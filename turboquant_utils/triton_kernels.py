"""Triton fused dequant + matmul kernels for on-the-fly inference.

These kernels avoid materializing the full dequantized weight by fusing
4-bit unpack → codebook lookup → matmul → norm rescale in one kernel launch.

Optimizations applied:
  1. Autotune — @triton.autotune searches (BLOCK_B, BLOCK_N, BLOCK_K, num_warps,
     num_stages) per problem shape; cached after first invocation.
  2. Shared-memory codebook — the 16-entry codebook (64 B) stays in L1/registers
     after first load in each K-tile; repeated gather hits cache.
  3. TF32 tensor cores — allow_tf32=True in tl.dot for ~2× throughput on
     fp32 Ampere+/Ada/Hopper.
  4. Pre-scaled norms — norms / sqrt(K) computed once on host, eliminating
     per-element division in the kernel epilogue.
  5. Software pipelining — num_stages in autotune configs controls prefetch depth.
  6. Transpose elimination — accumulates in natural (B, N) layout; no extra
     transpose required.

Main kernel: _turboquant_fused_matmul_kernel
  - Input: x_rot (pre-rotated activations), packed indices, codebook, norms_scaled
  - Output: x_rot @ codebook[indices].T * norms_scaled

Supports group-wise calls: pass a packed index slice (N, g_dim//2) with K=g_dim.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Autotune configurations — searched per unique (B, N, K) shape
# ---------------------------------------------------------------------------

_AUTOTUNE_CONFIGS = [
    # Small batch (inference with B=1..4)
    triton.Config({"BLOCK_B": 1,  "BLOCK_N": 32,  "BLOCK_K": 32},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK_B": 1,  "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_B": 4,  "BLOCK_N": 32,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 4,  "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=3),
    # Medium batch
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 32,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 64,  "BLOCK_K": 128}, num_warps=8, num_stages=3),
    # Large batch (tensor-core friendly ≥16 on all dims)
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 32,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 64,  "BLOCK_K": 128}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["B", "N", "K"])
@triton.jit
def _turboquant_fused_matmul_kernel(
    # Input
    input_ptr,        # (B, K) pre-rotated activations
    # Quantized weight
    indices_ptr,      # (N, K//2) packed uint8
    codebook_ptr,     # (n_levels,) float32
    norms_ptr,        # (N,) float32 — pre-scaled by 1/scale on host
    # Output
    output_ptr,       # (B, N)
    # Dims
    B, N, K,
    PACKED_K,         # K // 2 (stride for packed index rows)
    N_LEVELS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused dequant-matmul: output[b,n] = norms_scaled[n] * Σ_k x[b,k] * codebook[idx[n,k]]"""
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_b = rb < B
    mask_n = rn < N

    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        mask_k = rk < K

        # Load input tile: (BLOCK_B, BLOCK_K)
        inp_off = rb[:, None] * K + rk[None, :]
        inp_mask = mask_b[:, None] & mask_k[None, :]
        inp_tile = tl.load(input_ptr + inp_off, mask=inp_mask, other=0.0)

        # Load + unpack weight indices: (BLOCK_N, BLOCK_K)
        byte_col = rk // 2
        is_high = (rk % 2) == 1
        byte_off = rn[:, None] * PACKED_K + byte_col[None, :]
        w_mask = mask_n[:, None] & mask_k[None, :]
        packed = tl.load(indices_ptr + byte_off, mask=w_mask, other=0).to(tl.uint8)
        lo = packed & 0x0F
        hi = (packed >> 4) & 0x0F
        idx = tl.where(is_high[None, :], hi, lo)

        # Codebook lookup (16 entries — stays in L1/registers after first access)
        w_quant = tl.load(codebook_ptr + idx.to(tl.int32), mask=w_mask, other=0.0)

        # TF32 tensor-core MMA: (BLOCK_B, BLOCK_K) @ (BLOCK_K, BLOCK_N)
        acc += tl.dot(
            inp_tile.to(tl.float32),
            tl.trans(w_quant.to(tl.float32)),
            allow_tf32=True,
        )

    # Multiply by pre-scaled norms (norms / scale computed on host)
    norm_vals = tl.load(norms_ptr + rn, mask=mask_n, other=1.0)
    acc = acc * norm_vals[None, :]

    # Store
    out_off = rb[:, None] * N + rn[None, :]
    out_mask = mask_b[:, None] & mask_n[None, :]
    tl.store(output_ptr + out_off, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


def triton_fused_matmul(
    x_rot: torch.Tensor,           # (B, K) pre-rotated input
    indices_packed: torch.Tensor,   # (N, K//2) packed uint8
    codebook: torch.Tensor,         # (n_levels,) float32
    norms: torch.Tensor,            # (N,) float32
    K: int,                         # in_features (or group_size for per-group calls)
    scale: float | None = None,     # override sqrt(K) if needed
) -> torch.Tensor:
    """Fused dequant + matmul via Triton with autotune + TF32 tensor cores.

    Expects pre-rotated input: x_rot = x @ Pi.T

    Supports per-group calls: pass a slice of packed indices (N, g_dim//2)
    with K=g_dim. The kernel handles unpack + codebook lookup + matmul + norm
    rescale in one launch, avoiding materialization of the (N, K) float weight.

    Args:
        x_rot: (B, K) pre-rotated activations
        indices_packed: (N, K//2) packed 4-bit weight indices
        codebook: centroids
        norms: per-row weight norms (N,)
        K: dimension of this group (in_features or group_size)
        scale: norm divisor (default: sqrt(K))

    Returns:
        output: (B, N)
    """
    B = x_rot.shape[0]
    N = indices_packed.shape[0]
    PACKED_K = indices_packed.shape[1]
    if scale is None:
        scale = math.sqrt(K)

    # Pre-scale norms on host (avoids per-element division in kernel)
    norms_scaled = norms / scale

    output = torch.empty(B, N, dtype=torch.float32, device=x_rot.device)

    # Grid is a lambda so autotune can adapt it per config
    grid = lambda META: (
        triton.cdiv(B, META["BLOCK_B"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    _turboquant_fused_matmul_kernel[grid](
        x_rot, indices_packed, codebook, norms_scaled, output,
        B, N, K, PACKED_K,
        N_LEVELS=codebook.shape[0],
    )

    return output


# ---------------------------------------------------------------------------
# Dual-pass fused kernel: both residual passes in one launch
# ---------------------------------------------------------------------------

_DUAL_AUTOTUNE_CONFIGS = [
    # Small batch (inference with B=1..4) — tighter tiles for dual-pass register pressure
    triton.Config({"BLOCK_B": 1,  "BLOCK_N": 32,  "BLOCK_K": 32},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK_B": 1,  "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 4,  "BLOCK_N": 32,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 4,  "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
    # Medium batch
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 32,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 16, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
    # Large batch
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 32,  "BLOCK_K": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_B": 32, "BLOCK_N": 64,  "BLOCK_K": 64},  num_warps=8, num_stages=2),
]


@triton.autotune(configs=_DUAL_AUTOTUNE_CONFIGS, key=["B", "N", "K"])
@triton.jit
def _turboquant_fused_dual_matmul_kernel(
    # Pass 1
    input1_ptr,        # (B, K)
    indices1_ptr,      # (N, K//2) packed uint8
    codebook1_ptr,     # (n_levels,) float32
    norms1_ptr,        # (N,) float32 — pre-scaled
    # Pass 2
    input2_ptr,        # (B, K)
    indices2_ptr,      # (N, K//2) packed uint8
    codebook2_ptr,     # (n_levels,) float32
    norms2_ptr,        # (N,) float32 — pre-scaled
    # Output
    output_ptr,        # (B, N)
    # Dims
    B, N, K,
    PACKED_K,
    N_LEVELS: tl.constexpr,
    SAME_INPUT: tl.constexpr,  # 1 if input1==input2 (shared rotation), 0 otherwise
    BLOCK_B: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Dual-pass fused dequant-matmul: output = acc1*norms1 + acc2*norms2.

    Processes both residual passes in a single kernel launch, avoiding:
    - 2nd kernel launch overhead
    - Writing + reading pass1 output for the add
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_b = rb < B
    mask_n = rn < N

    acc1 = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        rk = k_start + tl.arange(0, BLOCK_K)
        mask_k = rk < K
        inp_mask = mask_b[:, None] & mask_k[None, :]
        w_mask = mask_n[:, None] & mask_k[None, :]

        # Shared index math for packed byte addressing
        byte_col = rk // 2
        is_high = (rk % 2) == 1
        byte_off = rn[:, None] * PACKED_K + byte_col[None, :]

        # ---- Pass 1: load input + unpack + codebook + dot ----
        inp1_off = rb[:, None] * K + rk[None, :]
        inp1 = tl.load(input1_ptr + inp1_off, mask=inp_mask, other=0.0)

        packed1 = tl.load(indices1_ptr + byte_off, mask=w_mask, other=0).to(tl.uint8)
        lo1 = packed1 & 0x0F
        hi1 = (packed1 >> 4) & 0x0F
        idx1 = tl.where(is_high[None, :], hi1, lo1)
        w1 = tl.load(codebook1_ptr + idx1.to(tl.int32), mask=w_mask, other=0.0)

        acc1 += tl.dot(inp1.to(tl.float32), tl.trans(w1.to(tl.float32)), allow_tf32=True)

        # ---- Pass 2: load input + unpack + codebook + dot ----
        if SAME_INPUT:
            inp2 = inp1
        else:
            inp2_off = rb[:, None] * K + rk[None, :]
            inp2 = tl.load(input2_ptr + inp2_off, mask=inp_mask, other=0.0)

        packed2 = tl.load(indices2_ptr + byte_off, mask=w_mask, other=0).to(tl.uint8)
        lo2 = packed2 & 0x0F
        hi2 = (packed2 >> 4) & 0x0F
        idx2 = tl.where(is_high[None, :], hi2, lo2)
        w2 = tl.load(codebook2_ptr + idx2.to(tl.int32), mask=w_mask, other=0.0)

        acc2 += tl.dot(inp2.to(tl.float32), tl.trans(w2.to(tl.float32)), allow_tf32=True)

    # Epilogue: combine both passes with their norms
    n1 = tl.load(norms1_ptr + rn, mask=mask_n, other=1.0)
    n2 = tl.load(norms2_ptr + rn, mask=mask_n, other=1.0)
    result = acc1 * n1[None, :] + acc2 * n2[None, :]

    out_off = rb[:, None] * N + rn[None, :]
    out_mask = mask_b[:, None] & mask_n[None, :]
    tl.store(output_ptr + out_off, result.to(output_ptr.dtype.element_ty), mask=out_mask)


def triton_fused_dual_matmul(
    x_rot1: torch.Tensor,           # (B, K) pre-rotated input for pass 1
    indices1_packed: torch.Tensor,   # (N, K//2) packed uint8 for pass 1
    codebook1: torch.Tensor,         # (n_levels,) float32
    norms1: torch.Tensor,            # (N,) float32
    x_rot2: torch.Tensor,           # (B, K) pre-rotated input for pass 2
    indices2_packed: torch.Tensor,   # (N, K//2) packed uint8 for pass 2
    codebook2: torch.Tensor,         # (n_levels,) float32
    norms2: torch.Tensor,            # (N,) float32
    K: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Dual-pass fused dequant + matmul: pass1 + pass2 in one Triton launch.

    Equivalent to:
        triton_fused_matmul(x_rot1, indices1, cb1, n1, K, scale)
      + triton_fused_matmul(x_rot2, indices2, cb2, n2, K, scale)

    but with one kernel launch, one output write, and shared index math.

    Args:
        x_rot1/x_rot2: (B, K) pre-rotated activations (may be same tensor)
        indices1/2_packed: (N, K//2) packed 4-bit weight indices
        codebook1/2: centroids for each pass
        norms1/2: per-row weight norms for each pass
        K: group dimension
        scale: norm divisor (default: sqrt(K))

    Returns:
        output: (B, N) = pass1_out + pass2_out
    """
    B = x_rot1.shape[0]
    N = indices1_packed.shape[0]
    PACKED_K = indices1_packed.shape[1]
    if scale is None:
        scale = math.sqrt(K)

    norms1_scaled = norms1 / scale
    norms2_scaled = norms2 / scale

    # Detect shared rotation (same rotated input → skip redundant load)
    same_input = x_rot1.data_ptr() == x_rot2.data_ptr()

    output = torch.empty(B, N, dtype=torch.float32, device=x_rot1.device)

    grid = lambda META: (
        triton.cdiv(B, META["BLOCK_B"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    _turboquant_fused_dual_matmul_kernel[grid](
        x_rot1, indices1_packed, codebook1, norms1_scaled,
        x_rot2, indices2_packed, codebook2, norms2_scaled,
        output,
        B, N, K, PACKED_K,
        N_LEVELS=codebook1.shape[0],
        SAME_INPUT=1 if same_input else 0,
    )

    return output
