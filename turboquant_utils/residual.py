"""Residual (multi-pass) TurboQuant quantization.

Residual quantization applies TurboQuant multiple times:
  Pass 1: Quantize W at bit_width b1 → W_hat1, residual R1 = W - W_hat1
  Pass 2: Quantize R1 at bit_width b2 → R_hat1
  Final:  W_approx = W_hat1 + R_hat1  (total bits = b1 + b2)

This achieves better quality than single-pass at the same total bit budget
because the residual has different structure from the original weight.

Shared-rotation multi-pass residual
------------------------------------
When all N residual passes use the **same** rotation matrix Π, the rotation
factors out of the sum of passes.  For each group g:

  Ŵ_total[:, g] = Σ_k diag(α_k) · C_k[i_k] / √d · Π_g
                 = [Σ_k diag(α_k) · C_k[i_k] / √d] · Π_g

The quantity Ỹ[:, g] = Σ_k α_k · C_k[i_k] / √d  is the **merged
rotated-domain matrix**.  This enables:
  1. Applying residual quantization an arbitrary number of times.
  2. Merging all passes into a single dense weight matrix (exact, lossless).
  3. Re-quantizing the merged representation for compressed single-pass
     inference — the merge and re-quantize operate entirely in the rotated
     domain, avoiding any inverse rotation computation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from turboquant_model.codebook import get_codebook
from turboquant_model.rotation import generate_rotation_matrix
from turboquant_model.quantize import (
    turboquant_quantize,
    turboquant_quantize_packed,
    pack_4bit,
    unpack_4bit,
)


@torch.no_grad()
def residual_quantize(
    W: torch.Tensor,
    bit_width_1: int = 4,
    bit_width_2: int = 4,
    group_size: Optional[int] = None,
    seed_1: int = 42,
    seed_2: int = 1042,
) -> torch.Tensor:
    """Two-pass residual TurboQuant: returns the dequantized approximation.

    Args:
        W: (M, N) weight matrix
        bit_width_1: bits for first pass
        bit_width_2: bits for second pass (residual)
        group_size: group size (None = full row)
        seed_1: rotation seed for pass 1
        seed_2: rotation seed for pass 2

    Returns:
        W_approx: same shape/dtype as W
    """
    # Pass 1
    W_hat1 = turboquant_quantize(W, bit_width=bit_width_1, group_size=group_size, seed=seed_1)

    # Residual
    residual = W.float() - W_hat1.float()

    # Pass 2
    R_hat = turboquant_quantize(residual, bit_width=bit_width_2, group_size=group_size, seed=seed_2)

    return (W_hat1.float() + R_hat.float()).to(W.dtype)


@torch.no_grad()
def residual_quantize_packed(
    W: torch.Tensor,
    bit_width_1: int = 4,
    bit_width_2: int = 4,
    group_size: Optional[int] = None,
    seed_1: int = 42,
    seed_2: int = 1042,
) -> dict:
    """Two-pass residual TurboQuant: returns packed representations for both passes.

    Args:
        W: (M, N) weight matrix
        bit_width_1: bits for first pass
        bit_width_2: bits for second pass
        group_size: group size (None = full row)
        seed_1: rotation seed for pass 1
        seed_2: rotation seed for pass 2

    Returns:
        dict with:
            pass1: dict (same format as turboquant_quantize_packed output)
            pass2: dict (same format, for the residual)
            total_bits: bit_width_1 + bit_width_2
    """
    M, N = W.shape
    if group_size is None:
        group_size = N

    # Pass 1: quantize and pack
    pass1 = turboquant_quantize_packed(W, bit_width=bit_width_1, group_size=group_size, seed=seed_1)

    # Reconstruct pass 1 to compute residual
    W_hat1 = _dequantize_from_packed(pass1, device=W.device)
    residual = W.float() - W_hat1

    # Pass 2: quantize residual
    pass2 = turboquant_quantize_packed(residual, bit_width=bit_width_2, group_size=group_size, seed=seed_2)

    return {
        "pass1": pass1,
        "pass2": pass2,
        "total_bits": bit_width_1 + bit_width_2,
    }


def _dequantize_from_packed(packed_data: dict, device: torch.device) -> torch.Tensor:
    """Reconstruct weight from packed representation.

    Args:
        packed_data: output from turboquant_quantize_packed
        device: target device

    Returns:
        W_approx: (M, N) float32 tensor
    """
    M, N = packed_data["shape"]
    group_size = packed_data["group_size"]
    seed = packed_data["seed"]

    indices_packed = packed_data["indices_packed"].to(device)
    codebook = packed_data["codebook"].to(device)
    norms = packed_data["norms"].to(device)

    indices = unpack_4bit(indices_packed, N if N % 2 == 0 else N + 1)
    indices = indices[:, :N]  # trim padding

    n_groups = math.ceil(N / group_size)
    W_approx = torch.zeros(M, N, dtype=torch.float32, device=device)

    for g in range(n_groups):
        g_start = g * group_size
        g_end = min(g_start + group_size, N)
        g_dim = g_end - g_start

        Pi = generate_rotation_matrix(g_dim, seed=seed + g_start).to(device)
        scale = math.sqrt(g_dim)

        Y_g = codebook[indices[:, g_start:g_end].long()] / scale
        W_g = Y_g @ Pi

        if norms.dim() == 1:
            W_g = W_g * norms.unsqueeze(1)
        else:
            W_g = W_g * norms[:, g].unsqueeze(1)

        W_approx[:, g_start:g_end] = W_g

    return W_approx


# ---------------------------------------------------------------------------
# Multi-pass residual with shared rotation (enables merging)
# ---------------------------------------------------------------------------


@torch.no_grad()
def multi_residual_quantize(
    W: torch.Tensor,
    n_passes: int = 3,
    bit_width: int = 4,
    group_size: Optional[int] = None,
    seed: int = 42,
) -> torch.Tensor:
    """N-pass residual TurboQuant with shared rotation: returns dequantized approximation.

    All passes share the same rotation matrix (seed), enabling lossless merging
    of the quantized representations.  Because the rotation factors out of the
    sum, passes can later be merged in the rotated domain.

    Pass 1: Quantize W        → Ŵ₁, residual R₁ = W − Ŵ₁
    Pass 2: Quantize R₁       → R̂₁, residual R₂ = R₁ − R̂₁
    …
    Pass N: Quantize R_{N−1}  → R̂_{N−1}
    Final:  W_approx = Ŵ₁ + R̂₁ + … + R̂_{N−1}

    Args:
        W: (M, N) weight matrix
        n_passes: number of residual passes (≥ 1)
        bit_width: bits per pass
        group_size: group size (None = full row)
        seed: rotation seed shared by all passes

    Returns:
        W_approx: same shape/dtype as W
    """
    current = W.float()
    W_approx = torch.zeros_like(current)

    for _ in range(n_passes):
        W_hat = turboquant_quantize(
            current, bit_width=bit_width, group_size=group_size, seed=seed,
        )
        W_approx += W_hat.float()
        current = current - W_hat.float()

    return W_approx.to(W.dtype)


@torch.no_grad()
def multi_residual_quantize_packed(
    W: torch.Tensor,
    n_passes: int = 3,
    bit_width: int = 4,
    group_size: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """N-pass residual TurboQuant with shared rotation: returns packed representations.

    All passes reuse the same rotation seed, which enables efficient merging
    via ``merge_residual_passes`` or ``merge_and_requantize``.

    Args:
        W: (M, N) weight matrix
        n_passes: number of residual passes (≥ 1)
        bit_width: bits per pass
        group_size: group size (None = full row)
        seed: rotation seed shared by all passes

    Returns:
        dict with:
            passes: list[dict] — one per pass (turboquant_quantize_packed format)
            n_passes: int
            total_bits: bit_width × n_passes
            shared_seed: int — the common rotation seed
    """
    M, N = W.shape
    if group_size is None:
        group_size = N

    passes: list[dict] = []
    current = W.float()

    for _ in range(n_passes):
        packed = turboquant_quantize_packed(
            current, bit_width=bit_width, group_size=group_size, seed=seed,
        )
        passes.append(packed)
        W_hat = _dequantize_from_packed(packed, device=W.device)
        current = current - W_hat

    return {
        "passes": passes,
        "n_passes": n_passes,
        "total_bits": bit_width * n_passes,
        "shared_seed": seed,
    }


# ---------------------------------------------------------------------------
# Alternating two-rotation multi-pass residual
# ---------------------------------------------------------------------------


@torch.no_grad()
def alternating_residual_quantize(
    W: torch.Tensor,
    n_passes: int = 3,
    bit_width: int = 4,
    group_size: Optional[int] = None,
    seed_a: int = 42,
    seed_b: int = 1042,
) -> torch.Tensor:
    """N-pass residual TurboQuant with two alternating rotations.

    Pass 0 uses seed_a, pass 1 uses seed_b, pass 2 uses seed_a, etc.
    Different rotations may decorrelate residual error differently,
    potentially capturing more information per pass than a single rotation.

    Trade-off vs shared rotation: inference requires 2 pre-rotated inputs
    instead of 1, and merge_and_requantize cannot be used (different rotation
    domains). However, merge_residual_passes (exact dense merge) still works.

    Args:
        W: (M, N) weight matrix
        n_passes: number of residual passes (≥ 1)
        bit_width: bits per pass
        group_size: group size (None = full row)
        seed_a: rotation seed for even passes (0, 2, 4, …)
        seed_b: rotation seed for odd passes (1, 3, 5, …)

    Returns:
        W_approx: same shape/dtype as W
    """
    current = W.float()
    W_approx = torch.zeros_like(current)

    for i in range(n_passes):
        seed = seed_a if i % 2 == 0 else seed_b
        W_hat = turboquant_quantize(
            current, bit_width=bit_width, group_size=group_size, seed=seed,
        )
        W_approx += W_hat.float()
        current = current - W_hat.float()

    return W_approx.to(W.dtype)


@torch.no_grad()
def alternating_residual_quantize_packed(
    W: torch.Tensor,
    n_passes: int = 3,
    bit_width: int = 4,
    group_size: Optional[int] = None,
    seed_a: int = 42,
    seed_b: int = 1042,
) -> dict:
    """N-pass residual TurboQuant with two alternating rotations: packed output.

    Args:
        W: (M, N) weight matrix
        n_passes: number of residual passes (≥ 1)
        bit_width: bits per pass
        group_size: group size (None = full row)
        seed_a: rotation seed for even passes
        seed_b: rotation seed for odd passes

    Returns:
        dict with:
            passes: list[dict] — one per pass
            n_passes: int
            total_bits: bit_width × n_passes
            seeds: (seed_a, seed_b)
    """
    M, N = W.shape
    if group_size is None:
        group_size = N

    passes: list[dict] = []
    current = W.float()

    for i in range(n_passes):
        seed = seed_a if i % 2 == 0 else seed_b
        packed = turboquant_quantize_packed(
            current, bit_width=bit_width, group_size=group_size, seed=seed,
        )
        passes.append(packed)
        W_hat = _dequantize_from_packed(packed, device=W.device)
        current = current - W_hat

    return {
        "passes": passes,
        "n_passes": n_passes,
        "total_bits": bit_width * n_passes,
        "seeds": (seed_a, seed_b),
    }


@torch.no_grad()
def merge_residual_passes(
    packed_data: dict,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Merge N shared-rotation residual passes into a single dense weight matrix.

    Because all passes share the same rotation matrix Π, the inverse rotations
    are identical and the passes can be merged by simple addition:

      W_merged = Σ_k Ŵ_k = [Σ_k α_k · C_k[i_k] / √d] · Π

    This is mathematically exact (lossless with respect to the quantized
    representations).

    Args:
        packed_data: output of ``multi_residual_quantize_packed``
        device: target device (default: CPU)

    Returns:
        W_merged: (M, N) float32 tensor — the exact sum of all dequantized passes
    """
    if device is None:
        device = torch.device("cpu")

    passes = packed_data["passes"]
    M, N = passes[0]["shape"]

    W_merged = torch.zeros(M, N, dtype=torch.float32, device=device)
    for pass_data in passes:
        W_merged += _dequantize_from_packed(pass_data, device=device)

    return W_merged


@torch.no_grad()
def merge_and_requantize(
    packed_data: dict,
    target_bit_width: int = 4,
    device: Optional[torch.device] = None,
) -> dict:
    """Merge N shared-rotation passes in the rotated domain and re-quantize.

    This is the key operation that exploits the shared rotation property.
    Instead of the naive merge-then-recompress path (which would require
    N inverse rotations + 1 forward rotation), this function:

    1. Sums the scaled codebook values across all passes **in the rotated
       domain** — no inverse rotation is ever computed.
    2. Re-normalizes and re-quantizes the merged representation directly.
    3. Returns a single-pass packed representation.

    The result encapsulates multi-pass quality with single-pass inference cost
    (only one rotation per group in the forward path).

    Args:
        packed_data: output of ``multi_residual_quantize_packed``
        target_bit_width: bits for the merged single-pass representation
        device: target device (default: CPU)

    Returns:
        dict in the same format as ``turboquant_quantize_packed`` — a single
        pass that approximates the sum of all input passes.
    """
    if device is None:
        device = torch.device("cpu")

    passes = packed_data["passes"]
    seed = packed_data["shared_seed"]
    M, N = passes[0]["shape"]
    group_size = passes[0]["group_size"]
    n_groups = math.ceil(N / group_size)

    centroids, boundaries = get_codebook(target_bit_width)
    centroids = centroids.to(device)
    boundaries = boundaries.to(device)

    all_merged_indices: list[torch.Tensor] = []
    all_merged_norms: list[torch.Tensor] = []

    for g in range(n_groups):
        g_start = g * group_size
        g_end = min(g_start + group_size, N)
        g_dim = g_end - g_start
        scale = math.sqrt(g_dim)

        # Sum rotated-domain values (with norms) across all passes.
        # Each pass contributes: norms_k · codebook_k[idx_k] / scale
        Y_merged = torch.zeros(M, g_dim, dtype=torch.float32, device=device)

        for pass_data in passes:
            cb = pass_data["codebook"].to(device)
            norms = pass_data["norms"].to(device)
            ip = pass_data["indices_packed"].to(device)

            indices = unpack_4bit(ip, N if N % 2 == 0 else N + 1)[:, :N]
            idx_g = indices[:, g_start:g_end].long()

            Y_g = cb[idx_g] / scale  # (M, g_dim) — rotated-domain values

            if norms.dim() == 1:
                Y_g = Y_g * norms.unsqueeze(1)
            else:
                Y_g = Y_g * norms[:, g].unsqueeze(1)

            Y_merged += Y_g

        # Re-normalize the merged rotated-domain matrix
        merged_norms = Y_merged.norm(dim=1, keepdim=True).clamp(min=1e-8)
        Y_normalized = Y_merged / merged_norms

        # Re-quantize directly (already in the rotated domain — no rotation
        # needed, which is the whole efficiency win)
        Y_scaled = Y_normalized * scale
        idx = torch.searchsorted(boundaries, Y_scaled.reshape(-1))
        idx = idx.clamp(0, len(centroids) - 1).reshape(M, g_dim)

        all_merged_indices.append(idx)
        all_merged_norms.append(merged_norms.squeeze(1))

    full_indices = torch.cat(all_merged_indices, dim=1)
    norms_out = (
        torch.stack(all_merged_norms, dim=1)
        if len(all_merged_norms) > 1
        else all_merged_norms[0]
    )

    if N % 2 != 0:
        full_indices = torch.nn.functional.pad(full_indices, (0, 1), value=0)

    packed = pack_4bit(full_indices)

    return {
        "indices_packed": packed,
        "codebook": centroids.cpu(),
        "norms": norms_out.cpu(),
        "seed": seed,
        "group_size": group_size,
        "shape": (M, N),
        "bit_width": target_bit_width,
    }
