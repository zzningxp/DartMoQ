"""TurboQuantLinear — Drop-in nn.Linear replacement with on-the-fly 4-bit dequantization.

Stores weights as packed 4-bit indices + per-row norms + shared codebook.
On-the-fly forward (Approach C: pre-rotate input):
  1. x_rot = x @ Pi.T           (rotate input, not weight)
  2. out = x_rot @ codebook[indices].T  (fused lookup + matmul)
  3. out = out * (norms / scale)  (rescale per output row)

Supports both single-pass and residual (two-pass) quantization.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from turboquant_model.rotation import (
    generate_rotation_matrix,
    hadamard_rotate,
    hadamard_rotate_inverse,
)
from turboquant_model.quantize import unpack_4bit, pack_4bit
from turboquant_model.codebook import get_codebook
from turboquant_model.norm_compression import factorize_norms, reconstruct_norms, FactoredNorms


class SharedScratchPool:
    """Double-buffered GPU scratch shared across all offloaded layers.

    Holds two sets of flat 1-D (indices, norms, codebook) buffers sized to
    the largest offloaded layer by *element count* (not per-dimension max).
    Layers use alternating slots (ping-pong) so one slot is written by H2D
    while the other is consumed by a kernel.

    Using flat buffers avoids wasting VRAM when layer shapes vary widely
    (e.g. one layer has many rows but few columns and another the reverse).

    VRAM cost: 2 × max_layer_numel  (vs N × layer_size without pooling).
    """

    def __init__(
        self,
        max_indices_numel: int,
        max_norms_numel: int,
        max_codebook_numel: int,
        device: torch.device,
    ):
        self.slots: list[dict[str, torch.Tensor]] = []
        for _ in range(2):
            self.slots.append({
                "indices": torch.empty(max_indices_numel, dtype=torch.uint8, device=device),
                "norms": torch.empty(max_norms_numel, dtype=torch.float32, device=device),
                "codebook": torch.empty(max_codebook_numel, dtype=torch.float32, device=device),
            })

    def get(self, slot: int, layer: "TurboQuantLinear") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (indices, norms, codebook) views reshaped to the layer's actual shape."""
        s = self.slots[slot % 2]
        idx_shape = layer._pass2_cpu_indices_packed.shape
        nrm_shape = layer._pass2_cpu_weight_norms.shape
        cb_shape = layer._pass2_cpu_codebook.shape

        idx = s["indices"][:math.prod(idx_shape)].view(idx_shape)
        nrm = s["norms"][:math.prod(nrm_shape)].view(nrm_shape)
        cb = s["codebook"][:math.prod(cb_shape)].view(cb_shape)
        return idx, nrm, cb

    def memory_bytes(self) -> int:
        """Total GPU bytes for both slots."""
        total = 0
        for s in self.slots:
            total += s["indices"].numel()  # uint8
            total += s["norms"].numel() * 4  # float32
            total += s["codebook"].numel() * 4  # float32
        return total

# Try to import fused kernels — prefers cuTile > Triton > Metal > PyTorch fallback
_HAS_CUTILE = False
_HAS_CUTILE_DUAL = False
try:
    from turboquant_model.cutile_kernels import cutile_fused_matmul, _CUTILE_AVAILABLE
    _HAS_CUTILE = _CUTILE_AVAILABLE
    if _CUTILE_AVAILABLE:
        from turboquant_model.cutile_kernels import cutile_fused_dual_matmul
        _HAS_CUTILE_DUAL = True
except ImportError:
    pass

_HAS_TRITON = False
_HAS_TRITON_DUAL = False
try:
    from turboquant_model.triton_kernels import triton_fused_matmul
    _HAS_TRITON = True
    from turboquant_model.triton_kernels import triton_fused_dual_matmul
    _HAS_TRITON_DUAL = True
except ImportError:
    pass

_HAS_METAL = False
try:
    from turboquant_model.metal_kernels import metal_fused_matmul, _METAL_AVAILABLE
    _HAS_METAL = _METAL_AVAILABLE
except ImportError:
    pass


class QuantizedEmbedding(nn.Module):
    """Drop-in replacement for nn.Embedding with quantized weight storage.

    Supports two modes:
      - INT8 (per-row symmetric): 1 scale per row, ~50% savings vs bf16
      - INT4 (per-group asymmetric, like Q4_K): scale+min per group of 32, ~75% savings

    Forward performs on-the-fly dequantization of only the looked-up rows,
    so the compute cost is negligible.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        mode: str = "int8",
        group_size: int = 32,
        padding_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.group_size = group_size
        self.padding_idx = padding_idx

        if mode == "int8":
            # Per-row symmetric: scale * int8
            self.register_buffer(
                "weight_int8",
                torch.zeros(num_embeddings, embedding_dim, dtype=torch.int8, device=device),
            )
            self.register_buffer(
                "weight_scale",
                torch.zeros(num_embeddings, dtype=torch.float32, device=device),
            )
        elif mode == "int4":
            # Per-group asymmetric (like Q4_K): packed uint8, scale+min per group
            n_groups_per_row = math.ceil(embedding_dim / group_size)
            self.register_buffer(
                "weight_packed",
                torch.zeros(num_embeddings, embedding_dim // 2, dtype=torch.uint8, device=device),
            )
            self.register_buffer(
                "weight_scale",
                torch.zeros(num_embeddings, n_groups_per_row, dtype=torch.float16, device=device),
            )
            self.register_buffer(
                "weight_min",
                torch.zeros(num_embeddings, n_groups_per_row, dtype=torch.float16, device=device),
            )
        else:
            raise ValueError(f"Unsupported QuantizedEmbedding mode: {mode}")

    @torch.no_grad()
    @staticmethod
    def from_float(
        embedding: nn.Embedding,
        mode: str = "int8",
        group_size: int = 32,
    ) -> "QuantizedEmbedding":
        """Quantize a float nn.Embedding in-place."""
        W = embedding.weight.data.float()
        V, D = W.shape
        device = W.device

        qe = QuantizedEmbedding(
            num_embeddings=V,
            embedding_dim=D,
            mode=mode,
            group_size=group_size,
            padding_idx=embedding.padding_idx,
            device=device,
        )

        if mode == "int8":
            # Symmetric per-row: scale = max(|w|) / 127
            amax = W.abs().amax(dim=1).clamp(min=1e-12)
            scale = amax / 127.0
            quantized = (W / scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
            qe.weight_int8.copy_(quantized)
            qe.weight_scale.copy_(scale)
        elif mode == "int4":
            gs = group_size
            n_groups = math.ceil(D / gs)
            packed = torch.zeros(V, D // 2, dtype=torch.uint8, device=device)
            scales = torch.zeros(V, n_groups, dtype=torch.float16, device=device)
            mins = torch.zeros(V, n_groups, dtype=torch.float16, device=device)

            for g in range(n_groups):
                start = g * gs
                end = min(start + gs, D)
                block = W[:, start:end]  # (V, gs)

                bmin = block.min(dim=1, keepdim=True).values
                bmax = block.max(dim=1, keepdim=True).values
                bmin = bmin.clamp(max=0)  # min <= 0 for asymmetric
                d = (bmax - bmin).clamp(min=1e-12) / 15.0

                quantized = ((block - bmin) / d).round().clamp(0, 15).to(torch.uint8)
                scales[:, g] = d.squeeze(1).half()
                mins[:, g] = (-bmin.squeeze(1)).half()

                # Pack pairs of 4-bit values into uint8
                half_start = start // 2
                for j in range(0, end - start, 2):
                    if start + j + 1 < D:
                        packed[:, half_start + j // 2] = (
                            quantized[:, j] | (quantized[:, j + 1] << 4)
                        )
                    else:
                        packed[:, half_start + j // 2] = quantized[:, j]

            qe.weight_packed.copy_(packed)
            qe.weight_scale.copy_(scales)
            qe.weight_min.copy_(mins)

        return qe

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.mode == "int8":
            rows_int8 = self.weight_int8[input]     # (..., D) int8
            rows_scale = self.weight_scale[input]    # (...,) float32
            return (rows_int8.float() * rows_scale.unsqueeze(-1)).to(
                torch.bfloat16 if self.weight_scale.dtype == torch.float32 else self.weight_scale.dtype
            )
        else:  # int4
            flat = input.reshape(-1)
            B = flat.shape[0]
            D = self.embedding_dim
            gs = self.group_size

            packed = self.weight_packed[flat]    # (B, D//2) uint8
            scales = self.weight_scale[flat]     # (B, n_groups) fp16
            mins = self.weight_min[flat]         # (B, n_groups) fp16

            # Unpack 4-bit values
            lo = (packed & 0x0F).to(torch.float16)
            hi = (packed >> 4).to(torch.float16)
            # Interleave: (B, D)
            unpacked = torch.stack([lo, hi], dim=-1).reshape(B, D)

            # Dequantize per group
            n_groups = scales.shape[1]
            unpacked = unpacked.reshape(B, n_groups, gs)
            dequant = unpacked * scales.unsqueeze(-1) - mins.unsqueeze(-1)
            result = dequant.reshape(B, D)

            return result.reshape(*input.shape, D).to(torch.bfloat16)

    def memory_bytes(self) -> int:
        """Approximate GPU memory for quantized storage."""
        if self.mode == "int8":
            return (
                self.weight_int8.numel()          # int8
                + self.weight_scale.numel() * 4   # float32
            )
        else:
            return (
                self.weight_packed.numel()         # uint8
                + self.weight_scale.numel() * 2    # float16
                + self.weight_min.numel() * 2      # float16
            )

    def dequantize(self) -> torch.Tensor:
        """Reconstruct the full float embedding table (for debugging)."""
        idx = torch.arange(self.num_embeddings, device=self.weight_scale.device)
        return self.forward(idx)


class TurboQuantLinear(nn.Module):
    """Linear layer with TurboQuant-compressed weights and on-the-fly dequantization.

    Storage per layer:
      - indices_packed: (M, N//2) uint8 — packed 4-bit quantization indices
      - weight_norms: (M,) or (M, n_groups) float32 — per-row norms
      - codebook: (16,) float32 — Lloyd-Max centroids (shared)
      - [optional] pass2_*: same buffers for residual pass

    Forward pass:
      x_rot = x @ Pi.T
      output = x_rot @ codebook[indices].T * (norms / sqrt(group_size))
      [+ residual pass if present]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        bit_width: int = 4,
        group_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        rotation: str = "qr",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bit_width = bit_width
        self.group_size = group_size or in_features
        self.n_levels = 2**bit_width
        self.rotation = rotation

        pack_factor = 8 // bit_width
        packed_dim = math.ceil(in_features / pack_factor)
        n_groups = math.ceil(in_features / self.group_size)

        # Pass 1 buffers
        self.register_buffer(
            "indices_packed",
            torch.zeros(out_features, packed_dim, dtype=torch.uint8, device=device),
        )
        if n_groups == 1:
            self.register_buffer(
                "weight_norms",
                torch.ones(out_features, dtype=torch.float32, device=device),
            )
        else:
            self.register_buffer(
                "weight_norms",
                torch.ones(out_features, n_groups, dtype=torch.float32, device=device),
            )
        self.register_buffer(
            "codebook",
            torch.zeros(self.n_levels, dtype=torch.float32, device=device),
        )

        # Pass 2 (residual) buffers — None until set
        self.register_buffer("pass2_indices_packed", None)
        self.register_buffer("pass2_weight_norms", None)
        self.register_buffer("pass2_codebook", None)
        self._pass2_seed: Optional[int] = None

        # Bias
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(out_features, dtype=torch.float32, device=device),
            )
        else:
            self.bias = None

        # Rotation cache: dict[seed_offset → Pi tensor]
        self._rotation_cache: dict[int, torch.Tensor] = {}
        self._rotation_seed: int = 42
        self._scale: float = math.sqrt(self.group_size)

        # Cached unpacked indices (lazy, freed on device change)
        self._cached_indices: Optional[torch.Tensor] = None
        self._cached_pass2_indices: Optional[torch.Tensor] = None
        self._n_groups: int = math.ceil(in_features / self.group_size)

        # Fused kernel priority: cuTile > Triton > Metal > PyTorch fallback
        self.use_cutile: bool = _HAS_CUTILE
        self.use_triton: bool = _HAS_TRITON
        self.use_metal: bool = _HAS_METAL

        # Variable bit-width support
        self._group_bit_widths: Optional[list[int]] = None
        self._group_codebooks: Optional[list[torch.Tensor]] = None
        self._indices_uint8: Optional[torch.Tensor] = None  # (M, N) uint8

        # Factored norm support
        self._factored_norms: Optional[FactoredNorms] = None
        self._use_factored_norms: bool = False

        # CPU offload for residual pass 2
        self._cpu_offload_pass2: bool = False
        self._pass2_cpu_indices_packed: Optional[torch.Tensor] = None  # pinned CPU
        self._pass2_cpu_weight_norms: Optional[torch.Tensor] = None   # pinned CPU
        self._pass2_cpu_codebook: Optional[torch.Tensor] = None       # pinned CPU
        self._copy_stream: Optional[torch.cuda.Stream] = None

        # Shared double-buffered scratch pool (set by enable_prefetch_chain)
        self._scratch_pool: Optional[SharedScratchPool] = None
        self._scratch_idx: int = 0  # which slot this layer uses (0 or 1)

        # Next-layer prefetch state
        self._prefetch_event: Optional[torch.cuda.Event] = None
        # Use object.__setattr__ to prevent nn.Module from registering
        # the linked layer as a child submodule.
        object.__setattr__(self, '_next_offloaded_layer', None)

    def set_rotation(self, seed: int):
        self._rotation_seed = seed
        self._rotation_cache.clear()

    def set_pass2(
        self,
        indices_packed: torch.Tensor,
        weight_norms: torch.Tensor,
        codebook: torch.Tensor,
        seed: int,
    ):
        """Set residual (pass 2) quantization data."""
        self.register_buffer("pass2_indices_packed", indices_packed)
        self.register_buffer("pass2_weight_norms", weight_norms)
        self.register_buffer("pass2_codebook", codebook)
        self._pass2_seed = seed
        self._cached_pass2_indices = None

    @property
    def has_residual(self) -> bool:
        return self.pass2_indices_packed is not None or self._pass2_cpu_indices_packed is not None

    @property
    def is_pass2_offloaded(self) -> bool:
        return self._cpu_offload_pass2 and self._pass2_cpu_indices_packed is not None

    def offload_pass2_to_cpu(self) -> None:
        """Move residual pass 2 buffers to pinned CPU memory.

        After calling this, GPU scratch is provided by a shared
        ``SharedScratchPool`` (set up by ``enable_prefetch_chain``).
        The forward pass will use a dedicated CUDA stream to asynchronously
        copy pass2 data from CPU to the shared scratch, overlapping the
        transfer with the pass 1 fused kernel.
        """
        if self.pass2_indices_packed is None:
            return

        device = self.indices_packed.device
        if device.type != "cuda":
            return  # CPU offload only meaningful for CUDA

        # Move to pinned CPU memory for fast async H2D
        self._pass2_cpu_indices_packed = self.pass2_indices_packed.cpu().pin_memory()
        self._pass2_cpu_weight_norms = self.pass2_weight_norms.cpu().pin_memory()
        self._pass2_cpu_codebook = self.pass2_codebook.cpu().pin_memory()

        # Remove GPU registered buffers to free VRAM
        self.register_buffer("pass2_indices_packed", None)
        self.register_buffer("pass2_weight_norms", None)
        self.register_buffer("pass2_codebook", None)

        # Create dedicated copy stream (may be replaced by shared stream later)
        self._copy_stream = torch.cuda.Stream(device=device)
        self._cpu_offload_pass2 = True
        self._cached_pass2_indices = None

    def reload_pass2_to_gpu(self) -> None:
        """Move offloaded pass2 data back to GPU (undo offload)."""
        if not self._cpu_offload_pass2 or self._pass2_cpu_indices_packed is None:
            return

        device = self.indices_packed.device
        self.register_buffer(
            "pass2_indices_packed", self._pass2_cpu_indices_packed.to(device),
        )
        self.register_buffer(
            "pass2_weight_norms", self._pass2_cpu_weight_norms.to(device),
        )
        self.register_buffer(
            "pass2_codebook", self._pass2_cpu_codebook.to(device),
        )

        # Free CPU pinned + shared scratch reference
        self._pass2_cpu_indices_packed = None
        self._pass2_cpu_weight_norms = None
        self._pass2_cpu_codebook = None
        self._scratch_pool = None
        self._scratch_idx = 0
        self._copy_stream = None
        self._cpu_offload_pass2 = False
        self._cached_pass2_indices = None

    def prefetch_pass2(self, stream: Optional[torch.cuda.Stream] = None) -> Optional[torch.cuda.Event]:
        """Start async H2D copy of pass2 data into the shared scratch pool.

        Called by the previous layer's forward hook to overlap the copy with
        the current layer's kernel execution.

        Returns a CUDA event that the default stream should wait on before
        using the scratch buffers, or None if not offloaded / no pool.
        """
        if not self._cpu_offload_pass2 or self._pass2_cpu_indices_packed is None:
            return None
        if self._scratch_pool is None:
            return None

        s = stream or self._copy_stream
        idx_buf, nrm_buf, cb_buf = self._scratch_pool.get(self._scratch_idx, self)

        with torch.cuda.stream(s):
            idx_buf.copy_(self._pass2_cpu_indices_packed, non_blocking=True)
            nrm_buf.copy_(self._pass2_cpu_weight_norms, non_blocking=True)
            cb_buf.copy_(self._pass2_cpu_codebook, non_blocking=True)

        event = torch.cuda.Event()
        s.record_event(event)
        self._prefetch_event = event
        return event

    def set_variable_bit_widths(
        self,
        group_bit_widths: list[int],
        group_codebooks: list[torch.Tensor],
        indices_uint8: torch.Tensor,
    ):
        """Set per-group variable bit-widths.

        Args:
            group_bit_widths: list of int bit-widths per group
            group_codebooks: list of (2^b_g,) codebook tensors per group
            indices_uint8: (M, N) uint8 indices
        """
        self._group_bit_widths = group_bit_widths
        self._group_codebooks = group_codebooks
        self._indices_uint8 = indices_uint8.to(self.indices_packed.device)

    def apply_norm_codec(self, method: str = "factored_int8"):
        """Apply norm compression (factorization + quantization).

        Args:
            method: "fp16", "factored_int8", or "factored_int4"
        """
        norms = self.weight_norms
        if method == "fp16":
            self.weight_norms.copy_(norms.half().float())
        elif method in ("factored_int8", "factored_int4"):
            residual_bits = 4 if method == "factored_int4" else 8
            self._factored_norms = factorize_norms(norms, residual_bits=residual_bits)
            # Replace weight_norms with reconstructed (lossy) version
            reconstructed = reconstruct_norms(self._factored_norms)
            self.weight_norms.copy_(reconstructed)
            self._use_factored_norms = True

    @property
    def has_variable_bit_widths(self) -> bool:
        return self._group_bit_widths is not None

    def _get_rotation(self, seed: int, g_start: int = 0) -> torch.Tensor:
        """Get cached rotation matrix for a specific group.

        Args:
            seed: base rotation seed
            g_start: group start column (each group uses seed + g_start)
        """
        key = seed + g_start
        if key not in self._rotation_cache:
            self._rotation_cache[key] = generate_rotation_matrix(
                self.group_size, seed=key
            ).to(self.indices_packed.device)
        return self._rotation_cache[key]

    def _get_indices(self) -> torch.Tensor:
        """Get unpacked indices (cached)."""
        if self._cached_indices is None:
            self._cached_indices = unpack_4bit(self.indices_packed, self.in_features)
        return self._cached_indices

    def _get_pass2_indices(self) -> torch.Tensor:
        if self._cached_pass2_indices is None and self.pass2_indices_packed is not None:
            self._cached_pass2_indices = unpack_4bit(self.pass2_indices_packed, self.in_features)
        return self._cached_pass2_indices

    def _forward_pass(
        self,
        x: torch.Tensor,
        indices: torch.Tensor | None,
        indices_packed: torch.Tensor,
        codebook: torch.Tensor,
        weight_norms: torch.Tensor,
        seed: int,
    ) -> torch.Tensor:
        """Single-pass on-the-fly dequant matmul with group-wise rotation.

        For each group g in [0, n_groups):
          x_rot_g = x[:, g_start:g_end] @ Pi_g.T     (rotate input slice)
          output += x_rot_g @ codebook[indices[:, g_start:g_end]].T * (norms_g / scale)

        Uses Triton fused kernel when available (operates on packed indices directly,
        avoiding unpack + codebook[idx] intermediate materialization).

        Args:
            x: (B, K) input (float32)
            indices: (N, K) unpacked int32 (None if using Triton path)
            indices_packed: (N, K//2) packed uint8 (for Triton path)
            codebook: (n_levels,) float32
            weight_norms: (N,) or (N, n_groups) float32
            seed: base rotation seed

        Returns:
            output: (B, N) float32
        """
        B = x.shape[0]
        N = indices_packed.shape[0]
        K = self.in_features
        device = x.device
        scale = self._scale

        output = torch.zeros(B, N, dtype=torch.float32, device=device)

        for g in range(self._n_groups):
            g_start = g * self.group_size
            g_end = min(g_start + self.group_size, K)
            g_dim = g_end - g_start

            # Rotate this group's input slice
            if self.rotation == "hadamard":
                x_rot_g = hadamard_rotate(x[:, g_start:g_end], seed + g_start)
            else:
                Pi_g = self._get_rotation(seed, g_start).to(device)
                x_rot_g = x[:, g_start:g_end] @ Pi_g.T  # (B, g_dim)

            # Per-group norms
            if weight_norms.dim() == 1:
                norms_g = weight_norms  # (N,) — same norm for all groups
            else:
                norms_g = weight_norms[:, g]  # (N,) — per-group norms

            if self.use_cutile and g_dim == self.group_size:
                # cuTile fused path: unpack + codebook lookup + matmul in one kernel
                packed_g = indices_packed[:, g_start // 2 : g_end // 2]
                out_g = cutile_fused_matmul(
                    x_rot_g.contiguous(), packed_g.contiguous(),
                    codebook, norms_g.contiguous(), g_dim, scale,
                )
            elif self.use_triton and g_dim == self.group_size:
                # Triton fused path: unpack + codebook lookup + matmul in one kernel
                packed_g = indices_packed[:, g_start // 2 : g_end // 2]  # (N, g_dim//2)
                out_g = triton_fused_matmul(
                    x_rot_g.contiguous(), packed_g.contiguous(),
                    codebook, norms_g.contiguous(), g_dim, scale,
                )
            elif self.use_metal and g_dim == self.group_size:
                # Metal fused path: unpack + codebook lookup + matmul in one kernel
                packed_g = indices_packed[:, g_start // 2 : g_end // 2]
                out_g = metal_fused_matmul(
                    x_rot_g.contiguous(), packed_g.contiguous(),
                    codebook, norms_g.contiguous(), g_dim, scale,
                )
            else:
                # PyTorch fallback: explicit unpack + lookup + matmul
                # Per-group codebook for variable bit-width
                if self._group_codebooks is not None and self._indices_uint8 is not None:
                    cb_g = self._group_codebooks[g]
                    idx_g = self._indices_uint8[:, g_start:g_end].long()
                    W_g = cb_g[idx_g]
                else:
                    if indices is None:
                        indices = unpack_4bit(indices_packed, K)
                    idx_g = indices[:, g_start:g_end]  # (N, g_dim)
                    W_g = codebook[idx_g.long()]       # (N, g_dim)
                out_g = x_rot_g @ W_g.T
                out_g = out_g * (norms_g[None, :] / scale)

            output += out_g

        return output

    def _forward_residual_fused(
        self,
        x: torch.Tensor,
        indices_packed1: torch.Tensor,
        codebook1: torch.Tensor,
        weight_norms1: torch.Tensor,
        seed1: int,
        indices_packed2: torch.Tensor,
        codebook2: torch.Tensor,
        weight_norms2: torch.Tensor,
        seed2: int,
        copy_done_event: Optional[torch.cuda.Event] = None,
    ) -> torch.Tensor:
        """Dual-pass fused forward: both residual passes in one kernel launch per group.

        Instead of 2 × n_groups kernel launches (pass1 then pass2), this does
        1 × n_groups launches using the dual-pass fused kernel, which merges
        unpack + codebook + matmul + norm-rescale for both passes and writes
        the summed output in a single store.

        When ``copy_done_event`` is provided (CPU-offload path), the rotation
        computations for all groups happen first (CPU work that overlaps with
        the async H2D copy), then the default stream waits on the event (device-
        side sync — no CPU blocking) before launching any dual kernel.

        Args:
            x: (B, K) input (float32)
            indices_packed1/2: (N, K//2) packed uint8 for each pass
            codebook1/2: (n_levels,) for each pass
            weight_norms1/2: (N,) or (N, n_groups) for each pass
            seed1/2: rotation seeds for each pass
            copy_done_event: CUDA event signaling pass2 H2D copy is complete

        Returns:
            output: (B, N) float32
        """
        B = x.shape[0]
        N = indices_packed1.shape[0]
        K = self.in_features
        device = x.device
        scale = self._scale

        output = torch.zeros(B, N, dtype=torch.float32, device=device)

        # Phase 1: Pre-compute all rotations (CPU work overlaps with H2D copy)
        rotations_1 = []
        rotations_2 = []
        for g in range(self._n_groups):
            g_start = g * self.group_size
            g_end = min(g_start + self.group_size, K)

            if self.rotation == "hadamard":
                x_rot1_g = hadamard_rotate(x[:, g_start:g_end], seed1 + g_start)
                x_rot2_g = hadamard_rotate(x[:, g_start:g_end], seed2 + g_start)
            else:
                Pi1_g = self._get_rotation(seed1, g_start).to(device)
                x_rot1_g = x[:, g_start:g_end] @ Pi1_g.T
                if seed1 == seed2:
                    x_rot2_g = x_rot1_g  # same tensor → SAME_INPUT optimization
                else:
                    Pi2_g = self._get_rotation(seed2, g_start).to(device)
                    x_rot2_g = x[:, g_start:g_end] @ Pi2_g.T

            rotations_1.append(x_rot1_g)
            rotations_2.append(x_rot2_g)

        # Phase 2: Ensure H2D copy is done (device-side sync, non-blocking to CPU)
        if copy_done_event is not None:
            torch.cuda.current_stream(device).wait_event(copy_done_event)

        # Phase 3: Launch dual kernels (pass2 data now guaranteed on GPU)
        for g in range(self._n_groups):
            g_start = g * self.group_size
            g_end = min(g_start + self.group_size, K)
            g_dim = g_end - g_start

            x_rot1_g = rotations_1[g]
            x_rot2_g = rotations_2[g]

            # Per-group norms
            n1_g = weight_norms1 if weight_norms1.dim() == 1 else weight_norms1[:, g]
            n2_g = weight_norms2 if weight_norms2.dim() == 1 else weight_norms2[:, g]

            packed1_g = indices_packed1[:, g_start // 2 : g_end // 2]
            packed2_g = indices_packed2[:, g_start // 2 : g_end // 2]

            if self.use_cutile and _HAS_CUTILE_DUAL and g_dim == self.group_size:
                out_g = cutile_fused_dual_matmul(
                    x_rot1_g.contiguous(), packed1_g.contiguous(),
                    codebook1, n1_g.contiguous(),
                    x_rot2_g.contiguous(), packed2_g.contiguous(),
                    codebook2, n2_g.contiguous(),
                    g_dim, scale,
                )
            elif self.use_triton and _HAS_TRITON_DUAL and g_dim == self.group_size:
                out_g = triton_fused_dual_matmul(
                    x_rot1_g.contiguous(), packed1_g.contiguous(),
                    codebook1, n1_g.contiguous(),
                    x_rot2_g.contiguous(), packed2_g.contiguous(),
                    codebook2, n2_g.contiguous(),
                    g_dim, scale,
                )
            else:
                # Fallback: two separate PyTorch forward passes
                indices1 = unpack_4bit(indices_packed1, K)
                indices2 = unpack_4bit(indices_packed2, K)
                idx1_g = indices1[:, g_start:g_end]
                idx2_g = indices2[:, g_start:g_end]
                W1_g = codebook1[idx1_g.long()]
                W2_g = codebook2[idx2_g.long()]
                out_g = (
                    x_rot1_g @ W1_g.T * (n1_g[None, :] / scale)
                    + x_rot2_g @ W2_g.T * (n2_g[None, :] / scale)
                )

            output += out_g

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """On-the-fly dequant forward pass with group-wise rotation.

        Handles 2D (B, K) and 3D (B, S, K) inputs.
        Uses dual-pass fused kernel when residual is present and a dual kernel
        is available (cuTile or Triton), falling back to two-pass otherwise.
        """
        device = x.device
        orig_shape = x.shape
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])

        x_f = x.float()

        _use_fused = self.use_cutile or self.use_triton or self.use_metal
        _has_dual = (self.use_cutile and _HAS_CUTILE_DUAL) or (self.use_triton and _HAS_TRITON_DUAL)

        if self.has_residual and _has_dual:
            # ---- Dual-pass fused path: 1 kernel launch per group ----

            copy_done_event = None

            # Resolve pass2 buffers (GPU-resident or CPU-offloaded)
            if self._cpu_offload_pass2:
                if self._prefetch_event is not None:
                    # Previous layer already started our H2D copy — use its event
                    copy_done_event = self._prefetch_event
                    self._prefetch_event = None
                elif self._scratch_pool is not None:
                    # Start async H2D copy into shared scratch pool
                    idx_buf, nrm_buf, cb_buf = self._scratch_pool.get(self._scratch_idx, self)
                    # Fence: prevent copy_stream from overwriting a slot
                    # still being read by the default stream (previous layer).
                    fence = torch.cuda.Event()
                    torch.cuda.current_stream(device).record_event(fence)
                    self._copy_stream.wait_event(fence)
                    with torch.cuda.stream(self._copy_stream):
                        idx_buf.copy_(self._pass2_cpu_indices_packed, non_blocking=True)
                        nrm_buf.copy_(self._pass2_cpu_weight_norms, non_blocking=True)
                        cb_buf.copy_(self._pass2_cpu_codebook, non_blocking=True)
                    copy_done_event = torch.cuda.Event()
                    self._copy_stream.record_event(copy_done_event)

                if self._scratch_pool is not None:
                    p2_indices, p2_norms, p2_codebook = self._scratch_pool.get(self._scratch_idx, self)
                else:
                    # Fallback: copy directly to temp tensors (no pool set up)
                    p2_indices = self._pass2_cpu_indices_packed.to(device, non_blocking=True)
                    p2_norms = self._pass2_cpu_weight_norms.to(device, non_blocking=True)
                    p2_codebook = self._pass2_cpu_codebook.to(device, non_blocking=True)
                    torch.cuda.current_stream(device).synchronize()
            else:
                p2_indices = self.pass2_indices_packed
                p2_norms = self.pass2_weight_norms
                p2_codebook = self.pass2_codebook

            output = self._forward_residual_fused(
                x_f,
                self.indices_packed, self.codebook, self.weight_norms, self._rotation_seed,
                p2_indices, p2_codebook, p2_norms, self._pass2_seed,
                copy_done_event=copy_done_event,
            )
        else:
            # ---- Standard path: pass 1, then optionally pass 2 ----

            copy_done_event = None

            # Start async H2D copy for offloaded pass 2 (overlaps with pass 1 compute)
            if self._cpu_offload_pass2 and self._pass2_cpu_indices_packed is not None:
                if self._prefetch_event is not None:
                    copy_done_event = self._prefetch_event
                    self._prefetch_event = None
                elif self._scratch_pool is not None:
                    idx_buf, nrm_buf, cb_buf = self._scratch_pool.get(self._scratch_idx, self)
                    # Fence: prevent copy_stream from overwriting a slot
                    # still being read by the default stream (previous layer).
                    fence = torch.cuda.Event()
                    torch.cuda.current_stream(device).record_event(fence)
                    self._copy_stream.wait_event(fence)
                    with torch.cuda.stream(self._copy_stream):
                        idx_buf.copy_(self._pass2_cpu_indices_packed, non_blocking=True)
                        nrm_buf.copy_(self._pass2_cpu_weight_norms, non_blocking=True)
                        cb_buf.copy_(self._pass2_cpu_codebook, non_blocking=True)
                    copy_done_event = torch.cuda.Event()
                    self._copy_stream.record_event(copy_done_event)

            # Pass 1 (runs on default stream, overlaps with H2D on copy_stream)
            if _use_fused or self._indices_uint8 is not None:
                indices = None
            else:
                indices = self._get_indices()
            output = self._forward_pass(
                x_f, indices, self.indices_packed, self.codebook,
                self.weight_norms, self._rotation_seed,
            )

            # Pass 2 (residual) if present
            if self.has_residual:
                if self._cpu_offload_pass2:
                    # Device-side wait: default stream pauses until H2D is done
                    # (non-blocking to CPU — pass 1 compute likely already hid the copy)
                    if copy_done_event is not None:
                        torch.cuda.current_stream(device).wait_event(copy_done_event)
                    if self._scratch_pool is not None:
                        p2_idx, p2_nrm, p2_cb = self._scratch_pool.get(self._scratch_idx, self)
                    else:
                        p2_idx = self._pass2_cpu_indices_packed.to(device)
                        p2_nrm = self._pass2_cpu_weight_norms.to(device)
                        p2_cb = self._pass2_cpu_codebook.to(device)
                    indices2 = None if _use_fused else unpack_4bit(p2_idx, self.in_features)
                    output += self._forward_pass(
                        x_f, indices2, p2_idx, p2_cb, p2_nrm, self._pass2_seed,
                    )
                else:
                    indices2 = None if _use_fused else self._get_pass2_indices()
                    output += self._forward_pass(
                        x_f, indices2, self.pass2_indices_packed, self.pass2_codebook,
                        self.pass2_weight_norms, self._pass2_seed,
                    )

        # Restore shape
        if len(orig_shape) == 3:
            output = output.reshape(orig_shape[0], orig_shape[1], self.out_features)

        # Prefetch next layer's pass2 data (overlaps with post-processing + next layer's pass1)
        if self._next_offloaded_layer is not None:
            # Record default-stream progress so copy_stream won't overwrite a
            # scratch slot still being read by an earlier layer's pass2 kernels.
            fence = torch.cuda.Event()
            torch.cuda.current_stream(device).record_event(fence)
            self._copy_stream.wait_event(fence)
            self._next_offloaded_layer.prefetch_pass2(stream=self._copy_stream)

        out = output.to(x.dtype)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    def dequantize(self) -> torch.Tensor:
        """Full dequantization: returns (M, N) bf16 weight (for debugging)."""
        indices = self._get_indices()
        scale = self._scale

        W = torch.zeros(
            self.out_features, self.in_features,
            dtype=torch.float32, device=self.indices_packed.device,
        )

        for g in range(self._n_groups):
            g_start = g * self.group_size
            g_end = min(g_start + self.group_size, self.in_features)

            if self.rotation == "hadamard":
                Y_g = self.codebook[indices[:, g_start:g_end].long()] / scale
                W_g = hadamard_rotate_inverse(Y_g, self._rotation_seed + g_start)
            else:
                Pi_g = self._get_rotation(self._rotation_seed, g_start)
                Y_g = self.codebook[indices[:, g_start:g_end].long()] / scale
                W_g = Y_g @ Pi_g[:g_end - g_start, :g_end - g_start]

            if self.weight_norms.dim() == 1:
                W_g = W_g * self.weight_norms.unsqueeze(1)
            else:
                W_g = W_g * self.weight_norms[:, g].unsqueeze(1)

            W[:, g_start:g_end] = W_g

        if self.has_residual:
            # Resolve pass2 tensors from GPU or CPU-offloaded source
            if self._cpu_offload_pass2:
                p2_indices_packed = self._pass2_cpu_indices_packed.to(device)
                p2_codebook = self._pass2_cpu_codebook.to(device)
                p2_norms = self._pass2_cpu_weight_norms.to(device)
                indices2 = unpack_4bit(p2_indices_packed, self.in_features)
            else:
                p2_codebook = self.pass2_codebook
                p2_norms = self.pass2_weight_norms
                indices2 = self._get_pass2_indices()
            for g in range(self._n_groups):
                g_start = g * self.group_size
                g_end = min(g_start + self.group_size, self.in_features)
                if self.rotation == "hadamard":
                    Y_g = p2_codebook[indices2[:, g_start:g_end].long()] / scale
                    W_g = hadamard_rotate_inverse(Y_g, self._pass2_seed + g_start)
                else:
                    Pi2_g = self._get_rotation(self._pass2_seed, g_start)
                    Y_g = p2_codebook[indices2[:, g_start:g_end].long()] / scale
                    W_g = Y_g @ Pi2_g[:g_end - g_start, :g_end - g_start]
                if p2_norms.dim() == 1:
                    W_g = W_g * p2_norms.unsqueeze(1)
                else:
                    W_g = W_g * p2_norms[:, g].unsqueeze(1)
                W[:, g_start:g_end] += W_g

        return W.to(torch.bfloat16)

    @torch.no_grad()
    def merge_passes(self) -> None:
        """Merge residual pass into the primary pass via rotated-domain addition.

        When both passes share the **same rotation seed**, the rotation matrix
        factors out of the sum and the merge happens entirely in the rotated
        domain — no inverse rotation is ever computed.  The merged rotated-domain
        values are re-normalized and re-quantized into a single set of packed
        4-bit indices.

        After calling this method ``has_residual`` becomes False and forward
        uses a single pass, cutting inference cost in half while retaining
        multi-pass quality (up to re-quantisation).

        When the rotation seeds differ, a dense dequantize-sum-requantize
        fallback is used (still correct, but involves inverse rotations).
        """
        if not self.has_residual:
            return

        # If pass2 is offloaded, reload to GPU first
        if self._cpu_offload_pass2:
            self.reload_pass2_to_gpu()

        device = self.indices_packed.device
        K = self.in_features
        N = self.out_features
        scale = self._scale

        centroids, boundaries = get_codebook(self.bit_width)
        centroids = centroids.to(device)
        boundaries = boundaries.to(device)

        same_rotation = self._pass2_seed == self._rotation_seed

        if same_rotation:
            # ---- Fast path: merge in the rotated domain ----
            indices1 = self._get_indices()
            indices2 = self._get_pass2_indices()

            all_merged_indices: list[torch.Tensor] = []
            all_merged_norms: list[torch.Tensor] = []

            for g in range(self._n_groups):
                g_start = g * self.group_size
                g_end = min(g_start + self.group_size, K)
                g_dim = g_end - g_start

                # Pass-1 contribution in rotated domain
                Y1 = self.codebook[indices1[:, g_start:g_end].long()] / scale
                n1 = (
                    self.weight_norms
                    if self.weight_norms.dim() == 1
                    else self.weight_norms[:, g]
                )
                Y1 = Y1 * n1.unsqueeze(1)

                # Pass-2 contribution in rotated domain
                Y2 = self.pass2_codebook[indices2[:, g_start:g_end].long()] / scale
                n2 = (
                    self.pass2_weight_norms
                    if self.pass2_weight_norms.dim() == 1
                    else self.pass2_weight_norms[:, g]
                )
                Y2 = Y2 * n2.unsqueeze(1)

                Y_merged = Y1 + Y2

                # Re-normalize
                merged_norms = Y_merged.norm(dim=1, keepdim=True).clamp(min=1e-8)
                Y_norm = Y_merged / merged_norms

                # Re-quantize (already rotated — no Pi needed)
                Y_scaled = Y_norm * scale
                idx = torch.searchsorted(boundaries, Y_scaled.reshape(-1))
                idx = idx.clamp(0, len(centroids) - 1).reshape(N, g_dim)

                all_merged_indices.append(idx)
                all_merged_norms.append(merged_norms.squeeze(1))

            full_indices = torch.cat(all_merged_indices, dim=1)
            norms_out = (
                torch.stack(all_merged_norms, dim=1)
                if len(all_merged_norms) > 1
                else all_merged_norms[0]
            )
        else:
            # ---- Fallback: dequantize both, sum, re-quantize ----
            W_merged = self.dequantize().float()
            all_indices: list[torch.Tensor] = []
            all_norms: list[torch.Tensor] = []

            for g_start in range(0, K, self.group_size):
                g_end = min(g_start + self.group_size, K)
                g_dim = g_end - g_start
                W_g = W_merged[:, g_start:g_end]

                norms = W_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
                W_norm = W_g / norms
                all_norms.append(norms.squeeze(1))

                Pi = generate_rotation_matrix(
                    g_dim, seed=self._rotation_seed + g_start
                ).to(device)
                Y = W_norm @ Pi.T
                Y_scaled = Y * scale

                idx = torch.searchsorted(boundaries, Y_scaled.reshape(-1))
                idx = idx.clamp(0, len(centroids) - 1).reshape(N, g_dim)
                all_indices.append(idx)

            full_indices = torch.cat(all_indices, dim=1)
            norms_out = (
                torch.stack(all_norms, dim=1)
                if len(all_norms) > 1
                else all_norms[0]
            )

        # Pack and replace buffers
        if K % 2 != 0:
            full_indices = torch.nn.functional.pad(full_indices, (0, 1), value=0)

        packed = pack_4bit(full_indices)
        self.indices_packed.copy_(packed)
        self.weight_norms.copy_(norms_out)
        self.codebook.copy_(centroids)

        # Clear residual buffers
        self.register_buffer("pass2_indices_packed", None)
        self.register_buffer("pass2_weight_norms", None)
        self.register_buffer("pass2_codebook", None)
        self._pass2_seed = None
        self._cached_indices = None
        self._cached_pass2_indices = None

    def memory_bytes(self) -> int:
        """Compressed storage in bytes (GPU VRAM only — excludes shared scratch pool)."""
        total = self.indices_packed.numel()  # uint8
        total += self.weight_norms.numel() * 4
        total += self.codebook.numel() * 4
        if self.bias is not None:
            total += self.bias.numel() * 4
        if not self._cpu_offload_pass2 and self.pass2_indices_packed is not None:
            total += self.pass2_indices_packed.numel()
            total += self.pass2_weight_norms.numel() * 4
            total += self.pass2_codebook.numel() * 4
        # When offloaded, pass2 lives on CPU — shared scratch pool VRAM is
        # amortized across all layers and reported separately via
        # SharedScratchPool.memory_bytes()
        return total

    def memory_bytes_cpu(self) -> int:
        """CPU memory used by offloaded pass2 buffers (0 when not offloaded)."""
        if not self._cpu_offload_pass2 or self._pass2_cpu_indices_packed is None:
            return 0
        total = self._pass2_cpu_indices_packed.numel()  # uint8
        total += self._pass2_cpu_weight_norms.numel() * 4
        total += self._pass2_cpu_codebook.numel() * 4
        return total

    def extra_repr(self) -> str:
        residual = ", residual=True" if self.has_residual else ""
        offload = ", cpu_offload=True" if self._cpu_offload_pass2 else ""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bit_width={self.bit_width}, group_size={self.group_size}{residual}{offload}, "
            f"compressed={self.memory_bytes() / 1024:.1f} KB"
        )
