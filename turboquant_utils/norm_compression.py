"""Norm tensor factorization and compression.

Decomposes the norm tensor α_{m,g} ∈ R^{M×G} via rank-1 SVD:
    α_{m,g} = β_m · γ_g · (1 + ε_{m,g})

where β_m is the row scale, γ_g is the group scale,
and ε_{m,g} is a small fractional residual quantized to int8.

Storage: M·16 + G·16 + M·G·8 + 32 bits
vs full: M·G·32 bits
Saving (d=128): 0.1875 BPW
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


def _pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 values (M, G) into uint8 (M, ceil(G/2)).

    Layout: byte = (lo & 0xF) | (hi << 4), where lo/hi are signed nibbles
    stored in offset binary (value + 8).
    """
    M, G = values.shape
    # Convert to offset binary: -7..+7 → 1..15 (0 reserved for exact zero edge case)
    offset = (values + 8).clamp(0, 15).to(torch.uint8)
    if G % 2 == 1:
        offset = torch.nn.functional.pad(offset, (0, 1))  # pad last dim
    lo = offset[:, 0::2]
    hi = offset[:, 1::2]
    return lo | (hi << 4)


def _unpack_int4(packed: torch.Tensor, M: int, G: int) -> torch.Tensor:
    """Unpack uint8 → signed int4 values (M, G)."""
    lo = (packed & 0x0F).to(torch.int8) - 8
    hi = ((packed >> 4) & 0x0F).to(torch.int8) - 8
    result = torch.stack([lo, hi], dim=-1).reshape(M, -1)
    return result[:, :G]


@dataclass
class FactoredNorms:
    """Factored norm representation."""
    row_scale: torch.Tensor       # (M,) float16
    group_scale: torch.Tensor     # (G,) float16
    residual_int8: torch.Tensor   # (M, G) int8 or (M, G//2) uint8 (packed int4)
    residual_scale: float         # scalar for dequantizing residual
    residual_bits: int = 8        # 8 for int8, 4 for packed int4


def factorize_norms(norms: torch.Tensor, residual_bits: int = 8) -> FactoredNorms:
    """Factorize (M, G) norm tensor into rank-1 + quantized residual.

    Args:
        norms: (M, G) or (M,) norm tensor
        residual_bits: 8 for int8, 4 for packed int4

    Returns:
        FactoredNorms with row/group scales and quantized residual
    """
    if norms.dim() == 1:
        return FactoredNorms(
            row_scale=norms.half(),
            group_scale=torch.ones(1, dtype=torch.float16, device=norms.device),
            residual_int8=torch.zeros(norms.shape[0], 1, dtype=torch.int8, device=norms.device),
            residual_scale=1.0,
            residual_bits=residual_bits,
        )

    M, G = norms.shape
    norms_f = norms.float()

    # SVD for rank-1 approximation
    U, S, Vh = torch.linalg.svd(norms_f, full_matrices=False)
    row_scale = U[:, 0] * S[0]  # (M,)
    group_scale = Vh[0, :]       # (G,)

    # Ensure positive (norms are always positive, but SVD can flip signs)
    if (row_scale < 0).sum() > M // 2:
        row_scale = -row_scale
        group_scale = -group_scale

    # Handle any remaining negative signs by flipping per-element
    row_sign = row_scale.sign().clamp(min=0) * 2 - 1  # default to +1 for zeros
    row_scale = row_scale.abs().clamp(min=1e-8)
    group_scale = group_scale * row_sign[0]  # correct for first-row sign
    # If group_scale has negatives, fallback to mean-based factorization
    if (group_scale <= 0).any():
        row_scale = norms_f.mean(dim=1).clamp(min=1e-8)
        group_scale = norms_f.mean(dim=0).clamp(min=1e-8)
        overall_mean = norms_f.mean().clamp(min=1e-8)
        group_scale = group_scale / overall_mean

    # Compute fractional residual
    rank1 = row_scale.unsqueeze(1) * group_scale.unsqueeze(0)  # (M, G)
    residual = norms_f / rank1.clamp(min=1e-8) - 1.0

    # Quantize residual
    res_amax = residual.abs().amax()
    if residual_bits == 4:
        max_val = 7  # signed 4-bit: -7..+7
    else:
        max_val = 127  # signed 8-bit: -127..+127

    if res_amax > 0:
        res_scale = res_amax.item() / max_val
        residual_q = (residual / res_scale).round().clamp(-max_val, max_val)
    else:
        res_scale = 1.0
        residual_q = torch.zeros(M, G, device=norms.device)

    if residual_bits == 4:
        # Pack two signed 4-bit values per uint8 byte
        residual_packed = _pack_int4(residual_q.to(torch.int8))
    else:
        residual_packed = residual_q.to(torch.int8)

    return FactoredNorms(
        row_scale=row_scale.half(),
        group_scale=group_scale.half(),
        residual_int8=residual_packed,
        residual_scale=res_scale,
        residual_bits=residual_bits,
    )


def reconstruct_norms(fn: FactoredNorms) -> torch.Tensor:
    """Reconstruct norms from factored representation.

    Returns:
        norms: (M, G) or (M,) float32 tensor
    """
    row_scale = fn.row_scale.float()
    group_scale = fn.group_scale.float()

    if fn.residual_int8.shape[1] == 1 and group_scale.shape[0] == 1:
        return row_scale

    rank1 = row_scale.unsqueeze(1) * group_scale.unsqueeze(0)

    if fn.residual_bits == 4:
        M = row_scale.shape[0]
        G = group_scale.shape[0]
        residual = _unpack_int4(fn.residual_int8, M, G).float() * fn.residual_scale
    else:
        residual = fn.residual_int8.float() * fn.residual_scale

    return rank1 * (1.0 + residual)


def norm_bpw(M: int, N: int, group_size: int, method: str = "fp32") -> float:
    """Compute norm BPW overhead.

    Args:
        M: out_features
        N: in_features
        group_size: columns per group
        method: "fp32", "fp16", "factored_int8", "factored_int4"

    Returns:
        BPW overhead (bits per weight element)
    """
    G = (N + group_size - 1) // group_size
    total_elements = M * N

    if method == "fp32":
        norm_bits = M * G * 32
    elif method == "fp16":
        norm_bits = M * G * 16
    elif method == "factored_int8":
        norm_bits = M * 16 + G * 16 + M * G * 8 + 32  # +32 for residual_scale
    elif method == "factored_int4":
        norm_bits = M * 16 + G * 16 + M * G * 4 + 32
    else:
        raise ValueError(f"Unknown norm method: {method}")

    return norm_bits / total_elements
