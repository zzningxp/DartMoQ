"""
MoE expert importance analysis utilities.

This module is intentionally not a runnable script. DartMoQ already owns model
loading, calibration data loading, and layer-wise activation capture. The public
entry point here only ranks neurons for one existing expert from one layer.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def analyze_expert_energy(
    expert,
    inps,
    reduce_ratio: float = 1.0,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """
    Return per-neuron importance rates for one routed expert.

    Args:
        inps: Hidden states captured before the MoE block, shaped
            ``[samples, seqlen, hidden_size]``.
        reduce_ratio: Blend between activation L2 and Linf statistics.
            ``0`` uses only L2, ``1`` uses only Linf.
        chunk_size: Number of flattened tokens processed at a time.

    Returns:
        A 1D tensor shaped ``[expert_intermediate_size]``. Larger values mean
        more important neurons.
    """
    if not 0.0 <= float(reduce_ratio) <= 1.0:
        raise ValueError("reduce_ratio must be in [0, 1]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    device = inps.device

    up_weight = expert.up_proj.weight
    gate_weight = expert.gate_proj.weight
    down_weight = expert.down_proj.weight
    act_fn = getattr(expert, "act_fn", F.silu)

    d_ff = up_weight.shape[0]
    base_energy = (down_weight.float() ** 2).sum(dim=0)
    exp_l2_square = torch.zeros(d_ff, device=device, dtype=torch.float32)
    exp_linf_square = torch.zeros(d_ff, device=device, dtype=torch.float32)

    flat_inps = inps.reshape(-1, inps.shape[-1])
    for start in range(0, flat_inps.shape[0], chunk_size):
        hidden_states = flat_inps[start:start + chunk_size]

        pre_act = F.linear(hidden_states, up_weight)
        gate = F.linear(hidden_states, gate_weight)
        activations = act_fn(gate) * pre_act
        activations = activations.float()

        exp_l2_square += (activations ** 2).sum(dim=0)
        exp_linf_square = torch.maximum(
            exp_linf_square,
            torch.max(torch.abs(activations), dim=0)[0] ** 2,
        )

    exp_stat = (1 - reduce_ratio) * exp_l2_square + reduce_ratio * exp_linf_square
    return (base_energy.to(device=device) * exp_stat).to(device=device)
