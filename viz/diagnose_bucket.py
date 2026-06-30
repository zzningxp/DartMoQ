#!/usr/bin/env python3
"""Diagnose: why is LDI bound much higher than DP curve on L1?"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viz._cache_io import load_all_layers


def _loss_from_neuron_bits(rates, neuron_bits):
    total = 0.0
    for b, vec in rates.items():
        mask = neuron_bits == b
        if mask.any():
            total += float(vec[mask].sum())
    return total


def _ldi_ratio(layer, bit):
    """Copy of _ldi_ratio from headroom.py."""
    from viz._cache_io import neuron_loss_matrix
    neurons = neuron_loss_matrix(layer, bit).flatten()
    neurons = neurons[neurons > 0]
    if neurons.size < 2:
        return float("nan")
    c = np.clip(neurons, 1e-30, None)
    return float(c.mean() / np.exp(np.mean(np.log(c)))), c


def _dp_uniform_over_dp(layer, bits, bpw, slice_expert_num):
    """Copy of _dp_uniform_over_dp from headroom.py."""
    if abs(bpw - round(bpw)) > 1e-6:
        return float("nan"), float("nan"), float("nan")
    b_uniform = int(round(bpw))
    if b_uniform not in layer.by_bit:
        return float("nan"), float("nan"), float("nan")

    # uniform reference
    n_experts = layer.n_experts
    activation = np.full(n_experts, 1.0 / n_experts)
    uniform_loss = float(sum(
        activation[e] * layer.by_bit[b_uniform][e].sum()
        for e in range(n_experts)
    ))

    # DP solution
    expert_rates = [
        {b: layer.by_bit[b][e] for b in bits if b in layer.by_bit}
        for e in range(n_experts)
    ]
    from dp_utils import enum_optimal_m_scheme_global_fast
    _, neuron_bits_per_expert = enum_optimal_m_scheme_global_fast(
        expert_rates, activation, slice_expert_num, target_bpw=bpw)
    dp_loss = sum(
        activation[e] * _loss_from_neuron_bits(expert_rates[e], neuron_bits_per_expert[e])
        for e in range(n_experts)
    )
    if dp_loss <= 0:
        return float("nan"), float("nan"), float("nan")
    return uniform_loss / dp_loss, uniform_loss, dp_loss


def diagnose_model(model_id: str, quantmode: str, rank_mode: str, bpw: int = 2,
                   bucket_counts=(1, 2, 4, 8, 16, 32)):
    layers = load_all_layers(model_id, quantmode, rank_mode, bits=(1, 2, 3, 4))
    if not layers:
        print("No layers found for", model_id)
        return

    bits = sorted(set.intersection(*[set(L.by_bit.keys()) for L in layers]))
    bits = [b for b in bits if b > 0]

    for li, L in enumerate(layers):
        print(f"\n{'='*60}")
        print(f"L{L.layer_idx}")
        print(f"{'='*60}")

        ldi_val, c = _ldi_ratio(L, bpw)
        print(f"LDI bound   = {ldi_val:.4f}x")
        print(f"  c.shape     = {c.shape}")
        print(f"  c.min/max   = {c.min():.3e} / {c.max():.3e}")
        print(f"  AM(c)       = {c.mean():.3e}")
        print(f"  GM(c)       = {np.exp(np.mean(np.log(c))):.3e}")

        print(f"\nDP curves (bits={bits}, bpw={bpw}):")
        print(f"  slice  |  ratio  | uniform_loss |  dp_loss  |")
        print(f"---------+---------+--------------+-----------+")
        for s in bucket_counts:
            if s > L.n_neurons:
                continue
            ratio, u_loss, d_loss = _dp_uniform_over_dp(L, bits, bpw, s)
            print(f"  {s:4d}  | {ratio:.4f}x | {u_loss:.4e} | {d_loss:.4e} |")


if __name__ == "__main__":
    from viz._cache_io import discover_models
    models = discover_models()
    if not models:
        print("No cached models found")
        sys.exit(1)

    print(f"Models found: {models}")
    print(f"\nDiagnosing first model: {models[0]}")
    diagnose_model(models[0], "turboquant", "turboquant_innerproduct")
