"""Headroom visualizations — DP loss curves with varying bucket counts.

This module provides visualization for DP loss as we vary the number of
buckets per expert (slice_expert_num).

All figures are driven by the cached sensitivity tensors in
``intermediate_result/quant_outlier_{quantmode}/{rank_mode}/{model_id}/``; no model needs to be
reloaded.

Usage
-----
    python -m viz.headroom                          # all models with cache
    python -m viz.headroom --model olmoe-7b-1b
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Make sibling modules importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viz._cache_io import (
    LayerSensitivity, apply_paper_style, discover_layers, discover_models,
    load_all_layers, model_label, resolve_model_id,
)

OUT_ROOT = "plot/headroom"
DEFAULT_QUANTMODE = "turboquant"
DEFAULT_RANK_MODE = "turboquant_innerproduct"
DEFAULT_BIT = 2  # probe bit used for static panels


def _multi_model_axes(models: List[str], width_per: float = 4.5, height: float = 4.0):
    """Create a 1xN figure (one subplot per model) with a shared layout."""
    n = max(len(models), 1)
    fig, axes = plt.subplots(1, n, figsize=(width_per * n, height), squeeze=False)
    return fig, axes[0]


def _loss_from_neuron_bits(
    rates: Dict[int, np.ndarray],
    neuron_bits: np.ndarray,
) -> float:
    """Sum loss using per-neuron bit assignment + per-bit loss table.

    `rates[b][i]` is the loss when neuron i is quantized at b bits, so the
    total loss is simply Σ_i rates[neuron_bits[i]][i].
    """
    total = 0.0
    for b, vec in rates.items():
        mask = neuron_bits == b
        if mask.any():
            total += float(vec[mask].sum())
    return total


def _compute_dp_loss(
    layer: LayerSensitivity,
    bits: List[int],
    bpw: float,
    slice_expert_num: int,
) -> float:
    """Compute DP loss for one layer at a given bucket count and bpw.

    Returns the absolute DP loss value (not a ratio).
    """
    n_experts = layer.n_experts
    activation = np.full(n_experts, 1.0 / n_experts)  # uniform weighting
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

    return dp_loss


def bucket_sweep(
    models: List[str],
    quantmode: str = DEFAULT_QUANTMODE,
    rank_mode: str = DEFAULT_RANK_MODE,
    bpw: int = 2,
    bucket_counts: Tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    num_layers: int = 4,
    out_dir: str = OUT_ROOT,
    save_pdf: bool = False,
) -> str:
    """DP loss curves as we vary bucket count per expert.

    Y-axis: DP loss (absolute value, log scale)
    X-axis: slice_expert_num (= buckets per expert) on log scale

    Each layer gets its own curve.
    """
    fig, axes = _multi_model_axes(models, width_per=4.5, height=4.0)

    for ax, model_id in zip(axes, models):
        print(f"\n=== bucket_sweep for model_id = {model_id} ===")
        # Load layers from start
        layers_start = load_all_layers(model_id, quantmode, rank_mode,
                                       bits=(1, 2, 3, 4), layer_start=0, num_layers=num_layers)
        # Load layers from end
        layers_end = load_all_layers(model_id, quantmode, rank_mode,
                                     bits=(1, 2, 3, 4), layer_start=-1, num_layers=num_layers)
        # Combine and deduplicate by layer index
        layer_dict = {}
        for l in layers_start:
            layer_dict[l.layer_idx] = l
        for l in layers_end:
            layer_dict[l.layer_idx] = l
        layers = list(layer_dict.values())
        layers.sort(key=lambda l: l.layer_idx)

        if not layers:
            ax.text(0.5, 0.5, "no cache", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(model_label(model_id))
            continue

        bits = sorted(set.intersection(*[set(L.by_bit.keys()) for L in layers]))
        bits = [b for b in bits if b > 0]

        n_layers = len(layers)
        cmap = plt.get_cmap("viridis", max(n_layers, 2))

        all_losses = []
        layer_L_inf_C = []

        for li, L in enumerate(layers):
            print(f"  Layer {L.layer_idx}")
            n_neurons = L.n_neurons
            dp_losses = []

            for s in bucket_counts:
                if s > n_neurons:
                    dp_losses.append(np.nan)
                    continue

                loss = _compute_dp_loss(L, bits, bpw, s)
                dp_losses.append(loss)

            dp_losses = np.asarray(dp_losses, dtype=float)
            all_losses.append((dp_losses, L.layer_idx, cmap(li)))

            color = cmap(li)
            # Plot DP loss curve for this layer
            ax.plot(bucket_counts, dp_losses, color=color, lw=2, marker="o",
                    ms=4, alpha=0.9, zorder=3,
                    label=f"L{L.layer_idx}")

            print(f"  DP losses: {dp_losses}")

        # Mark `expert` granularity (slice=1) and DartMoQP default (slice=8)
        ax.axvline(1, color="#cc7722", ls=":", lw=0.8, alpha=0.6)
        ax.axvline(8, color="#3a7ca5", ls=":", lw=0.8, alpha=0.6)

        # Draw theoretical L_inf and convergence curves if requested
        # if draw_L_inf:
        #     # Plot smooth theoretical curves
        #     m_smooth = np.logspace(0, np.log10(max(bucket_counts)), 100)
        #     for (L_inf, C, color, layer_idx), (dp_losses, _, _) in zip(layer_L_inf_C, all_losses):
        #         print(f"  Layer {layer_idx}:")
        #         print(f"    L_inf = {L_inf:.6g}, C = {C:.6g}")
        #         print(f"    DP loss at m={bucket_counts}: {dp_losses}")
        #         theory_at_points = [L_inf + C/m for m in bucket_counts]
        #         print(f"    L_inf + C/m at m={bucket_counts}: {theory_at_points}")

        #         # L(m) = L_inf + C/m
        #         L_theory = L_inf + C / m_smooth
        #         ax.plot(m_smooth, L_theory, color=color, lw=1.5, ls="--", alpha=0.7,
        #                 label=f"L{layer_idx}: $L^{{\\infty}}={L_inf:.3g}$")
        #         # Draw L_inf as horizontal line
        #         ax.axhline(L_inf, color=color, lw=1, ls=":", alpha=0.5)

        ax.set_xscale("log", base=2)
        # ax.set_yscale("log")
        ax.set_xticks([s for s in bucket_counts if s & (s - 1) == 0])
        ax.set_xticklabels([str(s) for s in bucket_counts if s & (s - 1) == 0],
                          fontsize=8)
        ax.set_xlabel("Buckets per expert (slice_expert_num)")
        ax.set_ylabel("DP loss (log scale)")
        # layer_str = f"first+last {num_layers} layers, total {len(layers)} layers"
        ax.set_title(f"{model_label(model_id)} @ {bpw}-bit")
        ax.legend(loc="lower right", fontsize=7, framealpha=0.5,
                  handlelength=1.5, borderpad=0.4)
        ax.grid(True, alpha=0.3, which="both")

    os.makedirs(out_dir, exist_ok=True)
    layer_suffix = f"_firstlast_{num_layers}"
    fp_png = os.path.join(out_dir, f"bucket_sweep_dp_loss_{rank_mode}_b{bpw}{layer_suffix}.png")
    plt.tight_layout()
    plt.savefig(fp_png)
    if save_pdf:
        fp_pdf = os.path.join(out_dir, f"bucket_sweep_dp_loss_{rank_mode}_b{bpw}{layer_suffix}.pdf")
        plt.savefig(fp_pdf)
    plt.close(fig)
    print(f"[bucket_sweep] saved figure: {fp_png}" + (f" and {fp_pdf}" if save_pdf else ""))

    return fp_png

def main():
    apply_paper_style()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="short cache id OR full model path; default: every model with cache")
    parser.add_argument("--quantmode", default=DEFAULT_QUANTMODE)
    parser.add_argument("--rank-mode", default=DEFAULT_RANK_MODE)
    parser.add_argument("--bit", type=int, default=DEFAULT_BIT)
    parser.add_argument("--num-layers", type=int, default=4,
                        help="number of layers to use from start and from end, default: 4")
    parser.add_argument("--pdf", action="store_true", default=False,
                        help="also save PDF copies alongside PNGs")
    args = parser.parse_args()

    if args.model:
        models = [resolve_model_id(args.model)]
    else:
        models = discover_models(args.quantmode, args.rank_mode)
        if not models:
            print(f"no models with cache under {args.quantmode}/{args.rank_mode}")
            return

    bucket_sweep(
        models, args.quantmode, args.rank_mode,
        bpw=args.bit,
        num_layers=args.num_layers,
        save_pdf=args.pdf,
    )

if __name__ == "__main__":
    main()
