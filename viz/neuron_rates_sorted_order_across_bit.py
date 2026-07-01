"""Neuron rates sorted order across bit visualization.

Shows how individual neuron losses change across bit widths, sorted by
a reference bit's loss. The sort_bit curve will be monotonic decreasing,
but other bits may not be!

All data sourced from the cached sensitivity tensors under
``intermediate_result/quant_outlier_{quantmode}/{rank_mode}/{model_id}/``;
no model is reloaded.

"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Make sibling modules importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dp_utils import extrapolate_0bit_loss_fix
from viz._cache_io import (
    LayerSensitivity, apply_paper_style, discover_layers, discover_models, load_layer,
    model_label, resolve_model_id,
)

OUT_ROOT = "plot/neuron_rates_sorted_order_across_bit"
DEFAULT_QUANTMODE = "turboquant"
DEFAULT_RANK_MODE = "turboquant_innerproduct"
DEFAULT_BITS = (1, 2, 3, 4)

BIT_COLORS = [
    (0.65, 0.65, 0.65),  # 0-bit
    (0.84, 0.96, 0.69),  # 1-bit
    (0.62, 0.89, 0.81),  # 2-bit
    (1.00, 0.75, 0.00),  # 3-bit
    (0.97, 0.55, 0.49),  # 4-bit
]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _load_rates_for_extrapolation(
    layer: LayerSensitivity,
    expert_idx: int,
    bits: Sequence[int],
) -> Dict[int, List[np.ndarray]]:
    """Format rates dict for extrapolate_0bit_loss_fix."""
    rates = {}
    for b in bits:
        if b in layer.by_bit:
            rates[b] = [layer.by_bit[b][expert_idx]]
    return rates


def _extrapolate_0bit_if_needed(
    layer: LayerSensitivity,
    expert_idx: int,
    bits: Sequence[int],
    quant_type: str = "turboquant",
) -> Tuple[List[int], Dict[int, np.ndarray]]:
    """Load available bits and extrapolate 0-bit if not present.

    Returns (bits_sorted, rates_dict) where rates_dict[b] is the loss array.
    """
    bits_available = sorted(set(bits) & set(layer.by_bit.keys()))
    rates_dict = {}
    for b in bits_available:
        rates_dict[b] = layer.by_bit[b][expert_idx]

    if 0 not in bits_available and len(bits_available) >= 2:
        rates_for_extrap = _load_rates_for_extrapolation(layer, expert_idx, bits_available)
        rates_0_list = extrapolate_0bit_loss_fix(
            rates_for_extrap, quant_type=quant_type, save_plots=False)
        if rates_0_list:
            rates_dict[0] = rates_0_list[0]
            bits_available = sorted([0] + bits_available)

    return bits_available, rates_dict


# ----------------------------------------------------------------------------
# neuron_rates_sorted_order_across_bit (single model)
# ----------------------------------------------------------------------------
def neuron_rates_sorted_order_across_bit_single(
    ax: plt.Axes,
    model_id: str,
    layer_idx: int,
    expert_idx: int = 0,
    sort_bit: int = 2,
    quantmode: str = DEFAULT_QUANTMODE,
    rank_mode: str = DEFAULT_RANK_MODE,
    bits: Tuple[int, ...] = DEFAULT_BITS,
    use_0bit: bool = True,
    max_neurons: int = 200,
) -> None:
    """Plot neuron rates sorted order across bit for a single model on given axes."""
    # If layer_idx == -1, use the last layer
    if layer_idx == -1:
        layers = discover_layers(quantmode, rank_mode, model_id)
        if not layers:
            ax.text(0.5, 0.5, "no layers", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{model_label(model_id)}", fontsize=10)
            return
        layer_idx = layers[-1]

    layer = load_layer(model_id, layer_idx, quantmode, rank_mode, bits=bits)
    if layer is None:
        ax.text(0.5, 0.5, "no cache", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{model_label(model_id)} L{layer_idx}", fontsize=10)
        return

    bits_sorted, rates_dict = _extrapolate_0bit_if_needed(
        layer, expert_idx, bits, quant_type=quantmode)
    if not bits_sorted:
        ax.text(0.5, 0.5, "no bits", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{model_label(model_id)} L{layer_idx}", fontsize=10)
        return

    if not use_0bit and 0 in rates_dict:
        rates_dict.pop(0)
        bits_sorted = sorted(bits_sorted)
        if 0 in bits_sorted:
            bits_sorted.remove(0)

    if sort_bit not in rates_dict:
        sort_bit = sorted(rates_dict.keys())[-1]

    # Sort neurons by sort_bit loss descending
    n_neurons = len(rates_dict[sort_bit])
    sorted_idx = np.argsort(-rates_dict[sort_bit])  # descending

    # Subsample if too many neurons
    if n_neurons > max_neurons:
        step = n_neurons // max_neurons
        sorted_idx = sorted_idx[::step]

    # Rates vs neuron index (lines per bit)
    x = np.arange(len(sorted_idx))
    for bit in bits_sorted:
        color = BIT_COLORS[bit]
        rate_values = rates_dict[bit][sorted_idx]
        ax.plot(x, rate_values, marker='', linestyle='-',
                 color=color, linewidth=1.5, alpha=0.7,
                 label=f"{bit}-bit")

    ax.set_xlabel(f"Neuron index (sorted by {sort_bit}-bit)", fontsize=9)
    ax.set_ylabel("Neuron loss (log scale)", fontsize=9)
    ax.set_title(f"{model_label(model_id)} L{layer_idx}", fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def neuron_rates_sorted_order_across_bit(
    model_id: Optional[str] = None,
    layer_idx: int = 0,
    expert_idx: int = 0,
    sort_bit: int = 2,
    quantmode: str = DEFAULT_QUANTMODE,
    rank_mode: str = DEFAULT_RANK_MODE,
    bits: Tuple[int, ...] = DEFAULT_BITS,
    use_0bit: bool = True,
    max_neurons: int = 200,
    out_dir: str = OUT_ROOT,
) -> str:
    """Neuron rates across bit widths, sorted by a reference bit.

    If model_id is None, will plot all discovered models in a row.
    If layer_idx == -1, use the last layer for each model.
    Always saves both PNG and PDF.
    """
    if model_id is None:
        # Plot all models
        models = discover_models(quantmode, rank_mode)
        if not models:
            print(f"[neuron_rates_sorted_order_across_bit] no models found")
            return ""

        n = len(models)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), squeeze=False)
        axes = axes[0]

        # Determine layer label for title
        layer_label = "last layer" if layer_idx == -1 else f"Layer {layer_idx}"

        for ax, model in zip(axes, models):
            neuron_rates_sorted_order_across_bit_single(
                ax, model, layer_idx, expert_idx, sort_bit,
                quantmode, rank_mode, bits, use_0bit, max_neurons
            )

        fig.suptitle(
            f"Neuron rates sorted order across bit (sorted by {sort_bit}-bit) — {layer_label}",
            fontsize=12, y=1.02,
        )

        os.makedirs(out_dir, exist_ok=True)
        layer_suffix = "last" if layer_idx == -1 else f"L{layer_idx}"
        fp_png = os.path.join(out_dir, f"neuron_rates_sorted_all_{layer_suffix}.png")
        fp_pdf = os.path.join(out_dir, f"neuron_rates_sorted_all_{layer_suffix}.pdf")
        plt.tight_layout()
        plt.savefig(fp_png)
        plt.savefig(fp_pdf)
        plt.close(fig)
        print(f"[neuron_rates_sorted_order_across_bit] saved {fp_png} and {fp_pdf}")
        return fp_png

    else:
        # Plot single model
        # If layer_idx == -1, use the last layer
        actual_layer_idx = layer_idx
        if layer_idx == -1:
            layers = discover_layers(quantmode, rank_mode, model_id)
            if not layers:
                print(f"[neuron_rates_sorted_order_across_bit] no layers found for {model_id}")
                return ""
            actual_layer_idx = layers[-1]

        layer = load_layer(model_id, actual_layer_idx, quantmode, rank_mode, bits=bits)
        if layer is None:
            print(f"[neuron_rates_sorted_order_across_bit] no cache found for {model_id} L{actual_layer_idx}")
            return ""

        bits_sorted, rates_dict = _extrapolate_0bit_if_needed(
            layer, expert_idx, bits, quant_type=quantmode)
        if not bits_sorted:
            print(f"[neuron_rates_sorted_order_across_bit] no usable bits in cache")
            return ""

        if not use_0bit and 0 in rates_dict:
            rates_dict.pop(0)
            bits_sorted = sorted(bits_sorted)
            if 0 in bits_sorted:
                bits_sorted.remove(0)

        if sort_bit not in rates_dict:
            sort_bit = sorted(rates_dict.keys())[-1]

        # Sort neurons by sort_bit loss descending
        n_neurons = len(rates_dict[sort_bit])
        sorted_idx = np.argsort(-rates_dict[sort_bit])  # descending

        # Subsample if too many neurons
        if n_neurons > max_neurons:
            step = n_neurons // max_neurons
            sorted_idx = sorted_idx[::step]

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        # Rates vs neuron index (lines per bit)
        x = np.arange(len(sorted_idx))
        for bit in bits_sorted:
            color = BIT_COLORS[bit]
            rate_values = rates_dict[bit][sorted_idx]
            ax.plot(x, rate_values, marker='', linestyle='-',
                     color=color, linewidth=1.5, alpha=0.7,
                     label=f"{bit}-bit")

        ax.set_xlabel(f"Neuron index (sorted by {sort_bit}-bit loss descending)")
        ax.set_ylabel("Neuron loss (log scale)")
        ax.set_title(f"Loss in sorted order\n(sort_bit={sort_bit} is monotonic, others may not be!)", fontsize=11)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")

        fig.suptitle(
            f"Neuron rates sorted order across bit — {model_label(model_id)} L{actual_layer_idx} Expert {expert_idx}",
            fontsize=12, y=1.02,
        )

        os.makedirs(out_dir, exist_ok=True)
        fp_png = os.path.join(out_dir, f"neuron_rates_sorted_{model_id}_L{actual_layer_idx}_exp{expert_idx}.png")
        fp_pdf = os.path.join(out_dir, f"neuron_rates_sorted_{model_id}_L{actual_layer_idx}_exp{expert_idx}.pdf")
        plt.tight_layout()
        plt.savefig(fp_png)
        plt.savefig(fp_pdf)
        plt.close(fig)
        print(f"[neuron_rates_sorted_order_across_bit] saved {fp_png} and {fp_pdf}")
        return fp_png


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    apply_paper_style()
    parser = argparse.ArgumentParser(
        description="Neuron rates sorted order across bit visualization")
    parser.add_argument("--model", default=None,
                        help="short cache id OR full model path; if not given, plot all models")
    parser.add_argument("--layer", type=int, default=0,
                        help="layer index")
    parser.add_argument("--expert", type=int, default=0,
                        help="expert index (only used for single model)")
    parser.add_argument("--sort-bit", type=int, default=2,
                        help="reference bit for sorting neurons")
    parser.add_argument("--max-neurons", type=int, default=200,
                        help="max neurons to show")
    parser.add_argument("--no-0bit", action="store_true",
                        help="don't extrapolate or show 0-bit")
    args = parser.parse_args()

    model_id = resolve_model_id(args.model) if args.model else None

    print("\n=== neuron_rates_sorted_order_across_bit ===")
    neuron_rates_sorted_order_across_bit(
        model_id, args.layer, args.expert,
        sort_bit=args.sort_bit, use_0bit=not args.no_0bit,
        max_neurons=args.max_neurons)


if __name__ == "__main__":
    main()
