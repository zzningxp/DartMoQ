"""Loss overlap and distribution visualizations.

Two motivation figures (each is one panel in the paper):

    overlap_quant_compare — *Observation 2.* Different quantization algorithms
                              (GPTQ vs TurboQuant) yield qualitatively different
                              per-neuron loss distributions for the same model
                              and layer. Tight clustering along the bit axis ⇒
                              uniform-ish distribution (low headroom). Wide
                              spread + heavy tail ⇒ heterogeneous distribution
                              (large headroom for mixed precision).
                              Side-by-side scatter, one panel per algorithm,
                              with a Gini coefficient in each title to quantify
                              the spread.

    overlap_quadratic_fit — *Observation 3.* Per-block log-loss is well-fit by
                              a quadratic in bit-width:
                                  log L(b) = p·b² + q·b + r.
                              Each (expert, block) gets a scatter of its
                              (bit, log loss) samples and the best-fit parabola
                              is overlaid. The median R² across all
                              (expert, block) fits is reported in the title.
                              This is the empirical foundation of the AM/GM
                              bound used in `viz.headroom.bucket_sweep`.

Both figures use the same scatter style as the legacy
`dp_utils.plot_block_losses_overlap`: x-axis = block-loss (log scale),
y-axis = bit-width, each expert offset slightly along the bit axis and
coloured by a viridis index so the per-expert structure is visible.

All data sourced from the cached sensitivity tensors under
``intermediate_result/quant_outlier_{quantmode}/{rank_mode}/{model_id}/``; no model is reloaded.

Usage
-----
    python -m viz.overlap_distribution                   # default model & layer
    python -m viz.overlap_distribution --model olmoe-7b-1b --layer 8
    python -m viz.overlap_distribution --skip quadratic_fit
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Make sibling modules importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dp_utils import get_sorted_and_block_losses
from viz._cache_io import (
    LayerSensitivity, apply_paper_style, load_layer, model_label,
    resolve_model_id,
)

OUT_ROOT = "plot/overlap_distribution"
DEFAULT_NUM_BLOCKS = 8


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _gini(x: np.ndarray) -> float:
    """Concentration index. 0 = uniform, 1 = single dominant element."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if x.size == 0:
        return float("nan")
    x = np.sort(x)
    n = x.size
    s = x.sum()
    if s == 0:
        return 0.0
    cum = np.cumsum(x)
    return (n + 1 - 2 * (cum.sum() / s)) / n


def _compute_block_losses(
    layer: LayerSensitivity,
    num_blocks: int,
    bits: Sequence[int],
) -> Tuple[List[int], dict, List[np.ndarray]]:
    """Compute per-expert block losses.

    Returns (bits_sorted, bit_to_idx, all_block_losses) where
    all_block_losses[e] has shape (n_bits, num_blocks).
    """
    bits_available = sorted(set(bits) & set(layer.by_bit.keys()))
    if not bits_available:
        return [], {}, []

    all_block_losses = []
    bit_to_idx = None
    for e in range(layer.n_experts):
        rates = {b: layer.by_bit[b][e] for b in bits_available}
        _, block_losses, bit_to_idx_e = get_sorted_and_block_losses(
            rates, bits_available, num_blocks)
        all_block_losses.append(block_losses)
        bit_to_idx = bit_to_idx_e   # same for every expert
    return bits_available, bit_to_idx, all_block_losses


def _block_loss_scatter(
    ax,
    bits_sorted: Sequence[int],
    bit_to_idx: dict,
    all_block_losses: List[np.ndarray],
    *,
    expert_cmap_name: str = "viridis",
    marker_size: float = 20.0,
    alpha: float = 0.5,
    show_legend: bool = True,
) -> None:
    """Render the block-loss scatter into ax (replicates plot_block_losses_overlap)."""
    n_experts = len(all_block_losses)
    if n_experts == 0:
        ax.text(0.5, 0.5, "no cache", ha="center", va="center",
                transform=ax.transAxes)
        return

    expert_spacing = 0.8 / max(n_experts, 1)
    expert_cmap = plt.get_cmap(expert_cmap_name, n_experts)

    # Draw scatter points (no labels here - we'll create legend manually)
    for bit in bits_sorted:
        for expert_idx, block_losses in enumerate(all_block_losses):
            loss_values = block_losses[bit_to_idx[bit], :]
            y_pos = float(bit) - 0.4 + expert_idx * expert_spacing + expert_spacing / 2
            r, g, b, _ = expert_cmap(expert_idx)
            ax.scatter(loss_values, np.full(len(loss_values), y_pos),
                       color=(r, g, b, alpha),
                       s=marker_size, alpha=alpha)

    # Create legend for experts (show a few representative ones, reversed order)
    if show_legend:
        n_experts = len(all_block_losses)
        legend_experts = min(16, n_experts)  # show up to 16 experts in legend
        step = max(1, n_experts // legend_experts)
        expert_indices = list(range(0, n_experts, step))[:legend_experts]
        for expert_idx in reversed(expert_indices):
            r, g, b, _ = expert_cmap(expert_idx)
            ax.scatter([], [], color=(r, g, b, alpha), s=marker_size,
                       label=f"Expert {expert_idx}")


# ----------------------------------------------------------------------------
# Observation 2 — different quantizers, different distributions
# ----------------------------------------------------------------------------
def overlap_quant_compare(
    model_id: str,
    layer_idx: int,
    quants: Sequence[Tuple[str, str, str]] = (
        ("GPTQ",       "gptq",       "gptq_quant_outlier"),
        ("TurboQuant", "turboquant", "turboquant_innerproduct"),
        ("TurboQuant-MSE", "turboquant", "turboquant_mse"),
    ),
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    bits: Tuple[int, ...] = (0, 1, 2, 3, 4),  # note: bit 0 is used for scatter,
    # but global x-range is computed from bit>0 to avoid axis stretching
    out_dir: str = OUT_ROOT,
    use_pdf: bool = False,
) -> str:
    """Compare per-neuron sensitivity distributions across quant algorithms.

    For one (model, layer), side-by-side panels (one per quant algorithm) overlay
    the block-losses of all experts at each bit. Tight clusters along the bit
    axis ⇒ uniform-ish distribution (low headroom). Wide spread + heavy tail
    ⇒ heterogeneous distribution (mixed precision pays off).

    Reports a Gini index per panel (computed over all block-loss values flattened
    across experts and bits>0) so the visual difference is also a single number.
    """
    n_q = len(quants)
    fig, axes = plt.subplots(1, n_q, figsize=(6.5 * n_q, 4.5), sharey=True)
    if n_q == 1:
        axes = [axes]

    # First pass: load everything, compute block losses, gather global x-range.
    per_quant_data = []
    all_loss_vals = []
    for name, quantmode, rank_mode in quants:
        layer = load_layer(model_id, layer_idx, quantmode, rank_mode, bits=bits)
        if layer is None:
            per_quant_data.append((name, None))
            continue
        bits_sorted, bit_to_idx, all_block_losses = _compute_block_losses(
            layer, num_blocks, bits)
        per_quant_data.append((name, (bits_sorted, bit_to_idx, all_block_losses)))
        # collect for global x-axis range and Gini (skip bit=0 — it's the
        # full-zero baseline and would stretch x-axis to oblivion)
        for bl in all_block_losses:
            for bit in bits_sorted:
                if bit == 0:
                    continue
                all_loss_vals.append(bl[bit_to_idx[bit], :].flatten())

    if not all_loss_vals:
        print(f"[dist_quant_compare] no cache found for {model_id} L{layer_idx}")
        plt.close(fig)
        return ""

    concat_loss = np.concatenate(all_loss_vals)
    concat_loss = concat_loss[concat_loss > 0]
    if concat_loss.size == 0:
        global_min = global_max = None
    else:
        global_min = float(np.min(concat_loss)) * 0.9
        # Use max from bit>0 as the main anchor, but reserve some headroom
        # for bit 0 points so they are not completely cut off. Cap the
        # expansion at 3x so the x-axis doesn't blow to infinity.
        bit0_losses = []
        for _, data in per_quant_data:
            if data is not None:
                _, bit_to_idx, all_block_losses = data
                if 0 in bit_to_idx:
                    for bl in all_block_losses:
                        bit0_losses.append(bl[bit_to_idx[0], :].flatten())
        bit0_max = 1.0
        if bit0_losses:
            bit0_flat = np.concatenate(bit0_losses)
            bit0_flat = bit0_flat[bit0_flat > 0]
            if bit0_flat.size > 0:
                bit0_max = float(np.max(bit0_flat))
        main_max = float(np.max(concat_loss))
        global_max = max(main_max, bit0_max * 0.5) * 1.1  # give bit0 some room
        global_max = min(global_max, 3.0 * main_max)     # cap expansion

    # Second pass: scatter + title per quant
    for ax, (name, data) in zip(axes, per_quant_data):
        if data is None:
            ax.text(0.5, 0.5, "no cache", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(name, fontsize=11)
            continue
        bits_sorted, bit_to_idx, all_block_losses = data

        # Gini per-bit (within the same bit, across blocks)
        per_bit_gini = {}
        all_losses = []
        all_bits = []
        for bit in bits_sorted:
            if bit == 0:
                continue
            bit_losses = np.concatenate([
                all_block_losses[e][bit_to_idx[bit], :]
                for e in range(len(all_block_losses))
            ])
            per_bit_gini[bit] = _gini(bit_losses)
            all_losses.extend(bit_losses)
            all_bits.extend([bit] * len(bit_losses))
        avg_per_bit_gini = np.mean(list(per_bit_gini.values()))

        # Print debug info
        print(f"\n--- {name} ---")
        print(f"  Bit range: {bits_sorted}")
        print(f"  Total blocks: {len(all_losses)}")
        for bit in sorted(per_bit_gini.keys()):
            bit_losses = [l for l, b in zip(all_losses, all_bits) if b == bit]
            print(f"  {bit}-bit: {len(bit_losses)} blocks, "
                  f"loss: min={np.min(bit_losses):.3e}, "
                  f"median={np.median(bit_losses):.3e}, "
                  f"max={np.max(bit_losses):.3e}, "
                  f"Gini={per_bit_gini[bit]:.3f}")
        print(f"  Avg per-bit Gini: {avg_per_bit_gini:.3f}")
        print(f"  [Key] High = blocks are distinguishable (good for DP), Low = all blocks same (bad for DP)")

        # Only show legend on the last plot
        is_last = (ax == axes[-1])
        _block_loss_scatter(ax, bits_sorted, bit_to_idx, all_block_losses, show_legend=is_last)

        ax.set_xscale("log")
        if global_min is not None:
            ax.set_xlim(global_min, global_max)
        # Ensure all bits are visible and equally spaced
        ax.set_ylim(-0.5, max(bits_sorted) + 0.5)
        ax.set_xlabel("Block loss (log scale)")
        ax.set_title(f"{name}\n(Per-bit Gini = {avg_per_bit_gini:.2f})", fontsize=11)
        ax.grid(True, alpha=0.3, axis="x")
        ax.legend(markerscale=1.5, fontsize=8, loc="upper right")

    axes[0].set_ylabel("Bit width")
    # set y ticks to integer bits
    used_bits = []
    for _, data in per_quant_data:
        if data is not None:
            used_bits = data[0]
            break
    if used_bits:
        axes[0].set_yticks(used_bits, [f"{b}-bit" for b in used_bits])

    fig.suptitle(
        f"Per-block loss distribution across quantizers — "
        f"{model_label(model_id)} L{layer_idx}, {num_blocks} blocks/expert",
        fontsize=11, y=1.02,
    )

    os.makedirs(out_dir, exist_ok=True)
    ext = "pdf" if use_pdf else "png"
    fp = os.path.join(out_dir, f"quant_compare_{model_id}_L{layer_idx}.{ext}")
    plt.tight_layout()
    plt.savefig(fp)
    plt.close(fig)
    print(f"[overlap_quant_compare] saved {fp}")
    return fp


# ----------------------------------------------------------------------------
# Observation 3 — log-quadratic fit
# ----------------------------------------------------------------------------
def overlap_quadratic_fit(
    model_id: str,
    layer_idx: int,
    quantmode: str = "turboquant",
    rank_mode: str = "turboquant_innerproduct",
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    bits: Tuple[int, ...] = (1, 2, 3, 4),
    out_dir: str = OUT_ROOT,
    use_pdf: bool = False,
) -> str:
    """Evidence that per-block log-loss is well-fit by a quadratic in bit-width.

    For each (expert, block), fit
        log L(b) = p·b² + q·b + r
    using the 4 cached integer bits (default 1..4) and report the median R²
    across all (expert, block) pairs. Overlay the fitted parabolas on the
    scatter so the reader can eyeball goodness of fit.

    0-bit is excluded because its loss is often exactly zero (the full-zero
    baseline) which breaks log-fits.
    """
    layer = load_layer(model_id, layer_idx, quantmode, rank_mode, bits=bits)
    if layer is None:
        print(f"[dist_quadratic_fit] no cache for {model_id} L{layer_idx} "
              f"({quantmode}/{rank_mode})")
        return ""

    bits_sorted, bit_to_idx, all_block_losses = _compute_block_losses(
        layer, num_blocks, bits)
    if not bits_sorted:
        print(f"[dist_quadratic_fit] no usable bits in cache")
        return ""

    bits_arr = np.asarray(bits_sorted, dtype=float)
    r2_values = []
    fit_curves = []  # list of (bit_grid, log_loss_grid) for plotting

    bit_grid = np.linspace(bits_arr.min(), bits_arr.max(), 50)

    for e, block_losses in enumerate(all_block_losses):
        for k in range(block_losses.shape[1]):
            y = block_losses[:, k].astype(float)   # shape (n_bits,)
            mask = y > 0
            if mask.sum() < 3:
                continue
            x = bits_arr[mask]
            log_y = np.log(y[mask])
            try:
                # polyfit returns [p, q, r] for ax^2 + bx + c
                coeffs = np.polyfit(x, log_y, deg=2)
            except (np.linalg.LinAlgError, ValueError):
                continue
            pred = np.polyval(coeffs, x)
            ss_res = float(np.sum((log_y - pred) ** 2))
            ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            r2_values.append(r2)
            fit_curves.append((bit_grid, np.polyval(coeffs, bit_grid)))

    median_r2 = float(np.nanmedian(r2_values)) if r2_values else float("nan")

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    _block_loss_scatter(ax, bits_sorted, bit_to_idx, all_block_losses,
                        alpha=0.35)

    # Overlay fitted parabolas (log L vs bit) → on plot: x = exp(log_loss), y = bit
    # Use a single soft red so the overlay reads as "trend" rather than another
    # data series. Draw only a subset (first N experts * first M blocks) so
    # the plot doesn't become spaghetti.
    draw_every_n = max(1, len(fit_curves) // 20)  # about 20 curves total
    for i, (bg, log_lg) in enumerate(fit_curves):
        if i % draw_every_n != 0:
            continue
        # bg is the bit grid (y on this plot), log_lg gives log loss (x on this plot)
        ax.plot(np.exp(log_lg), bg, color="#b5132e",
                alpha=0.4, lw=0.8, zorder=4)

    ax.set_xscale("log")
    ax.set_ylim(-0.5, max(bits_sorted) + 0.5)
    ax.set_xlabel("Block loss (log scale)")
    ax.set_ylabel("Bit width")
    ax.set_yticks(bits_sorted, [f"{b}-bit" for b in bits_sorted])
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend(markerscale=1.5, fontsize=8, loc="upper right")

    label = f"{model_label(model_id)} L{layer_idx}"
    ax.set_title(
        f"log L(b) = p·b² + q·b + r fit  —  {label}\n"
        f"median R² = {median_r2:.3f} across "
        f"{len(r2_values)} (expert × block) pairs  ·  red curves = fitted parabolas",
        fontsize=10,
    )

    os.makedirs(out_dir, exist_ok=True)
    ext = "pdf" if use_pdf else "png"
    fp = os.path.join(out_dir, f"quadratic_fit_{model_id}_L{layer_idx}.{ext}")
    plt.tight_layout()
    plt.savefig(fp)
    plt.close(fig)
    print(f"[overlap_quadratic_fit] saved {fp}  |  median R² = {median_r2:.3f}")
    return fp


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    apply_paper_style()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-v1-moe-16b",
                        help="short cache id OR full model path")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--num-blocks", type=int, default=DEFAULT_NUM_BLOCKS)
    parser.add_argument("--skip", nargs="+", default=[],
                        choices=["quant_compare", "quadratic_fit"],
                        help="panel names to skip")
    parser.add_argument("--pdf", action="store_true",
                        help="save as PDF instead of PNG")
    args = parser.parse_args()

    model_id = resolve_model_id(args.model)

    if "quant_compare" not in args.skip:
        print("\n=== overlap_quant_compare ===")
        overlap_quant_compare(model_id, args.layer, num_blocks=args.num_blocks, use_pdf=args.pdf)

    if "quadratic_fit" not in args.skip:
        print("\n=== overlap_quadratic_fit ===")
        overlap_quadratic_fit(model_id, args.layer, num_blocks=args.num_blocks, use_pdf=args.pdf)


if __name__ == "__main__":
    main()
