"""High-sensitivity neuron hypothesis validation.

This module implements visualization experiments to validate the hypothesis that
high-sensitivity neurons are the main cause of:
1) TurboQuant's higher ppl compared to GPTQ
2) TurboQuant's high seed sensitivity
3) DartMoQ's improvement after DP slicing

Two core experiments:
- sens_distribution: Compare TurboQuant vs GPTQ sensitivity distributions across experts
- dp_grouping: Analyze how DP grouping improves within-group homogeneity across experts
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional, Sequence, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dartmoq_utils import construct_experts_by_rates
from viz._cache_io import (
    LayerSensitivity,
    apply_paper_style,
    discover_layers,
    discover_models,
    expert_total_loss,
    load_all_layers,
    load_layer,
    model_label,
    neuron_loss_matrix,
    resolve_model_id,
)

OUT_ROOT = "plot/high_sensitivity_analysis"
DEFAULT_QUANTMODES = ["turboquant", "gptq"]
DEFAULT_RANK_MODES = {
    "turboquant": "turboquant_innerproduct",
    "gptq": "gptq_quant_outlier",
}
DEFAULT_BIT = 2
DEFAULT_SLICE_EXPERT_NUM = 8


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def _gini(x: np.ndarray) -> float:
    """Compute Gini coefficient of array."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if x.size == 0:
        return float("nan")
    s = float(x.sum())
    if s <= 0:
        return 0.0
    x = np.sort(x)[::-1]
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum.sum() / s)) / n)


def _topk_shares(x: np.ndarray, fractions: Sequence[float] = (0.01, 0.05, 0.10, 0.20, 0.50)) -> Dict[str, float]:
    """Compute what fraction of total is held by top-N% elements."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    total = float(x.sum())
    if total <= 0:
        return {f"{int(f*100)}%": float("nan") for f in fractions}
    sorted_x = np.sort(x)[::-1]
    shares = {}
    for f in fractions:
        k = max(1, int(np.ceil(f * x.size)))
        shares[f"{int(f*100)}%"] = float(sorted_x[:k].sum() / total)
    return shares


# ---------------------------------------------------------------------------
# Experiment 1: Sensitivity distribution comparison (multi-expert)
# ---------------------------------------------------------------------------
def plot_multi_expert_sens_distribution(
    models: List[str],
    tq_layers_dict: Dict[str, Sequence[LayerSensitivity]],
    gptq_layers_dict: Dict[str, Sequence[LayerSensitivity]],
    bit: int,
    targets_dict: Dict[str, Sequence[Tuple[int, int]]],
    out_dir: str,
    save_pdf: bool = False,
) -> str:
    """Plot multi-model, multi-expert sensitivity distribution comparison.

    Layout:
    - Top row: sensitivity distribution histograms (one panel per model)
    - Bottom row: Lorenz curves (one panel per model)
    """
    n_models = len(models)
    fig, axes = plt.subplots(2, n_models, figsize=(5.5 * n_models, 10))
    if n_models == 1:
        axes = axes.reshape(2, 1)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for col_idx, model_id in enumerate(models):
        tq_layers = tq_layers_dict.get(model_id, [])
        gptq_layers = gptq_layers_dict.get(model_id, [])
        targets = targets_dict.get(model_id, [])

        if not tq_layers or not gptq_layers or not targets:
            axes[0, col_idx].axis("off")
            axes[1, col_idx].axis("off")
            axes[0, col_idx].text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        # Build layer/expert dictionaries
        tq_dict = {l.layer_idx: l for l in tq_layers}
        gptq_dict = {l.layer_idx: l for l in gptq_layers}

        # --- Top: sensitivity distribution ---
        ax = axes[0, col_idx]
        bins = None
        for idx, (layer_idx, expert_idx) in enumerate(targets):
            if layer_idx not in tq_dict or layer_idx not in gptq_dict:
                continue
            tq_sens = tq_dict[layer_idx].by_bit[bit][expert_idx]
            gptq_sens = gptq_dict[layer_idx].by_bit[bit][expert_idx]

            log_tq = np.log10(np.clip(tq_sens, 1e-30, None))
            log_gptq = np.log10(np.clip(gptq_sens, 1e-30, None))

            if bins is None:
                # Compute shared bins from first expert
                min_val = min(log_tq.min(), log_gptq.min())
                max_val = max(log_tq.max(), log_gptq.max())
                bins = np.linspace(min_val, max_val, 55)

            # Plot TurboQuant
            ax.hist(log_tq, bins=bins, alpha=0.4, color="#b5132e",
                    histtype="stepfilled", linewidth=0,
                    label=f"L{layer_idx}E{expert_idx} (TQ)" if idx == 0 else None)
            # Plot GPTQ
            ax.hist(log_gptq, bins=bins, alpha=0.4, color="#3a7ca5",
                    histtype="stepfilled", linewidth=0,
                    label=f"L{layer_idx}E{expert_idx} (GPTQ)" if idx == 0 else None)

        # Add legend for TQ vs GPTQ (not per-expert)
        ax.plot([], [], color="#b5132e", lw=4, alpha=0.7, label="TurboQuant")
        ax.plot([], [], color="#3a7ca5", lw=4, alpha=0.7, label="GPTQ")
        ax.set_xlabel("log10(sensitivity)")
        ax.set_ylabel("Neuron count")
        ax.set_title(f"Sensitivity distribution — {model_label(model_id)}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        # --- Bottom: Lorenz curves ---
        ax = axes[1, col_idx]
        for idx, (layer_idx, expert_idx) in enumerate(targets):
            if layer_idx not in tq_dict or layer_idx not in gptq_dict:
                continue
            tq_sens = tq_dict[layer_idx].by_bit[bit][expert_idx]
            gptq_sens = gptq_dict[layer_idx].by_bit[bit][expert_idx]

            tq_sorted = np.sort(tq_sens)[::-1]
            gptq_sorted = np.sort(gptq_sens)[::-1]
            tq_cum = np.cumsum(tq_sorted) / max(float(tq_sorted.sum()), 1e-30)
            gptq_cum = np.cumsum(gptq_sorted) / max(float(gptq_sorted.sum()), 1e-30)
            x = np.arange(len(tq_cum)) / len(tq_cum)

            color = colors[idx % len(colors)]
            ax.plot(x, tq_cum, color=color, lw=1.5, alpha=0.8, linestyle="-",
                    label=f"L{layer_idx}E{expert_idx} (TQ)")
            ax.plot(x, gptq_cum, color=color, lw=1.5, alpha=0.8, linestyle="--",
                    label=f"L{layer_idx}E{expert_idx} (GPTQ)")

        ax.plot([0, 1], [0, 1], color="k", ls=":", lw=1.5, alpha=0.7, label="Uniform")
        ax.set_xlabel("Fraction of neurons (sorted desc by sensitivity)")
        ax.set_ylabel("Cumulative sensitivity share")
        ax.set_title(f"Lorenz curve — {model_label(model_id)}")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Sensitivity distribution analysis — {bit}-bit", y=0.995, fontsize=14)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    model_tag = "_".join(models)
    if len(model_tag) > 60:
        model_tag = f"{len(models)}models"
    fp_png = os.path.join(out_dir, f"multi_expert_sens_distribution_{model_tag}_b{bit}.png")
    fig.savefig(fp_png, dpi=200)
    if save_pdf:
        fig.savefig(os.path.splitext(fp_png)[0] + ".pdf")
    plt.close(fig)
    print(f"[sens_distribution] saved {fp_png}")

    return fp_png


def plot_sens_metrics_table(
    model_id: str,
    tq_layers: Sequence[LayerSensitivity],
    gptq_layers: Sequence[LayerSensitivity],
    bit: int,
    targets: Sequence[Tuple[int, int]],
    out_dir: str,
    save_pdf: bool = False,
) -> str:
    """Plot a table of sensitivity metrics per expert."""
    tq_dict = {l.layer_idx: l for l in tq_layers}
    gptq_dict = {l.layer_idx: l for l in gptq_layers}

    table_data = [
        ["Expert", "Quant", "Gini", "Top 1%", "Top 5%", "Top 10%", "Top 20%"],
    ]

    for layer_idx, expert_idx in targets:
        if layer_idx not in tq_dict or layer_idx not in gptq_dict:
            continue
        tq_sens = tq_dict[layer_idx].by_bit[bit][expert_idx]
        gptq_sens = gptq_dict[layer_idx].by_bit[bit][expert_idx]

        tq_gini = _gini(tq_sens)
        tq_shares = _topk_shares(tq_sens)
        gptq_gini = _gini(gptq_sens)
        gptq_shares = _topk_shares(gptq_sens)

        table_data.append([
            f"L{layer_idx}E{expert_idx}",
            "TQ",
            f"{tq_gini:.3f}" if np.isfinite(tq_gini) else "nan",
            f"{tq_shares['1%']:.1%}",
            f"{tq_shares['5%']:.1%}",
            f"{tq_shares['10%']:.1%}",
            f"{tq_shares['20%']:.1%}",
        ])
        table_data.append([
            "",
            "GPTQ",
            f"{gptq_gini:.3f}" if np.isfinite(gptq_gini) else "nan",
            f"{gptq_shares['1%']:.1%}",
            f"{gptq_shares['5%']:.1%}",
            f"{gptq_shares['10%']:.1%}",
            f"{gptq_shares['20%']:.1%}",
        ])

    # Add average row
    tq_ginis = []
    tq_1p = []
    gptq_ginis = []
    gptq_1p = []
    for layer_idx, expert_idx in targets:
        if layer_idx not in tq_dict or layer_idx not in gptq_dict:
            continue
        tq_sens = tq_dict[layer_idx].by_bit[bit][expert_idx]
        gptq_sens = gptq_dict[layer_idx].by_bit[bit][expert_idx]
        tq_ginis.append(_gini(tq_sens))
        tq_1p.append(_topk_shares(tq_sens)["1%"])
        gptq_ginis.append(_gini(gptq_sens))
        gptq_1p.append(_topk_shares(gptq_sens)["1%"])

    if tq_ginis:
        table_data.append(["---"] * 7)
        table_data.append([
            "AVG",
            "TQ",
            f"{np.mean(tq_ginis):.3f}",
            f"{np.mean(tq_1p):.1%}",
            "", "", "",
        ])
        table_data.append([
            "",
            "GPTQ",
            f"{np.mean(gptq_ginis):.3f}",
            f"{np.mean(gptq_1p):.1%}",
            "", "", "",
        ])
        table_data.append([
            "",
            "Ratio",
            f"{np.mean(tq_ginis)/np.mean(gptq_ginis):.2f}",
            f"{np.mean(tq_1p)/np.mean(gptq_1p):.2f}",
            "", "", "",
        ])

    fig, ax = plt.subplots(1, 1, figsize=(12, 3 + 0.4 * len(table_data)))
    ax.axis("off")
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Format table
    for i in range(len(table_data)):
        for j in range(7):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor("#e0e0e0")
                cell.set_text_props(weight="bold")
            elif table_data[i][0] == "---":
                cell.set_facecolor("#f0f0f0")
            elif table_data[i][1] == "TQ":
                cell.set_facecolor("#ffe0e0")
            elif table_data[i][1] == "GPTQ":
                cell.set_facecolor("#e0e0ff")
            elif table_data[i][1] == "Ratio":
                cell.set_facecolor("#e0ffe0")
                # Highlight ratio if >1.2 or <0.8
                for j_col in [2, 3]:
                    val_str = table_data[i][j_col]
                    try:
                        val = float(val_str)
                        if val > 1.2:
                            table[(i, j_col)].set_facecolor("#ff8080")
                        elif val < 0.8:
                            table[(i, j_col)].set_facecolor("#80ff80")
                    except:
                        pass

    fig.suptitle(f"Sensitivity metrics summary — {model_label(model_id)}", y=0.98, fontsize=13)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fp = os.path.join(out_dir, f"sens_metrics_table_{model_id}_b{bit}.png")
    fig.savefig(fp, dpi=150)
    if save_pdf:
        fig.savefig(os.path.splitext(fp)[0] + ".pdf")
    plt.close(fig)
    print(f"[sens_metrics] saved {fp}")
    return fp


# ---------------------------------------------------------------------------
# Experiment 2: DP grouping analysis (multi-expert)
# ---------------------------------------------------------------------------
def plot_multi_expert_dp_grouping(
    models: List[str],
    tq_layers_dict: Dict[str, Sequence[LayerSensitivity]],
    gptq_layers_dict: Dict[str, Sequence[LayerSensitivity]],
    bit: int,
    targets_dict: Dict[str, Sequence[Tuple[int, int]]],
    slice_expert_num: int = 8,
    out_dir: str = "",
    save_pdf: bool = False,
) -> str:
    """Plot DP grouping comparison between TurboQuant and GPTQ.

    Layout: one column per model, each panel shows side-by-side comparison
    of TurboQuant and GPTQ per-group sensitivity distributions.
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5), squeeze=False)
    axes = axes[0]
    colors_tq = "#b5132e"
    colors_gptq = "#3a7ca5"

    for col_idx, model_id in enumerate(models):
        ax = axes[col_idx]
        tq_layers = tq_layers_dict.get(model_id, [])
        gptq_layers = gptq_layers_dict.get(model_id, [])
        targets = targets_dict.get(model_id, [])

        if not tq_layers or not gptq_layers or not targets:
            ax.axis("off")
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        tq_dict = {l.layer_idx: l for l in tq_layers}
        gptq_dict = {l.layer_idx: l for l in gptq_layers}

        # Take the first target for simplicity
        layer_idx, expert_idx = targets[0]

        if layer_idx not in tq_dict or layer_idx not in gptq_dict:
            ax.axis("off")
            ax.text(0.5, 0.5, "Layer not found", ha="center", va="center")
            continue

        tq_sens = tq_dict[layer_idx].by_bit[bit][expert_idx]
        gptq_sens = gptq_dict[layer_idx].by_bit[bit][expert_idx]

        # Build DP groups (using TurboQuant sensitivity to define groups)
        rates = torch.as_tensor(tq_sens, dtype=torch.float32)
        groups, _ = construct_experts_by_rates(rates, slice_expert_num)
        groups = groups[1:]  # remove dummy first group

        # Get group data for both quant methods using the same group assignments
        tq_group_data = [tq_sens[np.asarray(g, dtype=int)] for g in groups]
        gptq_group_data = [gptq_sens[np.asarray(g, dtype=int)] for g in groups]

        # Position offset for side-by-side boxplots
        group_positions = np.arange(1, len(groups) + 1)
        offset = 0.25

        # Plot TurboQuant
        bp_tq = ax.boxplot(tq_group_data, positions=group_positions - offset, widths=0.4,
                           patch_artist=True, boxprops=dict(facecolor=colors_tq, alpha=0.7),
                           medianprops=dict(color="black"))
        # Plot GPTQ
        bp_gptq = ax.boxplot(gptq_group_data, positions=group_positions + offset, widths=0.4,
                            patch_artist=True, boxprops=dict(facecolor=colors_gptq, alpha=0.7),
                            medianprops=dict(color="black"))

        ax.set_xticks(group_positions)
        ax.set_xticklabels([f"G{i}" for i in range(len(groups))])
        ax.set_yscale("log")
        ax.set_ylabel("Sensitivity (log)")
        ax.set_title(f"{model_label(model_id)} — L{layer_idx}E{expert_idx}")
        ax.grid(True, alpha=0.3, axis="y")

        # Add legend
        ax.legend([bp_tq["boxes"][0], bp_gptq["boxes"][0]], ["TurboQuant", "GPTQ"], fontsize=9)

        # Add Gini text
        tq_full_gini = _gini(tq_sens)
        gptq_full_gini = _gini(gptq_sens)
        tq_group_ginis = [_gini(gd) for gd in tq_group_data]
        gptq_group_ginis = [_gini(gd) for gd in gptq_group_data]
        tq_avg_gini = np.mean([g for g in tq_group_ginis if np.isfinite(g)])
        gptq_avg_gini = np.mean([g for g in gptq_group_ginis if np.isfinite(g)])
        tq_reduction = 1.0 - tq_avg_gini / max(tq_full_gini, 1e-30)
        gptq_reduction = 1.0 - gptq_avg_gini / max(gptq_full_gini, 1e-30)

        info_text = (f"TQ Gini: {tq_full_gini:.3f} → {tq_avg_gini:.3f} (red {tq_reduction:.1%})\n"
                    f"GPTQ Gini: {gptq_full_gini:.3f} → {gptq_avg_gini:.3f} (red {gptq_reduction:.1%})")
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7), fontsize=8)

    fig.suptitle(f"DP Grouping Comparison (TurboQuant vs GPTQ) — {bit}-bit", y=1.02, fontsize=14)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    model_tag = "_".join(models)
    if len(model_tag) > 60:
        model_tag = f"{len(models)}models"
    fp_png = os.path.join(out_dir, f"multi_expert_dp_grouping_{model_tag}_b{bit}.png")
    fig.savefig(fp_png, dpi=180)
    if save_pdf:
        fig.savefig(os.path.splitext(fp_png)[0] + ".pdf")
    plt.close(fig)
    print(f"[dp_grouping] saved {fp_png}")
    return fp_png


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------
def select_target_experts(
    model_id: str,
    quantmode: str,
    rank_mode: str,
    bit: int,
    layers: Optional[Sequence[int]] = None,
    layer_start: Optional[int] = 0,
    num_layers: Optional[int] = 4,
    experts_per_layer: int = 1,
) -> Sequence[Tuple[int, int]]:
    """Select target experts based on sensitivity cache.

    Layer selection logic matches headroom.py:
    - If layers specified: use those layers
    - Else if layer_start and num_layers: use those
      * layer_start = -1 means from the end
    """
    available_layers = discover_layers(quantmode, rank_mode, model_id)
    if not available_layers:
        return []

    target_layers = []
    if layers:
        target_layers = [l for l in layers if l in available_layers]
    elif layer_start is not None:
        if layer_start == -1:
            # Last N layers
            if num_layers is not None:
                target_layers = available_layers[-num_layers:] if num_layers > 0 else []
            else:
                target_layers = available_layers[-4:]  # default 4
        else:
            # From layer_start onward, up to num_layers
            candidates = [l for l in available_layers if l >= layer_start]
            if num_layers is not None:
                target_layers = candidates[:num_layers]
            else:
                target_layers = candidates[:4]  # default 4
    else:
        # Default: first N layers
        target_layers = available_layers[:(num_layers or 4)]

    targets = []
    for layer_idx in target_layers:
        layer = load_layer(model_id, layer_idx, quantmode, rank_mode, bits=(bit,))
        if layer is None or bit not in layer.by_bit:
            continue
        # Select top experts by total sensitivity
        expert_sums = expert_total_loss(layer, bit)
        top_experts = np.argsort(expert_sums)[::-1][:experts_per_layer]
        for e in top_experts:
            targets.append((layer_idx, e))

    return targets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    apply_paper_style()
    parser = argparse.ArgumentParser(
        description="High-sensitivity neuron hypothesis validation experiments"
    )
    parser.add_argument("--model", default=None,
                        help="short cache id OR full model path; default: all with cache")
    parser.add_argument("--bit", type=int, default=DEFAULT_BIT,
                        help="bit width for sensitivity cache")
    parser.add_argument("--slice-expert-num", type=int, default=DEFAULT_SLICE_EXPERT_NUM,
                        help="number of groups per expert for DP analysis")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="specific layers to analyze")
    parser.add_argument("--layer-start", type=int, default=0,
                        help="start layer index (inclusive) to use; -1 means the last num_layers layers")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="number of layers to use")
    parser.add_argument("--experts-per-layer", type=int, default=1,
                        help="number of top experts per layer to analyze")
    parser.add_argument("--experiment", choices=["sens_distribution", "dp_grouping"],
                        action="append", default=[],
                        help="specific experiments to run; default: both")
    parser.add_argument("--out-dir", default=OUT_ROOT)
    parser.add_argument("--pdf", action="store_true", default=False,
                        help="also save PDF copies alongside PNGs")
    args = parser.parse_args()

    # Determine which experiments to run
    experiments = args.experiment if args.experiment else ["sens_distribution", "dp_grouping"]

    # Discover models
    if args.model:
        models = [resolve_model_id(args.model)]
    else:
        models = discover_models("turboquant", DEFAULT_RANK_MODES["turboquant"])
        if not models:
            print("No cached models found")
            return

    # Load data for all models first
    tq_rank = DEFAULT_RANK_MODES["turboquant"]
    gptq_rank = DEFAULT_RANK_MODES["gptq"]

    tq_layers_dict = {}
    gptq_layers_dict = {}
    targets_dict = {}
    valid_models = []

    for model_id in models:
        print(f"\n{'='*70}")
        print(f"Model: {model_label(model_id)}")
        print(f"{'='*70}")

        # Load both TurboQuant and GPTQ caches
        tq_layers = load_all_layers(model_id, "turboquant", tq_rank, bits=(args.bit,))
        gptq_layers = load_all_layers(model_id, "gptq", gptq_rank, bits=(args.bit,))

        if not tq_layers:
            print(f"No TurboQuant cache found for {model_id}")
            continue
        if not gptq_layers:
            print(f"No GPTQ cache found for {model_id}")
            continue

        # Select target experts (using TurboQuant sensitivity)
        targets = select_target_experts(
            model_id, "turboquant", tq_rank, args.bit,
            layers=args.layers, layer_start=args.layer_start, num_layers=args.num_layers,
            experts_per_layer=args.experts_per_layer
        )
        if not targets:
            print("No targets found")
            continue
        print(f"Selected {len(targets)} targets: {targets}")

        valid_models.append(model_id)
        tq_layers_dict[model_id] = tq_layers
        gptq_layers_dict[model_id] = gptq_layers
        targets_dict[model_id] = targets

    if not valid_models:
        print("\nNo valid models found")
        return

    # Experiment 1: Sensitivity distribution comparison (all models together)
    if "sens_distribution" in experiments:
        print(f"\n--- Sensitivity distribution comparison ({len(valid_models)} models) ---")
        plot_multi_expert_sens_distribution(
            valid_models, tq_layers_dict, gptq_layers_dict, args.bit, targets_dict,
            args.out_dir, args.pdf
        )

    # Experiment 2: DP grouping analysis (compare TQ vs GPTQ)
    if "dp_grouping" in experiments:
        print(f"\n--- DP grouping comparison ({len(valid_models)} models) ---")
        plot_multi_expert_dp_grouping(
            valid_models, tq_layers_dict, gptq_layers_dict, args.bit, targets_dict,
            args.slice_expert_num, args.out_dir, args.pdf
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
