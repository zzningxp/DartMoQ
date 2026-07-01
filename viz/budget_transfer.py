
"""预算转移可视化.

This module visualizes how DartMoQ can prune low-sensitivity neurons to int0
and reallocate those bits to high-sensitivity neurons (int4) for better
accuracy under the same bit budget.

Core idea: Same BPW, Better Loss.

Usage:
    python -m viz.budget_transfer                    # All models
    python -m viz.budget_transfer --model dsv1      # Specific model
    python -m viz.budget_transfer --bit 2           # Base bit width
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import ScalarFormatter, LogLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

OUT_ROOT = "plot/budget_transfer"
DEFAULT_QUANTMODE = "turboquant"
DEFAULT_RANK_MODE = "turboquant_innerproduct"
DEFAULT_BIT = 2

BIT_COLORS = [
    (0.95, 0.95, 0.95),
    (0.84, 0.96, 0.69),
    (0.62, 0.89, 0.81),
    (1.00, 0.75, 0.00),
    (0.97, 0.55, 0.49),
]
BIT_CMAP = ListedColormap(BIT_COLORS, name="bitwidth")


def collect_sensitivity_data(layer, bits=(0, 1, 2, 3, 4), expert_idx=0):
    """Collect sensitivity data and extrapolate int0 if needed."""
    result = {}
    for b in bits:
        if b in layer.by_bit:
            result[b] = layer.by_bit[b][expert_idx].copy()

    if 0 not in result and 1 in result and 2 in result:
        result[0] = extrapolate_0bit_for_visualization(result)

    return result


def extrapolate_0bit_for_visualization(rates):
    """Extrapolate int0 loss using log-quadratic fit."""
    bits_with_data = sorted([b for b in rates.keys() if b > 0])
    if len(bits_with_data) < 2:
        return rates[bits_with_data[0]] * 2.0

    n_neurons = len(rates[bits_with_data[0]])
    l0 = np.zeros(n_neurons, dtype=float)

    for i in range(n_neurons):
        losses = np.array([rates[b][i] for b in bits_with_data])
        b_array = np.array(bits_with_data, dtype=float)

        try:
            log_loss = np.log(np.clip(losses, 1e-30, None))
            p, q, r = np.polyfit(b_array, log_loss, deg=2)
            l0_i = np.exp(r)
            l1 = losses[0]
            if l0_i < l1:
                l0_i = l1 * 2.0
            l0[i] = l0_i
        except Exception:
            l0[i] = rates[1][i] * 2.0

    return l0


def compute_uniform_loss(rates, target_bpw, bits=(0, 1, 2, 3, 4)):
    """Compute uniform allocation loss."""
    available_bits = sorted([b for b in bits if b in rates])
    closest_bit = min(available_bits, key=lambda b: abs(b - target_bpw))
    n_neurons = len(rates[closest_bit])
    allocation = np.full(n_neurons, closest_bit, dtype=int)
    total_loss = rates[closest_bit].sum()
    return total_loss, allocation


def budget_transfer_allocation(rates, target_bpw, low_bit=0, high_bit=4):
    """Allocate bits by transferring from low-sensitivity to high-sensitivity."""
    n_neurons = len(rates[1])
    sens = rates[2] if 2 in rates else rates[1]
    sorted_idx = np.argsort(-sens)

    allocation = np.full(n_neurons, 2, dtype=int)
    total_bits = 2 * n_neurons

    # 应用对称的比特重新分配
    if abs(target_bpw - 2.0) < 0.01:
        n_low = int(0.125 * n_neurons)
        n_high = n_low
        allocation[sorted_idx[-n_low:]] = low_bit
        allocation[sorted_idx[:n_high]] = high_bit
        total_bits = 2 * n_neurons  # 平均bpw仍然是2.0
    else:
        # 正常逻辑
        target_bits = target_bpw * n_neurons
        max_low = 0
        for i in range(n_neurons, -1, -1):
            test_bits = 2 * (n_neurons - i)
            if test_bits <= target_bits:
                max_low = i
                break

        if max_low > 0:
            allocation[sorted_idx[-max_low:]] = low_bit
            total_bits = 2 * (n_neurons - max_low)

        extra_bits = target_bits - total_bits
        n_upgrade = int(extra_bits / (high_bit - 2))
        n_upgrade = min(n_upgrade, n_neurons - max_low)

        if n_upgrade > 0:
            allocation[sorted_idx[:n_upgrade]] = high_bit
            total_bits += (high_bit - 2) * n_upgrade

    total_loss = 0.0
    for i in range(n_neurons):
        b = allocation[i]
        total_loss += rates[b][i]

    stats = {
        "low_count": np.sum(allocation == low_bit),
        "low_fraction": float(np.sum(allocation == low_bit)) / n_neurons,
        "high_count": np.sum(allocation == high_bit),
        "high_fraction": float(np.sum(allocation == high_bit)) / n_neurons,
        "total_bits": total_bits,
        "avg_bpw": float(total_bits) / n_neurons,
        "sorted_idx": sorted_idx,
        "low_bit": low_bit,
        "high_bit": high_bit,
    }

    return allocation, total_loss, stats


def plot_multi_expert_allocation(model_id, layer, expert_indices=None, target_bpw=2.0,
                                 low_bit=0, high_bit=4, out_dir=OUT_ROOT, save_pdf=False):
    """Plot bit allocation for multiple experts."""
    if expert_indices is None:
        n_experts = len(layer.by_bit.get(2, []))
        expert_indices = list(range(min(4, n_experts)))  # 默认最多展示4个专家

    n_plots = len(expert_indices)
    if n_plots == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 7))
        axes = [axes]
    elif n_plots == 2:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    elif n_plots <= 4:
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()[:n_plots]

    fig.suptitle(f"{model_label(model_id)} | Layer {layer.layer_idx} | budget Transfer Allocation ({high_bit}/2/{low_bit}, {target_bpw:.1f} BPW)",
                 fontsize=16, fontweight='bold', y=0.99)

    for ax, expert_idx in zip(axes, expert_indices):
        rates = collect_sensitivity_data(layer, bits=[0, 1, 2, 3, 4], expert_idx=expert_idx)
        uniform_loss, uniform_alloc = compute_uniform_loss(rates, target_bpw)
        recycl_alloc, recycl_loss, recycl_stats = budget_transfer_allocation(
            rates, target_bpw, low_bit=low_bit, high_bit=high_bit
        )

        sens = rates[2] if 2 in rates else rates[1]

        plot_single_expert(ax, recycl_alloc, rates, sens, "Expert %d" % expert_idx,
                          recycl_stats, uniform_loss, recycl_loss)

    # 只在最后一个图显示colorbar
    if len(axes) > 0:
        cbar_ax = fig.add_axes([0.93, 0.12, 0.02, 0.76])
        import matplotlib as mpl
        norm = mpl.colors.Normalize(vmin=0, vmax=4)
        sm = plt.cm.ScalarMappable(cmap=BIT_CMAP, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, ticks=[low_bit, 2, high_bit])
        cbar.set_label('Bit Width', fontsize=12)

    plt.subplots_adjust(right=0.9)

    os.makedirs(out_dir, exist_ok=True)
    base_path = os.path.join(out_dir, "multi_expert_allocation_%s_L%d_%d-2-%d" %
                            (model_id, layer.layer_idx, high_bit, low_bit))
    fig.savefig(base_path + ".png", dpi=200, bbox_inches='tight')
    if save_pdf:
        fig.savefig(base_path + ".pdf", bbox_inches='tight')
    plt.close(fig)

    png_path = base_path + ".png"
    print(f"[Multi-Expert {high_bit}/2/{low_bit}] Saved to {png_path}")
    if save_pdf:
        pdf_path = base_path + ".pdf"
        print(f"[Multi-Expert {high_bit}/2/{low_bit}] Saved to {pdf_path}")
    return png_path


def plot_single_expert(ax, allocation, rates, sens, title, stats, uniform_loss, recycl_loss):
    """Plot single expert bit allocation with before/after comparison."""
    sorted_idx = np.argsort(-sens)
    sorted_sens = sens[sorted_idx]
    sorted_alloc = allocation[sorted_idx]

    low_bit = stats.get('low_bit', 0)
    high_bit = stats.get('high_bit', 4)

    # 获取变化前（都是2bit）和变化后的loss
    loss_before = rates[2][sorted_idx]  # 变化前都是2bit
    loss_after = np.array([rates[b][i] for b, i in zip(sorted_alloc, sorted_idx)])

    n_sample = min(1500, len(sorted_sens))
    sample_step = max(1, len(sorted_sens) // n_sample)
    sample_x = np.arange(0, len(sorted_sens), sample_step)
    sample_sens = sorted_sens[::sample_step]
    sample_alloc = sorted_alloc[::sample_step]
    sample_loss_before = loss_before[::sample_step]
    sample_loss_after = loss_after[::sample_step]

    # 画保持2bit不变的神经元（只画一个点）
    mask_keep2 = sample_alloc == 2
    if np.any(mask_keep2):
        ax.scatter(sample_x[mask_keep2], sample_loss_before[mask_keep2],
                  c=BIT_COLORS[2], s=12, marker='o', alpha=0.7, label='Keep (2-bit)')

    # 画从2bit→high_bit的神经元（画两个点）
    mask_upgrade = sample_alloc == high_bit
    if np.any(mask_upgrade):
        # 变化前的点
        ax.scatter(sample_x[mask_upgrade] - 2.5, sample_loss_before[mask_upgrade],
                  c=BIT_COLORS[2], s=12, marker='s', alpha=0.6, label='Before (2-bit)')
        # 变化后的点
        ax.scatter(sample_x[mask_upgrade] + 2.5, sample_loss_after[mask_upgrade],
                  c=BIT_COLORS[high_bit], s=12, marker='o', alpha=0.85, label=f'After ({high_bit}-bit)')

    # 画从2bit→low_bit的神经元（画两个点）
    mask_downgrade = sample_alloc == low_bit
    if np.any(mask_downgrade):
        # 变化前的点
        ax.scatter(sample_x[mask_downgrade] - 2.5, sample_loss_before[mask_downgrade],
                  c=BIT_COLORS[2], s=12, marker='s', alpha=0.6)
        # 变化后的点（low_bit=0加黑色边缘）
        edgecolor = 'black' if low_bit == 0 else None
        linewidth = 0.3 if low_bit == 0 else 0
        ax.scatter(sample_x[mask_downgrade] + 2.5, sample_loss_after[mask_downgrade],
                  c=BIT_COLORS[low_bit], s=12, marker='o', alpha=0.9,
                  edgecolors=edgecolor, linewidths=linewidth, label=f'After ({low_bit}-bit)')

    # 添加前后八分之一的分割线
    n_total = len(sorted_sens)
    cutoff1 = n_total * 0.125
    cutoff2 = n_total * (1 - 0.125)
    ax.axvline(x=cutoff1, color='lightgray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=cutoff2, color='lightgray', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_xlabel('Neurons (sorted by sensitivity)', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    # 只显示主要刻度：0.001, 0.01, 0.1, 1, ...
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    # 隐藏次刻度
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=[]))
    # 让刻度显示原始数值
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best', fontsize=9)

    improvement = (1 - recycl_loss / uniform_loss) * 100 if uniform_loss > 0 else 0
    stats_text = f'Int{low_bit}: {stats["low_fraction"]*100:.1f}%\nInt{high_bit}: {stats["high_fraction"]*100:.1f}%\nLoss Reduction: {improvement:.1f}%'

    ax.text(0.98, 0.98, stats_text, ha='right', va='top', fontsize=10,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='#4EA72E'))


def run_all_visualizations(model_id, layers, target_bit=DEFAULT_BIT, target_bpw=2.0,
                          expert_idx=0, out_dir=OUT_ROOT, save_pdf=False):
    """Run all visualizations for a model."""
    if not layers:
        print("No layers for %s" % model_id)
        return

    print("\n" + "="*70)
    print("Visualizing 预算转移: %s" % model_label(model_id))
    print("="*70)

    for layer in layers:
        print(f"\n  Processing Layer {layer.layer_idx}")
        try:
            plot_multi_expert_allocation(model_id, layer, target_bpw=target_bpw,
                                         low_bit=0, high_bit=4, out_dir=out_dir, save_pdf=save_pdf)
        except Exception as e:
            print(f"  Multi-Expert Allocation (4/2/0) failed for Layer {layer.layer_idx}: %s" % e)

        try:
            plot_multi_expert_allocation(model_id, layer, target_bpw=target_bpw,
                                         low_bit=1, high_bit=3, out_dir=out_dir, save_pdf=save_pdf)
        except Exception as e:
            print(f"  Multi-Expert Allocation (3/2/1) failed for Layer {layer.layer_idx}: %s" % e)


def main():
    apply_paper_style()

    parser = argparse.ArgumentParser(description="预算转移可视化")
    parser.add_argument("--model", default=None, help="Model ID or path")
    parser.add_argument("--bit", type=int, default=DEFAULT_BIT, help="Base bit width")
    parser.add_argument("--bpw", type=float, default=None, help="Target bits per weight")
    parser.add_argument("--quantmode", default=DEFAULT_QUANTMODE, choices=["gptq", "turboquant"])
    parser.add_argument("--rankmode", default=None)
    parser.add_argument("--layer-start", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--expert-idx", type=int, default=0)
    parser.add_argument("--out-dir", default=OUT_ROOT)
    parser.add_argument("--pdf", action="store_true", default=False)

    args = parser.parse_args()

    if args.rankmode is None:
        args.rankmode = "turboquant_innerproduct" if args.quantmode == "turboquant" else "gptq_quant_outlier"

    target_bpw = args.bpw if args.bpw is not None else float(args.bit)

    if args.model:
        models = [resolve_model_id(args.model)]
    else:
        models = discover_models(args.quantmode, args.rankmode)
        if not models:
            print("No cached models found for %s/%s" % (args.quantmode, args.rankmode))
            return

    print("Found %d models: %s" % (len(models), models))

    for model_id in models:
        layers = load_all_layers(model_id, args.quantmode, args.rankmode,
                                bits=(1, 2, 3, 4), layer_start=args.layer_start,
                                num_layers=args.num_layers)

        run_all_visualizations(model_id, layers, target_bit=args.bit,
                              target_bpw=target_bpw, expert_idx=args.expert_idx,
                              out_dir=args.out_dir, save_pdf=args.pdf)

    print("\nDone! All visualizations saved to %s" % args.out_dir)


if __name__ == "__main__":
    main()

