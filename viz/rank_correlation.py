"""
Rank correlation visualization utilities.

This module contains functions for visualizing rank correlations between
different bit widths.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path just in case
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INTERMEDIATE_RESULT_DIR = "intermediate_result"


def plot_diff_wbits_correlation(model_type, layer_idx, expert_num, rates_2, rates_3, rates_4):
    for x in range(expert_num):
        rates_2_x = rates_2[x].cpu().float().numpy()
        rates_3_x = rates_3[x].cpu().float().numpy()
        rates_4_x = rates_4[x].cpu().float().numpy()

        n_neurons = len(rates_2_x)
        bins = len(rates_3_x) // n_neurons
        # print(all_rates_x, rates_3_x)
        rank_x_2 = np.argsort(np.argsort(rates_2_x)) // bins
        rank_x_3 = np.argsort(np.argsort(rates_3_x)) // bins
        rank_x_4 = np.argsort(np.argsort(rates_4_x)) // bins
        # print(rank_x_2[:40], rank_x_3[:40], rank_x_4[:40],)

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        ax1 = axes[0]
        ax1.scatter(rank_x_2, rank_x_3, s=5, alpha=0.5)
        ax1.plot([1, n_neurons], [1, n_neurons], 'r--', linewidth=1)
        ax1.set_xlabel('2bit Rank (1=most important)')
        ax1.set_ylabel('3bit Rank (1=most important)')
        ax1.set_title('2bit vs 3bit Neuron Rank', fontsize=12, fontweight='bold')
        ax1.set_xlim(1, n_neurons)
        ax1.set_ylim(1, n_neurons)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.scatter(rank_x_2, rank_x_4, s=5, alpha=0.5)
        ax2.plot([1, n_neurons], [1, n_neurons], 'r--', linewidth=1)
        ax2.set_xlabel('2bit Rank (1=most important)')
        ax2.set_ylabel('4bit Rank (1=most important)')
        ax2.set_title('2bit vs 4bit Neuron Rank', fontsize=12, fontweight='bold')
        ax2.set_xlim(1, n_neurons)
        ax2.set_ylim(1, n_neurons)
        ax2.grid(True, alpha=0.3)

        ax3 = axes[2]
        ax3.scatter(rank_x_3, rank_x_4, s=5, alpha=0.5)
        ax3.plot([1, n_neurons], [1, n_neurons], 'r--', linewidth=1)
        ax3.set_xlabel('3bit Rank (1=most important)')
        ax3.set_ylabel('4bit Rank (1=most important)')
        ax3.set_title('3bit vs 4bit Neuron Rank', fontsize=12, fontweight='bold')
        ax3.set_xlim(1, n_neurons)
        ax3.set_ylim(1, n_neurons)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"plot/_rank_compare_{model_type}_{layer_idx}_{x}.png")
        plt.close()


def plot_spearman_rank_correlation(model_type, layer_idx, expert_num, rates_2, rates_3, rates_4):
    from scipy.stats import spearmanr
    import seaborn as sns

    corr_matrix_list = []
    for x in range(expert_num):
        rates_2_x = rates_2[x].cpu().float().numpy()
        rates_3_x = rates_3[x].cpu().float().numpy()
        rates_4_x = rates_4[x].cpu().float().numpy()

        ranks = {
            "2-bit": np.argsort(np.argsort(-rates_2_x)) + 1,
            "3-bit": np.argsort(np.argsort(-rates_3_x)) + 1,
            "4-bit": np.argsort(np.argsort(-rates_4_x)) + 1
        }

        corr_matrix = np.zeros((3, 3))
        methods = list(ranks.keys())
        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                corr_matrix[i, j], _ = spearmanr(ranks[m1], ranks[m2])
        corr_matrix_list.append(corr_matrix)

    cols = min(expert_num, 8)
    rows = (expert_num + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_2d(axes)

    for x, corr_matrix in enumerate(corr_matrix_list):
        row = x // cols
        col = x % cols
        ax = axes[row, col]

        sns.heatmap(corr_matrix,
                    annot=True,
                    fmt=".3f",
                    xticklabels=methods,
                    yticklabels=methods,
                    cmap="coolwarm",
                    vmin=0.8, vmax=1.0,
                    ax=ax)
        ax.set_title(f'Spearman Rank Correlation, Expert {x}', fontsize=8)

    for x in range(expert_num, rows * cols):
        row = x // cols
        col = x % cols
        axes[row, col].axis('off')

    fig.suptitle(f'Spearman Rank Correlation (Layer {layer_idx})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"plot/_spearman_rank_compare_{model_type}_{layer_idx}.png")
    plt.close()


def main():
    """Command-line interface for rank correlation visualizations."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Rank correlation visualization tools"
    )
    subparsers = parser.add_subparsers(title="Commands", dest="command")

    # rank_compare command
    parser_rank = subparsers.add_parser(
        "rank_compare", help="Plot rank correlation between bit widths"
    )
    parser_rank.add_argument("--model", required=True,
                            help="Model identifier")
    parser_rank.add_argument("--layer", type=int, required=True,
                            help="Layer index")
    parser_rank.add_argument("--expert-num", type=int, required=True,
                            help="Number of experts")
    parser_rank.add_argument("--quant", default="turboquant",
                            help="Quantization type")
    parser_rank.add_argument("--rank-mode", default="turboquant_innerproduct",
                            help="Rank mode (turboquant_innerproduct or gptq_quant_outlier)")

    # spearman command
    parser_spearman = subparsers.add_parser(
        "spearman", help="Plot Spearman rank correlation heatmaps"
    )
    parser_spearman.add_argument("--model", required=True,
                                help="Model identifier")
    parser_spearman.add_argument("--layer", type=int, required=True,
                                help="Layer index")
    parser_spearman.add_argument("--expert-num", type=int, required=True,
                                help="Number of experts")
    parser_spearman.add_argument("--quant", default="turboquant",
                                help="Quantization type")
    parser_spearman.add_argument("--rank-mode", default="turboquant_innerproduct",
                                help="Rank mode (turboquant_innerproduct or gptq_quant_outlier)")

    args = parser.parse_args()

    # Load cached data
    if args.command in ["rank_compare", "spearman"]:
        import torch
        rates_2, rates_3, rates_4 = None, None, None

        cache_dir = os.path.join(INTERMEDIATE_RESULT_DIR, f"quant_outlier_{args.quant}", args.rank_mode, args.model)

        for bit, rates_var in [(2, 'rates_2'), (3, 'rates_3'), (4, 'rates_4')]:
            cache_path = os.path.join(cache_dir, f"{args.model}_L{args.layer}_b{bit}.pt")
            try:
                data = torch.load(cache_path, map_location='cpu')
                locals()[rates_var] = data
                print(f"Loaded {bit}bit data from {cache_path}")
            except Exception as e:
                print(f"Failed to load {bit}bit data: {e}")

        if rates_2 is None or rates_3 is None or rates_4 is None:
            print("Error: Could not load all required bit data (2, 3, 4 bits)")
            return

        if args.command == "rank_compare":
            plot_diff_wbits_correlation(args.model, args.layer, args.expert_num,
                                        rates_2, rates_3, rates_4)
        else:
            plot_spearman_rank_correlation(args.model, args.layer, args.expert_num,
                                        rates_2, rates_3, rates_4)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
