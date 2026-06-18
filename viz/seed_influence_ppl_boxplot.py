"""Quantization PPL Boxplot Visualization.

This script reads perplexity data for different quantization methods and seeds,
then creates boxplots to show mean and variance.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_ROOT = "plot/quant_ppl_boxplot"
DEFAULT_DATA_PATH = "logs/quant_ppl_data.csv"


def apply_paper_style() -> None:
    """Tighter, paper-friendly matplotlib defaults."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 130,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_data(data_path: str) -> pd.DataFrame:
    """Load CSV data."""
    df = pd.read_csv(data_path)
    print(f"Loaded data from {data_path}")
    print(f"  Number of rows: {len(df)}")
    print(f"  Quant methods: {sorted(df['quant_method'].unique())}")
    return df


def plot_ppl_boxplots(
    df: pd.DataFrame,
    out_dir: str,
    save_pdf: bool = False,
):
    """Create boxplots for wiki_ppl and c4_ppl."""
    apply_paper_style()

    # Get unique quant methods
    quant_methods = sorted(df['quant_method'].unique())
    n_methods = len(quant_methods)

    # Define colors for each method
    method_colors = {
        "a8s8m2": "#3cb44b",
        "a8s8m22": "#4363d8",
        "a8s8m22222222": "#911eb4",
        "a8s8m32222222": "#f58231",
        "global-bpw-a8s8m2": "#e6194b",
    }

    # Create figure with two subplots (wiki and c4)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

    # Prepare data for boxplots
    wiki_data = []
    c4_data = []
    for method in quant_methods:
        method_df = df[df['quant_method'] == method]
        wiki_data.append(method_df['wiki_ppl'].values)
        c4_data.append(method_df['c4_ppl'].values)

    # Position settings
    positions = np.arange(n_methods)
    box_width = 0.6

    # Plot Wiki PPL
    bp1 = ax1.boxplot(
        wiki_data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=4, alpha=0.6)
    )

    # Set colors for wiki boxplot
    for patch, method in zip(bp1['boxes'], quant_methods):
        patch.set_facecolor(method_colors.get(method, "#808080"))
        patch.set_alpha(0.7)

    ax1.set_ylabel("WikiText PPL", fontsize=11)
    ax1.set_title("WikiText PPL by Quantization Method", fontsize=12)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(quant_methods, rotation=30, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot C4 PPL
    bp2 = ax2.boxplot(
        c4_data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=4, alpha=0.6)
    )

    # Set colors for c4 boxplot
    for patch, method in zip(bp2['boxes'], quant_methods):
        patch.set_facecolor(method_colors.get(method, "#808080"))
        patch.set_alpha(0.7)

    ax2.set_ylabel("C4 PPL", fontsize=11)
    ax2.set_title("C4 PPL by Quantization Method", fontsize=12)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(quant_methods, rotation=30, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "quant_ppl_boxplot.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if save_pdf:
        pdf_path = os.path.join(out_dir, "quant_ppl_boxplot.pdf")
        fig.savefig(pdf_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {out_path}")


def print_statistics(df: pd.DataFrame):
    """Print detailed statistics for each quant method."""
    print("\n" + "=" * 100)
    print("DETAILED STATISTICS")
    print("=" * 100)

    quant_methods = sorted(df['quant_method'].unique())

    print("\n--- Wiki PPL Statistics:")
    print(f"  {'Quant Method':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print(f"  {'-' * 75}")
    for method in quant_methods:
        method_df = df[df['quant_method'] == method]['wiki_ppl']
        print(f"  {method:<25} {method_df.mean():<10.4f} {method_df.std():<10.4f} "
              f"{method_df.min():<10.4f} {method_df.max():<10.4f}")

    print("\n--- C4 PPL Statistics:")
    print(f"  {'Quant Method':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print(f"  {'-' * 75}")
    for method in quant_methods:
        method_df = df[df['quant_method'] == method]['c4_ppl']
        print(f"  {method:<25} {method_df.mean():<10.4f} {method_df.std():<10.4f} "
              f"{method_df.min():<10.4f} {method_df.max():<10.4f}")


def main():
    parser = argparse.ArgumentParser(description="Quantization PPL Boxplot Visualization")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to CSV data file")
    parser.add_argument("--out-dir", default=OUT_ROOT, help="Output directory")
    parser.add_argument("--pdf", action="store_true", default=False, help="Also save PDF copies")
    args = parser.parse_args()

    apply_paper_style()

    # Load data
    df = load_data(args.data)

    # Print statistics
    print_statistics(df)

    # Create plots
    plot_ppl_boxplots(df, args.out_dir, args.pdf)

    print("\nDone!")


if __name__ == "__main__":
    main()
