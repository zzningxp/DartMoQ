"""Rank Mode 消融实验可视化"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

# Rank Mode 颜色
RANK_MODE_COLORS = {
    "turboquant_mse": "#e57373",        # 红色
    "turboquant_iipl": "#64b5f6",       # 蓝色
    "turboquant_innerproduct_fea": "#81c784",  # 绿色
    "tq_innerproduct": "#ffb74d"        # 橙色
}

# Rank Mode 显示名称
RANK_MODE_DISPLAY_NAMES = {
    "turboquant_mse": "MSE",
    "turboquant_iipl": "Input-Intermediate Product Loss",
    "turboquant_innerproduct_fea": "IP Full Expert Activation",
    "tq_innerproduct": "Inner Product (Recommend)"
}

MODEL_DISPLAY_NAMES = {
    "dsv1": "DeepSeek-V1",
    "dsv2": "DeepSeek-V2",
    "moon": "Moonlight",
    "olmoe": "OLMoE"
}


class RankModeAblationVisualizer:
    def __init__(self):
        pass

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """从CSV加载数据"""
        df = pd.read_csv(csv_path)
        return df

    def _setup_plot_style(self):
        """设置绘图样式"""
        import matplotlib as mpl
        mpl.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
        })

    def plot_rank_mode_comparison(self, df: pd.DataFrame, save_dir: Optional[str] = None):
        """绘制不同 Rank Mode 的对比柱状图"""
        self._setup_plot_style()

        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plot", "ablation_rank_mode")
        os.makedirs(save_dir, exist_ok=True)

        models = sorted(list(set(df['model'])))
        bpw_list = sorted(list(set(df['bpw'])))
        rank_modes = list(RANK_MODE_COLORS.keys())

        fig, ax = plt.subplots(1, 1, figsize=(14, 4.5))

        bar_width = 0.2
        model_spacing = 1.5
        bpw_spacing = 0.4
        rank_mode_spacing = 0.03

        # 设置y轴显示上限
        display_limit = 60

        current_x = 0
        group_ends = []

        for model in models:
            model_data = df[df['model'] == model]
            model_start = current_x

            for bpw in bpw_list:
                bpw_data = model_data[model_data['bpw'] == bpw]

                for rm_idx, rank_mode in enumerate(rank_modes):
                    rm_data = bpw_data[bpw_data['rank_mode'] == rank_mode]

                    if len(rm_data) > 0:
                        ppl_c4 = rm_data['ppl_c4'].iloc[0]

                        # 绘制柱子，超过显示上限的截断到上限
                        display_val = min(ppl_c4, display_limit * 0.98)

                        x_pos = current_x + rm_idx * (bar_width + rank_mode_spacing)

                        ax.bar(x_pos, display_val, bar_width,
                               color=RANK_MODE_COLORS[rank_mode],
                               label=RANK_MODE_DISPLAY_NAMES[rank_mode] if rm_idx == rm_idx else "")

                        # 如果超过显示上限，在柱子顶部标注实际值
                        if ppl_c4 > display_limit:
                            ax.text(x_pos, display_limit * 0.8,
                                   f"{ppl_c4:.0f}", ha='center', va='top', fontsize=7,
                                   bbox=dict(boxstyle='square,pad=0.1', facecolor='white', alpha=0.9,
                                             edgecolor=RANK_MODE_COLORS[rank_mode], linewidth=0.5))

                current_x += len(rank_modes) * (bar_width + rank_mode_spacing) + bpw_spacing

            group_end = current_x - bpw_spacing
            group_ends.append(group_end)
            current_x += model_spacing

        # 设置x轴标签 - 模型名
        xticks = []
        xticklabels = []
        current_x = 0
        for model_idx, model in enumerate(models):
            # 模型中心位置
            model_start = current_x
            for bpw in bpw_list:
                current_x += len(rank_modes) * (bar_width + rank_mode_spacing) + bpw_spacing
            model_end = current_x - bpw_spacing
            model_center = (model_start + model_end) / 2
            xticks.append(model_center)
            xticklabels.append(MODEL_DISPLAY_NAMES.get(model, model))
            current_x += model_spacing

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontweight='bold')

        # 添加 BPW 标签在每组柱子上方
        current_x = 0
        for model in models:
            for bpw in bpw_list:
                group_center = current_x + (len(rank_modes) * (bar_width + rank_mode_spacing)) / 2 - rank_mode_spacing / 2
                ax.text(group_center, display_limit * 0.96,
                       f"{bpw}", ha='center', va='top', fontsize=9,
                       bbox=dict(boxstyle='square,pad=0.1', facecolor='white', alpha=0.8, edgecolor='none'))
                current_x += len(rank_modes) * (bar_width + rank_mode_spacing) + bpw_spacing
            current_x += model_spacing

        # 画分隔线 - 在模型组之间
        for i in range(len(group_ends) - 1):
            sep = group_ends[i] + model_spacing / 2
            ax.axvline(x=sep, color='gray', linestyle='--', alpha=0.5)

        ax.set_ylabel('C4 Perplexity (PPL)')
        ax.set_title('Rank Mode Comparison (TurboQuant)', fontweight='bold')

        # 创建完整的legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(color=RANK_MODE_COLORS[rm], label=RANK_MODE_DISPLAY_NAMES[rm])
                          for rm in rank_modes]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.5, 0.85), ncol=2)

        ax.set_ylim(0, display_limit)

        plt.tight_layout()

        save_path_png = os.path.join(save_dir, 'ablation_rank_mode_c4.png')
        plt.savefig(save_path_png)
        print(f"PNG已保存到: {save_path_png}")

        save_path_pdf = os.path.join(save_dir, 'ablation_rank_mode_c4.pdf')
        plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
        print(f"PDF已保存到: {save_path_pdf}")

        plt.close()

    def plot_wiki_and_c4(self, df: pd.DataFrame, save_dir: Optional[str] = None):
        self._setup_plot_style()

        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plot", "ablation_rank_mode")
        os.makedirs(save_dir, exist_ok=True)

        models = sorted(list(set(df['model'])))
        bpw_list = sorted(list(set(df['bpw'])))
        rank_modes = list(RANK_MODE_COLORS.keys())

        fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))

        bar_width = 0.2
        model_spacing = 1.5
        bpw_spacing = 0.4
        rank_mode_spacing = 0.03

        # 设置y轴显示上限
        display_limit_wiki = 100
        display_limit_c4 = 100

        for ax_idx, (metric_name, display_limit, col_name) in enumerate([
            ("WikiText", display_limit_wiki, "ppl_wiki"),
            ("C4", display_limit_c4, "ppl_c4")
        ]):
            ax = axes[ax_idx]
            current_x = 0
            group_ends = []

            for model in models:
                model_data = df[df['model'] == model]

                for bpw in bpw_list:
                    bpw_data = model_data[model_data['bpw'] == bpw]

                    for rm_idx, rank_mode in enumerate(rank_modes):
                        rm_data = bpw_data[bpw_data['rank_mode'] == rank_mode]

                        if len(rm_data) > 0:
                            ppl_val = rm_data[col_name].iloc[0]

                            # 绘制柱子，超过显示上限的截断到上限
                            display_val = min(ppl_val, display_limit * 0.98)

                            x_pos = current_x + rm_idx * (bar_width + rank_mode_spacing)

                            ax.bar(x_pos, display_val, bar_width,
                                   color=RANK_MODE_COLORS[rank_mode])

                            # 如果超过显示上限，在柱子顶部标注实际值
                            if ppl_val > display_limit:
                                ax.text(x_pos, display_limit * 0.8,
                                       f"{ppl_val:.0f}", ha='center', va='top', fontsize=7,
                                       bbox=dict(boxstyle='square,pad=0.1', facecolor='white', alpha=0.9,
                                                 edgecolor=RANK_MODE_COLORS[rank_mode], linewidth=0.5))

                    current_x += len(rank_modes) * (bar_width + rank_mode_spacing) + bpw_spacing

                group_end = current_x - bpw_spacing
                group_ends.append(group_end)
                current_x += model_spacing

            # 设置x轴标签 - 模型名
            xticks = []
            xticklabels = []
            current_x = 0
            for model_idx, model in enumerate(models):
                model_start = current_x
                for bpw in bpw_list:
                    current_x += len(rank_modes) * (bar_width + rank_mode_spacing) + bpw_spacing
                model_end = current_x - bpw_spacing
                model_center = (model_start + model_end) / 2
                xticks.append(model_center)
                xticklabels.append(MODEL_DISPLAY_NAMES.get(model, model))
                current_x += model_spacing

            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, fontweight='bold')

            # 添加 BPW 标签在每组柱子上方
            current_x = 0
            for model in models:
                for bpw in bpw_list:
                    group_center = current_x + (len(rank_modes) * (bar_width + rank_mode_spacing)) / 2 - rank_mode_spacing / 2
                    ax.text(group_center, display_limit * 0.96,
                           f"{bpw}", ha='center', va='top', fontsize=9,
                           bbox=dict(boxstyle='square,pad=0.1', facecolor='white', alpha=0.8, edgecolor='none'))
                    current_x += len(rank_modes) * (bar_width + rank_mode_spacing) + bpw_spacing
                current_x += model_spacing

            # 画分隔线 - 在模型组之间
            for i in range(len(group_ends) - 1):
                sep = group_ends[i] + model_spacing / 2
                ax.axvline(x=sep, color='gray', linestyle='--', alpha=0.5)

            ax.set_ylabel(f'{metric_name} Perplexity (PPL)')
            ax.set_title(f'{metric_name} PPL', fontweight='bold')

            ax.set_ylim(0, display_limit)

        # 创建完整的legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(color=RANK_MODE_COLORS[rm], label=RANK_MODE_DISPLAY_NAMES[rm])
                          for rm in rank_modes]
        axes[0].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.28, 0.55), ncol=2)

        plt.tight_layout()

        save_path_png = os.path.join(save_dir, 'ablation_rank_mode_wiki_c4.png')
        plt.savefig(save_path_png)
        print(f"PNG已保存到: {save_path_png}")

        save_path_pdf = os.path.join(save_dir, 'ablation_rank_mode_wiki_c4.pdf')
        plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
        print(f"PDF已保存到: {save_path_pdf}")

        plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Rank Mode 消融实验可视化")
    parser.add_argument("--data_csv", type=str,
                        default="logs/ablation_rank_mode.csv",
                        help="数据 CSV 路径")
    parser.add_argument("--save_dir", type=str, default=None, help="保存目录")

    args = parser.parse_args()

    visualizer = RankModeAblationVisualizer()

    print("加载数据...")
    df = visualizer.load_data(args.data_csv)

    print("绘制 Rank Mode C4 对比图...")
    visualizer.plot_rank_mode_comparison(df, save_dir=args.save_dir)

    print("绘制 Rank Mode Wiki + C4 对比图...")
    visualizer.plot_wiki_and_c4(df, save_dir=args.save_dir)

    print("完成！")


if __name__ == "__main__":
    main()
