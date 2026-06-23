"""0bit_prune vs disable_0bit_prune 消融实验可视化"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

# 颜色配置
_0BIT_PRUNE_COLOR = "#e57373"  # 红色 - 0bit_prune
DISABLE_0BIT_PRUNE_COLOR = "#64b5f6"  # 蓝色 - disable_0bit_prune

MODEL_DISPLAY_NAMES = {
    "dsv1": "DeepSeek-V1",
    "dsv2": "DeepSeek-V2",
    "moon": "Moonlight",
    "olmoe": "OLMoE"
}


class ZeroBitPruneAblationVisualizer:
    def __init__(self):
        pass

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """从CSV加载数据"""
        df = pd.read_csv(csv_path, sep='\t')
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

    def _extract_bpw(self, quant_scheme: str) -> str:
        if 'a8s8m1.5' in quant_scheme:
            return 'a8s8m1.5'
        elif 'a8s8m2' in quant_scheme:
            return 'a8s8m2'
        return 'unknown'

    def plot_0bit_prune_comparison(self, df: pd.DataFrame, save_dir: Optional[str] = None):
        """绘制0bit_prune vs disable_0bit_prune的对比柱状图"""
        self._setup_plot_style()

        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plot", "ablation_0bit_prune")
        os.makedirs(save_dir, exist_ok=True)

        # 提取bpw信息
        df['bpw'] = df['quant_scheme'].apply(self._extract_bpw)

        models = sorted(list(set(df['model'])))
        bpw_list = sorted(list(set(df['bpw'])))
        quantmodes = ['turboquant', 'gptq']

        # 绘制两个子图：只显示 C4
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

        bar_width = 0.35
        model_spacing = 1.2

        dataset = 'c4'
        for col_idx, quantmode in enumerate(quantmodes):
            ax = axes[col_idx]

            df_qm = df[df['quantmode'] == quantmode]

            # 设置y轴显示上限
            display_limit = 40 if quantmode == 'turboquant' else 80

            current_x = 0

            for model in models:
                model_data = df_qm[df_qm['model'] == model]

                for bpw in bpw_list:
                    bpw_data = model_data[model_data['bpw'] == bpw]

                    if len(bpw_data) >= 2:
                        # 区分0bit_prune和disable_0bit_prune行
                        prune_rows = bpw_data[bpw_data['disable_0bit_prune'] == '0bit_prune']
                        disable_rows = bpw_data[bpw_data['disable_0bit_prune'] == 'disable_0bit_prune']

                        if len(prune_rows) > 0 and len(disable_rows) > 0:
                            ppl_col = f'ppl_{dataset}'
                            avg_prune = prune_rows[ppl_col].mean()
                            avg_disable = disable_rows[ppl_col].mean()

                            # 绘制柱子，超过显示上限的截断到上限
                            display_prune = min(avg_prune, display_limit * 0.98)
                            display_disable = min(avg_disable, display_limit * 0.98)

                            ax.bar(current_x - bar_width/2, display_prune, bar_width,
                                   color=_0BIT_PRUNE_COLOR, label='0bit_prune' if current_x == 0 else "")
                            ax.bar(current_x + bar_width/2, display_disable, bar_width,
                                   color=DISABLE_0BIT_PRUNE_COLOR, label='disable_0bit_prune' if current_x == 0 else "")

                            # 如果超过显示上限，在柱子顶部标注实际值
                            if avg_prune > display_limit:
                                ax.text(current_x - bar_width/2 - 0.5, display_limit * 0.85,
                                       f"{avg_prune:.2f}", ha='right', va='top', fontsize=8,
                                       bbox=dict(boxstyle='square,pad=0.1', facecolor='white', alpha=0.9, edgecolor=_0BIT_PRUNE_COLOR, linewidth=0.5))
                            if avg_disable > display_limit:
                                ax.text(current_x + bar_width/2 + 0.5, display_limit * 0.8,
                                       f"{avg_disable:.2f}", ha='left', va='top', fontsize=8,
                                       bbox=dict(boxstyle='square,pad=0.1', facecolor='white', alpha=0.9, edgecolor=DISABLE_0BIT_PRUNE_COLOR, linewidth=0.5))

                    current_x += 1
                current_x += model_spacing

            # 设置x轴标签
            xticks = []
            xticklabels = []
            current_x = 0
            for model in models:
                model_center = current_x + len(bpw_list)/2 - 0.5
                xticks.append(model_center)
                xticklabels.append(MODEL_DISPLAY_NAMES.get(model, model))
                current_x += len(bpw_list) + model_spacing

            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, fontweight='bold')

            current_x = 0
            for model in models:
                for i, bpw in enumerate(bpw_list):
                    ax.text(current_x, display_limit * (0.95 - i * 0.05),
                           f"{bpw}", ha='center', va='top', fontsize=8,
                           bbox=dict(boxstyle='square,pad=0.1', facecolor='white', alpha=0.8, edgecolor='none'))
                    current_x += 1
                current_x += model_spacing

            current_x = len(bpw_list) - 0.5
            for _ in range(len(models) - 1):
                ax.axvline(x=current_x + model_spacing/2, color='gray', linestyle='--', alpha=0.5)
                current_x += len(bpw_list) + model_spacing

            ax.set_ylabel('C4 Perplexity (PPL)')
            qm_title = "TurboQuant" if quantmode == "turboquant" else "GPTQ"
            ax.set_title(f'{qm_title}', fontweight='bold')

            ax.legend(loc='upper right', bbox_to_anchor=(0.3, 0.60))

            ax.set_ylim(0, display_limit)

        plt.tight_layout()

        save_path_png = os.path.join(save_dir, 'ablation_0bit_prune_vs_disable_comparison.png')
        plt.savefig(save_path_png)
        print(f"PNG已保存到: {save_path_png}")

        save_path_pdf = os.path.join(save_dir, 'ablation_0bit_prune_vs_disable_comparison.pdf')
        plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
        print(f"PDF已保存到: {save_path_pdf}")

        plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="0bit_prune 消融实验可视化")
    parser.add_argument("--csv", type=str,
                        default="logs/ablation_disable_0bit_prune.csv",
                        help="数据 CSV 路径")
    parser.add_argument("--save_dir", type=str, default=None, help="保存目录")

    args = parser.parse_args()

    visualizer = ZeroBitPruneAblationVisualizer()

    print("加载数据...")
    df = visualizer.load_data(args.csv)

    print("绘制 0bit_prune vs disable_0bit_prune 对比图...")
    visualizer.plot_0bit_prune_comparison(df, save_dir=args.save_dir)

    print("完成！")


if __name__ == "__main__":
    main()
