"""Global 消融实验可视化"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

GLOBAL_COLOR = "#e57373"  # 红色
NO_GLOBAL_COLOR = "#64b5f6"  # 蓝色

MODEL_SHORT_NAMES = {
    "deepseek-v1-moe-16b": "dsv1",
    "deepseek-v2-lite": "dsv2",
    "moonlight": "moon",
    "olmoe-7b-1b": "olmoe",
    "qwen3-30b-a3b": "qwen"
}

MODEL_DISPLAY_NAMES = {
    "dsv1": "DeepSeek-V1",
    "dsv2": "DeepSeek-V2",
    "moon": "Moonlight",
    "olmoe": "OLMoE",
    "qwen": "Qwen"
}


class GlobalAblationVisualizer:
    def __init__(self):
        pass

    def load_comparison_data(self, csv_path: str) -> pd.DataFrame:
        """从对比CSV加载数据"""
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

    def plot_global_vs_nonglobal_c4(self, df_comparison: pd.DataFrame, save_dir: Optional[str] = None):
        """绘制Global vs Non-Global的对比柱状图（只C4，分TurboQuant和GPTQ）"""
        self._setup_plot_style()

        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plot", "global_ablation")
        os.makedirs(save_dir, exist_ok=True)

        df_dp0 = df_comparison[df_comparison['config_global'] == 'dp+0cps'].copy()

        models = sorted(list(set(df_dp0['model_short'])))
        bpw_list = sorted(list(set(df_dp0['bpw'])))
        quantmodes = ['turboquant', 'gptq']

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

        bar_width = 0.35
        model_spacing = 1.2

        for ax_idx, quantmode in enumerate(quantmodes):
            ax = axes[ax_idx]

            df_qm = df_dp0[df_dp0['quantmode'] == quantmode]

            # 设置y轴显示上限
            display_limit = 100 if quantmode == 'turboquant' else 200

            current_x = 0

            for model in models:
                model_data = df_qm[df_qm['model_short'] == model]

                for bpw in bpw_list:
                    bpw_data = model_data[model_data['bpw'] == bpw]

                    if len(bpw_data) >= 2:
                        # 新格式：需要区分global和non-global行
                        # global行的quant_scheme_global带'global-'前缀
                        global_rows = bpw_data[bpw_data['quant_scheme_global'].str.startswith('global-', na=False)]
                        # non-global行的quant_scheme_global不带'global-'前缀
                        non_global_rows = bpw_data[~bpw_data['quant_scheme_global'].str.startswith('global-', na=False)]

                        if len(global_rows) > 0 and len(non_global_rows) > 0:
                            avg_non = non_global_rows['ppl_c4'].mean()
                            avg_global = global_rows['ppl_c4'].mean()

                            # 绘制柱子，超过显示上限的截断到上限
                            display_non = min(avg_non, display_limit * 0.98)
                            display_global = min(avg_global, display_limit * 0.98)

                            ax.bar(current_x - bar_width/2, display_non, bar_width,
                                   color=NO_GLOBAL_COLOR, label='w/o Global' if current_x == 0 else "")
                            ax.bar(current_x + bar_width/2, display_global, bar_width,
                                   color=GLOBAL_COLOR, label='Global' if current_x == 0 else "")

                            # 如果超过显示上限，在柱子顶部标注实际值 - 往左偏移
                            if avg_non > display_limit:
                                ax.text(current_x - bar_width/2 - 0.5, display_limit * 0.85,
                                       f"{avg_non:.2f}", ha='right', va='top', fontsize=8,
                                       bbox=dict(boxstyle='square,pad=0.1', facecolor='white', alpha=0.9, edgecolor=NO_GLOBAL_COLOR, linewidth=0.5))
                            if avg_global > display_limit:
                                ax.text(current_x + bar_width/2 + 0.5, display_limit * 0.8,
                                       f"{avg_global:.2f}", ha='left', va='top', fontsize=8,
                                       bbox=dict(boxstyle='square,pad=0.1', facecolor='white', alpha=0.9, edgecolor=GLOBAL_COLOR, linewidth=0.5))

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
                for bpw in bpw_list:
                    ax.text(current_x, display_limit * 0.95,
                           f"{bpw}", ha='center', va='top', fontsize=9,
                           bbox=dict(boxstyle='square,pad=0.1', facecolor='white', alpha=0.8, edgecolor='none'))
                    current_x += 1
                current_x += model_spacing

            current_x = len(bpw_list) - 0.5
            for _ in range(len(models) - 1):
                ax.axvline(x=current_x + model_spacing/2, color='gray', linestyle='--', alpha=0.5)
                current_x += len(bpw_list) + model_spacing

            ax.set_ylabel('C4 Perplexity (PPL)')
            qm_title = "TurboQuant" if quantmode == "turboquant" else "GPTQ"
            ax.set_title(f'{qm_title} (Global + dp+0cps)', fontweight='bold')

            ax.legend(loc='upper right', bbox_to_anchor=(0.21, 0.60))

            ax.set_ylim(0, display_limit)

        plt.tight_layout()

        save_path_png = os.path.join(save_dir, 'ablation_global_vs_nonglobal_c4_turboquant_gptq.png')
        plt.savefig(save_path_png)
        print(f"PNG已保存到: {save_path_png}")

        save_path_pdf = os.path.join(save_dir, 'ablation_global_vs_nonglobal_c4_turboquant_gptq.pdf')
        plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
        print(f"PDF已保存到: {save_path_pdf}")

        plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Global 消融实验可视化")
    parser.add_argument("--comparison_csv", type=str,
                        default="logs/global_vs_nonglobal_comparison.csv",
                        help="对比数据 CSV 路径")
    parser.add_argument("--save_dir", type=str, default=None, help="保存目录")

    args = parser.parse_args()

    visualizer = GlobalAblationVisualizer()

    print("加载对比数据...")
    df_comparison = visualizer.load_comparison_data(args.comparison_csv)

    print("绘制 Global vs Non-Global C4 对比图...")
    visualizer.plot_global_vs_nonglobal_c4(df_comparison, save_dir=args.save_dir)

    print("完成！")


if __name__ == "__main__":
    main()
