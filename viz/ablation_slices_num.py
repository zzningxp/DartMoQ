"""BPW 和 Slice 数消融实验可视化"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Optional
from dataclasses import dataclass

BIT_COLORS = [
    (0.95, 0.95, 0.95),
    (0.84, 0.96, 0.69),
    (0.62, 0.89, 0.81),
    (1.00, 0.75, 0.00),
    (0.97, 0.55, 0.49),
]


@dataclass
class AblationDataPoint:
    model: str
    num_slices: int
    bpw: float
    wiki_ppl: Optional[float]
    c4_ppl: Optional[float]


@dataclass
class AblationResult:
    models: list[str]
    num_slices_list: list[int]
    bpw_list: list[float]
    data: list[AblationDataPoint]


class AblationVisualizer:
    def __init__(self):
        self.model_display_names = {
            "dsv1": "DeepSeek-V1",
            "dsv2": "DeepSeek-V2",
            "moon": "Moon",
            "olmoe": "OLMoE",
            "qwen": "Qwen"
        }

    def load_data_from_csv(self, csv_path: str) -> AblationResult:
        df = pd.read_csv(csv_path)
        data_points = []

        for _, row in df.iterrows():
            wiki_ppl = float(row['wiki_ppl']) if pd.notna(row['wiki_ppl']) else None
            c4_ppl = float(row['c4_ppl']) if pd.notna(row['c4_ppl']) else None

            data_points.append(AblationDataPoint(
                model=str(row['model']),
                num_slices=int(row['num_slices']),
                bpw=float(row['bpw']),
                wiki_ppl=wiki_ppl,
                c4_ppl=c4_ppl
            ))

        models = sorted(list(set(d.model for d in data_points)))
        num_slices_list = sorted(list(set(d.num_slices for d in data_points)))
        bpw_list = sorted(list(set(d.bpw for d in data_points)))

        return AblationResult(
            models=models,
            num_slices_list=num_slices_list,
            bpw_list=bpw_list,
            data=data_points
        )

    def plot_single_figure(self, result: AblationResult, save_dir: Optional[str] = None):
        """单个综合图，横轴按模型分组，模型内按slice和bpw排列"""
        import matplotlib as mpl
        mpl.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
        })

        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plot", "ablation_bpw_slices")
        os.makedirs(save_dir, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(14, 4.5))

        n_models = len(result.models)
        n_slices = len(result.num_slices_list)
        n_bpw = len(result.bpw_list)

        model_spacing = 1.5
        slice_spacing = 0.15
        bar_width = 0.56

        # Map BPW values to BIT_COLORS: 0.5->1, 1.0->2, 1.5->3, 2.0->4
        bpw_to_color_idx = {0.5: 1, 1.0: 2, 1.5: 3, 2.0: 4}
        bpw_colors = [BIT_COLORS[bpw_to_color_idx.get(bpw, 2)] for bpw in result.bpw_list]

        x_tick_positions = []
        group_ends = []  # 记录每个模型组的结束位置
        current_x = 0

        position_map = {}

        for model in result.models:
            model_start = current_x
            for num_slices in result.num_slices_list:
                for bpw in result.bpw_list:
                    position_map[(model, num_slices, bpw)] = current_x
                    current_x += bar_width
                current_x += slice_spacing
            group_end = current_x - slice_spacing
            group_ends.append(group_end)
            model_center = (model_start + group_end) / 2
            x_tick_positions.append(model_center)
            current_x += model_spacing

        # 设置y轴显示上限
        display_limit = 160

        # 绘制 C4 PPL
        for model_idx, model in enumerate(result.models):
            model_data = [d for d in result.data if d.model == model]

            for slice_idx, num_slices in enumerate(result.num_slices_list):
                for bpw_idx, bpw in enumerate(result.bpw_list):
                    dp = next((d for d in model_data if d.num_slices == num_slices and d.bpw == bpw), None)
                    if dp and dp.c4_ppl is not None:
                        pos = position_map.get((model, num_slices, bpw))
                        if pos is not None:
                            # 超过显示上限的截断到上限
                            display_val = min(dp.c4_ppl, display_limit * 0.98)
                            ax.bar(pos, display_val, bar_width,
                                   color=bpw_colors[bpw_idx],
                                   edgecolor='white', linewidth=0.5)

                            # 如果超过显示上限，在柱子顶部标注实际值
                            if dp.c4_ppl > display_limit:
                                ax.text(pos, display_limit * 0.8,
                                       f"{dp.c4_ppl:.0f}", ha='center', va='top', fontsize=8,
                                       bbox=dict(boxstyle='square,pad=0.1', facecolor='white', alpha=0.9, edgecolor=bpw_colors[bpw_idx], linewidth=0.5))

        # 设置横轴标签 - 只显示模型名
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels([self.model_display_names.get(m, m) for m in result.models],
                           fontsize=11, fontweight='bold')

        # 画分隔线 - 在模型组之间
        for i in range(len(group_ends) - 1):
            sep = (group_ends[i] + group_ends[i] + model_spacing) / 2
            ax.axvline(x=sep, color='gray', linestyle='--', alpha=0.5)

        # 添加 slice 分组标记 - 放到柱子上方
        current_x = 0
        for model in result.models:
            for num_slices in result.num_slices_list:
                slice_center = current_x + (n_bpw * bar_width) / 2 - bar_width / 2
                ax.text(slice_center, display_limit * 0.96,
                        f"s{num_slices}", ha='center', va='top', fontsize=9,
                        bbox=dict(boxstyle='square,pad=0.1', facecolor='white', alpha=0.8, edgecolor='none'))
                current_x += n_bpw * bar_width + slice_spacing
            current_x += model_spacing

        ax.set_ylabel('C4 Perplexity (PPL)', fontsize=12)
        ax.set_title('C4 PPL', fontsize=13, fontweight='bold')

        # 创建图例 - 只放BPW，放到图内
        from matplotlib.patches import Patch
        legend_elements = []

        # BPW 颜色
        for bpw_idx, bpw in enumerate(result.bpw_list):
            legend_elements.append(Patch(color=bpw_colors[bpw_idx],
                                         label=f'BPW={bpw}'))

        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, bbox_to_anchor=(1, 0.85))

        # 设置y轴上限
        ax.set_ylim(0, display_limit)

        plt.tight_layout(rect=[0, 0.02, 1, 0.98])

        # 保存PNG
        save_path_png = os.path.join(save_dir, 'ablation_slices_num_c4_ppl.png')
        plt.savefig(save_path_png)
        print(f"PNG已保存到: {save_path_png}")

        # 保存PDF
        save_path_pdf = os.path.join(save_dir, 'ablation_slices_num_c4_ppl.pdf')
        plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
        print(f"PDF已保存到: {save_path_pdf}")
        plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="BPW 和 Slice 数消融实验可视化")
    parser.add_argument("--data_csv", type=str, default="logs/ablation_data.csv", help="数据 CSV 路径")
    parser.add_argument("--save_dir", type=str, default=None, help="保存目录")

    args = parser.parse_args()

    visualizer = AblationVisualizer()
    result = visualizer.load_data_from_csv(args.data_csv)
    visualizer.plot_single_figure(result, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
