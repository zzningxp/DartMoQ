"""微专家分布可视化量化

分析各层专家内重要微专家占比、敏感度分布曲线、跨层重要性偏移统计。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from tqdm import tqdm

from viz._cache_io import (
    load_all_layers, discover_layers, resolve_model_id, load_layer,
    apply_paper_style, model_label
)


@dataclass
class ExpertDistributionMetrics:
    """单个专家的分布度量"""
    layer_idx: int
    expert_idx: int
    # 重要微专家占比
    top10_frac: float  # Top 10% 神经元的敏感度占比
    top20_frac: float  # Top 20% 神经元的敏感度占比
    top50_frac: float  # Top 50% 神经元的敏感度占比
    # 敏感度分布
    gini_coefficient: float  # 基尼系数
    entropy: float  # 熵（归一化）
    skewness: float  # 偏度
    kurtosis: float  # 峰度
    # 敏感度曲线
    cumulative_energy: np.ndarray  # 累积敏感度曲线（保留变量名兼容性）


@dataclass
class LayerDistributionMetrics:
    """单层的分布度量"""
    layer_idx: int
    depth_category: str
    experts: List[ExpertDistributionMetrics]
    # 层级别汇总
    avg_top10_frac: float
    avg_top20_frac: float
    avg_top50_frac: float
    avg_gini: float
    avg_entropy: float
    # 跨专家重要性方差
    expert_importance_variance: float


@dataclass
class MicroExpertDistResult:
    """完整微专家分布结果"""
    model_id: str
    by_layer: List[LayerDistributionMetrics]
    summary_stats: Dict[str, Dict[str, float]]


class MicroExpertDistAnalyzer:
    """微专家分布分析器"""

    def __init__(
        self,
        model_id: str,
        quantmode: str = "turboquant",
        rank_mode: str = "turboquant_innerproduct",
        bits: List[int] = (1, 2, 3, 4),
        device: str = "cpu"
    ):
        self.model_id = resolve_model_id(model_id)
        self.quantmode = quantmode
        self.rank_mode = rank_mode
        self.bits = bits
        self.device = device

        # 按层位置分类
        self.depth_categories = {
            "shallow": lambda l, total: l < total // 3,
            "middle": lambda l, total: total // 3 <= l < 2 * total // 3,
            "deep": lambda l, total: l >= 2 * total // 3,
        }

    def get_depth_category(self, layer_idx: int, total_layers: int) -> str:
        """获取层的深度分类"""
        for cat, condition in self.depth_categories.items():
            if condition(layer_idx, total_layers):
                return cat
        return "middle"

    def compute_gini(self, x: np.ndarray) -> float:
        """计算基尼系数"""
        x_sorted = np.sort(x)
        n = len(x)
        if n == 0:
            return 0.0
        cumsum = np.cumsum(x_sorted)
        total = cumsum[-1]
        if total == 0:
            return 0.0
        # 基尼系数公式
        return (n + 1 - 2 * np.sum(cumsum) / total) / n

    def compute_entropy(self, x: np.ndarray) -> float:
        """计算归一化熵"""
        if np.sum(x) == 0:
            return 0.0
        p = x / np.sum(x)
        p = p[p > 0]  # 避免 log(0)
        entropy = -np.sum(p * np.log2(p))
        max_entropy = np.log2(len(x)) if len(x) > 0 else 1.0
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    def analyze_expert(
        self,
        sensitivity: np.ndarray,
        layer_idx: int,
        expert_idx: int
    ) -> ExpertDistributionMetrics:
        """分析单个专家的分布"""
        # 确保敏感度为正
        sens_pos = np.abs(sensitivity)
        if np.sum(sens_pos) == 0:
            sens_pos = np.ones_like(sensitivity)

        # 排序
        sens_sorted = np.sort(sens_pos)[::-1]
        total_energy = np.sum(sens_sorted)
        cumulative = np.cumsum(sens_sorted) / total_energy

        # 重要微专家占比
        n = len(sens_sorted)
        top10_idx = max(1, n // 10)
        top20_idx = max(1, n // 5)
        top50_idx = max(1, n // 2)

        top10_frac = float(cumulative[top10_idx - 1])
        top20_frac = float(cumulative[top20_idx - 1])
        top50_frac = float(cumulative[top50_idx - 1])

        # 能量分布度量
        gini = self.compute_gini(sens_pos)
        entropy = self.compute_entropy(sens_pos)
        skewness = float(stats.skew(sens_pos))
        kurtosis_val = float(stats.kurtosis(sens_pos))

        return ExpertDistributionMetrics(
            layer_idx=layer_idx,
            expert_idx=expert_idx,
            top10_frac=top10_frac,
            top20_frac=top20_frac,
            top50_frac=top50_frac,
            gini_coefficient=gini,
            entropy=entropy,
            skewness=skewness,
            kurtosis=kurtosis_val,
            cumulative_energy=cumulative
        )

    def analyze_layer(
        self,
        layer_idx: int,
        total_layers: int,
        bit: int
    ) -> Optional[LayerDistributionMetrics]:
        """分析单层"""
        layer = load_layer(
            self.model_id, layer_idx, self.quantmode, self.rank_mode, bits=[bit]
        )

        if layer is None or bit not in layer.by_bit:
            return None

        depth_cat = self.get_depth_category(layer_idx, total_layers)

        # 分析每个专家
        expert_metrics = []
        for exp_idx, sens in enumerate(layer.by_bit[bit]):
            exp_metrics = self.analyze_expert(sens, layer_idx, exp_idx)
            expert_metrics.append(exp_metrics)

        if not expert_metrics:
            return None

        # 层级别汇总
        avg_top10 = float(np.mean([e.top10_frac for e in expert_metrics]))
        avg_top20 = float(np.mean([e.top20_frac for e in expert_metrics]))
        avg_top50 = float(np.mean([e.top50_frac for e in expert_metrics]))
        avg_gini = float(np.mean([e.gini_coefficient for e in expert_metrics]))
        avg_entropy = float(np.mean([e.entropy for e in expert_metrics]))

        # 专家重要性方差（用总能量表示）
        expert_total_energy = [np.sum(np.abs(sens)) for sens in layer.by_bit[bit]]
        importance_variance = float(np.var(expert_total_energy))

        return LayerDistributionMetrics(
            layer_idx=layer_idx,
            depth_category=depth_cat,
            experts=expert_metrics,
            avg_top10_frac=avg_top10,
            avg_top20_frac=avg_top20,
            avg_top50_frac=avg_top50,
            avg_gini=avg_gini,
            avg_entropy=avg_entropy,
            expert_importance_variance=importance_variance
        )

    def analyze_all_layers(self, bit: int = 2) -> MicroExpertDistResult:
        """分析所有层"""
        layers = discover_layers(self.quantmode, self.rank_mode, self.model_id)
        total_layers = max(layers) + 1 if layers else 1

        print(f"\n{'='*60}")
        print(f"微专家分布分析: {model_label(self.model_id)}")
        print(f"模式: {self.rank_mode} @ {bit}bit")
        print(f"{'='*60}\n")

        by_layer = []
        for layer_idx in tqdm(layers, desc="分析层"):
            metrics = self.analyze_layer(layer_idx, total_layers, bit)
            if metrics is not None:
                by_layer.append(metrics)

        # 计算汇总统计
        summary_stats = self._compute_summary_stats(by_layer)

        # 打印表格
        self._print_summary_tables(summary_stats, by_layer)

        return MicroExpertDistResult(
            model_id=self.model_id,
            by_layer=by_layer,
            summary_stats=summary_stats
        )

    def _compute_summary_stats(
        self,
        by_layer: List[LayerDistributionMetrics]
    ) -> Dict[str, Dict[str, float]]:
        """计算汇总统计"""
        summary = {}

        # 按深度分类
        for depth in ["shallow", "middle", "deep"]:
            depth_layers = [l for l in by_layer if l.depth_category == depth]
            if not depth_layers:
                continue

            # 收集所有专家
            all_experts = [e for l in depth_layers for e in l.experts]

            summary[depth] = {
                "num_layers": len(depth_layers),
                "num_experts": len(all_experts),
                "top10_mean": float(np.mean([e.top10_frac for e in all_experts])),
                "top10_std": float(np.std([e.top10_frac for e in all_experts])),
                "top20_mean": float(np.mean([e.top20_frac for e in all_experts])),
                "top20_std": float(np.std([e.top20_frac for e in all_experts])),
                "top50_mean": float(np.mean([e.top50_frac for e in all_experts])),
                "top50_std": float(np.std([e.top50_frac for e in all_experts])),
                "gini_mean": float(np.mean([e.gini_coefficient for e in all_experts])),
                "gini_std": float(np.std([e.gini_coefficient for e in all_experts])),
                "entropy_mean": float(np.mean([e.entropy for e in all_experts])),
                "entropy_std": float(np.std([e.entropy for e in all_experts])),
            }

        # 总体统计
        if by_layer:
            all_experts_all = [e for l in by_layer for e in l.experts]
            summary["overall"] = {
                "num_layers": len(by_layer),
                "num_experts": len(all_experts_all),
                "top10_mean": float(np.mean([e.top10_frac for e in all_experts_all])),
                "top10_std": float(np.std([e.top10_frac for e in all_experts_all])),
                "top20_mean": float(np.mean([e.top20_frac for e in all_experts_all])),
                "top20_std": float(np.std([e.top20_frac for e in all_experts_all])),
                "top50_mean": float(np.mean([e.top50_frac for e in all_experts_all])),
                "top50_std": float(np.std([e.top50_frac for e in all_experts_all])),
                "gini_mean": float(np.mean([e.gini_coefficient for e in all_experts_all])),
                "gini_std": float(np.std([e.gini_coefficient for e in all_experts_all])),
                "entropy_mean": float(np.mean([e.entropy for e in all_experts_all])),
                "entropy_std": float(np.std([e.entropy for e in all_experts_all])),
            }

        return summary

    def _print_summary_tables(
        self,
        summary_stats: Dict[str, Dict[str, float]],
        by_layer: List[LayerDistributionMetrics]
    ):
        """打印汇总表格"""
        # 表1: 重要微专家占比
        print("\n" + "="*100)
        print("表1: 重要微专家敏感度占比汇总 (Top K% 神经元的敏感度占比)")
        print("="*100)
        print(f"\n{'深度分类':<15} {'专家数':<8} {'Top10%(mean±std)':<18} "
              f"{'Top20%(mean±std)':<18} {'Top50%(mean±std)':<18}")
        print("-"*100)

        for depth in ["shallow", "middle", "deep", "overall"]:
            if depth not in summary_stats:
                continue
            s = summary_stats[depth]
            depth_label = {
                "shallow": "浅层",
                "middle": "中层",
                "deep": "深层",
                "overall": "总体"
            }[depth]

            print(f"{depth_label:<15} {s['num_experts']:<8} "
                  f"{s['top10_mean']:.3f}±{s['top10_std']:.3f}".ljust(18) + " "
                  f"{s['top20_mean']:.3f}±{s['top20_std']:.3f}".ljust(18) + " "
                  f"{s['top50_mean']:.3f}±{s['top50_std']:.3f}".ljust(18))

        print("-"*100)

        # 表2: 敏感度分布度量
        print("\n" + "="*80)
        print("表2: 敏感度分布度量汇总")
        print("="*80)
        print(f"\n{'深度分类':<15} {'专家数':<8} {'基尼系数':<15} {'归一化熵':<15}")
        print("-"*80)

        for depth in ["shallow", "middle", "deep", "overall"]:
            if depth not in summary_stats:
                continue
            s = summary_stats[depth]
            depth_label = {
                "shallow": "浅层",
                "middle": "中层",
                "deep": "深层",
                "overall": "总体"
            }[depth]

            print(f"{depth_label:<15} {s['num_experts']:<8} "
                  f"{s['gini_mean']:.3f}±{s['gini_std']:.3f}".ljust(15) + " "
                  f"{s['entropy_mean']:.3f}±{s['entropy_std']:.3f}".ljust(15))

        print("-"*80)

        # 表3: 跨层重要性偏移
        self._print_layer_trend_table(by_layer)

    def _print_layer_trend_table(self, by_layer: List[LayerDistributionMetrics]):
        """打印层趋势表"""
        if not by_layer:
            return

        print("\n" + "="*100)
        print("表3: 跨层重要性偏移 (选择代表性层)")
        print("="*100)

        layers_sorted = sorted(by_layer, key=lambda x: x.layer_idx)

        # 选择代表性层
        if len(layers_sorted) > 10:
            step = len(layers_sorted) // 10
            display_layers = layers_sorted[::step]
        else:
            display_layers = layers_sorted

        print(f"\n{'层号':<6} {'深度':<10} {'Top10%':<10} {'Top20%':<10} "
              f"{'基尼系数':<10} {'熵':<10} {'专家重要性方差':<15}")
        print("-"*100)

        for layer in display_layers:
            depth_label = {
                "shallow": "浅",
                "middle": "中",
                "deep": "深"
            }[layer.depth_category]

            print(f"{layer.layer_idx:<6} {depth_label:<10} "
                  f"{layer.avg_top10_frac:<10.3f} {layer.avg_top20_frac:<10.3f} "
                  f"{layer.avg_gini:<10.3f} {layer.avg_entropy:<10.3f} "
                  f"{layer.expert_importance_variance:<15.2e}")

        print("-"*100 + "\n")

    def plot_results(
        self,
        results_by_bit: Dict[int, MicroExpertDistResult],
        save_dir: Optional[str] = None
    ):
        """绘制结果图"""
        apply_paper_style()

        if save_dir is None:
            save_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "plot", "stat_micro_expert_dist"
            )
        os.makedirs(save_dir, exist_ok=True)

        # 只画分布度量，按 bit 对比
        self._plot_distribution_metrics_by_bit(results_by_bit, save_dir)

        print(f"\n图表已保存到: {save_dir}")

    def _plot_distribution_metrics_by_bit(
        self,
        results_by_bit: Dict[int, MicroExpertDistResult],
        save_dir: str
    ):
        """绘制分布度量（按 bit 对比）"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        bit_order = sorted(results_by_bit.keys())
        bit_labels = [f'{b}bit' for b in bit_order]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(bit_order)]

        # 获取 model_id
        model_id = next(iter(results_by_bit.values())).model_id

        # 收集所有专家按 bit
        all_experts_by_bit = {}
        for bit in bit_order:
            result = results_by_bit[bit]
            all_experts_by_bit[bit] = [e for l in result.by_layer for e in l.experts]

        # 基尼系数箱线图
        ax1 = axes[0, 0]
        gini_data = [
            [e.gini_coefficient for e in all_experts_by_bit[b]]
            for b in bit_order
        ]
        if gini_data:
            bp1 = ax1.boxplot(gini_data, patch_artist=True)
            for patch, color in zip(bp1['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax1.set_xticklabels(bit_labels)
            ax1.set_ylabel('Gini Coefficient')
            ax1.set_title(f'{model_label(model_id)}: Gini Coefficient by Bit Width')
            ax1.grid(True, alpha=0.3, axis='y')

        # 熵箱线图
        ax2 = axes[0, 1]
        entropy_data = [
            [e.entropy for e in all_experts_by_bit[b]]
            for b in bit_order
        ]
        if entropy_data:
            bp2 = ax2.boxplot(entropy_data, patch_artist=True)
            for patch, color in zip(bp2['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax2.set_xticklabels(bit_labels)
            ax2.set_ylabel('Normalized Entropy')
            ax2.set_title(f'{model_label(model_id)}: Entropy by Bit Width')
            ax2.grid(True, alpha=0.3, axis='y')

        # 基尼系数 vs 熵散点图
        ax3 = axes[1, 0]
        for bit, color, label in zip(bit_order, colors, bit_labels):
            if not all_experts_by_bit[bit]:
                continue
            gini_vals = [e.gini_coefficient for e in all_experts_by_bit[bit]]
            entropy_vals = [e.entropy for e in all_experts_by_bit[bit]]
            ax3.scatter(gini_vals, entropy_vals, c=color, alpha=0.5, s=20, label=label)
        ax3.set_xlabel('Gini Coefficient')
        ax3.set_ylabel('Normalized Entropy')
        ax3.set_title(f'{model_label(model_id)}: Gini vs Entropy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Top10% vs 基尼系数散点图
        ax4 = axes[1, 1]
        for bit, color, label in zip(bit_order, colors, bit_labels):
            if not all_experts_by_bit[bit]:
                continue
            top10_vals = [e.top10_frac for e in all_experts_by_bit[bit]]
            gini_vals = [e.gini_coefficient for e in all_experts_by_bit[bit]]
            ax4.scatter(top10_vals, gini_vals, c=color, alpha=0.5, s=20, label=label)
        ax4.set_xlabel('Top 10% Sensitivity Fraction')
        ax4.set_ylabel('Gini Coefficient')
        ax4.set_title(f'{model_label(model_id)}: Top10% vs Gini')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{model_id}_distribution_metrics_by_bit.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()


def export_results_to_csv(
    result: MicroExpertDistResult,
    save_dir: Optional[str] = None,
    bit: int = 2
):
    """导出结果到 CSV"""
    if save_dir is None:
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "plot", "stat_micro_expert_dist"
        )
    os.makedirs(save_dir, exist_ok=True)

    # 专家级别数据
    expert_data = []
    for layer in result.by_layer:
        for exp in layer.experts:
            expert_data.append({
                "model_id": result.model_id,
                "layer_idx": exp.layer_idx,
                "expert_idx": exp.expert_idx,
                "depth_category": layer.depth_category,
                "top10_frac": exp.top10_frac,
                "top20_frac": exp.top20_frac,
                "top50_frac": exp.top50_frac,
                "gini_coefficient": exp.gini_coefficient,
                "entropy": exp.entropy,
                "skewness": exp.skewness,
                "kurtosis": exp.kurtosis,
            })

    df_expert = pd.DataFrame(expert_data)
    df_expert.to_csv(os.path.join(save_dir, f'{result.model_id}_expert_metrics_b{bit}.csv'),
                     index=False)

    # 层级别数据
    layer_data = []
    for layer in result.by_layer:
        layer_data.append({
            "model_id": result.model_id,
            "layer_idx": layer.layer_idx,
            "depth_category": layer.depth_category,
            "num_experts": len(layer.experts),
            "avg_top10_frac": layer.avg_top10_frac,
            "avg_top20_frac": layer.avg_top20_frac,
            "avg_top50_frac": layer.avg_top50_frac,
            "avg_gini": layer.avg_gini,
            "avg_entropy": layer.avg_entropy,
            "expert_importance_variance": layer.expert_importance_variance,
        })

    df_layer = pd.DataFrame(layer_data)
    df_layer.to_csv(os.path.join(save_dir, f'{result.model_id}_layer_metrics_b{bit}.csv'),
                    index=False)

    # 汇总统计
    summary_data = []
    for depth, stats in result.summary_stats.items():
        row = {"model_id": result.model_id, "depth_category": depth}
        row.update(stats)
        summary_data.append(row)

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(save_dir, f'{result.model_id}_summary_metrics_b{bit}.csv'),
                      index=False)

    print(f"CSV 已导出到: {save_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="微专家分布可视化量化")
    parser.add_argument("--model", type=str, default="deepseek-v1-moe-16b",
                        help="模型 ID 或路径")
    parser.add_argument("--quantmode", type=str, default="turboquant",
                        choices=["turboquant", "gptq"],
                        help="量化模式")
    parser.add_argument("--rank_mode", type=str, default="turboquant_innerproduct",
                        help="排序模式")
    parser.add_argument("--bits", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="比特位数")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="保存目录")
    parser.add_argument("--no_plot", action="store_true",
                        help="跳过绘图")
    parser.add_argument("--export_csv", action="store_true", default=True,
                        help="导出 CSV")

    args = parser.parse_args()

    analyzer = MicroExpertDistAnalyzer(
        model_id=args.model,
        quantmode=args.quantmode,
        rank_mode=args.rank_mode,
        bits=args.bits
    )

    results_by_bit = {}
    for bit in args.bits:
        print(f"\n{'='*80}")
        print(f"处理 {bit}bit")
        print(f"{'='*80}")

        result = analyzer.analyze_all_layers(bit=bit)

        if not result.by_layer:
            print(f"警告: 没有找到 {bit}bit 的数据")
            continue

        results_by_bit[bit] = result

        if args.export_csv:
            export_results_to_csv(result, save_dir=args.save_dir, bit=bit)

    if not args.no_plot and results_by_bit:
        analyzer.plot_results(results_by_bit, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
