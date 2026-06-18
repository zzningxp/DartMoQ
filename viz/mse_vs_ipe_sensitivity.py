"""
TurboQuant量化：MSE vs IPE 敏感度对比实验

包含4个核心实验：
1. 实验1：MSE vs IPE 的敏感度区分度对比
2. 实验2：正交旋转的能量均匀化效应可视化
3. 实验3：不同损失函数下DP切分的效果对比
4. 实验4：神经元量化异质性与比特秩相关性
"""

import sys
import os

# 添加父目录到路径，确保能找到 turboquant_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr

from turboquant_utils.rotation import generate_rotation_matrix
from turboquant_utils.rotation import hadamard_rotate, hadamard_rotate_inverse


class TurboQuantSensitivityTest:
    """TurboQuant敏感度测试主类"""

    def __init__(
        self,
        weight_matrix: torch.Tensor,
        calibration_inputs: Optional[torch.Tensor] = None,
        bits_list: List[int] = [0, 1, 2, 3, 4],
        groupsize: int = 128,
        seed: int = 42,
        device: str = None
    ):
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.weight_matrix = weight_matrix.to(device)
        self.calibration_inputs = (
            calibration_inputs.to(device) if calibration_inputs is not None else None
        )
        self.bits_list = bits_list
        self.groupsize = groupsize
        self.seed = seed
        self.device = device

        self.out_dim, self.in_dim = weight_matrix.shape

        if self.calibration_inputs is None:
            self._generate_calibration_inputs()

        self.rotation_matrix = None
        self.rotated_weight = None

    def _generate_calibration_inputs(self, num_samples: int = 128):
        torch.manual_seed(self.seed)
        self.calibration_inputs = torch.randn(
            num_samples, self.in_dim, device=self.device
        ) * 0.1
        print(f"Generated calibration inputs: {self.calibration_inputs.shape}")

    def apply_orthogonal_rotation(self, method: str = 'qr'):
        if method == 'qr':
            self.rotation_matrix = generate_rotation_matrix(
                self.in_dim, seed=self.seed
            ).to(self.device)
            self.rotated_weight = self.weight_matrix @ self.rotation_matrix.T
        elif method == 'hadamard':
            if (self.in_dim & (self.in_dim - 1)) != 0:
                print(f"Warning: in_dim={self.in_dim} not power of 2, using QR")
                self.rotation_matrix = generate_rotation_matrix(
                    self.in_dim, seed=self.seed
                ).to(self.device)
                self.rotated_weight = self.weight_matrix @ self.rotation_matrix.T
            else:
                self.rotated_weight = hadamard_rotate(
                    self.weight_matrix, seed=self.seed
                )
        else:
            raise ValueError(f"Unknown rotation method: {method}")

        print(f"Applied orthogonal rotation: {method}")
        return self.rotated_weight

    def quantize_weight(
        self,
        weight: torch.Tensor,
        bits: int,
        groupsize: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if bits == 0:
            return torch.zeros_like(weight), weight

        if groupsize is None:
            groupsize = self.groupsize

        n_groups = weight.shape[1] // groupsize
        weight_grouped = weight.reshape(weight.shape[0], n_groups, groupsize)

        w_min = weight_grouped.min(dim=-1, keepdim=True)[0]
        w_max = weight_grouped.max(dim=-1, keepdim=True)[0]

        scale = (w_max - w_min) / (2**bits - 1)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        q_weight = torch.round((weight_grouped - w_min) / scale)
        q_weight = torch.clamp(q_weight, 0, 2**bits - 1)

        dq_weight = q_weight * scale + w_min
        dq_weight = dq_weight.reshape(weight.shape[0], weight.shape[1])
        quant_error = weight - dq_weight

        return dq_weight, quant_error

    def calculate_mse(self, weight_original: torch.Tensor, weight_quant: torch.Tensor) -> torch.Tensor:
        return (weight_quant - weight_original).pow(2).mean(dim=-1)

    def calculate_ipe(
        self,
        weight_original: torch.Tensor,
        weight_quant: torch.Tensor,
        inputs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if inputs is None:
            inputs = self.calibration_inputs

        with torch.no_grad():
            orig_output = F.linear(inputs, weight_original)
            quant_output = F.linear(inputs, weight_quant)

        return (quant_output - orig_output).pow(2).mean(dim=0)

    def test1_mse_vs_ipe_distinction(
        self,
        bits: int = 2,
        use_rotation: bool = True,
        save_path: Optional[str] = None
    ) -> Dict:
        print("\n" + "="*60)
        print("实验1：MSE vs IPE 敏感度区分度对比")
        print("="*60)

        if use_rotation and self.rotated_weight is not None:
            test_weight = self.rotated_weight
            orig_weight = self.weight_matrix
        else:
            test_weight = self.weight_matrix
            orig_weight = self.weight_matrix

        quant_weight, _ = self.quantize_weight(test_weight, bits)

        if use_rotation and self.rotation_matrix is not None:
            quant_weight_orig_space = quant_weight @ self.rotation_matrix
        else:
            quant_weight_orig_space = quant_weight

        mse_per_neuron = self.calculate_mse(test_weight, quant_weight)
        ipe_per_neuron = self.calculate_ipe(orig_weight, quant_weight_orig_space)

        mse_sorted, _ = torch.sort(mse_per_neuron, descending=True)
        ipe_sorted, _ = torch.sort(ipe_per_neuron, descending=True)

        def stats_dict(errors):
            return {
                'mean': errors.mean().item(),
                'std': errors.std().item(),
                'max': errors.max().item(),
                'min': errors.min().item(),
                'cv': (errors.std() / (errors.mean() + 1e-12)).item(),
                'max_min_ratio': (errors.max() / (errors.min() + 1e-12)).item()
            }

        mse_stats = stats_dict(mse_per_neuron)
        ipe_stats = stats_dict(ipe_per_neuron)

        print(f"\nMSE 统计:")
        for k, v in mse_stats.items():
            print(f"  {k:15s}: {v:.6e}")

        print(f"\nIPE 统计:")
        for k, v in ipe_stats.items():
            print(f"  {k:15s}: {v:.6e}")

        if save_path is not None:
            self._plot_test1(mse_sorted, ipe_sorted, mse_stats, ipe_stats, save_path)

        return {
            'mse': mse_per_neuron.cpu().numpy(),
            'ipe': ipe_per_neuron.cpu().numpy(),
            'mse_sorted': mse_sorted.cpu().numpy(),
            'ipe_sorted': ipe_sorted.cpu().numpy(),
            'mse_stats': mse_stats,
            'ipe_stats': ipe_stats
        }

    def _plot_test1(self, mse_sorted, ipe_sorted, mse_stats, ipe_stats, save_path):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        x = np.arange(len(mse_sorted))
        ax1 = axes[0]
        ax1.plot(x, mse_sorted, 'b-', label='MSE', alpha=0.8, linewidth=2)
        ax1.plot(x, ipe_sorted, 'r-', label='IPE', alpha=0.8, linewidth=2)
        ax1.set_xlabel('Neuron Index (sorted)', fontsize=12)
        ax1.set_ylabel('Error Value', fontsize=12)
        ax1.set_title('MSE vs IPE (Linear Scale)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)

        stats_text = (
            f"MSE CV: {mse_stats['cv']:.3f}\n"
            f"IPE CV: {ipe_stats['cv']:.3f}\n"
            f"Ratio: {ipe_stats['cv']/(mse_stats['cv']+1e-6):.1f}x"
        )
        ax1.text(0.05, 0.95, stats_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax2 = axes[1]
        ax2.plot(x, mse_sorted + 1e-12, 'b-', label='MSE', alpha=0.8, linewidth=2)
        ax2.plot(x, ipe_sorted + 1e-12, 'r-', label='IPE', alpha=0.8, linewidth=2)
        ax2.set_yscale('log')
        ax2.set_xlabel('Neuron Index (sorted)', fontsize=12)
        ax2.set_ylabel('Error Value (log scale)', fontsize=12)
        ax2.set_title('MSE vs IPE (Log Scale)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"实验1图已保存: {save_path}")

    def test2_rotation_effect_on_weights(
        self,
        save_path: Optional[str] = None
    ) -> Dict:
        print("\n" + "="*60)
        print("实验2：正交旋转的能量均匀化效应可视化")
        print("="*60)

        if self.rotated_weight is None:
            self.apply_orthogonal_rotation()

        quant_weight_rot, _ = self.quantize_weight(self.rotated_weight, bits=2)
        quant_weight_orig = quant_weight_rot @ self.rotation_matrix
        ipe_per_neuron = self.calculate_ipe(self.weight_matrix, quant_weight_orig)

        ipe_sorted, sorted_indices = torch.sort(ipe_per_neuron, descending=True)
        high_idx = sorted_indices[0].item()
        low_idx = sorted_indices[-1].item()

        high_orig = self.weight_matrix[high_idx].cpu().numpy()
        high_rot = self.rotated_weight[high_idx].cpu().numpy()
        low_orig = self.weight_matrix[low_idx].cpu().numpy()
        low_rot = self.rotated_weight[low_idx].cpu().numpy()

        def stats_for_vec(vec):
            sorted_abs = np.sort(np.abs(vec))[::-1]
            top10 = sorted_abs[:len(sorted_abs)//10].sum()
            total = sorted_abs.sum()
            return {
                'top10_ratio': top10 / (total + 1e-12),
                'max_min_ratio': sorted_abs[0] / (sorted_abs[-1] + 1e-12)
            }

        stats = {
            'high_orig': stats_for_vec(high_orig),
            'high_rot': stats_for_vec(high_rot),
            'low_orig': stats_for_vec(low_orig),
            'low_rot': stats_for_vec(low_rot)
        }

        print(f"\n高敏感神经元 #{high_idx}:")
        print(f"  旋转前 - Top10%: {stats['high_orig']['top10_ratio']:.3f}")
        print(f"  旋转后 - Top10%: {stats['high_rot']['top10_ratio']:.3f}")
        print(f"\n低敏感神经元 #{low_idx}:")
        print(f"  旋转前 - Top10%: {stats['low_orig']['top10_ratio']:.3f}")
        print(f"  旋转后 - Top10%: {stats['low_rot']['top10_ratio']:.3f}")

        if save_path is not None:
            self._plot_test2(
                np.sort(np.abs(high_orig))[::-1],
                np.sort(np.abs(high_rot))[::-1],
                np.sort(np.abs(low_orig))[::-1],
                np.sort(np.abs(low_rot))[::-1],
                high_idx, low_idx, stats, save_path
            )

        return stats

    def _plot_test2(self, h_orig, h_rot, l_orig, l_rot, h_idx, l_idx, stats, save_path):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        x = np.arange(len(h_orig))

        ax1 = axes[0, 0]
        ax1.plot(x, h_orig, 'r-', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Weight Component (sorted)', fontsize=11)
        ax1.set_ylabel('Absolute Weight', fontsize=11)
        ax1.set_title(f'High-sensitive #{h_idx} (Before Rotation)', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.text(0.95, 0.95,
                f"Top10%: {stats['high_orig']['top10_ratio']:.3f}",
                transform=ax1.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax2 = axes[0, 1]
        ax2.plot(x, l_orig, 'b-', linewidth=2, alpha=0.8)
        ax2.set_xlabel('Weight Component (sorted)', fontsize=11)
        ax2.set_ylabel('Absolute Weight', fontsize=11)
        ax2.set_title(f'Low-sensitive #{l_idx} (Before Rotation)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.text(0.95, 0.95,
                f"Top10%: {stats['low_orig']['top10_ratio']:.3f}",
                transform=ax2.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax3 = axes[1, 0]
        ax3.plot(x, h_rot, 'r-', linewidth=2, alpha=0.8)
        ax3.set_xlabel('Weight Component (sorted)', fontsize=11)
        ax3.set_ylabel('Absolute Weight', fontsize=11)
        ax3.set_title(f'High-sensitive #{h_idx} (After Rotation)', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.text(0.95, 0.95,
                f"Top10%: {stats['high_rot']['top10_ratio']:.3f}",
                transform=ax3.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax4 = axes[1, 1]
        ax4.plot(x, l_rot, 'b-', linewidth=2, alpha=0.8)
        ax4.set_xlabel('Weight Component (sorted)', fontsize=11)
        ax4.set_ylabel('Absolute Weight', fontsize=11)
        ax4.set_title(f'Low-sensitive #{l_idx} (After Rotation)', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.text(0.95, 0.95,
                f"Top10%: {stats['low_rot']['top10_ratio']:.3f}",
                transform=ax4.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"实验2图已保存: {save_path}")

    def test3_dp_allocation_comparison(
        self,
        bits_list: Optional[List[int]] = None,
        bpw_values: Optional[List[float]] = None,
        save_path_prefix: Optional[str] = None
    ) -> Dict:
        print("\n" + "="*60)
        print("实验3：MSE vs IPE排序下DP切分效果对比")
        print("="*60)

        if bits_list is None:
            bits_list = [0, 2, 4]
        if bpw_values is None:
            bpw_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

        if self.rotated_weight is None:
            self.apply_orthogonal_rotation()

        quant_weight_rot, _ = self.quantize_weight(self.rotated_weight, bits=2)
        mse = self.calculate_mse(self.rotated_weight, quant_weight_rot).cpu().numpy()

        quant_weight_orig = quant_weight_rot @ self.rotation_matrix
        ipe = self.calculate_ipe(self.weight_matrix, quant_weight_orig).cpu().numpy()

        mse_results = []
        ipe_results = []

        for bpw in bpw_values:
            mse_alloc, mse_loss = self._greedy_bit_alloc(mse, bits_list, bpw)
            ipe_alloc, ipe_loss = self._greedy_bit_alloc(ipe, bits_list, bpw)

            mse_results.append({'bpw': bpw, 'loss': mse_loss, 'alloc': mse_alloc})
            ipe_results.append({'bpw': bpw, 'loss': ipe_loss, 'alloc': ipe_alloc})

            print(f"\nbpw={bpw:.1f}:")
            print(f"  MSE-based - loss: {mse_loss:.6e}, 0bit: {(mse_alloc == 0).sum()}")
            print(f"  IPE-based - loss: {ipe_loss:.6e}, 0bit: {(ipe_alloc == 0).sum()}")

        int0_ratios = np.linspace(0, 0.5, 11)
        mse_int0_losses = []
        ipe_int0_losses = []

        for ratio in int0_ratios:
            n = len(mse)
            n0 = int(n * ratio)
            mse_sorted = np.argsort(mse)
            ipe_sorted = np.argsort(ipe)

            mse_0_mask = np.zeros(n, dtype=bool)
            mse_0_mask[mse_sorted[:n0]] = True
            ipe_0_mask = np.zeros(n, dtype=bool)
            ipe_0_mask[ipe_sorted[:n0]] = True

            mse_int0_losses.append(((mse[mse_0_mask] * 10).sum() + (mse[~mse_0_mask]).sum()) / n)
            ipe_int0_losses.append(((ipe[ipe_0_mask] * 10).sum() + (ipe[~ipe_0_mask]).sum()) / n)

        if save_path_prefix is not None:
            self._plot_test3(bpw_values, [r['loss'] for r in mse_results],
                           [r['loss'] for r in ipe_results],
                           int0_ratios, mse_int0_losses, ipe_int0_losses,
                           save_path_prefix)

        return {
            'bpw_values': bpw_values,
            'mse_results': mse_results,
            'ipe_results': ipe_results
        }

    def _greedy_bit_alloc(self, loss_per_neuron, bits_list, target_bpw):
        n = len(loss_per_neuron)
        sorted_idx = np.argsort(loss_per_neuron)[::-1]
        bits_available = sorted([b for b in bits_list if b > 0], reverse=True)
        min_bit = min(b for b in bits_list if b > 0)

        allocation = np.ones(n, dtype=int) * min_bit
        remaining = target_bpw * n - min_bit * n

        for bit in bits_available:
            if bit <= min_bit:
                continue
            for i in sorted_idx:
                cost = bit - allocation[i]
                if remaining >= cost and allocation[i] < bit:
                    remaining -= cost
                    allocation[i] = bit

        if 0 in bits_list and remaining < 0:
            for i in sorted_idx[::-1]:
                if allocation[i] == min_bit:
                    freed = min_bit
                    allocation[i] = 0
                    remaining += freed
                    if remaining >= 0:
                        break

        total_loss = 0.0
        for i in range(n):
            bit = allocation[i]
            scale = 10.0 if bit == 0 else (4.0 / bit) ** 2
            total_loss += loss_per_neuron[i] * scale

        return allocation, total_loss / n

    def _plot_test3(self, bpw_values, mse_losses, ipe_losses, int0_ratios,
                  mse_int0, ipe_int0, save_prefix):
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 7))
        ax1.plot(bpw_values, mse_losses, 'b-o', linewidth=2, markersize=8, label='MSE-based')
        ax1.plot(bpw_values, ipe_losses, 'r-s', linewidth=2, markersize=8, label='IPE-based')
        ax1.set_xlabel('Target Bits per Weight', fontsize=13)
        ax1.set_ylabel('Average Loss', fontsize=13)
        ax1.set_title('Bit Budget vs Loss', fontsize=15, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)

        for i, bpw in enumerate(bpw_values):
            improvement = (mse_losses[i] - ipe_losses[i]) / (mse_losses[i] + 1e-12) * 100
            if improvement > 0:
                ax1.annotate(f'{improvement:.1f}%↓',
                           xy=(bpw, ipe_losses[i]),
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(f"{save_prefix}_bpw_vs_loss.png", dpi=150, bbox_inches='tight')
        plt.close()

        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 7))
        ax2.plot(int0_ratios * 100, mse_int0, 'b-o', linewidth=2, markersize=8, label='MSE-based')
        ax2.plot(int0_ratios * 100, ipe_int0, 'r-s', linewidth=2, markersize=8, label='IPE-based')
        ax2.set_xlabel('Int0 Pruning Ratio (%)', fontsize=13)
        ax2.set_ylabel('Average Loss', fontsize=13)
        ax2.set_title('Int0 Pruning Robustness', fontsize=15, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_prefix}_int0_ratio.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"实验3图已保存: {save_prefix}_*.png")

    def test4_bit_rank_correlation(
        self,
        bits_list: Optional[List[int]] = None,
        save_path_prefix: Optional[str] = None
    ) -> Dict:
        print("\n" + "="*60)
        print("实验4：不同比特下神经元敏感度秩相关性")
        print("="*60)

        if bits_list is None:
            bits_list = [1, 2, 3, 4]

        if self.rotated_weight is None:
            self.apply_orthogonal_rotation()

        ipe_per_bit = {}
        for bits in bits_list:
            qw_rot, _ = self.quantize_weight(self.rotated_weight, bits)
            qw_orig = qw_rot @ self.rotation_matrix
            ipe = self.calculate_ipe(self.weight_matrix, qw_orig).cpu().numpy()
            ipe_per_bit[bits] = ipe

        n_bits = len(bits_list)
        corr_matrix = np.zeros((n_bits, n_bits))

        for i, bit1 in enumerate(bits_list):
            for j, bit2 in enumerate(bits_list):
                corr, _ = spearmanr(ipe_per_bit[bit1], ipe_per_bit[bit2])
                corr_matrix[i, j] = corr

        print("\nSpearman秩相关矩阵:")
        print("     " + "  ".join([f"{b:4d}bit" for b in bits_list]))
        for i, bit1 in enumerate(bits_list):
            line = f"{bit1:2d}bit"
            for j, bit2 in enumerate(bits_list):
                line += f"  {corr_matrix[i, j]:.3f}"
            print(line)

        avg_ipe = np.mean([ipe_per_bit[b] for b in bits_list], axis=0)
        sorted_idx = np.argsort(avg_ipe)[::-1]
        step = len(sorted_idx) // 6
        sample_idx = sorted_idx[step : step*6 : step]
        sample_curves = {idx: [ipe_per_bit[b][idx] for b in bits_list] for idx in sample_idx}

        if save_path_prefix is not None:
            self._plot_test4(bits_list, corr_matrix, sample_curves, save_path_prefix)

        return {
            'bits_list': bits_list,
            'ipe_per_bit': ipe_per_bit,
            'corr_matrix': corr_matrix
        }

    def _plot_test4(self, bits_list, corr_matrix, sample_curves, save_prefix):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        ax1 = axes[0]
        im = ax1.imshow(corr_matrix, cmap='viridis', vmin=0, vmax=1)
        ax1.set_xticks(np.arange(len(bits_list)))
        ax1.set_yticks(np.arange(len(bits_list)))
        ax1.set_xticklabels([f"{b}bit" for b in bits_list], fontsize=11)
        ax1.set_yticklabels([f"{b}bit" for b in bits_list], fontsize=11)
        ax1.set_title('Spearman Rank Correlation', fontsize=14, fontweight='bold')

        for i in range(len(bits_list)):
            for j in range(len(bits_list)):
                ax1.text(j, i, f"{corr_matrix[i, j]:.3f}",
                        ha="center", va="center",
                        color="white" if corr_matrix[i, j] < 0.7 else "black", fontsize=12)

        plt.colorbar(im, ax=ax1, label='Correlation Coefficient')

        ax2 = axes[1]
        colors = plt.cm.tab10(np.linspace(0, 1, len(sample_curves)))
        for (idx, curve), color in zip(sample_curves.items(), colors):
            ax2.plot(bits_list, curve, 'o-', linewidth=2, markersize=8,
                    color=color, label=f'Neuron #{idx}')

        ax2.set_xlabel('Bit Width', fontsize=13)
        ax2.set_ylabel('IPE (log scale)', fontsize=13)
        ax2.set_yscale('log')
        ax2.set_title('Quantization Error vs Bit Width', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(bits_list)

        plt.tight_layout()
        plt.savefig(f"{save_prefix}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"实验4图已保存: {save_prefix}.png")

    def run_all_tests(
        self,
        save_dir: str = None,
        bits_for_test1: int = 2
    ) -> Dict:
        if save_dir is None:
            # 默认保存到项目根目录下的 plot/mse_vs_ipe
            current_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(os.path.dirname(current_dir), 'plot', 'mse_vs_ipe')
        os.makedirs(save_dir, exist_ok=True)
        results = {}

        results['test1'] = self.test1_mse_vs_ipe_distinction(
            bits=bits_for_test1,
            save_path=os.path.join(save_dir, 'test1_mse_vs_ipe.png')
        )
        results['test2'] = self.test2_rotation_effect_on_weights(
            save_path=os.path.join(save_dir, 'test2_rotation_effect.png')
        )
        results['test3'] = self.test3_dp_allocation_comparison(
            save_path_prefix=os.path.join(save_dir, 'test3_dp_comparison')
        )
        results['test4'] = self.test4_bit_rank_correlation(
            save_path_prefix=os.path.join(save_dir, 'test4_bit_correlation')
        )

        print("\n" + "="*60)
        print("所有测试完成！")
        print("="*60)

        return results


def create_synthetic_weight(
    out_dim: int = 256,
    in_dim: int = 512,
    seed: int = 42,
    device: str = 'cpu'
):
    torch.manual_seed(seed)
    weight = torch.randn(out_dim, in_dim, device=device) * 0.02

    n_high_sensitive = out_dim // 4
    n_important_dims = in_dim // 10

    for i in range(n_high_sensitive):
        important_dims = torch.randperm(in_dim, device=device)[:n_important_dims]
        weight[i, important_dims] += torch.randn(n_important_dims, device=device) * 0.1

    calib = torch.randn(128, in_dim, device=device) * 0.1
    return weight, calib


def run_demo(save_dir: str = None, device: str = None):
    if save_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(os.path.dirname(current_dir), 'plot', 'mse_vs_ipe')
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    print("创建合成权重...")
    weight, calib = create_synthetic_weight(out_dim=256, in_dim=512, seed=42, device=device)

    print("初始化测试...")
    tester = TurboQuantSensitivityTest(
        weight_matrix=weight,
        calibration_inputs=calib,
        seed=42,
        device=device
    )

    tester.apply_orthogonal_rotation()
    results = tester.run_all_tests(save_dir=save_dir)
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MSE vs IPE 敏感度对比')
    parser.add_argument('--save_dir', type=str, default=None, help='保存目录 (默认: ../plot/mse_vs_ipe)')
    parser.add_argument('--device', type=str, default=None, help='运行设备 (默认自动选择)')
    parser.add_argument('--small', action='store_true', help='用更小的矩阵(更快)')
    parser.add_argument('--demo', action='store_true', help='用合成数据运行演示')
    args = parser.parse_args()

    if args.demo:
        if args.save_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            args.save_dir = os.path.join(os.path.dirname(current_dir), 'plot', 'mse_vs_ipe')
        if args.small:
            import torch
            if args.device is None:
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            else:
                device = args.device
            print(f"使用设备: {device}")
            print("创建小尺寸合成权重 (128x256)...")
            weight, calib = create_synthetic_weight(out_dim=128, in_dim=256, seed=42, device=device)
            tester = TurboQuantSensitivityTest(weight, calib, seed=42, device=device)
            tester.apply_orthogonal_rotation()
            tester.run_all_tests(save_dir=args.save_dir)
        else:
            run_demo(args.save_dir, args.device)
