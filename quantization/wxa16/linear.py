#!/usr/bin/env python3
"""
WxA16 Linear Module - 存储 packed 量化权重，推理时反量化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

from turboquant_utils.quantize import (
    turboquant_quantize_packed_full,
    turboquant_dequantize_packed,
    unpack_nbit,
)
from turboquant_utils.quantize import get_codebook
from turboquant_utils.rotation import generate_rotation_matrix, hadamard_rotate_inverse
from turboquant_utils.triton_kernels import triton_fused_matmul_grouped


def _parse_dtype_str(dtype_str) -> torch.dtype:
    """把 "torch.bfloat16" 形式的字符串解析回 torch.dtype（与 quantize.py 的解析方式一致）。"""
    return getattr(torch, str(dtype_str).split(".")[-1])


class WxA16Linear(nn.Module):
    """
    WxA16 量化 Linear 层。

    存储 packed 权重而非 fp16，推理时先反量化再进行 GEMM。

    参数形状:
      - packed_indices: (out_features, packed_in_features)
      - norms: (out_features, n_groups) 或 (out_features,)
      - codebook: (2^bit_width,)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bit_width: int,
        group_size: int,
        packed_indices: torch.Tensor,
        codebook: torch.Tensor,
        norms: torch.Tensor,
        seed: int,
        rotation: str = "qr",
        bias = None,
        orig_dtype: torch.dtype = torch.float16,
        codebook_type: str = "lloydmax",
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bit_width = bit_width
        self.group_size = group_size
        self.seed = seed
        self.rotation = rotation
        self.orig_dtype = orig_dtype
        # "uniform"（WxA8 attention 用，indices 可直接映射为 int8 权重）
        # 或 "lloydmax"（默认，WxA16 传统码本）
        self.codebook_type = codebook_type

        # 存储量化参数 (注册为 buffers 以便 state_dict 保存)
        self.register_buffer("packed_indices", packed_indices)
        self.register_buffer("norms", norms)
        self.register_buffer("codebook", codebook)

        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

        # 计算分组数
        self.n_groups = (in_features + group_size - 1) // group_size

        # 预计算掩码
        self._bit_mask = (1 << bit_width) - 1
        self._elements_per_byte = 8 // bit_width

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        bit_width: int,
        group_size = None,
        seed: int = 42,
        rotation: str = "qr",
        keep_on_gpu: bool = True,
        uniform_codebook: bool = False,
    ) -> "WxA16Linear":
        """
        从 nn.Linear 量化转换为 WxA16Linear。

        Args:
            linear: 原始 nn.Linear
            bit_width: 1/2/4/8
            group_size: 分组大小 (默认 in_features)
            seed: 随机种子
            rotation: 旋转类型
            keep_on_gpu: 是否保持在 GPU
            uniform_codebook: 仅 bit_width==8 有效，均匀码本（WxA8 attention 用，
                见 turboquant_quantize_packed_full 的说明）

        Returns:
            WxA16Linear 实例
        """
        if group_size is None:
            group_size = linear.in_features

        # 执行 packed 量化
        packed_data = turboquant_quantize_packed_full(
            linear.weight.data,
            bit_width=bit_width,
            group_size=group_size,
            seed=seed,
            rotation=rotation,
            keep_on_gpu=keep_on_gpu,
            uniform_codebook=uniform_codebook,
        )

        # 处理 bias
        bias = linear.bias.data.clone() if linear.bias is not None else None

        # 创建实例
        return cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bit_width=bit_width,
            group_size=group_size,
            packed_indices=packed_data["indices_packed"],
            codebook=packed_data["codebook"],
            norms=packed_data["norms"],
            seed=seed,
            rotation=rotation,
            bias=bias,
            orig_dtype=linear.weight.dtype,
            codebook_type=packed_data["codebook_type"],
        )

    @classmethod
    def from_metadata(cls, meta: dict) -> "WxA16Linear":
        """
        从保存的元数据字典重建 WxA16Linear（checkpoint 加载路径）。

        只注册占位 buffer 并恢复普通属性，实际数据由 load_state_dict(assign=True) 回填。
        不修改现有量化路径（from_linear 不变）。
        """
        return cls(
            in_features=meta["in_features"],
            out_features=meta["out_features"],
            bit_width=meta["bit_width"],
            group_size=meta["group_size"],
            packed_indices=torch.empty(meta["indices_packed_shape"], dtype=torch.uint8),
            codebook=torch.empty(meta["codebook_shape"], dtype=torch.float16),
            norms=torch.empty(meta["norms_shape"], dtype=torch.float16),
            seed=meta["seed"],
            rotation=meta["rotation"],
            bias=torch.empty(meta["bias_shape"], dtype=torch.float16) if meta.get("has_bias", False) else None,
            orig_dtype=_parse_dtype_str(meta["orig_dtype"]),
            codebook_type=meta.get("codebook_type", "lloydmax"),
        )

    @torch.no_grad()
    def dequantize(self) -> torch.Tensor:
        """
        反量化回完整权重矩阵。

        Returns:
            W_approx: (out_features, in_features)
        """
        device = self.packed_indices.device
        codebook = self.codebook.to(device)

        M, N = self.out_features, self.in_features

        # 解包索引
        full_indices = unpack_nbit(self.packed_indices, self.bit_width, N)

        # 反量化
        W_approx = torch.zeros((M, N), dtype=torch.float32, device=device)

        # 确保 norms 是二维的
        norms = self.norms
        if norms.dim() == 1:
            norms = norms.unsqueeze(1)

        group_idx = 0
        for g_start in range(0, N, self.group_size):
            g_end = min(g_start + self.group_size, N)
            g_dim = g_end - g_start

            indices_g = full_indices[:, g_start:g_end]
            norms_g = norms[:, group_idx].unsqueeze(1)
            group_idx += 1

            # 从码本还原（float：码本存 fp16，但旋转矩阵 Pi 是 fp32，
            # 直接 fp16 @ fp32 会 dtype 不匹配；反量化本就不走 kernel，用 fp32 更准）
            Y_quant_scaled = codebook[indices_g].float()

            # 逆缩放
            scale = math.sqrt(g_dim)
            Y_unscaled = Y_quant_scaled / scale

            # 逆旋转
            if self.rotation == "none":
                W_g_approx = Y_unscaled
            elif self.rotation == "hadamard":
                W_g_approx = hadamard_rotate_inverse(Y_unscaled, seed=self.seed + g_start)
            else:  # qr
                Pi = generate_rotation_matrix(g_dim, seed=self.seed + g_start).to(device)
                W_g_approx = Y_unscaled @ Pi

            # 恢复原始尺度
            W_approx[:, g_start:g_end] = W_g_approx * norms_g

        return W_approx.to(self.orig_dtype)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向推理：使用 Triton Fused Kernel。

        Args:
            x: (batch_size, seq_len, in_features) 或 (batch_size, in_features)

        Returns:
            out: (batch_size, seq_len, out_features) 或 (batch_size, out_features)
        """
        orig_shape = x.shape
        batch_size = None
        seq_len = None

        # Reshape if needed
        if x.dim() == 3:
            batch_size, seq_len, in_features = x.shape
            x = x.reshape(batch_size * seq_len, in_features)

        # Ensure codebook is on the correct device
        codebook = self.codebook.to(x.device)

        # Use Triton fused kernel
        out = triton_fused_matmul_grouped(
            x, self.packed_indices, codebook, self.norms,
            self.seed, self.group_size, self.in_features, self.bit_width
        )

        # Add bias
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        # Reshape back if needed
        if batch_size is not None and seq_len is not None:
            out = out.reshape(batch_size, seq_len, self.out_features)

        return out.to(x.dtype)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bit_width={self.bit_width}, group_size={self.group_size}, "
                f"rotation={self.rotation}")

    def get_memory_usage(self) -> dict:
        """
        获取当前 WxA16Linear 的存储空间统计。

        Returns:
            dict with:
                - packed_bytes: packed 权重的字节数
                - metadata_bytes: 元数据的字节数 (norms, codebook, bias)
                - orig_fp16_bytes: 原始 fp16 权重的字节数 (用于对比)
                - compression_ratio: 压缩比
        """
        # packed_indices 的字节数
        packed_bytes = self.packed_indices.numel() * self.packed_indices.element_size()

        # norms 的字节数
        norms_bytes = self.norms.numel() * self.norms.element_size()

        # codebook 的字节数
        codebook_bytes = self.codebook.numel() * self.codebook.element_size()

        # bias 的字节数
        bias_bytes = 0
        if self.bias is not None:
            bias_bytes = self.bias.numel() * self.bias.element_size()

        metadata_bytes = norms_bytes + codebook_bytes + bias_bytes

        # 原始 fp16 的字节数 (权重 + bias)
        orig_weight_bytes = self.in_features * self.out_features * 2  # fp16 = 2 bytes
        orig_bias_bytes = self.out_features * 2 if self.bias is not None else 0
        orig_fp16_bytes = orig_weight_bytes + orig_bias_bytes

        # 压缩比
        total_bytes = packed_bytes + metadata_bytes
        compression_ratio = orig_fp16_bytes / total_bytes if total_bytes > 0 else float('inf')

        return {
            'packed_bytes': packed_bytes,
            'metadata_bytes': metadata_bytes,
            'total_bytes': total_bytes,
            'orig_fp16_bytes': orig_fp16_bytes,
            'compression_ratio': compression_ratio,
            'bit_width': self.bit_width,
        }


class W8A16Linear(WxA16Linear):
    """
    专用 W8A16 版本，对 8-bit 有潜在优化。

    默认 uniform_codebook=True：均匀码本让 indices 可以直接映射为 int8 权重
    （w_i8 = idx - 128），WxA8 attention 路径的 kernel 免查表。
    旧 checkpoint（Lloyd-Max 码本）加载不受影响 —— 加载路径走 from_metadata，
    codebook_type 从 meta.json 恢复；转换 WxA8 时会按码本类型判定。
    """

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size = None,
        seed: int = 42,
        rotation: str = "qr",
        keep_on_gpu: bool = True,
        uniform_codebook: bool = True,
    ) -> "W8A16Linear":
        return super().from_linear(
            linear,
            bit_width=8,
            group_size=group_size,
            seed=seed,
            rotation=rotation,
            keep_on_gpu=keep_on_gpu,
            uniform_codebook=uniform_codebook,
        )
