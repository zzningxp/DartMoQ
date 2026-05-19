"""Core TurboQuant quantization: single-pass rotation + Lloyd-Max scalar quantization.

Provides both simulation (returns fp32/bf16 approximation) and packed storage
(returns 4-bit packed indices + norms + codebook).

TurboQuant 是一种近最优的权重量化方法，结合了：
1. 随机正交旋转（QR分解或Hadamard变换）
2. Lloyd-Max标量量化（均匀量化码本）
3. 在线反量化，无需存储完整量化权重
"""

from __future__ import annotations

import math
from typing import Optional

import torch

# 导入码本生成模块
from turboquant_model.codebook import get_codebook
# 导入旋转矩阵生成和Hadamard变换模块
from turboquant_model.rotation import (
    generate_rotation_matrix,
    hadamard_rotate,
    hadamard_rotate_inverse,
)


# ---------------------------------------------------------------------------
# 4-bit 打包/解包工具函数
# 用于将量化索引高效存储：每字节存储2个4-bit索引
# ---------------------------------------------------------------------------


def pack_4bit(indices: torch.Tensor) -> torch.Tensor:
    """将4-bit索引(0-15)打包到uint8张量中，每字节存储2个索引。

    存储布局: byte = lo_nibble | (hi_nibble << 4)
    其中 lo = indices[..., 2i], hi = indices[..., 2i+1]

    Args:
        indices: 整数张量，形状为 (..., N)，值范围 [0, 15]，N 必须为偶数

    Returns:
        packed: uint8张量，形状为 (..., N//2)
    """
    # 确保最后一维长度为偶数，便于打包
    assert indices.shape[-1] % 2 == 0, "Last dim must be even for 4-bit packing"
    # 提取偶数位置的索引作为低4位
    lo = indices[..., 0::2].to(torch.uint8)
    # 提取奇数位置的索引作为高4位
    hi = indices[..., 1::2].to(torch.uint8)
    # 组合成uint8：低4位 | (高4位 << 4)
    return lo | (hi << 4)


def unpack_4bit(packed: torch.Tensor, N: int) -> torch.Tensor:
    """将uint8打包数据解包为4-bit索引张量。

    Args:
        packed: uint8张量，形状为 (..., N//2)
        N: 原始最后一维的长度

    Returns:
        indices: int32张量，形状为 (..., N)
    """
    # 提取低4位（掩码 0x0F）
    lo = (packed & 0x0F).to(torch.int32)
    # 右移4位后提取高4位
    hi = ((packed >> 4) & 0x0F).to(torch.int32)
    # 在最后一维堆叠 lo 和 hi
    result = torch.stack([lo, hi], dim=-1)
    # 重塑为原始形状
    return result.reshape(*packed.shape[:-1], N)


# ---------------------------------------------------------------------------
# 单遍量化（模拟模式）
# 返回反量化后的浮点近似值，用于评估量化效果
# ---------------------------------------------------------------------------


@torch.no_grad()
def turboquant_quantize(
    W: torch.Tensor,
    bit_width: int = 4,
    group_size: Optional[int] = None,
    seed: int = 42,
    rotation: str = "qr",
) -> torch.Tensor:
    """应用 TurboQuant 量化并返回反量化后的近似权重矩阵。

    TurboQuant 量化流程：
      1. 行归一化（Row-normalize）：将每行除以其L2范数
      2. 随机正交旋转：使用QR分解或Hadamard变换
      3. Lloyd-Max标量量化：使用均匀码本进行量化
      4. 反量化：查找码本中心点，逆旋转，恢复原始尺度

    Args:
        W: 权重矩阵，形状为 (out_features, in_features)
        bit_width: 每个坐标的量化位数（默认4-bit）
        group_size: 沿in_features维度的分组大小（None表示整行）
        seed: 旋转矩阵生成的随机种子
        rotation: 旋转类型，"qr" 或 "hadamard"

    Returns:
        W_approx: 量化后的近似权重矩阵，形状和 dtype 与输入相同
    """
    # 保存原始数据类型，用于最终输出
    orig_dtype = W.dtype
    # 转换为float32进行量化计算
    W = W.float()
    out_features, in_features = W.shape

    # 获取Lloyd-Max码本：centroids为量化中心点，boundaries为量化区间边界
    centroids, boundaries = get_codebook(bit_width)
    centroids = centroids.to(W.device)
    boundaries = boundaries.to(W.device)

    # 默认分组大小为整个输入特征维度
    if group_size is None:
        group_size = in_features

    # 初始化输出近似矩阵
    W_approx = torch.zeros_like(W)

    # 按分组遍历输入特征维度
    for g_start in range(0, in_features, group_size):
        g_end = min(g_start + group_size, in_features)
        g_dim = g_end - g_start
        W_g = W[:, g_start:g_end]  # 当前分组的权重子矩阵

        # Step 1: 行归一化
        # 计算每行的L2范数，防止除零
        norms = W_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_norm = W_g / norms

        # Step 2: 随机正交旋转
        if rotation == "hadamard":
            # 使用Hadamard变换进行旋转（计算更快）
            Y = hadamard_rotate(W_norm, seed=seed + g_start)
        else:
            # 使用QR分解生成随机正交矩阵
            Pi = generate_rotation_matrix(g_dim, seed=seed + g_start).to(W.device)
            Y = W_norm @ Pi.T

        # 归一化旋转后的方差
        scale = math.sqrt(g_dim)
        Y_scaled = Y * scale

        # Step 3: Lloyd-Max标量量化
        # 使用二分查找确定每个元素对应的码本索引
        indices = torch.searchsorted(boundaries, Y_scaled.reshape(-1))
        # 限制索引范围在有效码本范围内
        indices = indices.clamp(0, len(centroids) - 1)
        # 通过索引查找码本中心点，得到量化后的值
        Y_quant = centroids[indices].reshape(Y_scaled.shape)

        # Step 4: 反量化
        # 逆缩放
        Y_unscaled = Y_quant / scale
        # 逆旋转恢复原始空间
        if rotation == "hadamard":
            W_g_approx = hadamard_rotate_inverse(Y_unscaled, seed=seed + g_start)
        else:
            W_g_approx = Y_unscaled @ Pi
        # 恢复原始尺度
        W_approx[:, g_start:g_end] = W_g_approx * norms

    # 转换回原始数据类型并返回
    return W_approx.to(orig_dtype)


# ---------------------------------------------------------------------------
# 单遍量化（打包存储模式）
# 返回紧凑的量化表示，用于实际存储和推理部署
# ---------------------------------------------------------------------------


@torch.no_grad()
def turboquant_quantize_packed(
    W: torch.Tensor,
    bit_width: int = 4,
    group_size: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """量化权重矩阵并返回打包表示，用于存储和推理。

    与模拟模式不同，此函数不返回反量化后的权重，而是返回：
    - 打包的4-bit索引
    - 码本（量化中心点）
    - 归一化范数
    - 旋转种子和分组信息

    Args:
        W: 权重矩阵，形状为 (M, N)，M=out_features, N=in_features
        bit_width: 每个元素的量化位数（目前打包格式仅支持4-bit）
        group_size: 分组大小（None表示整行）
        seed: 旋转矩阵生成的随机种子

    Returns:
        dict 包含以下键值对:
            indices_packed: uint8张量，形状为 (M, N//2)，存储打包的4-bit索引
            codebook: float32张量，形状为 (2^bit_width,)，量化码本
            norms: float32张量，形状为 (M,) 或 (M, n_groups)，行范数
            seed: int，旋转种子
            group_size: int，分组大小
            shape: tuple (M, N)，原始权重矩阵形状
            bit_width: int，量化位数
    """
    # 当前打包格式仅支持4-bit量化
    assert bit_width == 4, "Packed format supports 4-bit only"

    M, N = W.shape
    # 默认分组大小为整个输入特征维度
    if group_size is None:
        group_size = N

    # 转换为float32进行量化计算
    W = W.float()
    # 获取Lloyd-Max码本
    centroids, boundaries = get_codebook(bit_width)
    centroids = centroids.to(W.device)
    boundaries = boundaries.to(W.device)

    # 收集每个分组的范数和索引
    all_norms = []
    all_indices = []

    # 按分组遍历输入特征维度
    for g_start in range(0, N, group_size):
        g_end = min(g_start + group_size, N)
        g_dim = g_end - g_start
        W_g = W[:, g_start:g_end]  # 当前分组的权重子矩阵

        # 行归一化
        norms = W_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_norm = W_g / norms
        all_norms.append(norms.squeeze(1))  # 保存范数（用于反量化）

        # 生成随机正交旋转矩阵并应用
        Pi = generate_rotation_matrix(g_dim, seed=seed + g_start).to(W.device)
        Y = W_norm @ Pi.T

        # 方差归一化缩放
        scale = math.sqrt(g_dim)
        Y_scaled = Y * scale

        # 标量量化：查找每个元素对应的码本索引
        indices = torch.searchsorted(boundaries, Y_scaled.reshape(-1))
        indices = indices.clamp(0, len(centroids) - 1).reshape(M, g_dim)
        all_indices.append(indices)

    # 拼接所有分组的索引和范数
    full_indices = torch.cat(all_indices, dim=1)
    norms_out = torch.stack(all_norms, dim=1) if len(all_norms) > 1 else all_norms[0]

    # 如果N为奇数，填充一个零索引以确保能被2整除（便于4-bit打包）
    if N % 2 != 0:
        full_indices = torch.nn.functional.pad(full_indices, (0, 1), value=0)

    # 将索引打包为4-bit格式
    packed = pack_4bit(full_indices)

    # 返回打包的量化表示（转移到CPU以便存储）
    return {
        "indices_packed": packed,
        "codebook": centroids.cpu(),
        "norms": norms_out.cpu(),
        "seed": seed,
        "group_size": group_size,
        "shape": (M, N),
        "bit_width": bit_width,
    }
