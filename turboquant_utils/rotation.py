"""TurboQuant 随机旋转矩阵生成模块。

提供两种随机正交旋转方法：

1. QR分解法 ("qr")：
   - 通过对标准高斯矩阵进行QR分解生成Haar分布的正交矩阵
   - 存储复杂度 O(d²)，计算复杂度 O(d²)
   - 适用于任意维度 d

2. Hadamard变换法 ("hadamard")：
   - 通过随机符号翻转 + 快速Walsh-Hadamard变换实现
   - 存储复杂度 O(d)，计算复杂度 O(d log d)
   - 要求维度 d 必须是 2 的幂次

旋转的作用：将权重矩阵投影到一个近似独立的坐标系，使得量化误差更均匀分布，
从而提高量化后的重建精度。
"""

from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# QR分解法随机旋转（Haar分布）
# 通过对高斯随机矩阵进行QR分解，生成Haar分布的正交矩阵
# ---------------------------------------------------------------------------


def generate_rotation_matrix(d: int, seed: int = 42) -> torch.Tensor:
    """通过QR分解生成Haar分布的随机正交矩阵。

    Haar分布的正交矩阵具有以下特性：
    - 任意单位向量经过旋转后均匀分布在单位超球面上
    - 旋转后的坐标近似独立，服从 N(0, 1/d) 分布
    - 保证旋转操作保持向量的L2范数不变

    算法流程：
    1. 生成 d×d 的标准高斯随机矩阵 G
    2. 对 G 进行QR分解：G = Q @ R
    3. 修正符号歧义：Q = Q @ diag(sign(diag(R)))

    Args:
        d: 矩阵维度
        seed: 随机种子，保证结果可复现

    Returns:
        Q: d×d 的正交矩阵，数据类型为 float32
    """
    # 创建指定种子的随机数生成器
    gen = torch.Generator().manual_seed(seed)
    # 生成标准高斯随机矩阵
    G = torch.randn(d, d, generator=gen)
    # QR分解：Q是正交矩阵，R是上三角矩阵
    Q, R = torch.linalg.qr(G)
    # 修正符号歧义：QR分解中Q的列符号不唯一
    # 通过乘以R对角线元素的符号，确保Q服从Haar分布
    diag_sign = torch.sign(torch.diag(R))
    Q = Q * diag_sign.unsqueeze(0)
    return Q


# ---------------------------------------------------------------------------
# Hadamard变换 + 随机符号旋转
# 通过随机符号翻转和快速Walsh-Hadamard变换实现高效旋转
# 特点：O(d)存储，O(d log d)计算，但要求d必须是2的幂次
# ---------------------------------------------------------------------------


def _generate_signs(d: int, seed: int) -> torch.Tensor:
    """生成d个随机 ±1 符号向量。

    Args:
        d: 向量维度
        seed: 随机种子

    Returns:
        signs: 形状为 (d,) 的张量，元素值为 ±1
    """
    gen = torch.Generator().manual_seed(seed)
    # 先生成0/1整数，再转换为-1/+1
    return torch.randint(0, 2, (d,), generator=gen).float() * 2 - 1


def _fwht(X: torch.Tensor) -> torch.Tensor:
    """对最后一维执行未归一化的快速Walsh-Hadamard变换（FWHT）。

    Walsh-Hadamard变换是一种正交变换，具有以下性质：
    - 变换矩阵是正交的（H^T = H^{-1}
    - 变换后的能量保持不变
    - 计算复杂度为 O(d log d)

    Args:
        X: 输入张量，形状为 (..., d)，d必须是2的幂次

    Returns:
        变换后的张量，形状与输入相同
    """
    d = X.shape[-1]
    h = 1  # 当前处理的块大小
    while h < d:
        # 重塑张量，使距离为h的元素相邻
        # 形状变为 (..., num_blocks, 2, block_size)
        X = X.view(*X.shape[:-1], d // (2 * h), 2, h)
        a = X[..., 0, :]  # 前半部分
        b = X[..., 1, :]  # 后半部分
        # Walsh-Hadamard核操作：[a+b, a-b]
        X = torch.stack([a + b, a - b], dim=-2)
        # 恢复原始形状
        X = X.view(*X.shape[:-3], d)
        h *= 2  # 块大小翻倍
    return X


def hadamard_rotate(X: torch.Tensor, seed: int) -> torch.Tensor:
    """正向Hadamard旋转：等价于 X @ Pi^T，其中 Pi = H @ D / sqrt(d)。

    H是Walsh-Hadamard矩阵，D是随机对角符号矩阵。
    由于FWHT本身是自逆的，正向和反向变换形式相似。

    步骤：
    1. 每行乘以随机 ±1 符号
    2. 应用归一化的FWHT

    Args:
        X: 输入张量，形状为 (..., d)，d必须是2的幂次
        seed: 符号向量的随机种子

    Returns:
        旋转后的张量，形状与输入相同
    """
    d = X.shape[-1]
    # 生成随机符号向量并移动到相同设备和数据类型
    signs = _generate_signs(d, seed).to(X.device, X.dtype)
    # 符号翻转 + FWHT + 归一化
    return _fwht(X * signs) / math.sqrt(d)


def hadamard_rotate_inverse(Y: torch.Tensor, seed: int) -> torch.Tensor:
    """反向Hadamard旋转：等价于 Y @ Pi，其中 Pi = H @ D / sqrt(d)。

    由于FWHT是自逆的，反向变换只需调整操作顺序：
    1. 应用归一化的FWHT
    2. 乘以相同的随机符号

    Args:
        Y: 输入张量，形状为 (..., d)，d必须是2的幂次
        seed: 随机种子（必须与正向变换相同）

    Returns:
        逆旋转后的张量，形状与输入相同
    """
    d = Y.shape[-1]
    # 使用相同种子生成符号向量
    signs = _generate_signs(d, seed).to(Y.device, Y.dtype)
    # FWHT + 归一化 + 符号翻转
    return _fwht(Y) / math.sqrt(d) * signs