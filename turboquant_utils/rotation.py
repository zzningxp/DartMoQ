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
from functools import lru_cache
from typing import Tuple

import torch

# ---------------------------------------------------------------------------
# 旋转矩阵缓存（LRU缓存）
# 缓存键: (d, seed)，缓存值: rotation_matrix
# 注意：由于 torch.Tensor 不能直接作为 lru_cache 的键值，我们用 CPU 的 ndarray 存储
# ---------------------------------------------------------------------------

_ROTATION_CACHE: dict[Tuple[int, int], torch.Tensor] = {}
_ROTATION_CACHE_DEV: dict[Tuple[int, int, str], torch.Tensor] = {}  # 按 device 缓存派生副本，避免每调用 .to(device) 重拷
_MAX_CACHE_SIZE = 2048  # 最多缓存2048个旋转矩阵

# P4-7: batch 旋转矩阵缓存
# 缓存键: (d, seed_base, num_groups, stride, device_str, dtype_str)
# 缓存值: (num_groups, d, d) 的 batch 旋转矩阵
_BATCH_ROTATION_CACHE: dict[Tuple, torch.Tensor] = {}
_MAX_BATCH_CACHE_SIZE = 512
# 注：down 方向 seed = base + expert_offset + group_offset，每 forward 工作集
# ~1032 个 key（256 experts × 4 groups 等），128 不够会导致每 forward 抖动、
# QR 反复重算。调到 2048 避免抖动，稳态后不再重算。CPU+GPU 各约 66MB 常驻。


# ---------------------------------------------------------------------------
# QR分解法随机旋转（Haar分布）
# 通过对高斯随机矩阵进行QR分解，生成Haar分布的正交矩阵
# ---------------------------------------------------------------------------


def generate_rotation_matrix(d: int, seed: int = 42, device: Optional[torch.device] = None) -> torch.Tensor:
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
        device: 目标设备（用于缓存查找）

    Returns:
        Q: d×d 的正交矩阵，数据类型为 float32
    """
    global _ROTATION_CACHE, _ROTATION_CACHE_DEV

    # 基矩阵（CPU）始终只计算一次，保证与量化侧（不传 device、在 CPU 算 QR）
    # 算出的矩阵值完全一致，避免推理侧在 GPU 上重算 QR 导致反量化静默出错。
    base_key = (d, seed)
    if base_key not in _ROTATION_CACHE:
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

        # 缓存结果（先存 CPU 上，节省显存）
        if len(_ROTATION_CACHE) >= _MAX_CACHE_SIZE:
            # 简单的LRU：移除最早的项
            first_key = next(iter(_ROTATION_CACHE.keys()))
            del _ROTATION_CACHE[first_key]

        _ROTATION_CACHE[base_key] = Q.cpu()

    base = _ROTATION_CACHE[base_key]

    if device is None:
        return base

    # 按 device 缓存派生副本：首次派生一次 .to(device) 后常驻，
    # 后续调用直接返回，消除每前向的 CPU→GPU 重拷（实验 6 拆出的隐藏开销）。
    dev_key = (d, seed, str(device))
    if dev_key in _ROTATION_CACHE_DEV:
        return _ROTATION_CACHE_DEV[dev_key]

    if len(_ROTATION_CACHE_DEV) >= _MAX_CACHE_SIZE:
        first_key = next(iter(_ROTATION_CACHE_DEV.keys()))
        del _ROTATION_CACHE_DEV[first_key]

    Qd = base.to(device)
    _ROTATION_CACHE_DEV[dev_key] = Qd
    return Qd


def clear_rotation_cache() -> None:
    """清空旋转矩阵缓存，释放内存"""
    global _ROTATION_CACHE, _ROTATION_CACHE_DEV, _BATCH_ROTATION_CACHE
    _ROTATION_CACHE.clear()
    _ROTATION_CACHE_DEV.clear()
    _BATCH_ROTATION_CACHE.clear()


def generate_batch_rotation_matrices(
    d: int,
    seed_base: int,
    num_groups: int,
    stride: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """生成 batch 旋转矩阵，形状 (num_groups, d, d)。

    用于 P4-7 旋转优化：将 num_groups 个小 matmul 合并为一次 bmm。
    每个 group 的 seed = seed_base + group_idx * stride。

    结果按 (d, seed_base, num_groups, stride, device, dtype) 缓存。

    Args:
        d: 每个旋转矩阵的维度
        seed_base: 起始 seed
        num_groups: group 数量
        stride: 每个 group 的 seed 增量（通常 = group_size）
        device: 目标设备
        dtype: 目标 dtype

    Returns:
        P_batch: (num_groups, d, d) 的 batch 正交矩阵
    """
    global _BATCH_ROTATION_CACHE

    key = (d, seed_base, num_groups, stride, str(device), str(dtype))
    if key in _BATCH_ROTATION_CACHE:
        return _BATCH_ROTATION_CACHE[key]

    # 逐个生成并堆叠（复用 generate_rotation_matrix 的 L1 缓存）
    mats = []
    for i in range(num_groups):
        seed = seed_base + i * stride
        P = generate_rotation_matrix(d, seed, device=device).to(dtype)
        mats.append(P)

    P_batch = torch.stack(mats, dim=0).contiguous()  # (num_groups, d, d)

    if len(_BATCH_ROTATION_CACHE) >= _MAX_BATCH_CACHE_SIZE:
        first_key = next(iter(_BATCH_ROTATION_CACHE.keys()))
        del _BATCH_ROTATION_CACHE[first_key]

    _BATCH_ROTATION_CACHE[key] = P_batch
    return P_batch


def batch_rotate_input(
    x: torch.Tensor,
    group_size: int,
    seed_base: int,
) -> torch.Tensor:
    """对输入 x 做分组旋转，使用 batch matmul 加速。

    P4-7 优化：把 num_groups 次小 matmul 合并成一次 bmm，
    消除 Python 循环 + 多次 kernel launch 开销。

    Args:
        x: 输入张量 (B, K_total)，K_total 必须是 group_size 的整数倍
        group_size: 每个 group 的大小
        seed_base: 起始 seed（每个 group 的 seed = seed_base + g_start）

    Returns:
        x_rot: 旋转后的张量 (B, K_total)，连续存储
    """
    B, K_total = x.shape
    num_groups = K_total // group_size

    # 获取 batch 旋转矩阵 (num_groups, group_size, group_size)
    P_batch = generate_batch_rotation_matrices(
        group_size, seed_base, num_groups, stride=group_size,
        device=x.device, dtype=x.dtype,
    )

    # reshape x 为 (num_groups, B, group_size) 以便 bmm
    x_groups = x.view(B, num_groups, group_size).transpose(0, 1).contiguous()
    # bmm: (num_groups, B, group_size) @ (num_groups, group_size, group_size).T
    #    = (num_groups, B, group_size)
    x_rot_groups = torch.bmm(x_groups, P_batch.transpose(1, 2))
    # 转回 (B, K_total)
    x_rot = x_rot_groups.transpose(0, 1).contiguous().view(B, K_total)
    return x_rot


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