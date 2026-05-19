"""Lloyd-Max 最优码本计算模块，用于高斯分布的标量量化。

Lloyd-Max 算法是一种迭代优化算法，用于计算给定概率分布下的最优量化码本。
对于 N(0,1) 高斯分布，算法通过交替更新量化区间边界和码本中心点（质心）
来最小化量化误差（均方误差）。

该模块提供：
1. _compute_lloyd_max_gaussian: 计算高斯分布的最优码本
2. get_codebook: 获取预计算的码本（带缓存机制）
"""

from __future__ import annotations

import numpy as np
import torch


def _compute_lloyd_max_gaussian(
    n_levels: int, n_iters: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """计算标准正态分布 N(0,1) 的 Lloyd-Max 最优码本。

    Lloyd-Max 算法迭代过程：
    1. 初始化量化区间边界
    2. 计算每个区间的最优质心（条件期望）
    3. 更新边界为相邻质心的中点
    4. 重复步骤2-3直到收敛

    Args:
        n_levels: 量化级数（码本大小）
        n_iters: 迭代次数（默认200次足够收敛）

    Returns:
        centroids: 形状为 (n_levels,) 的数组，表示各量化级的质心（已排序）
        boundaries: 形状为 (n_levels+1,) 的数组，表示量化区间边界
                    其中 boundaries[0]=-inf, boundaries[-1]=+inf
    """
    from scipy.stats import norm  # 延迟导入，避免不必要的依赖加载

    sigma = 1.0  # 高斯分布标准差

    # 初始化边界：在 [-3.5σ, 3.5σ] 范围内均匀分布
    # 覆盖约99.9%的高斯分布概率质量
    boundaries = np.linspace(-3.5 * sigma, 3.5 * sigma, n_levels + 1)
    boundaries[0] = -1e10  # 左边界设为负无穷
    boundaries[-1] = 1e10  # 右边界设为正无穷
    centroids = np.zeros(n_levels)  # 初始化质心数组

    # Lloyd-Max 迭代优化
    for _ in range(n_iters):
        # Step 1: 更新质心（条件期望）
        # 对于每个区间 [lo, hi]，质心 = E[X | X ∈ [lo, hi]]
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            # 计算区间内的概率质量
            p = norm.cdf(hi) - norm.cdf(lo)
            if p > 1e-15:  # 避免除以零
                # 高斯分布的条件期望公式：(φ(lo) - φ(hi)) / (Φ(hi) - Φ(lo))
                # 其中 φ 是概率密度函数，Φ 是累积分布函数
                centroids[i] = (norm.pdf(lo) - norm.pdf(hi)) / p
            else:
                # 概率质量极小时，使用区间中点作为质心
                centroids[i] = (max(lo, -3.5) + min(hi, 3.5)) / 2

        # Step 2: 更新边界（相邻质心的中点）
        for i in range(1, n_levels):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2

    return centroids, boundaries


# 码本缓存：键为 bit_width，值为 (centroids, boundaries) 元组
# 避免重复计算相同 bit_width 的码本
_CODEBOOK_CACHE: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def get_codebook(bit_width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """获取指定位宽的预计算 Lloyd-Max 码本（带缓存机制）。

    码本一旦计算就会被缓存，后续相同 bit_width 的请求直接返回缓存结果。

    Args:
        bit_width: 每元素的量化位数（如4-bit表示16个量化级）

    Returns:
        centroids: float32张量，形状为 (2^bit_width,)，表示量化质心
        boundaries: float32张量，形状为 (2^bit_width - 1,)，表示内部边界
                    （不含正负无穷边界，便于 torch.searchsorted 使用）
    """
    # 检查缓存，避免重复计算
    if bit_width not in _CODEBOOK_CACHE:
        n_levels = 2**bit_width  # 量化级数 = 2^bit_width
        centroids, boundaries = _compute_lloyd_max_gaussian(n_levels)
        # 缓存结果：转换为 PyTorch float32 张量
        # boundaries[1:-1] 移除正负无穷边界，保留内部边界
        _CODEBOOK_CACHE[bit_width] = (
            torch.tensor(centroids, dtype=torch.float32),
            torch.tensor(boundaries[1:-1], dtype=torch.float32),
        )
    return _CODEBOOK_CACHE[bit_width]
