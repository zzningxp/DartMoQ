"""cuTile 融合反量化 + 矩阵乘法内核，用于 TurboQuant 在线推理。

基于 NVIDIA cuTile Python (cuda-tile) 重新实现的优化版本，相比 Triton 基准版本
有多项性能优化。当 cuTile 不可用时会优雅降级。

相比 Triton 的优化点：
  1. 自动调优 — cuda.tile_experimental.autotune_launch 针对每个问题形状搜索
     最优 tile 大小 (TB, TN, TK)，首次运行后缓存结果。
  2. 共享内存码本 — 16个条目的码本(64字节)全程驻留 L1/寄存器，ct.gather
     提供隐式缓存。
  3. FP16/BF16 + Tensor Core — ct.mma 利用 Tensor Core 加速半精度输入，
     达到峰值 TFLOPS。
  4. TF32 Tensor Core — 对于 FP32 输入，使用 TF32(10位尾数)，在 Ampere/Ada/
     Blackwell 架构上实现约2倍吞吐量。
  5. 预缩放范数 — norms / sqrt(K) 在主机端计算一次，避免内核内的逐元素除法。
  6. 预取机制 — cuTile 的基于 tile 的加载模型自动优化内存流水线。
  7. 消除转置 — 以自然 (B, N) 布局累加并直接存储，避免输出的任何转置操作。

主要入口函数：
  cutile_fused_matmul            — 静态 tile 大小（快速，无调优开销）
  cutile_fused_matmul_autotuned  — 搜索最优 tile（首次调用较慢）
  cutile_fused_dual_matmul       — 双通路融合（用于残差量化）

支持的 GPU：Ampere (sm80), Ada (sm89), Blackwell (sm100+)
依赖：NVIDIA Driver r580+ 和 CUDA Toolkit 13.1+
"""

from __future__ import annotations

import math
from math import ceil

import torch

# ------------------------------
# cuTile 导入和可用性检查
# ------------------------------
try:
    import cuda.tile as ct

    # 常量整数类型，用于 cuTile 内核参数
    ConstInt = ct.Constant[int]
    _CUTILE_AVAILABLE = True
except ImportError:
    _CUTILE_AVAILABLE = False

# 可选的实验性自动调优器
_HAS_AUTOTUNE = False
if _CUTILE_AVAILABLE:
    try:
        from cuda.tile_experimental import autotune_launch

        _HAS_AUTOTUNE = True
    except ImportError:
        pass


def _next_power_of_2(n: int) -> int:
    """向上取整到下一个2的幂次。

    cuTile 内核要求 tile 大小必须是2的幂次，此函数用于计算满足要求的最小 tile 大小。

    Args:
        n: 输入整数

    Returns:
        大于等于n的最小2的幂次
    """
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


# ---------------------------------------------------------------------------
# cuTile 内核 — 仅在 cuda.tile 可用时定义
# ---------------------------------------------------------------------------

if _CUTILE_AVAILABLE:

    @ct.kernel
    def _turboquant_cutile_kernel(
        input_ptr,       # Array: (B, K) 预旋转后的激活值
        indices_ptr,     # Array: (N, PACKED_K) 打包的 uint8 索引
        codebook_ptr,    # Array: (N_LEVELS,) 码本中心点（float32）
        norms_ptr,       # Array: (N,) 预缩放后的范数（float32）
        output_ptr,      # Array: (B, N) 输出结果
        TB: ConstInt,    # batch 维度的 tile 大小
        TN: ConstInt,    # output (N) 维度的 tile 大小
        TK: ConstInt,    # reduction (K) 维度的 tile 大小，必须为偶数
    ):
        """融合 4-bit 解包 → 码本查找 → 矩阵乘法 → 范数缩放内核。

        每个 CTA (Cooperative Thread Array) 计算一个 (TB, TN) 的输出 tile，
        通过以 TK 为块大小迭代 K 维度来完成计算。打包的 uint8 索引被 gather，
        提取半字节(nibble)，在码本中查找对应值，然后送入 ct.mma 进行 Tensor Core 加速。

        整体计算流程：
        1. 加载输入 tile: (B, K) 的一部分
        2. 解包 4-bit 索引 → 转换为码本索引
        3. 码本查找 → 获取量化权重值
        4. Tensor Core 矩阵乘法累加
        5. 乘以预缩放的范数
        6. 存储输出
        """
        # 获取 block 索引：bid_b 对应 batch 维度，bid_n 对应 N 维度
        bid_b = ct.bid(0)
        bid_n = ct.bid(1)

        # 计算 K 维度需要的 tile 数量
        num_k_tiles = ct.num_tiles(input_ptr, axis=1, shape=(TB, TK))
        # 初始化累加器为0
        acc = ct.full((TB, TN), 0, dtype=ct.float32)
        # 零填充模式
        zero_pad = ct.PaddingMode.ZERO

        # 选择 MMA 数据类型：FP32 输入使用 TF32（Tensor Core加速），否则使用原生类型
        mma_dtype = (
            ct.tfloat32 if input_ptr.dtype == ct.float32 else input_ptr.dtype
        )

        # 当前 N-tile 的行索引（在 K 迭代中保持不变）
        rn = bid_n * TN + ct.arange(TN, dtype=ct.int32)

        # 遍历 K 维度的所有 tile
        for k_tile in range(num_k_tiles):
            k_start = k_tile * TK

            # ========== 加载输入 tile: (TB, TK) ==========
            inp = ct.load(
                input_ptr,
                index=(bid_b, k_tile),
                shape=(TB, TK),
                padding_mode=zero_pad,
            )

            # ========== 解包 4-bit 权重索引 ==========
            rk = ct.arange(TK, dtype=ct.int32)
            k_global = k_start + rk
            byte_col = k_global // 2            # 打包数组中的字节列
            is_high = (k_global % 2) == 1       # 是否为高半字节的标志

            # Gather 打包的字节：(TN, TK) — 每个字节只为其半字节获取
            packed = ct.gather(
                indices_ptr, (rn[:, None], byte_col[None, :])
            )

            # 提取每个 K 位置对应的半字节
            lo = ct.bitwise_and(packed, 0x0F)           # 低4位
            hi = ct.bitwise_and(ct.bitwise_rshift(packed, 4), 0x0F)  # 高4位
            idx = ct.where(is_high[None, :], hi, lo).astype(ct.int32)

            # ========== 码本查找：codebook[idx] → (TN, TK) ==========
            w_tile = ct.gather(codebook_ptr, idx)

            # ========== Tensor Core MMA: (TB, TK) @ (TK, TN) → (TB, TN) ==========
            a = inp.astype(mma_dtype)
            b = ct.transpose(w_tile).astype(mma_dtype)
            acc = ct.mma(a, b, acc)

        # ========== 乘以预缩放的范数 ==========
        norms = ct.load(
            norms_ptr, index=(bid_n,), shape=(TN,), padding_mode=zero_pad
        )
        acc = acc * norms[None, :]

        # ========== 存储结果 ==========
        ct.store(
            output_ptr,
            index=(bid_b, bid_n),
            tile=ct.astype(acc, output_ptr.dtype),
        )


# ---------------------------------------------------------------------------
# Python 包装函数 — 静态 tile 大小
# ---------------------------------------------------------------------------


def cutile_fused_matmul(
    x_rot: torch.Tensor,          # (B, K) 预旋转后的输入
    indices_packed: torch.Tensor,  # (N, K//2) 打包的 uint8 索引
    codebook: torch.Tensor,        # (n_levels,) 码本中心点
    norms: torch.Tensor,           # (N,) 每行权重的范数
    K: int,                        # 该分组的维度（in_features 或 group_size）
    scale: float | None = None,
) -> torch.Tensor:
    """通过 cuTile 实现融合反量化 + 矩阵乘法。

    ``triton_fused_matmul()`` 的直接替换版本。
    在主机端预缩放范数，并选择2的幂次作为 tile 大小。

    Args:
        x_rot: (B, K) 预旋转后的激活值
        indices_packed: (N, K//2) 打包的4-bit权重索引
        codebook: 码本中心点
        norms: 每行权重的范数 (N,)
        K: 该分组的维度（in_features 或 group_size）
        scale: 范数除数（默认: sqrt(K)）

    Returns:
        output: (B, N) 的矩阵乘法结果
    """
    # 检查 cuTile 是否可用
    if not _CUTILE_AVAILABLE:
        raise RuntimeError(
            "cuda-tile is not installed. Install with: pip install cuda-tile[tileiras]"
        )

    B = x_rot.shape[0]
    N = indices_packed.shape[0]

    # 默认缩放因子为 sqrt(K)
    if scale is None:
        scale = math.sqrt(K)

    # 在主机端预缩放范数（避免内核中的逐元素除法）
    norms_scaled = norms / scale

    # 分配输出张量
    output = torch.empty(B, N, dtype=torch.float32, device=x_rot.device)

    # 选择2的幂次的 tile 大小，上限为问题维度
    TB = min(32, _next_power_of_2(B))
    TN = min(64, _next_power_of_2(N))
    TK = min(64, _next_power_of_2(K))
    TK = max(TK, 2)  # 最小为2，用于打包半字节对齐

    # 计算 grid 大小
    grid = (ceil(B / TB), ceil(N / TN), 1)
    # 启动 cuTile 内核
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _turboquant_cutile_kernel,
        (x_rot, indices_packed, codebook, norms_scaled, output, TB, TN, TK),
    )
    return output


# ---------------------------------------------------------------------------
# 自动调优版本（使用 cuda.tile_experimental，如可用）
# ---------------------------------------------------------------------------


def cutile_fused_matmul_autotuned(
    x_rot: torch.Tensor,
    indices_packed: torch.Tensor,
    codebook: torch.Tensor,
    norms: torch.Tensor,
    K: int,
    scale: float | None = None,
) -> torch.Tensor:
    """自动调优的 cuTile 融合反量化 + 矩阵乘法。

    使用 ``cuda.tile_experimental.autotune_launch`` 搜索最佳吞吐量的 tile 大小
    {TB, TN, TK}。结果按 (形状, 数据类型) 缓存，因此每个唯一的问题形状只需搜索一次。

    当自动调优包不可用时，回退到 ``cutile_fused_matmul()``。

    参数 / 返回值: 与 ``cutile_fused_matmul()`` 相同。
    """
    # 如果自动调优不可用，回退到静态 tile 版本
    if not _HAS_AUTOTUNE:
        return cutile_fused_matmul(
            x_rot, indices_packed, codebook, norms, K, scale
        )

    B = x_rot.shape[0]
    N = indices_packed.shape[0]

    if scale is None:
        scale = math.sqrt(K)
    norms_scaled = norms / scale

    # 分配最终输出张量
    output = torch.empty(B, N, dtype=torch.float32, device=x_rot.device)

    # ========== 构建搜索空间 ==========
    def search_space():
        """生成所有可能的 (TB, TN, TK) 配置组合。"""
        configs = []
        max_tb = min(32, _next_power_of_2(B))
        for tb in [1, 2, 4, 8, 16, 32]:
            if tb > max_tb:
                continue
            for tn in [16, 32, 64, 128]:
                if tn > N:
                    continue
                for tk in [16, 32, 64, 128]:
                    if tk > K or tk < 2:
                        continue
                    configs.append((tb, tn, tk))
        return configs

    def grid_fn(cfg):
        """根据配置计算 grid 大小。"""
        tb, tn, _tk = cfg
        return (ceil(B / tb), ceil(N / tn), 1)

    def args_fn(cfg):
        """为每个调优试验生成参数（使用临时输出缓冲区避免覆盖）。"""
        tb, tn, tk = cfg
        tmp = torch.empty(B, N, dtype=torch.float32, device=x_rot.device)
        return (x_rot, indices_packed, codebook, norms_scaled, tmp, tb, tn, tk)

    def launch_args_fn(cfg):
        """最终启动时写入真实输出缓冲区。"""
        tb, tn, tk = cfg
        return (
            x_rot, indices_packed, codebook, norms_scaled, output, tb, tn, tk
        )

    # 启动自动调优
    autotune_launch(
        torch.cuda.current_stream(),
        grid_fn,
        _turboquant_cutile_kernel,
        args_fn,
        launch_args_fn=launch_args_fn,
        search_space=search_space,
        max_iter=30,      # 最大调优迭代次数
        seed=42,          # 随机种子
    )

    return output


# ---------------------------------------------------------------------------
# 双通路融合内核：一次启动处理两个残差通路
# ---------------------------------------------------------------------------

if _CUTILE_AVAILABLE:

    @ct.kernel
    def _turboquant_cutile_dual_kernel(
        # Pass 1 参数
        input1_ptr,       # Array: (B, K) 第一个通路的输入
        indices1_ptr,     # Array: (N, PACKED_K) 打包的 uint8 索引
        codebook1_ptr,    # Array: (N_LEVELS,) 码本中心点
        norms1_ptr,       # Array: (N,) 预缩放后的范数
        # Pass 2 参数
        input2_ptr,       # Array: (B, K) 第二个通路的输入
        indices2_ptr,     # Array: (N, PACKED_K) 打包的 uint8 索引
        codebook2_ptr,    # Array: (N_LEVELS,) 码本中心点
        norms2_ptr,       # Array: (N,) 预缩放后的范数
        # 输出
        output_ptr,       # Array: (B, N) 输出结果
        TB: ConstInt,     # batch 维度 tile 大小
        TN: ConstInt,     # output 维度 tile 大小
        TK: ConstInt,     # reduction 维度 tile 大小（必须为偶数）
        SAME_INPUT: ConstInt,  # 1 表示 input1==input2（共享旋转）
    ):
        """双通路融合 4-bit 解包 → 码本查找 → 矩阵乘法 → 结果合并。

        每个 CTA 计算一个 (TB, TN) 的输出 tile，同时处理两个通路，
        最终输出 = acc1*norms1 + acc2*norms2，只需一次存储操作。

        用于残差量化场景，其中需要对同一输入应用两个量化权重矩阵。
        """
        bid_b = ct.bid(0)
        bid_n = ct.bid(1)

        num_k_tiles = ct.num_tiles(input1_ptr, axis=1, shape=(TB, TK))
        # 为两个通路分别初始化累加器
        acc1 = ct.full((TB, TN), 0, dtype=ct.float32)
        acc2 = ct.full((TB, TN), 0, dtype=ct.float32)
        zero_pad = ct.PaddingMode.ZERO

        mma_dtype = (
            ct.tfloat32 if input1_ptr.dtype == ct.float32 else input1_ptr.dtype
        )

        rn = bid_n * TN + ct.arange(TN, dtype=ct.int32)

        for k_tile in range(num_k_tiles):
            k_start = k_tile * TK

            # 共享的索引计算（两个通路相同）
            rk = ct.arange(TK, dtype=ct.int32)
            k_global = k_start + rk
            byte_col = k_global // 2
            is_high = (k_global % 2) == 1

            # ========== 通路 1 ==========
            inp1 = ct.load(
                input1_ptr, index=(bid_b, k_tile),
                shape=(TB, TK), padding_mode=zero_pad,
            )
            packed1 = ct.gather(indices1_ptr, (rn[:, None], byte_col[None, :]))
            lo1 = ct.bitwise_and(packed1, 0x0F)
            hi1 = ct.bitwise_and(ct.bitwise_rshift(packed1, 4), 0x0F)
            idx1 = ct.where(is_high[None, :], hi1, lo1).astype(ct.int32)
            w1 = ct.gather(codebook1_ptr, idx1)
            acc1 = ct.mma(inp1.astype(mma_dtype), ct.transpose(w1).astype(mma_dtype), acc1)

            # ========== 通路 2 ==========
            # 如果两个输入相同（共享旋转），复用已加载的输入
            if SAME_INPUT == 1:
                inp2 = inp1
            else:
                inp2 = ct.load(
                    input2_ptr, index=(bid_b, k_tile),
                    shape=(TB, TK), padding_mode=zero_pad,
                )
            packed2 = ct.gather(indices2_ptr, (rn[:, None], byte_col[None, :]))
            lo2 = ct.bitwise_and(packed2, 0x0F)
            hi2 = ct.bitwise_and(ct.bitwise_rshift(packed2, 4), 0x0F)
            idx2 = ct.where(is_high[None, :], hi2, lo2).astype(ct.int32)
            w2 = ct.gather(codebook2_ptr, idx2)
            acc2 = ct.mma(inp2.astype(mma_dtype), ct.transpose(w2).astype(mma_dtype), acc2)

        # ========== Epilogue: 合并两个通路的结果 ==========
        n1 = ct.load(norms1_ptr, index=(bid_n,), shape=(TN,), padding_mode=zero_pad)
        n2 = ct.load(norms2_ptr, index=(bid_n,), shape=(TN,), padding_mode=zero_pad)
        result = acc1 * n1[None, :] + acc2 * n2[None, :]

        # 存储最终结果
        ct.store(
            output_ptr, index=(bid_b, bid_n),
            tile=ct.astype(result, output_ptr.dtype),
        )


def cutile_fused_dual_matmul(
    x_rot1: torch.Tensor,          # (B, K) 通路1的预旋转输入
    indices1_packed: torch.Tensor,  # (N, K//2) 通路1的打包索引
    codebook1: torch.Tensor,        # 通路1的码本
    norms1: torch.Tensor,           # 通路1的范数
    x_rot2: torch.Tensor,          # (B, K) 通路2的预旋转输入
    indices2_packed: torch.Tensor,  # (N, K//2) 通路2的打包索引
    codebook2: torch.Tensor,        # 通路2的码本
    norms2: torch.Tensor,           # 通路2的范数
    K: int,                        # 分组维度
    scale: float | None = None,
) -> torch.Tensor:
    """双通路融合反量化 + 矩阵乘法 via cuTile。

    等价于两次单通路调用的和：
        cutile_fused_matmul(x_rot1, idx1, cb1, n1, K, scale)
      + cutile_fused_matmul(x_rot2, idx2, cb2, n2, K, scale)

    但只需要一次内核启动、一次输出写入，并且共享索引计算，性能更优。

    主要用于残差量化场景，其中需要对同一输入应用两个量化权重矩阵。

    Args:
        x_rot1: (B, K) 通路1的预旋转激活值
        indices1_packed: (N, K//2) 通路1的打包4-bit索引
        codebook1: 通路1的码本
        norms1: 通路1的范数
        x_rot2: (B, K) 通路2的预旋转激活值
        indices2_packed: (N, K//2) 通路2的打包4-bit索引
        codebook2: 通路2的码本
        norms2: 通路2的范数
        K: 分组维度
        scale: 范数缩放因子（默认 sqrt(K)）

    Returns:
        output: (B, N) 两个通路结果的和
    """
    if not _CUTILE_AVAILABLE:
        raise RuntimeError(
            "cuda-tile is not installed. Install with: pip install cuda-tile[tileiras]"
        )

    B = x_rot1.shape[0]
    N = indices1_packed.shape[0]
    if scale is None:
        scale = math.sqrt(K)

    # 预缩放范数
    norms1_scaled = norms1 / scale
    norms2_scaled = norms2 / scale

    # 检查两个输入是否相同（共享旋转时可以复用输入加载）
    same_input = 1 if x_rot1.data_ptr() == x_rot2.data_ptr() else 0

    output = torch.empty(B, N, dtype=torch.float32, device=x_rot1.device)

    # 选择 tile 大小
    TB = min(32, _next_power_of_2(B))
    TN = min(64, _next_power_of_2(N))
    TK = min(64, _next_power_of_2(K))
    TK = max(TK, 2)

    grid = (ceil(B / TB), ceil(N / TN), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _turboquant_cutile_dual_kernel,
        (
            x_rot1, indices1_packed, codebook1, norms1_scaled,
            x_rot2, indices2_packed, codebook2, norms2_scaled,
            output, TB, TN, TK, same_input,
        ),
    )
    return output