#!/usr/bin/env python3
"""WxA8 融合 kernel：INT8 激活 × 混合 bit 权重 → INT32 累加 → FP16 输出。

与 triton_kernels.py 里的 WxA16 gf kernel 是同一套数据流，只改三处：

  1. 输入 x 是 int8，额外带一个 per-token-per-group 的激活 scale
  2. 权重码本是 int8，tl.dot 走 int8 → int32（IMMA）
  3. group 内用 int32 累加，出 group 时才乘 (激活 scale × 权重 scale) 转 fp32

之所以能这样原位替换，是因为 WxA16 kernel 本来就把 per-group 的权重 scale
放在内层 dot 之外（`total_acc += acc_g * norm_g`），per-token-per-group 的
激活 scale 塞进同一个 epilogue 即可，数据流结构完全不变。

—— 与 WxA16 的契约差异（重要）——
WxA8 kernel **不做旋转**。分组 QR 旋转和激活量化都是调用方的责任，
因为量化必须发生在旋转之后（旋转改变每个 group 的幅度分布），而
gate_up 和 down 用的是不同的旋转（down 的 seed 还带 expert 偏移），
不存在"在 MoE 入口一次量化两边共用"的可能。
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


# ===========================================================================
# 融合 kernel（group-first 布局）
# ===========================================================================

@triton.jit
def _wxa8_fused_matmul_kernel_grouped_gf(
    # Input
    x_ptr,            # (B, K_total) int8    —— 已旋转并量化
    xs_ptr,           # (B, NUM_GROUPS) fp32 —— per-token per-group 激活 scale
    # Quantized weight (group-first 布局)
    indices_ptr,      # (NUM_GROUPS_TOTAL, N, PACKED_PER_GROUP) uint8
    codebook_ptr,     # (n_levels,) int8
    norms_ptr,        # (NUM_GROUPS_TOTAL, N) fp16 —— 与 WxA16 的 norms_gf 同源
                      #   （1/sqrt(gs) 已预乘；码本的 int8 步长 cb_step 折在激活 scale 里）
    # Output
    output_ptr,       # (B, N)
    # Shape
    B, N,
    K_total,              # 总 K = num_groups * group_size
    INDICES_G0_STRIDE,    # indices 第 0 维 stride = N * PACKED_PER_GROUP
    NORMS_G0_STRIDE,      # norms 第 0 维 stride = N
    XS_ROW_STRIDE,        # xs 第 0 维 stride = NUM_GROUPS_TOTAL
    # Constexpr config
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BIT_WIDTH: tl.constexpr,
    N_LEVELS: tl.constexpr,
    IDENTITY_CB: tl.constexpr = False,
    BLOCK_B: tl.constexpr = 256,
    BLOCK_N: tl.constexpr = 32,
    BLOCK_K: tl.constexpr = 128,
):
    """WxA8 multi-group fused dequant + INT8 matmul kernel (group-first 布局)。

    注: indices_ptr / norms_ptr 指向切片后的起始位置（第 0 个待处理 group 的
    起点），kernel 内 g 从 0 到 NUM_GROUPS-1，通过 *_G0_STRIDE 算每个 group 基址。
    xs_ptr 则指向完整的 (B, NUM_GROUPS_TOTAL) 起点 —— down 路径下 x 已经是
    切片后的输入，其 scale 也是按切片后的 group 数排布的，故两者一致。

    IDENTITY_CB（仅 BIT_WIDTH==8）：均匀码本下 indices 直接就是 int8 权重
    （w_i8 = idx - 128），免查表 —— attention 路径用。此时 codebook_ptr 不会被
    访问（调用方仍须传一个合法指针，可传占位张量）。
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_b = rb < B
    mask_n = rn < N

    total_acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    ELEMENTS_PER_BYTE = 8 // BIT_WIDTH
    PACKED_PER_GROUP = GROUP_SIZE // ELEMENTS_PER_BYTE

    # 预计算行内偏移基（每个 group 内行 n 的基址）
    row_base = rn * PACKED_PER_GROUP  # (BLOCK_N,)

    for g in range(NUM_GROUPS):
        g_start = g * GROUP_SIZE
        g_base = g * INDICES_G0_STRIDE

        # 本 group 的权重 scale (BLOCK_N,) 与激活 scale (BLOCK_B,)
        norm_g = tl.load(norms_ptr + g * NORMS_G0_STRIDE + rn, mask=mask_n, other=0.0)
        xs_g = tl.load(xs_ptr + rb * XS_ROW_STRIDE + g, mask=mask_b, other=0.0)

        acc_i = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.int32)

        for k_start in range(0, GROUP_SIZE, BLOCK_K):
            rk = k_start + tl.arange(0, BLOCK_K)
            mask_k = rk < GROUP_SIZE

            # 激活 tile：int8，访存量是 fp16 路径的一半
            inp_off = rb[:, None] * K_total + (g_start + rk)[None, :]
            x_tile = tl.load(x_ptr + inp_off,
                             mask=mask_b[:, None] & mask_k[None, :], other=0)

            w_mask = mask_n[:, None] & mask_k[None, :]
            if BIT_WIDTH == 8:
                byte_off = g_base + row_base[:, None] + rk[None, :]
                idx = tl.load(indices_ptr + byte_off, mask=w_mask, other=0).to(tl.int32)
            else:
                BIT_MASK = (1 << BIT_WIDTH) - 1
                byte_col = rk // ELEMENTS_PER_BYTE
                pos_in_byte = rk % ELEMENTS_PER_BYTE
                byte_off = g_base + row_base[:, None] + byte_col[None, :]
                packed = tl.load(indices_ptr + byte_off, mask=w_mask, other=0).to(tl.uint8)
                shift = pos_in_byte * BIT_WIDTH
                idx = ((packed >> shift[None, :]) & BIT_MASK).to(tl.int32)

            if IDENTITY_CB:
                # 均匀码本：w_i8 = idx - 128（indices 就是 int8 权重，免查表）
                w_i8 = (idx - 128).to(tl.int8)
            else:
                w_i8 = tl.load(codebook_ptr + idx, mask=w_mask, other=0)

            acc_i += tl.dot(x_tile, tl.trans(w_i8), out_dtype=tl.int32)

        # 出 group：int32 → fp32，乘 (激活 scale × 权重 scale)
        total_acc += acc_i.to(tl.float32) * xs_g[:, None] * norm_g[None, :]

    tl.store(
        output_ptr + rb[:, None] * N + rn[None, :],
        total_acc.to(output_ptr.dtype.element_ty),
        mask=mask_b[:, None] & mask_n[None, :],
    )


# ===========================================================================
# P5-1: gate_up + silu×up 融合 kernel（group-first 布局）
# ===========================================================================
# 与 _wxa8_fused_matmul_kernel_grouped_gf 结构几乎相同，区别：
#   - 输出列数 = inter_size（N_out），但权重有 2*N_out 列（gate + up）
#   - 内部同时算 gate 和 up 两个累加器
#   - epilogue: out = silu(gate_out) * up_out
# 收益：省一次完整 gate_up 输出的读写（4MB/expert × 256 = 1GB/层）
#       + 省一次独立的 silu×up launch
# ===========================================================================

@triton.jit
def _wxa8_gate_up_silu_kernel_gf(
    # Input
    x_ptr,            # (B, K_total) int8    —— 已旋转并量化
    xs_ptr,           # (B, NUM_GROUPS) fp32 —— per-token per-group 激活 scale
    # Quantized weight (group-first 布局) —— 列数 = 2 * N_out（gate 在前，up 在后）
    indices_ptr,      # (NUM_GROUPS_TOTAL, 2*N_out, PACKED_PER_GROUP) uint8
    codebook_ptr,     # (n_levels,) int8
    norms_ptr,        # (NUM_GROUPS_TOTAL, 2*N_out) fp16
    # Output
    output_ptr,       # (B, N_out) fp16 —— silu(gate) * up
    # Shape
    B, N_OUT,         # N_OUT = inter_size（输出列数），权重列数 = 2*N_OUT
    K_total,              # 总 K = num_groups * group_size
    INDICES_G0_STRIDE,    # indices 第 0 维 stride = 2*N_out * PACKED_PER_GROUP
    NORMS_G0_STRIDE,      # norms 第 0 维 stride = 2*N_out
    XS_ROW_STRIDE,        # xs 第 0 维 stride = NUM_GROUPS_TOTAL
    # Constexpr config
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BIT_WIDTH: tl.constexpr,
    N_LEVELS: tl.constexpr,
    IDENTITY_CB: tl.constexpr = False,
    BLOCK_B: tl.constexpr = 256,
    BLOCK_N: tl.constexpr = 32,
    BLOCK_K: tl.constexpr = 128,
):
    """WxA8 gate_up + silu×up 融合 kernel（group-first 布局）。

    与独立 gate_up kernel + 独立 silu 等价，但中间结果不落地。
    权重布局不变（仍是 2*inter_size 列，gate 在前 up 在后），
    输出只有 inter_size 列（= silu(gate) * up）。
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # 输出列坐标（< N_OUT）
    mask_b = rb < B
    mask_n = rn < N_OUT

    gate_acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    ELEMENTS_PER_BYTE = 8 // BIT_WIDTH
    PACKED_PER_GROUP = GROUP_SIZE // ELEMENTS_PER_BYTE

    # gate 列和 up 列的权重基址（同一行号 n 在 gate 和 up 各有一列）
    row_base_gate = rn * PACKED_PER_GROUP          # gate 侧：列 n
    row_base_up = (rn + N_OUT) * PACKED_PER_GROUP  # up 侧：列 n + N_OUT

    for g in range(NUM_GROUPS):
        g_start = g * GROUP_SIZE
        g_base = g * INDICES_G0_STRIDE

        # 本 group 的 norms：gate 侧和 up 侧各取 BLOCK_N 列
        norm_gate = tl.load(norms_ptr + g * NORMS_G0_STRIDE + rn,
                            mask=mask_n, other=0.0)
        norm_up = tl.load(norms_ptr + g * NORMS_G0_STRIDE + rn + N_OUT,
                          mask=mask_n, other=0.0)
        xs_g = tl.load(xs_ptr + rb * XS_ROW_STRIDE + g, mask=mask_b, other=0.0)

        acc_gate_i = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.int32)
        acc_up_i = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.int32)

        for k_start in range(0, GROUP_SIZE, BLOCK_K):
            rk = k_start + tl.arange(0, BLOCK_K)
            mask_k = rk < GROUP_SIZE

            # 激活 tile（gate 和 up 共用同一份输入）
            inp_off = rb[:, None] * K_total + (g_start + rk)[None, :]
            x_tile = tl.load(x_ptr + inp_off,
                             mask=mask_b[:, None] & mask_k[None, :], other=0)

            w_mask = mask_n[:, None] & mask_k[None, :]
            if BIT_WIDTH == 8:
                # ---- gate 权重 ----
                byte_off_gate = g_base + row_base_gate[:, None] + rk[None, :]
                idx_gate = tl.load(indices_ptr + byte_off_gate, mask=w_mask, other=0).to(tl.int32)
                # ---- up 权重 ----
                byte_off_up = g_base + row_base_up[:, None] + rk[None, :]
                idx_up = tl.load(indices_ptr + byte_off_up, mask=w_mask, other=0).to(tl.int32)
            else:
                BIT_MASK = (1 << BIT_WIDTH) - 1
                byte_col = rk // ELEMENTS_PER_BYTE
                pos_in_byte = rk % ELEMENTS_PER_BYTE
                # ---- gate 权重 ----
                byte_off_gate = g_base + row_base_gate[:, None] + byte_col[None, :]
                packed_gate = tl.load(indices_ptr + byte_off_gate, mask=w_mask, other=0).to(tl.uint8)
                shift = pos_in_byte * BIT_WIDTH
                idx_gate = ((packed_gate >> shift[None, :]) & BIT_MASK).to(tl.int32)
                # ---- up 权重 ----
                byte_off_up = g_base + row_base_up[:, None] + byte_col[None, :]
                packed_up = tl.load(indices_ptr + byte_off_up, mask=w_mask, other=0).to(tl.uint8)
                idx_up = ((packed_up >> shift[None, :]) & BIT_MASK).to(tl.int32)

            if IDENTITY_CB:
                w_gate = (idx_gate - 128).to(tl.int8)
                w_up = (idx_up - 128).to(tl.int8)
            else:
                w_gate = tl.load(codebook_ptr + idx_gate, mask=w_mask, other=0)
                w_up = tl.load(codebook_ptr + idx_up, mask=w_mask, other=0)

            acc_gate_i += tl.dot(x_tile, tl.trans(w_gate), out_dtype=tl.int32)
            acc_up_i += tl.dot(x_tile, tl.trans(w_up), out_dtype=tl.int32)

        # 出 group：int32 → fp32，乘 scale
        gate_acc += acc_gate_i.to(tl.float32) * xs_g[:, None] * norm_gate[None, :]
        up_acc += acc_up_i.to(tl.float32) * xs_g[:, None] * norm_up[None, :]

    # Epilogue: out = silu(gate) * up
    out = gate_acc * tl.sigmoid(gate_acc) * up_acc

    tl.store(
        output_ptr + rb[:, None] * N_OUT + rn[None, :],
        out.to(output_ptr.dtype.element_ty),
        mask=mask_b[:, None] & mask_n[None, :],
    )


# ===========================================================================
# INT8 tile 配置表
#
# 必须与 WxA16 分开调：INT8 下每元素 1 字节、累加器是 int32，寄存器与
# shared 压力和 fp16 路径完全不同，实测最优 tile 差别很大 ——
# gate_up 沿用 WxA16 的 (64,128,32,w4,s4) 只有 1.24x，
# 换成 (256,32,128,w4,s2) 是 1.56x。
#
# 调优脚本: test/test_wxa8_tune.py（RTX 5090 / group_size=128 / gf 布局，
#           212s 全扫，搜索空间 BB∈{64,128,256} × BN∈{32,64,128} ×
#           BK∈{32,64,128} × warps∈{4,8} × stages∈{2,3,4}）
#
# 规律（与 WxA16 的最优点明显不同，值得记一笔）:
#   - BLOCK_K=128 全场景最优（= group_size，内层循环只跑一次）
#   - BLOCK_N=32 全场景最优：INT8 权重 tile 加载便宜，N 切窄换来更大的
#     BLOCK_B，反而更划算（WxA16 的 large 档是 BLOCK_N=128）
#   - large 档 BLOCK_B=256 / warps=4 / stages=2；small 档 BLOCK_B=64 / warps=8
#
# 格式: (BLOCK_B, BLOCK_N, BLOCK_K, num_warps, num_stages)
# 注释里的 us / relerr 是调优时实测值（relerr 对 fp32 参考解）
# ===========================================================================

# gate_up 方向（行切片，N=1024 K=2048）
_WXA8_CONFIG_GATE_UP = {
    "small": {   # B <= 256，B=128 下搜索
        1: (64, 32, 128, 8, 3),    # 20.7us  relerr=0.00649
        2: (64, 32, 128, 8, 4),    # 23.3us  relerr=0.00649
        4: (64, 32, 128, 8, 3),    # 29.2us  relerr=0.00861
    },
    "large": {   # B > 256，B=2048 下搜索（真实 eval 主力场景）
        1: (256, 32, 128, 4, 2),   # 40.8us  relerr=0.00647
        2: (256, 32, 128, 4, 2),   # 44.9us  relerr=0.00651
        4: (256, 32, 128, 4, 2),   # 59.4us  relerr=0.00860
    },
}

# down 方向（in_features 切片，N=2048 K=512）
_WXA8_CONFIG_DOWN = {
    "small": {
        1: (64, 32, 128, 8, 3),    #  9.8us  relerr=0.00650
        2: (64, 32, 128, 8, 3),    # 10.3us  relerr=0.00652
        4: (64, 32, 128, 8, 2),    # 12.2us  relerr=0.00860
    },
    "large": {
        1: (128, 32, 128, 4, 2),   # 25.8us  relerr=0.00647
        2: (256, 32, 128, 4, 2),   # 28.5us  relerr=0.00650
        4: (256, 32, 128, 4, 2),   # 33.0us  relerr=0.00860
    },
}

# attention 方向（完整矩阵无切片，8-bit 均匀码本 IDENTITY_CB 免查表）
# 扫描: /tmp/sweep_attn.py（B=16384, qkv N=4096 K=2048 / o N=2048 K=4096）
_WXA8_CONFIG_ATTN = {
    "small": {
        8: (64, 128, 128, 8, 3),   # 未单独扫描，按 large 档缩小 BLOCK_B 保守取值
    },
    "large": {
        8: (128, 128, 128, 8, 3),  # 649.6us / 659.6us, ~420 TOP/s（两形状一致）
    },
}

# B 自适应阈值，与 WxA16 保持一致（triton_kernels.py:_B_THRESHOLD_SMALL）
_B_THRESHOLD_SMALL = 256

# ---------------------------------------------------------------------------
# 静态派发（CUDA graph decode）专用配置：_wxa8_merged_kernel_gf 在 qwen3
# decode 形状（hidden=2048, inter=768, gs=128, capacity=T*top_k=8，worst-case
# 调度表 + GATHER_X + fp16 norms）下的最优 tile。与上面的 eager 表分开存：
# 两边 kernel 不同（merged vs 逐 expert）、形状不同，混用会互相回退。
# 搜索：test/test_wxa8_static_tune.py（bit2=43/4=9/1=9 成员取自 qwen3-1bpw
# 代表层；bitwise 自检保证换配置不改数值）。bit4 gate_up 用 BB=32、其余
# BB=16；down 统一 BB=16，BN 按 bit 分 128/32。
_WXA8_STATIC_CONFIG_GATE_UP = {   # capacity=8（B1 decode, T×top_k）下搜索
    2: (16, 32, 128, 4, 3),   # 34.9us（现行 small 档 49.9us，1.43x）
    4: (32, 32, 128, 4, 2),   # 50.6us（65.1us，1.29x）
    1: (16, 32, 128, 4, 3),   # 27.6us（41.3us，1.50x）
}
_WXA8_STATIC_CONFIG_DOWN = {      # 同上，down 方向（K=768 N=2048）
    2: (16, 128, 128, 4, 4),   # 18.5us（24.6us，1.33x）
    4: (16, 128, 128, 4, 4),   # 24.7us（32.4us，1.31x）
    1: (16, 32, 128, 4, 4),    # 15.1us（20.5us，1.36x）
}


def get_wxa8_static_config(bit_width: int, direction: str = "gate_up",
                           B: int | None = None):
    """静态派发池的 tile 配置（merged kernel, graph decode 形状）。

    表在 capacity=8（B1 decode）下搜索；其它 capacity（如 B4 的 32）暂
    沿用同表（同为 skinny-B 区间，比 eager small 档更贴近），未覆盖的
    bit 回退 get_wxa8_config。
    """
    table = (_WXA8_STATIC_CONFIG_DOWN if direction == "down"
             else _WXA8_STATIC_CONFIG_GATE_UP)
    return table.get(bit_width) or get_wxa8_config(bit_width, direction, B)

# 调优表未覆盖时的默认（取观测到的通用最优形状：BLOCK_K=128 一次吃掉整个
# group，BLOCK_N=32 切窄）。注意 bit=8 尚未调优 —— 8-bit 权重不 packing，
# 权重 tile 是 2-bit 的 4 倍大，最优点大概率不同，走 attention 路径前要补扫。
_WXA8_DEFAULT_CONFIG = (64, 32, 128, 4, 2)


def get_wxa8_config(bit_width: int, direction: str = "gate_up", B: int | None = None):
    """取指定 bit-width / 方向 / B 档位的最优 INT8 tile 配置。

    direction: "gate_up" / "down"（MoE 切片路径）/ "attn"（完整矩阵）
    """
    if direction == "attn":
        table = _WXA8_CONFIG_ATTN
    elif direction == "down":
        table = _WXA8_CONFIG_DOWN
    else:
        table = _WXA8_CONFIG_GATE_UP
    size_key = "small" if (B is not None and B <= _B_THRESHOLD_SMALL) else "large"
    return table[size_key].get(bit_width, _WXA8_DEFAULT_CONFIG)


# ---------------------------------------------------------------------------
# B 自适应分派：小 B 时 WxA8 不划算 —— 但实测结论是"不划算区间的绝对损失
# 远小于分派成本"，MoE 主路径**无条件走 A8**，这个判据保留给独立调用场景
# ---------------------------------------------------------------------------
# 实测盈亏交叉点（test/test_wxa8_kernel.py --crossover, bit=2, RTX 5090）：
#     B      gate_up   down
#      8     0.89x (+2.6us)   0.92x (+0.7us)   ← A8 亏，但绝对量很小
#     64     0.96x (+1.0us)   1.00x (+0.0us)
#    128     1.00x            1.04x            ← 交叉点
#    256     1.05x            1.07x
#   2048     1.60x (-26.6us)  1.38x (-10.5us)  ← 真实主力场景
# 小 B 下 kernel 远没吃满（B=128 时填充仅 37.6%，0.38 波），INT8 省下的
# 访存与 MMA 换不回多加载两组 scale 的成本。但为小 B 维护 fp16/int8 双缓冲
# （每 bit 约 128MB）或逐 expert 重算旋转（额外 kernel launch ~5us）都比
# 直接吃掉那 ≤2.8us 更亏，所以 MoE 模块无条件走 A8（见 WxA8BitPartitionedGroupMoE）。
WXA8_MIN_B = 256


def wxa8_is_profitable(B: int) -> bool:
    """该 batch 规模下 WxA8 是否比 WxA16 快。

    仅给"独立调用 WxA8 kernel、且本来就同时持有 fp16 与 int8 两份激活"
    的场景做分派参考。MoE 主路径不受此判据约束。
    """
    return B >= WXA8_MIN_B


# ===========================================================================
# 激活量化 / 权重码本转 INT8
# ===========================================================================

def quantize_act_per_token_group(x_rot: torch.Tensor, group_size: int,
                                 extra_scale: float = 1.0):
    """per-token per-group 对称量化（torch 参考实现）。

    x_rot 必须是**已经做过分组旋转**的激活。

    为什么是 per-group 而不是 per-token 整行：激活进 kernel 前已按 group
    分别做过独立的 QR 旋转，各 group 的幅度不同；per-group scale 让每个
    group 都吃满 int8 的 127 级。而现有 kernel epilogue 本来就逐 group
    乘 norm，加一个逐 group 的激活 scale 不增加结构复杂度。

    实测（test/test_wxa8_dot_spike.py）：2-bit MoE 下相对 fp32 参考的误差
    0.0065，其中几乎全部来自这一步（高斯下 per-group int8 的理论误差 0.66%）。

    Args:
        x_rot: (B, K) 已旋转的激活，K 必须是 group_size 的整数倍
        group_size: 分组大小
        extra_scale: 额外折进激活 scale 的常量，通常传 `build_int8_codebook`
            返回的 `cb_step`。这样权重侧的 `norms_gf` 可以**原封不动**沿用
            WxA16 的 fp16 buffer —— 既省掉一份 per-layer 的 fp32 副本
            （真实规模下 40 层约 1GB），也避免把 norms 乘小到 fp16 次正规区。

    Returns:
        x_i8:    (B, K) int8，contiguous
        x_scale: (B, K // group_size) fp32，contiguous（已含 extra_scale）
    """
    B, K = x_rot.shape
    if K % group_size != 0:
        raise ValueError(f"K ({K}) 必须是 group_size ({group_size}) 的整数倍")
    G = K // group_size

    xg = x_rot.reshape(B, G, group_size).float()
    amax = xg.abs().amax(dim=-1)                       # (B, G)
    scale = (amax / 127.0).clamp(min=1e-8)
    q = (xg / scale.unsqueeze(-1)).round().clamp_(-127, 127).to(torch.int8)
    if extra_scale != 1.0:
        scale = scale * extra_scale
    return q.reshape(B, K).contiguous(), scale.contiguous()


# ===========================================================================
# rotate + quantize 融合 kernel
# ===========================================================================

@triton.jit
def _rotate_quantize_kernel(
    x_ptr,            # (B, K_total) 未旋转激活
    rot_ptr,          # (NUM_GROUPS, G0_STRIDE) 每 group 一个旋转矩阵，
                      # 布局与 rotation.generate_batch_rotation_matrices 一致：
                      # 第 g 个矩阵的基址 = g * G0_STRIDE，行 stride = GROUP_SIZE
    x_i8_ptr,         # (B, K_total) int8 输出
    x_s_ptr,          # (B, NUM_GROUPS) fp16 输出 scale（已折入 extra_scale）
    B, K_total,
    ROT_G0_STRIDE,    # rot_ptr 第 0 维 stride
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EXTRA_SCALE: tl.constexpr,    # 折进 scale 的常量（cb_step）
    BLOCK_B: tl.constexpr = 32,
):
    """分组旋转 + per-token-per-group 对称量化的融合 kernel。

    与两段式（batch_rotate_input bmm + quantize_act_per_token_group）数学等价：
      x_rot[b, g, j] = Σ_k x[b, g, k] * P_g[k, j]
      scale[b, g]    = max_j |x_rot[b, g, j]| / 127 * EXTRA_SCALE
      x_i8[b, g, j]  = round(x_rot[b, g, j] / scale[b, g])
    差别只在浮点求和的顺序，误差在 int8 取整粒度之内。

    为什么两段式不划算：bmm 先把 (B, K) fp16 旋转结果整个写到显存再读回来
    量化（fp32 中间态还要再放大 2 倍）。真实规模下旋转本身已带宽受限
    （backlog:255），融合后 x 只读写各一趟，中间结果全部留在寄存器里。
    """
    pid_g = tl.program_id(0)
    pid_b = tl.program_id(1)
    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = rb < B

    # x 块: (BLOCK_B, GROUP_SIZE) —— 每个 program 处理一个 (group, row-block)
    x_off = rb[:, None] * K_total + pid_g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    x_tile = tl.load(x_ptr + x_off, mask=mask_b[:, None], other=0.0)  # fp16

    # 旋转: acc = x_tile @ P_g^T（对齐 batch_rotate_input 的 x @ P_batch^T 约定：
    # acc[b, j] = Σ_k x[b, k] * P_g[j, k]）
    acc = tl.zeros((BLOCK_B, GROUP_SIZE), dtype=tl.float32)
    rj = tl.arange(0, GROUP_SIZE)
    rk = tl.arange(0, GROUP_SIZE)
    # P_g 以 (j 行, k 列) 布局加载，dot 里转置回来
    p_off = pid_g * ROT_G0_STRIDE + rj[:, None] * GROUP_SIZE + rk[None, :]
    p_tile = tl.load(rot_ptr + p_off)                          # (G, G)
    acc += tl.dot(x_tile, tl.trans(p_tile), out_dtype=tl.float32)

    # per-row max → scale。注意：EXTRA_SCALE 只折进**存出去的** scale；
    # 量化除法必须用未折的 amax/127，否则 q = acc/scale 会偏小 cb_step 倍。
    amax = tl.max(tl.abs(acc), axis=1)                        # (BLOCK_B,)
    scale_unf = tl.where(amax < 1e-8 * 127.0, 1e-8, amax / 127.0)

    q = tl.extra.cuda.libdevice.round(acc / scale_unf[:, None])
    q_i8 = tl.clamp(q, -127, 127).to(tl.int8)
    tl.store(x_i8_ptr + x_off, q_i8, mask=mask_b[:, None])
    tl.store(x_s_ptr + rb * NUM_GROUPS + pid_g,
             (scale_unf * EXTRA_SCALE).to(tl.float16), mask=mask_b)


def rotate_quantize_fused(x: torch.Tensor, rot: torch.Tensor,
                          group_size: int, num_groups: int,
                          extra_scale: float = 1.0):
    """分组旋转 + 量化（融合 kernel 的 python wrapper）。

    Args:
        x: (B, K_total) 未旋转激活，K_total = num_groups * group_size
        rot: (num_groups, group_size, group_size) 每 group 一个旋转矩阵，
             与 generate_batch_rotation_matrices 的返回布局一致
        group_size / num_groups: 分组参数
        extra_scale: 折进 scale 的常量（cb_step）

    Returns:
        x_i8:    (B, K_total) int8
        x_scale: (B, num_groups) fp16
    """
    B, K_total = x.shape
    if K_total != num_groups * group_size:
        raise ValueError(f"K_total ({K_total}) != num_groups*group_size "
                         f"({num_groups * group_size})")
    if rot.shape != (num_groups, group_size, group_size):
        raise ValueError(f"rot 形状应为 {(num_groups, group_size, group_size)}, "
                         f"得到 {tuple(rot.shape)}")

    x_i8 = torch.empty((B, K_total), dtype=torch.int8, device=x.device)
    x_scale = torch.empty((B, num_groups), dtype=torch.float16, device=x.device)

    # BLOCK_B 固定 32：tl.dot 要求 M >= 16，极小 B（长尾 expert）不能缩块，
    # 靠 mask 丢掉多余行 —— 极小 expert 的绝对耗时本来就忽略不计。
    BLOCK_B = 32
    grid = (num_groups, triton.cdiv(B, BLOCK_B))
    _rotate_quantize_kernel[grid](
        x, rot, x_i8, x_scale, B, K_total, rot.stride(0),
        GROUP_SIZE=group_size, NUM_GROUPS=num_groups,
        EXTRA_SCALE=extra_scale, BLOCK_B=BLOCK_B,
        num_warps=4, num_stages=3,
    )
    return x_i8, x_scale


@triton.jit
def _rotate_quantize_kernel_pooled(
    x_ptr,            # (B_cap, K_total) fp16 —— 池缓冲（行序 = 成员序）
    rot_ptr,          # 旋转矩阵库（TOTAL, G0_STRIDE），第 k 个矩阵基址 = k*stride
    expert_info_ptr,  # (E, 4) int32 [token_start, token_count, rot_base, g_count]
    block_info_ptr,   # (num_blocks, 2) int32 [member_idx, block_b]
    x_i8_ptr,         # (B_cap, K_total) int8 输出
    x_s_ptr,          # (B_cap, NUM_GROUPS) fp16 输出 scale
    K_total,
    ROT_G0_STRIDE,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    EXTRA_SCALE: tl.constexpr,
    BLOCK_B: tl.constexpr = 32,
):
    """池内逐成员旋转 + 量化（静态派发 down 方向）。

    与 `_rotate_quantize_kernel` 同一数学、同一量化尾；差别只在旋转矩阵
    的来源：每个 expert 的 down 旋转 seed 含 expert 偏移，整批不能共用
    一个矩阵。池缓冲的行序是成员连续的（专家序），所以这里按"成员 ×
    行块"的最坏情况静态调度表分发程序，行段与矩阵基序号（rot_base =
    start // group_size）由 kernel 运行时从 expert_info 读——capture 期
    无 host 读、无 H2D。token_count=0 的成员整块早退（最坏情况表里空
    成员不产生访存）。

    expert_info 的列语义在 down 方向与 GEMM 完全一致（slice_start 即
    start//group_size），因此与静态 GEMM 共用同一份 expert_info 缓冲。
    """
    pid = tl.program_id(0)
    pid_g = tl.program_id(1)
    bi = block_info_ptr + pid * 2
    midx = tl.load(bi + 0).to(tl.int32)
    blk_b = tl.load(bi + 1).to(tl.int32)

    ei = expert_info_ptr + midx * 4
    token_start = tl.load(ei + 0).to(tl.int32)
    token_count = tl.load(ei + 1).to(tl.int32)
    rot_base = tl.load(ei + 2).to(tl.int32)
    if token_count <= 0:
        return

    rb = token_start + blk_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = rb < token_start + token_count

    x_off = rb[:, None] * K_total + pid_g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    x_tile = tl.load(x_ptr + x_off, mask=mask_b[:, None], other=0.0)

    acc = tl.zeros((BLOCK_B, GROUP_SIZE), dtype=tl.float32)
    rj = tl.arange(0, GROUP_SIZE)
    rk = tl.arange(0, GROUP_SIZE)
    p_off = (rot_base + pid_g) * ROT_G0_STRIDE + rj[:, None] * GROUP_SIZE + rk[None, :]
    p_tile = tl.load(rot_ptr + p_off)
    acc += tl.dot(x_tile, tl.trans(p_tile), out_dtype=tl.float32)

    amax = tl.max(tl.abs(acc), axis=1)
    scale_unf = tl.where(amax < 1e-8 * 127.0, 1e-8, amax / 127.0)

    q = tl.extra.cuda.libdevice.round(acc / scale_unf[:, None])
    q_i8 = tl.clamp(q, -127, 127).to(tl.int8)
    tl.store(x_i8_ptr + x_off, q_i8, mask=mask_b[:, None])
    tl.store(x_s_ptr + rb * NUM_GROUPS + pid_g,
             (scale_unf * EXTRA_SCALE).to(tl.float16), mask=mask_b)


def rotate_quantize_fused_pooled(x: torch.Tensor, rot_bank: torch.Tensor,
                                 expert_info: torch.Tensor,
                                 block_info: torch.Tensor,
                                 group_size: int, num_groups: int,
                                 extra_scale: float = 1.0):
    """池内逐成员旋转 + INT8 量化的 python wrapper（静态派发用）。

    Args:
        x: (B_cap, K_total) 池缓冲（行序 = 成员序，padding 行内容任意）
        rot_bank: 旋转矩阵库（total_groups, gs, gs），第 k 个矩阵对应
            seed = down_seed + k*group_size（与逐 expert 的批量生成同序）
        expert_info: (E, 4) int32 设备张量，列 2 = 旋转矩阵基序号
        block_info: (num_blocks, 2) int32 最坏情况调度表
        extra_scale: 折进激活 scale 的常量（cb_step）

    Returns:
        x_i8: (B_cap, K_total) int8；x_scale: (B_cap, num_groups) fp16
    """
    B_cap, K_total = x.shape
    if K_total != num_groups * group_size:
        raise ValueError(f"K_total ({K_total}) != num_groups*group_size "
                         f"({num_groups * group_size})")
    x_i8 = torch.empty((B_cap, K_total), dtype=torch.int8, device=x.device)
    x_scale = torch.empty((B_cap, num_groups), dtype=torch.float16,
                          device=x.device)
    grid = (block_info.shape[0], num_groups)
    _rotate_quantize_kernel_pooled[grid](
        x, rot_bank, expert_info, block_info, x_i8, x_scale,
        K_total, rot_bank.stride(0),
        GROUP_SIZE=group_size, NUM_GROUPS=num_groups,
        EXTRA_SCALE=extra_scale, BLOCK_B=32,
        num_warps=4, num_stages=3,
    )
    return x_i8, x_scale


@triton.jit
def _silu_mul_mask_kernel(
    gu_ptr,        # (C, 2*inter) fp16 —— 静态池 gate_up 输出（gate|up 拼接）
    mask_ptr,      # (C,) bool —— 行有效掩码（行号 < pool_total）
    out_ptr,       # (C, inter) fp16
    INTER,
    stride_gu, stride_out,
    BLOCK: tl.constexpr,
):
    """gate_up 尾部融合：silu(gate)*up*rowmask 一个 kernel 写出 fp16。

    替代 F.silu(gu[:, :inter]) * gu[:, inter:] 再乘 mask 的 3 个 PyTorch
    elementwise（P0 kernel 数量优化）。silu/mul 在 fp32 内完成、单次舍入，
    精度不低于原三段链（原链 silu 后、mul 后各舍入一次 fp16）。
    """
    pid = tl.program_id(0)
    n_col = tl.cdiv(INTER, BLOCK)
    row = pid // n_col
    col = (pid % n_col) * BLOCK + tl.arange(0, BLOCK)
    cm = col < INTER
    gate = tl.load(gu_ptr + row * stride_gu + col,
                   mask=cm, other=0.0).to(tl.float32)
    up = tl.load(gu_ptr + row * stride_gu + INTER + col,
                 mask=cm, other=0.0).to(tl.float32)
    act = gate * tl.sigmoid(gate) * up
    keep = tl.load(mask_ptr + row)
    act = tl.where(keep != 0, act, 0.0)
    tl.store(out_ptr + row * stride_out + col,
             act.to(tl.float16), mask=cm)


def silu_mul_mask_fused(gu: torch.Tensor, mask_rows: torch.Tensor,
                        inter: int):
    """_silu_mul_mask_kernel 的 python wrapper（静态池 A 融合）。

    gu: (C, 2*inter) fp16（wxa8 merged kernel 输出恒 fp16）
    mask_rows: (C,) bool；返回 (C, inter) fp16（padding 行恒零）。
    """
    C = gu.shape[0]
    out = torch.empty((C, inter), dtype=torch.float16, device=gu.device)
    BLOCK = 256
    grid = (C * triton.cdiv(inter, BLOCK),)
    _silu_mul_mask_kernel[grid](
        gu, mask_rows, out, inter, gu.stride(0), out.stride(0),
        BLOCK=BLOCK, num_warps=4,
    )
    return out


@triton.jit
def _weighted_scatter_add_kernel(
    dn_ptr,        # (C, H) fp16 —— down 输出（padding 行为分配器残留，不读）
    tokens_ptr,    # (C,) int64 —— 行 -> token 序号
    weights_ptr,   # (C,) fp16/bf16 —— topk 权重（x.dtype）
    mask_ptr,      # (C,) bool
    out_ptr,       # (T, H) —— final_hidden_states（x.dtype）
    H,
    stride_dn, stride_out,
    BLOCK: tl.constexpr,
):
    """scatter 尾融合：有效行 w*dn 原子累加到 final[tokens[r]]。

    替代 masked_fill_ + mul_ + index_add_ 三 kernel（P0 kernel 数量优化）。
    padding 行（mask=0）直接跳过——不读 dn 残留也不加零，与旧行为
    （清零后加 0）数学等价（0 + x == x，含 ±0 符号差异经 == 比较不可见），
    同时天然免疫 0×inf/NaN 残留问题（残留根本不进算术）。
    w*dn 在 fp32 完成后单次舍入到 out dtype（旧链 .to(x.dtype) 与 mul_
    各舍一次，精度不低于旧链）。
    """
    pid = tl.program_id(0)
    n_col = tl.cdiv(H, BLOCK)
    row = pid // n_col
    col = (pid % n_col) * BLOCK + tl.arange(0, BLOCK)
    keep = tl.load(mask_ptr + row)
    if keep == 0:
        return
    cm = col < H
    w = tl.load(weights_ptr + row).to(tl.float32)
    v = tl.load(dn_ptr + row * stride_dn + col,
                mask=cm, other=0.0).to(tl.float32)
    tok = tl.load(tokens_ptr + row)
    out_ty = out_ptr.dtype.element_ty
    tl.atomic_add(out_ptr + tok * stride_out + col,
                  (v * w).to(out_ty), mask=cm)


def weighted_scatter_add_fused(dn: torch.Tensor, tokens: torch.Tensor,
                               weights: torch.Tensor, mask_rows: torch.Tensor,
                               out: torch.Tensor):
    """_weighted_scatter_add_kernel 的 python wrapper（静态池 A 融合）。

    dn: (C, H) fp16；tokens: (C,) int64；weights: (C,) 与 out 同 dtype；
    mask_rows: (C,) bool；out: (T, H) 就地原子累加（= index_add_ 语义）。
    """
    C, H = dn.shape
    BLOCK = 256
    grid = (C * triton.cdiv(H, BLOCK),)
    _weighted_scatter_add_kernel[grid](
        dn, tokens, weights, mask_rows, out,
        H, dn.stride(0), out.stride(0),
        BLOCK=BLOCK, num_warps=4,
    )


@triton.jit
def _pool_bookkeeping_kernel(
    flat_e_ptr,     # (R,) int64 —— 路由 expert 序号（R = T*top_k）
    expert_bit_ptr,     # (E,) int64 —— expert -> 池号
    expert_local_ptr,   # (E,) int64 —— expert -> 池内成员序号
    ei_gu_ptr,       # (M, 4) int32 —— 列 0/1 = token_start/token_count
    ei_dn_ptr,       # (M, 4) int32
    pool_total_ptr,  # (1,) int32 —— 本池有效行总数
    pool_id, M, R,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """池簿记融合（P0 kernel 数量优化）：成员计数（LUT 匹配 flat_e，替代
    层级 zeros+index_add_ 的 counts 链）+ 排他前缀和 + expert_info 行段列
    回写 + pool_total，单 CTA。每成员计数 = 本池内 local == m 的路由行数，
    整数计数、无原子、结果确定。列 2/3（切片列）为常量不动。
    """
    ar = tl.arange(0, BLOCK_M)
    m = ar < M
    cnt = tl.zeros([BLOCK_M], dtype=tl.int32)
    for j0 in range(0, R, BLOCK_R):
        js = j0 + tl.arange(0, BLOCK_R)
        jm = js < R
        e = tl.load(flat_e_ptr + js, mask=jm, other=-1)
        lut_ok = e >= 0
        loc = tl.load(expert_local_ptr + e, mask=lut_ok, other=-1)
        pol = tl.load(expert_bit_ptr + e, mask=lut_ok, other=-1)
        hit = (pol == pool_id) & (loc[None, :] == ar[:, None]) & jm[None, :]
        cnt += tl.sum(hit.to(tl.int32), 1)
    cursor = tl.cumsum(cnt, 0) - cnt
    tl.store(ei_gu_ptr + ar * 4 + 0, cursor, mask=m)
    tl.store(ei_gu_ptr + ar * 4 + 1, cnt, mask=m)
    tl.store(ei_dn_ptr + ar * 4 + 0, cursor, mask=m)
    tl.store(ei_dn_ptr + ar * 4 + 1, cnt, mask=m)
    tl.store(pool_total_ptr, tl.sum(cnt, 0))


@triton.jit
def _pool_buffer_zero_kernel(
    tokens_ptr,      # (C+1,) int64
    weights_ptr,     # (C+1,) act_dtype
    slot_local_ptr,  # (C+1,) int64
    mask_ptr,        # (C,) bool —— s < pool_total
    pool_total_ptr,  # (1,) int32
    C,
    BLOCK: tl.constexpr,
):
    """池缓冲清零 + 行掩码融合（P0）：一个 kernel 清 tokens/weights/slot_local
    三缓冲并写 mask_rows（arange < pool_total），替代 3×zero_ + lt 共 4 个
    kernel。padding 槽必须为 0（tokens_c 的 padding 行 gather token 0，
    结果确定性依赖 0 初值）。四段虚拟拼接 [0, 3*(C+1)+C)，lane 级区域掩码
    （C 小于 BLOCK 时块内跨界，指针必须重定位到段内局部下标）。
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    CP1 = C + 1
    base3 = 3 * CP1
    k = idx // CP1
    local = idx - k * CP1
    in3 = idx < base3
    tl.store(tokens_ptr + local, 0, mask=in3 & (k == 0))
    tl.store(weights_ptr + local,
             tl.zeros([BLOCK], dtype=weights_ptr.dtype.element_ty),
             mask=in3 & (k == 1))
    tl.store(slot_local_ptr + local, 0, mask=in3 & (k == 2))
    s = idx - base3
    pt = tl.load(pool_total_ptr)
    tl.store(mask_ptr + s, s < pt, mask=(s >= 0) & (s < C))


@triton.jit
def _pool_scatter_rows_kernel(
    flat_e_ptr,      # (R,) int64 —— 路由 expert 序号（R == C == T*top_k）
    flat_w_ptr,      # (R,) act_dtype —— topk 权重
    expert_bit_ptr,  # (E,) int64 —— expert -> 池号
    expert_local_ptr,  # (E,) int64 —— expert -> 池内成员序号
    ei_gu_ptr,       # (M, 4) int32 —— 列 0 = 排他游标（bookkeeping 已写）
    tokens_ptr, weights_ptr, slot_local_ptr,   # (C+1,) 目标缓冲
    pool_id,         # 本池编号（int32）
    C, TOP_K, R,
    BLOCK_R: tl.constexpr,
):
    """行散写融合（P0）：每行算 dst（本池游标 + 同 expert 先前行数；非本池
    行写丢弃槽 C）并散写三缓冲。替代层级的 same/tril/sum 先导计数链与
    池内 in_pool/clamp/index_select/where + 3×index_copy_ 共 ~12 kernel；
    先导计数（prior）改为 kernel 内 O(R) 扫描（R = T*top_k，decode 下 ≤ 64）。
    token 序号 = r // TOP_K（替代 arange+repeat_interleave）。
    """
    r = tl.program_id(0)
    e = tl.load(flat_e_ptr + r)
    # prior：j < r 且 flat_e[j] == e 的行数（同 expert 第几行）
    prior = 0
    for j0 in range(0, R, BLOCK_R):
        js = j0 + tl.arange(0, BLOCK_R)
        jm = js < R
        ej = tl.load(flat_e_ptr + js, mask=jm, other=-1)
        prior += tl.sum(((js < r) & (ej == e)).to(tl.int32), 0)
    pool = tl.load(expert_bit_ptr + e)
    local = tl.load(expert_local_ptr + e)
    cursor = tl.load(ei_gu_ptr + local * 4 + 0)
    in_pool = pool == pool_id
    dst = tl.where(in_pool, cursor + prior, C).to(tl.int64)
    tl.store(tokens_ptr + dst, (r // TOP_K).to(tl.int64))
    w = tl.load(flat_w_ptr + r)
    tl.store(weights_ptr + dst, w)
    tl.store(slot_local_ptr + dst, local)


def static_pool_dispatch_fused(flat_e, flat_w, expert_bit,
                               expert_local, ps, pool_id, top_k):
    """静态派发簿记 + 缓冲准备三连（P0 kernel 数量优化）。

    替代 _forward_static_decode 层级的 counts zeros+index_add_（2 kernel）
    + 池循环里的 8 kernel 簿记（m_counts/cumsum/expert_info×4/pool_total）
    + 4 kernel 清零掩码 + 12 kernel 行散写链 = 3 个 kernel。成员计数在
    簿记 kernel 内用 expert_bit/local LUT 匹配 flat_e 直接数出。行/掩码
    缓冲为池状态常驻张量（指针跨步稳定，capture 合法）。R == C
    （容量 = T*top_k）。
    """
    R = flat_e.shape[0]
    M = ps.num_members
    _pool_bookkeeping_kernel[(1,)](
        flat_e, expert_bit, expert_local,
        ps.expert_info_gu, ps.expert_info_dn, ps.pool_total_buf,
        pool_id, M, R,
        BLOCK_M=triton.next_power_of_2(max(M, 2)),
        BLOCK_R=triton.next_power_of_2(max(R, 2)), num_warps=1,
    )
    n_blk = triton.cdiv(3 * (ps.capacity + 1) + ps.capacity, 256)
    _pool_buffer_zero_kernel[(n_blk,)](
        ps.tokens_buf, ps.weights_buf, ps.slot_local_buf, ps.mask_buf,
        ps.pool_total_buf, ps.capacity, BLOCK=256, num_warps=1,
    )
    _pool_scatter_rows_kernel[(R,)](
        flat_e, flat_w, expert_bit, expert_local, ps.expert_info_gu,
        ps.tokens_buf, ps.weights_buf, ps.slot_local_buf,
        pool_id, ps.capacity, top_k, R,
        BLOCK_R=triton.next_power_of_2(max(R, 2)), num_warps=1,
    )


def build_int8_codebook(codebook: torch.Tensor):
    """把 fp16 码本转成 int8，返回 (cb_i8, cb_step)。

    满足 `codebook[i] ≈ cb_i8[i] * cb_step`。

    低 bit 下这一步的代价很小（实测 relerr 对 fp32 参考解，激活量化本身
    贡献约 0.0065，下面是含激活的总误差）：
        1-bit: 0.00647   2-bit: 0.00651   → 码本贡献可忽略
        4-bit: 0.00860   → 码本额外贡献约 0.0056（16 级里最内侧质心
                            ±0.128 相对 max 2.73 只剩 6/127 级）
    4-bit 在真实部署里只覆盖 1~4 个 expert（backlog:141-156），影响有限。
    **MoE 不需要重新量化，现有 2bpw checkpoint 直接可用。**

    ⚠ 8-bit 是例外：8-bit 码本是 256 级 Lloyd-Max（非均匀，零点附近极密），
    映射到均匀的 int8 网格时零点附近多级会塌成同一个值，实测相对误差
    从 0.00075 掉到 0.0128（等效把 W8 降成约 W6）。attention 路径要走
    WxA8 必须先把 8-bit 量化改成均匀对称码本（见 roadmaps/wxa8-plan-260829.md）。
    """
    cb = codebook.float()
    cb_max = cb.abs().max()
    if cb_max <= 0:
        raise ValueError("码本全零，无法转 int8")
    cb_step = (cb_max / 127.0).item()
    cb_i8 = (cb / cb_step).round().clamp_(-127, 127).to(torch.int8)
    return cb_i8.contiguous(), cb_step


# ===========================================================================
# Wrapper：gate_up（行切片）/ down（in_features 切片）
# ===========================================================================

def _launch(x_i8, x_scale, indices_slice, cb_i8, norms_slice,
            group_size, num_groups, bit_width, direction,
            identity_cb=False, cfg=None):
    B = x_i8.shape[0]
    N = indices_slice.shape[1]
    K_total = x_i8.shape[1]

    out = torch.empty(B, N, dtype=torch.float16, device=x_i8.device)

    BLOCK_B, BLOCK_N, BLOCK_K, num_warps, num_stages = (
        cfg if cfg is not None
        else get_wxa8_config(bit_width, direction=direction, B=B))

    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(N, BLOCK_N))
    _wxa8_fused_matmul_kernel_grouped_gf[grid](
        x_i8, x_scale, indices_slice, cb_i8, norms_slice, out,
        B, N, K_total,
        indices_slice.stride(0), norms_slice.stride(0), x_scale.stride(0),
        GROUP_SIZE=group_size, NUM_GROUPS=num_groups,
        BIT_WIDTH=bit_width, N_LEVELS=cb_i8.shape[0],
        IDENTITY_CB=identity_cb,
        BLOCK_B=BLOCK_B, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def wxa8_matmul_grouped_slice_rows_gf(
    x_i8, x_scale, indices_packed_gf, codebook_i8, norms_gf,
    group_size, in_features, row_start, row_end, bit_width: int = 2,
    norms_prescaled: bool = False,
):
    """gate_up 路径：行切片（输出维度切片）。

    对应 WxA16 的 triton_fused_matmul_grouped_slice_rows_gf，但要求 x 已经
    旋转 + 量化（没有 x_is_rotated 开关 —— WxA8 kernel 永不旋转）。

    Args:
        x_i8: (B, in_features) int8，已旋转并量化
        x_scale: (B, in_features // group_size) fp32（已折入 cb_step）
        indices_packed_gf: (num_groups, N_total, packed_per_group) uint8
        codebook_i8: (n_levels,) int8
        norms_gf: (num_groups, N_total) fp16，与 WxA16 的 norms_gf 同源
        row_start, row_end: 行切片范围
        norms_prescaled: norms 是否已预乘 1/sqrt(group_size)。
            主路径恒为 True（P6-2 在建 gf 布局时预乘）；False 时这里补除，
            对齐 WxA16 的语义。

    Returns:
        output: (B, row_end - row_start) fp16
    """
    if bit_width not in {1, 2, 4, 8}:
        raise ValueError(f"bit_width must be 1/2/4/8, got {bit_width}")
    if in_features % group_size != 0:
        raise ValueError(
            f"WxA8 要求 in_features ({in_features}) 对齐 group_size ({group_size})")

    num_groups = in_features // group_size
    norms_slice = norms_gf[:, row_start:row_end]
    if not norms_prescaled:
        norms_slice = norms_slice.float() / math.sqrt(group_size)
    return _launch(
        x_i8, x_scale,
        indices_packed_gf[:, row_start:row_end, :],
        codebook_i8,
        norms_slice,
        group_size, num_groups, bit_width, "gate_up",
    )


def wxa8_gate_up_silu_fused_gf(
    x_i8, x_scale, indices_packed_gf, codebook_i8, norms_gf,
    group_size, in_features, inter_start, inter_end, bit_width: int = 2,
    norms_prescaled: bool = False,
):
    """gate_up + silu×up 融合 kernel（P5-1）。

    等价于：
        gate_up_out = wxa8_matmul_grouped_slice_rows_gf(2*inter_start, 2*inter_end)
        gate = gate_up_out[:, :inter_size]
        up = gate_up_out[:, inter_size:]
        act_out = silu(gate) * up

    但中间结果不落地，省一次完整 gate_up 输出的读写 + 一次独立 silu launch。

    Args:
        inter_start, inter_end: inter_size 维度的切片范围。
            权重的 gate_up 总列数 = 2 * total_inter_size，其中 gate 列 =
            [0, total_inter_size)，up 列 = [total_inter_size, 2*total_inter_size)。
            这里传的是 inter 侧的切片范围，函数内部自动映射到 gate 和 up 的列。
        其他参数同 wxa8_matmul_grouped_slice_rows_gf。

    Returns:
        output: (B, inter_end - inter_start) fp16 = silu(gate) * up
    """
    if bit_width not in {1, 2, 4, 8}:
        raise ValueError(f"bit_width must be 1/2/4/8, got {bit_width}")
    if in_features % group_size != 0:
        raise ValueError(
            f"WxA8 要求 in_features ({in_features}) 对齐 group_size ({group_size})")

    num_groups = in_features // group_size
    inter_size = inter_end - inter_start
    # gate 列: [inter_start, inter_end)，up 列: [total_inter+inter_start, total_inter+inter_end)
    # 但 indices/norms 的完整列数 = 2*total_inter_size，我们需要的是连续的 2*inter_size 列
    # 从 gate 侧的 inter_start 到 up 侧的 total_inter+inter_end —— 这不连续！
    #
    # 注意：bit_partitioned_moe 里 _gate_up_matmul 传的 row_start/row_end =
    # 2*start, 2*end，其中 start/end 是 inter_size 维度的偏移。
    # 也就是说，权重的 gate_up 按 [gate_0, up_0, gate_1, up_1, ..., gate_n, up_n] 排列？
    # 不，让我回去确认一下。
    #
    # WxA16 的 gate_up 是 gate_proj 和 up_proj 拼接在一起的：
    #   total N = 2 * inter_size，gate = 前 inter_size 列，up = 后 inter_size 列
    # 所以切片 [2*start, 2*end) = [gate_start..gate_end) + [up_start..up_end)
    # 其中 gate_start = 2*start, gate_end = start+end, up_start = start+end, up_end = 2*end？
    # 不对，让我想清楚。
    #
    # 看 bit_partitioned_moe.py 的 _gate_up_matmul:
    #   return gate_up_out, expert_tokens
    # 然后: gate_out = gate_up_out[:, :actual_inter_size]
    #       up_out = gate_up_out[:, actual_inter_size:]
    # 所以 gate_up_out 的列数 = 2 * actual_inter_size，
    # 前半 = gate，后半 = up。
    # 也就是说 row_start=2*start, row_end=2*end 切出来的是 gate 的 [start,end) + up 的 [start,end)。
    # 这是连续的 2*(end-start) 列，没错。
    #
    # 所以 fusion kernel 需要的权重就是 indices_packed_gf[:, 2*inter_start : 2*inter_end, :]
    # 和 norms_gf[:, 2*inter_start : 2*inter_end]
    # 输出列数 = inter_end - inter_start。
    #
    # 但权重的布局是 [gate_all, up_all]，所以 2*inter_start:2*inter_end 正好是
    # [gate_part, up_part]，连续的 2*inter_size 列。✅

    row_start = 2 * inter_start
    row_end = 2 * inter_end

    # 取切片：indices 和 norms 的列范围是 [row_start, row_end) = 2*inter_size 列
    indices_slice = indices_packed_gf[:, row_start:row_end, :]
    norms_slice = norms_gf[:, row_start:row_end]
    if not norms_prescaled:
        norms_slice = norms_slice.float() / math.sqrt(group_size)

    B = x_i8.shape[0]
    K_total = x_i8.shape[1]
    N_out = inter_size

    out = torch.empty(B, N_out, dtype=torch.float16, device=x_i8.device)

    BLOCK_B, BLOCK_N, BLOCK_K, num_warps, num_stages = (
        get_wxa8_config(bit_width, direction="gate_up", B=B))

    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(N_out, BLOCK_N))
    _wxa8_gate_up_silu_kernel_gf[grid](
        x_i8, x_scale, indices_slice, codebook_i8, norms_slice, out,
        B, N_out, K_total,
        indices_slice.stride(0), norms_slice.stride(0), x_scale.stride(0),
        GROUP_SIZE=group_size, NUM_GROUPS=num_groups,
        BIT_WIDTH=bit_width, N_LEVELS=codebook_i8.shape[0],
        IDENTITY_CB=False,
        BLOCK_B=BLOCK_B, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def wxa8_matmul_grouped_slice_in_features_gf(
    x_i8, x_scale, indices_packed_gf, codebook_i8, norms_gf,
    group_size, original_start, original_end, bit_width: int = 2,
    norms_prescaled: bool = False,
):
    """down 路径：in_features 切片（group 维切片，整块连续）。

    对应 WxA16 的 triton_fused_matmul_grouped_slice_in_features_gf。
    x_i8 已经是切片后的输入（每个 expert 自己的 act_out），且已用
    seed_base = seed + original_start 旋转过再量化。

    Args:
        x_i8: (B, original_end - original_start) int8
        x_scale: (B, num_groups_in_slice) fp32（已折入 cb_step）
        original_start, original_end: 原始全权重的 in_features 切片范围
        norms_prescaled: 同 slice_rows_gf

    Returns:
        output: (B, N) fp16
    """
    if bit_width not in {1, 2, 4, 8}:
        raise ValueError(f"bit_width must be 1/2/4/8, got {bit_width}")
    if original_start % group_size != 0:
        raise ValueError(
            f"original_start ({original_start}) 必须对齐 group_size ({group_size})")

    slice_in_features = original_end - original_start
    if slice_in_features % group_size != 0:
        raise ValueError(
            f"切片宽度 ({slice_in_features}) 必须对齐 group_size ({group_size})")

    g_start = original_start // group_size
    g_end = original_end // group_size

    norms_slice = norms_gf[g_start:g_end]
    if not norms_prescaled:
        norms_slice = norms_slice.float() / math.sqrt(group_size)
    return _launch(
        x_i8, x_scale,
        indices_packed_gf[g_start:g_end],
        codebook_i8,
        norms_slice,
        group_size, g_end - g_start, bit_width, "down",
    )


def wxa8_matmul_grouped_gf(
    x_i8, x_scale, indices_packed_gf, codebook_i8, norms_gf,
    group_size, num_groups, bit_width: int = 8,
    identity_cb: bool = False, cfg=None,
):
    """attention 路径：完整矩阵（无切片）。

    对应 WxA16 的 triton_fused_matmul_grouped（gf 布局变体），输入已经
    旋转 + 量化（调用方用 rotate_quantize_fused 做，cb_step 折进 x_scale）。

    identity_cb=True（8-bit 均匀码本）时 codebook_i8 不被访问，传任意
    形状为 (256,) 的 int8 占位张量即可。

    Args:
        x_i8: (B, K_total) int8
        x_scale: (B, num_groups) fp32
        indices_packed_gf: (num_groups, N, packed_per_group) uint8
        codebook_i8: (n_levels,) int8（identity_cb 时为占位）
        norms_gf: (num_groups, N) fp16，需已预乘 1/sqrt(group_size)

    Returns:
        output: (B, N) fp16
    """
    if bit_width not in {1, 2, 4, 8}:
        raise ValueError(f"bit_width must be 1/2/4/8, got {bit_width}")
    return _launch(
        x_i8, x_scale, indices_packed_gf, codebook_i8, norms_gf,
        group_size, num_groups, bit_width, "attn",
        identity_cb=identity_cb, cfg=cfg,
    )


# ===========================================================================
# P5-2 重做：按 bit 池合并多 expert 的 grouped kernel（2026-10-20）
#
# 动机（throughput-analysis-261020.md）：decode 每步 MoE 段 57.3ms 由
# 「320 次专家迭代 × ~10 个 Python/torch 调度小算子」主导（launch 本身
# ~10ms 不是大头）；prefill 由小 kernel SM 填充不足（单 expert 1.51 波）
# 主导。官方 FP8 的 grouped kernel 一次覆盖全部 active expert。
#
# 方案：同一 bit 池的所有 expert 合并进一次 kernel launch。
#   - token 已按 expert 排序（sorted_tokens），x 被 gather 成扁平
#     (B_total, K) 连续布局（每 bit 池一次 gather）
#   - expert_info (E, 4) int32: [token_start, token_count, row_start, row_count]
#     row_start/row_count 是 expert 在共享 indices/norms 上的列（行）切片
#   - block_info (num_blocks, 3) int32: [expert_idx, block_b, block_n]，
#     host 端预计算；1D grid，每 block 先读自己的 (expert, tile) 映射
#   - 输出扁平 (B_total, N_slice) 连续布局（本模型同 bit 池内 expert
#     inter_size 一致，行与列切片都连续），silu/scatter 整块一次做完
#
# 数学与逐 expert 路径完全一致：同一 tile 分解（BLOCK_B/BLOCK_N/BLOCK_K
# 同表）、同一累加顺序 → 期望 bit-exact。
# ===========================================================================


@triton.jit
def _wxa8_merged_kernel_gf(
    # Input
    x_ptr,            # (B_src, K_total) int8 —— GATHER_X=0 时行序即池行序
    xs_ptr,           # (B_src, NUM_GROUPS) fp32 —— 行与 x 对齐
    tokens_ptr,       # (B_total,) int64 —— GATHER_X=1 时的池行→源行索引
    # Quantized weight (group-first 布局，本 bit 池共享)
    indices_ptr,      # (NUM_GROUPS_TOTAL, N_total, PACKED_PER_GROUP) uint8
    codebook_ptr,     # (n_levels,) int8
    norms_ptr,        # (NUM_GROUPS_TOTAL, N_total) fp16（预乘态见 NORMS_PRESCALED）
    # Expert / block 布局
    expert_info_ptr,  # (E, 4) int32: [token_start, token_count, slice_start, slice_count]
    block_info_ptr,   # (num_blocks, 3) int32: [expert_idx, block_b, block_n]
    # Output
    output_ptr,       # (B_total, N_slice) fp16 扁平连续，行 stride = OUT_ROW_STRIDE
    # Shape / strides
    B_total, K_total, N_LIMIT,
    INDICES_G0_STRIDE, NORMS_G0_STRIDE, XS_ROW_STRIDE, OUT_ROW_STRIDE,
    NORM_INV,       # fp32 1/sqrt(gs)：NORMS_PRESCALED=False 时折入 norm
    # Constexpr config
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,     # 本 expert 的有效 group 数
    BIT_WIDTH: tl.constexpr,
    N_LEVELS: tl.constexpr,
    NORMS_PRESCALED: tl.constexpr = True,
    GATHER_X: tl.constexpr = False,
    SLICE_IS_GROUPS: tl.constexpr = False,
    BLOCK_B: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 32,
    BLOCK_K: tl.constexpr = 128,
):
    """按 bit 池合并多 expert 的 WxA8 grouped GEMM（双模式）。

    SLICE_IS_GROUPS=False（gate_up / 行切片）：slice_start/slice_count 是
    expert 在共享 indices/norms 上的**列**切片（rn 从 slice_start 起，输出
    列回到 [0, slice_count)），K 维是完整的 in_features（每 expert 相同）。
    SLICE_IS_GROUPS=True（down / in_features 切片）：slice_start/slice_count
    是 **group** 切片（indices/norms 从全局 group slice_start 起），K 维
    已是切片后的 inter（x 的列与 xs 都按切片后的 group 排布），N 维完整。
    行维两模式一致：从 token_start 起（B 维是 gather 后的扁平 token 序）。

    与 _wxa8_fused_matmul_kernel_grouped_gf 的 tile 数学完全一致（同一
    tile 分解、同一累加顺序）→ 期望与逐 expert 路径 bit-exact。
    """
    pid = tl.program_id(0)
    bi = block_info_ptr + pid * 3
    eid = tl.load(bi + 0).to(tl.int32)
    blk_b = tl.load(bi + 1).to(tl.int32)
    blk_n = tl.load(bi + 2).to(tl.int32)

    ei = expert_info_ptr + eid * 4
    token_start = tl.load(ei + 0).to(tl.int32)
    token_count = tl.load(ei + 1).to(tl.int32)
    slice_start = tl.load(ei + 2).to(tl.int32)
    slice_count = tl.load(ei + 3).to(tl.int32)

    # 静态派发下 expert_info 覆盖池内全部 expert（decode 时绝大多数
    # token_count=0）：空成员整块早退，权重流带与 dot 全免（对齐 pooled
    # rotate kernel 的写法）。host 编排路径成员恒非空，此分支不触发
    if token_count <= 0:
        return

    rb = token_start + blk_b * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = rb < token_start + token_count

    # GATHER_X=1：x/xs 按 tokens 索引取源行（免 host 侧 index_select 物化），
    # 输出行序仍是池行序 rb（padding 行 tokens=0，且被 mask_b 屏蔽不读）
    if GATHER_X:
        rb_x = tl.load(tokens_ptr + rb, mask=mask_b, other=0)
    else:
        rb_x = rb

    if SLICE_IS_GROUPS:
        # down：N 维完整（从 0 起），group 维按 slice_start 偏移
        rn = blk_n * BLOCK_N + tl.arange(0, BLOCK_N)
    else:
        # gate_up：N 维是共享权重的列切片，从 slice_start 起
        rn = slice_start + blk_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = rn < N_LIMIT

    total_acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

    ELEMENTS_PER_BYTE = 8 // BIT_WIDTH
    PACKED_PER_GROUP = GROUP_SIZE // ELEMENTS_PER_BYTE
    row_base = rn * PACKED_PER_GROUP

    for g in range(NUM_GROUPS):
        g_global = (slice_start + g) if SLICE_IS_GROUPS else g
        g_start = g * GROUP_SIZE
        g_base = g_global * INDICES_G0_STRIDE

        norm_g = tl.load(norms_ptr + g_global * NORMS_G0_STRIDE + rn,
                         mask=mask_n, other=0.0)
        if not NORMS_PRESCALED:
            # norms 未预乘 1/sqrt(gs)：kernel 内 fp32 乘倒数（与 host 侧
            # norms.float()/sqrt(gs) 同值——PyTorch 的标量除法在 CUDA 上
            # 就是乘 float 倒数），省掉每步 2 个 elementwise kernel
            norm_g = norm_g.to(tl.float32) * NORM_INV
        xs_g = tl.load(xs_ptr + rb_x * XS_ROW_STRIDE + g, mask=mask_b, other=0.0)

        acc_i = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.int32)

        for k_start in range(0, GROUP_SIZE, BLOCK_K):
            rk = k_start + tl.arange(0, BLOCK_K)
            mask_k = rk < GROUP_SIZE

            inp_off = rb_x[:, None] * K_total + (g_start + rk)[None, :]
            x_tile = tl.load(x_ptr + inp_off,
                             mask=mask_b[:, None] & mask_k[None, :], other=0)

            w_mask = mask_n[:, None] & mask_k[None, :]
            if BIT_WIDTH == 8:
                byte_off = g_base + row_base[:, None] + rk[None, :]
                idx = tl.load(indices_ptr + byte_off, mask=w_mask, other=0).to(tl.int32)
            else:
                BIT_MASK = (1 << BIT_WIDTH) - 1
                byte_col = rk // ELEMENTS_PER_BYTE
                pos_in_byte = rk % ELEMENTS_PER_BYTE
                byte_off = g_base + row_base[:, None] + byte_col[None, :]
                packed = tl.load(indices_ptr + byte_off, mask=w_mask, other=0).to(tl.uint8)
                shift = pos_in_byte * BIT_WIDTH
                idx = ((packed >> shift[None, :]) & BIT_MASK).to(tl.int32)

            w_i8 = tl.load(codebook_ptr + idx, mask=w_mask, other=0)

            acc_i += tl.dot(x_tile, tl.trans(w_i8), out_dtype=tl.int32)

        total_acc += acc_i.to(tl.float32) * xs_g[:, None] * norm_g[None, :]

    # 输出列坐标回到扁平缓冲（down 模式 rn 已从 0 起，rows 模式回退 slice_start）
    out_col = (rn - slice_start) if not SLICE_IS_GROUPS else rn
    out_off = rb[:, None] * OUT_ROW_STRIDE + out_col[None, :]
    tl.store(output_ptr + out_off, total_acc.to(output_ptr.dtype.element_ty),
             mask=mask_b[:, None] & mask_n[None, :])


def _build_merged_block_info(expert_info, BLOCK_B, BLOCK_N, n_extent, device):
    """host 端预计算 block_info：(num_blocks, 3) int32 [expert_idx, block_b, block_n]。

    n_extent 是 N 方向（列）的 tile 覆盖范围：rows 模式 = 每 expert 的列切片
    宽（2*inter）；in_features 模式 = 完整输出列数 N（hidden）。token 方向
    按各 expert 的 token_count 切。合并后总 grid = num_blocks（1D）。
    """
    blocks = []
    for e, (token_start, token_count, slice_start, slice_count) in enumerate(expert_info):
        if token_count <= 0 or slice_count <= 0:
            continue
        n_b = (token_count + BLOCK_B - 1) // BLOCK_B
        n_n = (n_extent + BLOCK_N - 1) // BLOCK_N
        for b in range(n_b):
            for n in range(n_n):
                blocks.append((e, b, n))
    if not blocks:
        raise ValueError("merged kernel: empty block list (no active experts)")
    return torch.tensor(blocks, dtype=torch.int32, device=device).contiguous()


def _launch_wxa8_merged_gf(
    x_i8, x_scale, indices_packed_gf, codebook_i8, norms_used,
    expert_info_dev, block_info, out, n_extent,
    group_size, num_groups, bit_width, slice_is_groups, cfg,
    norms_prescaled=True, tokens=None,
):
    """wxa8 merged kernel 统一发射入口（host 编排路径与静态派发路径共用）。

    参数序与 _wxa8_merged_kernel_gf 形参一一对应；expert_info_dev 必须是
    (E,4) int32 设备张量（kernel 运行时读行段/列段，行段可由设备端每步
    重写）；SLICE_IS_GROUPS 两模式语义见 kernel docstring。norms_prescaled
    =False 时 norms_used 直接传 fp16 原始 norms，1/sqrt(gs) 由 kernel 内
    乘 NORM_INV 折入（host 侧零 elementwise）。tokens 非 None 时 GATHER_X=1
    （x/xs 的源行由 tokens 索引，out 行数取 tokens 长度——静态 gate_up
    方向免两笔 index_select 物化）。
    """
    BLOCK_B, BLOCK_N, BLOCK_K, num_warps, num_stages = cfg
    _wxa8_merged_kernel_gf[(block_info.shape[0],)](
        x_i8, x_scale, tokens if tokens is not None else x_i8,
        indices_packed_gf, codebook_i8, norms_used,
        expert_info_dev, block_info, out,
        out.shape[0], x_i8.shape[1], n_extent,
        indices_packed_gf.stride(0), norms_used.stride(0), x_scale.stride(0),
        out.stride(0),
        1.0 / math.sqrt(group_size) if not norms_prescaled else 1.0,
        GROUP_SIZE=group_size, NUM_GROUPS=num_groups,
        BIT_WIDTH=bit_width, N_LEVELS=codebook_i8.shape[0],
        NORMS_PRESCALED=norms_prescaled,
        GATHER_X=tokens is not None,
        SLICE_IS_GROUPS=slice_is_groups,
        BLOCK_B=BLOCK_B, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )


def wxa8_matmul_merged_rows_gf(
    x_i8, x_scale, indices_packed_gf, codebook_i8, norms_gf,
    expert_info, group_size, in_features, bit_width: int = 2,
    norms_prescaled: bool = True, cfg=None,
):
    """gate_up 合并版：同一 bit 池所有 expert 一次 launch（行切片语义）。

    与逐 expert 的 wxa8_matmul_grouped_slice_rows_gf 逐段等价（数学
    bit-exact 期望）。x_i8 已按 sorted_tokens gather 成 (B_total, K) 扁平
    布局，x_scale 行对齐；expert_info 里 row_start/row_count 是该 expert
    在共享 indices/norms 上的列切片（主流程为 e*2*inter / 2*inter）。

    Returns:
        output: (B_total, 2*inter) fp16 扁平连续（按 expert 顺序行拼接）
    """
    if bit_width not in {1, 2, 4, 8}:
        raise ValueError(f"bit_width must be 1/2/4/8, got {bit_width}")
    if in_features % group_size != 0:
        raise ValueError(
            f"WxA8 要求 in_features ({in_features}) 对齐 group_size ({group_size})")

    expert_info = [tuple(int(v) for v in row) for row in expert_info]
    B_total = x_i8.shape[0]
    num_groups = in_features // group_size
    # 同 bit 池内 inter_size 一致 → N_slice = row_count 统一
    n_slice = expert_info[0][3] if expert_info else 0
    if any(row[3] != n_slice for row in expert_info):
        raise ValueError(
            f"merged rows kernel 要求池内列切片一致（inter_size 相同），"
            f"得到 row_count 集合 {sorted({row[3] for row in expert_info})}")

    norms_full = norms_gf if norms_prescaled else (
        norms_gf.float() / math.sqrt(group_size))

    BLOCK_B, BLOCK_N, BLOCK_K, num_warps, num_stages = (
        cfg if cfg is not None
        else get_wxa8_config(bit_width, direction="gate_up", B=B_total))

    block_info = _build_merged_block_info(expert_info, BLOCK_B, BLOCK_N,
                                          n_slice, x_i8.device)
    out = torch.empty(B_total, n_slice, dtype=torch.float16, device=x_i8.device)
    n_limit = max(row[2] for row in expert_info) + n_slice

    _launch_wxa8_merged_gf(
        x_i8, x_scale, indices_packed_gf, codebook_i8, norms_full,
        torch.tensor(expert_info, dtype=torch.int32, device=x_i8.device),
        block_info, out, n_limit,
        group_size, num_groups, bit_width, slice_is_groups=False,
        cfg=(BLOCK_B, BLOCK_N, BLOCK_K, num_warps, num_stages))
    return out


def wxa8_matmul_merged_in_features_gf(
    x_i8, x_scale, indices_packed_gf, codebook_i8, norms_gf,
    expert_info, group_size, bit_width: int = 2,
    norms_prescaled: bool = True, cfg=None,
):
    """down 合并版：同一 bit 池所有 expert 一次 launch（in_features 切片语义）。

    x_i8 已按 expert 顺序拼接成 (B_total, inter) 扁平布局（每 expert 的
    激活已用 seed+original_start 旋转并量化，v1 由调用方逐 expert 调
    rotate_quantize_fused）；expert_info 里 row_start/row_count 是 group
    维切片（主流程为 e*inter/gs / inter/gs），列 N 是共享的完整 hidden。

    Returns:
        output: (B_total, N) fp16 扁平连续（按 expert 顺序行拼接）
    """
    if bit_width not in {1, 2, 4, 8}:
        raise ValueError(f"bit_width must be 1/2/4/8, got {bit_width}")

    expert_info = [tuple(int(v) for v in row) for row in expert_info]
    B_total = x_i8.shape[0]
    g_count = expert_info[0][3] if expert_info else 0
    if any(row[3] != g_count for row in expert_info):
        raise ValueError(
            f"merged in_features kernel 要求池内 group 切片一致（inter_size "
            f"相同），得到集合 {sorted({row[3] for row in expert_info})}")

    norms_slice = norms_gf if norms_prescaled else (
        norms_gf.float() / math.sqrt(group_size))
    N = norms_slice.shape[1]

    BLOCK_B, BLOCK_N, BLOCK_K, num_warps, num_stages = (
        cfg if cfg is not None
        else get_wxa8_config(bit_width, direction="down", B=B_total))

    block_info = _build_merged_block_info(expert_info, BLOCK_B, BLOCK_N,
                                          N, x_i8.device)
    out = torch.empty(B_total, N, dtype=torch.float16, device=x_i8.device)

    _launch_wxa8_merged_gf(
        x_i8, x_scale, indices_packed_gf, codebook_i8, norms_slice,
        torch.tensor(expert_info, dtype=torch.int32, device=x_i8.device),
        block_info, out, N,
        group_size, g_count, bit_width, slice_is_groups=True,
        cfg=(BLOCK_B, BLOCK_N, BLOCK_K, num_warps, num_stages))
    return out


def wxa8_matmul_merged_rows_gf_static(
    x_i8, x_scale, indices_packed_gf, codebook_i8, norms_gf,
    expert_info_dev, block_info, n_slice, n_limit,
    group_size, in_features, bit_width: int = 2,
    norms_prescaled: bool = True, cfg=None, tokens=None,
):
    """gate_up 合并版·静态派发（CUDA graph decode 用）。

    与 wxa8_matmul_merged_rows_gf 同一 kernel、同一 tile 数学；差别仅在
    编排来源：expert_info（(E,4) int32，行段列每步设备端重写、列切片列
    为加载期常量）与 block_info（最坏情况静态调度表）均为设备常驻张量。
    函数内零 host 读、零 H2D、零形状变化——capture 期合法。norms 直传
    fp16 原始值（不预乘 1/sqrt(gs)），缩放由 kernel 内 NORM_INV 折入，
    消掉 host 侧 cast+mul 两个 elementwise kernel。tokens 非 None 时
    GATHER_X=1：x_i8/x_scale 传全量 token 旋转量化结果，池内源行由
    tokens 索引（免 index_select 物化），输出仍按池行序 (C, n_slice)。
    """
    out_rows = tokens.shape[0] if tokens is not None else x_i8.shape[0]
    out = torch.empty(out_rows, n_slice, dtype=torch.float16,
                      device=x_i8.device)
    if cfg is None:
        cfg = get_wxa8_config(bit_width, direction="gate_up", B=out_rows)
    _launch_wxa8_merged_gf(
        x_i8, x_scale, indices_packed_gf, codebook_i8, norms_gf,
        expert_info_dev, block_info, out, n_limit,
        group_size, in_features // group_size, bit_width,
        slice_is_groups=False, cfg=cfg,
        norms_prescaled=norms_prescaled, tokens=tokens)
    return out


def wxa8_matmul_merged_in_features_gf_static(
    x_i8, x_scale, indices_packed_gf, codebook_i8, norms_gf,
    expert_info_dev, block_info, group_size, g_count,
    bit_width: int = 2, norms_prescaled: bool = True, cfg=None,
):
    """down 合并版·静态派发（CUDA graph decode 用），契约同 rows 静态版。

    N（完整输出列数）取 norms_gf.shape[1]（静态形状）；g_count 为池内
    group 切片计数（静态常量，kernel 的 NUM_GROUPS）。norms 直传 fp16
    原始值（同 rows 静态版，缩放入 kernel）。
    """
    N = norms_gf.shape[1]
    out = torch.empty(x_i8.shape[0], N, dtype=torch.float16,
                      device=x_i8.device)
    if cfg is None:
        cfg = get_wxa8_config(bit_width, direction="down", B=x_i8.shape[0])
    _launch_wxa8_merged_gf(
        x_i8, x_scale, indices_packed_gf, codebook_i8, norms_gf,
        expert_info_dev, block_info, out, N,
        group_size, g_count, bit_width, slice_is_groups=True, cfg=cfg,
        norms_prescaled=norms_prescaled)
    return out
