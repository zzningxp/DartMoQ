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
