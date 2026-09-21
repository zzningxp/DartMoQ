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


def pack_nbit(indices: torch.Tensor, bit_width: int) -> torch.Tensor:
    """将n-bit索引打包到uint8张量中。

    支持 1/2/4/8 bit。

    Args:
        indices: 整数张量，形状为 (..., N)，值范围 [0, 2^bit_width - 1]
        bit_width: 1/2/4/8

    Returns:
        packed: uint8张量，形状为 (..., packed_N)
    """
    supported_bits = {1, 2, 4, 8}
    if bit_width not in supported_bits:
        raise ValueError(f"Only {supported_bits} bits are supported, got {bit_width}")

    elements_per_byte = 8 // bit_width
    N = indices.shape[-1]

    # 计算需要填充的数量
    pad = (elements_per_byte - (N % elements_per_byte)) % elements_per_byte
    if pad > 0:
        indices = torch.nn.functional.pad(indices, (0, pad), value=0)

    # 重塑为 (..., N_padded // elements_per_byte, elements_per_byte)
    packed_shape = indices.shape[:-1] + (-1, elements_per_byte)
    indices_reshaped = indices.reshape(packed_shape)

    # 打包
    packed = torch.zeros(indices.shape[:-1] + (indices.shape[-1] // elements_per_byte,),
                        dtype=torch.uint8, device=indices.device)

    mask = (1 << bit_width) - 1
    for i in range(elements_per_byte):
        packed |= (indices_reshaped[..., i].to(torch.uint8) & mask) << (i * bit_width)

    return packed


def unpack_nbit(packed: torch.Tensor, bit_width: int, N: int) -> torch.Tensor:
    """将uint8打包数据解包为n-bit索引张量。

    Args:
        packed: uint8张量，形状为 (..., packed_N)
        bit_width: 1/2/4/8
        N: 原始最后一维的长度

    Returns:
        indices: int32张量，形状为 (..., N)
    """
    supported_bits = {1, 2, 4, 8}
    if bit_width not in supported_bits:
        raise ValueError(f"Only {supported_bits} bits are supported, got {bit_width}")

    elements_per_byte = 8 // bit_width
    mask = (1 << bit_width) - 1

    # 重塑为 (..., packed_N, elements_per_byte)
    indices = torch.zeros(packed.shape[:-1] + (packed.shape[-1] * elements_per_byte,),
                         dtype=torch.int32, device=packed.device)

    for i in range(elements_per_byte):
        indices[..., i::elements_per_byte] = ((packed >> (i * bit_width)) & mask).to(torch.int32)

    # 截取原始长度
    return indices[..., :N]


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


@torch.no_grad()
def turboquant_quantize_packed_full(
    W: torch.Tensor,
    bit_width: int,
    group_size: Optional[int] = None,
    seed: int = 42,
    rotation: str = "qr",
    keep_on_gpu: bool = False,
    uniform_codebook: bool = False,
) -> dict:
    """全功能 TurboQuant 量化并返回打包表示，支持 1/2/4/8 bit。

    Args:
        W: 权重矩阵，形状为 (out_features, in_features)
        bit_width: 1/2/4/8
        group_size: 分组大小
        seed: 旋转种子
        rotation: "qr" 或 "hadamard" 或 "none"
        keep_on_gpu: 是否保持在 GPU 上（默认移到 CPU 节省显存）
        uniform_codebook: 仅 bit_width==8 有效。True 时码本改为均匀网格
            `cb[i] = (i - 128) * (3.5/128)`，使得 indices 可以直接映射为
            int8 权重（w_i8 = idx - 128，kernel 免查表）——WxA8 的 attention
            路径需要这个，因为 256 级 Lloyd-Max 非均匀码本塞进 int8 均匀
            网格会塌掉 61 级（实测等效约 7.6-bit，relerr 0.0128）。
            代价：量化误差约 2 倍（Lloyd-Max 8-bit SQNR 48.1 dB →
            均匀 41.9 dB），且 ±3.5σ 以外的尾部被 clip（约 0.05% 质量）。

    Returns:
        dict 包含完整的量化信息（codebook_type: "uniform" / "lloydmax"）
    """
    supported_bits = {1, 2, 4, 8}
    if bit_width not in supported_bits:
        raise ValueError(f"Only {supported_bits} bits are supported, got {bit_width}")
    if uniform_codebook and bit_width != 8:
        raise ValueError(f"uniform_codebook 只支持 bit_width=8，got {bit_width}")

    M, N = W.shape
    if group_size is None:
        group_size = N

    orig_dtype = W.dtype
    W = W.float()

    # 获取码本
    if uniform_codebook:
        # 均匀网格：cb[i] = (i-128)*step，step = 3.5/128。
        # cb[0] = -3.5、cb[255] = 127*step，w_i8 = idx-128 恰为 int8 全域。
        # 覆盖 N(0,1) 的 ±3.5σ（clip 掉约 0.05% 尾部质量）。
        n_levels = 256
        step = 3.5 / 128.0
        centroids = (torch.arange(n_levels, dtype=torch.float32) - 128) * step
        boundaries = (torch.arange(n_levels - 1, dtype=torch.float32) - 127.5) * step
        codebook_type = "uniform"
    else:
        centroids, boundaries = get_codebook(bit_width)
        codebook_type = "lloydmax"
    centroids = centroids.to(W.device)
    boundaries = boundaries.to(W.device)

    # 收集每个分组的信息
    all_norms = []
    all_indices = []
    all_rotations = []  # 如果需要存储旋转矩阵

    # 按分组遍历
    for g_start in range(0, N, group_size):
        g_end = min(g_start + group_size, N)
        g_dim = g_end - g_start
        W_g = W[:, g_start:g_end]

        # Step 1: 行归一化
        norms = W_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_norm = W_g / norms
        all_norms.append(norms.squeeze(1))

        # Step 2: 随机正交旋转
        if rotation == "none":
            Y = W_norm
        elif rotation == "hadamard":
            Y = hadamard_rotate(W_norm, seed=seed + g_start)
        else:  # qr
            Pi = generate_rotation_matrix(g_dim, seed=seed + g_start).to(W.device)
            Y = W_norm @ Pi.T

        # 归一化旋转后的方差
        scale = math.sqrt(g_dim)
        Y_scaled = Y * scale

        # Step 3: 标量量化（Lloyd-Max 或均匀码本，共用 searchsorted 边界查找）
        indices = torch.searchsorted(boundaries, Y_scaled.reshape(-1))
        indices = indices.clamp(0, len(centroids) - 1).reshape(M, g_dim)
        all_indices.append(indices)

    # 拼接所有分组的索引和范数
    full_indices = torch.cat(all_indices, dim=1)
    norms_out = torch.stack(all_norms, dim=1) if len(all_norms) > 1 else all_norms[0]

    # 打包索引
    packed = pack_nbit(full_indices, bit_width)

    # 准备输出
    target_device = W.device if keep_on_gpu else "cpu"

    # Step 2 (FP16 链路改造): codebook 和 norms 存 fp16
    # - codebook: 码本值是归一化标量，fp16 精度足够，误差极小
    # - norms: 行范数值，fp16 精度足够（范围通常在 ~1 左右）
    # 好处: 减少显存带宽，kernel 内 tl.dot 两边都是 fp16，可走 FP16 Tensor Core
    result = {
        "indices_packed": packed.to(target_device),
        "codebook": centroids.half().to("cpu"),  # 码本始终放 CPU，fp16 存储
        "norms": norms_out.half().to(target_device),  # norms 也改为 fp16
        "seed": seed,
        "group_size": group_size,
        "shape": (M, N),
        "bit_width": bit_width,
        "rotation": rotation,
        "orig_dtype": str(orig_dtype),
        "codebook_type": codebook_type,
    }

    return result


@torch.no_grad()
def turboquant_dequantize_packed(packed_data: dict, device: Optional[torch.device] = None) -> torch.Tensor:
    """从打包表示反量化回权重矩阵。

    Args:
        packed_data: `turboquant_quantize_packed_full()` 的返回值
        device: 目标设备（默认从 packed_data 推断）

    Returns:
        W_approx: 反量化后的权重矩阵
    """
    # 提取数据
    indices_packed = packed_data["indices_packed"]
    codebook = packed_data["codebook"]
    norms = packed_data["norms"]
    seed = packed_data["seed"]
    group_size = packed_data["group_size"]
    shape = packed_data["shape"]
    bit_width = packed_data["bit_width"]
    rotation = packed_data.get("rotation", "qr")
    orig_dtype_str = packed_data.get("orig_dtype", "torch.float16")

    # 解析原始 dtype
    orig_dtype = getattr(torch, orig_dtype_str.split(".")[-1])

    # 确定设备
    if device is None:
        device = indices_packed.device
    else:
        indices_packed = indices_packed.to(device)
        norms = norms.to(device)
    codebook = codebook.to(device)

    # Step 2: 反量化用 fp32 高精度计算
    codebook = codebook.float()
    norms = norms.float()

    M, N = shape

    # 解包索引
    full_indices = unpack_nbit(indices_packed, bit_width, N)

    # 反量化
    W_approx = torch.zeros((M, N), dtype=torch.float32, device=device)

    # 确保 norms 是二维的
    if norms.dim() == 1:
        norms = norms.unsqueeze(1)

    # 按分组反量化
    group_idx = 0
    for g_start in range(0, N, group_size):
        g_end = min(g_start + group_size, N)
        g_dim = g_end - g_start

        # 提取当前分组的索引和范数
        indices_g = full_indices[:, g_start:g_end]
        norms_g = norms[:, group_idx].unsqueeze(1)
        group_idx += 1

        # 从码本还原
        Y_quant_scaled = codebook[indices_g]

        # 逆缩放
        scale = math.sqrt(g_dim)
        Y_unscaled = Y_quant_scaled / scale

        # 逆旋转
        if rotation == "none":
            W_g_approx = Y_unscaled
        elif rotation == "hadamard":
            W_g_approx = hadamard_rotate_inverse(Y_unscaled, seed=seed + g_start)
        else:  # qr
            Pi = generate_rotation_matrix(g_dim, seed=seed + g_start, device=device)
            W_g_approx = Y_unscaled @ Pi

        # 恢复原始尺度
        W_approx[:, g_start:g_end] = W_g_approx * norms_g

    return W_approx.to(orig_dtype)


@torch.no_grad()
def turboquant_dequantize_packed_rows(
    packed_data: dict,
    row_start: int,
    row_end: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """只反量化指定行范围 [row_start:row_end, :]（优化2版本）

    优化2：减少 Python 循环开销！
    - 预先获取所有旋转矩阵（一次性）
    - 向量化更多操作

    Args:
        packed_data: `turboquant_quantize_packed_full()` 的返回值
        row_start: 起始行（包含）
        row_end: 结束行（不包含）
        device: 目标设备

    Returns:
        W_approx_slice: 反量化后的权重切片，形状为 (row_end-row_start, N)
    """
    # 提取数据
    indices_packed = packed_data["indices_packed"]
    codebook = packed_data["codebook"]
    norms = packed_data["norms"]
    seed = packed_data["seed"]
    group_size = packed_data["group_size"]
    shape = packed_data["shape"]
    bit_width = packed_data["bit_width"]
    rotation = packed_data.get("rotation", "qr")
    orig_dtype_str = packed_data.get("orig_dtype", "torch.float16")

    # 解析原始 dtype
    orig_dtype = getattr(torch, orig_dtype_str.split(".")[-1])

    # 确定设备
    if device is None:
        device = indices_packed.device
    else:
        indices_packed = indices_packed.to(device)
        norms = norms.to(device)
    codebook = codebook.to(device)

    # Step 2: 反量化用 fp32 高精度计算（codebook/norms 从 fp16 转回 fp32）
    # 反量化是参考/离线路径，不追求性能，精度优先
    codebook = codebook.float()
    norms = norms.float()

    M, N = shape

    # 边界检查
    row_start = max(0, row_start)
    row_end = min(M, row_end)
    if row_start >= row_end:
        return torch.zeros((0, N), dtype=orig_dtype, device=device)

    # 只切片需要的行！
    indices_packed_slice = indices_packed[row_start:row_end]  # (row_end-row_start, packed_N)

    # 解包索引（只解包切片的行）
    full_indices_slice = unpack_nbit(indices_packed_slice, bit_width, N)  # (row_end-row_start, N)

    # 切片 norms
    if norms.dim() == 1:
        norms_slice = norms[row_start:row_end].unsqueeze(1)  # (row_end-row_start, 1)
    else:
        norms_slice = norms[row_start:row_end]  # (row_end-row_start, n_groups)

    # 反量化
    num_rows = row_end - row_start
    W_approx = torch.zeros((num_rows, N), dtype=torch.float32, device=device)

    # 优化2：预先计算 group 边界和旋转矩阵
    n_groups = (N + group_size - 1) // group_size
    g_starts = [g for g in range(0, N, group_size)]
    g_ends = [min(g + group_size, N) for g in g_starts]
    g_dims = [g - g_start for g_start, g in zip(g_starts, g_ends)]

    # 预先获取所有需要的旋转矩阵（减少 Python 循环内的开销）
    Pis = []
    scales = []
    for i in range(n_groups):
        g_dim = g_dims[i]
        scale = math.sqrt(g_dim)
        scales.append(scale)
        if rotation == "qr":
            Pi = generate_rotation_matrix(g_dim, seed + g_starts[i], device=device)
            Pis.append(Pi)
        else:
            Pis.append(None)

    # 优化后的循环：减少循环内的操作
    for group_idx in range(n_groups):
        g_start = g_starts[group_idx]
        g_end = g_ends[group_idx]
        g_dim = g_dims[group_idx]

        # 提取当前分组的索引和范数
        indices_g = full_indices_slice[:, g_start:g_end]  # (num_rows, g_dim)
        norms_g = norms_slice[:, group_idx].unsqueeze(1)  # (num_rows, 1)

        # 从码本还原
        Y_quant_scaled = codebook[indices_g]

        # 逆缩放
        Y_unscaled = Y_quant_scaled / scales[group_idx]

        # 逆旋转
        if rotation == "none":
            W_g_approx = Y_unscaled
        elif rotation == "hadamard":
            W_g_approx = hadamard_rotate_inverse(Y_unscaled, seed=seed + g_start)
        else:  # qr
            W_g_approx = Y_unscaled @ Pis[group_idx]

        # 恢复原始尺度
        W_approx[:, g_start:g_end] = W_g_approx * norms_g

    return W_approx.to(orig_dtype)


@torch.no_grad()
def turboquant_dequantize_packed_cols(
    packed_data: dict,
    col_start: int,
    col_end: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """只反量化指定列范围 [:, col_start:col_end]（优化2版本）

    优化2：减少 Python 循环开销！

    Args:
        packed_data: `turboquant_quantize_packed_full()` 的返回值
        col_start: 起始列（包含）
        col_end: 结束列（不包含）
        device: 目标设备

    Returns:
        W_approx_slice: 反量化后的权重切片，形状为 (M, col_end-col_start)
    """
    # 提取数据
    indices_packed = packed_data["indices_packed"]
    codebook = packed_data["codebook"]
    norms = packed_data["norms"]
    seed = packed_data["seed"]
    group_size = packed_data["group_size"]
    shape = packed_data["shape"]
    bit_width = packed_data["bit_width"]
    rotation = packed_data.get("rotation", "qr")
    orig_dtype_str = packed_data.get("orig_dtype", "torch.float16")

    # 解析原始 dtype
    orig_dtype = getattr(torch, orig_dtype_str.split(".")[-1])

    # 确定设备
    if device is None:
        device = indices_packed.device
    else:
        indices_packed = indices_packed.to(device)
        norms = norms.to(device)
    codebook = codebook.to(device)

    # Step 2: 反量化用 fp32 高精度计算
    codebook = codebook.float()
    norms = norms.float()

    M, N = shape

    # 边界检查
    col_start = max(0, col_start)
    col_end = min(N, col_end)
    if col_start >= col_end:
        return torch.zeros((M, 0), dtype=orig_dtype, device=device)

    # 确定哪些 groups 包含我们需要的列
    g_start_first = (col_start // group_size) * group_size
    g_end_last = ((col_end + group_size - 1) // group_size) * group_size
    g_end_last = min(g_end_last, N)

    # 完整解包，但之后只处理相关 groups
    full_indices = unpack_nbit(indices_packed, bit_width, N)

    # 确保 norms 是二维的
    if norms.dim() == 1:
        norms = norms.unsqueeze(1)

    # 反量化
    num_cols = col_end - col_start
    W_approx = torch.zeros((M, num_cols), dtype=torch.float32, device=device)

    # 优化2：预先计算需要处理的 group 信息
    g_starts = []
    g_ends = []
    g_dims = []
    out_g_starts = []
    out_g_ends = []
    in_g_starts = []
    in_g_ends = []
    Pis = []
    scales = []

    for g_start in range(g_start_first, g_end_last, group_size):
        g_end = min(g_start + group_size, N)
        g_dim = g_end - g_start

        # 计算当前 group 和目标列范围的交集
        out_g_start = max(0, g_start - col_start)
        out_g_end = min(g_end - col_start, num_cols)

        if out_g_start >= out_g_end:
            continue

        g_starts.append(g_start)
        g_ends.append(g_end)
        g_dims.append(g_dim)
        out_g_starts.append(out_g_start)
        out_g_ends.append(out_g_end)
        in_g_starts.append(max(0, col_start - g_start))
        in_g_ends.append(min(col_end - g_start, g_dim))

        # 预先获取旋转矩阵
        scale = math.sqrt(g_dim)
        scales.append(scale)
        if rotation == "qr":
            Pi = generate_rotation_matrix(g_dim, seed + g_start, device=device)
            Pis.append(Pi)
        else:
            Pis.append(None)

    # 优化后的循环
    for i in range(len(g_starts)):
        g_start = g_starts[i]
        g_end = g_ends[i]
        g_dim = g_dims[i]
        out_g_start = out_g_starts[i]
        out_g_end = out_g_ends[i]
        in_g_start = in_g_starts[i]
        in_g_end = in_g_ends[i]

        group_idx = g_start // group_size

        # 提取当前分组的索引和范数
        indices_g = full_indices[:, g_start:g_end]
        norms_g = norms[:, group_idx].unsqueeze(1)

        # 从码本还原（完整 group）
        Y_quant_scaled = codebook[indices_g]

        # 逆缩放
        Y_unscaled = Y_quant_scaled / scales[i]

        # 逆旋转
        if rotation == "none":
            W_g_approx = Y_unscaled
        elif rotation == "hadamard":
            W_g_approx = hadamard_rotate_inverse(Y_unscaled, seed=seed + g_start)
        else:  # qr
            W_g_approx = Y_unscaled @ Pis[i]

        # 恢复原始尺度
        W_g_approx_scaled = W_g_approx * norms_g

        # 只保留需要的列切片
        W_approx[:, out_g_start:out_g_end] = W_g_approx_scaled[:, in_g_start:in_g_end]

    return W_approx.to(orig_dtype)
