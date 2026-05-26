"""DartMoQ fake-quant backend built on the local TurboQuant package.

This module implements integration plan 1:

- keep DartMoQ importance analysis and mixed-bit search unchanged;
- after DartMoQ decides the bit-width for a specific nn.Linear;
- use TurboQuant for every 1-15 bit weight approximation;
- keep the module as nn.Linear by writing the dequantized weight back.

The resulting model is still a normal dense model. This is intentional: it
keeps the first integration step small and makes PPL comparisons easy.

中文说明：
这个文件实现的是“方案 1”：只把最终量化阶段的一部分权重近似方式
从 GPTQ 换成 TurboQuant。它不会把 nn.Linear 替换成 TurboQuantLinear，
也不会生成 packed indices / norms / codebook 这种真实压缩推理格式。
因此它适合用来比较 PPL 和量化误差，但不能反映最终模型体积压缩。

论文来源: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
(Zandieh et al., 2025, arXiv:2504.19874)
项目来源：https://github.com/cksac/turboquant-model

"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
from typing import Any
from .quantize import turboquant_quantize

import torch
import torch.nn as nn
import torch.nn.functional as F


# 当前让 1-15 bit 宽度走 TurboQuant fake-quant。
# base/fp16 权重，不做 fake-quant。
MIN_TURBO_FAKE_QUANT_BIT = 1
MAX_TURBO_FAKE_QUANT_BIT = 15

def normalize_bit_width(bit_width: Any) -> int:
    """Convert DartMoQ/Torch bit-width values to a plain int."""

    if isinstance(bit_width, torch.Tensor):
        if bit_width.numel() != 1:
            raise ValueError(f"bit_width tensor must be scalar, got shape {tuple(bit_width.shape)}")
        return int(bit_width.item())
    return int(bit_width)

def get_linear_bit_from_dartmoq_quantizer(gptq_obj: Any) -> int:
    """Read the selected bit-width from DartMoQ's GPTQ wrapper.

    DartMoQ stores the selected bit-width at gptq[name].quantizer.bits after
    Quantizer.configure(...). This helper keeps the call site compact.
    """

    return normalize_bit_width(gptq_obj.quantizer.bits)

def is_turbo_fake_quant_supported(bit_width: Any) -> bool:
    """Return True only for bit-widths intended for TurboQuant fake-quant."""

    bit = normalize_bit_width(bit_width)
    return MIN_TURBO_FAKE_QUANT_BIT <= bit <= MAX_TURBO_FAKE_QUANT_BIT

@torch.no_grad()
def turbo_fake_quant_linear(
    linear: nn.Linear,
    bit_width: Any,
    group_size: int | None = 128,
    seed: int = 42,
    rotation: str = "qr",
    update: bool = True,
) -> nn.Linear:
    """Apply TurboQuant fake-quant to a Linear layer in-place.

    Args:
        linear: target nn.Linear.
        bit_width: positive integer bit-width.
        group_size: group size along in_features. Use 128 to match DartMoQ's
            current GPTQ group size. Use None for full-row groups.
        seed: TurboQuant rotation seed.
        rotation: "qr" is the safest default for arbitrary hidden sizes.

    Returns:
        The same nn.Linear object, with linear.weight.data replaced by the
        dequantized TurboQuant approximation.

    注意：
        这是 fake-quant。TurboQuant 会先量化权重，再把近似权重反量化
        成浮点 tensor 写回 linear.weight.data。模块类型仍然是 nn.Linear，
        因此不会带来 packed 权重的模型大小/显存收益。
    """

    bit = normalize_bit_width(bit_width)
    if not is_turbo_fake_quant_supported(bit):
        raise ValueError(
            f"TurboQuant fake-quant supports only "
            f"{MIN_TURBO_FAKE_QUANT_BIT}-{MAX_TURBO_FAKE_QUANT_BIT} bit, got {bit}"
        )
    if not isinstance(linear, nn.Linear):
        raise TypeError(f"expected nn.Linear, got {type(linear)!r}")

    orig_dtype = linear.weight.data.dtype
    orig_device = linear.weight.data.device

    # turboquant_quantize 返回的是“反量化后的近似权重”，不是 packed 表示。
    # group_size=128 默认对齐 DartMoQ 当前 GPTQ 的 groupsize 设置。
    qweight = turboquant_quantize(
        linear.weight.data,
        bit_width=bit,
        group_size=group_size,
        seed=seed,
        rotation=rotation,
    )

    # 原地 copy，保持 nn.Parameter 对象本身不变，减少对上层模型结构的影响。
    # if update == False, will not update the weight.
    quant_error = (linear.weight.data - qweight).pow(2)
    # quant_error = (linear.weight.data - qweight).abs()
    if update:
        linear.weight.data.copy_(qweight.to(device=orig_device, dtype=orig_dtype))
    return quant_error


@torch.no_grad()
def turboquant_outlier_activation_aware_rates(
    expert: nn.Module,
    flat_states: torch.Tensor,
    bit_width: Any,
    mode: str,
    group_size: int | None = 128,
    seed: int = 42,
    rotation: str = "qr",
) -> torch.Tensor:
    """Compute per-neuron TurboQuant outlier scores for one MoE expert."""

    if mode not in ("aw", "output"):
        raise ValueError(f"unknown TurboQuant outlier mode: {mode!r}")

    up_w = expert.up_proj.weight.data.float()
    gate_w = expert.gate_proj.weight.data.float()
    down_w = expert.down_proj.weight.data.float()

    up_q = turboquant_quantize(
        up_w,
        bit_width=bit_width,
        group_size=group_size,
        seed=seed,
        rotation=rotation,
    ).float()
    gate_q = turboquant_quantize(
        gate_w,
        bit_width=bit_width,
        group_size=group_size,
        seed=seed,
        rotation=rotation,
    ).float()
    down_q = turboquant_quantize(
        down_w,
        bit_width=bit_width,
        group_size=group_size,
        seed=seed,
        rotation=rotation,
    ).float()

    up_out = F.linear(flat_states, up_w)
    gate_out = F.linear(flat_states, gate_w)
    z_w = expert.act_fn(gate_out) * up_out
    zw_norm2 = z_w.pow(2).mean(dim=0)

    if mode == "aw":
        up_loss = (up_w - up_q).pow(2)
        gate_loss = (gate_w - gate_q).pow(2)
        down_loss = (down_w - down_q).pow(2)
        weight_score = up_loss.sum(dim=1) + gate_loss.sum(dim=1) + down_loss.sum(dim=0)
        return weight_score * zw_norm2
    else:
        up_q_out = F.linear(flat_states, up_q)
        gate_q_out = F.linear(flat_states, gate_q)
        z_q = expert.act_fn(gate_q_out) * up_q_out

        zq_norm2 = z_q.pow(2).mean(dim=0)
        zz = (z_w * z_q).mean(dim=0)

        wdown_norm2 = down_w.pow(2).sum(dim=0)
        qdown_norm2 = down_q.pow(2).sum(dim=0)
        down_dot = (down_w * down_q).sum(dim=0)
        rates = zw_norm2 * wdown_norm2 + zq_norm2 * qdown_norm2 - 2 * zz * down_dot
        # rates = ((z_w * down_w - z_q * down_q)).norm2() ??
        return rates.clamp_min(0)
