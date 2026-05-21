"""DartMoQ fake-quant backend built on the local TurboQuant package.

This module implements integration plan 1:

- keep DartMoQ importance analysis and mixed-bit search unchanged;
- after DartMoQ decides the bit-width for a specific nn.Linear;
- use TurboQuant for every 1-15 bit weight approximation;
- keep the module as nn.Linear by writing the dequantized weight back.

The resulting model is still a normal dense model. This is intentional: it
keeps the first integration step small and makes PPL comparisons easy.

中文说明：
这个文件实现的是"方案 1"：只把最终量化阶段的一部分权重近似方式
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

import torch
import torch.nn as nn

from .quantize import turboquant_quantize


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
    neuron_direction: str = None,
) -> torch.Tensor:
    """Apply TurboQuant fake-quant to a Linear layer in-place.

    Args:
        linear: target nn.Linear.
        bit_width: positive integer bit-width.
        group_size: group size along in_features. Use 128 to match DartMoQ's
            current GPTQ group size. Use None for full-row groups.
        seed: TurboQuant rotation seed.
        rotation: "qr" is the safest default for arbitrary hidden sizes.
        update: whether to update the linear weight in-place.
        neuron_direction: "up" or "down" for per-neuron quantization when update=False.
            "up": (hidden_size, neuron_size) -> split along neuron dim as (hidden_size, 1)
            "down": (neuron_size, hidden_size) -> split along neuron dim as (1, hidden_size)

    Returns:
        quant_error: quantization error with the same shape as linear.weight.

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
    weight = linear.weight.data

    if update:
        qweight = turboquant_quantize(
            weight,
            bit_width=bit,
            group_size=group_size,
            seed=seed,
            rotation=rotation,
        )
        quant_error = (weight - qweight).pow(2)
        linear.weight.data.copy_(qweight.to(device=orig_device, dtype=orig_dtype))
    else:
        quant_error = torch.zeros_like(weight)

        if neuron_direction == "down":
            hidden_size, neuron_size = weight.shape
            for i in range(neuron_size):
                # 取出第 i 个 neuron 的列向量，保持 (hidden_size, 1) 形状
                w_slice = weight[:, i:i+1]
                q_slice = turboquant_quantize(
                    w_slice,
                    bit_width=bit,
                    group_size=group_size,
                    seed=seed,
                    rotation=rotation,
                )
                quant_error[:, i:i+1] = (w_slice - q_slice).pow(2)
        elif neuron_direction == "up" or neuron_direction == "gate":
            neuron_size, hidden_size = weight.shape
            for i in range(neuron_size):
                # 取出第 i 个 neuron 的行向量，保持 (1, hidden_size) 形状
                w_slice = weight[i:i+1, :]
                q_slice = turboquant_quantize(
                    w_slice,
                    bit_width=bit,
                    group_size=group_size,
                    seed=seed,
                    rotation=rotation,
                )
                quant_error[i:i+1, :] = (w_slice - q_slice).pow(2)
                # if i % 64 == 0:
                #     print(f"{neuron_direction} {i} {w_slice.shape} {q_slice.shape} quant_error[i:i+1, :]: {quant_error[i:i+1, :].sum()}")
        else:
            raise ValueError(f"neuron_direction must be 'up' or 'down' when update=False, got {neuron_direction}")

    return quant_error

