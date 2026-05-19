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
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
from typing import Any

import torch
import torch.nn as nn


# 当前让 1-15 bit 宽度走 TurboQuant fake-quant。
# bit=0 保留给 DartMoQ 原来的剪枝/置零语义处理；bit=16 表示保持
# base/fp16 权重，不做 fake-quant。
MIN_TURBO_FAKE_QUANT_BIT = 1
MAX_TURBO_FAKE_QUANT_BIT = 15


@dataclass(frozen=True)
class TurboFakeQuantResult:
    """Result returned by quantize_linear_if_turbo_supported.

    handled:
        True 表示当前 bit 已经由这个 backend 处理，调用方不需要再跑 GPTQ。
        False 表示当前 bit 不归 TurboQuant 处理，调用方应该回退到原逻辑。
    bit_width:
        归一化后的 int bit 宽度。
    reason:
        用于日志/调试，说明本次处理或跳过的原因。
    weight_mse / weight_sse:
        TurboQuant fake-quant 前后权重的简单重构误差。这里只衡量权重
        本身的误差，不等价于 GPTQ 的 Hessian/activation-aware loss。
    """

    handled: bool
    bit_width: int
    reason: str
    weight_mse: torch.Tensor | None = None
    weight_sse: torch.Tensor | None = None


def _import_turboquant_quantize():
    """Lazy import the local TurboQuant quantizer.

    这里只从本项目的 turboquant_model 包导入。运行入口应从当前项目
    根目录启动，保证解析到的是仓库内的 TurboQuant 源码。
    """

    # The vendored directory in this repository is named ``turboquant_utils``,
    # while the upstream files use absolute ``turboquant_model.*`` imports.
    # Alias the current package before importing ``.quantize`` so fake-quant
    # uses the local source without requiring a package rename.
    package_name = __package__.split(".")[0]
    package = importlib.import_module(package_name)
    sys.modules.setdefault("turboquant_model", package)
    for module_name in ("codebook", "rotation"):
        local_name = f"{package_name}.{module_name}"
        upstream_name = f"turboquant_model.{module_name}"
        sys.modules.setdefault(upstream_name, importlib.import_module(local_name))

    from .quantize import turboquant_quantize
    return turboquant_quantize


def normalize_bit_width(bit_width: Any) -> int:
    """Convert DartMoQ/Torch bit-width values to a plain int."""

    # DartMoQ 的 Quantizer.bits 通常是 Python int，但有些路径可能传入
    # torch scalar。统一转成 int 后，后面的集合判断和日志更简单。
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

    turboquant_quantize = _import_turboquant_quantize()

    # 记录原 dtype/device，保证写回后不改变 DartMoQ/HF 模型原本的参数格式。
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
    linear.weight.data.copy_(qweight.to(device=orig_device, dtype=orig_dtype))
    return linear


@torch.no_grad()
def quantize_linear_if_turbo_supported(
    linear: nn.Linear,
    bit_width: Any,
    group_size: int | None = 128,
    seed: int = 42,
    rotation: str = "qr",
    zero_bit_policy: str = "zero",
) -> TurboFakeQuantResult:
    """Handle a DartMoQ-selected Linear bit-width when TurboQuant should own it.

    Returns handled=False for unsupported bits so the caller can fall back to
    DartMoQ/GPTQ unchanged.

    zero_bit_policy:
        - "zero": preserve DartMoQ's current bit=0 behavior by zeroing weight;
        - "fallback": return handled=False and let the caller handle bit=0.
    """

    bit = normalize_bit_width(bit_width)

    # bit=0 在 DartMoQ 中表示该权重被剪掉/置零。默认可以在这里直接置零；
    # 当前 dartmoq_utils.py 传入的是 "fallback"，让原 DartMoQ 分支继续处理。
    if bit == 0:
        if zero_bit_policy == "zero":
            linear.weight.data.zero_()
            return TurboFakeQuantResult(True, bit, "zeroed by bit=0 policy")
        if zero_bit_policy == "fallback":
            return TurboFakeQuantResult(False, bit, "bit=0 left to caller")
        raise ValueError(f"unknown zero_bit_policy: {zero_bit_policy!r}")

    if not is_turbo_fake_quant_supported(bit):
        return TurboFakeQuantResult(False, bit, "bit-width left to existing DartMoQ path")

    # 处理成功后返回 handled=True；调用方应该跳过 GPTQ fasterquant。
    orig_weight = linear.weight.data.float().clone()
    turbo_fake_quant_linear(
        linear,
        bit_width=bit,
        group_size=group_size,
        seed=seed,
        rotation=rotation,
    )
    quant_error = (orig_weight - linear.weight.data.float()).pow(2)
    return TurboFakeQuantResult(
        True,
        bit,
        "turboquant fake-quant applied",
        weight_mse=quant_error.mean(),
        weight_sse=quant_error.sum(),
    )
