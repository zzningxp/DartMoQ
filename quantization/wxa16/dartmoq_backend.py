#!/usr/bin/env python3
"""
WxA16 DartMoQ Backend - 真实量化，存储 packed 格式
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear import WxA16Linear, W8A16Linear
from turboquant_utils.quantize import turboquant_quantize_packed_full


@torch.no_grad()
def wxa16_quantize_linear(
    linear: nn.Linear,
    bit_width: int,
    group_size = None,
    seed: int = 42,
    rotation: str = "qr",
    keep_on_gpu: bool = True,
) -> WxA16Linear:
    """
    对 nn.Linear 进行 WxA16 真实量化，返回 WxA16Linear 模块。

    注意：此函数会创建新模块，不会修改原 linear。

    Args:
        linear: 原始 nn.Linear
        bit_width: 1/2/4/8
        group_size: 分组大小
        seed: 随机种子
        rotation: 旋转类型
        keep_on_gpu: 是否保持在 GPU

    Returns:
        WxA16Linear 实例
    """
    if bit_width == 8:
        return W8A16Linear.from_linear(
            linear,
            group_size=group_size,
            seed=seed,
            rotation=rotation,
            keep_on_gpu=keep_on_gpu,
        )
    else:
        return WxA16Linear.from_linear(
            linear,
            bit_width=bit_width,
            group_size=group_size,
            seed=seed,
            rotation=rotation,
            keep_on_gpu=keep_on_gpu,
        )


@torch.no_grad()
def wxa16_quantize_linear_inplace(
    linear: nn.Linear,
    bit_width: int,
    group_size = None,
    seed: int = 42,
    rotation: str = "qr",
    keep_on_gpu: bool = True,
) -> WxA16Linear:
    """
    对 nn.Linear 进行 WxA16 真实量化，原地替换为 WxA16Linear。

    与 fake quant 不同，这里会删除原始 fp16 权重以节省显存。

    Args:
        linear: 原始 nn.Linear（会被销毁）
        bit_width: 1/2/4/8
        group_size: 分组大小
        seed: 随机种子
        rotation: 旋转类型
        keep_on_gpu: 是否保持在 GPU

    Returns:
        WxA16Linear 实例（已经可以直接替换原 linear）
    """
    wxa16_linear = wxa16_quantize_linear(
        linear,
        bit_width=bit_width,
        group_size=group_size,
        seed=seed,
        rotation=rotation,
        keep_on_gpu=keep_on_gpu,
    )

    return wxa16_linear
