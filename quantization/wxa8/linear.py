#!/usr/bin/env python3
"""WxA8 Linear —— attention 路径的 INT8 激活 + 8-bit 均匀码本推理。

前置条件：权重必须用**均匀码本**量化（codebook_type="uniform"，
见 turboquant_utils/quantize.py 的 uniform_codebook 参数）。256 级
Lloyd-Max 非均匀码本塞进 int8 均匀网格会塌掉 61 级（实测等效约
7.6-bit），所以旧 checkpoint 的 8-bit linear 不能转 WxA8 ——
convert_model_to_wxa8 会按码本类型判定，非均匀的保持 W8A16。

与 WxA16Linear 的差别：
  1. 输入先分组旋转 + per-token-per-group 量化（rotate_quantize_fused，
     与 MoE 同一套融合 kernel），cb_step 折进激活 scale
  2. matmul 走 INT8 kernel；均匀码本下 indices 直接就是 int8 权重
     （w_i8 = idx - 128），kernel 免查表
  3. 权重侧 lazy 转 group-first 布局（与 MoE 的 _build_group_first 同一思路）

存储格式与 checkpoint 不变：packed_indices / codebook / norms 与
WxA16Linear 完全同源，加载后 convert_model_to_wxa8 原地切换。
"""

import math

import torch
import torch.nn as nn

from quantization.wxa16.linear import WxA16Linear
from turboquant_utils.rotation import generate_batch_rotation_matrices
from turboquant_utils.triton_kernels import convert_to_group_first
from turboquant_utils.triton_kernels_a8 import (
    rotate_quantize_fused,
    wxa8_matmul_grouped_gf,
)


def uniform_codebook_step(codebook: torch.Tensor) -> float:
    """从均匀码本反推量化步长 step（cb[i] = (i-128)*step）。

    用 (cb[-1] - cb[0]) / 255 推导，对 fp16 存储的舍入不敏感。
    码本不满足均匀性时抛 ValueError（转换前必须用 is_uniform_codebook 判定）。
    """
    cb = codebook.float()
    if cb.numel() != 256:
        raise ValueError(f"8-bit 码本应有 256 级，got {cb.numel()}")
    step = (cb[-1] - cb[0]) / 255.0
    if step <= 0:
        raise ValueError(f"非法码本步长 {step}")
    return float(step)


def is_uniform_codebook(codebook: torch.Tensor, rel_tol: float = 1e-2) -> bool:
    """判定码本是否均匀网格（相邻级差恒定）。

    Lloyd-Max 码本相邻级差从 ~0.01 变到 ~0.4（零点附近极密），均匀性
    判定直接失败 —— 这正是转换 WxA8 的安全阀。
    """
    if codebook.numel() != 256:
        return False
    d = codebook.float()[1:] - codebook.float()[:-1]
    mean = d.mean().abs()
    if mean <= 0:
        return False
    return bool((d - d.mean()).abs().max() < rel_tol * mean)


# 占位 int8 码本（IDENTITY_CB 路径下 kernel 不访问它，但签名要求传指针）
_DUMMY_CB_CACHE = {}


def _dummy_cb(device):
    if device not in _DUMMY_CB_CACHE:
        _DUMMY_CB_CACHE[device] = torch.zeros(256, device=device, dtype=torch.int8)
    return _DUMMY_CB_CACHE[device]


class WxA8Linear(WxA16Linear):
    """WxA8 attention linear（8-bit 均匀码本 + INT8 激活）。"""

    @classmethod
    def from_wxa16(cls, linear: WxA16Linear) -> "WxA8Linear":
        """把 WxA16Linear 原地切到 WxA8（零拷贝，与 MoE 的 from_wxa16 同理）。

        只允许 8-bit + 均匀码本；其他情况抛 ValueError（调用方应先判定）。
        """
        if not isinstance(linear, WxA16Linear):
            raise TypeError(f"期望 WxA16Linear，得到 {type(linear)}")
        if linear.bit_width != 8:
            raise ValueError(f"WxA8Linear 只支持 8-bit，got {linear.bit_width}")
        if not is_uniform_codebook(linear.codebook):
            raise ValueError(
                "非均匀码本（Lloyd-Max）不能转 WxA8：塞进 int8 网格会塌掉 "
                "约 61 级（等效 7.6-bit）。需要均匀码本重新量化。")
        linear.__class__ = cls
        linear._gf_built = False
        return linear

    def _ensure_gf(self, device):
        """lazy 转 group-first 布局 + 预乘 1/sqrt(group_size)（对齐 MoE 的
        _build_group_first 思路；不注册 buffer，state_dict 格式不变）。"""
        if getattr(self, "_gf_built", False):
            return
        if self.in_features % self.group_size != 0:
            raise ValueError(
                f"WxA8Linear 要求 in_features ({self.in_features}) 对齐 "
                f"group_size ({self.group_size})")
        indices_gf, norms_gf = convert_to_group_first(
            self.packed_indices, self.norms, self.group_size, self.bit_width)
        self._indices_gf = indices_gf.to(device).contiguous()
        self._norms_gf = (norms_gf.to(device) / math.sqrt(self.group_size)
                          ).half().contiguous()
        # 原始 buffer 移到 CPU 释放显存（与 MoE 的 offload_original_to_cpu 一致）
        self.packed_indices = self.packed_indices.cpu()
        self.norms = self.norms.cpu()
        self._gf_built = True

    @torch.no_grad()
    def drop_runtime_caches(self):
        """Drop the gf-layout caches (call before moving a layer between GPU
        and CPU, so old GPU tensors do not stay resident).

        In per-layer sequential eval, _indices_gf/_norms_gf keep references
        to the previous round's GPU tensors, which cannot be freed after the
        buffers move. After clearing, _ensure_gf rebuilds on the current
        device.
        """
        self._indices_gf = None
        self._norms_gf = None
        self._gf_built = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """INT8 激活推理：旋转+量化（融合）→ INT8 matmul → bias → 原 dtype。"""
        orig_shape = x.shape
        batch_size = seq_len = None
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            x = x.reshape(-1, self.in_features)

        self._ensure_gf(x.device)
        src = x if x.dtype == torch.float16 else x.half()

        gs = self.group_size
        G = self.in_features // gs
        rot = generate_batch_rotation_matrices(
            gs, self.seed, G, stride=gs, device=src.device, dtype=torch.float16)
        x_i8, x_scale = rotate_quantize_fused(
            src, rot, gs, G,
            extra_scale=uniform_codebook_step(self.codebook.to(x.device)))

        out = wxa8_matmul_grouped_gf(
            x_i8, x_scale, self._indices_gf, _dummy_cb(x.device), self._norms_gf,
            gs, G, self.bit_width, identity_cb=True)
        del x_i8, x_scale

        if self.bias is not None:
            out = out + self.bias.to(x.device, out.dtype)

        if batch_size is not None and seq_len is not None:
            out = out.reshape(batch_size, seq_len, self.out_features)
        return out.to(x.dtype)
