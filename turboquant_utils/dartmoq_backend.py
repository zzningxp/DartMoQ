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

import time
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

    # if update == False, will not update the weight.
    quant_error = (linear.weight.data - qweight).pow(2)
    # quant_error = (linear.weight.data - qweight).abs()
    if update:
        linear.weight.data.copy_(qweight.to(device=orig_device, dtype=orig_dtype))
    return quant_error


@torch.no_grad()
def collect_expert_activation_inputs(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    ori_expert_num: int,
    if_dense: bool = False,
) -> list[dict[str, torch.Tensor | dict[str, int]]]:
    """Collect real per-expert projection inputs without keeping them on GPU."""

    if if_dense or ori_expert_num == 1:
        assert ori_expert_num == 1, "dense model n == 1"
        experts = [layer.mlp]
    else:
        experts = [layer.mlp.experts[i] for i in range(ori_expert_num)]

    captured = [
        {"up_proj": [], "gate_proj": [], "down_proj": []}
        for _ in range(ori_expert_num)
    ]
    sample_counts = [
        {"up_proj": 0, "gate_proj": 0, "down_proj": 0}
        for _ in range(ori_expert_num)
    ]

    def capture(expert_idx: int, proj_name: str):
        def hook(_, inp, _out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            captured[expert_idx][proj_name].append(x.cpu())
            sample_counts[expert_idx][proj_name] += x.shape[0]

        return hook

    handles = []
    for expert_idx, expert in enumerate(experts):
        handles.append(expert.up_proj.register_forward_hook(capture(expert_idx, "up_proj")))
        handles.append(expert.gate_proj.register_forward_hook(capture(expert_idx, "gate_proj")))
        handles.append(expert.down_proj.register_forward_hook(capture(expert_idx, "down_proj")))

    try:
        for sample_idx in range(hidden_states.shape[0]):
            layer.mlp(hidden_states[sample_idx].unsqueeze(0))
    finally:
        for handle in handles:
            handle.remove()

    collected = []
    for expert_idx, _expert in enumerate(experts):
        collected.append({
            "up_proj": captured[expert_idx]["up_proj"],
            "gate_proj": captured[expert_idx]["gate_proj"],
            "down_proj": captured[expert_idx]["down_proj"],
            "sample_counts": sample_counts[expert_idx],
        })
    return collected


def _iter_activation_chunks(
    value: torch.Tensor | list[torch.Tensor] | None,
    device: torch.device,
    chunk_size: int = 4096,
):
    if value is None:
        return
    if isinstance(value, list):
        for tensor in value:
            for start in range(0, tensor.shape[0], chunk_size):
                yield tensor[start:start + chunk_size].to(device=device, non_blocking=True)
        return
    flat = value.reshape(-1, value.shape[-1])
    for start in range(0, flat.shape[0], chunk_size):
        yield flat[start:start + chunk_size].to(device=device, dtype=torch.float32, non_blocking=True)


def _activation_numel(value: torch.Tensor | list[torch.Tensor] | None) -> int:
    if value is None:
        return 0
    if isinstance(value, list):
        return sum(tensor.shape[0] for tensor in value)
    return value.reshape(-1, value.shape[-1]).shape[0]


def _stream_sum_sq(
    value: torch.Tensor | list[torch.Tensor] | None,
    device: torch.device,
    width: int,
) -> tuple[torch.Tensor, int]:
    total = torch.zeros(width, device=device, dtype=torch.float32)
    count = 0
    for chunk in _iter_activation_chunks(value, device):
        chunk = chunk.float()
        total += chunk.pow(2).sum(dim=0)
        count += chunk.shape[0]
    return total, count


def _stream_linear_sq_mean(
    value: torch.Tensor | list[torch.Tensor] | None,
    weight: torch.Tensor,
) -> torch.Tensor:
    total = torch.zeros(weight.shape[0], device=weight.device, dtype=torch.float32)
    count = 0
    for chunk in _iter_activation_chunks(value, weight.device):
        out = F.linear(chunk.float(), weight)
        total += out.pow(2).sum(dim=0)
        count += out.shape[0]
        del out, chunk
    if count == 0:
        return total
    return total / count


@torch.no_grad()
def turboquant_outlier_activation_aware_rates(
    expert: nn.Module,
    flat_states: torch.Tensor,
    bit_width: Any,
    mode: str,
    group_size: int | None = 128,
    seed: int = 42,
    rotation: str = "qr",
    hessian_samples: int | None = None,
    sketch_dim: int = 64,
    activation_inputs: dict[str, torch.Tensor] | None = None,
    activation_sample_counts: dict[str, int] | None = None,
) -> torch.Tensor:
    """Compute per-neuron TurboQuant outlier scores for one MoE expert."""

    if mode not in ("iipl", "innerproduct", "diagonal", "hessian", "qjl_sensitivity", "mse"):
        raise ValueError(f"unknown TurboQuant outlier mode: {mode!r}")
    if sketch_dim <= 0:
        raise ValueError("sketch_dim must be > 0")

    up_w = expert.up_proj.weight.data.float()
    gate_w = expert.gate_proj.weight.data.float()
    down_w = expert.down_proj.weight.data.float()

    if activation_inputs is None:
        up_inputs = flat_states
        gate_inputs = flat_states
        down_inputs = None
    else:
        up_inputs = activation_inputs["up_proj"]
        gate_inputs = activation_inputs["gate_proj"]
        down_inputs = activation_inputs["down_proj"]

    if _activation_numel(up_inputs) == 0 or _activation_numel(gate_inputs) == 0:
        return torch.zeros(up_w.shape[0], device=up_w.device, dtype=up_w.dtype)

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

    up_loss = (up_w - up_q).pow(2)
    gate_loss = (gate_w - gate_q).pow(2)
    down_loss = (down_w - down_q).pow(2)

    if activation_inputs is None:
        up_out = F.linear(up_inputs, up_w)
        gate_out = F.linear(gate_inputs, gate_w)
        z_w = expert.act_fn(gate_out) * up_out
        if z_w.numel() == 0:
            return torch.zeros(up_w.shape[0], device=up_w.device, dtype=up_w.dtype)
        zw_norm2 = z_w.pow(2).mean(dim=0)

        if mode == "mse":
            weight_score = up_loss.sum(dim=1) + gate_loss.sum(dim=1) + down_loss.sum(dim=0)
            return weight_score

        if mode == "iipl":
            weight_score = up_loss.sum(dim=1) + gate_loss.sum(dim=1) + down_loss.sum(dim=0)
            return weight_score * zw_norm2

        if mode == "diagonal":
            up_energy = up_inputs.pow(2).mean(dim=0)
            gate_energy = gate_inputs.pow(2).mean(dim=0)
            up_score = (up_loss * up_energy.unsqueeze(0)).sum(dim=1)
            gate_score = (gate_loss * gate_energy.unsqueeze(0)).sum(dim=1)
            down_score = down_loss.sum(dim=0) * zw_norm2
            return up_score + gate_score + down_score

        if mode == "hessian":
            sample_count = hessian_samples if hessian_samples is not None else flat_states.shape[0]
            up_hdiag = up_inputs.pow(2).sum(dim=0) * (2.0 / max(sample_count, 1))
            gate_hdiag = gate_inputs.pow(2).sum(dim=0) * (2.0 / max(sample_count, 1))
            z_hdiag = z_w.float().pow(2).sum(dim=0) * (2.0 / max(sample_count, 1))

            def hessian_score(W: torch.Tensor, Q: torch.Tensor, Hdiag: torch.Tensor) -> torch.Tensor:
                Hdiag = Hdiag.float().clamp_min(0)
                dead = Hdiag == 0
                Hdiag[dead] = 1
                W = W.clone()
                Q = Q.clone()
                W[:, dead] = 0
                Q[:, dead] = 0
                return (W - Q).pow(2) * Hdiag.unsqueeze(0)

            up_score = hessian_score(up_w, up_q, up_hdiag).sum(dim=1)
            gate_score = hessian_score(gate_w, gate_q, gate_hdiag).sum(dim=1)
            down_score = hessian_score(down_w, down_q, z_hdiag).sum(dim=0)
            return up_score + gate_score + down_score

        if mode == "qjl_sensitivity":
            up_residual = up_w - up_q
            gate_residual = gate_w - gate_q
            down_residual = down_w - down_q

            up_score = F.linear(up_inputs, up_residual).pow(2).mean(dim=0)
            gate_score = F.linear(gate_inputs, gate_residual).pow(2).mean(dim=0)
            z_energy = z_w.float().pow(2).mean(dim=0)

            generator = torch.Generator(device=down_w.device)
            generator.manual_seed(int(seed) + 104729)
            sketch = torch.empty(
                sketch_dim,
                down_residual.shape[0],
                device=down_residual.device,
                dtype=down_residual.dtype,
            )
            sketch.bernoulli_(0.5, generator=generator)
            sketch.mul_(2).sub_(1).div_(sketch_dim ** 0.5)

            down_sketch = torch.matmul(sketch, down_residual)
            down_score = z_energy * down_sketch.pow(2).sum(dim=0)
            return (up_score + gate_score + down_score).clamp_min(0)

        if mode == "innerproduct":
            up_q_out = F.linear(up_inputs, up_q)
            gate_q_out = F.linear(gate_inputs, gate_q)
            z_q = expert.act_fn(gate_q_out) * up_q_out

            zq_norm2 = z_q.pow(2).mean(dim=0)
            zz = (z_w * z_q).mean(dim=0)

            wdown_norm2 = down_w.pow(2).sum(dim=0)
            qdown_norm2 = down_q.pow(2).sum(dim=0)
            down_dot = (down_w * down_q).sum(dim=0)
            rates = zw_norm2 * wdown_norm2 + zq_norm2 * qdown_norm2 - 2 * zz * down_dot
            return rates.clamp_min(0)
        
        assert False, f"Unknown mode {mode}"

    else:
        up_sq, up_count = _stream_sum_sq(up_inputs, up_w.device, up_w.shape[1])
        gate_sq, gate_count = _stream_sum_sq(gate_inputs, gate_w.device, gate_w.shape[1])
        z_sq, z_count = _stream_sum_sq(down_inputs, down_w.device, down_w.shape[1])
        if z_count == 0:
            return torch.zeros(up_w.shape[0], device=up_w.device, dtype=up_w.dtype)
        zw_norm2 = z_sq / z_count

        if mode == "mse":
            weight_score = up_loss.sum(dim=1) + gate_loss.sum(dim=1) + down_loss.sum(dim=0)
            return weight_score

        if mode == "iipl":
            weight_score = up_loss.sum(dim=1) + gate_loss.sum(dim=1) + down_loss.sum(dim=0)
            return weight_score * zw_norm2

        if mode == "diagonal":
            up_energy = up_sq / max(up_count, 1)
            gate_energy = gate_sq / max(gate_count, 1)
            up_score = (up_loss * up_energy.unsqueeze(0)).sum(dim=1)
            gate_score = (gate_loss * gate_energy.unsqueeze(0)).sum(dim=1)
            down_score = down_loss.sum(dim=0) * zw_norm2
            return up_score + gate_score + down_score

        if mode == "hessian":
            sample_count = hessian_samples if hessian_samples is not None else flat_states.shape[0]
            up_count = activation_sample_counts.get("up_proj", sample_count) if activation_sample_counts else sample_count
            gate_count = activation_sample_counts.get("gate_proj", sample_count) if activation_sample_counts else sample_count
            down_count = activation_sample_counts.get("down_proj", sample_count) if activation_sample_counts else sample_count
            up_hdiag = up_sq * (2.0 / max(up_count, 1))
            gate_hdiag = gate_sq * (2.0 / max(gate_count, 1))
            z_hdiag = z_sq * (2.0 / max(down_count, 1))

            def hessian_score(W: torch.Tensor, Q: torch.Tensor, Hdiag: torch.Tensor) -> torch.Tensor:
                Hdiag = Hdiag.float().clamp_min(0)
                dead = Hdiag == 0
                Hdiag[dead] = 1
                W = W.clone()
                Q = Q.clone()
                W[:, dead] = 0
                Q[:, dead] = 0
                return (W - Q).pow(2) * Hdiag.unsqueeze(0)

            up_score = hessian_score(up_w, up_q, up_hdiag).sum(dim=1)
            gate_score = hessian_score(gate_w, gate_q, gate_hdiag).sum(dim=1)
            down_score = hessian_score(down_w, down_q, z_hdiag).sum(dim=0)
            return up_score + gate_score + down_score

        if mode == "qjl_sensitivity":
            up_residual = up_w - up_q
            gate_residual = gate_w - gate_q
            down_residual = down_w - down_q

            up_score = _stream_linear_sq_mean(up_inputs, up_residual)
            gate_score = _stream_linear_sq_mean(gate_inputs, gate_residual)
            z_energy = zw_norm2

            generator = torch.Generator(device=down_w.device)
            generator.manual_seed(int(seed) + 104729)
            sketch = torch.empty(
                sketch_dim,
                down_residual.shape[0],
                device=down_residual.device,
                dtype=down_residual.dtype,
            )
            sketch.bernoulli_(0.5, generator=generator)
            sketch.mul_(2).sub_(1).div_(sketch_dim ** 0.5)

            down_sketch = torch.matmul(sketch, down_residual)
            down_score = z_energy * down_sketch.pow(2).sum(dim=0)
            return (up_score + gate_score + down_score).clamp_min(0)

        if mode == "innerproduct":
            zq_sum = torch.zeros(down_w.shape[1], device=down_w.device, dtype=torch.float32)
            zz_sum = torch.zeros(down_w.shape[1], device=down_w.device, dtype=torch.float32)
            count = 0
            for up_chunk, gate_chunk, down_chunk in zip(
                _iter_activation_chunks(up_inputs, up_w.device),
                _iter_activation_chunks(gate_inputs, gate_w.device),
                _iter_activation_chunks(down_inputs, down_w.device),
            ):
                rows = min(up_chunk.shape[0], gate_chunk.shape[0], down_chunk.shape[0])
                up_chunk = up_chunk[:rows].float()
                gate_chunk = gate_chunk[:rows].float()
                down_chunk = down_chunk[:rows].float()
                up_q_out = F.linear(up_chunk, up_q)
                gate_q_out = F.linear(gate_chunk, gate_q)
                z_q = expert.act_fn(gate_q_out) * up_q_out
                zq_sum += z_q.pow(2).sum(dim=0)
                zz_sum += (down_chunk * z_q).sum(dim=0)
                count += rows
                del up_chunk, gate_chunk, down_chunk, up_q_out, gate_q_out, z_q

            if count == 0:
                return torch.zeros(up_w.shape[0], device=up_w.device, dtype=up_w.dtype)
            zq_norm2 = zq_sum / count
            zz = zz_sum / count

            wdown_norm2 = down_w.pow(2).sum(dim=0)
            qdown_norm2 = down_q.pow(2).sum(dim=0)
            down_dot = (down_w * down_q).sum(dim=0)
            rates = zw_norm2 * wdown_norm2 + zq_norm2 * qdown_norm2 - 2 * zz * down_dot
            return rates.clamp_min(0)
        
        assert False, f"Unknown mode {mode}"