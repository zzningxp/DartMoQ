"""
MoE expert importance analysis utilities.

This module is intentionally not a runnable script. DartMoQ already owns model
loading, calibration data loading, and layer-wise activation capture. The public
entry point here only ranks neurons for one existing expert from one layer.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


LINEAR_SOFTMAX_ROUTER_MODEL_TYPES = ("olmoe", "qwen3", "qwen3_moe")
CALLABLE_TOPK_ROUTER_MODEL_TYPES = ("deepseek", "deepseek_v2", "deepseek_v3", "moonlight")
SUPPORTED_PROJECT_MODEL_TYPES = (
    *LINEAR_SOFTMAX_ROUTER_MODEL_TYPES,
    *CALLABLE_TOPK_ROUTER_MODEL_TYPES,
)


@torch.no_grad()
def analyze_expert_importance(
    layer: nn.Module,
    expert_idx: int,
    inps: torch.Tensor,
    model_type: str,
    top_k: Optional[int] = None,
    reduce_ratio: float = 1.0,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """
    Return per-neuron importance rates for one routed expert.

    Args:
        layer: Decoder layer containing ``layer.mlp`` with ``gate`` and
            ``experts``.
        expert_idx: Index of the original expert to rank.
        inps: Hidden states captured before the MoE block, shaped
            ``[samples, seqlen, hidden_size]``.
        model_type: HF config model type. Only project-supported MoE types are
            accepted: olmoe, qwen3, qwen3_moe, deepseek, deepseek_v2,
            deepseek_v3, and moonlight.
        top_k: Number of routed experts per token. If omitted, this is inferred
            from the module/config when possible.
        reduce_ratio: Blend between activation L2 and Linf statistics.
            ``0`` uses only L2, ``1`` uses only Linf.
        chunk_size: Number of flattened tokens processed at a time.

    Returns:
        A 1D tensor shaped ``[expert_intermediate_size]``. Larger values mean
        more important neurons.
    """
    if not 0.0 <= float(reduce_ratio) <= 1.0:
        raise ValueError("reduce_ratio must be in [0, 1]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    model_type = _normalize_model_type(model_type)
    moe = _get_moe_module(layer)
    experts = moe.experts
    expert = experts[expert_idx]
    device = inps.device

    up_weight = expert.up_proj.weight
    gate_weight = expert.gate_proj.weight
    down_weight = expert.down_proj.weight
    act_fn = getattr(expert, "act_fn", F.silu)

    d_ff = up_weight.shape[0]
    base_energy = (down_weight.float() ** 2).sum(dim=0)
    exp_l2_square = torch.zeros(d_ff, device=device, dtype=torch.float32)
    exp_linf_square = torch.zeros(d_ff, device=device, dtype=torch.float32)

    flat_inps = inps.reshape(-1, inps.shape[-1])
    for start in range(0, flat_inps.shape[0], chunk_size):
        hidden_states = flat_inps[start:start + chunk_size]
        router_scores = _expert_router_scores(
            moe=moe,
            hidden_states=hidden_states,
            expert_idx=expert_idx,
            model_type=model_type,
            top_k=top_k,
        ).to(dtype=hidden_states.dtype)

        expert_mask = router_scores > 0
        if torch.count_nonzero(expert_mask) == 0:
            continue
        hidden_states = hidden_states[expert_mask]

        pre_act = F.linear(hidden_states, up_weight)
        gate = F.linear(hidden_states, gate_weight)
        activations = act_fn(gate) * pre_act
        activations = activations.float()

        exp_l2_square += (activations ** 2).sum(dim=0)
        exp_linf_square = torch.maximum(
            exp_linf_square,
            torch.max(torch.abs(activations), dim=0)[0] ** 2,
        )

    exp_stat = (1 - reduce_ratio) * exp_l2_square + reduce_ratio * exp_linf_square
    return (base_energy.to(device=device) * exp_stat).to(device=device)


def _get_moe_module(layer: nn.Module) -> nn.Module:
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
        return layer.mlp
    if hasattr(layer, "block_sparse_moe") and hasattr(layer.block_sparse_moe, "experts"):
        return layer.block_sparse_moe
    raise NotImplementedError("Could not find an MoE module on the given layer")


def _expert_router_scores(
    moe: nn.Module,
    hidden_states: torch.Tensor,
    expert_idx: int,
    model_type: str,
    top_k: Optional[int],
) -> torch.Tensor:
    if top_k is None:
        top_k = _infer_top_k(moe)

    if model_type in LINEAR_SOFTMAX_ROUTER_MODEL_TYPES:
        router_logits = moe.gate(hidden_states)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        top_values, top_indices = torch.topk(
            router_probs,
            k=min(top_k, router_probs.shape[-1]),
            dim=-1,
        )
        if _should_normalize_topk(moe):
            top_values = top_values / (top_values.sum(dim=-1, keepdim=True) + 1e-20)
        expert_scores = torch.zeros_like(router_probs)
        expert_scores.scatter_(dim=1, index=top_indices, src=top_values)
        return expert_scores[:, expert_idx]

    if model_type not in CALLABLE_TOPK_ROUTER_MODEL_TYPES:
        raise NotImplementedError(
            f"Unsupported model_type for importance analysis: {model_type}. "
            f"Supported project MoE types: {SUPPORTED_PROJECT_MODEL_TYPES}"
        )

    gate_hidden_states = hidden_states
    if gate_hidden_states.dim() == 2:
        gate_hidden_states = gate_hidden_states.unsqueeze(0)

    top_indices, top_values, _ = moe.gate(gate_hidden_states)
    top_indices = top_indices.reshape(-1, top_indices.shape[-1])
    top_values = top_values.reshape(-1, top_values.shape[-1]).float()
    return torch.where(
        top_indices == expert_idx,
        top_values,
        torch.zeros_like(top_values),
    ).sum(dim=-1)


def _normalize_model_type(model_type: str) -> str:
    normalized = model_type.lower().replace("-", "_")
    if normalized not in SUPPORTED_PROJECT_MODEL_TYPES:
        raise NotImplementedError(
            f"Unsupported model_type for importance analysis: {model_type}. "
            f"Supported project MoE types: {SUPPORTED_PROJECT_MODEL_TYPES}"
        )
    return normalized


def _infer_top_k(moe: nn.Module) -> int:
    if hasattr(moe, "top_k"):
        return int(moe.top_k)
    if hasattr(moe, "num_experts_per_tok"):
        return int(moe.num_experts_per_tok)
    if hasattr(moe, "config") and hasattr(moe.config, "num_experts_per_tok"):
        return int(moe.config.num_experts_per_tok)
    return 1


def _should_normalize_topk(moe: nn.Module) -> bool:
    if hasattr(moe, "norm_topk_prob"):
        return bool(moe.norm_topk_prob)
    if hasattr(moe, "config") and hasattr(moe.config, "norm_topk_prob"):
        return bool(moe.config.norm_topk_prob)
    return True
