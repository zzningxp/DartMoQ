#!/usr/bin/env python3
"""Router / shared-expert conventions for the five supported MoE models.

Different model families return different values from their MoE gates
(dispatched on ``model.config.model_type``):

  - deepseek / deepseek_v2 (dsv1/dsv2): gate returns a 3-tuple
    (topk_idx, topk_weight, aux_loss)
  - deepseek_v3 (moonlight): gate returns a 2-tuple (topk_idx, topk_weight)
  - qwen3_moe: with transformers 4.57 the gate is an nn.Linear returning
    logits (softmax + topk, then weight renormalization); with transformers
    5.x it returns a 3-tuple (router_logits, router_scores, router_indices)
  - olmoe: gate is an nn.Linear returning logits (softmax + topk, no renorm)

Shared-expert conventions:

  - deepseek family: ``shared_experts`` is a single MLP whose intermediate
    size is n_shared times larger; its output is added ungated
  - qwen3_moe (when a shared expert exists): ``shared_expert`` +
    ``shared_expert_gate`` with a sigmoid gate
  - olmoe / models without a shared expert: none

The quantize path (dartmoq_layer_reconstruct) and the checkpoint load path
(dartmoq_quant_io) use the same conventions; router_mode / shared_mode are
written to meta.json for restore. A new model family needs at most one new
table entry, no forward-logic changes.
"""

import torch


# model_type -> (router_mode, shared_mode)
MODEL_CONVENTIONS = {
    "deepseek": ("deepseek_tuple3", "sum"),
    "deepseek_v2": ("deepseek_tuple3", "sum"),
    "deepseek_v3": ("deepseek_tuple2", "sum"),
    "qwen3_moe": ("qwen3_moe", "gated"),
    "olmoe": ("softmax_topk", "none"),
}

# model_type -> whether the DecoderLayer expects the mlp to return a
# (hidden_states, router_logits) 2-tuple (olmoe unpacks the tuple to compute
# its aux loss; every other supported model assigns the plain tensor).
# This convention controls the packed MoE forward's return type.
MLP_RETURNS_TUPLE = {
    "olmoe": True,
}


def detect_conventions(config):
    """Return (router_mode, shared_mode) from config.model_type.

    Unknown model types fall back to the generic conventions
    (softmax_topk / none).
    """
    mt = getattr(config, "model_type", "")
    return MODEL_CONVENTIONS.get(mt, ("softmax_topk", "none"))


def mlp_returns_tuple(config):
    """Whether the DecoderLayer expects the mlp to return a 2-tuple."""
    mt = getattr(config, "model_type", "")
    return MLP_RETURNS_TUPLE.get(mt, False)


def router_forward(gate, hidden_states_3d, x_2d, mode, top_k):
    """Unified router forward; returns (topk_weights, topk_indices, router_logits).

    topk_weights / topk_indices have shape (N, top_k). router_logits is the
    raw gate output (passed through for logits-style conventions, None for
    tuple-style conventions) and is only consumed by layers that need it for
    aux-loss bookkeeping.

    DeepSeek-family gates require 3D input (bsz, seq, h), so the gate is
    always fed hidden_states_3d; the gate flattens it internally to (N, top_k).
    x_2d carries N / dtype info only.
    """
    gate_output = gate(hidden_states_3d)

    if mode == "deepseek_tuple3":
        # (topk_idx, topk_weight, aux_loss)
        topk_indices, topk_weights, _ = gate_output
        router_logits = None
    elif mode == "deepseek_tuple2":
        topk_indices, topk_weights = gate_output
        router_logits = None
    elif mode == "qwen3_moe":
        if isinstance(gate_output, tuple):
            # transformers 5.x: (router_logits, router_scores, router_indices)
            router_logits, topk_weights, topk_indices = gate_output
        else:
            # transformers 4.x: nn.Linear logits; softmax + topk + renormalize
            router_logits = gate_output
            router_probs = gate_output.softmax(dim=-1)
            topk_weights, topk_indices = router_probs.topk(top_k, dim=-1)
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    else:  # "softmax_topk" (olmoe and generic)
        router_logits = gate_output
        router_probs = gate_output.softmax(dim=-1)
        topk_weights, topk_indices = router_probs.topk(top_k, dim=-1)

    return topk_weights, topk_indices, router_logits


def shared_forward(moe, x_2d, mode):
    """Shared-expert forward; returns the increment to add to the MoE output,
    or None when the model has no shared expert.

    - "gated": shared_out * sigmoid(shared_expert_gate(x))  (qwen3-style)
    - "sum":   shared_experts(x), added ungated               (deepseek family)
    """
    if mode == "gated":
        shared_expert = getattr(moe, "shared_expert", None)
        shared_expert_gate = getattr(moe, "shared_expert_gate", None)
        if shared_expert is not None and shared_expert_gate is not None:
            shared_out = shared_expert(x_2d)
            shared_gate_val = torch.sigmoid(shared_expert_gate(x_2d))
            return shared_out * shared_gate_val
        return None

    if mode == "sum":
        shared_experts = getattr(moe, "shared_experts", None)
        if shared_experts is not None:
            return shared_experts(x_2d)
        return None

    return None
