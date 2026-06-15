"""cosine similarity 的高敏感神经元保护实验。

TurboQuant 低比特量化后，如果把 top-sensitive neuron 恢复为 fp16，
expert 局部输出方向是否会更接近原始 fp16 输出。

python viz/expert_cosine.py \
    --model "$modelname" \
    --rank-mode turboquant_innerproduct \
    --quantmode turboquant \
    --bit 2 \
    --layer-start 17 \
    --num-layers 2 \
    --expert-start 0 \
    --num-experts 2 \
    --pdf

"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DARTMOQ_ROOT = Path(__file__).resolve().parents[1]
if str(DARTMOQ_ROOT) not in sys.path:
    sys.path.insert(0, str(DARTMOQ_ROOT))

from dartmoq_utils import (  # type: ignore
    analyze_gptq_quant_outlier,
    analyze_turboquant_outlier_activation_aware,
)
from eval_dartmoq import load_model  # type: ignore
from data_utils import get_loaders  # type: ignore
from turboquant_utils.dartmoq_backend import collect_expert_activation_inputs  # type: ignore
from turboquant_utils.quantize import turboquant_quantize  # type: ignore
from viz._cache_io import (  # type: ignore
    model_label, resolve_model_id, resolve_model_path,
    load_expert_cosine_cache, save_expert_cosine_cache, discover_layers,
    load_layer,
)
from viz.seed_stability import Target, _get_expert_weights  # type: ignore


DEFAULT_QUANTMODE = "turboquant"
DEFAULT_RANK_MODE = "turboquant_innerproduct"
DEFAULT_BIT = 2
DEFAULT_SEEDS = [0, 42, 84, 126, 168]
DEFAULT_PRESERVE_FRACS = [0.0, 0.01, 0.02, 0.05]
OUT_ROOT = "plot/expert_cosine"


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _module_device(module: nn.Module) -> torch.device:
    for param in module.parameters():
        return param.device
    for buf in module.buffers():
        return buf.device
    return torch.device("cpu")


def _move_to_device(obj, device: torch.device):
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_move_to_device(x, device) for x in obj)
    if isinstance(obj, list):
        return [_move_to_device(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


def _save_fig(fig: plt.Figure, path: Path, save_pdf: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    if save_pdf:
        pdf_path = path.with_suffix(".pdf")
        fig.savefig(pdf_path, dpi=200, bbox_inches="tight")
        print(f"[expert_cosine] saved {path} and {pdf_path}")
    else:
        print(f"[expert_cosine] saved {path}")
    plt.close(fig)


def _concat_chunks(chunks: Sequence[torch.Tensor], device: torch.device, max_rows: Optional[int]) -> torch.Tensor:
    if not chunks:
        return torch.empty(0, 0, dtype=torch.float32, device=device)
    cat = torch.cat([chunk.float() for chunk in chunks], dim=0)
    if max_rows is not None and cat.shape[0] > max_rows:
        cat = cat[:max_rows]
    return cat.to(device=device, dtype=torch.float32)


def _n_experts(model) -> int:
    cfg = model.config
    if hasattr(cfg, "num_experts"):
        return int(cfg.num_experts)
    if hasattr(cfg, "n_routed_experts"):
        return int(cfg.n_routed_experts)
    raise ValueError("model config does not expose num_experts / n_routed_experts")


def _forward_decoder_layer(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    position_embeddings,
) -> torch.Tensor:
    kwargs = {"hidden_states": hidden_states}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    if position_embeddings is not None:
        kwargs["position_embeddings"] = position_embeddings
    outputs = layer(**kwargs)
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def _capture_first_layer_inputs(model, dataloader, nsamples: int, device: torch.device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    dtype = next(iter(model.parameters())).dtype
    bsz = 1
    inps = torch.zeros(
        (nsamples // bsz, bsz, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device="cpu",
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None, "position_embeddings": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            cache["position_embeddings"] = kwargs.get("position_embeddings")
            raise ValueError

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

    layers[0] = Catcher(layers[0])
    try:
        with torch.no_grad():
            for batch in dataloader:
                if cache["i"] >= inps.shape[0]:
                    break
                try:
                    model(batch[0].to(device))
                except ValueError:
                    pass
    finally:
        layers[0] = layers[0].module
        model.config.use_cache = use_cache

    return (
        inps.squeeze(1),
        cache["attention_mask"],
        cache["position_ids"],
        cache["position_embeddings"],
    )


@torch.no_grad()
def _collect_layer_mlp_inputs(
    model,
    layer_idx: int,
    dataloader,
    nsamples: int,
    device: torch.device,
) -> torch.Tensor:
    inps, attention_mask, position_ids, position_embeddings = _capture_first_layer_inputs(
        model, dataloader, nsamples, device
    )

    first_layer_device = _module_device(model.model.layers[0])
    current = inps.to(first_layer_device)
    for idx in range(layer_idx):
        layer = model.model.layers[idx]
        layer_device = _module_device(layer)
        current = current.to(layer_device)
        layer_attention_mask = _move_to_device(attention_mask, layer_device)
        layer_position_ids = _move_to_device(position_ids, layer_device)
        layer_position_embeddings = _move_to_device(position_embeddings, layer_device)
        outs = torch.zeros_like(current, device=layer_device)
        for sample_idx in range(current.shape[0]):
            outs[sample_idx:sample_idx + 1] = _forward_decoder_layer(
                layer,
                current[sample_idx:sample_idx + 1],
                layer_attention_mask,
                layer_position_ids,
                layer_position_embeddings,
            )
        current = outs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    layer = model.model.layers[layer_idx]
    layer_device = _module_device(layer)
    current = current.to(layer_device)
    layer_attention_mask = _move_to_device(attention_mask, layer_device)
    layer_position_ids = _move_to_device(position_ids, layer_device)
    layer_position_embeddings = _move_to_device(position_embeddings, layer_device)
    batchsize = current.shape[0]
    residual = current
    hidden_states_inorm = layer.input_layernorm(current)
    attn_out = torch.zeros_like(hidden_states_inorm, device=layer_device)
    for sample_idx in range(batchsize):
        if layer_position_embeddings is not None:
            attn_out[sample_idx:sample_idx + 1] = layer.self_attn(
                hidden_states=hidden_states_inorm[sample_idx:sample_idx + 1],
                attention_mask=layer_attention_mask,
                position_ids=layer_position_ids,
                position_embeddings=layer_position_embeddings,
            )[0]
        else:
            attn_out[sample_idx:sample_idx + 1] = layer.self_attn(
                hidden_states=hidden_states_inorm[sample_idx:sample_idx + 1],
                attention_mask=layer_attention_mask,
                position_ids=layer_position_ids,
            )[0]
    hidden_states = residual + attn_out
    mlp_inputs = layer.post_attention_layernorm(hidden_states)
    return mlp_inputs.to(device)


@torch.no_grad()
def _collect_expert_token_inputs(
    model,
    target: Target,
    device: torch.device,
    max_tokens: Optional[int],
    layer_inputs: Optional[torch.Tensor] = None,
    dataloader=None,
    nsamples: Optional[int] = None,
) -> torch.Tensor:
    if layer_inputs is None:
        if dataloader is None or nsamples is None:
            raise ValueError("dataloader and nsamples are required when layer_inputs is not provided")
        mlp_inputs = _collect_layer_mlp_inputs(model, target.layer_idx, dataloader, nsamples, device)
    else:
        mlp_inputs = layer_inputs
    layer = model.model.layers[target.layer_idx]
    captured = collect_expert_activation_inputs(layer, mlp_inputs, _n_experts(model), if_dense=False)
    expert_capture = captured[target.expert_idx]
    tokens = _concat_chunks(expert_capture["up_proj"], device, max_tokens)
    if tokens.numel() == 0:
        raise ValueError(
            f"No routed tokens collected for L{target.layer_idx} E{target.expert_idx}; "
            "try increasing nsamples or pick a more active expert."
        )
    return tokens


def _expert_forward_with_weights(
    expert: nn.Module,
    tokens: torch.Tensor,
    up_w: torch.Tensor,
    gate_w: torch.Tensor,
    down_w: torch.Tensor,
) -> torch.Tensor:
    up = F.linear(tokens, up_w)
    gate = expert.act_fn(F.linear(tokens, gate_w))
    hidden = gate * up
    return F.linear(hidden, down_w)


def _turboquant_weight(
    weight: torch.Tensor,
    bit_width: int,
    seed: int,
) -> torch.Tensor:
    return turboquant_quantize(
        weight,
        bit_width=bit_width,
        group_size=128,
        seed=seed,
        rotation="qr",
    ).float()


def _mask_indices_by_frac(sensitivity: np.ndarray, frac: float, largest: bool) -> np.ndarray:
    n = sensitivity.shape[0]
    if frac <= 0:
        return np.empty(0, dtype=int)
    k = max(1, int(np.ceil(frac * n)))
    order = np.argsort(sensitivity)
    if largest:
        return order[::-1][:k]
    return order[:k]


def _mask_indices_original_order(sensitivity: np.ndarray, frac: float) -> np.ndarray:
    n = sensitivity.shape[0]
    if frac <= 0:
        return np.empty(0, dtype=int)
    k = max(1, int(np.ceil(frac * n)))
    return np.arange(k)


def _restore_sensitive_neurons(
    w_q: torch.Tensor,
    w_ref: torch.Tensor,
    neuron_idx: np.ndarray,
    axis: int,
) -> torch.Tensor:
    out = w_q.clone()
    idx = torch.as_tensor(np.array(neuron_idx, copy=True), dtype=torch.long, device=out.device)
    if axis == 0:
        out[idx, :] = w_ref[idx, :]
    elif axis == 1:
        out[:, idx] = w_ref[:, idx]
    else:
        raise ValueError(f"unsupported axis {axis}")
    return out


def _rank_mode_to_analysis(rank_mode: str) -> tuple[str, str]:
    if rank_mode == "gptq_quant_outlier":
        return "gptq", ""
    turbo_map = {
        "turboquant_innerproduct": "innerproduct",
        "turboquant_innerproduct_fea": "innerproduct",
        "turboquant_iipl": "iipl",
        "turboquant_iipl_fea": "iipl",
        "turboquant_diagonal": "diagonal",
        "turboquant_hessian": "hessian",
        "turboquant_qjl_sensitivity": "qjl_sensitivity",
    }
    if rank_mode in turbo_map:
        return "turboquant", turbo_map[rank_mode]
    raise ValueError(f"Unsupported rank_mode for inline sensitivity: {rank_mode}")


def _compute_layer_sensitivity_inline(
    model,
    layer_idx: int,
    hidden_states: torch.Tensor,
    bit: int,
    rank_mode: str,
    sensitivity_seed: int,
):
    layer = model.model.layers[layer_idx]
    ori_expert_num = _n_experts(model)
    quantmode, turbo_mode = _rank_mode_to_analysis(rank_mode)
    if quantmode == "gptq":
        return analyze_gptq_quant_outlier(
            layer,
            layer_idx,
            hidden_states,
            ori_expert_num,
            wbits=bit,
            quantmode="gptq",
            save_path=None,
            seed=sensitivity_seed,
        )
    return analyze_turboquant_outlier_activation_aware(
        layer,
        layer_idx,
        hidden_states,
        ori_expert_num,
        wbits=bit,
        mode=turbo_mode,
        if_dense=False,
        save_path=None,
        use_activation_hooks=not rank_mode.endswith("_fea"),
        seed=sensitivity_seed,
    )


def _resolve_targets_with_sensitivity(
    model,
    model_id: str,
    dataloader,
    device: torch.device,
    layers: Optional[Sequence[int]],
    experts: Optional[Sequence[int]],
    rank_mode: str,
    quantmode: str,
    bit: int,
):
    layer_inputs_cache: dict[int, torch.Tensor] = {}
    sensitivity_map: dict[tuple[int, int], np.ndarray] = {}
    targets: list[Target] = []

    for layer_idx in layers:
        layer_inputs = _collect_layer_mlp_inputs(model, layer_idx, dataloader, 64, device)
        layer_inputs_cache[layer_idx] = layer_inputs

        layer_cache = load_layer(model_id, layer_idx, quantmode, rank_mode, bits=(bit,))
        if layer_cache is not None and bit in layer_cache.by_bit:
            all_rates = layer_cache.by_bit[bit]
        else:
            all_rates = _compute_layer_sensitivity_inline(
                model,
                layer_idx,
                layer_inputs,
                bit,
                rank_mode,
                42,
            )

        expert_scores = [
            (expert_idx, float(np.asarray(rates).sum()))
            for expert_idx, rates in enumerate(all_rates)
        ]
        if experts:
            expert_set = set(experts)
            expert_scores = [(e, s) for e, s in expert_scores if e in expert_set]

        for expert_idx, score in expert_scores:
            target = Target(layer_idx, expert_idx, score)
            targets.append(target)
            if layer_cache is not None and bit in layer_cache.by_bit:
                sensitivity_map[(layer_idx, expert_idx)] = np.asarray(all_rates[expert_idx], dtype=np.float32)
            else:
                sensitivity_map[(layer_idx, expert_idx)] = (
                    torch.as_tensor(all_rates[expert_idx]).detach().cpu().numpy().astype(np.float32, copy=False)
                )

    return targets, sensitivity_map, layer_inputs_cache


@dataclass
class CosineMetric:
    """cosine 越接近 1，说明量化后输出方向越接近原始 fp16 输出。
    one_minus_cosine 越小越好；画图时用它更直观地表示“角度漂移”。
    """
    cosine: float
    one_minus_cosine: float


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    """对多个 TurboQuant seed 的结果求均值和标准差，用来观察 seed 波动。"""
    arr = np.asarray(list(values), dtype=float)
    return float(arr.mean()), float(arr.std())


def _cosine_metric(ref: torch.Tensor, pred: torch.Tensor) -> CosineMetric:
    """比较原始 fp16 expert 输出和量化/恢复后的 expert 输出方向是否一致。"""
    cosine = float(F.cosine_similarity(ref.reshape(1, -1), pred.reshape(1, -1)).item())
    return CosineMetric(cosine=cosine, one_minus_cosine=1.0 - cosine)


def plot_from_cache(summary: dict, out_path: str, save_pdf: bool = False) -> None:
    """从 cache 中的 summary 直接画图，不需要模型"""
    model_id = summary["model_id"]
    targets = summary["targets"]

    n_plots = len(targets)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    # 定义不同策略的颜色和标记
    strategy_styles = {
        "top-sensitive": {"color": "#3a7ca5", "marker": "o", "label": "Top-sensitive"},
        "original-order": {"color": "#d62728", "marker": "s", "label": "Original order"},
    }

    for ax, target_data in zip(axes_flat, targets):
        layer = target_data["layer"]
        expert = target_data["expert"]
        per_strategy = target_data.get("strategies", {})

        if per_strategy:
            # 新格式：多个策略
            for strategy_name, per_frac in per_strategy.items():
                style = strategy_styles.get(strategy_name, {"color": "#333", "marker": "o", "label": strategy_name})
                x = [row["preserve_frac"] for row in per_frac]
                y = [row["mean_one_minus_cosine"] for row in per_frac]
                yerr = [row["std_one_minus_cosine"] for row in per_frac]
                ax.errorbar(x, y, yerr=yerr, marker=style["marker"], color=style["color"], label=style["label"], capsize=3)
            ax.legend(fontsize=8)
        else:
            # 兼容旧格式
            per_frac = target_data["rows"]
            x = [row["preserve_frac"] for row in per_frac]
            y = [row["mean_one_minus_cosine"] for row in per_frac]
            yerr = [row["std_one_minus_cosine"] for row in per_frac]
            ax.errorbar(x, y, yerr=yerr, marker="o", color="#3a7ca5")

        ax.set_xlabel("Preserved neuron fraction")
        ax.set_ylabel("1 - cosine similarity")
        ax.set_title(f"{model_label(model_id)} L{layer} E{expert}")
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)

    fig.suptitle(f"Cosine drift after fp16 protection - {model_label(model_id)}", y=1.01)
    _save_fig(fig, Path(out_path), save_pdf)
    print(f"[expert_cosine] plotted from cache, output to {out_path}")


def _get_out_path(
    model_id: str, layers: Sequence[int], experts: Optional[Sequence[int]],
    bit: int, rank_mode: str
) -> str:
    """构建输出路径：plot/expert_cosine/{model_id}_L{layers_str}_E{experts_str}_b{bit}_{rank_mode}"""
    layer_str = "_".join(str(l) for l in sorted(layers))
    if experts:
        expert_str = "_".join(str(e) for e in sorted(experts))
    else:
        expert_str = "all"
    safe_rank = rank_mode.replace("/", "_").replace("\\", "_")
    safe_rank = "".join(c if c.isalnum() or c in "_-" else "_" for c in safe_rank)
    return os.path.join(OUT_ROOT, f"{model_id}_L{layer_str}_E{expert_str}_b{bit}_{safe_rank}.png")


def _evaluate_strategy(
    expert: nn.Module,
    tokens: torch.Tensor,
    up_w: torch.Tensor,
    gate_w: torch.Tensor,
    down_w: torch.Tensor,
    ref_out: torch.Tensor,
    sensitivity: np.ndarray,
    strategy_name: str,
    quant_mode: str,
    bit: int,
    seeds: Sequence[int],
    preserve_fracs: Sequence[float],
) -> list[dict]:
    """评估一个选择策略，返回 per_frac 结果"""
    per_frac = []
    for frac in preserve_fracs:
        if strategy_name == "top-sensitive":
            keep_idx = _mask_indices_by_frac(sensitivity, frac, largest=True)
        elif strategy_name == "original-order":
            keep_idx = _mask_indices_original_order(sensitivity, frac)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        seed_metrics = []
        for seed in seeds:
            up_q = _turboquant_weight(up_w, bit, seed)
            gate_q = _turboquant_weight(gate_w, bit, seed)
            down_q = _turboquant_weight(down_w, bit, seed)

            up_mix = _restore_sensitive_neurons(up_q, up_w, keep_idx, axis=0)
            gate_mix = _restore_sensitive_neurons(gate_q, gate_w, keep_idx, axis=0)
            down_mix = _restore_sensitive_neurons(down_q, down_w, keep_idx, axis=1)

            pred = _expert_forward_with_weights(expert, tokens, up_mix, gate_mix, down_mix)
            metrics = _cosine_metric(ref_out, pred)
            seed_metrics.append({"seed": int(seed), **asdict(metrics)})

        mean_cos, std_cos = _mean_std(m["cosine"] for m in seed_metrics)
        mean_drift, std_drift = _mean_std(m["one_minus_cosine"] for m in seed_metrics)
        per_frac.append({
            "preserve_frac": frac,
            "n_preserved": int(len(keep_idx)),
            "mean_cosine": mean_cos,
            "std_cosine": std_cos,
            "mean_one_minus_cosine": mean_drift,
            "std_one_minus_cosine": std_drift,
            "per_seed": seed_metrics,
        })
    return per_frac


def run_cosine_experiment(
    model_id: str,
    layers: Sequence[int],
    experts: Optional[Sequence[int]],
    quantmode: str,
    rank_mode: str,
    bit: int,
    seeds: Sequence[int],
    preserve_fracs: Sequence[float],
    save_pdf: bool = False,
) -> dict:
    """运行 cosine 实验，返回 cache_summary"""
    device = _resolve_device()
    model_path = resolve_model_path(model_id)
    model, tokenizer = load_model(model_path)
    model.eval()

    dataloader, _ = get_loaders(
        "wikitext2",
        nsamples=64,
        seed=42,
        tokenizer=tokenizer,
        seqlen=model.seqlen,
    )

    targets, sensitivity_map, layer_inputs_cache = _resolve_targets_with_sensitivity(
        model, model_id, dataloader, device, layers, experts, rank_mode, quantmode, bit
    )
    if not targets:
        raise ValueError("No targets found from the selected sensitivity source.")

    summary = {
        "model_id": model_id,
        "rank_mode": rank_mode,
        "bit": bit,
        "seeds": [int(seed) for seed in seeds],
        "preserve_fracs": [float(x) for x in preserve_fracs],
        "targets": [],
    }

    results = []
    for target in targets:
        sensitivity = sensitivity_map[(target.layer_idx, target.expert_idx)]

        tokens = _collect_expert_token_inputs(
            model,
            target,
            device,
            4096,
            layer_inputs=layer_inputs_cache.get(target.layer_idx),
            dataloader=dataloader,
            nsamples=64,
        )

        layer = model.model.layers[target.layer_idx]
        expert = layer.mlp.experts[target.expert_idx]
        up_w, gate_w, down_w = _get_expert_weights(model, target, device)
        ref_out = _expert_forward_with_weights(expert, tokens, up_w, gate_w, down_w)

        # 评估多个策略
        strategies = ["top-sensitive", "original-order"]
        per_strategy = {}
        for strategy in strategies:
            print(f"[expert_cosine] L{target.layer_idx} E{target.expert_idx} - {strategy}")
            per_strategy[strategy] = _evaluate_strategy(
                expert, tokens, up_w, gate_w, down_w, ref_out, sensitivity,
                strategy, quantmode, bit, seeds, preserve_fracs
            )

        summary["targets"].append({
            "layer": target.layer_idx,
            "expert": target.expert_idx,
            "strategies": per_strategy,
        })
        results.append((target.layer_idx, target.expert_idx, per_strategy))

    # 绘图
    n_plots = len(results)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    strategy_styles = {
        "top-sensitive": {"color": "#3a7ca5", "marker": "o", "label": "Top-sensitive"},
        "original-order": {"color": "#d62728", "marker": "s", "label": "Original order"},
    }

    for ax, (layer, expert, per_strategy) in zip(axes_flat, results):
        for strategy_name, per_frac in per_strategy.items():
            style = strategy_styles.get(strategy_name, {"color": "#333", "marker": "o", "label": strategy_name})
            x = [row["preserve_frac"] for row in per_frac]
            y = [row["mean_one_minus_cosine"] for row in per_frac]
            yerr = [row["std_one_minus_cosine"] for row in per_frac]
            ax.errorbar(x, y, yerr=yerr, marker=style["marker"], color=style["color"], label=style["label"], capsize=3)
        ax.legend(fontsize=8)
        ax.set_xlabel("Preserved neuron fraction")
        ax.set_ylabel("1 - cosine similarity")
        ax.set_title(f"{model_label(model_id)} L{layer} E{expert}")
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)

    fig.suptitle(f"Cosine drift after fp16 protection - {model_label(model_id)}", y=1.01)

    out_path = _get_out_path(model_id, layers, experts, bit, rank_mode)
    _save_fig(fig, Path(out_path), save_pdf)

    return summary


def _select_layers_from_start(
    model_id: str, quantmode: str, rank_mode: str,
    layer_start: int, num_layers: int
) -> list[int]:
    """根据 layer_start 和 num_layers 选择 layers 列表"""
    available_layers = discover_layers(quantmode, rank_mode, model_id)
    if not available_layers:
        raise ValueError(f"No cache layers found for {model_id}")

    if layer_start == -1:
        return available_layers[-num_layers:]
    else:
        start_idx = 0
        for i, l in enumerate(available_layers):
            if l >= layer_start:
                start_idx = i
                break
        end_idx = min(start_idx + num_layers, len(available_layers))
        return available_layers[start_idx:end_idx]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepseek-v1-moe-16b",
                        help="short cache id OR full model path")
    parser.add_argument("--quantmode", default=DEFAULT_QUANTMODE, choices=["turboquant", "gptq"])
    parser.add_argument("--rank-mode", default=DEFAULT_RANK_MODE)
    parser.add_argument("--bit", type=int, default=DEFAULT_BIT)
    parser.add_argument("--layer-start", type=int, default=0,
                        help="start layer index (inclusive) to use; -1 means the last num-layers layers")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of layers to use, default: 1")
    parser.add_argument("--expert-start", type=int, default=0,
                        help="start expert index (inclusive) to use; -1 means the last num-experts experts")
    parser.add_argument("--num-experts", type=int, default=1,
                        help="number of experts to use per layer, default: 1")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--preserve-fracs", nargs="+", default=DEFAULT_PRESERVE_FRACS)
    parser.add_argument("--overwrite-cache", action="store_true",
                        help="overwrite existing cache and recompute")
    parser.add_argument("--pdf", action="store_true", default=False,
                        help="also save PDF copies alongside PNGs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_id = resolve_model_id(args.model)
    task_name = "cosine_similarity"

    layers = _select_layers_from_start(
        model_id, args.quantmode, args.rank_mode, args.layer_start, args.num_layers
    )
    print(f"[expert_cosine] selected layers: {layers}")

    expert_start = args.expert_start
    num_experts = args.num_experts
    experts = None
    if expert_start >= 0:
        experts = list(range(expert_start, expert_start + num_experts))
        print(f"[expert_cosine] selected experts: {experts}")

    # 定义我们使用的策略
    strategies = ["top-sensitive", "original-order"]
    cache = load_expert_cosine_cache(
        model_id, args.quantmode, args.rank_mode, args.bit, task_name, layers, experts,
        seeds=args.seeds, preserve_fracs=[float(x) for x in args.preserve_fracs],
        strategies=strategies
    )
    if cache is not None and not args.overwrite_cache:
        print(f"[expert_cosine] using cache for {model_id} L{layers} E{experts}")
        out_path = _get_out_path(model_id, layers, experts, args.bit, args.rank_mode)
        plot_from_cache(cache, out_path, args.pdf)
        return

    if cache is not None and args.overwrite_cache:
        print(f"[expert_cosine] overwriting existing cache for {model_id} L{layers} E{experts}")

    summary = run_cosine_experiment(
        model_id=model_id,
        layers=layers,
        experts=experts,
        quantmode=args.quantmode,
        rank_mode=args.rank_mode,
        bit=args.bit,
        seeds=args.seeds,
        preserve_fracs=[float(x) for x in args.preserve_fracs],
        save_pdf=args.pdf,
    )

    save_expert_cosine_cache(
        summary, model_id, args.quantmode, args.rank_mode, args.bit, task_name, layers, experts
    )


if __name__ == "__main__":
    main()
