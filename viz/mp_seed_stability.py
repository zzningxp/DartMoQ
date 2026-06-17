"""Multi-scheme seed stability analysis with IPE (Inner Product Error) metric.

This module implements the analysis for comparing three quantization schemes:
1. all 2bit
2. high-sensitivity-high-bit (32222221): top 1/8 3bit, bottom 1/8 1bit, middle 2bit
3. high-sensitivity-low-bit (12222223): top 1/8 1bit, bottom 1/8 3bit, middle 2bit

Usage:
    python viz/mp_seed_stability.py --model deepseek-v1-moe-16b --layers 17 18 --num-experts 4 --num-seeds 16
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquant_utils.quantize import turboquant_quantize
from viz._cache_io import (
    apply_paper_style,
    discover_layers,
    load_layer,
    model_label,
    resolve_model_id,
    resolve_model_path,
)
from viz.expert_cosine import (
    Target,
    _collect_layer_mlp_inputs,
    _collect_expert_token_inputs,
    _get_expert_weights,
    _resolve_device,
    _module_device,
)
from eval_dartmoq import load_model
from data_utils import get_loaders

OUT_ROOT = "plot/mp_seed_stability"
DEFAULT_MODEL = "deepseek-v1-moe-16b"
DEFAULT_QUANTMODE = "turboquant"
DEFAULT_RANK_MODE = "turboquant_innerproduct"
DEFAULT_BIT = 2
DEFAULT_SEED_INTERV = 42


@dataclass
class SchemeResult:
    """Results for one quantization scheme."""
    scheme_name: str
    ipe_values: List[float]


@dataclass
class ExpertResult:
    """Results for one expert across all schemes."""
    layer_idx: int
    expert_idx: int
    total_sensitivity: float
    scheme_results: Dict[str, SchemeResult]


def select_targets(
    model_id: str,
    quantmode: str,
    rank_mode: str,
    bit: int,
    layers: Optional[Sequence[int]],
    num_experts_per_layer: int,
    max_targets: int,
) -> List[Target]:
    """Select target experts - take first N experts by index (not sorted by sensitivity)."""
    rows: List[Target] = []

    for layer_idx in layers:
        layer = load_layer(model_id, layer_idx, quantmode, rank_mode, bits=(bit,))
        if layer is None or bit not in layer.by_bit:
            continue
        expert_scores = [
            (expert_idx, float(np.asarray(rates, dtype=float).sum()))
            for expert_idx, rates in enumerate(layer.by_bit[bit])
        ]

        for expert_idx, score in expert_scores[:num_experts_per_layer]:
            rows.append(Target(layer_idx, expert_idx, score))
            if len(rows) >= max_targets:
                return rows
    return rows


@torch.no_grad()
def compute_ipe(
    expert: nn.Module,
    tokens: torch.Tensor,
    up_w: torch.Tensor,
    gate_w: torch.Tensor,
    down_w: torch.Tensor,
    bit_width: int,
    seed: int,
    neuron_bit_mask: Optional[torch.Tensor] = None,
) -> float:
    """Compute IPE (Inner Product Error) for the whole expert."""
    device = up_w.device

    flat_tokens = tokens.to(device=device, dtype=torch.float32)

    up_out_ori = F.linear(flat_tokens, up_w)
    gate_out_ori = F.linear(flat_tokens, gate_w)

    if neuron_bit_mask is not None:
        up_q = torch.zeros_like(up_w)
        gate_q = torch.zeros_like(gate_w)
        down_q = torch.zeros_like(down_w)

        unique_bits = torch.unique(neuron_bit_mask)
        for bit in unique_bits:
            mask = (neuron_bit_mask == bit)
            if bit == 0:
                up_q[mask, :] = up_w[mask, :]
                gate_q[mask, :] = gate_w[mask, :]
                down_q[:, mask] = down_w[:, mask]
            else:
                up_q[mask, :] = turboquant_quantize(
                    up_w[mask, :], bit_width=int(bit), group_size=128, seed=seed, rotation="qr"
                )
                gate_q[mask, :] = turboquant_quantize(
                    gate_w[mask, :], bit_width=int(bit), group_size=128, seed=seed, rotation="qr"
                )
                down_q[:, mask] = turboquant_quantize(
                    down_w[:, mask], bit_width=int(bit), group_size=128, seed=seed, rotation="qr"
                )
    else:
        up_q = turboquant_quantize(up_w, bit_width=bit_width, group_size=128, seed=seed, rotation="qr")
        gate_q = turboquant_quantize(gate_w, bit_width=bit_width, group_size=128, seed=seed, rotation="qr")
        down_q = turboquant_quantize(down_w, bit_width=bit_width, group_size=128, seed=seed, rotation="qr")

    up_out_q = F.linear(flat_tokens, up_q)
    gate_out_q = F.linear(flat_tokens, gate_q)

    up_error = (up_out_ori - up_out_q).pow(2).mean(dim=0)
    gate_error = (gate_out_ori - gate_out_q).pow(2).mean(dim=0)

    z_ori = expert.act_fn(gate_out_ori) * up_out_ori
    down_out_ori = F.linear(z_ori, down_w)
    down_out_q = F.linear(z_ori, down_q)
    down_error = (down_out_ori - down_out_q).pow(2).mean(dim=0)

    total_ipe = float(up_error.sum().item() + gate_error.sum().item() + down_error.sum().item())
    return total_ipe


def create_bit_mask(sensitivity: np.ndarray, scheme: str = "32222221") -> np.ndarray:
    """Create bit mask for quantization schemes."""
    n = len(sensitivity)
    eighth = n // 8

    sorted_indices = np.argsort(sensitivity)[::-1]
    bit_mask = np.ones(n, dtype=int) * 2

    if scheme == "32222221":
        bit_mask[sorted_indices[:eighth]] = 3
        bit_mask[sorted_indices[-eighth:]] = 1
    elif scheme == "12222223":
        bit_mask[sorted_indices[:eighth]] = 1
        bit_mask[sorted_indices[-eighth:]] = 3

    return bit_mask


def print_expert_result(expert_result: ExpertResult):
    """Print results for a single expert."""
    print(f"\n  --- L{expert_result.layer_idx} E{expert_result.expert_idx} Results (Total Sensitivity: {expert_result.total_sensitivity:.4e})")
    scheme_names = ["all_2bit", "32222221", "12222223"]
    print(f"    {'Scheme':<15} {'IPE':<15}")
    print(f"    {'-' * 30}")
    for scheme_name in scheme_names:
        scheme_result = expert_result.scheme_results[scheme_name]
        ipe_arr = np.array(scheme_result.ipe_values)
        print(f"    {scheme_name:<15} {ipe_arr.mean():<15.4e}")


def evaluate_expert(
    model,
    target: Target,
    sensitivity: np.ndarray,
    tokens: torch.Tensor,
    seeds: Sequence[int],
    device: torch.device,
) -> ExpertResult:
    """Evaluate all three schemes for one expert."""
    print(f"  Evaluating L{target.layer_idx} E{target.expert_idx}")

    layer = model.model.layers[target.layer_idx]
    expert = layer.mlp.experts[target.expert_idx]
    up_w, gate_w, down_w = _get_expert_weights(model, target, device)

    scheme_results = {}

    print(f"    all_2bit")
    ipe_values = []
    for seed in seeds:
        ipe = compute_ipe(expert, tokens, up_w, gate_w, down_w, 2, seed)
        ipe_values.append(ipe)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    scheme_results["all_2bit"] = SchemeResult("all_2bit", ipe_values)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"    32222221")
    bit_mask_32222221 = create_bit_mask(sensitivity, scheme="32222221")
    ipe_values = []
    for seed in seeds:
        ipe = compute_ipe(
            expert, tokens, up_w, gate_w, down_w, 2, seed,
            neuron_bit_mask=torch.tensor(bit_mask_32222221, device=device)
        )
        ipe_values.append(ipe)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    scheme_results["32222221"] = SchemeResult("32222221", ipe_values)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"    12222223")
    bit_mask_12222223 = create_bit_mask(sensitivity, scheme="12222223")
    ipe_values = []
    for seed in seeds:
        ipe = compute_ipe(
            expert, tokens, up_w, gate_w, down_w, 2, seed,
            neuron_bit_mask=torch.tensor(bit_mask_12222223, device=device)
        )
        ipe_values.append(ipe)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    scheme_results["12222223"] = SchemeResult("12222223", ipe_values)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    expert_result = ExpertResult(
        layer_idx=target.layer_idx,
        expert_idx=target.expert_idx,
        total_sensitivity=target.total_sensitivity,
        scheme_results=scheme_results,
    )

    print_expert_result(expert_result)

    return expert_result


def print_statistics(expert_results: List[ExpertResult]):
    """Print detailed statistics for all experts and schemes."""
    print("\n" + "=" * 100)
    print("DETAILED STATISTICS")
    print("=" * 100)

    layer_results = {}
    for result in expert_results:
        if result.layer_idx not in layer_results:
            layer_results[result.layer_idx] = []
        layer_results[result.layer_idx].append(result)

    for layer_idx in sorted(layer_results.keys()):
        print(f"\n{'=' * 100}")
        print(f"LAYER {layer_idx}")
        print(f"{'=' * 100}")

        layer_experts = layer_results[layer_idx]

        print(f"\n--- Layer {layer_idx} Averages:")
        scheme_names = ["all_2bit", "32222221", "12222223"]
        print(f"  {'Scheme':<15} {'IPE':<15}")
        print(f"  {'-' * 30}")

        for scheme_name in scheme_names:
            all_ipe = []
            for expert_result in layer_experts:
                sr = expert_result.scheme_results[scheme_name]
                all_ipe.extend(sr.ipe_values)
            print(f"  {scheme_name:<15} {np.mean(all_ipe):<15.4e}")

    print(f"\n{'=' * 100}")
    print(f"OVERALL AVERAGES")
    print(f"{'=' * 100}")
    scheme_names = ["all_2bit", "32222221", "12222223"]
    print(f"\n  {'Scheme':<15} {'IPE':<15}")
    print(f"  {'-' * 30}")

    for scheme_name in scheme_names:
        all_ipe = []
        for expert_result in expert_results:
            sr = expert_result.scheme_results[scheme_name]
            all_ipe.extend(sr.ipe_values)
        print(f"  {scheme_name:<15} {np.mean(all_ipe):<15.4e}")


def plot_results(
    model_id: str,
    expert_results: List[ExpertResult],
    out_dir: str,
    save_pdf: bool = False,
):
    """Plot results for all experts in one figure."""
    apply_paper_style()

    layer_results = {}
    for result in expert_results:
        if result.layer_idx not in layer_results:
            layer_results[result.layer_idx] = []
        layer_results[result.layer_idx].append(result)

    scheme_colors = {
        "all_2bit": "#3cb44b",
        "32222221": "#911eb4",
        "12222223": "#f58231",
    }

    scheme_display_names = {
        "all_2bit": "All 2bit",
        "32222221": "32222221",
        "12222223": "12222223",
    }

    scheme_order = ["all_2bit", "32222221", "12222223"]
    n_schemes = len(scheme_order)

    all_experts = []
    for layer_idx in sorted(layer_results.keys()):
        all_experts.extend(layer_results[layer_idx])

    n_cols = len(all_experts)
    n_rows = 1

    # 这里就是要长条形的子图，以使得 boxplot 更加清晰
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10), squeeze=False)
    axes = axes[0]

    for expert_j, er in enumerate(all_experts):
        ax = axes[expert_j]

        data_list = []
        for scheme_name in scheme_order:
            sr = er.scheme_results[scheme_name]
            data_list.append(sr.ipe_values)

        bp = ax.boxplot(
            data_list,
            positions=np.arange(n_schemes),
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            boxprops=dict(linewidth=1.2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker='o', markersize=4, alpha=0.6)
        )

        for patch, scheme_name in zip(bp['boxes'], scheme_order):
            patch.set_facecolor(scheme_colors[scheme_name])
            patch.set_alpha(0.7)

        ax.set_ylabel(f"IPE", fontsize=10)
        ax.set_title(f"L{er.layer_idx}, E{er.expert_idx}", fontsize=11)
        ax.set_xticks(np.arange(n_schemes))
        ax.set_xticklabels([scheme_display_names[s] for s in scheme_order], rotation=30, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_yscale("log")

    fig.tight_layout()

    out_dir_with_model = os.path.join(out_dir, model_id)
    os.makedirs(out_dir_with_model, exist_ok=True)

    all_expert_indices = [(e.layer_idx, e.expert_idx) for e in all_experts]
    experts_str = "_".join(f"L{l}E{e}" for l, e in all_expert_indices)

    out_path = os.path.join(out_dir_with_model, f"mp_seed_stability_{experts_str}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if save_pdf:
        pdf_path = os.path.join(out_dir_with_model, f"mp_seed_stability_{experts_str}.pdf")
        fig.savefig(pdf_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-scheme seed stability analysis")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name or path")
    parser.add_argument("--quantmode", default=DEFAULT_QUANTMODE)
    parser.add_argument("--rank-mode", default=DEFAULT_RANK_MODE)
    parser.add_argument("--bit", type=int, default=DEFAULT_BIT, help="Bit width for sensitivity cache")
    parser.add_argument("--layers", nargs="+", type=int, default=None, help="Layer indices to analyze")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers to use if --layers not specified")
    parser.add_argument("--num-experts", type=int, default=4, help="Number of experts per layer")
    parser.add_argument("--num-seeds", type=int, default=16, help="Number of random seeds")
    parser.add_argument("--seed-start", type=int, default=0, help="Starting seed value")
    parser.add_argument("--seed-interv", type=int, default=DEFAULT_SEED_INTERV, help="Seed interval")
    parser.add_argument("--out-dir", default=OUT_ROOT)
    parser.add_argument("--pdf", action="store_true", default=False, help="Also save PDF copies")
    args = parser.parse_args()

    apply_paper_style()
    model_id = resolve_model_id(args.model)

    if args.layers is None:
        available_layers = discover_layers(args.quantmode, args.rank_mode, model_id)
        if not available_layers:
            print(f"No cached layers found for {model_id}")
            return
        args.layers = available_layers[:args.num_layers]

    print(f"Model: {model_label(model_id)}")
    print(f"Layers: {args.layers}")
    print(f"Experts per layer: {args.num_experts}")
    print(f"Number of seeds: {args.num_seeds} (start={args.seed_start}, interval={args.seed_interv})")

    targets = select_targets(
        model_id,
        args.quantmode,
        args.rank_mode,
        args.bit,
        args.layers,
        args.num_experts,
        max_targets=args.num_layers * args.num_experts,
    )
    if not targets:
        print("No targets found")
        return

    print(f"\nSelected targets:")
    for t in targets:
        print(f"  L{t.layer_idx} E{t.expert_idx} (sensitivity: {t.total_sensitivity:.4e})")

    device = _resolve_device()
    print(f"\nLoading model on {device}...")
    model_path = resolve_model_path(args.model)
    model, tokenizer = load_model(model_path)
    model.eval()

    dataloader, _ = get_loaders(
        "wikitext2",
        nsamples=64,
        seed=42,
        tokenizer=tokenizer,
        seqlen=model.seqlen,
    )

    seeds = [args.seed_start + i * args.seed_interv for i in range(args.num_seeds)]

    print(f"\nCollecting layer inputs...")
    layer_inputs_cache: dict[int, torch.Tensor] = {}
    sensitivity_map: dict[tuple[int, int], np.ndarray] = {}
    for layer_idx in args.layers:
        print(f"  Collecting inputs for layer {layer_idx}...")
        layer_inputs = _collect_layer_mlp_inputs(model, layer_idx, dataloader, 64, device)
        layer_inputs_cache[layer_idx] = layer_inputs.cpu()

        layer = load_layer(model_id, layer_idx, args.quantmode, args.rank_mode, bits=(args.bit,))
        if layer is not None and args.bit in layer.by_bit:
            for expert_idx, rates in enumerate(layer.by_bit[args.bit]):
                sensitivity_map[(layer_idx, expert_idx)] = np.asarray(rates, dtype=np.float32)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nStarting evaluation...")
    expert_results = []

    for target in targets:
        print(f"\nProcessing L{target.layer_idx} E{target.expert_idx}...")

        sensitivity = sensitivity_map.get((target.layer_idx, target.expert_idx))
        if sensitivity is None:
            print(f"Skipping: no sensitivity data")
            continue

        layer_inputs = layer_inputs_cache[target.layer_idx].to(device)
        tokens = _collect_expert_token_inputs(
            model,
            target,
            device,
            4096,
            layer_inputs=layer_inputs,
        )
        tokens_cpu = tokens.cpu()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        expert_result = evaluate_expert(
            model,
            target,
            sensitivity,
            tokens_cpu,
            seeds,
            device,
        )
        expert_results.append(expert_result)

        del tokens, tokens_cpu, layer_inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_statistics(expert_results)

    plot_results(model_id, expert_results, args.out_dir, args.pdf)

    print("\nDone!")


if __name__ == "__main__":
    main()
