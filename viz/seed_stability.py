"""TurboQuant seed-stability diagnostics after DartMoQ slicing.

This module does NOT run end-to-end perplexity or downstream benchmarks.  It is
an explanatory, local analysis: for high-sensitivity MoE experts, compare how
TurboQuant QR seeds affect raw up/gate/down quantization error when quantizing
(1) the full expert matrices and (2) DartMoQ-style neuron slices.

Usage
-----
    python -m viz.seed_stability --dry-run-targets
    python -m viz.seed_stability --layers 17 --experts 2 --seeds 0 1
    python -m viz.seed_stability --top-layers 3 --top-experts 1 --num-seeds 32
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dartmoq_utils import construct_experts_by_rates
from turboquant_utils.quantize import turboquant_quantize
from viz._cache_io import (
    apply_paper_style,
    discover_layers,
    load_layer,
    model_label,
    resolve_model_id,
    resolve_model_path,
)

OUT_ROOT = "plot/seed_stability"
DEFAULT_MODEL = "deepseek-v1-moe-16b"
DEFAULT_QUANTMODE = "turboquant"
DEFAULT_RANK_MODE = "turboquant_innerproduct"
DEFAULT_BIT = 2
DEFAULT_SLICE_EXPERT_NUM = 8


@dataclass(frozen=True)
class Target:
    layer_idx: int
    expert_idx: int
    total_sensitivity: float


@dataclass
class SeedResult:
    seed: int
    mode: str
    raw_error_sum: float
    weighted_error_sum: float
    weighted_to_raw_ratio: float
    error_gini: float
    top1_sens_error_share: float
    top5_sens_error_share: float
    top10_sens_error_share: float
    spearman_err_sens: float
    per_neuron_error: np.ndarray


# ---------------------------------------------------------------------------
# small statistics helpers
# ---------------------------------------------------------------------------
def _gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if x.size == 0:
        return float("nan")
    s = float(x.sum())
    if s <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum.sum() / s)) / n)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average-rank implementation sufficient for Spearman fallback."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(x, y)
        return float(rho)
    except Exception:
        rx = _rankdata(x)
        ry = _rankdata(y)
        rx = rx - rx.mean()
        ry = ry - ry.mean()
        denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
        return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _top_sensitivity_error_share(error: np.ndarray, sensitivity: np.ndarray, frac: float) -> float:
    error = np.asarray(error, dtype=float)
    sensitivity = np.asarray(sensitivity, dtype=float)
    total = float(error.sum())
    if total <= 0 or error.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(frac * error.size)))
    idx = np.argsort(sensitivity)[::-1][:k]
    return float(error[idx].sum() / total)


def _dispersion(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values > 0]
    if values.size == 0:
        return {"gini": float("nan"), "p95_p05": float("nan"), "amgm": float("nan")}
    p95 = float(np.percentile(values, 95))
    p05 = float(np.percentile(values, 5))
    clipped = np.clip(values, 1e-30, None)
    return {
        "gini": _gini(values),
        "p95_p05": p95 / max(p05, 1e-30),
        "amgm": float(clipped.mean() / np.exp(np.mean(np.log(clipped)))),
    }


def _cv(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or arr.mean() == 0:
        return float("nan")
    return float(arr.std() / arr.mean())


def _max_min(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.max() / max(arr.min(), 1e-30))


def _bad_good_gap(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.max() - arr.min())


# ---------------------------------------------------------------------------
# target selection / loading
# ---------------------------------------------------------------------------
def _parse_group_size(value: str) -> Optional[int]:
    if value.lower() in {"none", "full", "row"}:
        return None
    out = int(value)
    if out <= 0:
        raise argparse.ArgumentTypeError("--group-size must be positive or 'none'")
    return out


def _resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[seed_stability] requested {device}, but CUDA is unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(device)


def _load_target_model(model_arg: str):
    from eval_dartmoq import load_model

    model_path = resolve_model_path(model_arg)
    model, _ = load_model(model_path)
    model.eval()
    return model


def _candidate_layers(model_id: str, quantmode: str, rank_mode: str,
                      layers: Optional[Sequence[int]]) -> List[int]:
    if layers:
        return list(layers)
    return discover_layers(quantmode, rank_mode, model_id)


def select_targets(
    model_id: str,
    quantmode: str,
    rank_mode: str,
    bit: int,
    layers: Optional[Sequence[int]],
    experts: Optional[Sequence[int]],
    top_layers: int,
    top_experts: int,
    max_targets: int,
) -> List[Target]:
    rows: List[Target] = []
    layer_totals: List[Tuple[int, float, List[Tuple[int, float]]]] = []

    for layer_idx in _candidate_layers(model_id, quantmode, rank_mode, layers):
        layer = load_layer(model_id, layer_idx, quantmode, rank_mode, bits=(bit,))
        if layer is None or bit not in layer.by_bit:
            continue
        expert_scores = [
            (expert_idx, float(np.asarray(rates, dtype=float).sum()))
            for expert_idx, rates in enumerate(layer.by_bit[bit])
        ]
        if experts:
            expert_set = set(experts)
            expert_scores = [(e, s) for e, s in expert_scores if e in expert_set]
        expert_scores.sort(key=lambda item: item[1], reverse=True)
        if not expert_scores:
            continue
        layer_totals.append((layer_idx, float(sum(s for _, s in expert_scores)), expert_scores))

    if layers:
        selected_layers = layer_totals
    else:
        selected_layers = sorted(layer_totals, key=lambda item: item[1], reverse=True)[:top_layers]

    for layer_idx, _, expert_scores in selected_layers:
        chosen = expert_scores if experts else expert_scores[:top_experts]
        for expert_idx, score in chosen:
            rows.append(Target(layer_idx, expert_idx, score))
            if len(rows) >= max_targets:
                return rows
    return rows


def _load_expert_sensitivity(model_id: str, layer_idx: int, expert_idx: int,
                             bit: int, quantmode: str, rank_mode: str) -> np.ndarray:
    layer = load_layer(model_id, layer_idx, quantmode, rank_mode, bits=(bit,))
    if layer is None or bit not in layer.by_bit:
        raise FileNotFoundError(
            f"no sensitivity cache for {model_id} L{layer_idx} b{bit} "
            f"under {quantmode}/{rank_mode}"
        )
    if expert_idx >= len(layer.by_bit[bit]):
        raise IndexError(f"expert {expert_idx} out of range for L{layer_idx}")
    return np.asarray(layer.by_bit[bit][expert_idx], dtype=np.float32)


def _get_expert_weights(model, target: Target, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layer = model.model.layers[target.layer_idx]
    if not hasattr(layer.mlp, "experts"):
        raise ValueError(f"L{target.layer_idx} does not expose layer.mlp.experts")
    expert = layer.mlp.experts[target.expert_idx]
    return (
        expert.up_proj.weight.detach().to(device=device, dtype=torch.float32),
        expert.gate_proj.weight.detach().to(device=device, dtype=torch.float32),
        expert.down_proj.weight.detach().to(device=device, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# quantization error
# ---------------------------------------------------------------------------
@torch.no_grad()
def _quant_error_matrix(W: torch.Tensor, bit_width: int, group_size: Optional[int],
                        seed: int, rotation: str) -> torch.Tensor:
    q = turboquant_quantize(W, bit_width=bit_width, group_size=group_size,
                            seed=seed, rotation=rotation)
    return (W.float() - q.float()).pow(2)


@torch.no_grad()
def _compute_full_neuron_error(
    W_up: torch.Tensor,
    W_gate: torch.Tensor,
    W_down: torch.Tensor,
    seed: int,
    bit_width: int,
    group_size: Optional[int],
    rotation: str,
) -> torch.Tensor:
    up_err = _quant_error_matrix(W_up, bit_width, group_size, seed, rotation)
    gate_err = _quant_error_matrix(W_gate, bit_width, group_size, seed, rotation)
    down_err = _quant_error_matrix(W_down, bit_width, group_size, seed, rotation)
    return up_err.sum(dim=1) + gate_err.sum(dim=1) + down_err.sum(dim=0)


def _build_slice_groups(sensitivity: np.ndarray, slice_expert_num: int) -> List[List[int]]:
    rates = torch.as_tensor(sensitivity, dtype=torch.float32)
    groups, _ = construct_experts_by_rates(rates, slice_expert_num)
    groups = groups[1:]
    if not groups:
        raise ValueError("construct_experts_by_rates returned no non-dummy groups")
    return groups


@torch.no_grad()
def _compute_slice_neuron_error(
    W_up: torch.Tensor,
    W_gate: torch.Tensor,
    W_down: torch.Tensor,
    groups: Sequence[Sequence[int]],
    seed: int,
    bit_width: int,
    group_size: Optional[int],
    rotation: str,
    slice_seed_strategy: str,
) -> torch.Tensor:
    out = torch.zeros(W_up.shape[0], dtype=torch.float32, device=W_up.device)
    for slice_idx, group in enumerate(groups):
        if not group:
            continue
        idx = torch.as_tensor(group, dtype=torch.long, device=W_up.device)
        slice_seed = seed if slice_seed_strategy == "same" else seed + slice_idx * 1009
        up_err = _quant_error_matrix(W_up[idx, :], bit_width, group_size, slice_seed, rotation)
        gate_err = _quant_error_matrix(W_gate[idx, :], bit_width, group_size, slice_seed, rotation)
        down_err = _quant_error_matrix(W_down[:, idx], bit_width, group_size, slice_seed, rotation)
        out[idx] = up_err.sum(dim=1) + gate_err.sum(dim=1) + down_err.sum(dim=0)
    return out


def _summarize_seed_result(seed: int, mode: str, neuron_err: np.ndarray,
                           sensitivity: np.ndarray) -> SeedResult:
    neuron_err = np.asarray(neuron_err, dtype=np.float64)
    sensitivity = np.asarray(sensitivity, dtype=np.float64)
    sens_norm = sensitivity / max(float(np.mean(sensitivity)), 1e-30)
    raw = float(neuron_err.sum())
    weighted = float((neuron_err * sens_norm).sum())
    return SeedResult(
        seed=seed,
        mode=mode,
        raw_error_sum=raw,
        weighted_error_sum=weighted,
        weighted_to_raw_ratio=weighted / max(raw, 1e-30),
        error_gini=_gini(neuron_err),
        top1_sens_error_share=_top_sensitivity_error_share(neuron_err, sensitivity, 0.01),
        top5_sens_error_share=_top_sensitivity_error_share(neuron_err, sensitivity, 0.05),
        top10_sens_error_share=_top_sensitivity_error_share(neuron_err, sensitivity, 0.10),
        spearman_err_sens=_spearman(neuron_err, sensitivity),
        per_neuron_error=neuron_err,
    )


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def _save(fig, path: str, save_pdf: bool) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    if save_pdf:
        fig.savefig(os.path.splitext(path)[0] + ".pdf")
    plt.close(fig)
    print(f"[seed_stability] saved {path}" + (" and PDF" if save_pdf else ""))


def _best_worst(results: Sequence[SeedResult]) -> Tuple[SeedResult, SeedResult]:
    good = min(results, key=lambda r: r.weighted_error_sum)
    bad = max(results, key=lambda r: r.weighted_error_sum)
    return good, bad


def plot_seed_sweep(model_id: str, target: Target, bit: int,
                    full: Sequence[SeedResult], sliced: Sequence[SeedResult],
                    out_dir: str, save_pdf: bool) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    full_y = np.asarray([r.weighted_error_sum for r in full], dtype=float)
    slice_y = np.asarray([r.weighted_error_sum for r in sliced], dtype=float)
    denom = max(float(np.median(full_y)), 1e-30)
    ax.plot([r.seed for r in full], full_y / denom, marker="o", lw=1.4,
            color="#b5132e", label="full expert")
    ax.plot([r.seed for r in sliced], slice_y / denom, marker="s", lw=1.4,
            color="#3a7ca5", label="DartMoQ slices")
    good, bad = _best_worst(full)
    ax.axvline(good.seed, color="#2a9d8f", ls="--", lw=1, label=f"full good seed={good.seed}")
    ax.axvline(bad.seed, color="#7f1d1d", ls=":", lw=1.2, label=f"full bad seed={bad.seed}")
    ax.set_xlabel("TurboQuant QR seed")
    ax.set_ylabel("Weighted error / median(full)")
    ax.set_title(f"Seed sweep — {model_label(model_id)} L{target.layer_idx} E{target.expert_idx} ({bit}-bit cache)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fp = os.path.join(out_dir, f"seed_sweep_{model_id}_L{target.layer_idx}_E{target.expert_idx}_b{bit}.png")
    _save(fig, fp, save_pdf)
    return fp


def _shade_top_regions(ax, n: int) -> None:
    for frac, alpha, label in [(0.01, 0.16, "top 1% sens"), (0.05, 0.10, "top 5%"), (0.10, 0.06, "top 10%")]:
        ax.axvspan(0, max(1, int(np.ceil(frac * n))), color="#b5132e", alpha=alpha,
                   label=label if frac == 0.01 else None)


def plot_good_bad_neuron_error(model_id: str, target: Target, bit: int,
                               full: Sequence[SeedResult], sliced: Sequence[SeedResult],
                               sensitivity: np.ndarray, out_dir: str, save_pdf: bool) -> str:
    full_good, full_bad = _best_worst(full)
    slice_good, slice_bad = _best_worst(sliced)
    panels = [("full good", full_good), ("full bad", full_bad),
              ("slice good", slice_good), ("slice bad", slice_bad)]
    order = np.argsort(sensitivity)[::-1]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, sharey=True)
    eps = 1e-30
    for ax, (title, result) in zip(axes.flat, panels):
        y = result.per_neuron_error[order]
        _shade_top_regions(ax, y.size)
        ax.plot(np.arange(y.size), y + eps, color="#3a3a3a", lw=0.8)
        ax.set_yscale("log")
        ax.set_title(f"{title}: seed={result.seed}, top10={result.top10_sens_error_share:.2f}, ρ={result.spearman_err_sens:+.2f}")
        ax.grid(True, alpha=0.25)
    axes[1, 0].set_xlabel("Neurons sorted by sensitivity (desc)")
    axes[1, 1].set_xlabel("Neurons sorted by sensitivity (desc)")
    axes[0, 0].set_ylabel("Per-neuron quant error (log)")
    axes[1, 0].set_ylabel("Per-neuron quant error (log)")
    fig.suptitle(f"Good vs bad seed error placement — {model_label(model_id)} L{target.layer_idx} E{target.expert_idx}", y=1.02)
    fp = os.path.join(out_dir, f"good_bad_neuron_error_{model_id}_L{target.layer_idx}_E{target.expert_idx}_b{bit}.png")
    _save(fig, fp, save_pdf)
    return fp


def plot_alignment(model_id: str, target: Target, bit: int,
                   full: Sequence[SeedResult], sliced: Sequence[SeedResult],
                   sensitivity: np.ndarray, out_dir: str, save_pdf: bool) -> str:
    full_good, full_bad = _best_worst(full)
    slice_good, slice_bad = _best_worst(sliced)
    panels = [("full good", full_good, "#2a9d8f"), ("full bad", full_bad, "#b5132e"),
              ("slice good", slice_good, "#3a7ca5"), ("slice bad", slice_bad, "#cc7722")]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.0), sharex=True, sharey=True)
    x = np.asarray(sensitivity, dtype=float)
    eps = 1e-30
    for ax, (title, result, color) in zip(axes.flat, panels):
        ax.scatter(x + eps, result.per_neuron_error + eps, s=8, alpha=0.45, color=color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{title}: seed={result.seed}, ρ={result.spearman_err_sens:+.2f}, top10={result.top10_sens_error_share:.2f}")
        ax.grid(True, alpha=0.25)
    axes[1, 0].set_xlabel("Sensitivity (log)")
    axes[1, 1].set_xlabel("Sensitivity (log)")
    axes[0, 0].set_ylabel("Per-neuron quant error (log)")
    axes[1, 0].set_ylabel("Per-neuron quant error (log)")
    fig.suptitle(f"Error-sensitivity alignment — {model_label(model_id)} L{target.layer_idx} E{target.expert_idx}", y=1.02)
    fp = os.path.join(out_dir, f"error_sensitivity_alignment_{model_id}_L{target.layer_idx}_E{target.expert_idx}_b{bit}.png")
    _save(fig, fp, save_pdf)
    return fp


def plot_homogeneity(model_id: str, target: Target, bit: int, sensitivity: np.ndarray,
                     groups: Sequence[Sequence[int]], out_dir: str, save_pdf: bool) -> str:
    labels = ["full"] + [f"S{i}" for i in range(len(groups))]
    values = [sensitivity] + [sensitivity[np.asarray(g, dtype=int)] for g in groups]
    disp = [_dispersion(v) for v in values]
    metrics = ["gini", "p95_p05", "amgm"]
    titles = {"gini": "Gini", "p95_p05": "P95/P05", "amgm": "AM/GM"}
    colors = ["#b5132e"] + ["#3a7ca5"] * len(groups)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))
    for ax, metric in zip(axes, metrics):
        y = [d[metric] for d in disp]
        ax.bar(range(len(labels)), y, color=colors)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_title(titles[metric])
        if metric != "gini":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(f"Sensitivity homogeneity after slicing — {model_label(model_id)} L{target.layer_idx} E{target.expert_idx}", y=1.03)
    fp = os.path.join(out_dir, f"slice_homogeneity_{model_id}_L{target.layer_idx}_E{target.expert_idx}_b{bit}.png")
    _save(fig, fp, save_pdf)
    return fp


def plot_aggregate(model_id: str, bit: int, summaries: Sequence[Dict[str, float]],
                   out_dir: str, save_pdf: bool) -> str:
    if not summaries:
        return ""
    labels = [f"L{int(s['layer'])}E{int(s['expert'])}" for s in summaries]
    metrics = [("cv", "Weighted-error CV"), ("max_min", "Max / min"), ("gap_norm", "Bad-good gap / mean(full)")]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    x = np.arange(len(labels))
    width = 0.36
    for ax, (key, title) in zip(axes, metrics):
        full_y = [s[f"full_{key}"] for s in summaries]
        slice_y = [s[f"slice_{key}"] for s in summaries]
        ax.bar(x - width / 2, full_y, width, label="full", color="#b5132e")
        ax.bar(x + width / 2, slice_y, width, label="slice", color="#3a7ca5")
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
    axes[-1].legend(fontsize=8)
    fig.suptitle(f"Aggregate seed variability — {model_label(model_id)} ({bit}-bit cache)", y=1.03)
    fp = os.path.join(out_dir, f"aggregate_seed_variability_{model_id}_b{bit}.png")
    _save(fig, fp, save_pdf)
    return fp


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def _print_target_table(targets: Sequence[Target]) -> None:
    print("\nSelected targets:")
    if not targets:
        print("  (none)")
    for t in targets:
        print(f"  L{t.layer_idx:<3d} E{t.expert_idx:<3d} total_sensitivity={t.total_sensitivity:.6e}")


def _validate_shapes(sensitivity: np.ndarray, W_up: torch.Tensor, W_gate: torch.Tensor, W_down: torch.Tensor) -> None:
    n = sensitivity.shape[0]
    if W_up.shape[0] != n or W_gate.shape[0] != n or W_down.shape[1] != n:
        raise ValueError(
            "sensitivity and weight shapes do not align: "
            f"sens={n}, up={tuple(W_up.shape)}, gate={tuple(W_gate.shape)}, down={tuple(W_down.shape)}"
        )


def run_one_target(model, model_id: str, target: Target, seeds: Sequence[int], args) -> Dict[str, object]:
    print(f"\n=== seed_stability target: L{target.layer_idx} E{target.expert_idx} ===")
    sensitivity = _load_expert_sensitivity(model_id, target.layer_idx, target.expert_idx,
                                           args.bit, args.quantmode, args.rank_mode)
    device = _resolve_device(args.device)
    W_up, W_gate, W_down = _get_expert_weights(model, target, device)
    _validate_shapes(sensitivity, W_up, W_gate, W_down)
    groups = _build_slice_groups(sensitivity, args.slice_expert_num)

    full_results: List[SeedResult] = []
    slice_results: List[SeedResult] = []
    for seed in seeds:
        print(f"  seed {seed}")
        full_err = _compute_full_neuron_error(W_up, W_gate, W_down, seed,
                                              args.wbits, args.group_size, args.rotation)
        slice_err = _compute_slice_neuron_error(W_up, W_gate, W_down, groups, seed,
                                                args.wbits, args.group_size, args.rotation,
                                                args.slice_seed_strategy)
        full_np = full_err.detach().cpu().numpy()
        slice_np = slice_err.detach().cpu().numpy()
        full_results.append(_summarize_seed_result(seed, "full", full_np, sensitivity))
        slice_results.append(_summarize_seed_result(seed, "slice", slice_np, sensitivity))
        del full_err, slice_err
        if device.type == "cuda":
            torch.cuda.empty_cache()

    full_w = [r.weighted_error_sum for r in full_results]
    slice_w = [r.weighted_error_sum for r in slice_results]
    denom = max(float(np.mean(full_w)), 1e-30)
    summary = {
        "layer": float(target.layer_idx),
        "expert": float(target.expert_idx),
        "full_cv": _cv(full_w),
        "slice_cv": _cv(slice_w),
        "full_max_min": _max_min(full_w),
        "slice_max_min": _max_min(slice_w),
        "full_gap_norm": _bad_good_gap(full_w) / denom,
        "slice_gap_norm": _bad_good_gap(slice_w) / denom,
    }
    full_good, full_bad = _best_worst(full_results)
    slice_good, slice_bad = _best_worst(slice_results)
    print(
        f"  full : CV={summary['full_cv']:.4f}, max/min={summary['full_max_min']:.4f}, "
        f"good={full_good.seed}, bad={full_bad.seed}, bad top10={full_bad.top10_sens_error_share:.3f}"
    )
    print(
        f"  slice: CV={summary['slice_cv']:.4f}, max/min={summary['slice_max_min']:.4f}, "
        f"good={slice_good.seed}, bad={slice_bad.seed}, bad top10={slice_bad.top10_sens_error_share:.3f}"
    )

    if "seed_sweep" not in args.skip:
        plot_seed_sweep(model_id, target, args.bit, full_results, slice_results, args.out_dir, args.pdf)
    if "good_bad" not in args.skip:
        plot_good_bad_neuron_error(model_id, target, args.bit, full_results, slice_results,
                                   sensitivity, args.out_dir, args.pdf)
    if "alignment" not in args.skip:
        plot_alignment(model_id, target, args.bit, full_results, slice_results,
                       sensitivity, args.out_dir, args.pdf)
    if "homogeneity" not in args.skip:
        plot_homogeneity(model_id, target, args.bit, sensitivity, groups, args.out_dir, args.pdf)

    del W_up, W_gate, W_down
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"summary": summary, "full": full_results, "slice": slice_results}


def _seed_list(args) -> List[int]:
    if args.seeds is not None:
        return list(args.seeds)
    return list(range(args.seed_start, args.seed_start + args.num_seeds))


def main():
    apply_paper_style()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="short cache id OR full model path")
    parser.add_argument("--quantmode", default=DEFAULT_QUANTMODE)
    parser.add_argument("--rank-mode", default=DEFAULT_RANK_MODE)
    parser.add_argument("--bit", type=int, default=DEFAULT_BIT,
                        help="bit used to read sensitivity cache")
    parser.add_argument("--wbits", type=int, default=None,
                        help="TurboQuant bit width; default: same as --bit")
    parser.add_argument("--group-size", type=_parse_group_size, default=128,
                        help="TurboQuant group size; use 'none' for full-row groups")
    parser.add_argument("--rotation", default="qr", choices=["qr", "hadamard"])
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--experts", nargs="+", type=int, default=None)
    parser.add_argument("--top-layers", type=int, default=3)
    parser.add_argument("--top-experts", type=int, default=1)
    parser.add_argument("--max-targets", type=int, default=6)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=32)
    parser.add_argument("--slice-expert-num", type=int, default=DEFAULT_SLICE_EXPERT_NUM)
    parser.add_argument("--slice-seed-strategy", choices=["same", "offset"], default="same")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run-targets", action="store_true")
    parser.add_argument("--out-dir", default=OUT_ROOT)
    parser.add_argument("--pdf", action="store_true", default=False,
                        help="also save PDF copies alongside PNGs")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["seed_sweep", "good_bad", "alignment", "homogeneity", "aggregate"])
    args = parser.parse_args()
    if args.wbits is None:
        args.wbits = args.bit

    model_id = resolve_model_id(args.model)
    targets = select_targets(
        model_id, args.quantmode, args.rank_mode, args.bit,
        layers=args.layers, experts=args.experts,
        top_layers=args.top_layers, top_experts=args.top_experts,
        max_targets=args.max_targets,
    )
    _print_target_table(targets)
    if args.dry_run_targets:
        return
    if not targets:
        print(f"[seed_stability] no targets found for {model_id}/{args.quantmode}/{args.rank_mode}/b{args.bit}")
        return

    seeds = _seed_list(args)
    print(f"\nSeeds: {seeds}")
    print(f"Loading model {args.model} ...")
    model = _load_target_model(args.model)

    summaries = []
    for target in targets:
        result = run_one_target(model, model_id, target, seeds, args)
        summaries.append(result["summary"])

    if "aggregate" not in args.skip and len(summaries) > 1:
        plot_aggregate(model_id, args.bit, summaries, args.out_dir, args.pdf)


if __name__ == "__main__":
    main()
