"""Headroom visualizations — *MoE mixed-precision has large untapped headroom.*

Four motivation figures (each is one panel in the paper):

    amgm                      — per-layer AM/GM of neuron loss = closed-form
                                  headroom factor (uniform / oracle loss ratio,
                                  derived from the log-quadratic loss model).
                                  This IS the upper bound on what any neuron-
                                  level mixed-precision allocation can ever
                                  achieve over uniform-bit allocation.
    top10ratio                — share of total loss captured by the top-10%
                                  most sensitive neurons per layer. A simpler,
                                  more interpretable companion to `amgm`.
    act_vs_sens               — activation-rate ↔ sensitivity scatter. Spearman ρ
                                  near 0 ⇒ activation-rate (MoQE-style) and
                                  sensitivity (OWQ-style) are NOT interchangeable;
                                  a joint search is *necessary*.
    bucket_sweep              — bridge between `amgm` (closed-form bound) and
                                  `layer_expert_neuron_compare` (discrete DP).
                                  Sweeps the per-expert bucket count from 1
                                  (= expert granularity) toward n_neurons
                                  (theoretical limit) and shows how the realised
                                  uniform/DP loss ratio approaches the AM/GM
                                  bound. Justifies DartMoQP's choice of
                                  `slice_expert_num=8`.
    layer_expert_neuron_compare
                              — loss-model oracle loss at three granularities
                                  {layer, expert, neuron-bucket} over a bpw
                                  sweep; gap between curves = headroom captured
                                  by going one level finer.

All figures are driven by the cached sensitivity tensors in
``quant_outlier_{quantmode}/{rank_mode}/{model_id}/``; no model needs to be
reloaded. ``act_vs_sens`` additionally consumes per-expert activation-rate
tensors materialised by ``viz/dump_activation_rates.py``.

Usage
-----
    python -m viz.headroom                          # all models with cache
    python -m viz.headroom --model olmoe-7b-1b
    python -m viz.headroom --skip act_vs_sens       # skip a specific panel
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Make sibling modules importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viz._cache_io import (
    LayerSensitivity, apply_paper_style, discover_layers, discover_models,
    expert_total_loss, load_all_layers, load_layer, model_label,
    neuron_loss_matrix, resolve_model_id,
)

OUT_ROOT = "plot/headroom"
DEFAULT_QUANTMODE = "turboquant"
DEFAULT_RANK_MODE = "turboquant_innerproduct"
DEFAULT_BIT = 2  # probe bit used for static panels


# ----------------------------------------------------------------------------
# Headroom panels: top10ratio, amgm
# ----------------------------------------------------------------------------
def _headroom_compute(
    model_id: str,
    quantmode: str, rank_mode: str, bit: int,
    layer_start: Optional[int] = None,
    num_layers: Optional[int] = None,
) -> Tuple[List[int], List[float], List[float]]:
    """Shared per-layer computation for the two headroom panels.

    Returns: (layer_ids, top10%-share-per-layer, AM/GM-per-layer). Layers
    with no positive loss are skipped.
    """
    layers = load_all_layers(model_id, quantmode, rank_mode, bits=(bit,), layer_start=layer_start, num_layers=num_layers)
    layer_ids, head_fractions, am_gm_ratios = [], [], []
    for L in layers:
        neurons = neuron_loss_matrix(L, bit).flatten()
        neurons = neurons[neurons > 0]
        if neurons.size == 0:
            continue
        srt = np.sort(neurons)
        layer_ids.append(L.layer_idx)

        k = max(1, int(0.1 * len(srt)))
        head_fractions.append(srt[-k:].sum() / srt.sum())

        c = np.clip(srt, 1e-30, None)
        am = c.mean()
        gm = np.exp(np.mean(np.log(c)))
        am_gm_ratios.append(am / gm)
    return layer_ids, head_fractions, am_gm_ratios


def _multi_model_axes(models: List[str], width_per: float = 4.0, height: float = 3.8):
    """Create a 1xN figure (one subplot per model) with a shared layout."""
    n = max(len(models), 1)
    fig, axes = plt.subplots(1, n, figsize=(width_per * n, height), squeeze=False)
    return fig, axes[0]


def top10ratio(
    models: List[str],
    quantmode: str = DEFAULT_QUANTMODE,
    rank_mode: str = DEFAULT_RANK_MODE,
    bit: int = DEFAULT_BIT,
    layer_start: Optional[int] = None,
    num_layers: Optional[int] = None,
    out_dir: str = OUT_ROOT,
    save_pdf: bool = False,
) -> str:
    """Per-layer loss share captured by the top-10% most sensitive neurons.

    Bar > 0.10 means the layer is concentrated (uniform reference = 0.10).
    Companion to `amgm`: dispersion as a single interpretable number.
    """
    fig, axes = _multi_model_axes(models)
    all_layer_ids = None
    for ax, model_id in zip(axes, models):
        layer_ids, head_fractions, _ = _headroom_compute(
            model_id, quantmode, rank_mode, bit, layer_start=layer_start, num_layers=num_layers)
        if all_layer_ids is None:
            all_layer_ids = layer_ids
        if not layer_ids:
            ax.text(0.5, 0.5, "no cache", ha="center", va="center",
                    transform=ax.transAxes); ax.set_title(model_label(model_id))
            continue
        ax.bar(range(len(head_fractions)), head_fractions, color="#3a7ca5")
        ax.axhline(0.1, color="k", ls="--", lw=1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Decoder layer")
        layer_str = f"start={layer_start}, n={len(layer_ids)}" if layer_start is not None else f"all {len(layer_ids)} layers"
        ax.set_title(f"{model_label(model_id)}  "
                     f"max={max(head_fractions):.2f}")
    axes[0].set_ylabel("Loss share of top-10% neurons")
    layer_suffix = f"_start{layer_start}_n{len(all_layer_ids)}" if layer_start is not None and all_layer_ids is not None else ""
    fig.suptitle(f"Top-10% loss concentration — {bit}-bit, {layer_str}  "
                 f"(dashed = uniform reference 0.10)",
                 fontsize=11, y=1.02)

    os.makedirs(out_dir, exist_ok=True)
    fp_png = os.path.join(out_dir, f"top10ratio_{rank_mode}_b{bit}{layer_suffix}.png")
    plt.tight_layout()
    plt.savefig(fp_png)
    if save_pdf:
        layer_suffix = f"_start{layer_start}_n{len(all_layer_ids)}" if layer_start is not None and all_layer_ids is not None else ""
        fp_pdf = os.path.join(out_dir, f"top10ratio_{rank_mode}_b{bit}{layer_suffix}.pdf")
        plt.savefig(fp_pdf)
    plt.close(fig)
    print(f"[top10ratio] saved {fp_png}" + (f" and {fp_pdf}" if save_pdf else ""))
    return fp_png


def amgm(
    models: List[str],
    quantmode: str = DEFAULT_QUANTMODE,
    rank_mode: str = DEFAULT_RANK_MODE,
    bit: int = DEFAULT_BIT,
    layer_start: Optional[int] = None,
    num_layers: Optional[int] = None,
    out_dir: str = OUT_ROOT,
    save_pdf: bool = False,
) -> str:
    """AM/GM of per-neuron loss = closed-form headroom factor.

    Math bridge: under the log-quadratic loss model
        log L_i(b) = p b² + q b + r_i,
    the optimal-bit (oracle) total loss over a uniform-bit allocation equals
    AM(c_i) / GM(c_i) where c_i = exp(r_i). By AM-GM this ratio ≥ 1, with
    equality iff all neurons are equally hard to quantize. So this bar IS
    the upper bound on what any neuron-level mixed-precision method can
    reclaim against uniform-bit allocation, computed without solving any
    optimization.
    """
    fig, axes = _multi_model_axes(models)
    all_layer_ids = None
    for ax, model_id in zip(axes, models):
        layer_ids, _, am_gm_ratios = _headroom_compute(
            model_id, quantmode, rank_mode, bit, layer_start=layer_start, num_layers=num_layers)
        if all_layer_ids is None:
            all_layer_ids = layer_ids
        if not layer_ids:
            ax.text(0.5, 0.5, "no cache", ha="center", va="center",
                    transform=ax.transAxes); ax.set_title(model_label(model_id))
            continue
        ax.bar(range(len(am_gm_ratios)), am_gm_ratios, color="#b5132e")
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("Decoder layer")
        layer_str = f"start={layer_start}, n={len(layer_ids)}" if layer_start is not None else f"all {len(layer_ids)} layers"
        ax.set_title(f"{model_label(model_id)}  "
                     f"max={max(am_gm_ratios):.1f}×")
    axes[0].set_ylabel("AM / GM of per-neuron loss (log)")
    layer_suffix = f"_start{layer_start}_n{len(all_layer_ids)}" if layer_start is not None and all_layer_ids is not None else ""
    fig.suptitle(f"Headroom factor (uniform / oracle loss ratio) — {bit}-bit, {layer_str}  "
                 f"(dashed = no headroom 1.0)",
                 fontsize=11, y=1.02)

    os.makedirs(out_dir, exist_ok=True)
    fp_png = os.path.join(out_dir, f"amgm_{rank_mode}_b{bit}{layer_suffix}.png")
    plt.tight_layout()
    plt.savefig(fp_png)
    if save_pdf:
        layer_suffix = f"_start{layer_start}_n{len(all_layer_ids)}" if layer_start is not None and all_layer_ids is not None else ""
        fp_pdf = os.path.join(out_dir, f"amgm_{rank_mode}_b{bit}{layer_suffix}.pdf")
        plt.savefig(fp_pdf)
    plt.close(fig)
    print(f"[amgm] saved {fp_png}" + (f" and {fp_pdf}" if save_pdf else ""))
    return fp_png


# ----------------------------------------------------------------------------
# activation rate ↔ sensitivity scatter / Spearman
# ----------------------------------------------------------------------------
def _load_activation_rates(model_id: str) -> Dict[int, np.ndarray]:
    """Load per-layer expert activation rates dumped by viz/dump_activation_rates.py.

    Expected files:  logs/activation_rates/{model_id}/L{layer}.npy
                     shape = (n_experts,), values in [0,1] summing to top_k.
    """
    out = {}
    root = f"logs/activation_rates/{model_id}"
    if not os.path.isdir(root):
        return out
    for fn in sorted(os.listdir(root)):
        if not fn.endswith(".npy"):
            continue
        m = fn.replace(".npy", "")
        try:
            li = int(m.replace("L", ""))
        except ValueError:
            continue
        out[li] = np.load(os.path.join(root, fn))
    return out


def act_vs_sens(
    model_id: str,
    quantmode: str = DEFAULT_QUANTMODE,
    rank_mode: str = DEFAULT_RANK_MODE,
    bit: int = DEFAULT_BIT,
    layer_start: Optional[int] = None,
    num_layers: Optional[int] = None,
    out_dir: str = OUT_ROOT,
    save_pdf: bool = False,
) -> str:
    """Scatter (activation rate × quant sensitivity) per expert, coloured by
    layer; companion panel: histogram of per-layer Spearman ρ.

    Logical link to `amgm` — `amgm` shows the per-neuron loss distribution is
    long-tailed; readers might object "activation rate already explains this:
    more-used neurons get hit harder". `act_vs_sens` falsifies that hypothesis:
    activation rate and quantization sensitivity are nearly independent
    (Spearman ρ centred on 0), so neither signal is a substitute for the
    other — DartMoQP's joint search is *necessary*, not just convenient.
    """
    from scipy.stats import spearmanr

    layers = load_all_layers(model_id, quantmode, rank_mode, bits=(bit,), layer_start=layer_start, num_layers=num_layers)
    if not layers:
        print(f"[act_vs_sens] no cache for {model_id}/{quantmode}/{rank_mode}")
        return ""

    act_rates = _load_activation_rates(model_id)
    if not act_rates:
        print(f"[act_vs_sens] no activation-rate data at logs/activation_rates/{model_id}/; "
              f"run `python -m viz.dump_activation_rates --model {model_id}` first")
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    # left: scatter, colour=layer
    ax = axes[0]
    cmap = plt.get_cmap("viridis", len(layers))
    all_corrs = []
    n_layers_with_data = 0
    for r, L in enumerate(layers):
        if L.layer_idx not in act_rates:
            continue
        sens = expert_total_loss(L, bit)
        rates = act_rates[L.layer_idx][: len(sens)]
        if sens.std() == 0 or rates.std() == 0:
            continue
        rho, _ = spearmanr(rates, sens)
        all_corrs.append(rho)
        n_layers_with_data += 1
        ax.scatter(rates, sens, color=cmap(r), s=12, alpha=0.55)

    if n_layers_with_data == 0:
        print(f"[act_vs_sens] activation data found but no overlap with sensitivity layers")
        plt.close(fig); return ""

    ax.set_xlabel("Expert activation rate (measured on calibration set)")
    ax.set_ylabel(f"Expert {bit}-bit quant sensitivity")
    ax.set_yscale("log")
    layer_str = f"start={layer_start}, n={n_layers_with_data}" if layer_start is not None else f"all {n_layers_with_data} layers"
    ax.set_title(f"(A) {layer_str}, {model_label(model_id)}")

    # right: histogram of per-layer Spearman ρ
    ax = axes[1]
    ax.hist(all_corrs, bins=15, range=(-1, 1), color="#cc7722", edgecolor="black")
    mean_rho = float(np.mean(all_corrs))
    ax.axvline(mean_rho, color="k", ls="--",
               label=f"mean ρ = {mean_rho:+.2f}")
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman ρ (activation, sensitivity), per layer")
    ax.set_ylabel("# layers")
    ax.set_title("(B) Weak correlation ⇒ both signals needed")
    ax.legend()

    os.makedirs(out_dir, exist_ok=True)
    layer_suffix = f"_start{layer_start}_n{n_layers_with_data}" if layer_start is not None else ""
    fp_png = os.path.join(out_dir, f"act_vs_sens_{model_id}_{rank_mode}_b{bit}{layer_suffix}.png")
    plt.tight_layout()
    plt.savefig(fp_png)
    if save_pdf:
        layer_suffix = f"_start{layer_start}_n{n_layers_with_data}" if layer_start is not None else ""
        fp_pdf = os.path.join(out_dir, f"act_vs_sens_{model_id}_{rank_mode}_b{bit}{layer_suffix}.pdf")
        plt.savefig(fp_pdf)
    plt.close(fig)
    print(f"[act_vs_sens] saved {fp_png}" + (f" and {fp_pdf}" if save_pdf else "") + f"  | mean Spearman ρ = {mean_rho:+.3f}")
    return fp_png


# ----------------------------------------------------------------------------
# layer_expert_neuron_compare — granularity sweep, all models on one figure
# ----------------------------------------------------------------------------
def _loss_from_neuron_bits(
    rates: Dict[int, np.ndarray],
    neuron_bits: np.ndarray,
) -> float:
    """Sum loss using per-neuron bit assignment + per-bit loss table.

    `rates[b][i]` is the loss when neuron i is quantized at b bits, so the
    total loss is simply Σ_i rates[neuron_bits[i]][i].
    """
    total = 0.0
    for b, vec in rates.items():
        mask = neuron_bits == b
        if mask.any():
            total += float(vec[mask].sum())
    return total


def _dp_layer_total_loss(
    layer: LayerSensitivity,
    bits: List[int],
    granularity: str,
    target_bpw: float,
    slice_expert_num: int = 8,
) -> float:
    """Loss-model oracle total loss for one layer under the given granularity.

    All three granularities are evaluated on the SAME loss table
    `layer.by_bit[b][expert][neuron]`. Uniform activation weight is used here
    so that the curve isolates the benefit of *granularity*; `act_vs_sens` covers the
    activation-vs-sensitivity story separately.

    granularity:
        "layer"    — every neuron in the layer at the same integer bit
                     (emits NaN at half-bpw, because layer-wise schedules
                     are inherently integer-valued)
        "expert"   — each expert independently picks one bit
        "neuron"   — each expert is split into `slice_expert_num` neuron
                     buckets, each bucket picks its own bit (DartMoQ proper)

    Note: a separate "uniform" granularity is intentionally omitted — at
    integer bpw it is identical to "layer" (DP degree of freedom = 0), and
    at half-bpw it has no real-quantizer realisation. Including it would
    just draw a line that is either occluded by `layer` or NaN.
    """
    from dp_utils import enum_optimal_m_scheme_global_fast

    n_experts = layer.n_experts
    activation = np.full(n_experts, 1.0 / n_experts)  # uniform weighting
    expert_rates = [
        {b: layer.by_bit[b][e] for b in bits if b in layer.by_bit}
        for e in range(n_experts)
    ]

    # ----- layer (integer bpw only) -------------------------------------
    # All neurons in the layer share one bit. Half-bpw has no integer
    # representative, so emit NaN — the plot draws no marker there.
    if granularity == "layer":
        if abs(target_bpw - round(target_bpw)) > 1e-6:
            return float("nan")
        b = int(round(target_bpw))
        if b not in layer.by_bit:
            return float("nan")
        return float(sum(activation[e] * expert_rates[e][b].sum()
                         for e in range(n_experts)))

    # ----- expert -------------------------------------------------------
    # 1 bit per expert (slice_expert_num=1). DP returns per-neuron bit
    # assignment; re-evaluate the loss table to get a real loss number.
    if granularity == "expert":
        _, neuron_bits_per_expert = enum_optimal_m_scheme_global_fast(
            expert_rates, activation, 1, target_bpw=target_bpw)
        total = 0.0
        for e in range(n_experts):
            total += activation[e] * _loss_from_neuron_bits(
                expert_rates[e], neuron_bits_per_expert[e])
        return total

    # ----- neuron -------------------------------------------------------
    # slice_expert_num buckets per expert, each bucket picks its own bit.
    if granularity == "neuron":
        _, neuron_bits_per_expert = enum_optimal_m_scheme_global_fast(
            expert_rates, activation, slice_expert_num, target_bpw=target_bpw)
        total = 0.0
        for e in range(n_experts):
            total += activation[e] * _loss_from_neuron_bits(
                expert_rates[e], neuron_bits_per_expert[e])
        return total

    raise ValueError(f"unknown granularity {granularity}")


def layer_expert_neuron_compare(
    models: List[str],
    quantmode: str = DEFAULT_QUANTMODE,
    rank_mode: str = DEFAULT_RANK_MODE,
    bpws: Tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
    slice_expert_num: int = 8,
    layer_start: Optional[int] = None,
    num_layers: Optional[int] = None,
    out_dir: str = OUT_ROOT,
    save_pdf: bool = False,
) -> str:
    """Multi-model granularity sweep on a single figure.

    Layout: one column per model, four lines per panel (uniform / layer /
    expert / neuron). The y-axis (log) is the loss-model oracle total loss
    (averaged over the selected layers with full bit cache).
    """
    granularities = ["layer", "expert", "neuron"]
    colors = {"layer": "#3a7ca5", "expert": "#cc7722", "neuron": "#b5132e"}
    markers = {"layer": "o", "expert": "s", "neuron": "D"}

    # data: model -> granularity -> [loss per bpw]
    table: Dict[str, Dict[str, List[float]]] = {}

    for model_id in models:
        layers = load_all_layers(
            model_id, quantmode, rank_mode,
            bits=(0, 1, 2, 3, 4),
            layer_start=layer_start,
            num_layers=num_layers,
        )
        if not layers:
            print(f"[layer_expert_neuron_compare] no cache for {model_id}/{quantmode}/{rank_mode}; skipping")
            continue

        # intersect available bits across layers
        bits = sorted(set.intersection(*[set(L.by_bit.keys()) for L in layers]))
        if 0 in bits and not any(L.by_bit[0][0].sum() > 0 for L in layers):
            bits.remove(0)

        table[model_id] = {g: [] for g in granularities}
        for bpw in bpws:
            for g in granularities:
                agg, n_ok = 0.0, 0
                for L in layers:
                    v = _dp_layer_total_loss(L, bits, g, bpw, slice_expert_num)
                    if np.isfinite(v):
                        agg += v; n_ok += 1
                table[model_id][g].append(agg / max(n_ok, 1) if n_ok else float("nan"))

    if not table:
        print("[layer_expert_neuron_compare] no models had usable cache")
        return ""

    n = len(table)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (model_id, gd) in zip(axes, table.items()):
        bpws_arr = np.asarray(bpws, dtype=float)
        for g in granularities:
            y = np.asarray(gd[g], dtype=float)
            mask = np.isfinite(y)
            # `layer` only has values at integer bpw; drop NaNs so the line
            # connects across the gaps instead of breaking at every half-step.
            ax.plot(bpws_arr[mask], y[mask], color=colors[g], marker=markers[g],
                    lw=1.5, ms=5, label=g)
        ax.set_yscale("log")
        ax.set_xlabel("Target bpw")
        ax.set_title(model_label(model_id), fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Loss-model oracle total loss (log)")
    axes[-1].legend(loc="upper right", fontsize=8)
    if layer_start is not None:
        layer_str = f"start={layer_start}, n={len(layers)}"
    else:
        layer_str = f"first {max_layers} layers"
    fig.suptitle(f"Granularity headroom — {rank_mode}, {layer_str}",
                 fontsize=11, y=1.02)

    os.makedirs(out_dir, exist_ok=True)
    tag = "_".join(sorted(table.keys()))
    if len(tag) > 80:
        tag = f"{len(table)}models"
    if layer_start is not None:
        layer_suffix = f"_start{layer_start}_n{len(layers)}"
    else:
        layer_suffix = ""
    fp_png = os.path.join(out_dir, f"layer_expert_neuron_compare_{tag}_{rank_mode}{layer_suffix}.png")
    plt.tight_layout()
    plt.savefig(fp_png)
    if save_pdf:
        fp_pdf = os.path.join(out_dir, f"layer_expert_neuron_compare_{tag}_{rank_mode}{layer_suffix}.pdf")
        plt.savefig(fp_pdf)
    plt.close(fig)
    print(f"[layer_expert_neuron_compare] saved {fp_png}" + (f" and {fp_pdf}" if save_pdf else ""))
    return fp_png


# ----------------------------------------------------------------------------
# bucket_sweep — bridge between `amgm` (closed-form bound) and
#                `layer_expert_neuron_compare` (discrete DP at slice=8)
# ----------------------------------------------------------------------------
def _amgm_ratio(layer: LayerSensitivity, bit: int = None) -> float:
    """Per-layer AM/GM of neuron loss — computes the theoretical bound
    by fitting log-quadratic model to all available bits.

    The theoretical bound is AM(exp(r_i)) / GM(exp(r_i)) where r_i comes
    from fitting log L_i(b) = p b² + q b + r_i. This is a true upper bound
    for any bit allocation under the log-quadratic model.
    """
    # Collect all bits and their loss data
    bits = sorted(layer.by_bit.keys())
    if len(bits) < 2:
        # Fall back to single-bit AM/GM if not enough bits for fitting
        neurons = neuron_loss_matrix(layer, bit if bit is not None else bits[0]).flatten()
        neurons = neurons[neurons > 0]
        if neurons.size < 2:
            return float("nan"), float("nan"), float("nan"), float("nan")
        c = np.clip(neurons, 1e-30, None)
        return float(c.mean() / np.exp(np.mean(np.log(c)))), c.mean(), c.std(), np.exp(np.mean(np.log(c)))

    # Get per-neuron loss data across bits — by_bit is Dict[int, List[np.ndarray]]
    # where each list element is one expert
    all_losses = []

    for b in bits:
        # Stack all experts for this bit, then flatten
        layer_loss = neuron_loss_matrix(layer, b).flatten()
        all_losses.append(layer_loss)

    all_losses = np.stack(all_losses, axis=1)  # (n_neurons_total, n_bits)
    n_neurons_total = all_losses.shape[0]

    # Fit log-quadratic model for each neuron
    r_list = []

    for i in range(n_neurons_total):
        loss_i = all_losses[i, :]
        valid = loss_i > 0
        if not np.any(valid):
            continue

        try:
            # Fit log L_i(b) = p b² + q b + r_i
            log_loss = np.log(loss_i[valid])
            bits_valid = np.array(bits)[valid]
            if len(bits_valid) < 2:
                continue
            _, _, r = np.polyfit(bits_valid, log_loss, deg=2)
            r_list.append(r)
        except:
            continue

    if len(r_list) < 2:
        # Fall back
        use_bit = bit if bit is not None else bits[-1]
        neurons = neuron_loss_matrix(layer, use_bit).flatten()
        neurons = neurons[neurons > 0]
        if neurons.size < 2:
            return float("nan"), float("nan"), float("nan"), float("nan")
        c = np.clip(neurons, 1e-30, None)
        return float(c.mean() / np.exp(np.mean(np.log(c)))), c.mean(), c.std(), np.exp(np.mean(np.log(c)))

    # Compute theoretical AM/GM bound from the r_i (c_i = exp(r_i))
    r_array = np.array(r_list)
    c = np.exp(r_array)
    am = c.mean()
    gm = np.exp(np.mean(np.log(c)))
    return float(am / gm), float(am), float(c.std()), float(gm)


def _dp_uniform_over_dp(
    layer: LayerSensitivity,
    bits: List[int],
    bpw: float,
    slice_expert_num: int,
) -> float:
    """Ratio uniform_loss(bpw) / DP_loss(bpw, slice_expert_num).

    `uniform` here means "every neuron at the integer bit `round(bpw)`" — the
    same baseline `amgm` is implicitly comparing against. Returns NaN if bpw
    is not an integer (uniform is undefined) or the bit is not cached.
    """
    if abs(bpw - round(bpw)) > 1e-6:
        return float("nan")
    b_uniform = int(round(bpw))
    if b_uniform not in layer.by_bit:
        return float("nan")

    # uniform reference (matches the `layer` line in compare)
    n_experts = layer.n_experts
    activation = np.full(n_experts, 1.0 / n_experts)
    uniform_loss = float(sum(
        activation[e] * layer.by_bit[b_uniform][e].sum()
        for e in range(n_experts)
    ))

    # DP solution at the requested bucket count
    expert_rates = [
        {b: layer.by_bit[b][e] for b in bits if b in layer.by_bit}
        for e in range(n_experts)
    ]
    from dp_utils import enum_optimal_m_scheme_global_fast
    _, neuron_bits_per_expert = enum_optimal_m_scheme_global_fast(
        expert_rates, activation, slice_expert_num, target_bpw=bpw)
    dp_loss = sum(
        activation[e] * _loss_from_neuron_bits(expert_rates[e], neuron_bits_per_expert[e])
        for e in range(n_experts)
    )
    if dp_loss <= 0:
        return float("nan")
    return uniform_loss / dp_loss, uniform_loss, dp_loss


def bucket_sweep(
    models: List[str],
    quantmode: str = DEFAULT_QUANTMODE,
    rank_mode: str = DEFAULT_RANK_MODE,
    bpw: int = 2,
    bucket_counts: Tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    layer_start: int = 0,
    num_layers: int = 4,
    out_dir: str = OUT_ROOT,
    save_pdf: bool = False,
) -> str:
    """How close does discrete DP get to the AM/GM upper bound, as we shrink
    bucket size from "1 bucket per expert" (= expert granularity) toward
    finer per-expert buckets?

    Y-axis: `uniform_loss / DP_loss` — the realized loss-reduction multiplier.
    X-axis: `slice_expert_num` (= buckets per expert) on log scale.
    Yellow band: per-layer AM/GM range [min, max] across the layers in the
                 panel — each layer has its own headroom ceiling. The dashed
                 line inside is the cross-layer mean, included as a single
                 summary number; individual per-layer DP curves may exceed
                 this mean without violating the bound (they should stay
                 within the band).

    Note on the AM/GM "bound": `_amgm_ratio` uses the empirical neuron-loss
    distribution at `bpw` (rather than fitting the full log-quadratic model
    `log L_i(b) = p b² + q b + r_i` from multiple bits). This is a tight,
    practical proxy for the closed-form ceiling at that bpw and is what the
    `amgm` panel also plots — it is not a hard mathematical guarantee across
    all bit allocations, just the AM/GM of the per-neuron loss at b=bpw.

    The story this figure tells:
        • slice=1 (expert granularity): the multiplier `compare`'s `expert`
          curve is implicitly measuring at this bpw.
        • slice → n_neurons (theoretical right edge): the discrete DP would
          approach the continuous-bit AM/GM bound from below — we sweep up
          to 32 buckets/expert because DP state grows quadratically and
          past ~32 the curve has already plateaued in practice.
        • The plateau between slice=8 and slice=32 shows that few buckets
          already capture most of the closed-form headroom — which justifies
          DartMoQP's default `slice_expert_num=8` design choice.

    This is the missing logical bridge between `amgm` (closed-form, continuous
    bit) and `layer_expert_neuron_compare` (DP, discrete bit, slice=8).

    DP cost is roughly O((s · n_experts)² · bits) per layer-per-slice, so
    `max_layers` defaults to 4 to keep total runtime under ~5 minutes.
    """
    fig, axes = _multi_model_axes(models, width_per=4.5, height=4.0)

    for ax, model_id in zip(axes, models):
        print("\n=== bucket_sweep for model_id =", model_id, "===")
        layers = load_all_layers(model_id, quantmode, rank_mode,
                                 bits=(1, 2, 3, 4), layer_start=layer_start, num_layers=num_layers)
        if not layers:
            ax.text(0.5, 0.5, "no cache", ha="center", va="center",
                    transform=ax.transAxes); ax.set_title(model_label(model_id))
            continue

        bits = sorted(set.intersection(*[set(L.by_bit.keys()) for L in layers]))
        bits = [b for b in bits if b > 0]   # 0bit not used by uniform baseline

        # Per-layer ratio curves (light) + mean curve (bold).
        # Note: cmap is indexed by enumeration order (i.e. position in the
        # cached `layers` list), not by absolute layer_idx. Same-cmap-position
        # ----- per-layer DP curves + per-layer AM/GM bound, color-paired -----
        # Each layer gets a distinct viridis color. The DP curve (solid) and
        # its AM/GM bound (dashed horizontal of the SAME color) form a visual
        # pair, so a reader can read off "this DP curve approaches THAT bound"
        # without crossing legends.
        n_layers = len(layers)
        cmap = plt.get_cmap("viridis", max(n_layers, 2))
        for li, L in enumerate(layers):
            print(f"  Layer {L.layer_idx}")
            n_neurons = L.n_neurons
            curve = []
            uniform_loss_ = []
            dp_loss_ = []
            for s in bucket_counts:
                if s > n_neurons:
                    curve.append(np.nan); continue
                r, u, d = _dp_uniform_over_dp(L, bits, bpw, s)
                curve.append(r)
                uniform_loss_.append(u)
                dp_loss_.append(d.item())
            curve = np.asarray(curve, dtype=float)

            amgm, c_mean, c_std, c_gm = _amgm_ratio(L, bpw)
            print(f"Curve: {curve}, uniform_loss: {uniform_loss_}, dp_loss: {dp_loss_}, amgm: {amgm:.4f}, c_mean: {c_mean:.4f}, c_std: {c_std:.4f}, c_gm: {c_gm:.4f}")

            if np.isfinite(amgm) and amgm > 0:
                curve_normalized = curve / amgm

                color = cmap(li)
                # Normalized DP curve (solid) — colored per layer
                ax.plot(bucket_counts, curve_normalized, color=color, lw=2, marker="o",
                        ms=4, alpha=0.9, zorder=3,
                        label=f"L{L.layer_idx}: DP  (sup = {amgm:.2f}×)")
            else:
                print(f"  Skipping layer {L.layer_idx}: invalid amgm = {amgm}")

        # Common normalized upper bound at y=1
        ax.axhline(1.0, color="k", ls="--", lw=1.5, alpha=0.7, zorder=2,
                   label="Normalized AM/GM bound")

        y_top = 1.1
        ax.set_ylim(0.0, y_top)

        # Mark `expert` granularity (slice=1) and DartMoQP default (slice=8)
        ax.axvline(1, color="#cc7722", ls=":", lw=0.8, alpha=0.6)
        ax.axvline(8, color="#3a7ca5", ls=":", lw=0.8, alpha=0.6)

        ax.set_xscale("log", base=2)
        ax.set_xticks([s for s in bucket_counts if s & (s - 1) == 0])
        ax.set_xticklabels([str(s) for s in bucket_counts if s & (s - 1) == 0],
                           fontsize=8)
        ax.set_xlabel("Buckets per expert (slice_expert_num)")
        ax.set_title(f"{model_label(model_id)}  "
                     f"@ {bpw}-bit, start={layer_start} ({len(layers)} layers)")
        ax.legend(loc="best", fontsize=7, framealpha=0.85,
                  handlelength=1.5, borderpad=0.4)
        ax.grid(True, alpha=0.3, which="both")

    axes[0].set_ylabel("Normalized loss reduction (DP ratio / AM/GM bound)")
    # fig.suptitle(f"Bucket-count sweep — normalized — start layer {layer_start}",
    #              fontsize=11, y=1.02)

    os.makedirs(out_dir, exist_ok=True)
    fp_png = os.path.join(out_dir, f"bucket_sweep_{rank_mode}_b{bpw}_start{layer_start}_n{len(layers)}.png")
    plt.tight_layout()
    plt.savefig(fp_png)
    if save_pdf:
        fp_pdf = os.path.join(out_dir, f"bucket_sweep_{rank_mode}_b{bpw}_start{layer_start}_n{len(layers)}.pdf")
        plt.savefig(fp_pdf)
    plt.close(fig)
    print(f"[bucket_sweep] saved {fp_png}" + (f" and {fp_pdf}" if save_pdf else ""))
    return fp_png


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    apply_paper_style()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="short cache id OR full model path; default: every model with cache")
    parser.add_argument("--quantmode", default=DEFAULT_QUANTMODE)
    parser.add_argument("--rank-mode", default=DEFAULT_RANK_MODE)
    parser.add_argument("--bit", type=int, default=DEFAULT_BIT)
    parser.add_argument("--slice-expert-num", type=int, default=8)
    parser.add_argument("--layer-start", type=int, default=0,
                        help="start layer index (inclusive) to use; -1 means the last num-layers layers")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="number of layers to use, default: 4")
    parser.add_argument("--skip", nargs="+", default=[],
                        choices=["amgm", "top10ratio", "act_vs_sens",
                                 "bucket_sweep", "layer_expert_neuron_compare"],
                        help="panel names to skip")
    parser.add_argument("--pdf", action="store_true", default=False,
                        help="also save PDF copies alongside PNGs")
    args = parser.parse_args()

    if args.model:
        models = [resolve_model_id(args.model)]
    else:
        models = discover_models(args.quantmode, args.rank_mode)
        if not models:
            print(f"no models with cache under {args.quantmode}/{args.rank_mode}")
            return

    # Two multi-model bar figures (one figure per panel, all models side-by-side)
    if "top10ratio" not in args.skip:
        print("\n=== top10ratio ===")
        top10ratio(models, args.quantmode, args.rank_mode, args.bit,
                   layer_start=args.layer_start, num_layers=args.num_layers,
                   save_pdf=args.pdf)
    if "amgm" not in args.skip:
        print("\n=== amgm ===")
        amgm(models, args.quantmode, args.rank_mode, args.bit,
             layer_start=args.layer_start, num_layers=args.num_layers,
             save_pdf=args.pdf)

    # act_vs_sens — per-model (different layer counts / activation files)
    # if "act_vs_sens" not in args.skip:
    #     for model_id in models:
    #         print(f"\n=== act_vs_sens  model: {model_id} ===")
    #         act_vs_sens(model_id, args.quantmode, args.rank_mode, args.bit,
    #                    layer_start=args.layer_start, num_layers=args.num_layers,
    #                    save_pdf=args.pdf)

    # bucket_sweep — bridge between amgm (bound) and compare (slice=8 DP)
    if "bucket_sweep" not in args.skip:
        print("\n=== bucket_sweep (all models on one figure) ===")
        bucket_sweep(
            models, args.quantmode, args.rank_mode,
            bpw=args.bit,
            layer_start=args.layer_start,
            num_layers=args.num_layers,
            save_pdf=args.pdf,
        )

    # layer_expert_neuron_compare — single multi-model figure
    if "layer_expert_neuron_compare" not in args.skip:
        print("\n=== layer_expert_neuron_compare (all models on one figure) ===")
        layer_expert_neuron_compare(
            models, args.quantmode, args.rank_mode,
            slice_expert_num=args.slice_expert_num,
            layer_start=args.layer_start,
            num_layers=args.num_layers,
            save_pdf=args.pdf,
        )


if __name__ == "__main__":
    main()
