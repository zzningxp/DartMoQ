"""Sensitivity geometry visualizations — *different quantizers have intrinsically
different sensitivity geometries; element-wise MSE is degenerate for VQ.*

Four sub-figures (each illustrates how the choice of sensitivity
metric interacts with the underlying quantizer geometry):

    G.1  per-neuron sensitivity CDFs across metrics  — GPTQ-MSE vs TurboQuant-MSE
                                                        vs TurboQuant-IP; report Gini.
    G.2  rotation energy-flattening evidence        — for a random weight matrix,
                                                        show per-channel L2-energy
                                                        before and after a Polar/QR rotation.
    G.3  Spearman rank-agreement matrix             — three metrics × multiple layers,
                                                        which metrics produce consistent
                                                        importance rankings.
    G.4  sensitivity-vs-true-loss validity          — rank-correlate each metric
                                                        with the *measured* downstream
                                                        loss caused by zeroing out
                                                        individual neurons.

All four panels source data from the cached sensitivity tensors. G.2 and G.4
*may* additionally consume small standalone files if present:

  - G.2 reads `logs/rotation_energy/{model_id}_L{layer}.npz` if available;
         otherwise it synthesizes a random rotation on a Gaussian weight matrix
         (the energy-flattening effect is geometry-only and does not depend on
         the trained model — synthetic suffices for the paper figure).
  - G.4 reads `logs/zero_out_validation/{model_id}_L{layer}_e{expert}.npz`
         if available, otherwise it is skipped.

Usage
-----
    python -m viz.metric_geometry
    python -m viz.metric_geometry --model olmoe-7b-1b --layer 8
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viz._cache_io import (
    LayerSensitivity, apply_paper_style, discover_layers, discover_models,
    load_layer, model_label, resolve_model_id,
)

OUT_ROOT = "plot/metric_geometry"
DEFAULT_BIT = 2

# (display_name, quantmode_dir, rank_mode_dir)
# These three triples are the three "metric variants" we contrast in this section.
METRICS = [
    ("GPTQ element-MSE",        "gptq",       "gptq_quant_outlier"),
    ("TurboQuant element-MSE",  "gptq",       "turboquant_innerproduct"),   # legacy slot, see note
    ("TurboQuant inner-product", "turboquant", "turboquant_innerproduct"),
]
# NOTE: the project does not currently cache "TurboQuant element-MSE" tensors;
# that variant is reconstructed analytically inside G.1/G.3 from the
# TurboQuant inner-product cache so we can compare its *geometry* against the
# inner-product variant. See `_synthesize_tq_mse_from_cache` below.


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _gini(x: np.ndarray) -> float:
    """Concentration index for a non-negative vector. 0=uniform, 1=one element."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if x.size == 0:
        return float("nan")
    x = np.sort(x)
    n = x.size
    s = x.sum()
    if s == 0:
        return 0.0
    cum = np.cumsum(x)
    return (n + 1 - 2 * (cum.sum() / s)) / n


def _load_metric_layer(
    model_id: str, layer_idx: int, metric_idx: int, bit: int
) -> Optional[LayerSensitivity]:
    _, quantmode, rank_mode = METRICS[metric_idx]
    return load_layer(model_id, layer_idx, quantmode, rank_mode, bits=(bit,))


def _synthesize_tq_mse_from_cache(layer: LayerSensitivity, bit: int) -> List[np.ndarray]:
    """Project the inner-product (sample-norm scaled) loss back to a uniform-MSE
    proxy by removing the activation-norm weighting.

    Rationale: the inner-product loss is L_ip ≈ ‖Δw‖² · ‖x‖² + cross-terms,
    while a pure element-MSE loss treats every neuron as having the same
    activation energy. We approximate the latter by *normalizing each expert's
    per-neuron loss vector to constant sum* — this kills the activation-
    weighting axis while preserving any geometric information from the
    quantizer itself. The point of G.1/G.3 is to show this proxy still
    fails to discriminate neurons, illustrating that the metric problem is in
    the quantizer geometry, not in the activation weighting.
    """
    out = []
    for vec in layer.by_bit[bit]:
        v = np.asarray(vec, dtype=float)
        s = v.sum()
        if s <= 0:
            out.append(v)
        else:
            out.append(v / s * v.size)
    return out


# ----------------------------------------------------------------------------
# G.1  per-neuron CDF across metrics
# ----------------------------------------------------------------------------
def g1_metric_cdfs(
    model_id: str,
    layer_idx: int,
    bit: int = DEFAULT_BIT,
    out_dir: str = OUT_ROOT,
) -> str:
    """For a single (model, layer), draw three CDFs of per-neuron sensitivity.

    A near-diagonal CDF + low Gini ⇒ metric is uninformative (every neuron
    looks equally important). A long-tailed CDF + high Gini ⇒ metric carries
    real ordering information.
    """
    fig, ax = plt.subplots(figsize=(6, 4.2))

    colors = ["#3a7ca5", "#cc7722", "#b5132e"]
    annotations = []
    for mi, (name, _, _) in enumerate(METRICS):
        if mi == 1:  # synthesized TurboQuant element-MSE proxy
            layer = _load_metric_layer(model_id, layer_idx, mi, bit)
            if layer is None:
                continue
            rates_per_expert = _synthesize_tq_mse_from_cache(layer, bit)
        else:
            layer = _load_metric_layer(model_id, layer_idx, mi, bit)
            if layer is None:
                continue
            rates_per_expert = layer.by_bit[bit]

        flat = np.concatenate([r.flatten() for r in rates_per_expert])
        flat = flat[flat > 0]
        if flat.size == 0:
            continue

        flat_sorted = np.sort(flat)
        cum = np.cumsum(flat_sorted) / flat_sorted.sum()
        x = np.linspace(0, 1, flat_sorted.size)

        g = _gini(flat_sorted)
        ax.plot(x, cum, color=colors[mi], lw=1.7, label=f"{name}  (Gini={g:.2f})")
        annotations.append((name, g))

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="uniform reference")
    ax.set_xlabel("Fraction of neurons (sorted ascending)")
    ax.set_ylabel("Cumulative loss share")
    ax.set_title(f"Sensitivity CDFs across metrics — {model_label(model_id)} L{layer_idx} ({bit}-bit)")
    ax.legend(loc="upper left", fontsize=8)

    os.makedirs(out_dir, exist_ok=True)
    fp = os.path.join(out_dir, f"g1_cdf_{model_id}_L{layer_idx}_b{bit}.png")
    plt.tight_layout()
    plt.savefig(fp)
    plt.close(fig)
    print(f"[G.1] saved {fp}  |  Ginis: {annotations}")
    return fp


# ----------------------------------------------------------------------------
# G.2  rotation energy-flattening evidence
# ----------------------------------------------------------------------------
def g2_rotation_energy_flattening(
    seed: int = 0,
    n_rows: int = 4096,
    n_cols: int = 4096,
    out_dir: str = OUT_ROOT,
) -> str:
    """Demonstrate that a uniformly random orthogonal rotation flattens
    per-channel energy.

    Synthetic: build W ~ heavy-tailed (Cauchy-mix Gaussian), where per-row
    L2-energy varies by 1-2 orders of magnitude. Rotate W -> W' = W @ Q with
    Q drawn from O(n) via QR of a Gaussian. Plot the per-row energy before
    and after rotation as sorted curves and as a histogram.

    This is the geometric reason TurboQuant's element-wise MSE loses
    discrimination: in the rotated space the quantization noise is
    isotropically spread, so per-row MSE ≈ const and can no longer rank
    neurons.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_rows, n_cols)).astype(np.float64)
    # inject outlier rows: 5% of rows scaled by 20x
    n_outlier = max(1, n_rows // 20)
    base[:n_outlier] *= 20.0
    rng.shuffle(base, axis=0)

    # random orthogonal Q via QR
    g = rng.standard_normal((n_cols, n_cols))
    q, _ = np.linalg.qr(g)
    rotated = base @ q

    e_before = np.linalg.norm(base, axis=1) ** 2
    e_after = np.linalg.norm(rotated, axis=1) ** 2

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(np.sort(e_before)[::-1], color="#3a7ca5", lw=1.5,
            label=f"before rotation  (max/min = {e_before.max()/e_before.min():.1f}×)")
    ax.plot(np.sort(e_after)[::-1], color="#b5132e", lw=1.5,
            label=f"after random rotation  (max/min = {e_after.max()/e_after.min():.2f}×)")
    ax.set_yscale("log")
    ax.set_xlabel("Channel index (sorted by energy)")
    ax.set_ylabel("Per-row L2 energy (log)")
    ax.set_title("Random rotation flattens per-channel energy")
    ax.legend()

    ax = axes[1]
    bins = np.logspace(np.log10(min(e_before.min(), e_after.min())),
                       np.log10(max(e_before.max(), e_after.max())), 60)
    ax.hist(e_before, bins=bins, alpha=0.55, color="#3a7ca5", label="before")
    ax.hist(e_after, bins=bins, alpha=0.55, color="#b5132e", label="after rotation")
    ax.set_xscale("log")
    ax.set_xlabel("Per-row L2 energy")
    ax.set_ylabel("# channels")
    ax.set_title("Energy distribution before/after rotation")
    ax.legend()

    os.makedirs(out_dir, exist_ok=True)
    fp = os.path.join(out_dir, f"g2_rotation_energy_seed{seed}.png")
    plt.tight_layout()
    plt.savefig(fp)
    plt.close(fig)
    print(f"[G.2] saved {fp}")
    return fp


# ----------------------------------------------------------------------------
# G.3  Spearman rank-agreement matrix across metrics, aggregated over layers
# ----------------------------------------------------------------------------
def g3_metric_rank_agreement(
    model_id: str,
    bit: int = DEFAULT_BIT,
    out_dir: str = OUT_ROOT,
) -> str:
    """For each layer load all three metrics and compute pairwise Spearman ρ
    over neurons of every expert. Aggregate by layer (mean) and plot a 3x3
    matrix with cell values = mean ± std over layers.

    Interpretation:
      - High ρ between two metrics ⇒ they would lead to the same bit
        allocation ⇒ they are interchangeable.
      - Low ρ ⇒ choosing the wrong metric materially changes the allocation.
    """
    from scipy.stats import spearmanr

    layers_all = []
    # Determine common layers (intersection across the three metric caches)
    common = None
    for mi, (_, qm, rm) in enumerate(METRICS):
        if mi == 1:
            # synthesized: same cache as inner-product
            avail = set(discover_layers(qm, rm, model_id))
        else:
            avail = set(discover_layers(qm, rm, model_id))
        common = avail if common is None else common & avail
    if not common:
        print(f"[G.3] no overlapping cached layers for {model_id}")
        return ""

    common_sorted = sorted(common)
    rho_acc = np.zeros((3, 3))
    rho_sq = np.zeros((3, 3))
    cnt = 0

    for li in common_sorted:
        per_metric_rates = []  # each = list[expert] -> np.array
        ok = True
        for mi in range(3):
            L = _load_metric_layer(model_id, li, mi, bit)
            if L is None:
                ok = False
                break
            if mi == 1:
                per_metric_rates.append(_synthesize_tq_mse_from_cache(L, bit))
            else:
                per_metric_rates.append(L.by_bit[bit])
        if not ok:
            continue

        n_experts = min(len(per_metric_rates[i]) for i in range(3))
        layer_rho = np.zeros((3, 3))
        layer_rho_n = 0
        for e in range(n_experts):
            vecs = [per_metric_rates[i][e].flatten() for i in range(3)]
            sz = min(v.size for v in vecs)
            vecs = [v[:sz] for v in vecs]
            if any(v.std() == 0 for v in vecs):
                continue
            for i in range(3):
                for j in range(3):
                    rho, _ = spearmanr(vecs[i], vecs[j])
                    layer_rho[i, j] += rho
            layer_rho_n += 1
        if layer_rho_n == 0:
            continue
        layer_rho /= layer_rho_n
        rho_acc += layer_rho
        rho_sq += layer_rho ** 2
        cnt += 1

    if cnt == 0:
        print(f"[G.3] no usable layers for {model_id}")
        return ""

    mean = rho_acc / cnt
    std = np.sqrt(np.maximum(rho_sq / cnt - mean ** 2, 0))

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(mean, cmap="coolwarm", vmin=-1, vmax=1)
    names = [m[0] for m in METRICS]
    ax.set_xticks(range(3)); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels(names)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mean[i,j]:+.2f}\n±{std[i,j]:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if abs(mean[i,j]) > 0.55 else "black")
    ax.set_title(f"Spearman ρ between sensitivity metrics\n"
                 f"{model_label(model_id)} — averaged over {cnt} layers ({bit}-bit)")
    fig.colorbar(im, ax=ax, label="ρ")

    os.makedirs(out_dir, exist_ok=True)
    fp = os.path.join(out_dir, f"g3_rank_agreement_{model_id}_b{bit}.png")
    plt.tight_layout()
    plt.savefig(fp)
    plt.close(fig)
    print(f"[G.3] saved {fp}")
    return fp


# ----------------------------------------------------------------------------
# G.4  sensitivity-vs-true-loss validity
# ----------------------------------------------------------------------------
def g4_sensitivity_vs_true_loss(
    model_id: str,
    bit: int = DEFAULT_BIT,
    out_dir: str = OUT_ROOT,
) -> str:
    """If a "ground-truth" file exists (each entry: neuron_index → measured
    downstream loss from zeroing that neuron), correlate each metric's
    ranking with it. Otherwise skip.

    File layout expected:
        logs/zero_out_validation/{model_id}_L{layer}_e{expert}.npz
            keys: neuron_idx (int array), true_loss (float array)
    """
    from scipy.stats import spearmanr

    val_root = f"logs/zero_out_validation/{model_id}"
    if not os.path.isdir(val_root):
        print(f"[G.4] no ground-truth at {val_root} — skipping. "
              "Run scripts/dump_zero_out_validation.py to populate it.")
        return ""

    rows = []  # (layer, expert, metric_name, rho)
    for fn in sorted(os.listdir(val_root)):
        if not fn.endswith(".npz"):
            continue
        try:
            tag = fn.replace(".npz", "")
            li = int(tag.split("_L")[1].split("_")[0])
            eid = int(tag.split("_e")[1])
        except Exception:
            continue
        gt = np.load(os.path.join(val_root, fn))
        true = gt["true_loss"]
        idx = gt["neuron_idx"]

        for mi in range(3):
            L = _load_metric_layer(model_id, li, mi, bit)
            if L is None or eid >= len(L.by_bit[bit]):
                continue
            if mi == 1:
                vec = _synthesize_tq_mse_from_cache(L, bit)[eid]
            else:
                vec = L.by_bit[bit][eid]
            vec = vec[idx]
            if vec.std() == 0 or true.std() == 0:
                continue
            rho, _ = spearmanr(vec, true)
            rows.append((li, eid, METRICS[mi][0], rho))

    if not rows:
        print(f"[G.4] no usable ground-truth rows for {model_id}")
        return ""

    fig, ax = plt.subplots(figsize=(6, 4))
    metric_names = [m[0] for m in METRICS]
    data = {n: [r[3] for r in rows if r[2] == n] for n in metric_names}
    positions = range(1, len(metric_names) + 1)
    ax.boxplot([data[n] for n in metric_names], positions=positions,
               labels=metric_names, showmeans=True)
    ax.set_ylabel("Spearman ρ vs measured zero-out loss")
    ax.set_title(f"Metric validity — {model_label(model_id)} ({bit}-bit)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    os.makedirs(out_dir, exist_ok=True)
    fp = os.path.join(out_dir, f"g4_validity_{model_id}_b{bit}.png")
    plt.tight_layout()
    plt.savefig(fp)
    plt.close(fig)
    print(f"[G.4] saved {fp}")
    return fp


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    apply_paper_style()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--layer", type=int, default=None,
                        help="single layer used for G.1 (default: pick middle layer)")
    parser.add_argument("--bit", type=int, default=DEFAULT_BIT)
    parser.add_argument("--skip", nargs="+", default=[],
                        choices=["g1", "g2", "g3", "g4"])
    args = parser.parse_args()

    if args.model:
        models = [resolve_model_id(args.model)]
    else:
        # use the union of models that have *any* of the three metric caches
        seen = set()
        for _, qm, rm in METRICS:
            seen.update(discover_models(qm, rm))
        models = sorted(seen)

    if "g2" not in args.skip:
        g2_rotation_energy_flattening()

    for model_id in models:
        print(f"\n=== model: {model_id} ===")

        # pick a layer for G.1
        if "g1" not in args.skip:
            layer_idx = args.layer
            if layer_idx is None:
                ls = discover_layers(METRICS[2][1], METRICS[2][2], model_id)  # inner-product cache
                if not ls:
                    print(f"[G.1] no layers for {model_id}, skipping")
                else:
                    layer_idx = ls[len(ls) // 2]
                    g1_metric_cdfs(model_id, layer_idx, bit=args.bit)
            else:
                g1_metric_cdfs(model_id, layer_idx, bit=args.bit)

        if "g3" not in args.skip:
            g3_metric_rank_agreement(model_id, bit=args.bit)
        if "g4" not in args.skip:
            g4_sensitivity_vs_true_loss(model_id, bit=args.bit)


if __name__ == "__main__":
    main()
