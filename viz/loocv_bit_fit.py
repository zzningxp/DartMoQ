"""Leave-one-bit-out cross-validation (LOOCV) of the log-quadratic loss law.

Answers reviewer 998X6 Q: "can the model fit on b=2,3,4 predict the measured
1-bit loss?". The sample-in R^2 reported in the paper is computed by fitting
all four bits; this script instead trains on exactly 3 of {1,2,3,4}, predicts
the held-out bit, and reports the out-of-sample error.

Protocol (aligned with viz/bit_loss_fit.py and dp_utils.compute_r_squared_for_rates):
  * unit = per-neuron loss L_i(b), cached under
    intermediate_result/quant_outlier_{quantmode}/{rank_mode}/{model_id}/
  * fit   : log L_i(b) = p b^2 + q b + r  on the 3 training bits
  * predict: Lhat_i(b*) = exp(p b*^2 + q b* + r)
  * errors : relative error |Lhat - L| / L  and Spearman rank correlation
    between Lhat and L across neurons (per layer, per expert, pooled).

The full pass over 5 models x {gptq, turboquant} x all layers x all experts
scans ~1.6e8 neurons; to stay fast we
  (1) vectorise the quadratic fit across all neurons of a layer (3x3 normal
      equation solved in closed form via a cached inverse), and
  (2) optionally cache the per-unit [L(b) for b in 1..4] tensor under
      intermediate_result/loocv_bit_fit/ so that re-runs (e.g. after a paper
      table tweak) do not re-read every .pt file.

Outputs
  * results/loocv_bit_fit/summary.json         - machine-readable full dump
  * results/loocv_bit_fit/table_per_bit.csv    - per (model, quant, heldout b)
  * results/loocv_bit_fit/table_per_model.csv  - aggregated headline numbers
  * figs/loocv_bit_fit_relerr.png              - median rel-err bar chart
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from viz._cache_io import (
    KNOWN_MODELS,
    cache_dir,
    discover_layers,
    discover_models,
    load_layer,
)

# quantmode -> the rank_mode subdirectory that matches the paper's main tables
QUANTS: List[Tuple[str, str]] = [
    ("gptq", "gptq_quant_outlier"),
    ("turboquant", "turboquant_innerproduct"),
]

BITS_FIT: Tuple[int, int, int, int] = (1, 2, 3, 4)

CACHE_SUBDIR = os.path.join("intermediate_result", "loocv_bit_fit")
ZERO_CACHE_SUBDIR = os.path.join(CACHE_SUBDIR, "zero_bit")


@dataclass
class FoldStats:
    """Aggregate statistics for one held-out bit over a pool of neurons."""
    n_units: int
    median_rel_err: float
    mean_rel_err: float
    p90_rel_err: float
    p99_rel_err: float
    trimmed_mean_rel_err: float  # mean of the central 99% (drop top 1% outliers)
    spearman_rho: float   # pooled Spearman across all units (pred vs true)
    median_layer_rho: float  # median of per-layer Spearman (more robust)


@dataclass
class ZeroBitStats:
    """Comparison of extrapolated 0-bit loss with direct zeroing loss."""
    n_units: int
    n_units_direct: int          # units with direct-zeroing cache available
    spearman_rho: float          # pooled rank corr between Lhat(0) and L_direct(0)
    median_layer_rho: float      # median per-layer Spearman
    median_ratio: float          # median of Lhat(0) / L_direct(0)
    mean_ratio: float
    clip_rate: float             # fraction of units with Lhat(0) < L(1)
    clip_rate_direct: float      # same but restricted to units that also have direct cache


# ---------------------------------------------------------------------------
# closed-form vectorised quadratic fit
# ---------------------------------------------------------------------------

def _quad_design_matrix(bits: np.ndarray) -> np.ndarray:
    """Return X of shape (len(bits), 3) with columns [b^2, b, 1]."""
    b = bits.astype(np.float64)
    return np.stack([b * b, b, np.ones_like(b)], axis=1)


def fit_quadratic_vectorised(
    logL: np.ndarray,      # (n_units, n_bits)
    train_bits: np.ndarray,  # (n_bits,)
) -> np.ndarray:
    """Solve min_c ||X c - logL_i||_2 for every unit i.

    Returns coefficient matrix of shape (n_units, 3) with columns [p, q, r].
    Uses the closed-form normal equation c = (X^T X)^{-1} X^T y with X the
    same for all units, so the inverse is computed once.
    """
    X = _quad_design_matrix(train_bits)              # (k, 3)
    XtX_inv = np.linalg.inv(X.T @ X)                 # (3, 3)
    # coefficients: (n_units, 3)
    coef = logL @ X @ XtX_inv
    return coef


# Guard against exp() overflow: a handful of pathological units produce
# log-predictions > 709 which would otherwise yield inf and wreck the mean.
_LOG_CLIP = 600.0


def predict_quadratic(coef: np.ndarray, b: float) -> np.ndarray:
    """Evaluate exp(p b^2 + q b + r) for every unit, clipping the log-domain."""
    log_pred = coef[:, 0] * b * b + coef[:, 1] * b + coef[:, 2]
    return np.exp(np.clip(log_pred, -_LOG_CLIP, _LOG_CLIP))


# ---------------------------------------------------------------------------
# per-layer fold computation
# ---------------------------------------------------------------------------

def _layer_fold_errors(
    loss_by_bit: Dict[int, np.ndarray],
    holdout_bit: int,
    train_bits: Iterable[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (true, pred) arrays of shape (n_units,) for one held-out bit.

    Units with any non-positive loss among the 4 bits are dropped (they cannot
    be log-fitted; this matches the R^2 convention in
    dp_utils.compute_r_squared_for_rates where such units are excluded from
    the mean/median pool).
    """
    bits = np.asarray(sorted(train_bits), dtype=np.int64)
    # stack to (n_units, 4)
    L = np.stack([loss_by_bit[b] for b in BITS_FIT], axis=1).astype(np.float64)
    valid = np.all(L > 0, axis=1)
    L = L[valid]
    if L.size == 0:
        return np.empty(0), np.empty(0)

    # training log-loss
    train_idx = [BITS_FIT.index(b) for b in bits]
    logL_train = np.log(L[:, train_idx])           # (n_units, 3)
    coef = fit_quadratic_vectorised(logL_train, bits.astype(np.float64))
    pred = predict_quadratic(coef, float(holdout_bit))

    true = L[:, BITS_FIT.index(holdout_bit)]
    return true, pred


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation; falls back to NaN on failure."""
    if x.size < 2:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(x, y)
        return float(rho)
    except Exception:
        return float("nan")


def compute_fold_for_layer(
    loss_by_bit: Dict[int, np.ndarray],
    holdout_bit: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper used by tests."""
    train_bits = [b for b in BITS_FIT if b != holdout_bit]
    return _layer_fold_errors(loss_by_bit, holdout_bit, train_bits)


# ---------------------------------------------------------------------------
# cache: dump the per-unit [L1..L4] tensor so re-runs are cheap
# ---------------------------------------------------------------------------

def _flatten_layer_to_matrix(layer) -> np.ndarray:
    """Return (n_units, 4) float64 matrix of L(b) for b in BITS_FIT."""
    cols = []
    for b in BITS_FIT:
        arr_list = layer.by_bit.get(b)
        if arr_list is None:
            raise KeyError(f"layer L{layer.layer_idx} missing bit {b}")
        cols.append(np.concatenate([np.asarray(a, dtype=np.float64) for a in arr_list]))
    return np.stack(cols, axis=1)


def build_unit_cache(
    model_id: str,
    quantmode: str,
    rank_mode: str,
    out_path: str,
) -> Dict[str, np.ndarray]:
    """Load every cached layer and concatenate to a single (n_units, 4) array.

    Returns a dict with 'L' (the matrix) and 'layer_slices' (list of
    (layer_idx, start, end) so we can still compute per-layer stats).
    """
    layers = discover_layers(quantmode, rank_mode, model_id)
    if not layers:
        raise RuntimeError(f"no cached layers for {model_id}/{rank_mode}")

    mats = []
    layer_slices: List[Tuple[int, int, int]] = []
    cursor = 0
    for li in layers:
        layer = load_layer(model_id, li, quantmode, rank_mode, bits=BITS_FIT)
        if layer is None:
            continue
        # require all 4 bits present
        if any(b not in layer.by_bit for b in BITS_FIT):
            print(f"  [skip] {model_id} L{li}: missing bits, has {sorted(layer.by_bit)}")
            continue
        mat = _flatten_layer_to_matrix(layer)
        mats.append(mat)
        layer_slices.append((li, cursor, cursor + mat.shape[0]))
        cursor += mat.shape[0]

    if not mats:
        raise RuntimeError(f"no complete layers for {model_id}/{rank_mode}")

    L = np.concatenate(mats, axis=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(
        {
            "L": torch.from_numpy(L),
            "layer_slices": layer_slices,
            "bits": list(BITS_FIT),
            "model_id": model_id,
            "rank_mode": rank_mode,
        },
        out_path,
    )
    return {"L": L, "layer_slices": layer_slices}


def load_unit_cache(out_path: str) -> Dict[str, object]:
    blob = torch.load(out_path, map_location="cpu", weights_only=False)
    L = blob["L"]
    if torch.is_tensor(L):
        L = L.numpy()
    return {"L": np.asarray(L, dtype=np.float64), "layer_slices": blob["layer_slices"]}


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def _fold_stats(true: np.ndarray, pred: np.ndarray, per_layer_rhos: List[float]) -> FoldStats:
    rel = np.abs(pred - true) / true
    p99 = float(np.percentile(rel, 99))
    trimmed = rel[rel <= p99]
    return FoldStats(
        n_units=int(true.size),
        median_rel_err=float(np.median(rel)),
        mean_rel_err=float(np.mean(rel)),
        p90_rel_err=float(np.percentile(rel, 90)),
        p99_rel_err=p99,
        trimmed_mean_rel_err=float(np.mean(trimmed)) if trimmed.size else float("nan"),
        spearman_rho=_spearman_rho(pred, true),
        median_layer_rho=float(np.median(per_layer_rhos)) if per_layer_rhos else float("nan"),
    )


def _summarise_matrix(L: np.ndarray, layer_slices, holdout_bit: int) -> FoldStats:
    """Run one fold on the cached per-unit matrix."""
    train_bits = [b for b in BITS_FIT if b != holdout_bit]
    train_idx = [BITS_FIT.index(b) for b in train_bits]
    test_idx = BITS_FIT.index(holdout_bit)

    valid = np.all(L > 0, axis=1)
    Lv = L[valid]
    if Lv.size == 0:
        return FoldStats(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))

    coef = fit_quadratic_vectorised(np.log(Lv[:, train_idx]), np.asarray(train_bits, dtype=np.float64))
    pred = predict_quadratic(coef, float(holdout_bit))
    true = Lv[:, test_idx]

    # per-layer Spearman on the *valid* subset of each layer
    per_layer_rhos: List[float] = []
    valid_idx = np.flatnonzero(valid)
    for _, start, end in layer_slices:
        sel = valid_idx[(valid_idx >= start) & (valid_idx < end)] - start
        if sel.size < 2:
            continue
        # map back to the Lv row indices
        lv_rows = np.searchsorted(valid_idx, sel + start)
        rho = _spearman_rho(pred[lv_rows], true[lv_rows])
        if not np.isnan(rho):
            per_layer_rhos.append(rho)

    return _fold_stats(true, pred, per_layer_rhos)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_loocv(
    model_ids: List[str],
    use_cache: bool,
    save_cache: bool,
    out_dir: str,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Return nested results[model][quantmode]['loocv'|'zero_bit']."""
    os.makedirs(out_dir, exist_ok=True)
    results: Dict[str, Dict[str, Dict[str, object]]] = {}

    for model_id in model_ids:
        results[model_id] = {}
        for quantmode, rank_mode in QUANTS:
            # Skip models without a complete raw cache (e.g. qwen3.5 has no
            # quant_outlier data yet) instead of crashing the whole batch.
            if not discover_layers(quantmode, rank_mode, model_id):
                print(f"[skip]  {model_id} / {rank_mode}: no cached layers")
                continue

            cache_path = os.path.join(CACHE_SUBDIR, f"{model_id}__{rank_mode}.pt")
            payload = None
            if use_cache and os.path.exists(cache_path):
                print(f"[cache] load {cache_path}")
                payload = load_unit_cache(cache_path)
            else:
                print(f"[scan]  {model_id} / {rank_mode}")
                payload = build_unit_cache(model_id, quantmode, rank_mode, cache_path) \
                    if save_cache else _build_matrix_no_save(model_id, quantmode, rank_mode)

            fold_stats: Dict[int, FoldStats] = {}
            for holdout in BITS_FIT:
                stats = _summarise_matrix(payload["L"], payload["layer_slices"], holdout)
                fold_stats[holdout] = stats
                print(
                    f"  {model_id:>20s} {quantmode:<10s} holdout b={holdout}: "
                    f"median={stats.median_rel_err:.4f} mean={stats.mean_rel_err:.4f} "
                    f"trim-mean={stats.trimmed_mean_rel_err:.4f} p90={stats.p90_rel_err:.4f} "
                    f"p99={stats.p99_rel_err:.4f} rho={stats.spearman_rho:.4f} "
                    f"layer-rho={stats.median_layer_rho:.4f} (n={stats.n_units})"
                )

            # zero-bit extrapolation validation
            zpath = os.path.join(ZERO_CACHE_SUBDIR, f"{model_id}__{rank_mode}__zero.pt")
            if use_cache and os.path.exists(zpath):
                print(f"[cache] load {zpath}")
                zpayload = load_zero_bit_cache(zpath)
            else:
                print(f"[scan]  {model_id} / {rank_mode} zero-bit")
                zpayload = build_zero_bit_cache(model_id, quantmode, rank_mode, zpath)
            zstats = compute_zero_bit_stats(zpayload)
            print(
                f"  {model_id:>20s} {quantmode:<10s} zero-bit: "
                f"rho={zstats.spearman_rho:.4f} layer-rho={zstats.median_layer_rho:.4f} "
                f"ratio(med)={zstats.median_ratio:.4f} clip={zstats.clip_rate:.4%} "
                f"(n_direct={zstats.n_units_direct})"
            )

            results[model_id][quantmode] = {"loocv": fold_stats, "zero_bit": zstats}

    _write_outputs(results, out_dir)
    return results


def _build_matrix_no_save(model_id, quantmode, rank_mode):
    """Same as build_unit_cache but without writing the cache file."""
    layers = discover_layers(quantmode, rank_mode, model_id)
    mats = []
    layer_slices = []
    cursor = 0
    for li in layers:
        layer = load_layer(model_id, li, quantmode, rank_mode, bits=BITS_FIT)
        if layer is None or any(b not in layer.by_bit for b in BITS_FIT):
            continue
        mat = _flatten_layer_to_matrix(layer)
        mats.append(mat)
        layer_slices.append((li, cursor, cursor + mat.shape[0]))
        cursor += mat.shape[0]
    return {"L": np.concatenate(mats, axis=0), "layer_slices": layer_slices}


# ---------------------------------------------------------------------------
# 0-bit extrapolation validation
# ---------------------------------------------------------------------------

def _extrapolate_0bit_from_1234(L: np.ndarray) -> np.ndarray:
    """Fit on b=1,2,3,4 and extrapolate to b=0 -> Lhat(0) = exp(r).

    Matches the extrapolation used in the paper / dp_utils.extrapolate_0bit_loss_fix:
    fit on all four measured bits and read off the intercept.  Units with any
    non-positive L(b) are assigned NaN so downstream statistics can exclude
    them explicitly.
    """
    valid = np.all(L > 0, axis=1)
    Lhat0 = np.full(L.shape[0], np.nan, dtype=np.float64)
    if not np.any(valid):
        return Lhat0
    coef = fit_quadratic_vectorised(np.log(L[valid]), np.asarray([1.0, 2.0, 3.0, 4.0]))
    Lhat0[valid] = predict_quadratic(coef, 0.0)
    return Lhat0


def build_zero_bit_cache(
    model_id: str,
    quantmode: str,
    rank_mode: str,
    out_path: str,
) -> Dict[str, object]:
    """Build per-unit cache of [L_direct(0), L(1), Lhat(0) from b=2,3,4].

    Only layers that have a `_b0direct.pt` file contribute direct values;
    layers without direct cache still contribute L(1) and Lhat(0) so that
    clip-rate statistics use the full population.
    """
    layers = discover_layers(quantmode, rank_mode, model_id)
    if not layers:
        raise RuntimeError(f"no cached layers for {model_id}/{rank_mode}")

    base_dir = cache_dir(quantmode, rank_mode, model_id)
    direct_list: List[np.ndarray] = []
    l1_list: List[np.ndarray] = []
    lhat0_list: List[np.ndarray] = []
    layer_slices: List[Tuple[int, int, int]] = []
    layer_has_direct: List[Tuple[int, bool]] = []
    cursor = 0

    for li in layers:
        layer = load_layer(model_id, li, quantmode, rank_mode, bits=(1, 2, 3, 4))
        if layer is None or any(b not in layer.by_bit for b in (1, 2, 3, 4)):
            continue
        mat = _flatten_layer_to_matrix(layer)  # (n_units, 4)

        direct_path = os.path.join(base_dir, f"{model_id}_L{li}_b0direct.pt")
        has_direct = os.path.exists(direct_path)
        if has_direct:
            raw = torch.load(direct_path, map_location="cpu")
            direct = np.concatenate([
                t.detach().float().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
                for t in raw
            ]).astype(np.float64)
        else:
            direct = np.full(mat.shape[0], np.nan, dtype=np.float64)

        lhat0 = _extrapolate_0bit_from_1234(mat)

        direct_list.append(direct)
        l1_list.append(mat[:, 0])
        lhat0_list.append(lhat0)
        layer_slices.append((li, cursor, cursor + mat.shape[0]))
        layer_has_direct.append((li, has_direct))
        cursor += mat.shape[0]

    payload = {
        "L_direct": np.concatenate(direct_list),
        "L1": np.concatenate(l1_list),
        "Lhat0": np.concatenate(lhat0_list),
        "layer_slices": layer_slices,
        "layer_has_direct": layer_has_direct,
        "model_id": model_id,
        "rank_mode": rank_mode,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(payload, out_path)
    return payload


def load_zero_bit_cache(out_path: str) -> Dict[str, object]:
    blob = torch.load(out_path, map_location="cpu", weights_only=False)
    for k in ("L_direct", "L1", "Lhat0"):
        if torch.is_tensor(blob[k]):
            blob[k] = blob[k].numpy()
        blob[k] = np.asarray(blob[k], dtype=np.float64)
    return blob


def compute_zero_bit_stats(payload: Dict[str, object]) -> ZeroBitStats:
    """Compute extrapolation-vs-direct and clipping statistics."""
    L_direct = payload["L_direct"]
    L1 = payload["L1"]
    Lhat0 = payload["Lhat0"]
    layer_slices = payload["layer_slices"]
    layer_has_direct = payload["layer_has_direct"]

    n_units = int(L1.size)
    valid = np.isfinite(Lhat0) & (L1 > 0)
    valid_direct = valid & np.isfinite(L_direct) & (L_direct > 0)
    n_valid = int(valid.sum())
    n_direct = int(valid_direct.sum())

    # overall clip rate: Lhat(0) < L(1)
    clip_mask = (Lhat0 < L1) & valid
    clip_rate = float(clip_mask.sum() / max(n_valid, 1))

    # clip rate restricted to units that also have direct cache
    clip_rate_direct = float((clip_mask & valid_direct).sum() / max(n_direct, 1))

    # ratios on the direct subset
    if n_direct > 0:
        ratio = Lhat0[valid_direct] / L_direct[valid_direct]
        median_ratio = float(np.median(ratio))
        mean_ratio = float(np.mean(ratio))
        rho = _spearman_rho(Lhat0[valid_direct], L_direct[valid_direct])
    else:
        median_ratio = mean_ratio = rho = float("nan")

    # per-layer Spearman on the direct subset
    per_layer_rhos: List[float] = []
    for (li, start, end), (_, has_d) in zip(layer_slices, layer_has_direct):
        if not has_d:
            continue
        sl = slice(start, end)
        mask = valid_direct[sl]
        if mask.sum() < 2:
            continue
        r = _spearman_rho(Lhat0[sl][mask], L_direct[sl][mask])
        if not np.isnan(r):
            per_layer_rhos.append(r)

    return ZeroBitStats(
        n_units=n_units,
        n_units_direct=n_direct,
        spearman_rho=rho,
        median_layer_rho=float(np.median(per_layer_rhos)) if per_layer_rhos else float("nan"),
        median_ratio=median_ratio,
        mean_ratio=mean_ratio,
        clip_rate=clip_rate,
        clip_rate_direct=clip_rate_direct,
    )


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def _write_outputs(results, out_dir: str) -> None:
    # JSON dump
    json_path = os.path.join(out_dir, "summary.json")
    serial = {}
    for m, quants in results.items():
        serial[m] = {}
        for q, payload in quants.items():
            serial[m][q] = {
                "loocv": {str(b): asdict(s) for b, s in payload["loocv"].items()},
                "zero_bit": asdict(payload["zero_bit"]),
            }
    with open(json_path, "w") as f:
        json.dump(serial, f, indent=2)

    # per-bit CSV
    csv_path = os.path.join(out_dir, "table_per_bit.csv")
    with open(csv_path, "w") as f:
        f.write("model,quant,holdout_bit,n_units,median_rel_err,mean_rel_err,trimmed_mean_rel_err,p90_rel_err,p99_rel_err,spearman_rho,median_layer_rho\n")
        for m, quants in results.items():
            for q, payload in quants.items():
                for b, s in sorted(payload["loocv"].items()):
                    f.write(f"{m},{q},{b},{s.n_units},{s.median_rel_err:.6f},{s.mean_rel_err:.6f},{s.trimmed_mean_rel_err:.6f},{s.p90_rel_err:.6f},{s.p99_rel_err:.6f},{s.spearman_rho:.6f},{s.median_layer_rho:.6f}\n")

    # per-model CSV: headline = b=1 fold (the reviewer's question) + pooled stats
    model_path = os.path.join(out_dir, "table_per_model.csv")
    with open(model_path, "w") as f:
        f.write("model,quant,b1_median_rel_err,b1_mean_rel_err,b1_trimmed_mean,b1_spearman,pooled_median_rel_err,pooled_mean_rel_err,pooled_spearman\n")
        for m, quants in results.items():
            for q, payload in quants.items():
                folds = payload["loocv"]
                b1 = folds.get(1)
                rels = [s.median_rel_err for s in folds.values() if s.n_units > 0]
                means = [s.mean_rel_err for s in folds.values() if s.n_units > 0]
                rhos = [s.spearman_rho for s in folds.values() if s.n_units > 0 and not np.isnan(s.spearman_rho)]
                f.write(
                    f"{m},{q},{b1.median_rel_err:.6f},{b1.mean_rel_err:.6f},{b1.trimmed_mean_rel_err:.6f},{b1.spearman_rho:.6f},"
                    f"{np.median(rels):.6f},{np.mean(means):.6f},{np.mean(rhos):.6f}\n"
                )

    # zero-bit validation CSV
    zero_path = os.path.join(out_dir, "table_zero_bit.csv")
    with open(zero_path, "w") as f:
        f.write("model,quant,n_units,n_units_direct,spearman_rho,median_layer_rho,median_ratio,mean_ratio,clip_rate,clip_rate_direct\n")
        for m, quants in results.items():
            for q, payload in quants.items():
                s = payload["zero_bit"]
                f.write(
                    f"{m},{q},{s.n_units},{s.n_units_direct},{s.spearman_rho:.6f},{s.median_layer_rho:.6f},"
                    f"{s.median_ratio:.6f},{s.mean_ratio:.6f},{s.clip_rate:.6f},{s.clip_rate_direct:.6f}\n"
                )

    print(f"\n[write] {json_path}\n[write] {csv_path}\n[write] {model_path}\n[write] {zero_path}")


def plot_summary(results, out_path: str) -> None:
    """Bar chart of median relative error per (model, quant, heldout bit)."""
    import matplotlib.pyplot as plt

    models = list(results.keys())
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 3), sharey=True)
    if n_models == 1:
        axes = [axes]

    colors = {"gptq": "#e67e22", "turboquant": "#3498db"}
    width = 0.35
    x = np.arange(len(BITS_FIT))

    for ax, model in zip(axes, models):
        for qi, (quantmode, _) in enumerate(QUANTS):
            folds = results[model].get(quantmode, {}).get("loocv", {})
            vals = [folds.get(b, FoldStats(0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)).median_rel_err for b in BITS_FIT]
            ax.bar(
                x + (qi - 0.5) * width,
                vals,
                width,
                label="GPTQ" if quantmode == "gptq" else "TurboQuant",
                color=colors[quantmode],
                alpha=0.85,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"b={b}" for b in BITS_FIT])
        ax.set_title(model, fontsize=11)
        ax.set_xlabel("held-out bit")
        ax.grid(True, alpha=0.3, axis="y")
        if ax is axes[0]:
            ax.set_ylabel("median relative error")
            ax.legend(fontsize=9)

    fig.suptitle("LOOCV: log-quadratic fit on 3 bits predicts the held-out bit")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[write] {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=None,
                        help="cache model ids; default = all models with a gptq cache")
    parser.add_argument("--use-cache", action="store_true",
                        help="load per-unit L(b) from intermediate_result/loocv_bit_fit/")
    parser.add_argument("--save-cache", action="store_true",
                        help="build and write the per-unit cache (default if --use-cache is off)")
    parser.add_argument("--out-dir", default=os.path.join("results", "loocv_bit_fit"))
    parser.add_argument("--plot", default=os.path.join("figs", "loocv_bit_fit_relerr.png"))
    args = parser.parse_args()

    if args.models:
        model_ids = args.models
    else:
        # default: every model that has the GPTQ cache (the 5 paper models)
        model_ids = discover_models("gptq", "gptq_quant_outlier")

    save_cache = args.save_cache or not args.use_cache
    results = run_loocv(model_ids, use_cache=args.use_cache, save_cache=save_cache, out_dir=args.out_dir)
    if args.plot:
        plot_summary(results, args.plot)


if __name__ == "__main__":
    main()
