"""Plot DP-related sub-expert score distributions from cached q_rates.

The script builds the same DP-style sub-experts used by DartMoQ:
sort neurons inside every expert by ``sort_bit`` loss, then split into
``slices_per_expert`` groups. It supports five views:

1. ``raw_slice_error``: ``sum(q_rates[b])`` per slice, one plot per bit.
2. ``dp_bit_loss``: ``sum(q_rates[b]) * expert_activation`` per slice, one plot per bit.
3. ``ordering_score``: ``mean(q_rates[sort_bit]) * expert_activation``.
4. ``assigned_loss_by_expert``: ``sum(q_rates[assigned_bit]) * activation`` after DP.
5. ``assigned_loss_by_order``: same assigned loss, plotted along DP order.
6. ``assigned_total_vs_uniform_by_expert``: per-expert DP total versus no-sort fixed-bit baseline.
7. ``uniform_loss_by_expert_unsorted``: fixed-bit unsorted slice boxplot baseline.

NaN/Inf cache values are converted to 0 on load, so extrapolated b0 NaNs do not
poison sorting or losses.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dp_utils import enum_optimal_m_scheme_global_fast
from viz._cache_io import apply_paper_style, model_label, resolve_model_id


ALL_VIEWS = [
    "raw_slice_error",
    "raw_neuron_error_unsorted",
    "dp_bit_loss",
    "ordering_score",
    "assigned_loss_by_expert",
    "assigned_loss_by_order",
    "assigned_total_vs_uniform_by_expert",
    "uniform_loss_by_expert_unsorted",
]


def _candidate_cache_dirs(
    cache_root: str,
    quantmode: str,
    rank_mode: str,
    model_id: str,
) -> List[str]:
    roots = [cache_root]
    if cache_root == "auto":
        roots = ["intermediate_result", "."]

    out = []
    for root in roots:
        out.append(os.path.join(root, f"quant_outlier_{quantmode}", rank_mode, model_id))
    return out


def resolve_cache_dir(
    cache_root: str,
    quantmode: str,
    rank_mode: str,
    model_id: str,
) -> str:
    candidates = _candidate_cache_dirs(cache_root, quantmode, rank_mode, model_id)
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError("Cannot find cache directory. Tried:\n  " + "\n  ".join(candidates))


def discover_bits(cache_dir: str, model_id: str, layer: int) -> List[int]:
    pat = re.compile(rf"{re.escape(model_id)}_L{layer}_b(\d+)\.pt$")
    bits = []
    for name in os.listdir(cache_dir):
        match = pat.match(name)
        if match:
            bits.append(int(match.group(1)))
    return sorted(bits)


def discover_layers(cache_dir: str, model_id: str, probe_bit: int) -> List[int]:
    pat = re.compile(rf"{re.escape(model_id)}_L(\d+)_b{probe_bit}\.pt$")
    layers = []
    for name in os.listdir(cache_dir):
        match = pat.match(name)
        if match:
            layers.append(int(match.group(1)))
    return sorted(layers)


def load_loss_matrix(cache_dir: str, model_id: str, layer: int, bit: int) -> np.ndarray:
    path = os.path.join(cache_dir, f"{model_id}_L{layer}_b{bit}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    raw = torch.load(path, map_location="cpu")
    arrays = [
        t.detach().float().cpu().numpy() if torch.is_tensor(t) else np.asarray(t, dtype=np.float32)
        for t in raw
    ]
    matrix = np.stack(arrays, axis=0).astype(np.float64, copy=False)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D expert x neuron matrix, got shape {matrix.shape}")
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def load_activation_rates(
    model_id: str,
    layer: int,
    activation_root: str,
) -> np.ndarray:
    base = os.path.join(activation_root, model_id)
    candidates = [os.path.join(base, f"{model_id}_L{layer}.pt")]

    for path in candidates:
        if os.path.exists(path):
            raw = torch.load(path, map_location="cpu")
            arr = raw.detach().float().cpu().numpy() if torch.is_tensor(raw) else np.asarray(raw)
            return np.nan_to_num(arr.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

    raise FileNotFoundError("Cannot find expert activation cache. Tried:\n  " + "\n  ".join(candidates))


def load_rate_matrices(cache_dir: str, model_id: str, layer: int, bits: Sequence[int]) -> Dict[int, np.ndarray]:
    return {bit: load_loss_matrix(cache_dir, model_id, layer, bit) for bit in bits}


def build_slice_table(
    matrices: Dict[int, np.ndarray],
    activation_rates: np.ndarray,
    sort_bit: int,
    slices_per_expert: int,
) -> pd.DataFrame:
    sort_matrix = matrices[sort_bit]
    n_experts, n_neurons = sort_matrix.shape
    if len(activation_rates) < n_experts:
        raise ValueError(f"activation cache has {len(activation_rates)} experts, expected at least {n_experts}")

    rows = []
    for expert_id in range(n_experts):
        sorted_idx = np.argsort(-sort_matrix[expert_id])
        split_indices: Sequence[np.ndarray] = np.array_split(sorted_idx, slices_per_expert)
        act_rate = float(activation_rates[expert_id])

        for slice_id, indices in enumerate(split_indices):
            sort_loss_sum = float(sort_matrix[expert_id, indices].sum())
            row = {
                "expert": expert_id,
                "slice": slice_id,
                "num_neurons": int(len(indices)),
                "expert_activation_rate": act_rate,
                "sort_bit": sort_bit,
                "sort_bit_loss_sum": sort_loss_sum,
                "ordering_score": (sort_loss_sum / max(len(indices), 1)) * act_rate,
                "neuron_indices": " ".join(str(int(i)) for i in indices),
            }
            for bit, matrix in matrices.items():
                loss_sum = float(matrix[expert_id, indices].sum())
                row[f"raw_b{bit}"] = loss_sum
                row[f"dp_b{bit}"] = loss_sum * act_rate
            rows.append(row)

    df = pd.DataFrame(rows)
    df["dp_order"] = df["ordering_score"].rank(method="first", ascending=False).astype(int) - 1
    return df.sort_values("dp_order").reset_index(drop=True)


def build_raw_neuron_table(matrix: np.ndarray) -> pd.DataFrame:
    rows = []
    n_experts, n_neurons = matrix.shape
    for expert_id in range(n_experts):
        for neuron_id in range(n_neurons):
            rows.append({
                "expert": expert_id,
                "neuron": neuron_id,
                "score": float(matrix[expert_id, neuron_id]),
            })
    return pd.DataFrame(rows)


def build_unsorted_fixed_bit_slice_table(
    matrices: Dict[int, np.ndarray],
    activation_rates: np.ndarray,
    uniform_bit: int,
    slices_per_expert: int,
) -> pd.DataFrame:
    matrix = matrices[uniform_bit]
    n_experts, n_neurons = matrix.shape
    if len(activation_rates) < n_experts:
        raise ValueError(f"activation cache has {len(activation_rates)} experts, expected at least {n_experts}")

    rows = []
    for expert_id in range(n_experts):
        split_indices: Sequence[np.ndarray] = np.array_split(np.arange(n_neurons), slices_per_expert)
        act_rate = float(activation_rates[expert_id])
        for slice_id, indices in enumerate(split_indices):
            raw_sum = float(matrix[expert_id, indices].sum())
            rows.append({
                "expert": expert_id,
                "slice": slice_id,
                "num_neurons": int(len(indices)),
                "expert_activation_rate": act_rate,
                "uniform_bit": uniform_bit,
                "raw_loss": raw_sum,
                "score": raw_sum * act_rate,
                "neuron_indices": " ".join(str(int(i)) for i in indices),
            })
    return pd.DataFrame(rows)


def compute_assigned_bits(
    matrices: Dict[int, np.ndarray],
    activation_rates: np.ndarray,
    slices_per_expert: int,
    target_bpw: float,
    bits: Sequence[int],
    disable_0bit_compensation: bool,
) -> List[List[int]]:
    n_experts = next(iter(matrices.values())).shape[0]
    expert_rates_list = []
    for expert_idx in range(n_experts):
        expert_rates_list.append({bit: matrices[bit][expert_idx] for bit in bits})

    scheme, _ = enum_optimal_m_scheme_global_fast(
        expert_rates_list,
        activation_rates,
        slices_per_expert,
        target_bpw=target_bpw,
        enable_0bit_compensation=not disable_0bit_compensation,
    )
    return scheme


def attach_assigned_loss(
    df: pd.DataFrame,
    assigned_bits: List[List[int]],
) -> pd.DataFrame:
    df = df.copy()
    assigned = []
    losses = []
    for row in df.itertuples(index=False):
        bit = int(assigned_bits[int(row.expert)][int(row.slice)])
        assigned.append(bit)
        losses.append(float(getattr(row, f"dp_b{bit}")))
    df["assigned_bit"] = assigned
    df["score"] = losses
    return df


def resolve_uniform_bit(target_bpw: float, available_bits: Sequence[int]) -> int:
    rounded = int(round(target_bpw))
    if abs(float(target_bpw) - rounded) > 1e-9:
        raise ValueError(
            "no-sort fixed-bit baseline requires an integer --target-bpw, "
            f"got {target_bpw}. For example, bpw=2.0 uses b2 for every neuron."
        )
    if rounded not in available_bits:
        raise ValueError(f"fixed-bit baseline needs b{rounded} cache; available bits={list(available_bits)}")
    return rounded


def build_assigned_vs_uniform_table(
    assigned_df: pd.DataFrame,
    matrices: Dict[int, np.ndarray],
    activation_rates: np.ndarray,
    uniform_bit: int,
) -> pd.DataFrame:
    assigned_total = (
        assigned_df
        .groupby("expert", as_index=False)
        .agg(
            assigned_total_loss=("score", "sum"),
            assigned_mean_loss=("score", "mean"),
            assigned_min_bit=("assigned_bit", "min"),
            assigned_max_bit=("assigned_bit", "max"),
        )
    )
    bit_counts = (
        assigned_df
        .pivot_table(index="expert", columns="assigned_bit", values="slice", aggfunc="count", fill_value=0)
        .add_prefix("assigned_b")
        .reset_index()
    )
    assigned_total = assigned_total.merge(bit_counts, on="expert", how="left")

    rows = []
    uniform_matrix = matrices[uniform_bit]
    n_experts = uniform_matrix.shape[0]
    for expert_id in range(n_experts):
        act_rate = float(activation_rates[expert_id])
        raw_sum = float(uniform_matrix[expert_id].sum())
        rows.append({
            "expert": expert_id,
            "uniform_bit": uniform_bit,
            "expert_activation_rate": act_rate,
            "uniform_raw_loss": raw_sum,
            "uniform_loss": raw_sum * act_rate,
        })
    uniform_df = pd.DataFrame(rows)
    out = assigned_total.merge(uniform_df, on="expert", how="left")
    out["score"] = out["assigned_total_loss"]
    out["delta_assigned_minus_uniform"] = out["assigned_total_loss"] - out["uniform_loss"]
    out["ratio_assigned_over_uniform"] = out["assigned_total_loss"] / out["uniform_loss"].replace(0.0, np.nan)
    return out


def add_metadata(
    df: pd.DataFrame,
    model_id: str,
    quantmode: str,
    rank_mode: str,
    layer: int,
    view: str,
    bit: Optional[int],
    target_bpw: Optional[float],
) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "model", model_id)
    df.insert(1, "quantmode", quantmode)
    df.insert(2, "rank_mode", rank_mode)
    df.insert(3, "layer", layer)
    df.insert(4, "view", view)
    df.insert(5, "bit", bit if bit is not None else "")
    df.insert(6, "target_bpw", target_bpw if target_bpw is not None else "")
    return df


def plot_expert_boxplot(df: pd.DataFrame, out_base: str, title: str, ylabel: str) -> Tuple[str, str]:
    apply_paper_style()
    plot_df = df[np.isfinite(df["score"])].copy()
    if plot_df.empty:
        raise ValueError("No finite score values to plot")

    experts = sorted(plot_df["expert"].unique())
    grouped = [
        plot_df.loc[plot_df["expert"] == expert, "score"].to_numpy()
        for expert in experts
    ]

    width = max(12.0, min(34.0, 0.18 * len(experts) + 4.0))
    fig, ax = plt.subplots(figsize=(width, 4.8))
    ax.boxplot(
        grouped,
        positions=np.arange(len(experts)),
        widths=0.62,
        patch_artist=True,
        showfliers=True,
        flierprops={"marker": ".", "markersize": 2, "markerfacecolor": "#555555", "markeredgecolor": "#555555", "alpha": 0.55},
        boxprops={"facecolor": "white", "edgecolor": "#222222", "linewidth": 0.9},
        medianprops={"color": "#d62728", "linewidth": 1.2},
        whiskerprops={"color": "#222222", "linewidth": 0.8},
        capprops={"color": "#222222", "linewidth": 0.8},
    )
    ax.set_xlabel("Expert")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(experts)))
    ax.set_xticklabels([str(x) for x in experts], rotation=90, fontsize=6)
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")
    fig.tight_layout()

    png_path = out_base + ".png"
    pdf_path = out_base + ".pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def plot_order_scatter(df: pd.DataFrame, out_base: str, title: str, ylabel: str) -> Tuple[str, str]:
    apply_paper_style()
    plot_df = df.sort_values("dp_order").copy()

    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    scatter = ax.scatter(
        plot_df["dp_order"],
        plot_df["score"],
        c=plot_df.get("assigned_bit", plot_df["expert"]),
        cmap="viridis",
        s=12,
        alpha=0.8,
    )
    ax.set_xlabel("DP Sorted Sub-Expert Index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.01)
    cbar.set_label("Assigned Bit" if "assigned_bit" in plot_df else "Expert")
    fig.tight_layout()

    png_path = out_base + ".png"
    pdf_path = out_base + ".pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def plot_expert_comparison(df: pd.DataFrame, out_base: str, title: str, ylabel: str) -> Tuple[str, str]:
    apply_paper_style()
    plot_df = df.sort_values("expert").copy()

    fig, ax = plt.subplots(figsize=(14.0, 4.8))
    ax.plot(
        plot_df["expert"],
        plot_df["assigned_total_loss"],
        marker="o",
        markersize=2.5,
        linewidth=1.0,
        label="DP mixed assigned total",
    )
    ax.plot(
        plot_df["expert"],
        plot_df["uniform_loss"],
        marker="s",
        markersize=2.5,
        linewidth=1.0,
        label="No-sort fixed-bit baseline",
    )
    ax.set_xlabel("Expert")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    png_path = out_base + ".png"
    pdf_path = out_base + ".pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def save_view(
    df: pd.DataFrame,
    out_dir: str,
    stem: str,
    title: str,
    ylabel: str,
    plot_kind: str = "expert_box",
) -> Tuple[str, str, str]:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, stem + ".csv")
    df.to_csv(csv_path, index=False)
    out_base = os.path.join(out_dir, stem)
    if plot_kind == "order":
        png_path, pdf_path = plot_order_scatter(df, out_base, title, ylabel)
    elif plot_kind == "expert_compare":
        png_path, pdf_path = plot_expert_comparison(df, out_base, title, ylabel)
    else:
        png_path, pdf_path = plot_expert_boxplot(df, out_base, title, ylabel)
    return csv_path, png_path, pdf_path


def parse_views(values: Sequence[str]) -> List[str]:
    if not values or "all" in values:
        return ALL_VIEWS
    out = []
    for value in values:
        if value not in ALL_VIEWS:
            raise ValueError(f"unknown view {value!r}; choose from {ALL_VIEWS} or all")
        out.append(value)
    return out


def choose_random_layer(layers: Sequence[int], fixed_layer: int, seed: int) -> int:
    candidates = [layer for layer in layers if layer != fixed_layer]
    if not candidates:
        return fixed_layer
    rng = np.random.default_rng(seed)
    return int(rng.choice(candidates))


def run_for_layer(args: argparse.Namespace, model_id: str, cache_dir: str, layer: int) -> List[Tuple[str, str, str]]:
    available_bits = discover_bits(cache_dir, model_id, layer)
    bits = args.bits if args.bits else available_bits
    bits = [bit for bit in bits if bit in available_bits]
    if not bits:
        raise ValueError(f"No requested bits are available for L{layer}; available bits={available_bits}")

    views = parse_views(args.views)
    uniform_bit = None
    if "assigned_total_vs_uniform_by_expert" in views or "uniform_loss_by_expert_unsorted" in views:
        if args.target_bpw is None:
            raise ValueError("fixed-bit uniform baseline views require --target-bpw")
        uniform_bit = resolve_uniform_bit(args.target_bpw, available_bits)
        if uniform_bit not in bits:
            bits = sorted(set(bits + [uniform_bit]))

    only_unsorted = views == ["raw_neuron_error_unsorted"]
    sort_bit = args.sort_bit if args.sort_bit is not None else min(bits)
    if not only_unsorted:
        if sort_bit not in available_bits:
            raise ValueError(f"sort-bit b{sort_bit} cache not found for L{layer}; available bits={available_bits}")
        if sort_bit not in bits:
            bits = sorted(set(bits + [sort_bit]))

    matrices = load_rate_matrices(cache_dir, model_id, layer, bits)
    if only_unsorted:
        slice_df = pd.DataFrame()
    else:
        activation_rates = load_activation_rates(model_id, layer, args.activation_root)
        slice_df = build_slice_table(matrices, activation_rates, sort_bit, args.slices_per_expert)

    sort_dir = "unsorted" if only_unsorted else f"sort_b{sort_bit}"
    # Simplified: single output directory, all info in filename
    output_root = args.out_dir

    saved = []
    assigned_df = None
    assigned_bits = None
    if any(view.startswith("assigned_loss") for view in views) or "assigned_total_vs_uniform_by_expert" in views:
        if args.target_bpw is None:
            raise ValueError("assigned loss views require --target-bpw")
        assigned_bits = compute_assigned_bits(
            matrices=matrices,
            activation_rates=activation_rates,
            slices_per_expert=args.slices_per_expert,
            target_bpw=args.target_bpw,
            bits=bits,
            disable_0bit_compensation=args.disable_0bit_compensation,
        )
        assigned_df = attach_assigned_loss(slice_df, assigned_bits)

    for view in views:
        if view == "raw_slice_error":
            for bit in bits:
                df = slice_df.copy()
                df["score"] = df[f"raw_b{bit}"]
                df = add_metadata(df, model_id, args.quantmode, args.rank_mode, layer, view, bit, args.target_bpw)
                out_dir = output_root
                stem = f"{model_id}_{args.quantmode}_{args.rank_mode}_L{layer}_s{args.slices_per_expert}_{sort_dir}_{view}_b{bit}"
                saved.append(save_view(
                    df, out_dir, stem,
                    f"{model_label(model_id)} L{layer}: Raw Slice Error b{bit}",
                    f"sum(q_rates[b{bit}])",
                ))
        elif view == "raw_neuron_error_unsorted":
            for bit in bits:
                df = build_raw_neuron_table(matrices[bit])
                df = add_metadata(df, model_id, args.quantmode, args.rank_mode, layer, view, bit, args.target_bpw)
                out_dir = output_root
                stem = f"{model_id}_{args.quantmode}_{args.rank_mode}_L{layer}_s{args.slices_per_expert}_{sort_dir}_{view}_b{bit}"
                saved.append(save_view(
                    df, out_dir, stem,
                    f"{model_label(model_id)} L{layer}: Raw Unsorted Neuron Error b{bit}",
                    f"q_rates[b{bit}] per neuron",
                ))
        elif view == "dp_bit_loss":
            for bit in bits:
                df = slice_df.copy()
                df["score"] = df[f"dp_b{bit}"]
                df = add_metadata(df, model_id, args.quantmode, args.rank_mode, layer, view, bit, args.target_bpw)
                out_dir = output_root
                stem = f"{model_id}_{args.quantmode}_{args.rank_mode}_L{layer}_s{args.slices_per_expert}_{sort_dir}_{view}_b{bit}"
                saved.append(save_view(
                    df, out_dir, stem,
                    f"{model_label(model_id)} L{layer}: DP Bit Loss b{bit}",
                    f"sum(q_rates[b{bit}]) x activation",
                ))
        elif view == "ordering_score":
            df = slice_df.copy()
            df["score"] = df["ordering_score"]
            df = add_metadata(df, model_id, args.quantmode, args.rank_mode, layer, view, sort_bit, args.target_bpw)
            out_dir = output_root
            stem = f"{model_id}_{args.quantmode}_{args.rank_mode}_L{layer}_s{args.slices_per_expert}_{sort_dir}_{view}_sortb{sort_bit}"
            saved.append(save_view(
                df, out_dir, stem,
                f"{model_label(model_id)} L{layer}: DP Ordering Score",
                f"mean(q_rates[b{sort_bit}]) x activation",
            ))
        elif view == "assigned_loss_by_expert":
            assert assigned_df is not None
            df = add_metadata(assigned_df, model_id, args.quantmode, args.rank_mode, layer, view, None, args.target_bpw)
            comp = "no0comp" if args.disable_0bit_compensation else "0comp"
            out_dir = output_root
            stem = f"{model_id}_{args.quantmode}_{args.rank_mode}_L{layer}_s{args.slices_per_expert}_{sort_dir}_{view}_bpw{args.target_bpw:g}_{comp}"
            saved.append(save_view(
                df, out_dir, stem,
                f"{model_label(model_id)} L{layer}: Assigned Loss by Expert",
                "sum(q_rates[assigned_bit]) x activation",
            ))
        elif view == "assigned_loss_by_order":
            assert assigned_df is not None
            df = add_metadata(assigned_df, model_id, args.quantmode, args.rank_mode, layer, view, None, args.target_bpw)
            comp = "no0comp" if args.disable_0bit_compensation else "0comp"
            out_dir = output_root
            stem = f"{model_id}_{args.quantmode}_{args.rank_mode}_L{layer}_s{args.slices_per_expert}_{sort_dir}_{view}_bpw{args.target_bpw:g}_{comp}"
            saved.append(save_view(
                df, out_dir, stem,
                f"{model_label(model_id)} L{layer}: Assigned Loss by DP Order",
                "sum(q_rates[assigned_bit]) x activation",
                plot_kind="order",
            ))
        elif view == "assigned_total_vs_uniform_by_expert":
            assert assigned_df is not None
            assert uniform_bit is not None
            df = build_assigned_vs_uniform_table(assigned_df, matrices, activation_rates, uniform_bit)
            df = add_metadata(df, model_id, args.quantmode, args.rank_mode, layer, view, uniform_bit, args.target_bpw)
            comp = "no0comp" if args.disable_0bit_compensation else "0comp"
            out_dir = output_root
            stem = f"{model_id}_{args.quantmode}_{args.rank_mode}_L{layer}_s{args.slices_per_expert}_{sort_dir}_{view}_bpw{args.target_bpw:g}_{comp}_uniformb{uniform_bit}"
            saved.append(save_view(
                df, out_dir, stem,
                f"{model_label(model_id)} L{layer}: DP Mixed Total vs Fixed b{uniform_bit}",
                "per-expert total loss x activation",
                plot_kind="expert_compare",
            ))
        elif view == "uniform_loss_by_expert_unsorted":
            assert uniform_bit is not None
            df = build_unsorted_fixed_bit_slice_table(matrices, activation_rates, uniform_bit, args.slices_per_expert)
            df = add_metadata(df, model_id, args.quantmode, args.rank_mode, layer, view, uniform_bit, args.target_bpw)
            out_dir = output_root
            stem = f"{model_id}_{args.quantmode}_{args.rank_mode}_L{layer}_s{args.slices_per_expert}_unsorted_{view}_bpw{args.target_bpw:g}_uniformb{uniform_bit}"
            saved.append(save_view(
                df, out_dir, stem,
                f"{model_label(model_id)} L{layer}: No-Sort Fixed b{uniform_bit} Loss by Expert",
                f"sum(q_rates[b{uniform_bit}]) x activation",
            ))

    print(
        f"L{layer}: bits={bits}, sort_bit={sort_bit}, slices={args.slices_per_expert}, "
        f"records={len(slice_df)}, output_root={output_root}"
    )
    if assigned_bits is not None:
        flat_bits = [bit for expert_scheme in assigned_bits for bit in expert_scheme]
        print(f"L{layer}: assigned bit counts={dict(pd.Series(flat_bits).value_counts().sort_index())}")
    for csv_path, png_path, pdf_path in saved:
        print(f"  csv: {csv_path}")
        print(f"  png: {png_path}")
        print(f"  pdf: {pdf_path}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw DP-related sub-expert score views from cached q_rates.")
    parser.add_argument("--model", default="qwen3-30b-a3b", help="model id or model path")
    parser.add_argument("--quantmode", default="turboquant", choices=["turboquant", "gptq"])
    parser.add_argument("--rank-mode", "--rank_mode", dest="rank_mode", default="turboquant_innerproduct")
    parser.add_argument("--cache-root", default="auto", help="'auto', 'intermediate_result', '.', or another root")
    parser.add_argument("--activation-root", default=os.path.join("intermediate_result", "expert_activate"))
    parser.add_argument("--layers", type=int, nargs="+", default=[7], help="layers to plot")
    parser.add_argument("--include-random-layer", action="store_true", help="also plot one random available layer")
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--bits", type=int, nargs="+", default=[0, 1, 2, 3, 4], choices=[0, 1, 2, 3, 4])
    parser.add_argument("--sort-bit", type=int, default=None, choices=[0, 1, 2, 3, 4],
                        help="bit used to sort neurons before slicing; default is lowest requested available bit")
    parser.add_argument("--slices-per-expert", type=int, default=8)
    parser.add_argument("--target-bpw", type=float, default=None, help="required for assigned-loss views")
    parser.add_argument("--disable-0bit-compensation", action="store_true")
    parser.add_argument("--views", nargs="+", default=["all"], help=f"subset of {ALL_VIEWS}, or all")
    parser.add_argument("--out-dir", default=os.path.join("plot", "dp_score_views"))
    args = parser.parse_args()

    model_id = resolve_model_id(args.model)
    cache_dir = resolve_cache_dir(args.cache_root, args.quantmode, args.rank_mode, model_id)

    layers = list(dict.fromkeys(args.layers))
    if args.include_random_layer:
        probe_bit = args.bits[0] if args.bits else 0
        available_layers = discover_layers(cache_dir, model_id, probe_bit)
        random_layer = choose_random_layer(available_layers, fixed_layer=layers[0], seed=args.random_seed)
        if random_layer not in layers:
            layers.append(random_layer)

    all_saved = []
    for layer in layers:
        all_saved.extend(run_for_layer(args, model_id, cache_dir, layer))
    print(f"done: saved {len(all_saved)} view filesets")


if __name__ == "__main__":
    main()
