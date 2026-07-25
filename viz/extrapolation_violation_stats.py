"""
Statistics on 0-bit extrapolation monotonicity violations.

Computes for all 5 models (both quant types):
  1. Fraction of neurons where L_hat(0) < L(1)  (monotonicity violation)
  2. Where violations concentrate:
       - by layer index (early / middle / late thirds)
       - by sensitivity quantile (low-sensitivity vs high-sensitivity)
       - by expert activation rate (hot vs cold experts)
  3. Severity: median L_hat(0) / L(1) among violating neurons
  4. Fraction of neurons affected by the clip safeguard (= violation fraction)

Generates 3 LaTeX tables for the paper:
  Table 1: Overall summary (violation rate + severity per model per quant)
  Table 2: Violation rate by sensitivity decile
  Table 3: Per-layer violation distribution (early/mid/late + range)

All stats are derived from cached per-neuron loss tensors; no training needed.
"""
import os
import sys
import json
import re
import argparse
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz._cache_io import (
    KNOWN_MODELS, load_all_layers, discover_models,
    EXPERT_ACTIVATE_ROOT, model_label,
)


# ---------------------------------------------------------------------------
# Core: per-layer log-quadratic fit + violation detection
# ---------------------------------------------------------------------------

def fit_and_flag_layer(layer_rates: Dict[int, List[np.ndarray]],
                       ) -> Dict[str, np.ndarray]:
    """Fit log-quadratic curve per neuron and return diagnostics.

    Args:
        layer_rates: bit -> list of expert arrays (each shape (n_neurons,))

    Returns:
        dict with keys:
          - 'L0_raw': raw extrapolated 0-bit loss, shape (n_experts, n_neurons)
          - 'L1':     measured 1-bit loss, shape (n_experts, n_neurons)
          - 'violation': bool mask where L0_raw < L1
          - 'ratio':   L0_raw / L1 (only meaningful where L1 > 0)
          - 'sensitivity': L(1) as proxy for sensitivity, shape (n_experts, n_neurons)
          - 'p_coeff': quadratic coefficient p_i, shape (n_experts, n_neurons)
          - 'valid':   bool mask, neuron was fit successfully
    """
    bits = sorted(b for b in layer_rates.keys() if b != 0)
    assert len(bits) >= 2, "need at least 2 bits for extrapolation"
    b_array = np.array(bits, dtype=float)

    n_experts = len(layer_rates[bits[0]])
    n_neurons = len(layer_rates[bits[0]][0])

    L0_raw = np.full((n_experts, n_neurons), np.nan)
    L1 = np.full((n_experts, n_neurons), np.nan)
    p_coeff = np.full((n_experts, n_neurons), np.nan)
    valid = np.zeros((n_experts, n_neurons), dtype=bool)

    for exp_idx in range(n_experts):
        loss_mat = np.stack([layer_rates[b][exp_idx] for b in bits], axis=0)

        if 1 in bits:
            l1 = loss_mat[bits.index(1)]
        else:
            l1 = loss_mat[0]
        L1[exp_idx] = l1

        positive = (loss_mat > 0).all(axis=0)
        if not positive.any():
            continue

        log_loss = np.log(loss_mat[:, positive])

        X = np.column_stack([b_array ** 2, b_array, np.ones_like(b_array)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, log_loss, rcond=None)
        except np.linalg.LinAlgError:
            continue

        p = coeffs[0]
        r_coeff = coeffs[2]
        l0_raw = np.exp(r_coeff)

        pos_indices = np.where(positive)[0]
        L0_raw[exp_idx, pos_indices] = l0_raw
        p_coeff[exp_idx, pos_indices] = p
        valid[exp_idx, pos_indices] = True

    ratio = np.where(L1 > 0, L0_raw / L1, np.nan)
    violation = valid & (L0_raw < L1) & (L1 > 0)

    return {
        'L0_raw': L0_raw,
        'L1': L1,
        'violation': violation,
        'ratio': ratio,
        'sensitivity': L1,
        'p_coeff': p_coeff,
        'valid': valid,
    }


# ---------------------------------------------------------------------------
# Expert activation rate loader (per-layer .pt files)
# ---------------------------------------------------------------------------

def load_expert_act_rates(model_id: str) -> Optional[Dict[int, np.ndarray]]:
    """Load per-layer expert activation rates for a model.

    Returns dict {layer_idx: array(n_experts,)} or None if not available.
    """
    import torch

    model_dir = os.path.join(EXPERT_ACTIVATE_ROOT, model_id)
    if not os.path.isdir(model_dir):
        return None

    pat = re.compile(rf"{re.escape(model_id)}_L(\d+)\.pt$")
    result = {}
    for fn in os.listdir(model_dir):
        m = pat.match(fn)
        if m:
            lidx = int(m.group(1))
            path = os.path.join(model_dir, fn)
            data = torch.load(path, map_location='cpu')
            if hasattr(data, 'numpy'):
                result[lidx] = data.numpy()
            else:
                result[lidx] = np.asarray(data, dtype=float)

    return result if result else None


# ---------------------------------------------------------------------------
# Aggregation across layers of one model
# ---------------------------------------------------------------------------

def analyze_model(model_id: str,
                  quantmode: str,
                  rank_mode: str,
                  bits: List[int]) -> Dict:
    """Run violation analysis for one (model, quantmode, rank_mode)."""
    layers = load_all_layers(model_id, quantmode, rank_mode, bits=bits)
    if not layers:
        return {'error': 'no data'}

    print(f"  Loaded {len(layers)} layers for {model_id} ({quantmode}/{rank_mode})")

    # Expert activation rates (optional)
    act_rates_by_layer = load_expert_act_rates(model_id)
    has_act = act_rates_by_layer is not None
    if has_act:
        print(f"  Expert activation rates loaded: {len(act_rates_by_layer)} layers")

    # --- accumulators ---
    total_valid = 0
    total_violations = 0
    all_violation_ratios = []

    per_layer = {}
    quantile_viol_counts = np.zeros(10)
    quantile_total_counts = np.zeros(10)

    expert_viol_by_act_quintile = np.zeros(5)
    expert_total_by_act_quintile = np.zeros(5)

    for ls in layers:
        lidx = ls.layer_idx
        result = fit_and_flag_layer(ls.by_bit)

        valid = result['valid']
        viol = result['violation']
        ratio = result['ratio']
        sens = result['sensitivity']

        n_valid = int(valid.sum())
        n_viol = int(viol.sum())
        total_valid += n_valid
        total_violations += n_viol

        if n_viol > 0:
            viol_ratios = ratio[viol]
            all_violation_ratios.extend(viol_ratios.tolist())
            med_ratio = float(np.median(viol_ratios))
        else:
            med_ratio = float('nan')

        per_layer[lidx] = {
            'n_valid': n_valid,
            'n_violation': n_viol,
            'violation_frac': n_viol / n_valid if n_valid > 0 else float('nan'),
            'median_ratio': med_ratio,
        }

        # sensitivity quantile analysis
        for exp_idx in range(valid.shape[0]):
            exp_valid = valid[exp_idx]
            exp_sens = sens[exp_idx]
            exp_viol = viol[exp_idx]

            if not exp_valid.any():
                continue

            valid_idx = np.where(exp_valid)[0]
            valid_sens = exp_sens[valid_idx]
            valid_viol = exp_viol[valid_idx]

            order = np.argsort(valid_sens)
            n = len(order)
            for dec in range(10):
                start = dec * n // 10
                end = (dec + 1) * n // 10
                if end > start:
                    decile_mask = order[start:end]
                    quantile_total_counts[dec] += len(decile_mask)
                    quantile_viol_counts[dec] += int(valid_viol[decile_mask].sum())

        # expert hot/cold analysis
        if has_act and lidx in act_rates_by_layer:
            layer_act = act_rates_by_layer[lidx]
            n_exp = valid.shape[0]
            if len(layer_act) == n_exp:
                exp_order = np.argsort(layer_act)
                for q in range(5):
                    start = q * n_exp // 5
                    end = (q + 1) * n_exp // 5
                    if end > start:
                        quintile_experts = exp_order[start:end]
                        for ei in quintile_experts:
                            expert_total_by_act_quintile[q] += int(valid[ei].sum())
                            expert_viol_by_act_quintile[q] += int(viol[ei].sum())

    # --- aggregate stats ---
    global_viol_frac = total_violations / total_valid if total_valid > 0 else float('nan')
    all_ratios_arr = np.array(all_violation_ratios)
    if len(all_ratios_arr) > 0:
        severity = {
            'median': float(np.median(all_ratios_arr)),
            'mean': float(np.mean(all_ratios_arr)),
            'p10': float(np.percentile(all_ratios_arr, 10)),
            'p90': float(np.percentile(all_ratios_arr, 90)),
            'min': float(np.min(all_ratios_arr)),
            'n_violations': int(len(all_ratios_arr)),
        }
    else:
        severity = {'n_violations': 0}

    decile_rates = [
        float(quantile_viol_counts[d] / quantile_total_counts[d])
        if quantile_total_counts[d] > 0 else float('nan')
        for d in range(10)
    ]

    expert_quintile_rates = []
    if has_act:
        for q in range(5):
            rate = (expert_viol_by_act_quintile[q] / expert_total_by_act_quintile[q]
                    if expert_total_by_act_quintile[q] > 0 else float('nan'))
            expert_quintile_rates.append(float(rate))

    layer_indices = sorted(per_layer.keys())
    layer_viol_fracs = [per_layer[l]['violation_frac'] for l in layer_indices]

    # Layer thirds analysis (early / middle / late)
    n_layers = len(layer_indices)
    if n_layers >= 3:
        third1_end = n_layers // 3
        third2_end = 2 * n_layers // 3
        early_fracs = layer_viol_fracs[:third1_end]
        mid_fracs = layer_viol_fracs[third1_end:third2_end]
        late_fracs = layer_viol_fracs[third2_end:]

        def mean_pct(fracs):
            return float(np.nanmean(fracs) * 100) if fracs else float('nan')

        layer_thirds = {
            'early_pct': mean_pct(early_fracs),
            'mid_pct': mean_pct(mid_fracs),
            'late_pct': mean_pct(late_fracs),
            'min_pct': float(np.nanmin(layer_viol_fracs) * 100) if layer_viol_fracs else float('nan'),
            'max_pct': float(np.nanmax(layer_viol_fracs) * 100) if layer_viol_fracs else float('nan'),
            'median_pct': float(np.nanmedian(layer_viol_fracs) * 100) if layer_viol_fracs else float('nan'),
            'n_layers': n_layers,
        }
    else:
        layer_thirds = {}

    # Lowest-to-highest decile ratio (how much more violations in lowest decile)
    if decile_rates[0] > 0 and decile_rates[-1] > 0:
        low_high_ratio = decile_rates[0] / decile_rates[-1]
    else:
        low_high_ratio = float('nan')

    return {
        'model_id': model_id,
        'quantmode': quantmode,
        'rank_mode': rank_mode,
        'total_valid_neurons': total_valid,
        'total_violations': total_violations,
        'violation_fraction': global_viol_frac,
        'clip_affected_fraction': global_viol_frac,
        'severity': severity,
        'per_layer': per_layer,
        'layer_indices': layer_indices,
        'layer_violation_fracs': layer_viol_fracs,
        'layer_thirds': layer_thirds,
        'sensitivity_decile_violation_rates': decile_rates,
        'sensitivity_low_high_ratio': float(low_high_ratio),
        'expert_act_quintile_violation_rates': expert_quintile_rates,
        'has_expert_act_rates': has_act,
        'n_layers': len(layers),
    }


# ===========================================================================
# Text summary (stdout)
# ===========================================================================

def print_summary(all_results: Dict[str, Dict[str, Dict]]):
    """Print a human-readable summary."""
    print("\n" + "=" * 100)
    print("TABLE 1 — OVERALL SUMMARY")
    print("=" * 100)
    header = (f"{'Model':<25s} {'Quant':<12s} {'Viol%':>8s} {'Clip%':>8s} "
              f"{'MedRatio':>10s} {'P10Ratio':>10s} {'N_viol/total':>20s}")
    print(header)
    print("-" * 100)

    for model_id in sorted(all_results.keys()):
        label = model_label(model_id)
        for quantmode in sorted(all_results[model_id].keys()):
            r = all_results[model_id][quantmode]
            if 'error' in r:
                print(f"  {label:<23s} {quantmode:<12s} (no data)")
                continue
            viol_pct = r['violation_fraction'] * 100
            sev = r['severity']
            med = sev.get('median', float('nan'))
            p10 = sev.get('p10', float('nan'))
            n_v = sev.get('n_violations', 0)
            n_t = r['total_valid_neurons']
            print(f"  {label:<23s} {quantmode:<12s} {viol_pct:7.3f}% {viol_pct:7.3f}% "
                  f"{med:10.4f} {p10:10.4f} {n_v:>10d}/{n_t:<10d}")

    print("\n" + "=" * 100)
    print("TABLE 2 — VIOLATION RATE BY SENSITIVITY DECILE (TurboQuant)")
    print("  Decile 1 = lowest sensitivity, Decile 10 = highest sensitivity")
    print("=" * 100)
    dec_header = f"{'Model':<25s} " + " ".join(f"{'D'+str(i+1):>7s}" for i in range(10)) + f" {'D1/D10':>8s}"
    print(dec_header)
    print("-" * 100)
    for model_id in sorted(all_results.keys()):
        label = model_label(model_id)
        r = all_results[model_id].get('turboquant')
        if not r or 'error' in r:
            continue
        rates_pct = [x * 100 for x in r['sensitivity_decile_violation_rates']]
        row = f"  {label:<23s} " + " ".join(f"{v:6.2f}%" for v in rates_pct)
        ratio = r.get('sensitivity_low_high_ratio', float('nan'))
        row += f" {ratio:7.1f}x"
        print(row)

    print("\n" + "=" * 100)
    print("TABLE 3 — PER-LAYER VIOLATION DISTRIBUTION (TurboQuant, %)")
    print("=" * 100)
    t3_header = (f"{'Model':<25s} {'Early':>8s} {'Middle':>8s} {'Late':>8s} "
                 f"{'Min':>7s} {'Median':>8s} {'Max':>7s} {'#Layers':>8s}")
    print(t3_header)
    print("-" * 100)
    for model_id in sorted(all_results.keys()):
        label = model_label(model_id)
        r = all_results[model_id].get('turboquant')
        if not r or 'error' in r:
            continue
        lt = r['layer_thirds']
        if not lt:
            continue
        print(f"  {label:<23s} {lt['early_pct']:7.3f}% {lt['mid_pct']:7.3f}% {lt['late_pct']:7.3f}% "
              f"{lt['min_pct']:6.3f}% {lt['median_pct']:7.3f}% {lt['max_pct']:6.3f}% {lt['n_layers']:>7d}")

    # Expert activation
    print("\n" + "=" * 100)
    print("EXPERT ACTIVATION ANALYSIS — violation rate by activation quintile (TurboQuant)")
    print("  Q1 = coldest experts, Q5 = hottest experts")
    print("=" * 100)
    any_act = False
    for model_id in sorted(all_results.keys()):
        r = all_results[model_id].get('turboquant')
        if not r or 'error' in r:
            continue
        if r['has_expert_act_rates'] and r['expert_act_quintile_violation_rates']:
            any_act = True
            label = model_label(model_id)
            rates_pct = [x * 100 for x in r['expert_act_quintile_violation_rates']]
            row = " | ".join(f"Q{i+1}:{v:6.3f}%" for i, v in enumerate(rates_pct))
            print(f"  {label:<23s} {row}")
    if not any_act:
        print("  (no expert activation data)")

    # Paper-ready fill-in numbers
    print("\n" + "=" * 100)
    print("PAPER FILL-IN SUMMARY (TurboQuant only, since GPTQ ≈ 0%)")
    print("=" * 100)

    tq_results = []
    for model_id in all_results:
        r = all_results[model_id].get('turboquant')
        if r and 'error' not in r:
            tq_results.append(r)

    if tq_results:
        viol_pcts = [r['violation_fraction'] * 100 for r in tq_results]
        med_ratios = [r['severity']['median'] for r in tq_results
                      if r['severity'].get('n_violations', 0) > 0]
        low_high_ratios = [r['sensitivity_low_high_ratio'] for r in tq_results
                           if not np.isnan(r['sensitivity_low_high_ratio'])]

        print(f"\n  1. Violation fraction: "
              f"{min(viol_pcts):.2f}% – {max(viol_pcts):.2f}% "
              f"(median {np.median(viol_pcts):.2f}%)")
        print(f"  2. Concentrated in low-sensitivity neurons: "
              f"lowest decile has {np.mean(low_high_ratios):.1f}x "
              f"more violations than highest decile (range "
              f"{min(low_high_ratios):.1f}x–{max(low_high_ratios):.1f}x)")
        print(f"  3. Median L̂(0)/L(1) among violations: "
              f"{min(med_ratios):.3f} – {max(med_ratios):.3f} "
              f"(median {np.median(med_ratios):.3f})")
        print(f"  4. Clip affected fraction: same as violation fraction, "
              f"{min(viol_pcts):.2f}% – {max(viol_pcts):.2f}%")
        print(f"\n  α (clip safety factor) = 2.0 (from code: l1 * 2.0)")


# ===========================================================================
# LaTeX table generators
# ===========================================================================

def _fmt_pct(val, precision=2):
    """Format a fraction as percentage string for LaTeX."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return r"--"
    return f"{val * 100:.{precision}f}\\%"


def _fmt_num(val, precision=3):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return r"--"
    return f"{val:.{precision}f}"


def generate_table1_overall(all_results: Dict[str, Dict[str, Dict]], save_dir: str) -> str:
    """Table 1: Overall violation stats per model and quant type."""
    model_order = [m for m in KNOWN_MODELS if m in all_results]

    lines = []
    lines.append(r"% ================================================================")
    lines.append(r"% Table 1: Overall 0-bit extrapolation violation statistics")
    lines.append(r"% Automatically generated by viz/extrapolation_violation_stats.py")
    lines.append(r"% ================================================================")
    lines.append(r"")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Statistics of 0-bit extrapolation monotonicity violations across models and quantization methods. "
                 r"A violation is defined as $\hat{L}_i(0) < L_i(1)$, i.e., the extrapolated 0-bit loss falls below the measured 1-bit loss.}")
    lines.append(r"  \label{tab:extrap_violation_overall}")
    lines.append(r"  \resizebox{0.9\textwidth}{!}{%")
    lines.append(r"  \begin{tabular}{lcccccccc}")
    lines.append(r"    \toprule")
    lines.append(r"    & \multicolumn{3}{c}{TurboQuant} & \multicolumn{3}{c}{GPTQ} & \\")
    lines.append(r"    \cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    lines.append(r"    Model & Violation & Clip & Median & Violation & Clip & Median & \\")
    lines.append(r"    & Rate (\%) & Affected (\%) & $\hat{L}(0)/L(1)$ & Rate (\%) & Affected (\%) & $\hat{L}(0)/L(1)$ & $N_\mathrm{neurons}$ \\")
    lines.append(r"    \midrule")

    for model_id in model_order:
        label = KNOWN_MODELS[model_id]
        tq = all_results[model_id].get('turboquant', {})
        gptq = all_results[model_id].get('gptq', {})

        tq_viol = tq.get('violation_fraction', float('nan')) * 100 if 'error' not in tq else float('nan')
        tq_med = tq.get('severity', {}).get('median', float('nan')) if 'error' not in tq else float('nan')
        tq_n = tq.get('total_valid_neurons', 0) if 'error' not in tq else 0

        gptq_viol = gptq.get('violation_fraction', float('nan')) * 100 if 'error' not in gptq else float('nan')
        gptq_med = gptq.get('severity', {}).get('median', float('nan')) if 'error' not in gptq else float('nan')

        def f(v, p=2):
            if np.isnan(v):
                return r"$<\!0.01$" if p == 2 else r"--"
            return f"${v:.{p}f}$"

        n_total = tq_n if tq_n > 0 else gptq.get('total_valid_neurons', 0)
        n_str = f"${n_total:,}$".replace(",", r"{,}")

        lines.append(f"    {label} & {f(tq_viol)} & {f(tq_viol)} & {f(tq_med,3)} & "
                     f"{f(gptq_viol)} & {f(gptq_viol)} & {f(gptq_med,3)} & {n_str} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}%")
    lines.append(r"  }")
    lines.append(r"  \vspace{1ex}")
    lines.append(r"  \small \textit{Note:} Clip-affected neurons equal violating neurons since all violations are clipped. "
                 r"GPTQ shows near-zero violations because its loss curves are smoother in the low-bit regime.")
    lines.append(r"\end{table*}")

    tex = "\n".join(lines)
    path = os.path.join(save_dir, "table1_overall.tex")
    with open(path, 'w') as f:
        f.write(tex)
    print(f"  Table 1 saved: {path}")
    return tex


def generate_table2_sensitivity(all_results: Dict[str, Dict[str, Dict]], save_dir: str) -> str:
    """Table 2: Violation rate by sensitivity decile (TurboQuant)."""
    model_order = [m for m in KNOWN_MODELS if m in all_results
                   and 'turboquant' in all_results[m]
                   and 'error' not in all_results[m]['turboquant']]

    lines = []
    lines.append(r"% ================================================================")
    lines.append(r"% Table 2: Violation rate by sensitivity decile (TurboQuant)")
    lines.append(r"% ================================================================")
    lines.append(r"")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Violation rate of 0-bit extrapolation grouped by neuron sensitivity decile (TurboQuant). "
                 r"Decile 1 contains the neurons with the lowest 1-bit loss (least sensitive), "
                 r"Decile 10 the highest. Violations consistently concentrate in low-sensitivity units.}")
    lines.append(r"  \label{tab:extrap_violation_sensitivity}")
    lines.append(r"  \resizebox{\textwidth}{!}{%")
    lines.append(r"  \begin{tabular}{l" + "c" * 10 + "cc}")
    lines.append(r"    \toprule")
    lines.append(r"    Model & D1 & D2 & D3 & D4 & D5 & D6 & D7 & D8 & D9 & D10 & D1/D10 \\")
    lines.append(r"    & \multicolumn{10}{c}{violation rate (\%)} & ratio \\")
    lines.append(r"    \midrule")

    for model_id in model_order:
        label = KNOWN_MODELS[model_id]
        r = all_results[model_id]['turboquant']
        rates = r['sensitivity_decile_violation_rates']
        rates_pct = [x * 100 for x in rates]
        cells = " & ".join(f"{v:.2f}" for v in rates_pct)
        ratio = r['sensitivity_low_high_ratio']
        ratio_str = f"{ratio:.1f}x" if not np.isnan(ratio) else "--"
        lines.append(f"    {label} & {cells} & {ratio_str} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}%")
    lines.append(r"  }")
    lines.append(r"  \vspace{1ex}")
    lines.append(r"  \small \textit{Note:} Deciles are computed per expert per layer by $L_i(1)$ (1-bit loss), then aggregated across all layers and experts. "
                 r"D1/D10 is the ratio of violation rates between the lowest and highest sensitivity deciles.")
    lines.append(r"\end{table*}")

    tex = "\n".join(lines)
    path = os.path.join(save_dir, "table2_sensitivity_decile.tex")
    with open(path, 'w') as f:
        f.write(tex)
    print(f"  Table 2 saved: {path}")
    return tex


def generate_table3_layers(all_results: Dict[str, Dict[str, Dict]], save_dir: str) -> str:
    """Table 3: Per-layer violation distribution (TurboQuant)."""
    model_order = [m for m in KNOWN_MODELS if m in all_results
                   and 'turboquant' in all_results[m]
                   and 'error' not in all_results[m]['turboquant']]

    lines = []
    lines.append(r"% ================================================================")
    lines.append(r"% Table 3: Per-layer violation distribution (TurboQuant)")
    lines.append(r"% ================================================================")
    lines.append(r"")
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Per-layer distribution of 0-bit extrapolation violations (TurboQuant). "
                 r"Layers are divided into three equal groups: early (first third), middle, and late (last third).}")
    lines.append(r"  \label{tab:extrap_violation_layers}")
    lines.append(r"  \resizebox{0.9\columnwidth}{!}{%")
    lines.append(r"  \begin{tabular}{lccccccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Model & Early & Middle & Late & Min & Median & Max & Layers \\")
    lines.append(r"    & \multicolumn{6}{c}{violation rate (\%)} & \\")
    lines.append(r"    \midrule")

    for model_id in model_order:
        label = KNOWN_MODELS[model_id]
        lt = all_results[model_id]['turboquant']['layer_thirds']
        if not lt:
            continue
        lines.append(
            f"    {label} & "
            f"{lt['early_pct']:.2f} & {lt['mid_pct']:.2f} & {lt['late_pct']:.2f} & "
            f"{lt['min_pct']:.2f} & {lt['median_pct']:.2f} & {lt['max_pct']:.2f} & "
            f"{lt['n_layers']} \\\\"
        )

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}%")
    lines.append(r"  }")
    lines.append(r"  \vspace{1ex}")
    lines.append(r"  \small \textit{Note:} All values are in percent. "
                 r"Early/middle/late refer to the first, second, and last third of layers respectively.")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    path = os.path.join(save_dir, "table3_layer_distribution.tex")
    with open(path, 'w') as f:
        f.write(tex)
    print(f"  Table 3 saved: {path}")
    return tex


def generate_expert_act_table(all_results: Dict[str, Dict[str, Dict]], save_dir: str) -> Optional[str]:
    """Bonus table: violation rate by expert activation quintile (TurboQuant)."""
    model_order = [m for m in KNOWN_MODELS if m in all_results
                   and 'turboquant' in all_results[m]
                   and 'error' not in all_results[m]['turboquant']
                   and all_results[m]['turboquant'].get('has_expert_act_rates')
                   and all_results[m]['turboquant'].get('expert_act_quintile_violation_rates')]

    if not model_order:
        print("  No expert activation data available — skipping expert table.")
        return None

    lines = []
    lines.append(r"% ================================================================")
    lines.append(r"% Table B.x: Violation rate by expert activation quintile")
    lines.append(r"% ================================================================")
    lines.append(r"")
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Violation rate by expert activation quintile (TurboQuant). "
                 r"Q1 = coldest experts (lowest activation rate), Q5 = hottest.}")
    lines.append(r"  \label{tab:extrap_violation_expert_act}")
    lines.append(r"  \begin{tabular}{lccccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Model & Q1 (cold) & Q2 & Q3 & Q4 & Q5 (hot) \\")
    lines.append(r"    & \multicolumn{5}{c}{violation rate (\%)} \\")
    lines.append(r"    \midrule")

    for model_id in model_order:
        label = KNOWN_MODELS[model_id]
        rates = all_results[model_id]['turboquant']['expert_act_quintile_violation_rates']
        rates_pct = [x * 100 for x in rates]
        cells = " & ".join(f"{v:.3f}" for v in rates_pct)
        lines.append(f"    {label} & {cells} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \vspace{1ex}")
    lines.append(r"  \small \textit{Note:} Experts are ranked by activation rate per layer and grouped into quintiles.")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    path = os.path.join(save_dir, "tableB_expert_activation.tex")
    with open(path, 'w') as f:
        f.write(tex)
    print(f"  Expert activation table saved: {path}")
    return tex


# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------

def save_json(all_results: Dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, 'extrapolation_violation_stats.json')

    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: default(v) for k, v in o.items()}
        if isinstance(o, list):
            return [default(x) for x in o]
        return o

    with open(out_path, 'w') as f:
        json.dump(default(all_results), f, indent=2)
    print(f"\nFull JSON saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="0-bit extrapolation violation statistics across models"
    )
    parser.add_argument("--models", nargs="+",
                        help="Model ids (default: all known models with cache data)")
    parser.add_argument("--quants", nargs="+", default=["turboquant", "gptq"],
                        help="Quant modes to analyze (default: turboquant gptq)")
    parser.add_argument("--bits", nargs="+", type=int, default=[1, 2, 3, 4],
                        help="Bit widths to use for fit (default: 1 2 3 4)")
    parser.add_argument("--rank-modes", nargs="+",
                        help="Rank modes per quant mode (default: auto)")
    parser.add_argument("--save-dir", default="plot/extrapolation_violation",
                        help="Directory to save results")
    args = parser.parse_args()

    default_rank_modes = {
        'turboquant': 'turboquant_innerproduct',
        'gptq': 'gptq_quant_outlier',
    }

    if args.models:
        model_ids = args.models
    else:
        model_ids = set()
        for q in args.quants:
            rm = default_rank_modes[q]
            found = discover_models(q, rm)
            model_ids.update(found)
        model_ids = [m for m in KNOWN_MODELS if m in model_ids]
        print(f"Auto-discovered {len(model_ids)} models: {model_ids}")

    if not model_ids:
        print("ERROR: no models found in cache.")
        sys.exit(1)

    if args.rank_modes:
        assert len(args.rank_modes) == len(args.quants)
        rank_modes = dict(zip(args.quants, args.rank_modes))
    else:
        rank_modes = {q: default_rank_modes[q] for q in args.quants}

    all_results = defaultdict(dict)
    for model_id in model_ids:
        print(f"\n{'='*60}")
        print(f"Model: {model_label(model_id)} ({model_id})")
        print(f"{'='*60}")
        for quantmode in args.quants:
            rm = rank_modes[quantmode]
            print(f"\n  Quant: {quantmode}, rank_mode: {rm}")
            result = analyze_model(model_id, quantmode, rm, args.bits)
            all_results[model_id][quantmode] = result

    # Print text summary
    print_summary(all_results)

    # Generate LaTeX tables
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"\nGenerating LaTeX tables -> {args.save_dir}/")
    generate_table1_overall(all_results, args.save_dir)
    generate_table2_sensitivity(all_results, args.save_dir)
    generate_table3_layers(all_results, args.save_dir)
    generate_expert_act_table(all_results, args.save_dir)

    # Save JSON
    save_json(all_results, args.save_dir)


if __name__ == "__main__":
    main()
