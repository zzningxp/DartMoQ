# DartMoQ visualization toolkit

```
viz/
├── __init__.py                # package entry
├── _cache_io.py               # shared helpers: cache loading, model registry, plot style
├── headroom.py                # 4 motivation panels  — see table below
├── metric_geometry.py         # G.1 – G.4 — Challenge 1 (sensitivity geometry)
├── seed_stability.py          # S.1 – S.5 — TurboQuant seed variance diagnosis
├── dump_activation_rates.py   # one-off dumper: per-layer expert activation rates → intermediate_result/
├── micro_expert_rank_boxplot.py # shared implementation for DP score diagnostics
├── dp_score_tests/            # Exp.4/5 DP assigned-loss diagnostic entrypoints
└── legacy.py                  # re-exports of the old per-layer plotting functions
```

Each viz module exposes a `main()` callable as `python -m viz.<module>`.

`--model` accepts either a **short cache id** (`olmoe-7b-1b`,
`deepseek-v1-moe-16b`, `deepseek-v2-lite`, `moonlight`, `qwen3-30b-a3b`) **or
the full model path** you would pass to `run_dartmoq.py`
(e.g. `--model /home/.../OLMoE-1B-7B-0924/`). The path → id mapping lives in
`viz/_cache_io.py::resolve_model_id` and mirrors `eval_dartmoq.load_model`.

All inputs come from the `intermediate_result/quant_outlier_{quantmode}/{rank_mode}/{model_id}/`
cache directories. **`act_vs_sens` additionally requires** per-layer activation
rates materialised under `intermediate_result/expert_activate/{model_id}/` by
`viz/dump_activation_rates.py` (one-time cost per model).

---

## Module map → Paper section

| Module | Panels | Paper section |
|---|---|---|
| `headroom`        | `amgm`, `top10ratio`, `act_vs_sens`, `layer_expert_neuron_compare` | **Motivation** (§1 intro / §3.1 preliminary) |
| `metric_geometry` | G.1 – G.4 | **Challenge 1** (§4.1: finding the right sensitivity metric)       |
| `seed_stability`  | S.1 – S.5 | **Analysis**: why DartMoQ slicing reduces TurboQuant QR seed variance |
| `dp_score_tests`  | Exp.4 – Exp.5 | **Diagnostics**: inspect final DP assigned loss by expert and by global DP order |
| `dp_utils` internal | —      | **Challenge 2** (§4.2: quadratic log-fit & 0-bit extrapolation)    |
| `legacy`          | —         | Appendix (supplementary per-layer debugging figures)               |

> Note: an originally planned per-model neuron-loss-CDF panel was dropped —
> the same dispersion information is captured more compactly by `amgm` and
> `top10ratio`. The expert-sensitivity heatmap (a staple in every MoE
> quantization paper) was likewise omitted to avoid redundancy.

---

## Figure-by-figure

### Motivation panels (`viz/headroom.py`)

| Panel | Type | What it shows | How to generate |
|---|---|---|---|
| **`amgm`** | per-layer bar, multi-model side-by-side | AM/GM of per-neuron loss = **closed-form headroom factor**. Under the log-quadratic loss model `log L_i(b) = p b² + q b + r_i`, the uniform/oracle loss ratio equals `mean(c_i) / geomean(c_i)` where `c_i = exp(r_i)`. By AM-GM, this ratio ≥ 1; the bar height is the upper bound on what any neuron-level mixed-precision allocation can reclaim vs uniform-bit. | `python -m viz.headroom --skip top10ratio act_vs_sens layer_expert_neuron_compare` |
| **`top10ratio`** | per-layer bar, multi-model side-by-side | Loss share captured by the **top-10% most sensitive neurons** per layer. Uniform reference is 0.10; higher bars ⇒ more concentrated ⇒ companion interpretation of the `amgm` headroom bound. | `python -m viz.headroom --skip amgm act_vs_sens layer_expert_neuron_compare` |
| **`act_vs_sens`** | scatter + Spearman histogram (per model) | Expert activation rate × quant sensitivity; per-layer Spearman ρ centred near 0 ⇒ activation alone (MoQE-style) and sensitivity alone (OWQ-style) are NOT interchangeable. | `python -m viz.headroom --skip amgm top10ratio layer_expert_neuron_compare` |
| **`layer_expert_neuron_compare`** | multi-model line panel | Loss-model oracle total loss vs target bpw (1.0 – 4.0) for three granularities {layer, expert, neuron-bucket}; one column per model on a single figure. `layer` emits points only at integer bpw (half-bpw has no integer realisation); the dropped `uniform` line would coincide with `layer` at integer bpw and be undefined elsewhere. | `python -m viz.headroom --skip amgm top10ratio act_vs_sens` |

**Logical chain**
- **`amgm` / `top10ratio`** prove *within an expert, neuron-level loss is
  concentrated → AM/GM ≫ 1 → headroom exists.* AM/GM is a closed-form upper
  bound — no optimization is solved to obtain it.
- **`act_vs_sens`** falsifies the natural objection "activation rate already
  explains that dispersion": activation and sensitivity are nearly independent
  (mean Spearman ρ ≈ 0), so cheap signals cannot substitute for the explicit
  sensitivity model.
- **`layer_expert_neuron_compare`** turns the upper bound into a measured loss
  curve, sweeping bpw and granularity across all available models. The gap
  between adjacent curves is the headroom captured by going one level finer.

> ⚠️ **`act_vs_sens`** requires per-layer activation rates. Materialise them once with:
> ```bash
> python -m viz.dump_activation_rates --model deepseek-v1-moe-16b
> ```
> Output goes to `intermediate_result/expert_activate/<short_id>/<short_id>_L<layer>.pt`. If absent,
> `act_vs_sens` prints a hint and is silently skipped.

> ⚠️ **`layer_expert_neuron_compare`** runs the loss-model oracle (DP on
> cached sensitivities). It is *not* a brute-force oracle. See the abstract
> discussion of which oracle variant we adopt and why; the brute-force /
> LP-relaxation variants are documented in the paper appendix and live in a
> separate experiment script.

### G.* — Sensitivity geometry differences  (`viz/metric_geometry.py`)

| Figure | Type | What it shows | How to generate |
|---|---|---|---|
| **G.1** | CDF, 3 metrics      | GPTQ-MSE vs TurboQuant-MSE vs TurboQuant-IP; Gini coefficients per metric             | `python -m viz.metric_geometry --skip g2 g3 g4` |
| **G.2** | sorted-energy + hist | Random rotation flattens per-channel energy (synthetic: Gaussian weights + QR)      | `python -m viz.metric_geometry --skip g1 g3 g4` |
| **G.3** | 3×3 ρ matrix         | Spearman ρ between metrics, averaged over layers                                      | `python -m viz.metric_geometry --skip g1 g2 g4` |
| **G.4** | boxplot              | Metric ranking vs measured zero-out loss (requires ground-truth)                      | `python -m viz.metric_geometry --skip g1 g2 g3` |

> ⚠️ **G.4** requires ground-truth `logs/zero_out_validation/{model_id}/{model_id}_L{layer}_e{expert}.npz`
> (each `.npz` holds `neuron_idx` and `true_loss`). Produce it with a separate
> validation script that zeros individual neurons and measures the calibration
> loss. If absent, G.4 is silently skipped (you still get a console hint).

> ⚠️ **G.2** uses synthetic data on purpose — the energy-flattening property is
> a pure consequence of orthogonal rotation and does not depend on any trained
> model. If you want the figure on real weights instead, point the function at
> a stashed `(weight_matrix, rotation_matrix)` pair (small file under
> `logs/rotation_energy/`) — the loader hook is already present.

> ⚠️ **TurboQuant element-MSE** is *not* a separate cache — it is synthesized
> from the inner-product cache by stripping the activation-norm weighting
> (`viz/metric_geometry.py::_synthesize_tq_mse_from_cache`). This keeps the
> figure honest: any remaining ranking signal comes from the quantizer
> geometry itself, not from activations.

### S.* — TurboQuant seed stability after DartMoQ slicing (`viz/seed_stability.py`)

| Figure | Type | What it shows | How to generate |
|---|---|---|---|
| **S.1 `seed_sweep`** | line plot | Across the same QR seed set, full-expert TurboQuant has larger sensitivity-weighted error variation than DartMoQ neuron slices. | `python -m viz.seed_stability --skip good_bad alignment homogeneity aggregate` |
| **S.2 `good_bad`** | sorted per-neuron error | Bad full-matrix seeds place more error on the high-sensitivity neurons than good seeds; slice quantization dampens this effect. | `python -m viz.seed_stability --skip seed_sweep alignment homogeneity aggregate` |
| **S.3 `alignment`** | log-log scatter | Error–sensitivity rank alignment explains why a seed with similar raw MSE can be worse in sensitivity-weighted loss. | `python -m viz.seed_stability --skip seed_sweep good_bad homogeneity aggregate` |
| **S.4 `homogeneity`** | dispersion bars | DP slices have lower within-slice sensitivity dispersion than the full expert, supporting the “less heterogeneous submatrix” explanation. | `python -m viz.seed_stability --skip seed_sweep good_bad alignment aggregate` |
| **S.5 `aggregate`** | paired bars | Across high-loss DeepSeekMoE targets, slice quantization reduces weighted-error CV, max/min ratio, and bad–good seed gap. | `python -m viz.seed_stability --skip seed_sweep good_bad alignment homogeneity` |

This module is a diagnostic, not another end-to-end benchmark. It loads one
model only to access selected MoE expert weights, combines them with cached
per-neuron sensitivity, and compares full up/gate/down matrices against
DartMoQ-style neuron slices under the same TurboQuant QR seed sweep.

Quick target-only check (no model load):
```bash
python -m viz.seed_stability --model deepseek-v1-moe-16b --dry-run-targets
```

Small smoke run:
```bash
python -m viz.seed_stability --model deepseek-v1-moe-16b \
  --layers 17 --experts 2 --seeds 0 1 --bit 2 --wbits 2 \
  --slice-expert-num 8 --skip aggregate
```

### Exp.4-5 — DP assigned-loss diagnostics (`viz/dp_score_tests/`)

These scripts are PR-facing diagnostics for the bit-allocation pipeline. They
focus on the final DP assigned loss: Exp.4 groups the assigned losses by expert,
and Exp.5 plots the same assigned losses along the global DP ordering.

| Exp. | Entrypoint | Y-axis / formula | X-axis | Purpose |
|---|---|---|---|---|
| 4 | `python -m viz.dp_score_tests.exp_4.plot_exp4_assigned_loss_vs_uniform` | `sum(q_rates[assigned_bit]) * activation` | expert | Plot final assigned loss by expert; with `--include-uniform-baseline`, also outputs the no-sort, no-DP, fixed-bit boxplot at the same integer bpw. |
| 5 | `python -m viz.dp_score_tests.exp_5.plot_exp5_assigned_loss_by_order` | `sum(q_rates[assigned_bit]) * activation` | DP sorted sub-expert index | Check whether final assigned losses look reasonable along the DP sorted queue. |

Typical Qwen3 TurboQuant command for Exp.4:

```bash
python -m viz.dp_score_tests.exp_4.plot_exp4_assigned_loss_vs_uniform \
  --model qwen3-30b-a3b \
  --quantmode turboquant \
  --rank-mode turboquant_innerproduct \
  --layers 7 \
  --include-random-layer \
  --random-seed 123 \
  --bits 0 1 2 3 4 \
  --slices-per-expert 8 \
  --target-bpw 2.0 \
  --disable-0bit-compensation \
  --include-uniform-baseline \
  --comparison-out-dir figs/assigned_loss_bpw2_boxplot_compare
```

At `--target-bpw 2.0`, the baseline in Exp.4 uses fixed `b2` for every unsorted
slice: `sum(q_rates[b2][unsorted_slice]) * activation`. Non-integer target bpw
does not define a single fixed-bit no-mixed-precision baseline.

---

## Cache directory layout

```
intermediate_result/
  quant_outlier_{quantmode}/
    {rank_mode}/
      {model_id}/
        {model_id}_L{layer}_b{bit}.pt # list[Tensor], length = n_experts
  expert_activate/
    {model_id}/
      {model_id}_L{layer}.pt          # shape = (n_experts,)
```

`viz._cache_io.load_layer()` abstracts the sensitivity cache. The activation
rate cache is read directly by `viz.headroom._load_activation_rates`.

---

## Adding a new figure

1. Decide whether it belongs in `headroom.py` (untapped-gain story) or
   `metric_geometry.py` (sensitivity geometry story). If it's neither, start
   a new file named after the *story*, not the figure number.
2. Load data exclusively through `viz._cache_io` helpers (and
   `_load_activation_rates` when needed).
3. Add an `apply_paper_style()` call at the top of the script (or rely on the
   one in `main()`).
4. Save to `plot/<module-name>/<figure-id>_<descriptor>.png`.
5. Add a row to the figure table above and a one-line entry in the module's
   top-of-file docstring.

---

## Legacy function mapping

The old plotting functions are still importable for backwards compatibility:

| Old location → Old name | Now reachable via |
|---|---|
| `dp_utils.plot_neuron_rates_across_bits`            | `viz.legacy.plot_neuron_rates_across_bits` |
| `dp_utils.plot_bit_overlap`                         | `viz.legacy.plot_bit_overlap`              |
| `dp_utils.plot_block_losses_overlap`                | `viz.legacy.plot_block_losses_overlap`     |
| `visual_utils.plot_diff_wbits_correlation`          | `viz.legacy.plot_diff_wbits_correlation`   |
| `visual_utils.plot_spearman_rank_correlation`       | `viz.legacy.plot_spearman_rank_correlation`|

`viz.legacy` is a thin re-export module — the original bodies remain in
`dp_utils.py` and `visual_utils.py`.
