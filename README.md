
# DartMoQP: A MoE-Native Unified Framework for Mixed-Precision Quantization &amp; Structured Pruning

DartMoQP is a Mixture-of-Experts-native unified quantization and structured pruning framework. It brings quantization and pruning into a single mathematical framework for joint sensitivity modeling and global optimal search, with neuron-level expert reordering.

## Key Contributions

### Challenges Addressed

1. **Different quantization algorithms have fundamentally different error geometry characteristics**
2. **Quantization and pruning have inconsistent optimization objectives**

### Key Insights from Large-Scale Experiments

1. **Sensitivity Metric Design**:
   - For per-row quantization algorithms like GPTQ: Quantization error already incorporates second-order Hessian weighting from the input calibration set during iteration, so using element-wise MSE directly yields good sensitivity
   - For vector quantization algorithms like TurboQuant: Global random rotation causes energy homogenization, making element-wise MSE sensitivity poorly differentiated; an inner product loss based on the calibration input manifold is more suitable

2. **Unified Loss Space**:
   - For major quantization algorithms (GPTQ and TurboQuant), quantization loss follows a perfect quadratic distribution in the log domain
   - This allows reliable extrapolation of 0bit loss without any manual hyperparameters
   - Enables unified loss modeling of quantization and pruning for the first time

3. **Unified Dynamic Programming Search**:
   - A group-wise dynamic programming approach for optimal bit allocation
   - First, compute and cache quantization loss for each neuron at multiple bit widths (1-4 bits), then extrapolate 0bit loss (pruning) via log-quadratic fitting
   - Neurons within each expert are sorted by sensitivity, split into S groups, and all sub-experts are globally ranked by importance (sensitivity × expert activation rate)
   - Finally, monotonic DP search with non-increasing bit allocation constraint finds the optimal bit assignment at target bpw

4. **Stability to Random Seed**:
   - The random rotation matrix in TurboQuant causes different models to have varying sensitivity to different random seeds, leading to different quantization errors across models
   - This phenomenon is related to the weight characteristics of different models, resulting in different behaviors across models
   - Experiments show that our method can stabilize the impact of random seeds on quantization error

### Framework Design

DartMoQP adopts a quantization-method-agnostic global dynamic programming search pipeline that automatically matches the optimal sensitivity metric and bit allocation scheme for any quantization algorithm.

## Neuron-Level Expert Reordering

DartMoQP performs neuron-level expert reordering to optimize for mixed-precision quantization. The process is as follows:

[PLACEHOLDER FOR METHOD FIGURE - To be added later]

1. **Sensitivity Calculation**: For each neuron in each expert, compute its quantization sensitivity using the appropriate metric for the quantization algorithm (element-wise MSE for GPTQ, inner product loss for TurboQuant)
2. **Neuron Ranking**: Neurons within each expert are sorted in descending order of sensitivity (most sensitive first)
3. **Sub-expert Formation**: Sorted neurons are divided into S slices/sub-experts
4. **Global Merging in Hybrid Mode**: When using hybrid MoE, sub-experts can be dynamically merged based on their importance during inference

This reordering ensures that the most error-sensitive neurons receive more bits while less important neurons can be safely pruned (0bit) or quantized to lower precision.

## Hybrid MoE Wrapper Implementation

DartMoQP uses a wrapper-based approach that is compatible with all Transformers library-based models. The hybrid MoE structure features:

1. **Two-level Gating Mechanism**:
   - **First level**: Original MoE routing (same as the base model)
   - **Second level**: Mixed-precision selector that chooses sub-experts based on their assigned bit width

2. **Expert Merging**:
   - Sub-experts with the same bit width can be merged for efficient inference
   - Maintains compatibility with the original model architecture through wrapper composition

3. **Backward Compatibility**:
   - The wrapper preserves all original model interfaces
   - Works seamlessly with HuggingFace `generate()` and evaluation pipelines
   - Can be disabled with `--no-use-hybrid-moe` to use original experts


## Note on Implementation**: 

The current implementation is a simulated quantization framework. All quantized operations are dequantized back to fp16 for actual inference. While this does not provide real inference speedup, it enables accurate evaluation of quantization error and can guide the design of practical quantization algorithms.

## Loss Caching Mechanism

To avoid redundant computation during parameter sweeps, DartMoQP implements a loss caching mechanism:

1. **Cache Location**: Cached losses are stored in `quant_outlier_{gptq,turboquant}/{model_id}/`
2. **Cache Format**: Separate cache files for each bit width: `{model_id}_L{layer_idx}_b{bit}.pt`
3. **Contents**: Each cache file contains per-neuron quantization loss for all experts in that layer
4. **Reuse**: The cache is automatically reused for different rank modes, quant schemes, and seed values
5. **Groupsize**: All cache computations use a consistent groupsize of 128

This caching significantly speeds up hyperparameter searches and ablation studies.

## Results

DartMoQP achieves state-of-the-art performance across the full 0.5-4.0 bpw range on multiple mainstream MoE models:
- OLMoE-7B
- DeepSeekMoE-v1/v2 (16B-A3B)
- Moonlight (16B-A3B)
- Qwen3-30B-A3B

### Method Combinations

- **GPTQ-based methods**: Use GPTQ loss + dynamic programming (DP) + GPTQ quantization
- **Energy-based method**: Uses energy importance (from CAMERA) + DP + TurboQuant quantization. Note: Energy method does not support 0bit loss extrapolation, so it cannot be used for schemes below 1bit. At 1bit, it degrades to a non-optimized baseline.
- **Other TurboQuant-based methods**: Use TurboQuant loss + dynamic programming (DP) + TurboQuant quantization

Notably:
- **Extremely low bit regime (0.5-2 bpw)**: Order-of-magnitude performance improvement over baselines (though still not fully practical)
- **2bit scheme (industry standard)**: DartMoQP-TurboQuant consistently outperforms existing methods in downstream tasks

### ppl 
<img src="figs/result1-olmoe.png" width="500" alt="OLMoE-7B Results">
<img src="figs/result1-dsv1.png" width="500" alt="DeepSeekMoE-v1 Results">
<img src="figs/result1-dsv2.png" width="500" alt="DeepSeekMoE-v2 Results">
<img src="figs/result1-moonlight.png" width="500" alt="Moonlight Results">
<img src="figs/result1-qwen3.png" width="500" alt="Qwen3-30B-A3B Results">

The figures above show perplexity vs. bits-per-weight (bpw) comparisons between DartMoQP and representative quantization methods across five MoE models. DartMoQP-TurboQuant consistently achieves the lowest perplexity across all bit widths.

### eval-zero tasks

#### 1.0 bpw (+0.25)

| Model | Method | WikiText2 | C4 | Avg. | ARC-Challenge | ARC-Easy | PIQA | BoolQ | Winogrande | MNLI | Hellaswag | MMLU |
|-------|--------|-----------|----|-------|----------|---------------|------|-------|------------|-----------|------|-------|
| DSMoE-v1 | Energy | 278.704 | 573.5555 | 0.346737 | 0.244881 | 0.255892 | 0.519042 | 0.378899 | 0.505919 | 0.362506 | 0.27146 | 0.235294 |
| DSMoE-v1 | GPTQ | 13.2548 | 24.7183 | 0.487514 | 0.313993 | 0.602273 | 0.67519 | 0.61315 | 0.589582 | 0.362506 | 0.485262 | 0.258154 |
| DSMoE-v1 | IPE-TQ | **10.992** | **23.6676** | **0.508849** | 0.360068 | 0.65404 | 0.661045 | 0.674006 | 0.614838 | 0.393174 | 0.470922 | 0.2427 |
| DSv2-Lite | Energy | 37.7916 | 51.5081 | 0.384198 | 0.229522 | 0.315657 | 0.550054 | 0.571865 | 0.504341 | 0.335914 | 0.306513 | 0.259721 |
| DSv2-Lite | GPTQ | 47.3629 | 80.3963 | 0.375537 | 0.206485 | 0.294613 | 0.536997 | 0.574618 | 0.520126 | 0.340397 | 0.288787 | 0.242273 |
| DSv2-Lite | IPE-TQ | **9.6862** | **20.8551** | **0.487446** | 0.346416 | 0.635101 | 0.657236 | 0.534862 | 0.550908 | 0.369435 | 0.480084 | 0.325523 |
| Moonlight | Energy | 249.4769 | 333.5471 | 0.383909 | 0.21843 | 0.340488 | 0.551687 | 0.555963 | 0.501184 | 0.35354 | 0.296455 | 0.253525 |
| Moonlight | GPTQ | 69.8667 | 145.5026 | 0.382725 | 0.234642 | 0.324074 | 0.53482 | 0.584709 | 0.48382 | 0.344473 | 0.300737 | 0.254522 |
| Moonlight | IPE-TQ | **17.7703** | **44.9661** | **0.452666** | 0.28157 | 0.551768 | 0.597388 | 0.621407 | 0.516969 | 0.361284 | 0.405298 | 0.285643 |
| OLMoE | Energy | 16753.1133 | 8156.6748 | 0.373811 | 0.263652 | 0.291667 | 0.520675 | 0.565443 | 0.504341 | 0.319002 | 0.262498 | 0.26321 |
| OLMoE | GPTQ | 157.6327 | 278.5359 | 0.38802 | 0.227816 | 0.34133 | 0.534276 | 0.603364 | 0.513812 | 0.347224 | 0.292571 | 0.243769 |
| OLMoE | IPE-TQ | **26.214** | **50.9268** | **0.466753** | 0.337884 | 0.570707 | 0.617519 | 0.613761 | 0.553275 | 0.364952 | 0.40709 | 0.268836 |
| Qwen3 | Energy | 1886.7177 | 1422.7914 | 0.354788 | 0.230375 | 0.303451 | 0.527203 | 0.417125 | 0.513023 | 0.33622 | 0.268273 | 0.242629 |
| Qwen3 | GPTQ | 14.9775 | 26.5466 | 0.502423 | 0.302901 | 0.526094 | 0.638194 | 0.703058 | 0.636148 | 0.405196 | 0.522008 | 0.285786 |
| Qwen3 | IPE-TQ | **11.6826** | **20.9354** | **0.593427** | 0.436007 | 0.703283 | 0.666485 | 0.805199 | 0.66614 | 0.472542 | 0.550388 | 0.447372 |

#### 1.5 bpw (+0.25)

| Model | Method | WikiText2 | C4 | Avg. | ARC-Challenge | ARC-Easy | PIQA | BoolQ | Winogrande | MNLI | Hellaswag | MMLU |
|-------|--------|-----------|----|-------|----------|---------------|------|-------|------------|-----------|------|-------|
| DSMoE-v1 | Energy | 9.5556 | 15.5672 | 0.559036 | 0.401877 | 0.686869 | 0.695865 | 0.714679 | 0.681137 | 0.412532 | 0.612627 | 0.2667 |
| DSMoE-v1 | GPTQ | 9.1824 | **14.5733** | 0.556436 | 0.379693 | 0.671296 | 0.739935 | 0.629969 | 0.653512 | 0.418849 | 0.649871 | 0.308361 |
| DSMoE-v1 | IPE-TQ | **8.3029** | 14.5563 | **0.582868** | 0.43686 | 0.750842 | 0.735582 | 0.724465 | 0.670087 | 0.395721 | 0.628062 | 0.321322 |
| DSv2-Lite | Energy | 8.9815 | 14.2178 | 0.599845 | 0.438567 | 0.720539 | 0.714908 | 0.77156 | 0.640095 | 0.441671 | 0.638219 | 0.4332 |
| DSv2-Lite | GPTQ | 10.9595 | 19.3183 | 0.484888 | 0.350683 | 0.610269 | 0.662133 | 0.525382 | 0.553275 | 0.341416 | 0.514838 | 0.321108 |
| DSv2-Lite | IPE-TQ | **7.5585** | **13.2008** | **0.599571** | 0.462457 | 0.75968 | 0.735038 | 0.712538 | 0.646409 | 0.411717 | 0.640012 | 0.428714 |
| Moonlight | Energy | 17.3587 | 31.791 | 0.51768 | 0.358362 | 0.640152 | 0.67519 | 0.67156 | 0.569061 | 0.379012 | 0.507269 | 0.340835 |
| Moonlight | GPTQ | 21.8254 | 48.1886 | 0.454362 | 0.290102 | 0.487795 | 0.619151 | 0.635168 | 0.512234 | 0.371778 | 0.409381 | 0.309286 |
| Moonlight | IPE-TQ | **11.0685** | **24.6819** | **0.543562** | 0.393345 | 0.701178 | 0.699674 | 0.636697 | 0.572218 | 0.38543 | 0.556164 | 0.403789 |
| OLMoE | Energy | 33.461 | 46.6504 | 0.516981 | 0.374573 | 0.61069 | 0.682807 | 0.653517 | 0.586425 | 0.399796 | 0.540829 | 0.28721 |
| OLMoE | GPTQ | 24.1306 | 38.991 | 0.469537 | 0.306314 | 0.538721 | 0.631665 | 0.584404 | 0.549329 | 0.411513 | 0.468433 | 0.265917 |
| OLMoE | IPE-TQ | **15.985** | **23.9342** | **0.565929** | 0.438567 | 0.70665 | 0.694777 | 0.664832 | 0.655091 | 0.438309 | 0.585242 | 0.343968 |
| Qwen3 | Energy | 12.1434 | 19.1257 | 0.67474 | 0.56314 | 0.807239 | 0.732862 | 0.851376 | 0.67719 | 0.659908 | 0.480084 | 0.626122 |
| Qwen3 | GPTQ | 14.849 | 26.7429 | 0.488454 | 0.302048 | 0.520202 | 0.667029 | 0.723547 | 0.584057 | 0.407132 | 0.418044 | 0.285572 |
| Qwen3 | IPE-TQ | **10.0371** | **15.5216** | **0.717291** | 0.606655 | 0.842593 | 0.762242 | 0.87156 | 0.708761 | 0.658686 | 0.68094 | 0.606894 |

#### 2.0 bpw (+0.25)

| Model | Method | WikiText2 | C4 | Avg. | ARC-Challenge | ARC-Easy | PIQA | BoolQ | Winogrande | MNLI | Hellaswag | MMLU |
|-------|--------|-----------|----|-------|----------|---------------|------|-------|------------|-----------|------|-------|
| DSMoE-v1 | Energy | 7.8037 | 11.8563 | 0.623815 | 0.464164 | 0.750421 | 0.76333 | 0.763914 | 0.711918 | 0.44972 | 0.728839 | 0.358211 |
| DSMoE-v1 | GPTQ | 8.0116 | 12.27 | 0.604907 | 0.435154 | 0.738636 | 0.784548 | 0.679511 | 0.700079 | 0.424554 | 0.724258 | 0.352514 |
| DSMoE-v1 | IPE-TQ | **7.3066** | **11.4685** | **0.631659** | 0.475256 | 0.776094 | 0.784004 | 0.756881 | 0.708761 | 0.459501 | 0.71669 | 0.376086 |
| DSv2-Lite | Energy | 7.3503 | 11.267 | 0.66488 | 0.504266 | 0.789983 | 0.773667 | 0.796636 | 0.707182 | 0.502191 | 0.741984 | 0.503133 |
| DSv2-Lite | GPTQ | 8.0724 | 12.5196 | 0.598768 | 0.467577 | 0.742003 | 0.759521 | 0.607951 | 0.68824 | 0.377789 | 0.699761 | 0.447301 |
| DSv2-Lite | IPE-TQ | **6.8507** | **10.827** | **0.6863** | 0.505973 | 0.800084 | 0.784548 | 0.773394 | 0.704815 | 0.501172 | 0.734117 | - |
| Moonlight | Energy | 10.3568 | 20.7624 | 0.607224 | 0.467577 | 0.746633 | 0.732862 | 0.733945 | 0.594317 | 0.427509 | 0.62846 | 0.526492 |
| Moonlight | GPTQ | 10.2971 | 25.4002 | 0.540786 | 0.385666 | 0.688973 | 0.678455 | 0.656269 | 0.572218 | 0.37514 | 0.531169 | 0.438399 |
| Moonlight | IPE-TQ | **8.2281** | **17.3858** | **0.615681** | 0.493174 | 0.772306 | 0.746464 | 0.671254 | 0.617995 | 0.438003 | 0.649074 | 0.537174 |
| OLMoE | Energy | 17.2839 | 22.7481 | 0.613368 | 0.46843 | 0.733586 | 0.720892 | 0.747706 | 0.647987 | 0.496281 | 0.681936 | 0.410127 |
| OLMoE | GPTQ | 15.9601 | 22.2873 | 0.553202 | 0.379693 | 0.642256 | 0.686072 | 0.670948 | 0.600631 | 0.444524 | 0.624477 | 0.377012 |
| OLMoE | IPE-TQ | **12.5608** | **17.4607** | **0.639467** | 0.509386 | 0.760943 | 0.761697 | 0.740061 | 0.685083 | 0.505655 | 0.702051 | 0.450862 |
| Qwen3 | Energy | 10.3297 | 14.9735 | 0.746051 | 0.659556 | 0.862374 | 0.797606 | 0.882569 | 0.700079 | 0.787978 | 0.536547 | 0.741703 |
| Qwen3 | GPTQ | 11.4401 | 18.3998 | 0.619412 | 0.451365 | 0.726852 | 0.752448 | 0.822936 | 0.662983 | 0.575853 | 0.508365 | 0.454494 |
| Qwen3 | IPE-TQ | **9.4174** | **13.7852** | **0.762076** | 0.673208 | 0.867003 | 0.792709 | 0.884709 | 0.69929 | 0.743556 | 0.737801 | 0.698334 |

We prioritize outputting acc_norm from LM-Evaluation-Harness, Avg. and ARC-Challenge, ARC-Easy, PIQA, Hellaswag

### random seed effect

**Model**: deepseek-moe-16b-base/

#### Random seed stability comparison (2.0 bpw)

| Seed | Fixed scheme (a8s8m2) | | Global DP scheme (global-bpw-a8s8m2) | |
|------|-----------|----|----------|---------------|
| | WikiText2 | C4 | WikiText2 | C4 |
| 0 | 24.2966 | 33.808 | 7.332 | 11.4409 |
| 42 | 11.7607 | 16.495 | 7.2821 | 11.461 |
| 84 | 13.4714 | 22.1287 | 7.3532 | 11.472 |
| 126 | 15.6115 | 23.1076 | 7.301 | 11.4183 |
| 168 | 21.9776 | 32.4284 | 7.3165 | 11.4494 |
| 210 | 11.7193 | 18.9774 | 7.3161 | 11.4811 |
| 252 | 11.9574 | 19.2281 | 7.3351 | 11.5237 |
| 294 | 11.1694 | 17.5287 | 7.3026 | 11.4379 |

## Installation

### Prerequisites

```bash
conda env create -f environment.yml
conda activate dartmoq
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+
- Transformers
- Datasets
- NumPy
- Matplotlib

## Usage

### Basic Command

```bash
python run_dartmoq.py \
    <model_path> \
    <dataset> \
    [--slices N] \
    [--nsamples N] \
    [--rank-mode MODE] \
    [--quant-scheme SCHEME] \
    [--quantmode {gptq,turboquant}] \
    [--eval-zero] \
    [--save-model]
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `model` | Path to HuggingFace model checkpoint | **Required** |
| `dataset` | Calibration dataset: `wikitext2`, `ptb`, or `c4` | **Required** |
| `--seed` | Random seed for calibration sampling | 42 |
| `--nsamples` | Number of calibration samples | 128 |
| `--slices` | Number of sub-experts to slice (S) | 1 |
| `--rank-mode` | Neuron ranking mode for expert reordering | None |
| `--quant-scheme` | Quantization scheme (fixed or global) | None |
| `--quantmode` | Quantization algorithm: `gptq` or `turboquant` | `gptq` |
| `--eval-zero` | Enable zero-shot task evaluation | False |
| `--save-model` | Save quantized model to disk | False |
| `--standby-layer-cpu` | Move layers to CPU during quantization | False |
| `--no-use-hybrid-moe` | Disable hybrid MoE structure and use original experts | False (hybrid enabled by default) |

## Rank Modes (`--rank-mode`)

The rank mode determines how neurons are ordered within each expert for optimal quantization. Different modes are optimized for different quantization algorithms.

### Activation-Based Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `expert_activation` | Rank neurons by activation frequency in input samples | Baseline comparison |
| `energy` | Rank neurons by energy contribution (from CAMERA, for comparison only) to output | Interpretability-focused, baseline comparison |
| `random` | Random neuron ordering for baseline testing | Baseline comparison |
| `neuron_index` | Original neuron index order | Baseline comparison |

### GPTQ-Specific Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `gptq_quant_outlier` | Rank by GPTQ quantization loss, identifying error-sensitive neurons | **GPTQ quantization** |

### TurboQuant-Specific Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `turboquant_innerproduct` | TurboQuant outlier analysis using inner product loss | **Recommended for TurboQuant** |
| `turboquant_iipl` | TurboQuant with Input-Intermediate Product Loss | TurboQuant (alternative) |
| `turboquant_diagonal` | TurboQuant with diagonal Hessian approximation | Computationally constrained |
| `turboquant_hessian` | TurboQuant with full Hessian computation | Highest accuracy (slower) |
| `turboquant_qjl_sensitivity` | TurboQuant with quantized Johnson-Lindenstrauss sensitivity | Theoretical exploration |
| `turboquant_iipl_fea` | TurboQuant IIPL with full experts activation | Not recommended |
| `turboquant_innerproduct_fea` | TurboQuant inner product with full experts activation | Not recommended |

## Quantization Schemes (`--quant-scheme`)

The quant scheme determines how bits are allocated to neurons/blocks.

### Fixed Bit Schemes

Format: `a{A}s{S}m{BIT_STRING}`

- `A`: Number of experts (activation)
- `S`: Number of slices/sub-experts per expert
- `BIT_STRING`: Bit allocation for each slice (length must equal S)

Examples:
- `a8s8m22222222`: 8 experts, 8 slices each, all slices get 2 bits (2.0 bpw)
- `a8s8m44332211`: 8 experts, 8 slices each, bits decrease from 4 to 1 (2.5 bpw average)
- `a8s4m3322`: 8 experts, 4 slices each (2.5 bpw average)

### Global Dynamic Programming Schemes

Format: `global-bpw-a{A}s{S}m{BPW}`

Uses the global DP optimizer with monotonic non-increasing bit allocation constraint across all experts.

- `A`: Number of experts
- `S`: Number of slices per expert
- `BPW`: Target average bits per weight (can be fractional)

**Important Note**: The bpw values in all schemes (both fixed and `global-bpw`) refer to the weight bit allocation only. They do **not** include the additional overhead of:
- GPTQ: ~0.25 bpw for quantization parameters
- TurboQuant: ~0.252 bpw for quantization parameters

All computations use a consistent groupsize of 128. The actual total bpw will be approximately `target_bpw + 0.25` (GPTQ) or `target_bpw + 0.252` (TurboQuant).

Examples:
- `global-bpw-a8s8m0.5`: 8 experts, 8 slices each, ~0.5 bpw target (excluding overhead)
- `global-bpw-a8s8m1.0`: 8 experts, 8 slices each, ~1.0 bpw target (excluding overhead)
- `global-bpw-a8s8m1.5`: 8 experts, 8 slices each, ~1.5 bpw target (excluding overhead)
- `global-bpw-a8s8m2.0`: 8 experts, 8 slices each, ~2.0 bpw target (excluding overhead)
- `global-bpw-a8s8m2.5`: 8 experts, 8 slices each, ~2.5 bpw target (excluding overhead)

### How Global DP Works

1. **Per-expert neuron sorting**: Neurons in each expert are sorted by sensitivity
2. **Global sub-expert sorting**: All sub-experts are globally sorted by importance (sensitivity × expert activation rate)
3. **Monotonic DP search**: Find optimal bit allocation with non-increasing bit constraint
4. **Remap to per-expert schemes**: Map global allocation back to each expert

## Quantization Modes (`--quantmode`)

### GPTQ (`gptq`)

- **Type**: Per-row quantization
- **Sensitivity metric**: Element-wise MSE
- **Best rank mode**: `gptq_quant_outlier`
- **Strengths**: Well-understood, good stability, mature implementation
- **Overhead**: ~0.25 bpw additional (groupsize=128)

### TurboQuant (`turboquant`)

- **Type**: Vector quantization with global random rotation
- **Sensitivity metric**: Inner product loss on calibration manifold
- **Best rank mode**: `turboquant_innerproduct` (recommended)
- **Strengths**: Better compression at extremely low bits, energy homogenization
- **Overhead**: ~0.252 bpw additional (groupsize=128)

## Recommended Combinations

### For 2-bit Industry Deployment

```bash
# TurboQuant version (recommended for best quality)
python run_dartmoq.py \
    $MODEL_PATH \
    wikitext2 \
    --slices 8 \
    --nsamples 64 \
    --rank-mode turboquant_innerproduct \
    --quant-scheme a8s8m22222222 \
    --quantmode turboquant \
    --eval-zero

# GPTQ version
python run_dartmoq.py \
    $MODEL_PATH \
    wikitext2 \
    --slices 8 \
    --nsamples 64 \
    --rank-mode gptq_quant_outlier \
    --quant-scheme a8s8m22222222 \
    --quantmode gptq \
    --eval-zero
```

### For Global Optimal Search (Any BPW)

```bash
# TurboQuant with global DP
python run_dartmoq.py \
    $MODEL_PATH \
    wikitext2 \
    --slices 8 \
    --nsamples 64 \
    --rank-mode turboquant_innerproduct \
    --quant-scheme global-bpw-a8s8m1.5 \
    --quantmode turboquant \
    --eval-zero

# GPTQ with global DP
python run_dartmoq.py \
    $MODEL_PATH \
    wikitext2 \
    --slices 8 \
    --nsamples 64 \
    --rank-mode gptq_quant_outlier \
    --quant-scheme global-bpw-a8s8m1.5 \
    --quantmode gptq \
    --eval-zero
```

### For BPW Sweep

See `run.sh` for examples of sweeping across bpw values from 0.5 to 3.0.

## Supported Models

- DeepSeek-MoE-16B
- DeepSeek-V2-Lite
- OLMoE-1B-7B
- Moonlight-16B-A3B
- Qwen3-30B-A3B
- Most other MoE architectures with expert FFNs

## Calibration Datasets

- `wikitext2`: Wikitext-2 (recommended for most use cases)
- `c4`: C4 (Colossal Clean Crawled Corpus)
- `ptb`: Penn Treebank

64-128 samples are typically sufficient for good calibration.

## Output Files

- Quantization cache: `quant_outlier_{gptq,turboquant}/{model_id}/`
- Visualizations: `plot/`
- Saved models: `models/dartmoq_{model_type}_{rank_mode}_{quant_scheme}/`

## Visualization Tools

DartMoQP includes visualization modules in the `viz/` directory:

```bash
# Headroom analysis
python -m viz.headroom

# Metric geometry analysis
python -m viz.metric_geometry

# Loss distribution plots
python -m viz.distribution

# Activation rate analysis
python -m viz.dump_activation_rates
```

## Dense Model Quantization Support

DartMoQP is specifically designed for **Mixture-of-Experts (MoE) models** and does not support dense (non-MoE) models. For mixed-precision quantization of dense models, please refer to our dedicated method:

[DartMQ](https://github.com/zzningxp/DartMQ) — A unified framework for mixed-precision quantization of dense transformer models.

## Citation

If you use DartMoQP in your research, please cite:

```bibtex
@article{dartmoqp2024,
  title={DartMoQP: A MoE-Native Unified Framework for Mixed-Precision Quantization and Structured Pruning},
  author={Zhaoning Zhang},
  year={2026}
}
```

## License

This project is released under the same license as the base models it quantizes. Please refer to the original model licenses for details.

## Acknowledgments

- GPTQ for the per-row quantization baseline
- TurboQuant for the vector quantization approach
- CAMERA for energy-based importance estimation (for comparison only)
- All the MoE model authors for their open-source contributions

