
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
| DSMoEv1 | Energy | 278.704 | 573.556 | 0.347 | 0.245 | 0.256 | 0.519 | 0.379 | 0.506 | 0.363 | 0.271 | 0.235 |
| DSMoEv1 | GPTQ | 13.255 | 24.718 | 0.488 | 0.314 | 0.602 | 0.675 | 0.613 | 0.590 | 0.363 | 0.485 | 0.258 |
| DSMoEv1 | IPE-TQ | **10.992** | **23.668** | **0.509** | 0.360 | 0.654 | 0.661 | 0.674 | 0.615 | 0.393 | 0.471 | 0.243 |
| DSv2-Lite | Energy | 37.792 | 51.508 | 0.384 | 0.230 | 0.316 | 0.550 | 0.572 | 0.504 | 0.336 | 0.307 | 0.260 |
| DSv2-Lite | GPTQ | 47.363 | 80.396 | 0.376 | 0.206 | 0.295 | 0.537 | 0.575 | 0.520 | 0.340 | 0.289 | 0.242 |
| DSv2-Lite | IPE-TQ | **9.686** | **20.855** | **0.487** | 0.346 | 0.635 | 0.657 | 0.535 | 0.551 | 0.369 | 0.480 | 0.326 |
| Moonlight | Energy | 249.477 | 333.547 | 0.384 | 0.218 | 0.340 | 0.552 | 0.556 | 0.501 | 0.354 | 0.296 | 0.254 |
| Moonlight | GPTQ | 69.867 | 145.503 | 0.383 | 0.235 | 0.324 | 0.535 | 0.585 | 0.484 | 0.344 | 0.301 | 0.255 |
| Moonlight | IPE-TQ | **17.770** | **44.966** | **0.453** | 0.282 | 0.552 | 0.597 | 0.621 | 0.517 | 0.361 | 0.405 | 0.286 |
| OLMoE | Energy | 16753.113 | 8156.675 | 0.374 | 0.264 | 0.292 | 0.521 | 0.565 | 0.504 | 0.319 | 0.262 | 0.263 |
| OLMoE | GPTQ | 157.633 | 278.536 | 0.388 | 0.228 | 0.341 | 0.534 | 0.603 | 0.514 | 0.347 | 0.293 | 0.244 |
| OLMoE | IPE-TQ | **26.214** | **50.927** | **0.467** | 0.338 | 0.571 | 0.618 | 0.614 | 0.553 | 0.365 | 0.407 | 0.269 |
| Qwen3 | Energy | 1886.718 | 1422.791 | 0.355 | 0.230 | 0.303 | 0.527 | 0.417 | 0.513 | 0.336 | 0.268 | 0.243 |
| Qwen3 | GPTQ | 14.977 | 26.547 | 0.502 | 0.303 | 0.526 | 0.638 | 0.703 | 0.636 | 0.405 | 0.522 | 0.286 |
| Qwen3 | IPE-TQ | **11.683** | **20.935** | **0.593** | 0.436 | 0.703 | 0.666 | 0.805 | 0.666 | 0.473 | 0.550 | 0.447 |

#### 1.5 bpw (+0.25)

| Model | Method | WikiText2 | C4 | Avg. | ARC-Challenge | ARC-Easy | PIQA | BoolQ | Winogrande | MNLI | Hellaswag | MMLU |
|-------|--------|-----------|----|-------|----------|---------------|------|-------|------------|-----------|------|-------|
| DSMoEv1 | Energy | 9.556 | 15.567 | 0.559 | 0.402 | 0.687 | 0.696 | 0.715 | 0.681 | 0.413 | 0.613 | 0.267 |
| DSMoEv1 | GPTQ | 9.182 | **14.573** | 0.556 | 0.380 | 0.671 | 0.740 | 0.630 | 0.654 | 0.419 | 0.650 | 0.308 |
| DSMoEv1 | IPE-TQ | **8.303** | 14.556 | **0.583** | 0.437 | 0.751 | 0.736 | 0.724 | 0.670 | 0.396 | 0.628 | 0.321 |
| DSv2-Lite | Energy | 8.982 | 14.218 | 0.600 | 0.439 | 0.721 | 0.715 | 0.772 | 0.640 | 0.442 | 0.638 | 0.433 |
| DSv2-Lite | GPTQ | 10.960 | 19.318 | 0.485 | 0.351 | 0.610 | 0.662 | 0.525 | 0.553 | 0.341 | 0.515 | 0.321 |
| DSv2-Lite | IPE-TQ | **7.559** | **13.201** | **0.600** | 0.462 | 0.760 | 0.735 | 0.713 | 0.646 | 0.412 | 0.640 | 0.429 |
| Moonlight | Energy | 17.359 | 31.791 | 0.518 | 0.358 | 0.640 | 0.675 | 0.672 | 0.569 | 0.379 | 0.507 | 0.341 |
| Moonlight | GPTQ | 21.825 | 48.189 | 0.454 | 0.290 | 0.488 | 0.619 | 0.635 | 0.512 | 0.372 | 0.409 | 0.309 |
| Moonlight | IPE-TQ | **11.069** | **24.682** | **0.544** | 0.393 | 0.701 | 0.700 | 0.637 | 0.572 | 0.385 | 0.556 | 0.404 |
| OLMoE | Energy | 33.461 | 46.650 | 0.517 | 0.375 | 0.611 | 0.683 | 0.654 | 0.586 | 0.400 | 0.541 | 0.287 |
| OLMoE | GPTQ | 24.131 | 38.991 | 0.470 | 0.306 | 0.539 | 0.632 | 0.584 | 0.549 | 0.412 | 0.468 | 0.266 |
| OLMoE | IPE-TQ | **15.985** | **23.934** | **0.566** | 0.439 | 0.707 | 0.695 | 0.665 | 0.655 | 0.438 | 0.585 | 0.344 |
| Qwen3 | Energy | 12.143 | 19.126 | 0.675 | 0.563 | 0.807 | 0.733 | 0.851 | 0.677 | 0.660 | 0.480 | 0.626 |
| Qwen3 | GPTQ | 14.849 | 26.743 | 0.488 | 0.302 | 0.520 | 0.667 | 0.724 | 0.584 | 0.407 | 0.418 | 0.286 |
| Qwen3 | IPE-TQ | **10.037** | **15.522** | **0.717** | 0.607 | 0.843 | 0.762 | 0.872 | 0.709 | 0.659 | 0.681 | 0.607 |

#### 2.0 bpw (+0.25)

| Model | Method | WikiText2 | C4 | Avg. | ARC-Challenge | ARC-Easy | PIQA | BoolQ | Winogrande | MNLI | Hellaswag | MMLU |
|-------|--------|-----------|----|-------|----------|---------------|------|-------|------------|-----------|------|-------|
| DSMoEv1 | Energy | 7.804 | 11.856 | 0.624 | 0.464 | 0.750 | 0.763 | 0.764 | 0.712 | 0.450 | 0.729 | 0.358 |
| DSMoEv1 | GPTQ | 8.012 | 12.270 | 0.605 | 0.435 | 0.739 | 0.785 | 0.680 | 0.700 | 0.425 | 0.724 | 0.353 |
| DSMoEv1 | IPE-TQ | **7.307** | **11.469** | **0.632** | 0.475 | 0.776 | 0.784 | 0.757 | 0.709 | 0.460 | 0.717 | 0.376 |
| DSv2-Lite | Energy | 7.350 | 11.267 | 0.665 | 0.504 | 0.790 | 0.774 | 0.797 | 0.707 | 0.502 | 0.742 | 0.503 |
| DSv2-Lite | GPTQ | 8.072 | 12.520 | 0.599 | 0.468 | 0.742 | 0.760 | 0.608 | 0.688 | 0.378 | 0.700 | 0.447 |
| DSv2-Lite | IPE-TQ | **6.851** | **10.827** | **0.686** | 0.506 | 0.800 | 0.785 | 0.773 | 0.705 | 0.501 | 0.734 | - |
| Moonlight | Energy | 10.357 | 20.762 | 0.607 | 0.468 | 0.747 | 0.733 | 0.734 | 0.594 | 0.428 | 0.628 | 0.526 |
| Moonlight | GPTQ | 10.297 | 25.400 | 0.541 | 0.386 | 0.689 | 0.678 | 0.656 | 0.572 | 0.375 | 0.531 | 0.438 |
| Moonlight | IPE-TQ | **8.228** | **17.386** | **0.616** | 0.493 | 0.772 | 0.746 | 0.671 | 0.618 | 0.438 | 0.649 | 0.537 |
| OLMoE | Energy | 17.284 | 22.748 | 0.613 | 0.468 | 0.734 | 0.721 | 0.748 | 0.648 | 0.496 | 0.682 | 0.410 |
| OLMoE | GPTQ | 15.960 | 22.287 | 0.553 | 0.380 | 0.642 | 0.686 | 0.671 | 0.601 | 0.445 | 0.624 | 0.377 |
| OLMoE | IPE-TQ | **12.561** | **17.461** | **0.639** | 0.509 | 0.761 | 0.762 | 0.740 | 0.685 | 0.506 | 0.702 | 0.451 |
| Qwen3 | Energy | 10.330 | 14.973 | 0.746 | 0.660 | 0.862 | 0.798 | 0.883 | 0.700 | 0.788 | 0.537 | 0.742 |
| Qwen3 | GPTQ | 11.440 | 18.400 | 0.619 | 0.451 | 0.727 | 0.752 | 0.823 | 0.663 | 0.576 | 0.508 | 0.454 |
| Qwen3 | IPE-TQ | **9.417** | **13.785** | **0.762** | 0.673 | 0.867 | 0.793 | 0.885 | 0.699 | 0.744 | 0.738 | 0.698 |

We prioritize outputting acc_norm from LM-Evaluation-Harness. Tasks like ARC-Challenge, ARC-Easy, PIQA, and Hellaswag use acc_norm.

### random seed effect

**Model**: deepseek-moe-16b-base/

#### Random seed stability comparison (2.0 +0.25 bpw)

| Seed | Fixed scheme<br>(2slice) | | Fixed scheme<br>（8slices） | | Global DP scheme<br>(global-bpw-a8s8m2) | |
|------|-----------|----|----------|---------------|----|----|
| | WikiText2 | C4 | WikiText2 | C4 | WikiText2 | C4 |
| 0 | 24.297 | 33.808 | 23.944 | 33.586 | 7.332 | 11.441 |
| 42 | 11.761 | 16.495 | 11.620 | 16.665 | 7.282 | 11.461 |
| 84 | 13.471 | 22.129 | 13.319 | 22.085 | 7.353 | 11.472 |
| 126 | 15.611 | 23.108 | 15.648 | 23.039 | 7.301 | 11.418 |
| 168 | 21.978 | 32.428 | 21.296 | 32.163 | 7.316 | 11.449 |
| 210 | 11.719 | 18.977 | 11.612 | 18.716 | 7.316 | 11.481 |
| 252 | 11.957 | 19.228 | 11.971 | 19.062 | 7.335 | 11.524 |
| 294 | 11.169 | 17.529 | 11.121 | 17.327 | 7.303 | 11.438 |

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
| `--quantmode` | Quantization algorithm: `gptq` or `turboquant` | `turboquant` |
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

### For 2-bit ManualDeployment

```bash
# TurboQuant version (recommended for best quality)
python run_dartmoq.py \
    $MODEL_PATH \
    wikitext2 \
    --slices 8 \
    --nsamples 64 \
    --rank-mode turboquant_innerproduct \
    --quant-scheme global-a8s8m32222221 \
    --quantmode turboquant \
    --eval-zero

# GPTQ version
python run_dartmoq.py \
    $MODEL_PATH \
    wikitext2 \
    --slices 8 \
    --nsamples 64 \
    --rank-mode gptq_quant_outlier \
    --quant-scheme a8s8m44222220 \
    --quantmode gptq \
    --eval-zero
```
`global-a8s8m32222221` is the quantization scheme closest to 2 + 0.25 bpw in paper Camera-Q.

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

See `run.sh` for examples of sweeping across bpw values from 0.5 to 4.0.

## Supported Models

- `DeepSeek-MoE-16B` (16B-A3B)
- `DeepSeek-V2-Lite` (16B-A3B)
- `OLMoE-1B-7B` (7B-A1B)
- `Moonlight-16B-A3B`
- `Qwen3-30B-A3B`
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
- CAMERA (http://arxiv.org/abs/2508.02322) for energy-based importance estimation (for comparison only)
- All the MoE model authors for their open-source contributions

