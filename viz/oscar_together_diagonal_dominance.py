"""Verify diagonal dominance in FFN projections with DP sorted blocking.

This script:
1. Collects MoE FFN weights (up/gate/down proj) and activations
2. Applies Random QR rotation and TurboQuant quantization
3. Computes residual covariance along hidden_size dimension for all projections
   - up/gate: quantize along in_features (hidden_size)
   - down: transpose and quantize along out_features (hidden_size) to match up/gate
4. Performs DP-style sorted blocking analysis
5. Plots diagonal element distribution for full matrix and sorted blocks

Usage:
  python viz/oscar_together_residual_verify.py --model deepseek-moe-16b --layers 26
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from functools import wraps

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def timeit(func):
    """Decorator to print timing info for a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"  ⏱️ {func.__name__}: {elapsed:.2f}s")
        return result
    return wrapper


def print_timestamp(message: str) -> None:
    """Print a timestamped message."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

# Make sibling modules importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import get_loaders
from eval_dartmoq import load_model
from viz._cache_io import resolve_model_id, resolve_model_path, apply_paper_style
from turboquant_utils.quantize import turboquant_quantize
from turboquant_utils.rotation import generate_rotation_matrix
from dp_utils import get_unified_sorted_idx_general

DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def to_device(obj, device):
    """Recursively move tensors in nested structures (tuples/lists/dicts) to device."""
    if obj is None:
        return None
    if hasattr(obj, 'to'):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj


# Plot directory
PLOT_DIR = os.path.join('plot', 'oscar_together_diagonal_dominance')

# Number of blocks for DP-style blocking
DEFAULT_NUM_BLOCKS = 8


@torch.no_grad()
def compute_residual_covariance_with_turboquant(
    W: torch.Tensor,
    rotation_matrix: torch.Tensor | None,
    bit_width: int = 4,
    group_size: int | None = 128,
    seed: int = 42,
    transpose: bool = False,
) -> dict:
    """Compute residual covariance using TurboQuant quantization with Random QR rotation.

    Args:
        W: Weight matrix (out_features, in_features)
        rotation_matrix: Random QR rotation matrix (d, d)
        bit_width: Quantization bit width
        group_size: TurboQuant group size
        seed: Random seed
        transpose: If True, transpose W before quantizing (for down_proj)

    Returns:
        dict with residual covariance, statistics, and raw data for analysis
    """
    device = W.device

    # Transpose if needed (for down_proj)
    if transpose:
        W = W.T  # Now shape is (in_features, out_features), quantize along out_features (original hidden_size)

    d = W.shape[1]

    # Step 1: Apply Random QR rotation
    if rotation_matrix is not None:
        rotation_matrix = rotation_matrix.to(device)
        W_rotated = W @ rotation_matrix.T  # (out_features, d)
    else:
        W_rotated = W

    # Save weight dynamic range info
    W_range_before = {
        'min': float(W_rotated.min()),
        'max': float(W_rotated.max()),
        'mean': float(W_rotated.abs().mean()),
        'std': float(W_rotated.std()),
        'range': float(W_rotated.max() - W_rotated.min()),
        'per_col_min': W_rotated.abs().min(dim=0)[0].cpu().numpy(),
        'per_col_max': W_rotated.abs().max(dim=0)[0].cpu().numpy(),
        'per_col_range': (W_rotated.abs().max(dim=0)[0] - W_rotated.abs().min(dim=0)[0]).cpu().numpy(),
    }

    # Step 2: Quantize using TurboQuant with Random QR rotation
    W_quant_rotated = turboquant_quantize(
        W_rotated,
        bit_width=bit_width,
        group_size=group_size,
        seed=seed,
        rotation='qr'
    ).float()

    # Save quantized weight dynamic range info
    W_range_after = {
        'min': float(W_quant_rotated.min()),
        'max': float(W_quant_rotated.max()),
        'mean': float(W_quant_rotated.abs().mean()),
        'std': float(W_quant_rotated.std()),
        'range': float(W_quant_rotated.max() - W_quant_rotated.min()),
    }

    # Step 3: Compute residual
    residual = W_quant_rotated - W_rotated  # (out_features, d)

    # Step 4: Compute residual covariance E = sum_{i=1 to N} r_i^T r_i
    # Compute E before transposing back!
    E = residual.T @ residual  # (d, d)

    # Step 5: Compute statistics on E before transposing back
    diag_E = torch.diag(E)
    off_diag_E = E.clone()
    off_diag_E.diagonal().zero_()

    frob_E = E.norm('fro')
    frob_diag = diag_E.norm(2)
    frob_off_diag = off_diag_E.norm('fro')

    diagonal_dominance_ratio = float(frob_diag / frob_E.clamp_min(1e-12))

    # Check monotonicity of sorted diagonal (ascending)
    sorted_diag, _ = diag_E.sort()  # ascending
    monotonicity_violations = float(((sorted_diag[1:] - sorted_diag[:-1]) < -1e-9).float().sum())
    monotonicity_violation_rate = monotonicity_violations / max(len(sorted_diag) - 1, 1)

    max_off_diag = float(off_diag_E.abs().max())
    cond_number = float(diag_E.max() / diag_E.clamp_min(1e-12).min())

    # Collect off-diagonal elements
    triu_indices = torch.triu_indices(d, d, offset=1, device=device)
    off_diag_elements = off_diag_E[triu_indices[0], triu_indices[1]].cpu().numpy()

    # Transpose back if needed (for other metrics that require original shape)
    if transpose:
        W = W.T
        W_rotated = W_rotated.T
        W_quant_rotated = W_quant_rotated.T
        residual = residual.T
        # Keep E, diag_E, etc. remain as computed along quantized dimension!

    # Norm preservation - compute Frobenius norms
    frob_W_orig = W.norm('fro')
    frob_W_rot = W_rotated.norm('fro')
    frob_W_quant = W_quant_rotated.norm('fro')

    return {
        # Main metrics
        'diagonal_dominance_ratio': diagonal_dominance_ratio,
        'frob_E': float(frob_E),
        'frob_diag': float(frob_diag),
        'frob_off_diag': float(frob_off_diag),
        'max_off_diag': max_off_diag,
        'monotonicity_violation_rate': monotonicity_violation_rate,
        'cond_number': cond_number,

        # Norm preservation
        'frob_W_orig': float(frob_W_orig),
        'frob_W_rot': float(frob_W_rot),
        'frob_W_quant': float(frob_W_quant),
        'norm_ratio_rot_vs_orig': float(frob_W_rot / frob_W_orig.clamp_min(1e-12)),

        # Raw data for plotting
        'diag_E_raw': diag_E.cpu().numpy(),
        'sorted_diag_E': sorted_diag.cpu().numpy(),

        # Dynamic range
        'W_range_before': W_range_before,
        'W_range_after': W_range_after,

        # Off-diagonal elements
        'off_diag_elements': off_diag_elements,
        'E_matrix': E.cpu().numpy(),
    }


@torch.no_grad()
def compute_neuron_rates_for_dp(
    W: torch.Tensor,
    bits: list[int],
    group_size: int | None = 128,
    seed: int = 42,
    transpose: bool = False,
) -> dict[int, np.ndarray]:
    """
    Compute per-neuron loss rates for different bit widths for DP sorting.

    Args:
        W: Weight matrix (out_features, in_features)
        bits: List of bit widths to evaluate
        group_size: TurboQuant group size
        seed: Random seed
        transpose: If True, transpose W before quantizing (for down_proj)

    Returns:
        rates: Dict {bit: loss_array} where loss_array has shape (in_features,)
    """
    device = W.device

    # Transpose if needed (for down_proj)
    if transpose:
        W = W.T  # Now shape is (in_features, out_features), quantize along out_features

    d = W.shape[1]

    rates = {}

    for bit in bits:
        # Quantize with this bit width and Random QR rotation
        W_quant = turboquant_quantize(
            W, bit_width=bit, group_size=group_size, seed=seed, rotation='qr'
        ).float()

        # Compute per-neuron MSE loss (sum over output features)
        residual = W_quant - W
        per_neuron_loss = (residual ** 2).sum(dim=0).cpu().numpy()  # (in_features,)

        rates[bit] = per_neuron_loss

    return rates


@torch.no_grad()
def compute_block_residual_analysis_with_dp(
    W: torch.Tensor,
    rotation_matrix: torch.Tensor | None,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    target_bpw: float = 4.0,
    group_size: int | None = 128,
    seed: int = 42,
    transpose: bool = False,
) -> dict:
    """
    Compute residual analysis with sorted blocking (fixed all-same-bit scheme).

    Steps:
    1. Compute per-neuron loss rates for target bit width (for sorting)
    2. Sort neurons by sensitivity (loss descending)
    3. Split into blocks, all blocks use the same target bit width

    Args:
        W: Weight matrix (out_features, in_features)
        rotation_matrix: Random QR rotation matrix (unused, block-specific)
        activations: Input activations
        num_blocks: Number of blocks
        target_bpw: Target bits per weight (all blocks use this)
        group_size: TurboQuant group size
        seed: Random seed
        transpose: If True, transpose W before quantizing (for down_proj)

    Returns:
        dict with per-block analysis
    """
    # Use target_bpw as the fixed bit width for all blocks
    fixed_bit = int(target_bpw)

    device = W.device

    # Transpose if needed (for down_proj)
    if transpose:
        W = W.T  # Now shape is (in_features, out_features), quantize along out_features

    d = W.shape[1]

    print_timestamp(f"  Fixed scheme: Computing neuron rates for {fixed_bit} bit...")
    # Only compute rates for the fixed bit (for sorting)
    rates = compute_neuron_rates_for_dp(W, [fixed_bit], group_size=group_size, seed=seed, transpose=False)

    # Fixed scheme: all blocks get the same bit
    best_scheme = [fixed_bit] * num_blocks
    neuron_bits = np.full(d, fixed_bit)

    print(f"    Fixed scheme: {best_scheme}")
    print(f"    Fixed bpw: {np.mean(neuron_bits):.2f}")

    # Get sorted indices (using the only bit we have)
    sorted_idx = get_unified_sorted_idx_general(rates, [fixed_bit])

    # Verify sorting: check rates for first and last neurons
    print(f"    Sorting verification (bit={fixed_bit}):")
    print(f"      First 3 neurons (most sensitive): rates={rates[fixed_bit][sorted_idx[:3]]}")
    print(f"      Last 3 neurons (least sensitive): rates={rates[fixed_bit][sorted_idx[-3:]]}")

    # Now process each block with fixed bit width
    block_size = d // num_blocks
    block_results = []

    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = start + block_size if block_idx < num_blocks - 1 else d

        block_col_indices = sorted_idx[start:end]
        block_bit = fixed_bit

        W_block = W[:, block_col_indices]
        block_d = W_block.shape[1]

        # Compute block-specific Random QR rotation
        block_rot_matrix = generate_rotation_matrix(block_d, seed=seed + block_idx)
        if block_rot_matrix is not None:
            block_rot_matrix = block_rot_matrix.to(device)

        # Quantize with the allocated bit width
        block_analysis = compute_residual_covariance_with_turboquant(
            W_block, block_rot_matrix,
            bit_width=block_bit, group_size=group_size, seed=seed + block_idx,
            transpose=False  # Already transposed if needed at higher level
        )

        block_analysis['block_idx'] = block_idx
        block_analysis['block_size'] = end - start
        block_analysis['allocated_bit'] = block_bit
        block_analysis['col_indices'] = block_col_indices  # already numpy array

        block_results.append(block_analysis)

    return {
        'num_blocks': num_blocks,
        'block_size': block_size,
        'sorted_indices': sorted_idx,
        'neuron_bits': neuron_bits,
        'best_scheme': best_scheme,
        'rates': rates,
        'block_results': block_results,
        'transposed': transpose,
    }


@torch.no_grad()
@timeit
def verify_one_expert_enhanced(
    expert: nn.Module,
    expert_idx: int,
    layer_idx: int,
    modeltype: str,
    bit_width: int = 4,
    group_size: int | None = 128,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    seed: int = 42,
    skip_block: bool = False,
) -> dict:
    """Verify one expert's projections with DP sorted blocking.

    Returns:
        dict with verification results for up/gate/down projections
    """
    results = {}
    device = next(expert.parameters()).device

    for proj_name in ['up_proj', 'gate_proj', 'down_proj']:
        proj = getattr(expert, proj_name)
        W = proj.weight.data.float()

        # For down_proj, we transpose to quantize along hidden_size dimension
        is_down_proj = proj_name == 'down_proj'

        # Get the dimension for rotation matrix generation
        if is_down_proj:
            d = W.shape[0]  # Use out_features (hidden_size) for down_proj
            print(f"  Layer {layer_idx} Expert {expert_idx} {proj_name}: shape={tuple(W.shape)} (will transpose to quantize along hidden_size={d})")
        else:
            d = W.shape[1]  # Use in_features (hidden_size) for up/gate
            print(f"  Layer {layer_idx} Expert {expert_idx} {proj_name}: shape={tuple(W.shape)}")

        proj_results = {}

        # Generate Random QR rotation matrix for full matrix
        rot_matrix = generate_rotation_matrix(d, seed=seed + expert_idx + layer_idx)
        if rot_matrix is not None:
            rot_matrix = rot_matrix.to(device)

        # Full matrix analysis with Random QR
        proj_results['full_matrix'] = {}
        try:
            rot_seed = seed + expert_idx + layer_idx
            rot_result = compute_residual_covariance_with_turboquant(
                W, rot_matrix,
                bit_width=bit_width, group_size=group_size,
                seed=rot_seed,
                transpose=is_down_proj
            )
            proj_results['full_matrix'] = rot_result
            print(f"    Full matrix     "
                  f"diag_dominance={rot_result['diagonal_dominance_ratio']:.4f}, "
                  f"norm_ratio={rot_result['norm_ratio_rot_vs_orig']:.6f}, "
                  f"frob_E={rot_result['frob_E']:.4e}")
        except Exception as e:
            print(f"    Warning: Failed for full matrix: {e}")

        # Analysis for sorted blocking with fixed bit scheme
        if not skip_block:
            try:
                print_timestamp(f"  Starting sorted block analysis...")
                dp_block_result = compute_block_residual_analysis_with_dp(
                    W, rot_matrix,
                    num_blocks=num_blocks,
                    target_bpw=float(bit_width),  # Target same as nominal bit width
                    group_size=group_size,
                    seed=seed + expert_idx + layer_idx,
                    transpose=is_down_proj
                )
                proj_results['block_with_dp'] = dp_block_result

                # Print stats for plotting
                avg_diag_dominance = np.mean([
                    br['diagonal_dominance_ratio'] for br in dp_block_result['block_results']
                ])

                # Block 0 and 7 stats
                block_results = dp_block_result['block_results']
                b0_diag_dom = block_results[0]['diagonal_dominance_ratio'] if len(block_results) > 0 else 'N/A'
                b0_bit = block_results[0]['allocated_bit'] if len(block_results) > 0 else 'N/A'
                b0_diag_mean = np.mean(block_results[0]['diag_E_raw']) if len(block_results) > 0 else 'N/A'
                b7_diag_dom = block_results[7]['diagonal_dominance_ratio'] if len(block_results) > 7 else 'N/A'
                b7_bit = block_results[7]['allocated_bit'] if len(block_results) > 7 else 'N/A'
                b7_diag_mean = np.mean(block_results[7]['diag_E_raw']) if len(block_results) > 7 else 'N/A'

                print(f"    Sorted blocks   "
                      f"Fixed scheme: {dp_block_result['best_scheme']}")
                print(f"                    "
                      f"Block 0 (most sensitive): diag_dominance={b0_diag_dom:.4f}, bits={b0_bit}, diag_mean={b0_diag_mean:.4f}")
                print(f"                    "
                      f"Block 7 (least sensitive): diag_dominance={b7_diag_dom:.4f}, bits={b7_bit}, diag_mean={b7_diag_mean:.4f}")
                print(f"                    "
                      f"Average: diag_dominance={avg_diag_dominance:.4f}")
            except Exception as e:
                import traceback
                print(f"    Warning: Failed DP block: {e}")
                print(f"    Stack trace:\n{traceback.format_exc()}")

        results[proj_name] = proj_results

        # Clean up after each projection
        del W, rot_matrix

    # Clean up GPU cache and sleep to cool down
    torch.cuda.empty_cache()

    return results


@torch.no_grad()
def verify_layer(
    layer: nn.Module,
    layer_idx: int,
    modeltype: str,
    short_id: str,
    plot_dir: str = PLOT_DIR,
    skip_plot: bool = False,
    bit_width: int = 4,
    group_size: int | None = 128,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    seed: int = 42,
    skip_block: bool = False,
) -> dict | None:
    """Verify one layer."""
    print(f"  Layer {layer_idx}: verifying...")

    is_moe = hasattr(layer.mlp, 'gate') or hasattr(layer.mlp, 'experts')

    if not is_moe:
        print(f"  Layer {layer_idx}: not MoE, skipping")
        return None

    if hasattr(layer.mlp, 'experts'):
        ori_expert_num = len(layer.mlp.experts)
        experts = layer.mlp.experts
    else:
        ori_expert_num = 1
        experts = [layer.mlp]

    layer_results = {
        'layer_idx': layer_idx,
        'modeltype': modeltype,
        'num_experts': ori_expert_num,
        'num_blocks': num_blocks,
        'experts': {},
    }

    # Only process expert 0 for plotting (since we only plot expert 0)
    expert_idx = 0
    expert = experts[expert_idx] if ori_expert_num > 1 else experts[0]
    print_timestamp(f"Processing expert {expert_idx} (only, for plotting)...")
    expert_results = verify_one_expert_enhanced(
        expert, expert_idx, layer_idx, modeltype,
        bit_width=bit_width, group_size=group_size, num_blocks=num_blocks,
        seed=seed, skip_block=skip_block
    )
    layer_results['experts'][str(expert_idx)] = expert_results

    print(f"  Layer {layer_idx}: done")
    time.sleep(5.0)

    # Plot immediately after layer completes (before moving to next layer)
    if not skip_plot:
        try:
            print_timestamp(f"Plotting layer {layer_idx} immediately after processing...")
            plot_comprehensive_analysis(layer_results, layer_idx, short_id, plot_dir,
                                        bit_width=bit_width, group_size=group_size, num_blocks=num_blocks)
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            print(f"Warning: Plotting failed: {e}")
            print(f"Stack trace:\n{traceback.format_exc()}")

    return layer_results


@timeit
def plot_comprehensive_analysis(
    layer_result: dict,
    layer_idx: int,
    short_id: str,
    plot_dir: str = PLOT_DIR,
    bit_width: int = 4,
    group_size: int | None = 128,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
) -> None:
    # Plot main with sorted blocks
    _plot_2x3_with_dp(layer_result, layer_idx, short_id, plot_dir, bit_width, group_size, num_blocks)


def _plot_2x3_with_dp(
    layer_result: dict,
    layer_idx: int,
    short_id: str,
    plot_dir: str,
    bit_width: int,
    group_size: int | None,
    num_blocks: int,
) -> None:
    """
    Plot 1x3 layout with DP block analysis.

    Only row: Full matrix vs Blocked (DP) with individual blocks
    """
    apply_paper_style()
    os.makedirs(plot_dir, exist_ok=True)
    print(f"  [Plot] Saving DP plot to: {plot_dir}")

    num_experts = layer_result['num_experts']
    proj_names = ['up_proj', 'gate_proj', 'down_proj']

    if '0' not in layer_result['experts']:
        return
    expert0 = layer_result['experts']['0']

    # Get matrix dimensions from first projection
    d = None
    block_size = None
    for proj_name in proj_names:
        if proj_name in expert0 and 'full_matrix' in expert0[proj_name]:
            d = len(expert0[proj_name]['full_matrix']['diag_E_raw'])
            if d is not None and 'block_with_dp' in expert0[proj_name]:
                gdata = expert0[proj_name]['block_with_dp']
                if 'block_results' in gdata and gdata['block_results']:
                    block_size = gdata['block_results'][0]['block_size']
                    break
            if d is not None:
                break

    # Create figure: 1 row, 3 columns
    fig = plt.figure(figsize=(22, 6))
    gs = fig.add_gridspec(1, 3, hspace=0.35, wspace=0.28)

    # Main title
    fig.suptitle(f'Diagonal Dominance Density | {short_id} | Layer: {layer_idx} | Expert: 0 ', fontsize=16, fontweight='bold', y=0.98)

    # Fixed colors
    color_full = plt.cm.tab10(0)  # Blue for full matrix
    block_colors = [plt.cm.tab10(i) for i in range(1, 9)]  # Different colors for each block

    info_text = f'Quantization: {bit_width} bits target; '
    info_text += f'Quant group size: {group_size if group_size is not None else "None"}; '
    if d is not None:
        info_text += f'Full matrix size: {d}; '
    info_text += f'Num sorted blocks: {num_blocks}; '
    if block_size is not None:
        info_text += f'Sorted block size: {block_size}'

    fig.text(0.5, 0.92, info_text, transform=fig.transFigure,
             fontsize=12, verticalalignment='top', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # --- Only row: Full vs Sorted Blocks with individual blocks ---
    for col_idx, proj_name in enumerate(proj_names):
        if proj_name not in expert0:
            continue

        proj_data = expert0[proj_name]
        ax = fig.add_subplot(gs[0, col_idx])

        has_data = False

        # Plot Full matrix first (background)
        if 'full_matrix' in proj_data:
            full_res = proj_data['full_matrix']
            full_diag = full_res['diag_E_raw']
            ax.hist(full_diag, bins=50, alpha=0.4, label='Full Matrix (count)', density=True, color=color_full)
            has_data = True

        # Plot block 0 (most important) and block 7 (least important)
        if 'block_with_dp' in proj_data:
            dp_data = proj_data['block_with_dp']
            block_results = dp_data['block_results']
            best_scheme = dp_data['best_scheme']

            # Block 0 (most sensitive)
            if len(block_results) > 0:
                br_0 = block_results[0]
                block_diag_0 = br_0['diag_E_raw']
                block_bit_0 = br_0['allocated_bit']
                label_0 = f'Block 0 (most sensitive, {block_bit_0} bits, normalized)'
                ax.hist(block_diag_0, bins=50, alpha=0.4, label=label_0, density=True, color=block_colors[0])

            # Block 7 (least sensitive)
            if len(block_results) > 7:
                br_7 = block_results[7]
                block_diag_7 = br_7['diag_E_raw']
                block_bit_7 = br_7['allocated_bit']
                label_7 = f'Block 7 (least sensitive, {block_bit_7} bits, normalized)'
                ax.hist(block_diag_7, bins=50, alpha=0.4, label=label_7, density=True, color=block_colors[7])

            has_data = True

        if has_data:
            ax.set_xlabel('Diagonal Element Value', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'Random QR - Full vs Sorted Blocks - {proj_name}\nFixed Scheme: {best_scheme}', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3)
        else:
            # If no data, just plot note
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Random QR - {proj_name}', fontsize=13, fontweight='bold')

    # Adjust layout to leave space for titles
    fig.subplots_adjust(top=0.8)

    comp_path_png = os.path.join(plot_dir, f'oscar_together__{short_id}_L{layer_idx}_{bit_width}bit_sorted_blocks.png')
    comp_path_pdf = os.path.join(plot_dir, f'oscar_together__{short_id}_L{layer_idx}_{bit_width}bit_sorted_blocks.pdf')
    plt.savefig(comp_path_png, dpi=150, bbox_inches='tight')
    plt.savefig(comp_path_pdf, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved sorted blocks plot: {comp_path_png}")
    print(f"    Saved sorted blocks plot (PDF): {comp_path_pdf}")

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Verify diagonal dominance with DP sorted blocking')
    parser.add_argument('--model', required=True, help='Model path or short id')
    parser.add_argument('--dataset', default='wikitext2', help='Dataset for calibration')
    parser.add_argument('--nsamples', type=int, default=64, help='Number of calibration samples')
    parser.add_argument('--seqlen', type=int, default=2048, help='Sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--bits', type=int, default=2, help='Quantization bit width')
    parser.add_argument('--group_size', type=int, default=128, help='TurboQuant group size')
    parser.add_argument('--num_blocks', type=int, default=DEFAULT_NUM_BLOCKS, help='Number of blocks for blocked analysis')
    parser.add_argument('--layers', type=str, default=None, help='Comma-separated layer indices')
    parser.add_argument('--plot_dir', default=PLOT_DIR, help='Directory to save plots')
    parser.add_argument('--skip_plot', action='store_true', help='Skip plotting during computation')
    parser.add_argument('--skip_block', action='store_true', help='Skip DP-style blocked analysis')

    args = parser.parse_args()

    try:
        short_id = resolve_model_id(args.model)
    except Exception:
        short_id = args.model

    model_path = resolve_model_path(args.model)
    print_timestamp(f"Loading model from: {model_path}")
    model, tokenizer = load_model(model_path)
    model.seqlen = args.seqlen
    short_id = resolve_model_id(getattr(model, 'model_id', model_path))
    print(f"Model short id: {short_id}")

    os.makedirs(args.plot_dir, exist_ok=True)

    dataloader, _ = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed,
        seqlen=args.seqlen, tokenizer=tokenizer, bsz=1
    )

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    dtype = next(iter(model.parameters())).dtype

    inps = torch.zeros((args.nsamples, 1, args.seqlen, model.config.hidden_size),
                      dtype=dtype, device='cpu')
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            raise ValueError
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

    model.model.embed_tokens = model.model.embed_tokens.to(DEV)

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(DEV))
        except ValueError:
            pass
        if cache['i'] >= args.nsamples:
            break
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    inps = inps.squeeze(1)
    modeltype = model.config.model_type
    print(f"Model type: {modeltype}")

    if args.layers is not None:
        layer_indices = [int(idx.strip()) for idx in args.layers.split(',')]
    else:
        layer_indices = list(range(len(layers)))

    all_results = []

    for layer_idx in layer_indices:
        print_timestamp(f"Processing layer {layer_idx}...")

        if layer_idx < 0 or layer_idx >= len(layers):
            print(f"Warning: Layer {layer_idx} out of range, skipping")
            continue

        layer = layers[layer_idx]

        # Get layer's device
        device = next(layer.parameters()).device

        new_inps = torch.zeros_like(inps)

        for sample_idx in range(inps.shape[0]):
            inp = inps[sample_idx:sample_idx+1].to(device)

            residual = inp
            hidden_states_inorm = layer.input_layernorm(inp)

            # Move cache items to the same device
            attention_mask = cache['attention_mask']
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            position_ids = cache['position_ids']
            if position_ids is not None:
                position_ids = position_ids.to(device)

            if modeltype in ('olmoe', 'llama', 'qwen3', 'qwen3_moe'):
                attn_out = layer.self_attn(
                    hidden_states=hidden_states_inorm,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=cache.get('position_embeddings')
                )[0]
            else:
                attn_out = layer.self_attn(
                    hidden_states=hidden_states_inorm,
                    attention_mask=attention_mask,
                    position_ids=position_ids
                )[0]

            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)

            new_inps[sample_idx] = hidden_states.squeeze(0).cpu()

            mlp_out = layer.mlp(hidden_states)
            mlp_out = mlp_out[0] if isinstance(mlp_out, tuple) else mlp_out
            inps[sample_idx] = (mlp_out + residual).squeeze(0).cpu()

            del inp, residual, hidden_states_inorm, attn_out, hidden_states, mlp_out

        # Verify layer with immediate plotting inside the function
        layer_result = verify_layer(
            layer, layer_idx, modeltype,
            short_id=short_id, plot_dir=args.plot_dir, skip_plot=args.skip_plot,
            bit_width=args.bits, group_size=args.group_size, num_blocks=args.num_blocks,
            seed=args.seed, skip_block=args.skip_block
        )

        all_results.append(layer_result)

        # Clean up
        del new_inps
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache

    print_timestamp("Done!")


if __name__ == '__main__':
    main()
