"""
Bit loss extrapolation and visualization utilities.

This module contains functions for:
- Extrapolating 0bit loss using log-quadratic fit
- Visualizing neuron rates across bit widths with fit curves
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple

# Add parent directory to path to import dp_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dp_utils import extrapolate_0bit_loss_fix, extrapolate_0bit_loss, compute_r_squared_for_rates

INTERMEDIATE_RESULT_DIR = "intermediate_result"


def plot_lowest_r2_neurons(
    model_id: str,
    layer_idx: int,
    expert_idx: int,
    quant_type: str,
    neuron_r2: List[Tuple[int, float, np.ndarray]],
    outlier_bits: Set[int] = None,
    save_dir: str = None,
    use_pdf: bool = False,
):
    """Plot the lowest R² neurons with their fit curves.
    Args:
        model_id: Model identifier
        layer_idx: Layer index
        expert_idx: Expert index
        quant_type: Quantization type
        neuron_r2: List of (neuron_idx, r2_val, loss_array)
        outlier_bits: Set of bit widths
        save_dir: Directory to save plot
        use_pdf: Save as PDF instead of PNG
    """
    if outlier_bits is None:
        outlier_bits = {1, 2, 3, 4}
    bits_sorted = sorted(outlier_bits)
    b_array = np.array(bits_sorted, dtype=float)

    # Load full data to get all bits for these neurons
    rank_mode = 'turboquant_innerproduct' if quant_type == 'turboquant' else 'gptq_quant_outlier'
    cache_dir = os.path.join(INTERMEDIATE_RESULT_DIR, f"quant_outlier_{quant_type}", rank_mode, model_id)

    # Load all bit data for these neurons
    full_rates = {}
    for x in bits_sorted:
        cache_path = os.path.join(cache_dir, f"{model_id}_L{layer_idx}_b{x}.pt")
        if os.path.exists(cache_path):
            try:
                import torch
                cached_data = torch.load(cache_path, map_location='cpu')
                full_rates[x] = cached_data[expert_idx].detach().cpu().float().numpy()
            except Exception as e:
                print(f"Failed to load {cache_path}: {e}")

    # Create figure
    fig, axes = plt.subplots(1, len(neuron_r2), figsize=(5 * len(neuron_r2), 4))
    if len(neuron_r2) == 1:
        axes = [axes]

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']  # red, blue, green, purple, orange

    for ax_idx, (neuron_idx, r2_val, loss_array) in enumerate(neuron_r2):
        ax = axes[ax_idx]
        color = colors[ax_idx % len(colors)]

        # Plot actual data points
        actual_bits = []
        actual_losses = []
        for b in bits_sorted:
            if b in full_rates and neuron_idx < len(full_rates[b]):
                val = full_rates[b][neuron_idx]
                actual_bits.append(b)
                actual_losses.append(val)

        if actual_bits:
            ax.scatter(actual_bits, actual_losses, color=color, s=100, alpha=0.9, zorder=5, label='Data')

            # Try to compute and plot fit curve
            loss_arr = np.array(actual_losses)
            positive_mask = loss_arr > 0
            if positive_mask.sum() >= 3:
                try:
                    fit_bits = b_array[positive_mask]
                    fit_loss = loss_arr[positive_mask]
                    log_loss = np.log(fit_loss)
                    p, q, r = np.polyfit(fit_bits, log_loss, deg=2)

                    # Plot smooth fit curve
                    x_smooth = np.linspace(min(fit_bits) - 0.2, max(fit_bits) + 0.2, 100)
                    y_smooth_log = p * x_smooth**2 + q * x_smooth + r
                    y_smooth = np.exp(y_smooth_log)
                    ax.plot(x_smooth, y_smooth, color=color, linewidth=2, alpha=0.8, zorder=3, label='Fit')

                    # Also plot interpolated points
                    y_pred_log = p * fit_bits**2 + q * fit_bits + r
                    y_pred = np.exp(y_pred_log)
                    ax.scatter(fit_bits, y_pred, color=color, marker='x', s=80, alpha=0.7, zorder=4, label='Predicted')
                except Exception as e:
                    print(f"Failed to fit neuron {neuron_idx}: {e}")

            # Also try to extrapolate 0bit
            if len(actual_bits) >= 2:
                try:
                    # Create a temporary rates dict for extrapolation
                    temp_rates = {}
                    for b, val in zip(actual_bits, actual_losses):
                        temp_rates[b] = [np.array([val])]

                    rates_0_list = extrapolate_0bit_loss(temp_rates, quant_type=quant_type, save_plots=False)
                    l0 = rates_0_list[0][0]
                    ax.scatter([0], [l0], color=color, marker='*', s=150, alpha=0.9, zorder=6, label='0-bit')
                except Exception as e:
                    pass

        ax.set_xlabel('Bit Width', fontsize=10)
        ax.set_ylabel('Loss (log scale)', fontsize=10)
        ax.set_title(f'Neuron {neuron_idx}\nR²={r2_val:.4f}', fontsize=11, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, zorder=1)
        all_bits_with_0 = [0] + bits_sorted if len(actual_bits) >= 2 else bits_sorted
        ax.set_xlim(-0.2, max(all_bits_with_0) + 0.2)
        ax.set_xticks(all_bits_with_0)
        ax.legend(fontsize=8, loc='upper right')

    plt.suptitle(f'{model_id} Layer {layer_idx} Expert {expert_idx} ({quant_type}) - Lowest 5 R² Neurons',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    # Save plot
    if save_dir is None:
        save_dir = 'plot/neuron_rates_fit'
    os.makedirs(save_dir, exist_ok=True)
    ext = 'pdf' if use_pdf else 'png'
    save_path = os.path.join(save_dir, f'{model_id}_L{layer_idx}_exp{expert_idx}_{quant_type}_lowest_r2.{ext}')
    if use_pdf:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Debug plot saved to {save_path}")
    plt.close()


def compute_r_squared(x: np.ndarray, y: np.ndarray, p: float, q: float, r: float) -> float:
    """Compute coefficient of determination (R²) for log-quadratic fit.

    R² = 1 - (SSR / SST)
    where:
        SSR = sum((y_true - y_pred)^2)  (sum of squared residuals)
        SST = sum((y_true - y_mean)^2)  (total sum of squares)

    Args:
        x: bit values
        y: log(loss) values
        p, q, r: polynomial coefficients (y = p*x² + q*x + r)

    Returns:
        R² value, or NaN if computation fails
    """
    if len(y) < 2:
        return np.nan

    y_pred = p * x**2 + q * x + r
    y_mean = np.mean(y)

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)

    if ss_tot == 0:
        # All y values are the same
        return 1.0 if ss_res == 0 else 0.0

    return 1.0 - ss_res / ss_tot


def plot_neuron_rates_with_fit(
    model_id: str,
    layer_idx: int,
    expert_idx: int = 0,
    p: int = 20,
    n_show_neurons: int = 30,
    outlier_bits: Set[int] = None,
    use_0bit: bool = True,
    save_dir: str = None,
    use_pdf: bool = False,
):
    """
    Visualize neuron rates across different bit widths with fit curves.
    Three subplots: fit demo, TurboQuant, GPTQ.

    Args:
        model_id: Model identifier
        layer_idx: Layer index
        expert_idx: Expert index
        p: Number of neurons to plot
        outlier_bits: Set of bit widths to load, defaults to {1,2,3,4}
        use_0bit: Whether to extrapolate and include 0bit
        save_dir: Directory to save plot
    """
    if outlier_bits is None:
        outlier_bits = {1, 2, 3, 4}

    print(f"Plotting neuron rates with fit: model={model_id}, layer={layer_idx}, expert={expert_idx}, p={p}")

    # Load data for both quant types
    quants = [
        ('turboquant', 'turboquant_innerproduct'),
        ('gptq', 'gptq_quant_outlier'),
    ]

    all_data = {}
    for quant_type, rank_mode in quants:
        cache_dir = os.path.join(INTERMEDIATE_RESULT_DIR, f"quant_outlier_{quant_type}", rank_mode, model_id)
        rates = {}

        # Load data for each bit
        for x in outlier_bits:
            cache_path = os.path.join(cache_dir, f"{model_id}_L{layer_idx}_b{x}.pt")
            if os.path.exists(cache_path):
                try:
                    import torch
                    cached_data = torch.load(cache_path, map_location='cpu')
                    print(f"Loading cached data for {quant_type}: layer {layer_idx}, wbits={x}")
                    rates[x] = [cached_data[expert_idx]]
                except Exception as e:
                    print(f"Failed to load cached data for {quant_type} bit {x}: {e}")

        if not rates:
            print(f"No data loaded for {quant_type}!")
            all_data[quant_type] = None
            continue

        # Extrapolate 0bit if needed
        rates_0 = None
        fit_params = None
        if use_0bit and len(rates) >= 2:
            rates_copy = {k: v for k, v in rates.items()}
            # Note: dp_utils version doesn't return fit_params, so we need to handle this
            # For now, just use the extrapolation without the fit visualization
            rates_0_list = extrapolate_0bit_loss(rates_copy, quant_type=quant_type, save_plots=False)
            rates[0] = [rates_0_list[0]]
            rates_0 = rates_0_list[0]

        all_data[quant_type] = {
            'rates': rates,
            'rates_0': rates_0,
            'fit_params': None,
        }

    # Create figure with 3 subplots - adjust for colorbars
    fig = plt.figure(figsize=(20, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])

    # ------------------------------------------------------------------------
    # Subplot 1: Fit demo - show a few example neurons with their fit curves
    # ------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])

    # Use TurboQuant for the fit demo
    demo_quant = 'turboquant'

    if all_data[demo_quant] is not None:
        data = all_data[demo_quant]
        rates = data['rates']
        bits_sorted = sorted([b for b in rates.keys() if b != 0])
        n_neurons = int(len(rates[bits_sorted[0]][0]))

        # Pick a few interesting neurons to show (e.g., highest loss at 4-bit)
        highest_bit = max(bits_sorted)
        neuron_losses = [(i, rates[highest_bit][0][i].item() if hasattr(rates[highest_bit][0][i], 'item') else rates[highest_bit][0][i])
                         for i in range(n_neurons)]
        neuron_losses.sort(key=lambda x: x[1], reverse=True)

        # Show top 3 neurons
        demo_neurons = [idx for idx, _ in neuron_losses[:3]]
        colors = ['#e74c3c', '#3498db', '#2ecc71']  # red, blue, green

        for idx, neuron_idx in enumerate(demo_neurons):
            color = colors[idx % len(colors)]

            # Plot actual data points
            actual_bits = []
            actual_losses = []
            for b in bits_sorted:
                val = rates[b][0][neuron_idx]
                if hasattr(val, 'item'):
                    val = val.item()
                actual_bits.append(b)
                actual_losses.append(val)

            ax1.scatter(actual_bits, actual_losses, color=color, s=80, alpha=0.9,
                       label=f'Neuron {neuron_idx} (data)')

            # Plot 0-bit extrapolation
            if 0 in rates:
                l0 = rates[0][0][neuron_idx]
                if hasattr(l0, 'item'):
                    l0 = l0.item()
                ax1.scatter([0], [l0], color=color, marker='*', s=150, alpha=0.9,
                           label=f'Neuron {neuron_idx} (0-bit)')

        ax1.set_xlabel('Bit Width', fontsize=11)
        ax1.set_ylabel('Loss (log scale)', fontsize=11)
        ax1.set_title(f'Log-Quad Fit Demo\n{model_id} Layer {layer_idx} Expert {expert_idx}', fontsize=12, fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, loc='upper right')
        ax1.set_xlim(-0.2, max(bits_sorted) + 0.7)

    # ------------------------------------------------------------------------
    # Subplot 2: TurboQuant
    # ------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])

    if all_data['turboquant'] is not None:
        data = all_data['turboquant']
        rates = data['rates']
        bits_sorted = sorted([b for b in rates.keys() if b != 0])  # Exclude 0 for plotting
        if 0 in rates:
            bits_sorted_with_0 = [0] + bits_sorted
        else:
            bits_sorted_with_0 = bits_sorted
        n_neurons = int(len(rates[bits_sorted[0]][0]))

        print(f"TurboQuant: plotting {n_neurons} neurons, bits={bits_sorted_with_0}")

        # Use a colormap
        cmap = plt.get_cmap('turbo', n_show_neurons)

        all_losses = []
        legend_handles = []
        legend_labels = []
        for color_idx, neuron_idx in enumerate(range(min(n_show_neurons, n_neurons))):
            color = cmap(color_idx)

            # Plot 1-4 bit points - higher transparency
            rate_values_no0 = []
            bit_values_no0 = []
            for b in bits_sorted:
                val = rates[b][0][neuron_idx]
                if hasattr(val, 'item'):
                    val = val.item()
                bit_values_no0.append(b)
                rate_values_no0.append(val)
                all_losses.append(val)

            # Plot 1-4 bit with circles, higher transparency
            scatter = ax2.scatter(bit_values_no0, rate_values_no0, color=color, s=60, alpha=0.4, zorder=3)

            # Plot 0-bit with star if available - original alpha
            if 0 in rates:
                val_0 = rates[0][0][neuron_idx]
                if hasattr(val_0, 'item'):
                    val_0 = val_0.item()
                ax2.scatter([0], [val_0], color=color, marker='*', s=80, alpha=0.7, zorder=4)
                all_losses.append(val_0)

            # Collect handles for legend
            legend_handles.append(scatter)
            legend_labels.append(f'N{neuron_idx}')

        # Set y-axis limits based on data
        if all_losses:
            all_losses = np.array(all_losses)
            valid_losses = all_losses[all_losses > 0]
            if len(valid_losses) > 0:
                y_min = np.min(valid_losses) * 0.5
                y_max = np.max(valid_losses) * 2.0
                ax2.set_ylim(y_min, y_max)

        ax2.set_xlabel('Bit Width', fontsize=11)
        ax2.set_ylabel('Loss (log scale)', fontsize=11)
        ax2.set_title(f'TurboQuant\n{model_id} Layer {layer_idx} Expert {expert_idx}', fontsize=12, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3, zorder=1)
        ax2.set_xlim(-0.2, max(bits_sorted_with_0) + 0.2)
        ax2.set_xticks(bits_sorted_with_0)
        ax2.legend(legend_handles, legend_labels, fontsize=7, loc='upper right', ncol=2)

    # ------------------------------------------------------------------------
    # Subplot 3: GPTQ
    # ------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])

    if all_data['gptq'] is not None:
        data = all_data['gptq']
        rates = data['rates']
        bits_sorted = sorted([b for b in rates.keys() if b != 0])  # Exclude 0 for plotting
        if 0 in rates:
            bits_sorted_with_0 = [0] + bits_sorted
        else:
            bits_sorted_with_0 = bits_sorted
        n_neurons = int(len(rates[bits_sorted[0]][0]))

        print(f"GPTQ: plotting {n_neurons} neurons, bits={bits_sorted_with_0}")

        # Use a colormap
        cmap = plt.get_cmap('turbo', n_show_neurons)

        all_losses = []
        legend_handles = []
        legend_labels = []
        for color_idx, neuron_idx in enumerate(range(min(n_show_neurons, n_neurons))):
            color = cmap(color_idx)

            # Plot 1-4 bit points - higher transparency
            rate_values_no0 = []
            bit_values_no0 = []
            for b in bits_sorted:
                val = rates[b][0][neuron_idx]
                if hasattr(val, 'item'):
                    val = val.item()
                bit_values_no0.append(b)
                rate_values_no0.append(val)
                all_losses.append(val)

            # Plot 1-4 bit with circles, higher transparency
            scatter = ax3.scatter(bit_values_no0, rate_values_no0, color=color, s=60, alpha=0.4, zorder=3)

            # Plot 0-bit with star if available - original alpha
            if 0 in rates:
                val_0 = rates[0][0][neuron_idx]
                if hasattr(val_0, 'item'):
                    val_0 = val_0.item()
                ax3.scatter([0], [val_0], color=color, marker='*', s=80, alpha=0.7, zorder=4)
                all_losses.append(val_0)

            # Collect handles for legend
            legend_handles.append(scatter)
            legend_labels.append(f'N{neuron_idx}')

        # Set y-axis limits based on data
        if all_losses:
            all_losses = np.array(all_losses)
            valid_losses = all_losses[all_losses > 0]
            if len(valid_losses) > 0:
                y_min = np.min(valid_losses) * 0.5
                y_max = np.max(valid_losses) * 2.0
                ax3.set_ylim(y_min, y_max)

        ax3.set_xlabel('Bit Width', fontsize=11)
        ax3.set_ylabel('Loss (log scale)', fontsize=11)
        ax3.set_title(f'GPTQ\n{model_id} Layer {layer_idx} Expert {expert_idx}', fontsize=12, fontweight='bold')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3, zorder=1)
        ax3.set_xlim(-0.2, max(bits_sorted_with_0) + 0.2)
        ax3.set_xticks(bits_sorted_with_0)
        ax3.legend(legend_handles, legend_labels, fontsize=7, loc='upper right', ncol=2)

    plt.tight_layout()

    # Save plot
    if save_dir is None:
        save_dir = 'plot/neuron_rates_fit'
    os.makedirs(save_dir, exist_ok=True)
    ext = 'pdf' if use_pdf else 'png'
    save_path = os.path.join(save_dir, f'{model_id}_L{layer_idx}_exp{expert_idx}_fit_comparison.{ext}')
    if use_pdf:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def test_read_rates_from_file():
    outlier_bits = {1, 2, 3, 4}
    print(f"simulate quant outlier_bits {outlier_bits}")

    model_id = "deepseek-v1-moe-16b"
    layer_idx = 1
    quant_type = "turboquant"
    cache_dir = os.path.join(INTERMEDIATE_RESULT_DIR, f"quant_outlier_{quant_type}", "turboquant_innerproduct", model_id)

    p = 20
    expert_idx = 0
    rates = {}
    for x in outlier_bits:
        cache_path = os.path.join(cache_dir, f"{model_id}_L{layer_idx}_b{x}.pt")
        if os.path.exists(cache_path):
            try:
                import torch
                cached_data = torch.load(cache_path, map_location='cpu')
                print(f"Loading cached quant outlier data for layer {layer_idx}, wbits={x}")
                rates[x] = [cached_data[expert_idx]]
            except Exception as e:
                print(f"Failed to load cached data: {e}")

    rates[0] = extrapolate_0bit_loss(rates, quant_type=quant_type, save_plots=True)
    for i in range(p):
        print(i, end=', ')
        print(f"{rates[4][expert_idx][i].item():.4f}", end=', ')
        print(f"{rates[3][expert_idx][i].item():.4f}", end=', ')
        print(f"{rates[2][expert_idx][i].item():.4f}", end=', ')
        print(f"{rates[1][expert_idx][i].item():.4f}", end=', ')
        print(f"{rates[0][expert_idx][i].item():.4f}", end=', ')
        print()


def get_model_layer_stats(
    model_id: str,
    expert_idx: int = None,
    outlier_bits: Set[int] = None,
    debug_layers: List[int] = None,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Get R² mean stats per layer for a single model (both quant types)."""
    if outlier_bits is None:
        outlier_bits = {1, 2, 3, 4}

    # Discover all available layers
    all_layers = set()
    # prefix = '_nopr8fix'
    prefix = ''
    quants_for_discovery = [
        ('turboquant', f'turboquant_innerproduct{prefix}'),
        ('gptq', f'gptq_quant_outlier{prefix}'),
    ]
    found_any_cache = False
    for quant_type, rank_mode in quants_for_discovery:
        cache_dir = os.path.join(INTERMEDIATE_RESULT_DIR, f"quant_outlier_{quant_type}", rank_mode, model_id)
        if os.path.exists(cache_dir):
            found_any_cache = True
            for filename in os.listdir(cache_dir):
                if filename.endswith('_b1.pt') and model_id in filename:
                    parts = filename.split('_L')
                    if len(parts) > 1:
                        layer_part = parts[1].split('_b')[0]
                        try:
                            all_layers.add(int(layer_part))
                        except ValueError:
                            pass
        else:
            print(f"[WARNING] Cache directory not found for {quant_type}: {cache_dir}")

    layer_indices = sorted(all_layers)
    if not layer_indices:
        if not found_any_cache:
            print(f"[ERROR] No cache directories found for model {model_id} at all!")
        else:
            print(f"[ERROR] No layers found for model {model_id} in cache directories!")
        return {'turboquant': {}, 'gptq': {}}

    mode_str = f"ALL EXPERTS" if expert_idx is None else f"EXPERT {expert_idx}"
    print(f"Processing model {model_id} ({mode_str}), layers: {layer_indices}")

    result = {'turboquant': {}, 'gptq': {}}

    for quant_type, rank_mode in quants_for_discovery:
        cache_dir = os.path.join(INTERMEDIATE_RESULT_DIR, f"quant_outlier_{quant_type}", rank_mode, model_id)
        if not os.path.exists(cache_dir):
            print(f"[WARNING] {quant_type} cache directory not found: {cache_dir}")
            continue

        for lidx in layer_indices:
            rates = {}
            missing_bits = []
            for x in outlier_bits:
                cache_path = os.path.join(cache_dir, f"{model_id}_L{lidx}_b{x}.pt")
                if os.path.exists(cache_path):
                    try:
                        import torch
                        cached_data = torch.load(cache_path, map_location='cpu')
                        if expert_idx is None:
                            # Load ALL experts (default mode A)
                            rates[x] = cached_data
                        else:
                            # Load single expert only
                            expert_data = cached_data[expert_idx].detach().cpu().float().numpy()
                            rates[x] = expert_data
                    except Exception as e:
                        print(f"[WARNING] Failed to load {cache_path}: {e}")
                        missing_bits.append(x)
                else:
                    missing_bits.append(x)

            if not rates or len(rates) < 2:
                print(f"[WARNING] {quant_type} layer {lidx}: not enough data (only bits {sorted(rates.keys())} available, missing bits {missing_bits})")
                continue

            bits_sorted = sorted(rates.keys())

            # ========== NEW WAY (from dp_utils.py) ==========
            if expert_idx is None:
                # Default mode A: already all experts in rates[x], pass directly
                rates_for_dp_utils = rates
            else:
                # Single expert mode: wrap in list
                rates_for_dp_utils = {}
                for b in bits_sorted:
                    rates_for_dp_utils[b] = [rates[b]]

            new_stats = compute_r_squared_for_rates(rates_for_dp_utils)
            new_mean = new_stats['mean']
            new_median = new_stats['median']

            if not np.isnan(new_mean):
                result[quant_type][lidx] = {
                    'mean': new_mean,
                    'median': new_median
                }
                print(f"  {quant_type} layer {lidx}: mean R² = {new_mean:.4f}, median = {new_median:.4f}")
            else:
                print(f"  {quant_type} layer {lidx}: no valid data")

                # Debug mode temporarily disabled (old code removed)
                # if debug_layers and lidx in debug_layers:
                #     pass
    return result


def analyze_multi_model_r2(
    model_ids: List[str] = None,
    expert_idx: int = None,
    outlier_bits: Set[int] = None,
    save_dir: str = None,
    use_pdf: bool = False,
    debug_layers: List[int] = None,
):
    """Analyze R² for multiple models in a single row plot.

    Each subplot is one model, with two bars per layer: GPTQ (orange) and TurboQuant (blue).

    Args:
        model_ids: List of model identifiers (up to 5). If None, use all known models.
        expert_idx: Expert index
        outlier_bits: Set of bit widths to load
        save_dir: Directory to save plot
        use_pdf: Save as PDF instead of PNG
    """
    from viz._cache_io import KNOWN_MODELS

    # Use all known models if not provided
    use_all_models = model_ids is None or not model_ids
    if use_all_models:
        model_ids = sorted(KNOWN_MODELS.keys())
        print(f"Using all known models: {model_ids}")

    if len(model_ids) > 5:
        print(f"Warning: only first 5 models will be plotted (got {len(model_ids)})")
        model_ids = model_ids[:5]

    # Get stats for all models
    all_model_stats = []
    for model_id in model_ids:
        stats = get_model_layer_stats(model_id, expert_idx, outlier_bits, debug_layers)
        all_model_stats.append((model_id, stats))

    # Create figure: 1 row, N columns
    n_models = len(all_model_stats)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 3))
    if n_models == 1:
        axes = [axes]

    colors = {'turboquant': '#3498db', 'gptq': '#e67e22'}
    labels = {'turboquant': 'TQ', 'gptq': 'GPTQ'}

    for ax_idx, (model_id, stats) in enumerate(all_model_stats):
        ax = axes[ax_idx]

        # Get union of all layers for this model
        all_layers = set()
        for qt in ['turboquant', 'gptq']:
            all_layers.update(stats[qt].keys())
        layer_indices = sorted(all_layers)

        if not layer_indices:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(model_id)
            continue

        x = np.arange(len(layer_indices))
        width = 0.35

        # Plot bars (median) and mean markers for each quant type
        for qt_idx, quant_type in enumerate(['turboquant', 'gptq']):
            means = []
            medians = []
            for lidx in layer_indices:
                layer_stat = stats[quant_type].get(lidx)
                if layer_stat is not None:
                    means.append(layer_stat['mean'])
                    medians.append(layer_stat['median'])
                else:
                    means.append(np.nan)
                    medians.append(np.nan)

            offset = -width/2 if qt_idx == 0 else width/2

            # Plot median as bars
            label = f'{labels[quant_type]} (median)' if ax_idx == 0 else ""
            ax.bar(x + offset, medians, width, label=label,
                   color=colors[quant_type], alpha=0.8)

            # Plot mean as horizontal lines on the bars
            for xi, (mean_val, median_val) in enumerate(zip(means, medians)):
                if not np.isnan(mean_val) and not np.isnan(median_val):
                    line_x = xi + offset
                    # Draw a horizontal line for mean
                    ax.plot([line_x - width/3, line_x + width/3],
                            [mean_val, mean_val],
                            color='white', linewidth=2.5, solid_capstyle='butt')
                    ax.plot([line_x - width/3, line_x + width/3],
                            [mean_val, mean_val],
                            color='black', linewidth=1.5, solid_capstyle='butt')

        # Add reference lines
        if expert_idx is None:
            # All experts mode: 0.98 and 0.99
            ax.axhline(0.98, color='#27ae60', linestyle='--', alpha=0.7, linewidth=1.5, label='R²=0.98' if ax_idx == 0 else "")
            ax.axhline(0.99, color='#c0392b', linestyle=':', alpha=0.7, linewidth=1.5, label='R²=0.99' if ax_idx == 0 else "")
        else:
            # Single expert mode: 0.95 and 0.99 (keep original)
            ax.axhline(0.95, color='#27ae60', linestyle='--', alpha=0.7, linewidth=1.5, label='R²=0.95' if ax_idx == 0 else "")
            ax.axhline(0.99, color='#c0392b', linestyle=':', alpha=0.7, linewidth=1.5, label='R²=0.99' if ax_idx == 0 else "")

        # Add dummy lines for legend to explain mean marker
        if ax_idx == 0:
            ax.plot([], [], color='black', linewidth=1.5, label='mean')

        # Collect all R² values to determine y-axis range
        all_r2 = []
        for quant_type in ['turboquant', 'gptq']:
            for lidx in layer_indices:
                val = stats[quant_type].get(lidx)
                if val is not None:
                    all_r2.append(val['mean'])
                    all_r2.append(val['median'])

        # Dynamic y-axis limits
        if expert_idx is None:
            y_min = 0.9
        else:
            y_min = 0.8

        # Formatting
        ax.set_xlabel('Layer Index', fontsize=10)
        ax.set_ylabel('R²', fontsize=10)
        ax.set_title(model_id, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        if len(layer_indices) > 20:
            # Sparser labels when many layers
            step = max(2, len(layer_indices) // 15)  # aim for ~15 labels
            labels = [str(l) if i % step == 0 else "" for i, l in enumerate(layer_indices)]
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        else:
            ax.set_xticklabels([str(l) for l in layer_indices], rotation=45, ha='right', fontsize=8)
        ax.set_ylim(y_min, 1.005)
        # Set y-axis ticks
        if expert_idx is None:
            ax.set_yticks(np.arange(0.9, 1.0, 0.02))
        else:
            ax.set_yticks(np.arange(0.85, 1.01, 0.05))
        ax.grid(True, alpha=0.3, axis='y')

        # Only show legend on first plot
        if ax_idx == 0:
            ax.legend(fontsize=8, loc='lower right')

    plt.tight_layout()

    # Save plot
    if save_dir is None:
        save_dir = 'plot/neuron_rates_fit'
    os.makedirs(save_dir, exist_ok=True)
    if use_all_models:
        model_str = 'all_models'
    else:
        model_str = '_'.join([m.replace('-', '_') for m in model_ids])
    # Add expert identifier to filename if specified
    if expert_idx is None:
        expert_str = '_all_experts'
    else:
        expert_str = f'_expert{expert_idx}'
    # Save both PNG and PDF by default
    save_path_png = os.path.join(save_dir, f'r2_comparison_{model_str}{expert_str}.png')
    save_path_pdf = os.path.join(save_dir, f'r2_comparison_{model_str}{expert_str}.pdf')
    plt.savefig(save_path_png, dpi=150, bbox_inches='tight')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    print(f"\nMulti-model R² plots saved to:")
    print(f"  PNG: {save_path_png}")
    print(f"  PDF: {save_path_pdf}")
    plt.close()


def main():
    """Command-line interface for bit loss fit visualizations."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Bit loss extrapolation and visualization tools"
    )
    parser.add_argument("--models", nargs="+",
                      help="Model identifiers (up to 5, for R² analysis, default: auto-discover all)")
    parser.add_argument("--model",
                      help="Single model identifier (for neuron rate plotting)")
    parser.add_argument("--layer", type=int, default=1,
                      help="Layer index (for neuron rate plotting)")
    parser.add_argument("--expert", type=int, default=-1,
                      help="Expert index (default: None = all experts)")
    parser.add_argument("--p", type=int, default=20,
                      help="Number of neurons to plot (unused now, kept for compatibility)")
    parser.add_argument("--n-show-neurons", type=int, default=20,
                      help="Number of top neurons to show in plots")
    parser.add_argument("--bits", nargs="+", type=int, default=[1, 2, 3, 4],
                      help="Bit widths to load")
    parser.add_argument("--no-0bit", action="store_true",
                      help="Don't extrapolate 0bit")
    parser.add_argument("--save-dir",
                      help="Directory to save plot")
    parser.add_argument("--pdf", action="store_true",
                      help="Save as PDF instead of PNG")
    parser.add_argument("--analyze-r2", action="store_true",
                      help="Analyze multi-model R² comparison instead of plotting neuron rates")
    parser.add_argument("--debug-layers", nargs="+", type=int,
                      help="Layers to print detailed debug info for")

    args = parser.parse_args()

    if args.analyze_r2:
        # 优先用 --models，如果没指定但 --model 指定了，用 --model
        target_model_ids = args.models
        if not target_model_ids and args.model:
            target_model_ids = [args.model]
        # Convert -1 to None (use all experts)
        expert_param = args.expert if args.expert != -1 else None
        analyze_multi_model_r2(
            model_ids=target_model_ids,  # None = auto-discover
            expert_idx=expert_param,
            outlier_bits=set(args.bits),
            save_dir=args.save_dir,
            use_pdf=args.pdf,
            debug_layers=args.debug_layers,
        )
    else:
        # Default to the new fit comparison plot
        model_to_plot = args.model or "deepseek-v1-moe-16b"
        # plot_neuron_rates_with_fit still expects a single expert index, keep default 0
        expert_param = args.expert if args.expert != -1 else 0
        plot_neuron_rates_with_fit(
            model_id=model_to_plot,
            layer_idx=args.layer,
            expert_idx=expert_param,
            p=args.p,
            n_show_neurons=args.n_show_neurons,
            outlier_bits=set(args.bits),
            use_0bit=not args.no_0bit,
            save_dir=args.save_dir,
            use_pdf=args.pdf,
        )


if __name__ == "__main__":
    main()
