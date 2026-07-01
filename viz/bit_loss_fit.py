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

INTERMEDIATE_RESULT_DIR = "intermediate_result"


def extrapolate_0bit_loss(rates: Dict[int, List[np.ndarray]], quant_type: str = "gptq", save_plots: bool = False) -> List[np.ndarray]:
    bits = sorted(rates.keys())
    if 0 in bits:
        bits.remove(0)
        rates.pop(0)
    # print(bits)
    assert len(bits) >= 2, "at least 2 bits are required for extrapolation of 0bit loss"

    x_max = max(bits) + 1.0

    n_experts = len(rates[bits[0]])
    for b in bits:
        assert len(rates[b]) == n_experts, f"bit {b} has inconsistent number of experts"

    L0 = []

    for expert_idx in range(n_experts):
        print(f"Processing extrapolate 0bit loss for expert {expert_idx}")

        expert_rates = {}
        n_neurons = None
        for b in bits:
            expert_rates[b] = rates[b][expert_idx].detach().cpu().float().numpy()
            if n_neurons is None:
                n_neurons = len(expert_rates[b])
            else:
                assert len(expert_rates[b]) == n_neurons, \
                    f"expert {expert_idx}, bit {b} has inconsistent loss array length"

        b_array = np.array(bits, dtype=float)
        expert_L0 = np.zeros(n_neurons, dtype=float)

        for i in range(n_neurons):
            loss_array = np.array([expert_rates[b][i] for b in bits])

            # --------------------- log quadratic fit ---------------------
            # function: log(loss) = p*b² + q*b + r
            # loss(b) = exp(p*b² + q*b + r)
            # -------------------------------------------------------------
            r2 = np.nan
            try:
                log_loss = np.log(loss_array)

                p, q, r = np.polyfit(b_array, log_loss, deg=2)
                r2 = compute_r_squared(b_array, log_loss, p, q, r)

                l0 = np.exp(r)

                l1 = expert_rates[1][i] if 1 in expert_rates else expert_rates[bits[0]][i]
                if l0 < l1:
                    l0 = l1 * 2.0

            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                l1 = expert_rates[1][i] if 1 in expert_rates else expert_rates[bits[0]][i]
                l0 = l1 * 2.0

            expert_L0[i] = l0

            if save_plots:
                print(p, q, r, f"R²={r2:.4f}")
                plt.figure(figsize=(7, 4))

                plt.scatter(bits, loss_array, color='red', s=10, label='Original loss (1,2,3,4...)')

                b_dense = np.linspace(0.001, max(bits), 100)
                y_dense = np.exp(p * b_dense ** 2 + q * b_dense + r)
                r2_label = f', R²={r2:.4f}' if not np.isnan(r2) else ''
                plt.plot(b_dense, y_dense, 'b-', label=f'Log-quad fit (exp(pb²+qb+r)){r2_label}')

                plt.scatter(0, l0, color='green', s=10, label=f'L0 = {l0:.2f}')
                plt.scatter(1, l1, color='orange', s=10, label=f'L1 = {l1:.2f}')

                plt.title(f'Expert {expert_idx} | Neuron {i} | L0={l0:.2f}, L1={l1:.2f}'
                         f'{f", R²={r2:.4f}" if not np.isnan(r2) else ""}')
                plt.xlabel('bit')
                plt.ylabel('loss')
                plt.yscale('log')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                os.makedirs(f'plot/{quant_type}_bit_loss_fit', exist_ok=True)
                plt.savefig(f'plot/{quant_type}_bit_loss_fit/exp2_expert_{expert_idx}_neuron_{i}.png', dpi=150)
                plt.close()

        L0.append(expert_L0)

    return L0


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

    return 1.0 - (ss_res / ss_tot)


def extrapolate_0bit_loss_fix(rates: Dict[int, List[np.ndarray]], quant_type: str = "gptq", save_plots: bool = False) -> Tuple[List[np.ndarray], List[Dict]]:
    bits = sorted(rates.keys())
    if 0 in bits:
        bits.remove(0)
        rates.pop(0)
    # print(bits)
    assert len(bits) >= 2, "at least 2 bits are required for extrapolation of 0bit loss"

    x_max = max(bits) + 1.0

    n_experts = len(rates[bits[0]])
    for b in bits:
        assert len(rates[b]) == n_experts, f"bit {b} has inconsistent number of experts"

    L0 = []
    fit_params = []  # Store fit parameters for each neuron

    for expert_idx in range(n_experts):
        print(f"Processing extrapolate 0bit loss for expert {expert_idx}")

        expert_rates = {}
        n_neurons = None
        for b in bits:
            expert_rates[b] = rates[b][expert_idx].detach().cpu().float().numpy()
            if n_neurons is None:
                n_neurons = len(expert_rates[b])
            else:
                assert len(expert_rates[b]) == n_neurons, \
                    f"expert {expert_idx}, bit {b} has inconsistent loss array length"

        b_array = np.array(bits, dtype=float)
        expert_L0 = np.zeros(n_neurons, dtype=float)
        expert_fit_params = []

        for i in range(n_neurons):
            loss_array = np.array([expert_rates[b][i] for b in bits])
            positive_mask = loss_array > 0

            if not np.any(positive_mask):
                expert_L0[i] = 0.0
                expert_fit_params.append({'p': np.nan, 'q': np.nan, 'r': np.nan, 'r2': np.nan, 'valid': False})
                continue

            reference_bit = bits[0]
            if 1 in expert_rates:
                reference_bit = 1

            reference_loss = expert_rates[reference_bit][i]
            fallback_l0 = max(reference_loss * 2.0, 1e-12)

            # --------------------- log quadratic fit ---------------------
            # function: log(loss) = p*b² + q*b + r
            # loss(b) = exp(p*b² + q*b + r)
            # -------------------------------------------------------------
            p = q = r = np.nan
            r2 = np.nan
            valid_fit = False
            try:
                if positive_mask.sum() >= 3:
                    fit_bits = b_array[positive_mask]
                    fit_loss = loss_array[positive_mask]
                    log_loss = np.log(fit_loss)

                    p, q, r = np.polyfit(fit_bits, log_loss, deg=2)
                    r2 = compute_r_squared(fit_bits, log_loss, p, q, r)
                    l0 = np.exp(r)
                    valid_fit = True
                else:
                    l0 = fallback_l0

                if reference_loss > 0:
                    if l0 < reference_loss:
                        l0 = fallback_l0

            except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError):
                l0 = fallback_l0

            expert_L0[i] = l0
            expert_fit_params.append({'p': p, 'q': q, 'r': r, 'r2': r2, 'valid': valid_fit, 'bits': bits, 'losses': loss_array})

            if save_plots:
                print(p, q, r, f"R²={r2:.4f}")
                plt.figure(figsize=(7, 4))

                plt.scatter(bits, loss_array, color='red', s=10, label='Original loss (1,2,3,4...)')

                b_dense = np.linspace(0.001, max(bits), 100)
                y_dense = np.exp(p * b_dense ** 2 + q * b_dense + r)
                r2_label = f', R²={r2:.4f}' if valid_fit and not np.isnan(r2) else ''
                plt.plot(b_dense, y_dense, 'b-', label=f'Log-quad fit (exp(pb²+qb+r)){r2_label}')

                plt.scatter(0, l0, color='green', s=10, label=f'L0 = {l0:.2f}')
                plt.scatter(reference_bit, reference_loss, color='orange', s=10,
                            label=f'L{reference_bit} = {reference_loss:.2f}')

                plt.title(
                    f'Expert {expert_idx} | Neuron {i} | '
                    f'L0={l0:.2f}, L{reference_bit}={reference_loss:.2f}'
                    f'{f", R²={r2:.4f}" if valid_fit and not np.isnan(r2) else ""}'
                )
                plt.xlabel('bit')
                plt.ylabel('loss')
                plt.yscale('log')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                os.makedirs(f'plot/{quant_type}_bit_loss_fit', exist_ok=True)
                plt.savefig(f'plot/{quant_type}_bit_loss_fit/exp2_expert_{expert_idx}_neuron_{i}.png', dpi=150)
                plt.close()

        L0.append(expert_L0)
        fit_params.append(expert_fit_params)

    return L0, fit_params


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
            rates_0_list, fit_params = extrapolate_0bit_loss_fix(rates_copy, quant_type=quant_type, save_plots=False)
            rates[0] = [rates_0_list[0]]  # Make it consistent with other bits: [array]
            rates_0 = rates_0_list[0]

        all_data[quant_type] = {
            'rates': rates,
            'rates_0': rates_0,
            'fit_params': fit_params,
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
    fit_annotations = []  # Store fit formulas for legend/annotation

    if all_data[demo_quant] is not None:
        data = all_data[demo_quant]
        rates = data['rates']
        fit_params = data['fit_params']
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

        b_dense = np.linspace(0, max(bits_sorted) + 0.5, 100)

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

            # Plot fit curve if available
            if fit_params is not None and fit_params[0][neuron_idx]['valid']:
                fp = fit_params[0][neuron_idx]
                p_coef, q_coef, r_coef = fp['p'], fp['q'], fp['r']
                y_fit = np.exp(p_coef * b_dense ** 2 + q_coef * b_dense + r_coef)
                ax1.plot(b_dense, y_fit, color=color, linestyle='-', linewidth=2.5, alpha=0.8,
                        label=f'Neuron {neuron_idx} (fit)')

                # Store fit formula for annotation
                fit_formula = f'N{neuron_idx}: log(L) = {p_coef:.2e}b² + {q_coef:.2e}b + {r_coef:.2e}'
                fit_annotations.append((color, fit_formula))

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

        # Add fit formulas as text annotation in the lower left
        if fit_annotations:
            fit_text = "Fit Formulas:\n"
            for color, formula in fit_annotations:
                fit_text += formula + "\n"
            ax1.text(0.03, 0.03, fit_text.strip(), transform=ax1.transAxes,
                    fontsize=7, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

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

            # Plot fit curve if we have params - original alpha
            if data['fit_params'] is not None and data['fit_params'][0][neuron_idx]['valid']:
                fp = data['fit_params'][0][neuron_idx]
                p_coef, q_coef, r_coef = fp['p'], fp['q'], fp['r']
                b_dense = np.linspace(0, max(bits_sorted_with_0), 50)
                y_fit = np.exp(p_coef * b_dense ** 2 + q_coef * b_dense + r_coef)
                ax2.plot(b_dense, y_fit, color=color, linestyle='-', linewidth=2, alpha=0.5, zorder=2)

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

            # Plot fit curve if we have params - original alpha
            if data['fit_params'] is not None and data['fit_params'][0][neuron_idx]['valid']:
                fp = data['fit_params'][0][neuron_idx]
                p_coef, q_coef, r_coef = fp['p'], fp['q'], fp['r']
                b_dense = np.linspace(0, max(bits_sorted_with_0), 50)
                y_fit = np.exp(p_coef * b_dense ** 2 + q_coef * b_dense + r_coef)
                ax3.plot(b_dense, y_fit, color=color, linestyle='-', linewidth=2, alpha=0.5, zorder=2)

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
    expert_idx: int = 0,
    outlier_bits: Set[int] = None,
) -> Dict[str, Dict[int, float]]:
    """Get R² mean stats per layer for a single model (both quant types)."""
    if outlier_bits is None:
        outlier_bits = {1, 2, 3, 4}

    # Discover all available layers
    all_layers = set()
    quants_for_discovery = [
        ('turboquant', 'turboquant_innerproduct'),
        ('gptq', 'gptq_quant_outlier'),
    ]
    for quant_type, rank_mode in quants_for_discovery:
        cache_dir = os.path.join(INTERMEDIATE_RESULT_DIR, f"quant_outlier_{quant_type}", rank_mode, model_id)
        if os.path.exists(cache_dir):
            for filename in os.listdir(cache_dir):
                if filename.endswith('_b1.pt') and model_id in filename:
                    parts = filename.split('_L')
                    if len(parts) > 1:
                        layer_part = parts[1].split('_b')[0]
                        try:
                            all_layers.add(int(layer_part))
                        except ValueError:
                            pass

    layer_indices = sorted(all_layers)
    if not layer_indices:
        print(f"No layers found for model {model_id}")
        return {'turboquant': {}, 'gptq': {}}

    print(f"Processing model {model_id}, layers: {layer_indices}")

    result = {'turboquant': {}, 'gptq': {}}

    for quant_type, rank_mode in quants_for_discovery:
        cache_dir = os.path.join(INTERMEDIATE_RESULT_DIR, f"quant_outlier_{quant_type}", rank_mode, model_id)
        print(f"  {quant_type}: cache_dir = {cache_dir}, exists? {os.path.exists(cache_dir)}")

        for lidx in layer_indices:
            rates = {}
            for x in outlier_bits:
                cache_path = os.path.join(cache_dir, f"{model_id}_L{lidx}_b{x}.pt")
                if os.path.exists(cache_path):
                    try:
                        import torch
                        cached_data = torch.load(cache_path, map_location='cpu')
                        rates[x] = cached_data[expert_idx].detach().cpu().float().numpy()
                    except Exception as e:
                        print(f"    {quant_type} layer {lidx} bit {x} load error: {e}")
                else:
                    print(f"    {quant_type} layer {lidx} bit {x} missing: {cache_path}")

            if not rates or len(rates) < 2:
                print(f"  {quant_type} layer {lidx}: not enough bits ({len(rates)})")
                continue

            bits_sorted = sorted(rates.keys())
            n_neurons = len(rates[bits_sorted[0]])
            b_array = np.array(bits_sorted, dtype=float)

            r2_values = []
            n_pos_less3 = 0
            n_fit_fail = 0
            n_r2_nan = 0

            for i in range(n_neurons):
                loss_array = np.array([rates[b][i] for b in bits_sorted])
                positive_mask = loss_array > 0

                if positive_mask.sum() >= 3:
                    try:
                        fit_bits = b_array[positive_mask]
                        fit_loss = loss_array[positive_mask]
                        log_loss = np.log(fit_loss)

                        p, q, r = np.polyfit(fit_bits, log_loss, deg=2)
                        r2 = compute_r_squared(fit_bits, log_loss, p, q, r)

                        if not np.isnan(r2):
                            r2_values.append(r2)
                        else:
                            n_r2_nan += 1
                    except Exception:
                        n_fit_fail += 1
                else:
                    n_pos_less3 += 1

            if r2_values:
                result[quant_type][lidx] = np.mean(r2_values)
                print(f"  {quant_type} layer {lidx}: {len(r2_values)} neurons, mean R² = {result[quant_type][lidx]:.4f}")
            else:
                print(f"  {quant_type} layer {lidx}: no valid R² values (pos<3: {n_pos_less3}, fit_fail: {n_fit_fail}, r2_nan: {n_r2_nan})")

    print(f"  Result: TQ layers {sorted(result['turboquant'].keys())}, GPTQ layers {sorted(result['gptq'].keys())}")
    return result


def analyze_multi_model_r2(
    model_ids: List[str] = None,
    expert_idx: int = 0,
    outlier_bits: Set[int] = None,
    save_dir: str = None,
    use_pdf: bool = False,
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
    if model_ids is None or not model_ids:
        model_ids = sorted(KNOWN_MODELS.keys())
        print(f"Using all known models: {model_ids}")

    if len(model_ids) > 5:
        print(f"Warning: only first 5 models will be plotted (got {len(model_ids)})")
        model_ids = model_ids[:5]

    # Get stats for all models
    all_model_stats = []
    for model_id in model_ids:
        stats = get_model_layer_stats(model_id, expert_idx, outlier_bits)
        all_model_stats.append((model_id, stats))

    # Create figure: 1 row, N columns
    n_models = len(all_model_stats)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
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

        # Plot bars for each quant type
        for qt_idx, quant_type in enumerate(['turboquant', 'gptq']):
            means = []
            for lidx in layer_indices:
                means.append(stats[quant_type].get(lidx, np.nan))

            offset = -width/2 if qt_idx == 0 else width/2
            ax.bar(x + offset, means, width, label=labels[quant_type],
                   color=colors[quant_type], alpha=0.8)

        # Add reference lines
        ax.axhline(0.95, color='#27ae60', linestyle='--', alpha=0.7, linewidth=1.5, label='R²=0.95')
        ax.axhline(0.99, color='#c0392b', linestyle=':', alpha=0.7, linewidth=1.5, label='R²=0.99')

        # Collect all R² values to determine y-axis range
        all_r2 = []
        for quant_type in ['turboquant', 'gptq']:
            for lidx in layer_indices:
                val = stats[quant_type].get(lidx)
                if val is not None and not np.isnan(val):
                    all_r2.append(val)

        # Dynamic y-axis limits
        if all_r2:
            min_r2 = min(all_r2)
            y_min = min(0.8, min_r2 - 0.01)
        else:
            y_min = 0.94

        # Formatting
        ax.set_xlabel('Layer Index', fontsize=10)
        ax.set_ylabel('Mean R²', fontsize=10)
        ax.set_title(model_id, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in layer_indices], rotation=90, fontsize=8)
        ax.set_ylim(y_min, 1.005)
        ax.grid(True, alpha=0.3, axis='y')

        # Only show legend on first plot
        if ax_idx == 0:
            ax.legend(fontsize=8, loc='lower right')

    plt.tight_layout()

    # Save plot
    if save_dir is None:
        save_dir = 'plot/neuron_rates_fit'
    os.makedirs(save_dir, exist_ok=True)
    ext = 'pdf' if use_pdf else 'png'
    model_str = '_'.join([m.replace('-', '_') for m in model_ids])
    save_path = os.path.join(save_dir, f'multi_model_r2_comparison_{model_str}.{ext}')
    if use_pdf:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nMulti-model R² plot saved to {save_path}")
    plt.close()


def main():
    """Command-line interface for bit loss fit visualizations."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Bit loss extrapolation and visualization tools"
    )
    parser.add_argument("--models", nargs="+",
                      help="Model identifiers (up to 5, for R² analysis, default: auto-discover all)")
    parser.add_argument("--model", default="deepseek-v1-moe-16b",
                      help="Single model identifier (for neuron rate plotting)")
    parser.add_argument("--layer", type=int, default=1,
                      help="Layer index (for neuron rate plotting)")
    parser.add_argument("--expert", type=int, default=0,
                      help="Expert index")
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

    args = parser.parse_args()

    if args.analyze_r2:
        analyze_multi_model_r2(
            model_ids=args.models,  # None = auto-discover
            expert_idx=args.expert,
            outlier_bits=set(args.bits),
            save_dir=args.save_dir,
            use_pdf=args.pdf,
        )
    else:
        # Default to the new fit comparison plot
        plot_neuron_rates_with_fit(
            model_id=args.model,
            layer_idx=args.layer,
            expert_idx=args.expert,
            p=args.p,
            n_show_neurons=args.n_show_neurons,
            outlier_bits=set(args.bits),
            use_0bit=not args.no_0bit,
            save_dir=args.save_dir,
            use_pdf=args.pdf,
        )


if __name__ == "__main__":
    main()
