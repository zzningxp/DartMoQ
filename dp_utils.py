from collections import Counter
from pyexpat import model
import os
import time
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import curve_fit
from typing import Dict

INTERMEDIATE_RESULT_DIR = "intermediate_result"


def compute_r_squared_for_rates(rates: Dict[int, List[np.ndarray]]) -> Dict[str, float]:
    """
    Compute mean and median R² (coefficient of determination) for log-quadratic fit across all neurons in a layer.

    This function is aligned with the R² calculation in viz/bit_loss_fit.py to ensure consistency.

    R² measures how well the log-quadratic model fits the loss vs bit-width data.
    High R² (close to 1.0) indicates reliable 0-bit extrapolation.

    Statistically, most models (DeepSeek v1/v2, OLMoE, Moonlight) have mean R² > 0.99,
    while Qwen3 has a few layers below 0.99. We use 0.99 as the threshold.

    Args:
        rates: Dictionary mapping bit width to list of expert loss arrays

    Returns:
        Dictionary with 'mean' and 'median' R² values
    """
    bits = sorted(rates.keys())
    if 0 in bits:
        bits.remove(0)

    if len(bits) < 3:
        return {'mean': float('nan'), 'median': float('nan')}

    def compute_r_squared(x: np.ndarray, y: np.ndarray, p: float, q: float, r: float) -> float:
        """Compute coefficient of determination (R²) for log-quadratic fit - aligned with viz/bit_loss_fit.py."""
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

    all_r2_values = []
    b_array = np.array(bits, dtype=float)
    n_experts = len(rates[bits[0]])

    for expert_idx in range(n_experts):
        expert_rates = {}
        n_neurons = None
        for b in bits:
            val = rates[b][expert_idx]
            if hasattr(val, 'detach'):
                expert_rates[b] = val.detach().cpu().float().numpy()
            else:
                expert_rates[b] = np.asarray(val, dtype=float)
            if n_neurons is None:
                n_neurons = len(expert_rates[b])

        for i in range(n_neurons):
            loss_array = np.array([expert_rates[b][i] for b in bits])
            positive_mask = loss_array > 0

            # Perfect fit cases - aligned with viz/bit_loss_fit.py
            if positive_mask.sum() == 0:
                all_r2_values.append(1.0)
                continue
            if positive_mask.sum() >= 2 and (loss_array[positive_mask] == loss_array[positive_mask][0]).all():
                all_r2_values.append(1.0)
                continue

            # Normal R² calculation - aligned with viz/bit_loss_fit.py
            if positive_mask.sum() >= 3:
                try:
                    fit_bits = b_array[positive_mask]
                    fit_loss = loss_array[positive_mask]
                    log_loss = np.log(fit_loss)

                    p, q, r_coeff = np.polyfit(fit_bits, log_loss, deg=2)
                    r2 = compute_r_squared(fit_bits, log_loss, p, q, r_coeff)

                    if not np.isnan(r2):
                        all_r2_values.append(r2)
                except Exception:
                    pass

    if all_r2_values:
        r2_arr = np.array(all_r2_values)
        return {
            'mean': float(np.mean(r2_arr)),
            'median': float(np.median(r2_arr))
        }
    return {'mean': float('nan'), 'median': float('nan')}

def calculate_quant_overhead(groupsize: int, quant_type: str = "gptq") -> float:

    # Currently both use the same formula, but we keep the quant_type parameter
    # for future extensibility
    if quant_type not in ["gptq", "turboquant"]:
        raise ValueError(f"Unknown quant_type: {quant_type}, must be 'gptq' or 'turboquant'")
    
    return (groupsize * 32) / 128.0

def enum_optimal_m_scheme_global_fast_with_0bit_compensation(
    expert_rates_list: List[Dict[int, np.ndarray]],
    expert_activation_rates: List,
    slice_expert_num: int,
    target_bpw: float,
    epsilon: float,
    bits: List[int],
    expert_sorted_indices: List[np.ndarray],
    expert_subexpert_neurons: List[List[List[int]]],
    sorted_subexpert_ids: List[Tuple[int, int]],
    block_losses: np.ndarray,
    bit_to_idx: Dict[int, int],
    quant_overhead: float = 0.25
):
    """
    Global DP with monotonicity constraint and 0bit compensation.
    """
    n_experts = len(expert_rates_list)
    bits_sorted_asc = sorted(bits)
    bits_sorted_desc = sorted(bits, reverse=True)
    n_bits = len(bits)
    total_subexperts = n_experts * slice_expert_num
    
    # Calculate effective bits considering overhead
    def get_effective_bit(bit):
        if bit == 0:
            return 0.0
        else:
            return bit + quant_overhead
    
    # Target effective total bits: target_bpw is raw bpw, add overhead for each non-zero bit
    # We approximate by assuming all bits are non-zero for target calculation
    target_total_effective = (target_bpw + quant_overhead) * total_subexperts
    
    # Discretization setup
    search_scale_factor = 20
    max_effective_bit = max(get_effective_bit(b) for b in bits_sorted_asc)
    max_total_effective = max_effective_bit * total_subexperts
    
    discretized_budget = int(target_total_effective * search_scale_factor)
    discretized_max = int(max_total_effective * search_scale_factor)
    discretized_epsilon = int(epsilon * total_subexperts * search_scale_factor)
    
    INF = float('inf')
    
    # DP state: dp[k][w][b_idx] = min loss for first k blocks,
    #                             using w discretized effective bits,
    #                             where k-th block uses bits_sorted_desc[b_idx]
    prev_dp = np.full((discretized_max + 1, n_bits), INF)
    choice_history = []
    
    # Initialize for first block (k=0)
    for b_idx, bit in enumerate(bits_sorted_desc):
        effective_bit = get_effective_bit(bit)
        w = int(effective_bit * search_scale_factor)
        if w <= discretized_max:
            bit_idx_in_arr = bit_to_idx[bit]
            prev_dp[w, b_idx] = block_losses[bit_idx_in_arr, 0]
    
    # Fill DP table
    for k in range(1, total_subexperts):
        curr_dp = np.full((discretized_max + 1, n_bits), INF)
        curr_choice = [[(-1, -1) for __ in range(n_bits)] for _ in range(discretized_max + 1)]
        
        for w_prev in range(discretized_max + 1):
            for b_prev_idx in range(n_bits):
                if prev_dp[w_prev, b_prev_idx] == INF:
                    continue
                
                # Next block can only use <= previous bit (monotonic non-increasing)
                for b_curr_idx in range(b_prev_idx, n_bits):
                    bit_curr = bits_sorted_desc[b_curr_idx]
                    effective_bit = get_effective_bit(bit_curr)
                    w_add = int(effective_bit * search_scale_factor)
                    w_curr = w_prev + w_add
                    
                    if w_curr > discretized_max:
                        continue
                    
                    bit_idx_in_arr = bit_to_idx[bit_curr]
                    new_loss = prev_dp[w_prev, b_prev_idx] + block_losses[bit_idx_in_arr, k]
                    
                    if new_loss < curr_dp[w_curr, b_curr_idx]:
                        curr_dp[w_curr, b_curr_idx] = new_loss
                        curr_choice[w_curr][b_curr_idx] = (w_prev, b_prev_idx)
        
        prev_dp = curr_dp
        choice_history.append(curr_choice)
    
    # Find best w in epsilon range
    best_w = -1
    best_b_idx = -1
    best_loss = INF
    
    search_min = max(0, discretized_budget - discretized_epsilon)
    search_max = min(discretized_max, discretized_budget + discretized_epsilon)
    
    for w in range(search_min, search_max + 1):
        for b_idx in range(n_bits):
            if prev_dp[w, b_idx] < best_loss:
                best_loss = prev_dp[w, b_idx]
                best_w = w
                best_b_idx = b_idx
    
    if best_w == -1:
        raise ValueError("No feasible solution found with 0bit compensation, please check target_bpw and epsilon")
    
    # Backtrack to get global scheme (in sorted sub-expert order)
    global_scheme_sorted = [0] * total_subexperts
    current_w = best_w
    current_b_idx = best_b_idx
    
    global_scheme_sorted[-1] = bits_sorted_desc[current_b_idx]
    
    for k in reversed(range(total_subexperts - 1)):
        prev_w, prev_b_idx = choice_history[k][current_w][current_b_idx]
        global_scheme_sorted[k] = bits_sorted_desc[prev_b_idx]
        current_w, current_b_idx = prev_w, prev_b_idx
    
    # Calculate and print stats
    total_raw_bits = sum(global_scheme_sorted)
    total_effective_bits = sum(get_effective_bit(b) for b in global_scheme_sorted)
    avg_raw_bpw = total_raw_bits / total_subexperts
    avg_effective_bpw = total_effective_bits / total_subexperts
    zero_count = sum(1 for b in global_scheme_sorted if b == 0)
    
    print(f"Multi Bits Global DP mode with 0bit compensation: {Counter(global_scheme_sorted)}")
    print(f"  Raw average bpw: {avg_raw_bpw:.3f}, Effective average bpw: {avg_effective_bpw:.3f}",
          f"  Target bpw: {target_bpw:.3f} + {quant_overhead:.3f}, {zero_count}/{total_subexperts} sub-experts are 0bit")
    print(f"  Best loss: {best_loss:.4f}")
    
    # Step 5: Map back to per-expert scheme
    per_expert_scheme = [[0] * slice_expert_num for _ in range(n_experts)]
    for pos, (expert_idx, sub_idx) in enumerate(sorted_subexpert_ids):
        per_expert_scheme[expert_idx][sub_idx] = global_scheme_sorted[pos]
    
    # Step 6: Build per-expert neuron bits arrays
    per_expert_neuron_bits = []
    for expert_idx in range(n_experts):
        n_neurons = len(expert_sorted_indices[expert_idx])
        neuron_bits = np.zeros(n_neurons, dtype=int)
        for sub_idx in range(slice_expert_num):
            bit = per_expert_scheme[expert_idx][sub_idx]
            neurons_in_sub = expert_subexpert_neurons[expert_idx][sub_idx]
            for neuron_idx in neurons_in_sub:
                neuron_bits[neuron_idx] = bit
        per_expert_neuron_bits.append(neuron_bits)
    
    return per_expert_scheme, per_expert_neuron_bits

def neuron_level_dp_general_with_0bit_compensation(
    rates: Dict[int, np.ndarray],
    bits: List[int],
    target_bpw: float,
    epsilon: float = 0,
    quant_overhead: float = 0.25
) -> np.ndarray:
    """
    Neuron-level DP with 0bit compensation: 0bit does not incur quantization overhead.
    
    The key idea is that for each bit choice:
    - 0bit: cost = 0 (no overhead)
    - b>0: cost = b + quant_overhead
    
    We need to find a bit assignment where the total effective cost is within budget,
    while minimizing total loss.
    """
    bits_sorted = sorted(bits)
    n_neurons = len(rates[bits_sorted[0]])
    
    # Calculate effective bits considering overhead
    def get_effective_bit(bit):
        if bit == 0:
            return 0.0
        else:
            return bit + quant_overhead
    
    # Target effective total bits: target_bpw is raw bpw, add overhead for each non-zero bit
    # We approximate by assuming all bits are non-zero for target calculation
    target_total_effective = (target_bpw + quant_overhead) * n_neurons

    # We'll use a DP that tracks effective bits with some discretization
    # Scale factors for discretization
    search_scale_factor = 20  # Scale to handle floating points, 0.05 bit precision
    max_effective_bit = max(get_effective_bit(b) for b in bits_sorted)
    max_total_effective = max_effective_bit * n_neurons
    
    discretized_budget = int(target_total_effective * search_scale_factor)
    discretized_max = int(max_total_effective * search_scale_factor)
    discretized_epsilon = int(epsilon * n_neurons * search_scale_factor)
    
    INF = float('inf')
    
    # DP table: dp[i][w] = min loss for first i neurons with w discretized effective bits
    # We use a rolling array for memory efficiency
    prev_dp = np.full(discretized_max + 1, INF)
    prev_dp[0] = 0.0
    choice_history = []
    
    for i in range(n_neurons):
        curr_dp = np.full(discretized_max + 1, INF)
        curr_choice = np.full(discretized_max + 1, -1, dtype=int)
        
        for w_prev in range(discretized_max + 1):
            if prev_dp[w_prev] == INF:
                continue
            
            for bit in bits_sorted:
                effective_bit = get_effective_bit(bit)
                w_curr = w_prev + int(effective_bit * search_scale_factor)
                
                if w_curr <= discretized_max:
                    new_loss = prev_dp[w_prev] + rates[bit][i]
                    if new_loss < curr_dp[w_curr]:
                        curr_dp[w_curr] = new_loss
                        curr_choice[w_curr] = bit
        
        prev_dp = curr_dp
        choice_history.append(curr_choice)
    
    # Find the best solution within epsilon tolerance
    best_w = -1
    best_loss = INF
    search_min = max(0, discretized_budget - discretized_epsilon)
    search_max = min(discretized_max, discretized_budget + discretized_epsilon)
    
    for w in range(search_min, search_max + 1):
        if prev_dp[w] < best_loss:
            best_loss = prev_dp[w]
            best_w = w
    
    if best_w == -1:
        raise ValueError("No feasible solution found with 0bit compensation, please check target_bpw and epsilon")
    
    # Backtrack to get the solution
    neuron_bits = np.zeros(n_neurons, dtype=int)
    current_w = best_w
    
    for i in reversed(range(n_neurons)):
        choice = choice_history[i][current_w]
        neuron_bits[i] = choice
        effective_bit = get_effective_bit(choice)
        current_w -= int(effective_bit * search_scale_factor)
    
    # Calculate and print stats
    total_raw_bits = np.sum(neuron_bits)
    total_effective_bits = sum(get_effective_bit(b) for b in neuron_bits)
    avg_raw_bpw = total_raw_bits / n_neurons
    avg_effective_bpw = total_effective_bits / n_neurons
    zero_count = np.sum(neuron_bits == 0)
    
    print(f"0bit compensation enabled: {zero_count}/{n_neurons} neurons are 0bit")
    print(f"  Raw average bpw: {avg_raw_bpw:.3f}, Effective average bpw: {avg_effective_bpw:.3f}",
          f"  Target bpw: {target_bpw:.3f} + {quant_overhead:.3f}, Quant overhead per non-zero bit: {quant_overhead:.3f}")
    
    return neuron_bits

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
            val = rates[b][expert_idx]
            if hasattr(val, 'detach'):
                expert_rates[b] = val.detach().cpu().float().numpy()
            else:
                expert_rates[b] = np.asarray(val, dtype=float)
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
            # function: log(loss) = p*b^2 + q*b + r
            # loss(b) = exp(p*b^2 + q*b + r)
            # -------------------------------------------------------------
            try:
                log_loss = np.log(loss_array)

                p, q, r = np.polyfit(b_array, log_loss, deg=2)

                l0 = np.exp(r)

                l1 = expert_rates[1][i] if 1 in expert_rates else expert_rates[bits[0]][i]
                if l0 < l1:
                    l0 = l1 * 2.0

            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                l1 = expert_rates[1][i] if 1 in expert_rates else expert_rates[bits[0]][i]
                l0 = l1 * 2.0

            expert_L0[i] = l0

            if save_plots:
                print(p, q, r)
                plt.figure(figsize=(7, 4))

                plt.scatter(bits, loss_array, color='red', s=10, label='Original loss (1,2,3,4...)')

                b_dense = np.linspace(0.001, max(bits), 100)
                y_dense = np.exp(p * b_dense ** 2 + q * b_dense + r)
                plt.plot(b_dense, y_dense, 'b-', label=f'Log-quad fit (exp(pb²+qb+r))')

                plt.scatter(0, l0, color='green', s=10, label=f'L0 = {l0:.2f}')
                plt.scatter(1, l1, color='orange', s=10, label=f'L1 = {l1:.2f}')

                plt.title(f'Expert {expert_idx} | Neuron {i} | L0={l0:.2f}, L1={l1:.2f}')
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

def extrapolate_0bit_loss_fix(rates: Dict[int, List[np.ndarray]], quant_type: str = "gptq", save_plots: bool = False) -> List[np.ndarray]:
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
            val = rates[b][expert_idx]
            if hasattr(val, 'detach'):
                expert_rates[b] = val.detach().cpu().float().numpy()
            else:
                expert_rates[b] = np.asarray(val, dtype=float)
            if n_neurons is None:
                n_neurons = len(expert_rates[b])
            else:
                assert len(expert_rates[b]) == n_neurons, \
                    f"expert {expert_idx}, bit {b} has inconsistent loss array length"

        b_array = np.array(bits, dtype=float)
        expert_L0 = np.zeros(n_neurons, dtype=float)

        for i in range(n_neurons):
            loss_array = np.array([expert_rates[b][i] for b in bits])
            positive_mask = loss_array > 0

            if not np.any(positive_mask):
                expert_L0[i] = 0.0
                continue

            reference_bit = bits[0]
            if 1 in expert_rates:
                reference_bit = 1

            reference_loss = expert_rates[reference_bit][i]
            fallback_l0 = max(reference_loss * 2.0, 1e-12)

            # --------------------- log quadratic fit ---------------------
            # function: log(loss) = p*b^2 + q*b + r
            # loss(b) = exp(p*b^2 + q*b + r)
            # -------------------------------------------------------------
            p = q = r = np.nan
            try:
                if positive_mask.sum() >= 3:
                    fit_bits = b_array[positive_mask]
                    fit_loss = loss_array[positive_mask]
                    log_loss = np.log(fit_loss)

                    p, q, r = np.polyfit(fit_bits, log_loss, deg=2)
                    l0 = np.exp(r)
                else:
                    l0 = fallback_l0

                if reference_loss > 0:
                    if l0 < reference_loss:
                        l0 = fallback_l0

            except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError):
                l0 = fallback_l0

            expert_L0[i] = l0

            if save_plots:
                print(p, q, r)
                plt.figure(figsize=(7, 4))

                plt.scatter(bits, loss_array, color='red', s=10, label='Original loss (1,2,3,4...)')

                b_dense = np.linspace(0.001, max(bits), 100)
                y_dense = np.exp(p * b_dense ** 2 + q * b_dense + r)
                plt.plot(b_dense, y_dense, 'b-', label=f'Log-quad fit (exp(pb²+qb+r))')

                plt.scatter(0, l0, color='green', s=10, label=f'L0 = {l0:.2f}')
                plt.scatter(reference_bit, reference_loss, color='orange', s=10,
                            label=f'L{reference_bit} = {reference_loss:.2f}')

                plt.title(
                    f'Expert {expert_idx} | Neuron {i} | '
                    f'L0={l0:.2f}, L{reference_bit}={reference_loss:.2f}'
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

    return L0

def generate_valid_m_schemes_general(bits, s, target_bpw, epsilon):
    """
    generate_valid_m_schemes_general(bits, s, target_bpw, epsilon)
    incremental backtracking enumeration to generate all valid m-schemes that satisfy the bpw constraint (general bit version)
    """
    bits_sorted = sorted(bits, reverse=True)
    min_bit = bits_sorted[-1]
    max_bit = bits_sorted[0]

    target_total = target_bpw * s
    min_total = min_bit * s
    max_total = max_bit * s
    target_total_clipped = np.clip(target_total, min_total, max_total)
    valid_schemes = []

    def backtrack(pos, curr_total, curr_scheme, last_bit):
        if pos == s:
            if abs(curr_total - target_total_clipped) <= epsilon * s:
                valid_schemes.append(tuple(curr_scheme))
            return

        remaining = s - pos
        if curr_total + min_bit * remaining > target_total_clipped + epsilon * s:
            return
        if curr_total + max_bit * remaining < target_total_clipped - epsilon * s:
            return

        for bit in bits_sorted:
            if bit <= last_bit:
                backtrack(pos + 1, curr_total + bit, curr_scheme + [bit], bit)

    for first_bit in bits_sorted:
        backtrack(1, first_bit, [first_bit], first_bit)

    return valid_schemes

def get_unified_sorted_idx_general(rates: Dict[int, np.ndarray], bits: List[int]) -> np.ndarray:
    """
    core step 1: unified marginal gain sorting (general bit version)
    sorting criterion: use the lowest non-zero bit's loss
    """
    bits_sorted = sorted(bits)
    n_neurons = len(rates[bits_sorted[0]])
    idx = np.arange(n_neurons)
    lowest_bit = bits_sorted[0]
    # sort by the lowest bit's loss (descending)
    sorted_idx = idx[np.argsort(-rates[lowest_bit])]
    return sorted_idx

def precompute_block_losses_general(sorted_idx, rates, bits, s):
    n_neurons = len(sorted_idx)
    assert n_neurons % s == 0, "number of neurons must be divisible by the number of blocks"
    m = n_neurons // s

    bit_to_idx = {b: i for i, b in enumerate(bits)}
    block_losses = np.zeros((len(bits), s))

    for k in range(s):
        start = k * m
        end = start + m
        idx_in_block = sorted_idx[start:end]
        for bit in bits:
            bit_idx = bit_to_idx[bit]
            block_losses[bit_idx, k] = rates[bit][idx_in_block].sum()

    return block_losses, bit_to_idx

def precompute_block_losses_global(
    sorted_neurons: List[Tuple[int, int]],
    expert_rates_list: List[Dict[int, np.ndarray]],
    bits: List[int],
    num_blocks: int
):
    """
    Global version: precompute block losses from globally sorted neurons
    """
    total_neurons = len(sorted_neurons)
    assert total_neurons % num_blocks == 0, "total neurons must be divisible by num_blocks"
    neurons_per_block = total_neurons // num_blocks

    bit_to_idx = {b: i for i, b in enumerate(bits)}
    block_losses = np.zeros((len(bits), num_blocks))

    for block_idx in range(num_blocks):
        start = block_idx * neurons_per_block
        end = start + neurons_per_block
        block_neurons = sorted_neurons[start:end]

        for bit in bits:
            bit_idx = bit_to_idx[bit]
            loss_sum = 0.0
            for (expert_idx, neuron_idx) in block_neurons:
                loss_sum += expert_rates_list[expert_idx][bit][neuron_idx]
            block_losses[bit_idx, block_idx] = loss_sum

    return block_losses, bit_to_idx

def get_sorted_and_block_losses(rates, bits, num_blocks):
    """
    Sort neurons using get_unified_sorted_idx_general and compute block losses.

    Args:
        rates: Dict of {bit: rate_array}
        bits: List of bits
        num_blocks: Number of blocks (s)

    Returns:
        sorted_idx: Sorted neuron indices
        block_losses: Array of shape (len(bits), num_blocks)
        bit_to_idx: Mapping from bit to index
    """
    sorted_idx = get_unified_sorted_idx_general(rates, bits)
    block_losses, bit_to_idx = precompute_block_losses_general(sorted_idx, rates, bits, num_blocks)
    return sorted_idx, block_losses, bit_to_idx

def enum_optimal_m_scheme_separate_fast(rates, s, target_bpw, epsilon=0):
    bits = list(rates.keys())
    n_neurons = len(rates[bits[0]])
    for b in bits[1:]:
        assert len(rates[b]) == n_neurons, f"rates[{b}] length must be consistent with other bits"
    
    sorted_idx = get_unified_sorted_idx_general(rates, bits)
    
    block_losses, bit_to_idx = precompute_block_losses_general(sorted_idx, rates, bits, s)
    
    valid_schemes = generate_valid_m_schemes_general(bits, s, target_bpw, epsilon)
    if not valid_schemes:
        raise ValueError(f"No valid m scheme found for target_bpw={target_bpw}, please adjust parameters")
    
    best_loss = float('inf')
    best_scheme = None
    
    for scheme in valid_schemes:
        total_loss = 0.0
        for k, bit in enumerate(scheme):
            bit_idx = bit_to_idx[bit]
            total_loss += block_losses[bit_idx, k]
        
        # print(f"Scheme: {scheme}, Total Loss: {total_loss:.4f}")
        if total_loss < best_loss:
            best_loss = total_loss
            best_scheme = scheme
    
    print(f"{len(valid_schemes)} valid schemes... Optimal Scheme: {best_scheme}, Minimum Loss: {best_loss:.4f}")

    m = n_neurons // s
    neuron_bits = np.zeros(n_neurons, dtype=int)
    for k, bit in enumerate(best_scheme):
        start = k * m
        end = start + m
        neuron_bits[sorted_idx[start:end]] = bit

    return best_scheme, neuron_bits

def neuron_level_dp_general(
    rates: Dict[int, np.ndarray],
    bits: List[int],
    target_bpw: float,
    epsilon: float = 0,
    enable_0bit_compensation: bool = False,
    quant_overhead: float = 0.25
) -> np.ndarray:
    """
    Neuron-level dynamic programming for bit allocation.
    
    Args:
        rates: Dictionary mapping bit widths to loss arrays
        bits: List of available bit widths
        target_bpw: Target average bits per weight
        epsilon: Tolerance for budget constraint
        enable_0bit_compensation: Whether to enable 0bit compensation (0bit does not incur quant overhead)
        quant_overhead: Quantization overhead per non-zero bit (default 0.25)
    
    Returns:
        Array of bit assignments for each neuron
    """
    bits_sorted = sorted(bits)
    n_neurons = len(rates[bits_sorted[0]])
    min_bit = bits_sorted[0]
    max_bit = bits_sorted[-1]

    assert min_bit <= target_bpw <= max_bit, f"target_bpw must be in [{min_bit}, {max_bit}]"

    if enable_0bit_compensation:
        # Use the 0bit compensation version
        return neuron_level_dp_general_with_0bit_compensation(
            rates, bits, target_bpw, epsilon, quant_overhead
        )
    
    # Original DP without compensation
    target_total_w = round(target_bpw * n_neurons)
    min_total_w = min_bit * n_neurons
    max_total_w = max_bit * n_neurons
    target_total_w = int(np.clip(target_total_w, min_total_w, max_total_w))

    offset = min_total_w
    max_offset_w = max_total_w - offset
    target_offset_w = target_total_w - offset

    INF = float('inf')
    prev_dp = np.full(max_offset_w + 1, INF)
    prev_dp[0] = 0.0
    choice_history = []

    for i in range(n_neurons):
        curr_dp = np.full(max_offset_w + 1, INF)
        curr_choice = np.full(max_offset_w + 1, -1, dtype=int)

        for w_prev in range(max_offset_w + 1):
            if prev_dp[w_prev] == INF:
                continue

            for bit in bits_sorted:
                w_curr = w_prev + (bit - min_bit)
                if w_curr <= max_offset_w:
                    new_loss = prev_dp[w_prev] + rates[bit][i]
                    if new_loss < curr_dp[w_curr]:
                        curr_dp[w_curr] = new_loss
                        curr_choice[w_curr] = bit

        prev_dp = curr_dp
        choice_history.append(curr_choice)

    search_range = int(epsilon * n_neurons)
    best_w = -1
    best_loss = INF
    for w in range(max(0, target_offset_w - search_range),
                   min(max_offset_w, target_offset_w + search_range) + 1):
        if prev_dp[w] < best_loss:
            best_loss = prev_dp[w]
            best_w = w

    if best_w == -1:
        raise ValueError("No feasible solution found, please check target_bpw and epsilon")

    neuron_bits = np.zeros(n_neurons, dtype=int)
    current_w = best_w
    for i in reversed(range(n_neurons)):
        choice = choice_history[i][current_w]
        neuron_bits[i] = choice
        current_w -= (choice - min_bit)

    return neuron_bits

def enum_optimal_m_scheme_global_fast(
    expert_rates_list: List[Dict[int, np.ndarray]],
    expert_activation_rates: List,
    slice_expert_num: int,
    target_bpw: float,
    epsilon: float = 0,
    enable_0bit_compensation: bool = False,
    quant_overhead: float = 0.25
):
    """
    Global DP with monotonicity constraint with optional 0bit compensation.
    
    Args:
        enable_0bit_compensation: Whether to enable 0bit compensation
        quant_overhead: Quantization overhead per non-zero bit
    
    Returns:
        per_expert_scheme: list of lists, per_expert_scheme[expert_idx] is the bit scheme for that expert
        per_expert_neuron_bits: list of arrays, per_expert_neuron_bits[expert_idx] is the bit for each neuron
    """
    n_experts = len(expert_rates_list)
    bits = list(expert_rates_list[0].keys())

    for expert_rates in expert_rates_list:
        assert list(expert_rates.keys()) == bits, "all experts must have same bit set"

    # Step 1: For each expert, sort its neurons and split into sub-experts
    # Also compute each sub-expert's importance and precompute losses
    expert_sorted_indices = []  # expert_sorted_indices[expert_idx] = sorted neuron indices
    expert_subexpert_neurons = []  # expert_subexpert_neurons[expert_idx][sub_idx] = list of neuron indices in this sub-expert
    expert_act_rates = []  # cache act_rates for later use
    all_subexperts = []  # list of (expert_idx, sub_idx, importance)

    bits_sorted_asc = sorted(bits)
    bits_sorted_desc = sorted(bits, reverse=True)

    for expert_idx in range(n_experts):
        rates = expert_rates_list[expert_idx]
        sorted_idx = get_unified_sorted_idx_general(rates, bits)
        expert_sorted_indices.append(sorted_idx)

        # Get and cache act_rate for this expert
        act_rate = expert_activation_rates[expert_idx]
        if hasattr(act_rate, 'detach'):
            act_rate = float(act_rate.detach().cpu().numpy())
        elif hasattr(act_rate, 'item'):
            act_rate = float(act_rate.item())
        else:
            act_rate = float(act_rate)
        expert_act_rates.append(act_rate)

        n_neurons = len(sorted_idx)
        neurons_per_subexpert = n_neurons // slice_expert_num

        # Split into sub-experts
        subexpert_neurons = []
        for sub_idx in range(slice_expert_num):
            start = sub_idx * neurons_per_subexpert
            end = start + neurons_per_subexpert
            neurons_in_sub = sorted_idx[start:end]
            subexpert_neurons.append(neurons_in_sub)

            # Compute importance: average bit 0 loss × expert activation rate
            if len(bits_sorted_asc) == 1:
                combined_score = 1.0
            else:
                combined_score = 0.0
                low_bit = bits_sorted_asc[0]
                for neuron_idx in neurons_in_sub:
                    combined_score += rates[low_bit][neuron_idx]
                combined_score /= len(neurons_in_sub)

            combined_score *= act_rate

            all_subexperts.append((expert_idx, sub_idx, -combined_score))  # negative for ascending sort

        expert_subexpert_neurons.append(subexpert_neurons)

    # Step 2: Sort all sub-experts globally by importance
    all_subexperts_sorted = sorted(all_subexperts, key=lambda x: x[2])  # sort by negative score (ascending)
    # Now all_subexperts_sorted is sorted from most important to least important
    sorted_subexpert_ids = [(x[0], x[1]) for x in all_subexperts_sorted]  # list of (expert_idx, sub_idx)

    total_subexperts = n_experts * slice_expert_num

    # Step 3: Precompute block losses for the sorted sub-experts
    block_losses = np.zeros((len(bits), total_subexperts))
    bit_to_idx = {b: i for i, b in enumerate(bits)}

    for pos, (expert_idx, sub_idx) in enumerate(sorted_subexpert_ids):
        neurons_in_sub = expert_subexpert_neurons[expert_idx][sub_idx]
        rates = expert_rates_list[expert_idx]
        act_rate = expert_act_rates[expert_idx]
        for bit in bits:
            bit_idx = bit_to_idx[bit]
            loss_sum = 0.0
            for neuron_idx in neurons_in_sub:
                loss_sum += rates[bit][neuron_idx]
            block_losses[bit_idx, pos] = loss_sum * act_rate

    # Step 4: Monotonic DP on the sorted sub-experts with optional 0bit compensation
    min_bit = bits_sorted_asc[0]
    max_bit = bits_sorted_asc[-1]
    n_bits = len(bits)

    if enable_0bit_compensation:
        # Use the 0bit compensation version
        return enum_optimal_m_scheme_global_fast_with_0bit_compensation(
            expert_rates_list, expert_activation_rates, slice_expert_num, target_bpw, epsilon,
            bits, expert_sorted_indices, expert_subexpert_neurons,
            sorted_subexpert_ids, block_losses, bit_to_idx, quant_overhead
        )

    # Original DP without compensation
    # Total bit budget
    target_total = target_bpw * total_subexperts
    min_total = min_bit * total_subexperts
    max_total = max_bit * total_subexperts
    target_total_clipped = int(np.clip(target_total, min_total, max_total))

    # Use offset for DP (relative to min_bit) to reduce state space
    offset = min_bit * total_subexperts
    max_offset_w = max_total - offset
    target_offset_w = target_total_clipped - offset

    INF = float('inf')

    # DP state: dp[k][w][b_idx] = min loss for first k blocks, using w offset bits,
    #                             where k-th block uses bits_sorted_desc[b_idx]
    # We only keep previous step for memory efficiency
    prev_dp = np.full((max_offset_w + 1, n_bits), INF)
    # choice: (prev_w, prev_b_idx)
    choice_history = []

    # Initialize for first block (k=0)
    for b_idx, bit in enumerate(bits_sorted_desc):
        w = bit - min_bit
        if w <= max_offset_w:
            bit_idx_in_arr = bit_to_idx[bit]
            prev_dp[w, b_idx] = block_losses[bit_idx_in_arr, 0]

    # Fill DP table
    for k in range(1, total_subexperts):
        curr_dp = np.full((max_offset_w + 1, n_bits), INF)
        curr_choice = [[(-1, -1) for __ in range(n_bits)] for _ in range(max_offset_w + 1)]

        for w_prev in range(max_offset_w + 1):
            for b_prev_idx in range(n_bits):
                if prev_dp[w_prev, b_prev_idx] == INF:
                    continue

                # Next block can only use <= previous bit (monotonic non-increasing)
                # So b_curr_idx >= b_prev_idx in bits_sorted_desc
                for b_curr_idx in range(b_prev_idx, n_bits):
                    bit_curr = bits_sorted_desc[b_curr_idx]
                    w_add = bit_curr - min_bit
                    w_curr = w_prev + w_add

                    if w_curr > max_offset_w:
                        continue

                    bit_idx_in_arr = bit_to_idx[bit_curr]
                    new_loss = prev_dp[w_prev, b_prev_idx] + block_losses[bit_idx_in_arr, k]

                    if new_loss < curr_dp[w_curr, b_curr_idx]:
                        curr_dp[w_curr, b_curr_idx] = new_loss
                        curr_choice[w_curr][b_curr_idx] = (w_prev, b_prev_idx)

        prev_dp = curr_dp
        choice_history.append(curr_choice)

    # Find best w in epsilon range
    search_range = int(epsilon * total_subexperts)
    best_w = -1
    best_b_idx = -1
    best_loss = INF

    for w in range(max(0, target_offset_w - search_range),
                   min(max_offset_w, target_offset_w + search_range) + 1):
        for b_idx in range(n_bits):
            if prev_dp[w, b_idx] < best_loss:
                best_loss = prev_dp[w, b_idx]
                best_w = w
                best_b_idx = b_idx

    if best_w == -1:
        raise ValueError("No feasible solution found, please check target_bpw and epsilon")

    # Backtrack to get global scheme (in sorted sub-expert order)
    global_scheme_sorted = [0] * total_subexperts
    current_w = best_w
    current_b_idx = best_b_idx

    global_scheme_sorted[-1] = bits_sorted_desc[current_b_idx]

    for k in reversed(range(total_subexperts - 1)):
        prev_w, prev_b_idx = choice_history[k][current_w][current_b_idx]
        global_scheme_sorted[k] = bits_sorted_desc[prev_b_idx]
        current_w, current_b_idx = prev_w, prev_b_idx

    total_raw_bits = sum(global_scheme_sorted)
    total_effective_bits = sum(bit + quant_overhead if bit != 0 else 0 for bit in global_scheme_sorted)
    avg_raw_bpw = total_raw_bits / total_subexperts
    avg_effective_bpw = total_effective_bits / total_subexperts
    zero_count = sum(1 for bit in global_scheme_sorted if bit == 0)

    print(f"Multi Bits Global DP mode {Counter(global_scheme_sorted)}")
    print(f"  Raw average bpw: {avg_raw_bpw:.3f}, Effective average bpw: {avg_effective_bpw:.3f}",
          f"  Target bpw: {target_bpw:.3f} + {quant_overhead:.3f}, {zero_count}/{total_subexperts} sub-experts are 0bit")
    print(f"  best loss = {best_loss:.4f}")

    # Step 5: Map back to per-expert scheme
    per_expert_scheme = [[0] * slice_expert_num for _ in range(n_experts)]
    for pos, (expert_idx, sub_idx) in enumerate(sorted_subexpert_ids):
        per_expert_scheme[expert_idx][sub_idx] = global_scheme_sorted[pos]

    # Step 6: Build per-expert neuron bits arrays
    per_expert_neuron_bits = []
    for expert_idx in range(n_experts):
        n_neurons = len(expert_sorted_indices[expert_idx])
        neuron_bits = np.zeros(n_neurons, dtype=int)
        for sub_idx in range(slice_expert_num):
            bit = per_expert_scheme[expert_idx][sub_idx]
            neurons_in_sub = expert_subexpert_neurons[expert_idx][sub_idx]
            for neuron_idx in neurons_in_sub:
                neuron_bits[neuron_idx] = bit
        per_expert_neuron_bits.append(neuron_bits)

    return per_expert_scheme, per_expert_neuron_bits

def enum_optimal_m_scheme_energy_global_fast(
    expert_energy_list: List[np.ndarray],
    expert_activation_rates: List,
    slice_expert_num: int,
    target_bpw: float,
    bits: List[int] = None,
    epsilon: float = 0,
    enable_0bit_compensation: bool = False,
    quant_overhead: float = 0.25
):
    """
    Global DP for energy-based mode with optional 0bit compensation.
    
    Args:
        enable_0bit_compensation: Whether to enable 0bit compensation
        quant_overhead: Quantization overhead per non-zero bit
    
    Returns:
        per_expert_scheme: list of lists, per_expert_scheme[expert_idx] is the bit scheme for that expert
        per_expert_neuron_bits: list of arrays, per_expert_neuron_bits[expert_idx] is the bit for each neuron
    """
    if bits is None:
        bits = [1, 2, 3, 4]

    n_experts = len(expert_energy_list)

    # Step 1: For each expert, sort its neurons by energy (descending) and split into sub-experts
    expert_sorted_indices = []  # expert_sorted_indices[expert_idx] = sorted neuron indices (high energy first)
    expert_subexpert_neurons = []  # expert_subexpert_neurons[expert_idx][sub_idx] = list of neuron indices in this sub-expert
    all_subexperts = []  # list of (expert_idx, sub_idx, importance)

    bits_sorted_asc = sorted(bits)
    bits_sorted_desc = sorted(bits, reverse=True)

    for expert_idx in range(n_experts):
        energy = expert_energy_list[expert_idx]
        # Sort neurons by energy descending
        sorted_idx = np.argsort(-energy)
        expert_sorted_indices.append(sorted_idx)

        n_neurons = len(sorted_idx)
        neurons_per_subexpert = n_neurons // slice_expert_num

        # Split into sub-experts
        subexpert_neurons = []
        for sub_idx in range(slice_expert_num):
            start = sub_idx * neurons_per_subexpert
            end = start + neurons_per_subexpert
            neurons_in_sub = sorted_idx[start:end]
            subexpert_neurons.append(neurons_in_sub)

            # Compute importance: sum of energy in this sub-expert
            sub_energy = energy[neurons_in_sub].sum()

            # Multiply by expert activation rate
            act_rate = expert_activation_rates[expert_idx]
            if hasattr(act_rate, 'detach'):
                act_rate = float(act_rate.detach().cpu().numpy())
            elif hasattr(act_rate, 'item'):
                act_rate = float(act_rate.item())
            else:
                act_rate = float(act_rate)

            importance = sub_energy * act_rate
            all_subexperts.append((expert_idx, sub_idx, -importance))  # negative for ascending sort

        expert_subexpert_neurons.append(subexpert_neurons)

    # Step 2: Sort all sub-experts globally by importance
    all_subexperts_sorted = sorted(all_subexperts, key=lambda x: x[2])  # sort by negative importance (ascending)
    # Now all_subexperts_sorted is sorted from most important to least important
    sorted_subexpert_ids = [(x[0], x[1]) for x in all_subexperts_sorted]  # list of (expert_idx, sub_idx)

    total_subexperts = n_experts * slice_expert_num

    # Step 3: Precompute block losses for the sorted sub-experts
    # For energy mode, loss for bit b is: total_energy / (bit + 1)
    # Higher bit -> lower loss
    block_losses = np.zeros((len(bits), total_subexperts))
    bit_to_idx = {b: i for i, b in enumerate(bits)}

    for pos, (expert_idx, sub_idx) in enumerate(sorted_subexpert_ids):
        neurons_in_sub = expert_subexpert_neurons[expert_idx][sub_idx]
        energy = expert_energy_list[expert_idx]
        sub_energy = energy[neurons_in_sub].sum()
        for bit in bits:
            bit_idx = bit_to_idx[bit]
            # Loss = total_energy / (bit + 1), so higher bit has lower loss
            block_losses[bit_idx, pos] = sub_energy / (bit + 1)

    # Step 4: Monotonic DP on the sorted sub-experts with optional 0bit compensation
    min_bit = bits_sorted_asc[0]
    max_bit = bits_sorted_asc[-1]
    n_bits = len(bits)

    if enable_0bit_compensation:
        # Create fake rates dict for energy mode
        expert_rates_list = []
        for energy in expert_energy_list:
            rates = {}
            for bit in bits:
                rates[bit] = energy / (bit + 1)
            expert_rates_list.append(rates)
        
        # Use the 0bit compensation version
        return enum_optimal_m_scheme_global_fast_with_0bit_compensation(
            expert_rates_list, expert_activation_rates, slice_expert_num, target_bpw, epsilon,
            bits, expert_sorted_indices, expert_subexpert_neurons,
            sorted_subexpert_ids, block_losses, bit_to_idx, quant_overhead
        )

    # Original DP without compensation
    # Total bit budget
    target_total = target_bpw * total_subexperts
    min_total = min_bit * total_subexperts
    max_total = max_bit * total_subexperts
    target_total_clipped = int(np.clip(target_total, min_total, max_total))

    # Use offset for DP (relative to min_bit) to reduce state space
    offset = min_bit * total_subexperts
    max_offset_w = max_total - offset
    target_offset_w = target_total_clipped - offset

    INF = float('inf')

    # DP state: dp[k][w][b_idx] = min loss for first k blocks, using w offset bits,
    #                             where k-th block uses bits_sorted_desc[b_idx]
    # We only keep previous step for memory efficiency
    prev_dp = np.full((max_offset_w + 1, n_bits), INF)
    # choice: (prev_w, prev_b_idx)
    choice_history = []

    # Initialize for first block (k=0)
    for b_idx, bit in enumerate(bits_sorted_desc):
        w = bit - min_bit
        if w <= max_offset_w:
            bit_idx_in_arr = bit_to_idx[bit]
            prev_dp[w, b_idx] = block_losses[bit_idx_in_arr, 0]

    # Fill DP table
    for k in range(1, total_subexperts):
        curr_dp = np.full((max_offset_w + 1, n_bits), INF)
        curr_choice = [[(-1, -1) for __ in range(n_bits)] for _ in range(max_offset_w + 1)]

        for w_prev in range(max_offset_w + 1):
            for b_prev_idx in range(n_bits):
                if prev_dp[w_prev, b_prev_idx] == INF:
                    continue

                # Next block can only use <= previous bit (monotonic non-increasing)
                # So b_curr_idx >= b_prev_idx in bits_sorted_desc
                for b_curr_idx in range(b_prev_idx, n_bits):
                    bit_curr = bits_sorted_desc[b_curr_idx]
                    w_add = bit_curr - min_bit
                    w_curr = w_prev + w_add

                    if w_curr > max_offset_w:
                        continue

                    bit_idx_in_arr = bit_to_idx[bit_curr]
                    new_loss = prev_dp[w_prev, b_prev_idx] + block_losses[bit_idx_in_arr, k]

                    if new_loss < curr_dp[w_curr, b_curr_idx]:
                        curr_dp[w_curr, b_curr_idx] = new_loss
                        curr_choice[w_curr][b_curr_idx] = (w_prev, b_prev_idx)

        prev_dp = curr_dp
        choice_history.append(curr_choice)

    # Find best w in epsilon range
    search_range = int(epsilon * total_subexperts)
    best_w = -1
    best_b_idx = -1
    best_loss = INF

    for w in range(max(0, target_offset_w - search_range),
                   min(max_offset_w, target_offset_w + search_range) + 1):
        for b_idx in range(n_bits):
            if prev_dp[w, b_idx] < best_loss:
                best_loss = prev_dp[w, b_idx]
                best_w = w
                best_b_idx = b_idx

    if best_w == -1:
        raise ValueError("No feasible solution found, please check target_bpw and epsilon")

    # Backtrack to get global scheme (in sorted sub-expert order)
    global_scheme_sorted = [0] * total_subexperts
    current_w = best_w
    current_b_idx = best_b_idx

    global_scheme_sorted[-1] = bits_sorted_desc[current_b_idx]

    for k in reversed(range(total_subexperts - 1)):
        prev_w, prev_b_idx = choice_history[k][current_w][current_b_idx]
        global_scheme_sorted[k] = bits_sorted_desc[prev_b_idx]
        current_w, current_b_idx = prev_w, prev_b_idx

    total_raw_bits = sum(global_scheme_sorted)
    total_effective_bits = sum(bit + quant_overhead if bit != 0 else 0 for bit in global_scheme_sorted)
    avg_raw_bpw = total_raw_bits / total_subexperts
    avg_effective_bpw = total_effective_bits / total_subexperts
    zero_count = sum(1 for bit in global_scheme_sorted if bit == 0)

    print(f"Energy Global DP mode {Counter(global_scheme_sorted)}")
    print(f"  Raw average bpw: {avg_raw_bpw:.3f}, Effective average bpw: {avg_effective_bpw:.3f}",
          f"  Target bpw: {target_bpw:.3f} + {quant_overhead:.3f}, {zero_count}/{total_subexperts} sub-experts are 0bit")
    print(f"  best loss = {best_loss:.4f}")

    # Step 5: Map back to per-expert scheme
    per_expert_scheme = [[0] * slice_expert_num for _ in range(n_experts)]
    for pos, (expert_idx, sub_idx) in enumerate(sorted_subexpert_ids):
        per_expert_scheme[expert_idx][sub_idx] = global_scheme_sorted[pos]

    # Step 6: Build per-expert neuron bits arrays
    per_expert_neuron_bits = []
    for expert_idx in range(n_experts):
        n_neurons = len(expert_sorted_indices[expert_idx])
        neuron_bits = np.zeros(n_neurons, dtype=int)
        for sub_idx in range(slice_expert_num):
            bit = per_expert_scheme[expert_idx][sub_idx]
            neurons_in_sub = expert_subexpert_neurons[expert_idx][sub_idx]
            for neuron_idx in neurons_in_sub:
                neuron_bits[neuron_idx] = bit
        per_expert_neuron_bits.append(neuron_bits)

    return per_expert_scheme, per_expert_neuron_bits

#---- Test ----


def test_read_rates_from_file():
    outlier_bits = {1, 2, 3, 4}
    print(f"simulate quant outlier_bits {outlier_bits}")

    model_id = "deepseek-v1-moe-16b"
    layer_idx = 1
    quant_type = "turboquant"
    cache_dir = os.path.join(INTERMEDIATE_RESULT_DIR, f"quant_outlier_{quant_type}", model_id)

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
                rates[x] = [cached_data[expert_idx][:p]]
            except Exception as e:
                print(f"Failed to load cached data: {e}")

    rates[0] = extrapolate_0bit_loss_fix(rates, quant_type=quant_type, save_plots=True)
    for i in range(p):
        print(i, end=', ')
        print(f"{rates[4][expert_idx][i].item():.4f}", end=', ')
        print(f"{rates[3][expert_idx][i].item():.4f}", end=', ')
        print(f"{rates[2][expert_idx][i].item():.4f}", end=', ')
        print(f"{rates[1][expert_idx][i].item():.4f}", end=', ')
        print(f"{rates[0][expert_idx][i].item():.4f}", end=', ')
        print()

def test_dp_utils():
    np.random.seed(42)
    n_neurons = 1024

    bits = [2, 3, 4]

    rates = {}
    r_base = np.random.rand(n_neurons)

    for bit in sorted(bits, reverse=True):
        if bit == max(bits):
            rates[bit] = r_base
        else:
            higher_bit = bit + 1
            while higher_bit not in rates and higher_bit <= max(bits):
                higher_bit += 1
            if higher_bit not in rates:
                rates[bit] = r_base * (1.5 ** (max(bits) - bit)) + np.random.rand(n_neurons) * 0.1
            else:
                rates[bit] = rates[higher_bit] * 1.3 + np.random.rand(n_neurons) * 0.1

    s = 8
    target_bpw = 2.5
    epsilon = 0.1

    print(f"Neuron Level DP Config:bits={bits}, s={s}, target_bpw={target_bpw}, epsilon={epsilon}")

    print("\n--- Fast m-scheme Search ---")
    tick = time.time()
    best_scheme, neuron_bits_fast = enum_optimal_m_scheme_separate_fast(
        rates, s, target_bpw, epsilon
    )
    print(f"Fast m-scheme Search Time: {time.time() - tick:.4f} s")

def test_global_dp_utils():
    """Test global DP mode with multiple experts"""
    np.random.seed(42)
    n_neurons_per_expert = 1024
    n_experts = 8
    slice_expert_num = 8

    bits = [0, 1, 2, 3, 4]

    # Generate random expert activation rates - some high, some low
    expert_activation_rates = np.array([0.25, 0.2, 0.15, 0.1, 0.1, 0.08, 0.07, 0.05])

    # Generate rates for each expert
    expert_rates_list = []
    for expert_idx in range(n_experts):
        rates = {}
        # Base loss is scaled by expert activation (more active experts have higher loss sensitivity)
        r_base = np.random.rand(n_neurons_per_expert) * (0.5 + expert_activation_rates[expert_idx] * 2)

        for bit in sorted(bits, reverse=True):
            if bit == max(bits):
                rates[bit] = r_base
            else:
                higher_bit = bit + 1
                while higher_bit not in rates and higher_bit <= max(bits):
                    higher_bit += 1
                if higher_bit not in rates:
                    rates[bit] = r_base * (1.5 ** (max(bits) - bit)) + np.random.rand(n_neurons_per_expert) * 0.1
                else:
                    rates[bit] = rates[higher_bit] * 1.3 + np.random.rand(n_neurons_per_expert) * 0.1

        expert_rates_list.append(rates)

    target_bpw = 1.5
    epsilon = 0.1

    print(f"Global DP Config: n_experts={n_experts}, n_neurons_per_expert={n_neurons_per_expert}")
    print(f"  slice_expert_num={slice_expert_num}, bits={bits}, target_bpw={target_bpw}, epsilon={epsilon}")

    print(f"  expert_activation_rates: {expert_activation_rates}")

    print("\n--- Global DP Search ---")
    tick = time.time()
    per_expert_scheme, per_expert_neuron_bits = enum_optimal_m_scheme_global_fast(
        expert_rates_list, expert_activation_rates, slice_expert_num, target_bpw, epsilon
    )
    elapsed = time.time() - tick
    print(f"Global DP Search Time: {elapsed:.4f} s")

    # Verify each expert's scheme is non-increasing
    all_experts_non_increasing = True
    for expert_idx, expert_scheme in enumerate(per_expert_scheme):
        is_expert_non_increasing = all(expert_scheme[i] >= expert_scheme[i+1] for i in range(len(expert_scheme)-1))
        if not is_expert_non_increasing:
            all_experts_non_increasing = False
            print(f"Expert {expert_idx} scheme NOT non-increasing: {expert_scheme}")

    print(f"All experts' internal schemes are non-increasing: {all_experts_non_increasing}")

    # Verify and print stats
    print(f"\nPer-expert sub-expert schemes:")
    for expert_idx in range(n_experts):
        print(f"  Expert {expert_idx}: {per_expert_scheme[expert_idx]}")

    # Count scheme types
    from collections import Counter
    scheme_counter = Counter(tuple(s) for s in per_expert_scheme)
    print(f"\nScheme type count: {scheme_counter}")

    # Build per-expert stats
    expert_bit_counts = [{} for _ in range(n_experts)]
    for expert_idx in range(n_experts):
        neuron_bits = per_expert_neuron_bits[expert_idx]
        for bit in neuron_bits:
            bit = int(bit)
            expert_bit_counts[expert_idx][bit] = expert_bit_counts[expert_idx].get(bit, 0) + 1

    print("\nBit distribution per expert:")
    for expert_idx in range(n_experts):
        print(f"  Expert {expert_idx}: {dict(sorted(expert_bit_counts[expert_idx].items()))}")



if __name__ == "__main__":
    # Sanity checks for the DP / extrapolation logic.
    # test_read_rates_from_file()
    # test_dp_utils()
    # test_global_dp_utils()
    pass
