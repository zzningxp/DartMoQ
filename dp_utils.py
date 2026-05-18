from pyexpat import model
import os
import time
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import curve_fit
from typing import Dict


def _exp_decay_func(b: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    return A * np.exp(B * b) + C


def _poly2_func(b: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    return A * (b ** 2) + B * b + C


def _power_growth_func(x: np.ndarray, A: float, C: float, D: float, x_max: float) -> np.ndarray:
    return A * np.power(x_max - x + 1, C) + D


def extrapolate_0bit_loss(rates: Dict[int, List[np.ndarray]], save_plots: bool = False) -> List[np.ndarray]:
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
            expert_rates[b] = rates[b][expert_idx].detach().cpu().numpy()
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
                plt.savefig(f'plot/bit_loss_fit/exp2_expert_{expert_idx}_neuron_{i}.png', dpi=150)
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
    sorting criterion: combined marginal gain of adjacent bits
    """
    bits_sorted = sorted(bits)
    n_neurons = len(rates[bits_sorted[0]])
    idx = np.arange(n_neurons)

    if len(bits_sorted) == 1:
        return idx

    # compute combined score of marginal gains: sum(adjacent bit gains)
    combined_score = np.zeros(n_neurons)
    for i in range(len(bits_sorted) - 1):
        low_bit = bits_sorted[i]
        high_bit = bits_sorted[i + 1]
        # gain: high_bit replacing low_bit (rates[low] - rates[high])
        gain = rates[low_bit] - rates[high_bit]
        combined_score += gain

    sorted_idx = idx[np.argsort(-combined_score)]
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


def get_unified_sorted_idx_global(
    expert_rates_list: List[Dict[int, np.ndarray]],
    expert_activation_rates: List,
    bits: List[int]
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Global version: unified marginal gain sorting across all experts,
    each neuron is (expert_idx, neuron_idx) tuple, sorted by combined score * expert_activation_rate
    """
    bits_sorted = sorted(bits)
    n_experts = len(expert_rates_list)

    # Collect all neuron indices with expert info
    all_neurons = []  # list of (expert_idx, neuron_idx)
    all_combined_scores = []

    for expert_idx in range(n_experts):
        rates = expert_rates_list[expert_idx]
        n_neurons = len(rates[bits_sorted[0]])

        if len(bits_sorted) == 1:
            combined_score = np.ones(n_neurons)
        else:
            # compute combined score of marginal gains: sum(adjacent bit gains)
            combined_score = np.zeros(n_neurons)
            for i in range(len(bits_sorted) - 1):
                low_bit = bits_sorted[i]
                high_bit = bits_sorted[i + 1]
                gain = rates[low_bit] - rates[high_bit]
                combined_score += gain

        # Multiply by expert activation rate for global sorting (convert to numpy/float)
        act_rate = expert_activation_rates[expert_idx]
        if hasattr(act_rate, 'detach'):
            act_rate = float(act_rate.detach().cpu().numpy())
        elif hasattr(act_rate, 'item'):
            act_rate = float(act_rate.item())
        else:
            act_rate = float(act_rate)
        combined_score *= act_rate

        for neuron_idx in range(n_neurons):
            all_neurons.append((expert_idx, neuron_idx))
            all_combined_scores.append(combined_score[neuron_idx])

    # Sort all neurons globally by combined score
    all_combined_scores = np.array(all_combined_scores)
    sorted_positions = np.argsort(-all_combined_scores)
    sorted_neurons = [all_neurons[pos] for pos in sorted_positions]

    return sorted_neurons, sorted_positions


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


def enum_optimal_m_scheme_fast_general(rates, s, target_bpw, epsilon=0):
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
    epsilon: float = 0
) -> np.ndarray:
    bits_sorted = sorted(bits)
    n_neurons = len(rates[bits_sorted[0]])
    min_bit = bits_sorted[0]
    max_bit = bits_sorted[-1]

    assert min_bit <= target_bpw <= max_bit, f"target_bpw must be in [{min_bit}, {max_bit}]"

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
    epsilon: float = 0
):
    """
    Global DP with monotonicity constraint:
    1. Global neuron sorting (same as before)
    2. DP with monotonic non-increasing bit allocation
    Returns:
        per_expert_scheme: list of lists, per_expert_scheme[expert_idx] is the bit scheme for that expert
        per_expert_neuron_bits: list of arrays, per_expert_neuron_bits[expert_idx] is the bit for each neuron
    """
    n_experts = len(expert_rates_list)
    bits = list(expert_rates_list[0].keys())

    for expert_rates in expert_rates_list:
        assert list(expert_rates.keys()) == bits, "all experts must have same bit set"

    # Step 1: Global sorting of all neurons
    sorted_neurons, _ = get_unified_sorted_idx_global(
        expert_rates_list, expert_activation_rates, bits
    )

    total_neurons = len(sorted_neurons)
    total_blocks = n_experts * slice_expert_num

    # Step 2: Precompute block losses
    block_losses, bit_to_idx = precompute_block_losses_global(
        sorted_neurons, expert_rates_list, bits, total_blocks
    )

    # Step 3: Monotonic DP
    bits_sorted_asc = sorted(bits)  # ascending: [0, 1, 2, 3, 4]
    bits_sorted_desc = sorted(bits, reverse=True)  # descending: [4, 3, 2, 1, 0]
    min_bit = bits_sorted_asc[0]
    max_bit = bits_sorted_asc[-1]
    n_bits = len(bits)

    # Total bit budget
    target_total = target_bpw * total_blocks
    min_total = min_bit * total_blocks
    max_total = max_bit * total_blocks
    target_total_clipped = int(np.clip(target_total, min_total, max_total))

    # Use offset for DP (relative to min_bit) to reduce state space
    offset = min_bit * total_blocks
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
    for k in range(1, total_blocks):
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
    search_range = int(epsilon * total_blocks)
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

    # Backtrack to get scheme
    global_scheme = [0] * total_blocks
    current_w = best_w
    current_b_idx = best_b_idx

    global_scheme[-1] = bits_sorted_desc[current_b_idx]

    for k in reversed(range(total_blocks - 1)):
        prev_w, prev_b_idx = choice_history[k][current_w][current_b_idx]
        global_scheme[k] = bits_sorted_desc[prev_b_idx]
        current_w, current_b_idx = prev_w, prev_b_idx

    print(f"Global DP fast mode: best loss = {best_loss:.4f}")

    # Step 4: Assign bits to neurons based on our global scheme
    neurons_per_block = total_neurons // total_blocks

    neuron_bit_map = {}
    for block_idx, bit in enumerate(global_scheme):
        start = block_idx * neurons_per_block
        end = start + neurons_per_block
        for pos in range(start, end):
            expert_idx, neuron_idx = sorted_neurons[pos]
            if expert_idx not in neuron_bit_map:
                neuron_bit_map[expert_idx] = {}
            neuron_bit_map[expert_idx][neuron_idx] = bit

    # Step 5: Build per-expert return values
    n_neurons_per_expert = total_neurons // n_experts

    per_expert_scheme = []
    per_expert_neuron_bits = []

    # Split global_scheme into per-expert chunks
    for expert_idx in range(n_experts):
        start = expert_idx * slice_expert_num
        end = start + slice_expert_num
        expert_sub_scheme = global_scheme[start:end]
        per_expert_scheme.append(expert_sub_scheme)

    # Build neuron bits arrays
    for expert_idx in range(n_experts):
        neuron_bits = np.zeros(n_neurons_per_expert, dtype=int)
        if expert_idx in neuron_bit_map:
            for neuron_idx, bit in neuron_bit_map[expert_idx].items():
                neuron_bits[neuron_idx] = bit
        per_expert_neuron_bits.append(neuron_bits)

    return per_expert_scheme, per_expert_neuron_bits


#---- Test ----

def test_read_rates_from_file():
    outlier_bits = {1, 2, 3, 4}
    print(f"simulate quant outlier_bits {outlier_bits}")

    model_id = "deepseek-v1-moe-16b"
    layer_idx = 1
    cache_dir = f"quant_outlier_/{model_id}"

    p = 20
    rates = {}
    for x in outlier_bits:
        cache_path = os.path.join(cache_dir, f"{model_id}_L{layer_idx}_b{x}.pt")
        if os.path.exists(cache_path):
            try:
                import torch
                cached_data = torch.load(cache_path, map_location='cpu')
                print(f"Loading cached quant outlier data for layer {layer_idx}, wbits={x}")
                rates[x] = [cached_data[0][:p]]
            except Exception as e:
                print(f"Failed to load cached data: {e}")

    rates[0] = extrapolate_0bit_loss(rates, save_plots=True)
    for i in range(p):
        print(i, end=',')
        print(f"{rates[4][0][i].item():.4f}", end=',')
        print(f"{rates[3][0][i].item():.4f}", end=',')
        print(f"{rates[2][0][i].item():.4f}", end=',')
        print(f"{rates[1][0][i].item():.4f}", end=',')
        print(f"{rates[0][0][i].item():.4f}", end=',')
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
    best_scheme, neuron_bits_fast = enum_optimal_m_scheme_fast_general(
        rates, s, target_bpw, epsilon
    )
    print(f"Fast m-scheme Search Time: {time.time() - tick:.4f} s")

def test_global_dp_utils():
    """Test global DP mode with multiple experts"""
    np.random.seed(42)
    n_neurons_per_expert = 1024
    n_experts = 8
    slice_expert_num = 4

    bits = [2, 3, 4]

    # Generate random expert activation rates
    expert_activation_rates = np.random.rand(n_experts)
    expert_activation_rates = expert_activation_rates / expert_activation_rates.sum()

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

    target_bpw = 2.5
    epsilon = 0.1

    print(f"Global DP Config: n_experts={n_experts}, n_neurons_per_expert={n_neurons_per_expert}")
    print(f"  slice_expert_num={slice_expert_num}, bits={bits}, target_bpw={target_bpw}, epsilon={epsilon}")

    print(f"  expert_activation_rates (top 3): {np.sort(expert_activation_rates)[::-1][:3]}")

    print("\n--- Global DP Search ---")
    tick = time.time()
    per_expert_scheme, per_expert_neuron_bits = enum_optimal_m_scheme_global_fast(
        expert_rates_list, expert_activation_rates, slice_expert_num, target_bpw, epsilon
    )
    elapsed = time.time() - tick
    print(f"Global DP Search Time: {elapsed:.4f} s")

    # Rebuild global scheme to verify it's non-increasing
    global_scheme = []
    for expert_scheme in per_expert_scheme:
        global_scheme.extend(expert_scheme)

    is_non_increasing = all(global_scheme[i] >= global_scheme[i+1] for i in range(len(global_scheme)-1))
    print(f"Global scheme is non-increasing: {is_non_increasing}")

    # Verify each expert's scheme is non-increasing
    all_experts_non_increasing = True
    for expert_idx, expert_scheme in enumerate(per_expert_scheme):
        is_expert_non_increasing = all(expert_scheme[i] >= expert_scheme[i+1] for i in range(len(expert_scheme)-1))
        if not is_expert_non_increasing:
            all_experts_non_increasing = False
            print(f"Expert {expert_idx} scheme NOT non-increasing: {expert_scheme}")

    print(f"All experts' schemes are non-increasing: {all_experts_non_increasing}")

    # Verify and print stats
    print(f"\nPer-expert sub-expert schemes:")
    for expert_idx in range(min(3, n_experts)):
        print(f"  Expert {expert_idx}: {per_expert_scheme[expert_idx]}")

    # Count bit distribution
    bit_counts = {}
    for bit in global_scheme:
        bit_counts[bit] = bit_counts.get(bit, 0) + 1
    print(f"Bit distribution in global scheme: {dict(sorted(bit_counts.items()))}")

    # Build per-expert stats
    expert_bit_counts = [{} for _ in range(n_experts)]
    for expert_idx in range(n_experts):
        neuron_bits = per_expert_neuron_bits[expert_idx]
        for bit in neuron_bits:
            bit = int(bit)
            expert_bit_counts[expert_idx][bit] = expert_bit_counts[expert_idx].get(bit, 0) + 1

    print("\nBit distribution per expert:")
    for expert_idx in range(min(3, n_experts)):
        print(f"  Expert {expert_idx}: {dict(sorted(expert_bit_counts[expert_idx].items()))}")


if __name__ == "__main__":
    # test_extrapolate_0bit_loss()
    # test_read_rates_from_file()
    # test_dp_utils()
    test_global_dp_utils()
