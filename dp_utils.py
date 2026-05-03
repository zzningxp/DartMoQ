
import time
import numpy as np

def neuron_level_dp(rate_x, target_bpw, s = None, epsilon = 0):
    bits = list(rate_x.keys())
    print(bits)
    n_neurons = len(rate_x[bits[0]])
    assert 2.0 <= target_bpw <= 4.0, ""

    for b in bits:
        rate_x[b] = rate_x[b] / sum(rate_x[b])

    target_total_w = round(target_bpw * n_neurons)
    min_total_w = 2 * n_neurons
    max_total_w = 4 * n_neurons
    print(f"target: {n_neurons} {target_bpw} {target_total_w} {min_total_w} {max_total_w}")
    # target_total_w = np.clip(target_total_w, min_total_w, max_total_w)
    
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
            
            for bit in bits:
                w_curr = w_prev + bit - 2
                if w_curr <= max_offset_w:
                    new_loss = prev_dp[w_prev] + rate_x[bit][i]
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
        raise ValueError("NO feasible solution found, please check target_bpw")
        
    actual_total_w = best_w + offset
    actual_bpw = actual_total_w / n_neurons
    
    neuron_bits = np.zeros(n_neurons, dtype=int)
    current_w = best_w
    for i in reversed(range(n_neurons)):
        choice = choice_history[i][current_w]
        neuron_bits[i] = choice
        current_w -= choice - 2

    print("neuron_bits:", neuron_bits)
    print(f"2bit={np.sum(neuron_bits==2)}, "
          f"3bit={np.sum(neuron_bits==3)}, "
          f"4bit={np.sum(neuron_bits==4)}")
    print("actual_bpw:", actual_bpw)
    print("total_loss:", best_loss)
    print("target_bpw:", target_bpw) 
    print("n_neurons:", n_neurons)

def generate_valid_m_schemes(s, target_bpw, epsilon = 0):
    """
    生成所有满足bpw约束的【单调不增】规整m-scheme
    单调不增：bit数只能保持不变或降低，不能升高 (4 >= 3 >= 2)
    """
    target_total = target_bpw * s
    min_total, max_total = 2 * s, 4 * s
    target_total = np.clip(target_total, min_total, max_total)
    valid_schemes = []
    
    def backtrack(pos, curr_total, curr_scheme, last_bit):
        if pos == s:
            if abs(curr_total - target_total) <= epsilon * s:
                valid_schemes.append(tuple(curr_scheme))
            return
        
        remaining = s - pos
        if curr_total + 2 * remaining > target_total + epsilon * s:
            return
        if curr_total + 4 * remaining < target_total - epsilon * s:
            return

        max_possible_bit = last_bit
        min_possible_bit = 2
        
        for bit in range(max_possible_bit, min_possible_bit - 1, -1):
            backtrack(pos + 1, curr_total + bit, curr_scheme + [bit], bit)
    
    for first_bit in [4, 3, 2]:
        backtrack(1, first_bit, [first_bit], first_bit)
    
    return valid_schemes

def evaluate_scheme(scheme, rates):
    """
    【动态规划版】针对一个特定的scheme，用DP计算全局最优神经元分配、排序和总损失
    参数：
        scheme: m-scheme元组，如(3,3,2,2,4,4,3,3)
        rates: {2: arr, 3: arr, 4: arr}
    返回：
        (total_loss, sorted_idx, neuron_bits)
    """
    
    n_neurons = len(rates[2])
    s = len(scheme)
    assert n_neurons % s == 0, "神经元数必须能被分块数整除"
    m = n_neurons // s  # 每块大小
    
    # 1. 统计该scheme中2/3/4bit块的数量
    n2 = scheme.count(2)
    n3 = scheme.count(3)
    n4 = scheme.count(4)
    
    # 2. 计算需要分配给各bit的神经元总数
    K2 = n2 * m
    K3 = n3 * m
    K4 = n4 * m
    
    # 3. 提取rate数组
    r2 = rates[2]
    r3 = rates[3]
    r4 = rates[4]
    
    # ============== 4. 动态规划初始化 ==============
    INF = float('inf')
    # dp[k4][k3] = 选k4个4bit、k3个3bit时的最小总损失
    dp = np.full((K4 + 1, K3 + 1), INF)
    dp[0, 0] = 0.0  # 初始状态
    
    # 回溯路径保存：
    # prev_choice[k4][k3] = (prev_k4, prev_k3, choice, neuron_idx)
    # 注意：这里我们需要保存是处理哪个神经元时做的选择，方便回溯
    # 所以我们改用三维数组：prev_choice[i][k4][k3] 表示处理完第i个神经元后的状态
    # 为了节省空间，我们用两个二维数组交替，或者直接保存完整路径
    # 这里为了简单可靠，我们保存完整的三维回溯表
    prev_choice = [[[None for _ in range(K3 + 1)] for __ in range(K4 + 1)] for ___ in range(n_neurons + 1)]
    
    # ============== 5. 动态规划主循环 ==============
    for i in range(n_neurons):
        # 先把上一个状态复制过来（默认选择2bit，不改变k4,k3）
        # 注意：这里不能直接复制，必须显式转移
        for curr_k4 in range(K4 + 1):
            for curr_k3 in range(K3 + 1):
                if dp[curr_k4][curr_k3] == INF:
                    continue
                # 显式转移：选择2bit
                new_loss = dp[curr_k4][curr_k3] + r2[i]
                if new_loss < dp[curr_k4][curr_k3]:
                    dp[curr_k4][curr_k3] = new_loss
                    prev_choice[i+1][curr_k4][curr_k3] = (curr_k4, curr_k3, 0)
        
        # 从后往前遍历，避免覆盖
        for curr_k4 in range(K4, -1, -1):
            for curr_k3 in range(K3, -1, -1):
                if dp[curr_k4][curr_k3] == INF:
                    continue
                
                # 尝试选择3bit (k3+1)
                if curr_k3 < K3:
                    # 我们需要看"如果上一个状态是 (curr_k4, curr_k3)，那么选3bit后到 (curr_k4, curr_k3+1)"
                    # 所以我们需要先找到上一个状态的dp值
                    # 为了避免逻辑混乱，我们改用"滚动数组"的标准写法：
                    # 维护两个dp表：prev_dp和curr_dp
                    # 这里为了修复bug，我们重写DP逻辑，使用双数组
                    pass
    
    # ============== 【完全重写DP逻辑，使用双数组，彻底避免bug】 ==============
    # 重新初始化，使用双数组
    prev_dp = np.full((K4 + 1, K3 + 1), INF)
    prev_dp[0, 0] = 0.0
    # 回溯表：prev_choice[i][k4][k3] = (prev_k4, prev_k3, choice)
    backtrack = [[[None for _ in range(K3 + 1)] for __ in range(K4 + 1)] for ___ in range(n_neurons)]
    
    for i in range(n_neurons):
        curr_dp = np.full((K4 + 1, K3 + 1), INF)
        for k4 in range(K4 + 1):
            for k3 in range(K3 + 1):
                if prev_dp[k4][k3] == INF:
                    continue
                
                # 选择1：分配给2bit
                if curr_dp[k4][k3] > prev_dp[k4][k3] + r2[i]:
                    curr_dp[k4][k3] = prev_dp[k4][k3] + r2[i]
                    backtrack[i][k4][k3] = (k4, k3, 0)
                
                # 选择2：分配给3bit
                if k3 < K3:
                    if curr_dp[k4][k3+1] > prev_dp[k4][k3] + r3[i]:
                        curr_dp[k4][k3+1] = prev_dp[k4][k3] + r3[i]
                        backtrack[i][k4][k3+1] = (k4, k3, 1)
                
                # 选择3：分配给4bit
                if k4 < K4:
                    if curr_dp[k4+1][k3] > prev_dp[k4][k3] + r4[i]:
                        curr_dp[k4+1][k3] = prev_dp[k4][k3] + r4[i]
                        backtrack[i][k4+1][k3] = (k4, k3, 2)
        
        prev_dp = curr_dp
    
    # ============== 6. 回溯得到每个神经元的bit分配（修复版） ==============
    neuron_bits = np.zeros(n_neurons, dtype=int)
    curr_k4, curr_k3 = K4, K3
    
    # 检查是否有解
    if prev_dp[curr_k4][curr_k3] == INF:
        raise ValueError("DP未找到可行解，请检查scheme和rates")
    
    for i in reversed(range(n_neurons)):
        # 获取回溯信息
        bt = backtrack[i][curr_k4][curr_k3]
        if bt is None:
            # 理论上不应该走到这里，除非DP逻辑有误
            # 作为兜底，默认分配2bit
            neuron_bits[i] = 2
            continue
        
        prev_k4, prev_k3, choice = bt
        
        if choice == 0:
            neuron_bits[i] = 2
        elif choice == 1:
            neuron_bits[i] = 3
        elif choice == 2:
            neuron_bits[i] = 4
        
        # 更新状态
        curr_k4, curr_k3 = prev_k4, prev_k3
    
    # ============== 7. 生成最终排序 ==============
    idx = np.arange(n_neurons)
    four_bit_neurons = idx[neuron_bits == 4]
    three_bit_neurons = idx[neuron_bits == 3]
    two_bit_neurons = idx[neuron_bits == 2]
    
    # 同bit内按损失从大到小排序
    four_bit_sorted = four_bit_neurons[np.argsort(-r4[four_bit_neurons])]
    three_bit_sorted = three_bit_neurons[np.argsort(-r3[three_bit_neurons])]
    two_bit_sorted = two_bit_neurons[np.argsort(-r2[two_bit_neurons])]
    
    sorted_idx = np.concatenate([four_bit_sorted, three_bit_sorted, two_bit_sorted])
    
    # ============== 8. 计算总损失 ==============
    total_loss = prev_dp[K4, K3]
    
    return total_loss, sorted_idx, neuron_bits

def enum_optimal_m_scheme(rates, s, target_bpw, epsilon = 0):
    assert set(rates.keys()) == {2, 3, 4}, "rates must contain keys 2, 3, 4"
    n_neurons = len(rates[2])
    assert len(rates[3]) == n_neurons and len(rates[4]) == n_neurons, "rate arrays must have the same length"
    
    valid_schemes = generate_valid_m_schemes(s, target_bpw, epsilon)
    print(f"valid_schemes: {valid_schemes}")
    if not valid_schemes:
        raise ValueError(f"no valid scheme found for target_bpw={target_bpw}")
    
    best_total_loss = float('inf')
    best_result = None
    
    for scheme in valid_schemes:
        tick0 = time.time()
        print(f"current scheme: {scheme}")
        loss, sorted_idx, neuron_bits = evaluate_scheme(scheme, rates)
        if loss < best_total_loss:
            best_total_loss = loss
            best_result = {
                'scheme_tuple': scheme,
                'total_loss': loss,
                'sorted_idx': sorted_idx,
                'neuron_bits': neuron_bits,
                'actual_bpw': np.mean(scheme)
            }
        print(f"evaluate scheme {scheme} loss: {loss:.4f} cost: {time.time() - tick0:.4f} s")
    
    m_scheme_str = ''.join([str(b) for b in best_result['scheme_tuple']])
    best_result['m_scheme'] = f'{m_scheme_str}'
    best_result['valid_schemes_count'] = len(valid_schemes)
    best_result['all_valid_schemes'] = valid_schemes
    
    print(f"best mcheme: {best_result['m_scheme']}")
    return  best_result['scheme_tuple'], best_result['neuron_bits']

def get_unified_sorted_idx(rates):
    """
    【核心极速步骤1】一次统一的边际收益排序
    排序标准：综合考虑 gain_4to3 和 gain_3to2
    """
    r2, r3, r4 = rates[2], rates[3], rates[4]
    idx = np.arange(len(r2))
    
    # 计算综合边际收益
    gain_4to3 = r3 - r4  # 用4bit替代3bit的收益
    gain_3to2 = r2 - r3  # 用3bit替代2bit的收益
    # 综合得分：两个收益的加权和
    combined_score = gain_4to3 + gain_3to2
    
    # 按综合得分从大到小排序
    sorted_idx = idx[np.argsort(-combined_score)]
    return sorted_idx

def precompute_block_losses(sorted_idx, rates, s):
    """
    【核心极速步骤2】预计算块损失表
    返回：block_losses[bit_idx][k]，bit_idx=0→2bit,1→3bit,2→4bit
    """
    n_neurons = len(sorted_idx)
    assert n_neurons % s == 0
    m = n_neurons // s
    r2, r3, r4 = rates[2], rates[3], rates[4]
    
    block_losses = np.zeros((3, s))
    for k in range(s):
        start = k * m
        end = start + m
        idx_in_block = sorted_idx[start:end]
        block_losses[0, k] = r2[idx_in_block].sum()
        block_losses[1, k] = r3[idx_in_block].sum()
        block_losses[2, k] = r4[idx_in_block].sum()
    return block_losses

def enum_optimal_m_scheme_fast(rates, s, target_bpw, epsilon = 0):
    """
    【主函数】极速版枚举最优m-scheme
    速度：毫秒级 per 专家
    """
    # 1. 输入检查
    assert set(rates.keys()) == {2, 3, 4}
    n_neurons = len(rates[2])
    assert len(rates[3]) == n_neurons and len(rates[4]) == n_neurons
    
    # 2. 极速步骤1：一次统一排序
    sorted_idx = get_unified_sorted_idx(rates)
    
    # 3. 极速步骤2：预计算块损失表
    block_losses = precompute_block_losses(sorted_idx, rates, s)
    
    # 4. 极速步骤3：枚举单调scheme，查表选最优
    valid_schemes = generate_valid_m_schemes(s, target_bpw, epsilon)
    if not valid_schemes:
        raise ValueError("未找到有效方案")
    
    best_loss = float('inf')
    best_scheme = None
    
    for scheme in valid_schemes:
        total_loss = 0.0
        for k, bit in enumerate(scheme):
            total_loss += block_losses[bit - 2, k]
        if total_loss < best_loss:
            best_loss = total_loss
            best_scheme = scheme
        print(f"current scheme: {scheme} loss: {total_loss:.4f}")
    
    # 5. 生成结果
    m_scheme_str = ''.join([str(b) for b in best_scheme])
    actual_bpw = np.mean(best_scheme)
    
    m = n_neurons // s
    neuron_bits = np.zeros(n_neurons, dtype=int)
    for k, bit in enumerate(best_scheme):
        start = k * m
        end = start + m
        neuron_bits[sorted_idx[start:end]] = bit
    
    print(f"best scheme: {best_scheme}")
    return  best_scheme, neuron_bits


if __name__ == "__main__":
    # 1. 模拟生成rates dict
    np.random.seed(42)
    n_neurons = 1024
    r2 = np.random.rand(n_neurons)
    r3 = r2 * 0.7 + np.random.rand(n_neurons) * 0.1
    r4 = r3 * 0.7 + np.random.rand(n_neurons) * 0.1
    
    rates = {2: r2, 3: r3, 4: r4}
    
    # 2. 运行主函数
    s = 8
    target_bpw = 2.5
    # result = enum_optimal_m_scheme(rates, s, target_bpw, epsilon=0)
    result = enum_optimal_m_scheme_fast(rates, s, target_bpw, epsilon=0)
