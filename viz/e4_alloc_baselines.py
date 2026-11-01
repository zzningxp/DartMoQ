"""E4 同预算分配基线（rebuttal W5 可加性 / W4 单调约束代价的 proxy 侧）。

一个文件两种用法，均不改主流程代码：

  1) replay —— E4a：零前向 DP 回放（纯 CPU，读 intermediate_result 缓存）
     逐层重放全局单调 DP 得到 proxy objective，与 papers/e1_zero_bit_source_tables.md
     里已有的 PPL 配对，做「DP 目标 vs PPL」相关性分析（W5 证据链）；
     同时计算同 raw 预算的 uniform / random / unconstrained(MCKP 精确) 三种
     基线的 proxy（E6 的 proxy 侧差距预览）。
       python e4_alloc_baselines.py replay                 # 全部格（断点续跑）
       python e4_alloc_baselines.py replay --models olmoe-7b-1b --bpws 1.0
       python e4_alloc_baselines.py replay --summary-only  # 只重算相关性不回放

  2) run —— E4b：端到端基线（GPU，由 run.e4.sh 批量调；手动单跑也行）
     通过替换 dartmoq_layer_reconstruct 命名空间里的 DP 入口，在主流程里
     注入 --alloc-scheme {dp,uniform,random,unconstrained}，其余 argv 原样
     传给 run_dartmoq.py，协议与 run.zb.sh 完全一致：
       python e4_alloc_baselines.py run --alloc-scheme uniform \
           /path/to/model wikitext2 --slices 8 --nsamples 64 \
           --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.0 \
           --quantmode turboquant --zero-bit-source extrapolate --standby-layer-cpu

  辅助：
       python e4_alloc_baselines.py verify    # 快速 DP vs 主流程 canonical DP（真实缓存层）
       python e4_alloc_baselines.py selftest  # 合成小规模穷举校验 MCKP/基线/快速 DP

设计要点：三种基线全部与 dp 模式逐层对齐同 raw 预算（取 canonical DP 解的
raw 总 bit 数）：uniform 用原始 (expert, sub) 顺序的低/高 bit 确定性切分（不
用任何敏感度信息）；random 打乱 DP 解的 bit 多重集（同预算同直方图，只随机
"谁拿哪个 bit"）；unconstrained 在同 raw 预算下做无排序无单调约束的精确
multiple-choice knapsack（E6 oracle，向量化 ~0.1s/层）。packed mode（bits
非连续）不支持 uniform 的整 bit 切分，会显式报错。
"""

import os
import re
import csv
import sys
import glob
import time
import argparse
import runpy
import numpy as np

# viz 约定：脚本可从仓库根 `python viz/e4_alloc_baselines.py` 直跑，
# 根目录模块（dp_utils 及 runpy 顺跑的 run_dartmoq.py 的导入）由此可达
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dp_utils import get_unified_sorted_idx_general

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E1_TABLES = os.path.join(REPO, 'papers', 'e1_zero_bit_source_tables.md')
OUT_CSV = os.path.join(REPO, 'logs', 'e4_proxy_objectives.csv')
OUT_SUMMARY = os.path.join(REPO, 'logs', 'e4_correlation_summary.txt')
# 对齐版（分配级货币换算到 direct 价格）写独立 CSV，不覆盖原始 E4a 结果
OUT_CSV_ALIGNED = os.path.join(REPO, 'logs', 'e4_proxy_objectives_aligned.csv')

CACHE_RATES = os.path.join(REPO, 'intermediate_result', 'quant_outlier_turboquant', 'turboquant_innerproduct')
CACHE_ACT = os.path.join(REPO, 'intermediate_result', 'expert_activate')

# E1 五模型（与 run.zb.sh / md 表一致；model_id 即缓存目录名）
E1_MODELS = ['deepseek-v2-lite', 'deepseek-v1-moe-16b', 'moonlight', 'olmoe-7b-1b', 'qwen3-30b-a3b']

INF = float('inf')


# =====================================================================
# 共享的块准备（镜像 dp_utils 全局 DP 的 Step1-3，全部 numpy 化）
# =====================================================================

def expert_block_prep(expert_rates_list, act_rates, bits, slice_expert_num):
    """每专家统一排序切 s 个连续 sub-expert（与主流程 get_unified_sorted_idx_general
    同一排序口径），产出两张块损失表：
      losses_sorted: (B, T) 列序 = 全局重要度降序（单调 DP 用，须与主流程一致）
      losses_orig:   (B, T) 列序 = 原始 (expert, sub) 顺序（uniform/random/MCKP 用，
                     MCKP 与列序无关）
    重要度口径与主流程一致：act_e * mean(最低 bit 的 loss)。
    返回 dict。
    """
    bits = list(bits)
    E = len(expert_rates_list)
    s = slice_expert_num
    blocks = []            # blocks[e][sub] = 该专家排序序下的神经元索引数组
    losses_per_e = []      # losses_per_e[e] = (B, s)
    scores = []            # (e, sub, -importance) 与主流程同构造顺序
    for e in range(E):
        rates = expert_rates_list[e]
        sorted_idx = get_unified_sorted_idx_general(rates, bits)
        n = len(sorted_idx)
        m = n // s
        blk_losses = np.zeros((len(bits), s))
        sub_neurons = []
        for sub in range(s):
            neurons = sorted_idx[sub * m:(sub + 1) * m]
            sub_neurons.append(neurons)
            for bi, b in enumerate(bits):
                # 与主流程一致：块损失 = 该块该 bit 的 loss 之和 × 专家激活率
                blk_losses[bi, sub] = float(rates[b][neurons].sum()) * act_rates[e]
        blocks.append(sub_neurons)
        losses_per_e.append(blk_losses)
        low = rates[bits[0]] if len(bits) > 1 else None
        for sub in range(s):
            if len(bits) == 1:
                score = 1.0
            else:
                score = float(low[sub_neurons[sub]].mean())
            scores.append((e, sub, -(score * act_rates[e])))

    scores.sort(key=lambda x: x[2])          # 稳定排序，与主流程一致
    order_sorted = [(e, sub) for e, sub, _ in scores]
    T = E * s
    losses_sorted = np.zeros((len(bits), T))
    losses_orig = np.zeros((len(bits), T))
    for pos, (e, sub) in enumerate(order_sorted):
        losses_sorted[:, pos] = losses_per_e[e][:, sub]
    for e in range(E):
        for sub in range(s):
            losses_orig[:, e * s + sub] = losses_per_e[e][:, sub]
    return dict(bits=bits, blocks=blocks, order_sorted=order_sorted,
                losses_sorted=losses_sorted, losses_orig=losses_orig,
                E=E, T=T, s=s)


def proxy_of(flat_bits, losses, bit_to_idx):
    """给定按 losses 列序的每块 bit，算 act 加权总 proxy loss。"""
    idx = np.array([bit_to_idx[b] for b in flat_bits])
    return float(losses[idx, np.arange(losses.shape[1])].sum())


# =====================================================================
# 快速向量化单调 DP —— 精确复刻 dp_utils.enum_optimal_m_scheme_global_fast
# (enable_0bit_compensation=True 路径) 的搜索与口径：scale=20、effective
# ledger (bit>0 记 bit+overhead)、target=(bpw+overhead)*T、epsilon 窗口。
# 仅用于 replay/verify（CPU 分析），主流程仍走 canonical 函数。
# =====================================================================

def fast_monotone_dp(prep, target_bpw, epsilon, quant_overhead=0.25):
    losses = prep['losses_sorted']
    bits = prep['bits']
    T = prep['T']
    bits_desc = sorted(bits, reverse=True)
    bit_to_idx = {b: i for i, b in enumerate(bits)}
    B = len(bits_desc)
    scale = 20
    w_add = np.array([int((0.0 if b == 0 else b + quant_overhead) * scale) for b in bits_desc])
    W = int(((max(bits) + quant_overhead) if max(bits) > 0 else 0) * T * scale)
    target_w = int((target_bpw + quant_overhead) * T * scale)
    eps_w = int(epsilon * T * scale)

    dp = np.full((W + 1, B), INF)
    for j in range(B):
        w = int(w_add[j])
        if w <= W:
            dp[w, j] = losses[bit_to_idx[bits_desc[j]], 0]
    # choice[k, w, j] = 转移到 (w, j) 时的 prev bit 下标（desc 序）；-1 不可达
    choice = np.full((T, W + 1, B), -1, dtype=np.int8)

    for k in range(1, T):
        ndp = np.full((W + 1, B), INF)
        for j_prev in range(B):                     # 与 canonical 同序：b_prev 升序(=bit 降序)
            src = dp[:, j_prev]
            fin = np.isfinite(src)
            if not fin.any():
                continue
            for j in range(j_prev, B):              # 单调约束：b_curr <= b_prev
                c = int(w_add[j])
                lk = losses[bit_to_idx[bits_desc[j]], k]
                if c == 0:
                    cand = np.where(fin, src + lk, INF)
                    cur = ndp[:, j]
                    m = cand < cur
                    cur[m] = cand[m]
                    choice[k][m, j] = j_prev
                else:
                    cand = np.where(fin[:W + 1 - c], src[:W + 1 - c] + lk, INF)
                    cur = ndp[c:, j]
                    m = cand < cur
                    cur[m] = cand[m]
                    choice[k][c:][m, j] = j_prev
        dp = ndp

    lo, hi = max(0, target_w - eps_w), min(W, target_w + eps_w)
    window = dp[lo:hi + 1]
    if not np.isfinite(window).any():
        raise ValueError(f"fast DP: no feasible w in [{lo},{hi}] "
                         f"(target_bpw={target_bpw}, T={T})")
    # 展平 row-major argmin = 与 canonical 逐 w 再逐 b 的严格 < 扫描同 tie-break
    flat = int(np.argmin(window))
    w_best, j_best = lo + flat // B, flat % B
    best_loss = float(window[w_best - lo, j_best])

    scheme = [0] * T
    w, j = w_best, j_best
    scheme[T - 1] = bits_desc[j]
    for k in range(T - 1, 0, -1):
        j_prev = int(choice[k, w, j])
        w -= int(w_add[j])
        j = j_prev
        scheme[k - 1] = bits_desc[j]
        if j_prev < 0:
            raise ValueError("fast DP backtrack hit unreachable state")
    return best_loss, scheme


# =====================================================================
# 同 raw 预算基线：uniform / random / unconstrained(MCKP)
# =====================================================================

def mckp_at_budget(losses, bits, budget):
    """无约束精确 MCKP（列序任意）：每块独立任选 bit，raw 总 bit == budget。
    向量化 O(T*W*B)。返回 (best_loss, flat_bits)。"""
    bit_to_idx = {b: i for i, b in enumerate(bits)}
    B, T = losses.shape
    W = int(budget)
    dp = np.full(W + 1, INF)
    dp[0] = 0.0
    choice = np.full((T, W + 1), -1, dtype=np.int8)
    for k in range(T):
        ndp = np.full(W + 1, INF)
        for bi in range(B):
            c = bits[bi]                     # raw ledger：cost = bit 值
            lk = losses[bi, k]
            if c == 0:
                cand = np.where(np.isfinite(dp), dp + lk, INF)
                m = cand < ndp
                ndp[m] = cand[m]
                choice[k][m] = bi
            else:
                cand = np.where(np.isfinite(dp[:W + 1 - c]), dp[:W + 1 - c] + lk, INF)
                m = cand < ndp[c:]
                ndp[c:][m] = cand[m]
                choice[k][c:][m] = bi
        dp = ndp
    if not np.isfinite(dp[W]):
        raise ValueError(f"MCKP: budget {W} unreachable")
    flat = [0] * T
    w = W
    for k in range(T - 1, -1, -1):
        bi = int(choice[k, w])
        flat[k] = bits[bi]
        w -= bits[bi]
    return float(dp[W]), flat


def uniform_at_budget(bits, T, raw_total):
    """无敏感度信息：原始顺序前 k 块拿高 bit、其余拿低 bit，raw 总数严格对齐。
    要求 bits 为连续整数（E1/E4 协议 bits={0..4} 满足；packed mode 会在此报错）。"""
    bits = sorted(bits)
    if bits != list(range(bits[0], bits[0] + len(bits))):
        raise ValueError(f"uniform mode needs contiguous bits, got {bits} "
                         "(packed mode unsupported for E4)")
    bpw_per_block = raw_total / T
    lo = int(np.floor(bpw_per_block))
    lo = min(max(lo, bits[0]), bits[-1])
    hi = min(lo + 1, bits[-1])
    k = int(round(raw_total - lo * T))
    if k * (hi - lo) + lo * T != raw_total or k > T:
        raise ValueError(f"uniform split infeasible: T={T} raw={raw_total} lo={lo} hi={hi} k={k}")
    flat = [hi] * k + [lo] * (T - k)
    return flat


def permute_multiset(flat_bits, seed):
    rng = np.random.default_rng(seed)
    out = np.array(flat_bits, dtype=int)
    rng.shuffle(out)
    return [int(b) for b in out]


def deploy_flat_to_per_expert(flat_orig, prep):
    """原始 (e,s) 顺序的 flat bit -> 主流程同款返回契约
    (per_expert_scheme, per_expert_neuron_bits)。"""
    E, s = prep['E'], prep['s']
    per_expert_scheme = []
    per_expert_neuron_bits = []
    for e in range(E):
        scheme_e = [int(flat_orig[e * s + sub]) for sub in range(s)]
        per_expert_scheme.append(scheme_e)
        n = sum(len(b) for b in prep['blocks'][e])
        neuron_bits = np.zeros(n, dtype=int)
        for sub in range(s):
            for neuron_idx in prep['blocks'][e][sub]:
                neuron_bits[neuron_idx] = scheme_e[sub]
        per_expert_neuron_bits.append(neuron_bits)
    return per_expert_scheme, per_expert_neuron_bits


def scheme_sorted_to_flat_orig(scheme_sorted, prep):
    """sorted 列序的 DP 解 -> 原始 (e,s) 顺序 flat（便于与基线同口径算 proxy/预算）。"""
    flat = [0] * prep['T']
    for pos, (e, sub) in enumerate(prep['order_sorted']):
        flat[e * prep['s'] + sub] = scheme_sorted[pos]
    return flat


def eval_alloc_under_rates(flat_bits, prep, erl_rates, act_rates):
    """分配级货币换算：(e,s) 列序 flat bit -> erl_rates 价格下的总损失。

    块划分沿用 prep['blocks']（source 价格下的排序切块，块内神经元同 bit），
    只把价格表换成 erl_rates（典型为 direct）：
      aligned = Σ_e act_e * Σ_sub Σ_neuron erl_rates[e][b_sub][neuron]
    与 proxy_of 同口径（含 n mod s 余数神经元不计入的既有约定）。
    """
    s = prep['s']
    tot = 0.0
    for e, rates in enumerate(erl_rates):
        for sub in range(s):
            b = int(flat_bits[e * s + sub])
            neurons = prep['blocks'][e][sub]
            if len(neurons):
                tot += float(rates[b][neurons].sum()) * act_rates[e]
    return tot


def alloc_layer_all_modes(expert_rates_list, act_rates, bits, slice_expert_num,
                          target_bpw, epsilon, seed, quant_overhead=0.25):
    """一层四种模式的分配与 proxy（replay 与 run 共用）。
    返回 dict: raw_total, proxy_dp/uniform/random/unconstrained,
    flat_dp/flat_uniform/flat_random/flat_unc（均按原始 (e,s) 列序）。"""
    prep = expert_block_prep(expert_rates_list, act_rates, bits, slice_expert_num)
    bit_to_idx = {b: i for i, b in enumerate(prep['bits'])}
    loss_dp, scheme_sorted = fast_monotone_dp(prep, target_bpw, epsilon, quant_overhead)
    flat_dp = scheme_sorted_to_flat_orig(scheme_sorted, prep)
    raw_total = int(sum(flat_dp))

    flat_u = uniform_at_budget(prep['bits'], prep['T'], raw_total)
    loss_u = proxy_of(flat_u, prep['losses_orig'], bit_to_idx)

    flat_r = permute_multiset(flat_dp, seed)
    loss_r = proxy_of(flat_r, prep['losses_orig'], bit_to_idx)

    loss_c, flat_c = mckp_at_budget(prep['losses_orig'], prep['bits'], raw_total)

    return dict(prep=prep, raw_total=raw_total,
                proxy_dp=proxy_of(flat_dp, prep['losses_orig'], bit_to_idx),
                proxy_uniform=loss_u, proxy_random=loss_r, proxy_unconstrained=loss_c,
                flat_dp=flat_dp, flat_uniform=flat_u, flat_random=flat_r,
                flat_unconstrained=flat_c)


# =====================================================================
# E4b：run 模式 —— 注入 --alloc-scheme 后原样执行 run_dartmoq.py
# =====================================================================

def run_e4b(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--alloc-scheme', default='dp',
                        choices=['dp', 'uniform', 'random', 'unconstrained'])
    parser.add_argument('--alloc-seed', type=int, default=42)
    e4_args, rest = parser.parse_known_args(argv)
    if not rest or rest[0] in ('-h', '--help'):
        print(__doc__)
        return 0

    os.environ['DARTMOQ_ALLOC_SCHEME'] = e4_args.alloc_scheme
    os.environ['DARTMOQ_ALLOC_SEED'] = str(e4_args.alloc_seed)

    import dartmoq_layer_reconstruct as dlr
    orig_dp = dlr.enum_optimal_m_scheme_global_fast
    call_counter = {'n': 0}

    def patched(expert_rates_list, expert_activation_rates, slice_expert_num,
                target_bpw, epsilon=0, enable_0bit_compensation=False,
                quant_overhead=0.25, _orig=orig_dp):
        mode = os.environ.get('DARTMOQ_ALLOC_SCHEME', 'dp')
        # canonical DP 先跑：确定逐层 raw 预算锚点（random 还要它的直方图），
        # 并保持 dp 模式日志与主流程完全一致
        scheme_dp, nb_dp = _orig(expert_rates_list, expert_activation_rates,
                                 slice_expert_num, target_bpw, epsilon,
                                 enable_0bit_compensation, quant_overhead)
        if mode == 'dp':
            return scheme_dp, nb_dp

        act = []
        for a in expert_activation_rates:
            if hasattr(a, 'detach'):
                act.append(float(a.detach().cpu().numpy()))
            elif hasattr(a, 'item'):
                act.append(float(a.item()))
            else:
                act.append(float(a))
        bits = sorted(expert_rates_list[0].keys())
        seed = int(os.environ.get('DARTMOQ_ALLOC_SEED', 42)) + call_counter['n']
        call_counter['n'] += 1

        res = alloc_layer_all_modes(expert_rates_list, act, bits, slice_expert_num,
                                    target_bpw, epsilon, seed, quant_overhead)
        flat = res[f'flat_{mode}']
        eff = sum((b + quant_overhead) if b > 0 else 0.0 for b in flat) / len(flat)
        print(f"[e4-alloc] mode={mode} layer#{call_counter['n'] - 1} seed={seed} "
              f"proxy_loss={res['proxy_' + mode]:.4f} raw_bpw={res['raw_total'] / len(flat):.3f} "
              f"eff_bpw={eff:.3f} (dp_proxy={res['proxy_dp']:.4f}, "
              f"uniform={res['proxy_uniform']:.4f}, unc={res['proxy_unconstrained']:.4f})",
              flush=True)
        return deploy_flat_to_per_expert(flat, res['prep'])

    dlr.enum_optimal_m_scheme_global_fast = patched
    sys.argv = ['run_dartmoq.py'] + rest
    runpy.run_path(os.path.join(REPO, 'run_dartmoq.py'), run_name='__main__')
    return 0


# =====================================================================
# E4a：replay —— 缓存回放 + 与 E1 PPL 配对的相关性
# =====================================================================

def cell_ppl_fn(c):
    c = c.replace('*', '').strip()   # 粗体可能标在单元格内部（如 "x / **y**"）
    mm = re.match(r'^([\d.]+)\s*/\s*([\d.]+)$', c)
    return (float(mm.group(1)), float(mm.group(2))) if mm else None


def parse_e1_tables(path):
    """解析 papers/e1_zero_bit_source_tables.md -> {(model, source, bpw): (wiki, c4)}。
    兼容两种表：'## model' 节下的 bpw 行表（列 = source）；'模型 @x.xbpw' 的
    模型行表（L0 价格扫描：×N 列映射 direct-scaleN，direct/extrap/none/magnitude
    列与 E1 主表重复，跳过）。"""
    colmap = {'extrapolate': 'extrapolate', 'extrap(≈×1.6)': 'extrapolate',
              'direct': 'direct', 'none': 'none', 'magnitude': 'magnitude',
              'activation': 'activation', '×10': 'direct-scale10',
              '×100': 'direct-scale100', '×0.1': 'direct-scale0.1'}
    cells = {}
    dup = 0
    text = open(path).read()
    sections = re.split(r'^##\s+', text, flags=re.M)
    for sec in sections:
        lines = [l for l in sec.splitlines() if l.strip()]
        if not lines:
            continue
        title = lines[0].strip()
        sec_model = title if title in E1_MODELS else None
        hm = re.search(r'@([\d.]+)\s*bpw', title)
        sec_bpw = float(hm.group(1)) if hm else None
        header = None
        for line in lines[1:]:
            cols = [c.strip() for c in line.strip('|').split('|')] if '|' in line else []
            if not cols:
                continue
            if cols[0] == 'bpw':
                header = [colmap.get(c) for c in cols[1:]]
                continue
            if cols[0].startswith('模型'):
                header = [colmap.get(c) for c in cols[1:]]
                # '模型 @1.0bpw' 的 bpw 标在列头而非节标题
                hm2 = re.search(r'@([\d.]+)\s*bpw', cols[0])
                if hm2 and sec_bpw is None:
                    sec_bpw = float(hm2.group(1))
                continue
            if header is None:
                continue
            if sec_model is not None:
                # bpw 行表：行首 bpw，列 = sources
                try:
                    bpw = float(cols[0])
                except ValueError:
                    continue
                for src, c in zip(header, cols[1:]):
                    v = cell_ppl_fn(c)
                    if src and v:
                        key = (sec_model, src, bpw)
                        if key in cells:
                            dup += 1
                        cells[key] = v
            elif cols[0] in E1_MODELS and sec_bpw is not None:
                # 模型行表（L0 价格扫描）：direct/extrap/none/magnitude 列与
                # E1 主表重复，只收 direct-scaleN
                for src, c in zip(header, cols[1:]):
                    v = cell_ppl_fn(c)
                    if src and v and src.startswith('direct-scale'):
                        cells[(cols[0], src, sec_bpw)] = v
    print(f"parsed {len(cells)} PPL cells from {os.path.basename(path)} ({dup} in-section dups overwritten)")
    return cells


def b0_cache_suffix(source):
    if source in ('extrapolate',):
        return '_b0.pt', 1.0
    if source.startswith('direct-scale'):
        n = float(re.fullmatch(r'direct-scale([\d.]+)', source).group(1))
        return '_b0direct.pt', n
    return f'_b0{source}.pt', 1.0


def load_layer_inputs(model_id, layer_idx, source):
    """一层缓存 -> (expert_rates_list, act_rates, bits)。none 源无 b0。"""
    import torch
    rdir = os.path.join(CACHE_RATES, model_id)
    if source == 'none':
        bits = [1, 2, 3, 4]
    else:
        bits = [0, 1, 2, 3, 4]
    rates = {}
    for b in bits:
        if b == 0:
            suffix, scale = b0_cache_suffix(source)
        else:
            suffix, scale = f'_b{b}.pt', 1.0
        p = os.path.join(rdir, f'{model_id}_L{layer_idx}{suffix}')
        if not os.path.exists(p):
            return None
        arrs = torch.load(p, map_location='cpu')
        rates[b] = [a.float().numpy() * scale for a in arrs]
    act_p = os.path.join(CACHE_ACT, model_id, f'{model_id}_L{layer_idx}.pt')
    if not os.path.exists(act_p):
        return None
    act = torch.load(act_p, map_location='cpu').float().numpy()
    expert_rates_list = [{b: rates[b][e] for b in bits} for e in range(len(act))]
    return expert_rates_list, [float(a) for a in act], bits


def discover_layers(model_id):
    files = glob.glob(os.path.join(CACHE_RATES, model_id, f'{model_id}_L*_b1.pt'))
    return sorted(int(re.search(r'_L(\d+)_b1\.pt$', f).group(1)) for f in files)


def replay_cell(model_id, source, bpw, seed0=42, aligned=False):
    layers = discover_layers(model_id)
    if not layers:
        raise RuntimeError(f'no b1 caches for {model_id}')
    tot = dict(proxy_dp=0.0, proxy_uniform=0.0, proxy_random=0.0,
               proxy_unconstrained=0.0, raw_total=0, n=0)
    if aligned:
        for mode in ('dp', 'uniform', 'random', 'unconstrained'):
            tot[f'proxy_{mode}_aligned'] = 0.0
    t0 = time.time()
    for li, layer in enumerate(layers):
        inputs = load_layer_inputs(model_id, layer, source)
        if inputs is None:
            print(f"  ! missing caches layer {layer}, skipped")
            continue
        erl, act, bits = inputs
        res = alloc_layer_all_modes(erl, act, bits, slice_expert_num=8,
                                    target_bpw=bpw, epsilon=0.0, seed=seed0 + li)
        for k in ('proxy_dp', 'proxy_uniform', 'proxy_random', 'proxy_unconstrained'):
            tot[k] += res[k]
        if aligned:
            # 换算货币：同一批分配（PPL 不变），价格表换成 direct
            din = load_layer_inputs(model_id, layer, 'direct')
            if din is None:
                print(f"  ! missing direct caches layer {layer}, aligned skipped")
                continue
            erl_d = din[0]
            for mode in ('dp', 'uniform', 'random', 'unconstrained'):
                tot[f'proxy_{mode}_aligned'] += eval_alloc_under_rates(
                    res[f'flat_{mode}'], res['prep'], erl_d, act)
        tot['raw_total'] += res['raw_total']
        tot['n'] += 1
    T = tot['n'] * (len(act) * 8)
    tot['raw_bpw_mean'] = tot['raw_total'] / T if T else float('nan')
    tot['seconds'] = time.time() - t0
    return tot


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3:
        return float('nan')
    r = spearmanr(x, y).correlation
    return float(r) if r is not None else float('nan')


def stratified_pooled(rows, proxy_key, ppl_idx):
    """同 (model, bpw) 层内配对排秩后合并 —— 控制预算与模型尺度，
    直接回答"同预算下 proxy 排序是否预测 PPL 排序"。"""
    from scipy.stats import rankdata
    strata = {}
    for r in rows:
        strata.setdefault((r['model'], r['bpw']), []).append(r)
    rx, ry = [], []
    for cells in strata.values():
        if len(cells) < 2:
            continue
        px = [c[proxy_key] for c in cells]
        py = [c[ppl_idx] for c in cells]
        if len(set(px)) < 2 or len(set(py)) < 2:
            continue
        rx += list(rankdata(px))
        ry += list(rankdata(py))
    return spearman(rx, ry), len(rx)


def write_summary(rows):
    lines = []
    lines.append("E4a: DP proxy objective vs end-to-end PPL (rebuttal W5)")
    lines.append(f"cells: {len(rows)}  source: {OUT_CSV}")
    lines.append("")
    for model in sorted({r['model'] for r in rows}):
        rs = [r for r in rows if r['model'] == model]
        lines.append(f"[{model}] n={len(rs)}")
        for key in ('proxy_dp', 'proxy_uniform', 'proxy_random', 'proxy_unconstrained'):
            w = spearman([r[key] for r in rs], [r['ppl_wiki'] for r in rs])
            c = spearman([r[key] for r in rs], [r['ppl_c4'] for r in rs])
            lines.append(f"  {key:20s} spearman vs wiki={w:+.3f}  vs c4={c:+.3f}")
    lines.append("")
    lines.append("stratified (within model x bpw, rank-pooled):")
    for key in ('proxy_dp', 'proxy_uniform', 'proxy_random', 'proxy_unconstrained'):
        for ppl_idx, tag in (('ppl_wiki', 'wiki'), ('ppl_c4', 'c4')):
            rho, n = stratified_pooled(rows, key, ppl_idx)
            lines.append(f"  {key:20s} vs {tag}: rho={rho:+.3f} (n={n})")
    lines.append("")
    lines.append("same-budget proxy gaps (E6 preview, mean over cells):")
    for key, tag in (('proxy_unconstrained', 'unconstrained'), ('proxy_uniform', 'uniform'),
                     ('proxy_random', 'random')):
        gaps = [(r[key] - r['proxy_dp']) / r['proxy_dp'] * 100 for r in rows
                if r['proxy_dp'] > 0]
        lines.append(f"  dp vs {tag:14s}: mean +{np.mean(gaps):.2f}%  "
                     f"median +{np.median(gaps):.2f}%  max +{np.max(gaps):.2f}%")
    txt = '\n'.join(lines)
    with open(OUT_SUMMARY, 'w') as f:
        f.write(txt + '\n')
    print(txt)
    print(f"summary saved to {OUT_SUMMARY}")


def cmd_replay(args):
    cells = parse_e1_tables(E1_TABLES)
    keys = [k for k in cells
            if (not args.models or k[0] in args.models)
            and (not args.bpws or k[2] in args.bpws)
            and (not args.sources or k[1] in args.sources)]
    keys.sort()
    print(f"replay grid: {len(keys)} cells"
          + (" (aligned -> direct currency)" if args.aligned else ""))

    out_csv = OUT_CSV_ALIGNED if args.aligned else OUT_CSV
    fieldnames = ['model', 'source', 'bpw', 'n_layers', 'ppl_wiki', 'ppl_c4',
                  'proxy_dp', 'proxy_uniform', 'proxy_random', 'proxy_unconstrained',
                  'raw_bpw_mean', 'gap_uniform_pct', 'gap_unc_pct']
    if args.aligned:
        fieldnames += [f'proxy_{m}_aligned' for m in
                       ('dp', 'uniform', 'random', 'unconstrained')]

    done = set()
    rows = []
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            for r in csv.DictReader(f):
                rows.append({k: (v if k in ('model', 'source') else float(v)) for k, v in r.items()})
                done.add((r['model'], r['source'], float(r['bpw'])))
        print(f"resume: {len(done)} cells already in {out_csv}")

    write_header = not os.path.exists(out_csv)
    fout = open(out_csv, 'a', newline='')
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    for key in keys:
        if key in done:
            continue
        model, source, bpw = key
        wiki, c4 = cells[key]
        try:
            tot = replay_cell(model, source, bpw, aligned=args.aligned)
        except Exception as e:
            print(f"!! {key} failed: {e}")
            continue
        row = dict(model=model, source=source, bpw=bpw, n_layers=tot['n'],
                   ppl_wiki=wiki, ppl_c4=c4,
                   proxy_dp=tot['proxy_dp'], proxy_uniform=tot['proxy_uniform'],
                   proxy_random=tot['proxy_random'],
                   proxy_unconstrained=tot['proxy_unconstrained'],
                   raw_bpw_mean=tot['raw_bpw_mean'],
                   gap_uniform_pct=(tot['proxy_uniform'] - tot['proxy_dp']) / tot['proxy_dp'] * 100,
                   gap_unc_pct=(tot['proxy_unconstrained'] - tot['proxy_dp']) / tot['proxy_dp'] * 100)
        if args.aligned:
            for m in ('dp', 'uniform', 'random', 'unconstrained'):
                row[f'proxy_{m}_aligned'] = tot[f'proxy_{m}_aligned']
        writer.writerow(row)
        fout.flush()
        rows.append(row)
        tail = (f" aligned_dp={tot['proxy_dp_aligned']:.4g}" if args.aligned else "")
        print(f"[{model} {source} {bpw}] dp={tot['proxy_dp']:.4g} "
              f"unc={tot['proxy_unconstrained']:.4g} uniform={tot['proxy_uniform']:.4g} "
              f"random={tot['proxy_random']:.4g}{tail} "
              f"raw_bpw={tot['raw_bpw_mean']:.3f} "
              f"({tot['seconds']:.0f}s, {tot['n']} layers)")
    fout.close()
    if args.aligned:
        # 对齐版不写旧 summary（分析输出在 test/test_e4_monotone.py，两版对照）
        print(f"aligned replay done: {out_csv} "
              f"(analysis: conda run -n dart312 python test/test_e4_monotone.py)")
    else:
        write_summary(rows)


def cmd_summary_only(_args):
    rows = []
    with open(OUT_CSV) as f:
        for r in csv.DictReader(f):
            rows.append({k: (v if k in ('model', 'source') else float(v)) for k, v in r.items()})
    write_summary(rows)


# =====================================================================
# verify / selftest
# =====================================================================

def cmd_verify(args):
    """真实缓存层上：fast DP vs canonical dp_utils DP（loss 必须一致，scheme 报告一致性）。"""
    from dp_utils import enum_optimal_m_scheme_global_fast
    models = args.models or ['olmoe-7b-1b', 'deepseek-v1-moe-16b']
    n_layers = args.layers or 1
    ok = True
    for model in models:
        for layer in discover_layers(model)[:n_layers]:
            for source, bpw in (('extrapolate', 1.0), ('direct', 0.75)):
                inputs = load_layer_inputs(model, layer, source)
                if inputs is None:
                    continue
                erl, act, bits = inputs
                t0 = time.time()
                scheme_c, nb_c = enum_optimal_m_scheme_global_fast(
                    erl, act, 8, target_bpw=bpw, epsilon=0.0,
                    enable_0bit_compensation=True)
                t_c = time.time() - t0
                res = alloc_layer_all_modes(erl, act, bits, 8, bpw, 0.0, seed=0)
                # canonical 的 proxy / raw 从其返回值重算
                loss_c = 0.0
                raw_c = 0
                for e, (scheme_e, nb_e) in enumerate(zip(scheme_c, nb_c)):
                    raw_c += sum(scheme_e)
                    for b in bits:
                        loss_c += float(erl[e][b][nb_e == b].sum()) * act[e]
                rel = abs(loss_c - res['proxy_dp']) / max(abs(loss_c), 1e-12)
                hist_same = raw_c == res['raw_total']
                # rel 差来自 float32 求和顺序（numpy vs 逐个累加），1e-6 已远小于任何有意义差距
                status = 'OK' if rel < 1e-6 and hist_same else 'MISMATCH'
                if status != 'OK':
                    ok = False
                print(f"[{model} L{layer} {source}@{bpw}] canonical={loss_c:.6g} "
                      f"fast={res['proxy_dp']:.6g} rel={rel:.2e} raw_match={hist_same} "
                      f"canonical_t={t_c:.1f}s {status}")
    print('verify:', 'ALL OK' if ok else 'FAILED')
    return 0 if ok else 1


def cmd_selftest(_args):
    """合成小规模：MCKP == 穷举最优；uniform/random 预算对齐；fast DP vs canonical。"""
    from dp_utils import enum_optimal_m_scheme_global_fast
    rng = np.random.default_rng(0)
    bits = [0, 1, 2]
    E, s, n = 2, 4, 32
    erl, act = [], []
    for e in range(E):
        base = rng.random(n)
        rates = {2: base, 1: base * 2.5 + rng.random(n) * 0.05, 0: base * 7 + rng.random(n) * 0.1}
        erl.append(rates)
        act.append(0.1 + 0.2 * e)
    prep = expert_block_prep(erl, act, bits, s)
    loss_dp, scheme = fast_monotone_dp(prep, target_bpw=1.0, epsilon=0.0)
    flat_dp = scheme_sorted_to_flat_orig(scheme, prep)
    raw_total = sum(flat_dp)

    # 穷举所有 raw 总数 == raw_total 的分配（3^T 级别，T=8）
    import itertools
    T = prep['T']
    bit_to_idx = {b: i for i, b in enumerate(bits)}
    best, best_assign = INF, None
    for combo in itertools.product(bits, repeat=T):
        if sum(combo) == raw_total:
            v = proxy_of(list(combo), prep['losses_orig'], bit_to_idx)
            if v < best:
                best, best_assign = v, combo
    loss_mckp, flat_mckp = mckp_at_budget(prep['losses_orig'], bits, raw_total)
    print(f"raw_total={raw_total}  brute={best:.6g}  mckp={loss_mckp:.6g}  dp={loss_dp:.6g}")
    assert abs(best - loss_mckp) < 1e-9 * max(1, abs(best)), 'MCKP != brute force'
    assert loss_dp >= loss_mckp - 1e-9, 'DP loss below unconstrained optimum'
    fu = uniform_at_budget(bits, T, raw_total)
    assert sum(fu) == raw_total and fu == sorted(fu, reverse=True)
    fr = permute_multiset(flat_dp, seed=1)
    assert sum(fr) == raw_total and sorted(fr) == sorted(flat_dp)
    pu = proxy_of(fu, prep['losses_orig'], bit_to_idx)
    assert pu >= loss_mckp - 1e-9

    # run 模式的取数路径 res[f'flat_{mode}']/res['proxy_'+mode] 键一致性
    # （曾因 flat_unconstrained/flat_unc 缩写不一致在 run 模式秒崩，
    #  replay/selftest 只走 proxy 键所以没拦住）
    res_all = alloc_layer_all_modes(erl, act, bits, s, target_bpw=1.0, epsilon=0.0, seed=0)
    for m in ('dp', 'uniform', 'random', 'unconstrained'):
        assert f'flat_{m}' in res_all, f'flat key mismatch: flat_{m}'
        assert f'proxy_{m}' in res_all, f'proxy key mismatch: proxy_{m}'

    # 货币换算恒等式：erl_rates == source 价格时，eval_alloc_under_rates
    # 必须逐位复刻 proxy_of 的块表口径（含余数神经元不计入的约定）
    for m in ('dp', 'uniform', 'random', 'unconstrained'):
        v_aligned = eval_alloc_under_rates(res_all[f'flat_{m}'], res_all['prep'], erl, act)
        assert abs(v_aligned - res_all[f'proxy_{m}']) < 1e-9 * max(1.0, abs(v_aligned)), \
            f'aligned identity mismatch on {m}: {v_aligned} vs {res_all[f"proxy_{m}"]}'
    print(f"aligned identity ok: "
          f"{eval_alloc_under_rates(res_all['flat_dp'], res_all['prep'], erl, act):.6g} "
          f"== {res_all['proxy_dp']:.6g}")

    # fast DP vs canonical（同输入，comp 路径）
    scheme_c, nb_c = enum_optimal_m_scheme_global_fast(erl, act, s, target_bpw=1.0,
                                                       epsilon=0.0, enable_0bit_compensation=True)
    loss_c = 0.0
    for e, (se, nb) in enumerate(zip(scheme_c, nb_c)):
        for b in bits:
            loss_c += float(erl[e][b][nb == b].sum()) * act[e]
    print(f"canonical={loss_c:.6g} fast={loss_dp:.6g}")
    assert abs(loss_c - loss_dp) < 1e-9 * max(1, abs(loss_c)), 'fast DP != canonical'
    print('selftest: ALL OK')
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    pr = sub.add_parser('replay')
    pr.add_argument('--models', nargs='*')
    pr.add_argument('--sources', nargs='*')
    pr.add_argument('--bpws', nargs='*', type=float)
    pr.add_argument('--summary-only', action='store_true')
    # aligned: 分配级货币换算 —— 每 source 的 DP 分配不变（PPL 仍对得上
    # E1 实测），价格表统一换到 direct 下再算 proxy，写 OUT_CSV_ALIGNED
    pr.add_argument('--aligned', action='store_true')
    pv = sub.add_parser('verify')
    pv.add_argument('--models', nargs='*')
    pv.add_argument('--layers', type=int, default=1)
    sub.add_parser('selftest')
    # run 子命令的 --alloc-scheme/--alloc-seed 由 run_e4b 从 rest 里自行解析
    sub.add_parser('run')
    args, rest = p.parse_known_args()

    if args.cmd == 'replay':
        if args.summary_only:
            cmd_summary_only(args)
        else:
            cmd_replay(args)
    elif args.cmd == 'verify':
        sys.exit(cmd_verify(args))
    elif args.cmd == 'selftest':
        sys.exit(cmd_selftest(args))
    elif args.cmd == 'run':
        sys.exit(run_e4b(rest))


if __name__ == '__main__':
    main()
