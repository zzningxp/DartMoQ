"""
WxA16 MoE Kernel Tile Size 自动调优模块。

对 BLOCK_B / BLOCK_N / BLOCK_K / num_warps / num_stages 做全量网格搜索，
找出指定形状下的最优配置并硬编码到配置表。

【设计原则】
  1. 离线搜索，不在 runtime 做（避免首次编译开销）
  2. 测完整 kernel 路径（含 rotation 缓存命中），贴近真实调用
  3. 全量网格搜索（空间不大，200 组左右），结果可靠
  4. 用 torch.cuda.Event 精确计时，多次测量取中位数抗干扰
  5. 结果存 JSON，方便跨硬件/跨版本对比

【典型用法】

  1. 命令行一键搜索（见 test/test_p53_tune.py）：
     python test/test_p53_tune.py --bit 2 --B 32 --N 2816 --K 2048 \\
         --group-size 128 --direction gate_up --output result.json

  2. Python API：
     from turboquant_utils.kernel_autotune import search_best_config
     result = search_best_config(bit_width=2, B=32, N=2816, K=2048, ...)
     print(result['best'])

  3. 一键对当前芯片做全套基准并生成配置表：
     from turboquant_utils.kernel_autotune import run_full_autotune
     run_full_autotune(output_dir='results/autotune_rtx5090')

【搜索空间】
  BLOCK_B:   {16, 32, 64}
  BLOCK_N:   {16, 32, 64, 128}
  BLOCK_K:   {32, 64, 128}
  num_warps: {2, 4, 8}
  num_stages:{2, 3, 4}
  全组合 324 组，过滤后约 150-200 组有效配置。
"""

import itertools
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import triton

from .quantize import turboquant_quantize_packed_full
from .triton_kernels import (
    _turboquant_fused_matmul_kernel_grouped_gf,
    convert_to_group_first,
)


# ============================================================
# 搜索空间定义
# ============================================================

SEARCH_SPACE = {
    "BLOCK_B": [16, 32, 64],
    "BLOCK_N": [16, 32, 64, 128],
    "BLOCK_K": [32, 64, 128],
    "num_warps": [2, 4, 8],
    "num_stages": [2, 3, 4],
}


def generate_configs() -> List[Dict]:
    """生成搜索空间中的所有配置组合。"""
    keys = list(SEARCH_SPACE.keys())
    values = [SEARCH_SPACE[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def is_config_valid(cfg: Dict, B: int, N: int, group_size: int) -> bool:
    """过滤明显不合理的配置。

    过滤规则：
      - tile 比矩阵还大（padding 浪费严重）
      - BLOCK_K > group_size（内循环只跑一次，无意义）
      - 每个 warp 分到的输出元素太少（并行效率低）
      - BLOCK_K 不是 16 的倍数（mma 形状约束）
      - BLOCK_B / BLOCK_N 不是 16 的倍数
    """
    BLOCK_B = cfg["BLOCK_B"]
    BLOCK_N = cfg["BLOCK_N"]
    BLOCK_K = cfg["BLOCK_K"]
    num_warps = cfg["num_warps"]

    if BLOCK_B > B and B > 0:
        return False
    if BLOCK_N > N and N > 0:
        return False

    if BLOCK_K > group_size:
        return False

    # 每个 warp 至少负责 2 个输出元素（否则并行开销太大）
    elements_per_warp = (BLOCK_B * BLOCK_N) / num_warps
    if elements_per_warp < 32 * 2:  # 每线程 2 个元素
        return False

    # mma 形状约束
    if BLOCK_K % 16 != 0:
        return False
    if BLOCK_B % 16 != 0 or BLOCK_N % 16 != 0:
        return False

    return True


# ============================================================
# 单配置 benchmark
# ============================================================

def _call_kernel_gf(
    x_rot_concat, indices_packed_gf, codebook, norms_gf,
    group_size, num_groups, bit_width,
    BLOCK_B, BLOCK_N, BLOCK_K, num_warps, num_stages,
):
    """直接调用 group-first 布局的内部 kernel。"""
    B = x_rot_concat.shape[0]
    N = indices_packed_gf.shape[1]
    K_total = x_rot_concat.shape[1]

    indices_g0_stride = indices_packed_gf.stride(0)
    norms_g0_stride = norms_gf.stride(0)

    output = torch.empty(B, N, dtype=x_rot_concat.dtype, device=x_rot_concat.device)

    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(N, BLOCK_N))

    _turboquant_fused_matmul_kernel_grouped_gf[grid](
        x_rot_concat, indices_packed_gf, codebook, norms_gf, output,
        B, N, K_total,
        indices_g0_stride, norms_g0_stride,
        GROUP_SIZE=group_size, NUM_GROUPS=num_groups,
        BIT_WIDTH=bit_width, N_LEVELS=codebook.shape[0],
        BLOCK_B=BLOCK_B, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )

    return output


def benchmark_config(
    cfg: Dict,
    x_rot_concat, indices_packed_gf, codebook, norms_gf,
    group_size: int, num_groups: int, bit_width: int,
    n_warmup: int = 3, n_repeat: int = 10,
) -> Optional[Tuple[float, List[float]]]:
    """测试单个配置的性能。

    Returns:
        (median_ms, all_times_ms)，失败返回 None。
    """
    BLOCK_B = cfg["BLOCK_B"]
    BLOCK_N = cfg["BLOCK_N"]
    BLOCK_K = cfg["BLOCK_K"]
    num_warps = cfg["num_warps"]
    num_stages = cfg["num_stages"]

    try:
        # Warmup（含编译）
        for _ in range(n_warmup):
            out = _call_kernel_gf(
                x_rot_concat, indices_packed_gf, codebook, norms_gf,
                group_size, num_groups, bit_width,
                BLOCK_B, BLOCK_N, BLOCK_K, num_warps, num_stages,
            )
            del out
        torch.cuda.synchronize()

        # 正式测量
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        times = []

        for _ in range(n_repeat):
            start_event.record()
            out = _call_kernel_gf(
                x_rot_concat, indices_packed_gf, codebook, norms_gf,
                group_size, num_groups, bit_width,
                BLOCK_B, BLOCK_N, BLOCK_K, num_warps, num_stages,
            )
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
            del out

        times_sorted = sorted(times)
        median_ms = times_sorted[len(times_sorted) // 2]
        return median_ms, times

    except Exception:
        return None


# ============================================================
# 测试数据构造
# ============================================================

def setup_test_data(
    bit_width: int, B: int, N: int, K: int, group_size: int,
    device="cuda", seed: int = 42,
):
    """构造调优用的量化权重和旋转后输入。

    为了排除 rotation 开销对 tile 性能测量的干扰，直接用随机连续张量
    作为"旋转后"的输入（不影响 kernel 内部的 tile 性能对比）。

    Returns:
        (x_rot_concat, indices_packed_gf, codebook, norms_gf, num_groups)
    """
    assert K % group_size == 0, f"K={K} 必须是 group_size={group_size} 的整数倍"
    num_groups = K // group_size

    torch.manual_seed(seed)

    # 构造 fp16 权重并量化
    w_fp16 = torch.randn(N, K, device=device, dtype=torch.float16) * 0.02
    packed = turboquant_quantize_packed_full(
        w_fp16, bit_width=bit_width, group_size=group_size,
        seed=seed, keep_on_gpu=True,
    )

    indices_packed = packed["indices_packed"].to(device)
    codebook = packed["codebook"].to(device=device, dtype=torch.float16)
    norms = packed["norms"].to(device=device, dtype=torch.float16)

    # 转成 group-first 布局
    indices_packed_gf, norms_gf = convert_to_group_first(
        indices_packed, norms, group_size, bit_width
    )

    # 构造模拟的"旋转后"输入（连续存储）
    x_rot_concat = torch.randn(B, K, device=device, dtype=torch.float16).contiguous()
    indices_packed_gf = indices_packed_gf.contiguous()
    norms_gf = norms_gf.contiguous()

    return x_rot_concat, indices_packed_gf, codebook, norms_gf, num_groups


# ============================================================
# 主搜索函数
# ============================================================

def compute_flops(B: int, N: int, K: int) -> float:
    """估算一次 matmul 的 FLOPs（乘法+加法 = 2*B*N*K）。"""
    return 2.0 * B * N * K


def search_best_config(
    bit_width: int,
    B: int,
    N: int,
    K: int,
    group_size: int = 128,
    direction: str = "gate_up",
    n_warmup: int = 3,
    n_repeat: int = 10,
    seed: int = 42,
    limit: int = 0,
    baseline_config: Optional[Dict] = None,
    device="cuda",
) -> Dict:
    """对指定形状做全量网格搜索，返回排序后的结果。

    Args:
        bit_width: 1/2/4/8
        B: token 数（batch 维）
        N: 输出维度（neuron 数）
        K: 输入维度
        group_size: group 大小
        direction: "gate_up" 或 "down"（仅用于元数据标注）
        n_warmup: 每组配置的 warmup 次数
        n_repeat: 每组配置的正式测量次数
        seed: 随机种子
        limit: 只测前 N 个配置（0 = 不限制）
        baseline_config: baseline 配置字典。None 时用当前表中的配置。

    Returns:
        包含 params / baseline / best / results 的结果字典。
    """
    from .rotation import clear_rotation_cache
    clear_rotation_cache()

    # 构造测试数据
    x_rot_concat, indices_packed_gf, codebook, norms_gf, num_groups = setup_test_data(
        bit_width, B, N, K, group_size, device, seed
    )

    # 确定 baseline 配置
    if baseline_config is None:
        from .triton_kernels import _get_fused_grouped_config
        t = _get_fused_grouped_config(bit_width, direction=direction, B=B)
        baseline_config = {
            "BLOCK_B": t[0], "BLOCK_N": t[1], "BLOCK_K": t[2],
            "num_warps": t[3], "num_stages": t[4],
        }

    # 先测 baseline
    ref_result = benchmark_config(
        baseline_config, x_rot_concat, indices_packed_gf, codebook, norms_gf,
        group_size, num_groups, bit_width,
        n_warmup=n_warmup, n_repeat=n_repeat,
    )
    if ref_result is None:
        raise RuntimeError("Baseline 配置运行失败！")
    ref_median_ms, ref_times = ref_result
    ref_tflops = compute_flops(B, N, K) / (ref_median_ms / 1000) / 1e12

    # 生成所有配置
    all_configs = generate_configs()
    valid_configs = [c for c in all_configs if is_config_valid(c, B, N, group_size)]
    if limit > 0:
        valid_configs = valid_configs[:limit]

    # 跑搜索
    results = []
    failed_count = 0

    for cfg in valid_configs:
        # 跳过和 baseline 完全相同的配置
        if (cfg["BLOCK_B"] == baseline_config["BLOCK_B"] and
            cfg["BLOCK_N"] == baseline_config["BLOCK_N"] and
            cfg["BLOCK_K"] == baseline_config["BLOCK_K"] and
            cfg["num_warps"] == baseline_config["num_warps"] and
            cfg["num_stages"] == baseline_config["num_stages"]):
            results.append({
                "config": cfg,
                "median_ms": ref_median_ms,
                "tflops": ref_tflops,
                "speedup": 1.0,
                "is_baseline": True,
            })
            continue

        result = benchmark_config(
            cfg, x_rot_concat, indices_packed_gf, codebook, norms_gf,
            group_size, num_groups, bit_width,
            n_warmup=n_warmup, n_repeat=n_repeat,
        )

        if result is None:
            failed_count += 1
            continue

        median_ms, _ = result
        speedup = ref_median_ms / median_ms
        tflops = compute_flops(B, N, K) / (median_ms / 1000) / 1e12

        results.append({
            "config": cfg,
            "median_ms": median_ms,
            "tflops": tflops,
            "speedup": speedup,
            "is_baseline": False,
        })

    # 按性能排序
    results.sort(key=lambda r: r["median_ms"])
    best = results[0]

    return {
        "params": {
            "bit_width": bit_width,
            "direction": direction,
            "B": B,
            "N": N,
            "K": K,
            "group_size": group_size,
            "num_groups": num_groups,
            "n_warmup": n_warmup,
            "n_repeat": n_repeat,
            "seed": seed,
            "total_configs": len(valid_configs),
            "failed_count": failed_count,
        },
        "baseline": {
            "config": baseline_config,
            "median_ms": ref_median_ms,
            "tflops": ref_tflops,
        },
        "best": {
            "config": best["config"],
            "median_ms": best["median_ms"],
            "tflops": best["tflops"],
            "speedup": best["speedup"],
        },
        "results": results,
    }


def save_result(result: Dict, path: str):
    """保存搜索结果到 JSON 文件。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


# ============================================================
# 全套基准（针对当前 GPU）
# ============================================================

def run_full_autotune(
    output_dir: str = "results/kernel_autotune",
    group_size: int = 128,
    hidden_size: int = 2048,
    intermediate_size: int = 2816,
    n_warmup: int = 3,
    n_repeat: int = 10,
):
    """对当前 GPU 跑全套基准测试。

    覆盖：
      - 所有 bit_width: 1, 2, 4, 8
      - 两个方向: gate_up, down
      - 多个 B 值: 16, 32, 64（覆盖 MoE 典型范围）

    结果存到 output_dir 下，每个组合一个 JSON 文件。
    """
    bit_widths = [1, 2, 4, 8]
    B_values = [16, 32, 64]

    # 各方向的形状
    shapes = {
        "gate_up": {"N": 2 * intermediate_size, "K": hidden_size},
        "down": {"N": hidden_size, "K": intermediate_size},
    }

    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*80}")
    print(f"全套 kernel autotune — 结果输出到: {output_dir}")
    print(f"{'='*80}")

    total_start = time.time()
    total_jobs = len(bit_widths) * len(shapes) * len(B_values)
    done = 0

    for bit in bit_widths:
        for direction, shape in shapes.items():
            for B in B_values:
                done += 1
                fname = f"bit{bit}_{direction}_B{B}.json"
                fpath = os.path.join(output_dir, fname)

                print(f"\n[{done}/{total_jobs}] bit={bit}, dir={direction}, B={B}")
                print(f"  -> {fname}")

                t0 = time.time()
                result = search_best_config(
                    bit_width=bit, B=B, N=shape["N"], K=shape["K"],
                    group_size=group_size, direction=direction,
                    n_warmup=n_warmup, n_repeat=n_repeat,
                )
                save_result(result, fpath)

                print(f"  baseline: {result['baseline']['median_ms']:.4f} ms "
                      f"({result['baseline']['tflops']:.2f} TFLOPS)")
                print(f"  best:     {result['best']['median_ms']:.4f} ms "
                      f"({result['best']['tflops']:.2f} TFLOPS) "
                      f"[{result['best']['speedup']:.2f}x]")
                print(f"  耗时: {time.time() - t0:.1f}s")

    total_time = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"全部完成，总耗时 {total_time:.1f}s")
    print(f"{'='*80}")


# ============================================================
# 结果分析工具
# ============================================================

def print_topk(result: Dict, k: int = 20):
    """打印 Top-K 配置。"""
    results = result["results"]
    p = result["params"]

    print(f"\n{'='*90}")
    print(f"Top {min(k, len(results))} 最优配置 "
          f"(bit={p['bit_width']}, dir={p['direction']}, B={p['B']})")
    print(f"{'='*90}")

    header = (f"{'Rank':>4s}  {'BLOCK_B':>8s} {'BLOCK_N':>8s} {'BLOCK_K':>8s} "
              f"{'warps':>5s} {'stages':>6s}  {'Time(ms)':>9s} {'Speedup':>8s} "
              f"{'TFLOPS':>7s}  {'Note'}")
    print(header)
    print("-" * 90)

    for i, r in enumerate(results[:k]):
        cfg = r["config"]
        note = "★ baseline" if r.get("is_baseline") else ""
        print(f"{i+1:4d}  {cfg['BLOCK_B']:8d} {cfg['BLOCK_N']:8d} {cfg['BLOCK_K']:8d} "
              f"{cfg['num_warps']:5d} {cfg['num_stages']:6d}  "
              f"{r['median_ms']:9.4f} {r['speedup']:7.2f}x "
              f"{r['tflops']:7.2f}  {note}")


def generate_config_table(result_dir: str, bit_widths=None, B_threshold: int = 16):
    """从搜索结果中生成配置表。

    对每个 (bit, direction)，根据 B 阈值选择 small/large 档的最优配置。

    Args:
        result_dir: 搜索结果目录（run_full_autotune 的输出目录）
        bit_widths: 要包含的 bit_width 列表，None 表示全部
        B_threshold: small/large 的 B 阈值

    Returns:
        (gate_up_table, down_table) — 格式: {size_key: {bit: (BB, BN, BK, w, s)}}
    """
    if bit_widths is None:
        bit_widths = [1, 2, 4, 8]

    tables = {}
    for direction in ["gate_up", "down"]:
        table = {"small": {}, "large": {}}
        for bit in bit_widths:
            for size_key, B_val in [("small", B_threshold), ("large", B_threshold + 16)]:
                # 找最接近的 B 值的结果文件
                fname = f"bit{bit}_{direction}_B{B_val}.json"
                fpath = os.path.join(result_dir, fname)
                if not os.path.exists(fpath):
                    continue
                with open(fpath) as f:
                    result = json.load(f)
                best_cfg = result["best"]["config"]
                table[size_key][bit] = (
                    best_cfg["BLOCK_B"], best_cfg["BLOCK_N"], best_cfg["BLOCK_K"],
                    best_cfg["num_warps"], best_cfg["num_stages"],
                )
        tables[direction] = table

    return tables["gate_up"], tables["down"]
