#!/usr/bin/env python3
"""graph replay 逐 kernel profile（调优决策用：28.4ms 里到底是什么）。

目的：qwen3-30b-a3b-1bpw wxa8 的 CUDA graph decode 是 28.4 ms/step
（35.2 tok/s），dsv2 同管线 17.4 ms。本脚本对 g.replay() 套 torch.profiler
（CUPTI），给出：

  1. top-N kernel 表（名字 / 每 step 调用数 / 每 step 累计 ms / 均值 µs /
     占 step 比例）——判断该走 kernel 效率优化（少数大 kernel 慢）还是
     kernel 数量优化（海量小 kernel 的执行地板）
  2. 桶聚合：triton MoE / triton 量化 / attention / gemm / elementwise /
     memcpy / other
  3. replay 墙钟（CUDA event，无 profiler）vs Σ kernel 时间 → graph 内
     kernel 间隙
  4. 无 profiler 的 replay 中位耗时（与 graph_decode.py 的 replay_ms 对拍）

复用与复刻声明（代码复用规范）：
  - 复用 bench_throughput.load_bench_model / load_sample_tokens / _text_tower
    与 graph_decode.emb_device / reset_cache / static_prefill
  - capture 段（静态 buffer + feed/step + 侧流捂热 + 捕获）复刻自
    graph_decode.py main() 步骤 3-5（那边内联在 main 里无法 import）；
    graph_decode.py 的该段如有改动，此处需同步

与 graph_decode.py 的差别：不跑 eager 参考闭环（不需要 token 对拍，纯测时），
teacher-forced 喂样本 token 即可；静态派发下 kernel 集合与 token 值无关
（全池定形执行）。

用法（走 run.profile.sh，marker 断点续跑）：
    python profile_graph_decode.py --quant-dir quant_ckpt/qwen3-30b-a3b-1bpw-s1 \
        --mode wxa8 --prompt-len 128 --profile-steps 32
结果落 results/profile_graph_<model>_<mode>_<ts>.json。
"""

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_throughput import (  # noqa: E402
    _text_tower,
    load_bench_model,
    load_sample_tokens,
)
from graph_decode import emb_device, reset_cache, static_prefill  # noqa: E402

# 桶聚合的 kernel 名字模式（小写子串匹配；triton 内核名含源文件路径）
BUCKETS = [
    ("triton", ["triton"]),
    ("attention", ["flash", "fmha", "sdpa", "attn", "softmax"]),
    ("gemm", ["gemm", "cublas", "cutlass", "nvjet", "matmul", "igemm"]),
    ("elementwise", ["elementwise", "vectorized", "reduce", "cat_array",
                     "fill", "index", "copy_", "unroll"]),
    ("memcpy", ["memcpy"]),
]


def short_name(name):
    """kernel 长名压缩：取路径尾段 + 函数名（原始名保留在 JSON）。"""
    n = name.split("/")[-1]
    return n if len(n) <= 80 else n[:77] + "..."


def bucket_of(name):
    low = name.lower()
    for tag, pats in BUCKETS:
        if any(p in low for p in pats):
            return tag
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant-dir", default=None)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--mode", default="wxa8", choices=["wxa8"],
                    help="仅 wxa8（与 graph_decode.py 同：静态派发只移植了 wxa8）")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--settle-steps", type=int, default=8,
                    help="计时前无 profiler 的 settle replay 步数")
    ap.add_argument("--time-steps", type=int, default=32,
                    help="CUDA event 计时的 replay 步数（取中位）")
    ap.add_argument("--profile-steps", type=int, default=32,
                    help="torch.profiler 覆盖的 replay 步数")
    ap.add_argument("--mask-mode", choices=["native", "buffer"], default="native")
    ap.add_argument("--top", type=int, default=40, help="打印 top-N kernel")
    ap.add_argument("--dataset", default="wiki", choices=["wiki", "c4"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.quant_dir:
        ap.error("need --quant-dir")

    from transformers import StaticCache

    model, tokenizer, model_id, mode_label = load_bench_model(args)
    sample = load_sample_tokens(args.dataset, tokenizer)
    text = _text_tower(model)
    dev = emb_device(model)
    print(f"[profile] {model_id} {mode_label} | layers={len(text.layers)} "
          f"| prompt_len={args.prompt_len}", flush=True)

    # 静态派发必须在前向前打开（与 graph_decode.py main 步骤 1.5 相同）
    os.environ["DARTMOQ_STATIC_DISPATCH"] = "1"
    print("[profile] DARTMOQ_STATIC_DISPATCH=1", flush=True)

    max_cache_len = args.prompt_len + args.warmup_steps + args.settle_steps \
        + args.time_steps + args.profile_steps + 16
    static_cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
    static_prefill(model, sample, args.prompt_len, static_cache)

    # ---- 静态 buffer 与单步函数（复刻 graph_decode.py main 步骤 3）----
    ids_buf = torch.zeros(1, 1, dtype=torch.long, device=dev)
    pos_buf = torch.zeros(1, 1, dtype=torch.long, device=dev)
    if args.mask_mode == "buffer":
        dt = text.embed_tokens.weight.dtype
        mask_buf = torch.zeros(1, 1, 1, max_cache_len, dtype=dt, device=dev)
    else:
        mask_buf = None

    def feed(pos, token):
        ids_buf.fill_(token)
        pos_buf.fill_(pos)
        if mask_buf is not None:
            mask_buf.fill_(float("-inf"))
            mask_buf[..., : pos + 1].zero_()

    def step():
        out = model(input_ids=ids_buf, position_ids=pos_buf,
                    attention_mask=mask_buf, past_key_values=static_cache,
                    use_cache=True)
        return out.logits

    # teacher-forced 喂样本 token（kernel 集合与 token 值无关，纯测时）
    p0 = args.prompt_len
    feed_toks = sample[p0: p0 + max_cache_len].tolist()

    def feed_k(k):
        feed(p0 + k, feed_toks[k % len(feed_toks)])

    # ---- 侧流捂热 + 捕获（复刻 graph_decode.py main 步骤 4/5）----
    reset_cache(static_cache, dev)
    static_prefill(model, sample, args.prompt_len, static_cache)
    side = torch.cuda.Stream(device=dev)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for k in range(args.warmup_steps):
            feed_k(k)
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize(dev)

    print("[profile] capturing decode step ...", flush=True)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    print("[profile] capture ok", flush=True)

    def replay_loop(n, start_k):
        for k in range(start_k, start_k + n):
            feed_k(k)
            g.replay()
        torch.cuda.synchronize(dev)

    # ---- 1) 无 profiler 的 replay 计时（对拍 graph_decode 的 replay_ms）----
    replay_loop(args.settle_steps, args.warmup_steps)
    k0 = args.warmup_steps + args.settle_steps
    meds = []
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for k in range(k0, k0 + args.time_steps):
        feed_k(k)
        torch.cuda.synchronize(dev)
        ev0.record()
        g.replay()
        ev1.record()
        torch.cuda.synchronize(dev)
        meds.append(ev0.elapsed_time(ev1))
    replay_med_ms = statistics.median(meds)
    print(f"[profile] replay wall (no profiler, CUDA event): median "
          f"{replay_med_ms:.3f} ms/step over {args.time_steps} steps "
          f"({1000.0 / replay_med_ms:.2f} tok/s)", flush=True)

    # ---- 2) profiler 覆盖 replay（CUPTI 逐 kernel）----
    from torch.profiler import ProfilerActivity, profile
    k1 = k0 + args.time_steps
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        replay_loop(args.profile_steps, k1)

    rows = []
    for ev in prof.key_averages():
        dur_us = getattr(ev, "self_device_time_total", 0.0)
        cnt = ev.count
        # 只留真正在 device 上执行的 kernel/活动（CPU 侧 op 的 device 时间为 0）
        if dur_us <= 0 or cnt <= 0:
            continue
        rows.append({
            "name": ev.key,
            "short": short_name(ev.key),
            "bucket": bucket_of(ev.key),
            "count": cnt,
            "total_ms": dur_us / 1000.0,
            "mean_us": dur_us / cnt,
        })
    rows.sort(key=lambda r: -r["total_ms"])

    nsteps = args.profile_steps
    kern_total_ms = sum(r["total_ms"] for r in rows) / nsteps
    print(f"\n===== top {args.top} kernels (per step, {nsteps} steps "
          f"replayed, profiler) =====")
    print(f"{'ms/step':>8} {'%step':>6} {'calls':>7} {'mean_us':>8}  "
          f"{'bucket':<10} name")
    for r in rows[:args.top]:
        ms = r["total_ms"] / nsteps
        print(f"{ms:8.3f} {ms / replay_med_ms * 100:5.1f}% "
              f"{r['count'] / nsteps:7.1f} {r['mean_us']:8.1f}  "
              f"{r['bucket']:<10} {r['short']}")

    # ---- 3) 桶聚合 + 间隙 ----
    buckets = {}
    for r in rows:
        b = buckets.setdefault(r["bucket"], {"ms": 0.0, "calls": 0, "kernels": 0})
        b["ms"] += r["total_ms"] / nsteps
        b["calls"] += r["count"] / nsteps
        b["kernels"] += 1
    print("\n===== bucket summary (per step) =====")
    print(f"{'ms/step':>8} {'%step':>6} {'calls':>8} {'#kern':>6}  bucket")
    for tag, b in sorted(buckets.items(), key=lambda kv: -kv[1]["ms"]):
        print(f"{b['ms']:8.3f} {b['ms'] / replay_med_ms * 100:5.1f}% "
              f"{b['calls']:8.1f} {b['kernels']:6d}  {tag}")
    gap_ms = replay_med_ms - kern_total_ms
    print(f"\nΣ kernel = {kern_total_ms:.3f} ms/step | replay wall = "
          f"{replay_med_ms:.3f} ms/step | in-graph gap = {gap_ms:.3f} ms "
          f"({gap_ms / replay_med_ms * 100:.1f}% of wall)")
    print("[hint] 少数大 kernel 慢 → kernel 效率优先；海量小 kernel（均值 "
          "<20µs）占大头 → kernel 数量/合并优先；SM 填充度与占用率用 nsys "
          "profile 复跑本脚本（nsys 会吞 CUPTI，别同时开）深挖。")

    # ---- 4) 落盘 ----
    out = args.out or os.path.join(
        "results",
        f"profile_graph_{model_id}_{mode_label}_"
        f"{time.strftime('%m%d_%H%M%S')}.json")
    os.makedirs("results", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "model_id": model_id, "mode": mode_label,
                "mask_mode": args.mask_mode, "prompt_len": args.prompt_len,
                "profile_steps": nsteps, "time_steps": args.time_steps,
                "date": datetime.now().isoformat(timespec="seconds"),
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
            },
            "replay_wall_ms_median": replay_med_ms,
            "kernel_sum_ms_per_step": kern_total_ms,
            "in_graph_gap_ms": gap_ms,
            "buckets_per_step": buckets,
            "kernels": rows,
        }, f, indent=1, ensure_ascii=False)
    print(f"[profile] saved: {out}", flush=True)


if __name__ == "__main__":
    main()
