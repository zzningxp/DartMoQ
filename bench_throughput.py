#!/usr/bin/env python3
"""
Prefill/decode 吞吐基准（自 DartMoQ-Qwen3.5 移植的口径，int8 范围）。

本脚本在 batch=1 下分 prefill 段与 decode 段报 tok/s（serving 式单 prompt
前向），与现有整数据集批量吞吐口径互补。支持本项目量化 checkpoint
（--quant-dir + --mode wxa16|wxa8）；测速目标模型 dsv2 16B 与
qwen3-30b-a3b（其余模型加载路径未在本轮验证范围）。

评审报告口径（rebuttal）：输出里同时报 checkpoint 存储与显存——
  - checkpoint GB 与按类别分解（MoE packed / attn+shared packed / fp16 残余）
  - 含 scales/codebook/norms/mask 等全部 metadata 的 effective bpw
    （全部存储字节 x 8 / 模型总参数量，参数量由 meta.json 的原始 shape 反推）
  - 分相位峰值推理显存（after_load / warmup / prefill / decode 各档，
    及全程 overall peak allocated）
  - prefill/decode 延迟与 tok/s 的 batch x seq 网格（--batch-sizes，如 1,4）

与 DartMoQ-Qwen3.5 的 bench_throughput.py 同一口径纪律：
  - JIT 预热 + 热跑计时（每档 prefill 长度先 warmup_kernels，再 3 次取中位数）
  - prefill 只走 transformer（不含 lm_head，省显存）；
    TTFT 在 decode 循环里按 prefill 段 + 第一步完整报
  - decode 手动逐 token 顶层循环（DynamicCache + use_cache=True，greedy），
    机制与 eval 已验证路径一致
  - wall（serving 真实口径，含 Python 开销）与 CUDA event（纯 GPU 口径）双报
  - 量化模式顺带聚合各层 mlp.last_timings（init/kernel 拆分）

模型加载默认 trust_remote_code=False（transformers 5.20 原生 modeling）：
dsv2 的旧自定义 modeling 与 5.20 的 Cache API 不兼容（get_usable_length
已移除），原生 DeepseekV2ForCausalLM 的模块名与 checkpoint 的 sd key 一致
（strict 加载已验证）；qwen3_moe 本就是原生。graph_decode.py 同此约定。

用法:
  conda run -n dart312 python bench_throughput.py \
      --quant-dir quant_ckpt/dsv2-2bpw-s1 --mode wxa8 \
      --prefill-lens 16,128,2048 --decode-tokens 128,256

注意：本脚本是 eager 基线（CUDA graph 加速见 graph_decode.py）。若环境
变量 DARTMOQ_STATIC_DISPATCH=1 已导出，MoE 会走设备端定形派发（未捕获图），
启动时会打印告警。

HF 数据下载在本机可能挂起，数据集已缓存时建议加环境变量
HF_DATASETS_OFFLINE=1 运行。
"""

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import get_c4, get_git_hash, get_wikitext2
from dartmoq_utils import DEV

# MoE last_timings 里参与聚合的时间字段（active_experts 等计数类字段排除）
MOE_TIMING_KEYS = ("total", "init", "shared", "router", "sort_scatter_prep",
                   "compute", "triton", "dequant", "gemm", "cleanup_reshape")

# safetensors dtype 字符串 -> 每元素字节数（ckpt 头统计用；未知 dtype 按 4 兜底）
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
               "I16": 2, "F16": 2, "BF16": 2, "I32": 4, "F32": 4,
               "I64": 8, "F64": 8}


# ---------------------------------------------------------------------------
# checkpoint 存储统计（评审要求：ckpt GB / 含 metadata 与 mask 的 effective bpw）
# ---------------------------------------------------------------------------

def _orig_params_from_meta(meta):
    """从 meta.json 反推量化前参数量（量化模块的原始形状记录在 meta 里）。

    - wxa16_linear（attn/shared）：用 shape 或 in/out_features
    - wxa16_moe：bits 组内 shape 求和；0-bit 专家整只被剪掉、bits 组里
      无 shape，按 (E - 各 bit 专家数之和) 反推其原始参数量
    """
    n = 0
    for v in meta.get("modules", {}).values():
        if "shape" in v:
            n += math.prod(v["shape"])
        elif "in_features" in v and "out_features" in v:
            n += v["in_features"] * v["out_features"]
    for v in meta.get("layers", {}).values():
        for grp in v.get("bits", {}).values():
            for mat in grp.values():
                if "shape" in mat:
                    n += math.prod(mat["shape"])
        inter, E = v.get("intermediate_size"), v.get("num_experts")
        byb = v.get("inter_size_by_bit") or {}
        if inter and E and byb:
            zcnt = E - round(sum(sz / inter for sz in byb.values()))
            if zcnt > 0:
                n += zcnt * 3 * v.get("hidden_size", 0) * inter
    return n


def ckpt_report(quant_dir, unquantized_params):
    """按 safetensors 头逐键统计存储（不加载任何张量）。

    分类口径与 dartmoq_quant_io._print_size_breakdown 一致；effective bpw
    = 全部存储字节(含 scales/codebook/norms/mask 等全部 metadata) x 8 /
    模型总参数量（量化前 shape 反推 + 未量化参数）。
    """
    import json as _json
    from collections import defaultdict
    from safetensors import safe_open

    sd_path = os.path.join(quant_dir, "model.safetensors")
    with open(os.path.join(quant_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = _json.load(f)

    cats = defaultdict(int)  # bytes
    total = 0
    with safe_open(sd_path, framework="pt") as f:
        for k in f.keys():
            try:
                sl = f.get_slice(k)
                nbytes = math.prod(sl.get_shape() or [1]) * DTYPE_BYTES.get(sl.get_dtype(), 4)
            except Exception:
                nbytes = 0
            total += nbytes
            if k.startswith("qmeta/"):
                cats["meta scalars (qmeta)"] += nbytes
            elif "bit_weights" in k:
                cats["MoE packed (bit_weights)"] += nbytes
            elif k.endswith(("packed_indices", "codebook", "norms")):
                cats["Attn/Shared packed (indices/codebook/norms)"] += nbytes
            else:
                cats["fp16 remaining params"] += nbytes

    orig = _orig_params_from_meta(meta)
    params_total = orig + unquantized_params
    return {
        "ckpt_file_gb": os.path.getsize(sd_path) / 1024**3,
        "tensor_gb": total / 1024**3,
        "cat_gb": {c: b / 1024**3 for c, b in sorted(cats.items())},
        "params_quantized": orig,
        "params_unquantized": unquantized_params,
        "params_total": params_total,
        # 含 metadata/mask 的 effective bpw：全部存储折到每参数位数
        "eff_bpw_model": 8.0 * total / params_total if params_total else None,
        "quant_args": meta.get("quant_args", {}),
    }


class VramLog:
    """按相位记录 CUDA 显存（GB）：瞬时 allocated/reserved + 自上次 reset 的峰值；
    peak 全程累计（推理峰值显存取 overall peak）。"""

    def __init__(self):
        self.enabled = torch.cuda.is_available()
        self.phases = {}
        self.peak_allocated_gb = 0.0

    def snap(self, label):
        if not self.enabled:
            return
        pk = torch.cuda.max_memory_allocated() / 1024**3
        self.phases[label] = {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
            "peak_allocated_gb": round(pk, 3),
        }
        self.peak_allocated_gb = max(self.peak_allocated_gb, pk)
        torch.cuda.reset_peak_memory_stats()

    def as_dict(self):
        return {"phases": self.phases,
                "peak_allocated_gb": round(self.peak_allocated_gb, 3),
                "reserved_gb_final": (round(torch.cuda.memory_reserved() / 1024**3, 3)
                                      if self.enabled else None)}


def _batch_prompts(sample, B, L):
    """B 行尽量不重叠的 prompt 切片（MoE 各行激活分布尽量不同）；
    样本不够时按取模回绕（计时口径仍成立：kernel 时间由形状决定）。"""
    avail = len(sample) - L
    if avail <= 0:
        raise SystemExit(f"sample too short for prompt len {L}")
    rows = [sample[(i * L) % avail:(i * L) % avail + L] for i in range(B)]
    return torch.stack(rows)


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------

def load_bench_model(args):
    """加载量化 checkpoint，返回 (model, tokenizer, model_id, mode_label)。"""
    assert args.quant_dir, "need --quant-dir (外部基线模型不在本脚本支持范围)"
    from dartmoq_quant_io import load_quantized_model
    model, tokenizer = load_quantized_model(
        base_model_path=args.base_model,
        quant_dir=args.quant_dir,
        inference_quant_mode=args.mode,
        trust_remote_code=args.trust_remote_code,
    )
    model_id = getattr(model, "model_id",
                       os.path.basename(args.quant_dir.rstrip("/")))
    return model, tokenizer, model_id, args.mode


def load_sample_tokens(dataset, tokenizer):
    """从 wiki/c4 测试集取真实文本 token 序列（复用 data_utils 加载器）。

    返回 CPU 上的一维 LongTensor；全部测量（prefill 各档/decode/一致性检查）
    从同一个序列切片，固定 offset 保证可复现。
    """
    print(f"[data] loading {dataset} test set (tokenizing)...", flush=True)
    t0 = time.time()
    if dataset == "wiki":
        _, enc = get_wikitext2(1, 0, 2048, tokenizer)
    elif dataset == "c4":
        _, enc = get_c4(1, 0, 2048, tokenizer)
    else:
        raise SystemExit(f"unknown dataset: {dataset} (expected wiki|c4)")
    ids = enc.input_ids[0].long()
    print(f"[data] {len(ids)} tokens ready in {time.time() - t0:.1f}s")
    return ids


def _text_tower(model):
    """文本塔解析：CausalLM → model.model（本项目五个模型族统一此结构）。"""
    return getattr(model, "model", model)


# ---------------------------------------------------------------------------
# MoE stage 聚合（量化模式；无此属性的模型跳过）
# ---------------------------------------------------------------------------

def sum_moe_timings(model):
    """聚合所有 MoE 层 last_timings（本 forward 的 init/compute/triton/... 拆分）。

    无量化模块的模型返回 None。timings 是 wall-clock（MoE 模块内置），
    每 step 之后调用一次即可得到该 step 的 MoE 内部拆分。
    """
    try:
        layers = _text_tower(model).layers
    except AttributeError:
        return None
    acc = {}
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        lt = getattr(mlp, "last_timings", None)
        if lt is None:
            continue
        for k, v in lt.items():
            if k in MOE_TIMING_KEYS and isinstance(v, (int, float)):
                acc[k] = acc.get(k, 0.0) + float(v)
    return acc or None


# ---------------------------------------------------------------------------
# 预热
# ---------------------------------------------------------------------------

def warmup(model, prefill_lens, batch_sizes, sample, dev):
    """JIT 热跑：Triton kernels 按 (prefill 长度 x batch) 组合编译 +
    微型前向 + 短 decode（含最大 batch 形状）。"""
    print("[warmup] compiling Triton kernels (first run may look stuck; "
          "unseen shapes still compile lazily)...", flush=True)
    t0 = time.time()
    for module in model.modules():
        if not hasattr(module, "warmup_kernels"):
            continue
        for L in sorted(set(prefill_lens)):
            for B in sorted(set(batch_sizes)):
                try:
                    module.warmup_kernels(seq_len=L, batch_size=B)
                except Exception as e:  # 过小/过大的形状 kernel 可能不支持，热跑时再暴露
                    print(f"  [warmup] warmup_kernels(seq_len={L}, batch={B}) skipped: {e}")

    # 微型真前向 + cache 路径短 decode（尽早暴露整模型通路问题；最大 batch
    # 也走一遍，decode 首步不吃 JIT 编译时间）
    ids = torch.ones((1, 16), dtype=torch.long, device=dev)
    print("[warmup] micro forward (1x16) ...", flush=True)
    with torch.no_grad():
        model(input_ids=ids)
        print("[warmup] micro forward done; short decode loop (4 steps) ...",
              flush=True)
        decode_loop(model, sample[:32], 4, dev)
        max_b = max(batch_sizes)
        if max_b > 1:
            decode_loop(model, _batch_prompts(sample, max_b, 32), 4, dev)
    print("[warmup] decode loop done; final sync ...", flush=True)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    print(f"[warmup] done in {time.time() - t0:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# 测量核心
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_loop(model, prompt_ids, n_steps, dev, collect_logits=False):
    """逐 token 顶层 decode 循环（greedy，use_cache）。

    prompt_ids 为 [L] 时 batch=1（gen 每步是 int，与旧口径一致）；
    为 [B, L] 时批量 decode（gen 每步是每行 token 的 int 列表）。
    step 0 是 prefill（整段 prompt），step 1..n_steps 每步只喂 1 个 token。
    返回:
      gen: 生成的 token（长度 n_steps+1，gen[0] 由 prefill 末尾位置预测）
      wall_times: 每步 wall（含 Python 开销），长度 n_steps+1
      event_times: 每步 CUDA event 时间（CPU 上为空），长度 n_steps+1
      moe_times: 每步 MoE 层 last_timings 聚合（无量化模块时为空）
      step_logits: collect_logits=True 时逐步 logits [B, V]，长度 n_steps+1
    """
    single = prompt_ids.dim() == 1
    ids0 = prompt_ids.unsqueeze(0) if single else prompt_ids
    B = ids0.shape[0]
    _cfg = model.config
    cache_obj = _cfg if hasattr(_cfg, "use_cache") else None
    if cache_obj is not None:
        use_cache = cache_obj.use_cache
        cache_obj.use_cache = True
    else:
        use_cache = None
    try:
        past = None
        ids = ids0.to(dev)
        mask = torch.ones(ids.shape, dtype=torch.long, device=dev)
        gen, wall_times, event_times, moe_times, step_logits = [], [], [], [], []
        for step in range(n_steps + 1):
            cuda = ids.device.type == "cuda"
            ev0 = torch.cuda.Event(enable_timing=True) if cuda else None
            ev1 = torch.cuda.Event(enable_timing=True) if cuda else None
            if ev0 is not None:
                ev0.record()
            t0 = time.time()
            out = model(input_ids=ids, attention_mask=mask,
                        past_key_values=past, use_cache=True)
            if ev1 is not None:
                ev1.record()
            t1 = time.time()
            wall_times.append(t1 - t0)
            if ev1 is not None:
                ev0.synchronize()
                ev1.synchronize()
                event_times.append(ev0.elapsed_time(ev1) / 1000.0)

            past = out.past_key_values
            if collect_logits:
                step_logits.append(out.logits[:, -1].detach().clone())
            next_tok = out.logits[:, -1].argmax(dim=-1)      # [B]
            del out
            gen.append(int(next_tok[0]) if single else next_tok.tolist())
            ids = (torch.tensor([[int(next_tok[0])]], dtype=torch.long, device=dev)
                   if single else next_tok.unsqueeze(1))
            mask = torch.cat(
                [mask, torch.ones(B, 1, dtype=torch.long, device=dev)], dim=1)

            mt = sum_moe_timings(model)
            if mt:
                moe_times.append(mt)
            if step > 0 and n_steps >= 128 and (step % 64 == 0 or step == n_steps):
                print(f"  [decode] step {step}/{n_steps}, "
                      f"wall so far {sum(wall_times[1:]):.1f}s", flush=True)
    finally:
        if cache_obj is not None:
            cache_obj.use_cache = use_cache
    return {
        "gen": gen,
        "wall_times": wall_times,
        "event_times": event_times,
        "moe_times": moe_times,
        "step_logits": step_logits,
    }


@torch.no_grad()
def time_prefill(model, sample, lens, batches, reps, dev):
    """prefill 计时（无 cache，只走 transformer，不含 lm_head）：
    lens x batches 网格，每档 reps 次取中位数。B>1 时各行取互不重叠的样本
    切片（MoE 激活分布尽量不同；样本不足按取模回绕）。

    注意必须 no_grad：2048+ 档的 autograd 图会驻留全部层中间张量，
    OOM 排查结论见 DartMoQ-Qwen3.5 的 1015/1016.thrput 日志。
    """
    try:
        transformer = _text_tower(model)
    except AttributeError:
        transformer = model
    results = {}
    for L in lens:
        for B in batches:
            print(f"  [prefill] measuring len={L} batch={B}...", flush=True)
            if B == 1:
                ids = sample[:L].unsqueeze(0).to(dev)
            else:
                ids = _batch_prompts(sample, B, L).to(dev)
            walls, events = [], []
            for _ in range(reps):
                cuda = ids.device.type == "cuda"
                ev0 = torch.cuda.Event(enable_timing=True) if cuda else None
                ev1 = torch.cuda.Event(enable_timing=True) if cuda else None
                if ev0 is not None:
                    ev0.record()
                t0 = time.time()
                out = transformer(ids)
                if ev1 is not None:
                    ev1.record()
                t1 = time.time()
                walls.append(t1 - t0)
                if ev1 is not None:
                    ev0.synchronize()
                    ev1.synchronize()
                    events.append(ev0.elapsed_time(ev1) / 1000.0)
                del out
            row = {"L": L, "batch": B,
                   "latency_s": statistics.median(walls),
                   "tok_s": B * L / statistics.median(walls),
                   "reps": reps, "wall_all": walls}
            if events:
                row["latency_event_s"] = statistics.median(events)
                row["tok_s_event"] = B * L / statistics.median(events)
            results[f"L{L}B{B}"] = row
            # 档间释放激活临时显存，避免长档峰值滚进短档的分配池
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    return results


def consistency_check(model, tokenizer, sample, dev):
    """cache 路径 vs 无 cache 全序列前向的 logits 一致性守卫。

    64 token prefill + 8 步 decode（含 B=1/S=1 量化 kernel 形状），逐步
    logits 与全序列前向对比。

    判据按 dtype 分档：
      - fp32：max diff < 1e-3 且 argmax 全对（精确对拍）；
      - bf16/fp16：max diff < 8.0 且 argmax 一致率 >= 6/8。cache 路径与
        全序列前向的 kernel 形状不同，bf16 数值有差，几十层累积后 logits
        diff ~2 属正常噪声；真 bug（cache 错位/掩码错）会让 diff 到几十且
        argmax 几乎全错，仍会被抓住。
    """
    L, N = 64, 8
    if len(sample) < L:
        raise SystemExit(f"sample too short for consistency check: "
                         f"{len(sample)} < {L} (increase --sample-offset "
                         f"headroom or prompt len)")
    prompt = sample[:L]
    res = decode_loop(model, prompt, N, dev, collect_logits=True)
    gen = res["gen"]
    full = torch.cat([prompt, torch.tensor(gen[:N], dtype=torch.long)],
                     dim=0).unsqueeze(0).to(dev)
    with torch.no_grad():
        logits_full = model(input_ids=full).logits  # [1, L+N, V]

    max_diff = 0.0
    n_agree = 0
    for i in range(N):
        # step_logits 现为 [B, V]（batch 化返回），一致性检查取第 0 行
        d = (res["step_logits"][i][0].float()
             - logits_full[0, L + i - 1].float()).abs().max().item()
        max_diff = max(max_diff, d)
        if int(logits_full[0, L + i - 1].argmax().item()) == gen[i]:
            n_agree += 1
    text_head = tokenizer.decode(
        torch.cat([prompt, torch.tensor(gen[:N], dtype=torch.long)]).tolist(),
        skip_special_tokens=False)[:128]

    dtype = next(model.parameters()).dtype
    if dtype == torch.float32:
        tol, min_agree = 1e-3, N
    else:
        tol, min_agree = 8.0, 6
    return {"max_diff": max_diff, "tol": tol, "argmax_agree": f"{n_agree}/{N}",
            "min_agree": min_agree,
            "pass": max_diff < tol and n_agree >= min_agree,
            "text_head": text_head, "prompt_len": L, "n_steps": N}


# ---------------------------------------------------------------------------
# 统计与输出
# ---------------------------------------------------------------------------

def _per_step_stats(times):
    """均值/p50/p99（ms）。"""
    q = statistics.quantiles(times, n=100, method="inclusive")
    return {"mean_ms": statistics.fmean(times) * 1000,
            "p50_ms": q[49] * 1000, "p99_ms": q[98] * 1000}


def _mean_moe_times(moe_times):
    """decode 步 MoE 内部拆分均值（moe_times 非空时）。"""
    if not moe_times:
        return None
    keys = set().union(*(m.keys() for m in moe_times))
    return {k: statistics.fmean([m.get(k, 0.0) for m in moe_times])
            for k in keys}


def print_report(report):
    """终端表格（英文日志，对齐项目日志规范）。"""
    meta, prefill, decode, moe, check = (report["meta"], report["prefill"],
                                         report["decode"], report["moe_stage"],
                                         report["check"])
    storage, vram = report.get("storage"), report.get("vram")
    line = "=" * 72
    print(f"\n{line}\nPrefill/Decode Throughput Benchmark")
    print(f"model: {meta['model_id']} | mode: {meta['mode']} | "
          f"device: {meta['gpu']} | dataset: {meta['dataset']}")
    # checkpoint 标识行：同一日志多次运行（不同 checkpoint 同 mode）时，
    # logs_parser 靠它消歧
    print(f"checkpoint: {meta['checkpoint']}")
    print(f"prompt_len: {meta['prompt_len']} | "
          f"decode_tokens: {','.join(map(str, meta['decode_tokens']))} | "
          f"batch_sizes: {','.join(map(str, meta['batch_sizes']))} | "
          f"git: {meta['githash']}")
    if storage:
        print("-" * 72)
        print("Checkpoint storage (per review reporting):")
        print(f"  ckpt file: {storage['ckpt_file_gb']:.2f} GB "
              f"(tensors {storage['tensor_gb']:.2f} GB)")
        for cat, gb in storage["cat_gb"].items():
            print(f"    {cat}: {gb:.2f} GB")
        pt = storage["params_total"]
        print(f"  params total: {pt / 1e9:.2f} B "
              f"(quantized {storage['params_quantized'] / 1e9:.2f} B + "
              f"unquantized {storage['params_unquantized'] / 1e9:.2f} B)")
        print(f"  effective bpw (all storage / all params, "
              f"incl. scales/codebooks/masks): {storage['eff_bpw_model']:.3f}")
    if vram:
        print("-" * 72)
        print("VRAM (allocated GB; peak per phase):")
        for ph, v in vram["phases"].items():
            print(f"  {ph:<16} alloc={v['allocated_gb']:.2f} "
                  f"peak={v['peak_allocated_gb']:.2f} "
                  f"reserved={v['reserved_gb']:.2f}")
        print(f"  overall peak allocated: {vram['peak_allocated_gb']:.2f} GB")
    print("-" * 72)
    print("Prefill (no cache, transformer only, median):")
    for key in sorted(prefill, key=lambda k: (prefill[k]["L"], prefill[k]["batch"])):
        r = prefill[key]
        ev = f" | event {r['tok_s_event']:.1f} tok/s" if "tok_s_event" in r else ""
        print(f"  len={r['L']:<5} batch={r['batch']:<3} latency={r['latency_s']:.4f}s  "
              f"tok/s={r['tok_s']:.1f}{ev}")
    print("Decode (greedy, use_cache):")
    for tier in decode:
        n, B = tier["n_tokens"], tier["batch"]
        pw = tier["per_step_wall"]
        ln = (f"  tokens={n:<5} batch={B:<3} wall tok/s = {tier['tok_s_wall']:.2f} "
              f"({n * B} tokens / {tier['wall_s']:.2f}s, excl. prefill)")
        if "tok_s_event" in tier:
            ln += f" | event tok/s = {tier['tok_s_event']:.2f}"
        ln += (f" | per-step wall mean/p50/p99 = "
               f"{pw['mean_ms']:.1f}/{pw['p50_ms']:.1f}/{pw['p99_ms']:.1f} ms")
        print(ln)
    first = decode[0]
    print(f"  TTFT (batch=1 tier) = {first['ttft_s']:.3f}s "
          f"(prefill {first['wall_times'][0]:.3f}s + step1 {first['wall_times'][1]:.3f}s)")
    if moe:
        print("MoE stage (mean per decode step, all layers, s):")
        for k in MOE_TIMING_KEYS:
            if k in moe:
                print(f"  {k:<18} {moe[k]:.4f}")
    if check:
        verdict = "PASS" if check["pass"] else "FAIL"
        print(f"Consistency check: {verdict} "
              f"(max logits diff {check['max_diff']:.2e} < {check['tol']}, "
              f"argmax agree {check['argmax_agree']} >= {check['min_agree']})")
        print(f"  generated head: {check['text_head']!r}")
    print(f"Results: {meta['out_path']}\n{line}")


def save_report(report, args):
    os.makedirs("results", exist_ok=True)
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    safe_id = report["meta"]["model_id"].replace("/", "_")
    safe_mode = report["meta"]["mode"].replace("(", "_").replace(")", "")
    if args.out:
        path = args.out
    else:
        path = os.path.join("results",
                            f"throughput_{safe_id}_{safe_mode}_"
                            f"{report['meta']['githash']}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    report["meta"]["out_path"] = path
    print(f"[out] saved to {path}", flush=True)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Prefill/decode throughput benchmark (quantized ckpts)")
    p.add_argument("--quant-dir", default=None,
                   help="本项目量化 checkpoint 目录（配 --mode）")
    p.add_argument("--base-model", default=None,
                   help="base 模型路径（缺省读 meta.json，checkpoint 自包含则无需）")
    p.add_argument("--mode", default="wxa16", choices=["wxa16", "wxa8"],
                   help="推理模式（默认 wxa16）")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="用 checkpoint 内自定义 modeling（默认 False 走 "
                        "transformers 5.20 原生；dsv2 必须 False，见文件头）")
    p.add_argument("--prefill-lens", default="16,128,2048",
                   help="prefill 档位，逗号分隔（默认 16,128,2048）")
    p.add_argument("--batch-sizes", default="1",
                   help="batch 档位，逗号分隔（默认 1；评审网格建议 1,4）")
    p.add_argument("--decode-tokens", default="256",
                   help="decode 长度档位，逗号分隔（默认 256）")
    p.add_argument("--prompt-len", type=int, default=128,
                   help="decode 段的 prompt 长度（默认 128）")
    p.add_argument("--prefill-reps", type=int, default=3)
    p.add_argument("--dataset", default="wiki", choices=["wiki", "c4"])
    p.add_argument("--sample-offset", type=int, default=0,
                   help="样本序列切片起点（默认 0，固定可复现）")
    p.add_argument("--no-check", action="store_true",
                   help="跳过 logits 一致性守卫")
    p.add_argument("--out", default=None, help="结果 JSON 路径（默认自动命名）")
    args = p.parse_args()
    if not args.quant_dir:
        p.error("need --quant-dir")
    args.prefill_lens = [int(x) for x in args.prefill_lens.split(",")]
    args.batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    args.decode_tokens = [int(x) for x in args.decode_tokens.split(",")]
    if any(b < 1 for b in args.batch_sizes):
        p.error("--batch-sizes must be >= 1")
    return args


def main():
    args = parse_args()
    if os.environ.get("DARTMOQ_STATIC_DISPATCH", "0") == "1":
        print("[warn] DARTMOQ_STATIC_DISPATCH=1 is set: MoE decode rows will "
              "take the device-side static dispatch (uncaptured). Bench "
              "baseline expects the eager host path — unset it unless "
              "measuring static dispatch without graphs.", flush=True)
    model, tokenizer, model_id, mode_label = load_bench_model(args)
    dev = DEV
    print(f"[model] {model_id} on {dev} | mode: {mode_label}", flush=True)

    # checkpoint 存储报告（评审要求：ckpt GB / 含 metadata 与 mask 的
    # effective bpw）。未量化参数量从加载后的模型数（量化模块的原始
    # shape 在 meta.json 里反推）
    vram = VramLog()
    vram.snap("after_load")
    unquantized_params = sum(p.numel() for p in model.parameters())
    storage = ckpt_report(args.quant_dir, unquantized_params)
    print(f"[ckpt] file {storage['ckpt_file_gb']:.2f} GB | tensors "
          f"{storage['tensor_gb']:.2f} GB | params "
          f"{storage['params_total'] / 1e9:.2f} B | eff bpw "
          f"{storage['eff_bpw_model']:.3f}", flush=True)

    sample = load_sample_tokens(args.dataset, tokenizer)
    # 切片需覆盖 prefill 最大档、decode prompt 与一致性检查（64）；decode 生成
    # 的 token 由循环自己产生，不消耗样本
    need = max(max(args.prefill_lens), args.prompt_len, 64) + 16
    if args.sample_offset + need > len(sample):
        raise SystemExit(f"sample too short: offset {args.sample_offset} + "
                         f"{need} > {len(sample)}")
    sample = sample[args.sample_offset:args.sample_offset + need]

    warmup(model, args.prefill_lens, args.batch_sizes, sample, dev)
    vram.snap("warmup")
    # 释放预热期的 JIT 临时显存，让测量相位拿到干净的分配池
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    check = None
    if not args.no_check:
        print("[check] cache-vs-full logits consistency...", flush=True)
        check = consistency_check(model, tokenizer, sample, dev)
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    vram.snap("check")

    print("[bench] prefill...", flush=True)
    prefill = time_prefill(model, sample, args.prefill_lens, args.batch_sizes,
                           args.prefill_reps, dev)
    vram.snap("prefill")

    # decode 多档 x 多 batch：每档各自跑一遍 prefill + n 步 decode（完整会话）
    decode_tiers = []
    all_moe_times = []
    for n in args.decode_tokens:
        for B in args.batch_sizes:
            print(f"[bench] decode {n} tokens x batch {B}...", flush=True)
            prompt = (sample[:args.prompt_len] if B == 1
                      else _batch_prompts(sample, B, args.prompt_len))
            dec = decode_loop(model, prompt, n, dev)
            step_walls = dec["wall_times"][1:]  # 去掉 prefill 段
            stats = {
                "n_tokens": n, "batch": B,
                "wall_s": sum(step_walls),
                "tok_s_wall": n * B / sum(step_walls),
                "tok_s_wall_per_seq": n / sum(step_walls),
                "ttft_s": dec["wall_times"][0] + dec["wall_times"][1],
                "wall_times": dec["wall_times"],
                "per_step_wall": _per_step_stats(step_walls),
                "gen_tokens": dec["gen"][:n],
            }
            if dec["event_times"]:
                step_events = dec["event_times"][1:]
                stats["event_s"] = sum(step_events)
                stats["tok_s_event"] = n * B / sum(step_events)
                stats["per_step_event"] = _per_step_stats(step_events)
            decode_tiers.append(stats)
            all_moe_times.extend(dec["moe_times"])
            vram.snap(f"decode_{n}x{B}")
            # 档间释放 KV cache 与中间张量，防跨档显存累计
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    report = {
        "meta": {
            "model_id": model_id, "mode": mode_label,
            "checkpoint": args.quant_dir,
            "dataset": args.dataset, "sample_offset": args.sample_offset,
            "prompt_len": args.prompt_len, "decode_tokens": args.decode_tokens,
            "batch_sizes": args.batch_sizes,
            "prefill_lens": args.prefill_lens,
            "prefill_reps": args.prefill_reps,
            "trust_remote_code": args.trust_remote_code,
            "githash": get_git_hash(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "date": datetime.now().isoformat(timespec="seconds"),
        },
        "storage": storage,
        "vram": vram.as_dict(),
        "prefill": prefill,
        "decode": decode_tiers,
        "moe_stage": _mean_moe_times(all_moe_times),
        "check": check,
    }
    try:
        import triton
        report["meta"]["triton"] = triton.__version__
    except Exception:
        pass
    try:
        import transformers
        report["meta"]["transformers"] = transformers.__version__
    except Exception:
        pass

    save_report(report, args)
    print_report(report)


if __name__ == "__main__":
    main()
