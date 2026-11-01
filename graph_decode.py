#!/usr/bin/env python3
"""CUDA Graph decode（自 DartMoQ-Qwen3.5 移植的 whole 策略，int8 范围）。

把一个 decode 步捕获成 CUDA graph 重放，吃掉 host dispatch 开销。本
DartMoQ 移植与原版（Qwen3.5，见其文件头）的差异：

  - 仅 int8（--mode wxa8）：MoE 的 host 编排（token 统计 DtoH → Python
    池字典 → 变长 cat → H2D 调度表）与 capture 根本冲突，wxa8 已随静态
    派发移植（DARTMOQ_STATIC_DISPATCH=1，见 quantization/wxa16/
    bit_partitioned_moe.py 的移植注释）；wxa16 的静态池 kernel 未移植，
    本脚本直接拒绝（capture 会在 host 编排处死，无意义）。
  - 无线性注意力（Qwen3.5 的 GDN/双卡 segment 策略不适用）：目标模型
    dsv2 16B 与 qwen3-30b-a3b 全 full-attention、单卡放得下 → 只走
    whole 策略（整步一张图）。
  - 模型加载强制 transformers 5.20 原生 modeling（trust_remote_code=False
    默认）：dsv2 旧自定义 modeling 与 5.20 Cache API 不兼容；原生模块名
    与 checkpoint sd key 一致（strict 加载已验证，test_dsv2_native_cache.py）。
  - position_ids 用 (1,1) 2D 设备 buffer 逐步 fill_ 更新（原生 5.20
    StaticCache 的 cumulative_length 是设备张量、cudagraph 设计，无 host
    分支）；mask 两模式：native（框架自建，capture 合法性由本 PoC 直接
    检验）/ buffer（自持 (1,1,1,max_len) 加性 mask 逐步原地更新，语义
    完全自控）。默认 native，capture 报错或对拍 FAIL 时换 buffer。

正确性判据（与原版同款）：
  1) closed-loop greedy token 序列 vs eager decode_loop（pass B，硬判据；
     bf16 下平局步翻转属预期，pass B2 诊断区分 benign tie-flip / NaN /
     状态漂移）；
  2) teacher-forced 逐步 logits 对拍（pass C，信息性软判据）；
  3) 速度：replay-only CUDA event 计时（理论上限参考）+ 带 .item() 同步
     的整循环 wall（bench decode_loop 同口径）。

--steps / --prompt-len 均接受逗号多档全交叉出表；capture 是单步图，与
prompt 长度无关，一次捕获全档复用。eager 参考每档 prompt 只跑最长 steps
档——greedy 逐 token 生成，短档序列必然是最长档前缀，前缀对拍即可；
logits 对拍只在首档 prompt（capture 保真度与上下文长度无关）。

用法（本人手动跑；1bpw/2bpw checkpoint 现成（quant_ckpt/*-s1，slices=1
口径），量化保存入口见 run.quant.and.save.sh / run_dartmoq.py --save-quantized）：
    conda activate dart312
    export HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
    python graph_decode.py --quant-dir quant_ckpt/dsv2-2bpw-s1 --mode wxa8 \
        --prompt-len 16,128,2048 --steps 64,256
    python graph_decode.py --quant-dir quant_ckpt/qwen3-30b-a3b-1bpw-s1 \
        --mode wxa8 --prompt-len 128 --steps 256

结果 JSON 落 results/graph_decode_<model>_<mode>_<githash>_<ts>.json。
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
import traceback
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_throughput import (  # noqa: E402
    _text_tower,
    load_bench_model,
    load_sample_tokens,
)
from data_utils import get_git_hash  # noqa: E402


def emb_device(model):
    return model.get_input_embeddings().weight.device


@torch.no_grad()
def eager_reference(model, sample, prompt_len, steps):
    """bench decode_loop 同款 greedy（DynamicCache 默认路径），收 token/logits/计时。

    返回 tokens（steps+1 个）、logits（steps+1 份 cpu）、decode 段每步 ms。"""
    dev = emb_device(model)
    ids = sample[:prompt_len].unsqueeze(0).to(dev)
    mask = torch.ones(1, prompt_len, dtype=torch.long, device=dev)
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    out = model(input_ids=ids, attention_mask=mask,
                past_key_values=None, use_cache=True)
    ev1.record()
    ev0.synchronize()
    ev1.synchronize()
    # prefill 计时（与 static_prefill 的 StaticCache 口径对比速度差）
    print(f"[eager] prefill DynamicCache len={prompt_len}: "
          f"{ev0.elapsed_time(ev1):.1f} ms", flush=True)
    past = out.past_key_values
    tokens, logits, step_ms = [int(out.logits[0, -1].argmax().item())], \
        [out.logits[0, -1].float().cpu()], []
    ids = torch.tensor([[tokens[0]]], dtype=torch.long, device=dev)
    mask = torch.cat([mask, torch.ones(1, 1, dtype=torch.long, device=dev)], dim=1)

    for _ in range(steps):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        out = model(input_ids=ids, attention_mask=mask,
                    past_key_values=past, use_cache=True)
        ev1.record()
        past = out.past_key_values
        tok = int(out.logits[0, -1].argmax().item())
        ev0.synchronize()
        ev1.synchronize()
        step_ms.append(ev0.elapsed_time(ev1))
        tokens.append(tok)
        logits.append(out.logits[0, -1].float().cpu())
        del out
        ids = torch.tensor([[tok]], dtype=torch.long, device=dev)
        mask = torch.cat([mask, torch.ones(1, 1, dtype=torch.long, device=dev)], dim=1)
    del past
    torch.cuda.empty_cache()
    return tokens, logits, step_ms


def dump_exc_chain(tag, e, max_depth=8, tail_frames=4):
    """capture 失败聚合报错的解包打印。

    capture 期真实非法调用若在 fn() 内部先抛，graph.__exit__ 的 capture_end
    二次报错会把它链到 __context__ 上，表层只剩「previous error during
    capture」。沿 __context__/__cause__ 链逐层打印类型+首行消息+最内层
    traceback 帧（含模型文件行号），可直接看到是哪个 op、哪一行触发。"""
    seen, cur, depth = set(), e, 0
    while cur is not None and id(cur) not in seen and depth < max_depth:
        seen.add(id(cur))
        msg = str(cur).splitlines()[0] if str(cur) else ""
        print(f"[{tag}] exc[{depth}]: {type(cur).__name__}: {msg}", flush=True)
        for fr in traceback.extract_tb(cur.__traceback__)[-tail_frames:]:
            print(f"[{tag}]   at {os.path.basename(fr.filename)}:"
                  f"{fr.lineno} in {fr.name}", flush=True)
        cur = cur.__context__ or cur.__cause__
        depth += 1


def reset_cache(cache, dev):
    """把 StaticCache 复位到「未写」状态（复用同一批 buffer，不重新分配）。

    cumulative_length.zero_()（设备张量 in-place）+ KV 清零。
    has_previous_state / is_initialized 等 python 标志保持 True（正合意：
    懒初始化与首写分支已在 prefill 走过，decode 路径恒定）。"""
    n_full = 0
    for layer in cache.layers:
        if hasattr(layer, "keys"):  # StaticLayer（含滑动/索引变体）
            if layer.keys is None:
                # 惰性层尚未分配：首写 update 时按层所在设备建 buffer
                continue
            layer.cumulative_length.zero_()
            layer.keys.zero_()
            layer.values.zero_()
            n_full += 1
    torch.cuda.synchronize()
    return n_full


@torch.no_grad()
def static_prefill(model, sample, prompt_len, cache):
    """静态 cache 下的 eager prefill（图外），返回末位 logits。"""
    dev = emb_device(model)
    ids = sample[:prompt_len].unsqueeze(0).to(dev)
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    out = model(input_ids=ids, attention_mask=None,
                past_key_values=cache, use_cache=True)
    ev1.record()
    ev0.synchronize()
    ev1.synchronize()
    # prefill 计时（每次调用都打：B2 复跑多实例，可顺带看方差；
    # 与 eager_reference 的 DynamicCache 口径对比速度差）
    print(f"[static] prefill StaticCache len={prompt_len}: "
          f"{ev0.elapsed_time(ev1):.1f} ms", flush=True)
    logits = out.logits[0, -1].float().cpu()
    del out
    return logits


def run_eager_refs(model, sample, plen_list, steps_list, max_steps):
    """eager 参考（DynamicCache 默认路径，每档 prompt 只跑最长 steps 档）。

    greedy 逐 token 生成，短档序列必然是最长档前缀，前缀对拍即可；logits
    只留首档 prompt（对拍是软判据，capture 保真度与上下文长度无关）。"""
    refs = {}        # plen -> (tokens, step_ms)
    ref_logits = None
    eager_med = {}   # (plen, steps) -> ms
    for plen in plen_list:
        print(f"[graph] eager reference decode (plen={plen}, {max_steps} steps, "
              f"covers {steps_list} as prefixes) ...", flush=True)
        toks, lgs, ms = eager_reference(model, sample, plen, max_steps)
        refs[plen] = (toks, ms)
        if ref_logits is None:
            ref_logits = lgs
        else:
            del lgs
        for s in steps_list:
            eager_med[(plen, s)] = statistics.median(ms[:s])
            print(f"[graph] eager plen={plen:4d} {s:4d} steps: median "
                  f"{eager_med[(plen, s)]:.1f}ms/step "
                  f"({1000.0 / eager_med[(plen, s)]:.2f} tok/s)", flush=True)
    return refs, ref_logits, eager_med


def classify_b2_divergence(div, rerun_diff):
    """pass B2 分叉定性（NaN / gate bug / tie-flip / drift / ambiguous）。

    tie-flip 判据用参考 top-1 幅度下的 bf16 ulp 相对阈值：绝对阈值 1e-2
    在大幅度 logits 下永不可达（幅度 ~2^5 时 1 ulp=0.125；qwen3 1031 实测
    平局翻转 logits diff 0.1875 被误判 ambiguous）。老 JSON 无
    eager_top1_exp 键时回落旧 1e-2 绝对判据。
    """
    if div is not None and div["graph_logits_has_nan"]:
        # 闭环 step logits 含 NaN（argmax 落 0）：非 tie-flip，需逐层
        # NaN 溯源（静态 scatter 尾的 padding 清零已修，此处留证据链）
        return "NaN in closed-loop logits (state corruption, not a tie-flip)"
    if div is not None:
        if div["logits_max_abs"] < 1e-6:
            return "CONTRADICTION: identical logits but argmax differs (gate bug)"
        exp1 = div.get("eager_top1_exp")
        # bf16 尾数 7 位：幅度 2^(exp-1)~2^exp 区间内 ulp = 2^(exp-8)，
        # 4 ulp 余量吸收 vocab 内其它幅度条目的舍入差
        tol = 4.0 * 2.0 ** (exp1 - 8) if exp1 is not None else 1e-2
        if (div["logits_max_abs"] <= tol and div["graph_top2_gap"] <= tol
                and div["eager_top2_gap"] <= tol):
            return "tie-flip (likely benign)"
        if div["logits_max_abs"] > 1.0:
            return "state divergence (drift amplified)"
        return "ambiguous (inspect numbers below)"
    if rerun_diff == -1:
        return ("MATCH on rerun: pass B DIFF not reproduced "
                "(run-to-run nondeterminism)")
    return f"diverged again @step {rerun_diff} (non-p0 prompt, no logits evidence)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant-dir", default=None)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="用 checkpoint 内自定义 modeling（默认 False 走 "
                         "transformers 5.20 原生；dsv2 必须 False，见文件头）")
    ap.add_argument("--mode", default="wxa8", choices=["wxa8"],
                    help="推理模式。静态派发只移植了 wxa8（wxa16 的 MoE host "
                         "编排进 capture 必死，脚本不接受）")
    ap.add_argument("--prompt-len", default="128",
                    help="逗号分隔多档 prompt 长度（如 16,128,2048）；"
                         "与 --steps 全交叉，一次 capture 全档复用")
    ap.add_argument("--steps", default="64",
                    help="逗号分隔多档步数（如 64,256,512）；eager 参考"
                         "只跑最长档，短档按前缀对拍")
    ap.add_argument("--warmup-steps", type=int, default=3,
                    help="capture 前侧流捂热步数（也是惰性编译与静态派发"
                         "状态构建的触发步；消耗同名数量的 cache 槽位，"
                         "已计入 max_cache_len）")
    ap.add_argument("--mask-mode", choices=["native", "buffer"], default="native",
                    help="native=框架自建 mask（StaticCache 下形状静态，值若走"
                         "设备张量则 capture 正确——本 PoC 直接检验）；"
                         "buffer=自持 (1,1,1,max_len) 加性 mask 逐步原地更新")
    ap.add_argument("--reps", type=int, default=1, help="graph 循环重复轮数（取中位）")
    ap.add_argument("--dataset", default="wiki", choices=["wiki", "c4"])
    ap.add_argument("--sample-offset", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.quant_dir:
        ap.error("need --quant-dir")

    from transformers import StaticCache

    model, tokenizer, model_id, mode_label = load_bench_model(args)
    sample = load_sample_tokens(args.dataset, tokenizer)
    text = _text_tower(model)
    print(f"[graph] layers: total={len(text.layers)} "
          f"(full attention only in DartMoQ target models)", flush=True)

    dev = emb_device(model)
    steps_list = [int(s) for s in str(args.steps).split(",") if s.strip()]
    plen_list = [int(p) for p in str(args.prompt_len).split(",") if p.strip()]
    max_steps = max(steps_list)
    max_plen = max(plen_list)
    max_cache_len = max_plen + max_steps + args.warmup_steps + 16

    # ---- 1) eager 参考（DynamicCache 默认路径，每档 prompt 跑最长 steps 档）----
    refs, ref_logits, eager_med = run_eager_refs(
        model, sample, plen_list, steps_list, max_steps)

    # MoE 静态派发（CUDA graph 前置）：eager 基线保持 host 编排口径，此处
    # （基线之后、warmup 之前）打开；warmup 首个 decode 步以 eager 方式构建
    # 池状态，随后 capture/重放全程设备端定形。wxa8 已实现（本脚本只收
    # wxa8）；未实现静态化的子类会被 MoE forward 的 _STATIC_DISPATCH_READY
    # 守卫挡掉
    os.environ["DARTMOQ_STATIC_DISPATCH"] = "1"
    print("[graph] DARTMOQ_STATIC_DISPATCH=1 "
          f"(MoE static dispatch enabled for {args.mode})", flush=True)

    # ---- 2) 静态 cache + prefill（槽位按 max(prompt)+max(steps) 预分配）----
    static_cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
    p0 = plen_list[0]  # 捂热/捕获/logits 对拍的基准档
    pf_logits = static_prefill(model, sample, p0, static_cache)
    tok0 = int(pf_logits.argmax())
    print(f"[graph] static prefill ok; first token matches eager: "
          f"{tok0 == refs[p0][0][0]}", flush=True)

    # ---- 3) 静态 buffer 与单步函数 ----
    ids_buf = torch.zeros(1, 1, dtype=torch.long, device=dev)
    # 原生 5.20 modeling 的 position_ids 是 (batch, seq) 2D（Qwen3.5 的
    # (4,1,1) 是其自定义 wrapper 的形状约定，不适用于原生路径）
    pos_buf = torch.zeros(1, 1, dtype=torch.long, device=dev)
    if args.mask_mode == "buffer":
        # 激活 dtype 取 embed 权重 dtype（量化模型的激活 dtype 依据）
        dt = text.embed_tokens.weight.dtype
        mask_buf = torch.zeros(1, 1, 1, max_cache_len, dtype=dt, device=dev)
    else:
        mask_buf = None

    def feed(pos, token):
        """图外每步的静态 buffer 更新（in-place，无新分配）。"""
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

    # ---- 4) 侧流捂热（teacher-forced 喂参考 token，状态保持一致）----
    print(f"[graph] warmup {args.warmup_steps} steps on side stream "
          f"(triggers lazy kernel compile + static dispatch state build) ...",
          flush=True)
    reset_cache(static_cache, dev)
    static_prefill(model, sample, p0, static_cache)
    side = torch.cuda.Stream(device=dev)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for k in range(args.warmup_steps):
            feed(p0 + k, refs[p0][0][k])
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize(dev)

    # ---- 5) 捕获 ----
    print("[graph] capturing decode step ...", flush=True)
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            g_logits = step()
    except Exception as e:
        print(f"[graph] CAPTURE FAILED: {type(e).__name__}: {e}\n"
              f"[graph] hints: mask-mode native 的 host 分支 —— retry with "
              f"--mask-mode buffer；或模型含非静态化算子（看异常链）", flush=True)
        dump_exc_chain("graph", e)
        sys.exit(2)
    print("[graph] capture ok", flush=True)

    # ---- 6) pass B：prompt×steps 逐档 复位 → prefill → 重放（计时 + 序列对拍）----
    runs = []
    for plen in plen_list:
        for s in steps_list:
            ref_tokens = refs[plen][0]
            loop_stats, replay_stats, seq_ok, first_diff = [], [], True, -1
            for rep in range(args.reps):
                reset_cache(static_cache, dev)
                pf = static_prefill(model, sample, plen, static_cache)
                graph_tokens = [int(pf.argmax())]
                t0 = time.time()
                for i in range(s):
                    feed(plen + i, graph_tokens[-1])
                    ev0 = torch.cuda.Event(enable_timing=True)
                    ev1 = torch.cuda.Event(enable_timing=True)
                    ev0.record()
                    g.replay()
                    ev1.record()
                    tok = int(g_logits[0, -1].argmax().item())
                    ev0.synchronize()
                    ev1.synchronize()
                    replay_stats.append(ev0.elapsed_time(ev1))
                    graph_tokens.append(tok)
                loop_stats.append((time.time() - t0) / s * 1000)
                if seq_ok and graph_tokens != ref_tokens[: s + 1]:
                    seq_ok = False
                    for i, (a, b) in enumerate(zip(graph_tokens, ref_tokens)):
                        if a != b:
                            first_diff = i
                            break
            med_loop = statistics.median(loop_stats)
            runs.append({
                "prompt_len": plen,
                "steps": s,
                "loop_ms": med_loop,
                "replay_ms": statistics.median(replay_stats),
                "speedup": eager_med[(plen, s)] / med_loop,
                "tok_per_s": 1000.0 / med_loop,
                "token_match": seq_ok,
                "first_diff_step": first_diff,
            })

    print(f"\n===== graph decode (mask={args.mask_mode}) =====")
    print(f"{'plen':>4} {'steps':>5}  {'eager':>12}  {'graph':>12}  {'speedup':>7}  "
          f"{'replay-only':>12}  {'graph tok/s':>11}  tokens")
    for r in runs:
        plen, s = r["prompt_len"], r["steps"]
        label = "MATCH" if r["token_match"] else f"DIFF@{r['first_diff_step']}"
        print(f"{plen:>4} {s:>5}  {eager_med[(plen, s)]:>10.1f}ms  "
              f"{r['loop_ms']:>10.1f}ms  {r['speedup']:>6.2f}x  "
              f"{r['replay_ms']:>10.1f}ms  {r['tok_per_s']:>11.2f}  {label}")

    # ---- 6b) pass B2：闭环分叉诊断（仅 token_match=False 的档重跑一遍闭环；
    #       独立于 pass B 计时循环，不污染 loop_ms）----
    # 区分三类分叉：平局翻转（benign）/ 状态失配（drift 放大）/ logits 含
    # NaN（padding 行污染一类，静态 scatter 尾已修但留证据链）。
    #   - 分叉步 logits diff + 两侧 top-2 gap → 区分上述三类
    #   - prefill parity（vs eager 参考的 prefill logits）= 每次 run 的状态
    #     指纹；诊断 run 复现不了 pass B 的分叉 ⇒ run 间非确定性的直接证据
    # logits 对拍仅对首档 prompt 有效（ref_logits 只留首档），其余档只看
    # token 复现性。
    gate_diag = []
    for r in runs:
        if r["token_match"]:
            continue
        plen, s = r["prompt_len"], r["steps"]
        reset_cache(static_cache, dev)
        pf = static_prefill(model, sample, plen, static_cache)
        pf_parity = ((pf - ref_logits[0]).abs().max().item()
                     if plen == p0 else -1.0)
        tok0 = int(pf.argmax())
        # prefill token 是否与 eager 参考一致：B2 的 prefill 与 setup 时的
        # prefill 是两次独立执行，不一致 ⇒ prefill 本身非确定
        pf_tok_match = (None if plen != p0 else (tok0 == refs[p0][0][0]))
        diag_tokens = [tok0]
        cl_parity, div, matching = 0.0, None, True
        for i in range(s):
            feed(plen + i, diag_tokens[-1])
            g.replay()
            # detach：step() 的输出带 requires_grad（capture 在 no_grad 外），
            # 后续 float()/topk() 会触发 UserWarning
            gl = g_logits[0, -1].detach().float().cpu()
            tok = int(gl.argmax().item())
            diag_tokens.append(tok)
            if plen != p0 or not matching:
                continue
            d = (gl - ref_logits[i + 1]).abs().max().item()
            # NaN 用 max(0.0, nan) 会被吞掉（nan>x 为 False），显式保留
            if d != d or d > cl_parity:
                cl_parity = d
            if tok != refs[p0][0][i + 1]:
                matching = False
                gt = gl.topk(2).values
                et = ref_logits[i + 1].topk(2).values
                div = {
                    "step": i + 1,
                    "logits_max_abs": d,
                    "graph_logits_has_nan": bool(torch.isnan(gl).any().item()),
                    "graph_top2_gap": float(gt[0] - gt[1]),
                    "eager_top2_gap": float(et[0] - et[1]),
                    # 参考 top-1 的 frexp 阶数，B2 判据按该幅度下的 bf16
                    # ulp（2^(exp-8)，尾数 7 位）定阈值
                    "eager_top1_exp": math.frexp(abs(float(et[0])))[1],
                    "graph_tok": tok,
                    "eager_tok": refs[p0][0][i + 1],
                }
        rerun_diff = next(
            (i for i, (a, b) in enumerate(zip(diag_tokens, refs[plen][0]))
             if a != b), -1)
        verdict = classify_b2_divergence(div, rerun_diff)
        gate_diag.append({
            "prompt_len": plen, "steps": s,
            "prefill_parity_max_abs": pf_parity,
            "prefill_tok_match": pf_tok_match,
            "closed_loop_parity_max_abs": cl_parity,
            "rerun_first_diff_step": rerun_diff,
            "divergence": div,
            "verdict": verdict,
        })
        print(f"[graph] gate diag plen={plen} steps={s}: "
              f"prefill_parity={pf_parity:.3e} "
              f"prefill_tok_match={pf_tok_match} "
              f"closed_loop_parity={cl_parity:.3e} "
              f"rerun_diff@{rerun_diff} -> {verdict}", flush=True)
        if div is not None:
            print(f"[graph]   div step={div['step']} "
                  f"logits_diff={div['logits_max_abs']:.3e} "
                  f"graph_nan={div['graph_logits_has_nan']} "
                  f"graph_top2_gap={div['graph_top2_gap']:.3e} "
                  f"eager_top2_gap={div['eager_top2_gap']:.3e} "
                  f"tok graph={div['graph_tok']} eager={div['eager_tok']}",
                  flush=True)

    # ---- 7) pass C：teacher-forced logits 对拍（信息性；首档 prompt × 最长
    #      steps 档，覆盖该档全部前缀；capture 保真度与上下文长度无关）----
    reset_cache(static_cache, dev)
    static_prefill(model, sample, p0, static_cache)
    max_abs = 0.0
    for i in range(max_steps):
        feed(p0 + i, refs[p0][0][i])
        g.replay()
        d = (g_logits[0, -1].detach().float().cpu()
             - ref_logits[i + 1]).abs().max().item()
        # 同 B2：max(0.0, nan) 会吞 NaN，显式保留
        if d != d or d > max_abs:
            max_abs = d
    print(f"logits parity (teacher-forced, plen={p0}, {max_steps} steps): "
          f"max_abs={max_abs:.3e}")

    report = {
        "meta": {
            "model_id": model_id, "mode": mode_label, "mask_mode": args.mask_mode,
            "prompt_len_list": plen_list, "steps_list": steps_list,
            "warmup_steps": args.warmup_steps,
            "trust_remote_code": args.trust_remote_code,
            "static_dispatch": True,
            "githash": get_git_hash(),
            "gpu": torch.cuda.get_device_name(0),
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "eager_median_ms": {f"{p}:{s}": eager_med[(p, s)]
                            for p in plen_list for s in steps_list},
        "graph_runs": runs,
        "gate_diag": gate_diag,
        "logits_max_abs": {"prompt_len": p0, "steps": max_steps, "max_abs": max_abs},
    }
    out = args.out or os.path.join(
        "results", f"graph_decode_{model_id}_{mode_label}_"
                   f"{get_git_hash()}_{datetime.now().strftime('%y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[graph] saved: {out}")


if __name__ == "__main__":
    main()
