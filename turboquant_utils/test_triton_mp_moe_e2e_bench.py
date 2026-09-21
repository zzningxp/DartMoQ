#!/usr/bin/env python3
"""
混合比特 WxA16 MoE 端到端基准测试（mini-MoE 集成测试）。

【定位】
  介于单 kernel micro-bench 和全模型 run.q.sh 之间的桥梁。
  构造与真实 Qwen3.5-35B-A3B 相同形状的 WxA16BitPartitionedGroupMoE，
  权重随机初始化，不加载全模型，完整走一遍 forward 流程。

【为什么需要这个测试】
  单 kernel test 形状和调用次数与真实流程差太远，经常出现
  "micro 测出来有收益，run.q.sh 实测没效果甚至变差"的情况。
  本测试在以下维度贴近真实：
    1. 形状贴近：hidden_size / intermediate_size / num_experts / top_k 与真实一致
    2. 权重规模贴近：packed 总大小与真实 MoE 层同量级（超 L2）
    3. 调用模式贴近：per-expert × per-bit 完整循环，含 router / sort / scatter
    4. 缓存状态贴近：区分 cold 首跑与 warm 稳态，warm 才是性能结论
    5. 参数多样性贴近：col_start 有 num_experts × num_bits 种不同取值

【用法】
  PYTHONPATH=$PWD conda run -n dart312 python turboquant_utils/test_mp_moe_e2e_bench.py
  PYTHONPATH=$PWD conda run -n dart312 python turboquant_utils/test_mp_moe_e2e_bench.py --batch-size 4 --seq-len 256     # mini_batch 尺度
  PYTHONPATH=$PWD conda run -n dart312 python turboquant_utils/test_mp_moe_e2e_bench.py --batch-size 1 --seq-len 2048     # eval 单样本尺度
  PYTHONPATH=$PWD conda run -n dart312 python turboquant_utils/test_mp_moe_e2e_bench.py --num-experts 64 --top-k 4         # 小规模快速验证
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantization.wxa16.bit_partitioned_moe import WxA16BitPartitionedGroupMoE, WxA16Weights
from turboquant_utils.quantize import (
    turboquant_quantize_packed_full,
    turboquant_dequantize_packed_rows,
    turboquant_dequantize_packed_cols,
)
from turboquant_utils.rotation import clear_rotation_cache


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def build_mixed_bit_moe(
    num_experts,
    hidden_size,
    intermediate_size,
    top_k,
    bit_allocation,   # dict: {bit_width: neuron_count_per_expert}
    group_size=128,
    device='cuda',
    dtype=torch.float16,
):
    """
    构造一个混合比特的 WxA16BitPartitionedGroupMoE，权重随机初始化并量化。

    bit_allocation: 每个 bit 宽度对应的每 expert 神经元数
      例如 {2: 1024, 4: 1024} 表示每 expert 有 1024 个 2-bit 神经元 + 1024 个 4-bit 神经元
    """
    # 验证 neuron 数是 group_size 的整数倍
    for bit, count in bit_allocation.items():
        assert count % group_size == 0, \
            f"bit={bit}: {count} 不是 {group_size} 的整数倍"

    total_neurons = sum(bit_allocation.values())
    assert total_neurons <= intermediate_size, \
        f"总神经元 {total_neurons} > intermediate_size {intermediate_size}"

    # 创建 gate
    gate = nn.Linear(hidden_size, num_experts, bias=False, device=device, dtype=dtype)
    # 让 gate 权重有一定差异，模拟真实 router 分布
    nn.init.normal_(gate.weight, mean=0.0, std=0.02)

    # 创建 MoE 实例
    moe = WxA16BitPartitionedGroupMoE(
        gate=gate,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        shared_expert=None,
        shared_expert_gate=None,
    )

    moe.bit_list = sorted(bit_allocation.keys(), reverse=True)

    # 对每个 bit，构造连续排列的权重并量化
    # 所有 expert 的同一 bit 神经元连续排列（bit-partitioned 格式）
    for bit, neurons_per_exp in bit_allocation.items():
        bit_str = str(bit)
        total_neurons_bit = neurons_per_exp * num_experts

        # 构造 gate_up 和 down 的 fp16 权重（随机）
        # gate_up: 形状 (2*total_neurons_bit, hidden_size)
        gate_up_fp16 = torch.randn(2 * total_neurons_bit, hidden_size,
                                   device=device, dtype=dtype) * 0.02
        # down: 形状 (hidden_size, total_neurons_bit)
        down_fp16 = torch.randn(hidden_size, total_neurons_bit,
                                device=device, dtype=dtype) * 0.02

        # 量化为 packed 格式
        gate_up_packed = turboquant_quantize_packed_full(
            gate_up_fp16, bit_width=bit, group_size=group_size,
            seed=42 + bit, keep_on_gpu=True,
        )
        down_packed = turboquant_quantize_packed_full(
            down_fp16, bit_width=bit, group_size=group_size,
            seed=42 + bit + 1000, keep_on_gpu=True,
        )

        # 创建 WxA16Weights 并存入
        wxa16_weights = WxA16Weights(bit, hidden_size)
        wxa16_weights.set_packed_data(gate_up_packed, down_packed)
        moe.bit_weights[bit_str] = wxa16_weights

        # expert 偏移量
        offsets = torch.arange(0, (num_experts + 1) * neurons_per_exp, neurons_per_exp,
                               dtype=torch.long, device=device)
        moe.expert_offsets[bit_str] = offsets
        moe.inter_size_by_bit[bit] = neurons_per_exp

    return moe


def build_fp16_weight_cache(moe, device='cuda'):
    """
    预反量化所有 bit 的 gate_up / down 权重为 FP16，缓存起来。
    返回 {bit_str: {'gate_up': fp16_tensor, 'down': fp16_tensor}}。
    用于 FP16 baseline 性能测试时避免每次 forward 都反量化。
    """
    cache = {}
    for bit_str in moe.bit_weights.keys():
        wxa16_weights = moe.bit_weights[bit_str]
        gate_up_packed = wxa16_weights.gate_up_packed
        down_packed = wxa16_weights.down_packed

        # 完整反量化（所有行/列）
        M_gate_up = gate_up_packed['shape'][0]
        N_gate_up = gate_up_packed['shape'][1]
        w_gate_up_full = turboquant_dequantize_packed_rows(
            gate_up_packed, 0, M_gate_up, device=device
        ).to(torch.float16)

        M_down = down_packed['shape'][0]
        N_down = down_packed['shape'][1]
        w_down_full = turboquant_dequantize_packed_cols(
            down_packed, 0, N_down, device=device
        ).to(torch.float16)

        cache[bit_str] = {
            'gate_up': w_gate_up_full,
            'down': w_down_full,
        }
    return cache


def reference_forward_fp16(moe, hidden_states, fp16_weight_cache=None):
    """
    FP16 参考 forward：同样的 bit-partitioned MoE 结构，
    但用预反量化的 FP16 权重 + cuBLAS matmul 计算。

    逻辑完全镜像 WxA16BitPartitionedGroupMoE.forward：
    - router → sort → per-expert 循环 → per-bit 循环 → gate_up → silu → down → scatter

    fp16_weight_cache: 预反量化的权重缓存（build_fp16_weight_cache 的返回值）。
                       如果为 None，每次 forward 都实时反量化（慢，只用于正确性验证）。
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    x = hidden_states.reshape(-1, hidden_dim)
    final_hidden_states = torch.zeros_like(x)

    # Router
    gate_output = moe.gate(x)
    router_logits = gate_output.softmax(dim=-1)
    topk_weights, topk_indices = router_logits.topk(moe.top_k, dim=-1)
    del gate_output, router_logits

    # 按 expert 排序
    flat_expert_indices = topk_indices.flatten()
    flat_expert_weights = topk_weights.flatten()
    flat_token_indices = torch.arange(x.shape[0], device=x.device).repeat_interleave(moe.top_k)

    idxs = flat_expert_indices.argsort()
    sorted_experts = flat_expert_indices[idxs]
    sorted_weights = flat_expert_weights[idxs]
    sorted_tokens = flat_token_indices[idxs]

    tokens_per_expert = sorted_experts.bincount(minlength=moe.num_experts).cpu().numpy().cumsum(0)

    # per-expert 循环
    for expert_idx in range(moe.num_experts):
        end_idx = tokens_per_expert[expert_idx]
        start_idx = 0 if expert_idx == 0 else tokens_per_expert[expert_idx - 1]
        if start_idx == end_idx:
            continue

        exp_token_idx = sorted_tokens[start_idx:end_idx]
        expert_tokens = x[exp_token_idx]
        expert_weights = sorted_weights[start_idx:end_idx].unsqueeze(1)

        expert_out = torch.zeros_like(expert_tokens)

        for bit_str in moe.bit_weights.keys():
            bit = int(bit_str)
            wxa16_weights = moe.bit_weights[bit_str]
            expert_offsets = moe.expert_offsets[bit_str]

            start = int(expert_offsets[expert_idx].item())
            end = int(expert_offsets[expert_idx + 1].item())
            actual_inter_size = end - start

            if actual_inter_size == 0:
                continue

            if fp16_weight_cache is not None:
                # 使用预反量化缓存（性能测试用）
                w_cache = fp16_weight_cache[bit_str]
                w_gate_up_slice = w_cache['gate_up'][2*start:2*end, :]
                w_down_slice = w_cache['down'][:, start:end]
            else:
                # 实时反量化（正确性验证用）
                gate_up_packed = wxa16_weights.gate_up_packed
                w_gate_up_slice = turboquant_dequantize_packed_rows(
                    gate_up_packed, 2 * start, 2 * end, device=x.device
                ).to(torch.float16)
                down_packed = wxa16_weights.down_packed
                w_down_slice = turboquant_dequantize_packed_cols(
                    down_packed, start, end, device=x.device
                ).to(torch.float16)

            # gate_up matmul
            gate_up_out = expert_tokens @ w_gate_up_slice.T
            gate_out = gate_up_out[:, :actual_inter_size]
            up_out = gate_up_out[:, actual_inter_size:]
            del gate_up_out
            if fp16_weight_cache is None:
                del w_gate_up_slice

            act_out = F.silu(gate_out) * up_out
            del gate_out, up_out

            # down matmul
            down_out = act_out @ w_down_slice.T
            del act_out
            if fp16_weight_cache is None:
                del w_down_slice

            expert_out += down_out

        expert_out.mul_(expert_weights)
        final_hidden_states.scatter_reduce_(
            0,
            exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
            expert_out,
            reduce='sum'
        )
        del expert_out, expert_tokens, expert_weights, exp_token_idx

    del flat_expert_indices, flat_expert_weights, flat_token_indices
    del idxs, sorted_experts, sorted_weights, sorted_tokens, tokens_per_expert

    return final_hidden_states.reshape(batch_size, seq_len, hidden_dim)


def verify_correctness(moe, hidden_states, atol=0.5, rtol=1e-3):
    """
    验证 Triton MoE 输出与 FP16 reference 的数值一致性。
    返回 (max_diff, mean_diff, is_close)。
    """
    # Triton 输出
    out_triton = moe(hidden_states)
    # FP16 reference
    out_ref = reference_forward_fp16(moe, hidden_states)

    diff = (out_triton.float() - out_ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    ref_abs_max = out_ref.float().abs().max().item()
    is_close = max_diff < atol or max_diff / max(ref_abs_max, 1e-8) < rtol

    return max_diff, mean_diff, is_close, ref_abs_max


class SimpleRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class SimpleGQAttention(nn.Module):
    """
    简化的 GQA Attention，用作 MoE 之外的背景开销参考。
    参数规模与 Qwen3.5-35B-A3B 对齐：
      num_heads=16, num_kv_heads=2, head_dim=128 (hidden_size=2048)
    """
    def __init__(self, hidden_size, num_heads=16, num_kv_heads=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, x):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # GQA: expand kv heads to match q heads
        if self.num_heads != self.num_kv_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k[:, :, None, :, :].expand(B, self.num_kv_heads, n_rep, T, self.head_dim).reshape(B, self.num_heads, T, self.head_dim)
            v = v[:, :, None, :, :].expand(B, self.num_kv_heads, n_rep, T, self.head_dim).reshape(B, self.num_heads, T, self.head_dim)

        # scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        out = self.o_proj(out)
        return out


class SimpleTransformerLayer(nn.Module):
    """
    简化的 Transformer Decoder Layer:
      x → RMSNorm → Attention → residual → RMSNorm → MoE → residual

    用真实 Qwen3.5-35B-A3B 的形状参数。
    """
    def __init__(self, hidden_size, moe, num_heads=16, num_kv_heads=2):
        super().__init__()
        self.input_layernorm = SimpleRMSNorm(hidden_size)
        self.self_attn = SimpleGQAttention(hidden_size, num_heads, num_kv_heads)
        self.post_attention_layernorm = SimpleRMSNorm(hidden_size)
        self.mlp = moe  # WxA16BitPartitionedGroupMoE

        # 阶段计时
        self.timings = {}

    def forward(self, x):
        self.timings = {}

        torch.cuda.synchronize()
        t0 = time.time()

        # Attention block
        t_attn_start = time.time()
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = residual + x
        torch.cuda.synchronize()
        self.timings['attention'] = time.time() - t_attn_start

        # MoE block
        t_moe_start = time.time()
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        torch.cuda.synchronize()
        self.timings['moe'] = time.time() - t_moe_start

        torch.cuda.synchronize()
        self.timings['total'] = time.time() - t0

        return x

    def get_last_timings(self):
        return self.timings.copy()


def time_forward(module, hidden_states, n_warmup=2, n_repeat=5):
    """
    测 forward 时间，返回 (avg_time, times_list, avg_timings_dict)。
    如果 module 有 last_timings 属性或 get_last_timings() 方法，
    同时返回阶段平均时间。
    warmup 阶段不计入统计。
    """
    # Warmup
    for i in range(n_warmup):
        out = module(hidden_states)
        del out
    torch.cuda.synchronize()

    # 正式测量
    times = []
    timing_dicts = []
    for i in range(n_repeat):
        torch.cuda.synchronize()
        t0 = time.time()
        out = module(hidden_states)
        torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)
        # 支持 last_timings 属性 或 get_last_timings() 方法
        if hasattr(module, 'last_timings') and module.last_timings:
            timing_dicts.append(module.last_timings.copy())
        elif hasattr(module, 'get_last_timings'):
            timing_dicts.append(module.get_last_timings())
        del out

    avg = sum(times) / len(times)

    avg_timings = {}
    if timing_dicts:
        for key in timing_dicts[0].keys():
            vals = [d[key] for d in timing_dicts if key in d]
            if vals and isinstance(vals[0], (int, float)):
                avg_timings[key] = sum(vals) / len(vals)

    return avg, times, avg_timings


def estimate_per_expert_b(moe, hidden_states):
    """估算 per-expert 的平均 token 数（B）。"""
    batch_size, seq_len, hidden_dim = hidden_states.shape
    total_tokens = batch_size * seq_len * moe.top_k
    # 假设均匀分布，真实分布会更不均匀
    avg_b_per_expert = total_tokens / moe.num_experts
    return avg_b_per_expert


def main():
    parser = argparse.ArgumentParser(description="混合比特 WxA16 MoE 端到端基准测试")

    # 模型形状参数（默认接近 Qwen3.5-35B-A3B 的 MoE 层）
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=2816)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=128)

    # 输入形状
    parser.add_argument("--batch-size", type=int, default=4,
                        help="batch size（默认 4，对应 mini_batch 尺度）")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="seq len（默认 256，对应 mini_batch 尺度）")

    # bit 分布
    parser.add_argument("--bits", type=str, default="2:1408,4:1408",
                        help="bit 分布，格式 'bit:neurons_per_expert,...' "
                             "（默认 2-bit 1408 + 4-bit 1408，共 2816 神经元）")

    # 测试参数
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 解析 bit 分布
    bit_allocation = {}
    for part in args.bits.split(","):
        bit, count = part.strip().split(":")
        bit_allocation[int(bit)] = int(count)

    print("=" * 80)
    print("混合比特 WxA16 MoE 端到端基准测试")
    print("=" * 80)
    print(f"\n模型参数:")
    print(f"  hidden_size:       {args.hidden_size}")
    print(f"  intermediate_size: {args.intermediate_size}")
    print(f"  num_experts:       {args.num_experts}")
    print(f"  top_k:             {args.top_k}")
    print(f"  group_size:        {args.group_size}")
    print(f"  bit 分布:          {bit_allocation}")
    total_neurons = sum(bit_allocation.values())
    print(f"  神经元总数/expert: {total_neurons} "
          f"({total_neurons/args.intermediate_size*100:.1f}% of intermediate_size)")

    print(f"\n输入参数:")
    print(f"  batch_size:        {args.batch_size}")
    print(f"  seq_len:           {args.seq_len}")
    print(f"  总 token 数:       {args.batch_size * args.seq_len}")

    # 估算显存占用
    # packed 权重：每 neuron 约 bit/8 bytes + codebook + norms
    total_packed_bytes = 0
    for bit, neurons_per_exp in bit_allocation.items():
        total_neurons_bit = neurons_per_exp * args.num_experts
        # gate_up
        total_packed_bytes += total_neurons_bit * 2 * args.hidden_size * bit / 8
        # down
        total_packed_bytes += args.hidden_size * total_neurons_bit * bit / 8
        # norms (fp16)
        num_groups = args.hidden_size // args.group_size
        total_packed_bytes += total_neurons_bit * 2 * num_groups * 2  # gate_up norms
        total_packed_bytes += args.hidden_size * num_groups * 2        # down norms
        # codebook (fp16, 很小忽略) — Step 2: codebook 已改为 fp16

    print(f"\n权重规模估算:")
    print(f"  packed 总大小 ≈    {total_packed_bytes / 1024**2:.1f} MB")
    print(f"  (L2 缓存约 64MB，L2 驻留: {'否' if total_packed_bytes > 64*1024**2 else '是'})")

    # 估算 per-expert B
    total_tokens = args.batch_size * args.seq_len
    avg_b = total_tokens * args.top_k / args.num_experts
    print(f"\nPer-expert 估算:")
    print(f"  平均 B/expert:     ~{avg_b:.0f} tokens "
          f"(均匀分布假设，真实更不均匀)")
    print(f"  gate_up kernel 次/forward:  ~{args.num_experts * len(bit_allocation) * (args.hidden_size // args.group_size)}")
    print(f"  down kernel 次/forward:    ~{args.num_experts * len(bit_allocation) * (total_neurons // args.group_size)}")

    print("\n" + "=" * 80)
    print("构造 MoE 模型（随机权重 + 量化）...")
    print("=" * 80)

    t0 = time.time()
    moe = build_mixed_bit_moe(
        num_experts=args.num_experts,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        top_k=args.top_k,
        bit_allocation=bit_allocation,
        group_size=args.group_size,
        device=device,
        dtype=torch.float16,
    )
    torch.cuda.synchronize()
    t_build = time.time() - t0
    print(f"构造完成，耗时 {t_build:.2f}s")

    # 构造输入
    hidden_states = torch.randn(
        args.batch_size, args.seq_len, args.hidden_size,
        device=device, dtype=torch.float16
    )

    # ====== 测试 1：Cold 首跑（含 QR 重算 + 编译缓存） ======
    print("\n" + "=" * 80)
    print("[测试 1] Cold 首跑时间（含 QR 重算 + Triton 编译缓存冷启动）")
    print("=" * 80)

    # 确保缓存全冷
    clear_rotation_cache()

    torch.cuda.synchronize()
    t0 = time.time()
    out_cold = moe(hidden_states)
    torch.cuda.synchronize()
    t_cold = time.time() - t0
    del out_cold

    print(f"  Cold forward: {t_cold * 1000:.2f} ms")

    # ====== 测试 2：精度验证（Triton vs FP16 reference） ======
    print("\n" + "=" * 80)
    print("[测试 2] 数值正确性验证：Triton MoE vs FP16 反量化 reference")
    print("=" * 80)
    print("  （完整端到端 forward：router → sort → per-expert×per-bit → scatter）")

    max_diff, mean_diff, is_close, ref_abs_max = verify_correctness(moe, hidden_states)
    print(f"\n  参考输出 abs max: {ref_abs_max:.4f}")
    print(f"  最大绝对误差:    {max_diff:.6f}")
    print(f"  平均绝对误差:    {mean_diff:.6f}")
    print(f"  相对误差(max):   {max_diff / max(ref_abs_max, 1e-8) * 100:.4f}%")
    print(f"  结果:            {'✅ OK' if is_close else '❌ FAIL'} "
          f"(阈值: atol=0.5 或 rtol=0.1%)")

    # ====== 测试 3：Warm 稳态时间（真实推理场景） ======
    print("\n" + "=" * 80)
    print(f"[测试 3] Warm 稳态时间（{args.n_warmup} 次 warmup + {args.n_repeat} 次平均）")
    print("=" * 80)

    avg_warm, times_warm, warm_timings = time_forward(
        moe, hidden_states,
        n_warmup=args.n_warmup,
        n_repeat=args.n_repeat,
    )

    print(f"  平均: {avg_warm * 1000:.2f} ms")
    for i, t in enumerate(times_warm):
        print(f"    第 {i+1} 次: {t * 1000:.2f} ms")
    print(f"  Cold/Warm 比: {t_cold / avg_warm:.2f}x")

    # 阶段时间占比（warm 路径平均）
    if warm_timings:
        total = warm_timings.get('total', avg_warm)
        print(f"\n  阶段时间占比 (warm 平均, 总 {total*1000:.2f} ms):")
        stages = [
            ('router',            'Router (gate + topk)'),
            ('sort_scatter_prep', 'Sort + token 准备'),
            ('triton',            'Triton kernel (gate_up+down)'),
            ('compute',           'Compute (per-expert 循环总)'),
            ('cleanup_reshape',   'Cleanup + reshape'),
            ('shared',            'Shared expert'),
            ('init',              'Init (reshape+alloc)'),
        ]
        for key, label in stages:
            val = warm_timings.get(key, 0)
            if not isinstance(val, (int, float)):
                continue
            pct = val / total * 100 if total > 0 else 0
            bar = '#' * int(pct / 2)
            print(f"    {label:30s} {val*1000:8.2f} ms  {pct:5.1f}%  {bar}")
        ae = warm_timings.get('active_experts', '?')
        ab = warm_timings.get('active_bits', '?')
        print(f"    active_experts: {ae}, active_bits: {ab}")

    # ====== 测试 4：FP16 MoE baseline 端到端对比 ======
    print("\n" + "=" * 80)
    print("[测试 4] FP16 MoE vs Triton 混合比特 — 端到端速度对比")
    print("=" * 80)
    print("  （相同权重、相同 router、相同 token 分配，只有 matmul 实现不同：")
    print("     FP16 = 预反量化权重 + cuBLAS matmul；Triton = fused dequant + matmul kernel）")

    # 预反量化所有 bit 的权重（只做一次）
    print("\n  预反量化 FP16 权重...")
    t_dequant_start = time.time()
    fp16_weight_cache = build_fp16_weight_cache(moe, device=device)
    torch.cuda.synchronize()
    print(f"  完成，耗时 {time.time() - t_dequant_start:.2f}s")

    # warmup
    for _ in range(2):
        _ = reference_forward_fp16(moe, hidden_states, fp16_weight_cache)
    torch.cuda.synchronize()

    fp16_times = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.time()
        out_fp16 = reference_forward_fp16(moe, hidden_states, fp16_weight_cache)
        torch.cuda.synchronize()
        fp16_times.append(time.time() - t0)
        del out_fp16

    avg_fp16 = sum(fp16_times) / len(fp16_times)
    ratio = avg_warm / avg_fp16

    print(f"\n  FP16 MoE (cuBLAS):        {avg_fp16 * 1000:8.2f} ms")
    print(f"  Triton 混合比特 MoE:       {avg_warm * 1000:8.2f} ms")
    print(f"  比值 (Triton / FP16):      {ratio:5.2f}x")
    if ratio > 1:
        print(f"  → Triton 比 FP16 慢 {(ratio-1)*100:.0f}%")
    else:
        print(f"  → Triton 比 FP16 快 {(1-ratio)*100:.0f}%")

    # 估算总 FLOPs
    total_flops_gateup = 0
    total_flops_down = 0
    for bit, neurons_per_exp in bit_allocation.items():
        flops_per_exp = 2 * neurons_per_exp * 2 * args.hidden_size * avg_b
        total_flops_gateup += args.num_experts * flops_per_exp
        flops_per_exp_down = 2 * args.hidden_size * neurons_per_exp * avg_b
        total_flops_down += args.num_experts * flops_per_exp_down
    total_flops = total_flops_gateup + total_flops_down
    print(f"\n  估算总 FLOPs/forward:     {total_flops / 1e9:.2f} GFLOPs")
    print(f"  FP16 有效 TFLOPS:         {total_flops / avg_fp16 / 1e12:.2f} TFLOPS")
    print(f"  Triton 有效 TFLOPS:       {total_flops / avg_warm / 1e12:.2f} TFLOPS")
    print(f"  FP16 利用率 (FP16 TFLOPS / 80 TFLOPS peak): {total_flops / avg_fp16 / 80e12 * 100:.1f}%")
    print(f"  Triton 利用率 (Triton / 80 TFLOPS peak):    {total_flops / avg_warm / 80e12 * 100:.1f}%")

    del fp16_weight_cache

    # ====== 测试 5：不同 batch_size 下的扩展性 ======
    print("\n" + "=" * 80)
    print("[测试 5] 不同 token 数下的扩展性（warm 路径）")
    print("=" * 80)

    bsz_list = [1, 2, 4, 8]
    # 过滤掉太大的
    bsz_list = [b for b in bsz_list if b * args.seq_len * args.hidden_size * 2 < 2 * 1024**3]

    results_scaling = []
    for bsz in bsz_list:
        hs = torch.randn(bsz, args.seq_len, args.hidden_size,
                         device=device, dtype=torch.float16)
        avg_t, _, _ = time_forward(moe, hs, n_warmup=2, n_repeat=3)
        total_tok = bsz * args.seq_len
        results_scaling.append((bsz, total_tok, avg_t))
        # 估算 TFLOPS
        avg_b_per = total_tok * args.top_k / args.num_experts
        flops = 0
        for bit, neurons_per_exp in bit_allocation.items():
            flops += args.num_experts * (
                2 * neurons_per_exp * 2 * args.hidden_size * avg_b_per +
                2 * args.hidden_size * neurons_per_exp * avg_b_per
            )
        tflops = flops / avg_t / 1e12
        print(f"  batch={bsz:2d} ({total_tok:5d} tokens): "
              f"{avg_t * 1000:8.2f} ms  ({tflops:.2f} TFLOPS)")
        del hs

    # ====== 总结 ======
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print(f"  配置: num_experts={args.num_experts}, hidden={args.hidden_size}, "
          f"inter={args.intermediate_size}, top_k={args.top_k}")
    print(f"  bit 分布: {bit_allocation}")
    print(f"  输入: batch={args.batch_size}, seq_len={args.seq_len} "
          f"({args.batch_size * args.seq_len} tokens)")
    print(f"  Cold forward: {t_cold * 1000:.2f} ms")
    print(f"  Warm forward: {avg_warm * 1000:.2f} ms (平均)")
    print(f"  Cold/Warm:    {t_cold / avg_warm:.2f}x")
    print(f"  权重大小:     {total_packed_bytes / 1024**2:.1f} MB")
    print(f"  有效 TFLOPS:  {total_flops / avg_warm / 1e12:.2f} TFLOPS (warm)")

    # 显存占用
    if torch.cuda.is_available():
        print(f"\n  GPU 显存: {torch.cuda.memory_allocated() / 1024**2:.1f} MB / "
              f"{torch.cuda.max_memory_allocated() / 1024**2:.1f} MB (peak)")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
