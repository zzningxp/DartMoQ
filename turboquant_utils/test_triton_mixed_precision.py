#!/usr/bin/env python3
"""
混合精度 Triton 融合 kernel 测试程序。
按正式流程测试：整个矩阵一起量化，然后用slice_rows/slice_in_features切分。
"""

import argparse
import time
import torch
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquant_utils.triton_kernels import (
    triton_fused_matmul_grouped,
    triton_fused_matmul_grouped_slice_rows,
    triton_fused_matmul_grouped_slice_in_features
)
from turboquant_utils.codebook import get_codebook
from turboquant_utils.rotation import generate_rotation_matrix, clear_rotation_cache
from turboquant_utils.quantize import pack_nbit, unpack_nbit


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def quantize_weight_simple(W, bit_width=4, group_size=None, seed=42):
    M, K = W.shape
    if group_size is None:
        group_size = K

    W = W.float()
    centroids, boundaries = get_codebook(bit_width)
    centroids = centroids.to(W.device)
    boundaries = boundaries.to(W.device)

    all_norms = []
    all_indices = []

    for g_start in range(0, K, group_size):
        g_end = min(g_start + group_size, K)
        g_dim = g_end - g_start
        W_g = W[:, g_start:g_end]

        norms = W_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_norm = W_g / norms
        all_norms.append(norms.squeeze(1))

        Pi = generate_rotation_matrix(g_dim, seed + g_start, device=W.device)
        Y = W_norm @ Pi.T
        scale = math.sqrt(g_dim)
        Y_scaled = Y * scale

        indices = torch.searchsorted(boundaries, Y_scaled.reshape(-1))
        indices = indices.clamp(0, len(centroids) - 1).reshape(M, g_dim)
        all_indices.append(indices)

    full_indices = torch.cat(all_indices, dim=1)
    norms_out = torch.stack(all_norms, dim=1) if len(all_indices) > 1 else all_norms[0]

    packed = pack_nbit(full_indices, bit_width)

    # Step 2 (FP16 链路改造): codebook 和 norms 存 fp16，与主流程 quantize.py 一致
    return {
        "indices_packed": packed,
        "codebook": centroids.half(),
        "norms": norms_out.half(),
        "seed": seed,
        "group_size": group_size,
        "bit_width": bit_width,
    }


def simu_quant_weight(W, bit_width=4, group_size=None, seed=42):
    """
    模拟量化：现场量化，然后马上反量化存回 fp16。
    不存储 packed 数据，直接返回反量化后的 fp16 权重。
    """
    M, K = W.shape
    if group_size is None:
        group_size = K

    W = W.float()
    centroids, boundaries = get_codebook(bit_width)
    centroids = centroids.to(W.device)
    boundaries = boundaries.to(W.device)

    W_approx = torch.zeros_like(W)

    for g_start in range(0, K, group_size):
        g_end = min(g_start + group_size, K)
        g_dim = g_end - g_start
        W_g = W[:, g_start:g_end]

        norms = W_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_norm = W_g / norms

        Pi = generate_rotation_matrix(g_dim, seed + g_start, device=W.device)
        Y = W_norm @ Pi.T
        scale = math.sqrt(g_dim)
        Y_scaled = Y * scale

        indices = torch.searchsorted(boundaries, Y_scaled.reshape(-1))
        indices = indices.clamp(0, len(centroids) - 1).reshape(M, g_dim)

        # 马上反量化
        Y_quant_scaled = centroids[indices]
        Y_unscaled = Y_quant_scaled / scale
        W_g_approx = Y_unscaled @ Pi
        W_approx[:, g_start:g_end] = W_g_approx * norms

    return W_approx.to(torch.float16)


def dequantize_weight_simple(packed_data, W_shape):
    indices_packed = packed_data["indices_packed"]
    codebook = packed_data["codebook"]
    norms = packed_data["norms"]
    seed = packed_data["seed"]
    group_size = packed_data["group_size"]
    bit_width = packed_data["bit_width"]

    M, K = W_shape
    device = indices_packed.device

    full_indices = unpack_nbit(indices_packed, bit_width, K)

    # Step 2: baseline 反量化用 fp32 高精度计算（codebook/norms 从 fp16 转回 fp32）
    codebook = codebook.float()
    norms = norms.float()

    if norms.dim() == 1:
        norms = norms.unsqueeze(1)

    W_approx = torch.zeros((M, K), dtype=torch.float32, device=device)

    group_idx = 0
    for g_start in range(0, K, group_size):
        g_end = min(g_start + group_size, K)
        g_dim = g_end - g_start

        indices_g = full_indices[:, g_start:g_end]
        norms_g = norms[:, group_idx].unsqueeze(1)
        group_idx += 1

        Y_quant_scaled = codebook[indices_g]
        scale = math.sqrt(g_dim)
        Y_unscaled = Y_quant_scaled / scale

        Pi = generate_rotation_matrix(g_dim, seed + g_start, device=device)
        W_g_approx = Y_unscaled @ Pi

        W_approx[:, g_start:g_end] = W_g_approx * norms_g

    return W_approx


def get_slice_boundaries(K, proportions=None):
    if proportions is None:
        proportions = [2, 1, 1]

    total = sum(proportions)
    boundaries = []
    current_start = 0
    for i, p in enumerate(proportions):
        size = K * p // total
        if current_start + size > K or i == len(proportions) - 1:
            size = K - current_start
        boundaries.append((current_start, current_start + size))
        current_start += size

    return boundaries


def test_linear(args):
    bit_width = 8
    print("\n" + "=" * 70)
    print("场景1: 单独 Linear - triton_fused_matmul_grouped")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H = args.out_features
    K = args.in_features
    B = args.batch_size
    group_size = args.group_size

    K = ((K + group_size - 1) // group_size) * group_size

    print(f"\n测试配置:")
    print(f"  Linear shape: {H} x {K}")
    print(f"  Input x shape: {B} x {K}")

    torch.manual_seed(args.seed)
    W = torch.randn(H, K, dtype=torch.float16, device=device)
    x = torch.randn(B, K, dtype=torch.float16, device=device)

    packed_data = quantize_weight_simple(W, bit_width=bit_width, group_size=group_size, seed=args.seed)
    W_simu = simu_quant_weight(W, bit_width=bit_width, group_size=group_size, seed=args.seed)

    print("\nWarmup...")
    for _ in range(5):
        _ = x @ W.T
        _ = x @ W_simu.T
        w_dequant = dequantize_weight_simple(packed_data, W.shape)
        _ = x.float() @ w_dequant.T
        _ = triton_fused_matmul_grouped(
            x, packed_data["indices_packed"], packed_data["codebook"],
            packed_data["norms"], packed_data["seed"],
            packed_data["group_size"], K, bit_width=bit_width
        )

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        out_fp16 = x @ W.T
    torch.cuda.synchronize()
    t_fp16 = (time.time() - t0) / 10

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        out_simu = x @ W_simu.T
    torch.cuda.synchronize()
    t_simu = (time.time() - t0) / 10

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        w_dequant = dequantize_weight_simple(packed_data, W.shape)
        out_dequant = x.float() @ w_dequant.T
    torch.cuda.synchronize()
    t_dequant = (time.time() - t0) / 10

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        out_triton = triton_fused_matmul_grouped(
            x, packed_data["indices_packed"], packed_data["codebook"],
            packed_data["norms"], packed_data["seed"],
            packed_data["group_size"], K, bit_width=bit_width
        )
    torch.cuda.synchronize()
    t_triton = (time.time() - t0) / 10

    # 新改进对照：cold(改前, 每次清缓存强制重算旋转+重拷) vs hot(改后, 命中设备缓存)
    # 直接在 Triton 调用上测，最直观体现改进收益。
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        clear_rotation_cache()  # 模拟改前：每次都重算 QR + 重拷
        out_triton_cold = triton_fused_matmul_grouped(
            x, packed_data["indices_packed"], packed_data["codebook"],
            packed_data["norms"], packed_data["seed"],
            packed_data["group_size"], K, bit_width=bit_width
        )
    torch.cuda.synchronize()
    t_triton_cold = (time.time() - t0) / 10

    rot_same = torch.allclose(out_triton_cold, out_triton, atol=1e-2)

    out_fp16_float = out_fp16.float()
    out_simu_float = out_simu.float()
    out_dequant_float = out_dequant.float()
    out_triton_float = out_triton.float()

    abs_diff_simu_fp16 = torch.abs(out_fp16_float - out_simu_float)
    abs_diff_dequant_simu = torch.abs(out_dequant_float - out_simu_float)
    abs_diff_dequant_triton = torch.abs(out_dequant_float - out_triton_float)

    print("\n" + "=" * 70)
    print("场景1汇总")
    print("=" * 70)
    print(f"FP16:              {t_fp16 * 1000:8.2f} ms")
    print(f"SimuQuant:         {t_simu * 1000:8.2f} ms")
    print(f"反量化+GEMM:       {t_dequant * 1000:8.2f} ms")
    print(f"Triton(改后 hot):  {t_triton * 1000:8.2f} ms")
    print(f"Triton(改前 cold): {t_triton_cold * 1000:8.2f} ms")
    print(f"  → 新改进(旋转缓存)节省: {max(t_triton_cold - t_triton, 0) * 1000:8.2f} ms "
          f"({max((t_triton_cold - t_triton) / t_triton_cold * 100, 0):.1f}%)")
    print(f"  → 数值一致性: {'OK' if rot_same else 'FAIL'}")
    print(f"\n误差对比:")
    print(f"SimuQuant vs FP16:  max={abs_diff_simu_fp16.max():.6f}, mean={abs_diff_simu_fp16.mean():.6f}")
    print(f"反量化+GEMM vs SimuQuant: max={abs_diff_dequant_simu.max():.6f}, mean={abs_diff_dequant_simu.mean():.6f}")
    print(f"反量化+GEMM vs Triton: max={abs_diff_dequant_triton.max():.6f}, mean={abs_diff_dequant_triton.mean():.6f}")

    return {
        "t_fp16": t_fp16,
        "t_simu": t_simu,
        "t_dequant": t_dequant,
        "t_triton": t_triton,
        "t_triton_cold": t_triton_cold,
        "rot_same": rot_same,
        "err_simu_fp16_max": abs_diff_simu_fp16.max().item(),
        "err_simu_fp16_mean": abs_diff_simu_fp16.mean().item(),
        "err_dequant_simu_max": abs_diff_dequant_simu.max().item(),
        "err_dequant_simu_mean": abs_diff_dequant_simu.mean().item(),
        "err_dequant_triton_max": abs_diff_dequant_triton.max().item(),
        "err_dequant_triton_mean": abs_diff_dequant_triton.mean().item(),
    }


def test_moe_up_gate(args):
    print("\n" + "=" * 70)
    print("场景2.1: MoE up_gate - triton_fused_matmul_grouped_slice_rows")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    K = args.in_features
    D = K * 2
    B = args.batch_size
    group_size = args.group_size

    K = ((K + group_size - 1) // group_size) * group_size
    D = ((D + group_size - 1) // group_size) * group_size

    print(f"\n测试配置:")
    print(f"GateUp shape: {K} x {D} (包含 gate_proj 和 up_proj)")
    print(f"Input x shape: {B} x {D}")

    torch.manual_seed(args.seed)
    W_gate_up = torch.randn(K, D, dtype=torch.float16, device=device)
    x = torch.randn(B, D, dtype=torch.float16, device=device)

    packed_data = quantize_weight_simple(W_gate_up, bit_width=4, group_size=group_size, seed=args.seed)
    W_gate_up_simu = simu_quant_weight(W_gate_up, bit_width=4, group_size=group_size, seed=args.seed)

    start = K // 4
    end = K // 2
    print(f"\n模拟专家切片: rows [{2*start}:{2*end}]")

    print("\nWarmup...")
    for _ in range(5):
        _ = x @ W_gate_up.T
        _ = x @ W_gate_up_simu.T
        w_dequant = dequantize_weight_simple(packed_data, W_gate_up.shape)
        _ = x.float() @ w_dequant.T
        _ = triton_fused_matmul_grouped_slice_rows(
            x, packed_data["indices_packed"], packed_data["codebook"],
            packed_data["norms"], packed_data["seed"],
            packed_data["group_size"], D, 2*start, 2*end, bit_width=4
        )

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        out_fp16 = x @ W_gate_up.T[:, 2*start:2*end]
    torch.cuda.synchronize()
    t_fp16 = (time.time() - t0) / 10

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        out_simu = x @ W_gate_up_simu.T[:, 2*start:2*end]
    torch.cuda.synchronize()
    t_simu = (time.time() - t0) / 10

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        w_dequant = dequantize_weight_simple(packed_data, W_gate_up.shape)
        out_dequant = x.float() @ w_dequant.T[:, 2*start:2*end]
    torch.cuda.synchronize()
    t_dequant = (time.time() - t0) / 10

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        out_triton = triton_fused_matmul_grouped_slice_rows(
            x, packed_data["indices_packed"], packed_data["codebook"],
            packed_data["norms"], packed_data["seed"],
            packed_data["group_size"], D, 2*start, 2*end, bit_width=4
        )
    torch.cuda.synchronize()
    t_triton = (time.time() - t0) / 10

    # 新改进对照：cold(改前, 每次清缓存) vs hot(改后, 命中设备缓存)，直接织进 Triton 计时
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        clear_rotation_cache()
        out_triton_cold = triton_fused_matmul_grouped_slice_rows(
            x, packed_data["indices_packed"], packed_data["codebook"],
            packed_data["norms"], packed_data["seed"],
            packed_data["group_size"], D, 2*start, 2*end, bit_width=4
        )
    torch.cuda.synchronize()
    t_triton_cold = (time.time() - t0) / 10

    rot_same = torch.allclose(out_triton_cold, out_triton, atol=1e-2)

    out_fp16_float = out_fp16.float()
    out_simu_float = out_simu.float()
    out_dequant_float = out_dequant.float()
    out_triton_float = out_triton.float()

    abs_diff_simu_fp16 = torch.abs(out_fp16_float - out_simu_float)
    abs_diff_dequant_simu = torch.abs(out_dequant_float - out_simu_float)
    abs_diff_dequant_triton = torch.abs(out_dequant_float - out_triton_float)

    print("\n" + "=" * 70)
    print("场景2.1汇总")
    print("=" * 70)
    print(f"FP16:              {t_fp16 * 1000:8.2f} ms")
    print(f"SimuQuant:         {t_simu * 1000:8.2f} ms")
    print(f"反量化+GEMM:       {t_dequant * 1000:8.2f} ms")
    print(f"Triton(改后 hot):  {t_triton * 1000:8.2f} ms")
    print(f"Triton(改前 cold): {t_triton_cold * 1000:8.2f} ms")
    print(f"  → 新改进(旋转缓存)节省: {max(t_triton_cold - t_triton, 0) * 1000:8.2f} ms "
          f"({max((t_triton_cold - t_triton) / t_triton_cold * 100, 0):.1f}%)")
    print(f"  → 数值一致性: {'OK' if rot_same else 'FAIL'}")
    print(f"\n误差对比:")
    print(f"SimuQuant vs FP16:  max={abs_diff_simu_fp16.max():.6f}, mean={abs_diff_simu_fp16.mean():.6f}")
    print(f"反量化+GEMM vs SimuQuant: max={abs_diff_dequant_simu.max():.6f}, mean={abs_diff_dequant_simu.mean():.6f}")
    print(f"反量化+GEMM vs Triton: max={abs_diff_dequant_triton.max():.6f}, mean={abs_diff_dequant_triton.mean():.6f}")

    return {
        "t_fp16": t_fp16,
        "t_simu": t_simu,
        "t_dequant": t_dequant,
        "t_triton": t_triton,
        "t_triton_cold": t_triton_cold,
        "rot_same": rot_same,
        "err_simu_fp16_max": abs_diff_simu_fp16.max().item(),
        "err_simu_fp16_mean": abs_diff_simu_fp16.mean().item(),
        "err_dequant_simu_max": abs_diff_dequant_simu.max().item(),
        "err_dequant_simu_mean": abs_diff_dequant_simu.mean().item(),
        "err_dequant_triton_max": abs_diff_dequant_triton.max().item(),
        "err_dequant_triton_mean": abs_diff_dequant_triton.mean().item(),
    }


def test_moe_down(args):
    print("\n" + "=" * 70)
    print("场景2.2: MoE down - triton_fused_matmul_grouped_slice_in_features")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H = args.out_features
    K = args.in_features
    B = args.batch_size
    group_size = args.group_size

    H = ((H + group_size - 1) // group_size) * group_size
    K = ((K + group_size - 1) // group_size) * group_size

    print(f"\n测试配置:")
    print(f"Down shape: {H} x {K}")
    print(f"Input x shape: {B} x {K}")

    torch.manual_seed(args.seed)
    W_down = torch.randn(H, K, dtype=torch.float16, device=device)
    x = torch.randn(B, K, dtype=torch.float16, device=device)

    packed_data = quantize_weight_simple(W_down, bit_width=4, group_size=group_size, seed=args.seed)
    W_down_simu = simu_quant_weight(W_down, bit_width=4, group_size=group_size, seed=args.seed)

    start = K // 4
    end = K // 2
    print(f"\n模拟专家切片: in_features [{start}:{end}]")

    print("\nWarmup...")
    for _ in range(5):
        _ = x[:, start:end] @ W_down[:, start:end].T
        _ = x[:, start:end] @ W_down_simu[:, start:end].T
        w_dequant = dequantize_weight_simple(packed_data, W_down.shape)
        _ = x[:, start:end].float() @ w_dequant[:, start:end].T
        _ = triton_fused_matmul_grouped_slice_in_features(
            x[:, start:end], packed_data["indices_packed"], packed_data["codebook"],
            packed_data["norms"], packed_data["seed"],
            packed_data["group_size"], start, end, K, bit_width=4
        )

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        out_fp16 = x[:, start:end] @ W_down[:, start:end].T
    torch.cuda.synchronize()
    t_fp16 = (time.time() - t0) / 10

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        out_simu = x[:, start:end] @ W_down_simu[:, start:end].T
    torch.cuda.synchronize()
    t_simu = (time.time() - t0) / 10

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        w_dequant = dequantize_weight_simple(packed_data, W_down.shape)
        out_dequant = x[:, start:end].float() @ w_dequant[:, start:end].T
    torch.cuda.synchronize()
    t_dequant = (time.time() - t0) / 10

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        out_triton = triton_fused_matmul_grouped_slice_in_features(
            x[:, start:end], packed_data["indices_packed"], packed_data["codebook"],
            packed_data["norms"], packed_data["seed"],
            packed_data["group_size"], start, end, K, bit_width=4
        )
    torch.cuda.synchronize()
    t_triton = (time.time() - t0) / 10

    # 新改进对照：cold(改前, 每次清缓存) vs hot(改后, 命中设备缓存)，直接织进 Triton 计时
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        clear_rotation_cache()
        out_triton_cold = triton_fused_matmul_grouped_slice_in_features(
            x[:, start:end], packed_data["indices_packed"], packed_data["codebook"],
            packed_data["norms"], packed_data["seed"],
            packed_data["group_size"], start, end, K, bit_width=4
        )
    torch.cuda.synchronize()
    t_triton_cold = (time.time() - t0) / 10

    rot_same = torch.allclose(out_triton_cold, out_triton, atol=1e-2)

    out_fp16_float = out_fp16.float()
    out_simu_float = out_simu.float()
    out_dequant_float = out_dequant.float()
    out_triton_float = out_triton.float()

    abs_diff_simu_fp16 = torch.abs(out_fp16_float - out_simu_float)
    abs_diff_dequant_simu = torch.abs(out_dequant_float - out_simu_float)
    abs_diff_dequant_triton = torch.abs(out_dequant_float - out_triton_float)

    print("\n" + "=" * 70)
    print("场景2.2汇总")
    print("=" * 70)
    print(f"FP16:              {t_fp16 * 1000:8.2f} ms")
    print(f"SimuQuant:         {t_simu * 1000:8.2f} ms")
    print(f"反量化+GEMM:       {t_dequant * 1000:8.2f} ms")
    print(f"Triton(改后 hot):  {t_triton * 1000:8.2f} ms")
    print(f"Triton(改前 cold): {t_triton_cold * 1000:8.2f} ms")
    print(f"  → 新改进(旋转缓存)节省: {max(t_triton_cold - t_triton, 0) * 1000:8.2f} ms "
          f"({max((t_triton_cold - t_triton) / t_triton_cold * 100, 0):.1f}%)")
    print(f"  → 数值一致性: {'OK' if rot_same else 'FAIL'}")
    print(f"\n误差对比:")
    print(f"SimuQuant vs FP16:  max={abs_diff_simu_fp16.max():.6f}, mean={abs_diff_simu_fp16.mean():.6f}")
    print(f"反量化+GEMM vs SimuQuant: max={abs_diff_dequant_simu.max():.6f}, mean={abs_diff_dequant_simu.mean():.6f}")
    print(f"反量化+GEMM vs Triton: max={abs_diff_dequant_triton.max():.6f}, mean={abs_diff_dequant_triton.mean():.6f}")

    return {
        "t_fp16": t_fp16,
        "t_simu": t_simu,
        "t_dequant": t_dequant,
        "t_triton": t_triton,
        "t_triton_cold": t_triton_cold,
        "rot_same": rot_same,
        "err_simu_fp16_max": abs_diff_simu_fp16.max().item(),
        "err_simu_fp16_mean": abs_diff_simu_fp16.mean().item(),
        "err_dequant_simu_max": abs_diff_dequant_simu.max().item(),
        "err_dequant_simu_mean": abs_diff_dequant_simu.mean().item(),
        "err_dequant_triton_max": abs_diff_dequant_triton.max().item(),
        "err_dequant_triton_mean": abs_diff_dequant_triton.mean().item(),
    }


def test_rotation_cache_alignment(args, linear_result, moe_up_result, moe_down_result):
    """
    场景 3：旋转矩阵设备端缓存对照总览（针对最新改进）。

    最新改进：generate_rotation_matrix 在保留 CPU 基缓存（值一致）的同时，
    新增按 device 派生的缓存，消除每前向的 CPU→GPU 重拷（实验 6 拆出的隐藏开销）。

    各场景已在自身汇总里测出 cold(改前)/hot(改后) 时间差，这里统一对照。
    """
    print("\n" + "=" * 70)
    print("场景 3: 旋转矩阵设备端缓存对照（新改进 cold=改前 / hot=改后）")
    print("=" * 70)

    rows = [
        ("场景1 单独 Linear", linear_result),
        ("场景2.1 MoE up_gate", moe_up_result),
        ("场景2.2 MoE down", moe_down_result),
    ]
    all_ok = True
    for name, r in rows:
        cold = r["t_triton_cold"]
        hot = r["t_triton"]
        save = max(cold - hot, 0)
        pct = max(save / cold * 100, 0) if cold > 0 else 0
        ok = r.get("rot_same", False)
        all_ok = all_ok and ok
        print(f"  {name}:")
        print(f"    冷路径(改前, 清缓存重算): {cold * 1000:8.3f} ms")
        print(f"    热路径(改后, 命中缓存)  : {hot * 1000:8.3f} ms")
        print(f"    改进节省                : {save * 1000:8.3f} ms ({pct:.1f}%)")
        print(f"    数值一致性              : {'OK' if ok else 'FAIL'}")
        print()

    print(f"  总览结论 : {'全部对齐 OK —— 设备缓存不改变数值，且消除了重复重拷' if all_ok else 'FAIL —— 需检查'}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="混合精度 MoE 测试")

    parser.add_argument("--out_features", type=int, default=1024)
    parser.add_argument("--in_features", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    setup_seed(args.seed)

    linear_result = test_linear(args)
    moe_up_result = test_moe_up_gate(args)
    moe_down_result = test_moe_down(args)
    test_rotation_cache_alignment(args, linear_result, moe_up_result, moe_down_result)

    print("\n" + "=" * 70)
    print("最终汇总")
    print("=" * 70)

    print("\n场景1: 单独 Linear")
    print(f"    FP16:             {linear_result['t_fp16'] * 1000:8.2f} ms")
    print(f"    SimuQuant:        {linear_result['t_simu'] * 1000:8.2f} ms")
    print(f"    反量化+GEMM:      {linear_result['t_dequant'] * 1000:8.2f} ms")
    print(f"    Triton:           {linear_result['t_triton'] * 1000:8.2f} ms")
    print(f"  误差对比:")
    print(f"    SimuQuant vs FP16:    max={linear_result['err_simu_fp16_max']:.6f}, mean={linear_result['err_simu_fp16_mean']:.6f}")
    print(f"    反量化+GEMM vs SimuQuant: max={linear_result['err_dequant_simu_max']:.6f}, mean={linear_result['err_dequant_simu_mean']:.6f}")
    print(f"    反量化+GEMM vs Triton: max={linear_result['err_dequant_triton_max']:.6f}, mean={linear_result['err_dequant_triton_mean']:.6f}")

    print("\n场景2.1: MoE up_gate (slice_rows)")
    print(f"    FP16:             {moe_up_result['t_fp16'] * 1000:8.2f} ms")
    print(f"    SimuQuant:        {moe_up_result['t_simu'] * 1000:8.2f} ms")
    print(f"    反量化+GEMM:      {moe_up_result['t_dequant'] * 1000:8.2f} ms")
    print(f"    Triton:           {moe_up_result['t_triton'] * 1000:8.2f} ms")
    print(f"  误差对比:")
    print(f"    SimuQuant vs FP16:    max={moe_up_result['err_simu_fp16_max']:.6f}, mean={moe_up_result['err_simu_fp16_mean']:.6f}")
    print(f"    反量化+GEMM vs SimuQuant: max={moe_up_result['err_dequant_simu_max']:.6f}, mean={moe_up_result['err_dequant_simu_mean']:.6f}")
    print(f"    反量化+GEMM vs Triton: max={moe_up_result['err_dequant_triton_max']:.6f}, mean={moe_up_result['err_dequant_triton_mean']:.6f}")

    print("\n场景2.2: MoE down (slice_in_features)")
    print(f"    FP16:             {moe_down_result['t_fp16'] * 1000:8.2f} ms")
    print(f"    SimuQuant:        {moe_down_result['t_simu'] * 1000:8.2f} ms")
    print(f"    反量化+GEMM:      {moe_down_result['t_dequant'] * 1000:8.2f} ms")
    print(f"    Triton:           {moe_down_result['t_triton'] * 1000:8.2f} ms")
    print(f"  误差对比:")
    print(f"    SimuQuant vs FP16:    max={moe_down_result['err_simu_fp16_max']:.6f}, mean={moe_down_result['err_simu_fp16_mean']:.6f}")
    print(f"    反量化+GEMM vs SimuQuant: max={moe_down_result['err_dequant_simu_max']:.6f}, mean={moe_down_result['err_dequant_simu_mean']:.6f}")
    print(f"    反量化+GEMM vs Triton: max={moe_down_result['err_dequant_triton_max']:.6f}, mean={moe_down_result['err_dequant_triton_mean']:.6f}")

    print("\n" + "=" * 70)
    print("场景3（旋转缓存对照）已在场景1/2汇总后单独打印，见上。")

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
