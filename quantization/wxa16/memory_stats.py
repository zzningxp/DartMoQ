#!/usr/bin/env python3
"""
WxA16 存储空间统计工具
用于在量化前后统计并输出存储空间占比变化
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from collections import defaultdict


def count_bytes(obj) -> int:
    """计算一个对象的字节数，支持 tensor 或 packed dict"""
    if obj is None:
        return 0

    # 单个 tensor
    if hasattr(obj, 'numel') and hasattr(obj, 'element_size'):
        return obj.numel() * obj.element_size()

    # dict of objects
    if isinstance(obj, dict):
        total = 0
        for k, v in obj.items():
            total += count_bytes(v)
        return total

    # list/tuple of objects
    if isinstance(obj, (list, tuple)):
        total = 0
        for item in obj:
            total += count_bytes(item)
        return total

    return 0


def get_linear_memory_stats(linear: nn.Linear, prefix: str = "") -> Dict:
    """
    获取一个 nn.Linear 的存储空间统计

    Args:
        linear: nn.Linear 层
        prefix: 名称前缀

    Returns:
        dict with memory stats
    """
    weight_bytes = count_bytes(linear.weight)
    bias_bytes = count_bytes(linear.bias)
    total_bytes = weight_bytes + bias_bytes

    return {
        'name': prefix,
        'weight_bytes': weight_bytes,
        'bias_bytes': bias_bytes,
        'total_bytes': total_bytes,
        'type': 'linear_fp16',
        'dtype': str(linear.weight.dtype) if hasattr(linear, 'weight') else None,
    }


def get_wxa16_memory_stats(wxa16_linear, prefix: str = "") -> Dict:
    """
    获取一个 WxA16Linear 的存储空间统计

    Args:
        wxa16_linear: WxA16Linear 层
        prefix: 名称前缀

    Returns:
        dict with memory stats
    """
    mem = wxa16_linear.get_memory_usage()

    return {
        'name': prefix,
        'packed_bytes': mem['packed_bytes'],
        'metadata_bytes': mem['metadata_bytes'],
        'total_bytes': mem['total_bytes'],
        'orig_fp16_bytes': mem['orig_fp16_bytes'],
        'compression_ratio': mem['compression_ratio'],
        'bit_width': mem['bit_width'],
        'type': f'wxa16_{mem["bit_width"]}bit',
    }


def collect_all_linears(module, prefix="", result=None):
    """递归收集模块中的所有 nn.Linear 和 WxA16Linear"""
    if result is None:
        result = []

    # 检查当前模块是否是线性层
    if hasattr(module, 'get_memory_usage'):
        # WxA16Linear
        result.append((prefix, module, 'wxa16'))
    elif isinstance(module, nn.Linear):
        # 普通 nn.Linear
        result.append((prefix, module, 'linear'))
    else:
        # 递归检查子模块（但避免循环）
        for name, child in module.named_children():
            new_prefix = f"{prefix}.{name}" if prefix else name
            collect_all_linears(child, new_prefix, result)

    return result


def calculate_attn_total(attn_layer, is_quantized=False):
    """计算注意力层的总大小，通用方法"""
    total = 0
    linears = collect_all_linears(attn_layer, "attn")

    for name, module, _ in linears:
        if is_quantized and hasattr(module, 'get_memory_usage'):
            mem = module.get_memory_usage()
            total += mem['total_bytes']
            print(f"[DEBUG] Added {name}: {mem['total_bytes']} bytes (type: WxA16{mem['bit_width']})", flush=True)
        elif isinstance(module, nn.Linear):
            mem = get_linear_memory_stats(module)
            total += mem['total_bytes']
            print(f"[DEBUG] Added {name}: {mem['total_bytes']} bytes (type: {type(module).__name__})", flush=True)

    return total


def get_layer_memory_stats(layer, layer_idx: int) -> Dict:
    """
    统计一层的所有 Linear 模块的存储空间

    Args:
        layer: transformer 层
        layer_idx: 层索引

    Returns:
        dict with categorized stats
    """
    stats = {
        'layer_idx': layer_idx,
        'attention': defaultdict(int),
        'shared_expert': defaultdict(int),
        'router_expert': defaultdict(lambda: defaultdict(int)),  # bit -> size
        'total_orig': 0,
        'total_new': 0,
    }

    # Attention 层统计
    attn_layer = None
    attn_type = None
    if hasattr(layer, 'self_attn'):
        attn_layer = layer.self_attn
        attn_type = 'full'
    elif hasattr(layer, 'linear_attn'):
        attn_layer = layer.linear_attn
        attn_type = 'linear'

    if attn_layer is not None:
        if attn_type == 'full':
            # 标准注意力层
            attn_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
            for name in attn_names:
                if hasattr(attn_layer, name):
                    module = getattr(attn_layer, name)
                    prefix = f'attn.{name}'
                    if hasattr(module, 'get_memory_usage'):
                        # WxA16Linear
                        mem = get_wxa16_memory_stats(module, prefix)
                        stats['attention'][f'{name}_total'] += mem['total_bytes']
                        stats['attention'][f'{name}_orig'] += mem['orig_fp16_bytes']
                        stats['attention'][f'{name}_bit'] = mem['bit_width']
                    elif isinstance(module, nn.Linear):
                        # nn.Linear (fp16)
                        mem = get_linear_memory_stats(module, prefix)
                        stats['attention'][f'{name}_total'] += mem['total_bytes']
                        stats['attention'][f'{name}_orig'] += mem['total_bytes']
                        stats['attention'][f'{name}_bit'] = 16
        elif attn_type == 'linear':
            # 线性注意力层
            attn_names = ['in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj']
            for name in attn_names:
                if hasattr(attn_layer, name):
                    module = getattr(attn_layer, name)
                    prefix = f'attn.{name}'
                    if hasattr(module, 'get_memory_usage'):
                        # WxA16Linear
                        mem = get_wxa16_memory_stats(module, prefix)
                        stats['attention'][f'{name}_total'] += mem['total_bytes']
                        stats['attention'][f'{name}_orig'] += mem['orig_fp16_bytes']
                        stats['attention'][f'{name}_bit'] = mem['bit_width']
                    elif isinstance(module, nn.Linear):
                        # nn.Linear (fp16)
                        mem = get_linear_memory_stats(module, prefix)
                        stats['attention'][f'{name}_total'] += mem['total_bytes']
                        stats['attention'][f'{name}_orig'] += mem['total_bytes']
                        stats['attention'][f'{name}_bit'] = 16

    # Shared expert 统计
    if hasattr(layer.mlp, 'shared_expert') and layer.mlp.shared_expert is not None:
        shared = layer.mlp.shared_expert
        proj_names = ['gate_proj', 'up_proj', 'down_proj']
        for name in proj_names:
            if hasattr(shared, name):
                module = getattr(shared, name)
                prefix = f'shared.{name}'
                if hasattr(module, 'get_memory_usage'):
                    mem = get_wxa16_memory_stats(module, prefix)
                    stats['shared_expert'][f'{name}_total'] += mem['total_bytes']
                    stats['shared_expert'][f'{name}_orig'] += mem['orig_fp16_bytes']
                    stats['shared_expert'][f'{name}_bit'] = mem['bit_width']
                elif isinstance(module, nn.Linear):
                    mem = get_linear_memory_stats(module, prefix)
                    stats['shared_expert'][f'{name}_total'] += mem['total_bytes']
                    stats['shared_expert'][f'{name}_orig'] += mem['total_bytes']
                    stats['shared_expert'][f'{name}_bit'] = 16

    # Router expert 统计 (BitPartitionedGroupMoE 或 WxA16BitPartitionedGroupMoE)
    # 这里在量化前/后的阶段分别处理，因为结构不同

    return stats


def format_bytes(bytes_val: int) -> str:
    """将字节数格式化为可读字符串"""
    if bytes_val >= 1024**3:
        return f"{bytes_val / (1024**3):.2f} GB"
    elif bytes_val >= 1024**2:
        return f"{bytes_val / (1024**2):.2f} MB"
    elif bytes_val >= 1024:
        return f"{bytes_val / 1024:.2f} KB"
    else:
        return f"{bytes_val} B"


def print_memory_stats_layer_before(layer, layer_idx: int):
    """
    在量化前打印一层的存储空间统计

    Output format:
    [Layer X] Attention:  XXX MB
    [Layer X] Shared:     gate: XXX MB, up: XXX MB, down: XXX MB, total: XXX MB
    [Layer X] Router:     1bit: XXX MB, 2bit: XXX MB, 4bit: XXX MB, 8bit: XXX MB (但量化前都是 16bit)
    """
    print(f"\n{'='*80}", flush=True)
    print(f"[Layer {layer_idx}] Quantization PREPARE - Memory Stats (fp16 baseline)", flush=True)
    print(f"{'='*80}", flush=True)

    # Attention 统计
    attn_total = 0
    attn_layer = None
    attn_type = None

    if hasattr(layer, 'self_attn'):
        attn_layer = layer.self_attn
        attn_type = 'full'
    elif hasattr(layer, 'linear_attn'):
        attn_layer = layer.linear_attn
        attn_type = 'linear'
    else:
        # 尝试其他可能的属性名
        print(f"[DEBUG] No self_attn/linear_attn, checking all attrs...", flush=True)
        for attr in dir(layer):
            if 'attn' in attr.lower() or 'attention' in attr.lower():
                print(f"[DEBUG] Found possible attention attr: {attr}", flush=True)

    if attn_layer is not None:
        # print(f"[DEBUG] Attn layer type: {attn_type}, attributes: {[attr for attr in dir(attn_layer) if not attr.startswith('_')]}", flush=True)
        # 使用通用方法统计
        attn_total = calculate_attn_total(attn_layer, is_quantized=False)

    print(f"[Layer {layer_idx}] {'Attention':<12} {format_bytes(attn_total):>12} (16-bit fp16, {attn_type if attn_type else 'unknown'})", flush=True)

    # Shared expert 统计
    shared_gate = 0
    shared_up = 0
    shared_down = 0
    if hasattr(layer.mlp, 'shared_expert') and layer.mlp.shared_expert is not None:
        shared = layer.mlp.shared_expert
        if hasattr(shared, 'gate_proj') and isinstance(shared.gate_proj, nn.Linear):
            shared_gate = get_linear_memory_stats(shared.gate_proj)['total_bytes']
        if hasattr(shared, 'up_proj') and isinstance(shared.up_proj, nn.Linear):
            shared_up = get_linear_memory_stats(shared.up_proj)['total_bytes']
        if hasattr(shared, 'down_proj') and isinstance(shared.down_proj, nn.Linear):
            shared_down = get_linear_memory_stats(shared.down_proj)['total_bytes']

    shared_total = shared_gate + shared_up + shared_down
    print(f"[Layer {layer_idx}] {'Shared':<12} gate:{format_bytes(shared_gate):>10}, up:{format_bytes(shared_up):>10}, down:{format_bytes(shared_down):>10}, total:{format_bytes(shared_total):>10} (16-bit fp16)", flush=True)

    # Router expert 统计 - 从原始结构计算 (量化前)
    router_total = 0
    router_gate = 0
    router_up = 0
    router_down = 0
    num_experts = 0

    if hasattr(layer.mlp, 'experts'):
        experts = layer.mlp.experts

    # 检查原始 MoE 结构
    if hasattr(layer.mlp, 'experts'):
        experts = layer.mlp.experts

        # 检查是否是 grouped_gemm 格式 (gate_up_proj 和 down_proj 是大的 tensor)
        if hasattr(experts, 'gate_up_proj') and hasattr(experts, 'down_proj'):
            # Grouped GEMM 格式
            gate_up_proj = experts.gate_up_proj
            num_experts = gate_up_proj.shape[0]
            intermediate_size = gate_up_proj.shape[1] // 2
            hidden_size = gate_up_proj.shape[2]

            # 直接计算字节数: num_experts * (gate + up + down)
            # gate 和 up 每个都是 intermediate_size * hidden_size
            # down 是 hidden_size * intermediate_size
            # fp16 = 2 bytes
            per_expert_gate_bytes = intermediate_size * hidden_size * 2
            per_expert_up_bytes = intermediate_size * hidden_size * 2
            per_expert_down_bytes = hidden_size * intermediate_size * 2

            router_gate = num_experts * per_expert_gate_bytes
            router_up = num_experts * per_expert_up_bytes
            router_down = num_experts * per_expert_down_bytes
            print(f"[DEBUG] Grouped GEMM format: {num_experts} experts, gate bytes: {router_gate}", flush=True)

        elif hasattr(experts, '__len__'):
            # TraditionalMoEWrapper / DartMoQHybridWrapper
            num_experts = len(experts)
            print(f"[DEBUG] Traditional format: {num_experts} experts", flush=True)
            if num_experts > 0:
                # 看第一个 expert 的结构
                first_exp = experts[0]
                if hasattr(first_exp, 'sub_experts'):
                    # Hybrid wrapper 有多个 sub_experts，按 bit 分开
                    for expert in experts:
                        if hasattr(expert, 'sub_experts'):
                            for sub_expert in expert.sub_experts:
                                for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                                    if hasattr(sub_expert, proj_name):
                                        proj = getattr(sub_expert, proj_name)
                                        if isinstance(proj, nn.Linear):
                                            bytes_val = get_linear_memory_stats(proj)['total_bytes']
                                            if proj_name == 'gate_proj':
                                                router_gate += bytes_val
                                            elif proj_name == 'up_proj':
                                                router_up += bytes_val
                                            elif proj_name == 'down_proj':
                                                router_down += bytes_val
                else:
                    # 普通专家
                    for expert in experts:
                        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                            if hasattr(expert, proj_name):
                                proj = getattr(expert, proj_name)
                                if isinstance(proj, nn.Linear):
                                    bytes_val = get_linear_memory_stats(proj)['total_bytes']
                                    if proj_name == 'gate_proj':
                                        router_gate += bytes_val
                                    elif proj_name == 'up_proj':
                                        router_up += bytes_val
                                    elif proj_name == 'down_proj':
                                        router_down += bytes_val

    router_total = router_gate + router_up + router_down
    router_label = f"({num_experts} experts)" if num_experts > 0 else "baseline"
    print(f"[Layer {layer_idx}] {'Router':<12} gate:{format_bytes(router_gate):>10}, up:{format_bytes(router_up):>10}, down:{format_bytes(router_down):>10}, total:{format_bytes(router_total):>10} (16-bit fp16 {router_label})", flush=True)

    # 总计
    layer_total = attn_total + shared_total + router_total
    print(f"[Layer {layer_idx}] {'TOTAL':<12} {format_bytes(layer_total):>12}", flush=True)
    print(f"{'='*80}\n", flush=True)

    return {
        'layer_idx': layer_idx,
        'attn': attn_total,
        'shared_gate': shared_gate,
        'shared_up': shared_up,
        'shared_down': shared_down,
        'shared_total': shared_total,
        'router_gate': router_gate,
        'router_up': router_up,
        'router_down': router_down,
        'router_total': router_total,
        'num_experts': num_experts,
        'total': layer_total,
    }


def print_memory_stats_layer_after(layer, layer_idx: int, before_stats: Dict, qscheme: Dict = None):
    """
    在 WxA16 量化后打印一层的存储空间统计

    Output format:
    [Layer X] Attention:  XXX MB (8-bit) -> X.Xx compression
    [Layer X] Shared:     gate: XXX MB (X-bit), up: XXX MB (X-bit), down: XXX MB (X-bit), total: XXX MB -> X.Xx compression
    [Layer X] Router:     1bit: XXX MB, 2bit: XXX MB, 4bit: XXX MB, 8bit: XXX MB, total: XXX MB -> X.Xx compression
    """
    print(f"\n{'='*80}", flush=True)
    print(f"[Layer {layer_idx}] WxA16 Quantization DONE - Memory Stats", flush=True)
    print(f"{'='*80}", flush=True)

    # Attention 统计
    attn_total = 0
    attn_orig = before_stats['attn']
    attn_bit = 8  # default from qscheme
    if qscheme and 'attn' in qscheme:
        attn_bit = qscheme['attn'][0] if isinstance(qscheme['attn'], (list, tuple)) else qscheme['attn']

    attn_layer = None
    attn_type = None
    if hasattr(layer, 'self_attn'):
        attn_layer = layer.self_attn
        attn_type = 'full'
    elif hasattr(layer, 'linear_attn'):
        attn_layer = layer.linear_attn
        attn_type = 'linear'

    if attn_layer is not None:
        print(f"[DEBUG] Quantized - Attn layer type: {attn_type}", flush=True)
        # 使用通用方法统计量化后的大小
        attn_total = calculate_attn_total(attn_layer, is_quantized=True)

    attn_compression = attn_orig / attn_total if attn_total > 0 else 1.0
    print(f"[Layer {layer_idx}] {'Attention':<12} {format_bytes(attn_total):>12} ({attn_bit}-bit, {attn_type if attn_type else 'unknown'}) -> {attn_compression:.2f}x compression", flush=True)

    # Shared expert 统计
    shared_gate = 0
    shared_up = 0
    shared_down = 0
    shared_gate_bit = 16
    shared_up_bit = 16
    shared_down_bit = 16

    if hasattr(layer.mlp, 'shared_expert') and layer.mlp.shared_expert is not None:
        shared = layer.mlp.shared_expert
        if hasattr(shared, 'gate_proj'):
            module = shared.gate_proj
            if hasattr(module, 'get_memory_usage'):
                mem = module.get_memory_usage()
                shared_gate = mem['total_bytes']
                shared_gate_bit = mem['bit_width']
            elif isinstance(module, nn.Linear):
                shared_gate = get_linear_memory_stats(module)['total_bytes']

        if hasattr(shared, 'up_proj'):
            module = shared.up_proj
            if hasattr(module, 'get_memory_usage'):
                mem = module.get_memory_usage()
                shared_up = mem['total_bytes']
                shared_up_bit = mem['bit_width']
            elif isinstance(module, nn.Linear):
                shared_up = get_linear_memory_stats(module)['total_bytes']

        if hasattr(shared, 'down_proj'):
            module = shared.down_proj
            if hasattr(module, 'get_memory_usage'):
                mem = module.get_memory_usage()
                shared_down = mem['total_bytes']
                shared_down_bit = mem['bit_width']
            elif isinstance(module, nn.Linear):
                shared_down = get_linear_memory_stats(module)['total_bytes']

    shared_total = shared_gate + shared_up + shared_down
    shared_total_orig = before_stats['shared_total']
    shared_compression = shared_total_orig / shared_total if shared_total > 0 else 1.0

    print(f"[Layer {layer_idx}] {'Shared':<12} gate:{format_bytes(shared_gate):>10}({shared_gate_bit}-bit), up:{format_bytes(shared_up):>10}({shared_up_bit}-bit), down:{format_bytes(shared_down):>10}({shared_down_bit}-bit), total:{format_bytes(shared_total):>10} -> {shared_compression:.2f}x compression", flush=True)

    # Router expert 统计 - 从 WxA16BitPartitionedGroupMoE 结构计算
    router_total = 0
    router_by_bit = defaultdict(int)

    if hasattr(layer.mlp, 'bit_weights'):
        # WxA16BitPartitionedGroupMoE 或 BitPartitionedGroupMoE
        bit_weights = layer.mlp.bit_weights

        # 检查是 nn.ModuleDict 还是其他结构
        if hasattr(bit_weights, 'keys'):
            for bit_str in bit_weights.keys():
                bit = int(bit_str)
                w = bit_weights[bit_str]

                # WxA16Weights 对象
                if hasattr(w, 'gate_up_packed') and w.gate_up_packed is not None:
                    # 从 packed data dict 统计
                    gate_up_bytes = count_bytes(w.gate_up_packed)
                    router_by_bit[bit] += gate_up_bytes

                if hasattr(w, 'down_packed') and w.down_packed is not None:
                    down_bytes = count_bytes(w.down_packed)
                    router_by_bit[bit] += down_bytes

        # 也可能是 BitPartitionedGroupMoE 的属性
        elif hasattr(bit_weights, 'gate_up') and hasattr(bit_weights, 'down'):
            # fp16 fake quant 版本
            pass

    # 汇总 router expert
    for bit in sorted(router_by_bit.keys()):
        router_total += router_by_bit[bit]

    router_total_orig = before_stats['router_total']
    router_compression = router_total_orig / router_total if router_total > 0 else 1.0

    router_bit_strs = []
    for bit in sorted(router_by_bit.keys()):
        router_bit_strs.append(f"{bit}bit:{format_bytes(router_by_bit[bit]):>10}")
    router_bit_str = ", ".join(router_bit_strs)

    print(f"[Layer {layer_idx}] {'Router':<12} {router_bit_str}, total:{format_bytes(router_total):>10} -> {router_compression:.2f}x compression", flush=True)

    # 总计
    layer_total = attn_total + shared_total + router_total
    layer_total_orig = before_stats['total']
    layer_compression = layer_total_orig / layer_total if layer_total > 0 else 1.0

    print(f"[Layer {layer_idx}] {'TOTAL':<12} {format_bytes(layer_total):>12} (orig {format_bytes(layer_total_orig):>12}) -> {layer_compression:.2f}x compression", flush=True)
    print(f"{'='*80}\n", flush=True)

    return {
        'layer_idx': layer_idx,
        'attn': attn_total,
        'attn_orig': attn_orig,
        'attn_compression': attn_compression,
        'shared_gate': shared_gate,
        'shared_up': shared_up,
        'shared_down': shared_down,
        'shared_total': shared_total,
        'shared_total_orig': shared_total_orig,
        'shared_compression': shared_compression,
        'router_by_bit': dict(router_by_bit),
        'router_total': router_total,
        'router_total_orig': router_total_orig,
        'router_compression': router_compression,
        'total': layer_total,
        'total_orig': layer_total_orig,
        'total_compression': layer_compression,
    }
