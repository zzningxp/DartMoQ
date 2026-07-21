#!/usr/bin/env python3
"""
验证外推法
- 可以只用缓存数据验证
- 也可以生成缓存（需要加载模型）
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from typing import List, Optional, Tuple
from pathlib import Path
import gc

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_dartmoq import load_model  # type: ignore
from data_utils import get_loaders  # type: ignore
from turboquant_utils.dartmoq_backend import collect_expert_activation_inputs  # type: ignore

INTERMEDIATE_RESULT_DIR = "intermediate_result"


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _module_device(module: torch.nn.Module) -> torch.device:
    for param in module.parameters():
        return param.device
    for buf in module.buffers():
        return buf.device
    return torch.device("cpu")


def _move_to_device(obj, device: torch.device):
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_move_to_device(x, device) for x in obj)
    if isinstance(obj, list):
        return [_move_to_device(x, device) for x in obj]
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


def _forward_decoder_layer(
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    position_embeddings,
) -> torch.Tensor:
    kwargs = {"hidden_states": hidden_states}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    if position_embeddings is not None:
        kwargs["position_embeddings"] = position_embeddings
    outputs = layer(**kwargs)
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def _capture_first_layer_inputs(model, dataloader, nsamples: int, device: torch.device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    dtype = next(iter(model.parameters())).dtype
    bsz = 1
    inps = torch.zeros(
        (nsamples // bsz, bsz, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device="cpu",
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None, "position_embeddings": None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            cache["position_embeddings"] = kwargs.get("position_embeddings")
            raise ValueError

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

    layers[0] = Catcher(layers[0])
    try:
        with torch.no_grad():
            for batch in dataloader:
                if cache["i"] >= inps.shape[0]:
                    break
                try:
                    model(batch[0].to(device))
                except ValueError:
                    pass
    finally:
        layers[0] = layers[0].module
        model.config.use_cache = use_cache

    return (
        inps.squeeze(1),
        cache["attention_mask"],
        cache["position_ids"],
        cache["position_embeddings"],
    )


@torch.no_grad()
def _collect_layer_mlp_inputs(
    model,
    layer_idx: int,
    dataloader,
    nsamples: int,
    device: torch.device,
) -> torch.Tensor:
    inps, attention_mask, position_ids, position_embeddings = _capture_first_layer_inputs(
        model, dataloader, nsamples, device
    )

    first_layer_device = _module_device(model.model.layers[0])
    current = inps.to(first_layer_device)
    for idx in range(layer_idx):
        layer = model.model.layers[idx]
        layer_device = _module_device(layer)
        current = current.to(layer_device)
        layer_attention_mask = _move_to_device(attention_mask, layer_device)
        layer_position_ids = _move_to_device(position_ids, layer_device)
        layer_position_embeddings = _move_to_device(position_embeddings, layer_device)
        outs = torch.zeros_like(current, device=layer_device)
        for sample_idx in range(current.shape[0]):
            outs[sample_idx:sample_idx + 1] = _forward_decoder_layer(
                layer,
                current[sample_idx:sample_idx + 1],
                layer_attention_mask,
                layer_position_ids,
                layer_position_embeddings,
            )
        current = outs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    layer = model.model.layers[layer_idx]
    layer_device = _module_device(layer)
    current = current.to(layer_device)
    layer_attention_mask = _move_to_device(attention_mask, layer_device)
    layer_position_ids = _move_to_device(position_ids, layer_device)
    layer_position_embeddings = _move_to_device(position_embeddings, layer_device)
    batchsize = current.shape[0]
    residual = current
    hidden_states_inorm = layer.input_layernorm(current)
    attn_out = torch.zeros_like(hidden_states_inorm, device=layer_device)
    for sample_idx in range(batchsize):
        if layer_position_embeddings is not None:
            attn_out[sample_idx:sample_idx + 1] = layer.self_attn(
                hidden_states=hidden_states_inorm[sample_idx:sample_idx + 1],
                attention_mask=layer_attention_mask,
                position_ids=layer_position_ids,
                position_embeddings=layer_position_embeddings,
            )[0]
        else:
            attn_out[sample_idx:sample_idx + 1] = layer.self_attn(
                hidden_states=hidden_states_inorm[sample_idx:sample_idx + 1],
                attention_mask=layer_attention_mask,
                position_ids=layer_position_ids,
            )[0]
    hidden_states = residual + attn_out
    mlp_inputs = layer.post_attention_layernorm(hidden_states)
    return mlp_inputs.to(device)


def _concat_chunks(chunks: List[torch.Tensor], device: torch.device, max_rows: Optional[int]) -> torch.Tensor:
    if not chunks:
        return torch.empty(0, 0, dtype=torch.bfloat16, device=device)
    # 保留原始 dtype，不强制转换为 float32
    dtype = chunks[0].dtype if chunks else torch.bfloat16
    cat = torch.cat([chunk.to(dtype=dtype) for chunk in chunks], dim=0)
    if max_rows is not None and cat.shape[0] > max_rows:
        cat = cat[:max_rows]
    return cat.to(device=device, dtype=dtype)


def _n_experts(model) -> int:
    cfg = model.config
    if hasattr(cfg, "num_experts"):
        return int(cfg.num_experts)
    if hasattr(cfg, "n_routed_experts"):
        return int(cfg.n_routed_experts)
    raise ValueError("model config does not expose num_experts / n_routed_experts")


def _expert_forward_with_weights(
    expert: torch.nn.Module,
    tokens: torch.Tensor,
    up_w: torch.Tensor,
    gate_w: torch.Tensor,
    down_w: torch.Tensor,
) -> torch.Tensor:
    # 确保所有张量使用相同的 dtype
    dtype = up_w.dtype
    tokens = tokens.to(dtype=dtype)
    up = F.linear(tokens, up_w)
    gate = expert.act_fn(F.linear(tokens, gate_w))
    hidden = gate * up
    return F.linear(hidden, down_w)


@torch.no_grad()
def compute_true_0bit_loss_for_expert(
    expert,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """
    Compute true 0-bit loss for each neuron: directly compute mean(||output_orig - output_pruned||²)
    where output_pruned has neuron i set to 0 (others unchanged).

    Args:
        expert: The expert MLP module
        tokens: Input tokens to the expert (shape: [n_tokens, hidden_size])

    Returns:
        true_loss: Tensor of shape [n_neurons] with true 0-bit loss for each neuron
    """
    device = tokens.device

    # Get expert weights
    up_w = expert.up_proj.weight.clone()
    gate_w = expert.gate_proj.weight.clone()
    down_w = expert.down_proj.weight.clone()

    n_neurons = up_w.shape[0]

    # Compute original output
    output_orig = _expert_forward_with_weights(expert, tokens, up_w, gate_w, down_w)

    # Compute loss for each neuron
    true_loss = torch.zeros(n_neurons, device=device, dtype=torch.float32)

    for neuron_idx in range(n_neurons):
        # Create modified weights with neuron_idx set to 0
        up_w_pruned = up_w.clone()
        gate_w_pruned = gate_w.clone()
        down_w_pruned = down_w.clone()

        up_w_pruned[neuron_idx, :] = 0.0
        gate_w_pruned[neuron_idx, :] = 0.0
        down_w_pruned[:, neuron_idx] = 0.0

        # Compute pruned output
        output_pruned = _expert_forward_with_weights(expert, tokens, up_w_pruned, gate_w_pruned, down_w_pruned)

        # Compute MSE loss
        loss = (output_orig - output_pruned).pow(2).mean()
        true_loss[neuron_idx] = loss

    return true_loss


@torch.no_grad()
def compute_true_0bit_loss_for_layer(
    model,
    layer_idx: int,
    mlp_inputs: torch.Tensor,
    device: torch.device,
    num_experts_limit: Optional[int] = None,
) -> List[torch.Tensor]:
    """
    Compute true 0-bit loss for all experts in a layer.

    Args:
        model: The model
        layer_idx: Layer index
        mlp_inputs: Inputs to the MLP layer (shape: [nsamples, seqlen, hidden_size])
        device: Device to use
        num_experts_limit: Limit the number of experts to process (None for all)

    Returns:
        true_losses: List of true loss tensors, one per expert
    """
    layer = model.model.layers[layer_idx]
    ori_expert_num = _n_experts(model)

    if num_experts_limit is not None:
        ori_expert_num = min(ori_expert_num, num_experts_limit)

    # Collect expert activation inputs
    captured = collect_expert_activation_inputs(layer, mlp_inputs, _n_experts(model), if_dense=False)

    true_losses = []
    for expert_idx in range(ori_expert_num):
        print(f"    Computing true 0-bit loss for expert {expert_idx}...", flush=True)
        expert_capture = captured[expert_idx]
        tokens = _concat_chunks(expert_capture["up_proj"], device, max_rows=4096)

        if tokens.numel() == 0:
            print(f"    Warning: No tokens for expert {expert_idx}, using zero loss", flush=True)
            expert = layer.mlp.experts[expert_idx]
            n_neurons = expert.up_proj.weight.shape[0]
            true_losses.append(torch.zeros(n_neurons, device=device))
            continue

        expert = layer.mlp.experts[expert_idx]
        true_loss = compute_true_0bit_loss_for_expert(expert, tokens)
        true_losses.append(true_loss.cpu())

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return true_losses

def main():
    parser = argparse.ArgumentParser(description="验证外推法")
    parser.add_argument("model", nargs="?", help="模型路径（生成 cache 时需要）")
    parser.add_argument("--layers", type=int, nargs="+", default=[1], help="要处理的层")
    parser.add_argument("--model_id", type=str, default="deepseek-v1-moe-16b", help="模型 ID")
    parser.add_argument("--quantmode", type=str, choices=["gptq", "turboquant"], default="turboquant", help="量化模式")
    parser.add_argument("--rank_mode", type=str, default="turboquant_innerproduct", help="rank mode")
    parser.add_argument("--recompute", action="store_true", help="重新计算，不用缓存")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--nsamples", type=int, default=32, help="采样数")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--out_dir", type=str, default="plot/zero_bit_validation", help="输出目录")
    parser.add_argument("--num_experts", type=int, default=None, help="验证的专家数（None 表示全部）")
    args = parser.parse_args()

    print("=" * 60)
    print("验证外推法")
    print("=" * 60)
    print(f"model_id: {args.model_id}")
    print(f"quantmode: {args.quantmode}")
    print(f"rank_mode: {args.rank_mode}")
    print(f"layers: {args.layers}")
    print()

    # 加载模型（如果需要计算真实 0-bit 损失）
    model = None
    dataloader = None
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.model is None:
        print("错误: 计算真实 0-bit 损失需要提供模型路径")
        return

    print("加载模型...")
    model, tokenizer = load_model(args.model)
    model.eval()

    print("加载数据...")
    dataloader, _ = get_loaders(
        "wikitext2",
        nsamples=args.nsamples,
        seed=args.seed,
        tokenizer=tokenizer,
        seqlen=model.seqlen,
    )

    # 验证外推法
    print()
    print("验证外推法...")

    cache_dir = os.path.join(
        INTERMEDIATE_RESULT_DIR, f"quant_outlier_{args.quantmode}", args.rank_mode, args.model_id
    )

    if not os.path.exists(cache_dir):
        print(f"错误: 缓存目录不存在: {cache_dir}")
        print(f"请先用 --generate_cache 生成缓存，或者检查路径")
        return

    # 处理每一层
    all_results = []
    for layer_idx in args.layers:
        print()
        print(f"验证层 {layer_idx}...")

        # 加载 b0-b4
        cached_data = {}
        found = True
        path = os.path.join(cache_dir, f"{args.model_id}_L{layer_idx}_b0.pt")
        if os.path.exists(path):
            cached_data[0] = torch.load(path, map_location="cpu")
            print(f"  加载 b{0}: {len(cached_data[0])} experts")
        else:
            assert False, f"b{0} 未找到"

        true_0bit_losses = None

        true_0bit_cache_path = os.path.join(cache_dir, f"{args.model_id}_L{layer_idx}_true_0bit.pt")
        if os.path.exists(true_0bit_cache_path) and not args.recompute:
            print(f"  加载真实 0-bit 损失缓存: {true_0bit_cache_path}")
            true_0bit_losses = torch.load(true_0bit_cache_path, map_location="cpu")
        else:
            print(f"  计算真实 0-bit 损失...")
            mlp_inputs = _collect_layer_mlp_inputs(model, layer_idx, dataloader, args.nsamples, device)
            true_0bit_losses = compute_true_0bit_loss_for_layer(
                model, layer_idx, mlp_inputs, device, num_experts_limit=args.num_experts
            )
            print(f"  保存真实 0-bit 损失缓存: {true_0bit_cache_path}")
            torch.save(true_0bit_losses, true_0bit_cache_path)

        # 处理全部 expert
        all_actual = []
        all_extrapolated = []
        expert_results = []

        num_experts = len(cached_data[0]) if args.num_experts is None else min(args.num_experts, len(cached_data[0]))
        for expert_idx in range(num_experts):
            print(f"  处理 Expert {expert_idx}...")

            # 外推
            extrapolated = cached_data[0][expert_idx].numpy()

            # 获取真实 0-bit 损失
            if true_0bit_losses is not None and expert_idx < len(true_0bit_losses):
                actual = true_0bit_losses[expert_idx].numpy()
            else:
                assert False, f"    警告: 没有真实 0-bit 损失"

            # 过滤
            valid_mask = (
                (np.array(extrapolated) > 0) & (actual > 0) &
                np.isfinite(extrapolated) & np.isfinite(actual)
            )

            extrap_valid = np.array(extrapolated)[valid_mask]
            actual_valid = actual[valid_mask]

            if len(extrap_valid) > 0:
                # 统计
                corr, _ = spearmanr(actual_valid, extrap_valid)
                mae_linear = np.mean(np.abs(extrap_valid - actual_valid))

                print(f"    Spearman: {corr:.4f}, MAE: {mae_linear:.4e}")

                expert_results.append({
                    "layer_idx": layer_idx,
                    "expert_idx": expert_idx,
                    "spearman": corr,
                    "mae": mae_linear,
                    "n_neurons": len(extrap_valid),
                })

                all_actual.extend(actual_valid)
                all_extrapolated.extend(extrap_valid)

        # 汇总这层的图
        if len(all_actual) > 0:
            all_actual_np = np.array(all_actual)
            all_extrapolated_np = np.array(all_extrapolated)

            overall_corr, _ = spearmanr(all_actual_np, all_extrapolated_np)
            overall_mae = np.mean(np.abs(all_extrapolated_np - all_actual_np))

            # 计算专家级别的平均统计量
            expert_spearmans = [r["spearman"] for r in expert_results]
            expert_maes = [r["mae"] for r in expert_results]
            mean_spearman = np.mean(expert_spearmans)
            std_spearman = np.std(expert_spearmans)
            mean_mae = np.mean(expert_maes)
            std_mae = np.std(expert_maes)

            print()
            print(f"  层 {layer_idx} 结果:")
            print(f"    总体 Spearman: {overall_corr:.4f}")
            print(f"    总体 MAE: {overall_mae:.4e}")
            print(f"    专家平均 Spearman: {mean_spearman:.4f} ± {std_spearman:.4f}")
            print(f"    专家平均 MAE: {mean_mae:.4e} ± {std_mae:.4e}")
            print(f"    总神经元数: {len(all_actual_np)}")
            print(f"    专家数: {len(expert_results)}")

            all_results.append({
                "layer_idx": layer_idx,
                "overall_spearman": overall_corr,
                "overall_mae": overall_mae,
                "mean_spearman": mean_spearman,
                "std_spearman": std_spearman,
                "mean_mae": mean_mae,
                "std_mae": std_mae,
                "all_actual": all_actual_np,
                "all_extrapolated": all_extrapolated_np,
                "expert_results": expert_results,
            })

            # 画图
            plt.figure(figsize=(10, 8))
            plt.scatter(
                np.log10(all_actual_np + 1e-12),
                np.log10(all_extrapolated_np + 1e-12),
                alpha=0.3, s=5
            )

            min_val = min(
                np.log10(all_actual_np.min() + 1e-12),
                np.log10(all_extrapolated_np.min() + 1e-12)
            )
            max_val = max(
                np.log10(all_actual_np.max() + 1e-12),
                np.log10(all_extrapolated_np.max() + 1e-12)
            )
            plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5)

            stats_text = f"Spearman: {overall_corr:.4f}\nMAE: {overall_mae:.4e}"
            plt.text(
                0.05, 0.95, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9),
                fontsize=11
            )

            plt.xlabel("True 0-bit loss (log10)")
            plt.ylabel("Extrapolated 0-bit loss (log10)")
            plt.title(f"{args.model_id} Layer {layer_idx}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            os.makedirs(args.out_dir, exist_ok=True)
            out_path = os.path.join(args.out_dir, f"{args.model_id}_L{layer_idx}_extrapolation_check.png")
            plt.savefig(out_path, dpi=150)
            print(f"  图已保存至: {out_path}")
            plt.close()

    # 总体总结
    print()
    print("=" * 60)
    print("总体总结")
    print("=" * 60)
    for result in all_results:
        print(f"层 {result['layer_idx']}:")
        print(f"  总体 Spearman: {result['overall_spearman']:.4f}")
        print(f"  总体 MAE: {result['overall_mae']:.4e}")
        print(f"  专家平均 Spearman: {result['mean_spearman']:.4f} ± {result['std_spearman']:.4f}")
        print(f"  专家平均 MAE: {result['mean_mae']:.4e} ± {result['std_mae']:.4e}")
    print("=" * 60)


if __name__ == "__main__":
    main()
