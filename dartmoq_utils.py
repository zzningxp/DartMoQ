from urllib.parse import scheme_chars
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np

from typing import Optional, Tuple, List
import re
import time
import gc

from gptq_utils import GPTQ, Quantizer, find_layers

QBATCH = 256
DEV = torch.device('cuda:0')

@torch.no_grad()
def analyze_experts_activation(layer, layer_idx, inps, K, modeltype, save_path=None):

    tick0 = time.time()
    batch_size, seq_len, emb_size = inps.shape
    total_samples = batch_size * seq_len

    activation_markers = torch.zeros(layer.mlp.gate.weight.shape[0]).to(inps.device)
    if modeltype == 'olmoe' or modeltype == 'qwen3_moe' or modeltype == 'qwen3':
        with torch.no_grad():
            hidden_states = inps.view(-1, emb_size)
            router_logits = layer.mlp.gate(hidden_states)
            router_logits = F.softmax(router_logits, dim=-1)
            _, top_indices = torch.topk(router_logits, K, dim=-1)
            # print(top_indices.shape)
            del router_logits, hidden_states
    else:
        with torch.no_grad():
            top_indices, top_values, _ = layer.mlp.gate(inps)
            # print(top_indices.shape)

    for i in range(total_samples):
        activation_markers[top_indices[i]] += 1.0
    
    activation_rates = activation_markers / total_samples
    activation_rates = activation_rates.detach().cpu()

    if save_path:
        plt.figure(figsize=(10, 10))
        
        plt.subplot(1, 1, 1)
        plt.plot(range(activation_rates.shape[0]), np.sort(activation_rates.numpy()),  'b-', alpha=0.6)
        plt.title(f'Activation Rates for Layer {layer_idx}')
        plt.xlabel('Neuron Index')
        plt.ylabel('Activation Rate')
        plt.grid(True, alpha=0.3)
        
        # Add statistics for rates
        mean_rate = activation_rates.mean()
        std_rate = activation_rates.std()
        stats_text = (f'Mean rate: {mean_rate:.3f}\n'
                     f'Std rate: {std_rate:.3f}\n'
                     f'Max rate: {activation_rates.max():.3f}\n'
                     f'Min rate: {activation_rates.min():.3f}')
        
        plt.text(0.95, 0.95, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plt.savefig(save_path)
        plt.close()

    tick1 = time.time()
    print(f"analyze_experts_activation time for layer {layer_idx} is: {tick1 - tick0:.2f}")
    return activation_rates

@torch.no_grad()
def analyze_neuron_activations(act_fn, inps, gate_proj_weights, up_proj_weights, save_path: Optional[str] = None, sparsity = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    h = act_fn(F.linear(F.normalize(inps, p=2, dim=-1), F.normalize(gate_proj_weights, p=2, dim=1)))
    scores = h * F.linear(F.normalize(inps, p=2, dim=-1), F.normalize(up_proj_weights, p=2, dim=1))

    K = max(1, math.ceil(scores.shape[-1] * sparsity))  # Top 10% neurons 
    # print("K (top neurons per sample):", scores.shape, K)

    scores = scores.detach().cpu()
    batch_size, seq_len, inter_size = scores.shape
    total_samples = batch_size * seq_len

    flat_states = scores.reshape(-1, inter_size)
    activation_markers = torch.zeros_like(flat_states)
    # activation_values = torch.zeros_like(flat_states)
    
    for i in range(total_samples):
        sample_values = flat_states[i]
        abs_values = sample_values.abs().float()

        # Get indices of top-k absolute values
        top_values, top_indices = torch.topk(abs_values, k=K)
        activation_markers[i, top_indices] = 1.0
        # for idx in top_indices:
        #     activation_values[i, idx] = abs_values[idx]

    # Sum up activations across all samples
    activation_counts = activation_markers.sum(dim=0)
    # activation_values = activation_values.sum(dim=0)
    activation_rates = activation_counts / total_samples

    if save_path:
        plt.figure(figsize=(10, 10))
        
        # Plot 1: Activation rates histogram
        plt.subplot(3, 1, 1)  
        plt.hist(activation_rates.detach().to(dtype=torch.float32).numpy(), bins=500, edgecolor='black')
        plt.title('Distribution of Neuron Activation Rates')
        plt.xlabel('Activation Rate')
        plt.ylabel('Number of Neurons')
        plt.grid(True, alpha=0.3)
        
        # Add statistics for rates
        mean_rate = activation_rates.mean()
        std_rate = activation_rates.std()
        stats_text = (f'Mean rate: {mean_rate:.3f}\n'
                     f'Std rate: {std_rate:.3f}\n'
                     f'Max rate: {activation_rates.max():.3f}\n'
                     f'Min rate: {activation_rates.min():.3f}')
        
        plt.text(0.95, 0.95, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Plot 2: Neuron indices vs Activation counts
        plt.subplot(3, 1, 2)  
        neuron_indices = np.arange(inter_size)
        stats_text = (f'Mean count: {activation_counts.mean():.3f}\n'
                     f'Std count: {activation_counts.std():.3f}\n'
                     f'Max count: {activation_counts.max():.3f}\n'
                     f'Min count: {activation_counts.min():.3f}')
        
        activation_counts = activation_counts.detach().to(dtype=torch.float32).numpy()
        activation_counts = sorted(activation_counts, reverse=True)

        plt.plot(neuron_indices, activation_counts, 'b-', alpha=0.6)
        plt.title('Activation Counts per Neuron Index')
        plt.xlabel('Neuron Index')
        plt.ylabel('Activation Count')

        plt.text(0.95, 0.95, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 3: Neuron indices vs Activation values
        plt.subplot(3, 1, 3)  
        neuron_indices = np.arange(inter_size)
        stats_text = (f'Mean value: {activation_values.mean():.3f}\n'
                     f'Std value: {activation_values.std():.3f}\n'
                     f'Max value: {activation_values.max():.3f}\n'
                     f'Min value: {activation_values.min():.3f}')
        
        activation_values = activation_values.detach().to(dtype=torch.float32).numpy()
        activation_values = sorted(activation_values, reverse=True)

        plt.plot(neuron_indices, activation_values, 'b-', alpha=0.6)
        plt.title('Activation Values per Neuron Index')
        plt.xlabel('Neuron Index')
        plt.ylabel('Activation Value')

        plt.text(0.95, 0.95, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plt.savefig(save_path)
        plt.close()
    
    return activation_rates
    # return activation_counts, activation_values, activation_markers

@torch.no_grad()
def construct_experts_by_rates(origin_rates, num_experts):

    # print("origin_rates:", origin_rates.shape)
    hidden_size = origin_rates.shape[0]
    neurons_per_expert = hidden_size // num_experts

    expert_groups = []
    expert_rates = []
    rates = origin_rates.float()
    # markers = activation_markers.float()

    expert_groups.append([])
    expert_rates.append(0.0)

    top_values, top_indices = torch.topk(rates, hidden_size)
    for i in range(num_experts):
        expert_indices = top_indices[i*neurons_per_expert:(i+1)*neurons_per_expert].tolist()
        rates = top_values[i*neurons_per_expert:(i+1)*neurons_per_expert].sum().item()
        expert_groups.append(expert_indices)
        expert_rates.append(rates)

    # Normalize expert rates, 1e-8 to avoid division by zero as GPTQ loss can be zero
    expert_rates = [e / (sum(expert_rates) + 1e-8) for e in expert_rates]
    return expert_groups, expert_rates

@torch.no_grad()
def lowrank_compress_svd(weight_matrix, lowrank_sparsity, save_path=None):
    U, S, Vh = torch.linalg.svd(weight_matrix.float(), full_matrices=False)

    if lowrank_sparsity is not None:
        rank = int(weight_matrix.shape[1] * (1 - lowrank_sparsity))
    else:
        S_sum = S.sum()
        cum = torch.cumsum(S, 0)
        rank = torch.searchsorted(cum, 0.99 * S_sum).item() + 1

    ratio = S[:rank].sum() / S.sum()
    # print(f"Rank: {rank}, Ratio: {ratio}")
    
    U_reduced = U[:, :]
    S_reduced = S[:]
    Vh_reduced = Vh[:, :rank]
    # print(U_reduced.shape, S_reduced.shape, Vh_reduced.shape)
    
    low_rank_matrix = torch.mm(U_reduced, torch.mm(torch.diag(S_reduced), Vh_reduced))
    # print(weight_matrix.shape, rank, low_rank_matrix.shape)

    if save_path:
        plt.figure(figsize=(12, 8))
        
        neuron_indices = np.arange(S.shape[0])
        stats_text = ()
        
        S_cpu = S.cpu().numpy()
        plt.plot(neuron_indices, S_cpu, 'b-', alpha=0.6)
        plt.title(f'SVD decomposition of weight matrix, rank={rank}')
        plt.xlabel('Neuron Index')
        plt.ylabel('Weight Value')

        plt.text(0.95, 0.95, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
       
        plt.tight_layout()
        
        plt.savefig(save_path)
        plt.close()
    
    return low_rank_matrix.to(weight_matrix.dtype)

@torch.no_grad()
def analyze_quant_outlier(layer, layer_idx, hidden_states, ori_expert_num, wbits=2, if_dense=False, save_path=None):
    print(f"analyze_quant_outlier layer: {layer_idx} with {wbits} bits")
    nsample = hidden_states.shape[0]

    gptq = {}
    groupsize = 128
    act_order = True
    static_groups = False
    filters = ['up_proj', 'gate_proj', 'down_proj']
    loss = {}
    
    for ff in filters:
        qmodule_all = find_layers(layer, filters=[ff])
        qbatch = min(QBATCH, ori_expert_num)

        for qmi in range(0, len(qmodule_all.keys()), qbatch):
            tick0 = time.time()

            qmodule = {k: qmodule_all[k] for k in list(qmodule_all.keys())[qmi: qmi + qbatch]}
            if len(qmodule.keys()) == 0:
                continue
            for name in qmodule.keys():
                split_name = name.split('.')[-1]
                gptq[name] = GPTQ(qmodule[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(wbits, perchannel=True, sym=False, mse=False)

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data.detach(), out.data.detach())
                return tmp
            handles = []
            for name in qmodule.keys():
                handles.append(qmodule[name].register_forward_hook(add_batch(name)))
            
            for isample in range(nsample):
                if split_name in filters:
                    ffn_sample = hidden_states[isample].unsqueeze(0)
                    with torch.no_grad():
                        layer.mlp(ffn_sample)
                else:
                    assert False, f"Not quantize {name}"

            for handle in handles:
                handle.remove()
            del handles

            for name in qmodule.keys():
                loss[name] = gptq[name].fasterquant(name=f"layer_idx.{layer_idx}."+name, groupsize=groupsize, actorder=act_order, static_groups=static_groups, update=False)
                gptq[name].free()
                del gptq[name]
            
            tick1 = time.time()
            print(f"Simulate quant to find outliers, layer {layer_idx} {ff} {qmi}:{qmi + min(qbatch, len(qmodule.keys()))} bits: {wbits} time: {tick1 - tick0:.4f}")
            del qmodule

        del qmodule_all

    del gptq

    # print(loss)
    all_rates = []
    if if_dense:
        assert ori_expert_num == 1, "dense model n == 1"
    for expert_idx in range(ori_expert_num):
        if ori_expert_num == 1:
            u = f'mlp.up_proj'
            g = f'mlp.gate_proj'
            d = f'mlp.down_proj'
        else:
            u = f'mlp.experts.{expert_idx}.up_proj'
            g = f'mlp.experts.{expert_idx}.gate_proj'
            d = f'mlp.experts.{expert_idx}.down_proj'
        
        up_proj_loss = torch.sum(loss[u], dim=1)
        gate_proj_loss = torch.sum(loss[g], dim=1)
        down_proj_loss = torch.sum(loss[d], dim=0)
        # print(up_proj_loss.shape, gate_proj_loss.shape, down_proj_loss.shape)
        all_rates.append(up_proj_loss + gate_proj_loss + down_proj_loss)
        # rates = up_proj_loss / up_proj_loss.mean() + gate_proj_loss / gate_proj_loss.mean()
        
        del up_proj_loss, gate_proj_loss, down_proj_loss

    # print(f"Layer {layer_idx}, neural loss rates: ", all_rates)

    if save_path:
        rates = all_rates[0]
        plt.figure(figsize=(10, 10))
        
        for i, pp in enumerate([rates, up_proj_loss, gate_proj_loss]):
            plt.subplot(3, 1, i + 1)
            pps = pp.detach().cpu().to(dtype=torch.float32).numpy()
            # pps = sorted(pps, reverse=True)

            neuron_indices = np.arange(pp.shape[0])
            plt.plot(neuron_indices, pps, 'b-', alpha=0.6)
            plt.title('Distribution of Neuron Loss Rates')
            plt.xlabel('Neuron Index')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            
            mean_rate = rates.mean()
            std_rate = rates.std()
            stats_text = (f'Mean rate: {mean_rate:.3f}\n'
                        f'Std rate: {std_rate:.3f}\n'
                        f'Max rate: {rates.max():.3f}\n'
                        f'Min rate: {rates.min():.3f}')
            
            plt.text(0.95, 0.95, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        plt.savefig(save_path)
        plt.close()
    
    del loss
    torch.cuda.empty_cache()
    gc.collect()

    return all_rates

@torch.no_grad()
def quant_layer_mix_precision(layer, layer_idx, quant_attn, n_experts, slice_expert_num,
                attn_hidden_states, ffn_hidden_states, attention_mask, position_ids, position_embeddings, 
                qscheme):
    print(f"Quantize layer {layer_idx}")
    nsample = attn_hidden_states.shape[0]
    assert attn_hidden_states.shape[0] == ffn_hidden_states.shape[0], f"attn_hidden_states.shape: {attn_hidden_states.shape}, ffn_hidden_states.shape: {ffn_hidden_states.shape}"

    gptq = {}
    groupsize = 128
    act_order = True
    static_groups = False
    sym = False
    ffn_filters = ['up_proj', 'gate_proj', 'down_proj']
    attn_filters = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'kv_a_proj_with_mqa', 'kv_b_proj']
    if quant_attn:
        filters = attn_filters + ffn_filters
    else:
        filters = ffn_filters

    loss = {}

    for ff in filters:
        qmodule_all = find_layers(layer, filters=[ff])
        qbatch = min(QBATCH, n_experts)

        for qmi in range(0, len(qmodule_all.keys()), qbatch):
            tick0 = time.time()

            qmodule = {k: qmodule_all[k] for k in list(qmodule_all.keys())[qmi: qmi + qbatch]}
            if len(qmodule.keys()) == 0:
                continue
            for name in qmodule.keys():
                split_name = name.split('.')[-1]
                gptq[name] = GPTQ(qmodule[name])
                gptq[name].quantizer = Quantizer()

                if split_name in attn_filters:
                    bit = qscheme['attn']
                    gptq[name].quantizer.configure(bit[0], perchannel=True, sym=sym, mse=False)
                else:
                    match = re.search(r'mlp\.experts\.(\d+)', name)
                    expert_id = int(match.group(1)) if match else -1  ## shared expert id is -1
                    # print(expert_id, slice_expert_num)
                    if expert_id == -1:
                        bit = qscheme['share']
                        gptq[name].quantizer.configure(bit[0], perchannel=True, sym=sym, mse=False)
                    else:
                        bit = qscheme['expert']
                        gptq[name].quantizer.configure(bit[expert_id // slice_expert_num][expert_id % slice_expert_num], perchannel=True, sym=sym, mse=False)

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data.detach(), out.data.detach())
                return tmp
            handles = []
            for name in qmodule.keys():
                handles.append(qmodule[name].register_forward_hook(add_batch(name)))
            
            for isample in range(nsample):
                if split_name in attn_filters:
                    attn_sample = attn_hidden_states[isample].unsqueeze(0)
                    try:
                        with torch.no_grad():
                            layer.self_attn(
                                hidden_states=attn_sample, 
                                attention_mask=attention_mask, 
                                position_ids=position_ids,
                                position_embeddings=position_embeddings)
                    except:
                        with torch.no_grad():
                            layer.self_attn(
                                hidden_states=attn_sample, 
                                attention_mask=attention_mask, 
                                position_ids=position_ids)
                elif split_name in ffn_filters:
                    ffn_sample = ffn_hidden_states[isample].unsqueeze(0)
                    with torch.no_grad():
                        layer.mlp(ffn_sample)
                else:
                    assert False, f"Not quantize {name}"

            for handle in handles:
                handle.remove()
            del handles

            tick1 = time.time()

            forward_event = torch.cuda.Event()
            forward_event.record(torch.cuda.current_stream())

            streams = []
            for name in qmodule.keys():
                ss = torch.cuda.Stream()
                streams.append(ss)
                with torch.cuda.stream(ss):
                    ss.wait_event(forward_event)
                    if gptq[name].quantizer.bits == 0:
                        gptq[name].layer.weight = nn.Parameter(torch.zeros_like(gptq[name].layer.weight))
                        loss[name] = torch.zeros(1)
                    else:
                        loss[name] = gptq[name].fasterquant(
                            name=f"layer_idx.{layer_idx}."+name,
                            groupsize=groupsize,
                            actorder=act_order,
                            static_groups=static_groups
                        )

            for ss in streams:
                ss.synchronize()

            for name in qmodule.keys():
                if gptq[name] is not None:
                    gptq[name].free()
                    del gptq[name]

            # print(ff, qmodule, loss)
            tick2 = time.time()
            print(f"Quantize layer {layer_idx} {ff} {qmi}:{qmi + min(qbatch, len(qmodule.keys()))} time: {tick1 - tick0:.4f} + {tick2 - tick1:.4f} loss: {loss[name].sum():.6f}")
            del qmodule

        del qmodule_all
    
    torch.cuda.synchronize()

    del loss, gptq
    torch.cuda.empty_cache()
    gc.collect()