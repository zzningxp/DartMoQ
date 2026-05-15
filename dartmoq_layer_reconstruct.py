import torch
import torch.nn as nn
import os
import time
import gc
from dartmoq_utils import analyze_experts_activation
from dartmoq_utils import construct_experts_by_rates
from dartmoq_utils import analyze_neuron_activations
from dartmoq_utils import analyze_quant_outlier
from camera_utils import analyze_expert_energy
from dp_utils import enum_optimal_m_scheme_fast_general
from dp_utils import extrapolate_0bit_loss
from collections import Counter
from dartmoq_hybridmoe import DartMoQHybridWrapper
from dartmoq_hybridmoe import restructure_hybrid_qscheme

@torch.no_grad()
def reconstruct_moe_from_existing(model, layer, layer_idx, inps, 
                                  n_experts, n_activated, slice_expert_num, 
                                  ori_activated, device, qscheme, use_hybrid_moe, global_mode, args):
    if global_mode:
        expert_activation_rates = analyze_experts_activation(layer, layer_idx, inps, ori_activated, model.config.model_type)

    ori_expert_num = len(layer.mlp.experts)
    
    if use_hybrid_moe:
        # Hybrid MoE: keep original expert count at first level
        new_expert_num = ori_expert_num
    else:
        new_expert_num = ori_expert_num * slice_expert_num 
        scaling_factor = slice_expert_num

    ori_router_gate = layer.mlp.gate.weight
    
    if use_hybrid_moe:
        # Hybrid MoE uses nested structure: experts -> sub-experts
        all_new_experts = []  # List of lists (each expert has sub-experts)
    else:
        if type(layer.mlp.gate) == nn.Linear:
            new_router = nn.Linear(model.config.hidden_size, new_expert_num, dtype=ori_router_gate.dtype, bias=False).to(device)
        else:
            new_router = layer.mlp.gate.__class__(model.config).to(device).to(layer.mlp.gate.weight.dtype)
        all_new_experts = nn.ModuleList()

    total_neurons_processed = 0
    gate_start_idx = 0
    
    # For hybrid MoE, we need to track sub-expert bit configs
    sub_expert_bit_configs = []
    expert_to_subexperts = []

    probe_bit = 2
    if args.rank_mode == "quant_outlier":
        tick0 = time.time()

        q_rates = {}
        if 'target_bpw' not in qscheme:
            outlier_bits = {probe_bit}
        else:
            outlier_bits = {0, 1, 2, 3, 4}
        print(f"simulate quant outlier_bits {outlier_bits}")

        cache_dir = f"quant_outlier_/{model.model_id}"
        os.makedirs(cache_dir, exist_ok=True)
        
        for x in sorted(outlier_bits, reverse=True):  ## 0 bit should be extrapolated from other bit data, so we compute it at last
            cache_path = os.path.join(cache_dir, f"{model.model_id}_L{layer_idx}_b{x}.pt")
            if os.path.exists(cache_path):
                try:
                    cached_data = torch.load(cache_path, map_location=device)
                    print(f"Loading cached quant outlier data for layer {layer_idx}, wbits={x}", flush=True)
                    q_rates[x] = cached_data
                    continue
                except Exception as e:
                    print(f"Failed to load cached data {e}")
            
            if x == 0:
                print(f"Computing extrapolate 0 bit loss for layer {layer_idx}")
                q_rates[0] = extrapolate_0bit_loss(q_rates)
                q_rates[0] = [torch.from_numpy(q_rates[0][i]).to(device) for i in range(len(q_rates[0]))]
            else:
                print(f"Computing quant outlier for layer {layer_idx}, wbits={x}")
                q_rates[x] = analyze_quant_outlier(layer, layer_idx, inps, ori_expert_num, wbits=x, save_path=None)
            torch.save(q_rates[x], cache_path)
            print(f"Saved quant outlier data to {cache_path}")
        
        if 'target_bpw' not in qscheme:
            all_rates = q_rates[probe_bit]
        else:
            all_rates = []
            dpscheme_list = []
            for expert_idx in range(ori_expert_num):
                rates_x = {}
                for x in outlier_bits:
                    rates_x[x] = q_rates[x][expert_idx].detach().cpu().numpy()
                # print(f"expert_idx {expert_idx} scheme search:")
                dpscheme, rates = enum_optimal_m_scheme_fast_general(rates_x, slice_expert_num, target_bpw=qscheme['target_bpw'])
                dpscheme_list.append(dpscheme)
                rates = torch.from_numpy(rates).to(device)
                all_rates.append(rates)
            
        # from visual_utils import plot_diff_wbits_correlation, plot_spearman_rank_correlation
        # # plot_diff_wbits_correlation(model.config.model_type, layer_idx, ori_expert_num, q_rates[2], q_rates[3], q_rates[4])
        # plot_spearman_rank_correlation(model.config.model_type, layer_idx, ori_expert_num, q_rates[2], q_rates[3], q_rates[4])
        tick1 = time.time()
        print(f"analyze quant outlier time {tick1 - tick0}", flush=True)

    tick0 = time.time()

    all_new_expert_rates = []
    all_expert_groups = []  # Store groups for each expert
    
    for expert_idx, expert in enumerate(layer.mlp.experts):
        # print(f"\nProcessing original expert {expert_idx} / {ori_expert_num}")
        if args.rank_mode == "activation":
            ori_gate_proj_weights = expert.gate_proj.weight
            ori_up_proj_weights = expert.up_proj.weight
            ori_down_proj_weights = expert.down_proj.weight

            analyze_sparsity = 0.1
            rates = analyze_neuron_activations(expert.act_fn, inps, ori_gate_proj_weights, ori_up_proj_weights, sparsity=analyze_sparsity)
        elif args.rank_mode == "energy":
            rates = analyze_expert_energy(expert, inps)
        elif args.rank_mode == "quant_outlier":
            rates = all_rates[expert_idx]
        elif args.rank_mode == "random":
            rates = torch.randn(layer.mlp.intermediate_size, device=device)
        elif args.rank_mode == "neuron_index":
            rates = torch.arange(layer.mlp.intermediate_size, device=device)
        else:
            assert False, f"Unknown rank mode: {args.rank_mode}"
        
        expert_groups, expert_rates = construct_experts_by_rates(
            rates,
            num_experts = slice_expert_num
        )
        
        expert_groups = expert_groups[1:]
        all_expert_groups.append(expert_groups)
        
        if global_mode:
            _rates = [e * expert_activation_rates[expert_idx] for e in expert_rates[1:]]
            all_new_expert_rates.extend(_rates)
        else:
            all_new_expert_rates.extend(expert_rates[1:])

    # print(qscheme)
    if 'target_bpw' in qscheme:
        qscheme['expert'] = dpscheme_list
        counter = Counter(dpscheme_list)
        print(f"layer {layer_idx} {qscheme['target_bpw']} dpscheme_list scheme type count: {counter}")
    elif global_mode:
        ee = qscheme['econfig']
        e_bits = [int(e) for e in ee]

        if all_new_expert_rates is not None:
            _, sorted_index = torch.sort(torch.tensor(all_new_expert_rates), descending=True)
            # print(e_bits, sect_, new_expert_num)
            qscheme['expert'] = [[0] * slice_expert_num for i in range(ori_expert_num)]
            for i, idx in enumerate(sorted_index):
                # print(idx, all_new_expert_rates[idx])
                xi = int(idx // slice_expert_num)
                xj = int(idx % slice_expert_num)
                qscheme['expert'][xi][xj] = e_bits[i // ori_expert_num]
    else:
        qscheme['expert'] = [qscheme['econfig'] for i in range(ori_expert_num)]
    
    # For hybrid MoE: restructure qscheme to group by bit config
    if use_hybrid_moe:
        qscheme['slice_expert'] = qscheme['expert']
        qscheme['expert'] = restructure_hybrid_qscheme(qscheme['slice_expert'], slice_expert_num)

    for expert_idx, expert in enumerate(layer.mlp.experts):
        ori_gate_proj_weights = expert.gate_proj.weight
        ori_up_proj_weights = expert.up_proj.weight
        ori_down_proj_weights = expert.down_proj.weight
        
        # Get groups for this specific expert
        expert_groups = all_expert_groups[expert_idx]

        if use_hybrid_moe:
            # Hybrid MoE: group sub-experts by bit config
            expert_sub_experts = []
            expert_sub_sizes = []

            orig_bit_config = qscheme['slice_expert'][expert_idx]
            restructured_config = qscheme['expert'][expert_idx]
            # print("orig_bit_config:", orig_bit_config, "restructured_config:", restructured_config)

            bit_to_indices = {}
            bit_to_slice_count = {}
            
            for bit, group_indices in zip(orig_bit_config, expert_groups):
                if bit not in bit_to_indices:
                    bit_to_indices[bit] = []
                    bit_to_slice_count[bit] = 0
                bit_to_indices[bit].extend(group_indices)
                bit_to_slice_count[bit] += 1
            
            for bit in restructured_config:
                indices = bit_to_indices[bit]
                n_neurons = len(indices)

                # print(f"layer {layer_idx} expert {expert_idx} bit={bit} n_neurons={n_neurons}")
                # print(bit, indices)
                # if expert_idx < 2:
                #     print(f"layer {layer_idx} expert {expert_idx} bit={bit} n_neurons={n_neurons}, indices[:5]={indices[:5]} {indices[-5:]}")
                new_config = model.config
                new_config.intermediate_size = n_neurons
                expert_mlp = expert.__class__(new_config).to(device)
                
                with torch.no_grad():
                    indices_tensor = torch.tensor(indices, dtype=torch.long, device=ori_gate_proj_weights.device)
                    expert_mlp.gate_proj.weight.data = ori_gate_proj_weights[indices_tensor, :].detach().clone()
                    expert_mlp.up_proj.weight.data = ori_up_proj_weights[indices_tensor, :].detach().clone()
                    expert_mlp.down_proj.weight.data = ori_down_proj_weights[:, indices_tensor].detach().clone()
                
                expert_sub_experts.append(expert_mlp)
                expert_sub_sizes.append(n_neurons)
                total_neurons_processed += n_neurons
            
            all_new_experts.append(expert_sub_experts)
            sub_expert_bit_configs.append(tuple(restructured_config))
            expert_to_subexperts.append(list(range(len(expert_sub_experts))))
            
            # For hybrid MoE, router stays the same (one entry per original expert)
        else:
            # Original behavior: create separate expert for each slice
            for ii, group_indices in enumerate(expert_groups):
                n_neurons = len(group_indices)
                # if expert_idx < 2:
                #     print(f"layer {layer_idx} expert {expert_idx} slice {ii} n_neurons={n_neurons}, group_indices={group_indices[:5]} {group_indices[-5:]}")
                new_config = model.config
                new_config.intermediate_size = n_neurons
                expert_mlp = expert.__class__(new_config).to(device)
                
                with torch.no_grad():
                    group_indices_tensor = torch.tensor(group_indices, dtype=torch.long, device=ori_gate_proj_weights.device)
                    expert_mlp.gate_proj.weight.data = ori_gate_proj_weights[group_indices_tensor, :].detach().clone()
                    expert_mlp.up_proj.weight.data = ori_up_proj_weights[group_indices_tensor, :].detach().clone()
                    expert_mlp.down_proj.weight.data = ori_down_proj_weights[:, group_indices_tensor].detach().clone() * scaling_factor
                
                all_new_experts.append(expert_mlp)
                new_expert_intermediate_size = expert_mlp.up_proj.weight.shape[0]
                total_neurons_processed += new_expert_intermediate_size
            
            expanded_gate = ori_router_gate.data[expert_idx, :].unsqueeze(0).repeat(slice_expert_num, 1).to(device).detach().clone()
            new_router.weight.data[gate_start_idx: gate_start_idx + slice_expert_num, :] = expanded_gate
            gate_start_idx += slice_expert_num

        del ori_gate_proj_weights, ori_up_proj_weights, ori_down_proj_weights
        if 'group_indices_tensor' in locals():
            del group_indices_tensor
        if 'expanded_gate' in locals():
            del expanded_gate
        gc.collect()
        torch.cuda.empty_cache()

    tick1 = time.time()
    print(f"Layer {layer_idx}, {args.rank_mode} expert re- sort time: {tick1 - tick0}", flush=True)
    print("all_new_expert_rates:", len(all_new_expert_rates))

    if use_hybrid_moe:
        # Create hybrid MoE using the original model's MLP class
        moe = layer.mlp.__class__(model.config).to(device)
        
        # Keep gate and top_k configuration consistent with original
        moe.gate = layer.mlp.gate
        moe.num_experts = len(all_new_experts)
        
        # Replace experts with nn.ModuleList of DartMoQHybridWrapper wrappers
        # Each DartMoQHybridWrapper wraps multiple sub-experts with different bit configs
        moe.experts = nn.ModuleList([DartMoQHybridWrapper(sub_experts) for sub_experts in all_new_experts])
        
        counter = Counter(sub_expert_bit_configs)
        print("reconstruct moe with sub_expert_bit_configs: ", counter) 
        
        # Copy shared_experts if exists
        if hasattr(layer.mlp, 'shared_experts'):
            moe.shared_experts = layer.mlp.shared_experts
        moe.training = False
    else:
        # Original behavior
        moe = layer.mlp.__class__(model.config).to(device)
        moe.num_experts = len(all_new_experts)
        moe.top_k = n_activated
        moe.gate = new_router
        moe.experts = all_new_experts
        if hasattr(layer.mlp, 'shared_experts'):
            moe.shared_experts = layer.mlp.shared_experts
    gc.collect()
    torch.cuda.empty_cache()

    return moe
