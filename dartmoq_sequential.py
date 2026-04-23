import torch
import torch.nn as nn
import copy
import time
from tqdm import tqdm
from dartmoq_utils import *
from data_utils import *
from eval_dartmoq import cmoe_ppl_eval
from tool_utils import *

DEV = torch.device('cuda:0')

@torch.no_grad()
def reconstruct_moe_from_existing(model, layer, layer_idx, inps, n_experts, n_activated, slice_expert_num, ori_activated, device, args):

    expert_activation_rates = analyze_experts_activation(layer, layer_idx, inps, ori_activated, model.config.model_type) #, save_path="plot/{layer_idx}_experts_activation.png")

    ori_expert_num = len(layer.mlp.experts)
    new_expert_num = ori_expert_num * slice_expert_num 
    scaling_factor = slice_expert_num

    ori_router_gate = layer.mlp.gate.weight
    if type(layer.mlp.gate) == nn.Linear:
        new_router = nn.Linear(model.config.hidden_size, new_expert_num, dtype=ori_router_gate.dtype, bias=False).to(device)
    else:
        new_router = layer.mlp.gate.__class__(model.config).to(device).to(layer.mlp.gate.weight.dtype)
        # new_router.training = False ### for moonlight model
    # print(new_router)
    all_new_experts = nn.ModuleList()

    total_neurons_processed = 0
    gate_start_idx = 0

    tick0 = time.time()

    if args.rank_mode == "quant_outlier":
        all_rates = analyze_quant_outlier(layer, layer_idx, inps, ori_expert_num, if_dense=False, save_path=None)

    all_new_expert_rates = []
    for expert_idx, expert in enumerate(layer.mlp.experts):
        ori_gate_proj_weights = expert.gate_proj.weight
        ori_up_proj_weights = expert.up_proj.weight
        ori_down_proj_weights = expert.down_proj.weight

        # print(f"\nProcessing original expert {expert_idx} / {ori_expert_num}")
        if args.rank_mode == "activation":
            analyze_sparsity = 0.1
            rates = analyze_neuron_activations(expert.act_fn, inps, ori_gate_proj_weights, ori_up_proj_weights, sparsity=analyze_sparsity)
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
        _rates = [e * expert_activation_rates[expert_idx] for e in expert_rates[1:]]
        all_new_expert_rates.extend(_rates)

        # Create new experts for this original expert
        for ii, group_indices in enumerate(expert_groups):
            n_neurons = len(group_indices)
            expert_mlp = expert.__class__(model.config).to(device)
            
            with torch.no_grad():
                group_indices_tensor = torch.tensor(group_indices, dtype=torch.long, device=ori_gate_proj_weights.device)
                
                expert_mlp.gate_proj.weight.data = ori_gate_proj_weights[group_indices_tensor, :].detach().clone()
                expert_mlp.up_proj.weight.data = ori_up_proj_weights[group_indices_tensor, :].detach().clone()
                expert_mlp.down_proj.weight.data = ori_down_proj_weights[:, group_indices_tensor].detach().clone() * scaling_factor
                
            all_new_experts.append(expert_mlp)
            new_expert_intermediate_size = expert_mlp.up_proj.weight.shape[0]
            total_neurons_processed += new_expert_intermediate_size
            # print(expert_idx, ii, new_expert_intermediate_size, expert_mlp.gate_proj.weight.shape, expert_mlp.up_proj.weight.shape, expert_mlp.down_proj.weight.shape)
        
        expanded_gate = ori_router_gate.data[expert_idx, :].unsqueeze(0).repeat(slice_expert_num, 1).to(device).detach().clone()

        # print(f"gate_start_idx, slice_expert_num, expanded_gate.shape: {expert_idx, gate_start_idx, slice_expert_num, expanded_gate.shape}")
        new_router.weight.data[gate_start_idx: gate_start_idx + slice_expert_num, :] = expanded_gate
        gate_start_idx += slice_expert_num

        del group_indices_tensor, ori_gate_proj_weights, ori_up_proj_weights, ori_down_proj_weights, expanded_gate
        gc.collect()
        torch.cuda.empty_cache()

    tick1 = time.time()
    print(f"Layer {layer_idx}, {args.rank_mode} expert re- sort time: {tick1 - tick0}")
    print("all_new_expert_rates:", len(all_new_expert_rates))

    moe = layer.mlp.__class__(model.config).to(device)
    moe.num_experts = len(all_new_experts)
    moe.top_k = n_activated
    moe.gate = new_router
    moe.experts = all_new_experts
    if hasattr(layer.mlp, 'shared_experts'):
        moe.shared_experts = layer.mlp.shared_experts

    del all_rates
    gc.collect()
    torch.cuda.empty_cache()
    return moe, all_new_expert_rates

@torch.no_grad()
def construct_moe(model, moe_model_flag, layer, layer_idx, inp, attention_mask, position_ids, position_embeddings, 
                                n_experts, n_activated, slice_expert_num, n_shared, ori_activated, args):
    
    modeltype = model.config.model_type
    batchsize = inp.shape[0]

    device = next(layer.parameters()).device
    # print(layer, device)
    # print(inp.shape)

    # Forward attention
    inp = inp.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    if position_ids is not None:
        position_ids = position_ids.to(device)
    
    residual = inp
    with torch.no_grad():
        hidden_states_inorm = layer.input_layernorm(inp)

    tick0 = time.time()
    attn_out = torch.zeros_like(hidden_states_inorm)
    for b_i in range(0, batchsize):
        # print(modeltype)
        if modeltype == 'olmoe' or modeltype == 'llama' or modeltype == 'qwen3' or modeltype == 'qwen3_moe' or modeltype == 'deepseek_v3':
            with torch.no_grad():
                attn_out[b_i:b_i+1] = layer.self_attn(
                    hidden_states=hidden_states_inorm[b_i:b_i+1],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings)[0]
        else:
            with torch.no_grad():
                attn_out[b_i:b_i+1] = layer.self_attn(
                    hidden_states=hidden_states_inorm[b_i:b_i+1],
                    attention_mask=attention_mask, 
                    position_ids=position_ids)[0]
    tick1 = time.time()
    print(f"Inference in origin attention layer {layer_idx} with batch size {batchsize} time: {tick1 - tick0}")

    hidden_states = residual + attn_out
    residual = hidden_states
    with torch.no_grad():
        hidden_states = layer.post_attention_layernorm(hidden_states)

    # print(hidden_states.shape)
    is_moe_layer = hasattr(layer.mlp, 'gate') or hasattr(layer.mlp, 'experts') ## some moe model has no expert layer in the first few layers,
    
    all_new_expert_rates = None
    if moe_model_flag:
        if is_moe_layer:
            moe, all_new_expert_rates = reconstruct_moe_from_existing(model, layer, layer_idx, hidden_states, n_experts, n_activated, slice_expert_num, ori_activated, device, args)
            layer.mlp = moe
    else:
        # moe = reconstruct_moe_from_dense(model, layer, layer_idx, hidden_states, n_experts, n_activated, slice_expert_num, device, args)
        # layer.mlp = moe
        assert False, "Dense model is not supported"

    gc.collect()
    torch.cuda.empty_cache()

    if args.quant_scheme == "global":
        qscheme = {}
        qscheme['attn'] = [8]
        qscheme['share'] = [8]
        qscheme['expert'] = []
        
        # print(all_new_expert_rates)
        if all_new_expert_rates is not None:
            _, sorted_index = torch.sort(torch.tensor(all_new_expert_rates))
            high_ratio = 0.25
            low_ratio = 1 - high_ratio
            qscheme['expert'] = [[0] * slice_expert_num for i in range(n_experts // slice_expert_num)]
            for i, idx in enumerate(sorted_index):
                if i < int(n_experts * low_ratio):
                    bit = 2
                else:
                    bit = 4
                # print(idx, all_new_expert_rates[idx])
                xi = int(idx // slice_expert_num)
                xj = int(idx % slice_expert_num)
                qscheme['expert'][xi][xj] = bit

        print("global quant expert scheme:", qscheme['expert'])
    else:
        # args.quant_scheme should lik "a8s4m3221"
        qscheme_str = args.quant_scheme
        qscheme = {}
        qscheme['attn'] = [8]
        qscheme['share'] = [4]
        # qscheme['expert'] = [2, 2, 2, 2, 2, 2, 2, 2]
        if qscheme_str is not None:
            try:
                # sample: "a8s4m3221", "a8s4m33222222"
                match = re.search(r'a(\d)s(\d)m(\d+)', qscheme_str)
                aa = match.group(1)
                ss = match.group(2)
                ee = match.group(3)
                qscheme['attn'] = [int(aa)]
                qscheme['share'] = [int(ss)]
                qscheme['expert'] = [[int(e) for e in ee] for i in range(n_experts // slice_expert_num)]
                print(qscheme['expert'])
            except:
                assert False, f"Quant scheme {qscheme_str} is not valid."
    
    if_quant_attn = True
    quant_layer_mix_precision(layer, layer_idx, if_quant_attn, n_experts, slice_expert_num,
                hidden_states_inorm, hidden_states, attention_mask, position_ids, position_embeddings, 
                qscheme)
    gc.collect()
    torch.cuda.empty_cache()
    
    print(hidden_states.shape)
    tick0 = time.time()
    moe_out = torch.zeros_like(hidden_states)
    for b_i in range(0, batchsize):
        if modeltype == 'olmoe' or modeltype == 'qwen3_moe' or modeltype == 'qwen3':
            with torch.no_grad():
                moe_out[b_i:b_i+1], _ = layer.mlp(hidden_states[b_i:b_i+1])
        else:
            with torch.no_grad():
                moe_out[b_i:b_i+1] = layer.mlp(hidden_states[b_i:b_i+1])

    with torch.no_grad():
        moe_out = moe_out + residual

    del hidden_states, hidden_states_inorm, residual, attn_out, all_new_expert_rates

    gc.collect()
    torch.cuda.empty_cache()

    tick1 = time.time()
    print(f"Inference in new moe layer {layer_idx} with batch size {batchsize} time: {tick1 - tick0}", flush=True)
    return moe_out

@torch.no_grad()
def cmoe_sequential(model, tokenizer, dataloader, args):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    dtype = next(iter(model.parameters())).dtype
    bsz = 1
    
    inps = torch.zeros(
        (args.nsamples//bsz, bsz, model.seqlen, model.config.hidden_size), dtype=dtype, device='cpu'
    )
    print(inps.shape)
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None, 'position_embeddings': None}

    if args.standby_layer_cpu:
        model.model.embed_tokens = model.model.embed_tokens.to(DEV)

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):

            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            cache['position_embeddings'] = kwargs.get('position_embeddings')
            raise ValueError
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

    layers[0] = Catcher(layers[0])
    
    with torch.no_grad():
        for batch in dataloader:
            try:
                model(batch[0].to(DEV))
            except ValueError:
                pass

    layers[0] = layers[0].module

    torch.cuda.empty_cache()

    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']
    # print("position_embeddings:", position_embeddings)
    # print(cache)

    print('Ready.')
    # model.cuda()
    # layers.cuda()

    # MoE Carving
    moe_model_flag = False
    for layer in layers:
        moe_model_flag = moe_model_flag or hasattr(layer.mlp, 'gate') or hasattr(layer.mlp, 'experts')
    if moe_model_flag:
        if hasattr(model.config, 'num_experts'):         ## olmoe，
            slice_expert_num = args.nexperts // model.config.num_experts
            assert slice_expert_num * model.config.num_experts == args.nexperts, "n_experts must be multiple of existing expert num"
            model.config.num_experts = args.nexperts
        elif hasattr(model.config, 'n_routed_experts'):  ## DeepSeek-V1-MoE-16B
            slice_expert_num = args.nexperts // model.config.n_routed_experts
            assert slice_expert_num * model.config.n_routed_experts == args.nexperts, "n_experts must be multiple of existing expert num"
            model.config.n_routed_experts = args.nexperts

        ori_num_experts_per_tok = model.config.num_experts_per_tok
        model.config.num_experts_per_tok = args.nactivated
        if hasattr(model.config, 'moe_intermediate_size'): ## DeepSeek-V1-MoE-16B
            model.config.moe_intermediate_size = model.config.moe_intermediate_size // slice_expert_num
        elif hasattr(model.config, 'intermediate_size'): ## olmoe，
            model.config.intermediate_size = model.config.intermediate_size // slice_expert_num
        print("The model is already a MoE model. Proceeding to split experts. ")
        print(f"Slice expert by {slice_expert_num}: to {args.nexperts}, with {args.nactivated} activated experts.")
    else:
        print("The model is a dense model. Proceeding to carve MoE layers. ")
        slice_expert_num = args.nexperts
    
    inps = inps.squeeze(1)

    if args.standby_layer_cpu:
        layers_device = []
        for layer_idx, layer in enumerate(model.model.layers):
            dev = next(layer.parameters()).device
            layers_device.append(dev)
            # print(layer_idx, dev)
            if dev.type == 'cuda':
                layer = layer.to('cpu')
        for i in range(torch.cuda.device_count()):
            force_release_inactive_splits(device=i)
            print(f"CUDA {i} Allocated: {torch.cuda.memory_allocated(device=i) / 1024**3:.2f} GB")
            print(f"CUDA {i} Reserved: {torch.cuda.memory_reserved(device=i) / 1024**3:.2f} GB")       
        print(layers_device)
    
    for layer_idx, layer in tqdm(enumerate(layers), desc = 'Carving MoE layers...'):
        if args.standby_layer_cpu:
            layer = layer.to(layers_device[layer_idx])

        moe_out = construct_moe(model,
            moe_model_flag,
            layer, 
            layer_idx,
            inps, 
            attention_mask, 
            position_ids,
            position_embeddings,
            n_experts = args.nexperts,
            n_activated = args.nactivated,
            slice_expert_num = slice_expert_num,
            n_shared = args.nshared,
            ori_activated = ori_num_experts_per_tok,
            args = args
        )

        inps = moe_out

        if args.standby_layer_cpu:
            layer = layer.to('cpu')

        for i in range(torch.cuda.device_count()):
            # force_release_inactive_splits(device=i) # force to release inactive reserved memory
            print(f"CUDA {i} Allocated: {torch.cuda.memory_allocated(device=i) / 1024**3:.2f} GB")
            print(f"CUDA {i} Reserved: {torch.cuda.memory_reserved(device=i) / 1024**3:.2f} GB")
        print(flush=True)
        
    print("MoE carving done. Moving layers to GPU for evaluation...")

    if args.standby_layer_cpu:
        for i in range(torch.cuda.device_count()):
            force_release_inactive_splits(device=i)
        for layer_idx, layer in enumerate(model.model.layers):
            if layers_device[layer_idx].type == 'cuda':
                layer = layer.to(layers_device[layer_idx])
            for i in range(torch.cuda.device_count()):
                print(f"layer {layer_idx} CUDA {i} Allocated: {torch.cuda.memory_allocated(device=i) / 1024**3:.2f} GB")
                print(f"layer {layer_idx} CUDA {i} Reserved: {torch.cuda.memory_reserved(device=i) / 1024**3:.2f} GB")
        
        # for name, param in model.named_parameters():
        #     print(f"{name:<40} → {param.device}")

    # print('Training_free_ppl:')
    pre_ppl = []
    datasets = ['wikitext2', 'c4-new']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, tokenizer=tokenizer, seqlen=model.seqlen
        )
        print(dataset)
        eval_set = dataset
        ppl_i = cmoe_ppl_eval(model, testloader, eval_set, args)
        pre_ppl.append(f"{dataset}: {ppl_i}")
    
    return model
