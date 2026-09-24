import torch
import torch.nn as nn
import os
import time
import re
from dartmoq_utils import *
from data_utils import *
from eval_dartmoq import cmoe_ppl_eval
from tool_utils import *
from dartmoq_layer_reconstruct import reconstruct_moe_from_existing

DEV = torch.device('cuda:0')

@torch.no_grad()
def construct_moe(model, moe_model_flag, layer, layer_idx, inp, 
                    attention_mask, position_ids, position_embeddings, 
                    n_experts, n_activated, slice_expert_num, ori_activated, 
                    qscheme, args):
    
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

    # tick0 = time.time()
    # adapt the mask per mini-batch: the catcher caches a single-sample
    # (bsz=1) mask, and models that validate it strictly (dsv2/dsv3 etc.)
    # require the mask batch to match the input. All calibration samples share
    # the same seqlen and mask semantics, so broadcasting is correct; do not
    # modify attention_mask itself (the analyze stage runs single-sample
    # forwards that need the original batch=1 mask)
    def _mask_for(b_i, end_i):
        if attention_mask is None:
            return None
        mb = end_i - b_i
        if attention_mask.shape[0] == mb:
            return attention_mask
        if attention_mask.shape[0] == 1:
            return attention_mask.expand(mb, *attention_mask.shape[1:])
        return attention_mask

    attn_out = torch.zeros_like(hidden_states_inorm)
    mini_batch = 8
    for b_i in range(0, batchsize, mini_batch):
        end_i = min(b_i + mini_batch, batchsize)
        # print(modeltype)
        if modeltype == 'olmoe' or modeltype == 'llama' or modeltype == 'qwen3' or modeltype == 'qwen3_moe' or modeltype == 'deepseek_v3':
            with torch.no_grad():
                attn_out[b_i:end_i] = layer.self_attn(
                    hidden_states=hidden_states_inorm[b_i:end_i],
                    attention_mask=_mask_for(b_i, end_i),
                    position_ids=position_ids,
                    position_embeddings=position_embeddings)[0]
        else:
            with torch.no_grad():
                attn_out[b_i:end_i] = layer.self_attn(
                    hidden_states=hidden_states_inorm[b_i:end_i],
                    attention_mask=_mask_for(b_i, end_i),
                    position_ids=position_ids)[0]
    # tick1 = time.time()
    # print(f"Inference in origin attention layer {layer_idx} with batch size {batchsize} time: {tick1 - tick0}")

    hidden_states = residual + attn_out
    residual = hidden_states
    with torch.no_grad():
        hidden_states = layer.post_attention_layernorm(hidden_states)

    # print(hidden_states.shape)
    is_moe_layer = hasattr(layer.mlp, 'gate') or hasattr(layer.mlp, 'experts') ## some moe model has no expert layer in the first few layers,
    
    tick0 = time.time()
    use_hybrid_moe = getattr(args, 'use_hybrid_moe', False)
    quantmode = getattr(args, 'quantmode', 'gptq')
    global_mode = "global" in args.quant_scheme
    
    # WxA16 real-quantization switch: enabled explicitly by --wxa16, or
    # implicitly by --save-quantized (saving a packed checkpoint implies the
    # real quantization path)
    use_wxa16 = getattr(args, 'wxa16', False) or bool(getattr(args, 'save_quantized', None))
    if use_wxa16 and not use_hybrid_moe:
        assert False, "the WxA16 packed path requires the hybrid MoE structure (do not combine with --no-use-hybrid-moe)"

    if moe_model_flag:
        if is_moe_layer:
            moe, layer_metadata = reconstruct_moe_from_existing(model, layer, layer_idx, hidden_states,
                                                n_experts, n_activated, slice_expert_num, ori_activated, device,
                                                qscheme, use_hybrid_moe, global_mode, quantmode, args)
            layer.mlp = moe
    else:
        # moe = reconstruct_moe_from_dense(model, layer, layer_idx, hidden_states, n_experts, n_activated, slice_expert_num, device, args)
        # layer.mlp = moe
        assert False, "Dense model is not supported"
    gc.collect()
    torch.cuda.empty_cache()
    tick1 = time.time()
    print(f"reconstruct_moe_from_existing layer {layer_idx} time: {tick1 - tick0}", flush=True)

    tick0 = time.time()
    if_quant_attn = True
    if use_wxa16:
        # real quantization path: attention/shared become WxA16Linear,
        # the MoE is handled by from_build_block
        from dartmoq_utils import quant_layer_mix_precision_wxa16
        quant_layer_mix_precision_wxa16(layer, layer_idx, if_quant_attn, n_experts, slice_expert_num,
                    hidden_states_inorm, hidden_states, attention_mask, position_ids, position_embeddings,
                    qscheme, use_hybrid_moe, quantmode, seed=args.seed, group_size=GROUPSIZE)
    else:
        quant_layer_mix_precision(layer, layer_idx, if_quant_attn, n_experts, slice_expert_num,
                    hidden_states_inorm, hidden_states, attention_mask, position_ids, position_embeddings,
                    qscheme, use_hybrid_moe, quantmode, seed=args.seed)
    gc.collect()
    torch.cuda.empty_cache()
    tick1 = time.time()
    print(f"quant_layer_mix_precision layer {layer_idx} time: {tick1 - tick0}", flush=True)

    if use_wxa16 and moe_model_flag and is_moe_layer:
        # restructure into WxA16BitPartitionedGroupMoE: bit-bucketed packed
        # layout + triton-accelerated forward. Must run after attention/shared
        # quantization: from_build_block reuses the shared-expert submodules
        # that were already swapped to WxA16Linear (the same objects, keeping
        # the state_dict key structure consistent).
        tick_restructure = time.time()
        from quantization.wxa16.bit_partitioned_moe import WxA16BitPartitionedGroupMoE
        old_mlp = layer.mlp
        layer.mlp = WxA16BitPartitionedGroupMoE.from_build_block(old_mlp, layer_metadata, group_size=GROUPSIZE)

        # explicit cleanup of the old structure (gate/shared modules are
        # reused by the new structure, so only references are dropped; the
        # sub-expert weights were released block by block inside
        # from_build_block)
        if hasattr(old_mlp, 'experts'):
            for expert_wrapper in old_mlp.experts:
                if expert_wrapper is not None and hasattr(expert_wrapper, 'sub_experts'):
                    for sub_expert in expert_wrapper.sub_experts:
                        if hasattr(sub_expert, 'gate_proj'):
                            del sub_expert.gate_proj
                        if hasattr(sub_expert, 'up_proj'):
                            del sub_expert.up_proj
                        if hasattr(sub_expert, 'down_proj'):
                            del sub_expert.down_proj
                    del expert_wrapper.sub_experts
                if expert_wrapper is not None and hasattr(expert_wrapper, 'bit_to_indices'):
                    del expert_wrapper.bit_to_indices
            del old_mlp.experts
        if hasattr(old_mlp, 'gate'):
            del old_mlp.gate
        if hasattr(old_mlp, 'shared_expert'):
            del old_mlp.shared_expert
        if hasattr(old_mlp, 'shared_expert_gate'):
            del old_mlp.shared_expert_gate
        if hasattr(old_mlp, 'shared_experts'):
            del old_mlp.shared_experts
        del old_mlp
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Restructured to WxA16BitPartitionedGroupMoE layer {layer_idx}: {time.time() - tick_restructure:.4f}s", flush=True)

    # tick0 = time.time()
    # ---- DEBUG (device tracing for the packed path's final forward; remove after verified) ----
    print(f"  [DEBUG] final forward: hidden_states.device={hidden_states.device}, "
          f"layer param device={next(layer.parameters()).device}", flush=True)
    for _n, _m in layer.mlp.named_modules():
        if hasattr(_m, 'packed_indices'):
            print(f"  [DEBUG]   mlp.{_n}.packed_indices.device={_m.packed_indices.device}, "
                  f"norms.device={_m.norms.device}", flush=True)
    # ----------------------------------------------------------------
    moe_out = torch.zeros_like(hidden_states)
    mini_batch = 8
    for b_i in range(0, batchsize, mini_batch):
        end_i = min(b_i + mini_batch, batchsize)
        mlp_out = layer.mlp(hidden_states[b_i:end_i])
        if isinstance(mlp_out, tuple):
            moe_out[b_i:end_i] = mlp_out[0]
        else:
            moe_out[b_i:end_i] = mlp_out

    with torch.no_grad():
        moe_out = moe_out + residual

    # per-layer sync: async kernel errors (e.g. illegal memory accesses) are
    # reported lazily on CUDA; without the sync they surface at the next
    # layer's .to() as "CUDA error: unknown error", hiding the actual failing
    # layer/kernel. One sync per layer costs ~ms and is negligible.
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    del hidden_states, hidden_states_inorm, residual, attn_out

    gc.collect()
    torch.cuda.empty_cache()

    # tick1 = time.time()
    # print(f"Inference in new moe layer {layer_idx} with batch size {batchsize} time: {tick1 - tick0}", flush=True)
    return moe_out

@torch.no_grad()
def dartmoq_sequential(model, tokenizer, dataloader, args, test_ppl=True):
    print('Starting ...')
    tick_quant_start = time.time()

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

    # For CPU standby mode, we need to temporarily put embed_tokens and first layer on GPU
    # to collect the initial inputs
    if args.standby_layer_cpu:
        model.model.embed_tokens = model.model.embed_tokens.to(DEV)
        # Need at least layer 0 on DEV to collect inputs
        layers[0] = layers[0].to(DEV)

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):

            inps[cache['i']] = inp.cpu()  # Save to CPU
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

    # Move layer 0 back to CPU if needed
    if args.standby_layer_cpu:
        layers[0] = layers[0].to('cpu')

    torch.cuda.empty_cache()

    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    # Generate position_embeddings properly - it's not passed as kwargs to layers
    # but created by model.model.rotary_emb(hidden_states, position_ids)
    position_embeddings = cache.get('position_embeddings')
    if position_embeddings is None and hasattr(model.model, 'rotary_emb'):
        # For models like Qwen3 that use rotary_emb
        # Create a dummy hidden_states to generate position_embeddings
        with torch.no_grad():
            # Get the first batch to generate position_embeddings
            for batch in dataloader:
                dummy_input = batch[0].to(DEV)
                dummy_hidden = model.model.embed_tokens(dummy_input)
                if position_ids is None:
                    position_ids = torch.arange(0, dummy_input.shape[1], device=DEV).unsqueeze(0)
                position_embeddings = model.model.rotary_emb(dummy_hidden, position_ids)
                break
    # print("position_embeddings:", position_embeddings)
    # print(cache)

    print('Ready.')
    # model.cuda()
    # layers.cuda()

    moe_model_flag = False
    for layer in layers:
        moe_model_flag = moe_model_flag or hasattr(layer.mlp, 'gate') or hasattr(layer.mlp, 'experts')

    use_hybrid_moe = getattr(args, 'use_hybrid_moe', False)

    if moe_model_flag:
        slice_expert_num = args.slices

        # the packed path (WxA16) requires micro-expert slices to be multiples
        # of group_size (128): the gf layout and the down-slice kernels group
        # by 128, and a non-aligned slice (the basic unit of a bit bucket)
        # makes the bucket's total neuron count non-aligned, which breaks
        # convert_to_group_first's reshape. When slices does not divide inter
        # into 128-multiples, adjust it to inter // 128 automatically.
        if use_hybrid_moe and (getattr(args, 'wxa16', False) or bool(getattr(args, 'save_quantized', None))):
            if hasattr(model.config, 'moe_intermediate_size') and model.config.moe_intermediate_size is not None:
                moe_inter = model.config.moe_intermediate_size
            else:
                moe_inter = model.config.intermediate_size
            if moe_inter % GROUPSIZE != 0:
                assert False, (f"the packed path requires moe_intermediate_size "
                               f"({moe_inter}) to be a multiple of group_size ({GROUPSIZE})")
            aligned = (moe_inter % slice_expert_num == 0
                       and (moe_inter // slice_expert_num) % GROUPSIZE == 0)
            if not aligned:
                new_slices = moe_inter // GROUPSIZE
                print(f"packed path: slices {slice_expert_num} does not align slices "
                      f"to {GROUPSIZE}; auto-adjusting to {new_slices} "
                      f"(moe_intermediate {moe_inter})", flush=True)
                slice_expert_num = new_slices

        if hasattr(model.config, 'num_experts'):
            ori_num_experts = model.config.num_experts
            if use_hybrid_moe:
                # Hybrid MoE keeps original expert count at first level
                new_num_expert = model.config.num_experts
            else:
                new_num_expert = slice_expert_num * model.config.num_experts
                model.config.num_experts = new_num_expert
        elif hasattr(model.config, 'n_routed_experts'):
            ori_num_experts = model.config.n_routed_experts
            if use_hybrid_moe:
                new_num_expert = model.config.n_routed_experts
            else:
                new_num_expert = slice_expert_num * model.config.n_routed_experts
                model.config.n_routed_experts = new_num_expert

        ori_num_experts_per_tok = model.config.num_experts_per_tok
        if use_hybrid_moe:
            # Hybrid MoE keeps original activation count
            new_num_experts_per_tok = model.config.num_experts_per_tok
        else:
            new_num_experts_per_tok = slice_expert_num * model.config.num_experts_per_tok
            model.config.num_experts_per_tok = new_num_experts_per_tok

        # For hybrid MoE, we don't change intermediate_size since sub-experts have different sizes
        if not use_hybrid_moe:
            if hasattr(model.config, 'moe_intermediate_size'):
                model.config.moe_intermediate_size = model.config.moe_intermediate_size // slice_expert_num
            elif hasattr(model.config, 'intermediate_size'):
                model.config.intermediate_size = model.config.intermediate_size // slice_expert_num

        if use_hybrid_moe:
            print("The model is already a MoE model. Proceeding to create hybrid MoE structure.")
            print(f"Hybrid MoE: {ori_num_experts} experts with sub-experts sliced by {slice_expert_num}")
        else:
            print("The model is already a MoE model. Proceeding to split experts. ")
            print(f"Slice expert by {slice_expert_num}: to {new_num_expert}, with {new_num_experts_per_tok} activated experts.")
    else:
        assert False, "Dense model is not supported."

    inps = inps.squeeze(1)

    # For CPU standby mode, we want all layers on CPU initially
    if args.standby_layer_cpu:
        layers_device = []
        for layer_idx, layer in enumerate(model.model.layers):
            # Check if layer has parameters
            params = list(layer.parameters())
            if params:
                dev = params[0].device
            else:
                dev = torch.device('cpu')
            layers_device.append(dev)
            # print(layer_idx, dev)
            if dev.type == 'cuda':
                layer = layer.to('cpu')
        for i in range(torch.cuda.device_count()):
            force_release_inactive_splits(device=i)
            print(f"CUDA {i} Allocated: {torch.cuda.memory_allocated(device=i) / 1024**3:.2f} GB")
            print(f"CUDA {i} Reserved: {torch.cuda.memory_reserved(device=i) / 1024**3:.2f} GB")
        # print(layers_device)
    
    qscheme_str = args.quant_scheme
    qscheme = {}
    # qscheme['attn'] = [8]
    # qscheme['share'] = [4]
    # qscheme['expert'] = [2, 2, 2, 2, 2, 2, 2, 2]
    try:
        # sample: "a8s4m3221", "a8s4m33222222"
        match = re.search(r'a(\d)s(\d)m([\d.]+)', qscheme_str)
        aa = match.group(1)
        ss = match.group(2)
        ee = match.group(3)
        qscheme['attn'] = [int(aa)]
        qscheme['share'] = [int(ss)]
        if 'bpw' not in qscheme_str:
            assert len(ee) == slice_expert_num, f"Quant scheme {qscheme_str} should have {slice_expert_num} parts for expert quantization config."
            qscheme['econfig'] = [int(e) for e in ee]
            bpw = sum(qscheme['econfig']) * 1.0 / slice_expert_num
        else:
            bpw = float(ee)
            qscheme['target_bpw'] = bpw
        print(f"Quant expert scheme (ppl): {qscheme_str} with bpw {bpw} qscheme {qscheme}")
    except:
        assert False, f"Quant scheme {qscheme_str} is not valid."

    for layer_idx, layer in enumerate(layers):
        tick0 = time.time()
        if args.standby_layer_cpu:
            # Use the saved original device or default to DEV
            target_dev = layers_device[layer_idx] if layer_idx < len(layers_device) and layers_device[layer_idx].type == 'cuda' else DEV
            layer = layer.to(target_dev)

        moe_out = construct_moe(model,
            moe_model_flag,
            layer,
            layer_idx,
            inps,
            attention_mask,
            position_ids,
            position_embeddings,
            n_experts = new_num_expert,
            n_activated = new_num_experts_per_tok,
            slice_expert_num = slice_expert_num,
            ori_activated = ori_num_experts_per_tok,
            qscheme = qscheme,
            args = args
        )

        inps = moe_out

        if args.standby_layer_cpu:
            layer = layer.to('cpu')
            # packed path: drop runtime caches that hold tensors on the old
            # device (gf layout etc.), so stale GPU tensors do not stay
            # resident after layer.to('cpu')
            for sub in layer.modules():
                if hasattr(sub, 'drop_runtime_caches'):
                    sub.drop_runtime_caches()

        for i in range(torch.cuda.device_count()):
            # force_release_inactive_splits(device=i) # force to release inactive reserved memory
            print(f"CUDA {i} Allocated: {torch.cuda.memory_allocated(device=i) / 1024**3:.2f} GB")
            print(f"CUDA {i} Reserved: {torch.cuda.memory_reserved(device=i) / 1024**3:.2f} GB")

        tick1 = time.time()
        print(f"Layer {layer_idx} total reconstruct and quantization time: {tick1 - tick0:.2f} s", flush=True)
        print("." * 100, flush=True)

    print("MoE reconstruction and quantization done.")
    tick_quant_end = time.time()
    time_quant = tick_quant_end - tick_quant_start
    print(f"Runtime of quantization only: {time_quant:.2f}")

    # for name, param in model.named_parameters():
    #     print(f"{name:<40} → {param.device}")

    # print('Training_free_ppl:')
    # PPL 验证抽为 run_wiki_c4_ppl（见文件末尾）：run_dartmoq 主流程改为
    # 量化 → save → eval，eval 阶段崩溃不再丢失量化成果
    if test_ppl:
        run_wiki_c4_ppl(model, tokenizer, args)

    return model


def run_wiki_c4_ppl(model, tokenizer, args):
    """wikitext2/c4 PPL 验证（原 dartmoq_sequential 尾部 eval 块抽出）。

    run_dartmoq 主流程的调用顺序为 量化 → save → 本函数：eval 崩溃时
    checkpoint 已落盘（save_quantized_model 只做 CPU 拷贝，不改模型状态，
    save 后再 eval 结果不受影响）。
    """
    # args.sequential_eval defaults to False, not bound to args.standby_layer_cpu
    if getattr(args, 'sequential_eval', False):
        print("Will use sequential PPL evaluation (layers stay on CPU)")
    else:
        print("Will use normal PPL evaluation")

    tick_ppl_start = time.time()
    pre_ppl = []
    datasets = ['wikitext2', 'c4']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, tokenizer=tokenizer, seqlen=model.seqlen
        )
        print(dataset)
        eval_set = dataset
        ppl_i = cmoe_ppl_eval(model, testloader, eval_set, args)
        pre_ppl.append(f"{dataset}: {ppl_i}")
    tick_ppl_end = time.time()
    print(f"Runtime of wiki/c4 validation: {tick_ppl_end - tick_ppl_start:.2f}")
    return pre_ppl