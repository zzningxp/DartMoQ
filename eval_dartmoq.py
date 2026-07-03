from enum import auto
from os import name
import time

import torch
import torch.nn as nn

from dartmoq_utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer

@torch.no_grad()
def cmoe_ppl_eval_sequential(model, testloader, eval_set, args):
    """
    Sequential PPL evaluation: keeps layers on CPU and moves them to GPU one by one.
    Processes samples in batches to save memory.
    Used for models too large to fit entirely in GPU memory.
    """
    tick0 = time.time()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    testenc = testloader.input_ids
    nsamples = testenc.shape[1] // model.seqlen
    print('ppl evaluation samples (sequential mode, batch parallel):', nsamples)

    # Save original device for each layer
    layers = model.model.layers
    original_devices = []
    for layer in layers:
        if hasattr(layer, 'parameters') and len(list(layer.parameters())) > 0:
            original_devices.append(next(layer.parameters()).device)
        else:
            original_devices.append(torch.device('cpu'))

    # Move all layers to CPU first, but keep embed_tokens, norm, lm_head on GPU!
    print("Moving transformer layers to CPU for sequential evaluation...")
    for layer in layers:
        layer.to('cpu')

    # Keep embed_tokens, norm, lm_head on GPU permanently
    model.model.embed_tokens = model.model.embed_tokens.to(DEV)
    if hasattr(model.model, 'norm'):
        model.model.norm = model.model.norm.to(DEV)
    if hasattr(model, 'lm_head'):
        model.lm_head = model.lm_head.to(DEV)

    # Force cleanup
    import gc
    gc.collect()
    for i in range(torch.cuda.device_count()):
        torch.cuda.empty_cache()
        print(f"CUDA {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")

    # First, capture correct attention_mask, position_ids, position_embeddings using Catcher
    # Temporarily move layer 0 to DEV for capturing
    layers[0] = layers[0].to(DEV)

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            self.captured_kwargs = kwargs
            raise ValueError
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

    layers[0] = Catcher(layers[0])

    # Get first batch to capture the kwargs
    first_batch = testenc[:, :model.seqlen].to(DEV)
    try:
        model(first_batch)
    except ValueError:
        pass

    # Get the captured kwargs
    captured_kwargs = layers[0].captured_kwargs
    attention_mask = captured_kwargs.get('attention_mask')
    position_ids = captured_kwargs.get('position_ids')
    position_embeddings = captured_kwargs.get('position_embeddings')

    # Restore layer 0
    layers[0] = layers[0].module
    layers[0] = layers[0].to('cpu')

    torch.cuda.empty_cache()

    # Get all target ids first - keep on GPU!
    all_target_ids = testenc[:, :nsamples * model.seqlen].clone().to(DEV)

    # Batch sizes - use 64 for transformer layers, 4 for lm_head
    batch_size_transformer = 64
    batch_size_lm_head = 4

    # Precompute embeddings for all samples and keep on CPU to save GPU memory!
    print("Processing embeddings and caching on CPU...")
    all_embeddings = []
    for sample_idx in range(nsamples):
        batch = testenc[:, (sample_idx * model.seqlen):((sample_idx + 1) * model.seqlen)].to(DEV)
        with torch.no_grad():
            hidden = model.model.embed_tokens(batch)
        all_embeddings.append(hidden.cpu())  # Move to CPU immediately
        del batch, hidden
    gc.collect()
    torch.cuda.empty_cache()

    # If attention_mask exists, expand it for batch_size_transformer
    batch_attention_mask_64 = None
    if attention_mask is not None:
        if attention_mask.dim() == 4:
            batch_attention_mask_64 = attention_mask.repeat(batch_size_transformer, 1, 1, 1)
        elif attention_mask.dim() == 3:
            batch_attention_mask_64 = attention_mask.repeat(batch_size_transformer, 1, 1)
        elif attention_mask.dim() == 2:
            batch_attention_mask_64 = attention_mask.repeat(batch_size_transformer, 1)

    batch_position_ids_64 = None
    if position_ids is not None:
        batch_position_ids_64 = position_ids.repeat(batch_size_transformer, 1)

    batch_position_embeddings_64 = None
    if position_embeddings is not None:
        if isinstance(position_embeddings, tuple):
            batch_position_embeddings_64 = tuple(pe.repeat(batch_size_transformer, 1, 1) if pe is not None else None for pe in position_embeddings)
        else:
            batch_position_embeddings_64 = position_embeddings.repeat(batch_size_transformer, 1, 1)

    import inspect
    # Process each transformer layer sequentially, with batches of 64
    all_hidden_states = [emb for emb in all_embeddings]  # Start with embeddings

    for layer_idx, layer in enumerate(layers):
        if layer_idx % 5 == 0:
            print(f"Processing layer {layer_idx}/{len(layers)}...", flush=True)

        # Move layer to GPU
        layer = layer.to(DEV)

        # Process samples in batches of 64
        new_hidden_states = []
        for batch_start in range(0, nsamples, batch_size_transformer):
            batch_end = min(batch_start + batch_size_transformer, nsamples)
            actual_batch_size = batch_end - batch_start

            # Get this batch's hidden states and move to GPU
            batch_hidden = torch.cat(all_hidden_states[batch_start:batch_end], dim=0).to(DEV)

            # Prepare kwargs for this batch size
            layer_kwargs = {}
            forward_signature = inspect.signature(layer.forward)

            if 'attention_mask' in forward_signature.parameters and batch_attention_mask_64 is not None:
                if actual_batch_size != batch_size_transformer:
                    # Need to resize for last partial batch
                    if attention_mask.dim() == 4:
                        layer_kwargs['attention_mask'] = attention_mask.repeat(actual_batch_size, 1, 1, 1)
                    elif attention_mask.dim() == 3:
                        layer_kwargs['attention_mask'] = attention_mask.repeat(actual_batch_size, 1, 1)
                    elif attention_mask.dim() == 2:
                        layer_kwargs['attention_mask'] = attention_mask.repeat(actual_batch_size, 1)
                else:
                    layer_kwargs['attention_mask'] = batch_attention_mask_64

            if 'position_ids' in forward_signature.parameters and batch_position_ids_64 is not None:
                if actual_batch_size != batch_size_transformer:
                    layer_kwargs['position_ids'] = position_ids.repeat(actual_batch_size, 1)
                else:
                    layer_kwargs['position_ids'] = batch_position_ids_64

            if 'position_embeddings' in forward_signature.parameters and batch_position_embeddings_64 is not None:
                if actual_batch_size != batch_size_transformer and position_embeddings is not None:
                    if isinstance(position_embeddings, tuple):
                        layer_kwargs['position_embeddings'] = tuple(pe.repeat(actual_batch_size, 1, 1) if pe is not None else None for pe in position_embeddings)
                    else:
                        layer_kwargs['position_embeddings'] = position_embeddings.repeat(actual_batch_size, 1, 1)
                else:
                    layer_kwargs['position_embeddings'] = batch_position_embeddings_64

            # Forward pass
            with torch.no_grad():
                layer_outputs = layer(batch_hidden, **layer_kwargs)

            # Save output
            if isinstance(layer_outputs, tuple):
                batch_output = layer_outputs[0]
            else:
                batch_output = layer_outputs

            # Split batch into individual samples and move to CPU
            for i in range(actual_batch_size):
                new_hidden_states.append(batch_output[i:i+1].cpu())

            # Cleanup
            del batch_hidden, layer_outputs, batch_output
            gc.collect()
            torch.cuda.empty_cache()

        # Update hidden states for next layer
        all_hidden_states = new_hidden_states

        # Move layer back to CPU
        layer = layer.to('cpu')

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

        if layer_idx % 5 == 0:
            for i in range(torch.cuda.device_count()):
                print(f"  Layer {layer_idx}, CUDA {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")

    # Final norm and lm_head - process in batches of 4
    print("Processing final norm and lm_head...")
    nlls = []

    for batch_start in range(0, nsamples, batch_size_lm_head):
        batch_end = min(batch_start + batch_size_lm_head, nsamples)
        actual_batch_size = batch_end - batch_start

        # Get this batch's hidden states and move to GPU
        batch_hidden = torch.cat(all_hidden_states[batch_start:batch_end], dim=0).to(DEV)

        # Process this batch through norm and lm_head
        with torch.no_grad():
            batch_hidden = model.model.norm(batch_hidden)
            batch_logits = model.lm_head(batch_hidden)

            # Get labels for this batch - already on DEV
            batch_labels = all_target_ids[:, batch_start * model.seqlen : batch_end * model.seqlen]
            batch_labels = batch_labels.reshape(actual_batch_size, model.seqlen)

            # Calculate loss for this batch
            shift_logits = batch_logits[:, :-1, :].contiguous()
            shift_labels = batch_labels[:, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.reshape(actual_batch_size, model.seqlen - 1)
            neg_log_likelihood = loss.float().sum(dim=1)
            nlls.extend(list(neg_log_likelihood))

        # Clean up
        del batch_hidden, batch_logits, batch_labels
        gc.collect()
        torch.cuda.empty_cache()

    # Restore original devices
    print("Restoring layers to original devices...")
    for layer_idx, layer in enumerate(layers):
        layer.to(original_devices[layer_idx])

    # Calculate final ppl
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    tick1 = time.time()
    print(f'ppl on {eval_set} (sequential mode): {ppl.item():.4f} time: {tick1 - tick0:.2f}')
    model.config.use_cache = use_cache

    return ppl.item()


@torch.no_grad()
def cmoe_ppl_eval(model, testloader, eval_set, args):
    # Check if we should use sequential mode
    use_sequential = getattr(args, 'sequential_eval', False)
    use_standby_cpu = getattr(args, 'standby_layer_cpu', False)

    # If standby_layer_cpu is True, force sequential_eval to avoid device mismatch issues
    if use_standby_cpu and not use_sequential:
        print("Warning: standby_layer_cpu enabled, forcing sequential_eval for stability.")
        use_sequential = True

    if use_sequential:
        return cmoe_ppl_eval_sequential(model, testloader, eval_set, args)

    # Normal PPL evaluation - no manual distribution needed
    tick0 = time.time()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    testenc = testloader.input_ids
    # print("testenc.shape: ", testenc.shape)
    nsamples = testenc.shape[1] // model.seqlen
    # nsamples = 64
    print('ppl evaluation samples:', nsamples)

    def get_activation():
        def hook(model, input, output):
            isnan = torch.isnan(output)
            whereisnan = torch.where(isnan)
            if whereisnan[1].shape[0] > 0:
                # output[whereisnan] = 0.0
                print(whereisnan[1][0])
        return hook

    hooks = []
    hook_handles = []
    # print(model, model.config)
    # print(hasattr(model.config, 'num_experts'))
    # if hasattr(model.config, 'num_experts'): ## OLmoe
    #     for i in range(model.config.num_experts):
    #         hooks.append(model.model.layers[0].mlp.experts[i].up_proj)
    #         hooks.append(model.model.layers[0].mlp.experts[i].gate_proj)
    # if hasattr(model.config, 'n_routed_experts'): ## Deepseek-v3 / Moonlight
    #     for i in range(model.config.n_routed_experts):
    #         # for j
    #         hooks.append(model.model.layers[1].mlp.experts[i].up_proj)
    #         hooks.append(model.model.layers[1].mlp.experts[i].gate_proj)
    # hooks.append(model.model.layers[0].self_attn.kv_a_proj_with_mqa)
    # hooks.append(model.model.layers[0].self_attn.kv_b_proj)
    # hooks.append(model.model.layers[0].self_attn.q_proj)
    # hooks.append(model.model.layers[0].self_attn.o_proj)
    # hooks.append(model.model.layers[0].mlp)
    #
    # print(model)
    nlls = []

    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(DEV)
        target_ids = batch.clone()

        for hook in hooks:
            hook_handles.append(hook.register_forward_hook(get_activation()))

        with torch.no_grad():
            outputs = model(batch)
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = target_ids[:, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * model.seqlen
            nlls.append(neg_log_likelihood)

        for hook in hooks:
            hook_handles.pop().remove()

    # print(nlls)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    tick1 = time.time()
    print(f'ppl on {eval_set}: {ppl.item():.4f} time: {tick1 - tick0:.2f}')
    model.config.use_cache = use_cache

    return ppl.item()


def distribute_model_to_gpus(model):
    """
    按照分配策略把模型层分配到多个 GPU：
    - 前 1/N 层 → cuda:0
    - 后 1/N 层 → cuda:n-1
    - 中间层 → 中间的 GPU
    - embed_tokens → cuda:0
    - norm 和 lm_head → 最后一个 GPU
    """
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No CUDA devices found!")
        return model

    print(f"Distributing model across {num_gpus} GPUs...")

    if not hasattr(model.model, 'layers'):
        print("No model.model.layers found, moving whole model to cuda:0")
        return model.cuda()

    layers = model.model.layers
    num_layers = len(layers)
    print(f"Total layers: {num_layers}")

    layers_per_gpu = num_layers // num_gpus
    remainder = num_layers % num_gpus

    device_map = {}
    current_layer = 0

    for gpu_idx in range(num_gpus):
        num_layers_this_gpu = layers_per_gpu + (1 if gpu_idx < remainder else 0)
        for i in range(num_layers_this_gpu):
            device_map[current_layer] = f'cuda:{gpu_idx}'
            current_layer += 1

    if hasattr(model.model, 'embed_tokens'):
        model.model.embed_tokens = model.model.embed_tokens.to('cuda:0')
        print(f"embed_tokens → cuda:0")

    for layer_idx in range(num_layers):
        target_device = device_map[layer_idx]
        layers[layer_idx] = _move_to_device(layers[layer_idx], target_device)
        if layer_idx % max(1, num_layers // 10) == 0:  # 打印大约 10 条日志
            print(f"  layer {layer_idx}/{num_layers} → {target_device}")

    # 3. 分配 norm 和 lm_head 到最后一个 GPU
    last_device = f'cuda:{num_gpus - 1}'
    if hasattr(model.model, 'norm'):
        model.model.norm = _move_to_device(model.model.norm, last_device)
        print(f"norm → {last_device}")
    if hasattr(model, 'lm_head'):
        model.lm_head = _move_to_device(model.lm_head, last_device)
        print(f"lm_head → {last_device}")

    # 打印最终的 device map 摘要
    print("\n=== Final Device Map ===")
    from collections import defaultdict
    layer_count_by_gpu = defaultdict(int)
    for layer_idx, device in device_map.items():
        layer_count_by_gpu[device] += 1

    for gpu_idx in range(num_gpus):
        device = f'cuda:{gpu_idx}'
        count = layer_count_by_gpu.get(device, 0)
        if count > 0:
            print(f"{device}: {count} layers")

    # 强制同步和清理
    import gc
    gc.collect()
    for gpu_idx in range(num_gpus):
        torch.cuda.empty_cache()
        print(f"CUDA {gpu_idx}: {torch.cuda.memory_allocated(gpu_idx) / 1024**3:.2f} GB allocated")

    return model


def _move_to_device(module, target_device):
    """
    递归地将模块及其所有子模块移动到目标设备，并确保所有参数和缓冲区都被移动。
    同时验证移动是否成功。
    """
    module = module.to(target_device)

    for param in module.parameters(recurse=True):
        if param.device != torch.device(target_device):
            param.data = param.data.to(target_device)

    for buf in module.buffers(recurse=True):
        if buf is not None and buf.device != torch.device(target_device):
            buf.data = buf.data.to(target_device)

    for sub_module in module.modules():
        if sub_module is module:
            continue 
        has_params = any(True for _ in sub_module.parameters())
        if has_params:
            sub_module.to(target_device)
            for p in sub_module.parameters():
                if p.device != torch.device(target_device):
                    p.data = p.data.to(target_device)

    return module


def eval_zero_shot(model, task_list, tokenizer=None):
    tick0 = time.time()
    from lm_eval import tasks, evaluator, utils
    import tqdm
    from functools import partial
    evaluator.tqdm = partial(
        tqdm.tqdm,
        mininterval=5.0,
    )

    use_cache_original = model.config.use_cache
    model.config.use_cache = False

    has_cpu_params = False
    try:
        for _, param in model.named_parameters():
            if param.device.type == 'cpu':
                has_cpu_params = True
                break
    except:
        pass

    if has_cpu_params:
        print("Model is on CPU, distributing to GPUs for zero-shot evaluation...")
        model = distribute_model_to_gpus(model)
        print("Model distributed successfully.\n")
    else:
        devices = set()
        for _, param in model.named_parameters():
            devices.add(str(param.device))
        if len(devices) > 1:
            print(f"Model already distributed across devices: {sorted(devices)}")

    tick0 = time.time()
    for task in task_list:
        # for eval_batch_size in [16, 8, 4, 2, 1]:
        # for eval_batch_size in [8, 4, 2, 1]:
        for eval_batch_size in [4, 2, 1]:
            try:
            # Only support hf method now
                print(f"Evaluating {task} with batch size {eval_batch_size}")
                from lm_eval.models.huggingface import HFLM

                eval_model = HFLM(
                    pretrained=model,
                    trust_remote_code=True,
                    # 不要指定 device="cuda"，这会强制将所有内容移动到 cuda:0
                    batch_size=eval_batch_size,
                )
                eval_model.model.config.use_cache = False

                # 验证 eval_model 没有破坏设备分配
                eval_devices = set()
                for _, param in eval_model.model.named_parameters():
                    eval_devices.add(str(param.device))
                if len(eval_devices) > 1:
                    print(f"  Model remains distributed across: {sorted(eval_devices)}")

                tick_task = time.time()
                results = evaluator.simple_evaluate(
                    model=eval_model,
                    tasks=[task],
                    num_fewshot=5,
                    batch_size="auto",
                    # 不要指定 device，让 evaluator 使用模型已有的设备
                )
                tick1 = time.time()

                print(task, results["results"][task], f"time: {tick1 - tick_task}s")
                break
            except Exception as e:
                if eval_batch_size > 1:
                    print(f"Error evaluating {task} with batch size {eval_batch_size}: {e}")
                    pass
                elif eval_batch_size == 1:
                    assert False, f"Error evaluating {task} with batch size {eval_batch_size}, {e}"

    tick1 = time.time()
    print(f"Zero-shot evaluation time: {tick1 - tick0}")

    # Restore original use_cache setting
    model.config.use_cache = use_cache_original

def get_llama(model, device_map="auto"):
    def skip(*args, **kwargs): pass
    # torch.nn.init.kaiming_uniform_ = skip
    # torch.nn.init.uniform_ = skip
    # torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map
    )
    model.seqlen = 2048
    # model.seqlen = 4096
    return model

def get_llava(model, device_map="auto"):
    def skip(*args, **kwargs): pass
    # torch.nn.init.kaiming_uniform_ = skip
    # torch.nn.init.uniform_ = skip
    # torch.nn.init.normal_ = skip

    from llava.model import LlavaLlamaForCausalLM

    model = LlavaLlamaForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map
    )
    model.seqlen = 2048
    # model.seqlen = 4096

    return model

def get_olmoe(model_path, device_map="auto"):
    from transformers import OlmoeForCausalLM

    # model = OlmoeForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map = 'auto')
    # print(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map
    )

    model.seqlen = 2048
    # model.seqlen = 4096
    return model

def get_deepseek_moe_16b(model_path, device_map="auto"):
    # from transformers import DeepseekForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        trust_remote_code=True
    )

    model.seqlen = 2048
    # model.seqlen = 4096

    return model, tokenizer

def get_deepseek_v2_lite(model_path, device_map="auto"):

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        trust_remote_code=True
    )

    model.seqlen = 2048
    # model.seqlen = 4096

    return model, tokenizer

def get_qwen3_moe(model_path, device_map="auto"):
    from transformers import Qwen3MoeForCausalLM

    model = Qwen3MoeForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        trust_remote_code=True
    )

    model.seqlen = 2048
    # model.seqlen = 4096

    return model

def get_qwen3_30b_a3b(model_path, device_map="auto"):
    from transformers import Qwen3MoeForCausalLM
    model = Qwen3MoeForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        trust_remote_code=True
    )

    model.seqlen = 2048
    # model.seqlen = 4096

    return model

def get_qwen3(model_path, device_map="auto"):
    from transformers import Qwen3ForCausalLM

    model = Qwen3ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        trust_remote_code=True
    )

    model.seqlen = 2048
    # model.seqlen = 4096

    return model

def get_moonlight(model_path, device_map="auto"):
    from transformers import DeepseekV3ForCausalLM

    model = DeepseekV3ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        trust_remote_code=True
    )

    model.seqlen = 2048
    # model.seqlen = 4096

    return model

def get_auto(model_path, device_map="auto"):

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
        use_safetensors=True,
        trust_remote_code=True
    )

    model.seqlen = 2048
    # model.seqlen = 4096

    return model, tokenizer


def load_model(model_path, standby_cpu=False):
    """
    Load model with optional CPU standby mode for very large models.

    Args:
        model_path: Path to model
        standby_cpu: If True, loads model to CPU first for memory efficiency
    """
    print(model_path.lower())

    # For CPU standby mode, first check if model is on meta device and needs reload
    # If standby_cpu is True, we'll load to CPU directly
    device_map = "cpu" if standby_cpu else "auto"

    # Set model_id
    model_id = str(model_path).split('/')[-1].split('\\')[-1]

    # Try to infer model_id from path
    path_lower = model_path.lower()
    if 'llava' in path_lower:
        model_id = 'llava'
    elif 'olmoe' in path_lower:
        model_id = 'olmoe-7b-1b'
    elif "deepseek-moe-16b" in path_lower:
        model_id = 'deepseek-v1-moe-16b'
    elif 'deepseek-v2-lite' in path_lower:
        model_id = 'deepseek-v2-lite'
    elif 'llama-2-7b' in path_lower:
        model_id = 'llama2-7b'
    elif 'llama-2-13b' in path_lower:
        model_id = 'llama2-13b'
    elif 'llama-3-8b' in path_lower:
        model_id = 'llama3-8b'
    elif 'llama-3___1-8b' in path_lower:
        model_id = 'llama31-8b'
    elif 'qwen3-30b-a3b' in path_lower:
        model_id = 'qwen3-30b-a3b'
    elif 'qwen3_moe' in path_lower:
        model_id = 'qwen3-moe'
    elif 'qwen3' in path_lower:
        if 'qwen3-4b' in path_lower:
            model_id = 'qwen3-4b'
        elif 'qwen3-8b' in path_lower:
            model_id = 'qwen3-8b'
        else:
            model_id = 'qwen3'
    elif 'qwen2.5' in path_lower or 'qwen2___5' in path_lower:
        model_id = 'qwen2.5'
    elif 'moonlight' in path_lower:
        model_id = 'moonlight'

    # Original loading path for non-lazy mode
    if 'llava' in model_path.lower():
        model = get_llava(model_path, device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif 'olmoe' in model_path.lower():
        model = get_olmoe(model_path, device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif "deepseek-moe-16b" in model_path.lower():
        model, tokenizer = get_deepseek_moe_16b(model_path, device_map=device_map)
    elif 'deepseek-v2-lite' in model_path.lower():
        model, tokenizer = get_deepseek_v2_lite(model_path, device_map=device_map)
    elif 'llama' in model_path.lower():
        model = get_llama(model_path, device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif 'qwen3-30b-a3b' in model_path.lower():
        model = get_qwen3_30b_a3b(model_path, device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif 'qwen3_moe' in model_path.lower():
        model = get_qwen3_moe(model_path, device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif 'qwen3' in model_path.lower():
        model = get_qwen3(model_path, device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif 'qwen2.5' in model_path.lower() or 'qwen2___5' in model_path.lower():
        model, tokenizer = get_auto(model_path, device_map=device_map)
        print(model_path.lower(), model_id)
    elif 'moonlight' in model_path.lower():
        model = get_moonlight(model_path, device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(model_path.lower(), model_id)
    else:
        model, tokenizer = get_auto(model_path, device_map=device_map)

    model.eval()
    model.model_id = model_id
    if not model.model_id:
        model.model_id = getattr(model.config, '_name_or_path', None) or getattr(model.config, 'name_or_path', None) or model_path
        model.model_id = str(model.model_id).split('/')[-1].split('\\')[-1]

    # Mark if we're in CPU standby mode
    model._standby_cpu = standby_cpu
    model._model_path = model_path

    return model, tokenizer

if __name__ == '__main__':
    import argparse
    from data_utils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(        'model', type=str,
        help='Model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(        '--eval-zero',
        action='store_true', help='Evaluate zero-shot performance.'
    )
    parser.add_argument(        '--val-samples',
        type=int, default=256, help='Evaluate performance on x samples.'
    )
    parser.add_argument(        '--sequential-eval', action='store_true', default=False,
        help='Use sequential PPL evaluation (keeps layers on CPU, moves one by one).'
    )
    parser.add_argument(        '--standby-cpu', action='store_true', default=False,
        help='Use CPU standby mode (load model to CPU first for large models).'
    )


    args = parser.parse_args()

    print("", args.model)

    if not args.eval_zero:
        print("Loading model: ", args.model.lower())
        model, tokenizer = load_model(args.model, standby_cpu=args.standby_cpu)

        # If in CPU standby mode, make sure everything is on CPU
        if args.standby_cpu:
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        # for name, param in model.named_parameters():
        #     print(f"{name:<40} → {param.device}")

        print("model: ", args.model)
        # print(model)
        # print(model.config)
        ppl = []
        datasets = ['wikitext2', 'c4']
        for dataset in datasets:
            dataloader, testloader = get_loaders(
                dataset, nsamples=args.val_samples, seed=args.seed, tokenizer=tokenizer, seqlen=model.seqlen
            )

            print(dataset)
            eval_set = dataset
            ppl_i = cmoe_ppl_eval(model, testloader, eval_set, args)
            ppl.append(f"{dataset}: {ppl_i}")

    if args.eval_zero:
        print("Loading model: ", args.model.lower())
        model, tokenizer = load_model(args.model, standby_cpu=args.standby_cpu)

        task_list = ["arc_challenge", "arc_easy", "piqa", "boolq", "winogrande", "sciq", "mnli", "hellaswag", "gsm8k", "mmlu", "triviaqa"]
        # task_list = ["arc_challenge", "arc_easy", "boolq", "winogrande", "piqa", "sciq", "hellaswag", "mmlu", "gsm8k", "triviaqa"]
        # task_list = ["mnli"]
        eval_zero_shot(model, task_list, tokenizer=tokenizer)
