from enum import auto
from os import name
import time

import torch
import torch.nn as nn

from dartmoq_utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer

@torch.no_grad()
def cmoe_ppl_eval(model, testloader, eval_set, args):
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

def eval_zero_shot(model, task_list = ["arc_challenge", "arc_easy", "piqa", "boolq", "winogrande"]):
    tick0 = time.time()
    from lm_eval import tasks, evaluator, utils
    from lm_eval.models.huggingface import HFLM
    model = HFLM(
        pretrained=model,
        trust_remote_code=True,
        device="cuda",
    )

    for task in task_list:
        tick0 = time.time()
        results = evaluator.simple_evaluate(
            model=model,
            tasks=[task],
            num_fewshot=5,
            batch_size="auto",
            device="cuda"
        )
        tick1 = time.time()

        print(task, results["results"][task], f"time: {tick1 - tick0}s") 
    
    tick1 = time.time()
    print(f"Zero-shot evaluation time: {tick1 - tick0}")

def get_llama(model):
    def skip(*args, **kwargs):
        pass
    # torch.nn.init.kaiming_uniform_ = skip
    # torch.nn.init.uniform_ = skip
    # torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(
        model, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        device_map = "auto"
    )
    model.seqlen = 2048
    # model.seqlen = 4096
    return model

def get_llava(model):
    def skip(*args, **kwargs):
        pass
    # torch.nn.init.kaiming_uniform_ = skip
    # torch.nn.init.uniform_ = skip
    # torch.nn.init.normal_ = skip

    from llava.model import LlavaLlamaForCausalLM

    model = LlavaLlamaForCausalLM.from_pretrained(
        model, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        device_map = "auto"
    )
    model.seqlen = 2048
    # model.seqlen = 4096

    return model

def get_olmoe(model_path):
    from transformers import OlmoeForCausalLM

    # model = OlmoeForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map = 'auto')
    # print(model_path)
    device_map = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        device_map = device_map
    )

    model.seqlen = 2048
    # model.seqlen = 4096
    return model

def get_deepseek_moe_16b(model_path):
    # from transformers import DeepseekForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto"
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

def get_deepseek_v2_lite(model_path):

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto"
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
def get_qwen3_moe(model_path):
    from transformers import Qwen3MoeForCausalLM

    device_map = "auto"
    model = Qwen3MoeForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map = device_map,
        trust_remote_code=True
    )

    model.seqlen = 2048
    # model.seqlen = 4096

    return model

def get_qwen3_30b_a3b(model_path):
    from transformers import Qwen3MoeForCausalLM
    # device_map = {
    #             "model.embed_tokens": "cuda:0",
    #             "model.rotary_emb": "cuda:0",
    #             **{
    #                 f"model.layers.{k}": "cuda:0" for k in range(0, 16)
    #             },
    #             **{
    #                 f"model.layers.{k}": "cuda:1" for k in range(16, 32)
    #             },
    #             **{
    #                 f"model.layers.{k}": "cpu" for k in range(32, 48)
    #             },
    #             "model.norm": "cpu",
    #             "lm_head": "cpu",
    #         }
    # print(device_map)
    device_map = 'auto'
    model = Qwen3MoeForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map = device_map,
        trust_remote_code=True
    )

    model.seqlen = 2048
    # model.seqlen = 4096

    return model

def get_qwen3(model_path):
    from transformers import Qwen3ForCausalLM

    device_map = "auto"
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

def get_moonlight(model_path):
    from transformers import DeepseekV3ForCausalLM

    device_map = "auto"
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

def get_auto(model_path):

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    device_map = "auto"
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

def load_model(model_path):
    if 'llava' in model_path.lower():
        model = get_llava(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif 'olmoe' in model_path.lower():
        model = get_olmoe(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif "deepseek-moe-16b" in model_path.lower():
        model, tokenizer = get_deepseek_moe_16b(model_path)
    elif 'deepseek-v2-lite' in model_path.lower():
        model, tokenizer = get_deepseek_v2_lite(model_path)
    elif 'llama' in model_path.lower():
        model = get_llama(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif 'qwen3-30b-a3b' in model_path.lower():
        model = get_qwen3_30b_a3b(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif 'qwen3_moe' in model_path.lower():
        model = get_qwen3_moe(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif 'qwen3' in model_path.lower():
        model = get_qwen3(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    elif 'moonlight' in model_path.lower():
        model = get_moonlight(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        assert False, "Model type not supported."
    model.eval()
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


    args = parser.parse_args()
    
    print("", args.model)

    if not args.eval_zero:
        print("Loading model: ", args.model.lower())
        model, tokenizer = load_model(args.model)

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

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        task_list = ["arc_challenge", "arc_easy", "piqa", "boolq", "winogrande", "sciq", "mnli", "hellaswag", "gsm8k", "mmlu", "triviaqa"]
        # task_list = ["arc_challenge", "arc_easy", "boolq", "winogrande", "piqa", "sciq", "hellaswag", "mmlu", "gsm8k", "triviaqa"]
        # task_list = ["mnli"]
        eval_zero_shot(model, task_list)