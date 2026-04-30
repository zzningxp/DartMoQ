import time

import torch
import torch.nn as nn

from tqdm import *

import os 

import copy

from dartmoq_utils import *
from dartmoq_sequential import *
# from sft_utils import simple_sft
from eval_dartmoq import eval_zero_shot, load_model

def save_results(file_name, results):
    if results is not str:
        results = str(results)
    results = results + '\n'
    if not os.path.exists(file_name):
        with open(file_name, "w") as file:
            file.write(results)
    else:
        with open(file_name, "a") as file:
            file.write(results)


if __name__ == '__main__':
    import argparse
    from data_utils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(        'model', type=str,
        help='Model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(        '--nsamples', type=int, default=128,
        help='Number of Fine-tuning data for CMoE.'
    )
    parser.add_argument(        '--slices', type=int, default=1,
        help='Number of sub experts to slice.'
    )
    parser.add_argument(        '--eval-zero', action='store_true',
        help='Whether to run downstream tasks evaluation.'
    )
    parser.add_argument(        '--quant-scheme', 
        type=str, default=None,
        help='Quantization scheme like fix_scheme like a8s4m3221 or global scheme like global.'
    )
    parser.add_argument(        '--rank-mode', 
        type=str, default="quant_outlier",
        help='Rank mode for MoE reconstruction. activation|importance|quant_outlier|random|neuron_index'
    )
    parser.add_argument(        '--standby-layer-cpu', action='store_true', default=False,
        help='Whether to move quant layers to CPU before and after quantization.' 
    )

    args = parser.parse_args()
    
    print("-" * 50)
    print("Loading model: (ppl)", args.model)
    print("slices/quant-scheme/rank-mode: (ppl)", args.slices, args.quant_scheme, args.rank_mode)
    model, tokenizer = load_model(args.model)

    dataloader, _ = get_loaders(
        args.dataset, 
        nsamples=args.nsamples, 
        seed=args.seed, 
        tokenizer=tokenizer, 
        seqlen=model.seqlen
    )

    print("number of data: ", args.nsamples)
    print("model: ", args.model)
    print("cali_data: ", args.dataset)

    tick = time.time()

    with torch.no_grad():
        carved_model = cmoe_sequential(model, tokenizer, dataloader, args)
    save_carved_model = False
    if save_carved_model:
        carved_save_dir = f"model/carved_{model.config.model_type}_e{args.slices}_{args.quant_scheme}"
        print(carved_model)
        carved_model.save_pretrained(carved_save_dir)
        tokenizer.save_pretrained(carved_save_dir)

    if args.eval_zero:
        task_list = ["arc_challenge", "arc_easy", "piqa", "boolq", "winogrande", "sciq", "mnli", "hellaswag", "gsm8k", "mmlu", "triviaqa"]
        # task_list = ["mnli", "gsm8k", "mmlu", "triviaqa"]
        # task_list = ["arc_challenge", "arc_easy"]
        eval_zero_shot(model, task_list)
    
    # print(carved_model)

    tick1 = time.time()

    rt = time.time() - tick1
    print(f"Runtime of training-free construction (ppl): {tick1 - tick:.2f}")
    print(f"Runtime of fine-tuning construction: {rt:.2f}")
