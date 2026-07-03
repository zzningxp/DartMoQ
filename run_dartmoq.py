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
from dartmoq_io import save_dartmoq_model, load_dartmoq_model

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
        type=int, default=42, help='Seed for sampling the calibration data.'
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
        type=str, default=None,
        help='Rank mode for MoE reconstruction. Available modes:\n' \
        '  - expert_activation: Rank neurons by activation frequency in input samples\n' \
        '  - energy: Rank neurons by energy (from CAMERA) to the output\n' \
        '  - random: Random ordering for baseline testing\n' \
        '  - neuron_index: Original neuron index order\n' \
        '  - gptq_quant_outlier: Rank by GPTQ quantization loss, identifying error-sensitive neurons\n' \
        '  - turboquant_iipl: TurboQuant outlier analysis with IIPL (Input-Intermediate Product Loss)\n' \
        '  - turboquant_innerproduct: TurboQuant outlier analysis using inner product mode (**RECOMMENED**)\n' \
        '  - turboquant_diagonal: TurboQuant outlier analysis with diagonal Hessian approximation\n' \
        '  - turboquant_hessian: TurboQuant outlier analysis with full Hessian computation\n' \
        '  - turboquant_qjl_sensitivity: TurboQuant analysis with quantized Johnson-Lindenstrauss sensitivity\n' \
        '  - turboquant_iipl_fea: TurboQuant IIPL (Input-Intermediate Product Loss) mode with full experts activation (not recommended)\n' \
        '  - turboquant_innerproduct_fea: TurboQuant inner product mode with full experts activation (not recommended)\n'
    )
    parser.add_argument(
        '--disable-0bit-compensation',
        action='store_true',
        default=False,
        help='Disable 0bit compensation: 0bit weights incur quantization overhead'
    )
    parser.add_argument(
        '--disable-0bit-prune',
        action='store_true',
        default=False,
        help='Disable 0bit in DP search: only use 1-4 bits for bit allocation'
    )
    parser.add_argument(        '--standby-layer-cpu', action='store_true', default=False,
        help='Whether to move quant layers to CPU before and after quantization.'
    )
    parser.add_argument(        '--sequential-eval', action='store_true', default=False,
        help='Use sequential PPL evaluation (keeps layers on CPU, moves one by one).'
    )
    parser.add_argument(        '--no-use-hybrid-moe', dest='use_hybrid_moe', action='store_false', default=True,
        help='Disable hybrid MoE structure and use original experts instead.'
    )
    parser.add_argument(        '--quantmode', type=str, default='turboquant', choices=['gptq', 'turboquant'],
        help='Quantization mode: gptq (default) or turboquant.'
    )
    parser.add_argument(        '--save-model', action='store_true', default=False,
        help='Whether to save the model to disk.'
    )

    args = parser.parse_args()
    
    print("-" * 50)
    print(f"Current start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    print("Loading model: (ppl)", args.model)
    print("slices/quant-scheme/rank-mode/moe-struct/quantmode/disable-0bit-prune/standby-layer-cpu: (ppl)",
          args.slices, args.quant_scheme, args.rank_mode, "use_hybrid_moe" if args.use_hybrid_moe else " use_origin_moe", args.quantmode, args.disable_0bit_prune, args.standby_layer_cpu)
    model, tokenizer = load_model(args.model, standby_cpu=args.standby_layer_cpu)

    dataloader, _ = get_loaders(
        args.dataset, 
        nsamples=args.nsamples, 
        seed=args.seed, 
        tokenizer=tokenizer, 
        seqlen=model.seqlen
    )

    print("number of data: ", args.nsamples)
    print("model: ", args.model, model.model_id)
    print("cali_data: ", args.dataset)

    tick = time.time()

    with torch.no_grad():
        dartmoq_model = dartmoq_sequential(model, tokenizer, dataloader, args) #, test_ppl=False)

    if args.save_model:
        save_dir = f"models/dartmoq_{model.config.model_type}_{args.rank_mode}_{args.quant_scheme}"
        print("###: Save dartmoq model to: ", save_dir)
        save_dartmoq_model(dartmoq_model, tokenizer, save_dir, args)

    time_zero_eval = 0.0
    if args.eval_zero and not args.standby_layer_cpu:
        tick_zero_start = time.time()
        task_list = ["arc_challenge", "arc_easy", "piqa", "boolq", "winogrande", "mnli", "hellaswag", "mmlu"]
        # task_list = ["mnli", "hellaswag", "mmlu", "sciq"]
        # task_list = ["gsm8k", "triviaqa"]
        eval_zero_shot(dartmoq_model, task_list, tokenizer=tokenizer)
        tick_zero_done = time.time()
        time_zero_eval = tick_zero_done - tick_zero_start
        print(f"Runtime of zero-shot evaluation: {time_zero_eval:.2f}")
    elif args.eval_zero and args.standby_layer_cpu:
        print("Skipping zero-shot evaluation because standby_layer_cpu is enabled (standby mode takes precedence)")

    # print(model)

    tick1 = time.time()

    print(f"Runtime of training-free construction (ppl): {tick1 - tick:.2f}")
    print(f"Current finish time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    time.sleep(120)
