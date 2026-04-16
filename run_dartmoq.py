import time

import torch
import torch.nn as nn

from tqdm import *

import os 

import copy

from dartmoq_utils import *
from dartmoq_sequential import *
from sft_utils import simple_sft
from eval_dartmoq import cmoe_ppl_eval, load_model

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
    parser.add_argument(        '--extra-lr',
        type=float, default=0.001, 
        help='Initial learning rate for extra scale for router.'
    )
    parser.add_argument(        '--k-act', type=int, default=10,
        help='TopK number for the ATopK. K_a in paper.'
    )
    parser.add_argument(        '--bias-speed',
        type=float, default=0.001, 
        help='Bias update speed for load balancing. Gamma in paper.'
    )
    parser.add_argument(        '--nexperts', type=int, default=16,
        help='Total number of experts. N in paper.'
    )
    parser.add_argument(        '--nactivated', type=int, default=2,
        help='Number of activated routed experts.'
    )
    parser.add_argument(        '--nshared', type=int, default=2,
        help='Number of shared experts.'
    )
    parser.add_argument(        '--epoch', type=int, default=0,
        help='SFT epoch for CMoE.'
    )
    parser.add_argument(        '--sft-bsz', type=int, default=1,
        help='SFT batch size for CMoE.'
    )
    parser.add_argument(        '--carve-bsz', type=int, default=1,
        help='Carve batch size for CMoE.'
    )
    parser.add_argument(        '--eval-zero', action='store_true',
        help='Whether to run downstream tasks evaluation.'
    )
    parser.add_argument(        '--prefix', type=str, default=None,
        help='Prefix the results folder if needed.'
    )
    parser.add_argument(        '--quant-scheme', 
        type=str, default=None,
        help='Quantization scheme like fix_scheme like a8s4m3221 or global scheme like global.'
    )
    parser.add_argument(        '--rank-mode', 
        type=str, default="quant_outlier",
        help='Rank mode for MoE reconstruction. activation|quant_outlier|random|neuron_index'
    )
    parser.add_argument(        '--reconstruct_start_layer', type=int, default=0,
        help='Start layer for reconstruction.'
    )
    parser.add_argument(        '--reconstruct_end_layer', type=int, default=15,
        help='End layer for reconstruction.'
    )

    args = parser.parse_args()
    
    print("-" * 50)
    print("Loading model: ", args.model)
    print("quant-scheme/rank-mode: (ppl)", args.quant_scheme, args.rank_mode)
    model, tokenizer = load_model(args.model)

    dataloader, testloader = get_loaders(
        args.dataset, 
        nsamples=args.nsamples, 
        seed=args.seed, 
        tokenizer=tokenizer, 
        seqlen=model.seqlen, 
        bsz = args.carve_bsz
    )

    print("number of data: ", args.nsamples)
    print("model: ", args.model)
    print("cali_data: ", args.dataset)

    tick = time.time()
    # ori_ppl = cmoe_ppl_eval(model, testloader, args.dataset, args)
    # print(f"Original model ppl on {args.dataset}: {ori_ppl}")

    carved_model = cmoe_sequential(model, tokenizer, dataloader, args)
    save_carved_model = False
    if save_carved_model:
        carved_save_dir = f"model/carved_{model.config.model_type}_e{args.nexperts}a{args.nactivated}_{args.quant_scheme}"
        print(carved_model)
        carved_model.save_pretrained(carved_save_dir)
        tokenizer.save_pretrained(carved_save_dir)
    
    # print(carved_model)

    tick1 = time.time()

    sft_flag = args.epoch > 0
    if sft_flag:
        print('Starting SFT...')    

        carved_model.cuda()
        carved_model = simple_sft(carved_model, tokenizer, args, epoch = args.epoch)

        carved_model.eval()

        print('SFT_ppl:')
        ppl = []
        datasets = ['wikitext2', 'c4-new']
        for dataset in datasets:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, tokenizer=tokenizer, seqlen=carved_model.seqlen, bsz = args.carve_bsz
            )
            print(dataset)
            eval_set = dataset
            ppl_i = cmoe_ppl_eval(carved_model, testloader, eval_set, args)
            ppl.append(f"{dataset}: {ppl_i}")
        
        print("SFT_ppl: ", ppl)

    rt = time.time() - tick1
    print(f"Runtime of training-free construction (ppl): {tick1 - tick:.2f}")
    print(f"Runtime of fine-tuning construction: {rt:.2f}")
