#!/bin/sh

# MODEL_PATH=""
#       https://huggingface.co/allenai/OLMoE-1B-7B-0924

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_OFFLINE=1 
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2_divisions:4"

# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1.5 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1.625 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1.75 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1.875 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2.125 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2.25 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2.375 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2.5 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2.625 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2.75 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2.875 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m3 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m3.125 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m3.25 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m3.375 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m3.5 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m3.625 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m3.75 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m3.875 --use-hybrid-moe --quantmode turboquant

python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_outlier_aw --quant-scheme global-bpw-a8s8m0.5 --use-hybrid-moe --quantmode turboquant
python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_outlier_aw --quant-scheme global-bpw-a8s8m0.625 --use-hybrid-moe --quantmode turboquant
python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_outlier_aw --quant-scheme global-bpw-a8s8m1.5 --use-hybrid-moe --quantmode turboquant
python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_outlier_aw --quant-scheme global-bpw-a8s8m1.625 --use-hybrid-moe --quantmode turboquant
python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_outlier_aw --quant-scheme global-bpw-a8s8m2.5 --use-hybrid-moe --quantmode turboquant
python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_outlier_aw --quant-scheme global-bpw-a8s8m2.625 --use-hybrid-moe --quantmode turboquant
python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_outlier_aw --quant-scheme global-bpw-a8s8m3.5 --use-hybrid-moe --quantmode turboquant
python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_outlier_aw --quant-scheme global-bpw-a8s8m3.625 --use-hybrid-moe --quantmode turboquant

python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_outlier_output --quant-scheme global-bpw-a8s8m0.5 --use-hybrid-moe --quantmode turboquant
python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_outlier_output --quant-scheme global-bpw-a8s8m0.625 --use-hybrid-moe --quantmode turboquant
python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_outlier_output --quant-scheme global-bpw-a8s8m1.5 --use-hybrid-moe --quantmode turboquant
python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_outlier_output --quant-scheme global-bpw-a8s8m1.625 --use-hybrid-moe --quantmode turboquant
python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_outlier_output --quant-scheme global-bpw-a8s8m2.5 --use-hybrid-moe --quantmode turboquant
python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_outlier_output --quant-scheme global-bpw-a8s8m2.625 --use-hybrid-moe --quantmode turboquant
python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_outlier_output --quant-scheme global-bpw-a8s8m3.5 --use-hybrid-moe --quantmode turboquant
python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_outlier_output --quant-scheme global-bpw-a8s8m3.625 --use-hybrid-moe --quantmode turboquant

# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme global-bpw-a8s8m1.5 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme global-bpw-a8s8m1.625 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme global-bpw-a8s8m1.75 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme global-bpw-a8s8m1.875 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme global-bpw-a8s8m2 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme global-bpw-a8s8m2.125 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme global-bpw-a8s8m2.25 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme global-bpw-a8s8m2.375 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme global-bpw-a8s8m2.5 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme global-bpw-a8s8m2.625 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme global-bpw-a8s8m2.75 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme global-bpw-a8s8m2.875 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme global-bpw-a8s8m3 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme global-bpw-a8s8m3.125 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme global-bpw-a8s8m3.25 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme global-bpw-a8s8m3.375 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme global-bpw-a8s8m3.5 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme global-bpw-a8s8m3.625 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme global-bpw-a8s8m3.75 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme global-bpw-a8s8m3.875 --use-hybrid-moe --quantmode gptq


# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme bpw-a8s8m1.75 --use-hybrid-moe
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme bpw-a8s8m1.5 --use-hybrid-moe

# python run_dartmoq.py ~/models/Moonlight-16B-A3B/ wikitext2 --slices 1 --nsamples 64 --rank-mode quant_outlier --quant-scheme a8s8m3 --use-hybrid-moe
# python run_dartmoq.py ~/models/Moonlight-16B-A3B/ wikitext2 --slices 1 --nsamples 64 --rank-mode quant_outlier --quant-scheme a8s8m4 --use-hybrid-moe

# python run_dartmoq.py ~/models/Moonlight-16B-A3B/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme bpw-a8s8m1.5 --use-hybrid-moe
# python run_dartmoq.py ~/models/Moonlight-16B-A3B/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme bpw-a8s8m1.625 --use-hybrid-moe
# python run_dartmoq.py ~/models/Moonlight-16B-A3B/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme bpw-a8s8m1.75 --use-hybrid-moe
# python run_dartmoq.py ~/models/Moonlight-16B-A3B/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme bpw-a8s8m1.875 --use-hybrid-moe
# python run_dartmoq.py ~/models/Moonlight-16B-A3B/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme bpw-a8s8m2 --use-hybrid-moe
# python run_dartmoq.py ~/models/Moonlight-16B-A3B/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme bpw-a8s8m2.125 --use-hybrid-moe
# python run_dartmoq.py ~/models/Moonlight-16B-A3B/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme bpw-a8s8m2.25 --use-hybrid-moe
# python run_dartmoq.py ~/models/Moonlight-16B-A3B/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme bpw-a8s8m2.375 --use-hybrid-moe
# python run_dartmoq.py ~/models/Moonlight-16B-A3B/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme bpw-a8s8m2.5 --use-hybrid-moe
# python run_dartmoq.py ~/models/Moonlight-16B-A3B/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme bpw-a8s8m2.625 --use-hybrid-moe
# python run_dartmoq.py ~/models/Moonlight-16B-A3B/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme bpw-a8s8m2.75 --use-hybrid-moe
# python run_dartmoq.py ~/models/Moonlight-16B-A3B/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme bpw-a8s8m2.875 --use-hybrid-moe
# python run_dartmoq.py ~/models/Moonlight-16B-A3B/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme bpw-a8s8m3 --use-hybrid-moe
# python run_dartmoq.py ~/models/Moonlight-16B-A3B/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme bpw-a8s8m3.125 --use-hybrid-moe
# python run_dartmoq.py ~/models/Moonlight-16B-A3B/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme bpw-a8s8m3.25 --use-hybrid-moe
# python run_dartmoq.py ~/models/Moonlight-16B-A3B/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme bpw-a8s8m3.375 --use-hybrid-moe
# python run_dartmoq.py ~/models/Moonlight-16B-A3B/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme bpw-a8s8m3.5 --use-hybrid-moe
# python run_dartmoq.py ~/models/Moonlight-16B-A3B/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme bpw-a8s8m3.625 --use-hybrid-moe
# python run_dartmoq.py ~/models/Moonlight-16B-A3B/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme bpw-a8s8m3.75 --use-hybrid-moe
# python run_dartmoq.py ~/models/Moonlight-16B-A3B/ wikitext2 --slices 8 --nsamples 64 --rank-mode quant_outlier --quant-scheme bpw-a8s8m3.875 --use-hybrid-moe
