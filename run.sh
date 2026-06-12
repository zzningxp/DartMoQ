#!/bin/sh

# MODEL_PATH=""
#       https://huggingface.co/allenai/OLMoE-1B-7B-0924

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_OFFLINE=1 
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2_divisions:4"

modelname="$HOME/models/deepseek-moe-16b-base/"

python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m0.5 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m0.625 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m0.75 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m0.875 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.125 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.25 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.375 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.5 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.625 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.75 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.875 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.125 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.25 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.375 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.5 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.625 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.75 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.875 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m3 --quantmode turboquant

modelname="$HOME/models/OLMoE-1B-7B-0924-Instruct/"

python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m0.5 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m0.625 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m0.75 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m0.875 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.125 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.25 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.375 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.5 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.625 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.75 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.875 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.125 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.25 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.375 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.5 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.625 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.75 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.875 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m3 --quantmode turboquant

modelname="$HOME/models/DeepSeek-V2-Lite/"

python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m0.5 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m0.625 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m0.75 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m0.875 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.125 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.25 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.375 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.5 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.625 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.75 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.875 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.125 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.25 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.375 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.5 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.625 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.75 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.875 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m3 --quantmode turboquant

modelname="$HOME/models/Moonlight-16B-A3B/"

python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m0.5 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m0.625 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m0.75 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m0.875 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.125 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.25 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.375 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.5 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.625 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.75 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.875 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.125 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.25 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.375 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.5 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.625 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.75 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.875 --quantmode turboquant
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m3 --quantmode turboquant


# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1.125 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1.25 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1.375 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1.5 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1.625 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1.75 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1.875 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2.125 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2.25 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2.375 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2.5 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2.625 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2.75 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2.875 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m3 --quantmode turboquant

# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.5 --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.625 --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.75 --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.875 --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m1 --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m1.125 --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m1.25 --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m1.375 --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m1.5 --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m1.625 --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m1.75 --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m1.875 --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m2 --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m2.125 --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m2.25 --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m2.375 --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m2.5 --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m2.625 --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m2.75 --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m2.875 --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m3 --quantmode gptq

# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m0.5 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m0.625 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m0.75 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m0.875 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m1 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m1.125 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m1.25 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m1.375 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m1.5 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m1.625 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m1.75 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m1.875 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m2 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m2.125 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m2.25 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m2.375 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m2.5 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m2.625 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m2.75 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m2.875 --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m3 --quantmode turboquant
