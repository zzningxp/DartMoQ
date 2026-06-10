#!/bin/sh

# MODEL_PATH=""
#       https://huggingface.co/allenai/OLMoE-1B-7B-0924

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_OFFLINE=1 
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2_divisions:4"

modelname="$HOME/models/deepseek-moe-16b-base/"
# modelname="$HOME/models/OLMoE-1B-7B-0924-Instruct/"
# modelname="$HOME/models/DeepSeek-V2-Lite/"

python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m2 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m2 --quantmode gptq --eval-zero

python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.5 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1.5 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m1.5 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m1.5 --quantmode gptq --eval-zero

python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m1 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m1 --quantmode gptq --eval-zero

modelname="$HOME/models/OLMoE-1B-7B-0924-Instruct/"

python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m2 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m2 --quantmode gptq --eval-zero

python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.5 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1.5 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m1.5 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m1.5 --quantmode gptq --eval-zero

python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m1 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m1 --quantmode gptq --eval-zero

modelname="$HOME/models/DeepSeek-V2-Lite/"

python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m2 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m2 --quantmode gptq --eval-zero

python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.5 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1.5 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m1.5 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m1.5 --quantmode gptq --eval-zero

python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m1 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m1 --quantmode gptq --eval-zero

modelname="$HOME/models/Moonlight-16B-A3B/"

python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m2 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m2 --quantmode gptq --eval-zero

python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.5 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1.5 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m1.5 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m1.5 --quantmode gptq --eval-zero

python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m1 --quantmode turboquant --eval-zero
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m1 --quantmode gptq --eval-zero


# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m0.5 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m0.625 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m0.75 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m0.875 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.125 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.25 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.375 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.5 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.625 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.75 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m1.875 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.125 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.25 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.375 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.5 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.625 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.75 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m2.875 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m3 --use-hybrid-moe --quantmode turboquant

# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1.125 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1.25 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1.375 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1.5 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1.625 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1.75 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m1.875 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2.125 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2.25 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2.375 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2.5 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2.625 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2.75 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m2.875 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode energy --quant-scheme global-bpw-a8s8m3 --use-hybrid-moe --quantmode turboquant

# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.5 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.625 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.75 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.875 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m1 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m1.125 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m1.25 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m1.375 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m1.5 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m1.625 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m1.75 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m1.875 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m2 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m2.125 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m2.25 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m2.375 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m2.5 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m2.625 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m2.75 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m2.875 --use-hybrid-moe --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m3 --use-hybrid-moe --quantmode gptq

# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m0.5 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m0.625 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m0.75 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m0.875 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m1 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m1.125 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m1.25 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m1.375 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m1.5 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m1.625 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m1.75 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m1.875 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m2 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m2.125 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m2.25 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m2.375 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m2.5 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m2.625 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m2.75 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m2.875 --use-hybrid-moe --quantmode turboquant
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_iipl --quant-scheme global-bpw-a8s8m3 --use-hybrid-moe --quantmode turboquant
