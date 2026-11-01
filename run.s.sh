#!/bin/sh

# MODEL_PATH=""
#       https://huggingface.co/allenai/OLMoE-1B-7B-0924

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_OFFLINE=1 
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2_divisions:4"

modelname="$HOME/models/Qwen3-30B-A3B"
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m2 --quantmode gptq --standby-layer-cpu

modelname="$HOME/models/DeepSeek-V2-Lite/"
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m0.75 --quantmode turboquant --standby-layer-cpu
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.75 --quantmode gptq --standby-layer-cpu
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.75 --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.75 --quantmode gptq --standby-layer-cpu --eval-zero
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.75 --quantmode gptq --eval-zero

# modelname="$HOME/models/Moonlight-16B-A3B/"
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m0.75 --quantmode turboquant --standby-layer-cpu
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.75 --quantmode gptq --standby-layer-cpu
# # python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.75 --quantmode gptq
# # python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.75 --quantmode gptq --standby-layer-cpu --eval-zero
# # python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.75 --quantmode gptq --eval-zero

modelname="$HOME/models/OLMoE-1B-7B-0924-Instruct/"
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m0.75 --quantmode turboquant --standby-layer-cpu
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.75 --quantmode gptq --standby-layer-cpu
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.75 --quantmode gptq
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.75 --quantmode gptq --standby-layer-cpu --eval-zero
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.75 --quantmode gptq --eval-zero

# modelname="$HOME/models/deepseek-moe-16b-base/"
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m0.75 --quantmode turboquant --standby-layer-cpu
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.75 --quantmode gptq --standby-layer-cpu
# # python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.75 --quantmode gptq
# # python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.75 --quantmode gptq --standby-layer-cpu --eval-zero
# # python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.75 --quantmode gptq --eval-zero

# modelname="$HOME/models/Qwen3-30B-A3B"
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode turboquant_innerproduct --quant-scheme global-bpw-a8s8m0.75 --quantmode turboquant --standby-layer-cpu
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.75 --quantmode gptq --standby-layer-cpu
# # python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.75 --quantmode gptq --standby-layer-cpu --eval-zero
# # python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode gptq_quant_outlier --quant-scheme global-bpw-a8s8m0.75 --quantmode gptq --eval-zero
