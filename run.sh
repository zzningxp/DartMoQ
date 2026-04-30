#!/bin/sh

# MODEL_PATH=""
#       https://huggingface.co/allenai/OLMoE-1B-7B-0924

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_OFFLINE=1 
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2_divisions:4"

# python run_dartmoq.py ~/models/OLMoE-1B-7B-0924-Instruct/ wikitext2 --slices 1 --nsamples 64 --quant-scheme a8s8m8
# python run_dartmoq.py ~/models/OLMoE-1B-7B-0924-Instruct/ wikitext2 --slices 2 --nsamples 64 --quant-scheme a8s8m84
# python run_dartmoq.py ~/models/OLMoE-1B-7B-0924-Instruct/ wikitext2 --slices 2 --nsamples 64 --quant-scheme a8s8m42

# python run_dartmoq.py ~/models/OLMoE-1B-7B-0924-Instruct/ wikitext2 --slices 4 --nsamples 64 --quant-scheme a8s8m4222

# python run_dartmoq.py ~/models/DeepSeek-V2-Lite/ wikitext2 --slices 1 --nsamples 64 --quant-scheme a8s8m8
# python run_dartmoq.py ~/models/DeepSeek-V2-Lite/ wikitext2 --slices 1 --nsamples 64 --quant-scheme a8s4m4
# python run_dartmoq.py ~/models/DeepSeek-V2-Lite/ wikitext2 --slices 2 --nsamples 64 --quant-scheme a8s8m84
# python run_dartmoq.py ~/models/DeepSeek-V2-Lite/ wikitext2 --slices 4 --nsamples 64 --quant-scheme a8s8m4222

# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 4 --nsamples 64 --quant-scheme a8s8m4220
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 4 --nsamples 64 --quant-scheme a8s8m3220
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --quant-scheme a8s8m42222220 --standby-layer-cpu
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --quant-scheme a8s8m32222220 --standby-layer-cpu

# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --quant-scheme a8s8m42222200 --standby-layer-cpu
# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --quant-scheme a8s8m32222200 --standby-layer-cpu

# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --quant-scheme a8s8m32222221 --standby-layer-cpu

python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --quant-scheme global-a8s8m33322222 --standby-layer-cpu
python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --quant-scheme a8s8m33322222 --standby-layer-cpu

python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --quant-scheme global-a8s8m43322222 --standby-layer-cpu
python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --quant-scheme a8s8m43322222 --standby-layer-cpu

python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --quant-scheme global-a8s8m44422221 --standby-layer-cpu
python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --quant-scheme a8s8m44422221 --standby-layer-cpu

python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --quant-scheme global-a8s8m44444222 --standby-layer-cpu
python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --quant-scheme a8s8m44444222 --standby-layer-cpu

# python run_dartmoq.py ~/models/deepseek-moe-16b-base/ wikitext2 --slices 8 --nsamples 64 --quant-scheme global-a8s8m820 --standby-layer-cpu

# python run_dartmoq.py ~/models/OLMoE-1B-7B-0924-Instruct/ wikitext2 --slices 1 --nsamples 64 --quant-scheme a8s8m8 --standby-layer-cpu

# python run_dartmoq.py ~/models/Qwen3-30B-A3B/ wikitext2 --slices 1 --nsamples 64 --quant-scheme a8s8m8 --standby-layer-cpu
# python run_dartmoq.py ~/models/Qwen3-30B-A3B/ wikitext2 --slices 2 --nsamples 64 --quant-scheme a8s4m42 --standby-layer-cpu
# python run_dartmoq.py ~/models/Qwen3-30B-A3B/ wikitext2 --slices 4 --nsamples 64 --quant-scheme a8s4m4222 --standby-layer-cpu
# python run_dartmoq.py ~/models/Qwen3-30B-A3B/ wikitext2 --slices 4 --nsamples 64 --quant-scheme global --standby-layer-cpu
# python run_dartmoq.py ~/models/Qwen3-30B-A3B/ wikitext2 --slices 8 --nsamples 64 --quant-scheme global --standby-layer-cpu
# python run_dartmoq.py ~/models/Qwen3-30B-A3B/ wikitext2 --slices 16 --nsamples 64 --quant-scheme global --standby-layer-cpu