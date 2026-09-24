#!/bin/sh

# 按新 WxA16 真量化逻辑（--save-quantized）为五个模型各保存一份 1bpw 量化 checkpoint。
# --save-quantized 自动启用 packed 真量化路径（MoE 按 bit 分桶 + attention/shared 8bit，
# 落盘 safetensors + meta.json），量化完成后 run_dartmoq 会继续跑 wikitext2/c4 PPL
# eval 验证 packed 路径数值。加载推理见：
#   python eval_dartmoq.py --load-quantized <dir> --inference-quant-mode wxa16
#   python eval_dartmoq.py --load-quantized <dir> --inference-quant-mode wxa8
# 主流程环境建议 cmoe311（transformers 4.57.5，与 ModuleList 格式 checkpoint 匹配）。

# 主流程必须用 cmoe311（transformers 4.57.5）：dart312 的 transformers 5.x 专家为
# 3D grouped 格式（报 'OlmoeExperts' object is not iterable），且删除了 deepseek
# 远程建模代码依赖的 is_torch_fx_available。脚本内强制激活，避免误用默认环境。
source /home/daodao/anaconda3/etc/profile.d/conda.sh
conda activate cmoe311

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2_divisions:4"

# MoE 专家目标 bpw 可配置（0/1/2/4 bit 混合 DP 分配），attention 8bit 均匀码本，shared 8bit
# 用法：MOE_BPW=1.5 sh run.q.sh（支持小数，如 1.5/2），默认 1bpw
MOE_BPW=1
SCHEME="global-bpw-a8s8m${MOE_BPW}"
RANK_MODE="turboquant_innerproduct"

# # OLMoE-1B-7B
# modelname="$HOME/models/OLMoE-1B-7B-0924-Instruct/"
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode $RANK_MODE --quant-scheme $SCHEME --quantmode turboquant --standby-layer-cpu --save-quantized "quant_ckpt/olmoe-${MOE_BPW}bpw"

# # DeepSeekMoE-v1 16B
# modelname="$HOME/models/deepseek-moe-16b-base/"
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode $RANK_MODE --quant-scheme $SCHEME --quantmode turboquant --standby-layer-cpu --save-quantized "quant_ckpt/dsv1-${MOE_BPW}bpw"

# DeepSeek-V2-Lite
modelname="$HOME/models/DeepSeek-V2-Lite/"
python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode $RANK_MODE --quant-scheme $SCHEME --quantmode turboquant --standby-layer-cpu --save-quantized "quant_ckpt/dsv2-${MOE_BPW}bpw"

# # Moonlight-16B-A3B
# modelname="$HOME/models/Moonlight-16B-A3B/"
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode $RANK_MODE --quant-scheme $SCHEME --quantmode turboquant --standby-layer-cpu --save-quantized "quant_ckpt/moon-${MOE_BPW}bpw"

# # Qwen3-30B-A3B
# modelname="$HOME/models/Qwen3-30B-A3B"
# python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 --rank-mode $RANK_MODE --quant-scheme $SCHEME --quantmode turboquant --standby-layer-cpu --save-quantized "quant_ckpt/qwen3-30b-a3b-${MOE_BPW}bpw"
