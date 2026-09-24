#!/bin/sh

# 速度对比：五个模型 × 三个推理模式（fp16 原始 / wxa16 / wxa8 量化加载）。
# 量化 checkpoint 由 run.q.sh 生成（quant_ckpt/<name>-${MOE_BPW}bpw，每模型独立子目录）。
# 速度指标取 eval_dartmoq 打印的 wikitext2/c4 eval wall time。
# 三种模式统一走 sequential eval（32 样本批量、逐层搬移，与 3.5 项目口径一致）：
#   - fp16 基线加 --standby-cpu：原模型放不下 GPU，CPU 加载后逐层搬移
#   - wxa16/wxa8 加 --sequential-eval：量化层只有 100-200MB，搬移开销远小于 fp16
#     （常驻 GPU 的普通模式在 32 批量下会因注意力分数矩阵 + 大词表 logits OOM）
#   - 首次 wxa8 会触发 Triton JIT 编译（逐 expert 形状边跑边编译），
#     测速前先用任意一次 wxa8 eval 捂热磁盘缓存，第二次起才是稳态时间

# 主流程必须用 cmoe311（transformers 4.57.5，与 ModuleList 格式 checkpoint 匹配；
# dart312 的 5.x 专家为 3D grouped 格式且删除了旧接口）。脚本内强制激活。
source /home/daodao/anaconda3/etc/profile.d/conda.sh
conda activate cmoe311

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2_divisions:4"

CKPT="quant_ckpt"
MOE_BPW=1

# OLMoE-1B-7B
echo "################ olmoe fp16 ################"
python eval_dartmoq.py "$HOME/models/OLMoE-1B-7B-0924-Instruct/" --standby-cpu
echo "################ olmoe wxa16 ################"
python eval_dartmoq.py --load-quantized "$CKPT/olmoe-${MOE_BPW}bpw" --inference-quant-mode wxa16 --sequential-eval
echo "################ olmoe wxa8 ################"
python eval_dartmoq.py --load-quantized "$CKPT/olmoe-${MOE_BPW}bpw" --inference-quant-mode wxa8 --sequential-eval

# DeepSeekMoE-v1 16B
echo "################ dsv1 fp16 ################"
python eval_dartmoq.py "$HOME/models/deepseek-moe-16b-base/" --standby-cpu
echo "################ dsv1 wxa16 ################"
python eval_dartmoq.py --load-quantized "$CKPT/dsv1-${MOE_BPW}bpw" --inference-quant-mode wxa16 --sequential-eval
echo "################ dsv1 wxa8 ################"
python eval_dartmoq.py --load-quantized "$CKPT/dsv1-${MOE_BPW}bpw" --inference-quant-mode wxa8 --sequential-eval

# DeepSeek-V2-Lite
echo "################ dsv2 fp16 ################"
python eval_dartmoq.py "$HOME/models/DeepSeek-V2-Lite/" --standby-cpu
echo "################ dsv2 wxa16 ################"
python eval_dartmoq.py --load-quantized "$CKPT/dsv2-${MOE_BPW}bpw" --inference-quant-mode wxa16 --sequential-eval
echo "################ dsv2 wxa8 ################"
python eval_dartmoq.py --load-quantized "$CKPT/dsv2-${MOE_BPW}bpw" --inference-quant-mode wxa8 --sequential-eval

# Moonlight-16B-A3B
echo "################ moon fp16 ################"
python eval_dartmoq.py "$HOME/models/Moonlight-16B-A3B/" --standby-cpu
echo "################ moon wxa16 ################"
python eval_dartmoq.py --load-quantized "$CKPT/moon-${MOE_BPW}bpw" --inference-quant-mode wxa16 --sequential-eval
echo "################ moon wxa8 ################"
python eval_dartmoq.py --load-quantized "$CKPT/moon-${MOE_BPW}bpw" --inference-quant-mode wxa8 --sequential-eval

# Qwen3-30B-A3B
echo "################ qwen3-30b-a3b fp16 ################"
python eval_dartmoq.py "$HOME/models/Qwen3-30B-A3B" --standby-cpu
echo "################ qwen3-30b-a3b wxa16 ################"
python eval_dartmoq.py --load-quantized "$CKPT/qwen3-30b-a3b-${MOE_BPW}bpw" --inference-quant-mode wxa16 --sequential-eval
echo "################ qwen3-30b-a3b wxa8 ################"
python eval_dartmoq.py --load-quantized "$CKPT/qwen3-30b-a3b-${MOE_BPW}bpw" --inference-quant-mode wxa8 --sequential-eval
