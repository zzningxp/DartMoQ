#!/bin/sh
# 五个模型 × {1bpw, 2bpw} 量化并保存 checkpoint（quant_ckpt/<tag>-<n>bpw）。
# 口径与现有 quant_ckpt 一致（meta.json 实测）：
#   turboquant_innerproduct + turboquant + slices 8 + nsamples 64 + seed 42
# 保存后可直接用 bench_throughput.py / graph_decode.py --quant-dir 加载测速。
#
# 注意：quant_ckpt/ 下同名目录已存在，重复跑会覆盖重写。

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2_divisions:4"

# run_save <模型路径> <ckpt 标签> <量化方案>
run_save() {
    model="$1"; tag="$2"; scheme="$3"
    slices=1
    out="quant_ckpt/${tag}-${scheme#global-bpw-a8s8m}bpw-s${slices}"
    echo "###: [$tag] $scheme -> $out"
    python run_dartmoq.py "$model" wikitext2 \
        --slices "$slices" --nsamples 64 \
        --rank-mode turboquant_innerproduct \
        --quant-scheme "$scheme" \
        --quantmode turboquant \
        --standby-layer-cpu \
        --save-quantized "$out"
}

# 模型根目录（原 ~/models 下的五个模型已整体迁到 /mnt/new1t/models）
MODELS_ROOT="/mnt/new1t/models"

# ---- DeepSeek-V2-Lite (dsv2, 16B) ----
run_save "$MODELS_ROOT/DeepSeek-V2-Lite/" dsv2 global-bpw-a8s8m1
run_save "$MODELS_ROOT/DeepSeek-V2-Lite/" dsv2 global-bpw-a8s8m2
# run_save "$MODELS_ROOT/DeepSeek-V2-Lite/" dsv2 global-bpw-a4s4m1
# run_save "$MODELS_ROOT/DeepSeek-V2-Lite/" dsv2 global-bpw-a4s4m2

# ---- deepseek-moe-16b-base (dsv1, 16B) ----
run_save "$MODELS_ROOT/deepseek-moe-16b-base/" dsv1 global-bpw-a8s8m1
run_save "$MODELS_ROOT/deepseek-moe-16b-base/" dsv1 global-bpw-a8s8m2
# run_save "$MODELS_ROOT/deepseek-moe-16b-base/" dsv1 global-bpw-a4s4m1
# run_save "$MODELS_ROOT/deepseek-moe-16b-base/" dsv1 global-bpw-a4s4m2

# ---- Moonlight-16B-A3B (moon) ----
run_save "$MODELS_ROOT/Moonlight-16B-A3B/" moon global-bpw-a8s8m1
run_save "$MODELS_ROOT/Moonlight-16B-A3B/" moon global-bpw-a8s8m2
# run_save "$MODELS_ROOT/Moonlight-16B-A3B/" moon global-bpw-a4s4m1
# run_save "$MODELS_ROOT/Moonlight-16B-A3B/" moon global-bpw-a4s4m2

# ---- OLMoE-1B-7B-0924-Instruct (olmoe) ----
run_save "$MODELS_ROOT/OLMoE-1B-7B-0924-Instruct/" olmoe global-bpw-a8s8m1
run_save "$MODELS_ROOT/OLMoE-1B-7B-0924-Instruct/" olmoe global-bpw-a8s8m2
# run_save "$MODELS_ROOT/OLMoE-1B-7B-0924-Instruct/" olmoe global-bpw-a4s4m1
# run_save "$MODELS_ROOT/OLMoE-1B-7B-0924-Instruct/" olmoe global-bpw-a4s4m2

# ---- Qwen3-30B-A3B (qwen3-30b-a3b, 30B) ----
run_save "$MODELS_ROOT/Qwen3-30B-A3B" qwen3-30b-a3b global-bpw-a8s8m1
run_save "$MODELS_ROOT/Qwen3-30B-A3B" qwen3-30b-a3b global-bpw-a8s8m2
# run_save "$MODELS_ROOT/Qwen3-30B-A3B" qwen3-30b-a3b global-bpw-a4s4m1
# run_save "$MODELS_ROOT/Qwen3-30B-A3B" qwen3-30b-a3b global-bpw-a4s4m2
