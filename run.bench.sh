#!/bin/sh

# 推理速度数据采集（rebuttal 评审要求口径），用已存的 quant_ckpt/*-s1
# 十个 checkpoint（5 模型 x {1bpw, 2bpw}），不重新量化。
#
# 每格报（bench_throughput.py）：
#   - checkpoint GB（按类别分解：MoE packed / attn+shared packed / fp16 残余）
#   - 含 scales/codebook/norms/mask 等全部 metadata 的 effective bpw
#   - 分相位峰值推理显存（after_load/warmup/prefill/decode 各档 + overall peak）
#   - prefill/decode 延迟与 tok/s 的 batch x seq 网格（B={1,4} x L={16,128,2048}）
#
# 用法：bash run.bench.sh；断点续跑 marker 在 logs/bench_done/
#   （<模型>_<bpw>_<mode>）。FRESH=1 bash run.bench.sh 先清空全部 marker
#   从头跑完 20 格（完整重测到新 log 用这个）。GRAPH=1 bash run.bench.sh
#   额外跑 graph_decode.py（wxa8 CUDA graph decode，作为加速 decode 的
#   补充数字；失败不影响 bench 主结果）。
#
# 完整重跑示例（约 4-6h，每格 load+warmup+网格 约 10-20 min）：
#   FRESH=1 nohup bash run.bench.sh > logs/1102.2.full.bench.log 2>&1 &
#   （完成后 logs_parser.py 解析该 log 即得 20 格全表）

. /home/daodao/anaconda3/etc/profile.d/conda.sh
conda activate dart312

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2_divisions:4"

# <tag>|<quant-dir>|<extra flags>（tag 仅用于 marker/日志）
# dsv1(deepseek-moe-16b) 的 model_type "deepseek" 无 5.20 原生映射，必须
# --trust-remote-code 走 remote modeling；其 4.37 时代代码与 transformers
# 5.20 的三处不兼容（is_torch_fx_available import / rope_scaling 物化 /
# DynamicCache legacy 互转 API）已由 load_quantized_model 的 shim 补齐，
# 1102 bench GPU 实跑验证通过。dsv2/qwen3/olmoe 原生 modeling 不加 flag。
# moon(Moonlight) 1102 改走原生：model_type deepseek_v3 在 5.20 有原生
# DeepseekV3ForCausalLM，与 dsv2 同理（remote modeling 与 5.20 Cache API
# 不兼容），auto_map 在 trust_remote_code=False 时让位原生（config 探测
# DeepseekV3Config 解析 OK，module 命名与保存 sd key 同源）。
CKPTS="
dsv2-1bpw|quant_ckpt/dsv2-1bpw-s1|
dsv2-2bpw|quant_ckpt/dsv2-2bpw-s1|
qwen3-1bpw|quant_ckpt/qwen3-30b-a3b-1bpw-s1|
qwen3-2bpw|quant_ckpt/qwen3-30b-a3b-2bpw-s1|
olmoe-1bpw|quant_ckpt/olmoe-1bpw-s1|
olmoe-2bpw|quant_ckpt/olmoe-2bpw-s1|
dsv1-1bpw|quant_ckpt/dsv1-1bpw-s1|--trust-remote-code
dsv1-2bpw|quant_ckpt/dsv1-2bpw-s1|--trust-remote-code
moon-1bpw|quant_ckpt/moon-1bpw-s1|
moon-2bpw|quant_ckpt/moon-2bpw-s1|
"

MODES="wxa16 wxa8"
PREFILL_LENS="16,128,2048"
BATCH_SIZES="1,4"
DECODE_TOKENS="128,256"
GRAPH=${GRAPH:-0}

mkdir -p logs/bench_done
FRESH=${FRESH:-0}
if [ "$FRESH" = "1" ]; then
    echo "=== FRESH=1: clearing all bench_done markers (full 20-cell rerun) ==="
    rm -f logs/bench_done/*
fi
total_fail=0
# 连续失败熔断：单格失败继续下一格，但连败 5 格视为系统性 bug（加载/批量
# 路径崩）直接停，避免每格空烧一次模型加载；marker 保留进度，修好后重启
# 脚本续跑。1102 三轮修复后 20 格已全 PASS，此熔断为防未来改动的保险。
fail_streak=0

for entry in $CKPTS; do
    tag=${entry%%|*}
    rest=${entry#*|}
    qdir=${rest%%|*}
    extra=${rest#*|}
    for MODE in $MODES; do
        mtag="${tag}_${MODE}"
        if [ -f "logs/bench_done/${mtag}" ]; then
            echo "=== skip (marker exists): ${mtag} ==="
            continue
        fi
        echo "==============================================================="
        echo "bench: ckpt=${tag} mode=${MODE}"
        echo "start: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "==============================================================="
        if python bench_throughput.py --quant-dir "$qdir" --mode $MODE \
            --prefill-lens $PREFILL_LENS --batch-sizes $BATCH_SIZES \
            --decode-tokens $DECODE_TOKENS $extra; then
            touch "logs/bench_done/${mtag}"
            fail_streak=0
        else
            total_fail=$((total_fail + 1))
            fail_streak=$((fail_streak + 1))
            echo "!!! FAILED: ${mtag} (consecutive: ${fail_streak})"
            if [ "$fail_streak" -ge 5 ]; then
                echo "!!! 5 consecutive failures - likely systematic bench bug."
                echo "!!! Aborting (markers preserve progress; fix and relaunch)."
                exit 1
            fi
        fi
        if [ "$GRAPH" = "1" ] && [ "$MODE" = "wxa8" ]; then
            echo "--- graph_decode (aux) ${tag} ---"
            python graph_decode.py --quant-dir "$qdir" --mode wxa8 \
                || echo "!!! graph_decode FAILED: ${tag} (aux only)"
        fi
    done
done

echo "bench done. total_fail=${total_fail}"
