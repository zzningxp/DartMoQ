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
#   （<模型>_<bpw>_<mode>，删 marker 强制重跑）。GRAPH=1 bash run.bench.sh
#   额外跑 graph_decode.py（wxa8 CUDA graph decode，作为加速 decode 的
#   补充数字；失败不影响 bench 主结果）。
#
# 建议与 E4b 补格串行：
#   nohup bash -c 'bash run.e4.sh; bash run.bench.sh' > logs/1031.e4b2_bench.log 2>&1 &
#
# 量级：每格（load+warmup+网格）约 10-20 min，10 ckpt x 2 mode = 20 格
# 约 4-6h。moonlight/dsv1 的加载路径若不兼容（trust_remote_code 相关）
# 会记失败并继续，不熔断（bench 各格独立，与 E4b 不同）。

. /home/daodao/anaconda3/etc/profile.d/conda.sh
conda activate dart312

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2_divisions:4"

# <tag>|<quant-dir>|<extra flags>（tag 仅用于 marker/日志）
# dsv1(deepseek-moe-16b)/moon(Moonlight) 的 config 带 auto_map，必须
# --trust-remote-code 才能过 resolve_trust_remote_code（1031 bench 实测
# dsv1 挂在此处）；dsv2/qwen3/olmoe 保持原生 modeling 不加 flag。
CKPTS="
dsv2-1bpw|quant_ckpt/dsv2-1bpw-s1|
dsv2-2bpw|quant_ckpt/dsv2-2bpw-s1|
qwen3-1bpw|quant_ckpt/qwen3-30b-a3b-1bpw-s1|
qwen3-2bpw|quant_ckpt/qwen3-30b-a3b-2bpw-s1|
olmoe-1bpw|quant_ckpt/olmoe-1bpw-s1|
olmoe-2bpw|quant_ckpt/olmoe-2bpw-s1|
dsv1-1bpw|quant_ckpt/dsv1-1bpw-s1|--trust-remote-code
dsv1-2bpw|quant_ckpt/dsv1-2bpw-s1|--trust-remote-code
moon-1bpw|quant_ckpt/moon-1bpw-s1|--trust-remote-code
moon-2bpw|quant_ckpt/moon-2bpw-s1|--trust-remote-code
"

MODES="wxa16 wxa8"
PREFILL_LENS="16,128,2048"
BATCH_SIZES="1,4"
DECODE_TOKENS="128,256"
GRAPH=${GRAPH:-0}

mkdir -p logs/bench_done
total_fail=0
# 连续失败熔断：bench 未经 GPU 实跑验证，系统性 bug（加载/批量路径崩）时
# 每格仍要烧一次模型加载；连败 5 格直接停，修好后重启脚本（marker 续跑）。
# 阈值 5 的依据：dsv1/moon 两个未验证模型排最后，合法连败最多 4 不会误触。
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
