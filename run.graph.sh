#!/bin/sh

# CUDA graph decode 数据采集（qwen3-1bpw，wxa8 静态派发 whole 策略）。
# graph_decode.py 只接受 --mode wxa8（wxa16 host 编排进 capture 必死）；
# qwen3 走 transformers 5.20 原生 modeling，不需要 --trust-remote-code。
#
# 网格对齐 bench decode 档：prompt_len=128 x steps={128,256}（eager 参考
# 只跑最长档 256，短档按前缀对拍）。mask 默认 native；native 挂了按
# graph_decode.py 文件头预案自动换 buffer 补跑一次（结果 JSON 的
# meta.mask_mode 记录实际模式）。
#
# 单卡 0 即可（1bpw 峰值 <8GB）。marker 在 logs/graph_done/。
#
# 用法：
#   nohup bash run.graph.sh > logs/1104.2.graph_qwen3.log 2>&1 &
#   FRESH=1 nohup bash run.graph.sh > logs/<new>.log 2>&1 &
#   （FRESH=1 先清空全部 marker 从头重跑；内核/静态派发改动后重测用这个）

. /home/daodao/anaconda3/etc/profile.d/conda.sh
conda activate dart312

export CUDA_VISIBLE_DEVICES=0
export HF_DATASETS_OFFLINE=1

# <tag>|<quant-dir>（tag 仅用于 marker/日志）
CKPTS="
qwen3-1bpw|quant_ckpt/qwen3-30b-a3b-1bpw-s1
"

mkdir -p logs/graph_done
FRESH=${FRESH:-0}
if [ "$FRESH" = "1" ]; then
    echo "=== FRESH=1: clearing all graph_done markers (full rerun) ==="
    rm -f logs/graph_done/*
fi
total_fail=0

for entry in $CKPTS; do
    tag=${entry%%|*}
    qdir=${entry#*|}
    mtag="${tag}_wxa8"
    if [ -f "logs/graph_done/${mtag}" ]; then
        echo "=== skip (marker exists): ${mtag} ==="
        continue
    fi
    echo "==============================================================="
    echo "graph_decode: ckpt=${tag} mode=wxa8 mask=native"
    echo "start: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "==============================================================="
    if python graph_decode.py --quant-dir "$qdir" --mode wxa8 \
        --prompt-len 128 --steps 128,256; then
        touch "logs/graph_done/${mtag}"
    else
        total_fail=$((total_fail + 1))
        echo "!!! FAILED (native mask): ${mtag} — 按预案换 --mask-mode buffer 重试"
        echo "start: $(date '+%Y-%m-%d %H:%M:%S')"
        if python graph_decode.py --quant-dir "$qdir" --mode wxa8 \
            --prompt-len 128 --steps 128,256 --mask-mode buffer; then
            touch "logs/graph_done/${mtag}"
        else
            echo "!!! FAILED (buffer mask too): ${mtag}"
            total_fail=$((total_fail + 1))
        fi
    fi
done

echo "graph_decode done. total_fail=${total_fail}"
