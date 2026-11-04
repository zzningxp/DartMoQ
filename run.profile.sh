#!/bin/sh

# graph replay 逐 kernel profile（profile_graph_decode.py 的驱动）。
# 目的：qwen3-1bpw wxa8 28.4ms/step 的逐 kernel 归因，决定调优走 kernel
# 效率还是 kernel 数量（见 papers/inference_bench_1031.md 的调优路径）。
#
# 与 run.graph.sh 的差别：不跑 eager 参考与 token 对拍，纯测时 + CUPTI
# profile；单卡 0（<8GB）。marker 在 logs/profile_done/（删 marker 强制重跑）。
#
# 用法：
#   nohup bash run.profile.sh > logs/1101.profile_qwen3.log 2>&1 &

. /home/daodao/anaconda3/etc/profile.d/conda.sh
conda activate dart312

export CUDA_VISIBLE_DEVICES=0
export HF_DATASETS_OFFLINE=1

# <tag>|<quant-dir>|<extra flags>
CKPTS="
qwen3-1bpw|quant_ckpt/qwen3-30b-a3b-1bpw-s1|
dsv2-1bpw|quant_ckpt/dsv2-1bpw-s1|
"

mkdir -p logs/profile_done
total_fail=0

for entry in $CKPTS; do
    tag=${entry%%|*}
    rest=${entry#*|}
    qdir=${rest%%|*}
    extra=${rest#*|}
    mtag="${tag}_wxa8"
    if [ -f "logs/profile_done/${mtag}" ]; then
        echo "=== skip (marker exists): ${mtag} ==="
        continue
    fi
    echo "==============================================================="
    echo "profile: ckpt=${tag} mode=wxa8 mask=native"
    echo "start: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "==============================================================="
    if python profile_graph_decode.py --quant-dir "$qdir" --mode wxa8 \
        --prompt-len 128 $extra; then
        touch "logs/profile_done/${mtag}"
    else
        total_fail=$((total_fail + 1))
        echo "!!! FAILED (native mask): ${mtag} — 换 --mask-mode buffer 重试"
        if python profile_graph_decode.py --quant-dir "$qdir" --mode wxa8 \
            --prompt-len 128 --mask-mode buffer $extra; then
            touch "logs/profile_done/${mtag}"
        else
            echo "!!! FAILED (buffer mask too): ${mtag}"
            total_fail=$((total_fail + 1))
        fi
    fi
done

echo "profile done. total_fail=${total_fail}"
