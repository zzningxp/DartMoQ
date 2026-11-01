#!/bin/sh

# E1 端到端 0-bit 来源对照实验（rebuttal W1/W2/W3 四路对比的 DP 级部分）
#
# 用法：直接 bash run.zb.sh（或 sh run.zb.sh，两种都兼容），一次执行 MODELS ×
# ZB_SOURCES × BPWS 全交叉；要补/减组合就改下面三个列表。
#
#   ZB_SOURCES:
#     direct       (b) 直接测量 0-bit proxy loss（缓存 _b0direct.pt；五模型
#                      E1 已全层算齐，直接命中缓存）
#     direct-scale0.1 / direct-scale10 / direct-scale100
#                   direct 的 L0 幅值扫描（×0.1/×10/×100）：复用 _b0direct.pt
#                   缓存，系数仅在内存中乘，不写回缓存。×<1 = 0-bit 更便宜
#                   （剪得更狠），×>1 = 更贵（剪得更保守）
#     extrapolate  (c) log-quad 外推（复用 _b0.pt 缓存，五模型全层齐全）
#     magnitude    (d) 标准剪枝分数当 L₀：||gate行||²+||up行||²+||down列||²
#     activation   (d) 标准剪枝分数当 L₀：E[z²]（direct 无 down 范数因子版）
#     none         (a) 不允许 0-bit（--disable-0bit-prune 等价）
#   E1 四路全交叉已 115/115 收官（2026-10-28，见 papers/e1_zero_bit_source_tables.md）；
#   direct L0 幅值扫描 ×{10,100}@1.0 五模型已收官（2026-10-30，同表文件）；
#   当前批次为 rebuttal 表补格（dsv2/qwen3 × 0.5/1.5）。
#
# 补格批次量级（4 模型 × 7 格 = 28 格）：qwen3 7×~1.6h ≈ 11h，dsv1/moon 各
# ~2.2h，olmoe ~1.2h，合计 ~17h。qwen3 排最前（rebuttal 表要它的列）。
#
# 断点续跑：每格成功后在 logs/zb_done/ 落一个 marker（<模型>_<source>_<bpw>），
# 重新启动脚本时已有 marker 的格子自动跳过；要强制重跑就删掉对应 marker。
# none 在 bpw<1.0 时 DP 结构性不可行（无 0-bit 候选最低有效 1.25 bpw），自动跳过。
#
# 缓存说明：
#   b1-b4 敏感度复用现有缓存（_b1.pt ... _b4.pt）
#   direct 首跑会逐层计算并缓存 _b0direct.pt；外推值继续用 _b0.pt，互不覆盖
#
# 协议与论文 run.s.sh 完全一致（--slices 8 --nsamples 64，fake quant，
# standby-layer-cpu），保证与论文锚点数字可比。

. /home/daodao/anaconda3/etc/profile.d/conda.sh
conda activate cmoe311

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2_divisions:4"

# 每行一个模型：<短名>|<路径>（短名与缓存目录 model_id 一致）
# 当前批次：剩余四模型补格 —— {0.5,1.5} × {×0.1,×10,×100} + 1.0 × ×0.1
# = 每模型 7 格 × 4 模型 = 28 格。1.0 的 ×10/×100 首批已有 marker（注意
# bpw 字符串必须是 "1.0" 才能命中旧 marker，写 "1" 会导致重跑）。
# dsv2 已全齐（×0.1@1.0 那格 marker 是 "..._direct-scale0.1_1"，字符串不同，
# 别把 dsv2 加回来跑 1.0，否则该格会重跑一次）。
MODELS="
qwen3-30b-a3b|/mnt/new1t/models/Qwen3-30B-A3B
deepseek-v1-moe-16b|/mnt/new1t/models/deepseek-moe-16b-base
moonlight|/mnt/new1t/models/Moonlight-16B-A3B
olmoe-7b-1b|/mnt/new1t/models/OLMoE-1B-7B-0924-Instruct
"
# deepseek-v2-lite|/mnt/new1t/models/DeepSeek-V2-Lite

ZB_SOURCES="direct-scale0.1 direct-scale10 direct-scale100"
BPWS="0.5 1.0 1.5"

RANK_MODE="turboquant_innerproduct"

mkdir -p logs/zb_done

# 连续失败熔断：连续 3 个格子失败（如系统休眠/驱动崩）就整体中止，
# 修好后重新启动即可，marker 会保住进度
fail_streak=0

for entry in $MODELS; do
    modeltag=${entry%%|*}
    modelname=${entry#*|}
    for MOE_BPW in $BPWS; do
        SCHEME="global-bpw-a8s8m${MOE_BPW}"
        for ZB_SOURCE in $ZB_SOURCES; do
            if [ "$ZB_SOURCE" = "none" ] && awk -v b="$MOE_BPW" 'BEGIN{exit !(b<1.0)}'; then
                echo "=== skip: source=none bpw=${MOE_BPW} model=${modeltag} (<1.0 infeasible without 0bit) ==="
                continue
            fi
            tag="${modeltag}_${ZB_SOURCE}_${MOE_BPW}"
            if [ -f "logs/zb_done/${tag}" ]; then
                echo "=== skip (marker exists): ${tag} ==="
                continue
            fi
            echo "==============================================================="
            echo "E1 zero-bit A/B: source=${ZB_SOURCE} bpw=${MOE_BPW} model=${modeltag}"
            echo "start: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "==============================================================="
            if python run_dartmoq.py $modelname wikitext2 --slices 8 --nsamples 64 \
                --rank-mode $RANK_MODE --quant-scheme $SCHEME --quantmode turboquant \
                --zero-bit-source $ZB_SOURCE --standby-layer-cpu; then
                touch "logs/zb_done/${tag}"
                fail_streak=0
            else
                fail_streak=$((fail_streak + 1))
                echo "!!! FAILED: ${tag} (consecutive failures: ${fail_streak})"
                if [ "$fail_streak" -ge 3 ]; then
                    echo "!!! 3 consecutive failures - GPU/driver likely broken (system suspend?)."
                    echo "!!! Aborting. Fix environment and relaunch; markers preserve progress."
                    exit 1
                fi
            fi
        done
    done
done
