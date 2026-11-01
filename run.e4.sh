#!/bin/sh

# E4b 端到端同预算分配基线（rebuttal W5：DP 搜索的贡献要用 PPL 兑现）
#
# 用法：直接 bash run.e4.sh（或 sh run.e4.sh，两种都兼容）。逐格调
#   python viz/e4_alloc_baselines.py run --alloc-scheme $MODE --alloc-seed 42 \
#       <模型路径> wikitext2 --slices 8 ... (其余与 run.zb.sh 完全一致)
# 注入 --alloc-scheme {uniform,random,unconstrained} 后原样执行 run_dartmoq.py，
# 逐层与 canonical DP 解对齐同一 raw 预算（详见 viz/e4_alloc_baselines.py 头注释）。
#
# 三种基线语义（与 E4a replay 同一套代码 alloc_layer_all_modes）：
#   uniform        原始 (expert,sub) 顺序整 bit 切分，不用任何敏感度信息
#   random         打乱 DP 解的 bit 多重集（同预算同直方图，换"谁拿哪个 bit"）
#   unconstrained  同 raw 预算下无排序无单调约束的精确 MCKP（E6 oracle 侧）
#
# dp 模式不跑：E1 的 PPL 就是 dp 模式结果（同协议同代码路径），直接引用
# papers/e1_zero_bit_source_tables.md 做对照列。
#
# 每层会打印 [e4-alloc] 行（proxy/raw_bpw/eff_bpw），事后可用日志核对
# 每层预算确实与 dp 解对齐。random 用 --alloc-seed 42（逐层 seed=42+层号，
# 与 E4a replay 一致，proxy 可直接对上）。
#
# 默认批次 = 5 模型 × extrapolate@1.0 × 3 模式 = 15 格；
# 量级约 qwen3 3×~1.6h + 其余各 3×~0.5-2h，合计 ~15-20h（与 E1 单格同价，
# 注入的基线 DP 每层 ~1s 可忽略）。要改范围就改 MODES/BPWS/ZB_SOURCES/MODELS。
#
# 断点续跑：每格成功后在 logs/e4_done/ 落 marker（<模型>_<source>_<bpw>_<mode>），
# 重新启动脚本自动跳过已 marker 的格子；强制重跑就删对应 marker。
#
# 建议启动方式（与 E1 批次同款）：
#   nohup bash run.e4.sh > logs/1031.e4b.log 2>&1 &

. /home/daodao/anaconda3/etc/profile.d/conda.sh
conda activate cmoe311

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2_divisions:4"

# 每行一个模型：<短名>|<路径>（短名与缓存目录 model_id 一致）
MODELS="
qwen3-30b-a3b|/mnt/new1t/models/Qwen3-30B-A3B
deepseek-v1-moe-16b|/mnt/new1t/models/deepseek-moe-16b-base
moonlight|/mnt/new1t/models/Moonlight-16B-A3B
olmoe-7b-1b|/mnt/new1t/models/OLMoE-1B-7B-0924-Instruct
deepseek-v2-lite|/mnt/new1t/models/DeepSeek-V2-Lite
"

MODES="uniform random unconstrained"
ZB_SOURCES="extrapolate"
BPWS="1.0"
ALLOC_SEED=42

RANK_MODE="turboquant_innerproduct"

mkdir -p logs/e4_done

# 连续失败熔断：连续 3 个格子失败（如系统休眠/驱动崩）就整体中止，
# 修好后重新启动即可，marker 会保住进度
fail_streak=0

for entry in $MODELS; do
    modeltag=${entry%%|*}
    modelname=${entry#*|}
    for MOE_BPW in $BPWS; do
        SCHEME="global-bpw-a8s8m${MOE_BPW}"
        for ZB_SOURCE in $ZB_SOURCES; do
            for MODE in $MODES; do
                tag="${modeltag}_${ZB_SOURCE}_${MOE_BPW}_${MODE}"
                if [ -f "logs/e4_done/${tag}" ]; then
                    echo "=== skip (marker exists): ${tag} ==="
                    continue
                fi
                echo "==============================================================="
                echo "E4b alloc baseline: mode=${MODE} seed=${ALLOC_SEED} source=${ZB_SOURCE} bpw=${MOE_BPW} model=${modeltag}"
                echo "start: $(date '+%Y-%m-%d %H:%M:%S')"
                echo "==============================================================="
                if python viz/e4_alloc_baselines.py run --alloc-scheme $MODE --alloc-seed $ALLOC_SEED \
                    $modelname wikitext2 --slices 8 --nsamples 64 \
                    --rank-mode $RANK_MODE --quant-scheme $SCHEME --quantmode turboquant \
                    --zero-bit-source $ZB_SOURCE --standby-layer-cpu; then
                    touch "logs/e4_done/${tag}"
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
done
