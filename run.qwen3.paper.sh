#!/bin/sh

# 论文 Qwen3 列刷新批（2026-11-01 溯源结论的善后）
#
# 背景：论文主表 Qwen3 行出自 2026-06-18/06-20 的坏代码窗口（06-12 合入的
# 0bit compensation DP 当时分配劣化，layer0 有 53% 选 1-bit，@1.0 差 2.6 倍），
# 07-04 d62406d rollback 后代码已正确，但论文格子一直没刷新。出处对照：
#   @1.0 IPE-TQ-DP 28.18/48.2033  <- slurm-26251 块1（坏窗口）
#   @1.5 IPE-TQ-DP 13.7869/20.8771 <- slurm-26243 块1（坏窗口）
#   @2.0 IPE-TQ-DP 10.8818/15.5086 <- slurm-26251 块3（坏窗口）
#   @1.0 GPTQ-DP 982.38/1798.25   <- slurm-26251 块2（坏窗口）
#   @1.0 GPTQ-Origin 4221.86/4872.95 <- slurm-26243 块3（坏窗口）
#   @1.0 TQ-Origin 1514.07/1284.21  <- slurm-26233（同期）
#   @1.0 CAMERA-DP 1886.72/1422.79  <- slurm-25880（窗口前，旧代码；重跑求同代码公平）
# 交叉验证锚点：06-14 slurm-26225（同缓存、可行集更小）@1.0 = 11.68/20.94；
# 当前代码 E1 extrapolate @1.0 = 10.95/18.73（papers/e1_zero_bit_source_tables.md）。
#
# 用法：bash run.qwen3.paper.sh 2>&1 | tee logs/$(date +%m%d).qwen3.paper.log
#   每格成功后在 logs/qwen3paper_done/ 落 marker（<method>_<bpw>），重启自动跳过；
#   要强制重跑某格就删对应 marker。
#
# 覆盖（按优先级排序，跑不完至少保证前面的格子）：
#   1. IPE-TQ-DP   × {1.0, 1.5, 2.0}  论文方法行，必改
#   2. GPTQ-DP     × {1.0, 1.5, 2.0}  坏窗口产物，方法行变好后必须同代码重跑
#   3. CAMERA-DP   × {1.0, 1.5, 2.0}  旧代码产物，同代码统一口径
#   4. TQ-Origin / GPTQ-Origin × 1.0  坏窗口产物（固定 a8s8m1，slices=1）
#   GEMQ 为外部基线不在本脚本范围。
#   若论文 1.5/2.0 表也有 Origin 行，需用固定 econfig 形态补格
#   （参考 26243 的 global-a8s8m32222221，非整数 bpw 无法用 a8s8m1.5 直写）。
#
# 协议：与论文 run.s.sh / E1 完全一致（--slices 8 --nsamples 64，fake quant，
# standby-layer-cpu，默认 zero-bit-source=extrapolate 即论文方法的 b0 log-quad 外推）。
# b0-b4 敏感度缓存 E1 已算齐（_b0.pt/_b1.pt..._b4.pt），本批直接命中缓存。
# 量级：qwen3 turboquant 每格 ~1.6h，全部 11 格约 15h+，可分批跑。

. /home/daodao/anaconda3/etc/profile.d/conda.sh
conda activate cmoe311

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2_divisions:4"

MODEL="/mnt/new1t/models/Qwen3-30B-A3B"

# 每行一格：<marker名>|<额外参数>（公共部分：--nsamples 64 --standby-layer-cpu）
CELLS="
ipe-tq-dp_1.0|--slices 8 --rank-mode turboquant_innerproduct --quantmode turboquant --quant-scheme global-bpw-a8s8m1.0
ipe-tq-dp_1.5|--slices 8 --rank-mode turboquant_innerproduct --quantmode turboquant --quant-scheme global-bpw-a8s8m1.5
ipe-tq-dp_2.0|--slices 8 --rank-mode turboquant_innerproduct --quantmode turboquant --quant-scheme global-bpw-a8s8m2.0
gptq-dp_1.0|--slices 8 --rank-mode gptq_quant_outlier --quantmode gptq --quant-scheme global-bpw-a8s8m1.0
gptq-dp_1.5|--slices 8 --rank-mode gptq_quant_outlier --quantmode gptq --quant-scheme global-bpw-a8s8m1.5
gptq-dp_2.0|--slices 8 --rank-mode gptq_quant_outlier --quantmode gptq --quant-scheme global-bpw-a8s8m2.0
camera-dp_1.0|--slices 8 --rank-mode energy --quantmode turboquant --quant-scheme global-bpw-a8s8m1.0
camera-dp_1.5|--slices 8 --rank-mode energy --quantmode turboquant --quant-scheme global-bpw-a8s8m1.5
camera-dp_2.0|--slices 8 --rank-mode energy --quantmode turboquant --quant-scheme global-bpw-a8s8m2.0
tq-origin_1.0|--slices 1 --rank-mode turboquant_innerproduct --quantmode turboquant --quant-scheme a8s8m1
gptq-origin_1.0|--slices 1 --rank-mode gptq_quant_outlier --quantmode gptq --quant-scheme a8s8m1
"

mkdir -p logs/qwen3paper_done

# 连续失败熔断：连续 3 格失败（系统休眠/驱动崩）整体中止，marker 保住进度
fail_streak=0

for entry in $CELLS; do
    tag=${entry%%|*}
    extra=${entry#*|}
    if [ -f "logs/qwen3paper_done/${tag}" ]; then
        echo "=== skip (marker exists): ${tag} ==="
        continue
    fi
    echo "==============================================================="
    echo "paper qwen3 refresh: ${tag}"
    echo "start: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "==============================================================="
    if python run_dartmoq.py $MODEL wikitext2 --nsamples 64 $extra --standby-layer-cpu; then
        touch "logs/qwen3paper_done/${tag}"
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
