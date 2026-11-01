#!/bin/sh

# 轮询等待当前 run.e4.sh 批次结束后只做一件事：跑 run.bench.sh
# （推理速度数据采集，10 ckpt x {wxa16,wxa8}）。
#
# 注意：e4b 的 unconstrained 补格（qwen3/dsv1/moon/olmoe 4 格）**不在**
# 本脚本职责内——由本人手动择时跑（直接 bash run.e4.sh，marker 自动跳过
# 已完成格子）。
#
# 用法（脱离终端，日志聚合到一个文件）：
#   nohup bash run.e4.watch.sh > logs/1031.e4b2_bench.log 2>&1 &
#
# 轮询对象默认取启动时现存最早的 "bash run.e4.sh" 进程（pgrep 正则里
# [.] 防自匹配）；也可显式指定：WATCH_PID=<pid> bash run.e4.watch.sh。
# 要中止接力就 kill 掉本脚本进程（不影响正在跑的批次）。

POLL_SEC=${POLL_SEC:-60}

if [ -z "$WATCH_PID" ]; then
    WATCH_PID=$(pgrep -f "run[.]e4[.]sh" | head -1)
fi
if [ -z "$WATCH_PID" ]; then
    echo "[watch] no running run.e4.sh found; proceed immediately. $(date)"
else
    echo "[watch] waiting for run.e4.sh (pid ${WATCH_PID}) to finish: $(date)"
    while kill -0 "${WATCH_PID}" 2>/dev/null; do
        sleep ${POLL_SEC}
    done
    echo "[watch] run.e4.sh (pid ${WATCH_PID}) finished: $(date)"
fi

echo "[watch] === bench 开始 (run.bench.sh) : $(date) ==="
bash run.bench.sh
echo "[watch] bench 结束 exit=$?: $(date)"

echo "[watch] all done (e4b 补格需手动 bash run.e4.sh): $(date)"
