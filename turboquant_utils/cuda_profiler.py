"""CUDA event 分阶段计时工具（P6-0）。

背景
----
项目里原有的 forward 计时全部基于 `time.time()`，且中间没有任何
`torch.cuda.synchronize()`。在 CUDA 异步执行模型下，这类计时测到的是
**CPU 端 launch 时间**，而不是 GPU 端执行时间：

- CPU 会一路把 kernel 塞进 stream 然后跑到前面去，某一段的耗时会被记到
  "下一次 CPU 真正阻塞的地方"（通常是显存分配器要 cudaMalloc/cudaFree，
  或者出现 D2H 拷贝）。
- 典型症状：`wxa16_bit_partitioned_moe.py` 的 `init` 段只有一个 reshape +
  一个 zeros_like，却在 warm 层日志里占到 55%——那是分配器阻塞时把上一层
  排队未完成的 GPU 工作一起吸收进来了，不是这一段真的在干活。

本模块提供基于 `torch.cuda.Event` 的分阶段计时：event 只是往 stream 里插
标记，不打断流水线；forward 末尾统一 synchronize 一次再取 elapsed_time，
得到的是真实 GPU 执行时间。

设计要点
--------
- **event 复用**：每个 stage 在每一轮里可能被调用几百次（per-expert 循环），
  每次都 new 一个 Event 开销可观。这里按 (stage, 序号) 复用已分配的 event，
  轮次之间只重置游标，不重新分配。
- **零开销关闭**：`enabled=False` 时所有方法是空操作，主流程可以常驻调用。
- **嵌套安全**：用显式的 `stage(name)` 上下文管理器，避免忘记配对。

用法
----
    prof = CudaStageProfiler(enabled=True)

    prof.begin_round()                  # 每次 forward 开头
    with prof.stage("rotation"):
        x_rot = batch_rotate_input(...)
    with prof.stage("gate_up"):
        out = kernel(...)
    stats = prof.end_round()            # 这里做唯一一次 synchronize

    # stats: {"rotation": {"ms": 12.3, "calls": 1}, "gate_up": {...}, ...}
"""

from __future__ import annotations

import contextlib
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch


class CudaStageProfiler:
    """按 stage 累计 GPU 执行时间的轻量 profiler。

    Args:
        enabled: 关闭时所有方法退化为空操作，不产生任何 CUDA 调用。
        device: event 所属设备；None 表示使用当前设备。
    """

    def __init__(self, enabled: bool = False, device: Optional[torch.device] = None):
        self.enabled = bool(enabled) and torch.cuda.is_available()
        self.device = device

        # stage -> [(start_event, end_event), ...]，跨轮次复用
        self._pool: Dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        # stage -> 本轮已用到第几对 event
        self._cursor: Dict[str, int] = {}
        # 保持 stage 的首次出现顺序，方便打印时按执行顺序排列
        self._order: "OrderedDict[str, None]" = OrderedDict()

        self._in_round = False
        self.last_stats: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # 轮次控制
    # ------------------------------------------------------------------

    def begin_round(self) -> None:
        """开始一轮测量（通常对应一次 forward）。

        会先 synchronize 一次，把上一轮残留的异步工作排空，
        否则第一个 stage 会把上一层的尾巴算进来。
        """
        if not self.enabled:
            return
        torch.cuda.synchronize(self.device)
        for name in self._cursor:
            self._cursor[name] = 0
        self._in_round = True

    def end_round(self) -> Dict[str, Dict[str, float]]:
        """结束一轮，做唯一一次 synchronize 并汇总各 stage 的 GPU 时间。

        Returns:
            {stage: {"ms": 累计毫秒, "calls": 调用次数}}；关闭时返回空 dict。
        """
        if not self.enabled:
            return {}

        torch.cuda.synchronize(self.device)

        stats: Dict[str, Dict[str, float]] = {}
        for name in self._order:
            n = self._cursor.get(name, 0)
            if n == 0:
                continue
            total = 0.0
            for start_ev, end_ev in self._pool[name][:n]:
                total += start_ev.elapsed_time(end_ev)  # 毫秒
            stats[name] = {"ms": total, "calls": float(n)}

        self._in_round = False
        self.last_stats = stats
        return stats

    # ------------------------------------------------------------------
    # stage 计时
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def stage(self, name: str):
        """给一段代码打 GPU 计时标记。关闭时为零开销空操作。"""
        if not self.enabled:
            yield
            return

        start_ev, end_ev = self._acquire(name)
        start_ev.record()
        try:
            yield
        finally:
            end_ev.record()

    def _acquire(self, name: str) -> Tuple[torch.cuda.Event, torch.cuda.Event]:
        """取一对可用的 event，池子不够就扩容（扩容只在前几轮发生）。"""
        if name not in self._pool:
            self._pool[name] = []
            self._cursor[name] = 0
            self._order[name] = None

        idx = self._cursor[name]
        pool = self._pool[name]
        if idx >= len(pool):
            pool.append((
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            ))
        self._cursor[name] = idx + 1
        return pool[idx]

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------

    @staticmethod
    def format_stats(stats: Dict[str, Dict[str, float]], indent: str = "    ") -> str:
        """把 end_round() 的结果格式化成带占比的多行文本。"""
        if not stats:
            return f"{indent}(cuda profiler disabled / no data)"

        total = sum(v["ms"] for v in stats.values())
        lines = [f"{indent}[CUDA] measured total: {total:.3f} ms"]
        for name, v in stats.items():
            pct = (v["ms"] / total * 100.0) if total > 0 else 0.0
            calls = int(v["calls"])
            per_call = v["ms"] / calls if calls else 0.0
            lines.append(
                f"{indent}  {name:<22} {v['ms']:>9.3f} ms  {pct:>5.1f}%  "
                f"calls={calls:<6d} per_call={per_call * 1000:.1f} us"
            )
        return "\n".join(lines)

    def report(self, indent: str = "    ") -> str:
        """格式化最近一轮的结果。"""
        return self.format_stats(self.last_stats, indent=indent)


# ---------------------------------------------------------------------------
# SM 填充度 / 并行度分析（测试程序用，主流程不调用）
# ---------------------------------------------------------------------------

def sm_occupancy_report(
    grid_programs: int,
    device: Optional[torch.device] = None,
    label: str = "",
) -> str:
    """给出一次 kernel launch 的 SM 填充度估算。

    这是项目规范要求的「GPU 填充度分析 / 并行 SM 分析」的公共实现：
    kernel 的 program 数如果不是 SM 数的整数倍，最后一波就跑不满，
    program 数越接近 SM 数的整数倍，尾部浪费越小。

    Args:
        grid_programs: 该次 launch 的 program（CTA）总数。
        device: 目标设备，None 表示当前设备。
        label: 打印用的标签。

    Returns:
        单行描述字符串。
    """
    props = torch.cuda.get_device_properties(device if device is not None else torch.cuda.current_device())
    num_sm = props.multi_processor_count

    waves = grid_programs / num_sm
    full_waves = grid_programs // num_sm
    tail = grid_programs - full_waves * num_sm
    # 尾波利用率：最后一波用了几个 SM
    tail_util = (tail / num_sm * 100.0) if tail else 100.0
    # 整体利用率：总 program 数 / (向上取整的波数 × SM 数)
    import math as _math
    ceil_waves = _math.ceil(waves) if waves > 0 else 0
    overall = (grid_programs / (ceil_waves * num_sm) * 100.0) if ceil_waves else 0.0

    prefix = f"[{label}] " if label else ""
    return (
        f"{prefix}programs={grid_programs}  SM={num_sm}  waves={waves:.2f}  "
        f"tail_sm={tail}  tail_util={tail_util:.1f}%  overall_sm_util={overall:.1f}%"
    )
