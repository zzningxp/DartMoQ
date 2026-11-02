#!/usr/bin/env python3
"""Parse DartMoQ slurm logs into aligned benchmark rows.

The parser scans each log sequentially and starts a new row at every
"Loading model: (ppl) ..." line. Metrics that are missing because a run crashed
or skipped downstream evaluation are left empty instead of being shifted onto a
neighboring row.

Also parses bench_throughput.py logs (driven by run.bench.sh, see
papers/inference_bench_1031.md): one row per bench cell, with the storage
breakdown / eff bpw / VRAM peaks / prefill+decode grid / MoE-stage split /
consistency check. Bench logs are auto-detected per file, so a mixed list of
quantization and bench logs works.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Iterable


TASK_ALIASES = {
    "arc_challenge": "arc_c",
    "arc_easy": "arc_e",
    "piqa": "piqa",
    "boolq": "boolq",
    "winogrande": "wino",
    "mnli": "mnli",
    "hellaswag": "hella",
    "mmlu": "mmlu",
}

TASK_FIELDS = [
    "arc_c_acc", "arc_c_acc_norm",
    "arc_e_acc", "arc_e_acc_norm",
    "piqa_acc", "piqa_acc_norm",
    "boolq_acc",
    "wino_acc",
    "mnli_acc",
    "hella_acc", "hella_acc_norm",
    "mmlu_acc",
]

FIELDNAMES = [
    "git_hash", "model_name", "slices", "quant_scheme", "rank_mode",
    "moe_struct", "quantmode", "zero_bit_source", "disable_0bit_prune", "standby_layer_cpu",
    "quant_layers", "wxa16_real_quant", "bpw", "ppl_wikitext2", "ppl_c4",
    *TASK_FIELDS,
    "status", "runtime_ppl", "runtime_quant", "runtime_ppl_eval",
    "runtime_ppl_wikitext2", "runtime_ppl_c4", "runtime_zero_eval", "error",
]

LOADING_RE = re.compile(r"Loading model:\s*\(ppl\)\s*(?P<path>\S+)")
QUANTMODE_RE = re.compile(
    r"slices/quant-scheme/rank-mode/moe-struct/quantmode(?:/disable-0bit-prune)?(?:/standby-layer-cpu)?:\s*\(ppl\)\s+"
    r"(?P<slices>\S+)\s+(?P<quant_scheme>\S+)\s+(?P<rank_mode>\S+)\s+"
    r"(?P<moe_struct>\S+)\s+(?P<quantmode>\S+)(?:\s+(?P<disable_0bit_prune>\S+))?(?:\s+(?P<standby_layer_cpu>\S+))?"
)
BPW_RE = re.compile(r"\bwith bpw\s+(?P<bpw>[-+0-9.eE]+)")
PPL_RE = re.compile(r"ppl on (?P<dataset>wikitext2|c4)(?:\s+\([^)]+\))?:\s*(?P<value>[-+0-9.eE]+|nan|inf)")
TASK_RE = re.compile(r"^(?P<task>[A-Za-z0-9_]+)\s+\{(?P<body>.*)\}\s+time:")
METRIC_RE_TEMPLATE = r"['\"]{name}['\"]:\s*(?:np\.float64\()?([-+0-9.eE]+)"
RUNTIME_RE = re.compile(r"Runtime of training-free construction \(ppl\):\s*(?P<value>[-+0-9.eE]+)")
RUNTIME_QUANT_RE = re.compile(r"Runtime of quantization only:\s*(?P<value>[-+0-9.eE]+)")
RUNTIME_PPL_EVAL_RE = re.compile(r"Runtime of wiki/c4 validation:\s*(?P<value>[-+0-9.eE]+)")
RUNTIME_ZERO_EVAL_RE = re.compile(r"Runtime of zero-shot evaluation:\s*(?P<value>[-+0-9.eE]+)")
START_TIME_RE = re.compile(r"Current start time:\s*(?P<value>.+)")
FATAL_RE = re.compile(r"Segmentation fault|Traceback|RuntimeError|CUDA out of memory|Killed", re.IGNORECASE)
MODEL_NAME_RE = re.compile(r"^model:\s+(?P<path>\S+)\s+(?P<name>\S+)")
NUMERIC_RE = re.compile(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")

# ===========================================================================
# New-format logs (structured header printed by run_dartmoq.py /
# eval_dartmoq.py)
# ===========================================================================
START_RUN_RE = re.compile(r"DartMoQ Quantization Run")
EVAL_START_RUN_RE = re.compile(r"DartMoQ Evaluation")
NEW_MODEL_RE = re.compile(r"^Model:\s+(?P<path>\S+)")
NEW_GIT_HASH_RE = re.compile(r"^Git HEAD:\s+(?P<git_hash>[a-f0-9]+\+?)")
NEW_QUANT_SCHEME_RE = re.compile(r"^Quant scheme:\s+(?P<quant_scheme>\S+)")
NEW_RANK_MODE_RE = re.compile(r"^Rank mode:\s+(?P<rank_mode>\S+)")
NEW_SLICES_RE = re.compile(r"^Slices per expert:\s+(?P<slices>\S+)")
NEW_MOE_STRUCT_RE = re.compile(r"^MOE struct:\s+(?P<moe_struct>\S+)")
NEW_QUANTMODE_RE = re.compile(r"^Quant mode:\s+(?P<quantmode>\S+)")
NEW_STANDBY_CPU_RE = re.compile(r"^CPU standby:\s+(?P<standby_layer_cpu>\S+)")
NEW_WXA16_REAL_QUANT_RE = re.compile(r"^WxA16 real quantization:\s+(?P<wxa16_real_quant>\S+)")
NEW_ZB_SOURCE_RE = re.compile(r"^Zero-bit source:\s+(?P<zero_bit_source>\S+)")
NEW_START_TIME_RE = re.compile(r"^Current time:\s*(?P<value>.+)")
NEW_FINISH_TIME_RE = re.compile(r"^Finish time:\s*(?P<value>.+)")
NEW_PPL_RE = re.compile(r"ppl on (?P<dataset>wikitext2|c4)(?:\s+\([^)]+\))?:\s*(?P<value>[-+0-9.eE]+|nan|inf)(?:\s+time:\s*(?P<time_value>[-+0-9.eE]+))?")
# Eval format
EVAL_LOAD_QUANTIZED_RE = re.compile(r"^Load quantized checkpoint:\s+(?P<path>\S+)")
EVAL_INFERENCE_QUANT_MODE_RE = re.compile(r"^Inference quant mode:\s+(?P<mode>\S+)")
EVAL_SEQUENTIAL_RE = re.compile(r"^Sequential eval:\s+(?P<sequential>\S+)")
EVAL_STANDBY_CPU_RE = re.compile(r"^Standby CPU:\s+(?P<standby_cpu>\S+)")
EVAL_DATASETS_RE = re.compile(r"^Datasets:\s+(?P<datasets>.+)")

# For backward compatibility with old logs
LAYER_TIME_RE = re.compile(r"Layer (?P<layer>\d+) total reconstruct and quantization time:\s*(?P<value>[-+0-9.eE]+) s")
PPL_INDIVIDUAL_TIME_RE = re.compile(r"ppl on (?P<dataset>wikitext2|c4)(?:\s+\([^)]+\))?:\s*(?:[-+0-9.eE]+|nan|inf)\s*time:\s*(?P<value>[-+0-9.eE]+)")
ZERO_EVAL_OLD_RE = re.compile(r"Zero-shot evaluation time:\s*(?P<value>[-+0-9.eE]+)")

# ===========================================================================
# bpw fallback derived from quant_scheme (used when the log has no
# "with bpw" line)
# ===========================================================================
# Qwen3.5 style: w8a8 / w4a4 ...
QUANT_SCHEME_BPW_RE = re.compile(r"w(?P<weight_bits>\d+)(?:a(?P<act_bits>\d+))?")
# DartMoQ fixed schemes: a8s4m3221 -> MoE expert bitwidth list 3221 (mean = bpw)
DARTMOQ_SCHEME_BPW_RE = re.compile(r"a\d+s\d+m(?P<bits>[\d.]+)")


@dataclass
class RunRecord:
    source: str
    run_idx: int
    start_line: int
    end_line: int = 0
    status: str = "incomplete"
    start_time: str = ""
    model_path: str = ""
    model_name: str = ""
    slices: str = ""
    quant_scheme: str = ""
    rank_mode: str = ""
    moe_struct: str = ""
    quantmode: str = ""
    zero_bit_source: str = ""
    disable_0bit_prune: str = ""
    standby_layer_cpu: str = ""
    quant_layers: str = ""
    wxa16_real_quant: str = ""
    git_hash: str = ""
    bpw: str = ""
    ppl_wikitext2: str = ""
    ppl_c4: str = ""
    arc_c_acc: str = ""
    arc_c_acc_norm: str = ""
    arc_e_acc: str = ""
    arc_e_acc_norm: str = ""
    piqa_acc: str = ""
    piqa_acc_norm: str = ""
    boolq_acc: str = ""
    wino_acc: str = ""
    mnli_acc: str = ""
    hella_acc: str = ""
    hella_acc_norm: str = ""
    mmlu_acc: str = ""
    runtime_ppl: str = ""
    runtime_quant: str = ""
    runtime_ppl_eval: str = ""
    runtime_ppl_wikitext2: str = ""
    runtime_ppl_c4: str = ""
    runtime_zero_eval: str = ""
    error: str = ""
    _fatal: bool = field(default=False, repr=False)
    # Temporary fields for accumulating time from old logs
    _layer_times: list[float] = field(default_factory=list, repr=False)
    _ppl_eval_times: list[float] = field(default_factory=list, repr=False)

    def finalize(self, end_line: int) -> None:
        self.end_line = end_line
        # Only calculate and show quant time if we actually have ppl results
        # (meaning the quantization process completed successfully)
        if not self.runtime_quant and self._layer_times and (self.ppl_wikitext2 or self.ppl_c4):
            total = sum(self._layer_times)
            self.runtime_quant = f"{total:.2f}"
        # Calculate ppl eval time from individual ppl times if not already set
        if not self.runtime_ppl_eval and self._ppl_eval_times:
            total = sum(self._ppl_eval_times)
            self.runtime_ppl_eval = f"{total:.2f}"
        # Calculate bpw from quant_scheme if not already set
        if not self.bpw and self.quant_scheme:
            bpw = self._calculate_bpw_from_scheme(self.quant_scheme)
            if bpw is not None:
                self.bpw = f"{bpw:.2f}"
        # Calculate total time: time = t_q + t_ppl
        if not self.runtime_ppl:
            t_quant = 0.0
            t_ppl = 0.0
            try:
                if self.runtime_quant:
                    t_quant = float(self.runtime_quant)
            except ValueError:
                pass
            try:
                if self.runtime_ppl_eval:
                    t_ppl = float(self.runtime_ppl_eval)
            except ValueError:
                pass
            if t_quant > 0 or t_ppl > 0:
                self.runtime_ppl = f"{t_quant + t_ppl:.2f}"
        if self._fatal:
            self.status = "failed"
        elif self.ppl_wikitext2 and self.ppl_c4:
            # Have both ppl results
            self.status = "ok"
        elif self.runtime_ppl or any(getattr(self, name) for name in TASK_FIELDS):
            # Have other results but not both ppl（含纯 zero-shot 运行）
            self.status = "partial"
        else:
            # Have only one ppl or nothing at all
            self.status = "incomplete"

    @staticmethod
    def _calculate_bpw_from_scheme(quant_scheme: str) -> float | None:
        """Calculate bpw from the quant_scheme string.

        Supports:
          - DartMoQ fixed schemes "a8s4m3221" -> mean of the MoE expert
            bitwidths, (3+2+2+1)/4 = 2.0
          - w8a8 / w4a4 style (Qwen3.5) -> weight bits or (w+a)/2
        Returns None if parsing fails.
        """
        match = DARTMOQ_SCHEME_BPW_RE.search(quant_scheme)
        if match:
            bits = match.group("bits")
            if bits.isdigit():
                return sum(int(d) for d in bits) / len(bits)
            try:
                return float(bits)
            except ValueError:
                pass
        match = QUANT_SCHEME_BPW_RE.search(quant_scheme)
        if match:
            weight_bits = int(match.group("weight_bits"))
            act_bits = match.group("act_bits")
            if act_bits:
                return (weight_bits + int(act_bits)) / 2.0
            return float(weight_bits)
        return None

    def public_dict(self) -> dict[str, str | int]:
        row = asdict(self)
        row.pop("_fatal", None)
        return row


def _clean_line(line: str) -> str:
    return line.replace("\r", "").strip()


def _last_path_component(path: str) -> str:
    return os.path.basename(path.rstrip("/"))


def _metric(body: str, name: str) -> str:
    match = re.search(METRIC_RE_TEMPLATE.format(name=re.escape(name)), body)
    return match.group(1) if match else ""


def _append_error(record: RunRecord, line: str) -> None:
    message = line.strip()
    if not message:
        return
    if record.error:
        record.error += " | " + message
    else:
        record.error = message


def parse_log(path: str) -> list[RunRecord]:
    records: list[RunRecord] = []
    current: RunRecord | None = None
    pending_start_time = ""
    source = os.path.basename(path)
    last_line_no = 0
    # Track which format we're using
    using_new_format = False
    using_eval_format = False

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, raw_line in enumerate(f, 1):
            last_line_no = line_no
            line = _clean_line(raw_line)
            if not line:
                continue

            # Check for eval format start
            eval_start_match = EVAL_START_RUN_RE.search(line)
            if eval_start_match:
                if current is not None:
                    current.finalize(line_no - 1)
                    records.append(current)
                current = RunRecord(
                    source=source,
                    run_idx=len(records) + 1,
                    start_line=line_no,
                    start_time="",
                    model_path="",
                )
                current.quantmode = "fp16"
                using_eval_format = True
                using_new_format = False
                continue

            # Check for new format start
            start_run_match = START_RUN_RE.search(line)
            if start_run_match:
                if current is not None:
                    current.finalize(line_no - 1)
                    records.append(current)
                current = RunRecord(
                    source=source,
                    run_idx=len(records) + 1,
                    start_line=line_no,
                    start_time="",
                    model_path="",
                )
                using_new_format = True
                using_eval_format = False
                continue

            # If not using new format yet, check for old format patterns
            if not using_new_format and not using_eval_format:
                start_time_match = START_TIME_RE.search(line)
                if start_time_match:
                    pending_start_time = start_time_match.group("value").strip()

                loading_match = LOADING_RE.search(line)
                if loading_match:
                    if current is not None:
                        current.finalize(line_no - 1)
                        records.append(current)
                    current = RunRecord(
                        source=source,
                        run_idx=len(records) + 1,
                        start_line=line_no,
                        start_time=pending_start_time,
                        model_path=loading_match.group("path"),
                    )
                    current.model_name = _last_path_component(current.model_path)
                    using_new_format = False
                    using_eval_format = False
                    continue

            if current is None:
                continue

            # Parse layer times - do this first for all formats!
            layer_time_match = LAYER_TIME_RE.search(line)
            if layer_time_match:
                try:
                    t = float(layer_time_match.group("value"))
                    current._layer_times.append(t)
                except ValueError:
                    pass
                # Don't continue, fall through to other matches

            # Both new formats use model line
            model_match = NEW_MODEL_RE.search(line)
            if model_match:
                current.model_path = model_match.group("path")
                current.model_name = _last_path_component(current.model_path)
                continue

            # Parse git hash
            git_hash_match = NEW_GIT_HASH_RE.search(line)
            if git_hash_match:
                current.git_hash = git_hash_match.group("git_hash")
                continue

            # Eval format parsing
            if using_eval_format:
                eval_standby_cpu_match = EVAL_STANDBY_CPU_RE.search(line)
                if eval_standby_cpu_match:
                    current.standby_layer_cpu = eval_standby_cpu_match.group("standby_cpu")
                    continue

                # direct eval of a quantized checkpoint (--load-quantized)
                eval_quant_match = EVAL_LOAD_QUANTIZED_RE.search(line)
                if eval_quant_match:
                    qpath = eval_quant_match.group("path")
                    current.model_path = qpath
                    current.model_name = _last_path_component(qpath)
                    current.wxa16_real_quant = "Yes"
                    current.quantmode = "wxa16_load"
                    continue

                # inference quant mode: wxa8 overrides the default wxa16_load
                # marker (printed after the "Load quantized checkpoint" line)
                eval_inf_mode_match = EVAL_INFERENCE_QUANT_MODE_RE.search(line)
                if eval_inf_mode_match:
                    mode = eval_inf_mode_match.group("mode")
                    current.quantmode = f"{mode}_load"
                    continue

                # Eval uses the same PPL format
                new_ppl_match = NEW_PPL_RE.search(line)
                if new_ppl_match:
                    dataset = new_ppl_match.group("dataset")
                    setattr(current, f"ppl_{dataset}", new_ppl_match.group("value"))
                    time_value = new_ppl_match.groupdict().get("time_value")
                    if time_value:
                        try:
                            t = float(time_value)
                            current._ppl_eval_times.append(t)
                            setattr(current, f"runtime_ppl_{dataset}", f"{t:.2f}")
                        except ValueError:
                            pass
                    continue

            # New format parsing
            elif using_new_format:
                quant_scheme_match = NEW_QUANT_SCHEME_RE.search(line)
                if quant_scheme_match:
                    current.quant_scheme = quant_scheme_match.group("quant_scheme")
                    continue

                rank_mode_match = NEW_RANK_MODE_RE.search(line)
                if rank_mode_match:
                    current.rank_mode = rank_mode_match.group("rank_mode")
                    continue

                slices_match = NEW_SLICES_RE.search(line)
                if slices_match:
                    current.slices = slices_match.group("slices")
                    continue

                moe_struct_match = NEW_MOE_STRUCT_RE.search(line)
                if moe_struct_match:
                    current.moe_struct = moe_struct_match.group("moe_struct")
                    continue

                quantmode_match = NEW_QUANTMODE_RE.search(line)
                if quantmode_match:
                    current.quantmode = quantmode_match.group("quantmode")
                    continue

                standby_cpu_match = NEW_STANDBY_CPU_RE.search(line)
                if standby_cpu_match:
                    current.standby_layer_cpu = standby_cpu_match.group("standby_layer_cpu")
                    continue

                wxa16_real_quant_match = NEW_WXA16_REAL_QUANT_RE.search(line)
                if wxa16_real_quant_match:
                    current.wxa16_real_quant = wxa16_real_quant_match.group("wxa16_real_quant")
                    continue

                zb_source_match = NEW_ZB_SOURCE_RE.search(line)
                if zb_source_match:
                    current.zero_bit_source = zb_source_match.group("zero_bit_source")
                    continue

                start_time_match = NEW_START_TIME_RE.search(line)
                if start_time_match:
                    current.start_time = start_time_match.group("value").strip()
                    continue

                # New format PPL with time
                new_ppl_match = NEW_PPL_RE.search(line)
                if new_ppl_match:
                    dataset = new_ppl_match.group("dataset")
                    setattr(current, f"ppl_{dataset}", new_ppl_match.group("value"))
                    time_value = new_ppl_match.groupdict().get("time_value")
                    if time_value:
                        try:
                            t = float(time_value)
                            current._ppl_eval_times.append(t)
                            setattr(current, f"runtime_ppl_{dataset}", f"{t:.2f}")
                        except ValueError:
                            pass
                    continue

            # Old-format parsing (new-format lines that did not match fall
            # through here as well, so mixed logs keep working)
            quantmode_match = QUANTMODE_RE.search(line)
            if quantmode_match:
                for key, value in quantmode_match.groupdict().items():
                    setattr(current, key, value)
                continue

            bpw_match = BPW_RE.search(line)
            if bpw_match and not current.bpw:
                current.bpw = bpw_match.group("bpw")
                continue

            model_name_match = MODEL_NAME_RE.search(line)
            if model_name_match:
                current.model_name = model_name_match.group("name")
                continue

            # Backward compatibility: parse individual ppl eval times for old logs
            # Need to check this before PPL_RE because PPL_RE matches a subset
            ppl_ind_time_match = PPL_INDIVIDUAL_TIME_RE.search(line)
            if ppl_ind_time_match:
                try:
                    t = float(ppl_ind_time_match.group("value"))
                    current._ppl_eval_times.append(t)
                except ValueError:
                    pass
                # Still need to extract the ppl value, so don't continue here

            ppl_match = PPL_RE.search(line)
            if ppl_match:
                dataset = ppl_match.group("dataset")
                setattr(current, f"ppl_{dataset}", ppl_match.group("value"))
                continue

            task_match = TASK_RE.search(line)
            if task_match:
                task = task_match.group("task")
                prefix = TASK_ALIASES.get(task)
                if prefix:
                    body = task_match.group("body")
                    acc = _metric(body, "acc,none")
                    acc_norm = _metric(body, "acc_norm,none")
                    if acc:
                        setattr(current, f"{prefix}_acc", acc)
                    if acc_norm and f"{prefix}_acc_norm" in TASK_FIELDS:
                        setattr(current, f"{prefix}_acc_norm", acc_norm)
                continue

            runtime_match = RUNTIME_RE.search(line)
            if runtime_match:
                current.runtime_ppl = runtime_match.group("value")
                continue

            runtime_quant_match = RUNTIME_QUANT_RE.search(line)
            if runtime_quant_match:
                current.runtime_quant = runtime_quant_match.group("value")
                continue

            runtime_ppl_eval_match = RUNTIME_PPL_EVAL_RE.search(line)
            if runtime_ppl_eval_match:
                current.runtime_ppl_eval = runtime_ppl_eval_match.group("value")
                continue

            runtime_zero_eval_match = RUNTIME_ZERO_EVAL_RE.search(line)
            if runtime_zero_eval_match:
                current.runtime_zero_eval = runtime_zero_eval_match.group("value")
                continue

            # Backward compatibility: parse old zero-eval time format
            zero_eval_old_match = ZERO_EVAL_OLD_RE.search(line)
            if zero_eval_old_match and not current.runtime_zero_eval:
                current.runtime_zero_eval = zero_eval_old_match.group("value")
                continue

            if FATAL_RE.search(line):
                current._fatal = True
                _append_error(current, line)

    if current is not None:
        current.finalize(last_line_no)
        records.append(current)

    return records


def parse_logs(paths: Iterable[str]) -> list[RunRecord]:
    records: list[RunRecord] = []
    for path in paths:
        records.extend(parse_log(path))
    return records


# ===========================================================================
# Bench logs (bench_throughput.py, driven by run.bench.sh)
# ===========================================================================
BENCH_CELL_RE = re.compile(r"^bench: ckpt=(?P<ckpt_tag>\S+) mode=(?P<mode>\S+)$")
BENCH_SKIP_RE = re.compile(r"^=== skip \(marker exists\): (?P<mtag>\S+) ===$")
BENCH_START_TIME_RE = re.compile(r"^start: (?P<value>.+)$")
BENCH_MODEL_RE = re.compile(r"^\[model\] (?P<model>\S+) on (?P<device>\S+) \| mode: (?P<mode>\S+)$")
BENCH_CKPT_RE = re.compile(
    r"^\[ckpt\] file (?P<ckpt_file_gb>[\d.]+) GB \| tensors (?P<ckpt_tensors_gb>[\d.]+) GB "
    r"\| params (?P<params_total_b>[\d.]+) B \| eff bpw (?P<eff_bpw>[\d.]+)$")
BENCH_OUT_RE = re.compile(r"^\[out\] saved to (?P<path>\S+)$")
BENCH_SUMMARY_HDR_RE = re.compile(r"^Prefill/Decode Throughput Benchmark$")
BENCH_SUMMARY_META_RE = re.compile(
    r"^model: (?P<model>\S+) \| mode: (?P<mode>\S+) \| device: (?P<device>\S+) "
    r"\| dataset: (?P<dataset>\S+)$")
BENCH_SUMMARY_CKPT_RE = re.compile(r"^checkpoint: (?P<ckpt_dir>\S+)$")
BENCH_SUMMARY_MISC_RE = re.compile(
    r"^prompt_len: (?P<prompt_len>\d+) \| decode_tokens: (?P<decode_tokens>[\d,]+) "
    r"\| batch_sizes: (?P<batch_sizes>[\d,]+) \| git: (?P<git_hash>\S+)$")
BENCH_STORAGE_RES = [
    ("attn_shared_gb", re.compile(r"^Attn/Shared packed .*?: ([\d.]+) GB$")),
    ("moe_packed_gb", re.compile(r"^MoE packed \(bit_weights\): ([\d.]+) GB$")),
    ("fp16_rem_gb", re.compile(r"^fp16 remaining params: ([\d.]+) GB$")),
    ("qmeta_gb", re.compile(r"^meta scalars \(qmeta\): ([\d.]+) GB$")),
]
BENCH_PARAMS_RE = re.compile(
    r"^params total: [\d.]+ B \(quantized (?P<params_quant_b>[\d.]+) B "
    r"\+ unquantized (?P<params_unquant_b>[\d.]+) B\)$")
BENCH_VRAM_RE = re.compile(r"^(?P<phase>\S+)\s+alloc=[\d.]+ peak=(?P<peak>[\d.]+) reserved=[\d.]+$")
BENCH_VRAM_PEAK_RE = re.compile(r"^overall peak allocated: (?P<peak>[\d.]+) GB$")
BENCH_PREFILL_RE = re.compile(
    r"^len=(?P<length>\d+)\s+batch=(?P<batch>\d+)\s+latency=(?P<lat>[\d.]+)s\s+tok/s=(?P<tps>[\d.]+)")
BENCH_DECODE_RE = re.compile(
    r"^tokens=(?P<tokens>\d+)\s+batch=(?P<batch>\d+)\s+wall tok/s = (?P<tps>[\d.]+) "
    r"\(\d+ tokens / [\d.]+s, excl\. prefill\) \| event tok/s = (?P<ev_tps>[\d.]+) "
    r"\| per-step wall mean/p50/p99 = [\d.]+/(?P<p50>[\d.]+)/[\d.]+ ms$")
BENCH_TTFT_RE = re.compile(r"^TTFT \(batch=1 tier\) = (?P<ttft>[\d.]+)s \(prefill [\d.]+s \+ step1 [\d.]+s\)$")
BENCH_MOE_HDR_RE = re.compile(r"^MoE stage \(mean per decode step, all layers, s\):")
BENCH_MOE_ROW_RE = re.compile(r"^(?P<stage>\S+)\s+(?P<value>[\d.]+)$")
BENCH_CHECK_RE = re.compile(
    r"^Consistency check: (?P<check_status>PASS|FAIL) "
    r"\(max logits diff (?P<check_max_diff>[\d.eE+-]+) [<>=]+ [\d.]+, "
    r"argmax agree (?P<check_argmax>\d+/\d+)")
BENCH_RESULTS_RE = re.compile(r"^Results: (?P<path>\S+)$")
BENCH_EXC_RE = re.compile(r"^(?P<name>[A-Za-z_.]+(?:Error|Exception|Interrupt)): (?P<msg>.+)$")

# run.bench.sh 的固定网格（PREFILL_LENS=16,128,2048 × BATCH=1,4；
# DECODE_TOKENS=128,256 × BATCH=1,4）。其它网格的日志也能解析，
# 未出现在下列字段的组合只进 json（metrics 全量导出）。
PREFILL_GRID = [(16, 1), (16, 4), (128, 1), (128, 4), (2048, 1), (2048, 4)]
DECODE_GRID = [(128, 1), (128, 4), (256, 1), (256, 4)]
PREFILL_FIELDS = [f"pf_l{L}_b{B}_{kind}" for L, B in PREFILL_GRID for kind in ("lat_s", "tps")]
DECODE_FIELDS = [f"dec_t{T}_b{B}_{kind}" for T, B in DECODE_GRID for kind in ("tps", "ev_tps", "p50_ms")]
MOE_STAGES = ["total", "init", "shared", "router", "sort_scatter_prep", "compute",
              "triton", "dequant", "gemm", "cleanup_reshape"]
MOE_FIELDS = [f"moe_{s}_s" for s in MOE_STAGES]

BENCH_CORE_FIELDS = [
    "source", "run_idx", "status", "start_time", "ckpt_tag", "mode", "model",
    "device", "dataset", "git_hash", "prompt_len", "ckpt_dir",
    "ckpt_file_gb", "ckpt_tensors_gb", "attn_shared_gb", "moe_packed_gb",
    "fp16_rem_gb", "qmeta_gb", "params_total_b", "params_quant_b",
    "params_unquant_b", "eff_bpw",
    "vram_after_load_gb", "vram_prefill_peak_gb", "vram_peak_gb",
    *PREFILL_FIELDS, *DECODE_FIELDS, "ttft_s", *MOE_FIELDS,
    "check_status", "check_max_diff", "check_argmax", "results_json", "error",
]
BENCH_FIELDNAMES = BENCH_CORE_FIELDS

BENCH_DISPLAY_FIELDS = [
    "ckpt_tag", "mode", "eff_bpw", "ckpt_file_gb", "vram_peak_gb",
    "pf_l2048_b1_tps", "pf_l2048_b4_tps",
    "dec_t128_b1_tps", "dec_t128_b1_p50_ms", "dec_t256_b4_tps",
    "ttft_s", "moe_total_s", "check_status", "status", "error",
]
BENCH_PLAIN_HEADERS = {
    "ckpt_tag": "ckpt", "mode": "mode", "model": "model", "eff_bpw": "bpw",
    "ckpt_file_gb": "ckptGB", "vram_peak_gb": "vrPK",
    "pf_l2048_b1_tps": "pf2048x1", "pf_l2048_b4_tps": "pf2048x4",
    "dec_t128_b1_tps": "dec128x1", "dec_t128_b1_p50_ms": "p50ms",
    "dec_t256_b4_tps": "dec256x4", "ttft_s": "ttft",
    "moe_total_s": "moe_s", "check_status": "check", "status": "status",
    "error": "err",
}


@dataclass
class BenchRecord:
    source: str
    run_idx: int
    start_line: int
    end_line: int = 0
    status: str = "incomplete"
    start_time: str = ""
    ckpt_tag: str = ""
    mode: str = ""
    model: str = ""
    device: str = ""
    dataset: str = ""
    git_hash: str = ""
    prompt_len: str = ""
    ckpt_dir: str = ""
    ckpt_file_gb: str = ""
    ckpt_tensors_gb: str = ""
    attn_shared_gb: str = ""
    moe_packed_gb: str = ""
    fp16_rem_gb: str = ""
    qmeta_gb: str = ""
    params_total_b: str = ""
    params_quant_b: str = ""
    params_unquant_b: str = ""
    eff_bpw: str = ""
    vram_after_load_gb: str = ""
    vram_prefill_peak_gb: str = ""
    vram_peak_gb: str = ""
    ttft_s: str = ""
    check_status: str = ""
    check_max_diff: str = ""
    check_argmax: str = ""
    results_json: str = ""
    error: str = ""
    _fatal: bool = field(default=False, repr=False)
    _saw_summary: bool = field(default=False, repr=False)
    # 网格值（pf_* / dec_* / moe_*），public_dict 时摊平
    metrics: dict = field(default_factory=dict, repr=False)

    def finalize(self, end_line: int) -> None:
        self.end_line = end_line
        if self._fatal:
            self.status = "failed"
        elif self.check_status == "PASS" and self.results_json:
            self.status = "ok"
        elif self.check_status or self.metrics or self.results_json:
            self.status = "partial"
        else:
            self.status = "incomplete"

    def public_dict(self) -> dict[str, str | int]:
        row = asdict(self)
        row.pop("_fatal", None)
        row.pop("_saw_summary", None)
        row.pop("metrics", None)
        row.update(self.metrics)
        return row


def _new_bench_record(records: list[BenchRecord], current: BenchRecord | None,
                      line_no: int, source: str, **kwargs) -> BenchRecord:
    if current is not None:
        current.finalize(line_no - 1)
        records.append(current)
    return BenchRecord(source=source, run_idx=len(records) + 1,
                       start_line=line_no, **kwargs)


def parse_bench_log(path: str) -> list[BenchRecord]:
    records: list[BenchRecord] = []
    current: BenchRecord | None = None
    source = os.path.basename(path)
    last_line_no = 0
    in_moe_stage = False

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, raw_line in enumerate(f, 1):
            last_line_no = line_no
            line = _clean_line(raw_line)
            if not line:
                in_moe_stage = False
                continue

            # run.bench.sh 驱动的 cell 头（python 直跑 bench 无此行，靠汇总头开行）
            cell_match = BENCH_CELL_RE.search(line)
            if cell_match:
                current = _new_bench_record(
                    records, current, line_no, source,
                    ckpt_tag=cell_match.group("ckpt_tag"),
                    mode=cell_match.group("mode"))
                in_moe_stage = False
                continue

            skip_match = BENCH_SKIP_RE.search(line)
            if skip_match:
                mtag = skip_match.group("mtag")
                ckpt_tag, _, mode = mtag.rpartition("_")
                current = _new_bench_record(records, current, line_no, source,
                                            ckpt_tag=ckpt_tag, mode=mode)
                current.status = "skipped"
                records.append(current)
                current = None
                continue

            # 汇总块头：python 直跑时的开行；run.bench.sh 驱动时归并进当前 cell
            if BENCH_SUMMARY_HDR_RE.search(line):
                if current is None or current._saw_summary:
                    current = _new_bench_record(records, current, line_no, source)
                current._saw_summary = True
                in_moe_stage = False
                continue
            if current is None:
                continue

            start_time_match = BENCH_START_TIME_RE.search(line)
            if start_time_match and not current.start_time:
                current.start_time = start_time_match.group("value").strip()
                continue

            model_match = BENCH_MODEL_RE.search(line)
            if model_match:
                current.model = model_match.group("model")
                current.device = model_match.group("device")
                if not current.mode:
                    current.mode = model_match.group("mode")
                continue

            ckpt_match = BENCH_CKPT_RE.search(line)
            if ckpt_match:
                for key, value in ckpt_match.groupdict().items():
                    setattr(current, key, value)
                continue

            out_match = BENCH_OUT_RE.search(line)
            if out_match:
                current.results_json = out_match.group("path")
                continue

            meta_match = BENCH_SUMMARY_META_RE.search(line)
            if meta_match:
                current.model = meta_match.group("model")
                current.mode = meta_match.group("mode")
                current.device = meta_match.group("device")
                current.dataset = meta_match.group("dataset")
                continue

            summary_ckpt_match = BENCH_SUMMARY_CKPT_RE.search(line)
            if summary_ckpt_match:
                current.ckpt_dir = summary_ckpt_match.group("ckpt_dir")
                continue

            misc_match = BENCH_SUMMARY_MISC_RE.search(line)
            if misc_match:
                current.prompt_len = misc_match.group("prompt_len")
                current.git_hash = misc_match.group("git_hash")
                continue

            for field_name, rx in BENCH_STORAGE_RES:
                m = rx.search(line)
                if m:
                    setattr(current, field_name, m.group(1))
                    break
            else:
                params_match = BENCH_PARAMS_RE.search(line)
                if params_match:
                    current.params_quant_b = params_match.group("params_quant_b")
                    current.params_unquant_b = params_match.group("params_unquant_b")
                    continue

                vram_match = BENCH_VRAM_RE.search(line)
                if vram_match:
                    phase = vram_match.group("phase")
                    if phase == "after_load":
                        current.vram_after_load_gb = vram_match.group("peak")
                    elif phase == "prefill":
                        current.vram_prefill_peak_gb = vram_match.group("peak")
                    continue

                vram_peak_match = BENCH_VRAM_PEAK_RE.search(line)
                if vram_peak_match:
                    current.vram_peak_gb = vram_peak_match.group("peak")
                    continue

                prefill_match = BENCH_PREFILL_RE.search(line)
                if prefill_match:
                    L = prefill_match.group("length")
                    B = prefill_match.group("batch")
                    current.metrics[f"pf_l{L}_b{B}_lat_s"] = prefill_match.group("lat")
                    current.metrics[f"pf_l{L}_b{B}_tps"] = prefill_match.group("tps")
                    continue

                decode_match = BENCH_DECODE_RE.search(line)
                if decode_match:
                    T = decode_match.group("tokens")
                    B = decode_match.group("batch")
                    current.metrics[f"dec_t{T}_b{B}_tps"] = decode_match.group("tps")
                    current.metrics[f"dec_t{T}_b{B}_ev_tps"] = decode_match.group("ev_tps")
                    current.metrics[f"dec_t{T}_b{B}_p50_ms"] = decode_match.group("p50")
                    continue

                ttft_match = BENCH_TTFT_RE.search(line)
                if ttft_match:
                    current.ttft_s = ttft_match.group("ttft")
                    continue

                if in_moe_stage:
                    moe_match = BENCH_MOE_ROW_RE.search(line)
                    if moe_match:
                        current.metrics[f"moe_{moe_match.group('stage')}_s"] = moe_match.group("value")
                        continue

                if BENCH_MOE_HDR_RE.search(line):
                    in_moe_stage = True
                    continue

                check_match = BENCH_CHECK_RE.search(line)
                if check_match:
                    current.check_status = check_match.group("check_status")
                    current.check_max_diff = check_match.group("check_max_diff")
                    current.check_argmax = check_match.group("check_argmax")
                    continue

                results_match = BENCH_RESULTS_RE.search(line)
                if results_match:
                    current.results_json = results_match.group("path")
                    continue

                exc_match = BENCH_EXC_RE.search(line)
                if exc_match:
                    _append_error(current, line[:200])
                    current._fatal = True
                    continue

                if FATAL_RE.search(line):
                    current._fatal = True

    if current is not None:
        current.finalize(last_line_no)
        records.append(current)
    return records


def is_bench_log(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if BENCH_CELL_RE.search(line.replace("\r", "").strip()) \
                        or BENCH_SUMMARY_HDR_RE.search(line.replace("\r", "").strip()):
                    return True
    except OSError:
        pass
    return False


def _format_row(record: RunRecord, fields: list[str]) -> dict[str, str]:
    row = record.public_dict()
    return {field: _format_value(field, row.get(field, "")) for field in fields}


def _format_export_row(record: RunRecord) -> dict[str, str]:
    row = record.public_dict()
    return {EXPORT_HEADERS[field]: _format_value(field, row.get(field, "")) for field in FIELDNAMES}


def _csv_escape(value: str) -> str:
    if "," in value or '"' in value or "\n" in value:
        return f'"{value.replace(chr(34), chr(34)*2)}"'
    return value


def write_csv(records: list[RunRecord], out) -> None:
    headers = [EXPORT_HEADERS[field] for field in FIELDNAMES]
    out.write(",".join([_csv_escape(h) for h in headers]) + "\n")
    for record in records:
        row = _format_row(record, FIELDNAMES)
        out.write(",".join([_csv_escape(row.get(field, "")) for field in FIELDNAMES]) + "\n")


DISPLAY_FIELDS = [
    "git_hash", "model_name", "slices", "quant_scheme", "rank_mode", "moe_struct", "quantmode",
    "zero_bit_source", "quant_layers", "wxa16_real_quant", "bpw",
    "ppl_wikitext2", "ppl_c4",
    "status",
    "runtime_ppl", "runtime_quant", "runtime_ppl_eval",
    "runtime_ppl_wikitext2", "runtime_ppl_c4", "error",
]

PLAIN_HEADERS = {
    "model_name": "model",
    "slices": "sli",
    "quant_scheme": "qsch",
    "rank_mode": "rank",
    "moe_struct": "moe",
    "quantmode": "qmode",
    "zero_bit_source": "zbsrc",
    "disable_0bit_prune": "no0prune",
    "standby_layer_cpu": "stdbycpu",
    "quant_layers": "qlayers",
    "wxa16_real_quant": "wxa16",
    "git_hash": "git",
    "bpw": "bpw",
    "ppl_wikitext2": "wiki",
    "ppl_c4": "c4",
    "status": "status",
    "runtime_ppl": "time",
    "runtime_quant": "t_quant",
    "runtime_ppl_eval": "t_ppl",
    "runtime_ppl_wikitext2": "t_wiki",
    "runtime_ppl_c4": "t_c4",
    "runtime_zero_eval": "t_zero",
    "error": "err",
}

EXPORT_HEADERS = {
    "model_name": "model",
    "slices": "slices",
    "quant_scheme": "quant_scheme",
    "rank_mode": "rank_mode",
    "moe_struct": "moe_struct",
    "quantmode": "quantmode",
    "zero_bit_source": "zero_bit_source",
    "disable_0bit_prune": "disable_0bit_prune",
    "standby_layer_cpu": "standby_layer_cpu",
    "quant_layers": "quant_layers",
    "wxa16_real_quant": "wxa16_real_quant",
    "git_hash": "git_hash",
    "bpw": "bpw",
    "ppl_wikitext2": "ppl_wikitext2",
    "ppl_c4": "ppl_c4",
    "arc_c_acc": "arc_c_acc",
    "arc_c_acc_norm": "arc_c_acc_norm",
    "arc_e_acc": "arc_e_acc",
    "arc_e_acc_norm": "arc_e_acc_norm",
    "piqa_acc": "piqa_acc",
    "piqa_acc_norm": "piqa_acc_norm",
    "boolq_acc": "boolq_acc",
    "wino_acc": "wino_acc",
    "mnli_acc": "mnli_acc",
    "hella_acc": "hella_acc",
    "hella_acc_norm": "hella_acc_norm",
    "mmlu_acc": "mmlu_acc",
    "status": "status",
    "runtime_ppl": "total_time",
    "runtime_quant": "quant_time",
    "runtime_ppl_eval": "ppl_eval_time",
    "runtime_ppl_wikitext2": "t_wiki",
    "runtime_ppl_c4": "t_c4",
    "runtime_zero_eval": "zero_eval_time",
    "error": "error",
}


def _format_value(field: str, value) -> str:
    if field == "error":
        return ""
    # git_hash looks numeric but must not be formatted
    if field == "git_hash":
        return str(value).replace("\t", " ")
    text = str(value)
    if NUMERIC_RE.fullmatch(text):
        try:
            text = f"{float(text):.4f}".rstrip("0").rstrip(".")
        except (ValueError, TypeError):
            pass
    return text.replace("\t", " ")


def _format_plain_value(record: RunRecord, field: str, value) -> str:
    if record.status in ("failed", "partial", "incomplete"):
        if field in ("runtime_ppl", "error"):
            return "----"
        if field != "status" and value == "":
            return "----"
    return _format_value(field, value)


def _format_bench_plain_value(record: BenchRecord, field: str, value) -> str:
    if field == "error":
        # bench 的 error 是单行异常消息（如 "KeyError: 'type'"），plain 表里
        # 直接展示（截断），不像量化表那样清空
        text = str(value).replace("\t", " ")
        return text[:60] + "..." if len(text) > 60 else text
    if record.status in ("failed", "partial", "incomplete", "skipped"):
        if field != "status" and value == "":
            return "----"
    return _format_value(field, value)


def _render_plain(rows: list[list[str]], out) -> None:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        pieces = []
        for i, value in enumerate(row):
            pieces.append(value)
            if i != len(row) - 1:
                padding = " " * (widths[i] - len(value) + 2)
                pieces.append(padding)
        print("".join(pieces), file=out)


def write_plain(records: list[RunRecord], out) -> None:
    rows: list[list[str]] = []
    rows.append([PLAIN_HEADERS[field] for field in DISPLAY_FIELDS])
    for record in records:
        row = record.public_dict()
        rows.append([_format_plain_value(record, col, row.get(col, "")) for col in DISPLAY_FIELDS])
    _render_plain(rows, out)


def write_bench_plain(records: list[BenchRecord], out) -> None:
    rows: list[list[str]] = []
    rows.append([BENCH_PLAIN_HEADERS.get(f, f) for f in BENCH_DISPLAY_FIELDS])
    for record in records:
        row = record.public_dict()
        rows.append([_format_bench_plain_value(record, col, row.get(col, "")) for col in BENCH_DISPLAY_FIELDS])
    _render_plain(rows, out)


def _render_markdown(fields: list[str], headers: dict[str, str],
                     records: list[RunRecord] | list[BenchRecord], out,
                     plain: bool = False) -> None:
    print("<table>", file=out)
    print("  <thead>", file=out)
    print("    <tr>", file=out)
    for field in fields:
        print(f"      <th>{html.escape(headers.get(field, field))}</th>", file=out)
    print("    </tr>", file=out)
    print("  </thead>", file=out)
    print("  <tbody>", file=out)
    for record in records:
        row = record.public_dict()
        print("    <tr>", file=out)
        for field in fields:
            if plain:
                value = _format_plain_value(record, field, row.get(field, ""))  # type: ignore[arg-type]
            else:
                value = _format_value(field, row.get(field, ""))
            print(f"      <td>{html.escape(value)}</td>", file=out)
        print("    </tr>", file=out)
    print("  </tbody>", file=out)
    print("</table>", file=out)


def write_markdown(records: list[RunRecord], out) -> None:
    _render_markdown(FIELDNAMES, EXPORT_HEADERS, records, out, plain=True)


def write_bench_markdown(records: list[BenchRecord], out) -> None:
    _render_markdown(BENCH_FIELDNAMES, {f: f for f in BENCH_FIELDNAMES}, records, out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parse DartMoQ slurm logs without positional grep/awk alignment."
    )
    parser.add_argument("logs", nargs="+", help="slurm log files to parse")
    parser.add_argument(
        "--format", "-f", choices=("plain", "csv", "json", "md"), default="plain",
        help="output format. plain prints to stdout; csv/json/md write to <logfile>.<format>.",
    )
    parser.add_argument(
        "--complete-only", action="store_true",
        help="only emit rows whose status is ok",
    )
    args = parser.parse_args(argv)

    if args.format == "plain":
        # bench 日志与量化/eval 日志逐文件自动识别，两类各渲一张表
        bench_records: list[BenchRecord] = []
        run_records: list[RunRecord] = []
        for log_path in args.logs:
            if is_bench_log(log_path):
                bench_records.extend(parse_bench_log(log_path))
            else:
                run_records.extend(parse_log(log_path))
        if args.complete_only:
            bench_records = [r for r in bench_records if r.status == "ok"]
            run_records = [r for r in run_records if r.status == "ok"]
        if bench_records:
            write_bench_plain(bench_records, sys.stdout)
            print()
        if run_records:
            write_plain(run_records, sys.stdout)
        return 0

    for log_path in args.logs:
        bench_mode = is_bench_log(log_path)
        records = parse_bench_log(log_path) if bench_mode else parse_log(log_path)
        total_records = len(records)
        if args.complete_only:
            records = [record for record in records if record.status == "ok"]
        filtered_records = len(records)

        output_path = f"{log_path}.{args.format}"
        with open(output_path, "w", newline="", encoding="utf-8") as out:
            if args.format == "json":
                if bench_mode:
                    json.dump([r.public_dict() for r in records], out, indent=2, ensure_ascii=False)
                else:
                    json.dump([_format_export_row(record) for record in records], out, indent=2, ensure_ascii=False)
                print(file=out)
            elif args.format == "md":
                if bench_mode:
                    write_bench_markdown(records, out)
                else:
                    write_markdown(records, out)
            else:
                if bench_mode:
                    rows = [r.public_dict() for r in records]
                    out.write(",".join([_csv_escape(f) for f in BENCH_FIELDNAMES]) + "\n")
                    for row in rows:
                        out.write(",".join([_csv_escape(_format_value(f, row.get(f, "")))
                                            for f in BENCH_FIELDNAMES]) + "\n")
                else:
                    write_csv(records, out)

        # Print output log
        print(f"[LOG] Parsed: {log_path}" + (" (bench)" if bench_mode else ""))
        print(f"[LOG] Output: {output_path}")
        print(f"[LOG] Total runs: {total_records}")
        if args.complete_only:
            print(f"[LOG] Filtered to {filtered_records} complete runs (status='ok')")
        else:
            status_counts = {}
            for r in records:
                status_counts[r.status] = status_counts.get(r.status, 0) + 1
            status_str = ", ".join([f"{k}: {v}" for k, v in sorted(status_counts.items())])
            print(f"[LOG] Status breakdown: {status_str}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
