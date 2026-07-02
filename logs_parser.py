#!/usr/bin/env python3
"""Parse DartMoQ slurm logs into aligned benchmark rows.

The parser scans each log sequentially and starts a new row at every
"Loading model: (ppl) ..." line. Metrics that are missing because a run crashed
or skipped downstream evaluation are left empty instead of being shifted onto a
neighboring row.
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
    "model_name", "slices", "quant_scheme", "rank_mode",
    "moe_struct", "quantmode", "disable_0bit_prune", "standby_layer_cpu", "bpw", "ppl_wikitext2", "ppl_c4",
    *TASK_FIELDS,
    "status", "runtime_ppl", "runtime_quant", "runtime_ppl_eval", "runtime_zero_eval",
]

LOADING_RE = re.compile(r"Loading model:\s*\(ppl\)\s*(?P<path>\S+)")
QUANTMODE_RE = re.compile(
    r"slices/quant-scheme/rank-mode/moe-struct/quantmode(?:/disable-0bit-prune)?(?:/standby-layer-cpu)?:\s*\(ppl\)\s+"
    r"(?P<slices>\S+)\s+(?P<quant_scheme>\S+)\s+(?P<rank_mode>\S+)\s+"
    r"(?P<moe_struct>\S+)\s+(?P<quantmode>\S+)(?:\s+(?P<disable_0bit_prune>\S+))?(?:\s+(?P<standby_layer_cpu>\S+))?"
)
BPW_RE = re.compile(r"\bwith bpw\s+(?P<bpw>[-+0-9.eE]+)")
PPL_RE = re.compile(r"ppl on (?P<dataset>wikitext2|c4)(?:\s+\([^)]+\))?:\s*(?P<value>[-+0-9.eE]+)")
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

# For backward compatibility with old logs
LAYER_TIME_RE = re.compile(r"Layer (?P<layer>\d+) total reconstruct and quantization time:\s*(?P<value>[-+0-9.eE]+) s")
PPL_INDIVIDUAL_TIME_RE = re.compile(r"ppl on (?P<dataset>wikitext2|c4)(?:\s+\([^)]+\))?:\s*[-+0-9.eE]+\s*time:\s*(?P<value>[-+0-9.eE]+)")
ZERO_EVAL_OLD_RE = re.compile(r"Zero-shot evaluation time:\s*(?P<value>[-+0-9.eE]+)")


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
    disable_0bit_prune: str = ""
    standby_layer_cpu: str = ""
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
        if self._fatal:
            self.status = "failed"
        elif self.ppl_wikitext2 and self.ppl_c4:
            # Have both ppl results
            missing = [name for name in TASK_FIELDS if not getattr(self, name)]
            if missing:
                self.status = "partial"
            else:
                self.status = "ok"
        elif self.runtime_ppl or self.mmlu_acc:
            # Have other results but not both ppl
            missing = [name for name in TASK_FIELDS if not getattr(self, name)]
            if missing:
                self.status = "partial"
            else:
                self.status = "ok"
        else:
            # Have only one ppl or nothing at all
            self.status = "incomplete"

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

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, raw_line in enumerate(f, 1):
            last_line_no = line_no
            line = _clean_line(raw_line)
            if not line:
                continue

            start_match = START_TIME_RE.search(line)
            if start_match:
                pending_start_time = start_match.group("value").strip()

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
                continue

            if current is None:
                continue

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

            # Backward compatibility: parse layer times for old logs
            layer_time_match = LAYER_TIME_RE.search(line)
            if layer_time_match:
                try:
                    t = float(layer_time_match.group("value"))
                    current._layer_times.append(t)
                except ValueError:
                    pass
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
    "model_name", "slices", "quant_scheme", "rank_mode", "quantmode", "bpw",
    "ppl_wikitext2", "ppl_c4",
    "arc_c_acc", "arc_c_acc_norm", "arc_e_acc", "arc_e_acc_norm",
    "piqa_acc", "piqa_acc_norm", "boolq_acc", "wino_acc", "mnli_acc",
    "hella_acc", "hella_acc_norm", "mmlu_acc", "status",
    "runtime_ppl", "runtime_quant", "runtime_ppl_eval", "runtime_zero_eval", "error",
]

PLAIN_HEADERS = {
    "model_name": "model",
    "slices": "sli",
    "quant_scheme": "qsch",
    "rank_mode": "rank",
    "moe_struct": "moe",
    "quantmode": "qmode",
    "disable_0bit_prune": "no0prune",
    "standby_layer_cpu": "stdbycpu",
    "bpw": "bpw",
    "ppl_wikitext2": "wiki",
    "ppl_c4": "c4",
    "arc_c_acc": "arc_c",
    "arc_c_acc_norm": "arc_c_n",
    "arc_e_acc": "arc_e",
    "arc_e_acc_norm": "arc_e_n",
    "piqa_acc": "piqa",
    "piqa_acc_norm": "piqa_n",
    "boolq_acc": "boolq",
    "wino_acc": "wino",
    "mnli_acc": "mnli",
    "hella_acc": "hella",
    "hella_acc_norm": "hella_n",
    "mmlu_acc": "mmlu",
    "status": "status",
    "runtime_ppl": "time",
    "runtime_quant": "t_quant",
    "runtime_ppl_eval": "t_ppl",
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
    "disable_0bit_prune": "disable_0bit_prune",
    "standby_layer_cpu": "standby_layer_cpu",
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
    "runtime_zero_eval": "zero_eval_time",
}


def _format_value(field: str, value) -> str:
    if field == "error":
        return ""
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


def write_plain(records: list[RunRecord], out) -> None:
    rows: list[list[str]] = []
    rows.append([PLAIN_HEADERS[field] for field in DISPLAY_FIELDS])
    for record in records:
        row = record.public_dict()
        rows.append([_format_plain_value(record, col, row.get(col, "")) for col in DISPLAY_FIELDS])

    widths = [max(len(row[i]) for row in rows) for i in range(len(DISPLAY_FIELDS))]
    for row in rows:
        pieces = []
        for i, value in enumerate(row):
            pieces.append(value)
            if i != len(row) - 1:
                padding = " " * (widths[i] - len(value) + 2)
                pieces.append(padding)
        print("".join(pieces), file=out)


def write_markdown(records: list[RunRecord], out) -> None:
    print("<table>", file=out)
    print("  <thead>", file=out)
    print("    <tr>", file=out)
    for field in FIELDNAMES:
        print(f"      <th>{html.escape(EXPORT_HEADERS[field])}</th>", file=out)
    print("    </tr>", file=out)
    print("  </thead>", file=out)
    print("  <tbody>", file=out)
    for record in records:
        row = record.public_dict()
        print("    <tr>", file=out)
        for field in FIELDNAMES:
            value = _format_plain_value(record, field, row.get(field, ""))
            print(f"      <td>{html.escape(value)}</td>", file=out)
        print("    </tr>", file=out)
    print("  </tbody>", file=out)
    print("</table>", file=out)


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
        records = parse_logs(args.logs)
        if args.complete_only:
            records = [record for record in records if record.status == "ok"]
        write_plain(records, sys.stdout)
        return 0

    for log_path in args.logs:
        records = parse_log(log_path)
        total_records = len(records)
        if args.complete_only:
            records = [record for record in records if record.status == "ok"]
        filtered_records = len(records)

        output_path = f"{log_path}.{args.format}"
        with open(output_path, "w", newline="", encoding="utf-8") as out:
            if args.format == "json":
                json.dump([_format_export_row(record) for record in records], out, indent=2, ensure_ascii=False)
                print(file=out)
            elif args.format == "md":
                write_markdown(records, out)
            else:
                write_csv(records, out)

        # Print output log
        print(f"[LOG] Parsed: {log_path}")
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
