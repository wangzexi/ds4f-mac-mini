#!/usr/bin/env python3
"""Summarize the fixed-Mini exact selected-expert I/O matrix logs."""

from __future__ import annotations

import argparse
import pathlib
import re
import statistics


WALL = re.compile(r"^\s*([0-9.]+) real")
PREAD = re.compile(r"load_pread_avg=([0-9.]+) ms")
CACHE = re.compile(
    r"streaming expert cache .*?hits=(\d+) misses=(\d+) hit_rate=([0-9.]+)"
)
GPU = re.compile(r"gpu busy total=([0-9.]+) ms command_buffers=(\d+)")
THROUGHPUT = re.compile(r"ds4: prefill: ([0-9.]+) t/s, generation: ([0-9.]+) t/s")
DECODE_EVAL = re.compile(r"ds4: gpu decode eval \d+ took ([0-9.]+) ms")


def find(pattern: re.Pattern[str], text: str, group: int = 1) -> float | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    match = matches[-1]
    value = match[group - 1] if isinstance(match, tuple) else match
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=pathlib.Path)
    args = parser.parse_args()
    rows: dict[int, list[dict[str, float]]] = {}
    for path in sorted(args.directory.glob("pread-*-run-*.log")):
        match = re.match(r"pread-(\d+)-run-\d+\.log", path.name)
        if not match:
            continue
        text = path.read_text(errors="replace")
        hit = CACHE.findall(text)
        item: dict[str, float] = {}
        if (value := find(WALL, text)) is not None:
            item["wall_s"] = value
        if (value := find(PREAD, text)) is not None:
            item["pread_avg_ms"] = value
        if hit:
            h, m, rate = hit[-1]
            item["hits"] = float(h)
            item["misses"] = float(m)
            item["hit_rate"] = float(rate)
        if (value := find(GPU, text)) is not None:
            item["gpu_busy_ms"] = value
        if (throughput := THROUGHPUT.findall(text)):
            prefill_tps, generation_tps = throughput[-1]
            item["prefill_tps"] = float(prefill_tps)
            item["generation_tps"] = float(generation_tps)
        decode_eval_ms = [float(value) for value in DECODE_EVAL.findall(text)]
        if decode_eval_ms:
            item["decode_eval_ms_p50"] = statistics.median(decode_eval_ms)
            item["decode_eval_ms_p95"] = (
                sorted(decode_eval_ms)[max(0, (len(decode_eval_ms) * 95 + 99) // 100 - 1)]
            )
        rows.setdefault(int(match.group(1)), []).append(item)

    print(
        "pread_threads\truns\tgeneration_tps_p50\tdecode_eval_ms_p50\t"
        "decode_eval_ms_p95\tprefill_tps_p50\tpread_avg_ms_p50\thit_rate_p50\t"
        "misses_p50\tgpu_busy_ms_p50\twall_s_p50"
    )
    for threads, samples in sorted(rows.items()):
        def median(key: str) -> str:
            values = [sample[key] for sample in samples if key in sample]
            return f"{statistics.median(values):.3f}" if values else "-"

        print(
            f"{threads}\t{len(samples)}\t{median('generation_tps')}\t"
            f"{median('decode_eval_ms_p50')}\t{median('decode_eval_ms_p95')}\t"
            f"{median('prefill_tps')}\t{median('pread_avg_ms')}\t"
            f"{median('hit_rate')}\t{median('misses')}\t{median('gpu_busy_ms')}\t"
            f"{median('wall_s')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
