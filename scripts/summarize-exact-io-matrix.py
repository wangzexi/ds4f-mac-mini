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
        rows.setdefault(int(match.group(1)), []).append(item)

    print("pread_threads\truns\twall_s_p50\tpread_avg_ms_p50\thit_rate_p50\tmisses_p50\tgpu_busy_ms_p50")
    for threads, samples in sorted(rows.items()):
        def median(key: str) -> str:
            values = [sample[key] for sample in samples if key in sample]
            return f"{statistics.median(values):.3f}" if values else "-"

        print(
            f"{threads}\t{len(samples)}\t{median('wall_s')}\t{median('pread_avg_ms')}\t"
            f"{median('hit_rate')}\t{median('misses')}\t{median('gpu_busy_ms')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
