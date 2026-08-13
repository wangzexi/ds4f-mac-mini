#!/usr/bin/env python3
"""Summarize exact I/O measurements made through the actual Mini server."""

from __future__ import annotations

import argparse
import pathlib
import re
import statistics


PREAD = re.compile(r"load_pread_avg=([0-9.]+) ms")
CACHE = re.compile(r"streaming expert cache .*?hits=(\d+) misses=(\d+) hit_rate=([0-9.]+)")
GPU = re.compile(r"gpu busy total=([0-9.]+) ms command_buffers=(\d+)")
PREFILL = re.compile(r"prompt done ([0-9.]+)s")
DECODE = re.compile(r"decoding chunk=[0-9.]+ t/s avg=([0-9.]+) t/s ([0-9.]+)s")
TOTAL = re.compile(r"finish=[^ ]+ ([0-9.]+)s")
PHASE_MARK = re.compile(r"Metal memory (server exact (?:prefill|decode) phase):")
TIMING_DELTA = re.compile(
    r"streaming expert timing delta .*?load_calls=(\d+).*?load_pread_avg=([0-9.]+) ms"
    r".*?cache_all_resident=(\d+) cache_all_missing=(\d+) cache_mixed=(\d+)"
)


def last_float(pattern: re.Pattern[str], text: str, group: int = 1) -> float | None:
    found = pattern.findall(text)
    if not found:
        return None
    value = found[-1]
    if isinstance(value, tuple):
        value = value[group - 1]
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=pathlib.Path)
    args = parser.parse_args()
    rows: dict[int, list[dict[str, float]]] = {}
    for path in sorted(args.directory.glob("server-pread-*-run-*.log")):
        match = re.match(r"server-pread-(\d+)-run-\d+\.log", path.name)
        if not match:
            continue
        text = path.read_text(errors="replace")
        row: dict[str, float] = {}
        if (value := last_float(PREFILL, text)) is not None:
            row["prefill_s"] = value
        if (value := last_float(PREAD, text)) is not None:
            row["pread_avg_ms"] = value
        if (value := last_float(GPU, text)) is not None:
            row["gpu_busy_ms"] = value
        if (value := last_float(TOTAL, text)) is not None:
            row["total_s"] = value
        decode = DECODE.findall(text)
        if decode:
            row["decode_tps"] = float(decode[-1][0])
            row["decode_s"] = float(decode[-1][1])
        cache = CACHE.findall(text)
        if cache:
            hits, misses, rate = cache[-1]
            row["hits"] = float(hits)
            row["misses"] = float(misses)
            row["hit_rate"] = float(rate)
        phase = None
        for line in text.splitlines():
            mark = PHASE_MARK.search(line)
            if mark:
                phase = mark.group(1)
                continue
            timing = TIMING_DELTA.search(line)
            if phase and timing:
                loads, pread, all_resident, all_missing, mixed = timing.groups()
                prefix = "prefill" if "prefill" in phase else "decode"
                row[f"{prefix}_load_calls"] = float(loads)
                row[f"{prefix}_pread_avg_ms"] = float(pread)
                row[f"{prefix}_all_resident_layers"] = float(all_resident)
                row[f"{prefix}_all_missing_layers"] = float(all_missing)
                row[f"{prefix}_mixed_layers"] = float(mixed)
                phase = None
        rows.setdefault(int(match.group(1)), []).append(row)

    print(
        "pread_threads\truns\tdecode_tps_p50\tdecode_s_p50\tprefill_s_p50\t"
        "pread_avg_ms_p50\thit_rate_p50\t"
        "misses_p50\tgpu_busy_ms_p50\ttotal_s_p50"
    )
    for threads, samples in sorted(rows.items()):
        def median(key: str) -> str:
            values = [sample[key] for sample in samples if key in sample]
            return f"{statistics.median(values):.3f}" if values else "-"

        print(
            f"{threads}\t{len(samples)}\t{median('decode_tps')}\t{median('decode_s')}\t"
            f"{median('prefill_s')}\t{median('pread_avg_ms')}\t{median('hit_rate')}\t"
            f"{median('misses')}\t{median('gpu_busy_ms')}\t{median('total_s')}"
        )
        print(
            "  phase delta: "
            f"prefill loads={median('prefill_load_calls')} "
            f"decode loads={median('decode_load_calls')} "
            f"decode resident/all-miss/mixed="
            f"{median('decode_all_resident_layers')}/"
            f"{median('decode_all_missing_layers')}/"
            f"{median('decode_mixed_layers')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
