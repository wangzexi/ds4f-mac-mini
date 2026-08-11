#!/usr/bin/env python3
"""Summarize the fixed-Mini prefill measurement TSV."""

from __future__ import annotations

import collections
import pathlib
import statistics
import sys


def main() -> int:
    path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "cache/prefill-measurements.tsv")
    if not path.is_file():
        print(f"measurement table not found: {path}", file=sys.stderr)
        return 2

    groups: dict[tuple[str, int], list[tuple[int, float, float]]] = collections.defaultdict(list)
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) != 8:
            print(f"ignoring malformed line {lineno}", file=sys.stderr)
            continue
        _epoch, _pid, kind, layer, units, size, read_ms, wait_ms = fields
        groups[(kind, int(layer))].append((int(size), float(read_ms), float(wait_ms)))

    print("kind\tlayer\tsamples\tMiB\tread_ms_p50\twait_ms_p50\twait_ms_max")
    for (kind, layer), rows in sorted(groups.items()):
        sizes = [row[0] for row in rows]
        reads = [row[1] for row in rows]
        waits = [row[2] for row in rows]
        print(
            f"{kind}\t{layer}\t{len(rows)}\t{statistics.median(sizes) / 1048576:.2f}"
            f"\t{statistics.median(reads):.3f}\t{statistics.median(waits):.3f}"
            f"\t{max(waits):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
