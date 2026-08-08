#!/usr/bin/env python3
"""Compare DwarfStar --dump-logprobs JSON with a DS4F_TRACE_TOP log."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TRACE_RE = re.compile(r"^trace stage=(?P<stage>\w+) step=(?P<step>-?\d+)$")
TOP_RE = re.compile(r"^top\[(?P<rank>\d+)\]: token=(?P<id>\d+) logit=(?P<logit>\S+)$")


def parse_self_trace(path: Path) -> list[list[tuple[int, float]]]:
    """Return prefill/decode traces in the order they predict next tokens."""
    traces: list[list[tuple[int, float]]] = []
    current: list[tuple[int, float]] | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if TRACE_RE.match(raw):
            current = []
            traces.append(current)
            continue
        match = TOP_RE.match(raw)
        if match and current is not None:
            current.append((int(match["id"]), float(match["logit"])))
    return traces


def token_label(entry: dict) -> str:
    token = entry["token"]
    return f'{token["id"]} ({token["text"]!r})'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dwarfstar_json", type=Path)
    parser.add_argument("self_trace", type=Path)
    args = parser.parse_args()

    dwarf = json.loads(args.dwarfstar_json.read_text(encoding="utf-8"))
    expected_steps = dwarf["steps"]
    observed = parse_self_trace(args.self_trace)
    compared = min(len(expected_steps), len(observed))
    print(f"DwarfStar steps={len(expected_steps)}; self traces={len(observed)}; compared={compared}")
    if not compared:
        print("No comparable self traces found.")
        return 2

    first_top1 = None
    first_ranking = None
    for step in range(compared):
        expected_top = expected_steps[step]["top_logprobs"]
        actual_top = observed[step]
        expected_ids = [item["token"]["id"] for item in expected_top]
        actual_ids = [token_id for token_id, _ in actual_top]
        if first_top1 is None and actual_ids and expected_ids and actual_ids[0] != expected_ids[0]:
            first_top1 = step
        if first_ranking is None and expected_ids != actual_ids:
            first_ranking = step

    print(f"first top-k ranking mismatch: {first_ranking if first_ranking is not None else 'none'}")
    print(f"first selected-token mismatch: {first_top1 if first_top1 is not None else 'none'}")

    report_steps = {0}
    if first_top1 is not None:
        report_steps.add(first_top1)
    if first_ranking is not None:
        report_steps.add(first_ranking)
    for step in sorted(report_steps):
        expected_top = expected_steps[step]["top_logprobs"]
        actual_top = observed[step]
        expected_by_id = {item["token"]["id"]: item["logit"] for item in expected_top}
        actual_by_id = dict(actual_top)
        print(f"\nstep {step}:")
        print(f"  DwarfStar top-1: {token_label(expected_top[0])}, logit={expected_top[0]['logit']:.7g}")
        print(f"  self top-1:      {actual_top[0][0]}, logit={actual_top[0][1]:.7g}")
        for token_id in (expected_top[0]["token"]["id"], actual_top[0][0]):
            if token_id in expected_by_id and token_id in actual_by_id:
                delta = actual_by_id[token_id] - expected_by_id[token_id]
                print(f"  token {token_id} self-minus-DwarfStar={delta:+.7g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
