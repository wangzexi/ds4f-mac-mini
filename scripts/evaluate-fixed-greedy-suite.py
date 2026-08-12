#!/usr/bin/env python3
"""Run the fixed Mini greedy suite and compare a candidate to a Q4 baseline.

This is intentionally specific to ds4f-mini: each prompt opens the fixed
Metal/32K runner in a clean process, and candidate sidecar environment is
inherited unchanged.  It evaluates deterministic token IDs rather than
pretending that a small prompt set is a full quality benchmark.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path


TRACE = re.compile(r"trace token\[\d+\]=(\d+)")


def load_prompts(path: Path) -> list[str]:
    prompts = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            prompts.append(line)
    if len(prompts) != 16:
        raise SystemExit(f"expected exactly 16 fixed prompts, found {len(prompts)}")
    return prompts


def parse_tokens(stderr: str, count: int) -> list[int]:
    tokens = [int(value) for value in TRACE.findall(stderr)]
    if len(tokens) != count:
        raise RuntimeError(
            f"runner emitted {len(tokens)} traced tokens instead of {count}: {stderr[-800:]}"
        )
    return tokens


def lcp(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def read_reference(path: Path) -> dict[str, list[int]]:
    reference: dict[str, list[int]] = {}
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            reference[row["id"]] = [int(value) for value in row["token_ids"].split(",")]
    return reference


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=root / "results/quality/fixed-mini-greedy-prompts.txt",
    )
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    if args.tokens < 1 or args.repeat < 1:
        raise SystemExit("--tokens and --repeat must both be positive")
    if not args.runner.is_file() or not args.model.is_file():
        raise SystemExit("runner or model is unavailable")

    prompts = load_prompts(args.prompts)
    reference = read_reference(args.reference) if args.reference else {}
    if reference and set(reference) != {f"case_{index:02d}" for index in range(len(prompts))}:
        raise SystemExit("reference suite does not match the fixed prompt set")
    if any(len(tokens) != args.tokens for tokens in reference.values()):
        raise SystemExit("reference token count does not match --tokens")

    rows: list[dict[str, str]] = []
    for index, prompt in enumerate(prompts):
        case_id = f"case_{index:02d}"
        observed: list[list[int]] = []
        elapsed: list[float] = []
        for attempt in range(args.repeat):
            started = time.monotonic()
            result = subprocess.run(
                [str(args.runner), str(args.model), prompt, str(args.tokens)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                env={**os.environ, "DS4F_FAST_TRACE_IDS": "1"},
            )
            elapsed.append(time.monotonic() - started)
            if result.returncode:
                raise RuntimeError(f"{case_id} attempt {attempt + 1} failed: {result.stderr[-1200:]}")
            observed.append(parse_tokens(result.stderr, args.tokens))
        deterministic = all(tokens == observed[0] for tokens in observed[1:])
        if not deterministic:
            raise RuntimeError(f"{case_id} is not deterministic across {args.repeat} clean runs")

        base = reference.get(case_id, [])
        prefix = lcp(base, observed[0]) if base else 0
        rows.append({
            "id": case_id,
            "prompt": prompt,
            "token_ids": ",".join(map(str, observed[0])),
            "seconds": f"{sum(elapsed) / len(elapsed):.3f}",
            "reference_lcp": str(prefix) if base else "",
            "reference_first_match": str(int(bool(base) and base[0] == observed[0][0])) if base else "",
        })
        print(
            f"{case_id}: {sum(elapsed) / len(elapsed):.2f}s ids={rows[-1]['token_ids']}"
            + (f" q4_lcp={prefix}/{args.tokens}" if base else ""),
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    mean_seconds = sum(float(row["seconds"]) for row in rows) / len(rows)
    if reference:
        matched_first = sum(int(row["reference_first_match"]) for row in rows)
        matched_prefix = sum(int(row["reference_lcp"]) for row in rows)
        print(
            f"summary: cases={len(rows)} mean_seconds={mean_seconds:.3f} "
            f"q4_first_match={matched_first}/{len(rows)} "
            f"q4_token_lcp={matched_prefix}/{len(rows) * args.tokens}"
        )
    else:
        print(f"summary: cases={len(rows)} mean_seconds={mean_seconds:.3f} reference=written")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"fixed greedy suite failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
