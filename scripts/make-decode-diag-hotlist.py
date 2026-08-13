#!/usr/bin/env python3
"""Turn a recorded first-decode route trace into a post-prefill seed list.

The binary trace contains one little-endian six-int32 record per routed MoE
layer execution.  A run with N output tokens executes prompt_layers plus
(N - 1) decode-layer groups, so the final `--layers` records are precisely the
first decode token.  The output is the normal `layer expert priority` hotlist
format, but it is consumed only by the diagnostic post-prefill seed switch.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


EXPERTS_PER_RECORD = 6
RECORD_BYTES = EXPERTS_PER_RECORD * 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layers", type=int, default=43)
    args = parser.parse_args()

    if args.layers <= 0:
        parser.error("--layers must be positive")
    raw = args.trace.read_bytes()
    if len(raw) % RECORD_BYTES:
        parser.error(f"{args.trace} has {len(raw)} bytes, not a whole selected-ID trace")
    n_records = len(raw) // RECORD_BYTES
    if n_records < args.layers:
        parser.error(f"{args.trace} has only {n_records} records; need {args.layers}")

    records = struct.iter_unpack("<6i", raw)
    tail = list(records)[-args.layers:]
    lines = [
        "# Exact diagnostic seed: final records of first-decode selected-ID trace.",
        f"# records={n_records} decode_layers={args.layers}",
    ]
    n_entries = 0
    for layer, ids in enumerate(tail):
        for expert in sorted(set(ids)):
            if expert < 0:
                parser.error(f"negative selected expert in layer {layer}")
            lines.append(f"{layer} {expert} 1")
            n_entries += 1
    args.output.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"wrote {n_entries} exact selected-expert seeds across {args.layers} layers to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
