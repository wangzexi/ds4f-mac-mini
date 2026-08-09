#!/usr/bin/env python3
"""Verify the fixed DS4F-Mini Q4 tensor-type and file-size contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from verify_routed_copy import load


EXPECTED_OUTPUT_BYTES = 82_853_553_024
Q8_0 = 8
Q4_K = 12
F16 = 1


def expected_q4(name: str) -> bool:
    if name in {"token_embd.weight", "output.weight"}:
        return True
    if "_shexp.weight" in name:
        return True
    if ".indexer." in name or "indexer_" in name:
        return False
    return name.endswith((
        ".attn_kv.weight",
        ".attn_q_a.weight",
        ".attn_q_b.weight",
        ".attn_output_a.weight",
        ".attn_output_b.weight",
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    template = load(args.template)
    output = load(args.output)
    if set(template) != set(output):
        missing = sorted(set(template) - set(output))
        extra = sorted(set(output) - set(template))
        raise SystemExit(f"tensor name mismatch: missing={missing[:3]} extra={extra[:3]}")

    changed = []
    for name, (source_dims, source_type, _, _) in template.items():
        output_dims, output_type, _, _ = output[name]
        if source_dims != output_dims:
            raise SystemExit(f"shape changed: {name}")
        if expected_q4(name):
            wanted_source = F16 if name == "token_embd.weight" else Q8_0
            if source_type != wanted_source or output_type != Q4_K:
                raise SystemExit(
                    f"bad Q4 transition: {name} source={source_type} output={output_type}"
                )
            changed.append(name)
        elif source_type != output_type:
            raise SystemExit(
                f"unexpected type change: {name} source={source_type} output={output_type}"
            )

    if len(changed) != 346:
        raise SystemExit(f"expected 346 Q4 tensors, found {len(changed)}")
    if args.output.stat().st_size != EXPECTED_OUTPUT_BYTES:
        raise SystemExit(
            f"bad output size: {args.output.stat().st_size}, expected {EXPECTED_OUTPUT_BYTES}"
        )
    print(
        f"mini_model_contract: OK tensors={len(output)} q4_changes={len(changed)} "
        f"bytes={args.output.stat().st_size}"
    )


if __name__ == "__main__":
    main()
