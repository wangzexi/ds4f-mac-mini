#!/usr/bin/env python3
"""Verify the fixed Q2-core candidate against its Q4/IQ2 template."""

from __future__ import annotations

import argparse
from pathlib import Path

from verify_routed_copy import load


Q2_K = 10
Q4_K = 12
EXPECTED_Q2_CHANGES = 258


def expected_q2(name: str) -> bool:
    if "_shexp.weight" in name:
        return True
    if ".indexer." in name or "indexer_" in name:
        return False
    return name.endswith((
        ".attn_kv.weight",
        ".attn_q_a.weight",
        ".attn_q_b.weight",
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
    for name, (source_dims, source_type, _, source_size) in template.items():
        output_dims, output_type, _, output_size = output[name]
        if source_dims != output_dims:
            raise SystemExit(f"shape changed: {name}")
        if expected_q2(name):
            if source_type != Q4_K or output_type != Q2_K:
                raise SystemExit(
                    f"bad Q2-core transition: {name} source={source_type} output={output_type}"
                )
            if output_size >= source_size:
                raise SystemExit(
                    f"Q2-core tensor did not shrink: {name} source={source_size} output={output_size}"
                )
            changed.append(name)
        elif (source_type, source_size) != (output_type, output_size):
            raise SystemExit(
                f"unexpected tensor change: {name} source=({source_type}, {source_size}) "
                f"output=({output_type}, {output_size})"
            )

    if len(changed) != EXPECTED_Q2_CHANGES:
        raise SystemExit(f"expected {EXPECTED_Q2_CHANGES} Q2-core tensors, found {len(changed)}")
    if args.output.stat().st_size >= args.template.stat().st_size:
        raise SystemExit(
            f"candidate did not shrink: output={args.output.stat().st_size} "
            f"template={args.template.stat().st_size}"
        )
    print(
        f"q2_core_model_contract: OK tensors={len(output)} q2_changes={len(changed)} "
        f"bytes={args.output.stat().st_size}"
    )


if __name__ == "__main__":
    main()
