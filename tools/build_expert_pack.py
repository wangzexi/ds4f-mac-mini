#!/usr/bin/env python3
"""Pack each routed expert's gate/up/down payload into one contiguous record."""

from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path


VALUE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
LAYERS = 43
EXPERTS = 256
DATA_OFFSET = 4096


def read_u32(fp):
    return struct.unpack("<I", fp.read(4))[0]


def read_u64(fp):
    return struct.unpack("<Q", fp.read(8))[0]


def read_string(fp):
    return fp.read(read_u64(fp)).decode("utf-8")


def skip_value(fp, kind):
    if kind == 8:
        fp.seek(read_u64(fp), 1)
    elif kind == 9:
        element_kind = read_u32(fp)
        count = read_u64(fp)
        if element_kind == 8:
            for _ in range(count):
                fp.seek(read_u64(fp), 1)
        else:
            fp.seek(VALUE_SIZES[element_kind] * count, 1)
    else:
        fp.seek(VALUE_SIZES[kind], 1)


def load_spans(path: Path):
    with path.open("rb") as fp:
        if fp.read(4) != b"GGUF":
            raise ValueError(f"not GGUF: {path}")
        read_u32(fp)
        tensor_count = read_u64(fp)
        metadata_count = read_u64(fp)
        alignment = 32
        for _ in range(metadata_count):
            key = read_string(fp)
            kind = read_u32(fp)
            if key == "general.alignment" and kind == 4:
                alignment = read_u32(fp)
            else:
                skip_value(fp, kind)
        tensors = []
        for _ in range(tensor_count):
            name = read_string(fp)
            dims = tuple(read_u64(fp) for _ in range(read_u32(fp)))
            tensor_type = read_u32(fp)
            offset = read_u64(fp)
            tensors.append((name, dims, tensor_type, offset))
        data_start = (fp.tell() + alignment - 1) // alignment * alignment
    ordered = sorted(tensors, key=lambda row: row[3])
    spans = {}
    for index, (name, dims, tensor_type, offset) in enumerate(ordered):
        next_offset = ordered[index + 1][3] if index + 1 < len(ordered) else path.stat().st_size - data_start
        spans[name] = (dims, tensor_type, data_start + offset, next_offset - offset)
    return spans


def routed_name(layer: int, projection: str) -> str:
    return f"blk.{layer}.ffn_{projection}_exps.weight"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    spans = load_spans(args.model)
    rows = []
    gate_bytes = None
    down_bytes = None
    for layer in range(LAYERS):
        layer_rows = []
        for projection in ("gate", "up", "down"):
            name = routed_name(layer, projection)
            if name not in spans:
                raise SystemExit(f"missing routed tensor: {name}")
            dims, tensor_type, offset, size = spans[name]
            if not dims or dims[-1] != EXPERTS or size % EXPERTS:
                raise SystemExit(f"unexpected routed tensor layout: {name} dims={dims} size={size}")
            layer_rows.append((name, offset, size // EXPERTS, tensor_type))
        if layer_rows[0][2] != layer_rows[1][2]:
            raise SystemExit(f"gate/up expert size mismatch at layer {layer}")
        if gate_bytes is None:
            gate_bytes = layer_rows[0][2]
            down_bytes = layer_rows[2][2]
        if layer_rows[0][2] != gate_bytes or layer_rows[2][2] != down_bytes:
            raise SystemExit(f"mixed expert size class at layer {layer}")
        rows.append(layer_rows)

    assert gate_bytes is not None and down_bytes is not None
    slot_bytes = gate_bytes * 2 + down_bytes
    expected = DATA_OFFSET + LAYERS * EXPERTS * slot_bytes
    if args.dry_run:
        print(
            f"expert_pack: DRY-RUN layers={LAYERS} experts={EXPERTS} "
            f"gate_bytes={gate_bytes} down_bytes={down_bytes} "
            f"slot_bytes={slot_bytes} bytes={expected}"
        )
        return
    if args.verify_only:
        if args.output.stat().st_size != expected:
            raise SystemExit("packed size mismatch before verification")
        with args.model.open("rb", buffering=0) as source, args.output.open("rb", buffering=0) as packed:
            header = packed.read(48)
            expected_header = struct.pack(
                "<8sQIIQQQ",
                b"DS4FPK1\0",
                args.model.stat().st_size,
                LAYERS,
                EXPERTS,
                gate_bytes,
                down_bytes,
                DATA_OFFSET,
            )
            if header != expected_header:
                raise SystemExit("packed header mismatch")
            for layer, layer_rows in enumerate(rows):
                expert = (layer * 37 + 11) % EXPERTS
                packed.seek(DATA_OFFSET + (layer * EXPERTS + expert) * slot_bytes)
                actual = packed.read(slot_bytes)
                expected_parts = []
                for _name, base, expert_bytes, _tensor_type in layer_rows:
                    source.seek(base + expert * expert_bytes)
                    expected_parts.append(source.read(expert_bytes))
                if actual != b"".join(expected_parts):
                    raise SystemExit(f"packed payload mismatch layer={layer} expert={expert}")
                print(f"verified layer {layer + 1:02d}/{LAYERS} expert {expert:03d}", flush=True)
        print("expert_pack_verify: OK sampled_layers=43")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack(
        "<8sQIIQQQ",
        b"DS4FPK1\0",
        args.model.stat().st_size,
        LAYERS,
        EXPERTS,
        gate_bytes,
        down_bytes,
        DATA_OFFSET,
    )

    with args.model.open("rb", buffering=0) as source, args.output.open("wb", buffering=0) as output:
        output.write(header)
        output.write(bytes(DATA_OFFSET - len(header)))
        for layer, layer_rows in enumerate(rows):
            for expert in range(EXPERTS):
                for _name, base, expert_bytes, _tensor_type in layer_rows:
                    source.seek(base + expert * expert_bytes)
                    payload = source.read(expert_bytes)
                    if len(payload) != expert_bytes:
                        raise SystemExit(f"short read at layer={layer} expert={expert}")
                    output.write(payload)
            print(f"packed layer {layer + 1:02d}/{LAYERS}", flush=True)
        if args.sync:
            output.flush()
            os.fsync(output.fileno())

    actual = args.output.stat().st_size
    if actual != expected:
        raise SystemExit(f"packed size mismatch: expected={expected} actual={actual}")
    print(
        f"expert_pack: OK layers={LAYERS} experts={EXPERTS} "
        f"slot_bytes={slot_bytes} bytes={actual}"
    )


if __name__ == "__main__":
    main()
