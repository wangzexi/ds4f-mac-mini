#!/usr/bin/env python3
"""Verify that routed MoE tensor payloads are byte-identical in two GGUFs."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


VALUE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


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


def load(path):
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
    by_offset = sorted(tensors, key=lambda row: row[3])
    spans = {}
    for index, row in enumerate(by_offset):
        name, dims, tensor_type, offset = row
        next_offset = by_offset[index + 1][3] if index + 1 < len(by_offset) else path.stat().st_size - data_start
        spans[name] = (dims, tensor_type, data_start + offset, next_offset - offset)
    return spans


def is_routed(name):
    return "ffn_gate_exps.weight" in name or "ffn_down_exps.weight" in name or "ffn_up_exps.weight" in name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chunk-mib", type=int, default=8)
    args = parser.parse_args()

    template = load(args.template)
    output = load(args.output)
    names = [name for name in template if is_routed(name)]
    if not names:
        raise SystemExit("no routed tensors found")

    checked = 0
    checked_bytes = 0
    chunk_size = args.chunk_mib * 1024 * 1024
    with args.template.open("rb") as source_fp, args.output.open("rb") as output_fp:
        for name in names:
            if name not in output:
                raise SystemExit(f"missing output tensor: {name}")
            source_dims, source_type, source_offset, source_size = template[name]
            output_dims, output_type, output_offset, output_size = output[name]
            if (source_dims, source_type, source_size) != (output_dims, output_type, output_size):
                raise SystemExit(f"metadata/size mismatch: {name}")
            source_fp.seek(source_offset)
            output_fp.seek(output_offset)
            remaining = source_size
            at = 0
            while remaining:
                amount = min(remaining, chunk_size)
                source_data = source_fp.read(amount)
                output_data = output_fp.read(amount)
                if len(source_data) != amount or len(output_data) != amount:
                    raise SystemExit(f"short read: {name} at {at}")
                if source_data != output_data:
                    raise SystemExit(f"payload mismatch: {name} at chunk offset {at}")
                remaining -= amount
                at += amount
            checked += 1
            checked_bytes += source_size
            print(f"OK {checked:3d}/{len(names):3d} {name}", flush=True)
    print(f"routed_copy: OK tensors={checked} bytes={checked_bytes}")


if __name__ == "__main__":
    main()
