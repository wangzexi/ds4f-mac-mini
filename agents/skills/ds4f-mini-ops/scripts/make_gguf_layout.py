#!/usr/bin/env python3
"""Extract a weight-free GGUF layout manifest.

The quantizer needs GGUF metadata, tensor names, shapes, and ordering, but it
does not need the payload when routed experts are generated from HF shards.
This tool copies only the GGUF header and tensor-info region, leaving the
payload behind.  The resulting file is intentionally not loadable as a model;
it is a small build-time layout manifest.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


VALUE_SIZES = {
    0: 1,  # UINT8
    1: 1,  # INT8
    2: 2,  # UINT16
    3: 2,  # INT16
    4: 4,  # UINT32
    5: 4,  # INT32
    6: 4,  # FLOAT32
    7: 1,  # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}


def read_u32(fp) -> int:
    data = fp.read(4)
    if len(data) != 4:
        raise ValueError("short GGUF uint32")
    return struct.unpack("<I", data)[0]


def read_u64(fp) -> int:
    data = fp.read(8)
    if len(data) != 8:
        raise ValueError("short GGUF uint64")
    return struct.unpack("<Q", data)[0]


def read_string(fp) -> str:
    size = read_u64(fp)
    if size > 16 << 20:
        raise ValueError(f"unreasonable GGUF string at offset {fp.tell() - 8}: {size}")
    data = fp.read(size)
    if len(data) != size:
        raise ValueError("short GGUF string")
    return data.decode("utf-8")


def skip_value(fp, kind: int) -> None:
    if kind == 8:  # STRING
        fp.seek(read_u64(fp), 1)
        return
    if kind == 9:  # ARRAY
        element_kind = read_u32(fp)
        count = read_u64(fp)
        for _ in range(count):
            skip_value(fp, element_kind)
        return
    try:
        fp.seek(VALUE_SIZES[kind], 1)
    except KeyError as exc:
        raise ValueError(f"unsupported GGUF metadata type {kind}") from exc


def read_alignment_value(fp, kind: int) -> int | None:
    """Read general.alignment when it is encoded as an integer."""
    if kind == 4:  # UINT32
        return read_u32(fp)
    if kind == 10:  # UINT64
        return read_u64(fp)
    skip_value(fp, kind)
    return None


def extract_layout(source: Path, output: Path) -> tuple[int, int, int]:
    with source.open("rb") as src:
        if src.read(4) != b"GGUF":
            raise ValueError(f"not a GGUF file: {source}")
        version = read_u32(src)
        tensor_count = read_u64(src)
        metadata_count = read_u64(src)
        alignment = None
        for _ in range(metadata_count):
            key = read_string(src)
            kind = read_u32(src)
            if key == "general.alignment":
                value = read_alignment_value(src, kind)
                if value is not None:
                    alignment = value
            else:
                skip_value(src, kind)
        for _ in range(tensor_count):
            read_string(src)
            dims = read_u32(src)
            src.seek(8 * dims + 4 + 8, 1)
        if alignment is None:
            alignment = 32
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError(f"invalid GGUF alignment: {alignment}")
        header_end = src.tell()
        data_offset = (header_end + alignment - 1) // alignment * alignment

    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, output.open("wb") as dst:
        remaining = data_offset
        while remaining:
            chunk = src.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError("source ended before GGUF data offset")
            dst.write(chunk)
            remaining -= len(chunk)
    return version, tensor_count, data_offset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    version, tensors, bytes_written = extract_layout(args.source, args.output)
    print(
        f"gguf_layout: OK version={version} tensors={tensors} "
        f"bytes={bytes_written} output={args.output}"
    )


if __name__ == "__main__":
    main()
