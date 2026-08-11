#!/usr/bin/env python3
"""Inventory a GGUF file without loading tensor payloads.

This is intentionally standalone: it only reads the GGUF header and tensor
metadata, so it can be used before writing the DS4F-Mini runtime.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path


MAGIC = b"GGUF"

# GGUF metadata value types. Arrays are handled recursively.
TYPES = {
    0: ("uint8", "B"),
    1: ("int8", "b"),
    2: ("uint16", "H"),
    3: ("int16", "h"),
    4: ("uint32", "I"),
    5: ("int32", "i"),
    6: ("float32", "f"),
    7: ("bool", "?"),
    8: ("string", None),
    9: ("array", None),
    10: ("uint64", "Q"),
    11: ("int64", "q"),
    12: ("float64", "d"),
}


class Reader:
    def __init__(self, fp):
        self.fp = fp

    def read(self, n: int) -> bytes:
        data = self.fp.read(n)
        if len(data) != n:
            raise EOFError(f"short read: wanted {n}, got {len(data)}")
        return data

    def u8(self):
        return self.read(1)[0]

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def string(self):
        n = self.u64()
        if n > 16 * 1024 * 1024:
            raise ValueError(f"implausibly large GGUF string: {n}")
        return self.read(n).decode("utf-8", errors="replace")

    def value(self, kind: int, keep: bool = False):
        if kind == 8:
            return self.string()
        if kind == 9:
            element_kind = self.u32()
            n = self.u64()
            if n > 10_000_000:
                raise ValueError(f"implausibly large GGUF array: {n}")
            values = [self.value(element_kind, keep=keep) for _ in range(n)]
            return values if keep else None
        name, fmt = TYPES[kind]
        value = struct.unpack("<" + fmt, self.read(struct.calcsize(fmt)))[0]
        return value


def tensor_nbytes(dims, tensor_type: int) -> int | None:
    # GGML type block sizes / bytes per block for the types used by current
    # DeepSeek GGUFs. Unknown types are reported but not guessed.
    blocks = {
        0: (1, 4),       # F32
        1: (1, 2),       # F16
        2: (32, 18),     # Q4_0
        3: (32, 20),     # Q4_1
        6: (32, 22),     # Q5_0
        7: (32, 24),     # Q5_1
        8: (32, 34),     # Q8_0
        9: (32, 36),     # Q8_1
        10: (256, 84),   # Q2_K
        11: (256, 110),  # Q3_K
    }
    # Keep the table explicit; IQ and MXFP types vary by GGML revision.
    blocks.update({
        12: (256, 144),  # Q4_K
        13: (256, 176),  # Q5_K
        14: (256, 210),  # Q6_K
        15: (256, 292),  # Q8_K
        16: (256, 66),   # IQ2_XXS
    })
    if tensor_type not in blocks:
        return None
    block, bytes_per_block = blocks[tensor_type]
    elements = 1
    for dim in dims:
        elements *= dim
    if elements % block:
        return None
    return (elements // block) * bytes_per_block


def classify(name: str):
    lower = name.lower()
    match = re.search(r"(?:blk|block|layers?)[._-]?(\d+)", lower)
    layer = int(match.group(1)) if match else None
    if "expert" in lower or ".exps." in lower or "exps." in lower:
        group = "routed_expert"
    elif "output.weight" in lower or "lm_head" in lower:
        group = "output"
    elif "embed" in lower:
        group = "embedding"
    elif "attn" in lower or "attention" in lower:
        group = "attention"
    elif "ffn" in lower or "mlp" in lower:
        group = "ffn"
    else:
        group = "other"
    return layer, group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--json", type=Path, help="write full inventory JSON")
    args = ap.parse_args()

    with args.path.open("rb") as fp:
        r = Reader(fp)
        if r.read(4) != MAGIC:
            raise SystemExit("not a GGUF file")
        version = r.u32()
        tensor_count = r.u64()
        metadata_count = r.u64()
        metadata = {}
        interesting_metadata = {}
        for _ in range(metadata_count):
            key = r.string()
            kind = r.u32()
            keep = (
                key.startswith("general.")
                or "expert" in key
                or "block" in key
                or "attention" in key
                or "embedding" in key
                or "feed_forward" in key
                or "rope" in key
            )
            value = r.value(kind, keep=keep)
            if keep:
                interesting_metadata[key] = value
            metadata[key] = value
        alignment = int(metadata.get("general.alignment") or 32)
        tensor_infos = []
        for _ in range(tensor_count):
            name = r.string()
            n_dims = r.u32()
            dims = [r.u64() for _ in range(n_dims)]
            tensor_type = r.u32()
            offset = r.u64()
            tensor_infos.append({
                "name": name,
                "dims": dims,
                "type": tensor_type,
                "offset": offset,
                "nbytes": tensor_nbytes(dims, tensor_type),
            })
        data_start = (r.fp.tell() + alignment - 1) // alignment * alignment

    # The exact payload size is available from adjacent tensor offsets.  This
    # is more reliable than maintaining a copy of every GGML quant block table
    # (especially for IQ* and newer MXFP formats).  The final tensor is only
    # exact when the full file is present.
    ordered = sorted(tensor_infos, key=lambda item: item["offset"])
    for index, item in enumerate(ordered):
        item["file_offset"] = data_start + item["offset"]
        if index + 1 < len(ordered):
            item["nbytes_exact"] = ordered[index + 1]["offset"] - item["offset"]
        else:
            remaining = args.path.stat().st_size - item["file_offset"]
            item["nbytes_exact"] = remaining if remaining >= 0 else None
        item["layer"], item["group"] = classify(item["name"])

    by_layer = defaultdict(lambda: {
        "bytes": 0,
        "known_bytes": 0,
        "tensors": 0,
        "groups": Counter(),
        "group_bytes": Counter(),
        "start": None,
        "end": None,
    })
    type_bytes = Counter()
    unknown_bytes = 0
    for item in tensor_infos:
        layer = item["layer"] if item["layer"] is not None else -1
        row = by_layer[layer]
        row["tensors"] += 1
        row["groups"][item["group"]] += 1
        start = item["offset"]
        end = start + item["nbytes_exact"] if item["nbytes_exact"] is not None else None
        row["start"] = start if row["start"] is None else min(row["start"], start)
        if end is not None:
            row["end"] = end if row["end"] is None else max(row["end"], end)
        nbytes = item["nbytes_exact"]
        if nbytes is None:
            unknown_bytes += 1
        else:
            row["bytes"] += nbytes
            row["known_bytes"] += nbytes
            row["group_bytes"][item["group"]] += nbytes
            type_bytes[str(item["type"])] += nbytes

    summary = {
        "path": str(args.path),
        "file_bytes": args.path.stat().st_size,
        "gguf_version": version,
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "interesting_metadata": interesting_metadata,
        "alignment": alignment,
        "data_start": data_start,
        "unknown_tensor_sizes": unknown_bytes,
        "layers": {
            str(k): {
                "bytes": v["bytes"],
                "tensors": v["tensors"],
                "groups": dict(v["groups"]),
                "group_bytes": dict(v["group_bytes"]),
                "payload_start": v["start"],
                "payload_end": v["end"],
                "payload_span": (v["end"] - v["start"])
                if v["start"] is not None and v["end"] is not None else None,
            }
            for k, v in sorted(by_layer.items())
        },
        "type_bytes": dict(type_bytes),
        "tensors": tensor_infos,
    }

    print(json.dumps({k: v for k, v in summary.items() if k != "tensors"}, ensure_ascii=False, indent=2))
    if args.json:
        args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
