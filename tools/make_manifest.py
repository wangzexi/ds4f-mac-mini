#!/usr/bin/env python3
"""Build a compact layer/tensor manifest from gguf_inventory.py output."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inventory", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()

    inventory = json.loads(args.inventory.read_text())
    layers = defaultdict(list)
    global_tensors = []
    for tensor in inventory["tensors"]:
        item = {
            "name": tensor["name"],
            "dims": tensor["dims"],
            "type": tensor["type"],
            "file_offset": tensor["file_offset"],
            "nbytes": tensor["nbytes_exact"],
            "group": tensor["group"],
        }
        if tensor["group"] == "routed_expert" and tensor["dims"] and tensor["dims"][-1] == 256:
            if tensor["nbytes_exact"] is not None and tensor["nbytes_exact"] % 256 == 0:
                item["expert_count"] = 256
                item["expert_bytes"] = tensor["nbytes_exact"] // 256
        if tensor["layer"] is None:
            global_tensors.append(item)
        else:
            layers[tensor["layer"]].append(item)

    manifest = {
        "format": "ds4f-mini-manifest-v1",
        "source": inventory["path"],
        "file_bytes_at_inventory": inventory["file_bytes"],
        "data_start": inventory["data_start"],
        "metadata": inventory.get("interesting_metadata", {}),
        "global_tensors": global_tensors,
        "layers": {str(k): v for k, v in sorted(layers.items())},
    }
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {args.output}: {len(layers)} layers, {len(global_tensors)} global tensors")


if __name__ == "__main__":
    main()
