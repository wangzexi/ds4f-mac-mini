#!/usr/bin/env python3
"""Build the official 0731 Q4 model with one HF shard at a time.

The C quantizer owns the tensor mapping and writes selected tensor payloads at
 their final GGUF offsets.  This driver downloads only the shard group required
 for the current tensor group, quantizes it, records a checkpoint, and removes
 the source shard before continuing.  The result is one self-contained GGUF file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DEFAULT_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEFAULT_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
CHUNK_BYTES = 8 * 1024 * 1024


def die(message: str) -> "NoReturn":
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            block = fp.read(CHUNK_BYTES)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def auth_headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def resolve_url(repo: str, revision: str, name: str) -> str:
    # HF file names in the model index are simple basenames.  Refuse path
    # traversal before constructing a URL or a local staging path.
    if Path(name).name != name or name in ("", ".", ".."):
        die(f"unsafe shard name from official index: {name!r}")
    return f"https://huggingface.co/{repo}/resolve/{revision}/{name}?download=true"


def download_resumable(url: str, destination: Path, token: str | None) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    headers = auth_headers(token)
    if start:
        headers["Range"] = f"bytes={start}-"
    request = Request(url, headers=headers)
    try:
        response = urlopen(request, timeout=60)
    except HTTPError as exc:
        if start and exc.code == 416:
            os.replace(partial, destination)
            return destination.stat().st_size
        raise

    status = getattr(response, "status", 200)
    append = start > 0 and status == 206
    if start and not append:
        # Some mirrors ignore Range and return the whole object.  Restart
        # safely instead of appending a second copy to the partial file.
        start = 0
    mode = "ab" if append else "wb"
    total = response.headers.get("Content-Range", "").rsplit("/", 1)[-1]
    total_bytes = int(total) if total.isdigit() else None
    if total_bytes is None and response.headers.get("Content-Length", "").isdigit():
        length = int(response.headers["Content-Length"])
        total_bytes = start + length if append else length

    completed = start
    last_report = time.monotonic()
    with response, partial.open(mode) as fp:
        while True:
            block = response.read(CHUNK_BYTES)
            if not block:
                break
            fp.write(block)
            completed += len(block)
            now = time.monotonic()
            if now - last_report >= 5:
                suffix = f"/{total_bytes}" if total_bytes else ""
                print(f"download {destination.name}: {completed}{suffix} bytes", flush=True)
                last_report = now
        fp.flush()
        os.fsync(fp.fileno())
    if total_bytes is not None and completed != total_bytes:
        die(f"short download for {destination.name}: got {completed}, expected {total_bytes}")
    os.replace(partial, destination)
    return completed


def run_checked(command: list[str], *, capture: bool = False) -> str:
    print("+", " ".join(command), flush=True)
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    if result.returncode:
        raise SystemExit(result.returncode)
    return result.stdout or ""


def qargs(args: argparse.Namespace, *, out: Path) -> list[str]:
    command = [
        str(args.quantizer),
        "--hf",
        str(args.staging),
        "--template",
        str(args.layout),
        "--out",
        str(out),
        "--routed-w1",
        "iq2_xxs",
        "--routed-w2",
        "q2_k",
        "--routed-w3",
        "iq2_xxs",
        "--attention-proj",
        "q4_k",
        "--shared",
        "q4_k",
        "--embedding",
        "q4_k",
        "--output",
        "q4_k",
        "--threads",
        str(args.threads),
        "--overwrite",
    ]
    if args.imatrix:
        command.extend(["--imatrix", str(args.imatrix)])
    return command


def parse_kv(output: str, key: str) -> str:
    prefix = f"{key}: "
    for line in output.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    die(f"quantizer output has no {key}")


def read_stream_plan(args: argparse.Namespace, output: Path) -> list[tuple[int, str, tuple[str, ...]]]:
    output_text = run_checked(qargs(args, out=output) + ["--stream-plan"], capture=True)
    rows: list[tuple[int, str, tuple[str, ...]]] = []
    for line in output_text.splitlines():
        if not line.startswith("tensor\t"):
            continue
        fields = line.split("\t", 3)
        if len(fields) != 4:
            die(f"malformed stream-plan row: {line!r}")
        index = int(fields[1])
        files = tuple(item for item in fields[3].split(",") if item)
        if not files:
            die(f"stream-plan row has no source shards: {line!r}")
        rows.append((index, fields[2], files))
    rows.sort(key=lambda row: row[0])
    if [row[0] for row in rows] != list(range(len(rows))):
        die("stream plan tensor indexes are not a complete 0..N-1 sequence")
    return rows


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            parent = self.find(parent)
            self.parent[item] = parent
        return parent

    def union(self, left: str, right: str) -> None:
        self.add(left)
        self.add(right)
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def build_groups(rows: Iterable[tuple[int, str, tuple[str, ...]]]) -> list[tuple[tuple[str, ...], tuple[int, ...]]]:
    rows = list(rows)
    union = UnionFind()
    for _index, _name, files in rows:
        for filename in files:
            union.add(filename)
        for filename in files[1:]:
            union.union(files[0], filename)

    grouped: dict[str, dict[str, set[int]]] = {}
    for index, _name, files in rows:
        root = union.find(files[0])
        bucket = grouped.setdefault(root, {"files": set(), "indexes": set()})
        bucket["files"].update(files)
        bucket["indexes"].add(index)
    groups = [
        (tuple(sorted(bucket["files"])), tuple(sorted(bucket["indexes"])))
        for bucket in grouped.values()
    ]
    groups.sort(key=lambda group: group[1][0])
    return groups


def atomic_json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as fp:
        json.dump(value, fp, indent=2, sort_keys=True)
        fp.write("\n")
        fp.flush()
        os.fsync(fp.fileno())
        temporary = Path(fp.name)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    skill_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--staging", type=Path, default=Path("official-hf-staging"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layout", type=Path, default=skill_dir / "assets/deepseek-v4-flash-0731.layout.gguf")
    parser.add_argument("--quantizer", type=Path, required=True)
    parser.add_argument("--imatrix", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--keep-shards", action="store_true")
    parser.add_argument("--reset", action="store_true", help="remove only this build's checkpoint and output before starting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.staging = args.staging.resolve()
    args.out = args.out.resolve()
    args.layout = args.layout.resolve()
    args.quantizer = args.quantizer.resolve()
    token = os.environ.get("HF_TOKEN")

    if not args.layout.is_file():
        die(f"missing layout manifest: {args.layout}")
    if not args.quantizer.is_file() or not os.access(args.quantizer, os.X_OK):
        die(f"quantizer is not executable: {args.quantizer}")
    args.staging.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    state_path = args.out.with_name(args.out.name + ".stream-state.json")
    if args.reset:
        for path in (state_path, args.out):
            if path.exists():
                path.unlink()

    index_path = args.staging / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"downloading official index: {args.repo}@{args.revision}", flush=True)
        download_resumable(resolve_url(args.repo, args.revision, index_path.name), index_path, token)
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
    except (OSError, ValueError, KeyError) as exc:
        die(f"invalid official safetensors index {index_path}: {exc}")
    known_files = set(weight_map.values())
    if not known_files:
        die("official index has an empty weight_map")

    plan_rows = read_stream_plan(args, args.out)
    for _index, _name, files in plan_rows:
        unknown = set(files) - known_files
        if unknown:
            die(f"quantizer plan references files absent from official index: {sorted(unknown)}")
    groups = build_groups(plan_rows)
    print(f"stream plan: tensors={len(plan_rows)} shard_groups={len(groups)} source_shards={len(known_files)}", flush=True)

    dry_run = run_checked(qargs(args, out=args.out) + ["--dry-run"], capture=True)
    expected_gguf = int(parse_kv(dry_run, "approx_file_bytes"))

    completed: set[tuple[str, ...]] = set()
    state: dict[str, object] = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("repo") != args.repo or state.get("revision") != args.revision:
                die(f"checkpoint belongs to a different official source: {state_path}")
            if state.get("layout_sha256") != sha256_file(args.layout):
                die("layout manifest changed since the checkpoint was created")
            for item in state.get("completed_groups", []):
                completed.add(tuple(item))
        except (OSError, ValueError, TypeError) as exc:
            die(f"invalid stream checkpoint {state_path}: {exc}")

    initialized = bool(state.get("initialized")) and args.out.exists()
    if not initialized:
        if args.out.exists():
            die(f"output exists without a valid stream checkpoint; use --reset: {args.out}")
        run_checked(qargs(args, out=args.out) + ["--stream-init"])
        initialized = True
        state = {
            "repo": args.repo,
            "revision": args.revision,
            "layout_sha256": sha256_file(args.layout),
            "expected_gguf_bytes": expected_gguf,
            "completed_groups": [],
            "initialized": True,
        }
        atomic_json_write(state_path, state)
    elif args.out.stat().st_size != expected_gguf:
        die(f"initialized output size mismatch: got {args.out.stat().st_size}, expected {expected_gguf}")

    free_bytes = shutil.disk_usage(args.staging).free
    reserve_bytes = int(os.environ.get("DS4F_STREAM_TEMP_RESERVE_BYTES", str(6 * 1024**3)))
    if free_bytes < reserve_bytes:
        die(
            f"insufficient free space on staging volume: free={free_bytes} "
            f"need_at_least={reserve_bytes}"
        )
    print(
        f"disk preflight: free={free_bytes} reserve={reserve_bytes}",
        flush=True,
    )

    for group_number, (files, indexes) in enumerate(groups, start=1):
        if files in completed:
            print(f"skip completed group {group_number}/{len(groups)}: tensors={len(indexes)} files={len(files)}", flush=True)
            continue
        print(f"group {group_number}/{len(groups)}: tensors={len(indexes)} files={len(files)}", flush=True)
        for filename in files:
            destination = args.staging / filename
            if not destination.exists():
                download_resumable(resolve_url(args.repo, args.revision, filename), destination, token)
        run_checked(qargs(args, out=args.out) + ["--tensor-list", ",".join(map(str, indexes))])
        if not args.keep_shards:
            for filename in files:
                for candidate in (args.staging / filename, (args.staging / filename).with_name(filename + ".part")):
                    if candidate.exists():
                        candidate.unlink()
        completed.add(files)
        state["completed_groups"] = [list(item) for item in sorted(completed)]
        atomic_json_write(state_path, state)

    atomic_json_write(state_path, state)
    print(f"stream build complete: gguf={args.out} bytes={args.out.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
