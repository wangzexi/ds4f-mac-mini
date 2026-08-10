#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
source_dir="$project_dir/reference-ds4"
target_dir="$project_dir/speed-ds4"

if [ ! -d "$source_dir" ]; then
    printf '%s\n' 'ds4f-speed: missing reference-ds4' >&2
    exit 1
fi
mkdir -p "$target_dir"
rsync -a --delete --delete-excluded \
    --exclude=.git \
    --exclude=gguf \
    --exclude='*.gguf' \
    --exclude='*.o' \
    --exclude=ds4 \
    --exclude='ds4-*' \
    --exclude='*.dSYM' \
    "$source_dir/" "$target_dir/"
cd "$target_dir"
patch -p1 < "$project_dir/patches/cache-aware-experts.patch"
patch -p1 < "$project_dir/patches/decode-only-cache-aware.patch"
patch -p1 < "$project_dir/patches/cache-aware-entropy-guard.patch"
touch .ds4f-speed-prepared
