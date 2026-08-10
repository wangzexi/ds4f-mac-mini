#!/bin/sh
set -eu

if [ "$#" -gt 2 ]; then
    echo "usage: $0 [SOURCE_ENGINE] [OUTPUT_ENGINE]" >&2
    exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
source_engine=${1:-"$project_dir/reference-ds4"}
output_engine=${2:-"$project_dir/q4-ds4"}
patch_file="$project_dir/patches/q4-embedding-hc.patch"
dspark_calls_patch="$project_dir/patches/q4-embedding-dspark-calls.patch"

case "$output_engine" in
    "$project_dir"/q4-ds4|"$project_dir"/q4-speed-ds4) ;;
    *)
        echo "refusing unexpected generated-engine path: $output_engine" >&2
        exit 2
        ;;
esac

if [ ! -f "$source_engine/ds4.c" ] || [ ! -f "$patch_file" ] ||
   [ ! -f "$dspark_calls_patch" ]; then
    echo "source engine or Q4 embedding patch is unavailable" >&2
    exit 2
fi

mkdir -p "$output_engine"
rsync -a --delete --delete-excluded \
    --exclude gguf \
    --exclude '*.o' \
    --exclude ds4 \
    "$source_engine/" "$output_engine/"
patch -s -d "$output_engine" -p1 < "$patch_file"
patch -s -d "$output_engine" -p1 < "$dspark_calls_patch"

echo "$output_engine"
