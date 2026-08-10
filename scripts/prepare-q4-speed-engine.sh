#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)

"$project_dir/scripts/prepare-speed-engine.sh"
"$project_dir/scripts/prepare-q4-engine.sh" \
    "$project_dir/speed-ds4" \
    "$project_dir/q4-speed-ds4"
patch -s -d "$project_dir/q4-speed-ds4" -p1 \
    < "$project_dir/patches/packed-expert-sidecar.patch"

echo "$project_dir/q4-speed-ds4"
