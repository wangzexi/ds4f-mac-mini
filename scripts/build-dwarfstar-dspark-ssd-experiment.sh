#!/usr/bin/env bash
set -euo pipefail

# Build a separate experimental DwarfStar binary. The checked-out reference
# source stays untouched: two narrow patches are applied only to temporary
# compiler inputs.

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
reference_dir="${1:-"$project_dir/reference-ds4"}"
output_bin="${DS4F_DSPARK_SSD_BIN:-"$project_dir/bin/ds4-dspark-ssd"}"

for required in ds4.c ds4_metal.m ds4.h Makefile; do
    if [[ ! -f "$reference_dir/$required" ]]; then
        echo "missing DwarfStar reference file: $reference_dir/$required" >&2
        exit 1
    fi
done

make -C "$reference_dir" \
    ds4_cli.o ds4_help.o linenoise.o ds4_gpu_args.o ds4_distributed.o \
    ds4_tp.o ds4_ssd.o ds4_layer_pack.o

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/ds4f-dspark-ssd.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT
cp "$reference_dir/ds4.c" "$work_dir/ds4.c"
cp "$reference_dir/ds4_metal.m" "$work_dir/ds4_metal.m"

(
    cd "$work_dir"
    patch --batch -p0 < "$project_dir/patches/dwarfstar-dspark-ssd.c.patch"
    patch --batch -p0 < "$project_dir/patches/dwarfstar-dspark-ssd.metal.patch"
)

cc -O3 -ffast-math -g -mcpu=native -Wall -Wextra -std=c99 \
    -I "$reference_dir" -c "$work_dir/ds4.c" -o "$work_dir/ds4.o"
cc -O3 -ffast-math -g -mcpu=native -Wall -Wextra -fobjc-arc \
    -I "$reference_dir" -c "$work_dir/ds4_metal.m" -o "$work_dir/ds4_metal.o"

mkdir -p "$(dirname "$output_bin")"
temporary_bin="$output_bin.tmp"
cc -O3 -ffast-math -g -mcpu=native -Wall -Wextra -std=c99 \
    -I "$reference_dir" -o "$temporary_bin" \
    "$reference_dir/ds4_cli.o" "$reference_dir/ds4_help.o" \
    "$reference_dir/linenoise.o" "$reference_dir/ds4_gpu_args.o" \
    "$work_dir/ds4.o" "$reference_dir/ds4_distributed.o" \
    "$reference_dir/ds4_tp.o" "$reference_dir/ds4_ssd.o" \
    "$work_dir/ds4_metal.o" "$reference_dir/ds4_layer_pack.o" \
    -lm -pthread -framework Foundation -framework Metal
mv -f "$temporary_bin" "$output_bin"

printf 'built %s\n' "$output_bin"
