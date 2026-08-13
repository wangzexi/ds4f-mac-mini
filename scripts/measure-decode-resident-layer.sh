#!/bin/sh
# Strict single-layer decode measurement.  First run obtains the genuine
# routing choices; second run preloads exactly those choices *after prefill*
# and stage-profiles the requested layer.  No production setting is changed.
set -eu

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "usage: $0 LAYER PROMPT|@FILE [RESULT_DIR]" >&2
    exit 2
fi

layer=$1
prompt=$2
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
result_dir=${3:-"$project_dir/results/measurements/decode-resident-layer-${layer}-$(date +%Y%m%d-%H%M%S)"}
mkdir -p "$result_dir"

trace="$result_dir/first-decode.selected.bin"
hotlist="$result_dir/first-decode.exact-hotlist.txt"
baseline="$result_dir/baseline"
resident="$result_dir/resident"

echo "[1/2] Recording genuine routes for the first decode token..." >&2
env DS4_MOE_RECORD_SELECTED_IDS="$trace" \
    DS4_METAL_DECODE_STAGE_PROFILE=1 \
    DS4_METAL_DECODE_STAGE_PROFILE_LAYER="$layer" \
    "$script_dir/run-32k.sh" exact "$prompt" 2 >"$baseline.stdout" 2>"$baseline.stderr"
python3 "$script_dir/make-decode-diag-hotlist.py" "$trace" "$hotlist" --layers 43 \
    >"$result_dir/seed.stdout" 2>"$result_dir/seed.stderr"

echo "[2/2] Seeding those routes after prefill and profiling layer $layer..." >&2
env DS4_METAL_DIAG_POST_PREFILL_EXPERT_HOTLIST="$hotlist" \
    DS4_METAL_STREAMING_EXPERT_HOTLIST_PROFILE=1 \
    DS4_METAL_DECODE_STAGE_PROFILE=1 \
    DS4_METAL_DECODE_STAGE_PROFILE_LAYER="$layer" \
    DS4_METAL_STREAMING_EXPERT_LAYER_STATS=1 \
    DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1 \
    "$script_dir/run-32k.sh" exact "$prompt" 2 >"$resident.stdout" 2>"$resident.stderr"

echo "$result_dir"
