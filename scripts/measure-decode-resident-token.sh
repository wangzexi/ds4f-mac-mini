#!/bin/sh
# Measure a normal (un-instrumented) first decode with all its actual routed
# experts resident.  The second run has no stage profiler: its timing is the
# production Metal kernel cadence, while DS4_METAL_GRAPH_TOKEN_PROFILE reports
# the aggregate GPU execute interval across all transformer layers.
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 PROMPT|@FILE [RESULT_DIR]" >&2
    exit 2
fi

prompt=$1
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
result_dir=${2:-"$project_dir/results/measurements/decode-resident-token-$(date +%Y%m%d-%H%M%S)"}
mkdir -p "$result_dir"
trace="$result_dir/first-decode.selected.bin"
hotlist="$result_dir/first-decode.exact-hotlist.txt"
resident_trace="$result_dir/resident.selected.bin"

echo "[1/2] Recording the normal first-decode routes..." >&2
env DS4_MOE_RECORD_SELECTED_IDS="$trace" \
    "$script_dir/run-32k.sh" exact "$prompt" 2 >"$result_dir/baseline.stdout" 2>"$result_dir/baseline.stderr"
python3 "$script_dir/make-decode-diag-hotlist.py" "$trace" "$hotlist" --layers 43 \
    >"$result_dir/seed.stdout" 2>"$result_dir/seed.stderr"

echo "[2/2] Measuring the same routes with all selected experts resident..." >&2
env DS4_METAL_DIAG_POST_PREFILL_EXPERT_HOTLIST="$hotlist" \
    DS4_MOE_RECORD_SELECTED_IDS="$resident_trace" \
    DS4_METAL_GRAPH_TOKEN_PROFILE=1 \
    "$script_dir/run-32k.sh" exact "$prompt" 2 >"$result_dir/resident.stdout" 2>"$result_dir/resident.stderr"

python3 - "$trace" "$resident_trace" <<'PY'
import sys
from pathlib import Path

first, resident = (Path(p).read_bytes() for p in sys.argv[1:])
group = 43 * 24  # 43 layers, six i32 IDs each
if len(first) < group or len(resident) < group:
    raise SystemExit("selected-ID trace is shorter than one decode layer group")
if first[-group:] != resident[-group:]:
    raise SystemExit("diagnostic route mismatch: refusing to report resident timing")
print("verified: normal first-decode routes match exactly")
PY

grep 'metal SSD streaming token pos=' "$result_dir/resident.stderr" | tail -1 >&2 || true
echo "$result_dir"
