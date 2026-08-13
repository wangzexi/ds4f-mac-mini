#!/usr/bin/env bash
# Exact selected-expert I/O sweep using the actual single-session Mini server.
#
# This deliberately starts a fresh server for every measurement: it exercises
# the production L0/hash preload, static Decode trunk, request memory planner,
# and 1000-slot pageable/wired expert cache.  Do not run while production is
# serving traffic.  The caller is responsible for restoring production after
# this exclusive benchmark completes.
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
server="$project_dir/ds4f-server"
model="$project_dir/models/DeepSeek-V4-Flash-0731-Mini-Q4Trunk-IQ2Experts.gguf"
pack="$project_dir/models/DeepSeek-V4-Flash-0731-IQ2Experts-packed.bin"

prompt=${DS4F_BENCH_PROMPT:-你好}
tokens=${DS4F_BENCH_TOKENS:-32}
cache_experts=${DS4F_BENCH_SERVER_CACHE_EXPERTS:-1800}
repeats=${DS4F_BENCH_REPEATS:-2}
threads_csv=${DS4F_BENCH_PREAD_THREADS:-1,2,3,4,6}
port=${DS4F_BENCH_SERVER_PORT:-18080}
out_dir=${DS4F_BENCH_OUT_DIR:-"$project_dir/results/benchmarks/server-exact-io-$(date +%Y%m%d-%H%M%S)"}
record_selected_ids=${DS4F_BENCH_RECORD_SELECTED_IDS:-0}
if [[ $out_dir != /* ]]; then
    out_dir="$project_dir/$out_dir"
fi

if [[ ! -x "$server" || ! -r "$model" || ! -r "$pack" ]]; then
    echo "missing fixed Mini server, model, or packed experts" >&2
    exit 2
fi
if pgrep -x ds4f-server >/dev/null 2>&1; then
    echo "refusing to contend with a ds4f-server; stop production before this exclusive benchmark" >&2
    exit 2
fi
if [[ ! $tokens =~ ^[1-9][0-9]*$ ]] || [[ ! $cache_experts =~ ^[1-9][0-9]*$ ]] ||
   [[ ! $repeats =~ ^[1-9][0-9]*$ ]] || [[ ! $port =~ ^[1-9][0-9]*$ ]]; then
    echo "tokens, cache experts, repeats, and port must be positive integers" >&2
    exit 2
fi

mkdir -p "$out_dir"
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/ds4f-server-io.XXXXXX")
server_pid=
cleanup() {
    if [[ -n ${server_pid:-} ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    rm -rf "$tmp_dir"
}
# A benchmark is usually driven over SSH.  A signal handler must explicitly
# exit after cleanup: merely returning from a HUP trap lets Bash continue the
# loop and leaves a later private server alive after the controller is gone.
trap cleanup EXIT
trap 'cleanup; exit 129' HUP
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

printf 'prompt=%q\ntokens=%s\ncache_experts=%s\nrepeats=%s\nthreads=%s\nport=%s\n' \
    "$prompt" "$tokens" "$cache_experts" "$repeats" "$threads_csv" "$port" \
    > "$out_dir/config.env"
shasum -a 256 "$server" > "$out_dir/server.sha256"
stat -f '%N\t%z\t%m\t%i' "$model" "$pack" > "$out_dir/model-pack.stat.tsv"

wait_for_server() {
    local log=$1
    for _ in $(seq 1 240); do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "benchmark server exited during startup" >&2
            tail -80 "$log" >&2
            return 1
        fi
        if curl -fsS --connect-timeout 2 --max-time 5 \
                "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "benchmark server did not become ready within 240 seconds" >&2
    tail -80 "$log" >&2
    return 1
}

stop_server() {
    if [[ -n ${server_pid:-} ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid"
        # A signal-clean shutdown may still surface as a non-zero wait status;
        # the next start/readiness check is the meaningful benchmark gate.
        wait "$server_pid" || true
    fi
    server_pid=
}

reference_trace=
reference_response_hash=
IFS=, read -r -a threads_list <<< "$threads_csv"
for threads in "${threads_list[@]}"; do
    if [[ ! $threads =~ ^[1-9][0-9]*$ ]] || (( threads > 6 )); then
        echo "Decode has at most six packed expert reads; pread threads must be 1..6: $threads" >&2
        exit 2
    fi
    for ((repeat = 1; repeat <= repeats; repeat++)); do
        log="$out_dir/server-pread-${threads}-run-${repeat}.log"
        response="$out_dir/server-pread-${threads}-run-${repeat}.json"
        kv_dir="$tmp_dir/kv-${threads}-${repeat}"
        mkdir -p "$kv_dir"

        if [[ $record_selected_ids != 0 ]]; then
            export DS4_MOE_RECORD_SELECTED_IDS="$out_dir/server-pread-${threads}-run-${repeat}.selected-i32le"
        else
            unset DS4_MOE_RECORD_SELECTED_IDS || true
        fi
        env \
            DS4F_SERVER_HOST=127.0.0.1 \
            DS4F_SERVER_PORT="$port" \
            DS4F_SERVER_CACHE_EXPERTS="$cache_experts" \
            DS4F_SERVER_KV_CACHE_DIR="$kv_dir" \
            DS4F_SERVER_KV_CACHE_MIB=1 \
            DS4F_SERVER_KV_CACHE_MIN_TOKENS=32769 \
            DS4_METAL_STREAMING_EXPERT_PREAD_THREADS="$threads" \
            DS4_METAL_STREAMING_EXPERT_PREAD_POOL=1 \
            DS4_METAL_STREAMING_EXPERT_PREAD_PROFILE=1 \
            DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1 \
            DS4_METAL_STREAMING_EXPERT_LAYER_STATS=1 \
            DS4_METAL_STREAMING_EXPERT_LAYER_STATS_DELTA=1 \
            DS4_METAL_GPU_BUSY_PROFILE=1 \
            DS4_SERVER_EXPERT_PHASE_PROFILE=1 \
            DS4_SERVER_TRACE_TOKEN_IDS=1 \
            "$script_dir/run-server.sh" >"$log" 2>&1 &
        server_pid=$!
        wait_for_server "$log"

        curl -fsS --connect-timeout 5 --max-time 900 \
            -H 'Content-Type: application/json' \
            -d "$(python3 - "$prompt" "$tokens" <<'PY'
import json
import sys
print(json.dumps({
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": sys.argv[1]}],
    "max_tokens": int(sys.argv[2]),
    "temperature": 0,
    "stream": False,
}, ensure_ascii=False))
PY
)" \
            "http://127.0.0.1:${port}/v1/chat/completions" > "$response"

        trace=$(sed -n -E 's/.*trace token\[[0-9]+\]=([0-9]+).*/\1/p' "$log" | paste -sd, -)
        if [[ -z $trace ]]; then
            echo "no greedy token trace in $log" >&2
            exit 1
        fi
        response_hash=$(python3 - "$response" <<'PY'
import hashlib
import json
import sys
with open(sys.argv[1], encoding="utf-8") as fp:
    obj = json.load(fp)
text = obj["choices"][0]["message"]["content"]
print(hashlib.sha256(text.encode("utf-8")).hexdigest())
PY
)
        if [[ -z $reference_trace ]]; then
            reference_trace=$trace
            reference_response_hash=$response_hash
            printf '%s\n' "$trace" > "$out_dir/reference-trace.csv"
            printf '%s\n' "$response_hash" > "$out_dir/reference-response.sha256"
        elif [[ $trace != "$reference_trace" || $response_hash != "$reference_response_hash" ]]; then
            echo "exact server regression failed for pread_threads=$threads repeat=$repeat" >&2
            echo "expected trace: $reference_trace" >&2
            echo "actual trace:   $trace" >&2
            exit 1
        fi
        printf 'pread_threads=%s repeat=%s trace=%s response_sha256=%s\n' \
            "$threads" "$repeat" "$trace" "$response_hash" >> "$out_dir/runs.tsv"
        stop_server
    done
done

printf 'exact server I/O matrix passed: %s\n' "$out_dir"
