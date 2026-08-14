#!/usr/bin/env bash
# Exercise the fixed Mini exactly as its intended user does: one server and
# one continuing chat.  The second request carries the first answer back in
# the canonical chat transcript, so the server can reuse its in-memory / disk
# KV prefix while the benchmark records whether Decode's expert cache survives
# the intervening Prefill phase.
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
server="$project_dir/ds4f-server"
port=${DS4F_BENCH_SERVER_PORT:-18081}
repeats=${DS4F_BENCH_REPEATS:-2}
first_tokens=${DS4F_BENCH_FIRST_TOKENS:-8}
second_tokens=${DS4F_BENCH_SECOND_TOKENS:-32}
fresh_control=${DS4F_BENCH_FRESH_CONTROL:-1}
kv_cache_min_tokens=${DS4F_BENCH_KV_CACHE_MIN_TOKENS:-1}
out_dir=${DS4F_BENCH_OUT_DIR:-"$project_dir/results/benchmarks/session-continuity-$(date +%Y%m%d-%H%M%S)"}

if [[ $out_dir != /* ]]; then out_dir="$project_dir/$out_dir"; fi
if [[ ! -x $server ]] || [[ ! $repeats =~ ^[1-9][0-9]*$ ]] ||
   [[ ! $first_tokens =~ ^[1-9][0-9]*$ ]] || [[ ! $second_tokens =~ ^[1-9][0-9]*$ ]]; then
    echo "server, repeats, first_tokens, and second_tokens are invalid" >&2
    exit 2
fi
if [[ $fresh_control != 0 && $fresh_control != 1 ]]; then
    echo "fresh_control must be 0 or 1" >&2
    exit 2
fi
if [[ ! $kv_cache_min_tokens =~ ^[1-9][0-9]*$ ]]; then
    echo "kv_cache_min_tokens must be a positive integer" >&2
    exit 2
fi
if pgrep -x ds4f-server >/dev/null 2>&1; then
    echo "refusing to contend with a ds4f-server; use bench-exclusive-server.sh" >&2
    exit 2
fi

mkdir -p "$out_dir"
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/ds4f-session.XXXXXX")
server_pid=
cleanup() {
    if [[ -n ${server_pid:-} ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    rm -rf "$tmp_dir"
}
trap cleanup EXIT
trap 'cleanup; exit 129' HUP
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

wait_for_server() {
    local log=$1
    for _ in $(seq 1 240); do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            tail -80 "$log" >&2
            return 1
        fi
        if curl -fsS --connect-timeout 2 --max-time 5 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    tail -80 "$log" >&2
    return 1
}

first_trace_ref=
second_trace_ref=
first_hash_ref=
second_hash_ref=
for ((repeat = 1; repeat <= repeats; repeat++)); do
    log="$out_dir/server-run-$repeat.log"
    first="$out_dir/turn-1-run-$repeat.json"
    second="$out_dir/turn-2-run-$repeat.json"
    control="$out_dir/turn-2-fresh-control-run-$repeat.json"
    kv_dir="$tmp_dir/kv-$repeat"
    mkdir -p "$kv_dir"
    env \
        DS4F_SERVER_HOST=127.0.0.1 \
        DS4F_SERVER_PORT="$port" \
        DS4F_SERVER_KV_CACHE_DIR="$kv_dir" \
        DS4F_SERVER_KV_CACHE_MIB=64 \
        DS4F_SERVER_KV_CACHE_MIN_TOKENS="$kv_cache_min_tokens" \
        DS4_METAL_STREAMING_EXPERT_PREAD_PROFILE=1 \
        DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1 \
        DS4_METAL_STREAMING_EXPERT_LAYER_STATS=1 \
        DS4_METAL_STREAMING_EXPERT_LAYER_STATS_DELTA=1 \
        DS4_SERVER_EXPERT_PHASE_PROFILE=1 \
        DS4_SERVER_TRACE_TOKEN_IDS=1 \
        "$script_dir/run-server.sh" >"$log" 2>&1 &
    server_pid=$!
    wait_for_server "$log"

    first_seconds=$(curl -fsS --connect-timeout 5 --max-time 900 \
        -H 'Content-Type: application/json' \
        -d "$(python3 - "$first_tokens" <<'PY'
import json, sys
print(json.dumps({
    'model': 'deepseek-v4-flash',
    'messages': [{'role': 'user', 'content': '你好'}],
    'max_tokens': int(sys.argv[1]), 'temperature': 0, 'stream': False,
}, ensure_ascii=False))
PY
)" -o "$first" -w '%{time_total}' "http://127.0.0.1:$port/v1/chat/completions")

    second_payload="$tmp_dir/turn-2-$repeat.json"
    python3 - "$first" "$second_tokens" >"$second_payload" <<'PY'
import json, sys
first = json.load(open(sys.argv[1], encoding='utf-8'))
answer = first['choices'][0]['message']['content']
print(json.dumps({
    'model': 'deepseek-v4-flash',
    'messages': [
        {'role': 'user', 'content': '你好'},
        {'role': 'assistant', 'content': answer},
        {'role': 'user', 'content': '请继续用一句话说明。'},
    ],
    'max_tokens': int(sys.argv[2]), 'temperature': 0, 'stream': False,
}, ensure_ascii=False))
PY
    second_seconds=$(curl -fsS --connect-timeout 5 --max-time 900 \
        -H 'Content-Type: application/json' -d @"$second_payload" \
        -o "$second" -w '%{time_total}' "http://127.0.0.1:$port/v1/chat/completions")

    completion_counts=()
    while IFS= read -r value; do
        completion_counts+=("$value")
    done < <(python3 - "$first" "$second" <<'PY'
import json, sys
for path in sys.argv[1:]:
    value = json.load(open(path, encoding='utf-8')).get('usage', {}).get('completion_tokens')
    if not isinstance(value, int) or value <= 0:
        raise SystemExit(f'invalid completion token count in {path}: {value!r}')
    print(value)
PY
)
    if [[ ${#completion_counts[@]} -ne 2 ]]; then
        echo "missing completion counts" >&2
        exit 1
    fi
    actual_first=${completion_counts[0]}
    actual_second=${completion_counts[1]}
    traces=()
    while IFS= read -r value; do
        traces+=("$value")
    done < <(sed -n -E 's/.*trace token\[[0-9]+\]=([0-9]+).*/\1/p' "$log" | awk '
        NR <= first { a = a (NR == 1 ? "" : ",") $0; next }
        NR <= first + second { n = NR - first; b = b (n == 1 ? "" : ",") $0 }
        END { print a; print b }
    ' first="$actual_first" second="$actual_second")
    if [[ ${#traces[@]} -ne 2 ]] || [[ -z ${traces[0]} ]] || [[ -z ${traces[1]} ]]; then
        echo "missing turn trace in $log" >&2
        exit 1
    fi
    hashes=$(python3 - "$first" "$second" <<'PY'
import hashlib, json, sys
for path in sys.argv[1:]:
    text = json.load(open(path, encoding='utf-8'))['choices'][0]['message']['content']
    print(hashlib.sha256(text.encode()).hexdigest())
PY
)
    first_hash=$(printf '%s\n' "$hashes" | sed -n '1p')
    second_hash=$(printf '%s\n' "$hashes" | sed -n '2p')
    if [[ -z $first_trace_ref ]]; then
        first_trace_ref=${traces[0]}; second_trace_ref=${traces[1]}
        first_hash_ref=$first_hash; second_hash_ref=$second_hash
    elif [[ ${traces[0]} != "$first_trace_ref" || ${traces[1]} != "$second_trace_ref" ||
            $first_hash != "$first_hash_ref" || $second_hash != "$second_hash_ref" ]]; then
        echo "session continuity exact regression in repeat $repeat" >&2
        exit 1
    fi
    printf 'repeat=%s turn1_tokens=%s turn1_seconds=%s turn1_tps=%s turn2_tokens=%s turn2_seconds=%s turn2_tps=%s turn1_trace=%s turn2_trace=%s turn1_sha256=%s turn2_sha256=%s\n' \
        "$repeat" "$actual_first" "$first_seconds" "$(python3 - "$actual_first" "$first_seconds" <<'PY'
import sys
print(f'{int(sys.argv[1])/float(sys.argv[2]):.4f}')
PY
)" "$actual_second" "$second_seconds" "$(python3 - "$actual_second" "$second_seconds" <<'PY'
import sys
print(f'{int(sys.argv[1])/float(sys.argv[2]):.4f}')
PY
)" "${traces[0]}" "${traces[1]}" "$first_hash" "$second_hash" >>"$out_dir/runs.tsv"
    kill -TERM "$server_pid"
    wait "$server_pid" || true
    server_pid=

    if [[ $fresh_control == 1 ]]; then
        control_log="$out_dir/server-fresh-control-run-$repeat.log"
        control_kv="$tmp_dir/kv-control-$repeat"
        mkdir -p "$control_kv"
        env \
            DS4F_SERVER_HOST=127.0.0.1 \
            DS4F_SERVER_PORT="$port" \
            DS4F_SERVER_KV_CACHE_DIR="$control_kv" \
            DS4F_SERVER_KV_CACHE_MIB=64 \
            DS4F_SERVER_KV_CACHE_MIN_TOKENS="$kv_cache_min_tokens" \
            DS4_METAL_STREAMING_EXPERT_PREAD_PROFILE=1 \
            DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1 \
            DS4_METAL_STREAMING_EXPERT_LAYER_STATS=1 \
            DS4_METAL_STREAMING_EXPERT_LAYER_STATS_DELTA=1 \
            DS4_SERVER_EXPERT_PHASE_PROFILE=1 \
            DS4_SERVER_TRACE_TOKEN_IDS=1 \
            "$script_dir/run-server.sh" >"$control_log" 2>&1 &
        server_pid=$!
        wait_for_server "$control_log"
        control_seconds=$(curl -fsS --connect-timeout 5 --max-time 900 \
            -H 'Content-Type: application/json' -d @"$second_payload" \
            -o "$control" -w '%{time_total}' "http://127.0.0.1:$port/v1/chat/completions")
        control_tokens=$(python3 - "$control" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding='utf-8')).get('usage', {}).get('completion_tokens')
if not isinstance(value, int) or value <= 0:
    raise SystemExit(f'invalid completion token count: {value!r}')
print(value)
PY
)
        control_trace=$(sed -n -E 's/.*trace token\[[0-9]+\]=([0-9]+).*/\1/p' "$control_log" | paste -sd, -)
        control_hash=$(python3 - "$control" <<'PY'
import hashlib, json, sys
text = json.load(open(sys.argv[1], encoding='utf-8'))['choices'][0]['message']['content']
print(hashlib.sha256(text.encode()).hexdigest())
PY
)
        if [[ $control_tokens != "$actual_second" || $control_trace != "${traces[1]}" ||
                $control_hash != "$second_hash" ]]; then
            echo "fresh-control exact regression in repeat $repeat" >&2
            exit 1
        fi
        printf 'repeat=%s control_tokens=%s control_seconds=%s control_tps=%s control_trace=%s control_sha256=%s\n' \
            "$repeat" "$control_tokens" "$control_seconds" \
            "$(python3 - "$control_tokens" "$control_seconds" <<'PY'
import sys
print(f'{int(sys.argv[1])/float(sys.argv[2]):.4f}')
PY
)" "$control_trace" "$control_hash" >>"$out_dir/fresh-control.tsv"
        kill -TERM "$server_pid"
        wait "$server_pid" || true
        server_pid=
    fi
done

echo "exact session continuity benchmark passed: $out_dir"
