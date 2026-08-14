#!/usr/bin/env bash
# Run one server benchmark without leaving the M4 Mini's production server
# down.  The caller must pass the exact, already-inspected PID to stop.
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
stop_pid=${DS4F_BENCH_STOP_PID:?set DS4F_BENCH_STOP_PID to the inspected production PID}
restore_log=${DS4F_BENCH_RESTORE_LOG:-"$project_dir/results/server/production-after-exclusive-bench-$(date +%Y%m%d-%H%M%S).log"}
bench_kind=${DS4F_BENCH_KIND:-io-matrix}

if [[ ! $stop_pid =~ ^[1-9][0-9]*$ ]] || ! kill -0 "$stop_pid" 2>/dev/null; then
    echo "DS4F_BENCH_STOP_PID is not a live PID: $stop_pid" >&2
    exit 2
fi
if ! ps -p "$stop_pid" -o command= | grep -Fq "$project_dir/ds4f-server"; then
    echo "DS4F_BENCH_STOP_PID does not name this project's server: $stop_pid" >&2
    exit 2
fi

restore() {
    local pids
    pids=$(pgrep -x ds4f-server || true)
    if [[ -n $pids ]]; then
        kill -TERM $pids 2>/dev/null || true
    fi
    for _ in $(seq 1 30); do
        pgrep -x ds4f-server >/dev/null || break
        sleep 1
    done
    mkdir -p "$(dirname -- "$restore_log")"
    # Benchmark policy and diagnostic overrides must never leak into the
    # restored production server.  In particular, decode ablations preserve
    # dispatch timing by intentionally retaining stale buffers, so allowing
    # DS4_GLM_DECODE_ABLATE to survive this exec would make production answer
    # garbage.  Production uses one-token Decode probation, split threshold
    # 2, and no diagnostic routing/Metal profiling state.
    nohup env \
        -u DS4_GLM_DECODE_ABLATE \
        -u DS4_METAL_DECODE_STAGE_PROFILE \
        -u DS4_METAL_DECODE_STAGE_PROFILE_LAYER \
        -u DS4_METAL_DIAG_POST_PREFILL_EXPERT_HOTLIST \
        -u DS4_SERVER_DIAG_CLEAR_EXPERT_CACHE_BEFORE_PREFILL \
        -u DS4_MOE_RECORD_SELECTED_IDS \
        -u DS4_MOE_RECORD_SELECTED_IDS_TAGGED \
        -u DS4_MOE_RECORD_SELECTED_IDS_SYNC \
        -u DS4_METAL_GRAPH_TOKEN_PROFILE \
        -u DS4_METAL_GRAPH_TOKEN_PROFILE_SPLIT \
        -u DS4_METAL_ENABLE_STREAMING_EXPERT_ADDR_TABLE \
        -u DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE \
        -u DS4_METAL_ENABLE_STREAMING_COMPACT_ADDR \
        -u DS4_METAL_ENABLE_STREAMING_FULL_EXPERT_ADDR_TABLE \
        -u DS4_METAL_ENABLE_STREAMING_EXPERT_HIT_VALIDATOR \
        -u DS4_METAL_ENABLE_STREAMING_EXPERT_MASKED_ADDR \
        -u DS4_METAL_STREAMING_EXPERT_CHURN_PROFILE \
        -u DS4_METAL_STREAMING_EXPERT_PROBATION_LOOKUPS \
        -u DS4_METAL_ENABLE_STREAMING_PREFILL_CACHE_SEED \
        -u DS4_METAL_STREAMING_PREFILL_CACHE_SEED_K \
        -u DS4_METAL_STREAMING_PREFILL_CACHE_SEED_PROFILE \
        -u DS4_METAL_DISABLE_STREAMING_LAYER_BATCH \
        -u DS4_METAL_DISABLE_STREAMING_EXPERT_SPLIT \
        -u DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP \
        -u DS4_METAL_DISABLE_STREAMING_STATIC_MAP_STATE_CACHE \
        -u DS4_METAL_ENABLE_DECODE_HASH_PREFETCH \
        -u DS4_METAL_DISABLE_DECODE_HASH_PREFETCH \
        -u DS4_METAL_DECODE_HASH_PREFETCH_MASK \
        -u DS4F_SERVER_WORKING_SET_MIB \
        -u DS4F_SERVER_MEMORY_RESERVE_MIB \
        -u DS4F_SERVER_CACHE_EXPERTS \
        -u DS4F_SERVER_PREAD_THREADS \
        -u DS4F_SERVER_PINNED_MIB \
        -u DS4F_SERVER_DECODE_PINNED_MIB \
        DS4F_SERVER_DECODE_EVICTION_POLICY=probation-lru \
        DS4F_SERVER_DECODE_SPLIT_MIN_MISSES=2 \
        "$script_dir/run-server.sh" >"$restore_log" 2>&1 < /dev/null &
}
trap restore EXIT HUP INT TERM

kill -TERM "$stop_pid"
for _ in $(seq 1 30); do
    pgrep -x ds4f-server >/dev/null || break
    sleep 1
done
if pgrep -x ds4f-server >/dev/null; then
    echo "server did not stop before benchmark" >&2
    exit 1
fi

case "$bench_kind" in
    io-matrix)
        "$script_dir/bench-server-exact-io-matrix.sh"
        ;;
    resident-token)
        : "${DS4F_BENCH_PROMPT:?set DS4F_BENCH_PROMPT for resident-token}"
        "$script_dir/measure-decode-resident-token.sh" \
            "$DS4F_BENCH_PROMPT" \
            "${DS4F_BENCH_OUT_DIR:-$project_dir/results/measurements/decode-resident-token-$(date +%Y%m%d-%H%M%S)}"
        ;;
    session-continuity)
        "$script_dir/bench-server-session-continuity.sh"
        ;;
    *)
        echo "unknown DS4F_BENCH_KIND: $bench_kind (expected io-matrix, resident-token, or session-continuity)" >&2
        exit 2
        ;;
esac
