#!/usr/bin/env bash
# Run one server benchmark without leaving the M4 Mini's production server
# down.  The caller must pass the exact, already-inspected PID to stop.
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
stop_pid=${DS4F_BENCH_STOP_PID:?set DS4F_BENCH_STOP_PID to the inspected production PID}
restore_log=${DS4F_BENCH_RESTORE_LOG:-"$project_dir/results/server/production-after-exclusive-bench-$(date +%Y%m%d-%H%M%S).log"}

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
    nohup "$script_dir/run-server.sh" >"$restore_log" 2>&1 < /dev/null &
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

"$script_dir/bench-server-exact-io-matrix.sh"
