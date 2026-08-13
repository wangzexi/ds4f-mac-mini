#!/usr/bin/env bash
# Fixed-hardware exact Decode I/O sweep for the M4/16GB Flash 0731 runtime.
#
# This is deliberately not a general benchmark framework.  It isolates the
# selected-expert SSD path while keeping the deployed Q4 trunk, IQ2/Q2 expert
# sidecar, CPU router, greedy sampling, 32K context, and one fixed Chinese
# prompt unchanged.  Every run must produce the same token-ID trace as the
# first run or the sweep aborts.
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
runner="$project_dir/ds4f-q4-speed"
model="$project_dir/models/DeepSeek-V4-Flash-0731-Mini-Q4Trunk-IQ2Experts.gguf"
pack="$project_dir/models/DeepSeek-V4-Flash-0731-IQ2Experts-packed.bin"

prompt=${DS4F_BENCH_PROMPT:-你好}
tokens=${DS4F_BENCH_TOKENS:-32}
cache_experts=${DS4F_BENCH_CACHE_EXPERTS:-950}
repeats=${DS4F_BENCH_REPEATS:-2}
threads_csv=${DS4F_BENCH_PREAD_THREADS:-1,2,3,4,6,9}
out_dir=${DS4F_BENCH_OUT_DIR:-"$project_dir/results/benchmarks/exact-io-$(date +%Y%m%d-%H%M%S)"}

if [[ ! -x "$runner" || ! -r "$model" || ! -r "$pack" ]]; then
    echo "missing fixed Mini runner, model, or packed experts" >&2
    exit 2
fi
if pgrep -x ds4f-server >/dev/null 2>&1 &&
   [[ ${DS4F_BENCH_ALLOW_CONCURRENT_SERVER:-0} != 1 ]]; then
    echo "refusing to contend with production ds4f-server; stop it before this exclusive benchmark" >&2
    exit 2
fi
if [[ ! $tokens =~ ^[1-9][0-9]*$ ]] || [[ ! $cache_experts =~ ^[1-9][0-9]*$ ]] ||
   [[ ! $repeats =~ ^[1-9][0-9]*$ ]]; then
    echo "tokens, cache experts, and repeats must be positive integers" >&2
    exit 2
fi

mkdir -p "$out_dir"
printf 'pread_threads\truns\n' > "$out_dir/runs.tsv"
printf 'prompt=%q\ntokens=%s\ncache_experts=%s\nrepeats=%s\nthreads=%s\n' \
    "$prompt" "$tokens" "$cache_experts" "$repeats" "$threads_csv" \
    > "$out_dir/config.env"
# Hashing the 77+ GiB payload would itself contaminate the cold-I/O sweep.
# The immutable production model/pack are identified here by their filesystem
# identity; only the small executable is content-hashed.
shasum -a 256 "$runner" > "$out_dir/runner.sha256"
stat -f '%N\t%z\t%m\t%i' "$model" "$pack" > "$out_dir/model-pack.stat.tsv"

reference_trace=
run_index=0
IFS=, read -r -a threads_list <<< "$threads_csv"
for threads in "${threads_list[@]}"; do
    if [[ ! $threads =~ ^[1-9][0-9]*$ ]]; then
        echo "bad pread thread count: $threads" >&2
        exit 2
    fi
    for ((repeat = 1; repeat <= repeats; repeat++)); do
        run_index=$((run_index + 1))
        log="$out_dir/pread-${threads}-run-${repeat}.log"
        printf 'run=%d pread_threads=%s repeat=%s\n' "$run_index" "$threads" "$repeat" \
            | tee -a "$out_dir/runs.tsv"

        /usr/bin/time -lp env \
            DS4F_FAST_CACHE_EXPERTS="$cache_experts" \
            DS4F_FAST_TRACE_IDS=1 \
            DS4F_SPEED_CACHE_AWARE_MASS_PCT=100 \
            DS4F_SPEED_CACHE_AWARE_MAX_ENTROPY_PCT=100 \
            DS4_METAL_STREAMING_EXPERT_PACK_PATH="$pack" \
            DS4_METAL_ENABLE_STREAMING_IQ2_CPU_ROUTER=1 \
            DS4_METAL_STREAMING_EXPERT_PREAD_THREADS="$threads" \
            DS4_METAL_STREAMING_EXPERT_PREAD_POOL=1 \
            DS4_METAL_STREAMING_EXPERT_PREAD_PROFILE=1 \
            DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1 \
            DS4_METAL_STREAMING_EXPERT_LAYER_STATS=1 \
            DS4_METAL_STREAMING_EXPERT_LAYER_STATS_DELTA=1 \
            DS4_METAL_GPU_BUSY_PROFILE=1 \
            "$runner" "$model" "$prompt" "$tokens" \
            >"$log" 2>&1

        trace=$(sed -n -E 's/.*trace token\[[0-9]+\]=([0-9]+).*/\1/p' "$log" | paste -sd, -)
        if [[ -z $trace ]]; then
            echo "no greedy trace in $log" >&2
            exit 1
        fi
        if [[ -z $reference_trace ]]; then
            reference_trace=$trace
            printf '%s\n' "$reference_trace" > "$out_dir/reference-trace.csv"
        elif [[ $trace != "$reference_trace" ]]; then
            echo "exact regression failed for pread_threads=$threads repeat=$repeat" >&2
            echo "expected: $reference_trace" >&2
            echo "actual:   $trace" >&2
            exit 1
        fi
        printf 'trace=%s\n' "$trace" >> "$out_dir/runs.tsv"
    done
done

printf 'exact I/O matrix passed: %s\n' "$out_dir"
