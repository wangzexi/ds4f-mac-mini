#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
runtime_dir="$project_dir/runtime"
model=${DS4F_SERVER_MODEL:-"$project_dir/models/DeepSeek-V4-Flash-0731-Mini-Q4Trunk-IQ2Experts.gguf"}
pack=${DS4F_SERVER_EXPERT_PACK:-"$project_dir/models/DeepSeek-V4-Flash-0731-IQ2Experts-packed.bin"}

host=${DS4F_SERVER_HOST:-0.0.0.0}
port=${DS4F_SERVER_PORT:-8000}
context=${DS4F_SERVER_CONTEXT:-32768}
tokens=${DS4F_SERVER_TOKENS:-4096}
cache_experts=${DS4F_SERVER_CACHE_EXPERTS:-1800}
prefill_full_layer_parallel_pread=${DS4F_SERVER_PREFILL_FULL_LAYER_PARALLEL_PREAD:-1}
# Flash Decode selects at most six routed experts per transformer layer.  Six
# persistent readers fully cover that fixed I/O fan-out; the generic runtime
# default of nine only creates three permanently idle workers on this Mini.
pread_threads=${DS4F_SERVER_PREAD_THREADS:-6}
working_set_mib=${DS4F_SERVER_WORKING_SET_MIB:-11776}
pinned_mib=${DS4F_SERVER_PINNED_MIB:-4096}
# This M4 Mini's measured per-process wired-memory ceiling is just under
# 4 GiB (606 packed 6.75 MiB experts).  Decode may retain a larger pageable
# cache, but asking macOS to wire 6 GiB merely causes one predictable failed
# mlock on every fresh server process.
decode_pinned_mib=${DS4F_SERVER_DECODE_PINNED_MIB:-4096}
reserve_mib=${DS4F_SERVER_MEMORY_RESERVE_MIB:-512}
kv_cache_dir=${DS4F_SERVER_KV_CACHE_DIR:-$project_dir/cache/kv}
kv_cache_mib=${DS4F_SERVER_KV_CACHE_MIB:-10240}
kv_cache_min_tokens=${DS4F_SERVER_KV_CACHE_MIN_TOKENS:-1}
prefill_measurements=${DS4F_SERVER_PREFILL_MEASUREMENTS:-$project_dir/cache/prefill-measurements.tsv}
preload_static_decode=${DS4F_SERVER_PRELOAD_STATIC_DECODE_TRUNK:-1}
allow_shared_expert_sidecar=${DS4F_SERVER_ALLOW_SHARED_EXPERT_SIDECAR:-0}

if [ ! -x "$project_dir/ds4f-server" ]; then
    echo "missing ds4f-server; run: make server" >&2
    exit 1
fi
if [ ! -r "$model" ] || [ ! -r "$pack" ]; then
    echo "fixed model or packed expert sidecar is missing from $project_dir/models" >&2
    exit 1
fi

# A Q2 trunk candidate has a different GGUF total size but can reuse the
# byte-identical routed-expert sidecar.  This is opt-in only after the full
# routed-payload verifier has passed; production always keeps strict matching.
case "$allow_shared_expert_sidecar" in
    0)
        unset DS4_METAL_STREAMING_EXPERT_PACK_ALLOW_SAME_EXPERTS
        ;;
    1)
        export DS4_METAL_STREAMING_EXPERT_PACK_ALLOW_SAME_EXPERTS=1
        ;;
    *)
        echo "DS4F_SERVER_ALLOW_SHARED_EXPERT_SIDECAR must be 0 or 1" >&2
        exit 2
        ;;
esac

cd "$runtime_dir"
exec env \
    DS4_METAL_STREAMING_EXPERT_PACK_PATH="$pack" \
    DS4_METAL_ENABLE_STREAMING_IQ2_CPU_ROUTER=1 \
    DS4_METAL_PREFILL_STAGE_ALIAS=1 \
    DS4_METAL_KEEP_HASH_LAYER0=1 \
    DS4_METAL_PREFILL_FULL_LAYER_PARALLEL_PREAD="$prefill_full_layer_parallel_pread" \
    DS4_METAL_STREAMING_EXPERT_PREAD_THREADS="$pread_threads" \
    DS4_METAL_STREAMING_EXPERT_PREAD_POOL=1 \
    DS4_METAL_PREFILL_MEASUREMENTS_PATH="$prefill_measurements" \
    DS4_SERVER_PRELOAD_STATIC_DECODE_TRUNK="$preload_static_decode" \
    DS4_SERVER_WORKING_SET_MIB="$working_set_mib" \
    DS4_SERVER_PINNED_MIB="$pinned_mib" \
    DS4_SERVER_DECODE_PINNED_MIB="$decode_pinned_mib" \
    DS4_SERVER_MEMORY_RESERVE_MIB="$reserve_mib" \
    DS4_KVSTORE_STRICT_LRU=1 \
    "$project_dir/ds4f-server" \
    --model "$model" \
    --metal \
    --ssd-streaming \
    --ssd-streaming-cache-experts "$cache_experts" \
    --ctx "$context" \
    --tokens "$tokens" \
    --kv-disk-dir "$kv_cache_dir" \
    --kv-disk-space-mb "$kv_cache_mib" \
    --kv-cache-min-tokens "$kv_cache_min_tokens" \
    --kv-cache-cold-max-tokens 0 \
    --kv-cache-continued-interval-tokens 0 \
    --batched-session 1 \
    --host "$host" \
    --port "$port" \
    --cors \
    "$@"
