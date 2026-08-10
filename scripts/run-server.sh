#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
runtime_dir="$project_dir/runtime"
model="$project_dir/models/DeepSeek-V4-Flash-0731-Mini-Q4Trunk-IQ2Experts.gguf"
pack="$project_dir/models/DeepSeek-V4-Flash-0731-IQ2Experts-packed.bin"

host=${DS4F_SERVER_HOST:-0.0.0.0}
port=${DS4F_SERVER_PORT:-8000}
context=${DS4F_SERVER_CONTEXT:-32768}
tokens=${DS4F_SERVER_TOKENS:-4096}
cache_experts=${DS4F_SERVER_CACHE_EXPERTS:-600}
working_set_mib=${DS4F_SERVER_WORKING_SET_MIB:-11776}
pinned_mib=${DS4F_SERVER_PINNED_MIB:-4096}
reserve_mib=${DS4F_SERVER_MEMORY_RESERVE_MIB:-512}
kv_cache_dir=${DS4F_SERVER_KV_CACHE_DIR:-$project_dir/cache/kv}
kv_cache_mib=${DS4F_SERVER_KV_CACHE_MIB:-10240}
kv_cache_min_tokens=${DS4F_SERVER_KV_CACHE_MIN_TOKENS:-1}

if [ ! -x "$project_dir/ds4f-server" ]; then
    echo "missing ds4f-server; run: make server" >&2
    exit 1
fi
if [ ! -r "$model" ] || [ ! -r "$pack" ]; then
    echo "fixed model or packed expert sidecar is missing from $project_dir/models" >&2
    exit 1
fi

cd "$runtime_dir"
exec env \
    DS4_METAL_STREAMING_EXPERT_PACK_PATH="$pack" \
    DS4_METAL_ENABLE_STREAMING_IQ2_CPU_ROUTER=1 \
    DS4_METAL_PREFILL_STAGE_ALIAS=1 \
    DS4_SERVER_WORKING_SET_MIB="$working_set_mib" \
    DS4_SERVER_PINNED_MIB="$pinned_mib" \
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
