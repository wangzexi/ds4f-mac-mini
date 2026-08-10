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
tokens=${DS4F_SERVER_TOKENS:-128}
cache_experts=${DS4F_SERVER_CACHE_EXPERTS:-600}

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
    "$project_dir/ds4f-server" \
    --model "$model" \
    --metal \
    --ssd-streaming \
    --ssd-streaming-cache-experts "$cache_experts" \
    --ctx "$context" \
    --tokens "$tokens" \
    --batched-session 1 \
    --host "$host" \
    --port "$port" \
    --cors \
    "$@"
