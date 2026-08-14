#!/bin/sh
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 PROMPT|@FILE [MAX_TOKENS]" >&2
    exit 2
fi

prompt_input=$1
max_tokens=${2:-32}
cache_experts=${DS4F_FAST_CACHE_EXPERTS:-600}
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
model="$project_dir/models/DeepSeek-V4-Flash-0731-Mini-Q4Trunk-IQ2Experts.gguf"
pack="$project_dir/models/DeepSeek-V4-Flash-0731-IQ2Experts-packed.bin"

case "$prompt_input" in
    @*)
        prompt_file=${prompt_input#@}
        if [ ! -r "$prompt_file" ]; then
            echo "prompt file is not readable: $prompt_file" >&2
            exit 1
        fi
        prompt_bytes=$(wc -c < "$prompt_file" | tr -d ' ')
        if [ "$prompt_bytes" -gt 500000 ]; then
            echo "prompt file exceeds the 500000-byte argv safety limit" >&2
            exit 2
        fi
        prompt=$(cat -- "$prompt_file")
        ;;
    *) prompt=$prompt_input ;;
esac

if [ ! -x "$project_dir/ds4f-q4-speed" ]; then
    echo "missing runner; run: make ds4f-q4-speed" >&2
    exit 1
fi
if [ ! -r "$model" ] || [ ! -r "$pack" ]; then
    echo "fixed model or packed expert sidecar is missing" >&2
    exit 1
fi

exec env \
    DS4F_FAST_CACHE_EXPERTS="$cache_experts" \
    DS4_METAL_STREAMING_EXPERT_PACK_PATH="$pack" \
    DS4_METAL_ENABLE_STREAMING_IQ2_CPU_ROUTER=1 \
    "$project_dir/ds4f-q4-speed" "$model" "$prompt" "$max_tokens"
