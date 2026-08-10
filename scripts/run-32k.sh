#!/bin/sh
set -eu

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "usage: $0 exact|balanced|turbo PROMPT|@FILE [MAX_TOKENS]" >&2
    exit 2
fi

mode=$1
prompt_input=$2
max_tokens=${3:-32}
cache_experts=${DS4F_FAST_CACHE_EXPERTS:-600}
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
model="$project_dir/reference-ds4/gguf/DeepSeek-V4-Flash-0731-Mini-Q4Trunk-IQ2Experts.gguf"
pack="$project_dir/reference-ds4/gguf/DeepSeek-V4-Flash-0731-IQ2Experts-packed.bin"

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

case "$mode" in
    exact) mass=100; max_entropy=100 ;;
    balanced) mass=70; max_entropy=100 ;;
    turbo)
        mass=30
        max_entropy=100
        if [ "$max_tokens" -gt 32 ]; then
            echo "turbo is limited to 32 output tokens because longer runs drift" >&2
            exit 2
        fi
        ;;
    *)
        echo "unknown mode: $mode" >&2
        exit 2
        ;;
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
    DS4F_FAST_CONTEXT_K=32 \
    DS4F_FAST_CACHE_EXPERTS="$cache_experts" \
    DS4F_SPEED_CACHE_AWARE_MASS_PCT="$mass" \
    DS4F_SPEED_CACHE_AWARE_MAX_ENTROPY_PCT="$max_entropy" \
    DS4_METAL_STREAMING_EXPERT_PACK_PATH="$pack" \
    DS4_METAL_ENABLE_STREAMING_IQ2_CPU_ROUTER=1 \
    "$project_dir/ds4f-q4-speed" "$model" "$prompt" "$max_tokens"
