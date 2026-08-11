#!/bin/sh
set -eu

: "${HF_DIR:?set HF_DIR to the official DeepSeek-V4-Flash snapshot}"
: "${TEMPLATE:?set TEMPLATE to the verified 0731 Q4/IQ2 GGUF}"
: "${OUT:?set OUT to the new GGUF path}"

QUANTIZER=${QUANTIZER:-"$HOME/workspace/ds4f-mini-build/ds4/gguf-tools/deepseek4-quantize"}
THREADS=${THREADS:-8}

run_quantizer() {
    "$QUANTIZER" \
        --hf "$HF_DIR" \
        --template "$TEMPLATE" \
        --out "$OUT" \
        --preserve-routed-template \
        --attention-proj q2_k \
        --shared q2_k \
        --embedding q2_k \
        --output q2_k \
        --threads "$THREADS" "$@"
}

plan=$(run_quantizer --dry-run)
type_changes=$(printf '%s\n' "$plan" | sed -n 's/^type_changes: //p')
file_bytes=$(printf '%s\n' "$plan" | sed -n 's/^approx_file_bytes: //p')
if [ "$type_changes" != 346 ] || [ -z "$file_bytes" ] ||
   [ "$file_bytes" -ge 82853553024 ]; then
    printf '%s\n' "$plan" >&2
    printf 'unexpected Q2-full dry-run contract: type_changes=%s file_bytes=%s\n' \
        "${type_changes:-missing}" "${file_bytes:-missing}" >&2
    exit 1
fi
printf 'validated Q2-full dry-run: type_changes=%s file_bytes=%s\n' \
    "$type_changes" "$file_bytes" >&2

run_quantizer
