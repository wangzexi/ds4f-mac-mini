#!/bin/sh
set -eu

: "${HF_DIR:?set HF_DIR to the official DeepSeek-V4-Flash snapshot}"
: "${TEMPLATE:?set TEMPLATE to the verified 0731 IQ2/Q2 GGUF}"
: "${OUT:?set OUT to the new GGUF path}"

QUANTIZER=${QUANTIZER:-"$HOME/workspace/ds4f-mini-build/ds4/gguf-tools/deepseek4-quantize"}
THREADS=${THREADS:-8}

run_quantizer() {
    "$QUANTIZER" \
        --hf "$HF_DIR" \
        --template "$TEMPLATE" \
        --out "$OUT" \
        --preserve-routed-template \
        --attention-proj q4_k \
        --shared q4_k \
        --embedding q4_k \
        --output q4_k \
        --threads "$THREADS" "$@"
}

plan=$(run_quantizer --dry-run)
type_changes=$(printf '%s\n' "$plan" | sed -n 's/^type_changes: //p')
file_bytes=$(printf '%s\n' "$plan" | sed -n 's/^approx_file_bytes: //p')
if [ "$type_changes" != 346 ] || [ "$file_bytes" != 82853553024 ]; then
    printf '%s\n' "$plan" >&2
    printf 'unexpected Q4 dry-run contract: type_changes=%s file_bytes=%s\n' \
        "$type_changes" "$file_bytes" >&2
    exit 1
fi
printf 'validated Q4 dry-run: type_changes=%s file_bytes=%s\n' \
    "$type_changes" "$file_bytes" >&2

exec "$QUANTIZER" \
    --hf "$HF_DIR" \
    --template "$TEMPLATE" \
    --out "$OUT" \
    --preserve-routed-template \
    --attention-proj q4_k \
    --shared q4_k \
    --embedding q4_k \
    --output q4_k \
    --threads "$THREADS"
