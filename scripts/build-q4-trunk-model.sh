#!/bin/sh
set -eu

: "${HF_DIR:?set HF_DIR to the official DeepSeek-V4-Flash snapshot}"
: "${TEMPLATE:?set TEMPLATE to the verified 0731 IQ2/Q2 GGUF}"
: "${OUT:?set OUT to the new GGUF path}"

QUANTIZER=${QUANTIZER:-"$HOME/workspace/ds4f-mini-build/ds4/gguf-tools/deepseek4-quantize"}
THREADS=${THREADS:-8}

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
