#!/bin/sh
set -eu

: "${HF_DIR:?set HF_DIR to the official DeepSeek-V4-Flash snapshot}"
: "${TEMPLATE:?set TEMPLATE to the verified 0731 Q4/IQ2 GGUF}"
: "${OUT:?set OUT to the new GGUF path}"

QUANTIZER=${QUANTIZER:-"$HOME/workspace/ds4f-mini-build/ds4/gguf-tools/deepseek4-quantize"}
THREADS=${THREADS:-8}

# Q2_K is enabled only for the generic dense path.  The attention output
# projections stay Q4_K so Prefill retains its specialized grouped/batched
# output kernels instead of falling back to one generic matmul per token.
run_quantizer() {
    set -- "$@"
    layer=0
    while [ "$layer" -lt 43 ]; do
        set -- "$@" \
            --tensor-type "blk.$layer.attn_output_a.weight=q4_k" \
            --tensor-type "blk.$layer.attn_output_b.weight=q4_k"
        layer=$((layer + 1))
    done

    "$QUANTIZER" \
        --hf "$HF_DIR" \
        --template "$TEMPLATE" \
        --out "$OUT" \
        --preserve-routed-template \
        --attention-proj q2_k \
        --shared q2_k \
        --threads "$THREADS" "$@"
}

plan=$(run_quantizer --dry-run)
type_changes=$(printf '%s\n' "$plan" | sed -n 's/^type_changes: //p')
file_bytes=$(printf '%s\n' "$plan" | sed -n 's/^approx_file_bytes: //p')
if [ "$type_changes" != 258 ] || [ -z "$file_bytes" ] ||
   [ "$file_bytes" -ge 82853553024 ]; then
    printf '%s\n' "$plan" >&2
    printf 'unexpected Q2-core dry-run contract: type_changes=%s file_bytes=%s\n' \
        "${type_changes:-missing}" "${file_bytes:-missing}" >&2
    exit 1
fi
printf 'validated Q2-core dry-run: type_changes=%s file_bytes=%s\n' \
    "$type_changes" "$file_bytes" >&2

run_quantizer
