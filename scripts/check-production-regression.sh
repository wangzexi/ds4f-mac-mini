#!/usr/bin/env bash
# Verify the fixed Mini deployment against its accepted greedy token IDs.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 /path/to/ds4f-q4-speed /path/to/Flash-0731.gguf" >&2
    exit 2
fi

runner=$1
model=$2
if [[ ! -x $runner || ! -f $model ]]; then
    echo "runner or model is unavailable" >&2
    exit 2
fi

trace=$(mktemp "${TMPDIR:-/tmp}/ds4f-production-trace.XXXXXX")
trap 'rm -f "$trace"' EXIT

check_case() {
    local prompt=$1
    local count=$2
    shift 2
    local expected=("$@")
    local actual=()

    DS4F_FAST_TRACE_IDS=1 "$runner" "$model" "$prompt" "$count" >/dev/null 2>"$trace"
    while IFS= read -r token; do
        actual+=("$token")
    done < <(sed -n -E 's/.*trace token\[[0-9]+\]=([0-9]+).*/\1/p' "$trace")

    if [[ ${actual[*]} != "${expected[*]}" ]]; then
        echo "production token regression failed for: $prompt" >&2
        echo "expected: ${expected[*]}" >&2
        echo "actual:   ${actual[*]:-(none)}" >&2
        return 1
    fi
    echo "ok: ${actual[*]}"
}

check_case 'Explain one plus one in one word.' 3 6111 2004 28
check_case '用一句话解释为什么 1+1=2。' 4 23385 33951 6573 303

echo "production numerical regression passed"
