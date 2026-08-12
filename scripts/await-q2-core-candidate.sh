#!/usr/bin/env bash
# Fixed unattended build pipeline for the one useful first Q2 experiment.
# It never touches Mini's active model or production server.
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
win_host=${DS4F_WIN_HOST:-win}
poll_seconds=${DS4F_Q2_POLL_SECONDS:-60}
wait_once=${DS4F_Q2_WAIT_ONCE:-0}

hf_dir=/cygdrive/d/ds4f-build/hf
template=/cygdrive/d/ds4f-build/template/DeepSeek-V4-Flash-0731-Mini-Q4Trunk-IQ2Experts.gguf
tools_dir=/cygdrive/d/ds4f-build/q2-tools
out_dir=/cygdrive/d/ds4f-build/q2-candidates
out=$out_dir/DeepSeek-V4-Flash-0731-Mini-Q2CoreTrunk-IQ2Experts.gguf
python=/cygdrive/c/Users/Zexi/AppData/Local/Programs/Python/Python312/python.exe

win_bash() {
    ssh "$win_host" 'C:\cygwin64\bin\bash.exe -s' \
        <<< $'PATH=/usr/bin:/bin\nexport PATH\n'"$1"
}

download_complete() {
    local shard_count lock
    shard_count=$(win_bash "find '$hf_dir' -maxdepth 1 -type f -name 'model-*-of-00046.safetensors' | wc -l")
    lock=$(win_bash "find '$hf_dir/.cache/huggingface/download' -type f -name '*.lock' -print -quit 2>/dev/null")
    [[ $shard_count == 46 && -z $lock ]]
}

while ! download_complete; do
    shard_count=$(win_bash "find '$hf_dir' -maxdepth 1 -type f -name 'model-*-of-00046.safetensors' | wc -l")
    lock=$(win_bash "find '$hf_dir/.cache/huggingface/download' -type f -name '*.lock' -printf '%f\\n' -quit 2>/dev/null")
    printf 'waiting for official weights: shards=%s/46 active=%s\n' \
        "${shard_count:-0}" "${lock:-none}" >&2
    if [[ $wait_once == 1 ]]; then
        exit 4
    fi
    sleep "$poll_seconds"
done

printf 'official snapshot is complete; resuming verified Q4 template transfer\n' >&2
"$script_dir/resume-template-to-windows.sh"

win_bash "
set -eu
test -x '$tools_dir/deepseek4-quantize-preserve'
test -f '$template'
test ! -e '$out'
mkdir -p '$out_dir'
HF_DIR='$hf_dir' TEMPLATE='$template' OUT='$out' \\
QUANTIZER='$tools_dir/deepseek4-quantize-preserve' THREADS=8 \\
'$tools_dir/build-q2-core-model.sh'
"

win_bash "
set -eu
'$python' D:/ds4f-build/q2-tools/verify_q2_core_model.py --profile core \\
    D:/ds4f-build/template/DeepSeek-V4-Flash-0731-Mini-Q4Trunk-IQ2Experts.gguf \\
    D:/ds4f-build/q2-candidates/DeepSeek-V4-Flash-0731-Mini-Q2CoreTrunk-IQ2Experts.gguf
'$python' D:/ds4f-build/q2-tools/verify_routed_copy.py \\
    D:/ds4f-build/template/DeepSeek-V4-Flash-0731-Mini-Q4Trunk-IQ2Experts.gguf \\
    D:/ds4f-build/q2-candidates/DeepSeek-V4-Flash-0731-Mini-Q2CoreTrunk-IQ2Experts.gguf
sha256sum '$out'
"

printf 'Q2 core candidate is built and structurally verified on Windows: %s\n' "$out" >&2
