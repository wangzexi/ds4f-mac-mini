#!/usr/bin/env bash
# Resume the immutable Mini Q4/IQ2 template onto Windows D without staging it
# on the development Mac.  This is intentionally fixed to ds4f-mini's only
# template and destination; it is not a general file-copy utility.
set -euo pipefail

mini_host=${DS4F_MINI_HOST:-mini}
win_host=${DS4F_WIN_HOST:-win}
source=/Users/zexi/workspace/ds4f-mini/models/DeepSeek-V4-Flash-0731-Mini-Q4Trunk-IQ2Experts.gguf
destination=/cygdrive/d/ds4f-build/template/DeepSeek-V4-Flash-0731-Mini-Q4Trunk-IQ2Experts.gguf
partial=${destination}.part
block_bytes=1048576

win_bash() {
    ssh "$win_host" 'C:\cygwin64\bin\bash.exe -s' \
        <<< $'PATH=/usr/bin:/bin\nexport PATH\n'"$1"
}

# Concurrent download and template copying contend for the same external D
# drive.  Completion removes Hugging Face's lock files.  An explicit override
# exists only for recovery after independently confirming the download stopped.
if [[ ${DS4F_ALLOW_TEMPLATE_TRANSFER_DURING_DOWNLOAD:-0} != 1 ]] &&
   win_bash 'find /cygdrive/d/ds4f-build/hf/.cache/huggingface/download -name "*.lock" -print -quit 2>/dev/null | grep -q .'; then
    echo "official Hugging Face download is active; template transfer is intentionally deferred" >&2
    exit 3
fi

source_bytes=$(ssh "$mini_host" "test -r '$source' && stat -f %z '$source'")
if [[ ! $source_bytes =~ ^[0-9]+$ ]] || (( source_bytes == 0 )); then
    echo "could not read the Mini Q4 template size" >&2
    exit 1
fi

if win_bash "test -e '$destination'"; then
    echo "validated Windows template already exists: $destination" >&2
    exit 1
fi
win_bash "mkdir -p '\$(dirname '$partial')'"
partial_bytes=$(win_bash "if test -f '$partial'; then stat -c%s '$partial'; else echo 0; fi")
if [[ ! $partial_bytes =~ ^[0-9]+$ ]] || (( partial_bytes > source_bytes )); then
    echo "invalid Windows partial size: $partial_bytes (source $source_bytes)" >&2
    exit 1
fi

# Resume at an integral MiB.  Rewriting at most 1 MiB of the partial tail is
# deliberate: it avoids byte-sized dd I/O while retaining an exact final hash.
skip_blocks=$(( partial_bytes / block_bytes ))
resume_bytes=$(( skip_blocks * block_bytes ))
printf 'resuming template at %s MiB (%s / %s bytes already present)\n' \
    "$skip_blocks" "$partial_bytes" "$source_bytes" >&2

ssh "$mini_host" "dd if='$source' bs=$block_bytes skip=$skip_blocks status=progress" |
    ssh "$win_host" "C:\\cygwin64\\bin\\bash.exe -lc \"PATH=/usr/bin:/bin; export PATH; dd of='$partial' bs=$block_bytes seek=$skip_blocks conv=notrunc status=progress\""

final_bytes=$(win_bash "stat -c%s '$partial'")
if [[ $final_bytes != "$source_bytes" ]]; then
    echo "template transfer size mismatch: Windows=$final_bytes Mini=$source_bytes" >&2
    exit 1
fi

source_sha=$(ssh "$mini_host" "shasum -a 256 '$source' | sed 's/ .*//'")
dest_sha=$(win_bash "sha256sum '$partial' | sed 's/ .*//'")
if [[ $source_sha != "$dest_sha" ]]; then
    echo "template transfer SHA-256 mismatch" >&2
    exit 1
fi

win_bash "mv '$partial' '$destination'"
printf 'template transfer verified: sha256=%s\n' "$source_sha"
