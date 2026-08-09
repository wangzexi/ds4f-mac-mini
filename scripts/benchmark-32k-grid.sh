#!/bin/sh
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 MODEL.gguf [OUT_DIR]" >&2
    exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
model=$1
out_dir=${2:-"/tmp/ds4f-32k-grid-$(date +%Y%m%d-%H%M%S)"}
caches=${DS4F_BENCH_CACHE_LIST:-"440 560 700 850"}
masses=${DS4F_BENCH_MASS_LIST:-"100 95 90 85 80"}
tokens=${DS4F_BENCH_TOKENS:-32}
prompt=${DS4F_BENCH_PROMPT:-"Explain in two short paragraphs why a database should verify a backup before destructive maintenance."}

mkdir -p "$out_dir"
summary="$out_dir/summary.tsv"
printf 'cache_experts\tmass_pct\tstatus\tprefill_tps\tgeneration_tps\teffective_cap\toutput\n' > "$summary"

for cache in $caches; do
    for mass in $masses; do
        stem="cache-${cache}-mass-${mass}"
        stdout="$out_dir/$stem.out"
        stderr="$out_dir/$stem.err"
        status=0
        DS4F_FAST_CONTEXT_K=32 \
        DS4F_FAST_CACHE_EXPERTS=$cache \
        DS4F_SPEED_CACHE_AWARE_MASS_PCT=$mass \
        "$project_dir/ds4f-speed" "$model" "$prompt" "$tokens" \
            >"$stdout" 2>"$stderr" || status=$?

        prefill=$(sed -n 's/.*prefill: \([0-9.][0-9.]*\) t\/s.*/\1/p' "$stderr" | tail -1)
        generation=$(sed -n 's/.*generation: \([0-9.][0-9.]*\) t\/s.*/\1/p' "$stderr" | tail -1)
        effective=$(sed -n 's/.*runtime cache cap now \([0-9][0-9]*\) experts.*/\1/p' "$stderr" | tail -1)
        [ -n "$effective" ] || effective=$cache
        rendered=$(tr '\n\t' '  ' < "$stdout")
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$cache" "$mass" "$status" "${prefill:-NA}" \
            "${generation:-NA}" "$effective" "$rendered" >> "$summary"
        printf 'finished cache=%s mass=%s status=%s generation=%s\n' \
            "$cache" "$mass" "$status" "${generation:-NA}" >&2
    done
done

echo "$summary"
