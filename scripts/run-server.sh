#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
runtime_dir="$project_dir/runtime"
model=${DS4F_SERVER_MODEL:-"$project_dir/models/DeepSeek-V4-Flash-0731-Mini-Q4Trunk-IQ2Experts.gguf"}
pack=${DS4F_SERVER_EXPERT_PACK:-"$project_dir/models/DeepSeek-V4-Flash-0731-IQ2Experts-packed.bin"}

host=${DS4F_SERVER_HOST:-0.0.0.0}
port=${DS4F_SERVER_PORT:-8000}
context=${DS4F_SERVER_CONTEXT:-32768}
tokens=${DS4F_SERVER_TOKENS:-4096}
cache_experts=${DS4F_SERVER_CACHE_EXPERTS:-1800}
prefill_full_layer_parallel_pread=${DS4F_SERVER_PREFILL_FULL_LAYER_PARALLEL_PREAD:-1}
# Keep the full L0 slab in the canonical server path.  Releasing it gives a
# much faster long Prefill, but currently changes the first learned-router
# ordering at L3; it remains an explicit experiment until that numerical
# divergence is eliminated.
keep_hash_layer0=${DS4F_SERVER_KEEP_HASH_LAYER0:-1}
# Preserve the three exact hash-routed layers, then reclaim L0 before the
# learned stack.  On this fixed model that keeps the canonical L3 route while
# freeing one 256-expert slab for all later prefill/decode staging.
prefill_release_hash_layer0_after=${DS4F_SERVER_PREFILL_RELEASE_HASH_LAYER0_AFTER:-2}
# Flash Decode selects at most six routed experts per transformer layer.  On
# this Mini, a fresh 32-token exact A/B found five readers marginally faster
# than six (the sixth only contends for the same SSD); both remain exact.
pread_threads=${DS4F_SERVER_PREAD_THREADS:-5}
working_set_mib=${DS4F_SERVER_WORKING_SET_MIB:-11776}
pinned_mib=${DS4F_SERVER_PINNED_MIB:-4096}
# This M4 Mini's measured per-process wired-memory ceiling is just under
# 4 GiB (606 packed 6.75 MiB experts).  Decode may retain a larger pageable
# cache, but asking macOS to wire 6 GiB merely causes one predictable failed
# mlock on every fresh server process.
decode_pinned_mib=${DS4F_SERVER_DECODE_PINNED_MIB:-4096}
reserve_mib=${DS4F_SERVER_MEMORY_RESERVE_MIB:-512}
kv_cache_dir=${DS4F_SERVER_KV_CACHE_DIR:-$project_dir/cache/kv}
kv_cache_mib=${DS4F_SERVER_KV_CACHE_MIB:-10240}
kv_cache_min_tokens=${DS4F_SERVER_KV_CACHE_MIN_TOKENS:-1}
prefill_measurements=${DS4F_SERVER_PREFILL_MEASUREMENTS:-$project_dir/cache/prefill-measurements.tsv}
preload_static_decode=${DS4F_SERVER_PRELOAD_STATIC_DECODE_TRUNK:-1}
allow_shared_expert_sidecar=${DS4F_SERVER_ALLOW_SHARED_EXPERT_SIDECAR:-0}
# A newly read routed expert should survive until the next visit to its own
# layer.  On the fixed 43-layer/6-expert Flash path that is one Decode token
# (258 lookups).  Exact 64-token runs averaged 2.266 token/s, versus 2.239
# for plain LRU; retaining two token cycles was slower (2.253 token/s).
decode_eviction_policy=${DS4F_SERVER_DECODE_EVICTION_POLICY:-probation-lru}
# Decode on the fixed M4 Mini is usually a 4--5 resident / 1--2 missing
# expert mix.  With L2-after-L0 release, an interleaved 512-token/32-token
# A/B sweep found two misses the repeatable optimum: 2.42 token/s median,
# versus 2.40 for splitting a single miss and 2.38--2.41 for thresholds 3--6.
# All variants had the identical greedy trace.
decode_split_min_misses=${DS4F_SERVER_DECODE_SPLIT_MIN_MISSES:-2}
# The Metal GPU-address table is fast in isolation, but it made the fixed
# Flash runtime's answer depend on whether selected-expert buffers survived a
# previous request.  Keep it off until the resource-lifetime bug is fixed.
# The normal selected-slot cache remains enabled, so this does not discard the
# Mini's Decode expert residency or its SSD-read savings.
disable_streaming_expert_addr_table=${DS4F_SERVER_DISABLE_STREAMING_EXPERT_ADDR_TABLE:-1}

if [ ! -x "$project_dir/ds4f-server" ]; then
    echo "missing ds4f-server; run: make server" >&2
    exit 1
fi
if [ ! -r "$model" ] || [ ! -r "$pack" ]; then
    echo "fixed model or packed expert sidecar is missing from $project_dir/models" >&2
    exit 1
fi

# A Q2 trunk candidate has a different GGUF total size but can reuse the
# byte-identical routed-expert sidecar.  This is opt-in only after the full
# routed-payload verifier has passed; production always keeps strict matching.
case "$allow_shared_expert_sidecar" in
    0)
        unset DS4_METAL_STREAMING_EXPERT_PACK_ALLOW_SAME_EXPERTS
        ;;
    1)
        export DS4_METAL_STREAMING_EXPERT_PACK_ALLOW_SAME_EXPERTS=1
        ;;
    *)
        echo "DS4F_SERVER_ALLOW_SHARED_EXPERT_SIDECAR must be 0 or 1" >&2
        exit 2
        ;;
esac

case "$decode_eviction_policy" in
    lru|heat|reuse-heat|probation-lru)
        ;;
    *)
        echo "DS4F_SERVER_DECODE_EVICTION_POLICY must be lru, heat, reuse-heat, or probation-lru" >&2
        exit 2
        ;;
esac

case "$decode_split_min_misses" in
    1|2|3|4|5|6)
        ;;
    *)
        echo "DS4F_SERVER_DECODE_SPLIT_MIN_MISSES must be an integer 1..6" >&2
        exit 2
        ;;
esac

case "$disable_streaming_expert_addr_table" in
    0)
        unset DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE
        ;;
    1)
        export DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE=1
        ;;
    *)
        echo "DS4F_SERVER_DISABLE_STREAMING_EXPERT_ADDR_TABLE must be 0 or 1" >&2
        exit 2
        ;;
esac

case "$keep_hash_layer0" in
    0|1) ;;
    *)
        echo "DS4F_SERVER_KEEP_HASH_LAYER0 must be 0 or 1" >&2
        exit 2
        ;;
esac

case "$prefill_release_hash_layer0_after" in
    0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42) ;;
    *)
        echo "DS4F_SERVER_PREFILL_RELEASE_HASH_LAYER0_AFTER must be a layer index 0..42" >&2
        exit 2
        ;;
esac

cd "$runtime_dir"
exec env \
    DS4_METAL_STREAMING_EXPERT_PACK_PATH="$pack" \
    DS4_METAL_ENABLE_STREAMING_IQ2_CPU_ROUTER=1 \
    DS4_METAL_PREFILL_STAGE_ALIAS=1 \
    DS4_METAL_KEEP_HASH_LAYER0="$keep_hash_layer0" \
    DS4_METAL_PREFILL_RELEASE_HASH_LAYER0_AFTER="$prefill_release_hash_layer0_after" \
    DS4_METAL_PREFILL_FULL_LAYER_PARALLEL_PREAD="$prefill_full_layer_parallel_pread" \
    DS4_METAL_STREAMING_EXPERT_PREAD_THREADS="$pread_threads" \
    DS4_METAL_STREAMING_EXPERT_PREAD_POOL=1 \
    DS4_METAL_STREAMING_EXPERT_DECODE_EVICTION_POLICY="$decode_eviction_policy" \
    DS4_METAL_STREAMING_EXPERT_SPLIT_MIN_MISSES="$decode_split_min_misses" \
    DS4_METAL_PREFILL_MEASUREMENTS_PATH="$prefill_measurements" \
    DS4_SERVER_PRELOAD_STATIC_DECODE_TRUNK="$preload_static_decode" \
    DS4_SERVER_WORKING_SET_MIB="$working_set_mib" \
    DS4_SERVER_PINNED_MIB="$pinned_mib" \
    DS4_SERVER_DECODE_PINNED_MIB="$decode_pinned_mib" \
    DS4_SERVER_MEMORY_RESERVE_MIB="$reserve_mib" \
    DS4_KVSTORE_STRICT_LRU=1 \
    "$project_dir/ds4f-server" \
    --model "$model" \
    --metal \
    --ssd-streaming \
    --ssd-streaming-cache-experts "$cache_experts" \
    --ctx "$context" \
    --tokens "$tokens" \
    --kv-disk-dir "$kv_cache_dir" \
    --kv-disk-space-mb "$kv_cache_mib" \
    --kv-cache-min-tokens "$kv_cache_min_tokens" \
    --kv-cache-cold-max-tokens 0 \
    --kv-cache-continued-interval-tokens 0 \
    --batched-session 1 \
    --host "$host" \
    --port "$port" \
    --cors \
    "$@"
