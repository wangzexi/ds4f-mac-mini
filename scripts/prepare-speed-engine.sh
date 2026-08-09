#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
source_dir="$project_dir/reference-ds4"
target_dir="$project_dir/speed-ds4"

if [ ! -d "$source_dir" ]; then
    printf '%s\n' 'ds4f-speed: missing reference-ds4' >&2
    exit 1
fi
if [ -f "$target_dir/.ds4f-speed-prepared" ]; then
    exit 0
fi
if [ -e "$target_dir" ]; then
    printf '%s\n' 'ds4f-speed: speed-ds4 exists without its preparation marker; inspect or remove it explicitly' >&2
    exit 1
fi
mkdir "$target_dir"
rsync -a --exclude=.git --exclude=gguf --exclude='*.gguf' --exclude='*.o' --exclude=ds4 --exclude='ds4-*' --exclude='*.dSYM' "$source_dir/" "$target_dir/"
cd "$target_dir"
printf '%s\n' \
'g/Experimental approximate speed mode: anchor one route per layer./s//Anchored-route speed mode: intentionally approximate MoE expert selection./' \
'g/DS4F_EXPERIMENT_ROUTE_AFTER_TOKEN/s//DS4F_SPEED_ROUTE_AFTER_TOKEN/g' \
'g/DS4F_EXPERIMENT_ANCHORED_ROUTE/s//DS4F_SPEED_ANCHORED_ROUTE/g' \
'g/DS4F_EXPERIMENT_ROUTE_KEEP_TOP/s//DS4F_SPEED_KEEP_TOP/g' \
'/        getenv("DS4F_EXPERIMENT_DISABLE_ANCHORED_ROUTE") == NULL &&/d' \
'/        uint32_t                token) {/c' \
'        uint32_t                token,' \
'        uint32_t                pos) {' \
'.' \
'g/metal_graph_decode_cpu_router(g, model, layer, il, (uint32_t)token);/s//metal_graph_decode_cpu_router(g, model, layer, il, (uint32_t)token, (uint32_t)pos);/' \
'/    float \*cpu_router_norm;/a' \
'    /* Anchored-route speed mode: intentionally approximate MoE selection. */' \
'    bool speed_anchor_route_valid[DS4_MAX_LAYER];' \
'    int speed_anchor_route_ids[DS4_MAX_LAYER][DS4_MAX_EXPERT_USED];' \
'.' \
'/    const char \*speed_after_text = getenv("DS4F_SPEED_ROUTE_AFTER_TOKEN");/a' \
'    const bool speed_mode_requested =' \
'        getenv("DS4F_SPEED_ANCHORED_ROUTE") != NULL;' \
'    if (speed_mode_requested && pos == 0) {' \
'        memset(g->speed_anchor_route_valid, 0,' \
'               sizeof(g->speed_anchor_route_valid));' \
'    }' \
'.' \
'g/getenv("DS4F_SPEED_ANCHORED_ROUTE") != NULL &&/s//speed_mode_requested \&\&/' \
'w' \
'q' | ed -s ds4.c
touch .ds4f-speed-prepared
