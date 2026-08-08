#include "ds4f_gguf.h"

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { TARGET_LAYERS = 3, STAGES = 3, BLOCK = 5, MARKOV = 256, VOCAB = 129280 };

static int fail_count;

static const ds4f_tensor *must(const ds4f_gguf *g, char *name) {
    const ds4f_tensor *t = ds4f_gguf_find(g, name);
    if (!t) {
        fprintf(stderr, "missing: %s\n", name);
        fail_count++;
    }
    return t;
}

static void check_tensor(const ds4f_gguf *g, char *name, uint32_t type,
                         uint64_t d0, uint64_t d1) {
    const ds4f_tensor *t = must(g, name);
    if (!t) return;
    if (t->type != type || t->n_dims != 2 || t->dims[0] != d0 || t->dims[1] != d1) {
        fprintf(stderr, "bad layout: %s type=%s dims=%llu,%llu\n", name,
                ds4f_type_name(t->type), (unsigned long long)t->dims[0],
                (unsigned long long)(t->n_dims > 1 ? t->dims[1] : 1));
        fail_count++;
    }
}

static void check_expert(const ds4f_gguf *g, char *name, uint32_t type,
                         uint64_t d0, uint64_t d1, uint64_t d2) {
    const ds4f_tensor *t = must(g, name);
    if (!t) return;
    if (t->type != type || t->n_dims != 3 || t->dims[0] != d0 ||
        t->dims[1] != d1 || t->dims[2] != d2) {
        fprintf(stderr, "bad expert: %s type=%s dims=%llu,%llu,%llu\n", name,
                ds4f_type_name(t->type), (unsigned long long)t->dims[0],
                (unsigned long long)(t->n_dims > 1 ? t->dims[1] : 1),
                (unsigned long long)(t->n_dims > 2 ? t->dims[2] : 1));
        fail_count++;
    }
}

static void check_vector(const ds4f_gguf *g, char *name, uint32_t type,
                         uint64_t d0) {
    const ds4f_tensor *t = must(g, name);
    if (!t) return;
    if (t->type != type || t->n_dims != 1 || t->dims[0] != d0) {
        fprintf(stderr, "bad vector: %s type=%s dims=%llu\n", name,
                ds4f_type_name(t->type), (unsigned long long)t->dims[0]);
        fail_count++;
    }
}

static void stage_check(const ds4f_gguf *g, int stage) {
    char n[128];
#define V(S, T, D) do { snprintf(n, sizeof(n), "mtp.%d.%s.weight", stage, S); check_vector(g, n, T, D); } while (0)
#define M(S, T, A, B) do { snprintf(n, sizeof(n), "mtp.%d.%s.weight", stage, S); check_tensor(g, n, T, A, B); } while (0)
    V("attn_kv_a_norm", 0, 512); V("attn_norm", 0, 4096); V("attn_q_a_norm", 0, 1024);
    V("ffn_norm", 0, 4096); V("hc_attn_base", 0, 24); V("hc_attn_scale", 0, 3);
    V("hc_ffn_base", 0, 24); V("hc_ffn_scale", 0, 3);
    snprintf(n, sizeof(n), "mtp.%d.exp_probs_b.bias", stage); check_vector(g, n, 0, 256);
    M("attn_kv", 8, 4096, 512); M("attn_output_a", 8, 4096, 8192);
    M("attn_output_b", 8, 8192, 4096); M("attn_q_a", 8, 4096, 1024);
    M("attn_q_b", 8, 1024, 32768); M("ffn_gate_inp", 8, 4096, 256);
    M("ffn_gate_shexp", 8, 4096, 2048); M("ffn_up_shexp", 8, 4096, 2048);
    M("ffn_down_shexp", 8, 2048, 4096);
    snprintf(n, sizeof(n), "mtp.%d.ffn_gate_exps.weight", stage); check_expert(g, n, 16, 4096, 2048, 256);
    snprintf(n, sizeof(n), "mtp.%d.ffn_up_exps.weight", stage); check_expert(g, n, 16, 4096, 2048, 256);
    snprintf(n, sizeof(n), "mtp.%d.ffn_down_exps.weight", stage); check_expert(g, n, 10, 2048, 4096, 256);
    M("hc_attn_fn", 1, 16384, 24); M("hc_ffn_fn", 1, 16384, 24);
#undef V
#undef M
}

static void print_streaming_inventory(const ds4f_gguf *g) {
    uint64_t resident = 0;
    uint64_t routed = 0;
    uint64_t stage_routed[STAGES] = {0};
    for (uint64_t i = 0; i < g->tensor_count; ++i) {
        const ds4f_tensor *t = &g->tensors[i];
        const bool is_routed =
            strstr(t->name, ".ffn_gate_exps.weight") != NULL ||
            strstr(t->name, ".ffn_up_exps.weight") != NULL ||
            strstr(t->name, ".ffn_down_exps.weight") != NULL;
        if (!is_routed) {
            resident += t->nbytes;
            continue;
        }
        routed += t->nbytes;
        int stage = -1;
        if (sscanf(t->name, "mtp.%d.", &stage) == 1 &&
            stage >= 0 && stage < STAGES) {
            stage_routed[stage] += t->nbytes;
        }
    }
    printf("dspark streaming inventory: resident=%.2f GiB routed=%.2f GiB\n",
           (double)resident / 1073741824.0,
           (double)routed / 1073741824.0);
    for (int stage = 0; stage < STAGES; ++stage) {
        printf("  stage %d routed=%.2f GiB per_expert=%.2f MiB\n",
               stage,
               (double)stage_routed[stage] / 1073741824.0,
               (double)stage_routed[stage] / 256.0 / 1048576.0);
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s TARGET.gguf DSPARK_SUPPORT.gguf\n", argv[0]);
        return 2;
    }
    ds4f_gguf target, support;
    if (ds4f_gguf_open(&target, argv[1]) || ds4f_gguf_open(&support, argv[2])) {
        perror("GGUF");
        return 1;
    }
    printf("target tensors=%llu support tensors=%llu\n",
           (unsigned long long)target.tensor_count,
           (unsigned long long)support.tensor_count);
    if (support.dspark_stage_count != STAGES || support.dspark_n_layers != STAGES ||
        support.dspark_block_size != BLOCK || support.dspark_markov_rank != MARKOV ||
        support.dspark_target_layer_count != TARGET_LAYERS ||
        support.dspark_target_layers[0] != 40 || support.dspark_target_layers[1] != 41 ||
        support.dspark_target_layers[2] != 42 || support.dspark_noise_token_id >= VOCAB) {
        fprintf(stderr, "unexpected DSpark metadata: stages=%u layers=%u block=%u rank=%u targets=%u\n",
                support.dspark_stage_count, support.dspark_n_layers,
                support.dspark_block_size, support.dspark_markov_rank,
                support.dspark_target_layer_count);
        fail_count++;
    }
    const ds4f_tensor *main_proj = ds4f_gguf_find(&support, "mtp.0.main_proj.weight");
    if (!main_proj || main_proj->type != 8 || main_proj->n_dims != 2 ||
        main_proj->dims[0] != TARGET_LAYERS * 4096 || main_proj->dims[1] != 4096) {
        fprintf(stderr, "bad main_proj layout\n");
        fail_count++;
    }
    for (int s = 0; s < STAGES; ++s) stage_check(&support, s);
    check_tensor(&support, "mtp.2.confidence_head.proj.weight", 8, 4352, 1);
    check_tensor(&support, "mtp.2.markov_head.markov_w1.weight", 8, MARKOV, VOCAB);
    check_tensor(&support, "mtp.2.markov_head.markov_w2.weight", 8, MARKOV, VOCAB);
    check_vector(&support, "mtp.2.norm.weight", 0, 4096);
    print_streaming_inventory(&support);
    if (fail_count) printf("dspark probe: FAIL (%d issues)\n", fail_count);
    else printf("dspark probe: OK; Flash 0731 support is structurally compatible\n");
    ds4f_gguf_close(&support); ds4f_gguf_close(&target);
    return fail_count ? 1 : 0;
}
