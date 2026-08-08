#include "ds4f_gguf.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void gib(uint64_t n) { printf("%.3f GiB", (double)n / (1024.0*1024*1024)); }

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) { fprintf(stderr, "usage: %s MODEL.gguf [name-prefix]\n", argv[0]); return 2; }
    ds4f_gguf g;
    if (ds4f_gguf_open(&g, argv[1])) { perror("ds4f_gguf_open"); return 1; }
    printf("format: GGUF v%u, tensors=%" PRIu64 ", data_start=%" PRIu64 "\n",
           g.version, g.tensor_count, g.data_start);
    printf("file: "); gib(g.file_size); printf(", alignment=%" PRIu64 "\n", g.alignment);
    if (g.dspark_block_size || g.dspark_markov_rank || g.dspark_target_layer_count) {
        printf("dspark: stages=%u layers=%u block=%u markov_rank=%u noise_token=%u target_layers=%u",
               g.dspark_stage_count, g.dspark_n_layers, g.dspark_block_size,
               g.dspark_markov_rank,
               g.dspark_noise_token_id, g.dspark_target_layer_count);
        for (uint32_t i = 0; i < g.dspark_target_layer_count; ++i)
            printf(" %u", g.dspark_target_layers[i]);
        putchar('\n');
    }
    uint64_t global = 0, layers[128] = {0}; unsigned layer_count = 0;
    for (uint64_t i = 0; i < g.tensor_count; ++i) {
        ds4f_tensor *t = &g.tensors[i];
        if (t->layer < 0) global += t->nbytes;
        else if (t->layer < 128) { layers[t->layer] += t->nbytes; if ((unsigned)t->layer + 1 > layer_count) layer_count = (unsigned)t->layer + 1; }
    }
    printf("global payload: "); gib(global); printf("\n");
    printf("layers: %u\n", layer_count);
    for (unsigned l = 0; l < layer_count; ++l) { printf("  layer %u: ", l); gib(layers[l]); printf("\n"); }
    if (argc == 3) {
        for (uint64_t i = 0; i < g.tensor_count; ++i)
            if (!strncmp(g.tensors[i].name, argv[2], strlen(argv[2])))
                printf("tensor: %s type=%s dims=%" PRIu64 "x%" PRIu64 "\n",
                       g.tensors[i].name, ds4f_type_name(g.tensors[i].type),
                       g.tensors[i].dims[0], g.tensors[i].n_dims > 1 ? g.tensors[i].dims[1] : 1);
    }
    if (g.tensor_count) {
        unsigned char probe[64] = {0};
        ds4f_tensor *t = &g.tensors[0];
        if (ds4f_gguf_read(&g, t, 0, probe, sizeof(probe))) { perror("pread tensor"); ds4f_gguf_close(&g); return 1; }
        printf("pread: %s, type=%s, bytes=%" PRIu64 ", first64=", t->name, ds4f_type_name(t->type), t->nbytes);
        for (unsigned i = 0; i < sizeof(probe); ++i) printf("%02x", probe[i]);
        printf("\n");
    }
    ds4f_gguf_close(&g);
    return 0;
}
