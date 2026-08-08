#ifndef DS4F_GGUF_H
#define DS4F_GGUF_H

#include <stdint.h>

typedef struct {
    char *name;
    uint64_t dims[4];
    uint32_t n_dims;
    uint32_t type;
    uint64_t offset;
    uint64_t file_offset;
    uint64_t nbytes;
    int layer;
} ds4f_tensor;

typedef struct {
    int fd;
    uint64_t file_size;
    uint64_t data_start;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;
    uint64_t alignment;
    ds4f_tensor *tensors;
    uint32_t dspark_block_size;
    uint32_t dspark_stage_count;
    uint32_t dspark_n_layers;
    uint32_t dspark_markov_rank;
    uint32_t dspark_noise_token_id;
    uint32_t dspark_target_layer_count;
    uint32_t dspark_target_layers[8];
} ds4f_gguf;

int ds4f_gguf_open(ds4f_gguf *g, const char *path);
void ds4f_gguf_close(ds4f_gguf *g);
const ds4f_tensor *ds4f_gguf_find(const ds4f_gguf *g, const char *name);
int ds4f_gguf_read(const ds4f_gguf *g, const ds4f_tensor *t,
                   uint64_t relative_offset, void *dst, uint64_t bytes);
const char *ds4f_type_name(uint32_t type);

#endif
