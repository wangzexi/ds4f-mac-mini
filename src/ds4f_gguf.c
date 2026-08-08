#define _DARWIN_C_SOURCE
#include "ds4f_gguf.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct { int fd; uint64_t pos; } reader;

static int rd(reader *r, void *p, size_t n) {
    ssize_t got = pread(r->fd, p, n, (off_t)r->pos);
    if (got != (ssize_t)n) return -1;
    r->pos += n;
    return 0;
}
static int u32(reader *r, uint32_t *v) { return rd(r, v, 4); }
static int u64(reader *r, uint64_t *v) { return rd(r, v, 8); }
static int skip(reader *r, uint64_t n) {
    if (n > UINT64_MAX - r->pos) return -1;
    r->pos += n;
    return 0;
}
static int string(reader *r, char **out) {
    uint64_t n;
    if (u64(r, &n) || n > 64 * 1024 * 1024) return -1;
    char *s = calloc(1, (size_t)n + 1);
    if (!s || rd(r, s, (size_t)n)) { free(s); return -1; }
    *out = s;
    return 0;
}

static int skip_value(reader *r, uint32_t type) {
    static const uint8_t sizes[] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
    if (type == 8) { char *s = NULL; int e = string(r, &s); free(s); return e; }
    if (type == 9) {
        uint32_t elem; uint64_t n;
        if (u32(r, &elem) || u64(r, &n) || n > 10000000) return -1;
        for (uint64_t i = 0; i < n; ++i) if (skip_value(r, elem)) return -1;
        return 0;
    }
    if (type >= sizeof(sizes) || sizes[type] == 0) return -1;
    return skip(r, sizes[type]);
}

static int read_u32_value(reader *r, uint32_t type, uint32_t *out) {
    if (type == 4) return u32(r, out);
    if (type == 5) {
        int32_t v;
        if (rd(r, &v, sizeof(v)) || v < 0) return -1;
        *out = (uint32_t)v;
        return 0;
    }
    if (type == 10) {
        uint64_t v;
        if (u64(r, &v) || v > UINT32_MAX) return -1;
        *out = (uint32_t)v;
        return 0;
    }
    return -1;
}

static int read_u32_array(reader *r, uint32_t *out, uint32_t cap,
                          uint32_t *count) {
    uint32_t elem;
    uint64_t n;
    if (u32(r, &elem) || u64(r, &n) || n > 10000000) return -1;
    uint32_t saved = n < cap ? (uint32_t)n : cap;
    for (uint64_t i = 0; i < n; ++i) {
        uint32_t v = 0;
        if (read_u32_value(r, elem, &v)) return -1;
        if (i < saved) out[i] = v;
    }
    *count = saved;
    return 0;
}

static int read_alignment(reader *r, uint32_t type, uint64_t *alignment) {
    if (type == 10) return u64(r, alignment);
    if (type == 4) { uint32_t x; if (u32(r, &x)) return -1; *alignment = x; return 0; }
    return skip_value(r, type);
}

static int tensor_cmp(const void *a, const void *b) {
    const ds4f_tensor *x = a, *y = b;
    return x->offset < y->offset ? -1 : x->offset > y->offset;
}

static int layer_number(const char *name) {
    const char *p = strstr(name, "blk.");
    if (!p) p = strstr(name, "blk-");
    if (!p) return -1;
    p += 4;
    char *end = NULL;
    long n = strtol(p, &end, 10);
    return end != p && n >= 0 && n < 10000 ? (int)n : -1;
}

int ds4f_gguf_open(ds4f_gguf *g, const char *path) {
    memset(g, 0, sizeof(*g));
    g->fd = open(path, O_RDONLY);
    if (g->fd < 0) return -1;
    struct stat st;
    if (fstat(g->fd, &st)) goto fail;
    g->file_size = (uint64_t)st.st_size;
    reader r = {g->fd, 0};
    char magic[4];
    if (rd(&r, magic, 4) || memcmp(magic, "GGUF", 4) != 0 ||
        u32(&r, &g->version) || u64(&r, &g->tensor_count) ||
        u64(&r, &g->metadata_count) || g->tensor_count > 1000000) goto fail;
    g->alignment = 32;
    for (uint64_t i = 0; i < g->metadata_count; ++i) {
        char *key = NULL; uint32_t type;
        if (string(&r, &key) || u32(&r, &type)) { free(key); goto fail; }
        if (!strcmp(key, "general.alignment")) {
            if (read_alignment(&r, type, &g->alignment)) { free(key); goto fail; }
        } else if (!strcmp(key, "deepseek4.dspark.block_size") ||
                   !strcmp(key, "deepseek4.dspark_block_size") ||
                   !strcmp(key, "dspark.block_size")) {
            if (read_u32_value(&r, type, &g->dspark_block_size)) { free(key); goto fail; }
        } else if (!strcmp(key, "dspark.stage_count")) {
            if (read_u32_value(&r, type, &g->dspark_stage_count)) { free(key); goto fail; }
        } else if (!strcmp(key, "dspark.n_layers")) {
            if (read_u32_value(&r, type, &g->dspark_n_layers)) { free(key); goto fail; }
        } else if (!strcmp(key, "deepseek4.dspark.markov_rank") ||
                   !strcmp(key, "deepseek4.dspark_markov_rank") ||
                   !strcmp(key, "dspark.markov_rank")) {
            if (read_u32_value(&r, type, &g->dspark_markov_rank)) { free(key); goto fail; }
        } else if (!strcmp(key, "deepseek4.dspark.noise_token_id") ||
                   !strcmp(key, "deepseek4.dspark_noise_token_id") ||
                   !strcmp(key, "dspark.noise_token_id")) {
            if (read_u32_value(&r, type, &g->dspark_noise_token_id)) { free(key); goto fail; }
        } else if (!strcmp(key, "deepseek4.dspark.target_layer_ids") ||
                   !strcmp(key, "deepseek4.dspark_target_layer_ids") ||
                   !strcmp(key, "dspark.target_layer_ids")) {
            if (type != 9 || read_u32_array(&r, g->dspark_target_layers, 8,
                                             &g->dspark_target_layer_count)) {
                free(key); goto fail;
            }
        } else if (skip_value(&r, type)) { free(key); goto fail; }
        free(key);
    }
    g->tensors = calloc((size_t)g->tensor_count, sizeof(*g->tensors));
    if (!g->tensors) goto fail;
    for (uint64_t i = 0; i < g->tensor_count; ++i) {
        ds4f_tensor *t = &g->tensors[i];
        if (string(&r, &t->name) || u32(&r, &t->n_dims) || t->n_dims > 4)
            goto fail;
        for (uint32_t d = 0; d < t->n_dims; ++d) if (u64(&r, &t->dims[d])) goto fail;
        if (u32(&r, &t->type) || u64(&r, &t->offset)) goto fail;
        t->layer = layer_number(t->name);
    }
    g->data_start = (r.pos + g->alignment - 1) / g->alignment * g->alignment;
    for (uint64_t i = 0; i < g->tensor_count; ++i) {
        g->tensors[i].file_offset = g->data_start + g->tensors[i].offset;
        if (g->tensors[i].file_offset > g->file_size) goto fail;
    }
    ds4f_tensor *ordered = malloc((size_t)g->tensor_count * sizeof(*ordered));
    if (!ordered) goto fail;
    memcpy(ordered, g->tensors, (size_t)g->tensor_count * sizeof(*ordered));
    qsort(ordered, (size_t)g->tensor_count, sizeof(*ordered), tensor_cmp);
    for (uint64_t i = 0; i < g->tensor_count; ++i) {
        uint64_t n = i + 1 < g->tensor_count
            ? ordered[i + 1].offset - ordered[i].offset
            : g->file_size - ordered[i].file_offset;
        for (uint64_t j = 0; j < g->tensor_count; ++j)
            if (g->tensors[j].offset == ordered[i].offset &&
                !strcmp(g->tensors[j].name, ordered[i].name)) { g->tensors[j].nbytes = n; break; }
    }
    free(ordered);
    return 0;
fail:
    ds4f_gguf_close(g);
    return -1;
}

void ds4f_gguf_close(ds4f_gguf *g) {
    if (!g) return;
    if (g->tensors) for (uint64_t i = 0; i < g->tensor_count; ++i) free(g->tensors[i].name);
    free(g->tensors);
    if (g->fd > 0) close(g->fd);
    memset(g, 0, sizeof(*g));
    g->fd = -1;
}

const ds4f_tensor *ds4f_gguf_find(const ds4f_gguf *g, const char *name) {
    if (!g || !name) return NULL;
    for (uint64_t i = 0; i < g->tensor_count; ++i)
        if (!strcmp(g->tensors[i].name, name)) return &g->tensors[i];
    return NULL;
}

int ds4f_gguf_read(const ds4f_gguf *g, const ds4f_tensor *t,
                   uint64_t relative_offset, void *dst, uint64_t bytes) {
    if (!g || !t || relative_offset > t->nbytes || bytes > t->nbytes - relative_offset ||
        t->file_offset + relative_offset > INT64_MAX) { errno = EINVAL; return -1; }
    ssize_t got = pread(g->fd, dst, (size_t)bytes, (off_t)(t->file_offset + relative_offset));
    return got == (ssize_t)bytes ? 0 : -1;
}

const char *ds4f_type_name(uint32_t type) {
    switch (type) {
    case 0: return "f32"; case 1: return "f16"; case 8: return "q8_0";
    case 10: return "q2_k"; case 16: return "iq2_xxs"; case 20: return "iq2_xxs"; default: return "unknown";
    }
}
