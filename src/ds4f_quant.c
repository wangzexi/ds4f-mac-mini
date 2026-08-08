#include "ds4f_quant.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef DS4F_USE_METAL
extern int ds4f_metal_matvec_rows(const ds4f_gguf *g, const ds4f_tensor *t,
                                  const float *x, float *y,
                                  uint32_t row0, uint32_t rows);
extern int ds4f_metal_matvec_q8_pair(const ds4f_gguf *g,
                                     const ds4f_tensor *a, const ds4f_tensor *b,
                                     const float *x, float *ya, float *yb);
extern int ds4f_metal_matvec_pair_shared_input(const ds4f_gguf *g,
                                               const ds4f_tensor *a,
                                               const ds4f_tensor *b,
                                               const float *x,
                                               float *ya, float *yb);
extern int ds4f_metal_matvec_expert_q8k(const ds4f_gguf *g,
                                        const ds4f_tensor *t,
                                        uint32_t expert, const float *x,
                                        size_t n, float *y);
extern int ds4f_metal_matvec_expert_q8k_batch(const ds4f_gguf *g,
                                              const ds4f_tensor *t,
                                              const uint32_t *experts,
                                              size_t count, const float *x,
                                              size_t x_stride, size_t n,
                                              float *y, size_t y_stride);
extern int ds4f_metal_matvec_expert_q8k_pair(const ds4f_gguf *g,
                                             const ds4f_tensor *a,
                                             const ds4f_tensor *b,
                                             const uint32_t *experts,
                                             size_t count, const float *x,
                                             size_t x_stride, size_t n,
                                             float *ya, size_t ya_stride,
                                             float *yb, size_t yb_stride);
extern int ds4f_metal_matvec_expert_q8k_pair_prefetch(const ds4f_gguf *g,
                                                       const ds4f_tensor *a,
                                                       const ds4f_tensor *b,
                                                       const uint32_t *experts,
                                                       size_t count, const float *x,
                                                       size_t x_stride, size_t n,
                                                       float *ya, size_t ya_stride,
                                                       float *yb, size_t yb_stride,
                                                       const ds4f_tensor *prefetch);
extern int ds4f_metal_matvec_group(const ds4f_gguf *g, const ds4f_tensor *t,
                                   const float *x, uint32_t groups,
                                   uint32_t group_in, uint32_t group_out,
                                   float *y);
extern int ds4f_metal_router_top6(const float *logits, const float *bias,
                                  const uint32_t *fixed_ids, int use_fixed,
                                  uint32_t *out_ids, float *out_weights);
extern int ds4f_metal_swiglu_weight(const float *gate, const float *up,
                                    const float *weights, size_t count,
                                    size_t width, float *out);
#else
int ds4f_metal_iq2_probe(const ds4f_gguf *g, const ds4f_tensor *t,
                          uint32_t expert, const float *x, size_t n, float *out) {
    (void)g; (void)t; (void)expert; (void)x; (void)n; (void)out;
    return -1;
}
int ds4f_metal_router_top6(const float *logits, const float *bias,
                           const uint32_t *fixed_ids, int use_fixed,
                           uint32_t *out_ids, float *out_weights) {
    (void)logits; (void)bias; (void)fixed_ids; (void)use_fixed;
    (void)out_ids; (void)out_weights;
    return -1;
}
int ds4f_metal_swiglu_weight(const float *gate, const float *up,
                             const float *weights, size_t count, size_t width,
                             float *out) {
    (void)gate; (void)up; (void)weights; (void)count; (void)width; (void)out;
    return -1;
}
#endif

typedef struct { uint16_t d; int8_t qs[32]; } q8_0_block;
typedef struct { uint8_t scales[16]; uint8_t qs[64]; uint16_t d, dmin; } q2_k_block;
typedef struct { uint16_t d; uint16_t qs[32]; } iq2_xxs_block;
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } q8_k_block;

static uint32_t iq2_sign_bits(uint32_t code) {
    code &= 127u;
    return code | ((uint32_t)__builtin_popcount(code & 127u) & 1u) << 7u;
}

typedef struct {
    int fd;
    uint64_t file_offset;
    uint64_t nbytes;
    uint64_t age;
    uint8_t *data;
} expert_cache_entry;

static expert_cache_entry *g_expert_cache;
static size_t g_expert_cache_len;
static size_t g_expert_cache_cap;
static uint64_t g_expert_cache_bytes;
static uint64_t g_expert_cache_limit;
static uint64_t g_expert_cache_age;
static int g_expert_cache_initialized;

static void expert_cache_init(void) {
    if (g_expert_cache_initialized) return;
    g_expert_cache_initialized = 1;
    const char *s = getenv("DS4F_EXPERT_CACHE_GIB");
    double gib = s && s[0] ? strtod(s, NULL) : 4.0;
    if (!(gib > 0.0)) gib = 0.0;
    if (gib > 8.0) gib = 8.0;
    g_expert_cache_limit = (uint64_t)(gib * 1024.0 * 1024.0 * 1024.0);
    if (g_expert_cache_limit)
        fprintf(stderr, "ds4f: expert slice cache limit %.2f GiB\n", gib);
}

static void expert_cache_drop(size_t i) {
    if (i >= g_expert_cache_len) return;
    free(g_expert_cache[i].data);
    g_expert_cache_bytes -= g_expert_cache[i].nbytes;
    g_expert_cache[i] = g_expert_cache[g_expert_cache_len - 1];
    g_expert_cache_len--;
}

static void *expert_slice_load(const ds4f_gguf *g, const ds4f_tensor *t,
                               uint32_t expert, uint64_t expert_bytes,
                               int *cached_out) {
    expert_cache_init();
    const uint64_t file_offset = t->file_offset + (uint64_t)expert * expert_bytes;
    for (size_t i = 0; i < g_expert_cache_len; ++i) {
        expert_cache_entry *e = &g_expert_cache[i];
        if (e->fd == g->fd && e->file_offset == file_offset &&
            e->nbytes == expert_bytes) {
            e->age = ++g_expert_cache_age;
            *cached_out = 1;
            return e->data;
        }
    }

    void *raw = malloc((size_t)expert_bytes);
    if (!raw || ds4f_gguf_read(g, t, (uint64_t)expert * expert_bytes,
                               raw, expert_bytes)) {
        free(raw);
        return NULL;
    }
    *cached_out = 0;
    if (!g_expert_cache_limit || expert_bytes > g_expert_cache_limit) return raw;

    while (g_expert_cache_bytes + expert_bytes > g_expert_cache_limit &&
           g_expert_cache_len) {
        size_t oldest = 0;
        for (size_t i = 1; i < g_expert_cache_len; ++i)
            if (g_expert_cache[i].age < g_expert_cache[oldest].age) oldest = i;
        expert_cache_drop(oldest);
    }
    if (g_expert_cache_len == g_expert_cache_cap) {
        size_t next = g_expert_cache_cap ? g_expert_cache_cap * 2 : 64;
        expert_cache_entry *p = realloc(g_expert_cache, next * sizeof(*p));
        if (!p) return raw;
        g_expert_cache = p;
        g_expert_cache_cap = next;
    }
    expert_cache_entry *e = &g_expert_cache[g_expert_cache_len++];
    e->fd = g->fd;
    e->file_offset = file_offset;
    e->nbytes = expert_bytes;
    e->age = ++g_expert_cache_age;
    e->data = raw;
    g_expert_cache_bytes += expert_bytes;
    *cached_out = 1;
    return raw;
}

static const uint64_t iq2_grid[256] = {
    0x0808080808080808,0x080808080808082b,0x0808080808081919,0x0808080808082b08,
    0x0808080808082b2b,0x0808080808190819,0x0808080808191908,0x08080808082b0808,
    0x08080808082b082b,0x08080808082b2b08,0x08080808082b2b2b,0x0808080819080819,
    0x0808080819081908,0x0808080819190808,0x0808080819192b08,0x08080808192b0819,
    0x08080808192b1908,0x080808082b080808,0x080808082b08082b,0x080808082b082b2b,
    0x080808082b2b082b,0x0808081908080819,0x0808081908081908,0x0808081908190808,
    0x0808081908191919,0x0808081919080808,0x080808192b081908,0x080808192b192b08,
    0x0808082b08080808,0x0808082b0808082b,0x0808082b082b082b,0x0808082b2b08082b,
    0x0808190808080819,0x0808190808081908,0x0808190808190808,0x08081908082b0819,
    0x08081908082b1908,0x0808190819080808,0x080819081908082b,0x0808190819082b08,
    0x08081908192b0808,0x080819082b080819,0x080819082b081908,0x080819082b190808,
    0x080819082b2b1908,0x0808191908080808,0x080819190808082b,0x0808191908082b08,
    0x08081919082b0808,0x080819191908192b,0x08081919192b2b19,0x080819192b080808,
    0x080819192b190819,0x0808192b08082b19,0x0808192b08190808,0x0808192b19080808,
    0x0808192b2b081908,0x0808192b2b2b1908,0x08082b0808080808,0x08082b0808081919,
    0x08082b0808082b08,0x08082b0808191908,0x08082b08082b2b08,0x08082b0819080819,
    0x08082b0819081908,0x08082b0819190808,0x08082b081919082b,0x08082b082b082b08,
    0x08082b1908081908,0x08082b1919080808,0x08082b2b0808082b,0x08082b2b08191908,
    0x0819080808080819,0x0819080808081908,0x0819080808190808,0x08190808082b0819,
    0x0819080819080808,0x08190808192b0808,0x081908082b081908,0x081908082b190808,
    0x081908082b191919,0x0819081908080808,0x0819081908082b08,0x08190819082b0808,
    0x0819081919190808,0x0819081919192b2b,0x081908192b080808,0x0819082b082b1908,
    0x0819082b19081919,0x0819190808080808,0x0819190808082b08,0x08191908082b0808,
    0x08191908082b1919,0x0819190819082b19,0x081919082b080808,0x0819191908192b08,
    0x08191919192b082b,0x0819192b08080808,0x0819192b0819192b,0x08192b0808080819,
    0x08192b0808081908,0x08192b0808190808,0x08192b0819080808,0x08192b082b080819,
    0x08192b1908080808,0x08192b1908081919,0x08192b192b2b0808,0x08192b2b19190819,
    0x082b080808080808,0x082b08080808082b,0x082b080808082b2b,0x082b080819081908,
    0x082b0808192b0819,0x082b08082b080808,0x082b08082b08082b,0x082b0819082b2b19,
    0x082b081919082b08,0x082b082b08080808,0x082b082b0808082b,0x082b190808080819,
    0x082b190808081908,0x082b190808190808,0x082b190819080808,0x082b19081919192b,
    0x082b191908080808,0x082b191919080819,0x082b1919192b1908,0x082b192b2b190808,
    0x082b2b0808082b08,0x082b2b08082b0808,0x082b2b082b191908,0x082b2b2b19081908,
    0x1908080808080819,0x1908080808081908,0x1908080808190808,0x1908080808192b08,
    0x19080808082b0819,0x19080808082b1908,0x1908080819080808,0x1908080819082b08,
    0x190808081919192b,0x19080808192b0808,0x190808082b080819,0x190808082b081908,
    0x190808082b190808,0x1908081908080808,0x19080819082b0808,0x19080819192b0819,
    0x190808192b080808,0x190808192b081919,0x1908082b08080819,0x1908082b08190808,
    0x1908082b19082b08,0x1908082b1919192b,0x1908082b192b2b08,0x1908190808080808,
    0x1908190808082b08,0x19081908082b0808,0x190819082b080808,0x190819082b192b19,
    0x190819190819082b,0x19081919082b1908,0x1908192b08080808,0x19082b0808080819,
    0x19082b0808081908,0x19082b0808190808,0x19082b0819080808,0x19082b0819081919,
    0x19082b1908080808,0x19082b1919192b08,0x19082b19192b0819,0x19082b192b08082b,
    0x19082b2b19081919,0x19082b2b2b190808,0x1919080808080808,0x1919080808082b08,
    0x1919080808190819,0x1919080808192b19,0x19190808082b0808,0x191908082b080808,
    0x191908082b082b08,0x1919081908081908,0x191908191908082b,0x191908192b2b1908,
    0x1919082b2b190819,0x191919082b190808,0x191919082b19082b,0x1919191908082b2b,
    0x1919192b08080819,0x1919192b19191908,0x19192b0808080808,0x19192b0808190819,
    0x19192b0808192b19,0x19192b08192b1908,0x19192b1919080808,0x19192b2b08082b08,
    0x192b080808081908,0x192b080808190808,0x192b080819080808,0x192b0808192b2b08,
    0x192b081908080808,0x192b081919191919,0x192b082b08192b08,0x192b082b192b0808,
    0x192b190808080808,0x192b190808081919,0x192b191908190808,0x192b19190819082b,
    0x192b19192b081908,0x192b2b081908082b,0x2b08080808080808,0x2b0808080808082b,
    0x2b08080808082b2b,0x2b08080819080819,0x2b0808082b08082b,0x2b08081908081908,
    0x2b08081908192b08,0x2b08081919080808,0x2b08082b08190819,0x2b08190808080819,
    0x2b08190808081908,0x2b08190808190808,0x2b08190808191919,0x2b08190819080808,
    0x2b081908192b0808,0x2b08191908080808,0x2b0819191908192b,0x2b0819192b191908,
    0x2b08192b08082b19,0x2b08192b19080808,0x2b08192b192b0808,0x2b082b080808082b,
    0x2b082b1908081908,0x2b082b2b08190819,0x2b19080808081908,0x2b19080808190808,
    0x2b190808082b1908,0x2b19080819080808,0x2b1908082b2b0819,0x2b1908190819192b,
    0x2b1908192b080808,0x2b19082b19081919,0x2b19190808080808,0x2b191908082b082b,
    0x2b19190819081908,0x2b19191919190819,0x2b192b082b080819,0x2b192b19082b0808,
    0x2b2b08080808082b,0x2b2b080819190808,0x2b2b08082b081919,0x2b2b081908082b19,
    0x2b2b082b08080808,0x2b2b190808192b08,0x2b2b2b0819190808,0x2b2b2b1908081908,
};

const uint64_t *ds4f_iq2_grid_data(void) { return iq2_grid; }

float ds4f_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
        if (!mant) { float z = 0.0f; memcpy(&z, &sign, sizeof(z)); return z; }
        float f = (float)mant * 0x1.0p-24f;
        return (h & 0x8000u) ? -f : f;
    } else if (exp == 31) out = sign | 0x7f800000u | (mant << 13);
    else out = sign | ((exp + 112u) << 23) | (mant << 13);
    float f; memcpy(&f, &out, sizeof(f)); return f;
}

int ds4f_tensor_load(const ds4f_gguf *g, const ds4f_tensor *t, void **data_out) {
    if (!g || !t || !data_out || t->nbytes > SIZE_MAX) { errno = EINVAL; return -1; }
    void *p = malloc((size_t)t->nbytes);
    if (!p || ds4f_gguf_read(g, t, 0, p, t->nbytes)) { free(p); return -1; }
    *data_out = p; return 0;
}

static float dot_f32(const float *a, const float *b, size_t n) {
    float s = 0.0f; for (size_t i = 0; i < n; ++i) s += a[i] * b[i]; return s;
}
static float dot_f16(const uint16_t *w, const float *x, size_t n) {
    float s = 0.0f; for (size_t i = 0; i < n; ++i) s += ds4f_f16_to_f32(w[i]) * x[i]; return s;
}
static float dot_q8f32(const uint8_t *row,const float *x,size_t n){float s=0;for(size_t b=0;b<(n+31)/32;++b){uint16_t d;memcpy(&d,row+b*34,2);const int8_t *q=(const int8_t *)(row+b*34+2);size_t k=b*32,m=n-k<32?n-k:32;float z=0;for(size_t i=0;i<m;++i)z+=(float)q[i]*x[k+i];s+=ds4f_f16_to_f32(d)*z;}return s;}
static void quant_q8_0_activation(const float *x, int8_t *xq, float *scale, size_t n) {
    const size_t blocks = (n + 31u) / 32u;
    for (size_t b = 0; b < blocks; ++b) {
        const size_t i0 = b * 32u;
        const size_t bn = n - i0 < 32u ? n - i0 : 32u;
        float amax = 0.0f;
        for (size_t i = 0; i < bn; ++i) {
            float ax = fabsf(x[i0 + i]);
            if (ax > amax) amax = ax;
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        scale[b] = d;
        for (size_t i = 0; i < bn; ++i) {
            int v = (int)lrintf(x[i0 + i] * id);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            xq[i0 + i] = (int8_t)v;
        }
        memset(xq + i0 + bn, 0, 32u - bn);
    }
}

int ds4f_matvec_q8_0_cpu(const ds4f_gguf *g, const ds4f_tensor *t,
                          const float *x, float *y) {
    if (!g || !t || !x || !y || t->type != 8 || t->n_dims != 2) {
        errno = EINVAL;
        return -1;
    }
    const size_t in = (size_t)t->dims[0];
    const size_t out = (size_t)t->dims[1];
    const size_t blocks = (in + 31u) / 32u;
    const size_t rowbytes = blocks * 34u;
    int8_t *xq = malloc(blocks * 32u * sizeof(*xq));
    float *scale = malloc(blocks * sizeof(*scale));
    void *raw = NULL;
    if (!xq || !scale || ds4f_tensor_load(g, t, &raw)) {
        free(raw); free(scale); free(xq);
        return -1;
    }
    quant_q8_0_activation(x, xq, scale, in);
    for (size_t r = 0; r < out; ++r) {
        const uint8_t *row = (const uint8_t *)raw + r * rowbytes;
        float sum = 0.0f;
        for (size_t b = 0; b < blocks; ++b) {
            uint16_t bits;
            memcpy(&bits, row + b * 34u, sizeof(bits));
            const int8_t *wq = (const int8_t *)(row + b * 34u + 2u);
            int z = 0;
            for (size_t i = 0; i < 32u; ++i) z += (int)wq[i] * (int)xq[b * 32u + i];
            sum += ds4f_f16_to_f32(bits) * scale[b] * (float)z;
        }
        y[r] = sum;
    }
    free(raw); free(scale); free(xq);
    return 0;
}
static __attribute__((unused)) float dot_iq2(const iq2_xxs_block *w, const float *x, size_t n) {
    float s=0.0f; size_t nb=n/256;
    for (size_t b=0;b<nb;++b) {
        float d=ds4f_f16_to_f32(w[b].d); const uint16_t *q=w[b].qs;
        for (size_t g=0;g<8;++g) {
            uint32_t aux[2]; memcpy(aux,q,8); q+=4;
            uint32_t grid_index; const uint8_t *bytes=(const uint8_t *)&aux[0];
            for (size_t l=0;l<4;++l) {
                grid_index=bytes[l]; const uint8_t *gb=(const uint8_t *)&iq2_grid[grid_index];
                uint32_t sign=iq2_sign_bits((aux[1]>>(7*l))&127u);
                for (size_t j=0;j<8;++j) {
                    int v=(int)gb[j]; if (sign & (1u<<j)) v=-v;
                    s += (0.125f*d*(float)(2u*(aux[1]>>28)+1u))*(float)v*x[b*256+g*32+l*8+j];
                }
            }
        }
    }
    return s;
}
static __attribute__((unused)) float dot_q2(const q2_k_block *w, const float *x, size_t n) {
    float s=0.0f;
    for (size_t b=0;b<n/256;++b) for (size_t k=0;k<256;++k) {
        size_t group=k/16, l=k%16, base=32*(group/8)+16*(group&1), shift=((group/2)&3)*2;
        uint32_t q=(w[b].qs[base+l]>>shift)&3, sc=w[b].scales[group];
        float v=ds4f_f16_to_f32(w[b].d)*(float)(sc&15)*q-ds4f_f16_to_f32(w[b].dmin)*(float)(sc>>4);
        s+=v*x[b*256+k];
    }
    return s;
}

static void quant_q8k(const float *x, q8_k_block *y, size_t n) {
    for(size_t b=0;b<n/256;++b){float mx=0,amax=0;for(size_t i=0;i<256;++i){float ax=fabsf(x[b*256+i]);if(ax>amax){amax=ax;mx=x[b*256+i];}}
        if(!amax){y[b].d=0;memset(y[b].qs,0,256);memset(y[b].bsums,0,32);continue;}
        float is=-127.0f/mx; y[b].d=1.0f/is;for(size_t i=0;i<256;++i){int v=(int)lrintf(is*x[b*256+i]);if(v>127)v=127;if(v<-128)v=-128;y[b].qs[i]=(int8_t)v;}
        for(size_t j=0;j<16;++j){int s=0;for(size_t i=0;i<16;++i)s+=y[b].qs[j*16+i];y[b].bsums[j]=(int16_t)s;}
    }
}
static float dot_iq2_q8k(const iq2_xxs_block *w,const q8_k_block *y,size_t n){float s=0;for(size_t b=0;b<n/256;++b){const uint16_t *q=w[b].qs;float d=ds4f_f16_to_f32(w[b].d)*y[b].d;for(size_t ib=0;ib<8;++ib){uint32_t aux[2];memcpy(aux,q,8);q+=4;float scale=.125f*d*(float)(2u*(aux[1]>>28)+1u);const uint8_t *a=(const uint8_t *)aux;for(size_t l=0;l<4;++l){const uint8_t *gb=(const uint8_t *)&iq2_grid[a[l]];uint32_t sign=iq2_sign_bits((aux[1]>>(7*l))&127u);for(size_t j=0;j<8;++j){int v=(sign&(1u<<j))?-(int)gb[j]:(int)gb[j];s+=scale*(float)v*(float)y[b].qs[ib*32+l*8+j];}}}}return s;}
static float dot_q2_q8k(const q2_k_block *w, const q8_k_block *y, size_t n) {
    float sum = 0.0f;
    for (size_t b = 0; b < n / 256u; ++b) {
        int isum = 0;
        int summs = 0;
        for (size_t k = 0; k < 256u; ++k) {
            const size_t group = k / 16u;
            const size_t l = k % 16u;
            const size_t base = 32u * (group / 8u) + 16u * (group & 1u);
            const uint32_t shift = (uint32_t)((group / 2u) & 3u) * 2u;
            const uint32_t q = (w[b].qs[base + l] >> shift) & 3u;
            const uint8_t sc = w[b].scales[group];
            const int q8 = (int)y[b].qs[k];
            isum += (int)(sc & 15u) * (int)q * q8;
            summs += (int)(sc >> 4u) * q8;
        }
        sum += y[b].d * ds4f_f16_to_f32(w[b].d) * (float)isum -
               y[b].d * ds4f_f16_to_f32(w[b].dmin) * (float)summs;
    }
    return sum;
}

int ds4f_matvec_expert(const ds4f_gguf *g, const ds4f_tensor *t,
                       uint32_t expert, const float *x, float *y) {
    if (!g || !t || t->n_dims != 3 || t->dims[2] == 0 || expert >= t->dims[2]) { errno=EINVAL; return -1; }
    size_t in=(size_t)t->dims[0], out=(size_t)t->dims[1], expert_bytes=(size_t)(t->nbytes/t->dims[2]);
    size_t rowbytes=expert_bytes/out;
    int cached = 0;
    void *raw=expert_slice_load(g, t, expert, expert_bytes, &cached);
    if(!raw) return -1;
    for(size_t r=0;r<out;++r){const uint8_t *row=(const uint8_t *)raw+r*rowbytes;
        if(t->type==16)y[r]=dot_iq2((const iq2_xxs_block *)row,x,in);
        else if(t->type==10)y[r]=dot_q2((const q2_k_block *)row,x,in);
        else {if (!cached) free(raw);errno=ENOTSUP;return -1;}
    }
    if (!cached) free(raw);return 0;
}

int ds4f_matvec_expert_q8k_cpu(const ds4f_gguf *g, const ds4f_tensor *t,
                               uint32_t expert, const float *x, size_t n, float *y) {
    if (!g || !t || t->n_dims != 3 || t->dims[2] == 0 || expert >= t->dims[2] || n != t->dims[0] || n % 256) { errno=EINVAL; return -1; }
    size_t out=(size_t)t->dims[1], eb=(size_t)(t->nbytes/t->dims[2]), rb=eb/out;
    int cached = 0;
    void *raw=expert_slice_load(g, t, expert, eb, &cached);
    q8_k_block *qx=malloc((n/256)*sizeof(*qx));
    if(!raw||!qx){if (!cached) free(raw);free(qx);return -1;}
    quant_q8k(x,qx,n);
    for(size_t r=0;r<out;++r){const uint8_t *row=(const uint8_t *)raw+r*rb;if(t->type==16)y[r]=dot_iq2_q8k((const iq2_xxs_block *)row,qx,n);else if(t->type==10)y[r]=dot_q2_q8k((const q2_k_block *)row,qx,n);else{if (!cached) free(raw);free(qx);errno=ENOTSUP;return -1;}}
    if (!cached) free(raw);free(qx);return 0;
}

int ds4f_matvec_expert_q8k(const ds4f_gguf *g, const ds4f_tensor *t,
                           uint32_t expert, const float *x, size_t n, float *y) {
    if (!g || !t || t->n_dims != 3 || t->dims[2] == 0 || expert >= t->dims[2] || n != t->dims[0] || n % 256) { errno=EINVAL; return -1; }
#ifdef DS4F_USE_METAL
    if (ds4f_metal_matvec_expert_q8k(g, t, expert, x, n, y) == 0) return 0;
#endif
    return ds4f_matvec_expert_q8k_cpu(g, t, expert, x, n, y);
}

int ds4f_matvec_expert_q8k_batch(const ds4f_gguf *g, const ds4f_tensor *t,
                                 const uint32_t *experts, size_t count,
                                 const float *x, size_t x_stride, size_t n,
                                 float *y, size_t y_stride) {
    if (!g || !t || !experts || !count || !x || !y ||
        t->n_dims != 3 || t->dims[2] == 0 || n != t->dims[0] || n % 256 ||
        (x_stride && x_stride < n) || y_stride < t->dims[1]) {
        errno = EINVAL;
        return -1;
    }
#ifdef DS4F_USE_METAL
    if ((t->type == 16 || t->type == 10) &&
        ds4f_metal_matvec_expert_q8k_batch(g, t, experts, count, x,
                                           x_stride, n, y, y_stride) == 0)
        return 0;
#endif
    for (size_t i = 0; i < count; ++i) {
        const float *xi = x + (x_stride ? i * x_stride : 0);
        float *yi = y + i * y_stride;
        if (ds4f_matvec_expert_q8k(g, t, experts[i], xi, n, yi)) return -1;
    }
    return 0;
}

int ds4f_matvec_expert_q8k_pair(const ds4f_gguf *g,
                                const ds4f_tensor *a, const ds4f_tensor *b,
                                const uint32_t *experts, size_t count,
                                const float *x, size_t x_stride, size_t n,
                                float *ya, size_t ya_stride,
                                float *yb, size_t yb_stride) {
    if (!g || !a || !b || !experts || !count || !x || !ya || !yb ||
        a->n_dims != 3 || b->n_dims != 3 || a->type != b->type ||
        a->dims[0] != b->dims[0] || a->dims[2] != b->dims[2] ||
        n != a->dims[0] || n % 256 || (x_stride && x_stride < n) ||
        ya_stride < a->dims[1] || yb_stride < b->dims[1]) {
        errno = EINVAL;
        return -1;
    }
#ifdef DS4F_USE_METAL
    if ((a->type == 16 || a->type == 10) &&
        ds4f_metal_matvec_expert_q8k_pair(g, a, b, experts, count, x,
                                          x_stride, n, ya, ya_stride, yb,
                                          yb_stride) == 0)
        return 0;
#endif

    for (size_t i = 0; i < count; ++i) {
        const float *xi = x + (x_stride ? i * x_stride : 0);
        if (ds4f_matvec_expert_q8k(g, a, experts[i], xi, n,
                                   ya + i * ya_stride) ||
            ds4f_matvec_expert_q8k(g, b, experts[i], xi, n,
                                   yb + i * yb_stride)) return -1;
    }
    return 0;
}

int ds4f_matvec_expert_q8k_pair_prefetch(const ds4f_gguf *g,
                                         const ds4f_tensor *a, const ds4f_tensor *b,
                                         const uint32_t *experts, size_t count,
                                         const float *x, size_t x_stride, size_t n,
                                         float *ya, size_t ya_stride,
                                         float *yb, size_t yb_stride,
                                         const ds4f_tensor *prefetch) {
#ifndef DS4F_USE_METAL
    (void)prefetch;
#endif
#ifdef DS4F_USE_METAL
    if (prefetch && a && b && a->type == 16 && b->type == 16 &&
        ds4f_metal_matvec_expert_q8k_pair_prefetch(g, a, b, experts, count,
                                                    x, x_stride, n, ya, ya_stride,
                                                    yb, yb_stride, prefetch) == 0)
        return 0;
#endif
    return ds4f_matvec_expert_q8k_pair(g, a, b, experts, count, x, x_stride,
                                        n, ya, ya_stride, yb, yb_stride);
}

int ds4f_matvec(const ds4f_gguf *g, const ds4f_tensor *t, const float *x, float *y) {
    if (!g || !t || t->n_dims != 2) { errno=EINVAL; return -1; }
    size_t in=(size_t)t->dims[0], out=(size_t)t->dims[1], rowbytes=t->nbytes/out;
#ifdef DS4F_USE_METAL
    if (t->type == 8 && out <= UINT32_MAX && ds4f_metal_matvec_rows(g, t, x, y, 0, (uint32_t)out) == 0) return 0;
#endif
    void *raw=NULL; if (ds4f_tensor_load(g,t,&raw)) return -1;
    for (size_t r=0;r<out;++r) {
        const uint8_t *row=(const uint8_t *)raw+r*rowbytes;
        switch(t->type) {
        case 0: y[r]=dot_f32((const float *)row,x,in); break;
        case 1: y[r]=dot_f16((const uint16_t *)row,x,in); break;
        case 8: y[r]=dot_q8f32(row,x,in); break;
        default: free(raw); errno=ENOTSUP; return -1;
        }
    }
    free(raw); return 0;
}

int ds4f_matvec_q8_pair(const ds4f_gguf *g, const ds4f_tensor *a,
                         const ds4f_tensor *b, const float *x,
                         float *ya, float *yb) {
    if (!g || !a || !b || !x || !ya || !yb || a->n_dims != 2 || b->n_dims != 2 ||
        a->type != 8 || b->type != 8 || a->dims[0] != b->dims[0] ||
        a->dims[1] != b->dims[1]) {
        errno = EINVAL;
        return -1;
    }

#ifdef DS4F_USE_METAL
    if (ds4f_metal_matvec_q8_pair(g, a, b, x, ya, yb) == 0) return 0;
#endif
    return ds4f_matvec(g, a, x, ya) || ds4f_matvec(g, b, x, yb) ? -1 : 0;
}

int ds4f_matvec_pair_shared_input(const ds4f_gguf *g, const ds4f_tensor *a,
                                  const ds4f_tensor *b, const float *x,
                                  float *ya, float *yb) {
    if (!g || !a || !b || !x || !ya || !yb || a->n_dims != 2 ||
        b->n_dims != 2 || a->dims[0] != b->dims[0]) {
        errno = EINVAL;
        return -1;
    }
#ifdef DS4F_USE_METAL
    if (a->type == 8 && b->type == 8 &&
        ds4f_metal_matvec_pair_shared_input(g, a, b, x, ya, yb) == 0)
        return 0;
#endif
    return ds4f_matvec(g, a, x, ya) || ds4f_matvec(g, b, x, yb) ? -1 : 0;
}

int ds4f_matvec_group(const ds4f_gguf *g, const ds4f_tensor *t,
                      const float *x, uint32_t groups, uint32_t group_in,
                      uint32_t group_out, float *y) {
    if (!g || !t || t->n_dims != 2 || t->dims[0] != group_in || t->dims[1] != (uint64_t)groups*group_out) { errno=EINVAL; return -1; }
#ifdef DS4F_USE_METAL
    if (t->type == 8 &&
        ds4f_metal_matvec_group(g, t, x, groups, group_in, group_out, y) == 0)
        return 0;
#endif
    void *raw=NULL; if (ds4f_tensor_load(g,t,&raw)) return -1;
    size_t rowbytes=t->nbytes/t->dims[1];
    for (uint32_t gr=0;gr<groups;++gr) for (uint32_t r=0;r<group_out;++r) {
        size_t row=(size_t)gr*group_out+r; const uint8_t *p=(const uint8_t *)raw+row*rowbytes;
        y[row]=t->type==8?dot_q8f32(p,x+(size_t)gr*group_in,group_in):0.0f;
    }
    free(raw); return 0;
}

void ds4f_rms_norm(float *out,const float *x,const float *w,size_t n,float eps) {
    double ss=0.0; for(size_t i=0;i<n;++i) ss+=(double)x[i]*x[i]; float inv=1.0f/sqrtf((float)(ss/(double)n)+eps);
    for(size_t i=0;i<n;++i) out[i]=x[i]*inv*(w?w[i]:1.0f);
}
void ds4f_swiglu(float *out,const float *gate,const float *up,size_t n,float clamp) {
    for(size_t i=0;i<n;++i){float g=gate[i],u=up[i]; if(clamp>0){if(g>clamp)g=clamp;if(u>clamp)u=clamp;if(u<-clamp)u=-clamp;} out[i]=(g/(1.0f+expf(-g)))*u;}
}
