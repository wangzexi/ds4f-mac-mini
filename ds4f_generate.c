/* Reuse the already validated fixed-model math while replacing the one-shot
 * driver with a real token loop and per-layer KV state. */
#pragma clang diagnostic ignored "-Wunused-function"
#define main ds4f_first_unused_main
#include "ds4f_first.c"
#undef main

#include "ds4f_tokenizer.h"

#include <stdbool.h>
#include <sys/time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum { GEN_RAW = 128, GEN_MAX_INDEXER = 512 };

static const uint32_t gen_ratio[LAYERS] = {
    0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 0
};

typedef struct {
    uint32_t ratio;
    uint32_t n_raw;
    float *raw;
    uint32_t n_comp;
    uint32_t cap_comp;
    float *comp;
    float *state_kv;
    float *state_score;
    float *index_comp;
    float *index_state_kv;
    float *index_state_score;
    uint32_t n_index_comp;
} gen_layer_cache;

typedef struct {
    gen_layer_cache layer[LAYERS];
    uint32_t max_ctx;
} gen_cache;

static void gen_die(const char *s) { fprintf(stderr, "ds4f-generate: %s\n", s); exit(1); }

static double gen_now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

static void gen_cache_init(gen_cache *c, uint32_t max_ctx) {
    memset(c, 0, sizeof(*c));
    if (max_ctx < GEN_RAW) max_ctx = GEN_RAW;
    c->max_ctx = max_ctx;
    for (int l = 0; l < LAYERS; ++l) {
        gen_layer_cache *x = &c->layer[l];
        x->ratio = gen_ratio[l];
        x->raw = calloc((size_t)GEN_RAW * HD, sizeof(float));
        if (!x->raw) gen_die("raw KV allocation failed");
        if (!x->ratio) continue;
        x->cap_comp = max_ctx / x->ratio + 2;
        x->comp = calloc((size_t)x->cap_comp * HD, sizeof(float));
        uint32_t coff = x->ratio == 4 ? 2 : 1;
        uint32_t width = coff * HD;
        uint32_t state_rows = coff * x->ratio;
        x->state_kv = calloc((size_t)width * state_rows, sizeof(float));
        x->state_score = malloc((size_t)width * state_rows * sizeof(float));
        if (x->ratio == 4) {
            x->index_comp = calloc((size_t)x->cap_comp * 128, sizeof(float));
            x->index_state_kv = calloc((size_t)2 * 128 * 8, sizeof(float));
            x->index_state_score = malloc((size_t)2 * 128 * 8 * sizeof(float));
        }
        if (!x->comp || !x->state_kv || !x->state_score ||
            (x->ratio == 4 && (!x->index_comp || !x->index_state_kv || !x->index_state_score)))
            gen_die("compressed KV allocation failed");
        for (size_t i = 0; i < (size_t)width * state_rows; ++i) x->state_score[i] = -1e30f;
        if (x->ratio == 4) for (size_t i = 0; i < (size_t)2 * 128 * 8; ++i) x->index_state_score[i] = -1e30f;
    }
}

static void gen_cache_free(gen_cache *c) {
    for (int l = 0; l < LAYERS; ++l) {
        gen_layer_cache *x = &c->layer[l];
        free(x->raw); free(x->comp); free(x->state_kv); free(x->state_score);
        free(x->index_comp); free(x->index_state_kv); free(x->index_state_score);
    }
    memset(c, 0, sizeof(*c));
}

static void gen_push_raw(gen_layer_cache *c, const float *kv) {
    if (c->n_raw == GEN_RAW) {
        memmove(c->raw, c->raw + HD, (size_t)(GEN_RAW - 1) * HD * sizeof(float));
        c->n_raw = GEN_RAW - 1;
    }
    float *dst = c->raw + (size_t)c->n_raw * HD;
    for (int i = 0; i < HD; ++i) dst[i] = ds4f_f16_to_f32(f32_to_f16(kv[i]));
    c->n_raw++;
}

static void gen_push_comp(gen_layer_cache *c, const float *kv, int index) {
    uint32_t *n = index ? &c->n_index_comp : &c->n_comp;
    float *rows = index ? c->index_comp : c->comp;
    uint32_t dim = index ? 128 : HD;
    if (*n >= c->cap_comp) gen_die("compressed KV cache capacity exceeded");
    for (uint32_t i = 0; i < dim; ++i) rows[(size_t)*n * dim + i] = ds4f_f16_to_f32(f32_to_f16(kv[i]));
    (*n)++;
}

static float gen_rope_base(int layer) { return gen_ratio[layer] ? 160000.0f : 10000.0f; }
static void gen_rope(float *x, uint32_t heads_n, uint32_t dim, uint32_t nrot,
                     uint32_t pos, int layer, int inverse) {
    if (!pos) return;
    int compressed = gen_ratio[layer] != 0;
    float freq_scale = compressed ? 1.0f / 16.0f : 1.0f;
    float theta_scale = powf(gen_rope_base(layer), -2.0f / (float)nrot);
    float attn = compressed ? 1.0f / (1.0f + 0.1f * logf(16.0f)) : 1.0f;
    float lo = 0.0f, hi = (float)(nrot - 1);
    if (compressed) {
        lo = floorf((float)nrot * logf(65536.0f / (32.0f * 2.0f * (float)M_PI)) /
                    (2.0f * logf(160000.0f)));
        hi = ceilf((float)nrot * logf(65536.0f / (1.0f * 2.0f * (float)M_PI)) /
                   (2.0f * logf(160000.0f)));
        if (lo < 0) lo = 0; if (hi > (float)(nrot - 1)) hi = (float)(nrot - 1);
    }
    uint32_t nope = dim - nrot;
    for (uint32_t h = 0; h < heads_n; ++h) {
        float *tail = x + (size_t)h * dim + nope;
        float theta_extrap = (float)pos;
        for (uint32_t i = 0; i < nrot; i += 2) {
            float ramp = compressed ? 1.0f - fminf(1.0f, fmaxf(0.0f, ((float)(i / 2) - lo) / fmaxf(.001f, hi - lo))) : 0.0f;
            float theta = freq_scale * theta_extrap * (1.0f - ramp) + theta_extrap * ramp;
            float mscale = attn * (compressed ? 1.0f + 0.1f * logf(16.0f) : 1.0f);
            float c = cosf(theta) * mscale, s = sinf(theta) * mscale * (inverse ? -1.0f : 1.0f);
            float a = tail[i], b = tail[i + 1];
            tail[i] = a * c - b * s; tail[i + 1] = a * s + b * c;
            theta_extrap *= theta_scale;
        }
    }
}

static void gen_hadamard128(float *x) {
    for (uint32_t stride = 1; stride < 128; stride <<= 1)
        for (uint32_t base = 0; base < 128; base += 2 * stride)
            for (uint32_t i = 0; i < stride; ++i) {
                float a = x[base + i], b = x[base + stride + i];
                x[base + i] = a + b; x[base + stride + i] = a - b;
            }
    for (int i = 0; i < 128; ++i) x[i] *= 0.08838834764831845f;
}
static void gen_fp4(float *x) {
    static const float v[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    for (int off = 0; off < 128; off += 32) {
        float amax = 0;
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(x[off + i]));
        if (amax < 7.052966104933725e-38f) amax = 7.052966104933725e-38f;
        float scale = ldexpf(1.0f, (int)ceilf(log2f(amax / 6.0f)));
        for (int i = 0; i < 32; ++i) {
            float z = fminf(6.0f, fmaxf(-6.0f, x[off + i] / scale));
            int best = 0; float bd = fabsf(fabsf(z) - v[0]);
            for (int k = 1; k < 8; ++k) { float d = fabsf(fabsf(z) - v[k]); if (d < bd) { bd = d; best = k; } }
            x[off + i] = (z < 0 ? -1.0f : 1.0f) * v[best] * scale;
        }
    }
}

static void gen_pool(float *out, const float *kv, const float *score,
                     uint32_t ratio, uint32_t head_dim) {
    uint32_t coff = ratio == 4 ? 2 : 1, width = coff * head_dim;
    for (uint32_t j = 0; j < head_dim; ++j) {
        float mx = -1e30f;
        for (uint32_t r = 0; r < ratio; ++r) {
            mx = fmaxf(mx, score[(size_t)r * width + j]);
            if (ratio == 4) mx = fmaxf(mx, score[(size_t)(ratio + r) * width + head_dim + j]);
        }
        float den = 0, sum = 0;
        for (uint32_t r = 0; r < ratio; ++r) {
            float a = expf(score[(size_t)r * width + j] - mx);
            den += a; sum += a * kv[(size_t)r * width + j];
            if (ratio == 4) {
                float b = expf(score[(size_t)(ratio + r) * width + head_dim + j] - mx);
                den += b; sum += b * kv[(size_t)(ratio + r) * width + head_dim + j];
            }
        }
        out[j] = den > 0 ? sum / den : 0;
    }
}

static int gen_compress(const ds4f_gguf *g, int layer, const float *x,
                        uint32_t pos, float *out, int index, gen_layer_cache *c) {
    uint32_t ratio = c->ratio, dim = index ? 128 : HD;
    uint32_t coff = ratio == 4 ? 2 : 1, width = coff * dim;
    char n[96];
    const char *ks = index ? "indexer_compressor_kv" : "attn_compressor_kv";
    const char *gs = index ? "indexer_compressor_gate" : "attn_compressor_gate";
    const char *as = index ? "indexer_compressor_ape" : "attn_compressor_ape";
    const char *ns = index ? "indexer_compressor_norm" : "attn_compressor_norm";
    const ds4f_tensor *wk = layer_tensor(g, layer, ks, n, sizeof(n));
    const ds4f_tensor *wg = layer_tensor(g, layer, gs, n, sizeof(n));
    const ds4f_tensor *wa = layer_tensor(g, layer, as, n, sizeof(n));
    const ds4f_tensor *wn = layer_tensor(g, layer, ns, n, sizeof(n));
    float *kv = malloc((size_t)width * sizeof(float));
    float *sc = malloc((size_t)width * sizeof(float));
    float *ape = NULL, *norm = NULL;
    if (!kv || !sc) return -1;
    if (ds4f_matvec(g, wk, x, kv) || ds4f_matvec(g, wg, x, sc)) return -1;
    load_tensor(g, wa, (void **)&ape); load_tensor(g, wn, (void **)&norm);
    uint32_t pos_mod = pos % ratio, row = ratio == 4 ? ratio + pos_mod : pos_mod;
    float *state_kv = index ? c->index_state_kv : c->state_kv;
    float *state_sc = index ? c->index_state_score : c->state_score;
    for (uint32_t j = 0; j < width; ++j) {
        uint16_t h; memcpy(&h, (uint8_t *)ape + ((size_t)pos_mod * width + j) * 2, 2);
        sc[j] += ds4f_f16_to_f32(h);
    }
    memcpy(state_kv + (size_t)row * width, kv, (size_t)width * sizeof(float));
    memcpy(state_sc + (size_t)row * width, sc, (size_t)width * sizeof(float));
    int emit = ((pos + 1) % ratio) == 0;
    if (emit) {
        float *pooled = malloc((size_t)dim * sizeof(float));
        gen_pool(pooled, state_kv, state_sc, ratio, dim);
        float ss = 0;
        for (uint32_t i = 0; i < dim; ++i) ss += pooled[i] * pooled[i];
        float inv = 1.0f / sqrtf(ss / (float)dim + 1e-6f);
        for (uint32_t i = 0; i < dim; ++i) out[i] = pooled[i] * inv * norm[i];
        gen_rope(out, 1, dim, ROT, pos + 1 - ratio, layer, 0);
        if (index) { gen_hadamard128(out); gen_fp4(out); }
        else fp8_kv_round(out);
        free(pooled);
        if (ratio == 4) {
            for (uint32_t r = 0; r < ratio; ++r) {
                memcpy(state_kv + (size_t)r * width, state_kv + (size_t)(ratio + r) * width, (size_t)width * sizeof(float));
                memcpy(state_sc + (size_t)r * width, state_sc + (size_t)(ratio + r) * width, (size_t)width * sizeof(float));
            }
            for (uint32_t r = 0; r < ratio; ++r) {
                memcpy(state_kv + (size_t)(ratio + r) * width, state_kv + (size_t)r * width, (size_t)width * sizeof(float));
                memcpy(state_sc + (size_t)(ratio + r) * width, state_sc + (size_t)r * width, (size_t)width * sizeof(float));
            }
        }
        gen_push_comp(c, out, index);
    }
    free(norm); free(ape); free(sc); free(kv);
    return emit;
}

static bool *gen_index_allowed(const ds4f_gguf *g, int layer, const float *cur,
                               const float *qr_norm, gen_layer_cache *c, uint32_t pos) {
    if (c->n_index_comp <= GEN_MAX_INDEXER) return NULL;
    bool *allow = calloc(c->n_comp, sizeof(bool));
    char n[96]; float *q = malloc((size_t)HEAD * 128 * sizeof(float));
    float *w = malloc(HEAD * sizeof(float)); float *score = malloc(c->n_index_comp * sizeof(float));
    if (!allow || !q || !w || !score) gen_die("indexer allocation failed");
    if (ds4f_matvec(g, layer_tensor(g, layer, "indexer_attn_q_b", n, sizeof(n)), qr_norm, q) ||
        ds4f_matvec(g, layer_tensor(g, layer, "indexer_proj", n, sizeof(n)), cur, w)) gen_die("indexer matvec failed");
    gen_rope(q, HEAD, 128, ROT, pos, layer, 0); for (int h = 0; h < HEAD; ++h) gen_hadamard128(q + h * 128), gen_fp4(q + h * 128);
    for (uint32_t cidx = 0; cidx < c->n_index_comp; ++cidx) {
        score[cidx] = 0;
        for (int h = 0; h < HEAD; ++h) {
            float dot = 0; const float *qh = q + h * 128, *kv = c->index_comp + (size_t)cidx * 128;
            for (int j = 0; j < 128; ++j) dot += qh[j] * kv[j];
            if (dot > 0) score[cidx] += dot * w[h] / sqrtf((float)(HEAD * 128));
        }
    }
    for (int k = 0; k < GEN_MAX_INDEXER && (uint32_t)k < c->n_comp; ++k) {
        uint32_t best = 0; float bs = -1e30f;
        for (uint32_t i = 0; i < c->n_index_comp; ++i) if (!allow[i] && score[i] > bs) { best = i; bs = score[i]; }
        allow[best] = true;
    }
    free(score); free(w); free(q); return allow;
}

static void gen_attention(const ds4f_gguf *g, int layer, uint32_t pos,
                          const float *inp, float *out, gen_layer_cache *c) {
    char n[96];
    float *res = malloc((size_t)E * HC * 4), *cur = malloc(E * 4), *norm = malloc(E * 4);
    float *qr = malloc(QR * 4), *qrn = malloc(QR * 4), *q = malloc(Q * 4), *kv = malloc(HD * 4);
    float *heads = malloc(Q * 4), *low = calloc(GROUPS * LOW, 4), *aout = malloc(E * 4);
    float *post = malloc(HC * 4), *comb = malloc(HC * HC * 4);
    if (!res || !cur || !norm || !qr || !qrn || !q || !kv || !heads || !low || !aout || !post || !comb) gen_die("attention allocation failed");
    hc_pre(g, layer, "attn", inp, cur, res, post, comb);
    float *nw = NULL; load_tensor(g, layer_tensor(g, layer, "attn_norm", n, sizeof(n)), (void **)&nw);
    ds4f_rms_norm(norm, cur, nw, E, 1e-6f); free(nw);
    ds4f_matvec(g, layer_tensor(g, layer, "attn_q_a", n, sizeof(n)), norm, qr);
    float *qaw = NULL; load_tensor(g, layer_tensor(g, layer, "attn_q_a_norm", n, sizeof(n)), (void **)&qaw);
    ds4f_rms_norm(qrn, qr, qaw, QR, 1e-6f); free(qaw);
    ds4f_matvec(g, layer_tensor(g, layer, "attn_q_b", n, sizeof(n)), qrn, q);
    for (int h = 0; h < HEAD; ++h) ds4f_rms_norm(q + h * HD, q + h * HD, NULL, HD, 1e-6f);
    ds4f_matvec(g, layer_tensor(g, layer, "attn_kv", n, sizeof(n)), norm, kv);
    float *kvw = NULL; load_tensor(g, layer_tensor(g, layer, "attn_kv_a_norm", n, sizeof(n)), (void **)&kvw);
    ds4f_rms_norm(kv, kv, kvw, HD, 1e-6f); free(kvw);
    gen_rope(q, HEAD, HD, ROT, pos, layer, 0); gen_rope(kv, 1, HD, ROT, pos, layer, 0); fp8_kv_round(kv); gen_push_raw(c, kv);
    if (c->ratio) {
        float *comp_tmp = malloc((size_t)(c->ratio == 4 ? 2 * HD : HD) * sizeof(float));
        if (!comp_tmp) gen_die("attention compressor allocation failed");
        gen_compress(g, layer, norm, pos, comp_tmp, 0, c);
        if (c->ratio == 4) {
            float index_tmp[128];
            gen_compress(g, layer, norm, pos, index_tmp, 1, c);
        }
        free(comp_tmp);
    }
    bool *allow = c->ratio == 4 ? gen_index_allowed(g, layer, norm, qrn, c, pos) : NULL;
    float *sinks = NULL; load_tensor(g, layer_tensor(g, layer, "attn_sinks", n, sizeof(n)), (void **)&sinks);
    float inv = 1.0f / sqrtf((float)HD); uint32_t total = c->n_raw + c->n_comp;
    for (int h = 0; h < HEAD; ++h) {
        const float *qh = q + (size_t)h * HD; float maxs = sinks[h];
        for (uint32_t r = 0; r < total; ++r) {
            if (r >= c->n_raw && allow && !allow[r - c->n_raw]) continue;
            const float *x = r < c->n_raw ? c->raw + (size_t)r * HD : c->comp + (size_t)(r - c->n_raw) * HD;
            float s = 0; for (int j = 0; j < HD; ++j) s += qh[j] * x[j];
            if (s * inv > maxs) maxs = s * inv;
        }
        float den = expf(sinks[h] - maxs); float *oh = heads + (size_t)h * HD;
        memset(oh, 0, HD * sizeof(float));
        for (uint32_t r = 0; r < total; ++r) {
            if (r >= c->n_raw && allow && !allow[r - c->n_raw]) continue;
            const float *x = r < c->n_raw ? c->raw + (size_t)r * HD : c->comp + (size_t)(r - c->n_raw) * HD;
            float s = 0; for (int j = 0; j < HD; ++j) s += qh[j] * x[j];
            float wt = expf(s * inv - maxs); den += wt;
            for (int j = 0; j < HD; ++j) oh[j] += wt * x[j];
        }
        for (int j = 0; j < HD; ++j) oh[j] /= den;
    }
    free(sinks); free(allow);
    gen_rope(heads, HEAD, HD, ROT, pos, layer, 1);
    ds4f_matvec_group(g, layer_tensor(g, layer, "attn_output_a", n, sizeof(n)), heads, GROUPS, GROUP_IN, LOW, low);
    ds4f_matvec(g, layer_tensor(g, layer, "attn_output_b", n, sizeof(n)), low, aout);
    hc_post(out, aout, res, post, comb);
    free(comb); free(post); free(aout); free(low); free(heads); free(kv); free(q); free(qrn); free(qr); free(norm); free(cur); free(res);
}

static int gen_forward(const ds4f_gguf *g, gen_cache *cache, int token,
                       uint32_t pos, float *logits) {
    uint16_t *h = malloc(E * 2); float *plain = malloc(E * 4);
    float *cur = malloc(E * HC * 4), *next = malloc(E * HC * 4), *attn = malloc(E * HC * 4);
    if (!h || !plain || !cur || !next || !attn) gen_die("forward allocation failed");
    const ds4f_tensor *te = need(g, "token_embd.weight");
    if (ds4f_gguf_read(g, te, (uint64_t)token * E * 2, h, E * 2)) gen_die("embedding read failed");
    for (int i = 0; i < E; ++i) plain[i] = ds4f_f16_to_f32(h[i]);
    for (int hc = 0; hc < HC; ++hc) memcpy(cur + (size_t)hc * E, plain, E * 4);
    const int profile = getenv("DS4F_PROFILE") != NULL;
    double attention_ms = 0.0, ffn_ms = 0.0;
    for (int l = 0; l < LAYERS; ++l) {
        double t0 = profile ? gen_now_ms() : 0.0;
        gen_attention(g, l, pos, cur, attn, &cache->layer[l]);
        if (profile) attention_ms += gen_now_ms() - t0;
        t0 = profile ? gen_now_ms() : 0.0;
        if (ffn_layer(g, l, token, attn, next)) gen_die("FFN failed");
        if (profile) ffn_ms += gen_now_ms() - t0;
        float *tmp = cur; cur = next; next = tmp;
    }
    double t_head = profile ? gen_now_ms() : 0.0;
    int rc = logits ? output_head(g, cur, logits) : 0;
    if (profile) fprintf(stderr,
                         "profile pos=%u attention_ms=%.3f ffn_ms=%.3f head_ms=%.3f\n",
                         pos, attention_ms, ffn_ms, gen_now_ms() - t_head);
    free(attn); free(next); free(cur); free(plain); free(h); return rc;
}

static int argmax(const float *x) { int best = 0; for (int i = 1; i < VOCAB; ++i) if (x[i] > x[best]) best = i; return best; }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL.gguf PROMPT [new_tokens] [max_ctx]\n", argv[0]); return 2; }
    int n_new = argc > 3 ? atoi(argv[3]) : 8; uint32_t max_ctx = argc > 4 ? (uint32_t)atoi(argv[4]) : 4096;
    ds4f_gguf g; if (ds4f_gguf_open(&g, argv[1])) { perror("GGUF"); return 1; }
    ds4f_tokenizer tok; if (ds4f_tokenizer_open(&g, &tok)) gen_die("tokenizer load failed");
    ds4f_tokens prompt = {0}; if (ds4f_tokenize_chat(&tok, argv[2], 1, &prompt)) gen_die("chat tokenize failed");
    printf("prompt tokens=%zu:", prompt.len); for (size_t i = 0; i < prompt.len; ++i) printf(" %d", prompt.v[i]); putchar('\n');
    gen_cache cache; gen_cache_init(&cache, max_ctx);
    float *logits = malloc((size_t)VOCAB * sizeof(float)); if (!logits) gen_die("logits allocation failed");
    uint32_t pos = 0; int next = 0;
    double prefill_t0 = gen_now_ms();
    for (size_t i = 0; i < prompt.len; ++i) { if (gen_forward(&g, &cache, prompt.v[i], pos++, i + 1 == prompt.len ? logits : NULL)) gen_die("prefill failed"); }
    fprintf(stderr, "prefill_ms=%.3f tokens=%zu\n", gen_now_ms() - prefill_t0, prompt.len);
    if (getenv("DS4F_TRACE_TOP")) print_top(logits);
    for (int step = 0; step < n_new; ++step) {
        next = argmax(logits); char *text = NULL; size_t text_len = 0;
        ds4f_token_text(&tok, next, &text, &text_len);
        printf("gen[%d] token=%d text=", step, next); for (size_t i = 0; i < text_len; ++i) putchar(text[i]); putchar('\n'); free(text);
        if (ds4f_token_is_stop(&tok, next)) break;
        double decode_t0 = gen_now_ms();
        if (gen_forward(&g, &cache, next, pos++, logits)) gen_die("decode failed");
        fprintf(stderr, "decode[%d]_ms=%.3f\n", step, gen_now_ms() - decode_t0);
    }
    free(logits); gen_cache_free(&cache); ds4f_tokens_free(&prompt); ds4f_tokenizer_close(&tok); ds4f_gguf_close(&g); return 0;
}
