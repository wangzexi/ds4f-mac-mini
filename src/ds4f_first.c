#include "ds4f_gguf.h"
#include "ds4f_quant.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef DS4F_FFN_DUMP
#define DS4F_FFN_DUMP(name, x, n, layer) ((void)0)
#endif

#ifndef DS4F_HEAD_DUMP
#define DS4F_HEAD_DUMP(name, x, n) ((void)0)
#endif

/* Fixed target: DeepSeek V4 Flash 0731 on the 16 GB Mini. */
enum {
    E = 4096, HC = 4, HEAD = 64, HD = 512, Q = HEAD * HD,
    QR = 1024, GROUPS = 8, GROUP_IN = Q / GROUPS, LOW = 1024,
    FF = 2048, EXPERTS = 256, USED = 6, LAYERS = 43, VOCAB = 129280,
    ROT = 64
};

static const ds4f_tensor *need(const ds4f_gguf *g, const char *name) {
    const ds4f_tensor *t = ds4f_gguf_find(g, name);
    if (!t) {
        fprintf(stderr, "missing tensor: %s\n", name);
        exit(1);
    }
    return t;
}

static void load_tensor(const ds4f_gguf *g, const ds4f_tensor *t, void **p) {
    if (ds4f_tensor_load(g, t, p)) {
        perror(t->name);
        exit(1);
    }
}

static void layer_name(char *out, size_t cap, int layer, const char *suffix) {
    int n = snprintf(out, cap, "blk.%d.%s.weight", layer, suffix);
    if (n < 0 || (size_t)n >= cap) exit(1);
}

static const ds4f_tensor *layer_tensor(const ds4f_gguf *g, int layer,
                                       const char *suffix, char *name,
                                       size_t cap) {
    layer_name(name, cap, layer, suffix);
    return need(g, name);
}

static void hc_split(float *out, const float *mix, const float *scale,
                     const float *base) {
    for (int i = 0; i < HC; ++i)
        out[i] = 1.0f / (1.0f + expf(-(mix[i] * scale[0] + base[i]))) + 1e-6f;
    for (int i = 0; i < HC; ++i)
        out[HC + i] = 2.0f / (1.0f + expf(-(mix[HC + i] * scale[1] + base[HC + i])));

    float c[HC * HC];
    for (int dst = 0; dst < HC; ++dst) {
        float mx = -INFINITY;
        for (int src = 0; src < HC; ++src) {
            int k = src + dst * HC;
            c[k] = mix[2 * HC + k] * scale[2] + base[2 * HC + k];
            if (c[k] > mx) mx = c[k];
        }
        float sum = 0.0f;
        for (int src = 0; src < HC; ++src) {
            int k = src + dst * HC;
            c[k] = expf(c[k] - mx);
            sum += c[k];
        }
    for (int src = 0; src < HC; ++src)
        c[src + dst * HC] = c[src + dst * HC] / sum + 1e-6f;
    }
    /* The reference Sinkhorn sequence begins with a column pass, followed by
     * nineteen row/column passes (20 iterations in total). */
    for (int src = 0; src < HC; ++src) {
        float sum = 0.0f;
        for (int dst = 0; dst < HC; ++dst) sum += c[src + dst * HC];
        for (int dst = 0; dst < HC; ++dst) c[src + dst * HC] /= sum + 1e-6f;
    }
    for (int it = 1; it < 20; ++it) {
        for (int dst = 0; dst < HC; ++dst) {
            float sum = 0.0f;
            for (int src = 0; src < HC; ++src) sum += c[src + dst * HC];
            for (int src = 0; src < HC; ++src) c[src + dst * HC] /= sum + 1e-6f;
        }
        for (int src = 0; src < HC; ++src) {
            float sum = 0.0f;
            for (int dst = 0; dst < HC; ++dst) sum += c[src + dst * HC];
            for (int dst = 0; dst < HC; ++dst) c[src + dst * HC] /= sum + 1e-6f;
        }
    }
    memcpy(out + 2 * HC, c, sizeof(c));
}

static void weighted(float *out, const float *x, const float *w) {
    for (size_t d = 0; d < E; ++d) {
        out[d] = 0.0f;
        for (int h = 0; h < HC; ++h) out[d] += x[(size_t)h * E + d] * w[h];
    }
}

static void hc_post(float *out, const float *block, const float *res,
                    const float *post, const float *comb) {
    for (int dst = 0; dst < HC; ++dst) {
        for (size_t d = 0; d < E; ++d) {
            float v = block[d] * post[dst];
            for (int src = 0; src < HC; ++src)
                v += comb[dst + src * HC] * res[(size_t)src * E + d];
            out[(size_t)dst * E + d] = v;
        }
    }
}

static void hc_pre(const ds4f_gguf *g, int layer, const char *kind,
                   const float *inp, float *cur, float *res,
                   float *post, float *comb) {
    char n[96];
    const ds4f_tensor *fn = layer_tensor(g, layer,
                                         kind[0] == 'a' ? "hc_attn_fn" : "hc_ffn_fn",
                                         n, sizeof(n));
    const ds4f_tensor *st = layer_tensor(g, layer,
                                         kind[0] == 'a' ? "hc_attn_scale" : "hc_ffn_scale",
                                         n, sizeof(n));
    const ds4f_tensor *bt = layer_tensor(g, layer,
                                         kind[0] == 'a' ? "hc_attn_base" : "hc_ffn_base",
                                         n, sizeof(n));
    memcpy(res, inp, (size_t)E * HC * sizeof(float));
    float *flat = malloc((size_t)E * HC * sizeof(float));
    float *mix = malloc(24 * sizeof(float));
    float *scale = NULL, *base = NULL;
    if (!flat || !mix) exit(1);
    ds4f_rms_norm(flat, res, NULL, (size_t)E * HC, 1e-6f);
    if (ds4f_matvec(g, fn, flat, mix)) exit(1);
    load_tensor(g, st, (void **)&scale);
    load_tensor(g, bt, (void **)&base);
    float split[24];
    hc_split(split, mix, scale, base);
    weighted(cur, res, split);
    memcpy(post, split + HC, HC * sizeof(float));
    memcpy(comb, split + 2 * HC, HC * HC * sizeof(float));
    free(base);
    free(scale);
    free(mix);
    free(flat);
}

/* The first token is at position zero, so RoPE is the identity.  V4 still
 * rounds the non-RoPE KV tail through its E4M3 cache representation and then
 * through F16 before the dot products. */
static float e4m3_value(int i) {
    static const float exp_scale[16] = {
        0.0f, .015625f, .03125f, .0625f, .125f, .25f, .5f, 1.0f,
        2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f, 256.0f
    };
    int exp = (i >> 3) & 15, mant = i & 7;
    return exp == 0 ? (float)mant * .001953125f
                    : (1.0f + (float)mant * .125f) * exp_scale[exp];
}

static float e4m3_round(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (e4m3_value(mid) <= ax) lo = mid; else hi = mid - 1;
    }
    if (lo < 126) {
        float a = fabsf(ax - e4m3_value(lo));
        float b = fabsf(ax - e4m3_value(lo + 1));
        if (b < a || (b == a && (lo & 1) != 0)) ++lo;
    }
    return sign * e4m3_value(lo);
}

static __attribute__((unused)) uint16_t f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int exp = (int)((bits >> 23) & 255u) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t hm = mant >> shift;
        uint32_t rb = (mant >> (shift - 1)) & 1u;
        uint32_t sticky = mant & ((1u << (shift - 1)) - 1u);
        if (rb && (sticky || (hm & 1u))) ++hm;
        return (uint16_t)(sign | hm);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    uint32_t round = mant & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (half & 1u))) ++half;
    return (uint16_t)half;
}

static void fp8_kv_round(float *x) {
    for (int off = 0; off < HD - ROT; off += 64) {
        float amax = 0.0f;
        for (int i = 0; i < 64; ++i) amax = fmaxf(amax, fabsf(x[off + i]));
        if (amax < 1e-4f) amax = 1e-4f;
        float scale = ldexpf(1.0f, (int)ceilf(log2f(amax / 448.0f)));
        for (int i = 0; i < 64; ++i) x[off + i] = e4m3_round(x[off + i] / scale) * scale;
    }
}

static int attention_layer(const ds4f_gguf *g, int layer, const float *inp,
                           float *out) {
    char n[96];
    float *res = malloc((size_t)E * HC * sizeof(float));
    float *cur = malloc((size_t)E * sizeof(float));
    float *flat = malloc((size_t)E * HC * sizeof(float));
    float *norm = malloc((size_t)E * sizeof(float));
    float *mix = malloc(24 * sizeof(float));
    float *post = malloc(HC * sizeof(float));
    float *comb = malloc(HC * HC * sizeof(float));
    float *split = malloc(24 * sizeof(float));
    float *qr = malloc((size_t)QR * sizeof(float));
    float *qrn = malloc((size_t)QR * sizeof(float));
    float *q = malloc((size_t)Q * sizeof(float));
    float *kv = malloc((size_t)HD * sizeof(float));
    float *kvraw = malloc((size_t)HD * sizeof(float));
    float *heads = malloc((size_t)Q * sizeof(float));
    float *low = calloc((size_t)GROUPS * LOW, sizeof(float));
    float *aout = malloc((size_t)E * sizeof(float));
    if (!res || !cur || !flat || !norm || !mix || !post || !comb || !split ||
        !qr || !qrn || !q || !kv || !kvraw || !heads || !low || !aout) exit(1);

    hc_pre(g, layer, "attn", inp, cur, res, post, comb);
    ds4f_rms_norm(flat, res, NULL, (size_t)E * HC, 1e-6f);
    /* hc_pre already made cur; only the layer attention norm is weighted. */
    const ds4f_tensor *t = layer_tensor(g, layer, "attn_norm", n, sizeof(n));
    float *nw = NULL;
    load_tensor(g, t, (void **)&nw);
    ds4f_rms_norm(norm, cur, nw, E, 1e-6f);
    free(nw);

    if (ds4f_matvec(g, layer_tensor(g, layer, "attn_q_a", n, sizeof(n)), norm, qr)) return -1;
    float *qaw = NULL;
    load_tensor(g, layer_tensor(g, layer, "attn_q_a_norm", n, sizeof(n)), (void **)&qaw);
    ds4f_rms_norm(qrn, qr, qaw, QR, 1e-6f);
    free(qaw);
    if (ds4f_matvec(g, layer_tensor(g, layer, "attn_q_b", n, sizeof(n)), qrn, q)) return -1;
    for (int h = 0; h < HEAD; ++h) ds4f_rms_norm(q + (size_t)h * HD, q + (size_t)h * HD, NULL, HD, 1e-6f);

    if (ds4f_matvec(g, layer_tensor(g, layer, "attn_kv", n, sizeof(n)), norm, kvraw)) return -1;
    float *kvw = NULL;
    load_tensor(g, layer_tensor(g, layer, "attn_kv_a_norm", n, sizeof(n)), (void **)&kvw);
    ds4f_rms_norm(kv, kvraw, kvw, HD, 1e-6f);
    free(kvw);
    fp8_kv_round(kv);

    float *sinks = NULL;
    load_tensor(g, layer_tensor(g, layer, "attn_sinks", n, sizeof(n)), (void **)&sinks);
    float inv = 1.0f / sqrtf((float)HD);
    for (int h = 0; h < HEAD; ++h) {
        float score = 0.0f;
        for (int i = 0; i < HD; ++i) score += q[(size_t)h * HD + i] * kv[i];
        score *= inv;
        float mx = fmaxf(score, sinks[h]);
        float a = expf(score - mx), b = expf(sinks[h] - mx);
        float wt = a / (a + b);
        for (int i = 0; i < HD; ++i) heads[(size_t)h * HD + i] = kv[i] * wt;
    }
    free(sinks);

    if (ds4f_matvec_group(g, layer_tensor(g, layer, "attn_output_a", n, sizeof(n)),
                          heads, GROUPS, GROUP_IN, LOW, low)) return -1;
    if (ds4f_matvec(g, layer_tensor(g, layer, "attn_output_b", n, sizeof(n)), low, aout)) return -1;
    hc_post(out, aout, res, post, comb);

    free(aout); free(low); free(heads); free(kvraw); free(kv); free(q);
    free(qrn); free(qr); free(split); free(comb); free(post); free(mix);
    free(norm); free(flat); free(cur); free(res);
    return 0;
}

static float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

static void top6(const float *score, const float *bias, int *sel) {
    for (int i = 0; i < USED; ++i) sel[i] = -1;
    for (int i = 0; i < EXPERTS; ++i) {
        float v = score[i] + (bias ? bias[i] : 0.0f);
        for (int j = 0; j < USED; ++j) {
            float old = sel[j] < 0 ? -INFINITY : score[sel[j]] + (bias ? bias[sel[j]] : 0.0f);
            if (v > old) {
                for (int k = USED - 1; k > j; --k) sel[k] = sel[k - 1];
                sel[j] = i;
                break;
            }
        }
    }
}

static void print_stats(const char *name, const float *x, size_t n);

static void print_compare(const char *name, const float *gpu, const float *cpu, size_t n) {
    double diff2 = 0.0, gpu2 = 0.0, cpu2 = 0.0, dot = 0.0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float diff = gpu[i] - cpu[i];
        float ad = fabsf(diff);
        if (ad > max_abs) max_abs = ad;
        diff2 += (double)diff * diff;
        gpu2 += (double)gpu[i] * gpu[i];
        cpu2 += (double)cpu[i] * cpu[i];
        dot += (double)gpu[i] * cpu[i];
    }
    fprintf(stderr, "%s: diff_rms=%.8g max_abs=%.8g cosine=%.8g\n", name,
            sqrt(diff2 / (double)n), max_abs,
            dot / sqrt(fmax(gpu2, 1.0e-30) * fmax(cpu2, 1.0e-30)));
}

static int ffn_layer(const ds4f_gguf *g, int layer, int token,
                     const float *inp, float *out) {
    char n[96];
    float *res = malloc((size_t)E * HC * sizeof(float));
    float *cur = malloc((size_t)E * sizeof(float));
    float *flat = malloc((size_t)E * HC * sizeof(float));
    float *norm = malloc((size_t)E * sizeof(float));
    float *post = malloc(HC * sizeof(float));
    float *comb = malloc(HC * HC * sizeof(float));
    float *logits = malloc((size_t)EXPERTS * sizeof(float));
    float *probs = malloc((size_t)EXPERTS * sizeof(float));
    float *moe = calloc(E, sizeof(float));
    float *shared = malloc((size_t)E * sizeof(float));
    float *gate = malloc((size_t)FF * sizeof(float));
    float *up = malloc((size_t)FF * sizeof(float));
    float *mid = malloc((size_t)FF * sizeof(float));
    float *down = malloc((size_t)E * sizeof(float));
    float *gate_batch = malloc((size_t)USED * FF * sizeof(float));
    float *up_batch = malloc((size_t)USED * FF * sizeof(float));
    float *mid_batch = malloc((size_t)USED * FF * sizeof(float));
    float *down_batch = malloc((size_t)USED * E * sizeof(float));
    if (!res || !cur || !flat || !norm || !post || !comb || !logits || !probs ||
        !moe || !shared || !gate || !up || !mid || !down || !gate_batch ||
        !up_batch || !mid_batch || !down_batch) exit(1);

    hc_pre(g, layer, "ffn", inp, cur, res, post, comb);
    DS4F_FFN_DUMP("hc_ffn_pre", cur, E, layer);
    float *ffnw = NULL;
    load_tensor(g, layer_tensor(g, layer, "ffn_norm", n, sizeof(n)), (void **)&ffnw);
    ds4f_rms_norm(norm, cur, ffnw, E, 1e-6f);
    DS4F_FFN_DUMP("ffn_norm", norm, E, layer);
    free(ffnw);
    const ds4f_tensor *router = layer_tensor(g, layer, "ffn_gate_inp", n, sizeof(n));
    /* The Metal graph consumes the raw F32 normalization output for routers. */
    if (ds4f_matvec(g, router, norm, logits)) return -1;
    DS4F_FFN_DUMP("ffn_moe_logits", logits, EXPERTS, layer);
    for (int i = 0; i < EXPERTS; ++i) probs[i] = sqrtf(softplus(logits[i]));

    int sel[USED];
    float *router_bias = NULL;
    if (layer < 3) {
        int32_t hash[USED];
        const ds4f_tensor *ht = layer_tensor(g, layer, "ffn_gate_tid2eid", n, sizeof(n));
        if (ds4f_gguf_read(g, ht, (uint64_t)token * USED * sizeof(int32_t), hash,
                           sizeof(hash))) return -1;
        for (int i = 0; i < USED; ++i) sel[i] = hash[i];
    } else {
        char bn[96];
        int bn_len = snprintf(bn, sizeof(bn), "blk.%d.exp_probs_b.bias", layer);
        if (bn_len < 0 || (size_t)bn_len >= sizeof(bn)) return -1;
        const ds4f_tensor *bt = ds4f_gguf_find(g, bn);
        if (bt) load_tensor(g, bt, (void **)&router_bias);
        top6(probs, router_bias, sel);
    }
    float ew[USED], sum = 0.0f;
    for (int i = 0; i < USED; ++i) { ew[i] = probs[sel[i]]; sum += ew[i]; }
    for (int i = 0; i < USED; ++i) ew[i] = ew[i] / fmaxf(sum, 6.103515625e-5f) * 1.5f;
    if (getenv("DS4F_TRACE_EXPERTS")) {
        fprintf(stderr, "ds4f experts layer %d:", layer);
        for (int i = 0; i < USED; ++i) fprintf(stderr, " %d:%.7g", sel[i], ew[i]);
        fputc('\n', stderr);
    }

    if (getenv("DS4F_VALIDATE_METAL_ROUTER") && (layer == 0 || layer == 3)) {
        uint32_t fixed_ids[USED], gpu_ids[USED];
        float gpu_weights[USED];
        int mismatch = 0;
        float max_weight_diff = 0.0f;
        for (int i = 0; i < USED; ++i) fixed_ids[i] = (uint32_t)sel[i];
        if (ds4f_metal_router_top6(logits, router_bias, fixed_ids, layer < 3,
                                   gpu_ids, gpu_weights)) {
            fprintf(stderr, "ds4f Metal router diagnostic failed\n");
        } else {
            for (int i = 0; i < USED; ++i) {
                if (gpu_ids[i] != fixed_ids[i]) mismatch = 1;
                float diff = fabsf(gpu_weights[i] - ew[i]);
                if (diff > max_weight_diff) max_weight_diff = diff;
            }
            fprintf(stderr,
                    "ds4f Metal/CPU router layer %d: ids=%s max_weight_diff=%.8g\n",
                    layer, mismatch ? "MISMATCH" : "OK", max_weight_diff);
        }
    }
    free(router_bias);
    uint32_t expert_ids[USED];
    for (int s = 0; s < USED; ++s) expert_ids[s] = (uint32_t)sel[s];
    const ds4f_tensor *gt = layer_tensor(g, layer, "ffn_gate_exps", n, sizeof(n));
    const ds4f_tensor *ut = layer_tensor(g, layer, "ffn_up_exps", n, sizeof(n));
    const ds4f_tensor *dt = layer_tensor(g, layer, "ffn_down_exps", n, sizeof(n));
    if (ds4f_matvec_expert_q8k_pair_prefetch(g, gt, ut, expert_ids, USED, norm, 0, E,
                                             gate_batch, FF, up_batch, FF, dt)) return -1;
    if (getenv("DS4F_VALIDATE_METAL_EXPERTS") && layer == 0) {
        float *cpu_gate = malloc((size_t)FF * sizeof(*cpu_gate));
        float *cpu_up = malloc((size_t)FF * sizeof(*cpu_up));
        float *single_gate = malloc((size_t)FF * sizeof(*single_gate));
        if (!cpu_gate || !cpu_up || !single_gate ||
            ds4f_matvec_expert_q8k_cpu(g, gt, expert_ids[0], norm, E, cpu_gate) ||
            ds4f_matvec_expert_q8k_cpu(g, ut, expert_ids[0], norm, E, cpu_up) ||
            ds4f_matvec_expert_q8k(g, gt, expert_ids[0], norm, E, single_gate)) {
            free(single_gate); free(cpu_up); free(cpu_gate); return -1;
        }
        print_compare("ds4f Metal/CPU gate", gate_batch, cpu_gate, FF);
        print_compare("ds4f Metal/CPU up", up_batch, cpu_up, FF);
        print_compare("ds4f Metal-single/CPU gate", single_gate, cpu_gate, FF);
        float probe = 0.0f;
        if (ds4f_metal_iq2_probe(g, gt, expert_ids[0], norm, E, &probe) == 0) {
            fprintf(stderr, "ds4f Metal IQ2 probe row0: gpu=%.8g cpu=%.8g probe=%.8g\n",
                    gate_batch[0], cpu_gate[0], probe);
        }
        free(single_gate); free(cpu_up); free(cpu_gate);
    }
    if (getenv("DS4F_VALIDATE_METAL_SWIGLU") && layer == 0) {
        const size_t moe_width = (size_t)USED * FF;
        float *gpu_mid = malloc(moe_width * sizeof(*gpu_mid));
        float *cpu_mid = malloc(moe_width * sizeof(*cpu_mid));
        if (!gpu_mid || !cpu_mid ||
            ds4f_metal_swiglu_weight(gate_batch, up_batch, ew, USED, FF, gpu_mid)) {
            fprintf(stderr, "ds4f Metal SwiGLU diagnostic failed\n");
        } else {
            ds4f_swiglu(cpu_mid, gate_batch, up_batch, moe_width, 10.0f);
            for (int s = 0; s < USED; ++s)
                for (int i = 0; i < FF; ++i)
                    cpu_mid[(size_t)s * FF + i] *= ew[s];
            print_compare("ds4f Metal/CPU weighted SwiGLU", gpu_mid, cpu_mid, moe_width);
        }
        free(cpu_mid);
        free(gpu_mid);
    }
    for (int s = 0; s < USED; ++s) {
        float *expert_gate = gate_batch + (size_t)s * FF;
        float *expert_up = up_batch + (size_t)s * FF;
        for (int i = 0; i < FF; ++i) {
            if (expert_gate[i] > 10.0f) expert_gate[i] = 10.0f;
            if (expert_up[i] > 10.0f) expert_up[i] = 10.0f;
            if (expert_up[i] < -10.0f) expert_up[i] = -10.0f;
        }
    }
    DS4F_FFN_DUMP("ffn_moe_gate_clamped", gate_batch, (size_t)USED * FF, layer);
    DS4F_FFN_DUMP("ffn_moe_up_clamped", up_batch, (size_t)USED * FF, layer);
    for (int s = 0; s < USED; ++s) {
        if (getenv("DS4F_TRACE_EXPERT_DETAIL") && layer == 0) {
            char name[64];
            snprintf(name, sizeof(name), "ds4f expert %d gate", sel[s]);
            print_stats(name, gate_batch + (size_t)s * FF, FF);
            snprintf(name, sizeof(name), "ds4f expert %d up", sel[s]);
            print_stats(name, up_batch + (size_t)s * FF, FF);
        }
        ds4f_swiglu(mid_batch + (size_t)s * FF,
                    gate_batch + (size_t)s * FF,
                    up_batch + (size_t)s * FF, FF, 10.0f);
        /* Reference path applies router weight before re-quantizing for down. */
        for (int i = 0; i < FF; ++i) mid_batch[(size_t)s * FF + i] *= ew[s];
        if (getenv("DS4F_TRACE_EXPERT_DETAIL") && layer == 0) {
            char name[64];
            snprintf(name, sizeof(name), "ds4f expert %d mid", sel[s]);
            print_stats(name, mid_batch + (size_t)s * FF, FF);
        }
    }
    DS4F_FFN_DUMP("ffn_moe_weighted_swiglu", mid_batch, (size_t)USED * FF, layer);
    if (ds4f_matvec_expert_q8k_batch(g, dt, expert_ids, USED, mid_batch, FF, FF,
                                     down_batch, E)) return -1;
    for (int s = 0; s < USED; ++s)
        for (int i = 0; i < E; ++i) moe[i] += down_batch[(size_t)s * E + i];
    DS4F_FFN_DUMP("ffn_moe_out", moe, E, layer);
    if (getenv("DS4F_TRACE_EXPERT_DETAIL") && layer == 0) {
        for (int s = 0; s < USED; ++s) {
            char name[64];
            snprintf(name, sizeof(name), "ds4f expert %d down", sel[s]);
            print_stats(name, down_batch + (size_t)s * E, E);
        }
    }

    if (ds4f_matvec_q8_pair(g,
                             layer_tensor(g, layer, "ffn_gate_shexp", n, sizeof(n)),
                             layer_tensor(g, layer, "ffn_up_shexp", n, sizeof(n)),
                             norm, gate, up)) return -1;
    ds4f_swiglu(mid, gate, up, FF, 10.0f);
    if (ds4f_matvec(g, layer_tensor(g, layer, "ffn_down_shexp", n, sizeof(n)), mid, shared)) return -1;
    if (getenv("DS4F_TRACE_MOE")) {
        char name[48];
        snprintf(name, sizeof(name), "ds4f moe layer %d", layer);
        print_stats(name, moe, E);
        snprintf(name, sizeof(name), "ds4f shared layer %d", layer);
        print_stats(name, shared, E);
    }
    for (int i = 0; i < E; ++i) shared[i] += moe[i];
    if (getenv("DS4F_TRACE_MOE")) {
        char name[48];
        snprintf(name, sizeof(name), "ds4f ffn_out layer %d", layer);
        print_stats(name, shared, E);
    }
    hc_post(out, shared, res, post, comb);

    free(down_batch); free(mid_batch); free(up_batch); free(gate_batch);
    free(down); free(mid); free(up); free(gate); free(shared); free(moe);
    free(probs); free(logits); free(comb); free(post); free(norm); free(flat);
    free(cur); free(res);
    return 0;
}

static int output_head(const ds4f_gguf *g, const float *inp, float *logits) {
    char n[96];
    float *flat = malloc((size_t)E * HC * sizeof(float));
    float *pre = malloc(HC * sizeof(float));
    float *out = malloc((size_t)E * sizeof(float));
    float *norm = malloc((size_t)E * sizeof(float));
    if (!flat || !pre || !out || !norm) exit(1);
    ds4f_rms_norm(flat, inp, NULL, (size_t)E * HC, 1e-6f);
    if (ds4f_matvec(g, need(g, "output_hc_fn.weight"), flat, pre)) return -1;
    DS4F_HEAD_DUMP("result_hc_pre", pre, HC);
    float *scale = NULL, *base = NULL;
    load_tensor(g, need(g, "output_hc_scale.weight"), (void **)&scale);
    load_tensor(g, need(g, "output_hc_base.weight"), (void **)&base);
    float w[HC];
    for (int i = 0; i < HC; ++i) w[i] = 1.0f / (1.0f + expf(-(pre[i] * scale[0] + base[i]))) + 1e-6f;
    DS4F_HEAD_DUMP("result_hc_weights", w, HC);
    weighted(out, inp, w);
    DS4F_HEAD_DUMP("result_hc", out, E);
    free(base); free(scale); free(pre); free(flat);
    float *nw = NULL;
    load_tensor(g, need(g, "output_norm.weight"), (void **)&nw);
    ds4f_rms_norm(norm, out, nw, E, 1e-6f);
    DS4F_HEAD_DUMP("result_norm", norm, E);
    free(nw);
    int rc = ds4f_matvec(g, need(g, "output.weight"), norm, logits);
    DS4F_HEAD_DUMP("result_output", logits, VOCAB);
    free(norm); free(out);
    (void)n;
    return rc;
}

static void print_top(const float *logits) {
    int ids[10];
    for (int k = 0; k < 10; ++k) {
        ids[k] = 0;
        for (int i = 1; i < VOCAB; ++i) {
            int already = 0;
            for (int j = 0; j < k; ++j) if (ids[j] == i) already = 1;
            if (!already && logits[i] > logits[ids[k]]) ids[k] = i;
        }
        printf("top[%d]: token=%d logit=%.7g\n", k, ids[k], logits[ids[k]]);
    }
}

static void print_stats(const char *name, const float *x, size_t n) {
    float lo = x[0], hi = x[0];
    double ss = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (x[i] < lo) lo = x[i];
        if (x[i] > hi) hi = x[i];
        ss += (double)x[i] * x[i];
    }
    printf("%s: min=%.8g max=%.8g rms=%.8g\n", name, lo, hi,
           sqrtf((float)(ss / (double)n)));
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf [token_id]\n", argv[0]);
        return 2;
    }
    int token = argc > 2 ? atoi(argv[2]) : 67;
    if (token < 0 || token >= VOCAB) return 2;
    ds4f_gguf g;
    if (ds4f_gguf_open(&g, argv[1])) { perror("GGUF"); return 1; }

    uint16_t *emb16 = malloc((size_t)E * sizeof(uint16_t));
    float *emb = malloc((size_t)E * sizeof(float));
    float *cur = malloc((size_t)E * HC * sizeof(float));
    float *next = malloc((size_t)E * HC * sizeof(float));
    float *logits = malloc((size_t)VOCAB * sizeof(float));
    if (!emb16 || !emb || !cur || !next || !logits) exit(1);
    const ds4f_tensor *te = need(&g, "token_embd.weight");
    if (ds4f_gguf_read(&g, te, (uint64_t)token * E * sizeof(uint16_t), emb16,
                       (uint64_t)E * sizeof(uint16_t))) return 1;
    for (int i = 0; i < E; ++i) emb[i] = ds4f_f16_to_f32(emb16[i]);
    for (int h = 0; h < HC; ++h) memcpy(cur + (size_t)h * E, emb, (size_t)E * sizeof(float));
    printf("first-token input=%d; layers=%d\n", token, LAYERS);

    for (int layer = 0; layer < LAYERS; ++layer) {
        if (attention_layer(&g, layer, cur, next)) return 1;
        float *tmp = cur; cur = next; next = tmp;
        if (ffn_layer(&g, layer, token, cur, next)) return 1;
        tmp = cur; cur = next; next = tmp;
        if (getenv("DS4F_TRACE_LAYERS")) {
            char name[48];
            snprintf(name, sizeof(name), "layer %d final_hc", layer);
            print_stats(name, cur, (size_t)E * HC);
        }
        printf("layer %d/%d complete\n", layer + 1, LAYERS);
        const char *stop_after = getenv("DS4F_STOP_AFTER_LAYER");
        if (stop_after && atoi(stop_after) == layer) return 0;
    }
    if (getenv("DS4F_TRACE_STATE")) print_stats("first-token final_hc", cur, (size_t)E * HC);
    if (output_head(&g, cur, logits)) return 1;
    print_top(logits);
    printf("first-token: OK\n");

    free(logits); free(next); free(cur); free(emb); free(emb16);
    ds4f_gguf_close(&g);
    return 0;
}
