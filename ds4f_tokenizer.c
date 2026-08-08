#define _DARWIN_C_SOURCE
#include "ds4f_tokenizer.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct { int fd; uint64_t pos; } reader;
typedef struct { const char *ptr; size_t len; } span;
typedef struct { char *ptr; size_t len; } owned;
typedef struct { const char *ptr; size_t len; int value; uint8_t used; } map_entry;

static int read_at(reader *r, void *p, size_t n) {
    ssize_t got = pread(r->fd, p, n, (off_t)r->pos);
    if (got != (ssize_t)n) return -1;
    r->pos += n;
    return 0;
}
static int rd_u32(reader *r, uint32_t *v) { return read_at(r, v, 4); }
static int rd_u64(reader *r, uint64_t *v) { return read_at(r, v, 8); }
static int rd_string(reader *r, char **out, size_t *len_out) {
    uint64_t n;
    if (rd_u64(r, &n) || n > 64u * 1024u * 1024u || n > SIZE_MAX - 1) return -1;
    char *p = malloc((size_t)n + 1);
    if (!p || read_at(r, p, (size_t)n)) { free(p); return -1; }
    p[n] = 0;
    *out = p;
    if (len_out) *len_out = (size_t)n;
    return 0;
}
static int skip_bytes(reader *r, uint64_t n) {
    if (n > UINT64_MAX - r->pos) return -1;
    r->pos += n;
    return 0;
}
static int skip_value(reader *r, uint32_t type) {
    static const uint8_t size[] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
    if (type == 8) {
        char *s = NULL;
        int rc = rd_string(r, &s, NULL);
        free(s);
        return rc;
    }
    if (type == 9) {
        uint32_t elem; uint64_t n;
        if (rd_u32(r, &elem) || rd_u64(r, &n) || n > 10000000) return -1;
        for (uint64_t i = 0; i < n; ++i) if (skip_value(r, elem)) return -1;
        return 0;
    }
    if (type >= sizeof(size) || size[type] == 0) return -1;
    return skip_bytes(r, size[type]);
}
static uint64_t hash_bytes(const char *p, size_t n) {
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < n; ++i) { h ^= (uint8_t)p[i]; h *= UINT64_C(1099511628211); }
    return h;
}
static size_t pow2(size_t n) {
    size_t p = 1;
    while (p < n) { if (p > SIZE_MAX / 2) return 0; p <<= 1; }
    return p;
}
static map_entry *map_init(size_t n, size_t *cap) {
    *cap = pow2(n * 2 + 16);
    return *cap ? calloc(*cap, sizeof(map_entry)) : NULL;
}
static void map_put(map_entry *m, size_t cap, const char *p, size_t n, int value) {
    size_t i = (size_t)hash_bytes(p, n) & (cap - 1);
    while (m[i].used) {
        if (m[i].len == n && !memcmp(m[i].ptr, p, n)) { m[i].value = value; return; }
        i = (i + 1) & (cap - 1);
    }
    m[i] = (map_entry){p, n, value, 1};
}
static int map_get(const map_entry *m, size_t cap, const char *p, size_t n, int *value) {
    if (!m || !cap) return 0;
    size_t i = (size_t)hash_bytes(p, n) & (cap - 1);
    while (m[i].used) {
        if (m[i].len == n && !memcmp(m[i].ptr, p, n)) { *value = m[i].value; return 1; }
        i = (i + 1) & (cap - 1);
    }
    return 0;
}
static int read_string_array(reader *r, char ***out, size_t *count) {
    uint32_t elem; uint64_t n;
    if (rd_u32(r, &elem) || rd_u64(r, &n) || elem != 8 || n > 1000000) return -1;
    char **v = calloc((size_t)n, sizeof(*v));
    if (!v) return -1;
    for (uint64_t i = 0; i < n; ++i) {
        if (rd_string(r, &v[i], NULL)) {
            for (uint64_t j = 0; j < i; ++j) free(v[j]);
            free(v); return -1;
        }
    }
    *out = v; *count = (size_t)n; return 0;
}
static int read_u64_meta(reader *r, uint32_t type, int *out) {
    uint64_t v = 0;
    if (type == 4) { uint32_t x; if (rd_u32(r, &x)) return -1; v = x; }
    else if (type == 10) { if (rd_u64(r, &v)) return -1; }
    else if (type == 0) { uint8_t x; if (read_at(r, &x, 1)) return -1; v = x; }
    else if (type == 1) { uint16_t x; if (read_at(r, &x, 2)) return -1; v = x; }
    else if (type == 2) { uint32_t x; if (rd_u32(r, &x)) return -1; v = x; }
    else if (type == 3) { int32_t x; if (rd_u32(r, (uint32_t *)&x)) return -1; v = (uint32_t)x; }
    else return -1;
    *out = v > INT32_MAX ? -1 : (int)v;
    return 0;
}

static void free_string_array(char **v, size_t n) {
    if (!v) return;
    for (size_t i = 0; i < n; ++i) free(v[i]);
    free(v);
}

int ds4f_tokenizer_open(const ds4f_gguf *g, ds4f_tokenizer *t) {
    if (!g || !t) { errno = EINVAL; return -1; }
    memset(t, 0, sizeof(*t));
    t->bos_id = t->eos_id = t->user_id = t->assistant_id = -1;
    t->think_start_id = t->think_end_id = -1;
    reader r = {g->fd, 0};
    char magic[4]; uint32_t version; uint64_t tensors, metadata;
    if (read_at(&r, magic, 4) || memcmp(magic, "GGUF", 4) ||
        rd_u32(&r, &version) || rd_u64(&r, &tensors) || rd_u64(&r, &metadata)) return -1;
    (void)version; (void)tensors;
    char **raw_tokens = NULL, **raw_merges = NULL;
    size_t n_tokens = 0, n_merges = 0;
    for (uint64_t i = 0; i < metadata; ++i) {
        char *key = NULL; size_t key_len = 0; uint32_t type;
        if (rd_string(&r, &key, &key_len) || rd_u32(&r, &type)) { free(key); goto fail; }
        if (!strcmp(key, "tokenizer.ggml.tokens") && type == 9) {
            free_string_array(raw_tokens, n_tokens); raw_tokens = NULL; n_tokens = 0;
            if (read_string_array(&r, &raw_tokens, &n_tokens)) { free(key); goto fail; }
        } else if (!strcmp(key, "tokenizer.ggml.merges") && type == 9) {
            free_string_array(raw_merges, n_merges); raw_merges = NULL; n_merges = 0;
            if (read_string_array(&r, &raw_merges, &n_merges)) { free(key); goto fail; }
        } else if (!strcmp(key, "tokenizer.ggml.bos_token_id")) {
            if (read_u64_meta(&r, type, &t->bos_id)) { free(key); goto fail; }
        } else if (!strcmp(key, "tokenizer.ggml.eos_token_id")) {
            if (read_u64_meta(&r, type, &t->eos_id)) { free(key); goto fail; }
        } else if (skip_value(&r, type)) { free(key); goto fail; }
        free(key);
    }
    if (!raw_tokens || !raw_merges || n_tokens == 0) goto fail;
    t->token = calloc(n_tokens, sizeof(*t->token));
    t->merge = calloc(n_merges, sizeof(*t->merge));
    if (!t->token || (n_merges && !t->merge)) goto fail;
    t->n_vocab = n_tokens; t->n_merge = n_merges;
    for (size_t i = 0; i < n_tokens; ++i) {
        t->token[i].len = strlen(raw_tokens[i]);
        t->token[i].ptr = raw_tokens[i]; raw_tokens[i] = NULL;
    }
    for (size_t i = 0; i < n_merges; ++i) {
        t->merge[i] = raw_merges[i]; raw_merges[i] = NULL;
    }
    t->token_map = map_init(n_tokens, &t->token_map_cap);
    t->merge_map = map_init(n_merges, &t->merge_map_cap);
    if (!t->token_map || !t->merge_map) goto fail;
    for (size_t i = 0; i < n_tokens; ++i)
        map_put(t->token_map, t->token_map_cap, t->token[i].ptr, t->token[i].len, (int)i);
    for (size_t i = 0; i < n_merges; ++i)
        map_put(t->merge_map, t->merge_map_cap, t->merge[i], strlen(t->merge[i]), (int)i);
    const char *special[] = {
        "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜User｜>",
        "<｜Assistant｜>", "<think>", "</think>"
    };
    int *dst[] = {&t->bos_id, &t->eos_id, &t->user_id, &t->assistant_id,
                  &t->think_start_id, &t->think_end_id};
    for (size_t i = 0; i < sizeof(special) / sizeof(special[0]); ++i)
        if (*dst[i] < 0) map_get(t->token_map, t->token_map_cap,
                                 special[i], strlen(special[i]), dst[i]);
    free_string_array(raw_tokens, n_tokens);
    free_string_array(raw_merges, n_merges);
    return 0;
fail:
    free_string_array(raw_tokens, n_tokens);
    free_string_array(raw_merges, n_merges);
    ds4f_tokenizer_close(t);
    errno = EINVAL;
    return -1;
}

void ds4f_tokenizer_close(ds4f_tokenizer *t) {
    if (!t) return;
    if (t->token) for (size_t i = 0; i < t->n_vocab; ++i) free(t->token[i].ptr);
    if (t->merge) for (size_t i = 0; i < t->n_merge; ++i) free(t->merge[i]);
    free(t->token); free(t->merge); free(t->token_map); free(t->merge_map);
    memset(t, 0, sizeof(*t));
    t->bos_id = t->eos_id = t->user_id = t->assistant_id = -1;
    t->think_start_id = t->think_end_id = -1;
}

static void token_push(ds4f_tokens *t, int id) {
    if (t->len == t->cap) {
        size_t cap = t->cap ? t->cap * 2 : 64;
        int *p = realloc(t->v, cap * sizeof(*p));
        if (!p) exit(1);
        t->v = p; t->cap = cap;
    }
    t->v[t->len++] = id;
}
void ds4f_tokens_free(ds4f_tokens *t) { if (t) { free(t->v); memset(t, 0, sizeof(*t)); } }

static int utf8_len(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;
}
static void utf8_put(char **p, uint32_t cp) {
    if (cp <= 0x7f) *(*p)++ = (char)cp;
    else if (cp <= 0x7ff) { *(*p)++ = (char)(0xc0 | (cp >> 6)); *(*p)++ = (char)(0x80 | (cp & 63)); }
    else if (cp <= 0xffff) { *(*p)++ = (char)(0xe0 | (cp >> 12)); *(*p)++ = (char)(0x80 | ((cp >> 6) & 63)); *(*p)++ = (char)(0x80 | (cp & 63)); }
    else { *(*p)++ = (char)(0xf0 | (cp >> 18)); *(*p)++ = (char)(0x80 | ((cp >> 12) & 63)); *(*p)++ = (char)(0x80 | ((cp >> 6) & 63)); *(*p)++ = (char)(0x80 | (cp & 63)); }
}
static uint32_t byte_cp(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174) return b;
    uint32_t n = 0;
    for (uint32_t x = 0; x < 256; ++x) {
        if ((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || x >= 174) continue;
        if (x == b) return 256 + n;
        ++n;
    }
    return b;
}
static char *byte_encode(span s, size_t *out_len) {
    char *p = malloc(s.len * 4 + 1), *q = p;
    if (!p) exit(1);
    for (size_t i = 0; i < s.len; ++i) utf8_put(&q, byte_cp((uint8_t)s.ptr[i]));
    *q = 0; *out_len = (size_t)(q - p); return p;
}
static int map_rank(const ds4f_tokenizer *t, const owned *a, const owned *b) {
    size_t n = a->len + 1 + b->len;
    char stack[512]; char *p = n <= sizeof(stack) ? stack : malloc(n);
    if (!p) exit(1);
    memcpy(p, a->ptr, a->len); p[a->len] = ' '; memcpy(p + a->len + 1, b->ptr, b->len);
    int rank = -1;
    map_get((const map_entry *)t->merge_map, t->merge_map_cap, p, n, &rank);
    if (p != stack) free(p); return rank;
}
static void bpe_piece(const ds4f_tokenizer *t, span raw, ds4f_tokens *out) {
    size_t enc_len; char *enc = byte_encode(raw, &enc_len);
    size_t n = 0, cap = 32; owned *sym = calloc(cap, sizeof(*sym));
    for (size_t off = 0; off < enc_len;) {
        int k = utf8_len((uint8_t)enc[off]); if (off + (size_t)k > enc_len) k = 1;
        if (n == cap) { cap *= 2; sym = realloc(sym, cap * sizeof(*sym)); }
        sym[n].ptr = malloc((size_t)k); sym[n].len = (size_t)k; memcpy(sym[n].ptr, enc + off, (size_t)k); n++; off += (size_t)k;
    }
    for (;;) {
        int best = -1, rank = INT32_MAX;
        for (size_t i = 0; i + 1 < n; ++i) { int r = map_rank(t, &sym[i], &sym[i + 1]); if (r >= 0 && r < rank) { rank = r; best = (int)i; } }
        if (best < 0) break;
        owned m = {NULL, sym[best].len + sym[best + 1].len}; m.ptr = malloc(m.len);
        memcpy(m.ptr, sym[best].ptr, sym[best].len); memcpy(m.ptr + sym[best].len, sym[best + 1].ptr, sym[best + 1].len);
        free(sym[best].ptr); free(sym[best + 1].ptr); sym[best] = m;
        for (size_t j = (size_t)best + 1; j + 1 < n; ++j) sym[j] = sym[j + 1];
        --n;
    }
    for (size_t i = 0; i < n; ++i) {
        int id = -1;
        if (map_get((const map_entry *)t->token_map, t->token_map_cap, sym[i].ptr, sym[i].len, &id)) token_push(out, id);
        else for (size_t j = 0; j < sym[i].len;) {
            int k = utf8_len((uint8_t)sym[i].ptr[j]); if (j + (size_t)k > sym[i].len) k = 1;
            if (map_get((const map_entry *)t->token_map, t->token_map_cap, sym[i].ptr + j, (size_t)k, &id)) token_push(out, id);
            j += (size_t)k;
        }
        free(sym[i].ptr);
    }
    free(sym); free(enc);
}
static int is_alpha(uint8_t c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }
static int is_digit(uint8_t c) { return c >= '0' && c <= '9'; }
static int is_space(uint8_t c) { return c == ' ' || (c >= '\t' && c <= '\r'); }
static int is_nl(uint8_t c) { return c == '\n' || c == '\r'; }
static int is_punct(uint8_t c) { return (c >= '!' && c <= '/') || (c >= ':' && c <= '@') || (c >= '[' && c <= '`') || (c >= '{' && c <= '~'); }
static int cjk_at(const char *s, size_t n, size_t p) {
    if ((uint8_t)s[p] < 128) return 0;
    uint32_t cp = 0; uint8_t c = (uint8_t)s[p]; int k = utf8_len(c);
    if (p + (size_t)k > n) k = 1;
    if (k == 3) cp = ((uint32_t)(c & 15) << 12) | ((uint32_t)((uint8_t)s[p + 1] & 63) << 6) | ((uint8_t)s[p + 2] & 63);
    else if (k == 2) cp = ((uint32_t)(c & 31) << 6) | ((uint8_t)s[p + 1] & 63);
    else if (k == 4) cp = ((uint32_t)(c & 7) << 18) | ((uint32_t)((uint8_t)s[p + 1] & 63) << 12) | ((uint32_t)((uint8_t)s[p + 2] & 63) << 6) | ((uint8_t)s[p + 3] & 63);
    return (cp >= 0x4e00 && cp <= 0x9fa5) || (cp >= 0x3040 && cp <= 0x30ff);
}
static size_t next_char(const char *s, size_t n, size_t p) { int k = utf8_len((uint8_t)s[p]); return p + (p + (size_t)k <= n ? (size_t)k : 1); }
static int letter_like(const char *s, size_t p) { return (uint8_t)s[p] >= 128 || is_alpha((uint8_t)s[p]); }
static void tokenize_joyai(const ds4f_tokenizer *t, const char *text, ds4f_tokens *out) {
    size_t n = strlen(text), p = 0;
    while (p < n) {
        size_t start = p; uint8_t c = (uint8_t)text[p];
        if (is_digit(c)) { int k = 0; while (p < n && is_digit((uint8_t)text[p]) && k++ < 3) ++p; }
        else if (cjk_at(text, n, p)) while (p < n && cjk_at(text, n, p)) p = next_char(text, n, p);
        else if (is_punct(c) && p + 1 < n && is_alpha((uint8_t)text[p + 1])) { ++p; while (p < n && is_alpha((uint8_t)text[p])) ++p; }
        else if (letter_like(text, p)) while (p < n && letter_like(text, p)) p = next_char(text, n, p);
        else if (!is_nl(c) && !is_punct(c) && p + 1 < n && letter_like(text, p + 1)) { ++p; while (p < n && letter_like(text, p)) p = next_char(text, n, p); }
        else if (c == ' ' && p + 1 < n && is_punct((uint8_t)text[p + 1])) { ++p; while (p < n && is_punct((uint8_t)text[p])) ++p; while (p < n && is_nl((uint8_t)text[p])) ++p; }
        else if (is_punct(c)) { while (p < n && is_punct((uint8_t)text[p])) ++p; while (p < n && is_nl((uint8_t)text[p])) ++p; }
        else if (is_space(c)) {
            size_t q = p, last_nl = 0; while (q < n && is_space((uint8_t)text[q])) { if (is_nl((uint8_t)text[q])) last_nl = q + 1; ++q; }
            if (last_nl) p = last_nl; else if (q < n && q > p + 1 && (letter_like(text, q) || is_punct((uint8_t)text[q]))) p = q - 1; else p = q;
        } else p = next_char(text, n, p);
        if (p == start) p = next_char(text, n, p);
        bpe_piece(t, (span){text + start, p - start}, out);
    }
}

int ds4f_tokenize_text(const ds4f_tokenizer *t, const char *text, ds4f_tokens *out) {
    if (!t || !text || !out) return -1;
    tokenize_joyai(t, text, out); return 0;
}
int ds4f_tokenize_chat(const ds4f_tokenizer *t, const char *prompt, int no_think, ds4f_tokens *out) {
    if (!t || t->bos_id < 0 || t->user_id < 0 || t->assistant_id < 0) return -1;
    /* DwarfStar's V4 chat path starts directly with <｜User｜>; BOS is
     * reserved in the vocabulary but is not emitted by its prompt builder. */
    token_push(out, t->user_id);
    tokenize_joyai(t, prompt ? prompt : "", out); token_push(out, t->assistant_id);
    token_push(out, no_think ? t->think_end_id : t->think_start_id);
    return 0;
}
static uint32_t decode_cp(const char *s, size_t n, size_t *p) {
    uint8_t c = (uint8_t)s[(*p)++];
    if (c < 128) return c;
    int k = utf8_len(c); if (*p + (size_t)k - 1 > n) return c;
    uint32_t cp = 0;
    if (k == 2) cp = ((uint32_t)(c & 31) << 6) | ((uint8_t)s[(*p)++] & 63);
    else if (k == 3) cp = ((uint32_t)(c & 15) << 12) | ((uint32_t)((uint8_t)s[(*p)++] & 63) << 6) | ((uint8_t)s[(*p)++] & 63);
    else cp = ((uint32_t)(c & 7) << 18) | ((uint32_t)((uint8_t)s[(*p)++] & 63) << 12) | ((uint32_t)((uint8_t)s[(*p)++] & 63) << 6) | ((uint8_t)s[(*p)++] & 63);
    return cp;
}
static int cp_byte(uint32_t cp) {
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) return (int)cp;
    uint32_t n = 0; for (uint32_t b = 0; b < 256; ++b) { if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174) continue; if (cp == 256 + n) return (int)b; ++n; }
    return -1;
}
int ds4f_token_text(const ds4f_tokenizer *t, int id, char **out, size_t *out_len) {
    if (!t || !out || id < 0 || (size_t)id >= t->n_vocab) return -1;
    const ds4f_vocab_piece *s = &t->token[id]; char *p = malloc(s->len + 1); if (!p) return -1;
    size_t n = 0, pos = 0; while (pos < s->len) { uint32_t cp = decode_cp(s->ptr, s->len, &pos); int b = cp_byte(cp); if (b >= 0) p[n++] = (char)b; }
    p[n] = 0; *out = p; if (out_len) *out_len = n; return 0;
}
int ds4f_token_is_stop(const ds4f_tokenizer *t, int id) { return t && id == t->eos_id; }
