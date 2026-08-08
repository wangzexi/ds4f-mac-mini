#ifndef DS4F_TOKENIZER_H
#define DS4F_TOKENIZER_H

#include "ds4f_gguf.h"
#include <stddef.h>

typedef struct {
    int *v;
    size_t len;
    size_t cap;
} ds4f_tokens;

typedef struct {
    char *ptr;
    size_t len;
} ds4f_vocab_piece;

typedef struct {
    ds4f_vocab_piece *token;
    size_t n_vocab;
    char **merge;
    size_t n_merge;
    void *token_map;
    size_t token_map_cap;
    void *merge_map;
    size_t merge_map_cap;
    int bos_id;
    int eos_id;
    int user_id;
    int assistant_id;
    int think_start_id;
    int think_end_id;
} ds4f_tokenizer;

int ds4f_tokenizer_open(const ds4f_gguf *g, ds4f_tokenizer *t);
void ds4f_tokenizer_close(ds4f_tokenizer *t);
int ds4f_tokenize_text(const ds4f_tokenizer *t, const char *text,
                       ds4f_tokens *out);
int ds4f_tokenize_chat(const ds4f_tokenizer *t, const char *prompt,
                       int no_think, ds4f_tokens *out);
void ds4f_tokens_free(ds4f_tokens *t);
int ds4f_token_text(const ds4f_tokenizer *t, int id, char **out,
                    size_t *out_len);
int ds4f_token_is_stop(const ds4f_tokenizer *t, int id);

#endif
