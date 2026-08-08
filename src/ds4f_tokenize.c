#include "ds4f_gguf.h"
#include "ds4f_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL.gguf TEXT [--chat]\n", argv[0]); return 2; }
    ds4f_gguf g;
    if (ds4f_gguf_open(&g, argv[1])) { perror("GGUF"); return 1; }
    ds4f_tokenizer tok;
    if (ds4f_tokenizer_open(&g, &tok)) { perror("tokenizer"); return 1; }
    ds4f_tokens ids = {0};
    int rc = (argc > 3 && !strcmp(argv[3], "--chat"))
        ? ds4f_tokenize_chat(&tok, argv[2], 1, &ids)
        : ds4f_tokenize_text(&tok, argv[2], &ids);
    if (rc) { fprintf(stderr, "tokenize failed\n"); return 1; }
    printf("vocab=%zu bos=%d eos=%d user=%d assistant=%d think_end=%d\n",
           tok.n_vocab, tok.bos_id, tok.eos_id, tok.user_id, tok.assistant_id, tok.think_end_id);
    for (size_t i = 0; i < ids.len; ++i) printf("%s%d", i ? " " : "", ids.v[i]);
    putchar('\n');
    for (size_t i = 0; i < ids.len; ++i) { char *s = NULL; size_t n = 0; if (!ds4f_token_text(&tok, ids.v[i], &s, &n)) { printf("%zu:%d:", i, ids.v[i]); for (size_t j = 0; j < n; ++j) printf("%02x", (unsigned char)s[j]); putchar('\n'); free(s); } }
    ds4f_tokens_free(&ids); ds4f_tokenizer_close(&tok); ds4f_gguf_close(&g); return 0;
}
