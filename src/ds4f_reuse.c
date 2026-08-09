#include "ds4.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
/*
 * Persistent target-only deployment adapter. Each stdin line starts a fresh
 * chat context, but the verified DwarfStar Metal graph and SSD expert cache
 * remain alive so independent later requests avoid cache cold-start cost.
 */
enum {
    DS4F_FAST_CONTEXT_DEFAULT = 131072,
    DS4F_FAST_DEFAULT_TOKENS = 16,
};

static const uint64_t DS4F_FAST_DEFAULT_CACHE_BYTES = 6ull * 1024ull * 1024ull * 1024ull;

typedef struct {
    ds4_engine *engine;
    int emitted;
} output_state;

static void usage(const char *program) {
    fprintf(stderr,
            "usage: %s MODEL.gguf [TOKENS]\n"
            "Reads one prompt per stdin line. Each request gets a fresh context while the 6GiB SSD expert cache stays resident. "
            "Defaults to 128K context; set DS4F_FAST_CONTEXT_K=32, 128, or 256.\n",
            program);
}

static int parse_positive(const char *text, int fallback) {
    if (!text) return fallback;
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || end == text || *end || value < 1 || value > INT_MAX) return -1;
    return (int)value;
}

static int cache_bytes_from_env(uint64_t *out) {
    const char *text = getenv("DS4F_FAST_CACHE_GIB");
    if (!text || !text[0]) {
        *out = DS4F_FAST_DEFAULT_CACHE_BYTES;
        return 0;
    }
    char *end = NULL;
    errno = 0;
    double gib = strtod(text, &end);
    if (errno || end == text || *end || !(gib >= 1.0 && gib <= 6.0)) return -1;
    *out = (uint64_t)(gib * 1024.0 * 1024.0 * 1024.0);
    return *out ? 0 : -1;
}

static int context_tokens_from_env(int *out) {
    if (!out) return -1;
    const char *text = getenv("DS4F_FAST_CONTEXT_K");
    if (!text || !text[0]) {
        *out = DS4F_FAST_CONTEXT_DEFAULT;
        return 0;
    }
    char *end = NULL;
    errno = 0;
    long kib = strtol(text, &end, 10);
    if (errno || end == text || *end || (kib != 32 && kib != 128 && kib != 256)) return -1;
    *out = (int)kib * 1024;
    return 0;
}

static int enter_reference_dir(const char *program) {
    char executable[PATH_MAX];
    char reference_dir[PATH_MAX];
    if (!realpath(program, executable)) return -1;
    char *slash = strrchr(executable, '/');
    if (!slash) return -1;
    *slash = '\0';
    int written = snprintf(reference_dir, sizeof(reference_dir), "%s/reference-ds4",
                           executable);
    if (written < 0 || (size_t)written >= sizeof(reference_dir)) return -1;
    return chdir(reference_dir);
}

static void emit_token(void *ud, int token) {
    output_state *state = ud;
    size_t len = 0;
    char *text = ds4_token_text(state->engine, token, &len);
    if (!text) return;
    fwrite(text, 1, len, stdout);
    fflush(stdout);
    free(text);
    state->emitted++;
}

static void finish_output(void *ud) {
    output_state *state = ud;
    if (state->emitted) fputc('\n', stdout);
}
static int generate_prompt(ds4_engine *engine, const char *prompt, int tokens,
                           int context_size) {

    const char prefix[] = "<｜User｜>";
    const char suffix[] = "<｜Assistant｜></think>";
    const size_t prompt_len = strlen(prompt);
    if (prompt_len > SIZE_MAX - strlen(prefix) - strlen(suffix) - 1u) return -1;
    const size_t rendered_len = strlen(prefix) + prompt_len + strlen(suffix);
    char *rendered = malloc(rendered_len + 1u);
    if (!rendered) return -1;
    snprintf(rendered, rendered_len + 1u, "%s%s%s", prefix, prompt, suffix);

    ds4_tokens prompt_tokens = {0};
    ds4_tokenize_rendered_chat(engine, rendered, &prompt_tokens);
    free(rendered);
    if (prompt_tokens.len == 0) return -1;

    output_state output = { .engine = engine };
    int rc = ds4_engine_generate_argmax(engine, &prompt_tokens, tokens,
                                        context_size, emit_token,
                                        finish_output, &output, NULL, NULL);
    ds4_tokens_free(&prompt_tokens);
    return rc;
}

int main(int argc, char **argv) {
    if (argc > 1 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) {
        usage(argv[0]);
        return 0;
    }
    if (argc < 2 || argc > 3) {
        usage(argv[0]);
        return 2;
    }
    uint64_t cache_bytes = 0;
    if (cache_bytes_from_env(&cache_bytes)) {
        fprintf(stderr, "ds4f-reuse: DS4F_FAST_CACHE_GIB must be between 1 and 6\n");
        return 2;
    }
    int context_size = 0;
    if (context_tokens_from_env(&context_size)) {
        fprintf(stderr, "ds4f-reuse: DS4F_FAST_CONTEXT_K must be 32, 128, or 256\n");
        return 2;
    }
    char model_path[PATH_MAX];
    if (!realpath(argv[1], model_path)) {
        perror(argv[1]);
        return 2;
    }
    const int tokens = parse_positive(argc == 3 ? argv[2] : NULL,
                                      DS4F_FAST_DEFAULT_TOKENS);
    if (tokens < 0) {
        fprintf(stderr, "ds4f-reuse: TOKENS must be a positive integer\n");
        return 2;
    }
    if (enter_reference_dir(argv[0])) {
        fprintf(stderr, "ds4f-reuse: cannot locate reference-ds4 next to the executable\n");
        return 1;
    }

    ds4_engine_options options = {
        .model_path = model_path,
        .backend = DS4_BACKEND_METAL,
        .context_size = context_size,
        .mtp_draft_tokens = 1,
        .mtp_margin = 3.0f,
        .ssd_streaming = true,
        .ssd_streaming_cache_bytes = cache_bytes,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &options) != 0 || !engine) {
        fprintf(stderr, "ds4f-reuse: failed to initialize the Metal graph\n");
        return 1;
    }

    char *line = NULL;
    size_t line_cap = 0;
    int rc = 0;
    while (getline(&line, &line_cap, stdin) >= 0) {
        size_t line_len = strlen(line);
        while (line_len && (line[line_len - 1u] == '\n' || line[line_len - 1u] == '\r'))
            line[--line_len] = '\0';
        if (generate_prompt(engine, line, tokens, context_size) != 0) {
            fprintf(stderr, "ds4f-reuse: generation failed\n");
            rc = 1;
            break;
        }
    }
    if (ferror(stdin)) rc = 1;
    free(line);
    ds4_engine_close(engine);
    return rc;
}
