#include "ds4.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum {
    CONTEXT_TOKENS = 32 * 1024,
    DEFAULT_OUTPUT_TOKENS = 16,
    DEFAULT_CACHE_EXPERTS = 440,
    MAX_CACHE_EXPERTS = 1200,
};

typedef struct {
    ds4_engine *engine;
    int emitted;
} output_state;

static void usage(const char *program) {
    fprintf(stderr,
            "usage: %s MODEL.gguf [PROMPT] [TOKENS]\n"
            "Fixed configuration: Metal, 32K context, greedy decode, no MTP. "
            "Set DS4F_FAST_CACHE_EXPERTS=1..1200 to override the 440-expert cache.\n",
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

static int cache_experts_from_env(uint32_t *out) {
    const char *text = getenv("DS4F_FAST_CACHE_EXPERTS");
    if (!text || !text[0]) {
        *out = DEFAULT_CACHE_EXPERTS;
        return 0;
    }
    int value = parse_positive(text, -1);
    if (value < 1 || value > MAX_CACHE_EXPERTS) return -1;
    *out = (uint32_t)value;
    return 0;
}

static int enter_runtime_dir(const char *program) {
    char executable[PATH_MAX];
    char runtime_dir[PATH_MAX];
    if (!realpath(program, executable)) return -1;
    char *slash = strrchr(executable, '/');
    if (!slash) return -1;
    *slash = '\0';
    int written = snprintf(runtime_dir, sizeof(runtime_dir), "%s/runtime", executable);
    if (written < 0 || (size_t)written >= sizeof(runtime_dir)) return -1;
    return chdir(runtime_dir);
}

static char *render_prompt(const char *prompt) {
    static const char prefix[] = "<｜User｜>";
    static const char suffix[] = "<｜Assistant｜></think>";
    size_t size = strlen(prefix) + strlen(prompt) + strlen(suffix) + 1;
    char *rendered = malloc(size);
    if (!rendered) return NULL;
    snprintf(rendered, size, "%s%s%s", prefix, prompt, suffix);
    return rendered;
}

static void emit_token(void *ud, int token) {
    output_state *state = ud;
    size_t len = 0;
    char *text = ds4_token_text(state->engine, token, &len);
    if (!text) return;
    fwrite(text, 1, len, stdout);
    fflush(stdout);
    free(text);
    if (getenv("DS4F_FAST_TRACE_IDS")) {
        fprintf(stderr, "trace token[%d]=%d\n", state->emitted, token);
    }
    state->emitted++;
}

static void finish_output(void *ud) {
    output_state *state = ud;
    if (state->emitted) fputc('\n', stdout);
}

int main(int argc, char **argv) {
    if (argc > 1 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) {
        usage(argv[0]);
        return 0;
    }
    if (argc < 2 || argc > 4) {
        usage(argv[0]);
        return 2;
    }

    uint32_t cache_experts = 0;
    if (cache_experts_from_env(&cache_experts)) {
        fprintf(stderr, "ds4f-q4-speed: DS4F_FAST_CACHE_EXPERTS must be between 1 and 1200\n");
        return 2;
    }
    int output_tokens = parse_positive(argc >= 4 ? argv[3] : NULL,
                                       DEFAULT_OUTPUT_TOKENS);
    if (output_tokens < 0) {
        fprintf(stderr, "ds4f-q4-speed: TOKENS must be a positive integer\n");
        return 2;
    }

    char model_path[PATH_MAX];
    if (!realpath(argv[1], model_path)) {
        perror(argv[1]);
        return 2;
    }
    char *rendered = render_prompt(argc >= 3 ? argv[2] : "");
    if (!rendered) {
        perror("malloc");
        return 1;
    }
    if (enter_runtime_dir(argv[0])) {
        fprintf(stderr, "ds4f-q4-speed: cannot locate the bundled runtime\n");
        free(rendered);
        return 1;
    }

    (void)setenv("DS4_METAL_ENABLE_STREAMING_IQ2_CPU_ROUTER", "1", 0);
    ds4_engine_options options = {
        .model_path = model_path,
        .backend = DS4_BACKEND_METAL,
        .context_size = CONTEXT_TOKENS,
        .mtp_draft_tokens = 0,
        .mtp_margin = 3.0f,
        .ssd_streaming = true,
        .ssd_streaming_cache_experts = cache_experts,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &options) != 0 || !engine) {
        fprintf(stderr, "ds4f-q4-speed: failed to initialize the Metal graph\n");
        free(rendered);
        return 1;
    }

    ds4_tokens prompt_tokens = {0};
    ds4_tokenize_rendered_chat(engine, rendered, &prompt_tokens);
    free(rendered);
    if (prompt_tokens.len == 0) {
        fprintf(stderr, "ds4f-q4-speed: failed to encode the chat prompt\n");
        ds4_engine_close(engine);
        return 1;
    }

    output_state output = { .engine = engine };
    int rc = ds4_engine_generate_argmax(engine, &prompt_tokens, output_tokens,
                                        CONTEXT_TOKENS, emit_token,
                                        finish_output, &output, NULL, NULL);
    ds4_tokens_free(&prompt_tokens);
    ds4_engine_close(engine);
    return rc == 0 ? 0 : 1;
}
