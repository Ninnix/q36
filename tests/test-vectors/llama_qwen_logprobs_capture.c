#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama.h"

typedef struct {
    llama_token id;
    float logit;
    float logprob;
} token_prob;

static void *xmalloc(size_t size) {
    void *p = malloc(size ? size : 1);
    if (!p) {
        fputs("llama_qwen_logprobs_capture: out of memory\n", stderr);
        exit(1);
    }
    return p;
}

static char *read_file(const char *path, size_t *len) {
    FILE *fp = fopen(path, "rb");
    char *buf;
    long size;
    if (!fp || fseek(fp, 0, SEEK_END) != 0 ||
        (size = ftell(fp)) < 0 || fseek(fp, 0, SEEK_SET) != 0) {
        if (fp) fclose(fp);
        return NULL;
    }
    buf = xmalloc((size_t)size + 1);
    size_t nread = fread(buf, 1, (size_t)size, fp);
    int close_rc = fclose(fp);
    if (nread != (size_t)size || close_rc != 0) {
        free(buf);
        return NULL;
    }
    buf[size] = '\0';
    *len = (size_t)size;
    return buf;
}

static char *render_prompt(const char *user, size_t user_len, bool hf) {
    const char *prefix = hf ? "<|im_start|>user\n" :
                              "<|endoftext|><|im_start|>user\n";
    const char *suffix =
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    size_t prefix_len = strlen(prefix);
    size_t suffix_len = strlen(suffix);
    char *out = xmalloc(prefix_len + user_len + suffix_len + 1);
    memcpy(out, prefix, prefix_len);
    memcpy(out + prefix_len, user, user_len);
    memcpy(out + prefix_len + user_len, suffix, suffix_len + 1);
    return out;
}

static char *token_text(const struct llama_vocab *vocab, llama_token token,
                        size_t *len) {
    int cap = 256;
    char *buf = xmalloc((size_t)cap + 1);
    int n = llama_token_to_piece(vocab, token, buf, cap, 0, true);
    if (n < 0) {
        cap = -n + 8;
        char *newbuf = realloc(buf, (size_t)cap + 1);
        if (!newbuf) {
            free(buf);
            return NULL;
        }
        buf = newbuf;
        n = llama_token_to_piece(vocab, token, buf, cap, 0, true);
    }
    if (n < 0) {
        free(buf);
        return NULL;
    }
    buf[n] = '\0';
    *len = (size_t)n;
    return buf;
}

static void print_json_string(FILE *fp, const char *s, size_t len) {
    fputc('"', fp);
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '\\': fputs("\\\\", fp); break;
        case '"':  fputs("\\\"", fp); break;
        case '\b': fputs("\\b", fp); break;
        case '\f': fputs("\\f", fp); break;
        case '\n': fputs("\\n", fp); break;
        case '\r': fputs("\\r", fp); break;
        case '\t': fputs("\\t", fp); break;
        default:
            if (c < 0x20 || c >= 0x80) fprintf(fp, "\\u%04x", c);
            else fputc(c, fp);
            break;
        }
    }
    fputc('"', fp);
}

static void print_json_bytes(FILE *fp, const char *s, size_t len) {
    fputc('[', fp);
    for (size_t i = 0; i < len; i++) {
        if (i) fputs(", ", fp);
        fprintf(fp, "%u", (unsigned char)s[i]);
    }
    fputc(']', fp);
}

static llama_token *tokenize_prompt(const struct llama_vocab *vocab,
                                    const char *prompt, int *count) {
    int len = (int)strlen(prompt);
    int n = llama_tokenize(vocab, prompt, len, NULL, 0, false, true);
    llama_token *tokens;
    if (n >= 0) return NULL;
    tokens = xmalloc((size_t)-n * sizeof(*tokens));
    n = llama_tokenize(vocab, prompt, len, tokens, -n, false, true);
    if (n < 0) {
        free(tokens);
        return NULL;
    }
    *count = n;
    return tokens;
}

static void batch_set(struct llama_batch *batch, const llama_token *tokens,
                      int count, int pos0) {
    static llama_seq_id seq0;
    batch->n_tokens = count;
    for (int i = 0; i < count; i++) {
        batch->token[i] = tokens[i];
        batch->pos[i] = pos0 + i;
        batch->n_seq_id[i] = 1;
        batch->seq_id[i][0] = seq0;
        batch->logits[i] = i == count - 1;
    }
}

static token_prob *topk_from_logits(const float *logits, int n_vocab,
                                    int k, int *count) {
    float max_logit = -INFINITY;
    double sum = 0.0;
    double logsum;
    token_prob *top;
    int n = 0;
    if (k > n_vocab) k = n_vocab;
    if (k <= 0) return NULL;
    for (int i = 0; i < n_vocab; i++)
        if (logits[i] > max_logit) max_logit = logits[i];
    for (int i = 0; i < n_vocab; i++)
        sum += exp((double)logits[i] - max_logit);
    logsum = log(sum) + max_logit;

    top = xmalloc((size_t)k * sizeof(*top));
    for (int id = 0; id < n_vocab; id++) {
        float logit = logits[id];
        int pos;
        if (n < k) pos = n++;
        else if (logit <= top[n - 1].logit) continue;
        else pos = n - 1;
        while (pos > 0 && logit > top[pos - 1].logit) {
            top[pos] = top[pos - 1];
            pos--;
        }
        top[pos].id = id;
        top[pos].logit = logit;
        top[pos].logprob = (float)((double)logit - logsum);
    }
    *count = n;
    return top;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt_file = NULL;
    const char *out_path = NULL;
    const char *error = NULL;
    int ctx = 4096;
    int n_predict = 4;
    int top_k = 20;
    int threads = 1;
    bool hf_template = false;
    bool backend = false;
    struct llama_model *model = NULL;
    struct llama_context *lctx = NULL;
    struct llama_batch batch = {0};
    char *user = NULL;
    char *rendered = NULL;
    llama_token *prompt_tokens = NULL;
    int prompt_count = 0;
    FILE *fp = NULL;
    int rc = 1;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--prompt-file") && i + 1 < argc) prompt_file = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n-predict") && i + 1 < argc) n_predict = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--top-k") && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hf-template")) hf_template = true;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            puts("usage: llama_qwen_logprobs_capture --model FILE --prompt-file FILE --out FILE [--ctx N] [--n-predict N] [--top-k N] [--threads N] [--hf-template]");
            return 0;
        } else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 2;
        }
    }
    if (!model_path || !prompt_file || !out_path) {
        fputs("missing required arguments\n", stderr);
        return 2;
    }

    llama_backend_init();
    backend = true;
    {
        struct llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        /* Repacked weights live in anonymous RAM instead of the mmap. */
        mp.use_extra_bufts = false;
        mp.check_tensors = false;
        model = llama_model_load_from_file(model_path, mp);
    }
    if (!model) {
        error = "failed to load model";
        goto done;
    }
    {
        struct llama_context_params cp = llama_context_default_params();
        cp.n_ctx = (uint32_t)ctx;
        cp.n_batch = (uint32_t)ctx;
        cp.n_ubatch = (uint32_t)(ctx < 2048 ? ctx : 2048);
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        cp.type_k = GGML_TYPE_F32;
        cp.type_v = GGML_TYPE_F32;
        cp.no_perf = true;
        lctx = llama_init_from_model(model, cp);
    }
    if (!lctx) {
        error = "failed to create context";
        goto done;
    }

    {
        const struct llama_vocab *vocab = llama_model_get_vocab(model);
        size_t user_len;
        user = read_file(prompt_file, &user_len);
        if (!user) {
            error = "failed to read prompt";
            goto done;
        }
        rendered = render_prompt(user, user_len, hf_template);
        prompt_tokens = tokenize_prompt(vocab, rendered, &prompt_count);
        if (!prompt_tokens) {
            error = "failed to tokenize prompt";
            goto done;
        }
    }

    batch = llama_batch_init(ctx, 0, 1);
    batch_set(&batch, prompt_tokens, prompt_count, 0);
    if (llama_decode(lctx, batch) != 0) {
        error = "prompt decode failed";
        goto done;
    }
    fp = fopen(out_path, "wb");
    if (!fp) {
        error = "failed to open output file";
        goto done;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"prompt_tokens\": %d,\n", prompt_count);
    fprintf(fp, "  \"ctx\": %d,\n", ctx);
    fprintf(fp, "  \"top_k\": %d,\n", top_k);
    fprintf(fp, "  \"template\": \"%s\",\n",
            hf_template ? "hf-text-only" : "q36-fixed-nothink");
    fprintf(fp, "  \"steps\": [\n");

    {
        const struct llama_vocab *vocab = llama_model_get_vocab(model);
        int n_vocab = llama_vocab_n_tokens(vocab);
        int pos = prompt_count;
        for (int step = 0; step < n_predict; step++) {
            const float *logits = llama_get_logits_ith(lctx, batch.n_tokens - 1);
            token_prob *top;
            int top_count = 0;
            llama_token selected;
            char *selected_text;
            size_t selected_len;
            if (!logits) {
                error = "missing logits";
                goto done;
            }
            top = topk_from_logits(logits, n_vocab, top_k, &top_count);
            selected = top_count ? top[0].id : LLAMA_TOKEN_NULL;
            selected_text = token_text(vocab, selected, &selected_len);
            if (!selected_text) {
                free(top);
                error = "failed to convert token to piece";
                goto done;
            }

            if (step) fprintf(fp, ",\n");
            fprintf(fp, "    {\n");
            fprintf(fp, "      \"step\": %d,\n", step);
            fprintf(fp, "      \"selected\": {\n");
            fprintf(fp, "        \"id\": %d,\n", (int)selected);
            fprintf(fp, "        \"text\": ");
            print_json_string(fp, selected_text, selected_len);
            fprintf(fp, ",\n        \"bytes\": ");
            print_json_bytes(fp, selected_text, selected_len);
            fprintf(fp, "\n      },\n");
            fprintf(fp, "      \"top_logprobs\": [\n");
            for (int i = 0; i < top_count; i++) {
                size_t text_len;
                char *text = token_text(vocab, top[i].id, &text_len);
                if (!text) {
                    free(selected_text);
                    free(top);
                    error = "failed to convert token to piece";
                    goto done;
                }
                if (i) fprintf(fp, ",\n");
                fprintf(fp, "        {\n");
                fprintf(fp, "          \"token\": {\n");
                fprintf(fp, "            \"id\": %d,\n", (int)top[i].id);
                fprintf(fp, "            \"text\": ");
                print_json_string(fp, text, text_len);
                fprintf(fp, ",\n            \"bytes\": ");
                print_json_bytes(fp, text, text_len);
                fprintf(fp, "\n          },\n");
                fprintf(fp, "          \"logit\": %.9g,\n", top[i].logit);
                fprintf(fp, "          \"logprob\": %.9g\n", top[i].logprob);
                fprintf(fp, "        }");
                free(text);
            }
            fprintf(fp, "\n      ]\n");
            fprintf(fp, "    }");
            free(selected_text);
            free(top);

            if (selected == LLAMA_TOKEN_NULL ||
                llama_vocab_is_eog(vocab, selected)) break;
            batch_set(&batch, &selected, 1, pos++);
            if (llama_decode(lctx, batch) != 0) {
                error = "decode step failed";
                goto done;
            }
        }
    }
    fprintf(fp, "\n  ]\n}\n");
    if (fclose(fp) != 0) {
        fp = NULL;
        error = "failed to close output file";
        goto done;
    }
    fp = NULL;
    rc = 0;

done:
    if (error) fprintf(stderr, "llama_qwen_logprobs_capture: %s\n", error);
    if (fp) fclose(fp);
    if (batch.token) llama_batch_free(batch);
    if (lctx) llama_free(lctx);
    if (model) llama_model_free(model);
    if (backend) llama_backend_free();
    free(prompt_tokens);
    free(rendered);
    free(user);
    return rc;
}
