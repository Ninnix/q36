#define Q36_SERVER_TEST
#define Q36_SERVER_TEST_NO_MAIN
#include "../q36_server.c"
#include "../q36_quant.h"
#include "../q36_mtp.h"
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#ifdef Q36_WITH_LLAMA
#include "llama.h"
#include "ggml-cpu.h"
#endif
#ifndef Q36_NO_GPU
#include "../q36_gpu.h"
#endif

static bool test_is_cpu_only_build(void) {
#ifdef Q36_NO_GPU
    return true;
#else
    return false;
#endif
}

#ifndef Q36_NEG_INF
#define Q36_NEG_INF (-1.0e30f)
#endif

static q36_engine *test_engine;
static const char *test_model_override;
static const char *test_case_override;
static q36_dist_options test_parity_distributed;

static void test_skip(const char *name, const char *reason) {
    fprintf(stderr, "%s: SKIP (%s)\n", name, reason);
}

static bool test_model_available(const char *path) {
    if (!path || !path[0]) return false;
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    fclose(fp);
    return true;
}

static q36_backend test_default_backend(void) {
    return test_is_cpu_only_build() ? Q36_BACKEND_CPU : Q36_BACKEND_VULKAN;
}

static const char *test_model_path(void) {
    const char *model_path = getenv("Q36_TEST_MODEL");
    if (test_model_override) return test_model_override;
    return (model_path && model_path[0]) ? model_path : Q36_DEFAULT_MODEL_PATH;
}

static const char *test_mtp_path(void) {
    const char *path = getenv("Q36_TEST_MTP");
    return path && path[0] ? path : test_model_path();
}

static q36_engine *test_get_engine(void) {
    if (test_engine) return test_engine;

    q36_engine_options opt = {
        .model_path = test_model_path(),
        .backend = test_default_backend(),
    };
    TEST_ASSERT(q36_engine_open(&test_engine, &opt) == 0);
    return test_engine;
}

static void test_close_engine(void) {
    q36_engine_close(test_engine);
    test_engine = NULL;
}

static bool test_require_session_backend(const char *name, q36_engine *engine, int ctx_size) {
    q36_session *session = NULL;
    if (q36_session_create(&session, engine, ctx_size) == 0) {
        q36_session_free(session);
        return true;
    }
    test_skip(name, "Qwen model load is wired but inference/session path is not ported yet");
    return false;
}

static uint64_t test_round_up_u64(uint64_t n, uint64_t align) {
    return (n + align - 1) & ~(align - 1);
}

static uint16_t test_float_to_f16(float f) {
    union {
        float f;
        uint32_t u;
    } v = { .f = f };

    uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

static char *test_read_file(const char *path);
static q36_engine *test_open_runtime_engine(const char *model_path, int threads);

typedef struct {
    const char *model_path;
    const char *prompt_file;
    const char *case_id;
    const char *out_path;
    int tokens;
    int ctx;
    int top_k;
    int threads;
    bool hf_template;
} test_dump_config;

typedef struct {
    double *v;
    int len;
    int cap;
} parity_dvec;

typedef struct {
    int id;
    float q36_logit;
    float llama_logit;
    float diff;
    char text[160];
    char cls[32];
} parity_token_diff;

typedef struct {
    char *id;
    char *category;
    char *prompt;
} parity_prompt;

typedef struct {
    parity_prompt *v;
    int len;
    int cap;
} parity_prompt_vec;

typedef struct {
    char name[64];
    int cases;
    int positions;
    int top1_matches;
    int flagged_prompts;
    parity_dvec top5;
    parity_dvec kl;
    parity_dvec maxdiff;
} parity_category;

typedef struct {
    const parity_prompt *prompt;
    int prompt_tokens;
    int positions;
    int top1_matches;
    int first_div_step;
    int div_q36;
    int div_llama;
    int worst_step;
    double top5_median;
    double kl_median;
    double kl_p99;
    double maxdiff_p50;
    double maxdiff_p95;
    double maxdiff_p99;
    double worst_maxdiff;
    parity_token_diff worst_tokens[5];
    int worst_ntokens;
    parity_dvec top5;
    parity_dvec kl;
    parity_dvec maxdiff;
} parity_prompt_result;

typedef struct {
    const char *model_path;
    const char *report_path;
    int steps;
    int ctx_size;
    int threads;
    int quality_limit;
    int skip_long;
    int hf_template;
} parity_config;

static void *test_xrealloc(void *ptr, size_t n) {
    void *out = realloc(ptr, n);
    if (!out) {
        fprintf(stderr, "q36-test: out of memory\n");
        exit(1);
    }
    return out;
}

static char *test_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *out = malloc(n);
    TEST_ASSERT(out != NULL);
    if (!out) exit(1);
    memcpy(out, s, n);
    return out;
}

static void *test_xcalloc(size_t n, size_t size) {
    void *out = calloc(n ? n : 1, size ? size : 1);
    TEST_ASSERT(out != NULL);
    if (!out) exit(1);
    return out;
}

static int test_sample_argmax_local(const float *logits, int n) {
    int best = -1;
    float best_v = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        if (!isfinite(logits[i])) continue;
        if (best < 0 || logits[i] > best_v) {
            best = i;
            best_v = logits[i];
        }
    }
    return best;
}

static void test_dequant_row_cpu(uint32_t type, const void *src, float *dst, uint32_t n) {
    TEST_ASSERT(q36_quant_dequantize(type, src, dst, n));
}

static void test_quant_primitives(void) {
    typedef struct {
        uint16_t d;
        uint8_t qs[16];
    } q4_0_block;
    typedef struct {
        uint16_t d;
        int8_t qs[32];
    } q8_0_block;
    typedef struct {
        float d;
        int8_t qs[256];
        int16_t bsums[16];
    } q8_k_block;
    float x0[32];
    float xk[256];
    q8_0_block q0;
    q4_0_block q4;
    q8_k_block qk;
    float want = 0.0f;

    TEST_ASSERT(q36_quant_f32_to_f16(1.0f) == 0x3c00u);
    TEST_ASSERT(q36_quant_f32_to_f16(-2.0f) == 0xc000u);
    TEST_ASSERT(q36_quant_f16_to_f32(0x3c00u) == 1.0f);
    TEST_ASSERT(q36_quant_f16_to_f32(0xc000u) == -2.0f);

    for (uint32_t bits = 0; bits <= UINT16_MAX; bits++) {
        uint16_t bf16 = (uint16_t)bits;
        uint32_t got, want_bits = bits << 16;
        float value;
        TEST_ASSERT(q36_quant_dequantize(30, &bf16, &value, 1));
        memcpy(&got, &value, sizeof(got));
        TEST_ASSERT(got == want_bits);
    }

    for (int i = 0; i < 32; i++) x0[i] = (float)(i - 15) / 17.0f;
    q36_quant_q8_0(x0, &q0, 32);
    for (int i = 0; i < 32; i++) {
        float v = q36_quant_f16_to_f32(q0.d) * (float)q0.qs[i];
        want += v * v;
    }
    TEST_ASSERT(fabsf(q36_quant_dot_q8_0(&q0, &q0, 32) - want) < 1.0e-5f);

    for (int i = 0; i < 32; i++) x0[i] = (float)(i - 16);
    q36_quant_q4_0(x0, &q4, 32);
    {
        static const uint8_t want_qs[16] = {
            0x80, 0x91, 0x91, 0xa2, 0xa2, 0xb3, 0xb3, 0xc4,
            0xc4, 0xd5, 0xd5, 0xe6, 0xe6, 0xf7, 0xf7, 0xf8,
        };
        TEST_ASSERT(q4.d == 0x4000u);
        TEST_ASSERT(memcmp(q4.qs, want_qs, sizeof(want_qs)) == 0);
    }

    TEST_ASSERT(q36_parse_kv_cache_type("q4_0", &(q36_kv_cache_type){0}));
    TEST_ASSERT(!strcmp(q36_kv_cache_type_name(Q36_KV_CACHE_Q4_0), "q4_0"));
    for (int block = 0; block < 128; block++) {
        q4_0_block old, key;
        for (int i = 0; i < 32; i++) {
            float x = sinf((float)(block * 37 + i * 13)) * (0.2f + (float)(i % 7));
            if (block % 4 == 0) x += 2.0f;
            if (block % 4 == 1 && i == 0) x *= 20.0f;
            x0[i] = block == 0 ? 0.0f : x;
        }
        q36_quant_q4_0_kv(x0, &old, 32);
        q36_quant_q4_0_key(x0, &key, 32);
        double old_error = 0.0, key_error = 0.0;
        for (int i = 0; i < 32; i++) {
            int shift = i < 16 ? 0 : 4;
            int a_q = ((old.qs[i & 15] >> shift) & 15) - 8;
            int b_q = ((key.qs[i & 15] >> shift) & 15) - 8;
            double a = (double)x0[i] - (double)q36_quant_f16_to_f32(old.d) * a_q;
            double b = (double)x0[i] - (double)q36_quant_f16_to_f32(key.d) * b_q;
            old_error += a * a;
            key_error += b * b;
        }
        TEST_ASSERT(key_error <= old_error * 1.000001 + 1.0e-12);
    }
    TEST_ASSERT(q36_default_kv_cache_type_k(Q36_BACKEND_VULKAN, false) == Q36_KV_CACHE_Q8_0);
    TEST_ASSERT(q36_default_kv_cache_type_v(Q36_BACKEND_VULKAN, false) == Q36_KV_CACHE_Q4_0);
    TEST_ASSERT(q36_default_kv_cache_type_k(Q36_BACKEND_VULKAN, true) == Q36_KV_CACHE_F16);
    TEST_ASSERT(q36_default_kv_cache_type_v(Q36_BACKEND_VULKAN, true) == Q36_KV_CACHE_F16);
    TEST_ASSERT(q36_default_kv_cache_type_k(Q36_BACKEND_METAL, false) == Q36_KV_CACHE_Q8_0);
    TEST_ASSERT(q36_default_kv_cache_type_v(Q36_BACKEND_METAL, false) == Q36_KV_CACHE_Q4_0);
    TEST_ASSERT(q36_default_kv_cache_type_k(Q36_BACKEND_METAL, true) == Q36_KV_CACHE_F16);
    TEST_ASSERT(q36_default_kv_cache_type_v(Q36_BACKEND_METAL, true) == Q36_KV_CACHE_F16);
    TEST_ASSERT(q36_default_kv_cache_type_k(Q36_BACKEND_CPU, false) == Q36_KV_CACHE_F16);
    TEST_ASSERT(q36_default_kv_cache_type_v(Q36_BACKEND_CPU, false) == Q36_KV_CACHE_F16);

    /* Known correctly rounded Q8_K scales: fast reciprocal approximations
     * used to change these bytes in GPU builds of the CPU oracle. */
    for (int sign = 0; sign < 2; sign++) {
        memset(xk, 0, sizeof(xk));
        xk[0] = sign ? -0x1.fffffap-1f : 0x1.fffffap-1f;
        q36_quant_q8_k(xk, &qk, 256);
        uint32_t bits;
        memcpy(&bits, &qk.d, sizeof(bits));
        TEST_ASSERT(bits == (sign ? 0x3c010203u : 0xbc010203u));
        TEST_ASSERT(qk.qs[0] == -127 && qk.bsums[0] == -127);
        for (int i = 1; i < 256; i++) TEST_ASSERT(qk.qs[i] == 0);
    }
    for (int i = 0; i < 256; i++) xk[i] = (float)((i * 17) % 101 - 50) / 31.0f;
    q36_quant_q8_k(xk, &qk, 256);
    for (int group = 0; group < 16; group++) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += qk.qs[group * 16 + i];
        TEST_ASSERT(qk.bsums[group] == sum);
    }
}

static void test_ssd_cache_shrink(void) {
    q36_ssd_cache_plan plan;
    TEST_ASSERT(q36_ssd_auto_cache_plan(1000, 80, 0, 200, 10, 100, &plan));
    TEST_ASSERT(plan.model_target_bytes == 800 && plan.cache_experts == 60);
    TEST_ASSERT(q36_ssd_auto_cache_plan(1000, 80, 600, 200, 10, 100, &plan));
    TEST_ASSERT(plan.model_target_bytes == 600 && plan.cache_experts == 40);
    TEST_ASSERT(q36_ssd_auto_cache_plan(UINT64_MAX, 80, 1000, 200, 10, 30, &plan));
    TEST_ASSERT(plan.model_target_bytes == 1000 && plan.cache_experts == 30);
    TEST_ASSERT(!q36_ssd_auto_cache_plan(1000, 96, 0, 200, 10, 100, &plan));
    TEST_ASSERT(!q36_ssd_auto_cache_plan(0, 80, 0, 200, 10, 100, &plan));
    TEST_ASSERT(q36_ssd_shrink_cache_experts(0) == 0);
    TEST_ASSERT(q36_ssd_shrink_cache_experts(1) == 0);
    TEST_ASSERT(q36_ssd_shrink_cache_experts(2) == 1);
    TEST_ASSERT(q36_ssd_shrink_cache_experts(10) == 9);
    TEST_ASSERT(q36_ssd_shrink_cache_experts(11) == 9);
    TEST_ASSERT(q36_ssd_shrink_cache_experts(7281) == 6552);
}

static int test_parse_positive_int(int argc, char **argv, int *i, const char *name) {
    char *end = NULL;
    long v;
    if (*i + 1 >= argc) {
        fprintf(stderr, "q36-test: missing value for %s\n", name);
        exit(2);
    }
    v = strtol(argv[++(*i)], &end, 10);
    if (!argv[*i][0] || !end || *end || v <= 0 || v > INT32_MAX) {
        fprintf(stderr, "q36-test: bad value for %s: %s\n", name, argv[*i]);
        exit(2);
    }
    return (int)v;
}

static char *test_json_string_value(const char *line, const char *key) {
    char pat[128];
    const char *p;
    size_t cap;
    char *out;
    size_t n = 0;

    snprintf(pat, sizeof(pat), "\"%s\"", key);
    p = strstr(line, pat);
    if (!p) return NULL;
    p += strlen(pat);
    while (*p && *p != ':') p++;
    if (*p != ':') return NULL;
    p++;
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p != '"') return NULL;
    p++;

    cap = strlen(p) + 1;
    out = malloc(cap);
    TEST_ASSERT(out != NULL);
    if (!out) exit(1);
    while (*p && *p != '"') {
        unsigned char c = (unsigned char)*p++;
        if (c == '\\') {
            c = (unsigned char)*p++;
            if (c == 'n') out[n++] = '\n';
            else if (c == 'r') out[n++] = '\r';
            else if (c == 't') out[n++] = '\t';
            else out[n++] = (char)c;
        } else {
            out[n++] = (char)c;
        }
    }
    out[n] = '\0';
    return out;
}

static char *test_load_prompt_text(const char *prompt_file, const char *case_id) {
    FILE *fp = fopen(prompt_file, "rb");
    char line[8192];
    if (!fp) return NULL;
    while (fgets(line, sizeof(line), fp)) {
        char *id = test_json_string_value(line, "id");
        char *prompt = test_json_string_value(line, "prompt");
        if (id && prompt && !strcmp(id, case_id)) {
            free(id);
            fclose(fp);
            return prompt;
        }
        free(id);
        free(prompt);
    }
    fclose(fp);
    return NULL;
}

static void test_append_char(char **buf, size_t *len, size_t *cap, char c) {
    if (*len + 1 >= *cap) {
        *cap = *cap ? *cap * 2 : 64;
        *buf = test_xrealloc(*buf, *cap);
    }
    (*buf)[(*len)++] = c;
    (*buf)[*len] = '\0';
}

static void test_append_cstr(char **buf, size_t *len, size_t *cap, const char *s) {
    if (!s) return;
    while (*s) test_append_char(buf, len, cap, *s++);
}

static char *test_render_hf_chat_prompt(const char *system, const char *prompt, q36_think_mode think_mode) {
    char *out = NULL;
    size_t len = 0;
    size_t cap = 0;

    if (system && system[0]) {
        test_append_cstr(&out, &len, &cap, "<|im_start|>system\n");
        test_append_cstr(&out, &len, &cap, system);
        test_append_cstr(&out, &len, &cap, "<|im_end|>\n");
    }

    test_append_cstr(&out, &len, &cap, "<|im_start|>user\n");
    test_append_cstr(&out, &len, &cap, prompt ? prompt : "");
    test_append_cstr(&out, &len, &cap, "<|im_end|>\n<|im_start|>assistant\n");
    if (think_mode == Q36_THINK_NONE) {
        test_append_cstr(&out, &len, &cap, "<think>\n\n</think>\n\n");
    } else {
        test_append_cstr(&out, &len, &cap, "<think>\n");
    }

    if (!out) out = test_strdup("");
    return out;
}

static char *test_render_legacy_chat_prompt(const char *prompt) {
    char *out = NULL;
    size_t len = 0;
    size_t cap = 0;

    test_append_cstr(&out, &len, &cap, "<|endoftext|><|im_start|>user\n");
    test_append_cstr(&out, &len, &cap, prompt ? prompt : "");
    test_append_cstr(&out, &len, &cap,
                     "<|im_end|>\n<|im_start|>assistant\n</think>");
    return out;
}

static void test_encode_prompt_mode(q36_engine *engine,
                                    const char *system,
                                    const char *prompt_text,
                                    q36_think_mode think_mode,
                                    bool hf_template,
                                    q36_tokens *prompt) {
    if (!hf_template) {
        q36_encode_chat_prompt(engine, system, prompt_text, think_mode, prompt);
        return;
    }
    {
        char *rendered = test_render_hf_chat_prompt(system, prompt_text, think_mode);
        q36_tokenize_rendered_chat(engine, rendered, prompt);
        free(rendered);
    }
}

static char *test_token_repr(q36_engine *engine, int token) {
    size_t len = 0;
    char *text = q36_token_text(engine, token, &len);
    char *out = NULL;
    size_t n = 0;
    size_t cap = 0;
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c == '\n') {
            test_append_char(&out, &n, &cap, '\\');
            test_append_char(&out, &n, &cap, 'n');
        } else if (c == '\r') {
            test_append_char(&out, &n, &cap, '\\');
            test_append_char(&out, &n, &cap, 'r');
        } else if (c == '\t') {
            test_append_char(&out, &n, &cap, '\\');
            test_append_char(&out, &n, &cap, 't');
        } else if (c == '\\') {
            test_append_char(&out, &n, &cap, '\\');
            test_append_char(&out, &n, &cap, '\\');
        } else if (c < 32 || c >= 127) {
            char hex[5];
            snprintf(hex, sizeof(hex), "%02x", c);
            test_append_char(&out, &n, &cap, '\\');
            test_append_char(&out, &n, &cap, 'x');
            test_append_char(&out, &n, &cap, hex[0]);
            test_append_char(&out, &n, &cap, hex[1]);
        } else {
            test_append_char(&out, &n, &cap, (char)c);
        }
    }
    free(text);
    if (!out) out = test_strdup("");
    return out;
}

static FILE *test_open_output(const char *path) {
    if (!path || !strcmp(path, "-")) return stdout;
    return fopen(path, "wb");
}

static int test_write_case_dump(const test_dump_config *cfg) {
    q36_engine *engine = NULL;
    q36_engine_options opt = {
        .model_path = cfg->model_path,
        .backend = Q36_BACKEND_CPU,
        .n_threads = cfg->threads,
    };
    q36_session *session = NULL;
    q36_tokens prompt = {0};
    char *prompt_text = NULL;
    char err[160] = {0};
    FILE *out = NULL;
    int rc = 1;

    prompt_text = test_load_prompt_text(cfg->prompt_file, cfg->case_id);
    if (!prompt_text) {
        fprintf(stderr, "q36-test: prompt %s not found in %s\n", cfg->case_id, cfg->prompt_file);
        goto done;
    }
    if (q36_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "q36-test: failed to open engine for %s\n", cfg->case_id);
        goto done;
    }
    test_encode_prompt_mode(engine, "", prompt_text, Q36_THINK_NONE, cfg->hf_template, &prompt);
    if (q36_session_create(&session, engine, cfg->ctx) != 0) {
        fprintf(stderr, "q36-test: failed to create session for %s\n", cfg->case_id);
        goto done;
    }
    if (q36_session_sync(session, &prompt, err, sizeof(err)) != 0) {
        fprintf(stderr, "q36-test: sync failed for %s: %s\n", cfg->case_id, err);
        goto done;
    }
    out = test_open_output(cfg->out_path);
    if (!out) {
        fprintf(stderr, "q36-test: failed to open %s\n", cfg->out_path);
        goto done;
    }

    fprintf(out, "case\t%s\n", cfg->case_id);
    fprintf(out, "prompt_tokens\t%d\n", prompt.len);
    for (int step = 0; step < cfg->tokens; step++) {
        enum { TEST_MAX_TOPK = 64 };
        q36_token_score scores[TEST_MAX_TOPK];
        int k = cfg->top_k;
        int token;
        int nscore;
        char *selected_text;
        if (k > TEST_MAX_TOPK) k = TEST_MAX_TOPK;
        token = q36_session_argmax(session);
        if (token < 0) break;
        nscore = q36_session_top_logprobs(session, scores, k);
        selected_text = test_token_repr(engine, token);
        fprintf(out, "selected\t%d\t%d\t%s\n", step, token, selected_text);
        free(selected_text);
        for (int i = 0; i < nscore; i++) {
            char *score_text = test_token_repr(engine, scores[i].id);
            fprintf(out, "score\t%d\t%d\t%d\t%.9g\t%s\n",
                    step, i, scores[i].id, scores[i].logit, score_text);
            free(score_text);
        }
        if (q36_session_eval(session, token, err, sizeof(err)) != 0) break;
    }
    if (out != stdout) fclose(out);
    out = NULL;
    rc = 0;

done:
    if (out && out != stdout) fclose(out);
    q36_tokens_free(&prompt);
    q36_session_free(session);
    q36_engine_close(engine);
    free(prompt_text);
    return rc;
}

static int test_dump_chat_prompt_main(int argc, char **argv) {
    const char *model_path = test_model_path();
    const char *prompt_path = NULL;
    bool hf_template = false;
    q36_engine *engine = NULL;
    q36_tokens prompt = {0};
    char *prompt_text = NULL;
    int rc = 1;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--prompt-file") && i + 1 < argc) prompt_path = argv[++i];
        else if (!strcmp(argv[i], "--hf-template")) hf_template = true;
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            puts("usage: q36_test --dump-chat-prompt --prompt-file FILE [--model FILE] [--hf-template]");
            return 0;
        } else {
            fprintf(stderr, "q36-test: unknown option for --dump-chat-prompt: %s\n", argv[i]);
            return 2;
        }
    }
    if (!prompt_path) {
        fprintf(stderr, "q36-test: --dump-chat-prompt needs --prompt-file FILE\n");
        return 2;
    }
    prompt_text = test_read_file(prompt_path);
    if (!prompt_text) {
        fprintf(stderr, "q36-test: failed to read %s\n", prompt_path);
        goto done;
    }
    engine = test_open_runtime_engine(model_path, 0);
    if (!engine) goto done;
    test_encode_prompt_mode(engine, "", prompt_text, Q36_THINK_NONE, hf_template, &prompt);
    for (int i = 0; i < prompt.len; i++) {
        if (i) putchar(' ');
        printf("%d", prompt.v[i]);
    }
    putchar('\n');
    rc = 0;

done:
    free(prompt_text);
    q36_tokens_free(&prompt);
    q36_engine_close(engine);
    return rc;
}

static bool test_vector_case_selected(const char *id) {
    const char *filter = test_case_override ? test_case_override : getenv("Q36_TEST_VECTOR_CASE");
    char buf[256];
    if (!filter || !filter[0]) return true;
    snprintf(buf, sizeof(buf), "%s", filter);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        while (*tok && isspace((unsigned char)*tok)) tok++;
        if (*tok && strstr(id, tok)) return true;
    }
    return false;
}

static int test_dump_case_main(int argc, char **argv) {
    test_dump_config cfg = {
        .model_path = test_model_path(),
        .prompt_file = "gguf-tools/quality-testing/prompts.jsonl",
        .out_path = "-",
        .tokens = 4,
        .ctx = 32768,
        .top_k = 20,
        .threads = 1,
    };
    if (argc >= 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
        printf("usage: %s --dump-logprob-case CASE_ID [--model FILE] [--prompts FILE] [--out FILE] [--tokens N] [--ctx N] [--top-k N] [--threads N] [--hf-template]\n", "q36_test");
        return 0;
    }
    if (argc < 2) {
        fprintf(stderr, "usage: %s --dump-logprob-case CASE_ID [--model FILE] [--prompts FILE] [--out FILE] [--tokens N] [--ctx N] [--top-k N] [--threads N] [--hf-template]\n", argv[0]);
        return 2;
    }
    cfg.case_id = argv[1];
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--model")) cfg.model_path = argv[++i];
        else if (!strcmp(argv[i], "--prompts")) cfg.prompt_file = argv[++i];
        else if (!strcmp(argv[i], "--out")) cfg.out_path = argv[++i];
        else if (!strcmp(argv[i], "--tokens")) cfg.tokens = test_parse_positive_int(argc, argv, &i, argv[i]);
        else if (!strcmp(argv[i], "--ctx")) cfg.ctx = test_parse_positive_int(argc, argv, &i, argv[i]);
        else if (!strcmp(argv[i], "--top-k")) cfg.top_k = test_parse_positive_int(argc, argv, &i, argv[i]);
        else if (!strcmp(argv[i], "--threads")) cfg.threads = test_parse_positive_int(argc, argv, &i, argv[i]);
        else if (!strcmp(argv[i], "--hf-template")) cfg.hf_template = true;
        else {
            fprintf(stderr, "q36-test: unknown dump option: %s\n", argv[i]);
            return 2;
        }
    }
    return test_write_case_dump(&cfg);
}

static const char *test_format_metric(double v, char *buf, size_t cap) {
    if (v < 0.0 || !isfinite(v)) return "n/a";
    snprintf(buf, cap, "%.6g", v);
    return buf;
}

static double test_probe_maxdiff(const float *a, const float *b, uint64_t n) {
    double m = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}

static double test_probe_rms(const float *a, const float *b, uint64_t n) {
    double s = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        s += d * d;
    }
    return sqrt(s / (double)n);
}

static int test_probe_kl(const float *a, const float *b, int n, double *kl_out) {
    float mv_a = -FLT_MAX, mv_b = -FLT_MAX;
    double sa = 0.0, sb = 0.0, kl = 0.0;
    double lse_a, lse_b;
    for (int i = 0; i < n; i++) {
        if (isfinite(a[i]) && a[i] > mv_a) mv_a = a[i];
        if (isfinite(b[i]) && b[i] > mv_b) mv_b = b[i];
    }
    if (!isfinite(mv_a) || !isfinite(mv_b)) return -1;
    for (int i = 0; i < n; i++) {
        if (isfinite(a[i])) sa += exp((double)a[i] - (double)mv_a);
        if (isfinite(b[i])) sb += exp((double)b[i] - (double)mv_b);
    }
    lse_a = (double)mv_a + log(sa);
    lse_b = (double)mv_b + log(sb);
    for (int i = 0; i < n; i++) {
        if (isfinite(a[i]) && isfinite(b[i])) {
            double p = exp((double)a[i] - lse_a);
            kl += p * ((double)a[i] - lse_a - ((double)b[i] - lse_b));
        }
    }
    *kl_out = kl;
    return 0;
}

static void test_print_json_str(FILE *f, const char *s) {
    putc('"', f);
    for (; *s; s++) {
        if (*s == '\\') fputs("\\\\", f);
        else if (*s == '"') fputs("\\\"", f);
        else if (*s == '\n') fputs("\\n", f);
        else putc(*s, f);
    }
    putc('"', f);
}

static q36_engine *test_open_runtime_engine(const char *model_path, int threads) {
    q36_engine *engine = NULL;
    const char *env = getenv("Q36_TEST_THREADS");
    if (threads <= 0 && env && env[0]) threads = atoi(env);
    q36_engine_options opt = {
        .model_path = model_path,
        .backend = Q36_BACKEND_CPU,
        .n_threads = threads,
    };
    if (q36_engine_open(&engine, &opt) != 0) return NULL;
    return engine;
}

static q36_engine *test_open_backend_engine(const char *model_path,
                                            q36_backend backend,
                                            int threads) {
    q36_engine *engine = NULL;
    const char *env = getenv("Q36_TEST_THREADS");
    const char *stream = getenv("Q36_TEST_SSD_STREAMING");
    const char *cache_k = getenv("Q36_TEST_CACHE_TYPE_K");
    const char *cache_v = getenv("Q36_TEST_CACHE_TYPE_V");
    if (threads <= 0 && env && env[0]) threads = atoi(env);
    q36_engine_options opt = {
        .model_path = model_path,
        .backend = backend,
        .n_threads = threads,
    };
    if (backend == Q36_BACKEND_VULKAN)
        opt.distributed = test_parity_distributed;
    /* Keep the streaming override for checking the same Vulkan path against
     * a captured CPU trajectory on machines that cannot use residency. */
    if (backend == Q36_BACKEND_VULKAN && stream && stream[0] && atoi(stream)) {
        const char *cache_mb = getenv("Q36_TEST_SSD_STREAM_CACHE_MB");
        const char *full_layers = getenv("Q36_TEST_SSD_FULL_LAYERS");
        opt.ssd_streaming = true;
        if (full_layers && full_layers[0]) {
            opt.ssd_streaming_full_layers = (uint32_t)atoi(full_layers);
            opt.ssd_streaming_full_layers_set = true;
        }
        if (cache_mb && cache_mb[0] && atoi(cache_mb) > 0)
            opt.ssd_streaming_cache_bytes = (uint64_t)atoi(cache_mb) << 20;
    }
    if (cache_k && cache_k[0]) {
        if (!q36_parse_kv_cache_type(cache_k, &opt.cache_type_k)) return NULL;
    } else {
        opt.cache_type_k = q36_default_kv_cache_type_k(backend, opt.ssd_streaming);
    }
    if (cache_v && cache_v[0]) {
        if (!q36_parse_kv_cache_type(cache_v, &opt.cache_type_v)) return NULL;
    } else {
        opt.cache_type_v = q36_default_kv_cache_type_v(backend, opt.ssd_streaming);
    }
    if (q36_engine_open(&engine, &opt) != 0) return NULL;
    return engine;
}

static int test_layer_probe_main(int argc, char **argv) {
    const char *model_path = test_model_path();
    const char *prompt_file = "gguf-tools/quality-testing/prompts.jsonl";
    const char *case_id = NULL;
    const char *json_out = NULL;
    bool hf_template = false;
    int ctx_size = 32768;
    int threads = 1;
    q36_engine *q36_e = NULL;
    q36_engine *llama_e = NULL;
    q36_tokens prompt = {0};
    char *prompt_text = NULL;
    char err[160] = {0};
    float *q36_hidden = NULL;
    float *llama_hidden = NULL;
    float *q36_logits = NULL;
    float *llama_logits = NULL;
    FILE *out = stdout;
    int rc = 1;
    int n_layer = q36_qwen35_n_layer();
    int n_embd = q36_qwen35_n_embd();
    int n_vocab = q36_qwen35_n_vocab();
    struct timeval tv0, tv1;
    int n_warn = 0;
    int n_diverge = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model")) model_path = argv[++i];
        else if (!strcmp(argv[i], "--prompts")) prompt_file = argv[++i];
        else if (!strcmp(argv[i], "--json")) json_out = argv[++i];
        else if (!strcmp(argv[i], "--ctx")) ctx_size = test_parse_positive_int(argc, argv, &i, argv[i]);
        else if (!strcmp(argv[i], "--threads")) threads = test_parse_positive_int(argc, argv, &i, argv[i]);
        else if (!strcmp(argv[i], "--hf-template")) hf_template = true;
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            puts("usage: q36_test --layer-probe CASE_ID [--model FILE] [--prompts FILE] [--json FILE] [--ctx N] [--threads N] [--hf-template]");
            return 0;
        } else if (!case_id) case_id = argv[i];
        else {
            fprintf(stderr, "q36-test: unknown layer-probe option: %s\n", argv[i]);
            return 2;
        }
    }
    if (!case_id) {
        fprintf(stderr, "q36-test: missing CASE_ID for --layer-probe\n");
        return 2;
    }

    prompt_text = test_load_prompt_text(prompt_file, case_id);
    if (!prompt_text) {
        fprintf(stderr, "q36-test: prompt %s not found in %s\n", case_id, prompt_file);
        goto done;
    }
    q36_e = test_open_runtime_engine(model_path, threads);
    llama_e = test_open_runtime_engine(model_path, threads);
    if (!q36_e || !llama_e) {
        fprintf(stderr, "q36-test: failed to open engines for layer probe\n");
        goto done;
    }
    test_encode_prompt_mode(q36_e, "", prompt_text, Q36_THINK_NONE, hf_template, &prompt);
    if (prompt.len <= 0) {
        fprintf(stderr, "q36-test: empty prompt for %s\n", case_id);
        goto done;
    }

    q36_hidden = xmalloc((uint64_t)n_layer * n_embd * sizeof(float));
    llama_hidden = xmalloc((uint64_t)n_layer * n_embd * sizeof(float));
    q36_logits = xmalloc((uint64_t)n_vocab * sizeof(float));
    llama_logits = xmalloc((uint64_t)n_vocab * sizeof(float));
    gettimeofday(&tv0, NULL);
    if (q36_compare_prefill_layers(q36_e, llama_e, &prompt, ctx_size,
                                   q36_hidden, llama_hidden,
                                   q36_logits, llama_logits,
                                   err, sizeof(err)) != 0) {
        fprintf(stderr, "q36-test: layer probe failed for %s: %s\n", case_id, err);
        goto done;
    }

    if (json_out) {
        out = fopen(json_out, "wb");
        if (!out) {
            fprintf(stderr, "q36-test: failed to open %s\n", json_out);
            goto done;
        }
    }

    fprintf(out, "{\n  \"case\": ");
    test_print_json_str(out, case_id);
    fprintf(out, ",\n  \"prompt_tokens\": %d,\n", prompt.len);
    fprintf(out, "  \"layers\": [\n");
    for (int il = 0; il < n_layer; il++) {
        const float *a = q36_hidden + (uint64_t)il * n_embd;
        const float *b = llama_hidden + (uint64_t)il * n_embd;
        double md = test_probe_maxdiff(a, b, (uint64_t)n_embd);
        double rs = test_probe_rms(a, b, (uint64_t)n_embd);
        const char *status = "ok";
        if (md >= 1.0) { status = "diverge"; n_diverge++; }
        else if (md >= 0.1) { status = "warn"; n_warn++; }
        else if (md >= 0.01) status = "near";
        if (il) fputc(',', out);
        fprintf(out, "    {\"layer\":%d,\"maxdiff\":%.9g,\"rms\":%.9g,\"status\":\"%s\"}",
                il, md, rs, status);
        fprintf(stderr, "  %3d  maxdiff=%.6g  rms=%.6g  %s\n", il, md, rs, status);
    }
    {
        double kl = 0.0;
        int qt = test_sample_argmax_local(q36_logits, n_vocab);
        int lt = test_sample_argmax_local(llama_logits, n_vocab);
        double logit_md = test_probe_maxdiff(q36_logits, llama_logits, (uint64_t)n_vocab);
        double logit_rs = test_probe_rms(q36_logits, llama_logits, (uint64_t)n_vocab);
        int kl_ok = (test_probe_kl(q36_logits, llama_logits, n_vocab, &kl) == 0);
        fprintf(out, "\n  ],\n");
        fprintf(out, "  \"final_logits\": {\n");
        fprintf(out, "    \"maxdiff\": %.9g,\n", logit_md);
        fprintf(out, "    \"rms\": %.9g,\n", logit_rs);
        fprintf(out, "    \"kl\": %s,\n", kl_ok ? "true" : "false");
        if (kl_ok) fprintf(out, "    \"kl_value\": %.9g,\n", kl);
        fprintf(out, "    \"q36_top1\": %d,\n", qt);
        fprintf(out, "    \"llama_top1\": %d,\n", lt);
        fprintf(out, "    \"match\": %s\n", qt == lt ? "true" : "false");
        fprintf(out, "  },\n");
        fprintf(out, "  \"summary\": {\n");
        fprintf(out, "    \"n_warn\": %d,\n", n_warn);
        fprintf(out, "    \"n_diverge\": %d\n", n_diverge);
        fprintf(out, "  }\n}\n");
        fprintf(stderr, "\n=== Final logits: q36=%d llama=%d match=%s ===\n",
                qt, lt, qt == lt ? "YES" : "NO");
        fprintf(stderr, "maxdiff=%.9g  rms=%.9g  KL=%.9g\n", logit_md, logit_rs, kl_ok ? kl : -1.0);
        int64_t elapsed_ms;
        gettimeofday(&tv1, NULL);
        elapsed_ms = ((int64_t)tv1.tv_sec - (int64_t)tv0.tv_sec) * 1000 +
                     ((int64_t)tv1.tv_usec - (int64_t)tv0.tv_usec) / 1000;
        fprintf(stderr, "warn=%d  diverge=%d  time=%" PRId64 "ms\n",
                n_warn, n_diverge, elapsed_ms);
    }
    rc = 0;

done:
    if (out != stdout) fclose(out);
    free(q36_hidden);
    free(llama_hidden);
    free(q36_logits);
    free(llama_logits);
    q36_tokens_free(&prompt);
    q36_engine_close(q36_e);
    q36_engine_close(llama_e);
    free(prompt_text);
    return rc;
}

static void *parity_xrealloc(void *p, size_t n) {
    void *q = realloc(p, n);
    if (!q) {
        fprintf(stderr, "parity: out of memory\n");
        exit(1);
    }
    return q;
}

static char *parity_strdup(const char *s) {
    size_t n = strlen(s);
    char *out = xmalloc(n + 1);
    memcpy(out, s, n + 1);
    return out;
}

static void parity_dvec_push(parity_dvec *dv, double x) {
    if (dv->len == dv->cap) {
        dv->cap = dv->cap ? dv->cap * 2 : 64;
        dv->v = parity_xrealloc(dv->v, (size_t)dv->cap * sizeof(dv->v[0]));
    }
    dv->v[dv->len++] = x;
}

static int parity_double_cmp(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);
}

static double parity_percentile(const parity_dvec *dv, double p) {
    double *tmp;
    double out;
    int idx;
    if (!dv || dv->len <= 0) return 0.0;
    tmp = xmalloc((size_t)dv->len * sizeof(tmp[0]));
    memcpy(tmp, dv->v, (size_t)dv->len * sizeof(tmp[0]));
    qsort(tmp, (size_t)dv->len, sizeof(tmp[0]), parity_double_cmp);
    if (dv->len == 1) idx = 0;
    else {
        double pos = p * (double)(dv->len - 1);
        idx = (int)(pos + 0.5);
        if (idx < 0) idx = 0;
        if (idx >= dv->len) idx = dv->len - 1;
    }
    out = tmp[idx];
    free(tmp);
    return out;
}

static double parity_median(const parity_dvec *dv) {
    return parity_percentile(dv, 0.5);
}

static void parity_dvec_free(parity_dvec *dv) {
    free(dv->v);
    memset(dv, 0, sizeof(*dv));
}

static char *parity_read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    char *buf;
    long n;
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    n = ftell(fp);
    if (n < 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    buf = xmalloc((size_t)n + 1);
    if (n && fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        free(buf);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    buf[n] = '\0';
    return buf;
}

static void parity_prompt_vec_push(parity_prompt_vec *pv, const char *id, const char *cat, char *prompt) {
    if (pv->len == pv->cap) {
        pv->cap = pv->cap ? pv->cap * 2 : 128;
        pv->v = parity_xrealloc(pv->v, (size_t)pv->cap * sizeof(pv->v[0]));
    }
    pv->v[pv->len].id = parity_strdup(id);
    pv->v[pv->len].category = parity_strdup(cat);
    pv->v[pv->len].prompt = prompt;
    pv->len++;
}

static void parity_prompt_vec_free(parity_prompt_vec *pv) {
    for (int i = 0; i < pv->len; i++) {
        free(pv->v[i].id);
        free(pv->v[i].category);
        free(pv->v[i].prompt);
    }
    free(pv->v);
    memset(pv, 0, sizeof(*pv));
}

static char *parity_json_string(const char *line, const char *key) {
    return test_json_string_value(line, key);
}

static int parity_case_number(const char *id) {
    const char *p = strrchr(id, '_');
    if (!p) return -1;
    return atoi(p + 1);
}

static const char *parity_quality_category(const char *id) {
    int n = parity_case_number(id);
    if (n >= 0 && ((n / 10) % 2) == 1) return "quality/it";
    return "quality/en";
}

static bool parity_load_quality_prompts(parity_prompt_vec *pv, const char *path) {
    FILE *fp = fopen(path, "rb");
    char line[8192];
    if (!fp) return false;
    while (fgets(line, sizeof(line), fp)) {
        char *id = parity_json_string(line, "id");
        char *prompt = parity_json_string(line, "prompt");
        if (!id || !prompt) {
            free(id);
            free(prompt);
            fclose(fp);
            return false;
        }
        parity_prompt_vec_push(pv, id, parity_quality_category(id), prompt);
        free(id);
    }
    fclose(fp);
    return true;
}

static bool parity_load_test_prompt(parity_prompt_vec *pv, const char *path, const char *id, const char *cat) {
    char *prompt = parity_read_file(path);
    if (!prompt) return false;
    parity_prompt_vec_push(pv, id, cat, prompt);
    return true;
}

static int parity_parse_int_arg(int argc, char **argv, int *i, const char *name) {
    return test_parse_positive_int(argc, argv, i, name);
}

static parity_config parity_parse_args(int argc, char **argv) {
    parity_config cfg = {
        .model_path = test_model_path(),
        .report_path = "parity-report.md",
        .steps = 8,
        .ctx_size = 32768,
        .threads = 0,
        .quality_limit = 100,
        .skip_long = 0,
        .hf_template = 0,
    };
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") || !strcmp(argv[i], "-m")) cfg.model_path = argv[++i];
        else if (!strcmp(argv[i], "--out")) cfg.report_path = argv[++i];
        else if (!strcmp(argv[i], "--steps")) cfg.steps = parity_parse_int_arg(argc, argv, &i, argv[i]);
        else if (!strcmp(argv[i], "--ctx")) cfg.ctx_size = parity_parse_int_arg(argc, argv, &i, argv[i]);
        else if (!strcmp(argv[i], "--threads") || !strcmp(argv[i], "-t")) cfg.threads = parity_parse_int_arg(argc, argv, &i, argv[i]);
        else if (!strcmp(argv[i], "--quality-limit")) cfg.quality_limit = parity_parse_int_arg(argc, argv, &i, argv[i]);
        else if (!strcmp(argv[i], "--skip-long")) cfg.skip_long = 1;
        else if (!strcmp(argv[i], "--hf-template")) cfg.hf_template = 1;
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            puts("usage: q36_test --parity-report [--model FILE] [--out FILE] [--steps N] [--ctx N] [--threads N] [--quality-limit N] [--skip-long] [--hf-template]");
            exit(0);
        } else {
            fprintf(stderr, "parity: unknown option: %s\n", argv[i]);
            exit(2);
        }
    }
    if (cfg.steps > 64) cfg.steps = 64;
    return cfg;
}

static void parity_topk(const float *logits, int n, int k, int *ids, float *vals) {
    for (int i = 0; i < k; i++) {
        ids[i] = -1;
        vals[i] = Q36_NEG_INF;
    }
    for (int i = 0; i < n; i++) {
        float v = logits[i];
        if (!isfinite(v)) continue;
        for (int j = 0; j < k; j++) {
            if (ids[j] < 0 || v > vals[j]) {
                for (int l = k - 1; l > j; l--) {
                    ids[l] = ids[l - 1];
                    vals[l] = vals[l - 1];
                }
                ids[j] = i;
                vals[j] = v;
                break;
            }
        }
    }
}

static double parity_logsumexp(const float *x, int n) {
    float maxv = Q36_NEG_INF;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        if (isfinite(x[i]) && x[i] > maxv) maxv = x[i];
    }
    if (!isfinite(maxv)) return -INFINITY;
    for (int i = 0; i < n; i++) {
        if (isfinite(x[i])) sum += exp((double)x[i] - (double)maxv);
    }
    return (double)maxv + log(sum);
}

static void parity_append_char(char *out, size_t cap, size_t *n, char c) {
    if (*n + 1 < cap) out[(*n)++] = c;
}

static void parity_append_str(char *out, size_t cap, size_t *n, const char *s) {
    while (*s) parity_append_char(out, cap, n, *s++);
}

static void parity_token_repr_class(q36_engine *e, int token, char *repr, size_t rcap, char *cls, size_t ccap) {
    size_t len = 0;
    char *text = q36_token_text(e, token, &len);
    size_t n = 0;
    bool any_non_ascii = false;
    bool any_alpha = false;
    bool any_digit = false;
    bool any_space = false;
    bool any_control = false;
    bool all_space = len > 0;
    bool all_punct = len > 0;
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        if (c >= 128) any_non_ascii = true;
        if (isalpha(c)) any_alpha = true;
        if (isdigit(c)) any_digit = true;
        if (isspace(c)) any_space = true;
        if (c < 32 && !isspace(c)) any_control = true;
        if (!isspace(c)) all_space = false;
        if (!(ispunct(c) || isspace(c))) all_punct = false;
        if (c == '\n') parity_append_str(repr, rcap, &n, "\\n");
        else if (c == '\r') parity_append_str(repr, rcap, &n, "\\r");
        else if (c == '\t') parity_append_str(repr, rcap, &n, "\\t");
        else if (c == ' ') parity_append_str(repr, rcap, &n, "\\x20");
        else if (c == '\\') parity_append_str(repr, rcap, &n, "\\\\");
        else if (c == '`') parity_append_str(repr, rcap, &n, "\\x60");
        else if (c == '|') parity_append_str(repr, rcap, &n, "\\x7c");
        else if (c < 32 || c >= 127) {
            char buf[8];
            snprintf(buf, sizeof(buf), "\\x%02x", c);
            parity_append_str(repr, rcap, &n, buf);
        } else parity_append_char(repr, rcap, &n, (char)c);
    }
    if (rcap) repr[n < rcap ? n : rcap - 1] = '\0';
    if (len == 0) snprintf(cls, ccap, "empty");
    else if (text[0] == '<') snprintf(cls, ccap, "special");
    else if (all_space) snprintf(cls, ccap, any_space ? "whitespace" : "empty");
    else if (any_control) snprintf(cls, ccap, "control");
    else if (any_non_ascii) snprintf(cls, ccap, "non-ascii");
    else if (any_alpha) snprintf(cls, ccap, "ascii-text");
    else if (any_digit) snprintf(cls, ccap, "number");
    else if (all_punct) snprintf(cls, ccap, "punctuation");
    else snprintf(cls, ccap, "other");
    free(text);
}

static double parity_compare_logits(q36_engine *text_engine,
                                    const float *q36_logits,
                                    const float *llama_logits,
                                    int n_vocab,
                                    int *q36_top1,
                                    int *llama_top1,
                                    double *top5_overlap,
                                    double *kl_q36_llama,
                                    double *maxdiff_out,
                                    parity_token_diff *worst,
                                    int worst_cap) {
    int ntop[5], ltop[5];
    float nvals[5], lvals[5];
    double n_lse, l_lse, kl = 0.0;
    float maxdiff = 0.0f;
    int overlap = 0;
    parity_topk(q36_logits, n_vocab, 5, ntop, nvals);
    parity_topk(llama_logits, n_vocab, 5, ltop, lvals);
    *q36_top1 = ntop[0];
    *llama_top1 = ltop[0];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            if (ntop[i] >= 0 && ntop[i] == ltop[j]) {
                overlap++;
                break;
            }
        }
    }
    *top5_overlap = (double)overlap / 5.0;
    n_lse = parity_logsumexp(q36_logits, n_vocab);
    l_lse = parity_logsumexp(llama_logits, n_vocab);
    for (int i = 0; i < worst_cap; i++) {
        worst[i].id = -1;
        worst[i].diff = -1.0f;
    }
    for (int i = 0; i < n_vocab; i++) {
        float nl = q36_logits[i];
        float ll = llama_logits[i];
        float d = fabsf(nl - ll);
        if (isfinite(nl) && isfinite(ll)) {
            double lp = (double)nl - n_lse;
            double lq = (double)ll - l_lse;
            double p = exp(lp);
            kl += p * (lp - lq);
        } else if (isfinite(nl) && !isfinite(ll)) kl = INFINITY;
        if (d > maxdiff) maxdiff = d;
        for (int j = 0; j < worst_cap; j++) {
            if (worst[j].id < 0 || d > worst[j].diff) {
                for (int k = worst_cap - 1; k > j; k--) worst[k] = worst[k - 1];
                worst[j].id = i;
                worst[j].q36_logit = nl;
                worst[j].llama_logit = ll;
                worst[j].diff = d;
                break;
            }
        }
    }
    for (int i = 0; i < worst_cap; i++) {
        if (worst[i].id >= 0) parity_token_repr_class(text_engine, worst[i].id,
                                                      worst[i].text, sizeof(worst[i].text),
                                                      worst[i].cls, sizeof(worst[i].cls));
    }
    *kl_q36_llama = kl;
    *maxdiff_out = maxdiff;
    return maxdiff;
}

static parity_category *parity_get_category(parity_category *cats, int *ncat, const char *name) {
    for (int i = 0; i < *ncat; i++) if (!strcmp(cats[i].name, name)) return &cats[i];
    memset(&cats[*ncat], 0, sizeof(cats[*ncat]));
    snprintf(cats[*ncat].name, sizeof(cats[*ncat].name), "%s", name);
    return &cats[(*ncat)++];
}

static void parity_add_metrics(parity_dvec *global_top5, parity_dvec *global_kl, parity_dvec *global_maxdiff,
                               parity_category *cat, parity_dvec *pos_top5, parity_dvec *pos_kl,
                               parity_dvec *pos_maxdiff, int step,
                               double top5, double kl, double maxdiff) {
    parity_dvec_push(global_top5, top5);
    parity_dvec_push(global_kl, kl);
    parity_dvec_push(global_maxdiff, maxdiff);
    parity_dvec_push(&cat->top5, top5);
    parity_dvec_push(&cat->kl, kl);
    parity_dvec_push(&cat->maxdiff, maxdiff);
    parity_dvec_push(&pos_top5[step], top5);
    parity_dvec_push(&pos_kl[step], kl);
    parity_dvec_push(&pos_maxdiff[step], maxdiff);
}

static int parity_worst_cmp(const void *a, const void *b) {
    const parity_prompt_result *ra = *(const parity_prompt_result * const *)a;
    const parity_prompt_result *rb = *(const parity_prompt_result * const *)b;
    int ma = ra->positions - ra->top1_matches;
    int mb = rb->positions - rb->top1_matches;
    if (ma != mb) return mb - ma;
    if (ra->maxdiff_p99 < rb->maxdiff_p99) return 1;
    if (ra->maxdiff_p99 > rb->maxdiff_p99) return -1;
    if (ra->kl_p99 < rb->kl_p99) return 1;
    if (ra->kl_p99 > rb->kl_p99) return -1;
    return 0;
}

static void parity_write_status(FILE *fp, bool pass) {
    fprintf(fp, "%s", pass ? "PASS" : "FAIL");
}

static void parity_write_report(const parity_config *cfg,
                                const parity_prompt_vec *prompts,
                                parity_prompt_result *results,
                                parity_category *cats,
                                int ncat,
                                parity_dvec *pos_top5,
                                parity_dvec *pos_kl,
                                parity_dvec *pos_maxdiff,
                                int *pos_seen,
                                int *pos_match,
                                int total_positions,
                                int total_top1_matches,
                                const parity_dvec *global_top5,
                                const parity_dvec *global_kl,
                                const parity_dvec *global_maxdiff) {
    (void)pos_top5;
    FILE *fp = fopen(cfg->report_path, "wb");
    double top1_rate, top5_med, kl_med, kl_p99, max_p50, max_p95, max_p99, plot_max = 0.0;
    int flagged = 0;
    if (!fp) {
        fprintf(stderr, "parity: failed to write %s\n", cfg->report_path);
        return;
    }
    top1_rate = total_positions ? (double)total_top1_matches / (double)total_positions : 0.0;
    top5_med = parity_median(global_top5);
    kl_med = parity_median(global_kl);
    kl_p99 = parity_percentile(global_kl, 0.99);
    max_p50 = parity_percentile(global_maxdiff, 0.50);
    max_p95 = parity_percentile(global_maxdiff, 0.95);
    max_p99 = parity_percentile(global_maxdiff, 0.99);
    for (int i = 0; i < prompts->len; i++) if (results[i].maxdiff_p99 > 10.0) flagged++;
    {
        bool c1 = top1_rate >= 0.99;
        bool c2 = top5_med >= 0.95;
        bool c3 = kl_med < 5e-3 && kl_p99 < 2e-2;
        bool c4 = flagged == 0;
        bool all = c1 && c2 && c3 && c4;
        time_t now = time(NULL);
        struct tm tmv;
        char tbuf[64];
        localtime_r(&now, &tmv);
        strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S %Z", &tmv);
        fprintf(fp, "# q36 CPU vs llama.cpp Parity Report\n\n");
        fprintf(fp, "Top-line result: **%s**.\n\n", all ? "PASS" : "FAIL");
        fprintf(fp, "Generated: %s\n\n", tbuf);
        fprintf(fp, "Model: `%s`\n\n", cfg->model_path);
        fprintf(fp, "Corpus: `%d` prompts from `gguf-tools/quality-testing/prompts.jsonl` plus `%d` prompts from `tests/test-vectors/qwen3.6-35b-a3b/prompts`.\n\n",
                prompts->len - (cfg->skip_long ? 3 : 5), cfg->skip_long ? 3 : 5);
        fprintf(fp, "Run mode: empty system prompt, no-thinking chat rendering, greedy q36-token trajectory, max `%d` generated positions per prompt, ctx `%d`.\n\n", cfg->steps, cfg->ctx_size);
        fprintf(fp, "## Pass Criteria\n\n");
        fprintf(fp, "| # | Criterion | Threshold | Result | Status |\n|---:|---|---:|---:|---|\n");
        fprintf(fp, "| 1 | Greedy top-1 match rate | >= 99%% | %.3f%% (%d/%d) | ", top1_rate * 100.0, total_top1_matches, total_positions);
        parity_write_status(fp, c1); fprintf(fp, " |\n");
        fprintf(fp, "| 2 | Top-5 set overlap, median across positions | >= 95%% | %.1f%% | ", top5_med * 100.0);
        parity_write_status(fp, c2); fprintf(fp, " |\n");
        fprintf(fp, "| 3 | KL q36||llama | median < 5e-3, p99 < 2e-2 | median %.9g, p99 %.9g | ", kl_med, kl_p99);
        parity_write_status(fp, c3); fprintf(fp, " |\n");
        fprintf(fp, "| 4 | Max logit diff distribution | report p50/p95/p99; flag prompt p99 > 10 | p50 %.6g, p95 %.6g, p99 %.6g; flagged %d | ", max_p50, max_p95, max_p99, flagged);
        parity_write_status(fp, c4); fprintf(fp, " |\n\n");
        fprintf(fp, "## Category Breakdown\n\n");
        fprintf(fp, "| Category | Prompts | Positions | Top-1 match | Top-5 median | KL median | KL p99 | Max diff p99 | p99>10 prompts |\n");
        fprintf(fp, "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
        for (int i = 0; i < ncat; i++) {
            parity_category *c = &cats[i];
            double rate = c->positions ? (double)c->top1_matches / (double)c->positions : 0.0;
            fprintf(fp, "| %s | %d | %d | %.3f%% | %.1f%% | %.9g | %.9g | %.6g | %d |\n",
                    c->name, c->cases, c->positions, rate * 100.0,
                    parity_median(&c->top5) * 100.0,
                    parity_median(&c->kl), parity_percentile(&c->kl, 0.99),
                    parity_percentile(&c->maxdiff, 0.99), c->flagged_prompts);
        }
        fprintf(fp, "\n## Position-Index Drift Plot\n\nEach bar is scaled by per-position p99 max-logit-diff.\n\n```text\n");
        for (int i = 0; i < cfg->steps; i++) {
            double v = parity_percentile(&pos_maxdiff[i], 0.99);
            if (v > plot_max) plot_max = v;
        }
        if (plot_max <= 0.0) plot_max = 1.0;
        fprintf(fp, "step  seen  top1%%   KL_med      maxdiff_p99  drift\n");
        for (int i = 0; i < cfg->steps; i++) {
            double rate = pos_seen[i] ? (double)pos_match[i] / (double)pos_seen[i] : 0.0;
            double kmed = parity_median(&pos_kl[i]);
            double md99 = parity_percentile(&pos_maxdiff[i], 0.99);
            int bars = (int)((md99 / plot_max) * 32.0 + 0.5);
            if (bars < 0) bars = 0;
            if (bars > 32) bars = 32;
            fprintf(fp, "%4d  %4d  %6.2f  %.9g  %11.6g  ", i, pos_seen[i], rate * 100.0, kmed, md99);
            for (int b = 0; b < bars; b++) fputc('#', fp);
            fputc('\n', fp);
        }
        fprintf(fp, "```\n\n");
    }
    {
        parity_prompt_result **worst = xmalloc((size_t)prompts->len * sizeof(worst[0]));
        int nworst = prompts->len < 5 ? prompts->len : 5;
        for (int i = 0; i < prompts->len; i++) worst[i] = &results[i];
        qsort(worst, (size_t)prompts->len, sizeof(worst[0]), parity_worst_cmp);
        fprintf(fp, "## Top 5 Worst Prompts\n\n");
        fprintf(fp, "| Rank | Prompt | Category | Prompt tokens | Positions | Top-1 | Divergence point | q36 top1 | llama top1 | KL p99 | Max diff p99 | Worst diff step |\n");
        fprintf(fp, "|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|\n");
        for (int i = 0; i < nworst; i++) {
            parity_prompt_result *r = worst[i];
            char divbuf[32];
            if (r->first_div_step >= 0) snprintf(divbuf, sizeof(divbuf), "step %d", r->first_div_step);
            else snprintf(divbuf, sizeof(divbuf), "none");
            fprintf(fp, "| %d | %s | %s | %d | %d | %d/%d | %s | %d | %d | %.9g | %.6g | %d |\n",
                    i + 1, r->prompt->id, r->prompt->category, r->prompt_tokens,
                    r->positions, r->top1_matches, r->positions, divbuf,
                    r->div_q36, r->div_llama, r->kl_p99, r->maxdiff_p99, r->worst_step);
        }
        fprintf(fp, "\n## Worst-Prompt Token Diffs\n\n");
        for (int i = 0; i < nworst; i++) {
            parity_prompt_result *r = worst[i];
            fprintf(fp, "### %d. %s\n\n", i + 1, r->prompt->id);
            fprintf(fp, "Worst position: generated step `%d`; first divergence: `%s`", r->worst_step,
                    r->first_div_step >= 0 ? "step" : "none");
            if (r->first_div_step >= 0) fprintf(fp, " `%d`", r->first_div_step);
            fprintf(fp, ".\n\n");
            fprintf(fp, "| Token id | Vocab string | Class | q36 logit | llama logit | Abs diff |\n");
            fprintf(fp, "|---:|---|---|---:|---:|---:|\n");
            for (int j = 0; j < r->worst_ntokens; j++) {
                parity_token_diff *d = &r->worst_tokens[j];
                fprintf(fp, "| %d | `%s` | %s | %.9g | %.9g | %.9g |\n",
                        d->id, d->text, d->cls, d->q36_logit, d->llama_logit, d->diff);
            }
            fprintf(fp, "\n");
        }
        free(worst);
    }
    fprintf(fp, "## Notes\n\nThe two runtimes are compared at identical contexts. After each position, both sessions advance with the q36 greedy token, so later rows measure drift on one canonical greedy trajectory instead of mixing two different continuations.\n");
    fclose(fp);
}

static int test_parity_report_main(int argc, char **argv) {
    parity_config cfg = parity_parse_args(argc, argv);
    parity_prompt_vec prompts = {0};
    parity_prompt_result *results = NULL;
    parity_dvec global_top5 = {0}, global_kl = {0}, global_maxdiff = {0};
    parity_dvec *pos_top5 = NULL, *pos_kl = NULL, *pos_maxdiff = NULL;
    int *pos_seen = NULL, *pos_match = NULL;
    parity_category cats[16];
    int ncat = 0;
    int total_positions = 0;
    int total_top1_matches = 0;
    q36_engine *q36_runtime = NULL;
    q36_engine *llama = NULL;
    q36_session *ns = NULL;
    q36_session *ls = NULL;
    int rc = 1;

    memset(cats, 0, sizeof(cats));
    if (!parity_load_quality_prompts(&prompts, "gguf-tools/quality-testing/prompts.jsonl")) {
        fprintf(stderr, "parity: failed to load quality prompts\n");
        goto done;
    }
    if (cfg.quality_limit > 0 && prompts.len > cfg.quality_limit) {
        for (int i = cfg.quality_limit; i < prompts.len; i++) {
            free(prompts.v[i].id);
            free(prompts.v[i].category);
            free(prompts.v[i].prompt);
        }
        prompts.len = cfg.quality_limit;
    }
    if ((!cfg.skip_long &&
         (!parity_load_test_prompt(&prompts, "tests/test-vectors/qwen3.6-35b-a3b/prompts/long_code_audit.txt", "tv_long_code_audit", "test-vectors/long") ||
          !parity_load_test_prompt(&prompts, "tests/test-vectors/qwen3.6-35b-a3b/prompts/long_memory_archive.txt", "tv_long_memory_archive", "test-vectors/long"))) ||
        !parity_load_test_prompt(&prompts, "tests/test-vectors/qwen3.6-35b-a3b/prompts/short_code_completion.txt", "tv_short_code_completion", "test-vectors/short") ||
        !parity_load_test_prompt(&prompts, "tests/test-vectors/qwen3.6-35b-a3b/prompts/short_italian_fact.txt", "tv_short_italian_fact", "test-vectors/short") ||
        !parity_load_test_prompt(&prompts, "tests/test-vectors/qwen3.6-35b-a3b/prompts/short_reasoning_plain.txt", "tv_short_reasoning_plain", "test-vectors/short")) {
        fprintf(stderr, "parity: failed to load test-vector prompts\n");
        goto done;
    }

    q36_runtime = test_open_runtime_engine(cfg.model_path, cfg.threads);
    llama = test_open_runtime_engine(cfg.model_path, cfg.threads);
    if (!q36_runtime || !llama) goto done;
    if (q36_session_create(&ns, q36_runtime, cfg.ctx_size) != 0) {
        fprintf(stderr, "parity: q36 session failed\n");
        goto done;
    }
    if (q36_session_create(&ls, llama, cfg.ctx_size) != 0) {
        fprintf(stderr, "parity: llama session failed\n");
        goto done;
    }

    results = test_xcalloc((size_t)prompts.len, sizeof(results[0]));
    pos_top5 = test_xcalloc((size_t)cfg.steps, sizeof(pos_top5[0]));
    pos_kl = test_xcalloc((size_t)cfg.steps, sizeof(pos_kl[0]));
    pos_maxdiff = test_xcalloc((size_t)cfg.steps, sizeof(pos_maxdiff[0]));
    pos_seen = test_xcalloc((size_t)cfg.steps, sizeof(pos_seen[0]));
    pos_match = test_xcalloc((size_t)cfg.steps, sizeof(pos_match[0]));

    for (int i = 0; i < prompts.len; i++) {
        const parity_prompt *p = &prompts.v[i];
        parity_prompt_result *r = &results[i];
        parity_category *cat = parity_get_category(cats, &ncat, p->category);
        q36_tokens prompt = {0};
        char err[256];

        memset(r, 0, sizeof(*r));
        r->prompt = p;
        r->first_div_step = -1;
        r->div_q36 = -1;
        r->div_llama = -1;
        r->worst_step = -1;
        cat->cases++;

        test_encode_prompt_mode(q36_runtime, "", p->prompt, Q36_THINK_NONE, cfg.hf_template != 0, &prompt);
        r->prompt_tokens = prompt.len;
        if (prompt.len + cfg.steps + 1 >= cfg.ctx_size) {
            q36_tokens_free(&prompt);
            continue;
        }

        q36_session_invalidate(ns);
        q36_session_invalidate(ls);
        if (q36_session_sync(ns, &prompt, err, sizeof(err)) != 0) {
            fprintf(stderr, "parity: q36 sync failed for %s: %s\n", p->id, err);
            q36_tokens_free(&prompt);
            continue;
        }
        if (q36_session_sync(ls, &prompt, err, sizeof(err)) != 0) {
            fprintf(stderr, "parity: llama sync failed for %s: %s\n", p->id, err);
            q36_tokens_free(&prompt);
            continue;
        }

        for (int step = 0; step < cfg.steps; step++) {
            int q36_top1 = -1, llama_top1 = -1;
            double top5 = 0.0, kl = 0.0, maxdiff = 0.0;
            parity_token_diff worst[5];
            const float *qlogits = q36_session_logits(ns, NULL);
            const float *llogits = q36_session_logits(ls, NULL);
            parity_compare_logits(q36_runtime, qlogits, llogits, q36_qwen35_n_vocab(),
                                  &q36_top1, &llama_top1, &top5, &kl, &maxdiff, worst, 5);
            if (q36_top1 != llama_top1 && r->first_div_step < 0) {
                r->first_div_step = step;
                r->div_q36 = q36_top1;
                r->div_llama = llama_top1;
            }
            if (maxdiff > r->worst_maxdiff || r->worst_step < 0) {
                r->worst_maxdiff = maxdiff;
                r->worst_step = step;
                r->worst_ntokens = 5;
                memcpy(r->worst_tokens, worst, sizeof(worst));
            }
            r->positions++;
            if (q36_top1 == llama_top1) r->top1_matches++;
            cat->positions++;
            if (q36_top1 == llama_top1) cat->top1_matches++;
            total_positions++;
            if (q36_top1 == llama_top1) total_top1_matches++;
            pos_seen[step]++;
            if (q36_top1 == llama_top1) pos_match[step]++;
            parity_dvec_push(&r->top5, top5);
            parity_dvec_push(&r->kl, kl);
            parity_dvec_push(&r->maxdiff, maxdiff);
            parity_add_metrics(&global_top5, &global_kl, &global_maxdiff,
                               cat, pos_top5, pos_kl, pos_maxdiff, step,
                               top5, kl, maxdiff);
            if (q36_top1 == q36_token_eos(q36_runtime)) break;
            if (step + 1 < cfg.steps) {
                if (q36_session_eval(ns, q36_top1, err, sizeof(err)) != 0) break;
                if (q36_session_eval(ls, q36_top1, err, sizeof(err)) != 0) break;
            }
        }
        r->top5_median = parity_median(&r->top5);
        r->kl_median = parity_median(&r->kl);
        r->kl_p99 = parity_percentile(&r->kl, 0.99);
        r->maxdiff_p50 = parity_percentile(&r->maxdiff, 0.50);
        r->maxdiff_p95 = parity_percentile(&r->maxdiff, 0.95);
        r->maxdiff_p99 = parity_percentile(&r->maxdiff, 0.99);
        if (r->maxdiff_p99 > 10.0) cat->flagged_prompts++;
        q36_tokens_free(&prompt);
    }

    parity_write_report(&cfg, &prompts, results, cats, ncat,
                        pos_top5, pos_kl, pos_maxdiff, pos_seen, pos_match,
                        total_positions, total_top1_matches,
                        &global_top5, &global_kl, &global_maxdiff);
    fprintf(stderr, "parity: wrote %s\n", cfg.report_path);
    rc = 0;

done:
    if (results) {
        for (int i = 0; i < prompts.len; i++) {
            parity_dvec_free(&results[i].top5);
            parity_dvec_free(&results[i].kl);
            parity_dvec_free(&results[i].maxdiff);
        }
    }
    for (int i = 0; i < ncat; i++) {
        parity_dvec_free(&cats[i].top5);
        parity_dvec_free(&cats[i].kl);
        parity_dvec_free(&cats[i].maxdiff);
    }
    if (pos_top5) for (int i = 0; i < cfg.steps; i++) parity_dvec_free(&pos_top5[i]);
    if (pos_kl) for (int i = 0; i < cfg.steps; i++) parity_dvec_free(&pos_kl[i]);
    if (pos_maxdiff) for (int i = 0; i < cfg.steps; i++) parity_dvec_free(&pos_maxdiff[i]);
    parity_dvec_free(&global_top5);
    parity_dvec_free(&global_kl);
    parity_dvec_free(&global_maxdiff);
    free(pos_top5);
    free(pos_kl);
    free(pos_maxdiff);
    free(pos_seen);
    free(pos_match);
    free(results);
    q36_session_free(ns);
    q36_session_free(ls);
    q36_engine_close(q36_runtime);
    q36_engine_close(llama);
    parity_prompt_vec_free(&prompts);
    return rc;
}

static void test_metal_f16_matvec_fast_nr0_4(void) {
#ifdef Q36_NO_GPU
    test_skip("vulkan-kernels", "CPU-only build");
    return;
#else
    /*
     * This is the short regression for the long-context repetition failure.
     * Decode uses one-token F16 matvecs for several Q36 projections; the fast
     * nr0=4 variant must be numerically equivalent to the plain kernel.
     */
    const uint32_t in_dim = 4096;
    const uint32_t out_dim = 512;
    const uint64_t weight_bytes = (uint64_t)in_dim * out_dim * sizeof(uint16_t);
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint16_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            float w = (float)((int)((o * 3u + i * 5u) % 23u) - 11) / 64.0f;
            weights[(uint64_t)o * in_dim + i] = test_float_to_f16(w);
        }
    }

    q36_gpu_tensor *x = q36_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    q36_gpu_tensor *out = q36_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        q36_gpu_tensor_free(x);
        q36_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)in_dim * sizeof(float));
    float *out_host = malloc((size_t)out_dim * sizeof(float));
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        q36_gpu_tensor_free(x);
        q36_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t i = 0; i < in_dim; i++) {
        x_host[i] = (float)((int)(i % 31u) - 15) / 32.0f;
    }

    TEST_ASSERT(q36_gpu_tensor_write(x, 0, x_host, (uint64_t)in_dim * sizeof(float)) != 0);
    TEST_ASSERT(q36_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    TEST_ASSERT(q36_gpu_matmul_f16_tensor(out, weights_raw, weight_alloc, 0,
                                            in_dim, out_dim, x, 1) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(out, 0, out_host, (uint64_t)out_dim * sizeof(float)) != 0);

    float max_abs = 0.0f;
    for (uint32_t o = 0; o < out_dim; o++) {
        float ref = 0.0f;
        for (uint32_t i = 0; i < in_dim; i++) {
            float w = (float)((int)((o * 3u + i * 5u) % 23u) - 11) / 64.0f;
            ref += w * x_host[i];
        }
        float err = fabsf(out_host[o] - ref);
        if (err > max_abs) max_abs = err;
    }
    TEST_ASSERT(max_abs < 0.02f);

    free(x_host);
    free(out_host);
    q36_gpu_tensor_free(x);
    q36_gpu_tensor_free(out);
    free(weights_raw);
#endif
}

static float test_round_bf16(float value) {
    int exponent;
    if (!isfinite(value) || value == 0.0f) return value;
    frexp((double)value, &exponent);
    int shift = exponent - 8;
    if (shift < -133) shift = -133;
    return (float)ldexp(nearbyint(ldexp((double)value, -shift)), shift);
}

static void test_bf16_matmul(void) {
#ifdef Q36_NO_GPU
    test_skip("bf16-matmul", "CPU-only build");
#else
    /* Exercise scalar decode, verifier widths, and prefill against a double
     * reference. Odd widths also cover the exact shader's packed halfwords. */
    const uint32_t widths[] = {257, 5120}, counts[] = {1, 2, 5, 8, 129};
    const uint32_t out_dim = 48;
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(widths[0]); wi++) {
        uint32_t in_dim = widths[wi];
        size_t weight_bytes = (size_t)in_dim * out_dim * sizeof(uint16_t);
        uint16_t *weights = malloc(weight_bytes);
        float *input = malloc((size_t)129 * in_dim * sizeof(float));
        float *output = malloc((size_t)129 * out_dim * sizeof(float));
        q36_gpu_tensor *x = q36_gpu_tensor_alloc((uint64_t)129 * in_dim * sizeof(float));
        q36_gpu_tensor *y = q36_gpu_tensor_alloc((uint64_t)129 * out_dim * sizeof(float));
        TEST_ASSERT(weights && input && output && x && y);
        if (weights && input && output && x && y) {
            for (uint32_t i = 0; i < in_dim * out_dim; i++) {
                float value = sinf((float)i * 0.17f) * 0.03f;
                uint32_t bits;
                memcpy(&bits, &value, sizeof(bits));
                weights[i] = (uint16_t)(bits >> 16);
            }
            for (uint32_t i = 0; i < 129 * in_dim; i++)
                input[i] = cosf((float)i * 0.13f);
            TEST_ASSERT(q36_gpu_tensor_write(x, 0, input, (uint64_t)129 * in_dim * sizeof(float)));
            for (size_t ci = 0; ci < sizeof(counts) / sizeof(counts[0]); ci++) {
                uint32_t n = counts[ci];
                float scale = ci % 2 ? -0.75f : 1.0f;
                TEST_ASSERT(q36_gpu_matmul_bf16_scaled_tensor(y, weights, weight_bytes, 0,
                    in_dim, out_dim, x, n, scale));
                TEST_ASSERT(q36_gpu_tensor_read(y, 0, output, (uint64_t)n * out_dim * sizeof(float)));
                float max_abs = 0.0f;
                for (uint32_t t = 0; t < n; t++) {
                    for (uint32_t row = 0; row < out_dim; row++) {
                        double sum = 0.0;
                        for (uint32_t i = 0; i < in_dim; i++) {
                            uint32_t bits = (uint32_t)weights[(uint64_t)row * in_dim + i] << 16;
                            float value;
                            memcpy(&value, &bits, sizeof(value));
                            sum += (double)value * test_round_bf16(input[(uint64_t)t * in_dim + i]);
                        }
                        float got = output[(uint64_t)t * out_dim + row];
                        TEST_ASSERT(isfinite(got));
                        max_abs = fmaxf(max_abs, fabsf(got - (float)sum * scale));
                    }
                }
                TEST_ASSERT(max_abs < 1.0e-4f);
            }
        }
        q36_gpu_tensor_free(x);
        q36_gpu_tensor_free(y);
        q36_gpu_cleanup();
        free(weights);
        free(input);
        free(output);
    }
#endif
}

static void test_vulkan_f32_matvec(void) {
#ifdef Q36_NO_GPU
    test_skip("vulkan-kernels", "CPU-only build");
    return;
#else
    const uint32_t in_dim = 257;
    const uint32_t out_dim = 37;
    const uint64_t weight_bytes = (uint64_t)in_dim * out_dim * sizeof(float);
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    float *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            weights[(uint64_t)o * in_dim + i] =
                (float)((int)((o * 7u + i * 13u) % 29u) - 14) / 37.0f;
        }
    }

    q36_gpu_tensor *x = q36_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    q36_gpu_tensor *out = q36_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        q36_gpu_tensor_free(x);
        q36_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)in_dim * sizeof(float));
    float *out_host = malloc((size_t)out_dim * sizeof(float));
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        q36_gpu_tensor_free(x);
        q36_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t i = 0; i < in_dim; i++) {
        x_host[i] = (float)((int)(i % 19u) - 9) / 21.0f;
    }

    TEST_ASSERT(q36_gpu_tensor_write(x, 0, x_host, (uint64_t)in_dim * sizeof(float)) != 0);
    TEST_ASSERT(q36_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    TEST_ASSERT(q36_gpu_matmul_f32_tensor(out, weights_raw, weight_alloc, 0,
                                          in_dim, out_dim, x, 1) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(out, 0, out_host, (uint64_t)out_dim * sizeof(float)) != 0);

    float max_abs = 0.0f;
    for (uint32_t o = 0; o < out_dim; o++) {
        float ref = 0.0f;
        for (uint32_t i = 0; i < in_dim; i++) ref += weights[(uint64_t)o * in_dim + i] * x_host[i];
        float err = fabsf(out_host[o] - ref);
        if (err > max_abs) max_abs = err;
    }
    TEST_ASSERT(max_abs < 1.0e-4f);

    free(x_host);
    free(out_host);
    q36_gpu_tensor_free(x);
    q36_gpu_tensor_free(out);
    free(weights_raw);
#endif
}

static void test_vulkan_rms_norm(void) {
#ifdef Q36_NO_GPU
    test_skip("vulkan-kernels", "CPU-only build");
    return;
#else
    const uint32_t n = 37;
    const uint32_t rows = 3;
    const float eps = 1.0e-6f;
    const uint64_t bytes = (uint64_t)n * rows * sizeof(float);
    const uint64_t weight_bytes = (uint64_t)n * sizeof(float);
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;
    memset(weights_raw, 0, (size_t)weight_alloc);

    float *weights = weights_raw;
    float *x_host = malloc((size_t)bytes);
    float *out_host = malloc((size_t)bytes);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        free(weights_raw);
        return;
    }

    for (uint32_t i = 0; i < n; i++) weights[i] = 0.75f + (float)(i % 11u) * 0.03125f;
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t i = 0; i < n; i++) {
            x_host[(uint64_t)r * n + i] = (float)((int)((r * 17u + i * 5u) % 31u) - 15) / 9.0f;
        }
    }

    q36_gpu_tensor *x = q36_gpu_tensor_alloc(bytes);
    q36_gpu_tensor *out = q36_gpu_tensor_alloc(bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        q36_gpu_tensor_free(x);
        q36_gpu_tensor_free(out);
        free(x_host);
        free(out_host);
        free(weights_raw);
        return;
    }

    TEST_ASSERT(q36_gpu_tensor_write(x, 0, x_host, bytes) != 0);
    TEST_ASSERT(q36_gpu_rms_norm_plain_rows_tensor(out, x, n, rows, eps) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(out, 0, out_host, bytes) != 0);

    float max_abs = 0.0f;
    for (uint32_t r = 0; r < rows; r++) {
        const float *xr = x_host + (uint64_t)r * n;
        float ss = 0.0f;
        for (uint32_t i = 0; i < n; i++) ss += xr[i] * xr[i];
        float scale = 1.0f / sqrtf(ss / (float)n + eps);
        for (uint32_t i = 0; i < n; i++) {
            float ref = xr[i] * scale;
            float err = fabsf(out_host[(uint64_t)r * n + i] - ref);
            if (err > max_abs) max_abs = err;
        }
    }
    TEST_ASSERT(max_abs < 1.0e-5f);

    TEST_ASSERT(q36_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    TEST_ASSERT(q36_gpu_rms_norm_weight_rows_tensor(out, x, weights_raw, weight_alloc, 0, n, rows, eps) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(out, 0, out_host, bytes) != 0);

    max_abs = 0.0f;
    for (uint32_t r = 0; r < rows; r++) {
        const float *xr = x_host + (uint64_t)r * n;
        float ss = 0.0f;
        for (uint32_t i = 0; i < n; i++) ss += xr[i] * xr[i];
        float scale = 1.0f / sqrtf(ss / (float)n + eps);
        for (uint32_t i = 0; i < n; i++) {
            float ref = xr[i] * scale * weights[i];
            float err = fabsf(out_host[(uint64_t)r * n + i] - ref);
            if (err > max_abs) max_abs = err;
        }
    }
    TEST_ASSERT(max_abs < 1.0e-5f);

    q36_gpu_tensor_free(x);
    q36_gpu_tensor_free(out);
    free(x_host);
    free(out_host);
    free(weights_raw);
#endif
}

static void test_vulkan_split_norm(void) {
#if !defined(Q36_NO_GPU) && !defined(Q36_METAL)
    enum { max_n = 5157, max_rows = 9, guard = 64 };
    size_t bytes = (size_t)max_n * max_rows * sizeof(float);
    size_t weight_bytes = test_round_up_u64(max_n * sizeof(float), getpagesize());
    float *weights = NULL, *input = malloc(bytes), *want = malloc(bytes), *got = malloc(bytes);
    const char *saved = getenv("Q36_VK_SPLIT_NORM");
    char *old_env = saved ? strdup(saved) : NULL;
    uint8_t canary[guard], tail[guard];
    memset(canary, 0xa5, guard);
    q36_gpu_tensor *t[6] = {0};
    TEST_ASSERT(posix_memalign((void **)&weights, getpagesize(), weight_bytes) == 0);
    TEST_ASSERT(weights && input && want && got);
    if (!weights || !input || !want || !got) goto done;
    for (unsigned i = 0; i < max_n; i++) weights[i] = 0.75f + (i % 11u) * 0.03125f;
    TEST_ASSERT(q36_gpu_init() != 0);
    q36_gpu_set_dense_model(true);
    TEST_ASSERT(q36_gpu_set_model_map(weights, weight_bytes) != 0);
    for (unsigned i = 0; i < 6; i++) {
        t[i] = q36_gpu_tensor_alloc(bytes + guard);
        TEST_ASSERT(t[i] != NULL);
        if (!t[i]) goto done;
    }
    const unsigned widths[] = {2047, 2048, 4095, 4096, 5120, 5157};
    for (unsigned wi = 0; wi < sizeof(widths) / sizeof(*widths); wi++) {
        unsigned n = widths[wi];
        q36_gpu_set_dense_model(n >= 4095u);
        for (unsigned rows = 1; rows <= max_rows; rows++) {
            size_t nb = (size_t)n * rows * sizeof(float);
            for (unsigned fixture = 0; fixture < 3; fixture++) {
                for (size_t i = 0; i < nb / sizeof(float); i++)
                    input[i] = fixture == 0 ? sinf(i * 0.031f) * (0.25f + (i % 11u)) :
                               fixture == 1 ? (i % 7u ? 0.0001f : -1024.0f) : 0.0f;
                for (unsigned kind = 0; kind < 3; kind++) {
                    for (unsigned alias = 0; alias < 2; alias++) {
                        q36_gpu_tensor *out = alias ? t[0] : t[2];
                        q36_gpu_tensor *sum = alias ? t[1] : t[3];
                        TEST_ASSERT(q36_gpu_tensor_write(t[0], 0, input, nb) != 0);
                        TEST_ASSERT(q36_gpu_tensor_write(t[1], 0, input, nb) != 0);
                        TEST_ASSERT(q36_gpu_tensor_write(out, nb, canary, guard) != 0);
                        TEST_ASSERT(q36_gpu_tensor_write(sum, nb, canary, guard) != 0);
                        for (unsigned arm = 0; arm < 2; arm++) {
                            q36_gpu_tensor *dst = arm ? out : t[4];
                            setenv("Q36_VK_SPLIT_NORM", arm ? "1" : "0", 1);
                            if (kind == 0)
                                TEST_ASSERT(q36_gpu_rms_norm_plain_rows_tensor(dst, t[0], n, rows, 1e-6f) != 0);
                            else if (kind == 1)
                                TEST_ASSERT(q36_gpu_rms_norm_weight_rows_tensor(dst, t[0], weights,
                                    weight_bytes, 0, n, rows, 1e-6f) != 0);
                            else
                                TEST_ASSERT(q36_gpu_add_rms_norm_tensor(dst, arm ? sum : t[5], t[0], t[1],
                                    weights, weight_bytes, 0, n, rows, 1e-6f) != 0);
                            TEST_ASSERT(q36_gpu_tensor_read(dst, 0, arm ? got : want, nb) != 0);
                        }
                        TEST_ASSERT(memcmp(got, want, nb) == 0);
                        TEST_ASSERT(q36_gpu_tensor_read(out, nb, tail, guard) != 0);
                        TEST_ASSERT(memcmp(tail, canary, guard) == 0);
                        if (kind == 2) {
                            TEST_ASSERT(q36_gpu_tensor_read(sum, 0, got, nb) != 0);
                            TEST_ASSERT(q36_gpu_tensor_read(t[5], 0, want, nb) != 0);
                            TEST_ASSERT(memcmp(got, want, nb) == 0);
                            TEST_ASSERT(q36_gpu_tensor_read(sum, nb, tail, guard) != 0);
                            TEST_ASSERT(memcmp(tail, canary, guard) == 0);
                        }
                    }
                }
            }
        }
    }
done:
    if (old_env) setenv("Q36_VK_SPLIT_NORM", old_env, 1);
    else unsetenv("Q36_VK_SPLIT_NORM");
    free(old_env);
    for (unsigned i = 0; i < 6; i++) q36_gpu_tensor_free(t[i]);
    q36_gpu_set_dense_model(false);
    q36_gpu_cleanup();
    free(weights); free(input); free(want); free(got);
#endif
}

static void test_vulkan_add_rms_q8(void) {
#if defined(Q36_NO_GPU) || defined(Q36_METAL)
    test_skip("vulkan-add-rms-q8", "Vulkan-only test");
#else
    enum { max_n = 5120, bytes = max_n * 4, qbytes = max_n / 256 * 296 + 16 };
    float *weights = NULL;
    float input[max_n], got[max_n], want[max_n];
    uint8_t qgot[qbytes], qwant[qbytes];
    q36_gpu_tensor *a = NULL, *b = NULL, *sum = NULL, *norm = NULL, *q = NULL;
    q36_gpu_tensor *ref_sum = NULL, *ref_norm = NULL, *ref_q = NULL;
    TEST_ASSERT(posix_memalign((void **)&weights, (size_t)getpagesize(), bytes) == 0);
    if (!weights) return;
    for (unsigned i = 0; i < max_n; i++) weights[i] = 1.0f;
    TEST_ASSERT(q36_gpu_init() != 0);
    TEST_ASSERT(q36_gpu_set_model_map(weights, bytes) != 0);
    a = q36_gpu_tensor_alloc(bytes);
    b = q36_gpu_tensor_alloc(bytes);
    sum = q36_gpu_tensor_alloc(bytes);
    norm = q36_gpu_tensor_alloc(bytes);
    q = q36_gpu_tensor_alloc(qbytes);
    ref_sum = q36_gpu_tensor_alloc(bytes);
    ref_norm = q36_gpu_tensor_alloc(bytes);
    ref_q = q36_gpu_tensor_alloc(qbytes);
    TEST_ASSERT(a && b && sum && norm && q && ref_sum && ref_norm && ref_q);
    if (!a || !b || !sum || !norm || !q || !ref_sum || !ref_norm || !ref_q) goto cleanup;
    const unsigned widths[] = {256, 768, 3840, 4096, 5120};
    for (unsigned wi = 0; wi < sizeof(widths) / sizeof(widths[0]); wi++) {
        unsigned n = widths[wi], nb = n * 4, nq = n / 256 * 296;
        for (unsigned fixture = 0; fixture < 3; fixture++) {
            /* Equal opposite maxima cross lane boundaries; zeros exercise d=0. */
            for (unsigned i = 0; i < n; i++)
                input[i] = fixture == 0 ? sinf(i * 0.031f) * (0.25f + (i % 11) * 0.03125f) :
                           fixture == 1 ? (i & 1u ? -0.5f : 0.5f) : 0.0f;
            for (unsigned alias = 0; alias < 3; alias++) {
                q36_gpu_tensor *dst = alias == 0 ? sum : alias == 1 ? a : b;
                TEST_ASSERT(q36_gpu_tensor_write(a, 0, input, nb) != 0);
                TEST_ASSERT(q36_gpu_tensor_write(b, 0, input, nb) != 0);
                memset(qgot, 0xa5, qbytes);
                TEST_ASSERT(q36_gpu_tensor_write(q, 0, qgot, qbytes) != 0);
                TEST_ASSERT(q36_gpu_add_rms_norm_tensor(ref_norm, ref_sum, a, b,
                    weights, bytes, 0, n, 1, 1e-6f) != 0);
                TEST_ASSERT(q36_gpu_quantize_q8_k_tensor(ref_q, ref_norm, n, 1) != 0);
                q36_gpu_set_dense_model(true);
                TEST_ASSERT(q36_gpu_add_rms_norm_q8_k_tensor(norm, dst, a, b,
                    weights, bytes, 0, n, 1, 1e-6f, q) != 0);
                q36_gpu_set_dense_model(false);
                TEST_ASSERT(q36_gpu_tensor_read(dst, 0, got, nb) != 0);
                TEST_ASSERT(q36_gpu_tensor_read(ref_sum, 0, want, nb) != 0);
                TEST_ASSERT(memcmp(got, want, nb) == 0);
                TEST_ASSERT(q36_gpu_tensor_read(norm, 0, got, nb) != 0);
                TEST_ASSERT(q36_gpu_tensor_read(ref_norm, 0, want, nb) != 0);
                TEST_ASSERT(memcmp(got, want, nb) == 0);
                TEST_ASSERT(q36_gpu_tensor_read(q, 0, qgot, qbytes) != 0);
                TEST_ASSERT(q36_gpu_tensor_read(ref_q, 0, qwant, nq) != 0);
                TEST_ASSERT(memcmp(qgot, qwant, nq) == 0);
                for (unsigned i = nq; i < qbytes; i++) TEST_ASSERT(qgot[i] == 0xa5);
            }
        }
    }
    TEST_ASSERT(q36_gpu_add_rms_norm_q8_k_tensor(norm, sum, a, b,
        weights, bytes, 0, 257, 1, 1e-6f, q) == 0);
    TEST_ASSERT(q36_gpu_add_rms_norm_q8_k_tensor(norm, sum, a, b,
        weights, bytes, 0, 256, 2, 1e-6f, q) == 0);
cleanup:
    q36_gpu_tensor_free(a);
    q36_gpu_tensor_free(b);
    q36_gpu_tensor_free(sum);
    q36_gpu_tensor_free(norm);
    q36_gpu_tensor_free(q);
    q36_gpu_tensor_free(ref_sum);
    q36_gpu_tensor_free(ref_norm);
    q36_gpu_tensor_free(ref_q);
    q36_gpu_cleanup();
    free(weights);
#endif
}

static void test_vulkan_rms_norm_rope_one(uint32_t n_head) {
#ifdef Q36_NO_GPU
    (void)n_head;
    test_skip("vulkan-kernels", "CPU-only build");
#else
    const uint32_t head_dim = 256;
    const uint32_t src_stride = 512;
    const uint32_t n_tok = 2;
    const uint32_t rows = n_head * n_tok;
    const uint64_t src_floats = (uint64_t)rows * src_stride;
    const uint64_t dst_floats = (uint64_t)rows * head_dim;
    const uint64_t weight_bytes = (uint64_t)head_dim * sizeof(float);
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());
    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;
    memset(weights_raw, 0, (size_t)weight_alloc);
    float *weights = weights_raw;
    float *src_host = malloc(src_floats * sizeof(float));
    float *sep_host = malloc(dst_floats * sizeof(float));
    float *fused_host = malloc(dst_floats * sizeof(float));
    TEST_ASSERT(src_host && sep_host && fused_host);
    if (!src_host || !sep_host || !fused_host) goto done;

    for (uint32_t i = 0; i < head_dim; i++)
        weights[i] = 0.75f + (float)(i % 17u) / 32.0f;
    for (uint64_t i = 0; i < src_floats; i++)
        src_host[i] = (float)((int)((i * 13u + 7u) % 97u) - 48) / 31.0f;

    q36_gpu_tensor *src = q36_gpu_tensor_alloc(src_floats * sizeof(float));
    q36_gpu_tensor *sep = q36_gpu_tensor_alloc(dst_floats * sizeof(float));
    q36_gpu_tensor *fused = q36_gpu_tensor_alloc(dst_floats * sizeof(float));
    TEST_ASSERT(src && sep && fused);
    if (!src || !sep || !fused) {
        q36_gpu_tensor_free(src);
        q36_gpu_tensor_free(sep);
        q36_gpu_tensor_free(fused);
        goto done;
    }
    TEST_ASSERT(q36_gpu_tensor_write(src, 0, src_host, src_floats * sizeof(float)) != 0);
    TEST_ASSERT(q36_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    TEST_ASSERT(q36_gpu_extract_full_attn_q_tensor(sep, src, n_head, head_dim, n_tok) != 0);
    TEST_ASSERT(q36_gpu_rms_norm_weight_rows_tensor(sep, sep, weights_raw, weight_alloc,
                                                    0, head_dim, rows, 1.0e-6f) != 0);
    TEST_ASSERT(q36_gpu_rope_qwen_rows_tensor(sep, n_head, 19, n_tok) != 0);
    TEST_ASSERT(q36_gpu_rms_norm_rope_qwen_rows_tensor(fused, src, weights_raw, weight_alloc,
                                                       0, src_stride, n_head, 19, n_tok,
                                                       1.0e-6f) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(sep, 0, sep_host, dst_floats * sizeof(float)) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(fused, 0, fused_host, dst_floats * sizeof(float)) != 0);
    {
        float max_abs = 0.0f;
        uint64_t worst = 0;
        for (uint64_t i = 0; i < dst_floats; i++) {
            float err = fabsf(fused_host[i] - sep_host[i]);
            if (err > max_abs) {
                max_abs = err;
                worst = i;
            }
        }
        if (max_abs != 0.0f) {
            fprintf(stderr, "q36-test: norm-rope max_abs=%g worst=%" PRIu64 " fused=%g separate=%g\n",
                    max_abs, worst, fused_host[worst], sep_host[worst]);
        }
        TEST_ASSERT(max_abs == 0.0f);
    }
    q36_gpu_tensor_free(src);
    q36_gpu_tensor_free(sep);
    q36_gpu_tensor_free(fused);

done:
    free(src_host);
    free(sep_host);
    free(fused_host);
    free(weights_raw);
#endif
}

static void test_vulkan_rms_norm_rope(void) {
    test_vulkan_rms_norm_rope_one(16);
    test_vulkan_rms_norm_rope_one(24);
}

static void test_vulkan_rms_norm_rope_kv(void) {
#ifdef Q36_NO_GPU
    test_skip("vulkan-kernels", "CPU-only build");
#else
    const uint32_t n_head = 2, head_dim = 256, n_tok = 2, cap = 8, pos0 = 3;
    const uint32_t row = n_head * head_dim;
    const uint32_t cache_row_bytes = row * sizeof(uint16_t);
    const uint64_t values = (uint64_t)n_tok * row;
    const uint64_t cache_bytes = (uint64_t)cap * cache_row_bytes;
    const uint64_t weight_bytes = (uint64_t)head_dim * sizeof(float);
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());
    void *weights_raw = NULL;
    float *k_host = NULL, *v_host = NULL;
    unsigned char *a = NULL, *b = NULL;
    q36_gpu_tensor *k = NULL, *v = NULL, *sep = NULL;
    q36_gpu_tensor *ka = NULL, *va = NULL, *kb = NULL, *vb = NULL;

    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;
    memset(weights_raw, 0, (size_t)weight_alloc);
    k_host = malloc(values * sizeof(float));
    v_host = malloc(values * sizeof(float));
    a = malloc((size_t)n_tok * cache_row_bytes);
    b = malloc((size_t)n_tok * cache_row_bytes);
    TEST_ASSERT(k_host && v_host && a && b);
    if (!k_host || !v_host || !a || !b) goto done;
    for (uint32_t i = 0; i < head_dim; i++)
        ((float *)weights_raw)[i] = 0.75f + (float)(i % 17u) / 32.0f;
    for (uint64_t i = 0; i < values; i++) {
        k_host[i] = (float)((int)((i * 13u + 7u) % 97u) - 48) / 31.0f;
        v_host[i] = (float)((int)((i * 19u + 3u) % 89u) - 44) / 29.0f;
    }
    k = q36_gpu_tensor_alloc(values * sizeof(float));
    v = q36_gpu_tensor_alloc(values * sizeof(float));
    sep = q36_gpu_tensor_alloc(values * sizeof(float));
    ka = q36_gpu_tensor_alloc(cache_bytes);
    va = q36_gpu_tensor_alloc(cache_bytes);
    kb = q36_gpu_tensor_alloc(cache_bytes);
    vb = q36_gpu_tensor_alloc(cache_bytes);
    TEST_ASSERT(k && v && sep && ka && va && kb && vb);
    if (!k || !v || !sep || !ka || !va || !kb || !vb) goto done;
    TEST_ASSERT(q36_gpu_tensor_write(k, 0, k_host, values * sizeof(float)) != 0);
    TEST_ASSERT(q36_gpu_tensor_write(v, 0, v_host, values * sizeof(float)) != 0);
    TEST_ASSERT(q36_gpu_tensor_write(sep, 0, k_host, values * sizeof(float)) != 0);
    TEST_ASSERT(q36_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    TEST_ASSERT(q36_gpu_rms_norm_weight_rows_tensor(sep, sep, weights_raw, weight_alloc,
                                                    0, head_dim, n_head * n_tok, 1.0e-6f) != 0);
    TEST_ASSERT(q36_gpu_rope_qwen_rows_tensor(sep, n_head, pos0, n_tok) != 0);
    TEST_ASSERT(q36_gpu_attn_kv_store_tensor(ka, va, sep, v, pos0, n_tok, cap,
                                             row, row, 0, 0,
                                             cache_row_bytes, cache_row_bytes) != 0);
    TEST_ASSERT(q36_gpu_rms_norm_rope_qwen_kv_store_tensor(
                    kb, vb, k, v, weights_raw, weight_alloc, 0,
                    head_dim, n_head, pos0, n_tok, cap, 1.0e-6f) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(ka, (uint64_t)pos0 * cache_row_bytes,
                                    a, (uint64_t)n_tok * cache_row_bytes) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(kb, (uint64_t)pos0 * cache_row_bytes,
                                    b, (uint64_t)n_tok * cache_row_bytes) != 0);
    TEST_ASSERT(memcmp(a, b, (size_t)n_tok * cache_row_bytes) == 0);
    TEST_ASSERT(q36_gpu_tensor_read(va, (uint64_t)pos0 * cache_row_bytes,
                                    a, (uint64_t)n_tok * cache_row_bytes) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(vb, (uint64_t)pos0 * cache_row_bytes,
                                    b, (uint64_t)n_tok * cache_row_bytes) != 0);
    TEST_ASSERT(memcmp(a, b, (size_t)n_tok * cache_row_bytes) == 0);

done:
    q36_gpu_tensor_free(k);
    q36_gpu_tensor_free(v);
    q36_gpu_tensor_free(sep);
    q36_gpu_tensor_free(ka);
    q36_gpu_tensor_free(va);
    q36_gpu_tensor_free(kb);
    q36_gpu_tensor_free(vb);
    free(k_host);
    free(v_host);
    free(a);
    free(b);
    free(weights_raw);
#endif
}

/* GPU quantized KV store must be byte-identical to the host quantizers:
 * same blocks, same f16 scales, same packing, at the same cache offsets. */
static void test_kv_quant_diff(const char *label, const unsigned char *exp,
                               const unsigned char *got, size_t n, uint32_t blk_bytes) {
    for (size_t i = 0; i < n; i++) {
        if (exp[i] == got[i]) continue;
        size_t blk = i / blk_bytes, off = i - blk * blk_bytes;
        fprintf(stderr,
                "q36-test: kv-quant %s first diff at byte %zu (block %zu +%zu) "
                "exp=%02x got=%02x scale exp=%02x%02x got=%02x%02x\n",
                label, i, blk, off, exp[i], got[i],
                exp[blk * blk_bytes + 1], exp[blk * blk_bytes],
                got[blk * blk_bytes + 1], got[blk * blk_bytes]);
        return;
    }
}
static void test_vulkan_kv_store_quant_case(uint32_t type_k, uint32_t type_v) {
#ifdef Q36_NO_GPU
    (void)type_k; (void)type_v;
#else
    const uint32_t row = 512, n_tok = 5, cap = 8, pos0 = 2;
    const uint32_t k_row_bytes = (row / 32u) * (type_k == Q36_KV_CACHE_Q8_0 ? 34u : 18u);
    const uint32_t v_row_bytes = (row / 32u) * (type_v == Q36_KV_CACHE_Q8_0 ? 34u : 18u);
    const uint64_t values = (uint64_t)n_tok * row;
    float *k_host = malloc(values * sizeof(float));
    float *v_host = malloc(values * sizeof(float));
    unsigned char *k_exp = malloc((size_t)n_tok * k_row_bytes);
    unsigned char *v_exp = malloc((size_t)n_tok * v_row_bytes);
    unsigned char *k_got = malloc((size_t)n_tok * k_row_bytes);
    unsigned char *v_got = malloc((size_t)n_tok * v_row_bytes);
    q36_gpu_tensor *k = NULL, *v = NULL, *kc = NULL, *vc = NULL;

    TEST_ASSERT(k_host && v_host && k_exp && v_exp && k_got && v_got);
    if (!k_host || !v_host || !k_exp || !v_exp || !k_got || !v_got) goto done;
    for (uint64_t i = 0; i < values; i++) {
        k_host[i] = (float)((int)((i * 13u + 7u) % 97u) - 48) / 31.0f;
        v_host[i] = (float)((int)((i * 19u + 3u) % 89u) - 44) / 29.0f;
    }
    /* All-zero blocks exercise the d == 0 path. */
    for (uint32_t i = 0; i < 32; i++) {
        k_host[512 + 96 + i] = 0.0f;
        v_host[160 + i] = 0.0f;
    }
    for (uint32_t t = 0; t < n_tok; t++) {
        if (type_k == Q36_KV_CACHE_Q8_0)
            q36_quant_q8_0(k_host + (size_t)t * row, k_exp + (size_t)t * k_row_bytes, row);
        else
            q36_quant_q4_0_key(k_host + (size_t)t * row, k_exp + (size_t)t * k_row_bytes, row);
        if (type_v == Q36_KV_CACHE_Q8_0)
            q36_quant_q8_0(v_host + (size_t)t * row, v_exp + (size_t)t * v_row_bytes, row);
        else
            q36_quant_q4_0_kv(v_host + (size_t)t * row, v_exp + (size_t)t * v_row_bytes, row);
    }
    k = q36_gpu_tensor_alloc(values * sizeof(float));
    v = q36_gpu_tensor_alloc(values * sizeof(float));
    kc = q36_gpu_tensor_alloc((uint64_t)cap * k_row_bytes);
    vc = q36_gpu_tensor_alloc((uint64_t)cap * v_row_bytes);
    TEST_ASSERT(k && v && kc && vc);
    if (!k || !v || !kc || !vc) goto done;
    TEST_ASSERT(q36_gpu_tensor_write(k, 0, k_host, values * sizeof(float)) != 0);
    TEST_ASSERT(q36_gpu_tensor_write(v, 0, v_host, values * sizeof(float)) != 0);
    TEST_ASSERT(q36_gpu_attn_kv_store_tensor(kc, vc, k, v, pos0, n_tok, cap,
                                             row, row, type_k, type_v,
                                             k_row_bytes, v_row_bytes) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(kc, (uint64_t)pos0 * k_row_bytes,
                                    k_got, (uint64_t)n_tok * k_row_bytes) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(vc, (uint64_t)pos0 * v_row_bytes,
                                    v_got, (uint64_t)n_tok * v_row_bytes) != 0);
    test_kv_quant_diff("store-k", k_exp, k_got, (size_t)n_tok * k_row_bytes,
                       type_k == Q36_KV_CACHE_Q8_0 ? 34u : 18u);
    test_kv_quant_diff("store-v", v_exp, v_got, (size_t)n_tok * v_row_bytes,
                       type_v == Q36_KV_CACHE_Q8_0 ? 34u : 18u);
    TEST_ASSERT(memcmp(k_got, k_exp, (size_t)n_tok * k_row_bytes) == 0);
    TEST_ASSERT(memcmp(v_got, v_exp, (size_t)n_tok * v_row_bytes) == 0);
    /* Single-token append (decode path). */
    TEST_ASSERT(q36_gpu_attn_kv_store_tensor(kc, vc, k, v, cap - 1u, 1, cap,
                                             row, row, type_k, type_v,
                                             k_row_bytes, v_row_bytes) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(kc, (uint64_t)(cap - 1u) * k_row_bytes,
                                    k_got, k_row_bytes) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(vc, (uint64_t)(cap - 1u) * v_row_bytes,
                                    v_got, v_row_bytes) != 0);
    TEST_ASSERT(memcmp(k_got, k_exp, k_row_bytes) == 0);
    TEST_ASSERT(memcmp(v_got, v_exp, v_row_bytes) == 0);

done:
    q36_gpu_tensor_free(k);
    q36_gpu_tensor_free(v);
    q36_gpu_tensor_free(kc);
    q36_gpu_tensor_free(vc);
    free(k_host);
    free(v_host);
    free(k_exp);
    free(v_exp);
    free(k_got);
    free(v_got);
#endif
}

static void test_vulkan_kv_store_quant(void) {
#ifdef Q36_NO_GPU
    test_skip("vulkan-kernels", "CPU-only build");
#else
    test_vulkan_kv_store_quant_case(Q36_KV_CACHE_Q8_0, Q36_KV_CACHE_Q4_0);
    test_vulkan_kv_store_quant_case(Q36_KV_CACHE_Q4_0, Q36_KV_CACHE_Q8_0);
#endif
}

/* Fused norm+rope+quant kernel against the separate norm, rope and quant
 * store kernels: caches must match byte for byte, like the f16 twin above. */
static void test_vulkan_rms_norm_rope_kv_quant(void) {
#ifdef Q36_NO_GPU
    test_skip("vulkan-kernels", "CPU-only build");
#else
    const uint32_t n_head = 2, head_dim = 256, n_tok = 2, cap = 8, pos0 = 3;
    const uint32_t row = n_head * head_dim;
    const uint32_t k_row_bytes = n_head * (head_dim / 32u) * 34u;
    const uint32_t v_row_bytes = n_head * (head_dim / 32u) * 18u;
    const uint64_t values = (uint64_t)n_tok * row;
    const uint64_t weight_bytes = (uint64_t)head_dim * sizeof(float);
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());
    void *weights_raw = NULL;
    float *k_host = NULL, *v_host = NULL;
    unsigned char *a = NULL, *b = NULL;
    q36_gpu_tensor *k = NULL, *v = NULL, *sep = NULL;
    q36_gpu_tensor *ka = NULL, *va = NULL, *kb = NULL, *vb = NULL;

    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;
    memset(weights_raw, 0, (size_t)weight_alloc);
    k_host = malloc(values * sizeof(float));
    v_host = malloc(values * sizeof(float));
    a = malloc((size_t)n_tok * k_row_bytes);
    b = malloc((size_t)n_tok * k_row_bytes);
    TEST_ASSERT(k_host && v_host && a && b);
    if (!k_host || !v_host || !a || !b) goto done;
    for (uint32_t i = 0; i < head_dim; i++)
        ((float *)weights_raw)[i] = 0.75f + (float)(i % 17u) / 32.0f;
    for (uint64_t i = 0; i < values; i++) {
        k_host[i] = (float)((int)((i * 13u + 7u) % 97u) - 48) / 31.0f;
        v_host[i] = (float)((int)((i * 19u + 3u) % 89u) - 44) / 29.0f;
    }
    /* All-zero V block exercises the Q4_0 d == 0 path. */
    for (uint32_t i = 0; i < 32; i++) v_host[160 + i] = 0.0f;
    k = q36_gpu_tensor_alloc(values * sizeof(float));
    v = q36_gpu_tensor_alloc(values * sizeof(float));
    sep = q36_gpu_tensor_alloc(values * sizeof(float));
    ka = q36_gpu_tensor_alloc((uint64_t)cap * k_row_bytes);
    va = q36_gpu_tensor_alloc((uint64_t)cap * v_row_bytes);
    kb = q36_gpu_tensor_alloc((uint64_t)cap * k_row_bytes);
    vb = q36_gpu_tensor_alloc((uint64_t)cap * v_row_bytes);
    TEST_ASSERT(k && v && sep && ka && va && kb && vb);
    if (!k || !v || !sep || !ka || !va || !kb || !vb) goto done;
    TEST_ASSERT(q36_gpu_tensor_write(k, 0, k_host, values * sizeof(float)) != 0);
    TEST_ASSERT(q36_gpu_tensor_write(v, 0, v_host, values * sizeof(float)) != 0);
    TEST_ASSERT(q36_gpu_tensor_write(sep, 0, k_host, values * sizeof(float)) != 0);
    TEST_ASSERT(q36_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    TEST_ASSERT(q36_gpu_rms_norm_weight_rows_tensor(sep, sep, weights_raw, weight_alloc,
                                                    0, head_dim, n_head * n_tok, 1.0e-6f) != 0);
    TEST_ASSERT(q36_gpu_rope_qwen_rows_tensor(sep, n_head, pos0, n_tok) != 0);
    TEST_ASSERT(q36_gpu_attn_kv_store_tensor(ka, va, sep, v, pos0, n_tok, cap,
                                             row, row,
                                             Q36_KV_CACHE_Q8_0, Q36_KV_CACHE_Q4_0,
                                             k_row_bytes, v_row_bytes) != 0);
    TEST_ASSERT(q36_gpu_rms_norm_rope_qwen_kv_store_quant_tensor(
                    kb, vb, k, v, weights_raw, weight_alloc, 0,
                    head_dim, n_head, pos0, n_tok, cap, 1.0e-6f,
                    k_row_bytes, v_row_bytes) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(ka, (uint64_t)pos0 * k_row_bytes,
                                    a, (uint64_t)n_tok * k_row_bytes) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(kb, (uint64_t)pos0 * k_row_bytes,
                                    b, (uint64_t)n_tok * k_row_bytes) != 0);
    TEST_ASSERT(memcmp(a, b, (size_t)n_tok * k_row_bytes) == 0);
    TEST_ASSERT(q36_gpu_tensor_read(va, (uint64_t)pos0 * v_row_bytes,
                                    a, (uint64_t)n_tok * v_row_bytes) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(vb, (uint64_t)pos0 * v_row_bytes,
                                    b, (uint64_t)n_tok * v_row_bytes) != 0);
    TEST_ASSERT(memcmp(a, b, (size_t)n_tok * v_row_bytes) == 0);

done:
    q36_gpu_tensor_free(k);
    q36_gpu_tensor_free(v);
    q36_gpu_tensor_free(sep);
    q36_gpu_tensor_free(ka);
    q36_gpu_tensor_free(va);
    q36_gpu_tensor_free(kb);
    q36_gpu_tensor_free(vb);
    free(k_host);
    free(v_host);
    free(a);
    free(b);
    free(weights_raw);
#endif
}

static void test_vulkan_recurrent_conv_decode_case(uint32_t conv_dim) {
#ifdef Q36_NO_GPU
    (void)conv_dim;
    test_skip("vulkan-kernels", "CPU-only build");
#else
    const uint32_t n_taps = 4, hist = n_taps - 1u, n_tok = 7;
    const uint64_t row_bytes = (uint64_t)conv_dim * sizeof(float);
    const uint64_t cache_bytes = hist * row_bytes;
    const uint64_t window_bytes = (uint64_t)n_tok * n_taps * row_bytes;
    const uint64_t output_bytes = (uint64_t)n_tok * row_bytes;
    const uint64_t kernel_bytes = (uint64_t)n_taps * row_bytes;
    const uint64_t kernel_alloc = test_round_up_u64(kernel_bytes, (uint64_t)getpagesize());
    void *kernel_raw = NULL;
    float *cache_host = NULL, *cur_host = NULL, *a = NULL, *b = NULL;
    q36_gpu_tensor *cache_a = NULL, *cache_b = NULL, *cur = NULL;
    q36_gpu_tensor *window = NULL, *out_a = NULL, *out_b = NULL;

    TEST_ASSERT(posix_memalign(&kernel_raw, (size_t)getpagesize(), (size_t)kernel_alloc) == 0);
    if (!kernel_raw) return;
    memset(kernel_raw, 0, (size_t)kernel_alloc);
    cache_host = malloc((size_t)cache_bytes);
    cur_host = malloc((size_t)output_bytes);
    a = malloc((size_t)output_bytes);
    b = malloc((size_t)output_bytes);
    TEST_ASSERT(cache_host && cur_host && a && b);
    if (!cache_host || !cur_host || !a || !b) goto done;
    for (uint64_t i = 0; i < kernel_bytes / sizeof(float); i++)
        ((float *)kernel_raw)[i] = (float)((int)((i * 7u + 3u) % 29u) - 14) / 127.0f;
    for (uint64_t i = 0; i < cache_bytes / sizeof(float); i++)
        cache_host[i] = (float)((int)((i * 11u + 5u) % 41u) - 20) / 53.0f;
    for (uint64_t i = 0; i < output_bytes / sizeof(float); i++)
        cur_host[i] = (float)((int)((i * 13u + 1u) % 37u) - 18) / 47.0f;

    cache_a = q36_gpu_tensor_alloc(cache_bytes);
    cache_b = q36_gpu_tensor_alloc(cache_bytes);
    cur = q36_gpu_tensor_alloc(output_bytes);
    window = q36_gpu_tensor_alloc(window_bytes);
    out_a = q36_gpu_tensor_alloc(output_bytes);
    out_b = q36_gpu_tensor_alloc(output_bytes);
    TEST_ASSERT(cache_a && cache_b && cur && window && out_a && out_b);
    if (!cache_a || !cache_b || !cur || !window || !out_a || !out_b) goto done;
    TEST_ASSERT(q36_gpu_tensor_write(cache_a, 0, cache_host, cache_bytes) != 0);
    TEST_ASSERT(q36_gpu_tensor_write(cache_b, 0, cache_host, cache_bytes) != 0);
    TEST_ASSERT(q36_gpu_tensor_write(cur, 0, cur_host, output_bytes) != 0);
    TEST_ASSERT(q36_gpu_set_model_map(kernel_raw, kernel_alloc) != 0);
    TEST_ASSERT(q36_gpu_recurrent_conv_step_tensor(cache_a, cur, window, conv_dim, n_tok) != 0);
    TEST_ASSERT(q36_gpu_ssm_conv_silu_tensor(out_a, window, kernel_raw, kernel_alloc, 0,
                                             conv_dim, n_taps, n_tok) != 0);
    TEST_ASSERT(q36_gpu_recurrent_conv_silu_tensor(cache_b, cur, out_b,
                                                   kernel_raw, kernel_alloc, 0,
                                                   conv_dim, n_taps, n_tok) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(out_a, 0, a, output_bytes) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(out_b, 0, b, output_bytes) != 0);
    TEST_ASSERT(memcmp(a, b, (size_t)output_bytes) == 0);
    TEST_ASSERT(q36_gpu_tensor_read(cache_a, 0, a, cache_bytes) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(cache_b, 0, b, cache_bytes) != 0);
    TEST_ASSERT(memcmp(a, b, (size_t)cache_bytes) == 0);

done:
    q36_gpu_tensor_free(cache_a);
    q36_gpu_tensor_free(cache_b);
    q36_gpu_tensor_free(cur);
    q36_gpu_tensor_free(window);
    q36_gpu_tensor_free(out_a);
    q36_gpu_tensor_free(out_b);
    free(cache_host);
    free(cur_host);
    free(a);
    free(b);
    free(kernel_raw);
#endif
}

static void test_vulkan_delta_verify(void) {
#if !defined(Q36_NO_GPU) && !defined(Q36_METAL)
    enum { heads = 2, dim = 128, width = heads * dim, tokens = 128,
           state_count = heads * dim * dim };
    float input[3][tokens * width], gates[tokens * heads * 2];
    union { float f[state_count]; uint16_t h[state_count]; } initial;
    float a[state_count], b[state_count];
    q36_gpu_tensor *x[4] = {0}, *state[2] = {0}, *out[2] = {0}, *source = NULL;
    for (unsigned i = 0; i < tokens * width; i++) {
        input[0][i] = sinf(i * 0.13f) * 0.08f;
        input[1][i] = cosf(i * 0.17f) * 0.08f;
        input[2][i] = sinf(i * 0.19f) * 0.3f;
    }
    for (unsigned t = 0; t < tokens; t++)
        for (unsigned h = 0; h < heads; h++) {
            gates[2 * t * heads + h] = 0.93f + h * 0.02f;
            gates[(2 * t + 1) * heads + h] = 0.61f - (t % 8u) * 0.03f;
        }
    for (unsigned i = 0; i < 4; i++) {
        uint64_t bytes = i == 3 ? sizeof(gates) : sizeof(input[0]);
        x[i] = q36_gpu_tensor_alloc(bytes);
        TEST_ASSERT(x[i]);
        if (!x[i]) goto done;
        TEST_ASSERT(q36_gpu_tensor_write(x[i], 0, i == 3 ? gates : input[i], bytes));
    }
    for (unsigned half = 0; half < 2; half++) {
        uint64_t state_bytes = state_count * (half ? sizeof(uint16_t) : sizeof(float));
        for (unsigned i = 0; i < state_count; i++) {
            float v = sinf(i * 0.07f) * 0.125f;
            if (half) initial.h[i] = test_float_to_f16(v);
            else initial.f[i] = v;
        }
        for (unsigned i = 0; i < 2; i++) {
            state[i] = q36_gpu_tensor_alloc(state_bytes);
            out[i] = q36_gpu_tensor_alloc(sizeof(input[0]));
            TEST_ASSERT(state[i] && out[i]);
            if (!state[i] || !out[i]) goto done;
        }
        source = q36_gpu_tensor_alloc(state_bytes);
        TEST_ASSERT(source);
        if (!source) goto done;
        TEST_ASSERT(q36_gpu_tensor_write(source, 0, &initial, state_bytes));
        for (unsigned n = 1; n <= tokens; n = n < 8 ? n+1 : n == 8 ? 32 : n == 32 ? 128 : tokens+1) {
            for (unsigned i = 0; i < 2; i++)
                TEST_ASSERT(q36_gpu_tensor_write(state[i], 0, &initial, state_bytes));
            q36_gpu_set_micro_batch(false);
            for (unsigned t = 0; t < n; t++) {
                q36_gpu_tensor *row[5];
                for (unsigned i = 0; i < 5; i++) {
                    uint64_t bytes = (i == 3 ? 2u * heads : width) * sizeof(float);
                    row[i] = q36_gpu_tensor_view(i == 4 ? out[0] : x[i], t * bytes, bytes);
                    TEST_ASSERT(row[i]);
                }
                TEST_ASSERT(q36_gpu_delta_net_decode_tensor(state[0], row[0], row[1],
                    row[2], row[3], row[4], heads, dim, 1));
                for (unsigned i = 0; i < 5; i++) q36_gpu_tensor_free(row[i]);
            }
            q36_gpu_set_dense_prefill_exact(n > 8u);
            q36_gpu_set_micro_batch(n <= 8u);
            TEST_ASSERT(q36_gpu_delta_net_decode_tensor(state[1], x[0], x[1], x[2],
                x[3], out[1], heads, dim, n));
            q36_gpu_set_micro_batch(false);
            q36_gpu_set_dense_prefill_exact(false);
            TEST_ASSERT(q36_gpu_tensor_read(state[0], 0, a, state_bytes));
            TEST_ASSERT(q36_gpu_tensor_read(state[1], 0, b, state_bytes));
            TEST_ASSERT(memcmp(a, b, state_bytes) == 0);
            uint64_t out_bytes = (uint64_t)n * width * sizeof(float);
            TEST_ASSERT(q36_gpu_tensor_read(out[0], 0, a, out_bytes));
            TEST_ASSERT(q36_gpu_tensor_read(out[1], 0, b, out_bytes));
            TEST_ASSERT(memcmp(a, b, out_bytes) == 0);
            if (n > 8u) continue;
            /* The spare destination contains unrelated speculative state. */
            for (unsigned fallback = 0; fallback < (n == 1 ? 2u : 1u); fallback++) {
                memset(b, 0, state_bytes);
                TEST_ASSERT(q36_gpu_tensor_write(state[1], 0, b, state_bytes));
                q36_gpu_set_micro_batch(!fallback);
                TEST_ASSERT(q36_gpu_delta_net_decode_from_tensor(state[1], source,
                    x[0], x[1], x[2], x[3], out[1], heads, dim, n, true));
                q36_gpu_set_micro_batch(false);
                TEST_ASSERT(q36_gpu_tensor_read(state[0], 0, a, state_bytes));
                TEST_ASSERT(q36_gpu_tensor_read(state[1], 0, b, state_bytes));
                TEST_ASSERT(memcmp(a, b, state_bytes) == 0);
                TEST_ASSERT(q36_gpu_tensor_read(out[0], 0, a, out_bytes));
                TEST_ASSERT(q36_gpu_tensor_read(out[1], 0, b, out_bytes));
                TEST_ASSERT(memcmp(a, b, out_bytes) == 0);
                TEST_ASSERT(q36_gpu_tensor_read(source, 0, b, state_bytes));
                TEST_ASSERT(memcmp(&initial, b, state_bytes) == 0);
            }
            q36_gpu_set_micro_batch(true);
            TEST_ASSERT(q36_gpu_delta_net_decode_from_tensor(state[1], source,
                x[0], x[1], x[2], x[3], out[1], heads, dim, n, false));
            q36_gpu_set_micro_batch(false);
            TEST_ASSERT(q36_gpu_tensor_read(state[0], 0, a, state_bytes));
            TEST_ASSERT(q36_gpu_tensor_read(state[1], 0, b, state_bytes));
            TEST_ASSERT(memcmp(a, b, state_bytes) == 0);
        }
        q36_gpu_tensor_free(source); source = NULL;
        for (unsigned i = 0; i < 2; i++) {
            q36_gpu_tensor_free(state[i]); state[i] = NULL;
            q36_gpu_tensor_free(out[i]); out[i] = NULL;
        }
    }
done:
    q36_gpu_set_micro_batch(false);
    q36_gpu_tensor_free(source);
    for (unsigned i = 0; i < 4; i++) q36_gpu_tensor_free(x[i]);
    for (unsigned i = 0; i < 2; i++) {
        q36_gpu_tensor_free(state[i]);
        q36_gpu_tensor_free(out[i]);
    }
#endif
}

static void test_vulkan_delta_qkv_one(uint32_t n_heads) {
#ifdef Q36_NO_GPU
    (void)n_heads;
    test_skip("vulkan-kernels", "CPU-only build");
#else
    const uint32_t n_groups = 16, state_dim = 128;
    const uint32_t conv_stride = (2u * n_groups + n_heads) * state_dim, n_tok = 3;
    const uint64_t conv_bytes = (uint64_t)n_tok * conv_stride * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tok * n_heads * state_dim * sizeof(float);
    float *host = malloc((size_t)conv_bytes);
    float *a = malloc((size_t)out_bytes);
    float *b = malloc((size_t)out_bytes);
    q36_gpu_tensor *conv = NULL, *qa = NULL, *ka = NULL, *va = NULL;
    q36_gpu_tensor *qb = NULL, *kb = NULL, *vb = NULL;
    TEST_ASSERT(host && a && b);
    if (!host || !a || !b) goto done;
    for (uint64_t i = 0; i < conv_bytes / sizeof(float); i++) {
        float scale = i / conv_stride == 0 ? 1 : i / conv_stride == 1 ? 1e-7f : 0;
        host[i] = (float)((int)((i * 17u + 5u) % 101u) - 50) / 61.0f * scale;
    }
    conv = q36_gpu_tensor_alloc(conv_bytes);
    qa = q36_gpu_tensor_alloc(out_bytes);
    ka = q36_gpu_tensor_alloc(out_bytes);
    va = q36_gpu_tensor_alloc(out_bytes);
    qb = q36_gpu_tensor_alloc(out_bytes);
    kb = q36_gpu_tensor_alloc(out_bytes);
    vb = q36_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(conv && qa && ka && va && qb && kb && vb);
    if (!conv || !qa || !ka || !va || !qb || !kb || !vb) goto done;
    TEST_ASSERT(q36_gpu_tensor_write(conv, 0, host, conv_bytes) != 0);
    TEST_ASSERT(q36_gpu_delta_qk_l2_norm_tensor(qa, ka, conv, n_heads, n_groups,
                                                state_dim, conv_stride, n_tok, 1.0e-6f) != 0);
    TEST_ASSERT(q36_gpu_extract_recurrent_v_tensor(va, conv, conv_stride, n_heads * state_dim, n_tok) != 0);
    TEST_ASSERT(q36_gpu_delta_qkv_l2_norm_tensor(qb, kb, vb, conv, n_heads, n_groups,
                                                 state_dim, conv_stride, n_tok, 1.0e-6f) != 0);
#define TEST_SAME_TENSOR(old, fused) do { \
        TEST_ASSERT(q36_gpu_tensor_read(old, 0, a, out_bytes) != 0); \
        TEST_ASSERT(q36_gpu_tensor_read(fused, 0, b, out_bytes) != 0); \
        TEST_ASSERT(memcmp(a, b, (size_t)out_bytes) == 0); \
    } while (0)
    TEST_SAME_TENSOR(qa, qb);
    TEST_SAME_TENSOR(ka, kb);
    for (uint32_t kind = 0; kind < 2; kind++) {
        TEST_ASSERT(q36_gpu_tensor_read(kind ? ka : qa, 0, a, out_bytes) != 0);
        for (uint32_t tok = 0; tok < n_tok; tok++) {
            for (uint32_t h = 0; h < n_heads; h++) {
                const float *src = host + (uint64_t)tok * conv_stride +
                    (kind * n_groups + h % n_groups) * state_dim;
                const float *dst = a + ((uint64_t)tok * n_heads + h) * state_dim;
                double ss = 0, norm = 0;
                for (uint32_t i = 0; i < state_dim; i++) {
                    TEST_ASSERT(isfinite(dst[i]));
                    ss += (double)src[i] * src[i];
                    norm += (double)dst[i] * dst[i];
                }
                double expected = ss / (ss + 1e-6);
                TEST_ASSERT(fabs(norm - expected) <= 1e-6 * expected + 1e-12);
            }
        }
    }
    TEST_SAME_TENSOR(va, vb);
    for (uint32_t tok = 0; tok < n_tok; tok++) {
        TEST_ASSERT(memcmp(a + (uint64_t)tok * n_heads * state_dim,
            host + (uint64_t)tok * conv_stride + 2u * n_groups * state_dim,
            (size_t)n_heads * state_dim * sizeof(float)) == 0);
    }
#ifndef Q36_METAL
    const char *env = getenv("Q36_VK_GPU_RECURRENT");
    char *saved = env ? test_strdup(env) : NULL;
    setenv("Q36_VK_GPU_RECURRENT", "0", 1);
    TEST_ASSERT(q36_gpu_extract_recurrent_v_tensor(va, conv, conv_stride,
                    n_heads * state_dim, n_tok) != 0);
    if (saved) setenv("Q36_VK_GPU_RECURRENT", saved, 1);
    else unsetenv("Q36_VK_GPU_RECURRENT");
    free(saved);
    TEST_SAME_TENSOR(va, vb);
#endif
#undef TEST_SAME_TENSOR

done:
    q36_gpu_tensor_free(conv);
    q36_gpu_tensor_free(qa);
    q36_gpu_tensor_free(ka);
    q36_gpu_tensor_free(va);
    q36_gpu_tensor_free(qb);
    q36_gpu_tensor_free(kb);
    q36_gpu_tensor_free(vb);
    free(host);
    free(a);
    free(b);
#endif
}

static void test_vulkan_delta_qkv(void) {
    test_vulkan_delta_qkv_one(32);
    test_vulkan_delta_qkv_one(48);
}

static void test_vulkan_shared_ffn_decode(void) {
#ifdef Q36_NO_GPU
    test_skip("vulkan-kernels", "CPU-only build");
#else
    typedef struct {
        uint16_t d;
        int8_t qs[32];
    } test_q8_0_block;
    const uint32_t in_dim = 256, mid_dim = 64, out_dim = 37;
    const uint32_t in_blocks = in_dim / 32u, mid_blocks = mid_dim / 32u;
    const uint64_t gate_bytes = (uint64_t)mid_dim * in_blocks * sizeof(test_q8_0_block);
    const uint64_t down_bytes = (uint64_t)out_dim * mid_blocks * sizeof(test_q8_0_block);
    const uint64_t up_offset = gate_bytes;
    const uint64_t down_offset = up_offset + gate_bytes;
    const uint64_t weight_bytes = down_offset + down_bytes;
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());
    const float gate_scale = 0.75f, up_scale = 1.125f, down_scale = 0.625f;
    void *weights_raw = NULL;
    float *x_host = NULL, *out_host = NULL, *a = NULL, *b = NULL;
    q36_gpu_tensor *x = NULL, *gate = NULL, *up = NULL, *mid = NULL;
    q36_gpu_tensor *shared = NULL, *scalar = NULL, *sep = NULL, *fused = NULL;
    q36_gpu_tensor *fused_mid = NULL;

    TEST_ASSERT(sizeof(test_q8_0_block) == 34u);
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;
    memset(weights_raw, 0, (size_t)weight_alloc);
    for (uint32_t bank = 0; bank < 3u; bank++) {
        uint64_t offset = bank == 0u ? 0u : bank == 1u ? up_offset : down_offset;
        uint32_t rows = bank < 2u ? mid_dim : out_dim;
        uint32_t blocks = bank < 2u ? in_blocks : mid_blocks;
        test_q8_0_block *w = (test_q8_0_block *)((unsigned char *)weights_raw + offset);
        for (uint32_t row = 0; row < rows; row++) {
            for (uint32_t block = 0; block < blocks; block++) {
                test_q8_0_block *q = &w[(uint64_t)row * blocks + block];
                q->d = test_float_to_f16(0.003f * (float)(1u + (row + block + bank) % 9u));
                for (uint32_t i = 0; i < 32u; i++)
                    q->qs[i] = (int8_t)((int)((row * 7u + block * 11u + i * 5u + bank) % 31u) - 15);
            }
        }
    }
    x_host = malloc((size_t)in_dim * sizeof(float));
    out_host = malloc((size_t)out_dim * sizeof(float));
    a = malloc((size_t)out_dim * sizeof(float));
    b = malloc((size_t)out_dim * sizeof(float));
    TEST_ASSERT(x_host && out_host && a && b);
    if (!x_host || !out_host || !a || !b) goto done;
    for (uint32_t i = 0; i < in_dim; i++)
        x_host[i] = (float)((int)((i * 13u + 3u) % 53u) - 26) / 37.0f;
    for (uint32_t i = 0; i < out_dim; i++)
        out_host[i] = (float)((int)((i * 17u + 1u) % 41u) - 20) / 29.0f;

    x = q36_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    gate = q36_gpu_tensor_alloc((uint64_t)mid_dim * sizeof(float));
    up = q36_gpu_tensor_alloc((uint64_t)mid_dim * sizeof(float));
    mid = q36_gpu_tensor_alloc((uint64_t)mid_dim * sizeof(float));
    fused_mid = q36_gpu_tensor_alloc((uint64_t)mid_dim * sizeof(float));
    shared = q36_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    scalar = q36_gpu_tensor_alloc(sizeof(float));
    sep = q36_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    fused = q36_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    TEST_ASSERT(x && gate && up && mid && fused_mid && shared && scalar && sep && fused);
    if (!x || !gate || !up || !mid || !fused_mid || !shared || !scalar || !sep || !fused) goto done;
    float scalar_host = -0.375f;
    TEST_ASSERT(q36_gpu_tensor_write(x, 0, x_host, (uint64_t)in_dim * sizeof(float)) != 0);
    TEST_ASSERT(q36_gpu_tensor_write(scalar, 0, &scalar_host, sizeof(scalar_host)) != 0);
    TEST_ASSERT(q36_gpu_tensor_write(sep, 0, out_host, (uint64_t)out_dim * sizeof(float)) != 0);
    TEST_ASSERT(q36_gpu_tensor_write(fused, 0, out_host, (uint64_t)out_dim * sizeof(float)) != 0);
    TEST_ASSERT(q36_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    TEST_ASSERT(q36_gpu_matmul_q8_0_scaled_tensor(gate, weights_raw, weight_alloc, 0,
                                                  in_dim, mid_dim, x, 1, gate_scale) != 0);
    TEST_ASSERT(q36_gpu_matmul_q8_0_scaled_tensor(up, weights_raw, weight_alloc, up_offset,
                                                  in_dim, mid_dim, x, 1, up_scale) != 0);
    TEST_ASSERT(q36_gpu_swiglu_tensor(mid, gate, up, mid_dim, 0.0f, 1.0f) != 0);
    TEST_ASSERT(q36_gpu_matmul_q8_0_scaled_tensor(shared, weights_raw, weight_alloc, down_offset,
                                                  mid_dim, out_dim, mid, 1, down_scale) != 0);
    TEST_ASSERT(q36_gpu_ffn_tail_tensor(sep, shared, scalar, out_dim, 1) != 0);
    TEST_ASSERT(q36_gpu_shared_ffn_decode_tensor(
                    fused, fused_mid, x, scalar, weights_raw, weight_alloc,
                    0, up_offset, down_offset, in_dim, mid_dim, out_dim,
                    gate_scale, up_scale, down_scale) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(sep, 0, a, (uint64_t)out_dim * sizeof(float)) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(fused, 0, b, (uint64_t)out_dim * sizeof(float)) != 0);
#ifdef Q36_METAL
    {
        float max_rel = 0.0f;
        for (uint32_t i = 0; i < out_dim; i++) {
            float den = fabsf(a[i]) > 1.0f ? fabsf(a[i]) : 1.0f;
            float rel = fabsf(a[i] - b[i]) / den;
            if (rel > max_rel) max_rel = rel;
        }
        fprintf(stderr, "q36-test: Metal shared FFN fused max_rel=%g\n",
                (double)max_rel);
        /*
         * The fused Metal decode kernel uses a different simdgroup reduction
         * order from the component matvec kernels.  Require close numeric
         * agreement; bit identity is neither expected nor portable across
         * Apple GPU generations.
         */
        TEST_ASSERT(max_rel < 1.0e-4f);
    }
#else
    TEST_ASSERT(memcmp(a, b, (size_t)out_dim * sizeof(float)) == 0);
    {
        enum { rows = 8 };
        float inputs[rows * in_dim], seeds[rows * out_dim], scalars[rows];
        float want[rows * out_dim], got[rows * out_dim];
        q36_gpu_tensor *tx = q36_gpu_tensor_alloc(sizeof(inputs));
        q36_gpu_tensor *tm = q36_gpu_tensor_alloc((uint64_t)rows * mid_dim * sizeof(float));
        q36_gpu_tensor *ts = q36_gpu_tensor_alloc(sizeof(scalars));
        q36_gpu_tensor *ty = q36_gpu_tensor_alloc(sizeof(seeds));
        TEST_ASSERT(tx && tm && ts && ty);
        if (tx && tm && ts && ty) {
            for (unsigned t = 0; t < rows; t++) {
                scalars[t] = (float)t * 0.25f - 0.75f;
                for (unsigned i = 0; i < in_dim; i++)
                    inputs[t * in_dim + i] = x_host[i] * (0.5f + t * 0.125f);
                for (unsigned i = 0; i < out_dim; i++)
                    seeds[t * out_dim + i] = out_host[i] + t * 0.03125f;
                TEST_ASSERT(q36_gpu_tensor_write(x, 0, inputs + t * in_dim, in_dim * sizeof(float)) != 0);
                TEST_ASSERT(q36_gpu_tensor_write(scalar, 0, scalars + t, sizeof(float)) != 0);
                TEST_ASSERT(q36_gpu_tensor_write(fused, 0, seeds + t * out_dim, out_dim * sizeof(float)) != 0);
                TEST_ASSERT(q36_gpu_shared_ffn_decode_tensor(fused, fused_mid, x, scalar,
                    weights_raw, weight_alloc, 0, up_offset, down_offset,
                    in_dim, mid_dim, out_dim, gate_scale, up_scale, down_scale) != 0);
                TEST_ASSERT(q36_gpu_tensor_read(fused, 0, want + t * out_dim, out_dim * sizeof(float)) != 0);
            }
            TEST_ASSERT(q36_gpu_tensor_write(tx, 0, inputs, sizeof(inputs)) != 0);
            TEST_ASSERT(q36_gpu_tensor_write(ts, 0, scalars, sizeof(scalars)) != 0);
            for (unsigned n = 2; n <= rows; n++) {
                TEST_ASSERT(q36_gpu_tensor_write(ty, 0, seeds, sizeof(seeds)) != 0);
                int ok = q36_gpu_shared_ffn_rows_tensor(ty, tm, tx, ts, weights_raw, weight_alloc,
                    0, up_offset, down_offset, in_dim, mid_dim, out_dim,
                    gate_scale, up_scale, down_scale, n);
                if (!ok && n == 2u) {
                    test_skip("vulkan-shared-ffn-rows", "batched shared expert unavailable");
                    break;
                }
                TEST_ASSERT(ok != 0);
                TEST_ASSERT(q36_gpu_tensor_read(ty, 0, got, sizeof(got)) != 0);
                TEST_ASSERT(memcmp(got, want, n * out_dim * sizeof(float)) == 0);
                TEST_ASSERT(memcmp(got + n * out_dim, seeds + n * out_dim,
                    (rows - n) * out_dim * sizeof(float)) == 0);
            }
        }
        q36_gpu_tensor_free(tx);
        q36_gpu_tensor_free(tm);
        q36_gpu_tensor_free(ts);
        q36_gpu_tensor_free(ty);
    }
#endif

done:
    q36_gpu_tensor_free(x);
    q36_gpu_tensor_free(gate);
    q36_gpu_tensor_free(up);
    q36_gpu_tensor_free(mid);
    q36_gpu_tensor_free(fused_mid);
    q36_gpu_tensor_free(shared);
    q36_gpu_tensor_free(scalar);
    q36_gpu_tensor_free(sep);
    q36_gpu_tensor_free(fused);
    free(x_host);
    free(out_host);
    free(a);
    free(b);
    free(weights_raw);
#endif
}

#ifndef Q36_NO_GPU
enum {
    TEST_QK_K = 256,
    TEST_K_SCALE_SIZE = 12,
    TEST_TENSOR_Q2_K = 10,
    TEST_TENSOR_Q4_K = 12,
    TEST_TENSOR_Q5_K = 13,
    TEST_TENSOR_Q6_K = 14,
};

typedef struct {
    uint8_t scales[TEST_QK_K / 16];
    uint8_t qs[TEST_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} test_block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[TEST_K_SCALE_SIZE];
    uint8_t qs[TEST_QK_K / 2];
} test_block_q4_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[TEST_K_SCALE_SIZE];
    uint8_t qh[TEST_QK_K / 8];
    uint8_t qs[TEST_QK_K / 2];
} test_block_q5_K;

typedef struct {
    uint8_t ql[TEST_QK_K / 2];
    uint8_t qh[TEST_QK_K / 4];
    int8_t scales[TEST_QK_K / 16];
    uint16_t d;
} test_block_q6_K;

typedef struct {
    float d;
    float dmin;
    int8_t qs[TEST_QK_K];
    int16_t bsums[TEST_QK_K / 16];
} test_block_q8_K;


static float test_f16_to_float(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static float test_sigmoid_stable(float x);

static void test_quantize_q8_k(const float *x, test_block_q8_K *y) {
    float maxv = 0.0f;
    float amax = 0.0f;
    for (uint32_t i = 0; i < TEST_QK_K; i++) {
        float ax = fabsf(x[i]);
        if (ax > amax) {
            amax = ax;
            maxv = x[i];
        }
    }

    if (amax == 0.0f) {
        memset(y, 0, sizeof(*y));
        return;
    }

    const float iscale = -127.0f / maxv;
    y->d = 1.0f / iscale;
    y->dmin = 0.0f;
    for (uint32_t i = 0; i < TEST_QK_K; i++) {
        int v = (int)nearbyintf(iscale * x[i]);
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        y->qs[i] = (int8_t)v;
    }
    for (uint32_t j = 0; j < TEST_QK_K / 16; j++) {
        int sum = 0;
        for (uint32_t i = 0; i < 16; i++) sum += y->qs[j * 16u + i];
        y->bsums[j] = (int16_t)sum;
    }
}

static void test_k4_scale_min(const uint8_t *s, uint32_t j, uint32_t *scale, uint32_t *mn) {
    const uint32_t s0 = s[j];
    if (j < 4u) {
        *scale = s0 & 63u;
        *mn = s[4u + j] & 63u;
    } else {
        const uint32_t sj4 = s[j + 4u];
        const uint32_t sjm4 = s[j - 4u];
        *scale = (sj4 & 15u) | ((sjm4 >> 6) << 4);
        *mn = (sj4 >> 4) | ((s0 >> 6) << 4);
    }
}

static int test_q2_value(const test_block_q2_K *x, uint32_t idx) {
    uint32_t in = idx & 127u;
    uint32_t qoff = (idx >> 7u) * 32u + (in & 31u);
    uint32_t shift = (in >> 5u) * 2u;
    return (int)((x->qs[qoff] >> shift) & 3u);
}

static float test_dot_q2_k_q8_k(const test_block_q2_K *x, const test_block_q8_K *y) {
    const float d = test_f16_to_float(x->d) * y->d;
    const float dmin = -test_f16_to_float(x->dmin) * y->d;
    float lane[8] = {0};

    for (uint32_t k = 0; k < 8u; k++) {
        int minprod = (int)(x->scales[2u * k] >> 4) * (int)y->bsums[2u * k]
                    + (int)(x->scales[2u * k + 1u] >> 4) * (int)y->bsums[2u * k + 1u];
        int isum = 0;
        lane[k] = fmaf(dmin, (float)minprod, lane[k]);
        for (uint32_t half = 0; half < 2u; half++) {
            for (uint32_t s = 0; s < 4u; s++) {
                uint32_t group16 = half * 8u + 2u * s + (k >= 4u ? 1u : 0u);
                uint32_t e = half * 128u + s * 32u + (k < 4u ? 4u * k : 16u + 4u * (k - 4u));
                int s4 = test_q2_value(x, e + 0u) * (int)y->qs[e + 0u]
                       + test_q2_value(x, e + 1u) * (int)y->qs[e + 1u]
                       + test_q2_value(x, e + 2u) * (int)y->qs[e + 2u]
                       + test_q2_value(x, e + 3u) * (int)y->qs[e + 3u];
                isum += (int)(x->scales[group16] & 15u) * s4;
            }
        }
        lane[k] = fmaf(d, (float)isum, lane[k]);
    }

    return ((lane[0] + lane[4]) + (lane[2] + lane[6])) +
           ((lane[1] + lane[5]) + (lane[3] + lane[7]));
}

static float test_dot_q4_k_q8_k(const test_block_q4_K *x, const test_block_q8_K *y) {
    const float d = test_f16_to_float(x->d) * y->d;
    const float dmin = test_f16_to_float(x->dmin) * y->d;
    int isum = 0;
    int summs = 0;

    for (uint32_t g = 0; g < 8; g++) {
        uint32_t sc;
        uint32_t mn;
        int group = 0;
        test_k4_scale_min(x->scales, g, &sc, &mn);
        const uint8_t *q = x->qs + (g >> 1) * 32u;
        for (uint32_t i = 0; i < 32; i++) {
            uint32_t v = ((g & 1u) == 0u) ? (q[i] & 15u) : (q[i] >> 4);
            group += (int)v * (int)y->qs[g * 32u + i];
        }
        isum += (int)sc * group;
        summs += (int)mn * ((int)y->bsums[2u * g + 0u] + (int)y->bsums[2u * g + 1u]);
    }

    return d * (float)isum - dmin * (float)summs;
}

static float test_dot_q5_k_q8_k(const test_block_q5_K *x, const test_block_q8_K *y) {
    const float d = test_f16_to_float(x->d) * y->d;
    const float dmin = test_f16_to_float(x->dmin) * y->d;
    int isum = 0;
    int summs = 0;

    for (uint32_t g = 0; g < 8; g++) {
        uint32_t sc;
        uint32_t mn;
        int group = 0;
        test_k4_scale_min(x->scales, g, &sc, &mn);
        const uint8_t *q = x->qs + (g >> 1) * 32u;
        for (uint32_t i = 0; i < 32; i++) {
            uint32_t lo = ((g & 1u) == 0u) ? (q[i] & 15u) : (q[i] >> 4);
            uint32_t hi = (x->qh[i] >> g) & 1u;
            group += (int)(lo | (hi << 4)) * (int)y->qs[g * 32u + i];
        }
        isum += (int)sc * group;
        summs += (int)mn * ((int)y->bsums[2u * g + 0u] + (int)y->bsums[2u * g + 1u]);
    }

    return d * (float)isum - dmin * (float)summs;
}

static int test_q6_value(const test_block_q6_K *x, uint32_t idx) {
    const uint32_t half = idx >> 7;
    const uint32_t in = idx & 127u;
    uint32_t ql_off;
    uint32_t qh_off;
    uint32_t shift;
    uint32_t low;

    if (in < 32u) {
        ql_off = half * 64u + in;
        qh_off = half * 32u + in;
        shift = 0u;
        low = x->ql[ql_off] & 15u;
    } else if (in < 64u) {
        uint32_t l = in - 32u;
        ql_off = half * 64u + 32u + l;
        qh_off = half * 32u + l;
        shift = 2u;
        low = x->ql[ql_off] & 15u;
    } else if (in < 96u) {
        uint32_t l = in - 64u;
        ql_off = half * 64u + l;
        qh_off = half * 32u + l;
        shift = 4u;
        low = x->ql[ql_off] >> 4;
    } else {
        uint32_t l = in - 96u;
        ql_off = half * 64u + 32u + l;
        qh_off = half * 32u + l;
        shift = 6u;
        low = x->ql[ql_off] >> 4;
    }

    uint32_t hi = (x->qh[qh_off] >> shift) & 3u;
    return (int)(low | (hi << 4)) - 32;
}

static float test_dot_q6_k_q8_k(const test_block_q6_K *x, const test_block_q8_K *y) {
    const float d = test_f16_to_float(x->d) * y->d;
    int isum = 0;
    for (uint32_t g = 0; g < 16; g++) {
        int group = 0;
        for (uint32_t i = 0; i < 16; i++) {
            uint32_t idx = g * 16u + i;
            group += test_q6_value(x, idx) * (int)y->qs[idx];
        }
        isum += (int)x->scales[g] * group;
    }
    return d * (float)isum;
}

static void test_fill_q2_k(test_block_q2_K *b, uint32_t row) {
    memset(b, 0, sizeof(*b));
    for (uint32_t i = 0; i < TEST_QK_K / 16; i++) {
        uint8_t scale = (uint8_t)(1u + ((row * 7u + i * 5u) % 15u));
        uint8_t mn = (uint8_t)(1u + ((row * 11u + i * 3u) % 15u));
        b->scales[i] = (uint8_t)(scale | (mn << 4));
    }
    for (uint32_t i = 0; i < TEST_QK_K / 4; i++) b->qs[i] = (uint8_t)((row * 17u + i * 13u) & 255u);
    b->d = test_float_to_f16(0.0011f * (float)(row + 1u));
    b->dmin = test_float_to_f16(0.0004f * (float)(row + 2u));
}

static void test_fill_q4_k(test_block_q4_K *b, uint32_t row) {
    memset(b, 0, sizeof(*b));
    b->d = test_float_to_f16(0.001f * (float)(row + 1u));
    b->dmin = test_float_to_f16(0.0003f * (float)(row + 2u));
    for (uint32_t i = 0; i < TEST_K_SCALE_SIZE; i++) b->scales[i] = (uint8_t)(17u + ((row * 11u + i * 7u) & 63u));
    for (uint32_t i = 0; i < TEST_QK_K / 2; i++) b->qs[i] = (uint8_t)((row * 19u + i * 5u) & 255u);
}

static void test_fill_q5_k(test_block_q5_K *b, uint32_t row) {
    memset(b, 0, sizeof(*b));
    b->d = test_float_to_f16(0.0008f * (float)(row + 1u));
    b->dmin = test_float_to_f16(0.0002f * (float)(row + 3u));
    for (uint32_t i = 0; i < TEST_K_SCALE_SIZE; i++) b->scales[i] = (uint8_t)(9u + ((row * 13u + i * 5u) & 63u));
    for (uint32_t i = 0; i < TEST_QK_K / 8; i++) b->qh[i] = (uint8_t)((row * 23u + i * 17u) & 255u);
    for (uint32_t i = 0; i < TEST_QK_K / 2; i++) b->qs[i] = (uint8_t)((row * 3u + i * 29u) & 255u);
}

static void test_fill_q6_k(test_block_q6_K *b, uint32_t row) {
    memset(b, 0, sizeof(*b));
    for (uint32_t i = 0; i < TEST_QK_K / 2; i++) b->ql[i] = (uint8_t)((row * 31u + i * 7u) & 255u);
    for (uint32_t i = 0; i < TEST_QK_K / 4; i++) b->qh[i] = (uint8_t)((row * 5u + i * 11u) & 255u);
    for (uint32_t i = 0; i < TEST_QK_K / 16; i++) b->scales[i] = (int8_t)((int)((row * 3u + i * 5u) % 17u) - 8);
    b->d = test_float_to_f16(0.0007f * (float)(row + 1u));
}

static void test_vulkan_kquant_one(uint32_t type, uint64_t row_bytes) {
    const uint32_t in_dim = TEST_QK_K;
    const uint32_t out_dim = 9;
    const uint64_t weight_bytes = row_bytes * out_dim;
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;
    memset(weights_raw, 0, (size_t)weight_alloc);

    for (uint32_t o = 0; o < out_dim; o++) {
        uint8_t *row = (uint8_t *)weights_raw + (uint64_t)o * row_bytes;
        if (type == TEST_TENSOR_Q2_K) test_fill_q2_k((test_block_q2_K *)row, o);
        else if (type == TEST_TENSOR_Q4_K) test_fill_q4_k((test_block_q4_K *)row, o);
        else if (type == TEST_TENSOR_Q5_K) test_fill_q5_k((test_block_q5_K *)row, o);
        else if (type == TEST_TENSOR_Q6_K) test_fill_q6_k((test_block_q6_K *)row, o);
    }

    q36_gpu_tensor *x = q36_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    q36_gpu_tensor *out = q36_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    q36_gpu_tensor *q8 = q36_gpu_tensor_alloc(sizeof(test_block_q8_K));
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    TEST_ASSERT(q8 != NULL);
    if (!x || !out || !q8) {
        q36_gpu_tensor_free(q8);
        q36_gpu_tensor_free(x);
        q36_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)in_dim * sizeof(float));
    float *out_host = malloc((size_t)out_dim * sizeof(float));
    test_block_q8_K *xq = malloc(sizeof(*xq));
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    TEST_ASSERT(xq != NULL);
    if (!x_host || !out_host || !xq) {
        free(x_host);
        free(out_host);
        free(xq);
        q36_gpu_tensor_free(q8);
        q36_gpu_tensor_free(x);
        q36_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t i = 0; i < in_dim; i++) {
        x_host[i] = (float)((int)((i * 7u) % 23u) - 11) / 13.0f;
    }
    test_quantize_q8_k(x_host, xq);

    TEST_ASSERT(q36_gpu_tensor_write(x, 0, x_host, (uint64_t)in_dim * sizeof(float)) != 0);
    TEST_ASSERT(q36_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    TEST_ASSERT(q36_gpu_quantize_q8_k_tensor(q8, x, in_dim, 1) != 0);
    {
        test_block_q8_K q8_host;
        TEST_ASSERT(q36_gpu_tensor_read(q8, 0, &q8_host, sizeof(q8_host)) != 0);
        TEST_ASSERT(memcmp(&q8_host, xq, sizeof(q8_host)) == 0);
    }
    TEST_ASSERT(q36_gpu_matmul_k_quant_tensor(out, weights_raw, weight_alloc, 0,
                                              type, in_dim, out_dim, x, 1) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(out, 0, out_host, (uint64_t)out_dim * sizeof(float)) != 0);

    float max_abs = 0.0f;
    for (uint32_t o = 0; o < out_dim; o++) {
        float ref = 0.0f;
        const uint8_t *row = (const uint8_t *)weights_raw + (uint64_t)o * row_bytes;
        if (type == TEST_TENSOR_Q2_K) ref = test_dot_q2_k_q8_k((const test_block_q2_K *)row, xq);
        else if (type == TEST_TENSOR_Q4_K) ref = test_dot_q4_k_q8_k((const test_block_q4_K *)row, xq);
        else if (type == TEST_TENSOR_Q5_K) ref = test_dot_q5_k_q8_k((const test_block_q5_K *)row, xq);
        else if (type == TEST_TENSOR_Q6_K) ref = test_dot_q6_k_q8_k((const test_block_q6_K *)row, xq);
        float err = fabsf(out_host[o] - ref);
        if (err > max_abs) max_abs = err;
    }
    TEST_ASSERT(max_abs < 2.0e-3f);

    memset(out_host, 0, (size_t)out_dim * sizeof(float));
    TEST_ASSERT(q36_gpu_matmul_k_quant_q8_scaled_tensor(out, weights_raw, weight_alloc, 0,
                                                        type, in_dim, out_dim, q8, 1, 1.0f) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(out, 0, out_host, (uint64_t)out_dim * sizeof(float)) != 0);

    max_abs = 0.0f;
    for (uint32_t o = 0; o < out_dim; o++) {
        float ref = 0.0f;
        const uint8_t *row = (const uint8_t *)weights_raw + (uint64_t)o * row_bytes;
        if (type == TEST_TENSOR_Q2_K) ref = test_dot_q2_k_q8_k((const test_block_q2_K *)row, xq);
        else if (type == TEST_TENSOR_Q4_K) ref = test_dot_q4_k_q8_k((const test_block_q4_K *)row, xq);
        else if (type == TEST_TENSOR_Q5_K) ref = test_dot_q5_k_q8_k((const test_block_q5_K *)row, xq);
        else if (type == TEST_TENSOR_Q6_K) ref = test_dot_q6_k_q8_k((const test_block_q6_K *)row, xq);
        {
            float err = fabsf(out_host[o] - ref);
            if (err > max_abs) max_abs = err;
        }
    }
    TEST_ASSERT(max_abs < 2.0e-3f);

    free(xq);
    free(x_host);
    free(out_host);
    q36_gpu_tensor_free(q8);
    q36_gpu_tensor_free(x);
    q36_gpu_tensor_free(out);
    free(weights_raw);
}
#endif

static void test_vulkan_kquant_matvec(void) {
#ifdef Q36_NO_GPU
    test_skip("vulkan-kernels", "CPU-only build");
    return;
#else
    test_vulkan_kquant_one(TEST_TENSOR_Q2_K, sizeof(test_block_q2_K));
    test_vulkan_kquant_one(TEST_TENSOR_Q4_K, sizeof(test_block_q4_K));
    test_vulkan_kquant_one(TEST_TENSOR_Q5_K, sizeof(test_block_q5_K));
    test_vulkan_kquant_one(TEST_TENSOR_Q6_K, sizeof(test_block_q6_K));
#endif
}

#if !defined(Q36_NO_GPU) && !defined(Q36_METAL)
static void test_vulkan_decode_one(uint32_t type, uint32_t block_bytes,
                                  uint32_t in_dim, uint32_t out_dim) {
    /* An odd padded row count selects the existing generic decode kernel.
     * The first 1024 rows must be identical, including negative scales. */
    enum { tokens = 1 };
    const uint32_t ref_rows = out_dim + 3u;
    const uint32_t blocks = in_dim / (type == 20 ? 32u : 256u);
    const size_t ref_bytes = (size_t)tokens * ref_rows * sizeof(float);
    const uint32_t row_bytes = blocks * block_bytes;
    const size_t x_bytes = (size_t)tokens * in_dim * sizeof(float);
    const size_t y_bytes = (size_t)(out_dim + 1u) * sizeof(float);
    const size_t weight_bytes = (size_t)test_round_up_u64(
        (uint64_t)ref_rows * row_bytes, (uint64_t)getpagesize());
    uint8_t *weights = NULL;
    float *input = malloc(x_bytes), *got = malloc(y_bytes), *want = malloc(ref_bytes);
    q36_gpu_tensor *x = NULL, *out = NULL, *ref = NULL;
    TEST_ASSERT(posix_memalign((void **)&weights, (size_t)getpagesize(), weight_bytes) == 0);
    TEST_ASSERT(weights && input && got && want);
    if (!weights || !input || !got || !want) goto cleanup;
    memset(weights, 0, weight_bytes);
    for (uint32_t row = 0; row < ref_rows; row++) {
        for (uint32_t block = 0; block < blocks; block++) {
            uint8_t *p = weights + row * row_bytes + block * block_bytes;
            uint16_t d = test_float_to_f16(0.0001f * (float)(1 + row % 7 + block));
            for (uint32_t i = 0; i < block_bytes; i++)
                p[i] = (uint8_t)(row * 29u + block * 71u + i * 13u);
            uint32_t d_off = type == 10 ? 80 : type == 11 ? 108 : type == 14 ? 208 : 0;
            if ((row + block) & 1u) d ^= 0x8000u;
            memcpy(p + d_off, &d, sizeof(d));
            if (type == 10 || type == 12 || type == 13)
                memcpy(p + d_off + 2, &d, sizeof(d));
        }
    }
    for (uint32_t i = 0; i < tokens * in_dim; i++)
        input[i] = sinf((float)i * 0.031f) * (0.25f + (float)(i % 11) * 0.03125f);
    TEST_ASSERT(q36_gpu_init() != 0);
    q36_gpu_set_dense_model(true);
    x = q36_gpu_tensor_alloc(x_bytes);
    out = q36_gpu_tensor_alloc(y_bytes);
    ref = q36_gpu_tensor_alloc(ref_bytes);
    TEST_ASSERT(x && out && ref);
    if (!x || !out || !ref) goto cleanup;
    TEST_ASSERT(q36_gpu_set_model_map(weights, weight_bytes) != 0);
    TEST_ASSERT(q36_gpu_tensor_write(x, 0, input, x_bytes) != 0);
    int (*matmul)(q36_gpu_tensor *, const void *, uint64_t, uint64_t,
                  uint32_t, uint64_t, uint64_t, const q36_gpu_tensor *, uint64_t, float) =
        type < 16 ? q36_gpu_matmul_k_quant_scaled_tensor : q36_gpu_matmul_iq_quant_scaled_tensor;
    const float scales[] = {0.75f, -0.5f};
    for (unsigned si = 0; si < sizeof(scales) / sizeof(scales[0]); si++) {
        for (uint32_t i = 0; i <= out_dim; i++) got[i] = 12345.0f;
        TEST_ASSERT(q36_gpu_tensor_write(out, 0, got, y_bytes) != 0);
        TEST_ASSERT(matmul(ref, weights, weight_bytes, 0, type,
                           in_dim, ref_rows, x, 1, scales[si]) != 0);
        TEST_ASSERT(matmul(out, weights, weight_bytes, 0, type,
                           in_dim, out_dim, x, 1, scales[si]) != 0);
        TEST_ASSERT(q36_gpu_tensor_read(ref, 0, want, ref_bytes) != 0);
        TEST_ASSERT(q36_gpu_tensor_read(out, 0, got, y_bytes) != 0);
        bool nonzero = false;
        for (uint32_t i = 0; i < out_dim; i++) {
            TEST_ASSERT(isfinite(got[i]) && isfinite(want[i]));
            TEST_ASSERT(memcmp(got + i, want + i, sizeof(float)) == 0);
            nonzero = nonzero || got[i] != 0.0f;
        }
        TEST_ASSERT(nonzero);
        TEST_ASSERT(got[out_dim] == 12345.0f);
    }
cleanup:
    q36_gpu_tensor_free(x);
    q36_gpu_tensor_free(out);
    q36_gpu_tensor_free(ref);
    q36_gpu_set_dense_model(false);
    q36_gpu_cleanup();
    free(weights);
    free(input);
    free(got);
    free(want);
}

static void test_vulkan_small_dims_mode(uint32_t type, uint32_t block_bytes,
                                  uint32_t out_dim, uint32_t in_dim, bool exact_prefill) {
    /* Nine tokens select the large tile as an independent oracle. Odd block
     * counts and row counts cross alignment and every small tile boundary.
     * Q8 uses padded reference rows to select the original wide kernel. */
    const uint32_t tokens = exact_prefill ? 129u : 9u;
    const uint32_t blocks = (type == 8 || type == 20) ? in_dim / 32u : in_dim / 256u;
    const uint32_t ref_rows = type == 8 && out_dim < 131u ? 131u : out_dim;
    const size_t ref_bytes = (size_t)tokens * ref_rows * sizeof(float);
    const uint32_t row_bytes = blocks * block_bytes;
    const size_t x_bytes = (size_t)tokens * in_dim * sizeof(float);
    const size_t y_bytes = (size_t)tokens * out_dim * sizeof(float);
    const size_t weight_bytes = (size_t)test_round_up_u64(
        (uint64_t)ref_rows * row_bytes, (uint64_t)getpagesize());
    uint8_t *weights = NULL;
    float *input = malloc(x_bytes), *got = malloc(y_bytes), *want = malloc(ref_bytes);
    q36_gpu_tensor *x = NULL, *out = NULL, *ref = NULL;
    TEST_ASSERT(posix_memalign((void **)&weights, (size_t)getpagesize(), weight_bytes) == 0);
    TEST_ASSERT(weights && input && got && want);
    if (!weights || !input || !got || !want) goto cleanup;
    memset(weights, 0, weight_bytes);
    for (uint32_t row = 0; row < ref_rows; row++) {
        for (uint32_t block = 0; block < blocks; block++) {
            uint8_t *p = weights + row * row_bytes + block * block_bytes;
            uint16_t d = test_float_to_f16(0.0001f * (float)(1 + row % 7 + block));
            if ((type == 18 || type == 21 || type == 23) && in_dim >= 5120u && ((row + block) & 1u)) d ^= 0x8000u;
            for (uint32_t i = 0; i < block_bytes; i++)
                p[i] = (uint8_t)(row * 29u + block * 71u + i * 13u);
            uint32_t d_off = type == 10 ? 80 : type == 11 ? 108 : type == 14 ? 208 : 0;
            memcpy(p + d_off, &d, sizeof(d));
            if (type == 10 || type == 12 || type == 13)
                memcpy(p + d_off + 2, &d, sizeof(d));
        }
    }
    for (uint32_t i = 0; i < tokens * in_dim; i++) {
        input[i] = sinf((float)i * 0.031f) * (0.25f + (float)(i % 11) * 0.03125f);
        if ((type == 18 || type == 21 || type == 23) && in_dim >= 5120u && i / in_dim < 2u)
            input[i] = i / in_dim == 0u ? 1.0f : -1.0f;
    }
    TEST_ASSERT(q36_gpu_init() != 0);
    q36_gpu_set_dense_model(true);
    x = q36_gpu_tensor_alloc(x_bytes);
    out = q36_gpu_tensor_alloc(y_bytes);
    ref = q36_gpu_tensor_alloc(ref_bytes);
    TEST_ASSERT(x && out && ref);
    if (!x || !out || !ref) goto cleanup;
    TEST_ASSERT(q36_gpu_set_model_map(weights, weight_bytes) != 0);
    TEST_ASSERT(q36_gpu_tensor_write(x, 0, input, x_bytes) != 0);
    int (*matmul)(q36_gpu_tensor *, const void *, uint64_t, uint64_t,
                  uint32_t, uint64_t, uint64_t, const q36_gpu_tensor *, uint64_t, float) =
        type < 16 ? q36_gpu_matmul_k_quant_scaled_tensor : q36_gpu_matmul_iq_quant_scaled_tensor;
    if (exact_prefill) {
        q36_gpu_set_micro_batch(false);
        for (uint32_t t = 0; t < tokens; t++) {
            q36_gpu_tensor *xi = q36_gpu_tensor_view(x,(uint64_t)t*in_dim*sizeof(float),in_dim*sizeof(float));
            q36_gpu_tensor *yi = q36_gpu_tensor_view(ref,(uint64_t)t*out_dim*sizeof(float),out_dim*sizeof(float));
            TEST_ASSERT(xi && yi);
            if (xi && yi) TEST_ASSERT(type == 8 ?
                q36_gpu_matmul_q8_0_scaled_tensor(yi,weights,weight_bytes,0,in_dim,out_dim,xi,1,-0.5f) :
                matmul(yi,weights,weight_bytes,0,type,in_dim,out_dim,xi,1,-0.5f));
            q36_gpu_tensor_free(xi); q36_gpu_tensor_free(yi);
        }
        TEST_ASSERT(q36_gpu_tensor_read(ref,0,want,y_bytes));
        q36_gpu_set_dense_prefill_exact(true);
        const uint32_t counts[] = {2,7,8,9,15,16,17,31,32,33,64,128,129};
        for (unsigned c = 0; c < sizeof(counts)/sizeof(*counts); c++) {
            uint32_t n = counts[c];
            for (uint32_t i = 0; i < tokens*out_dim; i++) got[i] = 12345.0f;
            TEST_ASSERT(q36_gpu_tensor_write(out,0,got,y_bytes));
            TEST_ASSERT(type == 8 ?
                q36_gpu_matmul_q8_0_scaled_tensor(out,weights,weight_bytes,0,in_dim,out_dim,x,n,-0.5f) :
                matmul(out,weights,weight_bytes,0,type,in_dim,out_dim,x,n,-0.5f));
            TEST_ASSERT(q36_gpu_tensor_read(out,0,got,y_bytes));
            if (memcmp(got,want,(size_t)n*out_dim*sizeof(float))) {
                fprintf(stderr,"q36-test: exact prefill type=%u width=%u rows=%u tokens=%u\n",type,in_dim,out_dim,n);
                TEST_ASSERT(false);
            }
            for (uint32_t i = n*out_dim; i < tokens*out_dim; i++) TEST_ASSERT(got[i] == 12345.0f);
        }
        goto cleanup;
    }
    const float scales[] = {0.75f, -0.5f};
    for (unsigned s = 0; s < sizeof(scales) / sizeof(scales[0]); s++) {
        TEST_ASSERT(type == 8 ?
            q36_gpu_matmul_q8_0_scaled_tensor(ref, weights, weight_bytes, 0,
                in_dim, ref_rows, x, tokens, scales[s]) != 0 :
            matmul(ref, weights, weight_bytes, 0, type,
                in_dim, ref_rows, x, tokens, scales[s]) != 0);
        TEST_ASSERT(q36_gpu_tensor_read(ref, 0, want, ref_bytes) != 0);
        for (uint32_t n = 2; n <= (type == 8 ? 9u : 8u); n++) {
            for (uint32_t i = 0; i < tokens * out_dim; i++) got[i] = 12345.0f;
            TEST_ASSERT(q36_gpu_tensor_write(out, 0, got, y_bytes) != 0);
            TEST_ASSERT(type == 8 ?
                q36_gpu_matmul_q8_0_scaled_tensor(out, weights, weight_bytes, 0,
                    in_dim, out_dim, x, n, scales[s]) != 0 :
                matmul(out, weights, weight_bytes, 0, type,
                    in_dim, out_dim, x, n, scales[s]) != 0);
            TEST_ASSERT(q36_gpu_tensor_read(out, 0, got, y_bytes) != 0);
            bool nonzero = false;
            for (uint32_t i = 0; i < n * out_dim; i++) {
                uint32_t ri = (i / out_dim) * ref_rows + i % out_dim;
                TEST_ASSERT(isfinite(got[i]) && isfinite(want[ri]));
                TEST_ASSERT(memcmp(got + i, want + ri, sizeof(float)) == 0);
                nonzero = nonzero || got[i] != 0.0f;
            }
            TEST_ASSERT(nonzero);
            for (uint32_t i = n * out_dim; i < tokens * out_dim; i++)
                TEST_ASSERT(got[i] == 12345.0f);
        }

        /* The verifier shares weights across tokens but must preserve the
         * single-token accumulation, including non-power-of-two tails. */
        for (uint32_t t = 0; t < 8; t++) {
            q36_gpu_tensor *xi = q36_gpu_tensor_view(x, (uint64_t)t * in_dim * sizeof(float),
                                                     (uint64_t)in_dim * sizeof(float));
            q36_gpu_tensor *yi = q36_gpu_tensor_view(ref, (uint64_t)t * out_dim * sizeof(float),
                                                       (uint64_t)out_dim * sizeof(float));
            TEST_ASSERT(xi && yi);
            if (xi && yi) TEST_ASSERT(type == 8 ?
                q36_gpu_matmul_q8_0_scaled_tensor(yi, weights, weight_bytes, 0,
                    in_dim, out_dim, xi, 1, scales[s]) != 0 :
                matmul(yi, weights, weight_bytes, 0, type,
                    in_dim, out_dim, xi, 1, scales[s]) != 0);
            q36_gpu_tensor_free(xi);
            q36_gpu_tensor_free(yi);
        }
        TEST_ASSERT(q36_gpu_tensor_read(ref, 0, want, 8u * out_dim * sizeof(float)) != 0);
        if (type == 8 || type == 13 || type == 23) {
            uint32_t blocks = (out_dim + 63u) / 64u;
            uint32_t *ages = calloc(blocks, sizeof(*ages));
            q36_gpu_tensor *mask = q36_gpu_tensor_alloc((uint64_t)blocks * sizeof(*ages));
            TEST_ASSERT(ages && mask);
            if (ages && mask) {
                for (uint32_t i = 0; i < blocks; i++) ages[i] = (i & 1u) == 0u;
                TEST_ASSERT(q36_gpu_tensor_write(mask, 0, ages, (uint64_t)blocks * sizeof(*ages)) != 0);
                for (uint32_t i = 0; i < tokens * out_dim; i++) got[i] = 12345.0f;
                TEST_ASSERT(q36_gpu_tensor_write(out, 0, got, y_bytes) != 0);
                TEST_ASSERT(q36_gpu_mtp_head_tensor(out, weights, weight_bytes, 0,
                    type, in_dim, out_dim, x, mask, scales[s]) != 0);
                TEST_ASSERT(q36_gpu_tensor_read(out, 0, got, y_bytes) != 0);
                for (uint32_t i = 0; i < out_dim; i++) {
                    if (ages[i / 64u]) {
                        if (type == 8 || type == 13) TEST_ASSERT(memcmp(got + i, want + i, sizeof(float)) == 0);
                        else TEST_ASSERT(fabsf(got[i] - want[i]) <= 1.0e-6f * fmaxf(1.0f, fabsf(want[i])));
                    } else TEST_ASSERT(isinf(got[i]) && got[i] < 0.0f);
                }
                for (uint32_t i = out_dim; i < tokens * out_dim; i++)
                    TEST_ASSERT(got[i] == 12345.0f);
            }
            q36_gpu_tensor_free(mask);
            free(ages);
        }
        q36_gpu_set_micro_batch(true);
        for (uint32_t n = 2; n <= 8; n++) {
            for (uint32_t i = 0; i < tokens * out_dim; i++) got[i] = 12345.0f;
            TEST_ASSERT(q36_gpu_tensor_write(out, 0, got, y_bytes) != 0);
            TEST_ASSERT(type == 8 ?
                q36_gpu_matmul_q8_0_scaled_tensor(out, weights, weight_bytes, 0,
                    in_dim, out_dim, x, n, scales[s]) != 0 :
                matmul(out, weights, weight_bytes, 0, type,
                    in_dim, out_dim, x, n, scales[s]) != 0);
            TEST_ASSERT(q36_gpu_tensor_read(out, 0, got, y_bytes) != 0);
            if (memcmp(got, want, n * out_dim * sizeof(float))) {
                uint32_t i = 0;
                while (i < n * out_dim && got[i] == want[i]) i++;
                fprintf(stderr, "q36-test: verify type=%u rows=%u n=%u at=%u got=%g want=%g\n",
                        type, out_dim, n, i, got[i], want[i]);
                TEST_ASSERT(false);
            }
            for (uint32_t i = n * out_dim; i < tokens * out_dim; i++)
                TEST_ASSERT(got[i] == 12345.0f);
        }
        q36_gpu_set_micro_batch(false);
    }
    if (type == 21 && in_dim >= 5120u) {
        /* Reusing the same Q8 input across prefill/verify must distinguish
         * scaled f16 panels from raw i16 panels in both directions. */
        const uint32_t n = 3;
        const size_t bytes = (size_t)n * out_dim * sizeof(float);
        q36_gpu_tensor *q8 = q36_gpu_tensor_alloc((uint64_t)n * blocks * 296u);
        TEST_ASSERT(q8 != NULL);
        if (q8) {
            TEST_ASSERT(q36_gpu_quantize_q8_k_tensor(q8, x, in_dim, n) != 0);
            TEST_ASSERT(q36_gpu_matmul_iq_quant_q8_scaled_tensor(
                ref, weights, weight_bytes, 0, type, in_dim, out_dim, q8, n, scales[1]) != 0);
            q36_gpu_set_micro_batch(true);
            TEST_ASSERT(q36_gpu_matmul_iq_quant_q8_scaled_tensor(
                out, weights, weight_bytes, 0, type, in_dim, out_dim, q8, n, scales[1]) != 0);
            TEST_ASSERT(q36_gpu_tensor_read(out, 0, got, bytes) != 0);
            TEST_ASSERT(memcmp(got, want, bytes) == 0);
            q36_gpu_set_micro_batch(false);
            TEST_ASSERT(q36_gpu_matmul_iq_quant_q8_scaled_tensor(
                out, weights, weight_bytes, 0, type, in_dim, out_dim, q8, n, scales[1]) != 0);
            TEST_ASSERT(q36_gpu_tensor_read(out, 0, got, bytes) != 0);
            TEST_ASSERT(q36_gpu_tensor_read(ref, 0, want, bytes) != 0);
            TEST_ASSERT(memcmp(got, want, bytes) == 0);
        }
        q36_gpu_tensor_free(q8);
    }
cleanup:
    q36_gpu_tensor_free(x);
    q36_gpu_tensor_free(out);
    q36_gpu_tensor_free(ref);
    q36_gpu_set_dense_model(false);
    q36_gpu_cleanup();
    free(weights);
    free(input);
    free(got);
    free(want);
}

static void test_vulkan_small_dims(uint32_t type, uint32_t block_bytes,
                                   uint32_t out_dim, uint32_t in_dim) {
    test_vulkan_small_dims_mode(type,block_bytes,out_dim,in_dim,false);
}

static void test_vulkan_small_one(uint32_t type, uint32_t block_bytes,
                                  uint32_t out_dim) {
    test_vulkan_small_dims(type, block_bytes, out_dim, type == 8 ? 736u : 768u);
}
#endif

static void test_vulkan_small_tiles(void) {
#if defined(Q36_NO_GPU) || defined(Q36_METAL)
    test_skip("vulkan-small-tiles", "Vulkan-only test");
#else
    const uint32_t exact_types[] = {8,12,13,14,20,23},
        exact_bytes[] = {34,144,176,210,18,136},
        exact_widths[] = {5120,1280,5120,17408,5120,5120};
    for (unsigned i = 0; i < sizeof(exact_types)/sizeof(*exact_types); i++)
        test_vulkan_small_dims_mode(exact_types[i],exact_bytes[i],132,exact_widths[i],true);
    test_vulkan_small_dims_mode(13,176,131,768,true);
    const uint32_t q8_rows[] = {1, 7, 31, 48, 63, 131};
    for (unsigned i = 0; i < sizeof(q8_rows) / sizeof(q8_rows[0]); i++)
        test_vulkan_small_one(8, 34, q8_rows[i]);
    test_vulkan_small_dims(8, 34, 1028, 1056);
    test_vulkan_small_dims(8, 34, 65540, 1024);
    const uint32_t types[] = {10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23},
                   bytes[] = {84, 110, 144, 176, 210, 66, 74, 98, 50, 18, 110, 82, 136};
    for (unsigned i = 0; i < sizeof(types) / sizeof(types[0]); i++) {
        test_vulkan_small_one(types[i], bytes[i], 131);
        test_vulkan_small_one(types[i], bytes[i], 48);
    }
    const uint32_t iq3_shapes[][2] = {
        {5120, 6144}, {5120, 10240}, {5120, 12288}, {5120, 17408},
        {6144, 5120}, {17408, 5120},
    };
    for (unsigned i = 0; i < sizeof(iq3_shapes) / sizeof(iq3_shapes[0]); i++) {
        test_vulkan_small_dims(10, 84, iq3_shapes[i][1], iq3_shapes[i][0]);
        test_vulkan_small_dims(12, 144, iq3_shapes[i][1], iq3_shapes[i][0]);
        test_vulkan_small_dims(14, 210, iq3_shapes[i][1], iq3_shapes[i][0]);
        test_vulkan_small_dims(17, 74, iq3_shapes[i][1], iq3_shapes[i][0]);
        test_vulkan_small_dims(22, 82, iq3_shapes[i][1], iq3_shapes[i][0]);
        test_vulkan_small_dims(18, 98, iq3_shapes[i][1], iq3_shapes[i][0]);
        test_vulkan_small_dims(21, 110, iq3_shapes[i][1], iq3_shapes[i][0]);
        test_vulkan_small_dims(23, 136, iq3_shapes[i][1], iq3_shapes[i][0]);
    }
    test_vulkan_small_dims(23, 136, 1024, 5120);
    test_vulkan_small_dims(19, 50, 1024, 5120);
    test_vulkan_small_dims(19, 50, 17408, 5120);
#endif
}

static void test_vulkan_wide_tiles(uint32_t type, uint32_t block_bytes,
                                    uint32_t tokens, uint32_t out_dim) {
#if defined(Q36_NO_GPU) || defined(Q36_METAL)
    (void)type;
    (void)block_bytes;
    (void)tokens;
    (void)out_dim;
    test_skip("vulkan-wide-tiles", "Vulkan-only test");
#else
    /* Cross both tile edges with an odd number of weight blocks.
     * Split into batches of 128 plus three tokens to keep the F16 prefill
     * arithmetic throughout; a one-token remainder would select F32 decode. */
    const uint32_t in_dim = tokens >= 512 ? 4352 : 768;
    const uint32_t blocks = in_dim / (type == 20 ? 32 : 256);
    const uint32_t row_bytes = blocks * block_bytes;
    const size_t x_bytes = (size_t)tokens * in_dim * sizeof(float);
    const size_t y_bytes = (size_t)tokens * out_dim * sizeof(float);
    const size_t weight_bytes = (size_t)test_round_up_u64(
        (uint64_t)out_dim * row_bytes, (uint64_t)getpagesize());
    uint8_t *weights = NULL;
    float *input = malloc(x_bytes), *got = malloc(y_bytes), *want = malloc(y_bytes);
    q36_gpu_tensor *x = NULL, *out = NULL, *ref = NULL;
    TEST_ASSERT(posix_memalign((void **)&weights, (size_t)getpagesize(), weight_bytes) == 0);
    TEST_ASSERT(weights && input && got && want);
    if (!weights || !input || !got || !want) goto cleanup;
    memset(weights, 0, weight_bytes);
    for (uint32_t row = 0; row < out_dim; row++) {
        for (uint32_t block = 0; block < blocks; block++) {
            uint8_t *p = weights + row * row_bytes + block * block_bytes;
            uint16_t d = test_float_to_f16(0.0001f * (float)(1 + row % 7 + block));
            for (uint32_t i = 0; i < block_bytes; i++)
                p[i] = (uint8_t)(row * 29u + block * 71u + i * 13u);
            uint32_t d_off = type == 10 ? 80 : type == 11 ? 108 : 0;
            memcpy(p + d_off, &d, sizeof(d));
            if (type == 10 || type == 12 || type == 13)
                memcpy(p + d_off + 2, &d, sizeof(d));
        }
    }
    for (uint32_t i = 0; i < tokens * in_dim; i++)
        input[i] = sinf((float)i * 0.031f) * (0.25f + (float)(i % 11) * 0.03125f);
    TEST_ASSERT(q36_gpu_init() != 0);
    q36_gpu_set_dense_model(true);
    x = q36_gpu_tensor_alloc(x_bytes);
    out = q36_gpu_tensor_alloc(y_bytes);
    ref = q36_gpu_tensor_alloc(y_bytes);
    TEST_ASSERT(x && out && ref);
    if (!x || !out || !ref) goto cleanup;
    TEST_ASSERT(q36_gpu_set_model_map(weights, weight_bytes) != 0);
    TEST_ASSERT(q36_gpu_tensor_write(x, 0, input, x_bytes) != 0);
    int (*matmul)(q36_gpu_tensor *, const void *, uint64_t, uint64_t,
                  uint32_t, uint64_t, uint64_t, const q36_gpu_tensor *, uint64_t, float) =
        type < 16 ? q36_gpu_matmul_k_quant_scaled_tensor : q36_gpu_matmul_iq_quant_scaled_tensor;
    const float scales[] = {0.75f, -0.5f};
    for (unsigned si = 0; si < sizeof(scales) / sizeof(scales[0]); si++) {
        TEST_ASSERT(matmul(
            out, weights, weight_bytes, 0, type, in_dim, out_dim, x, tokens, scales[si]) != 0);
        for (uint32_t first = 0; first < tokens; first += 128) {
            uint32_t n = tokens - first < 128 ? tokens - first : 128;
            q36_gpu_tensor *xv = q36_gpu_tensor_view(
                x, (uint64_t)first * in_dim * sizeof(float), (uint64_t)n * in_dim * sizeof(float));
            q36_gpu_tensor *yv = q36_gpu_tensor_view(
                ref, (uint64_t)first * out_dim * sizeof(float), (uint64_t)n * out_dim * sizeof(float));
            TEST_ASSERT(xv && yv);
            if (xv && yv)
                TEST_ASSERT(matmul(
                    yv, weights, weight_bytes, 0, type, in_dim, out_dim, xv, n, scales[si]) != 0);
            q36_gpu_tensor_free(xv);
            q36_gpu_tensor_free(yv);
        }
        TEST_ASSERT(q36_gpu_tensor_read(out, 0, got, y_bytes) != 0);
        TEST_ASSERT(q36_gpu_tensor_read(ref, 0, want, y_bytes) != 0);
        bool nonzero = false;
        for (uint32_t i = 0; i < tokens * out_dim; i++) {
            TEST_ASSERT(isfinite(got[i]) && isfinite(want[i]));
            nonzero = nonzero || got[i] != 0.0f;
        }
        TEST_ASSERT(nonzero);
        TEST_ASSERT(memcmp(got, want, y_bytes) == 0);
    }
cleanup:
    q36_gpu_tensor_free(x);
    q36_gpu_tensor_free(out);
    q36_gpu_tensor_free(ref);
    q36_gpu_set_dense_model(false);
    q36_gpu_cleanup();
    free(weights);
    free(input);
    free(got);
    free(want);
#endif
}

#ifndef Q36_NO_GPU
static const char *test_dense_quant_type_name(uint32_t type) {
    static const char *names[] = {
        [10] = "Q2_K", [11] = "Q3_K", [12] = "Q4_K", [13] = "Q5_K",
        [14] = "Q6_K", [16] = "IQ2_XXS", [17] = "IQ2_XS",
        [18] = "IQ3_XXS", [19] = "IQ1_S", [20] = "IQ4_NL",
        [21] = "IQ3_S", [22] = "IQ2_S", [23] = "IQ4_XS",
        [29] = "IQ1_M", [30] = "BF16",
    };
    return type < sizeof(names) / sizeof(names[0]) && names[type]
        ? names[type] : "unknown";
}

static void test_dense_quant_model_rows(void) {
    static const uint32_t types[] = {
        10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 30,
    };
    enum { out_dim = 8 };
    q36_engine *engine = NULL;

    if (!test_model_available(test_model_path())) {
        test_skip("dense-quant-model-rows", "model file not found");
        return;
    }
    engine = test_open_runtime_engine(test_model_path(), 1);
    TEST_ASSERT(engine != NULL);
    if (!engine) return;
    TEST_ASSERT(q36_gpu_init() != 0);
    q36_gpu_set_dense_model(true);

    for (size_t ti = 0; ti < sizeof(types) / sizeof(types[0]); ti++) {
        const uint32_t type = types[ti];
        char tensor_name[256];
        uint32_t in_dim = 0, packed_type = 0, packed_n = 0;
        uint64_t row_bytes = 0;
        if (q36_engine_debug_first_tensor_of_type(
                engine, type, tensor_name, sizeof(tensor_name), &in_dim) != 0)
            continue;
        TEST_ASSERT(q36_engine_debug_tensor_row_packed(
            engine, tensor_name, 0, NULL, 0, &row_bytes, &packed_type,
            &packed_n) == 0);
        TEST_ASSERT(type == packed_type && in_dim == packed_n);
        if (!row_bytes || type != packed_type || in_dim != packed_n) continue;

        const uint64_t weight_bytes = row_bytes * out_dim;
        const uint64_t model_bytes =
            type == 18u ? 2u * weight_bytes : weight_bytes;
        const uint64_t weight_alloc = test_round_up_u64(
            model_bytes, (uint64_t)getpagesize());
        void *weights = NULL;
        float *dequant = malloc((size_t)in_dim * sizeof(*dequant));
        TEST_ASSERT(posix_memalign(&weights, (size_t)getpagesize(),
                                  (size_t)weight_alloc) == 0);
        TEST_ASSERT(dequant != NULL);
        if (!weights || !dequant) {
            free(weights);
            free(dequant);
            continue;
        }
        memset(weights, 0, (size_t)weight_alloc);
        for (uint32_t row = 0; row < out_dim; row++) {
            TEST_ASSERT(q36_engine_debug_tensor_row_packed(
                engine, tensor_name, row,
                (uint8_t *)weights + (uint64_t)row * row_bytes, row_bytes,
                NULL, NULL, NULL) == 0);
        }
        if (type == 18u)
            memcpy((uint8_t *)weights + weight_bytes, weights,
                   (size_t)weight_bytes);

        static const uint32_t token_counts[] = {1u, 2u, 5u, 8u, 129u};
        for (uint32_t tokens_i = 0;
             tokens_i < sizeof(token_counts) / sizeof(token_counts[0]);
             tokens_i++) {
            const uint32_t tokens = token_counts[tokens_i];
            if (tokens == 129u && type != 29u && type != 30u) continue;
            const uint64_t x_count = (uint64_t)tokens * in_dim;
            const uint64_t out_count = (uint64_t)tokens * out_dim;
            float *x_host = malloc((size_t)x_count * sizeof(*x_host));
            float *got = malloc((size_t)out_count * sizeof(*got));
            float *want = malloc((size_t)out_count * sizeof(*want));
            q36_gpu_tensor *x = q36_gpu_tensor_alloc(x_count * sizeof(float));
            q36_gpu_tensor *out = q36_gpu_tensor_alloc(out_count * sizeof(float));
            TEST_ASSERT(x_host && got && want && x && out);
            if (!x_host || !got || !want || !x || !out) {
                free(x_host); free(got); free(want);
                q36_gpu_tensor_free(x); q36_gpu_tensor_free(out);
                continue;
            }
            for (uint64_t i = 0; i < x_count; i++)
                x_host[i] = sinf((float)(i * 17u + type * 31u) * 0.013f)
                    * (0.25f + (float)(i % 11u) * 0.03125f);
            TEST_ASSERT(q36_gpu_tensor_write(
                x, 0, x_host, x_count * sizeof(float)) != 0);
            if (type == 30) {
                for (uint64_t i = 0; i < x_count; i++) x_host[i] = test_round_bf16(x_host[i]);
            }
#ifndef Q36_METAL
            /* Vulkan quant kernels consume Q8_K activations. Compare the
             * same matrix product rather than the unquantized input. */
            else {
                for (uint64_t base = 0; base < x_count; base += TEST_QK_K) {
                    struct {
                        float d;
                        int8_t qs[TEST_QK_K];
                        int16_t bsums[TEST_QK_K / 16];
                    } q;
                    q36_quant_q8_k(x_host + base, &q, TEST_QK_K);
                    for (uint32_t i = 0; i < TEST_QK_K; i++)
                        x_host[base + i] = q.d * q.qs[i];
                }
                if (tokens > 8u) {
                    for (uint64_t i = 0; i < x_count; i++)
                        x_host[i] = q36_quant_f16_to_f32(q36_quant_f32_to_f16(x_host[i]));
                }
            }
#endif
            for (uint32_t row = 0; row < out_dim; row++) {
                TEST_ASSERT(q36_quant_dequantize(
                    type, (const uint8_t *)weights + (uint64_t)row * row_bytes,
                    dequant, in_dim));
#ifndef Q36_METAL
                if (type != 30 && tokens > 8u) {
                    for (uint32_t i = 0; i < in_dim; i++)
                        dequant[i] = q36_quant_f16_to_f32(q36_quant_f32_to_f16(dequant[i]));
                }
#endif
                for (uint32_t tok = 0; tok < tokens; tok++) {
                    double sum = 0.0;
                    for (uint32_t i = 0; i < in_dim; i++)
                        sum += (double)dequant[i] * x_host[(uint64_t)tok * in_dim + i];
                    want[(uint64_t)tok * out_dim + row] = (float)sum;
                }
            }
            TEST_ASSERT(q36_gpu_set_model_map(weights, weight_alloc) != 0);
#ifndef Q36_METAL
            q36_gpu_set_micro_batch(tokens <= 8u);
#endif
            int ok = type == 30
                ? q36_gpu_matmul_bf16_scaled_tensor(
                    out, weights, weight_alloc, 0, in_dim, out_dim, x, tokens, 1.0f)
                : type <= 14
                ? q36_gpu_matmul_k_quant_scaled_tensor(
                    out, weights, weight_alloc, 0, type, in_dim, out_dim,
                    x, tokens, 1.0f)
                : q36_gpu_matmul_iq_quant_scaled_tensor(
                    out, weights, weight_alloc, 0, type, in_dim, out_dim,
                    x, tokens, 1.0f);
#ifndef Q36_METAL
            q36_gpu_set_micro_batch(false);
#endif
            TEST_ASSERT(ok != 0);
            TEST_ASSERT(q36_gpu_tensor_read(
                out, 0, got, out_count * sizeof(float)) != 0);
            double max_abs = 0.0, rms = 0.0, ref_rms = 0.0;
            for (uint64_t i = 0; i < out_count; i++) {
                double d = (double)got[i] - want[i];
                if (fabs(d) > max_abs) max_abs = fabs(d);
                rms += d * d;
                ref_rms += (double)want[i] * want[i];
            }
            rms = sqrt(rms / (double)out_count);
            ref_rms = sqrt(ref_rms / (double)out_count);
            fprintf(stderr,
                    "q36-test: dense quant %-7s tokens=%u tensor=%s max_abs=%.7g rel_rms=%.7g\n",
                    test_dense_quant_type_name(type), tokens, tensor_name,
                    max_abs, rms / fmax(ref_rms, 1.0e-20));
            TEST_ASSERT(isfinite(max_abs) && isfinite(rms));
            TEST_ASSERT(max_abs < (tokens == 1 ? 1.0e-5 : 2.0e-3));
            TEST_ASSERT(rms / fmax(ref_rms, 1.0e-20) <
                        (tokens == 1 ? 1.0e-5 : 1.0e-3));
#ifdef Q36_METAL
            if (type == 18u && tokens == 1u) {
                q36_gpu_tensor *out_b =
                    q36_gpu_tensor_alloc(out_count * sizeof(float));
                TEST_ASSERT(out_b != NULL);
                if (out_b) {
                    TEST_ASSERT(q36_gpu_matmul_iq3_xxs_pair_tensor(
                        out, out_b, weights, weight_alloc,
                        0, weight_bytes, in_dim, out_dim, x) != 0);
                    TEST_ASSERT(q36_gpu_tensor_read(
                        out, 0, got, out_count * sizeof(float)) != 0);
                    double pair_max_abs = 0.0;
                    for (uint64_t i = 0; i < out_count; i++)
                        pair_max_abs = fmax(
                            pair_max_abs, fabs((double)got[i] - want[i]));
                    TEST_ASSERT(q36_gpu_tensor_read(
                        out_b, 0, got, out_count * sizeof(float)) != 0);
                    for (uint64_t i = 0; i < out_count; i++)
                        pair_max_abs = fmax(
                            pair_max_abs, fabs((double)got[i] - want[i]));
                    fprintf(stderr,
                            "q36-test: dense IQ3_XXS pair max_abs=%.7g\n",
                            pair_max_abs);
                    TEST_ASSERT(isfinite(pair_max_abs) &&
                                pair_max_abs < 1.0e-5);
                    q36_gpu_tensor_free(out_b);
                }
            }
#endif
            free(x_host); free(got); free(want);
            q36_gpu_tensor_free(x); q36_gpu_tensor_free(out);
        }
        free(dequant);
        free(weights);
    }
    q36_gpu_set_dense_model(false);
    q36_gpu_cleanup();
    q36_engine_close(engine);
}
#else
static void test_dense_quant_model_rows(void) {
    test_skip("dense-quant-model-rows", "CPU-only build");
}
#endif

static float test_swiglu_ref(float gate, float up) {
    return gate / (1.0f + expf(-gate)) * up;
}

#ifndef Q36_NO_GPU
static void test_vulkan_moe_q4_one(bool streaming, uint32_t gate_type, uint32_t n_tok,
                                    uint32_t n_expert, uint32_t blocks, bool scaled) {
    enum {
        mid_dim = TEST_QK_K,
        out_dim = 11,
        n_used = 2,
    };
    const uint32_t in_dim = blocks * TEST_QK_K;
    const uint64_t block_bytes = gate_type == 23 ? 136 : sizeof(test_block_q4_K);
    const uint64_t row_bytes = blocks * block_bytes;
    const uint64_t down_row_bytes = sizeof(test_block_q4_K);
    const uint64_t gate_stride = row_bytes * mid_dim;
    const uint64_t down_stride = down_row_bytes * out_dim;
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = gate_offset + gate_stride * n_expert;
    const uint64_t down_offset = up_offset + gate_stride * n_expert;
    const uint64_t scale_offset = down_offset + down_stride * n_expert;
    const uint64_t model_bytes = scale_offset + 3u * n_expert * sizeof(float);
    const uint64_t model_alloc = test_round_up_u64(model_bytes, (uint64_t)getpagesize());
    uint32_t selected_host[n_tok * n_used];
    float weights_host[n_tok * n_used];
    void *model = NULL;
    float *x_host = NULL;
    float *out_host = NULL;
    float *want = NULL;
    float *mid = NULL;
    test_block_q8_K *midq = NULL;
    q36_gpu_tensor *x = NULL;
    q36_gpu_tensor *qx = NULL;
    q36_gpu_tensor *selected = NULL;
    q36_gpu_tensor *weights = NULL;
    q36_gpu_tensor *out = NULL;
    int ok;

    TEST_ASSERT(posix_memalign(&model, (size_t)getpagesize(), (size_t)model_alloc) == 0);
    TEST_ASSERT(model != NULL);
    if (!model) return;
    memset(model, 0, (size_t)model_alloc);

    for (uint32_t expert = 0; expert < n_expert; expert++) {
        for (uint32_t row = 0; row < mid_dim * blocks; row++) {
            uint8_t *g = (uint8_t *)model + gate_offset + expert * gate_stride + row * block_bytes;
            uint8_t *u = (uint8_t *)model + up_offset + expert * gate_stride + row * block_bytes;
            uint32_t seed = (1u + expert * 1000u + row) % 89u;
            if (gate_type == 23) {
                uint16_t d = test_float_to_f16(.0001f * (float)(seed + 1u));
                for (uint32_t j = 2; j < block_bytes; j++) {
                    g[j] = (uint8_t)(seed * 17u + j * 7u);
                    u[j] = (uint8_t)(seed * 11u + j * 13u);
                }
                memcpy(g, &d, sizeof(d));
                memcpy(u, &d, sizeof(d));
            } else {
                test_fill_q4_k((test_block_q4_K *)g, seed);
                test_fill_q4_k((test_block_q4_K *)u, (seed + 1u) % 89u);
            }
        }
        for (uint32_t row = 0; row < out_dim; row++) {
            test_fill_q4_k((test_block_q4_K *)((uint8_t *)model + down_offset + expert * down_stride + row * down_row_bytes),
                           (3u + expert * 1000u + row) % 89u);
        }
    }

    float *scales = (float *)((uint8_t *)model + scale_offset);
    for (uint32_t i = 0; i < 3u * n_expert; i++)
        scales[i] = 0.5f + (float)(i % 7u) * 0.125f;

    x_host = malloc((size_t)n_tok * in_dim * sizeof(*x_host));
    out_host = malloc((size_t)n_tok * out_dim * sizeof(*out_host));
    want = calloc((size_t)n_tok * out_dim, sizeof(*want));
    mid = malloc((size_t)n_tok * n_used * mid_dim * sizeof(*mid));
    midq = malloc((size_t)n_tok * n_used * sizeof(*midq));
    TEST_ASSERT(x_host && out_host && want && mid && midq);
    if (!x_host || !out_host || !want || !mid || !midq) goto done;

    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            x_host[(uint64_t)t * in_dim + i] =
                (float)((int)((t * 13u + i * 7u) % 31u) - 15) / 1900.0f;
        }
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        selected_host[t * n_used] = (t * 13u + n_expert - 1u) % n_expert;
        selected_host[t * n_used + 1u] = (selected_host[t * n_used] + 2u) % n_expert;
        weights_host[t * n_used] = 0.25f + 0.05f * (float)(t % 6u);
        weights_host[t * n_used + 1u] =
            1.0f - weights_host[t * n_used];
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        /* CPU q8_K omits the Vulkan layout's dmin padding word. */
        uint8_t cpu_q8[blocks * sizeof(test_block_q8_K)];
        q36_quant_q8_k(x_host + (uint64_t)t * in_dim, cpu_q8, in_dim);
        for (uint32_t u = 0; u < n_used; u++) {
            uint32_t expert = selected_host[t * n_used + u];
            for (uint32_t row = 0; row < mid_dim; row++) {
                const uint8_t *g = (const uint8_t *)model + gate_offset + expert * gate_stride + row * row_bytes;
                const uint8_t *up = (const uint8_t *)model + up_offset + expert * gate_stride + row * row_bytes;
                float gv, uv;
                TEST_ASSERT(q36_quant_dot_q8_k(gate_type, g, cpu_q8, in_dim, &gv));
                TEST_ASSERT(q36_quant_dot_q8_k(gate_type, up, cpu_q8, in_dim, &uv));
                if (scaled) {
                    gv *= scales[expert];
                    uv *= scales[n_expert + expert];
                }
                mid[((uint64_t)t * n_used + u) * mid_dim + row] = test_swiglu_ref(gv, uv);
            }
            test_quantize_q8_k(mid + ((uint64_t)t * n_used + u) * mid_dim,
                               &midq[t * n_used + u]);
            for (uint32_t row = 0; row < out_dim; row++) {
                const uint8_t *down = (const uint8_t *)model + down_offset + expert * down_stride + row * down_row_bytes;
                float dv = test_dot_q4_k_q8_k((const test_block_q4_K *)down, &midq[t * n_used + u]);
                if (scaled) dv *= scales[2u * n_expert + expert];
                want[(uint64_t)t * out_dim + row] += dv * weights_host[t * n_used + u];
            }
        }
    }

    x = q36_gpu_tensor_alloc((uint64_t)n_tok * in_dim * sizeof(float));
    qx = q36_gpu_tensor_alloc((uint64_t)n_tok * blocks * sizeof(test_block_q8_K));
    selected = q36_gpu_tensor_alloc(sizeof(selected_host));
    weights = q36_gpu_tensor_alloc(sizeof(weights_host));
    out = q36_gpu_tensor_alloc((uint64_t)n_tok * out_dim * sizeof(float));
    TEST_ASSERT(x && qx && selected && weights && out);
    if (!x || !qx || !selected || !weights || !out) goto done;

    TEST_ASSERT(q36_gpu_set_model_map(model, model_alloc) != 0);
    q36_gpu_set_ssd_streaming(streaming);
    q36_gpu_set_streaming_expert_cache_budget(streaming ? n_expert : 0);
    q36_gpu_set_streaming_expert_cache_expert_bytes(streaming ? gate_stride + gate_stride + down_stride : 0);
    TEST_ASSERT(q36_gpu_tensor_write(x, 0, x_host, (uint64_t)n_tok * in_dim * sizeof(float)) != 0);
    TEST_ASSERT(q36_gpu_tensor_write(selected, 0, selected_host, sizeof(selected_host)) != 0);
    TEST_ASSERT(q36_gpu_tensor_write(weights, 0, weights_host, sizeof(weights_host)) != 0);
    TEST_ASSERT(q36_gpu_quantize_q8_k_tensor(qx, x, in_dim, n_tok) != 0);

    q36_gpu_moe_weight gate = { gate_offset, gate_type, scale_offset, scaled };
    q36_gpu_moe_weight up = { up_offset, gate_type, scale_offset + n_expert * sizeof(float), scaled };
    q36_gpu_moe_weight down = { down_offset, 12, scale_offset + 2u * n_expert * sizeof(float), scaled };
    ok = q36_gpu_moe_ffn_q8_tensor(out, model, model_alloc,
                                   &gate, &up, &down,
                                   selected, weights, 0, n_used, qx, n_tok,
                                   in_dim, mid_dim, out_dim, n_expert);
    TEST_ASSERT(ok != 0);
    TEST_ASSERT(q36_gpu_tensor_read(out, 0, out_host, (uint64_t)n_tok * out_dim * sizeof(float)) != 0);

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    uint32_t max_i = 0;
    for (uint32_t i = 0; i < n_tok * out_dim; i++) {
        TEST_ASSERT(isfinite(out_host[i]));
        float err = fabsf(out_host[i] - want[i]);
        float den = fabsf(want[i]) > 1.0f ? fabsf(want[i]) : 1.0f;
        float rel = err / den;
        if (err > max_abs) {
            max_abs = err;
            max_i = i;
        }
        if (rel > max_rel) max_rel = rel;
    }
    if (max_abs >= 5.0e-2f && max_rel >= 5.0e-5f) {
        fprintf(stderr,
                "q36-test: routed q4 %s type=%u tokens=%u scaled=%d max_abs=%g max_rel=%g idx=%u got=%g want=%g\n",
                streaming ? "stream" : "resident", gate_type, n_tok, scaled,
                (double)max_abs,
                (double)max_rel,
                max_i,
                (double)out_host[max_i],
                (double)want[max_i]);
    }
    TEST_ASSERT(max_abs < 5.0e-2f || max_rel < 5.0e-5f);
#ifndef Q36_METAL
    if (gate_type == 23) {
        const char *env = getenv("Q36_VK_MOE_GATE_UP");
        char *saved = env ? test_strdup(env) : NULL;
        float *fallback = malloc((size_t)n_tok * out_dim * sizeof(float));
        TEST_ASSERT(fallback != NULL);
        setenv("Q36_VK_MOE_GATE_UP", "0", 1);
        TEST_ASSERT(q36_gpu_moe_ffn_q8_tensor(out, model, model_alloc,
                        &gate, &up, &down, selected, weights, 0, n_used, qx,
                        n_tok, in_dim, mid_dim, out_dim, n_expert) != 0);
        if (fallback) {
            TEST_ASSERT(q36_gpu_tensor_read(out, 0, fallback,
                            (uint64_t)n_tok * out_dim * sizeof(float)) != 0);
            TEST_ASSERT(memcmp(out_host, fallback,
                            (size_t)n_tok * out_dim * sizeof(float)) == 0);
        }
        if (saved) setenv("Q36_VK_MOE_GATE_UP", saved, 1);
        else unsetenv("Q36_VK_MOE_GATE_UP");
        free(saved);
        free(fallback);
    }
#endif

done:
    q36_gpu_set_ssd_streaming(false);
    q36_gpu_set_streaming_expert_cache_budget(0);
    q36_gpu_set_streaming_expert_cache_expert_bytes(0);
    q36_gpu_tensor_free(out);
    q36_gpu_tensor_free(weights);
    q36_gpu_tensor_free(selected);
    q36_gpu_tensor_free(qx);
    q36_gpu_tensor_free(x);
    free(midq);
    free(mid);
    free(want);
    free(out_host);
    free(x_host);
    free(model);
}
#endif

static void test_vulkan_moe_q4(void) {
#ifdef Q36_NO_GPU
    test_skip("vulkan-kernels", "CPU-only build");
    return;
#else
#ifndef Q36_METAL
    const uint32_t batches[] = {1, 2, 3, 8, 32, 33};
#endif
    for (int stream = 0; stream < 2; stream++) {
        test_vulkan_moe_q4_one(stream, 12, 1, 3, 1, false);
        test_vulkan_moe_q4_one(stream, 12, 2, 3, 1, false);
#ifndef Q36_METAL
        test_vulkan_moe_q4_one(stream, 23, 1, 3, 1, false);
        test_vulkan_moe_q4_one(stream, 23, 2, 3, 1, false);
        for (size_t i = 0; i < sizeof(batches) / sizeof(batches[0]); i++) {
            for (int scaled = 0; scaled < 2; scaled++) {
                test_vulkan_moe_q4_one(stream, 12, batches[i], 64, 2, scaled);
                test_vulkan_moe_q4_one(stream, 23, batches[i], 64, 2, scaled);
            }
        }
#endif
    }
#endif
}

static void test_routed_q4_quants(void) {
#ifndef Q36_NO_GPU
    if (!q36_gpu_init()) { test_skip("routed-q4-quants", "GPU unavailable"); return; }
#endif
    test_vulkan_moe_q4();
#ifndef Q36_NO_GPU
    q36_gpu_cleanup();
#endif
}

#ifndef Q36_NO_GPU
typedef struct {
    uint16_t d;
    uint16_t qs[TEST_QK_K / 8];
} test_block_iq2_xxs;

static void test_fill_iq2_xxs(test_block_iq2_xxs *b, uint32_t row) {
    uint32_t s = row * 2654435761u + 97u;
    b->d = test_float_to_f16(0.0005f * (float)(row % 43u + 1u));
    for (uint32_t i = 0; i < TEST_QK_K / 8; i++) {
        s = s * 1103515245u + 12345u;
        b->qs[i] = (uint16_t)(s >> 13);
    }
}

/* The routed f32 MoE path against Q36's CPU dequantized-row reference.
 * Metal checks its compact top-k=8 prefill MM, the standard DS4-style MM,
 * and the exact matvec fallback independently. */
static void test_vulkan_moe_gemm(void) {
    enum {
        in_dim = 2 * TEST_QK_K,
        mid_dim = TEST_QK_K,
#ifdef Q36_METAL
        out_dim = 128,
        n_expert = 8,
        n_used = 8,
#else
        out_dim = 131,
        n_expert = 4,
        n_used = 2,
#endif
#ifdef Q36_METAL
        n_tok = 64,
#else
        n_tok = 40,
#endif
    };
    const uint64_t gu_row = sizeof(test_block_iq2_xxs) * (in_dim / TEST_QK_K);
    const uint64_t down_row = sizeof(test_block_q2_K) * (mid_dim / TEST_QK_K);
    const uint64_t gate_stride = gu_row * mid_dim;
    const uint64_t down_stride = down_row * out_dim;
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = gate_offset + gate_stride * n_expert;
    const uint64_t down_offset = up_offset + gate_stride * n_expert;
    const uint64_t gsc_offset = down_offset + down_stride * n_expert;
    const uint64_t usc_offset = gsc_offset + n_expert * sizeof(float);
    const uint64_t dsc_offset = usc_offset + n_expert * sizeof(float);
    const uint64_t model_bytes = dsc_offset + n_expert * sizeof(float);
    const uint64_t model_alloc = test_round_up_u64(model_bytes, (uint64_t)getpagesize());
    void *model = NULL;
    float *x_host = NULL;
    float *want = NULL;
    float *got_gemm = NULL;
#ifdef Q36_METAL
    float *got_standard = NULL;
#endif
    float *got_mv = NULL;
    float *wrow = NULL;
    float *mid = NULL;
    q36_gpu_tensor *x = NULL;
    q36_gpu_tensor *selected = NULL;
    q36_gpu_tensor *weights = NULL;
    q36_gpu_tensor *out = NULL;
    uint32_t selected_host[n_tok * n_used];
    float weights_host[n_tok * n_used];
    int ok;

    TEST_ASSERT(posix_memalign(&model, (size_t)getpagesize(), (size_t)model_alloc) == 0);
    if (!model) return;
    memset(model, 0, (size_t)model_alloc);

    for (uint32_t expert = 0; expert < n_expert; expert++) {
        for (uint32_t row = 0; row < mid_dim; row++) {
            test_block_iq2_xxs *g = (test_block_iq2_xxs *)((uint8_t *)model + gate_offset + expert * gate_stride + row * gu_row);
            test_block_iq2_xxs *u = (test_block_iq2_xxs *)((uint8_t *)model + up_offset + expert * gate_stride + row * gu_row);
            for (uint32_t sb = 0; sb < in_dim / TEST_QK_K; sb++) {
                test_fill_iq2_xxs(g + sb, 1u + expert * 1000u + row * 2u + sb);
                test_fill_iq2_xxs(u + sb, 2u + expert * 1000u + row * 2u + sb);
            }
        }
        for (uint32_t row = 0; row < out_dim; row++) {
            /* Keep the fill's row-scaled d in the real model's Q2_K range:
             * huge synthetic scales make the down dot cancel thousands into
             * units, which amplifies the designed f16 staging rounding far
             * beyond anything the model produces. */
            test_fill_q2_k((test_block_q2_K *)((uint8_t *)model + down_offset + expert * down_stride + row * down_row),
                           (3u + expert * 1000u + row) % 89u);
        }
        ((float *)((uint8_t *)model + gsc_offset))[expert] = 1.0f + 0.10f * (float)expert;
        ((float *)((uint8_t *)model + usc_offset))[expert] = 1.0f - 0.05f * (float)expert;
        ((float *)((uint8_t *)model + dsc_offset))[expert] = 1.0f + 0.07f * (float)expert;
    }

    x_host = malloc((size_t)n_tok * in_dim * sizeof(*x_host));
    want = calloc((size_t)n_tok * out_dim, sizeof(*want));
    got_gemm = malloc((size_t)n_tok * out_dim * sizeof(*got_gemm));
#ifdef Q36_METAL
    got_standard = malloc((size_t)n_tok * out_dim * sizeof(*got_standard));
#endif
    got_mv = malloc((size_t)n_tok * out_dim * sizeof(*got_mv));
    wrow = malloc((size_t)in_dim * sizeof(*wrow));
    mid = malloc((size_t)mid_dim * sizeof(*mid));
#ifdef Q36_METAL
    TEST_ASSERT(x_host && want && got_gemm && got_standard &&
                got_mv && wrow && mid);
    if (!x_host || !want || !got_gemm || !got_standard ||
        !got_mv || !wrow || !mid) goto done;
#else
    TEST_ASSERT(x_host && want && got_gemm && got_mv && wrow && mid);
    if (!x_host || !want || !got_gemm || !got_mv || !wrow || !mid) goto done;
#endif

    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            x_host[(uint64_t)t * in_dim + i] =
                (float)((int)((t * 13u + i * 7u) % 61u) - 30) / 480.0f;
        }
#ifdef Q36_METAL
        float weight_sum = 0.0f;
        for (uint32_t u = 0; u < n_used; u++) {
            selected_host[t * n_used + u] =
                (uint32_t)((t + u) % n_expert);
            weights_host[t * n_used + u] =
                1.0f + 0.125f * (float)((t + 3u * u) % 5u);
            weight_sum += weights_host[t * n_used + u];
        }
        for (uint32_t u = 0; u < n_used; u++)
            weights_host[t * n_used + u] /= weight_sum;
#else
        /* Skewed routing covers full tiles, multi-tile experts, and tails. */
        selected_host[t * n_used + 0] = t < 28 ? 0u : 2u;
        selected_host[t * n_used + 1] = t < 28 ? 1u : 3u;
        weights_host[t * n_used + 0] =
            0.35f + 0.005f * (float)(t % 7u);
        weights_host[t * n_used + 1] =
            1.0f - weights_host[t * n_used + 0];
#endif
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t u = 0; u < n_used; u++) {
            uint32_t expert = selected_host[t * n_used + u];
            const float *xt = x_host + (uint64_t)t * in_dim;
            float gsc = ((const float *)((uint8_t *)model + gsc_offset))[expert];
            float usc = ((const float *)((uint8_t *)model + usc_offset))[expert];
            float dsc = ((const float *)((uint8_t *)model + dsc_offset))[expert];
            for (uint32_t row = 0; row < mid_dim; row++) {
                float gv = 0.0f;
                float uv = 0.0f;
                test_dequant_row_cpu(16, (uint8_t *)model + gate_offset + expert * gate_stride + row * gu_row,
                                     wrow, in_dim);
                for (uint32_t i = 0; i < in_dim; i++) gv += wrow[i] * xt[i];
                test_dequant_row_cpu(16, (uint8_t *)model + up_offset + expert * gate_stride + row * gu_row,
                                     wrow, in_dim);
                for (uint32_t i = 0; i < in_dim; i++) uv += wrow[i] * xt[i];
                mid[row] = test_swiglu_ref(gv * gsc, uv * usc);
            }
            for (uint32_t row = 0; row < out_dim; row++) {
                float dv = 0.0f;
                test_dequant_row_cpu(10, (uint8_t *)model + down_offset + expert * down_stride + row * down_row,
                                     wrow, mid_dim);
                for (uint32_t i = 0; i < mid_dim; i++) dv += wrow[i] * mid[i];
                want[(uint64_t)t * out_dim + row] += dv * dsc * weights_host[t * n_used + u];
            }
        }
    }

    x = q36_gpu_tensor_alloc((uint64_t)n_tok * in_dim * sizeof(float));
    selected = q36_gpu_tensor_alloc(sizeof(selected_host));
    weights = q36_gpu_tensor_alloc(sizeof(weights_host));
    out = q36_gpu_tensor_alloc((uint64_t)n_tok * out_dim * sizeof(float));
    TEST_ASSERT(x && selected && weights && out);
    if (!x || !selected || !weights || !out) goto done;

    TEST_ASSERT(q36_gpu_set_model_map(model, model_alloc) != 0);
    TEST_ASSERT(q36_gpu_tensor_write(x, 0, x_host, (uint64_t)n_tok * in_dim * sizeof(float)) != 0);
    TEST_ASSERT(q36_gpu_tensor_write(selected, 0, selected_host, sizeof(selected_host)) != 0);
    TEST_ASSERT(q36_gpu_tensor_write(weights, 0, weights_host, sizeof(weights_host)) != 0);

    q36_gpu_moe_weight gate = { gate_offset, 16, gsc_offset, true };
    q36_gpu_moe_weight up = { up_offset, 16, usc_offset, true };
    q36_gpu_moe_weight down = { down_offset, 10, dsc_offset, true };

#ifdef Q36_METAL
    setenv("Q36_METAL_MOE_MM", "1", 1);
    unsetenv("Q36_METAL_MOE_COMPACT");
    ok = q36_gpu_moe_ffn_f32_tensor(out, model, model_alloc,
                                    &gate, &up, &down,
                                    selected, weights, 0, n_used, x, n_tok,
                                    in_dim, mid_dim, out_dim, n_expert);
    TEST_ASSERT(ok != 0);
    TEST_ASSERT(q36_gpu_tensor_read(out, 0, got_gemm,
                (uint64_t)n_tok * out_dim * sizeof(float)) != 0);

    setenv("Q36_METAL_MOE_COMPACT", "0", 1);
    ok = q36_gpu_moe_ffn_f32_tensor(out, model, model_alloc,
                                    &gate, &up, &down,
                                    selected, weights, 0, n_used, x, n_tok,
                                    in_dim, mid_dim, out_dim, n_expert);
    TEST_ASSERT(ok != 0);
    TEST_ASSERT(q36_gpu_tensor_read(out, 0, got_standard,
                (uint64_t)n_tok * out_dim * sizeof(float)) != 0);

    setenv("Q36_METAL_MOE_MM", "0", 1);
    unsetenv("Q36_METAL_MOE_COMPACT");
#else
    setenv("Q36_VK_MOE_GEMM_MIN", "8", 1);
    ok = q36_gpu_moe_ffn_f32_tensor(out, model, model_alloc,
                                    &gate, &up, &down,
                                    selected, weights, 0, n_used, x, n_tok,
                                    in_dim, mid_dim, out_dim, n_expert);
    unsetenv("Q36_VK_MOE_GEMM_MIN");
    TEST_ASSERT(ok != 0);
    TEST_ASSERT(q36_gpu_tensor_read(out, 0, got_gemm, (uint64_t)n_tok * out_dim * sizeof(float)) != 0);

    setenv("Q36_VK_MOE_GEMM", "0", 1);
#endif
    ok = q36_gpu_moe_ffn_f32_tensor(out, model, model_alloc,
                                    &gate, &up, &down,
                                    selected, weights, 0, n_used, x, n_tok,
                                    in_dim, mid_dim, out_dim, n_expert);
#ifndef Q36_METAL
    unsetenv("Q36_VK_MOE_GEMM");
#else
    unsetenv("Q36_METAL_MOE_MM");
#endif
    TEST_ASSERT(ok != 0);
    TEST_ASSERT(q36_gpu_tensor_read(out, 0, got_mv, (uint64_t)n_tok * out_dim * sizeof(float)) != 0);

    {
        float mv_max_rel = 0.0f;
        float gemm_max_rel = 0.0f;
        bool differs = false;
        uint32_t gemm_worst = 0;
#ifdef Q36_METAL
        float standard_max_rel = 0.0f;
        bool compact_differs = false;
        uint32_t standard_worst = 0;
#endif
        uint32_t worst = 0;
        for (uint32_t i = 0; i < n_tok * out_dim; i++) {
            float den = fabsf(want[i]) > 1.0f ? fabsf(want[i]) : 1.0f;
            float mv_rel = fabsf(got_mv[i] - want[i]) / den;
            if (mv_rel > mv_max_rel) {
                mv_max_rel = mv_rel;
                worst = i;
            }
            float gemm_rel = fabsf(got_gemm[i] - want[i]) / den;
            if (gemm_rel > gemm_max_rel) {
                gemm_max_rel = gemm_rel;
                gemm_worst = i;
            }
            if (got_gemm[i] != got_mv[i]) differs = true;
#ifdef Q36_METAL
            float standard_rel = fabsf(got_standard[i] - want[i]) / den;
            if (standard_rel > standard_max_rel) {
                standard_max_rel = standard_rel;
                standard_worst = i;
            }
            if (got_gemm[i] != got_standard[i]) compact_differs = true;
#endif
        }
#ifdef Q36_METAL
        fprintf(stderr,
                "q36-test: routed Metal compact=%g standard=%g matvec=%g "
                "worst(compact=%u standard=%u matvec=%u) got=%g want=%g\n",
                (double)gemm_max_rel, (double)standard_max_rel,
                (double)mv_max_rel, gemm_worst, standard_worst, worst,
                (double)got_gemm[gemm_worst], (double)want[gemm_worst]);
        TEST_ASSERT(mv_max_rel < 1.0e-3f);
        TEST_ASSERT(gemm_max_rel < 5.0e-2f);
        TEST_ASSERT(standard_max_rel < 5.0e-2f);
        TEST_ASSERT(differs);
        TEST_ASSERT(!compact_differs);
#else
        fprintf(stderr,
                "q36-test: routed gemm mv_max_rel=%g gemm_max_rel=%g "
                "worst(gemm=%u matvec=%u) got=%g want=%g\n",
                (double)mv_max_rel, (double)gemm_max_rel,
                gemm_worst, worst,
                (double)got_gemm[gemm_worst], (double)want[gemm_worst]);
        /* The matvec path computes f32 dots; the GEMM path's bound is set
         * by its designed f16 staging noise through the cancelling down
         * dot.  Indexing or dequant bugs produce order-1 errors. */
        TEST_ASSERT(mv_max_rel < 1.0e-3f);
        TEST_ASSERT(gemm_max_rel < 5.0e-2f);
        /* If the two paths agree bit for bit the GEMM never ran. */
        TEST_ASSERT(differs);
#endif
    }

#ifndef Q36_METAL
    /* Verification widens the decode schedule; every row must retain the
     * ordinary fused gate/up and routing-sum arithmetic. */
    for (uint32_t t = 0; t < 8u; t++) {
        q36_gpu_tensor *xi = q36_gpu_tensor_view(x, (uint64_t)t * in_dim * sizeof(float),
                                                   (uint64_t)in_dim * sizeof(float));
        q36_gpu_tensor *si = q36_gpu_tensor_view(selected, (uint64_t)t * n_used * sizeof(uint32_t),
                                                   n_used * sizeof(uint32_t));
        q36_gpu_tensor *wi = q36_gpu_tensor_view(weights, (uint64_t)t * n_used * sizeof(float),
                                                   n_used * sizeof(float));
        TEST_ASSERT(xi && si && wi);
        if (xi && si && wi) {
            TEST_ASSERT(q36_gpu_moe_ffn_f32_tensor(out, model, model_alloc,
                &gate, &up, &down, si, wi, 0, n_used, xi, 1u,
                in_dim, mid_dim, out_dim, n_expert) != 0);
            TEST_ASSERT(q36_gpu_tensor_read(out, 0, got_mv + t * out_dim,
                                             out_dim * sizeof(float)) != 0);
        }
        q36_gpu_tensor_free(wi);
        q36_gpu_tensor_free(si);
        q36_gpu_tensor_free(xi);
    }
    q36_gpu_set_micro_batch(true);
    for (uint32_t n = 2; n <= 8u; n++) {
        TEST_ASSERT(q36_gpu_moe_ffn_f32_tensor(out, model, model_alloc,
            &gate, &up, &down, selected, weights, 0, n_used, x, n,
            in_dim, mid_dim, out_dim, n_expert) != 0);
        TEST_ASSERT(q36_gpu_tensor_read(out, 0, got_gemm, n * out_dim * sizeof(float)) != 0);
        TEST_ASSERT(memcmp(got_gemm, got_mv, n * out_dim * sizeof(float)) == 0);
    }
    q36_gpu_set_micro_batch(false);
#endif

done:
    q36_gpu_tensor_free(out);
    q36_gpu_tensor_free(weights);
    q36_gpu_tensor_free(selected);
    q36_gpu_tensor_free(x);
    free(mid);
    free(wrow);
    free(got_mv);
#ifdef Q36_METAL
    free(got_standard);
#endif
    free(got_gemm);
    free(want);
    free(x_host);
    free(model);
}
#endif

static void test_vulkan_moe_gemm_case(void) {
#ifdef Q36_NO_GPU
    test_skip("vulkan-kernels", "CPU-only build");
    return;
#else
    test_vulkan_moe_gemm();
#endif
}

static float test_sigmoid_stable(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

static void test_vulkan_swiglu_add(void) {
#ifdef Q36_NO_GPU
    test_skip("vulkan-kernels", "CPU-only build");
    return;
#else
    const uint32_t n = 513;
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    const float clamp = 1.25f;
    const float weight = 0.375f;

    q36_gpu_tensor *gate = q36_gpu_tensor_alloc(bytes);
    q36_gpu_tensor *up = q36_gpu_tensor_alloc(bytes);
    q36_gpu_tensor *mid = q36_gpu_tensor_alloc(bytes);
    q36_gpu_tensor *sum = q36_gpu_tensor_alloc(bytes);
    TEST_ASSERT(gate != NULL);
    TEST_ASSERT(up != NULL);
    TEST_ASSERT(mid != NULL);
    TEST_ASSERT(sum != NULL);
    if (!gate || !up || !mid || !sum) {
        q36_gpu_tensor_free(sum);
        q36_gpu_tensor_free(mid);
        q36_gpu_tensor_free(up);
        q36_gpu_tensor_free(gate);
        return;
    }

    float *gate_host = malloc((size_t)bytes);
    float *up_host = malloc((size_t)bytes);
    float *mid_host = malloc((size_t)bytes);
    float *sum_host = malloc((size_t)bytes);
    TEST_ASSERT(gate_host != NULL);
    TEST_ASSERT(up_host != NULL);
    TEST_ASSERT(mid_host != NULL);
    TEST_ASSERT(sum_host != NULL);
    if (!gate_host || !up_host || !mid_host || !sum_host) {
        free(sum_host);
        free(mid_host);
        free(up_host);
        free(gate_host);
        q36_gpu_tensor_free(sum);
        q36_gpu_tensor_free(mid);
        q36_gpu_tensor_free(up);
        q36_gpu_tensor_free(gate);
        return;
    }

    for (uint32_t i = 0; i < n; i++) {
        gate_host[i] = (float)((int)((i * 17u) % 41u) - 20) / 9.0f;
        up_host[i] = (float)((int)((i * 11u) % 37u) - 18) / 7.0f;
    }

    TEST_ASSERT(q36_gpu_tensor_write(gate, 0, gate_host, bytes) != 0);
    TEST_ASSERT(q36_gpu_tensor_write(up, 0, up_host, bytes) != 0);
    TEST_ASSERT(q36_gpu_swiglu_tensor(mid, gate, up, n, clamp, weight) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(mid, 0, mid_host, bytes) != 0);

    float max_abs = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float g = gate_host[i];
        float u = up_host[i];
        if (g > clamp) g = clamp;
        if (u > clamp) u = clamp;
        if (u < -clamp) u = -clamp;
        float ref = (g * test_sigmoid_stable(g)) * u * weight;
        float err = fabsf(mid_host[i] - ref);
        if (err > max_abs) max_abs = err;
    }
    TEST_ASSERT(max_abs < 1.0e-5f);

    TEST_ASSERT(q36_gpu_add_tensor(sum, gate, up, n) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(sum, 0, sum_host, bytes) != 0);

    max_abs = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float ref = gate_host[i] + up_host[i];
        float err = fabsf(sum_host[i] - ref);
        if (err > max_abs) max_abs = err;
    }
    TEST_ASSERT(max_abs < 1.0e-6f);

    free(sum_host);
    free(mid_host);
    free(up_host);
    free(gate_host);
    q36_gpu_tensor_free(sum);
    q36_gpu_tensor_free(mid);
    q36_gpu_tensor_free(up);
    q36_gpu_tensor_free(gate);
#endif
}

static void test_vulkan_directional_steering(void) {
#ifdef Q36_NO_GPU
    test_skip("vulkan-kernels", "CPU-only build");
#else
    const uint32_t width = 8;
    const uint32_t rows = 2;
    const uint32_t layer = 1;
    const float scale = 0.5f;
    float x[16];
    float expected[16];
    float directions[16] = {0};
    directions[8] = 0.5f;
    directions[9] = -0.5f;
    directions[10] = 0.5f;
    directions[11] = -0.5f;
    for (uint32_t i = 0; i < 16; i++) x[i] = (float)i * 0.25f - 1.0f;
    memcpy(expected, x, sizeof(x));
    for (uint32_t row = 0; row < rows; row++) {
        float dot = 0.0f;
        for (uint32_t i = 0; i < width; i++)
            dot += expected[(uint64_t)row * width + i] * directions[(uint64_t)layer * width + i];
        for (uint32_t i = 0; i < width; i++)
            expected[(uint64_t)row * width + i] -= scale * dot * directions[(uint64_t)layer * width + i];
    }
    q36_gpu_tensor *tx = q36_gpu_tensor_alloc(sizeof(x));
    q36_gpu_tensor *td = q36_gpu_tensor_alloc(sizeof(directions));
    TEST_ASSERT(tx != NULL);
    TEST_ASSERT(td != NULL);
    if (!tx || !td) {
        q36_gpu_tensor_free(td);
        q36_gpu_tensor_free(tx);
        return;
    }
    TEST_ASSERT(q36_gpu_tensor_write(tx, 0, x, sizeof(x)) != 0);
    TEST_ASSERT(q36_gpu_tensor_write(td, 0, directions, sizeof(directions)) != 0);
    TEST_ASSERT(q36_gpu_directional_steering_project_tensor(
                    tx, td, layer, width, rows, scale) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(tx, 0, x, sizeof(x)) != 0);
    for (uint32_t i = 0; i < 16; i++) TEST_ASSERT(fabsf(x[i] - expected[i]) < 1.0e-6f);
    q36_gpu_tensor_free(td);
    q36_gpu_tensor_free(tx);
#endif
}

#ifndef Q36_NO_GPU
static uint32_t test_kv_row_bytes(uint32_t type, uint32_t n) {
    if (type == Q36_KV_CACHE_F16) return n * (uint32_t)sizeof(uint16_t);
    if (type == Q36_KV_CACHE_Q8_0) return (n / 32u) * 34u;
    return (n / 32u) * 18u;
}

static void test_kv_store_row(uint8_t *dst, uint32_t type, const float *src, uint32_t n) {
    if (type == Q36_KV_CACHE_F16) {
        uint16_t *h = (uint16_t *)dst;
        for (uint32_t i = 0; i < n; i++) h[i] = test_float_to_f16(src[i]);
    } else if (type == Q36_KV_CACHE_Q8_0) {
        q36_quant_q8_0(src, dst, n);
    } else {
        q36_quant_q4_0_kv(src, dst, n);
    }
}

static float test_kv_at(const uint8_t *row, uint32_t type, uint32_t i) {
    if (type == Q36_KV_CACHE_F16)
        return test_f16_to_float(((const uint16_t *)row)[i]);
    if (type == Q36_KV_CACHE_Q8_0) {
        const uint8_t *b = row + (i / 32u) * 34u;
        int q = b[2u + (i & 31u)];
        if (q >= 128) q -= 256;
        return test_f16_to_float(*(const uint16_t *)b) * (float)q;
    }
    const uint8_t *b = row + (i / 32u) * 18u;
    uint32_t j = i & 31u;
    uint8_t q = b[2u + (j & 15u)];
    q = j < 16u ? q & 15u : q >> 4;
    return test_f16_to_float(*(const uint16_t *)b) * (float)((int)q - 8);
}

/* Fast-path attention against the CPU reference math, plus the
 * n_tok-invariance gate: a token must produce bit-identical output whether
 * it arrives in a causal batch (query-tiled kernel) or as a single decode
 * step (fused/split kernels).  Callers pick pos0/n_tok to cover partial
 * query tiles and multi-span causal ranges. */
static void test_vulkan_attn_case(uint32_t n_head, uint32_t pos0, uint32_t n_tok,
                                  uint32_t k_type, uint32_t v_type,
                                  uint32_t head_dim) {
    const uint32_t n_head_kv = 2;
    const uint32_t kv_max = pos0 + n_tok;
    const uint32_t cache_row = n_head_kv * head_dim;
    const uint32_t k_row_bytes = test_kv_row_bytes(k_type, cache_row);
    const uint32_t v_row_bytes = test_kv_row_bytes(v_type, cache_row);
    const uint64_t q_floats = (uint64_t)n_tok * n_head * head_dim;
    const uint64_t qg_floats = q_floats * 2u;
    const uint64_t cache_elems = (uint64_t)kv_max * cache_row;
    const uint64_t out_floats = q_floats;
    const uint64_t sinks_alloc = test_round_up_u64(n_head * sizeof(float),
                                                   (uint64_t)getpagesize());

    void *sinks_raw = NULL;
    TEST_ASSERT(posix_memalign(&sinks_raw, (size_t)getpagesize(), (size_t)sinks_alloc) == 0);
    if (!sinks_raw) return;
    float *sinks = sinks_raw;
    memset(sinks, 0, (size_t)sinks_alloc);
    for (uint32_t h = 0; h < n_head; h++)
        sinks[h] = (float)((int)(h * 11u % 13u) - 6) / 4.0f;

    float *q_host = malloc(q_floats * sizeof(float));
    float *qg_host = malloc(qg_floats * sizeof(float));
    float *kf_host = malloc(cache_elems * sizeof(float));
    float *vf_host = malloc(cache_elems * sizeof(float));
    uint8_t *kc_host = malloc((uint64_t)kv_max * k_row_bytes);
    uint8_t *vc_host = malloc((uint64_t)kv_max * v_row_bytes);
    float *out_host = malloc(out_floats * sizeof(float));
    float *out_single = malloc(out_floats * sizeof(float));
    TEST_ASSERT(q_host && qg_host && kf_host && vf_host && kc_host && vc_host && out_host && out_single);

    for (uint64_t i = 0; i < q_floats; i++)
        q_host[i] = (float)((int)((i * 7u + 3u) % 61u) - 30) / 97.0f;
    for (uint64_t i = 0; i < qg_floats; i++)
        qg_host[i] = (float)((int)((i * 13u + 5u) % 41u) - 20) / 11.0f;
    for (uint64_t i = 0; i < cache_elems; i++) {
        kf_host[i] = (float)((int)((i * 5u + 1u) % 53u) - 26) / 83.0f;
        vf_host[i] = (float)((int)((i * 3u + 7u) % 47u) - 23) / 31.0f;
    }
    for (uint32_t t = 0; t < kv_max; t++) {
        test_kv_store_row(kc_host + (uint64_t)t * k_row_bytes, k_type,
                          kf_host + (uint64_t)t * cache_row, cache_row);
        test_kv_store_row(vc_host + (uint64_t)t * v_row_bytes, v_type,
                          vf_host + (uint64_t)t * cache_row, cache_row);
    }

    /* scores scratch stays NULL: the fused path must not touch it. */
    q36_gpu_tensor *q = q36_gpu_tensor_alloc(q_floats * sizeof(float));
    q36_gpu_tensor *qg = q36_gpu_tensor_alloc(qg_floats * sizeof(float));
    q36_gpu_tensor *kc = q36_gpu_tensor_alloc((uint64_t)kv_max * k_row_bytes);
    q36_gpu_tensor *vc = q36_gpu_tensor_alloc((uint64_t)kv_max * v_row_bytes);
    q36_gpu_tensor *scores = NULL;
    q36_gpu_tensor *out = q36_gpu_tensor_alloc(out_floats * sizeof(float));
    TEST_ASSERT(q && qg && kc && vc && out);

    TEST_ASSERT(q36_gpu_tensor_write(q, 0, q_host, q_floats * sizeof(float)) != 0);
    TEST_ASSERT(q36_gpu_tensor_write(qg, 0, qg_host, qg_floats * sizeof(float)) != 0);
    TEST_ASSERT(q36_gpu_tensor_write(kc, 0, kc_host, (uint64_t)kv_max * k_row_bytes) != 0);
    TEST_ASSERT(q36_gpu_tensor_write(vc, 0, vc_host, (uint64_t)kv_max * v_row_bytes) != 0);
    TEST_ASSERT(q36_gpu_set_model_map(sinks_raw, sinks_alloc) != 0);

    TEST_ASSERT(q36_gpu_attn_decode_tensor(out, q, qg, kc, vc, scores,
                                           sinks_raw, sinks_alloc, 0, true,
                                           pos0, n_tok, n_head, n_head_kv,
                                           head_dim, k_type, v_type,
                                           k_row_bytes, v_row_bytes) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(out, 0, out_host, out_floats * sizeof(float)) != 0);

    /* CPU reference, the q36_forward_full_attn tail. */
    float max_err = 0.0f;
    for (uint32_t tok = 0; tok < n_tok; tok++) {
        uint32_t n_kv = pos0 + tok + 1u;
        for (uint32_t h = 0; h < n_head; h++) {
            uint32_t kvh = h / (n_head / n_head_kv);
            const float *qrow = q_host + ((uint64_t)tok * n_head + h) * head_dim;
            float max_score = sinks[h];
            float *w = malloc(n_kv * sizeof(float));
            TEST_ASSERT(w != NULL);
            for (uint32_t t = 0; t < n_kv; t++) {
                double dot = 0.0;
                const uint8_t *krow = kc_host + (uint64_t)t * k_row_bytes;
                for (uint32_t i = 0; i < head_dim; i++)
                    dot += (double)qrow[i] * (double)test_kv_at(krow, k_type, kvh * head_dim + i);
                w[t] = (float)(dot / sqrt((double)head_dim));
                if (w[t] > max_score) max_score = w[t];
            }
            double denom = exp((double)sinks[h] - (double)max_score);
            for (uint32_t t = 0; t < n_kv; t++) {
                w[t] = expf(w[t] - max_score);
                denom += w[t];
            }
            for (uint32_t j = 0; j < head_dim; j++) {
                float acc = 0.0f;
                for (uint32_t t = 0; t < n_kv; t++) {
                    const uint8_t *vrow = vc_host + (uint64_t)t * v_row_bytes;
                    acc += w[t] * test_kv_at(vrow, v_type, kvh * head_dim + j);
                }
                float gate = qg_host[((uint64_t)tok * n_head + h) * head_dim * 2u + head_dim + j];
                float ref = (float)(acc / denom) * (1.0f / (1.0f + expf(-gate)));
                float got = out_host[((uint64_t)tok * n_head + h) * head_dim + j];
                float err = fabsf(got - ref);
                if (err > max_err) max_err = err;
            }
            free(w);
        }
    }
    TEST_ASSERT(max_err < 2.0e-3f);

    /* n_tok invariance: replay each batch row as a decode step and require
     * bit equality with the batch output. */
    for (uint32_t tok = 0; tok < n_tok; tok++) {
        q36_gpu_tensor *qv = q36_gpu_tensor_view(q,
            (uint64_t)tok * n_head * head_dim * sizeof(float),
            (uint64_t)n_head * head_dim * sizeof(float));
        q36_gpu_tensor *qgv = q36_gpu_tensor_view(qg,
            (uint64_t)tok * n_head * head_dim * 2u * sizeof(float),
            (uint64_t)n_head * head_dim * 2u * sizeof(float));
        q36_gpu_tensor *ov = q36_gpu_tensor_view(out,
            (uint64_t)tok * n_head * head_dim * sizeof(float),
            (uint64_t)n_head * head_dim * sizeof(float));
        TEST_ASSERT(qv && qgv && ov);
        TEST_ASSERT(q36_gpu_attn_decode_tensor(ov, qv, qgv, kc, vc, scores,
                                               sinks_raw, sinks_alloc, 0, true,
                                               pos0 + tok, 1, n_head, n_head_kv,
                                               head_dim, k_type, v_type,
                                               k_row_bytes, v_row_bytes) != 0);
        q36_gpu_tensor_free(qv);
        q36_gpu_tensor_free(qgv);
        q36_gpu_tensor_free(ov);
    }
    TEST_ASSERT(q36_gpu_tensor_read(out, 0, out_single, out_floats * sizeof(float)) != 0);
    TEST_ASSERT(memcmp(out_host, out_single, out_floats * sizeof(float)) == 0);
#ifndef Q36_METAL
    if (n_tok <= 8u) {
        q36_gpu_set_micro_batch(true);
        TEST_ASSERT(q36_gpu_attn_decode_tensor(out, q, qg, kc, vc, scores,
            sinks_raw, sinks_alloc, 0, true, pos0, n_tok, n_head, n_head_kv,
            head_dim, k_type, v_type, k_row_bytes, v_row_bytes) != 0);
        q36_gpu_set_micro_batch(false);
        TEST_ASSERT(q36_gpu_tensor_read(out, 0, out_single, out_floats * sizeof(float)) != 0);
        TEST_ASSERT(memcmp(out_host, out_single, out_floats * sizeof(float)) == 0);
    }
#endif

    free(q_host);
    free(qg_host);
    free(kf_host);
    free(vf_host);
    free(kc_host);
    free(vc_host);
    free(out_host);
    free(out_single);
    q36_gpu_tensor_free(q);
    q36_gpu_tensor_free(qg);
    q36_gpu_tensor_free(kc);
    q36_gpu_tensor_free(vc);
    q36_gpu_tensor_free(scores);
    q36_gpu_tensor_free(out);
    free(sinks_raw);
}
#endif

static void test_vulkan_attn_decode_fused(void) {
#ifdef Q36_NO_GPU
    test_skip("vulkan-kernels", "CPU-only build");
#else
    /* Single span, partial query tile. */
    test_vulkan_attn_case(16, 129, 3, Q36_KV_CACHE_F16, Q36_KV_CACHE_F16, 256);
    /* Multiple spans with causal ends landing mid-tile; batch (qtile)
     * and decode (split) must agree bit for bit. */
    test_vulkan_attn_case(16, 2100, 12, Q36_KV_CACHE_F16, Q36_KV_CACHE_F16, 256);
    test_vulkan_attn_case(16, 129, 3, Q36_KV_CACHE_Q4_0, Q36_KV_CACHE_Q4_0, 256);
    test_vulkan_attn_case(12, 510, 3, Q36_KV_CACHE_Q4_0, Q36_KV_CACHE_Q4_0, 256);
    test_vulkan_attn_case(16, 4094, 5, Q36_KV_CACHE_Q4_0, Q36_KV_CACHE_Q4_0, 256);
    test_vulkan_attn_case(12, 129, 3, Q36_KV_CACHE_Q4_0, Q36_KV_CACHE_Q4_0, 64);
    test_vulkan_attn_case(16, 510, 3, Q36_KV_CACHE_Q4_0, Q36_KV_CACHE_Q4_0, 128);
    test_vulkan_attn_case(16, 129, 3, Q36_KV_CACHE_Q8_0, Q36_KV_CACHE_Q4_0, 256);
    test_vulkan_attn_case(16, 1534, 5, Q36_KV_CACHE_Q8_0, Q36_KV_CACHE_Q4_0, 256);
    test_vulkan_attn_case(16, 4094, 257, Q36_KV_CACHE_Q8_0, Q36_KV_CACHE_Q4_0, 256);
    /* Six-head GQA, partial tiles crossing span and span-group boundaries. */
    test_vulkan_attn_case(12, 510, 3, Q36_KV_CACHE_Q8_0, Q36_KV_CACHE_Q4_0, 256);
    test_vulkan_attn_case(12, 4094, 5, Q36_KV_CACHE_Q8_0, Q36_KV_CACHE_Q4_0, 256);
    test_vulkan_attn_case(12, 1534, 5, Q36_KV_CACHE_Q8_0, Q36_KV_CACHE_Q4_0, 256);
    test_vulkan_attn_case(12, 4094, 5, Q36_KV_CACHE_Q4_0, Q36_KV_CACHE_Q4_0, 256);
    test_vulkan_attn_case(12, 129, 3, Q36_KV_CACHE_Q4_0, Q36_KV_CACHE_Q4_0, 256);
    test_vulkan_attn_case(12, 1534, 5, Q36_KV_CACHE_Q4_0, Q36_KV_CACHE_Q4_0, 256);
    test_vulkan_attn_case(24, 8190, 5, Q36_KV_CACHE_Q4_0, Q36_KV_CACHE_Q4_0, 256);
#ifndef Q36_METAL
    /* Verification and decode share 512-key spans. Prefill folds eight
     * spans per group, so compare its long-context bits to a baseline run. */
    q36_gpu_set_micro_batch(true);
    test_vulkan_attn_case(12, 65534, 3, Q36_KV_CACHE_Q8_0, Q36_KV_CACHE_Q4_0, 256);
    q36_gpu_set_micro_batch(true);
    test_vulkan_attn_case(16, 32766, 3, Q36_KV_CACHE_Q8_0, Q36_KV_CACHE_Q4_0, 256);
    q36_gpu_set_micro_batch(false);
#endif
#endif
}

static void test_vulkan_host_overwrite(void) {
#if !defined(Q36_NO_GPU) && !defined(Q36_METAL)
    const float input[] = {3, -5, 7, 11}, zero[4] = {0};
    float got[4];
    q36_gpu_tensor *root = q36_gpu_tensor_alloc(256 + sizeof(input));
    q36_gpu_tensor *src = q36_gpu_tensor_alloc(sizeof(input));
    q36_gpu_tensor *dst = q36_gpu_tensor_alloc(sizeof(input));
    q36_gpu_tensor *z = q36_gpu_tensor_alloc(sizeof(zero));
    q36_gpu_tensor *view = q36_gpu_tensor_view(root, 256, sizeof(input));
    TEST_ASSERT(root && src && dst && z && view);
    if (!root || !src || !dst || !z || !view) goto done;
    TEST_ASSERT(q36_gpu_tensor_write(src, 0, input, sizeof(input)) != 0);
    TEST_ASSERT(q36_gpu_tensor_write(z, 0, zero, sizeof(zero)) != 0);
    for (int use_view = 0; use_view < 2; use_view++) {
        q36_gpu_tensor *a = use_view ? view : root;
        for (int copy = 0; copy < 2; copy++) {
            for (int count = 1; count <= 129; count += 128) {
                TEST_ASSERT(q36_gpu_tensor_copy(a, 0, src, 0, sizeof(input)) != 0);
                TEST_ASSERT(q36_gpu_tensor_read(a, 0, got, sizeof(got)) != 0);
                /* The earlier write is complete. The host must still wait
                 * for new readers, in both open and eagerly submitted batches. */
                for (int i = 0; i < count; i++) {
                    int ok = copy ? q36_gpu_tensor_copy(dst, 0, a, 0, sizeof(input))
                                  : q36_gpu_add_tensor(dst, a, z, 4);
                    TEST_ASSERT(ok != 0);
                }
                TEST_ASSERT(q36_gpu_tensor_write(a, 0, zero, sizeof(zero)) != 0);
                TEST_ASSERT(q36_gpu_tensor_read(dst, 0, got, sizeof(got)) != 0);
                TEST_ASSERT(memcmp(got, input, sizeof(input)) == 0);
            }
        }
    }
done:
    q36_gpu_tensor_free(view);
    q36_gpu_tensor_free(root);
    q36_gpu_tensor_free(src);
    q36_gpu_tensor_free(dst);
    q36_gpu_tensor_free(z);
#endif
}

static void test_mtp_lookup(void) {
    int history[27], ambiguous[43];
    for (int i = 0; i < 12; i++) history[i] = ambiguous[i] = ambiguous[16 + i] = i;
    for (int i = 0; i < 11; i++) history[16 + i] = ambiguous[32 + i] = i;
    for (int i = 0; i < 3; i++) history[12 + i] = ambiguous[12 + i] = ambiguous[28 + i] = 101 + i;
    history[15] = ambiguous[15] = -5;
    ambiguous[31] = -6;
    ambiguous[29] = 999;
    int draft[5] = {-1,-1,-1,-1,-1};
    const int want[] = {11,101,102,103,-1};
    TEST_ASSERT(q36_mtp_lookup(history, 27, 11, draft, 4) == 4);
    TEST_ASSERT(memcmp(draft, want, sizeof(want)) == 0);
    TEST_ASSERT(q36_mtp_lookup(history, 27, 11, draft, 1) == 0);
    TEST_ASSERT(q36_mtp_lookup(history, 5, 11, draft, 4) == 0);
    TEST_ASSERT(q36_mtp_lookup(history, 27, 99, draft, 4) == 0);
    TEST_ASSERT(q36_mtp_lookup(ambiguous, 43, 11, draft, 4) == 0);
    ambiguous[31] = -5;
    TEST_ASSERT(q36_mtp_lookup(ambiguous, 43, 11, draft, 4) == 4);
    TEST_ASSERT(draft[0] == 11 && draft[1] == 101 && draft[2] == 999 && draft[3] == 103);
    ambiguous[29] = 102;
    ambiguous[31] = -6;
    TEST_ASSERT(q36_mtp_lookup(ambiguous, 43, 11, draft, 4) == 4);
    TEST_ASSERT(memcmp(draft, want, sizeof(want)) == 0);
}

static void test_vulkan_top2_rows(void) {
#if !defined(Q36_NO_GPU) && !defined(Q36_METAL)
    enum { count = 33027, rows = 8, blocks = (count + 63) / 64 };
    float *values = malloc((size_t)count * rows * sizeof(float));
    int32_t got[rows * 2 + 1];
    uint32_t ages[blocks];
    q36_gpu_tensor *x = q36_gpu_tensor_alloc((uint64_t)count * rows * sizeof(float));
    q36_gpu_tensor *ids = q36_gpu_tensor_alloc(sizeof(got));
    q36_gpu_tensor *mask = q36_gpu_tensor_alloc(sizeof(ages));
    TEST_ASSERT(values && x && ids && mask);
    if (!values || !x || !ids || !mask) goto done;
    for (unsigned row = 0; row < rows; row++) {
        float *v = values + row * count;
        for (unsigned i = 0; i < count; i++) v[i] = -50.0f;
        v[5] = 20.0f;
        v[200 + row] = v[32768 + row] = 10.0f;
    }
    TEST_ASSERT(q36_gpu_tensor_write(x, 0, values, (uint64_t)count * rows * sizeof(float)) != 0);
    for (unsigned i = 0; i < rows * 2 + 1; i++) got[i] = -99;
    TEST_ASSERT(q36_gpu_tensor_write(ids, 0, got, sizeof(got)) != 0);
    TEST_ASSERT(q36_gpu_top2_rows_tensor(ids, x, count, rows, -1) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(ids, 0, got, sizeof(got)) != 0);
    for (unsigned row = 0; row < rows; row++) {
        TEST_ASSERT(got[2 * row] == 5);
        TEST_ASSERT(got[2 * row + 1] == 200 + (int)row);
    }
    TEST_ASSERT(got[rows * 2] == -99);
    TEST_ASSERT(q36_gpu_top2_rows_tensor(ids, x, count, rows, 5) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(ids, 0, got, sizeof(got)) != 0);
    for (unsigned row = 0; row < rows; row++) {
        TEST_ASSERT(got[2 * row] == 200 + (int)row);
        TEST_ASSERT(got[2 * row + 1] == 32768 + (int)row);
    }
    TEST_ASSERT(q36_gpu_mtp_head_mask_tensor(mask, x, ids, count, rows) != 0);
    TEST_ASSERT(q36_gpu_tensor_read(mask, 0, ages, sizeof(ages)) != 0);
    for (unsigned i = 0; i < blocks; i++) TEST_ASSERT(ages[i] == (i <= 512 ? 4u : 0u));
    /* Once the target alternatives move away, their blocks age out. */
    TEST_ASSERT(q36_gpu_top2_rows_tensor(ids, x, count, rows, -1) != 0);
    for (unsigned step = 1; step <= 4; step++) {
        TEST_ASSERT(q36_gpu_mtp_head_mask_tensor(mask, x, ids, count, rows) != 0);
        TEST_ASSERT(q36_gpu_tensor_read(mask, 0, ages, sizeof(ages)) != 0);
        TEST_ASSERT(ages[512] == 4u - step);
        TEST_ASSERT(ages[511] == 4u && ages[blocks - 1] == 0u);
    }
done:
    free(values);
    q36_gpu_tensor_free(x);
    q36_gpu_tensor_free(ids);
    q36_gpu_tensor_free(mask);
    q36_gpu_cleanup();
#endif
}

static void test_vulkan_router_subgroups(void) {
#if !defined(Q36_NO_GPU) && !defined(Q36_METAL)
    enum { tokens = 7, experts = 256, used = 8 };
    float input[tokens * experts], got_weights[tokens * used], want_weights[tokens * used];
    uint32_t got[tokens * used], want[tokens * used];
    q36_gpu_tensor *logits = q36_gpu_tensor_alloc(sizeof(input));
    q36_gpu_tensor *selected = q36_gpu_tensor_alloc(sizeof(got));
    q36_gpu_tensor *weights = q36_gpu_tensor_alloc(sizeof(got_weights));
    TEST_ASSERT(logits && selected && weights);
    char *old = getenv("Q36_VK_ROUTER_SUBGROUPS") ? strdup(getenv("Q36_VK_ROUTER_SUBGROUPS")) : NULL;
    if (!logits || !selected || !weights) goto done;
    const unsigned counts[] = {5, 32, 255, 256};
    for (unsigned c = 0; c < sizeof(counts) / sizeof(counts[0]); c++) {
        unsigned n = counts[c], k = n < used ? n : used;
        for (unsigned fixture = 0; fixture < 3; fixture++) {
            for (unsigned i = 0; i < tokens * n; i++)
                input[i] = fixture == 0 ? 0.0f : fixture == 1 ?
                    (float)((int)(i * 13u % 29u) - 14) : sinf(i * 0.37f) * 1000.0f;
            TEST_ASSERT(q36_gpu_tensor_write(logits, 0, input, tokens * n * sizeof(float)) != 0);
            for (unsigned arm = 0; arm < 2; arm++) {
                setenv("Q36_VK_ROUTER_SUBGROUPS", arm ? "1" : "0", 1);
                TEST_ASSERT(q36_gpu_router_topk_tensor(selected, weights, logits, n, k, tokens, 0.75f) != 0);
                TEST_ASSERT(q36_gpu_tensor_read(selected, 0, arm ? got : want, tokens * k * sizeof(uint32_t)) != 0);
                TEST_ASSERT(q36_gpu_tensor_read(weights, 0, arm ? got_weights : want_weights, tokens * k * sizeof(float)) != 0);
            }
            TEST_ASSERT(memcmp(got, want, tokens * k * sizeof(uint32_t)) == 0);
            TEST_ASSERT(memcmp(got_weights, want_weights, tokens * k * sizeof(float)) == 0);
        }
    }
done:
    if (old) setenv("Q36_VK_ROUTER_SUBGROUPS", old, 1);
    else unsetenv("Q36_VK_ROUTER_SUBGROUPS");
    free(old);
    q36_gpu_tensor_free(weights);
    q36_gpu_tensor_free(selected);
    q36_gpu_tensor_free(logits);
    q36_gpu_cleanup();
#endif
}

static void test_vulkan_kernels(void) {
    test_vulkan_router_subgroups();
    test_vulkan_top2_rows();
    test_vulkan_host_overwrite();
    test_metal_f16_matvec_fast_nr0_4();
    test_bf16_matmul();
    test_vulkan_f32_matvec();
    test_vulkan_rms_norm();
    test_vulkan_split_norm();
    test_vulkan_add_rms_q8();
    test_vulkan_rms_norm_rope();
    test_vulkan_rms_norm_rope_kv();
    test_vulkan_kv_store_quant();
    test_vulkan_rms_norm_rope_kv_quant();
    test_vulkan_recurrent_conv_decode_case(8192);
    test_vulkan_recurrent_conv_decode_case(10240);
    test_vulkan_delta_verify();
    test_vulkan_delta_qkv();
    test_vulkan_shared_ffn_decode();
    test_vulkan_kquant_matvec();
    test_vulkan_moe_q4();
    test_vulkan_moe_gemm_case();
    test_vulkan_swiglu_add();
    test_vulkan_directional_steering();
    test_vulkan_attn_decode_fused();
    const uint32_t wide_types[] = {10, 11, 12, 13, 16, 17, 18, 19, 20, 22, 23},
                   wide_bytes[] = {84, 110, 144, 176, 66, 74, 98, 50, 18, 82, 136};
    for (unsigned i = 0; i < sizeof(wide_types) / sizeof(wide_types[0]); i++)
        test_vulkan_wide_tiles(wide_types[i], wide_bytes[i], 131, wide_types[i] < 16 ? 4163 : 67);
    test_vulkan_wide_tiles(18, 98, 515, 8355);
    test_vulkan_wide_tiles(21, 110, 515, 8355);
    test_vulkan_wide_tiles(16, 66, 515, 8355);
    test_vulkan_wide_tiles(17, 74, 515, 8355);
    test_vulkan_wide_tiles(22, 82, 515, 8355);
    test_vulkan_wide_tiles(23, 136, 515, 8355);
#if !defined(Q36_NO_GPU) && !defined(Q36_METAL)
    const uint32_t decode_types[] = {11, 12, 13, 14, 18, 20, 21, 22},
                   decode_bytes[] = {110, 144, 176, 210, 98, 18, 110, 82};
    for (unsigned i = 0; i < sizeof(decode_types) / sizeof(decode_types[0]); i++) {
        test_vulkan_decode_one(decode_types[i], decode_bytes[i], 768, 1024);
        test_vulkan_decode_one(decode_types[i], decode_bytes[i], 5120, 1024);
    }
    test_vulkan_decode_one(21, 110, 5120, 5120);
    test_vulkan_decode_one(21, 110, 17408, 5120);
    test_vulkan_decode_one(12, 144, 5120, 4096);
    test_vulkan_decode_one(13, 176, 5120, 4096);
    test_vulkan_decode_one(14, 210, 6144, 4096);
    test_vulkan_decode_one(14, 210, 17408, 4096);
#endif
    test_vulkan_small_tiles();
}

static char *test_read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    char *s = malloc((size_t)len + 1);
    if (!s) {
        fclose(fp);
        return NULL;
    }
    size_t nread = fread(s, 1, (size_t)len, fp);
    fclose(fp);
    if (nread != (size_t)len) {
        free(s);
        return NULL;
    }
    s[len] = '\0';
    return s;
}

static int test_count_substr(const char *s, const char *needle) {
    int count = 0;
    size_t n = strlen(needle);
    const char *p = s;
    while ((p = strstr(p, needle)) != NULL) {
        count++;
        p += n;
    }
    return count;
}

static int test_hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static bool test_hex_to_bytes(const char *hex, unsigned char *out, int cap, int *len) {
    int n = 0;
    if (hex[0] == '-' && (!hex[1] || isspace((unsigned char)hex[1]))) {
        *len = 0;
        return true;
    }
    while (*hex && !isspace((unsigned char)*hex)) {
        int hi = test_hex_digit(hex[0]);
        int lo = test_hex_digit(hex[1]);
        if (hi < 0 || lo < 0 || n >= cap) return false;
        out[n++] = (unsigned char)((hi << 4) | lo);
        hex += 2;
    }
    *len = n;
    return true;
}

static bool test_token_bytes_equal(q36_engine *engine, int token,
                                   const unsigned char *want, int want_len) {
    size_t got_len = 0;
    char *got = q36_token_text(engine, token, &got_len);
    bool eq = got && got_len == (size_t)want_len &&
              memcmp(got, want, (size_t)want_len) == 0;
    free(got);
    return eq;
}

static int test_find_token_by_bytes(q36_engine *engine, int n_vocab,
                                    const unsigned char *want, int want_len) {
    for (int id = 0; id < n_vocab; id++) {
        if (test_token_bytes_equal(engine, id, want, want_len)) return id;
    }
    return -1;
}

static void test_long_prefill_progress(void *ud, const char *event, int current, int total) {
    (void)ud;
    if (strcmp(event, "prefill_chunk")) return;
    if (current == 0 || current == total || current % 8192 == 0) {
        fprintf(stderr, "q36-test: long-context prefill %d/%d\n", current, total);
    }
}

static void test_long_security_continuation(void) {
    if (!test_model_available(test_model_path())) {
        test_skip("long-context", "model file not found");
        return;
    }
    const char *prompt_path = getenv("Q36_TEST_LONG_PROMPT");
    if (!prompt_path || !prompt_path[0]) {
        prompt_path = "tests/long_context_security_prompt.txt";
    }
    char *prompt_text = test_read_file(prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    q36_backend backend = test_default_backend();
    q36_engine_options opt = {
        .model_path = test_model_path(),
        .backend = backend,
        .cache_type_k = q36_default_kv_cache_type_k(backend, false),
        .cache_type_v = q36_default_kv_cache_type_v(backend, false),
    };
    q36_engine *engine = NULL;
    TEST_ASSERT(q36_engine_open(&engine, &opt) == 0);
    if (!engine) {
        free(prompt_text);
        return;
    }
    if (!test_require_session_backend("long-context", engine, 100000)) {
        q36_engine_close(engine);
        free(prompt_text);
        return;
    }

    q36_tokens prompt = {0};
    q36_tokenize_rendered_chat(engine, prompt_text, &prompt);
    TEST_ASSERT(prompt.len > 30000);

    q36_session *session = NULL;
    TEST_ASSERT(q36_session_create(&session, engine, 100000) == 0);
    if (!session) {
        q36_tokens_free(&prompt);
        q36_engine_close(engine);
        free(prompt_text);
        return;
    }

    char err[160];
    q36_session_set_progress(session, test_long_prefill_progress, NULL);
    TEST_ASSERT(q36_session_sync(session, &prompt, err, sizeof(err)) == 0);
    q36_session_set_progress(session, NULL, NULL);

    buf out = {0};
    uint64_t rng = 12345;
    int generated = 0;
    bool decode_ok = true;
    for (; generated < 700; generated++) {
        int token = q36_session_sample(session, 0.8f, 40, 0.95f, 0.05f, &rng);
        if (token == q36_token_eos(engine)) break;

        size_t piece_len = 0;
        char *piece = q36_token_text(engine, token, &piece_len);
        buf_append(&out, piece, piece_len);
        free(piece);

        if (q36_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }

    const char *text = out.ptr ? out.ptr : "";
    const char *answer = strstr(text, "</think>");
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(generated > 0);
    TEST_ASSERT(answer != NULL);
    TEST_ASSERT(test_count_substr(text, "</think>") == 1);
    answer = answer ? answer + strlen("</think>") : text;
    TEST_ASSERT(test_count_substr(answer, "The most critical security issue") == 1);
    TEST_ASSERT(strstr(answer, "arbitrary file") != NULL);

    buf_free(&out);
    q36_session_free(session);
    q36_tokens_free(&prompt);
    q36_engine_close(engine);
    free(prompt_text);
}

#define TEST_VEC_MAX_STEPS 16
#define TEST_VEC_MAX_TOP 32
#define TEST_VEC_MAX_TOKEN_BYTES 128
/* Cross-quant gates: the default 2-bit-expert model scored against the
 * Q8_0 llama.cpp reference continuation, stored-reference continuation style. */
#define TEST_VEC_AVG_NLL_GATE 4.0
#define TEST_VEC_TOP20_MISS_GATE 1

typedef struct {
    unsigned char bytes[TEST_VEC_MAX_TOKEN_BYTES];
    int len;
    float logprob;
} test_vec_top;

typedef struct {
    unsigned char selected[TEST_VEC_MAX_TOKEN_BYTES];
    int selected_len;
    int ntop;
    test_vec_top top[TEST_VEC_MAX_TOP];
} test_vec_step;

typedef struct {
    char id[96];
    char prompt_path[512];
    int ctx;
    int nsteps;
    test_vec_step steps[TEST_VEC_MAX_STEPS];
} test_vec_case;

static bool test_read_vector_case(FILE *fp, test_vec_case *vc);
static bool test_fill_vector_case(FILE *fp, test_vec_case *vc);

typedef struct {
    char *id;
    char *prompt_file;
    char *llamacpp_file;
    int prompt_chars;
    int steps;
} test_vec_manifest_entry;

typedef struct {
    test_vec_manifest_entry *v;
    int len;
    int cap;
    char *template;
    char *checkpoint;
} test_vec_manifest;

typedef struct {
    test_vec_case vc;
    char *prompt;
} test_llamacpp_record;

static char *test_trim_line(char *line) {
    while (*line && isspace((unsigned char)*line)) line++;
    size_t n = strlen(line);
    while (n && isspace((unsigned char)line[n - 1])) line[--n] = '\0';
    return line;
}

static char *test_parent_dir(const char *path) {
    const char *slash = strrchr(path, '/');
    size_t n = slash ? (size_t)(slash - path) : 1;
    char *out = malloc(n + 1);
    TEST_ASSERT(out != NULL);
    if (!out) exit(1);
    if (slash) memcpy(out, path, n);
    else out[0] = '.';
    out[n] = '\0';
    return out;
}

static char *test_join_path(const char *dir, const char *file) {
    size_t ndir = strlen(dir);
    size_t nfile = strlen(file);
    bool need_slash = ndir > 0 && dir[ndir - 1] != '/';
    char *out = malloc(ndir + need_slash + nfile + 1);
    TEST_ASSERT(out != NULL);
    if (!out) exit(1);
    memcpy(out, dir, ndir);
    if (need_slash) out[ndir++] = '/';
    memcpy(out + ndir, file, nfile + 1);
    return out;
}

static bool test_json_expect(const char **p, char c) {
    json_ws(p);
    if (**p != c) return false;
    (*p)++;
    return true;
}

static void test_vec_manifest_free(test_vec_manifest *manifest) {
    if (!manifest) return;
    for (int i = 0; i < manifest->len; i++) {
        free(manifest->v[i].id);
        free(manifest->v[i].prompt_file);
        free(manifest->v[i].llamacpp_file);
    }
    free(manifest->v);
    free(manifest->template);
    free(manifest->checkpoint);
    memset(manifest, 0, sizeof(*manifest));
}

static bool test_parse_bytes_array(const char **p, unsigned char *out, int cap, int *len) {
    int n = 0;
    if (!test_json_expect(p, '[')) return false;
    json_ws(p);
    if (**p == ']') {
        (*p)++;
        *len = 0;
        return true;
    }
    for (;;) {
        int v = 0;
        if (!json_int(p, &v)) return false;
        if (n >= cap || v < 0 || v > 255) return false;
        out[n++] = (unsigned char)v;
        json_ws(p);
        if (**p == ']') {
            (*p)++;
            *len = n;
            return true;
        }
        if (**p != ',') return false;
        (*p)++;
    }
}

static bool test_parse_token_bytes_object(const char **p,
                                          unsigned char *out,
                                          int cap,
                                          int *len) {
    if (!test_json_expect(p, '{')) return false;
    *len = 0;
    json_ws(p);
    if (**p == '}') {
        (*p)++;
        return true;
    }
    for (;;) {
        char *key = NULL;
        bool ok = false;
        if (!json_string(p, &key)) return false;
        if (!test_json_expect(p, ':')) goto done;
        if (!strcmp(key, "bytes")) ok = test_parse_bytes_array(p, out, cap, len);
        else ok = json_skip_value(p);
done:
        free(key);
        if (!ok) return false;
        json_ws(p);
        if (**p == '}') {
            (*p)++;
            return true;
        }
        if (**p != ',') return false;
        (*p)++;
    }
}

static bool test_parse_top_logprobs_array(const char **p, test_vec_step *step) {
    if (!test_json_expect(p, '[')) return false;
    step->ntop = 0;
    json_ws(p);
    if (**p == ']') {
        (*p)++;
        return true;
    }
    for (;;) {
        unsigned char bytes[TEST_VEC_MAX_TOKEN_BYTES];
        int len = 0;
        float logprob = Q36_NEG_INF;
        if (!test_json_expect(p, '{')) return false;
        json_ws(p);
        if (**p != '}') {
            for (;;) {
                char *key = NULL;
                bool ok = false;
                if (!json_string(p, &key)) return false;
                if (!test_json_expect(p, ':')) goto alt_done;
                if (!strcmp(key, "token")) ok = test_parse_token_bytes_object(p, bytes, TEST_VEC_MAX_TOKEN_BYTES, &len);
                else if (!strcmp(key, "logprob")) {
                    double v = 0.0;
                    ok = json_number(p, &v);
                    logprob = (float)v;
                } else ok = json_skip_value(p);
alt_done:
                free(key);
                if (!ok) return false;
                json_ws(p);
                if (**p == '}') break;
                if (**p != ',') return false;
                (*p)++;
            }
        }
        (*p)++;
        if (logprob > -1000.0f && len > 0) {
            if (step->ntop >= TEST_VEC_MAX_TOP) return false;
            memcpy(step->top[step->ntop].bytes, bytes, (size_t)len);
            step->top[step->ntop].len = len;
            step->top[step->ntop].logprob = logprob;
            step->ntop++;
        }
        json_ws(p);
        if (**p == ']') {
            (*p)++;
            return true;
        }
        if (**p != ',') return false;
        (*p)++;
    }
}

static bool test_parse_llamacpp_step(const char **p, test_vec_step *step) {
    if (!test_json_expect(p, '{')) return false;
    memset(step, 0, sizeof(*step));
    json_ws(p);
    if (**p == '}') {
        (*p)++;
        return true;
    }
    for (;;) {
        char *key = NULL;
        bool ok = false;
        if (!json_string(p, &key)) return false;
        if (!test_json_expect(p, ':')) goto step_done;
        if (!strcmp(key, "token")) {
            ok = test_parse_token_bytes_object(p,
                                               step->selected,
                                               TEST_VEC_MAX_TOKEN_BYTES,
                                               &step->selected_len);
        } else if (!strcmp(key, "top_logprobs")) ok = test_parse_top_logprobs_array(p, step);
        else ok = json_skip_value(p);
step_done:
        free(key);
        if (!ok) return false;
        json_ws(p);
        if (**p == '}') {
            (*p)++;
            return true;
        }
        if (**p != ',') return false;
        (*p)++;
    }
}

static bool test_parse_manifest_entry(const char **p, test_vec_manifest_entry *entry) {
    if (!test_json_expect(p, '{')) return false;
    memset(entry, 0, sizeof(*entry));
    json_ws(p);
    if (**p == '}') {
        (*p)++;
        return true;
    }
    for (;;) {
        char *key = NULL;
        bool ok = false;
        if (!json_string(p, &key)) return false;
        if (!test_json_expect(p, ':')) goto done;
        if (!strcmp(key, "id")) ok = json_string(p, &entry->id);
        else if (!strcmp(key, "prompt_file")) ok = json_string(p, &entry->prompt_file);
        else if (!strcmp(key, "llamacpp_file")) ok = json_string(p, &entry->llamacpp_file);
        else if (!strcmp(key, "prompt_chars")) ok = json_int(p, &entry->prompt_chars);
        else if (!strcmp(key, "steps")) ok = json_int(p, &entry->steps);
        else ok = json_skip_value(p);
done:
        free(key);
        if (!ok) return false;
        json_ws(p);
        if (**p == '}') {
            (*p)++;
            return true;
        }
        if (**p != ',') return false;
        (*p)++;
    }
}

static bool test_parse_manifest(const char *json, test_vec_manifest *manifest) {
    const char *p = json;
    char *key = NULL;
    memset(manifest, 0, sizeof(*manifest));
    if (!test_json_expect(&p, '{')) return false;
    json_ws(&p);
    if (*p == '}') {
        p++;
        return true;
    }
    for (;;) {
        char *key = NULL;
        bool ok = false;
        if (!json_string(&p, &key)) goto fail;
        if (!test_json_expect(&p, ':')) goto fail;
        if (!strcmp(key, "checkpoint")) {
            ok = json_string(&p, &manifest->checkpoint);
        } else if (!strcmp(key, "template")) {
            ok = json_string(&p, &manifest->template);
        } else if (!strcmp(key, "prompts")) {
            if (!test_json_expect(&p, '[')) goto fail;
            json_ws(&p);
            if (*p != ']') {
                for (;;) {
                    test_vec_manifest_entry entry;
                    if (!test_parse_manifest_entry(&p, &entry)) goto fail;
                    if (manifest->len >= manifest->cap) {
                        manifest->cap = manifest->cap ? manifest->cap * 2 : 8;
                        manifest->v = test_xrealloc(manifest->v,
                                                    (size_t)manifest->cap * sizeof(*manifest->v));
                    }
                    manifest->v[manifest->len++] = entry;
                    json_ws(&p);
                    if (*p == ']') break;
                    if (*p != ',') goto fail;
                    p++;
                }
            }
            p++;
            ok = true;
        } else ok = json_skip_value(&p);
        free(key);
        if (!ok) goto fail;
        json_ws(&p);
        if (*p == '}') {
            p++;
            json_ws(&p);
            if (*p) goto fail;
            return true;
        }
        if (*p != ',') goto fail;
        p++;
    }
fail:
    free(key);
    test_vec_manifest_free(manifest);
    return false;
}

static bool test_parse_llamacpp_record(const char *json, test_llamacpp_record *record) {
    const char *p = json;
    char *key = NULL;
    memset(record, 0, sizeof(*record));
    if (!test_json_expect(&p, '{')) return false;
    json_ws(&p);
    if (*p == '}') {
        p++;
        return true;
    }
    for (;;) {
        char *key = NULL;
        bool ok = false;
        if (!json_string(&p, &key)) goto fail;
        if (!test_json_expect(&p, ':')) goto fail;
        if (!strcmp(key, "id")) {
            char *id = NULL;
            ok = json_string(&p, &id);
            if (ok) {
                snprintf(record->vc.id, sizeof(record->vc.id), "%s", id);
                free(id);
            }
        } else if (!strcmp(key, "prompt")) ok = json_string(&p, &record->prompt);
        else if (!strcmp(key, "request")) {
            if (!test_json_expect(&p, '{')) goto fail;
            json_ws(&p);
            ok = true;
            if (*p != '}') {
                for (;;) {
                    char *req_key = NULL;
                    if (!json_string(&p, &req_key)) {
                        ok = false;
                        break;
                    }
                    if (!test_json_expect(&p, ':')) {
                        free(req_key);
                        ok = false;
                        break;
                    }
                    if (!strcmp(req_key, "ctx")) ok = json_int(&p, &record->vc.ctx);
                    else ok = json_skip_value(&p);
                    free(req_key);
                    if (!ok) break;
                    json_ws(&p);
                    if (*p == '}') break;
                    if (*p != ',') {
                        ok = false;
                        break;
                    }
                    p++;
                }
            }
            if (!ok || *p != '}') goto fail;
            p++;
        } else if (!strcmp(key, "steps")) {
            if (!test_json_expect(&p, '[')) goto fail;
            json_ws(&p);
            ok = true;
            if (*p != ']') {
                for (;;) {
                    if (record->vc.nsteps >= TEST_VEC_MAX_STEPS) goto fail;
                    if (!test_parse_llamacpp_step(&p, &record->vc.steps[record->vc.nsteps++])) goto fail;
                    json_ws(&p);
                    if (*p == ']') break;
                    if (*p != ',') goto fail;
                    p++;
                }
            }
            p++;
        } else ok = json_skip_value(&p);
        free(key);
        if (!ok) goto fail;
        json_ws(&p);
        if (*p == '}') {
            p++;
            json_ws(&p);
            return *p == '\0';
        }
        if (*p != ',') goto fail;
        p++;
    }
fail:
    free(key);
    free(record->prompt);
    memset(record, 0, sizeof(*record));
    return false;
}

static void test_llamacpp_record_free(test_llamacpp_record *record) {
    if (!record) return;
    free(record->prompt);
    memset(record, 0, sizeof(*record));
}

static bool test_load_vector_cases(const char *path, test_vec_case **out, int *count) {
    FILE *fp = fopen(path, "rb");
    test_vec_case *cases = NULL;
    int len = 0;
    int cap = 0;
    if (!fp) return false;
    while (1) {
        test_vec_case vc;
        if (!test_read_vector_case(fp, &vc)) break;
        if (!test_fill_vector_case(fp, &vc)) {
            free(cases);
            fclose(fp);
            return false;
        }
        if (len >= cap) {
            cap = cap ? cap * 2 : 8;
            cases = test_xrealloc(cases, (size_t)cap * sizeof(*cases));
        }
        cases[len++] = vc;
    }
    fclose(fp);
    *out = cases;
    *count = len;
    return true;
}

static test_vec_case *test_find_vector_case(test_vec_case *cases, int count, const char *id) {
    for (int i = 0; i < count; i++) {
        if (!strcmp(cases[i].id, id)) return &cases[i];
    }
    return NULL;
}

static bool test_compare_vector_case(const test_vec_case *vec,
                                     const test_llamacpp_record *raw,
                                     const char *id) {
    if (vec->ctx != raw->vc.ctx || vec->nsteps != raw->vc.nsteps) {
        fprintf(stderr, "q36-test: fixture %s header mismatch vec(ctx=%d steps=%d) json(ctx=%d steps=%d)\n",
                id, vec->ctx, vec->nsteps, raw->vc.ctx, raw->vc.nsteps);
        return false;
    }
    for (int i = 0; i < vec->nsteps; i++) {
        const test_vec_step *a = &vec->steps[i];
        const test_vec_step *b = &raw->vc.steps[i];
        if (a->selected_len != b->selected_len ||
            memcmp(a->selected, b->selected, (size_t)a->selected_len) != 0) {
            fprintf(stderr, "q36-test: fixture %s step %d selected token mismatch\n", id, i);
            return false;
        }
        if (a->ntop != b->ntop) {
            fprintf(stderr, "q36-test: fixture %s step %d top count mismatch vec=%d json=%d\n",
                    id, i, a->ntop, b->ntop);
            return false;
        }
        for (int j = 0; j < a->ntop; j++) {
            if (a->top[j].len != b->top[j].len ||
                memcmp(a->top[j].bytes, b->top[j].bytes, (size_t)a->top[j].len) != 0) {
                fprintf(stderr, "q36-test: fixture %s step %d top %d token mismatch\n", id, i, j);
                return false;
            }
            if (fabsf(a->top[j].logprob - b->top[j].logprob) > 1.0e-5f) {
                fprintf(stderr,
                        "q36-test: fixture %s step %d top %d logprob mismatch vec=%g json=%g\n",
                        id, i, j, a->top[j].logprob, b->top[j].logprob);
                return false;
            }
        }
    }
    return true;
}

static bool test_validate_reference_vector_fixture(const char *vec_path,
                                                   bool *legacy_template) {
    char *root = test_parent_dir(vec_path);
    char *manifest_path = test_join_path(root, "manifest.json");
    char *manifest_text = test_read_file(manifest_path);
    test_vec_manifest manifest = {0};
    test_vec_case *cases = NULL;
    int ncases = 0;
    bool ok = false;

    if (!manifest_text) {
        fprintf(stderr, "q36-test: missing vector manifest %s\n", manifest_path);
        goto done;
    }
    if (!test_parse_manifest(manifest_text, &manifest)) {
        fprintf(stderr, "q36-test: failed to parse vector manifest %s\n", manifest_path);
        goto done;
    }
    const char *checkpoint = strrchr(root, '/');
    checkpoint = checkpoint ? checkpoint + 1 : root;
    if (!manifest.checkpoint || strcmp(manifest.checkpoint, checkpoint)) {
        fprintf(stderr, "q36-test: fixture checkpoint mismatch: directory=%s manifest=%s\n",
                checkpoint, manifest.checkpoint ? manifest.checkpoint : "missing");
        goto done;
    }
    if (legacy_template)
        *legacy_template = manifest.template &&
                           !strcmp(manifest.template, "legacy-q36-nothink");
    if (!test_load_vector_cases(vec_path, &cases, &ncases)) {
        fprintf(stderr, "q36-test: failed to parse vector fixture %s\n", vec_path);
        goto done;
    }
    if (ncases != manifest.len) {
        fprintf(stderr, "q36-test: vector case count mismatch vec=%d manifest=%d\n",
                ncases, manifest.len);
        goto done;
    }

    ok = true;
    for (int i = 0; i < manifest.len; i++) {
        test_vec_manifest_entry *entry = &manifest.v[i];
        test_vec_case *vec = test_find_vector_case(cases, ncases, entry->id);
        test_llamacpp_record raw = {0};
        char *prompt_path = NULL;
        char *raw_path = NULL;
        char *prompt_text = NULL;
        char *vec_prompt_text = NULL;
        char *raw_text = NULL;

        if (!vec) {
            fprintf(stderr, "q36-test: fixture %s missing from %s\n", entry->id, vec_path);
            ok = false;
            break;
        }
        prompt_path = test_join_path(root, entry->prompt_file);
        raw_path = test_join_path(root, entry->llamacpp_file);
        prompt_text = test_read_file(prompt_path);
        vec_prompt_text = test_read_file(vec->prompt_path);
        raw_text = test_read_file(raw_path);
        if (!prompt_text || !vec_prompt_text || !raw_text) {
            fprintf(stderr, "q36-test: fixture %s missing prompt/json payload\n", entry->id);
            ok = false;
            goto case_done;
        }
        if ((int)strlen(prompt_text) != entry->prompt_chars) {
            fprintf(stderr,
                    "q36-test: fixture %s prompt_chars mismatch manifest=%d actual=%zu\n",
                    entry->id, entry->prompt_chars, strlen(prompt_text));
            ok = false;
            goto case_done;
        }
        if (strcmp(prompt_text, vec_prompt_text) != 0) {
            fprintf(stderr, "q36-test: fixture %s prompt file mismatch between manifest and vec\n", entry->id);
            ok = false;
            goto case_done;
        }
        if (!test_parse_llamacpp_record(raw_text, &raw)) {
            fprintf(stderr, "q36-test: failed to parse raw llama fixture %s\n", raw_path);
            ok = false;
            goto case_done;
        }
        if (strcmp(raw.vc.id, entry->id) != 0) {
            fprintf(stderr, "q36-test: fixture %s raw json id mismatch (%s)\n", entry->id, raw.vc.id);
            ok = false;
            goto case_done;
        }
        if (strcmp(prompt_text, raw.prompt ? raw.prompt : "") != 0) {
            fprintf(stderr, "q36-test: fixture %s raw json prompt mismatch\n", entry->id);
            ok = false;
            goto case_done;
        }
        if (entry->steps != vec->nsteps || entry->steps != raw.vc.nsteps) {
            fprintf(stderr,
                    "q36-test: fixture %s step count mismatch manifest=%d vec=%d json=%d\n",
                    entry->id, entry->steps, vec->nsteps, raw.vc.nsteps);
            ok = false;
            goto case_done;
        }
        if (!test_compare_vector_case(vec, &raw, entry->id)) {
            ok = false;
            goto case_done;
        }

case_done:
        free(prompt_path);
        free(raw_path);
        free(prompt_text);
        free(vec_prompt_text);
        free(raw_text);
        test_llamacpp_record_free(&raw);
        if (!ok) break;
    }

done:
    free(root);
    free(manifest_path);
    free(manifest_text);
    free(cases);
    test_vec_manifest_free(&manifest);
    return ok;
}

static bool test_read_vector_case(FILE *fp, test_vec_case *vc) {
    char line[2048];
    memset(vc, 0, sizeof(*vc));
    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (sscanf(p, "case %95s %d %d %511s",
                   vc->id, &vc->ctx, &vc->nsteps, vc->prompt_path) == 4) {
            TEST_ASSERT(vc->nsteps > 0 && vc->nsteps <= TEST_VEC_MAX_STEPS);
            return true;
        }
        TEST_ASSERT(!"unexpected line before vector case");
    }
    return false;
}

static bool test_fill_vector_case(FILE *fp, test_vec_case *vc) {
    char line[2048];
    int step_index = -1;
    int top_index = 0;

    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (!strcmp(p, "end")) return true;

        if (!strncmp(p, "step ", 5)) {
            char hex[TEST_VEC_MAX_TOKEN_BYTES * 2 + 2];
            int ntop = 0;
            if (sscanf(p, "step %d %257s %d", &step_index, hex, &ntop) != 3) {
                TEST_ASSERT(!"bad vector step line");
                return false;
            }
            TEST_ASSERT(step_index >= 0 && step_index < vc->nsteps);
            TEST_ASSERT(ntop >= 0 && ntop <= TEST_VEC_MAX_TOP);
            vc->steps[step_index].ntop = ntop;
            TEST_ASSERT(test_hex_to_bytes(hex,
                                          vc->steps[step_index].selected,
                                          TEST_VEC_MAX_TOKEN_BYTES,
                                          &vc->steps[step_index].selected_len));
            top_index = 0;
            continue;
        }

        if (!strncmp(p, "top ", 4)) {
            char hex[TEST_VEC_MAX_TOKEN_BYTES * 2 + 2];
            float lp = 0.0f;
            TEST_ASSERT(step_index >= 0 && step_index < vc->nsteps);
            TEST_ASSERT(top_index < vc->steps[step_index].ntop);
            if (sscanf(p, "top %257s %f", hex, &lp) != 2) {
                TEST_ASSERT(!"bad vector top line");
                return false;
            }
            test_vec_top *top = &vc->steps[step_index].top[top_index++];
            top->logprob = lp;
            TEST_ASSERT(test_hex_to_bytes(hex, top->bytes,
                                          TEST_VEC_MAX_TOKEN_BYTES, &top->len));
            continue;
        }

        TEST_ASSERT(!"unexpected vector line");
        return false;
    }

    TEST_ASSERT(!"unterminated vector case");
    return false;
}

static void test_logprob_vector_case(q36_engine *engine, const test_vec_case *vc,
                                     bool hf_template, bool legacy_template) {
    char *prompt_text = test_read_file(vc->prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    q36_tokens prompt = {0};
    if (legacy_template) {
        char *rendered = test_render_legacy_chat_prompt(prompt_text);
        q36_tokenize_rendered_chat(engine, rendered, &prompt);
        free(rendered);
    } else {
        test_encode_prompt_mode(engine, "", prompt_text, Q36_THINK_NONE,
                                hf_template, &prompt);
    }
    free(prompt_text);

    q36_session *session = NULL;
    TEST_ASSERT(q36_session_create(&session, engine, vc->ctx) == 0);
    if (!session) {
        q36_tokens_free(&prompt);
        return;
    }

    char err[160];
    TEST_ASSERT(q36_session_sync(session, &prompt, err, sizeof(err)) == 0);

    /* stored-reference continuation scoring (gguf-tools/quality-testing): the
     * fixture is a stronger reference quant (llama.cpp Q8_0), so exact
     * token equality cannot hold for the 2-bit-expert default model.
     * Follow the reference trajectory and measure how much probability the
     * local model assigns to each reference token; a kernel regression
     * moves the NLL long before it flips visible quality. */
    q36_token_score scores[20];
    double nll_sum = 0.0;
    int top20_hits = 0;
    bool first_token_match = false;
    for (int i = 0; i < vc->nsteps; i++) {
        const test_vec_step *step = &vc->steps[i];
        int n_vocab = 0;
        const float *logits = q36_session_logits(session, &n_vocab);
        TEST_ASSERT(logits != NULL && n_vocab > 0);
        if (!logits || n_vocab <= 0) break;
        int ref_id = test_find_token_by_bytes(engine, n_vocab, step->selected, step->selected_len);
        TEST_ASSERT(ref_id >= 0);
        if (ref_id < 0) break;
        float mx = logits[0];
        for (int j = 1; j < n_vocab; j++) if (logits[j] > mx) mx = logits[j];
        TEST_ASSERT(isfinite(mx));
        double sum = 0.0;
        for (int j = 0; j < n_vocab; j++) sum += exp((double)logits[j] - mx);
        double lp = (double)logits[ref_id] - mx - log(sum);
        nll_sum += -lp;

        if (i == 0) {
            first_token_match = test_token_bytes_equal(engine, q36_session_argmax(session),
                                                       step->selected, step->selected_len);
        }
        int nscore = q36_session_top_logprobs(session, scores, 20);
        for (int j = 0; j < nscore; j++) {
            if (scores[j].id == ref_id) { top20_hits++; break; }
        }

        if (i + 1 < vc->nsteps) {
            TEST_ASSERT(q36_session_eval(session, ref_id, err, sizeof(err)) == 0);
        }
    }

    double avg_nll = vc->nsteps > 0 ? nll_sum / vc->nsteps : 0.0;
    fprintf(stderr,
            "q36-test: vector %s avg_nll=%.3f ref_in_top20=%d/%d first_token_match=%d\n",
            vc->id, avg_nll, top20_hits, vc->nsteps, first_token_match ? 1 : 0);
    /* Gates hold for the tracked default model against the Q8_0 reference
     * with ~2x headroom over measured values; see TODO.md for the
     * calibration run. */
    TEST_ASSERT(avg_nll <= TEST_VEC_AVG_NLL_GATE);
    TEST_ASSERT(top20_hits >= vc->nsteps - TEST_VEC_TOP20_MISS_GATE);

    q36_session_free(session);
    q36_tokens_free(&prompt);
}

static void test_logprob_vectors_one(const char *test_name,
                                     const char *env_name,
                                     const char *default_path,
                                     bool hf_template) {
    if (!test_model_available(test_model_path())) {
        test_skip(test_name, "model file not found");
        return;
    }
    const char *path = getenv(env_name);
    bool legacy_template = false;
    if (!path || !path[0]) path = default_path;
    TEST_ASSERT(test_validate_reference_vector_fixture(path, &legacy_template));
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return;

    q36_engine *engine = test_open_runtime_engine(test_model_path(), 0);
    if (!engine) {
        fclose(fp);
        return;
    }
    if (!test_require_session_backend(test_name, engine, 32768)) {
        fclose(fp);
        q36_engine_close(engine);
        return;
    }

    test_vec_case vc;
    while (test_read_vector_case(fp, &vc)) {
        if (!test_fill_vector_case(fp, &vc)) break;
        if (!test_vector_case_selected(vc.id)) continue;
        fprintf(stderr, "q36-test: %s %s\n", test_name, vc.id);
        test_logprob_vector_case(engine, &vc, hf_template, legacy_template);
    }
    fclose(fp);
    q36_engine_close(engine);
}

static void test_logprob_vectors(void) {
    test_logprob_vectors_one("logprob-vectors",
                             "Q36_TEST_VECTOR_FILE",
                             "tests/test-vectors/qwen3.6-35b-a3b/llama.vec",
                             false);
}

#ifdef Q36_WITH_LLAMA
typedef struct {
    const char *id;
    const char *prompt_path;
    int steps;
    int ctx;
} test_llama_case;
#endif

typedef struct {
    const char *id;
    const char *prompt_path;
    const char *prompt_text;
    q36_think_mode think_mode;
    bool hf_template;
    int steps;
    int ctx;
} test_backend_parity_case;

#ifdef Q36_WITH_LLAMA
typedef enum {
    TEST_LLAMA_SEQ,
    TEST_LLAMA_BATCH,
} test_llama_mode;
#endif

typedef struct {
    int ref_top1;
    int cand_top1;
    int overlap;
    int top5_overlap;
    int top15_overlap;
    int top20_overlap;
    int top64_overlap;
    int nonfinite;
    float rms;
    float max_abs;
    float top15_max_abs;
    float top20_max_abs;
    bool same_top1;
} test_logit_comparison;

typedef struct {
    const char *id;
    int last;
} test_llama_progress;

#ifdef Q36_WITH_LLAMA
static void test_llama_log_silent(enum ggml_log_level level, const char *text, void *user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}
#endif

static void test_llama_progress_cb(void *ud, const char *event, int current, int total) {
    test_llama_progress *p = (test_llama_progress *)ud;
    if (!p || !event || strcmp(event, "prefill_chunk")) return;
    if (current != total && current - p->last < 256) return;
    p->last = current;
    fprintf(stderr, "q36-test: llama-seq %s q36 prefill %d/%d\n", p->id, current, total);
}

static int test_reference_threads(void) {
    const char *env = getenv("Q36_TEST_THREADS");
    if (env && env[0]) {
        int v = atoi(env);
        if (v > 0) return v;
    }
    return 12;
}

#ifdef Q36_WITH_LLAMA
static bool test_llama_tokenize_exact(const struct llama_vocab *vocab, const char *text,
                                      llama_token **out, int *len) {
    int need;
    if (!vocab || !text || !out || !len) return false;
    need = llama_tokenize(vocab, text, (int)strlen(text), NULL, 0, false, true);
    if (need >= 0) return false;
    *len = -need;
    *out = test_xcalloc((size_t)*len, sizeof(**out));
    need = llama_tokenize(vocab, text, (int)strlen(text), *out, *len, false, true);
    if (need < 0) {
        free(*out);
        *out = NULL;
        *len = 0;
        return false;
    }
    *len = need;
    return true;
}

static void test_llama_batch_set_one(struct llama_batch *batch, llama_token token, int pos, bool logits) {
    static llama_seq_id seq0 = 0;
    batch->n_tokens = 1;
    batch->token[0] = token;
    batch->pos[0] = (llama_pos)pos;
    batch->n_seq_id[0] = 1;
    batch->seq_id[0][0] = seq0;
    batch->logits[0] = logits ? 1 : 0;
}

static void test_llama_batch_set_prompt(struct llama_batch *batch, const llama_token *tokens, int len) {
    static llama_seq_id seq0 = 0;
    batch->n_tokens = len;
    for (int i = 0; i < len; i++) {
        batch->token[i] = tokens[i];
        batch->pos[i] = (llama_pos)i;
        batch->n_seq_id[i] = 1;
        batch->seq_id[i][0] = seq0;
        batch->logits[i] = (i == len - 1) ? 1 : 0;
    }
}
#endif

static int test_topk_overlap_count(const float *a, const float *b, int n, int k) {
    int ida[64], idb[64];
    float va[64], vb[64];
    int overlap = 0;
    TEST_ASSERT(k > 0 && k <= 64);
    parity_topk(a, n, k, ida, va);
    parity_topk(b, n, k, idb, vb);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            if (ida[i] >= 0 && ida[i] == idb[j]) {
                overlap++;
                break;
            }
        }
    }
    return overlap;
}

static float test_topk_ref_max_abs(const float *ref, const float *cand, int n, int k) {
    int ref_ids[64];
    float ref_vals[64];
    float max_abs = 0.0f;
    TEST_ASSERT(k > 0 && k <= 64);
    parity_topk(ref, n, k, ref_ids, ref_vals);
    for (int i = 0; i < k; i++) {
        int id = ref_ids[i];
        float d;
        if (id < 0) continue;
        if (!isfinite(ref[id]) || !isfinite(cand[id])) continue;
        d = fabsf(cand[id] - ref[id]);
        if (d > max_abs) max_abs = d;
    }
    return max_abs;
}

static test_logit_comparison test_compare_logits(const float *cand, const float *ref, int n_vocab) {
    double sumsq = 0.0;
    float max_abs = 0.0f;
    int nonfinite = 0;
    for (int i = 0; i < n_vocab; i++) {
        float d;
        if (!isfinite(ref[i]) || !isfinite(cand[i])) {
            nonfinite++;
            continue;
        }
        d = cand[i] - ref[i];
        if (fabsf(d) > max_abs) max_abs = fabsf(d);
        sumsq += (double)d * (double)d;
    }
    {
        int ref_top5[5], cand_top5[5];
        int ref_top15[15], cand_top15[15];
        int ref_top20[20], cand_top20[20];
        int ref_top64[64], cand_top64[64];
        float ref_vals5[5], cand_vals5[5];
        float ref_vals15[15], cand_vals15[15];
        float ref_vals20[20], cand_vals20[20];
        float ref_vals64[64], cand_vals64[64];
        test_logit_comparison result;
        parity_topk(ref, n_vocab, 5, ref_top5, ref_vals5);
        parity_topk(cand, n_vocab, 5, cand_top5, cand_vals5);
        parity_topk(ref, n_vocab, 15, ref_top15, ref_vals15);
        parity_topk(cand, n_vocab, 15, cand_top15, cand_vals15);
        parity_topk(ref, n_vocab, 20, ref_top20, ref_vals20);
        parity_topk(cand, n_vocab, 20, cand_top20, cand_vals20);
        parity_topk(ref, n_vocab, 64, ref_top64, ref_vals64);
        parity_topk(cand, n_vocab, 64, cand_top64, cand_vals64);
        memset(&result, 0, sizeof(result));
        result.ref_top1 = ref_top20[0];
        result.cand_top1 = cand_top20[0];
        result.top5_overlap = test_topk_overlap_count(ref, cand, n_vocab, 5);
        result.top15_overlap = test_topk_overlap_count(ref, cand, n_vocab, 15);
        result.top20_overlap = test_topk_overlap_count(ref, cand, n_vocab, 20);
        result.top64_overlap = test_topk_overlap_count(ref, cand, n_vocab, 64);
        result.overlap = result.top20_overlap;
        result.nonfinite = nonfinite;
        result.rms = (float)sqrt(sumsq / (double)n_vocab);
        result.max_abs = max_abs;
        result.top15_max_abs = test_topk_ref_max_abs(ref, cand, n_vocab, 15);
        result.top20_max_abs = test_topk_ref_max_abs(ref, cand, n_vocab, 20);
        result.same_top1 = result.ref_top1 >= 0 && result.ref_top1 == result.cand_top1;
        return result;
    }
}

static void test_logit_comparison_print(const char *name, const char *case_id, int step,
                                const test_logit_comparison *result) {
    fprintf(stderr,
            "q36-test: %s %s step %d top1 ref=%d cand=%d top5_overlap=%d/5 "
            "top15_overlap=%d/15 top20_overlap=%d/20 top64_overlap=%d/64 rms=%g max_abs=%g top15_max_abs=%g top20_max_abs=%g\n",
            name, case_id, step,
            result->ref_top1, result->cand_top1,
            result->top5_overlap,
            result->top15_overlap,
            result->top20_overlap,
            result->top64_overlap,
            result->rms,
            result->max_abs,
            result->top15_max_abs,
            result->top20_max_abs);
}

static void test_logit_comparison_assert_strict(const char *name, const char *case_id, int step,
                                        const test_logit_comparison *result) {
    test_logit_comparison_print(name, case_id, step, result);
    TEST_ASSERT(result->same_top1);
    TEST_ASSERT(result->top5_overlap >= 4);
    TEST_ASSERT(result->top20_overlap >= 15);
    TEST_ASSERT(result->top64_overlap >= 40);
    TEST_ASSERT(result->top20_max_abs <= 8.0f);
}

static void test_logit_comparison_assert_loose(const char *name, const char *case_id, int step,
                                       const test_logit_comparison *result) {
    test_logit_comparison_print(name, case_id, step, result);
    TEST_ASSERT(result->top5_overlap >= 2);
    TEST_ASSERT(result->overlap >= 10);
    TEST_ASSERT(result->rms <= 4.0f);
    TEST_ASSERT(result->top20_max_abs <= 12.0f);
}

#ifdef Q36_WITH_LLAMA
static void test_llama_parity_case(q36_engine *engine,
                                 struct llama_model *llama_model,
                                 const test_llama_case *tc,
                                 test_llama_mode llama_mode,
                                 const char *test_name,
                                 bool strict_gate) {
    char *prompt_text = NULL;
    char *rendered = NULL;
    q36_tokens prompt = {0};
    llama_token *llama_tokens = NULL;
    int llama_len = 0;
    q36_session *session = NULL;
    struct llama_context *llama_ctx = NULL;
    struct llama_batch batch = {0};
    char err[160] = {0};
    const struct llama_vocab *vocab;
    test_llama_progress progress = {0};
    const int guard_threads = test_reference_threads();

    TEST_ASSERT(engine != NULL);
    TEST_ASSERT(llama_model != NULL);
    TEST_ASSERT(tc != NULL);
    prompt_text = test_read_file(tc->prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) goto done;
    rendered = test_render_hf_chat_prompt("", prompt_text, Q36_THINK_NONE);
    TEST_ASSERT(rendered != NULL);
    if (!rendered) goto done;
    q36_encode_chat_prompt(engine, "", prompt_text, Q36_THINK_NONE, &prompt);
    TEST_ASSERT(prompt.len > 0);
    vocab = llama_model_get_vocab(llama_model);
    TEST_ASSERT(test_llama_tokenize_exact(vocab, rendered, &llama_tokens, &llama_len));
    TEST_ASSERT(prompt.len == llama_len);
    for (int i = 0; i < prompt.len && i < llama_len; i++) {
        if (prompt.v[i] == (int)llama_tokens[i]) continue;
        fprintf(stderr, "q36-test: %s %s prompt token mismatch at %d q36=%d llama=%d\n",
                test_name, tc->id, i, prompt.v[i], (int)llama_tokens[i]);
        TEST_ASSERT(false);
        break;
    }
    TEST_ASSERT(q36_session_create(&session, engine, tc->ctx) == 0);
    TEST_ASSERT(session != NULL);
    if (!session) goto done;
    if (prompt.len >= 512) {
        progress.id = tc->id;
        q36_session_set_progress(session, test_llama_progress_cb, &progress);
    }
    TEST_ASSERT(q36_session_sync(session, &prompt, err, sizeof(err)) == 0);

    {
        struct llama_context_params cp = llama_context_default_params();
        cp.n_ctx = (uint32_t)tc->ctx;
        cp.n_batch = llama_mode == TEST_LLAMA_SEQ ? 1u : (uint32_t)tc->ctx;
        cp.n_ubatch = llama_mode == TEST_LLAMA_SEQ ? 1u : (uint32_t)(tc->ctx < 2048 ? tc->ctx : 2048);
        cp.n_threads = guard_threads;
        cp.n_threads_batch = guard_threads;
        cp.no_perf = true;
        cp.offload_kqv = false;
        cp.op_offload = false;
        cp.flash_attn_type = llama_mode == TEST_LLAMA_SEQ ?
            LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_ENABLED;
        llama_ctx = llama_init_from_model(llama_model, cp);
        TEST_ASSERT(llama_ctx != NULL);
        if (!llama_ctx) goto done;
        batch = llama_batch_init(llama_mode == TEST_LLAMA_SEQ ? 1 : tc->ctx, 0, 1);
        if (llama_mode == TEST_LLAMA_SEQ) {
            for (int i = 0; i < llama_len; i++) {
                test_llama_batch_set_one(&batch, llama_tokens[i], i, i == llama_len - 1);
                TEST_ASSERT(llama_decode(llama_ctx, batch) == 0);
                if (llama_len >= 512 && (i + 1 == llama_len || ((i + 1) % 256) == 0)) {
                    fprintf(stderr, "q36-test: %s %s llama prefill %d/%d\n", test_name, tc->id, i + 1, llama_len);
                }
            }
        } else {
            test_llama_batch_set_prompt(&batch, llama_tokens, llama_len);
            TEST_ASSERT(llama_decode(llama_ctx, batch) == 0);
        }
    }

    for (int step = 0; step < tc->steps; step++) {
        const float *q36_logits = q36_session_logits(session, NULL);
        const float *llama_logits = llama_get_logits_ith(llama_ctx, llama_mode == TEST_LLAMA_SEQ ? 0 : batch.n_tokens - 1);
        test_logit_comparison result;
        TEST_ASSERT(q36_logits != NULL);
        TEST_ASSERT(llama_logits != NULL);
        if (!q36_logits || !llama_logits) break;
        result = test_compare_logits(q36_logits, llama_logits, q36_qwen35_n_vocab());
        if (strict_gate) test_logit_comparison_assert_strict(test_name, tc->id, step, &result);
        else test_logit_comparison_assert_loose(test_name, tc->id, step, &result);
        if (step + 1 < tc->steps) {
            int token = q36_session_argmax(session);
            TEST_ASSERT(token >= 0);
            TEST_ASSERT(q36_session_eval(session, token, err, sizeof(err)) == 0);
            test_llama_batch_set_one(&batch, (llama_token)token, prompt.len + step, true);
            TEST_ASSERT(llama_decode(llama_ctx, batch) == 0);
        }
    }

done:
    if (batch.token) llama_batch_free(batch);
    if (llama_ctx) llama_free(llama_ctx);
    q36_session_free(session);
    q36_tokens_free(&prompt);
    free(llama_tokens);
    free(rendered);
    free(prompt_text);
}

static void test_llama_quant_dots(q36_engine *engine) {
    static const uint32_t types[] = {8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 30, 42};
    ggml_cpu_init();
    for (size_t ti = 0; ti < sizeof(types) / sizeof(types[0]); ti++) {
        uint32_t type = types[ti], n;
        char name[256];
        uint64_t row_bytes;
        if (q36_engine_debug_first_tensor_of_type(engine, type, name, sizeof(name), &n) != 0) continue;
        TEST_ASSERT(q36_engine_debug_tensor_row_packed(engine, name, 0, NULL, 0,
                                                      &row_bytes, NULL, NULL) == 0);
        enum ggml_type activation_type = ggml_get_type_traits_cpu(type)->vec_dot_type;
        TEST_ASSERT(activation_type == GGML_TYPE_Q8_K || activation_type == GGML_TYPE_Q8_0 ||
                    activation_type == GGML_TYPE_BF16);
        size_t qbytes = ggml_row_size(activation_type, n);
        float *input = test_xcalloc(n, sizeof(float));
        void *weights = test_xcalloc(1, row_bytes);
        void *activation = test_xcalloc(1, qbytes);
        void *reference = test_xcalloc(1, qbytes);
        for (uint32_t i = 0; i < n; i++) input[i] = sinf(i * 0.017f) * cosf(i * 0.03f);
        if (activation_type == GGML_TYPE_Q8_0) q36_quant_q8_0_activation(input,activation,n);
        else if (activation_type == GGML_TYPE_Q8_K) q36_quant_q8_k(input,activation,n);
        ggml_get_type_traits_cpu(activation_type)->from_float(input, reference, n);
        if (activation_type != GGML_TYPE_BF16) TEST_ASSERT(memcmp(activation, reference, qbytes) == 0);
        for (uint32_t row = 0; row < 8; row++) {
            float got, want = 0.0f;
            TEST_ASSERT(q36_engine_debug_tensor_row_packed(engine, name, row, weights,
                                                          row_bytes, NULL, NULL, NULL) == 0);
            if (activation_type == GGML_TYPE_BF16) got = q36_quant_dot_bf16(weights,input,n);
            else TEST_ASSERT(activation_type == GGML_TYPE_Q8_0 ?
                q36_quant_dot_q8_0_type(type,weights,activation,n,&got) :
                q36_quant_dot_q8_k(type, weights, activation, n, &got));
            ggml_get_type_traits_cpu(type)->vec_dot(n, &want, 0, weights, 0, reference, 0, 1);
            TEST_ASSERT(isfinite(got) && fabsf(got - want) <= 2.0e-5f + 1.0e-5f * fabsf(want));
        }
        free(input); free(weights); free(activation); free(reference);
    }
}

static void test_llama_quant_kernels(void) {
    if (!test_model_available(test_model_path())) {
        test_skip("llama-quant-dots", "model file not found");
        return;
    }
    q36_engine *engine = test_open_runtime_engine(test_model_path(), test_reference_threads());
    TEST_ASSERT(engine != NULL);
    if (!engine) return;
    test_llama_quant_dots(engine);
    q36_engine_close(engine);
}

static void test_llama_parity_seq(void) {
    static const test_llama_case cases[] = {
        {"short_italian_fact", "tests/test-vectors/qwen3.6-35b-a3b/prompts/short_italian_fact.txt", 4, 4096},
        {"short_code_completion", "tests/test-vectors/qwen3.6-35b-a3b/prompts/short_code_completion.txt", 2, 4096},
        {"short_reasoning_plain", "tests/test-vectors/qwen3.6-35b-a3b/prompts/short_reasoning_plain.txt", 3, 4096},
    };
    q36_engine *engine = NULL;
    struct llama_model *llama_model = NULL;
    struct llama_model_params mp;
    if (!test_model_available(test_model_path())) {
        test_skip("llama-parity-seq", "model file not found");
        return;
    }
    engine = test_open_runtime_engine(test_model_path(), test_reference_threads());
    if (!engine) return;
    if (!test_require_session_backend("llama-parity-seq", engine, 4096)) {
        q36_engine_close(engine);
        return;
    }
    llama_log_set(test_llama_log_silent, NULL);
    llama_backend_init();
    mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    mp.use_extra_bufts = false;
    llama_model = llama_model_load_from_file(test_model_path(), mp);
    TEST_ASSERT(llama_model != NULL);
    if (!llama_model) goto done;
    test_llama_quant_dots(engine);
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        if (!test_vector_case_selected(cases[i].id)) continue;
        fprintf(stderr, "q36-test: llama-parity-seq %s\n", cases[i].id);
        test_llama_parity_case(engine, llama_model, &cases[i], TEST_LLAMA_SEQ,
                             "llama-seq", true);
    }

done:
    if (llama_model) llama_model_free(llama_model);
    llama_backend_free();
    q36_engine_close(engine);
}

static void test_llama_parity_seq_long(void) {
    static const test_llama_case cases[] = {
        {"long_memory_archive", "tests/test-vectors/qwen3.6-35b-a3b/prompts/long_memory_archive.txt", 4, 4096},
        {"long_code_audit", "tests/test-vectors/qwen3.6-35b-a3b/prompts/long_code_audit.txt", 4, 4096},
    };
    q36_engine *engine = NULL;
    struct llama_model *llama_model = NULL;
    struct llama_model_params mp;
    if (!test_model_available(test_model_path())) {
        test_skip("llama-parity-seq-long", "model file not found");
        return;
    }
    engine = test_open_runtime_engine(test_model_path(), test_reference_threads());
    if (!engine) return;
    if (!test_require_session_backend("llama-parity-seq-long", engine, 4096)) {
        q36_engine_close(engine);
        return;
    }
    llama_log_set(test_llama_log_silent, NULL);
    llama_backend_init();
    mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    mp.use_extra_bufts = false;
    llama_model = llama_model_load_from_file(test_model_path(), mp);
    TEST_ASSERT(llama_model != NULL);
    if (!llama_model) goto done;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        if (!test_vector_case_selected(cases[i].id)) continue;
        fprintf(stderr, "q36-test: llama-parity-seq-long %s\n", cases[i].id);
        test_llama_parity_case(engine, llama_model, &cases[i], TEST_LLAMA_SEQ,
                             "llama-seq-long", true);
    }

done:
    if (llama_model) llama_model_free(llama_model);
    llama_backend_free();
    q36_engine_close(engine);
}

static void test_llama_parity_batch_loose(void) {
    static const test_llama_case cases[] = {
        {"short_italian_fact", "tests/test-vectors/qwen3.6-35b-a3b/prompts/short_italian_fact.txt", 4, 4096},
        {"short_code_completion", "tests/test-vectors/qwen3.6-35b-a3b/prompts/short_code_completion.txt", 2, 4096},
        {"short_reasoning_plain", "tests/test-vectors/qwen3.6-35b-a3b/prompts/short_reasoning_plain.txt", 3, 4096},
    };
    q36_engine *engine = NULL;
    struct llama_model *llama_model = NULL;
    struct llama_model_params mp;
    if (!test_model_available(test_model_path())) {
        test_skip("llama-parity-batch-loose", "model file not found");
        return;
    }
    engine = test_open_runtime_engine(test_model_path(), test_reference_threads());
    if (!engine) return;
    if (!test_require_session_backend("llama-parity-batch-loose", engine, 4096)) {
        q36_engine_close(engine);
        return;
    }
    llama_log_set(test_llama_log_silent, NULL);
    llama_backend_init();
    mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    mp.use_extra_bufts = false;
    llama_model = llama_model_load_from_file(test_model_path(), mp);
    TEST_ASSERT(llama_model != NULL);
    if (!llama_model) goto done;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        if (!test_vector_case_selected(cases[i].id)) continue;
        fprintf(stderr, "q36-test: llama-parity-batch-loose %s\n", cases[i].id);
        test_llama_parity_case(engine, llama_model, &cases[i], TEST_LLAMA_BATCH,
                             "llama-batch-loose", false);
    }

done:
    if (llama_model) llama_model_free(llama_model);
    llama_backend_free();
    q36_engine_close(engine);
}
#endif

static void test_vector_fixtures(void) {
    const char *path = getenv("Q36_TEST_LLAMA_VECTOR_FILE");
    if (!path || !path[0]) path = "tests/test-vectors/qwen3.6-35b-a3b/llama.vec";
    TEST_ASSERT(test_validate_reference_vector_fixture(path, NULL));
}

static const char *test_tool_call_request_json(void) {
    return
        "{"
        "\"model\":\"qwen3.6-35b-a3b\"," 
        "\"messages\":[{\"role\":\"user\",\"content\":\"List the files in the current directory. Use the provided tool; do not answer in prose.\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{"
            "\"name\":\"list_files\","
            "\"description\":\"List files in a directory.\","
            "\"parameters\":{\"type\":\"object\",\"properties\":{"
                "\"path\":{\"type\":\"string\",\"description\":\"Directory path to list.\"}"
            "},\"required\":[\"path\"]}"
        "}}],"
        "\"tool_choice\":\"auto\","
        "\"think\":false,"
        "\"temperature\":0,"
        "\"max_tokens\":256,"
        "\"stream\":false"
        "}";
}

static void test_tool_call_quality_one(void) {
    q36_engine *engine = test_get_engine();
    if (!engine) return;

    request r;
    char err[160];
    TEST_ASSERT(parse_chat_request(engine, NULL, test_tool_call_request_json(),
                                   512, 4096, &r, err, sizeof(err)));

    q36_session *session = NULL;
    TEST_ASSERT(q36_session_create(&session, engine, 4096) == 0);
    if (!session) {
        request_free(&r);
        return;
    }
    TEST_ASSERT(q36_session_sync(session, &r.prompt, err, sizeof(err)) == 0);

    buf text = {0};
    uint64_t rng = 123;
    bool decode_ok = true;
    bool saw_tool_start = false;
    bool saw_tool_end = false;
    qwen_tool_decode_tracker tracker;
    qwen_tool_decode_tracker_init(&tracker);
    for (int i = 0; i < r.max_tokens; i++) {
        int token = q36_session_sample(session, r.temperature, r.top_k,
                                       r.top_p, r.min_p, &rng);
        size_t piece_len = 0;
        char *piece = q36_token_text(engine, token, &piece_len);
        buf_append(&text, piece, piece_len);
        free(piece);
        qwen_tool_decode_tracker_update(&tracker, text.ptr, text.len);
        observe_tool_markers(&tracker, text.ptr, &saw_tool_start, &saw_tool_end, NULL);
        if (saw_tool_end) break;
        if (q36_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    bool parsed = parse_generated_message(text.ptr ? text.ptr : "",
                                          &content, &reasoning, &calls);
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(parsed);
    TEST_ASSERT(calls.len > 0);
    TEST_ASSERT(calls.len > 0 && !strcmp(calls.v[0].name, "list_files"));

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    buf_free(&text);
    q36_session_free(session);
    request_free(&r);
}

static void test_tool_call_quality(void) {
    if (!test_model_available(test_model_path())) {
        test_skip("tool-call-quality", "model file not found");
        return;
    }
    q36_engine *engine = test_get_engine();
    if (!engine) return;
    if (!test_require_session_backend("tool-call-quality", engine, 4096)) return;
    fprintf(stderr, "q36-test: tool-call quality\n");
    test_tool_call_quality_one();
    test_close_engine();
}

static void test_thinking_generation(void) {
    if (!test_model_available(test_model_path())) {
        test_skip("thinking-generation", "model file not found");
        return;
    }
    q36_engine *engine = test_get_engine();
    if (!engine) return;
    if (!test_require_session_backend("thinking-generation", engine, 4096)) return;

    const char *thk_prompt = "Reply with exactly two words: hello world.";
    q36_tokens prompt_none = {0};
    q36_encode_chat_prompt(engine, "", thk_prompt, Q36_THINK_NONE, &prompt_none);

    q36_tokens prompt_high = {0};
    q36_encode_chat_prompt(engine, "", thk_prompt, Q36_THINK_HIGH, &prompt_high);

    char *first_text = q36_token_text(engine, prompt_none.v[0], NULL);
    TEST_ASSERT(first_text && !strcmp(first_text, "<|im_start|>"));
    free(first_text);

    fprintf(stderr, "q36-test: think=HIGH prompt=%d tokens vs think=NONE prompt=%d tokens\n",
            prompt_high.len, prompt_none.len);

    size_t none_len = 0, high_len = 0;
    char *none_text = render_tokens_text(engine, &prompt_none, &none_len);
    char *high_text = render_tokens_text(engine, &prompt_high, &high_len);
    const char *none_suffix = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    const char *high_suffix = "<|im_start|>assistant\n<think>\n";
    /* Qwen puts newline tokens after the marker; validate the complete suffix. */
    TEST_ASSERT(none_text && none_len >= strlen(none_suffix) &&
                !strcmp(none_text + none_len - strlen(none_suffix), none_suffix));
    TEST_ASSERT(high_text && high_len >= strlen(high_suffix) &&
                !strcmp(high_text + high_len - strlen(high_suffix), high_suffix));

    free(none_text);
    free(high_text);

    q36_session *session = NULL;
    TEST_ASSERT(q36_session_create(&session, engine, 4096) == 0);
    if (!session) { q36_tokens_free(&prompt_none); q36_tokens_free(&prompt_high); return; }

    char err[160];
    TEST_ASSERT(q36_session_sync(session, &prompt_high, err, sizeof(err)) == 0);

    buf out = {0};
    uint64_t rng = 12345;
    bool decode_ok = true;
    int generated = 0;
    for (; generated < 128; generated++) {
        int token = q36_session_sample(session, 0.0f, 1, 1.0f, 0.0f, &rng);
        if (token == q36_token_eos(engine)) break;
        size_t piece_len = 0;
        char *piece = q36_token_text(engine, token, &piece_len);
        buf_append(&out, piece, piece_len);
        free(piece);
        if (q36_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
        if (strstr(out.ptr ? out.ptr : "", "hello") != NULL) break;
    }

    const char *text = out.ptr ? out.ptr : "";
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(generated > 0);
    fprintf(stderr, "q36-test: thinking-mode generated %d tokens: %.*s\n",
            generated, (int)(strlen(text) > 80 ? 80 : strlen(text)), text);

    buf_free(&out);
    q36_session_free(session);
    q36_tokens_free(&prompt_none);
    q36_tokens_free(&prompt_high);
}

typedef struct {
    int calls;
    int first_current;
    int last_current;
} test_progress_probe;

static void test_progress_probe_cb(void *ud, const char *event, int current, int total) {
    test_progress_probe *p = (test_progress_probe *)ud;
    (void)total;
    if (!p || strcmp(event, "prefill_chunk")) return;
    if (p->calls == 0) p->first_current = current;
    p->last_current = current;
    p->calls++;
}

static void test_session_sync_prefix_resume_cpu(void) {
    if (!test_model_available(test_model_path())) {
        test_skip("session-sync-prefix-cpu", "model file not found");
        return;
    }

    q36_engine *engine = test_open_runtime_engine(test_model_path(), 0);
    TEST_ASSERT(engine != NULL);
    if (!engine) return;

    q36_tokens full = {0};
    q36_encode_chat_prompt(engine,
                           "",
                           "Reply with exactly: sync prefix cpu check.",
                           Q36_THINK_NONE,
                           &full);
    TEST_ASSERT(full.len > 8);

    q36_tokens prefix = {0};
    for (int i = 0; i < full.len - 4; i++) q36_tokens_push(&prefix, full.v[i]);

    q36_session *warm = NULL;
    q36_session *cold = NULL;
    TEST_ASSERT(q36_session_create(&warm, engine, 4096) == 0);
    TEST_ASSERT(q36_session_create(&cold, engine, 4096) == 0);
    if (!warm || !cold) goto cleanup;

    char err[160];
    TEST_ASSERT(q36_session_sync(warm, &prefix, err, sizeof(err)) == 0);

    test_progress_probe probe = {0};
    q36_session_set_progress(warm, test_progress_probe_cb, &probe);
    TEST_ASSERT(q36_session_sync(warm, &full, err, sizeof(err)) == 0);
    q36_session_set_progress(warm, NULL, NULL);

    TEST_ASSERT(probe.calls == full.len - prefix.len);
    TEST_ASSERT(probe.first_current == prefix.len + 1);
    TEST_ASSERT(probe.last_current == full.len);

    TEST_ASSERT(q36_session_sync(cold, &full, err, sizeof(err)) == 0);
    TEST_ASSERT(q36_session_argmax(warm) == q36_session_argmax(cold));

cleanup:
    q36_session_free(cold);
    q36_session_free(warm);
    q36_tokens_free(&prefix);
    q36_tokens_free(&full);
    q36_engine_close(engine);
}

static void test_sampling_controls(void) {
    if (!test_model_available(test_model_path())) {
        test_skip("sampling-controls", "model file not found");
        return;
    }
    /* Replace host logits in a CPU session. Avoid worker threads here:
     * the later server rejection tests fork. */
    q36_engine *engine = test_open_runtime_engine(test_model_path(), 1);
    TEST_ASSERT(engine != NULL);
    if (!engine) return;

    q36_session *session = NULL;
    TEST_ASSERT(q36_session_create(&session, engine, 16) == 0);
    if (!session) goto done;
    q36_tokens seed = {0};
    char err[160] = {0};
    q36_tokens_push(&seed, 17);
    int rc = q36_session_sync(session, &seed, err, sizeof(err));
    q36_tokens_free(&seed);
    TEST_ASSERT(rc == 0);
    if (rc) goto done;

    int n_vocab = 0;
    float *logits = (float *)q36_session_logits(session, &n_vocab);
    TEST_ASSERT(logits != NULL);
    TEST_ASSERT(n_vocab > 4);
    if (!logits || n_vocab <= 4) goto done;

    for (int i = 0; i < n_vocab; i++) logits[i] = Q36_NEG_INF;
    logits[0] = 0.0f;
    logits[1] = 0.0f;

    uint64_t rng = 1;
    bool saw0 = false;
    bool saw1 = false;
    for (int i = 0; i < 64; i++) {
        int token = q36_session_sample(session, 1.0f, 0, 1.0f, 0.0f, &rng);
        if (token == 0) saw0 = true;
        if (token == 1) saw1 = true;
    }
    TEST_ASSERT(saw0);
    TEST_ASSERT(saw1);

    rng = 7;
    TEST_ASSERT(q36_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &rng) == 0);
    TEST_ASSERT(q36_session_sample(session, 1.0f, 1, 1.0f, 0.0f, &rng) == 0);

    for (int i = 0; i < n_vocab; i++) logits[i] = Q36_NEG_INF;
    logits[0] = 0.0f;
    logits[1] = -1.0f;
    logits[2] = -2.0f;

    rng = 11;
    TEST_ASSERT(q36_session_sample(session, 1.0f, 0, 0.5f, 0.0f, &rng) == 0);

    for (int i = 0; i < n_vocab; i++) logits[i] = Q36_NEG_INF;
    logits[0] = 0.0f;
    logits[1] = -0.1f;
    logits[2] = -2.0f;

    rng = 19;
    TEST_ASSERT(q36_session_sample(session, 1.0f, 0, 1.0f, 0.95f, &rng) == 0);

done:
    q36_session_free(session);
    q36_engine_close(engine);
}

static void test_kv_cache_save_restore(void) {
    if (!test_model_available(test_model_path())) {
        test_skip("kv-cache-save-restore", "model file not found");
        return;
    }
    q36_engine *engine = test_get_engine();
    if (!engine) return;
    if (!test_require_session_backend("kv-cache-save-restore", engine, 4096)) return;

    const char *prompt_text = "Reply with exactly: first second";
    q36_tokens prompt = {0};
    q36_encode_chat_prompt(engine, "", prompt_text, Q36_THINK_NONE, &prompt);

    q36_session *session = NULL;
    TEST_ASSERT(q36_session_create(&session, engine, 4096) == 0);
    if (!session) { q36_tokens_free(&prompt); return; }

    char err[160];
    TEST_ASSERT(q36_session_sync(session, &prompt, err, sizeof(err)) == 0);

    int argmax_before = q36_session_argmax(session);
    TEST_ASSERT(argmax_before >= 0);

    uint64_t payload_bytes = q36_session_payload_bytes(session);
    if (payload_bytes == 0) {
        test_skip("kv-cache-save-restore", "backend does not support payload serialization");
        q36_session_free(session);
        q36_tokens_free(&prompt);
        return;
    }

    q36_session_snapshot snap = {0};
    if (q36_session_save_snapshot(session, &snap, err, sizeof(err)) != 0) {
        test_skip("kv-cache-save-restore", err);
        q36_session_free(session);
        q36_tokens_free(&prompt);
        return;
    }
    TEST_ASSERT(snap.len > 0);

    q36_session_free(session);
    session = NULL;

    TEST_ASSERT(q36_session_create(&session, engine, 4096) == 0);

    if (q36_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) {
        test_skip("kv-cache-save-restore", err);
        q36_session_snapshot_free(&snap);
        q36_session_free(session);
        q36_tokens_free(&prompt);
        return;
    }

    int argmax_after = q36_session_argmax(session);
    if (argmax_after != argmax_before) {
        fprintf(stderr, "q36-test: kv-cache argmax mismatch before=%d after=%d\n",
                argmax_before, argmax_after);
        TEST_ASSERT(false);
    }

    TEST_ASSERT(q36_session_sync(session, &prompt, err, sizeof(err)) == 0);
    int argmax_after_sync = q36_session_argmax(session);
    TEST_ASSERT(argmax_after_sync == argmax_before);

    FILE *oversized = tmpfile();
    TEST_ASSERT(oversized != NULL);
    if (oversized) {
        uint32_t header[3] = {0x51563336u, 1u, 4096u};
        int32_t zero = 0;
        TEST_ASSERT(fwrite(header, sizeof(header), 1, oversized) == 1);
        for (uint32_t i = 0; i < header[2]; i++)
            TEST_ASSERT(fwrite(&zero, sizeof(zero), 1, oversized) == 1);
        rewind(oversized);
        err[0] = '\0';
        TEST_ASSERT(q36_session_load_payload(
                        session, oversized,
                        sizeof(header) + (uint64_t)header[2] * sizeof(zero),
                        err, sizeof(err)) != 0);
        TEST_ASSERT(strstr(err, "token-only checkpoint does not fit") != NULL);
        fclose(oversized);
    }

    q36_session_snapshot_free(&snap);
    q36_session_free(session);
    q36_tokens_free(&prompt);
}

static bool test_session_batch_mode(q36_kv_cache_type type_k,
                                    q36_kv_cache_type type_v,
                                    bool ssd_streaming) {
    const char *old_tmpdir_env = getenv("TMPDIR");
    char *old_tmpdir = old_tmpdir_env ? strdup(old_tmpdir_env) : NULL;
    setenv("TMPDIR", ".", 1);
    q36_engine *engine = NULL;
    q36_engine_options opt = {
        .model_path = test_model_path(),
        .backend = Q36_BACKEND_VULKAN,
        .cache_type_k = type_k,
        .cache_type_v = type_v,
        .prefill_chunk = 8,
        .ssd_streaming = ssd_streaming,
        .ssd_streaming_cold = ssd_streaming,
        .ssd_streaming_cache_experts = ssd_streaming ? 8u : 0u,
    };
    if (q36_engine_open(&engine, &opt) != 0) {
        if (old_tmpdir) setenv("TMPDIR", old_tmpdir, 1);
        else unsetenv("TMPDIR");
        free(old_tmpdir);
        return false;
    }

    const int counts[] = {1, 2, 4, 8};
    int first_count = 0;
    int n_counts = ssd_streaming ? 2 : 4;
    const char *count_env = getenv("Q36_TEST_SESSION_COUNT");
    if (count_env && count_env[0]) {
        int requested = atoi(count_env);
        first_count = -1;
        for (int i = 0; i < 4; i++) {
            if (counts[i] == requested) first_count = i;
        }
        if (first_count < 0) {
            fprintf(stderr, "q36-test: Q36_TEST_SESSION_COUNT must be 1, 2, 4, or 8\n");
            q36_engine_close(engine);
            if (old_tmpdir) setenv("TMPDIR", old_tmpdir, 1);
            else unsetenv("TMPDIR");
            free(old_tmpdir);
            return false;
        }
        n_counts = first_count + 1;
    }
    bool ok = true;
    for (int ci = first_count; ci < n_counts && ok; ci++) {
        int count = counts[ci];
        q36_session *sessions[8] = {0};
        q36_tokens prompts[8] = {0};
        float *oracle[8] = {0};
        q36_decode_item items[8] = {0};
        char err[160] = {0};

        for (int i = 0; i < count; i++) {
            char prompt_text[256];
            if (i < 2) {
                snprintf(prompt_text, sizeof(prompt_text),
                         "Return one word for the color of a clear daytime sky.");
            } else {
                snprintf(prompt_text, sizeof(prompt_text),
                         "Request %d: list %d compact facts about number %d, then answer in one word.%s",
                         i, i + 1, 17 + i * 13,
                         i & 1 ? " Add a short explanation first." : "");
            }
            q36_encode_chat_prompt(engine, "", prompt_text, Q36_THINK_NONE,
                                   &prompts[i]);
            TEST_ASSERT(q36_session_create(&sessions[i], engine, 512) == 0);
            if (!sessions[i] || q36_session_sync(sessions[i], &prompts[i],
                                                  err, sizeof(err)) != 0) {
                ok = false;
                break;
            }
            items[i].session = sessions[i];
            items[i].token = q36_session_argmax(sessions[i]);
            TEST_ASSERT(items[i].token >= 0);
            if (q36_session_eval(sessions[i], items[i].token,
                                 err, sizeof(err)) != 0) {
                ok = false;
                break;
            }
            int n_vocab = 0;
            const float *logits = q36_session_logits(sessions[i], &n_vocab);
            oracle[i] = malloc((size_t)n_vocab * sizeof(float));
            TEST_ASSERT(oracle[i] != NULL);
            if (!oracle[i]) {
                ok = false;
                break;
            }
            memcpy(oracle[i], logits, (size_t)n_vocab * sizeof(float));
            TEST_ASSERT(q36_session_sync(sessions[i], &prompts[i],
                                          err, sizeof(err)) == 0);
        }

        if (ok) {
            setenv("Q36_VK_SESSION_BATCH_LOG", "1", 1);
            TEST_ASSERT(q36_sessions_eval_batch(items, count, err, sizeof(err)) == 0);
            for (int i = 0; i < count; i++) {
                int n_vocab = 0;
                const float *logits = q36_session_logits(sessions[i], &n_vocab);
                test_logit_comparison cmp = test_compare_logits(
                        oracle[i], logits, n_vocab);
                fprintf(stderr,
                        "q36-test: Vulkan session batch rows=%d item=%d max_abs=%g rms=%g\n",
                        count, i, cmp.max_abs, cmp.rms);
                TEST_ASSERT(cmp.nonfinite == 0);
                TEST_ASSERT(q36_session_argmax(sessions[i]) ==
                            test_sample_argmax_local(oracle[i], n_vocab));
                TEST_ASSERT(cmp.max_abs <= 0.25f);
#ifndef Q36_METAL
                if (q36_engine_is_qwen38(engine) && !ssd_streaming)
                    TEST_ASSERT(!memcmp(oracle[i], logits, (size_t)n_vocab * sizeof(float)));
#endif
            }
            if (count >= 2) {
                for (int step = 0; step < 6; step++) {
                    int n0 = 0, n1 = 0;
                    const float *l0 = q36_session_logits(sessions[0], &n0);
                    const float *l1 = q36_session_logits(sessions[1], &n1);
                    TEST_ASSERT(n0 == n1);
                    TEST_ASSERT(!memcmp(l0, l1, (size_t)n0 * sizeof(float)));
                    items[0].token = q36_session_argmax(sessions[0]);
                    items[1].token = q36_session_argmax(sessions[1]);
                    TEST_ASSERT(items[0].token == items[1].token);
                    TEST_ASSERT(q36_sessions_eval_batch(items, 2,
                                                        err, sizeof(err)) == 0);
                }
            }

            if (count == 2) {
                q36_session_payload_file after = {0};
                int saved_argmax = q36_session_argmax(sessions[0]);
                TEST_ASSERT(q36_session_stage_payload(sessions[0], &after,
                                                       err, sizeof(err)) == 0);
                FILE *restore = fopen(after.path, "rb");
                TEST_ASSERT(restore != NULL);
                TEST_ASSERT(restore && q36_session_load_payload(
                        sessions[0], restore, after.bytes,
                        err, sizeof(err)) == 0);
                if (restore) fclose(restore);
                TEST_ASSERT(q36_session_argmax(sessions[0]) == saved_argmax);
                q36_session_payload_file_free(&after);

                int pos0 = q36_session_pos(sessions[0]);
                int pos1 = q36_session_pos(sessions[1]);
                q36_decode_item bad[2] = {items[0], items[1]};
                bad[1].token = -1;
                TEST_ASSERT(q36_sessions_eval_batch(bad, 2, err, sizeof(err)) != 0);
                TEST_ASSERT(q36_session_pos(sessions[0]) == pos0);
                TEST_ASSERT(q36_session_pos(sessions[1]) == pos1);

                for (int i = 0; i < 2; i++) {
                    TEST_ASSERT(q36_session_sync(sessions[i], &prompts[i],
                                                  err, sizeof(err)) == 0);
                    items[i].token = q36_session_argmax(sessions[i]);
                }
                setenv("Q36_VK_SESSION_BATCH", "0", 1);
                TEST_ASSERT(q36_sessions_eval_batch(items, 2, err, sizeof(err)) == 0);
                unsetenv("Q36_VK_SESSION_BATCH");
                for (int i = 0; i < 2; i++) {
                    int n_vocab = 0;
                    const float *logits = q36_session_logits(sessions[i], &n_vocab);
                    TEST_ASSERT(!memcmp(oracle[i], logits,
                                        (size_t)n_vocab * sizeof(float)));
                }

                q36_tokens mixed_prefill = {0};
                const q36_tokens *live = q36_session_tokens(sessions[0]);
                for (int i = 0; live && i < live->len; i++)
                    q36_tokens_push(&mixed_prefill, live->v[i]);
                q36_tokens_push(&mixed_prefill,
                                q36_session_argmax(sessions[0]));
                q36_decode_item mixed_decode = {
                    .session = sessions[1],
                    .token = q36_session_argmax(sessions[1]),
                };
                int mixed_pos0 = q36_session_pos(sessions[0]);
                int mixed_pos1 = q36_session_pos(sessions[1]);
                TEST_ASSERT(q36_sessions_eval_batch_with_prefill(
                        &mixed_decode, 1, sessions[0], &mixed_prefill,
                        err, sizeof(err)) == 0);
                TEST_ASSERT(q36_session_pos(sessions[0]) == mixed_pos0 + 1);
                TEST_ASSERT(q36_session_pos(sessions[1]) == mixed_pos1 + 1);
                q36_tokens_free(&mixed_prefill);
            }
        }

        unsetenv("Q36_VK_SESSION_BATCH_LOG");
        unsetenv("Q36_VK_SESSION_BATCH");
        for (int i = 0; i < count; i++) {
            free(oracle[i]);
            q36_tokens_free(&prompts[i]);
            q36_session_free(sessions[i]);
        }
    }
    q36_engine_close(engine);
    if (old_tmpdir) setenv("TMPDIR", old_tmpdir, 1);
    else unsetenv("TMPDIR");
    free(old_tmpdir);
    return ok;
}

static void test_vulkan_session_batch(void) {
    if (!test_model_available(test_model_path())) {
        test_skip("vulkan-session-batch", "model file not found");
        return;
    }
    if (test_is_cpu_only_build()) {
        test_skip("vulkan-session-batch", "requires Vulkan backend build");
        return;
    }
    const char *mode = getenv("Q36_TEST_BATCH_KV");
    if (mode && !strcmp(mode, "f16")) {
        TEST_ASSERT(test_session_batch_mode(Q36_KV_CACHE_F16,
                                            Q36_KV_CACHE_F16, false));
    } else if (mode && !strcmp(mode, "ssd")) {
        TEST_ASSERT(test_session_batch_mode(Q36_KV_CACHE_F16,
                                            Q36_KV_CACHE_F16, true));
    } else {
        TEST_ASSERT(test_session_batch_mode(Q36_KV_CACHE_Q8_0,
                                            Q36_KV_CACHE_Q4_0, false));
        TEST_ASSERT(test_session_batch_mode(Q36_KV_CACHE_Q4_0,
                                            Q36_KV_CACHE_Q4_0, false));
    }
}

static void test_session_sync_resume_matches_cold_rebuild(void) {
    if (!test_model_available(test_model_path())) {
        test_skip("session-sync-resume", "model file not found");
        return;
    }
    if (test_is_cpu_only_build()) {
        test_skip("session-sync-resume", "requires Vulkan backend build");
        return;
    }

    q36_engine *engine = NULL;
    q36_engine_options opt = {
        .model_path = test_model_path(),
        .backend = Q36_BACKEND_VULKAN,
    };
    if (q36_engine_open(&engine, &opt) != 0) {
        test_skip("session-sync-resume", "Vulkan backend unavailable");
        return;
    }

    /* The prompt must be long enough that the cold rebuild prefills
     * through the batched GEMM kernels while the warm session finishes
     * with a decode-order extension.  Warm == cold is a strict
     * distribution gate rather than bit equality: the prefill GEMMs keep
     * a fixed per-output k order for any n_tok, but that order is not
     * the decode matvec's, so the two paths legitimately differ at the
     * ~1e-3 logit level. */
    char prompt_text[8192];
    prompt_text[0] = '\0';
    for (int i = 0; i < 40; i++) {
        strcat(prompt_text,
               "Day after day the expedition journal records wind, snowfall, "
               "ration counts, sled repairs, and the slow drift of the pack ice. ");
    }
    strcat(prompt_text, "\nSummarize the journal in one word.");

    q36_tokens full = {0};
    q36_encode_chat_prompt(engine, "", prompt_text, Q36_THINK_NONE, &full);
    TEST_ASSERT(full.len > 600);

    q36_tokens prefix = {0};
    for (int i = 0; i < full.len - 2; i++) q36_tokens_push(&prefix, full.v[i]);

    q36_session *warm = NULL;
    q36_session *cold = NULL;
    char err[160] = {0};
    TEST_ASSERT(q36_session_create(&warm, engine, 4096) == 0);
    TEST_ASSERT(q36_session_create(&cold, engine, 4096) == 0);
    if (!warm || !cold) {
        q36_session_free(warm);
        q36_session_free(cold);
        q36_tokens_free(&prefix);
        q36_tokens_free(&full);
        q36_engine_close(engine);
        return;
    }

    TEST_ASSERT(q36_session_sync(warm, &prefix, err, sizeof(err)) == 0);
    int before_rewrite = q36_session_argmax(warm);
    TEST_ASSERT(q36_session_rewrite_from_common(warm, &full, prefix.len - 1,
                                                err, sizeof(err)) == Q36_SESSION_REWRITE_REBUILD_NEEDED);
    TEST_ASSERT(q36_session_pos(warm) == prefix.len);
    TEST_ASSERT(q36_session_argmax(warm) == before_rewrite);
    TEST_ASSERT(q36_session_sync(warm, &full, err, sizeof(err)) == 0);
    TEST_ASSERT(q36_session_sync(cold, &full, err, sizeof(err)) == 0);

    int warm_argmax = q36_session_argmax(warm);
    int cold_argmax = q36_session_argmax(cold);
    TEST_ASSERT(warm_argmax == cold_argmax);

    /* Q36 strict sequential gate plus a much tighter same-engine
     * absolute bound (measured drift is ~1e-2 at the top-20). */
    int warm_vocab = 0;
    int cold_vocab = 0;
    const float *warm_logits = q36_session_logits(warm, &warm_vocab);
    const float *cold_logits = q36_session_logits(cold, &cold_vocab);
    TEST_ASSERT(warm_logits && cold_logits && warm_vocab > 0 && warm_vocab == cold_vocab);
    test_logit_comparison result = test_compare_logits(warm_logits, cold_logits, warm_vocab);
    TEST_ASSERT(result.nonfinite == 0);
    test_logit_comparison_assert_strict("session-sync-resume", "warm-vs-cold", 0, &result);
    TEST_ASSERT(result.top20_max_abs <= 1.0f);

    q36_session_free(cold);
    q36_session_free(warm);
    q36_tokens_free(&prefix);
    q36_tokens_free(&full);
    q36_engine_close(engine);
}

#ifndef Q36_NO_GPU
static char *test_backend_parity_prompt_text(const test_backend_parity_case *tc) {
    if (!tc) return NULL;
    if (tc->prompt_text) return test_strdup(tc->prompt_text);
    if (tc->prompt_path) return test_read_file(tc->prompt_path);
    return NULL;
}

typedef struct {
    q36_tokens prompt;
    float **logits;
    int *tokens;
    int vocab;
    int steps;
} test_backend_parity_capture;

static void test_backend_parity_capture_free(test_backend_parity_capture *c) {
    if (!c) return;
    for (int i = 0; i < c->steps; i++) free(c->logits[i]);
    free(c->logits);
    free(c->tokens);
    q36_tokens_free(&c->prompt);
    memset(c, 0, sizeof(*c));
}

static void test_backend_parity_capture_cpu(q36_engine *engine,
                                            const test_backend_parity_case *tc,
                                            test_backend_parity_capture *out) {
    char *prompt_text = NULL;
    q36_session *session = NULL;
    char err[160] = {0};

    TEST_ASSERT(engine != NULL);
    TEST_ASSERT(tc != NULL);
    TEST_ASSERT(out != NULL);
    if (!engine || !tc || !out) return;

    prompt_text = test_backend_parity_prompt_text(tc);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) goto done;

    test_encode_prompt_mode(engine, "", prompt_text, tc->think_mode,
                            tc->hf_template, &out->prompt);
    TEST_ASSERT(out->prompt.len > 0);
    TEST_ASSERT(q36_session_create(&session, engine, tc->ctx) == 0);
    TEST_ASSERT(session != NULL);
    if (!session) goto done;
    TEST_ASSERT(q36_session_sync(session, &out->prompt, err, sizeof(err)) == 0);
    out->logits = test_xcalloc((size_t)tc->steps, sizeof(out->logits[0]));
    out->tokens = test_xcalloc((size_t)tc->steps, sizeof(out->tokens[0]));
    for (int step = 0; step < tc->steps; step++) {
        int vocab = 0;
        const float *logits = q36_session_logits(session, &vocab);
        TEST_ASSERT(logits != NULL && vocab > 0);
        if (!logits || vocab <= 0) break;
        if (out->vocab == 0) out->vocab = vocab;
        TEST_ASSERT(out->vocab == vocab);
        if (out->vocab != vocab) break;
        out->logits[step] = test_xcalloc((size_t)vocab, sizeof(float));
        memcpy(out->logits[step], logits, (size_t)vocab * sizeof(float));
        out->tokens[step] = q36_session_argmax(session);
        TEST_ASSERT(out->tokens[step] >= 0);
        if (out->tokens[step] < 0) break;
        out->steps++;
        if (step + 1 >= tc->steps || out->tokens[step] == q36_token_eos(engine)) break;
        TEST_ASSERT(q36_session_eval(session, out->tokens[step], err, sizeof(err)) == 0);
    }

done:
    q36_session_free(session);
    free(prompt_text);
}

/* CPU capture and GPU replay are separate phases. Keeping both engines
 * resident makes the 16 GB release target page-fault the model continuously. */
static void test_backend_parity_replay_gpu(q36_engine *engine,
                                           const test_backend_parity_case *tc,
                                           const test_backend_parity_capture *ref) {
    q36_session *session = NULL;
    char err[160] = {0};
    TEST_ASSERT(engine != NULL && tc != NULL && ref != NULL);
    if (!engine || !tc || !ref) return;
    TEST_ASSERT(q36_session_create(&session, engine, tc->ctx) == 0);
    TEST_ASSERT(session != NULL);
    if (!session) return;
    TEST_ASSERT(q36_session_sync(session, &ref->prompt, err, sizeof(err)) == 0);
    for (int step = 0; step < ref->steps; step++) {
        int vocab = 0;
        const float *logits = q36_session_logits(session, &vocab);
        TEST_ASSERT(logits != NULL && vocab == ref->vocab);
        if (!logits || vocab != ref->vocab) break;
        test_logit_comparison result =
            test_compare_logits(logits, ref->logits[step], vocab);
        TEST_ASSERT(result.nonfinite == 0);
        test_logit_comparison_assert_strict("gpu-cpu-parity", tc->id,
                                            step, &result);
        int token = q36_session_argmax(session);
        if (token != ref->tokens[step]) {
            char *cpu_text = test_token_repr(engine, ref->tokens[step]);
            char *gpu_text = test_token_repr(engine, token);
            fprintf(stderr,
                    "q36-test: gpu-cpu-parity %s step %d argmax mismatch cpu=%d(%s) gpu=%d(%s)\n",
                    tc->id, step,
                    ref->tokens[step], cpu_text ? cpu_text : "?",
                    token, gpu_text ? gpu_text : "?");
            free(cpu_text);
            free(gpu_text);
        }
        TEST_ASSERT(token == ref->tokens[step]);
        if (step + 1 < ref->steps)
            TEST_ASSERT(q36_session_eval(session, ref->tokens[step], err, sizeof(err)) == 0);
    }
    q36_session_free(session);
}

/* KV-type parity matrix support: Q36_TEST_PARITY_REF=vulkan swaps the
 * reference engine to Vulkan; Q36_TEST_REF_CACHE_TYPE_{K,V} and
 * Q36_TEST_VK_CACHE_TYPE_{K,V} pin the cache types of each engine through
 * the Q36_TEST_CACHE_TYPE_* variables read at engine open. */
static void test_parity_cache_env(const char *k, const char *v) {
    if (k && k[0]) setenv("Q36_TEST_CACHE_TYPE_K", k, 1);
    else unsetenv("Q36_TEST_CACHE_TYPE_K");
    if (v && v[0]) setenv("Q36_TEST_CACHE_TYPE_V", v, 1);
    else unsetenv("Q36_TEST_CACHE_TYPE_V");
}

static void test_gpu_cpu_parity(void) {
    static const test_backend_parity_case cases[] = {
        {"short_italian_fact", "tests/test-vectors/qwen3.6-35b-a3b/prompts/short_italian_fact.txt", NULL, Q36_THINK_NONE, true, 4, 4096},
        {"short_code_completion", "tests/test-vectors/qwen3.6-35b-a3b/prompts/short_code_completion.txt", NULL, Q36_THINK_NONE, true, 2, 4096},
        {"short_reasoning_plain", "tests/test-vectors/qwen3.6-35b-a3b/prompts/short_reasoning_plain.txt", NULL, Q36_THINK_NONE, true, 3, 4096},
        {"long_memory_archive", "tests/test-vectors/qwen3.6-35b-a3b/prompts/long_memory_archive.txt", NULL, Q36_THINK_NONE, true, 4, 4096},
        {"long_code_audit", "tests/test-vectors/qwen3.6-35b-a3b/prompts/long_code_audit.txt", NULL, Q36_THINK_NONE, true, 4, 4096},
    };
    enum { N_CASES = sizeof(cases) / sizeof(cases[0]) };
    test_backend_parity_capture captures[N_CASES];
    bool selected[N_CASES];
    memset(captures, 0, sizeof(captures));
    memset(selected, 0, sizeof(selected));
    q36_engine *cpu_engine = NULL;
    q36_engine *gpu_engine = NULL;
    if (!test_model_available(test_model_path())) {
        test_skip("gpu-cpu-parity", "model file not found");
        return;
    }
    const char *load = getenv("Q36_TEST_PARITY_LOAD_DIR");
    if (load && load[0]) {
        for (size_t i = 0; i < N_CASES; i++) {
            if (!test_vector_case_selected(cases[i].id)) continue;
            char path[1024];
            snprintf(path,sizeof(path),"%s/%s.bin",load,cases[i].id);
            FILE *fp = fopen(path,"rb");
            uint32_t shape[4];
            bool ok = fp && fread(shape,sizeof(shape),1,fp) == 1 &&
                shape[0] == 0x51333650u && shape[1] > 0 && shape[1] <= 1048576u &&
                shape[2] > 0 && shape[2] <= (uint32_t)cases[i].steps &&
                shape[3] > 0 && shape[3] < (uint32_t)cases[i].ctx;
            if (ok) {
                test_backend_parity_capture *c = &captures[i];
                c->vocab = shape[1]; c->steps = shape[2];
                c->prompt.len = c->prompt.cap = shape[3];
                c->prompt.v = test_xcalloc(shape[3],sizeof(int));
                c->tokens = test_xcalloc(shape[2],sizeof(int));
                c->logits = test_xcalloc(shape[2],sizeof(float *));
                ok = fread(c->prompt.v,sizeof(int),shape[3],fp) == shape[3] &&
                     fread(c->tokens,sizeof(int),shape[2],fp) == shape[2];
                for (int j = 0; j < c->steps; j++) {
                    c->logits[j] = test_xcalloc(c->vocab,sizeof(float));
                    if (fread(c->logits[j],sizeof(float),c->vocab,fp) != (size_t)c->vocab) ok = false;
                }
                if (fgetc(fp) != EOF) ok = false;
            }
            if (fp) fclose(fp);
            TEST_ASSERT(ok);
            if (!ok) goto done;
            selected[i] = true;
        }
        goto replay;
    }
    const char *ref_env = getenv("Q36_TEST_PARITY_REF");
    bool ref_vulkan = ref_env && !strcmp(ref_env, "vulkan");
    test_parity_cache_env(getenv("Q36_TEST_REF_CACHE_TYPE_K"),
                          getenv("Q36_TEST_REF_CACHE_TYPE_V"));
    cpu_engine = test_open_backend_engine(test_model_path(),
                                          ref_vulkan ? Q36_BACKEND_VULKAN
                                                     : Q36_BACKEND_CPU,
                                          ref_vulkan ? 1 : test_reference_threads());
    if (!cpu_engine) {
        test_skip("gpu-cpu-parity", "reference backend unavailable");
        return;
    }
    if (!test_require_session_backend("gpu-cpu-parity", cpu_engine, 4096)) {
        q36_engine_close(cpu_engine);
        return;
    }
    for (size_t i = 0; i < N_CASES; i++) {
        if (!test_vector_case_selected(cases[i].id)) continue;
        selected[i] = true;
        fprintf(stderr, "q36-test: gpu-cpu-parity CPU capture %s\n", cases[i].id);
        test_backend_parity_capture_cpu(cpu_engine, &cases[i], &captures[i]);
        const char *dump = getenv("Q36_TEST_PARITY_DUMP_DIR");
        if (dump && dump[0] && captures[i].steps > 0) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/%s.bin", dump, cases[i].id);
            FILE *fp = fopen(path, "wb");
            TEST_ASSERT(fp != NULL);
            if (fp) {
                const test_backend_parity_capture *c = &captures[i];
                uint32_t shape[] = {0x51333650u, (uint32_t)c->vocab,
                    (uint32_t)c->steps, (uint32_t)c->prompt.len};
                TEST_ASSERT(fwrite(shape, sizeof(shape), 1, fp) == 1);
                TEST_ASSERT(fwrite(c->prompt.v, sizeof(int), c->prompt.len, fp) == (size_t)c->prompt.len);
                TEST_ASSERT(fwrite(c->tokens, sizeof(int), c->steps, fp) == (size_t)c->steps);
                for (int step = 0; step < c->steps; step++)
                    TEST_ASSERT(fwrite(c->logits[step], sizeof(float), c->vocab, fp) == (size_t)c->vocab);
                TEST_ASSERT(fclose(fp) == 0);
            }
        }
    }
    q36_engine_close(cpu_engine);

replay:
    test_parity_cache_env(getenv("Q36_TEST_VK_CACHE_TYPE_K"),
                          getenv("Q36_TEST_VK_CACHE_TYPE_V"));
    gpu_engine = test_open_backend_engine(test_model_path(), Q36_BACKEND_VULKAN, 1);
    if (!gpu_engine) {
        test_skip("gpu-cpu-parity", "GPU backend unavailable");
        goto done;
    }
    if (!test_require_session_backend("gpu-cpu-parity", gpu_engine, 4096))
        goto done;
    for (size_t i = 0; i < N_CASES; i++) {
        if (!selected[i]) continue;
        fprintf(stderr, "q36-test: gpu-cpu-parity GPU replay %s\n", cases[i].id);
        test_backend_parity_replay_gpu(gpu_engine, &cases[i], &captures[i]);
    }

done:
    q36_engine_close(gpu_engine);
    for (size_t i = 0; i < N_CASES; i++)
        test_backend_parity_capture_free(&captures[i]);
}

static float *test_capture_streaming_logits(bool streaming, bool cold,
                                            uint32_t cache_experts,
                                            uint32_t full_layers,
                                            int *vocab_out) {
    q36_engine_options opt = {
        .model_path = test_model_path(),
        .backend = Q36_BACKEND_VULKAN,
        .ssd_streaming = streaming,
        .ssd_streaming_cold = cold,
        .ssd_streaming_cache_experts = cache_experts,
        .ssd_streaming_full_layers = full_layers,
    };
    q36_engine *engine = NULL;
    q36_session *session = NULL;
    q36_tokens prompt = {0};
    float *copy = NULL;
    char err[160] = {0};
    int vocab = 0;

    TEST_ASSERT(q36_engine_open(&engine, &opt) == 0);
    if (!engine) goto done;
    q36_encode_chat_prompt(engine, "", "Reply with one word: Rome.",
                           Q36_THINK_NONE, &prompt);
    TEST_ASSERT(q36_session_create(&session, engine, 4096) == 0);
    TEST_ASSERT(session != NULL);
    if (!session) goto done;
    TEST_ASSERT(q36_session_sync(session, &prompt, err, sizeof(err)) == 0);
    const float *logits = q36_session_logits(session, &vocab);
    TEST_ASSERT(logits != NULL && vocab > 0);
    if (logits && vocab > 0) {
        copy = malloc((size_t)vocab * sizeof(*copy));
        TEST_ASSERT(copy != NULL);
        if (copy) memcpy(copy, logits, (size_t)vocab * sizeof(*copy));
    }

done:
    if (vocab_out) *vocab_out = vocab;
    q36_tokens_free(&prompt);
    q36_session_free(session);
    q36_engine_close(engine);
    return copy;
}

static void test_ssd_streaming_parity(void) {
    if (!test_model_available(test_model_path())) {
        test_skip("ssd-streaming-parity", "model file not found");
        return;
    }
    int resident_vocab = 0, warm_vocab = 0, pressure_vocab = 0, full_vocab = 0;
    float *resident = test_capture_streaming_logits(false, false, 0, 0,
                                                    &resident_vocab);
    float *warm = test_capture_streaming_logits(true, false, 32, 0,
                                                &warm_vocab);
    float *pressure = test_capture_streaming_logits(true, true, 8, 0,
                                                    &pressure_vocab);
    float *full = test_capture_streaming_logits(true, true, 512, 1,
                                                &full_vocab);
    TEST_ASSERT(resident && warm && pressure && full);
    TEST_ASSERT(resident_vocab == warm_vocab);
    TEST_ASSERT(resident_vocab == pressure_vocab);
    TEST_ASSERT(resident_vocab == full_vocab);
    if (resident && warm && resident_vocab == warm_vocab) {
        test_logit_comparison result =
            test_compare_logits(warm, resident, resident_vocab);
        TEST_ASSERT(result.nonfinite == 0);
        test_logit_comparison_assert_strict("ssd-streaming-parity", "warm", 0,
                                            &result);
    }
    if (resident && pressure && resident_vocab == pressure_vocab) {
        test_logit_comparison result =
            test_compare_logits(pressure, resident, resident_vocab);
        TEST_ASSERT(result.nonfinite == 0);
        test_logit_comparison_assert_strict("ssd-streaming-parity", "cold-pressure",
                                            0, &result);
    }
    if (resident && full && resident_vocab == full_vocab) {
        test_logit_comparison result =
            test_compare_logits(full, resident, resident_vocab);
        TEST_ASSERT(result.nonfinite == 0);
        test_logit_comparison_assert_strict("ssd-streaming-parity", "full-layer",
                                            0, &result);
    }
    if (warm && pressure && warm_vocab == pressure_vocab) {
#ifdef Q36_METAL
        /*
         * Metal caps streaming prefill by the number of routes that fit in
         * the configured cache, so 32-slot and 8-slot runs use different
         * chunk shapes. Each is gated strictly against resident logits above;
         * bit identity across different reduction shapes is not expected.
         */
        test_logit_comparison result =
            test_compare_logits(pressure, warm, warm_vocab);
        TEST_ASSERT(result.nonfinite == 0);
        test_logit_comparison_assert_strict(
            "ssd-streaming-parity", "metal-warm-vs-pressure", 0, &result);
#else
        TEST_ASSERT(memcmp(warm, pressure,
                           (size_t)warm_vocab * sizeof(*warm)) == 0);
#endif
    }
    free(resident);
    free(warm);
    free(pressure);
    free(full);
}

typedef struct {
    const char *name;
    char *old;
    bool had_old;
} test_env_value;

static void test_fusion_env_set(test_env_value *env, size_t n, bool fallback) {
    for (size_t i = 0; i < n; i++) {
        const char *old = getenv(env[i].name);
        env[i].had_old = old != NULL;
        env[i].old = old ? test_strdup(old) : NULL;
        if (fallback) setenv(env[i].name, "0", 1);
        else unsetenv(env[i].name);
    }
}

static void test_fusion_env_restore(test_env_value *env, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (env[i].had_old) setenv(env[i].name, env[i].old, 1);
        else unsetenv(env[i].name);
        free(env[i].old);
    }
}

static float *test_capture_fusion_logits(bool fallback, int *vocab_out) {
    test_env_value env[] = {
#ifdef Q36_METAL
        {.name = "Q36_METAL_SHARED_VK"},
        {.name = "Q36_METAL_MOE_DOWN_SUM"},
        {.name = "Q36_METAL_RECURRENT_NORM_GATE"},
        {.name = "Q36_METAL_DELTA_R4"},
        {.name = "Q36_METAL_FUSED_NORM"},
#else
        {.name = "Q36_VK_SHARED_FFN_DECODE"},
        {.name = "Q36_VK_MOE_DOWN_SUM_DECODE"},
        {.name = "Q36_VK_RECURRENT_CONV_DECODE"},
        {.name = "Q36_VK_RECURRENT_PROJECTIONS"},
        {.name = "Q36_VK_GDN_FRONT"},
        {.name = "Q36_VK_IQ1_M_MMQ"},
        {.name = "Q36_VK_MOE_GEMM"},
#endif
    };
    q36_engine_options opt = {
        .model_path = test_model_path(),
        .backend = Q36_BACKEND_VULKAN,
    };
    q36_engine *engine = NULL;
    q36_session *session = NULL;
    q36_tokens prompt = {0};
    float *copy = NULL;
    char err[160] = {0};
    int vocab = 0;

    test_fusion_env_set(env, sizeof(env) / sizeof(env[0]), fallback);
    TEST_ASSERT(q36_engine_open(&engine, &opt) == 0);
    if (!engine) goto done;
    q36_encode_chat_prompt(engine, "", "Name Italy's capital.",
                           Q36_THINK_NONE, &prompt);
    TEST_ASSERT(q36_session_create(&session, engine, 4096) == 0);
    if (!session) goto done;
    TEST_ASSERT(q36_session_sync(session, &prompt, err, sizeof(err)) == 0);
    int token = q36_session_argmax(session);
    TEST_ASSERT(token >= 0);
    if (token >= 0)
        TEST_ASSERT(q36_session_eval(session, token, err, sizeof(err)) == 0);
    const float *logits = q36_session_logits(session, &vocab);
    TEST_ASSERT(logits != NULL && vocab > 0);
    if (logits && vocab > 0) {
        copy = malloc((size_t)vocab * sizeof(*copy));
        TEST_ASSERT(copy != NULL);
        if (copy) memcpy(copy, logits, (size_t)vocab * sizeof(*copy));
    }

done:
    if (vocab_out) *vocab_out = vocab;
    q36_tokens_free(&prompt);
    q36_session_free(session);
    q36_engine_close(engine);
    test_fusion_env_restore(env, sizeof(env) / sizeof(env[0]));
    return copy;
}

static void test_vulkan_fusion_parity(void) {
    if (!test_model_available(test_model_path())) {
        test_skip("vulkan-fusion-parity", "model file not found");
        return;
    }
    int fused_vocab = 0, fallback_vocab = 0;
    float *fused = test_capture_fusion_logits(false, &fused_vocab);
    float *fallback = test_capture_fusion_logits(true, &fallback_vocab);
    TEST_ASSERT(fused && fallback);
    TEST_ASSERT(fused_vocab == fallback_vocab);
    if (fused && fallback && fused_vocab == fallback_vocab) {
        test_logit_comparison result =
            test_compare_logits(fused, fallback, fused_vocab);
        TEST_ASSERT(result.nonfinite == 0);
        test_logit_comparison_assert_strict("vulkan-fusion-parity",
                                            "fused-vs-fallback", 1, &result);
    }
    free(fused);
    free(fallback);
}

static float *test_capture_queue_logits(bool ace, int *vocab_out,
                                        int *steps_out) {
#ifdef Q36_METAL
    (void)ace;
    (void)vocab_out;
    (void)steps_out;
    return NULL;
#else
    test_env_value env = {.name = "Q36_VK_ACE"};
    q36_engine_options opt = {
        .model_path = test_model_path(),
        .backend = Q36_BACKEND_VULKAN,
    };
    q36_engine *engine = NULL;
    q36_session *session = NULL;
    q36_tokens prompt = {0};
    float *copy = NULL;
    char err[160] = {0};
    int vocab = 0;
    int steps = 0;

    test_fusion_env_set(&env, 1, !ace);
    TEST_ASSERT(q36_engine_open(&engine, &opt) == 0);
    if (!engine) goto done;
    q36_encode_chat_prompt(engine, "", "Name Italy's capital.",
                           Q36_THINK_NONE, &prompt);
    TEST_ASSERT(q36_session_create(&session, engine, 4096) == 0);
    if (!session) goto done;
    TEST_ASSERT(q36_session_sync(session, &prompt, err, sizeof(err)) == 0);
    for (int step = 0; step < 4; step++) {
        const float *logits = q36_session_logits(session, &vocab);
        TEST_ASSERT(logits != NULL && vocab > 0);
        if (!logits || vocab <= 0) break;
        if (!copy) copy = malloc(4u * (size_t)vocab * sizeof(*copy));
        TEST_ASSERT(copy != NULL);
        if (!copy) break;
        memcpy(copy + (size_t)step * vocab, logits,
               (size_t)vocab * sizeof(*copy));
        steps++;
        if (step == 3) break;
        int token = q36_session_argmax(session);
        TEST_ASSERT(token >= 0);
        if (token < 0) break;
        TEST_ASSERT(q36_session_eval(session, token, err, sizeof(err)) == 0);
    }

done:
    if (vocab_out) *vocab_out = vocab;
    if (steps_out) *steps_out = steps;
    q36_tokens_free(&prompt);
    q36_session_free(session);
    q36_engine_close(engine);
    test_fusion_env_restore(&env, 1);
    return copy;
#endif
}

static void test_vulkan_queue_parity(void) {
#ifdef Q36_METAL
    test_skip("vulkan-queue-parity", "Vulkan-only test");
#else
    if (!test_model_available(test_model_path())) {
        test_skip("vulkan-queue-parity", "model file not found");
        return;
    }
    int fallback_vocab = 0, fallback_steps = 0;
    int ace_vocab = 0, ace_steps = 0;
    float *fallback = test_capture_queue_logits(false, &fallback_vocab,
                                                &fallback_steps);
    float *ace = test_capture_queue_logits(true, &ace_vocab, &ace_steps);
    TEST_ASSERT(fallback && ace);
    TEST_ASSERT(fallback_vocab == ace_vocab);
    TEST_ASSERT(fallback_steps == ace_steps);
    if (fallback && ace && fallback_vocab == ace_vocab &&
        fallback_steps == ace_steps) {
        TEST_ASSERT(memcmp(fallback, ace,
                           (size_t)ace_vocab * ace_steps * sizeof(*ace)) == 0);
    }
    free(fallback);
    free(ace);
#endif
}
#else
static void test_gpu_cpu_parity(void) {
    test_skip("gpu-cpu-parity", "CPU-only build");
}

static void test_ssd_streaming_parity(void) {
    test_skip("ssd-streaming-parity", "CPU-only build");
}

static void test_vulkan_fusion_parity(void) {
    test_skip("vulkan-fusion-parity", "CPU-only build");
}

static void test_vulkan_queue_parity(void) {
    test_skip("vulkan-queue-parity", "CPU-only build");
}
#endif

static void test_qwen_tool_call_format(void) {
    tool_calls calls = {0};
    tool_call tc = {0};
    tc.name = strdup("get_weather");
    tc.arguments = strdup("{\"location\":\"Rome\",\"units\":\"celsius\"}");
    tool_calls_push(&calls, tc);

    buf b = {0};
    append_qwen_tool_calls_text(&b, &calls, NULL, false);
    const char *rendered = b.ptr ? b.ptr : "";

    TEST_ASSERT(strstr(rendered, "<tool_call>") != NULL);
    TEST_ASSERT(strstr(rendered, "</tool_call>") != NULL);
    TEST_ASSERT(strstr(rendered, "<function=get_weather>") != NULL);
    TEST_ASSERT(strstr(rendered, "<parameter=location>\nRome\n</parameter>") != NULL);
    TEST_ASSERT(strstr(rendered, "<parameter=units>\ncelsius\n</parameter>") != NULL);

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls parsed = {0};
    char *cp = strdup(rendered);
    TEST_ASSERT(parse_generated_message(cp, &content, &reasoning, &parsed));
    TEST_ASSERT(parsed.len == 1);
    TEST_ASSERT(parsed.v[0].name && !strcmp(parsed.v[0].name, "get_weather"));
    TEST_ASSERT(parsed.v[0].arguments && strstr(parsed.v[0].arguments, "\"location\":\"Rome\"") != NULL);

    free(cp);
    free(content);
    free(reasoning);
    tool_calls_free(&parsed);
    buf_free(&b);
    tool_calls_free(&calls);

    tool_calls multi = {0};
    tc.name = strdup("bash");
    tc.arguments = strdup("{\"command\":\"ls\"}");
    tool_calls_push(&multi, tc);
    tc.name = strdup("edit");
    tc.arguments = strdup("{\"filePath\":\"/tmp/a\"}");
    tool_calls_push(&multi, tc);

    memset(&b, 0, sizeof(b));
    append_qwen_tool_calls_text(&b, &multi, NULL, false);
    rendered = b.ptr ? b.ptr : "";
    int count = 0;
    for (const char *p = rendered; (p = strstr(p, "<tool_call>")) != NULL; p++) count++;
    TEST_ASSERT(count == 2);

    buf_free(&b);
    tool_calls_free(&multi);

    /* Every token split must preserve literal markers in the parameter body. */
    const char *literal = "x </tool_call> <think>literal</think> </parameter> &lt;/parameter> &amp;lt;/parameter> &lt;ordinary&gt;";
    buf args = {0};
    buf_puts(&args, "{\"code\":");
    json_escape(&args, literal);
    buf_puts(&args, "}");
    tool_calls escaped = {0};
    tool_call escaped_call = {.name = strdup("write"), .arguments = strdup(args.ptr)};
    tool_calls_push(&escaped, escaped_call);
    append_qwen_tool_calls_text(&b, &escaped, NULL, false);
    for (size_t split = 0; split <= b.len; split++) {
        qwen_tool_decode_tracker tracker;
        qwen_tool_decode_tracker_init(&tracker);
        qwen_tool_decode_tracker_update(&tracker, b.ptr, split);
        if (split < b.len - 1) TEST_ASSERT(tracker.mode != QWEN_TOOL_TRACK_DONE);
        qwen_tool_decode_tracker_update(&tracker, b.ptr, b.len);
        TEST_ASSERT(tracker.mode == QWEN_TOOL_TRACK_DONE);
    }
    content = reasoning = NULL;
    cp = strdup(b.ptr);
    TEST_ASSERT(parse_generated_message(cp, &content, &reasoning, &parsed));
    TEST_ASSERT(parsed.len == 1);
    if (parsed.len == 1) TEST_ASSERT(!strcmp(parsed.v[0].arguments, args.ptr));
    TEST_ASSERT(!reasoning || !reasoning[0]);
    free(cp);
    free(content);
    free(reasoning);
    tool_calls_free(&parsed);
    tool_calls_free(&escaped);
    buf_free(&args);
    buf_free(&b);

    fprintf(stderr, "q36-test: Qwen tool format round-trip test passed\n");
}

static void test_qwen_tool_call_quality(void) {
    if (!test_model_available(test_model_path())) {
        test_skip("qwen-tool-call-quality", "model file not found");
        return;
    }
    q36_engine *engine = test_get_engine();
    if (!engine) return;
    if (!test_require_session_backend("qwen-tool-call-quality", engine, 4096)) return;

    const char *json =
        "{"
        "\"model\":\"qwen3.6-35b-a3b\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Rome? Use the get_weather tool.\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{"
            "\"name\":\"get_weather\","
            "\"description\":\"Get weather for a location.\","
            "\"parameters\":{\"type\":\"object\",\"properties\":{"
                "\"location\":{\"type\":\"string\",\"description\":\"City name.\"}"
            "},\"required\":[\"location\"]}"
        "}}],"
        "\"tool_choice\":\"auto\","
        "\"think\":false,"
        "\"temperature\":0,"
        "\"max_tokens\":256,"
        "\"stream\":false"
        "}";

    fprintf(stderr, "q36-test: Qwen tool call quality\n");
    request r;
    char err[160];
    TEST_ASSERT(parse_chat_request(engine, NULL, json, 512, 4096, &r, err, sizeof(err)));

    q36_session *session = NULL;
    TEST_ASSERT(q36_session_create(&session, engine, 4096) == 0);
    if (!session) { request_free(&r); return; }
    TEST_ASSERT(q36_session_sync(session, &r.prompt, err, sizeof(err)) == 0);

    buf text = {0};
    uint64_t rng = 123;
    bool decode_ok = true;
    bool saw_tool_start = false;
    bool saw_tool_end = false;
    qwen_tool_decode_tracker tracker;
    qwen_tool_decode_tracker_init(&tracker);
    for (int i = 0; i < r.max_tokens; i++) {
        int token = q36_session_sample(session, r.temperature, r.top_k, r.top_p, r.min_p, &rng);
        size_t piece_len = 0;
        char *piece = q36_token_text(engine, token, &piece_len);
        buf_append(&text, piece, piece_len);
        free(piece);
        qwen_tool_decode_tracker_update(&tracker, text.ptr, text.len);
        observe_tool_markers(&tracker, text.ptr, &saw_tool_start, &saw_tool_end, NULL);
        if (saw_tool_end) break;
        if (q36_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    bool parsed = parse_generated_message(text.ptr ? text.ptr : "", &content, &reasoning, &calls);
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(parsed);
    TEST_ASSERT(calls.len > 0);
    TEST_ASSERT(!strcmp(calls.v[0].name, "get_weather"));

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    buf_free(&text);
    q36_session_free(session);
    request_free(&r);
}

static void test_think_tool_recovery_live(void) {
    if (!test_model_available(test_model_path())) {
        test_skip("think-tool-recovery", "model file not found");
        return;
    }
    q36_engine *engine = test_get_engine();
    if (!engine) return;
    const char *json =
        "{"
        "\"model\":\"qwen3.6-35b-a3b\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"List the files in the current directory. Use the provided tool; do not answer in prose.\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{"
            "\"name\":\"list_files\","
            "\"description\":\"List files in a directory.\","
            "\"parameters\":{\"type\":\"object\",\"properties\":{"
                "\"path\":{\"type\":\"string\"}"
            "},\"required\":[\"path\"]}"
        "}}],"
        "\"tool_choice\":\"auto\","
        "\"think\":true,"
        "\"temperature\":0,"
        "\"max_tokens\":384"
        "}";
    request r;
    char err[192] = {0};
    TEST_ASSERT(parse_chat_request(engine, NULL, json, 512, 4096,
                                   &r, err, sizeof(err)));
    q36_session *session = NULL;
    TEST_ASSERT(q36_session_create(&session, engine, 4096) == 0);
    if (!session) {
        request_free(&r);
        return;
    }
    TEST_ASSERT(q36_session_sync(session, &r.prompt, err, sizeof(err)) == 0);

    thinking_state thinking = thinking_state_from_prompt(&r);
    TEST_ASSERT(thinking.inside);
    buf forced = {0};
    buf text = {0};
    buf_puts(&forced,
        "The user wants a directory listing. I will call list_files now.\n\n"
        Q36_TOOL_CALLS_START);
    q36_tokens toks = {0};
    q36_tokenize_rendered_chat(engine, forced.ptr, &toks);
    TEST_ASSERT(toks.len > 1);

    size_t scan_from = 0;
    for (int i = 0; i < toks.len; i++) {
        TEST_ASSERT(q36_session_eval(session, toks.v[i], err, sizeof(err)) == 0);
        size_t piece_len = 0;
        char *piece = q36_token_text(engine, toks.v[i], &piece_len);
        buf_append(&text, piece, piece_len);
        thinking_state_feed(&thinking, piece, piece_len);
        free(piece);
    }
    TEST_ASSERT(!complete_tool_call_inside_thinking(text.ptr, text.len, &scan_from));
    TEST_ASSERT(thinking.inside);

    bool recovered = false;
    bool decode_ok = true;
    int generated = 0;
    for (; generated < 256 && !recovered; generated++) {
        int token = q36_session_argmax(session);
        if (token < 0 || token == q36_token_eos(engine)) break;
        size_t piece_len = 0;
        char *piece = q36_token_text(engine, token, &piece_len);
        buf_append(&text, piece, piece_len);
        free(piece);
        recovered = complete_tool_call_inside_thinking(text.ptr, text.len,
                                                       &scan_from);
        if (recovered) break;
        if (q36_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    bool parsed = parse_generated_message_ex(text.ptr, true, NULL,
                                              &content, &reasoning, &calls);
    fprintf(stderr,
            "q36-test: think-tool-recovery generated=%d recovered=%d calls=%d name=%s\n",
            generated, recovered, calls.len,
            calls.len ? calls.v[0].name : "-");
    if (!recovered) fprintf(stderr, "q36-test: think-tool-recovery text=[%s]\n",
                          text.ptr ? text.ptr : "");
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(recovered);
    TEST_ASSERT(parsed);
    TEST_ASSERT(calls.len > 0 && !strcmp(calls.v[0].name, "list_files"));

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    q36_tokens_free(&toks);
    buf_free(&text);
    buf_free(&forced);
    q36_session_free(session);
    request_free(&r);
    test_close_engine();
}

static bool test_mtp_boundary_after_first(void *ud, int token) {
    int *calls = ud;
    (void)token;
    return ++(*calls) > 1;
}

static void test_mtp_verifier_replay_sampling(bool sampled) {
    if (test_is_cpu_only_build()) {
        test_skip("mtp-verifier", "MTP requires Vulkan");
        return;
    }
    if (!test_model_available(test_model_path()) || !test_model_available(test_mtp_path())) {
        test_skip("mtp-verifier", "model or MTP file not found");
        return;
    }
    const char *frontier_env = getenv("Q36_TEST_MTP_FRONTIER");
    int frontier = frontier_env ? atoi(frontier_env) : 0;
    const char *draft_env = getenv("Q36_TEST_MTP_DRAFT");
    int draft = draft_env ? atoi(draft_env) : 4;
    const char *tokens_env = getenv("Q36_TEST_MTP_TOKENS");
    int limit = tokens_env ? atoi(tokens_env) : 96;
    if (limit < 16) limit = 16;
    if (limit > 512) limit = 512;
    int ctx = frontier > 0 ? frontier + limit + 32 : 4096;
    bool compact = getenv("Q36_TEST_MTP_COMPACT") != NULL;
    q36_engine_options opt = {
        .model_path = test_model_path(),
        .mtp_path = test_mtp_path(),
        .backend = Q36_BACKEND_VULKAN,
        .mtp_draft_tokens = draft,
        .mtp_margin = 0.0f,
        .cache_type_k = compact ? Q36_KV_CACHE_Q8_0 : Q36_KV_CACHE_F16,
        .cache_type_v = compact ? Q36_KV_CACHE_Q4_0 : Q36_KV_CACHE_F16,
    };
    q36_engine *engine = NULL;
    q36_session *spec = NULL;
    q36_session *plain = NULL;
    q36_tokens prompt = {0};
    char err[192] = {0};
    uint64_t rng = 42, plain_rng = 42;
    const char *temp_env = getenv("Q36_TEST_MTP_TEMP");
    float temperature = sampled ? (temp_env ? strtof(temp_env, NULL) : 1.0f) : 0.0f;
    int sampled_non_greedy = 0;
    int committed = 0;
    int multi = 0;
    int partial = 0;
    int longest = 0;
    bool restored_generated = false;
    float worst_gap = 0.0f, worst_logits = 0.0f;

    TEST_ASSERT(q36_engine_open(&engine, &opt) == 0);
    if (!engine) return;
    TEST_ASSERT(q36_engine_has_mtp(engine));
    TEST_ASSERT(q36_engine_mtp_draft_tokens(engine) > 1);
    q36_encode_chat_prompt(engine, "",
        "Explain why speculative decoding is exact, then give a small C example.",
        Q36_THINK_NONE, &prompt);
    if (frontier > 0) {
        const char *path = getenv("Q36_TEST_MTP_PROMPT");
        char *text = test_read_file(path && path[0] ? path : "tests/long_context_story_prompt.txt");
        TEST_ASSERT(text != NULL);
        if (!text) goto done;
        q36_tokens_free(&prompt);
        q36_tokenize_text(engine, text, &prompt);
        free(text);
        TEST_ASSERT(prompt.len >= frontier);
        if (prompt.len < frontier) goto done;
        prompt.len = frontier;
    }
    TEST_ASSERT(q36_session_create(&spec, engine, ctx) == 0);
    TEST_ASSERT(q36_session_create(&plain, engine, ctx) == 0);
    if (!spec || !plain) goto done;
    TEST_ASSERT(q36_session_sync(spec, &prompt, err, sizeof(err)) == 0);
    TEST_ASSERT(q36_session_sync(plain, &prompt, err, sizeof(err)) == 0);

    FILE *payload = tmpfile();
    TEST_ASSERT(payload != NULL);
    if (payload) {
        uint64_t bytes = q36_session_payload_bytes(spec);
        TEST_ASSERT(bytes > sizeof(int32_t) * (uint64_t)prompt.len + 1024);
        TEST_ASSERT(q36_session_save_payload(spec, payload, err, sizeof(err)) == 0);
        TEST_ASSERT((uint64_t)ftell(payload) == bytes);
        rewind(payload);
        TEST_ASSERT(q36_session_load_payload(spec, payload, bytes, err, sizeof(err)) == 0);
        fclose(payload);
    }
    if (sampled) {
        int top = q36_session_argmax(spec);
        float saved = q36_session_logits(spec, NULL)[top];
        uint64_t excluded_rng = 7;
        int other = q36_session_sample_excluding(spec, top,
            temperature, 0, 1.0f, 0.05f, &excluded_rng);
        TEST_ASSERT(other >= 0 && other != top);
        TEST_ASSERT(q36_session_logits(spec, NULL)[top] == saved);
        TEST_ASSERT(q36_session_argmax(spec) == top);
    }
    while (committed < limit) {
        int first = frontier > 0 ?
            q36_session_argmax_excluding(spec, q36_token_eos(engine)) : q36_session_argmax(spec);
        if (sampled) {
            int greedy = first;
            first = q36_session_sample_excluding(spec, q36_token_eos(engine),
                temperature, 0, 1.0f, 0.05f, &rng);
            int reference = q36_session_sample_excluding(plain, q36_token_eos(engine),
                temperature, 0, 1.0f, 0.05f, &plain_rng);
            TEST_ASSERT(first == reference && rng == plain_rng);
            if (first != greedy) sampled_non_greedy++;
        }
        if (first < 0 || first == q36_token_eos(engine)) break;
        int accepted[17];
        int n = (frontier > 0 ? q36_session_eval_speculative_argmax_excluding : q36_session_eval_speculative_argmax)(
            spec, first, limit - committed, q36_token_eos(engine),
            accepted, (int)(sizeof(accepted) / sizeof(accepted[0])),
            err, sizeof(err));
        TEST_ASSERT(n > 0);
        if (n <= 0) {
            fprintf(stderr, "q36-test: MTP failed: %s\n", err);
            break;
        }
        TEST_ASSERT(accepted[0] == first);
        if (n > 1) multi++;
        if (n > longest) longest = n;
        if (n > 1 && n < q36_engine_mtp_draft_tokens(engine)) partial++;

        for (int i = 0; i < n; i++) {
            int vocab = 0;
            const float *logits = q36_session_logits(plain, &vocab);
            int top = frontier > 0 ?
                q36_session_argmax_excluding(plain, q36_token_eos(engine)) : q36_session_argmax(plain);
            TEST_ASSERT(logits && accepted[i] >= 0 && accepted[i] < vocab);
            TEST_ASSERT(top >= 0);
            if (!logits || top < 0 || accepted[i] < 0 || accepted[i] >= vocab) break;
            if (!sampled || i > 0) {
                float gap = logits[top] - logits[accepted[i]];
                if (gap > worst_gap) worst_gap = gap;
                TEST_ASSERT(gap <= 1.0e-6f);
            }
            TEST_ASSERT(q36_session_eval(plain, accepted[i], err, sizeof(err)) == 0);
            committed++;
            if ((!frontier && accepted[i] == q36_token_eos(engine)) || committed >= limit) break;
        }
        TEST_ASSERT(q36_session_pos(spec) == q36_session_pos(plain));
        TEST_ASSERT(q36_session_argmax(spec) == q36_session_argmax(plain));
        int vocab = q36_qwen35_n_vocab();
        const float *sl = q36_session_logits(spec, NULL);
        const float *pl = q36_session_logits(plain, NULL);
        TEST_ASSERT(sl && pl);
        if (!sl || !pl) break;
        float diff = 0.0f;
        for (int j = 0; j < vocab; j++) {
            TEST_ASSERT(isfinite(sl[j]) && isfinite(pl[j]));
            float d = fabsf(sl[j] - pl[j]);
            if (d > diff) diff = d;
        }
        if (diff > worst_logits) worst_logits = diff;
        if (diff > 1.0e-6f)
            fprintf(stderr, "q36-test: MTP boundary pos=%d n=%d max_diff=%g\n",
                    q36_session_pos(spec), n, diff);
        TEST_ASSERT(diff <= 1.0e-6f);
        if (!restored_generated && committed >= 8 && committed < limit) {
            FILE *saved = tmpfile();
            TEST_ASSERT(saved != NULL);
            if (saved) {
                uint64_t bytes = q36_session_payload_bytes(spec);
                TEST_ASSERT(q36_session_save_payload(spec, saved, err, sizeof(err)) == 0);
                rewind(saved);
                TEST_ASSERT(q36_session_load_payload(spec, saved, bytes, err, sizeof(err)) == 0);
                fclose(saved);
            }
            restored_generated = true;
        }
    }

    fprintf(stderr,
            "q36-test: MTP replay committed=%d multi=%d partial=%d longest=%d worst_argmax_gap=%g worst_logits_diff=%g\n",
            committed, multi, partial, longest, worst_gap, worst_logits);
    TEST_ASSERT(committed >= 16);
    TEST_ASSERT(multi > 0);
    if (sampled) {
        fprintf(stderr, "q36-test: opportunistic non-greedy boundaries=%d temperature=%g\n",
                sampled_non_greedy, temperature);
        if (temperature > 0.0f) TEST_ASSERT(sampled_non_greedy > 0);
    }
    const char *min_env = getenv("Q36_TEST_MTP_MIN_ACCEPT");
    if (min_env) TEST_ASSERT(longest >= atoi(min_env));
    if (draft > 2) TEST_ASSERT(partial > 0);
    int accepted[17];
    int boundary_calls = 0;
    int boundary_token = q36_session_argmax(spec);
    int boundary_n = q36_session_eval_speculative_argmax_with_boundary(spec,
        boundary_token, 8, -1, accepted, 17,
        test_mtp_boundary_after_first, &boundary_calls, err, sizeof(err));
    TEST_ASSERT(boundary_calls >= 2);
    TEST_ASSERT(boundary_n == 1 && accepted[0] == boundary_token);
    TEST_ASSERT(q36_session_eval(plain, boundary_token, err, sizeof(err)) == 0);
    TEST_ASSERT(q36_session_pos(spec) == q36_session_pos(plain));
    const float *boundary_logits = q36_session_logits(spec, NULL);
    const float *plain_logits = q36_session_logits(plain, NULL);
    TEST_ASSERT(boundary_logits && plain_logits);
    if (boundary_logits && plain_logits) {
        for (int i = 0; i < q36_qwen35_n_vocab(); i++)
            TEST_ASSERT(fabsf(boundary_logits[i] - plain_logits[i]) <= 1.0e-6f);
    }
    int pos = q36_session_pos(spec), first = q36_session_argmax(spec);
    TEST_ASSERT(q36_session_eval_speculative_argmax(spec, first, 0, -1,
        accepted, 17, err, sizeof(err)) == 0);
    TEST_ASSERT(q36_session_pos(spec) == pos);
    TEST_ASSERT(q36_session_eval_speculative_argmax(spec, first, 8, first,
        accepted, 17, err, sizeof(err)) == 1);
    TEST_ASSERT(accepted[0] == first && q36_session_pos(spec) == pos + 1);
    TEST_ASSERT(q36_session_eval_speculative_argmax(spec, q36_session_argmax(spec), 8, -1,
        accepted, 1, err, sizeof(err)) == 1);

done:
    q36_tokens_free(&prompt);
    q36_session_free(plain);
    q36_session_free(spec);
    q36_engine_close(engine);
}

static void test_mtp_verifier_replay(void) {
    test_mtp_verifier_replay_sampling(false);
}

static void test_mtp_opportunistic(void) {
    test_mtp_verifier_replay_sampling(true);
}

static void test_steering_sessions(void) {
    if (test_is_cpu_only_build() || !test_model_available(test_model_path())) {
        test_skip("steering-sessions", "requires a model and GPU backend");
        return;
    }
    q36_engine_options opt = {.model_path = test_model_path(), .backend = Q36_BACKEND_CPU};
    q36_engine *engine = NULL;
    TEST_ASSERT(q36_engine_open(&engine, &opt) == 0);
    if (!engine) return;
    int layers = q36_qwen35_n_layer(), width = q36_qwen35_n_embd();
    q36_engine_close(engine);
    engine = NULL;
    size_t count = (size_t)layers * width;
    float *bank = calloc(count, sizeof(float));
    TEST_ASSERT(bank != NULL);
    if (!bank) return;
    for (int i = 0; i < layers; i++) bank[(size_t)i * width] = 1.0f;
    char path[] = "/tmp/q36-steering-sessions-XXXXXX";
    int fd = mkstemp(path);
    TEST_ASSERT(fd >= 0);
    if (fd < 0) { free(bank); return; }
    FILE *fp = fdopen(fd, "wb");
    TEST_ASSERT(fp != NULL);
    if (!fp) { close(fd); unlink(path); free(bank); return; }
    TEST_ASSERT(fwrite(bank, sizeof(float), count, fp) == count);
    TEST_ASSERT(fclose(fp) == 0);
    free(bank);
    opt.backend = test_default_backend();
    opt.context_size = 256;
    opt.session_count = 3;
    opt.prefill_chunk = 8;
    opt.directional_steering_file = path;
    TEST_ASSERT(q36_engine_open(&engine, &opt) == 0);
    unlink(path);
    if (!engine) return;
    q36_session *sessions[2] = {0};
    q36_session *fresh = NULL;
    float *oracle[2] = {0};
    q36_tokens prompt = {0};
    char err[192] = {0};
    q36_encode_chat_prompt(engine, "", "Explain DNS.", Q36_THINK_NONE, &prompt);
    for (int i = 0; i < 2; i++) {
        TEST_ASSERT(q36_session_create(&sessions[i], engine, 256) == 0);
        if (!sessions[i]) goto done;
    }
    TEST_ASSERT(q36_session_directional_steering_key(sessions[0]) == 0);
    TEST_ASSERT(q36_session_set_directional_steering(sessions[0], 0.25f, 0.5f) == 0);
    uint64_t steering_key = q36_session_directional_steering_key(sessions[0]);
    TEST_ASSERT(steering_key != 0);
    TEST_ASSERT(q36_session_set_directional_steering(sessions[0], 0, 0) == 0);
    TEST_ASSERT(q36_session_directional_steering_key(sessions[0]) == 0);
    TEST_ASSERT(q36_session_set_directional_steering(sessions[0], 0, 0.5f) == 0);
    uint64_t zero_attn_key = q36_session_directional_steering_key(sessions[0]);
    TEST_ASSERT(q36_session_set_directional_steering(sessions[0], -0.0f, 0.5f) == 0);
    TEST_ASSERT(q36_session_directional_steering_key(sessions[0]) == zero_attn_key);
    TEST_ASSERT(q36_session_set_directional_steering(sessions[0], 0.25f, 0.5f) == 0);
    TEST_ASSERT(q36_session_directional_steering_key(sessions[0]) == steering_key);
    TEST_ASSERT(q36_session_directional_steering_ffn(sessions[1]) == 0.0f);
    TEST_ASSERT(q36_session_set_directional_steering(sessions[1], -0.25f, -0.5f) == 0);
    TEST_ASSERT(q36_session_directional_steering_key(sessions[1]) != steering_key);
    TEST_ASSERT(q36_session_directional_steering_ffn(sessions[0]) == 0.5f);
    TEST_ASSERT(q36_session_create(&fresh, engine, 256) == 0);
    TEST_ASSERT(fresh && q36_session_directional_steering_ffn(fresh) == 0.0f);
    q36_session_free(fresh);
    fresh = NULL;
    TEST_ASSERT(q36_session_set_directional_steering(sessions[0], NAN, 0) != 0);
    TEST_ASSERT(q36_session_set_directional_steering_ffn(sessions[0], 101) != 0);
    TEST_ASSERT(q36_session_directional_steering_key(sessions[0]) == steering_key);
    q36_decode_item items[2] = {0};
    for (int i = 0; i < 2; i++) {
        TEST_ASSERT(q36_session_sync(sessions[i], &prompt, err, sizeof(err)) == 0);
        items[i].session = sessions[i];
        items[i].token = q36_session_argmax(sessions[i]);
    }
    /* Use homogeneous batches as the oracle: single-token and batched
     * quant kernels have different rounding policies on dense models. */
    const float attn[2] = {0.25f, -0.25f}, ffn[2] = {0.5f, -0.5f};
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            TEST_ASSERT(q36_session_set_directional_steering(sessions[j], attn[j], ffn[j]) == 0);
            q36_session_invalidate(sessions[j]);
            TEST_ASSERT(q36_session_sync(sessions[j], &prompt, err, sizeof(err)) == 0);
            TEST_ASSERT(q36_session_set_directional_steering(sessions[j], attn[i], ffn[i]) == 0);
        }
        TEST_ASSERT(q36_sessions_eval_batch(items, 2, err, sizeof(err)) == 0);
        int vocab = 0;
        const float *logits = q36_session_logits(sessions[i], &vocab);
        oracle[i] = malloc((size_t)vocab * sizeof(float));
        TEST_ASSERT(oracle[i] != NULL);
        if (!oracle[i]) goto done;
        memcpy(oracle[i], logits, (size_t)vocab * sizeof(float));
    }
    for (int i = 0; i < 2; i++) {
        TEST_ASSERT(q36_session_set_directional_steering(sessions[i], attn[i], ffn[i]) == 0);
        q36_session_invalidate(sessions[i]);
        TEST_ASSERT(q36_session_sync(sessions[i], &prompt, err, sizeof(err)) == 0);
    }
    setenv("Q36_VK_SESSION_BATCH_LOG", "1", 1);
    TEST_ASSERT(q36_sessions_eval_batch(items, 2, err, sizeof(err)) == 0);
    for (int i = 0; i < 2; i++) {
        int vocab = 0;
        const float *logits = q36_session_logits(sessions[i], &vocab);
        test_logit_comparison cmp = test_compare_logits(oracle[i], logits, vocab);
        fprintf(stderr, "q36-test: steering session %d max_abs=%g rms=%g\n", i, cmp.max_abs, cmp.rms);
        TEST_ASSERT(cmp.nonfinite == 0 && cmp.max_abs <= 1e-5f);
        TEST_ASSERT(q36_session_argmax(sessions[i]) == test_sample_argmax_local(oracle[i], vocab));
    }
    TEST_ASSERT(q36_session_directional_steering_ffn(sessions[0]) == 0.5f);
    TEST_ASSERT(q36_session_directional_steering_ffn(sessions[1]) == -0.5f);
    unsetenv("Q36_VK_SESSION_BATCH_LOG");

done:
    q36_session_free(fresh);
    for (int i = 0; i < 2; i++) {
        free(oracle[i]);
        q36_session_free(sessions[i]);
    }
    q36_tokens_free(&prompt);
    q36_engine_close(engine);
}

static void test_dense_model_identity(void) {
    const char *small = getenv("Q36_TEST_DENSE_SMALL_MODEL");
    const char *quality = getenv("Q36_TEST_DENSE_QUALITY_MODEL");
    const char *paths[] = {
        small && small[0] ? small : "gguf/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf",
        quality && quality[0] ? quality : "gguf/Qwen3.8-27B-UD-IQ3_S.gguf",
    };
    int ids[2] = {0};
    for (int i = 0; i < 2; i++) {
        if (!test_model_available(paths[i])) {
            test_skip("dense-model-identity", "both dense GGUFs are required");
            return;
        }
    }
    for (int i = 0; i < 2; i++) {
        q36_engine *engine = test_open_runtime_engine(paths[i], 1);
        TEST_ASSERT(engine != NULL);
        if (!engine) return;
        TEST_ASSERT(q36_engine_is_qwen38(engine));
        TEST_ASSERT(!strcmp(q36_engine_model_name(engine), "qwen3.8-27b"));
        ids[i] = q36_engine_model_id(engine);
        q36_engine_close(engine);
    }
    TEST_ASSERT(ids[0] == 4 && ids[1] == 3);
}

static void test_server_unit_group(void) {
    q36_server_unit_tests_run();
}

typedef void (*test_fn)(void);

typedef struct {
    const char *flag;
    const char *name;
    const char *desc;
    test_fn fn;
} q36_test_entry;

static const q36_test_entry test_entries[] = {
    {"--routed-q4-quants", "routed-q4-quants", "resident and streaming Q4_K/IQ4_XS routed experts", test_routed_q4_quants},
    {"--quant-primitives", "quant-primitives", "Q36 half and activation quantization", test_quant_primitives},
    {"--ssd-cache-shrink", "ssd-cache-shrink", "SSD expert cache runtime shrink policy", test_ssd_cache_shrink},
    {"--long-context", "long-context", "long-context continuation regression", test_long_security_continuation},
    {"--tool-call-quality", "tool-call-quality", "model emits valid tool calls", test_tool_call_quality},
    {"--qwen-tool-call-quality", "qwen-tool-call-quality", "model emits native Qwen tool calls", test_qwen_tool_call_quality},
    {"--think-tool-recovery", "think-tool-recovery", "complete Qwen tool call is recovered from unclosed thinking", test_think_tool_recovery_live},
    {"--thinking-generation", "thinking-generation", "generation with thinking-mode enabled", test_thinking_generation},
    {"--session-sync-prefix-cpu", "session-sync-prefix-cpu", "CPU session sync extends matching prefixes", test_session_sync_prefix_resume_cpu},
    {"--sampling-controls", "sampling-controls", "CPU sampler honors temperature/top-k/top-p/min-p", test_sampling_controls},
    {"--kv-cache-save-restore", "kv-cache-save-restore", "disk KV cache save/restore cycle", test_kv_cache_save_restore},
    {"--session-sync-resume", "session-sync-resume", "warm session sync matches cold rebuild", test_session_sync_resume_matches_cold_rebuild},
    {"--vulkan-session-batch", "vulkan-session-batch", "native Vulkan 1/2/4/8 resident-session decode", test_vulkan_session_batch},
    {"--gpu-cpu-parity", "gpu-cpu-parity", "CPU/GPU logits top1/top5/top15/top64 parity", test_gpu_cpu_parity},
    {"--ssd-streaming-parity", "ssd-streaming-parity", "resident/warm/cold/cache-pressure GPU parity", test_ssd_streaming_parity},
    {"--vulkan-fusion-parity", "vulkan-fusion-parity", "fused Vulkan path against complete feature fallbacks", test_vulkan_fusion_parity},
    {"--vulkan-queue-parity", "vulkan-queue-parity", "ACE and graphics queues produce bit-identical logits", test_vulkan_queue_parity},
    {"--mtp-verifier", "mtp-verifier", "exact MTP acceptance, recurrent rollback and checkpoint restore", test_mtp_verifier_replay},
    {"--mtp-opportunistic", "mtp-opportunistic", "sampled block seeds, greedy tails and exact target state", test_mtp_opportunistic},
    {"--qwen-tool-call-format", "qwen-tool-call-format", "Qwen tool-call rendering format unit test", test_qwen_tool_call_format},
    {"--vector-fixtures", "vector-fixtures", "llama.cpp fixture consistency check", test_vector_fixtures},
    {"--logprob-vectors", "logprob-vectors", "reference top-logprob vector comparison", test_logprob_vectors},
#ifdef Q36_WITH_LLAMA
    {"--llama-quant-dots", "llama-quant-dots", "quantized weight rows against optional llama.cpp kernels", test_llama_quant_kernels},
    {"--llama-parity-seq", "llama-parity-seq", "HF-template Q36 parity against live sequential llama.cpp", test_llama_parity_seq},
    {"--llama-parity-seq-long", "llama-parity-seq-long", "HF-template Q36 parity on long prompts against live sequential llama.cpp", test_llama_parity_seq_long},
    {"--llama-parity-batch-loose", "llama-parity-batch-loose", "HF-template loose Q36 parity against live batch llama.cpp", test_llama_parity_batch_loose},
#endif
    {"--mtp-lookup", "mtp-lookup", "unambiguous prompt lookup and draft bounds", test_mtp_lookup},
    {"--vulkan-kernels", "vulkan-kernels", "isolated Vulkan-kernel numeric regressions", test_vulkan_kernels},
    {"--dense-quant-model-rows", "dense-quant-model-rows", "dense model quant kernels against CPU-dequantized GGUF rows", test_dense_quant_model_rows},
    {"--dense-model-identity", "dense-model-identity", "distinct checkpoint identities for the two dense GGUFs", test_dense_model_identity},
    {"--steering-sessions", "steering-sessions", "independent steering strengths in native session batches", test_steering_sessions},
    {"--server", "server", "server parser/rendering/cache unit tests", test_server_unit_group},
};

static void test_print_help(const char *prog) {
    printf("Usage: %s [--all | TEST...]\n\n", prog);
    puts("Special modes:");
    puts("  --dump-logprob-case CASE_ID [OPTIONS]");
      puts("      Dump greedy selected tokens and top-logit rows for one prompt case.");
    puts("");
    puts("Tests:");
    puts("  --all");
    puts("      Run every test. This is the default, ordered from slower to faster.");
    for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
        printf("  %-20s %s\n", test_entries[i].flag, test_entries[i].desc);
    }
    puts("  --list");
    puts("      Print test names only.");
    puts("  --model FILE");
    puts("      Override the model path for selected tests.");
    puts("  --case FILTER");
    puts("      Run vector cases whose names contain FILTER.");
    puts("  -h, --help");
    puts("      Show this help.");
    puts("\nEnvironment:");
    puts("  Q36_TEST_MODEL=FILE        Model path. Default: " Q36_DEFAULT_MODEL_PATH);
    puts("  Q36_TEST_THREADS=N         q36 CPU threads for test helpers (0/unset = runtime default).");
    puts("  Q36_TEST_PARITY_DUMP_DIR=DIR  Save CPU parity captures for later GPU replay.");
    puts("  Q36_TEST_PARITY_LOAD_DIR=DIR  Replay captures from the same model and CPU build.");
    puts("  Q36_TEST_LONG_PROMPT=FILE  Rendered long-context regression prompt.");
    puts("  Q36_TEST_VECTOR_FILE=FILE  Simple reference vector fixture.");
    puts("  Q36_TEST_LLAMA_VECTOR_FILE=FILE  llama.cpp vector fixture.");
    puts("  Q36_TEST_SSD_STREAMING=1   Open Vulkan test engines with SSD expert streaming.");
    puts("  Q36_TEST_SSD_STREAM_CACHE_MB=N  Cap the streaming expert cache instead of the auto budget.");
    puts("  Q36_TEST_SSD_FULL_LAYERS=N  Keep a resident prefix when testing streaming kernels.");
    puts("  --hf-template             Use the official HF text-only chat template in special parity modes.");
    puts("  Pipeline options apply to the GPU replay of --gpu-cpu-parity only.");
    q36_dist_usage(stdout);
}

static const q36_test_entry *test_find_entry(const char *arg) {
    if (!strcmp(arg, "--vulkan-cpu-parity"))
        arg = "--gpu-cpu-parity";
    for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
        if (!strcmp(arg, test_entries[i].flag)) return &test_entries[i];
    }
    return NULL;
}

static void test_run_entry(const q36_test_entry *entry) {
    int before = test_failures;
    fprintf(stderr, "%s:\n", entry->name);
    entry->fn();
    fprintf(stderr, "%s: ", entry->name);
    q36_log(stderr,
            test_failures == before ? Q36_LOG_OK : Q36_LOG_ERROR,
            "%s",
            test_failures == before ? "OK" : "ERR");
    fputc('\n', stderr);
}

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "--dump-chat-prompt")) {
        return test_dump_chat_prompt_main(argc - 1, argv + 1);
    }
    if (argc >= 3 && !strcmp(argv[1], "--dump-logprob-case")) {
        return test_dump_case_main(argc - 1, argv + 1);
    }
    bool run_all = argc == 1;
    bool selected[sizeof(test_entries) / sizeof(test_entries[0])] = {0};

    for (int i = 1; i < argc; i++) {
        char error[256];
        int rc = q36_dist_parse_cli_arg(argv[i],&i,argc,argv,&test_parity_distributed,error,sizeof(error));
        if (rc < 0) { fprintf(stderr,"q36-test: %s\n",error); return 2; }
        if (rc > 0) continue;
        if (!strcmp(argv[i], "--all")) {
            run_all = true;
        } else if (!strcmp(argv[i], "--model") && i + 1 < argc) {
            test_model_override = argv[++i];
        } else if (!strcmp(argv[i], "--case") && i + 1 < argc) {
            test_case_override = argv[++i];
        } else if (!strcmp(argv[i], "--list")) {
            for (size_t j = 0; j < sizeof(test_entries) / sizeof(test_entries[0]); j++) {
                puts(test_entries[j].flag);
            }
            return 0;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            test_print_help(argv[0]);
            return 0;
        } else {
            const q36_test_entry *entry = test_find_entry(argv[i]);
            if (!entry) {
                fprintf(stderr, "q36-test: unknown test switch: %s\n", argv[i]);
                test_print_help(argv[0]);
                return 2;
            }
            selected[(size_t)(entry - test_entries)] = true;
        }
    }

    if (q36_dist_enabled(&test_parity_distributed)) {
        const q36_test_entry *parity = test_find_entry("--gpu-cpu-parity");
        bool only_parity = !run_all && test_parity_distributed.role == Q36_DISTRIBUTED_COORDINATOR;
        for (size_t i = 0; i < sizeof(test_entries)/sizeof(*test_entries); i++)
            if (selected[i] && &test_entries[i] != parity) only_parity = false;
        if (!only_parity || !selected[parity-test_entries]) {
            fprintf(stderr,"q36-test: pipeline options require only --gpu-cpu-parity and a coordinator role\n");
            return 2;
        }
    }

    if (run_all) {
        for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
            test_run_entry(&test_entries[i]);
        }
    } else {
        for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
            if (selected[i]) test_run_entry(&test_entries[i]);
        }
    }

    test_close_engine();

    if (test_failures) {
        fprintf(stderr, "q36 tests: %d failure(s)\n", test_failures);
        return 1;
    }
    puts("q36 tests: ok");
    return 0;
}
