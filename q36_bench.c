#include "q36.h"
#include "q36_distributed.h"
#include "q36_ssd.h"
#ifndef Q36_NO_GPU
#include "q36_gpu.h"
#endif

/* Purpose-built throughput benchmark.
 *
 * The benchmark walks one fixed token sequence to configurable context
 * frontiers, measuring only the newest prefill interval at each frontier.  It
 * then snapshots the live session in memory, performs a fixed-length decode
 * run without allowing EOS, restores the snapshot, and continues to the next
 * frontier.  Snapshot save/restore time is intentionally outside both timing
 * windows.
 */

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    q36_dist_options distributed;
    const char *model_path;
    const char *mtp_path;
    const char *prompt_path;
    const char *chat_prompt_path;
    const char *system;
    const char *csv_path;
    q36_backend backend;
    int threads;
    int mtp_draft_tokens;
    float mtp_margin;
    float temperature;
    uint64_t seed;
    bool mtp_opportunistic;
    int ctx_start;
    int ctx_max;
    int ctx_alloc;
    int step_incr;
    int gen_tokens;
    double step_mul;
    uint32_t prefill_chunk;
    uint32_t ssd_streaming_cache_experts;
    uint32_t ssd_streaming_full_layers;
    uint64_t ssd_streaming_cache_bytes;
    uint32_t ssd_streaming_preload_experts;
    q36_kv_cache_type cache_type_k;
    q36_kv_cache_type cache_type_v;
    bool ssd_streaming;
    bool ssd_streaming_cold;
    bool ssd_streaming_full_layers_set;
    bool warm_weights;
    bool warmup;
    const char *dump_frontier_logits_dir;
} bench_config;

static double bench_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static void usage(FILE *fp) {
    fprintf(fp,
        "Usage: q36-bench --prompt-file FILE [options]\n"
        "\n"
        "Benchmarks instantaneous prefill and generation throughput at context\n"
        "frontiers such as 2048, 4096, 6144, ... . Generation defaults to greedy,\n"
        "runs for exactly --gen-tokens tokens, and skips EOS so every row is\n"
        "comparable.\n"
        "\n"
        "Input:\n"
        "  --prompt-file FILE\n"
        "      Raw benchmark text. The fixed token sequence is sliced at each frontier.\n"
        "  --chat-prompt-file FILE\n"
        "      Render FILE as one no-thinking chat user message, then slice that sequence.\n"
        "  -sys, --system TEXT\n"
        "      System prompt used only with --chat-prompt-file.\n"
        "\n"
        "Model and backend:\n"
        "  -m, --model FILE       GGUF model path. Default: " Q36_DEFAULT_MODEL_PATH "\n"
        "  --metal | --vulkan | --cpu | --backend NAME\n"
        "      Select Metal on Apple Silicon, Vulkan on Linux, or CPU.\n"
        "  -t, --threads N        CPU helper threads.\n"
        "  --prefill-chunk N      Override GPU graph prefill width. Default: model/runtime setting.\n"
        "  -ctk, --cache-type-k TYPE\n"
        "                         KV cache K type: f16, q8_0, or q4_0.\n"
        "                         Default: Vulkan/Metal resident q8_0, otherwise f16\n"
        "  -ctv, --cache-type-v TYPE\n"
        "                         KV cache V type: f16, q8_0, or q4_0.\n"
        "                         Default: Vulkan/Metal resident q4_0, otherwise f16\n"
        "  --warm-weights         Touch mapped tensor pages before benchmarking.\n"
        "  --warmup               Untimed prefill and 8 decode tokens at every frontier.\n"
        "  --mtp FILE             Optional MTP support GGUF: the greedy decode runs\n"
        "                         the speculative path when --mtp-draft > 1.\n"
        "  --mtp-draft N          Max MTP draft tokens per speculative step. Default: 1\n"
        "  --mtp-margin F         MTP draft confidence gate. Default: 3.0\n"
        "  --mtp-opportunistic    Sample block boundaries and accept greedy tails;\n"
        "                         changes the sampling distribution.\n"
        "  --temp F              Sampling temperature. Default: 0 (greedy).\n"
        "                         Other sampling defaults come from the loaded model.\n"
        "  --seed N              Sampling seed, reset at each frontier. Default: 1\n"
        "  --ssd-streaming        SSD-backed model streaming instead of full residency.\n"
        "  --ssd-streaming-cold   SSD streaming: skip the default expert-cache preload.\n"
        "  --ssd-streaming-cache-experts N|NGB\n"
        "                         Routed expert cache as count or GiB.\n"
        "  --ssd-streaming-full-layers N\n"
        "                         Keep the first N routed layers resident.\n"
        "  --ssd-streaming-preload-experts N\n"
        "                         Preload the first N expert slots.\n"
        "\n"
        "Sweep:\n"
        "  --ctx-start N          First measured frontier. Default: 2048\n"
        "  --ctx-max N            Last measured frontier. Default: 32768\n"
        "  --ctx-alloc N          Allocated context. Default: ctx-max + gen-tokens + 1\n"
        "  --step-mul F           Multiplicative step. Default: 1\n"
        "  --step-incr N          Linear step when --step-mul is 1. Default: 2048\n"
        "  --dump-frontier-logits-dir DIR  Write complete logits before decode.\n"
        "  --gen-tokens N         Decode tokens per frontier. Default: 128\n"
        "                         0 benchmarks prefill only: no snapshot, decode,\n"
        "                         or restore, so each frontier extends the last.\n"
        "\n"
        "Output:\n"
        "  --csv FILE             Write CSV there instead of stdout.\n"
        "  -h, --help             Show this help.\n");
}

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT_MAX) {
        fprintf(stderr, "q36-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static int parse_nonnegative_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v < 0 || v > INT_MAX) {
        fprintf(stderr, "q36-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static double parse_double_arg(const char *s, const char *opt) {
    char *end = NULL;
    double v = strtod(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v)) {
        fprintf(stderr, "q36-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "q36-bench: %s requires an argument\n", opt);
        exit(2);
    }
    return argv[++*i];
}

static q36_backend parse_backend(const char *s, const char *opt) {
    if (!strcmp(s, "metal")) return Q36_BACKEND_METAL;
    if (!strcmp(s, "vulkan")) return Q36_BACKEND_VULKAN;
    if (!strcmp(s, "cpu")) return Q36_BACKEND_CPU;
    fprintf(stderr, "q36-bench: invalid value for %s: %s\n", opt, s);
    fprintf(stderr, "q36-bench: valid backends are: metal, vulkan, cpu\n");
    exit(2);
}

static q36_backend default_backend(void) {
#ifdef Q36_NO_GPU
    return Q36_BACKEND_CPU;
#elif defined(__APPLE__)
    return Q36_BACKEND_METAL;
#else
    return Q36_BACKEND_VULKAN;
#endif
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "q36-bench: failed to open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "q36-bench: failed to seek %s\n", path);
        fclose(fp);
        exit(1);
    }
    long n = ftell(fp);
    if (n < 0) {
        fprintf(stderr, "q36-bench: failed to tell %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "q36-bench: failed to rewind %s\n", path);
        fclose(fp);
        exit(1);
    }
    char *buf = malloc((size_t)n + 1);
    if (!buf) {
        fprintf(stderr, "q36-bench: out of memory reading %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        fprintf(stderr, "q36-bench: failed to read %s\n", path);
        free(buf);
        fclose(fp);
        exit(1);
    }
    fclose(fp);
    buf[n] = '\0';
    return buf;
}

static bench_config parse_options(int argc, char **argv) {
    bench_config c = {
        .model_path = Q36_DEFAULT_MODEL_PATH,
        .system = "You are a helpful assistant.",
        .backend = default_backend(),
        .ctx_start = 2048,
        .ctx_max = 32768,
        .step_incr = 2048,
        .gen_tokens = 128,
        .step_mul = 1.0,
        .mtp_draft_tokens = 1,
        .mtp_margin = 3.0f,
        .seed = 1,
    };

    bool cache_type_k_set = false;
    bool cache_type_v_set = false;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        char dist_err[256] = {0};
        int dist_parse = q36_dist_parse_cli_arg(arg, &i, argc, argv,
                &c.distributed, dist_err, sizeof(dist_err));
        if (dist_parse < 0) { fprintf(stderr, "q36: %s\n", dist_err); exit(2); }
        if (dist_parse > 0) continue;
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            q36_dist_usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            c.prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--chat-prompt-file")) {
            c.chat_prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--ctx-start")) {
            c.ctx_start = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ctx-max")) {
            c.ctx_max = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ctx-alloc")) {
            c.ctx_alloc = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--step-incr")) {
            c.step_incr = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--step-mul")) {
            c.step_mul = parse_double_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--dump-frontier-logits-dir")) {
            c.dump_frontier_logits_dir = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--gen-tokens") || !strcmp(arg, "--tokens") || !strcmp(arg, "-n")) {
            c.gen_tokens = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--warmup")) {
            c.warmup = true;
        } else if (!strcmp(arg, "--csv")) {
            c.csv_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--prefill-chunk")) {
            c.prefill_chunk = (uint32_t)parse_int(
                need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.backend = parse_backend(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--vulkan")) {
            c.backend = Q36_BACKEND_VULKAN;
        } else if (!strcmp(arg, "--metal")) {
            c.backend = Q36_BACKEND_METAL;
        } else if (!strcmp(arg, "--cpu")) {
            c.backend = Q36_BACKEND_CPU;
        } else if (!strcmp(arg, "--cuda")) {
            fprintf(stderr, "q36-bench: %s is not supported; use --metal, --vulkan, or --cpu\n", arg);
            exit(2);
        } else if (!strcmp(arg, "-ctk") || !strcmp(arg, "--cache-type-k")) {
            if (!q36_parse_kv_cache_type(need_arg(&i, argc, argv, arg), &c.cache_type_k)) {
                fprintf(stderr, "q36-bench: invalid cache type for %s\n", arg);
                exit(2);
            }
            cache_type_k_set = true;
        } else if (!strcmp(arg, "-ctv") || !strcmp(arg, "--cache-type-v")) {
            if (!q36_parse_kv_cache_type(need_arg(&i, argc, argv, arg), &c.cache_type_v)) {
                fprintf(stderr, "q36-bench: invalid cache type for %s\n", arg);
                exit(2);
            }
            cache_type_v_set = true;
        } else if (!strcmp(arg, "--warm-weights")) {
            c.warm_weights = true;
        } else if (!strcmp(arg, "--mtp")) {
            c.mtp_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp-draft")) {
            c.mtp_draft_tokens = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--mtp-opportunistic")) {
            c.mtp_opportunistic = true;
        } else if (!strcmp(arg, "--temp")) {
            double v = parse_double_arg(need_arg(&i, argc, argv, arg), arg);
            if (v < 0.0 || v > 100.0) {
                fprintf(stderr, "q36-bench: --temp must be between 0 and 100\n");
                exit(2);
            }
            c.temperature = (float)v;
        } else if (!strcmp(arg, "--seed")) {
            const char *value = need_arg(&i, argc, argv, arg);
            char *end = NULL;
            errno = 0;
            c.seed = strtoull(value, &end, 10);
            if (!value[0] || value[0] == '-' || *end || errno) {
                fprintf(stderr, "q36-bench: invalid seed: %s\n", value);
                exit(2);
            }
        } else if (!strcmp(arg, "--mtp-margin")) {
            double v = parse_double_arg(need_arg(&i, argc, argv, arg), arg);
            if (v < 0.0) {
                fprintf(stderr, "q36-bench: --mtp-margin must be >= 0\n");
                exit(2);
            }
            c.mtp_margin = (float)v;
        } else if (!strcmp(arg, "--ssd-streaming")) {
            c.ssd_streaming = true;
        } else if (!strcmp(arg, "--ssd-streaming-cold")) {
            c.ssd_streaming_cold = true;
        } else if (!strcmp(arg, "--ssd-streaming-cache-experts")) {
            uint32_t experts = 0;
            uint64_t bytes = 0;
            if (!q36_parse_streaming_cache_experts_arg(
                    need_arg(&i, argc, argv, arg), &experts, &bytes)) {
                fprintf(stderr,
                        "q36-bench: --ssd-streaming-cache-experts must be a positive count or <number>GB\n");
                exit(2);
            }
            c.ssd_streaming_cache_experts = experts;
            c.ssd_streaming_cache_bytes = bytes;
        } else if (!strcmp(arg, "--ssd-streaming-full-layers")) {
            c.ssd_streaming = true;
            c.ssd_streaming_full_layers =
                (uint32_t)parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
            c.ssd_streaming_full_layers_set = true;
        } else if (!strcmp(arg, "--ssd-streaming-preload-experts")) {
            int v = parse_int(need_arg(&i, argc, argv, arg), arg);
            c.ssd_streaming_preload_experts = (uint32_t)v;
        } else {
            fprintf(stderr, "q36-bench: unknown option: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }

    if (!!c.prompt_path == !!c.chat_prompt_path) {
        fprintf(stderr, "q36-bench: specify exactly one of --prompt-file or --chat-prompt-file\n");
        exit(2);
    }
    if (c.ctx_start > c.ctx_max) {
        fprintf(stderr, "q36-bench: --ctx-start must be <= --ctx-max\n");
        exit(2);
    }
    if (c.step_mul < 1.0) {
        fprintf(stderr, "q36-bench: --step-mul must be >= 1\n");
        exit(2);
    }
    if (c.step_mul == 1.0 && c.step_incr <= 0) {
        fprintf(stderr, "q36-bench: --step-incr must be positive when --step-mul is 1\n");
        exit(2);
    }
    if (c.ctx_max > INT_MAX - c.gen_tokens - 1) {
        fprintf(stderr, "q36-bench: requested context is too large\n");
        exit(2);
    }
    if (c.ctx_max + c.gen_tokens > Q36_CONTEXT_MAX) {
        fprintf(stderr, "q36-bench: ctx-max + gen-tokens must not exceed %d\n",
                Q36_CONTEXT_MAX);
        exit(2);
    }
    if (c.ctx_alloc == 0) c.ctx_alloc = c.ctx_max + c.gen_tokens + 1;
    if (c.ctx_alloc <= c.ctx_max + c.gen_tokens) {
        fprintf(stderr, "q36-bench: --ctx-alloc must be greater than ctx-max + gen-tokens\n");
        exit(2);
    }
    if (c.ctx_alloc > Q36_CONTEXT_ALLOC_MAX) {
        fprintf(stderr, "q36-bench: --ctx-alloc must not exceed %d\n",
                Q36_CONTEXT_ALLOC_MAX);
        exit(2);
    }
    if (c.mtp_opportunistic && (!c.mtp_path || c.mtp_draft_tokens <= 1)) {
        fprintf(stderr, "q36-bench: --mtp-opportunistic requires --mtp FILE and --mtp-draft > 1\n");
        exit(2);
    }
    if (c.mtp_path && c.mtp_draft_tokens <= 1)
        fprintf(stderr, "q36-bench: --mtp with --mtp-draft <= 1 benchmarks the plain decode path\n");
    if (!cache_type_k_set)
        c.cache_type_k = q36_default_kv_cache_type_k(c.backend, c.ssd_streaming);
    if (!cache_type_v_set)
        c.cache_type_v = q36_default_kv_cache_type_v(c.backend, c.ssd_streaming);
    char dist_err[256] = {0};
    if (q36_dist_prepare_engine_options(&c.distributed, NULL, dist_err, sizeof(dist_err)) ||
        c.distributed.role == Q36_DISTRIBUTED_WORKER) {
        fprintf(stderr, "q36: %s\n", dist_err[0] ? dist_err : "run workers with q36");
        exit(2);
    }
    return c;
}

static void json_write_string(FILE *fp, const char *s) {
    fputc('"', fp);
    if (s) {
        for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
            switch (*p) {
            case '"':  fputs("\\\"", fp); break;
            case '\\': fputs("\\\\", fp); break;
            case '\b': fputs("\\b", fp); break;
            case '\f': fputs("\\f", fp); break;
            case '\n': fputs("\\n", fp); break;
            case '\r': fputs("\\r", fp); break;
            case '\t': fputs("\\t", fp); break;
            default:
                if (*p < 0x20) fprintf(fp, "\\u%04x", (unsigned)*p);
                else fputc((char)*p, fp);
                break;
            }
        }
    }
    fputc('"', fp);
}

static int write_frontier_logits_json(
        const bench_config *cfg,
        q36_engine         *engine,
        q36_session        *session,
        int                 frontier,
        int                 previous) {
    if (!cfg->dump_frontier_logits_dir) return 0;

    const int vocab = q36_engine_vocab_size(engine);
    int got = 0;
    const float *logits = q36_session_logits(session, &got);
    if (!logits || got != vocab) {
        fprintf(stderr, "q36-bench: missing frontier logits at %d\n", frontier);
        return 1;
    }
    for (int i = 0; i < vocab; i++) {
        if (!isfinite(logits[i])) {
            fprintf(stderr, "q36-bench: non-finite frontier logit %d at %d\n", i, frontier);
            return 1;
        }
    }

    char path[PATH_MAX];
    const int n = snprintf(path,
                           sizeof(path),
                           "%s/frontier_%06d.logits.json",
                           cfg->dump_frontier_logits_dir,
                           frontier);
    if (n <= 0 || (size_t)n >= sizeof(path)) {
        fprintf(stderr, "q36-bench: frontier logits path is too long\n");
        return 1;
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "q36-bench: failed to open %s: %s\n", path, strerror(errno));
        return 1;
    }

    const int argmax = q36_session_argmax(session);
    fprintf(fp, "{\n  \"source\":\"q36-bench\",\n  \"model\":");
    json_write_string(fp, cfg->model_path);
    fprintf(fp,
            ",\n  \"backend\":\"%s\",\n"
            "  \"quant_bits\":%d,\n  \"prompt_tokens\":%d,\n"
            "  \"frontier_tokens\":%d,\n  \"prefill_tokens\":%d,\n"
            "  \"ctx\":%d,\n  \"vocab\":%d,\n"
            "  \"argmax_id\":%d,\n  \"argmax_logit\":%.9g,\n  \"logits\":[",
            q36_backend_name(cfg->backend),
            q36_engine_routed_quant_bits(engine),
            frontier,
            frontier,
            frontier - previous,
            cfg->ctx_alloc,
            vocab,
            argmax,
            logits[argmax]);
    for (int i = 0; i < vocab; i++) {
        if (i) fputc(',', fp);
        if ((i % 8) == 0) fputs("\n    ", fp);
        if (isfinite(logits[i])) fprintf(fp, "%.9g", logits[i]);
        else fputs("null", fp);
    }
    fputs("\n  ]\n}\n", fp);
    int failed = ferror(fp);
    if (fclose(fp) != 0) failed = 1;
    if (failed) {
        fprintf(stderr, "q36-bench: failed to close %s\n", path);
        return 1;
    }
    return 0;
}

static int next_frontier(const bench_config *c, int cur) {
    if (cur >= c->ctx_max) return c->ctx_max;
    int next;
    if (c->step_mul == 1.0) {
        if (cur > INT_MAX - c->step_incr) next = c->ctx_max;
        else next = cur + c->step_incr;
    } else {
        const double v = ceil((double)cur * c->step_mul);
        next = v > (double)INT_MAX ? c->ctx_max : (int)v;
        if (next <= cur) next = cur + 1;
    }
    if (next > c->ctx_max) next = c->ctx_max;
    return next;
}

static void log_context_memory(const bench_config *c, q36_engine *engine) {
    q36_dist_model_info info;
    q36_engine_distributed_info(engine, &info);
    q36_context_memory m = q36_context_memory_estimate_configured(
            c->backend, c->ctx_alloc, info.prefill_cap,
            c->cache_type_k, c->cache_type_v);
    fprintf(stderr,
            "q36-bench: context buffers %.2f MiB (scratch=%.2f MiB, raw=%.2f MiB, compressed=%.2f MiB, ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            (double)m.scratch_bytes / (1024.0 * 1024.0),
            (double)m.raw_bytes / (1024.0 * 1024.0),
            (double)m.compressed_bytes / (1024.0 * 1024.0),
            c->ctx_alloc,
            q36_backend_name(c->backend),
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap);
}

int main(int argc, char **argv) {
    bench_config cfg = parse_options(argc, argv);

    q36_engine_options opt = {
        .distributed = cfg.distributed,
        .context_size = cfg.ctx_alloc,
        .model_path = cfg.model_path,
        .mtp_path = cfg.mtp_path,
        .backend = cfg.backend,
        .n_threads = cfg.threads,
        .prefill_chunk = cfg.prefill_chunk,
        .mtp_draft_tokens = cfg.mtp_draft_tokens,
        .mtp_margin = cfg.mtp_margin,
        .ssd_streaming_cache_experts = cfg.ssd_streaming_cache_experts,
        .ssd_streaming_full_layers = cfg.ssd_streaming_full_layers,
        .ssd_streaming_full_layers_set = cfg.ssd_streaming_full_layers_set,
        .ssd_streaming_cache_bytes = cfg.ssd_streaming_cache_bytes,
        .ssd_streaming_preload_experts = cfg.ssd_streaming_preload_experts,
        .ssd_streaming = cfg.ssd_streaming,
        .ssd_streaming_cold = cfg.ssd_streaming_cold,
        .cache_type_k = cfg.cache_type_k,
        .cache_type_v = cfg.cache_type_v,
        .warm_weights = cfg.warm_weights,
    };
    q36_engine *engine = NULL;
    if (q36_engine_open(&engine, &opt) != 0) return 1;
    log_context_memory(&cfg, engine);
    int top_k;
    float default_temp, top_p, min_p;
    q36_engine_sampling_defaults(engine, &default_temp, &top_k, &top_p, &min_p);
    if (cfg.temperature > 0.0f)
        fprintf(stderr, "q36-bench: sampling temperature=%g top_k=%d top_p=%g min_p=%g seed=%llu opportunistic=%d\n",
                cfg.temperature, top_k, top_p, min_p,
                (unsigned long long)cfg.seed, cfg.mtp_opportunistic);

    char *text = read_file(cfg.prompt_path ? cfg.prompt_path : cfg.chat_prompt_path);
    q36_tokens prompt = {0};
    if (cfg.chat_prompt_path) {
        q36_encode_chat_prompt(engine, cfg.system, text, Q36_THINK_NONE, &prompt);
    } else {
        q36_tokenize_text(engine, text, &prompt);
    }
    free(text);

    if (prompt.len < cfg.ctx_max) {
        fprintf(stderr,
                "q36-bench: prompt has %d tokens, need at least --ctx-max=%d\n",
                prompt.len,
                cfg.ctx_max);
        q36_tokens_free(&prompt);
        q36_engine_close(engine);
        return 1;
    }

    q36_session *session = NULL;
    if (q36_session_create(&session, engine, cfg.ctx_alloc) != 0) {
        fprintf(stderr, "q36-bench: failed to create session\n");
        q36_tokens_free(&prompt);
        q36_engine_close(engine);
        return 1;
    }

    const int eos = q36_token_eos(engine);
    const bool spec = (cfg.temperature <= 0.0f || cfg.mtp_opportunistic) &&
                      q36_engine_mtp_draft_tokens(engine) > 1 &&
                      getenv("Q36_MTP_SPEC_DISABLE") == NULL;
    if (cfg.warmup) {
        char warm_err[256] = {0};
        double started = bench_now_sec();
        /* Intermediate frontiers can select kernels absent at both ends. */
        for (int frontier = cfg.ctx_start; ; frontier = next_frontier(&cfg, frontier)) {
            q36_tokens prefix = {.v=prompt.v, .len=frontier, .cap=frontier};
            q36_session_invalidate(session);
            if (q36_session_sync(session, &prefix, warm_err, sizeof(warm_err))) {
                fprintf(stderr, "q36-bench: warm-up prefill failed: %s\n", warm_err);
                q36_session_free(session); q36_tokens_free(&prompt); q36_engine_close(engine);
                return 1;
            }
            for (int i = 0; i < 8 && q36_session_pos(session) + 1 < cfg.ctx_alloc;) {
                int token = q36_session_argmax_excluding(session, eos);
                int accepted[16], n = 1;
                int limit = cfg.ctx_alloc - q36_session_pos(session) - 1;
                if (limit > 8 - i) limit = 8 - i;
                if (token >= 0 && spec)
                    n = q36_session_eval_speculative_argmax_excluding(session, token,
                        limit, eos, accepted, 16, warm_err, sizeof(warm_err));
                else if (token < 0 || q36_session_eval(session, token, warm_err, sizeof(warm_err))) n = -1;
                if (n < 1) {
                    fprintf(stderr, "q36-bench: warm-up decode failed: %s\n", warm_err);
                    q36_session_free(session); q36_tokens_free(&prompt); q36_engine_close(engine);
                    return 1;
                }
                i += n;
            }
            if (frontier >= cfg.ctx_max) break;
        }
        q36_session_invalidate(session);
#ifndef Q36_NO_GPU
        q36_gpu_prof_reset();
#endif
        fprintf(stderr, "q36-bench: warm-up complete in %.3f s; measured run starts now\n",
                bench_now_sec() - started);
    }

    FILE *out = stdout;
    if (cfg.csv_path) {
        out = fopen(cfg.csv_path, "wb");
        if (!out) {
            fprintf(stderr, "q36-bench: failed to open %s: %s\n", cfg.csv_path, strerror(errno));
            q36_session_free(session);
            q36_tokens_free(&prompt);
            q36_engine_close(engine);
            return 1;
        }
    }
    fprintf(out, "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes\n");
    fflush(out);

    q36_session_snapshot snap = {0};
    char err[256];
    int previous = 0;
    int rc = 0;

    for (int frontier = cfg.ctx_start; ; frontier = next_frontier(&cfg, frontier)) {
        q36_tokens prefix = {
            .v = prompt.v,
            .len = frontier,
            .cap = frontier,
        };

        const bool prof_prefill = getenv("Q36_VK_PROF_PREFILL") != NULL;
#ifndef Q36_NO_GPU
        if (prof_prefill) q36_gpu_prof_reset();
#else
        (void)prof_prefill;
#endif
        const double prefill_t0 = bench_now_sec();
        if (q36_session_sync(session, &prefix, err, sizeof(err)) != 0) {
            fprintf(stderr, "q36-bench: prefill to %d failed: %s\n", frontier, err);
            rc = 1;
            break;
        }
        const double prefill_t1 = bench_now_sec();
#ifndef Q36_NO_GPU
        if (prof_prefill) q36_gpu_prof_report("prefill");
#endif
        const double prefill_sec = prefill_t1 - prefill_t0;
        const int prefill_tokens = frontier - previous;
        if (getenv("Q36_BENCH_TRACE"))
            fprintf(stderr, "q36-bench: prefill frontier=%d tokens=%d seconds=%.6f\n",
                    frontier, prefill_tokens, prefill_sec);
        if (write_frontier_logits_json(&cfg, engine, session, frontier, previous)) {
            rc = 1;
            break;
        }

        if (cfg.gen_tokens > 0 &&
            q36_session_save_snapshot(session, &snap, err, sizeof(err)) != 0) {
            fprintf(stderr, "q36-bench: snapshot at %d failed: %s\n", frontier, err);
            rc = 1;
            break;
        }

        const bool prof_decode = getenv("Q36_VK_PROF_DECODE") != NULL;
#ifndef Q36_NO_GPU
        if (prof_decode) q36_gpu_prof_reset();
#else
        (void)prof_decode;
#endif
        const bool spec_trace = getenv("Q36_BENCH_SPEC_TRACE") != NULL;
        unsigned spec_calls = 0, spec_tokens = 0, spec_hist[18] = {0};
        uint64_t rng = cfg.seed;
        const double gen_t0 = bench_now_sec();
        for (int i = 0; i < cfg.gen_tokens; ) {
            if (q36_session_pos(session) + 1 >= q36_session_ctx(session)) {
                fprintf(stderr, "q36-bench: generation would exceed allocated context at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            const int token = q36_session_sample_excluding(session, eos,
                cfg.temperature, top_k, top_p, min_p, &rng);
            if (token < 0) {
                fprintf(stderr, "q36-bench: failed to choose non-EOS token at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            if (spec) {
                /* Exclude EOS from both the sampled boundary and greedy tail
                 * so every arm measures the configured output length. */
                int toks[17];
                const int n = q36_session_eval_speculative_argmax_excluding(session,
                        token, cfg.gen_tokens - i, eos,
                        toks, (int)(sizeof(toks) / sizeof(toks[0])),
                        err, sizeof(err));
                if (n <= 0) {
                    fprintf(stderr, "q36-bench: speculative decode at frontier %d failed: %s\n",
                            frontier, n < 0 ? err : "no tokens accepted");
                    rc = 1;
                    break;
                }
                if (spec_trace) {
                    spec_calls++;
                    spec_tokens += n;
                    if (n <= 17) spec_hist[n]++;
                    if (spec_calls % 32 == 0)
                        fprintf(stderr, "q36-bench: spec calls=%u accepted=%u avg=%.3f\n",
                                spec_calls, spec_tokens, (double)spec_tokens / spec_calls);
                }
                i += n;
            } else {
                if (q36_session_eval(session, token, err, sizeof(err)) != 0) {
                    fprintf(stderr, "q36-bench: decode at frontier %d failed: %s\n", frontier, err);
                    rc = 1;
                    break;
                }
                i++;
            }
        }
        const double gen_t1 = bench_now_sec();
        if (spec_trace && spec_calls) {
            fprintf(stderr, "q36-bench: spec frontier=%d calls=%u avg=%.3f histogram=",
                    frontier, spec_calls, (double)spec_tokens / spec_calls);
            for (int n = 1; n <= 17; n++) fprintf(stderr, "%s%d:%u", n == 1 ? "" : ",", n, spec_hist[n]);
            fputc('\n', stderr);
        }
#ifndef Q36_NO_GPU
        if (prof_decode) q36_gpu_prof_report("decode");
#endif
        if (rc != 0) break;

        /* gen_tokens == 0 is a pure prefill benchmark: the live session stays
         * at the frontier and the next sync extends it.  Otherwise restore
         * the pre-decode state before extending the next frontier. The last
         * frontier needs no replay after its measurements are complete. */
        if (cfg.gen_tokens > 0 && frontier < cfg.ctx_max &&
            q36_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) {
            fprintf(stderr, "q36-bench: restore at %d failed: %s\n", frontier, err);
            rc = 1;
            break;
        }

        const double gen_sec = gen_t1 - gen_t0;
        fprintf(out,
                "%d,%d,%.2f,%d,%.2f,%llu\n",
                frontier,
                prefill_tokens,
                prefill_sec > 0.0 ? (double)prefill_tokens / prefill_sec : 0.0,
                cfg.gen_tokens,
                gen_sec > 0.0 ? (double)cfg.gen_tokens / gen_sec : 0.0,
                (unsigned long long)snap.len);
        fflush(out);

        previous = frontier;
        if (frontier >= cfg.ctx_max) break;
    }

    if (out != stdout) fclose(out);
    q36_session_snapshot_free(&snap);
    q36_session_free(session);
    q36_tokens_free(&prompt);
#ifndef Q36_NO_GPU
    if (cfg.ssd_streaming || getenv("Q36_VK_MEMORY_REPORT"))
        q36_gpu_print_memory_report("bench");
#endif
    q36_engine_close(engine);
    return rc;
}
