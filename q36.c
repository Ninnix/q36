/* =========================================================================
 * q36.c - Qwen 3.6 bootstrap.
 * =========================================================================
 *
 * This file is the one active q36 engine implementation.
 * It owns:
 * - GGUF loading and model validation
 * - Qwen 3.6 tensor binding
 * - tokenizer and chat rendering
 * - CPU inference/reference helpers
 * - session and runtime scaffolding
 *
 */

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "q36.h"
#include "q36_gpu.h"
#include "q36_iq_tables.h"
#include "q36_quant.h"
#include "q36_ssd.h"

#define Q36_GGUF_MAGIC 0x46554747u
#define Q36_MAX_DIMS 8
#define Q36_NEG_INF (-1.0e30f)
#define Q36_RMS_EPS (1.0e-6f)
/* DS4 enables Think Max at 3/8 of its full context window. */
#define Q36_THINK_MAX_MIN_CONTEXT 98304u
#define Q36_QK8_0 32u
#define Q36_Q8_0_BYTES 34u
#define Q36_QK4_0 32u
#define Q36_Q4_0_BYTES 18u
#define Q36_QK_K 256u
#define Q36_K_SCALE_SIZE 12u
#define Q36_Q8_K_BYTES 292u
#define Q36_VK_Q8_K_BYTES 296u
#define Q36_MAX_Q8_K_BYTES 4672u

#if defined(__GNUC__) || defined(__clang__)
#define Q36_MAYBE_UNUSED __attribute__((unused))
#else
#define Q36_MAYBE_UNUSED
#endif

static float q36_f16_to_f32(uint16_t h);
static uint16_t q36_f32_to_f16(float f);

static const char Q36_REASONING_EFFORT_MAX_PREFIX[] =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

enum {
    Q36_N_LAYER = 40,
    Q36_N_EMBD = 2048,
    Q36_N_VOCAB = 248320,
    Q36_TENSOR_COUNT = 733,
    Q36_CONTEXT_TRAIN = 262144,
    Q36_N_HEAD = 16,
    Q36_N_HEAD_KV = 2,
    Q36_N_HEAD_DIM = 256,
    Q36_N_VALUE_DIM = 256,
    Q36_N_ROT = 64,
    Q36_N_EXPERT = 256,
    Q36_N_EXPERT_USED = 8,
    Q36_N_FF_EXP = 512,
    Q36_N_FF_SHARED = 512,
    Q36_N_SSM_CONV = 4,
    Q36_N_SSM_STATE = 128,
    Q36_N_SSM_GROUP = 16,
    Q36_N_SSM_DT_RANK = 32,
    Q36_N_SSM_INNER = 4096,
    Q36_FULL_ATTENTION_INTERVAL = 4,
    Q36_MTP_MAX_DRAFT = 16,
    Q36_N_SSM_QK = Q36_N_SSM_GROUP * Q36_N_SSM_STATE,
    Q36_N_SSM_CONV_DIM = Q36_N_SSM_QK * 2u + Q36_N_SSM_INNER,
    Q36_CPU_SCRATCH_FLOATS = Q36_N_SSM_CONV * Q36_N_SSM_CONV_DIM,
    Q36_CPU_MAX_THREADS = 32,
    Q36_CPU_PREFILL_CHUNK_DEFAULT = 16,
};

#ifndef Q36_NO_GPU
#include "q36_streaming_hotlist.inc"
#endif

static volatile int q36_float_norm_accum_mode = 1;

static const uint32_t Q36_ROPE_SECTIONS[4] = {11, 11, 10, 0};

typedef struct {
    const char *ptr;
    uint64_t len;
} q36_str;

typedef q36_tokens token_vec;

typedef struct {
    const uint8_t *base;
    uint64_t size;
    uint64_t pos;
    char error[256];
} q36_cursor;

enum {
    GGUF_VALUE_UINT8   = 0,
    GGUF_VALUE_INT8    = 1,
    GGUF_VALUE_UINT16  = 2,
    GGUF_VALUE_INT16   = 3,
    GGUF_VALUE_UINT32  = 4,
    GGUF_VALUE_INT32   = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL    = 7,
    GGUF_VALUE_STRING  = 8,
    GGUF_VALUE_ARRAY   = 9,
    GGUF_VALUE_UINT64  = 10,
    GGUF_VALUE_INT64   = 11,
    GGUF_VALUE_FLOAT64 = 12,
};

typedef struct {
    const char *name;
    uint32_t block_elems;
    uint32_t block_bytes;
} gguf_type_info;

static const gguf_type_info gguf_types[] = {
    [0]  = {"f32", 1, 4},
    [1]  = {"f16", 1, 2},
    [8]  = {"q8_0", 32, 34},
    [10] = {"q2_k", 256, 84},
    [12] = {"q4_k", 256, 144},
    [13] = {"q5_k", 256, 176},
    [14] = {"q6_k", 256, 210},
    [15] = {"q8_k", 256, 292},
    [16] = {"iq2_xxs", 256, 66},
    [21] = {"iq3_s", 256, 110},
    [22] = {"iq2_s", 256, 82},
    [26] = {"i32", 1, 4},
};

enum {
    Q36_TENSOR_F32 = 0,
    Q36_TENSOR_F16 = 1,
    Q36_TENSOR_Q8_0 = 8,
    Q36_TENSOR_Q2_K = 10,
    Q36_TENSOR_Q4_K = 12,
    Q36_TENSOR_Q5_K = 13,
    Q36_TENSOR_Q6_K = 14,
    Q36_TENSOR_IQ2_XXS = 16,
    Q36_TENSOR_IQ3_S = 21,
    Q36_TENSOR_IQ2_S = 22,
    Q36_TENSOR_I32 = 26,
};

typedef struct {
    q36_str key;
    uint32_t type;
    uint64_t value_pos;
} q36_kv;

typedef struct {
    q36_str name;
    uint32_t ndim;
    uint64_t dim[Q36_MAX_DIMS];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t elements;
    uint64_t bytes;
} q36_tensor;

typedef struct {
    int fd;
    const uint8_t *map;
    uint64_t size;
    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint64_t alignment;
    uint64_t tensor_data_pos;
    uint64_t max_tensor_bytes;
    q36_kv *kv;
    q36_tensor *tensors;
} q36_model;

typedef struct {
    uint32_t type;
    uint64_t len;
    uint64_t data_pos;
} q36_array_ref;

typedef struct {
    uint16_t d;
    int8_t qs[Q36_QK8_0];
} q36_block_q8_0;

typedef struct {
    uint16_t d;
    uint8_t qs[Q36_QK4_0 / 2u];
} q36_block_q4_0;

typedef struct {
    float d;
    int8_t qs[Q36_QK_K];
    int16_t bsums[Q36_QK_K / 16];
} q36_block_q8_k;

typedef struct {
    uint8_t scales[Q36_QK_K / 16];
    uint8_t qs[Q36_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} q36_block_q2_k;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[Q36_K_SCALE_SIZE];
    uint8_t qs[Q36_QK_K / 2];
} q36_block_q4_k;

typedef struct {
    uint16_t d;
    uint16_t qs[Q36_QK_K / 8];
} q36_block_iq2_xxs;

typedef struct {
    uint16_t d;
    uint8_t qs[Q36_QK_K / 4];
    uint8_t qh[Q36_QK_K / 32];
    uint8_t scales[Q36_QK_K / 32];
} q36_block_iq2_s;

#define Q36_IQ3S_N_SCALE (Q36_QK_K / 64)
typedef struct {
    uint16_t d;
    uint8_t qs[Q36_QK_K / 4];
    uint8_t qh[Q36_QK_K / 32];
    uint8_t signs[Q36_QK_K / 8];
    uint8_t scales[Q36_IQ3S_N_SCALE];
} q36_block_iq3_s;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[Q36_K_SCALE_SIZE];
    uint8_t qh[Q36_QK_K / 8];
    uint8_t qs[Q36_QK_K / 2];
} q36_block_q5_k;

typedef struct {
    uint8_t ql[Q36_QK_K / 2];
    uint8_t qh[Q36_QK_K / 4];
    int8_t scales[Q36_QK_K / 16];
    uint16_t d;
} q36_block_q6_k;

typedef struct {
    q36_str key;
    int value;
    bool used;
} str_i32_entry;

typedef struct {
    str_i32_entry *entry;
    uint64_t cap;
} str_i32_table;

typedef struct q36_vocab {
    q36_str *token;
    int n_vocab;
    int bos_id;
    int eos_id;
    int im_start_id;
    int im_end_id;
    int think_start_id;
    int think_end_id;
    int vision_start_id;
    int vision_end_id;
    int image_pad_id;
    int video_pad_id;
    bool add_bos;
    str_i32_table token_to_id;
    str_i32_table merge_rank;
} q36_vocab;

typedef enum {
    Q36_LAYER_RECURRENT,
    Q36_LAYER_FULL_ATTN,
} q36_layer_kind;

typedef struct {
    q36_layer_kind kind;
    q36_tensor *attn_norm;
    q36_tensor *post_attention_norm;

    q36_tensor *attn_gate;
    q36_tensor *attn_qkv;
    q36_tensor *ssm_a;
    q36_tensor *ssm_alpha;
    q36_tensor *ssm_beta;
    q36_tensor *ssm_conv1d;
    q36_tensor *ssm_dt;
    q36_tensor *ssm_norm;
    q36_tensor *ssm_out;
    q36_tensor *attn_gate_scale;
    q36_tensor *attn_qkv_scale;
    q36_tensor *ssm_alpha_scale;
    q36_tensor *ssm_beta_scale;
    q36_tensor *ssm_out_scale;

    q36_tensor *attn_q;
    q36_tensor *attn_q_norm;
    q36_tensor *attn_k;
    q36_tensor *attn_k_norm;
    q36_tensor *attn_v;
    q36_tensor *attn_output;
    q36_tensor *attn_sinks;
    q36_tensor *attn_q_scale;
    q36_tensor *attn_k_scale;
    q36_tensor *attn_v_scale;
    q36_tensor *attn_output_scale;

    q36_tensor *ffn_gate_inp;
    q36_tensor *ffn_gate_inp_shexp;
    q36_tensor *ffn_gate_exps;
    q36_tensor *ffn_gate_shexp;
    q36_tensor *ffn_up_exps;
    q36_tensor *ffn_up_shexp;
    q36_tensor *ffn_down_exps;
    q36_tensor *ffn_down_shexp;
    q36_tensor *ffn_gate_exps_scale;
    q36_tensor *ffn_gate_shexp_scale;
    q36_tensor *ffn_up_exps_scale;
    q36_tensor *ffn_up_shexp_scale;
    q36_tensor *ffn_down_exps_scale;
    q36_tensor *ffn_down_shexp_scale;
} q36_layer_weights;

typedef struct {
    q36_tensor *token_embd;
    q36_tensor *output_norm;
    q36_tensor *output;
    q36_tensor *output_scale;
    q36_layer_weights layer[Q36_N_LAYER];
} q36_weights;

typedef struct {
    q36_tensor *token_embd;
    q36_tensor *output_norm;
    q36_tensor *output;
    q36_tensor *output_scale;
    q36_tensor *eh_proj;
    q36_tensor *enorm;
    q36_tensor *hnorm;
    q36_tensor *shared_head_norm;
    q36_layer_weights block;
} q36_mtp_weights;

struct q36_engine {
    q36_model model;
    q36_model mtp_model;
    q36_vocab vocab;
    q36_weights weights;
    q36_mtp_weights mtp_weights;
    q36_backend backend;
    uint32_t n_threads;
    uint32_t cpu_prefill_cap;
    uint32_t prefill_cap_override;
    int power_percent;
    int mtp_draft_tokens;
    float mtp_margin;
    float mtp_expert_weights_scale;
    q36_kv_cache_type cache_type_k;
    q36_kv_cache_type cache_type_v;
    char *directional_steering_file;
    float *directional_steering_dirs;
#ifndef Q36_NO_GPU
    q36_gpu_tensor *directional_steering_gpu;
    q36_gpu_tensor *session_batch_logits;
    uint32_t session_batch_logits_cap;
#endif
    float directional_steering_attn_scale;
    float directional_steering_ffn_scale;
    bool quality;
    int routed_quant_bits;
    float expert_weights_scale;
    uint32_t ssd_streaming_cache_experts;
    uint32_t ssd_streaming_full_layers;
    uint64_t ssd_streaming_cache_bytes;
    uint32_t ssd_streaming_preload_experts;
    q36_ssd_memory_lock simulated_memory;
    bool ssd_streaming;
    bool ssd_streaming_cold;
    bool ssd_streaming_full_layers_set;
    bool mtp_ready;
};

struct q36_session {
    q36_engine *engine;
    int ctx_size;
    q36_tokens checkpoint;
    bool checkpoint_valid;
    float *logits;
    float *mtp_logits;
    void *runtime;
    q36_session_progress_fn progress;
    void *progress_ud;
    q36_session_progress_fn display_progress;
    void *display_progress_ud;
    q36_session_cancel_fn cancel;
    void *cancel_ud;
    int mtp_draft_token;
    bool mtp_draft_valid;
    float mtp_draft_margin;
    int mtp_backoff;
    int mtp_backoff_len;
};

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t n_tokens;
} q36_payload_header;

typedef struct {
    uint32_t len;
    uint32_t cap;
    q36_kv_cache_type type_k;
    q36_kv_cache_type type_v;
    uint32_t k_row_bytes;
    uint32_t v_row_bytes;
    uint8_t *k;
    uint8_t *v;
} q36_full_attn_cache;

typedef struct {
    float *conv;
    float *state;
} q36_recurrent_cache;

typedef struct {
    q36_full_attn_cache full[Q36_N_LAYER];
    q36_recurrent_cache recurrent[Q36_N_LAYER];
    uint32_t prefill_cap;
    float *hidden;
    float *next_hidden;
    float *work0;
    float *work1;
    float *work2;
    float *work3;
    float *work4;
    float *work5;
    float *scores;
    float *batch_hidden;
    float *batch_next_hidden;
    float *batch_norm;
    float *batch_qg;
    float *batch_q;
    float *batch_k;
    float *batch_v;
    float *batch_attn_out;
    float *batch_recur_qkv;
    float *batch_recur_z;
    float *batch_recur_alpha;
    float *batch_recur_beta;
    float *batch_recur_proj;
    float *batch_ffn_gate_logits;
    float *batch_ffn_shared_gate;
    float *batch_ffn_shared_up;
    float *batch_ffn_shared_mid;
    float *batch_ffn_shared_out;
    float *batch_ffn_scalar;
    uint8_t *batch_xq;
} q36_cpu_runtime;

typedef void (*q36_parallel_fn)(void *ctx, uint64_t row0, uint64_t row1);

typedef struct {
    pthread_mutex_t mu;
    pthread_mutex_t submit_mu;
    pthread_cond_t work_cv;
    pthread_cond_t done_cv;
    pthread_t threads[Q36_CPU_MAX_THREADS];
    uint32_t n_threads;
    uint32_t n_workers;
    uint32_t active_threads;
    uint32_t generation;
    uint32_t done;
    bool initialized;
    bool shutdown;
    q36_parallel_fn fn;
    void *ctx;
    uint64_t n_rows;
} q36_cpu_pool;

static q36_cpu_pool q36_cpu_workers;
static pthread_mutex_t q36_cpu_workers_init_mu = PTHREAD_MUTEX_INITIALIZER;
static bool q36_cpu_workers_atexit;
static __thread int q36_cpu_parallel_depth;

#ifndef Q36_NO_GPU
/* Prefill batch width: the Vulkan scratch tensors hold this many token rows,
 * so q36_session_sync() can walk a prompt in chunks.  1024 keeps the routed
 * MoE GEMM tiles ~full (8192 slots, 32 per expert on average); the 8192-slot
 * schedule cap and the attention-partials scratch (which scales with
 * chunk x context spans) are why this does not go higher.  Chunk width was
 * never numerics-invariant once the prefill GEMM kernels landed; streaming
 * sessions keep their own 512 default. */
enum { Q36_VK_NTOK_DEFAULT = 1024 };

typedef struct {
    uint32_t cap;
    q36_kv_cache_type type_k;
    q36_kv_cache_type type_v;
    uint32_t k_row_bytes;
    uint32_t v_row_bytes;
    q36_gpu_tensor *k;
    q36_gpu_tensor *v;
} q36_vulkan_full_attn_cache;

typedef struct {
    q36_gpu_tensor *conv;
    q36_gpu_tensor *state;
} q36_vulkan_recurrent_cache;

typedef struct {
    uint32_t prefill_cap;
    bool recur_conv_fused;
    q36_vulkan_full_attn_cache full[Q36_N_LAYER];
    q36_vulkan_recurrent_cache recurrent[Q36_N_LAYER];
    q36_gpu_tensor *hidden;
    q36_gpu_tensor *next_hidden;
    /* Host-written embed staging, ping-ponged across prefill chunks: the
     * CPU fills the next chunk's rows while the GPU still runs the current
     * one, instead of mapping the GPU-written hidden tensor (which drains
     * the whole in-flight batch).  Two buffers suffice: the submit ring
     * bounds record-ahead to 4 batches, far less than one chunk. */
    q36_gpu_tensor *embed_stage[2];
    uint32_t embed_flip;
    q36_gpu_tensor *norm;
    q36_gpu_tensor *last_h;
    q36_gpu_tensor *inp_q8;
    q36_gpu_tensor *attn_qg;
    q36_gpu_tensor *attn_q;
    q36_gpu_tensor *attn_k;
    q36_gpu_tensor *attn_v;
    q36_gpu_tensor *attn_out;
    q36_gpu_tensor *recur_qkv;
    q36_gpu_tensor *recur_window;
    q36_gpu_tensor *recur_conv;
    q36_gpu_tensor *recur_z;
    q36_gpu_tensor *recur_alpha;
    q36_gpu_tensor *recur_beta;
    q36_gpu_tensor *recur_gb;
    q36_gpu_tensor *recur_q;
    q36_gpu_tensor *recur_k;
    q36_gpu_tensor *recur_v;
    q36_gpu_tensor *recur_proj;
    q36_gpu_tensor *ffn_gate_logits;
    q36_gpu_tensor *ffn_selected;
    q36_gpu_tensor *ffn_weights;
    q36_gpu_tensor *ffn_shared_gate;
    q36_gpu_tensor *ffn_shared_up;
    q36_gpu_tensor *ffn_shared_mid;
    q36_gpu_tensor *ffn_shared_out;
    q36_gpu_tensor *ffn_scalar;
    q36_gpu_tensor *logits;
    q36_gpu_tensor *scores;
    q36_vulkan_full_attn_cache mtp_full;
    q36_vulkan_recurrent_cache spec_recurrent[Q36_N_LAYER];
    q36_gpu_tensor *spec_logits;
    q36_gpu_tensor *mtp_tok_embd;
    q36_gpu_tensor *mtp_e_norm;
    q36_gpu_tensor *mtp_h_norm;
    q36_gpu_tensor *mtp_concat;
    q36_gpu_tensor *mtp_cur;
    q36_gpu_tensor *mtp_next;
    q36_gpu_tensor *mtp_head;
    q36_gpu_tensor *mtp_logits;
    bool mtp_enabled;
} q36_vulkan_runtime;
#endif

enum {
    Q36_PAYLOAD_MAGIC = 0x51563336u,
    Q36_PAYLOAD_VERSION_TOKEN_ONLY = 1,
    Q36_PAYLOAD_VERSION = 2,
    Q36_PAYLOAD_VERSION_TYPED_KV = 3,
    Q36_PAYLOAD_U32_FIELDS = 14,
    Q36_PAYLOAD_U32_FIELDS_TYPED_KV = 16,
};

static void q36_session_reset_runtime(q36_session *s);
static int sample_argmax(const float *logits, uint32_t n_vocab);
#ifndef Q36_NO_GPU
static int sample_argmax_margin(const float *logits, uint32_t n_vocab, float *margin);

static uint64_t q36_mtp_stat_calls;
static uint64_t q36_mtp_stat_drafted;
static uint64_t q36_mtp_stat_accepted;
static uint64_t q36_mtp_stat_full;
static bool q36_mtp_stat_registered;

static void q36_mtp_stats_report(void) {
    if (!q36_mtp_stat_calls) return;
    fprintf(stderr,
            "q36: MTP stats calls=%" PRIu64 " drafted=%" PRIu64 " accepted=%" PRIu64 " full=%" PRIu64 " accept=%.1f%%\n",
            q36_mtp_stat_calls,
            q36_mtp_stat_drafted,
            q36_mtp_stat_accepted,
            q36_mtp_stat_full,
            q36_mtp_stat_drafted ? 100.0 * (double)q36_mtp_stat_accepted / (double)q36_mtp_stat_drafted : 0.0);
}

static void q36_mtp_stats_add(int drafted, int accepted, bool full) {
    if (!getenv("Q36_MTP_STATS")) return;
    if (!q36_mtp_stat_registered) {
        atexit(q36_mtp_stats_report);
        q36_mtp_stat_registered = true;
    }
    q36_mtp_stat_calls++;
    if (drafted > 0) q36_mtp_stat_drafted += (uint64_t)drafted;
    if (accepted > 0) q36_mtp_stat_accepted += (uint64_t)accepted;
    if (full) q36_mtp_stat_full++;
}
#endif

static uint32_t q36_online_cpus(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) return 1;
    if ((unsigned long)n > Q36_CPU_MAX_THREADS) return Q36_CPU_MAX_THREADS;
    return (uint32_t)n;
}

static uint32_t q36_parse_env_u32(const char *name, uint32_t fallback) {
    const char *env = getenv(name);
    if (env && env[0]) {
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (end != env && v > 0) {
            if ((unsigned long)v > UINT32_MAX) return UINT32_MAX;
            return (uint32_t)v;
        }
    }
    return fallback;
}

static uint32_t q36_resolve_thread_count(int requested) {
    uint32_t n = q36_online_cpus();
    if (requested > 0) n = (uint32_t)requested;
    else n = q36_parse_env_u32("Q36_THREADS", n);
    if (n < 1) n = 1;
    if (n > Q36_CPU_MAX_THREADS) n = Q36_CPU_MAX_THREADS;
    return n;
}

static uint32_t q36_default_cpu_prefill_cap(void) {
    uint32_t cap = q36_parse_env_u32("Q36_CPU_PREFILL_CHUNK", Q36_CPU_PREFILL_CHUNK_DEFAULT);
    if (cap < 1) cap = 1;
    return cap;
}

#ifndef Q36_NO_GPU
static uint32_t q36_default_vk_prefill_cap(void) {
    uint32_t cap = q36_parse_env_u32("Q36_VK_PREFILL_CHUNK", Q36_VK_NTOK_DEFAULT);
    if (cap < 1) cap = 1;
    return cap;
}

/* SSD streaming prefill is IO-bound: each chunk re-streams close to the
 * whole routed expert set per layer, so expert bytes per token scale as
 * 1/chunk.  Measured on UD-Q4_K_M (ctx512, thermal-gated A/B): chunk 512
 * doubles prefill throughput over the resident-path default of 64.  An
 * explicit --prefill-chunk or Q36_VK_PREFILL_CHUNK still wins. */
static uint32_t q36_engine_vk_prefill_cap(const q36_engine *e) {
    if (e->prefill_cap_override) return e->prefill_cap_override;
    if (e->ssd_streaming && !getenv("Q36_VK_PREFILL_CHUNK")) return 512;
    return q36_default_vk_prefill_cap();
}
#endif

static uint32_t q36_q8k_row_bytes(uint32_t n) {
    return ((n + Q36_QK_K - 1u) / Q36_QK_K) * Q36_Q8_K_BYTES;
}

static uint32_t q36_q8_0_row_bytes(uint32_t n) {
    return ((n + Q36_QK8_0 - 1u) / Q36_QK8_0) * Q36_Q8_0_BYTES;
}

static uint32_t q36_q4_0_row_bytes(uint32_t n) {
    return ((n + Q36_QK4_0 - 1u) / Q36_QK4_0) * Q36_Q4_0_BYTES;
}

const char *q36_kv_cache_type_name(q36_kv_cache_type type) {
    switch (type) {
    case Q36_KV_CACHE_F16: return "f16";
    case Q36_KV_CACHE_Q8_0: return "q8_0";
    case Q36_KV_CACHE_Q4_0: return "q4_0";
    default: return "unknown";
    }
}

q36_kv_cache_type q36_default_kv_cache_type_k(q36_backend backend, bool ssd_streaming) {
    return backend == Q36_BACKEND_VULKAN && !ssd_streaming ?
           Q36_KV_CACHE_Q8_0 : Q36_KV_CACHE_F16;
}

q36_kv_cache_type q36_default_kv_cache_type_v(q36_backend backend, bool ssd_streaming) {
    return backend == Q36_BACKEND_VULKAN && !ssd_streaming ?
           Q36_KV_CACHE_Q4_0 : Q36_KV_CACHE_F16;
}

bool q36_parse_kv_cache_type(const char *s, q36_kv_cache_type *out) {
    if (!s || !out) return false;
    if (!strcmp(s, "f16")) {
        *out = Q36_KV_CACHE_F16;
        return true;
    }
    if (!strcmp(s, "q8_0") || !strcmp(s, "q8_o")) {
        *out = Q36_KV_CACHE_Q8_0;
        return true;
    }
    if (!strcmp(s, "q4_0") || !strcmp(s, "q4_o")) {
        *out = Q36_KV_CACHE_Q4_0;
        return true;
    }
    return false;
}

static uint32_t q36_kv_cache_row_bytes(q36_kv_cache_type type, uint32_t n) {
    switch (type) {
    case Q36_KV_CACHE_F16: return n * (uint32_t)sizeof(uint16_t);
    case Q36_KV_CACHE_Q8_0: return q36_q8_0_row_bytes(n);
    case Q36_KV_CACHE_Q4_0: return q36_q4_0_row_bytes(n);
    default: return 0;
    }
}

static float q36_kv_cache_at(const uint8_t *row, q36_kv_cache_type type, uint32_t i) {
    if (type == Q36_KV_CACHE_F16) {
        const uint16_t *h = (const uint16_t *)row;
        return q36_f16_to_f32(h[i]);
    }
    if (type == Q36_KV_CACHE_Q8_0) {
        const q36_block_q8_0 *b = (const q36_block_q8_0 *)row;
        const q36_block_q8_0 *x = b + i / Q36_QK8_0;
        return q36_f16_to_f32(x->d) * (float)x->qs[i % Q36_QK8_0];
    }
    if (type == Q36_KV_CACHE_Q4_0) {
        const q36_block_q4_0 *b = (const q36_block_q4_0 *)row;
        const q36_block_q4_0 *x = b + i / Q36_QK4_0;
        uint32_t j = i % Q36_QK4_0;
        uint8_t q = x->qs[j % (Q36_QK4_0 / 2u)];
        q = j < Q36_QK4_0 / 2u ? q & 0x0fu : q >> 4;
        return q36_f16_to_f32(x->d) * (float)((int)q - 8);
    }
    return 0.0f;
}

static void q36_kv_cache_store_row(uint8_t *dst, q36_kv_cache_type type, const float *src, uint32_t n) {
    if (type == Q36_KV_CACHE_F16) {
        uint16_t *h = (uint16_t *)dst;
        for (uint32_t i = 0; i < n; i++) h[i] = q36_f32_to_f16(src[i]);
    } else if (type == Q36_KV_CACHE_Q8_0) {
        q36_quant_q8_0(src, dst, (int64_t)n);
    } else if (type == Q36_KV_CACHE_Q4_0) {
        q36_quant_q4_0(src, dst, (int64_t)n);
    }
}

static uint64_t q36_cpu_prefill_scratch_bytes(uint32_t cap) {
    uint64_t total = 0;
    total += (uint64_t)Q36_N_EMBD * 2u * sizeof(float);
    total += (uint64_t)Q36_CPU_SCRATCH_FLOATS * 6u * sizeof(float);
    total += (uint64_t)cap * Q36_N_EMBD * 3u * sizeof(float);
    total += (uint64_t)cap * (Q36_N_HEAD * Q36_N_HEAD_DIM * 2u) * sizeof(float);
    total += (uint64_t)cap * Q36_N_SSM_INNER * 3u * sizeof(float);
    total += (uint64_t)cap * Q36_N_HEAD_KV * (Q36_N_HEAD_DIM + Q36_N_VALUE_DIM) * sizeof(float);
    total += (uint64_t)cap * Q36_N_SSM_CONV_DIM * sizeof(float);
    total += (uint64_t)cap * Q36_N_SSM_DT_RANK * 2u * sizeof(float);
    total += (uint64_t)cap * Q36_N_EXPERT * sizeof(float);
    total += (uint64_t)cap * Q36_N_FF_SHARED * 4u * sizeof(float);
    total += (uint64_t)cap * sizeof(float);
    total += (uint64_t)cap * Q36_MAX_Q8_K_BYTES;
    return total;
}

static void q36_cpu_threads_shutdown(void) {
    pthread_mutex_lock(&q36_cpu_workers_init_mu);
    if (!q36_cpu_workers.initialized) {
        pthread_mutex_unlock(&q36_cpu_workers_init_mu);
        return;
    }
    pthread_mutex_lock(&q36_cpu_workers.submit_mu);
    pthread_mutex_lock(&q36_cpu_workers.mu);
    q36_cpu_workers.shutdown = true;
    q36_cpu_workers.generation++;
    pthread_cond_broadcast(&q36_cpu_workers.work_cv);
    pthread_mutex_unlock(&q36_cpu_workers.mu);
    for (uint32_t i = 1; i < q36_cpu_workers.n_threads; i++) {
        pthread_join(q36_cpu_workers.threads[i], NULL);
    }
    pthread_mutex_unlock(&q36_cpu_workers.submit_mu);
    pthread_cond_destroy(&q36_cpu_workers.done_cv);
    pthread_cond_destroy(&q36_cpu_workers.work_cv);
    pthread_mutex_destroy(&q36_cpu_workers.submit_mu);
    pthread_mutex_destroy(&q36_cpu_workers.mu);
    memset(&q36_cpu_workers, 0, sizeof(q36_cpu_workers));
    pthread_mutex_unlock(&q36_cpu_workers_init_mu);
}

static void *q36_cpu_worker_main(void *arg) {
    uint32_t tid = (uint32_t)(uintptr_t)arg;
    uint32_t seen_generation = 0;
    for (;;) {
        pthread_mutex_lock(&q36_cpu_workers.mu);
        while (seen_generation == q36_cpu_workers.generation && !q36_cpu_workers.shutdown) {
            pthread_cond_wait(&q36_cpu_workers.work_cv, &q36_cpu_workers.mu);
        }
        if (q36_cpu_workers.shutdown) {
            pthread_mutex_unlock(&q36_cpu_workers.mu);
            return NULL;
        }
        seen_generation = q36_cpu_workers.generation;
        q36_parallel_fn fn = q36_cpu_workers.fn;
        void *ctx = q36_cpu_workers.ctx;
        uint64_t n_rows = q36_cpu_workers.n_rows;
        uint32_t n_threads = q36_cpu_workers.active_threads;
        pthread_mutex_unlock(&q36_cpu_workers.mu);

        if (fn && n_threads > 0) {
            uint64_t rows_per_thread = (n_rows + n_threads - 1u) / n_threads;
            uint64_t row0 = (uint64_t)tid * rows_per_thread;
            uint64_t row1 = row0 + rows_per_thread;
            if (row1 > n_rows) row1 = n_rows;
            if (row0 < row1) {
                q36_cpu_parallel_depth++;
                fn(ctx, row0, row1);
                q36_cpu_parallel_depth--;
            }
        }

        pthread_mutex_lock(&q36_cpu_workers.mu);
        q36_cpu_workers.done++;
        if (q36_cpu_workers.done == q36_cpu_workers.n_workers) {
            pthread_cond_signal(&q36_cpu_workers.done_cv);
        }
        pthread_mutex_unlock(&q36_cpu_workers.mu);
    }
}

static void q36_cpu_threads_reserve(uint32_t n_threads) {
    pthread_mutex_lock(&q36_cpu_workers_init_mu);
    if (!q36_cpu_workers.initialized) {
        pthread_mutex_init(&q36_cpu_workers.mu, NULL);
        pthread_mutex_init(&q36_cpu_workers.submit_mu, NULL);
        pthread_cond_init(&q36_cpu_workers.work_cv, NULL);
        pthread_cond_init(&q36_cpu_workers.done_cv, NULL);
        q36_cpu_workers.n_threads = 1;
        q36_cpu_workers.n_workers = 0;
        q36_cpu_workers.initialized = true;
        if (!q36_cpu_workers_atexit) {
            atexit(q36_cpu_threads_shutdown);
            q36_cpu_workers_atexit = true;
        }
    }
    if (n_threads > Q36_CPU_MAX_THREADS) n_threads = Q36_CPU_MAX_THREADS;
    while (q36_cpu_workers.n_threads < n_threads) {
        uint32_t tid = q36_cpu_workers.n_threads;
        if (pthread_create(&q36_cpu_workers.threads[tid], NULL, q36_cpu_worker_main,
                           (void *)(uintptr_t)tid) != 0) {
            break;
        }
        q36_cpu_workers.n_threads++;
        q36_cpu_workers.n_workers = q36_cpu_workers.n_threads - 1u;
    }
    pthread_mutex_unlock(&q36_cpu_workers_init_mu);
}

static void q36_parallel_for_rows(uint64_t n_rows,
                                  uint64_t min_parallel_rows,
                                  uint32_t requested_threads,
                                  q36_parallel_fn fn,
                                  void *ctx) {
    uint32_t n_threads;
    uint64_t rows_per_thread;
    uint64_t main_row1;
    if (!fn || n_rows == 0) return;
    if (requested_threads <= 1 || n_rows < min_parallel_rows || q36_cpu_parallel_depth > 0) {
        fn(ctx, 0, n_rows);
        return;
    }
    q36_cpu_threads_reserve(requested_threads);
    n_threads = q36_cpu_workers.n_threads;
    if (n_threads > requested_threads) n_threads = requested_threads;
    if (n_threads <= 1) {
        fn(ctx, 0, n_rows);
        return;
    }

    pthread_mutex_lock(&q36_cpu_workers.submit_mu);
    pthread_mutex_lock(&q36_cpu_workers.mu);
    q36_cpu_workers.fn = fn;
    q36_cpu_workers.ctx = ctx;
    q36_cpu_workers.n_rows = n_rows;
    q36_cpu_workers.active_threads = n_threads;
    q36_cpu_workers.done = 0;
    q36_cpu_workers.generation++;
    pthread_cond_broadcast(&q36_cpu_workers.work_cv);
    rows_per_thread = (n_rows + n_threads - 1u) / n_threads;
    main_row1 = rows_per_thread > n_rows ? n_rows : rows_per_thread;
    pthread_mutex_unlock(&q36_cpu_workers.mu);

    q36_cpu_parallel_depth++;
    fn(ctx, 0, main_row1);
    q36_cpu_parallel_depth--;

    pthread_mutex_lock(&q36_cpu_workers.mu);
    while (q36_cpu_workers.done < q36_cpu_workers.n_workers) {
        pthread_cond_wait(&q36_cpu_workers.done_cv, &q36_cpu_workers.mu);
    }
    pthread_mutex_unlock(&q36_cpu_workers.mu);
    pthread_mutex_unlock(&q36_cpu_workers.submit_mu);
}

static bool q36_backend_uses_graph(q36_backend backend) {
    return backend == Q36_BACKEND_VULKAN;
}

static bool q36_backend_supports_ssd_streaming(q36_backend backend) {
    return backend == Q36_BACKEND_VULKAN;
}

#ifndef Q36_NO_GPU
static bool q36_backend_supports_streaming_auto_cache(q36_backend backend) {
    return backend == Q36_BACKEND_VULKAN;
}
#endif

static bool q36_engine_uses_vulkan_runtime(const q36_engine *e) {
    return e && e->backend == Q36_BACKEND_VULKAN;
}

static void q36_die(const char *msg) {
    fprintf(stderr, "q36: %s\n", msg);
    exit(1);
}

static void q36_die_errno(const char *what, const char *path) {
    fprintf(stderr, "q36: %s '%s': %s\n", what, path, strerror(errno));
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) q36_die("out of memory");
    return p;
}

static void *xcalloc(size_t n, size_t size) {
    void *p = calloc(n ? n : 1, size ? size : 1);
    if (!p) q36_die("out of memory");
    return p;
}

static void *xrealloc(void *ptr, size_t n) {
    void *p = realloc(ptr, n ? n : 1);
    if (!p) q36_die("out of memory");
    return p;
}

static void *xmalloc_zeroed(size_t n, size_t size) {
    return xcalloc(n, size);
}

static char *q36_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *out = xmalloc(n);
    memcpy(out, s, n);
    return out;
}

static double q36_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static bool q36_str_eq(q36_str a, q36_str b) {
    return a.len == b.len && memcmp(a.ptr, b.ptr, a.len) == 0;
}

static bool q36_streq(q36_str s, const char *z) {
    size_t n = strlen(z);
    return s.len == n && memcmp(s.ptr, z, n) == 0;
}

static uint64_t hash_bytes(const void *ptr, uint64_t len) {
    const uint8_t *p = ptr;
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static const char *q36_log_color_code(q36_log_type type) {
    switch (type) {
    case Q36_LOG_PREFILL:
    case Q36_LOG_TIMING:
        return "\x1b[36m";
    case Q36_LOG_GENERATION:
    case Q36_LOG_OK:
        return "\x1b[32m";
    case Q36_LOG_KVCACHE:
        return "\x1b[33m";
    case Q36_LOG_TOOL:
        return "\x1b[90m";
    case Q36_LOG_WARNING:
        return "\x1b[38;5;208m";
    case Q36_LOG_ERROR:
        return "\x1b[31m";
    default:
        return "";
    }
}

bool q36_log_is_tty(FILE *fp) {
    int fd = fileno(fp);
    return fd >= 0 && isatty(fd) != 0;
}

void q36_log(FILE *fp, q36_log_type type, const char *fmt, ...) {
    va_list ap;
    const bool colorize = type != Q36_LOG_DEFAULT && q36_log_is_tty(fp);
    va_start(ap, fmt);
    if (colorize) fputs(q36_log_color_code(type), fp);
    vfprintf(fp, fmt, ap);
    if (colorize) fputs("\x1b[0m", fp);
    va_end(ap);
}

const char *q36_backend_name(q36_backend backend) {
    switch (backend) {
    case Q36_BACKEND_VULKAN: return "vulkan";
    case Q36_BACKEND_CPU: return "cpu";
    default: return "unknown";
    }
}

bool q36_think_mode_enabled(q36_think_mode mode) {
    return mode != Q36_THINK_NONE;
}

const char *q36_think_mode_name(q36_think_mode mode) {
    switch (mode) {
    case Q36_THINK_NONE: return "none";
    case Q36_THINK_HIGH: return "high";
    case Q36_THINK_MAX: return "max";
    default: return "unknown";
    }
}

const char *q36_think_max_prefix(void) {
    return Q36_REASONING_EFFORT_MAX_PREFIX;
}

uint32_t q36_think_max_min_context(void) {
    return Q36_THINK_MAX_MIN_CONTEXT;
}

q36_think_mode q36_think_mode_for_context(q36_think_mode mode, int ctx_size) {
    if (mode == Q36_THINK_MAX && ctx_size < (int)Q36_THINK_MAX_MIN_CONTEXT) {
        return Q36_THINK_HIGH;
    }
    return mode;
}

q36_context_memory q36_context_memory_estimate(q36_backend backend, int ctx_size) {
    q36_context_memory m = {0};
    uint32_t cpu_prefill_cap = q36_default_cpu_prefill_cap();
#ifndef Q36_NO_GPU
    uint32_t vk_prefill_cap = q36_default_vk_prefill_cap();
#endif
    if (ctx_size <= 0) return m;
#ifndef Q36_NO_GPU
    m.prefill_cap = backend == Q36_BACKEND_VULKAN ? vk_prefill_cap : cpu_prefill_cap;
#else
    m.prefill_cap = cpu_prefill_cap;
#endif
    m.raw_cap = backend == Q36_BACKEND_VULKAN ? (uint32_t)ctx_size : 0;
    m.comp_cap = 0;
    m.raw_bytes = (uint64_t)Q36_N_LAYER * Q36_N_HEAD_KV * Q36_N_HEAD_DIM * (uint64_t)ctx_size * sizeof(float);
    m.compressed_bytes = (uint64_t)Q36_N_LAYER * Q36_N_SSM_STATE * Q36_N_SSM_GROUP * sizeof(float);
    m.scratch_bytes = q36_cpu_prefill_scratch_bytes(m.prefill_cap);
    m.total_bytes = m.raw_bytes + m.compressed_bytes + m.scratch_bytes;
    return m;
}

q36_context_memory q36_context_memory_estimate_with_prefill(q36_backend backend,
                                                            int ctx_size,
                                                            uint32_t prefill_chunk) {
    q36_context_memory m = q36_context_memory_estimate(backend, ctx_size);
    if (prefill_chunk == 0) return m;
    m.prefill_cap = prefill_chunk;
    m.scratch_bytes = q36_cpu_prefill_scratch_bytes(prefill_chunk);
    m.total_bytes = m.raw_bytes + m.compressed_bytes + m.scratch_bytes;
    return m;
}

void q36_tokens_push(q36_tokens *tv, int token) {
    if (tv->len == tv->cap) {
        tv->cap = tv->cap ? tv->cap * 2 : 64;
        tv->v = xrealloc(tv->v, (size_t)tv->cap * sizeof(tv->v[0]));
    }
    tv->v[tv->len++] = token;
}

void q36_tokens_free(q36_tokens *tv) {
    free(tv->v);
    memset(tv, 0, sizeof(*tv));
}

void q36_tokens_copy(q36_tokens *dst, const q36_tokens *src) {
    dst->len = 0;
    for (int i = 0; i < src->len; i++) q36_tokens_push(dst, src->v[i]);
}

bool q36_tokens_starts_with(const q36_tokens *tokens, const q36_tokens *prefix) {
    if (!tokens || !prefix || prefix->len > tokens->len) return false;
    for (int i = 0; i < prefix->len; i++) {
        if (tokens->v[i] != prefix->v[i]) return false;
    }
    return true;
}

static float q36_f16_to_f32(uint16_t h) {
    return q36_quant_f16_to_f32(h);
}

static uint16_t q36_f32_to_f16(float f) {
    return q36_quant_f32_to_f16(f);
}

float q36_quant_f16_to_f32(uint16_t h) {
    const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x03ffu;
    uint32_t bits;
    float out;

    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t q36_quant_f32_to_f16(float f) {
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    uint32_t w;
    uint32_t out;
    float base;

    memcpy(&w, &f, sizeof(w));
    base = (fabsf(f) * scale_to_inf) * scale_to_zero;
    {
        const uint32_t shl1 = w + w;
        const uint32_t sign = w & UINT32_C(0x80000000);
        uint32_t bias = shl1 & UINT32_C(0xff000000);
        float bias_f;
        if (bias < UINT32_C(0x71000000)) bias = UINT32_C(0x71000000);
        bias = (bias >> 1) + UINT32_C(0x07800000);
        memcpy(&bias_f, &bias, sizeof(bias_f));
        base += bias_f;
        memcpy(&out, &base, sizeof(out));
        out = ((out >> 13) & UINT32_C(0x00007c00)) +
              (out & UINT32_C(0x00000fff));
        return (uint16_t)((sign >> 16) |
                          (shl1 > UINT32_C(0xff000000) ? UINT16_C(0x7e00) : out));
    }
}

void q36_quant_q8_0(const float *x, void *out, int64_t n) {
    q36_block_q8_0 *y = out;
    int64_t blocks = n / Q36_QK8_0;
    if (!x || !out || n < 0 || n % Q36_QK8_0) return;
    for (int64_t b = 0; b < blocks; b++) {
        float amax = 0.0f;
        for (int j = 0; j < (int)Q36_QK8_0; j++) {
            float a = fabsf(x[b * Q36_QK8_0 + j]);
            if (a > amax) amax = a;
        }
        float d = amax / 127.0f;
        float inv = d ? 1.0f / d : 0.0f;
        y[b].d = q36_quant_f32_to_f16(d);
        for (int j = 0; j < (int)Q36_QK8_0; j++)
            y[b].qs[j] = (int8_t)nearbyintf(x[b * Q36_QK8_0 + j] * inv);
    }
}

void q36_quant_q4_0(const float *x, void *out, int64_t n) {
    q36_block_q4_0 *y = out;
    int64_t blocks = n / Q36_QK4_0;
    if (!x || !out || n < 0 || n % Q36_QK4_0) return;
    for (int64_t b = 0; b < blocks; b++) {
        const float *row = x + b * Q36_QK4_0;
        float max = 0.0f;
        float amax = 0.0f;
        for (uint32_t i = 0; i < Q36_QK4_0; i++) {
            float a = fabsf(row[i]);
            if (a > amax) {
                amax = a;
                max = row[i];
            }
        }
        float d = max / -8.0f;
        float inv = d ? 1.0f / d : 0.0f;
        y[b].d = q36_quant_f32_to_f16(d);
        for (uint32_t i = 0; i < Q36_QK4_0 / 2u; i++) {
            int q0 = (int8_t)(row[i] * inv + 8.5f);
            int q1 = (int8_t)(row[i + Q36_QK4_0 / 2u] * inv + 8.5f);
            if (q0 > 15) q0 = 15;
            if (q1 > 15) q1 = 15;
            y[b].qs[i] = (uint8_t)(q0 | (q1 << 4));
        }
    }
}

void q36_quant_q8_k(const float *x, void *out, int64_t n) {
    q36_block_q8_k *y = out;
    int64_t blocks = n / Q36_QK_K;
    if (!x || !out || n < 0 || n % Q36_QK_K) return;
    for (int64_t b = 0; b < blocks; b++) {
        float max = 0.0f;
        float amax = 0.0f;
        const float *row = x + b * Q36_QK_K;
        for (int j = 0; j < (int)Q36_QK_K; j++) {
            float a = fabsf(row[j]);
            if (a > amax) {
                amax = a;
                max = row[j];
            }
        }
        if (amax == 0.0f) {
            memset(&y[b], 0, sizeof(y[b]));
            continue;
        }
        float inv = -127.0f / max;
        for (int j = 0; j < (int)Q36_QK_K; j++) {
            int v = (int)lrintf(inv * row[j]);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            y[b].qs[j] = (int8_t)v;
        }
        for (int j = 0; j < (int)Q36_QK_K / 16; j++) {
            int sum = 0;
            for (int i = 0; i < 16; i++) sum += y[b].qs[j * 16 + i];
            y[b].bsums[j] = (int16_t)sum;
        }
        y[b].d = 1.0f / inv;
    }
}

float q36_quant_dot_q8_0(const void *a, const void *b, int n) {
    const q36_block_q8_0 *x = a;
    const q36_block_q8_0 *y = b;
    float sum = 0.0f;
    if (!a || !b || n < 0 || n % Q36_QK8_0) return 0.0f;
    for (int block = 0; block < n / (int)Q36_QK8_0; block++) {
        int isum = 0;
        for (int i = 0; i < (int)Q36_QK8_0; i++) isum += x[block].qs[i] * y[block].qs[i];
        sum += q36_f16_to_f32(x[block].d) * q36_f16_to_f32(y[block].d) * (float)isum;
    }
    return sum;
}

static float q36_ref_rsqrtf(float x) {
    return 1.0f / sqrtf(x);
}

static void q36_ref_rms_norm(float *dst, const float *src, const float *weight, uint32_t n, float eps) {
    float scale;
    if (q36_float_norm_accum_mode) {
        float ss = 0.0f;
        for (uint32_t i = 0; i < n; i++) ss += src[i] * src[i];
        scale = q36_ref_rsqrtf(ss / (float)n + eps);
    } else {
        double ss = 0.0;
        for (uint32_t i = 0; i < n; i++) ss += (double)src[i] * (double)src[i];
        scale = q36_ref_rsqrtf((float)(ss / (double)n) + eps);
    }
    for (uint32_t i = 0; i < n; i++) {
        float v = src[i] * scale;
        dst[i] = weight ? v * weight[i] : v;
    }
}

static Q36_MAYBE_UNUSED void q36_ref_matmul_f32(float *out, const float *w, const float *x, uint32_t in_dim, uint32_t out_dim) {
    for (uint32_t row = 0; row < out_dim; row++) {
        const float *wr = w + (uint64_t)row * in_dim;
        double acc = 0.0;
        for (uint32_t i = 0; i < in_dim; i++) acc += (double)wr[i] * (double)x[i];
        out[row] = (float)acc;
    }
}

static float q36_sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float q36_siluf(float x) {
    return x / (1.0f + expf(-x));
}

static float q36_softplusf(float x) {
    if (x > 20.0f) return x;
    return logf(1.0f + expf(x));
}

static void q36_l2_norm(float *x, uint32_t n, float eps) {
    double ss = 0.0;
    for (uint32_t i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    if (ss <= 0.0) return;
    {
        float scale = 1.0f / fmaxf(sqrtf((float)ss), eps);
        for (uint32_t i = 0; i < n; i++) x[i] *= scale;
    }
}

static void q36_softmax_inplace(float *x, uint32_t n) {
    float maxv = -INFINITY;
    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) if (x[i] > maxv) maxv = x[i];
    for (uint32_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - maxv);
        sum += x[i];
    }
    if (sum == 0.0) return;
    {
        float inv = (float)(1.0 / sum);
        for (uint32_t i = 0; i < n; i++) x[i] *= inv;
    }
}

static bool q36_all_finite(const float *x, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        if (!isfinite(x[i])) return false;
    }
    return true;
}

static void q36_topk_indices(const float *x, uint32_t n, uint32_t k, uint32_t *idx, float *val) {
    for (uint32_t i = 0; i < k; i++) {
        idx[i] = 0;
        val[i] = -INFINITY;
    }
    for (uint32_t i = 0; i < n; i++) {
        float v = x[i];
        for (uint32_t j = 0; j < k; j++) {
            if (v > val[j]) {
                for (uint32_t t = k - 1; t > j; t--) {
                    idx[t] = idx[t - 1];
                    val[t] = val[t - 1];
                }
                idx[j] = i;
                val[j] = v;
                break;
            }
        }
    }
}

static void q36_get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63u;
        *m = q[j + 4] & 63u;
    } else {
        *d = (uint8_t)((q[j + 4] & 0x0fu) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

static bool tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes);
static void q36_dequantize_row_q8_0(const q36_block_q8_0 *x, float *y, uint64_t k);
static void q36_dequantize_row_q2_k(const q36_block_q2_k *x, float *y, uint64_t k);
static void q36_dequantize_row_q4_k(const q36_block_q4_k *x, float *y, uint64_t k);
static void q36_dequantize_row_q5_k(const q36_block_q5_k *x, float *y, uint64_t k);
static void q36_dequantize_row_q6_k(const q36_block_q6_k *x, float *y, uint64_t k);
static void q36_dequantize_row_iq2_xxs(const q36_block_iq2_xxs *x, float *y, uint64_t k);
static void q36_dequantize_row_iq2_s(const q36_block_iq2_s *x, float *y, uint64_t k);
static void q36_dequantize_row_iq3_s(const q36_block_iq3_s *x, float *y, uint64_t k);

static const uint8_t *q36_iq2xxs_grid_at(uint32_t idx) {
    return q36_iq2xxs_grid + (size_t)idx * 8u;
}

static const uint8_t *q36_iq2s_grid_at(uint32_t idx) {
    return (const uint8_t *)&q36_iq2s_grid[idx];
}

static const uint8_t *q36_iq3s_grid_at(uint32_t idx) {
    return (const uint8_t *)&q36_iq3s_grid[idx];
}

static bool q36_dequantize_row_from_ptr(uint32_t type, const uint8_t *src, float *dst, uint32_t n) {
    switch (type) {
    case Q36_TENSOR_F32:
        memcpy(dst, src, (size_t)n * sizeof(*dst));
        return true;
    case Q36_TENSOR_F16: {
        const uint16_t *h = (const uint16_t *)src;
        for (uint32_t i = 0; i < n; i++) dst[i] = q36_f16_to_f32(h[i]);
        return true;
    }
    case Q36_TENSOR_Q8_0:
        q36_dequantize_row_q8_0((const q36_block_q8_0 *)src, dst, n);
        return true;
    case Q36_TENSOR_Q2_K:
        q36_dequantize_row_q2_k((const q36_block_q2_k *)src, dst, n);
        return true;
    case Q36_TENSOR_Q4_K:
        q36_dequantize_row_q4_k((const q36_block_q4_k *)src, dst, n);
        return true;
    case Q36_TENSOR_Q5_K:
        q36_dequantize_row_q5_k((const q36_block_q5_k *)src, dst, n);
        return true;
    case Q36_TENSOR_Q6_K:
        q36_dequantize_row_q6_k((const q36_block_q6_k *)src, dst, n);
        return true;
    case Q36_TENSOR_IQ2_XXS:
        q36_dequantize_row_iq2_xxs((const q36_block_iq2_xxs *)src, dst, n);
        return true;
    case Q36_TENSOR_IQ2_S:
        q36_dequantize_row_iq2_s((const q36_block_iq2_s *)src, dst, n);
        return true;
    case Q36_TENSOR_IQ3_S:
        q36_dequantize_row_iq3_s((const q36_block_iq3_s *)src, dst, n);
        return true;
    default:
        return false;
    }
}

bool q36_quant_dequantize(uint32_t type, const void *src, float *out, uint32_t n) {
    return src && out && q36_dequantize_row_from_ptr(type, src, out, n);
}

static void q36_dequantize_row_q8_0(const q36_block_q8_0 *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % Q36_QK8_0) != 0) q36_die("q8_0 row size mismatch");
    nb = k / Q36_QK8_0;
    for (uint64_t i = 0; i < nb; i++) {
        float d = q36_f16_to_f32(x[i].d);
        for (uint32_t j = 0; j < Q36_QK8_0; j++) *y++ = d * (float)x[i].qs[j];
    }
}

static void q36_dequantize_row_q2_k(const q36_block_q2_k *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % Q36_QK_K) != 0) q36_die("q2_k row size mismatch");
    nb = k / Q36_QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *q = x[i].qs;
        float d = q36_f16_to_f32(x[i].d);
        float min = q36_f16_to_f32(x[i].dmin);
        int is = 0;
        for (uint32_t n = 0; n < Q36_QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc = x[i].scales[is++];
                float dl = d * (float)(sc & 0x0f);
                float ml = min * (float)(sc >> 4);
                for (int l = 0; l < 16; l++) *y++ = dl * (float)((q[l] >> shift) & 3u) - ml;

                sc = x[i].scales[is++];
                dl = d * (float)(sc & 0x0f);
                ml = min * (float)(sc >> 4);
                for (int l = 0; l < 16; l++) *y++ = dl * (float)((q[l + 16] >> shift) & 3u) - ml;

                shift += 2;
            }
            q += 32;
        }
    }
}

static void q36_dequantize_row_q4_k(const q36_block_q4_k *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % Q36_QK_K) != 0) q36_die("q4_k row size mismatch");
    nb = k / Q36_QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *q = x[i].qs;
        float d = q36_f16_to_f32(x[i].d);
        float min = q36_f16_to_f32(x[i].dmin);
        int is = 0;
        for (uint32_t j = 0; j < Q36_QK_K; j += 64) {
            uint8_t sc;
            uint8_t m;
            float d1;
            float d2;
            float m1;
            float m2;
            q36_get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            d1 = d * (float)sc;
            m1 = min * (float)m;
            q36_get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            d2 = d * (float)sc;
            m2 = min * (float)m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (float)(q[l] & 0x0f) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (float)(q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

static void q36_dequantize_row_q5_k(const q36_block_q5_k *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % Q36_QK_K) != 0) q36_die("q5_k row size mismatch");
    nb = k / Q36_QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *ql = x[i].qs;
        const uint8_t *qh = x[i].qh;
        float d = q36_f16_to_f32(x[i].d);
        float min = q36_f16_to_f32(x[i].dmin);
        int is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;
        for (uint32_t j = 0; j < Q36_QK_K; j += 64) {
            uint8_t sc;
            uint8_t m;
            float d1;
            float d2;
            float m1;
            float m2;
            q36_get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            d1 = d * (float)sc;
            m1 = min * (float)m;
            q36_get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            d2 = d * (float)sc;
            m2 = min * (float)m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (float)((ql[l] & 0x0f) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (float)((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 = (uint8_t)(u1 << 2);
            u2 = (uint8_t)(u2 << 2);
        }
    }
}

static void q36_dequantize_row_q6_k(const q36_block_q6_k *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % Q36_QK_K) != 0) q36_die("q6_k row size mismatch");
    nb = k / Q36_QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const float d = q36_f16_to_f32(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t *sc = x[i].scales;
        for (uint32_t n = 0; n < Q36_QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l + 0] & 0x0f) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * (float)sc[is + 0] * (float)q1;
                y[l + 32] = d * (float)sc[is + 2] * (float)q2;
                y[l + 64] = d * (float)sc[is + 4] * (float)q3;
                y[l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

static void q36_dequantize_row_iq2_xxs(const q36_block_iq2_xxs *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % Q36_QK_K) != 0) q36_die("iq2_xxs row size mismatch");
    nb = k / Q36_QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const float d = q36_f16_to_f32(x[i].d);
        for (uint32_t ib32 = 0; ib32 < Q36_QK_K / 32; ++ib32) {
            uint32_t aux32[2];
            const uint8_t *aux8;
            memcpy(aux32, x[i].qs + 4 * ib32, 2 * sizeof(uint32_t));
            aux8 = (const uint8_t *)aux32;
            {
                const float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t *grid = q36_iq2xxs_grid_at(aux8[l]);
                    const uint8_t signs = q36_ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127u];
                    for (int j = 0; j < 8; ++j) {
                        y[j] = db * (float)grid[j] * ((signs & q36_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                    }
                    y += 8;
                }
            }
        }
    }
}

static void q36_dequantize_row_iq2_s(const q36_block_iq2_s *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % Q36_QK_K) != 0) q36_die("iq2_s row size mismatch");
    nb = k / Q36_QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const float d = q36_f16_to_f32(x[i].d);
        const uint8_t *qs = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const uint8_t *signs = qs + Q36_QK_K / 8;
        for (uint32_t ib32 = 0; ib32 < Q36_QK_K / 32; ++ib32) {
            float db0 = d * (0.5f + (float)(x[i].scales[ib32] & 0x0f)) * 0.25f;
            float db1 = d * (0.5f + (float)(x[i].scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const float dl = l < 2 ? db0 : db1;
                const uint8_t *grid = q36_iq2s_grid_at((uint32_t)(qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * (float)grid[j] * ((signs[l] & q36_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                }
                y += 8;
            }
            qs += 4;
            signs += 4;
        }
    }
}

static void q36_dequantize_row_iq3_s(const q36_block_iq3_s *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % Q36_QK_K) != 0) q36_die("iq3_s row size mismatch");
    nb = k / Q36_QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const float d = q36_f16_to_f32(x[i].d);
        const uint8_t *qs = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const uint8_t *signs = x[i].signs;
        for (uint32_t ib32 = 0; ib32 < Q36_QK_K / 32; ib32 += 2) {
            const float db1 = d * (1.0f + 2.0f * (float)(x[i].scales[ib32 / 2] & 0x0f));
            const float db2 = d * (1.0f + 2.0f * (float)(x[i].scales[ib32 / 2] >> 4));
            for (int l = 0; l < 4; ++l) {
                const uint8_t *grid1 = q36_iq3s_grid_at((uint32_t)(qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256)));
                const uint8_t *grid2 = q36_iq3s_grid_at((uint32_t)(qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db1 * (float)grid1[j] * ((signs[l] & q36_kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                    y[j + 4] = db1 * (float)grid2[j] * ((signs[l] & q36_kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t *grid1 = q36_iq3s_grid_at((uint32_t)(qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256)));
                const uint8_t *grid2 = q36_iq3s_grid_at((uint32_t)(qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db2 * (float)grid1[j] * ((signs[l] & q36_kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                    y[j + 4] = db2 * (float)grid2[j] * ((signs[l] & q36_kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}

static bool q36_tensor_row_to_float(const q36_model *m, const q36_tensor *t, uint64_t row, float *dst, uint32_t n) {
    const uint8_t *src;
    uint64_t row_count;
    uint64_t row_bytes;
    if (!m || !t || !dst || t->ndim == 0 || t->dim[0] != (uint64_t)n) return false;
    row_count = t->elements / t->dim[0];
    if (row >= row_count || !tensor_nbytes(t->type, t->dim[0], &row_bytes)) return false;
    src = m->map + t->abs_offset + row * row_bytes;
    return q36_dequantize_row_from_ptr(t->type, src, dst, n);
}

static bool q36_tensor_row_ptr(const q36_model *m, const q36_tensor *t, uint64_t row, const uint8_t **src, uint64_t *row_bytes) {
    uint64_t row_count;
    uint64_t got_row_bytes;
    if (!m || !t || !src || t->ndim == 0) return false;
    row_count = t->elements / t->dim[0];
    if (row >= row_count || !tensor_nbytes(t->type, t->dim[0], &got_row_bytes)) return false;
    *src = m->map + t->abs_offset + row * got_row_bytes;
    if (row_bytes) *row_bytes = got_row_bytes;
    return true;
}

static bool q36_tensor_expert_row_ptr(const q36_model *m, const q36_tensor *t, uint32_t expert, uint32_t row,
                                      const uint8_t **src, uint64_t *row_bytes) {
    uint64_t rows_per_expert;
    uint64_t flat_row;
    if (!m || !t || !src || t->ndim != 3) return false;
    rows_per_expert = t->dim[1];
    if (expert >= t->dim[2] || row >= rows_per_expert) return false;
    flat_row = (uint64_t)expert * rows_per_expert + row;
    return q36_tensor_row_ptr(m, t, flat_row, src, row_bytes);
}

static Q36_MAYBE_UNUSED bool q36_tensor_expert_row_to_float(const q36_model *m, const q36_tensor *t, uint32_t expert, uint32_t row,
                                           float *dst, uint32_t n) {
    const uint8_t *src;
    uint64_t rows_per_expert;
    uint64_t flat_row;
    if (!m || !t || !dst || t->ndim != 3 || t->dim[0] != (uint64_t)n) return false;
    rows_per_expert = t->dim[1];
    if (expert >= t->dim[2] || row >= rows_per_expert) return false;
    flat_row = (uint64_t)expert * rows_per_expert + row;
    if (!q36_tensor_row_ptr(m, t, flat_row, &src, NULL)) return false;
    return q36_dequantize_row_from_ptr(t->type, src, dst, n);
}

static bool q36_tensor_get_plain(const q36_model *m, const q36_tensor *t, float *dst, uint32_t n) {
    if (!m || !t || !dst || t->ndim != 1 || t->dim[0] != (uint64_t)n) return false;
    return q36_dequantize_row_from_ptr(t->type, m->map + t->abs_offset, dst, n);
}

static float q36_tensor_scalar_or(const q36_model *m, const q36_tensor *t, float fallback) {
    float v = fallback;
    if (!t) return fallback;
    if (!q36_tensor_get_plain(m, t, &v, 1)) return fallback;
    return v;
}

static float q36_tensor_index_or(const q36_model *m, const q36_tensor *t, uint32_t idx, float fallback) {
    if (!m || !t || t->ndim != 1 || idx >= t->dim[0]) return fallback;
    if (t->type == Q36_TENSOR_F32) {
        const float *src = (const float *)(m->map + t->abs_offset);
        return src[idx];
    }
    if (t->type == Q36_TENSOR_F16) {
        const uint16_t *src = (const uint16_t *)(m->map + t->abs_offset);
        return q36_f16_to_f32(src[idx]);
    }
    return fallback;
}

static void q36_scale_inplace(float *x, uint32_t n, float scale) {
    if (!x || scale == 1.0f) return;
    for (uint32_t i = 0; i < n; i++) x[i] *= scale;
}

static bool q36_tensor_type_supports_q8k_dot(uint32_t type) {
    switch (type) {
    case Q36_TENSOR_Q2_K:
    case Q36_TENSOR_Q4_K:
    case Q36_TENSOR_Q5_K:
    case Q36_TENSOR_Q6_K:
    case Q36_TENSOR_IQ2_XXS:
    case Q36_TENSOR_IQ2_S:
    case Q36_TENSOR_IQ3_S:
        return true;
    default:
        return false;
    }
}

static bool q36_tensor_type_supports_q8_0_dot(uint32_t type) {
    return type == Q36_TENSOR_Q8_0;
}

static bool q36_tensor_type_supports_prequant_dot(uint32_t type) {
    return q36_tensor_type_supports_q8k_dot(type) || q36_tensor_type_supports_q8_0_dot(type);
}

static bool q36_tensor_pair_can_fuse(const q36_tensor *t0,
                                     const q36_tensor *t1,
                                     uint32_t ndim,
                                     uint32_t in_dim,
                                     uint32_t out_dim) {
    if (!t0 || !t1 || t0->type != t1->type || t0->ndim != ndim || t1->ndim != ndim) return false;
    if (t0->dim[0] != in_dim || t1->dim[0] != in_dim || t0->dim[1] != out_dim || t1->dim[1] != out_dim) return false;
    if (ndim == 3 && t0->dim[2] != t1->dim[2]) return false;
    return q36_tensor_type_supports_prequant_dot(t0->type);
}

static bool q36_tensor_row_dot_q8k(uint32_t type, const uint8_t *row, const uint8_t *xq, uint32_t n, float *out) {
    const q36_block_q8_k *q = (const q36_block_q8_k *)xq;
    float weights[Q36_QK_K];
    double acc = 0.0;
    uint32_t block_bytes;
    if (!row || !xq || !out || (n % Q36_QK_K) != 0) return false;
    if (type >= sizeof(gguf_types) / sizeof(gguf_types[0]) || !gguf_types[type].block_bytes) return false;
    block_bytes = gguf_types[type].block_bytes;
    for (uint32_t block = 0; block < n / Q36_QK_K; block++) {
        if (!q36_dequantize_row_from_ptr(type, row + (uint64_t)block * block_bytes,
                                         weights, Q36_QK_K)) return false;
        for (uint32_t i = 0; i < Q36_QK_K; i++)
            acc += (double)weights[i] * (double)(q[block].d * (float)q[block].qs[i]);
    }
    *out = (float)acc;
    return true;
}

static bool q36_tensor_row_dot_q8k_pair(uint32_t type,
                                        const uint8_t *row0,
                                        const uint8_t *row1,
                                        const uint8_t *xq,
                                        uint32_t n,
                                        float *out0,
                                        float *out1) {
    if (!row0 || !row1 || !xq || !out0 || !out1 || (n % Q36_QK_K) != 0) return false;
    return q36_tensor_row_dot_q8k(type, row0, xq, n, out0) &&
           q36_tensor_row_dot_q8k(type, row1, xq, n, out1);
}

static bool q36_tensor_row_dot_prequant(uint32_t type, const uint8_t *row, const uint8_t *xq, uint32_t n, float *out) {
    float acc = 0.0f;
    if (type == Q36_TENSOR_Q8_0) {
        if (!row || !xq || !out || (n % Q36_QK8_0) != 0) return false;
        acc = q36_quant_dot_q8_0(row, xq, (int)n);
        *out = acc;
        return true;
    }
    return q36_tensor_row_dot_q8k(type, row, xq, n, out);
}

static bool q36_tensor_row_dot_prequant_pair(uint32_t type,
                                             const uint8_t *row0,
                                             const uint8_t *row1,
                                             const uint8_t *xq,
                                             uint32_t n,
                                             float *out0,
                                             float *out1) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    if (type == Q36_TENSOR_Q8_0) {
        if (!row0 || !row1 || !xq || !out0 || !out1 || (n % Q36_QK8_0) != 0) return false;
        acc0 = q36_quant_dot_q8_0(row0, xq, (int)n);
        acc1 = q36_quant_dot_q8_0(row1, xq, (int)n);
        *out0 = acc0;
        *out1 = acc1;
        return true;
    }
    return q36_tensor_row_dot_q8k_pair(type, row0, row1, xq, n, out0, out1);
}

typedef enum {
    Q36_ACTIVATION_QUANT_NONE,
    Q36_ACTIVATION_QUANT_Q8_K,
    Q36_ACTIVATION_QUANT_Q8_0,
} q36_activation_quant_kind;

static q36_activation_quant_kind q36_activation_quant_kind_for_type(uint32_t type) {
    if (q36_tensor_type_supports_q8_0_dot(type)) return Q36_ACTIVATION_QUANT_Q8_0;
    if (q36_tensor_type_supports_q8k_dot(type)) return Q36_ACTIVATION_QUANT_Q8_K;
    return Q36_ACTIVATION_QUANT_NONE;
}

static bool q36_activation_quant_valid_dim(q36_activation_quant_kind kind, uint32_t n) {
    switch (kind) {
    case Q36_ACTIVATION_QUANT_NONE:
        return true;
    case Q36_ACTIVATION_QUANT_Q8_K:
        return (n % Q36_QK_K) == 0;
    case Q36_ACTIVATION_QUANT_Q8_0:
        return (n % Q36_QK8_0) == 0;
    default:
        return false;
    }
}

static uint32_t q36_activation_quant_row_bytes(q36_activation_quant_kind kind, uint32_t n) {
    switch (kind) {
    case Q36_ACTIVATION_QUANT_Q8_K:
        return q36_q8k_row_bytes(n);
    case Q36_ACTIVATION_QUANT_Q8_0:
        return q36_q8_0_row_bytes(n);
    default:
        return 0;
    }
}

static bool q36_quantize_activation_row(q36_activation_quant_kind kind, const float *x, uint8_t *xq, uint32_t n) {
    if (kind == Q36_ACTIVATION_QUANT_NONE) return true;
    if (!x || !xq || !q36_activation_quant_valid_dim(kind, n)) return false;
    if (kind == Q36_ACTIVATION_QUANT_Q8_0) q36_quant_q8_0(x, xq, (int64_t)n);
    else q36_quant_q8_k(x, xq, (int64_t)n);
    return true;
}

typedef struct {
    const float *x;
    uint8_t *xq;
    q36_activation_quant_kind kind;
    uint32_t in_dim;
    uint32_t row_bytes;
} q36_quantize_activation_batch_ctx;

typedef struct {
    float *out;
    const q36_engine *e;
    const q36_tensor *t;
    const float *x;
    const uint8_t *xq;
    uint32_t n_tok;
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t xq_row_bytes;
    float scale;
} q36_matmul_batch_ctx;

typedef struct {
    float *out0;
    float *out1;
    const q36_engine *e;
    const q36_tensor *t0;
    const q36_tensor *t1;
    const float *x;
    const uint8_t *xq;
    uint32_t n_tok;
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t xq_row_bytes;
    float scale0;
    float scale1;
} q36_matmul_pair_batch_ctx;

typedef struct {
    float *out;
    const q36_engine *e;
    const q36_tensor *t;
    uint32_t expert;
    const float *x;
    const uint8_t *xq;
    uint32_t n_tok;
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t xq_row_bytes;
    float scale;
} q36_expert_matmul_batch_ctx;

typedef struct {
    float *out0;
    float *out1;
    const q36_engine *e;
    const q36_tensor *t0;
    const q36_tensor *t1;
    uint32_t expert;
    const float *x;
    const uint8_t *xq;
    uint32_t n_tok;
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t xq_row_bytes;
    float scale0;
    float scale1;
} q36_expert_matmul_pair_batch_ctx;

typedef struct {
    float *out;
    const float *w;
    const float *x;
    uint32_t n_tok;
    uint32_t n;
} q36_vector_dot_batch_ctx;

typedef struct {
    float *x;
    uint32_t rows;
    uint32_t cols;
    float scale;
} q36_scale_rows_ctx;

typedef struct {
    float *dst;
    const float *a;
    const float *b;
    uint32_t rows;
    uint32_t cols;
} q36_add_rows_ctx;

typedef struct {
    float *dst;
    const float *src;
    const float *weight;
    uint32_t rows;
    uint32_t cols;
    float eps;
} q36_rms_norm_rows_ctx;

typedef struct {
    float *out;
    const float *gate;
    const float *up;
    uint32_t rows;
    uint32_t cols;
} q36_swiglu_rows_ctx;

static void q36_quantize_activation_batch_worker(void *opaque, uint64_t row0, uint64_t row1) {
    q36_quantize_activation_batch_ctx *ctx = (q36_quantize_activation_batch_ctx *)opaque;
    for (uint64_t t = row0; t < row1; t++) {
        q36_quantize_activation_row(ctx->kind,
                                    ctx->x + t * ctx->in_dim,
                                    ctx->xq + t * ctx->row_bytes,
                                    ctx->in_dim);
    }
}

static bool q36_quantize_activation_batch(q36_activation_quant_kind kind,
                                          const float *x,
                                          uint8_t *xq,
                                          uint32_t n_tok,
                                          uint32_t in_dim,
                                          uint32_t n_threads) {
    q36_quantize_activation_batch_ctx ctx;
    if (kind == Q36_ACTIVATION_QUANT_NONE) return true;
    if (!x || !xq || n_tok == 0 || !q36_activation_quant_valid_dim(kind, in_dim)) return false;
    ctx.x = x;
    ctx.xq = xq;
    ctx.kind = kind;
    ctx.in_dim = in_dim;
    ctx.row_bytes = q36_activation_quant_row_bytes(kind, in_dim);
    q36_parallel_for_rows(n_tok, 2, n_threads, q36_quantize_activation_batch_worker, &ctx);
    return true;
}

static bool q36_quantize_activation_batch_for_type(uint32_t type,
                                                   const float *x,
                                                   uint8_t *xq,
                                                   uint32_t n_tok,
                                                   uint32_t in_dim,
                                                   uint32_t n_threads) {
    return q36_quantize_activation_batch(q36_activation_quant_kind_for_type(type), x, xq, n_tok, in_dim, n_threads);
}

static void q36_matmul_batch_worker(void *opaque, uint64_t row0, uint64_t row1) {
    q36_matmul_batch_ctx *ctx = (q36_matmul_batch_ctx *)opaque;
    const q36_model *m = &ctx->e->model;
    for (uint64_t row = row0; row < row1; row++) {
        switch (ctx->t->type) {
        case Q36_TENSOR_F32: {
            const float *wr = (const float *)(m->map + ctx->t->abs_offset) + row * ctx->in_dim;
            for (uint32_t t = 0; t < ctx->n_tok; t++) {
                const float *xr = ctx->x + (uint64_t)t * ctx->in_dim;
                double acc = 0.0;
                for (uint32_t i = 0; i < ctx->in_dim; i++) acc += (double)wr[i] * (double)xr[i];
                ctx->out[(uint64_t)t * ctx->out_dim + row] = (float)acc * ctx->scale;
            }
            break;
        }
        case Q36_TENSOR_F16: {
            const uint16_t *wr = (const uint16_t *)(m->map + ctx->t->abs_offset) + row * ctx->in_dim;
            for (uint32_t t = 0; t < ctx->n_tok; t++) {
                const float *xr = ctx->x + (uint64_t)t * ctx->in_dim;
                double acc = 0.0;
                for (uint32_t i = 0; i < ctx->in_dim; i++) acc += (double)q36_f16_to_f32(wr[i]) * (double)xr[i];
                ctx->out[(uint64_t)t * ctx->out_dim + row] = (float)acc * ctx->scale;
            }
            break;
        }
        default: {
            const uint8_t *src;
            if (!ctx->xq || !q36_tensor_row_ptr(m, ctx->t, row, &src, NULL)) continue;
            for (uint32_t t = 0; t < ctx->n_tok; t++) {
                float v = 0.0f;
                if (!q36_tensor_row_dot_prequant(ctx->t->type, src,
                                                 ctx->xq + (uint64_t)t * ctx->xq_row_bytes,
                                                 ctx->in_dim, &v)) continue;
                ctx->out[(uint64_t)t * ctx->out_dim + row] = v * ctx->scale;
            }
            break;
        }
        }
    }
}

static void q36_matmul_pair_batch_worker(void *opaque, uint64_t row0, uint64_t row1) {
    q36_matmul_pair_batch_ctx *ctx = (q36_matmul_pair_batch_ctx *)opaque;
    const q36_model *m = &ctx->e->model;
    for (uint64_t row = row0; row < row1; row++) {
        const uint8_t *src0;
        const uint8_t *src1;
        if (!ctx->xq ||
            !q36_tensor_row_ptr(m, ctx->t0, row, &src0, NULL) ||
            !q36_tensor_row_ptr(m, ctx->t1, row, &src1, NULL)) {
            continue;
        }
        for (uint32_t t = 0; t < ctx->n_tok; t++) {
            float v0 = 0.0f;
            float v1 = 0.0f;
            if (!q36_tensor_row_dot_prequant_pair(ctx->t0->type, src0, src1,
                                                  ctx->xq + (uint64_t)t * ctx->xq_row_bytes,
                                                  ctx->in_dim, &v0, &v1)) {
                continue;
            }
            ctx->out0[(uint64_t)t * ctx->out_dim + row] = v0 * ctx->scale0;
            ctx->out1[(uint64_t)t * ctx->out_dim + row] = v1 * ctx->scale1;
        }
    }
}

static void q36_expert_matmul_batch_worker(void *opaque, uint64_t row0, uint64_t row1) {
    q36_expert_matmul_batch_ctx *ctx = (q36_expert_matmul_batch_ctx *)opaque;
    const q36_model *m = &ctx->e->model;
    for (uint64_t row = row0; row < row1; row++) {
        switch (ctx->t->type) {
        case Q36_TENSOR_F32: {
            uint64_t flat = (uint64_t)ctx->expert * ctx->out_dim + row;
            const float *wr = (const float *)(m->map + ctx->t->abs_offset) + flat * ctx->in_dim;
            for (uint32_t t = 0; t < ctx->n_tok; t++) {
                const float *xr = ctx->x + (uint64_t)t * ctx->in_dim;
                double acc = 0.0;
                for (uint32_t i = 0; i < ctx->in_dim; i++) acc += (double)wr[i] * (double)xr[i];
                ctx->out[(uint64_t)t * ctx->out_dim + row] = (float)acc * ctx->scale;
            }
            break;
        }
        case Q36_TENSOR_F16: {
            uint64_t flat = (uint64_t)ctx->expert * ctx->out_dim + row;
            const uint16_t *wr = (const uint16_t *)(m->map + ctx->t->abs_offset) + flat * ctx->in_dim;
            for (uint32_t t = 0; t < ctx->n_tok; t++) {
                const float *xr = ctx->x + (uint64_t)t * ctx->in_dim;
                double acc = 0.0;
                for (uint32_t i = 0; i < ctx->in_dim; i++) acc += (double)q36_f16_to_f32(wr[i]) * (double)xr[i];
                ctx->out[(uint64_t)t * ctx->out_dim + row] = (float)acc * ctx->scale;
            }
            break;
        }
        default: {
            const uint8_t *src;
            if (!ctx->xq || !q36_tensor_expert_row_ptr(m, ctx->t, ctx->expert, (uint32_t)row, &src, NULL)) continue;
            for (uint32_t t = 0; t < ctx->n_tok; t++) {
                float v = 0.0f;
                if (!q36_tensor_row_dot_prequant(ctx->t->type, src,
                                                 ctx->xq + (uint64_t)t * ctx->xq_row_bytes,
                                                 ctx->in_dim, &v)) continue;
                ctx->out[(uint64_t)t * ctx->out_dim + row] = v * ctx->scale;
            }
            break;
        }
        }
    }
}

static void q36_expert_matmul_pair_batch_worker(void *opaque, uint64_t row0, uint64_t row1) {
    q36_expert_matmul_pair_batch_ctx *ctx = (q36_expert_matmul_pair_batch_ctx *)opaque;
    const q36_model *m = &ctx->e->model;
    for (uint64_t row = row0; row < row1; row++) {
        const uint8_t *src0;
        const uint8_t *src1;
        if (!ctx->xq ||
            !q36_tensor_expert_row_ptr(m, ctx->t0, ctx->expert, (uint32_t)row, &src0, NULL) ||
            !q36_tensor_expert_row_ptr(m, ctx->t1, ctx->expert, (uint32_t)row, &src1, NULL)) {
            continue;
        }
        for (uint32_t t = 0; t < ctx->n_tok; t++) {
            float v0 = 0.0f;
            float v1 = 0.0f;
            if (!q36_tensor_row_dot_prequant_pair(ctx->t0->type, src0, src1,
                                                  ctx->xq + (uint64_t)t * ctx->xq_row_bytes,
                                                  ctx->in_dim, &v0, &v1)) {
                continue;
            }
            ctx->out0[(uint64_t)t * ctx->out_dim + row] = v0 * ctx->scale0;
            ctx->out1[(uint64_t)t * ctx->out_dim + row] = v1 * ctx->scale1;
        }
    }
}

static void q36_vector_dot_batch_worker(void *opaque, uint64_t row0, uint64_t row1) {
    q36_vector_dot_batch_ctx *ctx = (q36_vector_dot_batch_ctx *)opaque;
    for (uint64_t row = row0; row < row1; row++) {
        const float *xr = ctx->x + row * ctx->n;
        double acc = 0.0;
        for (uint32_t i = 0; i < ctx->n; i++) acc += (double)ctx->w[i] * (double)xr[i];
        ctx->out[row] = (float)acc;
    }
}

static void q36_scale_rows_worker(void *opaque, uint64_t row0, uint64_t row1) {
    q36_scale_rows_ctx *ctx = (q36_scale_rows_ctx *)opaque;
    for (uint64_t row = row0; row < row1; row++) {
        float *xr = ctx->x + row * ctx->cols;
        for (uint32_t i = 0; i < ctx->cols; i++) xr[i] *= ctx->scale;
    }
}

static void q36_add_rows_worker(void *opaque, uint64_t row0, uint64_t row1) {
    q36_add_rows_ctx *ctx = (q36_add_rows_ctx *)opaque;
    for (uint64_t row = row0; row < row1; row++) {
        float *dr = ctx->dst + row * ctx->cols;
        const float *ar = ctx->a + row * ctx->cols;
        const float *br = ctx->b + row * ctx->cols;
        for (uint32_t i = 0; i < ctx->cols; i++) dr[i] = ar[i] + br[i];
    }
}

static void q36_rms_norm_rows_worker(void *opaque, uint64_t row0, uint64_t row1) {
    q36_rms_norm_rows_ctx *ctx = (q36_rms_norm_rows_ctx *)opaque;
    for (uint64_t row = row0; row < row1; row++) {
        q36_ref_rms_norm(ctx->dst + row * ctx->cols,
                         ctx->src + row * ctx->cols,
                         ctx->weight,
                         ctx->cols,
                         ctx->eps);
    }
}

static void q36_swiglu_rows_worker(void *opaque, uint64_t row0, uint64_t row1) {
    q36_swiglu_rows_ctx *ctx = (q36_swiglu_rows_ctx *)opaque;
    for (uint64_t row = row0; row < row1; row++) {
        float *orow = ctx->out + row * ctx->cols;
        const float *grow = ctx->gate + row * ctx->cols;
        const float *urow = ctx->up + row * ctx->cols;
        for (uint32_t i = 0; i < ctx->cols; i++)
            orow[i] = grow[i] / (1.0f + expf(-grow[i])) * urow[i];
    }
}

static bool q36_tensor_matmul_batch_prequant(const q36_engine *e,
                                             const q36_tensor *t,
                                             const float *x,
                                             const uint8_t *xq,
                                             float *out,
                                             uint32_t n_tok,
                                             uint32_t in_dim,
                                             uint32_t out_dim,
                                             float scale) {
    q36_matmul_batch_ctx ctx;
    q36_activation_quant_kind kind;
    uint64_t ops;
    uint64_t min_rows;
    if (!e || !t || !x || !out || t->ndim != 2 || t->dim[0] != in_dim || t->dim[1] != out_dim) return false;
    kind = q36_activation_quant_kind_for_type(t->type);
    if (kind != Q36_ACTIVATION_QUANT_NONE && (!xq || !q36_activation_quant_valid_dim(kind, in_dim))) return false;
    ctx.out = out;
    ctx.e = e;
    ctx.t = t;
    ctx.x = x;
    ctx.xq = xq;
    ctx.n_tok = n_tok;
    ctx.in_dim = in_dim;
    ctx.out_dim = out_dim;
    ctx.xq_row_bytes = q36_activation_quant_row_bytes(kind, in_dim);
    ctx.scale = scale;
    ops = (uint64_t)n_tok * in_dim * out_dim;
    min_rows = ops >= 262144u ? 1u : 64u;
    q36_parallel_for_rows(out_dim, min_rows, e->n_threads, q36_matmul_batch_worker, &ctx);
    return true;
}

static bool q36_tensor_matmul_pair_batch_prequant(const q36_engine *e,
                                                  const q36_tensor *t0,
                                                  const q36_tensor *t1,
                                                  const float *x,
                                                  const uint8_t *xq,
                                                  float *out0,
                                                  float *out1,
                                                  uint32_t n_tok,
                                                  uint32_t in_dim,
                                                  uint32_t out_dim,
                                                  float scale0,
                                                  float scale1) {
    q36_matmul_pair_batch_ctx ctx;
    q36_activation_quant_kind kind;
    uint64_t ops;
    uint64_t min_rows;
    if (!e || !t0 || !t1 || !x || !out0 || !out1) return false;
    if (!q36_tensor_pair_can_fuse(t0, t1, 2, in_dim, out_dim)) return false;
    kind = q36_activation_quant_kind_for_type(t0->type);
    if (kind != Q36_ACTIVATION_QUANT_NONE && (!xq || !q36_activation_quant_valid_dim(kind, in_dim))) return false;
    ctx.out0 = out0;
    ctx.out1 = out1;
    ctx.e = e;
    ctx.t0 = t0;
    ctx.t1 = t1;
    ctx.x = x;
    ctx.xq = xq;
    ctx.n_tok = n_tok;
    ctx.in_dim = in_dim;
    ctx.out_dim = out_dim;
    ctx.xq_row_bytes = q36_activation_quant_row_bytes(kind, in_dim);
    ctx.scale0 = scale0;
    ctx.scale1 = scale1;
    ops = (uint64_t)n_tok * in_dim * out_dim * 2u;
    min_rows = ops >= 262144u ? 1u : 64u;
    q36_parallel_for_rows(out_dim, min_rows, e->n_threads, q36_matmul_pair_batch_worker, &ctx);
    return true;
}

static bool q36_tensor_expert_matmul_batch_prequant(const q36_engine *e,
                                                    const q36_tensor *t,
                                                    uint32_t expert,
                                                    const float *x,
                                                    const uint8_t *xq,
                                                    float *out,
                                                    uint32_t n_tok,
                                                    uint32_t in_dim,
                                                    uint32_t out_dim,
                                                    float scale) {
    q36_expert_matmul_batch_ctx ctx;
    q36_activation_quant_kind kind;
    uint64_t ops;
    uint64_t min_rows;
    if (!e || !t || !x || !out || t->ndim != 3 || t->dim[0] != in_dim || t->dim[1] != out_dim) return false;
    kind = q36_activation_quant_kind_for_type(t->type);
    if (kind != Q36_ACTIVATION_QUANT_NONE && (!xq || !q36_activation_quant_valid_dim(kind, in_dim))) return false;
    ctx.out = out;
    ctx.e = e;
    ctx.t = t;
    ctx.expert = expert;
    ctx.x = x;
    ctx.xq = xq;
    ctx.n_tok = n_tok;
    ctx.in_dim = in_dim;
    ctx.out_dim = out_dim;
    ctx.xq_row_bytes = q36_activation_quant_row_bytes(kind, in_dim);
    ctx.scale = scale;
    ops = (uint64_t)n_tok * in_dim * out_dim;
    min_rows = ops >= 262144u ? 1u : 64u;
    q36_parallel_for_rows(out_dim, min_rows, e->n_threads, q36_expert_matmul_batch_worker, &ctx);
    return true;
}

static bool q36_tensor_expert_matmul_pair_batch_prequant(const q36_engine *e,
                                                         const q36_tensor *t0,
                                                         const q36_tensor *t1,
                                                         uint32_t expert,
                                                         const float *x,
                                                         const uint8_t *xq,
                                                         float *out0,
                                                         float *out1,
                                                         uint32_t n_tok,
                                                         uint32_t in_dim,
                                                         uint32_t out_dim,
                                                         float scale0,
                                                         float scale1) {
    q36_expert_matmul_pair_batch_ctx ctx;
    q36_activation_quant_kind kind;
    uint64_t ops;
    uint64_t min_rows;
    if (!e || !t0 || !t1 || !x || !out0 || !out1) return false;
    if (!q36_tensor_pair_can_fuse(t0, t1, 3, in_dim, out_dim)) return false;
    kind = q36_activation_quant_kind_for_type(t0->type);
    if (kind != Q36_ACTIVATION_QUANT_NONE && (!xq || !q36_activation_quant_valid_dim(kind, in_dim))) return false;
    ctx.out0 = out0;
    ctx.out1 = out1;
    ctx.e = e;
    ctx.t0 = t0;
    ctx.t1 = t1;
    ctx.expert = expert;
    ctx.x = x;
    ctx.xq = xq;
    ctx.n_tok = n_tok;
    ctx.in_dim = in_dim;
    ctx.out_dim = out_dim;
    ctx.xq_row_bytes = q36_activation_quant_row_bytes(kind, in_dim);
    ctx.scale0 = scale0;
    ctx.scale1 = scale1;
    ops = (uint64_t)n_tok * in_dim * out_dim * 2u;
    min_rows = ops >= 262144u ? 1u : 64u;
    q36_parallel_for_rows(out_dim, min_rows, e->n_threads, q36_expert_matmul_pair_batch_worker, &ctx);
    return true;
}

static void q36_tensor_vector_dot_batch(float *out,
                                        const float *w,
                                        const float *x,
                                        uint32_t n_tok,
                                        uint32_t n,
                                        uint32_t n_threads) {
    q36_vector_dot_batch_ctx ctx;
    if (!out || !w || !x || n_tok == 0) return;
    ctx.out = out;
    ctx.w = w;
    ctx.x = x;
    ctx.n_tok = n_tok;
    ctx.n = n;
    q36_parallel_for_rows(n_tok, 2, n_threads, q36_vector_dot_batch_worker, &ctx);
}

static Q36_MAYBE_UNUSED void q36_scale_rows(float *x, uint32_t rows, uint32_t cols, float scale, uint32_t n_threads) {
    q36_scale_rows_ctx ctx;
    if (!x || scale == 1.0f || rows == 0) return;
    ctx.x = x;
    ctx.rows = rows;
    ctx.cols = cols;
    ctx.scale = scale;
    q36_parallel_for_rows(rows, (uint64_t)rows * cols >= 4096u ? 1u : 64u,
                          n_threads, q36_scale_rows_worker, &ctx);
}

static void q36_add_rows(float *dst,
                         const float *a,
                         const float *b,
                         uint32_t rows,
                         uint32_t cols,
                         uint32_t n_threads) {
    q36_add_rows_ctx ctx;
    if (!dst || !a || !b || rows == 0) return;
    ctx.dst = dst;
    ctx.a = a;
    ctx.b = b;
    ctx.rows = rows;
    ctx.cols = cols;
    q36_parallel_for_rows(rows, (uint64_t)rows * cols >= 4096u ? 1u : 64u,
                          n_threads, q36_add_rows_worker, &ctx);
}

static void q36_directional_steering_project_rows(float *x,
                                                   const float *dirs,
                                                   uint32_t layer,
                                                   uint32_t rows,
                                                   float scale) {
    if (!x || !dirs || rows == 0 || scale == 0.0f) return;
    const float *dir = dirs + (uint64_t)layer * Q36_N_EMBD;
    for (uint32_t row = 0; row < rows; row++) {
        float *xr = x + (uint64_t)row * Q36_N_EMBD;
        float dot = 0.0f;
        for (uint32_t i = 0; i < Q36_N_EMBD; i++) dot += xr[i] * dir[i];
        float coeff = scale * dot;
        for (uint32_t i = 0; i < Q36_N_EMBD; i++) xr[i] -= coeff * dir[i];
    }
}

static void q36_rms_norm_rows(float *dst,
                              const float *src,
                              const float *weight,
                              uint32_t rows,
                              uint32_t cols,
                              float eps,
                              uint32_t n_threads) {
    q36_rms_norm_rows_ctx ctx;
    if (!dst || !src || rows == 0) return;
    ctx.dst = dst;
    ctx.src = src;
    ctx.weight = weight;
    ctx.rows = rows;
    ctx.cols = cols;
    ctx.eps = eps;
    q36_parallel_for_rows(rows, (uint64_t)rows * cols >= 4096u ? 1u : 64u,
                          n_threads, q36_rms_norm_rows_worker, &ctx);
}

static void q36_swiglu_rows(float *out,
                            const float *gate,
                            const float *up,
                            uint32_t rows,
                            uint32_t cols,
                            uint32_t n_threads) {
    q36_swiglu_rows_ctx ctx;
    if (!out || !gate || !up || rows == 0) return;
    ctx.out = out;
    ctx.gate = gate;
    ctx.up = up;
    ctx.rows = rows;
    ctx.cols = cols;
    q36_parallel_for_rows(rows, (uint64_t)rows * cols >= 4096u ? 1u : 64u,
                          n_threads, q36_swiglu_rows_worker, &ctx);
}

static bool q36_tensor_matvec_prequant(const q36_engine *e,
                                       const q36_tensor *t,
                                       const float *x,
                                       const uint8_t *xq,
                                       float *out,
                                       float *rowbuf,
                                       uint32_t in_dim,
                                       uint32_t out_dim) {
    (void)rowbuf;
    return q36_tensor_matmul_batch_prequant(e, t, x, xq, out, 1, in_dim, out_dim, 1.0f);
}

static bool q36_tensor_matvec_pair_prequant(const q36_engine *e,
                                            const q36_tensor *t0,
                                            const q36_tensor *t1,
                                            const float *x,
                                            const uint8_t *xq,
                                            float *out0,
                                            float *out1,
                                            float *rowbuf,
                                            uint32_t in_dim,
                                            uint32_t out_dim,
                                            float scale0,
                                            float scale1) {
    (void)rowbuf;
    return q36_tensor_matmul_pair_batch_prequant(e, t0, t1, x, xq, out0, out1, 1, in_dim, out_dim, scale0, scale1);
}

static bool q36_tensor_matvec(const q36_engine *e, const q36_tensor *t, const float *x, float *out,
                              float *rowbuf, uint32_t in_dim, uint32_t out_dim) {
    uint8_t xq[Q36_MAX_Q8_K_BYTES];
    const uint8_t *xq_src = NULL;
    q36_activation_quant_kind kind;
    if (!e || !t || !x || !out) return false;
    kind = q36_activation_quant_kind_for_type(t->type);
    if (kind != Q36_ACTIVATION_QUANT_NONE &&
        in_dim <= Q36_N_SSM_INNER &&
        q36_activation_quant_valid_dim(kind, in_dim) &&
        q36_activation_quant_row_bytes(kind, in_dim) <= Q36_MAX_Q8_K_BYTES &&
        q36_quantize_activation_row(kind, x, xq, in_dim)) {
        xq_src = xq;
    }
    return q36_tensor_matvec_prequant(e, t, x, xq_src, out, rowbuf, in_dim, out_dim);
}

static bool q36_tensor_expert_matvec_prequant(const q36_engine *e,
                                              const q36_tensor *t,
                                              uint32_t expert,
                                              const float *x,
                                              const uint8_t *xq,
                                              float *out,
                                              float *rowbuf,
                                              uint32_t in_dim,
                                              uint32_t out_dim) {
    (void)rowbuf;
    return q36_tensor_expert_matmul_batch_prequant(e, t, expert, x, xq, out, 1, in_dim, out_dim, 1.0f);
}

static bool q36_tensor_expert_matvec_pair_prequant(const q36_engine *e,
                                                   const q36_tensor *t0,
                                                   const q36_tensor *t1,
                                                   uint32_t expert,
                                                   const float *x,
                                                   const uint8_t *xq,
                                                   float *out0,
                                                   float *out1,
                                                   float *rowbuf,
                                                   uint32_t in_dim,
                                                   uint32_t out_dim,
                                                   float scale0,
                                                   float scale1) {
    (void)rowbuf;
    return q36_tensor_expert_matmul_pair_batch_prequant(e, t0, t1, expert,
                                                         x, xq, out0, out1,
                                                         1, in_dim, out_dim,
                                                         scale0, scale1);
}

static bool q36_tensor_expert_matvec(const q36_engine *e, const q36_tensor *t, uint32_t expert,
                                     const float *x, float *out, float *rowbuf,
                                     uint32_t in_dim, uint32_t out_dim) {
    uint8_t xq[Q36_MAX_Q8_K_BYTES];
    const uint8_t *xq_src = NULL;
    q36_activation_quant_kind kind;
    if (!e || !t || !x || !out) return false;
    kind = q36_activation_quant_kind_for_type(t->type);
    if (kind != Q36_ACTIVATION_QUANT_NONE &&
        in_dim <= Q36_N_SSM_INNER &&
        q36_activation_quant_valid_dim(kind, in_dim) &&
        q36_activation_quant_row_bytes(kind, in_dim) <= Q36_MAX_Q8_K_BYTES &&
        q36_quantize_activation_row(kind, x, xq, in_dim)) {
        xq_src = xq;
    }
    return q36_tensor_expert_matvec_prequant(e, t, expert, x, xq_src, out, rowbuf, in_dim, out_dim);
}

static void cursor_error(q36_cursor *c, const char *msg) {
    if (c->error[0] == '\0') {
        snprintf(c->error, sizeof(c->error), "%s at byte %" PRIu64, msg, c->pos);
    }
}

static bool cursor_has(q36_cursor *c, uint64_t n) {
    if (n > c->size || c->pos > c->size - n) {
        cursor_error(c, "truncated GGUF file");
        return false;
    }
    return true;
}

static bool cursor_read(q36_cursor *c, void *dst, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    memcpy(dst, c->base + c->pos, (size_t)n);
    c->pos += n;
    return true;
}

static bool cursor_skip(q36_cursor *c, uint64_t n) {
    if (!cursor_has(c, n)) return false;
    c->pos += n;
    return true;
}

static bool cursor_u32(q36_cursor *c, uint32_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_u64(q36_cursor *c, uint64_t *v) {
    return cursor_read(c, v, sizeof(*v));
}

static bool cursor_string(q36_cursor *c, q36_str *s) {
    uint64_t len = 0;
    if (!cursor_u64(c, &len)) return false;
    if (!cursor_has(c, len)) return false;
    s->ptr = (const char *)(c->base + c->pos);
    s->len = len;
    c->pos += len;
    return true;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    uint64_t rem = value % alignment;
    return rem == 0 ? value : value + alignment - rem;
}

static uint64_t scalar_value_size(uint32_t type) {
    switch (type) {
    case GGUF_VALUE_UINT8:
    case GGUF_VALUE_INT8:
    case GGUF_VALUE_BOOL:
        return 1;
    case GGUF_VALUE_UINT16:
    case GGUF_VALUE_INT16:
        return 2;
    case GGUF_VALUE_UINT32:
    case GGUF_VALUE_INT32:
    case GGUF_VALUE_FLOAT32:
        return 4;
    case GGUF_VALUE_UINT64:
    case GGUF_VALUE_INT64:
    case GGUF_VALUE_FLOAT64:
        return 8;
    default:
        return 0;
    }
}

static bool skip_value(q36_cursor *c, uint32_t type, int depth) {
    if (depth > 8) {
        cursor_error(c, "metadata array nesting is too deep");
        return false;
    }

    {
        uint64_t scalar = scalar_value_size(type);
        if (scalar != 0) return cursor_skip(c, scalar);
    }

    if (type == GGUF_VALUE_STRING) {
        q36_str ignored;
        return cursor_string(c, &ignored);
    }

    if (type == GGUF_VALUE_ARRAY) {
        uint32_t item_type = 0;
        uint64_t len = 0;
        uint64_t item_size = 0;
        if (!cursor_u32(c, &item_type)) return false;
        if (!cursor_u64(c, &len)) return false;
        item_size = scalar_value_size(item_type);
        if (item_size != 0) {
            if (len > UINT64_MAX / item_size) {
                cursor_error(c, "metadata array is too large");
                return false;
            }
            return cursor_skip(c, len * item_size);
        }
        for (uint64_t i = 0; i < len; i++) {
            if (!skip_value(c, item_type, depth + 1)) return false;
        }
        return true;
    }

    cursor_error(c, "unknown GGUF metadata type");
    return false;
}

static const gguf_type_info *tensor_type(uint32_t type) {
    uint32_t n = (uint32_t)(sizeof(gguf_types) / sizeof(gguf_types[0]));
    if (type >= n || gguf_types[type].name == NULL) return NULL;
    return &gguf_types[type];
}

static const char *tensor_type_name(uint32_t type) {
    const gguf_type_info *info = tensor_type(type);
    return info ? info->name : "unknown";
}

static bool tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes) {
    const gguf_type_info *info = tensor_type(type);
    uint64_t blocks = 0;
    if (!info || info->block_elems == 0) return false;
    blocks = (elements + info->block_elems - 1) / info->block_elems;
    if (blocks > UINT64_MAX / info->block_bytes) return false;
    *bytes = blocks * info->block_bytes;
    return true;
}

static q36_cursor cursor_at(const q36_model *m, uint64_t pos) {
    q36_cursor c;
    c.base = m->map;
    c.size = m->size;
    c.pos = pos;
    c.error[0] = '\0';
    return c;
}

static q36_kv *model_find_kv(const q36_model *m, const char *key) {
    for (uint64_t i = 0; i < m->n_kv; i++) {
        if (q36_streq(m->kv[i].key, key)) return &m->kv[i];
    }
    return NULL;
}

static bool model_get_string(const q36_model *m, const char *key, q36_str *out) {
    q36_kv *kv = model_find_kv(m, key);
    q36_cursor c;
    if (!kv || kv->type != GGUF_VALUE_STRING) return false;
    c = cursor_at(m, kv->value_pos);
    return cursor_string(&c, out);
}

static bool model_get_u32(const q36_model *m, const char *key, uint32_t *out) {
    q36_kv *kv = model_find_kv(m, key);
    q36_cursor c;
    if (!kv || kv->type != GGUF_VALUE_UINT32) return false;
    c = cursor_at(m, kv->value_pos);
    return cursor_u32(&c, out);
}

static bool model_get_u64_compat(const q36_model *m, const char *key, uint64_t *out) {
    q36_kv *kv = model_find_kv(m, key);
    q36_cursor c;
    if (!kv) return false;
    c = cursor_at(m, kv->value_pos);
    if (kv->type == GGUF_VALUE_UINT64) return cursor_u64(&c, out);
    if (kv->type == GGUF_VALUE_UINT32) {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v)) return false;
        *out = v;
        return true;
    }
    return false;
}

static bool model_get_f32_compat(const q36_model *m, const char *key, float *out) {
    q36_kv *kv = model_find_kv(m, key);
    q36_cursor c;
    if (!kv) return false;
    c = cursor_at(m, kv->value_pos);
    if (kv->type == GGUF_VALUE_FLOAT32) return cursor_read(&c, out, sizeof(*out));
    if (kv->type == GGUF_VALUE_FLOAT64) {
        double v = 0.0;
        if (!cursor_read(&c, &v, sizeof(v))) return false;
        *out = (float)v;
        return true;
    }
    if (kv->type == GGUF_VALUE_UINT32) {
        uint32_t v = 0;
        if (!cursor_u32(&c, &v)) return false;
        *out = (float)v;
        return true;
    }
    if (kv->type == GGUF_VALUE_INT32) {
        int32_t v = 0;
        if (!cursor_read(&c, &v, sizeof(v))) return false;
        *out = (float)v;
        return true;
    }
    return false;
}

static bool model_get_bool(const q36_model *m, const char *key, bool *out) {
    q36_kv *kv = model_find_kv(m, key);
    q36_cursor c;
    uint8_t v = 0;
    if (!kv || kv->type != GGUF_VALUE_BOOL) return false;
    c = cursor_at(m, kv->value_pos);
    if (!cursor_read(&c, &v, sizeof(v))) return false;
    *out = v != 0;
    return true;
}

static bool model_get_array(const q36_model *m, const char *key, q36_array_ref *out) {
    q36_kv *kv = model_find_kv(m, key);
    q36_cursor c;
    if (!kv || kv->type != GGUF_VALUE_ARRAY) return false;
    c = cursor_at(m, kv->value_pos);
    if (!cursor_u32(&c, &out->type)) return false;
    if (!cursor_u64(&c, &out->len)) return false;
    out->data_pos = c.pos;
    return true;
}

static void model_close(q36_model *m) {
    if (!m) return;
    free(m->kv);
    free(m->tensors);
    if (m->map) munmap((void *)m->map, (size_t)m->size);
    if (m->fd >= 0) close(m->fd);
    memset(m, 0, sizeof(*m));
    m->fd = -1;
}

static void parse_metadata(q36_model *m, q36_cursor *c) {
    m->kv = xcalloc((size_t)m->n_kv, sizeof(m->kv[0]));
    m->alignment = 32;
    for (uint64_t i = 0; i < m->n_kv; i++) {
        q36_kv *kv = &m->kv[i];
        if (!cursor_string(c, &kv->key)) q36_die(c->error);
        if (!cursor_u32(c, &kv->type)) q36_die(c->error);
        kv->value_pos = c->pos;
        if (q36_streq(kv->key, "general.alignment") && kv->type == GGUF_VALUE_UINT32) {
            q36_cursor tmp = cursor_at(m, kv->value_pos);
            uint32_t align = 0;
            if (cursor_u32(&tmp, &align) && align != 0) m->alignment = align;
        }
        if (!skip_value(c, kv->type, 0)) q36_die(c->error);
    }
}

static void parse_tensors(q36_model *m, q36_cursor *c) {
    m->tensors = xcalloc((size_t)m->n_tensors, sizeof(m->tensors[0]));
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        q36_tensor *t = &m->tensors[i];
        if (!cursor_string(c, &t->name)) q36_die(c->error);
        if (!cursor_u32(c, &t->ndim)) q36_die(c->error);
        if (t->ndim == 0 || t->ndim > Q36_MAX_DIMS) q36_die("tensor has an unsupported number of dimensions");
        t->elements = 1;
        for (uint32_t d = 0; d < t->ndim; d++) {
            if (!cursor_u64(c, &t->dim[d])) q36_die(c->error);
            if (t->dim[d] != 0 && t->elements > UINT64_MAX / t->dim[d]) q36_die("tensor element count overflow");
            t->elements *= t->dim[d];
        }
        if (!cursor_u32(c, &t->type)) q36_die(c->error);
        if (!cursor_u64(c, &t->rel_offset)) q36_die(c->error);
        if (!tensor_nbytes(t->type, t->elements, &t->bytes)) {
            fprintf(stderr, "q36: unsupported tensor type %u for %.*s\n", t->type, (int)t->name.len, t->name.ptr);
            exit(1);
        }
    }
    m->tensor_data_pos = align_up(c->pos, m->alignment);
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        q36_tensor *t = &m->tensors[i];
        if (t->rel_offset > UINT64_MAX - m->tensor_data_pos) q36_die("tensor offset overflow");
        t->abs_offset = m->tensor_data_pos + t->rel_offset;
        if (t->bytes != 0 && (t->abs_offset > m->size || t->bytes > m->size - t->abs_offset)) {
            q36_die("tensor points outside GGUF file");
        }
        if (t->bytes > m->max_tensor_bytes) m->max_tensor_bytes = t->bytes;
    }
}

static void model_open(q36_model *m, const char *path, bool graph_mapping) {
    struct stat st;
    int fd;
    void *map;
    q36_cursor c;
    uint32_t magic = 0;
    memset(m, 0, sizeof(*m));
    m->fd = -1;
    fd = open(path, O_RDONLY);
    if (fd == -1) q36_die_errno("cannot open model", path);
    if (fstat(fd, &st) == -1) q36_die_errno("cannot stat model", path);
    if (st.st_size < 32) q36_die("model file is too small to be GGUF");
    map = mmap(NULL, (size_t)st.st_size, PROT_READ, graph_mapping ? MAP_SHARED : MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) q36_die_errno("cannot mmap model", path);
    m->fd = fd;
    m->map = map;
    m->size = (uint64_t)st.st_size;
    c = cursor_at(m, 0);
    if (!cursor_u32(&c, &magic)) q36_die(c.error);
    if (magic != Q36_GGUF_MAGIC) q36_die("model is not a GGUF file");
    if (!cursor_u32(&c, &m->version)) q36_die(c.error);
    if (!cursor_u64(&c, &m->n_tensors)) q36_die(c.error);
    if (!cursor_u64(&c, &m->n_kv)) q36_die(c.error);
    if (m->version != 3) q36_die("only GGUF v3 is supported");
    parse_metadata(m, &c);
    parse_tensors(m, &c);
}

static q36_tensor *model_find_tensor(const q36_model *m, const char *name) {
    size_t len = strlen(name);
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        if (m->tensors[i].name.len == len && memcmp(m->tensors[i].name.ptr, name, len) == 0) {
            return &m->tensors[i];
        }
    }
    return NULL;
}

static uint32_t required_u32(const q36_model *m, const char *key) {
    uint32_t v = 0;
    if (!model_get_u32(m, key, &v)) {
        fprintf(stderr, "q36: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}

static float required_f32(const q36_model *m, const char *key) {
    float v = 0.0f;
    if (!model_get_f32_compat(m, key, &v)) {
        fprintf(stderr, "q36: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}

static q36_str required_string(const q36_model *m, const char *key) {
    q36_str s = {0};
    if (!model_get_string(m, key, &s)) {
        fprintf(stderr, "q36: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return s;
}

static q36_tensor *required_tensor(const q36_model *m, const char *name) {
    q36_tensor *t = model_find_tensor(m, name);
    if (!t) {
        fprintf(stderr, "q36: required tensor is missing: %s\n", name);
        exit(1);
    }
    return t;
}

static q36_tensor *tensor_by_namef(const q36_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) q36_die("tensor name is too long");
    return model_find_tensor(m, name);
}

static q36_tensor *required_tensorf(const q36_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) q36_die("tensor name is too long");
    return required_tensor(m, name);
}

static void config_expect_u32(const char *name, uint32_t got, uint32_t expected) {
    if (got == expected) return;
    fprintf(stderr, "q36: expected %s=%u, got %u\n", name, expected, got);
    exit(1);
}

static void config_expect_f32(const char *name, float got, float expected) {
    float scale = fabsf(expected) > 1.0f ? fabsf(expected) : 1.0f;
    if (fabsf(got - expected) <= scale * 1.0e-6f) return;
    fprintf(stderr, "q36: expected %s=%.9g, got %.9g\n", name, (double)expected, (double)got);
    exit(1);
}

static void config_expect_string(const char *name, q36_str got, const char *expected) {
    if (q36_streq(got, expected)) return;
    fprintf(stderr, "q36: expected %s=%s\n", name, expected);
    exit(1);
}

static void config_expect_u32_array(const q36_model *m, const char *key, const uint32_t *expected, uint64_t n) {
    q36_array_ref arr;
    q36_cursor c;
    if (!model_get_array(m, key, &arr) || (arr.type != GGUF_VALUE_UINT32 && arr.type != GGUF_VALUE_INT32)) {
        fprintf(stderr, "q36: required metadata array is missing: %s\n", key);
        exit(1);
    }
    if (arr.len != n) {
        fprintf(stderr, "q36: expected %s length=%" PRIu64 ", got %" PRIu64 "\n", key, n, arr.len);
        exit(1);
    }
    c = cursor_at(m, arr.data_pos);
    for (uint64_t i = 0; i < n; i++) {
        uint32_t got = 0;
        if (arr.type == GGUF_VALUE_UINT32) {
            if (!cursor_u32(&c, &got)) q36_die(c.error);
        } else {
            int32_t v = 0;
            if (!cursor_read(&c, &v, sizeof(v))) q36_die(c.error);
            if (v < 0) q36_die("metadata array contains a negative value");
            got = (uint32_t)v;
        }
        if (got != expected[i]) {
            fprintf(stderr, "q36: unexpected %s[%" PRIu64 "]=%u, expected %u\n", key, i, got, expected[i]);
            exit(1);
        }
    }
}

static bool q36_layer_is_full_attention(uint32_t il) {
    return ((il + 1u) % Q36_FULL_ATTENTION_INTERVAL) == 0;
}

static bool tensor_type_is_routed_gate_up(uint32_t type) {
    return type == Q36_TENSOR_Q4_K || type == Q36_TENSOR_IQ2_XXS ||
           type == Q36_TENSOR_IQ2_S || type == Q36_TENSOR_IQ3_S ||
           type == Q36_TENSOR_Q8_0;
}

static bool tensor_type_is_routed_down(uint32_t type) {
    return type == Q36_TENSOR_Q2_K || type == Q36_TENSOR_Q4_K ||
           type == Q36_TENSOR_Q5_K || type == Q36_TENSOR_Q6_K ||
           type == Q36_TENSOR_IQ2_S || type == Q36_TENSOR_IQ3_S ||
           type == Q36_TENSOR_Q8_0;
}

static void tensor_expect_layout(const q36_tensor *t, uint32_t type, uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    uint64_t want[3];
    if (!t) q36_die("internal error: missing tensor while validating layout");
    if (t->type != type) {
        fprintf(stderr, "q36: tensor %.*s has type %s, expected %s\n",
                (int)t->name.len, t->name.ptr, tensor_type_name(t->type), tensor_type_name(type));
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr, "q36: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len, t->name.ptr, t->ndim, ndim);
        exit(1);
    }
    want[0] = d0;
    want[1] = d1;
    want[2] = d2;
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] == want[i]) continue;
        fprintf(stderr, "q36: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                (int)t->name.len, t->name.ptr, i, t->dim[i], want[i]);
        exit(1);
    }
}

static void tensor_expect_plain_layout(const q36_tensor *t, uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    if (!t) q36_die("internal error: missing tensor while validating layout");
    if (t->type != Q36_TENSOR_F32 && t->type != Q36_TENSOR_F16) {
        fprintf(stderr, "q36: tensor %.*s has type %s, expected f32 or f16\n",
                (int)t->name.len, t->name.ptr, tensor_type_name(t->type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}

static void tensor_expect_layout_or_q8_0(const q36_tensor *t, uint32_t type,
                                         uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    if (!t) q36_die("internal error: missing tensor while validating layout");
    if (t->type != type && t->type != Q36_TENSOR_Q8_0) {
        fprintf(stderr, "q36: tensor %.*s has type %s, expected %s or q8_0\n",
                (int)t->name.len, t->name.ptr, tensor_type_name(t->type), tensor_type_name(type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}

static void tensor_expect_layout_or_q6_k_or_q8_0(const q36_tensor *t, uint32_t type,
                                                uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    if (!t) q36_die("internal error: missing tensor while validating layout");
    if (t->type != type && t->type != Q36_TENSOR_Q6_K && t->type != Q36_TENSOR_Q8_0) {
        fprintf(stderr, "q36: tensor %.*s has type %s, expected %s, q6_k, or q8_0\n",
                (int)t->name.len, t->name.ptr, tensor_type_name(t->type), tensor_type_name(type));
        exit(1);
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}

static void tensor_expect_optional_plain(const q36_tensor *t, uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    if (t) tensor_expect_plain_layout(t, ndim, d0, d1, d2);
}

static void tensor_expect_routed_gate_up(const q36_tensor *t, uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    uint64_t want[3];
    if (!t) q36_die("internal error: missing routed expert tensor while validating layout");
    if (!tensor_type_is_routed_gate_up(t->type)) {
        fprintf(stderr, "q36: tensor %.*s has type %s, expected a routed gate/up expert quant type\n",
                (int)t->name.len, t->name.ptr, tensor_type_name(t->type));
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr, "q36: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len, t->name.ptr, t->ndim, ndim);
        exit(1);
    }
    want[0] = d0;
    want[1] = d1;
    want[2] = d2;
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] != want[i]) {
            fprintf(stderr, "q36: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                    (int)t->name.len, t->name.ptr, i, t->dim[i], want[i]);
            exit(1);
        }
    }
}

static void tensor_expect_routed_down(const q36_tensor *t, uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    uint64_t want[3];
    if (!t) q36_die("internal error: missing routed expert tensor while validating layout");
    if (!tensor_type_is_routed_down(t->type)) {
        fprintf(stderr, "q36: tensor %.*s has type %s, expected a routed down expert quant type\n",
                (int)t->name.len, t->name.ptr, tensor_type_name(t->type));
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr, "q36: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len, t->name.ptr, t->ndim, ndim);
        exit(1);
    }
    want[0] = d0;
    want[1] = d1;
    want[2] = d2;
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] != want[i]) {
            fprintf(stderr, "q36: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                    (int)t->name.len, t->name.ptr, i, t->dim[i], want[i]);
            exit(1);
        }
    }
}

static int q36_quant_bits_from_type(uint32_t type) {
    switch (type) {
    case Q36_TENSOR_IQ2_XXS:
    case Q36_TENSOR_IQ2_S:
    case Q36_TENSOR_Q2_K:
        return 2;
    case Q36_TENSOR_IQ3_S:
        return 3;
    case Q36_TENSOR_Q4_K:
        return 4;
    case Q36_TENSOR_Q5_K:
        return 5;
    case Q36_TENSOR_Q6_K:
        return 6;
    case Q36_TENSOR_Q8_0:
        return 8;
    default:
        return 0;
    }
}

static void weights_validate_layout(const q36_weights *w) {
    if (!w) q36_die("internal error: missing weights while validating layout");
    tensor_expect_layout_or_q8_0(w->token_embd, Q36_TENSOR_Q4_K, 2, Q36_N_EMBD, Q36_N_VOCAB, 0);
    tensor_expect_layout(w->output_norm, Q36_TENSOR_F32, 1, Q36_N_EMBD, 0, 0);
    tensor_expect_layout_or_q6_k_or_q8_0(w->output, Q36_TENSOR_Q4_K, 2, Q36_N_EMBD, Q36_N_VOCAB, 0);
    tensor_expect_optional_plain(w->output_scale, 1, 1, 0, 0);

    for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
        const q36_layer_weights *l = &w->layer[il];
        tensor_expect_layout(l->attn_norm, Q36_TENSOR_F32, 1, Q36_N_EMBD, 0, 0);
        tensor_expect_layout(l->post_attention_norm, Q36_TENSOR_F32, 1, Q36_N_EMBD, 0, 0);

        if (l->kind == Q36_LAYER_FULL_ATTN) {
            tensor_expect_layout_or_q8_0(l->attn_q, Q36_TENSOR_Q5_K, 2, Q36_N_EMBD, Q36_N_SSM_INNER * 2u, 0);
            tensor_expect_layout(l->attn_q_norm, Q36_TENSOR_F32, 1, Q36_N_HEAD_DIM, 0, 0);
            tensor_expect_layout_or_q8_0(l->attn_k, Q36_TENSOR_Q5_K, 2, Q36_N_EMBD, (uint64_t)Q36_N_HEAD_KV * Q36_N_HEAD_DIM, 0);
            tensor_expect_layout(l->attn_k_norm, Q36_TENSOR_F32, 1, Q36_N_HEAD_DIM, 0, 0);
            tensor_expect_layout_or_q8_0(l->attn_v, Q36_TENSOR_Q5_K, 2, Q36_N_EMBD, (uint64_t)Q36_N_HEAD_KV * Q36_N_VALUE_DIM, 0);
            tensor_expect_layout_or_q8_0(l->attn_output, Q36_TENSOR_Q5_K, 2, Q36_N_SSM_INNER, Q36_N_EMBD, 0);
            tensor_expect_optional_plain(l->attn_sinks, 1, Q36_N_HEAD, 0, 0);
            tensor_expect_optional_plain(l->attn_q_scale, 1, 1, 0, 0);
            tensor_expect_optional_plain(l->attn_k_scale, 1, 1, 0, 0);
            tensor_expect_optional_plain(l->attn_v_scale, 1, 1, 0, 0);
            tensor_expect_optional_plain(l->attn_output_scale, 1, 1, 0, 0);
        } else {
            tensor_expect_layout_or_q8_0(l->attn_gate, Q36_TENSOR_Q5_K, 2, Q36_N_EMBD, Q36_N_SSM_INNER, 0);
            tensor_expect_layout_or_q8_0(l->attn_qkv, Q36_TENSOR_Q5_K, 2, Q36_N_EMBD, Q36_N_SSM_INNER * 2u, 0);
            tensor_expect_layout(l->ssm_a, Q36_TENSOR_F32, 1, Q36_N_SSM_DT_RANK, 0, 0);
            tensor_expect_layout_or_q8_0(l->ssm_alpha, Q36_TENSOR_F32, 2, Q36_N_EMBD, Q36_N_SSM_DT_RANK, 0);
            tensor_expect_layout_or_q8_0(l->ssm_beta, Q36_TENSOR_F32, 2, Q36_N_EMBD, Q36_N_SSM_DT_RANK, 0);
            tensor_expect_layout(l->ssm_conv1d, Q36_TENSOR_F32, 2, Q36_N_SSM_CONV, Q36_N_SSM_INNER * 2u, 0);
            tensor_expect_layout(l->ssm_dt, Q36_TENSOR_F32, 1, Q36_N_SSM_DT_RANK, 0, 0);
            tensor_expect_layout(l->ssm_norm, Q36_TENSOR_F32, 1, Q36_N_SSM_STATE, 0, 0);
            tensor_expect_layout_or_q8_0(l->ssm_out, Q36_TENSOR_Q6_K, 2, Q36_N_SSM_INNER, Q36_N_EMBD, 0);
            tensor_expect_optional_plain(l->attn_gate_scale, 1, 1, 0, 0);
            tensor_expect_optional_plain(l->attn_qkv_scale, 1, 1, 0, 0);
            tensor_expect_optional_plain(l->ssm_alpha_scale, 1, 1, 0, 0);
            tensor_expect_optional_plain(l->ssm_beta_scale, 1, 1, 0, 0);
            tensor_expect_optional_plain(l->ssm_out_scale, 1, 1, 0, 0);
        }

        tensor_expect_layout_or_q8_0(l->ffn_gate_inp, Q36_TENSOR_F32, 2, Q36_N_EMBD, Q36_N_EXPERT, 0);
        tensor_expect_layout(l->ffn_gate_inp_shexp, Q36_TENSOR_F32, 1, Q36_N_EMBD, 0, 0);
        tensor_expect_routed_gate_up(l->ffn_gate_exps, 3, Q36_N_EMBD, Q36_N_FF_EXP, Q36_N_EXPERT);
        tensor_expect_layout_or_q8_0(l->ffn_gate_shexp, Q36_TENSOR_Q5_K, 2, Q36_N_EMBD, Q36_N_FF_SHARED, 0);
        tensor_expect_routed_gate_up(l->ffn_up_exps, 3, Q36_N_EMBD, Q36_N_FF_EXP, Q36_N_EXPERT);
        tensor_expect_layout_or_q8_0(l->ffn_up_shexp, Q36_TENSOR_Q5_K, 2, Q36_N_EMBD, Q36_N_FF_SHARED, 0);
        tensor_expect_routed_down(l->ffn_down_exps, 3, Q36_N_FF_EXP, Q36_N_EMBD, Q36_N_EXPERT);
        tensor_expect_layout_or_q8_0(l->ffn_down_shexp, Q36_TENSOR_Q6_K, 2, Q36_N_FF_SHARED, Q36_N_EMBD, 0);
        tensor_expect_optional_plain(l->ffn_gate_exps_scale, 1, Q36_N_EXPERT, 0, 0);
        tensor_expect_optional_plain(l->ffn_gate_shexp_scale, 1, 1, 0, 0);
        tensor_expect_optional_plain(l->ffn_up_exps_scale, 1, Q36_N_EXPERT, 0, 0);
        tensor_expect_optional_plain(l->ffn_up_shexp_scale, 1, 1, 0, 0);
        tensor_expect_optional_plain(l->ffn_down_exps_scale, 1, Q36_N_EXPERT, 0, 0);
        tensor_expect_optional_plain(l->ffn_down_shexp_scale, 1, 1, 0, 0);
    }
}

static void weights_bind(q36_weights *w, const q36_model *m) {
    memset(w, 0, sizeof(*w));
    w->token_embd = required_tensor(m, "token_embd.weight");
    w->output_norm = required_tensor(m, "output_norm.weight");
    w->output = required_tensor(m, "output.weight");
    w->output_scale = model_find_tensor(m, "output.scale");

    for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
        q36_layer_weights *l = &w->layer[il];
        l->kind = q36_layer_is_full_attention(il) ? Q36_LAYER_FULL_ATTN : Q36_LAYER_RECURRENT;
        l->attn_norm = required_tensorf(m, "blk.%u.attn_norm.weight", il);
        l->post_attention_norm = required_tensorf(m, "blk.%u.post_attention_norm.weight", il);
        if (l->kind == Q36_LAYER_FULL_ATTN) {
            l->attn_q = required_tensorf(m, "blk.%u.attn_q.weight", il);
            l->attn_q_norm = required_tensorf(m, "blk.%u.attn_q_norm.weight", il);
            l->attn_k = required_tensorf(m, "blk.%u.attn_k.weight", il);
            l->attn_k_norm = required_tensorf(m, "blk.%u.attn_k_norm.weight", il);
            l->attn_v = required_tensorf(m, "blk.%u.attn_v.weight", il);
            l->attn_output = required_tensorf(m, "blk.%u.attn_output.weight", il);
            l->attn_sinks = tensor_by_namef(m, "blk.%u.attn_sinks.weight", il);
            l->attn_q_scale = tensor_by_namef(m, "blk.%u.attn_q.scale", il);
            l->attn_k_scale = tensor_by_namef(m, "blk.%u.attn_k.scale", il);
            l->attn_v_scale = tensor_by_namef(m, "blk.%u.attn_v.scale", il);
            l->attn_output_scale = tensor_by_namef(m, "blk.%u.attn_output.scale", il);
        } else {
            l->attn_gate = required_tensorf(m, "blk.%u.attn_gate.weight", il);
            l->attn_qkv = required_tensorf(m, "blk.%u.attn_qkv.weight", il);
            l->ssm_a = required_tensorf(m, "blk.%u.ssm_a", il);
            l->ssm_alpha = required_tensorf(m, "blk.%u.ssm_alpha.weight", il);
            l->ssm_beta = required_tensorf(m, "blk.%u.ssm_beta.weight", il);
            l->ssm_conv1d = required_tensorf(m, "blk.%u.ssm_conv1d.weight", il);
            l->ssm_dt = required_tensorf(m, "blk.%u.ssm_dt.bias", il);
            l->ssm_norm = required_tensorf(m, "blk.%u.ssm_norm.weight", il);
            l->ssm_out = required_tensorf(m, "blk.%u.ssm_out.weight", il);
            l->attn_gate_scale = tensor_by_namef(m, "blk.%u.attn_gate.scale", il);
            l->attn_qkv_scale = tensor_by_namef(m, "blk.%u.attn_qkv.scale", il);
            l->ssm_alpha_scale = tensor_by_namef(m, "blk.%u.ssm_alpha.scale", il);
            l->ssm_beta_scale = tensor_by_namef(m, "blk.%u.ssm_beta.scale", il);
            l->ssm_out_scale = tensor_by_namef(m, "blk.%u.ssm_out.scale", il);
        }
        l->ffn_gate_inp = required_tensorf(m, "blk.%u.ffn_gate_inp.weight", il);
        l->ffn_gate_inp_shexp = required_tensorf(m, "blk.%u.ffn_gate_inp_shexp.weight", il);
        l->ffn_gate_exps = required_tensorf(m, "blk.%u.ffn_gate_exps.weight", il);
        l->ffn_gate_shexp = required_tensorf(m, "blk.%u.ffn_gate_shexp.weight", il);
        l->ffn_up_exps = required_tensorf(m, "blk.%u.ffn_up_exps.weight", il);
        l->ffn_up_shexp = required_tensorf(m, "blk.%u.ffn_up_shexp.weight", il);
        l->ffn_down_exps = required_tensorf(m, "blk.%u.ffn_down_exps.weight", il);
        l->ffn_down_shexp = required_tensorf(m, "blk.%u.ffn_down_shexp.weight", il);
        l->ffn_gate_exps_scale = tensor_by_namef(m, "blk.%u.ffn_gate_exps.scale", il);
        l->ffn_gate_shexp_scale = tensor_by_namef(m, "blk.%u.ffn_gate_shexp.scale", il);
        l->ffn_up_exps_scale = tensor_by_namef(m, "blk.%u.ffn_up_exps.scale", il);
        l->ffn_up_shexp_scale = tensor_by_namef(m, "blk.%u.ffn_up_shexp.scale", il);
        l->ffn_down_exps_scale = tensor_by_namef(m, "blk.%u.ffn_down_exps.scale", il);
        l->ffn_down_shexp_scale = tensor_by_namef(m, "blk.%u.ffn_down_shexp.scale", il);
    }
    weights_validate_layout(w);
}

#ifndef Q36_NO_GPU
static void mtp_weights_validate_layout(const q36_mtp_weights *w) {
    const q36_layer_weights *l;
    if (!w) q36_die("internal error: missing MTP weights while validating layout");
    l = &w->block;
    tensor_expect_layout(w->token_embd, Q36_TENSOR_Q8_0, 2, Q36_N_EMBD, Q36_N_VOCAB, 0);
    tensor_expect_layout(w->output_norm, Q36_TENSOR_F32, 1, Q36_N_EMBD, 0, 0);
    tensor_expect_layout(w->output, Q36_TENSOR_Q8_0, 2, Q36_N_EMBD, Q36_N_VOCAB, 0);
    tensor_expect_optional_plain(w->output_scale, 1, 1, 0, 0);
    tensor_expect_layout(w->eh_proj, Q36_TENSOR_Q8_0, 2, Q36_N_EMBD * 2u, Q36_N_EMBD, 0);
    tensor_expect_layout(w->enorm, Q36_TENSOR_F32, 1, Q36_N_EMBD, 0, 0);
    tensor_expect_layout(w->hnorm, Q36_TENSOR_F32, 1, Q36_N_EMBD, 0, 0);
    tensor_expect_layout(w->shared_head_norm, Q36_TENSOR_F32, 1, Q36_N_EMBD, 0, 0);

    tensor_expect_layout(l->attn_norm, Q36_TENSOR_F32, 1, Q36_N_EMBD, 0, 0);
    tensor_expect_layout(l->post_attention_norm, Q36_TENSOR_F32, 1, Q36_N_EMBD, 0, 0);
    tensor_expect_layout(l->attn_q, Q36_TENSOR_Q8_0, 2, Q36_N_EMBD, Q36_N_SSM_INNER * 2u, 0);
    tensor_expect_layout(l->attn_q_norm, Q36_TENSOR_F32, 1, Q36_N_HEAD_DIM, 0, 0);
    tensor_expect_layout(l->attn_k, Q36_TENSOR_Q8_0, 2, Q36_N_EMBD, (uint64_t)Q36_N_HEAD_KV * Q36_N_HEAD_DIM, 0);
    tensor_expect_layout(l->attn_k_norm, Q36_TENSOR_F32, 1, Q36_N_HEAD_DIM, 0, 0);
    tensor_expect_layout(l->attn_v, Q36_TENSOR_Q8_0, 2, Q36_N_EMBD, (uint64_t)Q36_N_HEAD_KV * Q36_N_VALUE_DIM, 0);
    tensor_expect_layout(l->attn_output, Q36_TENSOR_Q8_0, 2, Q36_N_SSM_INNER, Q36_N_EMBD, 0);
    tensor_expect_optional_plain(l->attn_sinks, 1, Q36_N_HEAD, 0, 0);
    tensor_expect_optional_plain(l->attn_q_scale, 1, 1, 0, 0);
    tensor_expect_optional_plain(l->attn_k_scale, 1, 1, 0, 0);
    tensor_expect_optional_plain(l->attn_v_scale, 1, 1, 0, 0);
    tensor_expect_optional_plain(l->attn_output_scale, 1, 1, 0, 0);

    tensor_expect_layout_or_q8_0(l->ffn_gate_inp, Q36_TENSOR_F32, 2, Q36_N_EMBD, Q36_N_EXPERT, 0);
    tensor_expect_layout(l->ffn_gate_inp_shexp, Q36_TENSOR_F32, 1, Q36_N_EMBD, 0, 0);
    tensor_expect_routed_gate_up(l->ffn_gate_exps, 3, Q36_N_EMBD, Q36_N_FF_EXP, Q36_N_EXPERT);
    tensor_expect_layout_or_q8_0(l->ffn_gate_shexp, Q36_TENSOR_Q5_K, 2, Q36_N_EMBD, Q36_N_FF_SHARED, 0);
    tensor_expect_routed_gate_up(l->ffn_up_exps, 3, Q36_N_EMBD, Q36_N_FF_EXP, Q36_N_EXPERT);
    tensor_expect_layout_or_q8_0(l->ffn_up_shexp, Q36_TENSOR_Q5_K, 2, Q36_N_EMBD, Q36_N_FF_SHARED, 0);
    tensor_expect_routed_down(l->ffn_down_exps, 3, Q36_N_FF_EXP, Q36_N_EMBD, Q36_N_EXPERT);
    tensor_expect_layout_or_q8_0(l->ffn_down_shexp, Q36_TENSOR_Q6_K, 2, Q36_N_FF_SHARED, Q36_N_EMBD, 0);
    tensor_expect_optional_plain(l->ffn_gate_exps_scale, 1, Q36_N_EXPERT, 0, 0);
    tensor_expect_optional_plain(l->ffn_gate_shexp_scale, 1, 1, 0, 0);
    tensor_expect_optional_plain(l->ffn_up_exps_scale, 1, Q36_N_EXPERT, 0, 0);
    tensor_expect_optional_plain(l->ffn_up_shexp_scale, 1, 1, 0, 0);
    tensor_expect_optional_plain(l->ffn_down_exps_scale, 1, Q36_N_EXPERT, 0, 0);
    tensor_expect_optional_plain(l->ffn_down_shexp_scale, 1, 1, 0, 0);
}

static void mtp_weights_bind(q36_mtp_weights *w, const q36_model *m) {
    const uint32_t il = Q36_N_LAYER;
    q36_layer_weights *l;
    memset(w, 0, sizeof(*w));
    w->token_embd = required_tensor(m, "token_embd.weight");
    w->output_norm = required_tensor(m, "output_norm.weight");
    w->output = required_tensor(m, "output.weight");
    w->output_scale = model_find_tensor(m, "output.scale");
    w->eh_proj = required_tensorf(m, "blk.%u.nextn.eh_proj.weight", il);
    w->enorm = required_tensorf(m, "blk.%u.nextn.enorm.weight", il);
    w->hnorm = required_tensorf(m, "blk.%u.nextn.hnorm.weight", il);
    w->shared_head_norm = required_tensorf(m, "blk.%u.nextn.shared_head_norm.weight", il);

    l = &w->block;
    l->kind = Q36_LAYER_FULL_ATTN;
    l->attn_norm = required_tensorf(m, "blk.%u.attn_norm.weight", il);
    l->post_attention_norm = required_tensorf(m, "blk.%u.post_attention_norm.weight", il);
    l->attn_q = required_tensorf(m, "blk.%u.attn_q.weight", il);
    l->attn_q_norm = required_tensorf(m, "blk.%u.attn_q_norm.weight", il);
    l->attn_k = required_tensorf(m, "blk.%u.attn_k.weight", il);
    l->attn_k_norm = required_tensorf(m, "blk.%u.attn_k_norm.weight", il);
    l->attn_v = required_tensorf(m, "blk.%u.attn_v.weight", il);
    l->attn_output = required_tensorf(m, "blk.%u.attn_output.weight", il);
    l->attn_sinks = tensor_by_namef(m, "blk.%u.attn_sinks.weight", il);
    l->attn_q_scale = tensor_by_namef(m, "blk.%u.attn_q.scale", il);
    l->attn_k_scale = tensor_by_namef(m, "blk.%u.attn_k.scale", il);
    l->attn_v_scale = tensor_by_namef(m, "blk.%u.attn_v.scale", il);
    l->attn_output_scale = tensor_by_namef(m, "blk.%u.attn_output.scale", il);
    l->ffn_gate_inp = required_tensorf(m, "blk.%u.ffn_gate_inp.weight", il);
    l->ffn_gate_inp_shexp = required_tensorf(m, "blk.%u.ffn_gate_inp_shexp.weight", il);
    l->ffn_gate_exps = required_tensorf(m, "blk.%u.ffn_gate_exps.weight", il);
    l->ffn_gate_shexp = required_tensorf(m, "blk.%u.ffn_gate_shexp.weight", il);
    l->ffn_up_exps = required_tensorf(m, "blk.%u.ffn_up_exps.weight", il);
    l->ffn_up_shexp = required_tensorf(m, "blk.%u.ffn_up_shexp.weight", il);
    l->ffn_down_exps = required_tensorf(m, "blk.%u.ffn_down_exps.weight", il);
    l->ffn_down_shexp = required_tensorf(m, "blk.%u.ffn_down_shexp.weight", il);
    l->ffn_gate_exps_scale = tensor_by_namef(m, "blk.%u.ffn_gate_exps.scale", il);
    l->ffn_gate_shexp_scale = tensor_by_namef(m, "blk.%u.ffn_gate_shexp.scale", il);
    l->ffn_up_exps_scale = tensor_by_namef(m, "blk.%u.ffn_up_exps.scale", il);
    l->ffn_up_shexp_scale = tensor_by_namef(m, "blk.%u.ffn_up_shexp.scale", il);
    l->ffn_down_exps_scale = tensor_by_namef(m, "blk.%u.ffn_down_exps.scale", il);
    l->ffn_down_shexp_scale = tensor_by_namef(m, "blk.%u.ffn_down_shexp.scale", il);
    mtp_weights_validate_layout(w);
}
#endif

static bool q36_streaming_tensor_expert_bytes(const q36_tensor *t, uint64_t *bytes_out) {
    uint64_t row_bytes = 0;
    uint64_t expert_bytes = 0;
    if (bytes_out) *bytes_out = 0;
    if (!t || !bytes_out || t->ndim < 3 || t->dim[2] == 0) return false;
    if (!tensor_nbytes(t->type, t->dim[0], &row_bytes) || row_bytes == 0) return false;
    if (t->dim[1] > UINT64_MAX / row_bytes) return false;
    expert_bytes = t->dim[1] * row_bytes;
    if (expert_bytes == 0) return false;
    *bytes_out = expert_bytes;
    return true;
}

static bool q36_streaming_layer_routed_expert_bytes(const q36_layer_weights *layer,
                                                    uint64_t *gate_bytes,
                                                    uint64_t *up_bytes,
                                                    uint64_t *down_bytes,
                                                    uint64_t *total_bytes) {
    uint64_t gate = 0, up = 0, down = 0, total = 0;
    if (gate_bytes) *gate_bytes = 0;
    if (up_bytes) *up_bytes = 0;
    if (down_bytes) *down_bytes = 0;
    if (total_bytes) *total_bytes = 0;
    if (!layer ||
        !q36_streaming_tensor_expert_bytes(layer->ffn_gate_exps, &gate) ||
        !q36_streaming_tensor_expert_bytes(layer->ffn_up_exps, &up) ||
        !q36_streaming_tensor_expert_bytes(layer->ffn_down_exps, &down)) {
        return false;
    }
    if (gate > UINT64_MAX - up) return false;
    total = gate + up;
    if (total > UINT64_MAX - down) return false;
    total += down;
    if (gate_bytes) *gate_bytes = gate;
    if (up_bytes) *up_bytes = up;
    if (down_bytes) *down_bytes = down;
    if (total_bytes) *total_bytes = total;
    return total != 0;
}

static bool q36_streaming_routed_expert_bytes(const q36_weights *weights,
                                              uint64_t *per_expert_bytes_out) {
    if (per_expert_bytes_out) *per_expert_bytes_out = 0;
    if (!weights || !per_expert_bytes_out) return false;
    for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
        if (q36_streaming_layer_routed_expert_bytes(&weights->layer[il], NULL, NULL, NULL,
                                                    per_expert_bytes_out)) {
            return true;
        }
    }
    return false;
}

#ifndef Q36_NO_GPU
static bool q36_weights_streaming_layer_experts_uniform(const q36_weights *weights,
                                                        uint32_t il) {
    uint64_t base = 0;
    uint64_t bytes = 0;
    if (!weights || il >= Q36_N_LAYER) return true;
    if (!q36_streaming_routed_expert_bytes(weights, &base)) return true;
    if (!q36_streaming_layer_routed_expert_bytes(&weights->layer[il], NULL, NULL, NULL, &bytes)) return true;
    return bytes == base;
}

static bool q36_streaming_full_layer_budget(const q36_weights *weights,
                                            uint32_t layers,
                                            uint32_t *slots_out,
                                            uint64_t *bytes_out) {
    uint64_t base = 0, bytes = 0;
    if (slots_out) *slots_out = 0;
    if (bytes_out) *bytes_out = 0;
    if (!weights || !slots_out || !bytes_out || layers > Q36_N_LAYER ||
        !q36_streaming_routed_expert_bytes(weights, &base) || base == 0) {
        return false;
    }
    for (uint32_t il = 0; il < layers; il++) {
        uint64_t per_expert = 0;
        if (!q36_streaming_layer_routed_expert_bytes(&weights->layer[il],
                                                      NULL, NULL, NULL,
                                                      &per_expert) ||
            per_expert > UINT64_MAX / Q36_N_EXPERT) {
            return false;
        }
        uint64_t layer_bytes = per_expert * Q36_N_EXPERT;
        if (bytes > UINT64_MAX - layer_bytes) return false;
        bytes += layer_bytes;
    }
    uint64_t slots = bytes / base + (bytes % base != 0);
    if (slots > UINT32_MAX) return false;
    *slots_out = (uint32_t)slots;
    *bytes_out = bytes;
    return true;
}
#endif

static uint32_t q36_streaming_cache_experts_for_byte_budget(const q36_weights *weights,
                                                            uint64_t bytes,
                                                            uint64_t *per_expert_bytes_out) {
    uint64_t per_expert_bytes = 0;
    if (per_expert_bytes_out) *per_expert_bytes_out = 0;
    if (!weights || bytes == 0 || !q36_streaming_routed_expert_bytes(weights, &per_expert_bytes)) return 0;
    if (per_expert_bytes_out) *per_expert_bytes_out = per_expert_bytes;
    return q36_ssd_cache_experts_for_byte_budget(bytes, per_expert_bytes);
}

#ifndef Q36_NO_GPU
static bool q36_weights_streaming_non_routed_bytes(const q36_weights *weights,
                                                   uint64_t *bytes_out) {
    uint64_t bytes = 0;
    if (bytes_out) *bytes_out = 0;
    if (!weights || !bytes_out) return false;
    const q36_tensor *skip[Q36_N_LAYER * 3u];
    uint32_t n_skip = 0;
    for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
        const q36_layer_weights *l = &weights->layer[il];
        skip[n_skip++] = l->ffn_gate_exps;
        skip[n_skip++] = l->ffn_up_exps;
        skip[n_skip++] = l->ffn_down_exps;
    }
    const q36_tensor *all[4 + Q36_N_LAYER * 40u];
    uint32_t n = 0;
    all[n++] = weights->token_embd;
    all[n++] = weights->output_norm;
    all[n++] = weights->output;
    all[n++] = weights->output_scale;
    for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
        const q36_layer_weights *l = &weights->layer[il];
        all[n++] = l->attn_norm;
        all[n++] = l->post_attention_norm;
        all[n++] = l->attn_q;
        all[n++] = l->attn_q_norm;
        all[n++] = l->attn_k;
        all[n++] = l->attn_k_norm;
        all[n++] = l->attn_v;
        all[n++] = l->attn_output;
        all[n++] = l->attn_sinks;
        all[n++] = l->attn_q_scale;
        all[n++] = l->attn_k_scale;
        all[n++] = l->attn_v_scale;
        all[n++] = l->attn_output_scale;
        all[n++] = l->attn_gate;
        all[n++] = l->attn_qkv;
        all[n++] = l->ssm_a;
        all[n++] = l->ssm_alpha;
        all[n++] = l->ssm_beta;
        all[n++] = l->ssm_conv1d;
        all[n++] = l->ssm_dt;
        all[n++] = l->ssm_norm;
        all[n++] = l->ssm_out;
        all[n++] = l->attn_gate_scale;
        all[n++] = l->attn_qkv_scale;
        all[n++] = l->ssm_alpha_scale;
        all[n++] = l->ssm_beta_scale;
        all[n++] = l->ssm_out_scale;
        all[n++] = l->ffn_gate_inp;
        all[n++] = l->ffn_gate_inp_shexp;
        all[n++] = l->ffn_gate_shexp;
        all[n++] = l->ffn_up_shexp;
        all[n++] = l->ffn_down_shexp;
        all[n++] = l->ffn_gate_exps_scale;
        all[n++] = l->ffn_gate_shexp_scale;
        all[n++] = l->ffn_up_exps_scale;
        all[n++] = l->ffn_up_shexp_scale;
        all[n++] = l->ffn_down_exps_scale;
        all[n++] = l->ffn_down_shexp_scale;
    }
    for (uint32_t i = 0; i < n; i++) {
        const q36_tensor *t = all[i];
        bool routed = false;
        if (!t) continue;
        for (uint32_t j = 0; j < n_skip; j++) {
            if (t == skip[j]) {
                routed = true;
                break;
            }
        }
        if (routed) continue;
        if (bytes > UINT64_MAX - t->bytes) return false;
        bytes += t->bytes;
    }
    *bytes_out = bytes;
    return true;
}
#endif

#ifndef Q36_NO_GPU
static q36_gpu_stream_expert_table q36_stream_expert_table_make(const q36_model *model,
                                                                const q36_layer_weights *layer,
                                                                uint32_t il) {
    q36_gpu_stream_expert_table table;
    uint64_t gate = 0, up = 0, down = 0;
    memset(&table, 0, sizeof(table));
    if (!model || !layer ||
        !q36_streaming_layer_routed_expert_bytes(layer, &gate, &up, &down, NULL)) {
        return table;
    }
    table.model_map = model->map;
    table.model_size = model->size;
    table.layer = il;
    table.n_total_expert = Q36_N_EXPERT;
    table.gate_offset = layer->ffn_gate_exps->abs_offset;
    table.up_offset = layer->ffn_up_exps->abs_offset;
    table.down_offset = layer->ffn_down_exps->abs_offset;
    table.gate_scales_offset = layer->ffn_gate_exps_scale ? layer->ffn_gate_exps_scale->abs_offset : 0;
    table.up_scales_offset = layer->ffn_up_exps_scale ? layer->ffn_up_exps_scale->abs_offset : 0;
    table.down_scales_offset = layer->ffn_down_exps_scale ? layer->ffn_down_exps_scale->abs_offset : 0;
    table.gate_expert_bytes = gate;
    table.up_expert_bytes = up;
    table.down_expert_bytes = down;
    table.gate_type = layer->ffn_gate_exps->type;
    table.up_type = layer->ffn_up_exps->type;
    table.down_type = layer->ffn_down_exps->type;
    table.has_gate_scales = layer->ffn_gate_exps_scale != NULL;
    table.has_up_scales = layer->ffn_up_exps_scale != NULL;
    table.has_down_scales = layer->ffn_down_exps_scale != NULL;
    return table;
}
#endif

static uint64_t q36_streaming_manual_cache_safe_bytes(void) {
#ifdef Q36_NO_GPU
    return 0;
#else
    const uint64_t gib = 1024ull * 1024ull * 1024ull;
    uint64_t recommended = q36_gpu_recommended_working_set_size();
    if (recommended == 0) return 0;
    uint64_t safe = recommended > UINT64_MAX / 7ull ? UINT64_MAX : (recommended * 7ull) / 10ull;
    return (safe / gib) * gib;
#endif
}

#ifndef Q36_NO_GPU
static bool q36_engine_configure_streaming_auto_cache(q36_engine *e) {
    if (!e || !e->ssd_streaming ||
        !q36_backend_supports_ssd_streaming(e->backend) ||
        e->ssd_streaming_cache_experts != 0 ||
        e->ssd_streaming_cache_bytes != 0) {
        return true;
    }
    if (!q36_backend_supports_streaming_auto_cache(e->backend)) return true;

    uint64_t recommended = q36_gpu_recommended_working_set_size();
    if (recommended == 0) {
        fprintf(stderr,
                "q36: SSD streaming auto cache: recommended working set unavailable; set --ssd-streaming-cache-experts N or NGB explicitly\n");
        return false;
    }
    uint64_t non_routed_bytes = 0;
    if (!q36_weights_streaming_non_routed_bytes(&e->weights, &non_routed_bytes)) {
        fprintf(stderr, "q36: SSD streaming auto cache could not measure non-routed model weights\n");
        return false;
    }
    uint64_t per_expert_bytes = 0;
    if (!q36_streaming_routed_expert_bytes(&e->weights, &per_expert_bytes)) {
        fprintf(stderr, "q36: SSD streaming auto cache could not measure routed expert size\n");
        return false;
    }
    q36_ssd_cache_plan plan;
    if (!q36_ssd_auto_cache_plan(recommended,
                                 non_routed_bytes,
                                 per_expert_bytes,
                                 (uint64_t)Q36_N_LAYER * Q36_N_EXPERT,
                                 &plan)) {
        fprintf(stderr, "q36: SSD streaming auto cache could not compute a valid cache budget\n");
        return false;
    }
    e->ssd_streaming_cache_experts = plan.cache_experts;
    fprintf(stderr, "q36: SSD streaming auto cache budget\n");
    fprintf(stderr, "q36:   %s recommends %.2f GiB working set\n",
            q36_backend_name(e->backend), (double)recommended / 1073741824.0);
    fprintf(stderr, "q36:   using 80%% total for model + cached experts: %.2f GiB\n",
            (double)plan.model_target_bytes / 1073741824.0);
    fprintf(stderr, "q36:   non-routed weights: %.2f GiB\n",
            (double)non_routed_bytes / 1073741824.0);
    fprintf(stderr, "q36:   routed expert size: %.2f MiB\n",
            (double)per_expert_bytes / 1048576.0);
    fprintf(stderr, "q36:   cached expert count: %u (%.2f GiB)\n",
            e->ssd_streaming_cache_experts,
            (double)plan.effective_cache_bytes / 1073741824.0);
    if (plan.model_target_bytes <= non_routed_bytes) {
        fprintf(stderr,
                "q36:   note: non-routed weights already fill the 80%% target; keeping a one-expert cache\n");
    }
    return true;
}
#endif

#ifndef Q36_NO_GPU
static bool q36_streaming_hotlist_add(uint32_t layer,
                                      uint32_t expert,
                                      uint32_t priority,
                                      int32_t experts[Q36_N_LAYER][Q36_N_EXPERT],
                                      uint32_t priorities[Q36_N_LAYER][Q36_N_EXPERT],
                                      uint32_t counts[Q36_N_LAYER],
                                      bool seen[Q36_N_LAYER][Q36_N_EXPERT],
                                      uint32_t *loaded) {
    if (layer >= Q36_N_LAYER || expert >= Q36_N_EXPERT) return true;
    if (seen[layer][expert]) return true;
    if (counts[layer] >= Q36_N_EXPERT) return false;
    seen[layer][expert] = true;
    experts[layer][counts[layer]] = (int32_t)expert;
    priorities[layer][counts[layer]] = priority ? priority : 1u;
    counts[layer]++;
    (*loaded)++;
    return true;
}

static bool q36_streaming_hotlist_load_file(const char *path,
                                            uint32_t max_entries,
                                            int32_t experts[Q36_N_LAYER][Q36_N_EXPERT],
                                            uint32_t priorities[Q36_N_LAYER][Q36_N_EXPERT],
                                            uint32_t counts[Q36_N_LAYER],
                                            bool seen[Q36_N_LAYER][Q36_N_EXPERT],
                                            uint32_t *loaded_out) {
    FILE *fp;
    char line[256];
    uint64_t lineno = 0;
    uint32_t loaded = 0;
    if (!path || !path[0] || max_entries == 0 || !loaded_out) return false;
    fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "q36: failed to open streaming expert hotlist %s: %s\n",
                path, strerror(errno));
        return false;
    }
    while (loaded < max_entries && fgets(line, sizeof(line), fp)) {
        char *p = line;
        char *end;
        unsigned long layer, expert;
        unsigned long long hits;
        lineno++;
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p == '\0' || *p == '#') continue;

        errno = 0;
        layer = strtoul(p, &end, 10);
        if (end == p || errno != 0) goto bad;
        p = end;
        while (*p && isspace((unsigned char)*p)) p++;

        errno = 0;
        expert = strtoul(p, &end, 10);
        if (end == p || errno != 0) goto bad;
        p = end;
        while (*p && isspace((unsigned char)*p)) p++;

        errno = 0;
        hits = strtoull(p, &end, 10);
        if (end == p || errno != 0) goto bad;
        if (hits == 0) continue;
        if (!q36_streaming_hotlist_add((uint32_t)layer,
                                       (uint32_t)expert,
                                       hits > UINT32_MAX ? UINT32_MAX : (uint32_t)hits,
                                       experts,
                                       priorities,
                                       counts,
                                       seen,
                                       &loaded)) {
            goto bad;
        }
    }
    if (ferror(fp)) {
        fprintf(stderr, "q36: failed to read streaming expert hotlist %s: %s\n",
                path, strerror(errno));
        fclose(fp);
        return false;
    }
    fclose(fp);
    *loaded_out = loaded;
    return true;

bad:
    fprintf(stderr, "q36: invalid streaming expert hotlist line %" PRIu64 " in %s\n",
            lineno, path);
    fclose(fp);
    return false;
}

static bool q36_streaming_hotlist_load_default(uint32_t max_entries,
                                               int32_t experts[Q36_N_LAYER][Q36_N_EXPERT],
                                               uint32_t priorities[Q36_N_LAYER][Q36_N_EXPERT],
                                               uint32_t counts[Q36_N_LAYER],
                                               bool seen[Q36_N_LAYER][Q36_N_EXPERT],
                                               uint32_t *loaded_out) {
    uint32_t loaded = 0;
    if (max_entries == 0 || !loaded_out) return false;
    for (uint32_t i = 0; i < q36_default_streaming_hotlist_count && loaded < max_entries; i++) {
        if (!q36_streaming_hotlist_add(q36_default_streaming_hotlist[i][0],
                                       q36_default_streaming_hotlist[i][1],
                                       max_entries - loaded,
                                       experts,
                                       priorities,
                                       counts,
                                       seen,
                                       &loaded)) {
            return false;
        }
    }
    *loaded_out = loaded;
    return true;
}

static bool q36_engine_seed_streaming_expert_cache_blind(q36_engine *e,
                                                         uint32_t preload,
                                                         uint32_t *seeded_out) {
    uint32_t remaining = preload;
    uint32_t seeded = 0;
    for (uint32_t il = 0; il < Q36_N_LAYER && remaining != 0; il++) {
        const q36_layer_weights *layer = &e->weights.layer[il];
        if (!q36_weights_streaming_layer_experts_uniform(&e->weights, il)) continue;
        q36_gpu_stream_expert_table table = q36_stream_expert_table_make(&e->model, layer, il);
        if (!table.model_map) continue;
        uint32_t n = remaining > Q36_N_EXPERT ? Q36_N_EXPERT : remaining;
        int32_t ids[Q36_N_EXPERT];
        uint32_t prio[Q36_N_EXPERT];
        for (uint32_t i = 0; i < n; i++) {
            ids[i] = (int32_t)i;
            prio[i] = n - i;
        }
        if (!q36_gpu_stream_expert_cache_seed_experts(&table, ids, prio, n)) {
            fprintf(stderr, "q36: SSD streaming failed to preload experts for layer %u\n", il);
            return false;
        }
        seeded += n;
        remaining -= n;
    }
    *seeded_out = seeded;
    return true;
}
#endif

#ifndef Q36_NO_GPU
static bool q36_engine_seed_streaming_expert_cache(q36_engine *e) {
    if (!e || !e->ssd_streaming || e->ssd_streaming_cold) return true;
    uint32_t budget = q36_gpu_stream_expert_cache_configured_count();
    uint32_t preload = e->ssd_streaming_preload_experts;
    if (preload == 0) preload = budget < 4096u ? budget : 4096u;
    if (preload == 0) return true;
    if (budget != 0 && preload > budget) preload = budget;
    uint32_t seeded = 0;
    uint32_t loaded = 0;
    int32_t experts[Q36_N_LAYER][Q36_N_EXPERT];
    uint32_t priorities[Q36_N_LAYER][Q36_N_EXPERT];
    uint32_t counts[Q36_N_LAYER];
    bool seen[Q36_N_LAYER][Q36_N_EXPERT];
    const char *path = getenv("Q36_VK_STREAMING_EXPERT_HOTLIST");
    const bool disable_default = getenv("Q36_VK_DISABLE_STREAMING_EXPERT_HOTLIST") != NULL;
    double t0 = q36_now_sec();

    memset(experts, 0, sizeof(experts));
    memset(priorities, 0, sizeof(priorities));
    memset(counts, 0, sizeof(counts));
    memset(seen, 0, sizeof(seen));

    if (path && path[0]) {
        if (!q36_streaming_hotlist_load_file(path, preload, experts, priorities, counts, seen, &loaded)) {
            return false;
        }
    } else if (!disable_default) {
        if (!q36_streaming_hotlist_load_default(preload, experts, priorities, counts, seen, &loaded)) {
            return false;
        }
    }

    if (loaded == 0) {
        if (!q36_engine_seed_streaming_expert_cache_blind(e, preload, &seeded)) return false;
    } else {
        for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
            uint32_t n = counts[il];
            if (n == 0) continue;
            const q36_layer_weights *layer = &e->weights.layer[il];
            if (!q36_weights_streaming_layer_experts_uniform(&e->weights, il)) continue;
            q36_gpu_stream_expert_table table = q36_stream_expert_table_make(&e->model, layer, il);
            if (!table.model_map) continue;
            if (!q36_gpu_stream_expert_cache_seed_experts(&table, experts[il], priorities[il], n)) {
                fprintf(stderr, "q36: SSD streaming failed to preload hotlist experts for layer %u\n", il);
                return false;
            }
            seeded += n;
        }
    }
    if (seeded != 0) {
        fprintf(stderr,
                "q36: SSD streaming preloaded %u expert slots%s in %.2fs\n",
                seeded,
                loaded ? " from hotlist" : "",
                q36_now_sec() - t0);
    }
    return true;
}
#endif

#ifndef Q36_NO_GPU
static bool q36_vulkan_prewarm_enabled(void) {
    const char *env = getenv("Q36_VK_PREWARM");
    return !env || !env[0] || env[0] != '0';
}

static int q36_weights_tensor_routed_layer(const q36_weights *w, const q36_tensor *t) {
    if (!w || !t) return -1;
    for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
        const q36_layer_weights *l = &w->layer[il];
        if (t == l->ffn_gate_exps || t == l->ffn_up_exps || t == l->ffn_down_exps)
            return (int)il;
    }
    return -1;
}

static bool q36_vulkan_prewarm_skip_tensor(const q36_engine *e, const q36_tensor *t) {
    if (t == e->weights.token_embd || t->bytes == 0) return true;
    int il = q36_weights_tensor_routed_layer(&e->weights, t);
    return e->ssd_streaming && il >= 0 && (uint32_t)il >= e->ssd_streaming_full_layers;
}

/* Prewarm reader: fills the page cache with pread() a bounded distance ahead
 * of the staging memcpy, so SSD reads overlap the copy into GPU memory.  The
 * bound keeps read-ahead pages from being evicted again before staging on
 * unified-memory boxes where the GPU heap and the page cache share RAM. */
typedef struct {
    const q36_engine *e;
    uint64_t staged;            /* written by the stager, read by the reader */
} q36_prewarm_reader;

static void *q36_prewarm_reader_main(void *arg) {
    q36_prewarm_reader *r = arg;
    const q36_engine *e = r->e;
    const uint64_t window = 512ull * 1024 * 1024;
    const size_t chunk = 4u * 1024 * 1024;
    char *buf = malloc(chunk);
    uint64_t read_bytes = 0;
    if (!buf) return NULL;
    for (uint64_t i = 0; i < e->model.n_tensors; i++) {
        const q36_tensor *t = &e->model.tensors[i];
        if (q36_vulkan_prewarm_skip_tensor(e, t)) continue;
        for (uint64_t off = 0; off < t->bytes; off += chunk) {
            uint64_t staged;
            while ((staged = __atomic_load_n(&r->staged, __ATOMIC_RELAXED)) < read_bytes &&
                   read_bytes - staged > window) {
                usleep(2000);
            }
            size_t n = t->bytes - off < chunk ? (size_t)(t->bytes - off) : chunk;
            if (pread(e->model.fd, buf, n, (off_t)(t->abs_offset + off)) < 0) break;
            read_bytes += n;
        }
    }
    free(buf);
    return NULL;
}

static void q36_vulkan_prewarm_weights(const q36_engine *e) {
    if (!e || !q36_vulkan_prewarm_enabled()) return;
    q36_prewarm_reader reader = { .e = e, .staged = 0 };
    pthread_t reader_thread;
    bool threaded = e->model.fd >= 0 &&
                    pthread_create(&reader_thread, NULL, q36_prewarm_reader_main, &reader) == 0;
    for (uint64_t i = 0; i < e->model.n_tensors; i++) {
        const q36_tensor *t = &e->model.tensors[i];
        if (q36_vulkan_prewarm_skip_tensor(e, t)) continue;
        (void)q36_gpu_cache_model_range(e->model.map, e->model.size, t->abs_offset, t->bytes, NULL);
        __atomic_add_fetch(&reader.staged, t->bytes, __ATOMIC_RELAXED);
    }
    if (threaded) {
        __atomic_store_n(&reader.staged, UINT64_MAX, __ATOMIC_RELAXED);
        pthread_join(reader_thread, NULL);
    }
    /* The MTP support model is fetched through the same whole-tensor cache
     * keys at draft time; without a prewarm the first drafts pay its whole
     * upload (~1.2s spread over the first replies). */
    if (e->mtp_ready) {
        for (uint64_t i = 0; i < e->mtp_model.n_tensors; i++) {
            const q36_tensor *t = &e->mtp_model.tensors[i];
            if (t == e->mtp_weights.token_embd || t->bytes == 0) continue;
            (void)q36_gpu_cache_model_range(e->mtp_model.map, e->mtp_model.size, t->abs_offset, t->bytes, NULL);
        }
    }
}
#endif

static void config_validate_model(const q36_model *m) {
    uint64_t ctx_train = 0;
    q36_str arch = required_string(m, "general.architecture");
    q36_str tok_model = required_string(m, "tokenizer.ggml.model");
    q36_str tok_pre = required_string(m, "tokenizer.ggml.pre");
    config_expect_string("general.architecture", arch, "qwen35moe");
    config_expect_string("tokenizer.ggml.model", tok_model, "gpt2");
    config_expect_string("tokenizer.ggml.pre", tok_pre, "qwen35");
    config_expect_u32("qwen35moe.block_count", required_u32(m, "qwen35moe.block_count"), Q36_N_LAYER);
    if (!model_get_u64_compat(m, "qwen35moe.context_length", &ctx_train)) {
        fprintf(stderr, "q36: required metadata key is missing: qwen35moe.context_length\n");
        exit(1);
    }
    if (ctx_train != Q36_CONTEXT_TRAIN) {
        fprintf(stderr, "q36: expected qwen35moe.context_length=%u, got %" PRIu64 "\n", Q36_CONTEXT_TRAIN, ctx_train);
        exit(1);
    }
    config_expect_u32("qwen35moe.embedding_length", required_u32(m, "qwen35moe.embedding_length"), Q36_N_EMBD);
    config_expect_u32("qwen35moe.attention.head_count", required_u32(m, "qwen35moe.attention.head_count"), Q36_N_HEAD);
    config_expect_u32("qwen35moe.attention.head_count_kv", required_u32(m, "qwen35moe.attention.head_count_kv"), Q36_N_HEAD_KV);
    config_expect_u32("qwen35moe.attention.key_length", required_u32(m, "qwen35moe.attention.key_length"), Q36_N_HEAD_DIM);
    config_expect_u32("qwen35moe.attention.value_length", required_u32(m, "qwen35moe.attention.value_length"), Q36_N_VALUE_DIM);
    config_expect_u32("qwen35moe.rope.dimension_count", required_u32(m, "qwen35moe.rope.dimension_count"), Q36_N_ROT);
    config_expect_f32("qwen35moe.rope.freq_base", required_f32(m, "qwen35moe.rope.freq_base"), 10000000.0f);
    config_expect_u32_array(m, "qwen35moe.rope.dimension_sections", Q36_ROPE_SECTIONS, 4);
    config_expect_f32("qwen35moe.attention.layer_norm_rms_epsilon",
                      required_f32(m, "qwen35moe.attention.layer_norm_rms_epsilon"), Q36_RMS_EPS);
    config_expect_u32("qwen35moe.expert_count", required_u32(m, "qwen35moe.expert_count"), Q36_N_EXPERT);
    config_expect_u32("qwen35moe.expert_used_count", required_u32(m, "qwen35moe.expert_used_count"), Q36_N_EXPERT_USED);
    config_expect_u32("qwen35moe.expert_feed_forward_length", required_u32(m, "qwen35moe.expert_feed_forward_length"), Q36_N_FF_EXP);
    config_expect_u32("qwen35moe.expert_shared_feed_forward_length", required_u32(m, "qwen35moe.expert_shared_feed_forward_length"), Q36_N_FF_SHARED);
    config_expect_u32("qwen35moe.ssm.conv_kernel", required_u32(m, "qwen35moe.ssm.conv_kernel"), Q36_N_SSM_CONV);
    config_expect_u32("qwen35moe.ssm.state_size", required_u32(m, "qwen35moe.ssm.state_size"), Q36_N_SSM_STATE);
    config_expect_u32("qwen35moe.ssm.group_count", required_u32(m, "qwen35moe.ssm.group_count"), Q36_N_SSM_GROUP);
    config_expect_u32("qwen35moe.ssm.time_step_rank", required_u32(m, "qwen35moe.ssm.time_step_rank"), Q36_N_SSM_DT_RANK);
    config_expect_u32("qwen35moe.ssm.inner_size", required_u32(m, "qwen35moe.ssm.inner_size"), Q36_N_SSM_INNER);
    config_expect_u32("qwen35moe.full_attention_interval", required_u32(m, "qwen35moe.full_attention_interval"), Q36_FULL_ATTENTION_INTERVAL);
    if (m->n_tensors != Q36_TENSOR_COUNT) {
        fprintf(stderr, "q36: expected %u tensors, got %" PRIu64 "\n", Q36_TENSOR_COUNT, m->n_tensors);
        exit(1);
    }
}

static void print_size(uint64_t bytes) {
    printf("%.2f GiB", (double)bytes / (1024.0 * 1024.0 * 1024.0));
}

static void model_summary(const q36_model *m) {
    q36_str name = {0};
    q36_str arch = {0};
    uint64_t ctx_train = 0;
    uint64_t tensor_bytes = 0;
    uint64_t params = 0;
    model_get_string(m, "general.name", &name);
    model_get_string(m, "general.architecture", &arch);
    model_get_u64_compat(m, "qwen35moe.context_length", &ctx_train);
    for (uint64_t i = 0; i < m->n_tensors; i++) {
        tensor_bytes += m->tensors[i].bytes;
        params += m->tensors[i].elements;
    }
    printf("model: %.*s\n", (int)name.len, name.ptr);
    printf("arch:  %.*s\n", (int)arch.len, arch.ptr);
    printf("gguf:  v%u, %" PRIu64 " metadata keys, %" PRIu64 " tensors\n", m->version, m->n_kv, m->n_tensors);
    printf("layers: %u\n", Q36_N_LAYER);
    printf("train context: %" PRIu64 "\n", ctx_train);
    printf("attention: heads=%u kv_heads=%u head_dim=%u full_interval=%u\n",
           Q36_N_HEAD, Q36_N_HEAD_KV, Q36_N_HEAD_DIM, Q36_FULL_ATTENTION_INTERVAL);
    printf("experts: count=%u used=%u ff=%u shared_ff=%u\n",
           Q36_N_EXPERT, Q36_N_EXPERT_USED, Q36_N_FF_EXP, Q36_N_FF_SHARED);
    printf("file size: ");
    print_size(m->size);
    printf("\n");
    printf("tensor bytes described by GGUF: ");
    print_size(tensor_bytes);
    printf("\n");
    printf("logical parameters: %.2f B\n", (double)params / 1000000000.0);
}

static q36_cpu_runtime *q36_cpu_runtime_create(int ctx_size,
                                               uint32_t prefill_cap,
                                               q36_kv_cache_type cache_type_k,
                                               q36_kv_cache_type cache_type_v) {
    q36_cpu_runtime *rt;
    uint32_t full_cap;
    uint64_t state_dim;
    size_t scratch_bytes;
    if (ctx_size <= 0) return NULL;
    if (prefill_cap < 1) prefill_cap = 1;
    uint32_t k_row_bytes = q36_kv_cache_row_bytes(cache_type_k, Q36_N_HEAD_KV * Q36_N_HEAD_DIM);
    uint32_t v_row_bytes = q36_kv_cache_row_bytes(cache_type_v, Q36_N_HEAD_KV * Q36_N_VALUE_DIM);
    if (!k_row_bytes || !v_row_bytes) return NULL;
    rt = xcalloc(1, sizeof(*rt));
    full_cap = (uint32_t)ctx_size;
    state_dim = (uint64_t)Q36_N_SSM_STATE * Q36_N_SSM_STATE * Q36_N_SSM_DT_RANK;
    scratch_bytes = (size_t)Q36_CPU_SCRATCH_FLOATS * sizeof(float);
    rt->prefill_cap = prefill_cap;
    rt->hidden = xmalloc((size_t)Q36_N_EMBD * sizeof(float));
    rt->next_hidden = xmalloc((size_t)Q36_N_EMBD * sizeof(float));
    rt->work0 = xmalloc(scratch_bytes);
    rt->work1 = xmalloc(scratch_bytes);
    rt->work2 = xmalloc(scratch_bytes);
    rt->work3 = xmalloc(scratch_bytes);
    rt->work4 = xmalloc(scratch_bytes);
    rt->work5 = xmalloc(scratch_bytes);
    rt->scores = xmalloc((size_t)full_cap * sizeof(float));
    rt->batch_hidden = xmalloc((size_t)prefill_cap * Q36_N_EMBD * sizeof(float));
    rt->batch_next_hidden = xmalloc((size_t)prefill_cap * Q36_N_EMBD * sizeof(float));
    rt->batch_norm = xmalloc((size_t)prefill_cap * Q36_N_EMBD * sizeof(float));
    rt->batch_qg = xmalloc((size_t)prefill_cap * Q36_N_HEAD * Q36_N_HEAD_DIM * 2u * sizeof(float));
    rt->batch_q = xmalloc((size_t)prefill_cap * Q36_N_SSM_INNER * sizeof(float));
    rt->batch_k = xmalloc((size_t)prefill_cap * Q36_N_HEAD_KV * Q36_N_HEAD_DIM * sizeof(float));
    rt->batch_v = xmalloc((size_t)prefill_cap * Q36_N_HEAD_KV * Q36_N_VALUE_DIM * sizeof(float));
    rt->batch_attn_out = xmalloc((size_t)prefill_cap * Q36_N_SSM_INNER * sizeof(float));
    rt->batch_recur_qkv = xmalloc((size_t)prefill_cap * Q36_N_SSM_CONV_DIM * sizeof(float));
    rt->batch_recur_z = xmalloc((size_t)prefill_cap * Q36_N_SSM_INNER * sizeof(float));
    rt->batch_recur_alpha = xmalloc((size_t)prefill_cap * Q36_N_SSM_DT_RANK * sizeof(float));
    rt->batch_recur_beta = xmalloc((size_t)prefill_cap * Q36_N_SSM_DT_RANK * sizeof(float));
    rt->batch_recur_proj = xmalloc((size_t)prefill_cap * Q36_N_SSM_INNER * sizeof(float));
    rt->batch_ffn_gate_logits = xmalloc((size_t)prefill_cap * Q36_N_EXPERT * sizeof(float));
    rt->batch_ffn_shared_gate = xmalloc((size_t)prefill_cap * Q36_N_FF_SHARED * sizeof(float));
    rt->batch_ffn_shared_up = xmalloc((size_t)prefill_cap * Q36_N_FF_SHARED * sizeof(float));
    rt->batch_ffn_shared_mid = xmalloc((size_t)prefill_cap * Q36_N_FF_SHARED * sizeof(float));
    rt->batch_ffn_shared_out = xmalloc((size_t)prefill_cap * Q36_N_EMBD * sizeof(float));
    rt->batch_ffn_scalar = xmalloc((size_t)prefill_cap * sizeof(float));
    rt->batch_xq = xmalloc((size_t)prefill_cap * Q36_MAX_Q8_K_BYTES);
    for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
        if (q36_layer_is_full_attention(il)) {
            rt->full[il].cap = full_cap;
            rt->full[il].type_k = cache_type_k;
            rt->full[il].type_v = cache_type_v;
            rt->full[il].k_row_bytes = k_row_bytes;
            rt->full[il].v_row_bytes = v_row_bytes;
            rt->full[il].k = xmalloc_zeroed((size_t)full_cap, k_row_bytes);
            rt->full[il].v = xmalloc_zeroed((size_t)full_cap, v_row_bytes);
        } else {
            rt->recurrent[il].conv = xmalloc_zeroed((size_t)(Q36_N_SSM_CONV - 1u) * Q36_N_SSM_CONV_DIM, sizeof(float));
            rt->recurrent[il].state = xmalloc_zeroed((size_t)state_dim, sizeof(float));
        }
    }
    return rt;
}

static void q36_cpu_runtime_reset(q36_cpu_runtime *rt) {
    uint64_t state_dim = (uint64_t)Q36_N_SSM_STATE * Q36_N_SSM_STATE * Q36_N_SSM_DT_RANK;
    if (!rt) return;
    memset(rt->hidden, 0, (size_t)Q36_N_EMBD * sizeof(float));
    memset(rt->next_hidden, 0, (size_t)Q36_N_EMBD * sizeof(float));
    for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
        rt->full[il].len = 0;
        if (rt->recurrent[il].conv) {
            memset(rt->recurrent[il].conv, 0, (size_t)(Q36_N_SSM_CONV - 1u) * Q36_N_SSM_CONV_DIM * sizeof(float));
        }
        if (rt->recurrent[il].state) {
            memset(rt->recurrent[il].state, 0, (size_t)state_dim * sizeof(float));
        }
    }
}

static void q36_cpu_runtime_free(q36_cpu_runtime *rt) {
    if (!rt) return;
    for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
        free(rt->full[il].k);
        free(rt->full[il].v);
        free(rt->recurrent[il].conv);
        free(rt->recurrent[il].state);
    }
    free(rt->hidden);
    free(rt->next_hidden);
    free(rt->work0);
    free(rt->work1);
    free(rt->work2);
    free(rt->work3);
    free(rt->work4);
    free(rt->work5);
    free(rt->scores);
    free(rt->batch_hidden);
    free(rt->batch_next_hidden);
    free(rt->batch_norm);
    free(rt->batch_qg);
    free(rt->batch_q);
    free(rt->batch_k);
    free(rt->batch_v);
    free(rt->batch_attn_out);
    free(rt->batch_recur_qkv);
    free(rt->batch_recur_z);
    free(rt->batch_recur_alpha);
    free(rt->batch_recur_beta);
    free(rt->batch_recur_proj);
    free(rt->batch_ffn_gate_logits);
    free(rt->batch_ffn_shared_gate);
    free(rt->batch_ffn_shared_up);
    free(rt->batch_ffn_shared_mid);
    free(rt->batch_ffn_shared_out);
    free(rt->batch_ffn_scalar);
    free(rt->batch_xq);
    free(rt);
}

#ifndef Q36_NO_GPU
static bool q36_gpu_tensor_zero(q36_gpu_tensor *tensor) {
    void *p;
    if (!tensor) return false;
    p = q36_gpu_tensor_contents_named(tensor, "submit_wait_tensor_zero");
    if (!p) return false;
    memset(p, 0, (size_t)q36_gpu_tensor_bytes(tensor));
    return true;
}

static bool q36_gpu_tensor_all_finite(const q36_gpu_tensor *tensor, uint32_t n) {
    float *p = q36_gpu_tensor_contents_named((q36_gpu_tensor *)tensor, "submit_wait_all_finite");
    return p && q36_all_finite(p, n);
}

static bool q36_gpu_tensor_scale_host(q36_gpu_tensor *tensor, uint32_t n, float scale) {
    float *p = q36_gpu_tensor_contents_named(tensor, "submit_wait_scale_host");
    if (!p) return false;
    q36_scale_inplace(p, n, scale);
    return true;
}

static bool q36_gpu_embed_tokens(const q36_model *m, const q36_tensor *t,
                                 const int *tokens, uint32_t n_tok, q36_gpu_tensor *dst) {
    float *p = q36_gpu_tensor_contents_named(dst, "submit_wait_embed_tokens");
    if (!p) return false;
    for (uint32_t i = 0; i < n_tok; i++) {
        if (tokens[i] < 0 || tokens[i] >= Q36_N_VOCAB) return false;
        if (!q36_tensor_row_to_float(m, t, (uint64_t)tokens[i],
                                     p + (uint64_t)i * Q36_N_EMBD, Q36_N_EMBD)) {
            return false;
        }
    }
    return true;
}

/* Build the n_tok conv windows in token order while sliding the persistent
 * conv history, exactly n_tok runs of q36_recurrent_conv_step(). */
static Q36_MAYBE_UNUSED bool q36_gpu_recurrent_conv_step(q36_gpu_tensor *cache_conv,
                                        const q36_gpu_tensor *cur,
                                        q36_gpu_tensor *window,
                                        uint32_t n_tok) {
    const uint32_t hist = Q36_N_SSM_CONV - 1u;
    const size_t row_bytes = (size_t)Q36_N_SSM_CONV_DIM * sizeof(float);
    float *cachep = q36_gpu_tensor_contents_named(cache_conv, "submit_wait_conv_cache");
    float *windowp = q36_gpu_tensor_contents_named(window, "submit_wait_conv_window");
    float *curp = q36_gpu_tensor_contents_named((q36_gpu_tensor *)cur, "submit_wait_conv_cur");
    if (!cachep || !windowp || !curp) return false;
    for (uint32_t t = 0; t < n_tok; t++) {
        float *wt = windowp + (uint64_t)t * Q36_N_SSM_CONV * Q36_N_SSM_CONV_DIM;
        const float *ct = curp + (uint64_t)t * Q36_N_SSM_CONV_DIM;
        for (uint32_t i = 0; i < hist; i++) {
            memcpy(wt + (uint64_t)i * Q36_N_SSM_CONV_DIM,
                   cachep + (uint64_t)i * Q36_N_SSM_CONV_DIM,
                   row_bytes);
        }
        memcpy(wt + (uint64_t)hist * Q36_N_SSM_CONV_DIM, ct, row_bytes);
        if (hist > 1u) {
            memmove(cachep,
                    cachep + Q36_N_SSM_CONV_DIM,
                    (size_t)(hist - 1u) * row_bytes);
        }
        memcpy(cachep + (uint64_t)(hist - 1u) * Q36_N_SSM_CONV_DIM, ct, row_bytes);
    }
    return true;
}

static Q36_MAYBE_UNUSED bool q36_gpu_extract_full_attn_q(q36_gpu_tensor *dst, const q36_gpu_tensor *qg, uint32_t n_tok) {
    const float *src = q36_gpu_tensor_contents_named((q36_gpu_tensor *)qg, "submit_wait_extract_q_src");
    float *out = q36_gpu_tensor_contents_named(dst, "submit_wait_extract_q_dst");
    if (!src || !out) return false;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t h = 0; h < Q36_N_HEAD; h++) {
            memcpy(out + ((uint64_t)t * Q36_N_HEAD + h) * Q36_N_HEAD_DIM,
                   src + (uint64_t)t * Q36_N_HEAD * Q36_N_HEAD_DIM * 2u + (uint64_t)h * Q36_N_HEAD_DIM * 2u,
                   (size_t)Q36_N_HEAD_DIM * sizeof(float));
        }
    }
    return true;
}

static Q36_MAYBE_UNUSED bool q36_gpu_extract_recurrent_v(q36_gpu_tensor *dst, const q36_gpu_tensor *conv, uint32_t n_tok) {
    const float *src = q36_gpu_tensor_contents_named((q36_gpu_tensor *)conv, "submit_wait_extract_v_src");
    float *out = q36_gpu_tensor_contents_named(dst, "submit_wait_extract_v_dst");
    if (!src || !out) return false;
    for (uint32_t t = 0; t < n_tok; t++) {
        memcpy(out + (uint64_t)t * Q36_N_SSM_INNER,
               src + (uint64_t)t * Q36_N_SSM_CONV_DIM + (uint64_t)Q36_N_SSM_QK * 2u,
               (size_t)Q36_N_SSM_INNER * sizeof(float));
    }
    return true;
}

static bool q36_gpu_tensor_matmul_scaled(const q36_model *m,
                                         const q36_tensor *t,
                                         const q36_gpu_tensor *x,
                                         q36_gpu_tensor *out,
                                         uint32_t in_dim,
                                         uint32_t out_dim,
                                         uint32_t n_tok,
                                         float scale) {
    bool ok = false;
    if (!m || !t || !x || !out || n_tok == 0) return false;
    if (t->ndim == 1) {
        if (out_dim != 1 || t->dim[0] != in_dim) return false;
    } else if (t->ndim == 2) {
        if (t->dim[0] != in_dim || t->dim[1] != out_dim) return false;
    } else {
        return false;
    }
    switch (t->type) {
    case Q36_TENSOR_F32:
        ok = q36_gpu_matmul_f32_scaled_tensor(out, m->map, m->size, t->abs_offset,
                                              in_dim, out_dim, x, n_tok, scale) != 0;
        break;
    case Q36_TENSOR_F16:
        ok = q36_gpu_matmul_f16_tensor(out, m->map, m->size, t->abs_offset,
                                       in_dim, out_dim, x, n_tok) != 0;
        if (ok && scale != 1.0f) ok = q36_gpu_tensor_scale_host(out, n_tok * out_dim, scale);
        break;
    case Q36_TENSOR_Q8_0:
        ok = q36_gpu_matmul_q8_0_scaled_tensor(out, m->map, m->size, t->abs_offset,
                                               in_dim, out_dim, x, n_tok, scale) != 0;
        break;
    case Q36_TENSOR_Q2_K:
    case Q36_TENSOR_Q4_K:
    case Q36_TENSOR_Q5_K:
    case Q36_TENSOR_Q6_K:
        ok = q36_gpu_matmul_k_quant_scaled_tensor(out, m->map, m->size, t->abs_offset,
                                                  t->type, in_dim, out_dim, x, n_tok, scale) != 0;
        break;
    default:
        return false;
    }
    return ok;
}

static bool q36_gpu_tensor_matmul_q8_scaled(const q36_model *m,
                                            const q36_tensor *t,
                                            const q36_gpu_tensor *xq,
                                            q36_gpu_tensor *out,
                                            uint32_t in_dim,
                                            uint32_t out_dim,
                                            uint32_t n_tok,
                                            float scale) {
    if (!m || !t || !xq || !out || n_tok == 0) return false;
    if (t->ndim == 1) {
        if (out_dim != 1 || t->dim[0] != in_dim) return false;
    } else if (t->ndim == 2) {
        if (t->dim[0] != in_dim || t->dim[1] != out_dim) return false;
    } else {
        return false;
    }
    switch (t->type) {
    case Q36_TENSOR_Q2_K:
    case Q36_TENSOR_Q4_K:
    case Q36_TENSOR_Q5_K:
    case Q36_TENSOR_Q6_K:
        return q36_gpu_matmul_k_quant_q8_scaled_tensor(out, m->map, m->size, t->abs_offset,
                                                        t->type, in_dim, out_dim, xq, n_tok, scale) != 0;
    default:
        return false;
    }
}

static bool q36_gpu_tensor_matmul_q8_or_float_scaled(const q36_model *m,
                                                     const q36_tensor *t,
                                                     const q36_gpu_tensor *x,
                                                     const q36_gpu_tensor *xq,
                                                     q36_gpu_tensor *out,
                                                     uint32_t in_dim,
                                                     uint32_t out_dim,
                                                     uint32_t n_tok,
                                                     float scale) {
    if (!t) return false;
    if (t->type == Q36_TENSOR_Q8_0) {
        return q36_gpu_tensor_matmul_scaled(m, t, x, out, in_dim, out_dim, n_tok, scale);
    }
    return q36_gpu_tensor_matmul_q8_scaled(m, t, xq, out, in_dim, out_dim, n_tok, scale);
}

static bool q36_vulkan_runtime_reset(q36_vulkan_runtime *rt) {
    uint64_t state_dim = (uint64_t)Q36_N_SSM_STATE * Q36_N_SSM_STATE * Q36_N_SSM_DT_RANK;
    if (!rt) return false;
    if (!q36_gpu_tensor_zero(rt->hidden) || !q36_gpu_tensor_zero(rt->next_hidden)) return false;
    if (rt->last_h && !q36_gpu_tensor_zero(rt->last_h)) return false;
    for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
        if (rt->recurrent[il].conv && !q36_gpu_tensor_zero(rt->recurrent[il].conv)) return false;
        if (rt->recurrent[il].state && !q36_gpu_tensor_zero(rt->recurrent[il].state)) return false;
        if (rt->recurrent[il].state && q36_gpu_tensor_bytes(rt->recurrent[il].state) != state_dim * sizeof(float)) return false;
    }
    return true;
}

static void q36_vulkan_runtime_free(q36_vulkan_runtime *rt) {
    if (!rt) return;
    for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
        q36_gpu_tensor_free(rt->full[il].k);
        q36_gpu_tensor_free(rt->full[il].v);
        q36_gpu_tensor_free(rt->recurrent[il].conv);
        q36_gpu_tensor_free(rt->recurrent[il].state);
        q36_gpu_tensor_free(rt->spec_recurrent[il].conv);
        q36_gpu_tensor_free(rt->spec_recurrent[il].state);
    }
    q36_gpu_tensor_free(rt->hidden);
    q36_gpu_tensor_free(rt->next_hidden);
    q36_gpu_tensor_free(rt->embed_stage[0]);
    q36_gpu_tensor_free(rt->embed_stage[1]);
    q36_gpu_tensor_free(rt->norm);
    q36_gpu_tensor_free(rt->last_h);
    q36_gpu_tensor_free(rt->inp_q8);
    q36_gpu_tensor_free(rt->attn_qg);
    q36_gpu_tensor_free(rt->attn_q);
    q36_gpu_tensor_free(rt->attn_k);
    q36_gpu_tensor_free(rt->attn_v);
    q36_gpu_tensor_free(rt->attn_out);
    q36_gpu_tensor_free(rt->recur_qkv);
    q36_gpu_tensor_free(rt->recur_window);
    q36_gpu_tensor_free(rt->recur_conv);
    q36_gpu_tensor_free(rt->recur_z);
    q36_gpu_tensor_free(rt->recur_alpha);
    q36_gpu_tensor_free(rt->recur_beta);
    q36_gpu_tensor_free(rt->recur_gb);
    q36_gpu_tensor_free(rt->recur_q);
    q36_gpu_tensor_free(rt->recur_k);
    q36_gpu_tensor_free(rt->recur_v);
    q36_gpu_tensor_free(rt->recur_proj);
    q36_gpu_tensor_free(rt->ffn_gate_logits);
    q36_gpu_tensor_free(rt->ffn_selected);
    q36_gpu_tensor_free(rt->ffn_weights);
    q36_gpu_tensor_free(rt->ffn_shared_gate);
    q36_gpu_tensor_free(rt->ffn_shared_up);
    q36_gpu_tensor_free(rt->ffn_shared_mid);
    q36_gpu_tensor_free(rt->ffn_shared_out);
    q36_gpu_tensor_free(rt->ffn_scalar);
    q36_gpu_tensor_free(rt->logits);
    q36_gpu_tensor_free(rt->scores);
    q36_gpu_tensor_free(rt->mtp_full.k);
    q36_gpu_tensor_free(rt->mtp_full.v);
    q36_gpu_tensor_free(rt->spec_logits);
    q36_gpu_tensor_free(rt->mtp_tok_embd);
    q36_gpu_tensor_free(rt->mtp_e_norm);
    q36_gpu_tensor_free(rt->mtp_h_norm);
    q36_gpu_tensor_free(rt->mtp_concat);
    q36_gpu_tensor_free(rt->mtp_cur);
    q36_gpu_tensor_free(rt->mtp_next);
    q36_gpu_tensor_free(rt->mtp_head);
    q36_gpu_tensor_free(rt->mtp_logits);
    free(rt);
}

static q36_vulkan_runtime *q36_vulkan_runtime_create(int ctx_size,
                                                     uint32_t prefill_cap,
                                                     bool enable_mtp,
                                                     bool quality,
                                                     bool ssd_streaming,
                                                     q36_kv_cache_type cache_type_k,
                                                     q36_kv_cache_type cache_type_v) {
    q36_vulkan_runtime *rt;
    uint64_t state_dim;
    uint64_t hist_rows;
    if (ctx_size <= 0 || !q36_gpu_init()) return NULL;
    uint32_t k_row_bytes = q36_kv_cache_row_bytes(cache_type_k, Q36_N_HEAD_KV * Q36_N_HEAD_DIM);
    uint32_t v_row_bytes = q36_kv_cache_row_bytes(cache_type_v, Q36_N_HEAD_KV * Q36_N_VALUE_DIM);
    if (!k_row_bytes || !v_row_bytes) return NULL;
    rt = xcalloc(1, sizeof(*rt));
    if (prefill_cap < 1) prefill_cap = 1;
    rt->prefill_cap = prefill_cap;
    rt->mtp_enabled = enable_mtp;
    {
        const char *env = getenv("Q36_VK_RECURRENT_CONV_DECODE");
        rt->recur_conv_fused = !quality && !ssd_streaming &&
                               (!env || !env[0] || env[0] != '0');
    }
    state_dim = (uint64_t)Q36_N_SSM_STATE * Q36_N_SSM_STATE * Q36_N_SSM_DT_RANK;
    hist_rows = Q36_N_SSM_CONV - 1u;
    /* Activation scratch carries prefill_cap token rows for batched prefill;
     * single-token decode just uses row 0. */
#define Q36_GPU_ALLOC_F32(field, n) do { \
        rt->field = q36_gpu_tensor_alloc((uint64_t)(n) * prefill_cap * sizeof(float)); \
        if (!rt->field) goto fail; \
    } while (0)
#define Q36_GPU_ALLOC_U32(field, n) do { \
        rt->field = q36_gpu_tensor_alloc((uint64_t)(n) * prefill_cap * sizeof(uint32_t)); \
        if (!rt->field) goto fail; \
    } while (0)
#define Q36_GPU_ALLOC_MTP_F32(field, n) do { \
        rt->field = q36_gpu_tensor_alloc((uint64_t)(n) * sizeof(float)); \
        if (!rt->field) goto fail; \
    } while (0)
    Q36_GPU_ALLOC_F32(hidden, Q36_N_EMBD);
    Q36_GPU_ALLOC_F32(next_hidden, Q36_N_EMBD);
    Q36_GPU_ALLOC_F32(embed_stage[0], Q36_N_EMBD);
    Q36_GPU_ALLOC_F32(embed_stage[1], Q36_N_EMBD);
    Q36_GPU_ALLOC_F32(norm, Q36_N_EMBD);
    rt->last_h = q36_gpu_tensor_alloc((uint64_t)Q36_N_EMBD * sizeof(float));
    if (!rt->last_h) goto fail;
    rt->inp_q8 = q36_gpu_tensor_alloc((uint64_t)((Q36_N_EMBD + Q36_QK_K - 1u) / Q36_QK_K) * Q36_VK_Q8_K_BYTES * prefill_cap);
    if (!rt->inp_q8) goto fail;
    Q36_GPU_ALLOC_F32(attn_qg, Q36_N_HEAD * Q36_N_HEAD_DIM * 2u);
    Q36_GPU_ALLOC_F32(attn_q, Q36_N_HEAD * Q36_N_HEAD_DIM);
    Q36_GPU_ALLOC_F32(attn_k, Q36_N_HEAD_KV * Q36_N_HEAD_DIM);
    Q36_GPU_ALLOC_F32(attn_v, Q36_N_HEAD_KV * Q36_N_VALUE_DIM);
    Q36_GPU_ALLOC_F32(attn_out, Q36_N_SSM_INNER);
    Q36_GPU_ALLOC_F32(recur_qkv, Q36_N_SSM_CONV_DIM);
    if (!rt->recur_conv_fused)
        Q36_GPU_ALLOC_F32(recur_window, (uint64_t)Q36_N_SSM_CONV * Q36_N_SSM_CONV_DIM);
    Q36_GPU_ALLOC_F32(recur_conv, Q36_N_SSM_CONV_DIM);
    Q36_GPU_ALLOC_F32(recur_z, Q36_N_SSM_INNER);
    Q36_GPU_ALLOC_F32(recur_alpha, Q36_N_SSM_DT_RANK);
    Q36_GPU_ALLOC_F32(recur_beta, Q36_N_SSM_DT_RANK);
    Q36_GPU_ALLOC_F32(recur_gb, Q36_N_SSM_DT_RANK * 2u);
    Q36_GPU_ALLOC_F32(recur_q, Q36_N_SSM_INNER);
    Q36_GPU_ALLOC_F32(recur_k, Q36_N_SSM_INNER);
    Q36_GPU_ALLOC_F32(recur_v, Q36_N_SSM_INNER);
    Q36_GPU_ALLOC_F32(recur_proj, Q36_N_SSM_INNER);
    Q36_GPU_ALLOC_F32(ffn_gate_logits, Q36_N_EXPERT);
    Q36_GPU_ALLOC_U32(ffn_selected, Q36_N_EXPERT_USED);
    Q36_GPU_ALLOC_F32(ffn_weights, Q36_N_EXPERT_USED);
    Q36_GPU_ALLOC_F32(ffn_shared_gate, Q36_N_FF_SHARED);
    Q36_GPU_ALLOC_F32(ffn_shared_up, Q36_N_FF_SHARED);
    Q36_GPU_ALLOC_F32(ffn_shared_mid, Q36_N_FF_SHARED);
    Q36_GPU_ALLOC_F32(ffn_shared_out, Q36_N_EMBD);
    Q36_GPU_ALLOC_F32(ffn_scalar, 1);
    /* The fused attention path never touches the scores scratch; skipping it
     * saves ctx_size * n_head * prefill_cap floats (128 MiB at ctx 32k). */
    if (!q36_gpu_attn_fused_enabled())
        Q36_GPU_ALLOC_F32(scores, (uint64_t)ctx_size * Q36_N_HEAD);
    rt->logits = q36_gpu_tensor_alloc((uint64_t)Q36_N_VOCAB * sizeof(float));
    if (!rt->logits) goto fail;
    if (enable_mtp) {
        rt->mtp_full.cap = (uint32_t)ctx_size;
        rt->mtp_full.type_k = cache_type_k;
        rt->mtp_full.type_v = cache_type_v;
        rt->mtp_full.k_row_bytes = k_row_bytes;
        rt->mtp_full.v_row_bytes = v_row_bytes;
        rt->mtp_full.k = q36_gpu_tensor_alloc((uint64_t)ctx_size * k_row_bytes);
        rt->mtp_full.v = q36_gpu_tensor_alloc((uint64_t)ctx_size * v_row_bytes);
        if (!rt->mtp_full.k || !rt->mtp_full.v) goto fail;
        rt->spec_logits = q36_gpu_tensor_alloc((uint64_t)Q36_MTP_MAX_DRAFT * Q36_N_VOCAB * sizeof(float));
        if (!rt->spec_logits) goto fail;
        Q36_GPU_ALLOC_MTP_F32(mtp_tok_embd, Q36_N_EMBD);
        Q36_GPU_ALLOC_MTP_F32(mtp_e_norm, Q36_N_EMBD);
        Q36_GPU_ALLOC_MTP_F32(mtp_h_norm, Q36_N_EMBD);
        Q36_GPU_ALLOC_MTP_F32(mtp_concat, Q36_N_EMBD * 2u);
        Q36_GPU_ALLOC_MTP_F32(mtp_cur, Q36_N_EMBD);
        Q36_GPU_ALLOC_MTP_F32(mtp_next, Q36_N_EMBD);
        Q36_GPU_ALLOC_MTP_F32(mtp_head, Q36_N_EMBD);
        Q36_GPU_ALLOC_MTP_F32(mtp_logits, Q36_N_VOCAB);
    }
    for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
        if (q36_layer_is_full_attention(il)) {
            rt->full[il].cap = (uint32_t)ctx_size;
            rt->full[il].type_k = cache_type_k;
            rt->full[il].type_v = cache_type_v;
            rt->full[il].k_row_bytes = k_row_bytes;
            rt->full[il].v_row_bytes = v_row_bytes;
            rt->full[il].k = q36_gpu_tensor_alloc((uint64_t)ctx_size * k_row_bytes);
            rt->full[il].v = q36_gpu_tensor_alloc((uint64_t)ctx_size * v_row_bytes);
            if (!rt->full[il].k || !rt->full[il].v) goto fail;
        } else {
            rt->recurrent[il].conv = q36_gpu_tensor_alloc(hist_rows * Q36_N_SSM_CONV_DIM * sizeof(float));
            rt->recurrent[il].state = q36_gpu_tensor_alloc(state_dim * sizeof(float));
            if (!rt->recurrent[il].conv || !rt->recurrent[il].state) goto fail;
            if (enable_mtp) {
                rt->spec_recurrent[il].conv = q36_gpu_tensor_alloc(hist_rows * Q36_N_SSM_CONV_DIM * sizeof(float));
                rt->spec_recurrent[il].state = q36_gpu_tensor_alloc(state_dim * sizeof(float));
                if (!rt->spec_recurrent[il].conv || !rt->spec_recurrent[il].state) goto fail;
            }
        }
    }
    if (!q36_vulkan_runtime_reset(rt)) goto fail;
#undef Q36_GPU_ALLOC_F32
#undef Q36_GPU_ALLOC_U32
#undef Q36_GPU_ALLOC_MTP_F32
    return rt;
fail:
#undef Q36_GPU_ALLOC_F32
#undef Q36_GPU_ALLOC_U32
#undef Q36_GPU_ALLOC_MTP_F32
    q36_vulkan_runtime_free(rt);
    return NULL;
}
#endif

static bool q36_embed_token(const q36_engine *e, int token, float *out) {
    if (!e || !out || token < 0 || token >= Q36_N_VOCAB) return false;
    return q36_tensor_row_to_float(&e->model, e->weights.token_embd, (uint64_t)token, out, Q36_N_EMBD);
}

/* Multi-section M-RoPE rotation, shared by the CPU reference and the Vulkan
 * scheduler (q36_vulkan_rope_heads): both engines must go through this one
 * compiled body, because a second copy of the rotation fuses differently
 * under -ffast-math and drifts by ulps. */
static void q36_apply_rope_one(float *x, uint32_t head_dim, uint32_t pos) {
    const uint32_t n_rot = Q36_N_ROT;
    const uint32_t half = n_rot / 2u;
    const float base = 10000000.0f;
    const uint32_t sect_dims = Q36_ROPE_SECTIONS[0] + Q36_ROPE_SECTIONS[1] + Q36_ROPE_SECTIONS[2] + Q36_ROPE_SECTIONS[3];
    const float theta_scale = powf(base, -2.0f / (float)n_rot);
    float theta[4] = {(float)pos, (float)pos, (float)pos, 0.0f};
    if (head_dim < n_rot) return;
    if (sect_dims == 0 || sect_dims > half) return;
    for (uint32_t pair = 0; pair < half; pair++) {
        uint32_t axis = 3;
        uint32_t sector = pair % sect_dims;
        if (sector < Q36_ROPE_SECTIONS[0]) {
            axis = 0;
        } else if (sector < Q36_ROPE_SECTIONS[0] + Q36_ROPE_SECTIONS[1]) {
            axis = 1;
        } else if (sector < Q36_ROPE_SECTIONS[0] + Q36_ROPE_SECTIONS[1] + Q36_ROPE_SECTIONS[2]) {
            axis = 2;
        }
        {
            float c = cosf(theta[axis]);
            float s = sinf(theta[axis]);
            float x0 = x[pair];
            float x1 = x[pair + half];
            x[pair] = x0 * c - x1 * s;
            x[pair + half] = x0 * s + x1 * c;
        }
        for (uint32_t i = 0; i < 4; i++) theta[i] *= theta_scale;
    }
}

static void q36_full_attn_cache_store(q36_full_attn_cache *cache, uint32_t pos, const float *k, const float *v) {
    uint8_t *dstk;
    uint8_t *dstv;
    if (!cache || pos >= cache->cap) return;
    dstk = cache->k + (uint64_t)pos * cache->k_row_bytes;
    dstv = cache->v + (uint64_t)pos * cache->v_row_bytes;
    q36_kv_cache_store_row(dstk, cache->type_k, k, Q36_N_HEAD_KV * Q36_N_HEAD_DIM);
    q36_kv_cache_store_row(dstv, cache->type_v, v, Q36_N_HEAD_KV * Q36_N_VALUE_DIM);
    if (cache->len <= pos) cache->len = pos + 1u;
}

static void q36_recurrent_conv_step(q36_recurrent_cache *cache, const float *cur, float *window) {
    uint32_t hist = Q36_N_SSM_CONV - 1u;
    for (uint32_t i = 0; i < hist; i++) {
        memcpy(window + (uint64_t)i * Q36_N_SSM_CONV_DIM,
               cache->conv + (uint64_t)i * Q36_N_SSM_CONV_DIM,
               (size_t)Q36_N_SSM_CONV_DIM * sizeof(float));
    }
    memcpy(window + (uint64_t)hist * Q36_N_SSM_CONV_DIM, cur, (size_t)Q36_N_SSM_CONV_DIM * sizeof(float));
    if (hist > 1u) {
        memmove(cache->conv,
                cache->conv + Q36_N_SSM_CONV_DIM,
                (size_t)(hist - 1u) * Q36_N_SSM_CONV_DIM * sizeof(float));
    }
    memcpy(cache->conv + (uint64_t)(hist - 1u) * Q36_N_SSM_CONV_DIM, cur, (size_t)Q36_N_SSM_CONV_DIM * sizeof(float));
}

static void q36_ssm_conv_apply(const float *window, const float *kernel, float *out) {
    for (uint32_t c = 0; c < Q36_N_SSM_CONV_DIM; c++) {
        double acc = 0.0;
        for (uint32_t t = 0; t < Q36_N_SSM_CONV; t++) {
            acc += (double)window[(uint64_t)t * Q36_N_SSM_CONV_DIM + c] * (double)kernel[(uint64_t)c * Q36_N_SSM_CONV + t];
        }
        out[c] = (float)acc;
    }
}

static void q36_ssm_conv_apply_f32(const float *window, const float *kernel, float *out) {
    for (uint32_t c = 0; c < Q36_N_SSM_CONV_DIM; c++) {
        float acc = 0.0f;
        for (uint32_t t = 0; t < Q36_N_SSM_CONV; t++) {
            acc += window[(uint64_t)t * Q36_N_SSM_CONV_DIM + c] * kernel[(uint64_t)c * Q36_N_SSM_CONV + t];
        }
        out[c] = acc;
    }
}

static void q36_delta_net_decode(float *state, const float *q, const float *k, const float *v,
                                 const float *g, const float *b, float *out) {
    for (uint32_t h = 0; h < Q36_N_SSM_DT_RANK; h++) {
        float *s = state + (uint64_t)h * Q36_N_SSM_STATE * Q36_N_SSM_STATE;
        const float *qh = q + (uint64_t)h * Q36_N_SSM_STATE;
        const float *kh = k + (uint64_t)h * Q36_N_SSM_STATE;
        const float *vh = v + (uint64_t)h * Q36_N_SSM_STATE;
        float decay = expf(g[h]);
        float beta = b[h];
        float sk[Q36_N_SSM_STATE];
        float d[Q36_N_SSM_STATE];
        for (uint32_t i = 0; i < Q36_N_SSM_STATE; i++) {
            for (uint32_t j = 0; j < Q36_N_SSM_STATE; j++) s[(uint64_t)i * Q36_N_SSM_STATE + j] *= decay;
        }
        for (uint32_t j = 0; j < Q36_N_SSM_STATE; j++) {
            double acc = 0.0;
            for (uint32_t i = 0; i < Q36_N_SSM_STATE; i++) acc += (double)s[(uint64_t)i * Q36_N_SSM_STATE + j] * (double)kh[i];
            sk[j] = (float)acc;
            d[j] = (vh[j] - sk[j]) * beta;
        }
        for (uint32_t i = 0; i < Q36_N_SSM_STATE; i++) {
            float ki = kh[i];
            for (uint32_t j = 0; j < Q36_N_SSM_STATE; j++) s[(uint64_t)i * Q36_N_SSM_STATE + j] += ki * d[j];
        }
        for (uint32_t j = 0; j < Q36_N_SSM_STATE; j++) {
            double acc = 0.0;
            for (uint32_t i = 0; i < Q36_N_SSM_STATE; i++) acc += (double)s[(uint64_t)i * Q36_N_SSM_STATE + j] * (double)qh[i];
            out[(uint64_t)h * Q36_N_SSM_STATE + j] = (float)(acc / sqrt((double)Q36_N_SSM_STATE));
        }
    }
}

static bool q36_forward_ffn(const q36_engine *e, const q36_layer_weights *l, const float *inp, float *out,
                            q36_cpu_runtime *rt) {
    float gate_logits[Q36_N_EXPERT];
    uint32_t top_idx[Q36_N_EXPERT_USED];
    float top_val[Q36_N_EXPERT_USED];
    float top_w[Q36_N_EXPERT_USED];
    uint8_t inpq[Q36_MAX_Q8_K_BYTES];
    const uint8_t *xq;
    q36_activation_quant_kind inpq_kind;
    float *gate = rt->work2;
    float *up = rt->work3;
    float *mid = rt->work4;
    float *rowbuf = rt->work5;
    float shared_gate;
    if (!q36_tensor_matvec(e, l->ffn_gate_inp, inp, gate_logits, rowbuf, Q36_N_EMBD, Q36_N_EXPERT)) return false;
    inpq_kind = q36_activation_quant_kind_for_type(l->ffn_gate_exps->type);
    if (!q36_quantize_activation_row(inpq_kind, inp, inpq, Q36_N_EMBD)) return false;
    xq = inpq_kind == Q36_ACTIVATION_QUANT_NONE ? NULL : inpq;
    q36_softmax_inplace(gate_logits, Q36_N_EXPERT);
    q36_topk_indices(gate_logits, Q36_N_EXPERT, Q36_N_EXPERT_USED, top_idx, top_val);
    {
        double sum = 0.0;
        for (uint32_t i = 0; i < Q36_N_EXPERT_USED; i++) sum += top_val[i];
        if (sum < 6.103515625e-5) sum = 6.103515625e-5;
        for (uint32_t i = 0; i < Q36_N_EXPERT_USED; i++) top_w[i] = (float)((double)top_val[i] / sum) * e->expert_weights_scale;
    }
    memset(out, 0, (size_t)Q36_N_EMBD * sizeof(float));
    for (uint32_t i = 0; i < Q36_N_EXPERT_USED; i++) {
        float gate_scale = q36_tensor_index_or(&e->model, l->ffn_gate_exps_scale, top_idx[i], 1.0f);
        float up_scale = q36_tensor_index_or(&e->model, l->ffn_up_exps_scale, top_idx[i], 1.0f);
        if (!q36_tensor_expert_matvec_pair_prequant(e,
                                                    l->ffn_gate_exps,
                                                    l->ffn_up_exps,
                                                    top_idx[i],
                                                    inp,
                                                    xq,
                                                    gate,
                                                    up,
                                                    rowbuf,
                                                    Q36_N_EMBD,
                                                    Q36_N_FF_EXP,
                                                    gate_scale,
                                                    up_scale)) {
            if (!q36_tensor_expert_matvec_prequant(e, l->ffn_gate_exps, top_idx[i], inp, xq,
                                                   gate, rowbuf, Q36_N_EMBD, Q36_N_FF_EXP)) {
                return false;
            }
            q36_scale_inplace(gate, Q36_N_FF_EXP, gate_scale);
            if (!q36_tensor_expert_matvec_prequant(e, l->ffn_up_exps, top_idx[i], inp, xq,
                                                   up, rowbuf, Q36_N_EMBD, Q36_N_FF_EXP)) {
                return false;
            }
            q36_scale_inplace(up, Q36_N_FF_EXP, up_scale);
        }
        for (uint32_t j = 0; j < Q36_N_FF_EXP; j++) mid[j] = q36_siluf(gate[j]) * up[j];
        if (!q36_tensor_expert_matvec(e, l->ffn_down_exps, top_idx[i], mid, gate, rowbuf, Q36_N_FF_EXP, Q36_N_EMBD)) return false;
        q36_scale_inplace(gate, Q36_N_EMBD,
                          q36_tensor_index_or(&e->model, l->ffn_down_exps_scale, top_idx[i], 1.0f));
        for (uint32_t j = 0; j < Q36_N_EMBD; j++) out[j] += gate[j] * top_w[i];
    }
    {
        float gate_scale = q36_tensor_scalar_or(&e->model, l->ffn_gate_shexp_scale, 1.0f);
        float up_scale = q36_tensor_scalar_or(&e->model, l->ffn_up_shexp_scale, 1.0f);
        q36_activation_quant_kind shared_kind = q36_activation_quant_kind_for_type(l->ffn_gate_shexp->type);
        if (shared_kind != inpq_kind) {
            if (!q36_quantize_activation_row(shared_kind, inp, inpq, Q36_N_EMBD)) return false;
            xq = shared_kind == Q36_ACTIVATION_QUANT_NONE ? NULL : inpq;
            inpq_kind = shared_kind;
        }
        if (!q36_tensor_matvec_pair_prequant(e,
                                             l->ffn_gate_shexp,
                                             l->ffn_up_shexp,
                                             inp,
                                             xq,
                                             gate,
                                             up,
                                             rowbuf,
                                             Q36_N_EMBD,
                                             Q36_N_FF_SHARED,
                                             gate_scale,
                                             up_scale)) {
            if (!q36_tensor_matvec_prequant(e, l->ffn_gate_shexp, inp, xq,
                                            gate, rowbuf, Q36_N_EMBD, Q36_N_FF_SHARED)) {
                return false;
            }
            q36_scale_inplace(gate, Q36_N_FF_SHARED, gate_scale);
            if (!q36_tensor_matvec_prequant(e, l->ffn_up_shexp, inp, xq,
                                            up, rowbuf, Q36_N_EMBD, Q36_N_FF_SHARED)) {
                return false;
            }
            q36_scale_inplace(up, Q36_N_FF_SHARED, up_scale);
        }
    }
    for (uint32_t j = 0; j < Q36_N_FF_SHARED; j++)
        mid[j] = gate[j] / (1.0f + expf(-gate[j])) * up[j];
    if (!q36_tensor_matvec(e, l->ffn_down_shexp, mid, gate, rowbuf, Q36_N_FF_SHARED, Q36_N_EMBD)) return false;
    q36_scale_inplace(gate, Q36_N_EMBD,
                      q36_tensor_scalar_or(&e->model, l->ffn_down_shexp_scale, 1.0f));
    if (!q36_tensor_get_plain(&e->model, l->ffn_gate_inp_shexp, rowbuf, Q36_N_EMBD)) return false;
    {
        double acc = 0.0;
        for (uint32_t i = 0; i < Q36_N_EMBD; i++) acc += (double)rowbuf[i] * (double)inp[i];
        up[0] = (float)acc;
    }
    shared_gate = q36_sigmoidf(up[0]);
    for (uint32_t j = 0; j < Q36_N_EMBD; j++) out[j] += gate[j] * shared_gate;
    return true;
}

static bool q36_forward_full_attn(const q36_engine *e, const q36_layer_weights *l, uint32_t il, uint32_t pos,
                                  const float *inp, float *out, q36_cpu_runtime *rt) {
    float *qg = rt->work5;
    float *q = rt->work0;
    float *k = rt->work1;
    float *v = rt->work2;
    float *acc = rt->work3;
    float sinks[Q36_N_HEAD];
    q36_full_attn_cache *cache = &rt->full[il];
    bool have_sinks = false;
    if (!q36_tensor_matvec(e, l->attn_q, inp, qg, rt->work4, Q36_N_EMBD, Q36_N_HEAD * Q36_N_HEAD_DIM * 2u)) {
        fprintf(stderr, "q36: full_attn attn_q failed at layer=%u pos=%u\n", il, pos);
        return false;
    }
    q36_scale_inplace(qg, Q36_N_HEAD * Q36_N_HEAD_DIM * 2u,
                      q36_tensor_scalar_or(&e->model, l->attn_q_scale, 1.0f));
    if (!q36_tensor_matvec(e, l->attn_k, inp, k, rt->work4, Q36_N_EMBD, Q36_N_HEAD_KV * Q36_N_HEAD_DIM)) {
        fprintf(stderr, "q36: full_attn attn_k failed at layer=%u pos=%u\n", il, pos);
        return false;
    }
    q36_scale_inplace(k, Q36_N_HEAD_KV * Q36_N_HEAD_DIM,
                      q36_tensor_scalar_or(&e->model, l->attn_k_scale, 1.0f));
    if (!q36_tensor_matvec(e, l->attn_v, inp, v, rt->work4, Q36_N_EMBD, Q36_N_HEAD_KV * Q36_N_VALUE_DIM)) {
        fprintf(stderr, "q36: full_attn attn_v failed at layer=%u pos=%u\n", il, pos);
        return false;
    }
    q36_scale_inplace(v, Q36_N_HEAD_KV * Q36_N_VALUE_DIM,
                      q36_tensor_scalar_or(&e->model, l->attn_v_scale, 1.0f));
    for (uint32_t h = 0; h < Q36_N_HEAD; h++) {
        q36_ref_rms_norm(q + (uint64_t)h * Q36_N_HEAD_DIM,
                         qg + (uint64_t)h * Q36_N_HEAD_DIM * 2u,
                         (const float *)(e->model.map + l->attn_q_norm->abs_offset),
                         Q36_N_HEAD_DIM,
                         Q36_RMS_EPS);
        q36_apply_rope_one(q + (uint64_t)h * Q36_N_HEAD_DIM, Q36_N_HEAD_DIM, pos);
    }
    for (uint32_t h = 0; h < Q36_N_HEAD_KV; h++) {
        q36_ref_rms_norm(k + (uint64_t)h * Q36_N_HEAD_DIM,
                         k + (uint64_t)h * Q36_N_HEAD_DIM,
                         (const float *)(e->model.map + l->attn_k_norm->abs_offset),
                         Q36_N_HEAD_DIM,
                         Q36_RMS_EPS);
        q36_apply_rope_one(k + (uint64_t)h * Q36_N_HEAD_DIM, Q36_N_HEAD_DIM, pos);
    }
    if (l->attn_sinks) have_sinks = q36_tensor_get_plain(&e->model, l->attn_sinks, sinks, Q36_N_HEAD);
    q36_full_attn_cache_store(cache, pos, k, v);
    for (uint32_t h = 0; h < Q36_N_HEAD; h++) {
        uint32_t kvh = h / (Q36_N_HEAD / Q36_N_HEAD_KV);
        float *head_out = acc + (uint64_t)h * Q36_N_VALUE_DIM;
        float max_score = have_sinks ? sinks[h] : -INFINITY;
        double denom = 0.0;
        memset(head_out, 0, (size_t)Q36_N_VALUE_DIM * sizeof(float));
        for (uint32_t t = 0; t <= pos && t < cache->len; t++) {
            const uint8_t *kc = cache->k + (uint64_t)t * cache->k_row_bytes + (uint64_t)kvh * q36_kv_cache_row_bytes(cache->type_k, Q36_N_HEAD_DIM);
            double dot = 0.0;
            for (uint32_t i = 0; i < Q36_N_HEAD_DIM; i++) dot += (double)q[(uint64_t)h * Q36_N_HEAD_DIM + i] * (double)q36_kv_cache_at(kc, cache->type_k, i);
            rt->scores[t] = (float)(dot / sqrt((double)Q36_N_HEAD_DIM));
            if (rt->scores[t] > max_score) max_score = rt->scores[t];
        }
        if (have_sinks) denom = exp((double)sinks[h] - (double)max_score);
        for (uint32_t t = 0; t <= pos && t < cache->len; t++) {
            const uint8_t *vc = cache->v + (uint64_t)t * cache->v_row_bytes + (uint64_t)kvh * q36_kv_cache_row_bytes(cache->type_v, Q36_N_VALUE_DIM);
            float w = expf(rt->scores[t] - max_score);
            denom += w;
            for (uint32_t i = 0; i < Q36_N_VALUE_DIM; i++) head_out[i] += w * q36_kv_cache_at(vc, cache->type_v, i);
        }
        if (denom > 0.0) {
            float inv = (float)(1.0 / denom);
            for (uint32_t i = 0; i < Q36_N_VALUE_DIM; i++) head_out[i] *= inv;
        }
        for (uint32_t i = 0; i < Q36_N_VALUE_DIM; i++) {
            uint64_t gate_off = (uint64_t)h * Q36_N_HEAD_DIM * 2u + Q36_N_HEAD_DIM + i;
            head_out[i] *= q36_sigmoidf(qg[gate_off]);
        }
    }
    if (!q36_tensor_matvec(e, l->attn_output, acc, out, rt->work4, Q36_N_SSM_INNER, Q36_N_EMBD)) {
        fprintf(stderr, "q36: full_attn attn_output failed at layer=%u pos=%u\n", il, pos);
        return false;
    }
    q36_scale_inplace(out, Q36_N_EMBD,
                      q36_tensor_scalar_or(&e->model, l->attn_output_scale, 1.0f));
    return true;
}

static bool q36_forward_recurrent(const q36_engine *e, const q36_layer_weights *l, uint32_t il,
                                  const float *inp, float *out, q36_cpu_runtime *rt) {
    q36_recurrent_cache *cache = &rt->recurrent[il];
    float *qkv = rt->work5;
    float *window = rt->work4;
    float *conv = rt->work3;
    float *z = rt->work2;
    float *beta = rt->work1;
    float *q = rt->work5;
    float *k = rt->work4;
    float *v = rt->work0;
    float *proj = rt->work3;
    float *gate = rt->work1 + Q36_N_SSM_DT_RANK;
    const float *kernel = (const float *)(e->model.map + l->ssm_conv1d->abs_offset);
    if (!q36_tensor_matvec(e, l->attn_qkv, inp, qkv, z, Q36_N_EMBD, Q36_N_SSM_CONV_DIM)) {
        fprintf(stderr, "q36: recurrent attn_qkv failed at layer=%u\n", il);
        return false;
    }
    q36_scale_inplace(qkv, Q36_N_SSM_CONV_DIM,
                      q36_tensor_scalar_or(&e->model, l->attn_qkv_scale, 1.0f));
    q36_recurrent_conv_step(cache, qkv, window);
    q36_ssm_conv_apply_f32(window, kernel, conv);
    for (uint32_t i = 0; i < Q36_N_SSM_CONV_DIM; i++) conv[i] = q36_siluf(conv[i]);
    if (!q36_all_finite(conv, Q36_N_SSM_CONV_DIM)) {
        fprintf(stderr, "q36: recurrent conv non-finite at layer=%u\n", il);
        return false;
    }
    if (!q36_tensor_matvec(e, l->ssm_beta, inp, beta, z, Q36_N_EMBD, Q36_N_SSM_DT_RANK)) {
        fprintf(stderr, "q36: recurrent ssm_beta failed at layer=%u\n", il);
        return false;
    }
    q36_scale_inplace(beta, Q36_N_SSM_DT_RANK,
                      q36_tensor_scalar_or(&e->model, l->ssm_beta_scale, 1.0f));
    if (!q36_tensor_matvec(e, l->ssm_alpha, inp, gate, z, Q36_N_EMBD, Q36_N_SSM_DT_RANK)) {
        fprintf(stderr, "q36: recurrent ssm_alpha failed at layer=%u\n", il);
        return false;
    }
    q36_scale_inplace(gate, Q36_N_SSM_DT_RANK,
                      q36_tensor_scalar_or(&e->model, l->ssm_alpha_scale, 1.0f));
    if (!q36_tensor_matvec(e, l->attn_gate, inp, z, window, Q36_N_EMBD, Q36_N_SSM_INNER)) {
        fprintf(stderr, "q36: recurrent attn_gate failed at layer=%u\n", il);
        return false;
    }
    q36_scale_inplace(z, Q36_N_SSM_INNER,
                      q36_tensor_scalar_or(&e->model, l->attn_gate_scale, 1.0f));
    for (uint32_t h = 0; h < Q36_N_SSM_DT_RANK; h++) {
        uint32_t gh = h % Q36_N_SSM_GROUP;
        memcpy(q + (uint64_t)h * Q36_N_SSM_STATE,
               conv + (uint64_t)gh * Q36_N_SSM_STATE,
               (size_t)Q36_N_SSM_STATE * sizeof(float));
        memcpy(k + (uint64_t)h * Q36_N_SSM_STATE,
               conv + (uint64_t)Q36_N_SSM_QK + (uint64_t)gh * Q36_N_SSM_STATE,
               (size_t)Q36_N_SSM_STATE * sizeof(float));
    }
    memcpy(v, conv + (uint64_t)Q36_N_SSM_QK * 2u,
           (size_t)Q36_N_SSM_DT_RANK * Q36_N_SSM_STATE * sizeof(float));
    for (uint32_t h = 0; h < Q36_N_SSM_DT_RANK; h++) {
        q36_l2_norm(q + (uint64_t)h * Q36_N_SSM_STATE, Q36_N_SSM_STATE, Q36_RMS_EPS);
        q36_l2_norm(k + (uint64_t)h * Q36_N_SSM_STATE, Q36_N_SSM_STATE, Q36_RMS_EPS);
        beta[h] = q36_sigmoidf(beta[h]);
        gate[h] = q36_softplusf(gate[h] + ((const float *)(e->model.map + l->ssm_dt->abs_offset))[h]) * ((const float *)(e->model.map + l->ssm_a->abs_offset))[h];
    }
    if (!q36_all_finite(beta, Q36_N_SSM_DT_RANK)) {
        fprintf(stderr, "q36: recurrent beta non-finite at layer=%u\n", il);
        return false;
    }
    if (!q36_all_finite(gate, Q36_N_SSM_DT_RANK)) {
        fprintf(stderr, "q36: recurrent gate non-finite at layer=%u\n", il);
        return false;
    }
    q36_delta_net_decode(cache->state, q, k, v, gate, beta, proj);
    if (!q36_all_finite(proj, Q36_N_SSM_INNER)) {
        fprintf(stderr, "q36: recurrent delta decode non-finite at layer=%u\n", il);
        return false;
    }
    for (uint32_t h = 0; h < Q36_N_SSM_DT_RANK; h++) {
        q36_ref_rms_norm(proj + (uint64_t)h * Q36_N_SSM_STATE,
                         proj + (uint64_t)h * Q36_N_SSM_STATE,
                         (const float *)(e->model.map + l->ssm_norm->abs_offset),
                         Q36_N_SSM_STATE,
                         Q36_RMS_EPS);
    }
    for (uint32_t i = 0; i < Q36_N_SSM_INNER; i++) proj[i] *= q36_siluf(z[i]);
    if (!q36_all_finite(proj, Q36_N_SSM_INNER)) {
        fprintf(stderr, "q36: recurrent gated proj non-finite at layer=%u\n", il);
        return false;
    }
    if (!q36_tensor_matvec(e, l->ssm_out, proj, out, q, Q36_N_SSM_INNER, Q36_N_EMBD)) {
        fprintf(stderr, "q36: recurrent ssm_out failed at layer=%u\n", il);
        return false;
    }
    q36_scale_inplace(out, Q36_N_EMBD,
                      q36_tensor_scalar_or(&e->model, l->ssm_out_scale, 1.0f));
    return true;
}

typedef struct {
    float *q;
    const float *qg;
    const float *weight;
    uint32_t pos0;
} q36_full_attn_q_rows_ctx;

typedef struct {
    float *k;
    const float *weight;
    uint32_t pos0;
} q36_full_attn_k_rows_ctx;

typedef struct {
    float *out;
    const float *q;
    const float *qg;
    const q36_full_attn_cache *cache;
    const float *sinks;
    uint32_t pos0;
    bool have_sinks;
} q36_full_attn_scores_ctx;

typedef struct {
    float *out;
    const q36_engine *e;
    const q36_layer_weights *l;
    const float *inp;
    float *gate_logits;
    const uint8_t *xq;
    uint32_t xq_row_bytes;
} q36_ffn_routed_batch_ctx;

typedef struct {
    float *out;
    const float *shared;
    const float *scalar;
} q36_ffn_combine_batch_ctx;

static void q36_full_attn_q_rows_worker(void *opaque, uint64_t row0, uint64_t row1) {
    q36_full_attn_q_rows_ctx *ctx = (q36_full_attn_q_rows_ctx *)opaque;
    for (uint64_t idx = row0; idx < row1; idx++) {
        uint32_t t = (uint32_t)(idx / Q36_N_HEAD);
        uint32_t h = (uint32_t)(idx % Q36_N_HEAD);
        const float *src = ctx->qg + (uint64_t)t * Q36_N_HEAD * Q36_N_HEAD_DIM * 2u + (uint64_t)h * Q36_N_HEAD_DIM * 2u;
        float *dst = ctx->q + (uint64_t)t * Q36_N_HEAD * Q36_N_HEAD_DIM + (uint64_t)h * Q36_N_HEAD_DIM;
        q36_ref_rms_norm(dst, src, ctx->weight, Q36_N_HEAD_DIM, Q36_RMS_EPS);
        q36_apply_rope_one(dst, Q36_N_HEAD_DIM, ctx->pos0 + t);
    }
}

static void q36_full_attn_k_rows_worker(void *opaque, uint64_t row0, uint64_t row1) {
    q36_full_attn_k_rows_ctx *ctx = (q36_full_attn_k_rows_ctx *)opaque;
    for (uint64_t idx = row0; idx < row1; idx++) {
        uint32_t t = (uint32_t)(idx / Q36_N_HEAD_KV);
        uint32_t h = (uint32_t)(idx % Q36_N_HEAD_KV);
        float *dst = ctx->k + (uint64_t)t * Q36_N_HEAD_KV * Q36_N_HEAD_DIM + (uint64_t)h * Q36_N_HEAD_DIM;
        q36_ref_rms_norm(dst, dst, ctx->weight, Q36_N_HEAD_DIM, Q36_RMS_EPS);
        q36_apply_rope_one(dst, Q36_N_HEAD_DIM, ctx->pos0 + t);
    }
}

static void q36_full_attn_scores_worker(void *opaque, uint64_t row0, uint64_t row1) {
    q36_full_attn_scores_ctx *ctx = (q36_full_attn_scores_ctx *)opaque;
    for (uint64_t idx = row0; idx < row1; idx++) {
        uint32_t t = (uint32_t)(idx / Q36_N_HEAD);
        uint32_t h = (uint32_t)(idx % Q36_N_HEAD);
        uint32_t kvh = h / (Q36_N_HEAD / Q36_N_HEAD_KV);
        uint32_t pos = ctx->pos0 + t;
        const float *qh = ctx->q + (uint64_t)t * Q36_N_HEAD * Q36_N_HEAD_DIM + (uint64_t)h * Q36_N_HEAD_DIM;
        const float *gate = ctx->qg + (uint64_t)t * Q36_N_HEAD * Q36_N_HEAD_DIM * 2u + (uint64_t)h * Q36_N_HEAD_DIM * 2u + Q36_N_HEAD_DIM;
        float *head_out = ctx->out + ((uint64_t)t * Q36_N_HEAD + h) * Q36_N_VALUE_DIM;
        float max_score = ctx->have_sinks ? ctx->sinks[h] : -INFINITY;
        double denom = 0.0;
        memset(head_out, 0, (size_t)Q36_N_VALUE_DIM * sizeof(float));
        for (uint32_t i = 0; i <= pos && i < ctx->cache->len; i++) {
            const uint8_t *kc = ctx->cache->k + (uint64_t)i * ctx->cache->k_row_bytes + (uint64_t)kvh * q36_kv_cache_row_bytes(ctx->cache->type_k, Q36_N_HEAD_DIM);
            double dot = 0.0;
            for (uint32_t j = 0; j < Q36_N_HEAD_DIM; j++) dot += (double)qh[j] * (double)q36_kv_cache_at(kc, ctx->cache->type_k, j);
            {
                float score = (float)(dot / sqrt((double)Q36_N_HEAD_DIM));
                if (score > max_score) max_score = score;
            }
        }
        if (ctx->have_sinks) denom = exp((double)ctx->sinks[h] - (double)max_score);
        for (uint32_t i = 0; i <= pos && i < ctx->cache->len; i++) {
            const uint8_t *kc = ctx->cache->k + (uint64_t)i * ctx->cache->k_row_bytes + (uint64_t)kvh * q36_kv_cache_row_bytes(ctx->cache->type_k, Q36_N_HEAD_DIM);
            const uint8_t *vc = ctx->cache->v + (uint64_t)i * ctx->cache->v_row_bytes + (uint64_t)kvh * q36_kv_cache_row_bytes(ctx->cache->type_v, Q36_N_VALUE_DIM);
            double dot = 0.0;
            for (uint32_t j = 0; j < Q36_N_HEAD_DIM; j++) dot += (double)qh[j] * (double)q36_kv_cache_at(kc, ctx->cache->type_k, j);
            {
                float w = expf((float)(dot / sqrt((double)Q36_N_HEAD_DIM)) - max_score);
                denom += w;
                for (uint32_t j = 0; j < Q36_N_VALUE_DIM; j++) head_out[j] += w * q36_kv_cache_at(vc, ctx->cache->type_v, j);
            }
        }
        if (denom > 0.0) {
            float inv = (float)(1.0 / denom);
            for (uint32_t j = 0; j < Q36_N_VALUE_DIM; j++) head_out[j] *= inv;
        }
        for (uint32_t j = 0; j < Q36_N_VALUE_DIM; j++) head_out[j] *= q36_sigmoidf(gate[j]);
    }
}

static void q36_ffn_routed_batch_worker(void *opaque, uint64_t row0, uint64_t row1) {
    q36_ffn_routed_batch_ctx *ctx = (q36_ffn_routed_batch_ctx *)opaque;
    for (uint64_t t = row0; t < row1; t++) {
        float top_val[Q36_N_EXPERT_USED];
        float top_w[Q36_N_EXPERT_USED];
        uint32_t top_idx[Q36_N_EXPERT_USED];
        float gate[Q36_N_FF_EXP];
        float up[Q36_N_FF_EXP];
        float mid[Q36_N_FF_EXP];
        float down[Q36_N_EMBD];
        float rowbuf[Q36_N_SSM_INNER];
        float *logits = ctx->gate_logits + t * Q36_N_EXPERT;
        float *out = ctx->out + t * Q36_N_EMBD;
        const float *inp = ctx->inp + t * Q36_N_EMBD;
        const uint8_t *inpq = ctx->xq ? ctx->xq + (uint64_t)t * ctx->xq_row_bytes : NULL;
        q36_softmax_inplace(logits, Q36_N_EXPERT);
        q36_topk_indices(logits, Q36_N_EXPERT, Q36_N_EXPERT_USED, top_idx, top_val);
        {
            double sum = 0.0;
            for (uint32_t i = 0; i < Q36_N_EXPERT_USED; i++) sum += top_val[i];
            if (sum < 6.103515625e-5) sum = 6.103515625e-5;
            for (uint32_t i = 0; i < Q36_N_EXPERT_USED; i++) top_w[i] = (float)((double)top_val[i] / sum) * ctx->e->expert_weights_scale;
        }
        memset(out, 0, (size_t)Q36_N_EMBD * sizeof(float));
        for (uint32_t i = 0; i < Q36_N_EXPERT_USED; i++) {
            float gate_scale = q36_tensor_index_or(&ctx->e->model, ctx->l->ffn_gate_exps_scale, top_idx[i], 1.0f);
            float up_scale = q36_tensor_index_or(&ctx->e->model, ctx->l->ffn_up_exps_scale, top_idx[i], 1.0f);
            if (!q36_tensor_expert_matvec_pair_prequant(ctx->e,
                                                        ctx->l->ffn_gate_exps,
                                                        ctx->l->ffn_up_exps,
                                                        top_idx[i],
                                                        inp,
                                                        inpq,
                                                        gate,
                                                        up,
                                                        rowbuf,
                                                        Q36_N_EMBD,
                                                        Q36_N_FF_EXP,
                                                        gate_scale,
                                                        up_scale)) {
                if (!q36_tensor_expert_matvec_prequant(ctx->e, ctx->l->ffn_gate_exps, top_idx[i], inp, inpq,
                                                       gate, rowbuf, Q36_N_EMBD, Q36_N_FF_EXP)) {
                    continue;
                }
                q36_scale_inplace(gate, Q36_N_FF_EXP, gate_scale);
                if (!q36_tensor_expert_matvec_prequant(ctx->e, ctx->l->ffn_up_exps, top_idx[i], inp, inpq,
                                                       up, rowbuf, Q36_N_EMBD, Q36_N_FF_EXP)) {
                    continue;
                }
                q36_scale_inplace(up, Q36_N_FF_EXP, up_scale);
            }
            for (uint32_t j = 0; j < Q36_N_FF_EXP; j++) mid[j] = q36_siluf(gate[j]) * up[j];
            if (!q36_tensor_expert_matvec(ctx->e, ctx->l->ffn_down_exps, top_idx[i], mid, down, rowbuf, Q36_N_FF_EXP, Q36_N_EMBD)) continue;
            q36_scale_inplace(down, Q36_N_EMBD,
                              q36_tensor_index_or(&ctx->e->model, ctx->l->ffn_down_exps_scale, top_idx[i], 1.0f));
            for (uint32_t j = 0; j < Q36_N_EMBD; j++) out[j] += down[j] * top_w[i];
        }
    }
}

static void q36_ffn_combine_batch_worker(void *opaque, uint64_t row0, uint64_t row1) {
    q36_ffn_combine_batch_ctx *ctx = (q36_ffn_combine_batch_ctx *)opaque;
    for (uint64_t t = row0; t < row1; t++) {
        float g = q36_sigmoidf(ctx->scalar[t]);
        float *out = ctx->out + t * Q36_N_EMBD;
        const float *shared = ctx->shared + t * Q36_N_EMBD;
        for (uint32_t j = 0; j < Q36_N_EMBD; j++) out[j] += shared[j] * g;
    }
}

static bool q36_forward_ffn_batch(const q36_engine *e,
                                  const q36_layer_weights *l,
                                  const float *inp,
                                  float *out,
                                  uint32_t n_tok,
                                  q36_cpu_runtime *rt) {
    q36_ffn_routed_batch_ctx routed_ctx;
    q36_ffn_combine_batch_ctx combine_ctx;
    const float *shared_gate_inp;
    const uint8_t *gate_inp_xq = NULL;
    q36_activation_quant_kind inpq_kind = q36_activation_quant_kind_for_type(l->ffn_gate_inp->type);
    if (inpq_kind != Q36_ACTIVATION_QUANT_NONE) {
        if (!q36_quantize_activation_batch(inpq_kind, inp, rt->batch_xq, n_tok, Q36_N_EMBD, e->n_threads)) return false;
        gate_inp_xq = rt->batch_xq;
    }
    if (!q36_tensor_matmul_batch_prequant(e, l->ffn_gate_inp, inp, gate_inp_xq,
                                          rt->batch_ffn_gate_logits,
                                          n_tok, Q36_N_EMBD, Q36_N_EXPERT, 1.0f)) return false;
    routed_ctx.out = out;
    routed_ctx.e = e;
    routed_ctx.l = l;
    routed_ctx.inp = inp;
    routed_ctx.gate_logits = rt->batch_ffn_gate_logits;
    inpq_kind = q36_activation_quant_kind_for_type(l->ffn_gate_exps->type);
    if (!q36_quantize_activation_batch(inpq_kind, inp, rt->batch_xq, n_tok, Q36_N_EMBD, e->n_threads)) return false;
    routed_ctx.xq = inpq_kind == Q36_ACTIVATION_QUANT_NONE ? NULL : rt->batch_xq;
    routed_ctx.xq_row_bytes = q36_activation_quant_row_bytes(inpq_kind, Q36_N_EMBD);
    q36_parallel_for_rows(n_tok, 2, e->n_threads, q36_ffn_routed_batch_worker, &routed_ctx);

    {
        float gate_scale = q36_tensor_scalar_or(&e->model, l->ffn_gate_shexp_scale, 1.0f);
        float up_scale = q36_tensor_scalar_or(&e->model, l->ffn_up_shexp_scale, 1.0f);
        q36_activation_quant_kind shared_kind = q36_activation_quant_kind_for_type(l->ffn_gate_shexp->type);
        if (shared_kind != inpq_kind) {
            if (!q36_quantize_activation_batch(shared_kind, inp, rt->batch_xq, n_tok, Q36_N_EMBD, e->n_threads)) return false;
            inpq_kind = shared_kind;
        }
        if (!q36_tensor_matmul_pair_batch_prequant(e,
                                                   l->ffn_gate_shexp,
                                                   l->ffn_up_shexp,
                                                   inp,
                                                   rt->batch_xq,
                                                   rt->batch_ffn_shared_gate,
                                                   rt->batch_ffn_shared_up,
                                                   n_tok,
                                                   Q36_N_EMBD,
                                                   Q36_N_FF_SHARED,
                                                   gate_scale,
                                                   up_scale)) {
            if (!q36_tensor_matmul_batch_prequant(e, l->ffn_gate_shexp, inp, rt->batch_xq,
                                                  rt->batch_ffn_shared_gate,
                                                  n_tok, Q36_N_EMBD, Q36_N_FF_SHARED,
                                                  gate_scale)) {
                return false;
            }
            if (!q36_tensor_matmul_batch_prequant(e, l->ffn_up_shexp, inp, rt->batch_xq,
                                                  rt->batch_ffn_shared_up,
                                                  n_tok, Q36_N_EMBD, Q36_N_FF_SHARED,
                                                  up_scale)) {
                return false;
            }
        }
    }
    q36_swiglu_rows(rt->batch_ffn_shared_mid,
                    rt->batch_ffn_shared_gate,
                    rt->batch_ffn_shared_up,
                    n_tok,
                    Q36_N_FF_SHARED,
                    e->n_threads);

    if (!q36_quantize_activation_batch_for_type(l->ffn_down_shexp->type, rt->batch_ffn_shared_mid,
                                               rt->batch_xq, n_tok, Q36_N_FF_SHARED, e->n_threads)) {
        return false;
    }
    if (!q36_tensor_matmul_batch_prequant(e, l->ffn_down_shexp, rt->batch_ffn_shared_mid, rt->batch_xq,
                                          rt->batch_ffn_shared_out,
                                          n_tok, Q36_N_FF_SHARED, Q36_N_EMBD,
                                          q36_tensor_scalar_or(&e->model, l->ffn_down_shexp_scale, 1.0f))) {
        return false;
    }

    shared_gate_inp = (const float *)(e->model.map + l->ffn_gate_inp_shexp->abs_offset);
    q36_tensor_vector_dot_batch(rt->batch_ffn_scalar, shared_gate_inp, inp, n_tok, Q36_N_EMBD, e->n_threads);
    combine_ctx.out = out;
    combine_ctx.shared = rt->batch_ffn_shared_out;
    combine_ctx.scalar = rt->batch_ffn_scalar;
    q36_parallel_for_rows(n_tok, 2, e->n_threads, q36_ffn_combine_batch_worker, &combine_ctx);
    return true;
}

static bool q36_forward_full_attn_batch(const q36_engine *e,
                                        const q36_layer_weights *l,
                                        uint32_t il,
                                        uint32_t pos0,
                                        const float *inp,
                                        float *out,
                                        uint32_t n_tok,
                                        q36_cpu_runtime *rt) {
    q36_full_attn_q_rows_ctx qctx;
    q36_full_attn_k_rows_ctx kctx;
    q36_full_attn_scores_ctx sctx;
    q36_full_attn_cache *cache = &rt->full[il];
    float sinks[Q36_N_HEAD];
    bool have_sinks = false;
    q36_activation_quant_kind xq_kind = q36_activation_quant_kind_for_type(l->attn_q->type);
    if (!q36_quantize_activation_batch(xq_kind, inp, rt->batch_xq, n_tok, Q36_N_EMBD, e->n_threads)) return false;
    if (!q36_tensor_matmul_batch_prequant(e, l->attn_q, inp, rt->batch_xq,
                                          rt->batch_qg,
                                          n_tok, Q36_N_EMBD, Q36_N_HEAD * Q36_N_HEAD_DIM * 2u,
                                          q36_tensor_scalar_or(&e->model, l->attn_q_scale, 1.0f))) {
        return false;
    }
    if (q36_activation_quant_kind_for_type(l->attn_k->type) != xq_kind) {
        xq_kind = q36_activation_quant_kind_for_type(l->attn_k->type);
        if (!q36_quantize_activation_batch(xq_kind, inp, rt->batch_xq, n_tok, Q36_N_EMBD, e->n_threads)) return false;
    }
    if (!q36_tensor_matmul_batch_prequant(e, l->attn_k, inp, rt->batch_xq,
                                          rt->batch_k,
                                          n_tok, Q36_N_EMBD, Q36_N_HEAD_KV * Q36_N_HEAD_DIM,
                                          q36_tensor_scalar_or(&e->model, l->attn_k_scale, 1.0f))) {
        return false;
    }
    if (q36_activation_quant_kind_for_type(l->attn_v->type) != xq_kind) {
        xq_kind = q36_activation_quant_kind_for_type(l->attn_v->type);
        if (!q36_quantize_activation_batch(xq_kind, inp, rt->batch_xq, n_tok, Q36_N_EMBD, e->n_threads)) return false;
    }
    if (!q36_tensor_matmul_batch_prequant(e, l->attn_v, inp, rt->batch_xq,
                                          rt->batch_v,
                                          n_tok, Q36_N_EMBD, Q36_N_HEAD_KV * Q36_N_VALUE_DIM,
                                          q36_tensor_scalar_or(&e->model, l->attn_v_scale, 1.0f))) {
        return false;
    }
    qctx.q = rt->batch_q;
    qctx.qg = rt->batch_qg;
    qctx.weight = (const float *)(e->model.map + l->attn_q_norm->abs_offset);
    qctx.pos0 = pos0;
    q36_parallel_for_rows((uint64_t)n_tok * Q36_N_HEAD, 1, e->n_threads, q36_full_attn_q_rows_worker, &qctx);
    kctx.k = rt->batch_k;
    kctx.weight = (const float *)(e->model.map + l->attn_k_norm->abs_offset);
    kctx.pos0 = pos0;
    q36_parallel_for_rows((uint64_t)n_tok * Q36_N_HEAD_KV, 1, e->n_threads, q36_full_attn_k_rows_worker, &kctx);
    if (l->attn_sinks) have_sinks = q36_tensor_get_plain(&e->model, l->attn_sinks, sinks, Q36_N_HEAD);
    for (uint32_t t = 0; t < n_tok; t++) {
        q36_full_attn_cache_store(cache,
                                  pos0 + t,
                                  rt->batch_k + (uint64_t)t * Q36_N_HEAD_KV * Q36_N_HEAD_DIM,
                                  rt->batch_v + (uint64_t)t * Q36_N_HEAD_KV * Q36_N_VALUE_DIM);
    }
    sctx.out = rt->batch_attn_out;
    sctx.q = rt->batch_q;
    sctx.qg = rt->batch_qg;
    sctx.cache = cache;
    sctx.sinks = sinks;
    sctx.pos0 = pos0;
    sctx.have_sinks = have_sinks;
    q36_parallel_for_rows((uint64_t)n_tok * Q36_N_HEAD, 1, e->n_threads, q36_full_attn_scores_worker, &sctx);
    if (!q36_quantize_activation_batch_for_type(l->attn_output->type, rt->batch_attn_out,
                                               rt->batch_xq, n_tok, Q36_N_SSM_INNER, e->n_threads)) {
        return false;
    }
    if (!q36_tensor_matmul_batch_prequant(e, l->attn_output, rt->batch_attn_out, rt->batch_xq,
                                          out,
                                          n_tok, Q36_N_SSM_INNER, Q36_N_EMBD,
                                          q36_tensor_scalar_or(&e->model, l->attn_output_scale, 1.0f))) {
        return false;
    }
    return true;
}

static bool q36_forward_recurrent_batch(const q36_engine *e,
                                        const q36_layer_weights *l,
                                        uint32_t il,
                                        const float *inp,
                                        float *out,
                                        uint32_t n_tok,
                                        q36_cpu_runtime *rt) {
    q36_recurrent_cache *cache = &rt->recurrent[il];
    const float *kernel = (const float *)(e->model.map + l->ssm_conv1d->abs_offset);
    const float *dt_bias = (const float *)(e->model.map + l->ssm_dt->abs_offset);
    const float *a = (const float *)(e->model.map + l->ssm_a->abs_offset);
    const float *ssm_norm = (const float *)(e->model.map + l->ssm_norm->abs_offset);
    q36_activation_quant_kind xq_kind = q36_activation_quant_kind_for_type(l->attn_qkv->type);
    if (!q36_quantize_activation_batch(xq_kind, inp, rt->batch_xq, n_tok, Q36_N_EMBD, e->n_threads)) return false;
    if (!q36_tensor_matmul_batch_prequant(e, l->attn_qkv, inp, rt->batch_xq,
                                          rt->batch_recur_qkv,
                                          n_tok, Q36_N_EMBD, Q36_N_SSM_CONV_DIM,
                                          q36_tensor_scalar_or(&e->model, l->attn_qkv_scale, 1.0f))) {
        return false;
    }
    if (q36_activation_quant_kind_for_type(l->attn_gate->type) != xq_kind) {
        xq_kind = q36_activation_quant_kind_for_type(l->attn_gate->type);
        if (!q36_quantize_activation_batch(xq_kind, inp, rt->batch_xq, n_tok, Q36_N_EMBD, e->n_threads)) return false;
    }
    if (!q36_tensor_matmul_batch_prequant(e, l->attn_gate, inp, rt->batch_xq,
                                          rt->batch_recur_z,
                                          n_tok, Q36_N_EMBD, Q36_N_SSM_INNER,
                                          q36_tensor_scalar_or(&e->model, l->attn_gate_scale, 1.0f))) {
        return false;
    }
    if (q36_activation_quant_kind_for_type(l->ssm_beta->type) != xq_kind) {
        xq_kind = q36_activation_quant_kind_for_type(l->ssm_beta->type);
        if (!q36_quantize_activation_batch(xq_kind, inp, rt->batch_xq, n_tok, Q36_N_EMBD, e->n_threads)) return false;
    }
    if (!q36_tensor_matmul_batch_prequant(e, l->ssm_beta, inp, rt->batch_xq,
                                          rt->batch_recur_beta,
                                          n_tok, Q36_N_EMBD, Q36_N_SSM_DT_RANK,
                                          q36_tensor_scalar_or(&e->model, l->ssm_beta_scale, 1.0f))) {
        return false;
    }
    if (q36_activation_quant_kind_for_type(l->ssm_alpha->type) != xq_kind) {
        xq_kind = q36_activation_quant_kind_for_type(l->ssm_alpha->type);
        if (!q36_quantize_activation_batch(xq_kind, inp, rt->batch_xq, n_tok, Q36_N_EMBD, e->n_threads)) return false;
    }
    if (!q36_tensor_matmul_batch_prequant(e, l->ssm_alpha, inp, rt->batch_xq,
                                          rt->batch_recur_alpha,
                                          n_tok, Q36_N_EMBD, Q36_N_SSM_DT_RANK,
                                          q36_tensor_scalar_or(&e->model, l->ssm_alpha_scale, 1.0f))) {
        return false;
    }
    for (uint32_t t = 0; t < n_tok; t++) {
        float *qkv = rt->batch_recur_qkv + (uint64_t)t * Q36_N_SSM_CONV_DIM;
        float *z = rt->batch_recur_z + (uint64_t)t * Q36_N_SSM_INNER;
        float *beta = rt->batch_recur_beta + (uint64_t)t * Q36_N_SSM_DT_RANK;
        float *gate = rt->batch_recur_alpha + (uint64_t)t * Q36_N_SSM_DT_RANK;
        float *window = rt->work4;
        float *conv = rt->work3;
        float *q = rt->work5;
        float *k = rt->work4;
        float *v = rt->work0;
        float *proj = rt->batch_recur_proj + (uint64_t)t * Q36_N_SSM_INNER;
        q36_recurrent_conv_step(cache, qkv, window);
        q36_ssm_conv_apply(window, kernel, conv);
        for (uint32_t i = 0; i < Q36_N_SSM_CONV_DIM; i++) conv[i] = q36_siluf(conv[i]);
        if (!q36_all_finite(conv, Q36_N_SSM_CONV_DIM)) return false;
        for (uint32_t h = 0; h < Q36_N_SSM_DT_RANK; h++) {
            uint32_t gh = h % Q36_N_SSM_GROUP;
            memcpy(q + (uint64_t)h * Q36_N_SSM_STATE,
                   conv + (uint64_t)gh * Q36_N_SSM_STATE,
                   (size_t)Q36_N_SSM_STATE * sizeof(float));
            memcpy(k + (uint64_t)h * Q36_N_SSM_STATE,
                   conv + (uint64_t)Q36_N_SSM_QK + (uint64_t)gh * Q36_N_SSM_STATE,
                   (size_t)Q36_N_SSM_STATE * sizeof(float));
        }
        memcpy(v, conv + (uint64_t)Q36_N_SSM_QK * 2u,
               (size_t)Q36_N_SSM_DT_RANK * Q36_N_SSM_STATE * sizeof(float));
        for (uint32_t h = 0; h < Q36_N_SSM_DT_RANK; h++) {
            q36_l2_norm(q + (uint64_t)h * Q36_N_SSM_STATE, Q36_N_SSM_STATE, Q36_RMS_EPS);
            q36_l2_norm(k + (uint64_t)h * Q36_N_SSM_STATE, Q36_N_SSM_STATE, Q36_RMS_EPS);
            beta[h] = q36_sigmoidf(beta[h]);
            gate[h] = q36_softplusf(gate[h] + dt_bias[h]) * a[h];
        }
        if (!q36_all_finite(beta, Q36_N_SSM_DT_RANK)) return false;
        if (!q36_all_finite(gate, Q36_N_SSM_DT_RANK)) return false;
        q36_delta_net_decode(cache->state, q, k, v, gate, beta, proj);
        if (!q36_all_finite(proj, Q36_N_SSM_INNER)) return false;
        for (uint32_t h = 0; h < Q36_N_SSM_DT_RANK; h++) {
            q36_ref_rms_norm(proj + (uint64_t)h * Q36_N_SSM_STATE,
                             proj + (uint64_t)h * Q36_N_SSM_STATE,
                             ssm_norm,
                             Q36_N_SSM_STATE,
                             Q36_RMS_EPS);
        }
        for (uint32_t i = 0; i < Q36_N_SSM_INNER; i++) proj[i] *= q36_siluf(z[i]);
        if (!q36_all_finite(proj, Q36_N_SSM_INNER)) return false;
    }
    if (!q36_quantize_activation_batch_for_type(l->ssm_out->type, rt->batch_recur_proj,
                                               rt->batch_xq, n_tok, Q36_N_SSM_INNER, e->n_threads)) {
        return false;
    }
    if (!q36_tensor_matmul_batch_prequant(e, l->ssm_out, rt->batch_recur_proj, rt->batch_xq,
                                          out,
                                          n_tok, Q36_N_SSM_INNER, Q36_N_EMBD,
                                          q36_tensor_scalar_or(&e->model, l->ssm_out_scale, 1.0f))) {
        return false;
    }
    return true;
}

static bool q36_forward_tokens_cpu(q36_session *s,
                                   const int *tokens,
                                   uint32_t n_tok,
                                   uint32_t pos0,
                                   bool compute_logits) {
    q36_engine *e;
    q36_cpu_runtime *rt;
    if (!s || !s->runtime || !tokens || n_tok == 0) return false;
    e = s->engine;
    rt = (q36_cpu_runtime *)s->runtime;
    if (n_tok > rt->prefill_cap) return false;
    for (uint32_t t = 0; t < n_tok; t++) {
        if (!q36_embed_token(e, tokens[t], rt->batch_hidden + (uint64_t)t * Q36_N_EMBD)) {
            fprintf(stderr, "q36: embed failed for token=%d pos=%u\n", tokens[t], pos0 + t);
            return false;
        }
    }
    for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
        const q36_layer_weights *l = &e->weights.layer[il];
        q36_rms_norm_rows(rt->batch_norm,
                          rt->batch_hidden,
                          (const float *)(e->model.map + l->attn_norm->abs_offset),
                          n_tok,
                          Q36_N_EMBD,
                          Q36_RMS_EPS,
                          e->n_threads);
        if (l->kind == Q36_LAYER_FULL_ATTN) {
            if (!q36_forward_full_attn_batch(e, l, il, pos0, rt->batch_norm, rt->batch_next_hidden, n_tok, rt)) {
                fprintf(stderr, "q36: full attention block failed at pos=%u layer=%u\n", pos0, il);
                return false;
            }
        } else {
            if (!q36_forward_recurrent_batch(e, l, il, rt->batch_norm, rt->batch_next_hidden, n_tok, rt)) {
                fprintf(stderr, "q36: recurrent block failed at pos=%u layer=%u\n", pos0, il);
                return false;
            }
        }
        q36_directional_steering_project_rows(rt->batch_next_hidden,
                                               e->directional_steering_dirs,
                                               il,
                                               n_tok,
                                               e->directional_steering_attn_scale);
        if (!q36_all_finite(rt->batch_next_hidden, n_tok * Q36_N_EMBD)) {
            fprintf(stderr, "q36: non-finite attn output at pos=%u layer=%u kind=%s\n",
                    pos0, il, l->kind == Q36_LAYER_FULL_ATTN ? "full" : "recurrent");
            return false;
        }
        q36_add_rows(rt->batch_next_hidden,
                     rt->batch_next_hidden,
                     rt->batch_hidden,
                     n_tok,
                     Q36_N_EMBD,
                     e->n_threads);
        q36_rms_norm_rows(rt->batch_norm,
                          rt->batch_next_hidden,
                          (const float *)(e->model.map + l->post_attention_norm->abs_offset),
                          n_tok,
                          Q36_N_EMBD,
                          Q36_RMS_EPS,
                          e->n_threads);
        if (!q36_forward_ffn_batch(e, l, rt->batch_norm, rt->batch_hidden, n_tok, rt)) {
            fprintf(stderr, "q36: ffn block failed at pos=%u layer=%u\n", pos0, il);
            return false;
        }
        q36_directional_steering_project_rows(rt->batch_hidden,
                                               e->directional_steering_dirs,
                                               il,
                                               n_tok,
                                               e->directional_steering_ffn_scale);
        if (!q36_all_finite(rt->batch_hidden, n_tok * Q36_N_EMBD)) {
            fprintf(stderr, "q36: non-finite ffn output at pos=%u layer=%u\n", pos0, il);
            return false;
        }
        q36_add_rows(rt->batch_hidden,
                     rt->batch_next_hidden,
                     rt->batch_hidden,
                     n_tok,
                     Q36_N_EMBD,
                     e->n_threads);
        if (!q36_all_finite(rt->batch_hidden, n_tok * Q36_N_EMBD)) {
            fprintf(stderr, "q36: non-finite hidden state at pos=%u layer=%u\n", pos0, il);
            return false;
        }
    }
    q36_rms_norm_rows(rt->batch_norm,
                      rt->batch_hidden,
                      (const float *)(e->model.map + e->weights.output_norm->abs_offset),
                      n_tok,
                      Q36_N_EMBD,
                      Q36_RMS_EPS,
                      e->n_threads);
    if (!q36_all_finite(rt->batch_norm, n_tok * Q36_N_EMBD)) {
        fprintf(stderr, "q36: non-finite output norm at pos=%u\n", pos0);
        return false;
    }
    if (compute_logits) {
        const float *last = rt->batch_norm + (uint64_t)(n_tok - 1u) * Q36_N_EMBD;
        if (!q36_tensor_matvec(e, e->weights.output, last, s->logits, rt->work1, Q36_N_EMBD, Q36_N_VOCAB)) {
            fprintf(stderr, "q36: output projection failed at pos=%u\n", pos0);
            return false;
        }
        q36_scale_inplace(s->logits, Q36_N_VOCAB,
                          q36_tensor_scalar_or(&e->model, e->weights.output_scale, 1.0f));
        if (!q36_all_finite(s->logits, Q36_N_VOCAB)) {
            fprintf(stderr, "q36: non-finite logits at pos=%u\n", pos0);
            return false;
        }
    }
    return true;
}

#ifndef Q36_NO_GPU
typedef struct {
    float *out;
    const float *shared;
    const float *scalar;
} q36_vulkan_ffn_tail_ctx;

static Q36_MAYBE_UNUSED void q36_vulkan_ffn_tail_rows(void *opaque, uint64_t row0, uint64_t row1) {
    q36_vulkan_ffn_tail_ctx *ctx = (q36_vulkan_ffn_tail_ctx *)opaque;

    for (uint64_t t = row0; t < row1; t++) {
        float g = q36_sigmoidf(ctx->scalar[t]);
        const float *st = ctx->shared + t * Q36_N_EMBD;
        float *ot = ctx->out + t * Q36_N_EMBD;
        for (uint32_t j = 0; j < Q36_N_EMBD; j++) ot[j] = fmaf(st[j], g, ot[j]);
    }
}

typedef struct {
    float *x;
    uint32_t n_head;
    uint32_t pos0;
} q36_vulkan_rope_ctx;

static void q36_vulkan_rope_rows(void *opaque, uint64_t row0, uint64_t row1) {
    q36_vulkan_rope_ctx *ctx = (q36_vulkan_rope_ctx *)opaque;

    for (uint64_t t = row0; t < row1; t++) {
        for (uint32_t h = 0; h < ctx->n_head; h++) {
            q36_apply_rope_one(ctx->x + (t * ctx->n_head + h) * Q36_N_HEAD_DIM,
                               Q36_N_HEAD_DIM, ctx->pos0 + (uint32_t)t);
        }
    }
}

static bool q36_forward_ffn_vulkan_model(q36_vulkan_runtime *rt,
                                         const q36_model *m,
                                         const q36_layer_weights *l,
                                         float expert_weights_scale,
                                         uint32_t il,
                                         uint32_t n_tok,
                                         const q36_gpu_tensor *inp,
                                         q36_gpu_tensor *out,
                                         bool session_batch) {
    q36_gpu_moe_weight gate;
    q36_gpu_moe_weight up;
    q36_gpu_moe_weight down;
    const float *shared;
    const float *scalar;
    float *outp;
    if (!rt || !m || !l || !inp || !out) return false;
    gate.offset = l->ffn_gate_exps->abs_offset;
    gate.type = l->ffn_gate_exps->type;
    gate.scales_offset = l->ffn_gate_exps_scale ? l->ffn_gate_exps_scale->abs_offset : 0;
    gate.has_scales = l->ffn_gate_exps_scale != NULL;
    up.offset = l->ffn_up_exps->abs_offset;
    up.type = l->ffn_up_exps->type;
    up.scales_offset = l->ffn_up_exps_scale ? l->ffn_up_exps_scale->abs_offset : 0;
    up.has_scales = l->ffn_up_exps_scale != NULL;
    down.offset = l->ffn_down_exps->abs_offset;
    down.type = l->ffn_down_exps->type;
    down.scales_offset = l->ffn_down_exps_scale ? l->ffn_down_exps_scale->abs_offset : 0;
    down.has_scales = l->ffn_down_exps_scale != NULL;
    /* The router and the shared-expert gate scalar run on the host from the
     * host-written norm, and the shared-expert gate/up dispatches record
     * before the routed pass so the routed block's one flush completes them
     * too. */
    if (!q36_gpu_tensor_matmul_scaled(m, l->ffn_gate_inp, inp, rt->ffn_gate_logits,
                                      Q36_N_EMBD, Q36_N_EXPERT, n_tok, 1.0f)) {
        return false;
    }
    if (!q36_gpu_router_topk_tensor(rt->ffn_selected, rt->ffn_weights, rt->ffn_gate_logits,
                                    Q36_N_EXPERT, Q36_N_EXPERT_USED, n_tok, expert_weights_scale)) {
        return false;
    }
    if (!q36_gpu_tensor_matmul_scaled(m, l->ffn_gate_inp_shexp, inp, rt->ffn_scalar,
                                      Q36_N_EMBD, 1, n_tok, 1.0f)) {
        return false;
    }
    /* Fast routed pass, decode and prefill: f32 activations straight into
     * the fused IQ2_XXS gate/up and Q2_K down kernels, expert schedule
     * built and consumed on GPU. When it runs, the q8_K input staging is
     * only needed for k-quant shared-expert projections, which this model
     * does not have. */
    bool routed_done =
        q36_gpu_moe_ffn_f32_tensor(out, m->map, m->size,
                                          &gate, &up, &down,
                                          rt->ffn_selected, rt->ffn_weights,
                                          il, Q36_N_EXPERT_USED, inp, n_tok,
                                          Q36_N_EMBD, Q36_N_FF_EXP, Q36_N_EMBD, Q36_N_EXPERT) != 0;
    if (routed_done && (n_tok == 1 || session_batch) &&
        l->ffn_gate_shexp->type == Q36_TENSOR_Q8_0 &&
        l->ffn_up_shexp->type == Q36_TENSOR_Q8_0 &&
        l->ffn_down_shexp->type == Q36_TENSOR_Q8_0) {
        const char *env = getenv("Q36_VK_SHARED_FFN_DECODE");
        if (!env || !env[0] || env[0] != '0') {
            if (n_tok == 1 && q36_gpu_shared_ffn_decode_tensor(
                    out, rt->ffn_shared_mid, inp, rt->ffn_scalar,
                    m->map, m->size,
                    l->ffn_gate_shexp->abs_offset,
                    l->ffn_up_shexp->abs_offset,
                    l->ffn_down_shexp->abs_offset,
                    Q36_N_EMBD, Q36_N_FF_SHARED, Q36_N_EMBD,
                    q36_tensor_scalar_or(m, l->ffn_gate_shexp_scale, 1.0f),
                    q36_tensor_scalar_or(m, l->ffn_up_shexp_scale, 1.0f),
                    q36_tensor_scalar_or(m, l->ffn_down_shexp_scale, 1.0f))) {
                return true;
            }
            bool ok = true;
            for (uint32_t row = 0; session_batch && row < n_tok && ok; row++) {
                q36_gpu_tensor *row_out = q36_gpu_tensor_view(
                    out, (uint64_t)row * Q36_N_EMBD * sizeof(float),
                    (uint64_t)Q36_N_EMBD * sizeof(float));
                q36_gpu_tensor *row_mid = q36_gpu_tensor_view(
                    rt->ffn_shared_mid,
                    (uint64_t)row * Q36_N_FF_SHARED * sizeof(float),
                    (uint64_t)Q36_N_FF_SHARED * sizeof(float));
                q36_gpu_tensor *row_inp = q36_gpu_tensor_view(
                    inp, (uint64_t)row * Q36_N_EMBD * sizeof(float),
                    (uint64_t)Q36_N_EMBD * sizeof(float));
                q36_gpu_tensor *row_scalar = q36_gpu_tensor_view(
                    rt->ffn_scalar, (uint64_t)row * sizeof(float),
                    sizeof(float));
                ok = row_out && row_mid && row_inp && row_scalar &&
                    q36_gpu_shared_ffn_decode_tensor(
                        row_out, row_mid, row_inp, row_scalar,
                        m->map, m->size,
                        l->ffn_gate_shexp->abs_offset,
                        l->ffn_up_shexp->abs_offset,
                        l->ffn_down_shexp->abs_offset,
                        Q36_N_EMBD, Q36_N_FF_SHARED, Q36_N_EMBD,
                        q36_tensor_scalar_or(m, l->ffn_gate_shexp_scale, 1.0f),
                        q36_tensor_scalar_or(m, l->ffn_up_shexp_scale, 1.0f),
                        q36_tensor_scalar_or(m, l->ffn_down_shexp_scale, 1.0f));
                q36_gpu_tensor_free(row_scalar);
                q36_gpu_tensor_free(row_inp);
                q36_gpu_tensor_free(row_mid);
                q36_gpu_tensor_free(row_out);
            }
            if (session_batch && ok) return true;
        }
    }
    if ((!routed_done ||
         l->ffn_gate_shexp->type != Q36_TENSOR_Q8_0 ||
         l->ffn_up_shexp->type != Q36_TENSOR_Q8_0) &&
        !q36_gpu_quantize_q8_k_tensor(rt->inp_q8, inp, Q36_N_EMBD, n_tok)) {
        return false;
    }
    if (!q36_gpu_tensor_matmul_q8_or_float_scaled(m, l->ffn_gate_shexp, inp, rt->inp_q8, rt->ffn_shared_gate,
                                                  Q36_N_EMBD, Q36_N_FF_SHARED, n_tok,
                                                  q36_tensor_scalar_or(m, l->ffn_gate_shexp_scale, 1.0f))) {
        return false;
    }
    if (!q36_gpu_tensor_matmul_q8_or_float_scaled(m, l->ffn_up_shexp, inp, rt->inp_q8, rt->ffn_shared_up,
                                                  Q36_N_EMBD, Q36_N_FF_SHARED, n_tok,
                                                  q36_tensor_scalar_or(m, l->ffn_up_shexp_scale, 1.0f))) {
        return false;
    }
    if (!routed_done &&
        !q36_gpu_moe_ffn_q8_tensor(out, m->map, m->size,
                                   &gate, &up, &down,
                                   rt->ffn_selected, rt->ffn_weights,
                                   il, Q36_N_EXPERT_USED, rt->inp_q8, n_tok,
                                   Q36_N_EMBD, Q36_N_FF_EXP, Q36_N_EMBD, Q36_N_EXPERT)) {
        return false;
    }
    if (!q36_gpu_swiglu_tensor(rt->ffn_shared_mid, rt->ffn_shared_gate, rt->ffn_shared_up,
                               n_tok * Q36_N_FF_SHARED, 0.0f, 1.0f)) {
        return false;
    }
    if (!q36_gpu_tensor_matmul_scaled(m, l->ffn_down_shexp, rt->ffn_shared_mid, rt->ffn_shared_out,
                                      Q36_N_FF_SHARED, Q36_N_EMBD, n_tok,
                                      q36_tensor_scalar_or(m, l->ffn_down_shexp_scale, 1.0f))) {
        return false;
    }
    /* Shared-expert tail on the host: the CPU compiles "out += shared * g"
     * to one fma after the down projection's scale, so the fused rounding
     * must be reproduced with fmaf() on the unscaled-by-g down output. */
    (void)shared;
    (void)scalar;
    (void)outp;
    return q36_gpu_ffn_tail_tensor(out, rt->ffn_shared_out, rt->ffn_scalar, Q36_N_EMBD, n_tok) != 0;
}

static bool q36_forward_ffn_vulkan(q36_session *s,
                                   const q36_layer_weights *l,
                                   uint32_t il,
                                   uint32_t n_tok,
                                   const q36_gpu_tensor *inp,
                                   q36_gpu_tensor *out) {
    q36_engine *e;
    if (!s) return false;
    e = s->engine;
    return q36_forward_ffn_vulkan_model((q36_vulkan_runtime *)s->runtime,
                                        &e->model,
                                        l,
                                        e->expert_weights_scale,
                                        il,
                                        n_tok,
                                        inp,
                                        out,
                                        false);
}

/* RoPE stays on the host even on the Vulkan path: the rotated values feed KV
 * storage, so they must match the CPU reference bit for bit, and only the
 * q36_apply_rope_one() body compiled in this unit guarantees that.
 * Token row t rotates at absolute position pos0 + t. */
static bool q36_vulkan_rope_heads(q36_gpu_tensor *x, uint32_t n_head, uint32_t pos0, uint32_t n_tok, bool quality) {
    const char *env = getenv("Q36_VK_GPU_ROPE");
    bool use_gpu = !env || !env[0] || env[0] != '0';
    if (!quality && use_gpu) return q36_gpu_rope_qwen_rows_tensor(x, n_head, pos0, n_tok) != 0;
    q36_vulkan_rope_ctx ctx;
    float *xp;
    if (q36_gpu_tensor_bytes(x) < (uint64_t)n_tok * n_head * Q36_N_HEAD_DIM * sizeof(float)) return false;
    xp = q36_gpu_tensor_contents_named(x, "submit_wait_host_rope");
    if (!xp) return false;
    ctx.x = xp;
    ctx.n_head = n_head;
    ctx.pos0 = pos0;
    if ((uint64_t)n_tok * n_head * Q36_N_HEAD_DIM >= 4096u) q36_gpu_parallel_for_rows(n_tok, 2, q36_vulkan_rope_rows, &ctx);
    else q36_vulkan_rope_rows(&ctx, 0, n_tok);
    return true;
}

static bool q36_vulkan_prepare_attn_heads(q36_gpu_tensor *dst,
                                          const q36_gpu_tensor *src,
                                          const q36_model *m,
                                          const q36_tensor *norm,
                                          uint32_t src_stride,
                                          uint32_t n_head,
                                          uint32_t pos0,
                                          uint32_t n_tok,
                                          bool quality) {
    const char *env = getenv("Q36_VK_NORM_ROPE");
    bool fused = !env || !env[0] || env[0] != '0';
    if (!quality && fused) {
        return q36_gpu_rms_norm_rope_qwen_rows_tensor(dst, src,
                                                       m->map, m->size,
                                                       norm->abs_offset,
                                                       src_stride, n_head,
                                                       pos0, n_tok,
                                                       Q36_RMS_EPS) != 0;
    }
    if (dst != src && !q36_gpu_extract_full_attn_q_tensor(dst, src, n_tok)) return false;
    if (!q36_gpu_rms_norm_weight_rows_tensor(dst, dst,
                                             m->map, m->size, norm->abs_offset,
                                             Q36_N_HEAD_DIM, n_tok * n_head,
                                             Q36_RMS_EPS)) {
        return false;
    }
    return q36_vulkan_rope_heads(dst, n_head, pos0, n_tok, quality);
}

static bool q36_forward_full_attn_vulkan_model(q36_vulkan_runtime *rt,
                                               const q36_model *m,
                                               const q36_layer_weights *l,
                                               q36_vulkan_full_attn_cache *cache,
                                               bool quality,
                                               uint32_t il,
                                               uint32_t pos0,
                                               uint32_t n_tok,
                                               const q36_gpu_tensor *inp,
                                               q36_gpu_tensor *out) {
    if (!rt || !m || !l || !cache || !inp || !out) return false;
    if ((l->attn_q->type != Q36_TENSOR_Q8_0 || l->attn_k->type != Q36_TENSOR_Q8_0 ||
         l->attn_v->type != Q36_TENSOR_Q8_0) &&
        !q36_gpu_quantize_q8_k_tensor(rt->inp_q8, inp, Q36_N_EMBD, n_tok)) {
        return false;
    }
    if (!q36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_q, inp, rt->inp_q8, rt->attn_qg,
                                                  Q36_N_EMBD, Q36_N_HEAD * Q36_N_HEAD_DIM * 2u, n_tok,
                                                  q36_tensor_scalar_or(m, l->attn_q_scale, 1.0f))) {
        fprintf(stderr, "q36: full_attn attn_q failed at layer=%u pos=%u\n", il, pos0);
        return false;
    }
    if (!q36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_k, inp, rt->inp_q8, rt->attn_k,
                                                  Q36_N_EMBD, Q36_N_HEAD_KV * Q36_N_HEAD_DIM, n_tok,
                                                  q36_tensor_scalar_or(m, l->attn_k_scale, 1.0f))) {
        fprintf(stderr, "q36: full_attn attn_k failed at layer=%u pos=%u\n", il, pos0);
        return false;
    }
    if (!q36_gpu_tensor_matmul_q8_or_float_scaled(m, l->attn_v, inp, rt->inp_q8, rt->attn_v,
                                                  Q36_N_EMBD, Q36_N_HEAD_KV * Q36_N_VALUE_DIM, n_tok,
                                                  q36_tensor_scalar_or(m, l->attn_v_scale, 1.0f))) {
        fprintf(stderr, "q36: full_attn attn_v failed at layer=%u pos=%u\n", il, pos0);
        return false;
    }
    if (!q36_vulkan_prepare_attn_heads(rt->attn_q, rt->attn_qg, m, l->attn_q_norm,
                                       Q36_N_HEAD_DIM * 2u, Q36_N_HEAD,
                                       pos0, n_tok, quality)) return false;
    {
        const char *env = getenv("Q36_VK_NORM_ROPE_KV");
        bool fused_kv = !quality && (!env || !env[0] || env[0] != '0') &&
                        cache->type_k == 0u && cache->type_v == 0u &&
                        cache->k_row_bytes == Q36_N_HEAD_KV * Q36_N_HEAD_DIM * sizeof(uint16_t) &&
                        cache->v_row_bytes == Q36_N_HEAD_KV * Q36_N_VALUE_DIM * sizeof(uint16_t);
        if (fused_kv) {
            if (!q36_gpu_rms_norm_rope_qwen_kv_store_tensor(
                    cache->k, cache->v, rt->attn_k, rt->attn_v,
                    m->map, m->size, l->attn_k_norm->abs_offset,
                    Q36_N_HEAD_DIM, Q36_N_HEAD_KV,
                    pos0, n_tok, cache->cap, Q36_RMS_EPS)) {
                return false;
            }
        } else if (!quality && (!env || !env[0] || env[0] != '0') &&
                   cache->type_k >= 1u && cache->type_k <= 2u &&
                   cache->type_v >= 1u && cache->type_v <= 2u &&
                   (Q36_N_HEAD_DIM % 32u) == 0u &&
                   (Q36_N_VALUE_DIM % 32u) == 0u &&
                   q36_gpu_rms_norm_rope_qwen_kv_store_quant_tensor(
                       cache->k, cache->v, rt->attn_k, rt->attn_v,
                       m->map, m->size, l->attn_k_norm->abs_offset,
                       Q36_N_HEAD_DIM, Q36_N_HEAD_KV,
                       pos0, n_tok, cache->cap, Q36_RMS_EPS,
                       cache->k_row_bytes, cache->v_row_bytes)) {
            /* Fused quantized KV store succeeded */
        } else {
            if (!q36_vulkan_prepare_attn_heads(rt->attn_k, rt->attn_k, m, l->attn_k_norm,
                                               Q36_N_HEAD_DIM, Q36_N_HEAD_KV,
                                               pos0, n_tok, quality)) return false;
            if (!q36_gpu_attn_kv_store_tensor(cache->k, cache->v,
                                              rt->attn_k, rt->attn_v,
                                              pos0, n_tok, cache->cap,
                                              Q36_N_HEAD_KV * Q36_N_HEAD_DIM,
                                              Q36_N_HEAD_KV * Q36_N_VALUE_DIM,
                                              cache->type_k,
                                              cache->type_v,
                                              cache->k_row_bytes,
                                              cache->v_row_bytes)) {
                return false;
            }
        }
    }
    if (!q36_gpu_attn_decode_tensor(rt->attn_out,
                                    rt->attn_q,
                                    rt->attn_qg,
                                     cache->k,
                                     cache->v,
                                     rt->scores,
                                     m->map,
                                     m->size,
                                     l->attn_sinks ? l->attn_sinks->abs_offset : 0,
                                    l->attn_sinks != NULL,
                                    pos0,
                                    n_tok,
                                    Q36_N_HEAD,
                                    Q36_N_HEAD_KV,
                                    Q36_N_HEAD_DIM,
                                    cache->type_k,
                                    cache->type_v,
                                    cache->k_row_bytes,
                                    cache->v_row_bytes)) {
        return false;
    }
    if (!q36_gpu_tensor_matmul_scaled(m, l->attn_output, rt->attn_out, out,
                                      Q36_N_SSM_INNER, Q36_N_EMBD, n_tok,
                                      q36_tensor_scalar_or(m, l->attn_output_scale, 1.0f))) {
        fprintf(stderr, "q36: full_attn attn_output failed at layer=%u pos=%u\n", il, pos0);
        return false;
    }
    return true;
}

static bool q36_forward_full_attn_vulkan(q36_session *s,
                                         const q36_layer_weights *l,
                                         uint32_t il,
                                         uint32_t pos0,
                                         uint32_t n_tok,
                                         const q36_gpu_tensor *inp,
                                         q36_gpu_tensor *out) {
    q36_engine *e;
    q36_vulkan_runtime *rt;
    if (!s) return false;
    e = s->engine;
    rt = (q36_vulkan_runtime *)s->runtime;
    return q36_forward_full_attn_vulkan_model(rt,
                                              &e->model,
                                              l,
                                              &rt->full[il],
                                              e->quality,
                                              il,
                                              pos0,
                                              n_tok,
                                              inp,
                                              out);
}

static bool q36_forward_recurrent_vulkan(q36_session *s,
                                         const q36_layer_weights *l,
                                         uint32_t il,
                                         uint32_t n_tok,
                                         const q36_gpu_tensor *inp,
                                         q36_gpu_tensor *out) {
    q36_engine *e;
    q36_vulkan_runtime *rt;
    q36_vulkan_recurrent_cache *cache;
    if (!s || !l || !inp || !out) return false;
    e = s->engine;
    rt = (q36_vulkan_runtime *)s->runtime;
    cache = &rt->recurrent[il];
    if ((l->attn_qkv->type != Q36_TENSOR_Q8_0 || l->attn_gate->type != Q36_TENSOR_Q8_0) &&
        !q36_gpu_quantize_q8_k_tensor(rt->inp_q8, inp, Q36_N_EMBD, n_tok)) {
        return false;
    }
    if (!q36_gpu_tensor_matmul_q8_or_float_scaled(&e->model, l->attn_qkv, inp, rt->inp_q8, rt->recur_qkv,
                                                  Q36_N_EMBD, Q36_N_SSM_CONV_DIM, n_tok,
                                                  q36_tensor_scalar_or(&e->model, l->attn_qkv_scale, 1.0f))) {
        fprintf(stderr, "q36: recurrent attn_qkv failed at layer=%u\n", il);
        return false;
    }
    /* The z gate dispatch records before the host conv step so the flush
     * the conv's qkv read triggers completes both projections. */
    if (!q36_gpu_tensor_matmul_q8_or_float_scaled(&e->model, l->attn_gate, inp, rt->inp_q8, rt->recur_z,
                                                  Q36_N_EMBD, Q36_N_SSM_INNER, n_tok,
                                                  q36_tensor_scalar_or(&e->model, l->attn_gate_scale, 1.0f))) {
        fprintf(stderr, "q36: recurrent attn_gate failed at layer=%u\n", il);
        return false;
    }
    {
        if (!rt->recur_conv_fused) {
            if (!q36_gpu_recurrent_conv_step_tensor(cache->conv, rt->recur_qkv,
                                                    rt->recur_window, n_tok)) return false;
            if (!q36_gpu_ssm_conv_silu_tensor(rt->recur_conv,
                                              rt->recur_window,
                                              e->model.map,
                                              e->model.size,
                                              l->ssm_conv1d->abs_offset,
                                              Q36_N_SSM_CONV_DIM,
                                              Q36_N_SSM_CONV,
                                              n_tok)) return false;
        } else if (!q36_gpu_recurrent_conv_silu_tensor(
                       cache->conv, rt->recur_qkv, rt->recur_conv,
                       e->model.map, e->model.size, l->ssm_conv1d->abs_offset,
                       Q36_N_SSM_CONV_DIM, Q36_N_SSM_CONV, n_tok)) {
            return false;
        }
    }
    if (e->quality && !q36_gpu_tensor_all_finite(rt->recur_conv, n_tok * Q36_N_SSM_CONV_DIM)) {
        fprintf(stderr, "q36: recurrent conv non-finite at layer=%u\n", il);
        return false;
    }
    if (!q36_gpu_tensor_matmul_scaled(&e->model, l->ssm_beta, inp, rt->recur_beta,
                                      Q36_N_EMBD, Q36_N_SSM_DT_RANK, n_tok,
                                      q36_tensor_scalar_or(&e->model, l->ssm_beta_scale, 1.0f))) {
        fprintf(stderr, "q36: recurrent ssm_beta failed at layer=%u\n", il);
        return false;
    }
    if (!q36_gpu_tensor_matmul_scaled(&e->model, l->ssm_alpha, inp, rt->recur_alpha,
                                      Q36_N_EMBD, Q36_N_SSM_DT_RANK, n_tok,
                                      q36_tensor_scalar_or(&e->model, l->ssm_alpha_scale, 1.0f))) {
        fprintf(stderr, "q36: recurrent ssm_alpha failed at layer=%u\n", il);
        return false;
    }
    {
        const char *env = getenv("Q36_VK_DELTA_QKV");
        bool fused = !e->quality && !e->ssd_streaming &&
                     (!env || !env[0] || env[0] != '0') &&
                     q36_gpu_delta_qkv_l2_norm_tensor(
                         rt->recur_q, rt->recur_k, rt->recur_v, rt->recur_conv,
                         Q36_N_SSM_DT_RANK, Q36_N_SSM_GROUP, Q36_N_SSM_STATE,
                         Q36_N_SSM_CONV_DIM, n_tok, Q36_RMS_EPS);
        if (!fused) {
            if (!q36_gpu_delta_qk_l2_norm_tensor(rt->recur_q,
                                                 rt->recur_k,
                                                 rt->recur_conv,
                                                 Q36_N_SSM_DT_RANK,
                                                 Q36_N_SSM_GROUP,
                                                 Q36_N_SSM_STATE,
                                                 Q36_N_SSM_CONV_DIM,
                                                 n_tok,
                                                 Q36_RMS_EPS)) {
                return false;
            }
            if (!q36_gpu_extract_recurrent_v_tensor(rt->recur_v, rt->recur_conv, n_tok)) return false;
        }
    }
    if (!q36_gpu_delta_net_gates_tensor(rt->recur_gb,
                                        rt->recur_alpha,
                                        rt->recur_beta,
                                        e->model.map,
                                        e->model.size,
                                        l->ssm_dt->abs_offset,
                                        l->ssm_a->abs_offset,
                                        Q36_N_SSM_DT_RANK,
                                        n_tok)) {
        return false;
    }
    if (e->quality && !q36_gpu_tensor_all_finite(rt->recur_gb, n_tok * Q36_N_SSM_DT_RANK * 2u)) {
        fprintf(stderr, "q36: recurrent gate non-finite at layer=%u\n", il);
        return false;
    }
    if (!q36_gpu_delta_net_decode_tensor(cache->state,
                                         rt->recur_q,
                                         rt->recur_k,
                                         rt->recur_v,
                                         rt->recur_gb,
                                         rt->recur_proj,
                                         Q36_N_SSM_DT_RANK,
                                         Q36_N_SSM_STATE,
                                         n_tok)) {
        return false;
    }
    if (e->quality && !q36_gpu_tensor_all_finite(rt->recur_proj, n_tok * Q36_N_SSM_INNER)) {
        fprintf(stderr, "q36: recurrent delta decode non-finite at layer=%u\n", il);
        return false;
    }
    if (!q36_gpu_rms_norm_weight_rows_tensor(rt->recur_proj, rt->recur_proj,
                                             e->model.map, e->model.size, l->ssm_norm->abs_offset,
                                             Q36_N_SSM_STATE, n_tok * Q36_N_SSM_DT_RANK, Q36_RMS_EPS)) {
        return false;
    }
    if (!q36_gpu_swiglu_tensor(rt->recur_proj, rt->recur_z, rt->recur_proj,
                               n_tok * Q36_N_SSM_INNER, 0.0f, 1.0f)) {
        return false;
    }
    if (e->quality && !q36_gpu_tensor_all_finite(rt->recur_proj, n_tok * Q36_N_SSM_INNER)) {
        fprintf(stderr, "q36: recurrent gated proj non-finite at layer=%u\n", il);
        return false;
    }
    if (!q36_gpu_tensor_matmul_scaled(&e->model, l->ssm_out, rt->recur_proj, out,
                                      Q36_N_SSM_INNER, Q36_N_EMBD, n_tok,
                                      q36_tensor_scalar_or(&e->model, l->ssm_out_scale, 1.0f))) {
        fprintf(stderr, "q36: recurrent ssm_out failed at layer=%u\n", il);
        return false;
    }
    return true;
}

/* Forward n_tok prompt tokens at positions pos0..pos0+n_tok-1 through the
 * Vulkan runtime, optionally leaving the last token's logits in s->logits.
 * Each token runs the exact arithmetic of the single-token path; batching
 * only widens the matmul dispatches and lets the host parity ops process
 * the whole chunk per GPU round-trip, which is where prefill time went. */
static bool q36_forward_tokens_vulkan_into(q36_session *s,
                                           const int *tokens,
                                           uint32_t n_tok,
                                           uint32_t pos0,
                                           q36_gpu_tensor *logits_out,
                                           bool logits_all_rows) {
    q36_engine *e;
    q36_vulkan_runtime *rt;
    if (!s || !s->runtime || !tokens || n_tok == 0) return false;
    e = s->engine;
    rt = (q36_vulkan_runtime *)s->runtime;
    if (n_tok > rt->prefill_cap) return false;
    q36_gpu_stream_expert_cache_note_tokens(n_tok);
    /* Embed into the host-only stage, then a queued device copy into the
     * GPU-written hidden tensor: mapping hidden here would wait out the
     * previous chunk's whole in-flight batch. */
    {
        q36_gpu_tensor *stage = rt->embed_stage[rt->embed_flip & 1u];
        rt->embed_flip ^= 1u;
        if (!q36_gpu_embed_tokens(&e->model, e->weights.token_embd, tokens, n_tok, stage)) {
            fprintf(stderr, "q36: embed failed for %u tokens at pos=%u\n", n_tok, pos0);
            return false;
        }
        if (!q36_gpu_copy_f32_tensor(rt->hidden, stage, n_tok * Q36_N_EMBD)) return false;
    }
    /* Every residual add feeds the next RMS norm (post-attention norm, the
     * next layer's attn norm, or the final output norm), so both adds fuse
     * with their norm; only layer 0's input norm stands alone. */
    if (!q36_gpu_rms_norm_weight_rows_tensor(rt->norm, rt->hidden,
                                             e->model.map, e->model.size,
                                             e->weights.layer[0].attn_norm->abs_offset,
                                             Q36_N_EMBD, n_tok, Q36_RMS_EPS)) {
        return false;
    }
    for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
        const q36_layer_weights *l = &e->weights.layer[il];
        if (l->kind == Q36_LAYER_FULL_ATTN) {
            if (!q36_forward_full_attn_vulkan(s, l, il, pos0, n_tok, rt->norm, rt->next_hidden)) {
                fprintf(stderr, "q36: full attention block failed at pos=%u layer=%u\n", pos0, il);
                return false;
            }
        } else {
            if (!q36_forward_recurrent_vulkan(s, l, il, n_tok, rt->norm, rt->next_hidden)) {
                fprintf(stderr, "q36: recurrent block failed at pos=%u layer=%u\n", pos0, il);
                return false;
            }
        }
        if (e->directional_steering_attn_scale != 0.0f &&
            !q36_gpu_directional_steering_project_tensor(
                rt->next_hidden,
                e->directional_steering_gpu,
                il,
                Q36_N_EMBD,
                n_tok,
                e->directional_steering_attn_scale)) {
            return false;
        }
        if (e->quality && !q36_gpu_tensor_all_finite(rt->next_hidden, n_tok * Q36_N_EMBD)) {
            fprintf(stderr, "q36: non-finite attn output at pos=%u layer=%u kind=%s\n",
                    pos0, il, l->kind == Q36_LAYER_FULL_ATTN ? "full" : "recurrent");
            return false;
        }
        if (!q36_gpu_add_rms_norm_tensor(rt->norm, rt->next_hidden, rt->next_hidden, rt->hidden,
                                         e->model.map, e->model.size, l->post_attention_norm->abs_offset,
                                         Q36_N_EMBD, n_tok, Q36_RMS_EPS)) {
            return false;
        }
        if (!q36_forward_ffn_vulkan(s, l, il, n_tok, rt->norm, rt->hidden)) {
            fprintf(stderr, "q36: ffn block failed at pos=%u layer=%u\n", pos0, il);
            return false;
        }
        if (e->directional_steering_ffn_scale != 0.0f &&
            !q36_gpu_directional_steering_project_tensor(
                rt->hidden,
                e->directional_steering_gpu,
                il,
                Q36_N_EMBD,
                n_tok,
                e->directional_steering_ffn_scale)) {
            return false;
        }
        if (e->quality && !q36_gpu_tensor_all_finite(rt->hidden, n_tok * Q36_N_EMBD)) {
            fprintf(stderr, "q36: non-finite ffn output at pos=%u layer=%u\n", pos0, il);
            return false;
        }
        {
            const q36_tensor *next_norm = il + 1u < Q36_N_LAYER ?
                e->weights.layer[il + 1u].attn_norm : e->weights.output_norm;
            if (!q36_gpu_add_rms_norm_tensor(rt->norm, rt->hidden, rt->next_hidden, rt->hidden,
                                             e->model.map, e->model.size, next_norm->abs_offset,
                                             Q36_N_EMBD, n_tok, Q36_RMS_EPS)) {
                return false;
            }
        }
        if (e->quality && !q36_gpu_tensor_all_finite(rt->hidden, n_tok * Q36_N_EMBD)) {
            fprintf(stderr, "q36: non-finite hidden state at pos=%u layer=%u\n", pos0, il);
            return false;
        }
    }
    if (e->quality && !q36_gpu_tensor_all_finite(rt->norm, n_tok * Q36_N_EMBD)) {
        fprintf(stderr, "q36: non-finite output norm at pos=%u\n", pos0);
        return false;
    }
    if (logits_out) {
        bool ok;
        if (logits_all_rows) {
            ok = q36_gpu_tensor_matmul_scaled(&e->model,
                                              e->weights.output,
                                              rt->norm,
                                              logits_out,
                                              Q36_N_EMBD,
                                              Q36_N_VOCAB,
                                              n_tok,
                                              q36_tensor_scalar_or(&e->model, e->weights.output_scale, 1.0f));
        } else {
            q36_gpu_tensor *last = q36_gpu_tensor_view(rt->norm,
                                                       (uint64_t)(n_tok - 1u) * Q36_N_EMBD * sizeof(float),
                                                       (uint64_t)Q36_N_EMBD * sizeof(float));
            ok = last &&
                 q36_gpu_tensor_matmul_scaled(&e->model, e->weights.output, last, logits_out,
                                              Q36_N_EMBD, Q36_N_VOCAB, 1,
                                              q36_tensor_scalar_or(&e->model, e->weights.output_scale, 1.0f));
            q36_gpu_tensor_free(last);
        }
        if (!ok) {
            fprintf(stderr, "q36: output projection failed at pos=%u\n", pos0);
            return false;
        }
        if (e->quality && !q36_gpu_tensor_all_finite(logits_out, Q36_N_VOCAB * (logits_all_rows ? n_tok : 1u))) {
            fprintf(stderr, "q36: non-finite logits at pos=%u\n", pos0);
            return false;
        }
    }
    return true;
}

static bool q36_vulkan_update_last_h(q36_vulkan_runtime *rt, uint32_t row) {
    if (!rt || !rt->last_h) return false;
    return q36_gpu_tensor_copy(rt->last_h,
                               0,
                               rt->norm,
                               (uint64_t)row * Q36_N_EMBD * sizeof(float),
                               (uint64_t)Q36_N_EMBD * sizeof(float)) != 0;
}

static q36_gpu_tensor *q36_vulkan_row(q36_gpu_tensor *t, uint32_t row,
                                      uint64_t width) {
    uint64_t bytes = width * sizeof(float);
    return q36_gpu_tensor_view(t, (uint64_t)row * bytes, bytes);
}

static bool q36_sessions_full_attn_vulkan(q36_decode_item *items, int count,
                                          const q36_layer_weights *l,
                                          uint32_t il,
                                          q36_gpu_tensor *inp,
                                          q36_gpu_tensor *out) {
    q36_session *first = items[0].session;
    q36_engine *e = first->engine;
    q36_vulkan_runtime *batch = first->runtime;
    uint32_t rows = (uint32_t)count;

    if ((l->attn_q->type != Q36_TENSOR_Q8_0 ||
         l->attn_k->type != Q36_TENSOR_Q8_0 ||
         l->attn_v->type != Q36_TENSOR_Q8_0) &&
        !q36_gpu_quantize_q8_k_tensor(batch->inp_q8, inp, Q36_N_EMBD, rows)) {
        return false;
    }
    if (!q36_gpu_tensor_matmul_q8_or_float_scaled(
            &e->model, l->attn_q, inp, batch->inp_q8, batch->attn_qg,
            Q36_N_EMBD, Q36_N_HEAD * Q36_N_HEAD_DIM * 2u, rows,
            q36_tensor_scalar_or(&e->model, l->attn_q_scale, 1.0f)) ||
        !q36_gpu_tensor_matmul_q8_or_float_scaled(
            &e->model, l->attn_k, inp, batch->inp_q8, batch->attn_k,
            Q36_N_EMBD, Q36_N_HEAD_KV * Q36_N_HEAD_DIM, rows,
            q36_tensor_scalar_or(&e->model, l->attn_k_scale, 1.0f)) ||
        !q36_gpu_tensor_matmul_q8_or_float_scaled(
            &e->model, l->attn_v, inp, batch->inp_q8, batch->attn_v,
            Q36_N_EMBD, Q36_N_HEAD_KV * Q36_N_VALUE_DIM, rows,
            q36_tensor_scalar_or(&e->model, l->attn_v_scale, 1.0f))) {
        return false;
    }

    for (int i = 0; i < count; i++) {
        q36_session *s = items[i].session;
        q36_vulkan_runtime *rt = s->runtime;
        q36_vulkan_full_attn_cache *cache = &rt->full[il];
        uint32_t pos = (uint32_t)s->checkpoint.len;
        q36_gpu_tensor *qg = q36_vulkan_row(batch->attn_qg, (uint32_t)i,
                Q36_N_HEAD * Q36_N_HEAD_DIM * 2u);
        q36_gpu_tensor *q = q36_vulkan_row(batch->attn_q, (uint32_t)i,
                Q36_N_HEAD * Q36_N_HEAD_DIM);
        q36_gpu_tensor *k = q36_vulkan_row(batch->attn_k, (uint32_t)i,
                Q36_N_HEAD_KV * Q36_N_HEAD_DIM);
        q36_gpu_tensor *v = q36_vulkan_row(batch->attn_v, (uint32_t)i,
                Q36_N_HEAD_KV * Q36_N_VALUE_DIM);
        q36_gpu_tensor *attn = q36_vulkan_row(batch->attn_out, (uint32_t)i,
                Q36_N_SSM_INNER);
        bool ok = qg && q && k && v && attn &&
            q36_vulkan_prepare_attn_heads(q, qg, &e->model, l->attn_q_norm,
                    Q36_N_HEAD_DIM * 2u, Q36_N_HEAD, pos, 1, e->quality);
        bool fused_kv = !e->quality && cache->type_k == Q36_KV_CACHE_F16 &&
                        cache->type_v == Q36_KV_CACHE_F16 &&
                        cache->k_row_bytes == Q36_N_HEAD_KV * Q36_N_HEAD_DIM * sizeof(uint16_t) &&
                        cache->v_row_bytes == Q36_N_HEAD_KV * Q36_N_VALUE_DIM * sizeof(uint16_t);
        const char *fused_env = getenv("Q36_VK_NORM_ROPE_KV");
        fused_kv = fused_kv && (!fused_env || !fused_env[0] || fused_env[0] != '0');
        if (ok && fused_kv) {
            ok = q36_gpu_rms_norm_rope_qwen_kv_store_tensor(
                    cache->k, cache->v, k, v, e->model.map, e->model.size,
                    l->attn_k_norm->abs_offset, Q36_N_HEAD_DIM,
                    Q36_N_HEAD_KV, pos, 1, cache->cap, Q36_RMS_EPS) != 0;
        } else if (ok) {
            ok = q36_vulkan_prepare_attn_heads(k, k, &e->model,
                    l->attn_k_norm, Q36_N_HEAD_DIM, Q36_N_HEAD_KV,
                    pos, 1, e->quality) &&
                 q36_gpu_attn_kv_store_tensor(
                    cache->k, cache->v, k, v, pos, 1, cache->cap,
                    Q36_N_HEAD_KV * Q36_N_HEAD_DIM,
                    Q36_N_HEAD_KV * Q36_N_VALUE_DIM,
                    cache->type_k, cache->type_v,
                    cache->k_row_bytes, cache->v_row_bytes) != 0;
        }
        if (ok) {
            ok = q36_gpu_attn_decode_tensor(
                    attn, q, qg, cache->k, cache->v, rt->scores,
                    e->model.map, e->model.size,
                    l->attn_sinks ? l->attn_sinks->abs_offset : 0,
                    l->attn_sinks != NULL, pos, 1,
                    Q36_N_HEAD, Q36_N_HEAD_KV, Q36_N_HEAD_DIM,
                    cache->type_k, cache->type_v,
                    cache->k_row_bytes, cache->v_row_bytes) != 0;
        }
        q36_gpu_tensor_free(attn);
        q36_gpu_tensor_free(v);
        q36_gpu_tensor_free(k);
        q36_gpu_tensor_free(q);
        q36_gpu_tensor_free(qg);
        if (!ok) return false;
    }

    return q36_gpu_tensor_matmul_scaled(
            &e->model, l->attn_output, batch->attn_out, out,
            Q36_N_SSM_INNER, Q36_N_EMBD, rows,
            q36_tensor_scalar_or(&e->model, l->attn_output_scale, 1.0f));
}

static bool q36_sessions_recurrent_vulkan(q36_decode_item *items, int count,
                                          const q36_layer_weights *l,
                                          uint32_t il,
                                          q36_gpu_tensor *inp,
                                          q36_gpu_tensor *out) {
    q36_session *first = items[0].session;
    q36_engine *e = first->engine;
    q36_vulkan_runtime *batch = first->runtime;
    uint32_t rows = (uint32_t)count;

    if ((l->attn_qkv->type != Q36_TENSOR_Q8_0 ||
         l->attn_gate->type != Q36_TENSOR_Q8_0) &&
        !q36_gpu_quantize_q8_k_tensor(batch->inp_q8, inp, Q36_N_EMBD, rows)) {
        return false;
    }
    if (!q36_gpu_tensor_matmul_q8_or_float_scaled(
            &e->model, l->attn_qkv, inp, batch->inp_q8, batch->recur_qkv,
            Q36_N_EMBD, Q36_N_SSM_CONV_DIM, rows,
            q36_tensor_scalar_or(&e->model, l->attn_qkv_scale, 1.0f)) ||
        !q36_gpu_tensor_matmul_q8_or_float_scaled(
            &e->model, l->attn_gate, inp, batch->inp_q8, batch->recur_z,
            Q36_N_EMBD, Q36_N_SSM_INNER, rows,
            q36_tensor_scalar_or(&e->model, l->attn_gate_scale, 1.0f)) ||
        !q36_gpu_tensor_matmul_scaled(
            &e->model, l->ssm_beta, inp, batch->recur_beta,
            Q36_N_EMBD, Q36_N_SSM_DT_RANK, rows,
            q36_tensor_scalar_or(&e->model, l->ssm_beta_scale, 1.0f)) ||
        !q36_gpu_tensor_matmul_scaled(
            &e->model, l->ssm_alpha, inp, batch->recur_alpha,
            Q36_N_EMBD, Q36_N_SSM_DT_RANK, rows,
            q36_tensor_scalar_or(&e->model, l->ssm_alpha_scale, 1.0f))) {
        return false;
    }

    for (int i = 0; i < count; i++) {
        q36_vulkan_runtime *rt = items[i].session->runtime;
        q36_vulkan_recurrent_cache *cache = &rt->recurrent[il];
        q36_gpu_tensor *qkv = q36_vulkan_row(batch->recur_qkv, (uint32_t)i,
                Q36_N_SSM_CONV_DIM);
        q36_gpu_tensor *conv = q36_vulkan_row(batch->recur_conv, (uint32_t)i,
                Q36_N_SSM_CONV_DIM);
        q36_gpu_tensor *q = q36_vulkan_row(batch->recur_q, (uint32_t)i,
                Q36_N_SSM_INNER);
        q36_gpu_tensor *k = q36_vulkan_row(batch->recur_k, (uint32_t)i,
                Q36_N_SSM_INNER);
        q36_gpu_tensor *v = q36_vulkan_row(batch->recur_v, (uint32_t)i,
                Q36_N_SSM_INNER);
        q36_gpu_tensor *gb = q36_vulkan_row(batch->recur_gb, (uint32_t)i,
                Q36_N_SSM_DT_RANK * 2u);
        q36_gpu_tensor *alpha = q36_vulkan_row(batch->recur_alpha, (uint32_t)i,
                Q36_N_SSM_DT_RANK);
        q36_gpu_tensor *beta = q36_vulkan_row(batch->recur_beta, (uint32_t)i,
                Q36_N_SSM_DT_RANK);
        q36_gpu_tensor *proj = q36_vulkan_row(batch->recur_proj, (uint32_t)i,
                Q36_N_SSM_INNER);
        bool ok = qkv && conv && q && k && v && gb && alpha && beta && proj;
        if (ok) {
            if (rt->recur_conv_fused) {
                ok = q36_gpu_recurrent_conv_silu_tensor(
                        cache->conv, qkv, conv, e->model.map, e->model.size,
                        l->ssm_conv1d->abs_offset, Q36_N_SSM_CONV_DIM,
                        Q36_N_SSM_CONV, 1) != 0;
            } else {
                ok = q36_gpu_recurrent_conv_step_tensor(
                        cache->conv, qkv, rt->recur_window, 1) != 0 &&
                     q36_gpu_ssm_conv_silu_tensor(
                        conv, rt->recur_window, e->model.map, e->model.size,
                        l->ssm_conv1d->abs_offset, Q36_N_SSM_CONV_DIM,
                        Q36_N_SSM_CONV, 1) != 0;
            }
        }
        if (ok && !q36_gpu_delta_qkv_l2_norm_tensor(
                    q, k, v, conv, Q36_N_SSM_DT_RANK, Q36_N_SSM_GROUP,
                    Q36_N_SSM_STATE, Q36_N_SSM_CONV_DIM, 1, Q36_RMS_EPS)) {
            ok = q36_gpu_delta_qk_l2_norm_tensor(
                    q, k, conv, Q36_N_SSM_DT_RANK, Q36_N_SSM_GROUP,
                    Q36_N_SSM_STATE, Q36_N_SSM_CONV_DIM, 1,
                    Q36_RMS_EPS) != 0 &&
                 q36_gpu_extract_recurrent_v_tensor(v, conv, 1) != 0;
        }
        if (ok) {
            ok = q36_gpu_delta_net_gates_tensor(
                    gb, alpha, beta, e->model.map, e->model.size,
                    l->ssm_dt->abs_offset, l->ssm_a->abs_offset,
                    Q36_N_SSM_DT_RANK, 1) != 0 &&
                 q36_gpu_delta_net_decode_tensor(
                    cache->state, q, k, v, gb, proj,
                    Q36_N_SSM_DT_RANK, Q36_N_SSM_STATE, 1) != 0;
        }
        q36_gpu_tensor_free(proj);
        q36_gpu_tensor_free(beta);
        q36_gpu_tensor_free(alpha);
        q36_gpu_tensor_free(gb);
        q36_gpu_tensor_free(v);
        q36_gpu_tensor_free(k);
        q36_gpu_tensor_free(q);
        q36_gpu_tensor_free(conv);
        q36_gpu_tensor_free(qkv);
        if (!ok) return false;
    }

    if (!q36_gpu_rms_norm_weight_rows_tensor(
            batch->recur_proj, batch->recur_proj,
            e->model.map, e->model.size, l->ssm_norm->abs_offset,
            Q36_N_SSM_STATE, rows * Q36_N_SSM_DT_RANK, Q36_RMS_EPS) ||
        !q36_gpu_swiglu_tensor(batch->recur_proj, batch->recur_z,
                               batch->recur_proj,
                               rows * Q36_N_SSM_INNER, 0.0f, 1.0f)) {
        return false;
    }
    return q36_gpu_tensor_matmul_scaled(
            &e->model, l->ssm_out, batch->recur_proj, out,
            Q36_N_SSM_INNER, Q36_N_EMBD, rows,
            q36_tensor_scalar_or(&e->model, l->ssm_out_scale, 1.0f));
}

static bool q36_sessions_eval_batch_vulkan(q36_decode_item *items, int count) {
    q36_session *first = items[0].session;
    q36_engine *e = first->engine;
    q36_vulkan_runtime *rt = first->runtime;
    uint32_t rows = (uint32_t)count;
    int tokens[8];

    for (int i = 0; i < count; i++) tokens[i] = items[i].token;
    q36_gpu_stream_expert_cache_note_tokens(rows);
    if (!q36_gpu_embed_tokens(&e->model, e->weights.token_embd,
                              tokens, rows, rt->embed_stage[0]) ||
        !q36_gpu_copy_f32_tensor(rt->hidden, rt->embed_stage[0],
                                 rows * Q36_N_EMBD) ||
        !q36_gpu_rms_norm_weight_rows_tensor(
                rt->norm, rt->hidden, e->model.map, e->model.size,
                e->weights.layer[0].attn_norm->abs_offset,
                Q36_N_EMBD, rows, Q36_RMS_EPS)) {
        return false;
    }

    q36_gpu_set_micro_batch(true);
    for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
        const q36_layer_weights *l = &e->weights.layer[il];
        bool ok = l->kind == Q36_LAYER_FULL_ATTN
            ? q36_sessions_full_attn_vulkan(items, count, l, il,
                                            rt->norm, rt->next_hidden)
            : q36_sessions_recurrent_vulkan(items, count, l, il,
                                             rt->norm, rt->next_hidden);
        if (ok && e->directional_steering_attn_scale != 0.0f) {
            ok = q36_gpu_directional_steering_project_tensor(
                    rt->next_hidden, e->directional_steering_gpu, il,
                    Q36_N_EMBD, rows,
                    e->directional_steering_attn_scale) != 0;
        }
        if (ok) {
            ok = q36_gpu_add_rms_norm_tensor(
                    rt->norm, rt->next_hidden, rt->next_hidden, rt->hidden,
                    e->model.map, e->model.size,
                    l->post_attention_norm->abs_offset,
                    Q36_N_EMBD, rows, Q36_RMS_EPS) != 0;
        }
        if (ok) {
            ok = q36_forward_ffn_vulkan_model(
                    rt, &e->model, l, e->expert_weights_scale, il,
                    rows, rt->norm, rt->hidden, true);
        }
        if (ok && e->directional_steering_ffn_scale != 0.0f) {
            ok = q36_gpu_directional_steering_project_tensor(
                    rt->hidden, e->directional_steering_gpu, il,
                    Q36_N_EMBD, rows,
                    e->directional_steering_ffn_scale) != 0;
        }
        if (ok) {
            const q36_tensor *next_norm = il + 1u < Q36_N_LAYER
                ? e->weights.layer[il + 1u].attn_norm : e->weights.output_norm;
            ok = q36_gpu_add_rms_norm_tensor(
                    rt->norm, rt->hidden, rt->next_hidden, rt->hidden,
                    e->model.map, e->model.size, next_norm->abs_offset,
                    Q36_N_EMBD, rows, Q36_RMS_EPS) != 0;
        }
        if (!ok) {
            q36_gpu_set_micro_batch(false);
            return false;
        }
    }
    q36_gpu_set_micro_batch(false);

    if (e->session_batch_logits_cap < rows) {
        q36_gpu_tensor *logits = q36_gpu_tensor_alloc(
                (uint64_t)rows * Q36_N_VOCAB * sizeof(float));
        if (!logits) return false;
        q36_gpu_tensor_free(e->session_batch_logits);
        e->session_batch_logits = logits;
        e->session_batch_logits_cap = rows;
    }
    if (!q36_gpu_tensor_matmul_scaled(
            &e->model, e->weights.output, rt->norm,
            e->session_batch_logits, Q36_N_EMBD, Q36_N_VOCAB, rows,
            q36_tensor_scalar_or(&e->model, e->weights.output_scale, 1.0f))) {
        return false;
    }
    for (int i = 0; i < count; i++) {
        q36_vulkan_runtime *dst = items[i].session->runtime;
        if (!q36_gpu_tensor_copy(
                dst->last_h, 0, rt->norm,
                (uint64_t)i * Q36_N_EMBD * sizeof(float),
                (uint64_t)Q36_N_EMBD * sizeof(float)) ||
            !q36_gpu_tensor_read(
                e->session_batch_logits,
                (uint64_t)i * Q36_N_VOCAB * sizeof(float),
                items[i].session->logits,
                (uint64_t)Q36_N_VOCAB * sizeof(float))) {
            return false;
        }
    }
    return true;
}

static bool q36_forward_tokens_vulkan(q36_session *s, const int *tokens, uint32_t n_tok,
                                      uint32_t pos0, bool compute_logits) {
    q36_vulkan_runtime *rt;
    if (!s) return false;
    rt = (q36_vulkan_runtime *)s->runtime;
    if (!q36_forward_tokens_vulkan_into(s,
                                        tokens,
                                        n_tok,
                                        pos0,
                                        compute_logits ? rt->logits : NULL,
                                        false)) {
        return false;
    }
    if (!q36_vulkan_update_last_h(rt, n_tok - 1u)) return false;
    if (compute_logits) {
        if (!q36_gpu_tensor_read(rt->logits, 0, s->logits, (uint64_t)Q36_N_VOCAB * sizeof(*s->logits))) {
            return false;
        }
    }
    return true;
}

static bool q36_forward_token_vulkan(q36_session *s, int token, uint32_t pos, bool compute_logits) {
    return q36_forward_tokens_vulkan(s, &token, 1, pos, compute_logits);
}

static uint64_t q36_vulkan_recurrent_conv_bytes(void) {
    return (uint64_t)(Q36_N_SSM_CONV - 1u) * Q36_N_SSM_CONV_DIM * sizeof(float);
}

static uint64_t q36_vulkan_recurrent_state_bytes(void) {
    return (uint64_t)Q36_N_SSM_STATE * Q36_N_SSM_STATE * Q36_N_SSM_DT_RANK * sizeof(float);
}

static bool q36_vulkan_spec_frontier_copy(q36_vulkan_recurrent_cache *dst,
                                          const q36_vulkan_recurrent_cache *src) {
    if (!dst || !src || !dst->conv || !src->conv || !dst->state || !src->state) return false;
    if (!q36_gpu_tensor_copy(dst->conv, 0, src->conv, 0, q36_vulkan_recurrent_conv_bytes())) return false;
    if (!q36_gpu_tensor_copy(dst->state, 0, src->state, 0, q36_vulkan_recurrent_state_bytes())) return false;
    return true;
}

static bool q36_vulkan_spec_frontier_snapshot(q36_vulkan_runtime *rt) {
    if (!rt) return false;
    for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
        if (q36_layer_is_full_attention(il)) continue;
        if (!q36_vulkan_spec_frontier_copy(&rt->spec_recurrent[il], &rt->recurrent[il])) return false;
    }
    return true;
}

static bool q36_vulkan_spec_frontier_restore(q36_vulkan_runtime *rt) {
    if (!rt) return false;
    for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
        if (q36_layer_is_full_attention(il)) continue;
        if (!q36_vulkan_spec_frontier_copy(&rt->recurrent[il], &rt->spec_recurrent[il])) return false;
    }
    return true;
}

static bool q36_vulkan_verify_suffix_tops(q36_session *s,
                                          const int *tokens,
                                          uint32_t n_tok,
                                          uint32_t pos0,
                                          int *row_tops) {
    q36_engine *e;
    q36_vulkan_runtime *rt;
    if (!s || !tokens || n_tok == 0 || n_tok > Q36_MTP_MAX_DRAFT) return false;
    e = s->engine;
    rt = (q36_vulkan_runtime *)s->runtime;
    if (!rt || !rt->spec_logits) return false;
    q36_gpu_set_micro_batch(true);
    {
        bool fwd = q36_forward_tokens_vulkan_into(s, tokens, n_tok, pos0, NULL, false) &&
                   q36_gpu_tensor_matmul_scaled(&e->model,
                                                e->weights.output,
                                                rt->norm,
                                                rt->spec_logits,
                                                Q36_N_EMBD,
                                                Q36_N_VOCAB,
                                                n_tok,
                                                q36_tensor_scalar_or(&e->model, e->weights.output_scale, 1.0f));
        q36_gpu_set_micro_batch(false);
        if (!fwd) return false;
    }
    for (uint32_t i = 0; i + 1u < n_tok; i++) {
        if (!q36_gpu_tensor_read(rt->spec_logits,
                                 (uint64_t)i * Q36_N_VOCAB * sizeof(*s->logits),
                                 s->logits,
                                 (uint64_t)Q36_N_VOCAB * sizeof(*s->logits))) return false;
        row_tops[i] = sample_argmax(s->logits, Q36_N_VOCAB);
    }
    return q36_gpu_tensor_read(rt->spec_logits,
                               (uint64_t)(n_tok - 1u) * Q36_N_VOCAB * sizeof(*s->logits),
                               s->logits,
                               (uint64_t)Q36_N_VOCAB * sizeof(*s->logits)) != 0;
}

static bool q36_mtp_concat(q36_gpu_tensor *out,
                           const q36_gpu_tensor *a,
                           const q36_gpu_tensor *b) {
    /* Device copies: mapping these GPU-written tensors on the host would
     * drain the whole in-flight draft pipeline. */
    return q36_gpu_tensor_copy(out, 0, a, 0, (uint64_t)Q36_N_EMBD * sizeof(float)) &&
           q36_gpu_tensor_copy(out, (uint64_t)Q36_N_EMBD * sizeof(float),
                               b, 0, (uint64_t)Q36_N_EMBD * sizeof(float));
}

static bool q36_mtp_eval_vulkan(q36_session *s,
                                int token,
                                uint32_t pos,
                                const q36_gpu_tensor *h_in) {
    q36_engine *e;
    q36_vulkan_runtime *rt;
    q36_mtp_weights *mw;
    q36_layer_weights *l;
    if (!s || !h_in) return false;
    e = s->engine;
    rt = (q36_vulkan_runtime *)s->runtime;
    if (!e || !e->mtp_ready || !rt || !rt->mtp_enabled || !s->mtp_logits) return false;
    mw = &e->mtp_weights;
    l = &mw->block;

    if (!q36_gpu_embed_tokens(&e->mtp_model, mw->token_embd, &token, 1, rt->mtp_tok_embd)) return false;
    if (!q36_gpu_rms_norm_weight_rows_tensor(rt->mtp_e_norm,
                                             rt->mtp_tok_embd,
                                             e->mtp_model.map,
                                             e->mtp_model.size,
                                             mw->enorm->abs_offset,
                                             Q36_N_EMBD,
                                             1,
                                             Q36_RMS_EPS)) return false;
    if (!q36_gpu_rms_norm_weight_rows_tensor(rt->mtp_h_norm,
                                             h_in,
                                             e->mtp_model.map,
                                             e->mtp_model.size,
                                             mw->hnorm->abs_offset,
                                             Q36_N_EMBD,
                                             1,
                                             Q36_RMS_EPS)) return false;
    if (!q36_mtp_concat(rt->mtp_concat, rt->mtp_e_norm, rt->mtp_h_norm)) return false;
    if (!q36_gpu_tensor_matmul_scaled(&e->mtp_model,
                                      mw->eh_proj,
                                      rt->mtp_concat,
                                      rt->mtp_cur,
                                      Q36_N_EMBD * 2u,
                                      Q36_N_EMBD,
                                      1,
                                      1.0f)) return false;

    if (!q36_gpu_rms_norm_weight_rows_tensor(rt->norm,
                                             rt->mtp_cur,
                                             e->mtp_model.map,
                                             e->mtp_model.size,
                                             l->attn_norm->abs_offset,
                                             Q36_N_EMBD,
                                             1,
                                             Q36_RMS_EPS)) return false;
    if (!q36_forward_full_attn_vulkan_model(rt,
                                            &e->mtp_model,
                                            l,
                                            &rt->mtp_full,
                                            e->quality,
                                            Q36_N_LAYER,
                                            pos,
                                            1,
                                            rt->norm,
                                            rt->mtp_next)) return false;
    if (!q36_gpu_add_tensor(rt->mtp_next, rt->mtp_next, rt->mtp_cur, Q36_N_EMBD)) return false;
    if (!q36_gpu_rms_norm_weight_rows_tensor(rt->norm,
                                             rt->mtp_next,
                                             e->mtp_model.map,
                                             e->mtp_model.size,
                                             l->post_attention_norm->abs_offset,
                                             Q36_N_EMBD,
                                             1,
                                             Q36_RMS_EPS)) return false;
    if (!q36_forward_ffn_vulkan_model(rt,
                                      &e->mtp_model,
                                      l,
                                      e->mtp_expert_weights_scale,
                                      Q36_N_LAYER,
                                      1,
                                      rt->norm,
                                      rt->mtp_cur,
                                      false)) return false;
    if (!q36_gpu_add_tensor(rt->mtp_cur, rt->mtp_next, rt->mtp_cur, Q36_N_EMBD)) return false;
    if (!q36_gpu_rms_norm_weight_rows_tensor(rt->mtp_head,
                                             rt->mtp_cur,
                                             e->mtp_model.map,
                                             e->mtp_model.size,
                                             mw->shared_head_norm->abs_offset,
                                             Q36_N_EMBD,
                                             1,
                                             Q36_RMS_EPS)) return false;
    if (!q36_gpu_tensor_matmul_scaled(&e->model,
                                      e->weights.output,
                                      rt->mtp_head,
                                      rt->mtp_logits,
                                      Q36_N_EMBD,
                                      Q36_N_VOCAB,
                                      1,
                                      q36_tensor_scalar_or(&e->model, e->weights.output_scale, 1.0f))) return false;
    if (e->quality && !q36_gpu_tensor_all_finite(rt->mtp_logits, Q36_N_VOCAB)) return false;
    return q36_gpu_tensor_read(rt->mtp_logits,
                               0,
                               s->mtp_logits,
                               (uint64_t)Q36_N_VOCAB * sizeof(*s->mtp_logits)) != 0;
}
#endif

static bool q36_forward_token_cpu(q36_session *s, int token, uint32_t pos, bool compute_logits) {
    q36_engine *e;
    q36_cpu_runtime *rt;
    if (!s || !s->runtime) return false;
    e = s->engine;
    rt = (q36_cpu_runtime *)s->runtime;
    if (!q36_embed_token(e, token, rt->hidden)) {
        fprintf(stderr, "q36: embed failed for token=%d pos=%u\n", token, pos);
        return false;
    }
    for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
        const q36_layer_weights *l = &e->weights.layer[il];
        q36_ref_rms_norm(rt->work0, rt->hidden, (const float *)(e->model.map + l->attn_norm->abs_offset), Q36_N_EMBD, Q36_RMS_EPS);
        if (l->kind == Q36_LAYER_FULL_ATTN) {
            if (!q36_forward_full_attn(e, l, il, pos, rt->work0, rt->next_hidden, rt)) {
                fprintf(stderr, "q36: full attention block failed for token=%d pos=%u layer=%u\n", token, pos, il);
                return false;
            }
        } else {
            if (!q36_forward_recurrent(e, l, il, rt->work0, rt->next_hidden, rt)) {
                fprintf(stderr, "q36: recurrent block failed for token=%d pos=%u layer=%u\n", token, pos, il);
                return false;
            }
        }
        q36_directional_steering_project_rows(rt->next_hidden,
                                               e->directional_steering_dirs,
                                               il,
                                               1,
                                               e->directional_steering_attn_scale);
        if (!q36_all_finite(rt->next_hidden, Q36_N_EMBD)) {
            fprintf(stderr, "q36: non-finite attn output at token=%d pos=%u layer=%u kind=%s\n",
                    token, pos, il, l->kind == Q36_LAYER_FULL_ATTN ? "full" : "recurrent");
            return false;
        }
        for (uint32_t i = 0; i < Q36_N_EMBD; i++) rt->next_hidden[i] += rt->hidden[i];
        q36_ref_rms_norm(rt->work0, rt->next_hidden, (const float *)(e->model.map + l->post_attention_norm->abs_offset), Q36_N_EMBD, Q36_RMS_EPS);
        if (!q36_forward_ffn(e, l, rt->work0, rt->work1, rt)) {
            fprintf(stderr, "q36: ffn block failed for token=%d pos=%u layer=%u\n", token, pos, il);
            return false;
        }
        q36_directional_steering_project_rows(rt->work1,
                                               e->directional_steering_dirs,
                                               il,
                                               1,
                                               e->directional_steering_ffn_scale);
        if (!q36_all_finite(rt->work1, Q36_N_EMBD)) {
            fprintf(stderr, "q36: non-finite ffn output at token=%d pos=%u layer=%u\n", token, pos, il);
            return false;
        }
        for (uint32_t i = 0; i < Q36_N_EMBD; i++) rt->hidden[i] = rt->next_hidden[i] + rt->work1[i];
        if (!q36_all_finite(rt->hidden, Q36_N_EMBD)) {
            fprintf(stderr, "q36: non-finite hidden state at token=%d pos=%u layer=%u\n", token, pos, il);
            return false;
        }
    }
    q36_ref_rms_norm(rt->work0, rt->hidden, (const float *)(e->model.map + e->weights.output_norm->abs_offset), Q36_N_EMBD, Q36_RMS_EPS);
    if (!q36_all_finite(rt->work0, Q36_N_EMBD)) {
        fprintf(stderr, "q36: non-finite output norm at token=%d pos=%u\n", token, pos);
        return false;
    }
    if (compute_logits) {
        if (!q36_tensor_matvec(e, e->weights.output, rt->work0, s->logits, rt->work1, Q36_N_EMBD, Q36_N_VOCAB)) {
            fprintf(stderr, "q36: output projection failed for token=%d pos=%u\n", token, pos);
            return false;
        }
        q36_scale_inplace(s->logits, Q36_N_VOCAB,
                          q36_tensor_scalar_or(&e->model, e->weights.output_scale, 1.0f));
        if (!q36_all_finite(s->logits, Q36_N_VOCAB)) {
            fprintf(stderr, "q36: non-finite logits at token=%d pos=%u\n", token, pos);
            return false;
        }
    }
    return true;
}

static uint64_t next_pow2(uint64_t n) {
    uint64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

static void table_init(str_i32_table *t, uint64_t expected) {
    t->cap = next_pow2(expected * 2 + 16);
    t->entry = xcalloc((size_t)t->cap, sizeof(t->entry[0]));
}

static void table_free(str_i32_table *t) {
    free(t->entry);
    memset(t, 0, sizeof(*t));
}

static void table_put(str_i32_table *t, q36_str key, int value) {
    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(key.ptr, key.len) & mask;
    while (t->entry[i].used) {
        if (q36_str_eq(t->entry[i].key, key)) {
            t->entry[i].value = value;
            return;
        }
        i = (i + 1) & mask;
    }
    t->entry[i].used = true;
    t->entry[i].key = key;
    t->entry[i].value = value;
}

static bool table_get(const str_i32_table *t, const char *ptr, uint64_t len, int *value) {
    uint64_t mask;
    uint64_t i;
    if (t->cap == 0) return false;
    mask = t->cap - 1;
    i = hash_bytes(ptr, len) & mask;
    while (t->entry[i].used) {
        q36_str key = t->entry[i].key;
        if (key.len == len && memcmp(key.ptr, ptr, len) == 0) {
            *value = t->entry[i].value;
            return true;
        }
        i = (i + 1) & mask;
    }
    return false;
}

static void utf8_put(char **p, uint32_t cp) {
    if (cp <= 0x7f) {
        *(*p)++ = (char)cp;
    } else if (cp <= 0x7ff) {
        *(*p)++ = (char)(0xc0 | (cp >> 6));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else if (cp <= 0xffff) {
        *(*p)++ = (char)(0xe0 | (cp >> 12));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else {
        *(*p)++ = (char)(0xf0 | (cp >> 18));
        *(*p)++ = (char)(0x80 | ((cp >> 12) & 0x3f));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    }
}

static uint32_t gpt2_byte_to_codepoint(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174) return b;
    {
        uint32_t n = 0;
        for (uint32_t x = 0; x < 256; x++) {
            if ((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || x >= 174) continue;
            if (x == b) return 256 + n;
            n++;
        }
    }
    return b;
}

static char *byte_encode(q36_str in, uint64_t *out_len) {
    char *out = xmalloc((size_t)in.len * 4 + 1);
    char *p = out;
    for (uint64_t i = 0; i < in.len; i++) utf8_put(&p, gpt2_byte_to_codepoint((uint8_t)in.ptr[i]));
    *p = '\0';
    *out_len = (uint64_t)(p - out);
    return out;
}

static int utf8_len_from_first_byte(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;
}

typedef struct {
    char *ptr;
    uint64_t len;
} owned_str;

static owned_str owned_copy(const char *ptr, uint64_t len) {
    owned_str s;
    s.ptr = xmalloc((size_t)len);
    memcpy(s.ptr, ptr, (size_t)len);
    s.len = len;
    return s;
}

static int bpe_rank(const q36_vocab *vocab, const owned_str *a, const owned_str *b) {
    uint64_t len = a->len + 1 + b->len;
    char stack[512];
    char *buf = len <= sizeof(stack) ? stack : xmalloc((size_t)len);
    int rank = -1;
    memcpy(buf, a->ptr, (size_t)a->len);
    buf[a->len] = ' ';
    memcpy(buf + a->len + 1, b->ptr, (size_t)b->len);
    table_get(&vocab->merge_rank, buf, len, &rank);
    if (buf != stack) free(buf);
    return rank;
}

static void bpe_emit_piece(const q36_vocab *vocab, q36_str raw_piece, token_vec *out) {
    uint64_t encoded_len = 0;
    char *encoded = byte_encode(raw_piece, &encoded_len);
    int n_sym = 0;
    int cap_sym = 32;
    owned_str *sym = xcalloc((size_t)cap_sym, sizeof(sym[0]));

    for (uint64_t off = 0; off < encoded_len;) {
        int n = utf8_len_from_first_byte((uint8_t)encoded[off]);
        if (off + (uint64_t)n > encoded_len) n = 1;
        if (n_sym == cap_sym) {
            cap_sym *= 2;
            sym = xrealloc(sym, (size_t)cap_sym * sizeof(sym[0]));
        }
        sym[n_sym++] = owned_copy(encoded + off, (uint64_t)n);
        off += (uint64_t)n;
    }

    for (;;) {
        int best_i = -1;
        int best_rank = INT32_MAX;
        for (int i = 0; i + 1 < n_sym; i++) {
            int rank = bpe_rank(vocab, &sym[i], &sym[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_i = i;
            }
        }
        if (best_i < 0) break;
        {
            owned_str merged;
            merged.len = sym[best_i].len + sym[best_i + 1].len;
            merged.ptr = xmalloc((size_t)merged.len);
            memcpy(merged.ptr, sym[best_i].ptr, (size_t)sym[best_i].len);
            memcpy(merged.ptr + sym[best_i].len, sym[best_i + 1].ptr, (size_t)sym[best_i + 1].len);
            free(sym[best_i].ptr);
            free(sym[best_i + 1].ptr);
            sym[best_i] = merged;
            for (int j = best_i + 1; j + 1 < n_sym; j++) sym[j] = sym[j + 1];
            n_sym--;
        }
    }

    for (int i = 0; i < n_sym; i++) {
        int token = -1;
        if (table_get(&vocab->token_to_id, sym[i].ptr, sym[i].len, &token)) {
            q36_tokens_push(out, token);
        } else {
            for (uint64_t j = 0; j < sym[i].len; j++) {
                if (table_get(&vocab->token_to_id, sym[i].ptr + j, 1, &token)) {
                    q36_tokens_push(out, token);
                }
            }
        }
        free(sym[i].ptr);
    }
    free(sym);
    free(encoded);
}

static uint64_t next_utf8_char(const char *s, uint64_t len, uint64_t pos) {
    int n = utf8_len_from_first_byte((uint8_t)s[pos]);
    if (pos + (uint64_t)n > len) n = 1;
    return pos + (uint64_t)n;
}

static bool ascii_alpha(uint8_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static bool ascii_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}

static bool ascii_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

static bool ascii_newline(uint8_t c) {
    return c == '\n' || c == '\r';
}

static bool qwen_letter_like_at(const char *s, uint64_t len, uint64_t pos) {
    (void)len;
    if ((uint8_t)s[pos] < 128) return ascii_alpha((uint8_t)s[pos]);
    return true;
}

static bool qwen_punct_symbol_at(const char *s, uint64_t len, uint64_t pos) {
    uint8_t c;
    (void)len;
    c = (uint8_t)s[pos];
    return c < 128 && !ascii_alpha(c) && !ascii_digit(c) && !ascii_space(c);
}

static bool qwen_match_suffix(const char *s, uint64_t len, uint64_t pos, uint64_t *out_pos) {
    static const char *suffixes[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    if (pos >= len || s[pos] != '\'') return false;
    for (size_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); i++) {
        size_t n = strlen(suffixes[i]);
        if (pos + n > len) continue;
        if (strncasecmp(s + pos, suffixes[i], n) == 0) {
            *out_pos = pos + n;
            return true;
        }
    }
    return false;
}

static uint64_t qwen_consume_letters(const char *s, uint64_t len, uint64_t pos) {
    while (pos < len && qwen_letter_like_at(s, len, pos)) pos = next_utf8_char(s, len, pos);
    return pos;
}

static void bpe_tokenize_text(const q36_vocab *vocab, const char *text, token_vec *out) {
    uint64_t len = strlen(text);
    uint64_t pos = 0;
    while (pos < len) {
        uint64_t start = pos;
        uint8_t c = (uint8_t)text[pos];

        if (c == ' ' && pos + 1 < len) {
            uint64_t next = pos + 1;
            uint8_t nc = (uint8_t)text[next];
            if (ascii_digit(nc)) {
                int ndigits = 0;
                pos++;
                while (pos < len && ascii_digit((uint8_t)text[pos]) && ndigits < 3) {
                    pos++;
                    ndigits++;
                }
            } else if (qwen_letter_like_at(text, len, next) ||
                       (!ascii_newline(nc) && !ascii_digit(nc) && !qwen_letter_like_at(text, len, next) &&
                        !ascii_space(nc) && next + 1 < len && qwen_letter_like_at(text, len, next + 1))) {
                uint64_t after = 0;
                pos++;
                if (!qwen_letter_like_at(text, len, pos)) pos = next_utf8_char(text, len, pos);
                pos = qwen_consume_letters(text, len, pos);
                if (qwen_match_suffix(text, len, pos, &after)) pos = after;
            } else if (qwen_punct_symbol_at(text, len, next)) {
                pos++;
                while (pos < len && qwen_punct_symbol_at(text, len, pos)) pos = next_utf8_char(text, len, pos);
                while (pos < len && ((uint8_t)text[pos] == '/' || ascii_newline((uint8_t)text[pos]))) pos++;
            }
        } else if (ascii_digit(c)) {
            int ndigits = 0;
            while (pos < len && ascii_digit((uint8_t)text[pos]) && ndigits < 3) {
                pos++;
                ndigits++;
            }
        } else if (qwen_letter_like_at(text, len, pos) ||
                   (!ascii_newline(c) && !ascii_digit(c) && !qwen_letter_like_at(text, len, pos) &&
                    !ascii_space(c) && pos + 1 < len && qwen_letter_like_at(text, len, pos + 1))) {
            uint64_t after = 0;
            if (!qwen_letter_like_at(text, len, pos)) pos = next_utf8_char(text, len, pos);
            pos = qwen_consume_letters(text, len, pos);
            if (qwen_match_suffix(text, len, pos, &after)) pos = after;
        } else if ((c == ' ' && pos + 1 < len && qwen_punct_symbol_at(text, len, pos + 1)) || qwen_punct_symbol_at(text, len, pos)) {
            if (c == ' ') pos++;
            while (pos < len && qwen_punct_symbol_at(text, len, pos)) pos = next_utf8_char(text, len, pos);
            while (pos < len && ((uint8_t)text[pos] == '/' || ascii_newline((uint8_t)text[pos]))) pos++;
        } else if (ascii_space(c)) {
            uint64_t p = pos;
            uint64_t last_newline_end = 0;
            while (p < len && ascii_space((uint8_t)text[p])) {
                uint8_t sc = (uint8_t)text[p++];
                if (ascii_newline(sc)) last_newline_end = p;
            }
            if (last_newline_end) {
                pos = last_newline_end;
            } else if (p < len && p > pos + 1 &&
                       (qwen_letter_like_at(text, len, p) || qwen_punct_symbol_at(text, len, p))) {
                pos = p - 1;
            } else {
                pos = p;
            }
        } else {
            pos = next_utf8_char(text, len, pos);
        }

        if (pos == start) pos = next_utf8_char(text, len, pos);
        bpe_emit_piece(vocab, (q36_str){ text + start, pos - start }, out);
    }
}

static int vocab_lookup(const q36_vocab *vocab, const char *text) {
    int token = -1;
    if (!table_get(&vocab->token_to_id, text, strlen(text), &token)) {
        fprintf(stderr, "q36: required tokenizer token is missing: %s\n", text);
        exit(1);
    }
    return token;
}

static void vocab_load(q36_vocab *vocab, const q36_model *model) {
    q36_array_ref tokens;
    q36_array_ref merges;
    q36_cursor c;
    memset(vocab, 0, sizeof(*vocab));
    if (!model_get_array(model, "tokenizer.ggml.tokens", &tokens) || tokens.type != GGUF_VALUE_STRING || tokens.len > INT32_MAX) {
        q36_die("GGUF tokenizer token table is missing or invalid");
    }
    if (!model_get_array(model, "tokenizer.ggml.merges", &merges) || merges.type != GGUF_VALUE_STRING) {
        q36_die("GGUF tokenizer merge table is missing or invalid");
    }
    vocab->n_vocab = (int)tokens.len;
    vocab->token = xcalloc((size_t)vocab->n_vocab, sizeof(vocab->token[0]));
    table_init(&vocab->token_to_id, tokens.len);
    c = cursor_at(model, tokens.data_pos);
    for (int i = 0; i < vocab->n_vocab; i++) {
        if (!cursor_string(&c, &vocab->token[i])) q36_die(c.error);
        table_put(&vocab->token_to_id, vocab->token[i], i);
    }
    table_init(&vocab->merge_rank, merges.len);
    c = cursor_at(model, merges.data_pos);
    for (uint64_t i = 0; i < merges.len; i++) {
        q36_str merge;
        if (!cursor_string(&c, &merge)) q36_die(c.error);
        table_put(&vocab->merge_rank, merge, (int)i);
    }
    vocab->bos_id = (int)required_u32(model, "tokenizer.ggml.bos_token_id");
    vocab->eos_id = (int)required_u32(model, "tokenizer.ggml.eos_token_id");
    vocab->add_bos = model_get_bool(model, "tokenizer.ggml.add_bos_token", &vocab->add_bos) ? vocab->add_bos : false;
    vocab->im_start_id = vocab_lookup(vocab, "<|im_start|>");
    vocab->im_end_id = vocab_lookup(vocab, "<|im_end|>");
    vocab->think_start_id = vocab_lookup(vocab, "<think>");
    vocab->think_end_id = vocab_lookup(vocab, "</think>");
    vocab->vision_start_id = vocab_lookup(vocab, "<|vision_start|>");
    vocab->vision_end_id = vocab_lookup(vocab, "<|vision_end|>");
    vocab->image_pad_id = vocab_lookup(vocab, "<|image_pad|>");
    vocab->video_pad_id = vocab_lookup(vocab, "<|video_pad|>");
}

static void vocab_free(q36_vocab *vocab) {
    free(vocab->token);
    table_free(&vocab->token_to_id);
    table_free(&vocab->merge_rank);
    memset(vocab, 0, sizeof(*vocab));
}

void q36_tokenize_text(q36_engine *e, const char *text, q36_tokens *out) {
    bpe_tokenize_text(&e->vocab, text ? text : "", out);
}

static bool special_token_at(const q36_vocab *vocab, const char *p, int *token, size_t *len) {
    struct special {
        const char *text;
        int token;
    } specials[] = {
        {"<|im_start|>", vocab->im_start_id},
        {"<|im_end|>", vocab->im_end_id},
        {"<think>", vocab->think_start_id},
        {"</think>", vocab->think_end_id},
        {"<|vision_start|>", vocab->vision_start_id},
        {"<|vision_end|>", vocab->vision_end_id},
        {"<|image_pad|>", vocab->image_pad_id},
        {"<|video_pad|>", vocab->video_pad_id},
    };
    for (size_t i = 0; i < sizeof(specials) / sizeof(specials[0]); i++) {
        size_t n = strlen(specials[i].text);
        if (!strncmp(p, specials[i].text, n)) {
            *token = specials[i].token;
            *len = n;
            return true;
        }
    }
    return false;
}

static void tokenize_span(const q36_vocab *vocab, const char *p, size_t n, token_vec *out) {
    char *tmp;
    if (!n) return;
    tmp = xmalloc(n + 1);
    memcpy(tmp, p, n);
    tmp[n] = '\0';
    bpe_tokenize_text(vocab, tmp, out);
    free(tmp);
}

static void tokenize_rendered_chat_vocab(const q36_vocab *vocab, const char *text, token_vec *out) {
    const char *span;
    const char *p;
    if (!text) text = "";
    span = text;
    p = text;
    while (*p) {
        int token = -1;
        size_t len = 0;
        if (special_token_at(vocab, p, &token, &len)) {
            tokenize_span(vocab, span, (size_t)(p - span), out);
            q36_tokens_push(out, token);
            p += len;
            span = p;
            continue;
        }
        p++;
    }
    tokenize_span(vocab, span, (size_t)(p - span), out);
}

void q36_tokenize_rendered_chat(q36_engine *e, const char *text, q36_tokens *out) {
    tokenize_rendered_chat_vocab(&e->vocab, text, out);
}

void q36_chat_begin(q36_engine *e, q36_tokens *tokens) {
    if (e->vocab.bos_id >= 0) q36_tokens_push(tokens, e->vocab.bos_id);
}

static void chat_append_open_role(q36_engine *e, token_vec *tokens, const char *role) {
    q36_tokenize_rendered_chat(e, "<|im_start|>", tokens);
    bpe_tokenize_text(&e->vocab, role, tokens);
    bpe_tokenize_text(&e->vocab, "\n", tokens);
}

static void chat_append_close_role(q36_engine *e, token_vec *tokens) {
    q36_tokenize_rendered_chat(e, "<|im_end|>", tokens);
    bpe_tokenize_text(&e->vocab, "\n", tokens);
}

void q36_chat_append_max_effort_prefix(q36_engine *e, q36_tokens *tokens) {
    bpe_tokenize_text(&e->vocab, Q36_REASONING_EFFORT_MAX_PREFIX, tokens);
}

void q36_chat_append_message(q36_engine *e, q36_tokens *tokens, const char *role, const char *content) {
    if (!role) role = "user";
    if (!content) content = "";
    if (!strcmp(role, "developer")) role = "system";
    if (!strcmp(role, "tool") || !strcmp(role, "function")) {
        chat_append_open_role(e, tokens, "user");
        bpe_tokenize_text(&e->vocab, "<tool_response>\n", tokens);
        bpe_tokenize_text(&e->vocab, content, tokens);
        bpe_tokenize_text(&e->vocab, "\n</tool_response>", tokens);
        chat_append_close_role(e, tokens);
        return;
    }
    chat_append_open_role(e, tokens, role);
    if (!strcmp(role, "assistant")) {
        q36_tokenize_rendered_chat(e, content, tokens);
    } else {
        bpe_tokenize_text(&e->vocab, content, tokens);
    }
    chat_append_close_role(e, tokens);
}

void q36_chat_append_assistant_prefix(q36_engine *e, q36_tokens *tokens, q36_think_mode think_mode) {
    chat_append_open_role(e, tokens, "assistant");
    if (think_mode == Q36_THINK_NONE) {
        q36_tokenize_rendered_chat(e, "</think>", tokens);
        return;
    }
    q36_tokenize_rendered_chat(e, "<think>", tokens);
    if (think_mode == Q36_THINK_MAX) {
        bpe_tokenize_text(&e->vocab, "\n", tokens);
        q36_chat_append_max_effort_prefix(e, tokens);
    }
}

void q36_encode_chat_prompt(q36_engine *e, const char *system, const char *prompt, q36_think_mode think_mode, q36_tokens *out) {
    q36_chat_begin(e, out);
    if (system && system[0]) q36_chat_append_message(e, out, "system", system);
    q36_chat_append_message(e, out, "user", prompt ? prompt : "");
    q36_chat_append_assistant_prefix(e, out, think_mode);
}

static void dump_tokens_fp(FILE *fp, const q36_vocab *vocab, const token_vec *tokens) {
    fprintf(fp, "[");
    for (int i = 0; i < tokens->len; i++) {
        if (i) fprintf(fp, ", ");
        fprintf(fp, "%d", tokens->v[i]);
    }
    fprintf(fp, "]\n");
    for (int i = 0; i < tokens->len; i++) {
        int id = tokens->v[i];
        if (id >= 0 && id < vocab->n_vocab) {
            fprintf(fp, "%6d  %.*s\n", id, (int)vocab->token[id].len, vocab->token[id].ptr);
        }
    }
}

static uint32_t utf8_decode_one(const char *s, uint64_t len, uint64_t *pos) {
    uint8_t c = (uint8_t)s[*pos];
    if (c < 0x80 || *pos + 1 >= len) {
        (*pos)++;
        return c;
    }
    if ((c & 0xe0) == 0xc0 && *pos + 1 < len) {
        uint32_t cp = ((uint32_t)(c & 0x1f) << 6) | ((uint8_t)s[*pos + 1] & 0x3f);
        *pos += 2;
        return cp;
    }
    if ((c & 0xf0) == 0xe0 && *pos + 2 < len) {
        uint32_t cp = ((uint32_t)(c & 0x0f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 2] & 0x3f);
        *pos += 3;
        return cp;
    }
    if ((c & 0xf8) == 0xf0 && *pos + 3 < len) {
        uint32_t cp = ((uint32_t)(c & 0x07) << 18) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 2] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 3] & 0x3f);
        *pos += 4;
        return cp;
    }
    (*pos)++;
    return c;
}

static int gpt2_codepoint_to_byte(uint32_t cp) {
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) return (int)cp;
    {
        uint32_t n = 0;
        for (uint32_t b = 0; b < 256; b++) {
            if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174) continue;
            if (cp == 256 + n) return (int)b;
            n++;
        }
    }
    return -1;
}

static bool vocab_token_is_literal_special(q36_str s) {
    return (s.len >= 2 && s.ptr[0] == '<' && s.ptr[1] == '|') ||
           q36_streq(s, "<think>") ||
           q36_streq(s, "</think>");
}

char *q36_token_text(q36_engine *e, int token, size_t *len) {
    q36_str s;
    char *out;
    size_t n = 0;
    uint64_t pos = 0;
    if (!e || token < 0 || token >= e->vocab.n_vocab) {
        out = xmalloc(1);
        out[0] = '\0';
        if (len) *len = 0;
        return out;
    }
    s = e->vocab.token[token];
    out = xmalloc((size_t)s.len + 1);
    if (vocab_token_is_literal_special(s)) {
        memcpy(out, s.ptr, (size_t)s.len);
        out[s.len] = '\0';
        if (len) *len = (size_t)s.len;
        return out;
    }
    while (pos < s.len) {
        uint32_t cp = utf8_decode_one(s.ptr, s.len, &pos);
        int b = gpt2_codepoint_to_byte(cp);
        if (b >= 0) out[n++] = (char)b;
    }
    out[n] = '\0';
    if (len) *len = n;
    return out;
}

int q36_token_eos(q36_engine *e) {
    return e ? e->vocab.eos_id : -1;
}

static int sample_argmax(const float *logits, uint32_t n_vocab) {
    int best = 0;
    float best_v = Q36_NEG_INF;
    for (uint32_t i = 0; i < n_vocab; i++) {
        float v = logits[i];
        if (v > best_v) {
            best_v = v;
            best = (int)i;
        }
    }
    return best;
}

#ifndef Q36_NO_GPU
static int sample_argmax_margin(const float *logits, uint32_t n_vocab, float *margin) {
    int best = 0;
    float best_v = Q36_NEG_INF;
    float second_v = Q36_NEG_INF;
    for (uint32_t i = 0; i < n_vocab; i++) {
        float v = logits[i];
        if (v > best_v) {
            second_v = best_v;
            best_v = v;
            best = (int)i;
        } else if (v > second_v) {
            second_v = v;
        }
    }
    if (margin) *margin = best_v - second_v;
    return best;
}
#endif

typedef struct {
    int id;
    float logit;
    float prob;
} q36_sample_candidate;

static uint64_t q36_sample_rng_next(uint64_t *state) {
    uint64_t x = *state;
    if (x == 0) x = 0x9e3779b97f4a7c15ull;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545f4914f6cdd1dull;
}

static float q36_sample_rng_f32(uint64_t *state) {
    const uint64_t x = q36_sample_rng_next(state);
    return (float)((x >> 40) & 0xffffffu) / 16777216.0f;
}

static int q36_sample_candidate_cmp_desc(const void *a, const void *b) {
    const q36_sample_candidate *ca = (const q36_sample_candidate *)a;
    const q36_sample_candidate *cb = (const q36_sample_candidate *)b;
    return (cb->logit > ca->logit) - (cb->logit < ca->logit);
}

static int q36_sample_full_vocab(const float *logits, uint32_t n_vocab,
                                 float temperature, float top_p, float min_p,
                                 uint64_t *rng) {
    float max_logit = Q36_NEG_INF;
    int best = 0;
    uint32_t finite = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        finite++;
        if (v > max_logit) {
            max_logit = v;
            best = (int)i;
        }
    }
    if (finite == 0) return sample_argmax(logits, n_vocab);

    if (top_p >= 1.0f) {
        float sum = 0.0f;
        const float min_rel = min_p > 0.0f ? min_p : 0.0f;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            {
                const float p = expf((v - max_logit) / temperature);
                if (p < min_rel) continue;
                sum += p;
            }
        }
        if (sum <= 0.0f || !isfinite(sum)) return best;
        {
            float r = q36_sample_rng_f32(rng) * sum;
            for (uint32_t i = 0; i < n_vocab; i++) {
                const float v = logits[i];
                if (!isfinite(v)) continue;
                {
                    const float p = expf((v - max_logit) / temperature);
                    if (p < min_rel) continue;
                    r -= p;
                    if (r <= 0.0f) return (int)i;
                }
            }
        }
        return best;
    }

    {
        q36_sample_candidate *cand = xmalloc((size_t)finite * sizeof(cand[0]));
        uint32_t n = 0;
        float sum = 0.0f;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            cand[n++] = (q36_sample_candidate){.id = (int)i, .logit = v, .prob = expf((v - max_logit) / temperature)};
            sum += cand[n - 1].prob;
        }
        if (sum <= 0.0f || !isfinite(sum)) {
            free(cand);
            return best;
        }

        qsort(cand, (size_t)n, sizeof(cand[0]), q36_sample_candidate_cmp_desc);
        {
            const float min_prob = (cand[0].prob / sum) * (min_p > 0.0f ? min_p : 0.0f);
            float filtered_sum = 0.0f;
            uint32_t filtered = 0;
            for (uint32_t i = 0; i < n; i++) {
                const float p = cand[i].prob / sum;
                if (i > 0 && p < min_prob) break;
                filtered_sum += cand[i].prob;
                filtered++;
                if (filtered_sum / sum >= top_p) break;
            }
            if (filtered == 0) {
                free(cand);
                return best;
            }
            {
                float r = q36_sample_rng_f32(rng) * filtered_sum;
                for (uint32_t i = 0; i < filtered; i++) {
                    r -= cand[i].prob;
                    if (r <= 0.0f) {
                        const int id = cand[i].id;
                        free(cand);
                        return id;
                    }
                }
                {
                    const int id = cand[filtered - 1].id;
                    free(cand);
                    return id;
                }
            }
        }
    }
}

static int q36_sample_top_p_min_p(const float *logits, uint32_t n_vocab,
                                  float temperature, int top_k,
                                  float top_p, float min_p,
                                  uint64_t *rng) {
    if (temperature <= 0.0f) return sample_argmax(logits, n_vocab);
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (min_p < 0.0f) min_p = 0.0f;
    if (top_k <= 0) return q36_sample_full_vocab(logits, n_vocab, temperature, top_p, min_p, rng);
    if (top_k > 1024) top_k = 1024;
    if ((uint32_t)top_k > n_vocab) top_k = (int)n_vocab;

    {
        int ids[1024];
        float vals[1024];
        float probs[1024];
        int n = 0;
        for (uint32_t i = 0; i < n_vocab; i++) {
            float v = logits[i];
            if (!isfinite(v)) continue;
            if (n == top_k && v <= vals[n - 1]) continue;
            {
                int j = n < top_k ? n++ : n - 1;
                while (j > 0 && vals[j - 1] < v) {
                    vals[j] = vals[j - 1];
                    ids[j] = ids[j - 1];
                    j--;
                }
                vals[j] = v;
                ids[j] = (int)i;
            }
        }
        if (n == 0) return sample_argmax(logits, n_vocab);

        {
            const float max_logit = vals[0];
            float sum = 0.0f;
            for (int i = 0; i < n; i++) {
                probs[i] = expf((vals[i] - max_logit) / temperature);
                sum += probs[i];
            }
            if (sum <= 0.0f || !isfinite(sum)) return ids[0];

            {
                const float min_prob = (probs[0] / sum) * min_p;
                float filtered_sum = 0.0f;
                int filtered = 0;
                for (int i = 0; i < n; i++) {
                    float p = probs[i] / sum;
                    if (i > 0 && p < min_prob) break;
                    filtered_sum += probs[i];
                    filtered++;
                    if (filtered_sum / sum >= top_p) break;
                }
                if (filtered <= 0) return ids[0];

                {
                    float r = q36_sample_rng_f32(rng) * filtered_sum;
                    for (int i = 0; i < filtered; i++) {
                        r -= probs[i];
                        if (r <= 0.0f) return ids[i];
                    }
                }
                return ids[filtered - 1];
            }
        }
    }
}

static bool q36_load_directional_steering(q36_engine *e) {
    uint64_t count;
    uint64_t bytes;
    struct stat st;
    FILE *fp;
    if (!e || (e->directional_steering_attn_scale == 0.0f &&
               e->directional_steering_ffn_scale == 0.0f)) {
        return true;
    }
    if (!e->directional_steering_file || !e->directional_steering_file[0]) {
        fprintf(stderr, "q36: directional steering needs --dir-steering-file\n");
        return false;
    }
    count = (uint64_t)Q36_N_LAYER * Q36_N_EMBD;
    bytes = count * sizeof(float);
    fp = fopen(e->directional_steering_file, "rb");
    if (!fp) {
        fprintf(stderr, "q36: failed to open directional steering file %s: %s\n",
                e->directional_steering_file, strerror(errno));
        return false;
    }
    if (fstat(fileno(fp), &st) != 0 || st.st_size < 0 || (uint64_t)st.st_size != bytes) {
        fprintf(stderr,
                "q36: directional steering file must contain exactly %u x %u f32 values (%" PRIu64 " bytes)\n",
                Q36_N_LAYER, Q36_N_EMBD, bytes);
        fclose(fp);
        return false;
    }
    e->directional_steering_dirs = xmalloc((size_t)bytes);
    if (fread(e->directional_steering_dirs, sizeof(float), (size_t)count, fp) != count ||
        fclose(fp) != 0) {
        fprintf(stderr, "q36: failed to read directional steering vectors from %s\n",
                e->directional_steering_file);
        free(e->directional_steering_dirs);
        e->directional_steering_dirs = NULL;
        return false;
    }
#ifndef Q36_NO_GPU
    if (e->backend == Q36_BACKEND_VULKAN) {
        e->directional_steering_gpu = q36_gpu_tensor_alloc(bytes);
        if (!e->directional_steering_gpu ||
            !q36_gpu_tensor_write(e->directional_steering_gpu, 0,
                                  e->directional_steering_dirs, bytes)) {
            fprintf(stderr, "q36: failed to upload directional steering vectors\n");
            return false;
        }
        free(e->directional_steering_dirs);
        e->directional_steering_dirs = NULL;
    }
#endif
    fprintf(stderr, "q36: directional steering enabled: %s attn=%g ffn=%g\n",
            e->directional_steering_file,
            (double)e->directional_steering_attn_scale,
            (double)e->directional_steering_ffn_scale);
    return true;
}

int q36_engine_open(q36_engine **out, const q36_engine_options *opt) {
    q36_engine *e;
    if (!out || !opt || !opt->model_path || !opt->model_path[0]) return 1;
#ifdef Q36_NO_GPU
    if (opt->backend == Q36_BACKEND_VULKAN) return 1;
#else
    if (opt->backend == Q36_BACKEND_VULKAN && !q36_gpu_init()) return 1;
#endif
    e = xcalloc(1, sizeof(*e));
    e->model.fd = -1;
    e->mtp_model.fd = -1;
    e->backend = opt->backend;
    e->n_threads = q36_resolve_thread_count(opt->n_threads);
    e->cpu_prefill_cap = opt->prefill_chunk ? opt->prefill_chunk : q36_default_cpu_prefill_cap();
    e->prefill_cap_override = opt->prefill_chunk;
    e->cache_type_k = opt->cache_type_k;
    e->cache_type_v = opt->cache_type_v;
    e->power_percent = opt->power_percent > 0 ? opt->power_percent : 100;
    e->quality = opt->quality;
    e->ssd_streaming = opt->ssd_streaming;
    e->ssd_streaming_cold = opt->ssd_streaming_cold;
    e->ssd_streaming_full_layers_set = opt->ssd_streaming_full_layers_set;
    e->ssd_streaming_cache_experts = opt->ssd_streaming_cache_experts;
    e->ssd_streaming_full_layers = opt->ssd_streaming_full_layers;
    e->ssd_streaming_cache_bytes = opt->ssd_streaming_cache_bytes;
    e->ssd_streaming_preload_experts = opt->ssd_streaming_preload_experts;
    e->mtp_draft_tokens = opt->mtp_draft_tokens > 0 ? opt->mtp_draft_tokens : 1;
    if (e->mtp_draft_tokens > Q36_MTP_MAX_DRAFT) e->mtp_draft_tokens = Q36_MTP_MAX_DRAFT;
    e->mtp_margin = opt->mtp_margin >= 0.0f ? opt->mtp_margin : 0.0f;
    if (opt->directional_steering_file && opt->directional_steering_file[0]) {
        e->directional_steering_file = q36_strdup(opt->directional_steering_file);
    }
    e->directional_steering_attn_scale = opt->directional_steering_attn;
    e->directional_steering_ffn_scale = opt->directional_steering_ffn;
    if (!q36_load_directional_steering(e)) {
        q36_engine_close(e);
        return 1;
    }
    if (opt->simulate_used_memory_bytes != 0 &&
        !q36_ssd_memory_lock_acquire(&e->simulated_memory,
                                     opt->simulate_used_memory_bytes)) {
        q36_engine_close(e);
        return 1;
    }
    model_open(&e->model, opt->model_path, q36_backend_uses_graph(opt->backend));
    config_validate_model(&e->model);
    vocab_load(&e->vocab, &e->model);
    weights_bind(&e->weights, &e->model);
    if (e->ssd_streaming && !q36_backend_supports_ssd_streaming(e->backend)) {
        fprintf(stderr, "q36: --ssd-streaming is currently supported only with --vulkan\n");
        q36_engine_close(e);
        return 1;
    }
    if (e->ssd_streaming && e->ssd_streaming_cache_bytes != 0) {
        uint64_t requested_cache_bytes = e->ssd_streaming_cache_bytes;
        uint64_t safe_cache_bytes = q36_streaming_manual_cache_safe_bytes();
        if (safe_cache_bytes != 0 && e->ssd_streaming_cache_bytes > safe_cache_bytes) {
            e->ssd_streaming_cache_bytes = safe_cache_bytes;
            fprintf(stderr,
                    "q36: %s SSD streaming cache budget %.2f GiB capped to %.2f GiB to keep expert buffers lockable\n",
                    q36_backend_name(e->backend),
                    (double)requested_cache_bytes / 1073741824.0,
                    (double)e->ssd_streaming_cache_bytes / 1073741824.0);
        }
        uint64_t per_expert_bytes = 0;
        uint32_t budget = q36_streaming_cache_experts_for_byte_budget(&e->weights,
                                                                      e->ssd_streaming_cache_bytes,
                                                                      &per_expert_bytes);
        if (budget == 0 || per_expert_bytes == 0) {
            fprintf(stderr,
                    "q36: --ssd-streaming-cache-experts byte budget is too small or invalid for this model\n");
            q36_engine_close(e);
            return 1;
        }
        e->ssd_streaming_cache_experts = budget;
        fprintf(stderr,
                "q36: %s SSD streaming cache budget %.2f GiB / %.2f MiB per expert = %u experts\n",
                q36_backend_name(e->backend),
                (double)e->ssd_streaming_cache_bytes / 1073741824.0,
                (double)per_expert_bytes / 1048576.0,
                budget);
    }
    e->routed_quant_bits = q36_quant_bits_from_type(e->weights.layer[0].ffn_gate_exps->type);
    e->expert_weights_scale = 1.0f;
    model_get_f32_compat(&e->model, "qwen35moe.expert_weights_scale", &e->expert_weights_scale);
    e->mtp_expert_weights_scale = e->expert_weights_scale;
    if (opt->mtp_path && opt->mtp_path[0]) {
        if (opt->backend != Q36_BACKEND_VULKAN) {
            fprintf(stderr, "q36: --mtp is currently supported only with --vulkan\n");
            q36_engine_close(e);
            return 1;
        }
#ifdef Q36_NO_GPU
        fprintf(stderr, "q36: --mtp is not available in CPU-only builds\n");
        q36_engine_close(e);
        return 1;
#else
        model_open(&e->mtp_model, opt->mtp_path, true);
        mtp_weights_bind(&e->mtp_weights, &e->mtp_model);
        model_get_f32_compat(&e->mtp_model, "qwen35moe.expert_weights_scale", &e->mtp_expert_weights_scale);
        e->mtp_ready = true;
        fprintf(stderr, "q36: MTP support model loaded: %s (draft=%d)\n",
                opt->mtp_path,
                e->mtp_draft_tokens);
#endif
    }
#ifndef Q36_NO_GPU
    if (opt->backend == Q36_BACKEND_VULKAN) {
        q36_gpu_set_quality(opt->quality);
        q36_gpu_set_ssd_streaming(e->ssd_streaming);
        if (!q36_engine_configure_streaming_auto_cache(e)) {
            q36_engine_close(e);
            return 1;
        }
        if (e->ssd_streaming_full_layers > Q36_N_LAYER) {
            fprintf(stderr, "q36: --ssd-streaming-full-layers must be between 0 and %u\n",
                    Q36_N_LAYER);
            q36_engine_close(e);
            return 1;
        }
        if (e->ssd_streaming && e->ssd_streaming_full_layers != 0) {
            uint32_t reserve = 0;
            uint64_t resident_bytes = 0;
            if (!q36_streaming_full_layer_budget(&e->weights,
                                                  e->ssd_streaming_full_layers,
                                                  &reserve,
                                                  &resident_bytes)) {
                fprintf(stderr, "q36: SSD streaming full-layer byte accounting failed\n");
                q36_engine_close(e);
                return 1;
            }
            uint64_t required = (uint64_t)reserve + Q36_N_EXPERT;
            if (required > e->ssd_streaming_cache_experts) {
                fprintf(stderr,
                        "q36: %u full resident layers require %u expert slots "
                        "plus %u for dynamic prefill, but the streaming budget has %u\n",
                        e->ssd_streaming_full_layers, reserve,
                        Q36_N_EXPERT,
                        e->ssd_streaming_cache_experts);
                q36_engine_close(e);
                return 1;
            }
            e->ssd_streaming_cache_experts -= (uint32_t)reserve;
            fprintf(stderr,
                    "q36: SSD streaming keeps %u full routed layers resident "
                    "(%.2f GiB); dynamic cache %u experts\n",
                    e->ssd_streaming_full_layers,
                    (double)resident_bytes / 1073741824.0,
                    e->ssd_streaming_cache_experts);
        } else if (e->ssd_streaming && e->ssd_streaming_full_layers_set) {
            fprintf(stderr, "q36: SSD streaming full resident layers disabled\n");
        }
        q36_gpu_set_streaming_full_layers(e->ssd_streaming ?
                                          e->ssd_streaming_full_layers : 0);
        q36_gpu_set_streaming_expert_cache_budget(e->ssd_streaming_cache_experts);
        if (e->ssd_streaming) {
            uint64_t slab_expert_bytes = 0;
            if (q36_streaming_routed_expert_bytes(&e->weights, &slab_expert_bytes)) {
                q36_gpu_set_streaming_expert_cache_expert_bytes(slab_expert_bytes);
                uint32_t routed = 0, boosted = 0;
                for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
                    const q36_layer_weights *l = &e->weights.layer[il];
                    if (!l->ffn_gate_exps || !l->ffn_up_exps || !l->ffn_down_exps) continue;
                    routed++;
                    if (!q36_weights_streaming_layer_experts_uniform(&e->weights, il)) boosted++;
                }
                if (boosted > 0) {
                    fprintf(stderr,
                            "q36: SSD streaming mixed-precision model: %u/%u routed layers off the slab size class will bypass the expert cache\n",
                            boosted,
                            routed);
                }
                if (boosted * 2 > routed && routed != 0) {
                    fprintf(stderr,
                            "q36: WARNING: most routed layers (%u/%u) are off the SSD streaming slab size class; expert-cache hit rate will be poor\n",
                            boosted,
                            routed);
                }
                uint32_t cache_floor = 2u * (routed - boosted) * Q36_N_EXPERT_USED;
                if (e->ssd_streaming_cache_experts < cache_floor) {
                    fprintf(stderr,
                            "q36: WARNING: SSD streaming cache has %u experts; %u are recommended to retain two decode routes per cacheable layer\n",
                            e->ssd_streaming_cache_experts, cache_floor);
                }
            }
        }
        if (!q36_gpu_set_model_map(e->model.map, e->model.size) ||
            !q36_gpu_set_model_fd(e->model.fd)) {
            q36_engine_close(e);
            return 1;
        }
        q36_vulkan_prewarm_weights(e);
        if (!q36_engine_seed_streaming_expert_cache(e)) {
            q36_engine_close(e);
            return 1;
        }
    }
#endif
    *out = e;
    return 0;
}

void q36_engine_close(q36_engine *e) {
    if (!e) return;
#ifndef Q36_NO_GPU
    q36_gpu_tensor_free(e->directional_steering_gpu);
    q36_gpu_tensor_free(e->session_batch_logits);
    if (e->backend == Q36_BACKEND_VULKAN) q36_gpu_cleanup();
#endif
    free(e->directional_steering_dirs);
    vocab_free(&e->vocab);
    model_close(&e->mtp_model);
    model_close(&e->model);
    free(e->directional_steering_file);
    q36_ssd_memory_lock_release(&e->simulated_memory);
    free(e);
}

void q36_engine_summary(q36_engine *e) {
    if (!e) return;
    model_summary(&e->model);
}

int q36_engine_vocab_size(q36_engine *e) {
    return e ? e->vocab.n_vocab : 0;
}

int q36_engine_power(q36_engine *e) {
    return e && e->power_percent > 0 ? e->power_percent : 100;
}

int q36_engine_set_power(q36_engine *e, int power_percent) {
    if (!e || power_percent < 1 || power_percent > 100) return 1;
    e->power_percent = power_percent;
    return 0;
}

const char *q36_engine_model_name(q36_engine *e) {
    (void)e;
    return "qwen3.6-35b-a3b";
}

int q36_engine_model_id(q36_engine *e) {
    (void)e;
    return 1;
}

int q36_inspect_model(const char *model_path) {
    q36_engine *e = NULL;
    q36_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model_path;
    opt.backend = Q36_BACKEND_CPU;
    if (q36_engine_open(&e, &opt) != 0) return 1;
    q36_engine_summary(e);
    q36_engine_close(e);
    return 0;
}

void q36_engine_dump_tokens(q36_engine *e, const q36_tokens *tokens) {
    if (!e || !tokens) return;
    dump_tokens_fp(stdout, &e->vocab, tokens);
}

int q36_dump_text_tokenization(const char *model_path, const char *text, FILE *fp) {
    q36_engine *e = NULL;
    q36_engine_options opt;
    q36_tokens tokens = {0};
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model_path;
    opt.backend = Q36_BACKEND_CPU;
    if (!fp) fp = stdout;
    if (q36_engine_open(&e, &opt) != 0) return 1;
    q36_tokenize_text(e, text ? text : "", &tokens);
    dump_tokens_fp(fp, &e->vocab, &tokens);
    q36_tokens_free(&tokens);
    q36_engine_close(e);
    return 0;
}

static int q36_unsupported_runtime(char *err, size_t errlen) {
    if (err && errlen) snprintf(err, errlen, "Qwen inference/session path is not wired on the new baseline yet");
    return 1;
}

int q36_engine_generate_argmax(q36_engine *e, const q36_tokens *prompt,
                               int n_predict, int ctx_size,
                               q36_token_emit_fn emit,
                               q36_generation_done_fn done,
                               void *emit_ud,
                               q36_session_progress_fn progress,
                               void *progress_ud) {
    q36_session *s = NULL;
    char err[160] = {0};
    int eos;
    int generated = 0;
    double t_prefill0;
    double t_prefill1;
    double t_decode0;
    double t_decode1;
    int rc = 1;
    if (!e || !prompt || n_predict < 0 || ctx_size <= 0) return 1;
    if (q36_session_create(&s, e, ctx_size) != 0 || !s) return 1;
    q36_session_set_progress(s, progress, progress_ud);
    t_prefill0 = q36_now_sec();
    if (q36_session_sync(s, prompt, err, sizeof(err)) != 0) goto done;
    t_prefill1 = q36_now_sec();
    eos = q36_token_eos(e);
    t_decode0 = q36_now_sec();
    for (int i = 0; i < n_predict; i++) {
        int token = q36_session_argmax(s);
        if (token < 0) goto done;
        token = q36_session_eos_to_think_close(s, token);
        if (token == eos) break;
        if (emit) emit(emit_ud, token);
        if (q36_session_eval(s, token, err, sizeof(err)) != 0) goto done;
        generated++;
    }
    t_decode1 = q36_now_sec();
    if (done) done(emit_ud);
    q36_log(stderr,
            Q36_LOG_TIMING,
            "q36: prefill: %.2f t/s, generation: %.2f t/s\n",
            t_prefill1 > t_prefill0 ? (double)prompt->len / (t_prefill1 - t_prefill0) : 0.0,
            t_decode1 > t_decode0 ? (double)generated / (t_decode1 - t_decode0) : 0.0);
    rc = 0;
done:
    q36_session_free(s);
    return rc;
}

int q36_engine_head_test(q36_engine *e, const q36_tokens *prompt) {
    (void)e;
    (void)prompt;
    return 1;
}

int q36_engine_first_token_test(q36_engine *e, const q36_tokens *prompt) {
    (void)e;
    (void)prompt;
    return 1;
}

int q36_engine_vulkan_graph_test(q36_engine *e, const q36_tokens *prompt) {
    (void)e;
    (void)prompt;
    return 1;
}

int q36_engine_vulkan_graph_full_test(q36_engine *e, const q36_tokens *prompt) {
    (void)e;
    (void)prompt;
    return 1;
}

int q36_engine_vulkan_graph_prompt_test(q36_engine *e, const q36_tokens *prompt, int ctx_size) {
    (void)e;
    (void)prompt;
    (void)ctx_size;
    return 1;
}

int q36_engine_debug_tensor_row(q36_engine *e, const char *tensor_name, uint64_t row, float *dst, uint32_t n) {
    q36_tensor *t;
    if (!e || !tensor_name || !dst) return 1;
    t = model_find_tensor(&e->model, tensor_name);
    if (!t) return 1;
    return q36_tensor_row_to_float(&e->model, t, row, dst, n) ? 0 : 1;
}

int q36_engine_debug_first_tensor_of_type(q36_engine *e, uint32_t type, char *name, size_t name_cap, uint32_t *n) {
    if (!e || !name || name_cap == 0) return 1;
    for (uint64_t i = 0; i < e->model.n_tensors; i++) {
        q36_tensor *t = &e->model.tensors[i];
        if (t->type != type || t->ndim == 0 || t->dim[0] == 0) continue;
        if (t->name.len + 1 > name_cap) return 1;
        memcpy(name, t->name.ptr, t->name.len);
        name[t->name.len] = '\0';
        if (n) *n = (uint32_t)t->dim[0];
        return 0;
    }
    return 1;
}

int q36_engine_debug_tensor_row_packed(q36_engine *e, const char *tensor_name, uint64_t row,
                                       void *dst, uint64_t dst_cap,
                                       uint64_t *row_bytes, uint32_t *type, uint32_t *n) {
    q36_tensor *t;
    uint64_t got_row_bytes;
    uint64_t row_count;
    const uint8_t *src;
    if (!e || !tensor_name) return 1;
    t = model_find_tensor(&e->model, tensor_name);
    if (!t || t->ndim == 0) return 1;
    row_count = t->elements / t->dim[0];
    if (row >= row_count || !tensor_nbytes(t->type, t->dim[0], &got_row_bytes)) return 1;
    if (row_bytes) *row_bytes = got_row_bytes;
    if (type) *type = t->type;
    if (n) *n = (uint32_t)t->dim[0];
    if (!dst) return 0;
    if (dst_cap < got_row_bytes) return 1;
    src = e->model.map + t->abs_offset + row * got_row_bytes;
    memcpy(dst, src, (size_t)got_row_bytes);
    return 0;
}

int q36_session_create(q36_session **out, q36_engine *e, int ctx_size) {
    q36_session *s;
    if (!out || !e || ctx_size <= 0) return 1;
    s = xcalloc(1, sizeof(*s));
    s->engine = e;
    s->ctx_size = ctx_size;
    s->logits = xmalloc((size_t)Q36_N_VOCAB * sizeof(*s->logits));
    memset(s->logits, 0, (size_t)Q36_N_VOCAB * sizeof(*s->logits));
    if (q36_engine_uses_vulkan_runtime(e)) {
#ifdef Q36_NO_GPU
        s->runtime = NULL;
#else
        s->runtime = q36_vulkan_runtime_create(ctx_size,
                                               q36_engine_vk_prefill_cap(e),
                                               e->mtp_ready,
                                               e->quality,
                                               e->ssd_streaming,
                                               e->cache_type_k,
                                               e->cache_type_v);
#endif
    } else {
        s->runtime = q36_cpu_runtime_create(ctx_size,
                                            e->cpu_prefill_cap,
                                            e->cache_type_k,
                                            e->cache_type_v);
    }
    if (!s->runtime) {
        free(s->logits);
        free(s->mtp_logits);
        free(s);
        return 1;
    }
    if (e->mtp_ready) {
        s->mtp_logits = xmalloc((size_t)Q36_N_VOCAB * sizeof(*s->mtp_logits));
        memset(s->mtp_logits, 0, (size_t)Q36_N_VOCAB * sizeof(*s->mtp_logits));
        s->mtp_draft_token = -1;
    }
    q36_session_reset_runtime(s);
    *out = s;
    return 0;
}

void q36_session_free(q36_session *s) {
    if (!s) return;
    q36_tokens_free(&s->checkpoint);
    if (q36_engine_uses_vulkan_runtime(s->engine)) {
#ifndef Q36_NO_GPU
        q36_vulkan_runtime_free((q36_vulkan_runtime *)s->runtime);
#endif
    } else {
        q36_cpu_runtime_free((q36_cpu_runtime *)s->runtime);
    }
    free(s->logits);
    free(s->mtp_logits);
    free(s);
}

void q36_session_set_progress(q36_session *s, q36_session_progress_fn fn, void *ud) {
    if (!s) return;
    s->progress = fn;
    s->progress_ud = ud;
}

int q36_session_power(q36_session *s) {
    return s ? q36_engine_power(s->engine) : 100;
}

int q36_session_set_power(q36_session *s, int power_percent) {
    return s ? q36_engine_set_power(s->engine, power_percent) : 1;
}

void q36_session_set_display_progress(q36_session *s, q36_session_progress_fn fn, void *ud) {
    if (!s) return;
    s->display_progress = fn;
    s->display_progress_ud = ud;
}

void q36_session_set_cancel(q36_session *s, q36_session_cancel_fn fn, void *ud) {
    if (!s) return;
    s->cancel = fn;
    s->cancel_ud = ud;
}

/* Evaluate prompt tokens [start, len) on top of the current checkpoint
 * (start must equal s->checkpoint.len), leaving the last token's logits in
 * s->logits.  The Vulkan runtime walks the range in prefill-sized batches;
 * the CPU path uses smaller layer-major chunks that keep recurrent state
 * updates ordered while batching projections and FFN work. */
static int q36_session_prefill_range(q36_session *s, const q36_tokens *prompt, int start,
                                     char *err, size_t errlen) {
    int i = start;
#ifndef Q36_NO_GPU
    if (start == 0 && q36_engine_uses_vulkan_runtime(s->engine)) {
        q36_gpu_stream_expert_cache_reset_route_hotness();
    }
#endif
    while (i < prompt->len) {
        if (s->cancel && s->cancel(s->cancel_ud)) {
            if (err && errlen) snprintf(err, errlen, "interrupted");
            return Q36_SESSION_SYNC_INTERRUPTED;
        }
        int n = 1;
        bool compute_logits;
        bool ok;
#ifndef Q36_NO_GPU
        if (q36_engine_uses_vulkan_runtime(s->engine)) {
            q36_vulkan_runtime *rt = (q36_vulkan_runtime *)s->runtime;
            n = prompt->len - i;
            if (rt && n > (int)rt->prefill_cap) n = (int)rt->prefill_cap;
            compute_logits = (i + n) == prompt->len;
            ok = q36_forward_tokens_vulkan(s, prompt->v + i, (uint32_t)n, (uint32_t)i, compute_logits);
        } else
#endif
        {
            q36_cpu_runtime *rt = (q36_cpu_runtime *)s->runtime;
            n = prompt->len - i;
            if (rt && n > (int)rt->prefill_cap) n = (int)rt->prefill_cap;
            compute_logits = (i + n) == prompt->len;
            ok = !q36_engine_uses_vulkan_runtime(s->engine) &&
                 (n == 1 && rt && rt->prefill_cap == 1
                  ? q36_forward_token_cpu(s, prompt->v[i], (uint32_t)i, compute_logits)
                  : q36_forward_tokens_cpu(s, prompt->v + i, (uint32_t)n, (uint32_t)i, compute_logits));
        }
        if (!ok) {
            if (err && errlen) snprintf(err, errlen, "%s prefill failed at token %d",
                                         q36_engine_uses_vulkan_runtime(s->engine) ? "Vulkan" : "CPU", i);
            s->checkpoint_valid = false;
            return 1;
        }
        for (int j = 0; j < n; j++) q36_tokens_push(&s->checkpoint, prompt->v[i + j]);
        s->checkpoint_valid = true;
        i += n;
        if (s->progress) s->progress(s->progress_ud, "prefill_chunk", i, prompt->len);
        if (s->display_progress) s->display_progress(s->display_progress_ud, "prefill_chunk", i, prompt->len);
    }
    return 0;
}

int q36_session_sync(q36_session *s, const q36_tokens *prompt, char *err, size_t errlen) {
    if (!s || !prompt) {
        if (err && errlen) snprintf(err, errlen, "invalid sync arguments");
        return 1;
    }
    if (prompt->len <= 0) {
        if (err && errlen) snprintf(err, errlen, "empty prompt");
        return 1;
    }
    if (prompt->len >= s->ctx_size) {
        if (err && errlen) snprintf(err, errlen, "prompt exceeds context");
        return 1;
    }
    if (s->checkpoint_valid &&
        prompt->len >= s->checkpoint.len &&
        q36_tokens_starts_with(prompt, &s->checkpoint))
    {
        return q36_session_prefill_range(s, prompt, s->checkpoint.len, err, errlen);
    }
    q36_session_reset_runtime(s);
    s->mtp_draft_valid = false;
    s->mtp_draft_token = -1;
    s->checkpoint.len = 0;
    s->checkpoint_valid = false;
    return q36_session_prefill_range(s, prompt, 0, err, errlen);
}

bool q36_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common) {
    if (live_len < 0 || canonical_len < 0 || common < 0) return true;
    if (common > live_len || common > canonical_len) return true;
    return common < live_len;
}

q36_session_rewrite_result q36_session_rewrite_from_common(q36_session *s, const q36_tokens *prompt, int common,
                                                           char *err, size_t errlen) {
    if (!s || !prompt || common < 0 || common > prompt->len) {
        if (err && errlen) snprintf(err, errlen, "invalid rewrite arguments");
        return Q36_SESSION_REWRITE_ERROR;
    }
    if (q36_session_rewrite_requires_rebuild(s->checkpoint.len, prompt->len, common)) {
        return q36_session_sync(s, prompt, err, errlen) == 0 ? Q36_SESSION_REWRITE_OK : Q36_SESSION_REWRITE_ERROR;
    }
    if (common == s->checkpoint.len && common == prompt->len) return Q36_SESSION_REWRITE_OK;
    s->checkpoint.len = common;
    s->checkpoint_valid = true;
    if (q36_session_prefill_range(s, prompt, common, err, errlen) != 0) {
        return Q36_SESSION_REWRITE_ERROR;
    }
    return Q36_SESSION_REWRITE_OK;
}

int q36_session_common_prefix(q36_session *s, const q36_tokens *prompt) {
    int n;
    int i;
    if (!s || !prompt || !s->checkpoint_valid) return 0;
    n = s->checkpoint.len < prompt->len ? s->checkpoint.len : prompt->len;
    i = 0;
    while (i < n && s->checkpoint.v[i] == prompt->v[i]) i++;
    return i;
}

int q36_session_argmax(q36_session *s) {
    if (!s || !s->logits) return -1;
    return sample_argmax(s->logits, Q36_N_VOCAB);
}

int q36_session_argmax_excluding(q36_session *s, int excluded_id) {
    int best = -1;
    float best_logit = Q36_NEG_INF;
    if (!s || !s->logits) return -1;
    for (uint32_t i = 0; i < Q36_N_VOCAB; i++) {
        float v;
        if ((int)i == excluded_id) continue;
        v = s->logits[i];
        if (best < 0 || v > best_logit) {
            best = (int)i;
            best_logit = v;
        }
    }
    return best;
}

int q36_session_sample(q36_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng) {
    if (!s || !s->logits) return -1;
    if (!rng) return q36_session_argmax(s);
    return q36_sample_top_p_min_p(s->logits, Q36_N_VOCAB, temperature, top_k, top_p, min_p, rng);
}

bool q36_session_in_think(q36_session *s) {
    if (!s || !s->checkpoint_valid) return false;
    const q36_vocab *vocab = &s->engine->vocab;
    for (int i = s->checkpoint.len - 1; i >= 0; i--) {
        int t = s->checkpoint.v[i];
        if (t == vocab->think_end_id) return false;
        if (t == vocab->think_start_id) return true;
    }
    return false;
}

/* An EOS while the transcript's <think> block is still unclosed would end the
 * reply with no visible text at all: quantized models near-tie </think> and
 * EOS right after the forced-open <think>.  Callers map that EOS to </think>
 * and keep decoding; at most one substitution per think block, since the
 * close itself ends the in-think state. */
int q36_session_eos_to_think_close(q36_session *s, int token) {
    if (!s || token != s->engine->vocab.eos_id) return token;
    int close = s->engine->vocab.think_end_id;
    if (close < 0 || !q36_session_in_think(s)) return token;
    return close;
}

int q36_session_top_logprobs(q36_session *s, q36_token_score *out, int k) {
    float max_logit = Q36_NEG_INF;
    double sum = 0.0;
    if (!s || !out || !s->logits || k <= 0) return 0;
    if (k > Q36_N_VOCAB) k = Q36_N_VOCAB;
    for (int i = 0; i < k; i++) {
        out[i].id = -1;
        out[i].logit = Q36_NEG_INF;
        out[i].logprob = Q36_NEG_INF;
    }
    for (uint32_t i = 0; i < Q36_N_VOCAB; i++) {
        float v = s->logits[i];
        if (!isfinite(v)) continue;
        if (v > max_logit) max_logit = v;
        for (int j = 0; j < k; j++) {
            if (out[j].id < 0 || v > out[j].logit) {
                for (int l = k - 1; l > j; l--) out[l] = out[l - 1];
                out[j].id = (int)i;
                out[j].logit = v;
                break;
            }
        }
    }
    if (!isfinite(max_logit)) return 0;
    for (uint32_t i = 0; i < Q36_N_VOCAB; i++) {
        float v = s->logits[i];
        if (isfinite(v)) sum += exp((double)v - (double)max_logit);
    }
    {
        double logsum = (double)max_logit + log(sum);
        for (int i = 0; i < k && out[i].id >= 0; i++) {
            out[i].logprob = (float)((double)out[i].logit - logsum);
        }
    }
    return k;
}

static int q36_session_eval_internal(q36_session *s, int token, bool update_mtp,
                                     char *err, size_t errlen) {
    uint32_t pos;
    if (!s) {
        if (err && errlen) snprintf(err, errlen, "missing session");
        return 1;
    }
    if (s->checkpoint.len >= s->ctx_size) {
        if (err && errlen) snprintf(err, errlen, "context exhausted");
        return 1;
    }
    s->mtp_draft_valid = false;
    pos = (uint32_t)s->checkpoint.len;
    if (!(q36_engine_uses_vulkan_runtime(s->engine)
#ifndef Q36_NO_GPU
          ? q36_forward_token_vulkan(s, token, pos, true)
#else
          ? false
#endif
          : q36_forward_token_cpu(s, token, pos, true))) {
        if (err && errlen) snprintf(err, errlen, "%s decode failed at pos %d",
                                     q36_engine_uses_vulkan_runtime(s->engine) ? "Vulkan" : "CPU",
                                     s->checkpoint.len);
        s->checkpoint_valid = false;
        return 1;
    }
#ifndef Q36_NO_GPU
    if (update_mtp && q36_engine_has_mtp(s->engine) && q36_engine_uses_vulkan_runtime(s->engine)) {
        q36_vulkan_runtime *rt = (q36_vulkan_runtime *)s->runtime;
        if (!q36_mtp_eval_vulkan(s, token, pos, rt->norm)) {
            if (err && errlen) snprintf(err, errlen, "MTP draft failed at pos %u", pos);
            s->checkpoint_valid = false;
            return 1;
        }
        s->mtp_draft_token = sample_argmax_margin(s->mtp_logits, Q36_N_VOCAB, &s->mtp_draft_margin);
        s->mtp_draft_valid = s->mtp_draft_token >= 0;
    }
#else
    (void)update_mtp;
#endif
    q36_tokens_push(&s->checkpoint, token);
    s->checkpoint_valid = true;
    return 0;
}

#ifndef Q36_NO_GPU
/* Decode one token with the MTP draft merged into the same submission:
 * the pipeline drains once (at the draft logits read) instead of once
 * per model, which is what makes the confidence gate affordable. */
static int q36_session_eval_with_draft(q36_session *s, int token, char *err, size_t errlen) {
    q36_vulkan_runtime *rt;
    uint32_t pos;
    if (s->checkpoint.len >= s->ctx_size) {
        if (err && errlen) snprintf(err, errlen, "context exhausted");
        return 1;
    }
    rt = (q36_vulkan_runtime *)s->runtime;
    s->mtp_draft_valid = false;
    pos = (uint32_t)s->checkpoint.len;
    {
        const bool timing = getenv("Q36_MTP_TIMING") != NULL;
        double t0 = timing ? q36_now_sec() : 0.0, t1 = t0, t2 = t0, t3 = t0, t4 = t0;
        bool ok = q36_forward_tokens_vulkan_into(s, &token, 1, pos, rt->logits, false) &&
                  q36_vulkan_update_last_h(rt, 0);
        if (timing) t1 = q36_now_sec();
        ok = ok && q36_mtp_eval_vulkan(s, token, pos, rt->norm);
        if (timing) t2 = q36_now_sec();
        ok = ok && q36_gpu_tensor_read(rt->logits, 0, s->logits, (uint64_t)Q36_N_VOCAB * sizeof(*s->logits));
        if (timing) t3 = q36_now_sec();
        if (!ok) {
            if (err && errlen) snprintf(err, errlen, "Vulkan decode+draft failed at pos %u", pos);
            s->checkpoint_valid = false;
            return 1;
        }
        s->mtp_draft_token = sample_argmax_margin(s->mtp_logits, Q36_N_VOCAB, &s->mtp_draft_margin);
        if (timing) {
            t4 = q36_now_sec();
            fprintf(stderr, "q36: mtp timing eval fwd=%.2f mtp=%.2f read=%.2f argmax=%.2f ms\n",
                    (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t3 - t2) * 1000.0, (t4 - t3) * 1000.0);
        }
    }
    s->mtp_draft_valid = s->mtp_draft_token >= 0;
    q36_tokens_push(&s->checkpoint, token);
    s->checkpoint_valid = true;
    return 0;
}
#endif

int q36_session_eval(q36_session *s, int token, char *err, size_t errlen) {
    return q36_session_eval_internal(s, token, false, err, errlen);
}

static bool q36_sessions_eval_batch_vulkan_supported(q36_decode_item *items,
                                                      int count,
                                                      q36_engine *e) {
#ifndef Q36_NO_GPU
    const char *enabled = getenv("Q36_VK_SESSION_BATCH");
    if ((enabled && enabled[0] && !strcmp(enabled, "0")) ||
        !items || count < 2 || count > 8 || !e ||
        e->backend != Q36_BACKEND_VULKAN || e->ssd_streaming) {
        return false;
    }
    for (int i = 0; i < count; i++) {
        q36_session *s = items[i].session;
        q36_vulkan_runtime *rt = s ? s->runtime : NULL;
        if (!s || s->engine != e || !s->checkpoint_valid || !rt ||
            rt->prefill_cap < (uint32_t)count) {
            return false;
        }
        for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
            if (!q36_layer_is_full_attention(il)) continue;
            q36_vulkan_full_attn_cache *cache = &rt->full[il];
            bool f16 = cache->type_k == Q36_KV_CACHE_F16 &&
                       cache->type_v == Q36_KV_CACHE_F16;
            bool resident = cache->type_k == Q36_KV_CACHE_Q8_0 &&
                            cache->type_v == Q36_KV_CACHE_Q4_0;
            if (!f16 && !resident) return false;
        }
    }
    return true;
#else
    (void)items;
    (void)count;
    (void)e;
    return false;
#endif
}

int q36_sessions_eval_batch(q36_decode_item *items, int count,
                            char *err, size_t errlen) {
    if (!items || count <= 0) {
        if (err && errlen) snprintf(err, errlen, "empty decode batch");
        return 1;
    }
    if (count == 1) {
        return q36_session_eval(items[0].session, items[0].token, err, errlen);
    }

    q36_session *first = items[0].session;
    if (!first || !first->engine) {
        if (err && errlen) snprintf(err, errlen, "decode batch has no session");
        return 1;
    }
    q36_engine *e = first->engine;
    for (int i = 0; i < count; i++) {
        q36_session *s = items[i].session;
        if (!s || s->engine != e) {
            if (err && errlen) {
                snprintf(err, errlen,
                         "decode batch item %d belongs to a different engine", i);
            }
            return 1;
        }
        if (items[i].token < 0 || items[i].token >= Q36_N_VOCAB) {
            if (err && errlen) {
                snprintf(err, errlen,
                         "decode batch item %d has an invalid token", i);
            }
            return 1;
        }
        if (s->checkpoint.len >= s->ctx_size) {
            if (err && errlen) {
                snprintf(err, errlen,
                         "decode batch item %d reached its context limit", i);
            }
            return 1;
        }
        for (int j = 0; j < i; j++) {
            if (items[j].session == s) {
                if (err && errlen) {
                    snprintf(err, errlen,
                             "decode batch repeats session at items %d and %d",
                             j, i);
                }
                return 1;
            }
        }
    }

#ifndef Q36_NO_GPU
    if (q36_sessions_eval_batch_vulkan_supported(items, count, e)) {
        bool ok = q36_sessions_eval_batch_vulkan(items, count);
        if (!ok) {
            for (int i = 0; i < count; i++) q36_session_invalidate(items[i].session);
            if (err && errlen) snprintf(err, errlen, "Vulkan batched decode failed");
            return 1;
        }
        for (int i = 0; i < count; i++) {
            q36_session *s = items[i].session;
            q36_tokens_push(&s->checkpoint, items[i].token);
            s->checkpoint_valid = true;
            s->mtp_draft_valid = false;
        }
        if (getenv("Q36_VK_SESSION_BATCH_LOG")) {
            fprintf(stderr, "q36: Vulkan session batch rows=%d native=1\n", count);
        }
        return 0;
    }
#endif

    for (int i = 0; i < count; i++) {
        if (q36_session_eval(items[i].session, items[i].token,
                             err, errlen) != 0) {
            for (int j = 0; j < count; j++) q36_session_invalidate(items[j].session);
            return 1;
        }
    }
    if (getenv("Q36_VK_SESSION_BATCH_LOG")) {
        fprintf(stderr, "q36: session batch rows=%d native=0 ordered=1\n", count);
    }
    return 0;
}

int q36_sessions_eval_batch_with_prefill(
        q36_decode_item *items, int count,
        q36_session *prefill_session, const q36_tokens *prefill_prompt,
        char *err, size_t errlen) {
    if (!items || count <= 0 || !prefill_session || !prefill_prompt ||
        !prefill_session->engine) {
        if (err && errlen) snprintf(err, errlen, "invalid mixed model batch");
        return 1;
    }
    if (!prefill_session->checkpoint_valid ||
        prefill_prompt->len <= prefill_session->checkpoint.len ||
        prefill_prompt->len >= prefill_session->ctx_size ||
        !q36_tokens_starts_with(prefill_prompt, &prefill_session->checkpoint)) {
        if (err && errlen) {
            snprintf(err, errlen,
                     "mixed prefill must extend a valid session checkpoint");
        }
        return 1;
    }
    for (int i = 0; i < count; i++) {
        q36_session *s = items[i].session;
        if (!s || s == prefill_session ||
            s->engine != prefill_session->engine ||
            !s->checkpoint_valid || s->checkpoint.len >= s->ctx_size ||
            items[i].token < 0 || items[i].token >= Q36_N_VOCAB) {
            if (err && errlen) snprintf(err, errlen, "invalid mixed decode item %d", i);
            return 1;
        }
        for (int j = 0; j < i; j++) {
            if (items[j].session == s) {
                if (err && errlen) {
                    snprintf(err, errlen,
                             "mixed decode repeats session at items %d and %d",
                             j, i);
                }
                return 1;
            }
        }
    }

    int rc = q36_session_sync(prefill_session, prefill_prompt, err, errlen);
    if (rc != 0) return rc;
    rc = q36_sessions_eval_batch(items, count, err, errlen);
    if (rc != 0) q36_session_invalidate(prefill_session);
    return rc;
}

int q36_session_eval_speculative_argmax(q36_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen) {
    q36_engine *e;
    if (!s || !accepted || accepted_cap <= 0 || max_tokens <= 0) return 0;
    e = s->engine;
    if (!e || !q36_engine_has_mtp(e) || e->mtp_draft_tokens <= 1 ||
        !q36_engine_uses_vulkan_runtime(e)) {
        if (q36_session_eval_internal(s, first_token, false, err, errlen) != 0) return -1;
        accepted[0] = first_token;
        return 1;
    }
#ifndef Q36_NO_GPU
    {
        q36_vulkan_runtime *rt = (q36_vulkan_runtime *)s->runtime;
        int verify[Q36_MTP_MAX_DRAFT];
        int row_tops[Q36_MTP_MAX_DRAFT];
        int draft_cap = e->mtp_draft_tokens - 1;
        int verify_n;
        int commit_n;
        uint32_t start_pos;
        int prev_token;
        if (!rt || !rt->last_h || !rt->logits) return 0;

        /* Draft backoff: after unconfident drafts, decode plainly for a
         * few tokens so hard content does not pay the draft head on every
         * token. Confident content resets to drafting every token. */
        if (s->mtp_backoff > 0) {
            s->mtp_backoff--;
            if (q36_session_eval_internal(s, first_token, false, err, errlen) != 0) return -1;
            accepted[0] = first_token;
            return 1;
        }

        /* Commit the target token with its MTP draft piggybacked on the
         * same submission; the confidence gate and the first-draft check
         * against the fresh target logits are then free. */
        if (q36_session_eval_with_draft(s, first_token, err, errlen) != 0) return -1;
        accepted[0] = first_token;

        if (draft_cap > Q36_MTP_MAX_DRAFT) draft_cap = Q36_MTP_MAX_DRAFT;
        if (draft_cap > accepted_cap - 1) draft_cap = accepted_cap - 1;
        if (draft_cap > max_tokens - 1) draft_cap = max_tokens - 1;
        if (draft_cap > s->ctx_size - s->checkpoint.len) draft_cap = s->ctx_size - s->checkpoint.len;
        if (rt->prefill_cap > 0 && draft_cap > (int)rt->prefill_cap) draft_cap = (int)rt->prefill_cap;
        if (first_token == eos_token || draft_cap <= 0 || !s->mtp_draft_valid) return 1;
        if ((e->mtp_margin > 0.0f && s->mtp_draft_margin < e->mtp_margin) ||
            s->mtp_draft_token != q36_session_argmax(s)) {
            s->mtp_backoff_len = s->mtp_backoff_len ? (s->mtp_backoff_len < 4 ? s->mtp_backoff_len * 2 : 4) : 1;
            s->mtp_backoff = s->mtp_backoff_len;
            q36_mtp_stats_add(0, 0, false);
            return 1;
        }
        s->mtp_backoff_len = 0;

        start_pos = (uint32_t)s->checkpoint.len;
        verify[0] = s->mtp_draft_token;
        verify_n = 1;
        prev_token = verify[0];
        while (verify_n < draft_cap && prev_token != eos_token) {
            float margin = 0.0f;
            if (!q36_mtp_eval_vulkan(s, prev_token, start_pos + (uint32_t)verify_n - 1u, rt->mtp_cur)) {
                if (err && errlen) snprintf(err, errlen, "MTP recursive draft failed at pos %u",
                                            start_pos + (uint32_t)verify_n - 1u);
                s->checkpoint_valid = false;
                return -1;
            }
            s->mtp_draft_token = sample_argmax_margin(s->mtp_logits, Q36_N_VOCAB, &margin);
            if (s->mtp_draft_token < 0) break;
            if (e->mtp_margin > 0.0f && margin < e->mtp_margin) break;
            verify[verify_n++] = s->mtp_draft_token;
            prev_token = s->mtp_draft_token;
        }

        if (!q36_vulkan_spec_frontier_snapshot(rt)) return 1;
        if (!q36_vulkan_verify_suffix_tops(s, verify, (uint32_t)verify_n, start_pos, row_tops)) {
            if (!q36_vulkan_spec_frontier_restore(rt)) q36_session_invalidate(s);
            if (err && errlen) snprintf(err, errlen, "MTP target verifier failed at pos %u", start_pos);
            return -1;
        }

        commit_n = 1;
        for (int i = 1; i < verify_n; i++) {
            if (row_tops[i - 1] != verify[i]) break;
            commit_n++;
        }
        for (int i = 0; i < commit_n; i++) {
            if (verify[i] == eos_token) {
                commit_n = i + 1;
                break;
            }
        }

        if (commit_n == verify_n) {
            /* Full accept: the verify pass already ran every committed
             * token's forward and left the last row's logits in s->logits;
             * adopting its KV/recurrent frontier makes the replay free. */
            if (!q36_vulkan_update_last_h(rt, (uint32_t)verify_n - 1u)) {
                if (err && errlen) snprintf(err, errlen, "MTP frontier adopt failed at pos %u", start_pos);
                q36_session_invalidate(s);
                return -1;
            }
        } else {
            if (!q36_vulkan_spec_frontier_restore(rt)) {
                if (err && errlen) snprintf(err, errlen, "MTP verifier rollback failed at pos %u", start_pos);
                q36_session_invalidate(s);
                return -1;
            }
            /* Replay the accepted prefix as one batch; refreshes s->logits
             * and last_h from the last accepted row. */
            bool replay;
            q36_gpu_set_micro_batch(true);
            replay = q36_forward_tokens_vulkan(s, verify, (uint32_t)commit_n, start_pos, true);
            q36_gpu_set_micro_batch(false);
            if (!replay) {
                if (err && errlen) snprintf(err, errlen, "MTP accepted-prefix replay failed at pos %u", start_pos);
                q36_session_invalidate(s);
                return -1;
            }
        }

        for (int i = 0; i < commit_n; i++) {
            accepted[1 + i] = verify[i];
            q36_tokens_push(&s->checkpoint, verify[i]);
        }
        s->checkpoint_valid = true;
        s->mtp_draft_valid = false;
        s->mtp_draft_token = -1;
        q36_mtp_stats_add(verify_n, commit_n, commit_n == verify_n);
        return 1 + commit_n;
    }
#else
    (void)eos_token;
    if (q36_session_eval_internal(s, first_token, false, err, errlen) != 0) return -1;
    accepted[0] = first_token;
    return 1;
#endif
}

const float *q36_session_logits(q36_session *s, int *n_vocab) {
    if (n_vocab) *n_vocab = Q36_N_VOCAB;
    return s ? s->logits : NULL;
}

static void q36_session_reset_runtime(q36_session *s) {
    if (!s) return;
    if (q36_engine_uses_vulkan_runtime(s->engine)) {
#ifndef Q36_NO_GPU
        (void)q36_vulkan_runtime_reset((q36_vulkan_runtime *)s->runtime);
#endif
    } else {
        q36_cpu_runtime_reset((q36_cpu_runtime *)s->runtime);
    }
}

void q36_session_invalidate(q36_session *s) {
    if (!s) return;
    q36_session_reset_runtime(s);
    s->mtp_draft_valid = false;
    s->mtp_draft_token = -1;
    s->checkpoint_valid = false;
    s->checkpoint.len = 0;
}

void q36_session_rewind(q36_session *s, int pos) {
    q36_tokens prefix = {0};
    if (!s) return;
    if (pos < 0) pos = 0;
    if (pos > s->checkpoint.len) pos = s->checkpoint.len;
    if (pos == s->checkpoint.len) return;
    for (int i = 0; i < pos; i++) q36_tokens_push(&prefix, s->checkpoint.v[i]);
    q36_session_reset_runtime(s);
    s->mtp_draft_valid = false;
    s->mtp_draft_token = -1;
    s->checkpoint.len = 0;
    s->checkpoint_valid = false;
    if (pos > 0) {
        if (q36_session_sync(s, &prefix, NULL, 0) != 0) {
            q36_session_invalidate(s);
        }
    }
    q36_tokens_free(&prefix);
}

int q36_session_pos(q36_session *s) {
    return s ? s->checkpoint.len : 0;
}

int q36_session_ctx(q36_session *s) {
    return s ? s->ctx_size : 0;
}

int q36_engine_routed_quant_bits(q36_engine *e) {
    return e ? e->routed_quant_bits : 0;
}

bool q36_engine_has_mtp(q36_engine *e) {
    return e && e->backend == Q36_BACKEND_VULKAN && e->mtp_ready;
}

int q36_engine_mtp_draft_tokens(q36_engine *e) {
    return q36_engine_has_mtp(e) ? e->mtp_draft_tokens : 1;
}

const q36_tokens *q36_session_tokens(q36_session *s) {
    return s ? &s->checkpoint : NULL;
}

static void q36_payload_set_err(char *err, size_t errlen, const char *msg) {
    if (err && errlen) snprintf(err, errlen, "%s", msg ? msg : "session payload error");
}

static int q36_payload_copy_file_bytes(FILE *src, FILE *dst, uint64_t bytes,
                                       char *err, size_t errlen) {
    unsigned char buf[64 * 1024];
    while (bytes) {
        size_t n = bytes < sizeof(buf) ? (size_t)bytes : sizeof(buf);
        if (fread(buf, 1, n, src) != n) {
            q36_payload_set_err(err, errlen, "failed to read staged session payload");
            return 1;
        }
        if (fwrite(buf, 1, n, dst) != n) {
            q36_payload_set_err(err, errlen, "failed to write staged session payload");
            return 1;
        }
        bytes -= n;
    }
    return 0;
}

static uint64_t q36_full_k_bytes(uint32_t rows, q36_kv_cache_type type) {
    return (uint64_t)rows * q36_kv_cache_row_bytes(type, Q36_N_HEAD_KV * Q36_N_HEAD_DIM);
}

static uint64_t q36_full_v_bytes(uint32_t rows, q36_kv_cache_type type) {
    return (uint64_t)rows * q36_kv_cache_row_bytes(type, Q36_N_HEAD_KV * Q36_N_VALUE_DIM);
}

static uint64_t q36_recurrent_conv_bytes(void) {
    return (uint64_t)(Q36_N_SSM_CONV - 1u) * Q36_N_SSM_CONV_DIM * sizeof(float);
}

static uint64_t q36_recurrent_state_bytes(void) {
    return (uint64_t)Q36_N_SSM_STATE * Q36_N_SSM_STATE * Q36_N_SSM_DT_RANK * sizeof(float);
}

static int q36_payload_write_bytes(FILE *fp, const void *ptr, uint64_t bytes,
                                   char *err, size_t errlen) {
    if (bytes == 0) return 0;
    if (!ptr || fwrite(ptr, 1, (size_t)bytes, fp) != bytes) {
        q36_payload_set_err(err, errlen, "failed to write session payload");
        return 1;
    }
    return 0;
}

static int q36_payload_read_bytes(FILE *fp, void *ptr, uint64_t bytes,
                                  uint64_t *remaining, char *err, size_t errlen) {
    if (bytes == 0) return 0;
    if (!remaining || *remaining < bytes || !ptr ||
        fread(ptr, 1, (size_t)bytes, fp) != bytes)
    {
        q36_payload_set_err(err, errlen, "failed to read session payload");
        return 1;
    }
    *remaining -= bytes;
    return 0;
}

static int q36_payload_write_u32(FILE *fp, uint32_t v, char *err, size_t errlen) {
    return q36_payload_write_bytes(fp, &v, sizeof(v), err, errlen);
}

static int q36_payload_read_u32(FILE *fp, uint32_t *v, uint64_t *remaining,
                                char *err, size_t errlen) {
    return q36_payload_read_bytes(fp, v, sizeof(*v), remaining, err, errlen);
}

#ifndef Q36_NO_GPU
static int q36_payload_write_tensor(FILE *fp, const q36_gpu_tensor *tensor,
                                    uint64_t bytes, char *err, size_t errlen) {
    unsigned char buf[64 * 1024];
    uint64_t off = 0;
    while (off < bytes) {
        uint64_t n = bytes - off;
        if (n > sizeof(buf)) n = sizeof(buf);
        if (!q36_gpu_tensor_read(tensor, off, buf, n) ||
            fwrite(buf, 1, (size_t)n, fp) != n)
        {
            q36_payload_set_err(err, errlen, "failed to write GPU session payload");
            return 1;
        }
        off += n;
    }
    return 0;
}

static int q36_payload_read_tensor(q36_gpu_tensor *tensor, FILE *fp,
                                   uint64_t bytes, uint64_t *remaining,
                                   char *err, size_t errlen) {
    unsigned char buf[64 * 1024];
    uint64_t off = 0;
    if (!remaining || *remaining < bytes) {
        q36_payload_set_err(err, errlen, "truncated GPU session payload");
        return 1;
    }
    while (off < bytes) {
        uint64_t n = bytes - off;
        if (n > sizeof(buf)) n = sizeof(buf);
        if (fread(buf, 1, (size_t)n, fp) != n ||
            !q36_gpu_tensor_write(tensor, off, buf, n))
        {
            q36_payload_set_err(err, errlen, "failed to read GPU session payload");
            return 1;
        }
        off += n;
    }
    *remaining -= bytes;
    return 0;
}
#endif

uint64_t q36_session_payload_bytes(q36_session *s) {
    if (!s) return 0;
    uint32_t rows = (uint32_t)s->checkpoint.len;
    bool typed = s->engine &&
        (s->engine->cache_type_k != Q36_KV_CACHE_F16 ||
         s->engine->cache_type_v != Q36_KV_CACHE_F16);
    uint64_t bytes = (uint64_t)(typed ? Q36_PAYLOAD_U32_FIELDS_TYPED_KV : Q36_PAYLOAD_U32_FIELDS) * sizeof(uint32_t);
    bytes += (uint64_t)rows * sizeof(int32_t);
    bytes += (uint64_t)Q36_N_VOCAB * sizeof(float);
    for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
        if (q36_layer_is_full_attention(il)) {
            bytes += sizeof(uint32_t) +
                     q36_full_k_bytes(rows, s->engine->cache_type_k) +
                     q36_full_v_bytes(rows, s->engine->cache_type_v);
        } else {
            bytes += q36_recurrent_conv_bytes() + q36_recurrent_state_bytes();
        }
    }
    return bytes;
}

int q36_session_write_staged_payload(const q36_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen) {
    if (!payload || !payload->path || !fp) {
        q36_payload_set_err(err, errlen, "invalid staged session payload");
        return 1;
    }
    FILE *src = fopen(payload->path, "rb");
    if (!src) {
        q36_payload_set_err(err, errlen, "failed to open staged session payload");
        return 1;
    }
    int rc = q36_payload_copy_file_bytes(src, fp, payload->bytes, err, errlen);
    if (fclose(src) != 0 && rc == 0) {
        q36_payload_set_err(err, errlen, "failed to close staged session payload");
        return 1;
    }
    return rc;
}

void q36_session_payload_file_free(q36_session_payload_file *payload) {
    if (!payload) return;
    if (payload->path) {
        unlink(payload->path);
        free(payload->path);
    }
    memset(payload, 0, sizeof(*payload));
}

int q36_session_stage_payload(q36_session *s, q36_session_payload_file *out,
                              char *err, size_t errlen) {
    if (!out) {
        q36_payload_set_err(err, errlen, "invalid session payload staging request");
        return 1;
    }
    memset(out, 0, sizeof(*out));
    if (!s || !s->checkpoint_valid) {
        q36_payload_set_err(err, errlen, "session has no valid checkpoint to stage");
        return 1;
    }

    const char *tmpdir = getenv("TMPDIR");
    if (!tmpdir || !tmpdir[0]) tmpdir = "/tmp";
    size_t tmpl_len = strlen(tmpdir) + sizeof("/q36-session-payload.XXXXXX");
    char *tmpl = xmalloc(tmpl_len);
    snprintf(tmpl, tmpl_len, "%s/q36-session-payload.XXXXXX", tmpdir);
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        free(tmpl);
        q36_payload_set_err(err, errlen, "failed to create staged session payload");
        return 1;
    }
    FILE *fp = fdopen(fd, "wb");
    if (!fp) {
        int saved = errno;
        close(fd);
        unlink(tmpl);
        free(tmpl);
        if (err && errlen)
            snprintf(err, errlen, "failed to open staged session payload: %s",
                     strerror(saved));
        return 1;
    }

    int rc = q36_session_save_payload(s, fp, err, errlen);
    if (rc == 0 && fflush(fp) != 0) {
        q36_payload_set_err(err, errlen, "failed to flush staged session payload");
        rc = 1;
    }
    off_t pos = -1;
    if (rc == 0) {
        pos = ftello(fp);
        if (pos < 0) {
            q36_payload_set_err(err, errlen, "failed to measure staged session payload");
            rc = 1;
        }
    }
    if (fclose(fp) != 0 && rc == 0) {
        q36_payload_set_err(err, errlen, "failed to close staged session payload");
        rc = 1;
    }
    if (rc != 0) {
        unlink(tmpl);
        free(tmpl);
        return 1;
    }
    out->path = q36_strdup(tmpl);
    out->bytes = (uint64_t)pos;
    free(tmpl);
    return 0;
}

int q36_session_save_payload(q36_session *s, FILE *fp, char *err, size_t errlen) {
    if (!s || !fp) {
        if (err && errlen) snprintf(err, errlen, "invalid payload save arguments");
        return 1;
    }
    if (!s->checkpoint_valid) {
        q36_payload_set_err(err, errlen, "session has no valid checkpoint to save");
        return 1;
    }
    uint32_t rows = (uint32_t)s->checkpoint.len;
    bool typed = s->engine &&
        (s->engine->cache_type_k != Q36_KV_CACHE_F16 ||
         s->engine->cache_type_v != Q36_KV_CACHE_F16);
    uint32_t h[Q36_PAYLOAD_U32_FIELDS_TYPED_KV] = {
        Q36_PAYLOAD_MAGIC,
        typed ? Q36_PAYLOAD_VERSION_TYPED_KV : Q36_PAYLOAD_VERSION,
        (uint32_t)s->ctx_size,
        s->engine && s->engine->backend == Q36_BACKEND_CPU ?
            ((q36_cpu_runtime *)s->runtime)->prefill_cap :
#ifndef Q36_NO_GPU
            ((q36_vulkan_runtime *)s->runtime)->prefill_cap,
#else
            0,
#endif
        rows,
        Q36_N_VOCAB,
        Q36_N_LAYER,
        Q36_N_HEAD_KV,
        Q36_N_HEAD_DIM,
        Q36_N_VALUE_DIM,
        Q36_N_SSM_CONV,
        Q36_N_SSM_CONV_DIM,
        Q36_N_SSM_STATE,
        Q36_N_SSM_DT_RANK,
        s->engine ? (uint32_t)s->engine->cache_type_k : 0,
        s->engine ? (uint32_t)s->engine->cache_type_v : 0,
    };
    uint32_t nfields = typed ? Q36_PAYLOAD_U32_FIELDS_TYPED_KV : Q36_PAYLOAD_U32_FIELDS;
    for (uint32_t i = 0; i < nfields; i++) {
        if (q36_payload_write_u32(fp, h[i], err, errlen) != 0) return 1;
    }
    if (q36_payload_write_bytes(fp, s->checkpoint.v,
                                (uint64_t)rows * sizeof(int32_t),
                                err, errlen) != 0) return 1;
    if (s->engine->backend == Q36_BACKEND_CPU) {
        q36_cpu_runtime *rt = (q36_cpu_runtime *)s->runtime;
        if (!rt) return q36_unsupported_runtime(err, errlen);
        if (q36_payload_write_bytes(fp, s->logits,
                                    (uint64_t)Q36_N_VOCAB * sizeof(float),
                                    err, errlen) != 0) return 1;
        for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
            if (q36_layer_is_full_attention(il)) {
                q36_full_attn_cache *c = &rt->full[il];
                if (!c->k || !c->v || c->len < rows) {
                    q36_payload_set_err(err, errlen, "CPU KV cache is shorter than checkpoint");
                    return 1;
                }
                if (q36_payload_write_u32(fp, rows, err, errlen) != 0 ||
                    q36_payload_write_bytes(fp, c->k, q36_full_k_bytes(rows, c->type_k), err, errlen) != 0 ||
                    q36_payload_write_bytes(fp, c->v, q36_full_v_bytes(rows, c->type_v), err, errlen) != 0)
                    return 1;
            } else {
                q36_recurrent_cache *c = &rt->recurrent[il];
                if (q36_payload_write_bytes(fp, c->conv, q36_recurrent_conv_bytes(), err, errlen) != 0 ||
                    q36_payload_write_bytes(fp, c->state, q36_recurrent_state_bytes(), err, errlen) != 0)
                    return 1;
            }
        }
        return 0;
    }
#ifdef Q36_NO_GPU
    return q36_unsupported_runtime(err, errlen);
#else
    q36_vulkan_runtime *rt = (q36_vulkan_runtime *)s->runtime;
    if (!rt) return q36_unsupported_runtime(err, errlen);
    if (q36_payload_write_tensor(fp, rt->logits,
                                 (uint64_t)Q36_N_VOCAB * sizeof(float),
                                 err, errlen) != 0) return 1;
    for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
        if (q36_layer_is_full_attention(il)) {
            q36_vulkan_full_attn_cache *c = &rt->full[il];
            if (!c->k || !c->v || c->cap < rows) {
                q36_payload_set_err(err, errlen, "Vulkan KV cache is shorter than checkpoint");
                return 1;
            }
            if (q36_payload_write_u32(fp, rows, err, errlen) != 0 ||
                q36_payload_write_tensor(fp, c->k, q36_full_k_bytes(rows, c->type_k), err, errlen) != 0 ||
                q36_payload_write_tensor(fp, c->v, q36_full_v_bytes(rows, c->type_v), err, errlen) != 0)
                return 1;
        } else {
            q36_vulkan_recurrent_cache *c = &rt->recurrent[il];
            if (q36_payload_write_tensor(fp, c->conv, q36_recurrent_conv_bytes(), err, errlen) != 0 ||
                q36_payload_write_tensor(fp, c->state, q36_recurrent_state_bytes(), err, errlen) != 0)
                return 1;
        }
    }
    return 0;
#endif
}

int q36_session_load_payload(q36_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen) {
    if (!s || !fp) {
        if (err && errlen) snprintf(err, errlen, "invalid payload load arguments");
        return 1;
    }
    uint64_t remaining = payload_bytes;
    uint32_t magic = 0, version = 0, third = 0;
    if (q36_payload_read_u32(fp, &magic, &remaining, err, errlen) != 0 ||
        q36_payload_read_u32(fp, &version, &remaining, err, errlen) != 0 ||
        q36_payload_read_u32(fp, &third, &remaining, err, errlen) != 0)
    {
        if (err && errlen) snprintf(err, errlen, "failed to read session payload header");
        return 1;
    }
    if (magic != Q36_PAYLOAD_MAGIC) {
        if (err && errlen) snprintf(err, errlen, "unsupported session payload format");
        return 1;
    }
    if (version == Q36_PAYLOAD_VERSION_TOKEN_ONLY) {
        uint32_t n_tokens = third;
        if (remaining != (uint64_t)n_tokens * sizeof(int32_t)) {
            if (err && errlen) snprintf(err, errlen, "mismatched token-only payload size");
            return 1;
        }
        q36_tokens prompt = {0};
        for (uint32_t i = 0; i < n_tokens; i++) {
            int32_t tok = 0;
            if (q36_payload_read_bytes(fp, &tok, sizeof(tok), &remaining, err, errlen) != 0) {
                q36_tokens_free(&prompt);
                return 1;
            }
            q36_tokens_push(&prompt, tok);
        }
        s->checkpoint.len = 0;
        s->checkpoint_valid = false;
        int rc = prompt.len ? q36_session_sync(s, &prompt, err, errlen) : 0;
        if (!prompt.len) q36_session_reset_runtime(s);
        q36_tokens_free(&prompt);
        if (rc != 0) return 1;
        s->checkpoint_valid = true;
        return 0;
    }
    if (version != Q36_PAYLOAD_VERSION && version != Q36_PAYLOAD_VERSION_TYPED_KV) {
        if (err && errlen) snprintf(err, errlen, "unsupported session payload version");
        return 1;
    }

    uint32_t nfields = version == Q36_PAYLOAD_VERSION_TYPED_KV ?
        Q36_PAYLOAD_U32_FIELDS_TYPED_KV : Q36_PAYLOAD_U32_FIELDS;
    uint32_t h[Q36_PAYLOAD_U32_FIELDS_TYPED_KV] = {0};
    h[0] = magic;
    h[1] = version;
    h[2] = third;
    for (uint32_t i = 3; i < nfields; i++) {
        if (q36_payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
    }
    uint32_t saved_ctx = h[2];
    uint32_t n_tokens = h[4];
    if (saved_ctx > (uint32_t)s->ctx_size || n_tokens >= (uint32_t)s->ctx_size) {
        q36_payload_set_err(err, errlen, "KV checkpoint does not fit current context");
        return 1;
    }
    if (h[5] != Q36_N_VOCAB || h[6] != Q36_N_LAYER ||
        h[7] != Q36_N_HEAD_KV || h[8] != Q36_N_HEAD_DIM ||
        h[9] != Q36_N_VALUE_DIM || h[10] != Q36_N_SSM_CONV ||
        h[11] != Q36_N_SSM_CONV_DIM || h[12] != Q36_N_SSM_STATE ||
        h[13] != Q36_N_SSM_DT_RANK)
    {
        q36_payload_set_err(err, errlen, "KV checkpoint was written for a different q36 layout");
        return 1;
    }
    q36_kv_cache_type payload_type_k = version == Q36_PAYLOAD_VERSION_TYPED_KV ?
        (q36_kv_cache_type)h[14] : Q36_KV_CACHE_F16;
    q36_kv_cache_type payload_type_v = version == Q36_PAYLOAD_VERSION_TYPED_KV ?
        (q36_kv_cache_type)h[15] : Q36_KV_CACHE_F16;
    if (payload_type_k > Q36_KV_CACHE_Q4_0 || payload_type_v > Q36_KV_CACHE_Q4_0 ||
        !s->engine ||
        s->engine->cache_type_k != payload_type_k ||
        s->engine->cache_type_v != payload_type_v)
    {
        q36_payload_set_err(err, errlen, "KV checkpoint was written with a different cache type");
        return 1;
    }

    q36_session_reset_runtime(s);
    s->checkpoint.len = 0;
    s->checkpoint_valid = false;
    for (uint32_t i = 0; i < n_tokens; i++) {
        int32_t tok = 0;
        if (q36_payload_read_bytes(fp, &tok, sizeof(tok), &remaining, err, errlen) != 0) return 1;
        q36_tokens_push(&s->checkpoint, tok);
    }
    if (s->engine->backend == Q36_BACKEND_CPU) {
        q36_cpu_runtime *rt = (q36_cpu_runtime *)s->runtime;
        if (!rt) return q36_unsupported_runtime(err, errlen);
        if (q36_payload_read_bytes(fp, s->logits,
                                   (uint64_t)Q36_N_VOCAB * sizeof(float),
                                   &remaining, err, errlen) != 0) return 1;
        for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
            if (q36_layer_is_full_attention(il)) {
                uint32_t rows = 0;
                q36_full_attn_cache *c = &rt->full[il];
                if (q36_payload_read_u32(fp, &rows, &remaining, err, errlen) != 0) return 1;
                if (rows != n_tokens || rows > c->cap) {
                    q36_payload_set_err(err, errlen, "KV checkpoint full-attention length mismatch");
                    return 1;
                }
                c->len = rows;
                if (q36_payload_read_bytes(fp, c->k, q36_full_k_bytes(rows, c->type_k), &remaining, err, errlen) != 0 ||
                    q36_payload_read_bytes(fp, c->v, q36_full_v_bytes(rows, c->type_v), &remaining, err, errlen) != 0)
                    return 1;
            } else {
                q36_recurrent_cache *c = &rt->recurrent[il];
                if (q36_payload_read_bytes(fp, c->conv, q36_recurrent_conv_bytes(), &remaining, err, errlen) != 0 ||
                    q36_payload_read_bytes(fp, c->state, q36_recurrent_state_bytes(), &remaining, err, errlen) != 0)
                    return 1;
            }
        }
    } else {
#ifdef Q36_NO_GPU
        return q36_unsupported_runtime(err, errlen);
#else
        q36_vulkan_runtime *rt = (q36_vulkan_runtime *)s->runtime;
        if (!rt) return q36_unsupported_runtime(err, errlen);
        if (q36_payload_read_tensor(rt->logits, fp,
                                    (uint64_t)Q36_N_VOCAB * sizeof(float),
                                    &remaining, err, errlen) != 0) return 1;
        for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
            if (q36_layer_is_full_attention(il)) {
                uint32_t rows = 0;
                q36_vulkan_full_attn_cache *c = &rt->full[il];
                if (q36_payload_read_u32(fp, &rows, &remaining, err, errlen) != 0) return 1;
                if (rows != n_tokens || rows > c->cap) {
                    q36_payload_set_err(err, errlen, "KV checkpoint full-attention length mismatch");
                    return 1;
                }
                if (q36_payload_read_tensor(c->k, fp, q36_full_k_bytes(rows, c->type_k), &remaining, err, errlen) != 0 ||
                    q36_payload_read_tensor(c->v, fp, q36_full_v_bytes(rows, c->type_v), &remaining, err, errlen) != 0)
                    return 1;
            } else {
                q36_vulkan_recurrent_cache *c = &rt->recurrent[il];
                if (q36_payload_read_tensor(c->conv, fp, q36_recurrent_conv_bytes(), &remaining, err, errlen) != 0 ||
                    q36_payload_read_tensor(c->state, fp, q36_recurrent_state_bytes(), &remaining, err, errlen) != 0)
                    return 1;
            }
        }
#endif
    }
    if (remaining != 0) {
        q36_payload_set_err(err, errlen, "trailing bytes in session payload");
        q36_session_invalidate(s);
        return 1;
    }
    s->checkpoint_valid = true;
    return 0;
}

int q36_session_save_snapshot(q36_session *s, q36_session_snapshot *snap, char *err, size_t errlen) {
    q36_payload_header hdr;
    uint64_t bytes;
    uint8_t *p;
    if (!s || !snap) {
        if (err && errlen) snprintf(err, errlen, "invalid snapshot save arguments");
        return 1;
    }
    bytes = sizeof(hdr) + (uint64_t)s->checkpoint.len * sizeof(int32_t);
    if (snap->cap < bytes) {
        snap->ptr = xrealloc(snap->ptr, (size_t)bytes);
        snap->cap = bytes;
    }
    hdr.magic = Q36_PAYLOAD_MAGIC;
    hdr.version = Q36_PAYLOAD_VERSION_TOKEN_ONLY;
    hdr.n_tokens = (uint32_t)s->checkpoint.len;
    memcpy(snap->ptr, &hdr, sizeof(hdr));
    p = snap->ptr + sizeof(hdr);
    if (hdr.n_tokens) memcpy(p, s->checkpoint.v, (size_t)hdr.n_tokens * sizeof(int32_t));
    snap->len = bytes;
    (void)err;
    (void)errlen;
    return 0;
}

int q36_session_load_snapshot(q36_session *s, const q36_session_snapshot *snap, char *err, size_t errlen) {
    const q36_payload_header *hdr;
    const int32_t *tok;
    if (!s || !snap || !snap->ptr || snap->len < sizeof(q36_payload_header)) {
        if (err && errlen) snprintf(err, errlen, "invalid snapshot payload");
        return 1;
    }
    hdr = (const q36_payload_header *)snap->ptr;
    if (hdr->magic != Q36_PAYLOAD_MAGIC ||
        hdr->version != Q36_PAYLOAD_VERSION_TOKEN_ONLY)
    {
        if (err && errlen) snprintf(err, errlen, "unsupported snapshot format");
        return 1;
    }
    if (snap->len != sizeof(*hdr) + (uint64_t)hdr->n_tokens * sizeof(int32_t)) {
        if (err && errlen) snprintf(err, errlen, "mismatched snapshot size");
        return 1;
    }
    tok = (const int32_t *)(snap->ptr + sizeof(*hdr));
    s->mtp_draft_valid = false;
    s->mtp_draft_token = -1;
    s->checkpoint.len = 0;
    s->checkpoint_valid = false;
    for (uint32_t i = 0; i < hdr->n_tokens; i++) q36_tokens_push(&s->checkpoint, tok[i]);
    if (s->checkpoint.len > 0) {
        q36_tokens prompt = {0};
        q36_tokens_copy(&prompt, &s->checkpoint);
        s->checkpoint.len = 0;
        s->checkpoint_valid = false;
        if (q36_session_sync(s, &prompt, err, errlen) != 0) {
            q36_tokens_free(&prompt);
            return 1;
        }
        q36_tokens_free(&prompt);
    } else {
        q36_session_reset_runtime(s);
    }
    s->checkpoint_valid = true;
    return 0;
}

void q36_session_snapshot_free(q36_session_snapshot *snap) {
    if (!snap) return;
    free(snap->ptr);
    memset(snap, 0, sizeof(*snap));
}

int q36_qwen35_n_layer(void) {
    return Q36_N_LAYER;
}

int q36_qwen35_n_embd(void) {
    return Q36_N_EMBD;
}

int q36_qwen35_n_vocab(void) {
    return Q36_N_VOCAB;
}

int q36_engine_prefill_layers(q36_engine *e,
                              const q36_tokens *prompt,
                              int ctx_size,
                              float *layer_out,
                              float *logits,
                              char *err,
                              size_t errlen) {
    (void)e;
    (void)prompt;
    (void)ctx_size;
    (void)layer_out;
    (void)logits;
    return q36_unsupported_runtime(err, errlen);
}

int q36_compare_prefill_layers(q36_engine *q36_e,
                               q36_engine *other_e,
                               const q36_tokens *prompt,
                               int ctx_size,
                               float *q36_layer_out,
                               float *other_layer_out,
                               float *q36_logits,
                               float *other_logits,
                               char *err,
                               size_t errlen) {
    (void)q36_e;
    (void)other_e;
    (void)prompt;
    (void)ctx_size;
    (void)q36_layer_out;
    (void)other_layer_out;
    (void)q36_logits;
    (void)other_logits;
    return q36_unsupported_runtime(err, errlen);
}
