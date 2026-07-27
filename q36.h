#ifndef Q36_H
#define Q36_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "q36_ssd.h"

#define Q36_DEFAULT_MODEL_FILE "Qwen3.6-35B-A3B-AntirezExperts-IQ2XXS-gateup-Q2K-down-Q8rest.gguf"
#define Q36_DEFAULT_MODEL_PATH "gguf/" Q36_DEFAULT_MODEL_FILE

typedef enum {
    Q36_BACKEND_VULKAN,
    Q36_BACKEND_CPU,
} q36_backend;

typedef enum {
    Q36_KV_CACHE_F16,
    Q36_KV_CACHE_Q8_0,
    Q36_KV_CACHE_Q4_0,
} q36_kv_cache_type;

typedef enum {
    Q36_THINK_NONE,
    Q36_THINK_HIGH,
    Q36_THINK_MAX,
} q36_think_mode;

typedef enum {
    Q36_LOG_DEFAULT,
    Q36_LOG_PREFILL,
    Q36_LOG_GENERATION,
    Q36_LOG_KVCACHE,
    Q36_LOG_TOOL,
    Q36_LOG_WARNING,
    Q36_LOG_TIMING,
    Q36_LOG_OK,
    Q36_LOG_ERROR,
} q36_log_type;

typedef struct {
    int *v;
    int len;
    int cap;
} q36_tokens;

typedef struct {
    int id;
    float logit;
    float logprob;
} q36_token_score;

/* ds4-style sampling: untouched temperature, no fixed-mass or fixed-count
 * truncation, min-p as the only filter.  The floor scales with the top
 * token, so confident steps prune hard while flat (quant-noise) steps drop
 * the junk tail that fixed top-p/top-k let through on the Q2 expert mix. */
#define Q36_DEFAULT_TEMPERATURE 1.0f
#define Q36_DEFAULT_TOP_P 1.0f
#define Q36_DEFAULT_MIN_P 0.05f

typedef struct q36_engine q36_engine;
typedef struct q36_session q36_session;

typedef struct {
    q36_session *session;
    int token;
} q36_decode_item;

typedef void (*q36_session_progress_fn)(void *ud, const char *event, int current, int total);
typedef bool (*q36_session_cancel_fn)(void *ud);

#define Q36_SESSION_SYNC_INTERRUPTED 2

typedef struct {
    const char *model_path;
    const char *mtp_path;
    q36_backend backend;
    int n_threads;
    uint32_t prefill_chunk;
    int mtp_draft_tokens;
    float mtp_margin;
    const char *directional_steering_file;
    float directional_steering_attn;
    float directional_steering_ffn;
    int power_percent;
    uint32_t ssd_streaming_cache_experts;
    uint32_t ssd_streaming_full_layers;
    uint64_t ssd_streaming_cache_bytes;
    uint32_t ssd_streaming_preload_experts;
    uint64_t simulate_used_memory_bytes;
    q36_kv_cache_type cache_type_k;
    q36_kv_cache_type cache_type_v;
    bool warm_weights;
    bool quality;
    bool ssd_streaming;
    bool ssd_streaming_cold;
    bool ssd_streaming_full_layers_set;
} q36_engine_options;

typedef void (*q36_token_emit_fn)(void *ud, int token);
typedef void (*q36_generation_done_fn)(void *ud);

typedef struct {
    uint64_t total_bytes;
    uint64_t raw_bytes;
    uint64_t compressed_bytes;
    uint64_t scratch_bytes;
    uint32_t prefill_cap;
    uint32_t raw_cap;
    uint32_t comp_cap;
} q36_context_memory;

typedef struct {
    uint8_t *ptr;
    uint64_t len;
    uint64_t cap;
} q36_session_snapshot;

typedef struct {
    char *path;
    uint64_t bytes;
} q36_session_payload_file;

int q36_engine_open(q36_engine **out, const q36_engine_options *opt);
void q36_engine_close(q36_engine *e);
void q36_engine_summary(q36_engine *e);
int q36_inspect_model(const char *model_path);
int q36_engine_vocab_size(q36_engine *e);
int q36_engine_power(q36_engine *e);
int q36_engine_set_power(q36_engine *e, int power_percent);
const char *q36_engine_model_name(q36_engine *e);
int q36_engine_model_id(q36_engine *e);
const char *q36_backend_name(q36_backend backend);
const char *q36_kv_cache_type_name(q36_kv_cache_type type);
bool q36_parse_kv_cache_type(const char *s, q36_kv_cache_type *out);
q36_kv_cache_type q36_default_kv_cache_type_k(q36_backend backend, bool ssd_streaming);
q36_kv_cache_type q36_default_kv_cache_type_v(q36_backend backend, bool ssd_streaming);
bool q36_think_mode_enabled(q36_think_mode mode);
const char *q36_think_mode_name(q36_think_mode mode);
const char *q36_think_max_prefix(void);
uint32_t q36_think_max_min_context(void);
q36_think_mode q36_think_mode_for_context(q36_think_mode mode, int ctx_size);
q36_context_memory q36_context_memory_estimate(q36_backend backend, int ctx_size);
q36_context_memory q36_context_memory_estimate_with_prefill(
        q36_backend backend,
        int ctx_size,
        uint32_t prefill_chunk);
q36_context_memory q36_context_memory_estimate_configured(
        q36_backend backend,
        int ctx_size,
        uint32_t prefill_chunk,
        q36_kv_cache_type cache_type_k,
        q36_kv_cache_type cache_type_v);
bool q36_log_is_tty(FILE *fp);
void q36_log(FILE *fp, q36_log_type type, const char *fmt, ...);
int q36_engine_generate_argmax(q36_engine *e, const q36_tokens *prompt,
                               int n_predict, int ctx_size,
                               q36_token_emit_fn emit,
                               q36_generation_done_fn done,
                               void *emit_ud,
                               q36_session_progress_fn progress,
                               void *progress_ud);
void q36_engine_dump_tokens(q36_engine *e, const q36_tokens *tokens);
int q36_dump_text_tokenization(const char *model_path, const char *text, FILE *fp);
int q36_engine_head_test(q36_engine *e, const q36_tokens *prompt);
int q36_engine_first_token_test(q36_engine *e, const q36_tokens *prompt);
int q36_engine_vulkan_graph_test(q36_engine *e, const q36_tokens *prompt);
int q36_engine_vulkan_graph_full_test(q36_engine *e, const q36_tokens *prompt);
int q36_engine_vulkan_graph_prompt_test(q36_engine *e, const q36_tokens *prompt, int ctx_size);

void q36_tokens_push(q36_tokens *tv, int token);
void q36_tokens_free(q36_tokens *tv);
void q36_tokens_copy(q36_tokens *dst, const q36_tokens *src);
bool q36_tokens_starts_with(const q36_tokens *tokens, const q36_tokens *prefix);

void q36_tokenize_text(q36_engine *e, const char *text, q36_tokens *out);
void q36_tokenize_rendered_chat(q36_engine *e, const char *text, q36_tokens *out);
void q36_chat_begin(q36_engine *e, q36_tokens *tokens);
void q36_encode_chat_prompt(
        q36_engine *e,
        const char *system,
        const char *prompt,
        q36_think_mode think_mode,
        q36_tokens *out);
void q36_chat_append_max_effort_prefix(q36_engine *e, q36_tokens *tokens);
void q36_chat_append_message(q36_engine *e, q36_tokens *tokens, const char *role, const char *content);
void q36_chat_append_assistant_prefix(q36_engine *e, q36_tokens *tokens, q36_think_mode think_mode);

char *q36_token_text(q36_engine *e, int token, size_t *len);
int q36_token_eos(q36_engine *e);

int q36_session_create(q36_session **out, q36_engine *e, int ctx_size);
void q36_session_free(q36_session *s);
int q36_session_power(q36_session *s);
int q36_session_set_power(q36_session *s, int power_percent);
void q36_session_set_progress(q36_session *s, q36_session_progress_fn fn, void *ud);
void q36_session_set_display_progress(q36_session *s, q36_session_progress_fn fn, void *ud);
void q36_session_set_cancel(q36_session *s, q36_session_cancel_fn fn, void *ud);

typedef enum {
    Q36_SESSION_REWRITE_ERROR = -1,
    Q36_SESSION_REWRITE_OK = 0,
    Q36_SESSION_REWRITE_REBUILD_NEEDED = 1,
} q36_session_rewrite_result;

int q36_session_sync(q36_session *s, const q36_tokens *prompt, char *err, size_t errlen);
bool q36_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common);
q36_session_rewrite_result q36_session_rewrite_from_common(
        q36_session *s, const q36_tokens *prompt, int common,
        char *err, size_t errlen);
int q36_session_common_prefix(q36_session *s, const q36_tokens *prompt);
int q36_session_argmax(q36_session *s);
int q36_session_argmax_excluding(q36_session *s, int excluded_id);
int q36_session_sample(q36_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
bool q36_session_in_think(q36_session *s);
int q36_session_eos_to_think_close(q36_session *s, int token);
int q36_session_top_logprobs(q36_session *s, q36_token_score *out, int k);
int q36_session_eval(q36_session *s, int token, char *err, size_t errlen);
int q36_sessions_eval_batch(q36_decode_item *items, int count,
                            char *err, size_t errlen);
int q36_sessions_eval_batch_with_prefill(
        q36_decode_item *items, int count,
        q36_session *prefill_session, const q36_tokens *prefill_prompt,
        char *err, size_t errlen);
int q36_session_eval_speculative_argmax(q36_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen);
const float *q36_session_logits(q36_session *s, int *n_vocab);
void q36_session_invalidate(q36_session *s);
void q36_session_rewind(q36_session *s, int pos);
int q36_session_pos(q36_session *s);
int q36_session_ctx(q36_session *s);
int q36_engine_routed_quant_bits(q36_engine *e);
bool q36_engine_has_mtp(q36_engine *e);
int q36_engine_mtp_draft_tokens(q36_engine *e);
const q36_tokens *q36_session_tokens(q36_session *s);
int q36_engine_debug_tensor_row(q36_engine *e, const char *tensor_name, uint64_t row, float *dst, uint32_t n);
int q36_engine_debug_first_tensor_of_type(q36_engine *e, uint32_t type, char *name, size_t name_cap, uint32_t *n);
int q36_engine_debug_tensor_row_packed(q36_engine *e, const char *tensor_name, uint64_t row,
                                       void *dst, uint64_t dst_cap,
                                       uint64_t *row_bytes, uint32_t *type, uint32_t *n);

uint64_t q36_session_payload_bytes(q36_session *s);
int q36_session_stage_payload(q36_session *s, q36_session_payload_file *out,
                              char *err, size_t errlen);
int q36_session_write_staged_payload(const q36_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen);
void q36_session_payload_file_free(q36_session_payload_file *payload);
int q36_session_save_payload(q36_session *s, FILE *fp, char *err, size_t errlen);
int q36_session_load_payload(q36_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);
int q36_session_save_snapshot(q36_session *s, q36_session_snapshot *snap, char *err, size_t errlen);
int q36_session_load_snapshot(q36_session *s, const q36_session_snapshot *snap, char *err, size_t errlen);
void q36_session_snapshot_free(q36_session_snapshot *snap);

int q36_qwen35_n_layer(void);
int q36_qwen35_n_embd(void);
int q36_qwen35_n_vocab(void);
int q36_engine_prefill_layers(q36_engine *e,
                              const q36_tokens *prompt,
                              int ctx_size,
                              float *layer_out,
                              float *logits,
                              char *err,
                              size_t errlen);
int q36_compare_prefill_layers(q36_engine *q36_e,
                               q36_engine *other_e,
                               const q36_tokens *prompt,
                               int ctx_size,
                               float *q36_layer_out,
                               float *other_layer_out,
                               float *q36_logits,
                               float *other_logits,
                               char *err,
                               size_t errlen);

#endif
