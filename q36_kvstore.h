#ifndef Q36_KVSTORE_H
#define Q36_KVSTORE_H

#include "q36.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define Q36_KVSTORE_FIXED_HEADER 48u
#define Q36_KVSTORE_DEFAULT_MB 4096
#define Q36_KVSTORE_HIT_HALF_LIFE_SECONDS (6ull * 60ull * 60ull)

#define Q36_KVSTORE_EXT_TOOL_MAP          (1u << 0)
#define Q36_KVSTORE_EXT_RESPONSES_VISIBLE (1u << 1)
#define Q36_KVSTORE_EXT_THINKING_VISIBLE  (1u << 2)
#define Q36_KVSTORE_EXT_SESSION_TITLE     (1u << 3)

typedef enum {
    Q36_KVSTORE_REASON_UNKNOWN   = 0,
    Q36_KVSTORE_REASON_COLD      = 1,
    Q36_KVSTORE_REASON_CONTINUED = 2,
    Q36_KVSTORE_REASON_EVICT     = 3,
    Q36_KVSTORE_REASON_SHUTDOWN  = 4,
    Q36_KVSTORE_REASON_AGENT_SYSTEM  = 5,
    Q36_KVSTORE_REASON_AGENT_SESSION = 6,
} q36_kvstore_reason;

typedef enum {
    Q36_KVSTORE_LOG_DEFAULT,
    Q36_KVSTORE_LOG_KVCACHE,
    Q36_KVSTORE_LOG_WARNING,
} q36_kvstore_log_type;

typedef struct {
    /* The file name is the rendered byte prefix, not the token sequence. The
     * payload still carries the exact tokens and graph state; the hash only
     * answers "does this checkpoint represent the bytes at the front of the
     * incoming prompt?" */
    char sha[41];
    char *path;
    uint8_t quant_bits;
    /* Stored in header byte 7.  Flash is 0 for backward compatibility with
     * older cache files where this reserved byte was always written as zero. */
    uint8_t model_id;
    uint8_t reason;
    uint32_t tokens;
    uint32_t hits;
    uint32_t ctx_size;
    uint8_t ext_flags;
    uint64_t created_at;
    uint64_t last_used;
    uint64_t payload_bytes;
    uint64_t text_bytes;
    uint64_t file_size;
} q36_kvstore_entry;

typedef struct {
    int min_tokens;
    int cold_max_tokens;
    int continued_interval_tokens;
    int boundary_trim_tokens;
    int boundary_align_tokens;
} q36_kvstore_options;

typedef struct {
    bool enabled;
    char *dir;
    uint64_t budget_bytes;
    bool reject_different_quant;
    q36_kvstore_options opt;
    int continued_last_store_tokens;
    q36_kvstore_entry *entry;
    int len;
    int cap;
    const char *log_name;
    void *log_ud;
    void (*log)(void *ud, q36_kvstore_log_type type, const char *msg);
} q36_kvstore;

typedef struct {
    const char *text;
    size_t text_len;
    uint8_t model_id;
    uint8_t quant_bits;
    uint32_t ctx_size;
    bool reject_different_quant;
} q36_kvstore_eviction_context;

typedef struct {
    void *ud;
    uint8_t ext_flag;
    bool (*serialized_size)(void *ud, const char *text, uint64_t *bytes_out);
    bool (*write)(void *ud, FILE *fp, const char *text, uint64_t *written_bytes);
    int (*load)(void *ud, FILE *fp, const void *wanted);
    const void *load_wanted;
} q36_kvstore_trailer_hooks;

typedef struct {
    int tokens;
    uint32_t text_bytes;
    uint8_t quant_bits;
    uint8_t ext_flags;
    double load_ms;
    char *path;
} q36_kvstore_load_result;

q36_kvstore_options q36_kvstore_default_options(void);
uint8_t q36_kvstore_reason_code(const char *reason);
const char *q36_kvstore_key_kind(uint8_t ext_flags);
bool q36_kvstore_quant_bits_valid(int quant_bits);

bool q36_kvstore_open(q36_kvstore *kc, const char *dir, uint64_t budget_mb,
                      bool reject_different_quant, q36_kvstore_options opt,
                      const char *log_name,
                      void (*log)(void *ud, q36_kvstore_log_type type, const char *msg),
                      void *log_ud);
void q36_kvstore_close(q36_kvstore *kc);
void q36_kvstore_clear(q36_kvstore *kc);
void q36_kvstore_entry_free(q36_kvstore_entry *e);

char *q36_kvstore_render_tokens_text(q36_engine *engine,
                                     const q36_tokens *tokens,
                                     size_t *out_len);
bool q36_kvstore_byte_prefix_match(const char *text, size_t text_len,
                                   const char *prefix, size_t prefix_len);
void q36_kvstore_tokens_copy_prefix(q36_tokens *dst, const q36_tokens *src, int n);
void q36_kvstore_build_prompt_from_exact_prefix_and_text_suffix(
        q36_engine *engine,
        const q36_tokens *exact_prefix,
        const char *suffix_text,
        q36_tokens *out);

int q36_kvstore_store_len(const q36_kvstore *kc, int tokens);
int q36_kvstore_chat_anchor_pos(const q36_kvstore *kc,
                                const q36_tokens *prompt,
                                int user_token_id,
                                int assistant_token_id);
int q36_kvstore_continued_store_target(const q36_kvstore *kc, int live_tokens);
void q36_kvstore_note_store(q36_kvstore *kc, int tokens);
int q36_kvstore_suppress_continued_store(q36_kvstore *kc, int tokens);
void q36_kvstore_restore_suppressed_continued(q36_kvstore *kc,
                                              int old_tokens,
                                              int suppressed_tokens);

bool q36_kvstore_file_size_fits(const q36_kvstore *kc,
                                uint64_t text_bytes,
                                uint64_t payload_bytes,
                                uint64_t trailer_bytes,
                                uint64_t *file_bytes_out,
                                uint64_t *required_bytes_out);
double q36_kvstore_entry_eviction_score(const q36_kvstore_entry *e,
                                        const q36_tokens *live,
                                        uint64_t now,
                                        const q36_kvstore_eviction_context *incoming);
void q36_kvstore_evict(q36_kvstore *kc, const q36_tokens *live,
                       uint64_t extra_bytes,
                       const q36_kvstore_eviction_context *incoming);
int q36_kvstore_find_text_prefix(q36_kvstore *kc, const char *prompt_text,
                                 int model_id, int quant_bits, int ctx_size);

bool q36_kvstore_store_live_prefix_text(q36_kvstore *kc,
                                        q36_engine *engine,
                                        q36_session *session,
                                        const q36_tokens *tokens,
                                        int store_len,
                                        const char *reason,
                                        const char *cache_text_override,
                                        uint8_t cache_text_ext,
                                        const char *cache_text_key,
                                        const q36_kvstore_trailer_hooks *hooks,
                                        char *err,
                                        size_t err_len);
bool q36_kvstore_store_live_prefix(q36_kvstore *kc,
                                   q36_engine *engine,
                                   q36_session *session,
                                   const q36_tokens *tokens,
                                   int store_len,
                                   const char *reason,
                                   const q36_kvstore_trailer_hooks *hooks,
                                   char *err,
                                   size_t err_len);
bool q36_kvstore_maybe_store_continued(q36_kvstore *kc,
                                       q36_engine *engine,
                                       q36_session *session,
                                       const q36_kvstore_trailer_hooks *hooks,
                                       char *err,
                                       size_t err_len);
int q36_kvstore_try_load_text(q36_kvstore *kc,
                              q36_engine *engine,
                              q36_session *session,
                              const char *prompt_text,
                              q36_tokens *effective_prompt,
                              q36_kvstore_load_result *result,
                              const q36_kvstore_trailer_hooks *hooks,
                              bool responses_protocol);
void q36_kvstore_load_result_free(q36_kvstore_load_result *result);

bool q36_kvstore_read_header(FILE *fp, q36_kvstore_entry *e,
                             uint32_t *text_bytes);
bool q36_kvstore_read_entry_file(const char *path, const char sha[41],
                                 q36_kvstore_entry *out);
void q36_kvstore_fill_header(uint8_t h[Q36_KVSTORE_FIXED_HEADER],
                             uint8_t model_id, uint8_t quant_bits,
                             uint8_t reason, uint8_t ext_flags,
                             uint32_t tokens, uint32_t hits, uint32_t ctx_size,
                             uint64_t created_at, uint64_t last_used,
                             uint64_t payload_bytes);
bool q36_kvstore_touch_file(const char *path, uint32_t hits);
bool q36_kvstore_sha_hex_name(const char *name, char sha[41]);
void q36_kvstore_sha1_bytes_hex(const void *ptr, size_t len, char out[41]);
char *q36_kvstore_path_join(const char *dir, const char *name);
char *q36_kvstore_path_for_sha(q36_kvstore *kc, const char sha[41]);
void q36_kvstore_le_put32(uint8_t *p, uint32_t v);
uint32_t q36_kvstore_le_get32(const uint8_t *p);

#endif
