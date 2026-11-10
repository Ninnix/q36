#ifndef Q36_DISTRIBUTED_H
#define Q36_DISTRIBUTED_H

#include "q36.h"

#define Q36_DIST_VERIFY_MAX 8u

typedef q36_distributed_options q36_dist_options;
typedef struct q36_dist_engine q36_dist_engine;
typedef struct q36_dist_session q36_dist_session;

typedef enum {
    Q36_DIST_CLI_ERROR = -1,
    Q36_DIST_CLI_NOT_MATCHED = 0,
    Q36_DIST_CLI_MATCHED = 1,
} q36_dist_cli_parse_result;

typedef struct {
    uint32_t n_layers, n_embd, n_vocab, prefill_cap, context_size;
    uint32_t n_hidden; /* Residual wire width; zero uses n_embd. */
    uint32_t layer_start, layer_end;
    bool speculation, mtp;
    uint64_t fingerprint;
} q36_dist_model_info;

bool q36_dist_enabled(const q36_dist_options *opt);
void q36_dist_usage(FILE *fp);
q36_dist_cli_parse_result q36_dist_parse_cli_arg(const char *arg, int *index,
        int argc, char **argv, q36_dist_options *opt, char *err, size_t errlen);
int q36_dist_prepare_engine_options(const q36_dist_options *opt,
        q36_engine_options *engine, char *err, size_t errlen);
int q36_dist_engine_open(q36_dist_engine **out, q36_engine *engine,
        const q36_dist_options *opt, const q36_dist_model_info *info);
void q36_dist_engine_close(q36_dist_engine *d);
int q36_dist_run(q36_engine *engine);
int q36_dist_session_create(q36_dist_session **out, q36_dist_engine *engine, int ctx);
void q36_dist_session_free(q36_dist_session *d);
void q36_dist_session_lock(q36_dist_session *d);
void q36_dist_session_unlock(q36_dist_session *d);
int q36_dist_session_sync(q36_dist_session *d, q36_session *owner,
        const q36_tokens *checkpoint, const q36_tokens *prompt, float *logits,
        char *err, size_t errlen);
int q36_dist_session_eval(q36_dist_session *d, q36_session *owner,
        const q36_tokens *checkpoint, int token, float *logits, char *err, size_t errlen);
int q36_dist_session_speculate(q36_dist_session *d, q36_session *owner,
        const q36_tokens *checkpoint, const int *tokens, uint32_t count,
        int eos, int excluded, float *logits, char *err, size_t errlen);

/* Each stage owns its recurrent and attention history. Slice input/output is
 * the raw residual stream; normalization belongs to the receiving layer. */
int q36_session_layer_slice_reset(q36_session *s, char *err, size_t errlen);
int q36_session_eval_layer_slice(q36_session *s, const int *tokens,
        uint32_t n_tokens, uint32_t pos0, uint32_t layer_start, uint32_t layer_end,
        const float *input_hidden, float *output_hidden, bool output_logits,
        float *logits, char *err, size_t errlen);
int q36_session_read_final_hidden(q36_session *s, uint32_t n_tokens,
        float *hidden, char *err, size_t errlen);
int q36_session_project_logits(q36_session *s, const float *hidden,
        float *logits, char *err, size_t errlen);
int q36_session_read_final_rows(q36_session *s, uint32_t count,
        float *hidden, char *err, size_t errlen);
int q36_session_pipeline_mtp_catchup(q36_session *s, const int *tokens,
        uint32_t count, uint32_t pos, const float *hidden, char *err, size_t errlen);
int q36_session_pipeline_spec_begin(q36_session *s, char *err, size_t errlen);
int q36_session_pipeline_spec_commit(q36_session *s, uint32_t count,
        char *err, size_t errlen);
int q36_session_pipeline_project(q36_session *s, const float *hidden,
        int excluded, int *top, char *err, size_t errlen);
int q36_session_pipeline_read_logits(q36_session *s, float *logits,
        char *err, size_t errlen);
int q36_session_verify_pipeline(q36_session *s, const int *tokens, uint32_t count,
        int eos, int excluded, char *err, size_t errlen);
void q36_engine_distributed_info(q36_engine *e, q36_dist_model_info *info);
q36_dist_engine *q36_engine_distributed(q36_engine *e);
bool q36_session_distributed_cancelled(q36_session *s);
void q36_session_distributed_progress(q36_session *s, int current, int total);

#endif
