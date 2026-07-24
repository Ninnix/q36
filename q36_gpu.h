#ifndef Q36_GPU_H
#define Q36_GPU_H

#include <stdbool.h>
#include <stdint.h>

/* =========================================================================
 * GPU Tensor and Command Lifetime.
 * =========================================================================
 *
 * Opaque device tensor used by the Q36-specific GPU executor.
 *
 * The public GPU API is tensor-resident: activations, KV/SSM state, and
 * scratch buffers stay device-owned across the whole prefill/decode command
 * sequence.
 */
typedef struct q36_gpu_tensor q36_gpu_tensor;

int q36_gpu_init(void);
void q36_gpu_cleanup(void);

/* Host-side row splitter for Vulkan parity ops. Each callback must preserve
 * the serial arithmetic order inside every row/token it handles. */
typedef void (*q36_gpu_parallel_fn)(void *ctx, uint64_t row0, uint64_t row1);
void q36_gpu_parallel_for_rows(uint64_t n_rows,
                               uint64_t min_parallel_rows,
                               q36_gpu_parallel_fn fn,
                               void *ctx);

q36_gpu_tensor *q36_gpu_tensor_alloc(uint64_t bytes);
q36_gpu_tensor *q36_gpu_tensor_view(const q36_gpu_tensor *base, uint64_t offset, uint64_t bytes);
void q36_gpu_tensor_free(q36_gpu_tensor *tensor);
uint64_t q36_gpu_tensor_bytes(const q36_gpu_tensor *tensor);
void *q36_gpu_tensor_contents(q36_gpu_tensor *tensor);
/* Same, but the flush this mapping may force shows up in the op profile
 * under `reason` instead of the generic submit_wait_tensor_contents. */
void *q36_gpu_tensor_contents_named(q36_gpu_tensor *tensor, const char *reason);
int q36_gpu_tensor_write(q36_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);
int q36_gpu_tensor_read(const q36_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);
int q36_gpu_tensor_copy(q36_gpu_tensor *dst, uint64_t dst_offset,
                          const q36_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes);

int q36_gpu_begin_commands(void);
int q36_gpu_flush_commands(void);
int q36_gpu_end_commands(void);
int q36_gpu_synchronize(void);

int q36_gpu_set_model_map(const void *model_map, uint64_t model_size);
int q36_gpu_set_model_fd(int fd);
int q36_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size);
int q36_gpu_set_model_map_spans(const void *model_map, uint64_t model_size, const uint64_t *offsets, const uint64_t *sizes, uint32_t count, uint64_t max_tensor_bytes);
int q36_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label);
int q36_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label);
void q36_gpu_set_quality(bool quality);
void q36_gpu_set_micro_batch(bool enabled);
/* True when the fused attention path will serve q36_gpu_attn_decode_tensor;
 * the runtime then skips the per-context scores scratch. Sampled once. */
bool q36_gpu_attn_fused_enabled(void);
void q36_gpu_set_ssd_streaming(bool enabled);
void q36_gpu_set_streaming_full_layers(uint32_t layers);
void q36_gpu_set_streaming_expert_cache_budget(uint32_t experts);
void q36_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes);
uint64_t q36_gpu_recommended_working_set_size(void);
uint32_t q36_gpu_stream_expert_cache_configured_count(void);
uint32_t q36_gpu_stream_expert_cache_current_count(void);

typedef struct q36_gpu_stream_expert_table {
    const void *model_map;
    uint64_t model_size;
    uint32_t layer;
    uint32_t n_total_expert;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_scales_offset;
    uint64_t up_scales_offset;
    uint64_t down_scales_offset;
    uint64_t gate_expert_bytes;
    uint64_t up_expert_bytes;
    uint64_t down_expert_bytes;
    uint32_t gate_type;
    uint32_t up_type;
    uint32_t down_type;
    bool has_gate_scales;
    bool has_up_scales;
    bool has_down_scales;
} q36_gpu_stream_expert_table;

void q36_gpu_stream_expert_cache_reset_route_hotness(void);
void q36_gpu_stream_expert_cache_note_tokens(uint32_t n_tokens);
void q36_gpu_stream_expert_cache_release_resident(void);
uint32_t q36_gpu_stream_expert_cache_budget_for_expert_size(uint64_t gate_expert_bytes, uint64_t up_expert_bytes, uint64_t down_expert_bytes);
int q36_gpu_stream_expert_cache_seed_selected(const q36_gpu_stream_expert_table *table, const int32_t *selected_ids, uint32_t n_selected);
int q36_gpu_stream_expert_cache_begin_selected_load(const q36_gpu_stream_expert_table *table, const int32_t *selected_ids, uint32_t n_selected);
int q36_gpu_stream_expert_cache_prepare_selected_batch(const q36_gpu_stream_expert_table *table, const int32_t *selected_ids, uint32_t n_tokens, uint32_t n_selected);
int q36_gpu_stream_expert_cache_load_layer(const q36_gpu_stream_expert_table *table);
int q36_gpu_stream_expert_cache_seed_from_layer_selected(const q36_gpu_stream_expert_table *table, const q36_gpu_tensor *selected, uint32_t n_tokens, uint32_t n_seed_tokens, uint32_t n_selected);
int q36_gpu_stream_expert_cache_release_layer_cache(void);
int q36_gpu_stream_expert_cache_seed_experts(const q36_gpu_stream_expert_table *table, const int32_t *expert_ids, const uint32_t *expert_priorities, uint32_t n_experts);
void q36_gpu_print_memory_report(const char *label);
void q36_gpu_prof_reset(void);
void q36_gpu_prof_report(const char *label);

/* =========================================================================
 * Dense Projections and Norms.
 * =========================================================================
 *
 * Matvec weights are read straight from the mmapped GGUF at weight_offset and
 * cached on device on first use.  The *_scaled variants apply the scalar from
 * the matching ".scale" model tensor to the output, like q36_scale_inplace()
 * after each CPU matvec.  The k-quant path quantizes activations to q8_K and
 * uses the same integer dot ggml's CPU kernels use.
 */

int q36_gpu_quantize_q8_k_tensor(
        q36_gpu_tensor       *out,
        const q36_gpu_tensor *x,
        uint64_t                in_dim,
        uint64_t                n_tok);

int q36_gpu_matmul_f16_tensor(
        q36_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const q36_gpu_tensor *x,
        uint64_t                n_tok);

int q36_gpu_matmul_f32_tensor(
        q36_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const q36_gpu_tensor *x,
        uint64_t                n_tok);

int q36_gpu_matmul_f32_scaled_tensor(
        q36_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const q36_gpu_tensor *x,
        uint64_t                n_tok,
        float                   scale);

int q36_gpu_matmul_q8_0_scaled_tensor(
        q36_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const q36_gpu_tensor *x,
        uint64_t                n_tok,
        float                   scale);

int q36_gpu_matmul_k_quant_tensor(
        q36_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const q36_gpu_tensor *x,
        uint64_t                n_tok);

int q36_gpu_matmul_k_quant_scaled_tensor(
        q36_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const q36_gpu_tensor *x,
        uint64_t                n_tok,
        float                   scale);

int q36_gpu_matmul_k_quant_q8_scaled_tensor(
        q36_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const q36_gpu_tensor *q8,
        uint64_t                n_tok,
        float                   scale);

int q36_gpu_rms_norm_plain_rows_tensor(
        q36_gpu_tensor       *out,
        const q36_gpu_tensor *x,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int q36_gpu_rms_norm_weight_rows_tensor(
        q36_gpu_tensor       *out,
        const q36_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int q36_gpu_swiglu_tensor(
        q36_gpu_tensor       *out,
        const q36_gpu_tensor *gate,
        const q36_gpu_tensor *up,
        uint32_t                n,
        float                   clamp,
        float                   weight);

int q36_gpu_ffn_tail_tensor(
        q36_gpu_tensor       *out,
        const q36_gpu_tensor *shared,
        const q36_gpu_tensor *scalar,
        uint32_t                out_dim,
        uint32_t                n_tok);

int q36_gpu_add_tensor(
        q36_gpu_tensor       *out,
        const q36_gpu_tensor *a,
        const q36_gpu_tensor *b,
        uint32_t                n);

int q36_gpu_directional_steering_project_tensor(
        q36_gpu_tensor       *x,
        const q36_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale);

/* out_sum = a + b, out_norm = rmsnorm(out_sum) * weight, fused per row.
 * out_sum may alias a or b. */
int q36_gpu_add_rms_norm_tensor(
        q36_gpu_tensor       *out_norm,
        q36_gpu_tensor       *out_sum,
        const q36_gpu_tensor *a,
        const q36_gpu_tensor *b,
        const void            *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

/* =========================================================================
 * Qwen3.6 Decode Kernels.
 * =========================================================================
 *
 * Faithful GPU ports of the CPU reference path in q36.c, one entry per step
 * of q36_forward_token_cpu().
 */

/* q36_l2_norm() over rows, in place. */
int q36_gpu_l2_norm_rows_tensor(
        q36_gpu_tensor *x,
        uint32_t          n,
        uint32_t          rows,
        float             eps);

int q36_gpu_recurrent_conv_step_tensor(
        q36_gpu_tensor       *cache_conv,
        const q36_gpu_tensor *cur,
        q36_gpu_tensor       *window,
        uint32_t                n_tok);

int q36_gpu_recurrent_conv_silu_tensor(
        q36_gpu_tensor       *cache_conv,
        const q36_gpu_tensor *cur,
        q36_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              kernel_offset,
        uint32_t              conv_dim,
        uint32_t              n_taps,
        uint32_t              n_tok);

int q36_gpu_extract_full_attn_q_tensor(
        q36_gpu_tensor       *dst,
        const q36_gpu_tensor *qg,
        uint32_t                n_tok);

/* Device-side contiguous f32 copy (dst[0..n) = src[0..n)), so host-staged
 * data reaches a GPU-written tensor without mapping it. */
int q36_gpu_copy_f32_tensor(
        q36_gpu_tensor       *dst,
        const q36_gpu_tensor *src,
        uint32_t                n_floats);

int q36_gpu_extract_recurrent_v_tensor(
        q36_gpu_tensor       *dst,
        const q36_gpu_tensor *conv,
        uint32_t                n_tok);

int q36_gpu_rope_qwen_rows_tensor(
        q36_gpu_tensor *x,
        uint32_t          n_head,
        uint32_t          pos0,
        uint32_t          n_tok);

int q36_gpu_rms_norm_rope_qwen_rows_tensor(
        q36_gpu_tensor       *dst,
        const q36_gpu_tensor *src,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              src_stride,
        uint32_t              n_head,
        uint32_t              pos0,
        uint32_t              n_tok,
        float                 eps);

int q36_gpu_rms_norm_rope_qwen_kv_store_tensor(
        q36_gpu_tensor       *k_cache,
        q36_gpu_tensor       *v_cache,
        const q36_gpu_tensor *k,
        const q36_gpu_tensor *v,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              src_stride,
        uint32_t              n_head,
        uint32_t              pos0,
        uint32_t              n_tok,
        uint32_t              cap,
        float                 eps);

int q36_gpu_rms_norm_rope_qwen_kv_store_quant_tensor(
        q36_gpu_tensor       *k_cache,
        q36_gpu_tensor       *v_cache,
        const q36_gpu_tensor *k,
        const q36_gpu_tensor *v,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              src_stride,
        uint32_t              n_head,
        uint32_t              pos0,
        uint32_t              n_tok,
        uint32_t              cap,
        float                 eps,
        uint32_t              k_row_bytes,
        uint32_t              v_row_bytes);

int q36_gpu_shared_ffn_decode_tensor(
        q36_gpu_tensor       *out,
        q36_gpu_tensor       *mid,
        const q36_gpu_tensor *x,
        const q36_gpu_tensor *scalar,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              down_offset,
        uint32_t              in_dim,
        uint32_t              mid_dim,
        uint32_t              out_dim,
        float                 gate_scale,
        float                 up_scale,
        float                 down_scale);


/* q36_ssm_conv_apply() + SiLU over n_tok windows: each window holds n_taps
 * rows of conv_dim, oldest first; the kernel weight is the model ssm_conv1d
 * tensor. */
int q36_gpu_ssm_conv_silu_tensor(
        q36_gpu_tensor       *out,
        const q36_gpu_tensor *window,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                kernel_offset,
        uint32_t                conv_dim,
        uint32_t                n_taps,
        uint32_t                n_tok);

/* Broadcast the conv output groups into per-head q/k and L2-normalize each
 * head (the gather + q36_l2_norm() loop in q36_forward_recurrent()), for
 * n_tok conv rows of conv_stride floats each. */
int q36_gpu_delta_qk_l2_norm_tensor(
        q36_gpu_tensor       *q_out,
        q36_gpu_tensor       *k_out,
        const q36_gpu_tensor *conv,
        uint32_t                n_heads,
        uint32_t                n_groups,
        uint32_t                state_dim,
        uint32_t                conv_stride,
        uint32_t                n_tok,
        float                   eps);

int q36_gpu_delta_qkv_l2_norm_tensor(
        q36_gpu_tensor       *q_out,
        q36_gpu_tensor       *k_out,
        q36_gpu_tensor       *v_out,
        const q36_gpu_tensor *conv,
        uint32_t              n_heads,
        uint32_t              n_groups,
        uint32_t              state_dim,
        uint32_t              conv_stride,
        uint32_t              n_tok,
        float                 eps);

/* Per-head decay/beta prep over n_tok rows: gb[h] = softplus(alpha+dt_bias)*a
 * and gb[n_heads+h] = sigmoid(beta). */
int q36_gpu_delta_net_gates_tensor(
        q36_gpu_tensor       *gb,
        const q36_gpu_tensor *alpha,
        const q36_gpu_tensor *beta,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                dt_bias_offset,
        uint64_t                a_offset,
        uint32_t                n_heads,
        uint32_t                n_tok);

/* q36_delta_net_decode(): n_tok recurrent state updates + readouts, applied
 * strictly in token order. */
int q36_gpu_delta_net_decode_tensor(
        q36_gpu_tensor       *state,
        const q36_gpu_tensor *q,
        const q36_gpu_tensor *k,
        const q36_gpu_tensor *v,
        const q36_gpu_tensor *gb,
        q36_gpu_tensor       *out,
        uint32_t                n_heads,
        uint32_t                state_dim,
        uint32_t                n_tok);

/* q36_full_attn_cache_store(): append n_tok K/V row pairs starting at pos0. */
int q36_gpu_attn_kv_store_tensor(
        q36_gpu_tensor       *k_cache,
        q36_gpu_tensor       *v_cache,
        const q36_gpu_tensor *k,
        const q36_gpu_tensor *v,
        uint32_t                pos0,
        uint32_t                n_tok,
        uint32_t                cap,
        uint32_t                k_row,
        uint32_t                v_row,
        uint32_t                k_cache_type,
        uint32_t                v_cache_type,
        uint32_t                k_cache_row_bytes,
        uint32_t                v_cache_row_bytes);

/* Full-attention decode over the typed KV cache: scores, softmax with optional
 * attention sinks, weighted V sum, and sigmoid output gating from the gate
 * half of the fused q projection (qg rows are [head][2*head_dim]).  Causal
 * batch: query row t sits at position pos0 + t and attends keys 0..pos0+t.
 * scores is caller scratch of n_tok*n_head*(pos0+n_tok) floats, clobbered. */
int q36_gpu_attn_decode_tensor(
        q36_gpu_tensor       *out,
        const q36_gpu_tensor *q,
        const q36_gpu_tensor *qg,
        const q36_gpu_tensor *k_cache,
        const q36_gpu_tensor *v_cache,
        q36_gpu_tensor       *scores,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        bool                    has_sinks,
        uint32_t                pos0,
        uint32_t                n_tok,
        uint32_t                n_head,
        uint32_t                n_head_kv,
        uint32_t                head_dim,
        uint32_t                k_cache_type,
        uint32_t                v_cache_type,
        uint32_t                k_cache_row_bytes,
        uint32_t                v_cache_row_bytes);

/* Router softmax + top-k + weight normalization over n_tok logit rows;
 * route_scale is the model expert_weights_scale. */
int q36_gpu_router_topk_tensor(
        q36_gpu_tensor       *selected,
        q36_gpu_tensor       *weights,
        const q36_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_used,
        uint32_t                n_tok,
        float                   route_scale);

/* One routed expert weight: GGUF offset/type of the 3D expert tensor plus the
 * optional per-expert ".scale" tensor (n_expert f32). */
typedef struct {
    uint64_t offset;
    uint32_t type;
    uint64_t scales_offset;
    bool has_scales;
} q36_gpu_moe_weight;

/* Routed expert FFN over n_tok rows of x with per-token expert selections,
 * mirroring the routed half of q36_forward_ffn() including all .scale
 * applications.  Supports Q2_K / Q4_K / Q5_K / Q6_K / IQ2_XXS / IQ2_S / IQ3_S expert quants. */
int q36_gpu_moe_ffn_tensor(
        q36_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        const q36_gpu_moe_weight *gate,
        const q36_gpu_moe_weight *up,
        const q36_gpu_moe_weight *down,
        const q36_gpu_tensor *selected,
        const q36_gpu_tensor *weights,
        uint32_t                layer,
        uint32_t                n_used,
        const q36_gpu_tensor *x,
        uint32_t                n_tok,
        uint32_t                in_dim,
        uint32_t                mid_dim,
        uint32_t                out_dim,
        uint32_t                n_expert);

int q36_gpu_moe_ffn_q8_tensor(
        q36_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        const q36_gpu_moe_weight *gate,
        const q36_gpu_moe_weight *up,
        const q36_gpu_moe_weight *down,
        const q36_gpu_tensor *selected,
        const q36_gpu_tensor *weights,
        uint32_t                layer,
        uint32_t                n_used,
        const q36_gpu_tensor *qx,
        uint32_t                n_tok,
        uint32_t                in_dim,
        uint32_t                mid_dim,
        uint32_t                out_dim,
        uint32_t                n_expert);

int q36_gpu_moe_ffn_f32_tensor(
        q36_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        const q36_gpu_moe_weight *gate,
        const q36_gpu_moe_weight *up,
        const q36_gpu_moe_weight *down,
        const q36_gpu_tensor *selected,
        const q36_gpu_tensor *weights,
        uint32_t                layer,
        uint32_t                n_used,
        const q36_gpu_tensor *x,
        uint32_t                n_tok,
        uint32_t                in_dim,
        uint32_t                mid_dim,
        uint32_t                out_dim,
        uint32_t                n_expert);

#endif
