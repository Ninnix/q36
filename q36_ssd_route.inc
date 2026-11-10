/* Advisory next-layer routing. The actual router remains authoritative;
 * a bad prediction can only waste read-ahead, never select an expert. */
static bool q36_route_enabled(const q36_session *s) {
#if defined(POSIX_FADV_WILLNEED) && !defined(Q36_METAL)
    const char *hint = getenv("Q36_SSD_ROUTE_PREFETCH");
    return s->engine->ssd_streaming && s->engine->backend == Q36_BACKEND_VULKAN &&
        !Q36_MODEL_DENSE && ((hint && atoi(hint) > 0) || getenv("Q36_SSD_ROUTE_PROFILE"));
#else
    (void)s;
    return false;
#endif
}

static const void *q36_route_read(q36_session *s, const q36_gpu_tensor *input, uint64_t bytes) {
    void *data = q36_gpu_tensor_contents((q36_gpu_tensor *)input);
    if (data) return data;
    if (!s->route_stage) s->route_stage = q36_gpu_tensor_alloc(Q36_N_EMBD*4u);
    if (!s->route_stage || !q36_gpu_tensor_copy(s->route_stage,0,input,0,bytes)) return NULL;
    return q36_gpu_tensor_contents(s->route_stage);
}

static void q36_route_observe(q36_session *s, uint32_t il, const uint32_t *ids, uint32_t count) {
    if (!s->route_pending || s->route_layer != il) return;
    for (uint32_t i = 0; i < 4; i++) {
        for (uint32_t j = 0; j < count; j++) if (s->route_ids[i] == ids[j]) {
            s->route_active_hits++;
            if (j < 4) s->route_top4_hits++;
        }
    }
    s->route_total += 4;
    s->route_pending = false;
    q36_ssd_prefetch_submit(s->engine->route_reader,NULL,0);
}

static void q36_route_prefetch(q36_session *s, uint32_t il, const q36_gpu_tensor *input,
                                const q36_gpu_tensor *residual) {
    q36_engine *e = s->engine;
    if (il + 1 >= Q36_N_LAYER || !q36_route_enabled(s)) return;
    const q36_layer_weights *next = e->weights.layer + il + 1;
    const q36_tensor *router = next->ffn_gate_inp;
    if (!router || router->dim[0] != Q36_N_EMBD || router->dim[1] != Q36_N_EXPERT) return;
    q36_ssd_prefetch_submit(e->route_reader,NULL,0);
    if (!s->route_stage) s->route_stage = q36_gpu_tensor_alloc(Q36_N_EMBD*4u);
    if (!s->route_stage) return;
    if (e->variant == Q36_VARIANT_FLASH_NEXT) {
#ifndef Q36_METAL
        uint64_t stride;
        if (!tensor_nbytes(router->type,Q36_N_EMBD,&stride) ||
            !q36_gpu_stream_matvec(s->route_stage,input,e->model.fd,router->abs_offset,
                router->type,Q36_N_EMBD,Q36_N_EXPERT,stride)) return;
#else
        return;
#endif
    } else {
        /* Reuse resident norms and routers. Only the current FFN delta and
         * the next attention delta are unknown at this point. */
        if (!residual || !next->post_attention_norm) return;
        if (!s->route_input) s->route_input = q36_gpu_tensor_alloc_scratch(Q36_N_EMBD*4u);
        if (!s->route_input || !q36_gpu_rms_norm_weight_rows_tensor(
                s->route_input,residual,e->model.map,e->model.size,
                next->post_attention_norm->abs_offset,Q36_N_EMBD,1,Q36_RMS_EPS) ||
            !q36_gpu_tensor_matmul_scaled(&e->model,router,s->route_input,s->route_stage,
                Q36_N_EMBD,Q36_N_EXPERT,1,1.0f)) return;
    }
    const float *logits = q36_gpu_tensor_contents(s->route_stage);
    if (!logits) return;
    float values[4];
    q36_topk_indices(logits,Q36_N_EXPERT,4,s->route_ids,values);
    s->route_layer = il + 1;
    s->route_pending = true;
#if defined(POSIX_FADV_WILLNEED) && !defined(Q36_METAL)
    const char *hint = getenv("Q36_SSD_ROUTE_PREFETCH");
    if (!hint || atoi(hint) <= 0) return;
    const q36_tensor *weights[] = {next->ffn_gate_exps,next->ffn_up_exps,next->ffn_down_exps};
    for (uint32_t j = 0; j < 3; j++)
        if (!weights[j] || weights[j]->bytes % Q36_N_EXPERT) return;
    q36_gpu_stream_expert_table table = {.model_map=e->model.map,.model_size=e->model.size,
        .layer=il+1,.n_total_expert=Q36_N_EXPERT,
        .gate_offset=weights[0]->abs_offset,.up_offset=weights[1]->abs_offset,.down_offset=weights[2]->abs_offset,
        .gate_expert_bytes=weights[0]->bytes/Q36_N_EXPERT,
        .up_expert_bytes=weights[1]->bytes/Q36_N_EXPERT,.down_expert_bytes=weights[2]->bytes/Q36_N_EXPERT};
    if (!e->route_reader) e->route_reader = q36_ssd_prefetch_create(e->model.fd);
    s->route_hint_bytes += q36_gpu_stream_hint_experts(&table,e->route_reader,s->route_ids,4);
#endif
}
