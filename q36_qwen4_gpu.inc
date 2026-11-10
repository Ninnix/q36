#if !defined(Q36_NO_GPU) && !defined(Q36_METAL)
/* Host work is limited to disk gathers and sparse routing. Residuals, PLE
 * history, GDN state and attention caches remain Vulkan tensors. */
enum {
    Q4_R, Q4_XN, Q4_LO, Q4_GATE, Q4_INJ, Q4_X, Q4_BLOCK,
    Q4_QKV, Q4_CONV, Q4_Z, Q4_BETA, Q4_ALPHA, Q4_Q, Q4_K,
    Q4_GB, Q4_O, Q4_TMP, Q4_TMP2, Q4_TMP3, Q4_W, Q4_LOGITS, Q4_SEL, Q4_NBUF
};
typedef struct {
    q36_gpu_tensor *v[Q4_NBUF], *base[Q4_NBUF], *row[32][Q4_NBUF];
    q36_gpu_tensor *state[48], *hist[48], *k[48], *value[48], *index[48];
    q36_gpu_tensor *ple_hist, *scores, *weight;
    uint32_t cap, first, last, row_cap, tokens, batch_cap, score_stride;
    bool resident, input_q8, batch, cache_index, radix_select;
} q36_q4_gpu;

static uint32_t q36_q4_gpu_prefill_cap(const q36_engine *e) {
    const char *batch = getenv("Q36_VK_FLASH_BATCH");
    const char *experts = getenv("Q36_VK_FLASH_EXPERTS");
    if (e->backend != Q36_BACKEND_VULKAN || e->ssd_streaming ||
        !q36_dist_enabled(&e->distributed) || !q36_gpu_qwen4_native() ||
        (experts && !strcmp(experts,"0"))) return 1;
    uint32_t cap = batch ? (uint32_t)atoi(batch) : e->prefill_cap_override ? e->prefill_cap_override : 32;
    if (cap < 1) cap = 1;
    if (cap > 32) cap = 32;
    if (e->prefill_cap_override && cap > e->prefill_cap_override) cap=e->prefill_cap_override;
    return cap;
}

static uint32_t q36_q4_gpu_stride(uint32_t buffer) {
    return buffer == Q4_BLOCK ? 131072 : 65536;
}

static void q36_q4_gpu_row(q36_q4_gpu *g, uint32_t row) {
    for (uint32_t i = 0; i < Q4_NBUF; i++) g->v[i]=g->row[row][i];
}

static bool q36_q4_gpu_rows(q36_q4_gpu *g, uint32_t n) {
    if (n <= g->row_cap) return true;
    uint32_t old_cap=g->row_cap, cap=n<=8 ? 8 : n<=16 ? 16 : 32;
    g->row_cap=cap;
    for (uint32_t i=0; i<Q4_NBUF; i++) {
        if (i==Q4_W || i==Q4_LOGITS) {
            for (uint32_t t=1; t<cap; t++) g->row[t][i]=g->base[i];
            continue;
        }
        q36_gpu_tensor *base=q36_gpu_tensor_alloc_uninitialized((uint64_t)cap*q36_q4_gpu_stride(i));
        if (!base) return false;
        for (uint32_t t=0; t<old_cap; t++)
            if (g->row[t][i] && g->row[t][i]!=g->base[i]) q36_gpu_tensor_free(g->row[t][i]);
        q36_gpu_tensor_free(g->base[i]); g->base[i]=base;
        for (uint32_t t=0; t<cap; t++) {
            g->row[t][i]=q36_gpu_tensor_view(base,(uint64_t)t*q36_q4_gpu_stride(i),q36_q4_gpu_stride(i));
            if (!g->row[t][i]) return false;
        }
    }
    q36_q4_gpu_row(g,0);
    return true;
}

static bool q36_q4_gpu_op(q36_q4_gpu *g, uint32_t op, uint32_t n, uint32_t groups,
    q36_gpu_tensor *a, q36_gpu_tensor *b, q36_gpu_tensor *c,
    q36_gpu_tensor *d, q36_gpu_tensor *e, q36_gpu_tensor *f,
    uint32_t rows, uint32_t pos, uint32_t aux, uint32_t mode) {
    q36_gpu_tensor *t[6] = {a,b,c,d,e,f};
    uint32_t p[8] = {op,n,rows,pos,0,aux,0,mode};
    if (g && g->batch) {
        for (uint32_t j = 0; j < 6; j++) {
            if (t[j] == g->scores) {
                p[4] |= (1u<<j) | (1u<<(16+j));
                p[6]=g->score_stride/sizeof(float);
                continue;
            }
            for (uint32_t i = 0; i < Q4_NBUF; i++) if (i != Q4_W && i != Q4_LOGITS && t[j] == g->v[i]) {
                p[4] |= 1u<<j;
                if (i == Q4_BLOCK) p[4] |= 1u<<(j+8);
                t[j]=g->base[i]; break;
            }
        }
        return q36_gpu_qwen4_op_rows(t,p,groups,g->tokens) != 0;
    }
    return q36_gpu_qwen4_op(t,p,groups) != 0;
}

static bool q36_q4_gpu_weight(q36_q4_gpu *g, const q36_model *m, const q36_tensor *t) {
    if (!t || t->elements * sizeof(float) > q36_gpu_tensor_bytes(g->v[Q4_W])) return false;
    if (g->resident && t->type == Q36_TENSOR_F32) {
        g->weight = q36_gpu_cached_tensor(m->map,m->size,t->abs_offset,t->bytes);
        return g->weight != NULL;
    }
    g->weight = g->v[Q4_W];
    float *w = xmalloc(t->elements * sizeof(float));
    bool ok = q36_dequantize_row_from_ptr(t->type, m->map + t->abs_offset, w, t->elements) &&
              q36_gpu_tensor_write(g->v[Q4_W],0,w,t->elements * sizeof(float));
    free(w);
    return ok;
}

static bool q36_q4_gpu_norm(q36_q4_gpu *g, const q36_model *m, const q36_tensor *w,
                            q36_gpu_tensor *out, q36_gpu_tensor *x, uint32_t width,
                            uint32_t rows, bool shared) {
    return q36_q4_gpu_weight(g,m,w) &&
        q36_q4_gpu_op(g,0,width,rows,out,x,g->weight,NULL,NULL,NULL,0,0,0,shared);
}

static bool q36_q4_gpu_mm(q36_q4_gpu *g, const q36_model *m, const q36_tensor *t, uint64_t first,
    uint32_t rows, q36_gpu_tensor *out, q36_gpu_tensor *x) {
    uint64_t stride;
    if (!t || !tensor_nbytes(t->type,t->dim[0],&stride)) return false;
    if (g->batch) {
        uint32_t a=Q4_NBUF,b=Q4_NBUF;
        for (uint32_t i = 0; i < Q4_NBUF; i++) { if (out == g->v[i]) a=i; if (x == g->v[i]) b=i; }
        if (a == Q4_NBUF || b == Q4_NBUF) return false;
        return q36_gpu_cached_matmul(g->base[a],g->base[b],m->map,m->size,t->abs_offset,t->bytes,
                first*stride,t->type,t->dim[0],rows,stride,g->tokens,q36_q4_gpu_stride(a),
                g->input_q8 && b == Q4_X ? g->base[Q4_Q] : NULL);
    }
    if (g->resident) return q36_gpu_cached_matvec(out,x,m->map,m->size,
        t->abs_offset,t->bytes,first*stride,t->type,t->dim[0],rows,stride,
        g->input_q8 && x == g->v[Q4_X] ? g->v[Q4_Q] : NULL);
    return q36_gpu_stream_matvec(out,x,m->fd,t->abs_offset + first*stride,
                                t->type,t->dim[0],rows,stride);
}
#define Q4_MM(t,out,in) q36_q4_gpu_mm(g,m,(t),0,(t)->ndim > 1 ? (t)->dim[1] : 1,g->v[out],g->v[in])
#define Q4_OP(op,n,groups,a,b,c,d,e,f) q36_q4_gpu_op(g,op,n,groups,g->v[a],g->v[b],g->v[c],g->v[d],g->v[e],g->v[f],0,0,0,0)

/* Q is dead until normalization. Share one packed input across independent
 * projections, just as the routed and shared experts do. */
static bool q36_q4_gpu_prepare_input(q36_q4_gpu *g) {
    g->input_q8=false;
    if (!g->resident || !q36_gpu_qwen4_native()) return true;
    g->input_q8=g->batch ? q36_gpu_qwen4_quantize(g->base[Q4_Q],g->base[Q4_X],g->tokens) :
        q36_gpu_quantize_q8_k_tensor(g->v[Q4_Q],g->v[Q4_X],2560,1);
    return g->input_q8;
}

static bool q36_q4_gpu_hc(q36_q4_gpu *g, const q36_model *m, const q36_tensor *norm,
    const q36_tensor *down, const q36_tensor *up, const q36_tensor *inject) {
    const char *setting=getenv("Q36_VK_FLASH_HC");
    bool fused=g->resident && !g->batch && inject && down->type==Q36_TENSOR_BF16 &&
        inject->type==Q36_TENSOR_BF16 && q36_gpu_qwen4_native() && (!setting || strcmp(setting,"0"));
    bool normalized=fused ? q36_q4_gpu_weight(g,m,norm) &&
        q36_q4_gpu_op(g,31,2560,4,g->v[Q4_XN],g->v[Q4_R],g->weight,g->v[Q4_TMP3],NULL,NULL,0,0,0,0) :
        q36_q4_gpu_norm(g,m,norm,g->v[Q4_XN],g->v[Q4_R],2560,4,false);
    if (!normalized) return false;
    bool projected=fused ? q36_gpu_qwen4_hc_down(g->v[Q4_LO],g->v[Q4_INJ],g->v[Q4_TMP3],m->map,m->size,
        down->abs_offset,down->bytes,inject->abs_offset,inject->bytes) : Q4_MM(down,Q4_LO,Q4_XN);
    if (!projected || !Q4_OP(1,320,2,Q4_LO,Q4_LO,Q4_LO,Q4_LO,Q4_LO,Q4_LO) ||
        !Q4_MM(up,Q4_GATE,Q4_LO) || !Q4_OP(2,2560,10,Q4_X,Q4_XN,Q4_GATE,Q4_X,Q4_X,Q4_X)) return false;
    return !inject || (fused ? Q4_OP(29,4,4,Q4_INJ,Q4_INJ,Q4_INJ,Q4_INJ,Q4_INJ,Q4_INJ) :
        Q4_MM(inject,Q4_INJ,Q4_XN) && Q4_OP(3,4,1,Q4_INJ,Q4_INJ,Q4_INJ,Q4_INJ,Q4_INJ,Q4_INJ));
}

static bool q36_q4_gpu_ple(q36_q4_gpu *g, const q36_model *m, const q36_layer_weights *l,
                          q36_q4_state *st, int token) {
    uint32_t rows[16];
    float emb[2560];
    q36_q4_ple_step(token,st->ple_prev,rows);
    if (!q36_q4_ngram_read(m,rows,16,emb) || !q36_gpu_tensor_write(g->v[Q4_X],0,emb,sizeof(emb)) ||
        !Q4_MM(l->ple_key,Q4_TMP,Q4_X) || !Q4_MM(l->ple_value,Q4_BLOCK,Q4_X) ||
        !q36_q4_gpu_norm(g,m,l->ple_norm_key,g->v[Q4_TMP2],g->v[Q4_TMP],2560,4,false) ||
        !q36_q4_gpu_norm(g,m,l->ple_norm_query,g->v[Q4_TMP3],g->v[Q4_R],2560,4,false) ||
        !Q4_OP(5,2560,4,Q4_TMP,Q4_TMP2,Q4_TMP3,Q4_BLOCK,Q4_TMP,Q4_TMP) ||
        !q36_q4_gpu_norm(g,m,l->ple_norm_conv,g->v[Q4_XN],g->v[Q4_TMP],2560,4,false) ||
        !q36_q4_gpu_weight(g,m,l->ple_conv)) return false;
    return q36_q4_gpu_op(g,6,10240,40,g->v[Q4_R],g->v[Q4_TMP],g->v[Q4_XN],g->ple_hist,g->weight,NULL,0,0,0,0);
}

static bool q36_q4_gpu_linear(q36_q4_gpu *g, const q36_model *m, const q36_layer_weights *l, uint32_t il) {
    if (!q36_q4_gpu_prepare_input(g)) return false;
    bool projected=Q4_MM(l->attn_qkv,Q4_QKV,Q4_X) && Q4_MM(l->attn_gate,Q4_Z,Q4_X) &&
        Q4_MM(l->ssm_beta,Q4_BETA,Q4_X) && Q4_MM(l->ssm_alpha,Q4_ALPHA,Q4_X);
    g->input_q8=false;
    if (!projected ||
        !q36_q4_gpu_weight(g,m,l->ssm_conv1d) ||
        !q36_q4_gpu_op(g,7,10240,40,g->v[Q4_CONV],g->v[Q4_QKV],g->weight,g->hist[il],NULL,NULL,0,0,0,0) ||
        !q36_q4_gpu_op(g,8,128,16,g->v[Q4_Q],g->v[Q4_K],g->v[Q4_CONV],NULL,NULL,NULL,0,0,0,1)) return false;
    /* The shared delta kernel scales Q; GB holds the decay and beta. */
    q36_gpu_tensor *decay, *dt;
    if (g->resident) {
        decay = q36_gpu_cached_tensor(m->map,m->size,l->ssm_a->abs_offset,192);
        dt = q36_gpu_cached_tensor(m->map,m->size,l->ssm_dt->abs_offset,192);
    } else {
        float controls[96];
        memcpy(controls,m->map + l->ssm_a->abs_offset,192);
        memcpy(controls + 48,m->map + l->ssm_dt->abs_offset,192);
        if (!q36_gpu_tensor_write(g->v[Q4_W],0,controls,sizeof(controls))) return false;
        decay = g->v[Q4_W];
        dt = q36_gpu_tensor_view(decay,192,192);
    }
    bool ok = decay && dt && q36_q4_gpu_op(g,9,48,1,g->v[Q4_GB],g->v[Q4_BETA],g->v[Q4_ALPHA],decay,dt,NULL,0,0,0,0);
    if (!g->resident) q36_gpu_tensor_free(dt);
    q36_gpu_tensor *v = q36_gpu_tensor_view(g->v[Q4_CONV],4096u * 4,6144u * 4);
    if (ok) ok = v && q36_gpu_delta_net_decode_tensor(g->state[il],g->v[Q4_Q],g->v[Q4_K],v,g->v[Q4_GB],g->v[Q4_O],48,128,1);
    q36_gpu_tensor_free(v);
    if (!ok || !q36_q4_gpu_weight(g,m,l->ssm_norm) ||
        !q36_q4_gpu_op(g,10,128,48,g->v[Q4_TMP],g->v[Q4_O],g->weight,g->v[Q4_Z],NULL,NULL,0,0,0,0)) return false;
    return Q4_MM(l->ssm_out,Q4_BLOCK,Q4_TMP);
}

static bool q36_q4_gpu_worse(const float *score, uint32_t a, uint32_t b) {
    return score[a] < score[b] || (score[a] == score[b] && a > b);
}

static void q36_q4_gpu_heap_down(const float *score, uint32_t heap[512], uint32_t i) {
    uint32_t value = heap[i];
    for (uint32_t child = i*2 + 1; child < 512; child = i*2 + 1) {
        if (child + 1 < 512 && q36_q4_gpu_worse(score,heap[child+1],heap[child])) child++;
        if (!q36_q4_gpu_worse(score,heap[child],value)) break;
        heap[i] = heap[child]; i = child;
    }
    heap[i] = value;
}

/* QSA needs a set, not sorted scores. Keep the worst retained block at the
 * root, preserving the reference's lower-index tie break in O(n log 512). */
static void q36_q4_gpu_select_blocks(const float *score, uint32_t n, bool *taken) {
    uint32_t heap[512];
    for (uint32_t i = 0; i < 512; i++) heap[i] = i;
    for (uint32_t i = 256; i > 0; i--) q36_q4_gpu_heap_down(score,heap,i-1);
    for (uint32_t i = 512; i < n; i++) if (q36_q4_gpu_worse(score,heap[0],i)) {
        heap[0] = i;
        q36_q4_gpu_heap_down(score,heap,0);
    }
    for (uint32_t i = 0; i < 512; i++) taken[heap[i]] = true;
}

static bool q36_q4_gpu_pool_index(q36_q4_gpu *g, uint32_t il, uint32_t pos, uint32_t blocks) {
    uint32_t first=pos/4;
    if (first<=512) first=0;
    if (!g->cache_index || blocks<=first) return true;
    /* Completed four-token blocks never change. Replace their first raw row
     * with the normalized, rotated key; partial blocks retain all raw rows. */
    return q36_q4_gpu_op(NULL,24,128,blocks-first,g->index[il],g->index[il],
            NULL,g->weight,NULL,NULL,blocks,0,first,0);
}

static bool q36_q4_gpu_attention(q36_q4_gpu *g, const q36_model *m, const q36_layer_weights *l,
                                 uint32_t il, uint32_t pos) {
    if (!q36_q4_gpu_prepare_input(g)) return false;
    bool projected=Q4_MM(l->attn_q,Q4_QKV,Q4_X) && Q4_MM(l->attn_k,Q4_K,Q4_X) && Q4_MM(l->attn_v,Q4_Z,Q4_X);
    g->input_q8=false;
    if (!projected || !q36_q4_gpu_weight(g,m,l->attn_q_norm) ||
        !q36_q4_gpu_op(g,11,256,24,g->v[Q4_Q],g->v[Q4_QKV],g->weight,NULL,NULL,NULL,0,pos,0,0) ||
        !q36_q4_gpu_weight(g,m,l->attn_k_norm) ||
        !q36_q4_gpu_op(g,11,256,2,g->v[Q4_TMP],g->v[Q4_K],g->weight,NULL,NULL,NULL,0,pos,0,1) ||
        !q36_q4_gpu_op(g,25,512,2,g->k[il],g->v[Q4_TMP],NULL,NULL,NULL,NULL,0,pos,g->cap,0) ||
        !q36_q4_gpu_op(g,17,512,2,g->value[il],g->v[Q4_Z],NULL,NULL,NULL,NULL,0,pos,0,0) ||
        !Q4_MM(l->indexer_k_proj,Q4_TMP,Q4_X) ||
        !q36_q4_gpu_op(g,17,128,1,g->index[il],g->v[Q4_TMP],NULL,NULL,NULL,NULL,0,pos,0,0)) return false;
    uint32_t selected[2052], count = 0, blocks = (pos + 1) / 4;
    if (blocks <= 512) {
        count=pos+1;
        if (!q36_q4_gpu_op(g,27,count,(count+255)/256,g->v[Q4_SEL],NULL,NULL,NULL,NULL,NULL,0,pos,0,0)) return false;
    } else {
        if (!Q4_MM(l->indexer_q_proj,Q4_TMP,Q4_X) || !q36_q4_gpu_weight(g,m,l->indexer_q_norm) ||
            !q36_q4_gpu_op(g,11,128,4,g->v[Q4_Q],g->v[Q4_TMP],g->weight,NULL,NULL,NULL,0,pos,0,1) ||
            !q36_q4_gpu_weight(g,m,l->indexer_k_norm) ||
            !q36_q4_gpu_pool_index(g,il,pos,blocks) ||
            !q36_q4_gpu_op(g,16,128,blocks,g->scores,g->index[il],g->v[Q4_Q],g->weight,NULL,NULL,blocks,0,0,g->cache_index)) return false;
        if (blocks<=2048) {
            count=2048+(pos+1)%4;
            if (!q36_q4_gpu_op(g,28,blocks,1,g->v[Q4_SEL],g->scores,NULL,NULL,NULL,NULL,0,pos,0,g->radix_select)) return false;
        } else {
        float *score = xmalloc(blocks * sizeof(float));
        bool *taken = xcalloc(blocks,sizeof(bool));
        bool ok = q36_gpu_tensor_read(g->scores,0,score,blocks * sizeof(float));
        if (ok) q36_q4_gpu_select_blocks(score,blocks,taken);
        if (ok) for (uint32_t j = 0; j < blocks; j++) if (taken[j]) for (uint32_t k = 0; k < 4; k++) selected[count++] = j * 4 + k;
        free(taken); free(score);
        if (!ok) return false;
        for (uint32_t j = blocks * 4; j <= pos; j++) selected[count++] = j;
        if (!q36_gpu_tensor_write(g->v[Q4_SEL],0,selected,count*sizeof(uint32_t))) return false;
        }
        /* QSA scoring reused the query scratch. Restore the attention query. */
        if (!q36_q4_gpu_weight(g,m,l->attn_q_norm) ||
            !q36_q4_gpu_op(g,11,256,24,g->v[Q4_Q],g->v[Q4_QKV],g->weight,NULL,NULL,NULL,0,pos,0,0)) return false;
    }
    return q36_q4_gpu_op(g,12,256,24,g->v[Q4_O],g->v[Q4_Q],g->k[il],g->value[il],g->v[Q4_QKV],g->v[Q4_SEL],count,0,g->cap,0) &&
        Q4_MM(l->attn_output,Q4_BLOCK,Q4_O);
}

static bool q36_q4_gpu_moe(q36_q4_gpu *g, const q36_model *m, const q36_layer_weights *l,
                           q36_session *s, uint32_t il) {
    float logits[512], value[10];
    uint32_t ids[10];
    const char *routed = getenv("Q36_VK_FLASH_EXPERTS");
    const char *router = getenv("Q36_VK_FLASH_ROUTER");
    bool gpu_router=g->resident && q36_gpu_qwen4_native() && !q36_route_enabled(s) &&
        (!routed || strcmp(routed,"0")) && (!router || strcmp(router,"0"));
    if (!Q4_MM(l->ffn_gate_inp,Q4_TMP,Q4_X)) return false;
    if (gpu_router) {
        if (!Q4_OP(23,Q36_N_EXPERT,1,Q4_SEL,Q4_TMP,Q4_SEL,Q4_SEL,Q4_SEL,Q4_SEL)) return false;
    } else {
    if (!q36_gpu_tensor_read(g->v[Q4_TMP],0,logits,Q36_N_EXPERT * sizeof(float))) return false;
    q36_topk_indices(logits,Q36_N_EXPERT,10,ids,value);
    q36_softmax_inplace(value,10);
    if (q36_route_enabled(s)) {
        q36_route_observe(s,il,ids,10);
        q36_route_prefetch(s,il,g->v[Q4_X],NULL);
    }
    }
    if (!Q4_OP(18,2560,10,Q4_BLOCK,Q4_BLOCK,Q4_BLOCK,Q4_BLOCK,Q4_BLOCK,Q4_BLOCK)) return false;
    /* Attention/GDN queries are dead here; reuse their buffer across experts. */
    g->input_q8 = g->resident && q36_gpu_quantize_q8_k_tensor(g->v[Q4_Q],g->v[Q4_X],2560,1);
    if (gpu_router && !g->input_q8) return false;
    if (g->resident && g->input_q8 && q36_gpu_qwen4_native() && (!routed || strcmp(routed,"0"))) {
        uint32_t meta[74] = {0};
        if (!gpu_router) { memcpy(meta,ids,sizeof(ids)); memcpy(meta+64,value,sizeof(value)); }
        q36_gpu_tensor *t[7] = {g->v[Q4_BLOCK],g->v[Q4_TMP],g->v[Q4_TMP2],g->v[Q4_TMP3],
                                g->v[Q4_GATE],g->v[Q4_SEL],g->v[Q4_Q]};
        const q36_tensor *w[3] = {l->ffn_gate_exps,l->ffn_up_exps,l->ffn_down_exps};
        uint64_t offsets[3],bytes[3]; uint32_t types[3];
        for (uint32_t i = 0; i < 3; i++) { offsets[i]=w[i]->abs_offset; bytes[i]=w[i]->bytes; types[i]=w[i]->type; }
        if ((!gpu_router && !q36_gpu_tensor_write(g->v[Q4_SEL],0,meta,sizeof(meta))) ||
            !q36_gpu_qwen4_experts(t,m->map,m->size,offsets,bytes,types,1) ||
            !Q4_OP(22,2560,10,Q4_BLOCK,Q4_BLOCK,Q4_SEL,Q4_BLOCK,Q4_BLOCK,Q4_BLOCK)) return false;
    } else {
    for (uint32_t i = 0; i < 10; i++) {
        if (!q36_q4_gpu_mm(g,m,l->ffn_gate_exps,(uint64_t)ids[i] * 640,640,g->v[Q4_TMP],g->v[Q4_X]) ||
            !q36_q4_gpu_mm(g,m,l->ffn_up_exps,(uint64_t)ids[i] * 640,640,g->v[Q4_TMP2],g->v[Q4_X]) ||
            !Q4_OP(13,640,3,Q4_TMP3,Q4_TMP,Q4_TMP2,Q4_TMP,Q4_TMP,Q4_TMP) ||
            !q36_q4_gpu_mm(g,m,l->ffn_down_exps,(uint64_t)ids[i] * 2560,2560,g->v[Q4_TMP],g->v[Q4_TMP3])) return false;
        uint32_t weight;
        memcpy(&weight,value + i,4);
        if (!q36_q4_gpu_op(g,14,2560,10,g->v[Q4_BLOCK],g->v[Q4_TMP],NULL,NULL,NULL,NULL,0,0,weight,0)) return false;
    }
    }
    bool ok = Q4_MM(l->ffn_gate_shexp,Q4_TMP,Q4_X) && Q4_MM(l->ffn_up_shexp,Q4_TMP2,Q4_X) &&
        Q4_OP(13,640,3,Q4_TMP3,Q4_TMP,Q4_TMP2,Q4_TMP,Q4_TMP,Q4_TMP) &&
        Q4_MM(l->ffn_down_shexp,Q4_TMP,Q4_TMP3) && Q4_MM(l->ffn_gate_inp_shexp,Q4_TMP2,Q4_X) &&
        Q4_OP(15,2560,10,Q4_BLOCK,Q4_TMP,Q4_TMP2,Q4_TMP,Q4_TMP,Q4_TMP);
    g->input_q8 = false;
    return ok;
}

static bool q36_q4_gpu_reserve(q36_q4_gpu *g, uint32_t cap, uint32_t rows) {
    if (cap <= g->cap) return true;
    for (uint32_t il = 3; il < 48; il += 4) {
        if (il < g->first || il > g->last) continue;
        q36_gpu_tensor **arrays[] = {&g->k[il],&g->value[il],&g->index[il]};
        uint32_t width[] = {512,512,128};
        for (uint32_t j = 0; j < 3; j++) {
            q36_gpu_tensor *t = q36_gpu_tensor_alloc_uninitialized((uint64_t)cap * width[j] * 4);
            bool copied=true;
            if (t && g->cap && rows) {
                if (j==0) {
                    q36_gpu_tensor *b[6]={t,*arrays[j],t,t,t,t};
                    uint32_t p[8]={26,512,rows,g->cap,0,cap};
                    copied=q36_gpu_qwen4_op(b,p,(512*rows+255)/256);
                } else copied=q36_gpu_tensor_copy(t,0,*arrays[j],0,(uint64_t)rows*width[j]*4);
            }
            if (!t || !copied) { q36_gpu_tensor_free(t); return false; }
            q36_gpu_tensor_free(*arrays[j]); *arrays[j] = t;
        }
    }
    q36_gpu_tensor_free(g->scores);
    g->score_stride=(cap/4+1)*sizeof(float);
    if (g->score_stride<65536) g->score_stride=65536;
    g->scores = q36_gpu_tensor_alloc_uninitialized((uint64_t)g->score_stride * (g->batch_cap ? g->batch_cap : 1));
    g->cap = cap;
    return g->scores != NULL;
}

static void q36_q4_gpu_free(q36_q4_gpu *g) {
    if (!g) return;
    for (uint32_t i = 0; i < Q4_NBUF; i++) {
        if (!g->base[i]) { q36_gpu_tensor_free(g->v[i]); continue; }
        for (uint32_t t = 0; t < g->row_cap; t++)
            if (g->row[t][i] && g->row[t][i] != g->base[i]) q36_gpu_tensor_free(g->row[t][i]);
        q36_gpu_tensor_free(g->base[i]);
    }
    for (uint32_t i = 0; i < 48; i++) {
        q36_gpu_tensor_free(g->state[i]); q36_gpu_tensor_free(g->hist[i]);
        q36_gpu_tensor_free(g->k[i]); q36_gpu_tensor_free(g->value[i]); q36_gpu_tensor_free(g->index[i]);
    }
    q36_gpu_tensor_free(g->ple_hist); q36_gpu_tensor_free(g->scores); free(g);
}

static bool q36_q4_gpu_init(q36_q4_state *st, const q36_engine *e) {
    q36_q4_gpu *g = xcalloc(1,sizeof(*g));
    st->gpu = g;
    g->first = e->layer_start;
    g->last = e->layer_end;
    g->resident = q36_dist_enabled(&e->distributed) && !e->ssd_streaming;
    const char *cache=getenv("Q36_VK_FLASH_QSA_CACHE"), *radix=getenv("Q36_VK_FLASH_QSA_RADIX");
    g->cache_index=!cache || strcmp(cache,"0");
    g->radix_select=!radix || strcmp(radix,"0");
    g->row_cap=g->tokens=1;
    g->batch_cap=q36_q4_gpu_prefill_cap(e);
    for (uint32_t i = 0; i < Q4_NBUF; i++) {
        uint64_t bytes = i == Q4_LOGITS ? 248320u * 4 : i == Q4_W ? 262144 : i == Q4_BLOCK ? 131072 : 65536;
        if (!(g->v[i] = q36_gpu_tensor_alloc(bytes))) return false;
        g->base[i]=g->row[0][i]=g->v[i];
    }
    if (g->first <= 1 && g->last >= 1) g->ple_hist = q36_gpu_tensor_alloc(9u * 10240 * 4);
    for (uint32_t il = g->first; il <= g->last; il++) if (!q36_layer_is_full_attention(il)) {
        g->state[il] = q36_gpu_tensor_alloc(48u * 128 * 128 * 4);
        g->hist[il] = q36_gpu_tensor_alloc(3u * 10240 * 4);
        if (!g->state[il] || !g->hist[il]) return false;
    }
    return (g->first > 1 || g->last < 1 || g->ple_hist) && q36_q4_gpu_reserve(g,st->cap,0);
}

static bool q36_q4_gpu_reset(q36_q4_state *st) {
    q36_q4_gpu *g=st->gpu;
    if (!g) return false;
    g->tokens=1; g->batch=false; g->input_q8=false;
    q36_q4_gpu_row(g,0);
    if (!q36_gpu_tensor_zero(g->v[Q4_R]) || !q36_gpu_tensor_zero(g->v[Q4_X]) ||
        !q36_gpu_tensor_zero(g->v[Q4_LOGITS])) return false;
    /* Attention and index rows are overwritten before reuse. Only recurrent
     * and PLE history need clearing; keep allocations and batch views. */
    for (uint32_t il=g->first; il<=g->last; il++) {
        if (g->state[il] && !q36_gpu_tensor_zero(g->state[il])) return false;
        if (g->hist[il] && !q36_gpu_tensor_zero(g->hist[il])) return false;
    }
    if (g->ple_hist && !q36_gpu_tensor_zero(g->ple_hist)) return false;
    for (uint32_t i=0; i<3; i++) st->ple_prev[i]=248044;
    memset(st->pos3,0,(uint64_t)st->cap*3*sizeof(uint32_t));
    memset(st->last_h,0,sizeof(st->last_h));
    st->mrope_delta=0;
    const char *cache=getenv("Q36_VK_FLASH_QSA_CACHE"), *radix=getenv("Q36_VK_FLASH_QSA_RADIX");
    g->cache_index=!cache || strcmp(cache,"0");
    g->radix_select=!radix || strcmp(radix,"0");
    return true;
}

static bool q36_q4_gpu_forward_slice(q36_session *s, int token, uint32_t pos, bool logits,
    uint32_t first, uint32_t last, const float *input, float *output) {
    q36_q4_state *st = s->runtime;
    q36_q4_gpu *g = st->gpu;
    const q36_model *m = &s->engine->model;
    const q36_weights *w = &s->engine->weights;
    float emb[2560];
    if (!g || !q36_q4_gpu_reserve(g,st->cap,pos)) return false;
    g->input_q8 = false;
    if (input) {
        if (!q36_gpu_tensor_write(g->v[Q4_R],0,input,10240u*4)) return false;
    } else {
        q36_q4_row(m,w->token_embd,token,emb);
        if (!q36_gpu_tensor_write(g->v[Q4_X],0,emb,sizeof(emb)) ||
            !Q4_OP(19,2560,40,Q4_R,Q4_X,Q4_X,Q4_X,Q4_X,Q4_X)) return false;
    }
    for (uint32_t il = first; il <= last; il++) {
        const q36_layer_weights *l = w->layer + il;
        if ((il == 1 && !q36_q4_gpu_ple(g,m,l,st,token)) ||
            !q36_q4_gpu_hc(g,m,l->hc_attn_norm,l->hc_attn_down,l->hc_attn_up,l->hc_attn_inject) ||
            !(q36_layer_is_full_attention(il) ? q36_q4_gpu_attention(g,m,l,il,pos) : q36_q4_gpu_linear(g,m,l,il)) ||
            !Q4_OP(4,2560,40,Q4_R,Q4_BLOCK,Q4_INJ,Q4_R,Q4_R,Q4_R) ||
            !q36_q4_gpu_hc(g,m,l->hc_ffn_norm,l->hc_ffn_down,l->hc_ffn_up,l->hc_ffn_inject) ||
            !q36_q4_gpu_moe(g,m,l,s,il) || !Q4_OP(4,2560,40,Q4_R,Q4_BLOCK,Q4_INJ,Q4_R,Q4_R,Q4_R)) {
            fprintf(stderr,"q36: Flash Next Vulkan failed at layer %u\n",il); return false;
        }
        if (getenv("Q36_Q4_DUMP_PREFIX")) {
            float row[10240];
            if (!q36_gpu_tensor_read(g->v[Q4_R],0,row,sizeof(row))) return false;
            q36_q4_dump("l_last",il,row,10240);
        }
    }
    if (last == 47 && (logits || q36_dist_enabled(&s->engine->distributed))) {
        if (!q36_q4_gpu_hc(g,m,w->output_hc_norm,w->output_hc_down,w->output_hc_up,NULL)) return false;
    }
    if (output && !q36_gpu_tensor_read(g->v[Q4_R],0,output,10240u*4)) return false;
    if (logits) {
        if (!Q4_MM(w->output,Q4_LOGITS,Q4_X) ||
            !q36_gpu_tensor_read(g->v[Q4_LOGITS],0,s->logits,248320u * 4)) return false;
        s->logits_host_valid = q36_all_finite(s->logits,248320);
        s->gpu_top2_valid = false;
        return s->logits_host_valid;
    }
    return q36_gpu_synchronize() != 0;
}

static bool q36_q4_gpu_forward(q36_session *s, int token, uint32_t pos, bool logits) {
    return q36_q4_gpu_forward_slice(s,token,pos,logits,0,47,NULL,NULL);
}
#include "q36_qwen4_batch.inc"
#undef Q4_MM
#undef Q4_OP
#endif
