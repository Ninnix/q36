/* Layer-major execution shares projections while recurrence and PLE retain
 * token order. The same kernels and buffers also serve single-token decode. */
static bool q36_q4_gpu_ple_batch(q36_q4_gpu *g, const q36_model *m,
    const q36_layer_weights *l, q36_q4_state *st, const int *tokens) {
    float *input=q36_gpu_tensor_contents_named(g->base[Q4_X],"flash_ple_gather");
    if (!input) return false;
    for (uint32_t t=0; t<g->tokens; t++) {
        uint32_t rows[16]; float emb[2560];
        q36_q4_ple_step(tokens[t],st->ple_prev,rows);
        if (!q36_q4_ngram_read(m,rows,16,emb)) return false;
        memcpy(input+(uint64_t)t*16384,emb,sizeof(emb));
    }
    q36_q4_gpu_row(g,0); g->batch=true;
    if (!Q4_MM(l->ple_key,Q4_TMP,Q4_X) || !Q4_MM(l->ple_value,Q4_BLOCK,Q4_X) ||
        !q36_q4_gpu_norm(g,m,l->ple_norm_key,g->v[Q4_TMP2],g->v[Q4_TMP],2560,4,false) ||
        !q36_q4_gpu_norm(g,m,l->ple_norm_query,g->v[Q4_TMP3],g->v[Q4_R],2560,4,false) ||
        !Q4_OP(5,2560,4,Q4_TMP,Q4_TMP2,Q4_TMP3,Q4_BLOCK,Q4_TMP,Q4_TMP) ||
        !q36_q4_gpu_norm(g,m,l->ple_norm_conv,g->v[Q4_XN],g->v[Q4_TMP],2560,4,false) ||
        !q36_q4_gpu_weight(g,m,l->ple_conv)) return false;
    g->batch=false;
    for (uint32_t t=0; t<g->tokens; t++) {
        q36_q4_gpu_row(g,t);
        if (!q36_q4_gpu_op(g,6,10240,40,g->v[Q4_R],g->v[Q4_TMP],g->v[Q4_XN],g->ple_hist,g->weight,NULL,0,0,0,0)) return false;
    }
    return true;
}

static bool q36_q4_gpu_linear_batch(q36_q4_gpu *g, const q36_model *m,
                                    const q36_layer_weights *l, uint32_t il) {
    if (!q36_q4_gpu_prepare_input(g)) return false;
    bool projected=Q4_MM(l->attn_qkv,Q4_QKV,Q4_X) && Q4_MM(l->attn_gate,Q4_Z,Q4_X) &&
        Q4_MM(l->ssm_beta,Q4_BETA,Q4_X) && Q4_MM(l->ssm_alpha,Q4_ALPHA,Q4_X);
    g->input_q8=false;
    if (!projected) return false;
    q36_gpu_tensor *decay=q36_gpu_cached_tensor(m->map,m->size,l->ssm_a->abs_offset,192);
    q36_gpu_tensor *dt=q36_gpu_cached_tensor(m->map,m->size,l->ssm_dt->abs_offset,192);
    if (!decay || !dt || !q36_q4_gpu_op(g,9,48,1,g->v[Q4_GB],g->v[Q4_BETA],g->v[Q4_ALPHA],decay,dt,NULL,0,0,0,0) ||
        !q36_q4_gpu_weight(g,m,l->ssm_conv1d)) return false;
    g->batch=false;
    if (!q36_q4_gpu_op(g,7,10240,40,g->base[Q4_CONV],g->base[Q4_QKV],g->weight,g->hist[il],NULL,NULL,0,0,0,g->tokens)) return false;
    q36_q4_gpu_row(g,0); g->batch=true;
    if (!q36_q4_gpu_op(g,8,128,16,g->v[Q4_Q],g->v[Q4_K],g->v[Q4_CONV],NULL,NULL,NULL,0,0,0,1)) return false;
    q36_gpu_tensor *delta[6]={g->state[il],g->base[Q4_Q],g->base[Q4_K],g->base[Q4_CONV],g->base[Q4_GB],g->base[Q4_O]};
    if (!q36_gpu_qwen4_delta(delta,g->tokens)) return false;
    q36_q4_gpu_row(g,0); g->batch=true;
    return q36_q4_gpu_weight(g,m,l->ssm_norm) &&
        q36_q4_gpu_op(g,10,128,48,g->v[Q4_TMP],g->v[Q4_O],g->weight,g->v[Q4_Z],NULL,NULL,0,0,0,0) &&
        Q4_MM(l->ssm_out,Q4_BLOCK,Q4_TMP);
}

static bool q36_q4_gpu_attention_batch(q36_q4_gpu *g, const q36_model *m,
    const q36_layer_weights *l, uint32_t il, uint32_t pos) {
    if (!q36_q4_gpu_prepare_input(g)) return false;
    bool projected=Q4_MM(l->attn_q,Q4_QKV,Q4_X) && Q4_MM(l->attn_k,Q4_K,Q4_X) &&
        Q4_MM(l->attn_v,Q4_Z,Q4_X) && Q4_MM(l->indexer_k_proj,Q4_TMP2,Q4_X);
    g->input_q8=false;
    if (!projected ||
        !q36_q4_gpu_weight(g,m,l->attn_q_norm) ||
        !q36_q4_gpu_op(g,11,256,24,g->v[Q4_Q],g->v[Q4_QKV],g->weight,NULL,NULL,NULL,0,pos,0,0) ||
        !q36_q4_gpu_weight(g,m,l->attn_k_norm) ||
        !q36_q4_gpu_op(g,11,256,2,g->v[Q4_TMP],g->v[Q4_K],g->weight,NULL,NULL,NULL,0,pos,0,1)) return false;
    if (!q36_q4_gpu_op(g,25,512,2,g->k[il],g->v[Q4_TMP],NULL,NULL,NULL,NULL,0,pos,g->cap,0) ||
        !q36_q4_gpu_op(g,17,512,2,g->value[il],g->v[Q4_Z],NULL,NULL,NULL,NULL,0,pos,0,0) ||
        !q36_q4_gpu_op(g,17,128,1,g->index[il],g->v[Q4_TMP2],NULL,NULL,NULL,NULL,0,pos,0,0)) return false;
    uint32_t blocks=(pos+g->tokens)/4;
    bool sparse=blocks>512;
    if (sparse) {
        if (blocks>16384 || !Q4_MM(l->indexer_q_proj,Q4_TMP,Q4_X) ||
            !q36_q4_gpu_weight(g,m,l->indexer_q_norm) ||
            !q36_q4_gpu_op(g,11,128,4,g->v[Q4_Q],g->v[Q4_TMP],g->weight,NULL,NULL,NULL,0,pos,0,1) ||
            !q36_q4_gpu_weight(g,m,l->indexer_k_norm) ||
            !q36_q4_gpu_pool_index(g,il,pos,blocks) ||
            !q36_q4_gpu_op(g,16,128,blocks,g->scores,g->index[il],g->v[Q4_Q],g->weight,NULL,NULL,blocks,pos,0,g->cache_index)) return false;
    }
    if (!sparse || blocks<=2048) {
        uint32_t count=pos+g->tokens;
        if (!q36_q4_gpu_op(g,sparse ? 28 : 27,sparse ? blocks : count,sparse ? 1 : (count+255)/256,
            g->v[Q4_SEL],sparse ? g->scores : NULL,NULL,NULL,NULL,NULL,0,pos,0,g->radix_select)) return false;
    } else {
    float *score=sparse ? xmalloc(blocks*4) : NULL;
    bool *taken=sparse ? xmalloc(blocks*sizeof(bool)) : NULL;
    bool ok=true;
    for (uint32_t t=0; ok && t<g->tokens; t++) {
        uint32_t selected[2052], count=0, complete=(pos+t+1)/4;
        if (complete<=512) {
            for (uint32_t i=0; i<=pos+t; i++) selected[count++]=i;
        } else {
            memset(taken,0,complete*sizeof(bool));
            ok=q36_gpu_tensor_read(g->scores,(uint64_t)t*65536,score,complete*4);
            if (!ok) break;
            q36_q4_gpu_select_blocks(score,complete,taken);
            for (uint32_t i=0; i<complete; i++) if (taken[i])
                for (uint32_t j=0; j<4; j++) selected[count++]=i*4+j;
            for (uint32_t i=complete*4; i<=pos+t; i++) selected[count++]=i;
        }
        ok=q36_gpu_tensor_write(g->row[t][Q4_SEL],0,selected,count*4) &&
           q36_gpu_tensor_write(g->row[t][Q4_SEL],4096u*4,&count,4);
    }
    free(score); free(taken);
    if (!ok) return false;
    }
    if (sparse && (!q36_q4_gpu_weight(g,m,l->attn_q_norm) ||
        !q36_q4_gpu_op(g,11,256,24,g->v[Q4_Q],g->v[Q4_QKV],g->weight,NULL,NULL,NULL,0,pos,0,0))) return false;
    return q36_q4_gpu_op(g,12,256,24,g->v[Q4_O],g->v[Q4_Q],g->k[il],g->value[il],g->v[Q4_QKV],g->v[Q4_SEL],0,0,g->cap,0) &&
           Q4_MM(l->attn_output,Q4_BLOCK,Q4_O);
}

static bool q36_q4_gpu_moe_batch(q36_q4_gpu *g, const q36_model *m,
    const q36_layer_weights *l, q36_session *s, uint32_t il) {
    if (!Q4_MM(l->ffn_gate_inp,Q4_TMP,Q4_X)) return false;
    const char *router=getenv("Q36_VK_FLASH_ROUTER");
    bool gpu_router=!q36_route_enabled(s) && (!router || strcmp(router,"0"));
    if (gpu_router && !Q4_OP(23,Q36_N_EXPERT,1,Q4_SEL,Q4_TMP,Q4_SEL,Q4_SEL,Q4_SEL,Q4_SEL)) return false;
    for (uint32_t t=0; !gpu_router && t<g->tokens; t++) {
        float logits[512], weights[10]; uint32_t ids[10], meta[74]={0};
        if (!q36_gpu_tensor_read(g->row[t][Q4_TMP],0,logits,Q36_N_EXPERT*4)) return false;
        q36_topk_indices(logits,Q36_N_EXPERT,10,ids,weights);
        q36_softmax_inplace(weights,10);
        if (q36_route_enabled(s)) {
            q36_route_observe(s,il,ids,10);
            q36_route_prefetch(s,il,g->row[t][Q4_X],NULL);
        }
        memcpy(meta,ids,sizeof(ids)); memcpy(meta+64,weights,sizeof(weights));
        if (!q36_gpu_tensor_write(g->row[t][Q4_SEL],0,meta,sizeof(meta))) return false;
    }
    g->input_q8=q36_gpu_qwen4_quantize(g->base[Q4_Q],g->base[Q4_X],g->tokens);
    q36_gpu_tensor *t[7]={g->base[Q4_BLOCK],g->base[Q4_TMP],g->base[Q4_TMP2],g->base[Q4_TMP3],
        g->base[Q4_GATE],g->base[Q4_SEL],g->base[Q4_Q]};
    const q36_tensor *w[3]={l->ffn_gate_exps,l->ffn_up_exps,l->ffn_down_exps};
    uint64_t offsets[3], bytes[3]; uint32_t types[3];
    for (uint32_t i=0; i<3; i++) { offsets[i]=w[i]->abs_offset; bytes[i]=w[i]->bytes; types[i]=w[i]->type; }
    if (!g->input_q8 || !q36_gpu_qwen4_experts(t,m->map,m->size,offsets,bytes,types,g->tokens) ||
        !Q4_OP(22,2560,10,Q4_BLOCK,Q4_BLOCK,Q4_SEL,Q4_BLOCK,Q4_BLOCK,Q4_BLOCK)) return false;
    bool ok=Q4_MM(l->ffn_gate_shexp,Q4_TMP,Q4_X) && Q4_MM(l->ffn_up_shexp,Q4_TMP2,Q4_X) &&
        Q4_OP(13,640,3,Q4_TMP3,Q4_TMP,Q4_TMP2,Q4_TMP,Q4_TMP,Q4_TMP) &&
        Q4_MM(l->ffn_down_shexp,Q4_TMP,Q4_TMP3) && Q4_MM(l->ffn_gate_inp_shexp,Q4_TMP2,Q4_X) &&
        Q4_OP(15,2560,10,Q4_BLOCK,Q4_TMP,Q4_TMP2,Q4_TMP,Q4_TMP,Q4_TMP);
    g->input_q8=false;
    return ok;
}

static bool q36_q4_gpu_forward_batch(q36_session *s, const int *tokens, uint32_t n,
    uint32_t pos, bool logits, uint32_t first, uint32_t last, const float *input, float *output) {
    q36_q4_state *st=s->runtime;
    q36_q4_gpu *g=st->gpu;
    const q36_model *m=&s->engine->model;
    const q36_weights *w=&s->engine->weights;
    if (!g || !n || n>32 || !q36_q4_gpu_rows(g,n)) return false;
    g->tokens=n; g->batch=false; g->input_q8=false;
    q36_q4_gpu_row(g,0);
    if (n==1) return q36_q4_gpu_forward_slice(s,tokens[0],pos,logits,first,last,input,output);
    if (!g->resident || !q36_q4_gpu_reserve(g,st->cap,pos)) return false;
    float *mapped=q36_gpu_tensor_contents_named(g->base[input ? Q4_R : Q4_X],"flash_input_gather");
    if (!mapped) return false;
    for (uint32_t t=0; t<n; t++) {
        if (input) memcpy(mapped+(uint64_t)t*16384,input+(uint64_t)t*10240,10240u*4);
        else {
            float emb[2560]; q36_q4_row(m,w->token_embd,tokens[t],emb);
            memcpy(mapped+(uint64_t)t*16384,emb,sizeof(emb));
        }
    }
    if (!input) {
        q36_q4_gpu_row(g,0); g->batch=true;
        if (!Q4_OP(19,2560,40,Q4_R,Q4_X,Q4_X,Q4_X,Q4_X,Q4_X)) return false;
        g->batch=false;
    }
    for (uint32_t il=first; il<=last; il++) {
        const q36_layer_weights *l=w->layer+il;
        if (il==1 && !q36_q4_gpu_ple_batch(g,m,l,st,tokens)) return false;
        q36_q4_gpu_row(g,0); g->batch=true;
        if (!q36_q4_gpu_hc(g,m,l->hc_attn_norm,l->hc_attn_down,l->hc_attn_up,l->hc_attn_inject) ||
            !(q36_layer_is_full_attention(il) ? q36_q4_gpu_attention_batch(g,m,l,il,pos) : q36_q4_gpu_linear_batch(g,m,l,il)) ||
            !Q4_OP(4,2560,40,Q4_R,Q4_BLOCK,Q4_INJ,Q4_R,Q4_R,Q4_R) ||
            !q36_q4_gpu_hc(g,m,l->hc_ffn_norm,l->hc_ffn_down,l->hc_ffn_up,l->hc_ffn_inject) ||
            !q36_q4_gpu_moe_batch(g,m,l,s,il) || !Q4_OP(4,2560,40,Q4_R,Q4_BLOCK,Q4_INJ,Q4_R,Q4_R,Q4_R)) {
            fprintf(stderr,"q36: Flash Next batched Vulkan failed at layer %u\n",il);
            return false;
        }
        g->batch=false;
    }
    q36_q4_gpu_row(g,n-1);
    if (last==47 && (logits || q36_dist_enabled(&s->engine->distributed)) &&
        !q36_q4_gpu_hc(g,m,w->output_hc_norm,w->output_hc_down,w->output_hc_up,NULL)) return false;
    if (output) for (uint32_t t=0; t<n; t++)
        if (!q36_gpu_tensor_read(g->row[t][Q4_R],0,output+(uint64_t)t*10240,10240u*4)) return false;
    if (logits) {
        if (!Q4_MM(w->output,Q4_LOGITS,Q4_X) || !q36_gpu_tensor_read(g->v[Q4_LOGITS],0,s->logits,248320u*4)) return false;
        s->logits_host_valid=q36_all_finite(s->logits,248320);
        s->gpu_top2_valid=false;
        return s->logits_host_valid;
    }
    return q36_gpu_synchronize()!=0;
}
