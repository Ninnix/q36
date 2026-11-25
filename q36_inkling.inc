#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize ("fp-contract=off")
#endif
/* Each virtual layer owns its KV and four causal convolutions. Sliding layers
 * reuse a 1024-token ring. Full layers grow with the live prefix. */
typedef struct {
    float history[INK_SLOTS][2*INK_K + 2*INK_D][3];
    uint8_t *kv[INK_SLOTS];
    uint32_t cap;
    float *scores;
} ink_state;

static void ink_reset(ink_state *st) {
    memset(st->history,0,sizeof(st->history));
}

static void ink_free(ink_state *st) {
    for (int i = 0; i < INK_SLOTS; i++) free(st->kv[i]);
    free(st->scores);
    free(st);
}

static bool ink_reserve(q36_session *s, uint32_t need) {
    ink_state *st = s->runtime;
    if (need <= st->cap) return true;
    unsigned cap = st->cap ? 2*st->cap : 8;
    if (cap < need) cap = need;
    if (cap > (unsigned)s->ctx_size) cap = s->ctx_size;
    unsigned row = ink_kv_row_bytes(s->engine->cache_type_k,s->engine->cache_type_v);
    for (int i = 0; i < INK_SLOTS; i++) {
        unsigned n = (i+1)%6 && cap > INK_WINDOW ? INK_WINDOW : cap;
        if (n <= st->cap) continue;
        st->kv[i] = xrealloc(st->kv[i],(size_t)n*row);
    }
    st->scores = xrealloc(st->scores,cap*sizeof(float));
    st->cap = cap;
    return true;
}

/* The score and weighted-state products have different matrix shapes.
 * Preserve both reduction orders through the next activation quantizer. */
static void ink_mix(float *out, const float *blocks, int count, const float *cur, const float *query, int queries) {
    float states[5][INK_D], scores[5];
    for (int j = 0; j <= count; j++) {
        const float *x = j == count ? cur : blocks + (count-1-j)*INK_D;
        q36_ref_rms_norm(states[j],x,NULL,INK_D,1e-6f);
        if (queries > 1 && count == 3) {
            float lanes[8] = {0};
            for (int i = 0; i < INK_D; i++) lanes[i%8] = fmaf(states[j][i],query[i],lanes[i%8]);
            scores[j] = q36_sum8(lanes);
        } else scores[j] = q36_dot_f32_strided(states[j],1,query,INK_D);
        scores[j] *= 1/sqrtf(INK_D);
    }
    q36_softmax_inplace(scores,count+1);
    for (int i = 0; i < INK_D; i++) {
        if (queries == 4) {
            float lanes[8] = {0};
            for (int j = 0; j <= count; j++) lanes[j] = states[j][i]*scores[j];
            out[i] = q36_sum8(lanes);
            continue;
        }
        float sum = 0;
        for (int j = 0; j <= count; j++) {
            if (count >= 3 && j < 4) sum += states[j][i]*scores[j];
            else sum = fmaf(states[j][i],scores[j],sum);
        }
        out[i] = sum;
    }
}

static void ink_mm(q36_engine *e, const q36_tensor *t, unsigned first, unsigned rows,
                   const float *x, float *out) {
    q36_tensor view = *t;
    uint64_t stride = t->bytes/(t->elements/t->dim[0]);
    view.ndim = 2;
    view.dim[1] = rows;
    view.elements = (uint64_t)rows*t->dim[0];
    view.abs_offset += first*stride;
    view.bytes = rows*stride;
    if (!q36_tensor_matvec(e,&view,x,out,NULL,view.dim[0],rows))
        q36_die("invalid Inkling projection");
}

static void ink_control(q36_engine *e, const ink_layer *l, int j, float *x, const float *msg) {
    float pair[2*INK_D], gate[64], gain[INK_D], forget[INK_D];
    q36_ref_rms_norm(pair,x,NULL,INK_D,1e-6f);
    q36_ref_rms_norm(pair+INK_D,msg,NULL,INK_D,1e-6f);
    ink_mm(e,l->ctrl_down[j],0,64,pair,gate);
    ink_mm(e,l->ctrl_gain[j],0,INK_D,gate,gain);
    ink_mm(e,l->ctrl_forget[j],0,INK_D,gate,forget);
    for (int i = 0; i < INK_D; i++)
        x[i] = (x[i]-tanhf(forget[i])*x[i]) + tanhf(gain[i])*msg[i];
}

static void ink_conv(float *x, unsigned n, float history[][3], const float *w, const float *dynamic) {
    float scale = 1/sqrtf(dynamic ? 9 : 5);
    for (unsigned i = 0; i < n; i++) {
        float *h = history[i];
        float now = x[i], v = 0;
        for (int j = 0; j < 3; j++) v += h[j]*w[4*i+j];
        v += now*w[4*i+3];
        if (dynamic) {
            const float *tap = dynamic + 4*(i/INK_HD);
            v += now*tap[0];
            v += h[2]*tap[1];
            v += h[1]*tap[2];
            v += h[0]*tap[3];
        }
        h[0] = h[1]; h[1] = h[2]; h[2] = now;
        x[i] = (now+v)*scale;
    }
}

static void ink_clip(float *x, const float *w, unsigned n, float limit, float tau) {
    if (w) for (unsigned i = 0; i < n; i++) x[i] *= w[i];
    double ss = 0;
    for (unsigned i = 0; i < n; i++) ss += (double)(x[i]*x[i]);
    float scale = 1/fmaxf(sqrtf((float)ss),limit);
    for (unsigned i = 0; i < n; i++) x[i] = ((x[i]*scale)*limit)*tau;
}

static void ink_attention(q36_session *s, const ink_layer *l, int slot, unsigned pos,
                          float in[3][INK_D], float *out) {
    q36_engine *e = s->engine;
    ink_state *st = s->runtime;
    bool full = l->dynamic[0] != NULL;
    unsigned cap = full || st->cap < INK_WINDOW ? st->cap : INK_WINDOW;
    unsigned kb = q36_kv_cache_row_bytes(e->cache_type_k,INK_HD);
    unsigned vb = q36_kv_cache_row_bytes(e->cache_type_v,INK_HD);
    unsigned row = INK_KH*(kb+vb);
    uint8_t *kv = st->kv[slot], *now = kv + (size_t)(pos%cap)*row;
    float (*history)[3] = st->history[slot];
    float q[INK_Q], k[INK_K], v[INK_K], rel[INK_H*INK_REL], sinks[INK_H], gate[INK_Q], ctx[INK_Q];
    float dynamic[2][4*INK_KH];
    ink_mm(e,l->q,0,INK_Q,in[0],q);
    ink_mm(e,l->k,0,INK_K,in[1],k);
    ink_mm(e,l->v,0,INK_K,in[2],v);
    ink_mm(e,l->gate,0,INK_Q,in[0],gate);
    ink_mm(e,l->rel,0,INK_H*INK_REL,in[0],rel);
    ink_mm(e,l->sink,0,INK_H,in[0],sinks);
    if (full) for (int i = 0; i < 2; i++) ink_mm(e,l->dynamic[i],0,32,in[i+1],dynamic[i]);
    ink_conv(k,INK_K,history,l->conv[0],full ? dynamic[0] : NULL);
    ink_conv(v,INK_K,history+INK_K,l->conv[1],full ? dynamic[1] : NULL);
    for (int h = 0; h < INK_K; h += INK_HD) {
        ink_clip(k+h,NULL,INK_HD,22.62741661f,1);
        q36_ref_rms_norm(v+h,v+h,NULL,INK_HD,1e-6f);
    }
    q36_kv_cache_store_row(now,e->cache_type_k,k,INK_K);
    q36_kv_cache_store_row(now+INK_KH*kb,e->cache_type_v,v,INK_K);
    unsigned start = !full && pos >= INK_WINDOW ? pos+1-INK_WINDOW : 0;
    float tau = full ? 1 + 0.1f*logf(fmaxf(1,(pos+1)/8192.0f)) : 1;
    for (int h = 0; h < INK_H; h++) {
        float *qh = q+h*INK_HD, *rh = rel+h*INK_REL, *ch = ctx+h*INK_HD;
        ink_clip(qh,l->clip_q+h*INK_HD,INK_HD,22.62741661f,tau);
        ink_clip(rh,l->clip_rel+h*INK_REL,INK_REL,8,tau);
        float top = 0;
        for (unsigned t = start; t <= pos; t++) {
            const uint8_t *key = kv + (size_t)(t%cap)*row + (h/3)*kb;
            float dot = q36_kv_dot(key,e->cache_type_k,qh,INK_HD);
            unsigned d = pos-t;
            float bias = 0;
            if (d < INK_WINDOW) for (int j = 0; j < INK_REL; j++) bias += rh[j]*l->bias_rel[d*INK_REL+j];
            dot = dot*(1/sqrtf(INK_HD)) + (bias-sinks[h]);
            st->scores[t-start] = dot;
            top = fmaxf(top,dot);
        }
        double sum = 0;
        for (unsigned t = start; t <= pos; t += 8) {
            float p[8] = {0};
            for (unsigned j = 0; j < 8 && t+j <= pos; j++) {
                p[j] = q36_vector_expf(st->scores[t+j-start]-top);
                st->scores[t+j-start] = p[j];
            }
            sum += q36_sum8(p);
        }
        sum += expf(-top);
        float inv = (float)(1.0/sum);
        float value_sum[32][INK_HD] = {{0}};
        for (unsigned t = start; t <= pos; t++) {
            const uint8_t *value = kv + (size_t)(t%cap)*row + INK_KH*kb + (h/3)*vb;
            float p = st->scores[t-start]*inv;
            float *acc = value_sum[t%32];
            for (int j = 0; j < INK_HD; j++) acc[j] = fmaf(p,q36_kv_cache_at(value,e->cache_type_v,j),acc[j]);
        }
        for (int j = 0; j < INK_HD; j++) {
            float lanes[8];
            for (int i = 0; i < 8; i++)
                lanes[i] = (value_sum[i][j]+value_sum[8+i][j]) + (value_sum[16+i][j]+value_sum[24+i][j]);
            ch[j] = q36_sum8(lanes);
        }
    }
    for (int i = 0; i < INK_Q; i++) ctx[i] *= q36_sigmoidf(gate[i]);
    ink_mm(e,l->out,0,INK_D,ctx,out);
    ink_conv(out,INK_D,history+2*INK_K,l->conv[2],NULL);
}

static void ink_route(const ink_layer *l, float *prob, uint32_t *ids, float *weights) {
    float score[128];
    for (int i = 0; i < 128; i++) {
        prob[i] = 1/(1+expf(-prob[i]));
        score[i] = prob[i]+l->bias[i];
    }
    q36_topk_indices(score,128,8,ids,weights);
    double total = 0;
    for (int i = 0; i < 8; i++) total += prob[ids[i]];
    float sum = (float)total + 1e-20f;
    for (int i = 0; i < 8; i++) weights[i] = (prob[ids[i]]/sum)*8;
}

static void ink_moe(q36_session *s, const ink_layer *l, int slot, const float *x, float *out) {
    uint32_t ids[8];
    float prob[128], weights[8];
    ink_mm(s->engine,l->router,0,128,x,prob);
    ink_route(l,prob,ids,weights);
    memset(out,0,INK_D*sizeof(float));
    for (int j = 0; j < 8; j++) {
        int id = ids[j];
        float gu[2*INK_F], down[INK_D];
        ink_mm(s->engine,l->gate_up,id*2*INK_F,2*INK_F,x,gu);
        for (int i = 0; i < INK_F; i++) {
            float scale = l->gate_scale ? l->gate_scale[id] : 1;
            float g = gu[i]*scale, u = gu[INK_F+i]*scale;
            gu[i] = (4*tanhf(g*0.25f)*q36_sigmoidf(g)) * (25*tanhf(u*0.04f));
        }
        ink_mm(s->engine,l->down,id*INK_D,INK_D,gu,down);
        float scale = l->down_scale ? l->down_scale[id] : 1;
        float weight = weights[j];
        for (int i = 0; i < INK_D; i++) out[i] += (down[i]*scale)*weight;
    }
    ink_state *st = s->runtime;
    ink_conv(out,INK_D,st->history[slot]+2*INK_K+INK_D,l->conv[3],NULL);
}

static unsigned ink_physical_layer(int slot) {
    return slot < 18 ? slot : slot < 42 ? 6+(slot-18)%12 : slot-24;
}

static bool ink_forward(q36_session *s, int token, unsigned pos, bool logits) {
    ink_weights *w = s->engine->ink;
    float cur[INK_D], prelude[INK_D], bank[2][INK_D], blocks[4][INK_D];
    if (token < 0 || token >= INK_VOCAB ||
        !q36_tensor_row_to_float(&s->engine->model,w->embd,token,cur,INK_D)) return false;
    q36_ref_rms_norm(cur,cur,w->embd_norm,INK_D,1e-6f);
    int count = 0;
    for (int slot = 0; slot < INK_SLOTS; slot++) {
        int il = ink_physical_layer(slot);
        ink_layer *l = &w->layer[il];
        bool repeat = slot >= 18 && slot < 42;
        if (slot == 6) memcpy(prelude,cur,sizeof(cur));
        if (repeat && il == 6) {
            /* Block 1 is the prelude, shared by every pass. Only the second
             * core block and the previous output carry messages forward. */
            memcpy(bank[0],blocks[2],sizeof(cur));
            memcpy(bank[1],cur,sizeof(cur));
            memcpy(cur,prelude,sizeof(cur));
            count = 1;
        }
        float in[3][INK_D], msg[4][INK_D], attn[INK_D], ffn[INK_D];
        if (repeat) for (int j = 0; j < 4; j++)
            ink_mix(msg[j],bank[0],1,bank[1],l->msg_query+j*INK_D,4);
        for (int j = 0; j < 3; j++) {
            if (!l->attn_query) memcpy(in[j],cur,sizeof(cur));
            else ink_mix(in[j],blocks[0],count,cur,l->attn_query+(l->dynamic[0] ? j*INK_D : 0),l->dynamic[0] ? 3 : 1);
            if (repeat) ink_control(s->engine,l,j,in[j],msg[j]);
            q36_ref_rms_norm(in[j],in[j],l->norm[j],INK_D,1e-6f);
        }
        ink_attention(s,l,slot,pos,in,attn);
        if (il%6 == 0) memcpy(blocks[count++],cur,sizeof(cur));
        else for (int i = 0; i < INK_D; i++) attn[i] += cur[i];
        ink_mix(in[0],blocks[0],count,attn,l->ffn_query,1);
        if (repeat) ink_control(s->engine,l,3,in[0],msg[3]);
        q36_ref_rms_norm(in[0],in[0],l->ffn_norm,INK_D,1e-6f);
        ink_moe(s,l,slot,in[0],ffn);
        for (int i = 0; i < INK_D; i++) cur[i] = attn[i]+ffn[i];
    }
    if (logits) {
        float x[INK_D];
        ink_mix(x,blocks[0],count,cur,w->output_query,1);
        q36_ref_rms_norm(x,x,w->output_norm,INK_D,1e-6f);
        ink_mm(s->engine,w->output,0,INK_VOCAB,x,s->logits);
        for (int i = 0; i < INK_VOCAB; i++) s->logits[i] = 30*tanhf(s->logits[i]*(1.0f/30));
    }
    s->logits_host_valid = logits;
    return true;
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
