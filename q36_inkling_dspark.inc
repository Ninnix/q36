/* Inkling's three-layer block drafter borrows the target embedding and output
 * projection. Weights belong to the engine. Attention history, target taps and
 * verification scratch belong to the session that produced them. */
enum {
    DS_MIX_TAPS, DS_ROTATE_QUERY, DS_STORE_KV, DS_ATTN_TILE, DS_ATTN_SUM,
    DS_SELECT_TILE, DS_RESTORE_HISTORY, DS_MARKOV_UPDATE=8, DS_MARKOV_EMBED, DS_SELECT_FINAL
};

enum {
    DS_CTX_MIX, DS_CTX_NORM, DS_ATTN_NORM, DS_QUERY, DS_KEY, DS_VALUE,
    DS_QUERY_NORM, DS_KEY_NORM, DS_CTX_KEY, DS_CTX_KEY_NORM, DS_CTX_VALUE,
    DS_ATTN_OUT, DS_FFN_NORM, DS_GATE, DS_UP, DS_DOWN, DS_LAYER_WEIGHTS
};
enum {
    DS_MARKOV_MIX, DS_MARKOV_NORM, DS_MARKOV_IN, DS_MARKOV_OUT, DS_MARKOV_BIAS,
    DS_JOINT_WEIGHT, DS_JOINT_BIAS, DS_MASK, DS_EMBED_NORM, DS_OUTPUT_NORM, DS_SHARED_WEIGHTS
};

enum { DS_VOCAB_CANDIDATES=61440, DS_VOCAB_RECENT=4096, DS_VOCAB_MAP_OFFSET=DS_VOCAB_CANDIDATES+DS_VOCAB_RECENT };
typedef struct ink_dspark_weights {
    uint8_t vocab_mask[(INK_VOCAB+7)/8];
    unsigned vocab_count;

    const q36_tensor *layer[3][DS_LAYER_WEIGHTS], *shared[DS_SHARED_WEIGHTS];
} ink_dspark_weights;

typedef struct { const char *name; unsigned width, rows, type; } ink_dspark_spec;

static bool ink_dspark_weights_bind(q36_model *m,int layer,const ink_dspark_spec *spec,
                                    const q36_tensor **out,unsigned count) {
    for (unsigned i=0;i<count;i++) {
        char name[96];
        if (layer>=0) snprintf(name,sizeof(name),"blk.%d.%s.weight",layer,spec[i].name);
        else snprintf(name,sizeof(name),"%s",spec[i].name);
        const q36_tensor *t=model_find_tensor(m,name);
        if (!t || t->type!=spec[i].type || t->ndim!=(spec[i].rows ? 2u : 1u) ||
            t->dim[0]!=spec[i].width || (spec[i].rows && t->dim[1]!=spec[i].rows)) {
            fprintf(stderr,"q36: incompatible DSpark tensor %s\n",name);
            return false;
        }
        out[i]=t;
    }
    return true;
}

typedef struct { float bias; uint32_t id; } ink_dspark_vocab_row;

static int ink_dspark_compare_bias(const void *a,const void *b) {
    const ink_dspark_vocab_row *x=a,*y=b;
    if (x->bias>y->bias) return -1;
    if (x->bias<y->bias) return 1;
    return (x->id>y->id)-(x->id<y->id);
}

/* The Markov bias supplies a model-derived vocabulary prior. The live prefix
 * fills in names, literals and other tokens specific to the request. */
static void ink_dspark_vocab_prepare(q36_engine *e) {
    ink_dspark_weights *w=e->dspark;
    if (e->mtp_margin>0) return;
    const float *bias=(const float *)(e->mtp_model.map+w->shared[DS_MARKOV_BIAS]->abs_offset);
    ink_dspark_vocab_row *rows=xmalloc(INK_VOCAB*sizeof(*rows));
    for (unsigned i=0;i<INK_VOCAB;i++)
        rows[i]=(ink_dspark_vocab_row){e->vocab.token[i].len<=1 ? INFINITY :
                                                     isfinite(bias[i]) ? bias[i] : -INFINITY,i};
    for (int i=0;i<e->vocab.n_special;i++) {
        unsigned id=e->vocab.special[i].id;
        if (id<INK_VOCAB) rows[id].bias=INFINITY;
    }
    qsort(rows,INK_VOCAB,sizeof(*rows),ink_dspark_compare_bias);
    for (unsigned i=0;i<DS_VOCAB_CANDIDATES;i++) {
        unsigned id=rows[i].id;
        w->vocab_mask[id/8]|=1u<<(id%8);
    }
    w->vocab_count=DS_VOCAB_CANDIDATES;
    free(rows);
}

static bool ink_dspark_bind(q36_engine *e,const char *path) {
    static const ink_dspark_spec layer[DS_LAYER_WEIGHTS]={
        {"ctx_mix",4,0,0},{"ctx_norm",INK_D,0,0},{"attn_norm",INK_D,0,0},
        {"attn_q",INK_D,INK_Q,8},{"attn_k",INK_D,INK_K,8},{"attn_v",INK_D,INK_K,8},
        {"attn_q_norm",128,0,0},{"attn_k_norm",128,0,0},{"attn_k_ctx",INK_D,INK_K,8},
        {"attn_k_ctx_norm",128,0,0},{"attn_v_ctx",INK_D,INK_K,8},{"attn_output",INK_Q,INK_D,8},
        {"ffn_norm",INK_D,0,0},{"ffn_gate",INK_D,9216,14},{"ffn_up",INK_D,9216,14},{"ffn_down",9216,INK_D,14}
    };
    static const ink_dspark_spec shared[DS_SHARED_WEIGHTS]={
        {"markov_ctx_mix.weight",4,0,0},{"markov_ctx_norm.weight",INK_D,0,0},
        {"markov_w1.weight",256,INK_VOCAB,14},{"markov_w2.weight",256,INK_VOCAB,14},
        {"markov_w2.bias",INK_VOCAB,0,0},{"joint_proj.weight",5120,768,8},{"joint_proj.bias",768,0,0},
        {"mask_embd.weight",INK_D,7,30},{"token_embd_norm.weight",INK_D,0,0},{"output_norm.weight",INK_D,0,0}
    };
    q36_model *m=&e->mtp_model;
    if (!path[0] || !q36_gpu_dspark_supported() || e->ink->output->type<12 ||
        e->ink->output->type>14 || access(path,R_OK)) return false;
    model_open(m,path,false);
    q36_str arch,tokenizer;
    q36_array_ref taps,window;
    static const struct { const char *key; unsigned value; } dimensions[]={
        {"dflash.block_count",3},{"dflash.block_size",8},{"dflash.embedding_length",INK_D},
        {"dflash.feed_forward_length",9216},{"dflash.attention.head_count",24},
        {"dflash.attention.head_count_kv",8},{"dflash.attention.key_length",128},
        {"dflash.attention.value_length",128},{"dflash.rope.dimension_count",128},
        {"dflash.attention.sliding_window",1025}
    };
    for (unsigned i=0;i<sizeof(dimensions)/sizeof(*dimensions);i++) {
        uint32_t n;
        if (!model_get_u32(m,dimensions[i].key,&n) || n!=dimensions[i].value) return false;
    }
    uint32_t indices[]={5,23,41,48};
    float freq,epsilon,softcap;
    bool anchor,confidence,markov;
    if (!model_get_string(m,"general.architecture",&arch) || !q36_streq(arch,"dflash") ||
        !model_get_string(m,"tokenizer.ggml.pre",&tokenizer) || !q36_streq(tokenizer,"inkling2") ||
        !model_get_f32_compat(m,"dflash.rope.freq_base",&freq) || freq!=1000000.f ||
        !model_get_f32_compat(m,"dflash.final_logit_softcapping",&softcap) || softcap!=30.f ||
        !model_get_f32_compat(m,"dflash.attention.layer_norm_rms_epsilon",&epsilon) || epsilon!=1e-6f ||
        !model_get_bool(m,"dflash.sample_from_anchor",&anchor) || anchor ||
        !model_get_bool(m,"dflash.has_confidence_head",&confidence) || confidence ||
        !model_get_bool(m,"dflash.has_markov_context",&markov) || !markov ||
        !model_get_array(m,"dflash.target_layers",&taps) || taps.len!=4 ||
        (taps.type!=GGUF_VALUE_UINT32 && taps.type!=GGUF_VALUE_INT32) ||
        memcmp(m->map+taps.data_pos,indices,sizeof(indices)) ||
        !model_get_array(m,"dflash.attention.sliding_window_pattern",&window) || window.len!=3 ||
        window.type!=GGUF_VALUE_BOOL || memcmp(m->map+window.data_pos,"\1\0\1",3) ||
        m->n_tensors!=3*DS_LAYER_WEIGHTS+DS_SHARED_WEIGHTS) return false;
    e->dspark=xcalloc(1,sizeof(*e->dspark));
    for (int i=0;i<3;i++)
        if (!ink_dspark_weights_bind(m,i,layer,e->dspark->layer[i],DS_LAYER_WEIGHTS)) return false;
    if (!ink_dspark_weights_bind(m,-1,shared,e->dspark->shared,DS_SHARED_WEIGHTS)) return false;
    ink_dspark_vocab_prepare(e);
    return true;
}

static bool ink_dspark_upload(q36_engine *e) {
    q36_model *m=&e->mtp_model;
    uint64_t sizes[3*DS_LAYER_WEIGHTS+DS_SHARED_WEIGHTS];
    unsigned count=0;
    const q36_tensor *mask=e->dspark->shared[DS_MASK];
    for (unsigned i=0;i<m->n_tensors;i++)
        if (&m->tensors[i]!=mask) sizes[count++]=m->tensors[i].bytes;
    if (!q36_gpu_plan_extra_weights(sizes,count)) return false;
    bool ok=true;
    for (unsigned i=0;i<m->n_tensors && ok;i++) {
        const q36_tensor *t=&m->tensors[i];
        if (t!=mask) ok=q36_gpu_cache_model_range(m->map,m->size,t->abs_offset,t->bytes,NULL);
    }
    if (!q36_gpu_finish_model_cache()) ok=false;
    q36_model_drop_pages(m);
    return ok;
}

enum {
    DS_CONTEXT, DS_NOISE, DS_NORM, DS_Q, DS_ROPE, DS_K, DS_V, DS_ATTN, DS_TEMP,
    DS_FF_GATE, DS_FF_UP, DS_BIAS, DS_TOP_TILES, DS_RESULT, DS_JOINT, DS_Z, DS_HIDDEN,
    DS_HISTORY, DS_QUANT, DS_VOCAB, DS_MASK_ROWS, DS_BUFFERS
};

typedef struct ink_dspark {
    q36_engine *engine;
    q36_gpu_tensor *buf[DS_BUFFERS], *mark, *kv[3], *taps, *mixed, *key, *value;
    q36_tokens flushes;
    uint64_t low_confidence_tokens, paused_tokens;
    unsigned panel_rows, cap, max_rows;
    unsigned pending, pending_pos, pause, pause_tokens, depth, lookup_skip;
    bool retry;
    q36_gpu_tensor *vocab_ids,*bias_scores;
    unsigned vocab_rows;
    int top[8][2];
} ink_dspark;

static q36_gpu_tensor *ink_dspark_alloc(uint64_t words) {
    return q36_gpu_tensor_alloc_uninitialized(words*sizeof(float));
}

static void ink_dspark_free(ink_dspark *d) {
    if (!d) return;
    if ((d->low_confidence_tokens || d->paused_tokens) && getenv("Q36_MTP_STATS"))
        fprintf(stderr,"q36: DSpark single-token steps low_confidence=%" PRIu64 " paused=%" PRIu64 "\n",
                d->low_confidence_tokens,d->paused_tokens);
    q36_gpu_tensor_free(d->mark);
    q36_gpu_tensor_free(d->vocab_ids);
    q36_gpu_tensor_free(d->bias_scores);
    for (unsigned i=0;i<DS_BUFFERS;i++) q36_gpu_tensor_free(d->buf[i]);
    for (unsigned i=0;i<3;i++) q36_gpu_tensor_free(d->kv[i]);
    q36_gpu_tensor_free(d->taps); q36_gpu_tensor_free(d->mixed);
    q36_gpu_tensor_free(d->key); q36_gpu_tensor_free(d->value);
    q36_tokens_free(&d->flushes);
    free(d);
}

static ink_dspark *ink_dspark_create(q36_engine *e,unsigned context) {
    static const unsigned words[DS_BUFFERS]={
        [DS_CONTEXT]=8*INK_D,[DS_NOISE]=8*INK_D,[DS_NORM]=8*INK_D,[DS_Q]=8*INK_Q,
        [DS_ROPE]=8*INK_Q,[DS_K]=8*INK_K,[DS_V]=8*INK_K,[DS_ATTN]=8*INK_Q,[DS_TEMP]=8*INK_D,
        [DS_FF_GATE]=8*9216,[DS_FF_UP]=8*9216,[DS_BIAS]=INK_VOCAB,[DS_TOP_TILES]=4*((INK_VOCAB+511)/512),
        [DS_RESULT]=24,[DS_JOINT]=5376,[DS_Z]=768,[DS_HIDDEN]=8*INK_D,
        [DS_HISTORY]=11*INK_SLOTS*(2*INK_K+2*INK_D),[DS_QUANT]=8*(9216/256)*74,
        [DS_VOCAB]=8*INK_PADDED,[DS_MASK_ROWS]=7*INK_D
    };
    ink_dspark *d=xcalloc(1,sizeof(*d));
    d->engine=e;
    d->pause_tokens=e->mtp_margin>0 ? INK_DSPARK_MAX_PAUSE : 0;
    const char *pause=getenv("Q36_DSPARK_PAUSE");
    if (pause && *pause) {
        char *end;
        long n=strtol(pause,&end,10);
        if (end!=pause && !*end && n>=0 && n<=INK_DSPARK_MAX_PAUSE) d->pause_tokens=(unsigned)n;
    }
    d->max_rows=context+8;
    for (unsigned i=0;i<DS_BUFFERS;i++) {
        unsigned n=i==DS_BIAS && e->dspark->vocab_count ? 2*DS_VOCAB_MAP_OFFSET : words[i];
        if (!(d->buf[i]=ink_dspark_alloc(n))) goto fail;
    }
    if (e->dspark->vocab_count) {
        uint64_t bytes=4ull*DS_VOCAB_MAP_OFFSET;
        d->bias_scores=q36_gpu_tensor_view(d->buf[DS_BIAS],0,bytes);
        d->vocab_ids=q36_gpu_tensor_view(d->buf[DS_BIAS],bytes,bytes);
        if (!d->bias_scores || !d->vocab_ids) goto fail;
        uint32_t *ids=q36_gpu_tensor_contents(d->vocab_ids);
        if (!ids) goto fail;
        for (unsigned id=0,n=0;id<INK_VOCAB;id++)
            if (e->dspark->vocab_mask[id/8]&(1u<<(id%8))) ids[n++]=id;
    }
    d->mark=q36_gpu_tensor_view(d->buf[DS_JOINT],5120*4,256*4);
    if (!d->mark) goto fail;
    for (int i=0;i<7;i++) {
        float row[INK_D];
        if (!q36_tensor_row_to_float(&e->mtp_model,e->dspark->shared[DS_MASK],i,row,INK_D) ||
            !q36_gpu_tensor_write(d->buf[DS_MASK_ROWS],i*sizeof(row),row,sizeof(row))) goto fail;
    }
    return d;
fail:
    ink_dspark_free(d);
    return NULL;
}

static unsigned ink_dspark_window(const ink_dspark *d,unsigned layer) {
    if (layer!=1) return 1024;
    return d->vocab_ids && d->max_rows>8192 ? 4096 : d->max_rows;
}

/* Reserve space beyond the window for the draft block's uncommitted rows. */
static unsigned ink_dspark_ring(const ink_dspark *d,unsigned il,unsigned cap) {
    unsigned rows=ink_dspark_window(d,il)+16;
    return cap<rows ? cap : rows;
}

static bool ink_dspark_reserve(ink_dspark *d,unsigned end) {
    if (end<=d->cap) return true;
    unsigned cap=(end+255u)&~255u;
    if (d->cap+d->cap/4>cap) cap=d->cap+d->cap/4;
    if (cap>d->max_rows) cap=d->max_rows;
    if (cap<end) return false;
    q36_gpu_tensor *next[3]={0};
    bool ok=true;
    for (unsigned i=0;i<3 && ok;i++) {
        uint64_t old=q36_gpu_tensor_bytes(d->kv[i]);
        uint64_t words=(uint64_t)ink_dspark_ring(d,i,cap)*INK_K;
        if (words*sizeof(float)<=old) continue;
        next[i]=ink_dspark_alloc(words);
        ok=next[i] && (!old || q36_gpu_tensor_copy(next[i],0,d->kv[i],0,old));
    }
    for (unsigned i=0;i<3;i++) {
        if (ok && next[i]) { q36_gpu_tensor_free(d->kv[i]); d->kv[i]=next[i]; }
        else q36_gpu_tensor_free(next[i]);
    }
    if (ok) d->cap=cap;
    return ok;
}

static void ink_dspark_trim_kv(ink_dspark *d,unsigned rows) {
    for (unsigned i=0;i<3;i++) {
        uint64_t bytes=4ull*ink_dspark_ring(d,i,rows+8)*INK_K;
        if (q36_gpu_tensor_bytes(d->kv[i])<=bytes) continue;
        q36_gpu_tensor_free(d->kv[i]);
        d->kv[i]=NULL;
    }
    d->cap=0;
}

static bool ink_dspark_tap(ink_dspark *d,unsigned slot,q36_gpu_tensor *x,unsigned nt) {
    unsigned rows=nt<16 ? 16 : nt;
    if (!slot && (rows>d->panel_rows || (rows==16 && d->panel_rows>=64))) {
        q36_gpu_tensor_free(d->taps); q36_gpu_tensor_free(d->mixed);
        q36_gpu_tensor_free(d->key); q36_gpu_tensor_free(d->value);
        d->taps=ink_dspark_alloc(4ull*rows*INK_D); d->mixed=ink_dspark_alloc((uint64_t)rows*INK_D);
        d->key=ink_dspark_alloc((uint64_t)rows*INK_K); d->value=ink_dspark_alloc((uint64_t)rows*INK_K);
        d->panel_rows=d->taps && d->mixed && d->key && d->value ? rows : 0;
    }
    return d->pending+nt<=d->panel_rows &&
        q36_gpu_tensor_copy(d->taps,4ull*(slot*d->panel_rows+d->pending)*INK_D,x,0,4ull*nt*INK_D);
}

static q36_gpu_tensor *ink_dspark_weight(ink_dspark *d,const q36_tensor *t) {
    q36_model *m=&d->engine->mtp_model;
    return q36_gpu_cached_tensor(m->map,m->size,t->abs_offset,t->bytes);
}

static bool ink_dspark_mm(ink_dspark *d,q36_gpu_tensor *out,const q36_tensor *t,
                          q36_gpu_tensor *in,unsigned nt) {
    if (nt>=2 && nt<=8 && t->type==Q36_TENSOR_Q6_K)
        return q36_gpu_quantize_q8_k_tensor(d->buf[DS_QUANT],in,t->dim[0],nt) &&
            q36_gpu_inkling_verify_matmul(out,d->buf[DS_QUANT],ink_dspark_weight(d,t),t->type,t->dim[0],t->dim[1],nt);
    return q36_gpu_tensor_matmul_scaled(&d->engine->mtp_model,t,in,out,t->dim[0],t->dim[1],nt,1);
}

static bool ink_dspark_norm(ink_dspark *d,q36_gpu_tensor *out,q36_gpu_tensor *in,const q36_tensor *t,unsigned n) {
    q36_model *m=&d->engine->mtp_model;
    return q36_gpu_rms_norm_weight_rows_tensor(out,in,m->map,m->size,t->abs_offset,t->dim[0],n,1e-6f);
}

static bool ink_dspark_gate_up(ink_dspark *d,const q36_tensor **layer,unsigned nt) {
    q36_gpu_tensor **b=d->buf;
    unsigned width=layer[DS_GATE]->dim[0],rows=layer[DS_GATE]->dim[1];
    return q36_gpu_quantize_q8_k_tensor(b[DS_QUANT],b[DS_NORM],width,nt) &&
        q36_gpu_inkling_verify_matmul(b[DS_FF_GATE],b[DS_QUANT],ink_dspark_weight(d,layer[DS_GATE]),
                                     Q36_TENSOR_Q6_K,width,rows,nt) &&
        q36_gpu_inkling_verify_matmul(b[DS_FF_UP],b[DS_QUANT],ink_dspark_weight(d,layer[DS_UP]),
                                     Q36_TENSOR_Q6_K,width,rows,nt) &&
        q36_gpu_swiglu_tensor(b[DS_FF_GATE],b[DS_FF_GATE],b[DS_FF_UP],nt*rows,0,1);
}

static bool ink_dspark_inject(ink_dspark *d,unsigned pos,unsigned nt) {
    if (!ink_dspark_reserve(d,pos+nt)) return false;
    ink_dspark_weights *w=d->engine->dspark;
    for (unsigned i=0;i<4;i++) {
        const q36_tensor *mix=i==3 ? w->shared[DS_MARKOV_MIX] : w->layer[i][DS_CTX_MIX];
        const q36_tensor *norm=i==3 ? w->shared[DS_MARKOV_NORM] : w->layer[i][DS_CTX_NORM];
        if (!q36_gpu_dspark_op(DS_MIX_TAPS,d->mixed,d->taps,ink_dspark_weight(d,mix),NULL,0,0,pos,d->panel_rows*INK_D,0,nt,36,1,nt) ||
            !ink_dspark_norm(d,d->mixed,d->mixed,norm,nt)) return false;
        if (i==3) {
            for (unsigned j=nt>8 ? nt-8 : 0;j<nt;j++)
                if (!q36_gpu_tensor_copy(d->buf[DS_CONTEXT],4ull*((pos+j)%8)*INK_D,d->mixed,4ull*j*INK_D,4*INK_D)) return false;
        } else {
            const q36_tensor **l=w->layer[i];
            if (!ink_dspark_mm(d,d->key,l[DS_CTX_KEY],d->mixed,nt) || !ink_dspark_mm(d,d->value,l[DS_CTX_VALUE],d->mixed,nt) ||
                !q36_gpu_dspark_op(DS_STORE_KV,d->kv[i],d->key,ink_dspark_weight(d,l[DS_CTX_KEY_NORM]),d->value,8,0,pos,
                                   ink_dspark_ring(d,i,d->cap),0,nt,8,1,nt)) return false;
        }
    }
    return true;
}

/* Serial decode can postpone feature projection until drafting resumes.
 * The tap panel lets one projection serve an entire pause. */
static bool ink_dspark_flush(ink_dspark *d,bool verifying) {
    if (!d->pending) return true;
    /* A short generated panel uses the verifier's Q8 reduction on replay
     * and when a new prompt follows a pause. */
    bool micro=!verifying && d->pending>=2 && d->pending<=8;
    if (micro) q36_gpu_set_micro_batch(true);
    bool ok=ink_dspark_inject(d,d->pending_pos,d->pending);
    if (micro) q36_gpu_set_micro_batch(false);
    if (!ok) return false;
    q36_tokens_push(&d->flushes,d->pending_pos+d->pending-1);
    d->pending=0;
    return true;
}

static bool ink_dspark_project(q36_session *s,q36_gpu_tensor *in,unsigned nt) {
    ink_dspark *d=((ink_vk *)s->runtime)->draft;
    const q36_tensor *t=s->engine->ink->output;
    q36_model *m=&s->engine->model;
    if (nt==1) return q36_gpu_cached_matvec(d->buf[DS_VOCAB],in,m->map,m->size,t->abs_offset,t->bytes,0,
                                            t->type,INK_D,INK_VOCAB,t->bytes/t->dim[1],NULL);
    return q36_gpu_quantize_q8_k_tensor(d->buf[DS_QUANT],in,INK_D,nt) &&
        q36_gpu_inkling_verify_matmul(d->buf[DS_VOCAB],d->buf[DS_QUANT],
            q36_gpu_cached_tensor(m->map,m->size,t->abs_offset,t->bytes),t->type,INK_D,INK_PADDED,nt);
}

static bool ink_dspark_vocab_recent(ink_dspark *d,const q36_tokens *ch,int anchor) {
    if (!d->vocab_ids) return true;
    unsigned len=0;
    uint32_t ids[DS_VOCAB_RECENT];
    uint8_t visited[sizeof(d->engine->dspark->vocab_mask)];
    const ink_dspark_weights *w=d->engine->dspark;
    memcpy(visited,w->vocab_mask,sizeof(visited));
    int start=ch->len>=DS_VOCAB_RECENT ? ch->len-DS_VOCAB_RECENT+1 : 0;
    for (int i=ch->len;i>=start;i--) {
        unsigned id=i==ch->len ? (unsigned)anchor : (unsigned)ch->v[i];
        if (id>=INK_VOCAB) continue;
        unsigned bit=1u<<(id%8);
        if (visited[id/8]&bit) continue;
        visited[id/8]|=bit;
        ids[len++]=id;
    }
    while (len%4) ids[len++]=UINT32_MAX;
    d->vocab_rows=w->vocab_count+len;
    return !len || q36_gpu_tensor_write(d->vocab_ids,4ull*w->vocab_count,ids,4ull*len);
}

static bool ink_dspark_vocab_project(ink_dspark *d,q36_gpu_tensor *out,q36_gpu_tensor *in,
         const q36_model *m,const q36_tensor *t,unsigned n) {
    unsigned width=t->dim[0];
    return q36_gpu_quantize_q8_k_tensor(d->buf[DS_QUANT],in,width,n) &&
           q36_gpu_inkling_draft_vocab(out,d->buf[DS_QUANT],
                   q36_gpu_cached_tensor(m->map,m->size,t->abs_offset,t->bytes),d->vocab_ids,
                   t->type,width,INK_PADDED,d->vocab_rows,n);
}

/* The anchor and active mask positions attend as one noncausal draft block. */
static bool ink_dspark_propose(q36_session *s,int *ids,int *length,int excluded,float threshold,bool probe,float *margins) {
    ink_vk *g=s->runtime;
    ink_dspark *d=g->draft;
    q36_engine *e=s->engine;
    q36_gpu_tensor **b=d->buf;
    const q36_tensor **w=e->dspark->shared;
    unsigned pos=s->checkpoint.len, count=*length;
    /* A second mask can inform the first proposal in the noncausal block. */
    unsigned width=count+(d->vocab_ids && count==2);
    float embedding[INK_D];
    if (!q36_gpu_tensor_write(b[DS_RESULT],0,ids,sizeof(*ids)) ||
        !ink_dspark_vocab_recent(d,&s->checkpoint,ids[0]) ||
        !ink_dspark_reserve(d,pos+width) ||
        !q36_tensor_row_to_float(&e->model,e->ink->embd,ids[0],embedding,INK_D) ||
        !q36_gpu_tensor_write(b[DS_NOISE],0,embedding,sizeof(embedding)) ||
        !ink_dspark_norm(d,b[DS_NOISE],b[DS_NOISE],w[DS_EMBED_NORM],1) ||
        !q36_gpu_tensor_copy(b[DS_NOISE],sizeof(embedding),b[DS_MASK_ROWS],0,(width-1)*sizeof(embedding))) return false;
    /* All draft attention reductions finish before the verifier uses scores. */
    unsigned end=g->cap>pos+width ? g->cap : pos+width;
    unsigned span=ink_dspark_window(d,1)+width;
    if (end>span) end=span;
    uint64_t bytes=4ull*width*24*((end+63)/64)*130;
    if (bytes>g->score_bytes && !ink_vk_resize_scores(g,bytes)) return false;
    for (unsigned i=0;i<3;i++) {
        const q36_tensor **l=e->dspark->layer[i];
        unsigned window=ink_dspark_window(d,i);
        if (window>pos) window=pos;
        unsigned tiles=(window+width+63)/64,ring=ink_dspark_ring(d,i,d->cap);
        if (!ink_dspark_norm(d,b[DS_NORM],b[DS_NOISE],l[DS_ATTN_NORM],width) ||
            !ink_dspark_mm(d,b[DS_Q],l[DS_QUERY],b[DS_NORM],width) ||
            !ink_dspark_mm(d,b[DS_K],l[DS_KEY],b[DS_NORM],width) ||
            !ink_dspark_mm(d,b[DS_V],l[DS_VALUE],b[DS_NORM],width) ||
            !q36_gpu_dspark_op(DS_ROTATE_QUERY,b[DS_ROPE],b[DS_Q],ink_dspark_weight(d,l[DS_QUERY_NORM]),NULL,24,0,pos,0,0,width,24,1,width) ||
            !q36_gpu_dspark_op(DS_STORE_KV,d->kv[i],b[DS_K],ink_dspark_weight(d,l[DS_KEY_NORM]),b[DS_V],8,0,pos,ring,0,width,8,1,width) ||
            !q36_gpu_dspark_op(DS_ATTN_TILE,g->scores,b[DS_ROPE],d->kv[i],NULL,0,tiles,pos,ring,window,width,tiles,24,width) ||
            !q36_gpu_dspark_op(DS_ATTN_SUM,b[DS_ATTN],g->scores,NULL,NULL,0,tiles,pos,0,window,width,1,24,width) ||
            !ink_dspark_mm(d,b[DS_TEMP],l[DS_ATTN_OUT],b[DS_ATTN],width) ||
            !q36_gpu_add_tensor(b[DS_NOISE],b[DS_NOISE],b[DS_TEMP],width*INK_D) ||
            !ink_dspark_norm(d,b[DS_NORM],b[DS_NOISE],l[DS_FFN_NORM],width) ||
            !ink_dspark_gate_up(d,l,width) ||
            !ink_dspark_mm(d,b[DS_TEMP],l[DS_DOWN],b[DS_FF_GATE],width) ||
            !q36_gpu_add_tensor(b[DS_NOISE],b[DS_NOISE],b[DS_TEMP],width*INK_D)) return false;
    }
    if (!ink_dspark_norm(d,b[DS_HIDDEN],b[DS_NOISE],w[DS_OUTPUT_NORM],count)) return false;
    q36_gpu_tensor *pred=q36_gpu_tensor_view(b[DS_HIDDEN],4*INK_D,4ull*(count-1)*INK_D);
    bool ok=pred && (d->vocab_ids ?
        ink_dspark_vocab_project(d,b[DS_VOCAB],pred,&e->model,e->ink->output,count-1) :
        ink_dspark_project(s,pred,count-1));
    q36_gpu_tensor_free(pred);
    if (!ok ||
        !q36_gpu_tensor_copy(b[DS_JOINT],4*2816,b[DS_CONTEXT],4ull*((pos-1)%8)*INK_D,4*INK_D)) return false;

    const q36_tensor *out=w[DS_MARKOV_OUT];
    unsigned vocab=d->vocab_ids ? d->vocab_rows : INK_VOCAB;
    unsigned tiles=(vocab+511)/512;
    q36_model *m=&e->mtp_model;
    for (unsigned i=1;i<count;i++) {
        q36_gpu_tensor *row=q36_gpu_tensor_view(b[DS_VOCAB],4ull*(i-1)*INK_PADDED,4ull*INK_VOCAB);
        ok=row && q36_gpu_dspark_op(DS_MARKOV_EMBED,b[DS_JOINT],b[DS_RESULT],ink_dspark_weight(d,w[DS_MARKOV_IN]),NULL,
                                    0,w[DS_MARKOV_IN]->type,i-1,excluded,0,1,4,1,1) &&
            q36_gpu_tensor_copy(b[DS_JOINT],4*512,b[DS_HIDDEN],4ull*i*INK_D,4*INK_D) &&
            ink_dspark_mm(d,b[DS_Z],w[DS_JOINT_WEIGHT],b[DS_JOINT],1) &&
            q36_gpu_dspark_op(DS_MARKOV_UPDATE,b[DS_JOINT],b[DS_Z],ink_dspark_weight(d,w[DS_JOINT_BIAS]),NULL,0,0,0,0,0,1,4,1,1) &&
            (d->vocab_ids ? ink_dspark_vocab_project(d,d->bias_scores,d->mark,m,out,1) :
             q36_gpu_cached_matvec(b[DS_BIAS],d->mark,m->map,m->size,out->abs_offset,out->bytes,0,out->type,
                                  256,INK_VOCAB,out->bytes/INK_VOCAB,NULL)) &&
            q36_gpu_dspark_op(DS_SELECT_TILE,b[DS_TOP_TILES],row,b[DS_BIAS],ink_dspark_weight(d,w[DS_MARKOV_BIAS]),
                              vocab,d->vocab_ids ? DS_VOCAB_MAP_OFFSET : 0,0,0,0,1,tiles,1,1) &&
            q36_gpu_dspark_op(DS_SELECT_FINAL,b[DS_RESULT],b[DS_TOP_TILES],NULL,NULL,tiles,0,i,0,0,1,1,1,1);
        q36_gpu_tensor_free(row);
        if (!ok) return false;
        /* A miss on the first proposal discards the whole tail. Check it
         * before running the remaining vocabulary corrections. */
        if (i==1 && count>2) {
            float margin;
            if (!q36_gpu_tensor_read(b[DS_RESULT],17*sizeof(float),&margin,sizeof(margin))) return false;
            if (probe && margin<threshold) {
                /* The anchor and first proposal are ready. Let the target
                 * check them before paying to score the remaining masks. */
                *length=count=2;
                break;
            }
            if (!probe && margin<1.5f*threshold) {
                *length=1;
                return true;
            }
        }
    }
    int best[24];
    if (!q36_gpu_tensor_read(b[DS_RESULT],0,best,sizeof(best))) return false;
    for (unsigned i=1;i<count;i++) {
        ids[i]=best[2*i]==excluded ? best[2*i+1] : best[2*i];
        if ((unsigned)ids[i]>=INK_VOCAB) return false;
    }
    memcpy(margins,best+16,8*sizeof(float));
    if (!probe && count==2 && margins[1]<1.5f*threshold) *length=1;
    return true;
}
