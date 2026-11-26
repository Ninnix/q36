#if !defined(Q36_NO_GPU) && !defined(Q36_METAL)
/* The 24 weight sets are resident once. Repeated layers have separate device
 * history and KV. Expert routing and causal state stay on the device across a batch. */
enum {
    IV_CUR, IV_PRELUDE, IV_BLOCKS, IV_BANK, IV_BANK_OUT, IV_MIX, IV_MSG,
    IV_IN_Q, IV_IN_K, IV_IN_V, IV_PAIR, IV_LOW, IV_GAIN, IV_FORGET,
    IV_Q, IV_K, IV_V, IV_REL, IV_SINK, IV_GATE, IV_DYNAMIC, IV_CONTEXT,
    IV_ATTN, IV_DENSE_PAIRS, IV_GU, IV_ACT, IV_QUANT, IV_ROUTER, IV_ROUTE, IV_LIVE, IV_MSG_SCALE, IV_PREP, IV_LOGITS, IV_COUNT
};
/* Attention projections die before the routed FFN. Their packed views share
 * the gate/up and activation buffers, including the earlier loop controls. */
static const struct { unsigned words, owner, offset; } ink_vk_layout[IV_LOGITS]={
    [IV_CUR]={INK_D}, [IV_PRELUDE]={INK_D}, [IV_BLOCKS]={4*INK_D}, [IV_BANK]={INK_D},
    [IV_BANK_OUT]={INK_D}, [IV_MIX]={5}, [IV_MSG]={4*INK_D}, [IV_MSG_SCALE]={4},
    [IV_IN_Q]={INK_D,IV_ACT,0}, [IV_IN_K]={INK_D,IV_ACT,INK_D}, [IV_IN_V]={INK_D,IV_ACT,2*INK_D},
    [IV_PAIR]={2*INK_D,IV_GU,0}, [IV_LOW]={64,IV_GU,4608}, [IV_GAIN]={INK_D,IV_GU,4672},
    [IV_FORGET]={INK_D,IV_GU,6976}, [IV_Q]={INK_Q,IV_GU,0}, [IV_K]={INK_K,IV_GU,3072},
    [IV_V]={INK_K,IV_GU,4096}, [IV_REL]={INK_REL*INK_H,IV_GU,5120}, [IV_SINK]={INK_H,IV_GU,5504},
    [IV_GATE]={INK_Q,IV_GU,5568}, [IV_DYNAMIC]={32,IV_GU,8640}, [IV_CONTEXT]={INK_Q,IV_GU,8704},
    [IV_ATTN]={INK_D}, [IV_DENSE_PAIRS]={INK_D,IV_GU,12640}, [IV_GU]={16*INK_F}, [IV_ACT]={8*INK_F},
    [IV_QUANT]={8*(INK_F/32)*9}, [IV_ROUTER]={128,IV_GU,0}, [IV_ROUTE]={16}, [IV_LIVE]={INK_D}, [IV_PREP]={INK_Q/32*9,IV_GU,11776}
};

static unsigned ink_vk_tiles(unsigned nt) {
    return 8*nt<nt+112 ? 8*nt : nt+112;
}

static uint64_t ink_vk_act_words(unsigned nt) {
    uint64_t words=(uint64_t)nt*ink_vk_layout[IV_ACT].words;
    // A tile holds eight scales and eight half-precision rows per block.
    // NVFP4 has the smallest blocks, so it bounds the gate and down layouts.
    uint64_t packed=(uint64_t)ink_vk_tiles(nt)*(INK_D/32)*136;
    return nt>=4 && packed>words ? packed : words;
}

/* Keep the score tile within 24 MiB, allowing more heads in a short final
 * batch. One KV head always needs three complete rows per query. */
enum { INK_SCORE_TILE = 32*65536 };
static unsigned ink_vk_head_tile(unsigned queries,unsigned end) {
    unsigned heads=INK_KH;
    while (heads>1 && (uint64_t)heads*queries*end>INK_SCORE_TILE) heads/=2;
    return heads;
}
static uint64_t ink_vk_score_words(unsigned nt,unsigned context,unsigned fmt) {
    uint64_t per_head=(uint64_t)(nt<32 ? nt : 32)*context, keys=per_head*INK_KH;
    if (nt>1 && q36_gpu_inkling_tiled(fmt) && keys>INK_SCORE_TILE)
        keys=per_head>INK_SCORE_TILE ? per_head : INK_SCORE_TILE;
    return (INK_H/INK_KH)*keys;
}
static uint64_t ink_vk_scratch_bytes(unsigned nt,unsigned context,unsigned fmt) {
    uint64_t words=1+9*(nt+112)+ink_vk_act_words(nt)-(uint64_t)nt*ink_vk_layout[IV_ACT].words;
    for (unsigned i=0;i<IV_LOGITS;i++)
        if (!ink_vk_layout[i].owner) words+=(uint64_t)nt*ink_vk_layout[i].words;
    if (nt<12) words+=(12-nt)*4*INK_D;
    if (nt*4<12*64) words+=12*64-nt*4;
    return 12ull*4*INK_D*4+4*(words+3ull*INK_VOCAB+ink_vk_score_words(nt,context,fmt));
}
typedef struct {
    q36_model *model;
    q36_gpu_tensor *buf[IV_COUNT], *history, *kv[INK_SLOTS], *scores;
    uint32_t cap, nt, batch, allocated, kv_format, kv_row_bytes;
    q36_gpu_tensor *cached[3], *queries;
    q36_gpu_tensor *messages[12], *message_scale[12], *pair[3], *low[3];
    struct ink_dspark *draft;
    bool verify, reuse, cached_k;
    uint64_t score_bytes;
} ink_vk;

static bool ink_vk_resize_scores(ink_vk *g,uint64_t bytes) {
    q36_gpu_tensor *scores=q36_gpu_tensor_alloc_uninitialized(bytes);
    if (!scores) return false;
    q36_gpu_tensor_free(g->scores);
    g->scores=scores;
    g->score_bytes=bytes;
    return true;
}

#include "q36_inkling_dspark.inc"

static uint32_t ink_bits(float f) {
    uint32_t u;
    memcpy(&u,&f,sizeof(u));
    return u;
}

static void ink_vk_free(ink_vk *g) {
    if (!g) return;
    for (unsigned i=0;i<3;i++) { q36_gpu_tensor_free(g->pair[i]); q36_gpu_tensor_free(g->low[i]); }
    for (int i=0;i<IV_COUNT;i++) q36_gpu_tensor_free(g->buf[i]);
    for (int i=0;i<INK_SLOTS;i++) q36_gpu_tensor_free(g->kv[i]);
    for (int i=0;i<12;i++) { q36_gpu_tensor_free(g->messages[i]); q36_gpu_tensor_free(g->message_scale[i]); }
    q36_gpu_tensor_free(g->queries);
    q36_gpu_tensor_free(g->history);
    q36_gpu_tensor_free(g->scores);
    ink_dspark_free(g->draft);
    free(g);
}

static ink_vk *ink_vk_create(q36_engine *e,unsigned context) {
    ink_vk *g=xcalloc(1,sizeof(*g));
    g->model=&e->model;
    g->kv_format=e->cache_type_k+3*e->cache_type_v;
    g->kv_row_bytes=ink_kv_row_bytes(e->cache_type_k,e->cache_type_v);
    g->batch=ink_vk_batch(e->prefill_cap_override);
    g->buf[IV_LOGITS]=q36_gpu_tensor_alloc_uninitialized(INK_VOCAB*4);
    if (!g->buf[IV_LOGITS]) goto fail;
    if (e->dspark && !(g->draft=ink_dspark_create(e,context))) goto fail;
    uint64_t bytes=INK_SLOTS*(2*INK_K+2*INK_D)*3u*4;
    g->history=g->draft ? q36_gpu_tensor_view(g->draft->buf[DS_HISTORY],0,bytes) :
                          q36_gpu_tensor_alloc_uninitialized(bytes);
    if (g->history) return g;
fail:
    ink_vk_free(g);
    return NULL;
}

/* All twelve message queries read the same two bank states during one
 * loop pass. Decode them together while the controller weights are idle. */
static bool ink_vk_message_views(ink_vk *g,const ink_weights *w) {
    const unsigned bytes=4*INK_D*sizeof(float);
    if (!g->queries) {
        g->queries=q36_gpu_tensor_alloc_uninitialized(12*bytes);
        if (!g->queries) return false;
        for (unsigned i=0;i<12;i++) {
            if (!q36_gpu_tensor_write(g->queries,i*bytes,w->layer[6+i].msg_query,bytes)) {
                q36_gpu_tensor_free(g->queries); g->queries=NULL; return false;
            }
        }
    }
    for (unsigned j=0;j<3;j++) {
        if (!g->pair[j]) g->pair[j]=q36_gpu_tensor_view(g->buf[IV_GU],j*2*INK_D*4,2*INK_D*4);
        if (!g->low[j]) g->low[j]=q36_gpu_tensor_view(g->buf[IV_QUANT],j*64*4,64*4);
        if (!g->pair[j] || !g->low[j]) return false;
    }
    for (unsigned i=0;i<12;i++) {
        if (!g->messages[i]) g->messages[i]=q36_gpu_tensor_view(g->buf[IV_MSG],i*bytes,bytes);
        if (!g->message_scale[i]) g->message_scale[i]=q36_gpu_tensor_view(g->buf[IV_MSG_SCALE],i*256,16);
        if (!g->messages[i] || !g->message_scale[i]) return false;
    }
    return true;
}

static bool ink_vk_resize_scratch(ink_vk *g, unsigned nt) {
    q36_gpu_tensor *buf[IV_LOGITS]={0};
    for (unsigned i=0;i<IV_LOGITS;i++) {
        if (ink_vk_layout[i].owner) continue;
        uint64_t words=(uint64_t)ink_vk_layout[i].words*nt;
        if (i==IV_MSG && nt<12) words=12*4*INK_D;
        if (i==IV_MSG_SCALE && words<12*64) words=12*64;
        if (i==IV_ROUTE) words+=1+9*(nt+112);
        if (i==IV_ACT) words=ink_vk_act_words(nt);
        if (!(buf[i]=q36_gpu_tensor_alloc_uninitialized(4*words))) goto fail;
    }
    for (unsigned i=0;i<IV_LOGITS;i++) {
        if (!ink_vk_layout[i].owner) continue;
        buf[i]=q36_gpu_tensor_view(buf[ink_vk_layout[i].owner],4ull*nt*ink_vk_layout[i].offset,
                                   4ull*nt*ink_vk_layout[i].words);
        if (!buf[i]) goto fail;
    }
    for (unsigned j=0;j<3;j++) {
        q36_gpu_tensor_free(g->pair[j]); g->pair[j]=NULL;
        q36_gpu_tensor_free(g->low[j]); g->low[j]=NULL;
    }
    for (unsigned i=0;i<12;i++) {
        q36_gpu_tensor_free(g->messages[i]); g->messages[i]=NULL;
        q36_gpu_tensor_free(g->message_scale[i]); g->message_scale[i]=NULL;
    }
    for (unsigned i=0;i<IV_LOGITS;i++) {
        q36_gpu_tensor_free(g->buf[i]);
        g->buf[i]=buf[i];
    }
    g->allocated=nt;
    return true;
fail:
    for (unsigned i=0;i<IV_LOGITS;i++) q36_gpu_tensor_free(buf[i]);
    return false;
}

static bool ink_vk_scratch(ink_vk *g, unsigned nt, unsigned end) {
    /* A single decode row needs none of the large prompt panels. Keep small
     * conversational batches warm, release large panels between turns. */
    bool trim=(nt==1 || g->verify) && g->allocated>=64;
    if (nt>g->allocated && !ink_vk_resize_scratch(g,nt)) return false;
    if (trim) (void)ink_vk_resize_scratch(g,nt);
    uint64_t bytes=4*ink_vk_score_words(nt,end,g->kv_format);
    if (bytes>g->score_bytes || (trim && g->score_bytes>4ull*INK_H*g->cap)) {
        uint64_t capacity=4*ink_vk_score_words(nt,g->cap,g->kv_format);
        if (!ink_vk_resize_scores(g,capacity) && bytes>g->score_bytes) return false;
    }
    g->nt=nt;
    return true;
}

static unsigned ink_vk_ring(const ink_vk *g, unsigned cap, bool full) {
    unsigned ring=INK_WINDOW+(g->batch<32 ? g->batch : 32)-1;
    return full || cap<ring ? cap : ring;
}

static bool ink_vk_reserve(q36_session *s, uint32_t need) {
    ink_vk *g=s->runtime;
    if (need<=g->cap) return true;
    unsigned cap=g->cap+g->cap/4;
    if (cap<need) cap=need;
    cap=(unsigned)ink_vk_rows(cap);
    if (cap>(unsigned)s->ctx_size) cap=s->ctx_size;
    for (int i=0;i<INK_SLOTS;i++) {
        unsigned n=ink_vk_ring(g,cap,(i+1)%6==0), old=ink_vk_ring(g,g->cap,(i+1)%6==0);
        if (ink_vk_rows(n)==ink_vk_rows(old)) continue;
        q36_gpu_tensor *kv=q36_gpu_tensor_alloc_uninitialized(ink_vk_rows(n)*g->kv_row_bytes);
        if (!kv) return false;
        if (g->kv[i] && !q36_gpu_tensor_copy(kv,0,g->kv[i],0,ink_vk_rows(old)*g->kv_row_bytes)) {
            q36_gpu_tensor_free(kv);
            return false;
        }
        q36_gpu_tensor_free(g->kv[i]);
        g->kv[i]=kv;
    }
    g->cap=cap;
    return true;
}

static q36_gpu_tensor *ink_vk_weight(ink_vk *g, const float *w, unsigned n) {
    if (!w) return NULL;
    q36_gpu_tensor *t=q36_gpu_cached_tensor(g->model->map,g->model->size,
        (const uint8_t *)w-g->model->map,(uint64_t)n*4);
    if (!t) q36_die("missing resident Inkling tensor");
    return t;
}

static bool ink_vk_op(ink_vk *g,unsigned op,unsigned groups,q36_gpu_tensor *a,q36_gpu_tensor *b,
        q36_gpu_tensor *c, q36_gpu_tensor *d, q36_gpu_tensor *e, q36_gpu_tensor *f,
        unsigned n, unsigned k, unsigned j, unsigned m, unsigned t) {
    q36_gpu_tensor *buf[6]={a,b,c,d,e,f};
    uint32_t p[8]={op,n,k,j,m,t,g->nt,groups};
    return q36_gpu_inkling_op(buf,p,g->kv_format);
}

static bool ink_vk_mul(ink_vk *g, const q36_tensor *t, unsigned mode, unsigned rows,
                      q36_gpu_tensor *x,q36_gpu_tensor *out,bool prepared) {
    unsigned width=t->dim[0], stride=t->bytes/(t->elements/width);
    if (rows==INK_VOCAB && t->type==Q36_TENSOR_Q5_K)
        return q36_gpu_cached_matvec(out,x,g->model->map,g->model->size,t->abs_offset,
            t->bytes,0,t->type,width,rows,stride,NULL);
    unsigned inputs=g->nt*(mode==2 ? 8 : 1);
    q36_gpu_tensor *w=q36_gpu_cached_tensor(g->model->map,g->model->size,t->abs_offset,t->bytes);
    if (!w) return false;
    q36_gpu_tensor *input=x;
    if (prepared) x=g->buf[IV_QUANT];
    else if (ink_packed_k(t->type)) {
        if ((!g->reuse || g->cached[0]!=x) && !q36_gpu_quantize_q8_k_tensor(g->buf[IV_QUANT],x,width,inputs)) return false;
        g->cached[0]=x;
        x=g->buf[IV_QUANT];
    } else if (t->type!=Q36_TENSOR_F32 && t->type!=Q36_TENSOR_BF16) {
        q36_gpu_tensor *buf[6]={g->buf[g->reuse ? IV_PREP : IV_QUANT],x};
        uint32_t p[8]={16,width,0,0,0,0,inputs,(width+127)/128};
        if ((!g->reuse || g->cached[1]!=x) && !q36_gpu_inkling_op(buf,p,0)) return false;
        g->cached[1]=x;
        x=buf[0];
    }
    if (g->verify && !mode && ((t->type>=12 && t->type<=14 && rows%4==0) ||
        (t->type==Q36_TENSOR_Q8_0 && rows>=128 && rows%8==0)))
        return q36_gpu_inkling_verify_matmul(out,x,w,t->type,width,rows,g->nt);
    bool iq=t->type==Q36_TENSOR_IQ2_XXS || t->type==Q36_TENSOR_IQ2_XS ||
            t->type==Q36_TENSOR_IQ3_XXS || t->type==Q36_TENSOR_IQ2_S;
    unsigned dense_tiles=(g->nt+7)/8;
    bool k=t->type>=Q36_TENSOR_Q4_K && t->type<=Q36_TENSOR_Q6_K;
    uint64_t pair_words=(uint64_t)dense_tiles*(k ? width/256*1096 : width/32*136);
    bool dense=!mode && (k || t->type==Q36_TENSOR_Q8_0) && pair_words<=(uint64_t)g->nt*INK_D;
    bool routed=mode && (iq || t->type==Q36_TENSOR_NVFP4);
    if ((!g->verify || !mode) && g->nt>=4 && rows%32==0 && (dense || routed) && q36_gpu_inkling_packed()) {
        q36_gpu_tensor *pairs=g->buf[mode ? IV_ACT : IV_DENSE_PAIRS];
        unsigned tiles=mode ? ink_vk_tiles(g->nt) : dense_tiles;
        unsigned pack=mode==1 && t->type==Q36_TENSOR_IQ3_XXS && width==INK_D && g->nt>=256 ? 35 : 26;
        if ((!dense || !g->reuse || g->cached[2]!=input || g->cached_k!=k) &&
            !ink_vk_op(g,pack,width/32,pairs,x,g->buf[IV_ROUTE],NULL,NULL,NULL,width,k ? 2 : iq,mode,tiles,0)) return false;
        if (dense) { g->cached[2]=input; g->cached_k=k; }
        mode|=16;
        x=pairs;
    }
    return q36_gpu_inkling_matvec(out,x,w,g->buf[IV_ROUTE],t->type,width,rows,stride,mode,g->nt,g->verify);
}

static bool ink_vk_mm(ink_vk *g,const q36_tensor *t,unsigned mode,unsigned rows,
                      q36_gpu_tensor *x,q36_gpu_tensor *out) {
    return ink_vk_mul(g,t,mode,rows,x,out,false);
}

static bool ink_vk_copy(ink_vk *g, q36_gpu_tensor *dst,unsigned ds,unsigned di,
                         q36_gpu_tensor *src,unsigned ss,unsigned si) {
    if (g->nt==1 || (ds==INK_D && ss==INK_D))
        return q36_gpu_tensor_copy(dst,di*4,src,si*4,4ull*g->nt*INK_D);
    return ink_vk_op(g,15,INK_D/128,dst,src,NULL,NULL,NULL,NULL,INK_D,ds,ss,di,si);
}

static bool ink_vk_norm(ink_vk *g, q36_gpu_tensor *out, unsigned offset, q36_gpu_tensor *in,
                         const float *w, unsigned width, unsigned heads) {
    return ink_vk_op(g,0,heads,out,in,ink_vk_weight(g,w,width*heads),NULL,NULL,NULL,
                     width,w!=NULL,offset,0,out==g->buf[IV_PAIR] ? 2*INK_D : width*heads);
}

static bool ink_vk_mix(ink_vk *g, q36_gpu_tensor *out, q36_gpu_tensor *blocks,
        int count, q36_gpu_tensor *cur, const float *query, unsigned queries, unsigned index, bool normalize) {
    q36_gpu_tensor *scratch=g->buf[IV_MIX], *w=ink_vk_weight(g,query,INK_D*queries);
    /* The fused mixer needs a batch of tokens to fill the GPU. */
    if (g->nt>=128 && q36_gpu_inkling_tiled(g->kv_format))
        return ink_vk_op(g,36,1,out,blocks,cur,w,g->buf[IV_LIVE],NULL,
                         count,index*INK_D,ink_bits(1/sqrtf(INK_D)),queries>1,normalize);
    return ink_vk_op(g,normalize ? 23 : 1,count+1,scratch,blocks,cur,w,g->buf[IV_LIVE],NULL,
                     count,index*INK_D,ink_bits(1/sqrtf(INK_D)),queries>1,0) &&
           ink_vk_op(g,2,INK_D/128,out,scratch,blocks,normalize ? g->buf[IV_LIVE] : cur,NULL,NULL,count,0,0,0,0);
}

static bool ink_vk_gates(ink_vk *g,const ink_layer *l,unsigned j,q36_gpu_tensor *x,
                         q36_gpu_tensor *dest,q36_gpu_tensor *msg,q36_gpu_tensor *low) {
    q36_gpu_tensor **v=g->buf;
    if ((g->nt==1 || g->verify) && q36_gpu_inkling_packed() &&
        l->ctrl_gain[j]->type==Q36_TENSOR_F32 && l->ctrl_forget[j]->type==Q36_TENSOR_F32) {
        const float *gain=(const float *)(g->model->map+l->ctrl_gain[j]->abs_offset);
        const float *forget=(const float *)(g->model->map+l->ctrl_forget[j]->abs_offset);
        q36_gpu_tensor *gw=ink_vk_weight(g,gain,INK_D*64), *fw=ink_vk_weight(g,forget,INK_D*64);
        bool gc=q36_gpu_tensor_bytes(gw)==2*INK_D*64, fc=q36_gpu_tensor_bytes(fw)==2*INK_D*64;
        if (gc==fc) return ink_vk_op(g,gc ? 27 : 19,INK_D/128,x,low,gw,fw,msg,dest,0,0,j,0,0);
    }
    return ink_vk_mm(g,l->ctrl_gain[j],0,INK_D,low,v[IV_GAIN]) &&
           ink_vk_mm(g,l->ctrl_forget[j],0,INK_D,low,v[IV_FORGET]) &&
           ink_vk_op(g,12,INK_D/128,x,msg,v[IV_GAIN],v[IV_FORGET],dest,NULL,INK_D,0,j,0,0);
}

static bool ink_vk_control(ink_vk *g,const ink_layer *l,unsigned j,q36_gpu_tensor *x,
                           q36_gpu_tensor *dest,q36_gpu_tensor *msg,q36_gpu_tensor *msg_scale) {
    q36_gpu_tensor **v=g->buf, *low=v[IV_LOW];
    /* The three local attention controllers read the same destination. Its
     * normalized half of the input pair survives all three projections. */
    bool reuse=dest!=x && j>0;
    bool paired=reuse ? ink_vk_op(g,9,INK_D/128,v[IV_PAIR],msg,msg_scale,NULL,NULL,NULL,0,0,j,0,0) :
        ink_vk_op(g,0,1,v[IV_PAIR],dest,NULL,msg,msg_scale,NULL,INK_D,0,0,j+1,2*INK_D);
    if (!paired || !ink_vk_mm(g,l->ctrl_down[j],0,64,v[IV_PAIR],low)) return false;
    return ink_vk_gates(g,l,j,x,dest,msg,low);
}

/* All three attention controllers read the same loop bank. Their input pairs
 * fit in the gate/up scratch until attention starts. */
static bool ink_vk_attention_controls(ink_vk *g,const ink_layer *l,q36_gpu_tensor *shared,
        q36_gpu_tensor *msg,q36_gpu_tensor *scales) {
    q36_gpu_tensor **v=g->buf, *dest=shared ? shared : v[IV_ACT];
    unsigned stride=g->allocated*INK_D;
    if (!ink_vk_op(g,28,3,v[IV_GU],dest,msg,scales,NULL,NULL,0,shared ? 0 : stride,2*INK_D,1,0))
        return false;
    for (unsigned j=0;j<3;j++)
        if (!ink_vk_mm(g,l->ctrl_down[j],0,64,g->pair[j],g->low[j])) return false;
    for (unsigned j=0;j<3;j++)
        if (!ink_vk_gates(g,l,j,v[IV_IN_Q+j],shared ? shared : v[IV_IN_Q+j],msg,g->low[j])) return false;
    return ink_vk_op(g,28,3,v[IV_ACT],v[IV_ACT],ink_vk_weight(g,l->norm[0],INK_D),
        ink_vk_weight(g,l->norm[1],INK_D),ink_vk_weight(g,l->norm[2],INK_D),NULL,0,stride,stride,0,0);
}

static bool ink_vk_conv(ink_vk *g, q36_gpu_tensor *x, const float *w, uint32_t n,
        uint32_t offset, q36_gpu_tensor *dynamic, unsigned pos, q36_gpu_tensor *residual) {
    return ink_vk_op(g,g->verify ? 40 : 3,n,x,ink_vk_weight(g,w,n*4),g->history,dynamic,residual,NULL,
        n,offset,(dynamic!=NULL)+2*(residual!=NULL),pos,ink_bits(1/sqrtf(dynamic ? 9 : 5)));
}

static bool ink_vk_attention(ink_vk *g, const ink_layer *l, unsigned slot, unsigned pos,
                             bool shared, q36_gpu_tensor *attn, q36_gpu_tensor *residual) {
    q36_gpu_tensor **v=g->buf, *kv=g->kv[slot];
    bool full=l->dynamic[0]!=NULL;
    unsigned hist=slot*(2*INK_K+2*INK_D), cap=ink_vk_ring(g,g->cap,full);
    unsigned window=full ? 0 : INK_WINDOW;
    g->reuse=true;
    memset(g->cached,0,sizeof(g->cached));
    bool projected=ink_vk_mm(g,l->q,0,INK_Q,v[IV_IN_Q],v[IV_Q]) &&
        ink_vk_mm(g,l->gate,0,INK_Q,v[IV_IN_Q],v[IV_GATE]) &&
        ink_vk_mm(g,l->rel,0,INK_REL*INK_H,v[IV_IN_Q],v[IV_REL]) &&
        ink_vk_mm(g,l->sink,0,INK_H,v[IV_IN_Q],v[IV_SINK]) &&
        ink_vk_mm(g,l->k,0,INK_K,v[shared ? IV_IN_Q : IV_IN_K],v[IV_K]) &&
        ink_vk_mm(g,l->v,0,INK_K,v[shared ? IV_IN_Q : IV_IN_V],v[IV_V]);
    g->reuse=false;
    if (!projected) return false;
    if ((g->nt==1 || g->verify) && !full) {
        if (!ink_vk_op(g,g->verify ? 39 : 29,2*INK_KH,kv,v[IV_K],g->history,ink_vk_weight(g,l->conv[0],4*INK_K),
                ink_vk_weight(g,l->conv[1],4*INK_K),v[IV_V],pos,hist,cap,0,ink_bits(1/sqrtf(5)))) return false;
    } else for (int j=0;j<2;j++) {
        q36_gpu_tensor *dyn=full ? v[IV_DYNAMIC] : NULL;
        if (full && !ink_vk_mm(g,l->dynamic[j],0,32,v[IV_IN_K+j],dyn)) return false;
        if (g->nt==1 || g->verify) {
            if (!ink_vk_op(g,(g->verify ? 37 : 21)+j,INK_KH,kv,v[IV_K+j],g->history,ink_vk_weight(g,l->conv[j],4*INK_K),
                dyn,NULL,pos,hist+j*INK_K,cap,1,ink_bits(1/3.f))) return false;
        } else if (!ink_vk_conv(g,v[IV_K+j],l->conv[j],INK_K,hist+j*INK_K,dyn,pos,NULL)) return false;
    }
    if (!ink_vk_op(g,4,2*INK_H,v[IV_Q],ink_vk_weight(g,l->clip_q,INK_Q),v[IV_REL],
            ink_vk_weight(g,l->clip_rel,INK_H*INK_REL),v[IV_GATE],NULL,0,0,0,0,full ? pos+1 : 0)) return false;
    unsigned nt=g->nt;
    for (unsigned j=0;j<nt;j+=32) {
        g->nt=nt-j<32 ? nt-j : 32;
        unsigned end=pos+j+g->nt;
        if (window && end>window) end=window;
        unsigned queries=nt==1 ? 1 : full && pos+j>=INK_WINDOW ? 16 : 8;
        bool packed_values=nt>1 && q36_gpu_inkling_tiled(g->kv_format);
        bool tiled=nt>1 && full && pos+j>=4096 && q36_gpu_inkling_tiled(g->kv_format);
        unsigned heads=tiled ? ink_vk_head_tile(g->nt,end) : INK_KH;
        bool ok=nt==1 || g->verify || ink_vk_op(g,5,2*INK_KH,kv,v[IV_K],v[IV_V],NULL,NULL,NULL,pos+j,0,cap,0,j);
        for (unsigned first=0;ok && first<INK_KH;first+=heads) {
            unsigned k=tiled ? 3*heads : window;
            bool paired=full && nt==1 && end>=1024 && q36_gpu_inkling_tiled(g->kv_format);
            unsigned score=tiled ? 31 : paired ? 34 : queries==1 ? 25 : queries==16 ? 30 : 6;
            ok=ink_vk_op(g,score,heads*(paired ? (end+31)/32 : (end+queries+14)/16),g->scores,v[IV_Q],v[IV_REL],kv,
                    ink_vk_weight(g,l->bias_rel,INK_WINDOW*INK_REL),v[IV_SINK],pos+j,k,cap,
                    tiled ? first : ink_bits(1/sqrtf(INK_HD)),j) &&
               ink_vk_op(g,tiled ? 32 : 7,3*heads,g->scores,NULL,tiled ? v[IV_SINK] : NULL,
                    NULL,NULL,NULL,pos+j,k,cap,first,j) &&
               ink_vk_op(g,tiled ? 33 : nt==1 ? 24 : 8,tiled ? heads*4 : INK_K/(packed_values ? 32 : 8),
                    v[IV_CONTEXT],g->scores,kv,v[IV_GATE],NULL,tiled ? v[IV_SINK] : NULL,
                    pos+j,k,cap,tiled ? first : packed_values,j);
        }
        g->nt=nt;
        if (!ok) return false;
    }
    return ink_vk_mm(g,l->out,0,INK_D,v[IV_CONTEXT],attn) &&
        ink_vk_conv(g,attn,l->conv[2],INK_D,hist+2*INK_K,NULL,pos,residual);
}

static bool ink_vk_moe(ink_vk *g, const ink_layer *l, unsigned slot, unsigned pos,q36_gpu_tensor *attn) {
    q36_gpu_tensor **v=g->buf;
    bool packed=l->down->type==Q36_TENSOR_NVFP4 || l->down->type==Q36_TENSOR_IQ4_NL || l->down->type==Q36_TENSOR_Q8_0;
    if (!ink_vk_mm(g,l->router,0,128,v[IV_IN_Q],v[IV_ROUTER]) ||
        !ink_vk_op(g,14,1,v[IV_ROUTE],v[IV_ROUTER],ink_vk_weight(g,l->bias,128),NULL,NULL,NULL,0,0,0,0,0) ||
        (g->nt>=4 && !g->verify && !ink_vk_op(g,17,1,v[IV_ROUTE],NULL,NULL,NULL,NULL,NULL,0,0,0,0,0)) ||
        !ink_vk_mm(g,l->gate_up,1,2*INK_F,v[IV_IN_Q],v[IV_GU]) ||
        !ink_vk_op(g,10,8*INK_F/128,v[packed ? IV_QUANT : IV_ACT],v[IV_GU],v[IV_ROUTE],ink_vk_weight(g,l->gate_scale,128),
                   NULL,NULL,INK_F,l->gate_scale!=NULL,0,packed,0) ||
        !ink_vk_mul(g,l->down,2,INK_D,v[IV_ACT],v[IV_GU],packed)) return false;
    unsigned hist=slot*(2*INK_K+2*INK_D)+2*INK_K+INK_D;
    return ink_vk_op(g,g->nt==1 ? 20 : g->verify ? 41 : 11,g->nt==1 ? INK_D/128 : INK_D,attn,v[IV_GU],v[IV_ROUTE],ink_vk_weight(g,l->down_scale,128),
        g->history,ink_vk_weight(g,l->conv[3],INK_D*4),INK_D,l->down_scale!=NULL,hist,pos,ink_bits(1/sqrtf(5)));
}

static bool ink_vk_forward(q36_session *s, const int *tokens, const float *embeddings,
                           unsigned nt, unsigned pos, bool logits) {
    ink_vk *g=s->runtime;
    ink_dspark *draft=s->vision_state || s->engine->mtp_draft_tokens<=1 ? NULL : g->draft;
    ink_weights *w=s->engine->ink;
    if (draft && (nt>1 || draft->pending==16) && !ink_dspark_flush(draft,g->verify)) return false;
    if (!ink_vk_scratch(g,nt,pos+nt)) return false;
    q36_gpu_tensor **v=g->buf, *cur=v[IV_CUR], *blocks=v[IV_BLOCKS], *attn=v[IV_ATTN], *x=v[IV_IN_Q];
    unsigned count=0;
    if (embeddings) {
        if (!q36_gpu_tensor_write(cur,0,embeddings,4ull*nt*INK_D)) return false;
    } else {
        float *emb=xmalloc(4ull*nt*INK_D);
        bool ok=true;
        for (unsigned i=0;i<nt && ok;i++)
            ok=tokens[i]>=0 && tokens[i]<INK_VOCAB &&
               q36_tensor_row_to_float(g->model,w->embd,tokens[i],emb+i*INK_D,INK_D);
        ok=ok && q36_gpu_tensor_write(cur,0,emb,4ull*nt*INK_D);
        free(emb);
        if (!ok || !ink_vk_norm(g,cur,0,cur,w->embd_norm,INK_D,1)) return false;
    }
    for (int slot=0;slot<INK_SLOTS;slot++) {
        if (draft && (slot==5 || slot==23 || slot==41) &&
            !ink_dspark_tap(draft,(slot-5)/18,cur,nt)) return false;
        unsigned il=ink_physical_layer(slot);
        ink_layer *l=&w->layer[il];
        unsigned queries=l->dynamic[0] ? 3 : 1;
        bool repeat=slot>=18 && slot<42;
        if (slot==6 && !q36_gpu_tensor_copy(v[IV_PRELUDE],0,cur,0,4ull*nt*INK_D)) return false;
        if (repeat && il==6) {
            if (!ink_vk_copy(g,v[IV_BANK],INK_D,0,blocks,4*INK_D,2*INK_D) ||
                !ink_vk_norm(g,v[IV_BANK_OUT],0,cur,NULL,INK_D,1) ||
                !q36_gpu_tensor_copy(cur,0,v[IV_PRELUDE],0,4ull*nt*INK_D)) return false;
            count=1;
        }
        q36_gpu_tensor *msg=v[IV_MSG], *msg_scale=v[IV_MSG_SCALE];
        if (repeat && nt==1) {
            if (il==6 && (!ink_vk_message_views(g,w) ||
                !ink_vk_op(g,18,48,msg,v[IV_BANK],msg_scale,v[IV_BANK_OUT],g->queries,NULL,
                           0,0,ink_bits(1/sqrtf(INK_D)),1,0))) return false;
            msg=g->messages[il-6]; msg_scale=g->message_scale[il-6];
        } else if (repeat && !ink_vk_op(g,18,4,v[IV_MSG],v[IV_BANK],v[IV_MSG_SCALE],v[IV_BANK_OUT],
                ink_vk_weight(g,l->msg_query,4*INK_D),NULL,0,0,ink_bits(1/sqrtf(INK_D)),0,0)) return false;
        if (!l->attn_query && !ink_vk_norm(g,v[IV_LIVE],0,cur,NULL,INK_D,1)) return false;
        if (repeat && queries==1 && !ink_vk_mix(g,attn,blocks,count,cur,l->attn_query,1,0,true)) return false;
        bool shared=!repeat && queries==1;
        bool together=repeat && nt==1 && q36_gpu_inkling_packed();
        for (unsigned j=0;j<3 && together;j++)
            together=l->ctrl_down[j]->type==Q36_TENSOR_F32 && l->ctrl_gain[j]->type==Q36_TENSOR_F32 &&
                     l->ctrl_forget[j]->type==Q36_TENSOR_F32;

        for (unsigned j=0;j<(shared ? 1u : 3u);j++) {
            q36_gpu_tensor *in=v[IV_IN_Q+j];
            q36_gpu_tensor *dest=repeat && queries==1 ? attn : in;
            if (dest==in) {
                if (l->attn_query) {
                    if (!ink_vk_mix(g,in,blocks,count,j ? v[IV_LIVE] : cur,l->attn_query,queries,j,j==0)) return false;
                } else if (!q36_gpu_tensor_copy(in,0,cur,0,4ull*nt*INK_D)) return false;
            }
            if (together) continue;
            if (repeat && !ink_vk_control(g,l,j,in,dest,msg,msg_scale)) return false;
            if (!ink_vk_norm(g,in,0,in,l->norm[j],INK_D,1)) return false;
        }
        if (together && !ink_vk_attention_controls(g,l,queries==1 ? attn : NULL,msg,msg_scale)) return false;
        if (!ink_vk_attention(g,l,slot,pos,shared,attn,il%6 ? cur : NULL)) return false;
        if (il%6==0) {
            if (!ink_vk_copy(g,blocks,4*INK_D,count++*INK_D,v[IV_LIVE],INK_D,0)) return false;
        }
        if (!ink_vk_mix(g,x,blocks,count,attn,l->ffn_query,1,0,true) ||
            (repeat && !ink_vk_control(g,l,3,x,x,msg,msg_scale)) ||
            !ink_vk_norm(g,x,0,x,l->ffn_norm,INK_D,1) || !ink_vk_moe(g,l,slot,pos,attn)) return false;
        q36_gpu_tensor *tmp=cur; cur=attn; attn=tmp;
    }
    s->gpu_top2_valid=false;
    s->logits_host_valid=false;
    if (logits || draft) {
        if (!ink_vk_mix(g,x,blocks,count,cur,w->output_query,1,0,true) ||
            !ink_vk_norm(g,x,0,x,w->output_norm,INK_D,1)) return false;
        if (draft) {
            if (!ink_dspark_tap(draft,3,x,nt)) return false;
            if (g->verify) {
                q36_gpu_tensor *vocab=draft->buf[DS_VOCAB];
                return ink_dspark_project(s,x,nt) &&
                    ink_vk_op(g,13,(INK_PADDED+127)/128,vocab,NULL,NULL,NULL,NULL,NULL,INK_PADDED,INK_VOCAB,0,0,0) &&
                    q36_gpu_top2_rows_tensor(v[IV_ROUTE],vocab,INK_PADDED,nt,-1) &&
                    q36_gpu_tensor_read(v[IV_ROUTE],0,draft->top,nt*sizeof(draft->top[0]));
            }
            if (nt==1) {
                if (!draft->pending) draft->pending_pos=pos;
                draft->pending++;
            } else if (!ink_dspark_inject(draft,pos,nt)) return false;
            if (!logits) return true;
        }
        g->nt=1;
        if (nt>1 && !q36_gpu_tensor_copy(cur,0,x,4ull*(nt-1)*INK_D,4*INK_D)) return false;
        if (!ink_vk_mm(g,w->output,0,INK_VOCAB,nt==1 ? x : cur,v[IV_LOGITS]) ||
            !ink_vk_op(g,13,(INK_VOCAB+127)/128,v[IV_LOGITS],NULL,NULL,NULL,NULL,NULL,INK_VOCAB,0,0,0,0)) return false;
        if (q36_gpu_top2_tensor(v[IV_ROUTE],v[IV_LOGITS],INK_VOCAB)) {
            if (!q36_gpu_tensor_read(v[IV_ROUTE],0,s->gpu_top2,sizeof(s->gpu_top2))) return false;
            s->gpu_top2_valid=s->gpu_top2[0]>=0 && s->gpu_top2[1]>=0;
        }
        if (!s->gpu_top2_valid) {
            if (!q36_gpu_tensor_read(v[IV_LOGITS],0,s->logits,INK_VOCAB*4)) return false;
            s->logits_host_valid=true;
        }
    }
    return true;
}
#endif
