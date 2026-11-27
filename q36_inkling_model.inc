/* Physical weights are shared by three passes over the middle twelve layers. */
enum {
    INK_D = 2304, INK_H = 24, INK_KH = 8, INK_HD = 128, INK_K = 1024,
    INK_Q = 3072, INK_F = 1152, INK_REL = 16, INK_WINDOW = 1024,
    INK_LAYERS = 24, INK_VOCAB = 200058, INK_PADDED = 201024
};

typedef struct {
    const q36_tensor *q, *k, *v, *out, *gate, *rel, *sink, *router, *gate_up, *down;
    const q36_tensor *dynamic[2], *ctrl_down[4], *ctrl_gain[4], *ctrl_forget[4];
    const float *norm[3], *clip_q, *clip_rel, *bias_rel, *conv[4];
    const float *attn_query, *ffn_query, *msg_query, *ffn_norm;
    const float *bias, *gate_scale, *down_scale;
} ink_layer;

typedef struct ink_weights {
    ink_layer layer[INK_LAYERS];
    const q36_tensor *embd, *output;
    const float *embd_norm, *output_norm, *output_query;
} ink_weights;

static void ink_config(const q36_model *m) {
    g_q36_shape = (q36_shape){
        .name = "Inkling 2 Mini", .arch = "inkling2_loop", .variant = Q36_VARIANT_INKLING,
        .n_layer = INK_LAYERS, .n_vocab = INK_VOCAB, .n_embd = INK_D,
        .n_head = INK_H, .n_head_kv = INK_KH, .n_head_dim = INK_HD, .n_value_dim = INK_HD,
        .n_expert = 128, .n_expert_used = 8, .n_ff_exp = INK_F, .full_attention_interval = 6
    };
    static const struct { const char *key; uint32_t n; } dims[] = {
        {"block_count",24},{"loop.count",3},{"loop.start",6},{"loop.end",18},{"loop.gate_rank",64},
        {"attn_res.block_size",6},{"shortconv.l_cache",4},{"attention.relative_length",16},
        {"attention.relative_extent",1024},{"attention.sliding_window",1024},{"attention.log_scale_floor",8192},
        {"expert_count",128},{"expert_used_count",8},{"expert_feed_forward_length",1152},
        {"embedding_length",2304},{"attention.key_length",128},{"attention.value_length",128},
        {"rope.dimension_count",0},{"vocab_size",INK_PADDED},{"vocab_size_unpadded",INK_VOCAB}
    };
    char key[128];
    for (size_t i = 0; i < sizeof(dims)/sizeof(*dims); i++) {
        snprintf(key,sizeof(key),"inkling2_loop.%s",dims[i].key);
        config_expect_u32(key, required_u32(m,key), dims[i].n);
    }
    static const struct { const char *key; float v; } params[] = {
        {"attention.layer_norm_rms_epsilon",1e-6f},{"attention.qk_clip",22.62741661f},
        {"attention.relative_clip",8},{"attention.log_scale_alpha",0.1f},{"expert_weights_scale",8},
        {"activation.situ_beta",4},{"activation.situ_linear_beta",25},{"final_logit_softcapping",30}
    };
    for (size_t i = 0; i < sizeof(params)/sizeof(*params); i++) {
        snprintf(key,sizeof(key),"inkling2_loop.%s",params[i].key);
        config_expect_f32(key,required_f32(m,key),params[i].v);
    }
    uint32_t heads[INK_LAYERS];
    for (int i = 0; i < INK_LAYERS; i++) heads[i] = INK_H;
    config_expect_u32_array(m,"inkling2_loop.attention.head_count",heads,INK_LAYERS);
    for (int i = 0; i < INK_LAYERS; i++) heads[i] = INK_KH;
    config_expect_u32_array(m,"inkling2_loop.attention.head_count_kv",heads,INK_LAYERS);
    q36_array_ref a;
    if (!model_get_array(m,"inkling2_loop.attention.sliding_window_pattern",&a) ||
        a.type != GGUF_VALUE_BOOL || a.len != INK_LAYERS) q36_die("invalid Inkling attention pattern");
    for (int i = 0; i < INK_LAYERS; i++)
        if (m->map[a.data_pos+i] != ((i+1)%6 != 0)) q36_die("unsupported Inkling attention pattern");
    bool varying = false;
    if (model_get_bool(m,"inkling2_loop.loop.varying",&varying) && varying) q36_die("unsupported variable Inkling loop");
    config_expect_string("tokenizer.ggml.model",required_string(m,"tokenizer.ggml.model"),"gpt2");
    config_expect_string("tokenizer.ggml.pre",required_string(m,"tokenizer.ggml.pre"),"inkling2");
}

static bool ink_packed_k(uint32_t type) {
    return type == Q36_TENSOR_Q4_K || type == Q36_TENSOR_Q5_K || type == Q36_TENSOR_Q6_K ||
           type == Q36_TENSOR_IQ2_XXS || type == Q36_TENSOR_IQ2_XS ||
           type == Q36_TENSOR_IQ3_XXS || type == Q36_TENSOR_IQ2_S;
}

static const q36_tensor *ink_tensor(const q36_model *m, int il, const char *suffix,
                                    unsigned x, unsigned y, unsigned z, bool plain) {
    char name[128];
    if (il < 0) snprintf(name,sizeof(name),"%s",suffix);
    else snprintf(name,sizeof(name),"blk.%d.%s",il,suffix);
    const q36_tensor *t = required_tensor(m,name);
    const gguf_type_info *fmt = tensor_type(t->type);
    bool supported = ink_packed_k(t->type) || t->type == Q36_TENSOR_F32 ||
                     t->type == Q36_TENSOR_BF16 || t->type == Q36_TENSOR_Q8_0 ||
                     t->type == Q36_TENSOR_IQ4_NL || t->type == Q36_TENSOR_NVFP4;
    if (!fmt || !supported || t->ndim != (z ? 3u : y ? 2u : 1u) || t->dim[0] != x ||
        (y && t->dim[1] != y) || (z && t->dim[2] != z) || x % fmt->block_elems ||
        (plain && t->type != Q36_TENSOR_F32)) {
        fprintf(stderr,"q36: invalid Inkling tensor %s\n",name);
        exit(1);
    }
    return t;
}

static ink_weights *ink_bind(const q36_model *m) {
    ink_weights *w = xcalloc(1,sizeof(*w));
    int il = -1;
#define MAT(name,x,y,z) ink_tensor(m,il,name ".weight",x,y,z,false)
#define DATA(name,x,y) ((const float *)(m->map + ink_tensor(m,il,name,x,y,0,true)->abs_offset))
    w->embd = MAT("token_embd",INK_D,INK_PADDED,0);
    w->output = MAT("output",INK_D,INK_PADDED,0);
    w->embd_norm = DATA("token_embd_norm.weight",INK_D,0);
    w->output_norm = DATA("output_norm.weight",INK_D,0);
    w->output_query = DATA("output_res_query.weight",INK_D,0);
    for (il = 0; il < INK_LAYERS; il++) {
        ink_layer *l = &w->layer[il];
        bool full = (il+1)%6 == 0;
        l->q = MAT("attn_q",INK_D,INK_Q,0);
        l->k = MAT("attn_k",INK_D,INK_K,0);
        l->v = MAT("attn_v",INK_D,INK_K,0);
        l->out = MAT("attn_output",INK_Q,INK_D,0);
        l->rel = MAT("attn_rel",INK_D,INK_REL*INK_H,0);
        l->gate = MAT("attn_gate",INK_D,INK_Q,0);
        l->sink = MAT("attn_sink_proj",INK_D,INK_H,0);
        l->clip_q = DATA("attn_q_clip.weight",INK_HD,INK_H);
        l->clip_rel = DATA("attn_rel_clip.weight",INK_REL,INK_H);
        l->bias_rel = DATA("attn_rel_bias.weight",INK_REL,INK_WINDOW);
        l->conv[0] = DATA("attn_k_conv.weight",4,INK_K);
        l->conv[1] = DATA("attn_v_conv.weight",4,INK_K);
        l->conv[2] = DATA("attn_post_conv.weight",4,INK_D);
        l->conv[3] = DATA("ffn_post_conv.weight",4,INK_D);
        if (full) {
            l->norm[0] = DATA("attn_q_inp_norm.weight",INK_D,0);
            l->norm[1] = DATA("attn_k_inp_norm.weight",INK_D,0);
            l->norm[2] = DATA("attn_v_inp_norm.weight",INK_D,0);
            l->dynamic[0] = MAT("attn_k_conv_proj",INK_D,32,0);
            l->dynamic[1] = MAT("attn_v_conv_proj",INK_D,32,0);
        } else l->norm[0] = l->norm[1] = l->norm[2] = DATA("attn_norm.weight",INK_D,0);
        if (il) l->attn_query = DATA("attn_res_query.weight",INK_D,full ? 3 : 0);
        l->ffn_query = DATA("ffn_res_query.weight",INK_D,0);
        l->ffn_norm = DATA("ffn_norm.weight",INK_D,0);
        l->router = MAT("ffn_gate_inp",INK_D,128,0);
        l->gate_up = MAT("ffn_gate_up_exps",INK_D,2*INK_F,128);
        l->down = MAT("ffn_down_exps",INK_F,INK_D,128);
        char key[80];
        snprintf(key,sizeof(key),"blk.%d.ffn_gate_up_exps.scale",il);
        if (model_find_tensor(m,key)) l->gate_scale = DATA("ffn_gate_up_exps.scale",128,0);
        snprintf(key,sizeof(key),"blk.%d.ffn_down_exps.scale",il);
        if (model_find_tensor(m,key)) l->down_scale = DATA("ffn_down_exps.scale",128,0);
        l->bias = DATA("exp_probs_b.bias",128,0);
        if (il >= 6 && il < 18) {
            l->msg_query = DATA("loop_msg_query.weight",INK_D,4);
            const char *ctrl[] = {"q","k","v","ffn"};
            for (int j = 0; j < 4; j++) {
                char name[80];
                snprintf(name,sizeof(name),"loop_%s_ctrl_down.weight",ctrl[j]);
                l->ctrl_down[j] = ink_tensor(m,il,name,2*INK_D,64,0,false);
                snprintf(name,sizeof(name),"loop_%s_ctrl_gain.weight",ctrl[j]);
                l->ctrl_gain[j] = ink_tensor(m,il,name,64,INK_D,0,false);
                snprintf(name,sizeof(name),"loop_%s_ctrl_forget.weight",ctrl[j]);
                l->ctrl_forget[j] = ink_tensor(m,il,name,64,INK_D,0,false);
            }
        }
    }
#undef DATA
#undef MAT
    return w;
}
