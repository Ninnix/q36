/* qwen4exp uses the existing tokenizer and quantizers, with a separate graph. */
static struct {
    uint64_t multiplier[3];
    uint32_t head_offset[16], head_vocab[16];
    uint64_t n_rows;
} g_q36_q4_ple;

static void q36_q4_array(const q36_model *m, const char *key, uint64_t *dst, uint32_t n) {
    q36_array_ref a;
    if (!model_get_array(m, key, &a) || a.len != n ||
        (a.type != GGUF_VALUE_UINT64 && a.type != GGUF_VALUE_UINT32 &&
         a.type != GGUF_VALUE_INT32)) q36_die("invalid Flash Next metadata array");
    q36_cursor c = cursor_at(m, a.data_pos);
    for (uint32_t i = 0; i < n; i++) {
        if (a.type == GGUF_VALUE_UINT64) {
            if (!cursor_u64(&c, dst + i)) q36_die(c.error);
        } else {
            uint32_t v;
            if (!cursor_u32(&c, &v)) q36_die(c.error);
            dst[i] = v;
        }
    }
}

static void q36_q4_config(const q36_model *m) {
    g_q36_shape = (q36_shape){
        .name = "Qwen3.8 Flash Next", .arch = "qwen4exp",
        .variant = Q36_VARIANT_FLASH_NEXT, .n_layer = 48, .n_embd = 2560,
        .n_vocab = 248320, .n_head = 24, .n_head_kv = 2,
        .n_head_dim = 256, .n_value_dim = 256, .n_rot = 64,
        .n_expert_used = 10, .n_ff_exp = 640, .n_ff_shared = 640,
        .n_ssm_conv = 4, .n_ssm_state = 128, .n_ssm_group = 16,
        .n_ssm_dt_rank = 48, .n_ssm_inner = 6144, .full_attention_interval = 4,
    };
    g_q36_shape.n_expert = required_u32(m, "qwen4exp.expert_count");
    if (Q36_N_EXPERT != 256 && Q36_N_EXPERT != 512)
        q36_die("Flash Next requires 256 or 512 experts");
    static const struct { const char *key; uint32_t value; } fields[] = {
        {"block_count",48}, {"embedding_length",2560},
        {"attention.head_count",24}, {"attention.head_count_kv",2},
        {"attention.key_length",256}, {"attention.value_length",256},
        {"expert_used_count",10}, {"expert_feed_forward_length",640},
        {"expert_shared_feed_forward_length",640}, {"hyper_connection.count",4},
        {"hyper_connection.low_rank",320}, {"attention.indexer.head_count",4},
        {"attention.indexer.key_length",128}, {"attention.indexer.top_k",2048},
        {"ssm.conv_kernel",4}, {"ssm.state_size",128}, {"ssm.group_count",16},
        {"ssm.time_step_rank",48}, {"ssm.inner_size",6144},
        {"full_attention_interval",4}, {"rope.dimension_count",64},
        {"ple.ngram_size",3}, {"ple.heads_per_ngram",8}, {"ple.conv_kernel",4},
        {"ple.eos_token_id",248044}, {"embedding_length_per_layer_input",160},
    };
    for (uint32_t i = 0; i < sizeof(fields)/sizeof(*fields); i++) {
        char key[128];
        snprintf(key, sizeof(key), "qwen4exp.%s", fields[i].key);
        config_expect_u32(key, required_u32(m, key), fields[i].value);
    }
    config_expect_f32("qwen4exp.rope.freq_base", required_f32(m, "qwen4exp.rope.freq_base"), 10000000);
    config_expect_f32("qwen4exp.attention.layer_norm_rms_epsilon",
                     required_f32(m, "qwen4exp.attention.layer_norm_rms_epsilon"), Q36_RMS_EPS);
    config_expect_string("tokenizer.ggml.model",required_string(m,"tokenizer.ggml.model"),"gpt2");
    config_expect_string("tokenizer.ggml.pre",required_string(m,"tokenizer.ggml.pre"),"qwen35");
    uint64_t values[48];
    q36_q4_array(m,"qwen4exp.rope.dimension_sections",values,4);
    for (uint32_t i = 0; i < 4; i++)
        if (values[i] != Q36_ROPE_SECTIONS[i]) q36_die("unsupported Flash Next rope sections");
    q36_q4_array(m, "qwen4exp.ple.layers", values, 1);
    if (values[0] != 1) q36_die("Flash Next PLE must be at layer 1");
    q36_q4_array(m, "qwen4exp.attention.compress_ratios", values, 48);
    for (uint32_t i = 0; i < 48; i++)
        if (values[i] != ((i + 1) % 4 == 0 ? 4u : 0u))
            q36_die("unsupported Flash Next attention compression");
    q36_q4_array(m, "qwen4exp.ple.layer_multipliers", g_q36_q4_ple.multiplier, 3);
    q36_q4_array(m, "qwen4exp.ple.head_offsets", values, 16);
    for (uint32_t i = 0; i < 16; i++) {
        if (values[i] > UINT32_MAX) q36_die("PLE offset overflow");
        g_q36_q4_ple.head_offset[i] = values[i];
    }
    q36_q4_array(m, "qwen4exp.ple.head_vocab_sizes", values, 16);
    g_q36_q4_ple.n_rows = 0;
    for (uint32_t i = 0; i < 16; i++) {
        if (!values[i] || values[i] > UINT32_MAX) q36_die("invalid PLE vocabulary");
        g_q36_q4_ple.head_vocab[i] = values[i];
        uint64_t end = values[i] + g_q36_q4_ple.head_offset[i];
        if (end > UINT32_MAX) q36_die("PLE table range overflow");
        if (end > g_q36_q4_ple.n_rows) g_q36_q4_ple.n_rows = end;
    }
}

static void q36_q4_expect(const q36_tensor *t, uint32_t rank, uint64_t x, uint64_t y, uint64_t z) {
    uint64_t stride;
    if (!t || t->ndim != rank || t->dim[0] != x ||
        (rank > 1 && t->dim[1] != y) || (rank > 2 && t->dim[2] != z) ||
        !tensor_nbytes(t->type,x,&stride)) q36_die("invalid Flash Next tensor layout");
}

static void q36_q4_plain(const q36_tensor *t, uint32_t size) {
    q36_q4_expect(t,1,size,0,0);
    if (t->type != Q36_TENSOR_F32) q36_die("Flash Next norm/state weights must be F32");
}

static void q36_q4_bind(q36_weights *w, const q36_model *m) {
    memset(w, 0, sizeof(*w));
    w->token_embd = required_tensor(m, "token_embd.weight");
    w->output = required_tensor(m, "output.weight");
    w->output_hc_norm = required_tensor(m, "output_hc_norm.weight");
    w->output_hc_down = required_tensor(m, "output_hc_down.weight");
    w->output_hc_up = required_tensor(m, "output_hc_up.weight");
    q36_q4_expect(w->token_embd,2,2560,248320,0);
    q36_q4_expect(w->output,2,2560,248320,0);
    q36_q4_plain(w->output_hc_norm,10240);
    q36_q4_expect(w->output_hc_down,2,10240,320,0);
    q36_q4_expect(w->output_hc_up,2,320,10240,0);
    for (uint32_t il = 0; il < Q36_N_LAYER; il++) {
        q36_layer_weights *l = w->layer + il;
#define Q4_BIND(field, name) l->field = required_tensorf(m, "blk.%u." name, il)
        Q4_BIND(hc_attn_norm,"hc_attn_norm.weight");
        Q4_BIND(hc_attn_down,"hc_attn_down.weight");
        Q4_BIND(hc_attn_up,"hc_attn_up.weight");
        Q4_BIND(hc_attn_inject,"hc_attn_inject.weight");
        Q4_BIND(hc_ffn_norm,"hc_ffn_norm.weight");
        Q4_BIND(hc_ffn_down,"hc_ffn_down.weight");
        Q4_BIND(hc_ffn_up,"hc_ffn_up.weight");
        Q4_BIND(hc_ffn_inject,"hc_ffn_inject.weight");
        Q4_BIND(ffn_gate_inp,"ffn_gate_inp.weight");
        Q4_BIND(ffn_gate_inp_shexp,"ffn_gate_inp_shexp.weight");
        Q4_BIND(ffn_gate_exps,"ffn_gate_exps.weight");
        Q4_BIND(ffn_up_exps,"ffn_up_exps.weight");
        Q4_BIND(ffn_down_exps,"ffn_down_exps.weight");
        Q4_BIND(ffn_gate_shexp,"ffn_gate_shexp.weight");
        Q4_BIND(ffn_up_shexp,"ffn_up_shexp.weight");
        Q4_BIND(ffn_down_shexp,"ffn_down_shexp.weight");
        if (q36_layer_is_full_attention(il)) {
            Q4_BIND(attn_q,"attn_q.weight"); Q4_BIND(attn_k,"attn_k.weight");
            Q4_BIND(attn_v,"attn_v.weight"); Q4_BIND(attn_output,"attn_output.weight");
            Q4_BIND(attn_q_norm,"attn_q_norm.weight"); Q4_BIND(attn_k_norm,"attn_k_norm.weight");
            Q4_BIND(indexer_q_proj,"indexer.q_proj.weight"); Q4_BIND(indexer_k_proj,"indexer.k_proj.weight");
            Q4_BIND(indexer_q_norm,"indexer.q_norm.weight"); Q4_BIND(indexer_k_norm,"indexer.k_norm.weight");
        } else {
            Q4_BIND(attn_qkv,"attn_qkv.weight"); Q4_BIND(attn_gate,"attn_gate.weight");
            Q4_BIND(ssm_a,"ssm_a"); Q4_BIND(ssm_dt,"ssm_dt.bias");
            Q4_BIND(ssm_alpha,"ssm_alpha.weight"); Q4_BIND(ssm_beta,"ssm_beta.weight");
            Q4_BIND(ssm_conv1d,"ssm_conv1d.weight"); Q4_BIND(ssm_norm,"ssm_norm.weight");
            Q4_BIND(ssm_out,"ssm_out.weight");
        }
        if (il == 1) {
            Q4_BIND(ple_key,"ple_key.weight"); Q4_BIND(ple_value,"ple_value.weight");
            Q4_BIND(ple_norm_key,"ple_norm_key.weight"); Q4_BIND(ple_norm_query,"ple_norm_query.weight");
            Q4_BIND(ple_norm_conv,"ple_norm_conv.weight"); Q4_BIND(ple_conv,"ple_conv1d.weight");
        }
#undef Q4_BIND
        q36_q4_plain(l->hc_attn_norm,10240); q36_q4_plain(l->hc_ffn_norm,10240);
        q36_q4_expect(l->hc_attn_down,2,10240,320,0); q36_q4_expect(l->hc_ffn_down,2,10240,320,0);
        q36_q4_expect(l->hc_attn_up,2,320,10240,0); q36_q4_expect(l->hc_ffn_up,2,320,10240,0);
        q36_q4_expect(l->hc_attn_inject,2,10240,4,0); q36_q4_expect(l->hc_ffn_inject,2,10240,4,0);
        q36_q4_expect(l->ffn_gate_inp,2,2560,Q36_N_EXPERT,0);
        q36_q4_expect(l->ffn_gate_inp_shexp,1,2560,0,0);
        q36_q4_expect(l->ffn_gate_exps,3,2560,640,Q36_N_EXPERT);
        q36_q4_expect(l->ffn_up_exps,3,2560,640,Q36_N_EXPERT);
        q36_q4_expect(l->ffn_down_exps,3,640,2560,Q36_N_EXPERT);
        q36_q4_expect(l->ffn_gate_shexp,2,2560,640,0);
        q36_q4_expect(l->ffn_up_shexp,2,2560,640,0);
        q36_q4_expect(l->ffn_down_shexp,2,640,2560,0);
        if (q36_layer_is_full_attention(il)) {
            q36_q4_expect(l->attn_q,2,2560,12288,0); q36_q4_expect(l->attn_k,2,2560,512,0);
            q36_q4_expect(l->attn_v,2,2560,512,0); q36_q4_expect(l->attn_output,2,6144,2560,0);
            q36_q4_plain(l->attn_q_norm,256); q36_q4_plain(l->attn_k_norm,256);
            q36_q4_expect(l->indexer_q_proj,2,2560,512,0); q36_q4_expect(l->indexer_k_proj,2,2560,128,0);
            q36_q4_plain(l->indexer_q_norm,128); q36_q4_plain(l->indexer_k_norm,128);
        } else {
            q36_q4_expect(l->attn_qkv,2,2560,10240,0); q36_q4_expect(l->attn_gate,2,2560,6144,0);
            q36_q4_expect(l->ssm_alpha,2,2560,48,0); q36_q4_expect(l->ssm_beta,2,2560,48,0);
            q36_q4_expect(l->ssm_conv1d,2,4,10240,0);
            if (l->ssm_conv1d->type != Q36_TENSOR_F32) q36_die("Flash Next GDN convolution must be F32");
            q36_q4_plain(l->ssm_a,48); q36_q4_plain(l->ssm_dt,48); q36_q4_plain(l->ssm_norm,128);
        }
        if (il == 1) {
            q36_q4_expect(l->ple_key,2,2560,10240,0); q36_q4_expect(l->ple_value,2,2560,2560,0);
            q36_q4_plain(l->ple_norm_key,10240); q36_q4_plain(l->ple_norm_query,10240); q36_q4_plain(l->ple_norm_conv,10240);
            q36_q4_expect(l->ple_conv,2,4,10240,0);
            if (l->ple_conv->type != Q36_TENSOR_F16 && l->ple_conv->type != Q36_TENSOR_F32)
                q36_die("unsupported Flash Next PLE convolution type");
        }
    }
}

static bool q36_q4_ngram_candidate(q36_model *m, const char *path) {
    q36_model other;
    model_open(&other, path, false);
    q36_tensor *t = model_find_tensor(&other, "per_layer_token_embd.weight");
    bool ok = t && t->ndim == 2 && t->dim[0] == 160 &&
              t->dim[1] >= g_q36_q4_ple.n_rows &&
              (t->type == Q36_TENSOR_IQ4_NL || t->type == Q36_TENSOR_BF16 ||
               t->type == Q36_TENSOR_F16 || t->type == Q36_TENSOR_F32);
    if (ok) {
        if (m->ngram_fd >= 0) q36_die("ambiguous n-gram shards; use matching split filenames");
        m->ngram = *t;
        m->ngram.name = (q36_str){0};
        m->ngram_fd = dup(other.fd);
        if (m->ngram_fd < 0) q36_die_errno("cannot open n-gram shard", path);
        fprintf(stderr, "q36: disk-only n-grams: %s (%s)\n", path, tensor_type_name(t->type));
    }
    model_close(&other);
    return ok;
}

static void q36_q4_open_ngrams(q36_model *m, const char *path) {
    q36_tensor *t = model_find_tensor(m, "per_layer_token_embd.weight");
    if (t) {
        m->ngram = *t;
        m->ngram_fd = dup(m->fd);
        if (m->ngram_fd < 0) q36_die_errno("cannot open n-grams", path);
        return;
    }
    const char *split = strstr(path, "-00001-of-00002.gguf");
    if (!split || split[20]) q36_die("Flash Next requires its n-gram GGUF shard");
    char *next = q36_strdup(path);
    next[split - path + 5] = '2';
    if (access(next, R_OK) == 0 && q36_q4_ngram_candidate(m, next)) {
        free(next);
        return;
    }
    free(next);
    /* Hybrid packs sometimes keep the original n-gram shard's filename. */
    char *dir = q36_strdup(path);
    char *slash = strrchr(dir, '/');
    if (slash) *slash = 0;
    else { free(dir); dir = q36_strdup("."); }
    DIR *d = opendir(dir);
    if (!d) q36_die_errno("cannot list model directory", dir);
    struct dirent *entry;
    while ((entry = readdir(d))) {
        const char *suffix = strstr(entry->d_name, "-00002-of-00002.gguf");
        if (!suffix || suffix[20]) continue;
        size_t n = strlen(dir) + strlen(entry->d_name) + 2;
        char *candidate = xmalloc(n);
        snprintf(candidate, n, "%s/%s", dir, entry->d_name);
        q36_q4_ngram_candidate(m, candidate);
        free(candidate);
    }
    closedir(d);
    free(dir);
    if (m->ngram_fd < 0) q36_die("no matching Flash Next n-gram shard found");
}
