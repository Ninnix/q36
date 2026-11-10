/* Qwen4exp reference math adapted from antirez/ds4 (MIT), revision b3f50180.
 * Norm gammas are already folded in GGUF; only ssm_norm uses the raw gamma. */

static uint32_t g_q36_q4_threads = 1;
static bool g_q36_q4_streaming;

static void q36_q4_row(const q36_model *m, const q36_tensor *t, uint64_t row, float *out) {
    if (!q36_tensor_row_to_float(m, t, row, out, t->dim[0]))
        q36_die("cannot decode Flash Next tensor row");
}

typedef struct {
    const q36_model *m;
    const q36_tensor *t;
    const float *x;
    const uint8_t *xq;
    float *out;
    uint64_t row0;
} q36_q4_mm;

static float q36_q4_bf16_dot(const uint16_t *w, const float *x, uint32_t n) {
    float lane[32] = {0};
    for (uint32_t i = 0; i < n; i += 32)
        for (uint32_t j = 0; j < 32; j++) lane[j] += q36_bf16_to_f32(w[i+j]) * x[i+j];
    float v[8];
    for (uint32_t j = 0; j < 8; j++) v[j] = (lane[j]+lane[j+16])+(lane[j+8]+lane[j+24]);
    for (uint32_t j = 0; j < 4; j++) v[j] += v[j+4];
    return (v[0]+v[2])+(v[1]+v[3]);
}

static void q36_q4_mm_rows(void *opaque, uint64_t begin, uint64_t end) {
    q36_q4_mm *c = opaque;
    uint32_t n = c->t->dim[0];
    float *row = xmalloc(n * sizeof(float));
    for (uint64_t r = begin; r < end; r++) {
        const uint8_t *packed;
        if (!q36_tensor_row_ptr(c->m,c->t,c->row0+r,&packed,NULL)) q36_die("invalid matrix row");
        if (c->t->type == Q36_TENSOR_BF16 && n % 32 == 0) {
            c->out[r] = q36_q4_bf16_dot((const uint16_t *)packed,c->x,n);
            continue;
        }
        if (c->xq) {
            bool small = c->t->type == Q36_TENSOR_Q2_0 || c->t->type == Q36_TENSOR_IQ4_NL || c->t->type == Q36_TENSOR_Q8_0;
            bool ok = small ? q36_quant_dot_q8_0_type(c->t->type,packed,c->xq,n,c->out+r) :
                              q36_quant_dot_q8_k(c->t->type,packed,c->xq,n,c->out+r);
            if (!ok) q36_die("unsupported quantized matrix dot");
            continue;
        }
        q36_q4_row(c->m, c->t, c->row0 + r, row);
        double sum = 0;
        for (uint32_t i = 0; i < n; i++) sum += (double)row[i] * c->x[i];
        c->out[r] = (float)sum;
    }
    free(row);
}

static void q36_q4_matvec_rows(const q36_model *m, const q36_tensor *t,
                                uint64_t row0, uint64_t rows, const float *x, float *out) {
    uint32_t n = t->dim[0];
    float *rounded = xmalloc(n * sizeof(float));
    uint8_t *quant = xmalloc(n * 2);
    const uint8_t *xq = NULL;
    if (t->type == Q36_TENSOR_BF16) {
        for (uint32_t i = 0; i < n; i++) rounded[i] = q36_round_bf16(x[i]);
    } else if (t->type == Q36_TENSOR_Q2_0 || t->type == Q36_TENSOR_IQ4_NL || t->type == Q36_TENSOR_Q8_0) {
        q36_quant_q8_0_activation(x, quant, n);
        xq = quant;
    } else if (q36_tensor_type_supports_q8k_dot(t->type) && n % 256 == 0) {
        q36_quant_q8_k(x, quant, n);
        xq = quant;
    } else memcpy(rounded, x, n * sizeof(float));
    q36_q4_mm c = {m, t, rounded, xq, out, row0};
    q36_parallel_for_rows(rows, 32, g_q36_q4_threads, q36_q4_mm_rows, &c);
    free(quant);
    free(rounded);
    if (g_q36_q4_streaming) {
        uint64_t stride;
        if (!tensor_nbytes(t->type, t->dim[0], &stride)) q36_die("invalid matrix stride");
        uintptr_t lo = (uintptr_t)m->map + t->abs_offset + row0 * stride;
        uintptr_t hi = lo + rows * stride;
        uintptr_t page = (uintptr_t)sysconf(_SC_PAGESIZE);
        lo = (lo + page - 1) / page * page;
        if (hi > lo) madvise((void *)lo, hi - lo, MADV_DONTNEED);
    }
}

static void q36_q4_matvec(const q36_model *m, const q36_tensor *t, const float *x, float *out) {
    q36_q4_matvec_rows(m, t, 0, t->ndim >= 2 ? t->dim[1] : 1, x, out);
}

static bool q36_q4_ngram_read(const q36_model *m, const uint32_t *rows, uint32_t count, float *out) {
    uint64_t stride;
    if (!tensor_nbytes(m->ngram.type, 160, &stride) || stride > 640) return false;
    uint8_t raw[640];
    for (uint32_t h = 0; h < count; h++) {
        if (rows[h] >= m->ngram.dim[1]) return false;
        uint64_t off = m->ngram.abs_offset + (uint64_t)rows[h] * stride;
        size_t done = 0;
        while (done < stride) {
            ssize_t n = pread(m->ngram_fd, raw + done, stride - done, off + done);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return false;
            done += (size_t)n;
        }
        if (!q36_dequantize_row_from_ptr(m->ngram.type, raw, out + h * 160, 160)) return false;
    }
    return true;
}

static void q36_q4_dump(const char *name, uint32_t layer, const float *data, uint32_t n) {
    const char *prefix = getenv("Q36_Q4_DUMP_PREFIX");
    if (!prefix) return;
    char path[4096];
    snprintf(path,sizeof(path),"%s.%s-%u.f32",prefix,name,layer);
    FILE *fp = fopen(path,"wb");
    if (fp) { fwrite(data,sizeof(float),n,fp); fclose(fp); }
}

static const float *q36_q4_f32(const q36_model *m, const q36_tensor *t) {
    if (t->type != Q36_TENSOR_F32) q36_die("qwen4 reference: expected an F32 tensor");
    return (const float *)(m->map + t->abs_offset);
}

static void q36_q4_rms(float *out, const float *x, const float *g, uint32_t n, float eps) {
    double ss = 0.0;
    for (uint32_t i = 0; i < n; i++) ss += (double)(x[i] * x[i]);
    const float scale = 1.0f / sqrtf((float)(ss / n) + eps);
    for (uint32_t i = 0; i < n; i++) out[i] = x[i] * scale * (g ? g[i] : 1.0f);
}

static void q36_q4_grouped_rms(float *out, const float *x, const float *g, uint32_t groups, uint32_t n, float eps) {
    for (uint32_t s = 0; s < groups; s++) {
        q36_q4_rms(out + (uint64_t)s * n, x + (uint64_t)s * n, g ? g + (uint64_t)s * n : NULL, n, eps);
    }
}

static void q36_q4_l2norm(float *x, uint32_t n) {
    double ss = 0.0;
    for (uint32_t i = 0; i < n; i++) ss += (double)(x[i] * x[i]);
    const float scale = 1.0f / sqrtf((float)ss + 1e-6f);
    for (uint32_t i = 0; i < n; i++) x[i] *= scale;
}

static void q36_q4_rope(float *x, uint32_t n_rot, const uint32_t *p3) {
    const uint32_t half = n_rot / 2u;
    for (uint32_t i = 0; i < half; i++) {
        const double theta = (double)p3[i % 3u] * (double)powf(10000000.0f, -(2.0f * i) / n_rot);
        const float c = (float)cos(theta), s = (float)sin(theta);
        const float x0 = x[i], x1 = x[i + half];
        x[i] = x0 * c - x1 * s;
        x[i + half] = x0 * s + x1 * c;
    }
}

static float q36_q4_softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

typedef struct {
    uint32_t cap;
    void *gpu;
    float *lin_state;   /* [layer][Hv][Dk][Dv] */
    float *lin_hist;    /* [layer][K-1][conv_dim], oldest first, raw qkv */
    float *ple_hist;    /* [(Kp-1)*ngram][hc_dim], oldest first, normed */
    int    ple_prev[3]; /* previous tokens, newest first */
    float *attn_k;      /* [layer][cap][Hkv*D] */
    float *attn_v;
    float *idx_k;       /* [layer][cap][Di] raw indexer keys */
    uint32_t *pos3;     /* [cap][3] rope positions */
    int32_t mrope_delta;
    float last_h[2560];
} q36_q4_state;

static void q36_q4_state_init(q36_q4_state *st, uint32_t cap) {
    const uint64_t L = Q36_N_LAYER;
    const uint64_t conv_dim = Q36_N_SSM_CONV_DIM;
    const uint64_t hc_dim = (uint64_t)Q36_N_EMBD * 4u;
    memset(st, 0, sizeof(*st));
    st->cap = cap;
    st->lin_state = xcalloc(L * Q36_N_SSM_DT_RANK * Q36_N_SSM_STATE * Q36_N_SSM_STATE, sizeof(float));
    st->lin_hist = xcalloc(L * (Q36_N_SSM_CONV - 1u) * conv_dim, sizeof(float));
    st->ple_hist = xcalloc((uint64_t)(4u - 1u) * 3u * hc_dim, sizeof(float));
    for (uint32_t i = 0; i < 3; i++) st->ple_prev[i] = 248044;
    st->attn_k = xcalloc((L / 4) * cap * Q36_N_HEAD_KV * Q36_N_HEAD_DIM, sizeof(float));
    st->attn_v = xcalloc((L / 4) * cap * Q36_N_HEAD_KV * Q36_N_HEAD_DIM, sizeof(float));
    st->idx_k = xcalloc((L / 4) * cap * 128u, sizeof(float));
    st->pos3 = xcalloc((uint64_t)cap * 3u, sizeof(uint32_t));
}

static void q36_q4_gpu_release(void *gpu);

static void q36_q4_state_free(q36_q4_state *st) {
    q36_q4_gpu_release(st->gpu);
    free(st->lin_state);
    free(st->lin_hist);
    free(st->ple_hist);
    free(st->attn_k);
    free(st->attn_v);
    free(st->idx_k);
    free(st->pos3);
    memset(st, 0, sizeof(*st));
}

/* hyper-connection mix: grouped norm, low-rank sigmoid gate, mean over
 * streams; inject (2*sigmoid(w/hc)) is the scatter weight of the combine */
static void q36_q4_hc_mix(
        const q36_model *m, const float *R,
        const q36_tensor *norm, const q36_tensor *down, const q36_tensor *up, const q36_tensor *inject,
        float *mixed, float *inj) {
    const uint32_t E = Q36_N_EMBD, hc = 4u, rank = 320u;
    const uint32_t hc_dim = E * hc;
    float *xn = xmalloc(hc_dim * sizeof(float));
    float *lo = xmalloc(rank * sizeof(float));
    float *gate = xmalloc(hc_dim * sizeof(float));
    q36_q4_grouped_rms(xn, R, q36_q4_f32(m, norm), hc, E, Q36_RMS_EPS);
    q36_q4_matvec(m, down, xn, lo);
    for (uint32_t r = 0; r < rank; r++) lo[r] = q36_siluf(lo[r] / (float)hc);
    q36_q4_matvec(m, up, lo, gate);
    for (uint32_t d = 0; d < E; d++) {
        float acc = 0.0f;
        for (uint32_t s = 0; s < hc; s++) {
            const uint32_t j = s * E + d;
            acc += q36_sigmoidf(gate[j]) * xn[j];
        }
        mixed[d] = acc / (float)hc;
    }
    if (inject && inj) {
        q36_q4_matvec(m, inject, xn, inj);
        for (uint32_t s = 0; s < hc; s++) inj[s] = 2.0f * q36_sigmoidf(inj[s] / (float)hc);
    }
    free(gate);
    free(lo);
    free(xn);
}

static void q36_q4_hc_combine(float *R, const float *out, const float *inj) {
    for (uint32_t s = 0; s < 4u; s++) {
        float *dst = R + (uint64_t)s * Q36_N_EMBD;
        for (uint32_t d = 0; d < Q36_N_EMBD; d++) dst[d] += inj[s] * out[d];
    }
}

/* n-gram hash rows for `token` over the newest-first history `prev`, which
 * then advances; EOS fills the context at and before the last EOS. */
static void q36_q4_ple_step(int token, int *prev, uint32_t *rows) {
    const uint32_t per = 8u;
    int ctx[3];
    bool cut = false;
    ctx[0] = token;
    for (uint32_t s = 1; s < 3u; s++) {
        const int t = cut ? 248044 : prev[s - 1];
        cut = cut || t == 248044;
        ctx[s] = cut ? 248044 : t;
    }
    for (uint32_t n = 2; n <= 3u; n++) {
        uint64_t mixed = (uint64_t)ctx[0] * g_q36_q4_ple.multiplier[0];
        for (uint32_t j = 1; j < n; j++) mixed ^= (uint64_t)ctx[j] * g_q36_q4_ple.multiplier[j];
        for (uint32_t g = 0; g < per; g++) {
            const uint32_t h = (n - 2u) * per + g;
            rows[h] = (uint32_t)(mixed % g_q36_q4_ple.head_vocab[h]) + g_q36_q4_ple.head_offset[h];
        }
    }
    for (uint32_t s = 3 - 1u; s > 0; s--) prev[s] = prev[s - 1];
    prev[0] = token;
}

static void q36_q4_ple(const q36_model *m, const q36_layer_weights *l,
                          q36_q4_state *st, int token, float *R) {
    const uint32_t E = Q36_N_EMBD, hc = 4u, hc_dim = E * hc;
    const uint32_t n_heads = 16u;
    const uint32_t hist_rows = (4u - 1u) * 3u;
    uint32_t rows[16];
    q36_q4_ple_step(token, st->ple_prev, rows);

    float *emb = xmalloc(E * sizeof(float));
    if (!q36_q4_ngram_read(m, rows, n_heads, emb))
        q36_die_errno("cannot read Qwen n-grams", "GGUF");

    q36_q4_dump("ple_embd",0,emb,E);
    float *key = xmalloc(hc_dim * sizeof(float));
    float *keyn = xmalloc(hc_dim * sizeof(float));
    float *query = xmalloc(hc_dim * sizeof(float));
    float *value = xmalloc(E * sizeof(float));
    float *gated = xmalloc(hc_dim * sizeof(float));
    float *normed = xmalloc(hc_dim * sizeof(float));
    q36_q4_matvec(m, l->ple_key, emb, key);
    q36_q4_grouped_rms(keyn, key, q36_q4_f32(m, l->ple_norm_key), hc, E, Q36_RMS_EPS);
    q36_q4_grouped_rms(query, R, q36_q4_f32(m, l->ple_norm_query), hc, E, Q36_RMS_EPS);
    q36_q4_matvec(m, l->ple_value, emb, value);
    for (uint32_t s = 0; s < hc; s++) {
        double dot = 0.0;
        for (uint32_t d = 0; d < E; d++) dot += (double)keyn[s * E + d] * query[s * E + d];
        float g = (float)(dot / sqrt((double)E));
        const float mag = sqrtf(fmaxf(fabsf(g), 1e-6f));
        g = q36_sigmoidf(g > 0.0f ? mag : (g < 0.0f ? -mag : 0.0f));
        for (uint32_t d = 0; d < E; d++) gated[s * E + d] = g * value[d];
    }
    q36_q4_dump("ple_gated_value",1,gated,hc_dim);
    q36_q4_grouped_rms(normed, gated, q36_q4_f32(m, l->ple_norm_conv), hc, E, Q36_RMS_EPS);

    /* depthwise conv, kernel K dilated by the n-gram size: tap k reads (K-1-k)*dil back */
    float *cw_f16 = NULL;
    const float *cw;
    if (l->ple_conv->type == Q36_TENSOR_F16) {
        cw_f16 = xmalloc((size_t)hc_dim * 4u * sizeof(float));
        for (uint32_t c = 0; c < hc_dim; c++) q36_q4_row(m, l->ple_conv, c, cw_f16 + (uint64_t)c * 4u);
        cw = cw_f16;
    } else {
        cw = q36_q4_f32(m, l->ple_conv);
    }
    for (uint32_t c = 0; c < hc_dim; c++) {
        double acc = 0.0;
        for (uint32_t k = 0; k < 4u; k++) {
            const uint32_t back = (4u - 1u - k) * 3u;
            const float xk = back == 0 ? normed[c] : st->ple_hist[(uint64_t)(hist_rows - back) * hc_dim + c];
            acc += (double)cw[(uint64_t)c * 4u + k] * xk;
        }
        R[c] += gated[c] + q36_siluf((float)acc);
    }
    q36_q4_dump("ple_output",1,R,hc_dim);
    free(cw_f16);
    memmove(st->ple_hist, st->ple_hist + hc_dim, (uint64_t)(hist_rows - 1u) * hc_dim * sizeof(float));
    memcpy(st->ple_hist + (uint64_t)(hist_rows - 1u) * hc_dim, normed, hc_dim * sizeof(float));

    free(normed); free(gated); free(value); free(query); free(keyn); free(key); free(emb);
}

static void q36_q4_linear(const q36_model *m, const q36_layer_weights *l, uint32_t il,
                             q36_q4_state *st, const float *x, float *out) {
    const uint32_t Hk = Q36_N_SSM_GROUP, Hv = Q36_N_SSM_DT_RANK, D = Q36_N_SSM_STATE, K = Q36_N_SSM_CONV;
    const uint32_t k_dim = Hk * D, v_dim = Hv * D, conv_dim = 2u * k_dim + v_dim;
    float *qkv = xmalloc(conv_dim * sizeof(float));
    float *conv = xmalloc(conv_dim * sizeof(float));
    float *z = xmalloc(v_dim * sizeof(float));
    float *b = xmalloc(Hv * sizeof(float));
    float *a = xmalloc(Hv * sizeof(float));
    float *o = xmalloc(v_dim * sizeof(float));
    q36_q4_matvec(m, l->attn_qkv, x, qkv);
    q36_q4_matvec(m, l->attn_gate, x, z);
    q36_q4_matvec(m, l->ssm_beta, x, b);
    q36_q4_matvec(m, l->ssm_alpha, x, a);

    float *hist = st->lin_hist + (uint64_t)il * (K - 1u) * conv_dim;
    const float *cw = q36_q4_f32(m, l->ssm_conv1d);
    for (uint32_t c = 0; c < conv_dim; c++) {
        double acc = (double)cw[(uint64_t)c * K + (K - 1u)] * qkv[c];
        for (uint32_t k = 0; k + 1u < K; k++) acc += (double)cw[(uint64_t)c * K + k] * hist[(uint64_t)k * conv_dim + c];
        conv[c] = q36_siluf((float)acc);
    }
    memmove(hist, hist + conv_dim, (uint64_t)(K - 2u) * conv_dim * sizeof(float));
    memcpy(hist + (uint64_t)(K - 2u) * conv_dim, qkv, conv_dim * sizeof(float));

    float *q = conv, *k = conv + k_dim, *v = conv + 2u * k_dim;
    const float qscale = 1.0f / sqrtf((float)D);
    for (uint32_t h = 0; h < Hk; h++) {
        q36_q4_l2norm(q + h * D, D);
        q36_q4_l2norm(k + h * D, D);
        for (uint32_t i = 0; i < D; i++) q[h * D + i] *= qscale;
    }
    const float *A = q36_q4_f32(m, l->ssm_a);
    const float *dt = q36_q4_f32(m, l->ssm_dt);
    const float *nw = q36_q4_f32(m, l->ssm_norm);
    for (uint32_t j = 0; j < Hv; j++) {
        const uint32_t kh = j % Hk;
        const float g = expf(A[j] * q36_q4_softplus(a[j] + dt[j]));
        const float beta = q36_sigmoidf(b[j]);
        float *S = st->lin_state + ((uint64_t)il * Hv + j) * D * D;   /* [dk][dv] */
        const float *qj = q + kh * D, *kj = k + kh * D, *vj = v + j * D;
        float *oj = o + j * D;
        for (uint32_t dv = 0; dv < D; dv++) {
            double kv = 0.0;
            for (uint32_t dk = 0; dk < D; dk++) {
                S[dk * D + dv] *= g;
                kv += (double)S[dk * D + dv] * kj[dk];
            }
            const float delta = (vj[dv] - (float)kv) * beta;
            double acc = 0.0;
            for (uint32_t dk = 0; dk < D; dk++) {
                S[dk * D + dv] += kj[dk] * delta;
                acc += (double)S[dk * D + dv] * qj[dk];
            }
            oj[dv] = (float)acc;
        }
        float tmp[128];
        if (D > 128) q36_die("Qwen3.8 reference: linear head dim exceeds 128");
        q36_q4_rms(tmp, oj, nw, D, Q36_RMS_EPS);
        for (uint32_t dv = 0; dv < D; dv++) oj[dv] = tmp[dv] * q36_sigmoidf(z[j * D + dv]);
    }
    q36_q4_matvec(m, l->ssm_out, o, out);
    free(o); free(a); free(b); free(z); free(conv); free(qkv);
}

/* QSA block selection for the query at `pos`: complete blocks of ratio
 * tokens ranked by relu-summed indexer scores, plus the incomplete tail. */
static uint32_t q36_q4_select(const q36_model *m, const q36_layer_weights *l, uint32_t il,
                                 q36_q4_state *st, const float *x, uint32_t pos, int32_t *sel) {
    const uint32_t ratio = 4u, Di = 128u, Hi = 4u;
    const uint32_t k_blocks = 2048u / ratio;
    float *qi = xmalloc((uint64_t)Hi * Di * sizeof(float));
    float *kraw = st->idx_k + ((uint64_t)(il / 4) * st->cap + pos) * Di;
    q36_q4_matvec(m, l->indexer_q_proj, x, qi);
    q36_q4_matvec(m, l->indexer_k_proj, x, kraw);

    const uint32_t n_vis = pos + 1u;
    const uint32_t n_blocks = n_vis / ratio;
    uint32_t n_sel = 0;
    if (n_blocks <= k_blocks) {
        for (uint32_t t = 0; t < n_vis; t++) sel[n_sel++] = (int32_t)t;
        free(qi);
        return n_sel;
    }
    const float *gq = q36_q4_f32(m, l->indexer_q_norm);
    const float *gk = q36_q4_f32(m, l->indexer_k_norm);
    for (uint32_t h = 0; h < Hi; h++) {
        float tmp[128];
        q36_q4_rms(tmp, qi + h * Di, gq, Di, Q36_RMS_EPS);
        q36_q4_rope(tmp, Q36_N_ROT, st->pos3 + (uint64_t)pos * 3u);
        memcpy(qi + h * Di, tmp, Di * sizeof(float));
    }
    float *score = xmalloc(n_blocks * sizeof(float));
    float *pooled = xmalloc(Di * sizeof(float));
    float *key = xmalloc(Di * sizeof(float));
    for (uint32_t b = 0; b < n_blocks; b++) {
        for (uint32_t d = 0; d < Di; d++) {
            double acc = 0.0;
            for (uint32_t t = 0; t < ratio; t++)
                acc += st->idx_k[((uint64_t)(il / 4) * st->cap + b * ratio + t) * Di + d];
            pooled[d] = (float)(acc / ratio);
        }
        q36_q4_rms(key, pooled, gk, Di, Q36_RMS_EPS);
        q36_q4_rope(key, Q36_N_ROT, st->pos3 + (uint64_t)b * ratio * 3u);
        double sum = 0.0;
        for (uint32_t h = 0; h < Hi; h++) {
            double dot = 0.0;
            for (uint32_t d = 0; d < Di; d++) dot += (double)qi[h * Di + d] * key[d];
            if (dot > 0.0) sum += dot;
        }
        score[b] = (float)sum;
    }
    /* top-k blocks, lower index first on ties */
    bool *taken = xcalloc(n_blocks, sizeof(bool));
    for (uint32_t i = 0; i < k_blocks; i++) {
        int best = -1;
        for (uint32_t b = 0; b < n_blocks; b++) {
            if (taken[b]) continue;
            if (best < 0 || score[b] > score[best]) best = (int)b;
        }
        taken[best] = true;
    }
    for (uint32_t b = 0; b < n_blocks; b++) {
        if (!taken[b]) continue;
        for (uint32_t t = 0; t < ratio; t++) sel[n_sel++] = (int32_t)(b * ratio + t);
    }
    for (uint32_t t = n_blocks * ratio; t < n_vis; t++) sel[n_sel++] = (int32_t)t;
    free(taken); free(key); free(pooled); free(score); free(qi);
    return n_sel;
}

static void q36_q4_attention(const q36_model *m, const q36_layer_weights *l, uint32_t il,
                                q36_q4_state *st, const float *x, uint32_t pos, float *out) {
    const uint32_t H = Q36_N_HEAD, Hkv = Q36_N_HEAD_KV, D = Q36_N_HEAD_DIM;
    const uint32_t q_dim = H * D, kv_dim = Hkv * D;
    float *qg = xmalloc(2u * q_dim * sizeof(float));
    float *q = xmalloc(q_dim * sizeof(float));
    float *gate = xmalloc(q_dim * sizeof(float));
    float *o = xmalloc(q_dim * sizeof(float));
    float *kc = st->attn_k + ((uint64_t)(il / 4) * st->cap + pos) * kv_dim;
    float *vc = st->attn_v + ((uint64_t)(il / 4) * st->cap + pos) * kv_dim;
    q36_q4_matvec(m, l->attn_q, x, qg);
    q36_q4_matvec(m, l->attn_k, x, kc);
    q36_q4_matvec(m, l->attn_v, x, vc);
    const float *gqn = q36_q4_f32(m, l->attn_q_norm);
    const float *gkn = q36_q4_f32(m, l->attn_k_norm);
    for (uint32_t h = 0; h < H; h++) {
        q36_q4_rms(q + h * D, qg + (uint64_t)h * 2u * D, gqn, D, Q36_RMS_EPS);
        q36_q4_rope(q + h * D, Q36_N_ROT, st->pos3 + (uint64_t)pos * 3u);
        memcpy(gate + h * D, qg + (uint64_t)h * 2u * D + D, D * sizeof(float));
    }
    for (uint32_t h = 0; h < Hkv; h++) {
        float tmp[256];
        q36_q4_rms(tmp, kc + h * D, gkn, D, Q36_RMS_EPS);
        q36_q4_rope(tmp, Q36_N_ROT, st->pos3 + (uint64_t)pos * 3u);
        memcpy(kc + h * D, tmp, D * sizeof(float));
    }

    int32_t *sel = xmalloc((uint64_t)(pos + 1u) * sizeof(int32_t));
    const uint32_t n_sel = q36_q4_select(m, l, il, st, x, pos, sel);
    float *p = xmalloc(n_sel * sizeof(float));
    const float scale = 1.0f / sqrtf((float)D);
    for (uint32_t h = 0; h < H; h++) {
        const uint32_t kvh = h / (H / Hkv);
        float mx = -FLT_MAX;
        for (uint32_t i = 0; i < n_sel; i++) {
            const float *kt = st->attn_k + ((uint64_t)(il / 4) * st->cap + (uint32_t)sel[i]) * kv_dim + kvh * D;
            double dot = 0.0;
            for (uint32_t d = 0; d < D; d++) dot += (double)q[h * D + d] * kt[d];
            p[i] = (float)dot * scale;
            if (p[i] > mx) mx = p[i];
        }
        double sum = 0.0;
        for (uint32_t i = 0; i < n_sel; i++) { p[i] = expf(p[i] - mx); sum += p[i]; }
        for (uint32_t d = 0; d < D; d++) {
            double acc = 0.0;
            for (uint32_t i = 0; i < n_sel; i++) {
                const float *vt = st->attn_v + ((uint64_t)(il / 4) * st->cap + (uint32_t)sel[i]) * kv_dim + kvh * D;
                acc += (double)p[i] * vt[d];
            }
            o[h * D + d] = (float)(acc / sum) * q36_sigmoidf(gate[h * D + d]);
        }
    }
    q36_q4_matvec(m, l->attn_output, o, out);
    free(p); free(sel); free(o); free(gate); free(q); free(qg);
}

static void q36_q4_moe(const q36_model *m, const q36_layer_weights *l, const float *x, float *out) {
    const uint32_t E = Q36_N_EMBD, NE = Q36_N_EXPERT, K = Q36_N_EXPERT_USED, F = Q36_N_FF_EXP;
    float *logits = xmalloc(NE * sizeof(float));
    double *prob = xmalloc(NE * sizeof(double));
    const uint32_t down_width = (uint32_t)l->ffn_down_exps->dim[0];
    float *g = xcalloc(down_width, sizeof(float));
    float *u = xmalloc(F * sizeof(float));
    float *y = xmalloc(E * sizeof(float));
    q36_q4_matvec(m, l->ffn_gate_inp, x, logits);
    double mx = -DBL_MAX, sum = 0.0;
    for (uint32_t e = 0; e < NE; e++) if (logits[e] > mx) mx = logits[e];
    for (uint32_t e = 0; e < NE; e++) { prob[e] = exp((double)logits[e] - mx); sum += prob[e]; }
    for (uint32_t e = 0; e < NE; e++) prob[e] /= sum;
    int sel[10];
    double wsum = 0.0;
    for (uint32_t i = 0; i < K; i++) {
        int best = -1;
        for (uint32_t e = 0; e < NE; e++) {
            bool used = false;
            for (uint32_t j = 0; j < i; j++) used |= sel[j] == (int)e;
            if (!used && (best < 0 || prob[e] > prob[best])) best = (int)e;
        }
        if (best < 0) q36_die("Qwen3.8 router selected more experts than available");
        sel[i] = best;
        wsum += prob[best];
    }
    for (uint32_t d = 0; d < E; d++) out[d] = 0.0f;
    for (uint32_t i = 0; i < K; i++) {
        const uint32_t e = (uint32_t)sel[i];
        const float wgt = (float)(prob[e] / wsum);
        q36_q4_matvec_rows(m, l->ffn_gate_exps, (uint64_t)e * F, F, x, g);
        q36_q4_matvec_rows(m, l->ffn_up_exps, (uint64_t)e * F, F, x, u);
        for (uint32_t f = 0; f < F; f++) g[f] = q36_siluf(g[f]) * u[f];
        q36_q4_matvec_rows(m, l->ffn_down_exps, (uint64_t)e * E, E, g, y);
        for (uint32_t d = 0; d < E; d++) out[d] += wgt * y[d];
    }
    q36_q4_matvec(m, l->ffn_gate_shexp, x, g);
    q36_q4_matvec(m, l->ffn_up_shexp, x, u);
    for (uint32_t f = 0; f < F; f++) g[f] = q36_siluf(g[f]) * u[f];
    q36_q4_matvec(m, l->ffn_down_shexp, g, y);
    float sg;
    q36_q4_matvec(m, l->ffn_gate_inp_shexp, x, &sg);
    sg = q36_sigmoidf(sg);
    for (uint32_t d = 0; d < E; d++) out[d] += sg * y[d];
    free(y); free(u); free(g); free(prob); free(logits);
}

/* one trunk-layer step on the wide residual R */
static void q36_q4_layer(const q36_model *m, const q36_weights *w, uint32_t il,
                            q36_q4_state *st, int token, uint32_t pos, float *R) {
    const q36_layer_weights *l = &w->layer[il];
    float mixed[2560], blk[2560], inj[4];

    if ((il == 1)) q36_q4_ple(m, l, st, token, R);
    q36_q4_hc_mix(m, R, l->hc_attn_norm, l->hc_attn_down, l->hc_attn_up, l->hc_attn_inject, mixed, inj);
    if ((!q36_layer_is_full_attention(il))) q36_q4_linear(m, l, il, st, mixed, blk);
    else q36_q4_attention(m, l, il, st, mixed, pos, blk);
    q36_q4_hc_combine(R, blk, inj);
    q36_q4_hc_mix(m, R, l->hc_ffn_norm, l->hc_ffn_down, l->hc_ffn_up, l->hc_ffn_inject, mixed, inj);
    q36_q4_moe(m, l, mixed, blk);
    q36_q4_hc_combine(R, blk, inj);
}

static void q36_q4_output(const q36_model *m, const q36_weights *w, const float *mixed, float *logits) {
    q36_q4_matvec(m, w->output, mixed, logits);
}

/* Forward `token` at `pos`; R_out (optional) receives the pre-mixer stream,
 * streams (optional) the stream after every trunk layer. */
static bool q36_q4_forward_slice_token(const q36_model *m, const q36_weights *w, q36_q4_state *st,
    int token, uint32_t pos, uint32_t first, uint32_t last, const float *input,
    float *logits, float *R_out, float *mixed_out, float *streams) {
    const uint32_t E = Q36_N_EMBD, hc = 4u, hc_dim = E * hc;
    float *R = xmalloc(hc_dim * sizeof(float));
    float *mixed = xmalloc(E * sizeof(float));
    if (input) memcpy(R,input,hc_dim*sizeof(float));
    else {
        q36_q4_row(m, w->token_embd, (uint64_t)token, R);
        for (uint32_t s = 1; s < hc; s++) memcpy(R + (uint64_t)s * E, R, E * sizeof(float));
    }
    for (uint32_t d = 0; d < 3; d++) st->pos3[(uint64_t)pos * 3 + d] = pos;
    for (uint32_t il = first; il <= last; il++) {
        q36_q4_layer(m, w, il, st, token, pos, R);
        if (!q36_all_finite(R, hc_dim)) {
            fprintf(stderr, "q36: non-finite Flash Next residual at layer %u\n", il);
            free(mixed); free(R);
            return false;
        }
        const char *prefix = getenv("Q36_Q4_DUMP_PREFIX");
        if (prefix) {
            char path[4096];
            snprintf(path, sizeof(path), "%s.l_last-%u.f32", prefix, il);
            FILE *fp = fopen(path, "wb");
            if (fp) { fwrite(R, sizeof(float), hc_dim, fp); fclose(fp); }
        }
        if (streams) memcpy(streams + (uint64_t)il * hc_dim, R, hc_dim * sizeof(float));
    }
    if (R_out) memcpy(R_out, R, hc_dim * sizeof(float));
    if (logits || mixed_out) {
        q36_q4_hc_mix(m, R, w->output_hc_norm, w->output_hc_down, w->output_hc_up, NULL, mixed, NULL);
        if (mixed_out) memcpy(mixed_out,mixed,E*sizeof(float));
        if (logits) q36_q4_output(m, w, mixed, logits);
    }
    free(mixed);
    free(R);
    return true;
}

static bool q36_q4_forward_token(const q36_model *m, const q36_weights *w, q36_q4_state *st,
    int token, uint32_t pos, float *logits, float *R_out, float *streams) {
    return q36_q4_forward_slice_token(m,w,st,token,pos,0,Q36_N_LAYER-1,NULL,
                                     logits,R_out,NULL,streams);
}


#include "q36_qwen4_gpu.inc"

static void q36_q4_gpu_release(void *gpu) {
#if !defined(Q36_NO_GPU) && !defined(Q36_METAL)
    q36_q4_gpu_free(gpu);
#else
    (void)gpu;
#endif
}

static bool q36_q4_reserve(q36_q4_state *st, uint32_t need) {
    if (need <= st->cap) return true;
    uint32_t cap = need < st->cap * 2 ? st->cap * 2 : need;
#if !defined(Q36_NO_GPU) && !defined(Q36_METAL)
    if (st->gpu) {
        const char *reuse=getenv("Q36_VK_FLASH_RESET_REUSE");
        if (!reuse || strcmp(reuse,"0")) {
            /* The Vulkan runtime owns these KV rows. CPU mirrors are unused. */
            st->pos3=xrealloc(st->pos3,(uint64_t)cap*3*sizeof(uint32_t));
            st->cap=cap;
            return true;
        }
    }
#endif
    uint64_t layers = Q36_N_LAYER / 4;
    float **arrays[] = {&st->attn_k, &st->attn_v, &st->idx_k};
    uint32_t widths[] = {Q36_N_HEAD_KV * Q36_N_HEAD_DIM, Q36_N_HEAD_KV * Q36_N_HEAD_DIM, 128};
    for (uint32_t a = 0; a < 3; a++) {
        float *data = xcalloc(layers * cap * widths[a], sizeof(float));
        for (uint32_t l = 0; l < layers; l++)
            memcpy(data + (uint64_t)l * cap * widths[a],
                   *arrays[a] + (uint64_t)l * st->cap * widths[a],
                   (uint64_t)st->cap * widths[a] * sizeof(float));
        free(*arrays[a]);
        *arrays[a] = data;
    }
    st->pos3 = xrealloc(st->pos3, (uint64_t)cap * 3 * sizeof(uint32_t));
    st->cap = cap;
    return true;
}

static bool q36_q4_forward(q36_session *s, int token, uint32_t pos, bool logits) {
    if (token < 0 || (uint32_t)token >= Q36_N_VOCAB || pos >= (uint32_t)s->ctx_size) return false;
#if !defined(Q36_NO_GPU) && !defined(Q36_METAL)
    if (s->engine->backend == Q36_BACKEND_VULKAN) return q36_q4_gpu_forward(s,token,pos,logits);
#endif
    if (!q36_q4_forward_token(&s->engine->model, &s->engine->weights, s->runtime,
                         token, pos, logits ? s->logits : NULL, NULL, NULL)) return false;
    if (logits) {
        s->logits_host_valid = q36_all_finite(s->logits, Q36_N_VOCAB);
        s->gpu_top2_valid = false;
        return s->logits_host_valid;
    }
    return true;
}
