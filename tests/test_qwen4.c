#include "../q36.c"

static void check(bool ok, const char *what) {
    if (!ok) { fprintf(stderr,"FAIL: %s\n",what); exit(1); }
}

static void test_q2_0(void) {
    uint8_t block[18];
    uint16_t scale = q36_f32_to_f16(0.5f);
    memcpy(block,&scale,2);
    memset(block + 2,0xe4,16);
    float decoded[64];
    check(q36_dequantize_row_from_ptr(Q36_TENSOR_Q2_0,block,decoded,64),"Q2_0 decode");
    for (uint32_t i = 0; i < 64; i++) check(decoded[i] == ((int)(i % 4) - 1) * 0.5f,"Q2_0 packing order");
    uint64_t bytes;
    check(tensor_nbytes(Q36_TENSOR_Q2_0,640,&bytes) && bytes == 180,"640-column Q2_0 row");
    check(!q36_dequantize_row_from_ptr(Q36_TENSOR_Q2_0,block,decoded,63),"Q2_0 partial block rejection");
}

static void test_activation_rounding(void) {
    float x[32];
    q36_block_q8_0 cache, activation;
    for (uint32_t i = 0; i < 32; i++)
        x[i] = (float)((int)(((352 + i) * 19 + 3) % 89) - 44) / 29.0f;
    q36_quant_q8_0(x,&cache,32);
    q36_quant_q8_0_activation(x,&activation,32);
    check(cache.d == 0x21d7 && activation.d == cache.d,"Q8 scale rounding");
    check(cache.qs[26] == 63,"existing KV-cache byte format");
    check(activation.qs[26] == 64,"GGML activation rounding boundary");
    q36_block_q8_0 pipeline;
    check(q36_quantize_activation_row(Q36_ACTIVATION_QUANT_Q8_0,x,(uint8_t *)&pipeline,32),
          "CPU activation quantization");
    check(!memcmp(&activation,&pipeline,sizeof(pipeline)),"shared CPU path uses activation rounding");
}

static void test_gdn_norm(void) {
    check(q36_softplusf(-50) > 0 &&
          fabsf(q36_softplusf(-50) / expf(-50) - 1) < 1e-6f,
          "softplus retains small positive gates");
    const float amplitudes[] = {0,1e-7f,1e-3f,1};
    for (uint32_t j = 0; j < sizeof(amplitudes)/sizeof(*amplitudes); j++) {
        float x[128];
        double ss = 0, norm = 0;
        for (uint32_t i = 0; i < 128; i++) {
            x[i] = amplitudes[j] * ((int)(i % 7) - 3);
            ss += (double)x[i] * x[i];
        }
        q36_l2_norm(x,128,1e-6f);
        for (uint32_t i = 0; i < 128; i++) {
            check(isfinite(x[i]),"GDN norm is finite for zero and small inputs");
            norm += (double)x[i] * x[i];
        }
        double expected = ss / (ss + 1e-6);
        check(fabs(norm - expected) <= 1e-6 * expected + 1e-12,
              "GDN epsilon belongs inside the square root");
    }
}

static void test_bf16_activation(void) {
    uint16_t weights[33];
    float input[33];
    for (uint32_t i = 0; i < 33; i++) {
        weights[i] = 0x3f80;
        input[i] = 1.0f + 1.0f / 256;
    }
    check(q36_quant_dot_bf16(weights,input,33) == 33,
          "BF16 activations round ties to even, including the scalar tail");
    for (uint32_t i = 0; i < 33; i++) input[i] = 1.0f + 3.0f / 256;
    check(q36_quant_dot_bf16(weights,input,33) == 33.0f * (1.0f + 1.0f / 64),
          "BF16 activation rounding carries into the retained mantissa");
}

static void test_rms_norm(void) {
    float x[5120], out[5120];
    for (uint32_t i = 0; i < 5120; i++) x[i] = 1.0001f;
    q36_ref_rms_norm(out,x,NULL,5120,1e-6f);
    long double value = x[0];
    float expected = (float)(value / sqrtl(value * value + 1e-6L));
    for (uint32_t i = 0; i < 5120; i++)
        check(fabsf(out[i] - expected) <= 2e-7f,"RMS norm accumulation at dense-model width");
}

static void test_dense_mixed_ffn(void) {
    enum {width=256, rows=3};
    q36_shape saved = g_q36_shape;
    g_q36_shape = Q36_SHAPE_27B;
    g_q36_shape.n_embd = g_q36_shape.n_ff_shared = width;
    size_t gate_bytes = width * 8 * sizeof(q36_block_iq4_nl);
    size_t up_bytes = width * sizeof(q36_block_q5_k);
    size_t bytes = gate_bytes + up_bytes + width * width * sizeof(float);
    uint8_t *map = xcalloc(1,bytes);
    q36_tensor gate = {.type=Q36_TENSOR_IQ4_NL,.ndim=2,.dim={width,width},.bytes=gate_bytes};
    q36_tensor up = {.type=Q36_TENSOR_Q5_K,.ndim=2,.dim={width,width},.bytes=up_bytes,.abs_offset=gate_bytes};
    q36_tensor down = {.type=Q36_TENSOR_F32,.ndim=2,.dim={width,width},
        .bytes=width*width*sizeof(float),.abs_offset=gate_bytes+up_bytes};
    for (unsigned i = 0; i < width*8; i++) {
        q36_block_iq4_nl *b = (q36_block_iq4_nl *)map+i;
        b->d = q36_f32_to_f16(.001f);
        memset(b->qs,0xa9,sizeof(b->qs));
    }
    for (unsigned i = 0; i < width; i++) {
        q36_block_q5_k *b = (q36_block_q5_k *)(map+gate_bytes)+i;
        b->d = q36_f32_to_f16(.002f);
        memset(b->scales,1,sizeof(b->scales));
        memset(b->qs,0x32,sizeof(b->qs));
        ((float *)(map+down.abs_offset))[i*width+i] = 1;
    }
    q36_engine e = {.model={.map=map,.size=bytes},.n_threads=1};
    float input[rows*width], output[rows*width], want[width], a[width], b[width];
    float work[4][rows*width];
    uint8_t xq[rows*sizeof(q36_block_q8_k)];
    q36_cpu_runtime rt = {.work2=work[0],.work3=work[1],.work4=work[2],.work5=work[3],
        .batch_ffn_shared_gate=work[0],.batch_ffn_shared_up=work[1],
        .batch_ffn_shared_mid=work[2],.batch_xq=xq};
    for (unsigned i = 0; i < rows*width; i++) input[i] = sinf(i*.03f)*.1f;
    q36_tensor *gates[] = {&gate,&up,&up,&down};
    q36_tensor *ups[] = {&up,&gate,&up,&up};
    for (unsigned c = 0; c < 4; c++) {
        q36_layer_weights l = {.ffn_gate_shexp=gates[c],.ffn_up_shexp=ups[c],.ffn_down_shexp=&down};
        check(q36_forward_ffn_batch(&e,&l,input,output,rows,&rt),"mixed dense FFN prefill");
        for (unsigned t = 0; t < rows; t++) {
            check(q36_tensor_matvec(&e,gates[c],input+t*width,a,NULL,width,width) &&
                  q36_tensor_matvec(&e,ups[c],input+t*width,b,NULL,width,width),"independent FFN projections");
            for (unsigned i = 0; i < width; i++) {
                want[i] = q36_siluf(a[i])*b[i];
                check(isfinite(output[t*width+i]) && fabsf(output[t*width+i]-want[i]) < 1e-6f,
                      "dense prefill uses each projection's activation format");
            }
            check(q36_forward_ffn(&e,&l,input+t*width,output,&rt),"mixed dense FFN decode");
            for (unsigned i = 0; i < width; i++)
                check(isfinite(output[i]) && fabsf(output[i]-want[i]) < 1e-6f,
                      "dense decode uses each projection's activation format");
        }
    }
    free(map);
    g_q36_shape = saved;
}

static void test_vector_exp(void) {
    for (int i = -256; i <= 256; i++) {
        float x = (float)i * 0.3125f;
        float expected = expf(x);
        check(fabsf(q36_vector_expf(x) / expected - 1) < 3e-7f,
              "vector exponential F32 accuracy");
    }
    check(q36_vector_expf(-INFINITY) == 0 && isinf(q36_vector_expf(INFINITY)),
          "vector exponential range fallback");
}

static void test_f32_dot(void) {
    float w[64] = {0}, x[64];
    for (uint32_t i = 0; i < 64; i++) x[i] = 1;
    w[0] = 0x1p30f; w[16] = -0x1p30f; w[32] = 1;
    check(q36_dot_f32_strided(w,1,x,64) == 0,
          "F32 vector accumulation rounds before cancellation");
    w[1] = 1; w[2] = -0x1p30f; w[3] = 0;
    check(q36_dot_f32_strided(w,1,x,4) == 1,
          "scalar F32 tail retains the reference double sum");
}

static void test_softmax(void) {
    float x[257];
    for (uint32_t i = 0; i < 257; i++) x[i] = 1000;
    q36_softmax_inplace(x,257);
    double sum = 0;
    for (uint32_t i = 0; i < 257; i++) {
        check(fabsf(x[i] - 1.0f / 257) < 1e-8f,"uniform softmax, including its scalar tail");
        sum += x[i];
    }
    check(fabs(sum - 1) < 1e-7,"softmax probabilities sum to one");
    x[0] = -1000; x[1] = 0;
    q36_softmax_inplace(x,2);
    check(x[0] == 0 && x[1] == 1,"softmax remains finite across a large logit gap");
}

static void test_delta_state(void) {
    q36_shape saved = g_q36_shape;
    g_q36_shape.n_ssm_state = 128;
    g_q36_shape.n_ssm_dt_rank = 1;
    float state[128 * 128] = {0}, q[128] = {1}, k[128] = {1};
    float v[128], out[128], gate = -0.5f, beta = 0.5f;
    for (uint32_t i = 0; i < 128; i++) v[i] = ((int)i - 63) / 128.0f;
    for (uint32_t step = 0; step < 2; step++) {
        q36_delta_net_decode(state,q,k,v,&gate,&beta,out);
        double factor = step ? 0.5 + 0.25 * exp((double)gate) : 0.5;
        for (uint32_t i = 0; i < 128; i++) {
            double expected = factor * v[i] / sqrt(128.0);
            check(fabs(out[i] - expected) < 1e-7,"GDN analytic recurrence and query scale");
        }
        for (uint32_t i = 128; i < 128 * 128; i++)
            check(state[i] == 0,"GDN state keeps key/value orientation");
    }
    g_q36_shape = saved;
}

static void test_attention_head(void) {
    q36_shape saved = g_q36_shape;
    g_q36_shape.n_head = 2; g_q36_shape.n_head_kv = 1;
    g_q36_shape.n_head_dim = g_q36_shape.n_value_dim = 32;
    uint16_t keys[96] = {0}, values[96];
    q36_full_attn_cache cache = {.k = (uint8_t *)keys, .v = (uint8_t *)values,
        .type_k = Q36_KV_CACHE_F16, .type_v = Q36_KV_CACHE_F16,
        .k_row_bytes = 64, .v_row_bytes = 64, .len = 3, .cap = 3};
    float q[32] = {0}, gate[32] = {0}, out[32], scores[3], sinks[2] = {0};
    for (uint32_t i = 0; i < 96; i++)
        values[i] = q36_f32_to_f16(i < 32 ? 0.5f : i < 64 ? -0.25f : 4096);
    for (uint32_t h = 0; h < 2; h++) {
        q36_full_attn_head(out,scores,&cache,q,gate,h,1,NULL);
        for (uint32_t i = 0; i < 32; i++)
            check(out[i] == 0.0625f,"grouped attention ignores future cache rows and applies its gate");
        q36_full_attn_head(out,scores,&cache,q,gate,h,1,sinks);
        for (uint32_t i = 0; i < 32; i++)
            check(fabsf(out[i] - 1.0f / 24) < 1e-7f,"attention sink participates only in the denominator");
    }
    g_q36_shape = saved;
}

static void test_ngram(void) {
    g_q36_q4_ple.multiplier[0] = 3;
    g_q36_q4_ple.multiplier[1] = 5;
    g_q36_q4_ple.multiplier[2] = 7;
    for (uint32_t i = 0; i < 16; i++) {
        g_q36_q4_ple.head_vocab[i] = 101 + i;
        g_q36_q4_ple.head_offset[i] = i * 200;
    }
    int prev[3] = {12,248044,37};
    uint32_t rows[16];
    q36_q4_ple_step(11,prev,rows);
    for (uint32_t h = 0; h < 16; h++) {
        uint64_t hash = (11u * 3u) ^ (12u * 5u);
        if (h >= 8) hash ^= (uint64_t)248044 * 7;
        check(rows[h] == hash % (101 + h) + h * 200,"n-gram hash and EOS boundary");
    }
    check(prev[0] == 11 && prev[1] == 12 && prev[2] == 248044,"n-gram history advance");
    FILE *fp = tmpfile();
    check(fp != NULL,"temporary n-gram file");
    float values[160];
    for (uint32_t i = 0; i < 160; i++) values[i] = (float)i / 16;
    check(fwrite(values,sizeof(float),160,fp) == 160 && fflush(fp) == 0,"n-gram fixture write");
    q36_model m = {.ngram_fd = fileno(fp),.ngram = {.type = Q36_TENSOR_F32,.dim = {160,1}}};
    uint32_t id = 0;
    float result[160];
    check(q36_q4_ngram_read(&m,&id,1,result) && !memcmp(values,result,sizeof(values)),"disk n-gram gather");
    id = 1;
    check(!q36_q4_ngram_read(&m,&id,1,result),"n-gram row bounds");
    m.ngram.dim[1] = 2;
    check(!q36_q4_ngram_read(&m,&id,1,result),"truncated n-gram read");
    fclose(fp);
}

static void test_sparse_tail(void) {
    g_q36_shape = Q36_SHAPE_35B_A3B;
    g_q36_shape.n_embd = 2560;
    g_q36_shape.n_layer = 48;
    g_q36_shape.n_ssm_dt_rank = 48;
    q36_q4_state st;
    q36_q4_state_init(&st,2054);
    size_t bytes = 2560u * 512 * 4 + 512 * 4;
    float *map = xcalloc(1,bytes);
    q36_model m = {.map = (const uint8_t *)map,.size = bytes};
    q36_tensor q = {.type=Q36_TENSOR_F32,.ndim=2,.dim={2560,512},.elements=2560u * 512};
    q36_tensor k = q; k.dim[1] = 128; k.elements = 2560u * 128;
    q36_tensor norm = {.type=Q36_TENSOR_F32,.ndim=1,.dim={128},.elements=128,.abs_offset=2560u * 512 * 4};
    for (uint32_t i = 0; i < 128; i++) map[norm.abs_offset / 4 + i] = 1;
    q36_layer_weights l = {.indexer_q_proj=&q,.indexer_k_proj=&k,.indexer_q_norm=&norm,.indexer_k_norm=&norm};
    float input[2560] = {0};
    int32_t selected[2054];
    uint32_t count = q36_q4_select(&m,&l,3,&st,input,2053,selected);
    check(count == 2050,"sparse attention selection budget and tail");
    for (uint32_t i = 0; i < 2048; i++) check(selected[i] == (int32_t)i,"sparse attention stable ties");
    check(selected[2048] == 2052 && selected[2049] == 2053,"incomplete sparse attention block stays visible");
    check(q36_q4_reserve(&st,2055) && st.cap >= 2055,"attention cache growth");
    q36_q4_state_free(&st);
    free(map);
}

static void test_layer_slices(const char *path) {
    q36_model m;
    q36_weights w;
    model_open(&m,path,false);
    config_validate_model(&m);
    q36_q4_bind(&w,&m);
    q36_q4_open_ngrams(&m,path);
    g_q36_q4_threads = 16;
    g_q36_q4_streaming = true;
    q36_q4_state full, prefix, suffix;
    q36_q4_state_init(&full,2);
    q36_q4_state_init(&prefix,2);
    q36_q4_state_init(&suffix,2);
    float residual[10240];
    float *want = xmalloc(Q36_N_VOCAB*sizeof(float));
    float *got = xmalloc(Q36_N_VOCAB*sizeof(float));
    int tokens[] = {9707,11};
    for (uint32_t pos = 0; pos < 2; pos++) {
        check(q36_q4_forward_token(&m,&w,&full,tokens[pos],pos,want,NULL,NULL),"CPU reference token");
        check(q36_q4_forward_slice_token(&m,&w,&prefix,tokens[pos],pos,0,25,NULL,
                                        NULL,residual,NULL,NULL),"CPU prefix HC streams");
        check(q36_q4_forward_slice_token(&m,&w,&suffix,tokens[pos],pos,26,47,residual,
                                        got,NULL,NULL,NULL),"CPU suffix and output HC mix");
        check(q36_all_finite(got,Q36_N_VOCAB) &&
              !memcmp(want,got,Q36_N_VOCAB*sizeof(float)),"CPU layer-slice logits are bit identical");
    }
    q36_q4_state_free(&full); q36_q4_state_free(&prefix); q36_q4_state_free(&suffix);
    free(want); free(got); model_close(&m);
    puts("CPU HC layer slices, PLE and recurrent history: two bit-identical positions");
}

int main(int argc, char **argv) {
    test_q2_0(); test_activation_rounding(); test_bf16_activation(); test_gdn_norm(); test_rms_norm();
    test_vector_exp(); test_f32_dot(); test_softmax(); test_delta_state(); test_dense_mixed_ffn();
    test_attention_head(); test_ngram(); test_sparse_tail();
    puts("Quantization, GDN/RMS norms, n-grams and sparse attention: OK");
    if (argc == 2) test_layer_slices(argv[1]);
    return 0;
}
