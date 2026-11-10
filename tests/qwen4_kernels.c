#include "../q36.c"

static bool resident_stream_errors(void) {
    uint8_t map[64] = {0}, frame[32] = {0};
    q36_gpu_tensor *out = q36_gpu_tensor_alloc(16);
    bool ok = out != NULL;
    for (uint32_t test = 0; ok && test < 4; test++) {
        memset(frame,0,sizeof(frame));
        frame[7] = test == 1 ? 16 : 8;
        frame[15] = test == 2 ? 20 : 16;
        FILE *file = tmpfile();
        if (!file) { ok = false; break; }
        uint32_t bytes = test == 0 ? 8 : test == 3 ? 20 : 16;
        ok = fwrite(frame,1,bytes,file) == bytes;
        rewind(file);
        ok = ok && !q36_gpu_cache_model_range_from_fd(map,sizeof(map),8,16,fileno(file)) &&
                   !q36_gpu_cached_copy(out,0,map,sizeof(map),8,16) &&
                   !q36_gpu_cached_tensor(map,sizeof(map),8,16);
        fclose(file);
    }
    q36_gpu_tensor_free(out);
    fprintf(stderr,"Resident stream rejects truncated/mismatched frames without publishing weights: %s\n",ok ? "OK" : "FAIL");
    return ok;
}

static bool suffix_cache_growth(void) {
    q36_q4_gpu *g = xcalloc(1,sizeof(*g));
    g->first = 26; g->last = 47;
    bool ok = q36_q4_gpu_reserve(g,2,0);
    float want[1024], got[2560];
    uint32_t widths[] = {512,512,128};
    for (uint32_t il = 3; ok && il < 48; il += 4) {
        if (il < g->first) {
            ok = !g->k[il] && !g->value[il] && !g->index[il];
            continue;
        }
        q36_gpu_tensor *buffers[] = {g->k[il],g->value[il],g->index[il]};
        for (uint32_t b = 0; ok && b < 3; b++) {
            for (uint32_t i = 0; i < 2*widths[b]; i++) want[i] = il+b+i*.125f;
            ok = q36_gpu_tensor_write(buffers[b],0,want,2*widths[b]*4);
        }
    }
    ok = ok && q36_q4_gpu_reserve(g,5,2);
    for (uint32_t il = 27; ok && il < 48; il += 4) {
        q36_gpu_tensor *buffers[] = {g->k[il],g->value[il],g->index[il]};
        for (uint32_t b = 0; ok && b < 3; b++) {
            for (uint32_t i = 0; i < 2*widths[b]; i++) want[i] = il+b+i*.125f;
            ok = q36_gpu_tensor_read(buffers[b],0,got,(b==0 ? 5 : 2)*widths[b]*4);
            if (b==0) {
                for (uint32_t j=0; ok && j<512; j++) for (uint32_t t=0; ok && t<2; t++)
                    ok=!memcmp(want+j*2+t,got+j*5+t,4);
            } else ok=ok && !memcmp(want,got,2*widths[b]*4);
        }
    }
    q36_q4_gpu_free(g);
    fprintf(stderr,"Suffix-only KV/index cache growth preserves existing rows: %s\n",ok ? "OK" : "FAIL");
    return ok;
}

static bool sparse_scores(void) {
    const uint32_t blocks = 513, cap = blocks * 4 + 2;
    float *keys = xmalloc((uint64_t)cap * 128 * 4);
    float query[512], gamma[128], want[513], got[513];
    for (uint32_t t = 0; t < cap; t++) for (uint32_t j = 0; j < 128; j++)
        keys[t * 128 + j] = sinf(t * 0.111f + j * 0.17f);
    for (uint32_t i = 0; i < 512; i++) query[i] = cosf(i * 0.13f);
    for (uint32_t i = 0; i < 128; i++) gamma[i] = 1;
    q36_gpu_tensor *k = q36_gpu_tensor_alloc((uint64_t)cap * 128 * 4);
    q36_gpu_tensor *q = q36_gpu_tensor_alloc(512 * 4), *raw = q36_gpu_tensor_alloc(512 * 4);
    q36_gpu_tensor *norm = q36_gpu_tensor_alloc(128 * 4), *score = q36_gpu_tensor_alloc(blocks * 4);
    bool ok = k && q && raw && norm && score &&
        q36_gpu_tensor_write(k,0,keys,(uint64_t)cap * 128 * 4) &&
        q36_gpu_tensor_write(raw,0,query,sizeof(query)) && q36_gpu_tensor_write(norm,0,gamma,sizeof(gamma));
    if (ok) ok = q36_q4_gpu_op(NULL,11,128,4,q,raw,norm,NULL,NULL,NULL,0,cap-1,0,1) &&
        q36_q4_gpu_op(NULL,16,128,blocks,score,k,q,norm,NULL,NULL,blocks,0,0,0) &&
        q36_gpu_tensor_read(score,0,got,sizeof(got));
    float cached[513];
    if (ok) ok=q36_q4_gpu_op(NULL,24,128,500,k,k,NULL,norm,NULL,NULL,blocks,0,0,0) &&
        q36_q4_gpu_op(NULL,24,128,13,k,k,NULL,norm,NULL,NULL,blocks,0,500,0) &&
        q36_q4_gpu_op(NULL,16,128,blocks,score,k,q,norm,NULL,NULL,blocks,0,0,1) &&
        q36_gpu_tensor_read(score,0,cached,sizeof(cached)) && !memcmp(got,cached,sizeof(got));
    fprintf(stderr,"QSA pooled keys cached in place preserve all score bits: %s\n",ok ? "OK" : "FAIL");
    uint32_t qp[3] = {cap-1,cap-1,cap-1};
    for (uint32_t h = 0; h < 4; h++) {
        q36_q4_rms(query+h*128,query+h*128,gamma,128,1e-6f);
        q36_q4_rope(query+h*128,64,qp);
    }
    uint32_t min_cpu = 0, min_gpu = 0;
    float max = 0;
    for (uint32_t b = 0; ok && b < blocks; b++) {
        float pooled[128];
        for (uint32_t j = 0; j < 128; j++) {
            double sum = 0;
            for (uint32_t t = 0; t < 4; t++) sum += keys[(b*4+t)*128+j];
            pooled[j] = (float)(sum * 0.25);
        }
        q36_q4_rms(pooled,pooled,gamma,128,1e-6f);
        uint32_t kp[3] = {b*4,b*4,b*4};
        q36_q4_rope(pooled,64,kp);
        double sum = 0;
        for (uint32_t h = 0; h < 4; h++) {
            double dot = 0;
            for (uint32_t j = 0; j < 128; j++) dot += (double)query[h*128+j] * pooled[j];
            if (dot > 0) sum += dot;
        }
        want[b] = (float)sum;
        if (!isfinite(got[b]) || fabsf(got[b]-want[b]) > 0.002f) ok = false;
        if (fabsf(got[b]-want[b]) > max) max = fabsf(got[b]-want[b]);
        if (want[b] < want[min_cpu]) min_cpu = b;
        if (got[b] < got[min_gpu]) min_gpu = b;
    }
    ok = ok && min_cpu == min_gpu;
    fprintf(stderr,"QSA 513 blocks: max_abs %.9g, excluded block %u/%u %s\n",max,min_cpu,min_gpu,ok ? "OK" : "FAIL");
    q36_gpu_tensor_free(k); q36_gpu_tensor_free(q); q36_gpu_tensor_free(raw);
    q36_gpu_tensor_free(norm); q36_gpu_tensor_free(score); free(keys);
    return ok;
}

static bool sparse_selection(void) {
    const uint32_t sizes[] = {512,513,1024,2048,4096};
    q36_gpu_tensor *input=q36_gpu_tensor_alloc(16384), *output=q36_gpu_tensor_alloc(65536);
    if (!input || !output) return false;
    uint32_t seed = 1;
    for (uint32_t k = 0; k < sizeof(sizes)/sizeof(*sizes); k++) for (uint32_t mode = 0; mode < 7; mode++) {
        uint32_t n = sizes[k];
        float *scores = xmalloc(n*4);
        bool *want = xcalloc(n,sizeof(bool)), *got = xcalloc(n,sizeof(bool));
        for (uint32_t i = 0; i < n; i++) {
            seed = seed*1664525u + 1013904223u;
            scores[i] = mode == 0 ? 0 : mode == 1 ? i : mode == 2 ? n-i :
                        mode == 3 ? (seed%17) : mode==4 ? (float)(int32_t)seed :
                        mode==5 ? (i%2 ? -0.0f : 0.0f) : (i%3 ? INFINITY : -INFINITY);
        }
        for (uint32_t i = 0; i < 512; i++) {
            uint32_t best = UINT32_MAX;
            for (uint32_t j = 0; j < n; j++)
                if (!want[j] && (best == UINT32_MAX || scores[j] > scores[best])) best = j;
            want[best] = true;
        }
        q36_q4_gpu_select_blocks(scores,n,got);
        bool ok = memcmp(want,got,n*sizeof(bool)) == 0;
        for (uint32_t radix=0; ok && radix<2; radix++) {
            if (n>2048 && !radix) continue;
            uint32_t selected[2052], expected[2052], count=0, tail=mode%4, length=0;
            for (uint32_t i=0; i<n; i++) if (want[i])
                for (uint32_t j=0; j<4; j++) expected[count++]=i*4+j;
            for (uint32_t i=0; i<tail; i++) expected[count++]=n*4+i;
            uint32_t p[8]={28,n,0,n*4+tail-1,0,0,0,radix};
            q36_gpu_tensor *t[6]={output,input,output,output,output,output};
            ok=q36_gpu_tensor_write(input,0,scores,n*4) && q36_gpu_qwen4_op(t,p,1) &&
                q36_gpu_tensor_read(output,0,selected,count*4) &&
                q36_gpu_tensor_read(output,4096*4,&length,4) && length==count &&
                !memcmp(selected,expected,count*4);
        }
        free(scores); free(want); free(got);
        if (!ok) return false;
    }
    q36_gpu_tensor_free(input); q36_gpu_tensor_free(output);
    fprintf(stderr,"QSA heap and GPU selection match serial selection, including ties and tails: OK\n");
    return true;
}

static bool streamed_head(const q36_model *m) {
    const q36_tensor *t = model_find_tensor(m,"output.weight");
    const uint32_t rows = 65539;
    if (!t || t->type != Q36_TENSOR_Q6_K || t->dim[1] < rows) return false;
    uint32_t width = t->dim[0];
    float *x = xmalloc(width*4), *got = xmalloc(rows*4);
    for (uint32_t i = 0; i < width; i++) x[i] = sinf(i*0.13f)+(i%7)*0.1f;
    q36_gpu_tensor *input = q36_gpu_tensor_alloc(width*4), *output = q36_gpu_tensor_alloc(rows*4);
    uint64_t stride;
    bool ok = input && output && tensor_nbytes(t->type,width,&stride) &&
        q36_gpu_tensor_write(input,0,x,width*4) &&
        q36_gpu_stream_matvec(output,input,m->fd,t->abs_offset,t->type,width,rows,stride) &&
        q36_gpu_tensor_read(output,0,got,rows*4);
    uint32_t sample[] = {0,1,3989,3991,3992,32768,65537,65538};
    float max = 0;
    for (uint32_t i = 0; ok && i < rows; i++) if (!isfinite(got[i])) ok = false;
    for (uint32_t i = 0; ok && i < sizeof(sample)/sizeof(*sample); i++) {
        float want;
        q36_q4_matvec_rows(m,t,sample[i],1,x,&want);
        float error = fabsf(got[sample[i]]-want);
        if (error > max) max = error;
        if (error > 0.0003f+fabsf(want)*0.00001f) ok = false;
    }
    fprintf(stderr,"Q6 streamed head, bank reuse and odd tail: max_abs %.9g %s\n",max,ok ? "OK" : "FAIL");
    q36_gpu_tensor_free(input); q36_gpu_tensor_free(output);
    free(x); free(got);
    return ok;
}

static bool batched_layers(q36_model *m) {
    const char *setting=getenv("Q36_VK_FLASH_OPS");
    char *saved=setting ? strdup(setting) : NULL;
    q36_engine engine={0};
    engine.model=*m; engine.backend=Q36_BACKEND_VULKAN;
    engine.distributed.role=Q36_DISTRIBUTED_WORKER;
    q36_q4_bind(&engine.weights,m);
    q36_q4_state state={0}; state.cap=64;
    q36_session session={0}; session.engine=&engine; session.runtime=&state;
    const uint32_t layers[]={0,3,11};
    int tokens[33]; for (uint32_t i=0; i<33; i++) tokens[i]=11+i;
    float *input=xmalloc(33u*10240*4), *before=xmalloc(33u*10240*4), *after=xmalloc(33u*10240*4);
    for (uint32_t i=0; i<33u*10240; i++) input[i]=sinf(i*.019f)+(i%11)*.03f;
    bool ok=true;
    for (uint32_t k=0; ok && k<3; k++) {
        uint32_t il=layers[k]; engine.layer_start=engine.layer_end=il;
        char prefix[64]; size_t len=snprintf(prefix,sizeof(prefix),"blk.%u.",il);
        for (uint64_t i=0; ok && i<m->n_tensors; i++) {
            const q36_tensor *t=m->tensors+i;
            if (t->name.len>=len && !memcmp(t->name.ptr,prefix,len))
                ok=q36_gpu_cache_model_range(m->map,m->size,t->abs_offset,t->bytes,NULL);
        }
        for (uint32_t mode=0; ok && mode<4; mode++) {
            setenv("Q36_VK_FLASH_OPS",mode ? "1" : "0",1);
            ok=q36_q4_gpu_init(&state,&engine);
            if (!ok) break;
            uint32_t count=mode==0 ? 1 : mode==1 ? 8 : mode==2 ? 24 : 32;
            for (uint32_t t=0; ok && t<33; ) {
                uint32_t n=33-t<count ? 33-t : count;
                if (n>1 && t+n==33) n--;
                ok=q36_q4_gpu_forward_batch(&session,tokens+t,n,t,false,il,il,
                        input+(uint64_t)t*10240,(mode ? after : before)+(uint64_t)t*10240);
                t+=n;
            }
            q36_q4_gpu_free(state.gpu); state.gpu=NULL;
            if (mode && ok) ok=!memcmp(before,after,33u*10240*4);
        }
        float max=0;
        for (uint32_t i=0; i<33u*10240; i++) if (fabsf(before[i]-after[i])>max) max=fabsf(before[i]-after[i]);
        ok=ok && !memcmp(before,after,33u*10240*4);
        fprintf(stderr,"Batched layer %u including following decode: max_abs %.9g %s\n",il,max,ok ? "OK" : "FAIL");
    }
    free(input); free(before); free(after);
    if (saved) { setenv("Q36_VK_FLASH_OPS",saved,1); free(saved); }
    else unsetenv("Q36_VK_FLASH_OPS");
    return ok;
}

static bool batched_matrices(q36_model *m) {
    bool seen[43][4] = {{0}};
    q36_gpu_tensor *input = q36_gpu_tensor_alloc(524288), *output = q36_gpu_tensor_alloc(524288);
    q36_gpu_tensor *single = q36_gpu_tensor_alloc(128);
    bool ok = input && output && single;
    for (uint64_t j = 0; ok && j < m->n_tensors; j++) {
        const q36_tensor *t = m->tensors+j;
        if (t->ndim < 2 || t->type > 42 || t->dim[0] > 10240) continue;
        uint32_t group = t->dim[0] < 1024 ? 0 : t->dim[0] < 4096 ? 1 : t->dim[0] < 8192 ? 2 : 3;
        if (seen[t->type][group]) continue;
        seen[t->type][group] = true;
        uint32_t width = t->dim[0], rows = t->dim[1] < 32 ? t->dim[1] : 32;
        uint64_t stride=0; float reference[8][32], got[32];
        float *x = xmalloc((uint64_t)width*4);
        ok = tensor_nbytes(t->type,width,&stride) && q36_gpu_cache_model_range(m->map,m->size,t->abs_offset,rows*stride,NULL);
        for (uint32_t k = 0; ok && k < 8; k++) {
            for (uint32_t i = 0; i < width; i++) x[i] = sinf(i*.13f+k*.21f)+(i%7)*.1f;
            q36_gpu_tensor *view = q36_gpu_tensor_view(input,(uint64_t)k*65536,(uint64_t)width*4);
            ok = view && q36_gpu_tensor_write(view,0,x,width*4) &&
                q36_gpu_cached_matvec(single,view,m->map,m->size,t->abs_offset,rows*stride,
                    0,t->type,width,rows,stride,NULL) && q36_gpu_tensor_read(single,0,reference[k],rows*4);
            q36_gpu_tensor_free(view);
        }
        ok = ok && q36_gpu_cached_matmul(output,input,m->map,m->size,t->abs_offset,rows*stride,
                0,t->type,width,rows,stride,8,65536,NULL);
        float max = 0;
        for (uint32_t k = 0; ok && k < 8; k++) {
            ok = q36_gpu_tensor_read(output,(uint64_t)k*65536,got,rows*4);
            for (uint32_t i = 0; i < rows; i++) if (fabsf(got[i]-reference[k][i]) > max) max=fabsf(got[i]-reference[k][i]);
            ok = ok && !memcmp(got,reference[k],rows*4);
        }
        fprintf(stderr,"Batch %s %ux%u: max_abs %.9g %s\n",tensor_type_name(t->type),width,rows,max,ok ? "OK" : "FAIL");
        free(x);
    }
    q36_gpu_tensor_free(input); q36_gpu_tensor_free(output); q36_gpu_tensor_free(single);
    return ok;
}

static bool gpu_router(void) {
    q36_gpu_tensor *input=q36_gpu_tensor_alloc(8*65536), *output=q36_gpu_tensor_alloc(8*65536);
    float x[512], values[10]; uint32_t ids[10], meta[74];
    bool ok=input && output;
    for (uint32_t pass=0; ok && pass<8; pass++) {
        uint32_t n=pass%2 ? 512 : 256;
        for (uint32_t row=0; ok && row<8; row++) {
            for (uint32_t i=0; i<n; i++) x[i]=pass==0 ? 0 : sinf(i*.13f+pass+row)*8+(i%7)*.1f;
            ok=q36_gpu_tensor_write(input,row*65536,x,n*4);
        }
        uint32_t p[8]={23,n,0,0,3};
        q36_gpu_tensor *t[6]={output,input,output,output,output,output};
        ok=ok && q36_gpu_qwen4_op_rows(t,p,1,8);
        for (uint32_t row=0; ok && row<8; row++) {
            for (uint32_t i=0; i<n; i++) x[i]=pass==0 ? 0 : sinf(i*.13f+pass+row)*8+(i%7)*.1f;
            q36_topk_indices(x,n,10,ids,values); q36_softmax_inplace(values,10);
            ok=q36_gpu_tensor_read(output,row*65536,meta,sizeof(meta));
            for (uint32_t i=0; ok && i<10; i++) {
                float weight; memcpy(&weight,meta+64+i,4);
                ok=ids[i]==meta[i] && !memcmp(values+i,&weight,4);
                if (!ok) fprintf(stderr,"router pass=%u row=%u lane=%u id=%u/%u weight=%.9g/%.9g\n",
                                 pass,row,i,ids[i],meta[i],values[i],weight);
            }
        }
    }
    fprintf(stderr,"GPU router: ordered ties and 64 batched softmax cases %s\n",ok ? "OK" : "FAIL");
    q36_gpu_tensor_free(input); q36_gpu_tensor_free(output);
    return ok;
}

static bool routed_experts(q36_model *m) {
    q36_engine engine = {0};
    q36_session session = {0};
    session.engine = &engine;
    q36_q4_gpu *g = xcalloc(1,sizeof(*g));
    g->resident = true;
    for (uint32_t i = 0; i < Q4_NBUF; i++)
        g->v[i] = q36_gpu_tensor_alloc(i == Q4_BLOCK ? 131072 : 65536);
    float x[2560], before[2560], after[2560];
    const uint32_t layers[] = {0,4,15,11,19,24};
    bool ok = true;
    for (uint32_t j = 0; ok && j < 6; j++) {
        q36_layer_weights l = {0};
        q36_tensor **fields[] = {&l.ffn_gate_exps,&l.ffn_up_exps,&l.ffn_down_exps,
            &l.ffn_gate_inp,&l.ffn_gate_inp_shexp,&l.ffn_gate_shexp,&l.ffn_up_shexp,&l.ffn_down_shexp};
        const char *names[] = {"ffn_gate_exps","ffn_up_exps","ffn_down_exps","ffn_gate_inp",
            "ffn_gate_inp_shexp","ffn_gate_shexp","ffn_up_shexp","ffn_down_shexp"};
        for (uint32_t i = 0; ok && i < 8; i++) {
            char name[128]; snprintf(name,sizeof(name),"blk.%u.%s.weight",layers[j],names[i]);
            *fields[i] = model_find_tensor(m,name);
            const q36_tensor *t = *fields[i];
            ok = t && q36_gpu_cache_model_range(m->map,m->size,t->abs_offset,t->bytes,NULL);
        }
        for (uint32_t pass = 0; ok && pass < 3; pass++) {
            for (uint32_t i = 0; i < 2560; i++) x[i] = sinf(i*.13f+pass) + (i%7)*.1f;
            setenv("Q36_VK_FLASH_EXPERTS","0",1);
            ok = q36_gpu_tensor_write(g->v[Q4_X],0,x,sizeof(x)) && q36_q4_gpu_moe(g,m,&l,&session,layers[j]) &&
                 q36_gpu_tensor_read(g->v[Q4_BLOCK],0,before,sizeof(before));
            setenv("Q36_VK_FLASH_EXPERTS","1",1);
            ok = ok && q36_gpu_tensor_write(g->v[Q4_X],0,x,sizeof(x)) && q36_q4_gpu_moe(g,m,&l,&session,layers[j]) &&
                 q36_gpu_tensor_read(g->v[Q4_BLOCK],0,after,sizeof(after));
            float maximum = 0;
            for (uint32_t i = 0; i < 2560; i++) if (fabsf(before[i]-after[i]) > maximum) maximum=fabsf(before[i]-after[i]);
            ok = ok && memcmp(before,after,sizeof(before)) == 0;
            fprintf(stderr,"Routed experts layer %u input %u max_abs %.9g %s\n",layers[j],pass,maximum,ok ? "OK" : "FAIL");
        }
    }
    unsetenv("Q36_VK_FLASH_EXPERTS");
    q36_q4_gpu_free(g);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 2) return 1;
    q36_model m;
    model_open(&m,argv[1],false);
    config_validate_model(&m);
    if (!q36_gpu_init()) return 1;
    bool seen[43][4] = {{0}};
    uint32_t tested = 0;
    for (uint64_t j = 0; j < m.n_tensors; j++) {
        const q36_tensor *t = m.tensors + j;
        if (t->ndim < 2 || t->type > 42 || t->dim[0] > 10240) continue;
        uint32_t group = t->dim[0] < 1024 ? 0 : t->dim[0] < 4096 ? 1 : t->dim[0] < 8192 ? 2 : 3;
        if (seen[t->type][group]) continue;
        seen[t->type][group] = true;
        bool tiled = t->type == Q36_TENSOR_IQ4_XS || (t->type >= 10 && t->type <= 14);
        for (uint32_t pass = 0; pass < (tiled ? 2u : 1u); pass++) {
        uint32_t limit = pass ? 8 : 7;
        uint32_t n = t->dim[0], rows = t->dim[1] < limit ? t->dim[1] : limit;
        float *x = xmalloc(n * 4), *cpu = xmalloc(rows * 4), *gpu = xmalloc(rows * 4);
        for (uint32_t i = 0; i < n; i++) x[i] = sinf(i * 0.13f) + (int)(i % 7) * 0.1f;
        q36_q4_matvec_rows(&m,t,0,rows,x,cpu);
        q36_gpu_tensor *input = q36_gpu_tensor_alloc(n * 4), *output = q36_gpu_tensor_alloc(rows * 4);
        uint64_t stride = 0;
        bool ok = input && output && tensor_nbytes(t->type,n,&stride) &&
            q36_gpu_tensor_write(input,0,x,n * 4) &&
            q36_gpu_stream_matvec(output,input,m.fd,t->abs_offset,t->type,n,rows,stride) &&
            q36_gpu_tensor_read(output,0,gpu,rows * 4);
        float *cached = xmalloc(rows*4);
        ok = ok && q36_gpu_cache_model_range(m.map,m.size,t->abs_offset,rows*stride,NULL) &&
            q36_gpu_cached_matvec(output,input,m.map,m.size,t->abs_offset,rows*stride,
                                  0,t->type,n,rows,stride,NULL) &&
            q36_gpu_tensor_read(output,0,cached,rows*4);
        if (ok && n % 256 == 0) {
            q36_gpu_tensor *packed = q36_gpu_tensor_alloc((uint64_t)(n/256)*296);
            ok = packed && q36_gpu_quantize_q8_k_tensor(packed,input,n,1) &&
                q36_gpu_cached_matvec(output,input,m.map,m.size,t->abs_offset,rows*stride,
                                      0,t->type,n,rows,stride,packed) &&
                q36_gpu_tensor_read(output,0,cached,rows*4);
            for (uint32_t i = 0; ok && i < rows; i++)
                if (memcmp(gpu+i,cached+i,sizeof(float))) ok = false;
            q36_gpu_tensor_free(packed);
        }
        FILE *frame = tmpfile();
        uint8_t header[16];
        uint64_t fields[] = {t->abs_offset,rows*stride};
        for (uint32_t f = 0; f < 2; f++) for (uint32_t b = 0; b < 8; b++)
            header[f*8+b] = fields[f] >> (56-b*8);
        ok = ok && frame && fwrite(header,1,16,frame) == 16 &&
            fwrite(m.map+t->abs_offset,1,rows*stride,frame) == rows*stride;
        if (frame) rewind(frame);
        ok = ok && q36_gpu_cache_model_range_from_fd(m.map,m.size,t->abs_offset,rows*stride,fileno(frame)) &&
            q36_gpu_cached_matvec(output,input,m.map,m.size,t->abs_offset,rows*stride,
                                  0,t->type,n,rows,stride,NULL) &&
            q36_gpu_tensor_read(output,0,cached,rows*4);
        if (frame) fclose(frame);
        float max = 0;
        for (uint32_t i = 0; ok && i < rows; i++) {
            float error = fabsf(cpu[i] - gpu[i]);
            if (!isfinite(gpu[i]) || error > 0.0003f + fabsf(cpu[i]) * 0.00001f) ok = false;
            if (memcmp(gpu+i,cached+i,sizeof(float))) ok = false;
            if (error > max) max = error;
        }
        fprintf(stderr,"%s %ux%u: max_abs %.9g %s\n",tensor_type_name(t->type),n,rows,max,ok ? "OK" : "FAIL");
        q36_gpu_tensor_free(input); q36_gpu_tensor_free(output);
        free(x); free(cpu); free(gpu); free(cached);
        if (!ok) return 1;
        tested++;
        }
    }
    if (!gpu_router() || !batched_layers(&m) || !batched_matrices(&m) || !routed_experts(&m) || !sparse_selection() || !streamed_head(&m) || !sparse_scores() || !resident_stream_errors() || !suffix_cache_growth()) return 1;
    q36_gpu_cleanup(); model_close(&m);
    printf("%u streamed matrix type/shape comparisons passed\n",tested);
    return 0;
}
