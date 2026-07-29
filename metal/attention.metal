// Qwen3.6 full, compressed, split-K, and GQA-tiled attention kernels.

kernel void q36_attention_f16(
        device float *out [[buffer(0)]],
        device float *scores [[buffer(1)]],
        device const float *q [[buffer(2)]],
        device const float *qg [[buffer(3)]],
        device const half *kcache [[buffer(4)]],
        device const half *vcache [[buffer(5)]],
        device const float *sinks [[buffer(6)]],
        constant q36_attention_args &args [[buffer(7)]],
        uint2 gid [[thread_position_in_grid]]) {
    uint head = gid.x, tok = gid.y;
    if (head >= args.heads) return;
    uint count = args.pos0 + tok + 1u;
    uint group = args.heads / args.kv_heads;
    uint kvh = head / group;
    float scale = rsqrt(float(args.dim));
    device const float *qh =
        q + ((ulong)tok * args.heads + head) * args.dim;
    uint score_capacity = args.score_stride / args.heads;
    device float *sh =
        scores + (ulong)tok * args.score_stride +
        (ulong)head * score_capacity;
    float maxv = args.has_sinks ? sinks[head] : -INFINITY;
    for (uint pos = 0; pos < count; pos++) {
        device const half *kh =
            kcache + ((ulong)pos * args.kv_heads + kvh) * args.dim;
        float dotv = 0.0f;
        for (uint i = 0; i < args.dim; i++)
            dotv = fma(qh[i], float(kh[i]), dotv);
        dotv *= scale;
        sh[pos] = dotv;
        maxv = max(maxv, dotv);
    }
    float denom = args.has_sinks ? exp(sinks[head] - maxv) : 0.0f;
    for (uint pos = 0; pos < count; pos++) {
        sh[pos] = exp(sh[pos] - maxv);
        denom += sh[pos];
    }
    float inv = denom > 0.0f ? 1.0f / denom : 1.0f;
    device const float *gate =
        qg + (ulong)tok * args.qg_stride +
        (ulong)head * args.dim * 2u + args.dim;
    device float *oh =
        out + ((ulong)tok * args.heads + head) * args.dim;
    for (uint j = 0; j < args.dim; j++) {
        float value = 0.0f;
        for (uint pos = 0; pos < count; pos++) {
            device const half *vh =
                vcache + ((ulong)pos * args.kv_heads + kvh) * args.dim;
            value = fma(sh[pos] * inv, float(vh[j]), value);
        }
        float g = gate[j];
        float sigmoid = g >= 0.0f ? 1.0f / (1.0f + exp(-g))
                                  : exp(g) / (1.0f + exp(g));
        oh[j] = value * sigmoid;
    }
}

kernel void q36_attention_q8_q4_parallel(
        device float *out [[buffer(0)]],
        device float *scores [[buffer(1)]],
        device const float *q [[buffer(2)]],
        device const float *qg [[buffer(3)]],
        device const uchar *kcache [[buffer(4)]],
        device const uchar *vcache [[buffer(5)]],
        device const float *sinks [[buffer(6)]],
        constant q36_attention_args &args [[buffer(7)]],
        threadgroup float *shared [[threadgroup(0)]],
        uint2 group_id [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    uint head = group_id.x, tok = group_id.y;
    if (head >= args.heads) return;
    uint count = args.pos0 + tok + 1u;
    uint kvh = head / (args.heads / args.kv_heads);
    device const float *qh =
        q + ((ulong)tok * args.heads + head) * args.dim;
    uint score_capacity = args.score_stride / args.heads;
    device float *sh = scores + (ulong)tok * args.score_stride +
        (ulong)head * score_capacity;
    uint k_row_bytes = args.kv_heads * (args.dim >> 5) * 34u;
    uint v_row_bytes = args.kv_heads * (args.dim >> 5) * 18u;
    float scale = rsqrt(float(args.dim));

    for (uint pos = simdgroup; pos < count; pos += 8u) {
        device const uchar *kr = kcache + (ulong)pos * k_row_bytes;
        float dotv = 0.0f;
        uint base = kvh * args.dim;
        for (uint j = lane; j < args.dim; j += 32u)
            dotv = fma(qh[j], q36_kv_q8_value(kr, base + j), dotv);
        dotv = simd_sum(dotv) * scale;
        if (lane == 0u) sh[pos] = dotv;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    float local_max = -INFINITY;
    for (uint pos = tid; pos < count; pos += 256u)
        local_max = max(local_max, sh[pos]);
    local_max = simd_max(local_max);
    if (lane == 0u) shared[simdgroup] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float maxv = args.has_sinks ? sinks[head] : -INFINITY;
        for (uint sg = 0; sg < 8u; sg++) maxv = max(maxv, shared[sg]);
        shared[0] = maxv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float maxv = shared[0];
    float local_sum = 0.0f;
    for (uint pos = tid; pos < count; pos += 256u) {
        float value = exp(sh[pos] - maxv);
        sh[pos] = value;
        local_sum += value;
    }
    local_sum = simd_sum(local_sum);
    if (lane == 0u) shared[simdgroup] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float denom = args.has_sinks ? exp(sinks[head] - maxv) : 0.0f;
        for (uint sg = 0; sg < 8u; sg++) denom += shared[sg];
        shared[0] = denom > 0.0f ? 1.0f / denom : 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    if (tid < args.dim) {
        float value = 0.0f;
        uint base = kvh * args.dim + tid;
        for (uint pos = 0; pos < count; pos++) {
            device const uchar *vr = vcache + (ulong)pos * v_row_bytes;
            value = fma(sh[pos], q36_kv_q4_value(vr, base), value);
        }
        device const float *gate = qg + (ulong)tok * args.qg_stride +
            (ulong)head * args.dim * 2u + args.dim;
        float g = gate[tid];
        float sigmoid = g >= 0.0f ? 1.0f / (1.0f + exp(-g))
                                  : exp(g) / (1.0f + exp(g));
        device float *oh = out + ((ulong)tok * args.heads + head) * args.dim;
        oh[tid] = value * shared[0] * sigmoid;
    }
}

struct q36_attention_split_args {
    uint pos0, heads, kv_heads, dim;
    uint qg_stride, has_sinks, n_spans;
};

kernel void q36_attention_q8_q4_split(
        device float *partials [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const uchar *kcache [[buffer(2)]],
        device const uchar *vcache [[buffer(3)]],
        constant q36_attention_split_args &args [[buffer(4)]],
        threadgroup float *shared [[threadgroup(0)]],
        uint3 group_id [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint head = group_id.x, tok = group_id.y, span = group_id.z;
    const uint count = args.pos0 + tok + 1u;
    const uint span0 = span * 512u;
    const uint span_end = min(span0 + 512u, count);
    if (span0 >= count) return;
    const uint kvh = head / (args.heads / args.kv_heads);
    const uint k_row_bytes = args.kv_heads * (args.dim >> 5) * 34u;
    const uint v_row_bytes = args.kv_heads * (args.dim >> 5) * 18u;
    const uint base_dim = kvh * args.dim;
    device const float *qh =
        q + ((ulong)tok * args.heads + head) * args.dim;
    threadgroup float *weights = shared;
    threadgroup float *reduce = shared + 512u;
    const float attn_scale = rsqrt(float(args.dim));

    for (uint pos = span0 + simdgroup; pos < span_end; pos += 8u) {
        device const uchar *kr = kcache + (ulong)pos * k_row_bytes;
        float dotv = 0.0f;
        for (uint j = lane; j < args.dim; j += 32u)
            dotv = fma(qh[j], q36_kv_q8_value(kr, base_dim + j), dotv);
        dotv = simd_sum(dotv) * attn_scale;
        if (lane == 0u) weights[pos - span0] = dotv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint span_n = span_end - span0;
    float vmax = -INFINITY;
    for (uint i = tid; i < span_n; i += 256u) vmax = max(vmax, weights[i]);
    vmax = simd_max(vmax);
    if (lane == 0u) reduce[simdgroup] = vmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float m = reduce[0];
        for (uint sg = 1u; sg < 8u; sg++) m = max(m, reduce[sg]);
        reduce[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float m = reduce[0];

    float lsum = 0.0f;
    for (uint i = tid; i < span_n; i += 256u) {
        float w = exp(weights[i] - m);
        weights[i] = w;
        lsum += w;
    }
    lsum = simd_sum(lsum);
    if (lane == 0u) reduce[simdgroup] = lsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float l = 0.0f;
        for (uint sg = 0u; sg < 8u; sg++) l += reduce[sg];
        reduce[0] = l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint pstride = args.dim + 2u;
    const ulong pbase =
        ((ulong)tok * args.heads * args.n_spans +
         (ulong)head * args.n_spans + span) * pstride;
    if (tid < args.dim) {
        float acc = 0.0f;
        for (uint i = 0u; i < span_n; i++) {
            device const uchar *vr =
                vcache + (ulong)(span0 + i) * v_row_bytes;
            acc = fma(weights[i], q36_kv_q4_value(vr, base_dim + tid), acc);
        }
        partials[pbase + tid] = acc;
    }
    if (tid == 0u) {
        partials[pbase + args.dim] = m;
        partials[pbase + args.dim + 1u] = reduce[0];
    }
}

// Prefill specialization for this model's 8:1 GQA shape.  One workgroup
// serves all eight query heads that share a KV head and folds eight adjacent
// 512-key spans into one partial.  K and V are therefore dequantized once
// instead of once per query head.
kernel void q36_attention_q8_q4_gqa8(
        device float *partials [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const uchar *kcache [[buffer(2)]],
        device const uchar *vcache [[buffer(3)]],
        constant q36_attention_split_args &args [[buffer(4)]],
        threadgroup float *shared [[threadgroup(0)]],
        uint3 group_id [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint HQ = 8u;
    constexpr uint SPANS_PER_GROUP = 8u;
    const uint kvh = group_id.x, tok = group_id.y, span_group = group_id.z;
    const uint h0 = kvh * HQ;
    const uint count = args.pos0 + tok + 1u;
    const uint first_span = span_group * SPANS_PER_GROUP;
    const uint first_key = first_span * 512u;
    if (first_key >= count) return;
    const uint last_span = min(first_span + SPANS_PER_GROUP,
                               (count + 511u) / 512u);
    const uint k_row_bytes = args.kv_heads * 8u * 34u;
    const uint v_row_bytes = args.kv_heads * 8u * 18u;
    const float attn_scale = rsqrt(256.0f);

    threadgroup float *sh_q = shared;
    threadgroup float *sh_w = sh_q + HQ * 256u;
    threadgroup float *reduce = sh_w + HQ * 512u;

    for (uint head = 0u; head < HQ; head++)
        sh_q[head * 256u + tid] =
            q[((ulong)tok * args.heads + h0 + head) * 256u + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float gm[HQ], gl[HQ], gacc[HQ];
    for (uint head = 0u; head < HQ; head++) {
        gm[head] = -INFINITY;
        gl[head] = 0.0f;
        gacc[head] = 0.0f;
    }

    for (uint span = first_span; span < last_span; span++) {
        const uint span0 = span * 512u;
        const uint span_end = min(span0 + 512u, count);
        const uint span_n = span_end - span0;

        // The eight SIMD groups score eight keys at a time.  Every lane
        // walks the same dimensions as q36_attention_q8_q4_split, but each
        // dequantized K value feeds all eight queries sharing this KV head.
        for (uint key0 = span0; key0 < span_end; key0 += 8u) {
            const uint key = key0 + simdgroup;
            float dots[HQ];
            for (uint head = 0u; head < HQ; head++) dots[head] = 0.0f;
            if (key < span_end) {
                device const uchar *kr =
                    kcache + (ulong)key * k_row_bytes;
                const uint kbase = kvh * 256u;
                for (uint d = lane; d < 256u; d += 32u) {
                    const float kval = q36_kv_q8_value(kr, kbase + d);
                    for (uint head = 0u; head < HQ; head++)
                        dots[head] = fma(
                            sh_q[head * 256u + d], kval, dots[head]);
                }
            }
            for (uint head = 0u; head < HQ; head++) {
                dots[head] = simd_sum(dots[head]) * attn_scale;
                if (lane == 0u && key < span_end)
                    sh_w[head * 512u + key - span0] = dots[head];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float sm[HQ], sl[HQ], sacc[HQ];
        for (uint head = 0u; head < HQ; head++) {
            float vmax = -INFINITY;
            for (uint i = tid; i < span_n; i += 256u)
                vmax = max(vmax, sh_w[head * 512u + i]);
            vmax = simd_max(vmax);
            if (lane == 0u) reduce[head * 8u + simdgroup] = vmax;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                float value = reduce[head * 8u];
                for (uint sg = 1u; sg < 8u; sg++)
                    value = max(value, reduce[head * 8u + sg]);
                reduce[head * 8u] = value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            sm[head] = reduce[head * 8u];

            float lsum = 0.0f;
            for (uint i = tid; i < span_n; i += 256u) {
                float w = exp(sh_w[head * 512u + i] - sm[head]);
                sh_w[head * 512u + i] = w;
                lsum += w;
            }
            lsum = simd_sum(lsum);
            if (lane == 0u) reduce[head * 8u + simdgroup] = lsum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                float value = 0.0f;
                for (uint sg = 0u; sg < 8u; sg++)
                    value += reduce[head * 8u + sg];
                reduce[head * 8u] = value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            sl[head] = reduce[head * 8u];
            sacc[head] = 0.0f;
        }

        // Same key-ordered V chain as split-K, with one q4 load shared by
        // all query heads.
        for (uint i = 0u; i < span_n; i++) {
            device const uchar *vr =
                vcache + (ulong)(span0 + i) * v_row_bytes;
            const float vv = q36_kv_q4_value(vr, kvh * 256u + tid);
            for (uint head = 0u; head < HQ; head++)
                sacc[head] = fma(
                    sh_w[head * 512u + i], vv, sacc[head]);
        }

        for (uint head = 0u; head < HQ; head++) {
            if (span == first_span) {
                gm[head] = sm[head];
                gl[head] = sl[head];
                gacc[head] = sacc[head];
            } else {
                float next_m = max(gm[head], sm[head]);
                float r0 = exp(gm[head] - next_m);
                float r1 = exp(sm[head] - next_m);
                gacc[head] = gacc[head] * r0 + sacc[head] * r1;
                gl[head] = gl[head] * r0 + sl[head] * r1;
                gm[head] = next_m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint pstride = args.dim + 2u;
    for (uint head = 0u; head < HQ; head++) {
        const ulong pbase =
            ((ulong)tok * args.heads * args.n_spans +
             (ulong)(h0 + head) * args.n_spans + span_group) * pstride;
        partials[pbase + tid] = gacc[head];
        if (tid == 0u) {
            partials[pbase + args.dim] = gm[head];
            partials[pbase + args.dim + 1u] = gl[head];
        }
    }
}

kernel void q36_attention_gqa8_combine(
        device float *out [[buffer(0)]],
        device const float *partials [[buffer(1)]],
        device const float *qg [[buffer(2)]],
        device const float *sinks [[buffer(3)]],
        constant q36_attention_split_args &args [[buffer(4)]],
        uint2 group_id [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint head = group_id.x, tok = group_id.y;
    if (tid >= args.dim) return;
    const uint count = args.pos0 + tok + 1u;
    const uint groups = (count + 4095u) / 4096u;
    const uint pstride = args.dim + 2u;
    ulong pbase = ((ulong)tok * args.heads * args.n_spans +
                   (ulong)head * args.n_spans) * pstride;
    float acc = partials[pbase + tid];
    float m = partials[pbase + args.dim];
    float l = partials[pbase + args.dim + 1u];
    for (uint group = 1u; group < groups; group++) {
        ulong gb = pbase + (ulong)group * pstride;
        float gm = partials[gb + args.dim];
        float gl = partials[gb + args.dim + 1u];
        float next_m = max(m, gm);
        float r0 = exp(m - next_m), r1 = exp(gm - next_m);
        acc = acc * r0 + partials[gb + tid] * r1;
        l = l * r0 + gl * r1;
        m = next_m;
    }
    if (args.has_sinks) l += exp(sinks[head] - m);
    device const float *gate = qg + (ulong)tok * args.qg_stride +
        (ulong)head * args.dim * 2u + args.dim;
    float g = gate[tid];
    float sigmoid = g >= 0.0f ? 1.0f / (1.0f + exp(-g))
                              : exp(g) / (1.0f + exp(g));
    out[((ulong)tok * args.heads + head) * args.dim + tid] =
        acc * (l > 0.0f ? 1.0f / l : 1.0f) * sigmoid;
}

kernel void q36_attention_split_combine(
        device float *out [[buffer(0)]],
        device const float *partials [[buffer(1)]],
        device const float *qg [[buffer(2)]],
        device const float *sinks [[buffer(3)]],
        constant q36_attention_split_args &args [[buffer(4)]],
        uint2 group_id [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint head = group_id.x, tok = group_id.y;
    if (tid >= args.dim) return;
    const uint count = args.pos0 + tok + 1u;
    const uint spans = (count + 511u) / 512u;
    const uint pstride = args.dim + 2u;
    ulong pbase = ((ulong)tok * args.heads * args.n_spans +
                   (ulong)head * args.n_spans) * pstride;
    float acc = partials[pbase + tid];
    float m = partials[pbase + args.dim];
    float l = partials[pbase + args.dim + 1u];
    for (uint span = 1u; span < spans; span++) {
        ulong sb = pbase + (ulong)span * pstride;
        float sm = partials[sb + args.dim];
        float sl = partials[sb + args.dim + 1u];
        float nm = max(m, sm);
        float r0 = exp(m - nm), r1 = exp(sm - nm);
        acc = acc * r0 + partials[sb + tid] * r1;
        l = l * r0 + sl * r1;
        m = nm;
    }
    if (args.has_sinks) l += exp(sinks[head] - m);
    device const float *gate = qg + (ulong)tok * args.qg_stride +
        (ulong)head * args.dim * 2u + args.dim;
    float g = gate[tid];
    float sigmoid = g >= 0.0f ? 1.0f / (1.0f + exp(-g))
                              : exp(g) / (1.0f + exp(g));
    out[((ulong)tok * args.heads + head) * args.dim + tid] =
        acc * (l > 0.0f ? 1.0f / l : 1.0f) * sigmoid;
}

kernel void q36_attention_f16_parallel(
        device float *out [[buffer(0)]],
        device float *scores [[buffer(1)]],
        device const float *q [[buffer(2)]],
        device const float *qg [[buffer(3)]],
        device const half *kcache [[buffer(4)]],
        device const half *vcache [[buffer(5)]],
        device const float *sinks [[buffer(6)]],
        constant q36_attention_args &args [[buffer(7)]],
        threadgroup float *shared [[threadgroup(0)]],
        uint2 group_id [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    uint head = group_id.x, tok = group_id.y;
    if (head >= args.heads) return;
    uint count = args.pos0 + tok + 1u;
    uint kvh = head / (args.heads / args.kv_heads);
    device const float *qh =
        q + ((ulong)tok * args.heads + head) * args.dim;
    uint score_capacity = args.score_stride / args.heads;
    device float *sh =
        scores + (ulong)tok * args.score_stride +
        (ulong)head * score_capacity;
    float scale = rsqrt(float(args.dim));

    for (uint pos = simdgroup; pos < count; pos += 8u) {
        device const half *kh =
            kcache + ((ulong)pos * args.kv_heads + kvh) * args.dim;
        float dotv = 0.0f;
        for (uint j = lane; j < args.dim; j += 32u)
            dotv = fma(qh[j], float(kh[j]), dotv);
        dotv = simd_sum(dotv) * scale;
        if (lane == 0u) sh[pos] = dotv;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float maxv = args.has_sinks ? sinks[head] : -INFINITY;
        for (uint pos = 0; pos < count; pos++) maxv = max(maxv, sh[pos]);
        float denom = args.has_sinks ? exp(sinks[head] - maxv) : 0.0f;
        for (uint pos = 0; pos < count; pos++) {
            sh[pos] = exp(sh[pos] - maxv);
            denom += sh[pos];
        }
        shared[0] = denom > 0.0f ? 1.0f / denom : 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    if (tid < args.dim) {
        float value = 0.0f;
        for (uint pos = 0; pos < count; pos++) {
            device const half *vh =
                vcache + ((ulong)pos * args.kv_heads + kvh) * args.dim;
            value = fma(sh[pos], float(vh[tid]), value);
        }
        device const float *gate =
            qg + (ulong)tok * args.qg_stride +
            (ulong)head * args.dim * 2u + args.dim;
        float g = gate[tid];
        float sigmoid = g >= 0.0f ? 1.0f / (1.0f + exp(-g))
                                  : exp(g) / (1.0f + exp(g));
        device float *oh =
            out + ((ulong)tok * args.heads + head) * args.dim;
        oh[tid] = value * shared[0] * sigmoid;
    }
}
