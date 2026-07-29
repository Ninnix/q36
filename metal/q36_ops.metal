// Qwen3.6 graph primitives that are small enough to keep shape-specialized.

kernel void q36_add_f32(device float *out [[buffer(0)]],
                         device const float *a [[buffer(1)]],
                         device const float *b [[buffer(2)]],
                         constant uint &n [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = a[i] + b[i];
}

kernel void q36_copy_f32(device float *out [[buffer(0)]],
                          device const float *in [[buffer(1)]],
                          constant uint &n [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = in[i];
}

kernel void q36_scale_f32(device float *x [[buffer(0)]],
                           constant uint &n [[buffer(1)]],
                           constant float &scale [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
    if (i < n) x[i] *= scale;
}

kernel void q36_swiglu_f32(device float *out [[buffer(0)]],
                            device const float *gate [[buffer(1)]],
                            device const float *up [[buffer(2)]],
                            constant uint &n [[buffer(3)]],
                            constant float &limit [[buffer(4)]],
                            constant float &weight [[buffer(5)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float g = gate[i], u = up[i];
    if (limit > 1.0e-6f) {
        g = min(g, limit);
        u = clamp(u, -limit, limit);
    }
    out[i] = (g / (1.0f + exp(-g))) * u * weight;
}

kernel void q36_ffn_tail_f32(device float *out [[buffer(0)]],
                              device const float *shared [[buffer(1)]],
                              device const float *scalar [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x < width) {
        float raw = scalar[gid.y];
        float gate = raw >= 0.0f ? 1.0f / (1.0f + exp(-raw))
                                 : exp(raw) / (1.0f + exp(raw));
        ulong i = (ulong)gid.y * width + gid.x;
        out[i] = fma(shared[i], gate, out[i]);
    }
}

struct q36_moe_reduce_args {
    uint width;
    uint used;
    uint tokens;
    uint has_down_scale;
};

struct q36_moe_activation_args {
    uint width;
    uint used;
    uint tokens;
    uint has_gate_scale;
    uint has_up_scale;
    uint has_down_scale;
};

kernel void q36_moe_activation_f32(
        device float *mid [[buffer(0)]],
        device const float *gate [[buffer(1)]],
        device const float *up [[buffer(2)]],
        device const int *selected [[buffer(3)]],
        device const float *route_weights [[buffer(4)]],
        device const float *gate_scales [[buffer(5)]],
        device const float *up_scales [[buffer(6)]],
        constant q36_moe_activation_args &args [[buffer(7)]],
        device const float *down_scales [[buffer(8)]],
        uint2 gid [[thread_position_in_grid]]) {
    ulong pair = gid.y;
    if (gid.x >= args.width || pair >= (ulong)args.tokens * args.used) return;
    int expert = selected[pair];
    float g = gate[pair * args.width + gid.x] *
              (args.has_gate_scale ? gate_scales[expert] : 1.0f);
    float u = up[pair * args.width + gid.x] *
              (args.has_up_scale ? up_scales[expert] : 1.0f);
    float d = args.has_down_scale ? down_scales[expert] : 1.0f;
    mid[pair * args.width + gid.x] =
        (g / (1.0f + exp(-g))) * u * route_weights[pair] * d;
}

kernel void q36_moe_reduce_f32(
        device float *out [[buffer(0)]],
        device const float *expert_out [[buffer(1)]],
        device const int *selected [[buffer(2)]],
        device const float *down_scales [[buffer(3)]],
        constant q36_moe_reduce_args &args [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.tokens) return;
    float sum = 0.0f;
    ulong pair = (ulong)gid.y * args.used;
    for (uint slot = 0; slot < args.used; ++slot) {
        int expert = selected[pair + slot];
        float scale = args.has_down_scale ? down_scales[expert] : 1.0f;
        sum = fma(expert_out[(pair + slot) * args.width + gid.x], scale, sum);
    }
    out[(ulong)gid.y * args.width + gid.x] = sum;
}

kernel void q36_rms_norm_f32(device float *out [[buffer(0)]],
                              device const float *in [[buffer(1)]],
                              device const float *weight [[buffer(2)]],
                              constant uint &width [[buffer(3)]],
                              constant float &eps [[buffer(4)]],
                              constant uint &has_weight [[buffer(5)]],
                              uint row [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
    float sum = 0.0f;
    for (uint i = lane; i < width; i += 32) {
        float v = in[row * width + i];
        sum += v * v;
    }
    sum = simd_sum(sum);
    float scale = rsqrt(sum / float(width) + eps);
    for (uint i = lane; i < width; i += 32)
        out[row * width + i] = in[row * width + i] * scale *
                               (has_weight ? weight[i] : 1.0f);
}

kernel void q36_l2_norm_f32(device float *x [[buffer(0)]],
                             constant uint &width [[buffer(1)]],
                             constant float &eps [[buffer(2)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
    float sum = 0.0f;
    for (uint i = lane; i < width; i += 32) {
        float v = x[row * width + i];
        sum += v * v;
    }
    sum = simd_sum(sum);
    float scale = rsqrt(sum + eps);
    for (uint i = lane; i < width; i += 32) x[row * width + i] *= scale;
}

struct q36_dense_args {
    uint in_dim;
    uint out_dim;
    uint tokens;
    float scale;
};

kernel void q36_matmul_f32(device float *out [[buffer(0)]],
                            device const float *weights [[buffer(1)]],
                            device const float *x [[buffer(2)]],
                            constant q36_dense_args &args [[buffer(3)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.out_dim || gid.y >= args.tokens) return;
    device const float *w = weights + (ulong)gid.x * args.in_dim;
    device const float *v = x + (ulong)gid.y * args.in_dim;
    float sum = 0.0f;
    for (uint i = 0; i < args.in_dim; i++) sum = fma(w[i], v[i], sum);
    out[(ulong)gid.y * args.out_dim + gid.x] = sum * args.scale;
}

kernel void q36_matmul_f16(device float *out [[buffer(0)]],
                            device const half *weights [[buffer(1)]],
                            device const float *x [[buffer(2)]],
                            constant q36_dense_args &args [[buffer(3)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.out_dim || gid.y >= args.tokens) return;
    device const half *w = weights + (ulong)gid.x * args.in_dim;
    device const float *v = x + (ulong)gid.y * args.in_dim;
    float sum = 0.0f;
    for (uint i = 0; i < args.in_dim; i++) sum = fma(float(w[i]), v[i], sum);
    out[(ulong)gid.y * args.out_dim + gid.x] = sum * args.scale;
}

struct q36_block_q8_0 {
    half d;
    char qs[32];
};

kernel void q36_matmul_q8_0(device float *out [[buffer(0)]],
                             device const q36_block_q8_0 *weights [[buffer(1)]],
                             device const float *x [[buffer(2)]],
                             constant q36_dense_args &args [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.out_dim || gid.y >= args.tokens) return;
    uint blocks = args.in_dim / 32;
    device const q36_block_q8_0 *w = weights + (ulong)gid.x * blocks;
    device const float *v = x + (ulong)gid.y * args.in_dim;
    float sum = 0.0f;
    for (uint b = 0; b < blocks; b++) {
        float dotv = 0.0f;
        for (uint i = 0; i < 32; i++)
            dotv = fma(float(w[b].qs[i]), v[b * 32 + i], dotv);
        sum = fma(float(w[b].d), dotv, sum);
    }
    out[(ulong)gid.y * args.out_dim + gid.x] = sum * args.scale;
}

struct q36_copy_rows_args {
    uint src_stride;
    uint dst_stride;
    uint src_offset;
    uint width;
    uint rows;
};

kernel void q36_copy_rows_f32(device float *dst [[buffer(0)]],
                               device const float *src [[buffer(1)]],
                               constant q36_copy_rows_args &args [[buffer(2)]],
                               uint2 gid [[thread_position_in_grid]]) {
    if (gid.x < args.width && gid.y < args.rows)
        dst[(ulong)gid.y * args.dst_stride + gid.x] =
            src[(ulong)gid.y * args.src_stride + args.src_offset + gid.x];
}

struct q36_recur_args { uint conv_dim; uint hist; uint tokens; };

kernel void q36_recurrent_window(device float *cache [[buffer(0)]],
                                  device const float *cur [[buffer(1)]],
                                  device float *window [[buffer(2)]],
                                  constant q36_recur_args &args [[buffer(3)]],
                                  uint c [[thread_position_in_grid]]) {
    if (c >= args.conv_dim) return;
    for (uint tok = 0; tok < args.tokens; tok++) {
        ulong wb = (ulong)tok * (args.hist + 1u) * args.conv_dim;
        for (uint i = 0; i < args.hist; i++)
            window[wb + (ulong)i * args.conv_dim + c] =
                cache[(ulong)i * args.conv_dim + c];
        window[wb + (ulong)args.hist * args.conv_dim + c] =
            cur[(ulong)tok * args.conv_dim + c];
        for (uint i = 0; i + 1u < args.hist; i++)
            cache[(ulong)i * args.conv_dim + c] =
                cache[(ulong)(i + 1u) * args.conv_dim + c];
        cache[(ulong)(args.hist - 1u) * args.conv_dim + c] =
            cur[(ulong)tok * args.conv_dim + c];
    }
}

struct q36_conv_args { uint conv_dim; uint taps; uint tokens; };

kernel void q36_conv_silu(device float *out [[buffer(0)]],
                           device const float *window [[buffer(1)]],
                           device const float *weights [[buffer(2)]],
                           constant q36_conv_args &args [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.conv_dim || gid.y >= args.tokens) return;
    ulong wb = (ulong)gid.y * args.taps * args.conv_dim;
    float sum = 0.0f;
    for (uint t = 0; t < args.taps; t++)
        sum = fma(window[wb + (ulong)t * args.conv_dim + gid.x],
                  weights[(ulong)gid.x * args.taps + t], sum);
    out[(ulong)gid.y * args.conv_dim + gid.x] =
        sum / (1.0f + exp(-sum));
}

kernel void q36_recurrent_conv_silu(
        device float *out [[buffer(0)]],
        device float *cache [[buffer(1)]],
        device const float *cur [[buffer(2)]],
        device const float *weights [[buffer(3)]],
        constant q36_conv_args &args [[buffer(4)]],
        uint c [[thread_position_in_grid]]) {
    if (c >= args.conv_dim || args.taps == 0u) return;
    for (uint tok = 0; tok < args.tokens; tok++) {
        float sum = 0.0f;
        for (uint t = 0; t + 1u < args.taps; t++)
            sum = fma(cache[(ulong)t * args.conv_dim + c],
                      weights[(ulong)c * args.taps + t], sum);
        float value = cur[(ulong)tok * args.conv_dim + c];
        sum = fma(value, weights[(ulong)c * args.taps + args.taps - 1u], sum);
        out[(ulong)tok * args.conv_dim + c] = sum / (1.0f + exp(-sum));
        for (uint t = 0; t + 2u < args.taps; t++)
            cache[(ulong)t * args.conv_dim + c] =
                cache[(ulong)(t + 1u) * args.conv_dim + c];
        cache[(ulong)(args.taps - 2u) * args.conv_dim + c] = value;
    }
}

struct q36_rope_args { uint heads; uint pos0; uint tokens; };

kernel void q36_rope_qwen(device float *x [[buffer(0)]],
                           constant q36_rope_args &args [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint pair = gid.x & 31u;
    uint head = gid.x >> 5u;
    uint tok = gid.y;
    if (head >= args.heads || tok >= args.tokens) return;
    ulong base = ((ulong)tok * args.heads + head) * 256u;
    float theta = float(args.pos0 + tok) *
                  pow(10000000.0f, -2.0f * float(pair) / 64.0f);
    uint axis = pair < 11u ? 0u : (pair < 22u ? 1u : 2u);
    if (axis == 3u) theta = 0.0f;
    float cs = cos(theta), sn = sin(theta);
    float a = x[base + pair], b = x[base + pair + 32u];
    x[base + pair] = fma(-b, sn, a * cs);
    x[base + pair + 32u] = fma(a, sn, b * cs);
}

struct q36_steer_args { uint layer; uint width; uint rows; float scale; };

kernel void q36_directional_steering(
        device float *x [[buffer(0)]],
        device const float *directions [[buffer(1)]],
        constant q36_steer_args &args [[buffer(2)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (row >= args.rows) return;
    ulong xb = (ulong)row * args.width;
    ulong db = (ulong)args.layer * args.width;
    float dotv = 0.0f;
    for (uint i = lane; i < args.width; i += 32u)
        dotv = fma(x[xb + i], directions[db + i], dotv);
    dotv = simd_sum(dotv) * args.scale;
    for (uint i = lane; i < args.width; i += 32u)
        x[xb + i] -= dotv * directions[db + i];
}

struct q36_delta_qkv_args {
    uint heads;
    uint groups;
    uint dim;
    uint conv_stride;
    uint tokens;
    float eps;
};

kernel void q36_delta_qkv(
        device float *qout [[buffer(0)]],
        device float *kout [[buffer(1)]],
        device float *vout [[buffer(2)]],
        device const float *conv [[buffer(3)]],
        constant q36_delta_qkv_args &args [[buffer(4)]],
        constant uint &write_v [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    uint head = group.x, tok = group.y;
    if (head >= args.heads || tok >= args.tokens) return;
    uint g = head % args.groups;
    ulong cb = (ulong)tok * args.conv_stride;
    ulong ob = ((ulong)tok * args.heads + head) * args.dim;
    float sq_q = 0.0f, sq_k = 0.0f;
    for (uint i = lane; i < args.dim; i += 32u) {
        float qv = conv[cb + (ulong)g * args.dim + i];
        float kv = conv[cb + (ulong)(args.groups + g) * args.dim + i];
        sq_q = fma(qv, qv, sq_q);
        sq_k = fma(kv, kv, sq_k);
    }
    float qs = 1.0f / max(sqrt(simd_sum(sq_q)), args.eps);
    float ks = 1.0f / max(sqrt(simd_sum(sq_k)), args.eps);
    for (uint i = lane; i < args.dim; i += 32u) {
        qout[ob + i] = conv[cb + (ulong)g * args.dim + i] * qs;
        kout[ob + i] = conv[cb + (ulong)(args.groups + g) * args.dim + i] * ks;
        if (write_v)
            vout[ob + i] = conv[cb + (ulong)2u * args.groups * args.dim +
                                      (ulong)head * args.dim + i];
    }
}

struct q36_delta_gate_args { uint heads; uint tokens; };

kernel void q36_delta_gates(
        device float *out [[buffer(0)]],
        device const float *alpha [[buffer(1)]],
        device const float *beta [[buffer(2)]],
        device const float *dt [[buffer(3)]],
        device const float *acoef [[buffer(4)]],
        constant q36_delta_gate_args &args [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.heads || gid.y >= args.tokens) return;
    ulong src = (ulong)gid.y * args.heads + gid.x;
    ulong dst = (ulong)gid.y * args.heads * 2u + gid.x;
    float z = alpha[src] + dt[gid.x];
    float softplus = z > 20.0f ? z : log(1.0f + exp(z));
    out[dst] = exp(softplus * acoef[gid.x]);
    float b = beta[src];
    out[dst + args.heads] = b >= 0.0f
        ? 1.0f / (1.0f + exp(-b))
        : exp(b) / (1.0f + exp(b));
}

struct q36_delta_net_args { uint heads; uint dim; uint tokens; };

kernel void q36_delta_net_decode(
        device float *state [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const float *k [[buffer(2)]],
        device const float *v [[buffer(3)]],
        device const float *gb [[buffer(4)]],
        device float *out [[buffer(5)]],
        constant q36_delta_net_args &args [[buffer(6)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    uint head = group.x;
    uint col0 = group.y * 32u;
    uint col = col0 + lane;
    if (head >= args.heads || col >= args.dim) return;
    ulong matrix = (ulong)head * args.dim * args.dim;
    float norm = rsqrt(float(args.dim));
    for (uint tok = 0; tok < args.tokens; tok++) {
        ulong vec = ((ulong)tok * args.heads + head) * args.dim;
        float decay = gb[(ulong)tok * 2u * args.heads + head];
        float beta = gb[(ulong)tok * 2u * args.heads + args.heads + head];
        float sk = 0.0f;
        for (uint i = 0; i < args.dim; i++) {
            ulong at = matrix + (ulong)i * args.dim + col;
            float s = state[at] * decay;
            state[at] = s;
            sk = fma(s, k[vec + i], sk);
        }
        float d = (v[vec + col] - sk) * beta;
        float y = 0.0f;
        for (uint i = 0; i < args.dim; i++) {
            ulong at = matrix + (ulong)i * args.dim + col;
            float s = fma(k[vec + i], d, state[at]);
            state[at] = s;
            y = fma(s, q[vec + i], y);
        }
        out[vec + col] = y * norm;
    }
}

/*
 * The recurrent state is private to the Metal backend, so keep it transposed
 * here: one contiguous row contains all input-state values for one output
 * channel.  Four SIMD groups update four output channels per threadgroup and
 * each lane retains four state values in registers across the token loop.
 */
kernel void q36_delta_net_decode_r4(
        device float *state [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const float *k [[buffer(2)]],
        device const float *v [[buffer(3)]],
        device const float *gb [[buffer(4)]],
        device float *out [[buffer(5)]],
        constant q36_delta_net_args &args [[buffer(6)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint R = 4u;
    uint col = group.x * R + simdgroup;
    uint head = group.y;
    if (head >= args.heads || col >= args.dim) return;

    ulong matrix = (ulong)head * args.dim * args.dim;
    device float *srow = state + matrix + (ulong)col * args.dim;
    float ls[R];
    for (uint j = 0; j < R; j++) {
        ls[j] = srow[lane * R + j];
    }

    float norm = rsqrt(float(args.dim));
    for (uint tok = 0; tok < args.tokens; tok++) {
        ulong vec = ((ulong)tok * args.heads + head) * args.dim;
        float decay = gb[(ulong)tok * 2u * args.heads + head];
        float beta = gb[(ulong)tok * 2u * args.heads + args.heads + head];
        float sk = 0.0f;
        for (uint j = 0; j < R; j++) {
            uint i = lane * R + j;
            ls[j] *= decay;
            sk = fma(ls[j], k[vec + i], sk);
        }
        sk = simd_sum(sk);
        float d = (v[vec + col] - sk) * beta;
        float y = 0.0f;
        for (uint j = 0; j < R; j++) {
            uint i = lane * R + j;
            ls[j] = fma(k[vec + i], d, ls[j]);
            y = fma(ls[j], q[vec + i], y);
        }
        y = simd_sum(y);
        if (lane == 0) out[vec + col] = y * norm;
    }

    for (uint j = 0; j < R; j++) {
        srow[lane * R + j] = ls[j];
    }
}

struct q36_kv_store_args {
    uint k_row;
    uint v_row;
    uint pos0;
    uint tokens;
};

kernel void q36_kv_store_f16(
        device half *kcache [[buffer(0)]],
        device half *vcache [[buffer(1)]],
        device const float *k [[buffer(2)]],
        device const float *v [[buffer(3)]],
        constant q36_kv_store_args &args [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, tok = gid.y;
    if (tok >= args.tokens) return;
    uint pos = args.pos0 + tok;
    if (i < args.k_row)
        kcache[(ulong)pos * args.k_row + i] =
            half(k[(ulong)tok * args.k_row + i]);
    if (i < args.v_row)
        vcache[(ulong)pos * args.v_row + i] =
            half(v[(ulong)tok * args.v_row + i]);
}

struct q36_router_args {
    uint experts;
    uint used;
    uint tokens;
    float scale;
};

kernel void q36_router_topk(
        device int *selected [[buffer(0)]],
        device float *weights [[buffer(1)]],
        device const float *logits [[buffer(2)]],
        constant q36_router_args &args [[buffer(3)]],
        uint tok [[thread_position_in_grid]]) {
    if (tok >= args.tokens || args.used > 8u) return;
    float best[8];
    int ids[8];
    for (uint k = 0; k < args.used; k++) {
        best[k] = -INFINITY;
        ids[k] = -1;
    }
    device const float *row = logits + (ulong)tok * args.experts;
    for (uint e = 0; e < args.experts; e++) {
        float value = row[e];
        for (uint k = 0; k < args.used; k++) {
            if (value > best[k] || (value == best[k] && int(e) < ids[k])) {
                for (uint j = args.used - 1u; j > k; j--) {
                    best[j] = best[j - 1u];
                    ids[j] = ids[j - 1u];
                }
                best[k] = value;
                ids[k] = int(e);
                break;
            }
        }
    }
    float sum = 0.0f;
    for (uint k = 0; k < args.used; k++) sum += exp(best[k] - best[0]);
    sum = max(sum, 6.103515625e-5f);
    for (uint k = 0; k < args.used; k++) {
        selected[(ulong)tok * args.used + k] = ids[k];
        weights[(ulong)tok * args.used + k] =
            exp(best[k] - best[0]) / sum * args.scale;
    }
}

struct q36_attention_args {
    uint pos0;
    uint heads;
    uint kv_heads;
    uint dim;
    uint qg_stride;
    uint has_sinks;
    uint score_stride;
};

static inline float q36_kv_q8_value(device const uchar *row, uint index) {
    device const uchar *block = row + (ulong)(index >> 5) * 34u;
    float d = float(*reinterpret_cast<device const half *>(block));
    return d * float(*reinterpret_cast<device const char *>(block + 2u + (index & 31u)));
}

static inline float q36_kv_q4_value(device const uchar *row, uint index) {
    device const uchar *block = row + (ulong)(index >> 5) * 18u;
    float d = float(*reinterpret_cast<device const half *>(block));
    uint i = index & 31u;
    uchar packed = block[2u + (i & 15u)];
    uint q = i < 16u ? (packed & 15u) : (packed >> 4);
    return d * float(int(q) - 8);
}

struct q36_kv_quant_args {
    uint k_row;
    uint v_row;
    uint pos0;
    uint tokens;
    uint k_row_bytes;
    uint v_row_bytes;
};

kernel void q36_kv_store_q8_q4(
        device uchar *kcache [[buffer(0)]],
        device uchar *vcache [[buffer(1)]],
        device const float *k [[buffer(2)]],
        device const float *v [[buffer(3)]],
        constant q36_kv_quant_args &args [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]]) {
    uint block = gid.x, tok = gid.y;
    if (tok >= args.tokens) return;
    uint kblocks = args.k_row >> 5;
    uint vblocks = args.v_row >> 5;
    if (block < kblocks) {
        device const float *src = k + (ulong)tok * args.k_row + block * 32u;
        float amax = 0.0f;
        for (uint i = 0; i < 32u; i++) amax = max(amax, abs(src[i]));
        float d = amax / 127.0f;
        device uchar *dst = kcache + (ulong)(args.pos0 + tok) *
            args.k_row_bytes + block * 34u;
        *reinterpret_cast<device half *>(dst) = half(d);
        float inv = d != 0.0f ? 1.0f / d : 0.0f;
        for (uint i = 0; i < 32u; i++) {
            int q = clamp(int(rint(src[i] * inv)), -127, 127);
            *reinterpret_cast<device char *>(dst + 2u + i) = char(q);
        }
    }
    if (block < vblocks) {
        device const float *src = v + (ulong)tok * args.v_row + block * 32u;
        float vmax = 0.0f;
        float amax = 0.0f;
        for (uint i = 0; i < 32u; i++) {
            float a = abs(src[i]);
            if (a > amax) {
                amax = a;
                vmax = src[i];
            }
        }
        float d = vmax / -8.0f;
        device uchar *dst = vcache + (ulong)(args.pos0 + tok) *
            args.v_row_bytes + block * 18u;
        *reinterpret_cast<device half *>(dst) = half(d);
        float inv = d != 0.0f ? 1.0f / d : 0.0f;
        for (uint i = 0; i < 16u; i++) {
            uint q0 = uint(clamp(int(src[i] * inv + 8.5f), 0, 15));
            uint q1 = uint(clamp(int(src[i + 16u] * inv + 8.5f), 0, 15));
            dst[2u + i] = uchar(q0 | (q1 << 4));
        }
    }
}

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
