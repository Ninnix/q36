// Qwen3.6 convolution, RoPE, steering, and gated delta-net kernels.

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

struct q36_norm_rope_args {
    uint src_stride;
    uint heads;
    uint pos0;
    uint tokens;
    float eps;
};

// Consume the interleaved Q/K projection row directly, apply the learned
// per-head weight and RMSNorm, then rotate the first 64 Qwen head dimensions.
// The lane reduction and RoPE expressions mirror the standalone kernels so
// this only changes dispatch and intermediate traffic, not the arithmetic.
kernel void q36_rms_norm_rope_qwen_f32(
        device float *dst [[buffer(0)]],
        device const float *src [[buffer(1)]],
        device const float *weight [[buffer(2)]],
        constant q36_norm_rope_args &args [[buffer(3)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint rows = args.heads * args.tokens;
    if (row >= rows) return;
    device const float *in = src + (ulong)row * args.src_stride;
    device float *out = dst + (ulong)row * 256u;

    float sum = 0.0f;
    for (uint i = lane; i < 256u; i += 32u) {
        const float v = in[i];
        sum += v * v;
    }
    sum = simd_sum(sum);
    const float scale = rsqrt(sum / 256.0f + args.eps);

    const uint pair = lane;
    const uint tok = row / args.heads;
    float theta = float(args.pos0 + tok) *
                  pow(10000000.0f, -2.0f * float(pair) / 64.0f);
    const uint axis = pair < 11u ? 0u : (pair < 22u ? 1u : 2u);
    if (axis == 3u) theta = 0.0f;
    const float cs = cos(theta);
    const float sn = sin(theta);
    const float a = in[pair] * scale * weight[pair];
    const float b = in[pair + 32u] * scale * weight[pair + 32u];
    out[pair] = fma(-b, sn, a * cs);
    out[pair + 32u] = fma(a, sn, b * cs);

    for (uint i = lane + 64u; i < 256u; i += 32u)
        out[i] = in[i] * scale * weight[i];
}

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

struct q36_recurrent_norm_gate_args {
    uint width;
    uint rows;
    float eps;
};

// Qwen3.5/3.6 recurrent output is laid out as 32 rows of 128 values. Fuse
// the per-row RMSNorm with the following elementwise SiLU gate so the
// normalized state never makes a round trip through device memory.
kernel void q36_recurrent_norm_gate(
        device float *state [[buffer(0)]],
        device const float *gate [[buffer(1)]],
        device const float *weight [[buffer(2)]],
        constant q36_recurrent_norm_gate_args &args [[buffer(3)]],
        uint row [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (row >= args.rows) return;
    const ulong base = (ulong)row * args.width;
    float sum = 0.0f;
    for (uint i = lane; i < args.width; i += 32u) {
        const float v = state[base + i];
        sum = fma(v, v, sum);
    }
    const float scale =
        rsqrt(simd_sum(sum) / float(args.width) + args.eps);
    for (uint i = lane; i < args.width; i += 32u) {
        const ulong at = base + i;
        const float z = gate[at];
        const float normalized = state[at] * scale * weight[i];
        state[at] = (z / (1.0f + exp(-z))) * normalized;
    }
}

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
