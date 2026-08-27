// Small Qwen3.6 graph primitives and shape-specialized dense operations.

kernel void q36_f32_to_f16(device const float *src [[buffer(0)]],
                            device half *dst [[buffer(1)]],
                            constant uint &count [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
    if (i < count) dst[i] = (half)src[i];
}

/* Match the CPU Q8_K activation boundary without materializing the packed
 * block: select the first signed maximum, round to int8, then immediately
 * reconstruct F32 for the native dense IQ3 matvec. */
kernel void q36_q8_k_roundtrip_f32(
        device const float *src [[buffer(0)]],
        device float *dst [[buffer(1)]],
        threadgroup float *signed_max [[threadgroup(0)]],
        uint block [[threadgroup_position_in_grid]],
        uint lane [[thread_position_in_threadgroup]]) {
    const uint i = block * 256u + lane;
    signed_max[lane] = src[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride; stride >>= 1u) {
        if (lane < stride &&
            abs(signed_max[lane + stride]) > abs(signed_max[lane]))
            signed_max[lane] = signed_max[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        const float maxv = signed_max[0];
        const float inv = maxv == 0.0f ? 0.0f : -127.0f / maxv;
        signed_max[0] = inv;
        signed_max[1] = inv == 0.0f ? 0.0f : 1.0f / inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = signed_max[0];
    if (inv == 0.0f) {
        dst[i] = 0.0f;
        return;
    }
    const int q = clamp(int(rint(inv * src[i])), -128, 127);
    dst[i] = float(q) * signed_max[1];
}

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

kernel void q36_top2_f32(
        device int *out_ids [[buffer(0)]],
        device const float *logits [[buffer(1)]],
        constant uint &count [[buffer(2)]],
        threadgroup float *scratch_values [[threadgroup(0)]],
        threadgroup int *scratch_ids [[threadgroup(1)]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    float best0 = -INFINITY, best1 = -INFINITY;
    int id0 = -1, id1 = -1;
    for (uint i = tid; i < count; i += ntg) {
        float value = logits[i];
        if (id0 < 0 || value > best0 ||
            (value == best0 && int(i) < id0)) {
            best1 = best0;
            id1 = id0;
            best0 = value;
            id0 = int(i);
        } else if (id1 < 0 || value > best1 ||
                   (value == best1 && int(i) < id1)) {
            best1 = value;
            id1 = int(i);
        }
    }
    scratch_values[tid] = best0;
    scratch_values[ntg + tid] = best1;
    scratch_ids[tid] = id0;
    scratch_ids[ntg + tid] = id1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0u) return;

    best0 = -INFINITY;
    best1 = -INFINITY;
    id0 = -1;
    id1 = -1;
    for (uint i = 0; i < ntg * 2u; i++) {
        float value = scratch_values[i];
        int id = scratch_ids[i];
        if (id < 0) continue;
        if (id0 < 0 || value > best0 ||
            (value == best0 && id < id0)) {
            if (id != id0) {
                best1 = best0;
                id1 = id0;
            }
            best0 = value;
            id0 = id;
        } else if (id != id0 && (id1 < 0 || value > best1 ||
                   (value == best1 && id < id1))) {
            best1 = value;
            id1 = id;
        }
    }
    out_ids[0] = id0;
    out_ids[1] = id1;
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

// The routed grouped matmul converts its dense RHS tile to half before the
// multiply. Storing the SwiGLU intermediate in that precision avoids the F32
// write/read without changing the values consumed by the down projection.
kernel void q36_moe_activation_f16(
        device half *mid [[buffer(0)]],
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
        (half)((g / (1.0f + exp(-g))) * u * route_weights[pair] * d);
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
