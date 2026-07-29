// Qwen3.6 routing and GPU-resident F16/Q8/Q4 KV-cache stores.

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

kernel void q36_router_topk_serial(
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

/*
 * Eight SIMD groups independently select their local top eight, then the
 * first SIMD group selects the global top eight from those 64 candidates.
 * This retains the 256-thread router reduction but replaces the old eight
 * full workgroup trees (64 barriers) with one workgroup rendezvous.
 */
kernel void q36_router_topk(
        device int *selected [[buffer(0)]],
        device float *weights [[buffer(1)]],
        device const float *logits [[buffer(2)]],
        constant q36_router_args &args [[buffer(3)]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint simdgroup [[simdgroup_index_in_threadgroup]],
        uint tok [[threadgroup_position_in_grid]]) {
    if (tok >= args.tokens || args.used > 8u) return;
    threadgroup float local_values[64];
    threadgroup uint local_ids[64];
    threadgroup uint winners[8];
    threadgroup float winner_values[8];

    float value = tid < args.experts
        ? logits[(ulong)tok * args.experts + tid] : -INFINITY;
    const uint id = tid;
    for (uint k = 0; k < args.used; ++k) {
        const float best_value = simd_max(value);
        const uint candidate =
            value == best_value ? id : 0xffffffffu;
        const uint best_id = simd_min(candidate);
        if (lane == 0u) {
            const uint at = simdgroup * args.used + k;
            local_values[at] = best_value;
            local_ids[at] = best_id;
        }
        if (id == best_id) value = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u) {
        const uint count = 8u * args.used;
        float value0 = lane < count ? local_values[lane] : -INFINITY;
        float value1 = lane + 32u < count
            ? local_values[lane + 32u] : -INFINITY;
        uint id0 = lane < count ? local_ids[lane] : 0xffffffffu;
        uint id1 = lane + 32u < count
            ? local_ids[lane + 32u] : 0xffffffffu;

        for (uint k = 0; k < args.used; ++k) {
            const bool take0 =
                value0 > value1 || (value0 == value1 && id0 < id1);
            const float local_value = take0 ? value0 : value1;
            const uint local_id = take0 ? id0 : id1;
            const float best_value = simd_max(local_value);
            const uint candidate =
                local_value == best_value ? local_id : 0xffffffffu;
            const uint best_id = simd_min(candidate);
            if (lane == 0u) {
                winners[k] = best_id;
                winner_values[k] = best_value;
            }
            if (id0 == best_id) value0 = -INFINITY;
            if (id1 == best_id) value1 = -INFINITY;
        }

        if (lane == 0u) {
            float sum = 0.0f;
            float top_exp[8];
            for (uint k = 0; k < args.used; ++k) {
                top_exp[k] =
                    exp(winner_values[k] - winner_values[0]);
                sum += top_exp[k];
            }
            sum = max(sum, 6.103515625e-5f);
            for (uint k = 0; k < args.used; ++k) {
                selected[(ulong)tok * args.used + k] =
                    int(winners[k]);
                weights[(ulong)tok * args.used + k] =
                    top_exp[k] / sum * args.scale;
            }
        }
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
