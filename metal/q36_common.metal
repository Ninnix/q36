#include <metal_stdlib>
#ifdef DS4_METAL_HAS_TENSOR
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
using namespace metal;
#ifdef DS4_METAL_HAS_TENSOR
using namespace mpp::tensor_ops;
#endif

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }
#define QK8_0 32
#ifndef QK_K
#define QK_K 256
#endif
#define N_SIMDWIDTH 32
#define N_R0_Q8_0 2
#define N_SG_Q8_0 4
#define FC_MUL_MV 600
#define FC_MUL_MM 700
#define FC_BIN 1300
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

// Reads one byte per stride to warm model-backed pages without copying the
// model. This is outside inference and exists only to reduce first-use stalls.
kernel void kernel_touch_u8_stride(
        device const uchar    *src        [[buffer(0)]],
        device uchar          *dst        [[buffer(1)]],
        constant ulong        &stride     [[buffer(2)]],
        constant ulong        &bytes      [[buffer(3)]],
        constant ulong        &dst_offset [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    ulong off = (ulong)gid * stride;
    if (off >= bytes) return;
    dst[dst_offset + (ulong)gid] = src[off];
}

enum ds4_sort_order {
    DS4_SORT_ORDER_ASC,
    DS4_SORT_ORDER_DESC,
};

struct block_q8_0 {
    half d;
    int8_t qs[QK8_0];
};

struct block_q8_K {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
};
