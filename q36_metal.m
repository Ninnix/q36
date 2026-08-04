#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "q36_gpu.h"
#include "q36_iq_tables.h"
#include "q36_ssd.h"
#define Q36_QUANT_LINKAGE __attribute__((weak_import))
#include "q36_quant.h"

#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <unistd.h>

/*
 * Metal backend foundation.
 *
 * Apple Silicon buffers use shared storage: CPU reference probes and Metal
 * kernels see the same allocation without staging copies.  Views retain their
 * owner buffer and only adjust the byte range, matching q36_gpu.h's lifetime
 * contract.  Operator kernels are added below this layer as they reach parity;
 * keeping resource and command ownership here prevents each ported Vulkan op
 * from inventing its own synchronization policy.
 */

struct q36_gpu_tensor {
    id<MTLBuffer> buffer;
    uint64_t offset;
    uint64_t bytes;
    bool owner;
};

static id<MTLDevice> q36_device;
static id<MTLCommandQueue> q36_queue;
static id<MTLLibrary> q36_library;
static id<MTLCommandBuffer> q36_batch;
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *q36_pipelines;
static NSMutableDictionary<NSString *, NSArray *> *q36_model_views;
static id q36_model_residency_set;
static bool q36_model_residency_added;
static pthread_mutex_t q36_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t q36_live_bytes;
static uint64_t q36_peak_bytes;
static const void *q36_model_map;
static uint64_t q36_model_size;
static int q36_model_fd = -1;
static bool q36_quality;
static bool q36_micro_batch;
static bool q36_ssd_streaming;
static bool q36_dense_model;

static uint64_t q36_gpu_system_memory_bytes(void) {
    uint64_t bytes = 0;
    size_t len = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &len, NULL, 0) != 0) return 0;
    return len == sizeof(bytes) ? bytes : 0;
}

static void q36_gpu_print_device_summary(void) {
    const char *name = q36_device.name ? q36_device.name.UTF8String : "unknown Metal device";
    uint64_t mem = q36_gpu_system_memory_bytes();
    if (mem) {
        fprintf(stderr, "q36: Metal device %s, %.2f GiB RAM\n",
                name, (double)mem / 1073741824.0);
    } else {
        fprintf(stderr, "q36: Metal device %s\n", name);
    }
}

enum {
    Q36_METAL_STREAM_MAX_LAYERS = 40,
    Q36_METAL_STREAM_MAX_EXPERTS = 256,
};
typedef struct {
    uint32_t layer;
    uint32_t expert;
    uint64_t last_used;
    uint32_t priority;
    bool valid;
} q36_metal_stream_entry;
typedef struct {
    id<MTLBuffer> gate;
    id<MTLBuffer> up;
    id<MTLBuffer> down;
    uint64_t gate_bytes;
    uint64_t up_bytes;
    uint64_t down_bytes;
    bool valid;
} q36_metal_full_layer;
static uint32_t q36_streaming_full_layers;
static uint32_t q36_streaming_expert_budget;
static uint32_t q36_streaming_expert_cap;
static uint32_t q36_streaming_expert_count;
static uint64_t q36_streaming_expert_bytes;
static uint64_t q36_streaming_config_gate_bytes;
static uint64_t q36_streaming_config_up_bytes;
static uint64_t q36_streaming_config_down_bytes;
static uint64_t q36_streaming_clock;
static uint64_t q36_streaming_hits;
static uint64_t q36_streaming_misses;
static uint64_t q36_streaming_loads;
static uint64_t q36_streaming_evictions;
static uint64_t q36_streaming_read_bytes;
static uint64_t q36_streaming_route_tokens;
static uint64_t q36_streaming_gate_bytes;
static uint64_t q36_streaming_up_bytes;
static uint64_t q36_streaming_down_bytes;
static q36_metal_stream_entry *q36_streaming_entries;
static int32_t
    q36_streaming_lookup[Q36_METAL_STREAM_MAX_LAYERS]
                        [Q36_METAL_STREAM_MAX_EXPERTS];
static uint32_t
    q36_streaming_bias[Q36_METAL_STREAM_MAX_LAYERS]
                      [Q36_METAL_STREAM_MAX_EXPERTS];
static uint32_t
    q36_streaming_hotness[Q36_METAL_STREAM_MAX_LAYERS]
                         [Q36_METAL_STREAM_MAX_EXPERTS];
static id<MTLBuffer> q36_streaming_gate;
static id<MTLBuffer> q36_streaming_up;
static id<MTLBuffer> q36_streaming_down;
static q36_metal_full_layer
    q36_streaming_full[Q36_METAL_STREAM_MAX_LAYERS];
static pthread_mutex_t q36_stream_mu = PTHREAD_MUTEX_INITIALIZER;
static id<MTLBuffer> q36_moe_gate_scratch;
static id<MTLBuffer> q36_moe_up_scratch;
static id<MTLBuffer> q36_moe_down_scratch;
static id<MTLBuffer> q36_moe_id_map;
static NSUInteger q36_moe_pair_scratch_bytes;
static NSUInteger q36_moe_down_scratch_bytes;
static NSUInteger q36_moe_id_map_bytes;
static id<MTLBuffer> q36_shared_gate_scratch;
static id<MTLBuffer> q36_shared_up_scratch;
static id<MTLBuffer> q36_shared_down_scratch;
static NSUInteger q36_shared_mid_scratch_bytes;
static NSUInteger q36_shared_down_scratch_bytes;
static bool q36_prof_active;
static double q36_prof_gpu_seconds;
static uint64_t q36_prof_command_buffers;

static void q36_streaming_cache_clear(bool clear_full_layers);

static void q36_model_residency_clear(void) {
    if (@available(macOS 15.0, *)) {
        if (q36_model_residency_set) {
            if (q36_model_residency_added && q36_queue &&
                [q36_queue respondsToSelector:@selector(removeResidencySet:)])
                [q36_queue removeResidencySet:q36_model_residency_set];
            [q36_model_residency_set endResidency];
            [q36_model_residency_set removeAllAllocations];
        }
    }
    q36_model_residency_set = nil;
    q36_model_residency_added = false;
}

static int q36_metal_wait(void) {
    id<MTLCommandBuffer> cb = nil;
    pthread_mutex_lock(&q36_mu);
    cb = q36_batch;
    q36_batch = nil;
    pthread_mutex_unlock(&q36_mu);
    if (!cb) return 1;
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "q36: Metal command buffer failed: %s\n",
                cb.error.localizedDescription.UTF8String);
        return 0;
    }
    if (q36_prof_active) {
        const double start = cb.GPUStartTime;
        const double end = cb.GPUEndTime;
        if (end > start) q36_prof_gpu_seconds += end - start;
        q36_prof_command_buffers++;
    }
    return 1;
}

static const char *q36_metal_base_source =
"#include <metal_stdlib>\n"
"#ifdef Q36_METAL_HAS_TENSOR\n"
"#include <metal_tensor>\n"
"#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
"#endif\n"
"using namespace metal;\n"
"#ifdef Q36_METAL_HAS_TENSOR\n"
"using namespace mpp::tensor_ops;\n"
"#endif\n"
"\n"
"#define MAX(x, y) ((x) > (y) ? (x) : (y))\n"
"#define MIN(x, y) ((x) < (y) ? (x) : (y))\n"
"#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }\n"
"#define QK8_0 32\n"
"#ifndef QK_K\n"
"#define QK_K 256\n"
"#endif\n"
"#define N_SIMDWIDTH 32\n"
"#define N_R0_Q8_0 2\n"
"#define N_SG_Q8_0 4\n"
"#define FC_MUL_MV 600\n"
"#define FC_MUL_MM 700\n"
"#define FC_BIN 1300\n"
"#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n"
"#ifndef M_PI_F\n"
"#define M_PI_F 3.14159265358979323846f\n"
"#endif\n"
"\n"
"struct block_q8_0 {\n"
"    half d;\n"
"    int8_t qs[QK8_0];\n"
"};\n"
"\n"
"struct block_q8_K {\n"
"    float d;\n"
"    int8_t qs[QK_K];\n"
"    int16_t bsums[QK_K / 16];\n"
"};\n";

static void q36_metal_append_u8_table(NSMutableString *source,
                                       NSString *name,
                                       const uint8_t *values, size_t count) {
    [source appendFormat:@"constant uchar %@[%zu] = {", name, count];
    for (size_t i = 0; i < count; i++)
        [source appendFormat:(i ? @",%u" : @"%u"), (unsigned)values[i]];
    [source appendString:@"};\n"];
}

static void q36_metal_append_u32_table(NSMutableString *source,
                                        NSString *name,
                                        const uint32_t *values, size_t count) {
    [source appendFormat:@"constant uint %@[%zu] = {", name, count];
    for (size_t i = 0; i < count; i++)
        [source appendFormat:(i ? @",%u" : @"%u"), values[i]];
    [source appendString:@"};\n"];
}

static void q36_metal_append_iq3_tables(NSMutableString *source) {
    q36_metal_append_u8_table(source, @"q36_dense_kmask_iq3",
                              q36_kmask_iq2xs,
                              sizeof(q36_kmask_iq2xs));
    q36_metal_append_u8_table(source, @"q36_dense_ksigns_iq3",
                              q36_ksigns_iq2xs,
                              sizeof(q36_ksigns_iq2xs));
    q36_metal_append_u32_table(source, @"q36_dense_iq3xxs_grid",
                               q36_iq3xxs_grid,
                               sizeof(q36_iq3xxs_grid) /
                                   sizeof(q36_iq3xxs_grid[0]));
    q36_metal_append_u32_table(source, @"q36_dense_iq3s_grid",
                               q36_iq3s_grid,
                               sizeof(q36_iq3s_grid) /
                                   sizeof(q36_iq3s_grid[0]));
}

static NSString *q36_metal_source(void) {
    NSArray<NSArray<NSString *> *> *required_sources = @[
        @[@"Q36_METAL_DENSE_SOURCE",    @"metal/dense.metal"],
        @[@"Q36_METAL_MOE_SOURCE",      @"metal/moe.metal"],
        @[@"Q36_METAL_NORM_SOURCE",     @"metal/norm.metal"],
        @[@"Q36_METAL_OPS_SOURCE",      @"metal/ops.metal"],
        @[@"Q36_METAL_RECURRENT_SOURCE", @"metal/recurrent.metal"],
        @[@"Q36_METAL_KV_SOURCE",       @"metal/kv.metal"],
        @[@"Q36_METAL_ATTN_SOURCE",     @"metal/attention.metal"]
    ];
    NSFileManager *fm = [NSFileManager defaultManager];
    NSMutableString *source = [NSMutableString
        stringWithUTF8String:q36_metal_base_source];
    q36_metal_append_iq3_tables(source);
    for (NSArray<NSString *> *spec in required_sources) {
        const char *override_path = getenv(spec[0].UTF8String);
        NSMutableArray<NSString *> *paths = [NSMutableArray array];
        if (override_path && override_path[0])
            [paths addObject:[NSString stringWithUTF8String:override_path]];
        [paths addObject:spec[1]];
        [paths addObject:[@"./" stringByAppendingString:spec[1]]];

        NSString *loaded = nil;
        NSString *loaded_path = nil;
        for (NSString *path in paths) {
            if (![fm fileExistsAtPath:path]) continue;
            NSError *error = nil;
            loaded = [NSString stringWithContentsOfFile:path
                                               encoding:NSUTF8StringEncoding
                                                  error:&error];
            if (!loaded) {
                fprintf(stderr, "q36: failed to read Metal source %s: %s\n",
                        path.UTF8String,
                        error.localizedDescription.UTF8String);
                return nil;
            }
            loaded_path = path;
            break;
        }
        if (!loaded) {
            fprintf(stderr,
                    "q36: Metal source %s not found (set %s to override)\n",
                    spec[1].UTF8String, spec[0].UTF8String);
            return nil;
        }
        [source appendFormat:@"\n// appended %@\n%@\n", loaded_path, loaded];
    }
    return source;
}

static id<MTLComputePipelineState> q36_pipeline(NSString *name) {
    id<MTLComputePipelineState> p = q36_pipelines[name];
    if (p) return p;
    id<MTLFunction> fn = [q36_library newFunctionWithName:name];
    if (!fn) {
        fprintf(stderr, "q36: Metal function %s is missing\n", name.UTF8String);
        return nil;
    }
    NSError *error = nil;
    p = [q36_device newComputePipelineStateWithFunction:fn error:&error];
    if (!p) {
        fprintf(stderr, "q36: Metal pipeline %s failed: %s\n",
                name.UTF8String, error.localizedDescription.UTF8String);
        return nil;
    }
    q36_pipelines[name] = p;
    return p;
}

static id<MTLComputePipelineState> q36_pipeline_nsg(NSString *name, int16_t nsg) {
    NSString *key = [NSString stringWithFormat:@"%@#%d", name, (int)nsg];
    id<MTLComputePipelineState> p = q36_pipelines[key];
    if (p) return p;
    MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
    [constants setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];
    NSError *error = nil;
    id<MTLFunction> fn = [q36_library newFunctionWithName:name
                                          constantValues:constants error:&error];
    if (!fn) {
        fprintf(stderr, "q36: Metal function %s specialization failed: %s\n",
                name.UTF8String, error.localizedDescription.UTF8String);
        return nil;
    }
    p = [q36_device newComputePipelineStateWithFunction:fn error:&error];
    if (!p) {
        fprintf(stderr, "q36: Metal pipeline %s specialization failed: %s\n",
                name.UTF8String, error.localizedDescription.UTF8String);
        return nil;
    }
    q36_pipelines[key] = p;
    return p;
}

static id<MTLComputePipelineState> q36_pipeline_mv_ext(
        NSString *name, int16_t nsg, int16_t nxpsg) {
    NSString *key = [NSString stringWithFormat:@"%@#ext%d:%d", name,
                     (int)nsg, (int)nxpsg];
    id<MTLComputePipelineState> p = q36_pipelines[key];
    if (p) return p;
    MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
    [constants setConstantValue:&nsg type:MTLDataTypeShort atIndex:600];
    [constants setConstantValue:&nxpsg type:MTLDataTypeShort atIndex:601];
    NSError *error = nil;
    id<MTLFunction> fn = [q36_library newFunctionWithName:name
                                          constantValues:constants error:&error];
    if (!fn) {
        fprintf(stderr, "q36: Metal ext function %s failed: %s\n",
                name.UTF8String, error.localizedDescription.UTF8String);
        return nil;
    }
    p = [q36_device newComputePipelineStateWithFunction:fn error:&error];
    if (!p) {
        fprintf(stderr, "q36: Metal ext pipeline %s failed: %s\n",
                name.UTF8String, error.localizedDescription.UTF8String);
        return nil;
    }
    q36_pipelines[key] = p;
    return p;
}

static id<MTLComputePipelineState> q36_pipeline_mm(
        NSString *name, bool bc_inp, bool bc_out) {
    NSString *key = [NSString stringWithFormat:@"%@#mm%d%d",
                     name, bc_inp ? 1 : 0, bc_out ? 1 : 0];
    id<MTLComputePipelineState> p = q36_pipelines[key];
    if (p) return p;
    MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
    [constants setConstantValue:&bc_inp type:MTLDataTypeBool atIndex:700];
    [constants setConstantValue:&bc_out type:MTLDataTypeBool atIndex:701];
    NSError *error = nil;
    id<MTLFunction> fn = [q36_library newFunctionWithName:name
                                          constantValues:constants error:&error];
    if (!fn) {
        fprintf(stderr, "q36: Metal MM function %s failed: %s\n",
                name.UTF8String, error.localizedDescription.UTF8String);
        return nil;
    }
    p = [q36_device newComputePipelineStateWithFunction:fn error:&error];
    if (!p) {
        fprintf(stderr, "q36: Metal MM pipeline %s failed: %s\n",
                name.UTF8String, error.localizedDescription.UTF8String);
        return nil;
    }
    q36_pipelines[key] = p;
    return p;
}

static id<MTLComputeCommandEncoder> q36_encoder(NSString *kernel) {
    if (!q36_batch && !q36_gpu_begin_commands()) return nil;
    id<MTLComputePipelineState> p = q36_pipeline(kernel);
    if (!p) return nil;
    id<MTLComputeCommandEncoder> enc = [q36_batch computeCommandEncoder];
    enc.label = kernel;
    [enc setComputePipelineState:p];
    return enc;
}

static id<MTLBuffer> q36_model_view(const void *map, uint64_t model_size,
                                    uint64_t offset, uint64_t bytes,
                                    uint64_t *inner) {
    if (!map || !q36_device || offset > model_size ||
        bytes > model_size - offset) return nil;
    uint64_t page = (uint64_t)getpagesize();
    uint64_t page_offset = offset & ~(page - 1u);
    uint64_t leading = offset - page_offset;
    if (bytes > UINT64_MAX - leading ||
        leading + bytes > UINT64_MAX - (page - 1u)) return nil;
    uint64_t view_bytes = (leading + bytes + page - 1u) & ~(page - 1u);
    if (view_bytes > model_size - page_offset)
        view_bytes = model_size - page_offset;
    if (leading + bytes > view_bytes ||
        view_bytes > (uint64_t)q36_device.maxBufferLength) return nil;
    NSString *key = [NSString stringWithFormat:@"%016" PRIx64 ":%016" PRIx64,
                                               offset, bytes];
    if (map == q36_model_map && model_size == q36_model_size) {
        NSArray *cached = q36_model_views[key];
        if (cached) {
            if (inner) *inner = [cached[1] unsignedLongLongValue];
            return cached[0];
        }
    }
    id<MTLBuffer> buffer =
        [q36_device newBufferWithBytesNoCopy:(void *)((uintptr_t)map + page_offset)
                                     length:(NSUInteger)view_bytes
                                    options:MTLResourceStorageModeShared
                                deallocator:nil];
    if (!buffer) return nil;
    if (inner) *inner = leading;
    if (map == q36_model_map && model_size == q36_model_size)
        q36_model_views[key] = @[buffer, @(leading)];
    return buffer;
}

typedef struct {
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t tokens;
    float scale;
} q36_dense_args;

typedef struct {
    int32_t ne00, ne01, ne02;
    uint64_t nb00, nb01, nb02, nb03;
    int32_t ne10, ne11, ne12;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t ne0, ne1, nr0;
    int16_t r2, r3;
} q36_q8_mv_args;

typedef struct {
    int32_t ne00, ne02;
    uint64_t nb01, nb02, nb03;
    int32_t ne12;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t ne0, ne1;
    int16_t r2, r3;
} q36_q8_mm_args;

static int q36_q8_prefill_mm(
        q36_gpu_tensor *out, const void *map, uint64_t size,
        uint64_t offset, uint64_t in_dim, uint64_t out_dim,
        const q36_gpu_tensor *x, uint64_t tokens, float scale) {
    if (!out || !x || !map || tokens < 2u || !in_dim || !out_dim ||
        (in_dim & 31u) || in_dim > INT32_MAX || out_dim > INT32_MAX ||
        tokens > INT32_MAX) return 0;
    uint64_t row_bytes = (in_dim / 32u) * 34u;
    if (out_dim > UINT64_MAX / row_bytes) return 0;
    uint64_t inner = 0;
    id<MTLBuffer> weights =
        q36_model_view(map, size, offset, out_dim * row_bytes, &inner);
    bool bc_inp = (in_dim % 32u) != 0;
    bool bc_out = (out_dim % 64u) != 0 || (tokens % 32u) != 0;
    id<MTLComputePipelineState> p =
        q36_pipeline_mm(@"kernel_mul_mm_q8_0_f32", bc_inp, bc_out);
    if (!weights || !p || (!q36_batch && !q36_gpu_begin_commands())) return 0;
    q36_q8_mm_args args = {
        (int32_t)in_dim, 1, row_bytes, row_bytes * out_dim,
        row_bytes * out_dim, 1, sizeof(float), in_dim * sizeof(float),
        in_dim * tokens * sizeof(float), in_dim * tokens * sizeof(float),
        (int32_t)out_dim, (int32_t)tokens, 1, 1
    };
    id<MTLComputeCommandEncoder> enc = [q36_batch computeCommandEncoder];
    enc.label = @"kernel_mul_mm_q8_0_f32";
    [enc setComputePipelineState:p];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:weights offset:(NSUInteger)inner atIndex:1];
    [enc setBuffer:x->buffer offset:x->offset atIndex:2];
    [enc setBuffer:out->buffer offset:out->offset atIndex:3];
    [enc setThreadgroupMemoryLength:(bc_out ? 8192u : 6144u) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((tokens + 31u) / 32u,
                                          (out_dim + 63u) / 64u, 1)
         threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    if (scale != 1.0f) {
        uint64_t count64 = tokens * out_dim;
        if (count64 > UINT32_MAX) return 0;
        uint32_t count = (uint32_t)count64;
        enc = q36_encoder(@"q36_scale_f32");
        if (!enc) return 0;
        [enc setBuffer:out->buffer offset:out->offset atIndex:0];
        [enc setBytes:&count length:sizeof(count) atIndex:1];
        [enc setBytes:&scale length:sizeof(scale) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    return 1;
}

/* Router-sized F32 projections are large enough to benefit from tiled
 * simdgroup matrix arithmetic during prefill. Keep both inputs in F32:
 * converting the router to half changes expert selection often enough to
 * fail the strict CPU-parity overlap checks. */
static int q36_f32_prefill_mm(
        q36_gpu_tensor *out, const void *map, uint64_t size,
        uint64_t offset, uint64_t in_dim, uint64_t out_dim,
        const q36_gpu_tensor *x, uint64_t tokens, float scale) {
    if (!out || !x || !map || tokens < 2u || out_dim < 64u ||
        !in_dim || (in_dim & 15u) || in_dim > INT32_MAX ||
        out_dim > INT32_MAX || tokens > INT32_MAX ||
        out_dim > UINT64_MAX / in_dim / sizeof(float))
        return 0;
    const uint64_t row_bytes = in_dim * sizeof(float);
    uint64_t inner = 0;
    id<MTLBuffer> weights =
        q36_model_view(map, size, offset, out_dim * row_bytes, &inner);
    const bool bc_inp = (in_dim % 32u) != 0;
    const bool bc_out = (out_dim % 64u) != 0 || (tokens % 32u) != 0;
    id<MTLComputePipelineState> p =
        q36_pipeline_mm(@"kernel_mul_mm_f32_ff32", bc_inp, bc_out);
    if (!weights || !p || (!q36_batch && !q36_gpu_begin_commands())) return 0;
    q36_q8_mm_args args = {
        (int32_t)in_dim, 1, row_bytes, row_bytes * out_dim,
        row_bytes * out_dim, 1, sizeof(float), in_dim * sizeof(float),
        in_dim * tokens * sizeof(float), in_dim * tokens * sizeof(float),
        (int32_t)out_dim, (int32_t)tokens, 1, 1
    };
    id<MTLComputeCommandEncoder> enc = [q36_batch computeCommandEncoder];
    enc.label = @"kernel_mul_mm_f32_ff32";
    [enc setComputePipelineState:p];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:weights offset:(NSUInteger)inner atIndex:1];
    [enc setBuffer:x->buffer offset:x->offset atIndex:2];
    [enc setBuffer:out->buffer offset:out->offset atIndex:3];
    [enc setThreadgroupMemoryLength:12288u atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((tokens + 31u) / 32u,
                                          (out_dim + 63u) / 64u, 1)
         threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    if (scale != 1.0f) {
        const uint64_t count64 = tokens * out_dim;
        if (count64 > UINT32_MAX) return 0;
        const uint32_t count = (uint32_t)count64;
        enc = q36_encoder(@"q36_scale_f32");
        if (!enc) return 0;
        [enc setBuffer:out->buffer offset:out->offset atIndex:0];
        [enc setBytes:&count length:sizeof(count) atIndex:1];
        [enc setBytes:&scale length:sizeof(scale) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    return 1;
}

static int q36_q8_decode(q36_gpu_tensor *out, const void *map, uint64_t size,
                         uint64_t offset, uint64_t in_dim, uint64_t out_dim,
                         const q36_gpu_tensor *x, float scale) {
    if (!out || !map || !x || !in_dim || !out_dim || (in_dim & 31u) ||
        in_dim > INT32_MAX || out_dim > INT32_MAX) return 0;
    uint64_t row_bytes = (in_dim / 32u) * 34u;
    if (out_dim > UINT64_MAX / row_bytes) return 0;
    uint64_t inner = 0;
    id<MTLBuffer> weights =
        q36_model_view(map, size, offset, out_dim * row_bytes, &inner);
    const char *rows_env = getenv("Q36_METAL_Q8_MV_ROWS");
    const int32_t nr0 = rows_env && rows_env[0] == '2' ? 2 : 4;
    int16_t nsg = 2;
    const char *nsg_env = getenv("Q36_METAL_Q8_MV_NSG");
    if (nsg_env && nsg_env[0] >= '1' && nsg_env[0] <= '8' &&
        nsg_env[1] == '\0')
        nsg = (int16_t)(nsg_env[0] - '0');
    id<MTLComputePipelineState> p =
        q36_pipeline_nsg(nr0 == 4
            ? @"kernel_mul_mv_q8_0_f32_r4"
            : @"kernel_mul_mv_q8_0_f32", nsg);
    if (!weights || !p || (!q36_batch && !q36_gpu_begin_commands())) return 0;
    q36_q8_mv_args args = {
        (int32_t)in_dim, (int32_t)out_dim, 1,
        34u, row_bytes, row_bytes * out_dim, row_bytes * out_dim,
        (int32_t)in_dim, 1, 1,
        sizeof(float), in_dim * sizeof(float), in_dim * sizeof(float),
        in_dim * sizeof(float), (int32_t)out_dim, 1, nr0, 1, 1
    };
    id<MTLComputeCommandEncoder> enc = [q36_batch computeCommandEncoder];
    enc.label = nr0 == 4 ? @"kernel_mul_mv_q8_0_f32_r4"
                         : @"kernel_mul_mv_q8_0_f32";
    [enc setComputePipelineState:p];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:weights offset:(NSUInteger)inner atIndex:1];
    [enc setBuffer:x->buffer offset:x->offset atIndex:2];
    [enc setBuffer:out->buffer offset:out->offset atIndex:3];
    [enc setThreadgroupMemoryLength:32u * (NSUInteger)nr0 * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(
            (out_dim + (uint64_t)nr0 - 1u) / (uint64_t)nr0, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)nsg, 1)];
    [enc endEncoding];
    if (scale != 1.0f) {
        uint32_t n = (uint32_t)out_dim;
        enc = q36_encoder(@"q36_scale_f32");
        if (!enc) return 0;
        [enc setBuffer:out->buffer offset:out->offset atIndex:0];
        [enc setBytes:&n length:sizeof(n) atIndex:1];
        [enc setBytes:&scale length:sizeof(scale) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    return 1;
}

static int q36_f32_decode(q36_gpu_tensor *out, const void *map, uint64_t size,
                          uint64_t offset, uint64_t in_dim, uint64_t out_dim,
                          const q36_gpu_tensor *x) {
    if (!out || !map || !x || !in_dim || !out_dim ||
        in_dim > INT32_MAX || out_dim > INT32_MAX ||
        in_dim > UINT64_MAX / sizeof(float) ||
        out_dim > UINT64_MAX / (in_dim * sizeof(float)) ||
        x->bytes < in_dim * sizeof(float) ||
        out->bytes < out_dim * sizeof(float)) return 0;
    const uint64_t row_bytes = in_dim * sizeof(float);
    uint64_t inner = 0;
    id<MTLBuffer> weights = q36_model_view(
        map, size, offset, out_dim * row_bytes, &inner);
    int16_t nsg = (int16_t)((in_dim + 127u) / 128u);
    if (nsg > 8) nsg = 8;
    const int32_t nr0 = 2;
    NSString *kernel = (in_dim & 3u)
        ? @"kernel_mul_mv_f32_f32" : @"kernel_mul_mv_f32_f32_4";
    id<MTLComputePipelineState> p = q36_pipeline_nsg(kernel, nsg);
    if (!weights || !p ||
        (!q36_batch && !q36_gpu_begin_commands())) return 0;
    q36_q8_mv_args args = {
        (int32_t)in_dim, (int32_t)out_dim, 1,
        sizeof(float), row_bytes, row_bytes * out_dim,
        row_bytes * out_dim, (int32_t)in_dim, 1, 1,
        sizeof(float), in_dim * sizeof(float),
        in_dim * sizeof(float), in_dim * sizeof(float),
        (int32_t)out_dim, 1, nr0, 1, 1
    };
    id<MTLComputeCommandEncoder> enc = [q36_batch computeCommandEncoder];
    enc.label = kernel;
    [enc setComputePipelineState:p];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:weights offset:(NSUInteger)inner atIndex:1];
    [enc setBuffer:x->buffer offset:x->offset atIndex:2];
    [enc setBuffer:out->buffer offset:out->offset atIndex:3];
    [enc setThreadgroupMemoryLength:32u * 2u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((out_dim + 1u) / 2u, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)nsg, 1)];
    [enc endEncoding];
    return 1;
}

static int q36_scale_rows(q36_gpu_tensor *out, uint64_t count64, float scale) {
    if (scale == 1.0f) return 1;
    if (!out || count64 > UINT32_MAX) return 0;
    uint32_t count = (uint32_t)count64;
    id<MTLComputeCommandEncoder> enc = q36_encoder(@"q36_scale_f32");
    if (!enc) return 0;
    [enc setBuffer:out->buffer offset:out->offset atIndex:0];
    [enc setBytes:&count length:sizeof(count) atIndex:1];
    [enc setBytes:&scale length:sizeof(scale) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(count, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    return 1;
}

static int q36_quant_float_matmul(
        NSString *mv_stem, NSString *mm_kernel, uint64_t block_bytes,
        q36_gpu_tensor *out, const void *map, uint64_t size, uint64_t offset,
        uint64_t in_dim, uint64_t out_dim, const q36_gpu_tensor *x,
        uint64_t tokens, float scale) {
    if (!mv_stem || !mm_kernel || !block_bytes || !out || !map || !x ||
        !in_dim || (in_dim & 255u) || !out_dim || !tokens ||
        in_dim > INT32_MAX || out_dim > INT32_MAX || tokens > INT32_MAX ||
        x->bytes < in_dim * tokens * sizeof(float) ||
        out->bytes < out_dim * tokens * sizeof(float)) return 0;
    const uint64_t row_bytes = (in_dim / 256u) * block_bytes;
    if (!row_bytes || out_dim > UINT64_MAX / row_bytes) return 0;
    uint64_t inner = 0;
    id<MTLBuffer> weights = q36_model_view(
        map, size, offset, out_dim * row_bytes, &inner);
    if (!weights || (!q36_batch && !q36_gpu_begin_commands())) return 0;

    if (tokens <= 5u) {
        NSString *kernel = [NSString stringWithFormat:@"%@%llu", mv_stem,
                            (unsigned long long)tokens];
        const int16_t nsg = 4, nxpsg = 32;
        id<MTLComputePipelineState> p =
            q36_pipeline_mv_ext(kernel, nsg, nxpsg);
        if (!p) return 0;
        q36_q8_mv_args args = {
            (int32_t)in_dim, (int32_t)out_dim, 1,
            block_bytes, row_bytes, row_bytes * out_dim,
            row_bytes * out_dim, (int32_t)in_dim, (int32_t)tokens, 1,
            sizeof(float), in_dim * sizeof(float),
            in_dim * tokens * sizeof(float),
            in_dim * tokens * sizeof(float),
            (int32_t)out_dim, (int32_t)tokens, 1, 1, 1
        };
        id<MTLComputeCommandEncoder> enc = [q36_batch computeCommandEncoder];
        enc.label = kernel;
        [enc setComputePipelineState:p];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:weights offset:(NSUInteger)inner atIndex:1];
        [enc setBuffer:x->buffer offset:x->offset atIndex:2];
        [enc setBuffer:out->buffer offset:out->offset atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((out_dim + nsg - 1u) / nsg,
                                              1, 1)
             threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
        [enc endEncoding];
    } else {
        const bool bc_inp = false;
        const bool bc_out = (out_dim % 64u) != 0 || (tokens % 32u) != 0;
        id<MTLComputePipelineState> p =
            q36_pipeline_mm(mm_kernel, bc_inp, bc_out);
        if (!p) return 0;
        q36_q8_mm_args args = {
            (int32_t)in_dim, 1, row_bytes, row_bytes * out_dim,
            row_bytes * out_dim, 1, sizeof(float), in_dim * sizeof(float),
            in_dim * tokens * sizeof(float), in_dim * tokens * sizeof(float),
            (int32_t)out_dim, (int32_t)tokens, 1, 1
        };
        id<MTLComputeCommandEncoder> enc = [q36_batch computeCommandEncoder];
        enc.label = mm_kernel;
        [enc setComputePipelineState:p];
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:weights offset:(NSUInteger)inner atIndex:1];
        [enc setBuffer:x->buffer offset:x->offset atIndex:2];
        [enc setBuffer:out->buffer offset:out->offset atIndex:3];
        [enc setThreadgroupMemoryLength:(bc_out ? 8192u : 6144u) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake((tokens + 31u) / 32u,
                                              (out_dim + 63u) / 64u, 1)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];
    }
    return q36_scale_rows(out, tokens * out_dim, scale);
}

static int q36_dense(NSString *kernel, q36_gpu_tensor *out,
                      const void *map, uint64_t model_size,
                      uint64_t weight_offset, uint64_t weight_bytes,
                      uint64_t in_dim, uint64_t out_dim,
                      const q36_gpu_tensor *x, uint64_t tokens, float scale) {
    if (!out || !x || !map || !in_dim || !out_dim || !tokens ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || tokens > UINT32_MAX ||
        in_dim > UINT64_MAX / tokens / sizeof(float) ||
        out_dim > UINT64_MAX / tokens / sizeof(float) ||
        x->bytes < in_dim * tokens * sizeof(float) ||
        out->bytes < out_dim * tokens * sizeof(float)) return 0;
    uint64_t inner = 0;
    id<MTLBuffer> weights =
        q36_model_view(map, model_size, weight_offset, weight_bytes, &inner);
    if (!weights) {
        fprintf(stderr, "q36: Metal cannot wrap model projection at %.3f GiB\n",
                (double)weight_offset / (1024.0 * 1024.0 * 1024.0));
        return 0;
    }
    id<MTLComputeCommandEncoder> enc = q36_encoder(kernel);
    if (!enc) return 0;
    q36_dense_args args = {
        (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)tokens, scale
    };
    [enc setBuffer:out->buffer offset:out->offset atIndex:0];
    [enc setBuffer:weights offset:(NSUInteger)inner atIndex:1];
    [enc setBuffer:x->buffer offset:x->offset atIndex:2];
    [enc setBytes:&args length:sizeof(args) atIndex:3];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)out_dim, (NSUInteger)tokens, 1)
   threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];
    return 1;
}

static void q36_dispatch_1d(id<MTLComputeCommandEncoder> enc, uint64_t n) {
    [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
}

int q36_gpu_init(void) {
    @autoreleasepool {
        if (q36_device) return 1;
        q36_device = MTLCreateSystemDefaultDevice();
        if (!q36_device) {
            fprintf(stderr, "q36: no Metal device is available\n");
            return 0;
        }
        q36_gpu_print_device_summary();
        q36_queue = [q36_device newCommandQueue];
        if (!q36_queue) {
            q36_device = nil;
            fprintf(stderr, "q36: failed to create Metal command queue\n");
            return 0;
        }
        NSError *error = nil;
        MTLCompileOptions *options = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeFast;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = YES;
#pragma clang diagnostic pop
        }
        q36_library = [q36_device newLibraryWithSource:q36_metal_source()
                                               options:options error:&error];
        if (!q36_library) {
            fprintf(stderr, "q36: Metal library compilation failed: %s\n",
                    error.localizedDescription.UTF8String);
            q36_queue = nil;
            q36_device = nil;
            return 0;
        }
        q36_pipelines = [NSMutableDictionary dictionary];
        q36_model_views = [NSMutableDictionary dictionary];
        return 1;
    }
}

void q36_gpu_cleanup(void) {
    @autoreleasepool {
        (void)q36_metal_wait();
        if (q36_live_bytes) {
            fprintf(stderr, "q36: Metal cleanup with %.2f MiB of live tensors\n",
                    (double)q36_live_bytes / (1024.0 * 1024.0));
        }
        q36_model_map = NULL;
        q36_model_size = 0;
        q36_model_residency_clear();
        q36_streaming_cache_clear(true);
        q36_model_fd = -1;
        q36_queue = nil;
        q36_moe_gate_scratch = nil;
        q36_moe_up_scratch = nil;
        q36_moe_down_scratch = nil;
        q36_moe_id_map = nil;
        q36_moe_pair_scratch_bytes = 0;
        q36_moe_down_scratch_bytes = 0;
        q36_moe_id_map_bytes = 0;
        q36_shared_gate_scratch = nil;
        q36_shared_up_scratch = nil;
        q36_shared_down_scratch = nil;
        q36_shared_mid_scratch_bytes = 0;
        q36_shared_down_scratch_bytes = 0;
        q36_pipelines = nil;
        q36_model_views = nil;
        q36_library = nil;
        q36_device = nil;
    }
}

q36_gpu_tensor *q36_gpu_tensor_alloc(uint64_t bytes) {
    if (!bytes || bytes > (uint64_t)NSUIntegerMax) return NULL;
    if (!q36_device && !q36_gpu_init()) return NULL;
    q36_gpu_tensor *tensor = calloc(1, sizeof(*tensor));
    if (!tensor) return NULL;
    tensor->buffer = [q36_device newBufferWithLength:(NSUInteger)bytes
                                             options:MTLResourceStorageModeShared];
    if (!tensor->buffer) {
        free(tensor);
        return NULL;
    }
    tensor->bytes = bytes;
    tensor->owner = true;
    pthread_mutex_lock(&q36_mu);
    q36_live_bytes += bytes;
    if (q36_live_bytes > q36_peak_bytes) q36_peak_bytes = q36_live_bytes;
    pthread_mutex_unlock(&q36_mu);
    return tensor;
}

q36_gpu_tensor *q36_gpu_tensor_alloc_scratch(uint64_t bytes) {
    /* Apple silicon uses unified memory, so Metal scratch has the same
     * storage mode as other tensors. Keep the backend-neutral scratch API
     * while Vulkan selects private VRAM in its implementation. */
    return q36_gpu_tensor_alloc(bytes);
}

q36_gpu_tensor *q36_gpu_tensor_view(const q36_gpu_tensor *base,
                                    uint64_t offset, uint64_t bytes) {
    if (!base || offset > base->bytes || bytes > base->bytes - offset) return NULL;
    q36_gpu_tensor *view = calloc(1, sizeof(*view));
    if (!view) return NULL;
    view->buffer = base->buffer;
    view->offset = base->offset + offset;
    view->bytes = bytes;
    view->owner = false;
    return view;
}

void q36_gpu_tensor_free(q36_gpu_tensor *tensor) {
    if (!tensor) return;
    if (tensor->owner) {
        pthread_mutex_lock(&q36_mu);
        q36_live_bytes -= tensor->bytes;
        pthread_mutex_unlock(&q36_mu);
    }
    tensor->buffer = nil;
    free(tensor);
}

uint64_t q36_gpu_tensor_bytes(const q36_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

void *q36_gpu_tensor_contents(q36_gpu_tensor *tensor) {
    if (!tensor || !q36_metal_wait()) return NULL;
    return (uint8_t *)tensor->buffer.contents + tensor->offset;
}

void *q36_gpu_tensor_contents_named(q36_gpu_tensor *tensor, const char *reason) {
    (void)reason;
    return q36_gpu_tensor_contents(tensor);
}

int q36_gpu_tensor_write(q36_gpu_tensor *tensor, uint64_t offset,
                         const void *data, uint64_t bytes) {
    if (!tensor || (!data && bytes) ||
        offset > tensor->bytes || bytes > tensor->bytes - offset ||
        !q36_metal_wait()) return 0;
    if (bytes) memcpy((uint8_t *)tensor->buffer.contents + tensor->offset + offset,
                      data, (size_t)bytes);
    return 1;
}

int q36_gpu_tensor_read(const q36_gpu_tensor *tensor, uint64_t offset,
                        void *data, uint64_t bytes) {
    if (!tensor || (!data && bytes) ||
        offset > tensor->bytes || bytes > tensor->bytes - offset ||
        !q36_metal_wait()) return 0;
    if (bytes) memcpy(data,
                      (const uint8_t *)tensor->buffer.contents + tensor->offset + offset,
                      (size_t)bytes);
    return 1;
}

int q36_gpu_tensor_copy(q36_gpu_tensor *dst, uint64_t dst_offset,
                        const q36_gpu_tensor *src, uint64_t src_offset,
                        uint64_t bytes) {
    if (!dst || !src || dst_offset > dst->bytes ||
        bytes > dst->bytes - dst_offset || src_offset > src->bytes ||
        bytes > src->bytes - src_offset) return 0;
    if (!bytes) return 1;
    if (!q36_batch && !q36_gpu_begin_commands()) return 0;
    id<MTLBlitCommandEncoder> blit = [q36_batch blitCommandEncoder];
    if (!blit) return 0;
    [blit copyFromBuffer:src->buffer sourceOffset:(NSUInteger)(src->offset + src_offset)
                toBuffer:dst->buffer destinationOffset:(NSUInteger)(dst->offset + dst_offset)
                    size:(NSUInteger)bytes];
    [blit endEncoding];
    return 1;
}

int q36_gpu_begin_commands(void) {
    if (!q36_device && !q36_gpu_init()) return 0;
    pthread_mutex_lock(&q36_mu);
    if (!q36_batch) q36_batch = [q36_queue commandBuffer];
    int ok = q36_batch != nil;
    pthread_mutex_unlock(&q36_mu);
    return ok;
}

int q36_gpu_flush_commands(void) { return q36_metal_wait(); }
int q36_gpu_end_commands(void) { return q36_metal_wait(); }
int q36_gpu_synchronize(void) { return q36_metal_wait(); }

int q36_gpu_set_model_map(const void *map, uint64_t size) {
    if (!map || !size) return 0;
    if (map != q36_model_map || size != q36_model_size) {
        q36_model_residency_clear();
        q36_streaming_cache_clear(true);
        [q36_model_views removeAllObjects];
    }
    q36_model_map = map;
    q36_model_size = size;
    return 1;
}

int q36_gpu_set_model_fd(int fd) {
    q36_model_fd = fd;
    return fd >= 0;
}

int q36_gpu_cache_model_range(const void *map, uint64_t size,
                              uint64_t offset, uint64_t bytes,
                              const char *label) {
    (void)label;
    if (!map || offset > size || bytes > size - offset) return 0;
    if (q36_ssd_streaming || map != q36_model_map ||
        size != q36_model_size) return 1;
    uint64_t inner = 0;
    return q36_model_view(map, size, offset, bytes, &inner) != nil;
}

int q36_gpu_finish_model_cache(void) {
    if (q36_ssd_streaming || getenv("Q36_METAL_NO_RESIDENCY") ||
        q36_model_residency_set || q36_model_views.count == 0)
        return 1;
    if (@available(macOS 15.0, *)) {
        MTLResidencySetDescriptor *desc =
            [[MTLResidencySetDescriptor alloc] init];
        desc.label = @"q36_model";
        desc.initialCapacity = q36_model_views.count;
        NSError *error = nil;
        q36_model_residency_set =
            [q36_device newResidencySetWithDescriptor:desc error:&error];
        if (!q36_model_residency_set) {
            fprintf(stderr,
                    "q36: Metal model residency set creation failed: %s\n",
                    error.localizedDescription.UTF8String);
            return 0;
        }
        for (NSArray *view in q36_model_views.allValues)
            [q36_model_residency_set addAllocation:view[0]];
        [q36_model_residency_set commit];
        [q36_model_residency_set requestResidency];
        if (q36_queue &&
            [q36_queue respondsToSelector:@selector(addResidencySet:)]) {
            [q36_queue addResidencySet:q36_model_residency_set];
            q36_model_residency_added = true;
        }
    }
    return 1;
}

int q36_gpu_cache_q8_f16_range(const void *map, uint64_t size,
                               uint64_t offset, uint64_t bytes,
                               uint64_t in_dim, uint64_t out_dim,
                               const char *label) {
    (void)in_dim; (void)out_dim;
    return q36_gpu_cache_model_range(map, size, offset, bytes, label);
}

void q36_gpu_set_quality(bool enabled) { q36_quality = enabled; }
void q36_gpu_set_dense_model(bool dense) { q36_dense_model = dense; }
void q36_gpu_set_micro_batch(bool enabled) { q36_micro_batch = enabled; }
bool q36_gpu_attn_fused_enabled(void) { return false; }

static bool q36_streaming_table_valid(
        const q36_gpu_stream_expert_table *table) {
    return table && table->model_map && table->model_size &&
           table->layer < Q36_METAL_STREAM_MAX_LAYERS &&
           table->n_total_expert > 0 &&
           table->n_total_expert <= Q36_METAL_STREAM_MAX_EXPERTS &&
           table->gate_expert_bytes && table->up_expert_bytes &&
           table->down_expert_bytes;
}

static bool q36_u64_add3(uint64_t a, uint64_t b, uint64_t c,
                         uint64_t *out) {
    if (!out || a > UINT64_MAX - b || a + b > UINT64_MAX - c) return false;
    *out = a + b + c;
    return true;
}

static bool q36_streaming_read(void *dst, uint64_t bytes, uint64_t offset) {
    if ((!dst && bytes) || !q36_model_map ||
        offset > q36_model_size || bytes > q36_model_size - offset ||
        bytes > (uint64_t)SIZE_MAX) {
        return false;
    }
    if (q36_model_fd < 0) {
        memcpy(dst, (const uint8_t *)q36_model_map + offset, (size_t)bytes);
        return true;
    }
    uint8_t *p = dst;
    uint64_t done = 0;
    while (done < bytes) {
        size_t chunk = (size_t)(bytes - done);
        if ((uint64_t)chunk != bytes - done) chunk = SIZE_MAX;
        ssize_t n = pread(q36_model_fd, p + done, chunk,
                          (off_t)(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        done += (uint64_t)n;
    }
    q36_streaming_read_bytes += bytes;
    return true;
}

static void q36_streaming_cache_clear(bool clear_full_layers) {
    pthread_mutex_lock(&q36_stream_mu);
    q36_streaming_gate = nil;
    q36_streaming_up = nil;
    q36_streaming_down = nil;
    free(q36_streaming_entries);
    q36_streaming_entries = NULL;
    q36_streaming_expert_cap = 0;
    q36_streaming_expert_count = 0;
    q36_streaming_gate_bytes = 0;
    q36_streaming_up_bytes = 0;
    q36_streaming_down_bytes = 0;
    for (uint32_t l = 0; l < Q36_METAL_STREAM_MAX_LAYERS; l++)
        for (uint32_t e = 0; e < Q36_METAL_STREAM_MAX_EXPERTS; e++)
            q36_streaming_lookup[l][e] = -1;
    if (clear_full_layers) {
        for (uint32_t l = 0; l < Q36_METAL_STREAM_MAX_LAYERS; l++) {
            q36_streaming_full[l].gate = nil;
            q36_streaming_full[l].up = nil;
            q36_streaming_full[l].down = nil;
            q36_streaming_full[l].gate_bytes = 0;
            q36_streaming_full[l].up_bytes = 0;
            q36_streaming_full[l].down_bytes = 0;
            q36_streaming_full[l].valid = false;
        }
        memset(q36_streaming_bias, 0, sizeof(q36_streaming_bias));
        memset(q36_streaming_hotness, 0, sizeof(q36_streaming_hotness));
        q36_streaming_clock = 0;
        q36_streaming_hits = 0;
        q36_streaming_misses = 0;
        q36_streaming_loads = 0;
        q36_streaming_evictions = 0;
        q36_streaming_read_bytes = 0;
        q36_streaming_route_tokens = 0;
    }
    pthread_mutex_unlock(&q36_stream_mu);
}

static bool q36_streaming_layout_matches(
        const q36_gpu_stream_expert_table *table) {
    return !q36_streaming_expert_cap ||
           (q36_streaming_gate_bytes >= table->gate_expert_bytes &&
            q36_streaming_up_bytes >= table->up_expert_bytes &&
            q36_streaming_down_bytes >= table->down_expert_bytes);
}

static bool q36_streaming_cache_init(
        const q36_gpu_stream_expert_table *table) {
    if (!q36_ssd_streaming || !q36_streaming_table_valid(table) ||
        q36_streaming_expert_budget == 0) return false;
    if (q36_streaming_expert_cap)
        return q36_streaming_layout_matches(table);
    if (q36_streaming_expert_bytes) {
        uint64_t total = 0;
        if (!q36_u64_add3(table->gate_expert_bytes,
                          table->up_expert_bytes,
                          table->down_expert_bytes, &total) ||
            total > q36_streaming_expert_bytes) {
            return false;
        }
    }

    const uint64_t gate_stride = q36_streaming_config_gate_bytes
        ? q36_streaming_config_gate_bytes : table->gate_expert_bytes;
    const uint64_t up_stride = q36_streaming_config_up_bytes
        ? q36_streaming_config_up_bytes : table->up_expert_bytes;
    const uint64_t down_stride = q36_streaming_config_down_bytes
        ? q36_streaming_config_down_bytes : table->down_expert_bytes;
    if (gate_stride < table->gate_expert_bytes ||
        up_stride < table->up_expert_bytes ||
        down_stride < table->down_expert_bytes)
        return false;
    uint32_t cap = q36_streaming_expert_budget;
    const uint32_t max_entries =
        Q36_METAL_STREAM_MAX_LAYERS * Q36_METAL_STREAM_MAX_EXPERTS;
    if (cap > max_entries) cap = max_entries;
    while (cap) {
        uint64_t gb = gate_stride * (uint64_t)cap;
        uint64_t ub = up_stride * (uint64_t)cap;
        uint64_t db = down_stride * (uint64_t)cap;
        bool overflow =
            gb / cap != gate_stride ||
            ub / cap != up_stride ||
            db / cap != down_stride ||
            gb > (uint64_t)NSUIntegerMax ||
            ub > (uint64_t)NSUIntegerMax ||
            db > (uint64_t)NSUIntegerMax;
        q36_metal_stream_entry *entries =
            overflow ? NULL : calloc(cap, sizeof(*entries));
        id<MTLBuffer> gate = entries ?
            [q36_device newBufferWithLength:(NSUInteger)gb
                                    options:MTLResourceStorageModeShared] : nil;
        id<MTLBuffer> up = gate ?
            [q36_device newBufferWithLength:(NSUInteger)ub
                                    options:MTLResourceStorageModeShared] : nil;
        id<MTLBuffer> down = up ?
            [q36_device newBufferWithLength:(NSUInteger)db
                                    options:MTLResourceStorageModeShared] : nil;
        if (entries && gate && up && down) {
            gate.label = @"q36_stream_expert_gate";
            up.label = @"q36_stream_expert_up";
            down.label = @"q36_stream_expert_down";
            q36_streaming_entries = entries;
            q36_streaming_gate = gate;
            q36_streaming_up = up;
            q36_streaming_down = down;
            q36_streaming_expert_cap = cap;
            q36_streaming_gate_bytes = gate_stride;
            q36_streaming_up_bytes = up_stride;
            q36_streaming_down_bytes = down_stride;
            fprintf(stderr,
                    "q36: Metal SSD expert cache allocated %u slots (%.2f GiB)\n",
                    cap, (double)(gb + ub + db) / 1073741824.0);
            return true;
        }
        free(entries);
        uint32_t next = q36_ssd_shrink_cache_experts(cap);
        fprintf(stderr,
                "q36: Metal SSD expert cache allocation failed at %u slots; shrinking to %u\n",
                cap, next);
        cap = next;
    }
    return false;
}

static bool q36_streaming_id_protected(
        const q36_metal_stream_entry *entry,
        const q36_gpu_stream_expert_table *table,
        const int32_t *ids, uint32_t count) {
    if (!entry || !entry->valid || entry->layer != table->layer) return false;
    for (uint32_t i = 0; i < count; i++)
        if (ids[i] >= 0 && entry->expert == (uint32_t)ids[i]) return true;
    return false;
}

static int32_t q36_streaming_pick_slot(
        const q36_gpu_stream_expert_table *table,
        const int32_t *protect_ids, uint32_t protect_count) {
    for (uint32_t i = 0; i < q36_streaming_expert_cap; i++)
        if (!q36_streaming_entries[i].valid) return (int32_t)i;
    int32_t best = -1;
    uint32_t best_priority = UINT32_MAX;
    uint64_t best_used = UINT64_MAX;
    for (uint32_t i = 0; i < q36_streaming_expert_cap; i++) {
        q36_metal_stream_entry *entry = &q36_streaming_entries[i];
        if (q36_streaming_id_protected(entry, table,
                                       protect_ids, protect_count)) continue;
        uint32_t priority = entry->priority;
        if (entry->layer < Q36_METAL_STREAM_MAX_LAYERS &&
            entry->expert < Q36_METAL_STREAM_MAX_EXPERTS) {
            priority += q36_streaming_bias[entry->layer][entry->expert];
            priority += q36_streaming_hotness[entry->layer][entry->expert];
        }
        if (best < 0 || priority < best_priority ||
            (priority == best_priority && entry->last_used < best_used)) {
            best = (int32_t)i;
            best_priority = priority;
            best_used = entry->last_used;
        }
    }
    return best;
}

static bool q36_streaming_load_slot(
        const q36_gpu_stream_expert_table *table,
        uint32_t expert, uint32_t slot, uint32_t priority) {
    if (slot >= q36_streaming_expert_cap ||
        expert >= table->n_total_expert) return false;
    q36_metal_stream_entry *entry = &q36_streaming_entries[slot];
    if (entry->valid) {
        q36_streaming_lookup[entry->layer][entry->expert] = -1;
        entry->valid = false;
        q36_streaming_evictions++;
    } else {
        q36_streaming_expert_count++;
    }
    uint64_t go = table->gate_offset +
                  (uint64_t)expert * table->gate_expert_bytes;
    uint64_t uo = table->up_offset +
                  (uint64_t)expert * table->up_expert_bytes;
    uint64_t doff = table->down_offset +
                    (uint64_t)expert * table->down_expert_bytes;
    uint8_t *gp = (uint8_t *)q36_streaming_gate.contents +
                  (uint64_t)slot * q36_streaming_gate_bytes;
    uint8_t *up = (uint8_t *)q36_streaming_up.contents +
                  (uint64_t)slot * q36_streaming_up_bytes;
    uint8_t *dp = (uint8_t *)q36_streaming_down.contents +
                  (uint64_t)slot * q36_streaming_down_bytes;
    if (!q36_streaming_read(gp, table->gate_expert_bytes, go) ||
        !q36_streaming_read(up, table->up_expert_bytes, uo) ||
        !q36_streaming_read(dp, table->down_expert_bytes, doff)) {
        q36_streaming_expert_count--;
        return false;
    }
    *entry = (q36_metal_stream_entry) {
        .layer = table->layer,
        .expert = expert,
        .last_used = ++q36_streaming_clock,
        .priority = priority,
        .valid = true,
    };
    q36_streaming_lookup[table->layer][expert] = (int32_t)slot;
    q36_streaming_loads++;
    return true;
}

static bool q36_streaming_prepare_locked(
        const q36_gpu_stream_expert_table *table,
        const int32_t *ids, const uint32_t *priorities, uint32_t count,
        uint32_t slot_by_expert[Q36_METAL_STREAM_MAX_EXPERTS]) {
    if (!ids || !q36_streaming_cache_init(table)) return false;
    bool seen[Q36_METAL_STREAM_MAX_EXPERTS] = {0};
    uint32_t unique = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (ids[i] < 0 || (uint32_t)ids[i] >= table->n_total_expert)
            return false;
        if (!seen[ids[i]]) {
            seen[ids[i]] = true;
            unique++;
        }
    }
    if (unique > q36_streaming_expert_cap) return false;
    if (slot_by_expert)
        for (uint32_t i = 0; i < Q36_METAL_STREAM_MAX_EXPERTS; i++)
            slot_by_expert[i] = UINT32_MAX;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t expert = (uint32_t)ids[i];
        int32_t slot = q36_streaming_lookup[table->layer][expert];
        uint32_t priority = priorities ? priorities[i] : 0;
        if (slot >= 0) {
            q36_metal_stream_entry *entry = &q36_streaming_entries[slot];
            entry->last_used = ++q36_streaming_clock;
            if (priority > entry->priority) entry->priority = priority;
            q36_streaming_hits++;
        } else {
            q36_streaming_misses++;
            slot = q36_streaming_pick_slot(table, ids, count);
            if (slot < 0 ||
                !q36_streaming_load_slot(table, expert, (uint32_t)slot,
                                         priority)) return false;
        }
        if (q36_streaming_hotness[table->layer][expert] != UINT32_MAX)
            q36_streaming_hotness[table->layer][expert]++;
        if (slot_by_expert) slot_by_expert[expert] = (uint32_t)slot;
    }
    return true;
}

static bool q36_streaming_prepare(
        const q36_gpu_stream_expert_table *table,
        const int32_t *ids, const uint32_t *priorities, uint32_t count,
        uint32_t slot_by_expert[Q36_METAL_STREAM_MAX_EXPERTS]) {
    if (!q36_metal_wait()) return false;
    pthread_mutex_lock(&q36_stream_mu);
    bool ok = q36_streaming_prepare_locked(table, ids, priorities, count,
                                            slot_by_expert);
    pthread_mutex_unlock(&q36_stream_mu);
    return ok;
}

static bool q36_streaming_full_layer_load(
        const q36_gpu_stream_expert_table *table) {
    if (!q36_streaming_table_valid(table) ||
        table->layer >= q36_streaming_full_layers) return false;
    q36_metal_full_layer *full = &q36_streaming_full[table->layer];
    if (full->valid)
        return full->gate_bytes ==
                   table->gate_expert_bytes * table->n_total_expert &&
               full->up_bytes ==
                   table->up_expert_bytes * table->n_total_expert &&
               full->down_bytes ==
                   table->down_expert_bytes * table->n_total_expert;
    uint64_t gb = table->gate_expert_bytes * table->n_total_expert;
    uint64_t ub = table->up_expert_bytes * table->n_total_expert;
    uint64_t db = table->down_expert_bytes * table->n_total_expert;
    if (gb > NSUIntegerMax || ub > NSUIntegerMax || db > NSUIntegerMax)
        return false;
    full->gate = [q36_device newBufferWithLength:(NSUInteger)gb
                                         options:MTLResourceStorageModeShared];
    full->up = full->gate ?
        [q36_device newBufferWithLength:(NSUInteger)ub
                                 options:MTLResourceStorageModeShared] : nil;
    full->down = full->up ?
        [q36_device newBufferWithLength:(NSUInteger)db
                                 options:MTLResourceStorageModeShared] : nil;
    if (!full->gate || !full->up || !full->down ||
        !q36_streaming_read(full->gate.contents, gb, table->gate_offset) ||
        !q36_streaming_read(full->up.contents, ub, table->up_offset) ||
        !q36_streaming_read(full->down.contents, db, table->down_offset)) {
        full->gate = nil;
        full->up = nil;
        full->down = nil;
        return false;
    }
    full->gate_bytes = gb;
    full->up_bytes = ub;
    full->down_bytes = db;
    full->valid = true;
    fprintf(stderr,
            "q36: Metal SSD full resident routed layer %u loaded (%.2f MiB)\n",
            table->layer, (double)(gb + ub + db) / 1048576.0);
    return true;
}

void q36_gpu_set_ssd_streaming(bool enabled) {
    if (q36_ssd_streaming != enabled) {
        (void)q36_metal_wait();
        q36_streaming_cache_clear(true);
        q36_ssd_streaming = enabled;
    }
}
void q36_gpu_set_streaming_full_layers(uint32_t layers) {
    if (layers != q36_streaming_full_layers) {
        (void)q36_metal_wait();
        q36_streaming_cache_clear(true);
        q36_streaming_full_layers = layers;
    }
}
void q36_gpu_set_streaming_expert_cache_budget(uint32_t experts) {
    if (experts != q36_streaming_expert_budget) {
        (void)q36_metal_wait();
        q36_streaming_cache_clear(false);
        q36_streaming_expert_budget = experts;
    }
}
void q36_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes) {
    if (bytes != q36_streaming_expert_bytes) {
        (void)q36_metal_wait();
        q36_streaming_cache_clear(false);
        q36_streaming_expert_bytes = bytes;
        q36_streaming_config_gate_bytes = 0;
        q36_streaming_config_up_bytes = 0;
        q36_streaming_config_down_bytes = 0;
    }
}
void q36_gpu_set_streaming_expert_cache_layout(
        uint64_t gate_bytes, uint64_t up_bytes, uint64_t down_bytes) {
    uint64_t total = 0;
    if (!q36_u64_add3(gate_bytes, up_bytes, down_bytes, &total)) total = 0;
    if (gate_bytes != q36_streaming_config_gate_bytes ||
        up_bytes != q36_streaming_config_up_bytes ||
        down_bytes != q36_streaming_config_down_bytes ||
        total != q36_streaming_expert_bytes) {
        (void)q36_metal_wait();
        q36_streaming_cache_clear(false);
        q36_streaming_config_gate_bytes = gate_bytes;
        q36_streaming_config_up_bytes = up_bytes;
        q36_streaming_config_down_bytes = down_bytes;
        q36_streaming_expert_bytes = total;
    }
}
uint32_t q36_gpu_stream_expert_cache_configured_count(void) {
    return q36_streaming_expert_cap ?
        q36_streaming_expert_cap : q36_streaming_expert_budget;
}
uint32_t q36_gpu_stream_expert_cache_current_count(void) {
    return q36_streaming_expert_count;
}
void q36_gpu_stream_expert_cache_reset_route_hotness(void) {
    pthread_mutex_lock(&q36_stream_mu);
    memset(q36_streaming_hotness, 0, sizeof(q36_streaming_hotness));
    pthread_mutex_unlock(&q36_stream_mu);
}
void q36_gpu_stream_expert_cache_note_tokens(uint32_t tokens) {
    if (!q36_ssd_streaming || !tokens) return;
    pthread_mutex_lock(&q36_stream_mu);
    q36_streaming_route_tokens += tokens;
    if ((q36_streaming_route_tokens & 15u) < tokens) {
        for (uint32_t l = 0; l < Q36_METAL_STREAM_MAX_LAYERS; l++)
            for (uint32_t e = 0; e < Q36_METAL_STREAM_MAX_EXPERTS; e++)
                q36_streaming_hotness[l][e] >>= 1;
    }
    pthread_mutex_unlock(&q36_stream_mu);
}
void q36_gpu_stream_expert_cache_release_resident(void) {
    (void)q36_metal_wait();
    q36_streaming_cache_clear(true);
}
uint32_t q36_gpu_stream_expert_cache_budget_for_expert_size(
        uint64_t gate_bytes, uint64_t up_bytes, uint64_t down_bytes) {
    uint64_t total = 0;
    if (!q36_u64_add3(gate_bytes, up_bytes, down_bytes, &total) ||
        (q36_streaming_expert_bytes &&
         q36_streaming_expert_bytes < total) ||
        (q36_streaming_config_gate_bytes &&
         q36_streaming_config_gate_bytes < gate_bytes) ||
        (q36_streaming_config_up_bytes &&
         q36_streaming_config_up_bytes < up_bytes) ||
        (q36_streaming_config_down_bytes &&
         q36_streaming_config_down_bytes < down_bytes)) return 0;
    return q36_gpu_stream_expert_cache_configured_count();
}

int q36_gpu_stream_expert_cache_seed_selected(
        const q36_gpu_stream_expert_table *table, const int32_t *ids,
        uint32_t count) {
    return q36_streaming_prepare(table, ids, NULL, count, NULL) ? 1 : 0;
}
int q36_gpu_stream_expert_cache_begin_selected_load(
        const q36_gpu_stream_expert_table *table, const int32_t *ids,
        uint32_t count) {
    return q36_gpu_stream_expert_cache_seed_selected(table, ids, count);
}
int q36_gpu_stream_expert_cache_prepare_selected_batch(
        const q36_gpu_stream_expert_table *table, const int32_t *ids,
        uint32_t tokens, uint32_t selected) {
    uint64_t count = (uint64_t)tokens * selected;
    if (count > UINT32_MAX) return 0;
    return q36_streaming_prepare(table, ids, NULL, (uint32_t)count, NULL)
        ? 1 : 0;
}
int q36_gpu_stream_expert_cache_load_layer(
        const q36_gpu_stream_expert_table *table) {
    if (!table) return 0;
    if (table->layer < q36_streaming_full_layers) {
        if (!q36_metal_wait()) return 0;
        pthread_mutex_lock(&q36_stream_mu);
        bool ok = q36_streaming_full_layer_load(table);
        pthread_mutex_unlock(&q36_stream_mu);
        return ok ? 1 : 0;
    }
    uint32_t count = table->n_total_expert;
    uint32_t budget = q36_gpu_stream_expert_cache_configured_count();
    if (budget && count > budget) count = budget;
    int32_t *ids = malloc((size_t)count * sizeof(*ids));
    if (!ids) return 0;
    for (uint32_t i = 0; i < count; i++) ids[i] = (int32_t)i;
    int ok = q36_streaming_prepare(table, ids, NULL, count, NULL) ? 1 : 0;
    free(ids);
    return ok;
}
int q36_gpu_stream_expert_cache_seed_from_layer_selected(
        const q36_gpu_stream_expert_table *table,
        const q36_gpu_tensor *selected, uint32_t tokens,
        uint32_t seed_tokens, uint32_t n_selected) {
    if (!selected || !n_selected) return 0;
    if (seed_tokens > tokens) seed_tokens = tokens;
    uint64_t count = (uint64_t)seed_tokens * n_selected;
    if (count > UINT32_MAX ||
        selected->bytes < count * sizeof(int32_t)) return 0;
    const int32_t *ids =
        q36_gpu_tensor_contents((q36_gpu_tensor *)selected);
    return ids && q36_streaming_prepare(table, ids, NULL,
                                         (uint32_t)count, NULL) ? 1 : 0;
}
int q36_gpu_stream_expert_cache_release_layer_cache(void) { return 1; }
int q36_gpu_stream_expert_cache_seed_experts(
        const q36_gpu_stream_expert_table *table, const int32_t *ids,
        const uint32_t *priorities, uint32_t count) {
    return q36_streaming_prepare(table, ids, priorities, count, NULL) ? 1 : 0;
}
int q36_gpu_stream_expert_cache_bias_experts(
        const q36_gpu_stream_expert_table *table, const int32_t *ids,
        const uint32_t *priorities, uint32_t count) {
    if (!q36_streaming_table_valid(table) || !ids) return 0;
    pthread_mutex_lock(&q36_stream_mu);
    for (uint32_t i = 0; i < count; i++) {
        if (ids[i] < 0 || (uint32_t)ids[i] >= table->n_total_expert) {
            pthread_mutex_unlock(&q36_stream_mu);
            return 0;
        }
        uint32_t priority = priorities ? priorities[i] : 1u;
        uint32_t *bias =
            &q36_streaming_bias[table->layer][(uint32_t)ids[i]];
        if (priority > *bias) *bias = priority;
    }
    pthread_mutex_unlock(&q36_stream_mu);
    return 1;
}

void q36_gpu_prof_reset(void) {
    q36_prof_gpu_seconds = 0.0;
    q36_prof_command_buffers = 0;
    q36_prof_active = true;
}
void q36_gpu_prof_report(const char *label) {
    q36_prof_active = false;
    fprintf(stderr,
            "q36: Metal %s profile command_buffers=%" PRIu64
            " gpu_busy_ms=%.3f avg_gpu_ms=%.3f\n",
            label ? label : "GPU",
            q36_prof_command_buffers,
            q36_prof_gpu_seconds * 1000.0,
            q36_prof_command_buffers
                ? q36_prof_gpu_seconds * 1000.0 /
                      (double)q36_prof_command_buffers
                : 0.0);
}

int q36_gpu_set_model_map_range(const void *map, uint64_t size,
                                uint64_t offset, uint64_t bytes) {
    if (!map || offset > size || bytes > size - offset) return 0;
    return q36_gpu_set_model_map(map, size);
}

int q36_gpu_set_model_map_spans(const void *map, uint64_t size,
                                const uint64_t *offsets, const uint64_t *sizes,
                                uint32_t count, uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!map || (!offsets && count) || (!sizes && count)) return 0;
    for (uint32_t i = 0; i < count; i++)
        if (offsets[i] > size || sizes[i] > size - offsets[i]) return 0;
    return q36_gpu_set_model_map(map, size);
}

void q36_gpu_parallel_for_rows(uint64_t rows, uint64_t grain,
                               q36_gpu_parallel_fn fn, void *ctx) {
    (void)grain;
    if (fn && rows) fn(ctx, 0, rows);
}

uint64_t q36_gpu_recommended_working_set_size(void) {
    if (!q36_device && !q36_gpu_init()) return 0;
    return q36_device ? (uint64_t)q36_device.recommendedMaxWorkingSetSize : 0;
}

void q36_gpu_print_memory_report(const char *label) {
    uint64_t cache_bytes = 0;
    uint64_t full_bytes = 0;
    pthread_mutex_lock(&q36_stream_mu);
    if (q36_streaming_gate) cache_bytes += q36_streaming_gate.length;
    if (q36_streaming_up) cache_bytes += q36_streaming_up.length;
    if (q36_streaming_down) cache_bytes += q36_streaming_down.length;
    for (uint32_t l = 0; l < Q36_METAL_STREAM_MAX_LAYERS; l++) {
        if (!q36_streaming_full[l].valid) continue;
        full_bytes += q36_streaming_full[l].gate_bytes;
        full_bytes += q36_streaming_full[l].up_bytes;
        full_bytes += q36_streaming_full[l].down_bytes;
    }
    fprintf(stderr,
            "q36: Metal memory%s%s: live %.2f MiB, peak %.2f MiB "
            "ssd=%s cache=%u/%u cache_gib=%.2f full_gib=%.2f "
            "hits=%" PRIu64 " misses=%" PRIu64 " loads=%" PRIu64
            " evictions=%" PRIu64 " reads_gib=%.2f\n",
            label ? " " : "", label ? label : "",
            (double)q36_live_bytes / (1024.0 * 1024.0),
            (double)q36_peak_bytes / (1024.0 * 1024.0),
            q36_ssd_streaming ? "on" : "off",
            q36_streaming_expert_count,
            q36_streaming_expert_cap ?
                q36_streaming_expert_cap : q36_streaming_expert_budget,
            (double)cache_bytes / 1073741824.0,
            (double)full_bytes / 1073741824.0,
            q36_streaming_hits, q36_streaming_misses,
            q36_streaming_loads, q36_streaming_evictions,
            (double)q36_streaming_read_bytes / 1073741824.0);
    pthread_mutex_unlock(&q36_stream_mu);
}

int q36_gpu_add_tensor(q36_gpu_tensor *out, const q36_gpu_tensor *a,
                       const q36_gpu_tensor *b, uint32_t n) {
    if (!out || !a || !b || out->bytes < (uint64_t)n * 4 ||
        a->bytes < (uint64_t)n * 4 || b->bytes < (uint64_t)n * 4) return 0;
    id<MTLComputeCommandEncoder> enc = q36_encoder(@"q36_add_f32");
    if (!enc) return 0;
    [enc setBuffer:out->buffer offset:out->offset atIndex:0];
    [enc setBuffer:a->buffer offset:a->offset atIndex:1];
    [enc setBuffer:b->buffer offset:b->offset atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    q36_dispatch_1d(enc, n);
    return 1;
}

int q36_gpu_copy_f32_tensor(q36_gpu_tensor *out, const q36_gpu_tensor *in,
                            uint32_t n) {
    if (!out || !in || out->bytes < (uint64_t)n * 4 ||
        in->bytes < (uint64_t)n * 4) return 0;
    id<MTLComputeCommandEncoder> enc = q36_encoder(@"q36_copy_f32");
    if (!enc) return 0;
    [enc setBuffer:out->buffer offset:out->offset atIndex:0];
    [enc setBuffer:in->buffer offset:in->offset atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    q36_dispatch_1d(enc, n);
    return 1;
}

int q36_gpu_swiglu_tensor(q36_gpu_tensor *out, const q36_gpu_tensor *gate,
                          const q36_gpu_tensor *up, uint32_t n,
                          float limit, float weight) {
    if (!out || !gate || !up || out->bytes < (uint64_t)n * 4 ||
        gate->bytes < (uint64_t)n * 4 || up->bytes < (uint64_t)n * 4) return 0;
    id<MTLComputeCommandEncoder> enc = q36_encoder(@"q36_swiglu_f32");
    if (!enc) return 0;
    [enc setBuffer:out->buffer offset:out->offset atIndex:0];
    [enc setBuffer:gate->buffer offset:gate->offset atIndex:1];
    [enc setBuffer:up->buffer offset:up->offset atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    [enc setBytes:&limit length:sizeof(limit) atIndex:4];
    [enc setBytes:&weight length:sizeof(weight) atIndex:5];
    q36_dispatch_1d(enc, n);
    return 1;
}

int q36_gpu_ffn_tail_tensor(q36_gpu_tensor *out,
                            const q36_gpu_tensor *shared,
                            const q36_gpu_tensor *scalar,
                            uint32_t width, uint32_t rows) {
    if (!out || !shared || !scalar) return 0;
    id<MTLComputeCommandEncoder> enc = q36_encoder(@"q36_ffn_tail_f32");
    if (!enc) return 0;
    [enc setBuffer:out->buffer offset:out->offset atIndex:0];
    [enc setBuffer:shared->buffer offset:shared->offset atIndex:1];
    [enc setBuffer:scalar->buffer offset:scalar->offset atIndex:2];
    [enc setBytes:&width length:sizeof(width) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(width, rows, 1)
   threadsPerThreadgroup:MTLSizeMake(MIN(width, 256u), 1, 1)];
    [enc endEncoding];
    return 1;
}

static int q36_norm(q36_gpu_tensor *out, const q36_gpu_tensor *in,
                     const void *map, uint64_t size, uint64_t weight_offset,
                     uint32_t width, uint32_t rows, float eps) {
    if (!out || !in || (uint64_t)width * rows * 4 > out->bytes ||
        (uint64_t)width * rows * 4 > in->bytes) return 0;
    id<MTLBuffer> wbuf = nil;
    uint64_t weight_inner = 0;
    uint32_t has_weight = map != NULL;
    if (map) {
        const uint64_t weight_bytes = (uint64_t)width * sizeof(float);
        if (weight_offset > size || weight_bytes > size - weight_offset)
            return 0;
        wbuf = q36_model_view(map, size, weight_offset, weight_bytes,
                              &weight_inner);
        if (!wbuf) return 0;
    } else {
        wbuf = in->buffer;
    }
    id<MTLComputeCommandEncoder> enc = q36_encoder(@"q36_rms_norm_f32");
    if (!enc) return 0;
    [enc setBuffer:out->buffer offset:out->offset atIndex:0];
    [enc setBuffer:in->buffer offset:in->offset atIndex:1];
    [enc setBuffer:wbuf
            offset:(NSUInteger)(map ? weight_inner : in->offset)
           atIndex:2];
    [enc setBytes:&width length:sizeof(width) atIndex:3];
    [enc setBytes:&eps length:sizeof(eps) atIndex:4];
    [enc setBytes:&has_weight length:sizeof(has_weight) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];
    return 1;
}

int q36_gpu_rms_norm_plain_rows_tensor(q36_gpu_tensor *out,
                                        const q36_gpu_tensor *in,
                                        uint32_t width, uint32_t rows,
                                        float eps) {
    return q36_norm(out, in, NULL, 0, 0, width, rows, eps);
}

int q36_gpu_rms_norm_weight_rows_tensor(q36_gpu_tensor *out,
                                         const q36_gpu_tensor *in,
                                         const void *map, uint64_t size,
                                         uint64_t offset, uint32_t width,
                                         uint32_t rows, float eps) {
    if (!map || offset > size || (uint64_t)width * 4 > size - offset) return 0;
    return q36_norm(out, in, map, size, offset, width, rows, eps);
}

int q36_gpu_recurrent_norm_gate_tensor(
        q36_gpu_tensor *state, const q36_gpu_tensor *gate,
        const void *map, uint64_t size, uint64_t weight_offset,
        uint32_t width, uint32_t rows, float eps) {
    const char *env = getenv("Q36_METAL_RECURRENT_NORM_GATE");
    if (env && env[0] == '0') return 0;
    const uint64_t bytes = (uint64_t)width * rows * sizeof(float);
    if (!state || !gate || !map || !width || !rows ||
        bytes > state->bytes || bytes > gate->bytes ||
        weight_offset > size ||
        (uint64_t)width * sizeof(float) > size - weight_offset)
        return 0;
    uint64_t inner = 0;
    id<MTLBuffer> weight = q36_model_view(
        map, size, weight_offset, (uint64_t)width * sizeof(float), &inner);
    if (!weight) return 0;
    struct { uint32_t width, rows; float eps; } args =
        { width, rows, eps };
    id<MTLComputeCommandEncoder> enc =
        q36_encoder(@"q36_recurrent_norm_gate");
    if (!enc) return 0;
    [enc setBuffer:state->buffer offset:state->offset atIndex:0];
    [enc setBuffer:gate->buffer offset:gate->offset atIndex:1];
    [enc setBuffer:weight offset:(NSUInteger)inner atIndex:2];
    [enc setBytes:&args length:sizeof(args) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];
    return 1;
}

int q36_gpu_l2_norm_rows_tensor(q36_gpu_tensor *x, uint32_t width,
                                 uint32_t rows, float eps) {
    if (!x || (uint64_t)width * rows * 4 > x->bytes) return 0;
    id<MTLComputeCommandEncoder> enc = q36_encoder(@"q36_l2_norm_f32");
    if (!enc) return 0;
    [enc setBuffer:x->buffer offset:x->offset atIndex:0];
    [enc setBytes:&width length:sizeof(width) atIndex:1];
    [enc setBytes:&eps length:sizeof(eps) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];
    return 1;
}

int q36_gpu_matmul_f16_tensor(q36_gpu_tensor *out, const void *map,
                              uint64_t size, uint64_t offset,
                              uint64_t in_dim, uint64_t out_dim,
                              const q36_gpu_tensor *x, uint64_t tokens) {
    if (!in_dim || !out_dim ||
        in_dim > UINT64_MAX / out_dim / sizeof(uint16_t)) return 0;
    return q36_dense(@"q36_matmul_f16", out, map, size, offset,
                     in_dim * out_dim * sizeof(uint16_t),
                     in_dim, out_dim, x, tokens, 1.0f);
}

int q36_gpu_matmul_f32_scaled_tensor(q36_gpu_tensor *out, const void *map,
                                     uint64_t size, uint64_t offset,
                                     uint64_t in_dim, uint64_t out_dim,
                                     const q36_gpu_tensor *x, uint64_t tokens,
                                     float scale) {
    if (!in_dim || !out_dim ||
        in_dim > UINT64_MAX / out_dim / sizeof(float)) return 0;
    if (tokens == 1u && scale == 1.0f &&
        q36_f32_decode(out, map, size, offset, in_dim, out_dim, x))
        return 1;
    if (tokens > 1u &&
        q36_f32_prefill_mm(out, map, size, offset, in_dim, out_dim,
                           x, tokens, scale))
        return 1;
    return q36_dense(@"q36_matmul_f32", out, map, size, offset,
                     in_dim * out_dim * sizeof(float),
                     in_dim, out_dim, x, tokens, scale);
}

int q36_gpu_matmul_f32_tensor(q36_gpu_tensor *out, const void *map,
                              uint64_t size, uint64_t offset,
                              uint64_t in_dim, uint64_t out_dim,
                              const q36_gpu_tensor *x, uint64_t tokens) {
    return q36_gpu_matmul_f32_scaled_tensor(out, map, size, offset,
                                            in_dim, out_dim, x, tokens, 1.0f);
}

int q36_gpu_matmul_q8_0_scaled_tensor(q36_gpu_tensor *out, const void *map,
                                      uint64_t size, uint64_t offset,
                                      uint64_t in_dim, uint64_t out_dim,
                                      const q36_gpu_tensor *x, uint64_t tokens,
                                      float scale) {
    if (!in_dim || !out_dim || (in_dim & 31u) ||
        in_dim / 32u > UINT64_MAX / out_dim / 34u) return 0;
    if (tokens == 1 &&
        q36_q8_decode(out, map, size, offset, in_dim, out_dim, x, scale))
        return 1;
    if (tokens > 1 &&
        q36_q8_prefill_mm(out, map, size, offset, in_dim, out_dim,
                          x, tokens, scale))
        return 1;
    return q36_dense(@"q36_matmul_q8_0", out, map, size, offset,
                     (in_dim / 32u) * out_dim * 34u,
                     in_dim, out_dim, x, tokens, scale);
}

typedef struct {
    uint32_t src_stride, dst_stride, src_offset, width, rows;
} q36_copy_rows_args;

static int q36_copy_rows(q36_gpu_tensor *dst, const q36_gpu_tensor *src,
                          q36_copy_rows_args args) {
    if (!dst || !src || !args.width || !args.rows) return 0;
    uint64_t src_end = (uint64_t)(args.rows - 1u) * args.src_stride +
                       args.src_offset + args.width;
    uint64_t dst_end = (uint64_t)(args.rows - 1u) * args.dst_stride +
                       args.width;
    if (src_end > src->bytes / sizeof(float) ||
        dst_end > dst->bytes / sizeof(float)) return 0;
    id<MTLComputeCommandEncoder> enc = q36_encoder(@"q36_copy_rows_f32");
    if (!enc) return 0;
    [enc setBuffer:dst->buffer offset:dst->offset atIndex:0];
    [enc setBuffer:src->buffer offset:src->offset atIndex:1];
    [enc setBytes:&args length:sizeof(args) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(args.width, args.rows, 1)
   threadsPerThreadgroup:MTLSizeMake(MIN(args.width, 256u), 1, 1)];
    [enc endEncoding];
    return 1;
}

int q36_gpu_extract_full_attn_q_tensor(q36_gpu_tensor *dst,
                                        const q36_gpu_tensor *qg,
                                        uint32_t tokens) {
    return q36_copy_rows(dst, qg, (q36_copy_rows_args){
        512u, 256u, 0u, 256u, tokens * 16u
    });
}

int q36_gpu_extract_recurrent_v_tensor(q36_gpu_tensor *dst,
                                        const q36_gpu_tensor *conv,
                                        uint32_t tokens) {
    return q36_copy_rows(dst, conv, (q36_copy_rows_args){
        8192u, 4096u, 4096u, 4096u, tokens
    });
}

int q36_gpu_recurrent_conv_step_tensor(q36_gpu_tensor *cache,
                                        const q36_gpu_tensor *cur,
                                        q36_gpu_tensor *window,
                                        uint32_t tokens) {
    const uint32_t conv_dim = 8192u, hist = 3u;
    if (!cache || !cur || !window || !tokens ||
        cache->bytes < (uint64_t)hist * conv_dim * 4u ||
        cur->bytes < (uint64_t)tokens * conv_dim * 4u ||
        window->bytes < (uint64_t)tokens * (hist + 1u) * conv_dim * 4u)
        return 0;
    struct { uint32_t conv_dim, hist, tokens; } args =
        { conv_dim, hist, tokens };
    id<MTLComputeCommandEncoder> enc = q36_encoder(@"q36_recurrent_window");
    if (!enc) return 0;
    [enc setBuffer:cache->buffer offset:cache->offset atIndex:0];
    [enc setBuffer:cur->buffer offset:cur->offset atIndex:1];
    [enc setBuffer:window->buffer offset:window->offset atIndex:2];
    [enc setBytes:&args length:sizeof(args) atIndex:3];
    q36_dispatch_1d(enc, conv_dim);
    return 1;
}

static int q36_conv_dispatch(NSString *kernel, q36_gpu_tensor *out,
                              q36_gpu_tensor *cache,
                              const q36_gpu_tensor *input,
                              const void *map, uint64_t size, uint64_t offset,
                              uint32_t dim, uint32_t taps, uint32_t tokens) {
    if (!out || !input || !map || !dim || !taps || !tokens ||
        dim > UINT64_MAX / taps / sizeof(float)) return 0;
    uint64_t inner = 0;
    id<MTLBuffer> weights = q36_model_view(map, size, offset,
        (uint64_t)dim * taps * sizeof(float), &inner);
    if (!weights) return 0;
    struct { uint32_t dim, taps, tokens; } args = { dim, taps, tokens };
    id<MTLComputeCommandEncoder> enc = q36_encoder(kernel);
    if (!enc) return 0;
    [enc setBuffer:out->buffer offset:out->offset atIndex:0];
    NSUInteger index = 1;
    if (cache) [enc setBuffer:cache->buffer offset:cache->offset atIndex:index++];
    [enc setBuffer:input->buffer offset:input->offset atIndex:index++];
    [enc setBuffer:weights offset:(NSUInteger)inner atIndex:index++];
    [enc setBytes:&args length:sizeof(args) atIndex:index];
    if (cache) q36_dispatch_1d(enc, dim);
    else {
        [enc dispatchThreads:MTLSizeMake(dim, tokens, 1)
       threadsPerThreadgroup:MTLSizeMake(MIN(dim, 256u), 1, 1)];
        [enc endEncoding];
    }
    return 1;
}

int q36_gpu_ssm_conv_silu_tensor(q36_gpu_tensor *out,
                                  const q36_gpu_tensor *window,
                                  const void *map, uint64_t size,
                                  uint64_t offset, uint32_t dim,
                                  uint32_t taps, uint32_t tokens) {
    return q36_conv_dispatch(@"q36_conv_silu", out, NULL, window,
                             map, size, offset, dim, taps, tokens);
}

int q36_gpu_recurrent_conv_silu_tensor(q36_gpu_tensor *cache,
                                        const q36_gpu_tensor *cur,
                                        q36_gpu_tensor *out,
                                        const void *map, uint64_t size,
                                        uint64_t offset, uint32_t dim,
                                        uint32_t taps, uint32_t tokens) {
    return q36_conv_dispatch(@"q36_recurrent_conv_silu", out, cache, cur,
                             map, size, offset, dim, taps, tokens);
}

int q36_gpu_rope_qwen_rows_tensor(q36_gpu_tensor *x, uint32_t heads,
                                   uint32_t pos0, uint32_t tokens) {
    if (!x || !heads || !tokens ||
        x->bytes < (uint64_t)heads * tokens * 256u * sizeof(float)) return 0;
    struct { uint32_t heads, pos0, tokens; } args = { heads, pos0, tokens };
    id<MTLComputeCommandEncoder> enc = q36_encoder(@"q36_rope_qwen");
    if (!enc) return 0;
    [enc setBuffer:x->buffer offset:x->offset atIndex:0];
    [enc setBytes:&args length:sizeof(args) atIndex:1];
    [enc dispatchThreads:MTLSizeMake(heads * 32u, tokens, 1)
   threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];
    return 1;
}

int q36_gpu_directional_steering_project_tensor(
        q36_gpu_tensor *x, const q36_gpu_tensor *directions,
        uint32_t layer, uint32_t width, uint32_t rows, float scale) {
    if (!x || !directions || !width || !rows) return 0;
    struct { uint32_t layer, width, rows; float scale; } args =
        { layer, width, rows, scale };
    id<MTLComputeCommandEncoder> enc =
        q36_encoder(@"q36_directional_steering");
    if (!enc) return 0;
    [enc setBuffer:x->buffer offset:x->offset atIndex:0];
    [enc setBuffer:directions->buffer offset:directions->offset atIndex:1];
    [enc setBytes:&args length:sizeof(args) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];
    return 1;
}

int q36_gpu_add_rms_norm_tensor(q36_gpu_tensor *out_norm,
                                 q36_gpu_tensor *out_sum,
                                 const q36_gpu_tensor *a,
                                 const q36_gpu_tensor *b,
                                 const void *map, uint64_t size,
                                 uint64_t weight_offset,
                                 uint32_t width, uint32_t rows, float eps) {
    const char *fused_norm_env = getenv("Q36_METAL_FUSED_NORM");
    bool fused_norm = (!fused_norm_env || !fused_norm_env[0] ||
                       fused_norm_env[0] != '0') &&
                      width >= 512u && (width & 3u) == 0u;
    uint64_t bytes = (uint64_t)width * rows * sizeof(float);
    if (fused_norm && out_norm && out_sum && a && b && map &&
        bytes <= out_norm->bytes && bytes <= out_sum->bytes &&
        bytes <= a->bytes && bytes <= b->bytes &&
        weight_offset <= size &&
        (uint64_t)width * sizeof(float) <= size - weight_offset) {
        const uint64_t weight_bytes =
            (uint64_t)width * sizeof(float);
        uint64_t weight_inner = 0;
        id<MTLBuffer> weight =
            q36_model_view(map, size, weight_offset, weight_bytes,
                            &weight_inner);
        struct {
            int32_t ne00, ne00_t;
            uint64_t nb1, nb2, nb3;
            float eps;
            int32_t nef1[3], nef2[3], nef3[3];
            uint64_t nbf1[3], nbf2[3], nbf3[3];
        } args = {0};
        args.ne00 = (int32_t)width;
        args.ne00_t = (int32_t)(width / 4u);
        args.nb1 = (uint64_t)width * sizeof(float);
        args.nb2 = args.nb1 * rows;
        args.nb3 = args.nb2;
        args.eps = eps;
        for (uint32_t i = 0; i < 3; i++) {
            args.nef1[i] = args.nef2[i] = args.nef3[i] = 1;
        }
        args.nef1[0] = (int32_t)rows;
        args.nbf1[0] = (uint64_t)width * sizeof(float);
        id<MTLComputeCommandEncoder> enc =
            q36_encoder(@"kernel_add_rms_norm_mul_f32_4");
        if (!weight || !enc) return 0;
        [enc setBytes:&args length:sizeof(args) atIndex:0];
        [enc setBuffer:a->buffer offset:a->offset atIndex:1];
        [enc setBuffer:b->buffer offset:b->offset atIndex:2];
        [enc setBuffer:weight offset:(NSUInteger)weight_inner atIndex:3];
        [enc setBuffer:out_sum->buffer offset:out_sum->offset atIndex:4];
        [enc setBuffer:out_norm->buffer offset:out_norm->offset atIndex:5];
        [enc setThreadgroupMemoryLength:32u * sizeof(float) atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        return 1;
    }
    return q36_gpu_add_tensor(out_sum, a, b, width * rows) &&
           q36_gpu_rms_norm_weight_rows_tensor(
               out_norm, out_sum, map, size, weight_offset, width, rows, eps);
}

static int q36_delta_qkv_dispatch(q36_gpu_tensor *q,
                                   q36_gpu_tensor *k,
                                   q36_gpu_tensor *v,
                                   const q36_gpu_tensor *conv,
                                   uint32_t heads, uint32_t groups,
                                   uint32_t dim, uint32_t stride,
                                   uint32_t tokens, float eps) {
    if (!q || !k || !conv || !heads || !groups || !dim || !tokens ||
        stride < 2u * groups * dim + (v ? heads * dim : 0u)) return 0;
    struct {
        uint32_t heads, groups, dim, stride, tokens;
        float eps;
    } args = { heads, groups, dim, stride, tokens, eps };
    uint32_t write_v = v != NULL;
    id<MTLComputeCommandEncoder> enc = q36_encoder(@"q36_delta_qkv");
    if (!enc) return 0;
    [enc setBuffer:q->buffer offset:q->offset atIndex:0];
    [enc setBuffer:k->buffer offset:k->offset atIndex:1];
    [enc setBuffer:(v ? v->buffer : q->buffer)
            offset:(v ? v->offset : q->offset) atIndex:2];
    [enc setBuffer:conv->buffer offset:conv->offset atIndex:3];
    [enc setBytes:&args length:sizeof(args) atIndex:4];
    [enc setBytes:&write_v length:sizeof(write_v) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(heads, tokens, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];
    return 1;
}

int q36_gpu_delta_qk_l2_norm_tensor(
        q36_gpu_tensor *q, q36_gpu_tensor *k,
        const q36_gpu_tensor *conv, uint32_t heads, uint32_t groups,
        uint32_t dim, uint32_t stride, uint32_t tokens, float eps) {
    return q36_delta_qkv_dispatch(q, k, NULL, conv, heads, groups, dim,
                                  stride, tokens, eps);
}

int q36_gpu_delta_qkv_l2_norm_tensor(
        q36_gpu_tensor *q, q36_gpu_tensor *k, q36_gpu_tensor *v,
        const q36_gpu_tensor *conv, uint32_t heads, uint32_t groups,
        uint32_t dim, uint32_t stride, uint32_t tokens, float eps) {
    return q36_delta_qkv_dispatch(q, k, v, conv, heads, groups, dim,
                                  stride, tokens, eps);
}

int q36_gpu_delta_net_gates_tensor(
        q36_gpu_tensor *out, const q36_gpu_tensor *alpha,
        const q36_gpu_tensor *beta, const void *map, uint64_t size,
        uint64_t dt_offset, uint64_t a_offset,
        uint32_t heads, uint32_t tokens) {
    uint64_t bytes = (uint64_t)heads * sizeof(float);
    uint64_t dt_inner = 0, a_inner = 0;
    id<MTLBuffer> dt = q36_model_view(map, size, dt_offset, bytes, &dt_inner);
    id<MTLBuffer> a = q36_model_view(map, size, a_offset, bytes, &a_inner);
    if (!out || !alpha || !beta || !dt || !a || !heads || !tokens) return 0;
    struct { uint32_t heads, tokens; } args = { heads, tokens };
    id<MTLComputeCommandEncoder> enc = q36_encoder(@"q36_delta_gates");
    if (!enc) return 0;
    [enc setBuffer:out->buffer offset:out->offset atIndex:0];
    [enc setBuffer:alpha->buffer offset:alpha->offset atIndex:1];
    [enc setBuffer:beta->buffer offset:beta->offset atIndex:2];
    [enc setBuffer:dt offset:(NSUInteger)dt_inner atIndex:3];
    [enc setBuffer:a offset:(NSUInteger)a_inner atIndex:4];
    [enc setBytes:&args length:sizeof(args) atIndex:5];
    [enc dispatchThreads:MTLSizeMake(heads, tokens, 1)
   threadsPerThreadgroup:MTLSizeMake(MIN(heads, 64u), 1, 1)];
    [enc endEncoding];
    return 1;
}

int q36_gpu_delta_net_decode_tensor(
        q36_gpu_tensor *state, const q36_gpu_tensor *q,
        const q36_gpu_tensor *k, const q36_gpu_tensor *v,
        const q36_gpu_tensor *gb, q36_gpu_tensor *out,
        uint32_t heads, uint32_t dim, uint32_t tokens) {
    if (!state || !q || !k || !v || !gb || !out ||
        !heads || !dim || !tokens || (dim & 31u)) return 0;
    struct { uint32_t heads, dim, tokens; } args = { heads, dim, tokens };
    const char *r4_env = getenv("Q36_METAL_DELTA_R4");
    bool use_r4 = dim == 128u && (!r4_env || strcmp(r4_env, "0") != 0);
    id<MTLComputeCommandEncoder> enc = q36_encoder(
        use_r4 ? @"q36_delta_net_decode_r4" : @"q36_delta_net_decode");
    if (!enc) return 0;
    [enc setBuffer:state->buffer offset:state->offset atIndex:0];
    [enc setBuffer:q->buffer offset:q->offset atIndex:1];
    [enc setBuffer:k->buffer offset:k->offset atIndex:2];
    [enc setBuffer:v->buffer offset:v->offset atIndex:3];
    [enc setBuffer:gb->buffer offset:gb->offset atIndex:4];
    [enc setBuffer:out->buffer offset:out->offset atIndex:5];
    [enc setBytes:&args length:sizeof(args) atIndex:6];
    if (use_r4) {
        [enc dispatchThreadgroups:MTLSizeMake((dim + 3u) / 4u, heads, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    } else {
        [enc dispatchThreadgroups:MTLSizeMake(heads, dim / 32u, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
    [enc endEncoding];
    return 1;
}

int q36_gpu_attn_kv_store_tensor(
        q36_gpu_tensor *kcache, q36_gpu_tensor *vcache,
        const q36_gpu_tensor *k, const q36_gpu_tensor *v,
        uint32_t pos0, uint32_t tokens, uint32_t cap,
        uint32_t k_row, uint32_t v_row,
        uint32_t k_type, uint32_t v_type,
        uint32_t k_row_bytes, uint32_t v_row_bytes) {
    if (!kcache || !vcache || !k || !v || !tokens ||
        pos0 > cap || tokens > cap - pos0) return 0;
    if (k_type != 0u || v_type != 0u) {
        if ((k_type != 1u && k_type != 2u) ||
            (v_type != 1u && v_type != 2u) ||
            (k_row & 31u) || (v_row & 31u) ||
            k_row_bytes != (k_row / 32u) * (k_type == 1u ? 34u : 18u) ||
            v_row_bytes != (v_row / 32u) * (v_type == 1u ? 34u : 18u))
            return 0;
        if (k_type != 1u || v_type != 2u) {
            const float *kp = q36_gpu_tensor_contents((q36_gpu_tensor *)k);
            const float *vp = q36_gpu_tensor_contents((q36_gpu_tensor *)v);
            uint8_t *kc = q36_gpu_tensor_contents(kcache);
            uint8_t *vc = q36_gpu_tensor_contents(vcache);
            if (!kp || !vp || !kc || !vc) return 0;
            for (uint32_t tok = 0; tok < tokens; tok++) {
                void *kd = kc + (uint64_t)(pos0 + tok) * k_row_bytes;
                void *vd = vc + (uint64_t)(pos0 + tok) * v_row_bytes;
                if (k_type == 1u) q36_quant_q8_0(kp + (uint64_t)tok * k_row, kd, k_row);
                else q36_quant_q4_0_kv(kp + (uint64_t)tok * k_row, kd, k_row);
                if (v_type == 1u) q36_quant_q8_0(vp + (uint64_t)tok * v_row, vd, v_row);
                else q36_quant_q4_0_kv(vp + (uint64_t)tok * v_row, vd, v_row);
            }
            return 1;
        }
        struct {
            uint32_t k_row, v_row, pos0, tokens;
            uint32_t k_row_bytes, v_row_bytes;
        } args = { k_row, v_row, pos0, tokens, k_row_bytes, v_row_bytes };
        id<MTLComputeCommandEncoder> enc =
            q36_encoder(@"q36_kv_store_q8_q4");
        if (!enc) return 0;
        [enc setBuffer:kcache->buffer offset:kcache->offset atIndex:0];
        [enc setBuffer:vcache->buffer offset:vcache->offset atIndex:1];
        [enc setBuffer:k->buffer offset:k->offset atIndex:2];
        [enc setBuffer:v->buffer offset:v->offset atIndex:3];
        [enc setBytes:&args length:sizeof(args) atIndex:4];
        uint32_t blocks = MAX(k_row, v_row) / 32u;
        [enc dispatchThreads:MTLSizeMake(blocks, tokens, 1)
           threadsPerThreadgroup:MTLSizeMake(MIN(blocks, 256u), 1, 1)];
        [enc endEncoding];
        return 1;
    }
    if (k_row_bytes != k_row * sizeof(uint16_t) ||
        v_row_bytes != v_row * sizeof(uint16_t)) return 0;
    struct { uint32_t k_row, v_row, pos0, tokens; } args =
        { k_row, v_row, pos0, tokens };
    id<MTLComputeCommandEncoder> enc = q36_encoder(@"q36_kv_store_f16");
    if (!enc) return 0;
    [enc setBuffer:kcache->buffer offset:kcache->offset atIndex:0];
    [enc setBuffer:vcache->buffer offset:vcache->offset atIndex:1];
    [enc setBuffer:k->buffer offset:k->offset atIndex:2];
    [enc setBuffer:v->buffer offset:v->offset atIndex:3];
    [enc setBytes:&args length:sizeof(args) atIndex:4];
    uint32_t width = k_row > v_row ? k_row : v_row;
    [enc dispatchThreads:MTLSizeMake(width, tokens, 1)
   threadsPerThreadgroup:MTLSizeMake(MIN(width, 256u), 1, 1)];
    [enc endEncoding];
    return 1;
}

int q36_gpu_router_topk_tensor(
        q36_gpu_tensor *selected, q36_gpu_tensor *weights,
        const q36_gpu_tensor *logits, uint32_t experts,
        uint32_t used, uint32_t tokens, float scale) {
    if (!selected || !weights || !logits || !experts || !used ||
        used > 8u || !tokens) return 0;
    struct { uint32_t experts, used, tokens; float scale; } args =
        { experts, used, tokens, scale };
    const char *parallel_env = getenv("Q36_METAL_ROUTER_PARALLEL");
    const bool parallel = !parallel_env || parallel_env[0] != '0';
    id<MTLComputeCommandEncoder> enc =
        q36_encoder(parallel ? @"q36_router_topk"
                             : @"q36_router_topk_serial");
    if (!enc) return 0;
    [enc setBuffer:selected->buffer offset:selected->offset atIndex:0];
    [enc setBuffer:weights->buffer offset:weights->offset atIndex:1];
    [enc setBuffer:logits->buffer offset:logits->offset atIndex:2];
    [enc setBytes:&args length:sizeof(args) atIndex:3];
    if (parallel) {
        [enc dispatchThreadgroups:MTLSizeMake(tokens, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    } else {
        q36_dispatch_1d(enc, tokens);
    }
    return 1;
}

int q36_gpu_top2_tensor(q36_gpu_tensor *out_ids,
                        const q36_gpu_tensor *logits,
                        uint32_t count) {
    const char *env = getenv("Q36_METAL_GPU_ARGMAX");
    const uint32_t threads = 256u;
    if ((env && env[0] == '0') || !out_ids || !logits || count < 2u ||
        out_ids->bytes < 2u * sizeof(int32_t) ||
        logits->bytes < (uint64_t)count * sizeof(float)) {
        return 0;
    }
    id<MTLComputeCommandEncoder> enc = q36_encoder(@"q36_top2_f32");
    if (!enc) return 0;
    [enc setBuffer:out_ids->buffer offset:out_ids->offset atIndex:0];
    [enc setBuffer:logits->buffer offset:logits->offset atIndex:1];
    [enc setBytes:&count length:sizeof(count) atIndex:2];
    [enc setThreadgroupMemoryLength:threads * 2u * sizeof(float) atIndex:0];
    [enc setThreadgroupMemoryLength:threads * 2u * sizeof(int32_t) atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [enc endEncoding];
    return 1;
}

typedef struct {
    float d;
    float dmin;
    int8_t qs[256];
    int16_t bsums[16];
} q36_metal_q8_k;

typedef struct {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
} q36_metal_q8_k_packed;

typedef struct {
    uint8_t scales[16], qs[64];
    uint16_t d, dmin;
} q36_metal_q2_k;
typedef struct {
    uint16_t d, dmin;
    uint8_t scales[12], qs[128];
} q36_metal_q4_k;
typedef struct {
    uint16_t d, dmin;
    uint8_t scales[12], qh[32], qs[128];
} q36_metal_q5_k;
typedef struct {
    uint8_t ql[128], qh[64];
    int8_t scales[16];
    uint16_t d;
} q36_metal_q6_k;

static void q36_k4_scale_min(const uint8_t *s, uint32_t j,
                              uint32_t *scale, uint32_t *mn) {
    if (j < 4u) {
        *scale = s[j] & 63u;
        *mn = s[4u + j] & 63u;
    } else {
        *scale = (s[j + 4u] & 15u) | ((s[j - 4u] >> 6) << 4);
        *mn = (s[j + 4u] >> 4) | ((s[j] >> 6) << 4);
    }
}

static int q36_q2_k_value(const q36_metal_q2_k *x, uint32_t idx) {
    uint32_t in = idx & 127u;
    uint32_t qoff = (idx >> 7u) * 32u + (in & 31u);
    return (x->qs[qoff] >> ((in >> 5u) * 2u)) & 3u;
}

static int q36_q6_k_value(const q36_metal_q6_k *x, uint32_t idx) {
    uint32_t half = idx >> 7u, in = idx & 127u;
    uint32_t ql, qh, shift, low;
    if (in < 32u) {
        ql = half * 64u + in; qh = half * 32u + in; shift = 0u;
        low = x->ql[ql] & 15u;
    } else if (in < 64u) {
        uint32_t i = in - 32u;
        ql = half * 64u + 32u + i; qh = half * 32u + i; shift = 2u;
        low = x->ql[ql] & 15u;
    } else if (in < 96u) {
        uint32_t i = in - 64u;
        ql = half * 64u + i; qh = half * 32u + i; shift = 4u;
        low = x->ql[ql] >> 4;
    } else {
        uint32_t i = in - 96u;
        ql = half * 64u + 32u + i; qh = half * 32u + i; shift = 6u;
        low = x->ql[ql] >> 4;
    }
    return (int)(low | (((x->qh[qh] >> shift) & 3u) << 4)) - 32;
}

static float q36_kquant_dot_block(uint32_t type, const void *vx,
                                   const q36_metal_q8_k *y) {
    if (type == 10u) {
        const q36_metal_q2_k *x = vx;
        float d = q36_quant_f16_to_f32(x->d) * y->d;
        float dm = -q36_quant_f16_to_f32(x->dmin) * y->d;
        float lane[8] = {0};
        for (uint32_t k = 0; k < 8u; k++) {
            int mp = (x->scales[2u*k] >> 4) * y->bsums[2u*k]
                   + (x->scales[2u*k+1u] >> 4) * y->bsums[2u*k+1u];
            int sum = 0;
            lane[k] = fmaf(dm, (float)mp, lane[k]);
            for (uint32_t h = 0; h < 2u; h++) for (uint32_t s = 0; s < 4u; s++) {
                uint32_t g = h * 8u + 2u*s + (k >= 4u);
                uint32_t e = h * 128u + s * 32u +
                    (k < 4u ? 4u*k : 16u + 4u*(k-4u));
                int v = 0;
                for (uint32_t i = 0; i < 4u; i++)
                    v += q36_q2_k_value(x, e+i) * y->qs[e+i];
                sum += (x->scales[g] & 15u) * v;
            }
            lane[k] = fmaf(d, (float)sum, lane[k]);
        }
        return ((lane[0]+lane[4])+(lane[2]+lane[6])) +
               ((lane[1]+lane[5])+(lane[3]+lane[7]));
    }
    if (type == 12u || type == 13u) {
        const q36_metal_q4_k *x4 = vx;
        const q36_metal_q5_k *x5 = vx;
        const uint8_t *scales = type == 12u ? x4->scales : x5->scales;
        const uint8_t *qs = type == 12u ? x4->qs : x5->qs;
        uint16_t hd = type == 12u ? x4->d : x5->d;
        uint16_t hm = type == 12u ? x4->dmin : x5->dmin;
        int isum = 0, summs = 0;
        for (uint32_t g = 0; g < 8u; g++) {
            uint32_t sc, mn; int group = 0;
            q36_k4_scale_min(scales, g, &sc, &mn);
            const uint8_t *q = qs + (g >> 1u) * 32u;
            for (uint32_t i = 0; i < 32u; i++) {
                uint32_t v = (g & 1u) ? q[i] >> 4 : q[i] & 15u;
                if (type == 13u) v |= ((x5->qh[i] >> g) & 1u) << 4;
                group += (int)v * y->qs[g*32u+i];
            }
            isum += (int)sc * group;
            summs += (int)mn * (y->bsums[2u*g] + y->bsums[2u*g+1u]);
        }
        return q36_quant_f16_to_f32(hd) * y->d * (float)isum -
               q36_quant_f16_to_f32(hm) * y->d * (float)summs;
    }
    if (type == 14u) {
        const q36_metal_q6_k *x = vx;
        int isum = 0;
        for (uint32_t g = 0; g < 16u; g++) {
            int group = 0;
            for (uint32_t i = 0; i < 16u; i++) {
                uint32_t j = g*16u+i;
                group += q36_q6_k_value(x, j) * y->qs[j];
            }
            isum += x->scales[g] * group;
        }
        return q36_quant_f16_to_f32(x->d) * y->d * (float)isum;
    }
    return 0.0f;
}

static uint64_t q36_kquant_row_bytes(uint32_t type, uint64_t in_dim) {
    uint64_t block_bytes = 0;
    switch (type) {
    case 10: block_bytes = 84u; break;
    case 12: block_bytes = 144u; break;
    case 13: block_bytes = 176u; break;
    case 14: block_bytes = 210u; break;
    default: return 0;
    }
    return (in_dim / 256u) * block_bytes;
}

int q36_gpu_quantize_q8_k_tensor(q36_gpu_tensor *out,
                                  const q36_gpu_tensor *x,
                                  uint64_t in_dim, uint64_t tokens) {
    /* Dense Metal consumes the original float activation in native IQ3/K
     * kernels.  Its callers retain the shared Vulkan q8_K staging calls, so
     * acknowledge those without introducing a command-buffer wait. */
    if (q36_dense_model) return out && x && in_dim && tokens;
    if (!q36_quant_q8_k || !out || !x || !in_dim || (in_dim & 255u) ||
        !tokens || out->bytes < tokens * (in_dim / 256u) * sizeof(q36_metal_q8_k) ||
        x->bytes < tokens * in_dim * sizeof(float)) return 0;
    const float *src = q36_gpu_tensor_contents((q36_gpu_tensor *)x);
    unsigned char *dst = q36_gpu_tensor_contents(out);
    if (!src || !dst) return 0;
    uint64_t blocks = in_dim / 256u;
    q36_metal_q8_k_packed *tmp =
        malloc((size_t)blocks * sizeof(q36_metal_q8_k_packed));
    if (!tmp) return 0;
    for (uint64_t tok = 0; tok < tokens; tok++) {
        q36_quant_q8_k(src + tok * in_dim, tmp, (int64_t)in_dim);
        q36_metal_q8_k *row = (q36_metal_q8_k *)dst + tok * blocks;
        for (uint64_t b = 0; b < blocks; b++) {
            row[b].d = tmp[b].d;
            row[b].dmin = 0.0f;
            memcpy(row[b].qs, tmp[b].qs, sizeof(row[b].qs));
            memcpy(row[b].bsums, tmp[b].bsums, sizeof(row[b].bsums));
        }
    }
    free(tmp);
    return 1;
}

static int q36_kquant_q8_host(q36_gpu_tensor *out, const void *map,
                              uint64_t size, uint64_t offset, uint32_t type,
                              uint64_t in_dim, uint64_t out_dim,
                              const q36_metal_q8_k *xq, uint64_t tokens,
                              float scale) {
    uint64_t row_bytes = q36_kquant_row_bytes(type, in_dim);
    if (!q36_quant_f16_to_f32 || !out || !map || !row_bytes || !out_dim ||
        offset > size || row_bytes > (size - offset) / out_dim) return 0;
    float *dst = q36_gpu_tensor_contents(out);
    if (!dst || !xq) return 0;
    const unsigned char *weights = (const unsigned char *)map + offset;
    uint64_t blocks = in_dim / 256u;
    uint64_t block_bytes = row_bytes / blocks;
    for (uint64_t r = 0; r < out_dim; r++) {
        for (uint64_t tok = 0; tok < tokens; tok++) {
            float sum = 0.0f;
            for (uint64_t b = 0; b < blocks; b++)
                sum += q36_kquant_dot_block(
                    type, weights + r * row_bytes + b * block_bytes,
                    xq + tok * blocks + b);
            dst[tok * out_dim + r] = sum * scale;
        }
    }
    return 1;
}

int q36_gpu_matmul_k_quant_scaled_tensor(
        q36_gpu_tensor *out, const void *map, uint64_t size, uint64_t offset,
        uint32_t type, uint64_t in_dim, uint64_t out_dim,
        const q36_gpu_tensor *x, uint64_t tokens, float scale) {
    if (type == 12u || type == 14u) {
        return q36_quant_float_matmul(
            type == 12u ? @"kernel_mul_mv_ext_q4_K_f32_r1_"
                        : @"kernel_mul_mv_ext_q6_K_f32_r1_",
            type == 12u ? @"kernel_mul_mm_q4_K_f32"
                        : @"kernel_mul_mm_q6_K_f32",
            type == 12u ? 144u : 210u,
            out, map, size, offset, in_dim, out_dim, x, tokens, scale);
    }
    if (!x || !in_dim || (in_dim & 255u) ||
        x->bytes < tokens * in_dim * sizeof(float)) return 0;
    const float *src = q36_gpu_tensor_contents((q36_gpu_tensor *)x);
    uint64_t blocks = in_dim / 256u;
    q36_metal_q8_k *q8 = calloc((size_t)(tokens * blocks), sizeof(*q8));
    q36_metal_q8_k_packed *tmp =
        malloc((size_t)blocks * sizeof(q36_metal_q8_k_packed));
    if (!src || !q8 || !tmp || !q36_quant_q8_k) {
        free(tmp); free(q8); return 0;
    }
    for (uint64_t tok = 0; tok < tokens; tok++) {
        q36_quant_q8_k(src + tok * in_dim, tmp, (int64_t)in_dim);
        for (uint64_t b = 0; b < blocks; b++) {
            q8[tok*blocks+b].d = tmp[b].d;
            memcpy(q8[tok*blocks+b].qs, tmp[b].qs, 256u);
            memcpy(q8[tok*blocks+b].bsums, tmp[b].bsums, 32u);
        }
    }
    free(tmp);
    int ok = q36_kquant_q8_host(out, map, size, offset, type,
                                in_dim, out_dim, q8, tokens, scale);
    free(q8);
    return ok;
}

int q36_gpu_matmul_k_quant_tensor(
        q36_gpu_tensor *out, const void *map, uint64_t size, uint64_t offset,
        uint32_t type, uint64_t in_dim, uint64_t out_dim,
        const q36_gpu_tensor *x, uint64_t tokens) {
    return q36_gpu_matmul_k_quant_scaled_tensor(
        out, map, size, offset, type, in_dim, out_dim, x, tokens, 1.0f);
}

int q36_gpu_matmul_k_quant_q8_scaled_tensor(
        q36_gpu_tensor *out, const void *map, uint64_t size, uint64_t offset,
        uint32_t type, uint64_t in_dim, uint64_t out_dim,
        const q36_gpu_tensor *q8, uint64_t tokens, float scale) {
    if (!q8 || !in_dim || (in_dim & 255u) ||
        q8->bytes < tokens * (in_dim / 256u) * sizeof(q36_metal_q8_k))
        return 0;
    const q36_metal_q8_k *blocks =
        q36_gpu_tensor_contents((q36_gpu_tensor *)q8);
    return blocks && q36_kquant_q8_host(out, map, size, offset, type,
                                        in_dim, out_dim, blocks, tokens, scale);
}

int q36_gpu_matmul_iq_quant_scaled_tensor(
        q36_gpu_tensor *out, const void *map, uint64_t size, uint64_t offset,
        uint32_t type, uint64_t in_dim, uint64_t out_dim,
        const q36_gpu_tensor *x, uint64_t tokens, float scale) {
    if (type != 18u && type != 21u) return 0;
    return q36_quant_float_matmul(
        type == 18u ? @"kernel_mul_mv_ext_iq3_xxs_f32_r1_"
                    : @"kernel_mul_mv_ext_iq3_s_f32_r1_",
        type == 18u ? @"kernel_mul_mm_iq3_xxs_f32"
                    : @"kernel_mul_mm_iq3_s_f32",
        type == 18u ? 98u : 110u,
        out, map, size, offset, in_dim, out_dim, x, tokens, scale);
}

int q36_gpu_matmul_iq_quant_q8_scaled_tensor(
        q36_gpu_tensor *out, const void *map, uint64_t size, uint64_t offset,
        uint32_t type, uint64_t in_dim, uint64_t out_dim,
        const q36_gpu_tensor *q8, uint64_t tokens, float scale) {
    if (!q8 || (type != 18u && type != 21u) || !in_dim ||
        (in_dim & 255u) || !tokens) return 0;
    const uint64_t blocks = in_dim / 256u;
    if (q8->bytes < tokens * blocks * sizeof(q36_metal_q8_k)) return 0;
    q36_gpu_tensor *x = q36_gpu_tensor_alloc(tokens * in_dim * sizeof(float));
    if (!x) return 0;
    const q36_metal_q8_k *src =
        q36_gpu_tensor_contents((q36_gpu_tensor *)q8);
    float *dst = q36_gpu_tensor_contents(x);
    if (!src || !dst) {
        q36_gpu_tensor_free(x);
        return 0;
    }
    for (uint64_t t = 0; t < tokens; t++) {
        for (uint64_t b = 0; b < blocks; b++) {
            const q36_metal_q8_k *qb = src + t*blocks + b;
            for (uint32_t i = 0; i < 256u; i++)
                dst[t*in_dim + b*256u + i] = qb->d * qb->qs[i];
        }
    }
    int ok = q36_gpu_matmul_iq_quant_scaled_tensor(
        out, map, size, offset, type, in_dim, out_dim, x, tokens, scale);
    q36_gpu_tensor_free(x);
    return ok;
}

int q36_gpu_matmul_q8_0_pair_scaled_tensor(
        q36_gpu_tensor *out_a, q36_gpu_tensor *out_b,
        const void *map, uint64_t size,
        uint64_t offset_a, uint64_t offset_b,
        uint64_t in_dim, uint64_t out_a_dim, uint64_t out_b_dim,
        const q36_gpu_tensor *x, float scale_a, float scale_b) {
    if (!out_a || !out_b || !x || !map || !in_dim || (in_dim & 31u) ||
        !out_a_dim || !out_b_dim || in_dim > INT32_MAX ||
        out_a_dim > INT32_MAX || out_b_dim > INT32_MAX ||
        x->bytes < in_dim * sizeof(float) ||
        out_a->bytes < out_a_dim * sizeof(float) ||
        out_b->bytes < out_b_dim * sizeof(float)) return 0;
    const uint64_t row_bytes = (in_dim / 32u) * 34u;
    if (out_a_dim > UINT64_MAX / row_bytes ||
        out_b_dim > UINT64_MAX / row_bytes) return 0;
    uint64_t inner_a = 0, inner_b = 0;
    id<MTLBuffer> wa = q36_model_view(
        map, size, offset_a, out_a_dim * row_bytes, &inner_a);
    id<MTLBuffer> wb = q36_model_view(
        map, size, offset_b, out_b_dim * row_bytes, &inner_b);
    int16_t nsg_a = out_a_dim > 65536u ? 8 : 4;
    int16_t nsg_b = out_b_dim > 65536u ? 8 : 4;
    if (!wa || !wb || nsg_a != nsg_b) return 0;
    id<MTLComputePipelineState> p =
        q36_pipeline_nsg(@"kernel_mul_mv_q8_0_f32_pair", nsg_a);
    if (!p || (!q36_batch && !q36_gpu_begin_commands())) return 0;
#define Q36_Q8_PAIR_ARGS(dim_) (q36_q8_mv_args) { \
        (int32_t)in_dim, (int32_t)(dim_), 1, 34u, row_bytes, \
        row_bytes * (dim_), row_bytes * (dim_), (int32_t)in_dim, 1, 1, \
        sizeof(float), in_dim * sizeof(float), in_dim * sizeof(float), \
        in_dim * sizeof(float), (int32_t)(dim_), 1, 2, 1, 1 }
    q36_q8_mv_args args_a = Q36_Q8_PAIR_ARGS(out_a_dim);
    q36_q8_mv_args args_b = Q36_Q8_PAIR_ARGS(out_b_dim);
#undef Q36_Q8_PAIR_ARGS
    id<MTLComputeCommandEncoder> enc = [q36_batch computeCommandEncoder];
    enc.label = @"kernel_mul_mv_q8_0_f32_pair";
    [enc setComputePipelineState:p];
    [enc setBytes:&args_a length:sizeof(args_a) atIndex:0];
    [enc setBytes:&args_b length:sizeof(args_b) atIndex:1];
    [enc setBuffer:wa offset:(NSUInteger)inner_a atIndex:2];
    [enc setBuffer:wb offset:(NSUInteger)inner_b atIndex:3];
    [enc setBuffer:x->buffer offset:x->offset atIndex:4];
    [enc setBuffer:out_a->buffer offset:out_a->offset atIndex:5];
    [enc setBuffer:out_b->buffer offset:out_b->offset atIndex:6];
    [enc setThreadgroupMemoryLength:32u * 4u * sizeof(float) atIndex:0];
    uint64_t max_dim = out_a_dim > out_b_dim ? out_a_dim : out_b_dim;
    [enc dispatchThreadgroups:MTLSizeMake((max_dim + 1u) / 2u, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(32, (NSUInteger)nsg_a, 1)];
    [enc endEncoding];
    q36_gpu_tensor *outs[2] = {out_a, out_b};
    float scales[2] = {scale_a, scale_b};
    uint64_t dims[2] = {out_a_dim, out_b_dim};
    for (uint32_t i = 0; i < 2; i++) {
        if (scales[i] == 1.0f) continue;
        uint32_t n = (uint32_t)dims[i];
        enc = q36_encoder(@"q36_scale_f32");
        if (!enc) return 0;
        [enc setBuffer:outs[i]->buffer offset:outs[i]->offset atIndex:0];
        [enc setBytes:&n length:sizeof(n) atIndex:1];
        [enc setBytes:&scales[i] length:sizeof(scales[i]) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    return 1;
}

int q36_gpu_shared_ffn_decode_tensor(
        q36_gpu_tensor *out, q36_gpu_tensor *mid,
        const q36_gpu_tensor *x, const q36_gpu_tensor *scalar,
        const void *map, uint64_t size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim,
        float gate_scale, float up_scale, float down_scale) {
    const char *vk_style_env = getenv("Q36_METAL_SHARED_VK");
    const bool vk_style = !vk_style_env || vk_style_env[0] != '0';
    if (vk_style && out && mid && x && scalar && map && in_dim &&
        !(in_dim & 31u) && mid_dim && !(mid_dim & 31u) && out_dim &&
        x->bytes >= (uint64_t)in_dim * sizeof(float) &&
        scalar->bytes >= sizeof(float) &&
        mid->bytes >= (uint64_t)mid_dim * sizeof(float) &&
        out->bytes >= (uint64_t)out_dim * sizeof(float)) {
        const uint64_t gate_row_bytes = (uint64_t)(in_dim / 32u) * 34u;
        const uint64_t down_row_bytes = (uint64_t)(mid_dim / 32u) * 34u;
        uint64_t go = 0, uo = 0, doff = 0;
        id<MTLBuffer> gw = q36_model_view(
            map, size, gate_offset, gate_row_bytes * mid_dim, &go);
        id<MTLBuffer> uw = q36_model_view(
            map, size, up_offset, gate_row_bytes * mid_dim, &uo);
        id<MTLBuffer> dw = q36_model_view(
            map, size, down_offset, down_row_bytes * out_dim, &doff);
        if (gw && uw && dw && (q36_batch || q36_gpu_begin_commands())) {
            struct {
                uint32_t mid_dim, blocks;
                float gate_scale, up_scale;
            } ga = { mid_dim, in_dim / 32u, gate_scale, up_scale };
            id<MTLComputeCommandEncoder> enc =
                q36_encoder(@"q36_shared_gate_up_decode");
            if (!enc) return 0;
            [enc setBuffer:gw offset:(NSUInteger)go atIndex:0];
            [enc setBuffer:uw offset:(NSUInteger)uo atIndex:1];
            [enc setBuffer:x->buffer offset:x->offset atIndex:2];
            [enc setBuffer:mid->buffer offset:mid->offset atIndex:3];
            [enc setBytes:&ga length:sizeof(ga) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(mid_dim, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];

            struct {
                uint32_t out_dim, blocks;
                float scale;
            } da = { out_dim, mid_dim / 32u, down_scale };
            enc = q36_encoder(@"q36_shared_down_tail_decode");
            if (!enc) return 0;
            [enc setBuffer:dw offset:(NSUInteger)doff atIndex:0];
            [enc setBuffer:mid->buffer offset:mid->offset atIndex:1];
            [enc setBuffer:scalar->buffer offset:scalar->offset atIndex:2];
            [enc setBuffer:out->buffer offset:out->offset atIndex:3];
            [enc setBytes:&da length:sizeof(da) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(out_dim, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [enc endEncoding];
            return 1;
        }
    }
    const NSUInteger mid_bytes = (NSUInteger)mid_dim * sizeof(float);
    const NSUInteger down_bytes = (NSUInteger)out_dim * sizeof(float);
    if (q36_shared_mid_scratch_bytes < mid_bytes) {
        q36_shared_gate_scratch =
            [q36_device newBufferWithLength:mid_bytes options:MTLResourceStorageModePrivate];
        q36_shared_up_scratch =
            [q36_device newBufferWithLength:mid_bytes options:MTLResourceStorageModePrivate];
        if (!q36_shared_gate_scratch || !q36_shared_up_scratch) return 0;
        q36_shared_mid_scratch_bytes = mid_bytes;
    }
    if (q36_shared_down_scratch_bytes < down_bytes) {
        q36_shared_down_scratch =
            [q36_device newBufferWithLength:down_bytes options:MTLResourceStorageModePrivate];
        if (!q36_shared_down_scratch) return 0;
        q36_shared_down_scratch_bytes = down_bytes;
    }
    q36_gpu_tensor gate = {
        .buffer = q36_shared_gate_scratch, .bytes = mid_bytes
    };
    q36_gpu_tensor up = {
        .buffer = q36_shared_up_scratch, .bytes = mid_bytes
    };
    q36_gpu_tensor down = {
        .buffer = q36_shared_down_scratch, .bytes = down_bytes
    };
    return q36_gpu_matmul_q8_0_pair_scaled_tensor(
               &gate, &up, map, size, gate_offset, up_offset,
               in_dim, mid_dim, mid_dim, x, gate_scale, up_scale) &&
           q36_gpu_swiglu_tensor(mid, &gate, &up, mid_dim, 0.0f, 1.0f) &&
           q36_gpu_matmul_q8_0_scaled_tensor(
               &down, map, size, down_offset, mid_dim, out_dim,
               mid, 1, down_scale) &&
           q36_gpu_ffn_tail_tensor(out, &down, scalar, out_dim, 1);
}

int q36_gpu_rms_norm_rope_qwen_rows_tensor(
        q36_gpu_tensor *dst, const q36_gpu_tensor *src,
        const void *map, uint64_t size, uint64_t weight_offset,
        uint32_t src_stride, uint32_t heads,
        uint32_t pos0, uint32_t tokens, float eps) {
    if (!heads || !tokens || tokens > UINT32_MAX / heads) return 0;
    uint32_t rows = heads * tokens;
    const char *fused_env = getenv("Q36_METAL_NORM_ROPE");
    const bool fused = !fused_env || fused_env[0] != '0';
    if (src_stride < 256u || !dst || !src || !map ||
        dst->bytes < (uint64_t)rows * 256u * sizeof(float) ||
        src->bytes < ((uint64_t)(rows - 1u) * src_stride + 256u) *
                     sizeof(float) ||
        weight_offset > size ||
        256u * sizeof(float) > size - weight_offset) return 0;
    if (fused) {
        uint64_t weight_inner = 0;
        id<MTLBuffer> weight = q36_model_view(
            map, size, weight_offset, 256u * sizeof(float), &weight_inner);
        id<MTLComputeCommandEncoder> enc = weight
            ? q36_encoder(@"q36_rms_norm_rope_qwen_f32") : nil;
        if (enc) {
            struct {
                uint32_t src_stride, heads, pos0, tokens;
                float eps;
            } args = { src_stride, heads, pos0, tokens, eps };
            [enc setBuffer:dst->buffer offset:dst->offset atIndex:0];
            [enc setBuffer:src->buffer offset:src->offset atIndex:1];
            [enc setBuffer:weight offset:(NSUInteger)weight_inner atIndex:2];
            [enc setBytes:&args length:sizeof(args) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [enc endEncoding];
            return 1;
        }
    }
    if (!q36_copy_rows(dst, src, (q36_copy_rows_args){
            src_stride, 256u, 0u, 256u, rows
        })) return 0;
    return q36_gpu_rms_norm_weight_rows_tensor(
               dst, dst, map, size, weight_offset, 256u, rows, eps) &&
           q36_gpu_rope_qwen_rows_tensor(dst, heads, pos0, tokens);
}

int q36_gpu_rms_norm_rope_qwen_kv_store_tensor(
        q36_gpu_tensor *kcache, q36_gpu_tensor *vcache,
        const q36_gpu_tensor *k, const q36_gpu_tensor *v,
        const void *map, uint64_t size, uint64_t weight_offset,
        uint32_t src_stride, uint32_t heads,
        uint32_t pos0, uint32_t tokens, uint32_t cap, float eps) {
    uint64_t row = (uint64_t)heads * 256u;
    q36_gpu_tensor *kn = q36_gpu_tensor_alloc(row * tokens * sizeof(float));
    int ok = kn &&
        q36_gpu_rms_norm_rope_qwen_rows_tensor(
            kn, k, map, size, weight_offset, src_stride,
            heads, pos0, tokens, eps) &&
        q36_gpu_attn_kv_store_tensor(
            kcache, vcache, kn, v, pos0, tokens, cap,
            (uint32_t)row, (uint32_t)row, 0u, 0u,
            (uint32_t)row * 2u, (uint32_t)row * 2u);
    q36_gpu_tensor_free(kn);
    return ok;
}

int q36_gpu_rms_norm_rope_qwen_kv_store_quant_tensor(
        q36_gpu_tensor *kcache, q36_gpu_tensor *vcache,
        const q36_gpu_tensor *k, const q36_gpu_tensor *v,
        const void *map, uint64_t size, uint64_t weight_offset,
        uint32_t src_stride, uint32_t heads,
        uint32_t pos0, uint32_t tokens, uint32_t cap, float eps,
        uint32_t k_row_bytes, uint32_t v_row_bytes) {
    if (!q36_quant_q8_0 || !q36_quant_q4_0 || !kcache || !vcache ||
        !v || !heads || !tokens || pos0 > cap || tokens > cap - pos0)
        return 0;
    uint32_t row = heads * 256u;
    if (k_row_bytes != (row / 32u) * 34u ||
        v_row_bytes != (row / 32u) * 18u) return 0;
    q36_gpu_tensor *kn =
        q36_gpu_tensor_alloc((uint64_t)row * tokens * sizeof(float));
    if (!kn || !q36_gpu_rms_norm_rope_qwen_rows_tensor(
            kn, k, map, size, weight_offset, src_stride,
            heads, pos0, tokens, eps)) {
        q36_gpu_tensor_free(kn);
        return 0;
    }
    int ok = q36_gpu_attn_kv_store_tensor(
        kcache, vcache, kn, v, pos0, tokens, cap, row, row,
        1u, 2u, k_row_bytes, v_row_bytes);
    q36_gpu_tensor_free(kn);
    return ok;
}

static float q36_kv_quant_value(const uint8_t *row, uint32_t type,
                                uint32_t index) {
    if (type == 0u) {
        const uint8_t *p = row + (uint64_t)index * sizeof(uint16_t);
        return q36_quant_f16_to_f32(
            (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8)));
    }
    const uint8_t *block = row + (uint64_t)(index / 32u) *
        (type == 1u ? 34u : 18u);
    float d = q36_quant_f16_to_f32(
        (uint16_t)((uint16_t)block[0] | ((uint16_t)block[1] << 8)));
    uint32_t i = index & 31u;
    if (type == 1u) return d * (float)(int8_t)block[2u + i];
    uint8_t packed = block[2u + (i & 15u)];
    uint8_t q = i < 16u ? (packed & 15u) : (packed >> 4);
    return d * (float)((int)q - 8);
}

static int q36_attention_quant_host(
        q36_gpu_tensor *out, const q36_gpu_tensor *q,
        const q36_gpu_tensor *qg, const q36_gpu_tensor *kcache,
        const q36_gpu_tensor *vcache, q36_gpu_tensor *scores,
        const void *map, uint64_t size, uint64_t sinks_offset,
        bool has_sinks, uint32_t pos0, uint32_t tokens,
        uint32_t heads, uint32_t kv_heads, uint32_t dim,
        uint32_t k_type, uint32_t v_type,
        uint32_t k_row_bytes, uint32_t v_row_bytes) {
    if (!q36_quant_f16_to_f32 || !out || !q || !qg || !kcache || !vcache ||
        !tokens || !heads || !kv_heads || heads % kv_heads ||
        !dim || k_type > 2u || v_type > 2u || (dim & 31u) ||
        k_row_bytes != (k_type == 0u
            ? kv_heads * dim * sizeof(uint16_t)
            : kv_heads * (dim / 32u) * (k_type == 1u ? 34u : 18u)) ||
        v_row_bytes != (v_type == 0u
            ? kv_heads * dim * sizeof(uint16_t)
            : kv_heads * (dim / 32u) * (v_type == 1u ? 34u : 18u)))
        return 0;
    float *op = q36_gpu_tensor_contents(out);
    const float *qp = q36_gpu_tensor_contents((q36_gpu_tensor *)q);
    const float *qgp = q36_gpu_tensor_contents((q36_gpu_tensor *)qg);
    const uint8_t *kc = q36_gpu_tensor_contents((q36_gpu_tensor *)kcache);
    const uint8_t *vc = q36_gpu_tensor_contents((q36_gpu_tensor *)vcache);
    float *sp = scores ? q36_gpu_tensor_contents(scores) : NULL;
    const float *sinks = has_sinks && map && sinks_offset <= size &&
                         (uint64_t)heads * sizeof(float) <= size - sinks_offset
        ? (const float *)((const uint8_t *)map + sinks_offset) : NULL;
    bool own_scores = false;
    if (!sp) {
        sp = calloc((size_t)tokens * heads * (pos0 + tokens), sizeof(float));
        own_scores = true;
    }
    if (!op || !qp || !qgp || !kc || !vc || !sp || (has_sinks && !sinks)) {
        if (own_scores) free(sp);
        return 0;
    }
    const uint32_t score_stride = heads * (pos0 + tokens);
    const uint32_t score_cap = pos0 + tokens;
    const uint32_t group = heads / kv_heads;
    const float attn_scale = 1.0f / sqrtf((float)dim);
    for (uint32_t tok = 0; tok < tokens; tok++) {
        uint32_t count = pos0 + tok + 1u;
        for (uint32_t head = 0; head < heads; head++) {
            uint32_t kvh = head / group;
            const float *qh = qp + ((uint64_t)tok * heads + head) * dim;
            float *sh = sp + (uint64_t)tok * score_stride +
                        (uint64_t)head * score_cap;
            float maxv = sinks ? sinks[head] : -INFINITY;
            for (uint32_t pos = 0; pos < count; pos++) {
                const uint8_t *kr = kc + (uint64_t)pos * k_row_bytes;
                float dot = 0.0f;
                for (uint32_t j = 0; j < dim; j++)
                    dot = fmaf(qh[j],
                               q36_kv_quant_value(kr, k_type, kvh * dim + j),
                               dot);
                sh[pos] = dot * attn_scale;
                if (sh[pos] > maxv) maxv = sh[pos];
            }
            float denom = sinks ? expf(sinks[head] - maxv) : 0.0f;
            for (uint32_t pos = 0; pos < count; pos++) {
                sh[pos] = expf(sh[pos] - maxv);
                denom += sh[pos];
            }
            float inv = denom > 0.0f ? 1.0f / denom : 1.0f;
            const float *gate = qgp + (uint64_t)tok * heads * dim * 2u +
                                (uint64_t)head * dim * 2u + dim;
            float *oh = op + ((uint64_t)tok * heads + head) * dim;
            for (uint32_t j = 0; j < dim; j++) {
                float value = 0.0f;
                for (uint32_t pos = 0; pos < count; pos++) {
                    const uint8_t *vr = vc + (uint64_t)pos * v_row_bytes;
                    value = fmaf(sh[pos] * inv,
                                 q36_kv_quant_value(vr, v_type, kvh * dim + j),
                                 value);
                }
                float g = gate[j];
                float sig = g >= 0.0f ? 1.0f / (1.0f + expf(-g))
                                      : expf(g) / (1.0f + expf(g));
                oh[j] = value * sig;
            }
        }
    }
    if (own_scores) free(sp);
    return 1;
}

int q36_gpu_attn_decode_tensor(
        q36_gpu_tensor *out, const q36_gpu_tensor *q,
        const q36_gpu_tensor *qg, const q36_gpu_tensor *kcache,
        const q36_gpu_tensor *vcache, q36_gpu_tensor *scores,
        const void *map, uint64_t size, uint64_t sinks_offset,
        bool has_sinks, uint32_t pos0, uint32_t tokens,
        uint32_t heads, uint32_t kv_heads, uint32_t dim,
        uint32_t k_type, uint32_t v_type,
        uint32_t k_row_bytes, uint32_t v_row_bytes) {
    if (!scores || ((k_type != 0u || v_type != 0u) &&
                    !(k_type == 1u && v_type == 2u)))
        return q36_attention_quant_host(
            out, q, qg, kcache, vcache, scores, map, size, sinks_offset,
            has_sinks, pos0, tokens, heads, kv_heads, dim,
            k_type, v_type, k_row_bytes, v_row_bytes);
    bool quant = k_type == 1u && v_type == 2u &&
        k_row_bytes == kv_heads * (dim / 32u) * 34u &&
        v_row_bytes == kv_heads * (dim / 32u) * 18u;
    if (!out || !q || !qg || !kcache || !vcache || !scores ||
        !tokens || !heads || !kv_heads || heads % kv_heads ||
        !dim || (!quant &&
        (k_type != 0u || v_type != 0u ||
         k_row_bytes != kv_heads * dim * sizeof(uint16_t) ||
         v_row_bytes != kv_heads * dim * sizeof(uint16_t)))) return 0;
    uint64_t sink_inner = 0;
    id<MTLBuffer> sink_buffer = has_sinks
        ? q36_model_view(map, size, sinks_offset,
                         (uint64_t)heads * sizeof(float), &sink_inner)
        : qg->buffer;
    if (!sink_buffer) return 0;
    struct {
        uint32_t pos0, heads, kv_heads, dim;
        uint32_t qg_stride, has_sinks, score_stride;
    } args = {
        pos0, heads, kv_heads, dim, heads * dim * 2u,
        has_sinks ? 1u : 0u, heads * (pos0 + tokens)
    };
    uint32_t n_spans = (pos0 + tokens + 511u) / 512u;
    const char *split_env = getenv("Q36_METAL_ATTN_SPLIT");
    bool split_enabled = !split_env || !split_env[0] || split_env[0] != '0';
    const char *gqa_env = getenv("Q36_METAL_ATTN_GQA");
    bool gqa_enabled = !gqa_env || !gqa_env[0] || gqa_env[0] != '0';
    uint32_t gqa_groups = (pos0 + tokens + 4095u) / 4096u;
    bool gqa = gqa_enabled && quant && tokens >= 2u &&
        pos0 + tokens > 1024u && heads == 16u && kv_heads == 2u &&
        dim == 256u &&
        (uint64_t)tokens * heads * gqa_groups * (dim + 2u) * sizeof(float)
            <= scores->bytes;
    if (gqa) {
        struct {
            uint32_t pos0, heads, kv_heads, dim;
            uint32_t qg_stride, has_sinks, n_spans;
        } gqa_args = {
            pos0, heads, kv_heads, dim, heads * dim * 2u,
            has_sinks ? 1u : 0u, gqa_groups
        };
        id<MTLComputeCommandEncoder> gqa_enc =
            q36_encoder(@"q36_attention_q8_q4_gqa8");
        if (!gqa_enc) return 0;
        [gqa_enc setBuffer:scores->buffer offset:scores->offset atIndex:0];
        [gqa_enc setBuffer:q->buffer offset:q->offset atIndex:1];
        [gqa_enc setBuffer:kcache->buffer offset:kcache->offset atIndex:2];
        [gqa_enc setBuffer:vcache->buffer offset:vcache->offset atIndex:3];
        [gqa_enc setBytes:&gqa_args length:sizeof(gqa_args) atIndex:4];
        [gqa_enc setThreadgroupMemoryLength:6208u * sizeof(float) atIndex:0];
        [gqa_enc dispatchThreadgroups:MTLSizeMake(kv_heads, tokens, gqa_groups)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [gqa_enc endEncoding];
        id<MTLComputeCommandEncoder> combine =
            q36_encoder(@"q36_attention_gqa8_combine");
        if (!combine) return 0;
        [combine setBuffer:out->buffer offset:out->offset atIndex:0];
        [combine setBuffer:scores->buffer offset:scores->offset atIndex:1];
        [combine setBuffer:qg->buffer offset:qg->offset atIndex:2];
        [combine setBuffer:sink_buffer
                    offset:(has_sinks ? (NSUInteger)sink_inner : qg->offset)
                   atIndex:3];
        [combine setBytes:&gqa_args length:sizeof(gqa_args) atIndex:4];
        [combine dispatchThreadgroups:MTLSizeMake(heads, tokens, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [combine endEncoding];
        return 1;
    }
    bool split = split_enabled && quant && pos0 + tokens > 1024u &&
        (uint64_t)tokens * heads * n_spans * (dim + 2u) * sizeof(float)
            <= scores->bytes;
    if (split) {
        struct {
            uint32_t pos0, heads, kv_heads, dim;
            uint32_t qg_stride, has_sinks, n_spans;
        } split_args = {
            pos0, heads, kv_heads, dim, heads * dim * 2u,
            has_sinks ? 1u : 0u, n_spans
        };
        id<MTLComputeCommandEncoder> split_enc =
            q36_encoder(@"q36_attention_q8_q4_split");
        if (!split_enc) return 0;
        [split_enc setBuffer:scores->buffer offset:scores->offset atIndex:0];
        [split_enc setBuffer:q->buffer offset:q->offset atIndex:1];
        [split_enc setBuffer:kcache->buffer offset:kcache->offset atIndex:2];
        [split_enc setBuffer:vcache->buffer offset:vcache->offset atIndex:3];
        [split_enc setBytes:&split_args length:sizeof(split_args) atIndex:4];
        [split_enc setThreadgroupMemoryLength:520u * sizeof(float) atIndex:0];
        [split_enc dispatchThreadgroups:MTLSizeMake(heads, tokens, n_spans)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [split_enc endEncoding];
        id<MTLComputeCommandEncoder> combine =
            q36_encoder(@"q36_attention_split_combine");
        if (!combine) return 0;
        [combine setBuffer:out->buffer offset:out->offset atIndex:0];
        [combine setBuffer:scores->buffer offset:scores->offset atIndex:1];
        [combine setBuffer:qg->buffer offset:qg->offset atIndex:2];
        [combine setBuffer:sink_buffer
                    offset:(has_sinks ? (NSUInteger)sink_inner : qg->offset)
                   atIndex:3];
        [combine setBytes:&split_args length:sizeof(split_args) atIndex:4];
        [combine dispatchThreadgroups:MTLSizeMake(heads, tokens, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [combine endEncoding];
        return 1;
    }
    id<MTLComputeCommandEncoder> enc =
        q36_encoder(quant ? @"q36_attention_q8_q4_parallel"
                          : @"q36_attention_f16_parallel");
    if (!enc) return 0;
    [enc setBuffer:out->buffer offset:out->offset atIndex:0];
    [enc setBuffer:scores->buffer offset:scores->offset atIndex:1];
    [enc setBuffer:q->buffer offset:q->offset atIndex:2];
    [enc setBuffer:qg->buffer offset:qg->offset atIndex:3];
    [enc setBuffer:kcache->buffer offset:kcache->offset atIndex:4];
    [enc setBuffer:vcache->buffer offset:vcache->offset atIndex:5];
    [enc setBuffer:sink_buffer
            offset:(has_sinks ? (NSUInteger)sink_inner : qg->offset)
           atIndex:6];
    [enc setBytes:&args length:sizeof(args) atIndex:7];
    [enc setThreadgroupMemoryLength:8u * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(heads, tokens, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    return 1;
}

static uint64_t q36_quant_row_bytes(uint32_t type, uint32_t width) {
    uint32_t block = 0, bytes = 0;
    switch (type) {
    case 8:  block = 32u;  bytes = 34u;  break;
    case 10: block = 256u; bytes = 84u;  break;
    case 12: block = 256u; bytes = 144u; break;
    case 13: block = 256u; bytes = 176u; break;
    case 14: block = 256u; bytes = 210u; break;
    case 16: block = 256u; bytes = 66u;  break;
    case 21: block = 256u; bytes = 110u; break;
    case 22: block = 256u; bytes = 82u;  break;
    default: return 0;
    }
    return width % block ? 0 : (uint64_t)(width / block) * bytes;
}

typedef struct {
    int32_t nei0, nei1; uint64_t nbi1;
    int32_t ne00, ne01, ne02; uint64_t nb00, nb01, nb02;
    int32_t ne10, ne11, ne12, ne13; uint64_t nb10, nb11, nb12;
    int32_t ne0, ne1; uint64_t nb1; int32_t nr0;
    int32_t tp_rank, tp_world, tp_addend, tp_expert_base;
} q36_moe_mv_args;

typedef struct { uint32_t width, used, tokens, has_down_scale; } q36_moe_reduce_args;
typedef struct {
    uint32_t width, used, tokens, has_gate_scale, has_up_scale,
             has_down_scale;
} q36_moe_activation_args;
typedef struct {
    int32_t ne02, ne10, ne11;
    uint64_t nb11, nb12;
    int32_t ne21, ne20;
    uint64_t nb21;
} q36_moe_mm_map_args;
typedef struct {
    int32_t ne00, ne02;
    uint64_t nb01, nb02, nb03;
    int32_t ne11;
    uint64_t nb10, nb11, nb12, nb13;
    int32_t ne20, ne21, ne0, ne1;
    int16_t r2, r3;
    int32_t tp_rank, tp_world, tp_expert_base;
} q36_moe_mm_args;

static q36_moe_mv_args q36_moe_make_args(
        uint32_t cols, uint32_t rows, uint32_t experts,
        uint64_t row_bytes, uint64_t expert_bytes,
        uint32_t rhs_rows, uint32_t used, uint32_t tokens) {
    return (q36_moe_mv_args) {
        .nei0 = (int32_t)used, .nei1 = (int32_t)tokens,
        .nbi1 = (uint64_t)used * sizeof(int32_t),
        .ne00 = (int32_t)cols, .ne01 = (int32_t)rows, .ne02 = (int32_t)experts,
        .nb00 = row_bytes / (cols / 256u), .nb01 = row_bytes, .nb02 = expert_bytes,
        .ne10 = (int32_t)cols, .ne11 = (int32_t)rhs_rows,
        .ne12 = (int32_t)tokens, .ne13 = 1, .nb10 = sizeof(float),
        .nb11 = (uint64_t)cols * sizeof(float),
        .nb12 = (uint64_t)rhs_rows * cols * sizeof(float),
        .ne0 = (int32_t)rows, .ne1 = (int32_t)used,
        .nb1 = (uint64_t)rows * sizeof(float), .nr0 = 4, .tp_world = 1,
    };
}

static int q36_moe_ensure_scratch(uint64_t pair_bytes, uint64_t down_bytes) {
    if (pair_bytes > NSUIntegerMax || down_bytes > NSUIntegerMax) return 0;
    if (q36_moe_pair_scratch_bytes < pair_bytes) {
        q36_moe_gate_scratch = [q36_device newBufferWithLength:(NSUInteger)pair_bytes options:MTLResourceStorageModePrivate];
        if (!q36_moe_gate_scratch) return 0;
        q36_moe_pair_scratch_bytes = (NSUInteger)pair_bytes;
    }
    /* Up activations are consumed when the in-place SiLU writes gate scratch;
     * only then does down projection need its output buffer. Reuse the down
     * allocation for up, keeping gate/mid separate because down reads it. */
    const uint64_t shared_bytes = down_bytes > pair_bytes ? down_bytes : pair_bytes;
    if (q36_moe_down_scratch_bytes < shared_bytes) {
        q36_moe_down_scratch = [q36_device
            newBufferWithLength:(NSUInteger)shared_bytes
                        options:MTLResourceStorageModePrivate];
        if (!q36_moe_down_scratch) return 0;
        q36_moe_down_scratch_bytes = (NSUInteger)shared_bytes;
    }
    q36_moe_up_scratch = q36_moe_down_scratch;
    return 1;
}

static const char *q36_moe_mm_map_name(uint32_t used, bool compact) {
    if (compact) {
        return used == 8u
            ? "kernel_mul_mm_id_map0_ne20_8_compact"
            : NULL;
    }
    switch (used) {
    case 1:  return "kernel_mul_mm_id_map0_ne20_1";
    case 2:  return "kernel_mul_mm_id_map0_ne20_2";
    case 4:  return "kernel_mul_mm_id_map0_ne20_4";
    case 5:  return "kernel_mul_mm_id_map0_ne20_5";
    case 6:  return "kernel_mul_mm_id_map0_ne20_6";
    case 8:  return "kernel_mul_mm_id_map0_ne20_8";
    case 10: return "kernel_mul_mm_id_map0_ne20_10";
    case 16: return "kernel_mul_mm_id_map0_ne20_16";
    case 22: return "kernel_mul_mm_id_map0_ne20_22";
    default: return NULL;
    }
}

static int q36_moe_ensure_id_map(
        uint32_t experts, uint32_t tokens, uint32_t used, bool compact) {
    uint64_t words = (uint64_t)experts * (tokens + 1u);
    if (compact) {
        const uint64_t max_blocks =
            ((uint64_t)tokens * used + 31u) / 32u + experts;
        if (max_blocks > (UINT64_MAX - words) / 2u) return 0;
        words += 2u * max_blocks;
    }
    if (words > NSUIntegerMax / sizeof(int32_t)) return 0;
    const NSUInteger bytes = (NSUInteger)(words * sizeof(int32_t));
    if (q36_moe_id_map_bytes < bytes) {
        q36_moe_id_map = [q36_device
            newBufferWithLength:bytes
                        options:MTLResourceStorageModePrivate];
        if (!q36_moe_id_map) return 0;
        q36_moe_id_map_bytes = bytes;
    }
    return 1;
}

static q36_moe_mm_args q36_moe_make_mm_args(
        uint32_t cols, uint32_t rows, uint32_t experts,
        uint64_t row_bytes, uint64_t expert_bytes,
        uint32_t rhs_rows, uint32_t used, uint32_t tokens) {
    return (q36_moe_mm_args) {
        .ne00 = (int32_t)cols, .ne02 = (int32_t)experts,
        .nb01 = row_bytes, .nb02 = expert_bytes,
        .nb03 = expert_bytes * experts, .ne11 = (int32_t)rhs_rows,
        .nb10 = sizeof(float), .nb11 = (uint64_t)cols * sizeof(float),
        .nb12 = (uint64_t)rhs_rows * cols * sizeof(float),
        .nb13 = (uint64_t)tokens * rhs_rows * cols * sizeof(float),
        .ne20 = (int32_t)used, .ne21 = (int32_t)tokens,
        .ne0 = (int32_t)rows, .ne1 = (int32_t)used,
        .r2 = 1, .r3 = 1, .tp_world = 1,
    };
}

typedef struct {
    id<MTLBuffer> gate;
    id<MTLBuffer> up;
    id<MTLBuffer> down;
    uint64_t gate_expert_stride;
    uint64_t up_expert_stride;
    uint64_t down_expert_stride;
    id<MTLBuffer> selected;
    NSUInteger selected_offset;
    uint32_t experts;
} q36_metal_stream_binding;

static bool q36_moe_stream_binding(
        q36_metal_stream_binding *binding,
        const void *map, uint64_t size,
        const q36_gpu_moe_weight *gate,
        const q36_gpu_moe_weight *up,
        const q36_gpu_moe_weight *down,
        const q36_gpu_tensor *selected,
        uint32_t tokens, uint32_t used, uint32_t experts,
        uint32_t layer, uint64_t gate_expert_bytes,
        uint64_t up_expert_bytes, uint64_t down_expert_bytes) {
    if (!binding || !selected) return false;
    *binding = (q36_metal_stream_binding) {
        .gate_expert_stride = gate_expert_bytes,
        .up_expert_stride = up_expert_bytes,
        .down_expert_stride = down_expert_bytes,
        .selected = selected->buffer,
        .selected_offset = (NSUInteger)selected->offset,
        .experts = experts,
    };
    if (!q36_ssd_streaming) return true;
    q36_gpu_stream_expert_table table = {
        .model_map = map,
        .model_size = size,
        .layer = layer,
        .n_total_expert = experts,
        .gate_offset = gate->offset,
        .up_offset = up->offset,
        .down_offset = down->offset,
        .gate_scales_offset = gate->scales_offset,
        .up_scales_offset = up->scales_offset,
        .down_scales_offset = down->scales_offset,
        .gate_expert_bytes = gate_expert_bytes,
        .up_expert_bytes = up_expert_bytes,
        .down_expert_bytes = down_expert_bytes,
        .gate_type = gate->type,
        .up_type = up->type,
        .down_type = down->type,
        .has_gate_scales = gate->has_scales,
        .has_up_scales = up->has_scales,
        .has_down_scales = down->has_scales,
    };
    if (layer < q36_streaming_full_layers) {
        if (!q36_metal_wait()) return false;
        pthread_mutex_lock(&q36_stream_mu);
        bool ok = q36_streaming_full_layer_load(&table);
        if (ok) {
            q36_metal_full_layer *full = &q36_streaming_full[layer];
            binding->gate = full->gate;
            binding->up = full->up;
            binding->down = full->down;
        }
        pthread_mutex_unlock(&q36_stream_mu);
        return ok;
    }

    uint64_t count64 = (uint64_t)tokens * used;
    if (!count64 || count64 > UINT32_MAX ||
        selected->bytes < count64 * sizeof(int32_t)) return false;
    const int32_t *ids =
        q36_gpu_tensor_contents((q36_gpu_tensor *)selected);
    if (!ids) return false;
    uint32_t slots[Q36_METAL_STREAM_MAX_EXPERTS];
    if (!q36_streaming_prepare(&table, ids, NULL, (uint32_t)count64,
                                slots)) return false;
    uint32_t *mapped = malloc((size_t)count64 * sizeof(*mapped));
    if (!mapped) return false;
    for (uint64_t i = 0; i < count64; i++) {
        uint32_t expert = (uint32_t)ids[i];
        if (expert >= experts || slots[expert] == UINT32_MAX) {
            free(mapped);
            return false;
        }
        mapped[i] = slots[expert];
    }
    id<MTLBuffer> mapped_buffer =
        [q36_device newBufferWithBytes:mapped
                                length:(NSUInteger)(count64 * sizeof(*mapped))
                               options:MTLResourceStorageModeShared];
    free(mapped);
    if (!mapped_buffer) return false;
    mapped_buffer.label = @"q36_stream_selected_slots";
    pthread_mutex_lock(&q36_stream_mu);
    binding->gate = q36_streaming_gate;
    binding->up = q36_streaming_up;
    binding->down = q36_streaming_down;
    binding->gate_expert_stride = q36_streaming_gate_bytes;
    binding->up_expert_stride = q36_streaming_up_bytes;
    binding->down_expert_stride = q36_streaming_down_bytes;
    binding->experts = q36_streaming_expert_cap;
    pthread_mutex_unlock(&q36_stream_mu);
    binding->selected = mapped_buffer;
    binding->selected_offset = 0;
    return binding->gate && binding->up && binding->down &&
           binding->experts != 0;
}

static int q36_moe_gpu(
        q36_gpu_tensor *out, const void *map, uint64_t size,
        const q36_gpu_moe_weight *gate, const q36_gpu_moe_weight *up,
        const q36_gpu_moe_weight *down, const q36_gpu_tensor *selected,
        const q36_gpu_tensor *route_weights, uint32_t used,
        const q36_gpu_tensor *x, uint32_t tokens, uint32_t in_dim,
        uint32_t mid_dim, uint32_t out_dim, uint32_t experts,
        uint32_t layer) {
    if (!out || !map || !gate || !up || !down || !selected ||
        !route_weights || !x || !used || !tokens ||
        (in_dim & 255u) || (mid_dim & 255u)) return 0;
    const bool iq2_q2 =
        gate->type == 16u && up->type == 16u && down->type == 10u;
    const bool q4 =
        gate->type == 12u && up->type == 12u && down->type == 12u;
    if (!iq2_q2 && !q4) return 0;
    const uint64_t grb = q36_quant_row_bytes(gate->type, in_dim);
    const uint64_t urb = q36_quant_row_bytes(up->type, in_dim);
    const uint64_t drb = q36_quant_row_bytes(down->type, mid_dim);
    const uint64_t geb = grb * mid_dim, ueb = urb * mid_dim, deb = drb * out_dim;
    const uint64_t pairs = (uint64_t)tokens * used;
    if (gate->offset > size || geb * experts > size - gate->offset ||
        up->offset > size || ueb * experts > size - up->offset ||
        down->offset > size || deb * experts > size - down->offset ||
        selected->bytes < pairs * sizeof(int32_t) ||
        route_weights->bytes < pairs * sizeof(float) ||
        x->bytes < (uint64_t)tokens * in_dim * sizeof(float) ||
        out->bytes < (uint64_t)tokens * out_dim * sizeof(float) ||
        !q36_moe_ensure_scratch(pairs * mid_dim * sizeof(float),
                                pairs * out_dim * sizeof(float))) return 0;

    q36_metal_stream_binding stream = {0};
    if (!q36_moe_stream_binding(&stream, map, size, gate, up, down,
                                 selected, tokens, used, experts, layer,
                                 geb, ueb, deb)) return 0;
    const uint32_t weight_experts = stream.experts;
    id<MTLBuffer> weight_selected = stream.selected;
    const NSUInteger weight_selected_offset = stream.selected_offset;
    uint64_t go = 0, uo = 0, doff = 0, gso = 0, uso = 0, dso = 0;
    id<MTLBuffer> gb = stream.gate;
    id<MTLBuffer> ub = stream.up;
    id<MTLBuffer> db = stream.down;
    uint64_t gate_stride = stream.gate_expert_stride;
    uint64_t up_stride = stream.up_expert_stride;
    uint64_t down_stride = stream.down_expert_stride;
    /* The paired gate/up kernels share one expert-stride field.  Q36 model
     * layouts keep these tensors symmetric; reject malformed asymmetric
     * layouts instead of indexing the up buffer with the gate stride. */
    if (gate_stride != up_stride) return 0;
    if (!q36_ssd_streaming) {
        gb = q36_model_view(map, size, gate->offset, geb * experts, &go);
        ub = q36_model_view(map, size, up->offset, ueb * experts, &uo);
        db = q36_model_view(map, size, down->offset, deb * experts, &doff);
    }
    id<MTLBuffer> gsb = route_weights->buffer, usb = route_weights->buffer, dsb = route_weights->buffer;
    if (gate->has_scales) gsb = q36_model_view(map, size, gate->scales_offset, (uint64_t)experts * 4u, &gso);
    if (up->has_scales) usb = q36_model_view(map, size, up->scales_offset, (uint64_t)experts * 4u, &uso);
    if (down->has_scales) dsb = q36_model_view(map, size, down->scales_offset, (uint64_t)experts * 4u, &dso);
    if (!gb || !ub || !db || !gsb || !usb || !dsb || (!q36_batch && !q36_gpu_begin_commands())) return 0;

    const char *mm_env = getenv("Q36_METAL_MOE_MM");
    const char *compact_env = getenv("Q36_METAL_MOE_COMPACT");
    const bool compact = iq2_q2 && used == 8u &&
        (!compact_env || compact_env[0] != '0');
    const char *map_name = q36_moe_mm_map_name(used, compact);
    bool use_mm = tokens >= 64u && !q36_micro_batch && map_name &&
        (!mm_env || mm_env[0] != '0');
    id<MTLComputePipelineState> map_pipeline = nil;
    id<MTLComputePipelineState> gate_pipeline = nil;
    id<MTLComputePipelineState> down_pipeline = nil;
    if (use_mm) {
        map_pipeline = q36_pipeline_mm(
            [NSString stringWithUTF8String:map_name], false, false);
        gate_pipeline = q36_pipeline_mm(
            compact ? @"kernel_mul_mm_id_iq2_xxs_f32_compact" :
            (q4 ? @"kernel_mul_mm_id_q4_K_f32"
                : @"kernel_mul_mm_id_iq2_xxs_f32"),
            false, true);
        down_pipeline = q36_pipeline_mm(
            compact ? @"kernel_mul_mm_id_q2_K_f32_compact" :
            (q4 ? @"kernel_mul_mm_id_q4_K_f32"
                : @"kernel_mul_mm_id_q2_K_f32"),
            false, true);
        const uint64_t map_shmem =
            (uint64_t)weight_experts * used * sizeof(uint16_t);
        use_mm = map_pipeline && gate_pipeline && down_pipeline &&
            weight_experts <= map_pipeline.maxTotalThreadsPerThreadgroup &&
            128u <= gate_pipeline.maxTotalThreadsPerThreadgroup &&
            128u <= down_pipeline.maxTotalThreadsPerThreadgroup &&
            map_shmem <= q36_device.maxThreadgroupMemoryLength &&
            8192u <= q36_device.maxThreadgroupMemoryLength;
        if (use_mm) {
            use_mm = q36_moe_ensure_id_map(
                weight_experts, tokens, used, compact);
        }
    }

    q36_moe_mm_args gate_mm = {0};
    q36_moe_mm_args down_mm = {0};
    NSUInteger map_ids_offset = 0;
    NSUInteger mm_blocks = 0;
    if (use_mm) {
        map_ids_offset =
            (NSUInteger)weight_experts * sizeof(int32_t);
        q36_moe_mm_map_args ma = {
            (int32_t)weight_experts, (int32_t)in_dim, 1,
            (uint64_t)in_dim * sizeof(float),
            (uint64_t)in_dim * sizeof(float),
            (int32_t)tokens, (int32_t)used,
            (uint64_t)used * sizeof(int32_t)
        };
        id<MTLComputeCommandEncoder> menc =
            [q36_batch computeCommandEncoder];
        if (!menc) return 0;
        menc.label = [NSString stringWithUTF8String:map_name];
        [menc setComputePipelineState:map_pipeline];
        [menc setBytes:&ma length:sizeof(ma) atIndex:0];
        [menc setBuffer:weight_selected
                 offset:weight_selected_offset atIndex:1];
        [menc setBuffer:q36_moe_id_map offset:0 atIndex:2];
        [menc setBuffer:q36_moe_id_map offset:map_ids_offset atIndex:3];
        [menc setThreadgroupMemoryLength:
            (NSUInteger)weight_experts * used * sizeof(uint16_t) atIndex:0];
        [menc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(weight_experts, 1, 1)];
        [menc endEncoding];
        gate_mm = q36_moe_make_mm_args(
            in_dim, mid_dim, weight_experts, grb, gate_stride,
            1, used, tokens);
        down_mm = q36_moe_make_mm_args(
            mid_dim, out_dim, weight_experts, drb, down_stride,
            used, used, tokens);
        mm_blocks = compact
            ? (NSUInteger)(((uint64_t)tokens * used + 31u) / 32u +
                           weight_experts)
            : weight_experts;
    }

    /* Every routed slot for a token consumes the same input row.  The id
     * kernel's ne11/nb12 mapping must therefore expose one RHS row per token;
     * using `used` here walks into following token rows (and out of bounds
     * during decode). */
    q36_moe_mv_args ga =
        q36_moe_make_args(in_dim, mid_dim, weight_experts,
                          grb, gate_stride, 1, used, tokens);
    ga.nr0 = q4 ? 2 : 4;
    id<MTLComputePipelineState> gp = use_mm
        ? gate_pipeline
        : q36_pipeline_nsg(
            q4 ? @"kernel_mul_mv_id_q4_K_pair_f32"
               : @"kernel_mul_mv_id_iq2_xxs_pair_f32", 2);
    id<MTLComputeCommandEncoder> enc = gp ? [q36_batch computeCommandEncoder] : nil;
    if (!enc) return 0;
    enc.label = use_mm
        ? @"moe.gate.kernel_mul_mm_id_iq2_xxs_f32"
        : @"moe.gate_up.kernel_mul_mv_id_iq2_xxs_pair_f32";
    [enc setComputePipelineState:gp];
    if (use_mm) {
        [enc setBytes:&gate_mm length:sizeof(gate_mm) atIndex:0];
        [enc setBuffer:gb offset:(NSUInteger)go atIndex:1];
        [enc setBuffer:x->buffer offset:x->offset atIndex:2];
        [enc setBuffer:q36_moe_id_map offset:0 atIndex:3];
        [enc setBuffer:q36_moe_id_map offset:map_ids_offset atIndex:4];
        [enc setBuffer:q36_moe_gate_scratch offset:0 atIndex:5];
        [enc setThreadgroupMemoryLength:8192u atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(
                compact ? 1u : (tokens + 31u) / 32u,
                (mid_dim + 63u) / 64u, mm_blocks)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];

        enc = [q36_batch computeCommandEncoder];
        enc.label = @"moe.up.kernel_mul_mm_id_iq2_xxs_f32";
        [enc setComputePipelineState:gate_pipeline];
        [enc setBytes:&gate_mm length:sizeof(gate_mm) atIndex:0];
        [enc setBuffer:ub offset:(NSUInteger)uo atIndex:1];
        [enc setBuffer:x->buffer offset:x->offset atIndex:2];
        [enc setBuffer:q36_moe_id_map offset:0 atIndex:3];
        [enc setBuffer:q36_moe_id_map offset:map_ids_offset atIndex:4];
        [enc setBuffer:q36_moe_up_scratch offset:0 atIndex:5];
        [enc setThreadgroupMemoryLength:8192u atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(
                compact ? 1u : (tokens + 31u) / 32u,
                (mid_dim + 63u) / 64u, mm_blocks)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];
    } else {
        [enc setBytes:&ga length:sizeof(ga) atIndex:0];
        [enc setBuffer:gb offset:(NSUInteger)go atIndex:1];
        [enc setBuffer:ub offset:(NSUInteger)uo atIndex:2];
        [enc setBuffer:x->buffer offset:x->offset atIndex:3];
        [enc setBuffer:q36_moe_gate_scratch offset:0 atIndex:4];
        [enc setBuffer:q36_moe_up_scratch offset:0 atIndex:5];
        [enc setBuffer:weight_selected
                 offset:weight_selected_offset atIndex:6];
        [enc setThreadgroupMemoryLength:q4 ? 0u : 2176u atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(
                (mid_dim + (q4 ? 3u : 7u)) / (q4 ? 4u : 8u),
                1, (NSUInteger)pairs)
             threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
        [enc endEncoding];
    }

    const char *down_sum_env = getenv("Q36_METAL_MOE_DOWN_SUM");
    const bool down_sum = tokens == 1u &&
        (!down_sum_env || down_sum_env[0] != '0');
    q36_moe_activation_args aa = {
        mid_dim, used, tokens, gate->has_scales, up->has_scales,
        down_sum && down->has_scales
    };
    enc = q36_encoder(@"q36_moe_activation_f32");
    if (!enc) return 0;
    /* Activation consumes gate element i before replacing the same element
     * with SiLU(gate)*up.  Reusing the gate buffer removes one 16 MiB
     * 1024-token routed-expert scratch allocation without changing order. */
    [enc setBuffer:q36_moe_gate_scratch offset:0 atIndex:0];
    [enc setBuffer:q36_moe_gate_scratch offset:0 atIndex:1];
    [enc setBuffer:q36_moe_up_scratch offset:0 atIndex:2];
    [enc setBuffer:selected->buffer offset:selected->offset atIndex:3];
    [enc setBuffer:route_weights->buffer offset:route_weights->offset atIndex:4];
    [enc setBuffer:gsb offset:(NSUInteger)gso atIndex:5];
    [enc setBuffer:usb offset:(NSUInteger)uso atIndex:6];
    [enc setBytes:&aa length:sizeof(aa) atIndex:7];
    [enc setBuffer:dsb offset:(NSUInteger)dso atIndex:8];
    [enc dispatchThreads:MTLSizeMake(mid_dim, (NSUInteger)pairs, 1)
   threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];

    q36_moe_mv_args da = q36_moe_make_args(
        mid_dim, out_dim, weight_experts,
        drb, down_stride, used, used, tokens);
    da.nr0 = q4 ? 2 : 4;
    id<MTLComputePipelineState> dp = use_mm
        ? down_pipeline
        : q36_pipeline_nsg(
            down_sum
                ? (q4 ? @"kernel_mul_mv_id_q4_K_sum6_f32"
                      : @"kernel_mul_mv_id_q2_K_sum6_f32")
                : (q4 ? @"kernel_mul_mv_id_q4_K_f32"
                      : @"kernel_mul_mv_id_q2_K_f32"), 2);
    enc = dp ? [q36_batch computeCommandEncoder] : nil;
    if (!enc) return 0;
    enc.label = use_mm
        ? @"moe.down.kernel_mul_mm_id_q2_K_f32"
        : (down_sum
            ? @"moe.down.kernel_mul_mv_id_q2_K_sum6_f32"
            : @"moe.down.kernel_mul_mv_id_q2_K_f32");
    [enc setComputePipelineState:dp];
    if (use_mm) {
        [enc setBytes:&down_mm length:sizeof(down_mm) atIndex:0];
        [enc setBuffer:db offset:(NSUInteger)doff atIndex:1];
        [enc setBuffer:q36_moe_gate_scratch offset:0 atIndex:2];
        [enc setBuffer:q36_moe_id_map offset:0 atIndex:3];
        [enc setBuffer:q36_moe_id_map offset:map_ids_offset atIndex:4];
        [enc setBuffer:q36_moe_down_scratch offset:0 atIndex:5];
        [enc setThreadgroupMemoryLength:8192u atIndex:0];
        [enc dispatchThreadgroups:MTLSizeMake(
                compact ? 1u : (tokens + 31u) / 32u,
                (out_dim + 63u) / 64u, mm_blocks)
             threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else {
        [enc setBytes:&da length:sizeof(da) atIndex:0];
        [enc setBuffer:db offset:(NSUInteger)doff atIndex:1];
        [enc setBuffer:q36_moe_gate_scratch offset:0 atIndex:2];
        [enc setBuffer:(down_sum ? out->buffer : q36_moe_down_scratch)
                offset:(down_sum ? out->offset : 0) atIndex:3];
        [enc setBuffer:weight_selected
                 offset:weight_selected_offset atIndex:4];
    }
    if (down_sum) {
        [enc setBuffer:out->buffer offset:out->offset atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(
                (out_dim + (q4 ? 3u : 7u)) / (q4 ? 4u : 8u),
                tokens, 1)
             threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    } else if (!use_mm) {
        [enc dispatchThreadgroups:MTLSizeMake(
                (out_dim + (q4 ? 3u : 7u)) / (q4 ? 4u : 8u),
                1, (NSUInteger)pairs)
             threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    }
    [enc endEncoding];

    if (down_sum) return 1;

    q36_moe_reduce_args ra = { out_dim, used, tokens, down->has_scales };
    enc = q36_encoder(@"q36_moe_reduce_f32");
    if (!enc) return 0;
    [enc setBuffer:out->buffer offset:out->offset atIndex:0];
    [enc setBuffer:q36_moe_down_scratch offset:0 atIndex:1];
    [enc setBuffer:selected->buffer offset:selected->offset atIndex:2];
    [enc setBuffer:dsb offset:(NSUInteger)dso atIndex:3]; [enc setBytes:&ra length:sizeof(ra) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(out_dim, tokens, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];
    return 1;
}

static int q36_moe_host(
        q36_gpu_tensor *out, const void *map, uint64_t size,
        const q36_gpu_moe_weight *gate, const q36_gpu_moe_weight *up,
        const q36_gpu_moe_weight *down,
        const q36_gpu_tensor *selected, const q36_gpu_tensor *route_weights,
        uint32_t used, const float *x, uint32_t tokens,
        uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim,
        uint32_t experts) {
    if (!q36_quant_dequantize || !out || !map || !gate || !up || !down ||
        !selected || !route_weights || !x || !used || !tokens ||
        !in_dim || !mid_dim || !out_dim || !experts) return 0;
    uint64_t grb = q36_quant_row_bytes(gate->type, in_dim);
    uint64_t urb = q36_quant_row_bytes(up->type, in_dim);
    uint64_t drb = q36_quant_row_bytes(down->type, mid_dim);
    if (!grb || !urb || !drb ||
        gate->offset > size || grb > (size - gate->offset) / mid_dim / experts ||
        up->offset > size || urb > (size - up->offset) / mid_dim / experts ||
        down->offset > size || drb > (size - down->offset) / out_dim / experts)
        return 0;
    const int32_t *ids =
        q36_gpu_tensor_contents((q36_gpu_tensor *)selected);
    const float *rw =
        q36_gpu_tensor_contents((q36_gpu_tensor *)route_weights);
    float *dst = q36_gpu_tensor_contents(out);
    float *grow = malloc((size_t)in_dim * sizeof(float));
    float *urow = malloc((size_t)in_dim * sizeof(float));
    float *drow = malloc((size_t)mid_dim * sizeof(float));
    float *mid = malloc((size_t)mid_dim * sizeof(float));
    if (!ids || !rw || !dst || !grow || !urow || !drow || !mid) {
        free(mid); free(drow); free(urow); free(grow);
        return 0;
    }
    memset(dst, 0, (size_t)tokens * out_dim * sizeof(float));
    const unsigned char *base = map;
    const float *gsc = gate->has_scales
        ? (const float *)(base + gate->scales_offset) : NULL;
    const float *usc = up->has_scales
        ? (const float *)(base + up->scales_offset) : NULL;
    const float *dsc = down->has_scales
        ? (const float *)(base + down->scales_offset) : NULL;
    for (uint32_t tok = 0; tok < tokens; tok++) {
        const float *xt = x + (uint64_t)tok * in_dim;
        for (uint32_t slot = 0; slot < used; slot++) {
            int32_t expert = ids[(uint64_t)tok * used + slot];
            if (expert < 0 || (uint32_t)expert >= experts) continue;
            float gs = gsc ? gsc[expert] : 1.0f;
            float us = usc ? usc[expert] : 1.0f;
            float ds = dsc ? dsc[expert] : 1.0f;
            const unsigned char *gw = base + gate->offset +
                (uint64_t)expert * mid_dim * grb;
            const unsigned char *uw = base + up->offset +
                (uint64_t)expert * mid_dim * urb;
            for (uint32_t j = 0; j < mid_dim; j++) {
                if (!q36_quant_dequantize(gate->type, gw + (uint64_t)j * grb,
                                          grow, in_dim) ||
                    !q36_quant_dequantize(up->type, uw + (uint64_t)j * urb,
                                          urow, in_dim)) {
                    free(mid); free(drow); free(urow); free(grow);
                    return 0;
                }
                double ga = 0.0, ua = 0.0;
                for (uint32_t i = 0; i < in_dim; i++) {
                    ga += (double)grow[i] * xt[i];
                    ua += (double)urow[i] * xt[i];
                }
                float g = (float)ga * gs;
                mid[j] = (g / (1.0f + expf(-g))) * ((float)ua * us);
            }
            const unsigned char *dw = base + down->offset +
                (uint64_t)expert * out_dim * drb;
            float route = rw[(uint64_t)tok * used + slot];
            for (uint32_t j = 0; j < out_dim; j++) {
                if (!q36_quant_dequantize(down->type, dw + (uint64_t)j * drb,
                                          drow, mid_dim)) {
                    free(mid); free(drow); free(urow); free(grow);
                    return 0;
                }
                double acc = 0.0;
                for (uint32_t i = 0; i < mid_dim; i++)
                    acc += (double)drow[i] * mid[i];
                uint64_t at = (uint64_t)tok * out_dim + j;
                dst[at] = fmaf((float)acc * ds, route, dst[at]);
            }
        }
    }
    free(mid); free(drow); free(urow); free(grow);
    return 1;
}

int q36_gpu_moe_ffn_f32_tensor(
        q36_gpu_tensor *out, const void *map, uint64_t size,
        const q36_gpu_moe_weight *gate, const q36_gpu_moe_weight *up,
        const q36_gpu_moe_weight *down,
        const q36_gpu_tensor *selected, const q36_gpu_tensor *weights,
        uint32_t layer, uint32_t used, const q36_gpu_tensor *x,
        uint32_t tokens, uint32_t in_dim, uint32_t mid_dim,
        uint32_t out_dim, uint32_t experts) {
    const char *gpu_moe = getenv("Q36_METAL_MOE_GPU");
    const char *gpu_moe_required = getenv("Q36_METAL_MOE_GPU_REQUIRED");
    const char *force_mv = getenv("Q36_VK_MOE_GEMM");
    if ((!gpu_moe || gpu_moe[0] != '0') &&
        (!force_mv || force_mv[0] != '0') &&
        q36_moe_gpu(out, map, size, gate, up, down, selected, weights,
                    used, x, tokens, in_dim, mid_dim, out_dim, experts,
                    layer))
        return 1;
    if (gpu_moe_required && gpu_moe_required[0] != '0') return 0;
    const float *xp = q36_gpu_tensor_contents((q36_gpu_tensor *)x);
    return xp && q36_moe_host(out, map, size, gate, up, down,
                              selected, weights, used, xp, tokens,
                              in_dim, mid_dim, out_dim, experts);
}

int q36_gpu_moe_ffn_q8_tensor(
        q36_gpu_tensor *out, const void *map, uint64_t size,
        const q36_gpu_moe_weight *gate, const q36_gpu_moe_weight *up,
        const q36_gpu_moe_weight *down,
        const q36_gpu_tensor *selected, const q36_gpu_tensor *weights,
        uint32_t layer, uint32_t used, const q36_gpu_tensor *q8,
        uint32_t tokens, uint32_t in_dim, uint32_t mid_dim,
        uint32_t out_dim, uint32_t experts) {
    (void)layer;
    if (!q8 || !in_dim || (in_dim & 255u)) return 0;
    const q36_metal_q8_k *blocks =
        q36_gpu_tensor_contents((q36_gpu_tensor *)q8);
    /*
     * K-quant routed FFNs deliberately keep the Q8_K boundary between both
     * matrix products.  Besides matching the Vulkan ABI, this avoids turning
     * the integer dot into a subtly different dequantized-float operation.
     */
    if (blocks && gate && up && down && gate->type >= 10u && gate->type <= 14u &&
        up->type >= 10u && up->type <= 14u &&
        down->type >= 10u && down->type <= 14u &&
        !(mid_dim & 255u) && q36_quant_q8_k) {
        const int32_t *ids = q36_gpu_tensor_contents((q36_gpu_tensor *)selected);
        const float *rw = q36_gpu_tensor_contents((q36_gpu_tensor *)weights);
        float *dst = q36_gpu_tensor_contents(out);
        uint64_t grb = q36_kquant_row_bytes(gate->type, in_dim);
        uint64_t urb = q36_kquant_row_bytes(up->type, in_dim);
        uint64_t drb = q36_kquant_row_bytes(down->type, mid_dim);
        uint64_t in_blocks = in_dim / 256u, mid_blocks = mid_dim / 256u;
        float *mid = malloc((size_t)mid_dim * sizeof(float));
        q36_metal_q8_k *midq = calloc((size_t)mid_blocks, sizeof(*midq));
        q36_metal_q8_k_packed *tmp =
            malloc((size_t)mid_blocks * sizeof(*tmp));
        if (!ids || !rw || !dst || !grb || !urb || !drb ||
            !mid || !midq || !tmp) {
            free(tmp); free(midq); free(mid); return 0;
        }
        memset(dst, 0, (size_t)tokens * out_dim * sizeof(float));
        const uint8_t *base = map;
        const float *gsc = gate->has_scales
            ? (const float *)(base + gate->scales_offset) : NULL;
        const float *usc = up->has_scales
            ? (const float *)(base + up->scales_offset) : NULL;
        const float *dsc = down->has_scales
            ? (const float *)(base + down->scales_offset) : NULL;
        for (uint32_t tok = 0; tok < tokens; tok++) {
            const q36_metal_q8_k *xq = blocks + (uint64_t)tok * in_blocks;
            for (uint32_t slot = 0; slot < used; slot++) {
                int32_t e = ids[(uint64_t)tok * used + slot];
                if (e < 0 || (uint32_t)e >= experts) continue;
                const uint8_t *gw = base + gate->offset +
                    (uint64_t)e * mid_dim * grb;
                const uint8_t *uw = base + up->offset +
                    (uint64_t)e * mid_dim * urb;
                for (uint32_t j = 0; j < mid_dim; j++) {
                    float ga = 0.0f, ua = 0.0f;
                    for (uint64_t b = 0; b < in_blocks; b++) {
                        ga += q36_kquant_dot_block(
                            gate->type, gw + (uint64_t)j*grb +
                            b*(grb/in_blocks), xq+b);
                        ua += q36_kquant_dot_block(
                            up->type, uw + (uint64_t)j*urb +
                            b*(urb/in_blocks), xq+b);
                    }
                    ga *= gsc ? gsc[e] : 1.0f;
                    ua *= usc ? usc[e] : 1.0f;
                    mid[j] = (ga / (1.0f + expf(-ga))) * ua;
                }
                q36_quant_q8_k(mid, tmp, mid_dim);
                for (uint64_t b = 0; b < mid_blocks; b++) {
                    midq[b].d = tmp[b].d;
                    memcpy(midq[b].qs, tmp[b].qs, 256u);
                    memcpy(midq[b].bsums, tmp[b].bsums, 32u);
                }
                const uint8_t *dw = base + down->offset +
                    (uint64_t)e * out_dim * drb;
                float route = rw[(uint64_t)tok * used + slot];
                float ds = dsc ? dsc[e] : 1.0f;
                for (uint32_t j = 0; j < out_dim; j++) {
                    float acc = 0.0f;
                    for (uint64_t b = 0; b < mid_blocks; b++)
                        acc += q36_kquant_dot_block(
                            down->type, dw + (uint64_t)j*drb +
                            b*(drb/mid_blocks), midq+b);
                    uint64_t at = (uint64_t)tok*out_dim+j;
                    dst[at] = fmaf(acc*ds, route, dst[at]);
                }
            }
        }
        free(tmp); free(midq); free(mid);
        return 1;
    }
    float *x = malloc((size_t)tokens * in_dim * sizeof(float));
    if (!blocks || !x) { free(x); return 0; }
    uint32_t nb = in_dim / 256u;
    for (uint32_t tok = 0; tok < tokens; tok++)
        for (uint32_t b = 0; b < nb; b++)
            for (uint32_t i = 0; i < 256u; i++)
                x[(uint64_t)tok * in_dim + (uint64_t)b * 256u + i] =
                    blocks[(uint64_t)tok * nb + b].d *
                    blocks[(uint64_t)tok * nb + b].qs[i];
    int ok = q36_moe_host(out, map, size, gate, up, down,
                           selected, weights, used, x, tokens,
                           in_dim, mid_dim, out_dim, experts);
    free(x);
    return ok;
}

int q36_gpu_moe_ffn_tensor(
        q36_gpu_tensor *out, const void *map, uint64_t size,
        const q36_gpu_moe_weight *gate, const q36_gpu_moe_weight *up,
        const q36_gpu_moe_weight *down,
        const q36_gpu_tensor *selected, const q36_gpu_tensor *weights,
        uint32_t layer, uint32_t used, const q36_gpu_tensor *x,
        uint32_t tokens, uint32_t in_dim, uint32_t mid_dim,
        uint32_t out_dim, uint32_t experts) {
    return q36_gpu_moe_ffn_f32_tensor(
        out, map, size, gate, up, down, selected, weights,
        layer, used, x, tokens, in_dim, mid_dim, out_dim, experts);
}
