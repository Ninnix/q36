#include "q36_gpu.h"
#include "q36_iq2_tables_vulkan.inc"
#include "q36_quant.h"
#include "q36_ssd.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <vulkan/vulkan.h>

enum {
    Q36_VK_QK8_0 = 32,
    Q36_VK_TENSOR_F32     = 0,
    Q36_VK_TENSOR_F16     = 1,
    Q36_VK_TENSOR_Q2_K    = 10,
    Q36_VK_TENSOR_Q4_K    = 12,
    Q36_VK_TENSOR_Q5_K    = 13,
    Q36_VK_TENSOR_Q6_K    = 14,
    Q36_VK_TENSOR_IQ2_XXS = 16,
    Q36_VK_TENSOR_IQ3_S   = 21,
    Q36_VK_TENSOR_IQ2_S   = 22,
};

typedef struct q36_gpu_tensor {
    VkBuffer buffer;
    VkDeviceMemory memory;
    unsigned char *data;
    uint64_t bytes;
    uint64_t alloc_bytes;
    uint64_t offset;
    struct q36_gpu_tensor *owner;
    bool device_local;
    bool weight_only;           /* arena-backed weights: uploaded once, never
                                 * written by dispatches, skipped by the
                                 * batch hazard tracker */
    bool gpu_written;           /* some recorded command wrote this root */
    uint64_t last_use_seq;      /* submit seq of the last recorded GPU use
                                 * of this root; 0 = never referenced */
} q36_gpu_tensor;

typedef struct q36_vk_kernel {
    const char *path;
    uint32_t bindings;
    uint32_t push_bytes;
    uint32_t write_mask;        /* bit per binding the shader writes */
    VkShaderModule shader;
    VkDescriptorSetLayout set_layout;
    VkPipelineLayout layout;
    VkPipeline pipeline;
    bool ready;
} q36_vk_kernel;

#define Q36_VK_KERNEL(path_, bindings_, push_bytes_, write_mask_) \
    ((q36_vk_kernel){ \
        .path = (path_), \
        .bindings = (bindings_), \
        .push_bytes = (push_bytes_), \
        .write_mask = (write_mask_), \
    })

typedef struct q36_vk_weight {
    const void *source;
    uint64_t bytes;
    q36_gpu_tensor *tensor;
    struct q36_vk_weight *next;
} q36_vk_weight;

typedef struct q36_vk_packed_weight {
    const void *source;
    uint64_t src_bytes;
    uint32_t type;
    uint32_t in_dim;
    uint32_t out_dim;
    q36_gpu_tensor *tensor;
    struct q36_vk_packed_weight *next;
} q36_vk_packed_weight;

/* Weight arena: cached weights suballocate from large blocks instead of
 * one VkDeviceMemory each.  Every device memory is an amdgpu buffer object
 * the kernel validates on EVERY queue submit, and with ~700 per-weight
 * allocations that validation cost ~2 ms per submit -- the single largest
 * share of a forward pass on this stack.  A handful of arena blocks makes
 * submits cheap again.  Weights live until the model map changes, so a bump
 * allocator with whole-arena teardown is enough. */
typedef struct q36_vk_arena_block {
    q36_gpu_tensor *base;
    uint64_t used;
    struct q36_vk_arena_block *next;
} q36_vk_arena_block;

typedef struct {
    VkInstance instance;
    VkPhysicalDevice physical;
    VkDevice device;
    VkQueue queue;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkFence fence;
    VkQueryPool query_pool;
    VkDescriptorPool descriptor_pool;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties mem_props;
    uint32_t queue_family;
    uint32_t timestamp_valid_bits;
    uint32_t api_version;
    bool ready;
    bool prof_ops;
    bool prof_kernel;
    bool have_f16;

    q36_vk_kernel matmul_f16;
    q36_vk_kernel matmul_f32;
    q36_vk_kernel matmul_f32_fast;
    q36_vk_kernel add;
    q36_vk_kernel directional_steering;
    q36_vk_kernel rms_norm;
    q36_vk_kernel add_rms_norm;
    q36_vk_kernel swiglu;
    q36_vk_kernel rope;
    q36_vk_kernel rms_norm_rope;
    q36_vk_kernel rms_norm_rope_kv;
    q36_vk_kernel copy_rows;
    q36_vk_kernel recur_window;
    q36_vk_kernel conv_silu;
    q36_vk_kernel recur_conv_silu_decode;
    q36_vk_kernel delta_qk;
    q36_vk_kernel delta_qkv;
    q36_vk_kernel delta_gates;
    q36_vk_kernel quantize_q8_0;
    q36_vk_kernel quantize_q8_k;
    q36_vk_kernel matmul_q8_0;
    q36_vk_kernel matmul_q8_0_q36;
    q36_vk_kernel matmul_q8_0_f32b;
    q36_vk_kernel shared_gate_up_decode;
    q36_vk_kernel shared_down_tail_decode;
    q36_vk_kernel matmul_q8_0_mm;
    q36_vk_kernel matmul_q8_0_mm_f16;
    q36_vk_kernel matmul_q8_0_mm_f16_out32;
    q36_vk_kernel matmul_q8_0_f32b_nx;
    q36_vk_kernel matmul_q8_0_decode;
    q36_vk_kernel matmul_q8_0_decode_q36;
    q36_vk_kernel matmul_q8_0_decode_b64;
    q36_vk_kernel matmul_kquant;
    q36_vk_kernel matmul_q5k_mmq;
    q36_vk_kernel matmul_q6k_mmq;
    q36_vk_kernel matmul_q5k_mmq_fast;
    q36_vk_kernel matmul_q6k_mmq_fast;
    q36_vk_kernel delta_net;
    q36_vk_kernel delta_net_fast;
    q36_vk_kernel delta_net_cols;
    q36_vk_kernel delta_net_decode;
    q36_vk_kernel attn_scores;
    q36_vk_kernel attn_post;
    q36_vk_kernel attn_reduce;
    q36_vk_kernel attn_decode_fused;
    q36_vk_kernel attn_decode_split;
    q36_vk_kernel attn_prefill_qtile;
    q36_vk_kernel attn_combine;
    q36_vk_kernel moe_gate_up;
    q36_vk_kernel router_topk;
    q36_vk_kernel moe_tiles;
    q36_vk_kernel kv_store;
    q36_vk_kernel kv_store_quant;
    q36_vk_kernel rms_norm_rope_kv_quant;
    q36_vk_kernel moe_gate_up_f32b;
    q36_vk_kernel moe_down_q2k_f32b;
    q36_vk_kernel moe_down_q2k_sum_decode;
    q36_vk_kernel moe_gate_up_gemm;
    q36_vk_kernel moe_down_gemm;
    q36_vk_kernel moe_matvec;
    q36_vk_kernel moe_matvec_fast;
    q36_vk_kernel moe_reduce;
    q36_vk_kernel ffn_tail;
    q36_vk_weight *weights;
    q36_vk_packed_weight *packed_weights;
    q36_vk_arena_block *arena;
    q36_gpu_tensor *iq_tables;
    q36_gpu_tensor *attn_part;
    uint64_t attn_part_bytes;
} q36_vulkan_runtime;

enum {
    Q36_VK_QK_K = 256,
    Q36_VK_MATMUL_TOK_TILE = 4,
    Q36_VK_MOE_PAIR_TILE = 8,
    Q36_VK_MOE_TILE_WORDS = 1 + Q36_VK_MOE_PAIR_TILE,
    Q36_VK_MOE_GEMM_TILE = 16,
    Q36_VK_N_HEAD = 16,
    Q36_VK_N_HEAD_DIM = 256,
    Q36_VK_N_SSM_CONV = 4,
    Q36_VK_N_SSM_STATE = 128,
    Q36_VK_N_SSM_GROUP = 16,
    Q36_VK_N_SSM_QK = Q36_VK_N_SSM_GROUP * Q36_VK_N_SSM_STATE,
    Q36_VK_N_SSM_CONV_DIM = Q36_VK_N_SSM_QK * 2u + 4096u,
};

typedef struct {
    uint16_t d;
    int8_t qs[Q36_VK_QK8_0];
} q36_vk_block_q8_0;

typedef char q36_vk_block_q8_0_size[(sizeof(q36_vk_block_q8_0) == 34) ? 1 : -1];

typedef struct {
    float d;
    float dmin;
    int8_t qs[Q36_VK_QK_K];
    int16_t bsums[Q36_VK_QK_K / 16];
} q36_vk_block_q8_K;

typedef char q36_vk_block_q8_k_size[(sizeof(q36_vk_block_q8_K) == 296) ? 1 : -1];

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[Q36_VK_QK_K / 8];
    uint8_t qs[Q36_VK_QK_K / 2];
} q36_vk_block_q5_K_src;

typedef struct {
    uint8_t ql[Q36_VK_QK_K / 2];
    uint8_t qh[Q36_VK_QK_K / 4];
    int8_t scales[Q36_VK_QK_K / 16];
    uint16_t d;
} q36_vk_block_q6_K_src;

typedef struct {
    uint16_t d[8];
    uint16_t dmin[8];
    uint8_t scales[64];
    uint8_t mins[64];
    uint8_t qh[(Q36_VK_QK_K / 8) * 8];
    uint8_t qs[(Q36_VK_QK_K / 2) * 8];
} q36_vk_block_q5_Kx8;

typedef char q36_vk_block_q5_kx8_size[(sizeof(q36_vk_block_q5_Kx8) == 1440) ? 1 : -1];

typedef struct {
    uint16_t d[8];
    int8_t scales[(Q36_VK_QK_K / 16) * 8];
    uint8_t ql[(Q36_VK_QK_K / 2) * 8];
    uint8_t qh[(Q36_VK_QK_K / 4) * 8];
} q36_vk_block_q6_Kx8;

typedef char q36_vk_block_q6_kx8_size[(sizeof(q36_vk_block_q6_Kx8) == 1680) ? 1 : -1];

static q36_vulkan_runtime q36_vk;
static pthread_mutex_t q36_vk_mu = PTHREAD_MUTEX_INITIALIZER;
static const void *q36_gpu_model_map;
static uint64_t q36_gpu_model_size;
static int q36_gpu_model_fd = -1;
static bool q36_gpu_quality;
static bool q36_gpu_commands_open;
static uint64_t q36_gpu_live_bytes;
static uint64_t q36_gpu_peak_bytes;

/* Routed-expert banks are cached on device like any other weight, which
 * removes the per-token expert staging memcpy.  If the heaps cannot hold
 * them the runtime falls back to per-call packing (same numerics, slower)
 * and never retries, so a full memory cannot turn into a per-call
 * vkAllocateMemory failure loop. */
static bool q36_vk_moe_bank_cache = true;

enum {
    Q36_VK_STREAM_MAX_LAYERS = 40,
    Q36_VK_STREAM_MAX_EXPERTS = 256,
};

typedef struct {
    bool valid;
    uint32_t layer;
    uint32_t expert;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t last_used;
    bool loading;               /* reserved this prepare pass, data in flight */
} q36_vk_stream_entry;

typedef struct {
    bool enabled;
    uint32_t budget;
    uint32_t full_layers;
    uint32_t runtime_cap;
    uint32_t allocation_failures;
    bool indexed;
    uint64_t configured_expert_bytes;
    uint32_t cap;
    uint64_t clock;
    uint64_t hits;
    uint64_t misses;
    uint64_t loads;
    uint64_t evictions;
    uint64_t lookup_steps;
    uint64_t gate_expert_bytes;
    uint64_t up_expert_bytes;
    uint64_t down_expert_bytes;
    uint32_t gate_type;
    uint32_t up_type;
    uint32_t down_type;
    bool has_gate_scales;
    bool has_up_scales;
    bool has_down_scales;
    q36_gpu_tensor *gate;
    q36_gpu_tensor *up;
    q36_gpu_tensor *down;
    q36_gpu_tensor *gate_scales;
    q36_gpu_tensor *up_scales;
    q36_gpu_tensor *down_scales;
    q36_vk_stream_entry *entry;
    uint32_t slot_of[Q36_VK_STREAM_MAX_LAYERS][Q36_VK_STREAM_MAX_EXPERTS];
    uint32_t route_hotness[Q36_VK_STREAM_MAX_LAYERS][Q36_VK_STREAM_MAX_EXPERTS];
    uint64_t route_tokens;
    uint64_t hotness_decay_token;
} q36_vk_stream_cache;

static q36_vk_stream_cache q36_vk_stream;

/* Free-list of retired scratch tensors.  Decode allocates the same handful
 * of small buffers every layer (q8 activations, expert mids, attention aux);
 * reusing them skips a vkAllocateMemory/vkFreeMemory round-trip per buffer.
 * Reused tensors are re-zeroed so semantics match a fresh allocation. */
enum {
    Q36_VK_POOL_CAP = 256,
    Q36_VK_POOL_MAX_BYTES = 1u << 20,
    Q36_VK_MAX_THREADS = 32,
};
static q36_gpu_tensor *q36_vk_pool[Q36_VK_POOL_CAP];
static uint32_t q36_vk_pool_n;

typedef struct {
    pthread_mutex_t mu;
    pthread_mutex_t submit_mu;
    pthread_cond_t work_cv;
    pthread_cond_t done_cv;
    pthread_t threads[Q36_VK_MAX_THREADS];
    uint32_t n_threads;
    uint32_t n_workers;
    uint32_t generation;
    uint32_t done;
    bool initialized;
    bool shutdown;
    q36_gpu_parallel_fn fn;
    void *ctx;
    uint64_t n_rows;
} q36_vk_host_pool;

static q36_vk_host_pool q36_vk_host;
static pthread_mutex_t q36_vk_host_init_mu = PTHREAD_MUTEX_INITIALIZER;
static __thread int q36_vk_parallel_depth;

/* Lazy submission, the scheme ds4's Metal runtime uses: dispatches are
 * recorded into one open command buffer with full compute barriers between
 * them, and the batch is submitted and fence-waited only when the host
 * actually reads a tensor.  Every host-side op fetches its pointers through
 * q36_gpu_tensor_contents(), so the flush point is automatic and the
 * host-exact parity ops keep working unchanged.  Tensors freed while a
 * batch is open are parked on the retire list until the fence signals,
 * since recorded commands still reference their memory. */
enum {
    Q36_VK_BATCH_FLUSH_AT = 32,
    Q36_VK_RETIRE_CAP = 1024,
};
static bool q36_vk_batch_recording;
static uint32_t q36_vk_batch_count;
static q36_gpu_tensor *q36_vk_retired[Q36_VK_RETIRE_CAP];
static uint32_t q36_vk_retired_n;

/* Submission sequence: every vkQueueSubmit gets the next seq, roots are
 * stamped with the seq of the batch that last references them, and a
 * fence wait advances completed_seq (one queue, so fences signal in
 * submission order).  A root whose last_use_seq is <= completed_seq has
 * no pending GPU references and can retire without draining anything. */
static uint64_t q36_vk_submit_seq;
static uint64_t q36_vk_completed_seq;

/* Q36_VK_PROF=1 wall-clock accounting: where a forward pass actually waits.
 * submit_ns covers vkQueueSubmit to fence signal (GPU batch execution as
 * seen by the host), record_ns the per-dispatch descriptor + command
 * recording.  Printed by q36_gpu_print_memory_report(). */
static uint64_t q36_vk_prof_dispatches;
static uint64_t q36_vk_prof_flushes;
static uint64_t q36_vk_prof_submit_ns;
static uint64_t q36_vk_prof_record_ns;

enum {
    Q36_VK_PROF_QUERY_CAP = 2048,
    Q36_VK_PROF_KERNEL_CAP = 64,
    Q36_VK_PROF_OP_CAP = 128,
};

typedef struct {
    const char *path;
    uint64_t dispatches;
    uint64_t groups;
    uint64_t gpu_ns;
} q36_vk_prof_kernel_row;

typedef struct {
    const char *name;
    uint64_t dispatches;
    uint64_t flushes;
    uint64_t groups;
    uint64_t record_ns;
    uint64_t submit_ns;
    uint64_t gpu_ns;
    uint64_t wall_ns;
} q36_vk_prof_op_row;

typedef struct {
    const char *name;
    uint64_t t0;
    bool active;
} q36_vk_prof_scope;

static const char *q36_vk_prof_query_path[Q36_VK_PROF_QUERY_CAP / 2];
static const char *q36_vk_prof_query_op[Q36_VK_PROF_QUERY_CAP / 2];
static uint64_t q36_vk_prof_query_groups[Q36_VK_PROF_QUERY_CAP / 2];
static uint32_t q36_vk_prof_query_count;
static q36_vk_prof_kernel_row q36_vk_prof_kernel_rows[Q36_VK_PROF_KERNEL_CAP];
static uint32_t q36_vk_prof_kernel_n;
static q36_vk_prof_op_row q36_vk_prof_op_rows[Q36_VK_PROF_OP_CAP];
static uint32_t q36_vk_prof_op_n;

enum {
    Q36_VK_STREAM_PROFILE_LAYERS = 40,
    Q36_VK_STREAM_PROFILE_EXPERTS = 256,
};

typedef struct {
    bool active;
    bool reported;
    const char *path;
    uint64_t hist[Q36_VK_STREAM_PROFILE_LAYERS][Q36_VK_STREAM_PROFILE_EXPERTS];
    uint64_t total;
} q36_vk_stream_profile;

typedef struct {
    uint32_t layer;
    uint32_t expert;
    uint64_t count;
} q36_vk_stream_profile_entry;

static q36_vk_stream_profile q36_vk_stream_prof;

static uint64_t q36_vk_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void q36_vk_prof_report(void);
static void q36_vk_stream_profile_report(void);

static void q36_vk_prof_reset(void) {
    q36_vk_prof_dispatches = 0;
    q36_vk_prof_flushes = 0;
    q36_vk_prof_submit_ns = 0;
    q36_vk_prof_record_ns = 0;
    q36_vk_prof_query_count = 0;
    q36_vk_prof_kernel_n = 0;
    q36_vk_prof_op_n = 0;
}

static int q36_vk_stream_profile_cmp(const void *a, const void *b) {
    const q36_vk_stream_profile_entry *ea = a;
    const q36_vk_stream_profile_entry *eb = b;
    if (ea->count < eb->count) return 1;
    if (ea->count > eb->count) return -1;
    if (ea->layer != eb->layer) return ea->layer < eb->layer ? -1 : 1;
    if (ea->expert != eb->expert) return ea->expert < eb->expert ? -1 : 1;
    return 0;
}

static void q36_vk_stream_profile_init(void) {
    const char *path = getenv("Q36_VK_STREAMING_EXPERT_PROFILE");
    const char *hotlist = getenv("Q36_VK_STREAMING_EXPERT_HOTLIST_OUT");
    if (hotlist && hotlist[0]) path = hotlist;
    if (!path || !path[0] || q36_vk_stream_prof.active) return;
    memset(&q36_vk_stream_prof, 0, sizeof(q36_vk_stream_prof));
    q36_vk_stream_prof.active = true;
    q36_vk_stream_prof.path = path;
    atexit(q36_vk_stream_profile_report);
    fprintf(stderr, "q36: Vulkan streaming expert profiler active: %s\n", path);
}

static void q36_vk_stream_profile_record_unlocked(const q36_gpu_stream_expert_table *t,
                                                  const uint32_t *selected,
                                                  uint64_t n_selected) {
    if (!q36_vk_stream_prof.active || !t || !selected) return;
    if (t->layer >= Q36_VK_STREAM_PROFILE_LAYERS) return;
    for (uint64_t i = 0; i < n_selected; i++) {
        uint32_t expert = selected[i];
        if (expert >= Q36_VK_STREAM_PROFILE_EXPERTS) continue;
        q36_vk_stream_prof.hist[t->layer][expert]++;
        q36_vk_stream_prof.total++;
    }
}

static void q36_vk_stream_profile_report(void) {
    q36_vk_stream_profile *p = &q36_vk_stream_prof;
    q36_vk_stream_profile_entry entries[Q36_VK_STREAM_PROFILE_LAYERS *
                                        Q36_VK_STREAM_PROFILE_EXPERTS];
    FILE *fp;
    uint32_t n = 0;
    if (!p->active || p->reported || !p->path || !p->path[0]) return;
    p->reported = true;
    for (uint32_t il = 0; il < Q36_VK_STREAM_PROFILE_LAYERS; il++) {
        for (uint32_t expert = 0; expert < Q36_VK_STREAM_PROFILE_EXPERTS; expert++) {
            entries[n++] = (q36_vk_stream_profile_entry) {
                .layer = il,
                .expert = expert,
                .count = p->hist[il][expert],
            };
        }
    }
    qsort(entries, n, sizeof(entries[0]), q36_vk_stream_profile_cmp);
    fp = fopen(p->path, "wb");
    if (!fp) {
        fprintf(stderr, "q36: failed to write streaming expert profile %s: %s\n",
                p->path, strerror(errno));
        return;
    }
    fprintf(fp,
            "# q36 expert hotlist v1\n"
            "# layers %u\n"
            "# experts %u\n"
            "# selections %" PRIu64 "\n"
            "# columns: layer expert hits weight\n",
            Q36_VK_STREAM_PROFILE_LAYERS,
            Q36_VK_STREAM_PROFILE_EXPERTS,
            p->total);
    for (uint32_t i = 0; i < n; i++) {
        fprintf(fp, "%u %u %" PRIu64 " 0\n",
                entries[i].layer,
                entries[i].expert,
                entries[i].count);
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "q36: failed to close streaming expert profile %s: %s\n",
                p->path, strerror(errno));
    } else {
        fprintf(stderr, "q36: wrote Vulkan streaming expert profile %s (%" PRIu64 " selections)\n",
                p->path, p->total);
    }
}

static const char *q36_vk_prof_op_name(const char *name) {
    return (name && name[0]) ? name : "other";
}

static q36_vk_prof_op_row *q36_vk_prof_op_get(const char *name) {
    name = q36_vk_prof_op_name(name);
    for (uint32_t i = 0; i < q36_vk_prof_op_n; i++) {
        if (q36_vk_prof_op_rows[i].name == name || !strcmp(q36_vk_prof_op_rows[i].name, name)) {
            return &q36_vk_prof_op_rows[i];
        }
    }
    if (q36_vk_prof_op_n == Q36_VK_PROF_OP_CAP) return NULL;
    memset(&q36_vk_prof_op_rows[q36_vk_prof_op_n], 0, sizeof(q36_vk_prof_op_rows[q36_vk_prof_op_n]));
    q36_vk_prof_op_rows[q36_vk_prof_op_n].name = name;
    return &q36_vk_prof_op_rows[q36_vk_prof_op_n++];
}

static void q36_vk_prof_op_add(const char *name,
                               uint64_t dispatches,
                               uint64_t flushes,
                               uint64_t groups,
                               uint64_t record_ns,
                               uint64_t submit_ns,
                               uint64_t gpu_ns,
                               uint64_t wall_ns) {
    if (!q36_vk.prof_ops) return;
    q36_vk_prof_op_row *r = q36_vk_prof_op_get(name);
    if (!r) return;
    r->dispatches += dispatches;
    r->flushes += flushes;
    r->groups += groups;
    r->record_ns += record_ns;
    r->submit_ns += submit_ns;
    r->gpu_ns += gpu_ns;
    r->wall_ns += wall_ns;
}

static void q36_vk_prof_scope_begin(q36_vk_prof_scope *s, const char *name) {
    if (!s) return;
    s->active = q36_vk.prof_ops;
    s->name = q36_vk_prof_op_name(name);
    s->t0 = s->active ? q36_vk_now_ns() : 0;
}

static void q36_vk_prof_scope_end(q36_vk_prof_scope *s) {
    if (!s || !s->active) return;
    q36_vk_prof_op_add(s->name, 0, 0, 0, 0, 0, 0, q36_vk_now_ns() - s->t0);
    s->active = false;
}

static void q36_vk_prof_kernel_add(const char *path, uint64_t groups, uint64_t gpu_ns) {
    for (uint32_t i = 0; i < q36_vk_prof_kernel_n; i++) {
        if (q36_vk_prof_kernel_rows[i].path == path || !strcmp(q36_vk_prof_kernel_rows[i].path, path)) {
            q36_vk_prof_kernel_rows[i].dispatches++;
            q36_vk_prof_kernel_rows[i].groups += groups;
            q36_vk_prof_kernel_rows[i].gpu_ns += gpu_ns;
            return;
        }
    }
    if (q36_vk_prof_kernel_n == Q36_VK_PROF_KERNEL_CAP) return;
    q36_vk_prof_kernel_rows[q36_vk_prof_kernel_n++] = (q36_vk_prof_kernel_row){
        .path = path,
        .dispatches = 1,
        .groups = groups,
        .gpu_ns = gpu_ns,
    };
}

static void q36_vk_prof_kernel_collect_unlocked(void) {
    if (!q36_vk.prof_kernel || !q36_vk.query_pool || q36_vk_prof_query_count == 0) return;

    uint64_t ticks[Q36_VK_PROF_QUERY_CAP];
    VkResult rc = vkGetQueryPoolResults(q36_vk.device,
                                        q36_vk.query_pool,
                                        0,
                                        q36_vk_prof_query_count,
                                        sizeof(ticks[0]) * q36_vk_prof_query_count,
                                        ticks,
                                        sizeof(ticks[0]),
                                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (rc != VK_SUCCESS) {
        q36_vk_prof_query_count = 0;
        return;
    }

    uint64_t mask = UINT64_MAX;
    if (q36_vk.timestamp_valid_bits > 0 && q36_vk.timestamp_valid_bits < 64) {
        mask = (1ull << q36_vk.timestamp_valid_bits) - 1ull;
    }
    for (uint32_t q = 0; q + 1 < q36_vk_prof_query_count; q += 2) {
        uint64_t a = ticks[q] & mask;
        uint64_t b = ticks[q + 1u] & mask;
        uint64_t dt = (b >= a) ? (b - a) : ((mask - a) + b + 1u);
        uint64_t ns = (uint64_t)((double)dt * (double)q36_vk.props.limits.timestampPeriod);
        q36_vk_prof_kernel_add(q36_vk_prof_query_path[q / 2u], q36_vk_prof_query_groups[q / 2u], ns);
        q36_vk_prof_op_add(q36_vk_prof_query_op[q / 2u], 0, 0, 0, 0, 0, ns, 0);
    }
    q36_vk_prof_query_count = 0;
}

static void *q36_vk_host_worker_main(void *arg) {
    const uint32_t tid = (uint32_t)(uintptr_t)arg;
    uint32_t seen_generation = 0;

    for (;;) {
        pthread_mutex_lock(&q36_vk_host.mu);
        while (seen_generation == q36_vk_host.generation && !q36_vk_host.shutdown) {
            pthread_cond_wait(&q36_vk_host.work_cv, &q36_vk_host.mu);
        }
        if (q36_vk_host.shutdown) {
            pthread_mutex_unlock(&q36_vk_host.mu);
            return NULL;
        }

        seen_generation = q36_vk_host.generation;
        q36_gpu_parallel_fn fn = q36_vk_host.fn;
        void *ctx = q36_vk_host.ctx;
        uint64_t n_rows = q36_vk_host.n_rows;
        uint32_t n_threads = q36_vk_host.n_threads;
        pthread_mutex_unlock(&q36_vk_host.mu);

        {
            uint64_t rows_per_thread = (n_rows + n_threads - 1u) / n_threads;
            uint64_t row0 = (uint64_t)tid * rows_per_thread;
            uint64_t row1 = row0 + rows_per_thread;
            if (row1 > n_rows) row1 = n_rows;
            if (row0 < row1) {
                q36_vk_parallel_depth++;
                fn(ctx, row0, row1);
                q36_vk_parallel_depth--;
            }
        }

        pthread_mutex_lock(&q36_vk_host.mu);
        q36_vk_host.done++;
        if (q36_vk_host.done == q36_vk_host.n_workers) pthread_cond_signal(&q36_vk_host.done_cv);
        pthread_mutex_unlock(&q36_vk_host.mu);
    }
}

static void q36_vk_host_threads_init(void) {
    pthread_mutex_lock(&q36_vk_host_init_mu);
    if (q36_vk_host.initialized) {
        pthread_mutex_unlock(&q36_vk_host_init_mu);
        return;
    }

    uint32_t n_threads = 12;
    long online_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    const char *env = getenv("Q36_THREADS");
    if (online_cpus > 0) n_threads = online_cpus < 12 ? (uint32_t)online_cpus : 12;
    if (env && env[0]) {
        long v = strtol(env, NULL, 10);
        if (v > 0) n_threads = (uint32_t)v;
    }
    if (n_threads > Q36_VK_MAX_THREADS) n_threads = Q36_VK_MAX_THREADS;
    if (n_threads == 0) n_threads = 1;

    pthread_mutex_init(&q36_vk_host.mu, NULL);
    pthread_mutex_init(&q36_vk_host.submit_mu, NULL);
    pthread_cond_init(&q36_vk_host.work_cv, NULL);
    pthread_cond_init(&q36_vk_host.done_cv, NULL);
    q36_vk_host.n_threads = 1;
    q36_vk_host.n_workers = 0;
    q36_vk_host.generation = 0;
    q36_vk_host.done = 0;
    q36_vk_host.shutdown = false;
    q36_vk_host.initialized = true;

    for (uint32_t i = 1; i < n_threads; i++) {
        if (pthread_create(&q36_vk_host.threads[i], NULL,
                           q36_vk_host_worker_main, (void *)(uintptr_t)i) != 0) {
            fprintf(stderr, "q36: Vulkan host pool started %u/%u threads\n", i, n_threads);
            break;
        }
        q36_vk_host.n_threads = i + 1u;
        q36_vk_host.n_workers = i;
    }
    pthread_mutex_unlock(&q36_vk_host_init_mu);
}

static void q36_vk_host_threads_shutdown(void) {
    pthread_mutex_lock(&q36_vk_host_init_mu);
    if (!q36_vk_host.initialized) {
        pthread_mutex_unlock(&q36_vk_host_init_mu);
        return;
    }

    pthread_mutex_lock(&q36_vk_host.submit_mu);
    pthread_mutex_lock(&q36_vk_host.mu);
    q36_vk_host.shutdown = true;
    q36_vk_host.generation++;
    pthread_cond_broadcast(&q36_vk_host.work_cv);
    pthread_mutex_unlock(&q36_vk_host.mu);
    for (uint32_t i = 1; i < q36_vk_host.n_threads; i++) pthread_join(q36_vk_host.threads[i], NULL);
    pthread_mutex_unlock(&q36_vk_host.submit_mu);

    pthread_cond_destroy(&q36_vk_host.done_cv);
    pthread_cond_destroy(&q36_vk_host.work_cv);
    pthread_mutex_destroy(&q36_vk_host.submit_mu);
    pthread_mutex_destroy(&q36_vk_host.mu);
    memset(&q36_vk_host, 0, sizeof(q36_vk_host));
    pthread_mutex_unlock(&q36_vk_host_init_mu);
}

void q36_gpu_parallel_for_rows(uint64_t n_rows,
                               uint64_t min_parallel_rows,
                               q36_gpu_parallel_fn fn,
                               void *ctx) {
    uint64_t rows_per_thread;
    uint64_t main_row1;

    if (!fn || n_rows == 0) return;
    q36_vk_host_threads_init();
    if (q36_vk_parallel_depth > 0 || q36_vk_host.n_threads <= 1 || n_rows < min_parallel_rows) {
        fn(ctx, 0, n_rows);
        return;
    }

    pthread_mutex_lock(&q36_vk_host.submit_mu);
    pthread_mutex_lock(&q36_vk_host.mu);
    q36_vk_host.fn = fn;
    q36_vk_host.ctx = ctx;
    q36_vk_host.n_rows = n_rows;
    q36_vk_host.done = 0;
    q36_vk_host.generation++;
    pthread_cond_broadcast(&q36_vk_host.work_cv);
    rows_per_thread = (n_rows + q36_vk_host.n_threads - 1u) / q36_vk_host.n_threads;
    main_row1 = rows_per_thread > n_rows ? n_rows : rows_per_thread;
    pthread_mutex_unlock(&q36_vk_host.mu);

    if (main_row1 > 0) {
        q36_vk_parallel_depth++;
        fn(ctx, 0, main_row1);
        q36_vk_parallel_depth--;
    }

    pthread_mutex_lock(&q36_vk_host.mu);
    while (q36_vk_host.done < q36_vk_host.n_workers) {
        pthread_cond_wait(&q36_vk_host.done_cv, &q36_vk_host.mu);
    }
    pthread_mutex_unlock(&q36_vk_host.mu);
    pthread_mutex_unlock(&q36_vk_host.submit_mu);
}

/* Root tensors written by the open batch.  q36_gpu_tensor_contents() only
 * needs to submit when the host touches one of these (or the set
 * overflowed); reads of host-produced tensors (router selection, q8
 * staging, swiglu outputs) keep recording.  This is safe because the
 * schedulers never host-write a tensor that recorded-but-unsubmitted
 * commands read: activations flow forward, so a host write always precedes
 * the recording of its GPU consumers. */
/* A 512-token prefill chunk writes more distinct roots than this, so
 * batch_dirty_all trips once per chunk.  That used to drain the pipeline
 * on the next chunk's embed-stage map (~190ms/chunk); now contents()
 * consults the per-root gpu_written flag, so host-only tensors skip the
 * flush even under dirty_all, and scratch memory is bounded by the
 * seq-based retirement sweep on each eager submit rather than by the
 * accidental per-chunk drain. */
enum { Q36_VK_WRITTEN_CAP = 64 };
static q36_gpu_tensor *q36_vk_batch_written[Q36_VK_WRITTEN_CAP];
static uint32_t q36_vk_batch_written_n;
static bool q36_vk_batch_dirty_all;

/* Hazard sets since the last pipeline barrier in the open batch.  A barrier
 * is emitted only when a dispatch touches a tensor written since then, or
 * writes one that was read (RAW/WAW/WAR); independent dispatches overlap on
 * the GPU instead of draining the pipeline ~1300 times per decoded token.
 * Weight-arena roots never enter the sets: they are uploaded from the host
 * before use and no shader writes them. */
enum { Q36_VK_HAZARD_CAP = 48 };
static q36_gpu_tensor *q36_vk_hazard_written[Q36_VK_HAZARD_CAP];
static q36_gpu_tensor *q36_vk_hazard_read[Q36_VK_HAZARD_CAP];
static uint32_t q36_vk_hazard_written_n;
static uint32_t q36_vk_hazard_read_n;
static bool q36_vk_hazard_all;

static bool q36_vk_hazard_in(q36_gpu_tensor **set, uint32_t n, const q36_gpu_tensor *root) {
    for (uint32_t i = 0; i < n; i++) {
        if (set[i] == root) return true;
    }
    return false;
}

static void q36_vk_hazard_add(q36_gpu_tensor **set, uint32_t *n, q36_gpu_tensor *root) {
    if (root->weight_only || q36_vk_hazard_in(set, *n, root)) return;
    if (*n == Q36_VK_HAZARD_CAP) q36_vk_hazard_all = true;
    else set[(*n)++] = root;
}

static void q36_vk_hazard_reset(void) {
    q36_vk_hazard_written_n = 0;
    q36_vk_hazard_read_n = 0;
    q36_vk_hazard_all = false;
}

/* GPU work is outstanding while a batch records or an eager submit's fence
 * has not been waited on. Frees and host reads must respect both. */
static bool q36_vk_gpu_busy_unlocked(void);
static const char *q36_vk_flush_reason = "submit_wait_other";
static __thread const char *q36_vk_tensor_contents_reason = "submit_wait_tensor_contents";

static int q36_vk_flush_reason_unlocked(const char *reason);
static void *q36_vk_tensor_contents_labeled(q36_gpu_tensor *tensor, const char *reason);
static int q36_vk_begin_batch_unlocked(VkCommandBuffer *cmd_out);
static void q36_vk_note_written_tensor(q36_gpu_tensor *tensor);

static uint64_t q36_round_up_u64(uint64_t n, uint64_t align) {
    return (n + align - 1u) & ~(align - 1u);
}

static bool q36_u64_mul_ok(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > UINT64_MAX / a) return false;
    *out = a * b;
    return true;
}

static bool q36_gpu_tensor_range_ok(const q36_gpu_tensor *tensor, uint64_t offset, uint64_t bytes);
static const unsigned char *q36_gpu_weight_bytes(const void *model_map,
                                                  uint64_t model_size,
                                                  uint64_t weight_offset,
                                                  uint64_t bytes);
static q36_gpu_tensor *q36_vk_quantize_q8_k_alloc_tensor(const q36_gpu_tensor *x,
                                                         uint64_t in_dim,
                                                         uint64_t n_tok);
static int q36_vk_run_unlocked(
        const char                 *op,
        q36_vk_kernel              *kernel,
        const q36_gpu_tensor *const *bindings,
        const void                 *push,
        uint32_t                    push_bytes,
        uint32_t                    groups_x,
        uint32_t                    groups_y,
        uint32_t                    groups_z);

typedef struct {
    float d;
    int8_t qs[Q36_VK_QK_K];
    int16_t bsums[Q36_VK_QK_K / 16];
} q36_vk_block_q8_K_packed;

enum { Q36_MAX_VK_Q8_BLOCKS = 32 };

typedef struct {
    q36_vk_block_q8_K *dst;
    const float *x;
    uint64_t in_dim;
    uint64_t blocks;
} q36_vk_quantize_q8_k_ctx;

static void q36_vk_quantize_q8_k_rows(void *opaque, uint64_t row0, uint64_t row1) {
    q36_vk_quantize_q8_k_ctx *ctx = (q36_vk_quantize_q8_k_ctx *)opaque;
    q36_vk_block_q8_K_packed tmp[Q36_MAX_VK_Q8_BLOCKS];

    for (uint64_t t = row0; t < row1; t++) {
        q36_vk_block_q8_K *qt = ctx->dst + t * ctx->blocks;
        q36_quant_q8_k(ctx->x + t * ctx->in_dim, tmp, (int64_t)ctx->in_dim);
        for (uint64_t b = 0; b < ctx->blocks; b++) {
            qt[b].d = tmp[b].d;
            qt[b].dmin = 0.0f;
            memcpy(qt[b].qs, tmp[b].qs, sizeof(qt[b].qs));
            memcpy(qt[b].bsums, tmp[b].bsums, sizeof(qt[b].bsums));
        }
    }
}

static void q36_vk_quantize_q8_k_host(q36_vk_block_q8_K *dst,
                                      const float *x,
                                      uint64_t n_tok,
                                      uint64_t in_dim) {
    const uint64_t blocks = in_dim / Q36_VK_QK_K;
    q36_vk_quantize_q8_k_ctx ctx;

    if (blocks > Q36_MAX_VK_Q8_BLOCKS) abort();
    ctx.dst = dst;
    ctx.x = x;
    ctx.in_dim = in_dim;
    ctx.blocks = blocks;
    if (n_tok * in_dim >= 8192u) q36_gpu_parallel_for_rows(n_tok, 4, q36_vk_quantize_q8_k_rows, &ctx);
    else q36_vk_quantize_q8_k_rows(&ctx, 0, n_tok);
}

static void q36_vk_q5_scale_min(const uint8_t *src, uint32_t j, uint8_t *scale, uint8_t *mn) {
    const uint32_t s0 = src[j];
    if (j < 4u) {
        *scale = (uint8_t)(s0 & 63u);
        *mn = (uint8_t)(src[4u + j] & 63u);
    } else {
        const uint32_t sj4 = src[j + 4u];
        const uint32_t sjm4 = src[j - 4u];
        *scale = (uint8_t)((sj4 & 15u) | ((sjm4 >> 6) << 4));
        *mn = (uint8_t)((sj4 >> 4) | ((s0 >> 6) << 4));
    }
}

static uint64_t q36_vk_kquant_mmq_block_bytes(uint32_t type) {
    switch (type) {
    case Q36_VK_TENSOR_Q5_K: return sizeof(q36_vk_block_q5_Kx8);
    case Q36_VK_TENSOR_Q6_K: return sizeof(q36_vk_block_q6_Kx8);
    default: return 0;
    }
}

static bool q36_vk_use_mmq(void) {
    const char *env = getenv("Q36_VK_MMQ");
    return env && env[0] && env[0] != '0';
}

static bool q36_vk_use_mmq_fast(void) {
    const char *env = getenv("Q36_VK_MMQ_FAST");
    return env && env[0] && env[0] != '0';
}

static bool q36_vk_env_default_on(const char *name) {
    const char *env = getenv(name);
    return !env || !env[0] || env[0] != '0';
}

static bool q36_vk_use_gpu_attn_post(void) {
    return q36_vk_env_default_on("Q36_VK_GPU_ATTN_POST");
}

/* Prefill batches at or above this token count route routed-MoE work
 * through the GEMM kernels; below it the matvec path keeps
 * decode-identical numerics for chunk tails and speculative windows. */
static uint32_t q36_vk_moe_gemm_min(void) {
    const char *env = getenv("Q36_VK_MOE_GEMM_MIN");
    long v = env ? strtol(env, NULL, 10) : 0;
    return v > 0 ? (uint32_t)v : 128u;
}

static bool q36_vk_use_moe_down_sum_decode(void) {
    return q36_vk_env_default_on("Q36_VK_MOE_DOWN_SUM_DECODE");
}

/* Sampled once: the runtime skips the scores scratch allocation when the
 * fused attention path is on, so the choice must not change mid-process. */
static bool q36_vk_use_attn_qtile(void) {
    static int cached = -1;
    if (cached < 0) cached = q36_vk_env_default_on("Q36_VK_ATTN_QTILE") ? 1 : 0;
    return cached != 0;
}

static bool q36_vk_use_attn_fused(void) {
    static int cached = -1;
    if (cached < 0) cached = q36_vk_env_default_on("Q36_VK_ATTN_FUSED") ? 1 : 0;
    return cached != 0;
}

bool q36_gpu_attn_fused_enabled(void) {
    return !q36_gpu_quality && q36_vk_use_attn_fused();
}

static bool q36_vk_use_gpu_rms(void) {
    return q36_vk_env_default_on("Q36_VK_GPU_RMS");
}

static bool q36_vk_use_add_rms(void) {
    return q36_vk_env_default_on("Q36_VK_ADD_RMS");
}

static bool q36_vk_use_gpu_recurrent(void) {
    return q36_vk_env_default_on("Q36_VK_GPU_RECURRENT");
}

static bool q36_vk_use_delta_fast_prefill(void) {
    return q36_vk_env_default_on("Q36_VK_DELTA_FAST_PREFILL");
}

static bool q36_vk_use_delta_col_prefill(void) {
    return q36_vk_env_default_on("Q36_VK_DELTA_COL_PREFILL");
}

static bool q36_vk_use_delta_decode(void) {
    return q36_vk_env_default_on("Q36_VK_DELTA_DECODE");
}

static bool q36_vk_use_attn_splitk(void) {
    return q36_vk_env_default_on("Q36_VK_ATTN_SPLITK");
}

static uint32_t q36_vk_kv_cache_row_bytes(uint32_t type, uint32_t n) {
    if (type == 0u) return n * (uint32_t)sizeof(uint16_t);
    if (type == 1u) return ((n + Q36_VK_QK8_0 - 1u) / Q36_VK_QK8_0) * 34u;
    if (type == 2u) return ((n + Q36_VK_QK8_0 - 1u) / Q36_VK_QK8_0) * 18u;
    return 0;
}

static bool q36_vk_use_f32_fast(void) {
    return q36_vk_env_default_on("Q36_VK_F32_FAST");
}

static bool q36_vk_use_gpu_ffn_tail(void) {
    return q36_vk_env_default_on("Q36_VK_GPU_FFN_TAIL");
}

static bool q36_vk_use_gpu_swiglu(void) {
    return q36_vk_env_default_on("Q36_VK_GPU_SWIGLU");
}

static bool q36_vk_use_moe_fast(void) {
    return q36_vk_env_default_on("Q36_VK_MOE_FAST");
}

static bool q36_vk_use_moe_gate_up(void) {
    return q36_vk_env_default_on("Q36_VK_MOE_GATE_UP");
}

static bool q36_vk_use_router_fast_prefill(void) {
    const char *env = getenv("Q36_VK_ROUTER_FAST_PREFILL");
    return env && env[0] && env[0] != '0';
}

static bool q36_vk_use_q8_0_gpu(void) {
    return q36_vk_env_default_on("Q36_VK_Q8_0_GPU");
}

static bool q36_vk_use_q8_0_f32b(void) {
    return q36_vk_env_default_on("Q36_VK_Q8_0_F32B");
}

static int q36_vk_quantize_q8_k_dispatch(q36_gpu_tensor *q8,
                                         const q36_gpu_tensor *x,
                                         uint64_t in_dim,
                                         uint64_t n_tok) {
    uint64_t x_bytes = 0;
    uint64_t q8_bytes = 0;
    const uint64_t blocks = in_dim / Q36_VK_QK_K;

    if (in_dim == 0 || (in_dim % Q36_VK_QK_K) != 0) return 0;
    if (in_dim > UINT32_MAX || n_tok > UINT32_MAX || blocks > UINT32_MAX) return 0;
    if (!q36_u64_mul_ok(n_tok, in_dim, &x_bytes) ||
        !q36_u64_mul_ok(x_bytes, sizeof(float), &x_bytes)) {
        return 0;
    }
    if (!q36_u64_mul_ok(n_tok, blocks, &q8_bytes) ||
        !q36_u64_mul_ok(q8_bytes, sizeof(q36_vk_block_q8_K), &q8_bytes)) {
        return 0;
    }
    if (!q36_gpu_tensor_range_ok(x, 0, x_bytes) ||
        !q36_gpu_tensor_range_ok(q8, 0, q8_bytes)) {
        return 0;
    }

    struct {
        uint32_t in_dim;
        uint32_t blocks;
        uint32_t n_tok;
    } push = { (uint32_t)in_dim, (uint32_t)blocks, (uint32_t)n_tok };
    const q36_gpu_tensor *bindings[2] = { x, q8 };

    pthread_mutex_lock(&q36_vk_mu);
    int ok = q36_vk_run_unlocked("q8_k_quant",
                                 &q36_vk.quantize_q8_k,
                                 bindings,
                                 &push,
                                 sizeof(push),
                                 (uint32_t)blocks,
                                 (uint32_t)n_tok,
                                 1);
    pthread_mutex_unlock(&q36_vk_mu);
    return ok;
}

int q36_gpu_quantize_q8_k_tensor(q36_gpu_tensor *out,
                                 const q36_gpu_tensor *x,
                                 uint64_t in_dim,
                                 uint64_t n_tok) {
    uint64_t blocks = in_dim / Q36_VK_QK_K;
    uint64_t x_bytes = 0;
    uint64_t q8_count = 0;
    uint64_t q8_bytes = 0;

    if (in_dim == 0 || (in_dim % Q36_VK_QK_K) != 0 || n_tok == 0) return 0;
    if (!q36_u64_mul_ok(n_tok, in_dim, &x_bytes) ||
        !q36_u64_mul_ok(x_bytes, sizeof(float), &x_bytes) ||
        !q36_u64_mul_ok(n_tok, blocks, &q8_count) ||
        !q36_u64_mul_ok(q8_count, sizeof(q36_vk_block_q8_K), &q8_bytes)) {
        return 0;
    }
    if (!q36_gpu_tensor_range_ok(x, 0, x_bytes) || !q36_gpu_tensor_range_ok(out, 0, q8_bytes)) return 0;

    if (!q36_gpu_quality && in_dim == 512u && n_tok == 8u && q36_vk_env_default_on("Q36_VK_Q8_K_MOE_GPU") &&
        q36_vk_quantize_q8_k_dispatch(out, x, in_dim, n_tok)) {
        return 1;
    }
    /* Decode/MTP-verify activations quantize on GPU too: the host quantizer
     * here was the largest remaining decode flush cause after GPU RMS. */
    if (!q36_gpu_quality && n_tok <= 8u && q36_vk_env_default_on("Q36_VK_Q8_K_MOE_GPU") &&
        q36_vk_quantize_q8_k_dispatch(out, x, in_dim, n_tok)) {
        return 1;
    }
    if (!q36_gpu_quality && n_tok > 8u && q36_vk_env_default_on("Q36_VK_Q8_GPU") &&
        q36_vk_quantize_q8_k_dispatch(out, x, in_dim, n_tok)) {
        return 1;
    }

    const float *x_host = q36_vk_tensor_contents_labeled((q36_gpu_tensor *)x, "submit_wait_q8k_quant_x");
    q36_vk_block_q8_K *q8_host = q36_vk_tensor_contents_labeled(out, "submit_wait_q8k_quant_out");
    if (!x_host || !q8_host) return 0;
    q36_vk_quantize_q8_k_host(q8_host, x_host, n_tok, in_dim);
    return 1;
}


static float q36_vk_sigmoid_stable(float x) {
    return 1.0f / (1.0f + expf(-x));
}




static q36_gpu_tensor *q36_gpu_tensor_root(const q36_gpu_tensor *tensor, uint64_t *offset) {
    if (!tensor) return NULL;
    if (!tensor->owner) {
        if (offset) *offset = tensor->offset;
        return (q36_gpu_tensor *)tensor;
    }
    uint64_t root_offset = 0;
    q36_gpu_tensor *root = q36_gpu_tensor_root(tensor->owner, &root_offset);
    if (offset) *offset = root_offset + tensor->offset;
    return root;
}

static bool q36_gpu_tensor_range_ok(const q36_gpu_tensor *tensor, uint64_t offset, uint64_t bytes) {
    return tensor && offset <= tensor->bytes && bytes <= tensor->bytes - offset;
}

static const unsigned char *q36_gpu_weight_bytes(const void *model_map,
                                                  uint64_t model_size,
                                                  uint64_t weight_offset,
                                                  uint64_t bytes) {
    const unsigned char *map = model_map ? (const unsigned char *)model_map :
                                           (const unsigned char *)q36_gpu_model_map;
    const uint64_t size = model_map ? model_size : q36_gpu_model_size;
    if (!map) return NULL;
    if (weight_offset > size || bytes > size - weight_offset) return NULL;
    return map + weight_offset;
}

static uint32_t q36_vk_find_memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < q36_vk.mem_props.memoryTypeCount; i++) {
        if ((bits & (1u << i)) == 0) continue;
        if ((q36_vk.mem_props.memoryTypes[i].propertyFlags & want) == want) return i;
    }
    return UINT32_MAX;
}

static void q36_vk_kernel_destroy(q36_vk_kernel *k) {
    if (!q36_vk.device || !k) return;
    if (k->pipeline) vkDestroyPipeline(q36_vk.device, k->pipeline, NULL);
    if (k->layout) vkDestroyPipelineLayout(q36_vk.device, k->layout, NULL);
    if (k->set_layout) vkDestroyDescriptorSetLayout(q36_vk.device, k->set_layout, NULL);
    if (k->shader) vkDestroyShaderModule(q36_vk.device, k->shader, NULL);
    k->pipeline = VK_NULL_HANDLE;
    k->layout = VK_NULL_HANDLE;
    k->set_layout = VK_NULL_HANDLE;
    k->shader = VK_NULL_HANDLE;
    k->ready = false;
}

static void *q36_read_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len <= 0 || (len & 3) != 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    void *buf = malloc((size_t)len);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    if (fread(buf, 1, (size_t)len, fp) != (size_t)len) {
        free(buf);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    *len_out = (size_t)len;
    return buf;
}

static int q36_vk_kernel_init(q36_vk_kernel *k) {
    if (k->ready) return 1;

    size_t code_size = 0;
    void *code = q36_read_file(k->path, &code_size);
    if (!code) {
        fprintf(stderr, "q36: failed to read Vulkan shader %s\n", k->path);
        return 0;
    }

    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code_size,
        .pCode = code,
    };
    VkResult rc = vkCreateShaderModule(q36_vk.device, &smci, NULL, &k->shader);
    free(code);
    if (rc != VK_SUCCESS) {
        fprintf(stderr, "q36: vkCreateShaderModule failed for %s (%d)\n", k->path, rc);
        return 0;
    }

    VkDescriptorSetLayoutBinding bindings[8];
    if (k->bindings > 8) return 0;
    for (uint32_t i = 0; i < k->bindings; i++) {
        bindings[i] = (VkDescriptorSetLayoutBinding){
            .binding = i,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        };
    }
    VkDescriptorSetLayoutCreateInfo dsci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = k->bindings,
        .pBindings = bindings,
    };
    rc = vkCreateDescriptorSetLayout(q36_vk.device, &dsci, NULL, &k->set_layout);
    if (rc != VK_SUCCESS) {
        fprintf(stderr, "q36: vkCreateDescriptorSetLayout failed for %s (%d)\n", k->path, rc);
        q36_vk_kernel_destroy(k);
        return 0;
    }

    VkPushConstantRange push = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = k->push_bytes,
    };
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &k->set_layout,
        .pushConstantRangeCount = k->push_bytes ? 1u : 0u,
        .pPushConstantRanges = k->push_bytes ? &push : NULL,
    };
    rc = vkCreatePipelineLayout(q36_vk.device, &plci, NULL, &k->layout);
    if (rc != VK_SUCCESS) {
        fprintf(stderr, "q36: vkCreatePipelineLayout failed for %s (%d)\n", k->path, rc);
        q36_vk_kernel_destroy(k);
        return 0;
    }

    VkPipelineShaderStageCreateInfo stage = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = k->shader,
        .pName = "main",
    };
    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = stage,
        .layout = k->layout,
    };
    rc = vkCreateComputePipelines(q36_vk.device, VK_NULL_HANDLE, 1, &cpci, NULL, &k->pipeline);
    if (rc != VK_SUCCESS) {
        fprintf(stderr, "q36: vkCreateComputePipelines failed for %s (%d)\n", k->path, rc);
        q36_vk_kernel_destroy(k);
        return 0;
    }

    k->ready = true;
    return 1;
}

static q36_gpu_tensor *q36_vk_tensor_alloc_kind_unlocked(uint64_t bytes, bool device_pref) {
    q36_gpu_tensor *tensor = calloc(1, sizeof(*tensor));
    if (!tensor) return NULL;
    tensor->bytes = bytes;
    tensor->alloc_bytes = bytes ? q36_round_up_u64(bytes, 4) : 4;

    VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = tensor->alloc_bytes,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (vkCreateBuffer(q36_vk.device, &bci, NULL, &tensor->buffer) != VK_SUCCESS) {
        free(tensor);
        return NULL;
    }

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(q36_vk.device, tensor->buffer, &req);
    /* Prefer cached (snooped) host-visible memory: on the BC-250 the
     * write-combined GTT type is not reliably coherent for device writes
     * followed by host reads, and every host op reads GPU output.
     * Weights are the exception (device_pref): written once by memcpy and
     * only ever read by shaders, so they go to the device-local heap, which
     * is several times larger than the host-cached one and must hold the
     * routed expert banks. */
    uint32_t mt = UINT32_MAX;
    if (device_pref) {
        mt = q36_vk_find_memory_type(req.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        tensor->device_local = mt != UINT32_MAX;
    }
    if (mt == UINT32_MAX) {
        mt = q36_vk_find_memory_type(req.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                     VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    }
    if (mt == UINT32_MAX) {
        mt = q36_vk_find_memory_type(req.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    if (mt == UINT32_MAX) {
        vkDestroyBuffer(q36_vk.device, tensor->buffer, NULL);
        free(tensor);
        return NULL;
    }
    VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = mt,
    };
    if (vkAllocateMemory(q36_vk.device, &mai, NULL, &tensor->memory) != VK_SUCCESS) {
        vkDestroyBuffer(q36_vk.device, tensor->buffer, NULL);
        free(tensor);
        return NULL;
    }
    if (vkBindBufferMemory(q36_vk.device, tensor->buffer, tensor->memory, 0) != VK_SUCCESS ||
        vkMapMemory(q36_vk.device, tensor->memory, 0, req.size, 0, (void **)&tensor->data) != VK_SUCCESS)
    {
        vkFreeMemory(q36_vk.device, tensor->memory, NULL);
        vkDestroyBuffer(q36_vk.device, tensor->buffer, NULL);
        free(tensor);
        return NULL;
    }
    memset(tensor->data, 0, (size_t)tensor->alloc_bytes);

    q36_gpu_live_bytes += bytes;
    if (q36_gpu_live_bytes > q36_gpu_peak_bytes) q36_gpu_peak_bytes = q36_gpu_live_bytes;
    return tensor;
}

static void q36_vk_tensor_free_unlocked(q36_gpu_tensor *tensor) {
    if (!tensor) return;
    if (!tensor->owner) {
        q36_gpu_live_bytes -= tensor->bytes;
        if (tensor->memory) vkUnmapMemory(q36_vk.device, tensor->memory);
        if (tensor->memory) vkFreeMemory(q36_vk.device, tensor->memory, NULL);
        if (tensor->buffer) vkDestroyBuffer(q36_vk.device, tensor->buffer, NULL);
    }
    free(tensor);
}

/* Pool-or-free for a tensor with no pending GPU references. A full pool
 * evicts round-robin instead of rejecting: otherwise a prefill chunk fills
 * every slot with chunk-sized scratch and decode-sized buffers can never
 * enter, turning each decode step into fresh vkAllocateMemory calls. */
static uint32_t q36_vk_pool_rr;
static void q36_vk_tensor_retire_unlocked(q36_gpu_tensor *tensor) {
    if (!tensor->owner && !tensor->device_local && tensor->data &&
        tensor->alloc_bytes <= Q36_VK_POOL_MAX_BYTES) {
        q36_gpu_live_bytes -= tensor->bytes;
        if (q36_vk_pool_n < Q36_VK_POOL_CAP) {
            q36_vk_pool[q36_vk_pool_n++] = tensor;
        } else {
            /* Pooled entries already left the live-byte count at retire. */
            q36_vk_pool[q36_vk_pool_rr]->bytes = 0;
            q36_vk_tensor_free_unlocked(q36_vk_pool[q36_vk_pool_rr]);
            q36_vk_pool[q36_vk_pool_rr] = tensor;
            q36_vk_pool_rr = (q36_vk_pool_rr + 1u) % Q36_VK_POOL_CAP;
        }
    } else {
        q36_vk_tensor_free_unlocked(tensor);
    }
}

/* Retire now if the tensor's last GPU reference has completed, else park
 * it on the retired queue for the next sweep. */
static void q36_vk_tensor_release_unlocked(q36_gpu_tensor *tensor) {
    if (!tensor) return;
    if (tensor->last_use_seq > q36_vk_completed_seq) {
        if (q36_vk_retired_n == Q36_VK_RETIRE_CAP) q36_vk_flush_reason_unlocked("submit_wait_retire_cap");
        if (tensor->last_use_seq > q36_vk_completed_seq) {
            q36_vk_retired[q36_vk_retired_n++] = tensor;
            return;
        }
    }
    q36_vk_tensor_retire_unlocked(tensor);
}

static void q36_vk_stream_cache_release_unlocked(bool keep_config) {
    bool enabled = q36_vk_stream.enabled;
    uint32_t budget = q36_vk_stream.budget;
    uint32_t full_layers = q36_vk_stream.full_layers;
    uint32_t runtime_cap = q36_vk_stream.runtime_cap;
    uint32_t allocation_failures = q36_vk_stream.allocation_failures;
    uint64_t expert_bytes = q36_vk_stream.configured_expert_bytes;

    q36_vk_tensor_free_unlocked(q36_vk_stream.gate);
    q36_vk_tensor_free_unlocked(q36_vk_stream.up);
    q36_vk_tensor_free_unlocked(q36_vk_stream.down);
    q36_vk_tensor_free_unlocked(q36_vk_stream.gate_scales);
    q36_vk_tensor_free_unlocked(q36_vk_stream.up_scales);
    q36_vk_tensor_free_unlocked(q36_vk_stream.down_scales);
    free(q36_vk_stream.entry);
    memset(&q36_vk_stream, 0, sizeof(q36_vk_stream));
    if (keep_config) {
        q36_vk_stream.enabled = enabled;
        q36_vk_stream.budget = budget;
        q36_vk_stream.full_layers = full_layers;
        q36_vk_stream.runtime_cap = runtime_cap;
        q36_vk_stream.allocation_failures = allocation_failures;
        q36_vk_stream.configured_expert_bytes = expert_bytes;
    }
}

static bool q36_vk_stream_copy_model_range(const void *model_map,
                                           uint64_t model_size,
                                           uint64_t offset,
                                           void *dst,
                                           uint64_t bytes) {
    if (bytes == 0) return true;
    if (!dst || bytes > (uint64_t)SIZE_MAX) return false;
    if (q36_gpu_model_fd >= 0) {
        uint64_t done = 0;
        while (done < bytes) {
            size_t n = (size_t)(bytes - done);
            if (n > (size_t)(256u << 20)) n = (size_t)(256u << 20);
            ssize_t got = pread(q36_gpu_model_fd,
                                (unsigned char *)dst + done,
                                n,
                                (off_t)(offset + done));
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) break;
            done += (uint64_t)got;
        }
        if (done == bytes) return true;
    }

    const unsigned char *src = q36_gpu_weight_bytes(model_map, model_size, offset, bytes);
    if (!src) return false;
    memcpy(dst, src, (size_t)bytes);
    return true;
}

/* Submit the open batch, wait for it, then settle everything that was
 * deferred while it was recording: retired tensors and the descriptor
 * pool (all its sets belong to the finished batch). */
/* Command-buffer ring: the batch cap submits eagerly WITHOUT waiting, so
 * the GPU chews on earlier work while the host records ahead; only a host
 * read waits for everything. Hazard sets survive eager submits because the
 * pending work still executes in submission order. */
enum { Q36_VK_CB_RING = 4 };
typedef struct {
    VkCommandBuffer cb;
    VkFence fence;
    VkDescriptorPool pool;
    bool pending;
    uint64_t seq;
} q36_vk_cb_slot;
static q36_vk_cb_slot q36_vk_ring[Q36_VK_CB_RING];
static uint32_t q36_vk_ring_cur;

static bool q36_vk_gpu_busy_unlocked(void) {
    if (q36_vk_batch_recording) return true;
    for (uint32_t i = 0; i < Q36_VK_CB_RING; i++) {
        if (q36_vk_ring[i].pending) return true;
    }
    return false;
}

static int q36_vk_slot_wait_unlocked(q36_vk_cb_slot *s) {
    VkResult rc = VK_SUCCESS;
    if (!s->pending) return 1;
    /* Decode batches finish in tens of microseconds; spinning on the fence
     * status dodges the syscall + thread wakeup of a blocking wait. */
    for (uint32_t spin = 0; spin < 4096; spin++) {
        rc = vkGetFenceStatus(q36_vk.device, s->fence);
        if (rc != VK_NOT_READY) break;
    }
    if (rc == VK_NOT_READY) rc = vkWaitForFences(q36_vk.device, 1, &s->fence, VK_TRUE, UINT64_MAX);
    vkResetFences(q36_vk.device, 1, &s->fence);
    if (s->pool) vkResetDescriptorPool(q36_vk.device, s->pool, 0);
    s->pending = false;
    if (rc == VK_SUCCESS && s->seq > q36_vk_completed_seq) q36_vk_completed_seq = s->seq;
    return rc == VK_SUCCESS;
}

static int q36_vk_submit_current_unlocked(bool host_read) {
    q36_vk_cb_slot *s = &q36_vk_ring[q36_vk_ring_cur];
    if (!q36_vk_batch_recording) return 1;
    if (host_read) {
        /* Host ops read these buffers right after the fence.  The fence's
         * implicit host visibility is not reliable on the BC-250's RADV
         * (intermittent stale reads), so flush shader writes to host
         * explicitly. */
        VkMemoryBarrier host_barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
        };
        vkCmdPipelineBarrier(q36_vk.command_buffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &host_barrier, 0, NULL, 0, NULL);
    }
    if (vkEndCommandBuffer(q36_vk.command_buffer) != VK_SUCCESS) return 0;
    {
        VkSubmitInfo si = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &q36_vk.command_buffer,
        };
        if (vkQueueSubmit(q36_vk.queue, 1, &si, s->fence) != VK_SUCCESS) return 0;
    }
    s->pending = true;
    s->seq = ++q36_vk_submit_seq;
    q36_vk_batch_recording = false;
    q36_vk_batch_count = 0;
    return 1;
}

static void q36_vk_ring_activate_unlocked(uint32_t idx) {
    q36_vk_ring_cur = idx;
    q36_vk.command_buffer = q36_vk_ring[idx].cb;
    q36_vk.fence = q36_vk_ring[idx].fence;
    q36_vk.descriptor_pool = q36_vk_ring[idx].pool;
}

/* Retire every parked tensor whose last GPU reference has completed.
 * Runs on each eager-submit fence wait so scratch memory drains as the
 * ring advances instead of piling up until the next full flush. */
static void q36_vk_retired_sweep_unlocked(void) {
    uint32_t kept = 0;
    for (uint32_t i = 0; i < q36_vk_retired_n; i++) {
        q36_gpu_tensor *t = q36_vk_retired[i];
        if (t->last_use_seq > q36_vk_completed_seq) q36_vk_retired[kept++] = t;
        else q36_vk_tensor_retire_unlocked(t);
    }
    q36_vk_retired_n = kept;
}

static int q36_vk_submit_eager_unlocked(void) {
    uint64_t t0 = q36_vk_now_ns();
    int ok = q36_vk_submit_current_unlocked(false);
    if (ok) {
        uint32_t next = (q36_vk_ring_cur + 1u) % Q36_VK_CB_RING;
        ok = q36_vk_slot_wait_unlocked(&q36_vk_ring[next]);
        q36_vk_ring_activate_unlocked(next);
        q36_vk_retired_sweep_unlocked();
    }
    q36_vk_prof_op_add("submit_eager", 0, 1, 0, 0, q36_vk_now_ns() - t0, 0, 0);
    return ok;
}

static int q36_vk_flush_unlocked(void) {
    bool had_work = q36_vk_batch_recording;
    int ok = q36_vk_submit_current_unlocked(true);
    uint64_t t0 = q36_vk_now_ns();
    for (uint32_t i = 0; i < Q36_VK_CB_RING; i++) {
        had_work = had_work || q36_vk_ring[i].pending;
        if (!q36_vk_slot_wait_unlocked(&q36_vk_ring[i])) ok = 0;
    }
    if (had_work) {
        q36_vk_prof_flushes++;
        {
            uint64_t submit_ns = q36_vk_now_ns() - t0;
            q36_vk_prof_submit_ns += submit_ns;
            q36_vk_prof_op_add(q36_vk_flush_reason, 0, 1, 0, 0, submit_ns, 0, 0);
        }
        if (ok) q36_vk_prof_kernel_collect_unlocked();
        q36_vk_prof_query_count = 0;
        q36_vk_batch_written_n = 0;
        q36_vk_batch_dirty_all = false;
        q36_vk_hazard_reset();
    }
    q36_vk_retired_sweep_unlocked();
    return ok;
}

static int q36_vk_begin_batch_unlocked(VkCommandBuffer *cmd_out) {
    q36_vk_cb_slot *slot = &q36_vk_ring[q36_vk_ring_cur];
    if (!slot->cb) {
        VkCommandBufferAllocateInfo cai = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = q36_vk.command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        if (vkAllocateCommandBuffers(q36_vk.device, &cai, &slot->cb) != VK_SUCCESS) return 0;
    }
    if (!slot->fence) {
        VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        if (vkCreateFence(q36_vk.device, &fci, NULL, &slot->fence) != VK_SUCCESS) return 0;
    }
    q36_vk.command_buffer = slot->cb;
    q36_vk.fence = slot->fence;
    if (!q36_vk_batch_recording) {
        VkCommandBufferBeginInfo bi = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        if (vkBeginCommandBuffer(q36_vk.command_buffer, &bi) != VK_SUCCESS) return 0;
        q36_vk_batch_recording = true;
    }
    if (cmd_out) *cmd_out = q36_vk.command_buffer;
    return 1;
}

static void q36_vk_note_written_tensor(q36_gpu_tensor *tensor) {
    q36_gpu_tensor *root = q36_gpu_tensor_root(tensor, NULL);
    if (!root) return;
    uint32_t i = 0;
    while (i < q36_vk_batch_written_n && q36_vk_batch_written[i] != root) i++;
    if (i < q36_vk_batch_written_n) return;
    if (q36_vk_batch_written_n == Q36_VK_WRITTEN_CAP) q36_vk_batch_dirty_all = true;
    else q36_vk_batch_written[q36_vk_batch_written_n++] = root;
}

static int q36_vk_flush_reason_unlocked(const char *reason) {
    const char *prev = q36_vk_flush_reason;
    q36_vk_flush_reason = reason ? reason : "submit_wait_other";
    int ok = q36_vk_flush_unlocked();
    q36_vk_flush_reason = prev;
    return ok;
}

static void q36_vk_weight_cache_free_unlocked(void) {
    q36_vk_weight *w = q36_vk.weights;
    q36_vk_packed_weight *pw = q36_vk.packed_weights;
    q36_vk_arena_block *b = q36_vk.arena;
    q36_vk_stream_cache_release_unlocked(true);
    while (w) {
        q36_vk_weight *next = w->next;
        q36_vk_tensor_free_unlocked(w->tensor);
        free(w);
        w = next;
    }
    q36_vk.weights = NULL;
    while (pw) {
        q36_vk_packed_weight *next = pw->next;
        q36_vk_tensor_free_unlocked(pw->tensor);
        free(pw);
        pw = next;
    }
    q36_vk.packed_weights = NULL;
    while (b) {
        q36_vk_arena_block *next = b->next;
        q36_vk_tensor_free_unlocked(b->base);
        free(b);
        b = next;
    }
    q36_vk.arena = NULL;
}

/* Suballocate weight bytes from the arena; new blocks prefer the
 * device-local heap and fall back to host-cached memory when it is full,
 * like the old per-weight allocations did.  256-byte alignment satisfies
 * any minStorageBufferOffsetAlignment the spec allows. */
enum { Q36_VK_ARENA_BLOCK_BYTES = 512u << 20 };

static q36_gpu_tensor *q36_vk_arena_alloc_unlocked(uint64_t bytes) {
    const uint64_t need = q36_round_up_u64(bytes ? bytes : 4, 256);
    q36_vk_arena_block *block = NULL;
    for (q36_vk_arena_block *b = q36_vk.arena; b; b = b->next) {
        if (b->base->alloc_bytes - b->used >= need) {
            block = b;
            break;
        }
    }
    if (!block) {
        uint64_t block_bytes = need > Q36_VK_ARENA_BLOCK_BYTES ? need : Q36_VK_ARENA_BLOCK_BYTES;
        q36_gpu_tensor *base = q36_vk_tensor_alloc_kind_unlocked(block_bytes, true);
        if (!base && block_bytes > need) base = q36_vk_tensor_alloc_kind_unlocked(need, true);
        if (!base) base = q36_vk_tensor_alloc_kind_unlocked(need, false);
        if (!base) return NULL;
        block = calloc(1, sizeof(*block));
        if (!block) {
            q36_vk_tensor_free_unlocked(base);
            return NULL;
        }
        base->weight_only = true;
        block->base = base;
        block->next = q36_vk.arena;
        q36_vk.arena = block;
    }
    {
        q36_gpu_tensor *view = q36_gpu_tensor_view(block->base, block->used, bytes);
        if (view) block->used += need;
        return view;
    }
}

/* Writes into the host-visible weight arena go to write-combined memory and
 * top out well below the SSD on a single thread, so bulk staging (prewarm)
 * splits the copy across the host pool.  Inference-time lazy staging keeps
 * the plain memcpy: it runs under q36_vk_mu in paths where the pool may be
 * busy with host compute. */
static bool q36_vk_weight_copy_parallel = false;

typedef struct {
    unsigned char *dst;
    const unsigned char *src;
    uint64_t bytes;
} q36_vk_weight_copy_ctx;

static void q36_vk_weight_copy_rows(void *ctx, uint64_t row0, uint64_t row1) {
    const q36_vk_weight_copy_ctx *c = ctx;
    uint64_t off0 = row0 << 20, off1 = row1 << 20;
    if (off1 > c->bytes) off1 = c->bytes;
    if (off0 < off1) memcpy(c->dst + off0, c->src + off0, (size_t)(off1 - off0));
}

static q36_gpu_tensor *q36_vk_weight_get_unlocked(const void *source, uint64_t bytes) {
    /* Move hits to the head: the cache is a linear list scanned per matvec
     * and a token touches the same few hundred weights in layer order. */
    for (q36_vk_weight *w = q36_vk.weights, *prev = NULL; w; prev = w, w = w->next) {
        if (w->source == source && w->bytes == bytes) {
            if (prev) {
                prev->next = w->next;
                w->next = q36_vk.weights;
                q36_vk.weights = w;
            }
            return w->tensor;
        }
    }

    q36_gpu_tensor *tensor = q36_vk_arena_alloc_unlocked(bytes);
    if (!tensor) return NULL;
    {
        uint64_t offset = 0;
        q36_gpu_tensor *root = q36_gpu_tensor_root(tensor, &offset);
        if (q36_vk_weight_copy_parallel) {
            q36_vk_weight_copy_ctx c = { root->data + offset, source, bytes };
            q36_gpu_parallel_for_rows((bytes + ((1u << 20) - 1)) >> 20, 8, q36_vk_weight_copy_rows, &c);
        } else {
            memcpy(root->data + offset, source, (size_t)bytes);
        }
    }

    q36_vk_weight *w = calloc(1, sizeof(*w));
    if (!w) {
        q36_vk_tensor_free_unlocked(tensor);
        return NULL;
    }
    w->source = source;
    w->bytes = bytes;
    w->tensor = tensor;
    w->next = q36_vk.weights;
    q36_vk.weights = w;
    return tensor;
}

typedef enum {
    Q36_VK_STREAM_PART_NONE = 0,
    Q36_VK_STREAM_PART_GATE,
    Q36_VK_STREAM_PART_UP,
    Q36_VK_STREAM_PART_DOWN,
} q36_vk_stream_part;

static bool q36_vk_stream_table_valid(const q36_gpu_stream_expert_table *t) {
    return t && t->model_map && t->model_size != 0 &&
           t->n_total_expert != 0 && t->n_total_expert <= 256 &&
           t->gate_expert_bytes != 0 &&
           t->up_expert_bytes != 0 &&
           t->down_expert_bytes != 0;
}

static uint64_t q36_vk_stream_table_expert_bytes(const q36_gpu_stream_expert_table *t) {
    if (!t) return 0;
    if (t->gate_expert_bytes > UINT64_MAX - t->up_expert_bytes) return 0;
    uint64_t sum = t->gate_expert_bytes + t->up_expert_bytes;
    if (sum > UINT64_MAX - t->down_expert_bytes) return 0;
    return sum + t->down_expert_bytes;
}

static bool q36_vk_stream_layout_matches_unlocked(const q36_gpu_stream_expert_table *t) {
    if (q36_vk_stream.cap == 0) return true;
    return q36_vk_stream.gate_expert_bytes == t->gate_expert_bytes &&
           q36_vk_stream.up_expert_bytes == t->up_expert_bytes &&
           q36_vk_stream.down_expert_bytes == t->down_expert_bytes &&
           q36_vk_stream.gate_type == t->gate_type &&
           q36_vk_stream.up_type == t->up_type &&
           q36_vk_stream.down_type == t->down_type &&
           q36_vk_stream.has_gate_scales == t->has_gate_scales &&
           q36_vk_stream.has_up_scales == t->has_up_scales &&
           q36_vk_stream.has_down_scales == t->has_down_scales;
}

static bool q36_vk_stream_cache_init_unlocked(const q36_gpu_stream_expert_table *t) {
    if (!q36_vk_stream.enabled || q36_vk_stream.budget == 0 || !q36_vk_stream_table_valid(t)) return false;
    uint64_t expert_bytes = q36_vk_stream_table_expert_bytes(t);
    if (expert_bytes == 0) return false;
    if (q36_vk_stream.configured_expert_bytes != 0 &&
        q36_vk_stream.configured_expert_bytes != expert_bytes) {
        return false;
    }
    if (q36_vk_stream.cap != 0) return q36_vk_stream_layout_matches_unlocked(t);

    uint32_t cap = q36_vk_stream.runtime_cap ? q36_vk_stream.runtime_cap : q36_vk_stream.budget;
    while (cap != 0) {
        uint64_t gate_bytes = 0, up_bytes = 0, down_bytes = 0, scale_bytes = 0;
        bool sizes_ok = q36_u64_mul_ok(cap, t->gate_expert_bytes, &gate_bytes) &&
                        q36_u64_mul_ok(cap, t->up_expert_bytes, &up_bytes) &&
                        q36_u64_mul_ok(cap, t->down_expert_bytes, &down_bytes) &&
                        q36_u64_mul_ok(cap, sizeof(float), &scale_bytes);
        if (sizes_ok) {
            q36_vk_stream.entry = calloc(cap, sizeof(q36_vk_stream.entry[0]));
            if (q36_vk_stream.entry)
                q36_vk_stream.gate = q36_vk_tensor_alloc_kind_unlocked(gate_bytes, true);
            if (q36_vk_stream.gate)
                q36_vk_stream.up = q36_vk_tensor_alloc_kind_unlocked(up_bytes, true);
            if (q36_vk_stream.up)
                q36_vk_stream.down = q36_vk_tensor_alloc_kind_unlocked(down_bytes, true);
            if (q36_vk_stream.down && t->has_gate_scales)
                q36_vk_stream.gate_scales = q36_vk_tensor_alloc_kind_unlocked(scale_bytes, true);
            if (q36_vk_stream.down && (!t->has_gate_scales || q36_vk_stream.gate_scales) && t->has_up_scales)
                q36_vk_stream.up_scales = q36_vk_tensor_alloc_kind_unlocked(scale_bytes, true);
            if (q36_vk_stream.down && (!t->has_gate_scales || q36_vk_stream.gate_scales) &&
                (!t->has_up_scales || q36_vk_stream.up_scales) && t->has_down_scales)
                q36_vk_stream.down_scales = q36_vk_tensor_alloc_kind_unlocked(scale_bytes, true);
        }
        if (sizes_ok && q36_vk_stream.entry && q36_vk_stream.gate &&
            q36_vk_stream.up && q36_vk_stream.down &&
            (!t->has_gate_scales || q36_vk_stream.gate_scales) &&
            (!t->has_up_scales || q36_vk_stream.up_scales) &&
            (!t->has_down_scales || q36_vk_stream.down_scales)) {
            break;
        }

        q36_vk_stream_cache_release_unlocked(true);
        uint32_t next = q36_ssd_shrink_cache_experts(cap);
        q36_vk_stream.runtime_cap = next;
        q36_vk_stream.allocation_failures++;
        fprintf(stderr,
                "q36: Vulkan streaming expert cache allocation failed at %u experts / %.2f GiB\n",
                cap, (double)cap * (double)expert_bytes / 1073741824.0);
        if (next != 0) {
            fprintf(stderr,
                    "q36:   shrinking resident cache margin by %u experts / %.2f GiB; runtime cache cap now %u experts\n",
                    cap - next,
                    (double)(cap - next) * (double)expert_bytes / 1073741824.0,
                    next);
        }
        cap = next;
    }
    if (cap == 0) return false;

    q36_vk_stream.cap = cap;
    q36_vk_stream.gate_expert_bytes = t->gate_expert_bytes;
    q36_vk_stream.up_expert_bytes = t->up_expert_bytes;
    q36_vk_stream.down_expert_bytes = t->down_expert_bytes;
    q36_vk_stream.gate_type = t->gate_type;
    q36_vk_stream.up_type = t->up_type;
    q36_vk_stream.down_type = t->down_type;
    q36_vk_stream.has_gate_scales = t->has_gate_scales;
    q36_vk_stream.has_up_scales = t->has_up_scales;
    q36_vk_stream.has_down_scales = t->has_down_scales;
    {
        const char *env = getenv("Q36_VK_STREAM_CACHE_INDEX");
        q36_vk_stream.indexed = !env || !env[0] || env[0] != '0';
    }
    return true;
}

static bool q36_vk_stream_cache_entry_matches(const q36_vk_stream_entry *e,
                                               const q36_gpu_stream_expert_table *t,
                                               uint32_t expert) {
    return e && e->valid && e->layer == t->layer && e->expert == expert &&
           e->gate_offset == t->gate_offset &&
           e->up_offset == t->up_offset &&
           e->down_offset == t->down_offset;
}

static int q36_vk_stream_cache_find_unlocked(const q36_gpu_stream_expert_table *t, uint32_t expert) {
    if (q36_vk_stream.indexed && t->layer < Q36_VK_STREAM_MAX_LAYERS &&
        expert < Q36_VK_STREAM_MAX_EXPERTS) {
        uint32_t mapped = q36_vk_stream.slot_of[t->layer][expert];
        q36_vk_stream.lookup_steps++;
        if (mapped == 0) return -1;
        uint32_t slot = mapped - 1u;
        if (slot < q36_vk_stream.cap &&
            q36_vk_stream_cache_entry_matches(&q36_vk_stream.entry[slot], t, expert)) {
            return (int)slot;
        }
        q36_vk_stream.slot_of[t->layer][expert] = 0;
        return -1;
    }
    for (uint32_t i = 0; i < q36_vk_stream.cap; i++) {
        q36_vk_stream_entry *e = &q36_vk_stream.entry[i];
        q36_vk_stream.lookup_steps++;
        if (q36_vk_stream_cache_entry_matches(e, t, expert)) return (int)i;
    }
    return -1;
}

static void q36_vk_stream_cache_unmap_slot_unlocked(uint32_t slot) {
    if (!q36_vk_stream.indexed || slot >= q36_vk_stream.cap) return;
    q36_vk_stream_entry *e = &q36_vk_stream.entry[slot];
    if (!e->valid || e->layer >= Q36_VK_STREAM_MAX_LAYERS ||
        e->expert >= Q36_VK_STREAM_MAX_EXPERTS) return;
    uint32_t *mapped = &q36_vk_stream.slot_of[e->layer][e->expert];
    if (*mapped == slot + 1u) *mapped = 0;
}

static uint32_t q36_vk_stream_route_hotness_unlocked(uint32_t layer, uint32_t expert) {
    if (layer >= Q36_VK_STREAM_MAX_LAYERS || expert >= Q36_VK_STREAM_MAX_EXPERTS) return 0;
    return q36_vk_stream.route_hotness[layer][expert];
}

static void q36_vk_stream_note_hotness_unlocked(uint32_t layer, uint32_t expert,
                                                uint32_t amount) {
    if (layer >= Q36_VK_STREAM_MAX_LAYERS || expert >= Q36_VK_STREAM_MAX_EXPERTS || amount == 0) return;
    uint32_t *hotness = &q36_vk_stream.route_hotness[layer][expert];
    *hotness = *hotness > UINT32_MAX - amount ? UINT32_MAX : *hotness + amount;
}

static void q36_vk_stream_decay_hotness_unlocked(void) {
    while (q36_vk_stream.route_tokens - q36_vk_stream.hotness_decay_token >= 16u) {
        for (uint32_t layer = 0; layer < Q36_VK_STREAM_MAX_LAYERS; layer++) {
            for (uint32_t expert = 0; expert < Q36_VK_STREAM_MAX_EXPERTS; expert++) {
                q36_vk_stream.route_hotness[layer][expert] >>= 1;
            }
        }
        q36_vk_stream.hotness_decay_token += 16u;
    }
}

static uint32_t q36_vk_stream_cache_pick_slot_unlocked(void) {
    uint32_t best = UINT32_MAX;
    for (uint32_t i = 0; i < q36_vk_stream.cap; i++) {
        if (q36_vk_stream.entry[i].loading) continue;
        if (!q36_vk_stream.entry[i].valid) return i;
        uint32_t hotness = q36_vk_stream_route_hotness_unlocked(
            q36_vk_stream.entry[i].layer, q36_vk_stream.entry[i].expert);
        uint32_t best_hotness = best == UINT32_MAX ? UINT32_MAX :
            q36_vk_stream_route_hotness_unlocked(
                q36_vk_stream.entry[best].layer, q36_vk_stream.entry[best].expert);
        if (best == UINT32_MAX || hotness < best_hotness ||
            (hotness == best_hotness &&
             q36_vk_stream.entry[i].last_used < q36_vk_stream.entry[best].last_used)) {
            best = i;
        }
    }
    if (best != UINT32_MAX) q36_vk_stream.evictions++;
    return best;
}

static bool q36_vk_stream_cache_copy_scale_unlocked(q36_gpu_tensor *dst,
                                                    uint32_t slot,
                                                    const q36_gpu_stream_expert_table *t,
                                                    uint64_t scale_offset,
                                                    uint32_t expert) {
    if (!dst) return true;
    float value = 1.0f;
    if (!q36_vk_stream_copy_model_range(t->model_map,
                                        t->model_size,
                                        scale_offset + (uint64_t)expert * sizeof(float),
                                        &value,
                                        sizeof(value))) {
        return false;
    }
    memcpy(dst->data + (uint64_t)slot * sizeof(float), &value, sizeof(value));
    return true;
}

/* Copy one expert's gate/up/down (+scales) into its cache slot.  Pure data
 * movement into disjoint slot ranges, so the miss batch fans out across the
 * host pool while the prepare pass holds q36_vk_mu: parallel preads keep the
 * SSD queue full where one blocking read per expert leaves it idle. */
static bool q36_vk_stream_slot_fetch(uint32_t slot,
                                     const q36_gpu_stream_expert_table *t,
                                     uint32_t expert) {
    unsigned char *gate_dst = q36_vk_stream.gate->data + (uint64_t)slot * q36_vk_stream.gate_expert_bytes;
    unsigned char *up_dst = q36_vk_stream.up->data + (uint64_t)slot * q36_vk_stream.up_expert_bytes;
    unsigned char *down_dst = q36_vk_stream.down->data + (uint64_t)slot * q36_vk_stream.down_expert_bytes;
    return q36_vk_stream_copy_model_range(t->model_map,
                                          t->model_size,
                                          t->gate_offset + (uint64_t)expert * t->gate_expert_bytes,
                                          gate_dst,
                                          t->gate_expert_bytes) &&
           q36_vk_stream_copy_model_range(t->model_map,
                                          t->model_size,
                                          t->up_offset + (uint64_t)expert * t->up_expert_bytes,
                                          up_dst,
                                          t->up_expert_bytes) &&
           q36_vk_stream_copy_model_range(t->model_map,
                                          t->model_size,
                                          t->down_offset + (uint64_t)expert * t->down_expert_bytes,
                                          down_dst,
                                          t->down_expert_bytes) &&
           (!t->has_gate_scales || q36_vk_stream_cache_copy_scale_unlocked(q36_vk_stream.gate_scales, slot, t, t->gate_scales_offset, expert)) &&
           (!t->has_up_scales || q36_vk_stream_cache_copy_scale_unlocked(q36_vk_stream.up_scales, slot, t, t->up_scales_offset, expert)) &&
           (!t->has_down_scales || q36_vk_stream_cache_copy_scale_unlocked(q36_vk_stream.down_scales, slot, t, t->down_scales_offset, expert));
}

typedef struct {
    uint32_t slot;
    uint32_t expert;
    bool ok;
} q36_vk_stream_miss;

typedef struct {
    const q36_gpu_stream_expert_table *t;
    q36_vk_stream_miss *miss;
} q36_vk_stream_fetch_ctx;

static void q36_vk_stream_fetch_rows(void *ctx, uint64_t row0, uint64_t row1) {
    q36_vk_stream_fetch_ctx *c = ctx;
    for (uint64_t i = row0; i < row1; i++) {
        c->miss[i].ok = q36_vk_stream_slot_fetch(c->miss[i].slot, c->t, c->miss[i].expert);
    }
}

/* Hit-or-reserve: hits bump recency, misses claim a slot and record the
 * pending fetch.  The reserved entry carries its final identity immediately
 * so duplicate selections in the same list hit it. */
static bool q36_vk_stream_cache_prepare_one_unlocked(const q36_gpu_stream_expert_table *t,
                                                     uint32_t expert,
                                                     uint32_t priority,
                                                     uint32_t *slot_map,
                                                     q36_vk_stream_miss *miss,
                                                     uint32_t *n_miss,
                                                     bool *need_flush) {
    if (expert >= t->n_total_expert) return false;
    q36_vk_stream_note_hotness_unlocked(t->layer, expert, priority ? priority : 1u);
    int slot = q36_vk_stream_cache_find_unlocked(t, expert);
    if (slot >= 0) {
        q36_vk_stream.hits++;
    } else {
        if (*n_miss >= 256u) return false;
        q36_vk_stream.misses++;
        uint32_t pick = q36_vk_stream_cache_pick_slot_unlocked();
        if (pick == UINT32_MAX) return false;
        q36_vk_stream_entry *entry = &q36_vk_stream.entry[pick];
        if (entry->valid) *need_flush = true;
        q36_vk_stream_cache_unmap_slot_unlocked(pick);
        entry->valid = true;
        entry->loading = true;
        entry->layer = t->layer;
        entry->expert = expert;
        entry->gate_offset = t->gate_offset;
        entry->up_offset = t->up_offset;
        entry->down_offset = t->down_offset;
        if (q36_vk_stream.indexed && t->layer < Q36_VK_STREAM_MAX_LAYERS &&
            expert < Q36_VK_STREAM_MAX_EXPERTS) {
            q36_vk_stream.slot_of[t->layer][expert] = pick + 1u;
        }
        miss[*n_miss] = (q36_vk_stream_miss) { pick, expert, false };
        (*n_miss)++;
        slot = (int)pick;
    }

    q36_vk_stream_entry *entry = &q36_vk_stream.entry[slot];
    entry->last_used = ++q36_vk_stream.clock;
    if (slot_map) slot_map[expert] = (uint32_t)slot;
    return true;
}

/* Flush once if any reserved slot evicts live data, then fetch the whole
 * miss batch in parallel and finalize the entries. */
static bool q36_vk_stream_cache_finish_unlocked(const q36_gpu_stream_expert_table *t,
                                                q36_vk_stream_miss *miss,
                                                uint32_t n_miss,
                                                bool need_flush,
                                                bool prepared) {
    bool ok = prepared;
    if (n_miss == 0) return ok;
    if (ok && need_flush && !q36_vk_flush_reason_unlocked("submit_wait_stream_cache_evict")) ok = false;
    if (ok) {
        q36_vk_stream_fetch_ctx ctx = { t, miss };
        q36_gpu_parallel_for_rows(n_miss, 2, q36_vk_stream_fetch_rows, &ctx);
    }
    for (uint32_t i = 0; i < n_miss; i++) {
        q36_vk_stream_entry *entry = &q36_vk_stream.entry[miss[i].slot];
        entry->loading = false;
        if (ok && miss[i].ok) {
            q36_vk_stream.loads++;
        } else {
            q36_vk_stream_cache_unmap_slot_unlocked(miss[i].slot);
            entry->valid = false;
            ok = false;
        }
    }
    return ok;
}

static bool q36_vk_stream_cache_prepare_u32(const q36_gpu_stream_expert_table *t,
                                            const uint32_t *selected,
                                            uint64_t n_selected,
                                            uint32_t *slot_map) {
    if (!selected || !slot_map || n_selected == 0) return false;
    for (uint32_t i = 0; i < 256; i++) slot_map[i] = UINT32_MAX;

    bool seen[256] = {0};
    uint32_t unique = 0;
    for (uint64_t i = 0; i < n_selected; i++) {
        uint32_t expert = selected[i];
        if (expert >= 256) return false;
        if (!seen[expert]) {
            seen[expert] = true;
            unique++;
        }
    }
    pthread_mutex_lock(&q36_vk_mu);
    bool ok = q36_vk_stream_cache_init_unlocked(t);
    if (ok && unique > q36_vk_stream.cap) ok = false;
    if (ok) q36_vk_stream_profile_record_unlocked(t, selected, n_selected);
    q36_vk_stream_miss miss[256];
    uint32_t n_miss = 0;
    bool need_flush = false;
    for (uint64_t i = 0; ok && i < n_selected; i++) {
        uint32_t expert = selected[i];
        if (expert >= t->n_total_expert) {
            ok = false;
            break;
        }
        ok = q36_vk_stream_cache_prepare_one_unlocked(t, expert, 1, slot_map,
                                                      miss, &n_miss, &need_flush);
    }
    ok = q36_vk_stream_cache_finish_unlocked(t, miss, n_miss, need_flush, ok);
    pthread_mutex_unlock(&q36_vk_mu);
    return ok;
}

static uint32_t q36_vk_stream_cache_cap(const q36_gpu_stream_expert_table *t) {
    pthread_mutex_lock(&q36_vk_mu);
    uint32_t cap = q36_vk_stream_cache_init_unlocked(t) ? q36_vk_stream.cap : 0;
    pthread_mutex_unlock(&q36_vk_mu);
    return cap;
}

static void q36_vk_stream_note_routes(const q36_gpu_stream_expert_table *t,
                                      const uint32_t *selected,
                                      uint64_t n_selected) {
    pthread_mutex_lock(&q36_vk_mu);
    q36_vk_stream_profile_record_unlocked(t, selected, n_selected);
    for (uint64_t i = 0; i < n_selected; i++) {
        if (selected[i] < t->n_total_expert) {
            q36_vk_stream_note_hotness_unlocked(t->layer, selected[i], 1u);
        }
    }
    pthread_mutex_unlock(&q36_vk_mu);
}

static bool q36_vk_stream_cache_prepare_i32(const q36_gpu_stream_expert_table *t,
                                            const int32_t *selected,
                                            const uint32_t *priorities,
                                            uint32_t n_selected,
                                            uint32_t *slot_map) {
    if (!selected || n_selected == 0) return false;
    if (slot_map) for (uint32_t i = 0; i < 256; i++) slot_map[i] = UINT32_MAX;

    pthread_mutex_lock(&q36_vk_mu);
    bool ok = q36_vk_stream_cache_init_unlocked(t);
    q36_vk_stream_miss miss[256];
    uint32_t n_miss = 0;
    bool need_flush = false;
    for (uint32_t i = 0; ok && i < n_selected; i++) {
        if (selected[i] < 0 || (uint32_t)selected[i] >= t->n_total_expert) {
            ok = false;
            break;
        }
        uint32_t priority = priorities ? priorities[i] : 1u;
        ok = q36_vk_stream_cache_prepare_one_unlocked(t, (uint32_t)selected[i], priority, slot_map,
                                                      miss, &n_miss, &need_flush);
    }
    ok = q36_vk_stream_cache_finish_unlocked(t, miss, n_miss, need_flush, ok);
    pthread_mutex_unlock(&q36_vk_mu);
    return ok;
}

static const q36_gpu_tensor *q36_vk_stream_weight_tensor_unlocked(q36_vk_stream_part part) {
    if (q36_vk_stream.cap == 0) return NULL;
    if (part == Q36_VK_STREAM_PART_GATE) return q36_vk_stream.gate;
    if (part == Q36_VK_STREAM_PART_UP) return q36_vk_stream.up;
    if (part == Q36_VK_STREAM_PART_DOWN) return q36_vk_stream.down;
    return NULL;
}

static const q36_gpu_tensor *q36_vk_stream_scale_tensor_unlocked(q36_vk_stream_part part) {
    if (q36_vk_stream.cap == 0) return NULL;
    if (part == Q36_VK_STREAM_PART_GATE) return q36_vk_stream.gate_scales;
    if (part == Q36_VK_STREAM_PART_UP) return q36_vk_stream.up_scales;
    if (part == Q36_VK_STREAM_PART_DOWN) return q36_vk_stream.down_scales;
    return NULL;
}

static int q36_vk_descriptor_for_tensor(const q36_gpu_tensor *tensor, VkDescriptorBufferInfo *info) {
    uint64_t root_offset = 0;
    q36_gpu_tensor *root = q36_gpu_tensor_root(tensor, &root_offset);
    if (!root || !root->buffer) return 0;
    uint64_t range = tensor->bytes ? q36_round_up_u64(tensor->bytes, 4) : 1;
    if (root_offset > root->alloc_bytes) return 0;
    if (range > root->alloc_bytes - root_offset) range = root->alloc_bytes - root_offset;
    if (range == 0) range = 1;
    *info = (VkDescriptorBufferInfo){
        .buffer = root->buffer,
        .offset = root_offset,
        .range = range,
    };
    return 1;
}

static int q36_vk_alloc_descriptor_unlocked(q36_vk_kernel *k, VkDescriptorSet *set) {
    VkDescriptorSetAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = q36_vk.descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &k->set_layout,
    };
    VkResult rc = vkAllocateDescriptorSets(q36_vk.device, &ai, set);
    if (rc == VK_SUCCESS) return 1;
    /* The pool's live sets belong to the open batch; rotate to the next
     * ring slot (its pool was reset when its fence completed). */
    if (!q36_vk_submit_eager_unlocked()) return 0;
    ai.descriptorPool = q36_vk.descriptor_pool;
    rc = vkAllocateDescriptorSets(q36_vk.device, &ai, set);
    return rc == VK_SUCCESS;
}

static int q36_vk_run_unlocked(
        const char                 *op,
        q36_vk_kernel              *kernel,
        const q36_gpu_tensor *const *bindings,
        const void                 *push,
        uint32_t                    push_bytes,
        uint32_t                    groups_x,
        uint32_t                    groups_y,
        uint32_t                    groups_z) {
    const char *prof_op = q36_vk_prof_op_name(op);
    uint64_t prof_t0 = q36_vk_now_ns();
    uint64_t groups_total = (uint64_t)groups_x * groups_y * groups_z;
    if (!q36_vk_kernel_init(kernel)) return 0;
    if (push_bytes != kernel->push_bytes) return 0;
    if (q36_vk.prof_kernel && q36_vk.query_pool && q36_vk_prof_query_count + 2u > Q36_VK_PROF_QUERY_CAP) {
        if (!q36_vk_flush_reason_unlocked("submit_wait_query_pool")) return 0;
    }

    VkDescriptorBufferInfo infos[8];
    VkWriteDescriptorSet writes[8];
    for (uint32_t i = 0; i < kernel->bindings; i++) {
        if (!q36_vk_descriptor_for_tensor(bindings[i], &infos[i])) return 0;
    }

    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!q36_vk_alloc_descriptor_unlocked(kernel, &set)) return 0;
    for (uint32_t i = 0; i < kernel->bindings; i++) {
        writes[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[i],
        };
    }
    vkUpdateDescriptorSets(q36_vk.device, kernel->bindings, writes, 0, NULL);

    /* Record into the open batch; submission happens at the next host read
     * (q36_vk_flush_unlocked).  One reusable command buffer and fence are
     * enough because the flush always waits before recording resumes. */
    VkCommandBuffer cmd;
    if (!q36_vk_begin_batch_unlocked(&cmd)) return 0;
    q36_gpu_tensor *roots[16];
    for (uint32_t i = 0; i < kernel->bindings; i++) {
        roots[i] = q36_gpu_tensor_root(bindings[i], NULL);
    }
    {
        /* Barrier only on a real hazard against work recorded since the
         * last barrier - including work in earlier eager-submitted buffers,
         * which executes in submission order and is still covered by a
         * barrier recorded here. Independent dispatches overlap. */
        bool need_barrier = q36_vk_hazard_all;
        for (uint32_t i = 0; !need_barrier && i < kernel->bindings; i++) {
            if (roots[i]->weight_only) continue;
            if (q36_vk_hazard_in(q36_vk_hazard_written, q36_vk_hazard_written_n, roots[i])) need_barrier = true;
            else if (((kernel->write_mask >> i) & 1u) &&
                     q36_vk_hazard_in(q36_vk_hazard_read, q36_vk_hazard_read_n, roots[i])) need_barrier = true;
        }
        if (need_barrier) {
            VkMemoryBarrier compute_barrier = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            };
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &compute_barrier, 0, NULL, 0, NULL);
            q36_vk_hazard_reset();
        }
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kernel->layout, 0, 1, &set, 0, NULL);
    if (push_bytes) vkCmdPushConstants(cmd, kernel->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
    uint32_t query0 = UINT32_MAX;
    if (q36_vk.prof_kernel && q36_vk.query_pool && q36_vk_prof_query_count + 2u <= Q36_VK_PROF_QUERY_CAP) {
        query0 = q36_vk_prof_query_count;
        q36_vk_prof_query_count += 2u;
        q36_vk_prof_query_path[query0 / 2u] = kernel->path;
        q36_vk_prof_query_op[query0 / 2u] = prof_op;
        q36_vk_prof_query_groups[query0 / 2u] = groups_total;
        vkCmdResetQueryPool(cmd, q36_vk.query_pool, query0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, q36_vk.query_pool, query0);
    }
    vkCmdDispatch(cmd, groups_x, groups_y, groups_z);
    if (query0 != UINT32_MAX) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, q36_vk.query_pool, query0 + 1u);
    }
    for (uint32_t i = 0; i < kernel->bindings; i++) {
        roots[i]->last_use_seq = q36_vk_submit_seq + 1;
        if ((kernel->write_mask >> i) & 1u) {
            roots[i]->gpu_written = true;
            q36_vk_hazard_add(q36_vk_hazard_written, &q36_vk_hazard_written_n, roots[i]);
            q36_vk_note_written_tensor((q36_gpu_tensor *)bindings[i]);
        } else {
            q36_vk_hazard_add(q36_vk_hazard_read, &q36_vk_hazard_read_n, roots[i]);
        }
    }
    q36_vk_prof_dispatches++;
    {
        uint64_t record_ns = q36_vk_now_ns() - prof_t0;
        q36_vk_prof_record_ns += record_ns;
        q36_vk_prof_op_add(prof_op, 1, 0, groups_total, record_ns, 0, 0, 0);
    }
    if (++q36_vk_batch_count >= Q36_VK_BATCH_FLUSH_AT) return q36_vk_submit_eager_unlocked();
    return 1;
}

int q36_gpu_init(void) {
    pthread_mutex_lock(&q36_vk_mu);
    if (q36_vk.ready) {
        pthread_mutex_unlock(&q36_vk_mu);
        return 1;
    }

    memset(&q36_vk, 0, sizeof(q36_vk));
    q36_vk.prof_ops = getenv("Q36_VK_PROF_OP") || getenv("Q36_VK_PROF") || getenv("Q36_VK_PROF_KERNEL");
    /* write_mask marks the bindings each shader writes; the lazy-flush
     * tracker uses it so host reads of input-only tensors do not submit
     * the open batch. */
    q36_vk.matmul_f16 = Q36_VK_KERNEL("vulkan/matmul_f16.spv", 3, 16, 1u << 2);
    q36_vk.matmul_f32 = Q36_VK_KERNEL("vulkan/matmul_f32.spv", 3, 16, 1u << 2);
    q36_vk.matmul_f32_fast = Q36_VK_KERNEL("vulkan/matmul_f32_fast.spv", 3, 16, 1u << 2);
    q36_vk.add = Q36_VK_KERNEL("vulkan/add.spv", 3, 4, 1u << 2);
    q36_vk.directional_steering = Q36_VK_KERNEL("vulkan/directional_steering.spv", 2, 16, 1u << 0);
    q36_vk.rms_norm = Q36_VK_KERNEL("vulkan/rms_norm.spv", 3, 12, 1u << 2);
    q36_vk.add_rms_norm = Q36_VK_KERNEL("vulkan/add_rms_norm.spv", 5, 8, (1u << 3) | (1u << 4));
    q36_vk.swiglu = Q36_VK_KERNEL("vulkan/swiglu.spv", 3, 12, 1u << 2);
    q36_vk.rope = Q36_VK_KERNEL("vulkan/rope_qwen.spv", 1, 16, 1u << 0);
    q36_vk.rms_norm_rope = Q36_VK_KERNEL("vulkan/rms_norm_rope_qwen.spv", 3, 24, 1u << 2);
    q36_vk.rms_norm_rope_kv = Q36_VK_KERNEL("vulkan/rms_norm_rope_kv_qwen.spv", 5, 24,
                                            (1u << 2) | (1u << 4));
    q36_vk.copy_rows = Q36_VK_KERNEL("vulkan/copy_rows.spv", 2, 20, 1u << 1);
    q36_vk.recur_window = Q36_VK_KERNEL("vulkan/recur_window.spv", 3, 12, (1u << 0) | (1u << 2));
    q36_vk.conv_silu = Q36_VK_KERNEL("vulkan/conv_silu.spv", 3, 12, 1u << 2);
    q36_vk.recur_conv_silu_decode = Q36_VK_KERNEL("vulkan/recur_conv_silu_decode.spv", 4, 12,
                                                   (1u << 0) | (1u << 3));
    q36_vk.delta_qk = Q36_VK_KERNEL("vulkan/delta_qk.spv", 3, 24, (1u << 1) | (1u << 2));
    q36_vk.delta_qkv = Q36_VK_KERNEL("vulkan/delta_qkv.spv", 4, 24,
                                     (1u << 1) | (1u << 2) | (1u << 3));
    q36_vk.delta_gates = Q36_VK_KERNEL("vulkan/delta_gates.spv", 5, 8, 1u << 4);
    q36_vk.quantize_q8_0 = Q36_VK_KERNEL("vulkan/quantize_q8_0.spv", 2, 12, 1u << 1);
    q36_vk.quantize_q8_k = Q36_VK_KERNEL("vulkan/quantize_q8_k.spv", 2, 12, 1u << 1);
    q36_vk.matmul_q8_0 = Q36_VK_KERNEL("vulkan/matmul_q8_0.spv", 3, 20, 1u << 2);
    q36_vk.matmul_q8_0_q36 = Q36_VK_KERNEL("vulkan/matmul_q8_0_q36.spv", 3, 20, 1u << 2);
    q36_vk.matmul_q8_0_f32b = Q36_VK_KERNEL("vulkan/matmul_q8_0_f32b.spv", 3, 20, 1u << 2);
    q36_vk.shared_gate_up_decode = Q36_VK_KERNEL("vulkan/shared_gate_up_decode.spv", 4, 16, 1u << 3);
    q36_vk.shared_down_tail_decode = Q36_VK_KERNEL("vulkan/shared_down_tail_decode.spv", 4, 12, 1u << 3);
    q36_vk.matmul_q8_0_mm = Q36_VK_KERNEL("vulkan/matmul_q8_0_mm.spv", 3, 20, 1u << 2);
    q36_vk.matmul_q8_0_mm_f16 = Q36_VK_KERNEL("vulkan/matmul_q8_0_mm_f16.spv", 3, 20, 1u << 2);
    q36_vk.matmul_q8_0_mm_f16_out32 = Q36_VK_KERNEL("vulkan/matmul_q8_0_mm_f16_out32.spv", 3, 20, 1u << 2);
    q36_vk.matmul_q8_0_f32b_nx = Q36_VK_KERNEL("vulkan/matmul_q8_0_f32b_nx.spv", 3, 20, 1u << 2);
    q36_vk.matmul_q8_0_decode = Q36_VK_KERNEL("vulkan/matmul_q8_0_decode.spv", 3, 20, 1u << 2);
    q36_vk.matmul_q8_0_decode_q36 = Q36_VK_KERNEL("vulkan/matmul_q8_0_decode_q36.spv", 3, 20, 1u << 2);
    q36_vk.matmul_q8_0_decode_b64 = Q36_VK_KERNEL("vulkan/matmul_q8_0_decode_b64.spv", 3, 20, 1u << 2);
    q36_vk.matmul_kquant = Q36_VK_KERNEL("vulkan/matmul_kquant.spv", 3, 28, 1u << 2);
    q36_vk.matmul_q5k_mmq = Q36_VK_KERNEL("vulkan/matmul_q5k_mmq.spv", 3, 16, 1u << 2);
    q36_vk.matmul_q6k_mmq = Q36_VK_KERNEL("vulkan/matmul_q6k_mmq.spv", 3, 16, 1u << 2);
    q36_vk.matmul_q5k_mmq_fast = Q36_VK_KERNEL("vulkan/matmul_q5k_mmq_fast.spv", 3, 16, 1u << 2);
    q36_vk.matmul_q6k_mmq_fast = Q36_VK_KERNEL("vulkan/matmul_q6k_mmq_fast.spv", 3, 16, 1u << 2);
    q36_vk.delta_net = Q36_VK_KERNEL("vulkan/delta_net.spv", 6, 12, (1u << 0) | (1u << 5));
    q36_vk.delta_net_fast = Q36_VK_KERNEL("vulkan/delta_net_fast.spv", 6, 12, (1u << 0) | (1u << 5));
    q36_vk.delta_net_cols = Q36_VK_KERNEL("vulkan/delta_net_cols.spv", 6, 16, (1u << 0) | (1u << 5));
    q36_vk.delta_net_decode = Q36_VK_KERNEL("vulkan/delta_net_decode.spv", 6, 12, (1u << 0) | (1u << 5));
    q36_vk.attn_scores = Q36_VK_KERNEL("vulkan/attn_scores.spv", 3, 28, 1u << 2);
    q36_vk.attn_post = Q36_VK_KERNEL("vulkan/attn_post.spv", 4, 28, (1u << 0) | (1u << 3));
    q36_vk.attn_reduce = Q36_VK_KERNEL("vulkan/attn_reduce.spv", 4, 28, 1u << 3);
    q36_vk.attn_decode_fused = Q36_VK_KERNEL("vulkan/attn_decode_fused.spv", 6, 40, 1u << 5);
    q36_vk.attn_decode_split = Q36_VK_KERNEL("vulkan/attn_decode_split.spv", 5, 40, 1u << 4);
    q36_vk.attn_prefill_qtile = Q36_VK_KERNEL("vulkan/attn_prefill_qtile.spv", 5, 44, 1u << 4);
    q36_vk.attn_combine = Q36_VK_KERNEL("vulkan/attn_combine.spv", 4, 24, 1u << 3);
    q36_vk.moe_gate_up = Q36_VK_KERNEL("vulkan/moe_gate_up.spv", 8, 28, 1u << 6);
    q36_vk.router_topk = Q36_VK_KERNEL("vulkan/router_topk.spv", 3, 16, (1u << 1) | (1u << 2));
    q36_vk.moe_tiles = Q36_VK_KERNEL("vulkan/moe_tiles.spv", 3, 16, 1u << 1);
    q36_vk.kv_store = Q36_VK_KERNEL("vulkan/kv_store.spv", 4, 16, (1u << 0) | (1u << 1));
    q36_vk.kv_store_quant = Q36_VK_KERNEL("vulkan/kv_store_quant.spv", 2, 20, 1u << 0);
    q36_vk.rms_norm_rope_kv_quant = Q36_VK_KERNEL("vulkan/rms_norm_rope_kv_qwen_quant.spv", 5, 32, (1u << 2) | (1u << 4));
    q36_vk.moe_gate_up_f32b = Q36_VK_KERNEL("vulkan/moe_gate_up_f32b.spv", 8, 24, 1u << 6);
    q36_vk.moe_down_q2k_f32b = Q36_VK_KERNEL("vulkan/moe_down_q2k_f32b.spv", 5, 24, 1u << 4);
    q36_vk.moe_down_q2k_sum_decode = Q36_VK_KERNEL("vulkan/moe_down_q2k_sum_decode.spv", 6, 24, 1u << 5);
    q36_vk.moe_gate_up_gemm = Q36_VK_KERNEL("vulkan/moe_gate_up_gemm.spv", 8, 24, 1u << 6);
    q36_vk.moe_down_gemm = Q36_VK_KERNEL("vulkan/moe_down_gemm.spv", 5, 24, 1u << 4);
    q36_vk.moe_matvec = Q36_VK_KERNEL("vulkan/moe_matvec.spv", 6, 36, 1u << 4);
    q36_vk.moe_matvec_fast = Q36_VK_KERNEL("vulkan/moe_matvec_fast.spv", 6, 36, 1u << 4);
    q36_vk.moe_reduce = Q36_VK_KERNEL("vulkan/moe_reduce.spv", 3, 8, 1u << 2);
    q36_vk.ffn_tail = Q36_VK_KERNEL("vulkan/ffn_tail.spv", 3, 8, 1u << 0);

    q36_vk.api_version = VK_API_VERSION_1_0;
    PFN_vkEnumerateInstanceVersion enumerate_instance_version =
        (PFN_vkEnumerateInstanceVersion)vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion");
    if (enumerate_instance_version) (void)enumerate_instance_version(&q36_vk.api_version);

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "q36",
        .applicationVersion = 1,
        .pEngineName = "q36_vulkan",
        .engineVersion = 1,
        .apiVersion = q36_vk.api_version,
    };
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    if (vkCreateInstance(&ici, NULL, &q36_vk.instance) != VK_SUCCESS) goto fail;

    uint32_t ndev = 0;
    if (vkEnumeratePhysicalDevices(q36_vk.instance, &ndev, NULL) != VK_SUCCESS || ndev == 0) goto fail;
    VkPhysicalDevice *devs = calloc(ndev, sizeof(devs[0]));
    if (!devs) goto fail;
    if (vkEnumeratePhysicalDevices(q36_vk.instance, &ndev, devs) != VK_SUCCESS) {
        free(devs);
        goto fail;
    }

    for (uint32_t di = 0; di < ndev && !q36_vk.physical; di++) {
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[di], &nq, NULL);
        VkQueueFamilyProperties *qprops = calloc(nq, sizeof(qprops[0]));
        if (!qprops) continue;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[di], &nq, qprops);
        for (uint32_t qi = 0; qi < nq; qi++) {
            if ((qprops[qi].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) continue;
            q36_vk.physical = devs[di];
            q36_vk.queue_family = qi;
            q36_vk.timestamp_valid_bits = qprops[qi].timestampValidBits;
            break;
        }
        free(qprops);
    }
    free(devs);
    if (!q36_vk.physical) goto fail;

    vkGetPhysicalDeviceProperties(q36_vk.physical, &q36_vk.props);
    vkGetPhysicalDeviceMemoryProperties(q36_vk.physical, &q36_vk.mem_props);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = q36_vk.queue_family,
        .queueCount = 1,
        .pQueuePriorities = &prio,
    };
    /* The matmul / attention / delta-net kernels accumulate in fp64 and
     * emulate correctly rounded f32 fma via int64 bit ops to stay bit-exact
     * against the CPU reference engine, so shaderFloat64 and shaderInt64 are
     * hard requirements (RADV exposes both on the BC-250 target). */
    VkPhysicalDeviceFeatures supported;
    vkGetPhysicalDeviceFeatures(q36_vk.physical, &supported);
    if (!supported.shaderFloat64 || !supported.shaderInt64) goto fail;
    VkPhysicalDeviceFeatures enabled = { .shaderFloat64 = VK_TRUE, .shaderInt64 = VK_TRUE };
    /* f16 arithmetic is optional: the packed-f16 prefill GEMM needs it, and
     * falls back to the f32 kernel when the device lacks the feature. */
    VkPhysicalDeviceShaderFloat16Int8Features f16_enable = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
        .shaderFloat16 = VK_TRUE,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
        .pEnabledFeatures = &enabled,
    };
    if (q36_vk.api_version >= VK_API_VERSION_1_2 &&
        q36_vk.props.apiVersion >= VK_API_VERSION_1_2) {
        VkPhysicalDeviceShaderFloat16Int8Features f16_query = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
        };
        VkPhysicalDeviceFeatures2 feat2 = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &f16_query,
        };
        vkGetPhysicalDeviceFeatures2(q36_vk.physical, &feat2);
        if (f16_query.shaderFloat16) {
            q36_vk.have_f16 = true;
            dci.pNext = &f16_enable;
        }
    }
    if (vkCreateDevice(q36_vk.physical, &dci, NULL, &q36_vk.device) != VK_SUCCESS) goto fail;
    vkGetDeviceQueue(q36_vk.device, q36_vk.queue_family, 0, &q36_vk.queue);

    VkCommandPoolCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = q36_vk.queue_family,
    };
    if (vkCreateCommandPool(q36_vk.device, &cpci, NULL, &q36_vk.command_pool) != VK_SUCCESS) goto fail;

    VkDescriptorPoolSize ps = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 16384,
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 4096,
        .poolSizeCount = 1,
        .pPoolSizes = &ps,
    };
    for (uint32_t i = 0; i < Q36_VK_CB_RING; i++) {
        if (vkCreateDescriptorPool(q36_vk.device, &dpci, NULL, &q36_vk_ring[i].pool) != VK_SUCCESS) goto fail;
    }
    q36_vk_ring_activate_unlocked(0);

    if (getenv("Q36_VK_PROF_KERNEL") && q36_vk.timestamp_valid_bits != 0) {
        VkQueryPoolCreateInfo qpci = {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = Q36_VK_PROF_QUERY_CAP,
        };
        if (vkCreateQueryPool(q36_vk.device, &qpci, NULL, &q36_vk.query_pool) == VK_SUCCESS) {
            q36_vk.prof_kernel = true;
        }
    }

    q36_vk.ready = true;
    q36_vk_stream_profile_init();
    if (getenv("Q36_VK_PROF") || getenv("Q36_VK_PROF_KERNEL") || getenv("Q36_VK_PROF_OP")) atexit(q36_vk_prof_report);
    pthread_mutex_unlock(&q36_vk_mu);
    return 1;

fail:
    if (q36_vk.device) vkDeviceWaitIdle(q36_vk.device);
    if (q36_vk.query_pool) vkDestroyQueryPool(q36_vk.device, q36_vk.query_pool, NULL);
    for (uint32_t i = 0; i < Q36_VK_CB_RING; i++) {
        if (q36_vk_ring[i].pool) vkDestroyDescriptorPool(q36_vk.device, q36_vk_ring[i].pool, NULL);
        if (q36_vk_ring[i].fence) vkDestroyFence(q36_vk.device, q36_vk_ring[i].fence, NULL);
    }
    memset(q36_vk_ring, 0, sizeof(q36_vk_ring));
    q36_vk_ring_cur = 0;
    if (q36_vk.command_pool) vkDestroyCommandPool(q36_vk.device, q36_vk.command_pool, NULL);
    if (q36_vk.device) vkDestroyDevice(q36_vk.device, NULL);
    if (q36_vk.instance) vkDestroyInstance(q36_vk.instance, NULL);
    memset(&q36_vk, 0, sizeof(q36_vk));
    pthread_mutex_unlock(&q36_vk_mu);
    return 0;
}

static void q36_vk_prof_report(void) {
    fprintf(stderr,
            "q36: Vulkan prof dispatches=%" PRIu64 " record_ms=%.1f flushes=%" PRIu64 " submit_wait_ms=%.1f\n",
            q36_vk_prof_dispatches,
            (double)q36_vk_prof_record_ns / 1e6,
            q36_vk_prof_flushes,
            (double)q36_vk_prof_submit_ns / 1e6);
    if (q36_vk_prof_op_n != 0) {
        fprintf(stderr, "q36: Vulkan op profile\n");
        for (uint32_t i = 0; i < q36_vk_prof_op_n; i++) {
            const q36_vk_prof_op_row *r = &q36_vk_prof_op_rows[i];
            fprintf(stderr,
                    "q36:   op %-28s dispatches=%" PRIu64 " flushes=%" PRIu64 " groups=%" PRIu64
                    " record_ms=%.3f submit_wait_ms=%.3f gpu_ms=%.3f wall_ms=%.3f\n",
                    r->name,
                    r->dispatches,
                    r->flushes,
                    r->groups,
                    (double)r->record_ns / 1e6,
                    (double)r->submit_ns / 1e6,
                    (double)r->gpu_ns / 1e6,
                    (double)r->wall_ns / 1e6);
        }
    }
    if (q36_vk_prof_kernel_n != 0) {
        uint64_t total_ns = 0;
        for (uint32_t i = 0; i < q36_vk_prof_kernel_n; i++) total_ns += q36_vk_prof_kernel_rows[i].gpu_ns;
        fprintf(stderr, "q36: Vulkan kernel GPU time total_ms=%.3f\n", (double)total_ns / 1e6);
        for (uint32_t i = 0; i < q36_vk_prof_kernel_n; i++) {
            const q36_vk_prof_kernel_row *r = &q36_vk_prof_kernel_rows[i];
            const double pct = total_ns ? 100.0 * (double)r->gpu_ns / (double)total_ns : 0.0;
            fprintf(stderr,
                    "q36:   %-32s dispatches=%" PRIu64 " groups=%" PRIu64 " gpu_ms=%.3f pct=%.1f\n",
                    r->path,
                    r->dispatches,
                    r->groups,
                    (double)r->gpu_ns / 1e6,
                    pct);
        }
    }
}

void q36_gpu_cleanup(void) {
    pthread_mutex_lock(&q36_vk_mu);
    if (q36_vk.device) vkDeviceWaitIdle(q36_vk.device);
    q36_vk_stream_profile_report();
    q36_vk_batch_recording = false;
    q36_vk_batch_count = 0;
    q36_vk_prof_query_count = 0;
    q36_vk_batch_written_n = 0;
    q36_vk_batch_dirty_all = false;
    q36_vk_submit_seq = 0;
    q36_vk_completed_seq = 0;
    q36_vk_hazard_reset();
    while (q36_vk_retired_n) q36_vk_tensor_free_unlocked(q36_vk_retired[--q36_vk_retired_n]);
    while (q36_vk_pool_n) q36_vk_tensor_free_unlocked(q36_vk_pool[--q36_vk_pool_n]);
    q36_vk_weight_cache_free_unlocked();
    q36_vk_tensor_free_unlocked(q36_vk.iq_tables);
    q36_vk.iq_tables = NULL;
    q36_vk_tensor_free_unlocked(q36_vk.attn_part);
    q36_vk.attn_part = NULL;
    q36_vk.attn_part_bytes = 0;
    q36_vk_kernel_destroy(&q36_vk.ffn_tail);
    q36_vk_kernel_destroy(&q36_vk.moe_reduce);
    q36_vk_kernel_destroy(&q36_vk.moe_matvec_fast);
    q36_vk_kernel_destroy(&q36_vk.moe_matvec);
    q36_vk_kernel_destroy(&q36_vk.moe_gate_up);
    q36_vk_kernel_destroy(&q36_vk.router_topk);
    q36_vk_kernel_destroy(&q36_vk.moe_tiles);
    q36_vk_kernel_destroy(&q36_vk.kv_store);
    q36_vk_kernel_destroy(&q36_vk.kv_store_quant);
    q36_vk_kernel_destroy(&q36_vk.rms_norm_rope_kv_quant);
    q36_vk_kernel_destroy(&q36_vk.moe_gate_up_f32b);
    q36_vk_kernel_destroy(&q36_vk.moe_down_q2k_f32b);
    q36_vk_kernel_destroy(&q36_vk.moe_down_q2k_sum_decode);
    q36_vk_kernel_destroy(&q36_vk.moe_gate_up_gemm);
    q36_vk_kernel_destroy(&q36_vk.moe_down_gemm);
    q36_vk_kernel_destroy(&q36_vk.attn_decode_fused);
    q36_vk_kernel_destroy(&q36_vk.attn_decode_split);
    q36_vk_kernel_destroy(&q36_vk.attn_prefill_qtile);
    q36_vk_kernel_destroy(&q36_vk.attn_combine);
    q36_vk_kernel_destroy(&q36_vk.attn_reduce);
    q36_vk_kernel_destroy(&q36_vk.attn_post);
    q36_vk_kernel_destroy(&q36_vk.attn_scores);
    q36_vk_kernel_destroy(&q36_vk.delta_net_cols);
    q36_vk_kernel_destroy(&q36_vk.delta_net_decode);
    q36_vk_kernel_destroy(&q36_vk.delta_net_fast);
    q36_vk_kernel_destroy(&q36_vk.delta_net);
    q36_vk_kernel_destroy(&q36_vk.delta_gates);
    q36_vk_kernel_destroy(&q36_vk.delta_qk);
    q36_vk_kernel_destroy(&q36_vk.delta_qkv);
    q36_vk_kernel_destroy(&q36_vk.recur_conv_silu_decode);
    q36_vk_kernel_destroy(&q36_vk.conv_silu);
    q36_vk_kernel_destroy(&q36_vk.recur_window);
    q36_vk_kernel_destroy(&q36_vk.copy_rows);
    q36_vk_kernel_destroy(&q36_vk.rms_norm_rope_kv);
    q36_vk_kernel_destroy(&q36_vk.rms_norm_rope);
    q36_vk_kernel_destroy(&q36_vk.rope);
    q36_vk_kernel_destroy(&q36_vk.swiglu);
    q36_vk_kernel_destroy(&q36_vk.rms_norm);
    q36_vk_kernel_destroy(&q36_vk.add_rms_norm);
    q36_vk_kernel_destroy(&q36_vk.quantize_q8_0);
    q36_vk_kernel_destroy(&q36_vk.matmul_q6k_mmq_fast);
    q36_vk_kernel_destroy(&q36_vk.matmul_q5k_mmq_fast);
    q36_vk_kernel_destroy(&q36_vk.matmul_q6k_mmq);
    q36_vk_kernel_destroy(&q36_vk.matmul_q5k_mmq);
    q36_vk_kernel_destroy(&q36_vk.matmul_q8_0_decode_b64);
    q36_vk_kernel_destroy(&q36_vk.matmul_q8_0_decode_q36);
    q36_vk_kernel_destroy(&q36_vk.matmul_q8_0_decode);
    q36_vk_kernel_destroy(&q36_vk.matmul_q8_0_f32b);
    q36_vk_kernel_destroy(&q36_vk.shared_down_tail_decode);
    q36_vk_kernel_destroy(&q36_vk.shared_gate_up_decode);
    q36_vk_kernel_destroy(&q36_vk.matmul_q8_0_mm);
    q36_vk_kernel_destroy(&q36_vk.matmul_q8_0_mm_f16);
    q36_vk_kernel_destroy(&q36_vk.matmul_q8_0_mm_f16_out32);
    q36_vk_kernel_destroy(&q36_vk.matmul_q8_0_f32b_nx);
    q36_vk_kernel_destroy(&q36_vk.matmul_q8_0_q36);
    q36_vk_kernel_destroy(&q36_vk.matmul_q8_0);
    q36_vk_kernel_destroy(&q36_vk.matmul_kquant);
    q36_vk_kernel_destroy(&q36_vk.quantize_q8_k);
    q36_vk_kernel_destroy(&q36_vk.directional_steering);
    q36_vk_kernel_destroy(&q36_vk.add);
    q36_vk_kernel_destroy(&q36_vk.matmul_f32);
    q36_vk_kernel_destroy(&q36_vk.matmul_f32_fast);
    q36_vk_kernel_destroy(&q36_vk.matmul_f16);
    for (uint32_t i = 0; i < Q36_VK_CB_RING; i++) {
        if (q36_vk_ring[i].pool) vkDestroyDescriptorPool(q36_vk.device, q36_vk_ring[i].pool, NULL);
        if (q36_vk_ring[i].fence) vkDestroyFence(q36_vk.device, q36_vk_ring[i].fence, NULL);
    }
    memset(q36_vk_ring, 0, sizeof(q36_vk_ring));
    q36_vk_ring_cur = 0;
    if (q36_vk.query_pool) vkDestroyQueryPool(q36_vk.device, q36_vk.query_pool, NULL);
    if (q36_vk.command_pool) vkDestroyCommandPool(q36_vk.device, q36_vk.command_pool, NULL);
    if (q36_vk.device) vkDestroyDevice(q36_vk.device, NULL);
    if (q36_vk.instance) vkDestroyInstance(q36_vk.instance, NULL);
    memset(&q36_vk, 0, sizeof(q36_vk));
    q36_gpu_model_map = NULL;
    q36_gpu_model_size = 0;
    q36_gpu_model_fd = -1;
    q36_gpu_quality = false;
    q36_gpu_commands_open = false;
    q36_vk_moe_bank_cache = true;
    pthread_mutex_unlock(&q36_vk_mu);
    q36_vk_host_threads_shutdown();
}

q36_gpu_tensor *q36_gpu_tensor_alloc(uint64_t bytes) {
    if (!q36_gpu_init()) return NULL;
    pthread_mutex_lock(&q36_vk_mu);
    q36_gpu_tensor *tensor = NULL;
    const uint64_t alloc_bytes = bytes ? q36_round_up_u64(bytes, 4) : 4;
    for (uint32_t i = 0; i < q36_vk_pool_n; i++) {
        if (q36_vk_pool[i]->alloc_bytes != alloc_bytes) continue;
        tensor = q36_vk_pool[i];
        q36_vk_pool[i] = q36_vk_pool[--q36_vk_pool_n];
        tensor->bytes = bytes;
        tensor->gpu_written = false;
        tensor->last_use_seq = 0;
        memset(tensor->data, 0, (size_t)alloc_bytes);
        q36_gpu_live_bytes += bytes;
        if (q36_gpu_live_bytes > q36_gpu_peak_bytes) q36_gpu_peak_bytes = q36_gpu_live_bytes;
        break;
    }
    if (!tensor) tensor = q36_vk_tensor_alloc_kind_unlocked(bytes, false);
    pthread_mutex_unlock(&q36_vk_mu);
    return tensor;
}

q36_gpu_tensor *q36_gpu_tensor_view(const q36_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!q36_gpu_tensor_range_ok(base, offset, bytes)) return NULL;
    q36_gpu_tensor *view = calloc(1, sizeof(*view));
    if (!view) return NULL;
    view->bytes = bytes;
    view->offset = offset;
    view->owner = (q36_gpu_tensor *)base;
    return view;
}

void q36_gpu_tensor_free(q36_gpu_tensor *tensor) {
    if (!tensor) return;
    pthread_mutex_lock(&q36_vk_mu);
    if (tensor->owner) {
        q36_vk_tensor_free_unlocked(tensor);
    } else {
        q36_vk_tensor_release_unlocked(tensor);
    }
    pthread_mutex_unlock(&q36_vk_mu);
}

uint64_t q36_gpu_tensor_bytes(const q36_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

/* The lazy-submission sync point: any host access to tensor bytes lands
 * here first, so flushing the open batch makes every host-side op see
 * completed GPU results without explicit synchronization at call sites. */
void *q36_gpu_tensor_contents(q36_gpu_tensor *tensor) {
    uint64_t offset = 0;
    q36_gpu_tensor *root = q36_gpu_tensor_root(tensor, &offset);
    if (!root || !root->data || offset > root->bytes) return NULL;
    bool flush = false;
    pthread_mutex_lock(&q36_vk_mu);
    if (q36_vk_gpu_busy_unlocked()) {
        if (root->gpu_written) {
            /* dirty_all only means the written set overflowed; a root no
             * dispatch ever wrote cannot have pending GPU writes. */
            flush = q36_vk_batch_dirty_all;
            for (uint32_t i = 0; !flush && i < q36_vk_batch_written_n; i++) {
                flush = q36_vk_batch_written[i] == root;
            }
        } else {
            /* Host-written staging the GPU only reads (embed stage, conv
             * window): the caller may overwrite it, so wait out a pending
             * read; with double-buffered stages this is almost never hit. */
            flush = root->last_use_seq > q36_vk_completed_seq;
        }
    }
    pthread_mutex_unlock(&q36_vk_mu);
    if (flush) {
        pthread_mutex_lock(&q36_vk_mu);
        int ok = q36_vk_flush_reason_unlocked(q36_vk_tensor_contents_reason);
        pthread_mutex_unlock(&q36_vk_mu);
        if (!ok) return NULL;
    }
    return root->data + offset;
}

static void *q36_vk_tensor_contents_labeled(q36_gpu_tensor *tensor, const char *reason) {
    const char *prev = q36_vk_tensor_contents_reason;
    q36_vk_tensor_contents_reason = reason ? reason : "submit_wait_tensor_contents";
    void *p = q36_gpu_tensor_contents(tensor);
    q36_vk_tensor_contents_reason = prev;
    return p;
}

void *q36_gpu_tensor_contents_named(q36_gpu_tensor *tensor, const char *reason) {
    return q36_vk_tensor_contents_labeled(tensor, reason);
}

int q36_gpu_tensor_write(q36_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    unsigned char *dst = q36_vk_tensor_contents_labeled(tensor, "submit_wait_tensor_write");
    if (!dst || !data || !q36_gpu_tensor_range_ok(tensor, offset, bytes)) return 0;
    memcpy(dst + offset, data, (size_t)bytes);
    return 1;
}

int q36_gpu_tensor_read(const q36_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    unsigned char *src = q36_vk_tensor_contents_labeled((q36_gpu_tensor *)tensor, "submit_wait_tensor_read");
    if (!src || !data || !q36_gpu_tensor_range_ok(tensor, offset, bytes)) return 0;
    memcpy(data, src + offset, (size_t)bytes);
    return 1;
}

int q36_gpu_tensor_copy(q36_gpu_tensor *dst, uint64_t dst_offset,
                        const q36_gpu_tensor *src, uint64_t src_offset,
                        uint64_t bytes) {
    if (!bytes) return 1;
    if (!q36_gpu_tensor_range_ok(dst, dst_offset, bytes) ||
        !q36_gpu_tensor_range_ok(src, src_offset, bytes)) return 0;
    if (q36_vk.ready && dst && src) {
        uint64_t root_dst_offset = 0;
        uint64_t root_src_offset = 0;
        q36_gpu_tensor *dst_root = q36_gpu_tensor_root(dst, &root_dst_offset);
        q36_gpu_tensor *src_root = q36_gpu_tensor_root(src, &root_src_offset);
        root_dst_offset += dst_offset;
        root_src_offset += src_offset;
        bool overlap = dst_root == src_root &&
                       root_dst_offset < root_src_offset + bytes &&
                       root_src_offset < root_dst_offset + bytes;
        if (dst_root && src_root && dst_root->buffer && src_root->buffer && !overlap) {
            pthread_mutex_lock(&q36_vk_mu);
            VkCommandBuffer cmd;
            int ok = q36_vk_begin_batch_unlocked(&cmd);
            if (ok) {
                bool need_barrier = q36_vk_hazard_all ||
                    q36_vk_hazard_in(q36_vk_hazard_written, q36_vk_hazard_written_n, src_root) ||
                    q36_vk_hazard_in(q36_vk_hazard_written, q36_vk_hazard_written_n, dst_root) ||
                    q36_vk_hazard_in(q36_vk_hazard_read, q36_vk_hazard_read_n, dst_root);
                if (need_barrier) {
                    VkMemoryBarrier barrier = {
                        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                    };
                    vkCmdPipelineBarrier(cmd,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         0, 1, &barrier, 0, NULL, 0, NULL);
                    q36_vk_hazard_reset();
                }
            }
            if (ok) {
                VkBufferCopy copy = {
                    .srcOffset = root_src_offset,
                    .dstOffset = root_dst_offset,
                    .size = bytes,
                };
                vkCmdCopyBuffer(cmd, src_root->buffer, dst_root->buffer, 1, &copy);
                src_root->last_use_seq = q36_vk_submit_seq + 1;
                dst_root->last_use_seq = q36_vk_submit_seq + 1;
                dst_root->gpu_written = true;
                q36_vk_hazard_add(q36_vk_hazard_read, &q36_vk_hazard_read_n, src_root);
                q36_vk_hazard_add(q36_vk_hazard_written, &q36_vk_hazard_written_n, dst_root);
                q36_vk_note_written_tensor(dst_root);
                q36_vk_prof_dispatches++;
                q36_vk_prof_op_add("tensor_copy", 1, 0, 1, 0, 0, 0, 0);
                if (++q36_vk_batch_count >= Q36_VK_BATCH_FLUSH_AT) ok = q36_vk_submit_eager_unlocked();
            }
            pthread_mutex_unlock(&q36_vk_mu);
            return ok;
        }
    }
    unsigned char *dstp = q36_vk_tensor_contents_labeled(dst, "submit_wait_tensor_copy_dst");
    unsigned char *srcp = q36_vk_tensor_contents_labeled((q36_gpu_tensor *)src, "submit_wait_tensor_copy_src");
    if (!dstp || !srcp) return 0;
    memmove(dstp + dst_offset, srcp + src_offset, (size_t)bytes);
    return 1;
}

int q36_gpu_begin_commands(void) {
    q36_gpu_commands_open = true;
    return 1;
}

int q36_gpu_flush_commands(void) {
    return q36_gpu_commands_open ? 1 : 0;
}

int q36_gpu_end_commands(void) {
    if (!q36_gpu_commands_open) return 0;
    q36_gpu_commands_open = false;
    return 1;
}

int q36_gpu_synchronize(void) {
    if (!q36_vk.ready) return 1;
    pthread_mutex_lock(&q36_vk_mu);
    int ok = q36_vk_flush_reason_unlocked("submit_wait_explicit");
    pthread_mutex_unlock(&q36_vk_mu);
    return ok && vkDeviceWaitIdle(q36_vk.device) == VK_SUCCESS;
}

int q36_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    if (!model_map || model_size == 0) return 0;
    pthread_mutex_lock(&q36_vk_mu);
    q36_vk_weight_cache_free_unlocked();
    q36_gpu_model_map = model_map;
    q36_gpu_model_size = model_size;
    pthread_mutex_unlock(&q36_vk_mu);
    return 1;
}

int q36_gpu_set_model_fd(int fd) {
    q36_gpu_model_fd = fd;
    return fd >= 0;
}

int q36_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    if (!model_map || map_offset > model_size || map_size > model_size - map_offset) return 0;
    pthread_mutex_lock(&q36_vk_mu);
    q36_vk_weight_cache_free_unlocked();
    q36_gpu_model_map = (const unsigned char *)model_map + map_offset;
    q36_gpu_model_size = map_size;
    pthread_mutex_unlock(&q36_vk_mu);
    return 1;
}

int q36_gpu_set_model_map_spans(const void *model_map, uint64_t model_size, const uint64_t *offsets, const uint64_t *sizes, uint32_t count, uint64_t max_tensor_bytes) {
    (void)offsets;
    (void)sizes;
    (void)count;
    (void)max_tensor_bytes;
    return q36_gpu_set_model_map(model_map, model_size);
}

int q36_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label) {
    (void)label;
    const unsigned char *source = q36_gpu_weight_bytes(model_map, model_size, offset, bytes);
    int ok;
    if (!source) return 0;
    pthread_mutex_lock(&q36_vk_mu);
    q36_vk_weight_copy_parallel = true;
    ok = q36_vk_weight_get_unlocked(source, bytes) != NULL;
    q36_vk_weight_copy_parallel = false;
    pthread_mutex_unlock(&q36_vk_mu);
    return ok;
}

int q36_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label) {
    (void)in_dim;
    (void)out_dim;
    return q36_gpu_cache_model_range(model_map, model_size, offset, bytes, label);
}

void q36_gpu_set_quality(bool quality) {
    q36_gpu_quality = quality;
}

void q36_gpu_set_ssd_streaming(bool enabled) {
    pthread_mutex_lock(&q36_vk_mu);
    if (q36_vk_stream.enabled != enabled) {
        q36_vk_stream.runtime_cap = 0;
        q36_vk_stream.allocation_failures = 0;
        q36_vk_stream_cache_release_unlocked(true);
        q36_vk_stream.enabled = enabled;
    }
    q36_vk_moe_bank_cache = !enabled;
    pthread_mutex_unlock(&q36_vk_mu);
}

void q36_gpu_set_streaming_full_layers(uint32_t layers) {
    pthread_mutex_lock(&q36_vk_mu);
    q36_vk_stream.full_layers = layers;
    q36_vk_moe_bank_cache = !q36_vk_stream.enabled || layers != 0;
    pthread_mutex_unlock(&q36_vk_mu);
}

void q36_gpu_set_streaming_expert_cache_budget(uint32_t experts) {
    pthread_mutex_lock(&q36_vk_mu);
    if (q36_vk_stream.budget != experts) {
        q36_vk_stream.runtime_cap = 0;
        q36_vk_stream.allocation_failures = 0;
        q36_vk_stream_cache_release_unlocked(true);
        q36_vk_stream.budget = experts;
    }
    pthread_mutex_unlock(&q36_vk_mu);
}

void q36_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes) {
    pthread_mutex_lock(&q36_vk_mu);
    if (q36_vk_stream.configured_expert_bytes != bytes) {
        q36_vk_stream.runtime_cap = 0;
        q36_vk_stream.allocation_failures = 0;
        q36_vk_stream_cache_release_unlocked(true);
        q36_vk_stream.configured_expert_bytes = bytes;
    }
    pthread_mutex_unlock(&q36_vk_mu);
}

uint64_t q36_gpu_recommended_working_set_size(void) {
    uint64_t best = 0;
    if (!q36_vk.ready) return 0;
    for (uint32_t i = 0; i < q36_vk.mem_props.memoryHeapCount; i++) {
        if ((q36_vk.mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) continue;
        if (q36_vk.mem_props.memoryHeaps[i].size > best) best = q36_vk.mem_props.memoryHeaps[i].size;
    }
    if (best != 0) return best;
    for (uint32_t i = 0; i < q36_vk.mem_props.memoryHeapCount; i++) {
        if (q36_vk.mem_props.memoryHeaps[i].size > best) best = q36_vk.mem_props.memoryHeaps[i].size;
    }
    return best;
}

uint32_t q36_gpu_stream_expert_cache_configured_count(void) {
    return q36_vk_stream.runtime_cap ? q36_vk_stream.runtime_cap : q36_vk_stream.budget;
}

uint32_t q36_gpu_stream_expert_cache_current_count(void) {
    pthread_mutex_lock(&q36_vk_mu);
    uint32_t count = 0;
    for (uint32_t i = 0; i < q36_vk_stream.cap; i++) count += q36_vk_stream.entry[i].valid ? 1u : 0u;
    pthread_mutex_unlock(&q36_vk_mu);
    return count;
}

void q36_gpu_stream_expert_cache_reset_route_hotness(void) {
    pthread_mutex_lock(&q36_vk_mu);
    memset(q36_vk_stream.route_hotness, 0, sizeof(q36_vk_stream.route_hotness));
    q36_vk_stream.hotness_decay_token = q36_vk_stream.route_tokens;
    pthread_mutex_unlock(&q36_vk_mu);
}

void q36_gpu_stream_expert_cache_note_tokens(uint32_t n_tokens) {
    if (!q36_vk_stream.enabled || n_tokens == 0) return;
    pthread_mutex_lock(&q36_vk_mu);
    if (q36_vk_stream.route_tokens <= UINT64_MAX - n_tokens) {
        q36_vk_stream.route_tokens += n_tokens;
    } else {
        q36_vk_stream.route_tokens = UINT64_MAX;
    }
    q36_vk_stream_decay_hotness_unlocked();
    pthread_mutex_unlock(&q36_vk_mu);
}

void q36_gpu_stream_expert_cache_release_resident(void) {
    pthread_mutex_lock(&q36_vk_mu);
    q36_vk_stream_cache_release_unlocked(true);
    pthread_mutex_unlock(&q36_vk_mu);
}

uint32_t q36_gpu_stream_expert_cache_budget_for_expert_size(uint64_t gate_expert_bytes, uint64_t up_expert_bytes, uint64_t down_expert_bytes) {
    if (gate_expert_bytes == 0 || up_expert_bytes == 0 || down_expert_bytes == 0) return 0;
    uint64_t total = gate_expert_bytes + up_expert_bytes;
    if (total < gate_expert_bytes || total > UINT64_MAX - down_expert_bytes) return 0;
    total += down_expert_bytes;
    if (q36_vk_stream.configured_expert_bytes != 0 && q36_vk_stream.configured_expert_bytes != total) return 0;
    return q36_vk_stream.runtime_cap ? q36_vk_stream.runtime_cap : q36_vk_stream.budget;
}

int q36_gpu_stream_expert_cache_seed_selected(const q36_gpu_stream_expert_table *table, const int32_t *selected_ids, uint32_t n_selected) {
    return q36_vk_stream_cache_prepare_i32(table, selected_ids, NULL, n_selected, NULL) ? 1 : 0;
}

int q36_gpu_stream_expert_cache_begin_selected_load(const q36_gpu_stream_expert_table *table, const int32_t *selected_ids, uint32_t n_selected) {
    return q36_gpu_stream_expert_cache_seed_selected(table, selected_ids, n_selected);
}

int q36_gpu_stream_expert_cache_prepare_selected_batch(const q36_gpu_stream_expert_table *table, const int32_t *selected_ids, uint32_t n_tokens, uint32_t n_selected) {
    uint64_t n = (uint64_t)n_tokens * n_selected;
    if (n > UINT32_MAX) return 0;
    return q36_vk_stream_cache_prepare_i32(table, selected_ids, NULL, (uint32_t)n, NULL) ? 1 : 0;
}

int q36_gpu_stream_expert_cache_load_layer(const q36_gpu_stream_expert_table *table) {
    if (!table || table->n_total_expert == 0) return 0;
    uint32_t n = table->n_total_expert;
    uint32_t budget = q36_gpu_stream_expert_cache_configured_count();
    if (budget != 0 && n > budget) n = budget;
    int32_t *ids = malloc((size_t)n * sizeof(ids[0]));
    if (!ids) return 0;
    for (uint32_t i = 0; i < n; i++) ids[i] = (int32_t)i;
    int ok = q36_vk_stream_cache_prepare_i32(table, ids, NULL, n, NULL) ? 1 : 0;
    free(ids);
    return ok;
}

int q36_gpu_stream_expert_cache_seed_from_layer_selected(const q36_gpu_stream_expert_table *table, const q36_gpu_tensor *selected, uint32_t n_tokens, uint32_t n_seed_tokens, uint32_t n_selected) {
    if (!selected || n_selected == 0) return 0;
    if (n_seed_tokens > n_tokens) n_seed_tokens = n_tokens;
    uint64_t n = (uint64_t)n_seed_tokens * n_selected;
    if (n > UINT32_MAX || !q36_gpu_tensor_range_ok(selected, 0, n * sizeof(int32_t))) return 0;
    const int32_t *ids = q36_vk_tensor_contents_labeled((q36_gpu_tensor *)selected, "submit_wait_stream_seed_selected");
    if (!ids) return 0;
    return q36_vk_stream_cache_prepare_i32(table, ids, NULL, (uint32_t)n, NULL) ? 1 : 0;
}

int q36_gpu_stream_expert_cache_release_layer_cache(void) {
    return 1;
}

int q36_gpu_stream_expert_cache_seed_experts(const q36_gpu_stream_expert_table *table, const int32_t *expert_ids, const uint32_t *expert_priorities, uint32_t n_experts) {
    return q36_vk_stream_cache_prepare_i32(table, expert_ids, expert_priorities, n_experts, NULL) ? 1 : 0;
}

void q36_gpu_print_memory_report(const char *label) {
    fprintf(stderr,
            "q36: Vulkan memory%s%s live=%" PRIu64 " peak=%" PRIu64 " fd=%d quality=%s ssd_streaming=%s cache=%u/%u requested=%u allocation_failures=%u lookup=%s lookup_steps=%" PRIu64 " hits=%" PRIu64 " misses=%" PRIu64 " loads=%" PRIu64 " evictions=%" PRIu64 " device=%s api=%u.%u.%u\n",
            label ? " " : "",
            label ? label : "",
            q36_gpu_live_bytes,
            q36_gpu_peak_bytes,
            q36_gpu_model_fd,
            q36_gpu_quality ? "high" : "fast",
            q36_vk_stream.enabled ? "on" : "off",
            q36_gpu_stream_expert_cache_current_count(),
            q36_vk_stream.cap ? q36_vk_stream.cap : q36_gpu_stream_expert_cache_configured_count(),
            q36_vk_stream.budget,
            q36_vk_stream.allocation_failures,
            q36_vk_stream.indexed ? "index" : "scan",
            q36_vk_stream.lookup_steps,
            q36_vk_stream.hits,
            q36_vk_stream.misses,
            q36_vk_stream.loads,
            q36_vk_stream.evictions,
            q36_vk.props.deviceName[0] ? q36_vk.props.deviceName : "uninitialized",
            VK_VERSION_MAJOR(q36_vk.api_version),
            VK_VERSION_MINOR(q36_vk.api_version),
            VK_VERSION_PATCH(q36_vk.api_version));
}

void q36_gpu_prof_reset(void) {
    pthread_mutex_lock(&q36_vk_mu);
    (void)q36_vk_flush_reason_unlocked("submit_wait_prof_reset");
    q36_vk_prof_reset();
    pthread_mutex_unlock(&q36_vk_mu);
}

void q36_gpu_prof_report(const char *label) {
    pthread_mutex_lock(&q36_vk_mu);
    (void)q36_vk_flush_reason_unlocked("submit_wait_prof_report");
    pthread_mutex_unlock(&q36_vk_mu);
    if (label && label[0]) fprintf(stderr, "q36: Vulkan prof label=%s\n", label);
    q36_vk_prof_report();
}

static int q36_vk_matmul_dense(q36_vk_kernel *kernel,
                               q36_gpu_tensor *out,
                               const void *model_map,
                               uint64_t model_size,
                               uint64_t weight_offset,
                               uint64_t weight_bytes,
                               uint64_t in_dim,
                               uint64_t out_dim,
                               const q36_gpu_tensor *x,
                               uint64_t n_tok,
                               float scale) {
    uint64_t out_bytes = 0;
    uint64_t in_bytes = 0;
    if (!q36_u64_mul_ok(n_tok, out_dim, &out_bytes) || !q36_u64_mul_ok(out_bytes, sizeof(float), &out_bytes)) return 0;
    if (!q36_u64_mul_ok(n_tok, in_dim, &in_bytes) || !q36_u64_mul_ok(in_bytes, sizeof(float), &in_bytes)) return 0;
    if (!q36_gpu_tensor_range_ok(out, 0, out_bytes) || !q36_gpu_tensor_range_ok(x, 0, in_bytes)) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;

    const unsigned char *source = q36_gpu_weight_bytes(model_map, model_size, weight_offset, weight_bytes);
    if (!source) return 0;

    pthread_mutex_lock(&q36_vk_mu);
    q36_gpu_tensor *weights = q36_vk_weight_get_unlocked(source, weight_bytes);
    if (!weights) {
        pthread_mutex_unlock(&q36_vk_mu);
        return 0;
    }

    struct {
        uint32_t in_dim;
        uint32_t out_dim;
        uint32_t n_tok;
        float scale;
    } push = { (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)n_tok, scale };
    const q36_gpu_tensor *bindings[3] = { weights, x, out };
    /* out_dim workgroups; the f16/f32 shaders index rows by global id, so
     * trailing groups simply exit.  These dense paths see little runtime
     * use: f32 matvecs run on the host and this model has no f16 matvec. */
    int ok = q36_vk_run_unlocked(kernel == &q36_vk.matmul_f16 ? "dense_f16" : "dense_f32",
                                 kernel, bindings, &push, sizeof(push),
                                 (uint32_t)out_dim, (uint32_t)n_tok, 1);
    pthread_mutex_unlock(&q36_vk_mu);
    return ok;
}


int q36_gpu_matmul_f16_tensor(q36_gpu_tensor *out,
                              const void *model_map,
                              uint64_t model_size,
                              uint64_t weight_offset,
                              uint64_t in_dim,
                              uint64_t out_dim,
                              const q36_gpu_tensor *x,
                              uint64_t n_tok) {
    uint64_t elems = 0;
    if (!q36_u64_mul_ok(in_dim, out_dim, &elems)) return 0;
    return q36_vk_matmul_dense(&q36_vk.matmul_f16, out, model_map, model_size, weight_offset,
                               elems * sizeof(uint16_t), in_dim, out_dim, x, n_tok, 1.0f);
}

/* The f32 tensors in this model are the small control projections (expert
 * router logits, delta-net alpha/beta, the shared-expert gate scalar):
 * out_dim is 1..256, far too little parallelism to feed the GPU, and the
 * CPU-exact serial fp64 chain is latency-bound there.  Running the same
 * loop as q36_tensor_matvec() on the host is faster than a dispatch and,
 * because the inputs are host-produced norms, costs no batch flush at
 * all -- the whole router stays off the GPU timeline. */
typedef struct {
    const float *w;
    const float *x;
    float *out;
    uint64_t in_dim;
    uint64_t out_dim;
    float scale;
} q36_vk_matmul_f32_ctx;

static void q36_vk_matmul_f32_rows(void *opaque, uint64_t row0, uint64_t row1) {
    q36_vk_matmul_f32_ctx *ctx = (q36_vk_matmul_f32_ctx *)opaque;

    for (uint64_t tok = row0; tok < row1; tok++) {
        const float *xt = ctx->x + tok * ctx->in_dim;
        float *ot = ctx->out + tok * ctx->out_dim;
        uint64_t row = 0;
        /* Four independent row chains per pass: each keeps the exact
         * sequential rounding of q36_tensor_matvec(), the interleave only
         * hides the fp64 add latency. */
        for (; row + 4 <= ctx->out_dim; row += 4) {
            const float *w0 = ctx->w + row * ctx->in_dim;
            const float *w1 = w0 + ctx->in_dim;
            const float *w2 = w1 + ctx->in_dim;
            const float *w3 = w2 + ctx->in_dim;
            double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
            for (uint64_t i = 0; i < ctx->in_dim; i++) {
                double xv = (double)xt[i];
                a0 += (double)w0[i] * xv;
                a1 += (double)w1[i] * xv;
                a2 += (double)w2[i] * xv;
                a3 += (double)w3[i] * xv;
            }
            ot[row] = (float)a0 * ctx->scale;
            ot[row + 1] = (float)a1 * ctx->scale;
            ot[row + 2] = (float)a2 * ctx->scale;
            ot[row + 3] = (float)a3 * ctx->scale;
        }
        for (; row < ctx->out_dim; row++) {
            const float *wr = ctx->w + row * ctx->in_dim;
            double acc = 0.0;
            for (uint64_t i = 0; i < ctx->in_dim; i++) acc += (double)wr[i] * (double)xt[i];
            ot[row] = (float)acc * ctx->scale;
        }
    }
}

int q36_gpu_matmul_f32_scaled_tensor(q36_gpu_tensor *out,
                                     const void *model_map,
                                     uint64_t model_size,
                                     uint64_t weight_offset,
                                     uint64_t in_dim,
                                     uint64_t out_dim,
                                     const q36_gpu_tensor *x,
                                     uint64_t n_tok,
                                     float scale) {
    uint64_t elems = 0;
    uint64_t out_bytes = 0;
    uint64_t in_bytes = 0;
    if (!q36_u64_mul_ok(in_dim, out_dim, &elems)) return 0;
    if (!q36_u64_mul_ok(n_tok, out_dim, &out_bytes) || !q36_u64_mul_ok(out_bytes, sizeof(float), &out_bytes)) return 0;
    if (!q36_u64_mul_ok(n_tok, in_dim, &in_bytes) || !q36_u64_mul_ok(in_bytes, sizeof(float), &in_bytes)) return 0;
    if (!q36_gpu_tensor_range_ok(out, 0, out_bytes) || !q36_gpu_tensor_range_ok(x, 0, in_bytes)) return 0;

    /* Fast mode keeps these small control matvecs on GPU: a host fallback
     * here would force a flush now that nothing else reads activations back
     * during decode.  Default is the f32-FMA kernel; Q36_VK_F32_FAST=0
     * restores the fp64 shader whose 29 guard bits make a router near-tie
     * flip impossible (--quality keeps the host-exact route). */
    if (!q36_gpu_quality &&
        q36_vk_matmul_dense(q36_vk_use_f32_fast() ? &q36_vk.matmul_f32_fast : &q36_vk.matmul_f32,
                            out, model_map, model_size, weight_offset,
                            elems * sizeof(float), in_dim, out_dim, x, n_tok, scale)) {
        return 1;
    }

    const float *w = (const float *)q36_gpu_weight_bytes(model_map, model_size, weight_offset,
                                                         elems * sizeof(float));
    const float *x_host = q36_vk_tensor_contents_labeled((q36_gpu_tensor *)x, "submit_wait_matmul_f32_x");
    float *out_host = q36_vk_tensor_contents_labeled(out, "submit_wait_matmul_f32_out");
    q36_vk_matmul_f32_ctx ctx;
    if (!w || !x_host || !out_host) return 0;

    ctx.w = w;
    ctx.x = x_host;
    ctx.out = out_host;
    ctx.in_dim = in_dim;
    ctx.out_dim = out_dim;
    ctx.scale = scale;
    if (n_tok > 1) q36_gpu_parallel_for_rows(n_tok, 2, q36_vk_matmul_f32_rows, &ctx);
    else q36_vk_matmul_f32_rows(&ctx, 0, n_tok);
    return 1;
}

int q36_gpu_matmul_f32_tensor(q36_gpu_tensor *out,
                              const void *model_map,
                              uint64_t model_size,
                              uint64_t weight_offset,
                              uint64_t in_dim,
                              uint64_t out_dim,
                              const q36_gpu_tensor *x,
                              uint64_t n_tok) {
    return q36_gpu_matmul_f32_scaled_tensor(out, model_map, model_size, weight_offset,
                                            in_dim, out_dim, x, n_tok, 1.0f);
}

typedef struct {
    unsigned char *xq;
    const float *x;
    uint64_t in_dim;
    uint64_t row_bytes;
} q36_vk_quantize_q8_0_ctx;

static void q36_vk_quantize_q8_0_rows(void *opaque, uint64_t row0, uint64_t row1) {
    q36_vk_quantize_q8_0_ctx *ctx = (q36_vk_quantize_q8_0_ctx *)opaque;

    for (uint64_t t = row0; t < row1; t++) {
        q36_quant_q8_0(ctx->x + t * ctx->in_dim,
                       ctx->xq + t * ctx->row_bytes,
                       (int64_t)ctx->in_dim);
    }
}

typedef struct {
    const unsigned char *w;
    const unsigned char *xq;
    float *out;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t n_tok;
    uint64_t row_bytes;
    float scale;
} q36_vk_matmul_q8_0_ctx;

static void q36_vk_matmul_q8_0_rows(void *opaque, uint64_t row0, uint64_t row1) {
    q36_vk_matmul_q8_0_ctx *ctx = (q36_vk_matmul_q8_0_ctx *)opaque;

    for (uint64_t row = row0; row < row1; row++) {
        const void *wr = ctx->w + row * ctx->row_bytes;
        for (uint64_t tok = 0; tok < ctx->n_tok; tok++) {
            float acc = q36_quant_dot_q8_0(wr,
                                           ctx->xq + tok * ctx->row_bytes,
                                           (int)ctx->in_dim);
            ctx->out[tok * ctx->out_dim + row] = acc * ctx->scale;
        }
    }
}

static const char *q36_vk_q8_0_op_name(uint64_t out_dim, uint64_t blocks, uint64_t n_tok) {
    uint64_t in_dim = blocks * Q36_VK_QK8_0;
    if (n_tok == 1) {
        if (out_dim == 2048 && in_dim == 4096) return "dense_q8_0_d_2048x4096";
        if (out_dim == 2048 && in_dim == 512) return "dense_q8_0_d_2048x512";
        if (out_dim == 32 && in_dim == 2048) return "dense_q8_0_d_32x2048";
        if (out_dim == 4096 && in_dim == 2048) return "dense_q8_0_d_4096x2048";
        if (out_dim == 512 && in_dim == 2048) return "dense_q8_0_d_512x2048";
        if (out_dim == 8192 && in_dim == 2048) return "dense_q8_0_d_8192x2048";
        if (in_dim == 2048 && out_dim > 100000) return "dense_q8_0_d_vocabx2048";
        return "dense_q8_0_d_other";
    }
    if (out_dim == 2048 && in_dim == 4096) return "dense_q8_0_p_2048x4096";
    if (out_dim == 2048 && in_dim == 512) return "dense_q8_0_p_2048x512";
    if (out_dim == 32 && in_dim == 2048) return "dense_q8_0_p_32x2048";
    if (out_dim == 4096 && in_dim == 2048) return "dense_q8_0_p_4096x2048";
    if (out_dim == 512 && in_dim == 2048) return "dense_q8_0_p_512x2048";
    if (out_dim == 8192 && in_dim == 2048) return "dense_q8_0_p_8192x2048";
    if (in_dim == 2048 && out_dim > 100000) return "dense_q8_0_p_vocabx2048";
    return "dense_q8_0_p_other";
}

static const char *q36_vk_q8_0_quant_wait_name(uint64_t out_dim, uint64_t blocks, uint64_t n_tok) {
    uint64_t in_dim = blocks * Q36_VK_QK8_0;
    if (n_tok == 1) {
        if (out_dim == 2048 && in_dim == 4096) return "submit_wait_q8_0_q_2048x4096";
        if (out_dim == 2048 && in_dim == 512) return "submit_wait_q8_0_q_2048x512";
        if (out_dim == 32 && in_dim == 2048) return "submit_wait_q8_0_q_32x2048";
        if (out_dim == 4096 && in_dim == 2048) return "submit_wait_q8_0_q_4096x2048";
        if (out_dim == 512 && in_dim == 2048) return "submit_wait_q8_0_q_512x2048";
        if (out_dim == 8192 && in_dim == 2048) return "submit_wait_q8_0_q_8192x2048";
        if (in_dim == 2048 && out_dim > 100000) return "submit_wait_q8_0_q_vocabx2048";
        return "submit_wait_q8_0_q_other";
    }
    return "submit_wait_q8_0_quant_x";
}

static int q36_vk_matmul_q8_0_gpu(q36_gpu_tensor *out,
                                  const unsigned char *w,
                                  uint64_t weight_bytes,
                                  const unsigned char *xq,
                                  uint64_t xq_bytes,
                                  uint64_t row_bytes,
                                  uint64_t out_dim,
                                  uint64_t n_tok,
                                  uint64_t blocks,
                                  float scale) {
    q36_gpu_tensor *xq_tensor = q36_gpu_tensor_alloc(xq_bytes);
    if (!xq_tensor) return 0;
    unsigned char *xq_host = q36_vk_tensor_contents_labeled(xq_tensor, "submit_wait_q8_0_stage_xq");
    if (!xq_host) {
        q36_gpu_tensor_free(xq_tensor);
        return 0;
    }
    memcpy(xq_host, xq, (size_t)xq_bytes);

    struct {
        uint32_t out_dim;
        uint32_t n_tok;
        uint32_t blocks;
        uint32_t row_bytes;
        float scale;
    } push = { (uint32_t)out_dim, (uint32_t)n_tok, (uint32_t)blocks, (uint32_t)row_bytes, scale };

    pthread_mutex_lock(&q36_vk_mu);
    q36_gpu_tensor *weights = q36_vk_weight_get_unlocked(w, weight_bytes);
    int ok = weights != NULL;
    if (ok) {
        const q36_gpu_tensor *bindings[3] = { weights, xq_tensor, out };
        const char *op = q36_vk.prof_ops ? q36_vk_q8_0_op_name(out_dim, blocks, n_tok) : "dense_q8_0";
        if (n_tok == 1) {
            q36_vk_kernel *kernel = &q36_vk.matmul_q8_0_decode;
            if (blocks == 64u && row_bytes == 64u * 34u) kernel = &q36_vk.matmul_q8_0_decode_b64;
            ok = q36_vk_run_unlocked(op, kernel, bindings, &push, sizeof(push),
                                     (uint32_t)((out_dim + 15u) / 16u), 1, 1);
        } else {
            ok = q36_vk_run_unlocked(op, &q36_vk.matmul_q8_0, bindings, &push, sizeof(push),
                                     (uint32_t)((out_dim + 7u) / 8u),
                                     (uint32_t)((n_tok + 3u) / 4u),
                                     1);
        }
    }
    pthread_mutex_unlock(&q36_vk_mu);
    q36_gpu_tensor_free(xq_tensor);
    return ok;
}

static bool q36_vk_use_q8_mm_f16(void) {
    return q36_vk.have_f16 && q36_vk_env_default_on("Q36_VK_Q8_MM_F16");
}

static bool q36_vk_use_q8_mm_f16_out32(void) {
    return q36_vk_env_default_on("Q36_VK_Q8_MM_F16_OUT32");
}

/* MTP verify/replay window: micro batches there must be bit-identical to
 * the decode matvec, not to the prefill GEMM, so speculative rows equal
 * plain greedy decode. Session-sync small extends must keep the GEMM or
 * warm resume would stop matching the chunked cold rebuild. */
static bool q36_vk_micro_batch;

void q36_gpu_set_micro_batch(bool enabled) {
    q36_vk_micro_batch = enabled;
}

/* Prefill GEMM against raw f32 activations: 64x64 shared-memory tiles,
 * weights dequantized once per k-block for all tokens. Default is the
 * packed-f16 kernel; Q36_VK_Q8_MM_F16=0 keeps the f32 tiles. */
static int q36_vk_matmul_q8_0_mm(q36_gpu_tensor *out,
                                 const unsigned char *w,
                                 uint64_t weight_bytes,
                                 const q36_gpu_tensor *x,
                                 uint64_t row_bytes,
                                 uint64_t out_dim,
                                 uint64_t n_tok,
                                 uint64_t blocks,
                                 float scale) {
    struct {
        uint32_t out_dim;
        uint32_t n_tok;
        uint32_t blocks;
        uint32_t row_bytes;
        float scale;
    } push = { (uint32_t)out_dim, (uint32_t)n_tok, (uint32_t)blocks, (uint32_t)row_bytes, scale };

    pthread_mutex_lock(&q36_vk_mu);
    q36_gpu_tensor *weights = q36_vk_weight_get_unlocked(w, weight_bytes);
    int ok = weights != NULL;
    if (ok) {
        const q36_gpu_tensor *bindings[3] = { weights, x, out };
        const char *op = q36_vk.prof_ops ? q36_vk_q8_0_op_name(out_dim, blocks, n_tok) : "dense_q8_0";
        int f16 = q36_vk_use_q8_mm_f16();
        int out32 = f16 && out_dim == 32u && q36_vk_use_q8_mm_f16_out32();
        q36_vk_kernel *kernel = out32 ? &q36_vk.matmul_q8_0_mm_f16_out32 :
                                f16 ? &q36_vk.matmul_q8_0_mm_f16 :
                                      &q36_vk.matmul_q8_0_mm;
        uint32_t gx = out32 ? 1u :
                      f16 ? (uint32_t)((out_dim + 127u) / 128u) :
                            (uint32_t)((out_dim + 63u) / 64u);
        uint32_t gy = out32 ? (uint32_t)((n_tok + 63u) / 64u) :
                              (uint32_t)((n_tok + 63u) / 64u);
        ok = q36_vk_run_unlocked(op, kernel, bindings, &push, sizeof(push),
                                 gx, gy, 1);
    }
    pthread_mutex_unlock(&q36_vk_mu);
    return ok;
}

/* Dense decode matvec against raw f32 activations: no activation
 * quantization, so no host read of x and no q8 staging upload. Numerics
 * differ from the CPU path (which dots q8-quantized activations), so this
 * stays out of --quality. */
static int q36_vk_matmul_q8_0_f32b(q36_gpu_tensor *out,
                                   const unsigned char *w,
                                   uint64_t weight_bytes,
                                   const q36_gpu_tensor *x,
                                   uint64_t row_bytes,
                                   uint64_t out_dim,
                                   uint64_t blocks,
                                   float scale) {
    struct {
        uint32_t out_dim;
        uint32_t n_tok;
        uint32_t blocks;
        uint32_t row_bytes;
        float scale;
    } push = { (uint32_t)out_dim, 1u, (uint32_t)blocks, (uint32_t)row_bytes, scale };
    uint32_t gx = out_dim > 16384u ? 16384u : (uint32_t)out_dim;
    uint32_t gy = (uint32_t)((out_dim + 16383u) / 16384u);

    pthread_mutex_lock(&q36_vk_mu);
    q36_gpu_tensor *weights = q36_vk_weight_get_unlocked(w, weight_bytes);
    int ok = weights != NULL;
    if (ok) {
        const q36_gpu_tensor *bindings[3] = { weights, x, out };
        const char *op = q36_vk.prof_ops ? q36_vk_q8_0_op_name(out_dim, blocks, 1) : "dense_q8_0";
        ok = q36_vk_run_unlocked(op, &q36_vk.matmul_q8_0_f32b, bindings, &push, sizeof(push),
                                 gx, gy, 1);
    }
    pthread_mutex_unlock(&q36_vk_mu);
    return ok;
}

/* Micro-batch decode matvec: 2..8 tokens accumulated in registers per
 * weight load. Weight traffic equals one matvec and each token's
 * accumulation order equals the single-token kernel, so MTP verify rows
 * are bit-identical to plain decode of the same tokens. */
static int q36_vk_matmul_q8_0_f32b_nx(q36_gpu_tensor *out,
                                      const unsigned char *w,
                                      uint64_t weight_bytes,
                                      const q36_gpu_tensor *x,
                                      uint64_t row_bytes,
                                      uint64_t out_dim,
                                      uint64_t n_tok,
                                      uint64_t blocks,
                                      float scale) {
    struct {
        uint32_t out_dim;
        uint32_t n_tok;
        uint32_t blocks;
        uint32_t row_bytes;
        float scale;
    } push = { (uint32_t)out_dim, (uint32_t)n_tok, (uint32_t)blocks, (uint32_t)row_bytes, scale };
    uint32_t gx = out_dim > 16384u ? 16384u : (uint32_t)out_dim;
    uint32_t gy = (uint32_t)((out_dim + 16383u) / 16384u);

    pthread_mutex_lock(&q36_vk_mu);
    q36_gpu_tensor *weights = q36_vk_weight_get_unlocked(w, weight_bytes);
    int ok = weights != NULL;
    if (ok) {
        const q36_gpu_tensor *bindings[3] = { weights, x, out };
        const char *op = q36_vk.prof_ops ? q36_vk_q8_0_op_name(out_dim, blocks, n_tok) : "dense_q8_0";
        ok = q36_vk_run_unlocked(op, &q36_vk.matmul_q8_0_f32b_nx, bindings, &push, sizeof(push),
                                 gx, gy, 1);
    }
    pthread_mutex_unlock(&q36_vk_mu);
    return ok;
}

static int q36_vk_matmul_q8_0_gpu_quant(q36_gpu_tensor *out,
                                        const unsigned char *w,
                                        uint64_t weight_bytes,
                                        const q36_gpu_tensor *x,
                                        uint64_t row_bytes,
                                        uint64_t out_dim,
                                        uint64_t n_tok,
                                        uint64_t blocks,
                                        float scale) {
    uint64_t xq_bytes;
    if (!q36_u64_mul_ok(n_tok, blocks, &xq_bytes) ||
        !q36_u64_mul_ok(xq_bytes, 36u, &xq_bytes)) return 0;
    q36_gpu_tensor *xq_tensor = q36_gpu_tensor_alloc(xq_bytes);
    if (!xq_tensor) return 0;

    struct {
        uint32_t in_dim;
        uint32_t blocks;
        uint32_t n_tok;
    } qpush = { (uint32_t)(blocks * Q36_VK_QK8_0), (uint32_t)blocks, (uint32_t)n_tok };

    struct {
        uint32_t out_dim;
        uint32_t n_tok;
        uint32_t blocks;
        uint32_t row_bytes;
        float scale;
    } mpush = { (uint32_t)out_dim, (uint32_t)n_tok, (uint32_t)blocks, (uint32_t)row_bytes, scale };

    pthread_mutex_lock(&q36_vk_mu);
    q36_gpu_tensor *weights = q36_vk_weight_get_unlocked(w, weight_bytes);
    int ok = weights != NULL;
    if (ok) {
        const q36_gpu_tensor *qbindings[2] = { x, xq_tensor };
        ok = q36_vk_run_unlocked("q8_0_quant", &q36_vk.quantize_q8_0, qbindings, &qpush, sizeof(qpush),
                                 (uint32_t)blocks, (uint32_t)n_tok, 1);
    }
    if (ok) {
        const q36_gpu_tensor *mbindings[3] = { weights, xq_tensor, out };
        const char *op = q36_vk.prof_ops ? q36_vk_q8_0_op_name(out_dim, blocks, n_tok) : "dense_q8_0";
        if (n_tok == 1) {
            ok = q36_vk_run_unlocked(op, &q36_vk.matmul_q8_0_decode_q36, mbindings, &mpush, sizeof(mpush),
                                     (uint32_t)((out_dim + 15u) / 16u), 1, 1);
        } else {
            ok = q36_vk_run_unlocked(op, &q36_vk.matmul_q8_0_q36, mbindings, &mpush, sizeof(mpush),
                                     (uint32_t)((out_dim + 7u) / 8u),
                                     (uint32_t)((n_tok + 3u) / 4u),
                                     1);
        }
    }
    pthread_mutex_unlock(&q36_vk_mu);
    q36_gpu_tensor_free(xq_tensor);
    return ok;
}

int q36_gpu_matmul_q8_0_scaled_tensor(q36_gpu_tensor *out,
                                      const void *model_map,
                                      uint64_t model_size,
                                      uint64_t weight_offset,
                                      uint64_t in_dim,
                                      uint64_t out_dim,
                                      const q36_gpu_tensor *x,
                                      uint64_t n_tok,
                                      float scale) {
    uint64_t blocks;
    uint64_t row_bytes = 0;
    uint64_t out_bytes = 0;
    uint64_t in_bytes = 0;
    uint64_t weight_bytes = 0;
    uint64_t xq_bytes = 0;
    unsigned char *xq;

    if (in_dim == 0 || n_tok == 0 || (in_dim % Q36_VK_QK8_0) != 0 || in_dim > INT32_MAX) return 0;
    blocks = in_dim / Q36_VK_QK8_0;
    if (!q36_u64_mul_ok(blocks, sizeof(q36_vk_block_q8_0), &row_bytes)) return 0;
    if (!q36_u64_mul_ok(n_tok, out_dim, &out_bytes) || !q36_u64_mul_ok(out_bytes, sizeof(float), &out_bytes)) return 0;
    if (!q36_u64_mul_ok(n_tok, in_dim, &in_bytes) || !q36_u64_mul_ok(in_bytes, sizeof(float), &in_bytes)) return 0;
    if (!q36_u64_mul_ok(row_bytes, out_dim, &weight_bytes)) return 0;
    if (!q36_u64_mul_ok(row_bytes, n_tok, &xq_bytes) || xq_bytes > SIZE_MAX) return 0;
    if (!q36_gpu_tensor_range_ok(out, 0, out_bytes) || !q36_gpu_tensor_range_ok(x, 0, in_bytes)) return 0;

    const unsigned char *w = q36_gpu_weight_bytes(model_map, model_size, weight_offset, weight_bytes);
    if (!w) return 0;

    if (!q36_gpu_quality && n_tok == 1 && q36_vk_use_q8_0_f32b() && q36_vk_use_q8_0_gpu() &&
        out_dim <= UINT32_MAX && blocks <= UINT32_MAX && row_bytes <= UINT32_MAX &&
        q36_vk_matmul_q8_0_f32b(out, w, weight_bytes, x, row_bytes, out_dim, blocks, scale)) {
        return 1;
    }

    if (!q36_gpu_quality && q36_vk_micro_batch && n_tok >= 2 && n_tok <= 8 &&
        q36_vk_use_q8_0_f32b() && q36_vk_use_q8_0_gpu() &&
        out_dim <= UINT32_MAX && blocks <= UINT32_MAX && row_bytes <= UINT32_MAX &&
        q36_vk_matmul_q8_0_f32b_nx(out, w, weight_bytes, x, row_bytes, out_dim, n_tok, blocks, scale)) {
        return 1;
    }

    if (!q36_gpu_quality && n_tok >= 2 && q36_vk_use_q8_0_f32b() && q36_vk_use_q8_0_gpu() &&
        out_dim <= UINT32_MAX && n_tok <= UINT32_MAX && blocks <= UINT32_MAX && row_bytes <= UINT32_MAX &&
        q36_vk_matmul_q8_0_mm(out, w, weight_bytes, x, row_bytes, out_dim, n_tok, blocks, scale)) {
        return 1;
    }

    if (!q36_gpu_quality && n_tok > 1u && q36_vk_env_default_on("Q36_VK_Q8_0_GPU_QUANT") && q36_vk_use_q8_0_gpu() &&
        out_dim <= UINT32_MAX && n_tok <= UINT32_MAX && blocks <= UINT32_MAX && row_bytes <= UINT32_MAX &&
        q36_vk_matmul_q8_0_gpu_quant(out, w, weight_bytes, x, row_bytes, out_dim, n_tok, blocks, scale)) {
        return 1;
    }

    const float *x_host = q36_vk_tensor_contents_labeled((q36_gpu_tensor *)x,
                                                         q36_vk.prof_ops ? q36_vk_q8_0_quant_wait_name(out_dim, blocks, n_tok) :
                                                                          "submit_wait_q8_0_quant_x");
    if (!x_host) return 0;

    xq = malloc((size_t)xq_bytes);
    if (!xq) return 0;

    q36_vk_quantize_q8_0_ctx qctx = { xq, x_host, in_dim, row_bytes };
    if (n_tok * in_dim >= 8192u) q36_gpu_parallel_for_rows(n_tok, 4, q36_vk_quantize_q8_0_rows, &qctx);
    else q36_vk_quantize_q8_0_rows(&qctx, 0, n_tok);

    if (!q36_gpu_quality && q36_vk_use_q8_0_gpu() &&
        out_dim <= UINT32_MAX && n_tok <= UINT32_MAX && blocks <= UINT32_MAX && row_bytes <= UINT32_MAX &&
        q36_vk_matmul_q8_0_gpu(out, w, weight_bytes, xq, xq_bytes, row_bytes, out_dim, n_tok, blocks, scale)) {
        free(xq);
        return 1;
    }

    float *out_host = q36_vk_tensor_contents_labeled(out, "submit_wait_q8_0_host_out");
    if (!out_host) {
        free(xq);
        return 0;
    }
    q36_vk_matmul_q8_0_ctx mctx = { w, xq, out_host, in_dim, out_dim, n_tok, row_bytes, scale };
    if (out_dim >= 64u) q36_gpu_parallel_for_rows(out_dim, 64, q36_vk_matmul_q8_0_rows, &mctx);
    else q36_vk_matmul_q8_0_rows(&mctx, 0, out_dim);
    free(xq);
    return 1;
}

int q36_gpu_shared_ffn_decode_tensor(q36_gpu_tensor *out,
                                      q36_gpu_tensor *mid,
                                      const q36_gpu_tensor *x,
                                      const q36_gpu_tensor *scalar,
                                      const void *model_map,
                                      uint64_t model_size,
                                      uint64_t gate_offset,
                                      uint64_t up_offset,
                                      uint64_t down_offset,
                                      uint32_t in_dim,
                                      uint32_t mid_dim,
                                      uint32_t out_dim,
                                      float gate_scale,
                                      float up_scale,
                                      float down_scale) {
    uint64_t in_blocks, mid_blocks, gate_bytes, down_bytes;
    if (q36_gpu_quality || !q36_vk_use_q8_0_f32b() || !q36_vk_use_q8_0_gpu() ||
        !out || !mid || !x || !scalar || in_dim == 0 || mid_dim == 0 || out_dim == 0 ||
        in_dim % Q36_VK_QK8_0 != 0 || mid_dim % Q36_VK_QK8_0 != 0) {
        return 0;
    }
    in_blocks = in_dim / Q36_VK_QK8_0;
    mid_blocks = mid_dim / Q36_VK_QK8_0;
    if (!q36_u64_mul_ok(in_blocks * sizeof(q36_vk_block_q8_0), mid_dim, &gate_bytes) ||
        !q36_u64_mul_ok(mid_blocks * sizeof(q36_vk_block_q8_0), out_dim, &down_bytes) ||
        !q36_gpu_tensor_range_ok(x, 0, (uint64_t)in_dim * sizeof(float)) ||
        !q36_gpu_tensor_range_ok(mid, 0, (uint64_t)mid_dim * sizeof(float)) ||
        !q36_gpu_tensor_range_ok(scalar, 0, sizeof(float)) ||
        !q36_gpu_tensor_range_ok(out, 0, (uint64_t)out_dim * sizeof(float))) {
        return 0;
    }
    const unsigned char *gate = q36_gpu_weight_bytes(model_map, model_size, gate_offset, gate_bytes);
    const unsigned char *up = q36_gpu_weight_bytes(model_map, model_size, up_offset, gate_bytes);
    const unsigned char *down = q36_gpu_weight_bytes(model_map, model_size, down_offset, down_bytes);
    if (!gate || !up || !down) return 0;

    pthread_mutex_lock(&q36_vk_mu);
    q36_gpu_tensor *gate_tensor = q36_vk_weight_get_unlocked(gate, gate_bytes);
    q36_gpu_tensor *up_tensor = q36_vk_weight_get_unlocked(up, gate_bytes);
    q36_gpu_tensor *down_tensor = q36_vk_weight_get_unlocked(down, down_bytes);
    int ok = gate_tensor && up_tensor && down_tensor;
    if (ok) {
        struct {
            uint32_t mid_dim;
            uint32_t blocks;
            float gate_scale;
            float up_scale;
        } push = { mid_dim, (uint32_t)in_blocks, gate_scale, up_scale };
        const q36_gpu_tensor *bindings[4] = { gate_tensor, up_tensor, x, mid };
        ok = q36_vk_run_unlocked("shared_gate_up_decode", &q36_vk.shared_gate_up_decode,
                                 bindings, &push, sizeof(push), mid_dim, 1, 1);
    }
    if (ok) {
        struct {
            uint32_t out_dim;
            uint32_t blocks;
            float scale;
        } push = { out_dim, (uint32_t)mid_blocks, down_scale };
        const q36_gpu_tensor *bindings[4] = { down_tensor, mid, scalar, out };
        ok = q36_vk_run_unlocked("shared_down_tail_decode", &q36_vk.shared_down_tail_decode,
                                 bindings, &push, sizeof(push), out_dim, 1, 1);
    }
    pthread_mutex_unlock(&q36_vk_mu);
    return ok;
}

static uint64_t q36_vk_kquant_row_bytes(uint32_t type) {
    switch (type) {
    case Q36_VK_TENSOR_Q2_K: return 84;
    case Q36_VK_TENSOR_Q4_K: return 144;
    case Q36_VK_TENSOR_Q5_K: return 176;
    case Q36_VK_TENSOR_Q6_K: return 210;
    default: return 0;
    }
}

static void q36_vk_pack_q5_kx8_matrix(unsigned char *dst,
                                      const unsigned char *src,
                                      uint32_t out_dim,
                                      uint32_t blocks,
                                      uint32_t src_row_bytes) {
    const uint32_t row_tiles = (out_dim + 7u) / 8u;
    for (uint32_t tile = 0; tile < row_tiles; tile++) {
        for (uint32_t b = 0; b < blocks; b++) {
            q36_vk_block_q5_Kx8 packed;
            memset(&packed, 0, sizeof(packed));
            for (uint32_t r = 0; r < 8u; r++) {
                uint32_t row = tile * 8u + r;
                uint8_t sc = 0;
                uint8_t mn = 0;
                if (row >= out_dim) continue;
                const q36_vk_block_q5_K_src *blk = (const q36_vk_block_q5_K_src *)(src + (uint64_t)row * src_row_bytes + (uint64_t)b * sizeof(*blk));
                packed.d[r] = blk->d;
                packed.dmin[r] = blk->dmin;
                for (uint32_t g = 0; g < 8u; g++) {
                    q36_vk_q5_scale_min(blk->scales, g, &sc, &mn);
                    packed.scales[g * 8u + r] = sc;
                    packed.mins[g * 8u + r] = mn;
                }
                for (uint32_t i = 0; i < Q36_VK_QK_K / 8u; i++) packed.qh[i * 8u + r] = blk->qh[i];
                for (uint32_t i = 0; i < Q36_VK_QK_K / 2u; i++) packed.qs[i * 8u + r] = blk->qs[i];
            }
            memcpy(dst + ((uint64_t)tile * blocks + b) * sizeof(packed), &packed, sizeof(packed));
        }
    }
}

static void q36_vk_pack_q6_kx8_matrix(unsigned char *dst,
                                      const unsigned char *src,
                                      uint32_t out_dim,
                                      uint32_t blocks,
                                      uint32_t src_row_bytes) {
    const uint32_t row_tiles = (out_dim + 7u) / 8u;
    for (uint32_t tile = 0; tile < row_tiles; tile++) {
        for (uint32_t b = 0; b < blocks; b++) {
            q36_vk_block_q6_Kx8 packed;
            memset(&packed, 0, sizeof(packed));
            for (uint32_t r = 0; r < 8u; r++) {
                uint32_t row = tile * 8u + r;
                if (row >= out_dim) continue;
                const q36_vk_block_q6_K_src *blk = (const q36_vk_block_q6_K_src *)(src + (uint64_t)row * src_row_bytes + (uint64_t)b * sizeof(*blk));
                packed.d[r] = blk->d;
                for (uint32_t i = 0; i < Q36_VK_QK_K / 16u; i++) packed.scales[i * 8u + r] = blk->scales[i];
                for (uint32_t i = 0; i < Q36_VK_QK_K / 2u; i++) packed.ql[i * 8u + r] = blk->ql[i];
                for (uint32_t i = 0; i < Q36_VK_QK_K / 4u; i++) packed.qh[i * 8u + r] = blk->qh[i];
            }
            memcpy(dst + ((uint64_t)tile * blocks + b) * sizeof(packed), &packed, sizeof(packed));
        }
    }
}

static q36_gpu_tensor *q36_vk_packed_weight_get_unlocked(const unsigned char *source,
                                                         uint64_t src_bytes,
                                                         uint32_t type,
                                                         uint32_t in_dim,
                                                         uint32_t out_dim) {
    for (q36_vk_packed_weight *w = q36_vk.packed_weights; w; w = w->next) {
        if (w->source == source && w->src_bytes == src_bytes && w->type == type &&
            w->in_dim == in_dim && w->out_dim == out_dim) {
            return w->tensor;
        }
    }

    const uint32_t blocks = in_dim / Q36_VK_QK_K;
    const uint32_t row_tiles = (out_dim + 7u) / 8u;
    const uint64_t pack_block_bytes = q36_vk_kquant_mmq_block_bytes(type);
    const uint64_t src_row_bytes = q36_vk_kquant_row_bytes(type) * (uint64_t)blocks;
    const uint64_t packed_bytes = (uint64_t)row_tiles * blocks * pack_block_bytes;
    if (pack_block_bytes == 0 || blocks == 0 || row_tiles == 0) return NULL;

    q36_gpu_tensor *tensor = q36_vk_arena_alloc_unlocked(packed_bytes);
    if (!tensor) return NULL;
    uint64_t offset = 0;
    q36_gpu_tensor *root = q36_gpu_tensor_root(tensor, &offset);
    if (!root) {
        q36_vk_tensor_free_unlocked(tensor);
        return NULL;
    }

    if (type == Q36_VK_TENSOR_Q5_K) {
        q36_vk_pack_q5_kx8_matrix(root->data + offset, source, out_dim, blocks, (uint32_t)src_row_bytes);
    } else if (type == Q36_VK_TENSOR_Q6_K) {
        q36_vk_pack_q6_kx8_matrix(root->data + offset, source, out_dim, blocks, (uint32_t)src_row_bytes);
    } else {
        q36_vk_tensor_free_unlocked(tensor);
        return NULL;
    }

    q36_vk_packed_weight *entry = calloc(1, sizeof(*entry));
    if (!entry) {
        q36_vk_tensor_free_unlocked(tensor);
        return NULL;
    }
    entry->source = source;
    entry->src_bytes = src_bytes;
    entry->type = type;
    entry->in_dim = in_dim;
    entry->out_dim = out_dim;
    entry->tensor = tensor;
    entry->next = q36_vk.packed_weights;
    q36_vk.packed_weights = entry;
    return tensor;
}

static int q36_vk_matmul_kquant_mmq_unlocked(q36_gpu_tensor *out,
                                             const q36_gpu_tensor *weights,
                                             const q36_gpu_tensor *q8,
                                             uint32_t type,
                                             uint32_t out_dim,
                                             uint32_t n_tok,
                                             uint32_t blocks,
                                             float scale) {
    q36_vk_kernel *kernel = NULL;
    bool fast = q36_vk_use_mmq_fast();
    if (type == Q36_VK_TENSOR_Q5_K) kernel = fast ? &q36_vk.matmul_q5k_mmq_fast : &q36_vk.matmul_q5k_mmq;
    else if (type == Q36_VK_TENSOR_Q6_K) kernel = fast ? &q36_vk.matmul_q6k_mmq_fast : &q36_vk.matmul_q6k_mmq;
    else return 0;
    struct {
        uint32_t out_dim;
        uint32_t n_tok;
        uint32_t blocks;
        float scale;
    } push = { out_dim, n_tok, blocks, scale };
    const q36_gpu_tensor *bindings[3] = { weights, q8, out };
    return q36_vk_run_unlocked("dense_kquant_mmq",
                               kernel,
                               bindings,
                               &push,
                               sizeof(push),
                               (out_dim + 7u) / 8u,
                               (n_tok + 3u) / 4u,
                               1);
}

int q36_gpu_matmul_k_quant_q8_scaled_tensor(q36_gpu_tensor *out,
                                            const void *model_map,
                                            uint64_t model_size,
                                            uint64_t weight_offset,
                                            uint32_t weight_type,
                                            uint64_t in_dim,
                                            uint64_t out_dim,
                                            const q36_gpu_tensor *q8,
                                            uint64_t n_tok,
                                            float scale) {
    if (in_dim == 0 || (in_dim % 256u) != 0) return 0;
    if (in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    const uint64_t block_bytes = q36_vk_kquant_row_bytes(weight_type);
    if (block_bytes == 0) return 0;
    const uint64_t blocks = in_dim / Q36_VK_QK_K;
    const uint64_t row_bytes = blocks * block_bytes;
    if (row_bytes == 0 || row_bytes > UINT32_MAX) return 0;

    uint64_t out_bytes = 0;
    uint64_t q8_bytes = 0;
    uint64_t weight_bytes = 0;
    if (!q36_u64_mul_ok(n_tok, out_dim, &out_bytes) || !q36_u64_mul_ok(out_bytes, sizeof(float), &out_bytes)) return 0;
    if (!q36_u64_mul_ok(n_tok, blocks, &q8_bytes) || !q36_u64_mul_ok(q8_bytes, sizeof(q36_vk_block_q8_K), &q8_bytes)) return 0;
    if (!q36_u64_mul_ok(row_bytes, out_dim, &weight_bytes)) return 0;
    if (!q36_gpu_tensor_range_ok(out, 0, out_bytes) || !q36_gpu_tensor_range_ok(q8, 0, q8_bytes)) return 0;

    const unsigned char *source = q36_gpu_weight_bytes(model_map, model_size, weight_offset, weight_bytes);
    if (!source) return 0;

    if (!q36_gpu_quality && q36_vk_use_mmq() &&
        (weight_type == Q36_VK_TENSOR_Q5_K || weight_type == Q36_VK_TENSOR_Q6_K) &&
        (n_tok % 4u) == 0u && n_tok >= 4u) {
        pthread_mutex_lock(&q36_vk_mu);
        q36_gpu_tensor *packed = q36_vk_packed_weight_get_unlocked(source, weight_bytes, weight_type,
                                                                   (uint32_t)in_dim, (uint32_t)out_dim);
        int ok = packed && q36_vk_matmul_kquant_mmq_unlocked(out, packed, q8, (uint32_t)weight_type,
                                                             (uint32_t)out_dim, (uint32_t)n_tok,
                                                             (uint32_t)blocks, scale);
        pthread_mutex_unlock(&q36_vk_mu);
        return ok;
    }

    pthread_mutex_lock(&q36_vk_mu);
    q36_gpu_tensor *weights = q36_vk_weight_get_unlocked(source, weight_bytes);
    int ok = weights != NULL;

    if (ok) {
        struct {
            uint32_t in_dim;
            uint32_t out_dim;
            uint32_t n_tok;
            uint32_t blocks;
            uint32_t row_bytes;
            uint32_t type;
            float scale;
        } mpush = {
            (uint32_t)in_dim,
            (uint32_t)out_dim,
            (uint32_t)n_tok,
            (uint32_t)blocks,
            (uint32_t)row_bytes,
            weight_type,
            scale,
        };
        const q36_gpu_tensor *mbindings[3] = { weights, q8, out };
        /* 8 rows x 8 AVX2 lanes per workgroup. */
        ok = q36_vk_run_unlocked("dense_kquant",
                                 &q36_vk.matmul_kquant,
                                 mbindings,
                                 &mpush,
                                 sizeof(mpush),
                                 ((uint32_t)out_dim + 7u) / 8u,
                                 ((uint32_t)n_tok + Q36_VK_MATMUL_TOK_TILE - 1u) / Q36_VK_MATMUL_TOK_TILE,
                                 1);
    }

    pthread_mutex_unlock(&q36_vk_mu);
    return ok;
}

int q36_gpu_matmul_k_quant_scaled_tensor(q36_gpu_tensor *out,
                                         const void *model_map,
                                         uint64_t model_size,
                                         uint64_t weight_offset,
                                         uint32_t weight_type,
                                         uint64_t in_dim,
                                         uint64_t out_dim,
                                         const q36_gpu_tensor *x,
                                         uint64_t n_tok,
                                         float scale) {
    q36_gpu_tensor *q8 = q36_vk_quantize_q8_k_alloc_tensor(x, in_dim, n_tok);
    if (!q8) return 0;
    int ok = q36_gpu_matmul_k_quant_q8_scaled_tensor(out, model_map, model_size, weight_offset,
                                                     weight_type, in_dim, out_dim, q8, n_tok, scale);
    q36_gpu_tensor_free(q8);
    return ok;
}

int q36_gpu_matmul_k_quant_tensor(q36_gpu_tensor *out,
                                  const void *model_map,
                                  uint64_t model_size,
                                  uint64_t weight_offset,
                                  uint32_t weight_type,
                                  uint64_t in_dim,
                                  uint64_t out_dim,
                                  const q36_gpu_tensor *x,
                                  uint64_t n_tok) {
    return q36_gpu_matmul_k_quant_scaled_tensor(out, model_map, model_size, weight_offset,
                                                weight_type, in_dim, out_dim, x, n_tok, 1.0f);
}

/* Row-wise RMS norm on the host, the exact float-accumulation numerics of
 * q36_ref_rms_norm() (the mode the CPU parity gates were tuned for): scalar
 * square sum, 1/sqrtf scale, optional f32 weight. */
static void q36_vk_rms_norm_row(float *dst, const float *src, const float *weight, uint32_t n, float eps) {
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; i++) ss += src[i] * src[i];
    {
        float scale = 1.0f / sqrtf(ss / (float)n + eps);
        for (uint32_t i = 0; i < n; i++) {
            float v = src[i] * scale;
            dst[i] = weight ? v * weight[i] : v;
        }
    }
}

/* Fixed-size clone of the CPU engine's 128-wide norm: gcc compiles that
 * constant-size copy to an 8-lane ymm square-sum reduced in hsum pair
 * order; the explicit lanes reproduce those roundings exactly. */
static void q36_vk_rms_norm_row_128(float *dst, const float *src, const float *weight, float eps) {
    float lanes[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    float ss;
    for (uint32_t i = 0; i < 128u; i += 8u) {
        for (uint32_t k = 0; k < 8u; k++) lanes[k] += src[i + k] * src[i + k];
    }
    ss = ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6]))
       + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]));
    {
        float scale = 1.0f / sqrtf(ss / 128.0f + eps);
        for (uint32_t i = 0; i < 128u; i++) {
            float v = src[i] * scale;
            dst[i] = weight ? v * weight[i] : v;
        }
    }
}

typedef struct {
    float *out;
    const float *x;
    const float *weight;
    uint32_t n;
    float eps;
} q36_vk_rms_norm_ctx;

static void q36_vk_rms_norm_rows(void *opaque, uint64_t row0, uint64_t row1) {
    q36_vk_rms_norm_ctx *ctx = (q36_vk_rms_norm_ctx *)opaque;

    for (uint64_t r = row0; r < row1; r++) {
        if (ctx->n == 128u) q36_vk_rms_norm_row_128(ctx->out + r * 128u, ctx->x + r * 128u, ctx->weight, ctx->eps);
        else q36_vk_rms_norm_row(ctx->out + r * ctx->n, ctx->x + r * ctx->n, ctx->weight, ctx->n, ctx->eps);
    }
}

static int q36_vk_rms_norm_tensor(q36_gpu_tensor *out,
                                  const q36_gpu_tensor *x,
                                  const float *weight,
                                  uint32_t n,
                                  uint32_t rows,
                                  float eps) {
    if (n == 0 || rows == 0) return 0;
    uint64_t elems = 0;
    uint64_t bytes = 0;
    if (!q36_u64_mul_ok(n, rows, &elems) || !q36_u64_mul_ok(elems, sizeof(float), &bytes)) return 0;
    if (!q36_gpu_tensor_range_ok(out, 0, bytes) || !q36_gpu_tensor_range_ok(x, 0, bytes)) return 0;

    /* Fast mode always norms on GPU: with f32-B dense matvecs the host no
     * longer reads activations afterwards, so a host norm here would be the
     * only remaining decode flush cause. */
    if (!q36_gpu_quality && q36_vk_use_gpu_rms()) {
        struct {
            uint32_t n;
            uint32_t has_weight;
            float eps;
        } push = { n, weight ? 1u : 0u, eps };
        pthread_mutex_lock(&q36_vk_mu);
        q36_gpu_tensor *wt = weight ? q36_vk_weight_get_unlocked(weight, (uint64_t)n * sizeof(float)) : (q36_gpu_tensor *)x;
        int ok = wt != NULL;
        if (ok) {
            const q36_gpu_tensor *bindings[3] = { x, wt, out };
            ok = q36_vk_run_unlocked("norm", &q36_vk.rms_norm, bindings, &push, sizeof(push), rows, 1, 1);
        }
        pthread_mutex_unlock(&q36_vk_mu);
        return ok;
    }

    q36_vk_rms_norm_ctx ctx;
    const float *xp = q36_vk_tensor_contents_labeled((q36_gpu_tensor *)x, "submit_wait_rms_norm_x");
    float *op = q36_vk_tensor_contents_labeled(out, "submit_wait_rms_norm_out");
    if (!xp || !op) return 0;
    ctx.out = op;
    ctx.x = xp;
    ctx.weight = weight;
    ctx.n = n;
    ctx.eps = eps;
    if (elems >= 8192u) q36_gpu_parallel_for_rows(rows, 2, q36_vk_rms_norm_rows, &ctx);
    else q36_vk_rms_norm_rows(&ctx, 0, rows);
    return 1;
}

int q36_gpu_rms_norm_plain_rows_tensor(q36_gpu_tensor *out,
                                       const q36_gpu_tensor *x,
                                       uint32_t n,
                                       uint32_t rows,
                                       float eps) {
    return q36_vk_rms_norm_tensor(out, x, NULL, n, rows, eps);
}

int q36_gpu_rms_norm_weight_rows_tensor(q36_gpu_tensor *out,
                                         const q36_gpu_tensor *x,
                                         const void *model_map,
                                        uint64_t model_size,
                                        uint64_t weight_offset,
                                        uint32_t n,
                                        uint32_t rows,
                                        float eps) {
    const float *weight = (const float *)q36_gpu_weight_bytes(model_map, model_size, weight_offset,
                                                              (uint64_t)n * sizeof(float));
    if (!weight) return 0;
    return q36_vk_rms_norm_tensor(out, x, weight, n, rows, eps);
}

/* SwiGLU on the host: q36_siluf()'s exact x / (1 + expf(-x)) with the same
 * scalar libm expf the CPU engine calls. */
typedef struct {
    float *out;
    const float *gate;
    const float *up;
    float clamp;
    float weight;
} q36_vk_swiglu_ctx;

static void q36_vk_swiglu_rows(void *opaque, uint64_t row0, uint64_t row1) {
    q36_vk_swiglu_ctx *ctx = (q36_vk_swiglu_ctx *)opaque;

    for (uint64_t i = row0; i < row1; i++) {
        float g = ctx->gate[i];
        float u = ctx->up[i];
        if (ctx->clamp > 1.0e-6f) {
            if (g > ctx->clamp) g = ctx->clamp;
            if (u > ctx->clamp) u = ctx->clamp;
            if (u < -ctx->clamp) u = -ctx->clamp;
        }
        ctx->out[i] = g / (1.0f + expf(-g)) * u * ctx->weight;
    }
}

int q36_gpu_swiglu_tensor(q36_gpu_tensor *out,
                          const q36_gpu_tensor *gate,
                          const q36_gpu_tensor *up,
                          uint32_t n,
                          float clamp,
                          float weight) {
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    if (n == 0) return 0;
    if (!q36_gpu_tensor_range_ok(out, 0, bytes) ||
        !q36_gpu_tensor_range_ok(gate, 0, bytes) ||
        !q36_gpu_tensor_range_ok(up, 0, bytes)) {
        return 0;
    }

    if (!q36_gpu_quality && q36_vk_use_gpu_swiglu()) {
        struct {
            uint32_t n;
            float clamp;
            float weight;
        } push = { n, clamp, weight };
        const q36_gpu_tensor *bindings[3] = { gate, up, out };
        pthread_mutex_lock(&q36_vk_mu);
        int ok = q36_vk_run_unlocked("swiglu", &q36_vk.swiglu, bindings, &push, sizeof(push), (n + 255u) / 256u, 1, 1);
        pthread_mutex_unlock(&q36_vk_mu);
        return ok;
    }

    q36_vk_swiglu_ctx ctx;
    const float *gp = q36_vk_tensor_contents_labeled((q36_gpu_tensor *)gate, "submit_wait_swiglu_gate");
    const float *upp = q36_vk_tensor_contents_labeled((q36_gpu_tensor *)up, "submit_wait_swiglu_up");
    float *op = q36_vk_tensor_contents_labeled(out, "submit_wait_swiglu_out");
    if (!gp || !upp || !op) return 0;

    ctx.out = op;
    ctx.gate = gp;
    ctx.up = upp;
    ctx.clamp = clamp;
    ctx.weight = weight;
    if (n >= 16384u) q36_gpu_parallel_for_rows(n, 1024, q36_vk_swiglu_rows, &ctx);
    else q36_vk_swiglu_rows(&ctx, 0, n);
    return 1;
}

int q36_gpu_ffn_tail_tensor(q36_gpu_tensor *out,
                            const q36_gpu_tensor *shared,
                            const q36_gpu_tensor *scalar,
                            uint32_t out_dim,
                            uint32_t n_tok) {
    if (out_dim == 0 || n_tok == 0) return 0;
    if (!q36_gpu_tensor_range_ok(out, 0, (uint64_t)out_dim * n_tok * sizeof(float)) ||
        !q36_gpu_tensor_range_ok(shared, 0, (uint64_t)out_dim * n_tok * sizeof(float)) ||
        !q36_gpu_tensor_range_ok(scalar, 0, (uint64_t)n_tok * sizeof(float))) {
        return 0;
    }
    if (q36_gpu_quality || !q36_vk_use_gpu_ffn_tail()) {
        const float *sp = q36_vk_tensor_contents_labeled((q36_gpu_tensor *)shared, "submit_wait_ffn_tail_shared");
        const float *gp = q36_vk_tensor_contents_labeled((q36_gpu_tensor *)scalar, "submit_wait_ffn_tail_scalar");
        float *op = q36_vk_tensor_contents_labeled(out, "submit_wait_ffn_tail_out");
        if (!sp || !gp || !op) return 0;
        for (uint32_t t = 0; t < n_tok; t++) {
            float g = q36_vk_sigmoid_stable(gp[t]);
            if (!isfinite(g)) return 0;
            for (uint32_t j = 0; j < out_dim; j++) {
                uint64_t idx = (uint64_t)t * out_dim + j;
                op[idx] = fmaf(sp[idx], g, op[idx]);
            }
        }
        return 1;
    }
    struct {
        uint32_t out_dim;
        uint32_t n_tok;
    } push = { out_dim, n_tok };
    const q36_gpu_tensor *bindings[3] = { out, shared, scalar };
    pthread_mutex_lock(&q36_vk_mu);
    int ok = q36_vk_run_unlocked("moe_tail", &q36_vk.ffn_tail, bindings, &push, sizeof(push), (out_dim + 255u) / 256u, n_tok, 1);
    pthread_mutex_unlock(&q36_vk_mu);
    return ok;
}

int q36_gpu_add_tensor(q36_gpu_tensor *out,
                       const q36_gpu_tensor *a,
                       const q36_gpu_tensor *b,
                       uint32_t n) {
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    if (n == 0) return 0;
    if (!q36_gpu_tensor_range_ok(out, 0, bytes) ||
        !q36_gpu_tensor_range_ok(a, 0, bytes) ||
        !q36_gpu_tensor_range_ok(b, 0, bytes)) {
        return 0;
    }

    struct {
        uint32_t n;
    } push = { n };

    const q36_gpu_tensor *bindings[3] = { a, b, out };
    pthread_mutex_lock(&q36_vk_mu);
    int ok = q36_vk_run_unlocked("residual_add",
                                 &q36_vk.add,
                                 bindings,
                                 &push,
                                 sizeof(push),
                                 (n + 255u) / 256u,
                                 1,
                                 1);
    pthread_mutex_unlock(&q36_vk_mu);
    return ok;
}

int q36_gpu_directional_steering_project_tensor(
        q36_gpu_tensor       *x,
        const q36_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale) {
    uint64_t x_elems;
    uint64_t dir_elems;
    if (!x || !directions || width == 0 || rows == 0 || scale == 0.0f) return 0;
    if (!q36_u64_mul_ok(width, rows, &x_elems) ||
        !q36_u64_mul_ok((uint64_t)layer + 1u, width, &dir_elems) ||
        !q36_gpu_tensor_range_ok(x, 0, x_elems * sizeof(float)) ||
        !q36_gpu_tensor_range_ok(directions, 0, dir_elems * sizeof(float))) {
        return 0;
    }
    struct {
        uint32_t layer;
        uint32_t width;
        uint32_t rows;
        float scale;
    } push = { layer, width, rows, scale };
    const q36_gpu_tensor *bindings[2] = { x, directions };
    pthread_mutex_lock(&q36_vk_mu);
    int ok = q36_vk_run_unlocked("directional_steering",
                                 &q36_vk.directional_steering,
                                 bindings,
                                 &push,
                                 sizeof(push),
                                 rows,
                                 1,
                                 1);
    pthread_mutex_unlock(&q36_vk_mu);
    return ok;
}

/* Fused residual add + weighted RMS norm: out_sum = a + b (the new residual)
 * and out_norm = rmsnorm(out_sum) * w in one dispatch.  Bit-identical to
 * q36_gpu_add_tensor() + q36_gpu_rms_norm_weight_rows_tensor(), which stays
 * the fallback for quality mode and Q36_VK_ADD_RMS=0.  out_sum may alias a
 * or b.  The shader keeps each thread's row slice in registers, so n is
 * capped at 16*256 lanes. */
int q36_gpu_add_rms_norm_tensor(q36_gpu_tensor *out_norm,
                                q36_gpu_tensor *out_sum,
                                const q36_gpu_tensor *a,
                                const q36_gpu_tensor *b,
                                const void *model_map,
                                uint64_t model_size,
                                uint64_t weight_offset,
                                uint32_t n,
                                uint32_t rows,
                                float eps) {
    if (n == 0 || rows == 0) return 0;
    if (q36_gpu_quality || !q36_vk_use_add_rms() || !q36_vk_use_gpu_rms() || n > 4096u) {
        return q36_gpu_add_tensor(out_sum, a, b, n * rows) &&
               q36_gpu_rms_norm_weight_rows_tensor(out_norm, out_sum,
                                                   model_map, model_size, weight_offset,
                                                   n, rows, eps);
    }
    uint64_t elems = 0;
    uint64_t bytes = 0;
    if (!q36_u64_mul_ok(n, rows, &elems) || !q36_u64_mul_ok(elems, sizeof(float), &bytes)) return 0;
    if (!q36_gpu_tensor_range_ok(out_norm, 0, bytes) || !q36_gpu_tensor_range_ok(out_sum, 0, bytes) ||
        !q36_gpu_tensor_range_ok(a, 0, bytes) || !q36_gpu_tensor_range_ok(b, 0, bytes)) {
        return 0;
    }
    const float *weight = (const float *)q36_gpu_weight_bytes(model_map, model_size, weight_offset,
                                                              (uint64_t)n * sizeof(float));
    if (!weight) return 0;

    struct {
        uint32_t n;
        float eps;
    } push = { n, eps };
    pthread_mutex_lock(&q36_vk_mu);
    q36_gpu_tensor *wt = q36_vk_weight_get_unlocked(weight, (uint64_t)n * sizeof(float));
    int ok = wt != NULL;
    if (ok) {
        const q36_gpu_tensor *bindings[5] = { a, b, wt, out_sum, out_norm };
        ok = q36_vk_run_unlocked("add_rms_norm", &q36_vk.add_rms_norm, bindings, &push, sizeof(push), rows, 1, 1);
    }
    pthread_mutex_unlock(&q36_vk_mu);
    return ok;
}







/* ===========================================================================
 * Qwen3.6 decode kernels.
 * ===========================================================================
 *
 * Faithful GPU ports of the CPU reference path in q36.c.  Each wrapper mirrors
 * one step of q36_forward_token_cpu(): the multi-section RoPE, the full
 * attention score/softmax/weighted-sum with sinks and output gating, the
 * delta-net recurrent decode, the SSM conv window, and the routed MoE with the
 * model's per-tensor/per-expert .scale factors.
 */

/* Codebook grids for the IQ expert quants, uploaded once and bound to the
 * moe_matvec kernel.  Layout must match the TAB_* offsets in the shader. */
static q36_gpu_tensor *q36_vk_iq_tables_unlocked(void) {
    if (q36_vk.iq_tables) return q36_vk.iq_tables;
    q36_gpu_tensor *t = q36_vk_tensor_alloc_kind_unlocked(Q36_VK_TAB_TOTAL_BYTES, true);
    if (!t) return NULL;
    q36_vk_pack_iq_tables(t->data);
    q36_vk.iq_tables = t;
    return t;
}

static float q36_vk_softplus(float x) {
    if (x > 20.0f) return x;
    return logf(1.0f + expf(x));
}

/* In-place row L2 norm on the host: q36_l2_norm()'s float accumulation. */
static void q36_vk_l2_norm_row(float *x, uint32_t n, float eps) {
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; i++) ss += x[i] * x[i];
    if (ss <= 0.0f) return;
    {
        float scale = 1.0f / fmaxf(sqrtf(ss), eps);
        for (uint32_t i = 0; i < n; i++) x[i] *= scale;
    }
}

/* Fixed-size clone, same reasoning as q36_vk_rms_norm_row_128(). */
static void q36_vk_l2_norm_row_128(float *x, float eps) {
    float ss = 0.0f;
    for (uint32_t i = 0; i < 128u; i++) ss += x[i] * x[i];
    if (ss <= 0.0f) return;
    {
        float scale = 1.0f / fmaxf(sqrtf(ss), eps);
        for (uint32_t i = 0; i < 128u; i++) x[i] *= scale;
    }
}

int q36_gpu_l2_norm_rows_tensor(q36_gpu_tensor *x, uint32_t n, uint32_t rows, float eps) {
    uint64_t elems = 0;
    uint64_t bytes = 0;
    float *xp;
    if (n == 0 || rows == 0) return 0;
    if (!q36_u64_mul_ok(n, rows, &elems) || !q36_u64_mul_ok(elems, sizeof(float), &bytes)) return 0;
    if (!q36_gpu_tensor_range_ok(x, 0, bytes)) return 0;
    xp = q36_vk_tensor_contents_labeled(x, "submit_wait_l2_norm_x");
    if (!xp) return 0;
    for (uint32_t r = 0; r < rows; r++) {
        if (n == 128u) q36_vk_l2_norm_row_128(xp + (uint64_t)r * 128u, eps);
        else q36_vk_l2_norm_row(xp + (uint64_t)r * n, n, eps);
    }
    return 1;
}

int q36_gpu_recurrent_conv_step_tensor(q36_gpu_tensor *cache_conv,
                                       const q36_gpu_tensor *cur,
                                       q36_gpu_tensor *window,
                                       uint32_t n_tok) {
    const uint32_t hist = Q36_VK_N_SSM_CONV - 1u;
    uint64_t row_bytes = (uint64_t)Q36_VK_N_SSM_CONV_DIM * sizeof(float);
    if (n_tok == 0) return 0;
    if (!q36_gpu_tensor_range_ok(cache_conv, 0, hist * row_bytes) ||
        !q36_gpu_tensor_range_ok(cur, 0, (uint64_t)n_tok * row_bytes) ||
        !q36_gpu_tensor_range_ok(window, 0, (uint64_t)n_tok * Q36_VK_N_SSM_CONV * row_bytes)) {
        return 0;
    }
    if (q36_gpu_quality || !q36_vk_use_gpu_recurrent()) {
        float *cachep = q36_vk_tensor_contents_labeled(cache_conv, "submit_wait_recur_window_cache");
        float *windowp = q36_vk_tensor_contents_labeled(window, "submit_wait_recur_window_out");
        float *curp = q36_vk_tensor_contents_labeled((q36_gpu_tensor *)cur, "submit_wait_recur_window_cur");
        if (!cachep || !windowp || !curp) return 0;
        for (uint32_t t = 0; t < n_tok; t++) {
            float *wt = windowp + (uint64_t)t * Q36_VK_N_SSM_CONV * Q36_VK_N_SSM_CONV_DIM;
            const float *ct = curp + (uint64_t)t * Q36_VK_N_SSM_CONV_DIM;
            for (uint32_t i = 0; i < hist; i++) memcpy(wt + (uint64_t)i * Q36_VK_N_SSM_CONV_DIM, cachep + (uint64_t)i * Q36_VK_N_SSM_CONV_DIM, (size_t)row_bytes);
            memcpy(wt + (uint64_t)hist * Q36_VK_N_SSM_CONV_DIM, ct, (size_t)row_bytes);
            if (hist > 1u) memmove(cachep, cachep + Q36_VK_N_SSM_CONV_DIM, (size_t)(hist - 1u) * (size_t)row_bytes);
            memcpy(cachep + (uint64_t)(hist - 1u) * Q36_VK_N_SSM_CONV_DIM, ct, (size_t)row_bytes);
        }
        return 1;
    }
    struct {
        uint32_t conv_dim;
        uint32_t hist;
        uint32_t n_tok;
    } push = { Q36_VK_N_SSM_CONV_DIM, hist, n_tok };
    const q36_gpu_tensor *bindings[3] = { cache_conv, cur, window };
    pthread_mutex_lock(&q36_vk_mu);
    int ok = q36_vk_run_unlocked("recurrent_window", &q36_vk.recur_window, bindings, &push, sizeof(push),
                                 (Q36_VK_N_SSM_CONV_DIM + 255u) / 256u, 1, 1);
    pthread_mutex_unlock(&q36_vk_mu);
    return ok;
}

int q36_gpu_recurrent_conv_silu_tensor(q36_gpu_tensor *cache_conv,
                                       const q36_gpu_tensor *cur,
                                       q36_gpu_tensor *out,
                                       const void *model_map,
                                       uint64_t model_size,
                                       uint64_t kernel_offset,
                                       uint32_t conv_dim,
                                       uint32_t n_taps,
                                       uint32_t n_tok) {
    uint64_t row_bytes = (uint64_t)conv_dim * sizeof(float);
    uint64_t kernel_bytes;
    if (q36_gpu_quality || !q36_vk_use_gpu_recurrent() || conv_dim == 0 || n_taps != 4u || n_tok == 0 ||
        !q36_u64_mul_ok(row_bytes, n_taps, &kernel_bytes) ||
        !q36_gpu_tensor_range_ok(cache_conv, 0, (uint64_t)(n_taps - 1u) * row_bytes) ||
        !q36_gpu_tensor_range_ok(cur, 0, (uint64_t)n_tok * row_bytes) ||
        !q36_gpu_tensor_range_ok(out, 0, (uint64_t)n_tok * row_bytes)) {
        return 0;
    }
    const float *kernel = (const float *)q36_gpu_weight_bytes(model_map, model_size,
                                                              kernel_offset, kernel_bytes);
    if (!kernel) return 0;

    pthread_mutex_lock(&q36_vk_mu);
    q36_gpu_tensor *kt = q36_vk_weight_get_unlocked(kernel, kernel_bytes);
    int ok = kt != NULL;
    if (ok) {
        struct {
            uint32_t conv_dim;
            uint32_t n_taps;
            uint32_t n_tok;
        } push = { conv_dim, n_taps, n_tok };
        const q36_gpu_tensor *bindings[4] = { cache_conv, cur, kt, out };
        ok = q36_vk_run_unlocked("recurrent_conv_silu_decode", &q36_vk.recur_conv_silu_decode,
                                 bindings, &push, sizeof(push),
                                 (conv_dim + 255u) / 256u, 1, 1);
    }
    pthread_mutex_unlock(&q36_vk_mu);
    return ok;
}

int q36_gpu_copy_f32_tensor(q36_gpu_tensor *dst,
                            const q36_gpu_tensor *src,
                            uint32_t n_floats) {
    if (n_floats == 0) return 0;
    struct {
        uint32_t src_stride;
        uint32_t dst_stride;
        uint32_t src_offset;
        uint32_t n;
        uint32_t rows;
    } push = { n_floats, n_floats, 0u, n_floats, 1u };
    const q36_gpu_tensor *bindings[2] = { src, dst };
    pthread_mutex_lock(&q36_vk_mu);
    int ok = q36_vk_run_unlocked("embed_stage_copy", &q36_vk.copy_rows, bindings, &push, sizeof(push),
                                 (n_floats + 255u) / 256u, 1, 1);
    pthread_mutex_unlock(&q36_vk_mu);
    return ok;
}

int q36_gpu_extract_full_attn_q_tensor(q36_gpu_tensor *dst,
                                       const q36_gpu_tensor *qg,
                                       uint32_t n_tok) {
    if (n_tok == 0) return 0;
    struct {
        uint32_t src_stride;
        uint32_t dst_stride;
        uint32_t src_offset;
        uint32_t n;
        uint32_t rows;
    } push = { Q36_VK_N_HEAD_DIM * 2u, Q36_VK_N_HEAD_DIM, 0u, Q36_VK_N_HEAD_DIM, n_tok * Q36_VK_N_HEAD };
    const q36_gpu_tensor *bindings[2] = { qg, dst };
    pthread_mutex_lock(&q36_vk_mu);
    int ok = q36_vk_run_unlocked("attn_extract_q", &q36_vk.copy_rows, bindings, &push, sizeof(push),
                                 (Q36_VK_N_HEAD_DIM + 255u) / 256u, n_tok * Q36_VK_N_HEAD, 1);
    pthread_mutex_unlock(&q36_vk_mu);
    return ok;
}

int q36_gpu_extract_recurrent_v_tensor(q36_gpu_tensor *dst,
                                       const q36_gpu_tensor *conv,
                                       uint32_t n_tok) {
    if (n_tok == 0) return 0;
    if (q36_gpu_quality || !q36_vk_use_gpu_recurrent()) {
        const float *src = q36_vk_tensor_contents_labeled((q36_gpu_tensor *)conv, "submit_wait_recur_extract_v_src");
        float *out = q36_vk_tensor_contents_labeled(dst, "submit_wait_recur_extract_v_out");
        if (!src || !out) return 0;
        for (uint32_t t = 0; t < n_tok; t++) {
            memcpy(out + (uint64_t)t * 4096u,
                   src + (uint64_t)t * Q36_VK_N_SSM_CONV_DIM + (uint64_t)Q36_VK_N_SSM_QK * 2u,
                   (size_t)4096u * sizeof(float));
        }
        return 1;
    }
    struct {
        uint32_t src_stride;
        uint32_t dst_stride;
        uint32_t src_offset;
        uint32_t n;
        uint32_t rows;
    } push = { Q36_VK_N_SSM_CONV_DIM, 4096u, Q36_VK_N_SSM_QK * 2u, 4096u, n_tok };
    const q36_gpu_tensor *bindings[2] = { conv, dst };
    pthread_mutex_lock(&q36_vk_mu);
    int ok = q36_vk_run_unlocked("recurrent_extract_v", &q36_vk.copy_rows, bindings, &push, sizeof(push),
                                 (4096u + 255u) / 256u, n_tok, 1);
    pthread_mutex_unlock(&q36_vk_mu);
    return ok;
}

int q36_gpu_rope_qwen_rows_tensor(q36_gpu_tensor *x,
                                  uint32_t n_head,
                                  uint32_t pos0,
                                  uint32_t n_tok) {
    if (n_head == 0 || n_tok == 0) return 0;
    struct {
        uint32_t n_head;
        uint32_t head_dim;
        uint32_t pos0;
        uint32_t n_tok;
    } push = { n_head, Q36_VK_N_HEAD_DIM, pos0, n_tok };
    const q36_gpu_tensor *bindings[1] = { x };
    pthread_mutex_lock(&q36_vk_mu);
    int ok = q36_vk_run_unlocked("attn_rope", &q36_vk.rope, bindings, &push, sizeof(push), n_head, n_tok, 1);
    pthread_mutex_unlock(&q36_vk_mu);
    return ok;
}

int q36_gpu_rms_norm_rope_qwen_rows_tensor(q36_gpu_tensor *dst,
                                           const q36_gpu_tensor *src,
                                           const void *model_map,
                                           uint64_t model_size,
                                           uint64_t weight_offset,
                                           uint32_t src_stride,
                                           uint32_t n_head,
                                           uint32_t pos0,
                                           uint32_t n_tok,
                                           float eps) {
    const uint32_t head_dim = Q36_VK_N_HEAD_DIM;
    uint64_t rows = (uint64_t)n_head * n_tok;
    uint64_t src_floats;
    uint64_t dst_floats;
    if (src_stride < head_dim || rows == 0) return 0;
    src_floats = (rows - 1u) * src_stride + head_dim;
    dst_floats = rows * head_dim;
    if (!q36_gpu_tensor_range_ok(src, 0, src_floats * sizeof(float)) ||
        !q36_gpu_tensor_range_ok(dst, 0, dst_floats * sizeof(float))) {
        return 0;
    }
    const float *weight = (const float *)q36_gpu_weight_bytes(model_map, model_size,
                                                              weight_offset,
                                                              head_dim * sizeof(float));
    if (!weight) return 0;
    struct {
        uint32_t src_stride;
        uint32_t n_head;
        uint32_t pos0;
        uint32_t n_tok;
        float eps;
        uint32_t pad;
    } push = { src_stride, n_head, pos0, n_tok, eps, 0u };
    pthread_mutex_lock(&q36_vk_mu);
    q36_gpu_tensor *wt = q36_vk_weight_get_unlocked(weight, head_dim * sizeof(float));
    int ok = wt != NULL;
    if (ok) {
        const q36_gpu_tensor *bindings[3] = { src, wt, dst };
        ok = q36_vk_run_unlocked("norm_rope", &q36_vk.rms_norm_rope,
                                 bindings, &push, sizeof(push),
                                 n_head, n_tok, 1);
    }
    pthread_mutex_unlock(&q36_vk_mu);
    return ok;
}

int q36_gpu_rms_norm_rope_qwen_kv_store_tensor(q36_gpu_tensor *k_cache,
                                                q36_gpu_tensor *v_cache,
                                                const q36_gpu_tensor *k,
                                                const q36_gpu_tensor *v,
                                                const void *model_map,
                                                uint64_t model_size,
                                                uint64_t weight_offset,
                                                uint32_t src_stride,
                                                uint32_t n_head,
                                                uint32_t pos0,
                                                uint32_t n_tok,
                                                uint32_t cap,
                                                float eps) {
    const uint32_t head_dim = Q36_VK_N_HEAD_DIM;
    uint64_t rows = (uint64_t)n_head * n_tok;
    uint64_t cache_values = (uint64_t)cap * n_head * head_dim;
    uint64_t src_floats;
    if (src_stride < head_dim || rows == 0 || cap == 0 ||
        pos0 >= cap || n_tok > cap - pos0) {
        return 0;
    }
    src_floats = (rows - 1u) * src_stride + head_dim;
    if (!q36_gpu_tensor_range_ok(k, 0, src_floats * sizeof(float)) ||
        !q36_gpu_tensor_range_ok(v, 0, rows * head_dim * sizeof(float)) ||
        !q36_gpu_tensor_range_ok(k_cache, 0, cache_values * sizeof(uint16_t)) ||
        !q36_gpu_tensor_range_ok(v_cache, 0, cache_values * sizeof(uint16_t))) {
        return 0;
    }
    const float *weight = (const float *)q36_gpu_weight_bytes(model_map, model_size,
                                                              weight_offset,
                                                              head_dim * sizeof(float));
    if (!weight) return 0;
    struct {
        uint32_t src_stride;
        uint32_t n_head;
        uint32_t pos0;
        uint32_t n_tok;
        uint32_t cap;
        float eps;
    } push = { src_stride, n_head, pos0, n_tok, cap, eps };
    pthread_mutex_lock(&q36_vk_mu);
    q36_gpu_tensor *wt = q36_vk_weight_get_unlocked(weight, head_dim * sizeof(float));
    int ok = wt != NULL;
    if (ok) {
        const q36_gpu_tensor *bindings[5] = { k, wt, k_cache, v, v_cache };
        ok = q36_vk_run_unlocked("norm_rope_kv", &q36_vk.rms_norm_rope_kv,
                                 bindings, &push, sizeof(push),
                                 n_head, n_tok, 1);
    }
    pthread_mutex_unlock(&q36_vk_mu);
    return ok;
}

/* Fused RMS norm + Qwen RoPE + quantized KV store (Q8_0 K / Q4_0 V).
 * Same semantics as q36_gpu_rms_norm_rope_qwen_kv_store_tensor but writes
 * quantized blocks directly on the GPU, eliminating the host round-trip. */
int q36_gpu_rms_norm_rope_qwen_kv_store_quant_tensor(q36_gpu_tensor *k_cache,
                                                q36_gpu_tensor *v_cache,
                                                const q36_gpu_tensor *k,
                                                const q36_gpu_tensor *v,
                                                const void *model_map,
                                                uint64_t model_size,
                                                uint64_t weight_offset,
                                                uint32_t src_stride,
                                                uint32_t n_head,
                                                uint32_t pos0,
                                                uint32_t n_tok,
                                                uint32_t cap,
                                                float eps,
                                                uint32_t k_row_bytes,
                                                uint32_t v_row_bytes) {
    const uint32_t head_dim = Q36_VK_N_HEAD_DIM;
    uint64_t rows = (uint64_t)n_head * n_tok;
    if (src_stride < head_dim || rows == 0 || cap == 0 ||
        pos0 >= cap || n_tok > cap - pos0) return 0;
    const float *weight = (const float *)q36_gpu_weight_bytes(model_map, model_size,
                                                              weight_offset,
                                                              head_dim * sizeof(float));
    if (!weight) return 0;
    struct {
        uint32_t src_stride;
        uint32_t n_head;
        uint32_t pos0;
        uint32_t n_tok;
        uint32_t cap;
        float eps;
        uint32_t k_row_bytes;
        uint32_t v_row_bytes;
    } push = { src_stride, n_head, pos0, n_tok, cap, eps, k_row_bytes, v_row_bytes };
    pthread_mutex_lock(&q36_vk_mu);
    q36_gpu_tensor *wt = q36_vk_weight_get_unlocked(weight, head_dim * sizeof(float));
    int ok = wt != NULL;
    if (ok) {
        const q36_gpu_tensor *bindings[5] = { k, wt, k_cache, v, v_cache };
        ok = q36_vk_run_unlocked("norm_rope_kv_quant", &q36_vk.rms_norm_rope_kv_quant,
                                 bindings, &push, sizeof(push),
                                 n_head, n_tok, 1);
    }
    pthread_mutex_unlock(&q36_vk_mu);
    return ok;
}

/* Short causal conv1d + SiLU on the host:
 * fp64 tap accumulation like
 * q36_ssm_conv_apply() and q36_siluf()'s exact formula with libm expf.
 * Batched over n_tok windows of n_taps rows each. */
typedef struct {
    float *out;
    const float *window;
    const float *kernel;
    uint32_t conv_dim;
    uint32_t n_taps;
} q36_vk_conv_silu_ctx;

static void q36_vk_conv_silu_rows(void *opaque, uint64_t row0, uint64_t row1) {
    q36_vk_conv_silu_ctx *ctx = (q36_vk_conv_silu_ctx *)opaque;

    for (uint64_t tok = row0; tok < row1; tok++) {
        const float *wt = ctx->window + tok * ctx->n_taps * ctx->conv_dim;
        float *ot = ctx->out + tok * ctx->conv_dim;
        for (uint32_t c = 0; c < ctx->conv_dim; c++) {
            double acc = 0.0;
            for (uint32_t t = 0; t < ctx->n_taps; t++) {
                acc += (double)wt[(uint64_t)t * ctx->conv_dim + c] * (double)ctx->kernel[(uint64_t)c * ctx->n_taps + t];
            }
            {
                float a = (float)acc;
                ot[c] = a / (1.0f + expf(-a));
            }
        }
    }
}

int q36_gpu_ssm_conv_silu_tensor(q36_gpu_tensor *out,
                                 const q36_gpu_tensor *window,
                                 const void *model_map,
                                 uint64_t model_size,
                                 uint64_t kernel_offset,
                                 uint32_t conv_dim,
                                 uint32_t n_taps,
                                 uint32_t n_tok) {
    uint64_t dim_bytes = (uint64_t)conv_dim * sizeof(float);
    uint64_t kernel_bytes = dim_bytes * n_taps;
    if (conv_dim == 0 || n_taps == 0 || n_tok == 0) return 0;
    if (!q36_gpu_tensor_range_ok(out, 0, dim_bytes * n_tok) ||
        !q36_gpu_tensor_range_ok(window, 0, kernel_bytes * n_tok)) {
        return 0;
    }

    if (!q36_gpu_quality && q36_vk_use_gpu_recurrent()) {
        const float *kernel = (const float *)q36_gpu_weight_bytes(model_map, model_size, kernel_offset, kernel_bytes);
        if (!kernel) return 0;
        struct {
            uint32_t conv_dim;
            uint32_t n_taps;
            uint32_t n_tok;
        } push = { conv_dim, n_taps, n_tok };
        pthread_mutex_lock(&q36_vk_mu);
        q36_gpu_tensor *kt = q36_vk_weight_get_unlocked(kernel, kernel_bytes);
        int ok = kt != NULL;
        if (ok) {
            const q36_gpu_tensor *bindings[3] = { window, kt, out };
            ok = q36_vk_run_unlocked("recurrent_conv_silu", &q36_vk.conv_silu, bindings, &push, sizeof(push),
                                     (conv_dim + 255u) / 256u, n_tok, 1);
        }
        pthread_mutex_unlock(&q36_vk_mu);
        return ok;
    }

    q36_vk_conv_silu_ctx ctx;
    const float *kernel;
    const float *win;
    float *op;
    kernel = (const float *)q36_gpu_weight_bytes(model_map, model_size, kernel_offset, kernel_bytes);
    win = q36_vk_tensor_contents_labeled((q36_gpu_tensor *)window, "submit_wait_recur_conv_window");
    op = q36_vk_tensor_contents_labeled(out, "submit_wait_recur_conv_out");
    if (!kernel || !win || !op) return 0;

    ctx.out = op;
    ctx.window = win;
    ctx.kernel = kernel;
    ctx.conv_dim = conv_dim;
    ctx.n_taps = n_taps;
    if ((uint64_t)n_tok * conv_dim >= 8192u) q36_gpu_parallel_for_rows(n_tok, 2, q36_vk_conv_silu_rows, &ctx);
    else q36_vk_conv_silu_rows(&ctx, 0, n_tok);
    return 1;
}

/* Delta-net q/k head assembly on the host: group gather plus the
 * q36_l2_norm() float accumulation, like the loop in q36_forward_recurrent().
 * conv_stride is the per-token float stride of the conv tensor (its rows
 * carry the v part after the q/k groups). */
typedef struct {
    float *q;
    float *k;
    const float *conv;
    uint32_t n_heads;
    uint32_t n_groups;
    uint32_t state_dim;
    uint32_t conv_stride;
    uint64_t head_floats;
    float eps;
} q36_vk_delta_qk_ctx;

static void q36_vk_delta_qk_rows(void *opaque, uint64_t row0, uint64_t row1) {
    q36_vk_delta_qk_ctx *ctx = (q36_vk_delta_qk_ctx *)opaque;

    for (uint64_t tok = row0; tok < row1; tok++) {
        const float *ct = ctx->conv + tok * ctx->conv_stride;
        float *qt = ctx->q + tok * ctx->head_floats;
        float *kt = ctx->k + tok * ctx->head_floats;
        for (uint32_t h = 0; h < ctx->n_heads; h++) {
            uint32_t g = h % ctx->n_groups;
            memcpy(qt + (uint64_t)h * ctx->state_dim, ct + (uint64_t)g * ctx->state_dim,
                   (size_t)ctx->state_dim * sizeof(float));
            memcpy(kt + (uint64_t)h * ctx->state_dim, ct + ((uint64_t)ctx->n_groups + g) * ctx->state_dim,
                   (size_t)ctx->state_dim * sizeof(float));
            if (ctx->state_dim == 128u) {
                q36_vk_l2_norm_row_128(qt + (uint64_t)h * 128u, ctx->eps);
                q36_vk_l2_norm_row_128(kt + (uint64_t)h * 128u, ctx->eps);
            } else {
                q36_vk_l2_norm_row(qt + (uint64_t)h * ctx->state_dim, ctx->state_dim, ctx->eps);
                q36_vk_l2_norm_row(kt + (uint64_t)h * ctx->state_dim, ctx->state_dim, ctx->eps);
            }
        }
    }
}

int q36_gpu_delta_qk_l2_norm_tensor(q36_gpu_tensor *q_out,
                                    q36_gpu_tensor *k_out,
                                    const q36_gpu_tensor *conv,
                                    uint32_t n_heads,
                                    uint32_t n_groups,
                                    uint32_t state_dim,
                                    uint32_t conv_stride,
                                    uint32_t n_tok,
                                    float eps) {
    uint64_t head_floats = (uint64_t)n_heads * state_dim;
    uint64_t head_bytes = head_floats * n_tok * sizeof(float);
    uint64_t conv_bytes;
    if (n_heads == 0 || n_groups == 0 || state_dim == 0 || n_tok == 0) return 0;
    if (conv_stride < 2u * n_groups * state_dim) return 0;
    conv_bytes = ((uint64_t)conv_stride * (n_tok - 1u) + 2u * (uint64_t)n_groups * state_dim) * sizeof(float);
    if (!q36_gpu_tensor_range_ok(q_out, 0, head_bytes) ||
        !q36_gpu_tensor_range_ok(k_out, 0, head_bytes) ||
        !q36_gpu_tensor_range_ok(conv, 0, conv_bytes)) {
        return 0;
    }

    if (!q36_gpu_quality && q36_vk_use_gpu_recurrent()) {
        struct {
            uint32_t n_heads;
            uint32_t n_groups;
            uint32_t state_dim;
            uint32_t conv_stride;
            uint32_t n_tok;
            float eps;
        } push = { n_heads, n_groups, state_dim, conv_stride, n_tok, eps };
        const q36_gpu_tensor *bindings[3] = { conv, q_out, k_out };
        pthread_mutex_lock(&q36_vk_mu);
        int ok = q36_vk_run_unlocked("delta_qk_l2", &q36_vk.delta_qk, bindings, &push, sizeof(push), n_heads, n_tok, 1);
        pthread_mutex_unlock(&q36_vk_mu);
        return ok;
    }

    q36_vk_delta_qk_ctx ctx;
    const float *cp;
    float *qp;
    float *kp;
    cp = q36_vk_tensor_contents_labeled((q36_gpu_tensor *)conv, "submit_wait_delta_qk_conv");
    qp = q36_vk_tensor_contents_labeled(q_out, "submit_wait_delta_qk_q");
    kp = q36_vk_tensor_contents_labeled(k_out, "submit_wait_delta_qk_k");
    if (!cp || !qp || !kp) return 0;

    ctx.q = qp;
    ctx.k = kp;
    ctx.conv = cp;
    ctx.n_heads = n_heads;
    ctx.n_groups = n_groups;
    ctx.state_dim = state_dim;
    ctx.conv_stride = conv_stride;
    ctx.head_floats = head_floats;
    ctx.eps = eps;
    if ((uint64_t)n_tok * n_heads * state_dim >= 16384u) q36_gpu_parallel_for_rows(n_tok, 2, q36_vk_delta_qk_rows, &ctx);
    else q36_vk_delta_qk_rows(&ctx, 0, n_tok);
    return 1;
}

int q36_gpu_delta_qkv_l2_norm_tensor(q36_gpu_tensor *q_out,
                                     q36_gpu_tensor *k_out,
                                     q36_gpu_tensor *v_out,
                                     const q36_gpu_tensor *conv,
                                     uint32_t n_heads,
                                     uint32_t n_groups,
                                     uint32_t state_dim,
                                     uint32_t conv_stride,
                                     uint32_t n_tok,
                                     float eps) {
    uint64_t head_floats = (uint64_t)n_heads * state_dim;
    uint64_t head_bytes = head_floats * n_tok * sizeof(float);
    uint64_t conv_floats;
    if (q36_gpu_quality || !q36_vk_use_gpu_recurrent() ||
        n_heads == 0 || n_groups == 0 || state_dim == 0 || n_tok == 0) return 0;
    if (conv_stride < 2u * n_groups * state_dim + head_floats) return 0;
    conv_floats = (uint64_t)conv_stride * (n_tok - 1u) +
                  2u * (uint64_t)n_groups * state_dim + head_floats;
    if (!q36_gpu_tensor_range_ok(q_out, 0, head_bytes) ||
        !q36_gpu_tensor_range_ok(k_out, 0, head_bytes) ||
        !q36_gpu_tensor_range_ok(v_out, 0, head_bytes) ||
        !q36_gpu_tensor_range_ok(conv, 0, conv_floats * sizeof(float))) {
        return 0;
    }
    struct {
        uint32_t n_heads;
        uint32_t n_groups;
        uint32_t state_dim;
        uint32_t conv_stride;
        uint32_t n_tok;
        float eps;
    } push = { n_heads, n_groups, state_dim, conv_stride, n_tok, eps };
    const q36_gpu_tensor *bindings[4] = { conv, q_out, k_out, v_out };
    pthread_mutex_lock(&q36_vk_mu);
    int ok = q36_vk_run_unlocked("delta_qkv_l2", &q36_vk.delta_qkv, bindings,
                                 &push, sizeof(push), n_heads, n_tok, 1);
    pthread_mutex_unlock(&q36_vk_mu);
    return ok;
}

/* Host-side per-head gate prep (64 scalars): decay = expf(softplus(alpha +
 * dt_bias) * a) and beta = sigmoid(beta_raw), exactly the elementwise loop
 * before q36_delta_net_decode() plus its expf, so the shader never needs a
 * GPU exp that would round differently from libm. */
typedef struct {
    float *gb;
    const float *alpha;
    const float *beta;
    const float *dt;
    const float *a;
    uint32_t n_heads;
} q36_vk_delta_gates_ctx;

static void q36_vk_delta_gates_rows(void *opaque, uint64_t row0, uint64_t row1) {
    q36_vk_delta_gates_ctx *ctx = (q36_vk_delta_gates_ctx *)opaque;

    for (uint64_t tok = row0; tok < row1; tok++) {
        const float *at = ctx->alpha + tok * ctx->n_heads;
        const float *bt = ctx->beta + tok * ctx->n_heads;
        float *gt = ctx->gb + tok * 2u * ctx->n_heads;
        for (uint32_t h = 0; h < ctx->n_heads; h++) {
            gt[h] = expf(q36_vk_softplus(at[h] + ctx->dt[h]) * ctx->a[h]);
            gt[ctx->n_heads + h] = q36_vk_sigmoid_stable(bt[h]);
        }
    }
}

int q36_gpu_delta_net_gates_tensor(q36_gpu_tensor *gb,
                                   const q36_gpu_tensor *alpha,
                                   const q36_gpu_tensor *beta,
                                   const void *model_map,
                                   uint64_t model_size,
                                   uint64_t dt_bias_offset,
                                   uint64_t a_offset,
                                   uint32_t n_heads,
                                   uint32_t n_tok) {
    uint64_t bytes = (uint64_t)n_heads * sizeof(float);
    if (n_heads == 0 || n_tok == 0) return 0;
    if (!q36_gpu_tensor_range_ok(gb, 0, 2u * bytes * n_tok) ||
        !q36_gpu_tensor_range_ok(alpha, 0, bytes * n_tok) ||
        !q36_gpu_tensor_range_ok(beta, 0, bytes * n_tok)) {
        return 0;
    }

    if (!q36_gpu_quality && q36_vk_use_gpu_recurrent()) {
        const float *dt = (const float *)q36_gpu_weight_bytes(model_map, model_size, dt_bias_offset, bytes);
        const float *a = (const float *)q36_gpu_weight_bytes(model_map, model_size, a_offset, bytes);
        if (!dt || !a) return 0;
        struct {
            uint32_t n_heads;
            uint32_t n_tok;
        } push = { n_heads, n_tok };
        pthread_mutex_lock(&q36_vk_mu);
        q36_gpu_tensor *dt_t = q36_vk_weight_get_unlocked(dt, bytes);
        q36_gpu_tensor *a_t = dt_t ? q36_vk_weight_get_unlocked(a, bytes) : NULL;
        int ok = dt_t != NULL && a_t != NULL;
        if (ok) {
            const q36_gpu_tensor *bindings[5] = { alpha, beta, dt_t, a_t, gb };
            ok = q36_vk_run_unlocked("delta_gates", &q36_vk.delta_gates, bindings, &push, sizeof(push),
                                     (n_heads + 63u) / 64u, n_tok, 1);
        }
        pthread_mutex_unlock(&q36_vk_mu);
        return ok;
    }

    q36_vk_delta_gates_ctx ctx;
    const float *dt;
    const float *a;
    const float *alpha_host;
    const float *beta_host;
    float *gb_host;
    dt = (const float *)q36_gpu_weight_bytes(model_map, model_size, dt_bias_offset, bytes);
    a = (const float *)q36_gpu_weight_bytes(model_map, model_size, a_offset, bytes);
    alpha_host = q36_vk_tensor_contents_labeled((q36_gpu_tensor *)alpha, "submit_wait_delta_gates_alpha");
    beta_host = q36_vk_tensor_contents_labeled((q36_gpu_tensor *)beta, "submit_wait_delta_gates_beta");
    gb_host = q36_vk_tensor_contents_labeled(gb, "submit_wait_delta_gates_out");
    if (!dt || !a || !alpha_host || !beta_host || !gb_host) return 0;

    ctx.gb = gb_host;
    ctx.alpha = alpha_host;
    ctx.beta = beta_host;
    ctx.dt = dt;
    ctx.a = a;
    ctx.n_heads = n_heads;
    if ((uint64_t)n_tok * n_heads >= 128u) q36_gpu_parallel_for_rows(n_tok, 2, q36_vk_delta_gates_rows, &ctx);
    else q36_vk_delta_gates_rows(&ctx, 0, n_tok);
    return 1;
}

/* n_tok sequential state updates in one dispatch: the kernel loops tokens
 * inside the workgroup, preserving the exact per-head chain order, so a
 * prefill batch costs one dispatch instead of n_tok barrier-separated
 * ones (per-dispatch overhead is what bounds this stack). */
int q36_gpu_delta_net_decode_tensor(q36_gpu_tensor *state,
                                    const q36_gpu_tensor *q,
                                    const q36_gpu_tensor *k,
                                    const q36_gpu_tensor *v,
                                    const q36_gpu_tensor *gb,
                                    q36_gpu_tensor *out,
                                    uint32_t n_heads,
                                    uint32_t state_dim,
                                    uint32_t n_tok) {
    uint64_t vec_bytes = (uint64_t)n_heads * state_dim * sizeof(float);
    uint64_t gb_bytes = 2u * (uint64_t)n_heads * sizeof(float);
    uint64_t state_bytes = vec_bytes * state_dim;
    if (n_heads == 0 || state_dim == 0 || state_dim > 128 || n_tok == 0) return 0;
    if (!q36_gpu_tensor_range_ok(state, 0, state_bytes) ||
        !q36_gpu_tensor_range_ok(q, 0, vec_bytes * n_tok) ||
        !q36_gpu_tensor_range_ok(k, 0, vec_bytes * n_tok) ||
        !q36_gpu_tensor_range_ok(v, 0, vec_bytes * n_tok) ||
        !q36_gpu_tensor_range_ok(gb, 0, gb_bytes * n_tok) ||
        !q36_gpu_tensor_range_ok(out, 0, vec_bytes * n_tok)) {
        return 0;
    }

    struct {
        uint32_t n_heads;
        uint32_t state_dim;
        uint32_t n_tok;
    } push = { n_heads, state_dim, n_tok };
    const q36_gpu_tensor *bindings[6] = { state, q, k, v, gb, out };
    q36_vk_kernel *kernel = &q36_vk.delta_net;
    const char *name = "delta_net";
    uint32_t groups_y = 1;
    if (!q36_gpu_quality) {
        if (n_tok >= 32u && state_dim == 128u && q36_vk_use_delta_col_prefill()) {
            /* Sliced dispatch: 64-token windows keep each run's q/k/v
             * stream L2-resident (one 512-token run cost +60% pure GPU
             * time).  The state RAW hazard chains the slices in order and
             * the per-column arithmetic is identical to a single run. */
            int ok = 1;
            pthread_mutex_lock(&q36_vk_mu);
            for (uint32_t t0 = 0; ok && t0 < n_tok; t0 += 64u) {
                struct {
                    uint32_t n_heads;
                    uint32_t state_dim;
                    uint32_t tok0;
                    uint32_t tok_end;
                } cpush = { n_heads, state_dim, t0,
                            t0 + 64u < n_tok ? t0 + 64u : n_tok };
                ok = q36_vk_run_unlocked("delta_net_cols", &q36_vk.delta_net_cols,
                                         bindings, &cpush, sizeof(cpush),
                                         n_heads, state_dim, 1);
            }
            pthread_mutex_unlock(&q36_vk_mu);
            return ok;
        } else if (state_dim == 128u && q36_vk_use_delta_decode()) {
            /* Bit-identical to delta_net_fast with half the state traffic;
             * covers decode and chunk tails (n_tok < 32). */
            kernel = &q36_vk.delta_net_decode;
            name = "delta_net_decode";
            groups_y = state_dim / 32u;
        } else if (q36_vk_use_delta_fast_prefill()) {
            kernel = &q36_vk.delta_net_fast;
            name = "delta_net_fast";
        }
    }
    pthread_mutex_lock(&q36_vk_mu);
    int ok = q36_vk_run_unlocked(name, kernel, bindings, &push, sizeof(push),
                                 n_heads, groups_y, 1);
    pthread_mutex_unlock(&q36_vk_mu);
    return ok;
}

/* KV cache append, the GPU twin of q36_full_attn_cache_store(): row t of
 * k/v lands at position pos0 + t.  Rows are host-converted when the fast
 * f16 GPU append path is not applicable. */
int q36_gpu_attn_kv_store_tensor(q36_gpu_tensor *k_cache,
                                 q36_gpu_tensor *v_cache,
                                 const q36_gpu_tensor *k,
                                 const q36_gpu_tensor *v,
                                 uint32_t pos0,
                                 uint32_t n_tok,
                                 uint32_t cap,
                                 uint32_t k_row,
                                 uint32_t v_row,
                                 uint32_t k_cache_type,
                                 uint32_t v_cache_type,
                                 uint32_t k_cache_row_bytes,
                                 uint32_t v_cache_row_bytes) {
    const float *k_host;
    const float *v_host;
    unsigned char *kc;
    unsigned char *vc;
    q36_vk_prof_scope prof;
    int ok = 0;
    uint32_t k_expect;
    uint32_t v_expect;
    if (cap == 0 || n_tok == 0 || pos0 >= cap || n_tok > cap - pos0 || k_row == 0 || v_row == 0) return 0;
    if (k_cache_type > 2u || v_cache_type > 2u || k_cache_row_bytes == 0 || v_cache_row_bytes == 0) return 0;
    k_expect = q36_vk_kv_cache_row_bytes(k_cache_type, k_row);
    v_expect = q36_vk_kv_cache_row_bytes(v_cache_type, v_row);
    if (k_cache_row_bytes != k_expect || v_cache_row_bytes != v_expect) return 0;
    if (!q36_gpu_tensor_range_ok(k, 0, (uint64_t)n_tok * k_row * sizeof(float)) ||
        !q36_gpu_tensor_range_ok(v, 0, (uint64_t)n_tok * v_row * sizeof(float)) ||
        !q36_gpu_tensor_range_ok(k_cache, 0, (uint64_t)cap * k_cache_row_bytes) ||
        !q36_gpu_tensor_range_ok(v_cache, 0, (uint64_t)cap * v_cache_row_bytes)) {
        return 0;
    }
    /* Fast decode appends KV on GPU; packHalf2x16 rounds to nearest-even,
     * matching Q36's host half conversion. */
    if (k_cache_type == 0u && v_cache_type == 0u &&
        k_cache_row_bytes == k_row * sizeof(uint16_t) &&
        v_cache_row_bytes == v_row * sizeof(uint16_t) &&
        !q36_gpu_quality && (k_row % 2u) == 0u && (v_row % 2u) == 0u &&
        q36_vk_env_default_on("Q36_VK_GPU_KV_STORE")) {
        struct {
            uint32_t k_row;
            uint32_t v_row;
            uint32_t pos0;
            uint32_t n_tok;
        } push = { k_row, v_row, pos0, n_tok };
        const q36_gpu_tensor *bindings[4] = { k_cache, v_cache, k, v };
        uint32_t row_max = k_row > v_row ? k_row : v_row;
        pthread_mutex_lock(&q36_vk_mu);
        ok = q36_vk_run_unlocked("attn_kv_store", &q36_vk.kv_store, bindings, &push, sizeof(push),
                                 (row_max / 2u + 255u) / 256u, n_tok, 1);
        pthread_mutex_unlock(&q36_vk_mu);
        if (ok) return 1;
        ok = 0;
    }

    /* GPU quantized KV store: writes Q8_0/Q4_0 blocks directly on the GPU,
     * avoiding the host round-trip that costs ~200 pipeline flushes per run. */
    if (!q36_gpu_quality && n_tok > 1u &&
        (k_row % 32u) == 0u && (v_row % 32u) == 0u &&
        k_cache_type >= 1u && k_cache_type <= 2u &&
        v_cache_type >= 1u && v_cache_type <= 2u &&
        q36_vk_env_default_on("Q36_VK_GPU_KV_STORE")) {
        struct { uint32_t row_elems; uint32_t cache_row_bytes; uint32_t pos0; uint32_t n_tok; uint32_t qtype; } kpush =
            { k_row, k_cache_row_bytes, pos0, n_tok, k_cache_type };
        struct { uint32_t row_elems; uint32_t cache_row_bytes; uint32_t pos0; uint32_t n_tok; uint32_t qtype; } vpush =
            { v_row, v_cache_row_bytes, pos0, n_tok, v_cache_type };
        const q36_gpu_tensor *kbind[2] = { k_cache, k };
        const q36_gpu_tensor *vbind[2] = { v_cache, v };
        pthread_mutex_lock(&q36_vk_mu);
        ok = q36_vk_run_unlocked("kv_store_quant_k", &q36_vk.kv_store_quant,
                                 kbind, &kpush, sizeof(kpush), 1, n_tok, 1);
        if (ok) {
            ok = q36_vk_run_unlocked("kv_store_quant_v", &q36_vk.kv_store_quant,
                                     vbind, &vpush, sizeof(vpush), 1, n_tok, 1);
        }
        pthread_mutex_unlock(&q36_vk_mu);
        if (ok) return 1;
        ok = 0;
    }

    q36_vk_prof_scope_begin(&prof, "attn_kv_store_host");
    k_host = q36_vk_tensor_contents_labeled((q36_gpu_tensor *)k, "submit_wait_attn_kv_k");
    v_host = q36_vk_tensor_contents_labeled((q36_gpu_tensor *)v, "submit_wait_attn_kv_v");
    kc = q36_vk_tensor_contents_labeled(k_cache, "submit_wait_attn_kv_k_cache");
    vc = q36_vk_tensor_contents_labeled(v_cache, "submit_wait_attn_kv_v_cache");
    if (!k_host || !v_host || !kc || !vc) goto done;

    for (uint32_t t = 0; t < n_tok; t++) {
        uint64_t pos = pos0 + t;
        const float *kt = k_host + (uint64_t)t * k_row;
        const float *vt = v_host + (uint64_t)t * v_row;
        unsigned char *kdst = kc + pos * k_cache_row_bytes;
        unsigned char *vdst = vc + pos * v_cache_row_bytes;
        if (k_cache_type == 0u) {
            uint16_t *h = (uint16_t *)kdst;
            for (uint32_t i = 0; i < k_row; i++) h[i] = q36_quant_f32_to_f16(kt[i]);
        } else if (k_cache_type == 1u) {
            q36_quant_q8_0(kt, kdst, (int64_t)k_row);
        } else {
            q36_quant_q4_0(kt, kdst, (int64_t)k_row);
        }
        if (v_cache_type == 0u) {
            uint16_t *h = (uint16_t *)vdst;
            for (uint32_t i = 0; i < v_row; i++) h[i] = q36_quant_f32_to_f16(vt[i]);
        } else if (v_cache_type == 1u) {
            q36_quant_q8_0(vt, vdst, (int64_t)v_row);
        } else {
            q36_quant_q4_0(vt, vdst, (int64_t)v_row);
        }
    }
    ok = 1;
done:
    q36_vk_prof_scope_end(&prof);
    return ok;
}

/* Full-attention decode: score pass then softmax/weighted-sum/gating pass,
 * mirroring the per-head loop in q36_forward_full_attn().  Causal batch:
 * query row t (absolute position pos0 + t) attends keys 0..pos0+t.  One
 * score dispatch covers the whole chunk (token on workgroup z), the softmax
 * runs on the host for every token after one flush, then one reduce
 * dispatch finishes it, so the cost is one GPU round-trip and two
 * dispatches per batch instead of per token.  scores is caller scratch of
 * n_tok * n_head * (pos0 + n_tok) floats and is clobbered. */
typedef struct {
    float *aux;
    float *scores;
    const float *qg;
    const float *sinks;
    uint64_t score_stride;
    uint64_t qg_stride;
    uint64_t aux_floats;
    uint32_t pos0;
    uint32_t n_head;
    uint32_t head_dim;
    bool has_sinks;
} q36_vk_attn_host_ctx;

static void q36_vk_attn_host_rows(void *opaque, uint64_t row0, uint64_t row1) {
    q36_vk_attn_host_ctx *ctx = (q36_vk_attn_host_ctx *)opaque;

    for (uint64_t tok = row0; tok < row1; tok++) {
        uint32_t n_kv = ctx->pos0 + (uint32_t)tok + 1u;
        float *stok = ctx->scores + tok * ctx->score_stride;
        float *atok = ctx->aux + tok * ctx->aux_floats;
        const float *gtok = ctx->qg + tok * ctx->qg_stride;
        for (uint32_t h = 0; h < ctx->n_head; h++) {
            float *sh = stok + (uint64_t)h * n_kv;
            float max_score = ctx->has_sinks ? ctx->sinks[h] : -INFINITY;
            double denom = 0.0;
            for (uint32_t t = 0; t < n_kv; t++) {
                if (sh[t] > max_score) max_score = sh[t];
            }
            if (ctx->has_sinks) denom = exp((double)ctx->sinks[h] - (double)max_score);
            for (uint32_t t = 0; t < n_kv; t++) {
                float w = expf(sh[t] - max_score);
                denom += w;
                sh[t] = w;
            }
            atok[h] = denom > 0.0 ? (float)(1.0 / denom) : 1.0f;
            for (uint32_t j = 0; j < ctx->head_dim; j++) {
                float g = gtok[(uint64_t)h * 2u * ctx->head_dim + ctx->head_dim + j];
                atok[ctx->n_head + (uint64_t)h * ctx->head_dim + j] = 1.0f / (1.0f + expf(-g));
            }
        }
    }
}

int q36_gpu_attn_decode_tensor(q36_gpu_tensor *out,
                               const q36_gpu_tensor *q,
                               const q36_gpu_tensor *qg,
                               const q36_gpu_tensor *k_cache,
                               const q36_gpu_tensor *v_cache,
                               q36_gpu_tensor *scores,
                               const void *model_map,
                               uint64_t model_size,
                               uint64_t sinks_offset,
                               bool has_sinks,
                               uint32_t pos0,
                               uint32_t n_tok,
                               uint32_t n_head,
                               uint32_t n_head_kv,
                               uint32_t head_dim,
                               uint32_t k_cache_type,
                               uint32_t v_cache_type,
                               uint32_t k_cache_row_bytes,
                               uint32_t v_cache_row_bytes) {
    uint32_t kv_max = pos0 + n_tok;
    uint64_t q_bytes = (uint64_t)n_head * head_dim * sizeof(float);
    uint64_t qg_stride = 2u * (uint64_t)n_head * head_dim;
    uint64_t k_cache_bytes = (uint64_t)kv_max * k_cache_row_bytes;
    uint64_t v_cache_bytes = (uint64_t)kv_max * v_cache_row_bytes;
    uint64_t score_stride = (uint64_t)n_head * kv_max * sizeof(float);
    uint64_t aux_floats = (uint64_t)n_head + (uint64_t)n_head * head_dim;
    const float *sinks = NULL;
    q36_gpu_tensor *aux;
    int ok = 1;
    uint32_t k_expect;
    uint32_t v_expect;
    if (n_tok == 0 || n_head == 0 || n_head_kv == 0 || head_dim == 0) return 0;
    if (n_head % n_head_kv != 0 || kv_max < n_tok ||
        k_cache_type > 2u || v_cache_type > 2u ||
        k_cache_row_bytes == 0 || v_cache_row_bytes == 0) return 0;
    k_expect = q36_vk_kv_cache_row_bytes(k_cache_type, n_head_kv * head_dim);
    v_expect = q36_vk_kv_cache_row_bytes(v_cache_type, n_head_kv * head_dim);
    if (k_cache_row_bytes != k_expect || v_cache_row_bytes != v_expect) return 0;
    if (!q36_gpu_tensor_range_ok(out, 0, q_bytes * n_tok) ||
        !q36_gpu_tensor_range_ok(q, 0, q_bytes * n_tok) ||
        !q36_gpu_tensor_range_ok(qg, 0, 2u * q_bytes * n_tok) ||
        !q36_gpu_tensor_range_ok(k_cache, 0, k_cache_bytes) ||
        !q36_gpu_tensor_range_ok(v_cache, 0, v_cache_bytes)) {
        return 0;
    }
    if (has_sinks) {
        sinks = (const float *)q36_gpu_weight_bytes(model_map, model_size, sinks_offset,
                                                    (uint64_t)n_head * sizeof(float));
        if (!sinks) return 0;
    }

    /* Fused fast path: scores, online softmax, V sum, and gating in one
     * dispatch, f32 with a per-token order that depends only on n_kv (so
     * warm==cold holds).  scores may be NULL here; the runtime skips that
     * scratch when this path is on.  --quality keeps the bit-exact fp64
     * chain below. */
    if (q36_gpu_attn_fused_enabled() &&
        head_dim <= 256 && (head_dim & 1u) == 0u) {
        /* Query-tiled prefill: one workgroup per (head, 8-token tile, span)
         * reuses each K/V fetch across the tile's queries instead of
         * streaming the cache once per token.  It always goes through the
         * split partials + ordered combine; single-span partials are
         * bit-identical to the fused kernel, so decode (n_tok == 1) staying
         * on fused/split cannot diverge from prefill. */
        if (n_tok >= 2u && q36_vk_use_attn_qtile()) {
            uint32_t n_spans = (kv_max + 511u) / 512u;
            uint64_t part_bytes = (uint64_t)n_tok * n_head * n_spans * (head_dim + 2u) * sizeof(float);
            struct {
                uint32_t pos0;
                uint32_t n_head;
                uint32_t n_head_kv;
                uint32_t head_dim;
                uint32_t n_spans;
                uint32_t n_tok;
                uint32_t has_sinks;
                uint32_t k_type;
                uint32_t v_type;
                uint32_t k_row_bytes;
                uint32_t v_row_bytes;
            } qpush = { pos0, n_head, n_head_kv, head_dim, n_spans, n_tok, has_sinks ? 1u : 0u,
                        k_cache_type, v_cache_type, k_cache_row_bytes, v_cache_row_bytes };
            struct {
                uint32_t pos0;
                uint32_t n_head;
                uint32_t head_dim;
                uint32_t n_spans;
                uint32_t qg_stride;
                uint32_t has_sinks;
            } cpush = { pos0, n_head, head_dim, n_spans, (uint32_t)qg_stride, has_sinks ? 1u : 0u };
            pthread_mutex_lock(&q36_vk_mu);
            q36_gpu_tensor *part = q36_vk.attn_part;
            if (!part || q36_vk.attn_part_bytes < part_bytes) {
                q36_vk_tensor_release_unlocked(q36_vk.attn_part);
                part = q36_vk_tensor_alloc_kind_unlocked(part_bytes, true);
                q36_vk.attn_part = part;
                q36_vk.attn_part_bytes = part ? part_bytes : 0;
            }
            q36_gpu_tensor *sinks_t = sinks ?
                q36_vk_weight_get_unlocked(sinks, (uint64_t)n_head * sizeof(float)) :
                (q36_gpu_tensor *)qg;
            ok = part != NULL && sinks_t != NULL;
            if (ok) {
                const q36_gpu_tensor *qbind[5] = { q, k_cache, v_cache, sinks_t, part };
                ok = q36_vk_run_unlocked("attn_prefill_qtile", &q36_vk.attn_prefill_qtile,
                                         qbind, &qpush, sizeof(qpush),
                                         n_head, (n_tok + 7u) / 8u, n_spans);
            }
            if (ok) {
                const q36_gpu_tensor *cbind[4] = { part, qg, sinks_t, out };
                ok = q36_vk_run_unlocked("attn_combine", &q36_vk.attn_combine,
                                         cbind, &cpush, sizeof(cpush),
                                         n_head, n_tok, 1);
            }
            pthread_mutex_unlock(&q36_vk_mu);
            return ok;
        }

        /* Split-K when the causal range spans multiple 512-key windows:
         * per-span partials then an ordered combine.  Decode occupancy goes
         * from n_head workgroups to n_head * spans.  A single-span split is
         * bit-identical to the fused kernel (span 0 seeds its max with the
         * sink), so gating on kv_max keeps per-token n_tok-invariance. */
        uint32_t n_spans = (kv_max + 511u) / 512u;
        if (n_spans > 1u && head_dim <= 256u && q36_vk_use_attn_splitk()) {
            uint64_t part_bytes = (uint64_t)n_tok * n_head * n_spans * (head_dim + 2u) * sizeof(float);
            struct {
                uint32_t pos0;
                uint32_t n_head;
                uint32_t n_head_kv;
                uint32_t head_dim;
                uint32_t n_spans;
                uint32_t has_sinks;
                uint32_t k_type;
                uint32_t v_type;
                uint32_t k_row_bytes;
                uint32_t v_row_bytes;
            } spush = { pos0, n_head, n_head_kv, head_dim, n_spans, has_sinks ? 1u : 0u,
                        k_cache_type, v_cache_type, k_cache_row_bytes, v_cache_row_bytes };
            struct {
                uint32_t pos0;
                uint32_t n_head;
                uint32_t head_dim;
                uint32_t n_spans;
                uint32_t qg_stride;
                uint32_t has_sinks;
            } cpush = { pos0, n_head, head_dim, n_spans, (uint32_t)qg_stride, has_sinks ? 1u : 0u };
            pthread_mutex_lock(&q36_vk_mu);
            /* The partials scratch lives on the runtime and only grows: an
             * alloc/free per attention call showed up as ~0.2ms of host
             * record time per dispatch. */
            q36_gpu_tensor *part = q36_vk.attn_part;
            if (!part || q36_vk.attn_part_bytes < part_bytes) {
                q36_vk_tensor_release_unlocked(q36_vk.attn_part);
                part = q36_vk_tensor_alloc_kind_unlocked(part_bytes, true);
                q36_vk.attn_part = part;
                q36_vk.attn_part_bytes = part ? part_bytes : 0;
            }
            q36_gpu_tensor *sinks_t = sinks ?
                q36_vk_weight_get_unlocked(sinks, (uint64_t)n_head * sizeof(float)) :
                (q36_gpu_tensor *)qg;
            ok = part != NULL && sinks_t != NULL;
            if (ok) {
                const q36_gpu_tensor *sbind[5] = { q, k_cache, v_cache, sinks_t, part };
                ok = q36_vk_run_unlocked("attn_decode_split", &q36_vk.attn_decode_split,
                                         sbind, &spush, sizeof(spush),
                                         n_head, n_tok, n_spans);
            }
            if (ok) {
                const q36_gpu_tensor *cbind[4] = { part, qg, sinks_t, out };
                ok = q36_vk_run_unlocked("attn_combine", &q36_vk.attn_combine,
                                         cbind, &cpush, sizeof(cpush),
                                         n_head, n_tok, 1);
            }
            pthread_mutex_unlock(&q36_vk_mu);
            return ok;
        }
        struct {
            uint32_t pos0;
            uint32_t n_head;
            uint32_t n_head_kv;
            uint32_t head_dim;
            uint32_t qg_stride;
            uint32_t has_sinks;
            uint32_t k_type;
            uint32_t v_type;
            uint32_t k_row_bytes;
            uint32_t v_row_bytes;
        } fpush = {
            pos0, n_head, n_head_kv, head_dim,
            (uint32_t)qg_stride, has_sinks ? 1u : 0u,
            k_cache_type, v_cache_type, k_cache_row_bytes, v_cache_row_bytes,
        };
        pthread_mutex_lock(&q36_vk_mu);
        q36_gpu_tensor *sinks_t = sinks ?
            q36_vk_weight_get_unlocked(sinks, (uint64_t)n_head * sizeof(float)) :
            (q36_gpu_tensor *)qg;
        ok = sinks_t != NULL;
        if (ok) {
            const q36_gpu_tensor *fbindings[6] = { q, qg, k_cache, v_cache, sinks_t, out };
            ok = q36_vk_run_unlocked("attn_decode_fused", &q36_vk.attn_decode_fused,
                                     fbindings, &fpush, sizeof(fpush),
                                     n_head, n_tok, 1);
        }
        pthread_mutex_unlock(&q36_vk_mu);
        return ok;
    }

    if (!q36_gpu_tensor_range_ok(scores, 0, score_stride * n_tok)) return 0;

    {
        struct {
            uint32_t pos0;
            uint32_t n_head;
            uint32_t n_head_kv;
            uint32_t head_dim;
            uint32_t score_stride;
            uint32_t k_type;
            uint32_t k_row_bytes;
        } spush = { pos0, n_head, n_head_kv, head_dim, (uint32_t)(score_stride / sizeof(float)),
                    k_cache_type, k_cache_row_bytes };
        const q36_gpu_tensor *sbindings[3] = { q, k_cache, scores };
        pthread_mutex_lock(&q36_vk_mu);
        ok = q36_vk_run_unlocked("attn_scores", &q36_vk.attn_scores, sbindings, &spush, sizeof(spush),
                                 (kv_max + 63u) / 64u, n_head, n_tok);
        pthread_mutex_unlock(&q36_vk_mu);
    }
    if (!ok) return 0;
    aux = q36_gpu_tensor_alloc(aux_floats * n_tok * sizeof(float));
    if (!aux) return 0;

    if (!q36_gpu_quality && q36_vk_use_gpu_attn_post()) {
        struct {
            uint32_t pos0;
            uint32_t n_head;
            uint32_t head_dim;
            uint32_t score_stride;
            uint32_t qg_stride;
            uint32_t aux_stride;
            uint32_t has_sinks;
        } ppush = {
            pos0,
            n_head,
            head_dim,
            (uint32_t)(score_stride / sizeof(float)),
            (uint32_t)qg_stride,
            (uint32_t)aux_floats,
            has_sinks ? 1u : 0u,
        };
        pthread_mutex_lock(&q36_vk_mu);
        q36_gpu_tensor *sinks_t = sinks ? q36_vk_weight_get_unlocked(sinks, (uint64_t)n_head * sizeof(float)) : (q36_gpu_tensor *)qg;
        ok = sinks_t != NULL;
        if (ok) {
            const q36_gpu_tensor *pbindings[4] = { scores, qg, sinks_t, aux };
            ok = q36_vk_run_unlocked("attn_post", &q36_vk.attn_post, pbindings, &ppush, sizeof(ppush), n_head, n_tok, 1);
        }
        pthread_mutex_unlock(&q36_vk_mu);
    } else {
        q36_vk_prof_scope prof;
        q36_vk_prof_scope_begin(&prof, "attn_softmax_host");
        float *aux_host = q36_vk_tensor_contents_labeled(aux, "submit_wait_attn_softmax_aux");
        float *scores_host = q36_vk_tensor_contents_labeled(scores, "submit_wait_attn_softmax_scores");
        const float *qg_host = q36_vk_tensor_contents_labeled((q36_gpu_tensor *)qg, "submit_wait_attn_softmax_qg");
        q36_vk_attn_host_ctx hctx;
        if (!aux_host || !scores_host || !qg_host) {
            q36_vk_prof_scope_end(&prof);
            q36_gpu_tensor_free(aux);
            return 0;
        }
        hctx.aux = aux_host;
        hctx.scores = scores_host;
        hctx.qg = qg_host;
        hctx.sinks = sinks;
        hctx.score_stride = score_stride / sizeof(float);
        hctx.qg_stride = qg_stride;
        hctx.aux_floats = aux_floats;
        hctx.pos0 = pos0;
        hctx.n_head = n_head;
        hctx.head_dim = head_dim;
        hctx.has_sinks = has_sinks;
        if (n_tok > 1) q36_gpu_parallel_for_rows(n_tok, 2, q36_vk_attn_host_rows, &hctx);
        else q36_vk_attn_host_rows(&hctx, 0, n_tok);
        q36_vk_prof_scope_end(&prof);
    }
    if (!ok) {
        q36_gpu_tensor_free(aux);
        return 0;
    }

    {
        struct {
            uint32_t pos0;
            uint32_t n_head;
            uint32_t n_head_kv;
            uint32_t head_dim;
            uint32_t score_stride;
            uint32_t v_type;
            uint32_t v_row_bytes;
        } rpush = { pos0, n_head, n_head_kv, head_dim, (uint32_t)(score_stride / sizeof(float)),
                    v_cache_type, v_cache_row_bytes };
        const q36_gpu_tensor *rbindings[4] = { scores, v_cache, aux, out };
        pthread_mutex_lock(&q36_vk_mu);
        ok = q36_vk_run_unlocked("attn_reduce", &q36_vk.attn_reduce, rbindings, &rpush, sizeof(rpush),
                                 (head_dim + 63u) / 64u, n_head, n_tok);
        pthread_mutex_unlock(&q36_vk_mu);
    }
    q36_gpu_tensor_free(aux);
    return ok;
}

/* Expert router on the host: softmax, top-k, and routing-weight
 * normalization replicate q36_softmax_inplace() + q36_topk_indices() and the
 * weight loop in q36_forward_ffn() operation for operation.  256 scalars per
 * call; doing it on the host makes expert selection bit-equal to the CPU
 * engine, so a near-tie can never route to a different expert. */
typedef struct {
    uint32_t *selected;
    float *weights;
    const float *logits;
    uint32_t n_expert;
    uint32_t n_used;
    float route_scale;
} q36_vk_router_ctx;

static void q36_vk_router_rows(void *opaque, uint64_t row0, uint64_t row1) {
    q36_vk_router_ctx *ctx = (q36_vk_router_ctx *)opaque;
    float prob[256];
    uint32_t idx[8];
    float val[8];

    for (uint64_t tok = row0; tok < row1; tok++) {
        const float *lt = ctx->logits + tok * ctx->n_expert;
        uint32_t *st = ctx->selected + tok * ctx->n_used;
        float *wt = ctx->weights + tok * ctx->n_used;
        {
            float maxv = -INFINITY;
            double sum = 0.0;
            for (uint32_t i = 0; i < ctx->n_expert; i++) if (lt[i] > maxv) maxv = lt[i];
            for (uint32_t i = 0; i < ctx->n_expert; i++) {
                prob[i] = expf(lt[i] - maxv);
                sum += prob[i];
            }
            if (sum != 0.0) {
                float inv = (float)(1.0 / sum);
                for (uint32_t i = 0; i < ctx->n_expert; i++) prob[i] *= inv;
            }
        }
        for (uint32_t i = 0; i < ctx->n_used; i++) {
            idx[i] = 0;
            val[i] = -INFINITY;
        }
        for (uint32_t i = 0; i < ctx->n_expert; i++) {
            float v = prob[i];
            for (uint32_t j = 0; j < ctx->n_used; j++) {
                if (v > val[j]) {
                    for (uint32_t t = ctx->n_used - 1; t > j; t--) {
                        idx[t] = idx[t - 1];
                        val[t] = val[t - 1];
                    }
                    idx[j] = i;
                    val[j] = v;
                    break;
                }
            }
        }
        {
            double sum = 0.0;
            for (uint32_t i = 0; i < ctx->n_used; i++) sum += val[i];
            if (sum < 6.103515625e-5) sum = 6.103515625e-5;
            for (uint32_t i = 0; i < ctx->n_used; i++) {
                st[i] = idx[i];
                wt[i] = (float)((double)val[i] / sum) * ctx->route_scale;
            }
        }
    }
}

static void q36_vk_router_fast_rows(void *opaque, uint64_t row0, uint64_t row1) {
    q36_vk_router_ctx *ctx = (q36_vk_router_ctx *)opaque;
    uint32_t idx[8];
    float val[8];
    float top_exp[8];

    for (uint64_t tok = row0; tok < row1; tok++) {
        const float *lt = ctx->logits + tok * ctx->n_expert;
        uint32_t *st = ctx->selected + tok * ctx->n_used;
        float *wt = ctx->weights + tok * ctx->n_used;

        for (uint32_t i = 0; i < ctx->n_used; i++) {
            idx[i] = 0;
            val[i] = -INFINITY;
        }
        for (uint32_t i = 0; i < ctx->n_expert; i++) {
            float v = lt[i];
            for (uint32_t j = 0; j < ctx->n_used; j++) {
                if (v > val[j]) {
                    for (uint32_t t = ctx->n_used - 1; t > j; t--) {
                        idx[t] = idx[t - 1];
                        val[t] = val[t - 1];
                    }
                    idx[j] = i;
                    val[j] = v;
                    break;
                }
            }
        }

        double sum = 0.0;
        float maxv = val[0];
        for (uint32_t i = 0; i < ctx->n_used; i++) {
            top_exp[i] = expf(val[i] - maxv);
            sum += top_exp[i];
        }
        if (sum < 6.103515625e-5) sum = 6.103515625e-5;
        for (uint32_t i = 0; i < ctx->n_used; i++) {
            st[i] = idx[i];
            wt[i] = (float)((double)top_exp[i] / sum) * ctx->route_scale;
        }
    }
}

int q36_gpu_router_topk_tensor(q36_gpu_tensor *selected,
                               q36_gpu_tensor *weights,
                               const q36_gpu_tensor *logits,
                               uint32_t n_expert,
                               uint32_t n_used,
                               uint32_t n_tok,
                               float route_scale) {
    const float *logits_host;
    uint32_t *selected_host;
    float *weights_host;
    q36_vk_router_ctx ctx;
    q36_vk_prof_scope prof;
    int ok = 0;
    if (n_expert == 0 || n_expert > 256 || n_used == 0 || n_used > 8 || n_used > n_expert || n_tok == 0) return 0;
    if (!q36_gpu_tensor_range_ok(logits, 0, (uint64_t)n_tok * n_expert * sizeof(float)) ||
        !q36_gpu_tensor_range_ok(selected, 0, (uint64_t)n_tok * n_used * sizeof(uint32_t)) ||
        !q36_gpu_tensor_range_ok(weights, 0, (uint64_t)n_tok * n_used * sizeof(float))) {
        return 0;
    }
    /* Fast-mode routing runs on GPU for every batch size so the expert
     * selection never crosses the host and a token gets identical routing
     * numerics whether it arrives in a prefill chunk, a chunk tail, or a
     * single decode step (warm/cold session equality depends on that). */
    if (!q36_gpu_quality && q36_vk_env_default_on("Q36_VK_ROUTER_GPU")) {
        struct {
            uint32_t n_expert;
            uint32_t n_used;
            uint32_t n_tok;
            float route_scale;
        } push = { n_expert, n_used, n_tok, route_scale };
        const q36_gpu_tensor *bindings[3] = { logits, selected, weights };
        pthread_mutex_lock(&q36_vk_mu);
        ok = q36_vk_run_unlocked("router_topk", &q36_vk.router_topk, bindings, &push, sizeof(push),
                                 n_tok, 1, 1);
        pthread_mutex_unlock(&q36_vk_mu);
        if (ok) return 1;
    }

    q36_vk_prof_scope_begin(&prof, "router_topk_host");
    logits_host = q36_vk_tensor_contents_labeled((q36_gpu_tensor *)logits, "submit_wait_router_logits");
    selected_host = q36_vk_tensor_contents_labeled(selected, "submit_wait_router_selected");
    weights_host = q36_vk_tensor_contents_labeled(weights, "submit_wait_router_weights");
    if (!logits_host || !selected_host || !weights_host) goto done;

    ctx.selected = selected_host;
    ctx.weights = weights_host;
    ctx.logits = logits_host;
    ctx.n_expert = n_expert;
    ctx.n_used = n_used;
    ctx.route_scale = route_scale;
    q36_gpu_parallel_fn fn = (!q36_gpu_quality && n_tok >= 32u && q36_vk_use_router_fast_prefill()) ? q36_vk_router_fast_rows : q36_vk_router_rows;
    if ((uint64_t)n_tok * n_expert >= 1024u) q36_gpu_parallel_for_rows(n_tok, 2, fn, &ctx);
    else fn(&ctx, 0, n_tok);
    ok = 1;
done:
    q36_vk_prof_scope_end(&prof);
    return ok;
}

static uint64_t q36_vk_iq_block_bytes(uint32_t type) {
    switch (type) {
    case Q36_VK_TENSOR_Q2_K: return 84;
    case Q36_VK_TENSOR_Q4_K: return 144;
    case Q36_VK_TENSOR_Q5_K: return 176;
    case Q36_VK_TENSOR_Q6_K: return 210;
    case Q36_VK_TENSOR_IQ2_XXS: return 66;
    case Q36_VK_TENSOR_IQ3_S: return 110;
    case Q36_VK_TENSOR_IQ2_S: return 82;
    default: return 0;
    }
}

static q36_gpu_tensor *q36_vk_quantize_q8_k_alloc_tensor(const q36_gpu_tensor *x,
                                                         uint64_t in_dim,
                                                         uint64_t n_tok) {
    uint64_t blocks = in_dim / Q36_VK_QK_K;
    uint64_t q8_count = 0;
    uint64_t q8_bytes = 0;
    q36_gpu_tensor *q8;

    if (!q36_u64_mul_ok(n_tok, blocks, &q8_count) ||
        !q36_u64_mul_ok(q8_count, sizeof(q36_vk_block_q8_K), &q8_bytes)) {
        return NULL;
    }
    q8 = q36_gpu_tensor_alloc(q8_bytes);
    if (!q8) return NULL;
    if (!q36_gpu_quantize_q8_k_tensor(q8, x, in_dim, n_tok)) {
        q36_gpu_tensor_free(q8);
        return NULL;
    }
    return q8;
}

/* Expert-major routed-pair tiles for moe_matvec: workgroup y owns one expert
 * and up to Q36_VK_MOE_PAIR_TILE pair ids, so the shader can load each expert
 * row block once and update that many accumulators before advancing. */
/* Upper bound on the packed tile count without reading the schedule back:
 * sum(ceil(c_e/4)) <= floor(n_slot/4) + #experts hit.  Expert-major packing
 * is dense from tile 0, so consumers can dispatch this bound instead of the
 * n_slot worst case and skip most dead-tile workgroups. */
static uint32_t q36_vk_moe_tile_bound(uint32_t n_slot, uint32_t n_expert, uint32_t pair_tile) {
    uint32_t live = n_expert < n_slot ? n_expert : n_slot;
    uint32_t bound = n_slot / pair_tile + live;
    return bound < n_slot ? bound : n_slot;
}

/* GPU tile schedule straight from the router selection, no host read.
 * Unused trailing tiles keep the invalid marker and the kernels exit early.
 * With slot_map (SSD streaming) the tile expert fields carry cache slot ids
 * so the consumers index the stream cache buffers directly. */
static q36_gpu_tensor *q36_vk_moe_tiles_gpu(const q36_gpu_tensor *selected, uint32_t n_slot, uint32_t n_expert,
                                            const uint32_t *slot_map, uint32_t pair_tile) {
    if (n_slot == 0 || n_slot > 8192u || n_expert == 0 || n_expert > 256u) return NULL;
    /* n_slot extra words: the build kernel stages the expert-grouped slot
     * ids in the buffer tail instead of LDS. */
    q36_gpu_tensor *tiles = q36_gpu_tensor_alloc((uint64_t)n_slot * (2u + pair_tile) * sizeof(uint32_t));
    q36_gpu_tensor *map = NULL;
    if (!tiles) return NULL;
    if (slot_map) {
        map = q36_gpu_tensor_alloc(256u * sizeof(uint32_t));
        if (!map || !q36_gpu_tensor_write(map, 0, slot_map, 256u * sizeof(uint32_t))) {
            q36_gpu_tensor_free(map);
            q36_gpu_tensor_free(tiles);
            return NULL;
        }
    }
    struct {
        uint32_t n_slot;
        uint32_t n_expert;
        uint32_t streamed;
        uint32_t pair_tile;
    } push = { n_slot, n_expert, slot_map ? 1u : 0u, pair_tile };
    const q36_gpu_tensor *bindings[3] = { selected, tiles, map ? map : tiles };
    pthread_mutex_lock(&q36_vk_mu);
    int ok = q36_vk_run_unlocked("moe_tiles", &q36_vk.moe_tiles, bindings, &push, sizeof(push), 1, 1, 1);
    pthread_mutex_unlock(&q36_vk_mu);
    q36_gpu_tensor_free(map);
    if (!ok) {
        q36_gpu_tensor_free(tiles);
        return NULL;
    }
    return tiles;
}

static q36_gpu_tensor *q36_vk_moe_build_tiles_ex(const uint32_t *selected_host,
                                                  uint64_t n_slot,
                                                  uint32_t n_expert,
                                                  const uint32_t *expert_slot_map,
                                                  uint32_t pair_tile,
                                                  bool skip_unmapped,
                                                  uint32_t *tile_count_out) {
    uint32_t counts[256] = {0};
    uint32_t offsets[257];
    uint32_t next[256];
    uint32_t *grouped_pairs = NULL;
    q36_gpu_tensor *tiles = NULL;
    uint32_t *tiles_host;
    uint64_t words = 0;
    uint64_t bytes = 0;
    uint32_t tile = 0;

    if (!selected_host || !tile_count_out || n_expert == 0 || n_expert > 256 || n_slot == 0 || n_slot > UINT32_MAX) {
        return NULL;
    }
    for (uint64_t pair = 0; pair < n_slot; pair++) {
        uint32_t expert = selected_host[pair];
        if (skip_unmapped && expert_slot_map && expert_slot_map[expert] == UINT32_MAX) continue;
        counts[expert]++;
    }
    offsets[0] = 0;
    *tile_count_out = 0;
    for (uint32_t expert = 0; expert < n_expert; expert++) {
        offsets[expert + 1u] = offsets[expert] + counts[expert];
        *tile_count_out += (counts[expert] + pair_tile - 1u) / pair_tile;
    }
    if (*tile_count_out == 0) return NULL;

    memcpy(next, offsets, (size_t)n_expert * sizeof(next[0]));
    grouped_pairs = malloc((size_t)n_slot * sizeof(grouped_pairs[0]));
    if (!grouped_pairs) return NULL;
    for (uint32_t pair = 0; pair < (uint32_t)n_slot; pair++) {
        uint32_t expert = selected_host[pair];
        if (skip_unmapped && expert_slot_map && expert_slot_map[expert] == UINT32_MAX) continue;
        grouped_pairs[next[expert]++] = pair;
    }
    if (!q36_u64_mul_ok(*tile_count_out, 1u + pair_tile, &words) ||
        !q36_u64_mul_ok(words, sizeof(uint32_t), &bytes)) {
        free(grouped_pairs);
        return NULL;
    }

    tiles = q36_gpu_tensor_alloc(bytes);
    tiles_host = tiles ? q36_vk_tensor_contents_labeled(tiles, "submit_wait_moe_tiles_build") : NULL;
    if (!tiles || !tiles_host) {
        q36_gpu_tensor_free(tiles);
        free(grouped_pairs);
        return NULL;
    }
    memset(tiles_host, 0xff, (size_t)bytes);
    for (uint32_t expert = 0; expert < n_expert; expert++) {
        if (counts[expert] != 0 && expert_slot_map && expert_slot_map[expert] == UINT32_MAX) {
            free(grouped_pairs);
            q36_gpu_tensor_free(tiles);
            return NULL;
        }
        for (uint32_t i = offsets[expert]; i < offsets[expert + 1u]; i += pair_tile) {
            uint32_t base = tile * (1u + pair_tile);
            tiles_host[base] = expert_slot_map ? expert_slot_map[expert] : expert;
            for (uint32_t u = 0; u < pair_tile && i + u < offsets[expert + 1u]; u++) {
                tiles_host[base + 1u + u] = grouped_pairs[i + u];
            }
            tile++;
        }
    }
    free(grouped_pairs);
    return tiles;
}

static q36_gpu_tensor *q36_vk_moe_build_tiles(const uint32_t *selected_host,
                                               uint64_t n_slot,
                                               uint32_t n_expert,
                                               const uint32_t *expert_slot_map,
                                               uint32_t *tile_count_out) {
    return q36_vk_moe_build_tiles_ex(selected_host, n_slot, n_expert,
                                     expert_slot_map, Q36_VK_MOE_PAIR_TILE,
                                     false, tile_count_out);
}

/* One routed expert matvec batch over n_tok tokens.  Pairs are grouped by
 * expert on the host so workgroup y can own one expert and up to
 * Q36_VK_MOE_PAIR_TILE pair ids, loading each expert row block once and
 * updating that many q8 accumulation chains before advancing.
 *
 * The whole expert bank and its .scale tensor are cached on device through
 * the weight cache and the router's expert ids index them directly, so a
 * decode step moves no expert bytes at all.  If the bank does not fit the
 * heaps the runtime falls back to packing one expert copy per tile; both
 * paths feed the shader the same expert-major tile schedule. */
static int q36_vk_moe_matvec(q36_gpu_tensor *out,
                              const void *model_map,
                              uint64_t model_size,
                              const q36_gpu_moe_weight *w,
                              const q36_gpu_tensor *qx,
                              bool x_per_expert,
                              bool fast_allowed,
                              q36_vk_stream_part stream_part,
                              const uint32_t *stream_slot_map,
                              bool resident_bank,
                              const q36_gpu_tensor *selected,
                              uint32_t n_used,
                              uint32_t n_tok,
                             uint32_t in_dim,
                             uint32_t out_dim,
                             uint32_t n_expert) {
    uint64_t block_bytes = q36_vk_iq_block_bytes(w->type);
    uint64_t blocks;
    uint64_t row_bytes;
    uint64_t expert_stride;
    uint64_t weight_bytes;
    uint64_t scale_bytes = (uint64_t)n_expert * sizeof(float);
    uint64_t n_slot = (uint64_t)n_tok * n_used;
    uint64_t packed_weight_bytes = 0;
    uint64_t packed_scale_bytes = 0;
    const unsigned char *source;
    const unsigned char *scales_source = NULL;
    const uint32_t *selected_host;
    q36_gpu_tensor *tiles = NULL;
    q36_gpu_tensor *packed_weights = NULL;
    q36_gpu_tensor *packed_scales = NULL;
    const uint32_t *tiles_host;
    uint32_t tile_count = 0;
    unsigned char *packed_weight_host;
    float *packed_scales_host = NULL;

    if (block_bytes == 0 || in_dim == 0 || (in_dim % Q36_VK_QK_K) != 0 || out_dim == 0 || n_tok == 0) return 0;
    blocks = in_dim / Q36_VK_QK_K;
    row_bytes = blocks * block_bytes;
    expert_stride = row_bytes * out_dim;
    if (expert_stride > UINT32_MAX ||
        !q36_u64_mul_ok(expert_stride, n_expert, &weight_bytes)) {
        return 0;
    }
    source = q36_gpu_weight_bytes(model_map, model_size, w->offset, weight_bytes);
    if (!source) return 0;
    if (w->has_scales) {
        scales_source = q36_gpu_weight_bytes(model_map, model_size, w->scales_offset, scale_bytes);
        if (!scales_source) return 0;
    }
    if (!q36_gpu_tensor_range_ok(out, 0, n_slot * out_dim * sizeof(float)) ||
        !q36_gpu_tensor_range_ok(selected, 0, n_slot * sizeof(uint32_t)) ||
        !q36_gpu_tensor_range_ok(qx, 0, (x_per_expert ? n_slot : n_tok) * blocks * sizeof(q36_vk_block_q8_K))) {
        return 0;
    }

    tiles_host = NULL;
    if (!q36_gpu_quality && n_slot <= 4096u && stream_part == Q36_VK_STREAM_PART_NONE &&
        (!q36_vk_stream.enabled || resident_bank) && q36_vk_moe_bank_cache) {
        tiles = q36_vk_moe_tiles_gpu(selected, (uint32_t)n_slot, n_expert, NULL, Q36_VK_MOE_PAIR_TILE);
        tile_count = q36_vk_moe_tile_bound((uint32_t)n_slot, n_expert, Q36_VK_MOE_PAIR_TILE);
    }
    if (!tiles) {
        selected_host = q36_vk_tensor_contents_labeled((q36_gpu_tensor *)selected, "submit_wait_moe_selected");
        if (!selected_host) return 0;
        for (uint64_t slot = 0; slot < n_slot; slot++) {
            if (selected_host[slot] >= n_expert) return 0;
        }
        tiles = q36_vk_moe_build_tiles(selected_host,
                                       n_slot,
                                       n_expert,
                                       stream_part != Q36_VK_STREAM_PART_NONE ? stream_slot_map : NULL,
                                       &tile_count);
        tiles_host = tiles ? q36_vk_tensor_contents_labeled(tiles, "submit_wait_moe_tiles_read") : NULL;
        if (!tiles || !tiles_host || tile_count == 0) {
            q36_gpu_tensor_free(tiles);
            return 0;
        }
    }

    struct {
        uint32_t out_dim;
        uint32_t blocks;
        uint32_t row_bytes;
        uint32_t expert_stride;
        uint32_t type;
        uint32_t n_used;
        uint32_t has_scales;
        uint32_t x_per_expert;
        uint32_t packed_weights;
    } push = {
        out_dim, (uint32_t)blocks, (uint32_t)row_bytes, (uint32_t)expert_stride,
        w->type, n_used, w->has_scales ? 1u : 0u, x_per_expert ? 1u : 0u, 0u,
    };

    if (stream_part != Q36_VK_STREAM_PART_NONE && stream_slot_map) {
        pthread_mutex_lock(&q36_vk_mu);
        q36_gpu_tensor *tables = q36_vk_iq_tables_unlocked();
        const q36_gpu_tensor *bank = q36_vk_stream_weight_tensor_unlocked(stream_part);
        const q36_gpu_tensor *bank_scales = q36_vk_stream_scale_tensor_unlocked(stream_part);
        int ok = tables && bank && (!w->has_scales || bank_scales);
        if (ok) {
            const q36_gpu_tensor *bindings[6] = {
                bank, qx, tiles, w->has_scales ? bank_scales : bank, out, tables,
            };
            q36_vk_kernel *kernel = (!q36_gpu_quality && q36_vk_use_moe_fast() && fast_allowed) ? &q36_vk.moe_matvec_fast : &q36_vk.moe_matvec;
            ok = q36_vk_run_unlocked(w->type == Q36_VK_TENSOR_Q2_K && x_per_expert ? "moe_q2k_down_stream" :
                                     (w->type == Q36_VK_TENSOR_IQ2_XXS ? "moe_iq2_matvec_stream" : "moe_matvec_stream"),
                                     kernel, bindings, &push, sizeof(push),
                                     (out_dim + 7u) / 8u, tile_count, 1);
        }
        pthread_mutex_unlock(&q36_vk_mu);
        if (ok) {
            q36_gpu_tensor_free(tiles);
            return 1;
        }
    }

    if ((!q36_vk_stream.enabled || resident_bank) && q36_vk_moe_bank_cache) {
        pthread_mutex_lock(&q36_vk_mu);
        q36_gpu_tensor *tables = q36_vk_iq_tables_unlocked();
        q36_gpu_tensor *bank = tables ? q36_vk_weight_get_unlocked(source, weight_bytes) : NULL;
        q36_gpu_tensor *bank_scales = NULL;
        if (bank && w->has_scales) {
            bank_scales = q36_vk_weight_get_unlocked(scales_source, scale_bytes);
            if (!bank_scales) bank = NULL;
        }
        if (bank) {
            const q36_gpu_tensor *bindings[6] = {
                bank, qx, tiles, bank_scales ? bank_scales : bank, out, tables,
            };
            q36_vk_kernel *kernel = (!q36_gpu_quality && q36_vk_use_moe_fast() && fast_allowed) ? &q36_vk.moe_matvec_fast : &q36_vk.moe_matvec;
            int ok = q36_vk_run_unlocked(w->type == Q36_VK_TENSOR_Q2_K && x_per_expert ? "moe_q2k_down" :
                                         (w->type == Q36_VK_TENSOR_IQ2_XXS ? "moe_iq2_matvec" : "moe_matvec"),
                                         kernel, bindings, &push, sizeof(push),
                                         (out_dim + 7u) / 8u, tile_count, 1);
            pthread_mutex_unlock(&q36_vk_mu);
            q36_gpu_tensor_free(tiles);
            return ok;
        }
        q36_vk_moe_bank_cache = false;
        pthread_mutex_unlock(&q36_vk_mu);
        fprintf(stderr, "q36: Vulkan expert bank cache exhausted device memory, "
                        "falling back to per-call expert staging\n");
    }

    /* Packed per-call staging needs expert ids on the host; GPU-built tiles
     * only serve the resident-bank dispatch above. */
    if (!tiles_host) {
        q36_gpu_tensor_free(tiles);
        return 0;
    }
    if (!q36_u64_mul_ok(tile_count, expert_stride, &packed_weight_bytes)) {
        q36_gpu_tensor_free(tiles);
        return 0;
    }
    packed_weights = q36_gpu_tensor_alloc(packed_weight_bytes);
    if (w->has_scales) {
        packed_scale_bytes = (uint64_t)tile_count * sizeof(float);
        packed_scales = q36_gpu_tensor_alloc(packed_scale_bytes);
    }
    packed_weight_host = packed_weights ? q36_vk_tensor_contents_labeled(packed_weights, "submit_wait_moe_pack_weights") : NULL;
    packed_scales_host = packed_scales ? q36_vk_tensor_contents_labeled(packed_scales, "submit_wait_moe_pack_scales") : NULL;
    if (!packed_weights || !packed_weight_host ||
        (w->has_scales && (!packed_scales || !packed_scales_host))) {
        q36_gpu_tensor_free(packed_scales);
        q36_gpu_tensor_free(packed_weights);
        q36_gpu_tensor_free(tiles);
        return 0;
    }
    for (uint32_t tile = 0; tile < tile_count; tile++) {
        const uint32_t expert = tiles_host[(uint64_t)tile * Q36_VK_MOE_TILE_WORDS];
        memcpy(packed_weight_host + (uint64_t)tile * expert_stride,
               source + (uint64_t)expert * expert_stride,
               (size_t)expert_stride);
        if (packed_scales_host) packed_scales_host[tile] = ((const float *)scales_source)[expert];
    }
    push.packed_weights = 1u;

    pthread_mutex_lock(&q36_vk_mu);
    q36_gpu_tensor *tables = q36_vk_iq_tables_unlocked();
    int ok = tables != NULL;
    if (ok) {
        const q36_gpu_tensor *bindings[6] = {
            packed_weights,
            qx,
            tiles,
            w->has_scales ? packed_scales : packed_weights,
            out,
            tables,
        };
        q36_vk_kernel *kernel = (!q36_gpu_quality && q36_vk_use_moe_fast() && fast_allowed) ? &q36_vk.moe_matvec_fast : &q36_vk.moe_matvec;
        ok = q36_vk_run_unlocked(w->type == Q36_VK_TENSOR_Q2_K && x_per_expert ? "moe_q2k_down" :
                                 (w->type == Q36_VK_TENSOR_IQ2_XXS ? "moe_iq2_matvec" : "moe_matvec"),
                                 kernel, bindings, &push, sizeof(push),
                                 (out_dim + 7u) / 8u, tile_count, 1);
    }
    pthread_mutex_unlock(&q36_vk_mu);
    q36_gpu_tensor_free(packed_scales);
    q36_gpu_tensor_free(packed_weights);
    q36_gpu_tensor_free(tiles);
    return ok;
}

static int q36_vk_moe_gate_up_iq2_swiglu(q36_gpu_tensor *out,
                                          const void *model_map,
                                          uint64_t model_size,
                                          const q36_gpu_moe_weight *gate,
                                          const q36_gpu_moe_weight *up,
                                          const q36_gpu_tensor *selected,
                                          const q36_gpu_tensor *qx,
                                          const uint32_t *stream_slot_map,
                                          bool resident_bank,
                                          uint32_t n_used,
                                          uint32_t n_tok,
                                          uint32_t in_dim,
                                         uint32_t mid_dim,
                                         uint32_t n_expert) {
    const uint64_t block_bytes = 66;
    uint64_t blocks;
    uint64_t row_bytes;
    uint64_t expert_stride;
    uint64_t weight_bytes;
    uint64_t scale_bytes = (uint64_t)n_expert * sizeof(float);
    uint64_t n_slot = (uint64_t)n_tok * n_used;
    const unsigned char *gate_source;
    const unsigned char *up_source;
    const unsigned char *gate_scales_source = NULL;
    const unsigned char *up_scales_source = NULL;
    const uint32_t *selected_host;
    q36_gpu_tensor *tiles = NULL;
    uint32_t tile_count = 0;
    int ok = 0;

    if (q36_gpu_quality || !q36_vk_use_moe_gate_up() || (n_tok > 1u && n_tok < 32u) ||
        (!q36_vk_moe_bank_cache && !stream_slot_map)) return 0;
    if (!gate || !up || !selected || !qx || !out || n_used == 0 || n_tok == 0 || n_expert == 0) return 0;
    if (gate->type != Q36_VK_TENSOR_IQ2_XXS || up->type != Q36_VK_TENSOR_IQ2_XXS) return 0;
    if (in_dim == 0 || (in_dim % Q36_VK_QK_K) != 0 || mid_dim == 0) return 0;
    blocks = in_dim / Q36_VK_QK_K;
    row_bytes = blocks * block_bytes;
    expert_stride = row_bytes * mid_dim;
    if (expert_stride > UINT32_MAX || row_bytes > UINT32_MAX || blocks > UINT32_MAX ||
        !q36_u64_mul_ok(expert_stride, n_expert, &weight_bytes)) {
        return 0;
    }
    gate_source = q36_gpu_weight_bytes(model_map, model_size, gate->offset, weight_bytes);
    up_source = q36_gpu_weight_bytes(model_map, model_size, up->offset, weight_bytes);
    if (!gate_source || !up_source) return 0;
    if (gate->has_scales) {
        gate_scales_source = q36_gpu_weight_bytes(model_map, model_size, gate->scales_offset, scale_bytes);
        if (!gate_scales_source) return 0;
    }
    if (up->has_scales) {
        up_scales_source = q36_gpu_weight_bytes(model_map, model_size, up->scales_offset, scale_bytes);
        if (!up_scales_source) return 0;
    }
    if (!q36_gpu_tensor_range_ok(out, 0, n_slot * mid_dim * sizeof(float)) ||
        !q36_gpu_tensor_range_ok(selected, 0, n_slot * sizeof(uint32_t)) ||
        !q36_gpu_tensor_range_ok(qx, 0, (uint64_t)n_tok * blocks * sizeof(q36_vk_block_q8_K))) {
        return 0;
    }

    if (n_slot <= 4096u && !stream_slot_map &&
        (!q36_vk_stream.enabled || resident_bank)) {
        tiles = q36_vk_moe_tiles_gpu(selected, (uint32_t)n_slot, n_expert, NULL, Q36_VK_MOE_PAIR_TILE);
        tile_count = q36_vk_moe_tile_bound((uint32_t)n_slot, n_expert, Q36_VK_MOE_PAIR_TILE);
    }
    if (!tiles) {
        selected_host = q36_vk_tensor_contents_labeled((q36_gpu_tensor *)selected, "submit_wait_moe_gate_up_selected");
        if (!selected_host) return 0;
        for (uint64_t slot = 0; slot < n_slot; slot++) {
            if (selected_host[slot] >= n_expert) return 0;
        }
        tiles = q36_vk_moe_build_tiles(selected_host,
                                       n_slot,
                                       n_expert,
                                       stream_slot_map,
                                       &tile_count);
        if (!tiles || tile_count == 0) {
            q36_gpu_tensor_free(tiles);
            return 0;
        }
    }

    struct {
        uint32_t out_dim;
        uint32_t blocks;
        uint32_t row_bytes;
        uint32_t expert_stride;
        uint32_t n_used;
        uint32_t has_gate_scales;
        uint32_t has_up_scales;
    } push = {
        mid_dim, (uint32_t)blocks, (uint32_t)row_bytes, (uint32_t)expert_stride,
        n_used, gate->has_scales ? 1u : 0u, up->has_scales ? 1u : 0u,
    };

    pthread_mutex_lock(&q36_vk_mu);
    q36_gpu_tensor *tables = q36_vk_iq_tables_unlocked();
    const q36_gpu_tensor *gate_bank = NULL;
    const q36_gpu_tensor *up_bank = NULL;
    const q36_gpu_tensor *gate_scales = NULL;
    const q36_gpu_tensor *up_scales = NULL;
    if (stream_slot_map) {
        gate_bank = q36_vk_stream_weight_tensor_unlocked(Q36_VK_STREAM_PART_GATE);
        up_bank = q36_vk_stream_weight_tensor_unlocked(Q36_VK_STREAM_PART_UP);
        gate_scales = q36_vk_stream_scale_tensor_unlocked(Q36_VK_STREAM_PART_GATE);
        up_scales = q36_vk_stream_scale_tensor_unlocked(Q36_VK_STREAM_PART_UP);
    } else {
        gate_bank = tables ? q36_vk_weight_get_unlocked(gate_source, weight_bytes) : NULL;
        up_bank = gate_bank ? q36_vk_weight_get_unlocked(up_source, weight_bytes) : NULL;
        if (up_bank && gate->has_scales) gate_scales = q36_vk_weight_get_unlocked(gate_scales_source, scale_bytes);
        if (up_bank && (!gate->has_scales || gate_scales) && up->has_scales) up_scales = q36_vk_weight_get_unlocked(up_scales_source, scale_bytes);
    }
    if (tables && up_bank && (!gate->has_scales || gate_scales) && (!up->has_scales || up_scales)) {
        const q36_gpu_tensor *bindings[8] = {
            gate_bank,
            up_bank,
            qx,
            tiles,
            gate->has_scales ? gate_scales : gate_bank,
            up->has_scales ? up_scales : up_bank,
            out,
            tables,
        };
        ok = q36_vk_run_unlocked("moe_iq2_gate_up", &q36_vk.moe_gate_up, bindings, &push, sizeof(push),
                                 (mid_dim + 7u) / 8u, tile_count, 1);
    }
    pthread_mutex_unlock(&q36_vk_mu);
    q36_gpu_tensor_free(tiles);
    return ok;
}

/* Routed expert FFN, the routed half of q36_forward_ffn() including all
 * .scale applications, batched over n_tok rows of x with per-token expert
 * selections.  The fast IQ2_XXS path fuses gate/up/SwiGLU into one dispatch;
 * other layouts and --quality keep the separate exact gate/up/SwiGLU route. */
int q36_gpu_moe_ffn_q8_tensor(q36_gpu_tensor *out,
                              const void *model_map,
                              uint64_t model_size,
                              const q36_gpu_moe_weight *gate,
                              const q36_gpu_moe_weight *up,
                              const q36_gpu_moe_weight *down,
                              const q36_gpu_tensor *selected,
                              const q36_gpu_tensor *weights,
                              uint32_t layer,
                              uint32_t n_used,
                              const q36_gpu_tensor *qx,
                              uint32_t n_tok,
                              uint32_t in_dim,
                              uint32_t mid_dim,
                              uint32_t out_dim,
                              uint32_t n_expert) {
    uint64_t mid_bytes = (uint64_t)n_used * mid_dim * sizeof(float);
    uint64_t down_bytes = (uint64_t)n_used * out_dim * sizeof(float);
    q36_gpu_tensor *gate8 = NULL;
    q36_gpu_tensor *up8 = NULL;
    q36_gpu_tensor *mid8 = NULL;
    q36_gpu_tensor *qmid = NULL;
    q36_gpu_tensor *down8 = NULL;
    uint32_t stream_slot_map[256];
    const uint32_t *stream_slots = NULL;
    bool resident_bank = q36_vk_stream.enabled && layer < q36_vk_stream.full_layers;
    int ok = 0;

    if (!gate || !up || !down || !qx || n_used == 0 || n_expert == 0 || n_tok == 0) return 0;
    if (in_dim == 0 || (in_dim % Q36_VK_QK_K) != 0 ||
        mid_dim == 0 || (mid_dim % Q36_VK_QK_K) != 0 || out_dim == 0) {
        return 0;
    }
    if (!q36_gpu_tensor_range_ok(out, 0, (uint64_t)n_tok * out_dim * sizeof(float)) ||
        !q36_gpu_tensor_range_ok(qx, 0, (uint64_t)n_tok * (in_dim / Q36_VK_QK_K) * sizeof(q36_vk_block_q8_K)) ||
        !q36_gpu_tensor_range_ok(selected, 0, (uint64_t)n_tok * n_used * sizeof(uint32_t)) ||
        !q36_gpu_tensor_range_ok(weights, 0, (uint64_t)n_tok * n_used * sizeof(float))) {
        return 0;
    }

    if (q36_vk_stream.enabled && !resident_bank && n_expert <= 256) {
        uint64_t gate_block = q36_vk_iq_block_bytes(gate->type);
        uint64_t up_block = q36_vk_iq_block_bytes(up->type);
        uint64_t down_block = q36_vk_iq_block_bytes(down->type);
        uint64_t gate_row = gate_block * (in_dim / Q36_VK_QK_K);
        uint64_t up_row = up_block * (in_dim / Q36_VK_QK_K);
        uint64_t down_row = down_block * (mid_dim / Q36_VK_QK_K);
        const uint32_t *selected_host = q36_vk_tensor_contents_labeled((q36_gpu_tensor *)selected,
                                                                       "submit_wait_moe_stream_selected");
        q36_gpu_stream_expert_table table;
        memset(&table, 0, sizeof(table));
        table.model_map = model_map;
        table.model_size = model_size;
        table.layer = layer;
        table.n_total_expert = n_expert;
        table.gate_offset = gate->offset;
        table.up_offset = up->offset;
        table.down_offset = down->offset;
        table.gate_scales_offset = gate->scales_offset;
        table.up_scales_offset = up->scales_offset;
        table.down_scales_offset = down->scales_offset;
        table.gate_type = gate->type;
        table.up_type = up->type;
        table.down_type = down->type;
        table.has_gate_scales = gate->has_scales;
        table.has_up_scales = up->has_scales;
        table.has_down_scales = down->has_scales;
        if (gate_block && up_block && down_block && selected_host &&
            !q36_u64_mul_ok(gate_row, mid_dim, &table.gate_expert_bytes)) {
            selected_host = NULL;
        }
        if (selected_host && !q36_u64_mul_ok(up_row, mid_dim, &table.up_expert_bytes)) selected_host = NULL;
        if (selected_host && !q36_u64_mul_ok(down_row, out_dim, &table.down_expert_bytes)) selected_host = NULL;
        if (selected_host && q36_vk_stream_cache_prepare_u32(&table,
                                                             selected_host,
                                                             (uint64_t)n_tok * n_used,
                                                             stream_slot_map)) {
            stream_slots = stream_slot_map;
        }
    }

    mid8 = q36_gpu_tensor_alloc(mid_bytes * n_tok);
    down8 = q36_gpu_tensor_alloc(down_bytes * n_tok);
    ok = qx && mid8 && down8 &&
          q36_vk_moe_gate_up_iq2_swiglu(mid8, model_map, model_size, gate, up,
                                         selected, qx, stream_slots, resident_bank,
                                         n_used, n_tok, in_dim, mid_dim, n_expert);
    if (!ok && mid8 && down8) {
        gate8 = q36_gpu_tensor_alloc(mid_bytes * n_tok);
        up8 = q36_gpu_tensor_alloc(mid_bytes * n_tok);
        ok = gate8 && up8 &&
             q36_vk_moe_matvec(gate8, model_map, model_size, gate, qx, false, false,
                                stream_slots ? Q36_VK_STREAM_PART_GATE : Q36_VK_STREAM_PART_NONE,
                                stream_slots, resident_bank,
                                selected, n_used, n_tok, in_dim, mid_dim, n_expert) &&
             q36_vk_moe_matvec(up8, model_map, model_size, up, qx, false, true,
                                stream_slots ? Q36_VK_STREAM_PART_UP : Q36_VK_STREAM_PART_NONE,
                                stream_slots, resident_bank,
                                selected, n_used, n_tok, in_dim, mid_dim, n_expert) &&
             q36_gpu_swiglu_tensor(mid8, gate8, up8, n_tok * n_used * mid_dim, 0.0f, 1.0f);
    }
    if (ok) {
        qmid = q36_vk_quantize_q8_k_alloc_tensor(mid8, mid_dim, (uint64_t)n_tok * n_used);
        ok = qmid &&
               q36_vk_moe_matvec(down8, model_map, model_size, down, qmid, true,
                                  down->type == Q36_VK_TENSOR_Q2_K,
                                  stream_slots ? Q36_VK_STREAM_PART_DOWN : Q36_VK_STREAM_PART_NONE,
                                  stream_slots, resident_bank,
                                  selected, n_used, n_tok, mid_dim, out_dim, n_expert);
    }
    if (ok) {
        struct {
            uint32_t n;
            uint32_t n_used;
        } push = { out_dim, n_used };
        const q36_gpu_tensor *bindings[3] = { down8, weights, out };
        pthread_mutex_lock(&q36_vk_mu);
        ok = q36_vk_run_unlocked("moe_reduce", &q36_vk.moe_reduce, bindings, &push, sizeof(push),
                                 (out_dim + 255u) / 256u, n_tok, 1);
        pthread_mutex_unlock(&q36_vk_mu);
    }

    q36_gpu_tensor_free(down8);
    q36_gpu_tensor_free(qmid);
    q36_gpu_tensor_free(mid8);
    q36_gpu_tensor_free(up8);
    q36_gpu_tensor_free(gate8);
    return ok;
}

/* Whole routed FFN against f32 activations for decode and prefill chunks:
 * GPU tile schedule, fused IQ2_XXS gate/up/SwiGLU, Q2_K down, weighted
 * reduce. No q8_K stage anywhere, no host read of the selection. */
int q36_gpu_moe_ffn_f32_tensor(q36_gpu_tensor *out,
                                      const void *model_map,
                                      uint64_t model_size,
                                      const q36_gpu_moe_weight *gate,
                                      const q36_gpu_moe_weight *up,
                                      const q36_gpu_moe_weight *down,
                                      const q36_gpu_tensor *selected,
                                      const q36_gpu_tensor *weights,
                                      uint32_t layer,
                                      uint32_t n_used,
                                      const q36_gpu_tensor *x,
                                      uint32_t n_tok,
                                      uint32_t in_dim,
                                      uint32_t mid_dim,
                                      uint32_t out_dim,
                                      uint32_t n_expert) {
    uint64_t gu_stride = (uint64_t)(in_dim / Q36_VK_QK_K) * 66u * mid_dim;
    uint64_t down_stride = (uint64_t)(mid_dim / Q36_VK_QK_K) * 84u * out_dim;
    uint64_t scale_bytes = (uint64_t)n_expert * sizeof(float);
    uint64_t n_slot = (uint64_t)n_tok * n_used;
    q36_gpu_tensor *tiles = NULL;
    q36_gpu_tensor *sel_slots = NULL;
    q36_gpu_tensor *mid8 = NULL;
    q36_gpu_tensor *down8 = NULL;
    const uint32_t *selected_host = NULL;
    q36_gpu_stream_expert_table stream_table = {0};
    uint32_t selected_unique[256];
    uint32_t unique = 0;
    uint32_t stream_cap = 0;
    bool overflow = false;
    bool streamed = q36_vk_stream.enabled && layer >= q36_vk_stream.full_layers;
    int ok = 0;

    if (q36_gpu_quality || n_tok == 0 || n_used == 0 || n_used > 8u || n_slot > 8192u) return 0;
    if (!q36_vk_env_default_on("Q36_VK_MOE_F32B")) return 0;
    if (!streamed && !q36_vk_moe_bank_cache) return 0;
    if (gate->type != Q36_VK_TENSOR_IQ2_XXS || up->type != Q36_VK_TENSOR_IQ2_XXS ||
        down->type != Q36_VK_TENSOR_Q2_K) return 0;
    if (in_dim == 0 || (in_dim % Q36_VK_QK_K) != 0 || (in_dim % 4u) != 0 ||
        mid_dim == 0 || (mid_dim % Q36_VK_QK_K) != 0 || out_dim == 0) return 0;
    if (gu_stride * n_expert > UINT32_MAX || down_stride * n_expert > UINT32_MAX) return 0;

    const unsigned char *gate_src = q36_gpu_weight_bytes(model_map, model_size, gate->offset, gu_stride * n_expert);
    const unsigned char *up_src = q36_gpu_weight_bytes(model_map, model_size, up->offset, gu_stride * n_expert);
    const unsigned char *down_src = q36_gpu_weight_bytes(model_map, model_size, down->offset, down_stride * n_expert);
    if (!gate_src || !up_src || !down_src) return 0;
    const unsigned char *gate_sc = gate->has_scales ? q36_gpu_weight_bytes(model_map, model_size, gate->scales_offset, scale_bytes) : NULL;
    const unsigned char *up_sc = up->has_scales ? q36_gpu_weight_bytes(model_map, model_size, up->scales_offset, scale_bytes) : NULL;
    const unsigned char *down_sc = down->has_scales ? q36_gpu_weight_bytes(model_map, model_size, down->scales_offset, scale_bytes) : NULL;
    if ((gate->has_scales && !gate_sc) || (up->has_scales && !up_sc) || (down->has_scales && !down_sc)) return 0;
    if (!q36_gpu_tensor_range_ok(out, 0, (uint64_t)n_tok * out_dim * sizeof(float)) ||
        !q36_gpu_tensor_range_ok(x, 0, (uint64_t)n_tok * in_dim * sizeof(float)) ||
        !q36_gpu_tensor_range_ok(selected, 0, n_slot * sizeof(uint32_t)) ||
        !q36_gpu_tensor_range_ok(weights, 0, n_slot * sizeof(float))) {
        return 0;
    }

    /* SSD streaming: the selected experts stream into the cache and the
     * tile schedule carries cache slot ids, so the same fused kernels read
     * the stream buffers at slot * stride — identical bytes, identical
     * math, no q8_K staging.  The selection read is a host sync, but the
     * cache prepare needs it on the host anyway. */
    uint32_t stream_slot_map[256];
    if (streamed) {
        if (n_expert > 256u) return 0;
        selected_host = q36_vk_tensor_contents_labeled((q36_gpu_tensor *)selected,
                                                       "submit_wait_moe_stream_selected");
        if (!selected_host) return 0;
        stream_table.model_map = model_map;
        stream_table.model_size = model_size;
        stream_table.layer = layer;
        stream_table.n_total_expert = n_expert;
        stream_table.gate_offset = gate->offset;
        stream_table.up_offset = up->offset;
        stream_table.down_offset = down->offset;
        stream_table.gate_scales_offset = gate->scales_offset;
        stream_table.up_scales_offset = up->scales_offset;
        stream_table.down_scales_offset = down->scales_offset;
        stream_table.gate_type = gate->type;
        stream_table.up_type = up->type;
        stream_table.down_type = down->type;
        stream_table.has_gate_scales = gate->has_scales;
        stream_table.has_up_scales = up->has_scales;
        stream_table.has_down_scales = down->has_scales;
        stream_table.gate_expert_bytes = gu_stride;
        stream_table.up_expert_bytes = gu_stride;
        stream_table.down_expert_bytes = down_stride;
        stream_cap = q36_vk_stream_cache_cap(&stream_table);
        if (stream_cap == 0) return 0;
        bool seen[256] = {0};
        for (uint64_t i = 0; i < n_slot; i++) {
            uint32_t expert = selected_host[i];
            if (expert >= n_expert) return 0;
            if (!seen[expert]) {
                seen[expert] = true;
                selected_unique[unique++] = expert;
            }
        }
        overflow = unique > stream_cap;
        if (overflow) q36_vk_stream_note_routes(&stream_table, selected_host, n_slot);
        if (!overflow && !q36_vk_stream_cache_prepare_u32(
                &stream_table, selected_host, n_slot, stream_slot_map)) return 0;
        /* The kernels compute weight offsets in 32-bit halfword units, so
         * the whole slot arena must stay indexable. */
        if (gu_stride * q36_vk_stream.cap > (uint64_t)UINT32_MAX * 2u ||
            down_stride * q36_vk_stream.cap > (uint64_t)UINT32_MAX * 2u) return 0;
        if (!overflow && n_tok == 1u) {
            /* Decode identity schedule with slot ids instead of expert ids. */
            uint32_t mapped[8];
            for (uint32_t i = 0; i < (uint32_t)n_slot; i++) mapped[i] = stream_slot_map[selected_host[i]];
            sel_slots = q36_gpu_tensor_alloc(n_slot * sizeof(uint32_t));
            if (!sel_slots || !q36_gpu_tensor_write(sel_slots, 0, mapped, n_slot * sizeof(uint32_t))) goto done;
        }
    }

    /* Decode: one distinct expert per slot, so the selection itself is the
     * schedule and the tile build dispatch is skipped. */
    bool identity = n_tok == 1u && !overflow;
    bool down_sum_decode = identity && q36_vk_use_moe_down_sum_decode();
    /* Prefill batches route to the register-tiled GEMM kernels: wider
     * expert token tiles amortize the dequant and weight reads that bound
     * the matvec path. Small batches (chunk tails, MTP verify windows)
     * keep the matvec kernels, whose per-slot arithmetic order matches
     * decode bit for bit; the GEMM's k order is fixed but differs from
     * decode, which the relaxed warm==cold session gate absorbs. */
    bool gemm = !identity && !q36_vk_micro_batch &&
                n_tok >= q36_vk_moe_gemm_min() &&
                q36_vk_env_default_on("Q36_VK_MOE_GEMM");
    uint32_t pair_tile = gemm ? Q36_VK_MOE_GEMM_TILE : Q36_VK_MOE_PAIR_TILE;
    uint32_t tile_count = (uint32_t)n_slot;
    if (!identity && !overflow) {
        tiles = q36_vk_moe_tiles_gpu(selected, (uint32_t)n_slot, n_expert,
                                     streamed ? stream_slot_map : NULL, pair_tile);
        if (!tiles) goto done;
        tile_count = q36_vk_moe_tile_bound((uint32_t)n_slot, n_expert, pair_tile);
    }
    mid8 = q36_gpu_tensor_alloc(n_slot * mid_dim * sizeof(float));
    if (!down_sum_decode) down8 = q36_gpu_tensor_alloc(n_slot * out_dim * sizeof(float));
    if (!mid8 || (!down_sum_decode && !down8)) goto done;

    uint32_t wave = 0;
    ok = 1;
    /* Overflow runs the original full-batch expert tiles in cache-sized
     * waves and reduces once. Pair arithmetic is unchanged, while a small
     * resident cache never selects the Q8 fallback. */
    do {
        q36_gpu_tensor *wave_tiles = tiles;
        uint32_t wave_tile_count = tile_count;
        uint32_t wave_experts = unique - wave;
        if (overflow) {
            if (wave_experts > stream_cap) wave_experts = stream_cap;
            if (!q36_vk_stream_cache_prepare_u32(&stream_table,
                                                  selected_unique + wave,
                                                  wave_experts,
                                                  stream_slot_map)) {
                ok = 0;
                break;
            }
            wave_tiles = q36_vk_moe_build_tiles_ex(selected_host, n_slot,
                                                    n_expert, stream_slot_map,
                                                    pair_tile, true,
                                                    &wave_tile_count);
            if (!wave_tiles || wave_tile_count == 0) {
                q36_gpu_tensor_free(wave_tiles);
                ok = 0;
                break;
            }
        }

        pthread_mutex_lock(&q36_vk_mu);
        q36_gpu_tensor *tables = q36_vk_iq_tables_unlocked();
        q36_gpu_tensor *gate_bank, *up_bank, *down_bank;
        q36_gpu_tensor *gate_scales, *up_scales, *down_scales;
        if (streamed) {
            gate_bank = q36_vk_stream.gate;
            up_bank = q36_vk_stream.up;
            down_bank = q36_vk_stream.down;
            gate_scales = gate->has_scales ? q36_vk_stream.gate_scales : NULL;
            up_scales = up->has_scales ? q36_vk_stream.up_scales : NULL;
            down_scales = down->has_scales ? q36_vk_stream.down_scales : NULL;
            ok = tables && gate_bank && up_bank && down_bank;
        } else {
            gate_bank = tables ? q36_vk_weight_get_unlocked(gate_src, gu_stride * n_expert) : NULL;
            up_bank = gate_bank ? q36_vk_weight_get_unlocked(up_src, gu_stride * n_expert) : NULL;
            down_bank = up_bank ? q36_vk_weight_get_unlocked(down_src, down_stride * n_expert) : NULL;
            gate_scales = gate_sc && down_bank ? q36_vk_weight_get_unlocked(gate_sc, scale_bytes) : NULL;
            up_scales = up_sc && down_bank ? q36_vk_weight_get_unlocked(up_sc, scale_bytes) : NULL;
            down_scales = down_sc && down_bank ? q36_vk_weight_get_unlocked(down_sc, scale_bytes) : NULL;
            ok = tables && down_bank;
        }
        ok = ok &&
             (!gate->has_scales || gate_scales) && (!up->has_scales || up_scales) && (!down->has_scales || down_scales);
        if (ok) {
            struct {
                uint32_t mid_dim;
                uint32_t blocks;
                uint32_t n_used;
                uint32_t has_gate_scales;
                uint32_t has_up_scales;
                uint32_t identity;
            } gpush = { mid_dim, in_dim / Q36_VK_QK_K, n_used,
                        gate->has_scales ? 1u : 0u, up->has_scales ? 1u : 0u,
                        identity ? 1u : 0u };
            const q36_gpu_tensor *gb[8] = {
                gate_bank, up_bank, x, identity ? (sel_slots ? sel_slots : selected) : wave_tiles,
                gate_scales ? gate_scales : gate_bank,
                up_scales ? up_scales : up_bank,
                mid8, tables,
            };
            ok = gemm ? q36_vk_run_unlocked("moe_iq2_gate_up_gemm", &q36_vk.moe_gate_up_gemm, gb, &gpush, sizeof(gpush),
                                            (mid_dim + 63u) / 64u, wave_tile_count, 1)
                      : q36_vk_run_unlocked("moe_iq2_gate_up", &q36_vk.moe_gate_up_f32b, gb, &gpush, sizeof(gpush),
                                            mid_dim, wave_tile_count, 1);
        }
        if (ok) {
            struct {
                uint32_t out_dim;
                uint32_t in_dim;
                uint32_t blocks;
                uint32_t n_used;
                uint32_t has_scales;
                uint32_t identity;
            } dpush = { out_dim, mid_dim, mid_dim / Q36_VK_QK_K, n_used, down->has_scales ? 1u : 0u,
                        identity ? 1u : 0u };
            const q36_gpu_tensor *schedule = identity ? (sel_slots ? sel_slots : selected) : wave_tiles;
            if (down_sum_decode) {
                const q36_gpu_tensor *db[6] = {
                    down_bank, mid8, schedule,
                    down_scales ? down_scales : down_bank,
                    weights, out,
                };
                ok = q36_vk_run_unlocked("moe_q2k_down_sum_decode",
                                         &q36_vk.moe_down_q2k_sum_decode,
                                         db, &dpush, sizeof(dpush),
                                         out_dim, 1, 1);
            } else {
                const q36_gpu_tensor *db[5] = {
                    down_bank, mid8, schedule,
                    down_scales ? down_scales : down_bank,
                    down8,
                };
                ok = gemm ? q36_vk_run_unlocked("moe_q2k_down_gemm", &q36_vk.moe_down_gemm, db, &dpush, sizeof(dpush),
                                                (out_dim + 63u) / 64u, wave_tile_count, 1)
                          : q36_vk_run_unlocked("moe_q2k_down", &q36_vk.moe_down_q2k_f32b, db, &dpush, sizeof(dpush),
                                                out_dim, wave_tile_count, 1);
            }
        }
        pthread_mutex_unlock(&q36_vk_mu);
        if (overflow) q36_gpu_tensor_free(wave_tiles);
        wave += wave_experts;
    } while (ok && overflow && wave < unique);

    if (ok && !down_sum_decode) {
        struct {
            uint32_t n;
            uint32_t n_used;
        } rpush = { out_dim, n_used };
        const q36_gpu_tensor *rb[3] = { down8, weights, out };
        pthread_mutex_lock(&q36_vk_mu);
        ok = q36_vk_run_unlocked("moe_reduce", &q36_vk.moe_reduce, rb, &rpush, sizeof(rpush),
                                 (out_dim + 255u) / 256u, n_tok, 1);
        pthread_mutex_unlock(&q36_vk_mu);
    }

done:
    q36_gpu_tensor_free(down8);
    q36_gpu_tensor_free(mid8);
    q36_gpu_tensor_free(tiles);
    q36_gpu_tensor_free(sel_slots);
    return ok;
}

int q36_gpu_moe_ffn_tensor(q36_gpu_tensor *out,
                           const void *model_map,
                           uint64_t model_size,
                           const q36_gpu_moe_weight *gate,
                           const q36_gpu_moe_weight *up,
                           const q36_gpu_moe_weight *down,
                           const q36_gpu_tensor *selected,
                           const q36_gpu_tensor *weights,
                           uint32_t layer,
                           uint32_t n_used,
                           const q36_gpu_tensor *x,
                           uint32_t n_tok,
                           uint32_t in_dim,
                           uint32_t mid_dim,
                           uint32_t out_dim,
                           uint32_t n_expert) {
    q36_gpu_tensor *qx = q36_vk_quantize_q8_k_alloc_tensor(x, in_dim, n_tok);
    if (!qx) return 0;
    int ok = q36_gpu_moe_ffn_q8_tensor(out, model_map, model_size,
                                       gate, up, down,
                                       selected, weights, layer, n_used,
                                       qx, n_tok, in_dim, mid_dim, out_dim, n_expert);
    q36_gpu_tensor_free(qx);
    return ok;
}
