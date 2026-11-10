#ifndef Q36_SSD_H
#define Q36_SSD_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    void *ptr;
    uint64_t bytes;
} q36_ssd_memory_lock;

typedef struct {
    uint64_t model_target_bytes;
    uint64_t cache_bytes;
    uint64_t effective_cache_bytes;
    uint32_t cache_experts;
} q36_ssd_cache_plan;

typedef struct q36_ssd_prefetch q36_ssd_prefetch;
typedef struct {
    uint64_t offset, bytes;
} q36_ssd_range;

q36_ssd_prefetch *q36_ssd_prefetch_create(int fd);
uint64_t q36_ssd_prefetch_submit(q36_ssd_prefetch *p, const q36_ssd_range *ranges,
                                uint32_t count);
void q36_ssd_prefetch_free(q36_ssd_prefetch *p);

bool q36_parse_gib_arg(const char *s, uint64_t *bytes);
bool q36_parse_streaming_cache_experts_arg(const char *s,
                                           uint32_t   *experts,
                                           uint64_t   *bytes);

uint32_t q36_ssd_cache_experts_for_byte_budget(uint64_t bytes,
                                               uint64_t per_expert_bytes);
uint32_t q36_ssd_shrink_cache_experts(uint32_t experts);
bool q36_ssd_auto_cache_plan(uint64_t            recommended_bytes,
                             uint32_t            percent,
                             uint64_t            model_limit_bytes,
                             uint64_t            non_routed_bytes,
                             uint64_t            per_expert_bytes,
                             uint64_t            max_model_experts,
                             q36_ssd_cache_plan *out);

bool q36_ssd_memory_lock_acquire(q36_ssd_memory_lock *lock,
                                 uint64_t             bytes);
void q36_ssd_memory_lock_release(q36_ssd_memory_lock *lock);

#endif
