#include "q36_ssd.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

static const uint64_t Q36_GIB = 1024ull * 1024ull * 1024ull;

/* Only the latest prediction matters. Replacing the pending batch cancels
 * stale guesses without waiting for already-issued kernel read-ahead. */
struct q36_ssd_prefetch {
    int fd;
    pthread_t thread;
    pthread_mutex_t mu;
    pthread_cond_t ready;
    q36_ssd_range ranges[12];
    uint64_t generation;
    uint32_t count;
    bool stop;
};

static void *q36_ssd_prefetch_worker(void *opaque) {
    q36_ssd_prefetch *p = opaque;
    for (;;) {
        pthread_mutex_lock(&p->mu);
        while (!p->count && !p->stop) pthread_cond_wait(&p->ready,&p->mu);
        if (p->stop) { pthread_mutex_unlock(&p->mu); break; }
        q36_ssd_range ranges[12];
        uint32_t count = p->count;
        uint64_t generation = p->generation;
        memcpy(ranges,p->ranges,count*sizeof(*ranges));
        p->count = 0;
        pthread_mutex_unlock(&p->mu);
        for (uint32_t i = 0; i < count; i++) {
            pthread_mutex_lock(&p->mu);
            bool current = !p->stop && generation == p->generation;
            pthread_mutex_unlock(&p->mu);
            if (!current) break;
#ifdef POSIX_FADV_WILLNEED
            (void)posix_fadvise(p->fd,ranges[i].offset,ranges[i].bytes,POSIX_FADV_WILLNEED);
#endif
        }
    }
    return NULL;
}

q36_ssd_prefetch *q36_ssd_prefetch_create(int fd) {
    q36_ssd_prefetch *p = calloc(1,sizeof(*p));
    if (!p) return NULL;
    p->fd = fcntl(fd,F_DUPFD_CLOEXEC,0);
    if (p->fd < 0) { free(p); return NULL; }
    if (pthread_mutex_init(&p->mu,NULL)) { close(p->fd); free(p); return NULL; }
    if (pthread_cond_init(&p->ready,NULL)) {
        pthread_mutex_destroy(&p->mu); close(p->fd); free(p); return NULL;
    }
    if (pthread_create(&p->thread,NULL,q36_ssd_prefetch_worker,p)) {
        pthread_cond_destroy(&p->ready); pthread_mutex_destroy(&p->mu);
        close(p->fd); free(p); return NULL;
    }
    return p;
}

uint64_t q36_ssd_prefetch_submit(q36_ssd_prefetch *p, const q36_ssd_range *ranges,
                                uint32_t count) {
    if (!p || count > 12 || (count && !ranges)) return 0;
    uint64_t bytes = 0;
    for (uint32_t i = 0; i < count; i++) bytes += ranges[i].bytes;
    pthread_mutex_lock(&p->mu);
    if (count) memcpy(p->ranges,ranges,count*sizeof(*ranges));
    p->generation++;
    p->count = count;
    if (count) pthread_cond_signal(&p->ready);
    pthread_mutex_unlock(&p->mu);
    return bytes;
}

void q36_ssd_prefetch_free(q36_ssd_prefetch *p) {
    if (!p) return;
    pthread_mutex_lock(&p->mu);
    p->stop = true;
    pthread_cond_signal(&p->ready);
    pthread_mutex_unlock(&p->mu);
    pthread_join(p->thread,NULL);
    pthread_cond_destroy(&p->ready);
    pthread_mutex_destroy(&p->mu);
    close(p->fd);
    free(p);
}

bool q36_parse_gib_arg(const char *s, uint64_t *bytes) {
    if (bytes) *bytes = 0;
    if (!s || !s[0] || !bytes) return false;

    size_t len = strlen(s);
    if (len > 2 &&
        (s[len - 2] == 'g' || s[len - 2] == 'G') &&
        (s[len - 1] == 'b' || s[len - 1] == 'B')) {
        len -= 2;
    }
    if (len == 0) return false;
    for (size_t i = 0; i < len; i++) {
        if (!isdigit((unsigned char)s[i])) return false;
    }

    char numbuf[32];
    if (len >= sizeof(numbuf)) return false;
    memcpy(numbuf, s, len);
    numbuf[len] = '\0';

    errno = 0;
    unsigned long long v = strtoull(numbuf, NULL, 10);
    if (errno != 0 || v == 0 || v > UINT64_MAX / Q36_GIB) return false;

    *bytes = (uint64_t)v * Q36_GIB;
    return true;
}

bool q36_parse_streaming_cache_experts_arg(const char *s,
                                           uint32_t   *experts,
                                           uint64_t   *bytes) {
    if (experts) *experts = 0;
    if (bytes) *bytes = 0;
    if (!s || !s[0] || !experts || !bytes) return false;

    const size_t len = strlen(s);
    if (len > 2 &&
        (s[len - 2] == 'g' || s[len - 2] == 'G') &&
        (s[len - 1] == 'b' || s[len - 1] == 'B')) {
        return q36_parse_gib_arg(s, bytes);
    }

    for (size_t i = 0; i < len; i++) {
        if (!isdigit((unsigned char)s[i])) return false;
    }

    errno = 0;
    unsigned long v = strtoul(s, NULL, 10);
    if (errno != 0 || v == 0 || v > UINT32_MAX) return false;

    *experts = (uint32_t)v;
    return true;
}

uint32_t q36_ssd_cache_experts_for_byte_budget(uint64_t bytes,
                                               uint64_t per_expert_bytes) {
    if (bytes == 0 || per_expert_bytes == 0) return 0;
    const uint64_t experts = bytes / per_expert_bytes;
    if (experts == 0 || experts > UINT32_MAX) return 0;
    return (uint32_t)experts;
}

uint32_t q36_ssd_shrink_cache_experts(uint32_t experts) {
    if (experts == 0) return 0;
    const uint32_t release = (experts + 9u) / 10u;
    return experts > release ? experts - release : 0;
}

bool q36_ssd_auto_cache_plan(uint64_t            recommended_bytes,
                             uint32_t            percent,
                             uint64_t            model_limit_bytes,
                             uint64_t            non_routed_bytes,
                             uint64_t            per_expert_bytes,
                             uint64_t            max_model_experts,
                             q36_ssd_cache_plan *out) {
    if (!out) return false;
    memset(out, 0, sizeof(*out));
    if (recommended_bytes == 0 || per_expert_bytes == 0 || percent < 50 || percent > 95) return false;

    out->model_target_bytes = (recommended_bytes / 100) * percent +
                              (recommended_bytes % 100) * percent / 100;
    if (model_limit_bytes && out->model_target_bytes > model_limit_bytes)
        out->model_target_bytes = model_limit_bytes;
    if (out->model_target_bytes > non_routed_bytes) {
        out->cache_bytes = out->model_target_bytes - non_routed_bytes;
    }

    uint64_t cache_experts = out->cache_bytes / per_expert_bytes;
    if (cache_experts == 0) cache_experts = 1;
    if (max_model_experts != 0 && cache_experts > max_model_experts) {
        cache_experts = max_model_experts;
    }
    if (cache_experts > UINT32_MAX) cache_experts = UINT32_MAX;

    out->cache_experts = (uint32_t)cache_experts;
    out->effective_cache_bytes = cache_experts * per_expert_bytes;
    return out->cache_experts != 0;
}

bool q36_ssd_memory_lock_acquire(q36_ssd_memory_lock *lock,
                                 uint64_t             bytes) {
    if (!lock) return false;
    lock->ptr = NULL;
    lock->bytes = 0;
    if (bytes == 0) return true;
    if (bytes > (uint64_t)SIZE_MAX) {
        fprintf(stderr,
                "q36: --simulate-used-memory is too large for this process\n");
        return false;
    }

    void *ptr = mmap(NULL,
                     (size_t)bytes,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS,
                     -1,
                     0);
    if (ptr == MAP_FAILED) {
        fprintf(stderr,
                "q36: --simulate-used-memory mmap %.2f GiB failed: %s\n",
                (double)bytes / (double)Q36_GIB,
                strerror(errno));
        return false;
    }

    const long page_long = sysconf(_SC_PAGESIZE);
    const uint64_t page = page_long > 0 ? (uint64_t)page_long : 4096ull;
    const uint64_t chunk_bytes = 256ull * 1024ull * 1024ull;
    volatile unsigned char *p = (volatile unsigned char *)ptr;

    /*
     * Touch and lock in bounded chunks.  A single very large mlock() is harder
     * to diagnose when it fails and can create long uninterruptible VM work on
     * macOS; chunking mirrors the standalone diagnostic utility.
     */
    uint64_t locked = 0;
    for (uint64_t off = 0; off < bytes; off += chunk_bytes) {
        uint64_t len = bytes - off;
        if (len > chunk_bytes) len = chunk_bytes;

        for (uint64_t pos = off; pos < off + len; pos += page) {
            p[pos] = (unsigned char)(pos / page);
        }
        if (len != 0) p[off + len - 1u] = 1;

        if (mlock((void *)(p + off), (size_t)len) != 0) {
            fprintf(stderr,
                    "q36: --simulate-used-memory mlock failed after %.2f/%.2f GiB: %s\n",
                    (double)locked / (double)Q36_GIB,
                    (double)bytes / (double)Q36_GIB,
                    strerror(errno));
            if (locked != 0) munlock(ptr, (size_t)locked);
            munmap(ptr, (size_t)bytes);
            return false;
        }
        locked += len;
    }

    lock->ptr = ptr;
    lock->bytes = bytes;
    fprintf(stderr,
            "q36: simulated used memory: locked %.2f GiB before model load\n",
            (double)bytes / (double)Q36_GIB);
    return true;
}

void q36_ssd_memory_lock_release(q36_ssd_memory_lock *lock) {
    if (!lock || !lock->ptr || lock->bytes == 0) return;
    munlock(lock->ptr, (size_t)lock->bytes);
    munmap(lock->ptr, (size_t)lock->bytes);
    lock->ptr = NULL;
    lock->bytes = 0;
}
