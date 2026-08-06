#define _POSIX_C_SOURCE 200809L

#include "quants.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define Q36_GGUF_ALIGNMENT 32
#define Q36_DEFAULT_IN "gguf/Qwen3.6-35B-A3B-Q8_0.gguf"
#define Q36_KV_IMATRIX_FILE "quantize.imatrix.file"
#define Q36_KV_IMATRIX_DATASET "quantize.imatrix.dataset"
#define Q36_KV_IMATRIX_ENTRIES "quantize.imatrix.entries_count"
#define Q36_KV_IMATRIX_CHUNKS "quantize.imatrix.chunks_count"

typedef enum {
    GGUF_TYPE_UINT8 = 0,
    GGUF_TYPE_INT8 = 1,
    GGUF_TYPE_UINT16 = 2,
    GGUF_TYPE_INT16 = 3,
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_INT32 = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_TYPE_UINT64 = 10,
    GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_type;

typedef struct {
    char *key;
    int value;
} hslot;

typedef struct {
    hslot *slots;
    int cap;
} hmap;

typedef struct {
    uint8_t *data;
    size_t size;
} bytes;

typedef struct {
    char *name;
    int n_dims;
    int64_t ne[Q36Q_MAX_DIMS];
    q36q_type type;
    uint64_t old_offset;
    uint64_t new_offset;
    size_t size;
    const uint8_t *data;
} tensor;

typedef struct gguf {
    char *path;
    int fd;
    uint8_t *map;
    size_t size;
    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    bytes kv;
    size_t alignment;
    size_t data_offset;
    tensor *tensors;
    hmap tmap;
    int split_no;
    int split_count;
    int64_t split_tensors;
    struct gguf *shards;
    int n_shards;
} gguf;

typedef struct {
    char *name;
    float *v;
    int n;
} im_entry;

typedef struct {
    char *file;
    char *dataset;
    im_entry *entries;
    int n_entries;
    int chunks;
    bool strict;
} imatrix;

typedef enum {
    DEC_KEEP,
    DEC_GATE_UP,
    DEC_DOWN,
    DEC_Q4,
} decision;

typedef struct {
    tensor *tensors;
    uint64_t n_tensors;
    uint64_t extra_kv;
    size_t meta_size;
    size_t data_offset;
    size_t tensor_bytes;
    size_t alignment;
    size_t n_gate_up;
    size_t n_down;
    size_t n_q4;
    size_t n_keep;
} plan;

typedef struct {
    char **inputs;
    int n_inputs;
    char *out;
    char *imatrix;
    char *audit;
    bool dry_run;
    bool force;
    bool strict_imatrix;
    bool synthetic_imatrix;
    int threads;
    int q4_start;
    int q4_end;
    int q4_last;
} params;

static void die(const char *msg) {
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

static void die_errno(const char *what, const char *path) {
    fprintf(stderr, "error: %s %s: %s\n", what, path ? path : "", strerror(errno));
    exit(1);
}

static void die_pthread(int err, const char *what) {
    fprintf(stderr, "error: %s: %s\n", what, strerror(err));
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    if (sz && n > SIZE_MAX / sz) die("allocation size overflow");
    void *p = calloc(n ? n : 1, sz ? sz : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) die("out of memory");
    return q;
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static size_t size_add(size_t a, size_t b) {
    if (a > SIZE_MAX - b) die("size overflow");
    return a + b;
}

static size_t size_mul(size_t a, size_t b) {
    if (a && b > SIZE_MAX / a) die("size overflow");
    return a * b;
}

static uint64_t fnv1a(const char *s) {
    uint64_t h = 1469598103934665603ull;
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= 1099511628211ull;
    }
    return h;
}

static void hmap_build(hmap *m, char **keys, int n) {
    if (n < 0 || n > INT_MAX / 2) die("too many tensor names");
    int cap = 1;
    while (cap < n * 2) cap <<= 1;
    m->cap = cap;
    m->slots = xcalloc((size_t)cap, sizeof(m->slots[0]));
    for (int i = 0; i < n; i++) {
        int p = (int)(fnv1a(keys[i]) & (uint64_t)(cap - 1));
        while (m->slots[p].key) p = (p + 1) & (cap - 1);
        m->slots[p].key = keys[i];
        m->slots[p].value = i;
    }
}

static int hmap_get(const hmap *m, const char *key) {
    if (!m->slots) return -1;
    int p = (int)(fnv1a(key) & (uint64_t)(m->cap - 1));
    while (m->slots[p].key) {
        if (!strcmp(m->slots[p].key, key)) return m->slots[p].value;
        p = (p + 1) & (m->cap - 1);
    }
    return -1;
}

static void hmap_free(hmap *m) {
    free(m->slots);
    memset(m, 0, sizeof(*m));
}

static bool starts(const char *s, const char *p) {
    return strncmp(s, p, strlen(p)) == 0;
}

static bool ends(const char *s, const char *p) {
    size_t sl = strlen(s), pl = strlen(p);
    return sl >= pl && memcmp(s + sl - pl, p, pl) == 0;
}

static size_t pad(size_t x, size_t n) {
    if (!n) die("zero GGUF alignment");
    size_t rem = x % n;
    return rem ? size_add(x, n - rem) : x;
}

static uint16_t get_u16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t get_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t get_u64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i);
    return v;
}

static void put_u32(FILE *fp, uint32_t v) {
    if (fwrite(&v, 4, 1, fp) != 1) die("write u32 failed");
}

static void put_u64(FILE *fp, uint64_t v) {
    if (fwrite(&v, 8, 1, fp) != 1) die("write u64 failed");
}

static void put_str(FILE *fp, const char *s) {
    uint64_t n = strlen(s);
    put_u64(fp, n);
    if (n && fwrite(s, 1, (size_t)n, fp) != (size_t)n) die("write string failed");
}

static uint32_t read_u32(FILE *fp, const char *what) {
    uint8_t b[4];
    if (fread(b, 1, 4, fp) != 4) {
        fprintf(stderr, "error: short read for %s\n", what);
        exit(1);
    }
    return get_u32(b);
}

static int32_t read_i32(FILE *fp, const char *what) {
    return (int32_t)read_u32(fp, what);
}

static size_t file_remaining(FILE *fp) {
    off_t pos = ftello(fp);
    if (pos < 0 || fseeko(fp, 0, SEEK_END) != 0) die("cannot inspect input size");
    off_t end = ftello(fp);
    if (fseeko(fp, pos, SEEK_SET) != 0 || end < pos) die("cannot inspect input size");
    if ((uintmax_t)(end - pos) > SIZE_MAX) die("input file is too large");
    return (size_t)(end - pos);
}

static size_t scalar_size(uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: return 8;
    }
    return 0;
}

static uint64_t pull_u64(const uint8_t *base, size_t size, size_t *p) {
    if (*p > size || size - *p < 8) die("bad GGUF: short u64");
    uint64_t v = get_u64(base + *p);
    *p += 8;
    return v;
}

static uint16_t pull_u16(const uint8_t *base, size_t size, size_t *p) {
    if (*p > size || size - *p < 2) die("bad GGUF: short u16");
    uint16_t v = get_u16(base + *p);
    *p += 2;
    return v;
}

static uint32_t pull_u32(const uint8_t *base, size_t size, size_t *p) {
    if (*p > size || size - *p < 4) die("bad GGUF: short u32");
    uint32_t v = get_u32(base + *p);
    *p += 4;
    return v;
}

static char *pull_str(const uint8_t *base, size_t size, size_t *p) {
    uint64_t n = pull_u64(base, size, p);
    if (n > SIZE_MAX - 1 || *p > size || n > size - *p)
        die("bad GGUF: short string");
    char *s = xmalloc((size_t)n + 1);
    memcpy(s, base + *p, (size_t)n);
    s[n] = 0;
    *p += (size_t)n;
    return s;
}

static void skip_value(const uint8_t *base, size_t size, size_t *p, uint32_t type) {
    if (type == GGUF_TYPE_STRING) {
        uint64_t n = pull_u64(base, size, p);
        if (*p > size || n > size - *p) die("bad GGUF: short string value");
        *p += (size_t)n;
        return;
    }
    if (type == GGUF_TYPE_ARRAY) {
        uint32_t et = pull_u32(base, size, p);
        uint64_t n = pull_u64(base, size, p);
        if (et == GGUF_TYPE_STRING) {
            for (uint64_t i = 0; i < n; i++) {
                uint64_t len = pull_u64(base, size, p);
                if (*p > size || len > size - *p) die("bad GGUF: short string array");
                *p += (size_t)len;
            }
            return;
        }
        size_t sz = scalar_size(et);
        if (!sz) die("bad GGUF: unsupported array type");
        if (n > SIZE_MAX / sz || *p > size || (size_t)n * sz > size - *p)
            die("bad GGUF: short array");
        *p += (size_t)n * sz;
        return;
    }
    size_t sz = scalar_size(type);
    if (!sz) die("bad GGUF: unsupported value type");
    if (*p > size || sz > size - *p) die("bad GGUF: short scalar");
    *p += sz;
}

static size_t str_size(const char *s) {
    return size_add(8, strlen(s));
}

static bool imatrix_kv(const char *key) {
    return starts(key, "quantize.imatrix.");
}

static bool split_kv(const char *key) {
    return starts(key, "split.");
}

static size_t tensor_elems(const tensor *t) {
    size_t n = 1;
    for (int i = 0; i < t->n_dims; i++) {
        if (t->ne[i] <= 0 || (uint64_t)t->ne[i] > SIZE_MAX) die("bad tensor shape");
        n = size_mul(n, (size_t)t->ne[i]);
    }
    return n;
}

static size_t tensor_size(q36q_type type, const int64_t *ne, int n_dims) {
    if (n_dims < 1 || ne[0] <= 0) return 0;
    size_t row = q36q_row_size(type, ne[0]);
    for (int i = 1; i < n_dims; i++) {
        if (ne[i] <= 0 || (uint64_t)ne[i] > SIZE_MAX) return 0;
        row = size_mul(row, (size_t)ne[i]);
    }
    return row;
}

static int tensor_rows(const tensor *t) {
    size_t rows = tensor_elems(t) / (size_t)t->ne[0];
    if (rows > INT_MAX) die("tensor has too many rows");
    return (int)rows;
}

static int64_t tensor_nelems_i64(const tensor *t) {
    int64_t n = 1;
    for (int i = 0; i < t->n_dims; i++) {
        if (t->ne[i] <= 0 || n > INT64_MAX / t->ne[i]) die("tensor element count overflow");
        n *= t->ne[i];
    }
    return n;
}

static gguf gguf_open(const char *path) {
    gguf g = { .fd = -1, .split_no = -1, .split_count = -1, .split_tensors = -1 };
    g.path = xstrdup(path);
    g.fd = open(path, O_RDONLY);
    if (g.fd < 0) die_errno("open", path);
    struct stat st;
    if (fstat(g.fd, &st) != 0) die_errno("stat", path);
    if (st.st_size <= 0) die("empty GGUF");
    if ((uintmax_t)st.st_size > SIZE_MAX) die("GGUF is too large to map");
    g.size = (size_t)st.st_size;
    g.map = mmap(NULL, g.size, PROT_READ, MAP_PRIVATE, g.fd, 0);
    if (g.map == MAP_FAILED) die_errno("mmap", path);
    size_t p = 0;
    if (g.size < 24 || memcmp(g.map, "GGUF", 4) != 0) die("bad GGUF magic");
    p += 4;
    g.version = pull_u32(g.map, g.size, &p);
    g.n_tensors = pull_u64(g.map, g.size, &p);
    g.n_kv = pull_u64(g.map, g.size, &p);
    g.alignment = Q36_GGUF_ALIGNMENT;
    if (g.n_kv > (g.size - p) / 12 || g.n_kv > (SIZE_MAX - 1) / 2)
        die("bad GGUF: metadata count exceeds file size");

    size_t kv_start = p;
    size_t keep_cap = size_add(size_mul((size_t)g.n_kv, 2), 1);
    size_t *spans = xcalloc(keep_cap, sizeof(spans[0]));
    size_t keep_n = 0;
    for (uint64_t i = 0; i < g.n_kv; i++) {
        size_t rec_start = p;
        char *key = pull_str(g.map, g.size, &p);
        uint32_t type = pull_u32(g.map, g.size, &p);
        if (!strcmp(key, "general.alignment") && type == GGUF_TYPE_UINT32) {
            g.alignment = pull_u32(g.map, g.size, &p);
        } else if (!strcmp(key, "split.no") && type == GGUF_TYPE_UINT16) {
            g.split_no = pull_u16(g.map, g.size, &p);
        } else if (!strcmp(key, "split.count") && type == GGUF_TYPE_UINT16) {
            g.split_count = pull_u16(g.map, g.size, &p);
        } else if (!strcmp(key, "split.tensors.count") && type == GGUF_TYPE_INT32) {
            g.split_tensors = (int32_t)pull_u32(g.map, g.size, &p);
        } else {
            skip_value(g.map, g.size, &p, type);
        }
        if (!imatrix_kv(key) && !split_kv(key)) {
            spans[keep_n++] = rec_start - kv_start;
            spans[keep_n++] = p - kv_start;
        }
        free(key);
    }

    size_t kv_len = p - kv_start;
    for (size_t i = 0; i < keep_n; i += 2)
        g.kv.size = size_add(g.kv.size, spans[i + 1] - spans[i]);
    g.kv.data = xmalloc(g.kv.size);
    size_t q = 0;
    for (size_t i = 0; i < keep_n; i += 2) {
        size_t n = spans[i + 1] - spans[i];
        memcpy(g.kv.data + q, g.map + kv_start + spans[i], n);
        q = size_add(q, n);
    }
    g.n_kv = keep_n / 2;
    free(spans);
    (void)kv_len;

    if (g.alignment == 0) die("bad GGUF: zero alignment");
    if (g.n_tensors > (g.size - p) / 32 || g.n_tensors > INT_MAX ||
        g.n_tensors > SIZE_MAX / sizeof(g.tensors[0]))
        die("bad GGUF: tensor count exceeds file size");
    g.tensors = xcalloc((size_t)g.n_tensors, sizeof(g.tensors[0]));
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        tensor *t = &g.tensors[i];
        t->name = pull_str(g.map, g.size, &p);
        t->n_dims = (int)pull_u32(g.map, g.size, &p);
        if (t->n_dims < 1 || t->n_dims > Q36Q_MAX_DIMS) die("bad tensor rank");
        for (int j = 0; j < t->n_dims; j++) {
            uint64_t dim = pull_u64(g.map, g.size, &p);
            if (dim == 0 || dim > INT64_MAX) die("bad tensor shape");
            t->ne[j] = (int64_t)dim;
        }
        t->type = (q36q_type)pull_u32(g.map, g.size, &p);
        t->old_offset = pull_u64(g.map, g.size, &p);
        t->size = tensor_size(t->type, t->ne, t->n_dims);
        if (!t->size) {
            fprintf(stderr, "error: unsupported tensor type or shape: %s type=%u\n",
                    t->name, (unsigned)t->type);
            exit(1);
        }
    }
    g.data_offset = pad(p, g.alignment);
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        tensor *t = &g.tensors[i];
        if (t->old_offset > SIZE_MAX - g.data_offset ||
            t->size > SIZE_MAX - (g.data_offset + (size_t)t->old_offset) ||
            g.data_offset + (size_t)t->old_offset + t->size > g.size) {
            fprintf(stderr, "error: tensor data outside file: %s\n", t->name);
            exit(1);
        }
        t->data = g.map + g.data_offset + t->old_offset;
    }
    char **keys = xmalloc(size_mul((size_t)g.n_tensors, sizeof(keys[0])));
    for (uint64_t i = 0; i < g.n_tensors; i++) keys[i] = g.tensors[i].name;
    hmap_build(&g.tmap, keys, (int)g.n_tensors);
    free(keys);
    return g;
}

static gguf gguf_open_many(char **paths, int n) {
    if (n == 1) {
        gguf g = gguf_open(paths[0]);
        if (g.split_count > 1) die("split GGUF input requires one --in per shard");
        return g;
    }
    gguf g = { .fd = -1, .n_shards = n };
    g.path = xstrdup(paths[0]);
    g.shards = xcalloc((size_t)n, sizeof(g.shards[0]));
    for (int i = 0; i < n; i++) g.shards[i] = gguf_open(paths[i]);

    gguf *first = &g.shards[0];
    g.version = first->version;
    g.alignment = first->alignment;
    g.n_kv = first->n_kv;
    g.kv.size = first->kv.size;
    g.kv.data = xmalloc(g.kv.size);
    memcpy(g.kv.data, first->kv.data, g.kv.size);
    for (int i = 0; i < n; i++) {
        gguf *s = &g.shards[i];
        if (s->version != g.version || s->alignment != g.alignment) {
            die("GGUF shards have incompatible version or alignment");
        }
        if (s->split_no != i || s->split_count != n || s->split_tensors < 0) {
            die("GGUF shards are missing, duplicated, or out of order");
        }
        if (g.n_tensors > UINT64_MAX - s->n_tensors) die("GGUF shard tensor count overflow");
        g.n_tensors += s->n_tensors;
    }
    if ((uint64_t)first->split_tensors != g.n_tensors) die("GGUF shard tensor count mismatch");
    if (g.n_tensors > INT_MAX || g.n_tensors > SIZE_MAX / sizeof(g.tensors[0]))
        die("too many tensors across GGUF shards");

    g.tensors = xcalloc((size_t)g.n_tensors, sizeof(g.tensors[0]));
    uint64_t k = 0;
    for (int i = 0; i < n; i++) {
        gguf *s = &g.shards[i];
        for (uint64_t j = 0; j < s->n_tensors; j++) {
            tensor *dst = &g.tensors[k++];
            *dst = s->tensors[j];
            dst->name = xstrdup(s->tensors[j].name);
        }
    }
    char **keys = xmalloc(size_mul((size_t)g.n_tensors, sizeof(keys[0])));
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        keys[i] = g.tensors[i].name;
        for (uint64_t j = 0; j < i; j++) {
            if (!strcmp(keys[i], keys[j])) {
                fprintf(stderr, "error: duplicate tensor across shards: %s\n", keys[i]);
                exit(1);
            }
        }
    }
    hmap_build(&g.tmap, keys, (int)g.n_tensors);
    free(keys);
    return g;
}

static void gguf_close(gguf *g) {
    for (int i = 0; i < g->n_shards; i++) gguf_close(&g->shards[i]);
    free(g->shards);
    if (g->map && g->map != MAP_FAILED) munmap(g->map, g->size);
    if (g->fd >= 0) close(g->fd);
    for (uint64_t i = 0; i < g->n_tensors; i++) free(g->tensors[i].name);
    free(g->tensors);
    free(g->kv.data);
    free(g->path);
    hmap_free(&g->tmap);
    memset(g, 0, sizeof(*g));
}

static void imatrix_load_gguf(imatrix *im, const char *path) {
    gguf g = gguf_open(path);
    int cap = 128;
    im->entries = xcalloc((size_t)cap, sizeof(im->entries[0]));
    im->n_entries = 0;
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        tensor *sum = &g.tensors[i];
        const char *suffix = ".in_sum2";
        if (!ends(sum->name, suffix)) continue;
        if (sum->type != Q36Q_TYPE_F32) die("GGUF imatrix in_sum2 tensor is not f32");

        size_t base_len = strlen(sum->name) - strlen(suffix);
        char *base = xmalloc(base_len + 1);
        memcpy(base, sum->name, base_len);
        base[base_len] = 0;

        char *count_name = xmalloc(base_len + strlen(".counts") + 1);
        memcpy(count_name, base, base_len);
        memcpy(count_name + base_len, ".counts", strlen(".counts") + 1);
        int cidx = hmap_get(&g.tmap, count_name);
        free(count_name);
        if (cidx < 0) {
            fprintf(stderr, "error: missing GGUF imatrix counts for %s\n", base);
            exit(1);
        }
        tensor *counts = &g.tensors[cidx];
        if (counts->type != Q36Q_TYPE_F32) die("GGUF imatrix counts tensor is not f32");

        int64_t nval = tensor_nelems_i64(sum);
        int64_t ncounts = tensor_nelems_i64(counts);
        if (nval < 1 || ncounts < 1 || nval % ncounts) {
            fprintf(stderr, "error: bad GGUF imatrix shape for %s\n", base);
            exit(1);
        }
        if (nval > INT32_MAX) die("GGUF imatrix entry too large");
        if (im->n_entries == cap) {
            if (cap > INT_MAX / 2) die("too many imatrix entries");
            cap *= 2;
            im->entries = xrealloc(im->entries,
                                   size_mul((size_t)cap, sizeof(im->entries[0])));
        }

        float *v = xmalloc(size_mul((size_t)nval, sizeof(float)));
        const float *sdata = (const float *)sum->data;
        const float *cdata = (const float *)counts->data;
        int64_t ne0 = nval / ncounts;
        for (int64_t j = 0; j < ncounts; j++) {
            float count = cdata[j];
            if (count > 0.0f) {
                for (int64_t k = 0; k < ne0; k++) {
                    float x = sdata[j * ne0 + k] / count;
                    if (!isfinite(x)) die("non-finite GGUF imatrix value");
                    v[j * ne0 + k] = x;
                }
            } else {
                for (int64_t k = 0; k < ne0; k++) v[j * ne0 + k] = 1.0f;
            }
        }

        im->entries[im->n_entries++] = (im_entry){ base, v, (int)nval };
    }
    if (im->n_entries < 1) die("GGUF imatrix has no entries");
    gguf_close(&g);
}

static void imatrix_load(imatrix *im, const char *path, bool strict) {
    memset(im, 0, sizeof(*im));
    im->file = xstrdup(path);
    im->strict = strict;
    im->chunks = -1;
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open imatrix", path);
    uint8_t magic[4];
    if (fread(magic, 1, sizeof(magic), fp) != sizeof(magic)) die("short imatrix read");
    if (!memcmp(magic, "GGUF", 4)) {
        fclose(fp);
        imatrix_load_gguf(im, path);
        return;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) die_errno("seek imatrix", path);
    int32_t n = read_i32(fp, "imatrix entry count");
    if (n < 1) die("imatrix has no entries");
    if ((size_t)n > file_remaining(fp) / 17) die("imatrix entry count exceeds file size");
    im->entries = xcalloc((size_t)n, sizeof(im->entries[0]));
    im->n_entries = n;
    for (int i = 0; i < n; i++) {
        int32_t len = read_i32(fp, "imatrix name length");
        if (len < 1 || len > 4096) die("bad imatrix name length");
        char *name = xmalloc((size_t)len + 1);
        if (fread(name, 1, (size_t)len, fp) != (size_t)len) die("short imatrix name read");
        name[len] = 0;
        int32_t ncall = read_i32(fp, "imatrix calls");
        int32_t nval = read_i32(fp, "imatrix values");
        if (nval < 1) die("bad imatrix value count");
        if ((size_t)nval > file_remaining(fp) / sizeof(float))
            die("imatrix value count exceeds file size");
        float *v = xmalloc(size_mul((size_t)nval, sizeof(float)));
        if (fread(v, sizeof(float), (size_t)nval, fp) != (size_t)nval) die("short imatrix value read");
        if (ncall > 0) {
            for (int j = 0; j < nval; j++) v[j] /= (float)ncall;
        }
        for (int j = 0; j < nval; j++) {
            if (!isfinite(v[j])) die("non-finite imatrix value");
        }
        im->entries[i] = (im_entry){ name, v, nval };
    }
    if (fgetc(fp) != EOF && fseeko(fp, -1, SEEK_CUR) == 0) {
        im->chunks = read_i32(fp, "imatrix chunks");
        int32_t len = read_i32(fp, "imatrix dataset length");
        if (len > 0 && len < (1 << 20) && (size_t)len <= file_remaining(fp)) {
            im->dataset = xmalloc((size_t)len + 1);
            if (fread(im->dataset, 1, (size_t)len, fp) == (size_t)len) im->dataset[len] = 0;
            else {
                free(im->dataset);
                im->dataset = NULL;
            }
        }
    }
    fclose(fp);
}

static void imatrix_free(imatrix *im) {
    for (int i = 0; i < im->n_entries; i++) {
        free(im->entries[i].name);
        free(im->entries[i].v);
    }
    free(im->entries);
    free(im->file);
    free(im->dataset);
}

static const im_entry *imatrix_find(const imatrix *im, const char *name) {
    if (!im || !im->n_entries) return NULL;
    for (int i = 0; i < im->n_entries; i++) {
        im_entry *e = &im->entries[i];
        if (strcmp(e->name, name)) continue;
        return e;
    }
    if (im->strict) {
        fprintf(stderr, "error: missing imatrix entry for %s\n", name);
        exit(1);
    }
    return NULL;
}

static const float *imatrix_row(const imatrix *im, const tensor *t, int64_t row, int64_t ncols) {
    const im_entry *e = imatrix_find(im, t->name);
    if (!e) return NULL;
    if ((int64_t)e->n == ncols) return e->v;
    if (t->n_dims == 3 && t->ne[1] > 0 && t->ne[2] > 0 &&
        (int64_t)e->n == ncols * t->ne[2]) {
        int64_t expert = row / t->ne[1];
        if (expert >= 0 && expert < t->ne[2]) return e->v + (size_t)expert * (size_t)ncols;
    }
    fprintf(stderr, "error: imatrix size mismatch for %s: got %d expected %" PRId64,
            t->name, e->n, ncols);
    if (t->n_dims == 3 && t->ne[2] > 0) fprintf(stderr, " or %" PRId64, ncols * t->ne[2]);
    fprintf(stderr, "\n");
    exit(1);
}

static bool parse_routed(const char *name, int *layerp, char *kind, size_t klen) {
    int layer = -1, pos = 0;
    char part[32];
    if (sscanf(name, "blk.%d.ffn_%31[^_]_exps.weight%n", &layer, part, &pos) != 2) return false;
    if (pos != (int)strlen(name)) return false;
    if (strcmp(part, "gate") && strcmp(part, "up") && strcmp(part, "down") && strcmp(part, "out")) return false;
    if (layerp) *layerp = layer;
    snprintf(kind, klen, "%s", part);
    return true;
}

static const char *layer_suffix(const char *name) {
    if (!starts(name, "blk.")) return NULL;
    const char *p = strchr(name + 4, '.');
    return p ? p + 1 : NULL;
}

static const char *keep_reason(const char *name) {
    static const char *top[] = {
        "output.weight", "output_norm.weight", "token_embd.weight",
    };
    static const char *layer[] = {
        "attn_gate.weight", "attn_k.weight", "attn_k_norm.weight",
        "attn_norm.weight", "attn_output.weight", "attn_q.weight",
        "attn_q_norm.weight", "attn_qkv.weight", "attn_v.weight",
        "ffn_down_shexp.weight", "ffn_gate_inp.weight",
        "ffn_gate_inp_shexp.weight", "ffn_gate_shexp.weight",
        "ffn_up_shexp.weight", "post_attention_norm.weight", "ssm_a",
        "ssm_alpha.weight", "ssm_beta.weight", "ssm_conv1d.weight",
        "ssm_dt.bias", "ssm_norm.weight", "ssm_out.weight",
    };
    for (size_t i = 0; i < sizeof(top) / sizeof(top[0]); i++) {
        if (!strcmp(name, top[i])) return "top-level Q8/output/embed/norm keep-list";
    }
    const char *s = layer_suffix(name);
    if (!s) return NULL;
    for (size_t i = 0; i < sizeof(layer) / sizeof(layer[0]); i++) {
        if (!strcmp(s, layer[i])) return "per-layer non-routed Q8 keep-list";
    }
    return NULL;
}

static bool q4_layer(const params *pa, int layer) {
    return pa->q4_start >= 0 && layer >= pa->q4_start && layer <= pa->q4_end;
}

static decision classify_tensor(const tensor *t, const params *pa, q36q_type *type, const char **why) {
    int layer = -1;
    char kind[32];
    if (parse_routed(t->name, &layer, kind, sizeof(kind))) {
        if (q4_layer(pa, layer)) {
            *type = Q36Q_TYPE_Q4_K;
            *why = "routed expert q4 window";
            return DEC_Q4;
        }
        if (!strcmp(kind, "gate") || !strcmp(kind, "up")) {
            *type = Q36Q_TYPE_IQ2_XXS;
            *why = "routed expert gate/up";
            return DEC_GATE_UP;
        }
        *type = Q36Q_TYPE_Q2_K;
        *why = "routed expert down/out";
        return DEC_DOWN;
    }
    const char *r = keep_reason(t->name);
    if (!r) {
        fprintf(stderr, "error: tensor is neither routed expert nor explicit keep-list entry: %s\n", t->name);
        exit(1);
    }
    *type = t->type == Q36Q_TYPE_BF16 || t->type == Q36Q_TYPE_F16 ?
            Q36Q_TYPE_Q8_0 : t->type;
    *why = r;
    return DEC_KEEP;
}

static int max_routed_layer(const gguf *in) {
    int max = -1;
    for (uint64_t i = 0; i < in->n_tensors; i++) {
        int layer = -1;
        char kind[32];
        if (parse_routed(in->tensors[i].name, &layer, kind, sizeof(kind)) && layer > max) max = layer;
    }
    if (max < 0) die("no routed expert tensors found");
    return max;
}

static size_t extra_imatrix_kv_size(const imatrix *im) {
    if (!im || !im->n_entries) return 0;
    size_t n = 0;
    n = size_add(n, size_add(size_add(str_size(Q36_KV_IMATRIX_FILE), 4), str_size(im->file)));
    n = size_add(n, size_add(str_size(Q36_KV_IMATRIX_ENTRIES), 12));
    if (im->dataset)
        n = size_add(n, size_add(size_add(str_size(Q36_KV_IMATRIX_DATASET), 4), str_size(im->dataset)));
    if (im->chunks > 0) n = size_add(n, size_add(str_size(Q36_KV_IMATRIX_CHUNKS), 12));
    return n;
}

static uint64_t extra_imatrix_kv_count(const imatrix *im) {
    if (!im || !im->n_entries) return 0;
    return 2 + (im->dataset != NULL) + (im->chunks > 0);
}

static void write_imatrix_kv(FILE *fp, const imatrix *im) {
    if (!im || !im->n_entries) return;
    put_str(fp, Q36_KV_IMATRIX_FILE);
    put_u32(fp, GGUF_TYPE_STRING);
    put_str(fp, im->file);
    put_str(fp, Q36_KV_IMATRIX_ENTRIES);
    put_u32(fp, GGUF_TYPE_UINT64);
    put_u64(fp, (uint64_t)im->n_entries);
    if (im->dataset) {
        put_str(fp, Q36_KV_IMATRIX_DATASET);
        put_u32(fp, GGUF_TYPE_STRING);
        put_str(fp, im->dataset);
    }
    if (im->chunks > 0) {
        put_str(fp, Q36_KV_IMATRIX_CHUNKS);
        put_u32(fp, GGUF_TYPE_UINT64);
        put_u64(fp, (uint64_t)im->chunks);
    }
}

static plan make_plan(const gguf *in, const imatrix *im, const params *pa) {
    plan p = {0};
    p.n_tensors = in->n_tensors;
    p.alignment = in->alignment;
    p.extra_kv = extra_imatrix_kv_count(im);
    p.tensors = xcalloc((size_t)p.n_tensors, sizeof(p.tensors[0]));
    size_t off = 0, info = 0;
    for (uint64_t i = 0; i < p.n_tensors; i++) {
        tensor *dst = &p.tensors[i];
        const tensor *src = &in->tensors[i];
        *dst = *src;
        while (dst->n_dims > 1 && dst->ne[dst->n_dims - 1] == 1) dst->n_dims--;
        const char *why = NULL;
        q36q_type target = src->type;
        decision d = classify_tensor(src, pa, &target, &why);
        (void)why;
        if (d == DEC_GATE_UP) p.n_gate_up++;
        else if (d == DEC_DOWN) p.n_down++;
        else if (d == DEC_Q4) p.n_q4++;
        else p.n_keep++;
        if (target != src->type && src->type != Q36Q_TYPE_Q8_0 &&
            src->type != Q36Q_TYPE_BF16 && src->type != Q36Q_TYPE_F16 &&
            src->type != Q36Q_TYPE_F32) {
            fprintf(stderr, "error: unsupported quantization source: %s is %s\n",
                    src->name, q36q_type_name(src->type));
            exit(1);
        }
        if (q36q_can_quantize(target) && src->ne[0] % q36q_block_size(target)) {
            fprintf(stderr, "error: %s ne[0]=%" PRId64 " not divisible by %s block size %" PRId64 "\n",
                    src->name, src->ne[0], q36q_type_name(target), q36q_block_size(target));
            exit(1);
        }
        dst->type = target;
        dst->size = tensor_size(target, dst->ne, dst->n_dims);
        dst->new_offset = off;
        off = size_add(off, pad(dst->size, p.alignment));
        size_t shape = size_mul((size_t)dst->n_dims, 8);
        info = size_add(info, size_add(str_size(dst->name), size_add(shape, 16)));
    }
    p.tensor_bytes = off;
    p.meta_size = size_add(24, in->kv.size);
    p.meta_size = size_add(p.meta_size, extra_imatrix_kv_size(im));
    p.meta_size = size_add(p.meta_size, info);
    p.data_offset = pad(p.meta_size, p.alignment);
    return p;
}

static void free_plan(plan *p) {
    free(p->tensors);
    memset(p, 0, sizeof(*p));
}

static void write_padding(FILE *fp, size_t n) {
    static uint8_t z[4096];
    while (n) {
        size_t c = n < sizeof(z) ? n : sizeof(z);
        if (fwrite(z, 1, c, fp) != c) die("write padding failed");
        n -= c;
    }
}

static void copy_bytes(FILE *fp, const uint8_t *p, size_t n) {
    while (n) {
        size_t c = n > (1u << 20) ? (1u << 20) : n;
        if (fwrite(p, 1, c, fp) != c) die("write tensor failed");
        p += c;
        n -= c;
    }
}

static void decode_q8_rows(const uint8_t *src, int64_t ncols, int rows, float *dst) {
    const int64_t blocks = ncols / 32;
    const size_t row_size = q36q_row_size(Q36Q_TYPE_Q8_0, ncols);
    for (int r = 0; r < rows; r++) {
        const uint8_t *row = src + (size_t)r * row_size;
        float *out = dst + (size_t)r * (size_t)ncols;
        for (int64_t b = 0; b < blocks; b++) {
            const uint8_t *blk = row + (size_t)b * 34;
            float d = q36q_f16_to_f32(get_u16(blk));
            const int8_t *q = (const int8_t *)(blk + 2);
            for (int j = 0; j < 32; j++) out[(size_t)b * 32 + (size_t)j] = d * (float)q[j];
        }
    }
}

static void decode_rows(q36q_type type, const uint8_t *src,
                        int64_t ncols, int rows, float *dst) {
    if (type == Q36Q_TYPE_Q8_0) {
        decode_q8_rows(src, ncols, rows, dst);
        return;
    }
    size_t n = size_mul((size_t)rows, (size_t)ncols);
    if (type == Q36Q_TYPE_F32) {
        memcpy(dst, src, size_mul(n, sizeof(float)));
        return;
    }
    if (type == Q36Q_TYPE_F16 || type == Q36Q_TYPE_BF16) {
        for (size_t i = 0; i < n; i++) {
            uint16_t v = get_u16(src + i * sizeof(uint16_t));
            dst[i] = type == Q36Q_TYPE_F16 ? q36q_f16_to_f32(v) : q36q_bf16_to_f32(v);
        }
        return;
    }
    die("unsupported tensor decode type");
}

static float *synthetic_imatrix(const tensor *t) {
    const int64_t ncols = t->ne[0];
    const int rows = tensor_rows(t);
    const size_t src_row = q36q_row_size(t->type, ncols);
    float *acc = xcalloc((size_t)ncols, sizeof(float));
    size_t row_bytes = size_mul((size_t)ncols, sizeof(float));
    int chunk = (int)((16u << 20) / row_bytes);
    if (chunk < 1) chunk = 1;
    float *f32 = xmalloc(size_mul((size_t)chunk, row_bytes));
    const uint8_t *base = t->data;
    for (int r = 0; r < rows; r += chunk) {
        int nr = rows - r < chunk ? rows - r : chunk;
        decode_rows(t->type, base + (size_t)r * src_row, ncols, nr, f32);
        for (int i = 0; i < nr; i++) {
            float *row = f32 + (size_t)i * (size_t)ncols;
            for (int64_t c = 0; c < ncols; c++) acc[c] += row[c] * row[c];
        }
    }
    free(f32);
    return acc;
}

typedef struct {
    const tensor *src;
    const tensor *dst;
    const imatrix *im;
    const float *imat;
    uint8_t *out;
    int rows;
    int chunk;
    int next;
    int64_t ncols;
    size_t src_row;
    size_t dst_row;
    bool synthetic;
    bool use_imatrix;
    pthread_mutex_t lock;
} quant_job;

static void *quant_worker(void *arg) {
    quant_job *j = arg;
    size_t row_bytes = size_mul((size_t)j->ncols, sizeof(float));
    float *f32 = xmalloc(size_mul((size_t)j->chunk, row_bytes));
    q36q_quantize_init(j->dst->type);
    for (;;) {
        pthread_mutex_lock(&j->lock);
        int r = j->next;
        if (r >= j->rows) {
            pthread_mutex_unlock(&j->lock);
            break;
        }
        int nr = j->rows - r < j->chunk ? j->rows - r : j->chunk;
        if (j->use_imatrix && !j->synthetic && j->src->n_dims == 3 && j->src->ne[1] > 0) {
            int left = (int)(j->src->ne[1] - (int64_t)r % j->src->ne[1]);
            if (nr > left) nr = left;
        }
        j->next += nr;
        pthread_mutex_unlock(&j->lock);
        if (nr <= 0) die("empty tensor conversion chunk");

        const uint8_t *base = j->src->data;
        const float *imat = j->imat;
        if (j->use_imatrix && !j->synthetic && j->src->n_dims == 3 && j->src->ne[1] > 0) {
            imat = imatrix_row(j->im, j->src, r, j->ncols);
        }
        decode_rows(j->src->type, base + (size_t)r * j->src_row, j->ncols, nr, f32);
        size_t n = q36q_quantize_chunk(j->dst->type, f32,
                                       j->out + (size_t)r * j->dst_row,
                                       0, nr, j->ncols, imat);
        if (n != (size_t)nr * j->dst_row) die("quantized chunk size mismatch");
    }
    free(f32);
    return NULL;
}

static void convert_tensor(FILE *fp, const tensor *src, const tensor *dst,
                           const imatrix *im, bool synthetic_ok, int threads) {
    const int64_t ncols = src->ne[0];
    const int rows = tensor_rows(src);
    const size_t src_row = q36q_row_size(src->type, ncols);
    const size_t dst_row = q36q_row_size(dst->type, ncols);
    bool use_imatrix = dst->type != Q36Q_TYPE_Q8_0;
    const float *imat = use_imatrix ? imatrix_row(im, src, 0, ncols) : NULL;
    float *synthetic = NULL;
    if (!imat && q36q_requires_imatrix(dst->type)) {
        if (!synthetic_ok) {
            fprintf(stderr, "error: %s needs imatrix for %s; pass --allow-synthetic-imatrix to use weight energy\n",
                    src->name, q36q_type_name(dst->type));
            exit(1);
        }
        synthetic = synthetic_imatrix(src);
        imat = synthetic;
    }
    size_t row_bytes = size_mul((size_t)ncols, sizeof(float));
    int chunk = (int)((16u << 20) / row_bytes);
    if (chunk < 1) chunk = 1;
    if (threads < 1) threads = 1;
    if (threads > rows) threads = rows;

    size_t out_bytes = size_mul((size_t)rows, dst_row);
    uint8_t *out = xmalloc(out_bytes);
    quant_job job = {
        .src = src, .dst = dst, .im = im, .imat = imat, .out = out,
        .rows = rows, .chunk = chunk, .ncols = ncols,
        .src_row = src_row, .dst_row = dst_row, .synthetic = synthetic != NULL,
        .use_imatrix = use_imatrix,
    };
    pthread_mutex_init(&job.lock, NULL);
    pthread_t *tid = xcalloc((size_t)threads, sizeof(tid[0]));
    for (int i = 1; i < threads; i++) {
        int err = pthread_create(&tid[i], NULL, quant_worker, &job);
        if (err) die_pthread(err, "pthread_create");
    }
    quant_worker(&job);
    for (int i = 1; i < threads; i++) {
        int err = pthread_join(tid[i], NULL);
        if (err) die_pthread(err, "pthread_join");
    }
    pthread_mutex_destroy(&job.lock);
    free(tid);
    if (fwrite(out, 1, out_bytes, fp) != out_bytes) {
        die("write quantized tensor failed");
    }
    free(out);
    free(synthetic);
}

static void write_gguf(const gguf *in, const plan *p, const imatrix *im, const params *pa) {
    FILE *fp = fopen(pa->out, "wb");
    if (!fp) die_errno("open output", pa->out);
    if (fwrite("GGUF", 1, 4, fp) != 4) die("write magic failed");
    put_u32(fp, in->version);
    put_u64(fp, in->n_tensors);
    put_u64(fp, in->n_kv + p->extra_kv);
    if (fwrite(in->kv.data, 1, in->kv.size, fp) != in->kv.size) die("write KV failed");
    write_imatrix_kv(fp, im);
    for (uint64_t i = 0; i < p->n_tensors; i++) {
        const tensor *t = &p->tensors[i];
        put_str(fp, t->name);
        put_u32(fp, (uint32_t)t->n_dims);
        for (int j = 0; j < t->n_dims; j++) put_u64(fp, (uint64_t)t->ne[j]);
        put_u32(fp, (uint32_t)t->type);
        put_u64(fp, t->new_offset);
    }
    long pos = ftell(fp);
    if (pos < 0 || (size_t)pos > p->data_offset) die("metadata size mismatch");
    write_padding(fp, p->data_offset - (size_t)pos);

    for (uint64_t i = 0; i < p->n_tensors; i++) {
        const tensor *src = &in->tensors[i];
        const tensor *dst = &p->tensors[i];
        fprintf(stderr, "[%4" PRIu64 "/%4" PRIu64 "] %s %s -> %s\n",
                i + 1, p->n_tensors, src->name, q36q_type_name(src->type), q36q_type_name(dst->type));
        if (src->type == dst->type) {
            copy_bytes(fp, src->data, src->size);
        } else {
            convert_tensor(fp, src, dst, im, pa->synthetic_imatrix, pa->threads);
        }
        write_padding(fp, pad(dst->size, p->alignment) - dst->size);
    }
    long end = ftell(fp);
    if (end < 0 || (size_t)end != size_add(p->data_offset, p->tensor_bytes))
        die("output size mismatch");
    fclose(fp);
}

static void write_audit(const gguf *in, const plan *p, const params *pa, const char *path) {
    FILE *fp = fopen(path, "w");
    if (!fp) die_errno("open audit", path);
    fprintf(fp, "tensor\tsource_type\ttarget_type\tdecision\n");
    for (uint64_t i = 0; i < p->n_tensors; i++) {
        const char *why = NULL;
        q36q_type type = in->tensors[i].type;
        classify_tensor(&in->tensors[i], pa, &type, &why);
        fprintf(fp, "%s\t%s\t%s\t%s\n",
                in->tensors[i].name,
                q36q_type_name(in->tensors[i].type),
                q36q_type_name(p->tensors[i].type),
                why);
    }
    fclose(fp);
}

static void print_plan(const gguf *in, const plan *p, const params *pa) {
    size_t changed = 0;
    for (uint64_t i = 0; i < p->n_tensors; i++) {
        if (in->tensors[i].type != p->tensors[i].type) {
            changed++;
            printf("type_change: %s %s -> %s\n",
                   in->tensors[i].name,
                   q36q_type_name(in->tensors[i].type),
                   q36q_type_name(p->tensors[i].type));
        }
    }
    if (pa->q4_start >= 0) printf("q4_expert_layers: %d-%d\n", pa->q4_start, pa->q4_end);
    printf("n_tensors: %" PRIu64 "\n", p->n_tensors);
    printf("routed_gate_up_iq2_xxs: %zu\n", p->n_gate_up);
    printf("routed_down_out_q2_k: %zu\n", p->n_down);
    printf("routed_q4_k: %zu\n", p->n_q4);
    printf("explicit_keep_list: %zu\n", p->n_keep);
    printf("type_changes: %zu\n", changed);
    printf("meta_bytes: %zu\n", p->data_offset);
    printf("tensor_bytes_unpadded: %zu\n", p->tensor_bytes);
    printf("approx_file_bytes: %zu\n", size_add(p->data_offset, p->tensor_bytes));
}

static bool parse_range(const char *s, int *start, int *end) {
    int a = -1, b = -1, pos = 0;
    if (sscanf(s, "%d-%d%n", &a, &b, &pos) != 2) return false;
    if (pos != (int)strlen(s) || a < 0 || b < a) return false;
    *start = a;
    *end = b;
    return true;
}

static bool exists(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return false;
    close(fd);
    return true;
}

static char *need(int argc, char **argv, int *i, const char *arg) {
    if (++*i >= argc) {
        fprintf(stderr, "error: missing value for %s\n", arg);
        exit(1);
    }
    return argv[*i];
}

static void add_input(params *p, char *path) {
    p->inputs = xrealloc(p->inputs, (size_t)(p->n_inputs + 1) * sizeof(p->inputs[0]));
    p->inputs[p->n_inputs++] = path;
}

static void usage(const char *argv0) {
    printf("usage: %s [--in MODEL.gguf ...] --imatrix q36.dat --out OUT.gguf [options]\n", argv0);
    printf("\nQwen3.6/Q36 mixed quantizer. Inputs are mmaped and decoded in chunks.\n\n");
    printf("recipe:\n");
    printf("  blk.N.ffn_gate_exps.weight -> iq2_xxs\n");
    printf("  blk.N.ffn_up_exps.weight   -> iq2_xxs\n");
    printf("  blk.N.ffn_down_exps.weight -> q2_k\n");
    printf("  optional q4 expert window   -> q4_k for gate/up/down\n");
    printf("  explicit keep-list f16/bf16 tensors -> q8_0; f32 tensors stay f32\n\n");
    printf("options:\n");
    printf("  --in FILE                  input GGUF; repeat in shard order, default %s\n", Q36_DEFAULT_IN);
    printf("  --out FILE                 output GGUF\n");
    printf("  --imatrix FILE             llama-imatrix legacy .dat or GGUF imatrix file\n");
    printf("  --imatrix-strict           fail if any quantized tensor has no imatrix entry\n");
    printf("  --allow-synthetic-imatrix  fallback to weight-energy importance for iq2_xxs\n");
    printf("  --audit FILE               write TSV tensor decision audit\n");
    printf("  --threads N                quantization workers, default 1\n");
    printf("  --q4-expert-layers A-B     quantize routed experts in layers A..B as q4_k\n");
    printf("  --q4-expert-last N         quantize routed experts in the last N layers as q4_k\n");
    printf("  --dry-run                  parse metadata and print plan only\n");
    printf("  --force                    overwrite output\n");
}

static params parse_args(int argc, char **argv) {
    params p = { .threads = 1, .q4_start = -1, .q4_end = -1 };
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(argv[0]);
            exit(0);
        } else if (!strcmp(a, "--in") || !strcmp(a, "--model")) {
            add_input(&p, need(argc, argv, &i, a));
        } else if (!strcmp(a, "--out")) {
            p.out = need(argc, argv, &i, a);
        } else if (!strcmp(a, "--imatrix")) {
            p.imatrix = need(argc, argv, &i, a);
        } else if (!strcmp(a, "--audit")) {
            p.audit = need(argc, argv, &i, a);
        } else if (!strcmp(a, "--dry-run")) {
            p.dry_run = true;
        } else if (!strcmp(a, "--force")) {
            p.force = true;
        } else if (!strcmp(a, "--imatrix-strict")) {
            p.strict_imatrix = true;
        } else if (!strcmp(a, "--allow-synthetic-imatrix")) {
            p.synthetic_imatrix = true;
        } else if (!strcmp(a, "--threads") || !strcmp(a, "-t")) {
            p.threads = atoi(need(argc, argv, &i, a));
        } else if (!strcmp(a, "--q4-expert-layers")) {
            if (!parse_range(need(argc, argv, &i, a), &p.q4_start, &p.q4_end)) {
                die("bad --q4-expert-layers, expected A-B");
            }
        } else if (!strcmp(a, "--q4-expert-last")) {
            p.q4_last = atoi(need(argc, argv, &i, a));
            if (p.q4_last < 1) die("--q4-expert-last must be >= 1");
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", a);
            exit(1);
        }
    }
    if (p.threads < 1) die("--threads must be >= 1");
    if (p.q4_last && p.q4_start >= 0) die("use either --q4-expert-last or --q4-expert-layers");
    if (!p.dry_run && !p.out) die("--out is required unless --dry-run is used");
    if (!p.imatrix && !p.synthetic_imatrix) die("--imatrix is required for iq2_xxs; or pass --allow-synthetic-imatrix");
    if (p.out && exists(p.out) && !p.force) die("output exists; use --force");
    if (!p.n_inputs) add_input(&p, Q36_DEFAULT_IN);
    return p;
}

int main(int argc, char **argv) {
    params pa = parse_args(argc, argv);
    imatrix im = {0};
    if (pa.imatrix) imatrix_load(&im, pa.imatrix, pa.strict_imatrix);
    gguf in = gguf_open_many(pa.inputs, pa.n_inputs);
    if (pa.q4_last) {
        int last = max_routed_layer(&in);
        if (pa.q4_last > last + 1) die("--q4-expert-last exceeds routed layer count");
        pa.q4_end = last;
        pa.q4_start = last - pa.q4_last + 1;
    }
    plan p = make_plan(&in, &im, &pa);
    print_plan(&in, &p, &pa);
    if (pa.audit) write_audit(&in, &p, &pa, pa.audit);
    if (!pa.dry_run) write_gguf(&in, &p, &im, &pa);
    free_plan(&p);
    gguf_close(&in);
    imatrix_free(&im);
    free(pa.inputs);
    return 0;
}
