/* Layer pipelines follow DwarfStar's coordinator/worker model and CLI.
 * Prefix and suffix stages retain independent KV state. The coordinator
 * projects the final hidden row, avoiding vocabulary-sized return packets.
 * Prefill sends a bounded window of chunks: stage A can compute the next
 * chunk while stage B consumes the previous one. MTP verification pipelines
 * candidate tokens in the same way; plain decode is serial. */
#include "q36_distributed.h"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define DIST_MAGIC UINT32_C(0x51333644)
#define DIST_VERSION 4u
#define DIST_MAX_FRAME (64u * 1024u * 1024u)
#define DIST_HELLO 1u
#define DIST_ERROR 2u
#define DIST_TRANSPORT_ERROR 2
#define DIST_CREATE 3u
#define DIST_WORK 4u
#define DIST_RESULT 5u
#define DIST_FREE 6u
#define DIST_COMMIT 7u
#define DIST_RESET 1u
#define DIST_LOGITS 2u
#define DIST_ROWS 4u
#define DIST_SPEC 8u
#define DIST_BEGIN 16u

struct q36_dist_engine {
    q36_engine *engine;
    q36_dist_options opt;
    q36_dist_model_info info, peer;
    pthread_mutex_t mutex;
    int listener, fd;
    uint32_t next_id, generation;
};

struct q36_dist_session {
    q36_dist_engine *engine;
    uint32_t id, generation, ctx;
};

typedef struct dist_worker_session {
    uint32_t id, ctx, spec_pos;
    bool spec;
    q36_session *session;
    struct dist_worker_session *next;
} dist_worker_session;

/* HC models exchange the complete residual stream; the vocabulary head
 * still consumes one embedding-width row returned by the output mixer. */
static uint32_t dist_hidden_width(const q36_dist_model_info *info) {
    return info->n_hidden ? info->n_hidden : info->n_embd;
}

static int dist_error(char *err, size_t len, const char *fmt, ...) {
    if (err && len) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, len, fmt, ap);
        va_end(ap);
    }
    return 1;
}

static double dist_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Bound every stalled transfer, and let the frontend cancel a waiting route.
 * SIGPIPE is suppressed per socket/send, never by changing process handlers. */
static int dist_io(int fd, void *buffer, size_t len, bool write,
                   q36_session *owner) {
    unsigned char *p = buffer;
    double deadline = dist_now() + 120;
    while (len) {
        if (owner && q36_session_distributed_cancelled(owner)) {
            errno = ECANCELED;
            return 1;
        }
        struct pollfd pollfd = {fd, write ? POLLOUT : POLLIN, 0};
        int ready = poll(&pollfd, 1, 100);
        if (ready < 0 && errno == EINTR) continue;
        if (ready < 0) return 1;
        if (!ready) {
            if (dist_now() < deadline) continue;
            errno = ETIMEDOUT;
            return 1;
        }
        ssize_t n;
        if (write) {
#ifdef MSG_NOSIGNAL
            n = send(fd, p, len, MSG_NOSIGNAL | MSG_DONTWAIT);
#else
            n = send(fd, p, len, MSG_DONTWAIT);
#endif
        } else n = recv(fd, p, len, MSG_DONTWAIT);
        if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        if (n <= 0) {
            if (!n) errno = ECONNRESET;
            return 1;
        }
        p += n;
        len -= (size_t)n;
        deadline = dist_now() + 120;
    }
    return 0;
}

static void dist_socket_options(int fd) {
    /* Agent shell tools must not keep the route or listener alive after exit. */
    fcntl(fd, F_SETFD, FD_CLOEXEC);
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
#ifdef SO_NOSIGPIPE
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
}

static int dist_endpoint(const char *host, int port, bool listen_socket) {
    char service[16];
    struct addrinfo hints = {0}, *list = NULL;
    snprintf(service, sizeof(service), "%d", port);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = listen_socket ? AI_PASSIVE : 0;
    if (getaddrinfo(host, service, &hints, &list)) return -1;
    int fd = -1;
    for (struct addrinfo *a = list; a; a = a->ai_next) {
        fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd < 0) continue;
        dist_socket_options(fd);
        if (listen_socket) {
            int one = 1;
            setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
            if (!bind(fd, a->ai_addr, a->ai_addrlen) && !listen(fd, 8)) break;
        } else if (!connect(fd, a->ai_addr, a->ai_addrlen)) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(list);
    return fd;
}

/* All words, including float bit patterns, use network byte order. */
static void dist_put(uint8_t *p, uint32_t value) {
    value = htonl(value);
    memcpy(p, &value, 4);
}

static uint32_t dist_get(const uint8_t *p) {
    uint32_t value;
    memcpy(&value, p, 4);
    return ntohl(value);
}

static int dist_send(int fd, uint32_t type, void *data, uint32_t bytes,
                     q36_session *owner) {
    uint32_t h[3] = {htonl(DIST_MAGIC), htonl(type), htonl(bytes)};
    return dist_io(fd, h, sizeof(h), true, owner) ||
           dist_io(fd, data, bytes, true, owner);
}

static int dist_recv(int fd, uint32_t *type, uint8_t **data, uint32_t *bytes,
                     q36_session *owner, char *err, size_t errlen) {
    uint32_t h[3];
    *data = NULL;
    if (dist_io(fd, h, sizeof(h), false, owner)) {
        dist_error(err, errlen, "pipeline receive: %s", strerror(errno));
        return DIST_TRANSPORT_ERROR;
    }
    *type = ntohl(h[1]);
    *bytes = ntohl(h[2]);
    if (ntohl(h[0]) != DIST_MAGIC || *bytes > DIST_MAX_FRAME)
        return dist_error(err, errlen, "invalid pipeline frame");
    *data = malloc((size_t)*bytes + 1u);
    if (!*data) return dist_error(err, errlen, "pipeline allocation failed");
    if (dist_io(fd, *data, *bytes, false, owner)) {
        free(*data);
        *data = NULL;
        dist_error(err, errlen, "pipeline payload: %s", strerror(errno));
        return DIST_TRANSPORT_ERROR;
    }
    (*data)[*bytes] = 0;
    if (*type == DIST_ERROR) {
        dist_error(err, errlen, "worker: %.*s", (int)(*bytes > 255 ? 255 : *bytes), *data);
        free(*data);
        *data = NULL;
        return 1;
    }
    return 0;
}

static void dist_disconnect(q36_dist_engine *d) {
    if (d->fd >= 0) close(d->fd);
    d->fd = -1;
}

static int dist_hello(q36_dist_engine *d, int fd, q36_session *owner,
                      char *err, size_t errlen) {
    q36_dist_model_info *i = &d->info;
    uint32_t words[] = {DIST_VERSION, i->n_layers, i->n_embd, i->n_vocab,
        i->layer_start, i->layer_end, i->context_size, i->prefill_cap,
        (uint32_t)(i->fingerprint >> 32), (uint32_t)i->fingerprint,
        i->speculation, i->mtp, dist_hidden_width(i)};
    uint8_t hello[sizeof(words)];
    for (size_t k = 0; k < sizeof(words)/sizeof(*words); k++) dist_put(hello + 4*k, words[k]);
    if (dist_send(fd, DIST_HELLO, hello, sizeof(hello), owner)) {
        dist_error(err, errlen, "pipeline hello: %s", strerror(errno));
        return DIST_TRANSPORT_ERROR;
    }
    /* A coordinator accepts the route on its first inference request. An idle
     * REPL may wait arbitrarily long before that request, so workers wait too. */
    if (d->opt.role == Q36_DISTRIBUTED_WORKER) {
        struct pollfd p = {fd, POLLIN, 0};
        int ready;
        do { ready = poll(&p, 1, -1); } while (ready < 0 && errno == EINTR);
        if (ready < 0) {
            dist_error(err, errlen, "pipeline hello poll: %s", strerror(errno));
            return DIST_TRANSPORT_ERROR;
        }
    }
    uint32_t type, bytes;
    uint8_t *data;
    int received = dist_recv(fd, &type, &data, &bytes, owner, err, errlen);
    if (received) return received;
    int rc = 1;
    if (type != DIST_HELLO || bytes != sizeof(hello)) {
        dist_error(err, errlen, "invalid pipeline hello");
        goto done;
    }
    uint32_t peer[13];
    for (int k = 0; k < 13; k++) peer[k] = dist_get(data + 4*k);
    if (peer[0] != DIST_VERSION || peer[1] != i->n_layers || peer[2] != i->n_embd ||
        peer[3] != i->n_vocab || peer[8] != words[8] || peer[9] != words[9] ||
        !peer[6] || peer[6] > Q36_CONTEXT_ALLOC_MAX || !peer[7] || peer[7] > 8192 ||
        peer[10] > 1u || peer[11] > 1u || peer[12] != words[12]) {
        dist_error(err, errlen, "pipeline model, protocol, context or batch mismatch");
        goto done;
    }
    bool coordinator = d->opt.role == Q36_DISTRIBUTED_COORDINATOR;
    if ((coordinator && (peer[4] != i->layer_end + 1u || peer[5] + 1u != i->n_layers)) ||
        (!coordinator && (peer[4] != 0 || peer[5] + 1u != i->layer_start))) {
        dist_error(err, errlen, "pipeline layer ranges have a gap or overlap");
        goto done;
    }
    if ((coordinator && peer[11]) || ((i->mtp || peer[11]) && (!i->speculation || !peer[10]))) {
        dist_error(err, errlen, "pipeline MTP requires resident Vulkan stages and a coordinator MTP head");
        goto done;
    }
    d->peer = (q36_dist_model_info){.n_layers=peer[1], .n_embd=peer[2],
        .n_vocab=peer[3], .layer_start=peer[4], .layer_end=peer[5],
        .context_size=peer[6], .prefill_cap=peer[7], .speculation=peer[10], .mtp=peer[11], .n_hidden=peer[12]};
    rc = 0;
done:
    free(data);
    return rc;
}

static int dist_route(q36_dist_engine *d, q36_session *owner,
                      char *err, size_t errlen) {
    if (d->fd >= 0) return 0;
    fprintf(stderr, "q36: waiting for pipeline worker on %s:%d\n",
            d->opt.listen_host, d->opt.listen_port);
    for (;;) {
        if (q36_session_distributed_cancelled(owner))
            return dist_error(err, errlen, "interrupted waiting for pipeline worker");
        struct pollfd p = {d->listener, POLLIN, 0};
        int n = poll(&p, 1, 100);
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) return dist_error(err, errlen, "pipeline accept: %s", strerror(errno));
        if (!n) continue;
        int fd = accept(d->listener, NULL, NULL);
        if (fd < 0) continue;
        dist_socket_options(fd);
        if (dist_hello(d, fd, owner, err, errlen)) { close(fd); return 1; }
        d->fd = fd;
        d->generation++;
        fprintf(stderr, "q36: pipeline ready: layers %u:%u -> %u:output, FP32 activations, coordinator output head\n",
                d->info.layer_start, d->info.layer_end, d->peer.layer_start);
        return 0;
    }
}

static int dist_result_rows(q36_dist_session *s, q36_session *owner, uint32_t pos,
        float *logits, float *hidden, uint32_t rows, char *err, size_t errlen) {
    uint32_t type, bytes;
    uint8_t *data;
    q36_dist_engine *d = s->engine;
    if (dist_recv(d->fd, &type, &data, &bytes, owner, err, errlen)) return 1;
    uint64_t values = (uint64_t)rows * d->info.n_embd;
    uint64_t expected = 8u + values * sizeof(float);
    int rc = 0;
    bool allocated = !hidden && rows;
    if (type != DIST_RESULT || bytes != expected || dist_get(data) != s->id ||
        dist_get(data + 4) != pos) rc = dist_error(err, errlen, "pipeline result is out of sequence");
    else if (rows) {
        if (allocated) hidden = malloc((size_t)values * sizeof(float));
        if (!hidden) rc = dist_error(err, errlen, "pipeline output allocation failed");
        else {
            for (uint64_t j = 0; j < values; j++) {
                uint32_t bits = dist_get(data + 8u + 4u*j);
                memcpy(hidden + j, &bits, 4);
            }
            if (logits) rc = q36_session_project_logits(owner,
                hidden + (uint64_t)(rows - 1u) * d->info.n_embd, logits, err, errlen);
        }
    }
    if (allocated) free(hidden);
    free(data);
    return rc;
}

static int dist_result(q36_dist_session *s, q36_session *owner, uint32_t pos,
        float *logits, char *err, size_t errlen) {
    return dist_result_rows(s, owner, pos, logits, NULL, logits ? 1u : 0u, err, errlen);
}

static int dist_session_ready(q36_dist_session *s, q36_session *owner,
                              char *err, size_t errlen) {
    q36_dist_engine *d = s->engine;
    if (dist_route(d, owner, err, errlen)) return 1;
    if (s->generation == d->generation) return 0;
    if (s->ctx > d->peer.context_size)
        return dist_error(err, errlen, "worker --ctx is smaller than the coordinator session");
    uint8_t create[8];
    dist_put(create, s->id);
    dist_put(create + 4, s->ctx);
    if (dist_send(d->fd, DIST_CREATE, create, sizeof(create), owner) ||
        dist_result(s, owner, 0, NULL, err, errlen)) return 1;
    s->generation = d->generation;
    return 0;
}

bool q36_dist_enabled(const q36_dist_options *opt) {
    return opt && opt->role != Q36_DISTRIBUTED_NONE;
}

int q36_dist_engine_open(q36_dist_engine **out, q36_engine *engine,
        const q36_dist_options *opt, const q36_dist_model_info *info) {
    q36_dist_engine *d = calloc(1, sizeof(*d));
    if (!d) return 1;
    d->engine = engine;
    d->opt = *opt;
    d->info = *info;
    d->listener = d->fd = -1;
    d->next_id = 1;
    pthread_mutexattr_t attr;
    if (pthread_mutexattr_init(&attr)) { free(d); return 1; }
    int rc = pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    if (!rc) rc = pthread_mutex_init(&d->mutex, &attr);
    pthread_mutexattr_destroy(&attr);
    if (rc) { free(d); return 1; }
    if (opt->role == Q36_DISTRIBUTED_COORDINATOR) {
        d->listener = dist_endpoint(opt->listen_host, opt->listen_port, true);
        if (d->listener < 0) {
            fprintf(stderr, "q36: cannot listen for pipeline worker at %s:%d\n", opt->listen_host, opt->listen_port);
            q36_dist_engine_close(d);
            return 1;
        }
    }
    *out = d;
    return 0;
}

void q36_dist_engine_close(q36_dist_engine *d) {
    if (!d) return;
    dist_disconnect(d);
    if (d->listener >= 0) close(d->listener);
    pthread_mutex_destroy(&d->mutex);
    free(d);
}

int q36_dist_session_create(q36_dist_session **out, q36_dist_engine *engine, int ctx) {
    q36_dist_session *s = calloc(1, sizeof(*s));
    if (!s) return 1;
    s->engine = engine;
    s->ctx = (uint32_t)ctx;
    pthread_mutex_lock(&engine->mutex);
    s->id = engine->next_id++;
    pthread_mutex_unlock(&engine->mutex);
    if (!s->id) { free(s); return 1; }
    *out = s;
    return 0;
}

void q36_dist_session_lock(q36_dist_session *s) { pthread_mutex_lock(&s->engine->mutex); }
void q36_dist_session_unlock(q36_dist_session *s) { pthread_mutex_unlock(&s->engine->mutex); }

void q36_dist_session_free(q36_dist_session *s) {
    if (!s) return;
    q36_dist_engine *d = s->engine;
    pthread_mutex_lock(&d->mutex);
    if (d->fd >= 0 && s->generation == d->generation) {
        uint8_t data[4];
        dist_put(data, s->id);
        if (dist_send(d->fd, DIST_FREE, data, sizeof(data), NULL) ||
            dist_result(s, NULL, 0, NULL, NULL, 0)) dist_disconnect(d);
    }
    pthread_mutex_unlock(&d->mutex);
    free(s);
}

typedef struct {
    float *hidden;
    int tops[Q36_DIST_VERIFY_MAX], excluded, eos;
    uint32_t count;
    bool stopped;
} dist_verify;

typedef struct {
    int fd, result, error;
    uint8_t *data;
    uint32_t bytes;
} dist_write;

static void *dist_writer(void *opaque) {
    dist_write *w = opaque;
    w->result = dist_send(w->fd, DIST_WORK, w->data, w->bytes, NULL);
    w->error = errno;
    return NULL;
}

typedef struct {
    q36_dist_session *session;
    const int *tokens;
    float *input, *target;
    uint32_t count, pos;
    bool final, done;
    int result;
    char error[256];
    pthread_mutex_t mutex;
    pthread_cond_t ready;
} dist_prefill;

static int dist_eval_span_body(q36_dist_session *s, q36_session *owner,
        const int *tokens, uint32_t count, uint32_t pos0, bool report_progress,
        float *logits, dist_verify *verify, dist_prefill *prepared,
        char *err, size_t errlen) {
    q36_dist_engine *d = s->engine;
    uint32_t cap = verify ? 1u : d->info.prefill_cap;
    if (cap > count) cap = count;
    if (cap > d->peer.prefill_cap) cap = d->peer.prefill_cap;
    if (d->opt.prefill_chunk && cap > d->opt.prefill_chunk) cap = d->opt.prefill_chunk;
    uint32_t frame_cap = (DIST_MAX_FRAME - 16u) / ((1u + dist_hidden_width(&d->info)) * 4u);
    if (cap > frame_cap) cap = frame_cap;
    uint32_t local_cap = prepared ? count : cap;
    uint32_t window = d->opt.prefill_window ? d->opt.prefill_window : 2;
    uint64_t max_bytes = 16u + (uint64_t)cap * (1u + dist_hidden_width(&d->info)) * 4u;
    if (max_bytes > DIST_MAX_FRAME) return dist_error(err, errlen, "pipeline chunk exceeds frame limit");
    uint8_t *data = malloc((size_t)max_bytes);
    float *hidden = prepared ? prepared->input : malloc((size_t)local_cap * dist_hidden_width(&d->info) * sizeof(float));
    if (!data || !hidden) {
        free(data);
        if (!prepared) free(hidden);
        return dist_error(err, errlen, "pipeline allocation failed");
    }
    struct { uint32_t end, count, index; } pending[64];
    uint32_t head = 0, tail = 0, i = 0, local_start = 0, local_end = prepared ? count : 0;
    int rc = 0;
    double started = dist_now(), local_seconds = 0, wire_seconds = 0;
    double local_started, wire_started;
    while (i < count || head != tail) {
        if (owner && q36_session_distributed_cancelled(owner)) {
            rc = Q36_SESSION_SYNC_INTERRUPTED;
            dist_error(err, errlen, "interrupted");
            break;
        }
        bool receive = false, writing = false;
        pthread_t writer;
        dist_write write_job = {.fd=d->fd};
        if (i < count && tail - head < window) {
            if (i == local_end) {
                uint32_t batch = count - i;
                if (batch > local_cap) batch = local_cap;
                if (pos0 + i == 0 && q36_session_layer_slice_reset(owner, err, errlen)) { rc = 1; break; }
                /* Each stage can use its own batch size. Retain prefix rows
                 * while smaller suffix chunks travel through the window. */
                local_started = dist_now();
                if (q36_session_eval_layer_slice(owner, tokens + i, batch, pos0 + i,
                        d->info.layer_start, d->info.layer_end, NULL, hidden,
                        false, NULL, err, errlen)) { rc = 1; break; }
                local_seconds += dist_now() - local_started;
                local_start = i;
                local_end = i + batch;
            }
            uint32_t n = local_end - i;
            if (n > cap) n = cap;
            bool last = i + n == count && (!prepared || prepared->final);
            dist_put(data, s->id);
            dist_put(data + 4, pos0 + i);
            dist_put(data + 8, n);
            uint32_t flags = (pos0 + i == 0 ? DIST_RESET : 0) | (last || verify ? DIST_LOGITS : 0);
            if (d->info.mtp) flags |= DIST_ROWS;
            if (verify) flags |= DIST_SPEC | (i == 0 ? DIST_BEGIN : 0);
            dist_put(data + 12, flags);
            for (uint32_t j = 0; j < n; j++) dist_put(data + 16u + 4u*j, (uint32_t)tokens[i+j]);
            uint64_t values = (uint64_t)n * dist_hidden_width(&d->info);
            for (uint64_t j = 0; j < values; j++) {
                uint32_t bits;
                memcpy(&bits, hidden + (uint64_t)(i - local_start) * dist_hidden_width(&d->info) + j, 4);
                dist_put(data + 16u + 4u*n + 4u*j, bits);
            }
            uint32_t bytes = 16u + n * (1u + dist_hidden_width(&d->info)) * 4u;
            if (d->info.mtp && head != tail) {
                /* Large target-row replies must drain while the next frame
                 * is sent, including when the link has small TCP buffers. */
                write_job.data = data;
                write_job.bytes = bytes;
                int created = pthread_create(&writer, NULL, dist_writer, &write_job);
                if (created) { rc = dist_error(err, errlen, "pipeline sender: %s", strerror(created)); break; }
                writing = receive = true;
            } else if (dist_send(d->fd, DIST_WORK, data, bytes, owner)) {
                rc = dist_error(err, errlen, "pipeline send: %s", strerror(errno));
                break;
            }
            uint32_t slot = tail++ % 64;
            pending[slot].end = pos0 + i + n;
            pending[slot].count = n;
            pending[slot].index = i;
            i += n;
        } else receive = true;
        if (receive) {
            uint32_t slot = head++ % 64;
            uint32_t end = pending[slot].end, n = pending[slot].count, index = pending[slot].index;
            bool last = end == pos0 + count && (!prepared || prepared->final);
            float *out = !verify && last ? logits : NULL;
            float *rows = prepared ? prepared->target + (d->info.mtp ? (uint64_t)index * d->info.n_embd : 0) :
                verify ? verify->hidden + (uint64_t)index * d->info.n_embd : hidden;
            uint32_t returned = d->info.mtp ? n : last ? 1u : 0u;
            wire_started = dist_now();
            if (dist_result_rows(s, owner, end, out, rows, returned, err, errlen) ||
                (!prepared && d->info.mtp && !verify && q36_session_pipeline_mtp_catchup(owner,
                    tokens + index, n, pos0 + index, rows, err, errlen))) rc = 1;
            if (!rc && verify && !verify->stopped) {
                int top;
                rc = q36_session_pipeline_project(owner, rows, verify->excluded,
                    &top, err, errlen);
                /* Drain rows already in flight, but stop issuing rejected tails. */
                if (!rc) {
                    verify->tops[index] = top;
                    if (tokens[index] == verify->eos ||
                        (index + 1u < count && top != tokens[index + 1u])) {
                        verify->stopped = true;
                        count = i;
                    }
                }
            }
            wire_seconds += dist_now() - wire_started;
            if (!rc && report_progress)
                q36_session_distributed_progress(owner, (int)end, (int)(pos0 + count));
        }
        if (writing) {
            if (rc) shutdown(d->fd, SHUT_RDWR);
            pthread_join(writer, NULL);
            if (!rc && write_job.result)
                rc = dist_error(err, errlen, "pipeline send: %s", strerror(write_job.error));
        }
        if (rc) break;
    }
    if (d->opt.debug && !prepared) fprintf(stderr, "q36: pipeline %u tokens at %u, window %u, local_batch=%u wire_batch=%u, %.3f s local=%.3f s remote_wait=%.3f s\n",
                              count, pos0, window, local_cap, cap, dist_now() - started, local_seconds, wire_seconds);
    if (verify) verify->count = i;
    if (!prepared) free(hidden);
    free(data);
    return rc;
}

static void *dist_prefill_exchange(void *opaque) {
    dist_prefill *p = opaque;
    p->result = dist_eval_span_body(p->session, NULL, p->tokens, p->count,
        p->pos, false, NULL, NULL, p, p->error, sizeof(p->error));
    pthread_mutex_lock(&p->mutex);
    p->done = true;
    pthread_cond_signal(&p->ready);
    pthread_mutex_unlock(&p->mutex);
    return NULL;
}

/* The I/O thread only moves prepared rows. GPU catchup remains on the caller,
 * after the next prefix batch has been saved in its other host buffer. */
static int dist_prefill_span(q36_dist_session *s, q36_session *owner,
        const int *tokens, uint32_t count, uint32_t pos0, uint32_t cap,
        bool progress, float *logits, char *err, size_t errlen) {
    q36_dist_engine *d = s->engine;
    uint32_t wire = d->peer.prefill_cap;
    if (d->opt.prefill_chunk && wire > d->opt.prefill_chunk) wire = d->opt.prefill_chunk;
    uint32_t frame = (DIST_MAX_FRAME - 16u) / ((1u + dist_hidden_width(&d->info)) * 4u);
    if (wire > frame) wire = frame;
    uint64_t bytes = (uint64_t)cap * dist_hidden_width(&d->info) * sizeof(float);
    float *current = malloc(bytes), *next = malloc(bytes);
    float *target = malloc(d->info.mtp ? bytes : (uint64_t)d->info.n_embd * sizeof(float));
    dist_prefill job = {.session=s, .target=target};
    int rc = 1;
    if (!current || !next || !target) { dist_error(err, errlen, "pipeline allocation failed"); goto done; }
    if (pthread_mutex_init(&job.mutex, NULL)) { dist_error(err, errlen, "pipeline mutex failed"); goto done; }
    if (pthread_cond_init(&job.ready, NULL)) {
        pthread_mutex_destroy(&job.mutex);
        dist_error(err, errlen, "pipeline condition failed");
        goto done;
    }
    double started = dist_now(), local = 0, wait = 0, t;
    if (!pos0 && q36_session_layer_slice_reset(owner, err, errlen)) goto destroy;
    t = dist_now();
    if (q36_session_eval_layer_slice(owner, tokens, cap, pos0,
            d->info.layer_start, d->info.layer_end, NULL, current, false, NULL, err, errlen)) goto destroy;
    local += dist_now() - t;
    rc = 0;
    for (uint32_t i = 0; i < count;) {
        uint32_t n = count - i;
        if (n > cap) n = cap;
        if (q36_session_distributed_cancelled(owner)) {
            dist_error(err, errlen, "interrupted"); rc = Q36_SESSION_SYNC_INTERRUPTED; break;
        }
        job.tokens = tokens + i; job.input = current;
        job.count = n; job.pos = pos0 + i; job.final = i + n == count; job.done = false;
        pthread_t io;
        int created = pthread_create(&io, NULL, dist_prefill_exchange, &job);
        if (created) { rc = dist_error(err, errlen, "pipeline exchange: %s", strerror(created)); break; }
        uint32_t following = count - i - n;
        if (following > cap) following = cap;
        t = dist_now();
        if (following) rc = q36_session_eval_layer_slice(owner, tokens + i + n, following, pos0 + i + n,
            d->info.layer_start, d->info.layer_end, NULL, next, false, NULL, err, errlen);
        local += dist_now() - t;
        t = dist_now();
        pthread_mutex_lock(&job.mutex);
        while (!job.done) {
            pthread_mutex_unlock(&job.mutex);
            if (q36_session_distributed_cancelled(owner)) {
                dist_error(err, errlen, "interrupted"); rc = Q36_SESSION_SYNC_INTERRUPTED;
            }
            if (rc) shutdown(d->fd, SHUT_RDWR);
            pthread_mutex_lock(&job.mutex);
            if (!job.done) {
                struct timespec deadline;
                clock_gettime(CLOCK_REALTIME, &deadline);
                deadline.tv_nsec += 100000000;
                if (deadline.tv_nsec >= 1000000000) { deadline.tv_sec++; deadline.tv_nsec -= 1000000000; }
                pthread_cond_timedwait(&job.ready, &job.mutex, &deadline);
            }
        }
        pthread_mutex_unlock(&job.mutex);
        pthread_join(io, NULL);
        if (!rc && job.result) rc = dist_error(err, errlen, "%s", job.error);
        if (rc) break;
        if (d->info.mtp) {
            for (uint32_t j = 0; !rc && j < n;) {
                uint32_t rows = n - j;
                if (rows > wire) rows = wire;
                rc = q36_session_pipeline_mtp_catchup(owner, tokens + i + j, rows, pos0 + i + j,
                    target + (uint64_t)j * d->info.n_embd, err, errlen);
                j += rows;
            }
        }
        if (!rc && job.final) rc = q36_session_project_logits(owner,
            target + (d->info.mtp ? (uint64_t)(n - 1u) * d->info.n_embd : 0), logits, err, errlen);
        wait += dist_now() - t;
        if (rc) break;
        i += n;
        if (progress) q36_session_distributed_progress(owner, (int)(pos0 + i), (int)(pos0 + count));
        float *swap = current; current = next; next = swap;
    }
    if (d->opt.debug) fprintf(stderr, "q36: pipeline %u tokens at %u, local_batch=%u wire_batch=%u, overlapped, %.3f s local=%.3f s remote_wait=%.3f s\n",
        count, pos0, cap, wire, dist_now() - started, local, wait);
destroy:
    pthread_cond_destroy(&job.ready);
    pthread_mutex_destroy(&job.mutex);
done:
    free(current); free(next); free(target);
    return rc;
}

static int dist_eval_span(q36_dist_session *s, q36_session *owner,
        const int *tokens, uint32_t count, uint32_t pos0, bool progress,
        float *logits, dist_verify *verify, char *err, size_t errlen) {
    q36_dist_engine *d = s->engine;
    uint32_t cap = d->info.prefill_cap, wire = d->peer.prefill_cap;
    if (d->opt.prefill_chunk && wire > d->opt.prefill_chunk) wire = d->opt.prefill_chunk;
    uint32_t frame = (DIST_MAX_FRAME - 16u) / ((1u + dist_hidden_width(&d->info)) * 4u);
    if (wire > frame) wire = frame;
    if (!verify && count > cap && cap > wire)
        return dist_prefill_span(s, owner, tokens, count, pos0, cap, progress, logits, err, errlen);
    return dist_eval_span_body(s, owner, tokens, count, pos0, progress, logits, verify, NULL, err, errlen);
}

int q36_dist_session_sync(q36_dist_session *s, q36_session *owner,
        const q36_tokens *checkpoint, const q36_tokens *prompt, float *logits,
        char *err, size_t errlen) {
    q36_dist_engine *d = s->engine;
    pthread_mutex_lock(&d->mutex);
    bool replay = d->fd < 0 || s->generation != d->generation;
    int rc = dist_session_ready(s, owner, err, errlen);
    uint32_t start = replay ? 0 : (uint32_t)checkpoint->len;
    if (!rc && start < (uint32_t)prompt->len)
        rc = dist_eval_span(s, owner, prompt->v + start, (uint32_t)prompt->len - start,
                            start, true, logits, NULL, err, errlen);
    if (rc) {
        dist_disconnect(d);
        if (q36_session_distributed_cancelled(owner)) rc = Q36_SESSION_SYNC_INTERRUPTED;
    }
    pthread_mutex_unlock(&d->mutex);
    return rc;
}

int q36_dist_session_eval(q36_dist_session *s, q36_session *owner,
        const q36_tokens *checkpoint, int token, float *logits, char *err, size_t errlen) {
    q36_dist_engine *d = s->engine;
    pthread_mutex_lock(&d->mutex);
    bool replay = d->fd < 0 || s->generation != d->generation;
    int rc = dist_session_ready(s, owner, err, errlen);
    if (!rc && replay && checkpoint->len)
        rc = dist_eval_span(s, owner, checkpoint->v, (uint32_t)checkpoint->len, 0, true, logits, NULL, err, errlen);
    if (!rc) rc = dist_eval_span(s, owner, &token, 1, (uint32_t)checkpoint->len, false, logits, NULL, err, errlen);
    if (rc) {
        dist_disconnect(d);
        if (q36_session_distributed_cancelled(owner)) rc = Q36_SESSION_SYNC_INTERRUPTED;
    }
    pthread_mutex_unlock(&d->mutex);
    return rc;
}

/* The wire lock spans verification and commit: another session must never
 * interleave a transaction's saved recurrent frontier. */
int q36_dist_session_speculate(q36_dist_session *s, q36_session *owner,
        const q36_tokens *checkpoint, const int *tokens, uint32_t count,
        int eos, int excluded, float *logits, char *err, size_t errlen) {
    q36_dist_engine *d = s->engine;
    if (!d->info.mtp || !tokens || !logits || !checkpoint->len || !count ||
        count > Q36_DIST_VERIFY_MAX || (uint32_t)checkpoint->len > s->ctx ||
        count > s->ctx - (uint32_t)checkpoint->len) {
        dist_error(err, errlen, "invalid pipeline speculative block");
        return -1;
    }
    dist_verify verify = {
        .hidden = malloc((size_t)count * d->info.n_embd * sizeof(float)),
        .excluded = excluded, .eos = eos,
    };
    if (!verify.hidden) {
        dist_error(err, errlen, "pipeline verification allocation failed");
        return -1;
    }
    pthread_mutex_lock(&d->mutex);
    bool replay = d->fd < 0 || s->generation != d->generation;
    int rc = dist_session_ready(s, owner, err, errlen);
    uint32_t pos = (uint32_t)checkpoint->len;
    if (!rc && replay)
        rc = dist_eval_span(s, owner, checkpoint->v, pos, 0, false, logits, NULL, err, errlen);
    if (!rc) rc = q36_session_pipeline_spec_begin(owner, err, errlen);
    if (!rc) rc = dist_eval_span(s, owner, tokens, count, pos, false, NULL, &verify, err, errlen);
    uint32_t commit = 1;
    if (!rc) {
        while (commit < verify.count && tokens[commit - 1u] != eos) {
            if (verify.tops[commit - 1u] != tokens[commit]) break;
            commit++;
        }
    }
    if (!rc) {
        uint8_t data[8];
        dist_put(data, s->id);
        dist_put(data + 4, commit);
        /* Rebuild both frontiers concurrently; publish tokens after the ACK. */
        if (dist_send(d->fd, DIST_COMMIT, data, sizeof(data), owner) ||
            q36_session_pipeline_spec_commit(owner, commit, err, errlen) ||
            q36_session_pipeline_mtp_catchup(owner, tokens, commit, pos, verify.hidden, err, errlen) ||
            q36_session_pipeline_read_logits(owner, logits, err, errlen) ||
            dist_result(s, owner, pos + commit, NULL, err, errlen)) rc = 1;
    }
    if (d->opt.debug && !rc) fprintf(stderr, "q36: pipeline MTP drafted=%u verified=%u committed=%u\n",
                                    count, verify.count, commit);
    if (rc) dist_disconnect(d);
    pthread_mutex_unlock(&d->mutex);
    free(verify.hidden);
    return rc ? -1 : (int)commit;
}

static int dist_worker_result(int fd, uint32_t id, uint32_t pos,
                              const float *hidden, uint32_t n_embd) {
    uint32_t bytes = 8u + (hidden ? n_embd * 4u : 0);
    uint8_t *data = malloc(bytes);
    if (!data) return 1;
    dist_put(data, id);
    dist_put(data + 4, pos);
    if (hidden) for (uint32_t j = 0; j < n_embd; j++) {
        uint32_t bits;
        memcpy(&bits, hidden + j, 4);
        dist_put(data + 8u + 4u*j, bits);
    }
    int rc = dist_send(fd, DIST_RESULT, data, bytes, NULL);
    free(data);
    return rc;
}

static int dist_worker_loop(q36_dist_engine *d, int fd, char *err, size_t errlen) {
    dist_worker_session *sessions = NULL;
    int rc = 0;
    for (;;) {
        if (err && errlen) err[0] = '\0';
        struct pollfd idle = {fd, POLLIN, 0};
        int ready;
        do { ready = poll(&idle, 1, -1); } while (ready < 0 && errno == EINTR);
        if (ready < 0) { rc = dist_error(err, errlen, "worker poll failed"); break; }
        uint32_t type, bytes;
        uint8_t *data;
        if (dist_recv(fd, &type, &data, &bytes, NULL, err, errlen)) { rc = 1; break; }
        uint32_t id = bytes >= 4 ? dist_get(data) : 0;
        dist_worker_session **link = &sessions;
        while (*link && (*link)->id != id) link = &(*link)->next;
        dist_worker_session *s = *link;
        if (!id) rc = dist_error(err, errlen, "missing pipeline session id");
        else if (type == DIST_CREATE && bytes == 8 && !s) {
            uint32_t ctx = dist_get(data + 4);
            if (!ctx || ctx > d->info.context_size) rc = dist_error(err, errlen, "worker context exceeded");
            else {
                s = calloc(1, sizeof(*s));
                if (!s || q36_session_create(&s->session, d->engine, (int)ctx)) {
                    free(s);
                    rc = dist_error(err, errlen, "cannot create worker session");
                } else {
                    s->id = id;
                    s->ctx = ctx;
                    *link = s;
                    rc = dist_worker_result(fd, id, 0, NULL, 0);
                }
            }
        } else if (type == DIST_FREE && bytes == 4 && s) {
            *link = s->next;
            q36_session_free(s->session);
            free(s);
            rc = dist_worker_result(fd, id, 0, NULL, 0);
        } else if (type == DIST_COMMIT && bytes == 8 && s && s->spec) {
            uint32_t count = dist_get(data + 4);
            rc = q36_session_pipeline_spec_commit(s->session, count, err, errlen);
            if (!rc) {
                s->spec = false;
                rc = dist_worker_result(fd, id, s->spec_pos + count, NULL, 0);
            }
        } else if (type == DIST_WORK && bytes >= 16 && s) {
            uint32_t pos = dist_get(data + 4), n = dist_get(data + 8), flags = dist_get(data + 12);
            uint64_t expected = 16u + (uint64_t)n * (1u + dist_hidden_width(&d->info)) * 4u;
            if (!n || n > d->info.prefill_cap || expected != bytes ||
                pos > s->ctx || n > s->ctx - pos ||
                flags & ~(DIST_RESET | DIST_LOGITS | DIST_ROWS | DIST_SPEC | DIST_BEGIN) ||
                ((flags & DIST_RESET) && pos) ||
                ((flags & DIST_ROWS) && !d->peer.mtp) ||
                ((flags & DIST_BEGIN) && (!(flags & DIST_SPEC) || s->spec || !pos)) ||
                ((flags & DIST_SPEC) && (!d->peer.mtp || !d->info.speculation || n != 1u ||
                    !(flags & DIST_ROWS) || (flags & DIST_RESET) || (!(flags & DIST_BEGIN) && !s->spec))) ||
                (s->spec && !(flags & DIST_SPEC))) rc = dist_error(err, errlen, "invalid pipeline work shape");
            else {
                int *tokens = malloc((size_t)n * sizeof(int));
                float *hidden = malloc((size_t)n * dist_hidden_width(&d->info) * sizeof(float));
                uint32_t returned = flags & DIST_ROWS ? n : flags & DIST_LOGITS ? 1u : 0u;
                float *result = returned ? malloc((size_t)returned * d->info.n_embd * sizeof(float)) : NULL;
                if (!tokens || !hidden || (returned && !result)) rc = dist_error(err, errlen, "worker allocation failed");
                else {
                    for (uint32_t j = 0; j < n; j++) tokens[j] = (int)dist_get(data + 16u + 4u*j);
                    for (uint64_t j = 0; j < (uint64_t)n * dist_hidden_width(&d->info); j++) {
                        uint32_t bits = dist_get(data + 16u + 4u*n + 4u*j);
                        memcpy(hidden + j, &bits, 4);
                    }
                    if (flags & DIST_RESET) rc = q36_session_layer_slice_reset(s->session, err, errlen);
                    if (!rc && (flags & DIST_BEGIN)) {
                        rc = q36_session_pipeline_spec_begin(s->session, err, errlen);
                        if (!rc) { s->spec = true; s->spec_pos = pos; }
                    }
                    if (!rc) rc = q36_session_eval_layer_slice(s->session, tokens, n, pos,
                            d->info.layer_start, d->info.layer_end, hidden, NULL,
                            false, NULL, err, errlen);
                    if (!rc && result) rc = flags & DIST_ROWS ?
                        q36_session_read_final_rows(s->session, n, result, err, errlen) :
                        q36_session_read_final_hidden(s->session, n, result, err, errlen);
                    if (!rc) rc = dist_worker_result(fd, id, pos + n, result, returned * d->info.n_embd);
                }
                free(result);
                free(hidden);
                free(tokens);
            }
        } else rc = dist_error(err, errlen, "invalid pipeline operation or session");
        free(data);
        if (rc) {
            char *message = err && errlen && err[0] ? err : "worker response failed";
            dist_send(fd, DIST_ERROR, message, (uint32_t)strlen(message), NULL);
            break;
        }
    }
    while (sessions) {
        dist_worker_session *next = sessions->next;
        q36_session_free(sessions->session);
        free(sessions);
        sessions = next;
    }
    return rc;
}

int q36_dist_run(q36_engine *engine) {
    q36_dist_engine *d = q36_engine_distributed(engine);
    if (!d || d->opt.role != Q36_DISTRIBUTED_WORKER) return 1;
    fprintf(stderr, "q36: pipeline worker layers %u:output, coordinator %s:%d\n",
            d->info.layer_start, d->opt.coordinator_host, d->opt.coordinator_port);
    for (;;) {
        int fd = dist_endpoint(d->opt.coordinator_host, d->opt.coordinator_port, false);
        if (fd < 0) { struct timespec t = {1, 0}; nanosleep(&t, NULL); continue; }
        char err[256] = {0};
        int hello = dist_hello(d, fd, NULL, err, sizeof(err));
        if (hello) {
            fprintf(stderr, "q36: %s\n", err);
            close(fd);
            if (hello == DIST_TRANSPORT_ERROR) {
                /* A proxy can accept TCP while its upstream is still down. */
                struct timespec t = {1, 0};
                nanosleep(&t, NULL);
                continue;
            }
            return 1;
        }
        fprintf(stderr, "q36: pipeline worker connected\n");
        dist_worker_loop(d, fd, err, sizeof(err));
        close(fd);
        fprintf(stderr, "q36: pipeline disconnected: %s; waiting to reconnect\n", err);
    }
}

static bool dist_u32(const char *text, uint32_t *value) {
    if (!text || !*text) return false;
    uint64_t n = 0;
    for (const char *p = text; *p; p++) {
        if (*p < '0' || *p > '9') return false;
        n = n * 10u + (unsigned)(*p - '0');
        if (n > UINT32_MAX) return false;
    }
    *value = (uint32_t)n;
    return true;
}

void q36_dist_usage(FILE *fp) {
    fprintf(fp,
        "\nDistributed inference (two-stage TCP pipeline):\n"
        "  --role coordinator|worker\n"
        "  --layers A:B       Inclusive layers; use A:output for the worker.\n"
        "  --listen HOST PORT Coordinator TCP bind address.\n"
        "  --coordinator HOST PORT  Worker connects to this IP or hostname.\n"
        "  --dist-prefill-chunk N   Maximum tokens per pipeline chunk.\n"
        "  --dist-prefill-window N  Chunks in flight, 1..64 (default 2).\n"
        "  --debug            Report pipeline timing.\n");
}

q36_dist_cli_parse_result q36_dist_parse_cli_arg(const char *arg, int *index,
        int argc, char **argv, q36_dist_options *opt, char *err, size_t errlen) {
    bool role = !strcmp(arg, "--role"), layers = !strcmp(arg, "--layers");
    bool listen_arg = !strcmp(arg, "--listen"), coord = !strcmp(arg, "--coordinator");
    bool chunk = !strcmp(arg, "--dist-prefill-chunk"), window = !strcmp(arg, "--dist-prefill-window");
    if (!strcmp(arg, "--debug")) { opt->debug = true; return Q36_DIST_CLI_MATCHED; }
    if (!role && !layers && !listen_arg && !coord && !chunk && !window) return Q36_DIST_CLI_NOT_MATCHED;
    int needed = listen_arg || coord ? 2 : 1;
    if (*index + needed >= argc) goto invalid;
    const char *value = argv[++*index];
    uint32_t n;
    if (role) {
        if (opt->role) goto invalid;
        if (!strcmp(value, "coordinator")) opt->role = Q36_DISTRIBUTED_COORDINATOR;
        else if (!strcmp(value, "worker")) opt->role = Q36_DISTRIBUTED_WORKER;
        else goto invalid;
    } else if (layers) {
        const char *colon = strchr(value, ':');
        char start[16];
        if (!colon || colon == value || colon-value >= (int)sizeof(start) || opt->layers.set) goto invalid;
        memcpy(start, value, (size_t)(colon-value));
        start[colon-value] = 0;
        if (!dist_u32(start, &opt->layers.start)) goto invalid;
        opt->layers.has_output = !strcmp(colon + 1, "output");
        if (!opt->layers.has_output && (!dist_u32(colon + 1, &opt->layers.end) ||
            opt->layers.end < opt->layers.start)) goto invalid;
        opt->layers.set = true;
    } else if (listen_arg || coord) {
        if (!*value || !dist_u32(argv[++*index], &n) || !n || n > 65535) goto invalid;
        if (listen_arg) {
            if (opt->listen_host) goto invalid;
            opt->listen_host = value; opt->listen_port = (int)n;
        } else {
            if (opt->coordinator_host) goto invalid;
            opt->coordinator_host = value; opt->coordinator_port = (int)n;
        }
    } else {
        if (!dist_u32(value, &n) || !n || n > (window ? 64u : 8192u)) goto invalid;
        if (chunk) opt->prefill_chunk = n;
        else opt->prefill_window = n;
    }
    return Q36_DIST_CLI_MATCHED;
invalid:
    dist_error(err, errlen, "invalid or missing value for %s", arg);
    return Q36_DIST_CLI_ERROR;
}

int q36_dist_prepare_engine_options(const q36_dist_options *opt,
        q36_engine_options *engine, char *err, size_t errlen) {
    if (!opt) return 0;
    if (!q36_dist_enabled(opt)) {
        if (opt->layers.set || opt->listen_host || opt->coordinator_host ||
            opt->prefill_chunk || opt->prefill_window)
            return dist_error(err, errlen, "distributed options require --role");
        return 0;
    }
    if (!opt->layers.set || (opt->role != Q36_DISTRIBUTED_COORDINATOR && opt->role != Q36_DISTRIBUTED_WORKER) ||
        opt->prefill_window > 64 || opt->prefill_chunk > 8192)
        return dist_error(err, errlen, "pipeline role requires a valid --layers range");
    if (opt->role == Q36_DISTRIBUTED_COORDINATOR) {
        if (!opt->listen_host || opt->listen_port < 1 || opt->listen_port > 65535 ||
            opt->coordinator_host || opt->layers.start || opt->layers.has_output)
            return dist_error(err, errlen, "coordinator requires --layers 0:N --listen HOST PORT");
    } else if (!opt->coordinator_host || opt->coordinator_port < 1 || opt->coordinator_port > 65535 ||
               opt->listen_host || !opt->layers.start || !opt->layers.has_output ||
               opt->prefill_chunk || opt->prefill_window)
        return dist_error(err, errlen, "worker requires --layers N:output --coordinator HOST PORT");
    if (engine) engine->distributed = *opt;
    return 0;
}
