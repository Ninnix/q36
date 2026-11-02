/* Exercise the actual framing, worker session lifecycle and prefill window
 * with a tiny recurrent stage, without loading a GGUF or requiring a GPU. */
#include "../q36_distributed.c"
#include <assert.h>
#include <math.h>
#include <signal.h>
#include <sys/wait.h>

struct q36_engine { q36_dist_engine *dist; };
struct q36_session {
    q36_engine *engine;
    uint32_t pos, ctx;
    float state;
    int progress;
    bool cancel, spec;
    uint32_t spec_pos, spec_n, mtp_pos;
    float history[Q36_DIST_VERIFY_MAX], rows[64];
    float candidate_logits[3];
    unsigned projections;
};

q36_dist_engine *q36_engine_distributed(q36_engine *e) { return e->dist; }
bool q36_session_distributed_cancelled(q36_session *s) { return s->cancel; }
void q36_session_distributed_progress(q36_session *s, int current, int total) {
    assert(current <= total);
    s->progress = current;
}
int q36_session_create(q36_session **out, q36_engine *e, int ctx) {
    *out = calloc(1, sizeof(**out));
    assert(*out);
    (*out)->engine = e;
    (*out)->ctx = (uint32_t)ctx;
    return 0;
}
void q36_session_free(q36_session *s) { free(s); }
int q36_session_layer_slice_reset(q36_session *s, char *err, size_t errlen) {
    (void)err; (void)errlen;
    s->pos = 0;
    s->state = 0;
    s->spec = false;
    s->mtp_pos = 0;
    return 0;
}
int q36_session_eval_layer_slice(q36_session *s, const int *tokens,
        uint32_t n, uint32_t pos, uint32_t first, uint32_t last,
        const float *input, float *output, bool output_logits,
        float *logits, char *err, size_t errlen) {
    (void)last;
    assert(!output_logits);
    if (pos != s->pos || !n || pos + n > s->ctx)
        return dist_error(err, errlen, "fake stage position mismatch");
    for (uint32_t i = 0; i < n; i++) {
        if (tokens[i] < 0) return dist_error(err, errlen, "invalid token");
        float value = first ? input[i * 2] : (float)tokens[i];
        s->state += value;
        s->rows[2u*i] = s->state;
        s->rows[2u*i + 1u] = -s->state;
        if (s->spec) {
            assert(s->spec_n < Q36_DIST_VERIFY_MAX);
            s->history[s->spec_n++] = s->state;
        }
        if (output) {
            output[i * 2] = s->state;
            output[i * 2 + 1] = -s->state;
        }
    }
    if (output_logits) {
        logits[0] = s->state;
        logits[1] = -s->state;
        logits[2] = (float)(pos + n);
    }
    s->pos += n;
    return 0;
}

int q36_session_read_final_hidden(q36_session *s, uint32_t n,
        float *hidden, char *err, size_t errlen) {
    (void)n; (void)err; (void)errlen;
    hidden[0] = s->state;
    hidden[1] = -s->state;
    return 0;
}
int q36_session_project_logits(q36_session *s, const float *hidden,
        float *logits, char *err, size_t errlen) {
    (void)s; (void)err; (void)errlen;
    logits[0] = hidden[0];
    logits[1] = hidden[1];
    logits[2] = fabsf(hidden[0]) * .001f;
    return 0;
}

int q36_session_pipeline_project(q36_session *s, const float *hidden,
        int excluded, int *top, char *err, size_t len) {
    float *logits = s->candidate_logits;
    int rc = q36_session_project_logits(s, hidden, logits, err, len);
    *top = -1;
    for (int i = 0; i < 3; i++)
        if (i != excluded && (*top < 0 || logits[i] > logits[*top])) *top = i;
    s->projections++;
    return rc;
}
int q36_session_pipeline_read_logits(q36_session *s, float *logits,
        char *err, size_t len) {
    (void)err; (void)len;
    memcpy(logits, s->candidate_logits, 3 * sizeof(float));
    return 0;
}

int q36_session_read_final_rows(q36_session *s, uint32_t n, float *hidden, char *err, size_t len) {
    (void)err; (void)len;
    assert(n <= 32);
    memcpy(hidden, s->rows, n * 2u * sizeof(float));
    return 0;
}
int q36_session_pipeline_mtp_catchup(q36_session *s, const int *tokens,
        uint32_t n, uint32_t pos, const float *hidden, char *err, size_t len) {
    (void)tokens; (void)hidden;
    if (pos != s->mtp_pos) return dist_error(err, len, "fake MTP history mismatch");
    s->mtp_pos += n;
    return 0;
}
int q36_session_pipeline_spec_begin(q36_session *s, char *err, size_t len) {
    if (s->spec) return dist_error(err, len, "nested fake transaction");
    s->spec = true;
    s->spec_pos = s->pos;
    s->spec_n = 0;
    return 0;
}
int q36_session_pipeline_spec_commit(q36_session *s, uint32_t count, char *err, size_t len) {
    if (!s->spec || !count || count > s->spec_n) return dist_error(err, len, "bad fake commit");
    s->state = s->history[count - 1u];
    s->pos = s->spec_pos + count;
    s->spec = false;
    return 0;
}

static void *worker_main(void *arg) {
    q36_dist_engine *d = arg;
    char err[256];
    dist_worker_loop(d, d->fd, err, sizeof(err));
    close(d->fd);
    return NULL;
}

static int parse(int argc, char **argv, q36_dist_options *opt) {
    char err[256];
    for (int i = 1; i < argc; i++) {
        int rc = q36_dist_parse_cli_arg(argv[i], &i, argc, argv, opt, err, sizeof(err));
        if (rc != Q36_DIST_CLI_MATCHED) return 1;
    }
    return q36_dist_prepare_engine_options(opt, NULL, err, sizeof(err));
}

static void test_options(void) {
    q36_dist_options opt = {0};
    char *good[] = {"q36", "--role", "coordinator", "--layers", "0:19", "--listen", "127.0.0.1", "9911",
                    "--dist-prefill-window", "3", "--dist-prefill-chunk", "8"};
    assert(!parse(12, good, &opt));
    assert(opt.layers.start == 0 && opt.layers.end == 19 && opt.prefill_window == 3);
    char *worker[] = {"q36", "--role", "worker", "--layers", "20:output", "--coordinator", "::1", "9911"};
    memset(&opt, 0, sizeof(opt));
    assert(!parse(8, worker, &opt) && opt.layers.has_output);
    const char *bad[][4] = {
        {"q36", "--layers", "-1:20", NULL}, {"q36", "--layers", "3:2", NULL},
        {"q36", "--layers", "0:", NULL}, {"q36", "--layers", "0:4294967296", NULL},
        {"q36", "--role", "peer", NULL}, {"q36", "--role", NULL, NULL},
        {"q36", "--dist-prefill-window", "65", NULL}, {"q36", "--dist-prefill-chunk", "0", NULL},
        {"q36", "--layers", "0:1", NULL}, {"q36", "--listen", "127.0.0.1", "70000"},
    };
    for (size_t j = 0; j < sizeof(bad)/sizeof(*bad); j++) {
        int argc = 0;
        while (argc < 4 && bad[j][argc]) argc++;
        memset(&opt, 0, sizeof(opt));
        assert(parse(argc, (char **)bad[j], &opt));
    }
}

static void test_pipeline(uint32_t window) {
    int fds[2];
    assert(!socketpair(AF_UNIX, SOCK_STREAM, 0, fds));
    q36_dist_engine coord = {.fd=fds[0], .listener=-1, .next_id=1, .generation=1,
        .info={.n_layers=4, .n_embd=2, .n_vocab=3, .prefill_cap=3, .layer_start=0, .layer_end=1},
        .peer={.prefill_cap=2, .context_size=32}, .opt={.prefill_window=window}};
    assert(!pthread_mutex_init(&coord.mutex, NULL));
    q36_engine engine = {0};
    q36_dist_engine worker = {.fd=fds[1], .engine=&engine,
        .info={.n_embd=2, .n_vocab=3, .prefill_cap=2, .context_size=32, .layer_start=2, .layer_end=3}};
    engine.dist = &worker;
    pthread_t thread;
    assert(!pthread_create(&thread, NULL, worker_main, &worker));
    q36_dist_session *a, *b;
    assert(!q36_dist_session_create(&a, &coord, 32));
    assert(!q36_dist_session_create(&b, &coord, 32));
    q36_session owner = {.ctx=32}, other = {.ctx=32};
    int tokens[] = {1, 2, 3, 4, 5, 6, 7};
    q36_tokens empty = {0}, prompt = {.v=tokens, .len=7};
    float logits[3];
    char err[256];
    assert(!q36_dist_session_sync(a, &owner, &empty, &prompt, logits, err, sizeof(err)));
    assert(logits[0] == 84 && logits[1] == -84 && owner.pos == 7 && owner.progress == 7);
    assert(!q36_dist_session_eval(a, &owner, &prompt, 8, logits, err, sizeof(err)));
    assert(logits[0] == 120 && owner.pos == 8 && owner.progress == 7);
    assert(!q36_dist_session_sync(b, &other, &empty, &prompt, logits, err, sizeof(err)));
    assert(logits[0] == 84);
    /* A divergent timeline must clear both stages' recurrent history. */
    prompt.len = 3;
    assert(!q36_dist_session_sync(a, &owner, &empty, &prompt, logits, err, sizeof(err)));
    assert(logits[0] == 10 && owner.pos == 3);
    prompt.len = 5;
    q36_tokens prefix = {.v=tokens, .len=3};
    assert(!q36_dist_session_sync(a, &owner, &prefix, &prompt, logits, err, sizeof(err)));
    assert(logits[0] == 35);
    /* Invalid work tears down the route; a partial local stage is never committed. */
    assert(q36_dist_session_eval(a, &owner, &prompt, -1, logits, err, sizeof(err)));
    assert(coord.fd == -1);
    pthread_join(thread, NULL);
    /* A new worker has no KV state: recreate each session and replay its tokens. */
    assert(!socketpair(AF_UNIX, SOCK_STREAM, 0, fds));
    coord.fd = fds[0];
    coord.generation++;
    worker.fd = fds[1];
    assert(!pthread_create(&thread, NULL, worker_main, &worker));
    assert(!q36_dist_session_eval(a, &owner, &prompt, 6, logits, err, sizeof(err)));
    assert(logits[0] == 56 && owner.pos == 6);
    assert(!q36_dist_session_sync(b, &other, &prompt, &prompt, logits, err, sizeof(err)));
    assert(logits[0] == 35);
    q36_dist_session_free(b);
    owner.cancel = true;
    assert(q36_dist_session_eval(a, &owner, &prompt, 7, logits, err, sizeof(err)) == Q36_SESSION_SYNC_INTERRUPTED);
    assert(coord.fd == -1);
    q36_dist_session_free(a);
    pthread_join(thread, NULL);
    pthread_mutex_destroy(&coord.mutex);
}

static void test_speculation(uint32_t window) {
    int fds[2];
    assert(!socketpair(AF_UNIX, SOCK_STREAM, 0, fds));
    q36_dist_engine coord = {.fd=fds[0], .listener=-1, .next_id=1, .generation=1,
        .info={.n_layers=4, .n_embd=2, .n_vocab=3, .prefill_cap=3, .mtp=true,
               .speculation=true, .layer_start=0, .layer_end=1},
        .peer={.prefill_cap=2, .context_size=32, .speculation=true}, .opt={.prefill_window=window}};
    assert(!pthread_mutex_init(&coord.mutex, NULL));
    q36_engine engine = {0};
    q36_dist_engine worker = {.fd=fds[1], .engine=&engine, .peer={.mtp=true},
        .info={.n_embd=2, .n_vocab=3, .prefill_cap=2, .context_size=32,
               .speculation=true, .layer_start=2, .layer_end=3}};
    engine.dist = &worker;
    pthread_t thread;
    assert(!pthread_create(&thread, NULL, worker_main, &worker));
    q36_dist_session *session;
    assert(!q36_dist_session_create(&session, &coord, 32));
    q36_session owner = {.ctx=32};
    int history[32] = {1, 2, 3};
    q36_tokens empty = {0}, checkpoint = {.v=history, .len=3};
    float logits[3]; char err[256];
    assert(!q36_dist_session_sync(session, &owner, &empty, &checkpoint, logits, err, sizeof(err)));
    assert(owner.mtp_pos == 3);
    int cases[][3] = {{1,0,0}, {1,0,2}, {1,2,2}, {1,0,0}, {1,2,2}};
    int accepted[] = {3,2,1,2,3};
    for (unsigned c = 0; c < sizeof(cases)/sizeof(*cases); c++) {
        unsigned projected = owner.projections;
        int count = q36_dist_session_speculate(session, &owner, &checkpoint,
            cases[c], 3, c == 3 ? 0 : -1, c == 4 ? 0 : -1, logits, err, sizeof(err));
        assert(count == accepted[c]);
        assert(owner.projections - projected == (unsigned)count);
        for (int i = 0; i < count; i++) history[checkpoint.len++] = cases[c][i];
        assert(owner.pos == (uint32_t)checkpoint.len && owner.mtp_pos == owner.pos);
        float prefix = 0, suffix = 0;
        for (int i = 0; i < checkpoint.len; i++) { prefix += history[i]; suffix += prefix; }
        assert(owner.state == prefix && logits[0] == suffix && logits[1] == -suffix);
    }
    assert(!q36_dist_session_eval(session, &owner, &checkpoint, 2, logits, err, sizeof(err)));
    history[checkpoint.len++] = 2;
    owner.cancel = true;
    assert(q36_dist_session_speculate(session, &owner, &checkpoint,
        cases[0], 3, -1, -1, logits, err, sizeof(err)) < 0 && coord.fd == -1);
    q36_dist_session_free(session);
    pthread_join(thread, NULL);
    pthread_mutex_destroy(&coord.mutex);
}

typedef struct { int fd; uint8_t *data; uint32_t bytes; } duplex_job;
static void *duplex_peer(void *opaque) {
    duplex_job *j = opaque;
    assert(!dist_send(j->fd, DIST_RESULT, j->data, j->bytes, NULL));
    uint32_t type, bytes; uint8_t *data; char err[128];
    assert(!dist_recv(j->fd, &type, &data, &bytes, NULL, err, sizeof(err)));
    assert(type == DIST_WORK && bytes == j->bytes && !memcmp(data, j->data, bytes));
    free(data);
    return NULL;
}
static void test_duplex(void) {
    int fds[2], small = 1024;
    assert(!socketpair(AF_UNIX, SOCK_STREAM, 0, fds));
    for (int i = 0; i < 2; i++) {
        assert(!setsockopt(fds[i], SOL_SOCKET, SO_SNDBUF, &small, sizeof(small)));
        assert(!setsockopt(fds[i], SOL_SOCKET, SO_RCVBUF, &small, sizeof(small)));
    }
    uint32_t bytes = 256u * 1024u;
    uint8_t *payload = malloc(bytes); assert(payload);
    for (uint32_t i = 0; i < bytes; i++) payload[i] = (uint8_t)(i * 13u);
    duplex_job job = {fds[1], payload, bytes};
    dist_write send = {.fd=fds[0], .data=payload, .bytes=bytes};
    pthread_t peer, writer;
    assert(!pthread_create(&peer, NULL, duplex_peer, &job));
    assert(!pthread_create(&writer, NULL, dist_writer, &send));
    uint32_t type, received; uint8_t *data; char err[128];
    assert(!dist_recv(fds[0], &type, &data, &received, NULL, err, sizeof(err)));
    assert(type == DIST_RESULT && received == bytes && !memcmp(data, payload, bytes));
    free(data);
    pthread_join(writer, NULL); pthread_join(peer, NULL);
    assert(!send.result);
    close(fds[0]); close(fds[1]);
    assert(!socketpair(AF_UNIX, SOCK_STREAM, 0, fds));
    send.fd = fds[0]; send.result = 0;
    assert(!pthread_create(&writer, NULL, dist_writer, &send));
    shutdown(fds[0], SHUT_RDWR);
    pthread_join(writer, NULL);
    assert(send.result);
    close(fds[0]); close(fds[1]); free(payload);
}

static void test_frames(void) {
    int fds[2];
    assert(!socketpair(AF_UNIX, SOCK_STREAM, 0, fds));
    uint32_t h[] = {htonl(DIST_MAGIC), htonl(DIST_WORK), htonl(DIST_MAX_FRAME + 1u)};
    assert(!dist_io(fds[0], h, sizeof(h), true, NULL));
    uint32_t type, bytes;
    uint8_t *data;
    char err[256];
    assert(dist_recv(fds[1], &type, &data, &bytes, NULL, err, sizeof(err)) && !data);
    close(fds[0]); close(fds[1]);
    assert(!socketpair(AF_UNIX, SOCK_STREAM, 0, fds));
    close(fds[0]);
    assert(dist_recv(fds[1], &type, &data, &bytes, NULL, err, sizeof(err)) == DIST_TRANSPORT_ERROR && !data);
    close(fds[1]);
    uint8_t word[4];
    dist_put(word, UINT32_C(0xfedcba98));
    assert(word[0] == 0xfe && word[3] == 0x98 && dist_get(word) == UINT32_C(0xfedcba98));
}

typedef struct { q36_dist_engine *engine; int fd, result; } hello_job;

static void *hello_main(void *opaque) {
    hello_job *job = opaque;
    char err[256];
    job->result = dist_hello(job->engine, job->fd, NULL, err, sizeof(err));
    return NULL;
}

static void test_hello(void) {
    for (int variant = 0; variant < 7; variant++) {
        int fds[2];
        assert(!socketpair(AF_UNIX, SOCK_STREAM, 0, fds));
        q36_dist_engine coord = {.opt={.role=Q36_DISTRIBUTED_COORDINATOR},
            .info={.n_layers=4, .n_embd=2, .n_vocab=3, .prefill_cap=8,
                   .context_size=32, .layer_start=0, .layer_end=1, .fingerprint=1234}};
        q36_dist_engine worker = coord;
        worker.opt.role = Q36_DISTRIBUTED_WORKER;
        worker.info.layer_start = 2;
        worker.info.layer_end = 3;
        if (variant == 1) worker.info.fingerprint++;
        if (variant == 2) worker.info.layer_start = 3;
        if (variant == 3) worker.info.n_vocab++;
        if (variant >= 4) {
            coord.info.mtp = true;
            coord.info.speculation = variant != 6;
            worker.info.speculation = variant != 4;
        }
        hello_job job = {&worker, fds[1], -1};
        pthread_t thread;
        assert(!pthread_create(&thread, NULL, hello_main, &job));
        char err[256];
        int result = dist_hello(&coord, fds[0], NULL, err, sizeof(err));
        pthread_join(thread, NULL);
        bool bad = variant != 0 && variant != 5;
        assert((result != 0) == bad);
        assert((job.result != 0) == bad);
        close(fds[0]); close(fds[1]);
    }
}

/* Full-logit and pre-speculation protocols must both be rejected. */
static void test_legacy_peer(void) {
    for (uint32_t version = 1; version <= 2; version++) {
        int fds[2];
        assert(!socketpair(AF_UNIX, SOCK_STREAM, 0, fds));
        q36_dist_engine coord = {.opt={.role=Q36_DISTRIBUTED_COORDINATOR},
            .info={.n_layers=4, .n_embd=2, .n_vocab=3, .prefill_cap=8,
                   .context_size=32, .layer_start=0, .layer_end=1, .fingerprint=1234}};
        uint32_t words[] = {version, 4, 2, 3, 2, 3, 32, 8, 0, 1234};
        uint8_t hello[sizeof(words)];
        for (size_t i = 0; i < sizeof(words)/sizeof(*words); i++) dist_put(hello + 4*i, words[i]);
        assert(!dist_send(fds[1], DIST_HELLO, hello, sizeof(hello), NULL));
        char err[256];
        assert(dist_hello(&coord, fds[0], NULL, err, sizeof(err)));
        close(fds[0]); close(fds[1]);
    }
}

static void test_tcp_endpoint(const char *bind_host, const char *connect_host) {
    int listener = dist_endpoint(bind_host, 0, true);
    if (listener < 0 && !strcmp(bind_host, "::1") &&
        (errno == EAFNOSUPPORT || errno == EPROTONOSUPPORT || errno == EADDRNOTAVAIL)) {
        puts("distributed tests: IPv6 loopback unavailable, skipped");
        return;
    }
    assert(listener >= 0 && (fcntl(listener, F_GETFD) & FD_CLOEXEC));
    struct sockaddr_storage address;
    socklen_t size = sizeof(address);
    assert(!getsockname(listener, (struct sockaddr *)&address, &size));
    int port = address.ss_family == AF_INET6 ?
        ntohs(((struct sockaddr_in6 *)&address)->sin6_port) :
        ntohs(((struct sockaddr_in *)&address)->sin_port);
    int client = dist_endpoint(connect_host, port, false);
    assert(client >= 0 && (fcntl(client, F_GETFD) & FD_CLOEXEC));
    int peer = accept(listener, NULL, NULL);
    assert(peer >= 0);
    dist_socket_options(peer);
    assert(fcntl(peer, F_GETFD) & FD_CLOEXEC);

    char payload[] = "TCP pipeline";
    assert(!dist_send(client, DIST_WORK, payload, sizeof(payload), NULL));
    uint32_t type, bytes;
    uint8_t *data;
    char err[256];
    assert(!dist_recv(peer, &type, &data, &bytes, NULL, err, sizeof(err)));
    assert(type == DIST_WORK && bytes == sizeof(payload) && !memcmp(data, payload, bytes));
    free(data);
    close(client); close(peer); close(listener);
}

/* An accepting proxy with no upstream must not cause a hot reconnect loop. */
static void test_retry_pacing(void) {
    int listener = dist_endpoint("127.0.0.1", 0, true);
    assert(listener >= 0);
    struct sockaddr_in address;
    socklen_t size = sizeof(address);
    assert(!getsockname(listener, (struct sockaddr *)&address, &size));
    pid_t child = fork();
    assert(child >= 0);
    if (!child) {
        close(listener);
        q36_dist_engine worker = {.opt={.role=Q36_DISTRIBUTED_WORKER,
            .coordinator_host="127.0.0.1", .coordinator_port=ntohs(address.sin_port)},
            .info={.n_layers=4, .n_embd=2, .n_vocab=3, .context_size=32,
                   .prefill_cap=2, .layer_start=2, .layer_end=3}};
        q36_engine engine = {.dist=&worker};
        _exit(q36_dist_run(&engine));
    }
    struct pollfd p = {listener, POLLIN, 0};
    bool first = poll(&p, 1, 5000) > 0, premature = false, retried = false;
    if (first) {
        int peer = accept(listener, NULL, NULL);
        assert(peer >= 0);
        uint32_t type, bytes;
        uint8_t *data;
        char err[256];
        int rc = dist_recv(peer, &type, &data, &bytes, NULL, err, sizeof(err));
        free(data);
        close(peer);
        if (!rc && type == DIST_HELLO) {
            premature = poll(&p, 1, 100) > 0;
            if (!premature) retried = poll(&p, 1, 5000) > 0;
        }
    }
    kill(child, SIGTERM);
    int status;
    assert(waitpid(child, &status, 0) == child);
    close(listener);
    assert(first && !premature && retried && WIFSIGNALED(status));
}

int main(void) {
    test_options();
    test_tcp_endpoint("127.0.0.1", "127.0.0.1");
    test_tcp_endpoint("127.0.0.1", "localhost");
    test_tcp_endpoint("0.0.0.0", "127.0.0.1");
    test_tcp_endpoint("::1", "::1");
    test_frames();
    test_duplex();
    test_hello();
    test_legacy_peer();
    test_pipeline(1);
    test_pipeline(3);
    test_speculation(1);
    test_speculation(2);
    test_speculation(3);
    test_retry_pacing();
    puts("distributed tests: ok");
    return 0;
}
