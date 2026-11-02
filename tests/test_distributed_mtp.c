#include "q36.h"
#include "q36_distributed.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static char error[256];
static void check(int rc) {
    if (rc) { fprintf(stderr, "%s\n", error); exit(1); }
}
static int top(q36_session *s, int excluded) {
    int n, best = -1;
    const float *row = q36_session_logits(s, &n);
    assert(row);
    for (int i = 0; i < n; i++)
        if (i != excluded && (best < 0 || row[i] > row[best])) best = i;
    return best;
}
static void same(q36_session *a, q36_session *b) {
    int na, nb;
    const float *x = q36_session_logits(a, &na), *y = q36_session_logits(b, &nb);
    assert(x && y && na == nb && q36_session_pos(a) == q36_session_pos(b));
    float worst = 0;
    for (int i = 0; i < na; i++) {
        assert(isfinite(x[i]) && isfinite(y[i]));
        float d = fabsf(x[i] - y[i]);
        if (d > worst) worst = d;
    }
    if (worst > 1e-6f) fprintf(stderr, "pipeline MTP max logit difference=%g\n", worst);
    assert(worst <= 1e-6f);
}
static bool boundary(void *ud, int token) {
    (void)token;
    return ++*(int *)ud > 1;
}
int main(int argc, char **argv) {
    q36_engine_options opt = {.backend=Q36_BACKEND_VULKAN, .context_size=1024,
        .session_count=2, .prefill_chunk=32, .mtp_draft_tokens=4, .mtp_margin=0,
        .cache_type_k=Q36_KV_CACHE_Q8_0, .cache_type_v=Q36_KV_CACHE_Q4_0};
    for (int i = 1; i < argc; i++) {
        int rc = q36_dist_parse_cli_arg(argv[i], &i, argc, argv, &opt.distributed, error, sizeof(error));
        if (rc < 0) check(1);
        if (rc > 0) continue;
        if (!strcmp(argv[i], "--model") && i+1 < argc) opt.model_path = argv[++i];
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    assert(opt.model_path && opt.distributed.role == Q36_DISTRIBUTED_COORDINATOR);
    opt.mtp_path = opt.model_path;
    q36_engine *e = NULL;
    q36_session *spec = NULL, *plain = NULL;
    check(q36_engine_open(&e, &opt));
    assert(q36_engine_has_mtp(e));
    check(q36_session_create(&spec, e, 1024));
    check(q36_session_create(&plain, e, 1024));
    int eos = q36_token_eos(e), multi = 0, blocks = 0;
    const char *prompts[] = {
        "Explain why the sky looks blue and sunsets look red. Use plain language.",
        "Write a Python function to merge two sorted lists in linear time, with an example.",
        "A tank has 40 litres. A pump adds 8 litres per minute and a leak drains 3. When will it reach 100 litres? Check your reasoning."
    };
    for (unsigned c = 0; c < sizeof(prompts)/sizeof(*prompts); c++) {
        q36_tokens prompt = {0};
        q36_encode_chat_prompt(e, "You are a helpful assistant.", prompts[c], Q36_THINK_NONE, &prompt);
        for (int sampled = 0; sampled < 2; sampled++) {
            q36_session_invalidate(spec); q36_session_invalidate(plain);
            check(q36_session_sync(spec, &prompt, error, sizeof(error)));
            check(q36_session_sync(plain, &prompt, error, sizeof(error)));
            same(spec, plain);
            uint64_t rng = 1234567;
            for (int generated = 0; generated < 24;) {
                int accepted[16];
                int seed = q36_session_sample_excluding(spec, eos, sampled ? .7f : 0, 20, .8f, 0, &rng);
                int n = q36_session_eval_speculative_argmax_excluding(spec, seed, 24-generated,
                    eos, accepted, 16, error, sizeof(error));
                if (n < 1) check(1);
                assert(n <= 24-generated && accepted[0] == seed);
                for (int i = 0; i < n; i++) {
                    if (i) assert(accepted[i] == top(plain, eos));
                    check(q36_session_eval(plain, accepted[i], error, sizeof(error)));
                }
                same(spec, plain);
                generated += n; blocks++; multi += n > 1;
            }
            fprintf(stderr, "pipeline MTP case=%u sampled=%d: exact\n", c, sampled);
        }
        q36_tokens_free(&prompt);
    }
    /* Force a rejection at the second token, rather than relying on draft quality. */
    int draft[4] = {top(spec, eos), 0, 0, 0};
    check(q36_session_eval(plain, draft[0], error, sizeof(error)));
    draft[1] = (top(plain, eos) + 1) % q36_qwen35_n_vocab();
    int n = q36_session_verify_pipeline(spec, draft, 4, -1, eos, error, sizeof(error));
    if (n < 1) check(1);
    assert(n == 1); same(spec, plain);
    q36_session_snapshot snapshot = {0};
    check(q36_session_save_snapshot(spec, &snapshot, error, sizeof(error)));
    check(q36_session_eval(spec, 17, error, sizeof(error)));
    check(q36_session_load_snapshot(spec, &snapshot, error, sizeof(error)));
    check(q36_session_load_snapshot(plain, &snapshot, error, sizeof(error)));
    same(spec, plain);
    FILE *payload = tmpfile(); assert(payload);
    check(q36_session_save_payload(spec, payload, error, sizeof(error)));
    long bytes = ftell(payload); assert(bytes > 0);
    check(q36_session_eval(spec, 17, error, sizeof(error)));
    rewind(payload);
    check(q36_session_load_payload(spec, payload, (uint64_t)bytes, error, sizeof(error)));
    fclose(payload); same(spec, plain);
    int calls = 0, accepted[16], seed = top(spec, eos);
    n = q36_session_eval_speculative_argmax_with_boundary(spec, seed, 8, -1,
        accepted, 16, boundary, &calls, error, sizeof(error));
    assert(n == 1 && calls >= 2 && accepted[0] == seed);
    check(q36_session_eval(plain, seed, error, sizeof(error)));
    same(spec, plain);
    seed = top(spec, eos);
    n = q36_session_eval_speculative_argmax_excluding(spec, seed, 4, eos,
        accepted, 16, error, sizeof(error));
    if (n < 1) check(1);
    for (int i = 0; i < n; i++) {
        if (i) assert(accepted[i] == top(plain, eos));
        check(q36_session_eval(plain, accepted[i], error, sizeof(error)));
    }
    same(spec, plain);
    assert(multi > 0);
    fprintf(stderr, "pipeline MTP blocks=%d multi=%d; rejection, snapshot, payload, boundary: exact\n", blocks, multi);
    q36_session_snapshot_free(&snapshot);
    q36_session_free(plain); q36_session_free(spec); q36_engine_close(e);
    puts("distributed MTP tests: ok");
    return 0;
}
