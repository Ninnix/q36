/* Real-model session checks. Run once without --role to record logits, then
 * repeat with a coordinator and --reference to compare the distributed route. */
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

static void compare(q36_session *s, FILE *fp, bool reference, const char *label) {
    int n;
    const float *logits = q36_session_logits(s, &n);
    assert(logits);
    if (!reference) { assert(fwrite(logits, sizeof(float), (size_t)n, fp) == (size_t)n); return; }
    float *want = malloc((size_t)n * sizeof(float));
    assert(want && fread(want, sizeof(float), (size_t)n, fp) == (size_t)n);
    double sum = 0;
    float max = 0;
    int best = 0;
    for (int i = 0; i < n; i++) {
        assert(isfinite(want[i]) && isfinite(logits[i]));
        float diff = fabsf(want[i] - logits[i]);
        sum += (double)diff * diff;
        if (diff > max) max = diff;
        if (want[i] > want[best]) best = i;
    }
    double rms = sqrt(sum / n);
    fprintf(stderr, "%s: top1=%d reference=%d rms=%.6g max=%.6g bit_exact=%s\n",
            label, q36_session_argmax(s), best, rms, max,
            memcmp(logits, want, (size_t)n * sizeof(float)) == 0 ? "yes" : "no");
    assert(q36_session_argmax(s) == best && rms < .005 && max < .05);
    free(want);
}

int main(int argc, char **argv) {
    q36_engine_options opt = {.backend=Q36_BACKEND_VULKAN, .context_size=512,
        .prefill_chunk=8, .session_count=2, .cache_type_k=Q36_KV_CACHE_Q8_0,
        .cache_type_v=Q36_KV_CACHE_Q4_0};
    const char *path = NULL;
    bool reference = false;
    for (int i = 1; i < argc; i++) {
        int rc = q36_dist_parse_cli_arg(argv[i], &i, argc, argv, &opt.distributed, error, sizeof(error));
        if (rc < 0) check(1);
        if (rc > 0) continue;
        if (!strcmp(argv[i], "--cpu")) opt.backend = Q36_BACKEND_CPU;
        else if (!strcmp(argv[i], "--ssd-streaming")) opt.ssd_streaming = true;
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) opt.model_path = argv[++i];
        else if ((!strcmp(argv[i], "--record") || !strcmp(argv[i], "--reference")) && i + 1 < argc) {
            reference = !strcmp(argv[i], "--reference"); path = argv[++i];
        } else { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    }
    assert(path && opt.model_path);
    check(q36_dist_prepare_engine_options(&opt.distributed, &opt, error, sizeof(error)));
    FILE *fp = fopen(path, reference ? "rb" : "wb");
    assert(fp);
    q36_engine *e = NULL;
    q36_session *a = NULL, *b = NULL;
    check(q36_engine_open(&e, &opt));
    check(q36_session_create(&a, e, 512));
    check(q36_session_create(&b, e, 512));
    q36_tokens first = {0}, second = {0};
    q36_encode_chat_prompt(e, "You are a helpful assistant.", "What is 2 + 2?", Q36_THINK_NONE, &first);
    q36_encode_chat_prompt(e, "You are a helpful assistant.", "Name the capital of Italy.", Q36_THINK_NONE, &second);
    check(q36_session_sync(a, &first, error, sizeof(error)));
    compare(a, fp, reference, "prefill");
    q36_session_snapshot snapshot = {0};
    check(q36_session_save_snapshot(a, &snapshot, error, sizeof(error)));
    FILE *payload = tmpfile();
    assert(payload);
    check(q36_session_save_payload(a, payload, error, sizeof(error)));
    long payload_bytes = ftell(payload);
    assert(payload_bytes > 0);
    check(q36_session_eval(a, 17, error, sizeof(error)));
    compare(a, fp, reference, "decode");
    check(q36_session_sync(b, &second, error, sizeof(error)));
    compare(b, fp, reference, "second session");
    check(q36_session_sync(a, &second, error, sizeof(error)));
    compare(a, fp, reference, "divergent prompt");
    check(q36_session_load_snapshot(a, &snapshot, error, sizeof(error)));
    compare(a, fp, reference, "snapshot replay");
    rewind(payload);
    check(q36_session_load_payload(a, payload, (uint64_t)payload_bytes, error, sizeof(error)));
    compare(a, fp, reference, "payload replay");
    q36_tokens_push(&first, 17);
    check(q36_session_sync(a, &first, error, sizeof(error)));
    compare(a, fp, reference, "prefix extension");
    q36_decode_item items[] = {{a, 478}, {b, 198}};
    check(q36_sessions_eval_batch(items, 2, error, sizeof(error)));
    compare(a, fp, reference, "ordered decode A");
    compare(b, fp, reference, "ordered decode B");
    if (reference) assert(fgetc(fp) == EOF);
    fclose(fp); fclose(payload);
    q36_session_snapshot_free(&snapshot);
    q36_tokens_free(&first); q36_tokens_free(&second);
    q36_session_free(a); q36_session_free(b);
    q36_engine_close(e);
    puts("distributed model tests: ok");
    return 0;
}
