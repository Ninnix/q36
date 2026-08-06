#!/bin/sh
set -eu

model=${Q36_TEST_MODEL:-gguf/Qwen3.6-35B-A3B-AntirezExperts-IQ2XXS-gateup-Q2K-down-Q8rest.gguf}
prompt=${Q36_RELEASE_BENCH_PROMPT:-tests/long_context_story_prompt.txt}
min_prefill=${Q36_RELEASE_MIN_PREFILL_TPS:-50}
min_decode=${Q36_RELEASE_MIN_DECODE_TPS:-40}
warm=$(mktemp /tmp/q36-release-bench-warm.XXXXXX)
run1=$(mktemp /tmp/q36-release-bench-1.XXXXXX)
run2=$(mktemp /tmp/q36-release-bench-2.XXXXXX)
run3=$(mktemp /tmp/q36-release-bench-3.XXXXXX)
trap 'rm -f "$warm" "$run1" "$run2" "$run3"' EXIT INT TERM

run_bench() {
    ./q36-bench -m "$model" --vulkan --prompt-file "$prompt" \
        --ctx-start 256 --ctx-max 256 --ctx-alloc 512 --gen-tokens 16 \
        --csv "$1"
}

run_bench "$warm"
run_bench "$run1"
run_bench "$run2"
run_bench "$run3"

awk -F, -v min_prefill="$min_prefill" -v min_decode="$min_decode" '
function median3(a, b, c, lo, hi) {
    lo = a < b ? (a < c ? a : c) : (b < c ? b : c)
    hi = a > b ? (a > c ? a : c) : (b > c ? b : c)
    return a + b + c - lo - hi
}
FNR == 2 {
    prefill[++found] = $3 + 0
    decode[found] = $5 + 0
}
END {
    if (found != 3) {
        print "release benchmark did not produce three data rows" > "/dev/stderr"
        exit 1
    }
    p = median3(prefill[1], prefill[2], prefill[3])
    d = median3(decode[1], decode[2], decode[3])
    printf "release benchmark median: prefill %.2f t/s, decode %.2f t/s\n", p, d
    if (p < min_prefill) {
        printf "prefill gate failed: %.2f < %.2f t/s\n", p, min_prefill > "/dev/stderr"
        exit 1
    }
    if (d < min_decode) {
        printf "decode gate failed: %.2f < %.2f t/s\n", d, min_decode > "/dev/stderr"
        exit 1
    }
}
' "$run1" "$run2" "$run3"
