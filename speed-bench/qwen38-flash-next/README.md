# Qwen3.8 Flash Next measurements

Recorded November 8–9, 2026, on AMD BC-250 boards with 16 CPU threads.
Weights: `Qwen3.8-Flash-Next-GSQ-RCO-Coder-22GB-down-00001-of-00002.gguf`,
with the original IQ4_NL n-gram shard. These are historical measurements;
the commands below use the shared tools for future comparisons.

## Correctness

The CPU reference used llama.cpp revision
`a691b2345844ce1b430201fc5fd404ce062e05f2`. The graph follows DS4 revision
`b3f50180ec2c7324892be1f50bc83555ccb8d5e7`; that DS4 loader did not accept
this quantized/pruned pack, so no DS4 runtime speed comparison is claimed.

Both short suites passed at all nine positions across Italian fact, code and
reasoning prompts. Gates require identical top-1, overlaps of at least 4/5,
15/20 and 40/64, and reference top-20 logit error at most 8.0.

| Comparison | Top-1 matches | Minimum overlap: top-5 / top-20 / top-64 | Maximum top-20 error |
| --- | ---: | --- | ---: |
| CPU / llama.cpp CPU | 9/9 | 4 / 18 / 57 | 1.09455 |
| CPU / Vulkan SSD streaming | 9/9 | 4 / 17 / 56 | 0.973371 |

Rechecked November 10 with llama.cpp
`351025ec532f825c7eb6c7ae1017d8e6d46d0cc9`, after the shared CPU arithmetic
fixes. Both comparisons pass all nine positions with the same prompts,
16 CPU threads and unchanged gates:

| Comparison | Top-1 matches | Minimum overlap: top-5 / top-20 / top-64 | Maximum top-20 error |
| --- | ---: | --- | ---: |
| CPU / llama.cpp CPU | 9/9 | 5 / 16 / 57 | 0.894385 |
| CPU / Vulkan SSD streaming | 9/9 | 4 / 15 / 56 | 1.14117 |

The November 8–9 distributed optimization also passed the shared CPU/Vulkan gates.
Five full-vocabulary frontier vectors matched the control byte for byte.
Nine long-session checks with a 2,083-token prefix matched exactly, including
partial-block decode, isolated sessions, divergent prompts, snapshots,
payload replay and prefix extension. Invalidation cleared exposed logits.

Use the [shared parity commands](../../README.md#qwen38-flash-next) and
[distributed reference/replay test](../../docs/DISTRIBUTED.md#validation).
`make test-qwen4 test-ssd-prefetch` covers CPU n-grams, quantization, sparse
selection and SSD queue ownership/cancellation. `tests/qwen4_kernels` covers
Vulkan arithmetic, batching, QSA ties/tails and cache growth. These replace
the individual development microbenchmarks and probe scripts.

## SSD streaming on one board

Sixteen generated tokens, original parity-correct implementation versus the
optimized streaming path:

| Prompt tokens | Prefill t/s before → after | Decode t/s before → after |
| ---: | ---: | ---: |
| 32 | 0.74 → 2.49 | 0.68 → 2.37 |
| 128 | 0.65 → 2.43 | 0.60 → 2.24 |

At 32 tokens, process peak RSS changed from 123.4 to 124.3 MiB; GPU allocation
peak stayed at 144.6 MiB, including the shared 16 MiB streaming slab.
Final temperatures were 43.6–47.8 C. OS page cache was uncontrolled; these
short runs do not establish cold-SSD throughput. Next-layer router hints
showed no consistent gain on either Flash Next or Qwen3.6 and remain disabled.
Hint-on/off logits matched exactly at all four checked positions per model.

## Resident distributed pipeline

The fixed 24/24 split keeps PLE on the main SSD and resident transformer
weights on both boards. No clock or CU detection selects the split.
Batch 32, allocation 8,192, eight greedy decode tokens, complete warm-up per
arm, sequential runs. The original fixture was a synthetic C audit prompt;
the existing story fixture in the command below gives a new workload, so its
absolute rates need not match this table.

| Context | New prefill tokens | Prefill t/s control → current | Decode t/s control → current |
| ---: | ---: | ---: | ---: |
| 128 | 128 | 128.63 → 160.01 | 29.75 → 30.09 |
| 2,048 | 1,920 | 160.36 → 160.40 | 22.88 → 23.18 |
| 3,968 | 1,920 | 130.97 → 133.40 | 20.52 → 22.96 |
| 5,888 | 1,920 | 127.93 → 132.42 | 18.62 → 22.83 |
| 7,808 | 1,920 | 123.64 → 130.06 | 16.17 → 22.71 |

Main GPU tensor peak stayed at 11,436,079,744 bytes. Worker high-water RSS
fell from 354.95 to 71.37 MiB. Main measured median RSS rose from 55.20 to
1,379.54 MiB; the final split was 26.18 MiB anonymous and 1,353.36 MiB
file-backed. Main high-water RSS was 5,085.56 / 6,125.71 MiB. Process RSS
and GPU allocation counters describe different memory and cannot be added.
Final main CPU temperature was 52.125–77.875 C; worker CPU 58.75–74.375 C
and GPU 55–75 C. Normal policies were unchanged: main GPU 1,000–1,850 MHz,
worker 1,500 MHz. Worker measured disk reads were zero in the final arm.

At 8K, decode improved 40.45% and prefill 5.19%. The 150/30 t/s targets
remain unmet at long contexts. Flash Next MTP is not implemented.

Start the [worker](../../docs/DISTRIBUTED.md#qwen38-flash-next) with
`--ctx 8192`, then run the existing benchmark on the coordinator:

```sh
MODEL=gguf/Qwen3.8-Flash-Next-GSQ-RCO-Coder-22GB-down-00001-of-00002.gguf
mkdir -p /tmp/flash-next-logits
Q36_THREADS=16 Q36_SSD_ROUTE_PREFETCH=0 Q36_VK_MEMORY_REPORT=1 \
  ./q36-bench -m "$MODEL" --vulkan -t 16 \
  --role coordinator --layers 0:23 --listen 10.36.0.1 9911 \
  --prompt-file tests/long_context_story_prompt.txt \
  --ctx-start 128 --ctx-max 7808 --step-incr 1920 --ctx-alloc 8192 \
  --prefill-chunk 32 --gen-tokens 8 --warmup \
  --dump-frontier-logits-dir /tmp/flash-next-logits
```

Replace addresses for your hosts. For a matched control, set
`Q36_VK_FLASH_QSA_CACHE=0`, `Q36_VK_FLASH_QSA_RADIX=0` and
`Q36_VK_FLASH_RESET_REUSE=0` on both stages; the worker helper accepts
repeated `--env Q36_NAME=value`. Restart both arms with identical settings
and compare dumped frontier vectors. Keep measurements and raw outputs
outside the source tree.
