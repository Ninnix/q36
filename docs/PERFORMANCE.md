# Performance and Benchmarking

[README](../README.md#speed)

## Context sweeps

Compare the same model, quantization, context, sampling settings, and power
state. Record the commit and whether weights were resident, streamed, or
distributed. Keep other GPU workloads idle and repeat comparisons.

`q36-bench` measures prefill and generation at successive context frontiers:

```sh
./q36-bench --prompt-file tests/long_context_story_prompt.txt
```

Each prefill number measures the newly added token interval. Generation uses
128 greedy non-EOS tokens, then restores the prefix before continuing.
Defaults cover 2K through 32K context in 2K steps. Output is CSV.
`--warmup` runs an untimed prefill and eight decode tokens at every frontier.
Keep cold startup, prefix replay, and resident inference timings separate.

Apple Silicon commands and results are in [Metal](METAL.md#recorded-measurements).

## BC-250 context sweeps

These recorded BC-250 sweeps use resident models, Q8_0 keys / Q4_0 values,
and 128 greedy generation tokens per frontier. Prefill measures the newly
added interval. MTP uses the embedded head with draft limit eight and
adaptive confidence (`--mtp-margin 0`). All rates are in tokens per second.

### Qwen3.6-35B-A3B

`gguf/Qwen3.6-35B-A3B-IQ2XXS-Q2K-Q8.gguf`

| Context | Prefill | Decode | MTP decode |
| ---: | ---: | ---: | ---: |
| 2,048 | 634.96 | 90.63 | 151.22 |
| 4,096 | 634.74 | 91.21 | 131.39 |
| 8,192 | 510.59 | 89.37 | 127.17 |
| 16,384 | 371.32 | 83.78 | 143.14 |
| 32,768 | 236.85 | 74.43 | 77.96 |
| 65,536 | 137.01 | 64.72 | 83.00 |

![BC-250 Qwen3.6-35B-A3B prefill, decode, and MTP decode](../speed-bench/bc250-20261025/qwen36_ts.svg)

### Qwen3.8-27B

`gguf/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`

| Context | Prefill | Decode | MTP decode |
| ---: | ---: | ---: | ---: |
| 2,048 | 207.64 | 27.07 | 36.44 |
| 4,096 | 194.88 | 26.94 | 40.31 |
| 8,192 | 169.60 | 26.22 | 34.53 |
| 16,384 | 135.21 | 25.11 | 37.49 |
| 32,768 | 95.43 | 22.06 | 30.61 |
| 65,536 | 59.48 | 20.18 | 25.32 |

![BC-250 Qwen3.8-27B prefill, decode, and MTP decode](../speed-bench/bc250-20261025/qwen38-xxs_ts.svg)

### Qwen3.8-27B (IQ3_S)

`gguf/Qwen3.8-27B-UD-IQ3_S.gguf`

| Context | Prefill | Decode | MTP decode |
| ---: | ---: | ---: | ---: |
| 2,048 | 212.70 | 25.61 | 39.47 |
| 4,096 | 199.71 | 25.45 | 47.52 |
| 8,192 | 173.53 | 24.80 | 45.21 |
| 16,384 | 137.72 | 23.82 | 34.88 |
| 32,768 | 96.68 | 21.48 | 29.32 |
| 65,536 | 59.46 | 19.28 | 26.01 |

![BC-250 Qwen3.8-27B IQ3_S prefill, decode, and MTP decode](../speed-bench/bc250-20261025/qwen38-iq3s_ts.svg)

Recorded on 2026-10-25. The charts are in
[the sweep directory](../speed-bench/bc250-20261025). These are recorded
baselines, not a fresh benchmark of every commit.

## Earlier measurements

[October 24 sweeps](../speed-bench/bc250-20261024/README.md) retain the previous
resident measurements and reproduction commands. The [historical BC-250 chart](../speed-bench/bc250_ts.svg)
includes the earlier Qwen3.6 resident and SSD-streaming runs.

The published BC-250 measurements use a 40-CU unlock, a 500-2000 MHz governor
range, and an 85 C thermal ceiling. A stock 24-CU board will be slower.

## MTP measurements

### Qwen3.6 MoE

BC-250 resident Vulkan measurements with the embedded head and Q8_0 K / Q4_0 V
cover prose, code, reasoning with thinking enabled, and structured output.
The table gives MTP speedups over ordinary decode in the same build:

| Workload | Greedy speedup | Sampled speedup |
| --- | ---: | ---: |
| Prose | 1.17–1.28× | 1.20–1.24× |
| Code | 1.21–1.99× | 1.48–1.58× |
| Reasoning | 1.40–1.50× | 1.51–1.60× |
| JSON | 1.39–1.50× | 1.52–1.62× |

These are serialized, warmed sweeps at 128, 2K, and 4K context with 192
generated tokens, draft ceiling eight, and margin zero. All 24 greedy cases
produce identical token sequences with MTP enabled or disabled. The 12
sampled cases use temperature 0.6 and seed 42; opportunistic MTP samples each
block boundary and uses a verified greedy tail, so its sampling distribution
differs from ordinary per-token sampling.

## What to compare next

- For SSD streaming, record the effective cache and separate cold startup from
  warm-cache inference.
- For multiple sessions, report individual latency and aggregate throughput.
- For MTP, compare ordinary decode with the same prompt and context.
- For distributed inference, keep the model and prompt equal. Resident weights
  on two boards versus streamed weights on one measures capacity as well as
  parallel execution.
