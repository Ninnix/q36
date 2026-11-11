# Qwen3.8-27B Dynamic Q6_K pipeline

Measured November 11, 2026, on two AMD BC-250 boards. Model:
`unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf`, 21,983,677,344 bytes.
SHA-256: `c9c206812fbe4ac7b76a729e25928b63f2ae89d37f69da7a71c20aec763cd436`.

## Correctness

The CPU path uses each dense FFN projection's activation format. The Dynamic
Q6_K pack has an IQ4_NL gate beside a Q5_K up projection in layer 11; reusing
Q8_0 gate activations for the Q8_K up projection previously produced non-finite
outputs. Scalar decode and batched prefill now handle this combination.
`make test-qwen4` covers both directions, a float/quantized pair, and a
same-format control without loading a model.

Developer CPU parity checks used 16 threads and unchanged parity gates.

| Comparison | Positions / top-1 matches | Minimum top-5 / top-20 / top-64 overlap | Maximum top-20 error |
| --- | ---: | --- | ---: |
| CPU / external developer reference | 4 / 4 | 4 / 19 / 61 | 0.351967 |
| CPU / distributed Vulkan | 9 / 9 | 4 / 18 / 56 | 1.11089 |

The developer reference check covers the complete Italian fact prompt and
three decode steps. The remaining cases were stopped after this case; the
model exceeds physical RAM and CPU inference repeatedly pages its weights.
The distributed check covers fact, code, and reasoning prompts, including
prefill and decode. CPU results were captured before opening the coordinator.

Distributed MTP passed all six prose/code/reasoning arms with greedy and
sampled block boundaries. Of 64 speculative blocks, 43 accepted multiple
tokens. Accepted tokens matched ordinary target decoding; full-vocabulary
logits differed by at most `1e-6`. Forced rejection, snapshot replay, payload
replay, and generation boundaries also passed.

The plain session test recorded a 128-token pipeline run, then replayed it
with both the chunk size and window reduced to one. All nine complete
logit vectors matched byte for byte: prefill, decode, isolated sessions,
divergent prompts, snapshots, payload replay, prefix extension, and ordered
decode on two sessions.

`make -j4 test` passed, including transport, quantization, sampling, agent,
server, n-gram, SSD prefetch, and worker catalogue tests. The new staging
catalogue was also checked against the existing Qwen3.6 Q2/Q4, Qwen3.8 IQ3_S,
Qwen3.8 IQ3_XXS with MTP, and Flash Next model files. The final
`./q36_test --vulkan-kernels` regression also passed.

## Setup

The coordinator owns layers `0:33`, the output projection, and the optional
MTP block. The worker owns `34:output`. Resident target weights are 10.483 GiB
and 8.682 GiB; MTP adds 0.327 GiB on the coordinator. Embeddings remain mmap
backed on the coordinator. The worker stages its slice directly into Vulkan,
using a sparse catalogue and fingerprint samples rather than a second GGUF
copy on USB storage.

The worker has an 8 GiB VRAM / 8 GiB system-memory split. Its weight arena uses
the existing host-visible fallback when VRAM fills. Initial staging transferred
9,322,327,296 bytes over the LAN in about 80 seconds. Inference uses an SSH
reverse tunnel because the coordinator's inbound LAN port is blocked. The
normal CPU/GPU clock and thermal policies were left in place. The coordinator
uses an ACE compute queue; the worker driver exposes a graphics queue for
compute. Both boards run native builds with identical prefill shader hashes.

See [launch and parity commands](../../docs/DISTRIBUTED.md#qwen38-27b-q6_k).
Raw logs and CPU captures are under the ignored local directory
`tests/test-vectors/local/q38-q6-20261111/` and
`tests/test-vectors/local/q38-q6-speedup-20261111/` on the test coordinator.
The latest optimization run is in
`tests/test-vectors/local/q38-q6-next-20261111/`.

## Pipeline arithmetic

The previous fast batched path changed both projection rounding and the
FP16 recurrent-state rounding interval. Those differences accumulated until
4K logits failed parity. Prefill now batches native decode arithmetic:
Q5_K/Q6_K/IQ4_NL tiles share weight loads across tokens. Q8_0 now reuses
the verification tile across four output rows and eight tokens while retaining
its native FP32 reduction. Paired Q5_K/Q6_K token tiles share weights through
cache; Q5_K widens once per dot while keeping every i16 partial in range.
Recurrent state rounds after each token. The fused
convolution/gating front processes ordered rows in one dispatch, reusing
existing scratch so future QKV inputs survive output aliasing. Incomplete
tiles and other formats reuse the existing verification kernels.

This profile is selected only for dense models marked Q6_K in GGUF. Plain
decode, MTP verification, and other quantizations keep their existing kernel
selection. No additional dependency or repacked weight copy is required.
The default distributed prefill capacity is 64 tokens. Complete vocabulary
logits match the saved token-by-token reference exactly at 128, 2,048, and
4,096 tokens. Kernel tests also cover 2–129 rows, including irregular tails,
odd output dimensions, large row grids, and untouched output sentinels. FP16
recurrent output and state match sequential evaluation byte for byte at 32
and 128 rows.

## Pipeline performance

Measurements use the 34/30 layer split, explicit 64-token stage and
transport chunks, a two-chunk window, 4,608-token allocation, and 24 generated
tokens per point. Each process warms all measured frontiers before timing.
The prose sweep measures 128/2K/4K context scaling. Other workload/sampling
arms use 128 tokens. Sampling uses temperature 0.7 and seed 1234567; MTP uses
four drafts and zero margin, with opportunistic MTP for sampling. Arms run
sequentially under the normal clock and thermal policies. These are single
runs; small differences need repetition.

Greedy prose, in tokens/s:

| Context | Plain prefill | MTP prefill | Plain decode | MTP decode | Decode gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 85.66 | 83.88 | 14.76 | 18.03 | 22.2% |
| 2048 | 131.03 | 116.38 | 14.56 | 18.08 | 24.2% |
| 4096 | 124.62 | 109.79 | 14.53 | 17.59 | 21.1% |

At 4K, plain prefill improves from the previous exact profile's 101.62 to
124.62 tokens/s (22.6%). The earlier scalar fallback reached
23.76 tokens/s. The 150–200 tokens/s target was not reached.

The latest chunk sweep retained exact full-vocabulary logits against the
saved token-by-token reference at all three context frontiers:

| Chunk | Prefill at 128 | Prefill at 2K | Prefill at 4K | Planned scratch per stage |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 101.20 | 122.08 | 116.21 | 12.53 MiB |
| 64 | 84.46 | 130.67 | 123.56 | 24.09 MiB |
| 128 | 63.26 | 126.19 | 120.25 | 47.21 MiB |

The 64-token profile gives the best measured long-prefill rate and reduces
planned scratch from 47.21 to 24.09 MiB per stage. A 32-token override improves
short-prefill latency but loses throughput at 2K/4K. There are no new buffers
or repacked weight copies. Plain decode and MTP verification retain their
existing kernel selection; other quantizations retain their prefill profile.

Short-context decode, in tokens/s:

| Workload | Sampling | Plain | MTP | Gain |
| --- | --- | ---: | ---: | ---: |
| Prose | Greedy | 14.76 | 18.03 | 22.2% |
| Prose | Temperature 0.7 | 14.00 | 16.41 | 17.2% |
| Code | Greedy | 14.71 | 18.54 | 26.0% |
| Code | Temperature 0.7 | 13.99 | 21.28 | 52.1% |
| Reasoning | Greedy | 14.70 | 18.91 | 28.6% |
| Reasoning | Temperature 0.7 | 14.37 | 18.37 | 27.8% |

The geometric mean MTP decode gain across these six arms is
28.6%. At 4K, MTP prefill changes 124.62 to 109.79 tokens/s.
Raw measurements, including sampled prefill controls, stay in the ignored
local results directory.

Reasoning MTP initially measured 18.91/18.37 tokens/s for greedy/sampled
generation, below the previous 20.07/19.59. Repeating the same new profile
measured 20.15/19.67. Acceptance histograms were unchanged: eight speculative
calls produced 24 tokens, with five four-token blocks. The table retains the
first measurements; these runs do not establish a stable decode improvement.

Public Vulkan tensor counters peaked at 10.61 GiB on the plain
coordinator, 11.02 GiB with MTP, and 8.88 GiB on the worker.
Separate private scratch counters peaked at 26.4 MiB on the coordinator
and 49.3 MiB on the worker. These exclude pageable GGUF mappings and
general process memory. No Vulkan allocation failed. End-of-arm GPU
temperatures were 52–58°C on the coordinator and 59–72°C on the worker.

With the worker staged as described above, the plain prose sweep is:

```sh
./q36-bench -m gguf/Qwen3.8-27B-UD-Q6_K.gguf --vulkan -t 16 \
  --role coordinator --layers 0:33 --listen 192.168.1.90 9911 \
  --prefill-chunk 64 --dist-prefill-chunk 64 --ctx-alloc 4608 \
  --ctx-start 128 --ctx-max 4096 --step-mul 16 --gen-tokens 24 --warmup \
  --prompt-file tests/long_context_story_prompt.txt
```

The runtime defaults to 64 tokens for this model. The benchmark memory
estimate uses the resolved engine capacity, including model defaults.
For MTP, add
`--mtp gguf/Qwen3.8-27B-UD-Q6_K.gguf --mtp-draft 4 --mtp-margin 0`.
For sampling, add `--temp .7 --seed 1234567`, and add `--mtp-opportunistic`
to its MTP arm. Initialization, staging, snapshot replay, and warm-up are
excluded from rates.
