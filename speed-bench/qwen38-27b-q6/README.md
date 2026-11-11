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

## Pipeline arithmetic

The previous fast batched path changed both projection rounding and the
FP16 recurrent-state rounding interval. Those differences accumulated until
4K logits failed parity. Prefill now batches native decode arithmetic:
Q5_K/Q6_K/IQ4_NL tiles share weight loads across tokens, Q8_0 retains its
native FP32 reduction, and recurrent state rounds after each token. The fused
convolution/gating front processes ordered rows in one dispatch, reusing
existing scratch so future QKV inputs survive output aliasing. Incomplete
tiles and other formats reuse the existing verification kernels.

This profile is selected only for dense models marked Q6_K in GGUF. Plain
decode, MTP verification, and other quantizations keep their existing kernel
selection. No additional dependency or repacked weight copy is required.
The default distributed prefill capacity is 128 tokens. Complete vocabulary
logits match the saved token-by-token reference exactly at 128, 2,048, and
4,096 tokens. Kernel tests also cover 2–129 rows, including irregular tails,
odd output dimensions, and untouched output sentinels. FP16 recurrent output
and state match sequential evaluation byte for byte at 32 and 128 rows.

## Pipeline performance

Measurements use a 34/30 layer split, 128-token stage and transport chunks,
a two-chunk window, 4,608-token allocation, and 24 generated tokens per point.
Each process warms all measured frontiers before timing. The 128/2K/4K prose
sweep checks context scaling; code, reasoning, and sampled generation use
128 tokens. Sampled runs use temperature 0.7 and seed 1234567. MTP uses four
drafts and zero margin, with opportunistic MTP for sampling. Arms run
sequentially. These are single runs; small differences need repetition.

Greedy prose, in tokens/s:

| Context | Plain prefill | MTP prefill | Plain decode | MTP decode | Decode gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 53.14 | 52.62 | 14.74 | 17.72 | 20.2% |
| 2048 | 104.66 | 97.96 | 14.52 | 17.92 | 23.4% |
| 4096 | 101.62 | 95.39 | 14.50 | 17.48 | 20.6% |

The 4K plain prefill rate is 4.28 times the earlier 23.76 tokens/s
scalar fallback. The 150–200 tokens/s target was not reached.

The chunk sweep retained exact full-vocabulary logits at every frontier:

| Chunk | Prefill at 2K | Prefill at 4K | Planned scratch per stage |
| ---: | ---: | ---: | ---: |
| 128 | 104.64 | 101.86 | 47.21 MiB |
| 256 | 92.61 | 89.19 | 93.46 MiB |
| 512 | 80.25 | 79.18 | 185.95 MiB |

Larger chunks increased pipeline bubbles without reducing stage compute
time. The earlier 30/34 split reached 94.53/91.70 tokens/s at 2K/4K.
The default 128-token profile uses about 40.5 MiB more planned scratch than
the earlier 16-token scalar fallback, while sharing existing convolution and
activation panels. The
weight arena and recurrent-state layouts are unchanged by the kernels.

Short-context decode, in tokens/s:

| Workload | Sampling | Plain | MTP | Gain |
| --- | --- | ---: | ---: | ---: |
| Prose | Greedy | 14.74 | 17.72 | 20.2% |
| Prose | Temperature 0.7 | 14.08 | 16.37 | 16.3% |
| Code | Greedy | 14.13 | 17.64 | 24.8% |
| Code | Temperature 0.7 | 14.29 | 21.05 | 47.3% |
| Reasoning | Greedy | 13.87 | 20.07 | 44.7% |
| Reasoning | Temperature 0.7 | 14.17 | 19.59 | 38.2% |

The geometric mean decode gain across these six arms is 31.4%.
Raw measurements, including sampled prefill controls, remain in the ignored
local results directory. MTP adds prefill work: at 4K it changes
101.62 to 95.39 tokens/s.

Public Vulkan tensor counters peaked at 10.64 GiB on the plain
coordinator, 11.05 GiB with MTP, and 8.89 GiB on the worker.
Separate private scratch counters peaked at 52.6 MiB on the coordinator
and 51.0 MiB on the worker. These exclude pageable GGUF mappings and
general process memory. No Vulkan allocation failed. End-of-arm GPU
temperatures were 50–57°C on the coordinator and 57–70°C on the worker.

With the worker staged as described above, the plain prose sweep is:

```sh
./q36-bench -m gguf/Qwen3.8-27B-UD-Q6_K.gguf --vulkan -t 16 \
  --role coordinator --layers 0:33 --listen 192.168.1.90 9911 \
  --prefill-chunk 128 --dist-prefill-chunk 128 --ctx-alloc 4608 \
  --ctx-start 128 --ctx-max 4096 --step-mul 16 --gen-tokens 24 --warmup \
  --prompt-file tests/long_context_story_prompt.txt
```

For MTP, add `--mtp gguf/Qwen3.8-27B-UD-Q6_K.gguf --mtp-draft 4 --mtp-margin 0`.
For sampling, add `--temp .7 --seed 1234567`, and add `--mtp-opportunistic`
to its MTP arm. Initialization, staging, snapshot replay, and warm-up are
excluded from rates.
