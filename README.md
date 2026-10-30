<p align="center">
  <img src="logo.svg" alt="QuarkStar logo" width="220">
</p>

**QuarkStar** is a small native inference engine for **Qwen3.6-35B-A3B**,
**Qwen3.8-27B**.
It is self-contained and deliberately narrow, not a general GGUF runner. 
The main paths are `qwen35moe`-specific Vulkan and Metal graph executors with
Q36-specific loading, prompt rendering, tool calls, KV state, HTTP server,
and coding agent. The repository also includes tools and data for GGUF,
imatrix, quality, and speed.

This project would not exist without **DwarfStar**, **llama.cpp and GGML**, make
sure to read the acknowledgements section, a big thank you to Salvatore Sanfilippo 
aka Antirez and Georgi Gerganov and all the other contributors.

Model support is intentionally opportunistic. The project follows the best open
weights for useful local machine sizes, especially 16 GB machines and 24/32 GB
workstations. A model may be removed when a better replacement arrives.

Supported runtimes:

* **Metal** on Apple Silicon M1 or newer. A 16 GB Mac can run the default
  release model resident; 8 GB Macs are intended to use SSD streaming.
* **Vulkan** on Linux. The backend is generic; its current fast path was
  developed and tuned on the AMD BC-250 with RADV.
* **CPU** as a portable reference and debugging runtime, not the normal
  production path.

# So, what can I do with this software?

* You can run a quite capable model on your very cheap hardware, a $150 machine, 
the BC-250. Even if you don't have enough RAM, with SSD streaming, you can still 
run it at a decent speed.
* You can run the same CLI, agent, evaluation harness, benchmark, and server on
  Apple Silicon through the native Metal graph runtime.

## Requirements

QuarkStar has two native GPU backends:

* **Linux with Vulkan 1.1 or newer** and support for shader `float64` and
  `int64`. Subgroup arithmetic, 16-bit storage, and native `float16` are
  detected at runtime; optimized kernels fall back when an optional feature
  is unavailable.
* **AMD BC-250** (Cyan Skillfish, 24 CUs, 16 GB unified GDDR6) is the primary
  tested Vulkan device. Its vendor/device ID selects the tuned fast path
  automatically. Follow [BC250.md](BC250.md) for the RADV, UMA, kernel-memory,
  governor, build, and optional 40-CU setup.
* **Apple Silicon M1 or newer** with macOS 11 or newer and Xcode or the Command
  Line Tools providing the macOS SDK and Metal framework. Build with
  `make metal` and run with `./q36 --metal`. Shader sources are compiled through
  Metal at runtime, so the build does not require a separate offline
  `metal` invocation. Metal builds use an Apple Silicon/macOS 11 deployment
  target; newer residency APIs are selected only at runtime.

Windows is not currently a build target. The Metal backend uses unified-memory,
mmap-backed model buffers and is independent of the Vulkan build.

## Motivations

* Small open-weight models are already good and fit on normal personal machines, 
and they'll keep getting better.
* AI providers' flat plans keep raising prices, and not everyone can afford $3k–$5k machines.
* BC-250 is the perfect machine for that: $150, a powerful GPU, fast memory, 
decent SSD speed support, and unified memory.
* Qwen3.6-35B-A3B tolerate aggressive routed-expert quantization (recipe by Antirez).
* Qwen3.6-35B-A3B is fast as hell and can potentially run even on a toaster, Vulkan 
is compatible on paper with a huge range of devices, and with DZN and Lavapipe it 
opens up some very interesting possibilities in the future.
* Compressed KV caches and fast local SSDs make long contexts practical.
* The idea of an inference system specialized for a few models.

# AI full disclosure

* This software is developed with **strong assistance from GPT 5.5, 5.6, Claude Fable**
 and with humans leading the ideas, testing, and debugging. We say this openly because 
 it shaped how the project was built. If you are not happy with AI-developed code, this 
 software is not for you. The acknowledgement below is equally important: this would not 
 exist without `llama.cpp` and GGML, largely written by hand.

## Acknowledgements

### To antirez and ds4

QuarkStar is essentially a port of [DwarfStar](https://github.com/antirez/ds4),
redesigned for the Vulkan runtime and retargeted at Qwen3.6-35B-A3B. The Vulkan
engine adapts DwarfStar's ideas to a different model and device. The Apple
runtime also retains and specializes DwarfStar's mature Metal kernel library
for Qwen3.6, including the routed IQ2_XXS/Q2_K expert path.

**Special thanks to Salvatore**, he is a continuous source of inspiration for
me, and his content on YouTube has greatly improved me as a software engineer
and as a person.

This project was born with the intent of improving my skills in LLMs. It's
useful for me for inference and for learning, and I hope it will be useful for
you too.

### To llama.cpp and GGML

`q36.c` does not link against GGML, but it **exists thanks to the path opened by the
llama.cpp project and the kernels, quantization formats, GGUF ecosystem, and hard-won
engineering knowledge developed there**.
We are thankful and indebted to [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
and its contributors. Their implementation, kernels, tests, and design choices were
an essential reference while building this Qwen3.6-35B-A3B specific inference path.
Some source-level pieces are retained or adapted here under the MIT license: GGUF
quant layouts and tables, CPU quant/dot logic, and certain kernels. For this
reason, and because we are genuinely grateful, we keep the GGML authors copyright
notice in our `LICENSE` file.

## Status

The software is currently very fast changing. Consider it alpha quality.
Before each release, a big QA run is executed, however instabilities
are definitely possible.

## More Documentation

The focused documents below cover development, release checks, model details,
and offline tooling. For normal usage, keep reading the next sections.

- [CONTRIBUTING.md](CONTRIBUTING.md): correctness and speed regression testing
  guide for contributors. **Read this before sending a pull request**.
- [BC250.md](BC250.md): Linux, RADV, unified-memory, governor, and build setup
  for the primary Vulkan device.
- [QA_BEFORE_RELEASES.md](QA_BEFORE_RELEASES.md): the complete release test
  matrix.
- [MODEL_CARD.md](MODEL_CARD.md): the fixed Qwen3.6 architecture, tokenizer,
  quantization, and sampling assumptions used by Q36.
- [gguf-tools/README.md](gguf-tools/README.md): offline GGUF generation,
  imatrix collection, quantization tooling, and quality checks.
- [gguf-tools/imatrix/dataset/README.md](gguf-tools/imatrix/dataset/README.md):
  how the calibration prompt corpus is generated.
- [gguf-tools/quality-testing/README.md](gguf-tools/quality-testing/README.md):
  how local GGUFs are scored against OpenRouter continuations.
- [dir-steering/README.md](dir-steering/README.md): vector extraction, scale sweeps, and live
  steering for Qwen models.
- [tests/test-vectors/README.md](tests/test-vectors/README.md): tracked
  llama.cpp continuation vectors used for regression checks.

## Model Weights

This implementation works with Qwen3.6-35B-A3B and dense Qwen3.8-27B GGUFs.
It is not a general GGUF loader, and arbitrary GGUF files will not have
the tensor layout, quantization mix, metadata, or optional MTP state expected by
the engine. The 2 bit quantizations provided here are verified to be actually
high quality: they behave well, work under coding agents, call tools in a reliable way.

Dense Qwen3.8-27B has two supported options, both with an embedded MTP head:

| Download target | GGUF | File size | Use |
|---|---|---:|---|
| `27b` | `Qwen3.8-27B-UD-IQ3_S.gguf` | 11.21 GiB | Higher quality weights |
| `27b-iq3-xxs` | `Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` | 9.73 GiB | More memory for context |

The GSQ-RCO file replaces the earlier unofficial UD-IQ3_XXS option. Its
mixed tensor formats, including BF16 recurrent projections, run natively on
the GPU. Both models use the same CLI, server, agent, evaluator, session
cache, and Qwen3.8 vision projector.
GSQ-RCO uses a separate disk cache namespace and agent checkpoint identity
so switching models cannot restore KV state produced by the other weights.

Download and start the smaller dense model explicitly:

```sh
./download_model.sh 27b-iq3-xxs
./q36-server -m gguf/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf --mtp --vulkan
```

Select IQ3_S for higher quality weights:

```sh
./q36-server \
  -m gguf/Qwen3.8-27B-UD-IQ3_S.gguf \
  --vulkan
```

### Vision

Q36 accepts the matching Qwen3-VL projector as a disk-backed sidecar. It maps
the file without copying the model and streams one weight tensor at a time
through a reusable Vulkan buffer. The full 0.9 GiB projector is never retained
in RAM or VRAM, and streamed pages are released after each dispatch.

```sh
./q36 --vision gguf/Qwen3.6-35B-A3B-mmproj-F16.gguf
./q36 -m gguf/Qwen3.8-27B-UD-IQ3_S.gguf \
  --vision gguf/Qwen3.8-27B-mmproj-F16.gguf
```

At the interactive prompt, `/read photo.jpg` and `/read image.png` submit an
image turn. An optional prompt can be passed along with the image (e.g.
`/read photo.jpg Describe this image` or `/read "path/with spaces/img.png" What is here?`).
JPEG and PNG decoding is built in. The sidecar output dimension is
validated against the selected standard or dense language model before use.
Vision prompt execution currently requires Vulkan. `q36-agent --vision FILE`
exposes `view_image` for local files. `q36-server --vision FILE` accepts OpenAI
image URLs containing PNG/JPEG data URIs, Responses image blocks, and Anthropic
base64 images, including images returned by tools. Remote image URLs are rejected.
The server caches at most 32 MiB of decoded image embeddings and encoded keys;
the projector weights remain streamed from disk. Prefix reuse checks the image
fingerprints and geometry, so replacing pixels with the same number of image
pads rebuilds the context. Vision sessions currently cannot be saved as agent
sessions or disk KV checkpoints.

The 2 bit quants use a very asymmetrical quantization: only the routed MoE
experts are quantized, up/gate at `IQ2_XXS`, down at `Q2_K`. They are the
majority of all the model space: the other components (shared experts,
projections, routing) are left untouched to guarantee quality. The resulting
weight file is about 12.1 GB including the embedded MTP head, which fits
the BC-250 with room left for KV cache and OS.

Download one main model.

```sh
./download_model.sh q2  # Qwen3.6 MoE with an embedded 333 MB MTP head
./download_model.sh 27b # Dense Qwen3.8-27B IQ3_S
```

The `q2` target matches the compiled default path. Pass `--token TOKEN`
if authentication is needed, and `--mtp` to enable its embedded head at runtime.

The script downloads Qwen3.6 models from
`https://huggingface.co/Ninnix96/Qwen3.6-35B-A3B-gguf` and dense Qwen3.8-27B
from `https://huggingface.co/unsloth/Qwen3.8-27B-GGUF`. It stores files under
`./gguf/`, resumes partial downloads with `curl -C -`, and updates
`./q36moe.gguf` to point at the selected model for older scripts.

Then build for the target platform:

```sh
make                  # Linux / generic Vulkan, automatic BC-250 fast path
make vulkan-generic   # explicit generic Vulkan build
make vulkan-bc250     # Vulkan build that requires a BC-250 at runtime
make metal            # macOS / Apple Silicon
```

The normal and CPU-only builds are self-contained. They do not require a
llama.cpp checkout or GGML libraries; those are used only by explicit optional
reference targets under `make test-llama` and `make test-vectors-local`.

The vendored JPEG/PNG decoder in `third_party/iris` is Copyright (c) 2026
Salvatore Sanfilippo and distributed under its included MIT license.

`gguf/Qwen3.6-35B-A3B-IQ2XXS-Q2K-Q8.gguf` is
the default model path used by all runtime binaries. Pass `-m` to select another
supported GGUF from `./gguf/`. Run `./q36 --help` and
`./q36-server --help` for the full flag list.

If you want to regenerate GGUF files, quantize community fine-tunes / abliterated models
(via `--strip-nextn`), or collect a new imatrix, see
[gguf-tools/README.md](gguf-tools/README.md). Those tools are meant for offline
Qwen3.6 model-building work. The native quantizer accepts Q8, F16, or BF16
inputs; imatrix collection still uses the optional llama.cpp tooling.

The Qwen3.6 `q2` download includes its MTP support in the main GGUF;
enable it explicitly with `--mtp`.
Embedded heads are supported in 41-block MoE GGUFs and in both supported
65-block Qwen3.8-27B GGUFs. Use `--mtp` without a filename to select the
embedded head, or `--mtp FILE` for a separate support GGUF. In `q36`,
`q36-server`, and `q36-agent`, this enables the tuned BC-250 configuration:
up to eight tokens, adaptive draft confidence (`--mtp-margin 0`), and
opportunistic sampling. Explicit depth, margin, and sampling options override
these defaults regardless of argument order. The embedding and output weights
remain shared. With `--mtp` omitted, the runtime excludes the extra block from
the resident GPU weight cache.

The MoE build `gguf/Qwen3.6-35B-A3B-IQ2XXS-Q2K-Q8.gguf` retains the
Antirez target tensors and embeds a 332.78 MB MTP block quantized from
Unsloth's original BF16 weights. Its draft gate/up experts use IQ2_XXS,
down experts use Q4_K, projections and shared experts use Q8_0, and
routers/norms remain F32. Enable it with:

```sh
./q36 -m gguf/Qwen3.6-35B-A3B-IQ2XXS-Q2K-Q8.gguf --mtp --temp 0 --nothink \
  -p "Write a C function that reverses a linked list."
```

BC-250 resident Vulkan measurements with this head and Q8_0 K / Q4_0 V
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

The current kernels improve MTP throughput by 19% geometrically over
`d5b4e4b`, with 12–27% gains across the greedy cases. Q8 verification shares
activation loads across output rows, shared experts verify multiple tokens
together, and eight-head attention reuses K/V loads. The MoE normalization
scratch adds 16 KiB; the model and head weights are unchanged. Measurements,
temperatures, dense-optimization audit, and validation are in
[the local speedup report](tests/test-vectors/local/q36-moe-speed-20261023/README.md).
The [head-selection report](tests/test-vectors/local/q36-moe-mtp-20261023/README.md)
records the earlier quantization comparisons.

For greedy Vulkan generation with the dense embedded head:

```sh
model=gguf/Qwen3.8-27B-UD-IQ3_S.gguf
./q36 -m "$model" --mtp \
  --temp 0 --nothink -p "Write a C function that reverses a linked list."
```

The verifier uses up to eight tokens per pass. Draft rejection restores the
accepted recurrent state without replaying target matrices. The MTP cache
is populated during prefill and corrected from accepted target hidden states.
Verification shares quantized weights across rows and keeps token selection
on the GPU. The draft output head uses a small mask of recent target
alternatives; target verification always scores the full vocabulary.

Resident Vulkan chooses recursive MTP lengths between two and four rows from
the current draft's confidence. With `--mtp-draft 8`,
a matching 12–16-token suffix can supply a longer copied tail after MTP
confirms its first proposal. Positive `--mtp-margin` values use MTP alone.
The regression tests compare committed tokens and every target logit with
ordinary decode, including quantized KV, checkpoint restore and verifier
widths up to eight. MTP also matches a separate ordinary engine with MTP disabled.
The IQ2_XXS/Q2_K MoE target also verifies rows together, retaining ordinary
decode routing, shared-expert and recurrent arithmetic. Other MoE tensor
mixes and SSD streaming retain scalar verification.

Opportunistic sampling is the default with `--mtp` at a positive temperature:

```sh
./q36 -m "$model" --mtp --temp 1 --seed 42 --nothink \
  -p "Write a C function that reverses a linked list."
```

Each block samples its first token with the requested temperature, top-k,
top-p and min-p, then accepts the draft tail while it matches the target's
greedy continuation. This deliberately changes the sampling distribution.
The next block samples again from the last committed target state. Target
logits, KV updates and rejection rollback use the same verifier as greedy MTP.
This also works in `q36-agent` and in `q36-server` for non-batched text
requests with zero presence/frequency penalties; other server requests keep
ordinary sampling. Use `--no-mtp-opportunistic` to sample every token at
positive temperature while retaining exact greedy MTP at temperature zero.
`--mtp-draft 1` disables speculative blocks.

With `--mtp-margin 0`, dense Vulkan drafting uses the MTP head's
confidence to choose its length. It verifies at least one proposal and
extends up to four rows while the top-two logit gap is at least 1.0.
Confirmed history tails can still extend to eight. This avoids wider
verification of weak prefixes and removes wall-clock timing from
block-length decisions. Target acceptance still checks every proposed token.

MTP remains experimental. These BC-250 comparisons use the pushed
`e497fd7` build and the new implementation, 128 output tokens, and Q8_0 K /
Q4_0 V cache. Each sweep starts at 128 and extends through 2K and 4K.

Greedy generation:

| Context | Story, pushed → new t/s | C source, pushed → new t/s |
| ---: | ---: | ---: |
| 128 | 30.22 → 33.06 | 25.80 → 27.77 |
| 2048 | 32.34 → 34.08 | 30.01 → 30.35 |
| 3968 | 26.68 → 29.01 | 30.70 → 30.74 |
| 4096 | 31.96 → 32.45 | 30.57 → 34.52 |

Opportunistic generation at temperature 1, top-k 0, top-p 1 and min-p 0.05
(all cells are pushed → new t/s):

| Context | Story, seed 1 | Story, seed 42 | C source, seed 1 | C source, seed 42 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 26.91 → 35.56 | 29.26 → 29.33 | 23.36 → 29.01 | 26.57 → 29.02 |
| 2048 | 30.14 → 31.18 | 26.28 → 28.49 | 31.96 → 36.23 | 29.07 → 36.62 |
| 3968 | 26.64 → 30.16 | 25.45 → 30.03 | 29.12 → 32.08 | 28.35 → 31.12 |
| 4096 | 28.69 → 29.86 | 28.94 → 27.07 | 28.28 → 30.87 | 31.23 → 30.20 |

The regressed 2K code/seed-42 case reaches 36.62 t/s,
with 36.46 t/s in the final repeat. Additional pushed-build controls
measured 28.97 and 28.72 t/s; the older pre-regression build measured
30.94 and 31.03 t/s. Fresh 8K story/seed-1 generation changes from
30.74 to 32.63 t/s.

The largest sampled regression is 6.5% for story, seed 42, context 4096.
Sampled arms can produce different continuations, so these end-to-end
comparisons include continuation variation. A general 35 t/s rate or
attained hardware limit is not established. The paired IQ4 lookup is
independently 6–17% faster in warmed kernel measurements with identical
scalar outputs. It adds no tensor or weight buffers.

All greedy/sampled replay, quantized KV, checkpoint restore, scalar fallback,
MoE regression, separate-engine 8K and fresh 16-thread CPU/Vulkan checks
through top-64 pass. Every MTP replay has zero target-logit difference.
Maximum recorded benchmark temperature is 73°C.
Plain decoding remains the default. Commands, raw controls, rejected probes,
frozen sources and complete results are in
[the local benchmark artifacts](tests/test-vectors/local/q38-mtp-20261021-v6/README.md).

## Speed

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

![BC-250 Qwen3.6-35B-A3B prefill, decode, and MTP decode](speed-bench/bc250-20261025/qwen36_ts.svg)

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

![BC-250 Qwen3.8-27B prefill, decode, and MTP decode](speed-bench/bc250-20261025/qwen38-xxs_ts.svg)

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

![BC-250 Qwen3.8-27B IQ3_S prefill, decode, and MTP decode](speed-bench/bc250-20261025/qwen38-iq3s_ts.svg)

Recorded on 2026-10-25. See [performance and benchmarking](docs/PERFORMANCE.md)
for the raw data, commands, Metal measurements, and earlier baselines.

## Running Models Larger Than Available Memory

Metal and Vulkan SSD streaming keep non-routed weights resident and load
selected routed experts into a bounded backend cache. The planner reserves memory
for the configured context, prefill scratch, planned server sessions, and a staging
margin before assigning the remaining model budget to static weights and experts.
Explicit cache sizes are targets capped by the same budget. Creating a larger or
additional session reduces the expert cache when necessary; it fails cleanly if
static weights and the minimum working cache leave insufficient room. Startup
logs show the resolved context and model budgets.

```sh
./q36 --metal --ssd-streaming -p "Explain radix trees."
./q36 --vulkan --ssd-streaming --ssd-streaming-cache-experts 256 -p "Hello"
./q36 --metal --ssd-streaming --ssd-streaming-cache-experts 1GB -p "Hello"
```

`NGB` is a routed-expert byte budget. Q36 converts it to the number of expert
slots that fit the release GGUF. A plain integer requests an expert-slot count;
both forms are capped to leave room for the configured contexts.
Non-routed weights, KV cache, activations, and graph scratch need additional
memory. Startup prints the resolved slot count and actual cache allocation.

A built-in or `Q36_VK_STREAMING_EXPERT_HOTLIST` profile biases eviction
without reading expert weights at startup. It does not fill the cache with
arbitrary experts when no profile exists. Use `--ssd-streaming-cold` for an
empty cache, or `--ssd-streaming-preload-experts N` to request an explicit
weight preload. Metal stores mixed expert sizes in component-wise padded cache
slots, so the IQ2/Q2 and Q4 routed layers share one bounded cache and one
hotlist policy. Vulkan retains its original single-size-class behavior. The
`Q36_VK_STREAMING_EXPERT_HOTLIST` and
`Q36_VK_DISABLE_STREAMING_EXPERT_HOTLIST` names are retained for compatibility
and apply to both graph runtimes.

As in DS4, an explicit routed prefix can stay fully resident:

```sh
./q36 --ssd-streaming --ssd-streaming-full-layers 4 -p "Hello"
```

Full layers are charged at their actual byte size and the remaining budget
must still hold one layer of dynamic expert slots for prefill. The default is
zero because the dynamic-only cache is faster on the BC-250. Pass
`--ssd-streaming-full-layers 0` to disable an explicit setting.

### Metal SSD streaming

Metal follows DS4's bounded expert-cache design. Routed expert bytes are read
from the GGUF into size-limited shared Metal buffers, selected IDs are remapped
to cache slots, and cache misses use a biased least-recently-used eviction
policy. `--ssd-streaming-cache-experts`, preload, cold-cache, and
full-resident-layer controls therefore allocate and constrain real Metal
buffers; they are not hints to macOS VM paging.

For the mixed 13 GB model, each dynamic slot is sized for the largest routed
tensor layout. Smaller IQ2/Q2 experts and the six Q4_K expert layers can remain
cached together; changing precision at layer 34 does not reset the cache or
fall back to the CPU.

On a 16 GB Mac, start with the resident q2 model:

```sh
./q36 --metal -p "Hello"
```

For the larger release GGUF, or on an 8 GB Mac, start conservatively:

```sh
./q36 --metal --ssd-streaming --ctx 4096 -p "Hello"
./q36 --metal --ssd-streaming \
  --ssd-streaming-cache-experts 512 --ctx 4096 -p "Hello"
```

The automatic form uses `recommendedMaxWorkingSetSize`. The explicit example
allocates about 0.42 GiB for the q2 model or 0.85 GiB for the mixed model,
whose slots are padded for Q4_K. It leaves more headroom for the OS and KV
state. Increase it only while memory pressure, swap, and responsiveness remain
acceptable. A physical 8 GB Apple Silicon run is part of release QA;
`--simulate-used-memory 8GB` on a larger Mac is diagnostic evidence, not a
replacement.

`--ssd-streaming-preload-experts N` performs an actual startup read. The normal
automatic hotlist only biases eviction and does not read expert weights.
`--ssd-streaming-full-layers N` loads the first `N` routed layers into separate
full-resident Metal buffers and deducts their exact bytes from the dynamic
budget. `q36-bench` prints cache slots, cache/full-layer GiB, hits, misses,
loads, evictions, and GGUF bytes read in its final memory report.

## Native Agent

Q36 includes a native coding agent. Inference is controlled inside the agent
itself, without a socket or API boundary, so the transcript and live KV state
are one session. The tools and system prompt use Qwen3.6's native tagged tool
format directly. This provides a few advantages:

- Low latency for generated text, tool calls, and new sessions.
- Live progress during long prefills.
- No OpenAI, Anthropic, or Hermes conversion in the model loop.
- The transcript and KV state cannot drift apart.
- Built-in file, search, shell, process, and web tools tuned for the model.
- Saved sessions can be switched without prefill when their KV payload is
  present.

Start the agent in the current directory, another project, or one-shot mode:

```sh
./q36-agent
./q36-agent --chdir /path/to/project
./q36-agent -m gguf/Qwen3.8-27B-UD-IQ3_S.gguf --mtp --chdir /path/to/project
./q36-agent --non-interactive -p "Inspect the tests and fix the failure."
```

`--mtp` uses the same Vulkan verifier as the CLI and server. The agent streams
accepted tokens individually and decodes tags, thinking closure, and automatic
edit-old shortening one token at a time. Tool calls therefore finish before
the next assistant round, and the transcript stays aligned with the live KV.

Adaptive thinking closure starts after 50000 thinking tokens by default. For
budgets of at least 8000, its allowed `</think>` rank rises to the top 64 over
the next `min(N/2, 8192)` tokens, where `N` is `--thinking-budget`. Smaller
budgets keep the previous ranking schedule. The rank can rise further if
thinking continues. This is a soft budget; `--tokens` is the hard output limit.

With Qwen3.8-27B, `--thinking-budget` also selects the starting effort:
up to 8000 tokens is `low`, up to 16000 is `medium`, up to 24000 is `high`,
and larger budgets use `xhigh`. The default is `xhigh` with a 50000-token
closure target. `high` uses Qwen3.8's native `xhigh` instruction. Explicit
`--think-low`, `--think-medium`, `--think`, or `--think-xhigh` overrides the
starting mode without changing the budget. At an empty idle agent prompt,
Tab cycles `low → medium → high → xhigh` for Qwen3.8 and `off ↔ on` for
Qwen3.6. Tab leaves the closure budget unchanged; the footer shows both.

Agent user and system messages accept `<|think_on|>` and `<|think_off|>`.
The marker is removed before rendering and remains in effect for later turns.
Historical thinking stays in the append-only transcript until compaction.

Resident Metal and Vulkan both use Q8_0 keys with Q4_0 values and default to a
100000-token agent context. The backend's automatic resident GPU prefill width
resolves to 1024 tokens. Compact Metal attention scratch keeps this faster
chunk width practical at long contexts. CPU uses F16 KV. SSD-streamed model
weights also default to a 100000-token context with F16 KV:

```sh
./q36-agent --metal
./q36-agent --metal --ssd-streaming --ssd-streaming-cache-experts 512
```

Explicit `--ctx`, `-ctk`, and `-ctv` values override the preset.

`q36-agent --chdir` loads its model and runtime assets from the launch directory,
then changes to the requested project for agent tools. Built-in path tools also
expand `~` to the current user's home directory.

When a command such as `sudo` requests a terminal password, the interactive
agent opens a private prompt with input hidden. Type the password there;
Ctrl+C cancels the command. Password input is sent directly to the command,
without entering chat history, traces, or captured tool output. Other command
stdin reads receive EOF; `--non-interactive` cannot prompt for passwords.

Use `/hints on` for occasional short explanations of programming concepts behind
the current work, rendered as teal blockquotes. `/hints off` disables them. This
setting lasts for the current process and is reapplied after context compaction.

Both `q36` and `q36-agent` accept `--prefix-file FILE`. The file contains alternating
`USER:` and `ASSISTANT:` lines, starting with a user and ending with an assistant.
Turn content may span multiple lines. The CLI prefills this conversation before
the first interactive prompt; the agent keeps it through reset and compaction.
For example:

```text
USER: Our project uses C99 and has no external dependencies.
ASSISTANT: I will follow those constraints.
```

Sessions are stored in `~/.q36/kvcache`. Use `/save` to persist the current
session, `/list` to show saved sessions, and `/switch <sha>` to resume one.
The session ID remains stable across later saves. `/del <sha>` removes a saved
session. `/strip <sha>` keeps its transcript and title but removes the KV
payload; switching to a stripped session rebuilds the KV cache by prefilling
the saved text. `/compact` compacts the current context immediately.
Exiting during generation stops the worker before asking whether to save.
Sessions containing images cannot be saved yet; declining to exit after a save
failure returns to the current chat.

## Benchmarking

`q36-bench` measures instantaneous prefill and generation throughput at
context frontiers instead of reporting one whole-run average. It loads the
model once, walks a fixed token sequence to frontiers such as 2048, 4096, and
6144, and uses incremental prefill so each row measures only the newly added
token interval. After each frontier it saves the live KV state to memory,
generates a fixed greedy non-EOS probe, restores the snapshot, and continues
prefill.

```sh
./q36-bench --vulkan --prompt-file tests/long_context_story_prompt.txt \
  --ctx-start 2048 --ctx-max 32768 --gen-tokens 128

./q36-bench --metal --prompt-file tests/long_context_story_prompt.txt \
  --ctx-start 2048 --ctx-max 32768 --gen-tokens 128
```

## Capability Evaluation

`q36-eval` is a real-model integration benchmark, not a leaderboard runner.
Its 92 embedded questions are a regression subset: 25 GPQA Diamond, 25
curated SuperGPQA, 25 AIME 2025, and 17 COMPSEC cases. It loads the GGUF,
renders Qwen3.6 chat prompts, streams sampled tokens in a TUI, grades the final
answers, and prints prompt-token, generated-token, and pass/fail results.

```sh
./q36-eval --trace /tmp/q36-eval.txt
./q36-eval --metal --ssd-streaming --trace /tmp/q36-eval-metal-ssd.txt
```

The default run uses a 16000-token generation budget and thinking mode. The
context is sized from the largest selected prompt plus that budget, up to the
model's 262144-token native context. Press `p` to pause, `q` to stop and print
the report, Up/Down to select a question, and Enter to queue it next. `--plain`
disables the TUI.

Use `--regrade-trace /path/to/trace.txt` to rerun the current answer extractor
and scorer on a saved trace without loading the model. A short deterministic
smoke run is:

```sh
./q36-eval --plain --questions 4 --tokens 2048 --temp 0 --seed 1
```

## CLI

One-shot prompt:

```sh
./q36 -p "Explain Redis streams in one paragraph."
```

Without `-p`, Q36 starts an interactive multi-turn chat:

```sh
./q36
q36>
```

The CLI keeps the rendered transcript and live graph KV checkpoint, so each
turn extends the previous conversation. Useful commands are `/help`, `/think`,
`/think-max`, `/nothink`, `/ctx N`, `/read FILE [PROMPT]`, and `/quit`. Ctrl+C interrupts
the current generation and returns to `q36>`.

Thinking mode is enabled by default. Use `/nothink` or `--nothink` for direct
answers. `--mtp` enables the embedded MTP head with the tuned defaults above;
`--mtp MTP.gguf` selects a separate support model. Greedy decoding stays exact.
At positive temperature, opportunistic MTP samples block starts and accepts
greedy tails; `--no-mtp-opportunistic` restores sampling for every token.

## Server

Start a local OpenAI/Anthropic-compatible server:

```sh
./q36-server --vulkan --ctx 32768 \
  --kv-disk-dir /tmp/q36-kv --kv-disk-space-mb 8192

./q36-server --metal --ctx 32768 \
  --kv-disk-dir /tmp/q36-kv --kv-disk-space-mb 8192
```

On an 8 GB Mac, reduce context and use the bounded Metal cache:

```sh
./q36-server --metal --ssd-streaming \
  --ssd-streaming-cache-experts 512 --ctx 4096
```

### Web UI

`q36-server` serves a chat web UI at `http://HOST:PORT/` (default
`http://127.0.0.1:8000/`). It is built into the binary at compile time from
`webui/`, so there is nothing to deploy and no Node.js build step.

- Streaming Markdown with syntax-highlighted code, tables, LaTeX math, copy and
  wrap buttons, and a sandboxed live preview for `html`/`svg` code blocks.
- Collapsible reasoning ("Thought for ...") with a thinking-mode picker
  (default, off, low, medium, high, xhigh, max).
- Branching conversations: editing a prompt or regenerating a reply keeps the
  old version and adds a `< 2/3 >` switcher.
- Image attachments (needs `--vision`), text/code file attachments, paste and
  drag-and-drop, a context-usage meter, per-reply tok/s and cached-prompt stats.
- Chats are stored in the browser (IndexedDB) with search, pinning, rename,
  Markdown/JSON export and JSON import. Sampling parameters, system prompt,
  light/dark theme, English/Russian UI and keyboard shortcuts (`?` lists them)
  are in Settings.
- Pass `--open-browser` to have the server launch your default browser at the
  UI URL (`xdg-open`/`open`) as soon as it starts listening, the same way
  llama.cpp's server does; it is off by default so headless/remote servers
  don't try to pop a browser.

The UI resends each conversation verbatim, including `reasoning_content`, so
the server's live KV prefix reuse keeps working across turns. Chat titles are
derived locally; model-written titles are an opt-in setting because they would
evict that cache.

Without extra options the server keeps one mutable backend/KV checkpoint and
uses the original single graph worker. Stateless clients that resend a longer
version of the same prompt can reuse that prefix instead of pre-filling from
token zero.

`--batched-session N` opts into `N` independent resident sessions:

```sh
./q36-server --ctx 32768 --batched-session 4
```

Use `--mixed-prefill-quantum N` to tune how many prefill tokens a batched
session runs per scheduling turn while another session is generating. The
default is 128; smaller values favor decode latency, larger values favor
prefill throughput.

Each active request owns one slot until it finishes; excess requests wait for
an idle slot. Assignment prefers the resident live/token prefix with the
longest match. When disk KV caching is enabled, an unmatched idle slot is
persisted before reuse. Thinking state, tool continuations, RNG, logits,
recurrent state, and full-attention KV remain session-local.

One coordinator owns model execution. Decode-ready slots coalesce for up to
2 ms and advance in one model step. Prefills run round-robin in bounded
quanta: 2048 tokens while no generation is active and 128 while any generation
is active. This prevents one large prompt from blocking every active decoder.
MTP speculative decoding is disabled in batched mode.

Resident context memory is multiplied by `N`; startup prints both the
per-session estimate and the total. Choose `N` and `--ctx` so all session KV,
recurrent state, and graph scratch fit. Backend behavior is:

| Backend | Multi-session execution |
| --- | --- |
| Vulkan resident, 2-8 decode-ready rows | Native row-batched shared projections and FFN work with private positions, recurrent state, and typed KV when graph scratch can hold every row. F16/F16 and Q8_0/Q4_0 KV are supported; other KV pairs and unsupported shapes use ordered fallback. |
| Vulkan SSD streaming | Deterministic ordered fallback, preserving expert-cache ownership. |
| Metal resident, 2-8 decode-ready rows | Native row-batched graph execution where the kernel shape is supported; ordered exact fallback otherwise. |
| Metal SSD streaming | Deterministic ordered fallback, preserving bounded expert-cache ownership. |
| CPU, batches above 8, or forced `Q36_VK_SESSION_BATCH=0` | Deterministic ordered fallback. The compatibility environment variable also controls Metal session batching. |

Batch size one calls `q36_session_eval()` directly. If a batched step fails,
all members are invalidated so none can silently continue from a partially
advanced frontier. Ordered fallback preserves concurrency and scheduling
fairness, but does not provide the aggregate throughput gain of native GPU
batching.

Supported endpoints:

- `GET /v1/models`
- `GET /v1/models/<loaded-model-id>`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/completions`
- `POST /v1/messages`

`/v1/chat/completions` accepts the usual OpenAI-style `messages`,
`max_tokens`/`max_completion_tokens`, `temperature`, `top_p`, `top_k`,
`min_p`, `presence_penalty`, `frequency_penalty`, `seed`, `stream`,
`stream_options.include_usage`, `tools`, and `tool_choice`. Tool schemas and
calls use the native Qwen3 Coder tagged format, and generated calls are mapped
back to OpenAI tool calls.

Both Qwen3.6 and Qwen3.8 accept the fixed-template controls:

```json
"chat_template_kwargs": {
  "enable_thinking": false,
  "preserve_thinking": true
}
```

`preserve_thinking` defaults to `true`, retaining earlier assistant reasoning
verbatim so later prompts remain a prefix-cache match. Set it to `false` to
strip reasoning before the latest user query. System and user messages may also
contain `<|think_on|>` or `<|think_off|>`; Q36 removes the control marker before
rendering and applies it to subsequent turns.

When omitted by the client, Qwen uses `temperature=1`, `top_p=1`, no top-k cap,
and `min_p=0.05`. CLI, agent, and server use the loaded model's defaults. Eval
stays fixed at the Qwen sampling defaults for comparable runs. Explicit values
always win.

`/v1/responses` accepts string or message-array input, instructions, direct
tool schemas, function-call continuations, function-call outputs, sampling
controls, reasoning controls, and `max_output_tokens`. It returns native
Responses API message, reasoning, and function-call output items. With
`stream:true`, it emits Responses API SSE events through
`response.completed`.

`/v1/messages` is the Anthropic-compatible endpoint used by Claude Code
style clients. It accepts `system`, `messages`, `tools`, `tool_choice`,
`max_tokens`, `temperature`, `top_p`, `top_k`, `stream`, `stop_sequences`,
and thinking controls. Tool uses are returned as Anthropic `tool_use`
blocks.

Both APIs support SSE streaming. In thinking mode, reasoning is streamed in
the native API shape instead of being mixed into final text. OpenAI chat
streaming also streams tool calls as soon as the `<tool_call>` opening is
recognized: the tool header is sent first, then each completed native
parameter is forwarded as a `tool_calls[].function.arguments` delta while
generation continues. The Anthropic endpoint streams thinking and text live, then emits
structured `tool_use` blocks when the generated tool block is complete.

Pass `--cors` to add `Access-Control-Allow-Origin`, methods, and headers and
to answer browser `OPTIONS` preflight requests. CORS headers are disabled by
default.

### Tool call handling and canonicalization

Qwen3.6-35B-A3B emits tool calls in its native tagged format. Tool definitions
are provided in the system prompt inside `<tools>...</tools>`. A call has one
`<function=name>` block inside `<tool_call>`, with one
`<parameter=name>` block per argument:

```text
<tool_call>
<function=list_files>
<parameter=pattern>
*.c
</parameter>
<parameter=max_depth>
2
</parameter>
</function>
</tool_call>
```

Agent clients do not send that same text back on the next request: they send
normalized OpenAI/Anthropic JSON tool-call objects. **If the server
re-rendered those objects slightly differently, the rendered byte prefix
would no longer match the live KV checkpoint** and the next turn would have
to be rebuilt.

All markers are plain ASCII. Q36 keeps the exact replay and canonicalization
machinery inherited from `ds4`, because a sampled call and its next-turn API
rendering must still be byte-identical to avoid silent KV drift.

The first line of defense is exact replay. Every tool call gets an
unguessable API tool ID, and the server remembers `tool id -> exact sampled
<tool_call> block` in a bounded in-memory map backed by radix trees. When
the client later sends that tool ID back, the prompt renderer uses the exact
bytes the model sampled, not a freshly formatted approximation. This map
can also be saved inside KV cache files, so exact replay survives server
restarts for cached histories.

**Canonicalization is only the backup path**. If the exact sampled block is
missing, or exact replay is disabled with
`--disable-exact-tool-replay`, the server renders deterministic native Qwen
tags from the JSON tool object, following schema property order. After a
tool-call turn, it compares the live sampled token stream with the prompt
that the next client request will render. If needed, it rewrites the live
checkpoint, or falls back to an older disk KV snapshot and replays only the
suffix. This keeps the model continuation aligned with the stateless API
transcript.

During generation, the server also treats native Qwen syntax differently from
payload. When the model is emitting stable protocol structure — tool,
function, and parameter tags — sampling is forced to `temperature=0` so
the tool call stays parseable. This greedy mode does **not** apply to
argument values: string contents inside the arguments JSON, including file
contents and edit text, use the request's normal sampling settings. That
separation is important: deterministic decoding is helpful for syntax, but
can create repeated text when applied to long code or file bodies.

Minimal OpenAI example:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3.6-35b-a3b",
    "messages":[{"role":"user","content":"List three Redis design principles."}],
    "stream":true
  }'
```

For connecting a separate UI through the server, see [UI integration](docs/UI.md).

### Agent Client Usage

`q36-server` can be used by local coding agents that speak OpenAI-compatible
chat completions. Start the server first, and set the client context limit
no higher than the `--ctx` value you started the server with:

```sh
./q36-server --ctx 32768 --kv-disk-dir /tmp/q36-kv --kv-disk-space-mb 8192
```

On a BC-250 with 16 GB of unified memory, weights take ~10–11 GB at Q2,
which leaves roughly 2–4 GB for KV cache, scratch buffers, OS and your
client. The model's native context is 256K tokens, but the live context must
still fit in available memory. Disk KV checkpoints avoid repeated prefill and
preserve sessions across restarts; they do not enlarge the active context
window.

The `262144` output limit in the configs below matches the model's native
context ceiling. The server stops earlier when its configured context window
is full.

For **opencode**, add a provider and agent entry to
`~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "q36": {
      "name": "q36.c (local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "q36-local"
      },
      "models": {
        "qwen3.6-35b-a3b": {
          "name": "Qwen 3.6 MoE (q36.c local)",
          "limit": {
            "context": 32768,
            "output": 262144
          }
        }
      }
    }
  },
  "agent": {
    "q36": {
      "description": "Qwen 3.6 MoE served by local q36-server",
      "model": "q36/qwen3.6-35b-a3b",
      "temperature": 0
    }
  }
}
```

For **Pi**, add a provider to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "q36": {
      "name": "q36.c local",
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "q36-local",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_tokens",
        "supportsStrictMode": false,
        "thinkingFormat": "qwen",
        "requiresReasoningContentOnAssistantMessages": true
      },
      "models": [
        {
          "id": "qwen3.6-35b-a3b",
          "name": "Qwen 3.6 MoE (q36.c local)",
          "reasoning": true,
          "thinkingLevelMap": {
            "off": null,
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh"
          },
          "input": ["text"],
          "contextWindow": 32768,
          "maxTokens": 262144,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
```

Optionally make it the default Pi model in `~/.pi/agent/settings.json`:

```json
{
  "defaultProvider": "q36",
  "defaultModel": "qwen3.6-35b-a3b"
}
```

For **Claude Code**, use the Anthropic-compatible endpoint. A wrapper like
this matches the local `~/bin/claude-q36` setup:

```sh
#!/bin/sh
unset ANTHROPIC_API_KEY

export ANTHROPIC_BASE_URL="${Q36_ANTHROPIC_BASE_URL:-http://127.0.0.1:8000}"
export ANTHROPIC_AUTH_TOKEN="${Q36_API_KEY:-q36-local}"
export ANTHROPIC_MODEL="qwen3.6-35b-a3b"

export ANTHROPIC_CUSTOM_MODEL_OPTION="qwen3.6-35b-a3b"
export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="Qwen 3.6 MoE local q36"
export ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION="q36.c local GGUF"

export ANTHROPIC_DEFAULT_SONNET_MODEL="qwen3.6-35b-a3b"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="qwen3.6-35b-a3b"
export ANTHROPIC_DEFAULT_OPUS_MODEL="qwen3.6-35b-a3b"
export CLAUDE_CODE_SUBAGENT_MODEL="qwen3.6-35b-a3b"

export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1
export CLAUDE_STREAM_IDLE_TIMEOUT_MS=600000

exec "$HOME/.local/bin/claude" "$@"
```

Claude Code may send a large initial prompt, often around 25k tokens,
before it starts doing useful work. Keep `--kv-disk-dir` enabled: after the
first expensive prefill, the disk KV cache lets later continuations or
restarted sessions reuse the saved prefix instead of processing the whole
prompt again.

## Thinking Modes

Qwen3.6-35B-A3B and Qwen3.8-27B have distinct non-thinking and thinking modes, controlled
natively by `<think>...</think>` blocks rendered into the prompt. The server
defaults to thinking mode.

Mapping of API thinking controls to prompt rendering:

- Anthropic `thinking: {"type":"enabled"}` → thinking on (default).
- Anthropic `thinking: {"type":"disabled"}` → thinking off.
- Qwen3.8-27B supports `low`, `medium`, and `xhigh` in its chat template.
  `high` and the default select `xhigh`; `minimal` selects `low`.
- Qwen3.6-35B-A3B has only thinking on or off. Positive efforts all enable
  thinking without a model-specific effort instruction.
- Numeric effort strings map `"0"` to thinking off, `"1"`–`"33"` to `low`,
  `"34"`–`"66"` to `medium`, and `"67"`–`"100"` to `xhigh`. On Qwen3.6,
  all positive values enable the same thinking mode. Only `"max"` selects
  QuarkStar Think Max.
- `thinking.budget_tokens` on chat or Anthropic messages starts adaptive
  `</think>` ranking at that token count. On Qwen3.8 it also selects `low`
  through 8000, `medium` through 16000, `high` through 24000, and `xhigh`
  above 24000. Explicit `none` and `max` efforts keep their mode.
- Explicit non-thinking: `thinking:{"type":"disabled"}`, `think:false`, or
  `chat_template_kwargs:{"enable_thinking":false}`. Both profiles use the
  `<think>\n\n</think>\n\n` non-thinking prefix.

## KV Cache Quantization

Q36 supports `f16`, `q8_0`, and `q4_0` KV rows. Select key and value types
independently with `-ctk` and `-ctv`:

```sh
./q36 -ctk q8_0 -ctv q4_0 -p "Explain radix trees."
./q36-server -ctk f16 -ctv f16
```

Resident Metal and Vulkan frontends default to Q8_0 keys and Q4_0 values. CPU
and SSD streaming default to F16 for both. Explicit flags always override
these defaults. KV quantization reduces context memory; it does not change
model weight quantization or extend Qwen3.6's 262144-token native context.

## Disk KV Cache

Chat/completion APIs are stateless: agent clients usually resend the whole
conversation every request. `q36-server` first tries the cheap exact
token-prefix check, then falls back to comparing rendered prompt bytes with
decoded checkpoint bytes. The live in-memory checkpoint covers the current
session; the disk KV cache makes useful prefixes survive session switches
and server restarts.

The default server has one live KV cache. With `--batched-session N`, each
resident slot has one and active slots are never evicted. When an unrelated
request reuses an idle slot, its old checkpoint can only be resumed without
re-processing if it was written to the disk KV cache.

Enable it with:

```sh
./q36-server --kv-disk-dir /tmp/q36-kv --kv-disk-space-mb 8192
```

The cache key is the SHA1 of the rendered byte prefix, and files are named
`<sha1>.kv`. The Q36 payload still stores the exact token IDs and graph
state for that prefix. This matters for continued chats: the model may have
generated one token whose decoded text is later sent back by a client as two
canonical prompt tokens. A rendered byte-prefix hit can still reuse the
checkpoint and tokenize only the new suffix. The file is intentionally
written with ordinary `read`/`write` I/O, not `mmap`, so restoring cache
entries does not add more VM mappings to a process that already maps the
model.

Tool calls also keep a bounded exact-replay map keyed by unguessable tool
IDs, so client JSON history can be rendered back to the exact sampled text.
The RAM map keeps up to 100000 IDs by default; tune it with
`--tool-memory-max-ids`. Use `--disable-exact-tool-replay` to disable this
and fall back to canonical JSON-to-Qwen rendering.

On disk, a cache file is:

```text
KVC fixed header, 48 bytes
u32 rendered_text_bytes
rendered_text_bytes of UTF-8-ish token text
Q36 session payload, payload_bytes from the KVC header
optional tool-id map section
```

The fixed header is little-endian:

```text
0   u8[3]  magic = "KVC"
3   u8     version = 1
4   u8     representative model tensor quant bits: 1-6 or 8
5   u8     save reason: 0 unknown, 1 cold, 2 continued, 3 evict, 4 shutdown
6   u8     extension flags, bit 0 = appended tool-id map
7   u8     reserved
8   u32    cached token count
12  u32    hit count
16  u32    context size the snapshot was written for
20  u8[4]  reserved
24  u64    creation Unix time
32  u64    last-used Unix time
40  u64    Q36 session payload byte count
```

The rendered text is the tokenizer-decoded text for the cached token
prefix. It is both the human-inspectable prefix and the lookup identity:
its SHA1 is the filename, and a file is reusable only when those bytes are
a prefix of the incoming rendered prompt. After load, the exact checkpoint
tokens from the Q36 payload remain authoritative, and only the incoming
text suffix after the cached bytes is tokenized.

The optional tool-id map is present only when header extension bit 0 is
set. Appended sections use fixed bit order, so future extension bits can
add fields without ambiguity. The map stores unguessable API tool call IDs
back to the exact `<tool_call>` block the model sampled. Only mappings whose
block is present in the rendered cached text are stored. This lets restarted
servers render later client history byte-for-byte like the original model
output, even if the client reorders JSON arguments.

The current tool-id map section is:

```text
0   u8[3]  magic = "KTM"
3   u8     version = 1
4   u32    entry count

For each entry:
0   u32    tool id byte length
4   u32    sampled block byte length
8   bytes  tool id
... bytes  exact sampled <tool_call> block
```

The section is auxiliary replay memory, not model state. A cache hit
restores the session payload first, then loads the map if present. Before
rendering a request, the server can also scan cache files for the tool IDs
present in the client history and load just those mappings, so an exact
replay can survive server restarts even when the matching KV snapshot is
not the one ultimately used for the rendered-prefix hit.

The current Q36 session payload starts with fourteen little-endian `u32`
fields for version 2, or sixteen fields for typed-KV version 3:

```text
0   magic = "Q36 "
1   payload version = 2 (f16 KV) or 3 (typed KV)
2   saved context size
3   prefill chunk size
4   checkpoint token count
5   vocabulary size
6   layer count
7   KV head count
8   key head dimension
9   value head dimension
10  recurrent convolution width
11  recurrent convolution dimension
12  recurrent state dimension
13  recurrent dt rank
14  K cache type (version 3 only)
15  V cache type (version 3 only)
```

Then it stores:

- `u32[token_count]` checkpoint token IDs.
- `float32[vocab_size]` logits for the next token after that checkpoint.
- For each full-attention layer: a `u32` row count followed by all K and V
  rows in the selected `f16`, `q8_0`, or `q4_0` cache type.
- For each recurrent layer: its `float32` convolution history and recurrent
  state tensors.

Version 1 stores tokens and rebuilds runtime state by prefilling the saved
sequence. MTP sessions write this version so loading reconstructs both the
target state and the shifted MTP history. Other sessions write version 2 or
3 and persist the complete Qwen full-attention and recurrent state.

The logits are raw IEEE-754 `float32` values from the host `q36_session`
buffer. They are saved immediately after the checkpoint tokens so a loaded
snapshot can sample or continue from the exact next-token distribution
without running one extra decode step. MTP draft logits/state are not
persisted; an MTP checkpoint rebuilds both models from its saved tokens.

The tensor payload is q36-specific KV/session state, not a generic
inference graph dump. It is expected to be portable only across compatible
`q36.c` builds for this model layout.

The cache stores checkpoints at four moments:

- `cold`: after a long first prompt reaches a stable prefix, before
  generation.
- `continued`: when prefill or generation reaches the next absolute aligned
  frontier.
- `evict`: before an unrelated request replaces the live in-memory session.
- `shutdown`: when the server exits cleanly.

Cold saves intentionally trim a small token suffix and align down to a
prefill chunk boundary. This avoids common BPE boundary retokenization
misses when a future request appends text to the same prompt. The defaults
are conservative: store prefixes of at least 512 tokens, cold-save prompts
up to 30000 tokens, trim 32 tail tokens, and align to 2048-token chunks.

Continued saves use the same alignment and are written only when the live
graph naturally reaches an absolute frontier. With the defaults this means
roughly every 10k tokens, independent of where the first cold checkpoint
landed, so long generations leave restart points behind without persisting
the fragile final few tokens.

Important knobs:

- `--kv-cache-min-tokens`
- `--kv-cache-cold-max-tokens`
- `--kv-cache-continued-interval-tokens`
- `--kv-cache-boundary-trim-tokens`
- `--kv-cache-boundary-align-tokens`
- `--tool-memory-max-ids`
- `--disable-exact-tool-replay`

The cache directory is disposable. If behavior looks suspicious, stop the
server and remove it. You can investigate what is cached with `hexdump`,
since the KV cache files include the verbatim prompt cached.

## Backends

Q36 is multi-runtime at the source and API level, but Metal and Vulkan are
separate native builds rather than one fat executable. Each build produces the
same command names—`q36`, `q36-server`, `q36-bench`, `q36-agent`, `q36-eval`,
and `q36_test`—linked to the selected graph runtime:

| Platform/runtime | Build | Explicit invocation |
| --- | --- | --- |
| Linux / generic Vulkan | `make` or `make vulkan-generic` | `./q36 --vulkan -p "Hello"` |
| Linux / BC-250 checked Vulkan | `make vulkan-bc250` | `./q36 --vulkan -p "Hello"` |
| macOS / Metal | `make metal` | `./q36 --metal -p "Hello"` |
| CPU reference | `make cpu` | `./q36 --cpu -p "Hello"` |

Building another runtime overwrites those local executable names. Use separate
worktrees or copy build artifacts if Metal, Vulkan, and CPU binaries must be
kept side by side. The `--backend metal|vulkan|cpu` spelling is equivalent to
the short backend switches and is accepted by the CLI, server, agent,
benchmark, and evaluation harness. A Metal-linked binary rejects `--vulkan`,
and a Vulkan-linked binary rejects `--metal`, so deployment mistakes fail
before model loading.

### Vulkan device and compatibility policy

The generic and BC-250 targets contain the same Vulkan backend and SPIR-V.
`vulkan-generic` queries device capabilities and uses portable fallbacks when
the subgroup or 16-bit features required by a tuned kernel are absent. A
BC-250 is recognized at runtime and keeps the current optimized path.
`vulkan-bc250` adds a vendor/device check so a board-specific release artifact
cannot silently run on different hardware.

The generic target removes the hard device identity restriction, not the need
to validate a new GPU. Before calling another Vulkan device supported, run the
isolated kernel suite and the short CPU/GPU parity gate, then record prefill
and decode throughput at the intended context sizes.

### Metal device and compatibility policy

Like DS4, a normal local Metal process uses `MTLCreateSystemDefaultDevice`.
Apple Silicon exposes its unified GPU as one logical Metal device, so Q36 does
not require or expose a device list for M1, M2, M3, M4, or later families.
There is no layer split or multi-Mac distributed mode: one Q36 process owns one
default local Metal device and one model. Run only one large model process at a
time when validating memory and performance.

The Metal binary targets macOS 11. Newer facilities are optional:

* macOS 15 residency sets and `MTLMathMode` are selected only after runtime
  availability checks.
* Older releases use the same shared/no-copy buffers and baseline Metal
  kernels without residency sets.
* Host code defaults to the Apple M1 instruction baseline, so a binary built
  on a newer Mac remains usable on M1. Use
  `NATIVE_CPU_FLAG=-mcpu=native make metal` only for a local-only build.
* Routed MoE prefill follows DS4's expert-major batch-MM dispatch and selects
  the map specialization for the model's actual routed-expert count. If the
  shape, pipeline, thread count, or threadgroup memory is unsupported, Metal
  automatically uses the exact matvec path. `Q36_METAL_MOE_MM=0` forces that
  fallback for diagnostics.
* Cache auto-sizing uses the device's `recommendedMaxWorkingSetSize`, not a
  hard-coded machine-memory percentage.
* All model buffers use unified storage; no discrete-GPU transfer or eGPU
  placement path is provided.

The minimum deployment target proves link compatibility, not every driver and
GPU generation. Release QA must still run on a physical M1 with the oldest
supported macOS, a representative newer Mac, a 16 GB resident configuration,
and a physical 8 GB SSD-streaming configuration.

Metal shaders are compiled at runtime from `metal/*.metal`. Run the CLI,
server, benchmark, and evaluation harness from the Q36 project tree, and ship
the `metal` directory with binary packages. A frontend that changes working
directory before model startup must provide absolute paths through
`Q36_METAL_DENSE_SOURCE`, `Q36_METAL_MOE_SOURCE`,
`Q36_METAL_NORM_SOURCE`, `Q36_METAL_OPS_SOURCE`,
`Q36_METAL_RECURRENT_SOURCE`, `Q36_METAL_KV_SOURCE`, and
`Q36_METAL_ATTN_SOURCE`.

### CPU reference runtime

Do not treat the CPU path as the production target. The CLI and `q36-server`
support the CPU backend for reference/debug use and share the same KV
session and snapshot format as Metal and Vulkan, but normal inference should
use a GPU graph runtime.

## Steering

Q36 can edit attention and FFN outputs with the same projection used by DS4:

```text
y = y - scale * direction[layer] * dot(direction[layer], y)
```

The steering file is a flat native-endian `f32` matrix with one normalized
direction per layer: `64 x 5120` for Qwen3.8/3.6 27B or `40 x 2048` for
Qwen3.6 35B A3B. Positive scales remove the represented
direction; negative scales amplify it. FFN steering defaults to `1` when a
file is supplied without an explicit scale:

```sh
./q36 -p "Write tersely" \
  --dir-steering-file qwen-direction.f32 \
  --dir-steering-ffn 0.8
```

`--dir-steering-attn F` applies the same edit after attention outputs. With no
file, or with both scales set to zero, inference follows the normal path. CPU
and both GPU runtimes apply the same operation during prefill and decode; Metal
and Vulkan keep the matrix resident and project activations in place.

In the CLI and agent REPL, `/steer` displays the FFN scale and `/steer F`
sets it for subsequent tokens, without rebuilding the existing KV cache.
Load with `--dir-steering-ffn 0` to start with steering disabled and enable it
later. Scales range from `-100` to `100`; start near `-1` to `1`.
Each session keeps its own steering strengths; the vector bank is shared.
The agent steers its decisions and prose, while processing prompt text and
generating tool calls without steering to preserve exact arguments and file
contents. CLI and server steering covers both prefill and decode.
The MTP predictor remains unsteered; its drafts are verified by the steered
target model.

See [dir-steering/README.md](dir-steering/README.md) to build a vector for
`gguf/Qwen3.8-27B-UD-IQ3_S.gguf` from the bundled paired prompt sets.

## Testing And Release QA

The release checks are split so quick source tests do not require loading the
model:

```sh
make test                 # unit, parser, protocol, cache and fixture tests
make test-vulkan          # isolated Vulkan kernel coverage
make test-model           # generation, CPU/Vulkan and fusion parity
make test-metal           # Metal unit and isolated numeric kernel coverage
make test-metal-model     # Metal generation, CPU parity, state, and streaming
make test-session-batch   # Vulkan 1/2/4/8-session full-logit/state oracle
make test-server-live     # live HTTP, CORS and Responses API smoke test
make test-webui           # Web UI unit tests (Node.js, no model needed)
make test-server-live-metal # live Metal HTTP/CORS/Responses smoke test
make test-server-live-metal-ssd # same live Metal surface through SSD streaming
make test-server-batching # concurrent requests against one 4-slot server
make test-server-batching-metal # same concurrent server gate on Metal
make test-server-batching-metal-ssd # Metal batching through bounded SSD cache
make benchmark-session-batch # old, 1/2/4/8-slot, and ordered-fallback server runs
make test-streaming       # resident/warm/cold/pressure/full-layer matrix
make benchmark-gate       # conservative BC-250 throughput floor
make release-build-check  # generic/BC-250 Vulkan and CPU builds with -Werror
make release-build-check-metal # Metal release build with -Werror
```

`make test-release` runs the complete sequence, including reference vectors.
The manual hardware, server, agent, long-context, power and sign-off checklist
is in [`QA_BEFORE_RELEASES.md`](QA_BEFORE_RELEASES.md). Distributed inference
is outside Q36's release scope.

## Test Vectors

`tests/test-vectors/qwen3.6-35b-a3b` contains committed short and long-context
continuation vectors captured from llama.cpp with the Qwen3.6 Q8_0 reference
GGUF. Checkpoint-scoped directories keep fixtures from different model
revisions separate. They are the default offline reference, so building and
testing Q36 does not require llama.cpp or GGML libraries.

The fixtures use greedy decoding, thinking disabled, and `top_logprobs=20`.
Private vectors are generated by the optional local llama.cpp capture tool and
compared by token bytes, so tokenizer/template or attention regressions show
up before they become long generation failures.

The reference workflows are:

```sh
make test-reference        # tracked, reproducible results
make test-vectors-local LLAMA_BUILD_DIR=llama.cpp/build
make test-reference-local  # ignored local capture
make test-llama LLAMA_BUILD_DIR=llama.cpp/build  # optional live comparison
```

Hosted official-model comparisons use OpenRouter and the native Q36 scorer;
see `gguf-tools/quality-testing/README.md`. The API key and private response
captures stay outside version control.

## Debugging Notes

When a generation looks wrong, three small tools are usually enough to get
a first answer:

```sh
./q36 --dump-tokens -p "..."
./q36 --dump-logprobs /tmp/out.json --logprobs-top-k 20 --temp 0 -p "..."
./q36-server --trace /tmp/q36-trace.txt ...
```

- `--dump-tokens` tokenizes the `-p` or `--prompt-file` string exactly as
  written, recognizes Qwen protocol specials (`<|im_start|>`, `<|im_end|>`,
  `<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<tools>`,
  `</tools>`), and then exits before inference starts. Useful for confirming
  that a tool block tokenizes the way you expect: the closing marker
  `</tool_call>` is plain ASCII and splits as ordinary BPE tokens, not as a
  single special.
- `--dump-logprobs` stores a greedy continuation with the top local
  alternatives at each step, which helps separate sampling choices from
  logit/model issues.
- `q36-server --trace` writes the rendered prompts, cache decisions,
  generated text, and tool-parser events for a whole agent session.

## Logo

The QwarkStar logo is an AI-edited version of the DwarfStar logo.
DwarfStar is designed by hand by Salvatore Sanfilippo, made more
graphical with AI, and manually reworked by Ben Gnomino, whose human touch made it rock. As always, all credits go to Salvatore.

## Extended evaluation and regression checks

`q36-eval --suite hard-smoke` selects 12 harder cases; `--suite hard` selects all
50. The original 92 cases remain the default `core` suite. Use `--list-cases`,
`--validate-cases`, and `--source`, `--domain`, or `--case-id` to inspect or filter
cases. `--retry-incomplete` retries answers that hit their output limit.
[EVAL_DATA.md](EVAL_DATA.md) records sources and licenses.

Server requests may set `ignore_eos: true` with an explicit `temperature: 0` for
fixed-length greedy generation. Context limits, stop strings, and client stops
still apply. Cache usage is reported in each API's usage fields.

See [tests/REGRESSIONS.md](tests/REGRESSIONS.md) for serial model and client checks.
