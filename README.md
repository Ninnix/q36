<p align="center">
  <img src="logo.svg" alt="QuarkStar logo" width="220">
</p>

**QuarkStar** is a small native inference engine for **Qwen3.6-35B-A3B**,
**Qwen3.8-27B**, and **Qwen3.8 Flash Next**.
It is self-contained and deliberately narrow, not a general GGUF runner.
Use the GGUF files the project provides; they are part of the project itself.
The main paths are model-specific Vulkan and Metal graph executors with
Q36-specific loading, prompt rendering, tool calls, KV state, HTTP server,
and coding agent. The repository also includes tools and data for GGUF,
imatrix, quality, and speed.

This project would not exist without **DwarfStar**, **llama.cpp and GGML**, make
sure to read the acknowledgements section, a big thank you to Salvatore Sanfilippo
aka Antirez and Georgi Gerganov and all the other contributors.

Supported runtimes:

* **Vulkan** on Linux. Its current fast path was developed and tuned on the
  AMD BC-250 with RADV. The generic backend is untested on other GPUs; use a
  coding agent to adapt and validate it for your hardware.
* **Metal** on Apple Silicon M1 or newer. A 16 GB Mac can run the default
  release model resident; 8 GB Macs are intended to use SSD streaming.
* **CPU** as a portable reference and debugging runtime, not the normal
  production path.

Model support is intentionally opportunistic. The project follows the best open
weights for useful local machine sizes, especially 16 GB machines and 24/32 GB
workstations. A model may be removed when a better replacement arrives.

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
  automatically. Follow [BC250.md](docs/BC250.md) for the RADV, UMA, kernel-memory,
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
* BC-250 is the perfect machine for that: cheap, a powerful GPU, fast memory and unified memory.
* Qwen3.8-27B and Qwen3.6-35B-A3B tolerate aggressive quantization. The MoE
  model uses Antirez's routed-expert recipe.
* Qwen3.6-35B-A3B is fast as hell and can potentially run even on a toaster, Vulkan is compatible on paper with a huge range of devices, and with DZN and Lavapipe it opens up some very interesting possibilities in the future.
* Compressed KV caches and fast local SSDs make long contexts practical.
* The idea of an inference system specialized for a few models.

# AI full disclosure

This software is developed with strong assistance from AI coding agents and with humans leading the ideas, testing, and debugging. We say this openly because it shaped how the project was built. If you are not happy with AI-developed code, this software is not for you. The acknowledgement below is equally important: this would not exist without llama.cpp and GGML, largely written by hand.

## Acknowledgements

### To antirez and ds4

QuarkStar is essentially a port of [DwarfStar](https://github.com/antirez/ds4),
redesigned for the Vulkan runtime and retargeted at Qwen3.8-27B and Qwen3.6-35B-A3B. The Vulkan
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

### To Unsloth and ISTA

The supported dense Qwen3.8-27B weights come from
[Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) and
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF).
Thank you to both teams for making these quantized models available:
Unsloth provides the UD-IQ3_S and UD-Q6_K releases, and ISTA-DASLab provides
the smaller GSQ-RCO IQ3_XXS release. Their work gives QuarkStar dense-model options for
different memory budgets.

Download the 20.5 GiB Q6_K model with `./download_model.sh 27b-q6`.
It supports distribuited inference
[two-board Vulkan pipeline with MTP](docs/DISTRIBUTED.md#qwen38-27b-q6_k).

## Status

The software is currently very fast changing. Consider it alpha quality.
Before each release, a big QA run is executed, however instabilities
are definitely possible.

# How to use this project?

Like Salvatore with DwarfStar, I believe AI changes how projects can be
shipped and used. The main differences today are:

1. Users can make significant changes with less effort and cost, even without
   deep knowledge of the domain. For example, a QuarkStar user can ask a coding
   agent to improve prefill and generation speed on their hardware, preserve
   correctness, and run a deep QA pass.
2. Software can provide working examples for its main use cases without
   covering every possible setup. QuarkStar's Vulkan and Metal implementations
   can guide adaptations for another GPU, model, or memory budget.

The project aims to work for the featured models and hardware. If you have
access to coding agents, consider using one to explore the code, make changes,
and build a setup for your needs. Features we do not ship or document may be
straightforward to add for your particular use case.

## Start Here

```sh
git clone https://github.com/Ninnix/q36.git
cd q36
```

Choose your build. The platform guides cover prerequisites, memory sizing,
and hardware-specific setups:

| Platform guide | Build |
| --- | --- |
| [BC-250](docs/BC250.md) | `make` |
| [Metal on Apple Silicon](docs/METAL.md) | `make metal` |

For a first run on a 16 GB machine, download Qwen3.6-35B-A3B Q2:

```sh
./download_model.sh q2
```

Downloads go in `gguf/`. Repeat the command to resume an interrupted download.
The downloader verifies the exact file size and SHA-256, including for existing
files. If either check fails, it stops and keeps the file for inspection; move
that file aside before retrying. Verification requires `sha256sum` or `shasum`.
Leave memory for the context and runtime buffers as well as the model.
See [other models](docs/MODELS.md) or use [SSD streaming](docs/SSD_STREAMING.md)
on a smaller Mac.

## Qwen3.8 Flash Next

The `qwen4exp` text model runs on CPU and Vulkan. The GSQ-RCO Coder pack with
256 experts uses Q2_0 down projections and its original IQ4_NL n-gram shard.
Keep both GGUF files in the same directory. The loader accepts the differing
filenames in this pack when exactly one compatible second shard is present.

```sh
./download_model.sh flash-next
model=gguf/Qwen3.8-Flash-Next-GSQ-RCO-Coder-22GB-IQ2_XXS-00001-of-00002.gguf
./q36 -m "$model" --cpu --ssd-streaming --ctx 4096 --nothink -p "Hello"
./q36 -m "$model" --vulkan --ssd-streaming --ctx 4096 --nothink -p "Hello"
```

The complete download is about 50.8 GB: 22.0 GB of transformer weights and
28.8 GB of disk-backed n-grams. Both shards come from the same model pack.

N-gram rows are read with `pread`. Vulkan also streams dense weights and selected
experts through two alternating 8 MiB banks in a shared 16 MiB slab. Parallel
reads overlap GPU execution; packed projections reuse the existing K/IQ decode
kernels. Activations and recurrent state stay on the GPU. CPU projections reuse
the existing quantizers and thread pool. SSD-streaming prefill processes one
token at a time; resident Vulkan supports batches. Attention caches use F32. Checkpoints
reuse the token-replay format, which restores n-gram history as well as GDN and
attention state. The existing [two-board pipeline](docs/DISTRIBUTED.md#qwen38-flash-next)
supports this model, with PLE/n-grams only on the coordinator and resident suffix
weights on the worker. MTP, vision, steering and Metal are not implemented for
this architecture.

The graph follows [DS4](https://github.com/antirez/ds4/blob/b3f50180ec2c7324892be1f50bc83555ccb8d5e7/ds4.c).
Its current loader requires BF16 n-grams and does not directly accept this
quantized/pruned pack. Comparisons were rechecked November 10, 2026, against
[llama.cpp](https://github.com/ggml-org/llama.cpp/tree/351025ec532f825c7eb6c7ae1017d8e6d46d0cc9).
On the BC-250, both strict parity suites pass with 16 CPU threads and all three
short prompts (Italian fact, code and reasoning), as required by `SPEEDUP.md`:

| Comparison | Checked positions | Top-1 matches | Strict gates |
| --- | ---: | ---: | --- |
| CPU vs llama.cpp CPU | 9 | 9/9 | PASS |
| CPU vs Vulkan, SSD streaming | 9 | 9/9 | PASS |

The existing gates are unchanged: identical top-1, at least 4/5, 15/20 and 40/64
top-k overlap, and at most 8.0 absolute error on reference top-20 logits. Norms,
BF16 reductions and quantized activation rounding follow the reference operation
order. Quantized matrix products retain integer dot products until scale
application. Metrics and validation results are retained in the
[benchmark summary](speed-bench/qwen38-flash-next/README.md#correctness).
Raw logs are generated locally and excluded from the repository.

```sh
Q36_TEST_THREADS=16 ./q36_llama_test --llama-parity-seq --model "$reference_model" --case short
Q36_TEST_THREADS=16 Q36_TEST_SSD_STREAMING=1 ./q36_test --gpu-cpu-parity --model "$model" --case short
```

Build `q36_llama_test` against the llama.cpp revision above. For that loader,
`reference_model` must name the same weights through matching split filenames;
the recorded run used symlinks `model-00001-of-00002.gguf` and
`model-00002-of-00002.gguf` to the two original files.

Short SSD-mode benchmarks on the BC-250, with 16 threads and 16 generated
tokens, compare the original parity-correct streaming path with these changes:

| Prompt tokens | Original prefill | Current prefill | Original decode | Current decode |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 0.74 t/s | 2.49 t/s | 0.68 t/s | 2.37 t/s |
| 128 | 0.65 t/s | 2.43 t/s | 0.60 t/s | 2.24 t/s |

The 128-token decode gain is 3.7x. At 32 tokens, process peak RSS changed from
123.4 to 124.3 MiB; the Vulkan allocation peak stayed at 144.6 MiB. The shared
weight slab remains 16 MiB. OS page cache was not controlled, so these numbers
do not establish sustained cold-SSD throughput. The final runs stayed between
43.6 and 47.8 C. Long 2K/4K benchmarks are deferred at the user's request.
See the [SSD measurements](speed-bench/qwen38-flash-next/README.md#ssd-streaming-on-one-board).

The resident two-BC-250 pipeline defaults to a 50/50 layer split, with any odd
remaining layer on the coordinator. This model has 48 layers: main `0:23`,
worker `24:output`. The launcher makes no hardware-based split decisions.
It measures 160.01 t/s prefill and 30.09 t/s plain decode at 128 tokens;
133.40/22.96 t/s at 3,968 tokens; and 130.06/22.71 t/s at 7,808 tokens.
Decode stays near 23 t/s from 2K through 8K: the matched control fell from
22.88 to 16.17 t/s. QSA caches completed key transforms in existing storage
and uses radix selection instead of quadratic ranking. Prompt resets reuse
GPU buffers, and Vulkan sessions avoid unused CPU KV mirrors. GPU tensor
peak is unchanged; worker peak RSS falls from 354.95 to 71.37 MiB.
Longer prefill points measure 1,920-token extensions with an 8,192-token
allocation. Both stages retain their weights; only main gathers PLE/n-grams
from SSD. All five full-vocabulary frontier vectors and nine long-context
session/replay checks are bit-identical to the control. CPU/Vulkan parity
passes all existing gates. The 150/30 t/s targets remain unmet at long
contexts. Flash Next MTP is unavailable. See
[launch commands](docs/DISTRIBUTED.md#qwen38-flash-next), the
[current context-scaling results](speed-bench/qwen38-flash-next/README.md#resident-distributed-pipeline).

Use the shared benchmark for a short SSD-mode check:

```sh
Q36_THREADS=16 ./q36-bench -m "$model" --vulkan -t 16 --ssd-streaming \
  --prompt-file tests/long_context_story_prompt.txt \
  --ctx-start 32 --ctx-max 128 --step-incr 96 --ctx-alloc 256 \
  --gen-tokens 16 --warmup
```

The optional next-layer router experiment follows
[DwarfStar's Kimi K3 prefetch design](https://github.com/antirez/ds4/blob/bc9334c7c75fe346ab0fec0e1b6defe71a5003c1/docs/KIMI_K3.md).
`Q36_SSD_ROUTE_PREFETCH=4` predicts four experts and queues cancellable SSD hints
while the current FFN runs. The actual router still selects every expert.
It supports Flash Next and Qwen3.6 MoE and remains off by default: final short
trials showed no consistent gain. Flash Next reached 3.79 t/s with hints versus
3.93 t/s in the final warm control; Qwen3.6 controls were 5.40/5.60 t/s versus
4.57/5.64 t/s with hints. Cache warming strongly affected the first Flash Next
control. Hint-on/off logit vectors were bit-identical at all four checked token
positions on both models. Set `Q36_SSD_ROUTE_PROFILE=1` for prediction counters;
measure performance separately with profiling disabled. The prediction omits
the intervening FFN and attention updates, and Flash Next also omits the next
HC mix. Correct membership in the active expert set is different from matching
the top four by score. These are architectural approximations, not the same
prediction accuracy measured for DwarfStar's AttnRes model.

`make test-qwen4` tests Q2_0 packing, disk n-gram reads, EOS history and sparse
attention selection beyond 2048 tokens. Build `tests/qwen4_kernels` for Vulkan
matrix, batching, QSA and cache-growth checks. Use the shared parity commands
above and [distributed session/replay tests](docs/DISTRIBUTED.md#validation)
for full-model validation; `make test-ssd-prefetch` checks read-ahead queue
cancellation, descriptor ownership and shutdown.

## Everyday Use

Once built and with a model downloaded:

```sh
./q36
./q36 -p "Explain Redis streams in one paragraph."
./q36-agent
./q36-server --ctx 32768
```

The default model is `gguf/Qwen3.6-35B-A3B-IQ2XXS-Q2K-Q8.gguf`.
Pass `-m FILE` to choose another supported model. Main-model downloads
also update `q36moe.gguf` for older scripts. Run commands from the repository
root; `q36-agent --chdir /path/to/project` sets the agent's tool directory
after loading its model and runtime assets.

The server listens at `http://127.0.0.1:8000` by default; see [serving](docs/SERVER.md)
for API access and multiple sessions.

The interactive CLI keeps a multi-turn conversation. Use `/help`, `/read FILE`,
`/ctx N`, and `/quit`. Ctrl+C interrupts generation and returns to the prompt.
Run each binary with `--help` for its full options.

## Update

To update QuarkStar from source, run inside the `q36` directory:

```sh
git pull
make clean
make vulkan-bc250
```

On Apple Silicon, use `make metal` for the last command.

## Native Agent

`q36-agent` runs inference directly, without a separate HTTP server. It keeps
the transcript and live model state together, shows prefill progress, and
uses Qwen's native tool format. Built-in tools cover files, search, shell
commands, processes, and the web.

```sh
./q36-agent
./q36-agent --chdir /path/to/project
./q36-agent --non-interactive -p "Inspect the tests and fix the failure."
```

Sessions are stored in `~/.q36/kvcache`. Use `/save`, `/list`, and
`/switch <sha>` to save and resume work. Compatible KV snapshots avoid
rebuilding the prompt. `/strip <sha>` keeps the text and title while removing
the large KV payload; switching to that session requires prefill. Sessions
containing images cannot yet be saved.

Use `/hints on` for occasional short explanations of the programming choices
behind the work, and `/hints off` to stop them. See the [agent guide](docs/AGENT.md)
for thinking budgets, context, prefix files, terminal input, and compaction.

For Pi, OpenCode, or Claude Code, use `q36-server` and follow
[client setup](docs/CLIENTS.md).

### Models, images, and speculation

[Models and vision](docs/MODELS.md) lists the supported GGUFs, download
targets, and matching encoders. Vision currently requires Vulkan. With the
encoder passed as `--vision FILE`, use `/read image.png` in the CLI or
`view_image` in the agent.

Speculative decoding is opt-in with `--mtp`. Both supported model families
include an embedded head. It can improve generation, but the gain depends on
the workload. Read [speculative decoding](docs/SPECULATIVE_DECODING.md) for
setup and the difference between opportunistic and ordinary sampling.

### Output and power

Thinking is enabled by default. Use `--nothink` or `/nothink` for direct
answers, and `--think` or `/think` to enable it again. The normal sampling
defaults are temperature 1, top-p 1, and min-p 0.05; `--temp 0` selects
greedy output. Qwen3.8 also has reasoning effort levels; see the
[agent guide](docs/AGENT.md#generation-and-thinking) and
[server controls](docs/SERVER.md#thinking-modes).

For BC-250 clocks and thermal limits, use the system governor described in
the [hardware guide](docs/BC250.md#4-install-the-gpu-governor).

Directional steering can adjust the model's behavior during generation.
Load a vector with `--dir-steering-file FILE`; `/steer F` changes its scale
for subsequent tokens. See [steering](docs/STEERING.md) for runtime behavior
and vector extraction.

## Server

`q36-server` exposes OpenAI-compatible chat and Responses APIs and an
Anthropic-compatible Messages API. It supports streaming, tools, independent
sessions, and disk KV checkpoints. See [serving](docs/SERVER.md) for setup.

QuarkStar intentionally keeps the UI separate to keep the engine small and
self-contained. Use an existing UI, or build your own, and connect it through
the server. [UI setup](docs/UI.md) includes an Open WebUI example.

## Capability Evaluation

`q36-eval` runs embedded capability regression tests against a real GGUF.
These are QuarkStar integration checks, not official leaderboard scores.

```sh
./q36-eval --trace /tmp/q36-eval.txt
./q36-eval --suite hard-smoke
./q36-eval --suite hard --retry-incomplete
```

The default suite is `core`; `--suite all` runs core and hard cases.
`--list-cases` lists tests without loading a model. `--plain` selects
non-interactive output, and `--regrade-trace FILE` scores a saved trace
without generating again. Sources and licenses are in [EVAL_DATA.md](EVAL_DATA.md).
See [testing](docs/TESTING.md) for regression and release checks.

## Speed

The recorded BC-250 sweeps use resident Qwen3.6 Q2 and Qwen3.8 IQ3 models,
Q8_0 keys / Q4_0 values, and 128 greedy generation tokens at each frontier.
They are baselines, not a fresh benchmark of every commit.

![BC-250 Qwen3.6-35B-A3B throughput](speed-bench/bc250-20261025/qwen36_ts.svg)

See [performance and benchmarking](docs/PERFORMANCE.md) for the complete
tables, dense-model charts, Metal results, measurement conditions, and commands.

## Detailed Guides

- [BC-250](docs/BC250.md): Linux, RADV, UMA memory, governor, and build setup.
- [Vulkan](docs/VULKAN.md): requirements, native builds, and device validation.
- [Metal](docs/METAL.md): Apple Silicon setup, memory, and compatibility.
- [Models and vision](docs/MODELS.md): downloads, quantization, and encoders.
- [Native agent](docs/AGENT.md): sessions, thinking, context, and tools.
- [Distributed inference](docs/DISTRIBUTED.md): split a supported model across two
  Linux machines over Ethernet, Wi-Fi, or other TCP networks.
- [SSD streaming](docs/SSD_STREAMING.md): run the MoE model with a bounded expert cache.
- [Speculative decoding](docs/SPECULATIVE_DECODING.md): MTP verification and sampling.
- [Serving](docs/SERVER.md): APIs, images, and multiple sessions.
- [Coding agent clients](docs/CLIENTS.md): Pi, OpenCode, and Claude Code.
- [UI setup](docs/UI.md): separate interfaces connected through the server.
- [KV caches](docs/KV_CACHE.md): precision, disk checkpoints, and file formats.
- [Steering](docs/STEERING.md): activation edits and vector extraction.
- [Performance](docs/PERFORMANCE.md): benchmarks and recorded baselines.
- [Testing and development](docs/TESTING.md): regression tests, debugging, and offline tools.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before sending a pull request.

## Logo

The QwarkStar logo is an AI-edited version of the DwarfStar logo.
DwarfStar is designed by hand by Salvatore Sanfilippo, made more
graphical with AI, and manually reworked by Ben Gnomino, whose human touch made it rock. As always, all credits go to Salvatore.
