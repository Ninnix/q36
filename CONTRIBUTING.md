# Contributing

QuarkStar changes should be tested against the failure mode they can
realistically affect. The project has two regression tracks: correctness and
speed. Include the commands you ran, the machine/backend, the model quant, and
any notable failures in the PR or commit notes.

Do not send changes affecting inference, Vulkan kernels, prompt rendering, KV
cache, or the server API without checking that the resulting code is still
correct. Speed regressions are acceptable only when they fix an important
correctness bug and the tradeoff is explicit.

## Correctness Regression Tests

Build both Vulkan targets first. The generic binary still selects the BC-250
fast path automatically when it sees that device; the checked target rejects
other devices at startup.

```sh
make clean
make vulkan-generic
make vulkan-bc250
```

The C test runner is `q36_test`. Running it without arguments is equivalent to
`--all` and includes model-backed groups. The default Makefile target runs the
model-independent evaluator and agent tests plus selected Q36 unit, cache,
protocol, fixture, and server groups:

```sh
make test                  # default model-independent suite
./q36_test --all           # every registered test; model/Vulkan required
```

Useful narrower checks:

```sh
./q36_test --server
./q36_test --logprob-vectors
./q36_test --long-context
./q36_test --tool-call-quality
./q36_test --qwen-tool-call-quality
./q36_test --vulkan-kernels
./q36_test --gpu-cpu-parity
```

What they cover:

- `--server`: request parsing, Qwen chat rendering, streaming, native tool-call
  parsing, thinking controls, disk-KV cache bookkeeping, and other server-side
  logic. This is the best quick check for API and prompt-rendering changes.
- `--logprob-vectors`: compares local token bytes and top-logprob slices against
  the llama.cpp reference fixtures in `tests/test-vectors/`. This catches
  tokenizer, template, attention, and logits regressions.
- `--long-context`: runs the long-context security continuation regression from
  `tests/long_context_security_prompt.txt` unless `Q36_TEST_LONG_PROMPT` is set.
- `--tool-call-quality`: exercises actual model behavior for tool-call emission
  in both fast and exact paths.
- `--qwen-tool-call-quality`: exercises native Qwen tool-call emission through
  the OpenAI-compatible request path.
- `--vulkan-kernels`: isolated Vulkan kernel numeric checks.
- `--gpu-cpu-parity`: compares live CPU and Metal/Vulkan logits on the same
  generated suffix with Q36 top-1 and top-k overlap gates.

The runner defaults to the q36 model path compiled into `q36.h`. Override paths
and fixtures when needed:

```sh
Q36_TEST_MODEL=/path/to/model.gguf ./q36_test --logprob-vectors
Q36_TEST_VECTOR_FILE=/path/to/llama.vec ./q36_test --logprob-vectors
Q36_TEST_LONG_PROMPT=/path/to/prompt.txt ./q36_test --long-context
Q36_TEST_VECTOR_CASE=short ./q36_test --gpu-cpu-parity
```

For CPU portability, verify that the CPU-only target still builds:

```sh
make cpu
```

The CPU backend is a reference/debug path, not the production performance
target. Avoid large CPU inference runs unless that is the change being tested.

## Reference Vector Refresh

The default reference test consumes committed llama.cpp results and does not
link llama.cpp:

```sh
make test-reference
```

Keep experimental captures out of Git:

```sh
make test-vectors-local LLAMA_BUILD_DIR=llama.cpp/build
make test-reference-local
```

The local capture lives under the ignored `tests/test-vectors/local/`
directory. To run the live CPU-vs-llama.cpp gates, build the optional runner:

```sh
make test-llama LLAMA_BUILD_DIR=llama.cpp/build
make test-llama-long LLAMA_BUILD_DIR=llama.cpp/build
```

Only `make test-vectors-refresh` updates the tracked reference corpus. Review
its JSON, manifest, and compact vector diff together.

## Speed Regression Tests

Use `q36-bench` for throughput regressions. It reports instantaneous prefill and
generation speed at context frontiers, not one whole-run average. Prefill is
incremental: each row measures only the newly processed suffix since the
previous frontier.

Short smoke benchmark:

```sh
./q36-bench \
  --prompt-file tests/long_context_story_prompt.txt \
  --ctx-start 32 \
  --ctx-max 64 \
  --ctx-alloc 128 \
  --gen-tokens 8
```

Before/after backend benchmark:

```sh
./q36-bench \
  -m gguf/Qwen3.6-35B-A3B-AntirezExperts-IQ2XXS-gateup-Q2K-down-Q8rest.gguf \
  --prompt-file tests/long_context_security_prompt.txt \
  --ctx-start 2048 \
  --ctx-max 32768 \
  --step-incr 2048 \
  --gen-tokens 128 \
  --csv /tmp/q36-speed.csv
```

Use the same machine, backend, model file, context sweep, power/thermal state,
and background load when comparing two commits. For Vulkan work, run at least
one before/after CSV and compare both `prefill_tps` and `gen_tps`. Generation is
greedy and skips EOS so each frontier gets the same number of generated tokens.

## Reporting Session Bugs

For debugging a failing generation, keep the trace:

```sh
./q36-server --trace /tmp/q36-trace.txt ...
```

Attach the trace, command line, model file, backend, context size, and whether
disk KV cache was enabled.
