# Qwen 3.6 MoE Test Vectors

These vectors are captured directly from `llama.cpp` through a tiny local
`libllama` capture tool, using the same rendered Qwen chat prompt shape as
`q36 --nothink`, greedy decode, and `top_logprobs=20`. The capture script now
defaults to
`gguf/Qwen3.6-35B-A3B-Q8_0.gguf`;
regenerate the fixtures before treating this directory as a refreshed Q8_0
parity corpus.

`qwen3.6-35b-a3b/` is the tracked parity corpus. Checkpoint-scoped directories
keep fixtures from different model revisions from being mixed. Each manifest records the exact
template used for each capture; the current fixture predates the fixed
`--nothink` render, while new captures use it. Normal builds and tests do not
need a llama.cpp checkout.

Files:

- `<checkpoint>/prompts/*.txt`: exact user prompts.
- `<checkpoint>/llamacpp/*.llamacpp.json`: raw llama.cpp continuations and top-logprobs.
- `<checkpoint>/llama.vec`: compact C-test fixture generated from the llama.cpp JSON.

Check the tracked raw JSON against its compact form without a model:

```sh
make test-reference
```

With a model present, the same command also scores the stored continuation
against Q36. `Q36_TEST_MODEL` selects a different local GGUF.

The C runner consumes `llama.vec` directly:

```sh
Q36_TEST_MODEL=gguf/Qwen3.6-35B-A3B-Q8_0.gguf ./q36_reference_test --logprob-vectors
```

The runner uses the normal non-quality path and compares q36 against the
llama.cpp reference fixture token-by-token by byte sequence.

`llama.vec` is intentionally trivial to parse from C: each case points to a
prompt file and each expected token is hex-encoded by bytes. The llama.cpp JSON
files remain in the tree so the compact fixture can be audited against the raw
reference capture.

For a private refresh, keep a llama.cpp checkout in `./llama.cpp` and build
static CPU libraries:

```sh
cmake -S llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF
cmake --build llama.cpp/build -j
```

Then run:

```sh
make test-vectors-local LLAMA_BUILD_DIR=llama.cpp/build
make test-reference-local
```

This writes `tests/test-vectors/local/`, which is ignored by Git. Use
`LLAMA_CPP_DIR` and `LLAMA_BUILD_DIR` when the checkout or build lives
elsewhere. Only run `make test-vectors-refresh` when intentionally replacing
the tracked corpus.

Live CPU-vs-llama.cpp logit gates remain available as an optional binary:

```sh
make test-llama LLAMA_BUILD_DIR=llama.cpp/build
make test-llama-long LLAMA_BUILD_DIR=llama.cpp/build
```

OpenRouter reference collection and scoring are documented in
`gguf-tools/quality-testing/README.md`; API keys and raw private captures must
not be committed.
