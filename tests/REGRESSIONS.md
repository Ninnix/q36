# Frontend and memory regressions

Run model-free checks with `make test`: atomic file operations, bounded reads and
searches, independent shell jobs, private password entry, terminal redraw and
fragmented input, tool parsers, compaction boundaries, image ownership and cache
limits, prefix parsing, quantization guards, scoring validators, and eval graders.
No additional Python packages are needed.

Model tests use the standard Qwen3.6 MoE unless `--model` overrides it. Run one
heavy process at a time, as required by [SPEEDUP.md](../SPEEDUP.md).

```sh
python3 tests/test_server_vision.py --output /tmp/q36-vision-check
python3 tests/test_server_vision.py --slots 2 --output /tmp/q36-vision-batched
python3 tests/test_server_vision.py --pi /usr/bin/pi --output /tmp/q36-pi-vision
python3 tests/test_agent_compaction.py \
  --model gguf/Qwen3.6-35B-A3B-AntirezExperts-IQ2XXS-gateup-Q2K-down-Q8rest.gguf \
  --ctx 8192 --tokens 2400 --output /tmp/q36-compaction-check
python3 tests/test_agent_sessions.py \
  --vision gguf/Qwen3.6-35B-A3B-mmproj-F16.gguf \
  --output /tmp/q36-session-check
Q36_TEST_THREADS=16 ./q36_test --gpu-cpu-parity --ssd-streaming-parity --case short
```

After changing shared CPU arithmetic, also run the optional live reference:

```sh
model=gguf/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf
make q36_llama_test LLAMA_BUILD_DIR=llama.cpp/build-q36-cpu
Q36_TEST_THREADS=16 ./q36_llama_test --llama-quant-dots --model "$model"
Q36_TEST_THREADS=16 ./q36_llama_test --llama-parity-seq --model "$model" --case short
```

On November 10, 2026, `Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` matched llama.cpp
`351025ec532f825c7eb6c7ae1017d8e6d46d0cc9` exactly at all nine short-suite
positions with 16 CPU threads. This covers the Italian top-5 failure against
the updated reference. The fix aligns GDN epsilon placement, BF16 activations,
quantized reductions, SiLU, RoPE and attention probability normalization.
The existing parity gates and prompts are unchanged. llama.cpp remains an
optional test/reference dependency.

The same model also passes all nine CPU/Vulkan positions with the default
Q8_0 key and Q4_0 value caches. Vulkan norms round their squares to F32 before
accumulating in double, matching the reference; host fallbacks reuse the CPU
norm and activation routines.

`Qwen3.8-27B-UD-IQ3_S.gguf` also passes both nine-position comparisons.
The Qwen3.6 35B IQ2XXS/Q2K/Q8 and IQ4XS/Q4K/Q8 models also match the same
llama.cpp revision exactly at all nine positions. Both pass CPU/Vulkan parity;
the IQ4 model uses SSD streaming. F32 router projections, Q8 dot reductions
and routing normalization preserve the reference rounding boundaries.
Flash Next passes both comparisons in SSD streaming; its updated results
are recorded in [the model measurements](../speed-bench/qwen38-flash-next/README.md#correctness).
Its GPU router preserves the CPU vector exponential, reduction order and
ties-to-even conversion. Ordered ties, 64 batched router cases and the
real-weight routed-expert comparisons pass without relaxing their exact checks.

Resident Vulkan speed was checked against `c29767c` on the BC-250 at 128,
2,048, 3,968 and 4,096 tokens, with chunk 128 and warm-up at every frontier.
The two dense models and IQ2 MoE retain identical Vulkan allocation peaks.
The initial IQ3_S 4K slowdown did not reproduce in the targeted repeats:

| IQ3_S at 4K | Prefill t/s before → after | Plain decode t/s before → after |
| --- | ---: | ---: |
| Initial 128-token extension, 16 decode tokens | 125.14 → 107.30 | 24.99 → 19.38 |
| Repeated 128-token extension, 64 decode tokens | 127.07 → 129.97 | 24.85 → 24.92 |
| Fresh 4,096-token prefill, 64 decode tokens | 144.41 → 149.83 | 25.15 → 25.13 |

The old and corrected IQ3_S builds produce different 4K logits, including
top-1; reference parity was checked on the short suite above. Within each
build, fresh prefill and snapshot restore reproduce the entire 4K vector
byte for byte. This distinguishes the numerical changes from restore drift.

The full unit suite, SSD cache/fusion/queue checks, dense and MoE session/tool
checks, greedy and sampled MTP, live and batched HTTP server, dense and MoE
vision, agent save/restore/cancellation and 8K agent compaction all passed.
Raw captures, timings, temperatures and hashes stay outside the repository.

`--pi-api openai-responses` selects an individual Pi API; `--ctx` sets the
vision server context. Use `--resident` for dense models, which have no routed
expert cache. The compaction driver accepts `--prefix-file FILE` to
check that every compacted transcript retains the complete initial prefix.
For the thinking-mode code fixture, use `--think --tokens 6000` at context 8192
to leave room for both reasoning and all 120 functions.

The vision test saves each request and response, checks fresh and reused image
history in Chat Completions, Responses and Anthropic Messages, replaces images
with identical geometry, appends another image, exercises image tool results,
and checks fixed-length generation and invalid inputs. It also checks exact pending
tool IDs, image-preserving tool-only continuations, appended images, and isolation
from unknown tool IDs. `--pi` additionally runs
a real coding client that reads two images, edits Python, runs it, and checks the
result against an independent oracle. Pi settings are isolated under the output
directory and point only at the local server.

The compaction test uses a real PTY. It fills the context, verifies tool execution
after compaction, compiles 120 generated C functions across a generation boundary, checks
the carried output budget, rejects an oversized user input, and checks recovery.
`Q36_AGENT_CACHE_DIR` isolates its session cache without changing the user's home.
Terminal ANSI and model traces remain in its output directory for inspection.

The session test saves and restores a chat, exits during generation, checks
idle CPU use at the save prompt, and verifies that cancelling an unsupported
image-session save returns to a usable chat. `--web` additionally permits
browser startup and checks Google result links and a real page visit.

The two image fixtures contain synthetic ticket text and colored geometric shapes.
