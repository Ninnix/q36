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
