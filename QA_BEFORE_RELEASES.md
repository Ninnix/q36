# QA Before Releases

This is the release gate for QuarkStar. Run it before tagging or publishing a
release build. It focuses on the paths that have historically regressed:
Qwen prompt rendering, Metal and Vulkan kernels and fusions, SSD expert
streaming, long context state, disk KV checkpoints, server protocols, and the
native agent.

Do not run multiple model processes at once. Record the commit, BC-250 firmware
and Mesa version, GGUF filename and checksum, context size, cache types, and all
non-default flags for every model-backed run.

## 1. Repository And Build Sanity

- Start from a clean tree: `git status --short`.
- Run `git diff --check` before committing.
- Build generic Vulkan, BC-250-checked Vulkan, and CPU release configurations
  with warnings promoted to errors:
  `make release-build-check`.
- On macOS, run `make release-build-check-metal`, inspect every release binary
  with `otool -l`, and require the documented minimum macOS deployment target.
- Do not fix release warnings with a global suppression. A narrow suppression
  is acceptable only for a test translation unit that intentionally includes
  another C file and leaves unrelated static functions unused.
- Verify `./q36 --help`, `./q36-server --help`, `./q36-agent --help`,
  `./q36-bench --help`, and `./q36-eval --help` render without stale options.
- Confirm the release binaries use the intended Vulkan loader and Mesa RADV
  driver with `ldd ./q36` and `vulkaninfo --summary`.
- Confirm a Metal build reports the expected Apple GPU, unified memory, and a
  nonzero `recommendedMaxWorkingSetSize`. A normal single-Mac process must use
  the system default Metal device, matching DS4's local-device policy.

## 2. Core Regression Tests

- Run `make test`.
- Run the isolated Vulkan numeric suite: `make test-vulkan`.
- Run the model-backed suite: `make test-model`.
- Run the tracked reference vectors: `make test-reference`.
- Run `./q36-eval --self-test-extractors` explicitly after evaluator changes.
- Any assertion, hang, non-finite logit, invalid tool call, or unexplained skip
  is a release blocker.

## 3. Continuation Quality Gates

Tokenizer, template, routing, quantization, KV, attention, or output-head
changes require teacher-forced continuation testing. One sampled chat answer is
not evidence of parity.

- Run `make test-reference` against the committed llama.cpp vectors.
- Run the 100-case OpenRouter fixture for every released GGUF using
  `gguf-tools/quality-testing/score_openrouter`.
- Preserve the `summary` and `api_summary` lines for every model.
- Compare the candidate TSV with the previous release using
  `python3 gguf-tools/quality-testing/compare_scores.py OLD.tsv NEW.tsv`.
- Treat a material NLL, first-token, API top-1, pair-order, or extra-logit
  regression as blocking unless the release notes describe an intentional
  quality tradeoff.
- Keep API keys and raw private response captures outside version control.

## 4. Vulkan Inference

- Build and start both `make vulkan-generic` and `make vulkan-bc250` on the
  BC-250. Both must report the BC-250 fast path. On any additional Vulkan test
  host, use `make vulkan-generic` and record the reported device and subgroup.
- Run `make test-vulkan`; all isolated kernels must pass their numeric gates.
- Run `make test-model`; CPU/Vulkan short and long prompt parity must retain the
  same greedy token and satisfy the top-5/top-20/top-64 overlap gates.
- Exercise thinking and no-thinking prompts.
- Run one short tool-calling prompt and require native Qwen
  `<function>/<parameter>` output.
- After changing a fusion, compare the enabled path with its environment-variable
  fallback on the same forced tokens. Matching one sampled answer is not enough.
- Run `make benchmark-gate` and compare its CSV with the previous release on the
  same board and power state. Investigate any repeatable regression over 5% even
  when the conservative absolute gate still passes.
- Preserve a profiler run when dispatch count or a hot fusion changes.

## 5. Metal Inference

- Run `make test-metal` and `make test-metal-model`.
- Run resident inference, benchmark, evaluator, live server, and native-agent
  smoke tests one model process at a time.
- Require the CPU/Metal short and long prompt parity gates and every isolated
  Metal numeric kernel gate to pass.
- Exercise thinking, no-thinking, native tool calls, session save/restore,
  disk KV, MTP when shipped, and fusion fallbacks.
- Run the release GGUF on every advertised Apple GPU generation. At minimum,
  retain results from an M1-class machine and each materially different GPU
  generation used for tuning; never infer an all-device pass from one Mac.
- Exercise the oldest supported macOS release as well as the current release.
  APIs newer than the deployment target must be protected by runtime
  availability checks and have an older-OS fallback.
- Verify the binary deployment target and runtime Metal shader compilation on
  the oldest supported macOS release. A successful build against the newest
  SDK alone is not sufficient.
- Record the selected device name, OS version, RAM, recommended working-set
  size, and whether the model is resident or VM-backed SSD streaming.

## 6. SSD Streaming

- Run `make test-streaming`. It compares the same short prompt under:
  full residency, warm hotlist streaming, cold streaming with an eight-expert
  cache-pressure budget, and a one-layer resident prefix.
- Require the same greedy token, finite logits, and the strict top-logit overlap
  gate for all modes.
- Run the OpenRouter quality scorer once resident and once streaming for every
  released GGUF. Aggregate quality must remain in the same band.
- Benchmark resident, warm streaming, and cold streaming one process at a time.
  Record prefill and decode speed separately.
- Verify automatic hotlist startup reports an eviction bias and reads no
  expert weights. With no hotlist, warm and `--ssd-streaming-cold` must both
  avoid blind startup reads. An explicit preload count must still preload.
- Test an explicit expert count and an `NGB` budget. The resolved budget must
  fit the reported working set and must never become zero silently.
- Run a cache-pressure prompt long enough to evict expert slots. It must finish
  without stale-slot logits, allocation growth, or a device loss.
- Full-layer streaming must pass short-prompt parity and exact byte-budget
  accounting. Keep the default at zero unless repeated benchmarks improve over
  routed-expert streaming on the target hardware.
- On Metal, test the release GGUF on a physical 8 GB Apple Silicon machine.
  Require the bounded shared-Metal expert cache to stay within its resolved
  byte budget; record peak memory, swap, SSD reads, eviction behavior, and
  post-pressure recovery explicitly. A `--simulate-used-memory` run on a
  larger Mac is useful diagnostics, but does not replace the physical gate.
- Require `--ssd-streaming-cache-experts`, preload, cold-cache, eviction, and
  full-resident-layer controls to enforce the same byte accounting and
  correctness gates on Metal and Vulkan.

## 7. Long Context And Session State

- Run `./q36_test --long-context` with the release context and cache types.
- Run `./q36_test --session-sync-resume` and
  `./q36_test --kv-cache-save-restore`.
- Repeat save/restore with `f16`, `q8_0`, and `q4_0` K/V cache types.
- Verify payload v2 and typed-KV payload v3 load correctly. Token-only v1 is a
  compatibility input, not the current writer format.
- Interrupt a long prefill, restart from a disk checkpoint, and compare the
  next greedy token with an uninterrupted run.
- Corrupt or truncate a disposable cache file and require a clean rejection,
  never a partial restore.

## 8. Server APIs

- Run `./q36_test --server` after HTTP, SSE, prompt, cache, or tool changes.
- Run `make test-server-live` for an actual 4K-context server process, CORS
  preflight, chat completion, and Responses API SSE cycle.
- On macOS, also run `make test-server-live-metal`.
- Run `make test-server-live-metal-ssd` to exercise the same HTTP surfaces
  through the bounded Metal expert cache.
- Run `make test-session-batch` with resident Q8_0/Q4_0 and F16/F16 KV. The
  1/2/4/8-session oracle must preserve full logits, token choices, snapshots,
  invalid-input frontiers, and the forced ordered fallback.
- Run `make test-server-batching`. It starts one four-slot model process and
  sends more concurrent streaming and non-streaming requests than fit in the
  resident slots. Seeded pairs must match and the log must contain a native
  Vulkan batch larger than one.
- On macOS, run `make test-server-batching-metal` and require a native Metal
  batch larger than one.
- Before a Metal release, also run `make test-server-batching-metal-ssd` so
  cache eviction is covered under concurrent sessions and cancellation.
- Exercise non-streaming and streaming requests for:
  `/v1/chat/completions`, `/v1/responses`, `/v1/completions`, and
  `/v1/messages`.
- For `/v1/responses`, test string input, message input, function calls,
  function-call outputs, reasoning summary output, and SSE completion events.
- With `--cors`, send an `OPTIONS` request and require all three
  `Access-Control-Allow-*` headers. Without the option, require none.
- Run a multi-turn tool call. The second request must reuse the exact sampled
  Qwen tool text through its call ID, including after a disk-cache restart.
- Disconnect a streaming client while generation is active. The worker must
  return control without hanging later clients.
- Repeat concurrent short greedy requests with 1, 2, 4, and 8 resident slots.
  Record aggregate decode tokens/s, per-request latency, and the batch-size
  distribution from `Q36_SERVER_BATCH_LOG=1`. Record the old single-session
  path and `Q36_VK_SESSION_BATCH=0` ordered fallback separately. The sequential
  one-process-at-a-time command is `make benchmark-session-batch`.
- Exercise mixed short/long prefills, more requests than slots, tool-result
  continuations, Responses and Anthropic continuations, a disconnected SSE
  client, disk KV slot reuse/restore, and shutdown with active and waiting
  clients. Require no state mix-up, deadlock, starvation, or leaked tool IDs.
- SSD streaming must use the ordered correctness fallback. Native resident
  Vulkan batching supports 2-8 rows; other row counts must fall back exactly.

## 9. Native Agent

- Run `./q36_agent_test`.
- Start the agent with no streaming flag on each release GPU backend and require
  full residency in the startup log. Repeat with `--ssd-streaming` and require
  SSD streaming.
- Run one non-interactive tool call and one interactive edit/read loop.
- Verify generated calls use Qwen native tags and Hermes tool schemas; no
  foreign protocol parser or prompt text may be present.
- Exercise queued input and require `+QUARKSTAR_QUEUED` and
  `+QUARKSTAR_WAITING` markers.
- Exercise `/save`, `/list`, `/switch`, `/compact`, `/history`, and `/new`.
- Interrupt generation with Ctrl+C and ensure the TUI remains usable.
- Run a context-compaction cycle near the configured limit and verify the task
  summary, tool state, and current working directory survive.

## 10. Directional Steering And MTP

- `make test-vulkan` must pass the directional-steering numeric test.
- Run a zero direction file and require identical greedy output.
- Run one nonzero direction with FFN steering and one with attention steering.
- If MTP is shipped, compare greedy text with MTP disabled and enabled, record
  acceptance, and require clean verifier rollback on a partial accept.

## 11. Model Files And Download Workflow

- Run `./download_model.sh --help` and one resumable download in a disposable
  directory.
- Verify the published filename, size, checksum, tensor count, architecture,
  context length, and embedded Qwen template.
- Run `./q36 --inspect -m MODEL` before inference.
- Test every GGUF advertised in the release notes. Do not infer compatibility
  from a similar filename.

## 12. Performance And Power

- Put the BC-250 in the documented performance mode and record temperatures.
- Run `make benchmark-gate` with the default thresholds, then repeat the full
  benchmark sweep used for the previous release.
- Record resident and streaming results independently.
- Repeat one decode at reduced `--power` and verify throttling does not change
  greedy output or corrupt session state.
- On macOS, retain resident and SSD-streaming `q36-bench` results per Apple GPU
  generation; compare like-for-like OS, power state, context, and cache types.

## 13. Release Sign-off

Record:

- Commit and version/tag.
- Hardware, OS/kernel, Metal device or Mesa/RADV and Vulkan loader, and firmware.
- GGUF path, size, and checksum.
- Warning-free Metal, Vulkan, and CPU build results for the platforms tested.
- Unit, Metal, Vulkan, model, reference, streaming, server, and agent results.
- OpenRouter quality summaries.
- Resident and streaming benchmark CSVs.
- Known skips and why they are safe.

Do not publish when a required host or model was unavailable without stating
the missing coverage explicitly in the release notes.
