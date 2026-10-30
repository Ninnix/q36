# Serving Models

[README](../README.md) | [Client setup](CLIENTS.md) | [UI setup](UI.md)

## Start locally

```sh
./q36-server --vulkan --ctx 32768
```

On Apple Silicon, use `--metal`. The default address is
`http://127.0.0.1:8000`. Use `--host` and `--port` to change it; pass `-m FILE`
to select a supported GGUF. Run from the repository root.

An 8 GB Mac needs a smaller context and bounded MoE expert cache:

```sh
./q36-server --metal --ssd-streaming --ssd-streaming-cache-experts 512 --ctx 4096
```

## APIs

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

### Prompt and thinking controls

Both Qwen3.6 and Qwen3.8 accept the fixed-template controls:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": false,
    "preserve_thinking": true
  }
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

Chat, Responses, and Anthropic support SSE streaming. In thinking mode,
reasoning is streamed in the native API shape instead of being mixed into
final text. OpenAI chat
streaming also streams tool calls as soon as the `<tool_call>` opening is
recognized: the tool header is sent first, then each completed native
parameter is forwarded as a `tool_calls[].function.arguments` delta while
generation continues. The Anthropic endpoint streams thinking and text live, then emits
structured `tool_use` blocks when the generated tool block is complete.

Pass `--cors` to add `Access-Control-Allow-Origin`, methods, and headers and
to answer browser `OPTIONS` preflight requests. CORS headers are disabled by
default.

For fixed-length greedy generation, requests may set `ignore_eos: true`
with an explicit `temperature: 0`. Context limits, stop strings, and client
stops still apply. Cache usage is reported in each API's usage fields.

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

## Session scheduling

Without extra options the server keeps one mutable backend/KV checkpoint and
uses one graph worker. Stateless clients that resend a longer version of the
same prompt can reuse that prefix instead of pre-filling from token zero.

## Multiple sessions

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

## Images

Start with the matching language model and `--vision FILE`; see
[models and vision](MODELS.md#vision). OpenAI chat and Responses accept inline
PNG/JPEG data URIs. Anthropic accepts base64 image sources. Remote image URLs
are rejected. Vision execution currently requires Vulkan.

## Thinking modes

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

## Tool history

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

## Disk KV cache

Enable disk checkpoints to reuse compatible prefixes across slot changes
and server restarts:

```sh
./q36-server --ctx 32768 --kv-disk-dir /tmp/q36-kv --kv-disk-space-mb 8192
```

See [KV caches](KV_CACHE.md) for precision, save policy, exact tool replay,
and the on-disk format. For external coding agents, follow
[client setup](CLIENTS.md); for a separate web UI, see [UI setup](UI.md).
