# KV Caches

[README](../README.md) | [Server](SERVER.md) | [Native agent sessions](AGENT.md#saved-sessions)

## Precision

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

## Server disk cache

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

## Save policy

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

## On-disk format

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
