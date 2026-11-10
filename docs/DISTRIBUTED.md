# Distributed inference

[README](../README.md) | [Vulkan](VULKAN.md) | [SSD streaming](SSD_STREAMING.md)

QuarkStar splits a supported model between two machines using a TCP layer pipeline.
The coordinator runs a prefix of the transformer and the vocabulary projection;
the worker runs the remaining layers and final normalization. Each stage keeps
its own attention and recurrent state and caches only its assigned weights.
The GGUF stays mmap-backed.

The machines can run different Linux distributions. The transport uses ordinary
TCP over Ethernet, Wi-Fi, a VPN, or another routed IP network. It does not depend
on interface names, USB adapters, a network manager, SSH, or a particular firewall.
The project's existing model and backend requirements still apply.

## Prepare both machines

Build the same QuarkStar revision separately on each machine using the
[Linux build instructions](VULKAN.md#build-and-run). Each machine needs its own
working Vulkan installation for GPU inference. CPU stages are also supported
for reference and debugging. Building locally accommodates different C libraries,
drivers, and CPU instruction sets; the executables need not be identical.

Use the same model file on both machines. Flash Next also supports the
[resident worker staging](#qwen38-flash-next) path without copying its n-gram
shard or a second full GGUF. Local paths may differ. Download a
supported model on each host, or copy the completed GGUF and compare its
`sha256sum`. For example:

```sh
./download_model.sh q4
```

The handshake checks protocol version, model metadata, sampled weight
fingerprints, dimensions, context capacity, and contiguous layer coverage.
Choose a layer split that fits each machine's memory, allowing space for its
KV/recurrent state and scratch buffers. The coordinator also holds the output
projection weights. An equal layer count need not be the best memory or speed
balance.

## Choose the network addresses

Use an address or hostname of the coordinator that the worker can reach.
Both machines can use their existing network connections; a dedicated cable
or static subnet is optional. With DHCP, use a stable hostname or an address
reservation if the coordinator's address may change.

| Option | Address to use |
| --- | --- |
| Coordinator `--listen HOST PORT` | A local address to bind, or `0.0.0.0` for all local IPv4 interfaces |
| Worker `--coordinator HOST PORT` | The coordinator's reachable IP address or DNS hostname |

IPv4, IPv6, and DNS hostnames are supported. For IPv6, bind `::` or a local
IPv6 address and connect to the coordinator's reachable IPv6 address. Host and
port are separate arguments, so IPv6 literals do not need brackets. A wildcard
bind address is not a destination address for the worker.

Permit the selected TCP port on the coordinator from the worker using your
host's firewall configuration. Routed networks must allow that connection;
Wi-Fi access points with client isolation may block communication between peers.
SSH is optional for administration or file copies and does not carry inference.

The inference transport has no authentication or encryption. Use a trusted
network or an encrypted tunnel/VPN when the network is not trusted.

## Start the pipeline

The following example splits a 40-layer model into layers `0:19` and
`20:output`. Ranges are inclusive; `output` includes the remaining transformer
layers and final normalization. The normalized state returns to the coordinator
for the vocabulary projection. Adjust the ranges for your supported model and
each machine's available memory.

For Qwen3.6 35B Q4, this is a 50/50 split: 20 target layers per board. Its
GGUF reports 41 blocks because the last block is the optional MTP head, which
stays on the coordinator and is not part of the target layer split. Layer
ranges are explicit frontend options; they are not selected from GPU clocks
or CU counts.

On the worker, replace `coordinator.example` with the coordinator's reachable
hostname or IP address, and set the local model path:

```sh
MODEL=/path/to/model.gguf
COORDINATOR_HOST=coordinator.example

./q36 -m "$MODEL" --vulkan --ctx 4096 --prefill-chunk 32 \
  -ctk q8_0 -ctv q4_0 \
  --role worker --layers 20:output --coordinator "$COORDINATOR_HOST" 9911
```

On the coordinator, set its local path to the same model:

```sh
MODEL=/path/to/model.gguf

./q36 -m "$MODEL" --vulkan --ctx 4096 --prefill-chunk 32 \
  -ctk q8_0 -ctv q4_0 \
  --role coordinator --layers 0:19 --listen 0.0.0.0 9911 \
  --temp 0 --nothink -p 'What is 2 + 2?' -n 16
```

Either process can start first. The worker retries until the coordinator is
available. Transport failures during connection setup wait one second before
retrying, including when a proxy accepts TCP but its upstream is unavailable.
Port `9911` is an example; choose any available TCP port and use it
on both sides. Use a specific local address with `--listen` to restrict the
coordinator to one network.

Omit `-p` for interactive chat. `q36-server`, `q36-agent`, `q36-bench`, and
`q36-eval` accept the coordinator options and use their regular session APIs.
Workers always run `q36`. For the HTTP server, `--listen HOST PORT` is the
pipeline socket; `--host` and `--port` select the separate HTTP socket.
The coordinator's session context must not exceed the worker's `--ctx`.

Resident Vulkan stages preload their assigned weights into device memory;
startup fails if the full slice cannot be cached. Adjust the split or use
`--ssd-streaming` on a stage whose weights and runtime buffers do not fit.
Resident and streaming modes may differ between machines. A slow model drive
affects loading, while streaming can also make drive speed part of inference.

## Prefill and network performance

`--dist-prefill-chunk N` caps each transmitted chunk. Its size is bounded by
both stages' prefill capacities. The coordinator can evaluate a larger local
batch and split its residual rows into smaller worker chunks. This lets each
stage use its own `--prefill-chunk` capacity without reducing the larger stage
to the smaller stage's batch size. A bounded I/O exchange overlaps the worker
with computation of the next local batch; GPU work and progress/cancellation
callbacks remain on the inference thread. Short spans keep the smaller batch
to avoid an extra pipeline fill. `--dist-prefill-window N` controls
the number of chunks in flight, from 1 to 64; the default is 2. A window larger
than one lets the coordinator compute the next chunk while the worker processes
the previous chunk. These controls work with any network transport supported
by the host's TCP stack.

Activations use lossless FP32 transport in network byte order. For the final
token of a prefill span or decode call, the worker returns one normalized hidden
row. The coordinator projects it to full logits for sampling and logprobs.
This keeps the return transfer proportional to the hidden width rather than
the vocabulary size. Protocol version 4 also negotiates the residual width
independently of the embedding width, so HC models retain every stream. Rebuild
both stages when upgrading; older protocols are rejected.

Without MTP, decode of one conversation remains serial: the next token depends on both
stages finishing the current one. Throughput depends on both machines, the
layer split, memory/storage mode, bandwidth, latency, and network congestion.
Wi-Fi and routed links can have different throughput and jitter from a direct
Ethernet link. Measure on the network you intend to use.

When prefill is limited by the prefix stage, try increasing only the
coordinator's `--prefill-chunk`, keeping worker chunks and the window small.
Larger local batches can use more efficient GPU kernels, at the cost of more
scratch memory and startup latency. Different batch shapes can change
floating-point rounding; compare logits and memory as well as throughput.

Vulkan coordinators retain projected logits on the GPU. Greedy decode reads
only the top token ids; sampling, logprobs, and the full-logits API read the
vocabulary row when needed. CPU stages keep their logits in host memory.

`--debug` reports route and timing information. Its `remote_wait` time includes
the coordinator's vocabulary projection after receiving the final row.

## Qwen3.8 Flash Next

Flash Next uses the same coordinator/worker options, framing, session recovery
and resident weight arena. Its forward transfer contains all four HC streams
(10,240 FP32 values per token). The suffix runs the final HC mixer and returns
one 2,560-value row; the vocabulary head stays beside sampling on the
coordinator. Layer 1 and its PLE history must belong to the coordinator. Only
that machine opens the n-gram shard.

The default split gives each machine half the transformer layers, with an odd
remaining layer assigned to the coordinator. This 48-layer model uses `0:23`
on the main BC-250 and `24:output` on the worker. The staging helper reads the
layer count from the GGUF and prints both ranges before loading weights.
It does not detect clocks or CU counts. Both stages can be resident while the
main machine gathers n-grams from its SSD. No MTP head is implemented for
Flash Next.

Build the same source and shaders on both hosts first. The worker can use the
ordinary first GGUF shard, without the n-gram shard. If its model drive is too
slow, stage only the suffix weights directly into its resident GPU cache:

```sh
MODEL=/path/to/Qwen3.8-Flash-Next-GSQ-RCO-Coder-22GB-IQ2_XXS-00001-of-00002.gguf

python3 tools/flash_next_worker.py --model "$MODEL" --ssh root@moraine \
  --remote-dir /root/q36-flash-next \
  --coordinator 10.36.0.1 --ctx 4096 --bulk-host 10.36.0.2
```

The example uses the existing dedicated USB Ethernet subnet. Replace the SSH
destination, paths and addresses for your hosts. The helper uses ordinary SSH
authentication; `--control-path` and repeated `--ssh-option` arguments support
an existing SSH connection. `--bulk-host` sends the weight payload over the
trusted direct link, accepting only the coordinator address; omit it to carry
the payload through SSH instead. Ports default to 9911 for inference and 9912
for the temporary weight transfer.

The helper creates a sparse `.q36-resident` descriptor in `/dev/shm`, containing
the original tensor catalogue and handshake fingerprint samples. It occupies
about 14.7 MiB even though its logical size matches the original shard. This is
not a complete GGUF: the engine rejects it without the resident weight stream.
Tensor offsets and sizes are checked while data enters the existing GPU arena;
inference has no demand-upload fallback. No full weight file or PLE/n-gram shard
is stored on the worker. Restart the helper after a worker reboot or process
exit to stage the weights again. Install the executable and shaders on an
executable filesystem; `/dev/shm` may be mounted with `noexec`.

Leave that helper running. On the main machine, start any existing frontend:

```sh
Q36_THREADS=16 ./q36 -m "$MODEL" --vulkan --ctx 4096 -t 16 \
  --role coordinator --layers 0:23 --listen 10.36.0.1 9911 \
  --temp 0 --nothink -p 'Say hello.' -n 16
```

The current QSA implementation transforms each completed four-token key block
once, retaining it in the first raw-key row of that block. Partial blocks keep
their raw rows until complete. Radix selection replaces quadratic ranking of
all key blocks; it preserves score ties, signed zeros and natural block order.
Dense rows at the sparse-transition boundary directly retain every token.
Neither optimization allocates another GPU cache. Contexts beyond 2,048 key
blocks retain the existing CPU heap-selection fallback.

Prompt reset clears recurrent/PLE state and exposed outputs in place while
retaining existing GPU KV and batch buffers. New context growth does not copy
unused CPU KV mirrors for Vulkan sessions. CPU inference keeps its existing
state allocation and arithmetic.

The matched 24/24 sweep uses 16 threads, batch 32, context allocation 8,192,
complete warm-up at every frontier and eight greedy tokens per point. At
7,808 tokens, QSA caching/radix selection and buffer reuse improve plain
decode from 16.17 to 22.71 t/s and prefill from 123.64 to 130.06 t/s.
Longer prefill points measure 1,920-token extensions. Main GPU tensor peak
is unchanged; worker high-water RSS falls from 354.95 to 71.37 MiB.
See the [benchmark summary](../speed-bench/qwen38-flash-next/README.md)
for the full context sweep, memory breakdown, temperatures and commands
using the shared `q36-bench` executable. Use `--ctx 8192` on the worker.

All five complete 248,320-logit frontier vectors are byte-identical to the
control. Nine additional checks with a 2,083-token prefix cover prefill,
decode completing a partial block, isolated sessions, divergent prompts,
snapshots, payload replay and prefix extension. These remain bit-identical,
and invalidation clears exposed logits. Fresh 16-thread CPU/Vulkan tests pass
all unchanged top-1 through top-64 gates. Regression and kernel tests pass.
These measurements use resident experts with PLE on the main SSD. Flash Next
MTP remains unavailable; the 150/30 t/s targets are not met at long contexts.

## Distributed MTP

This section applies to the model families with an implemented MTP graph.
Flash Next still rejects MTP. Its supplied shards contain no nextn head;
[published shared heads](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/main/MTP/README.md)
exist, but need a separate graph, target-tensor borrowing, full final HC
residual transfer and speculative state rollback. The checked shared head has
512 experts, independent of this target's pruned 256-expert layers. No Flash
Next MTP performance is claimed here.

Enable the existing MTP option on the coordinator only:

```sh
./q36 --vulkan -m "$MODEL" --mtp --mtp-draft 4 --mtp-margin 0 \
  --role coordinator --layers 0:19 --listen 0.0.0.0 9911 \
  --ctx 8192 --prefill-chunk 32 -ctk q8_0 -ctv q4_0
```

Start the worker with its matching suffix range as above, without `--mtp`.
Both stages must use resident Vulkan inference. The coordinator owns the MTP
block, embeddings, and vocabulary head; the worker does not need a duplicate
MTP head. Existing sampled opportunistic MTP, confidence margins, draft limits,
and generation boundaries apply to the pipeline too.

The coordinator proposes a short continuation. Its prefix stage can verify
candidate token N+1 while the suffix stage verifies token N. Each stage uses
ordinary single-token arithmetic and keeps a recurrent rollback point. Only
the matching target prefix is committed on both machines. Rejected tails are
rolled back using saved recurrent inputs, without rerunning model weights.
Verification keeps vocabulary logits on the coordinator GPU and reads only
the top token for each candidate. Once a continuation fails, queued tail rows
are drained without projecting them. The final accepted vocabulary row stays
on the GPU until sampling, logprobs, or an API caller needs it on the CPU.

MTP also needs normalized target rows to maintain its shifted attention
history. Requests and replies are exchanged concurrently so large hidden-row
returns cannot block the next request. GPU work remains on the inference
thread. `--dist-prefill-window` bounds work in flight during prefill and candidate
verification. Decode gains depend on draft acceptance, the layer split, and
network latency; compare `--mtp` against plain decode at several context sizes
before selecting a profile.

## Sessions and recovery

Sessions have separate worker KV/recurrent state. Multi-session decode runs in
order. Divergent prompts reset and rebuild both stages. On transport failure,
the current operation fails and the connection is discarded. The next session
operation reconnects and reconstructs that session from its token checkpoint.
A failed or cancelled operation never commits a partial distributed checkpoint.
This recovery also applies when a Wi-Fi connection drops or a worker restarts.

Pipeline disk payloads use the token-only checkpoint format. Loading one, or
restoring a snapshot, replays the saved tokens across both stages. A distributed
checkpoint does not contain a complete single-machine KV cache. Full resident
KV payloads from a single-machine run cannot be loaded into a pipeline session.
MTP history is rebuilt during replay as well. Token-only restore follows cold
prefill arithmetic, which can differ from the earlier decode schedule's rounding.

## Measure resident throughput

Start the worker and wait for both stages to report
`resident pipeline weights: ... ready`. Use an untimed warm-up to exclude
initial connection setup and shader compilation as well as model loading.
For the example split above, run on the coordinator:

```sh
./q36-bench -m "$MODEL" --vulkan \
  --role coordinator --layers 0:19 --listen 0.0.0.0 9911 \
  --prefill-chunk 32 --dist-prefill-window 2 -ctk q8_0 -ctv q4_0 \
  --prompt-file tests/long_context_story_prompt.txt \
  --ctx-start 256 --ctx-max 1024 --step-incr 256 --ctx-alloc 4096 \
  --gen-tokens 128 --warmup
```

`--warmup` prefills every measured context frontier and decodes eight tokens
at each, then resets the session before timing. This also exercises attention
kernels selected only at intermediate context sizes. At later frontiers, `prefill_tokens` is
the newly added suffix; `ctx_tokens` is the resulting total context.

`Q36_VK_MEMORY_REPORT=1` reports weight uploads. On the coordinator the counter
resets after warm-up, so zero confirms no further weight staging during the
measured run. On the worker, compare the counter before and after the run.
Compare speeds using the same model, state precision, context and output
lengths, backend settings, and warm-up procedure. Keep cold-start measurements
separate from resident inference measurements.

## Validation

The transport tests need no model or GPU:

```sh
make test-distributed
```

They cover framing, option validation, address handling, pipelined chunks,
resets, isolated sessions, cancellation, protocol compatibility, and worker
replacement with transcript replay.

For a real model, record a single-machine reference first:

```sh
make tests/test_distributed_model
./tests/test_distributed_model --model "$MODEL" --ssd-streaming \
  --record /tmp/q36-reference.bin
```

Start a worker with the matching suffix range, `--ctx 512 --prefill-chunk 8
-ctk q8_0 -ctv q4_0`, then compare the coordinator against the reference:

```sh
./tests/test_distributed_model --model "$MODEL" --ssd-streaming \
  --role coordinator --layers 0:19 --listen 0.0.0.0 9911 \
  --reference /tmp/q36-reference.bin
```

For Flash Next, use `0:23` / `24:output`. Add `--prefill-tokens 2083` to both
the recording and comparison commands to exercise sparse attention and a
partial key block, and give the worker at least `--ctx 2115`.

The test compares complete logits for prefill, decode, a second session,
divergent prompts, snapshots, disk checkpoint restore, prefix extension, and
ordered multi-session decode. Match prefill caps, KV types, and recurrent-state
precision. Resident Vulkan normally stores recurrent state in FP16, while
streaming uses FP32. For FP32 reference comparisons, set
`Q36_VK_RECURRENT_STATE_F16=0`, `Q36_VK_RECURRENT_CONV_DECODE=0`, and
`Q36_VK_DELTA_QKV=0` on both processes. Different hardware and backends can also
affect floating-point rounding; assess correctness on the actual pair.

To check distributed MTP against plain target decoding on the actual pair:

```sh
make tests/test_distributed_mtp
./tests/test_distributed_mtp --model "$MODEL" \
  --role coordinator --layers 0:19 --listen 0.0.0.0 9911
```

Use a worker with `--ctx 1024 --prefill-chunk 32` and matching KV/recurrent
settings. The test checks greedy and sampled boundaries on prose, code, and
reasoning; full logits after accepted blocks; a forced rejection; checkpoint
replay; and generation boundaries. The Qwen3.6 Q4 path has been validated with
default Vulkan settings and with FP32 recurrent state and separate
convolution/QKV operations as described above.

To exercise unequal stage capacities, use a smaller worker capacity and add
`--prefill-chunk 128 --prefill-tokens 256` to the MTP test above. Its padded
prompts cover multiple local batches, smaller worker chunks, and partial tails.

## Current limits

A route has exactly one prefix coordinator and one suffix worker. Plain
pipeline inference supports CPU and Vulkan stages; distributed MTP requires
resident Vulkan on both. Vision, directional steering,
`--ssd-streaming-full-layers`, and whole-model diagnostic flags are unavailable
in pipeline mode. Multi-machine Metal inference has not been validated. The
pipeline does not add support for model architectures or tensor layouts outside
the existing runtime.
