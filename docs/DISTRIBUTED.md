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

Use the same model file on both machines. Local paths may differ. Download a
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

`--dist-prefill-chunk N` caps the coordinator's chunk size. The actual size is
bounded by both stages' prefill capacities. `--dist-prefill-window N` controls
the number of chunks in flight, from 1 to 64; the default is 2. A window larger
than one lets the coordinator compute the next chunk while the worker processes
the previous chunk. These controls work with any network transport supported
by the host's TCP stack.

Activations use lossless FP32 transport in network byte order. For the final
token of a prefill span or decode call, the worker returns one normalized hidden
row. The coordinator projects it to full logits for sampling and logprobs.
This keeps the return transfer proportional to the hidden width rather than
the vocabulary size. Protocol version 2 rejects older full-logit peers.

Decode of one conversation remains serial: the next token depends on both
stages finishing the current one. Throughput depends on both machines, the
layer split, memory/storage mode, bandwidth, latency, and network congestion.
Wi-Fi and routed links can have different throughput and jitter from a direct
Ethernet link. Measure on the network you intend to use.

`--debug` reports route and timing information. Its `remote_wait` time includes
the coordinator's vocabulary projection after receiving the final row.

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

The test compares complete logits for prefill, decode, a second session,
divergent prompts, snapshots, disk checkpoint restore, prefix extension, and
ordered multi-session decode. Match prefill caps, KV types, and recurrent-state
precision. Resident Vulkan normally stores recurrent state in FP16, while
streaming uses FP32. For FP32 reference comparisons, set
`Q36_VK_RECURRENT_STATE_F16=0`, `Q36_VK_RECURRENT_CONV_DECODE=0`, and
`Q36_VK_DELTA_QKV=0` on both processes. Different hardware and backends can also
affect floating-point rounding; assess correctness on the actual pair.

## Current limits

A route has exactly one prefix coordinator and one suffix worker. MTP, vision,
directional steering, `--ssd-streaming-full-layers`, and whole-model diagnostic
flags are unavailable in pipeline mode. CPU and Vulkan stages are supported;
multi-machine Metal inference has not been validated. The pipeline does not
add support for model architectures or tensor layouts outside the existing
runtime.
