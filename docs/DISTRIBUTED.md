# Inference Across Machines

[README](../README.md) | [BC-250 setup](BC250.md) | [SSD streaming](SSD_STREAMING.md)

QuarkStar supports a two-machine layer pipeline over TCP. The coordinator
runs the first layers and vocabulary projection; the worker runs the remaining
layers and final normalization. Each machine keeps weights and KV state for
its own layers.

Use the same commit and model on both machines. Local model paths may differ.
The transport has no authentication or encryption: use a trusted network or
an encrypted tunnel.

## Prepare both machines

Follow the [BC-250 setup](BC250.md) on each board, then build and download
the same model from the repository root:

```sh
make vulkan-bc250
./download_model.sh q4
```

Qwen3.6 Q4 and Qwen3.8 Flash Next are recommended for distributed inference
on two BC-250s. Both also work with SSD streaming on one board, at modest
speeds. Each board needs room for its weight slice, context, and scratch
buffers. The coordinator also holds the vocabulary projection.

Use the coordinator's reachable IP address or hostname. Any working TCP
network is sufficient; a dedicated cable and static subnet are unnecessary.
Allow the chosen TCP port through the coordinator's firewall from the worker.

## Start the pair

This example assigns 20 of Qwen3.6 Q4's 40 target layers to each board.
Replace `192.168.1.10` with the coordinator's address.

On the worker:

```sh
./q36 -m gguf/Qwen3.6-35B-A3B-IQ4XS-Q4K-Q8.gguf \
  --role worker --layers 20:output --coordinator 192.168.1.10 9911
```

On the coordinator:

```sh
./q36 -m gguf/Qwen3.6-35B-A3B-IQ4XS-Q4K-Q8.gguf \
  --role coordinator --layers 0:19 --listen 192.168.1.10 9911
```

Either process can start first; the worker retries until the coordinator is
ready. Keep both running. The coordinator opens the normal interactive chat.

Layer ranges are inclusive. `20:output` includes the remaining target layers
and final normalization; the vocabulary projection stays on the coordinator.
The optional MTP block is separate from the 40 target layers. Choose adjacent
ranges that cover the model exactly once. An equal layer count is a starting
point, not a guarantee of equal memory use or execution time.

The coordinator can also be `q36-server`, `q36-agent`, `q36-bench`, or
`q36-eval`, using the same route options. Workers always run `q36`.
For the server, `--listen` is the pipeline socket; `--host` and `--port`
select the separate HTTP socket. Set the worker's `--ctx` at least as large
as the coordinator's session context.

## Tuning and memory

Start with the default prefill sizes. `--dist-prefill-chunk N` caps the
transmitted chunk size; `--dist-prefill-window N` sets the number of chunks
in flight, from 1 to 64, with a default of 2. Each stage can use its own
`--prefill-chunk`. `--debug` reports the route, batch sizes, and timings.

Prefill can overlap the two stages. Plain generation remains serial: each
token must finish both stages before the next is sampled. Use a pipeline
primarily to fit larger models; decode gains depend on the split and network.
Activations use lossless FP32 transport.

Resident stages must fit their entire assigned slice. Adjust the split or
add `--ssd-streaming` to a stage whose weights and runtime buffers do not
fit. The two stages may use different residency modes.

## Distributed MTP

For models with an implemented MTP head, add `--mtp` to the coordinator
command. Leave it off on the worker. Both stages must use resident Vulkan.
The coordinator owns the head and shares the target embeddings and output
weights with it.

Draft verification can overlap work on the two boards. Accepted tokens are
committed on both stages; rejected tails restore recurrent state. Speed
depends on draft acceptance and network latency. Compare against ordinary
decoding with the same prompt and context. See [speculative decoding](SPECULATIVE_DECODING.md).

## Qwen3.8 Flash Next

Use the same setup with the `flash-next` download target. Its 48 layers use
`0:23` on the coordinator and `24:output` on the worker. Pass the first shard,
`gguf/Qwen3.8-Flash-Next-GSQ-RCO-Coder-22GB-IQ2_XXS-00001-of-00002.gguf`,
with `-m` on both machines.

Only the coordinator needs the second, n-gram shard. Keep both shards together
there. Both stages can keep their transformer weights resident while the
coordinator reads n-grams from SSD. Flash Next has no implemented MTP head.

See the [recorded pipeline measurements](../speed-bench/qwen38-flash-next/README.md#resident-distributed-pipeline)
for context sweeps and memory use.

## Sessions and recovery

Sessions keep separate worker state and run in order. A failed connection
fails the current operation. The next session operation reconnects and
replays its token checkpoint across both stages. Failed or cancelled work
does not commit a partial checkpoint.

Pipeline snapshots and disk checkpoints contain tokens, so restoring them
requires prefill. Single-machine KV payloads cannot be loaded into a pipeline.

## Validation

The transport tests need no model or GPU:

```sh
make test-distributed
```

For model checks, [tests/test_distributed_model.c](../tests/test_distributed_model.c)
compares full logits across prefill, decode, isolated sessions, divergent
prompts, snapshots, checkpoint replay, and prefix extension.
[tests/test_distributed_mtp.c](../tests/test_distributed_mtp.c) compares
speculation with ordinary target decoding, including rejection and replay.
Match KV types, prefill sizes, and recurrent precision when comparing runs.

## Current limits

A route has one coordinator and one worker. CPU stages are available for
reference use; distributed MTP requires resident Vulkan. Multi-machine Metal
has not been validated. Vision, directional steering, full-layer streaming,
and whole-model diagnostic flags are unavailable in pipeline mode.

For protocol details, see [q36_distributed.c](../q36_distributed.c).
