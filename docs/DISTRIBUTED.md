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

## Qwen3.8-27B Q6_K

Download the Unsloth Dynamic Q6_K model with `./download_model.sh 27b-q6`.
Its mixed tensor types and embedded MTP block use the existing dense kernels.
The measured split is `0:33` on the coordinator and `34:output` on the worker:
about 10.48 GiB and 8.68 GiB of resident weights respectively, excluding the
coordinator's pageable embeddings and optional 335 MiB MTP block. Runtime
scratch, recurrent state, and KV cache need additional space.

With the complete GGUF on both boards, use the normal launch commands above
with these layer ranges. For a worker with slow or limited storage, stage
only its suffix directly into Vulkan instead:

```sh
python3 tools/pipeline_worker.py \
  --model gguf/Qwen3.8-27B-UD-Q6_K.gguf \
  --ssh root@worker --remote-dir /root/q36 \
  --metadata /dev/shm/q36-q6.q36-resident \
  --layer-start 34 --coordinator 192.168.1.90 --ctx 4096 \
  --env Q36_VK_PREFILL_CHUNK=64
```

Keep this helper running. It starts the worker, sends the catalogue and
fingerprint samples, then uploads only assigned target weights, including
final normalization. The MTP block stays on the coordinator. No full model
copy is stored on the worker. `--dry-run` reports the weight slice without
connecting. The old `tools/flash_next_worker.py` entry point remains available.

If the coordinator's firewall blocks inbound inference traffic, an SSH
reverse tunnel can carry it without changing firewall rules:

```sh
ssh -N -R 127.0.0.1:19911:192.168.1.90:9911 root@worker
```

In that case, pass `--coordinator 127.0.0.1 --port 19911` to the helper.
The coordinator still listens on `192.168.1.90:9911`. Optional
`--bulk-host WORKER_LAN_IP --bulk-peer COORDINATOR_LAN_IP` sends the initial
weight payload over the trusted LAN while inference uses the tunnel.

Start the coordinator with the matching route:

```sh
./q36 -m gguf/Qwen3.8-27B-UD-Q6_K.gguf --vulkan \
  --role coordinator --layers 0:33 --listen 192.168.1.90 9911 \
  --ctx 4096 --prefill-chunk 64 --dist-prefill-chunk 64
```

Add `--mtp` only on the coordinator to use the embedded head. Both boards
must run the same build. The sparse `.q36-resident` descriptor is valid only
with the helper's weight stream; it is not a standalone model file.

Dense Q6 prefill batches the existing native arithmetic across tokens. It
reuses the verification kernels, shares Q5_K/Q6_K/IQ4_NL weight loads, and
rounds FP16 recurrent state after every token. This fixes the accumulated
4K drift without evaluating the whole model one token at a time. The default
stage capacity is 64 tokens. Other quantizations retain their existing
prefill selection. See the [measurements](../speed-bench/qwen38-27b-q6/README.md).

For interactive CPU inference on one board:

```sh
./q36 -m gguf/Qwen3.8-27B-UD-Q6_K.gguf --cpu -t 16 --ctx 4096
```

To check the distributed Vulkan path against CPU, start the helper with
`--ctx 4096`, then run:

```sh
Q36_TEST_THREADS=16 ./q36_test --gpu-cpu-parity --case short \
  --model gguf/Qwen3.8-27B-UD-Q6_K.gguf \
  --role coordinator --layers 0:33 --listen 192.168.1.90 9911
```

This uses the existing top-1 through top-64 parity gates, capturing CPU
results before opening the Vulkan coordinator. CPU runs page
weights from disk when the model exceeds RAM and can be slow.

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
