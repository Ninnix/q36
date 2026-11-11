# Models and Vision

[README](../README.md#start-here) | [Distributed inference](DISTRIBUTED.md) | [SSD streaming](SSD_STREAMING.md)

QuarkStar runs the project's Qwen3.6-35B-A3B, dense Qwen3.8-27B, and
Qwen3.8 Flash Next GGUFs. It is not a general GGUF loader. Other files may
have unsupported tensor layouts, quantization mixes, metadata, or MTP state.
Use the download targets below; `./download_model.sh --help` lists filenames
and options.

## Downloads

| Target | Model | File size | Use |
| --- | --- | --- | --- |
| `q2` | Qwen3.6-35B-A3B IQ2_XXS/Q2_K | About 12.1 GB | Default resident MoE model for 16 GB machines |
| `q4` | Qwen3.6-35B-A3B IQ4_XS/Q4_K | 19.05 GiB | Recommended for two-machine distributed inference |
| `27b` | Qwen3.8-27B UD-IQ3_S | 11.21 GiB | Resident dense model |
| `27b-iq3-xxs` | Qwen3.8-27B GSQ-RCO IQ3_XXS | 9.73 GiB | Smaller dense weights, leaving more room for context |
| `flash-next` | Qwen3.8 Flash Next GSQ-RCO Coder IQ2_XXS | About 50.8 GB, two shards | Recommended for two-machine distributed inference |

For a first run on the BC-250:

```sh
./download_model.sh q2
./q36
```

Qwen3.6 Q4 and Flash Next are suggested for [distributed inference](DISTRIBUTED.md)
on two BC-250s. Both also work with [SSD streaming](SSD_STREAMING.md) on one
board, with modest performance.

Files go in `gguf/`. Repeat an interrupted download to resume it. Public
weights normally need no authentication; the script uses `HF_TOKEN` or the
local Hugging Face token cache when present.

Main-model downloads update the legacy `q36moe.gguf` link. The runtime default
is `gguf/Qwen3.6-35B-A3B-IQ2XXS-Q2K-Q8.gguf`; pass `-m FILE` for other
models. The binaries do not follow the legacy link automatically.

Qwen3.6 and both dense Qwen3.8 downloads include an MTP head. Enable it with
`--mtp`; see [speculative decoding](SPECULATIVE_DECODING.md). Flash Next has
no implemented MTP head.

## Qwen3.6-35B-A3B

The Q2 recipe uses IQ2_XXS gate/up and Q2_K down projections for routed
experts, which account for most of the weights. Shared experts, projections,
routing, and other tensors retain higher precision. The 12.1 GB file includes
a 333 MB MTP head and leaves room for context and the OS on a BC-250.

Q4 uses IQ4_XS gate/up and Q4_K down experts. Its 19.05 GiB weights need a
two-board pipeline or SSD streaming on a single 16 GB board. The pipeline
split is `0:19` / `20:output`; follow the [two-machine walkthrough](DISTRIBUTED.md#start-the-pair).

Both downloads come from
[Ninnix96/Qwen3.6-35B-A3B-gguf](https://huggingface.co/Ninnix96/Qwen3.6-35B-A3B-gguf).

## Qwen3.8-27B

Both dense releases use the same CLI, server, agent, evaluation, and vision
projector. Download and run the smaller model:

```sh
./download_model.sh 27b-iq3-xxs
./q36 -m gguf/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf
```

For IQ3_S, use the `27b` target and select
`gguf/Qwen3.8-27B-UD-IQ3_S.gguf`. Dense weights must fit resident; the MoE
expert-streaming cache does not apply to them.

GSQ-RCO replaces the earlier unofficial UD-IQ3_XXS option. Its mixed formats,
including BF16 recurrent projections, run natively on the GPU. Each quant has
its own disk-cache and agent-checkpoint identity.

The downloads come from
[Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) and
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF).

## Qwen3.8 Flash Next

The download contains about 22.0 GB of transformer weights and 28.8 GB of
disk-backed n-grams. Keep both shards together and select the first with `-m`:

```sh
./download_model.sh flash-next
./q36 \
  -m gguf/Qwen3.8-Flash-Next-GSQ-RCO-Coder-22GB-IQ2_XXS-00001-of-00002.gguf \
  --ssd-streaming --ctx 4096
```

This is the single-board SSD-streaming setup, with modest prefill and decode
performance. For the recommended resident two-board pipeline, use `0:23` on
the coordinator and `24:output` on the worker. Only the coordinator needs
the n-gram shard; see [distributed Flash Next](DISTRIBUTED.md#qwen38-flash-next).

N-gram rows are read from SSD on demand in either mode. The n-gram shard is
never loaded as a resident table. Checkpoints restore by replaying saved
tokens. This architecture supports Vulkan and the CPU reference; MTP, vision,
steering, and Metal are unavailable.

The shards come from
[Ninnix96/Qwen3.8-Flash-Next-GSQ-RCO-Coder-22GB-gguf](https://huggingface.co/Ninnix96/Qwen3.8-Flash-Next-GSQ-RCO-Coder-22GB-gguf).

## Vision

For Qwen3.6, download the matching encoder and start the CLI:

```sh
./download_model.sh qwen36-vision
./q36 --vision gguf/Qwen3.6-35B-A3B-mmproj-F16.gguf
```

For dense Qwen3.8-27B, use the `qwen38-vision` target and pass
`--vision gguf/Qwen3.8-27B-mmproj-F16.gguf` with the selected dense model.
Encoder downloads do not change the legacy main-model link.

At the interactive prompt, `/read photo.jpg` submits an image turn.
Append a question after the filename to describe what you want. JPEG and PNG
decoding is built in. The matching projector stays disk-backed and streams
one weight tensor at a time through a reusable Vulkan buffer.

Vision execution requires Vulkan. `q36-agent --vision FILE` exposes
`view_image` for local files. The [server image API](SERVER.md#images) accepts
inline PNG/JPEG images; remote image URLs are rejected. Image fingerprints
and geometry participate in prefix reuse. Vision sessions cannot be saved
as agent sessions or disk KV checkpoints.

The vendored JPEG/PNG decoder in `third_party/iris` is Copyright (c) 2026
Salvatore Sanfilippo and distributed under its included MIT license.

## Build weights

See [GGUF tools](../gguf-tools/README.md) for conversion, quantization,
fine-tunes, and imatrix collection. The native quantizer accepts Q8, F16,
or BF16 inputs; imatrix collection uses the optional llama.cpp tooling.
