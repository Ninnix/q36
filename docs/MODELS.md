# Models and Vision

[README](../README.md#start-here) | [SSD streaming](SSD_STREAMING.md)

QuarkStar runs the project's Qwen3.6-35B-A3B and dense Qwen3.8-27B GGUFs.
It is not a general GGUF loader. Other files may have unsupported tensor
layouts, quantization mixes, metadata, or MTP state. Use the download targets
below; `./download_model.sh --help` lists filenames and options.

## Downloads

| Target | Model | File size | Use |
| --- | --- | --- | --- |
| `q2` | Qwen3.6-35B-A3B IQ2_XXS/Q2_K | About 12.1 GB | Default MoE model for 16 GB machines |
| `27b` | Qwen3.8-27B UD-IQ3_S | 11.21 GiB | Higher-quality dense weights |
| `27b-iq3-xxs` | Qwen3.8-27B GSQ-RCO IQ3_XXS | 9.73 GiB | Smaller dense weights, leaving more memory for context |

For a first run:

```sh
./download_model.sh q2
./q36
```

Files go in `gguf/`. Downloads resume with `curl -C -`; repeat the command
if a transfer is interrupted. Public weights normally need no authentication.
Pass `--token TOKEN` or set `HF_TOKEN` when needed; the script also checks
the local Hugging Face token cache.

Main-model downloads update the legacy `q36moe.gguf` link. The runtime default
is `gguf/Qwen3.6-35B-A3B-IQ2XXS-Q2K-Q8.gguf`, so pass `-m FILE` to select a
dense model or another supported release GGUF. The binaries do not follow
the legacy link automatically.

All three downloads include an MTP head. Enable it with `--mtp`; see
[speculative decoding](SPECULATIVE_DECODING.md) for sampling and backend limits.

## Qwen3.6-35B-A3B

The Q2 recipe uses IQ2_XXS gate/up and Q2_K down projections for the routed
MoE experts, which account for most of the weights. Shared experts,
projections, routing, and other tensors retain higher precision. The release
file is about 12.1 GB including its 333 MB embedded MTP head, leaving room
for context and the OS on the BC-250.

The download comes from
[Ninnix96/Qwen3.6-35B-A3B-gguf](https://huggingface.co/Ninnix96/Qwen3.6-35B-A3B-gguf).
For a smaller machine, use [SSD streaming](SSD_STREAMING.md).

## Qwen3.8-27B

The IQ3_S and GSQ-RCO IQ3_XXS releases use the same CLI, server, agent,
evaluator, session cache, and Qwen3.8 vision projector. GSQ-RCO replaces the
earlier unofficial UD-IQ3_XXS option; its mixed formats, including BF16
recurrent projections, run natively on the GPU. It has a separate disk cache
namespace and agent checkpoint identity, so switching quantizations cannot
restore incompatible model state.

Download and start the smaller dense model:

```sh
./download_model.sh 27b-iq3-xxs
./q36-server -m gguf/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf --mtp --vulkan
```

For IQ3_S:

```sh
./download_model.sh 27b
./q36-server -m gguf/Qwen3.8-27B-UD-IQ3_S.gguf --vulkan
```

The downloads come from
[unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
and [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF).
Dense weights must fit resident; the MoE expert-streaming cache does not
apply to them.

## Vision

Download the encoder for the selected language model:

```sh
./download_model.sh qwen36-vision
./download_model.sh qwen38-vision
```

Encoder downloads do not change the legacy main-model link.

Q36 accepts the matching Qwen3-VL projector as a disk-backed sidecar. It maps
the file without copying the model and streams one weight tensor at a time
through a reusable Vulkan buffer. The full 0.9 GiB projector is never retained
in RAM or VRAM, and streamed pages are released after each dispatch.

```sh
./q36 --vision gguf/Qwen3.6-35B-A3B-mmproj-F16.gguf
./q36 -m gguf/Qwen3.8-27B-UD-IQ3_S.gguf \
  --vision gguf/Qwen3.8-27B-mmproj-F16.gguf
```

At the interactive prompt, `/read photo.jpg` and `/read image.png` submit an
image turn. An optional prompt can be passed along with the image (e.g.
`/read photo.jpg Describe this image` or `/read "path/with spaces/img.png" What is here?`).
JPEG and PNG decoding is built in. The sidecar output dimension is
validated against the selected standard or dense language model before use.
Vision prompt execution currently requires Vulkan. `q36-agent --vision FILE`
exposes `view_image` for local files. `q36-server --vision FILE` accepts OpenAI
image URLs containing PNG/JPEG data URIs, Responses image blocks, and Anthropic
base64 images, including images returned by tools. Remote image URLs are rejected.
The server caches at most 32 MiB of decoded image embeddings and encoded keys;
the projector weights remain streamed from disk. Prefix reuse checks the image
fingerprints and geometry, so replacing pixels with the same number of image
pads rebuilds the context. Vision sessions currently cannot be saved as agent
sessions or disk KV checkpoints.

The vendored JPEG/PNG decoder in `third_party/iris` is Copyright (c) 2026
Salvatore Sanfilippo and distributed under its included MIT license.

## Build weights

If you want to regenerate GGUF files, quantize community fine-tunes / abliterated models
(via `--strip-nextn`), or collect a new imatrix, see
[gguf-tools/README.md](../gguf-tools/README.md). Those tools are meant for offline
Qwen3.6 model-building work. The native quantizer accepts Q8, F16, or BF16
inputs; imatrix collection still uses the optional llama.cpp tooling.
