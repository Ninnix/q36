# Qwen3.6-35B-A3B model card synopsis

This document summarizes the Q36 target model and GGUF facts that matter for
local inference, Vulkan development, and benchmark interpretation.

Sources:

- GGUF repo: https://huggingface.co/Ninnix96/Qwen3.6-35B-A3B-gguf
- Base model metadata: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- Local GGUF inspected: `gguf/Qwen3.6-35B-A3B-AntirezExperts-IQ2XXS-gateup-Q2K-down-Q8rest.gguf`

The GGUF repository currently has no detailed upstream model-card body. The
facts below are taken from the target GGUF metadata and from Q36's runtime
assumptions.

## Model Family

Qwen3.6-35B-A3B is a Qwen Mixture-of-Experts language model.

| Model | Total parameters | Active parameters | Context length |
|---|---:|---:|---:|
| Qwen3.6-35B-A3B | 35B | about 3B | 256K tokens |

Q36 is intentionally not a general GGUF runner. It targets this model family
and, by default, the AntirezExperts GGUF:

```text
Qwen3.6-35B-A3B-AntirezExperts-IQ2XXS-gateup-Q2K-down-Q8rest.gguf
```

## Architecture

The inspected GGUF reports:

| Field | Value |
|---|---:|
| Architecture | `qwen35moe` |
| Transformer blocks | 40 |
| Embedding length | 2048 |
| Native context | 262144 tokens |
| Attention heads | 16 |
| KV heads | 2 |
| Key length | 256 |
| Value length | 256 |
| RoPE dimension count | 64 |
| RoPE sections | `[11, 11, 10, 0]` |
| RoPE frequency base | 10000000 |
| RMS norm epsilon | 1e-6 |
| Experts | 256 |
| Active experts per token | 8 |
| Expert FFN length | 512 |
| Shared expert FFN length | 512 |
| Full-attention interval | 4 |

The model is not a plain full-attention transformer at every layer. The GGUF
metadata includes the Qwen3.6 recurrent/SSM fields used by Q36's graph:

| Field | Value |
|---|---:|
| SSM convolution kernel | 4 |
| SSM state size | 128 |
| SSM group count | 16 |
| SSM time-step rank | 32 |
| SSM inner size | 4096 |

Q36 maps these metadata values to a fixed Vulkan execution graph with full
attention layers, recurrent layers, routed MoE, shared experts, and model
specific RoPE handling.

## Precision And Weights

The target GGUF uses an asymmetric quantization scheme designed for Q36's 16 GB
unified-memory target:

| Tensor group | Quantization |
|---|---|
| Routed expert gate weights | `IQ2_XXS` |
| Routed expert up weights | `IQ2_XXS` |
| Routed expert down weights | `Q2_K` |
| Shared experts, attention, recurrent, projections | mostly `Q8_0` |
| Norms, gates, biases and small tensors | `F32` |

The inspected file has 733 tensors with this type count:

| Type | Tensor count |
|---|---:|
| `F32` | 301 |
| `Q8_0` | 312 |
| `IQ2_XXS` | 80 |
| `Q2_K` | 40 |

The routed MoE tensors dominate the weight footprint, so quantizing only those
paths yields a roughly 10-11 GB model while keeping non-expert components at
higher precision. Q36's Vulkan path uploads GGML IQ codebook tables once and
binds them to the routed MoE kernels. For the default GGUF, `IQ2_XXS` is the
IQ table that is required by the gate/up experts; the runtime also keeps the
`IQ2_S` and `IQ3_S` codebooks for generic routed-matvec coverage.

GGUF metadata for the inspected file also reports:

| Field | Value |
|---|---|
| Quantized by | `Ninnix96` |
| Quantization version | 2 |
| File type | 7 |
| imatrix entries | 120 |
| imatrix chunks | 6249 |

## Tokenizer And Prompting

The GGUF tokenizer metadata is:

| Field | Value |
|---|---|
| tokenizer model | `gpt2` |
| tokenizer pre-tokenizer | `qwen35` |
| tokens | 248320 |
| EOS token ID | 248046 |
| BOS token ID | 248044 |
| padding token ID | 248055 |
| add BOS | false |

Q36 renders ChatML prompts with Qwen protocol tokens such as `<|im_start|>` and
`<|im_end|>`. Thinking mode is controlled with native `<think>...</think>`
blocks. Q36 defaults to thinking mode; explicit non-thinking requests render
the chat-template flag equivalent of `enable_thinking=false`. Historical
reasoning is preserved by default for stable autoregressive and KV-cache
prefixes; clients can opt out with `preserve_thinking=false`.

Tool definitions are rendered in the system prompt inside `<tools>...</tools>`.
Calls use Qwen3.6's native tagged form: `<tool_call>` contains a
`<function=name>` block whose arguments are `<parameter=name>` blocks. These
markers are plain ASCII and tokenize as ordinary BPE pieces, so Q36's streaming
parser treats them as syntax rather than special tokens.

## Local Running Notes

Q36 targets AMD BC-250 on Linux using Vulkan/RADV and
`VK_EXT_external_memory_host`, mapping the GGUF directly as a host-backed
Vulkan buffer. The CPU backend exists for correctness checks and diagnostics.

The model's native context is 256K tokens, but the live context window is
limited by the memory left after weights and runtime scratch buffers. Q36's
disk KV cache reuses prefixes across stateless API requests and server
restarts; it does not extend the active context beyond the allocated window.

Default sampling metadata in the GGUF is:

| Setting | Value |
|---|---:|
| temperature | 1.0 |
| top_p | 0.95 |
| top_k | 20 |

Q36 ignores the GGUF sampling metadata and defaults to `temperature=1.0`,
`top_p=1.0`, `top_k=0`, and `min_p=0.05`: on the Q2 expert mix any fixed-mass
or fixed-count truncation admits quantization-noise tail tokens, while the
min-p floor scales with the top token's confidence.

## Licensing

The inspected GGUF metadata reports the base model license as Apache-2.0 and
links to the Qwen model license page.
