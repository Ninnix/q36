# Metal on Apple Silicon

[README](../README.md#start-here) | [Models](MODELS.md) | [SSD streaming](SSD_STREAMING.md)

## Build and run

Use an Apple Silicon M1 or newer Mac with macOS 11 or newer. Install Xcode or
the Command Line Tools for the SDK and Metal framework:

```sh
xcode-select --install
```

From the repository root:

```sh
make metal
./download_model.sh q2
./q36 --metal
```

The build produces the same CLI, agent, server, benchmark, and evaluator as
Vulkan. Shader sources compile through Metal at runtime; a separate offline
`metal` invocation is unnecessary. Run commands from the repository root so
the runtime can find `metal/*.metal`.

## Memory

On a 16 GB Mac, start with the resident Qwen3.6 Q2 model and leave memory for
the context and runtime buffers. On an 8 GB Mac, start with a bounded expert
cache and a smaller context:

```sh
./q36 --metal --ssd-streaming --ssd-streaming-cache-experts 512 --ctx 4096
```

The cache size is additional to non-routed weights, context, and scratch.
See [SSD streaming](SSD_STREAMING.md#metal-cache) before increasing it.
Dense Qwen3.8 requires resident weights; use the supported smaller quant when
memory is tight, and check the startup estimate.

## Device and compatibility policy

A normal local Metal process uses `MTLCreateSystemDefaultDevice`.
Apple Silicon exposes its unified GPU as one logical Metal device, so Q36 does
not require or expose a device list for M1, M2, M3, M4, or later families.
There is no layer split or multi-Mac distributed mode: one Q36 process owns one
default local Metal device and one model. Run only one large model process at a
time when validating memory and performance.

The Metal binary targets macOS 11. Newer facilities are optional:

* macOS 15 residency sets and `MTLMathMode` are selected only after runtime
  availability checks.
* Older releases use the same shared/no-copy buffers and baseline Metal
  kernels without residency sets.
* Host code defaults to the Apple M1 instruction baseline, so a binary built
  on a newer Mac remains usable on M1. Use
  `NATIVE_CPU_FLAG=-mcpu=native make metal` only for a local-only build.
* Routed MoE prefill follows DS4's expert-major batch-MM dispatch and selects
  the map specialization for the model's actual routed-expert count. If the
  shape, pipeline, thread count, or threadgroup memory is unsupported, Metal
  automatically uses the exact matvec path. `Q36_METAL_MOE_MM=0` forces that
  fallback for diagnostics.
* Cache auto-sizing uses the device's `recommendedMaxWorkingSetSize`, not a
  hard-coded machine-memory percentage.
* All model buffers use unified storage; no discrete-GPU transfer or eGPU
  placement path is provided.

The minimum deployment target proves link compatibility, not every driver and
GPU generation. Release QA must still run on a physical M1 with the oldest
supported macOS, a representative newer Mac, a 16 GB resident configuration,
and a physical 8 GB SSD-streaming configuration.

Metal shaders are compiled at runtime from `metal/*.metal`. Run the CLI,
server, benchmark, and evaluation harness from the Q36 project tree, and ship
the `metal` directory with binary packages. A frontend that changes working
directory before model startup must provide absolute paths through
`Q36_METAL_DENSE_SOURCE`, `Q36_METAL_MOE_SOURCE`,
`Q36_METAL_NORM_SOURCE`, `Q36_METAL_OPS_SOURCE`,
`Q36_METAL_RECURRENT_SOURCE`, `Q36_METAL_KV_SOURCE`, and
`Q36_METAL_ATTN_SOURCE`.

## Next steps

- [Models and vision](MODELS.md): vision execution currently requires Vulkan.
- [Batched serving](SERVER.md#multiple-sessions).
- [Measurements and benchmark commands](PERFORMANCE.md).
- [Metal regression and release checks](TESTING.md).
