# Vulkan on Linux

[README](../README.md#start-here) | [BC-250 setup](BC250.md) | [Testing](TESTING.md)

The AMD BC-250 with Mesa RADV is the tested Vulkan target. The backend also
queries other devices and provides portable kernel fallbacks, but removing
the device restriction is not evidence that an untested GPU works correctly.
Use the [regression checks](TESTING.md#regression-checks) when adapting
QuarkStar to another device.

## Build and run

Install a C toolchain, Vulkan headers and loader, the GPU driver, and
`glslangValidator`. The backend requires Vulkan 1.1 or newer and shader
`float64` and `int64`. Subgroup arithmetic, 16-bit storage, and native
`float16` are detected at runtime.

```sh
make
./download_model.sh q2
./q36 --vulkan -p "Explain Redis streams in one paragraph."
```

The [BC-250 guide](BC250.md) covers package commands, RADV, memory settings,
and the governor for that board. Windows is not currently a build target.

## Device selection and validation

The generic and BC-250 targets contain the same Vulkan backend and SPIR-V.
`vulkan-generic` queries device capabilities and uses portable fallbacks when
the subgroup or 16-bit features required by a tuned kernel are absent. A
BC-250 is recognized at runtime and keeps the current optimized path.
`vulkan-bc250` adds a vendor/device check so a board-specific release artifact
cannot silently run on different hardware.

The generic target removes the hard device identity restriction, not the need
to validate a new GPU. Before calling another Vulkan device supported, run the
isolated kernel suite and the short CPU/GPU parity gate, then record prefill
and decode throughput at the intended context sizes.

## Native builds

Q36 is multi-runtime at the source and API level, but Metal and Vulkan are
separate native builds rather than one fat executable. Each build produces the
same command names—`q36`, `q36-server`, `q36-bench`, `q36-agent`, `q36-eval`,
and `q36_test`—linked to the selected graph runtime:

| Platform/runtime | Build | Explicit invocation |
| --- | --- | --- |
| Linux / generic Vulkan | `make` or `make vulkan-generic` | `./q36 --vulkan -p "Hello"` |
| Linux / BC-250 checked Vulkan | `make vulkan-bc250` | `./q36 --vulkan -p "Hello"` |
| macOS / Metal | `make metal` | `./q36 --metal -p "Hello"` |
| CPU reference | `make cpu` | `./q36 --cpu -p "Hello"` |

Building another runtime overwrites those local executable names. Use separate
worktrees or copy build artifacts if Metal, Vulkan, and CPU binaries must be
kept side by side. The `--backend metal|vulkan|cpu` spelling is equivalent to
the short backend switches and is accepted by the CLI, server, agent,
benchmark, and evaluation harness. A Metal-linked binary rejects `--vulkan`,
and a Vulkan-linked binary rejects `--metal`, so deployment mistakes fail
before model loading.

The normal and CPU-only builds are self-contained. They do not require a
llama.cpp checkout or GGML libraries. Those are used only by explicit optional
reference targets, including `make test-llama` and `make test-vectors-local`.

## CPU reference

Do not treat the CPU path as the production target. The CLI and `q36-server`
support the CPU backend for reference/debug use and share the same KV
session and snapshot format as Metal and Vulkan, but normal inference should
use a GPU graph runtime.
