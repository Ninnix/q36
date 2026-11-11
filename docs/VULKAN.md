# Vulkan on Linux

[README](../README.md#start-here) | [BC-250 setup](BC250.md) | [Testing](TESTING.md)

The AMD BC-250 with Mesa RADV is the tested Vulkan target. Follow the
[board setup](BC250.md) for packages, GPU-visible memory, and the governor.

## Build and run

From the repository root:

```sh
make vulkan-bc250
./download_model.sh q2
./q36
```

The build selects Vulkan by default and detects the BC-250 fast path
automatically. No backend flag is needed.

The build produces `q36`, `q36-server`, `q36-agent`, `q36-bench`,
`q36-eval`, and `q36_test`. Run from the project tree so the backend can
find the compiled shaders in `vulkan/`. Binary packages need that directory.

Vulkan 1.1, shader `float64`, and shader `int64` are required. Subgroup
arithmetic, 16-bit storage, and native `float16` are detected at runtime.
Windows is not currently a build target.

## Other devices

The BC-250 target checks the device identity at startup. The generic Vulkan
target uses the same backend and shaders without that restriction, selecting
portable fallbacks when optional GPU features are missing. It is available
for porting; other GPUs have not been validated.

Before calling another device supported, run the [regression checks](TESTING.md#regression-checks),
CPU/GPU parity, and context sweeps on the actual hardware. Removing the device
check alone does not establish support.

## Native builds

Vulkan, [Metal](METAL.md), and the CPU reference are separate native builds.
They produce the same executable names, so changing the runtime overwrites
the local binaries. Keep separate worktrees when retaining several builds.
Each GPU binary rejects the other runtime's backend option before loading
weights.

Normal builds are self-contained. They need no llama.cpp checkout or GGML
libraries; those are used only by explicit reference-test targets. CPU
inference is intended for reference and debugging.
