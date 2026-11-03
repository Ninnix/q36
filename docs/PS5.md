# Native PS5 smoke test

This branch prepares the native homebrew title requested in
[PS5LM issue #48](https://github.com/cobanov/PS5LM/issues/48#issuecomment-6555518117).
It targets the PS5 SDK and PS5_Vulkan's RADV, with the same one-prompt invocation
proposed at q36 `d9a665a`. The engine sources come from this branch, including
subsequent upstream fixes. Retail-console model loading and generation are
**unverified**. A cross-build is not the console proof requested in the issue.

The output is a separate title, `PPSA99576`, named **q36 smoke test**. It runs
the existing q36 CLI once, writes a log, and closes through
`sceSystemServiceLoadExec("exit")`. There is no dashboard or engine selector.

## Toolchain

Use x86-64 Linux and the public homebrew tools used by PS5LM. No console or
proprietary firmware is needed for compilation. Host prerequisites are listed
in PS5LM's [BUILDING.md](https://github.com/cobanov/PS5LM/blob/main/docs/BUILDING.md)
and [native CI workflow](https://github.com/cobanov/PS5LM/blob/main/.github/workflows/native-build.yml).
The engine has no new library dependency; the PS5 title links the platform SDK,
RADV, and its C++ compiler runtime.

Prepare the same revisions as PS5LM's native build at
`80a38fe2d0092a97cbd7ff0d7548c30d54eb2b07`. The PS5LM copy only supplies the three
Mesa patches; its llama.cpp and UI submodules are not needed:

```sh
git clone https://github.com/cobanov/PS5LM.git /tmp/PS5LM
git -C /tmp/PS5LM checkout 80a38fe2d0092a97cbd7ff0d7548c30d54eb2b07
mkdir -p /tmp/PS5LM/.deps/src
cd /tmp/PS5LM/.deps/src
git clone https://github.com/mihawk-99/PS5_Vulkan.git
git -C PS5_Vulkan checkout f410e67a0241b3e8edcdbcd2a9b285dfc4a412ed
git clone https://github.com/mihawk-99/PS5_PayloadSDK.git
git -C PS5_PayloadSDK checkout b83202be73e930050e00de0f1b4d9ec46c0391bf
git clone https://github.com/mihawk-99/PS5_Mesa.git
git -C PS5_Mesa checkout 1ba9874f478657ccf6da72ce55f0260164cc0bd0
git -C PS5_Mesa am /tmp/PS5LM/patches/mesa/*.patch
export PS5_VULKAN_ROOT=/tmp/PS5LM/.deps/src/PS5_Vulkan
cd "$PS5_VULKAN_ROOT"
bash tools/setup-native-dependencies.sh
PS5_MESA_REVISION=$(git -C ../PS5_Mesa rev-parse HEAD) bash tools/build-radv.sh release
make libc
mkdir -p build/host
cp build/runtime-shim/ps5-native-tool build/host/
```

Keep the terminal environment and return to the q36 checkout:

```sh
make test-ps5                       # host test of ICD dispatch and failure cleanup
BUILD_JOBS=2 make ps5               # compile, link and sign the native title
```

`ps5/build.sh --objects-only` compiles target objects and all SPIR-V without
requiring a built RADV archive or signing tool. `VULKAN_HEADERS_ROOT` can name
the host directory containing `vulkan/` and `vk_video/` (default `/usr/include`).
All generated files live under `build/ps5`, separate from host binaries and
objects. The title folder is `build/ps5/title/PPSA99576`; it contains `eboot.bin`,
`sce_sys/param.json`, `sce_module/libc.prx`, and the runtime `vulkan/*.spv` files.
`build-info.txt` records the engine, PS5_Vulkan, and SDK revisions;
`radv-PROVENANCE.txt` records the driver build.

The Vulkan entry points are resolved from the statically linked RADV ICD after
instance creation. No Linux Vulkan loader or BC-250 device-ID requirement is
used. The SDK compiler targets `x86_64-sie-ps5`, with Zen 2 CPU instructions.
The RADV link recipe supplies direct-memory allocation and thread wrappers.
`isatty` reports no terminal; `posix_fadvise` is a local cache
hint stub because no title module exports it. Engine error exits use the same
title shutdown path as normal completion.

On 2026-11-03, the full cross-build passed with Clang 23.1, the revisions above,
and PS5LM's three Mesa patches. The packaged ELF and FSELF each reported
`StaticErrors: 0`, the signed container reported valid integrity, and all 197
SPIR-V shaders were included. `make test-ps5` also passed its host dispatch and
failure-cleanup tests. No console was available, so this records build and
static package validation only.

## Console handoff

Use the HEN, ShadowMountPlus, and firmware combination already working with
PS5LM. This branch establishes no additional supported firmware versions.
Close PS5LM before testing so its GPU allocations are released. Install the
complete title folder under `/data/homebrew/PPSA99576`, register it through
ShadowMountPlus, and launch **q36 smoke test** from the home screen for the
foreground user. Sending `eboot.elf` directly to elfldr would run the wrong
kind of process and is not this test.

Allow the new title access to `/data`, as PS5LM requires. For OnionHEN that means
adding `PPSA99576` to `app_jailbreak.exact_title_ids`. The title sends the same
`/download0/etahen_jailbreak` PID request as PS5LM and waits up to ten seconds.
It also locates its shaders under `/mnt/sandbox/PPSA99576_<slot>/app0` when the
HEN moves it outside the sandbox. The default model path is:

```text
/data/PS5LM/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf
```

Download the model on the PC with `./download_model.sh 27b-iq3-xxs` and transfer
it with the console's payload FTP server, or reuse the existing file. Do not
bundle the model into the title. For a different location, place its absolute
console path on the first line of `/data/q36/model.txt`. The app uses
`/download0/q36` when it cannot write `/data/q36/smoke.log`; in that case `model.txt`
and `smoke.log` live there. A USB model path must also be visible to the title.

Each launch uses these fixed arguments, with MTP disabled:

```sh
Q36_THREADS=2 ./q36 --vulkan \
  -m /data/PS5LM/models/Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf \
  --ctx 2048 -t 2 --temp 0 -n 64 \
  -p "Explain what a radix tree is."
```

The title writes stdout and stderr unbuffered to `/data/q36/smoke.log`, replacing
the previous run. After a normal run, the log includes the generated text,
q36's generation statistics, `q36-ps5: one-prompt returned 0`, and
`q36-ps5: finished status=0`. A nonzero status or missing completion line is
not a pass. Check that generation actually produced tokens; initialization or
model inspection alone does not count. The TV has no progress UI.

For the issue, share the log, `build-info.txt`, `radv-PROVENANCE.txt`, model
filename and SHA-256, firmware, HEN and ShadowMountPlus versions. Include the
Vulkan device/features, model-load result, generated output and token count.
If it fails, keep the log and kernel log from that run. Memory mapping, pool
capacity, shader execution, shutdown and performance all still need testing
on a retail console; BC-250/Linux results do not establish them.
