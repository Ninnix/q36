#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vk=${PS5_VULKAN_ROOT:?set PS5_VULKAN_ROOT to PS5_Vulkan built as described in docs/PS5.md}
vk=$(cd -- "$vk" && pwd)
export PS5_VULKAN_ROOT="$vk"
export PS5_PAYLOAD_SDK="$vk/.deps/native/ps5-payload-sdk"
sdk=$PS5_PAYLOAD_SDK
out="$root/build/ps5"
archive="$vk/.deps/native/radv-release/lib/libvulkan_radeon.ps5.a"
tool="$vk/build/host/ps5-native-tool"
jobs=${BUILD_JOBS:-2}
[[ $jobs =~ ^[1-9][0-9]*$ ]] || { echo 'ps5: invalid BUILD_JOBS' >&2; exit 2; }
[[ $# == 0 || ( $# == 1 && $1 == --objects-only ) ]] || {
    echo 'usage: ps5/build.sh [--objects-only]' >&2; exit 2;
}
for command in clang clang++ glslc make python3; do
    command -v "$command" >/dev/null || { echo "ps5: missing $command" >&2; exit 2; }
done
for file in "$vk/tooling/prospero-clang18" "$sdk/bin/prospero-lld" "$sdk/target/include/stdio.h"; do
    [[ -f $file ]] || { echo "ps5: missing $file" >&2; exit 2; }
done
mkdir -p "$out/vulkan" "$out/headers" "$out/stubs"
# Do not expose the host's libc headers to the target compiler.
header_root=${VULKAN_HEADERS_ROOT:-/usr/include}
for directory in vulkan vk_video; do
    cp -r "$header_root/$directory" "$out/headers/"
done
python3 "$root/ps5/gen_dispatch.py" "$root/q36_vulkan.c" "$out"
revision=$(git -C "$root" describe --always --dirty)
make -B -C "$out" -f "$root/ps5/Makefile" -j "$jobs" \
    ROOT="$root" CC="$root/ps5/cc" HEADERS="$out/headers" \
    GLSLC="$(command -v glslc)" REVISION="$revision" objects
if [[ ${1:-} == --objects-only ]]; then
    echo "ps5: target objects and SPIR-V compiled in $out; title linking and console execution untested"
    exit 0
fi
for file in "$archive" "$tool" "$vk/runtime/libc.prx"; do
    [[ -f $file ]] || { echo "ps5: missing $file; see docs/PS5.md" >&2; exit 2; }
done
cxx() { PS5_CLANG="${PS5_CLANG_CXX:-clang++}" sh "$vk/tooling/prospero-clang18" -std=c++20 -O2 "$@"; }
cxx -fno-exceptions -fno-rtti -c "$vk/tooling/native/app_crt.cpp" -o "$out/app_crt.o"
cxx -fno-exceptions -fno-rtti -c "$vk/tooling/native/app_cpp_runtime.cpp" -o "$out/app_cpp_runtime.o"
"$root/ps5/cc" -O2 -c "$root/ps5/compat.c" -o "$out/compat.o"
"$sdk/bin/llvm-ar" rcs "$out/libq36_compat.a" "$out/compat.o"

# The driver calls AGC through system-module imports, as PS5LM does.
imports=$("$sdk/bin/llvm-nm" -u "$archive" | awk '{print $NF}' | grep '^sceAgc' | sort -u)
for lib in libSceAgc libSceAgcDriver; do
    stub="$out/stubs/$lib.c"
    : > "$stub"
    for name in $imports; do
        case $lib:$name in
            libSceAgcDriver:sceAgcDriver*) echo "void $name(void) {}" >> "$stub" ;;
            libSceAgc:sceAgcDriver*) ;;
            libSceAgc:*) echo "void $name(void) {}" >> "$stub" ;;
        esac
    done
    "$root/ps5/cc" -O2 -fPIC -c "$stub" -o "$out/stubs/$lib.o"
    "$sdk/bin/prospero-lld" --shared -soname "$lib.prx" -o "$out/stubs/$lib.so" "$out/stubs/$lib.o"
done

# radv-link.sh expects the Debian/Arch builtins layout. Cover other layouts too.
builtins=$(clang --print-file-name=libclang_rt.builtins.a)
[[ -f $builtins ]] || builtins="$(clang --print-resource-dir)/lib/linux/libclang_rt.builtins-x86_64.a"
mkdir -p "$out/clang/resource/lib/linux"
ln -sf "$builtins" "$out/clang/resource/lib/linux/libclang_rt.builtins-x86_64.a"
printf '#!/bin/sh\n[ "$1" = --print-resource-dir ] && { echo "%s"; exit 0; }\nexec clang "$@"\n' \
    "$out/clang/resource" > "$out/clang/clang"
chmod +x "$out/clang/clang"
source "$vk/tools/radv-link.sh"
PS5_CLANG="$out/clang/clang" radv_link_recipe "$vk" "$sdk" "$archive"
"$sdk/bin/prospero-lld" "${radv_linker_script[@]}" "${radv_link_flags[@]}" \
    --version-script "$vk/tooling/native/app-symbols.map" --exclude-libs=ALL --wrap=exit \
    -e _start -o "$out/llvm-pie.elf" "$out/app_crt.o" "$out/app_cpp_runtime.o" \
    "$out/main.o" "$out/dispatch.o" "$out"/engine-*.o "$out/libq36_compat.a" \
    "$out/stubs/libSceAgc.so" "$out/stubs/libSceAgcDriver.so" \
    "${radv_link_inputs[@]}" --as-needed "$sdk"/target/lib/*.so
"$tool" link --in "$out/llvm-pie.elf" --out "$out/eboot.elf" \
    --stub-dir "$sdk/target/lib" --stub "$out/stubs/libSceAgc.so" \
    --stub "$out/stubs/libSceAgcDriver.so" --module-sdk 0x02000009 \
    --companion-sdk 0x08050001 --file-name eboot.elf
app="$out/title/PPSA99576"
mkdir -p "$app/sce_sys" "$app/sce_module" "$app/vulkan"
"$tool" self --sign --in "$out/eboot.elf" --out "$app/eboot.bin" --magic 0x1D3D154F
python3 - "$app/sce_sys/param.json" <<'PY'
import json
import sys
from pathlib import Path
param = {
    'titleId': 'PPSA99576', 'contentId': 'UP9000-PPSA99576_00-Q36SMOKETEST0001',
    'conceptId': '99576', 'contentVersion': '01.000.000', 'masterVersion': '01.00',
    'applicationCategoryType': 0, 'applicationDrmType': 'free', 'contentBadgeType': 1,
    'attribute': 0, 'attribute2': 0, 'attribute3': 0, 'downloadDataSize': 16384,
    'requiredSystemSoftwareVersion': '0x0000000000000000', 'sdkVersion': '0x0000000000000000',
    'localizedParameters': {'defaultLanguage': 'en-US', 'en-US': {'titleName': 'q36 smoke test'}},
}
Path(sys.argv[1]).write_text(json.dumps(param, indent=2) + '\n')
PY
cp "$vk/runtime/libc.prx" "$app/sce_module/"
cp "$out"/vulkan/*.spv "$app/vulkan/"
cp "$root/LICENSE" "$app/q36-LICENSE.txt"
"$tool" self --inspect --file "$app/eboot.bin"
printf 'q36 %s\nPS5_Vulkan %s\nSDK %s\n' "$revision" \
    "$(git -C "$vk" rev-parse HEAD)" "$(cat "$sdk/.ps5-sdk-revision")" > "$app/build-info.txt"
cp "$vk/.deps/native/radv-release/PROVENANCE.txt" "$app/radv-PROVENANCE.txt"
bash "$vk/tools/inspect.sh" "$out/eboot.elf"
bash "$vk/tools/inspect.sh" "$app/eboot.bin"
echo "ps5: title packaged in $app; console execution untested"
