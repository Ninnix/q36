#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
compiler=${METALC:-}
if [ -z "$compiler" ]; then
    for candidate in \
        /private/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-*/Metal.xctoolchain/usr/bin/metal
    do
        if [ -x "$candidate" ]; then
            compiler=$candidate
            break
        fi
    done
fi

if [ -z "$compiler" ] || [ ! -x "$compiler" ]; then
    echo "q36: Metal compiler not found; run: xcodebuild -downloadComponent MetalToolchain" >&2
    exit 1
fi

source_file=$(mktemp "${TMPDIR:-/tmp}/q36-metal-XXXXXX.metal")
module_cache=$(mktemp -d "${TMPDIR:-/tmp}/q36-metal-cache-XXXXXX")
air_file=${1:-"${TMPDIR:-/tmp}/q36-metal.air"}
trap 'rm -f "$source_file"; rm -rf "$module_cache"' EXIT HUP INT TERM

for name in \
    q36_common flash_attn dense moe dsv4_hc unary dsv4_kv dsv4_rope \
    dsv4_misc argsort cpy concat get_rows sum_rows softmax repeat glu \
    norm bin set_rows q36_ops
do
    sed -n '1,$p' "$root/metal/$name.metal" >> "$source_file"
done

"$compiler" -fmodules-cache-path="$module_cache" -c "$source_file" -o "$air_file"
