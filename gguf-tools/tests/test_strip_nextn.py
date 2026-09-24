#!/usr/bin/env python3

import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def scalar(key, kind, value):
    fmt = {2: "H", 4: "I", 5: "i"}[kind]
    return string(key) + struct.pack("<I" + fmt, kind, value)


def write_gguf(path, name, block_count, shard=None, split_count=2):
    metadata = [scalar("general.alignment", 4, 32),
                scalar("qwen35moe.block_count", 4, block_count)]
    if block_count == 41:
        metadata.append(scalar("qwen35moe.nextn_predict_layers", 4, 1))
    if shard is not None:
        metadata += [scalar("split.no", 2, shard), scalar("split.count", 2, 2),
                     scalar("split.tensors.count", 5, split_count)]
    info = string(name) + struct.pack("<I Q I Q", 1, 8, 0, 0)
    header = struct.pack("<4sIQQ", b"GGUF", 3, 1, len(metadata)) + b"".join(metadata) + info
    payload = struct.pack("<8f", *range(shard or 0, (shard or 0) + 8))
    path.write_bytes(header + bytes(-len(header) % 32) + payload)
    return payload


def inspect(path):
    data = path.read_bytes()
    magic, version, n_tensors, n_kv = struct.unpack_from("<4sIQQ", data)
    assert (magic, version) == (b"GGUF", 3)
    pos = 24

    def read_string():
        nonlocal pos
        length = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        value = data[pos:pos + length].decode()
        pos += length
        return value

    metadata = {}
    for _ in range(n_kv):
        key = read_string()
        kind = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        fmt = {2: "H", 4: "I", 5: "i"}[kind]
        metadata[key] = struct.unpack_from("<" + fmt, data, pos)[0]
        pos += struct.calcsize(fmt)
    tensors = {}
    for _ in range(n_tensors):
        name = read_string()
        rank = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        assert rank == 1
        size, kind, offset = struct.unpack_from("<QIQ", data, pos)
        pos += 20
        assert (size, kind) == (8, 0)
        tensors[name] = offset
    start = (pos + 31) // 32 * 32
    return metadata, {name: data[start + offset:start + offset + 32]
                      for name, offset in tensors.items()}


def convert(binary, inputs, out, strip=False, good=True):
    args = [binary]
    for path in inputs:
        args += ["--in", str(path)]
    args += ["--out", str(out), "--allow-synthetic-imatrix"]
    if strip:
        args.append("--strip-nextn")
    result = subprocess.run(args, text=True, capture_output=True)
    if good:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0 and "GGUF shard tensor count mismatch" in result.stderr


def main():
    binary = str(Path(sys.argv[1] if len(sys.argv) > 1 else "gguf-tools/qwen36-quantize").resolve())
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        mtp_name = "blk.40.nextn.enorm.weight"
        base_name = "output_norm.weight"
        mtp = root / "mtp.gguf"
        base = root / "base.gguf"
        mtp_data = write_gguf(mtp, mtp_name, 41, shard=0)
        base_data = write_gguf(base, base_name, 41, shard=1)

        stripped = root / "stripped.gguf"
        convert(binary, [mtp, base], stripped, strip=True)
        metadata, tensors = inspect(stripped)
        assert metadata["qwen35moe.block_count"] == 40
        assert "qwen35moe.nextn_predict_layers" not in metadata
        assert "split.tensors.count" not in metadata
        assert tensors == {base_name: base_data}

        kept = root / "kept.gguf"
        convert(binary, [mtp, base], kept)
        metadata, tensors = inspect(kept)
        assert metadata["qwen35moe.block_count"] == 41
        assert metadata["qwen35moe.nextn_predict_layers"] == 1
        assert tensors == {mtp_name: mtp_data, base_name: base_data}

        bad = root / "bad.gguf"
        write_gguf(bad, mtp_name, 41, shard=0, split_count=3)
        convert(binary, [bad, base], root / "rejected.gguf", strip=True, good=False)

        standard = root / "standard.gguf"
        write_gguf(standard, base_name, 40)
        for strip in (False, True):
            out = root / f"standard-{strip}.gguf"
            convert(binary, [standard], out, strip=strip)
            metadata, tensors = inspect(out)
            assert metadata["qwen35moe.block_count"] == 40
            assert list(tensors) == [base_name]
    print("strip-nextn: split strip, split keep, count validation, and standard path OK")


if __name__ == "__main__":
    main()
