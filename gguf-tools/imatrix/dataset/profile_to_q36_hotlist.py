#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


def load(path: Path, limit: int) -> list[tuple[int, int, int]]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(f"bad profile line: {line}")
            layer, expert, hits = map(int, fields[:3])
            if not 0 <= layer < 40 or not 0 <= expert < 256 or hits <= 0:
                raise ValueError(f"bad profile entry: {line}")
            out.append((layer, expert, hits))
            if len(out) == limit:
                break
    return out


def write(path: Path, entries: list[tuple[int, int, int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("/* Generated from a balanced rendered-prompt router profile. */\n")
        f.write("static const uint32_t q36_default_streaming_hotlist[][3] = {\n")
        for layer, expert, hits in entries:
            f.write(f"    {{{layer}, {expert}, {hits}}},\n")
        f.write("};\n")
        f.write(
            "static const uint32_t q36_default_streaming_hotlist_count =\n"
            "    sizeof(q36_default_streaming_hotlist) /\n"
            "    sizeof(q36_default_streaming_hotlist[0]);\n"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", type=Path)
    ap.add_argument("--out", type=Path, default=Path("q36_streaming_hotlist.inc"))
    ap.add_argument("--limit", type=int, default=4096)
    args = ap.parse_args()
    if args.limit <= 0:
        ap.error("--limit must be positive")
    entries = load(args.profile, args.limit)
    if not entries:
        ap.error("profile has no entries")
    write(args.out, entries)
    print(f"wrote {len(entries)} entries to {args.out}")


if __name__ == "__main__":
    main()
