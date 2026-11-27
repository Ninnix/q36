#!/usr/bin/env python3
"""Build a Q36 directional-steering vector from paired prompt sets.

The extractor asks q36 to dump one hidden activation row per executed transformer
layer, averages the target and control rows, and writes a flat f32 file.
At runtime q36 applies:

    y = y - scale * direction[layer] * dot(direction[layer], y)

Positive scale suppresses the target direction.  Negative scale amplifies it.
"""

from __future__ import annotations

import argparse
import array
from contextlib import contextmanager
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path


MODEL_PROFILES = {
    "qwen3.8-27b": (64, 5120),
    "qwen3.6-27b": (64, 5120),
    "qwen3.6-35b-a3b": (40, 2048),
    "inkling-2-mini": (48, 2304),
}


def read_prompt_file(path: Path) -> list[str]:
    """Read one prompt per non-empty line, ignoring shell-style comments."""
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)
    if not prompts:
        raise SystemExit(f"{path}: no prompts found")
    return prompts


def normalize(v: list[float]) -> list[float]:
    n2 = sum(x * x for x in v)
    if n2 <= 0.0:
        return v
    inv = 1.0 / math.sqrt(n2)
    return [x * inv for x in v]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@contextmanager
def capture_process(q36, model, system, think, ctx, component, work):
    env = os.environ.copy()
    env["Q36_GRAPH_DUMP_PREFIX"] = str(work / "dump")
    env["Q36_GRAPH_DUMP_NAME"] = component
    env["Q36_GRAPH_DUMP_LAYER"] = "all"
    env["Q36_GRAPH_DUMP_POS"] = "last"
    cmd = [str(q36), "-m", str(model), "--ctx", str(ctx),
           "--system", system, "--temp", "0", "-n", "1",
           "--think" if think else "--nothink"]
    process = subprocess.Popen(cmd, cwd=q36.parent, env=env, text=True,
                               stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
    try:
        yield process
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stderr.close()


def run_capture(process, prompt, component, n_layer, n_embd, work):
    """Clear the native REPL for each prompt while keeping model weights loaded."""
    prompt_path = work / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    process.stdin.write(f'/clear\n/read "{prompt_path}"\n')
    process.stdin.flush()
    log = []
    while True:
        line = process.stderr.readline()
        log.append(line)
        if not line or "prompt processing failed:" in line:
            raise RuntimeError("q36 activation capture failed:\n" + "".join(log))
        if "q36: prefill:" in line and "generation:" in line:
            break

    rows: list[list[float]] = []
    for layer in range(n_layer):
        path = work / f"dump_{component}-{layer}_pos0.bin"
        data = array.array("f")
        with path.open("rb") as f:
            data.fromfile(f, path.stat().st_size // 4)
        if len(data) != n_embd or not all(math.isfinite(x) for x in data):
            raise RuntimeError(f"bad dump shape for {path}: {len(data)} floats")
        rows.append(list(data[-n_embd:]))
    return rows


def add_rows(total: list[list[float]], rows: list[list[float]], n_layer: int) -> None:
    for layer in range(n_layer):
        dst = total[layer]
        src = rows[layer]
        for i, value in enumerate(src):
            dst[i] += value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q36", default="./q36", help="path to the q36 CLI")
    ap.add_argument("--model", default="gguf/Qwen3.8-27B-UD-IQ3_S.gguf", help="GGUF model path")
    ap.add_argument("--profile", choices=MODEL_PROFILES,
                    default="qwen3.8-27b",
                    help="model shape; the CLI renders the native model chat template")
    ap.add_argument("--good-file", required=True,
                    help="desired/target prompts, one per line")
    ap.add_argument("--bad-file", required=True,
                    help="contrast/control prompts, one per line")
    ap.add_argument("--out", default="dir-steering/out/direction.json",
                    help="metadata JSON path; .f32 is written next to it")
    ap.add_argument("--source", default="",
                    help="optional corpus source or provenance note")
    ap.add_argument("--limit", type=int, default=0,
                    help="capture only the first N pairs; zero uses all pairs")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--component", default="ffn_out",
                    choices=("ffn_out", "attn_out", "residual_pre", "residual_attn",
                             "residual_post", "attn_norm", "ffn_norm"),
                    help="hidden-width activation stream used to estimate the direction")
    ap.add_argument("--think", action="store_true",
                    help="capture after <think>; default captures direct answers")
    ap.add_argument("--pair-normalize", action="store_true",
                    help="average normalized per-pair differences")
    ap.add_argument("--no-orthogonalize", action="store_true",
                    help="do not remove the component parallel to the control mean")
    args = ap.parse_args()

    q36 = Path(args.q36).resolve()
    model_arg = Path(args.model)
    model = model_arg.resolve()
    n_layer, n_embd = MODEL_PROFILES[args.profile]
    good_prompts = read_prompt_file(Path(args.good_file))
    bad_prompts = read_prompt_file(Path(args.bad_file))
    if args.limit < 0:
        ap.error("--limit must be nonnegative")
    if len(good_prompts) != len(bad_prompts):
        ap.error("target and control files must have the same number of prompts")
    n = len(good_prompts)
    if args.limit:
        n = min(n, args.limit)
    good_prompts = good_prompts[:n]
    bad_prompts = bad_prompts[:n]

    good_sum = [[0.0] * n_embd for _ in range(n_layer)]
    bad_sum = [[0.0] * n_embd for _ in range(n_layer)]
    pair_sum = [[0.0] * n_embd for _ in range(n_layer)]

    with tempfile.TemporaryDirectory(prefix="q36-dir-steer-") as td:
        root = Path(td)
        with capture_process(q36, model, args.system, args.think,
                             args.ctx, args.component, root) as process:
            for i, (good, bad) in enumerate(zip(good_prompts, bad_prompts), 1):
                print(f"pair {i}/{n}", flush=True)
                good_rows = run_capture(process, good, args.component,
                                        n_layer, n_embd, root)
                bad_rows = run_capture(process, bad, args.component,
                                       n_layer, n_embd, root)
                add_rows(good_sum, good_rows, n_layer)
                add_rows(bad_sum, bad_rows, n_layer)
                if args.pair_normalize:
                    for layer in range(n_layer):
                        diff = normalize([
                            good_rows[layer][j] - bad_rows[layer][j]
                            for j in range(n_embd)
                        ])
                        for j, value in enumerate(diff):
                            pair_sum[layer][j] += value

    layers = []
    for layer in range(n_layer):
        good_mean = [x / n for x in good_sum[layer]]
        bad_mean = [x / n for x in bad_sum[layer]]
        if args.pair_normalize:
            direction = normalize([x / n for x in pair_sum[layer]])
        else:
            direction = normalize([
                good_mean[i] - bad_mean[i]
                for i in range(n_embd)
            ])
        if not args.no_orthogonalize:
            base = normalize(bad_mean)
            projection = dot(direction, base)
            direction = normalize([
                direction[i] - projection * base[i]
                for i in range(n_embd)
            ])
        layers.append(direction)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "q36-directional-steering-v1",
        "shape": [n_layer, n_embd],
        "profile": args.profile,
        "pairs": n,
        "component": args.component,
        "thinking": bool(args.think),
        "pair_normalize": bool(args.pair_normalize),
        "orthogonalize_control_mean": not args.no_orthogonalize,
        "good_file": str(Path(args.good_file)),
        "bad_file": str(Path(args.bad_file)),
        "good_sha256": hashlib.sha256(Path(args.good_file).read_bytes()).hexdigest(),
        "bad_sha256": hashlib.sha256(Path(args.bad_file).read_bytes()).hexdigest(),
        "model": str(model_arg),
        "source": args.source,
        "note": "runtime positive scale suppresses this direction; negative scale amplifies it",
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    flat = array.array("f")
    for direction in layers:
        flat.extend(direction)
    f32_out = out.with_suffix(".f32")
    with f32_out.open("wb") as f:
        flat.tofile(f)
    print(f"wrote {out}")
    print(f"wrote {f32_out}")


if __name__ == "__main__":
    main()
