#!/usr/bin/env python3
"""Exercise vector loading, projection, capture lifetime, and the live REPL."""

import argparse
import array
import math
import os
import re
from pathlib import Path
import subprocess
import tempfile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--q36", default="./q36")
    ap.add_argument("--model", default="gguf/Qwen3.8-27B-UD-IQ3_S.gguf")
    profiles = {"qwen3.8-27b": (64, 5120), "qwen3.6-35b-a3b": (40, 2048),
                "inkling-2-mini": (48, 2304)}
    ap.add_argument("--profile", choices=profiles,
                    default="qwen3.8-27b")
    ap.add_argument("--backend", choices=("cpu", "vulkan", "metal"))
    args = ap.parse_args()
    layers, width = profiles[args.profile]
    inkling = args.profile == "inkling-2-mini"
    base = [str(Path(args.q36).resolve()), "-m", str(Path(args.model).resolve()),
            "--ctx", "256", "--nothink", "--temp", "0"]
    if args.backend:
        base.append(f"--{args.backend}")
    env = os.environ.copy()
    env.pop("Q36_GRAPH_DUMP_PREFIX", None)
    env.pop("Q36_GRAPH_DUMP_NAME", None)
    env.pop("Q36_GRAPH_DUMP_LAYER", None)
    env.pop("Q36_GRAPH_DUMP_POS", None)

    def run(options, *, capture=None, commands=None, success=True, filters=None):
        child_env = env.copy()
        if capture:
            child_env["Q36_GRAPH_DUMP_PREFIX"] = str(capture[0])
            child_env["Q36_GRAPH_DUMP_NAME"] = capture[1]
        if filters:
            child_env.update(filters)
        result = subprocess.run(base + options, input=commands, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=child_env, timeout=120)
        output = result.stdout + result.stderr
        assert (result.returncode == 0) == success, output
        return output

    with tempfile.TemporaryDirectory(prefix="q36-steering-test-") as td:
        root = Path(td)
        bank = root / "direction.f32"
        vectors = array.array("f", [0.0]) * (layers * width)

        def write_bank(layer):
            for i in range(len(vectors)):
                vectors[i] = 0.0
            vectors[layer * width] = 1.0
            bank.write_bytes(vectors.tobytes())

        def read_dump(prefix, component):
            rows = []
            for layer in range(layers):
                path = Path(f"{prefix}_{component}-{layer}_pos0.bin")
                assert path.stat().st_size == width * 4, path
                row = array.array("f")
                row.frombytes(path.read_bytes())
                assert all(math.isfinite(x) for x in row), path
                rows.append(row)
            return rows

        # A fresh process must detect the model shape before reading the bank,
        # even when the initial scale is zero and /steer will enable it later.
        write_bank(0)
        steering = ["--dir-steering-file", str(bank), "--dir-steering-ffn", "0"]
        output = run(steering, commands="/steer\n/steer -1\n/steer\n/steer nan\n/ctx 384\n/steer\n/steer 0\n/exit\n")
        assert "Steering FFN: 0." in output, output
        assert output.count("Steering FFN: -1.") >= 3, output
        assert "/steer must be between -100 and 100" in output, output
        assert "needs a vector loaded" not in output, output

        bank.write_bytes(b"bad")
        output = run(steering + ["--inspect"], success=False)
        assert f"{layers} x {width}" in output, output
        write_bank(0)
        vectors[0] = math.nan
        bank.write_bytes(vectors.tobytes())
        output = run(steering + ["--inspect"], success=False)
        assert "failed to read directional steering vectors" in output, output

        chunk = "8" if inkling else "1"
        prompt = ["-p", "Explain DNS.", "--prefill-chunk", chunk, "-n", "1"]
        targets = [("ffn_out", "--dir-steering-ffn", 18 if inkling else 0),
                   ("attn_out", "--dir-steering-attn", 0),
                   ("attn_out", "--dir-steering-attn", 47 if inkling else 3)]
        for component, option, layer in targets:
            write_bank(layer)
            prefix = root / f"baseline-{component}-{layer}"
            run(prompt, capture=(prefix, component))
            baseline = read_dump(prefix, component)
            assert abs(baseline[layer][0]) > 1e-6
            for scale in (1.0, -0.5, 0.0):
                prefix = root / f"scaled-{component}-{layer}-{scale}"
                options = prompt + ["--dir-steering-file", str(bank), option, str(scale)]
                # More decode tokens must leave the final prompt capture intact.
                run(options + ["-n", "3"], capture=(prefix, component))
                actual = read_dump(prefix, component)
                expected = baseline[layer][0] * (1.0 - scale)
                assert abs(actual[layer][0] - expected) < 1e-4, (component, layer, scale)
                assert max(abs(a - b) for a, b in
                           zip(actual[layer][1:], baseline[layer][1:])) < 1e-4
                if scale == 0:
                    assert all(a == b for a, b in zip(actual, baseline))
                if inkling and layer:
                    assert actual[:layer] == baseline[:layer]

        prefix = root / "all-positions"
        layer = 1 if inkling else 0
        layer_filter = str(layer)
        output = run(prompt + ["-n", "3"], capture=(prefix, "all"), filters={
            "Q36_GRAPH_DUMP_LAYER": layer_filter, "Q36_GRAPH_DUMP_POS": "all"})
        match = re.search(r"processing (\d+) input tokens", output)
        assert match, output
        prompt_tokens = int(match[1])
        names = ["residual_pre", "attn_norm", "attn_out", "residual_attn",
                 "ffn_norm", "ffn_out", "residual_post"]
        files = list(root.glob("all-positions_*.bin"))
        assert len(files) == (prompt_tokens + 3) * len(names), len(files)

        def row_at(prefix, name, pos, layer=layer):
            values = array.array("f")
            values.frombytes(Path(f"{prefix}_{name}-{layer}_pos{pos}.bin").read_bytes())
            assert len(values) == width and all(math.isfinite(x) for x in values)
            return values

        for pos in (0, prompt_tokens - 1, prompt_tokens + 2):
            values = {name: row_at(prefix, name, pos) for name in names}
            for left, right, total in [("residual_pre", "attn_out", "residual_attn"),
                                        ("residual_attn", "ffn_out", "residual_post")]:
                assert max(abs(a + b - c) for a, b, c in
                           zip(values[left], values[right], values[total])) < 1e-4
        baseline = root / f"baseline-ffn_out-{targets[0][2]}"
        assert row_at(prefix, "ffn_out", prompt_tokens - 1) == row_at(baseline, "ffn_out", 0)
        selected = root / "selected-position"
        run(prompt, capture=(selected, "all"), filters={
            "Q36_GRAPH_DUMP_LAYER": layer_filter, "Q36_GRAPH_DUMP_POS": "1"})
        assert len(list(root.glob("selected-position_*.bin"))) == len(names)
        for name in names:
            assert row_at(selected, name, 1) == row_at(prefix, name, 1)
        if inkling:
            selected = root / "final-slot"
            run(prompt, capture=(selected, "all"), filters={
                "Q36_GRAPH_DUMP_LAYER": str(layers - 1), "Q36_GRAPH_DUMP_POS": "1"})
            row_at(selected, "output_norm", 1, layers - 1)

            selected = root / "first-slot"
            run(prompt, capture=(selected, "all"), filters={"Q36_GRAPH_DUMP_LAYER": "0"})
            assert row_at(selected, "residual_attn", 0, 0) == row_at(selected, "attn_out", 0, 0)

    print(f"PASS steering ({args.profile}): loading, zero-scale enable, invalid banks, "
          "FFN and attention projection, final prompt capture, layer/position filters, "
          "residual identities, /steer and /ctx")


if __name__ == "__main__":
    main()
