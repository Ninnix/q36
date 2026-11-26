"""Compare Inkling decode on the same prose, code and reasoning frontiers."""
import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
for name in ("prose", "code", "reasoning"):
    parser.add_argument(name, type=Path)
parser.add_argument("--completion", nargs=2, action="append", default=[],
                    metavar=("TOKENS", "FILE"), help="Also generate at this exact source position.")
parser.add_argument("--model", type=Path, default=root / "gguf/Inkling-2-Mini-UD-IQ3_XXS.gguf")
parser.add_argument("--head", type=Path, default=root / "gguf/Inkling-2-Mini-DSpark-333M.gguf")
parser.add_argument("--no-lookup", action="store_true", help="Measure the trained head alone.")
parser.add_argument("--min-speedup", type=float, help="Require this ratio in every measured case.")
parser.add_argument("--repeats", type=int, default=1, help="Runs per arm, alternating which arm runs first.")
args = parser.parse_args()
env = os.environ.copy()
if args.no_lookup:
    env["Q36_MTP_LOOKUP_OFF"] = "1"
if "Q36_MTP_SPEC_DISABLE" in env:
    parser.error("Q36_MTP_SPEC_DISABLE disables the speculative arm")
if args.min_speedup is not None and not (math.isfinite(args.min_speedup) and args.min_speedup > 0):
    parser.error("--min-speedup must be positive")
if args.repeats < 1:
    parser.error("--repeats must be positive")
print("Prompt lookup", "off" if "Q36_MTP_LOOKUP_OFF" in env else "on", file=sys.stderr)

workloads = [(name, getattr(args, name).resolve(), (128, 2048, 4096))
             for name in ("prose", "code", "reasoning")]
for position, filename in args.completion:
    if not position.isascii() or not position.isdecimal() or int(position) < 1:
        parser.error("--completion requires a positive token position")
    path = Path(filename).resolve()
    workloads.append((path.name, path, (int(position),)))

def measure(prompt, frontiers, sampled, spec, output):
    tokens = 64 if sampled else 96
    cmd = [str(root / "q36-bench"), "--vulkan", "-m", args.model.resolve(), "-t", "16",
           "-ctk", "q8_0", "-ctv", "q4_0", "--warmup", "--ctx-start", str(frontiers[0]),
           "--ctx-max", str(frontiers[-1]), "--step-mul", "16", "--prompt-file", str(prompt),
           "--gen-tokens", str(tokens), "--csv", str(output)]
    if sampled:
        cmd += ["--temp", "1", "--seed", "42"]
    if spec:
        cmd += ["--mtp", args.head.resolve(), "--mtp-draft", "8", "--mtp-margin", "0"]
        if sampled:
            cmd += ["--mtp-opportunistic"]
    subprocess.run(cmd, check=True, cwd=root, env=env)
    with output.open() as f:
        rows = {}
        for r in csv.DictReader(f):
            ctx = int(r["ctx_tokens"])
            if ctx in rows or int(r["gen_tokens"]) != tokens:
                raise ValueError(f"Unexpected decode measurement for {prompt} at {ctx} tokens")
            rows[ctx] = float(r["gen_tps"])
    if set(rows) != set(frontiers) or any(not math.isfinite(v) or v <= 0 for v in rows.values()):
        raise ValueError(f"Incomplete decode measurements for {prompt}")
    return rows

worst = float("inf")
writer = csv.writer(sys.stdout, lineterminator="\n")
writer.writerow(("sampling", "domain", "context", "plain_tps", "dspark_tps", "speedup"))
sys.stdout.flush()
with tempfile.TemporaryDirectory() as tmp:
    output = Path(tmp) / "run.csv"
    for sampled in (False, True):
        mode = "sampled" if sampled else "greedy"
        seconds = [0.0, 0.0]
        tokens = 64 if sampled else 96
        total = tokens * args.repeats
        for name, prompt, frontiers in workloads:
            elapsed = [{}, {}]
            for repeat in range(args.repeats):
                for speculative in ((True, False) if repeat % 2 else (False, True)):
                    times = elapsed[speculative]
                    for ctx, tps in measure(prompt, frontiers, sampled, speculative, output).items():
                        times[ctx] = times.get(ctx, 0.0) + tokens / tps
            for ctx in sorted(elapsed[0]):
                plain, spec = (total / times[ctx] for times in elapsed)
                ratio = spec / plain
                worst = min(worst, ratio)
                seconds[0] += elapsed[0][ctx]
                seconds[1] += elapsed[1][ctx]
                writer.writerow((mode, name, ctx, f"{plain:.2f}", f"{spec:.2f}", f"{ratio:.3f}"))
            sys.stdout.flush()
        measured = total * sum(len(frontiers) for _, _, frontiers in workloads)
        writer.writerow((mode, "all", "all", f"{measured/seconds[0]:.2f}",
                         f"{measured/seconds[1]:.2f}", f"{seconds[0]/seconds[1]:.3f}"))
        sys.stdout.flush()

if args.min_speedup and worst < args.min_speedup:
    raise SystemExit(f"Lowest speedup {worst:.3f}x fell below {args.min_speedup:g}x")
