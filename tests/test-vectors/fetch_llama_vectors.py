#!/usr/bin/env python3
"""Capture Qwen 3.6 MoE logprob vectors through a direct libllama tool."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


MODEL_PATH = "gguf/Qwen3.6-35B-A3B-Q8_0.gguf"
MODEL_NAME = "Qwen3.6-35B-A3B-Q8_0.gguf"
CAPTURE_BIN = ".opencode-tmp/llama_qwen_logprobs_capture"
DEFAULT_OUT = "tests/test-vectors"
TOP_LOGPROBS = 20
MAX_TOKENS = 4
CTX_BY_ID = {
    "short_italian_fact": 16384,
    "short_code_completion": 4096,
    "short_reasoning_plain": 4096,
    "long_memory_archive": 16384,
    "long_code_audit": 16384,
}


def long_memory_prompt() -> str:
    block = (
        "Record {i:03d}: the archive entry says that component alpha keeps a "
        "compressed index, component beta keeps raw observations, and component "
        "gamma reports anomalies only after the checksum phrase appears. "
        "Do not summarize yet; retain the exact final question.\n"
    )
    body = "".join(block.format(i=i) for i in range(72))
    return (
        "You are checking a long technical archive. Read the repeated records "
        "and answer only the final question with one short sentence.\n\n"
        + body
        + "\nFinal question: which component reports anomalies after the checksum phrase appears?"
    )


def long_code_prompt() -> str:
    stanza = (
        "Function f_{i} validates a queue entry, calls normalize_path(), then "
        "appends a compact audit line. The invariant is that strlen() must not "
        "be recomputed when a trusted length returned by snprintf() is already "
        "available. Security note {i}: reject negative sizes before casting.\n"
    )
    body = "".join(stanza.format(i=i) for i in range(68))
    return (
        "Review this generated C-code audit log. After the log, complete the "
        "sentence with the most likely next words.\n\n"
        + body
        + "\nCompletion target: The most important code quality issue is"
    )


PROMPTS = [
    {
        "id": "short_italian_fact",
        "kind": "short",
        "prompt": "Rispondi in italiano con una frase: chi era Ada Lovelace?",
    },
    {
        "id": "short_code_completion",
        "kind": "short",
        "prompt": "Complete the C statement with the next exact token only:\nreturn snprintf(buf, sizeof(buf), \"%d\", value",
    },
    {
        "id": "short_reasoning_plain",
        "kind": "short",
        "prompt": "Answer with only the number: 2048 divided by 128 is",
    },
    {
        "id": "long_memory_archive",
        "kind": "long",
        "prompt": long_memory_prompt(),
    },
    {
        "id": "long_code_audit",
        "kind": "long",
        "prompt": long_code_prompt(),
    },
]


def hex_bytes(values: list[int]) -> str:
    return "".join(f"{int(x):02x}" for x in values)


def fixture_hex(values: list[int]) -> str:
    out = hex_bytes(values)
    return out if out else "-"


def capture_record(spec: dict, capture_bin: Path, model: Path, prompt_path: Path, hf_template: bool, threads: int) -> dict:
    raw_path = prompt_path.with_suffix(".capture.json")
    try:
        cmd = [
            str(capture_bin),
            "--model",
            str(model),
            "--prompt-file",
            str(prompt_path),
            "--out",
            str(raw_path),
            "--ctx",
            str(CTX_BY_ID[spec["id"]]),
            "--n-predict",
            str(MAX_TOKENS),
            "--top-k",
            str(TOP_LOGPROBS),
            "--threads",
            str(threads),
        ]
        if hf_template:
            cmd.append("--hf-template")
        subprocess.run(cmd, check=True)
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    finally:
        raw_path.unlink(missing_ok=True)

    steps = []
    for item in raw.get("steps", []):
        top = []
        for alt in item.get("top_logprobs", []) or []:
            tok = alt.get("token", {})
            top.append(
                {
                    "token": {
                        "id": tok.get("id"),
                        "text": tok.get("text", ""),
                        "bytes": [int(x) for x in tok.get("bytes", [])],
                    },
                    "logit": alt.get("logit"),
                    "logprob": alt.get("logprob"),
                }
            )
        tok = item.get("selected", {})
        steps.append(
            {
                "step": item.get("step"),
                "token": {
                    "id": tok.get("id"),
                    "text": tok.get("text", ""),
                    "bytes": [int(x) for x in tok.get("bytes", [])],
                },
                "top_logprobs": top,
            }
        )

    return {
        "schema": "q36-llamacpp-logprobs-v1",
        "source": "llama.cpp-direct-logits",
        "model": model.name,
        "runtime": "llama.cpp",
        "id": spec["id"],
        "kind": spec["kind"],
        "prompt": spec["prompt"],
        "request": {
            "system": "",
            "temperature": -1,
            "max_tokens": MAX_TOKENS,
            "ctx": CTX_BY_ID[spec["id"]],
            "top_logprobs": TOP_LOGPROBS,
            "threads": threads,
            "think": False,
            "template": "hf-text-only" if hf_template else "q36-fixed-nothink",
        },
        "prompt_tokens": raw.get("prompt_tokens"),
        "ctx": raw.get("ctx"),
        "top_k": raw.get("top_k"),
        "template": raw.get("template", "hf-text-only" if hf_template else "q36-fixed-nothink"),
        "steps": steps,
    }


def write_compact_fixture(root: Path, manifest: dict) -> None:
    lines = [
        "# q36-llama-logprob-vectors-v1",
        "# case <id> <ctx> <steps> <prompt-file>",
        "# step <index> <selected-hex> <top-count>",
        "# top <token-hex> <llamacpp-logprob>",
        "",
    ]
    for prompt in manifest["prompts"]:
        vector_id = prompt["id"]
        record = json.loads((root / prompt["llamacpp_file"]).read_text(encoding="utf-8"))
        steps = record["steps"]
        prompt_file = root / prompt["prompt_file"]
        lines.append(f"case {vector_id} {CTX_BY_ID[vector_id]} {len(steps)} {prompt_file}")
        for i, step in enumerate(steps):
            top = []
            for alt in step.get("top_logprobs", []):
                lp = float(alt.get("logprob", -9999))
                if lp <= -1000:
                    continue
                token_hex = fixture_hex(alt["token"]["bytes"])
                top.append((token_hex, lp))
            lines.append(f"step {i} {fixture_hex(step['token']['bytes'])} {len(top)}")
            for token_hex, lp in top:
                lines.append(f"top {token_hex} {lp:.9g}")
        lines.append("end")
        lines.append("")
    (root / "llama.vec").write_text("\n".join(lines), encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="output directory")
    parser.add_argument("--model", default=MODEL_PATH, help="Qwen GGUF path")
    parser.add_argument("--capture-bin", default=CAPTURE_BIN, help="direct logits capture binary")
    parser.add_argument("--threads", type=int, default=1, help="kept for compatibility")
    parser.add_argument("--only", action="append", help="fetch only the named prompt id")
    parser.add_argument("--reuse-existing", action="store_true",
                        help="reuse existing raw llamacpp json files when present")
    parser.add_argument("--hf-template", action="store_true",
                        help="capture using the official HF text-only chat template")
    args = parser.parse_args()

    if args.hf_template and not args.out:
        raise SystemExit("--hf-template requires --out; the tracked corpus in "
                         f"{DEFAULT_OUT} uses the Q36 chat render")
    root = Path(args.out or DEFAULT_OUT)
    model = Path(args.model)
    capture_bin = Path(args.capture_bin)
    prompt_dir = root / "prompts"
    llamacpp_dir = root / "llamacpp"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    llamacpp_dir.mkdir(parents=True, exist_ok=True)

    if not capture_bin.exists():
        raise SystemExit(f"missing capture binary: {capture_bin}")
    if not model.exists():
        raise SystemExit(f"missing model: {model}")

    wanted = set(args.only or [])
    selected = [spec for spec in PROMPTS if not wanted or spec["id"] in wanted]
    template = "hf-text-only" if args.hf_template else "q36-fixed-nothink"
    manifest = {
        "schema": "q36-test-vector-manifest-v1",
        "source": "llama.cpp-direct-logits",
        "template": template,
        "model": model.name,
        "top_logprobs": TOP_LOGPROBS,
        "max_tokens": MAX_TOKENS,
        "prompts": [],
    }

    for spec in selected:
        prompt_path = prompt_dir / f"{spec['id']}.txt"
        prompt_path.write_text(spec["prompt"], encoding="utf-8")

        out_path = llamacpp_dir / f"{spec['id']}.llamacpp.json"
        if args.reuse_existing and out_path.exists():
            record = json.loads(out_path.read_text(encoding="utf-8"))
        else:
            record = capture_record(spec, capture_bin, model, prompt_path, args.hf_template, args.threads)
            out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if record.get("template") != template:
            raise SystemExit(f"{out_path}: template is {record.get('template')!r}, expected {template!r}")
        manifest["prompts"].append(
            {
                "id": spec["id"],
                "kind": spec["kind"],
                "prompt_file": str(prompt_path.relative_to(root)),
                "llamacpp_file": str(out_path.relative_to(root)),
                "prompt_chars": len(spec["prompt"]),
                "steps": len(record["steps"]),
            }
        )
        print(f"wrote {out_path}")

    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not wanted:
        write_compact_fixture(root, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
