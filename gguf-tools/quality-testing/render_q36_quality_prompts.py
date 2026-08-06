#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def qwen_prompt(text, mode):
    assistant_prefix = {
        "think": "<think>\n",
        "nothink": "<think>\n\n</think>\n\n",
        "none": "",
    }[mode]
    return (
        "<|im_start|>user\n"
        + text
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
        + assistant_prefix
    )


def load_prompts(path, limit):
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            obj = json.loads(line)
            case_id = obj.get("id")
            prompt = obj.get("prompt")
            if not case_id or not prompt:
                raise SystemExit(f"bad prompt row: {line[:120]}")
            rows.append((case_id, prompt))
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise SystemExit(f"no prompts in {path}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(ROOT / "prompts.jsonl"))
    ap.add_argument("--out-text", default=str(ROOT / "rendered_prompts.txt"))
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "prompts"))
    ap.add_argument("--manifest", default=str(ROOT / "data" / "manifest.tsv"))
    ap.add_argument("--mode", choices=("think", "nothink", "none"), default="nothink")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = load_prompts(Path(args.prompts), args.limit)
    out_text = Path(args.out_text)
    out_dir = Path(args.out_dir)
    manifest = Path(args.manifest)
    out_text.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    with out_text.open("w", encoding="utf-8") as all_fp, \
            manifest.open("w", encoding="utf-8") as man_fp:
        man_fp.write("# id\tprompt_file\n")
        for case_id, prompt in rows:
            rendered = qwen_prompt(prompt, args.mode)
            prompt_path = out_dir / f"{case_id}.txt"
            prompt_path.write_text(rendered, encoding="utf-8")
            all_fp.write(rendered)
            all_fp.write("\n")
            man_fp.write(f"{case_id}\t{prompt_path}\n")

    print(f"rendered {len(rows)} prompts to {out_text}")
    print(f"wrote manifest to {manifest}")


if __name__ == "__main__":
    main()
