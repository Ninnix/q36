#!/usr/bin/env python3
"""Render Q36 prompts.jsonl into Qwen3.6 chat-template text for llama-imatrix."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

TOOL_INSTRUCTIONS = (
    "# Tools\n\n"
    "You have access to the following functions:\n\n"
    "<tools>{tools}\n</tools>"
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_1>\n"
    "value_1\n"
    "</parameter>\n"
    "<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\n"
    "that can span\n"
    "multiple lines\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n\n"
    "<IMPORTANT>\n"
    "Reminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>"
)


def render_content(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for item in content:
            if not isinstance(item, dict):
                raise ValueError("unexpected content item")
            if "text" in item:
                out.append(str(item["text"]))
            elif "image" in item or "image_url" in item or item.get("type") == "image":
                out.append("<|vision_start|><|image_pad|><|vision_end|>")
            elif "video" in item or item.get("type") == "video":
                out.append("<|vision_start|><|video_pad|><|vision_end|>")
            else:
                raise ValueError("unexpected content item")
        return "".join(out)
    raise ValueError("unexpected content type")


def system_text(messages: list[dict], tools: list[dict]) -> str:
    first_system = messages and messages[0].get("role") == "system"
    if tools:
        tool_lines = "".join(
            "\n" + json.dumps(t, ensure_ascii=False, separators=(",", ":"))
            for t in tools
        )
        content = TOOL_INSTRUCTIONS.format(tools=tool_lines)
        if first_system:
            extra = render_content(messages[0].get("content")).strip()
            if extra:
                content += "\n\n" + extra
        return f"{IM_START}system\n{content}{IM_END}\n"
    if first_system:
        content = render_content(messages[0].get("content")).strip()
        return f"{IM_START}system\n{content}{IM_END}\n"
    return ""


def last_query_index(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") != "user":
            continue
        content = render_content(msg.get("content")).strip()
        if not (content.startswith("<tool_response>") and content.endswith("</tool_response>")):
            return i
    raise ValueError("no user query found")


def render_tool_calls(content: str, calls: list[dict]) -> str:
    out = []
    for i, call in enumerate(calls):
        fn = call.get("function", call)
        name = fn.get("name")
        if not name:
            continue
        if i == 0 and content.strip():
            out.append(f"\n\n<tool_call>\n<function={name}>\n")
        elif i == 0:
            out.append(f"<tool_call>\n<function={name}>\n")
        else:
            out.append(f"\n<tool_call>\n<function={name}>\n")
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        for key, value in args.items():
            out.append(f"<parameter={key}>\n")
            if isinstance(value, str):
                out.append(value)
            else:
                out.append(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
            out.append("\n</parameter>\n")
        out.append("</function>\n</tool_call>")
    return "".join(out)


def render(messages: list[dict], mode: str, tools: list[dict] | None = None,
           add_generation_prompt: bool = True, preserve_thinking: bool = False) -> str:
    tools = tools or []
    out = [system_text(messages, tools)]
    last_query = last_query_index(messages)

    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = render_content(msg.get("content")).strip()
        if role == "system":
            if i != 0:
                raise ValueError("system message must be first")
            continue
        if role == "user":
            out.append(f"{IM_START}user\n{content}{IM_END}\n")
        elif role == "assistant":
            reasoning = str(msg.get("reasoning_content") or "").strip()
            if preserve_thinking or i > last_query:
                body = f"<think>\n{reasoning}\n</think>\n\n{content}"
            else:
                body = content
            calls = msg.get("tool_calls")
            if isinstance(calls, list):
                body += render_tool_calls(content, calls)
            out.append(f"{IM_START}assistant\n{body}{IM_END}\n")
        elif role == "tool":
            if i > 0 and messages[i - 1].get("role") != "tool":
                out.append(f"{IM_START}user")
            out.append(f"\n<tool_response>\n{content}\n</tool_response>")
            if i == len(messages) - 1 or messages[i + 1].get("role") != "tool":
                out.append(f"{IM_END}\n")
        else:
            raise ValueError(f"unexpected message role: {role}")

    if add_generation_prompt and (not messages or messages[-1].get("role") != "assistant"):
        out.append(f"{IM_START}assistant\n")
        if mode == "nothink":
            out.append("<think>\n\n</think>\n\n")
        else:
            out.append("<think>\n")
    return "".join(out)


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def sample_records(records: list[dict], per_group: int) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        groups[(record.get("category"), record.get("mode"))].append(record)

    out = []
    for i in range(per_group):
        for key in sorted(groups):
            if i < len(groups[key]):
                out.append(groups[key][i])
    return out


def write_mode(path: Path, records: list[dict], mode: str) -> None:
    records = [r for r in records if r.get("mode") == mode]
    with path.open("w", encoding="utf-8") as f:
        for i, obj in enumerate(records):
            if i:
                f.write("\n\n")
            f.write(f"===== Q36_IMATRIX_PROMPT {obj['id']} {obj['category']} {mode} {obj['source']} =====\n")
            f.write(render(obj["messages"], mode, tools=obj.get("tools")))


def write_all(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        first = True
        for obj in records:
            mode = obj.get("mode", "nothink")
            if not first:
                f.write("\n\n")
            first = False
            f.write(f"===== Q36_IMATRIX_PROMPT {obj['id']} {obj['category']} {mode} {obj['source']} =====\n")
            f.write(render(obj["messages"], mode, tools=obj.get("tools")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=Path(__file__).resolve().parent / "prompts.jsonl")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--out", type=Path, help="write only the combined file to this path")
    ap.add_argument("--sample-per-category-mode", type=int, default=0,
                    help="select N records from every category/mode pair")
    args = ap.parse_args()

    records = load_records(args.inp)
    if args.sample_per_category_mode < 0:
        ap.error("--sample-per-category-mode must be non-negative")
    if args.sample_per_category_mode:
        records = sample_records(records, args.sample_per_category_mode)
    if args.out:
        write_all(args.out, records)
        print(f"rendered {len(records)} prompts to {args.out}")
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined = args.out_dir / "rendered_prompts.txt"
    think = args.out_dir / "rendered_prompts_think.txt"
    nothink = args.out_dir / "rendered_prompts_nothink.txt"
    write_all(combined, records)
    write_mode(think, records, "think")
    write_mode(nothink, records, "nothink")
    n_think = sum(1 for r in records if r.get("mode") == "think")
    n_nothink = sum(1 for r in records if r.get("mode") == "nothink")
    print(f"rendered {len(records)} prompts to {combined}")
    print(f"rendered {n_think} prompts to {think}")
    print(f"rendered {n_nothink} prompts to {nothink}")


if __name__ == "__main__":
    main()
