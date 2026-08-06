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
    "- You can use the <think></think> block to plan your next tool call OR to synthesize data and formulate your final response to the user.\n"
    "- ALL explanation and reasoning MUST be placed strictly inside the <think></think> block.\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags.\n"
    "- If you choose to call a tool, output the <tool_call> block immediately after thinking, with no conversational text before it.\n"
    "- The <tool_call> and <function> tags must begin a new line with no indentation.\n"
    "- To call multiple functions, output a separate, completely closed <tool_call></tool_call> block for each function.\n"
    "- If you have all necessary data, provide your final answer directly to the user without any tool call.\n"
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


def apply_thinking_control(content: str, thinking: bool) -> tuple[str, bool]:
    if "<|think_off|>" in content:
        return content.replace("<|think_off|>", "").strip(), False
    if "<|think_on|>" in content:
        return content.replace("<|think_on|>", "").strip(), True
    return content.strip(), thinking


def is_system(role: object) -> bool:
    return role in ("system", "developer")


def system_text(messages: list[dict], tools: list[dict], thinking: bool) -> tuple[str, int, bool]:
    first_system = bool(messages and is_system(messages[0].get("role")))
    skip = 1 if first_system else 0
    extra = render_content(messages[0].get("content")) if first_system else ""
    extra, thinking = apply_thinking_control(extra, thinking)
    if tools:
        tool_lines = "".join(
            "\n" + json.dumps(t, ensure_ascii=False, separators=(",", ":"))
            for t in tools
        )
        content = TOOL_INSTRUCTIONS.format(tools=tool_lines)
        if extra:
            content += "\n\n" + extra
        return f"{IM_START}system\n{content}{IM_END}\n", skip, thinking
    if first_system:
        if extra:
            return f"{IM_START}system\n{extra}{IM_END}\n", skip, thinking
        return "", skip, thinking
    return "", skip, thinking


def last_query_index(messages: list[dict]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") != "user":
            continue
        content = render_content(msg.get("content")).strip()
        if not (content.startswith("<tool_response>") and content.endswith("</tool_response>")):
            return i
    if len(messages) > 50:
        return len(messages) - 1
    return 0


def assistant_parts(msg: dict, content: str) -> tuple[str, str]:
    reasoning = msg.get("reasoning_content")
    if reasoning is None:
        reasoning = msg.get("thinking")
    if reasoning is not None:
        return str(reasoning).strip(), content

    found = []
    for tag in ("</think>", "</thinking>"):
        if content.startswith(tag):
            found.append((0, tag))
    for tag in ("\n</think>", "\n</thinking>", "\n</ think>", "\n</think >"):
        pos = content.find(tag)
        if pos >= 0:
            found.append((pos, tag))
    if not found:
        return "", content
    pos, tag = min(found)
    reasoning = content[:pos]
    opening = "<thinking>" if "thinking" in tag else "<think>"
    if opening in reasoning:
        reasoning = reasoning.rsplit(opening, 1)[-1]
    return reasoning.strip(), content[pos + len(tag):].strip()


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
            out.append(f"\n\n<tool_call>\n<function={name}>\n")
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


def tool_response_is_error(content: str) -> bool:
    lower = content.lower()
    if len(content) >= 500 or "$ " in content or "took " in lower:
        return False
    head = lower[:80]
    return any(mark in head for mark in (
        '"error":', "error:", "err!", "fatal:", "exception:",
        "traceback", "command not found", "invalid syntax", "failed to",
    ))


def tool_error_warning(failures: int) -> str:
    if failures >= 2:
        return (f"\n\n⚠️ SYSTEM WARNING: {failures} consecutive tool errors detected. "
                "Your previous approach is incorrect. You MUST use a fundamentally "
                "different approach or corrected arguments.")
    if failures == 1:
        return ("\n\n⚠️ SYSTEM WARNING: The previous tool call returned an error. "
                "Diagnose the failure and retry with completely corrected arguments.")
    return ""


def render(messages: list[dict], mode: str, tools: list[dict] | None = None,
           add_generation_prompt: bool = True, preserve_thinking: bool = True) -> str:
    tools = tools or []
    thinking = mode != "nothink"
    system, skip, thinking = system_text(messages, tools, thinking)
    out = [system]
    body_messages = messages[skip:]
    last_query = last_query_index(body_messages)
    failures = 0

    for i, msg in enumerate(body_messages):
        role = msg.get("role")
        content = render_content(msg.get("content")).strip()
        if is_system(role):
            content, thinking = apply_thinking_control(content, thinking)
            out.append(f"{IM_START}system\n{content}{IM_END}\n")
        elif role == "user":
            content, thinking = apply_thinking_control(content, thinking)
            failures = 0
            out.append(f"{IM_START}user\n{content}{IM_END}\n")
        elif role == "assistant":
            reasoning, content = assistant_parts(msg, content)
            if reasoning and (preserve_thinking or i > last_query):
                body = f"<think>\n{reasoning}\n</think>\n\n{content}"
            else:
                body = content
            calls = msg.get("tool_calls")
            if isinstance(calls, list):
                body += render_tool_calls(content, calls)
            out.append(f"{IM_START}assistant\n{body}{IM_END}\n")
        elif role == "tool":
            failures = failures + 1 if tool_response_is_error(content) else 0
            if i == 0 or body_messages[i - 1].get("role") != "tool":
                out.append(f"{IM_START}user")
            out.append(f"\n<tool_response>\n{content}{tool_error_warning(failures)}\n</tool_response>")
            if i == len(body_messages) - 1 or body_messages[i + 1].get("role") != "tool":
                out.append(f"{IM_END}\n")
        else:
            out.append(f"{IM_START}user\n[{role}]: {content}{IM_END}\n")

    if add_generation_prompt and (not messages or messages[-1].get("role") != "assistant"):
        out.append(f"{IM_START}assistant\n")
        if not thinking or failures >= 2:
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
