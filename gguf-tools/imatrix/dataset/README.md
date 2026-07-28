# Q36 Imatrix Calibration Dataset

This directory builds the Qwen3.6/Q36 calibration corpus used before mixed
routed-expert quantization. The tracked JSONL is the source of truth and the
rendered files use the Qwen3.6 chat format.

Tool records preserve tool schemas and calls in Qwen3.6 format:

- top-level `tools` arrays use OpenAI-style function schemas;
- assistant calls use Qwen-compatible `tool_calls`;
- tool results remain `role: "tool"` messages and render as Qwen
  `<tool_response>` blocks.

`prompts.jsonl` is the source-of-truth intermediate. Do not pass it directly to
`llama-imatrix`; render it to plain prompt text with the Qwen3.6 chat template
first.

## 1. Build `prompts.jsonl`

```sh
python3 gguf-tools/imatrix/dataset/build_q36_imatrix_dataset.py
```

This writes:

- `gguf-tools/imatrix/dataset/prompts.jsonl`
- `gguf-tools/imatrix/dataset/manifest.json`

Expected count: `4690` JSONL records, split as `2345` `think` and `2345`
`nothink`, matching Q36 inference modes.

## 2. Render Qwen3.6 Chat Prompts

```sh
python3 gguf-tools/imatrix/dataset/render_q36_prompts.py \
  --in gguf-tools/imatrix/dataset/prompts.jsonl
```

The renderer follows the Qwen3.6 chat template shape: `<|im_start|>` /
`<|im_end|>`, `# Tools` system blocks, `<tool_call>` blocks, and
`<tool_response>` blocks. For records that do not already end with an assistant
message, it adds the Qwen assistant generation prefix. `think` mode starts the
assistant with `<think>\n`; `nothink` mode starts it with an empty
`<think>\n\n</think>\n\n` block.

It writes three rendered corpus files:

- `rendered_prompts.txt`: combined `think` and `nothink` calibration input.
- `rendered_prompts_think.txt`: Qwen `enable_thinking=True` style prompts.
- `rendered_prompts_nothink.txt`: Qwen `enable_thinking=False` style prompts.

For normal imatrix collection use `rendered_prompts.txt`. The split files are
kept for ablations and debugging. If the model metadata chat template changes,
update the renderer first and regenerate all three rendered files.

For a balanced streaming-router profile without processing the full corpus:

```sh
python3 gguf-tools/imatrix/dataset/render_q36_prompts.py \
  --sample-per-category-mode 2 \
  --out /tmp/q36-streaming-prompts.txt
```

This samples each category and thinking mode, then interleaves the groups.

## 3. Run `llama-imatrix`

Use the local Q8 GGUF and keep mmap enabled:

```sh
llama.cpp/build/bin/llama-imatrix \
  -m gguf/Qwen3.6-35B-A3B-Q8_0.gguf \
  -f gguf-tools/imatrix/dataset/rendered_prompts.txt \
  --mmap \
  --parse-special \
  --output-format dat \
  -o gguf/q36-qwen36-imatrix.dat
```

The `dat` output is the format consumed by `gguf-tools/qwen36-quantize`.

## 4. Quantize With The Imatrix

```sh
make -C gguf-tools

gguf-tools/qwen36-quantize \
  --in gguf/Qwen3.6-35B-A3B-Q8_0.gguf \
  --imatrix gguf/q36-qwen36-imatrix.dat \
  --imatrix-strict \
  --audit gguf/q36-mixed-audit.tsv \
  --out gguf/Qwen3.6-35B-A3B-Q36-IQ2XXS-gateup-Q2K-down-Q8rest.gguf
```

The quantizer changes only exact routed expert tensors:

- `blk.N.ffn_gate_exps.weight` -> `IQ2_XXS`
- `blk.N.ffn_up_exps.weight` -> `IQ2_XXS`
- `blk.N.ffn_down_exps.weight` -> `Q2_K`

All shared experts, routers, embeddings, output heads, norms, attention, SSM,
and other non-routed tensors are explicit keep-list entries.
