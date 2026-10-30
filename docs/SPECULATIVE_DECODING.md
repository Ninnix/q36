# Speculative Decoding

[README](../README.md) | [Models](MODELS.md) | [Performance](PERFORMANCE.md#mtp-measurements)

MTP is opt-in. It drafts several tokens and verifies them against the target
model. Plain decoding remains the default; speedups depend on the prompt,
context, and accepted draft length.

## Enable MTP

The Qwen3.6 `q2` download includes its MTP support in the main GGUF;
enable it explicitly with `--mtp`.
Embedded heads are supported in 41-block MoE GGUFs and in both supported
65-block Qwen3.8-27B GGUFs. Use `--mtp` without a filename to select the
embedded head, or `--mtp FILE` for a separate support GGUF. In `q36`,
`q36-server`, and `q36-agent`, this enables the tuned BC-250 configuration:
up to eight tokens, adaptive draft confidence (`--mtp-margin 0`), and
opportunistic sampling. Explicit depth, margin, and sampling options override
these defaults regardless of argument order. The embedding and output weights
remain shared. With `--mtp` omitted, the runtime excludes the extra block from
the resident GPU weight cache.

The MoE build `gguf/Qwen3.6-35B-A3B-IQ2XXS-Q2K-Q8.gguf` retains the
Antirez target tensors and embeds a 332.78 MB MTP block quantized from
Unsloth's original BF16 weights. Its draft gate/up experts use IQ2_XXS,
down experts use Q4_K, projections and shared experts use Q8_0, and
routers/norms remain F32. Enable it with:

```sh
./q36 -m gguf/Qwen3.6-35B-A3B-IQ2XXS-Q2K-Q8.gguf --mtp --temp 0 --nothink \
  -p "Write a C function that reverses a linked list."
```

## Dense verification

For greedy Vulkan generation with the dense embedded head:

```sh
model=gguf/Qwen3.8-27B-UD-IQ3_S.gguf
./q36 -m "$model" --mtp \
  --temp 0 --nothink -p "Write a C function that reverses a linked list."
```

The verifier uses up to eight tokens per pass. Draft rejection restores the
accepted recurrent state without replaying target matrices. The MTP cache
is populated during prefill and corrected from accepted target hidden states.
Verification shares quantized weights across rows and keeps token selection
on the GPU. The draft output head uses a small mask of recent target
alternatives; target verification always scores the full vocabulary.

Resident Vulkan chooses recursive MTP lengths between two and four rows from
the current draft's confidence. With `--mtp-draft 8`,
a matching 12–16-token suffix can supply a longer copied tail after MTP
confirms its first proposal. Positive `--mtp-margin` values use MTP alone.
The regression tests compare committed tokens and every target logit with
ordinary decode, including quantized KV, checkpoint restore and verifier
widths up to eight. MTP also matches a separate ordinary engine with MTP disabled.
The IQ2_XXS/Q2_K MoE target also verifies rows together, retaining ordinary
decode routing, shared-expert and recurrent arithmetic. Other MoE tensor
mixes and SSD streaming retain scalar verification.

## Sampling

Opportunistic sampling is the default with `--mtp` at a positive temperature:

```sh
./q36 -m "$model" --mtp --temp 1 --seed 42 --nothink \
  -p "Write a C function that reverses a linked list."
```

Each block samples its first token with the requested temperature, top-k,
top-p and min-p, then accepts the draft tail while it matches the target's
greedy continuation. This deliberately changes the sampling distribution.
The next block samples again from the last committed target state. Target
logits, KV updates and rejection rollback use the same verifier as greedy MTP.
This also works in `q36-agent` and in `q36-server` for non-batched text
requests with zero presence/frequency penalties; other server requests keep
ordinary sampling. Use `--no-mtp-opportunistic` to sample every token at
positive temperature while retaining exact greedy MTP at temperature zero.
`--mtp-draft 1` disables speculative blocks.

With `--mtp-margin 0`, dense Vulkan drafting uses the MTP head's
confidence to choose its length. It verifies at least one proposal and
extends up to four rows while the top-two logit gap is at least 1.0.
Confirmed history tails can still extend to eight. This avoids wider
verification of weak prefixes and removes wall-clock timing from
block-length decisions. Target acceptance still checks every proposed token.

## Backend limits

The row-batched verifier described above is the resident Vulkan path.
SSD streaming and other MoE tensor mixes use scalar verification. The server
disables MTP with `--batched-session`; choose speculation for one session or
batched serving for several.

See [recorded MTP measurements](PERFORMANCE.md#mtp-measurements) for the
workloads, sampling conditions, historical comparisons, and validation.
