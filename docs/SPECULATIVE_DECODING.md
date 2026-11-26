# Speculative Decoding

[README](../README.md) | [Models](MODELS.md) | [Performance](PERFORMANCE.md#mtp-measurements)

MTP is opt-in. It drafts several tokens and verifies them against the target
model. Plain decoding remains the default; speedups depend on the prompt,
context, and accepted draft length.

## Inkling 2 Mini on the BC-250

Download `inkling-2-mini` and `inkling-2-mini-dspark` with
`download_model.sh`. In `q36`, `q36-server`, or `q36-agent`, pass the target
GGUF with `-m` and the 333 MB draft GGUF with `--mtp FILE`. The three-layer
head shares Inkling's token embedding and output projection. Each block begins
with a target token and can propose up to seven more. The target checks the
proposals and commits their matching prefix.

At the default `--mtp-margin 0`, the head drafts from a shortlist built from
its learned vocabulary bias and recent tokens. The target scores its full
vocabulary during verification. The draft and verifier share attention
scratch, and verification extends the committed convolution history in place.
When the draft cache capacity exceeds 8K, its full-attention layer uses the
most recent 4,096 tokens. Target verification uses the full context.

The head starts with four positions.
After a miss or a shortened block, it verifies an uncertain first proposal
alongside the target anchor. A rejected probe reduces the next block to two
positions. Two-position probes use a second draft mask for context.
A matching pair lets the block grow again. The probe keeps its
lower confidence threshold until a longer prefix is accepted. An unambiguous
prompt suffix can supply a longer tail. If its first proposal fails
verification, Inkling skips the next four suffix matches.

`Q36_DSPARK_PAUSE=0` is the default with the adaptive margin. Values
from 1 to 12 set the pause length after repeated misses, with 12 restoring
the previous behavior. An explicit positive `--mtp-margin` selects the
trained head alone and retains the twelve-token pause unless overridden.
Set `Q36_MTP_LOOKUP_OFF=1` to use the head alone with the adaptive margin.

For greedy generation use `--temp 0`. At positive temperature, opportunistic
sampling draws each block's first token from the target and checks greedy
tails. Use `--no-mtp-opportunistic` for ordinary target sampling. The head
runs on a single BC-250 Vulkan device. Image-conditioned requests use ordinary
target decode.

On one BC-250 with Q8_0 keys and Q4_0 values, six held-out short questions
averaged 49.76 t/s plain and 58.35 t/s with the head alone. Six questions
after 2K or 4K documents averaged 45.76 and 61.30 t/s, respectively. The
slowest question in each set gained 1.07× and 1.13×. Each answer generated
96 greedy tokens and matched plain decode on both runs.

With the previous twelve-token pause, six unfinished C and Python functions
averaged 50.41 t/s plain and 82.55 t/s with the head alone (1.64×).
An 8K C-file continuation spent its first 55 tokens finishing a comment and
gained 1.08× with lookup. At a 4K struct-field list, lookup-assisted decode reached
64.52 t/s against 46.38 t/s plain. The head added about 352 MiB of resident GPU
memory during 2K decode.

`make test-inkling-dspark` checks greedy and sampled-anchor replay against
serial target logits, including rejected proposals and restored checkpoints.
The following command takes three workload files with at least 4,096 tokens each.

```sh
python3 tests/bench_inkling_dspark.py prose.txt source.c reasoning.txt --no-lookup
```

The benchmark measures plain and speculative generation at 128, 2,048 and
4,096 tokens. It reports each result and aggregates by elapsed decode time.
Use `--completion 116 queue.c` to add a run starting after token 116 of
that file. Choosing a position inside a function distinguishes code completion
from continuing a comment or documentation.
Omit `--no-lookup` to include matching continuations from the prompt.
`--repeats 3` repeats each measurement and alternates which arm runs first.
`--min-speedup 1.5` checks the requested ratio in every measured case.
Set `Q36_BENCH_SPEC_TRACE=1` to see committed tokens per call at each frontier.
`Q36_MTP_STATS=1` also counts low-confidence and paused single-token steps.

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

The server uses ordinary decoding for requests that include tools. A tool call
can end or change sampling inside a speculative block, and restoring that
boundary currently requires replaying the context. Other server requests can
still use MTP.

The MoE build `gguf/Qwen3.6-35B-A3B-IQ2XXS-Q2K-Q8.gguf` retains the
Antirez target tensors and embeds a 332.78 MB MTP block quantized from
Unsloth's original BF16 weights. Its draft gate/up experts use IQ2_XXS,
down experts use Q4_K, projections and shared experts use Q8_0, and
routers/norms remain F32. Enable it with:

```sh
./q36 --mtp --temp 0 \
  -p "Write a C function that reverses a linked list."
```

## Dense verification

For greedy Vulkan generation with the dense embedded head:

```sh
./q36 -m gguf/Qwen3.8-27B-UD-IQ3_S.gguf --mtp --temp 0 \
  -p "Write a C function that reverses a linked list."
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
./q36 -m gguf/Qwen3.8-27B-UD-IQ3_S.gguf --mtp --temp 1 \
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
