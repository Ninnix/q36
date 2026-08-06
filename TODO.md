# DS4 Follow-up

- [x] Make owned JSON field replacement atomic and reject non-finite integers.
- [x] Bound agent cache text and title lengths by bytes remaining in the file.
- [x] Bound GGUF metadata/tensor counts and all quantizer shape/size arithmetic.
- [x] Reject oversized token-only session payloads before allocation.
- [x] Route unterminated thinking output to reasoning content.
- [x] Recover complete tool calls inside unclosed reasoning without injecting tokens.
- [x] Keep a repeated reasoning pass out of streamed answer content.
- [x] Add `--mixed-prefill-quantum` to the server interface.
- [x] Retain successfully loaded disk KV checkpoints and let normal eviction manage them.
- [x] Scope reference fixtures by model checkpoint.
- [x] Make the release benchmark discard a warm-up and gate the median of three runs.
- [x] Add focused regressions and run the broader CPU/Vulkan/server/agent gates.

Validation note: the Vulkan `short_code_completion` reference case remains at
avg NLL 4.670, above its existing gate. The unmodified test harness produces
the same value against the same runtime and model, so this is not a regression
from these changes.
