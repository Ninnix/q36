# Native Coding Agent

[README](../README.md#native-agent) | [Server clients](CLIENTS.md)

`q36-agent` runs inference directly, without a separate HTTP server. It
keeps the transcript and live model state together, shows prefill progress,
and uses Qwen's native tagged tool format. Built-in tools cover files,
search, shell commands, processes, and the web.

Start the agent in the current directory, another project, or one-shot mode:

```sh
./q36-agent
./q36-agent --chdir /path/to/project
./q36-agent -m gguf/Qwen3.8-27B-UD-IQ3_S.gguf --mtp --chdir /path/to/project
./q36-agent --non-interactive -p "Inspect the tests and fix the failure."
```

## Generation and thinking

`--mtp` uses the same Vulkan verifier as the CLI and server. The agent streams
accepted tokens individually and decodes tags, thinking closure, and automatic
edit-old shortening one token at a time. Tool calls therefore finish before
the next assistant round, and the transcript stays aligned with the live KV.

Adaptive thinking closure starts after 50000 thinking tokens by default. For
budgets of at least 8000, its allowed `</think>` rank rises to the top 64 over
the next `min(N/2, 8192)` tokens, where `N` is `--thinking-budget`. Smaller
budgets keep the previous ranking schedule. The rank can rise further if
thinking continues. This is a soft budget; `--tokens` is the hard output limit.

With Qwen3.8-27B, `--thinking-budget` also selects the starting effort:
up to 8000 tokens is `low`, up to 16000 is `medium`, up to 24000 is `high`,
and larger budgets use `xhigh`. The default is `xhigh` with a 50000-token
closure target. `high` uses Qwen3.8's native `xhigh` instruction. Explicit
`--think-low`, `--think-medium`, `--think`, or `--think-xhigh` overrides the
starting mode without changing the budget. At an empty idle agent prompt,
Tab cycles `low → medium → high → xhigh` for Qwen3.8 and `off ↔ on` for
Qwen3.6. Tab leaves the closure budget unchanged; the footer shows both.

Agent user and system messages accept `<|think_on|>` and `<|think_off|>`.
The marker is removed before rendering and remains in effect for later turns.
Historical thinking stays in the append-only transcript until compaction.

## Context and runtime

Resident Metal and Vulkan both use Q8_0 keys with Q4_0 values and default to a
100000-token agent context. The backend's automatic resident GPU prefill width
resolves to 1024 tokens. Compact Metal attention scratch keeps this faster
chunk width practical at long contexts. CPU uses F16 KV. SSD-streamed model
weights also default to a 100000-token context with F16 KV:

```sh
./q36-agent --metal
./q36-agent --metal --ssd-streaming --ssd-streaming-cache-experts 512
```

Explicit `--ctx`, `-ctk`, and `-ctv` values override the preset.

`q36-agent --chdir` loads its model and runtime assets from the launch directory,
then changes to the requested project for agent tools. Built-in path tools also
expand `~` to the current user's home directory.

## Terminal input

When a command such as `sudo` requests a terminal password, the interactive
agent opens a private prompt with input hidden. Type the password there;
Ctrl+C cancels the command. Password input is sent directly to the command,
without entering chat history, traces, or captured tool output. Other command
stdin reads receive EOF; `--non-interactive` cannot prompt for passwords.

## Hints and prefix files

Use `/hints on` for occasional short explanations of programming concepts behind
the current work, rendered as teal blockquotes. `/hints off` disables them. This
setting lasts for the current process and is reapplied after context compaction.

Both `q36` and `q36-agent` accept `--prefix-file FILE`. The file contains alternating
`USER:` and `ASSISTANT:` lines, starting with a user and ending with an assistant.
Turn content may span multiple lines. The CLI prefills this conversation before
the first interactive prompt; the agent keeps it through reset and compaction.
For example:

```text
USER: Our project uses C99 and has no external dependencies.
ASSISTANT: I will follow those constraints.
```

## Saved sessions

Sessions are stored in `~/.q36/kvcache`. Use `/save` to persist the current
session, `/list` to show saved sessions, and `/switch <sha>` to resume one.
The session ID remains stable across later saves. `/del <sha>` removes a saved
session. `/strip <sha>` keeps its transcript and title but removes the KV
payload; switching to a stripped session rebuilds the KV cache by prefilling
the saved text. `/compact` compacts the current context immediately.
Exiting during generation stops the worker before asking whether to save.
Sessions containing images cannot be saved yet; declining to exit after a save
failure returns to the current chat.
