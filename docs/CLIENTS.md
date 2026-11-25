# Coding Agent Clients

[README](../README.md) | [Server](SERVER.md) | [Native agent](AGENT.md)

`q36-server` can be used by local coding agents that speak OpenAI-compatible
chat completions. Start the server first, and set the client context limit
no higher than the `--ctx` value you started the server with:

```sh
./q36-server --ctx 32768 --kv-disk-dir /tmp/q36-kv --kv-disk-space-mb 8192
```

On a BC-250 with 16 GB of unified memory, Q2 weights take about 11.3 GiB,
which leaves roughly 2–4 GB for KV cache, scratch buffers, OS and your
client. The model's native context is 256K tokens, but the live context must
still fit in available memory. Disk KV checkpoints avoid repeated prefill and
preserve sessions across restarts; they do not enlarge the active context
window.

The examples allow 16384 output tokens within a 32768-token context.
Prompt, thinking, and output share that window; a client limit does not
enlarge the server allocation.

## OpenCode

Add a provider and agent entry to
`~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "q36": {
      "name": "q36.c (local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "q36-local"
      },
      "models": {
        "qwen3.6-35b-a3b": {
          "name": "Qwen 3.6 MoE (q36.c local)",
          "limit": {
            "context": 32768,
            "output": 16384
          }
        }
      }
    }
  },
  "agent": {
    "q36": {
      "description": "Qwen 3.6 MoE served by local q36-server",
      "model": "q36/qwen3.6-35b-a3b",
      "temperature": 0
    }
  }
}
```

## Pi

Add a provider to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "q36": {
      "name": "q36.c local",
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "q36-local",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_tokens",
        "supportsStrictMode": false,
        "thinkingFormat": "qwen",
        "requiresReasoningContentOnAssistantMessages": true
      },
      "models": [
        {
          "id": "qwen3.6-35b-a3b",
          "name": "Qwen 3.6 MoE (q36.c local)",
          "reasoning": true,
          "thinkingLevelMap": {
            "off": null,
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh"
          },
          "input": ["text"],
          "contextWindow": 32768,
          "maxTokens": 16384,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
```

Optionally make it the default Pi model in `~/.pi/agent/settings.json`:

```json
{
  "defaultProvider": "q36",
  "defaultModel": "qwen3.6-35b-a3b"
}
```

## Inkling 2 Mini on one board

Start the native Vulkan server with the Inkling GGUF.

```sh
./q36-server --vulkan -t 16 --ctx 16384 -m gguf/Inkling-2-Mini-UD-IQ3_XXS.gguf
```

Enable image input with the separate projector.

```sh
./download_model.sh inkling-2-mini-vision
./q36-server --vulkan -t 16 --ctx 16384 \
  -m gguf/Inkling-2-Mini-UD-IQ3_XXS.gguf \
  --vision gguf/Inkling-2-Mini-mmproj-F16.gguf
```

QuarkStar reads the projector matrices from disk into a reusable 32 MiB Vulkan weight buffer. The same `--vision` flag enables `/read FILE [PROMPT]` in `q36` and `view_image` in `q36-agent`.

OpenCode 2.0.29 accepts this project-local `opencode.json`. Each variant sends an explicit effort in the request body.

```json
{
  "model": "q36/inkling-2-mini",
  "providers": {
    "q36": {
      "name": "QuarkStar",
      "package": "@ai-sdk/openai-compatible",
      "settings": {"baseURL": "http://127.0.0.1:8000/v1", "apiKey": "local"},
      "models": {
        "inkling-2-mini": {
          "name": "Inkling 2 Mini",
          "limit": {"context": 16384, "output": 768},
          "capabilities": {"tools": true, "reasoning": {"supported": true}},
          "variants": [
            {"id": "none", "body": {"reasoning_effort": "none"}},
            {"id": "minimal", "body": {"reasoning_effort": "minimal"}},
            {"id": "low", "body": {"reasoning_effort": "low"}},
            {"id": "medium", "body": {"reasoning_effort": "medium"}},
            {"id": "high", "body": {"reasoning_effort": "high"}},
            {"id": "xhigh", "body": {"reasoning_effort": "xhigh"}}
          ]
        }
      }
    }
  },
  "websearch": {"provider": "tinyfish"}
}
```

For Pi 0.74.0, use the Pi example above with model ID `inkling-2-mini`, `contextWindow` set to 16384, and `maxTokens` set to 768. Set `compat.thinkingFormat` to `"openai"` so Pi sends the selected level. In `thinkingLevelMap`, use `"off": "none"` and `"minimal": "minimal"`. Pi supplies its own tools. Web search requires a search extension.

```sh
opencode run --standalone --model 'q36/inkling-2-mini#low' 'Read README.md and give its first heading.'
pi -p --provider q36 --model inkling-2-mini --thinking medium 'Read README.md and give its first heading.'
```

## Claude Code

Use the Anthropic-compatible endpoint. Save this wrapper with a name
different from `claude`, such as `claude-q36`:

```sh
#!/bin/sh
unset ANTHROPIC_API_KEY

export ANTHROPIC_BASE_URL="${Q36_ANTHROPIC_BASE_URL:-http://127.0.0.1:8000}"
export ANTHROPIC_AUTH_TOKEN="${Q36_API_KEY:-q36-local}"
export ANTHROPIC_MODEL="qwen3.6-35b-a3b"

export ANTHROPIC_CUSTOM_MODEL_OPTION="qwen3.6-35b-a3b"
export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="Qwen 3.6 MoE local q36"
export ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION="q36.c local GGUF"

export ANTHROPIC_DEFAULT_SONNET_MODEL="qwen3.6-35b-a3b"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="qwen3.6-35b-a3b"
export ANTHROPIC_DEFAULT_OPUS_MODEL="qwen3.6-35b-a3b"
export CLAUDE_CODE_SUBAGENT_MODEL="qwen3.6-35b-a3b"

export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1
export CLAUDE_STREAM_IDLE_TIMEOUT_MS=600000

exec claude "$@"
```

Claude Code may send a large initial prompt, often around 25k tokens,
before it starts doing useful work. Keep `--kv-disk-dir` enabled: after the
first expensive prefill, the disk KV cache lets later continuations or
restarted sessions reuse the saved prefix instead of processing the whole
prompt again.

The examples select Qwen3.6. For a dense model, pass its GGUF to the server
and use the model ID returned by `/v1/models`. The `q36-local` API key is a
placeholder, not server authentication.
