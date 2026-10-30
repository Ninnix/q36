# UI

[README](../README.md#server)

QuarkStar intentionally does not include a graphical or web UI. The goal is
to keep the repository small and self-contained: C for the engine, native
code and kernels for GPU runtimes such as Vulkan and Metal, and no external
application dependencies. Future GPU backends can follow the same approach.

Use a separate UI, or build your own, and connect it to `q36-server` through
its OpenAI- or Anthropic-compatible HTTP API. The UI owns presentation and
its own dependencies; QuarkStar owns inference. Installing a UI does not
change how QuarkStar is built.

## Connect a UI

Start the server:

```sh
./q36-server --vulkan --ctx 32768
```

On Apple Silicon, use `--metal` instead of `--vulkan`. The default address
is `http://127.0.0.1:8000`.

- OpenAI clients use base URL `http://127.0.0.1:8000/v1` and the
  `/v1/chat/completions` endpoint.
- Anthropic clients use base URL `http://127.0.0.1:8000` and the
  `/v1/messages` endpoint.

Choose a UI that supports the protocol you intend to use. Both endpoints
support streaming and tool calls. Available model IDs are returned by
`GET /v1/models`.

The UI must be able to reach the server from the machine or container where
its requests originate. For a browser that calls Q36 directly, pass `--cors`
to enable cross-origin requests. A UI that forwards requests through its own
backend does not need CORS.

## Open WebUI

Open WebUI is one example of a separate UI. Connect it through QuarkStar's
OpenAI-compatible API.

In **Settings > Admin > Connections**, add an OpenAI connection with URL
`http://127.0.0.1:8000/v1` and API key `q36-local`. The key is a placeholder;
the server does not authenticate it. Select the model returned by `/v1/models`,
such as `qwen3.6-35b-a3b`.

When Open WebUI runs in a Linux Docker or Podman container, `--network host`
lets its backend reach the same loopback address. The connection can also be
configured with `OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1`,
`OPENAI_API_KEY=q36-local`, and `ENABLE_OLLAMA_API=false`. CORS is unnecessary
for this backend-to-backend connection.

## Context and embeddings

Open WebUI's built-in tools add several thousand prompt tokens: a 4096-token
server context can be too small even for a short question. Chat history,
reasoning, and output must also fit within `--ctx`. Document retrieval and
long-term memory need a separate embedding model; Q36 has no embeddings
endpoint.
