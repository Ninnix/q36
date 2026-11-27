# Directional Steering

[README](../README.md) | [Vector extraction](../dir-steering/README.md)

Q36 can edit attention and FFN outputs with the same projection used by DS4:

```text
y = y - scale * direction[layer] * dot(direction[layer], y)
```

The steering file is a flat native-endian `f32` matrix with one normalized
direction per layer. Qwen3.8/3.6 27B uses `64 x 5120`, Qwen3.6 35B A3B uses
`40 x 2048`, and Inkling 2 Mini uses `48 x 2304`. Inkling uses one row per
executed layer in forward order, including separate rows for all three passes
through its middle block. Positive scales remove the represented direction.
Negative scales amplify it. Supplying a file sets the FFN scale to `1` by default.

```sh
./q36 -p "Write tersely" \
  --dir-steering-file qwen-direction.f32 \
  --dir-steering-ffn 0.8
```

`--dir-steering-attn F` applies the same edit after attention outputs. With no
file, or with both scales set to zero, inference follows the normal path. CPU
and both GPU runtimes apply the same operation during prefill and decode; Metal
and Vulkan keep the matrix resident and project activations in place.
Inkling 2 Mini runs on CPU and Vulkan.

In the CLI and agent REPL, `/steer` displays the FFN scale and `/steer F`
sets it for subsequent tokens, without rebuilding the existing KV cache.
Load with `--dir-steering-ffn 0` to start with steering disabled and enable it
later. Scales range from `-100` to `100`; start near `-1` to `1`.
Each session keeps its own steering strengths; the vector bank is shared.
The agent steers its decisions and prose, while processing prompt text and
generating tool calls without steering to preserve exact arguments and file
contents. CLI and server steering covers both prefill and decode.
The MTP predictor has no steering projection and the steered target verifies
its drafts. Inkling's DSpark predictor receives features from the target.

See [dir-steering/README.md](../dir-steering/README.md) to build a vector for
`gguf/Qwen3.8-27B-UD-IQ3_S.gguf` from the bundled paired prompt sets.
