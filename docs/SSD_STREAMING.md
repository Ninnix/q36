# Models Larger Than RAM

[README](../README.md) | [Models](MODELS.md) | [Metal setup](METAL.md#memory)

Use resident inference when the model and runtime fit: it is faster.
SSD streaming keeps a bounded cache of routed experts and reads missing
experts from the GGUF. Use a local SSD and leave memory for the OS.

The Qwen3.6 MoE model supports expert streaming. Dense Qwen3.8-27B weights
must fit resident. Flash Next has a separate streaming path described below.
Qwen3.6 Q4 and Flash Next are recommended for [distributed inference](DISTRIBUTED.md)
on two BC-250s; SSD streaming on one board offers modest performance.

## Start with the automatic budget

On a BC-250, use streaming for the larger Qwen3.6 Q4 model:

```sh
./download_model.sh q4
./q36 -m gguf/Qwen3.6-35B-A3B-IQ4XS-Q4K-Q8.gguf --ssd-streaming
```

The planner keeps non-routed weights resident and reserves memory for context,
prefill scratch, server sessions, and staging before sizing the expert cache.
Startup reports the effective slot count and allocation. Prefer this automatic
budget for a first run.

Larger contexts and additional sessions can shrink the cache. A session fails
cleanly if the mandatory weights and minimum working cache cannot fit.
Starting successfully does not guarantee useful generation speed: try a short
conversation before committing to a long task.

## Adjusting the cache

To leave more memory for context or other sessions, set an expert byte budget:

```sh
./q36 -m gguf/Qwen3.6-35B-A3B-IQ4XS-Q4K-Q8.gguf \
  --ssd-streaming --ssd-streaming-cache-experts 1GB
```

A plain integer requests expert slots instead of bytes. Both forms are targets
capped by the available memory. Non-routed weights, KV state, activations,
and scratch are additional to this cache budget.

The default hotlist biases eviction without reading expert weights at startup.
`--ssd-streaming-cold` starts empty; `--ssd-streaming-preload-experts N`
requests an actual startup read. These are useful for controlled measurements.

`--ssd-streaming-full-layers N` keeps the first `N` routed layers resident,
deducting their actual size from the dynamic cache budget. Leave its default
of zero on the BC-250: the dynamic-only cache is faster there.

For Apple Silicon commands and cache behavior, see [Metal](METAL.md#memory).

## Qwen3.8 Flash Next

Flash Next streams transformer weights through bounded Vulkan buffers and
reads n-gram rows from the second shard on demand:

```sh
./download_model.sh flash-next
./q36 \
  -m gguf/Qwen3.8-Flash-Next-GSQ-RCO-Coder-22GB-IQ2_XXS-00001-of-00002.gguf \
  --ssd-streaming --ctx 4096
```

Keep both shards together on the SSD. This path does not use the Qwen3.6
expert-cache controls. Prefill runs one token at a time, and checkpoint
restore replays tokens. For resident transformer weights across two boards,
see [distributed inference](DISTRIBUTED.md#qwen38-flash-next).

See [benchmarking](PERFORMANCE.md) for cache and throughput comparisons.
