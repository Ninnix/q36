# Models Larger Than RAM

[README](../README.md) | [Models](MODELS.md) | [Metal](METAL.md)

Use resident inference when the model and runtime fit. SSD streaming trades
speed for capacity by reading missing routed experts from the GGUF. This
path is for the Qwen3.6 MoE model; it does not make the dense Qwen3.8 model
an expert-streamed model.

## Start with the automatic budget

Metal and Vulkan SSD streaming keep non-routed weights resident and load
selected routed experts into a bounded backend cache. The planner reserves memory
for the configured context, prefill scratch, planned server sessions, and a staging
margin before assigning the remaining model budget to static weights and experts.
Explicit cache sizes are targets capped by the same budget. Creating a larger or
additional session reduces the expert cache when necessary; it fails cleanly if
static weights and the minimum working cache leave insufficient room. Startup
logs show the resolved context and model budgets.

```sh
./q36 --metal --ssd-streaming -p "Explain radix trees."
./q36 --vulkan --ssd-streaming --ssd-streaming-cache-experts 256 -p "Hello"
./q36 --metal --ssd-streaming --ssd-streaming-cache-experts 1GB -p "Hello"
```

`NGB` is a routed-expert byte budget. Q36 converts it to the number of expert
slots that fit the release GGUF. A plain integer requests an expert-slot count;
both forms are capped to leave room for the configured contexts.
Non-routed weights, KV cache, activations, and graph scratch need additional
memory. Startup prints the resolved slot count and actual cache allocation.

A built-in or `Q36_VK_STREAMING_EXPERT_HOTLIST` profile biases eviction
without reading expert weights at startup. It does not fill the cache with
arbitrary experts when no profile exists. Use `--ssd-streaming-cold` for an
empty cache, or `--ssd-streaming-preload-experts N` to request an explicit
weight preload. Metal stores mixed expert sizes in component-wise padded cache
slots, so the IQ2/Q2 and Q4 routed layers share one bounded cache and one
hotlist policy. Vulkan retains its original single-size-class behavior. The
`Q36_VK_STREAMING_EXPERT_HOTLIST` and
`Q36_VK_DISABLE_STREAMING_EXPERT_HOTLIST` names are retained for compatibility
and apply to both graph runtimes.

An explicit routed prefix can stay fully resident:

```sh
./q36 --ssd-streaming --ssd-streaming-full-layers 4 -p "Hello"
```

Full layers are charged at their actual byte size and the remaining budget
must still hold one layer of dynamic expert slots for prefill. The default is
zero because the dynamic-only cache is faster on the BC-250. Pass
`--ssd-streaming-full-layers 0` to disable an explicit setting.

## Metal cache

Metal follows DS4's bounded expert-cache design. Routed expert bytes are read
from the GGUF into size-limited shared Metal buffers, selected IDs are remapped
to cache slots, and cache misses use a biased least-recently-used eviction
policy. `--ssd-streaming-cache-experts`, preload, cold-cache, and
full-resident-layer controls therefore allocate and constrain real Metal
buffers; they are not hints to macOS VM paging.

For the mixed 13 GB model, each dynamic slot is sized for the largest routed
tensor layout. Smaller IQ2/Q2 experts and the six Q4_K expert layers can remain
cached together; changing precision at layer 34 does not reset the cache or
fall back to the CPU.

On a 16 GB Mac, start with the resident q2 model:

```sh
./q36 --metal -p "Hello"
```

For the larger release GGUF, or on an 8 GB Mac, start conservatively:

```sh
./q36 --metal --ssd-streaming --ctx 4096 -p "Hello"
./q36 --metal --ssd-streaming \
  --ssd-streaming-cache-experts 512 --ctx 4096 -p "Hello"
```

The automatic form uses `recommendedMaxWorkingSetSize`. The explicit example
allocates about 0.42 GiB for the q2 model or 0.85 GiB for the mixed model,
whose slots are padded for Q4_K. It leaves more headroom for the OS and KV
state. Increase it only while memory pressure, swap, and responsiveness remain
acceptable. A physical 8 GB Apple Silicon run is part of release QA;
`--simulate-used-memory 8GB` on a larger Mac is diagnostic evidence, not a
replacement.

`--ssd-streaming-preload-experts N` performs an actual startup read. The normal
automatic hotlist only biases eviction and does not read expert weights.
`--ssd-streaming-full-layers N` loads the first `N` routed layers into separate
full-resident Metal buffers and deducts their exact bytes from the dynamic
budget. `q36-bench` prints cache slots, cache/full-layer GiB, hits, misses,
loads, evictions, and GGUF bytes read in its final memory report.
