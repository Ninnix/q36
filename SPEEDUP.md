# Speedup Rules

This file is a rulebook for implementing new Quarkstar speedups and kernel optimizations to improve prefill and decode tokens per second (t/s) for a target runtime (Vulkan, Metal) and a target model.

* A speedup must remain effective as the context size grows. Benchmark `q36_bench.c` with multiple context sizes, such as 128, 2K, and 4K tokens. In some cases, also test 8K and 16K.

* After a change brings a speedup, always verify that it is mathematically equivalent and produces logits consistent with the previous implementation. A good check is the parity test in `q36_test.c`:

  * CPU vs. llama: use 16 threads and short prompts only. This is required only if the `q36.c` CPU path was touched by the changes.
  * CPU vs. GPU (Vulkan or Metal): use short prompts and 16 CPU threads.
  * All parity thresholds from top-1 through top-64 must pass.

* RAM efficiency is very important for this tool. Even an additional 100 MB can be a problem. As a general rule, a speedup should not increase RAM usage. An exception can be made for a very good trade-off. For example, a speculative decoding layer that provides a 2x decode speedup at the cost of 500 MB of additional RAM can be a good fit.

* Check temperatures during benchmarks, since temperature can affect performance. Optimizations that reduce temperature or power usage while maintaining or improving performance are also valuable.

Never launch two heavy inference processes at the same time. Do not run benchmarks in parallel. Models are often larger than 11 GB, while test machines may have only 16 GB of RAM.
