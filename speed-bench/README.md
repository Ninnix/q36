## Benchmarking

Here we collect prefill and generation speed obtained with the BC-250.

Run `q36-bench` as:

```
./q36-bench \
  -m gguf/Qwen3.6-35B-A3B-AntirezExperts-IQ2XXS-gateup-Q2K-down-Q8rest.gguf \
  --prompt-file tests/long_context_story_prompt.txt \
  --ctx-start 2048 \
  --ctx-max 30720 \
  --step-incr 2048 \
  --gen-tokens 128 \
  -ctk q8_0 -ctv q4_0 \
  --csv speed-bench/bc250.csv
```

The current tokenizer produces 31217 tokens from the story fixture, making
30720 the last complete 2048-token frontier.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/bc250.csv --title "BC-250 t/s"
```
