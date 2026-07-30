#!/bin/sh
set -eu

backend=${1:---vulkan}
case "$backend" in
    --metal|--vulkan) ;;
    *)
        echo "usage: $0 [--metal|--vulkan] [--ssd-streaming]" >&2
        exit 2
        ;;
esac
streaming=${2:-}
case "$streaming" in
    ""|--ssd-streaming) ;;
    *)
        echo "usage: $0 [--metal|--vulkan] [--ssd-streaming]" >&2
        exit 2
        ;;
esac
stream_args=
if [ "$streaming" = "--ssd-streaming" ]; then
    stream_args="--ssd-streaming --ssd-streaming-cache-experts ${Q36_TEST_STREAM_CACHE_EXPERTS:-512}"
fi
model=${Q36_TEST_MODEL:-gguf/Qwen3.6-35B-A3B-AntirezExperts-IQ2XXS-gateup-Q2K-down-Q8rest.gguf}
port=${Q36_TEST_SERVER_PORT:-$((18000 + $$ % 1000))}
base=http://127.0.0.1:$port
tmp=$(mktemp -d /tmp/q36-server-live.XXXXXX)
pid=

cleanup() {
    if [ -n "$pid" ]; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    rm -rf "$tmp"
}
trap cleanup EXIT INT TERM

# stream_args is either empty or two validated CLI options. The cache value is
# validated again by q36-server.
./q36-server -m "$model" "$backend" $stream_args --ctx 4096 --tokens 8 \
    --host 127.0.0.1 --port "$port" --cors >"$tmp/server.log" 2>&1 &
pid=$!

ready=0
for _ in $(seq 1 120); do
    if curl -fsS --max-time 1 "$base/v1/models" >"$tmp/models.json"; then
        ready=1
        break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        cat "$tmp/server.log" >&2
        exit 1
    fi
    sleep 1
done
if [ "$ready" != 1 ]; then
    cat "$tmp/server.log" >&2
    echo "q36 server did not become ready" >&2
    exit 1
fi

grep -q 'qwen3.6-35b-a3b' "$tmp/models.json"
curl -fsS --max-time 5 -D "$tmp/cors.headers" -o /dev/null \
    -X OPTIONS "$base/v1/responses"
grep -qi '^Access-Control-Allow-Origin: \*' "$tmp/cors.headers"
grep -qi '^Access-Control-Allow-Methods: GET, POST, OPTIONS' "$tmp/cors.headers"
grep -qi '^Access-Control-Allow-Headers: \*' "$tmp/cors.headers"

curl -fsS --max-time 120 "$base/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Reply OK."}],"max_tokens":1,"think":false,"stream":false}' \
    >"$tmp/chat.json"
grep -q '"object":"chat.completion"' "$tmp/chat.json"

curl -fsS --max-time 120 "$base/v1/responses" \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.6-35b-a3b","input":"Reply OK.","max_output_tokens":1,"think":false,"stream":true}' \
    >"$tmp/responses.sse"
grep -q '^event: response.created' "$tmp/responses.sse"
grep -q '^event: response.completed' "$tmp/responses.sse"

echo "live server smoke: ok"
