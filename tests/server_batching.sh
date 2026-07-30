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
port=${Q36_TEST_SERVER_PORT:-$((19000 + $$ % 1000))}
base=http://127.0.0.1:$port
tmp=$(mktemp -d /tmp/q36-server-batch.XXXXXX)
pid=
client_pids=
status=1

cleanup() {
    for client_pid in $client_pids; do
        kill "$client_pid" 2>/dev/null || true
        wait "$client_pid" 2>/dev/null || true
    done
    if [ -n "$pid" ]; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    if [ "$status" = 0 ]; then
        rm -rf "$tmp"
    else
        echo "batched server artifacts: $tmp" >&2
    fi
}
trap cleanup EXIT INT TERM

if ./q36-server --batched-session 0 >/dev/null 2>&1; then
    echo "--batched-session 0 was accepted" >&2
    exit 1
fi
if ./q36-server --batched-session -1 >/dev/null 2>&1; then
    echo "--batched-session -1 was accepted" >&2
    exit 1
fi

# stream_args is either empty or two validated CLI options. The cache value is
# validated again by q36-server.
Q36_SERVER_BATCH_LOG=1 Q36_VK_SESSION_BATCH_LOG=1 \
./q36-server -m "$model" "$backend" $stream_args --ctx 4096 --prefill-chunk 8 --tokens 32 \
    --batched-session 4 --kv-disk-dir "$tmp/kv" --kv-disk-space-mb 1024 \
    --kv-cache-min-tokens 32 --kv-cache-cold-max-tokens 4096 \
    --kv-cache-boundary-trim-tokens 0 --kv-cache-boundary-align-tokens 32 \
    --host 127.0.0.1 --port "$port" >"$tmp/server.log" 2>&1 &
pid=$!

ready=0
for _ in $(seq 1 180); do
    if curl -fsS --max-time 1 "$base/v1/models" >"$tmp/models.json" 2>/dev/null; then
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
    exit 1
fi

python3 tests/test_server_batching.py --url "$base" --pairs "${Q36_TEST_BATCH_PAIRS:-5}" \
    --nonce "batch-$port" \
    >"$tmp/nonstream.json"
python3 tests/test_server_batching.py --url "$base" --pairs "${Q36_TEST_BATCH_PAIRS:-5}" --stream \
    --nonce "batch-$port" \
    >"$tmp/stream.json"
grep -q '"status": "PASS"' "$tmp/nonstream.json"
grep -q '"status": "PASS"' "$tmp/stream.json"

# Drop a streaming client while its request is active. The resident worker must
# release its slot and the coordinator must remain stoppable.
curl -sS --max-time 1 "$base/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Write a long numbered explanation of batching fairness."}],"max_tokens":1024,"temperature":0,"think":false,"stream":true}' \
    >"$tmp/disconnect.sse" 2>/dev/null || true

# Shut down with more requests than slots so active and queued clients are both
# forced through their cancellation paths.
for i in $(seq 1 8); do
    curl -sS --max-time 30 "$base/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"qwen3.6-35b-a3b\",\"messages\":[{\"role\":\"user\",\"content\":\"Shutdown batching request $i: write a long explanation.\"}],\"max_tokens\":1024,\"temperature\":0,\"think\":false,\"stream\":true}" \
        >"$tmp/shutdown-$i.sse" 2>/dev/null &
    client_pids="$client_pids $!"
done
sleep 1
kill -TERM "$pid"
wait "$pid"
pid=
for client_pid in $client_pids; do
    wait "$client_pid" 2>/dev/null || true
done

grep -q 'resident_sessions=4' "$tmp/server.log"
if [ "$streaming" = "--ssd-streaming" ]; then
    grep -Eq 'session batch rows=[2-8] native=0 ordered=1' "$tmp/server.log"
    grep -q 'SSD expert cache allocated' "$tmp/server.log"
else
    grep -Eq '(metal|vulkan) session batch rows=[2-8] native=1' "$tmp/server.log"
fi
grep -q 'kv cache hit text' "$tmp/server.log"
grep -q 'shutdown requested' "$tmp/server.log"
cat "$tmp/nonstream.json"
cat "$tmp/stream.json"
echo "batched server: ok"
status=0
