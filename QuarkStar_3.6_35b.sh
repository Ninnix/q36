#!/bin/bash
# QuarkStar 35B — запуск q36-server с проверкой состояния

DIR="/home/deck/q36-gui"
MODEL="$DIR/gguf/Qwen3.6-35B-A3B-IQ2XXS-Q2K-Q8.gguf"
PORT=9080
URL="http://localhost:$PORT"

# При запуске двойным кликом — открыть себя в konsole
if [ -z "$QUARKSTAR_IN_KONSOLE" ]; then
    export QUARKSTAR_IN_KONSOLE=1
    exec konsole -e bash "$(readlink -f "$0")"
fi

port_open() {
    (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null
}

server_running() {
    pgrep -f "q36-server" >/dev/null 2>&1 || port_open
}

wait_key() {
    echo
    read -n 1 -s -r -p "Нажмите любую клавишу для закрытия окна..."
}

cd "$DIR" || { echo "Ошибка: папка $DIR не найдена"; wait_key; exit 1; }

echo "=== QuarkStar 35B ==="

# Проверка: сервер уже запущен?
if server_running; then
    echo "[OK] Сервер уже запущен: $URL"
    echo "Останавливать его не буду. Закройте окно или нажмите клавишу."
    wait_key
    exit 0
fi

if [ ! -f "$MODEL" ]; then
    echo "Ошибка: модель не найдена: $MODEL"
    wait_key
    exit 1
fi

echo "Сервер не запущен. Запускаю..."
echo "Загрузка модели может занять некоторое время."
echo

./q36-server -m "$MODEL" --vulkan --mtp --ctx 32768 \
  --kv-disk-dir /tmp/q36-kv --kv-disk-space-mb 8192 --port 9080 --open-browser &
SERVER_PID=$!

# Остановить сервер при закрытии окна / Ctrl+C
trap 'echo; echo "Останавливаю сервер..."; kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null' EXIT INT TERM

# Ждём, пока порт откроется (до 10 минут)
for i in $(seq 1 600); do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo
        echo "[ОШИБКА] Сервер завершился во время запуска. См. сообщения выше."
        wait_key
        exit 1
    fi
    if port_open; then
        echo
        echo "=========================================="
        echo "[OK] СЕРВЕР ЗАПУЩЕН: $URL"
        echo "=========================================="
        echo "Не закрывайте это окно — закрытие остановит сервер."
        break
    fi
    sleep 1
done

if ! port_open; then
    echo "[ВНИМАНИЕ] Порт $PORT не открылся за 10 минут."
fi

wait $SERVER_PID
echo
echo "Сервер остановлен."
wait_key
