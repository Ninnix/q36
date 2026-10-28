#!/usr/bin/env python3
"""Mock q36-server for WebUI browser tests: serves webui/ from disk plus a fake OpenAI API."""
import json
import mimetypes
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "webui")
REQUESTS = []


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send_json(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/health":
            return self.send_json({"status": "ok"})
        if path == "/props":
            return self.send_json({"model": "mock-q36", "n_ctx": 32768, "default_max_tokens": 4096,
                                   "vision": True, "slots": 1, "qwen38": False})
        if path == "/__requests":
            return self.send_json(REQUESTS)
        if path == "/":
            path = "/index.html"
        full = os.path.normpath(os.path.join(ROOT, path.lstrip("/")))
        if not full.startswith(os.path.normpath(ROOT)) or not os.path.isfile(full):
            return self.send_json({"error": {"message": "not found"}}, 404)
        with open(full, "rb") as f:
            data = f.read()
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype + "; charset=utf-8" if ctype.startswith("text/") or ctype.endswith("javascript") else ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if not os.path.basename(full).startswith("preview."):
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
                             "frame-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path != "/v1/chat/completions":
            return self.send_json({"error": {"message": "not found"}}, 404)
        REQUESTS.append(body)
        msgs = body.get("messages", [])
        last = ""
        for m in reversed(msgs):
            if m.get("role") == "user":
                c = m.get("content")
                last = c if isinstance(c, str) else " ".join(p.get("text", "") for p in c if p.get("type") == "text")
                break
        if "TOOLONG" in last:
            return self.send_json({"error": {"message": "prompt too long", "code": "context_length_exceeded",
                                             "n_prompt_tokens": 40000, "n_ctx": 32768}}, 400)
        slow = "SLOW" in last
        reasoning = "Let me think about: " + last[:40]
        content = ("Here is **bold** text and `code`.\n\n```python\ndef add(a, b):\n    return a + b\n```\n\n"
                   "| a | b |\n|---|---|\n| 1 | 2 |\n\nInline math $x^2 + \\frac{1}{2}$ done.\n\n"
                   "```html\n<h1 id=\"hi\">Preview OK</h1>\n```\n")
        if slow:
            content = "word " * 400
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def ev(obj):
            self.wfile.write(b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n")
            self.wfile.flush()

        try:
            ev({"choices": [{"delta": {"role": "assistant"}, "index": 0}]})
            if body.get("chat_template_kwargs", {}).get("enable_thinking") is not False and not slow:
                for i in range(0, len(reasoning), 8):
                    ev({"choices": [{"delta": {"reasoning_content": reasoning[i:i + 8]}, "index": 0}]})
                    time.sleep(0.01)
            step = 6
            for i in range(0, len(content), step):
                ev({"choices": [{"delta": {"content": content[i:i + step]}, "index": 0}]})
                time.sleep(0.05 if slow else 0.008)
            ev({"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]})
            ev({"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 80, "total_tokens": 200,
                                         "prompt_tokens_details": {"cached_tokens": 100}}})
            ev(b"[DONE]")
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
