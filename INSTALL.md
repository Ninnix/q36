# Installation

Steps to clone, build, and run QuarkStar (q36) with its chat Web UI.

## 1. Clone

    git clone https://github.com/Shalaykin1/q36-gui.git
    cd q36-gui

## 2. Requirements

Pick one backend:

* **Linux with Vulkan 1.1+** (shader `float64`/`int64` support). Primary
  tested device is the AMD BC-250; see [BC250.md](BC250.md) for RADV,
  unified-memory, and governor setup.
* **Apple Silicon (M1 or newer)** with macOS 11+ and Xcode or the Command
  Line Tools (for Metal).
* **CPU** as a portable fallback with no GPU dependency (slower, but no
  extra setup).

Windows is not currently a build target.

## 3. Build

    make vulkan   # Linux with Vulkan (e.g. BC-250)
    # or
    make metal    # Apple Silicon
    # or
    make cpu      # CPU-only, no GPU required

## 4. Download model weights

    ./download_model.sh q2            # Qwen3.6-35B-A3B MoE, ~12 GiB (BC-250 default)
    ./download_model.sh 27b           # Dense Qwen3.8-27B IQ3_S, 11.2 GiB
    ./download_model.sh 27b-iq3-xxs   # Dense Qwen3.8-27B IQ3_XXS, 9.7 GiB

This downloads into `gguf/` and creates a `./q36moe.gguf` symlink for the
default target. Pass `--token TOKEN` or set `HF_TOKEN` if the HuggingFace
repo requires authentication. Downloads resume if interrupted.

## 5. Run the server (with Web UI)

    ./q36-server -m gguf/Qwen3.6-35B-A3B-IQ2XXS-Q2K-Q8.gguf --vulkan --mtp --ctx 16384 \
      --kv-disk-dir /tmp/q36-kv --kv-disk-space-mb 8192 --port 9080 --open-browser

Adjust `-m` to the downloaded model file and the backend flag
(`--vulkan`, `--metal`, or omit both for a CPU build). `--ctx`, `--kv-disk-dir`,
`--kv-disk-space-mb`, and `--port` are optional tuning flags shown above as an
example; drop or change them as needed.

`--open-browser` opens the chat UI automatically (at `http://127.0.0.1:PORT/`,
`PORT` defaulting to 8000 or whatever `--port` is set to) once the server
starts listening — use it when the server runs on the same machine as your
browser. If you connect to the server remotely (e.g. over SSH or a tunnel),
drop the flag and open that URL in your browser by hand.

See `./q36-server --help` for all flags (context size, `--vision` for image
input, `--kv-disk-dir`/`--kv-disk-space-mb` for disk KV cache, `--cors`,
etc.) and the `## Server` and `### Web UI` sections of
[README.md](README.md) for details.
