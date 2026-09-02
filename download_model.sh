#!/bin/sh
set -e

REPO="Ninnix96/Qwen3.6-35B-A3B-gguf"
KAT_REPO="Ninnix96/KAT-Coder-V2.5-Dev-gguf"
DENSE_REPO="unsloth/Qwen3.8-27B-GGUF"
VISION_REPO="unsloth/Qwen3.6-35B-A3B-GGUF"
Q2_FILE="Qwen3.6-35B-A3B-AntirezExperts-IQ2XXS-gateup-Q2K-down-Q8rest.gguf"
Q2_Q4_FILE="Qwen3.6-35B-A3B-Layers34-39Q4KExperts-OtherExpertLayersIQ2XXSGateUp-Q2KDown-Q8Rest-imatrix.gguf"
MTP_FILE="Qwen3.6-35B-A3B-MTP-Q4K-Q8_0-F32.gguf"
KAT_FILE="KAT-Coder-V2.5-Dev-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-imatrix.gguf"
DENSE_FILE="Qwen3.8-27B-UD-IQ3_S.gguf"
VISION_REMOTE="mmproj-F16.gguf"
VISION_FILE="Qwen3.6-35B-A3B-mmproj-F16.gguf"
DENSE_VISION_FILE="Qwen3.8-27B-mmproj-F16.gguf"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT_DIR=${Q36_GGUF_DIR:-"$ROOT/gguf"}
case "$OUT_DIR" in
    /*) ;;
    *) OUT_DIR="$ROOT/$OUT_DIR" ;;
esac
TOKEN=${HF_TOKEN:-}

usage() {
    cat <<EOF
QuarkStar GGUF downloader

Usage:
  ./download_model.sh q2-imatrix [--token TOKEN]
  ./download_model.sh q2-q4-imatrix [--token TOKEN]
  ./download_model.sh kat-coder [--token TOKEN]
  ./download_model.sh 27b [--token TOKEN]
  ./download_model.sh qwen36-vision [--token TOKEN]
  ./download_model.sh qwen38-vision [--token TOKEN]
  ./download_model.sh mtp [--token TOKEN]

Targets:
  q2-imatrix
       q2-imatrix quant from $REPO, about 10 GB on disk.
       Routed MoE experts use IQ2_XXS for gate/up and Q2_K for down.
       The rest of the tensors are left at higher precision. This model
       fits a BC-250 with 16 GB of unified GDDR6 fully resident.

  q2-q4-imatrix
       Mixed quant: same as q2-imatrix but the last 6 layers (34..39) use
       Q4_K routed experts. About 13 GB on disk, higher quality inference.
       On a 16 GB BC-250 run it with --ssd-streaming to keep memory free.

  kat-coder
       KAT-Coder V2.5 Dev IQ2_XXS imatrix quant from $KAT_REPO,
       about 11 GB on disk.

  27b
       Dense Qwen3.8 27B Dynamic 3.0 IQ3_S quant from $DENSE_REPO.

  qwen36-vision
       Qwen3.6 35B A3B F16 vision projector from $VISION_REPO.
       It is downloaded as $VISION_FILE and does not update ./q36moe.gguf.

  qwen38-vision
       Qwen3.8 27B F16 vision projector from $DENSE_REPO.
       It is downloaded as $DENSE_VISION_FILE and does not update ./q36moe.gguf.

  mtp  Optional speculative decoding component for either main-model target.
       It must be enabled explicitly with --mtp when starting a runtime.

Options:
  --token TOKEN  Hugging Face token. Otherwise HF_TOKEN or the local HF
                 token cache is used if present. The main repository is
                 public, so a token is usually not required.

Environment:
  Q36_GGUF_DIR   Directory used for downloaded GGUF files.
                 Default: ./gguf

After a main-model download the script updates the legacy link:
  ./q36moe.gguf -> <download directory>/<selected model>

With the default output directory, q2-imatrix matches the compiled model path:
  ./q36 -p "Hello"
  ./q36-server --ctx 32768

Select q2-q4-imatrix explicitly:
  ./q36 -m $OUT_DIR/$Q2_Q4_FILE --ssd-streaming -p "Hello"

After downloading mtp, enable it explicitly, for example:
  ./q36 --mtp <download directory>/$MTP_FILE --mtp-draft 2

EOF
}

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

MODEL=$1
shift
LINK_MODEL=1
MODEL_REPO=$REPO
MODEL_REMOTE=

case "$MODEL" in
    q2-imatrix) MODEL_FILE=$Q2_FILE ;;
    q2-q4-imatrix) MODEL_FILE=$Q2_Q4_FILE ;;
    kat-coder) MODEL_REPO=$KAT_REPO; MODEL_FILE=$KAT_FILE ;;
    27b) MODEL_REPO=$DENSE_REPO; MODEL_FILE=$DENSE_FILE ;;
    qwen36-vision)
        MODEL_REPO=$VISION_REPO
        MODEL_REMOTE=$VISION_REMOTE
        MODEL_FILE=$VISION_FILE
        LINK_MODEL=0
        ;;
    qwen38-vision)
        MODEL_REPO=$DENSE_REPO
        MODEL_REMOTE=$VISION_REMOTE
        MODEL_FILE=$DENSE_VISION_FILE
        LINK_MODEL=0
        ;;
    mtp) MODEL_FILE=$MTP_FILE; LINK_MODEL=0 ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown model: $MODEL" >&2
        echo >&2
        usage >&2
        exit 1
        ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --token)
            shift
            if [ $# -eq 0 ]; then
                echo "Missing value after --token" >&2
                exit 1
            fi
            TOKEN=$1
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

if [ -z "$MODEL_REMOTE" ]; then
    MODEL_REMOTE=$MODEL_FILE
fi

if [ -z "$TOKEN" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

download_one() {
    repo=$1
    remote=$2
    file=$3
    out="$OUT_DIR/$file"
    part="$out.part"
    aria2_part="$out.aria2"
    url="https://huggingface.co/$repo/resolve/main/$remote"

    mkdir -p "$OUT_DIR"

    # Refuse to interleave with a half-done aria2 download. The user
    # probably wanted that one to finish, not a parallel curl.
    if [ -e "$aria2_part" ]; then
        echo "Found incomplete aria2 download sidecar: $aria2_part" >&2
        echo "Finish or remove that partial download before using this curl downloader." >&2
        exit 1
    fi

    if [ -s "$out" ]; then
        echo "Already downloaded: $out"
        return
    fi

    echo "Downloading $remote as $file"
    echo "from https://huggingface.co/$repo"
    echo "If the download stops, run the same command again to resume it."

    if [ -n "$TOKEN" ]; then
        curl -fL --progress-meter -C - -H "Authorization: Bearer $TOKEN" -o "$part" "$url"
    else
        curl -fL --progress-meter -C - -o "$part" "$url"
    fi

    mv "$part" "$out"
}

download_one "$MODEL_REPO" "$MODEL_REMOTE" "$MODEL_FILE"

if [ "$MODEL" = "mtp" ]; then
    echo
    echo "MTP is an optional component for q2-imatrix."
    echo "Enable it explicitly, for example:"
    echo "  ./q36 --mtp $OUT_DIR/$MTP_FILE --mtp-draft 2"
elif [ "$MODEL" = "qwen36-vision" ]; then
    echo
    echo "Vision projector downloaded. Use it with the standard model:"
    echo "  ./q36 --vision $OUT_DIR/$VISION_FILE"
elif [ "$MODEL" = "qwen38-vision" ]; then
    echo
    echo "Vision projector downloaded. Use it with the dense model:"
    echo "  ./q36 -m $OUT_DIR/$DENSE_FILE --vision $OUT_DIR/$DENSE_VISION_FILE"
elif [ "$LINK_MODEL" -eq 1 ]; then
    cd "$ROOT"
    ln -sfn "$OUT_DIR/$MODEL_FILE" q36moe.gguf
    echo "Linked ./q36moe.gguf -> $OUT_DIR/$MODEL_FILE"
fi

echo
echo "Done."
