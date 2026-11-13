#!/bin/sh
set -e

REPO="Ninnix96/Qwen3.6-35B-A3B-gguf"
DENSE_REPO="unsloth/Qwen3.8-27B-GGUF"
DENSE_SMALL_REPO="ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF"
FLASH_REPO="Ninnix96/Qwen3.8-Flash-Next-GSQ-RCO-Coder-22GB-gguf"
VISION_REPO="unsloth/Qwen3.6-35B-A3B-GGUF"
Q2_FILE="Qwen3.6-35B-A3B-IQ2XXS-Q2K-Q8.gguf"
Q4_FILE="Qwen3.6-35B-A3B-IQ4XS-Q4K-Q8.gguf"
DENSE_FILE="Qwen3.8-27B-UD-IQ3_S.gguf"
DENSE_Q6_FILE="Qwen3.8-27B-UD-Q6_K.gguf"
DENSE_SMALL_FILE="Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"
FLASH_FILE="Qwen3.8-Flash-Next-GSQ-RCO-Coder-22GB-IQ2_XXS-00001-of-00002.gguf"
FLASH_NGRAM_FILE="Qwen3.8-Flash-Next-GSQ-RCO-Coder-22GB-IQ2_XXS-00002-of-00002.gguf"
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
  ./download_model.sh q2 [--token TOKEN]
  ./download_model.sh q4 [--token TOKEN]
  ./download_model.sh 27b [--token TOKEN]
  ./download_model.sh 27b-q6 [--token TOKEN]
  ./download_model.sh 27b-iq3-xxs [--token TOKEN]
  ./download_model.sh flash-next [--token TOKEN]
  ./download_model.sh qwen36-vision [--token TOKEN]
  ./download_model.sh qwen38-vision [--token TOKEN]

Targets:
  q2
       Qwen3.6 MoE quant from $REPO, about 12.1 GB on disk.
       Routed MoE experts use IQ2_XXS for gate/up and Q2_K for down.
       The rest of the tensors are left at higher precision. This model
       fits a BC-250 with 16 GB of unified GDDR6 fully resident.
       Includes a 333 MB MTP head; enable it with --mtp.

  q4
       Qwen3.6 MoE IQ4_XS gate/up and Q4_K down, about 19.05 GiB.
       Use SSD streaming on one BC-250, or a two-board layer pipeline.

  27b
       Dense Qwen3.8 27B Dynamic 3.0 IQ3_S quant from $DENSE_REPO.

  27b-q6
       Dense Qwen3.8 27B Dynamic Q6_K quant from $DENSE_REPO, about 20.5 GiB.
       Use a two-board layer pipeline on BC-250. Includes an embedded MTP
       head; enable it with --mtp on the coordinator. CPU inference supported.

  27b-iq3-xxs
       Dense Qwen3.8 27B GSQ-RCO IQ3_XXS from $DENSE_SMALL_REPO.
       Smaller weights leave more memory for context. Includes an embedded
       MTP head; enable it with --mtp. Uses the qwen38-vision projector.

  flash-next (also qwen38-flash-next)
       Qwen3.8 Flash Next GSQ-RCO Coder from $FLASH_REPO.
       Downloads both required GGUF shards, about 50.8 GB in total.
       Keep them together and pass the first shard with -m. The n-gram
       shard stays on the coordinator in distributed mode. No MTP head.

  qwen36-vision
       Qwen3.6 35B A3B F16 vision projector from $VISION_REPO.
       It is downloaded as $VISION_FILE and does not update ./q36moe.gguf.

  qwen38-vision
       Qwen3.8 27B F16 vision projector from $DENSE_REPO.
       It is downloaded as $DENSE_VISION_FILE and does not update ./q36moe.gguf.

Options:
  --token TOKEN  Hugging Face token. Otherwise HF_TOKEN or the local HF
                 token cache is used if present. The main repository is
                 public, so a token is usually not required.

Environment:
  Q36_GGUF_DIR   Directory used for downloaded GGUF files.
                 Default: ./gguf

After a main-model download the script updates the legacy link:
  ./q36moe.gguf -> <download directory>/<selected model>

Every file is checked against its expected size and SHA-256 before use,
including existing downloads. A mismatch stops the script and keeps the file
for inspection. Move it aside before retrying. Requires sha256sum or shasum.

With the default output directory, q2 matches the compiled model path:
  ./q36 -p "Hello"
  ./q36-server --ctx 32768

Enable the embedded MTP head:
  ./q36 --mtp -p "Hello"

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
MODEL_EXTRA_FILE=

case "$MODEL" in
    q2) MODEL_FILE=$Q2_FILE ;;
    q4) MODEL_FILE=$Q4_FILE ;;
    27b) MODEL_REPO=$DENSE_REPO; MODEL_FILE=$DENSE_FILE ;;
    27b-q6) MODEL_REPO=$DENSE_REPO; MODEL_FILE=$DENSE_Q6_FILE ;;
    27b-iq3-xxs) MODEL_REPO=$DENSE_SMALL_REPO; MODEL_FILE=$DENSE_SMALL_FILE ;;
    flash-next|qwen38-flash-next)
        MODEL_REPO=$FLASH_REPO
        MODEL_FILE=$FLASH_FILE
        MODEL_EXTRA_FILE=$FLASH_NGRAM_FILE
        ;;
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

if ! command -v sha256sum >/dev/null 2>&1 && ! command -v shasum >/dev/null 2>&1; then
    echo "Download verification requires sha256sum or shasum." >&2
    exit 1
fi

# Sizes and SHA-256 hashes from the Hugging Face LFS metadata.
artifact_identity() {
    case "$1" in
        "$Q2_FILE")
            expected_bytes=12070095168
            expected_sha=4dab0f76b675cc1215b7ee599e85d2a5bbc97a20bd6fca18a8f35e343d90c013
            ;;
        "$Q4_FILE")
            expected_bytes=20458702976
            expected_sha=17597eb07526022e9f0c73730174cad3daf5ae53b2ce4a64ddfe4461323dc5fe
            ;;
        "$DENSE_FILE")
            expected_bytes=12040883104
            expected_sha=d847e2c1e4aa276e4b7b8e9ad7628050e61e165d49ab995407bc36677a6f3864
            ;;
        "$DENSE_Q6_FILE")
            expected_bytes=21980441504
            expected_sha=6cac55f876015d6d4b94b1656d23f79f424ac9b583652aedf0633dba116576a8
            ;;
        "$DENSE_SMALL_FILE")
            expected_bytes=10442827840
            expected_sha=63f29a2189a6b4cc31f81e093d3856ad293a5583114439f41ae1ed7af4093262
            ;;
        "$FLASH_FILE")
            expected_bytes=21986609760
            expected_sha=f519acc9e3c6d570299224f589b5d1b0a7868698ea1736809480f87dbb07a1e9
            ;;
        "$FLASH_NGRAM_FILE")
            expected_bytes=28800138432
            expected_sha=316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113
            ;;
        "$VISION_FILE")
            expected_bytes=899283680
            expected_sha=8971ee4f331ff0a4c609374f32984b3d4e6dc086c0aa35f1d637fad1829e887f
            ;;
        "$DENSE_VISION_FILE")
            expected_bytes=927607488
            expected_sha=cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e
            ;;
        *)
            echo "Missing artifact identity: $1" >&2
            exit 1
            ;;
    esac
}

verify_download() {
    artifact_identity "$1"
    if [ "$(wc -c < "$2")" -ne "$expected_bytes" ]; then
        echo "Incorrect file size: $2. Move the file aside and retry." >&2
        exit 1
    fi
    echo "Verifying SHA-256: $2"
    if command -v sha256sum >/dev/null 2>&1; then
        actual_sha=$(sha256sum < "$2")
    else
        actual_sha=$(shasum -a 256 < "$2")
    fi
    actual_sha=${actual_sha%% *}
    if [ "$actual_sha" != "$expected_sha" ]; then
        echo "Checksum mismatch: $2. Move the file aside and retry; it was not accepted." >&2
        exit 1
    fi
}

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

    if [ -e "$out" ]; then
        verify_download "$file" "$out"
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

    verify_download "$file" "$part"
    mv "$part" "$out"
}

download_one "$MODEL_REPO" "$MODEL_REMOTE" "$MODEL_FILE"
if [ -n "$MODEL_EXTRA_FILE" ]; then
    download_one "$MODEL_REPO" "$MODEL_EXTRA_FILE" "$MODEL_EXTRA_FILE"
fi

if [ "$MODEL" = "qwen36-vision" ]; then
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
    if [ "$MODEL" = "q2" ] || [ "$MODEL" = "27b-iq3-xxs" ]; then
        echo "Enable the embedded MTP head with:"
        echo "  ./q36 -m $OUT_DIR/$MODEL_FILE --mtp"
    elif [ -n "$MODEL_EXTRA_FILE" ]; then
        echo "Both GGUF shards are required. Run using the first shard:"
        echo "  ./q36 -m \"$OUT_DIR/$MODEL_FILE\" --vulkan --ssd-streaming"
    fi
fi

echo
echo "Done."
