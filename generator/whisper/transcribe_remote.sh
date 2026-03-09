#!/usr/bin/env bash
# Transcribe an audio file using Whisper on gpu-server.
#
# Usage:
#   ./transcribe_remote.sh podcast.mp3
#   ./transcribe_remote.sh podcast.mp3 --language en
#   ./transcribe_remote.sh podcast.mp3 --model medium
#
# The transcript is saved next to the input file as .txt

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <audio-file> [--language CODE] [--model SIZE]"
    exit 1
fi

AUDIO_FILE="$1"
shift

if [ ! -f "$AUDIO_FILE" ]; then
    echo "Error: $AUDIO_FILE not found"
    exit 1
fi

BASENAME="$(basename "$AUDIO_FILE")"
STEM="${BASENAME%.*}"
LOCAL_DIR="$(dirname "$(realpath "$AUDIO_FILE")")"
REMOTE_DIR="~/transcribe_tmp"
EXTRA_ARGS="$*"

echo "==> Uploading $BASENAME to gpu-server..."
ssh gpu-server "mkdir -p $REMOTE_DIR"
scp "$AUDIO_FILE" "gpu-server:$REMOTE_DIR/$BASENAME"

echo "==> Transcribing on gpu-server..."
ssh gpu-server "source ~/podcast-generator/vox-env/bin/activate && \
    python ~/podcast-generator/transcribe.py \
    $REMOTE_DIR/$BASENAME \
    --output $REMOTE_DIR/$STEM.txt \
    $EXTRA_ARGS"

echo "==> Downloading transcript..."
scp "gpu-server:$REMOTE_DIR/$STEM.txt" "$LOCAL_DIR/$STEM.txt"

echo "==> Cleaning up remote files..."
ssh gpu-server "rm -f $REMOTE_DIR/$BASENAME $REMOTE_DIR/$STEM.txt"

echo "==> Done! Transcript saved to: $LOCAL_DIR/$STEM.txt"
