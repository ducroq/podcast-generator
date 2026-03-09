#!/usr/bin/env python3
"""Transcribe audio using faster-whisper on GPU.

Usage (on gpu-server):
    python transcribe.py input.mp3
    python transcribe.py input.mp3 --model medium --language en
    python transcribe.py input.mp3 --output /path/to/output.txt
"""

import argparse
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel


def transcribe(audio_path: str, model_size: str = "large-v3", language: str = None) -> str:
    print(f"Loading model: {model_size}", file=sys.stderr)
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    print(f"Transcribing: {audio_path}", file=sys.stderr)
    start = time.time()

    segments, info = model.transcribe(audio_path, language=language, beam_size=5)

    print(f"Detected language: {info.language} (probability {info.language_probability:.2f})", file=sys.stderr)

    lines = []
    for segment in segments:
        lines.append(segment.text.strip())

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f}s", file=sys.stderr)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with faster-whisper")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--model", default="large-v3", help="Model size (default: large-v3)")
    parser.add_argument("--language", default=None, help="Language code (e.g. en, nl). Auto-detect if omitted.")
    parser.add_argument("--output", default=None, help="Output text file path. Prints to stdout if omitted.")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: {audio_path} not found", file=sys.stderr)
        sys.exit(1)

    text = transcribe(str(audio_path), model_size=args.model, language=args.language)

    if args.output:
        out = Path(args.output)
        out.write_text(text, encoding="utf-8")
        print(f"Saved to {out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
