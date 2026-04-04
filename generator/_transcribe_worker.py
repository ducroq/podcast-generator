#!/usr/bin/env python3
"""Worker script for transcription — called by validate_tts.py via subprocess.

Separated to avoid loading the Whisper model in the main process and to
prevent code injection (no f-string interpolation into source code).
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--audio", required=True)
parser.add_argument("--model", default="base")
parser.add_argument("--language", default="en")
args = parser.parse_args()

from faster_whisper import WhisperModel

device = "cuda"
try:
    import torch
    if not torch.cuda.is_available():
        device = "cpu"
except ImportError:
    device = "cpu"

compute_type = "float16" if device == "cuda" else "int8"
model = WhisperModel(args.model, device=device, compute_type=compute_type)
segments, _ = model.transcribe(args.audio, language=args.language)
print(" ".join(s.text.strip() for s in segments))
