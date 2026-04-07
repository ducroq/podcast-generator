#!/usr/bin/env python3
"""
Assemble multi-speaker intro lines into a single voiceover track.

Concatenates individually generated intro WAV files with per-speaker
pause durations between them. The output is a single WAV suitable
for mixing under a music bed.

Pipeline position: after TTS generation of intro lines, before mix_episode.

Usage:
    python generator/assemble_intro.py intro_dir/ --lines intro_lines.txt -o intro.wav
    python generator/assemble_intro.py intro_dir/ --lines intro_lines.txt \
        --speaker-pauses '{"morgan": 0.4, "zara": 0.4}' -o intro.wav
"""

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False


def parse_intro_lines(path):
    """Parse intro lines from a text file.

    Format: Speaker: [emotion] Text
    Returns list of dicts: [{index, speaker, text}]
    """
    entries = []
    with open(path, encoding="utf-8") as fh:
        for i, raw in enumerate(fh):
            stripped = raw.strip()
            if not stripped:
                continue
            match = re.match(r"(.+?):\s*(?:\[([^\]]*)\]\s*)?(.*)", stripped)
            if match:
                speaker = match.group(1).strip().lower()
                text = match.group(3).strip()
                if text:
                    entries.append({"index": i, "speaker": speaker, "text": text})
    return entries


def assemble_intro(wav_dir, lines, output_path, sr=24000,
                   default_pause=0.15, speaker_pauses=None):
    """Concatenate intro line WAVs into a single voiceover track.

    Args:
        wav_dir: directory containing intro_NNN_speaker.wav files
        lines: list of dicts from parse_intro_lines
        output_path: output WAV path
        sr: target sample rate
        default_pause: default pause between lines (seconds)
        speaker_pauses: optional dict of speaker → pause duration overrides

    Returns total duration in seconds.
    """
    if not HAS_AUDIO:
        raise RuntimeError("assemble_intro requires numpy and soundfile: "
                           "pip install numpy soundfile")

    wav_dir = Path(wav_dir)
    speaker_pauses = speaker_pauses or {}
    parts = []

    for line in lines:
        speaker = line["speaker"]
        idx = line["index"]
        filename = f"intro_{idx:03d}_{speaker}.wav"
        wav_path = wav_dir / filename

        if not wav_path.exists():
            print(f"  WARNING: Missing {filename}")
            continue

        audio, file_sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            new_len = int(len(audio) * sr / file_sr)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)

        parts.append(audio)

        # Add pause after this line
        pause = speaker_pauses.get(speaker, default_pause)
        if pause > 0:
            parts.append(np.zeros(int(sr * pause), dtype=np.float32))

    if not parts:
        print("  WARNING: No intro lines assembled")
        return 0.0

    full = np.concatenate(parts)
    sf.write(str(output_path), full, sr)
    duration = len(full) / sr
    print(f"  Intro assembled: {duration:.1f}s -> {output_path}")
    return duration


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Assemble intro line WAVs into a single voiceover track.",
    )
    p.add_argument("intro_dir",
                    help="Directory containing intro WAV files")
    p.add_argument("--lines", required=True,
                    help="Intro lines text file (speaker: [emotion] text)")
    p.add_argument("-o", "--output", default="intro_assembled.wav",
                    help="Output WAV path (default: intro_assembled.wav)")
    p.add_argument("--sr", type=int, default=24000,
                    help="Target sample rate (default: 24000)")
    p.add_argument("--default-pause", type=float, default=0.15,
                    help="Default pause between lines (default: 0.15s)")
    p.add_argument("--speaker-pauses", type=str, default=None,
                    help='Per-speaker pause durations as JSON (e.g. \'{"morgan": 0.4}\')')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    lines = parse_intro_lines(args.lines)
    print(f"Parsed {len(lines)} intro lines from {args.lines}")

    speaker_pauses = None
    if args.speaker_pauses:
        speaker_pauses = json.loads(args.speaker_pauses)

    duration = assemble_intro(
        args.intro_dir, lines, args.output,
        sr=args.sr, default_pause=args.default_pause,
        speaker_pauses=speaker_pauses,
    )
    print(f"Total: {duration:.1f}s")


if __name__ == "__main__":
    main()
