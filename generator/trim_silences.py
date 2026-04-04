#!/usr/bin/env python3
"""
Post-process ElevenLabs text_to_dialogue output to shorten excessive pauses.
Loudness normalization is off by default (must be the final pipeline step).

Usage:
    python trim_silences.py input.mp3 [output.mp3] [--max-pause 0.35] [--loudnorm]
"""

import argparse
import subprocess
import sys
from pathlib import Path

from audio_utils import detect_silences, get_duration


def trim_silences(input_path, output_path, max_pause=0.35, noise_db=-35,
                  min_duration=0.3, loudnorm=False):
    """Shorten silences to max_pause duration and optionally normalize loudness."""
    silences = detect_silences(input_path, noise_db, min_duration)

    if not silences:
        print("No silences detected — nothing to trim.")
        return False

    total_duration = get_duration(input_path)

    # Build segments: audio parts + shortened silences
    segments = []
    current_pos = 0.0

    for silence in silences:
        # Audio before this silence
        if silence['start'] > current_pos:
            segments.append((current_pos, silence['start']))

        # Shortened silence (keep max_pause, clamped to not eat adjacent speech)
        if silence['duration'] > max_pause:
            keep_start = max(silence['start'], silence['end'] - max_pause)
            segments.append((keep_start, keep_start + max_pause))
        else:
            # Keep short silences as-is
            segments.append((silence['start'], silence['end']))

        current_pos = silence['end']

    # Remaining audio after last silence
    if current_pos < total_duration:
        segments.append((current_pos, total_duration))

    # Build ffmpeg filter_complex
    filter_parts = []
    concat_inputs = []
    for i, (start, end) in enumerate(segments):
        filter_parts.append(f'[0:a]atrim={start:.6f}:{end:.6f},asetpts=N/SR/TB[s{i}]')
        concat_inputs.append(f'[s{i}]')

    concat = ''.join(concat_inputs) + f'concat=n={len(segments)}:v=0:a=1[trimmed]'

    if loudnorm:
        full_filter = ';'.join(filter_parts) + ';' + concat + \
                      ';[trimmed]loudnorm=I=-16:TP=-1.5:LRA=11[out]'
        out_label = '[out]'
    else:
        full_filter = ';'.join(filter_parts) + ';' + concat
        out_label = '[trimmed]'

    cmd = [
        'ffmpeg', '-y', '-i', str(input_path),
        '-filter_complex', full_filter,
        '-map', out_label,
        '-codec:a', 'libmp3lame', '-b:a', '192k',
        str(output_path)
    ]

    # Print stats
    original_silence = sum(s['duration'] for s in silences)
    trimmed_silence = sum(min(s['duration'], max_pause) for s in silences)
    saved = original_silence - trimmed_silence

    print(f"Silences: {len(silences)} regions")
    print(f"Original silence: {original_silence:.1f}s / {total_duration:.1f}s total")
    print(f"Trimmed to: {trimmed_silence:.1f}s (saved {saved:.1f}s)")
    print(f"Max pause: {max_pause * 1000:.0f}ms")
    if loudnorm:
        print(f"Loudness: -16 LUFS (podcast standard)")
    print(f"Processing...")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: ffmpeg failed")
        print(result.stderr[-500:])
        return False

    in_size = Path(input_path).stat().st_size
    out_size = Path(output_path).stat().st_size
    new_duration = get_duration(output_path)
    print(f"Result: {total_duration:.1f}s -> {new_duration:.1f}s ({in_size/1024:.0f}KB -> {out_size/1024:.0f}KB)")
    print(f"Done: {output_path}")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Shorten excessive pauses in TTS audio output"
    )
    parser.add_argument('input', help='Input audio file (mp3/wav)')
    parser.add_argument('output', nargs='?', help='Output file (default: input_processed.mp3)')
    parser.add_argument('--max-pause', type=float, default=0.35,
                        help='Maximum pause duration in seconds (default: 0.35)')
    parser.add_argument('--loudnorm', action='store_true',
                        help='Apply loudness normalization (-16 LUFS). Only use as final step after mixing.')

    args = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        sys.exit(1)

    output_file = Path(args.output) if args.output else \
        input_file.with_stem(input_file.stem + '_processed')

    trim_silences(input_file, output_file, max_pause=args.max_pause,
                  loudnorm=args.loudnorm)
