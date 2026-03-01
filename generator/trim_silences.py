#!/usr/bin/env python3
"""
Post-process ElevenLabs text_to_dialogue output to shorten excessive pauses.
Optionally applies loudness normalization to podcast standard (-16 LUFS).

Usage:
    python trim_silences.py input.mp3 [output.mp3] [--max-pause 0.35] [--no-loudnorm]
"""

import subprocess
import re
import sys
from pathlib import Path


def detect_silences(input_path, noise_db=-35, min_duration=0.3):
    """Detect silence regions using ffmpeg silencedetect."""
    cmd = [
        'ffmpeg', '-i', str(input_path),
        '-af', f'silencedetect=noise={noise_db}dB:d={min_duration}',
        '-f', 'null', '-'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    silences = []
    starts = []
    for line in result.stderr.split('\n'):
        start_match = re.search(r'silence_start:\s*([\d.]+)', line)
        end_match = re.search(r'silence_end:\s*([\d.]+)\s*\|\s*silence_duration:\s*([\d.]+)', line)
        if start_match:
            starts.append(float(start_match.group(1)))
        if end_match and starts:
            silences.append({
                'start': starts.pop(0),
                'end': float(end_match.group(1)),
                'duration': float(end_match.group(2))
            })
    return silences


def get_duration(input_path):
    """Get audio duration in seconds."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(input_path)
    ]
    return float(subprocess.run(cmd, capture_output=True, text=True).stdout.strip())


def trim_silences(input_path, output_path, max_pause=0.35, noise_db=-35,
                  min_duration=0.3, loudnorm=True):
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

        # Shortened silence (keep max_pause, centered on original)
        if silence['duration'] > max_pause:
            center = (silence['start'] + silence['end']) / 2
            segments.append((center - max_pause / 2, center + max_pause / 2))
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
    if len(sys.argv) < 2:
        print("Usage: python trim_silences.py input.mp3 [output.mp3] [--max-pause 0.35] [--no-loudnorm]")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    max_pause = 0.35
    loudnorm = True
    output_file = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--max-pause' and i + 1 < len(sys.argv):
            max_pause = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--no-loudnorm':
            loudnorm = False
            i += 1
        elif not output_file:
            output_file = Path(sys.argv[i])
            i += 1
        else:
            i += 1

    if not output_file:
        output_file = input_file.with_stem(input_file.stem + '_processed')

    trim_silences(input_file, output_file, max_pause=max_pause, loudnorm=loudnorm)
