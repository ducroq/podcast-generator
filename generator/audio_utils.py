"""Shared audio utility functions for ffmpeg-based processing."""

import re
import subprocess


def detect_silences(input_path, noise_db=-35, min_duration=0.3):
    """Detect silence regions using ffmpeg silencedetect."""
    noise_db = float(noise_db)
    min_duration = float(min_duration)
    if not (-100 <= noise_db <= 0):
        raise ValueError(f"noise_db must be between -100 and 0, got {noise_db}")
    if not (0.01 <= min_duration <= 10.0):
        raise ValueError(f"min_duration must be between 0.01 and 10.0, got {min_duration}")
    cmd = [
        'ffmpeg', '-i', str(input_path),
        '-af', f'silencedetect=noise={noise_db}dB:d={min_duration}',
        '-f', 'null', '-'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # ffmpeg returns 0 even on silencedetect, but check for total failure
    if result.returncode != 0 and 'silencedetect' not in result.stderr:
        raise RuntimeError(f"ffmpeg silencedetect failed on {input_path}: {result.stderr[:200]}")

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
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {input_path}: {result.stderr[:200]}")
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise RuntimeError(f"ffprobe returned unexpected output for {input_path}: {result.stdout[:100]}")


def get_sample_rate(input_path):
    """Get audio sample rate."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=sample_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(input_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {input_path}: {result.stderr[:200]}")
    try:
        return int(result.stdout.strip())
    except ValueError:
        raise RuntimeError(f"ffprobe returned unexpected output for {input_path}: {result.stdout[:100]}")
