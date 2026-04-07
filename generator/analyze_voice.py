#!/usr/bin/env python3
"""
Voice register analysis: F0 (fundamental frequency) and spectral centroid.

Use when assigning voices to a podcast cast to ensure speakers are
distinguishable by register. Recommended minimum separation: ~80-100 Hz F0.

Pipeline position: before generation — during voice casting.

Usage:
    python generator/analyze_voice.py voices/alex.mp3 voices/lisa.mp3 voices/zara.mp3
    python generator/analyze_voice.py voices/*.mp3 --json
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False


def load_audio_mono(path, target_sr=16000):
    """Load audio as mono at target sample rate using ffmpeg."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(path),
            "-ar", str(target_sr), "-ac", "1", "-f", "wav", tmp.name,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to load {path}")
        audio, sr = sf.read(tmp.name, dtype="float32")
        return audio, sr
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def estimate_f0(audio, sr, fmin=50, fmax=500):
    """Estimate fundamental frequency using autocorrelation.

    Returns median F0 in Hz over voiced frames, or None if no voiced
    frames detected.
    """
    frame_size = int(sr * 0.03)   # 30ms frames
    hop = int(sr * 0.01)          # 10ms hop
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)

    f0_values = []

    for start in range(0, len(audio) - frame_size, hop):
        frame = audio[start:start + frame_size]

        # Skip silent frames
        if np.max(np.abs(frame)) < 0.01:
            continue

        # Autocorrelation
        frame = frame - np.mean(frame)
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2:]  # positive lags only

        # Normalize
        if corr[0] > 0:
            corr = corr / corr[0]

        # Find peak in valid lag range
        if max_lag > len(corr):
            continue
        search = corr[min_lag:max_lag]
        if len(search) == 0:
            continue

        peak_idx = np.argmax(search)
        peak_val = search[peak_idx]

        # Voicing threshold
        if peak_val > 0.3:
            lag = min_lag + peak_idx
            if lag == 0:
                continue
            f0 = sr / lag
            f0_values.append(f0)

    if not f0_values:
        return None
    return float(np.median(f0_values))


def compute_spectral_centroid(audio, sr):
    """Compute mean spectral centroid in Hz.

    The spectral centroid indicates the "brightness" of the voice —
    higher values mean a brighter, more forward voice.
    """
    frame_size = 2048
    hop = 512
    centroids = []

    for start in range(0, len(audio) - frame_size, hop):
        frame = audio[start:start + frame_size]

        # Skip silent frames
        if np.max(np.abs(frame)) < 0.01:
            continue

        spectrum = np.abs(np.fft.rfft(frame * np.hanning(frame_size)))
        freqs = np.fft.rfftfreq(frame_size, 1 / sr)

        total = np.sum(spectrum)
        if total > 0:
            centroid = np.sum(freqs * spectrum) / total
            centroids.append(centroid)

    if not centroids:
        return None
    return float(np.mean(centroids))


def analyze_voice(audio_path):
    """Analyze a voice reference file.

    Returns dict with F0, spectral centroid, and duration.
    """
    if not HAS_AUDIO:
        sys.exit("Requires numpy and soundfile: pip install numpy soundfile")

    audio, sr = load_audio_mono(str(audio_path))
    duration = len(audio) / sr

    f0 = estimate_f0(audio, sr)
    centroid = compute_spectral_centroid(audio, sr)

    return {
        "file": str(Path(audio_path).name),
        "duration": round(duration, 1),
        "f0_hz": round(f0, 1) if f0 else None,
        "centroid_hz": round(centroid, 1) if centroid else None,
    }


def check_separation(results, min_f0_gap=80):
    """Check if voices have sufficient F0 separation.

    Returns list of warnings for voice pairs that are too close.
    """
    warnings = []
    voiced = [r for r in results if r["f0_hz"] is not None]
    voiced.sort(key=lambda r: r["f0_hz"])

    for i in range(len(voiced) - 1):
        gap = voiced[i + 1]["f0_hz"] - voiced[i]["f0_hz"]
        if gap < min_f0_gap:
            warnings.append(
                f"{voiced[i]['file']} ({voiced[i]['f0_hz']} Hz) and "
                f"{voiced[i + 1]['file']} ({voiced[i + 1]['f0_hz']} Hz): "
                f"only {gap:.0f} Hz apart (recommended ≥{min_f0_gap} Hz)"
            )
    return warnings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Analyze voice register (F0, spectral centroid) for cast assignment.",
    )
    p.add_argument("files", nargs="+",
                    help="Voice reference audio files to analyze")
    p.add_argument("--min-gap", type=float, default=80,
                    help="Minimum F0 separation in Hz (default: 80)")
    p.add_argument("--json", action="store_true",
                    help="Output as JSON")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    results = []
    log = sys.stderr if args.json else sys.stdout
    for path in args.files:
        path = Path(path)
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping", file=log)
            continue
        print(f"  Analyzing {path.name}...", end=" ", flush=True, file=log)
        result = analyze_voice(path)
        results.append(result)
        if result["f0_hz"]:
            print(f"F0={result['f0_hz']} Hz, centroid={result['centroid_hz']} Hz",
                  file=log)
        else:
            print("(no voiced frames detected)", file=log)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    # Summary table
    print(f"\n{'Voice':<25} {'F0 (Hz)':>8} {'Centroid (Hz)':>14} {'Register':>10}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x["f0_hz"] or 0):
        f0_str = f"{r['f0_hz']:.0f}" if r["f0_hz"] else "?"
        cent_str = f"{r['centroid_hz']:.0f}" if r["centroid_hz"] else "?"
        register = ""
        if r["f0_hz"]:
            if r["f0_hz"] < 165:
                register = "Low"
            elif r["f0_hz"] < 255:
                register = "Mid"
            else:
                register = "High"
        print(f"{r['file']:<25} {f0_str:>8} {cent_str:>14} {register:>10}")

    # Check separation
    warnings = check_separation(results, args.min_gap)
    if warnings:
        print(f"\nWARNINGS:")
        for w in warnings:
            print(f"  ⚠ {w}")
    else:
        print(f"\nAll voices have ≥{args.min_gap} Hz F0 separation.")


if __name__ == "__main__":
    main()
