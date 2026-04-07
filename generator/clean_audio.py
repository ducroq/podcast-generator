#!/usr/bin/env python3
"""
Post-TTS audio cleanup: click detection/repair, silence trimming, fade-in/out.

Runs on individual WAV files after TTS generation, before mixing. Operates
in-place using numpy/soundfile (not ffmpeg). Designed for gpu-server but
works anywhere numpy + soundfile are installed.

Pipeline position: after TTS generation, before validate_tts / mix_episode.

Usage:
    python generator/clean_audio.py output_dir/           # clean all WAVs
    python generator/clean_audio.py single_file.wav       # clean one file
    python generator/clean_audio.py output_dir/ --dry-run # preview only
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
except ImportError:
    sys.exit("Requires numpy and soundfile: pip install numpy soundfile")


# ---------------------------------------------------------------------------
# Click detection & repair
# ---------------------------------------------------------------------------

def detect_clicks(audio, sr, threshold=0.15):
    """Detect click artifacts (large sample-to-sample jumps).

    Returns list of dicts: [{pos_ms, severity, samples}].
    Groups adjacent click indices within 5ms.
    """
    diff = np.abs(np.diff(audio))
    click_indices = np.where(diff > threshold)[0]

    if len(click_indices) == 0:
        return []

    # Group adjacent indices (within 5ms)
    window = max(1, int(sr * 0.005))
    groups = []
    current_group = [click_indices[0]]
    for idx in click_indices[1:]:
        if idx - current_group[-1] < window:
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]
    groups.append(current_group)

    clicks = []
    for group in groups:
        pos_ms = (group[0] / sr) * 1000
        severity = float(np.max(diff[group]))
        clicks.append({
            "pos_ms": round(pos_ms, 1),
            "severity": round(severity, 3),
            "samples": len(group),
        })
    return clicks


def repair_clicks(audio, sr, threshold=0.15):
    """Repair click artifacts by linear interpolation over them.

    Blends 3 samples before and after each click point.
    Returns (repaired_audio, count_repaired).
    """
    audio = audio.copy()
    diff = np.abs(np.diff(audio))
    click_indices = np.where(diff > threshold)[0]
    repaired = 0

    for idx in click_indices:
        start = max(0, idx - 3)
        end = min(len(audio) - 1, idx + 4)
        audio[start:end] = np.linspace(
            audio[start], audio[min(end, len(audio) - 1)],
            end - start, dtype=np.float32,
        )
        repaired += 1

    return audio, repaired


# ---------------------------------------------------------------------------
# Silence trimming (numpy-based, not ffmpeg)
# ---------------------------------------------------------------------------

def trim_leading_silence(audio, sr, threshold_db=-35):
    """Trim leading silence, keeping 5ms before the first sound."""
    threshold = 10 ** (threshold_db / 20)
    window = max(1, int(sr * 0.01))
    abs_audio = np.abs(audio)
    for i in range(0, len(audio) - window, window):
        if np.max(abs_audio[i:i + window]) > threshold:
            start = max(0, i - int(sr * 0.005))
            return audio[start:]
    return audio


def trim_trailing_silence(audio, sr, threshold_db=-35):
    """Trim trailing silence, keeping 20ms after the last sound."""
    threshold = 10 ** (threshold_db / 20)
    window = max(1, int(sr * 0.01))
    abs_audio = np.abs(audio)
    for i in range(len(audio) - window, 0, -window):
        if np.max(abs_audio[i:i + window]) > threshold:
            end = min(len(audio), i + window + int(sr * 0.02))
            return audio[:end]
    return audio


# ---------------------------------------------------------------------------
# Fade-in / fade-out
# ---------------------------------------------------------------------------

def apply_fades(audio, sr, fade_ms=8):
    """Apply linear fade-in/fade-out for clean zero crossings."""
    audio = audio.copy()
    fade_samples = int(sr * fade_ms / 1000)
    if len(audio) > fade_samples * 2:
        audio[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        audio[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    return audio


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def clean_file(wav_path, threshold=0.15, fade_ms=8, trim=True, dry_run=False):
    """Clean a single WAV file: detect/repair clicks, trim silence, apply fades.

    Modifies the file in-place (unless dry_run=True).
    Returns a report dict with details of what was done.
    """
    wav_path = Path(wav_path)
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    original_len = len(audio)
    report = {
        "file": str(wav_path.name),
        "original_duration": round(original_len / sr, 3),
        "clicks_detected": 0,
        "clicks_repaired": 0,
        "trimmed_ms": 0.0,
        "fades_applied": False,
    }

    # 1. Click detection and repair
    clicks = detect_clicks(audio, sr, threshold)
    report["clicks_detected"] = len(clicks)
    if clicks:
        report["click_details"] = clicks
        if not dry_run:
            audio, n_repaired = repair_clicks(audio, sr, threshold)
            report["clicks_repaired"] = n_repaired

    # 2. Trim silence
    if trim and not dry_run:
        audio = trim_leading_silence(audio, sr)
        audio = trim_trailing_silence(audio, sr)

    # 3. Apply fades
    if not dry_run:
        audio = apply_fades(audio, sr, fade_ms)
        report["fades_applied"] = True

    report["cleaned_duration"] = round(len(audio) / sr, 3)
    report["trimmed_ms"] = round((original_len - len(audio)) / sr * 1000, 1)

    # Write back
    if not dry_run:
        sf.write(str(wav_path), audio, sr)

    return report


def clean_directory(directory, threshold=0.15, fade_ms=8, trim=True,
                    dry_run=False):
    """Clean all WAV files in a directory.

    Returns list of report dicts.
    """
    directory = Path(directory)
    wav_files = sorted(directory.glob("*.wav"))

    if not wav_files:
        print(f"No .wav files found in {directory}")
        return []

    results = []
    total_clicks = 0
    total_trimmed = 0.0

    for wav_path in wav_files:
        report = clean_file(wav_path, threshold, fade_ms, trim, dry_run)
        results.append(report)
        total_clicks += report["clicks_repaired"]
        total_trimmed += report["trimmed_ms"]
        status = "DRY" if dry_run else "OK"
        clicks_str = f" clicks={report['clicks_detected']}" if report["clicks_detected"] else ""
        trim_str = f" trimmed={report['trimmed_ms']:.0f}ms" if report["trimmed_ms"] else ""
        print(f"  [{status}] {report['file']}{clicks_str}{trim_str}")

    print(f"\nSummary: {len(results)} files, "
          f"{total_clicks} clicks repaired, "
          f"{total_trimmed:.0f}ms trimmed")

    # Write report
    if not dry_run:
        report_path = directory / "clean_report.json"
        with open(report_path, "w") as f:
            json.dump({
                "total_files": len(results),
                "total_clicks_repaired": total_clicks,
                "total_trimmed_ms": round(total_trimmed),
                "results": results,
            }, f, indent=2)
        print(f"Report: {report_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Clean TTS audio: click repair, silence trim, fades.",
    )
    p.add_argument("input",
                    help="WAV file or directory of WAV files")
    p.add_argument("--threshold", type=float, default=0.15,
                    help="Click detection threshold (default: 0.15)")
    p.add_argument("--fade-ms", type=int, default=8,
                    help="Fade-in/out duration in ms (default: 8)")
    p.add_argument("--no-trim", action="store_true",
                    help="Skip silence trimming")
    p.add_argument("--dry-run", action="store_true",
                    help="Preview only — don't modify files")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)

    if input_path.is_dir():
        clean_directory(input_path, args.threshold, args.fade_ms,
                        trim=not args.no_trim, dry_run=args.dry_run)
    elif input_path.is_file():
        report = clean_file(input_path, args.threshold, args.fade_ms,
                            trim=not args.no_trim, dry_run=args.dry_run)
        print(json.dumps(report, indent=2))
    else:
        sys.exit(f"Not found: {input_path}")


if __name__ == "__main__":
    main()
