#!/usr/bin/env python3
"""
Podcast mastering chain using Spotify's Pedalboard.

Applies a professional mastering chain to podcast audio:
  1. Highpass (80Hz) — remove rumble
  2. Compressor — even out dynamics
  3. NoiseGate — clean up quiet sections
  4. Limiter — prevent clipping
  5. Gain to -16 LUFS (ITU-R BS.1770)

Usage:
    python generator/master.py input.mp3 -o mastered.mp3
    python generator/master.py input.mp3 -o mastered.mp3 --target-lufs -14
    python generator/master.py input.mp3 --analyze  # just measure, don't process
"""

import argparse
import sys
from pathlib import Path

try:
    import numpy as np
    from pedalboard import (
        Pedalboard, Compressor, HighpassFilter, LowShelfFilter,
        HighShelfFilter, Limiter, NoiseGate, Gain,
    )
    from pedalboard.io import AudioFile
except ImportError:
    sys.exit("Missing dependencies: pip install pedalboard numpy")

try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None


# ---------------------------------------------------------------------------
# Loudness measurement
# ---------------------------------------------------------------------------

def measure_lufs(audio: np.ndarray, sample_rate: int) -> float:
    """Measure integrated loudness (LUFS) using pyloudnorm or peak-based fallback."""
    if pyln is not None:
        meter = pyln.Meter(sample_rate)
        # pyloudnorm expects (samples, channels)
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        elif audio.shape[0] < audio.shape[1]:
            audio = audio.T  # pedalboard uses (channels, samples)
        return meter.integrated_loudness(audio)
    else:
        # Rough RMS-based estimate when pyloudnorm is not available
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-10:
            return -70.0
        return 20 * np.log10(rms) - 0.691  # approximate LUFS from RMS


def compute_gain_db(current_lufs: float, target_lufs: float) -> float:
    """Compute gain in dB to reach target LUFS."""
    return target_lufs - current_lufs


# ---------------------------------------------------------------------------
# Mastering chain
# ---------------------------------------------------------------------------

def build_master_chain(target_lufs: float = -16.0,
                       highpass_hz: float = 80.0,
                       compress: bool = True,
                       noise_gate: bool = True) -> Pedalboard:
    """Build the mastering effects chain (without final gain — applied separately)."""
    effects = []

    # 1. Highpass — remove rumble and low-frequency noise
    effects.append(HighpassFilter(cutoff_frequency_hz=highpass_hz))

    # 2. Gentle low-shelf cut — reduce mud in 200-400Hz range
    effects.append(LowShelfFilter(cutoff_frequency_hz=300.0, gain_db=-2.0))

    # 3. Presence boost — add clarity in speech range
    effects.append(HighShelfFilter(cutoff_frequency_hz=3000.0, gain_db=1.5))

    # 4. Compressor — even out dynamics
    if compress:
        effects.append(Compressor(
            threshold_db=-18.0,
            ratio=3.0,
            attack_ms=5.0,
            release_ms=150.0,
        ))

    # 5. Noise gate — clean up quiet sections
    if noise_gate:
        effects.append(NoiseGate(
            threshold_db=-40.0,
            attack_ms=1.0,
            release_ms=100.0,
        ))

    # 6. Limiter — prevent clipping
    effects.append(Limiter(threshold_db=-1.0))

    return Pedalboard(effects)


def master_audio(input_path: str, output_path: str,
                 target_lufs: float = -16.0,
                 highpass_hz: float = 80.0,
                 compress: bool = True,
                 noise_gate: bool = True) -> dict:
    """Apply mastering chain and normalize to target LUFS.

    Returns dict with before/after measurements.
    """
    # Read input
    with AudioFile(str(input_path)) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate

    # Measure input
    input_lufs = measure_lufs(audio, sample_rate)

    # Apply effects chain
    board = build_master_chain(target_lufs, highpass_hz, compress, noise_gate)
    processed = board(audio, sample_rate)

    # Measure after processing (before final gain)
    post_chain_lufs = measure_lufs(processed, sample_rate)

    # Apply final gain to hit target LUFS
    gain_db = compute_gain_db(post_chain_lufs, target_lufs)
    gain_linear = 10 ** (gain_db / 20.0)
    processed = processed * gain_linear

    # Clip to prevent any overs
    processed = np.clip(processed, -1.0, 1.0)

    # Measure final
    output_lufs = measure_lufs(processed, sample_rate)

    # Write output
    with AudioFile(str(output_path), 'w', sample_rate,
                   num_channels=processed.shape[0]) as f:
        f.write(processed)

    return {
        'input_lufs': round(input_lufs, 1),
        'post_chain_lufs': round(post_chain_lufs, 1),
        'gain_applied_db': round(gain_db, 1),
        'output_lufs': round(output_lufs, 1),
        'sample_rate': sample_rate,
        'duration_s': round(audio.shape[1] / sample_rate, 1),
    }


def analyze_audio(input_path: str) -> dict:
    """Measure loudness without processing."""
    with AudioFile(str(input_path)) as f:
        audio = f.read(f.frames)
        sample_rate = f.samplerate

    lufs = measure_lufs(audio, sample_rate)
    peak_db = 20 * np.log10(max(np.abs(audio).max(), 1e-10))
    rms_db = 20 * np.log10(max(np.sqrt(np.mean(audio ** 2)), 1e-10))

    return {
        'lufs': round(lufs, 1),
        'peak_db': round(peak_db, 1),
        'rms_db': round(rms_db, 1),
        'sample_rate': sample_rate,
        'channels': audio.shape[0],
        'duration_s': round(audio.shape[1] / sample_rate, 1),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Master podcast audio using Pedalboard DSP chain.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python generator/master.py episode.mp3 -o mastered.mp3\n"
               "  python generator/master.py episode.mp3 --analyze\n"
               "  python generator/master.py episode.mp3 -o out.mp3 --target-lufs -14\n",
    )
    p.add_argument("input", help="Input audio file")
    p.add_argument("-o", "--output", help="Output file (required unless --analyze)")
    p.add_argument("--target-lufs", type=float, default=-16.0,
                    help="Target loudness in LUFS (default: -16)")
    p.add_argument("--highpass", type=float, default=80.0,
                    help="Highpass filter frequency in Hz (default: 80)")
    p.add_argument("--no-compress", action="store_true",
                    help="Skip compression")
    p.add_argument("--no-gate", action="store_true",
                    help="Skip noise gate")
    p.add_argument("--analyze", action="store_true",
                    help="Measure loudness only (no processing)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if not Path(args.input).exists():
        sys.exit(f"Not found: {args.input}")

    if args.analyze:
        info = analyze_audio(args.input)
        print(f"File: {args.input}")
        print(f"  Duration: {info['duration_s']}s")
        print(f"  Sample rate: {info['sample_rate']} Hz")
        print(f"  Channels: {info['channels']}")
        print(f"  LUFS: {info['lufs']}")
        print(f"  Peak: {info['peak_db']} dBFS")
        print(f"  RMS: {info['rms_db']} dBFS")
        return

    if not args.output:
        sys.exit("Output path required (-o). Use --analyze for measurement only.")

    print(f"Mastering: {args.input}")
    result = master_audio(
        args.input, args.output,
        target_lufs=args.target_lufs,
        highpass_hz=args.highpass,
        compress=not args.no_compress,
        noise_gate=not args.no_gate,
    )
    print(f"  Input:  {result['input_lufs']} LUFS")
    print(f"  Chain:  {result['post_chain_lufs']} LUFS")
    print(f"  Gain:   {result['gain_applied_db']:+.1f} dB")
    print(f"  Output: {result['output_lufs']} LUFS")
    print(f"  Duration: {result['duration_s']}s")
    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
