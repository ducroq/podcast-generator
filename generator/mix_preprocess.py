#!/usr/bin/env python3
"""
Pre-processing for individual WAV lines before mixing: room reverb,
per-speaker volume adjustment, RMS normalization, clip fades.

Operates on WAV files in-place using numpy/soundfile. Runs BEFORE the
ffmpeg-based mix_episode.py pipeline (concat, music bed, mastering).

Pipeline position: after clean_audio, before mix_episode.

Usage:
    python generator/mix_preprocess.py output_dir/ --manifest manifest.json
    python generator/mix_preprocess.py output_dir/ --manifest manifest.json \
        --reverb-decay 0.15 --reverb-mix 0.02 \
        --speaker-volume '{"zara": 2.5}' --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
    from scipy.signal import fftconvolve
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


# ---------------------------------------------------------------------------
# Room impulse response
# ---------------------------------------------------------------------------

def generate_room_ir(sr, decay_time=0.15):
    """Generate a synthetic room impulse response.

    Short decay (~0.15s) simulates a small studio room. Early reflections
    at 5/12/20/35ms add spatial cues. The result is barely audible at
    mix=0.02 but glues spliced lines together.
    """
    num_samples = int(sr * decay_time)
    rng = np.random.default_rng(42)
    ir = rng.standard_normal(num_samples).astype(np.float32)

    # Exponential decay
    decay = np.exp(-np.linspace(0, 6, num_samples)).astype(np.float32)
    ir *= decay

    # Early reflections
    for delay_ms, gain in [(5, 0.4), (12, 0.25), (20, 0.15), (35, 0.1)]:
        pos = int(sr * delay_ms / 1000)
        if pos < num_samples:
            ir[pos] += gain

    # Normalize
    peak = np.max(np.abs(ir))
    if peak > 0:
        ir /= peak
    return ir


# ---------------------------------------------------------------------------
# Audio processing functions
# ---------------------------------------------------------------------------

def apply_reverb(audio, ir, mix=0.02):
    """Apply convolution reverb at a wet/dry ratio.

    mix=0.02 is barely audible — just enough to make spliced lines
    feel like they were recorded in the same room.
    """
    wet = fftconvolve(audio, ir, mode="full")[:len(audio)].astype(np.float32)
    return (audio * (1 - mix) + wet * mix).astype(np.float32)


def apply_speaker_volume(audio, speaker, volume_map=None):
    """Apply per-speaker volume adjustment in dB.

    volume_map: {"zara": 2.5, "alex": 0.0, ...}
    Positive values boost, negative values cut.
    """
    if not volume_map:
        return audio
    db = volume_map.get(speaker, 0.0)
    if db == 0.0:
        return audio
    return (audio * (10 ** (db / 20))).astype(np.float32)


def rms_normalize(audio, target_rms=0.1):
    """Normalize audio to a target RMS level."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        audio = (audio * (target_rms / rms)).astype(np.float32)
    return audio


def apply_clip_fades(audio, sr, fade_ms=20):
    """Apply fade-in/fade-out at clip boundaries."""
    audio = audio.copy()
    fade_samples = int(sr * fade_ms / 1000)
    if len(audio) > fade_samples * 2:
        audio[:fade_samples] *= np.linspace(0, 1, fade_samples, dtype=np.float32)
        audio[-fade_samples:] *= np.linspace(1, 0, fade_samples, dtype=np.float32)
    return audio


# ---------------------------------------------------------------------------
# Per-line processing
# ---------------------------------------------------------------------------

def preprocess_line(wav_path, speaker, sr, room_ir=None, reverb_mix=0.02,
                    volume_map=None, target_rms=0.1, fade_ms=20):
    """Load a WAV line and apply all pre-processing.

    Returns processed numpy array. Does NOT write back — caller decides.
    """
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

    audio = rms_normalize(audio, target_rms)
    audio = apply_speaker_volume(audio, speaker, volume_map)
    if room_ir is not None:
        audio = apply_reverb(audio, room_ir, reverb_mix)
    audio = apply_clip_fades(audio, sr, fade_ms)
    return audio


# ---------------------------------------------------------------------------
# Directory-level processing
# ---------------------------------------------------------------------------

def preprocess_directory(directory, manifest, sr=24000,
                         reverb_decay=0.15, reverb_mix=0.02,
                         volume_map=None, target_rms=0.1, fade_ms=20,
                         dry_run=False):
    """Pre-process all WAV lines in a directory using manifest for speaker info.

    Manifest format: [{"file": "001_speaker.wav", "speaker": "alex", ...}, ...]
    Writes processed WAVs back in-place.

    Returns list of result dicts.
    """
    if not HAS_DEPS:
        sys.exit("Requires numpy, soundfile, scipy: "
                 "pip install numpy soundfile scipy")

    directory = Path(directory)
    room_ir = generate_room_ir(sr, reverb_decay) if reverb_mix > 0 else None

    results = []
    for entry in manifest:
        filename = entry["file"]
        speaker = entry.get("speaker", "unknown")

        # Path traversal guard
        fp = Path(filename)
        if fp.is_absolute() or '..' in fp.parts:
            results.append({"file": filename, "status": "REJECTED"})
            continue
        wav_path = (directory / fp).resolve()
        if not wav_path.is_relative_to(directory.resolve()):
            results.append({"file": filename, "status": "REJECTED"})
            continue

        if not wav_path.exists():
            print(f"  [MISSING] {filename} — file not found, skipping")
            results.append({"file": filename, "status": "MISSING"})
            continue

        if dry_run:
            results.append({
                "file": filename, "speaker": speaker, "status": "DRY",
            })
            continue

        audio = preprocess_line(
            wav_path, speaker, sr,
            room_ir=room_ir, reverb_mix=reverb_mix,
            volume_map=volume_map, target_rms=target_rms, fade_ms=fade_ms,
        )
        sf.write(str(wav_path), audio, sr)
        results.append({
            "file": filename, "speaker": speaker,
            "duration": round(len(audio) / sr, 3),
            "status": "OK",
        })
        print(f"  [OK] {filename} ({speaker})")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Pre-process WAV lines: room reverb, speaker volume, RMS normalize.",
    )
    p.add_argument("input_dir",
                    help="Directory containing WAV line files")
    p.add_argument("--manifest", required=True,
                    help="JSON manifest with file/speaker info")
    p.add_argument("--reverb-decay", type=float, default=0.15,
                    help="Room reverb decay time in seconds (default: 0.15)")
    p.add_argument("--reverb-mix", type=float, default=0.02,
                    help="Reverb wet/dry ratio (default: 0.02)")
    p.add_argument("--speaker-volume", type=str, default=None,
                    help='Per-speaker dB adjustments as JSON (e.g. \'{"zara": 2.5}\')')
    p.add_argument("--target-rms", type=float, default=0.1,
                    help="RMS normalization target (default: 0.1)")
    p.add_argument("--fade-ms", type=int, default=20,
                    help="Clip fade-in/out in ms (default: 20)")
    p.add_argument("--sr", type=int, default=24000,
                    help="Target sample rate (default: 24000)")
    p.add_argument("--dry-run", action="store_true",
                    help="Preview only — don't modify files")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    with open(args.manifest, encoding="utf-8") as f:
        manifest_data = json.load(f)

    # Support both flat list and {"main": [...]} format
    if isinstance(manifest_data, list):
        manifest = manifest_data
    elif isinstance(manifest_data, dict):
        manifest = manifest_data.get("main", [])
        manifest.extend(manifest_data.get("intro", []))
    else:
        sys.exit("Manifest must be a JSON array or object with 'main'/'intro' keys")

    volume_map = None
    if args.speaker_volume:
        try:
            volume_map = json.loads(args.speaker_volume)
        except json.JSONDecodeError as e:
            sys.exit(f"Invalid JSON for --speaker-volume: {e}\n"
                     f'Expected format: \'{{"speaker": dB}}\' e.g. \'{{"zara": 2.5}}\'')
        if not isinstance(volume_map, dict):
            sys.exit("--speaker-volume must be a JSON object")
        for k, v in volume_map.items():
            if not isinstance(v, (int, float)) or not (-30 <= v <= 30):
                sys.exit(f"--speaker-volume: value for '{k}' must be a number "
                         f"between -30 and 30 dB")

    print(f"Pre-processing {len(manifest)} lines in {args.input_dir}")
    if volume_map:
        print(f"  Speaker volumes: {volume_map}")
    print(f"  Reverb: decay={args.reverb_decay}s, mix={args.reverb_mix}")
    print(f"  RMS target: {args.target_rms}")

    results = preprocess_directory(
        args.input_dir, manifest, sr=args.sr,
        reverb_decay=args.reverb_decay, reverb_mix=args.reverb_mix,
        volume_map=volume_map, target_rms=args.target_rms, fade_ms=args.fade_ms,
        dry_run=args.dry_run,
    )

    ok = sum(1 for r in results if r["status"] == "OK")
    dry = sum(1 for r in results if r["status"] == "DRY")
    missing = sum(1 for r in results if r["status"] == "MISSING")
    print(f"\nDone: {ok} processed, {dry} dry, {missing} missing")


if __name__ == "__main__":
    main()
