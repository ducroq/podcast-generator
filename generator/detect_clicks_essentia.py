"""LPC-based click detection using Essentia (Linux/gpu-server only).

Essentia's ClickDetector uses linear predictive coding to model the
expected signal, then flags positions where the actual signal diverges
significantly. This is much more accurate than amplitude thresholding
for detecting perceptual clicks/pops in TTS output.

This module is designed to run as part of TTS validation on the gpu-server,
not in the local processing pipeline. It detects clicks but does not
repair them — flagged lines should be regenerated.

Usage:
    from detect_clicks_essentia import detect_clicks, detect_clicks_batch

    # Single file
    clicks = detect_clicks("path/to/audio.wav")

    # Batch (reads manifest)
    report = detect_clicks_batch(manifest, tts_dir)
"""

import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

try:
    import essentia.standard as es
    _HAS_ESSENTIA = True
except ImportError:
    _HAS_ESSENTIA = False


def detect_clicks(audio_path, frame_size=512, hop_size=256,
                  silence_threshold=-50, order=12):
    """Detect clicks in an audio file using Essentia's ClickDetector.

    Args:
        audio_path: path to WAV file
        frame_size: analysis frame size (samples)
        hop_size: hop between frames (samples)
        silence_threshold: skip frames below this level (dB)
        order: LPC model order (higher = more sensitive, slower)

    Returns:
        list of dicts: [{start_ms, end_ms, frame_idx}]
        Empty list if no clicks found or Essentia unavailable.
    """
    if not _HAS_ESSENTIA:
        logger.warning("Essentia not available — skipping click detection")
        return []

    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    detector = es.ClickDetector(
        frameSize=frame_size,
        hopSize=hop_size,
        silenceThreshold=silence_threshold,
        order=order,
    )

    clicks = []
    for frame_idx in range(0, len(audio) - frame_size, hop_size):
        frame = audio[frame_idx:frame_idx + frame_size]
        starts, ends = detector(frame)
        for s, e in zip(starts, ends):
            clicks.append({
                "start_ms": round((frame_idx + s) / sr * 1000, 1),
                "end_ms": round((frame_idx + e) / sr * 1000, 1),
                "frame_idx": frame_idx,
            })

    return clicks


def detect_clicks_batch(manifest, tts_dir, **kwargs):
    """Run click detection on all TTS files in a manifest.

    Args:
        manifest: manifest dict with lines
        tts_dir: Path to TTS directory
        **kwargs: passed to detect_clicks()

    Returns:
        dict: {total, clean, flagged, results: [{hash, speaker, text, clicks}]}
    """
    if not _HAS_ESSENTIA:
        logger.warning("Essentia not available — skipping batch click detection")
        return {"total": 0, "clean": 0, "flagged": 0, "results": []}

    tts_dir = Path(tts_dir)
    results = []
    total = 0
    flagged = 0

    for h, info in manifest["lines"].items():
        if info.get("status") != "exists":
            continue
        wav_path = tts_dir / info["file"]
        if not wav_path.exists():
            continue

        total += 1
        clicks = detect_clicks(wav_path, **kwargs)

        if clicks:
            flagged += 1
            results.append({
                "hash": h,
                "speaker": info["speaker"],
                "text": info["text"][:60],
                "clicks": clicks,
                "click_count": len(clicks),
            })

    return {
        "total": total,
        "clean": total - flagged,
        "flagged": flagged,
        "results": results,
    }


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Detect clicks in TTS audio (Essentia)")
    parser.add_argument("path", help="WAV file or directory of WAVs")
    parser.add_argument("--frame-size", type=int, default=512)
    parser.add_argument("--hop-size", type=int, default=256)
    parser.add_argument("--order", type=int, default=12)
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not _HAS_ESSENTIA:
        sys.exit("Essentia not installed. Run: pip install essentia (Linux only)")

    path = Path(args.path)
    if path.is_file():
        clicks = detect_clicks(path, frame_size=args.frame_size,
                               hop_size=args.hop_size, order=args.order)
        if args.json:
            print(json.dumps(clicks, indent=2))
        elif clicks:
            print(f"{path.name}: {len(clicks)} clicks")
            for c in clicks:
                print(f"  {c['start_ms']:.1f}ms - {c['end_ms']:.1f}ms")
        else:
            print(f"{path.name}: clean")
    elif path.is_dir():
        wavs = sorted(path.glob("*.wav"))
        all_results = []
        for wav in wavs:
            clicks = detect_clicks(wav, frame_size=args.frame_size,
                                   hop_size=args.hop_size, order=args.order)
            if clicks:
                all_results.append({"file": wav.name, "clicks": clicks})
                print(f"{wav.name}: {len(clicks)} clicks")
            else:
                print(f"{wav.name}: clean")
        if args.json:
            print(json.dumps(all_results, indent=2))
        print(f"\n{len(all_results)}/{len(wavs)} files have clicks")
