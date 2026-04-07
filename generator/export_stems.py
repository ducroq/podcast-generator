#!/usr/bin/env python3
"""
Export per-speaker stems for DAW editing.

Creates one WAV per speaker, all same length, silence-padded, aligned
for drag-and-drop into any DAW. Also generates an Audacity LOF import file.

Pipeline position: after TTS generation + clean, before or alongside mixing.

Usage:
    python generator/export_stems.py output_dir/ --manifest manifest.json --script script.txt
    python generator/export_stems.py output_dir/ --manifest manifest.json --script script.txt \
        -o stems/ --sr 24000
"""

import argparse
import json
import re
import sys
from pathlib import Path


def _safe_filename(speaker):
    """Sanitize a speaker name for use as a filename component."""
    return re.sub(r'[^\w]', '_', speaker)

try:
    import numpy as np
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False


def parse_script_timing(script_path):
    """Parse a script into ordered line entries with speaker info.

    Returns list of dicts: [{speaker, text, index}]
    """
    entries = []
    index = 0
    with open(script_path, encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped or stripped.startswith("=") or stripped.startswith("#"):
                continue
            if re.match(r"^[A-Z][A-Z\s':,\-]+$", stripped):
                continue
            if re.match(r"^\[.*\]$", stripped):
                continue
            match = re.match(r"(.+?):\s*(?:\[([^\]]*)\]\s*)?(.*)", stripped)
            if match:
                speaker = match.group(1).strip().lower()
                text = match.group(3).strip()
                if text:
                    entries.append({"speaker": speaker, "text": text, "index": index})
                    index += 1
    return entries


def build_timeline(audio_dir, entries, sr=24000,
                   speaker_change_pause=0.15, same_speaker_pause=0.08):
    """Build a timeline of line positions from WAV files.

    Returns (total_samples, list of {speaker, start_sample, audio_array}).
    """
    if not HAS_AUDIO:
        sys.exit("Requires numpy and soundfile: pip install numpy soundfile")

    audio_dir = Path(audio_dir)
    timeline = []
    skipped = []
    pos = 0
    prev_speaker = None

    for entry in entries:
        speaker = entry["speaker"]
        idx = entry["index"]
        filename = f"{idx:03d}_{_safe_filename(speaker)}.wav"
        wav_path = audio_dir / filename

        if not wav_path.exists():
            # Try manifest-style filename
            for candidate in audio_dir.glob(f"{idx:03d}_*.wav"):
                wav_path = candidate
                break

        if not wav_path.exists():
            skipped.append(filename)
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

        # Add pause
        if prev_speaker is not None:
            pause = speaker_change_pause if speaker != prev_speaker else same_speaker_pause
            pos += int(sr * pause)

        timeline.append({
            "speaker": speaker,
            "start_sample": pos,
            "audio": audio,
        })
        pos += len(audio)
        prev_speaker = speaker

    if skipped:
        print(f"  WARNING: {len(skipped)} missing WAV files skipped:")
        for s in skipped[:10]:
            print(f"    {s}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")

    return pos, timeline


def export_stems(audio_dir, entries, output_dir, sr=24000,
                 speaker_change_pause=0.15, same_speaker_pause=0.08):
    """Export per-speaker stem WAVs, all same length.

    Returns dict: {speaker: output_path}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_samples, timeline = build_timeline(
        audio_dir, entries, sr, speaker_change_pause, same_speaker_pause,
    )

    if not timeline:
        print("No audio files found for stems.")
        return {}

    # Group by speaker
    speakers = sorted(set(t["speaker"] for t in timeline))

    stem_paths = {}
    for speaker in speakers:
        stem = np.zeros(total_samples, dtype=np.float32)
        line_count = 0
        for t in timeline:
            if t["speaker"] == speaker:
                start = t["start_sample"]
                end = start + len(t["audio"])
                if end <= total_samples:
                    stem[start:end] = t["audio"]
                else:
                    stem[start:total_samples] = t["audio"][:total_samples - start]
                line_count += 1

        filename = f"stem_{_safe_filename(speaker)}.wav"
        out_path = output_dir / filename
        sf.write(str(out_path), stem, sr)
        stem_paths[speaker] = str(out_path)
        print(f"  {filename}: {line_count} lines, {total_samples / sr:.1f}s")

    # Generate Audacity LOF file
    lof_path = output_dir / "import.lof"
    with open(lof_path, "w") as f:
        for speaker in speakers:
            filename = f"stem_{_safe_filename(speaker)}.wav"
            f.write(f'file "{filename}" offset 0.0\n')
    print(f"  import.lof: Audacity import file")

    # Summary
    total_dur = total_samples / sr
    print(f"\n{len(speakers)} stems exported, {total_dur:.1f}s each")

    return stem_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Export per-speaker stems for DAW editing.",
    )
    p.add_argument("audio_dir",
                    help="Directory containing generated WAV line files")
    p.add_argument("--script", required=True,
                    help="Script file (for speaker/line ordering)")
    p.add_argument("-o", "--output-dir", default="stems",
                    help="Output directory for stems (default: stems/)")
    p.add_argument("--sr", type=int, default=24000,
                    help="Sample rate (default: 24000)")
    p.add_argument("--speaker-pause", type=float, default=0.15,
                    help="Pause between different speakers (default: 0.15s)")
    p.add_argument("--same-pause", type=float, default=0.08,
                    help="Pause between same speaker lines (default: 0.08s)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    entries = parse_script_timing(args.script)
    print(f"Script: {len(entries)} lines")
    speakers = sorted(set(e["speaker"] for e in entries))
    print(f"Speakers: {', '.join(speakers)}")

    export_stems(
        args.audio_dir, entries, args.output_dir, sr=args.sr,
        speaker_change_pause=args.speaker_pause,
        same_speaker_pause=args.same_pause,
    )


if __name__ == "__main__":
    main()
