#!/usr/bin/env python3
"""
Place backchannel clips into mixed podcast audio at strategic positions.

Rules from production experience:
- Only after long turns (>6s by default)
- From a third person (not the current or next speaker)
- Maximum N total per episode (default: 12)
- Minimum gap between placements (default: 5 lines)
- Placed ~100ms before the next speaker starts

Usage:
    # Typically called from mix_episode.py, not directly
    python generator/place_backchannels.py mixed.wav --manifest manifest.json \
        --backchannels bc_clips/ -o mixed_with_bc.wav
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import numpy as np
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False


def plan_backchannel_placement(line_positions, speakers,
                                max_count=12, min_gap=5,
                                min_turn_duration=6.0,
                                sr=24000, rng=None):
    """Plan where to place backchannels in the timeline.

    Args:
        line_positions: list of dicts with {pos_samples, speaker, duration}
        speakers: list of all speaker names in the episode
        max_count: maximum total backchannels
        min_gap: minimum lines between placements
        min_turn_duration: only place after turns longer than this (seconds)
        rng: numpy random generator (for reproducibility)

    Returns list of placement dicts:
        [{position_samples, reactor_speaker, after_line_idx}]
    """
    if rng is None:
        rng = np.random.default_rng(42)

    placements = []
    last_bc_idx = -min_gap  # allow first placement

    for i in range(1, len(line_positions)):
        if len(placements) >= max_count:
            break

        prev = line_positions[i - 1]
        curr = line_positions[i]

        # Only when speaker changes after a long turn
        if (prev["speaker"] == curr["speaker"]
                or prev["duration"] < min_turn_duration
                or i - last_bc_idx < min_gap):
            continue

        # Pick third person (not prev speaker, not current speaker)
        possible = [s for s in speakers
                    if s != prev["speaker"] and s != curr["speaker"]]
        if not possible:
            continue

        reactor = possible[int(rng.integers(len(possible)))]

        placements.append({
            "position_samples": curr["pos_samples"] - int(sr * 0.1),
            "reactor_speaker": reactor,
            "after_line_idx": i - 1,
        })
        last_bc_idx = i

    return placements


def place_backchannels(audio, placements, bc_clips, sr=24000,
                       volume_db=-3.0, rng=None):
    """Overlay backchannel clips onto audio at planned positions.

    Args:
        audio: numpy array (will be modified in-place)
        placements: from plan_backchannel_placement
        bc_clips: {speaker: [np.array, ...]} from load_backchannel_library
        sr: sample rate
        volume_db: backchannel volume relative to full scale
        rng: numpy random generator

    Returns (modified_audio, count_placed).
    """
    if not HAS_AUDIO:
        sys.exit("Requires numpy: pip install numpy")

    if rng is None:
        rng = np.random.default_rng(42)

    audio = audio.copy()
    volume = 10 ** (volume_db / 20)
    placed = 0
    last_clip_per_speaker = {}

    for placement in placements:
        speaker = placement["reactor_speaker"]
        pos = placement["position_samples"]

        if speaker not in bc_clips or not bc_clips[speaker]:
            continue

        # Pick a clip, avoiding the last one used for this speaker
        n_clips = len(bc_clips[speaker])
        available = list(range(n_clips))
        last_used = last_clip_per_speaker.get(speaker, -1)
        if last_used in available and len(available) > 1:
            available.remove(last_used)

        clip_idx = available[int(rng.integers(len(available)))]
        clip = bc_clips[speaker][clip_idx]
        last_clip_per_speaker[speaker] = clip_idx

        # Overlay
        if pos > 0 and pos + len(clip) < len(audio):
            audio[pos:pos + len(clip)] += clip * volume
            placed += 1

    # Clamp to prevent clipping from overlaid backchannels
    np.clip(audio, -1.0, 1.0, out=audio)
    return audio, placed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Place backchannel clips into mixed podcast audio.",
    )
    p.add_argument("input", help="Input audio WAV file")
    p.add_argument("--manifest", required=True,
                    help="Manifest JSON with line positions and speakers")
    p.add_argument("--backchannels", required=True,
                    help="Directory with backchannel clips (bc_speaker_NN.wav)")
    p.add_argument("-o", "--output", default=None,
                    help="Output WAV file (default: input_bc.wav)")
    p.add_argument("--max-count", type=int, default=12,
                    help="Maximum backchannel insertions (default: 12)")
    p.add_argument("--min-gap", type=int, default=5,
                    help="Minimum lines between backchannels (default: 5)")
    p.add_argument("--min-turn", type=float, default=6.0,
                    help="Minimum turn duration to trigger (default: 6.0s)")
    p.add_argument("--volume", type=float, default=-3.0,
                    help="Backchannel volume in dB (default: -3.0)")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    p.add_argument("--dry-run", action="store_true",
                    help="Show placement plan without processing")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    from generate_backchannels import load_backchannel_library

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)

    # Build line_positions from manifest
    entries = manifest if isinstance(manifest, list) else manifest.get("main", [])
    speakers = sorted(set(e.get("speaker", "") for e in entries))

    # For dry run, just show the plan
    rng = np.random.default_rng(args.seed)

    # Build approximate positions (would need actual audio analysis for precise)
    line_positions = []
    pos = 0
    sr = 24000
    for entry in entries:
        dur = entry.get("duration", 2.0)
        line_positions.append({
            "pos_samples": pos,
            "speaker": entry.get("speaker", ""),
            "duration": dur,
        })
        pos += int(dur * sr) + int(0.15 * sr)

    placements = plan_backchannel_placement(
        line_positions, speakers,
        max_count=args.max_count, min_gap=args.min_gap,
        min_turn_duration=args.min_turn, rng=rng,
    )

    print(f"Planned {len(placements)} backchannel placements "
          f"(max {args.max_count})")

    if args.dry_run:
        for p in placements:
            print(f"  Line {p['after_line_idx']}: {p['reactor_speaker']} "
                  f"at {p['position_samples'] / sr:.1f}s")
        return

    # Load audio and clips
    bc_clips = load_backchannel_library(args.backchannels)
    audio, file_sr = sf.read(str(args.input), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    rng = np.random.default_rng(args.seed)
    audio, placed = place_backchannels(
        audio, placements, bc_clips, sr=file_sr,
        volume_db=args.volume, rng=rng,
    )

    output = args.output or str(Path(args.input).with_stem(
        Path(args.input).stem + "_bc"))
    sf.write(output, audio, file_sr)
    print(f"Placed {placed} backchannels -> {output}")


if __name__ == "__main__":
    main()
