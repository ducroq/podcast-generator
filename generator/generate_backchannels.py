#!/usr/bin/env python3
"""
Generate backchannel clip libraries for podcast voices.

Produces short reaction sounds (mmhm, right, yeah, huh) per voice
for layering onto dialogue during non-speaking moments. Clips are
organized by speaker and tagged by type (agreement, surprise, tracking).

The output is a reusable library — generate once per voice, use across episodes.

Usage (on gpu-server):
    python generator/generate_backchannels.py --voices voices.json -o bc_clips/
    python generator/generate_backchannels.py --list bc_clips/
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False


# ---------------------------------------------------------------------------
# Backchannel word library
# ---------------------------------------------------------------------------

BACKCHANNEL_WORDS = {
    "agreement": ["Mmhm.", "Right.", "Yeah.", "Mm."],
    "surprise": ["Huh.", "Oh.", "Wow."],
    "tracking": ["Right.", "Okay.", "Right, right.", "Yeah, yeah."],
}

# All unique words across categories
ALL_WORDS = sorted(set(w for words in BACKCHANNEL_WORDS.values() for w in words))


def get_words_for_type(bc_type):
    """Get backchannel words for a given type."""
    return BACKCHANNEL_WORDS.get(bc_type, [])


def get_all_words():
    """Get all unique backchannel words."""
    return ALL_WORDS


# ---------------------------------------------------------------------------
# Library I/O
# ---------------------------------------------------------------------------

def load_backchannel_library(bc_dir):
    """Load all backchannel WAV clips from a directory, grouped by speaker.

    Expects filenames: bc_<speaker>_<NN>.wav
    Returns: {speaker: [np.array, ...]}
    """
    if not HAS_AUDIO:
        sys.exit("Requires numpy and soundfile: pip install numpy soundfile")

    bc_dir = Path(bc_dir)
    clips = {}

    for wav_path in sorted(bc_dir.glob("bc_*.wav")):
        parts = wav_path.stem.split("_")
        if len(parts) >= 3:
            speaker = parts[1]
            audio, sr = sf.read(str(wav_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if speaker not in clips:
                clips[speaker] = []
            clips[speaker].append(audio)

    return clips


def load_backchannel_manifest(bc_dir):
    """Load backchannel manifest from a directory.

    Returns manifest list or None if no manifest exists.
    """
    manifest_path = Path(bc_dir) / "bc_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def list_library(bc_dir):
    """List contents of a backchannel clip library."""
    bc_dir = Path(bc_dir)
    clips = {}
    for wav_path in sorted(bc_dir.glob("bc_*.wav")):
        parts = wav_path.stem.split("_")
        if len(parts) >= 3:
            speaker = parts[1]
            if speaker not in clips:
                clips[speaker] = []
            clips[speaker].append(wav_path.name)

    if not clips:
        print(f"No backchannel clips found in {bc_dir}")
        return

    total = sum(len(v) for v in clips.values())
    print(f"Backchannel library: {total} clips, {len(clips)} speakers")
    for speaker, files in sorted(clips.items()):
        print(f"  {speaker}: {len(files)} clips")
        for f in files:
            print(f"    {f}")


# ---------------------------------------------------------------------------
# Generation (requires TTS engine — runs on gpu-server)
# ---------------------------------------------------------------------------

def plan_backchannel_clips(voice_config, output_dir, words=None):
    """Plan backchannel clips for one voice (manifest only, no audio).

    Returns manifest list: [{"file": "bc_morgan_00.wav", "word": "Mmhm.", "speaker": "morgan"}]
    Use this to prepare the manifest structure before generating audio on gpu-server.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    speaker = voice_config["name"]
    words = words or ALL_WORDS

    manifest = []
    for i, word in enumerate(words):
        filename = f"bc_{speaker}_{i:02d}.wav"
        manifest.append({
            "file": filename,
            "word": word,
            "speaker": speaker,
        })

    return manifest


def generate_backchannel_clips(voice_config, output_dir, engine="qwen",
                                words=None):
    """Generate backchannel clips for one voice.

    Requires a TTS engine running on gpu-server. This function raises
    NotImplementedError — use plan_backchannel_clips() to create the
    manifest, then generate WAV files on gpu-server with the appropriate
    TTS engine.
    """
    raise NotImplementedError(
        "TTS generation requires gpu-server. Use plan_backchannel_clips() "
        "to create the manifest, then generate WAV files on gpu-server."
    )


def save_manifest(manifest, output_dir):
    """Save backchannel manifest to output directory."""
    manifest_path = Path(output_dir) / "bc_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Generate or inspect backchannel clip libraries.",
    )
    p.add_argument("--voices", type=str, default=None,
                    help="Voice config JSON for generation")
    p.add_argument("-o", "--output-dir", default="./bc_clips",
                    help="Output directory for clips (default: ./bc_clips)")
    p.add_argument("--engine", default="qwen",
                    help="TTS engine (default: qwen)")
    p.add_argument("--list", type=str, default=None,
                    help="List contents of an existing backchannel library")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.list:
        list_library(args.list)
    elif args.voices:
        with open(args.voices, encoding="utf-8") as f:
            voices = json.load(f)

        all_manifest = []
        for i, voice in enumerate(voices):
            if "name" not in voice:
                print(f"  WARNING: voice entry {i} missing 'name', skipping")
                continue
            print(f"  Planning manifest for {voice['name']}...")
            manifest = plan_backchannel_clips(voice, args.output_dir)
            all_manifest.extend(manifest)

        save_manifest(all_manifest, args.output_dir)
        print(f"\nManifest saved: {args.output_dir}/bc_manifest.json")
        print(f"Total: {len(all_manifest)} clip entries planned")
        print(f"\nWARNING: No audio generated. This tool creates the manifest only.")
        print(f"Generate WAV files on gpu-server using the manifest, then use")
        print(f"place_backchannels.py to insert them into your mix.")
    else:
        print("Provide --voices for generation or --list for inspection")
        sys.exit(1)


if __name__ == "__main__":
    main()
