#!/usr/bin/env python3
"""
TTS overrides: split long lines into segments with controlled pauses.

Engine-agnostic. The override format works for Qwen, Chatterbox, or any
per-line TTS engine. ElevenLabs text_to_dialogue is unaffected (it generates
full sections, not individual lines).

Override JSON format:
    {
        "overrides": {
            "015": "simple text replacement",
            "019": [
                {"text": "First segment...", "pause_after": 0.3},
                {"text": "Second segment.", "pause_after": 0.0}
            ]
        }
    }

Usage:
    python generator/tts_overrides.py overrides.json --list
    python generator/tts_overrides.py overrides.json --check 015
    python generator/tts_overrides.py overrides.json --validate
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


def load_overrides(path):
    """Load overrides from a JSON file.

    Returns the inner "overrides" dict. If the file has no "overrides" key,
    returns an empty dict.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("overrides", {})


def get_override(overrides, line_index):
    """Look up an override by line index.

    Keys are zero-padded to 3 digits (e.g. "015").
    Returns str (simple replacement), list[dict] (segments), or None.
    """
    key = f"{line_index:03d}"
    return overrides.get(key)


def is_segmented(override):
    """True if the override is a list of segment dicts (not a simple string)."""
    return isinstance(override, list)


def apply_override_text(original_text, override):
    """Apply an override to the original text.

    - Simple string override: returns the replacement text.
    - Segmented override: returns the list of segments unchanged.
    - None: returns original_text.
    """
    if override is None:
        return original_text
    if isinstance(override, str):
        return override
    return override  # list of segments


def get_segment_texts(override):
    """Extract text strings from a segmented override.

    Returns list of text strings (for TTS generation of each segment).
    """
    if not is_segmented(override):
        return [override] if isinstance(override, str) else []
    return [seg["text"] for seg in override]


def get_segment_pauses(override):
    """Extract pause durations from a segmented override.

    Returns list of pause durations in seconds (one per segment).
    """
    if not is_segmented(override):
        return [0.0]
    return [seg.get("pause_after", 0.0) for seg in override]


def assemble_segments(segment_wavs, pauses, sr, output_path):
    """Concatenate WAV segment files with silence pauses between them.

    Args:
        segment_wavs: list of Path objects to WAV segment files
        pauses: list of pause durations in seconds (one per segment)
        sr: target sample rate
        output_path: output WAV path

    Returns total duration in seconds.
    """
    if not HAS_AUDIO:
        raise RuntimeError("assemble_segments requires numpy and soundfile: "
                           "pip install numpy soundfile")

    parts = []
    for i, wav_path in enumerate(segment_wavs):
        audio, file_sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            # Simple resampling via linear interpolation
            new_len = int(len(audio) * sr / file_sr)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
        parts.append(audio)

        pause = pauses[i] if i < len(pauses) else 0.0
        if pause > 0:
            parts.append(np.zeros(int(sr * pause), dtype=np.float32))

    if not parts:
        raise ValueError("No segments to assemble")

    full = np.concatenate(parts)
    sf.write(str(output_path), full, sr)
    return len(full) / sr


def validate_overrides(overrides):
    """Validate override structure. Returns list of issues (empty = valid)."""
    issues = []
    for key, value in overrides.items():
        if not key.isdigit():
            issues.append(f"Key '{key}' is not a numeric string")
            continue
        if isinstance(value, str):
            if not value.strip():
                issues.append(f"Key '{key}': empty string replacement")
        elif isinstance(value, list):
            for i, seg in enumerate(value):
                if not isinstance(seg, dict):
                    issues.append(f"Key '{key}' segment {i}: not a dict")
                    continue
                if "text" not in seg:
                    issues.append(f"Key '{key}' segment {i}: missing 'text' field")
                elif not seg["text"].strip():
                    issues.append(f"Key '{key}' segment {i}: empty text")
                if "pause_after" in seg:
                    pause = seg["pause_after"]
                    if not isinstance(pause, (int, float)):
                        issues.append(f"Key '{key}' segment {i}: pause_after not a number")
                    elif pause < 0:
                        issues.append(f"Key '{key}' segment {i}: negative pause_after")
        else:
            issues.append(f"Key '{key}': value must be string or list, got {type(value).__name__}")
    return issues


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="TTS override utility: inspect and validate override JSON files.",
    )
    p.add_argument("overrides_file",
                    help="Path to overrides JSON file")
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true",
                         help="List all overrides")
    action.add_argument("--check", type=int,
                         help="Show override for a specific line index")
    action.add_argument("--validate", action="store_true",
                         help="Validate JSON structure")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not Path(args.overrides_file).exists():
        sys.exit(f"Overrides file not found: {args.overrides_file}")
    overrides = load_overrides(args.overrides_file)

    if args.list:
        print(f"Overrides: {len(overrides)} entries")
        for key, value in sorted(overrides.items()):
            if isinstance(value, str):
                print(f"  {key}: \"{value[:60]}{'...' if len(value) > 60 else ''}\"")
            elif isinstance(value, list):
                print(f"  {key}: [{len(value)} segments]")
                for i, seg in enumerate(value):
                    text = seg.get("text", "")
                    pause = seg.get("pause_after", 0)
                    print(f"    {i}: \"{text[:50]}{'...' if len(text) > 50 else ''}\" "
                          f"(pause={pause}s)")

    elif args.check is not None:
        idx = args.check
        override = get_override(overrides, idx)
        if override is None:
            print(f"No override for line {idx:03d}")
        else:
            print(json.dumps({f"{idx:03d}": override}, indent=2))

    elif args.validate:
        issues = validate_overrides(overrides)
        if issues:
            print(f"INVALID: {len(issues)} issue(s)")
            for issue in issues:
                print(f"  - {issue}")
            sys.exit(1)
        else:
            print(f"VALID: {len(overrides)} overrides, all well-formed")


if __name__ == "__main__":
    main()
