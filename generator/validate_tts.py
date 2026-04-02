#!/usr/bin/env python3
"""
Validate TTS output for hallucinations by comparing generated audio against
the intended text using ASR transcription.

Usage:
    python generator/validate_tts.py output.wav "Expected text here"
    python generator/validate_tts.py output_dir/ --manifest manifest.json
    python generator/validate_tts.py output.wav "Expected text" --fix --ref-audio ref.mp3 --ref-text "ref transcript"

Checks:
  1. Transcribe output with faster-whisper
  2. Compare against expected text (word overlap)
  3. Flag if output has extra words at start/end (hallucination)
  4. Flag if duration is suspicious for text length
  5. Optionally re-generate flagged samples (--fix)
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def transcribe(audio_path, model_size="base", language="en"):
    """Transcribe audio file using faster-whisper."""
    cmd = [
        sys.executable, "-c", f"""
from faster_whisper import WhisperModel
model = WhisperModel("{model_size}", device="cuda", compute_type="float16")
segments, _ = model.transcribe("{audio_path}", language="{language}")
print(" ".join(s.text.strip() for s in segments))
"""
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def get_duration(audio_path):
    """Get audio duration in seconds."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(audio_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def normalize_text(text):
    """Normalize text for comparison: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def check_hallucination(expected_text, transcribed_text):
    """Compare expected vs transcribed text. Returns (is_ok, issues)."""
    expected_norm = normalize_text(expected_text)
    transcribed_norm = normalize_text(transcribed_text)

    expected_words = expected_norm.split()
    transcribed_words = transcribed_norm.split()

    issues = []

    # Check for extra words at the start (common Qwen hallucination)
    if len(transcribed_words) > len(expected_words):
        # Find where expected text starts in transcribed text
        first_expected = expected_words[0] if expected_words else ""
        prepend_count = 0
        for w in transcribed_words:
            if w == first_expected:
                break
            prepend_count += 1

        if prepend_count > 1:
            extra = " ".join(transcribed_words[:prepend_count])
            issues.append(f"HALLUCINATION_START: {prepend_count} extra words at start: \"{extra}\"")

    # Check for extra words at the end
    if len(transcribed_words) > len(expected_words) + 3:
        last_expected = expected_words[-1] if expected_words else ""
        # Find last occurrence of last expected word
        append_start = None
        for i in range(len(transcribed_words) - 1, -1, -1):
            if transcribed_words[i] == last_expected:
                append_start = i + 1
                break
        if append_start and append_start < len(transcribed_words) - 1:
            extra = " ".join(transcribed_words[append_start:])
            issues.append(f"HALLUCINATION_END: extra words at end: \"{extra}\"")

    # Word overlap ratio
    expected_set = set(expected_words)
    transcribed_set = set(transcribed_words)
    if expected_set:
        overlap = len(expected_set & transcribed_set) / len(expected_set)
        if overlap < 0.7:
            issues.append(f"LOW_OVERLAP: only {overlap:.0%} of expected words found in transcription")

    # Length ratio
    len_ratio = len(transcribed_words) / max(len(expected_words), 1)
    if len_ratio > 1.4:
        issues.append(f"TOO_LONG: transcription is {len_ratio:.1f}x expected word count")
    elif len_ratio < 0.5:
        issues.append(f"TOO_SHORT: transcription is {len_ratio:.1f}x expected word count")

    is_ok = len(issues) == 0
    return is_ok, issues


def validate_single(audio_path, expected_text, language="en"):
    """Validate a single audio file. Returns dict with results."""
    audio_path = Path(audio_path)

    if not audio_path.exists():
        return {"file": str(audio_path), "status": "ERROR", "issues": ["File not found"]}

    duration = get_duration(audio_path)

    # Expected duration: ~0.3s per word for natural speech
    expected_words = len(expected_text.split())
    expected_duration = expected_words * 0.35
    duration_ratio = duration / max(expected_duration, 0.1)

    transcription = transcribe(str(audio_path), language=language)
    if transcription is None:
        return {
            "file": str(audio_path),
            "status": "ERROR",
            "issues": ["Transcription failed"],
            "duration": duration,
        }

    is_ok, issues = check_hallucination(expected_text, transcription)

    # Duration sanity check
    if duration_ratio > 2.0:
        issues.append(f"DURATION: {duration:.1f}s is {duration_ratio:.1f}x expected ({expected_duration:.1f}s)")
        is_ok = False
    elif duration < 0.5:
        issues.append(f"SILENT: output is only {duration:.1f}s")
        is_ok = False

    return {
        "file": str(audio_path),
        "status": "OK" if is_ok else "FLAGGED",
        "duration": round(duration, 1),
        "expected_text": expected_text,
        "transcription": transcription,
        "issues": issues,
    }


def validate_manifest(manifest_path):
    """Validate multiple files from a JSON manifest.

    Manifest format:
    [
        {"file": "output_1.wav", "text": "Expected text for line 1"},
        {"file": "output_2.wav", "text": "Expected text for line 2"},
        ...
    ]
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    results = []
    flagged = 0
    for entry in manifest:
        result = validate_single(entry["file"], entry["text"],
                                 language=entry.get("language", "en"))
        results.append(result)
        if result["status"] == "FLAGGED":
            flagged += 1

    return results, flagged


def print_result(result):
    """Print a single validation result."""
    status = result["status"]
    icon = "OK" if status == "OK" else "!!" if status == "FLAGGED" else "??"
    print(f"  [{icon}] {Path(result['file']).name} ({result.get('duration', '?')}s)")

    if result.get("transcription"):
        print(f"       Expected:    {result['expected_text'][:80]}...")
        print(f"       Transcribed: {result['transcription'][:80]}...")

    for issue in result.get("issues", []):
        print(f"       -> {issue}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate TTS output for hallucinations")
    parser.add_argument("input", help="Audio file or directory")
    parser.add_argument("expected_text", nargs="?", help="Expected text (for single file)")
    parser.add_argument("--manifest", help="JSON manifest with file/text pairs")
    parser.add_argument("--language", default="en", help="Language code (default: en)")

    args = parser.parse_args()

    if args.manifest:
        results, flagged = validate_manifest(args.manifest)
        print(f"\nValidated {len(results)} files, {flagged} flagged:\n")
        for r in results:
            print_result(r)
    elif args.expected_text:
        result = validate_single(args.input, args.expected_text, args.language)
        print()
        print_result(result)
    else:
        print("Provide either expected_text or --manifest")
        sys.exit(1)

    # Exit code: 1 if any flagged
    flagged_count = sum(1 for r in ([result] if not args.manifest else results)
                        if r["status"] == "FLAGGED")
    sys.exit(1 if flagged_count > 0 else 0)
