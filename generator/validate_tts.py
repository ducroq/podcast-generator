#!/usr/bin/env python3
"""
Validate TTS output for hallucinations by comparing generated audio against
the intended text using ASR transcription. Always writes a validation report.

Usage:
    python generator/validate_tts.py output.wav "Expected text here"
    python generator/validate_tts.py . --manifest manifest.json
    python generator/validate_tts.py . --manifest manifest.json --revalidate-flagged

The validation report (validation.json) is always saved alongside the audio.
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from audio_utils import get_duration


def transcribe(audio_path, model_size="base", language="en"):
    """Transcribe audio file using faster-whisper via subprocess."""
    script = str(Path(__file__).parent / "_transcribe_worker.py")
    cmd = [
        sys.executable, script,
        "--audio", str(audio_path),
        "--model", model_size,
        "--language", language,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


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
        append_start = None
        for i in range(len(transcribed_words) - 1, -1, -1):
            if transcribed_words[i] == last_expected:
                append_start = i + 1
                break
        if append_start is not None and append_start < len(transcribed_words):
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


WORD_DURATION = {
    "en": 0.35,
    "de": 0.45,
    "nl": 0.35,
}


def validate_single(audio_path, expected_text, language="en", ref_path=None):
    """Validate a single audio file. Returns dict with results.

    Runs core checks (ASR + duration) always, plus optional quality checks
    (UTMOS MOS, speaker similarity, language ID) when their dependencies
    are installed.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        return {"file": str(audio_path), "status": "ERROR", "issues": ["File not found"]}

    duration = get_duration(audio_path)

    expected_words = len(expected_text.split())
    expected_duration = expected_words * WORD_DURATION.get(language, 0.35)
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

    if duration_ratio > 2.0:
        issues.append(f"DURATION: {duration:.1f}s is {duration_ratio:.1f}x expected ({expected_duration:.1f}s)")
        is_ok = False
    elif duration < 0.5:
        issues.append(f"SILENT: output is only {duration:.1f}s")
        is_ok = False

    result = {
        "file": str(audio_path),
        "status": "OK" if is_ok else "FLAGGED",
        "duration": round(duration, 1),
        "expected_text": expected_text,
        "transcription": transcription,
        "issues": issues,
    }

    # Optional quality checks (gracefully skip if dependencies not installed)
    from quality_checks import run_quality_checks
    quality = run_quality_checks(audio_path, ref_path=ref_path,
                                 expected_language=language)
    if quality:
        result["quality"] = quality
        # Flag based on quality thresholds
        if quality.get("mos") is not None and quality["mos"] < 3.5:
            issues.append(f"LOW_MOS: {quality['mos']}/5.0 (threshold: 3.5)")
            result["status"] = "FLAGGED"
        if quality.get("speaker_similarity") is not None and quality["speaker_similarity"] < 0.75:
            issues.append(f"VOICE_DRIFT: speaker similarity {quality['speaker_similarity']:.2f} (threshold: 0.75)")
            result["status"] = "FLAGGED"
        if quality.get("language_match") is False:
            issues.append(f"WRONG_LANGUAGE: detected '{quality.get('detected_language')}' "
                         f"(confidence: {quality.get('language_confidence', '?')})")
            result["status"] = "FLAGGED"

    return result


def validate_manifest(manifest_path, skip_passed=False):
    """Validate files from a JSON manifest. Returns (results, flagged_count).

    Manifest format:
    [
        {"file": "output_1.wav", "text": "Expected text for line 1"},
        {"file": "output_2.wav", "text": "Expected text for line 2"},
        ...
    ]

    If skip_passed=True and a previous validation.json exists, only re-validates
    entries that were FLAGGED or ERROR last time.
    """
    manifest_dir = Path(manifest_path).parent

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Load previous report if skipping passed entries
    previous = {}
    if skip_passed:
        report = load_report(manifest_dir)
        if report:
            for r in report.get("results", []):
                previous[r["file"]] = r

    results = []
    flagged = 0
    skipped = 0
    for entry in manifest:
        file_path = Path(entry["file"])

        # Reject absolute paths and path traversal
        if file_path.is_absolute() or '..' in file_path.parts:
            results.append({
                "file": str(file_path),
                "status": "ERROR",
                "issues": ["Rejected path: must be relative without '..'"],
            })
            flagged += 1
            continue

        resolved = (manifest_dir / file_path).resolve()
        if not resolved.is_relative_to(manifest_dir.resolve()):
            results.append({
                "file": str(file_path),
                "status": "ERROR",
                "issues": ["Rejected path: resolves outside manifest directory"],
            })
            flagged += 1
            continue

        # Skip previously passed entries
        prev = previous.get(str(resolved))
        if prev and prev.get("status") == "OK":
            results.append(prev)
            skipped += 1
            continue

        result = validate_single(str(resolved), entry["text"],
                                 language=entry.get("language", "en"),
                                 ref_path=entry.get("ref_audio"))
        results.append(result)
        if result["status"] == "FLAGGED":
            flagged += 1

    if skipped:
        print(f"  Skipped {skipped} previously passed entries")

    return results, flagged


def build_report(results, language="en", engine=None, manifest_path=None):
    """Build a structured validation report."""
    ok = sum(1 for r in results if r["status"] == "OK")
    flagged = sum(1 for r in results if r["status"] == "FLAGGED")
    errors = sum(1 for r in results if r["status"] == "ERROR")

    return {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "language": language,
        "engine": engine,
        "manifest": str(manifest_path) if manifest_path else None,
        "summary": {
            "total": len(results),
            "ok": ok,
            "flagged": flagged,
            "errors": errors,
        },
        "results": results,
    }


def save_report(report, output_dir):
    """Save validation report to output_dir/validation.json.

    If a previous report exists, it is preserved as validation_prev.json
    so you can compare across runs.
    """
    output_dir = Path(output_dir)
    report_path = output_dir / "validation.json"
    prev_path = output_dir / "validation_prev.json"

    if report_path.exists():
        # Rotate: current → prev (keep one generation of history)
        if prev_path.exists():
            prev_path.unlink()
        report_path.rename(prev_path)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report_path


def load_report(output_dir):
    """Load an existing validation report, or None if not found."""
    report_path = Path(output_dir) / "validation.json"
    if not report_path.exists():
        return None
    with open(report_path, encoding="utf-8") as f:
        return json.load(f)


def print_result(result):
    """Print a single validation result."""
    status = result["status"]
    icon = "OK" if status == "OK" else "!!" if status == "FLAGGED" else "??"
    print(f"  [{icon}] {Path(result['file']).name} ({result.get('duration', '?')}s)")

    if result.get("transcription"):
        expected = result['expected_text']
        transcribed = result['transcription']
        print(f"       Expected:    {expected[:80]}{'...' if len(expected) > 80 else ''}")
        print(f"       Transcribed: {transcribed[:80]}{'...' if len(transcribed) > 80 else ''}")

    if result.get("quality"):
        q = result["quality"]
        parts = []
        if q.get("mos") is not None:
            parts.append(f"MOS={q['mos']}")
        if q.get("speaker_similarity") is not None:
            parts.append(f"spk_sim={q['speaker_similarity']:.2f}")
        if q.get("language_match") is not None:
            lang_icon = "ok" if q["language_match"] else "MISMATCH"
            parts.append(f"lang={q.get('detected_language', '?')}({lang_icon})")
        if parts:
            print(f"       Quality:     {', '.join(parts)}")

    for issue in result.get("issues", []):
        print(f"       -> {issue}")


def print_summary(report):
    """Print the report summary."""
    s = report["summary"]
    print(f"\n{'='*50}")
    print(f"VALIDATION REPORT — {report['validated_at'][:10]}")
    if report.get("engine"):
        print(f"Engine: {report['engine']}")
    print(f"Language: {report['language']}")
    print(f"Total: {s['total']}  OK: {s['ok']}  Flagged: {s['flagged']}  Errors: {s['errors']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate TTS output for hallucinations")
    parser.add_argument("input", help="Audio file or directory")
    parser.add_argument("expected_text", nargs="?", help="Expected text (for single file)")
    parser.add_argument("--manifest", help="JSON manifest with file/text pairs")
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    parser.add_argument("--engine", help="TTS engine used (stored in report)")
    parser.add_argument("--ref-audio", help="Voice reference audio for speaker similarity check")
    parser.add_argument("--revalidate-flagged", action="store_true",
                        help="Only re-validate previously FLAGGED/ERROR entries")

    args = parser.parse_args()

    # Report available quality checks
    from quality_checks import get_available_checks
    available = get_available_checks()
    if available:
        print(f"Quality checks available: {', '.join(available)}")
    else:
        print("Quality checks: none (install speechmos, resemblyzer, speechbrain for enhanced validation)")

    if args.manifest:
        results, flagged = validate_manifest(
            args.manifest,
            skip_passed=args.revalidate_flagged,
        )
        report = build_report(results, language=args.language,
                              engine=args.engine, manifest_path=args.manifest)

        # Always save report next to manifest
        report_dir = Path(args.manifest).parent
        report_path = save_report(report, report_dir)

        print_summary(report)
        print()
        for r in results:
            print_result(r)
        print(f"\nReport saved: {report_path}")

    elif args.expected_text:
        result = validate_single(args.input, args.expected_text, args.language,
                                 ref_path=args.ref_audio)
        results = [result]
        report = build_report(results, language=args.language, engine=args.engine)

        # Save report next to the audio file
        report_dir = Path(args.input).parent
        report_path = save_report(report, report_dir)

        print_summary(report)
        print()
        print_result(result)
        print(f"\nReport saved: {report_path}")

    else:
        print("Provide either expected_text or --manifest")
        sys.exit(1)

    # Exit code: 1 if any flagged
    flagged_count = sum(1 for r in results if r["status"] == "FLAGGED")
    sys.exit(1 if flagged_count > 0 else 0)
