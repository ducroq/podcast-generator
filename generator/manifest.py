"""Content-addressed manifest for podcast TTS pipeline.

Parses podcast scripts into a manifest where each line is identified by a
content hash (first 8 chars of SHA-256 of normalized text + speaker). This
decouples file naming from line order — reordering lines in the script does
not invalidate existing TTS audio files.

Corpus-size assumption: designed for podcasts with up to ~1000 lines per
series. At 8 hex chars (32 bits), collision probability is negligible for
this scale. For larger corpora, increase the `length` parameter.

Usage:
    from manifest import parse_script, build_manifest, content_hash

    entries = parse_script("script.txt")
    manifest = build_manifest(entries, audio_dir=Path("tts/"))
    save_manifest(manifest, Path("manifest.json"))
"""

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except ImportError:
    sf = None
    _HAS_SOUNDFILE = False

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1

# Canonical status values for manifest lines
STATUS_MISSING = "missing"
STATUS_EXISTS = "exists"
STATUS_FAILED = "failed"

# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------

# Regex to strip emotion/direction tags like [excited], [dry], [to Zara]
_EMOTION_RE = re.compile(r"\[[^\]]*\]")

# Characters allowed in speaker names for filenames
_SPEAKER_SANITIZE_RE = re.compile(r"[^\w]")


def normalize_text(text):
    """Normalize text for hashing: strip emotions, lowercase, collapse whitespace.

    Note: punctuation including apostrophes is stripped, so "it's" and "its"
    normalize identically. This is intentional — TTS output for both is the
    same audio.
    """
    text = _EMOTION_RE.sub("", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # strip punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def content_hash(text, speaker=None, length=8):
    """SHA-256 hash of normalized text + speaker, truncated to `length` hex chars.

    Including the speaker ensures that Alex saying "Yeah." and Morgan saying
    "Yeah." get different hashes and different audio files.
    """
    normalized = normalize_text(text)
    if speaker:
        normalized = f"{speaker.lower()}:{normalized}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:length]


def _sanitize_speaker(speaker):
    """Sanitize speaker name for use in filenames: only [a-z0-9_]."""
    return _SPEAKER_SANITIZE_RE.sub("_", speaker.lower())


# ---------------------------------------------------------------------------
# Script parsing
# ---------------------------------------------------------------------------

# Default pause durations (can be overridden by config)
DEFAULT_BEAT_PAUSE = 0.5
DEFAULT_LONG_PAUSE = 1.5
DEFAULT_EXTRA_LONG_PAUSE = 2.5


def parse_script(path, beat_pause=DEFAULT_BEAT_PAUSE):
    """Parse a podcast script into a list of entries.

    Returns a list of dicts, each with a "type" key:
        - "line":          spoken dialogue {speaker, text, emotion, hash, section}
        - "section_break": section transition {from_section, to_section}
        - "backchannel":   reaction cue {reactor, bc_type}
        - "pause":         explicit pause {duration}

    Lines are NOT assigned positional indices. Use the list order as the
    canonical sequence. Each line gets a content hash for file naming.

    Duplicate lines (same speaker, same text) get a stable suffix:
    first occurrence → base hash, second → base_hash_2, etc. The suffix
    is visible in the filename for debuggability.
    """
    entries = []
    current_section = None
    # Track hash occurrences to disambiguate identical lines (e.g. two
    # Morgan "...Yeah." lines). Second occurrence gets hash_2, etc.
    hash_counts = {}

    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped or stripped.startswith("=" * 10):
                continue

            # Section headers: all-caps lines like "COLD OPEN", "THE TURN"
            if re.match(r"^[A-Z][A-Z\s\':,\-]+$", stripped):
                if current_section is not None:
                    entries.append({
                        "type": "section_break",
                        "from_section": current_section,
                        "to_section": stripped,
                    })
                current_section = stripped
                continue

            # Bracketed directives: pauses, backchannels, stage directions
            if re.match(r"^\[.*\]$", stripped):
                lower = stripped.lower()

                # Backchannel: [react: speaker type]
                # Open-ended type matching — downstream decides what to do
                react_match = re.match(
                    r"^\[react:\s*(\w+)\s+(\w+)\]$", lower,
                )
                if react_match:
                    entries.append({
                        "type": "backchannel",
                        "reactor": react_match.group(1),
                        "bc_type": react_match.group(2),
                    })
                    continue

                # Structured pause: [pause: N.N] or [pause: Ns]
                timed_match = re.match(
                    r"^\[pause:\s*(\d+(?:\.\d+)?)\s*s?\]$", lower,
                )
                if timed_match:
                    entries.append({
                        "type": "pause",
                        "duration": float(timed_match.group(1)),
                    })
                    continue

                # Natural language pause directives
                if any(w in lower for w in ["pause", "silence", "beat"]):
                    if "two second" in lower or "three second" in lower:
                        duration = DEFAULT_EXTRA_LONG_PAUSE
                    elif "long" in lower:
                        duration = DEFAULT_LONG_PAUSE
                    else:
                        duration = beat_pause
                    entries.append({"type": "pause", "duration": duration})
                # Other bracketed lines (stage directions) are ignored
                logger.debug("Dropping stage direction: %s", stripped)
                continue

            # Dialogue lines: "Speaker: [emotion] text"
            match = re.match(r"(.+?):\s*(?:\[([^\]]*)\]\s*)?(.*)", stripped)
            if match:
                speaker = match.group(1).strip().lower()
                emotion = match.group(2).strip() if match.group(2) else None
                text = _EMOTION_RE.sub("", match.group(3)).strip()
                text = re.sub(r"\s{2,}", " ", text)  # collapse double spaces
                if text:
                    base_hash = content_hash(text, speaker=speaker)
                    # Disambiguate duplicates with stable suffix
                    hash_counts[base_hash] = hash_counts.get(base_hash, 0) + 1
                    if hash_counts[base_hash] == 1:
                        h = base_hash
                    else:
                        h = f"{base_hash}_{hash_counts[base_hash]}"
                    entries.append({
                        "type": "line",
                        "speaker": speaker,
                        "text": text,
                        "emotion": emotion,
                        "hash": h,
                        "section": current_section,
                    })

    return entries


# ---------------------------------------------------------------------------
# Manifest building
# ---------------------------------------------------------------------------


def _line_filename(speaker, h):
    """Generate content-addressed filename: speaker_hash.wav"""
    safe_speaker = _sanitize_speaker(speaker)
    return f"{safe_speaker}_{h}.wav"


def build_manifest(entries, audio_dir=None, script_path=None, episode=None):
    """Build a manifest from parsed script entries.

    Scans `audio_dir` for existing content-addressed audio files and marks
    lines as "exists" or "missing". Non-line entries (pauses, backchannels,
    section breaks) are included in the order array but not in the lines dict.

    Returns:
        {
            "meta": {version, script, episode, generated_at},
            "lines": {hash: {speaker, text, emotion, file, status, duration, ...}},
            "order": [{"type": ..., "hash": ... or other fields}, ...],
        }

    Note: ``duration`` reflects the file state at manifest-build time. It is
    ``None`` for missing files and may be stale if files are regenerated after
    the manifest is built. The mixer should read durations from disk at mix
    time rather than relying solely on this field.
    """
    meta = {
        "version": MANIFEST_VERSION,
        "script": str(script_path) if script_path else None,
        "episode": episode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    lines = {}
    order = []

    for entry in entries:
        if entry["type"] == "line":
            h = entry["hash"]

            if h in lines:
                logger.warning(
                    "Hash collision: %s already in manifest (speaker=%s, text='%s'). "
                    "Second entry will overwrite the first.",
                    h, entry["speaker"], entry["text"][:50],
                )

            filename = _line_filename(entry["speaker"], h)

            # Check if audio exists
            status = STATUS_MISSING
            duration = None
            if audio_dir and (audio_dir / filename).exists():
                status = STATUS_EXISTS
                if _HAS_SOUNDFILE:
                    try:
                        info = sf.info(str(audio_dir / filename))
                        duration = round(info.duration, 2)
                    except Exception as exc:
                        logger.warning(
                            "Could not read duration for %s: %s", filename, exc,
                        )

            lines[h] = {
                "speaker": entry["speaker"],
                "text": entry["text"],
                "emotion": entry["emotion"],
                "file": filename,
                "section": entry["section"],
                "status": status,
                "duration": duration,
                "engine": None,
            }
            order.append({"type": "line", "hash": h})

        elif entry["type"] == "section_break":
            order.append({
                "type": "section_break",
                "from_section": entry["from_section"],
                "to_section": entry["to_section"],
            })

        elif entry["type"] == "backchannel":
            order.append({
                "type": "backchannel",
                "reactor": entry["reactor"],
                "bc_type": entry["bc_type"],
            })

        elif entry["type"] == "pause":
            order.append({
                "type": "pause",
                "duration": entry["duration"],
            })

    return {"meta": meta, "lines": lines, "order": order}


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def save_manifest(manifest, path):
    """Write manifest to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def load_manifest(path):
    """Read manifest from JSON file.

    Validates that the required top-level keys are present.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not all(k in data for k in ("meta", "lines", "order")):
        raise ValueError(
            f"Invalid manifest format in {path}: "
            f"expected dict with 'meta', 'lines', and 'order' keys"
        )
    return data


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def validate_script_for_tts(entries, max_words=35):
    """Check parsed script entries for lines that may produce poor TTS prosody.

    Flags lines that are too long, contain multiple emotional beats, or have
    excessive pauses. This is advisory — flagged lines are not blocked, just
    logged as warnings for manual review.

    Args:
        entries: list of entry dicts from parse_script()
        max_words: word-count threshold for "long" flag (default 35)

    Returns:
        dict with {clean: int, flagged: int, flags: [{speaker, text, emotion, word_count, issues}]}
    """
    flags = []
    clean = 0
    _TRANSITION_WORDS = {"then", "but", "to", "and"}

    for entry in entries:
        if entry["type"] != "line":
            continue

        text = entry["text"]
        word_count = len(text.split())
        emotion = entry.get("emotion")
        issues = []

        # Long lines produce flat prosody
        if word_count > max_words:
            issues.append("long")

        # Multiple emotion beats in a single tag: "amused, then dry"
        if emotion:
            lower_emotion = emotion.lower()
            if "," in lower_emotion:
                parts = [p.strip() for p in lower_emotion.split(",")]
                for part in parts:
                    first_word = part.split()[0] if part.split() else ""
                    if first_word in _TRANSITION_WORDS:
                        issues.append("multi_emotion")
                        break

        # Many pauses (3+ ellipses) — TTS engines struggle with timing
        if text.count("...") >= 3:
            issues.append("many_pauses")

        # Combination: long AND 2+ pauses — strong candidate for splitting
        if "long" in issues and text.count("...") >= 2:
            issues.append("split_candidate")

        if issues:
            flags.append({
                "speaker": entry["speaker"],
                "text": text,
                "emotion": emotion,
                "word_count": word_count,
                "issues": issues,
            })
            logger.warning(
                "TTS flag [%s] %s (%d words): %s",
                entry["speaker"], ", ".join(issues), word_count,
                text[:80] + ("..." if len(text) > 80 else ""),
            )
        else:
            clean += 1

    result = {"clean": clean, "flagged": len(flags), "flags": flags}
    if flags:
        logger.info(
            "Script validation: %d clean, %d flagged", clean, len(flags),
        )
    else:
        logger.info("Script validation: all %d lines clean", clean)
    return result


def line_count(manifest):
    """Count spoken lines in manifest."""
    return len(manifest["lines"])


def missing_lines(manifest):
    """Return list of hashes for lines without audio."""
    return [h for h, info in manifest["lines"].items() if info["status"] == STATUS_MISSING]


def lines_to_generate(manifest):
    """Return full line dicts (with hash) for all lines needing TTS.

    Each dict includes all fields from the manifest plus the "hash" key,
    ready to iterate over for TTS generation.
    """
    return [
        {"hash": h, **info}
        for h, info in manifest["lines"].items()
        if info["status"] == STATUS_MISSING
    ]


def section_names(manifest):
    """Extract ordered list of unique section names from manifest.

    Works for both single-section and multi-section scripts by scanning
    line entries for their section assignments, not just section breaks.
    """
    seen = set()
    names = []
    for entry in manifest["order"]:
        if entry["type"] == "line":
            section = manifest["lines"][entry["hash"]].get("section")
            if section and section not in seen:
                seen.add(section)
                names.append(section)
    return names
