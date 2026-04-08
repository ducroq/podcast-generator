"""Content-addressed manifest for podcast TTS pipeline.

Parses podcast scripts into a manifest where each line is identified by a
content hash (first 8 chars of SHA-256 of normalized text). This decouples
file naming from line order — reordering lines in the script does not
invalidate existing TTS audio files.

Usage:
    from manifest import parse_script, build_manifest, content_hash

    entries = parse_script("script.txt")
    manifest = build_manifest(entries, audio_dir=Path("tts/"))
    save_manifest(manifest, Path("manifest.json"))
"""

import hashlib
import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------

# Regex to strip emotion/direction tags like [excited], [dry], [to Zara]
_EMOTION_RE = re.compile(r"\[[^\]]*\]")


def normalize_text(text):
    """Normalize text for hashing: strip emotions, lowercase, collapse whitespace."""
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
    """
    entries = []
    current_section = None
    # Track hash occurrences to disambiguate identical lines (e.g. two
    # Morgan "...Yeah." lines). Second occurrence gets hash + "_2", etc.
    hash_counts = {}

    for raw in open(path, encoding="utf-8"):
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
            react_match = re.match(
                r"^\[react:\s*(\w+)\s+(laugh|breath)\]$", lower,
            )
            if react_match:
                entries.append({
                    "type": "backchannel",
                    "reactor": react_match.group(1),
                    "bc_type": react_match.group(2),
                })
                continue

            # Pause directives
            if any(w in lower for w in ["pause", "silence", "beat"]):
                if "two second" in lower or "three second" in lower:
                    duration = DEFAULT_EXTRA_LONG_PAUSE
                elif "long" in lower:
                    duration = DEFAULT_LONG_PAUSE
                else:
                    duration = beat_pause
                entries.append({"type": "pause", "duration": duration})
            # Other bracketed lines (stage directions) are ignored
            continue

        # Dialogue lines: "Speaker: [emotion] text"
        match = re.match(r"(.+?):\s*(?:\[([^\]]*)\]\s*)?(.*)", stripped)
        if match:
            speaker = match.group(1).strip().lower()
            emotion = match.group(2).strip() if match.group(2) else None
            text = match.group(3).strip()
            if text:
                base_hash = content_hash(text, speaker=speaker)
                # Disambiguate duplicates (same speaker, same text)
                hash_counts[base_hash] = hash_counts.get(base_hash, 0) + 1
                if hash_counts[base_hash] == 1:
                    h = base_hash
                else:
                    # Append occurrence number to create unique hash
                    suffix = str(hash_counts[base_hash])
                    h = content_hash(text + suffix, speaker=speaker)
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
    return f"{speaker.replace(' ', '_')}_{h}.wav"


def build_manifest(entries, audio_dir=None):
    """Build a manifest from parsed script entries.

    Scans `audio_dir` for existing content-addressed audio files and marks
    lines as "exists" or "missing". Non-line entries (pauses, backchannels,
    section breaks) are included in the order array but not in the lines dict.

    Returns:
        {
            "lines": {hash: {speaker, text, emotion, file, status, ...}},
            "order": [{"type": ..., "hash": ... or other fields}, ...],
        }
    """
    lines = {}
    order = []

    for entry in entries:
        if entry["type"] == "line":
            h = entry["hash"]
            filename = _line_filename(entry["speaker"], h)

            # Check if audio exists
            status = "missing"
            duration = None
            if audio_dir and (audio_dir / filename).exists():
                status = "exists"
                # Try to get duration without importing soundfile at module level
                try:
                    import soundfile as sf
                    info = sf.info(str(audio_dir / filename))
                    duration = round(info.duration, 2)
                except Exception:
                    pass

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

    return {"lines": lines, "order": order}


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def save_manifest(manifest, path):
    """Write manifest to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def load_manifest(path):
    """Read manifest from JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def line_count(manifest):
    """Count spoken lines in manifest."""
    return len(manifest["lines"])


def missing_lines(manifest):
    """Return list of hashes for lines without audio."""
    return [h for h, info in manifest["lines"].items() if info["status"] == "missing"]


def section_names(manifest):
    """Extract ordered list of section names from manifest."""
    names = []
    for entry in manifest["order"]:
        if entry["type"] == "section_break":
            if not names:
                names.append(entry["from_section"])
            names.append(entry["to_section"])
    return names
