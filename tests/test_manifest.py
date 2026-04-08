"""Tests for generator/manifest.py — content-addressed manifest for podcast TTS pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from manifest import (
    content_hash,
    normalize_text,
    parse_script,
    build_manifest,
    save_manifest,
    load_manifest,
    line_count,
    missing_lines,
    section_names,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SCRIPT = """\
====================
COLD OPEN
====================

[Office meeting room. Fluorescent hum.]

Junior Manager: [earnest] So I've been reading Atomic Habits. Has anyone read Atomic Habits?

[Brief pause.]

Team Member 1: [agreeable] Yeah. Yeah, it's great.

Team Member 2: [flat] I started it.

====================
OPENING
====================

Alex: [quiet recognition] I've been in that meeting. I've been that guy.

Morgan: [dry] We've all been that guy.
[react: zara breath]

Alex: I don't know, I think I was worse than most.

[Silence. Two seconds.]

Alex: Morgan was telling me the other day... she said something about the nod.
[react: morgan laugh]

[Beat.]

Morgan: [amused] You're welcome.
"""


@pytest.fixture
def script_file(tmp_path):
    """Write sample script to a temp file."""
    path = tmp_path / "test_script.txt"
    path.write_text(SAMPLE_SCRIPT, encoding="utf-8")
    return path


@pytest.fixture
def audio_dir_with_files(tmp_path):
    """Create an audio dir with one matching WAV file."""
    audio_dir = tmp_path / "tts"
    audio_dir.mkdir()
    # Create a short WAV for the first line
    text = "So I've been reading Atomic Habits. Has anyone read Atomic Habits?"
    h = content_hash(text, speaker="junior manager")
    filename = f"junior_manager_{h}.wav"
    audio = np.zeros(24000, dtype=np.float32)  # 1s silence at 24kHz
    sf.write(str(audio_dir / filename), audio, 24000)
    return audio_dir


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_deterministic(self):
        """Same text produces same hash every time."""
        text = "We bought Good to Great in bulk."
        assert content_hash(text) == content_hash(text)

    def test_different_text_different_hash(self):
        assert content_hash("Hello world") != content_hash("Goodbye world")

    def test_ignores_emotion_tags(self):
        """[excited] Hello and [calm] Hello produce the same hash."""
        assert content_hash("[excited] Hello world", speaker="alex") == content_hash("[calm] Hello world", speaker="alex")

    def test_ignores_no_emotion(self):
        """Text with and without emotion tag produce the same hash."""
        assert content_hash("[dry] We've all been that guy.", speaker="morgan") == content_hash("We've all been that guy.", speaker="morgan")

    def test_different_speaker_different_hash(self):
        """Same text, different speaker → different hash."""
        assert content_hash("Yeah.", speaker="alex") != content_hash("Yeah.", speaker="morgan")

    def test_ignores_whitespace(self):
        """Extra spaces don't change hash."""
        assert content_hash("hello  world") == content_hash("hello world")

    def test_ignores_case(self):
        assert content_hash("Hello World") == content_hash("hello world")

    def test_ignores_punctuation(self):
        assert content_hash("Hello, world!") == content_hash("Hello world")

    def test_length_default(self):
        """Default hash is 8 hex chars."""
        h = content_hash("test")
        assert len(h) == 8
        assert all(c in "0123456789abcdef" for c in h)

    def test_length_custom(self):
        h = content_hash("test", length=12)
        assert len(h) == 12


class TestNormalizeText:
    def test_strips_emotion_tags(self):
        assert normalize_text("[excited] Hello!") == "hello"

    def test_strips_multiple_tags(self):
        assert normalize_text("[to Zara] [quiet] Hello") == "hello"

    def test_collapses_whitespace(self):
        assert normalize_text("hello   world") == "hello world"

    def test_strips_punctuation(self):
        assert normalize_text("Hello, world! How's it going?") == "hello world hows it going"


# ---------------------------------------------------------------------------
# Script parsing
# ---------------------------------------------------------------------------


class TestParseScript:
    def test_line_count(self, script_file):
        """Correct number of spoken lines parsed."""
        entries = parse_script(script_file)
        lines = [e for e in entries if e["type"] == "line"]
        assert len(lines) == 8

    def test_speakers(self, script_file):
        entries = parse_script(script_file)
        lines = [e for e in entries if e["type"] == "line"]
        speakers = [l["speaker"] for l in lines]
        assert "junior manager" in speakers
        assert "alex" in speakers
        assert "morgan" in speakers

    def test_emotion_parsed(self, script_file):
        entries = parse_script(script_file)
        lines = [e for e in entries if e["type"] == "line"]
        # First line has [earnest]
        assert lines[0]["emotion"] == "earnest"
        # Alex's third line has no emotion
        no_emotion = [l for l in lines if l["text"].startswith("I don't know")]
        assert len(no_emotion) == 1
        assert no_emotion[0]["emotion"] is None

    def test_each_line_has_hash(self, script_file):
        entries = parse_script(script_file)
        lines = [e for e in entries if e["type"] == "line"]
        for line in lines:
            assert "hash" in line
            assert len(line["hash"]) == 8

    def test_sections_detected(self, script_file):
        entries = parse_script(script_file)
        breaks = [e for e in entries if e["type"] == "section_break"]
        assert len(breaks) == 1
        assert breaks[0]["from_section"] == "COLD OPEN"
        assert breaks[0]["to_section"] == "OPENING"

    def test_section_assigned_to_lines(self, script_file):
        entries = parse_script(script_file)
        lines = [e for e in entries if e["type"] == "line"]
        # First 3 lines are in COLD OPEN
        assert lines[0]["section"] == "COLD OPEN"
        assert lines[2]["section"] == "COLD OPEN"
        # Lines after section break are in OPENING
        assert lines[3]["section"] == "OPENING"

    def test_react_cues_parsed(self, script_file):
        entries = parse_script(script_file)
        bcs = [e for e in entries if e["type"] == "backchannel"]
        assert len(bcs) == 2
        assert bcs[0]["reactor"] == "zara"
        assert bcs[0]["bc_type"] == "breath"
        assert bcs[1]["reactor"] == "morgan"
        assert bcs[1]["bc_type"] == "laugh"

    def test_pauses_parsed(self, script_file):
        entries = parse_script(script_file)
        pauses = [e for e in entries if e["type"] == "pause"]
        assert len(pauses) == 3  # [Brief pause.], [Silence. Two seconds.], [Beat.]
        durations = sorted([p["duration"] for p in pauses])
        # [Brief pause.] = 0.5, [Beat.] = 0.5, [Silence. Two seconds.] = 2.5
        assert durations == [0.5, 0.5, 2.5]

    def test_stage_directions_ignored(self, script_file):
        """Bracketed stage directions that aren't pauses or reacts are skipped."""
        entries = parse_script(script_file)
        types = {e["type"] for e in entries}
        assert types == {"line", "section_break", "backchannel", "pause"}

    def test_order_preserved(self, script_file):
        """Entries are in script order."""
        entries = parse_script(script_file)
        # First entry is a line (Junior Manager), then pause, then lines,
        # then section break, then more lines with BCs and pauses interspersed
        assert entries[0]["type"] == "line"
        assert entries[0]["speaker"] == "junior manager"


# ---------------------------------------------------------------------------
# Manifest building
# ---------------------------------------------------------------------------


class TestBuildManifest:
    def test_empty_dir(self, script_file, tmp_path):
        """No audio files → all lines status 'missing'."""
        entries = parse_script(script_file)
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        manifest = build_manifest(entries, audio_dir=empty_dir)
        assert all(info["status"] == "missing" for info in manifest["lines"].values())

    def test_finds_existing(self, script_file, audio_dir_with_files):
        """Existing content-addressed file → status 'exists'."""
        entries = parse_script(script_file)
        manifest = build_manifest(entries, audio_dir=audio_dir_with_files)
        # The first line (Junior Manager) has a matching file
        first_line = [e for e in entries if e["type"] == "line"][0]
        h = first_line["hash"]
        assert manifest["lines"][h]["status"] == "exists"
        assert manifest["lines"][h]["duration"] is not None

    def test_missing_lines_helper(self, script_file, audio_dir_with_files):
        entries = parse_script(script_file)
        manifest = build_manifest(entries, audio_dir=audio_dir_with_files)
        missing = missing_lines(manifest)
        # 8 lines total, 1 exists → 7 missing
        assert len(missing) == 7

    def test_line_count_helper(self, script_file):
        entries = parse_script(script_file)
        manifest = build_manifest(entries)
        assert line_count(manifest) == 8

    def test_order_includes_all_types(self, script_file):
        entries = parse_script(script_file)
        manifest = build_manifest(entries)
        types_in_order = {e["type"] for e in manifest["order"]}
        assert "line" in types_in_order
        assert "section_break" in types_in_order
        assert "backchannel" in types_in_order
        assert "pause" in types_in_order

    def test_order_lines_reference_hashes(self, script_file):
        entries = parse_script(script_file)
        manifest = build_manifest(entries)
        line_orders = [e for e in manifest["order"] if e["type"] == "line"]
        for lo in line_orders:
            assert lo["hash"] in manifest["lines"]

    def test_no_audio_dir(self, script_file):
        """Building without audio_dir → all missing, no crash."""
        entries = parse_script(script_file)
        manifest = build_manifest(entries, audio_dir=None)
        assert all(info["status"] == "missing" for info in manifest["lines"].values())

    def test_section_names_helper(self, script_file):
        entries = parse_script(script_file)
        manifest = build_manifest(entries)
        names = section_names(manifest)
        assert names == ["COLD OPEN", "OPENING"]


class TestManifestOrderSurvivesReorder:
    def test_reorder_preserves_hashes(self, tmp_path):
        """Swapping two lines → same hashes, different order."""
        script_v1 = tmp_path / "v1.txt"
        script_v2 = tmp_path / "v2.txt"

        script_v1.write_text(
            "====================\nSECTION\n====================\n\n"
            "Alex: First line.\n\nMorgan: Second line.\n",
            encoding="utf-8",
        )
        script_v2.write_text(
            "====================\nSECTION\n====================\n\n"
            "Morgan: Second line.\n\nAlex: First line.\n",
            encoding="utf-8",
        )

        entries_v1 = parse_script(script_v1)
        entries_v2 = parse_script(script_v2)
        m1 = build_manifest(entries_v1)
        m2 = build_manifest(entries_v2)

        # Same hashes in lines dict
        assert set(m1["lines"].keys()) == set(m2["lines"].keys())

        # Different order
        order_v1 = [e["hash"] for e in m1["order"] if e["type"] == "line"]
        order_v2 = [e["hash"] for e in m2["order"] if e["type"] == "line"]
        assert order_v1 != order_v2
        assert set(order_v1) == set(order_v2)


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


class TestDuplicateLines:
    def test_same_speaker_same_text_unique_hashes(self, tmp_path):
        """Two identical lines from the same speaker get unique hashes."""
        script = tmp_path / "dupes.txt"
        script.write_text(
            "====================\nSECTION\n====================\n\n"
            "Morgan: ...Yeah.\n\nAlex: Something else.\n\nMorgan: ...Yeah.\n",
            encoding="utf-8",
        )
        entries = parse_script(script)
        lines = [e for e in entries if e["type"] == "line"]
        hashes = [l["hash"] for l in lines]
        # All 3 hashes should be unique
        assert len(set(hashes)) == 3

    def test_first_occurrence_gets_base_hash(self, tmp_path):
        """First occurrence of a line gets the base hash (no suffix)."""
        script = tmp_path / "dupes.txt"
        script.write_text(
            "====================\nSECTION\n====================\n\n"
            "Morgan: ...Yeah.\n\nMorgan: ...Yeah.\n",
            encoding="utf-8",
        )
        entries = parse_script(script)
        lines = [e for e in entries if e["type"] == "line"]
        # First occurrence should match base hash
        base = content_hash("...Yeah.", speaker="morgan")
        assert lines[0]["hash"] == base
        # Second should be different
        assert lines[1]["hash"] != base


class TestManifestIO:
    def test_roundtrip(self, script_file, tmp_path):
        """Write manifest to JSON, read back, identical."""
        entries = parse_script(script_file)
        manifest = build_manifest(entries)

        path = tmp_path / "manifest.json"
        save_manifest(manifest, path)
        loaded = load_manifest(path)

        assert loaded["lines"] == manifest["lines"]
        assert loaded["order"] == manifest["order"]

    def test_json_valid(self, script_file, tmp_path):
        """Output is valid JSON."""
        entries = parse_script(script_file)
        manifest = build_manifest(entries)

        path = tmp_path / "manifest.json"
        save_manifest(manifest, path)

        with open(path) as f:
            data = json.load(f)
        assert "lines" in data
        assert "order" in data
