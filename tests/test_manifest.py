"""Tests for generator/manifest.py — content-addressed manifest for podcast TTS pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from manifest import (
    MANIFEST_VERSION,
    content_hash,
    normalize_text,
    parse_script,
    build_manifest,
    save_manifest,
    load_manifest,
    line_count,
    missing_lines,
    lines_to_generate,
    section_names,
    validate_script_for_tts,
    _sanitize_speaker,
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
        """Same text, different speaker -> different hash."""
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


class TestSanitizeSpeaker:
    def test_spaces_replaced(self):
        assert _sanitize_speaker("junior manager") == "junior_manager"

    def test_slashes_replaced(self):
        assert _sanitize_speaker("speaker/evil") == "speaker_evil"

    def test_dots_replaced(self):
        assert _sanitize_speaker("..") == "__"

    def test_normal_name_unchanged(self):
        assert _sanitize_speaker("alex") == "alex"

    def test_uppercase_lowered(self):
        assert _sanitize_speaker("Alex") == "alex"


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
        assert lines[0]["emotion"] == "earnest"
        no_emotion = [l for l in lines if l["text"].startswith("I don't know")]
        assert len(no_emotion) == 1
        assert no_emotion[0]["emotion"] is None

    def test_each_line_has_hash(self, script_file):
        entries = parse_script(script_file)
        lines = [e for e in entries if e["type"] == "line"]
        for line in lines:
            assert "hash" in line
            # Base hash is 8 chars; duplicates get _N suffix (e.g. 10+ chars)
            base = line["hash"].split("_")[0] if "_" in line["hash"] else line["hash"]
            assert len(base) == 8
            assert all(c in "0123456789abcdef" for c in base)

    def test_sections_detected(self, script_file):
        entries = parse_script(script_file)
        breaks = [e for e in entries if e["type"] == "section_break"]
        assert len(breaks) == 1
        assert breaks[0]["from_section"] == "COLD OPEN"
        assert breaks[0]["to_section"] == "OPENING"

    def test_section_assigned_to_lines(self, script_file):
        entries = parse_script(script_file)
        lines = [e for e in entries if e["type"] == "line"]
        assert lines[0]["section"] == "COLD OPEN"
        assert lines[2]["section"] == "COLD OPEN"
        assert lines[3]["section"] == "OPENING"

    def test_react_cues_parsed(self, script_file):
        entries = parse_script(script_file)
        bcs = [e for e in entries if e["type"] == "backchannel"]
        assert len(bcs) == 2
        assert bcs[0]["reactor"] == "zara"
        assert bcs[0]["bc_type"] == "breath"
        assert bcs[1]["reactor"] == "morgan"
        assert bcs[1]["bc_type"] == "laugh"

    def test_react_cue_open_type(self, tmp_path):
        """Backchannel types beyond laugh/breath are accepted."""
        script = tmp_path / "bc.txt"
        script.write_text(
            "====================\nS\n====================\n\n"
            "Alex: Hello.\n[react: morgan sigh]\n[react: zara hmm]\n",
            encoding="utf-8",
        )
        entries = parse_script(script)
        bcs = [e for e in entries if e["type"] == "backchannel"]
        assert len(bcs) == 2
        assert bcs[0]["bc_type"] == "sigh"
        assert bcs[1]["bc_type"] == "hmm"

    def test_pauses_parsed(self, script_file):
        entries = parse_script(script_file)
        pauses = [e for e in entries if e["type"] == "pause"]
        assert len(pauses) == 3
        durations = sorted([p["duration"] for p in pauses])
        assert durations == [0.5, 0.5, 2.5]

    def test_structured_pause(self, tmp_path):
        """[pause: N.N] syntax is parsed."""
        script = tmp_path / "pause.txt"
        script.write_text(
            "====================\nS\n====================\n\n"
            "Alex: Hello.\n[pause: 1.5]\nMorgan: Hi.\n[pause: 0.3s]\n",
            encoding="utf-8",
        )
        entries = parse_script(script)
        pauses = [e for e in entries if e["type"] == "pause"]
        assert len(pauses) == 2
        assert pauses[0]["duration"] == 1.5
        assert pauses[1]["duration"] == 0.3

    def test_stage_directions_ignored(self, script_file):
        entries = parse_script(script_file)
        types = {e["type"] for e in entries}
        assert types == {"line", "section_break", "backchannel", "pause"}

    def test_order_preserved(self, script_file):
        entries = parse_script(script_file)
        assert entries[0]["type"] == "line"
        assert entries[0]["speaker"] == "junior manager"

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_script(tmp_path / "does_not_exist.txt")


# ---------------------------------------------------------------------------
# Single-section scripts
# ---------------------------------------------------------------------------


class TestSingleSection:
    def test_section_names_single_section(self, tmp_path):
        """Script with one section (no section_break) still reports that section."""
        script = tmp_path / "single.txt"
        script.write_text(
            "====================\nONLY SECTION\n====================\n\n"
            "Alex: Hello.\nMorgan: Hi.\n",
            encoding="utf-8",
        )
        entries = parse_script(script)
        manifest = build_manifest(entries)
        names = section_names(manifest)
        assert names == ["ONLY SECTION"]

    def test_no_section_at_all(self, tmp_path):
        """Script without any section header."""
        script = tmp_path / "nosec.txt"
        script.write_text("Alex: Hello.\nMorgan: Hi.\n", encoding="utf-8")
        entries = parse_script(script)
        manifest = build_manifest(entries)
        names = section_names(manifest)
        assert names == []


# ---------------------------------------------------------------------------
# Duplicate line handling
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
        assert len(set(hashes)) == 3

    def test_first_occurrence_gets_base_hash(self, tmp_path):
        """First occurrence gets base hash, second gets hash_2."""
        script = tmp_path / "dupes.txt"
        script.write_text(
            "====================\nSECTION\n====================\n\n"
            "Morgan: ...Yeah.\n\nMorgan: ...Yeah.\n",
            encoding="utf-8",
        )
        entries = parse_script(script)
        lines = [e for e in entries if e["type"] == "line"]
        base = content_hash("...Yeah.", speaker="morgan")
        assert lines[0]["hash"] == base
        assert lines[1]["hash"] == f"{base}_2"

    def test_suffix_visible_in_filename(self, tmp_path):
        """Duplicate hash suffix is visible in the generated filename."""
        script = tmp_path / "dupes.txt"
        script.write_text(
            "====================\nSECTION\n====================\n\n"
            "Morgan: ...Yeah.\n\nMorgan: ...Yeah.\n",
            encoding="utf-8",
        )
        entries = parse_script(script)
        manifest = build_manifest(entries)
        lines = list(manifest["lines"].values())
        filenames = [l["file"] for l in lines]
        assert any("_2" in f for f in filenames)


# ---------------------------------------------------------------------------
# Manifest building
# ---------------------------------------------------------------------------


class TestBuildManifest:
    def test_empty_dir(self, script_file, tmp_path):
        entries = parse_script(script_file)
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        manifest = build_manifest(entries, audio_dir=empty_dir)
        assert all(info["status"] == "missing" for info in manifest["lines"].values())

    def test_finds_existing(self, script_file, audio_dir_with_files):
        entries = parse_script(script_file)
        manifest = build_manifest(entries, audio_dir=audio_dir_with_files)
        first_line = [e for e in entries if e["type"] == "line"][0]
        h = first_line["hash"]
        assert manifest["lines"][h]["status"] == "exists"
        assert manifest["lines"][h]["duration"] is not None

    def test_missing_lines_helper(self, script_file, audio_dir_with_files):
        entries = parse_script(script_file)
        manifest = build_manifest(entries, audio_dir=audio_dir_with_files)
        missing = missing_lines(manifest)
        assert len(missing) == 7

    def test_lines_to_generate_helper(self, script_file, audio_dir_with_files):
        entries = parse_script(script_file)
        manifest = build_manifest(entries, audio_dir=audio_dir_with_files)
        to_gen = lines_to_generate(manifest)
        assert len(to_gen) == 7
        # Each entry has hash, speaker, text, file
        for entry in to_gen:
            assert "hash" in entry
            assert "speaker" in entry
            assert "text" in entry
            assert "file" in entry

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
        entries = parse_script(script_file)
        manifest = build_manifest(entries, audio_dir=None)
        assert all(info["status"] == "missing" for info in manifest["lines"].values())

    def test_section_names_helper(self, script_file):
        entries = parse_script(script_file)
        manifest = build_manifest(entries)
        names = section_names(manifest)
        assert names == ["COLD OPEN", "OPENING"]

    def test_meta_block_present(self, script_file):
        entries = parse_script(script_file)
        manifest = build_manifest(entries, script_path="test.txt", episode="ep01")
        assert "meta" in manifest
        assert manifest["meta"]["version"] == MANIFEST_VERSION
        assert manifest["meta"]["script"] == "test.txt"
        assert manifest["meta"]["episode"] == "ep01"
        assert manifest["meta"]["generated_at"] is not None

    def test_sanitized_filenames(self, tmp_path):
        """Speaker names with special chars produce safe filenames."""
        script = tmp_path / "special.txt"
        script.write_text(
            "====================\nS\n====================\n\n"
            "Team Member 1: Hello.\n",
            encoding="utf-8",
        )
        entries = parse_script(script)
        manifest = build_manifest(entries)
        line = list(manifest["lines"].values())[0]
        assert "/" not in line["file"]
        assert "\\" not in line["file"]
        assert ".." not in line["file"].split("_")[0]


class TestManifestOrderSurvivesReorder:
    def test_reorder_preserves_hashes(self, tmp_path):
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
        m1 = build_manifest(parse_script(script_v1))
        m2 = build_manifest(parse_script(script_v2))
        assert set(m1["lines"].keys()) == set(m2["lines"].keys())
        order_v1 = [e["hash"] for e in m1["order"] if e["type"] == "line"]
        order_v2 = [e["hash"] for e in m2["order"] if e["type"] == "line"]
        assert order_v1 != order_v2
        assert set(order_v1) == set(order_v2)


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


class TestManifestIO:
    def test_roundtrip(self, script_file, tmp_path):
        entries = parse_script(script_file)
        manifest = build_manifest(entries)
        path = tmp_path / "manifest.json"
        save_manifest(manifest, path)
        loaded = load_manifest(path)
        assert loaded["lines"] == manifest["lines"]
        assert loaded["order"] == manifest["order"]
        assert loaded["meta"]["version"] == MANIFEST_VERSION

    def test_json_valid(self, script_file, tmp_path):
        entries = parse_script(script_file)
        manifest = build_manifest(entries)
        path = tmp_path / "manifest.json"
        save_manifest(manifest, path)
        with open(path) as f:
            data = json.load(f)
        assert "lines" in data
        assert "order" in data
        assert "meta" in data

    def test_load_invalid_format_raises(self, tmp_path):
        """Loading a non-manifest JSON raises ValueError."""
        path = tmp_path / "bad.json"
        path.write_text('{"foo": "bar"}', encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid manifest format"):
            load_manifest(path)

    def test_load_missing_meta_raises(self, tmp_path):
        """Loading manifest without meta key raises ValueError."""
        path = tmp_path / "nometa.json"
        path.write_text('{"lines": {}, "order": []}', encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid manifest format"):
            load_manifest(path)

    def test_load_legacy_list_raises(self, tmp_path):
        """Loading old flat-list manifest raises ValueError."""
        path = tmp_path / "legacy.json"
        path.write_text('[{"file": "test.wav"}]', encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid manifest format"):
            load_manifest(path)


# ---------------------------------------------------------------------------
# TTS script validation
# ---------------------------------------------------------------------------


class TestValidateScriptForTts:
    def test_clean_lines_not_flagged(self):
        """Short lines with simple emotions pass cleanly."""
        entries = [
            {"type": "line", "speaker": "alex", "text": "Yeah, that makes sense.", "emotion": "thoughtful"},
            {"type": "line", "speaker": "morgan", "text": "I agree.", "emotion": "dry"},
        ]
        result = validate_script_for_tts(entries)
        assert result["clean"] == 2
        assert result["flagged"] == 0
        assert result["flags"] == []

    def test_long_line_flagged(self):
        """Lines exceeding max_words are flagged as long."""
        long_text = " ".join(["word"] * 40)
        entries = [
            {"type": "line", "speaker": "alex", "text": long_text, "emotion": None},
        ]
        result = validate_script_for_tts(entries, max_words=35)
        assert result["flagged"] == 1
        assert "long" in result["flags"][0]["issues"]
        assert result["flags"][0]["word_count"] == 40

    def test_custom_max_words(self):
        """Custom max_words threshold is respected."""
        text = " ".join(["word"] * 20)
        entries = [
            {"type": "line", "speaker": "alex", "text": text, "emotion": None},
        ]
        # Should flag at 15, not at 25
        result_flag = validate_script_for_tts(entries, max_words=15)
        result_clean = validate_script_for_tts(entries, max_words=25)
        assert result_flag["flagged"] == 1
        assert result_clean["flagged"] == 0

    def test_multi_emotion_flagged(self):
        """Emotion tags with comma + transition word are flagged."""
        entries = [
            {"type": "line", "speaker": "morgan", "text": "Sure.", "emotion": "amused, then dry"},
        ]
        result = validate_script_for_tts(entries)
        assert result["flagged"] == 1
        assert "multi_emotion" in result["flags"][0]["issues"]

    def test_multi_emotion_variants(self):
        """Various transition words trigger multi_emotion."""
        for transition in ["but", "to", "and", "then"]:
            entries = [
                {"type": "line", "speaker": "alex", "text": "Ok.",
                 "emotion": f"warm, {transition} sharp"},
            ]
            result = validate_script_for_tts(entries)
            assert result["flagged"] == 1, f"transition '{transition}' not caught"
            assert "multi_emotion" in result["flags"][0]["issues"]

    def test_simple_comma_emotion_not_flagged(self):
        """Comma without transition word is not flagged (e.g. 'warm, quiet')."""
        entries = [
            {"type": "line", "speaker": "alex", "text": "Ok.", "emotion": "warm, quiet"},
        ]
        result = validate_script_for_tts(entries)
        assert result["flagged"] == 0

    def test_many_pauses_flagged(self):
        """Lines with 3+ ellipses are flagged."""
        entries = [
            {"type": "line", "speaker": "zara", "text": "Well... I mean... it's just... yeah.", "emotion": None},
        ]
        result = validate_script_for_tts(entries)
        assert result["flagged"] == 1
        assert "many_pauses" in result["flags"][0]["issues"]

    def test_split_candidate_flagged(self):
        """Long line with 2+ pauses gets split_candidate."""
        long_with_pauses = " ".join(["word"] * 36) + "... something... else"
        entries = [
            {"type": "line", "speaker": "alex", "text": long_with_pauses, "emotion": None},
        ]
        result = validate_script_for_tts(entries, max_words=35)
        assert result["flagged"] == 1
        issues = result["flags"][0]["issues"]
        assert "long" in issues
        assert "split_candidate" in issues

    def test_non_line_entries_ignored(self):
        """Pauses, backchannels, section breaks are skipped."""
        entries = [
            {"type": "pause", "duration": 1.5},
            {"type": "backchannel", "reactor": "zara", "bc_type": "laugh"},
            {"type": "section_break", "from_section": "A", "to_section": "B"},
        ]
        result = validate_script_for_tts(entries)
        assert result["clean"] == 0
        assert result["flagged"] == 0

    def test_no_emotion_not_flagged_as_multi(self):
        """Lines with no emotion tag don't trigger multi_emotion."""
        entries = [
            {"type": "line", "speaker": "alex", "text": "Hello.", "emotion": None},
        ]
        result = validate_script_for_tts(entries)
        assert result["flagged"] == 0

    def test_with_parsed_script(self, script_file):
        """Validation works on real parse_script output."""
        entries = parse_script(script_file)
        result = validate_script_for_tts(entries)
        # SAMPLE_SCRIPT has short lines, so all should be clean
        assert result["clean"] + result["flagged"] > 0
        assert isinstance(result["flags"], list)
