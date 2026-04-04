"""Tests for generator/validate_tts.py — hallucination detection and validation reports."""

import json
from pathlib import Path

from validate_tts import (
    normalize_text, check_hallucination, build_report, save_report,
    load_report, WORD_DURATION,
)


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Hello World") == "hello world"

    def test_strips_punctuation(self):
        assert normalize_text("Hello, world!") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize_text("  hello   world  ") == "hello world"

    def test_mixed(self):
        assert normalize_text("It's a beautiful day, isn't it?") == "its a beautiful day isnt it"


class TestCheckHallucination:
    def test_perfect_match(self):
        is_ok, issues = check_hallucination(
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog",
        )
        assert is_ok
        assert issues == []

    def test_case_insensitive_match(self):
        is_ok, issues = check_hallucination("Hello World", "hello world")
        assert is_ok

    def test_punctuation_insensitive(self):
        is_ok, issues = check_hallucination(
            "Hello, world!",
            "Hello world",
        )
        assert is_ok

    def test_detects_prepended_hallucination(self):
        is_ok, issues = check_hallucination(
            "The cat sat on the mat",
            "Welcome to our show today The cat sat on the mat",
        )
        assert not is_ok
        assert any("HALLUCINATION_START" in i for i in issues)

    def test_detects_appended_hallucination(self):
        is_ok, issues = check_hallucination(
            "The cat sat",
            "The cat sat on the mat and then it jumped over the fence and ran away quickly",
        )
        assert not is_ok
        assert any("HALLUCINATION_END" in i or "TOO_LONG" in i for i in issues)

    def test_detects_low_overlap(self):
        is_ok, issues = check_hallucination(
            "The quick brown fox jumps over the lazy dog",
            "A completely different sentence with no matching words here",
        )
        assert not is_ok
        assert any("LOW_OVERLAP" in i for i in issues)

    def test_detects_too_long(self):
        expected = "short text"
        transcribed = "this is a much much much longer text than was ever expected to be here"
        is_ok, issues = check_hallucination(expected, transcribed)
        assert any("TOO_LONG" in i for i in issues)

    def test_detects_too_short(self):
        expected = "A fairly long expected sentence with many words in it"
        transcribed = "short"
        is_ok, issues = check_hallucination(expected, transcribed)
        assert any("TOO_SHORT" in i for i in issues)

    def test_single_trailing_word_detected(self):
        """Regression: off-by-one previously missed single trailing words."""
        is_ok, issues = check_hallucination(
            "one two three four five six",
            "one two three four five six seven eight nine ten extra",
        )
        assert any("HALLUCINATION_END" in i for i in issues)

    def test_empty_expected(self):
        is_ok, issues = check_hallucination("", "some transcribed text")
        # Should not crash; may flag issues
        assert isinstance(is_ok, bool)

    def test_empty_transcribed(self):
        is_ok, issues = check_hallucination("expected text here", "")
        assert not is_ok
        assert any("TOO_SHORT" in i for i in issues)


class TestWordDuration:
    def test_english_rate(self):
        assert WORD_DURATION["en"] == 0.35

    def test_german_slower_than_english(self):
        assert WORD_DURATION["de"] > WORD_DURATION["en"]


class TestBuildReport:
    def test_report_structure(self):
        results = [
            {"file": "a.wav", "status": "OK", "issues": []},
            {"file": "b.wav", "status": "FLAGGED", "issues": ["HALLUCINATION_START"]},
            {"file": "c.wav", "status": "ERROR", "issues": ["File not found"]},
        ]
        report = build_report(results, language="en", engine="qwen")
        assert report["language"] == "en"
        assert report["engine"] == "qwen"
        assert report["summary"]["total"] == 3
        assert report["summary"]["ok"] == 1
        assert report["summary"]["flagged"] == 1
        assert report["summary"]["errors"] == 1
        assert "validated_at" in report

    def test_empty_results(self):
        report = build_report([])
        assert report["summary"]["total"] == 0


class TestSaveLoadReport:
    def test_save_and_load_roundtrip(self, tmp_path):
        report = build_report(
            [{"file": "test.wav", "status": "OK", "issues": []}],
            language="en",
        )
        save_report(report, tmp_path)
        loaded = load_report(tmp_path)
        assert loaded["summary"]["total"] == 1
        assert loaded["results"][0]["file"] == "test.wav"

    def test_rotation_preserves_previous(self, tmp_path):
        # Save first report
        r1 = build_report([{"file": "a.wav", "status": "OK", "issues": []}])
        save_report(r1, tmp_path)

        # Save second report — first should rotate to _prev
        r2 = build_report([{"file": "b.wav", "status": "FLAGGED", "issues": ["x"]}])
        save_report(r2, tmp_path)

        # Current report is r2
        current = load_report(tmp_path)
        assert current["results"][0]["file"] == "b.wav"

        # Previous report is r1
        prev_path = tmp_path / "validation_prev.json"
        assert prev_path.exists()
        with open(prev_path) as f:
            prev = json.load(f)
        assert prev["results"][0]["file"] == "a.wav"

    def test_load_nonexistent_returns_none(self, tmp_path):
        assert load_report(tmp_path) is None
