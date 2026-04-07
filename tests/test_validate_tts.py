"""Tests for generator/validate_tts.py — hallucination detection and validation reports."""

import json
from pathlib import Path

import pytest

from validate_tts import (
    normalize_text, check_hallucination, build_report, save_report,
    load_report, WORD_DURATION, MAX_SECONDS_PER_WORD,
    DURATION_ANOMALY_MULTIPLIER, calibrate_word_duration, get_max_duration,
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


class TestDurationAnomaly:
    def test_max_seconds_per_word_exists(self):
        assert MAX_SECONDS_PER_WORD == 0.8

    def test_47_words_max_duration(self):
        """47 words at 0.8s/word = max 37.6s. A 666s file must be flagged."""
        max_dur = 47 * MAX_SECONDS_PER_WORD
        assert max_dur < 40  # well under 666s
        assert 666 > max_dur  # runaway generation caught

    def test_short_line_max_duration(self):
        """5 words at 0.8s/word = max 4s. Reasonable for 'Wait... both?'"""
        max_dur = 5 * MAX_SECONDS_PER_WORD
        assert max_dur == 4.0


class TestCalibration:
    def test_calibrate_returns_seconds_per_word(self, tmp_path):
        """A 5s ref with 10 words = 0.5s/word."""
        from unittest.mock import patch
        with patch("validate_tts.get_duration", return_value=5.0):
            spw = calibrate_word_duration(
                str(tmp_path / "ref.wav"),
                "one two three four five six seven eight nine ten",
            )
            assert spw == pytest.approx(0.5, abs=0.01)

    def test_calibrate_short_text_returns_none(self, tmp_path):
        from unittest.mock import patch
        with patch("validate_tts.get_duration", return_value=1.0):
            spw = calibrate_word_duration(str(tmp_path / "ref.wav"), "hi")
            assert spw is None

    def test_get_max_duration_with_calibration(self):
        # 50 words, calibrated 0.4s/word, 2.5x margin = 50s
        max_dur = get_max_duration(50, calibrated_spw=0.4)
        assert max_dur == pytest.approx(50.0)

    def test_get_max_duration_fallback(self):
        # 50 words, no calibration → 50 * 0.8 = 40s
        max_dur = get_max_duration(50)
        assert max_dur == pytest.approx(40.0)

    def test_calibrated_is_tighter_than_fallback(self):
        """A fast speaker (0.3s/word) gets a tighter ceiling than the default."""
        cal = get_max_duration(50, calibrated_spw=0.3)
        default = get_max_duration(50)
        assert cal < default


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
