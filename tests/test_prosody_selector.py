"""Tests for generator/prosody_selector.py — emotion-to-ref mapping."""

import json
from pathlib import Path

import pytest

from prosody_selector import (
    ProsodySelector,
    EMOTION_MAP,
    DEFAULT_PROSODY,
)


@pytest.fixture
def manifest(tmp_path):
    """Create a test prosody manifest."""
    data = {
        "emma": {
            "excited": {"file": "/refs/emma_excited.wav", "text": "Wow!"},
            "calm": {"file": "/refs/emma_calm.wav", "text": "Calm."},
            "emphatic": {"file": "/refs/emma_emphatic.wav", "text": "Point!"},
            "contemplative": {"file": "/refs/emma_contemplative.wav", "text": "Hmm."},
            "urgent": {"file": "/refs/emma_urgent.wav", "text": "Now!"},
        },
        "felix": {
            "excited": {"file": "/refs/felix_excited.wav", "text": "Yes!"},
            "calm": {"file": "/refs/felix_calm.wav", "text": "Okay."},
        },
    }
    path = tmp_path / "prosody_manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def selector(manifest):
    return ProsodySelector(manifest)


class TestProsodySelector:
    def test_direct_match(self, selector):
        assert selector.select("emma", "excited") == "/refs/emma_excited.wav"
        assert selector.select("emma", "calm") == "/refs/emma_calm.wav"

    def test_mapped_emotion(self, selector):
        # "fascinated" maps to "excited"
        assert selector.select("emma", "fascinated") == "/refs/emma_excited.wav"
        # "thoughtful" maps to "contemplative"
        assert selector.select("emma", "thoughtful") == "/refs/emma_contemplative.wav"
        # "building" maps to "emphatic"
        assert selector.select("emma", "building") == "/refs/emma_emphatic.wav"

    def test_unknown_emotion_falls_back_to_default(self, selector):
        # Unknown emotion should fall back to DEFAULT_PROSODY ("calm")
        assert selector.select("emma", "nonexistent_emotion") == "/refs/emma_calm.wav"

    def test_unknown_voice_returns_none(self, selector):
        assert selector.select("unknown_voice", "excited") is None

    def test_case_insensitive(self, selector):
        assert selector.select("EMMA", "EXCITED") == "/refs/emma_excited.wav"
        assert selector.select("Emma", "Calm") == "/refs/emma_calm.wav"

    def test_voices_list(self, selector):
        assert selector.voices() == ["emma", "felix"]

    def test_emotions_list(self, selector):
        assert selector.emotions("emma") == ["calm", "contemplative", "emphatic", "excited", "urgent"]
        assert selector.emotions("felix") == ["calm", "excited"]
        assert selector.emotions("unknown") == []

    def test_select_with_text(self, selector):
        result = selector.select_with_text("emma", "excited")
        assert result == ("/refs/emma_excited.wav", "Wow!")

    def test_select_with_text_mapped(self, selector):
        result = selector.select_with_text("emma", "passionate")
        assert result == ("/refs/emma_excited.wav", "Wow!")

    def test_select_with_text_unknown_voice(self, selector):
        assert selector.select_with_text("unknown", "calm") is None

    def test_fallback_when_prosody_missing(self, selector):
        # felix only has excited and calm — "emphatic" should fall back to calm
        assert selector.select("felix", "emphatic") == "/refs/felix_calm.wav"


class TestEmotionMap:
    def test_all_values_are_valid_prosody_categories(self):
        valid = {"excited", "calm", "emphatic", "contemplative", "urgent"}
        for emotion, prosody in EMOTION_MAP.items():
            assert prosody in valid, f"{emotion} maps to invalid prosody '{prosody}'"

    def test_default_prosody_is_valid(self):
        assert DEFAULT_PROSODY in {"excited", "calm", "emphatic", "contemplative", "urgent"}

    def test_common_emotions_are_mapped(self):
        common = ["excited", "calm", "warm", "curious", "thoughtful", "emphatic",
                   "surprised", "building", "passionate", "skeptical", "neutral"]
        for emotion in common:
            assert emotion in EMOTION_MAP, f"Common emotion '{emotion}' not in EMOTION_MAP"
