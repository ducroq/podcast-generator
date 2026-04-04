"""Tests for generator/elevenlabs/src/voice_settings.py — shared voice config."""

import logging
import pytest

from src.voice_settings import (
    EMOTIONAL_VARIANTS, BASE_SIMILARITY, STABILITY_OFFSET,
    get_voice_settings, parse_line,
)


class TestParseLineFormat:
    def test_standard_line(self):
        speaker, emotion, text = parse_line("Emma: [excited] This is amazing!")
        assert speaker == "emma"
        assert emotion == "excited"
        assert text == "This is amazing!"

    def test_multi_word_emotion(self):
        speaker, emotion, text = parse_line("Lucas: [genuinely interested] Tell me more.")
        assert speaker == "lucas"
        assert emotion == "genuinely interested"
        assert text == "Tell me more."

    def test_no_emotion_tag_returns_none(self):
        speaker, emotion, text = parse_line("Emma: Just plain text here.")
        assert speaker is None
        assert emotion is None
        assert text is None

    def test_empty_string(self):
        speaker, emotion, text = parse_line("")
        assert speaker is None

    def test_whitespace_handling(self):
        speaker, emotion, text = parse_line("  Emma:  [calm]  Some spaced text  ")
        assert speaker == "emma"
        assert emotion == "calm"
        assert text == "Some spaced text"


class TestGetVoiceSettings:
    def test_known_speaker_known_emotion(self):
        settings = get_voice_settings("emma", "excited")
        assert settings.similarity_boost == BASE_SIMILARITY
        # Excited stability: 0.3 + STABILITY_OFFSET(-0.10) = 0.20
        assert abs(settings.stability - 0.20) < 0.01
        assert settings.style == 0.55

    def test_known_speaker_unknown_emotion_falls_back_to_default(self):
        settings = get_voice_settings("emma", "nonexistent_emotion")
        default = EMOTIONAL_VARIANTS["emma"]["default"]
        expected_stability = max(0.15, default["stability"] + STABILITY_OFFSET)
        assert abs(settings.stability - expected_stability) < 0.01
        assert settings.style == default["style"]

    def test_unknown_speaker_falls_back_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            settings = get_voice_settings("unknown_speaker", "excited")
        # Falls back to hardcoded {stability: 0.4, style: 0.4}
        expected_stability = max(0.15, 0.4 + STABILITY_OFFSET)
        assert abs(settings.stability - expected_stability) < 0.01
        assert settings.style == 0.4
        assert any("Unknown speaker" in r.message for r in caplog.records)

    def test_stability_never_below_minimum(self):
        """Even with large negative offset, stability should not go below 0.15."""
        settings = get_voice_settings("emma", "excited")
        assert settings.stability >= 0.15

    def test_all_variants_have_stability_and_style(self):
        for speaker, emotions in EMOTIONAL_VARIANTS.items():
            for emotion, values in emotions.items():
                assert "stability" in values, f"{speaker}/{emotion} missing stability"
                assert "style" in values, f"{speaker}/{emotion} missing style"
                assert 0 <= values["stability"] <= 1, f"{speaker}/{emotion} stability out of range"
                assert 0 <= values["style"] <= 1, f"{speaker}/{emotion} style out of range"
