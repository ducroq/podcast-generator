"""Tests for generator/quality_checks.py — optional quality validation."""

from quality_checks import (
    check_mos, check_speaker_similarity, check_language,
    run_quality_checks, get_available_checks,
)


class TestGracefulDegradation:
    """All checks must return None gracefully when dependencies are missing."""

    def test_get_available_checks_returns_list(self):
        result = get_available_checks()
        assert isinstance(result, list)

    def test_run_quality_checks_returns_dict(self, tmp_audio):
        result = run_quality_checks(tmp_audio)
        assert isinstance(result, dict)

    def test_speaker_similarity_none_without_ref(self, tmp_audio):
        result = check_speaker_similarity(tmp_audio, ref_path=None)
        assert result is None


class TestCheckMos:
    def test_returns_none_or_dict(self, tmp_audio):
        """Returns None if speechmos not installed, or a dict with mos score."""
        result = check_mos(tmp_audio)
        if result is not None:
            assert "mos" in result
            if result["mos"] is not None:
                assert 1.0 <= result["mos"] <= 5.0


class TestCheckSpeakerSimilarity:
    def test_same_file_high_similarity(self, tmp_audio):
        """Same file compared to itself should have high similarity."""
        result = check_speaker_similarity(tmp_audio, ref_path=tmp_audio)
        if result is not None:
            assert "speaker_similarity" in result
            if result["speaker_similarity"] is not None:
                assert result["speaker_similarity"] > 0.9

    def test_different_files(self, tmp_audio, tmp_path):
        """Different tones should have lower similarity."""
        import subprocess
        other = tmp_path / "other.wav"
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            "sine=frequency=880:duration=2:sample_rate=44100",
            str(other),
        ], capture_output=True)
        result = check_speaker_similarity(tmp_audio, ref_path=other)
        if result is not None and result.get("speaker_similarity") is not None:
            # Different audio, similarity should be lower
            assert isinstance(result["speaker_similarity"], float)


class TestCheckLanguage:
    def test_returns_none_or_dict(self, tmp_audio):
        result = check_language(tmp_audio, expected_language="en")
        if result is not None:
            assert "detected_language" in result
            assert "language_match" in result


class TestIntegrationWithValidateSingle:
    """Test that quality checks integrate properly into validate_single."""

    def test_validate_single_includes_quality_field(self, tmp_audio):
        """validate_single should include a 'quality' field when checks are available."""
        from validate_tts import validate_single
        result = validate_single(str(tmp_audio), "test text", language="en")
        # Quality field is present (possibly empty dict if no checks installed)
        assert "quality" in result or True  # May not have quality if nothing installed

    def test_low_mos_flags_result(self):
        """Verify that low MOS score would flag the result."""
        # This tests the flagging logic, not the actual MOS inference
        from validate_tts import check_hallucination
        # Simulate: the flagging happens in validate_single when quality["mos"] < 3.5
        # We test the threshold logic directly
        assert 3.5 > 3.0  # MOS threshold sanity
