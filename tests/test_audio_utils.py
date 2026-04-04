"""Tests for generator/audio_utils.py — shared ffmpeg helpers."""

import pytest
from audio_utils import detect_silences, get_duration, get_sample_rate


class TestGetDuration:
    def test_returns_float(self, tmp_audio):
        duration = get_duration(tmp_audio)
        assert isinstance(duration, float)

    def test_expected_duration(self, tmp_audio):
        # Our fixture is ~5 seconds
        duration = get_duration(tmp_audio)
        assert 4.5 < duration < 5.5

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="ffprobe failed"):
            get_duration(tmp_path / "nonexistent.wav")


class TestGetSampleRate:
    def test_returns_44100(self, tmp_audio):
        sr = get_sample_rate(tmp_audio)
        assert sr == 44100

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="ffprobe failed"):
            get_sample_rate(tmp_path / "nonexistent.wav")


class TestDetectSilences:
    def test_finds_silences(self, tmp_audio):
        silences = detect_silences(tmp_audio, noise_db=-30, min_duration=0.3)
        assert len(silences) >= 2, f"Expected at least 2 silences, got {len(silences)}"

    def test_silence_structure(self, tmp_audio):
        silences = detect_silences(tmp_audio, noise_db=-30, min_duration=0.3)
        for s in silences:
            assert "start" in s
            assert "end" in s
            assert "duration" in s
            assert s["end"] > s["start"]
            assert s["duration"] > 0

    def test_no_silences_in_continuous_tone(self, tmp_path):
        """A continuous tone should have no silences."""
        tone = tmp_path / "tone.wav"
        import subprocess
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            "sine=frequency=440:duration=2:sample_rate=44100",
            str(tone),
        ], capture_output=True)
        silences = detect_silences(tone, noise_db=-30, min_duration=0.3)
        assert len(silences) == 0

    def test_nonexistent_file_returns_empty_or_raises(self, tmp_path):
        # ffmpeg will fail, detect_silences should raise RuntimeError
        with pytest.raises(RuntimeError):
            detect_silences(tmp_path / "nonexistent.wav")
