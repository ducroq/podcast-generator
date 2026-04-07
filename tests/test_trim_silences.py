"""Tests for generator/trim_silences.py — silence trimming."""

import subprocess
from pathlib import Path

from audio_utils import get_duration
from trim_silences import trim_silences


class TestTrimSilences:
    def test_output_shorter_than_input(self, tmp_audio):
        output = tmp_audio.parent / "trimmed.mp3"
        result = trim_silences(tmp_audio, output, max_pause=0.2, loudnorm=False)
        assert result is True
        assert output.exists()
        original = get_duration(tmp_audio)
        trimmed = get_duration(output)
        assert trimmed < original, f"Trimmed ({trimmed:.1f}s) should be shorter than original ({original:.1f}s)"

    def test_loudnorm_off_by_default(self, tmp_audio):
        """Verify loudnorm defaults to False (pipeline constraint)."""
        import inspect
        sig = inspect.signature(trim_silences)
        assert sig.parameters["loudnorm"].default is False

    def test_loudnorm_on_produces_output(self, tmp_audio):
        output = tmp_audio.parent / "trimmed_loud.mp3"
        result = trim_silences(tmp_audio, output, max_pause=0.2, loudnorm=True)
        assert result is True
        assert output.exists()

    def test_max_pause_respected(self, tmp_audio):
        """Output should have no silence longer than max_pause (approximately)."""
        output = tmp_audio.parent / "trimmed.mp3"
        trim_silences(tmp_audio, output, max_pause=0.25, loudnorm=False)
        from audio_utils import detect_silences
        remaining = detect_silences(output, noise_db=-30, min_duration=0.3)
        # After trimming with max_pause=0.25, no silence should be >= 0.3s
        assert len(remaining) == 0, f"Found {len(remaining)} silences longer than 0.3s after trimming to 0.25s"

    def test_no_silences_returns_true(self, tmp_path):
        """A continuous tone has no silences — trim should return True (no-op success)."""
        tone = tmp_path / "tone.wav"
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            "sine=frequency=440:duration=2:sample_rate=44100",
            str(tone),
        ], capture_output=True)
        output = tmp_path / "trimmed.mp3"
        result = trim_silences(tone, output, loudnorm=False)
        assert result is True


class TestSilenceClamping:
    def test_leading_silence_does_not_eat_speech(self, tmp_path):
        """Regression: centering was eating adjacent speech at boundaries.
        Now uses clamping instead."""
        # Create audio: 1.5s silence + 1s tone + 0.5s silence
        path = tmp_path / "leading_silence.wav"
        filter_str = (
            "anullsrc=r=44100:cl=mono[s1];[s1]atrim=duration=1.5[silence1];"
            "aevalsrc=sin(440*2*PI*t):s=44100:d=1.0[tone];"
            "anullsrc=r=44100:cl=mono[s2];[s2]atrim=duration=0.5[silence2];"
            "[silence1][tone][silence2]concat=n=3:v=0:a=1[out]"
        )
        subprocess.run([
            "ffmpeg", "-y", "-filter_complex", filter_str,
            "-map", "[out]", str(path),
        ], capture_output=True)

        output = tmp_path / "trimmed.mp3"
        trim_silences(path, output, max_pause=0.35, loudnorm=False)

        # The output should still contain the tone
        out_duration = get_duration(output)
        # Original is 3.0s, after trimming silences it should be shorter but > 1s (the tone)
        assert out_duration >= 1.0, f"Output too short ({out_duration:.1f}s) — speech may have been eaten"
