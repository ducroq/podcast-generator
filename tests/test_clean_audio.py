"""Tests for generator/clean_audio.py — click detection, repair, trimming, fades."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generator"))
from clean_audio import (
    apply_fades,
    build_parser,
    clean_directory,
    clean_file,
    detect_clicks,
    repair_clicks,
    trim_leading_silence,
    trim_trailing_silence,
)

SR = 24000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def silent_audio():
    """1 second of silence."""
    return np.zeros(SR, dtype=np.float32)


@pytest.fixture
def tone_audio():
    """0.5s of 440Hz tone at moderate volume."""
    t = np.linspace(0, 0.5, int(SR * 0.5), dtype=np.float32)
    return 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def audio_with_clicks(tone_audio):
    """Tone with 3 artificial clicks inserted."""
    audio = tone_audio.copy()
    # Insert sharp jumps
    audio[1000] = 0.9
    audio[1001] = -0.8
    audio[5000] = 0.85
    audio[5001] = -0.85
    audio[9000] = 0.7
    audio[9001] = -0.9
    return audio


@pytest.fixture
def audio_with_silence():
    """Silence + tone + silence."""
    silence_lead = np.zeros(int(SR * 0.2), dtype=np.float32)
    t = np.linspace(0, 0.3, int(SR * 0.3), dtype=np.float32)
    tone = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    silence_trail = np.zeros(int(SR * 0.3), dtype=np.float32)
    return np.concatenate([silence_lead, tone, silence_trail])


@pytest.fixture
def wav_file(tmp_path, tone_audio):
    """Write tone_audio to a WAV file."""
    path = tmp_path / "test.wav"
    sf.write(str(path), tone_audio, SR)
    return path


@pytest.fixture
def wav_with_clicks(tmp_path, audio_with_clicks):
    """Write audio with clicks to a WAV file."""
    path = tmp_path / "clicks.wav"
    sf.write(str(path), audio_with_clicks, SR)
    return path


@pytest.fixture
def wav_dir(tmp_path, audio_with_clicks, audio_with_silence):
    """Directory with multiple WAV files."""
    d = tmp_path / "wavs"
    d.mkdir()
    sf.write(str(d / "001_speaker.wav"), audio_with_clicks, SR)
    sf.write(str(d / "002_speaker.wav"), audio_with_silence, SR)
    return d


# ---------------------------------------------------------------------------
# detect_clicks
# ---------------------------------------------------------------------------

class TestDetectClicks:
    def test_no_clicks_in_clean_tone(self, tone_audio):
        clicks = detect_clicks(tone_audio, SR)
        assert clicks == []

    def test_no_clicks_in_silence(self, silent_audio):
        clicks = detect_clicks(silent_audio, SR)
        assert clicks == []

    def test_detects_inserted_clicks(self, audio_with_clicks):
        clicks = detect_clicks(audio_with_clicks, SR)
        assert len(clicks) >= 3

    def test_click_has_required_fields(self, audio_with_clicks):
        clicks = detect_clicks(audio_with_clicks, SR)
        for click in clicks:
            assert "pos_ms" in click
            assert "severity" in click
            assert "samples" in click
            assert click["severity"] > 0.15

    def test_position_is_plausible(self, audio_with_clicks):
        clicks = detect_clicks(audio_with_clicks, SR)
        positions_ms = [c["pos_ms"] for c in clicks]
        # First click at sample 1000 → ~41.7ms at 24kHz
        assert any(30 < p < 60 for p in positions_ms)

    def test_higher_threshold_detects_fewer(self, audio_with_clicks):
        clicks_low = detect_clicks(audio_with_clicks, SR, threshold=0.1)
        clicks_high = detect_clicks(audio_with_clicks, SR, threshold=0.5)
        assert len(clicks_low) >= len(clicks_high)

    def test_empty_audio(self):
        audio = np.array([], dtype=np.float32)
        clicks = detect_clicks(audio, SR)
        assert clicks == []


# ---------------------------------------------------------------------------
# repair_clicks
# ---------------------------------------------------------------------------

class TestRepairClicks:
    def test_repairs_all_clicks(self, audio_with_clicks):
        repaired, count = repair_clicks(audio_with_clicks, SR)
        assert count > 0
        remaining = detect_clicks(repaired, SR)
        assert len(remaining) < len(detect_clicks(audio_with_clicks, SR))

    def test_does_not_modify_clean_audio(self, tone_audio):
        repaired, count = repair_clicks(tone_audio, SR)
        assert count == 0
        np.testing.assert_array_equal(repaired, tone_audio)

    def test_returns_copy(self, audio_with_clicks):
        original = audio_with_clicks.copy()
        repaired, _ = repair_clicks(audio_with_clicks, SR)
        # Original should be unchanged (repair_clicks copies)
        np.testing.assert_array_equal(audio_with_clicks, original)

    def test_output_same_length(self, audio_with_clicks):
        repaired, _ = repair_clicks(audio_with_clicks, SR)
        assert len(repaired) == len(audio_with_clicks)


# ---------------------------------------------------------------------------
# trim_leading_silence
# ---------------------------------------------------------------------------

class TestTrimLeadingSilence:
    def test_removes_leading_silence(self, audio_with_silence):
        trimmed = trim_leading_silence(audio_with_silence, SR)
        assert len(trimmed) < len(audio_with_silence)

    def test_preserves_margin(self, audio_with_silence):
        trimmed = trim_leading_silence(audio_with_silence, SR)
        # Should have removed most of the 200ms leading silence
        # but kept ~5ms margin
        removed_samples = len(audio_with_silence) - len(trimmed)
        removed_ms = removed_samples / SR * 1000
        assert removed_ms > 150  # removed most of 200ms

    def test_no_change_when_no_leading_silence(self, tone_audio):
        trimmed = trim_leading_silence(tone_audio, SR)
        # Should be same or very close to original length
        assert len(trimmed) >= len(tone_audio) - int(SR * 0.01)

    def test_all_silence_returns_original(self, silent_audio):
        trimmed = trim_leading_silence(silent_audio, SR)
        assert len(trimmed) == len(silent_audio)


# ---------------------------------------------------------------------------
# trim_trailing_silence
# ---------------------------------------------------------------------------

class TestTrimTrailingSilence:
    def test_removes_trailing_silence(self, audio_with_silence):
        trimmed = trim_trailing_silence(audio_with_silence, SR)
        assert len(trimmed) < len(audio_with_silence)

    def test_preserves_margin(self, audio_with_silence):
        trimmed = trim_trailing_silence(audio_with_silence, SR)
        # Should have removed most of 300ms trailing silence
        removed_samples = len(audio_with_silence) - len(trimmed)
        removed_ms = removed_samples / SR * 1000
        assert removed_ms > 200  # removed most of 300ms

    def test_all_silence_returns_original(self, silent_audio):
        trimmed = trim_trailing_silence(silent_audio, SR)
        assert len(trimmed) == len(silent_audio)


# ---------------------------------------------------------------------------
# apply_fades
# ---------------------------------------------------------------------------

class TestApplyFades:
    def test_starts_at_zero(self, tone_audio):
        faded = apply_fades(tone_audio, SR, fade_ms=8)
        assert faded[0] == pytest.approx(0.0, abs=0.001)

    def test_ends_at_zero(self, tone_audio):
        faded = apply_fades(tone_audio, SR, fade_ms=8)
        assert faded[-1] == pytest.approx(0.0, abs=0.001)

    def test_middle_unchanged(self, tone_audio):
        faded = apply_fades(tone_audio, SR, fade_ms=8)
        fade_samples = int(SR * 0.008)
        mid = len(tone_audio) // 2
        np.testing.assert_array_almost_equal(
            faded[fade_samples + 10:mid],
            tone_audio[fade_samples + 10:mid],
        )

    def test_returns_copy(self, tone_audio):
        original = tone_audio.copy()
        apply_fades(tone_audio, SR, fade_ms=8)
        np.testing.assert_array_equal(tone_audio, original)

    def test_output_same_length(self, tone_audio):
        faded = apply_fades(tone_audio, SR, fade_ms=8)
        assert len(faded) == len(tone_audio)

    def test_short_audio_no_crash(self):
        """Audio shorter than 2x fade samples should not crash."""
        short = np.array([0.1, 0.2, 0.1], dtype=np.float32)
        faded = apply_fades(short, SR, fade_ms=8)
        assert len(faded) == 3


# ---------------------------------------------------------------------------
# clean_file
# ---------------------------------------------------------------------------

class TestCleanFile:
    def test_repairs_clicks_in_file(self, wav_with_clicks):
        report = clean_file(wav_with_clicks)
        assert report["clicks_repaired"] > 0

    def test_report_has_required_fields(self, wav_file):
        report = clean_file(wav_file)
        assert "file" in report
        assert "original_duration" in report
        assert "cleaned_duration" in report
        assert "clicks_detected" in report
        assert "fades_applied" in report

    def test_dry_run_does_not_modify(self, wav_with_clicks):
        original_data, _ = sf.read(str(wav_with_clicks), dtype="float32")
        report = clean_file(wav_with_clicks, dry_run=True)
        after_data, _ = sf.read(str(wav_with_clicks), dtype="float32")
        np.testing.assert_array_equal(original_data, after_data)
        assert report["clicks_repaired"] == 0
        assert report["fades_applied"] is False

    def test_no_trim_flag(self, wav_file):
        report = clean_file(wav_file, trim=False)
        assert report["trimmed_ms"] == 0.0

    def test_modifies_file_in_place(self, wav_with_clicks):
        original_data, _ = sf.read(str(wav_with_clicks), dtype="float32")
        clean_file(wav_with_clicks)
        after_data, _ = sf.read(str(wav_with_clicks), dtype="float32")
        # Should differ (clicks repaired + fades applied)
        assert not np.array_equal(original_data, after_data)


# ---------------------------------------------------------------------------
# clean_directory
# ---------------------------------------------------------------------------

class TestCleanDirectory:
    def test_processes_all_wavs(self, wav_dir):
        results = clean_directory(wav_dir)
        assert len(results) == 2

    def test_writes_report_json(self, wav_dir):
        clean_directory(wav_dir)
        report_path = wav_dir / "clean_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["total_files"] == 2

    def test_dry_run_no_report(self, wav_dir):
        clean_directory(wav_dir, dry_run=True)
        assert not (wav_dir / "clean_report.json").exists()

    def test_empty_directory(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        results = clean_directory(empty)
        assert results == []


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

class TestCLI:
    def test_defaults(self):
        args = build_parser().parse_args(["somedir"])
        assert args.threshold == 0.15
        assert args.fade_ms == 8
        assert args.no_trim is False
        assert args.dry_run is False

    def test_custom_args(self):
        args = build_parser().parse_args([
            "somefile.wav", "--threshold", "0.2",
            "--fade-ms", "12", "--no-trim", "--dry-run",
        ])
        assert args.threshold == 0.2
        assert args.fade_ms == 12
        assert args.no_trim is True
        assert args.dry_run is True
