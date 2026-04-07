"""Tests for generator/mix_preprocess.py — room reverb, speaker volume, RMS normalize."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generator"))
from mix_preprocess import (
    apply_clip_fades,
    apply_reverb,
    apply_speaker_volume,
    build_parser,
    generate_room_ir,
    preprocess_directory,
    preprocess_line,
    rms_normalize,
)

SR = 24000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tone_audio():
    """0.5s of 440Hz tone."""
    t = np.linspace(0, 0.5, int(SR * 0.5), dtype=np.float32)
    return 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def room_ir():
    return generate_room_ir(SR, decay_time=0.15)


@pytest.fixture
def wav_dir(tmp_path, tone_audio):
    """Directory with WAV files and a manifest."""
    d = tmp_path / "lines"
    d.mkdir()
    sf.write(str(d / "001_alex.wav"), tone_audio, SR)
    sf.write(str(d / "002_zara.wav"), tone_audio * 0.5, SR)

    manifest = [
        {"file": "001_alex.wav", "speaker": "alex", "text": "Hello"},
        {"file": "002_zara.wav", "speaker": "zara", "text": "Hi"},
    ]
    manifest_path = d / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return d, manifest, manifest_path


# ---------------------------------------------------------------------------
# generate_room_ir
# ---------------------------------------------------------------------------

class TestGenerateRoomIR:
    def test_returns_numpy_array(self):
        ir = generate_room_ir(SR)
        assert isinstance(ir, np.ndarray)

    def test_length_matches_decay(self):
        ir = generate_room_ir(SR, decay_time=0.15)
        expected_samples = int(SR * 0.15)
        assert len(ir) == expected_samples

    def test_normalized_to_unit_peak(self):
        ir = generate_room_ir(SR)
        assert np.max(np.abs(ir)) == pytest.approx(1.0, abs=0.01)

    def test_decays_over_time(self):
        ir = generate_room_ir(SR, decay_time=0.3)
        first_quarter = np.mean(np.abs(ir[:len(ir) // 4]))
        last_quarter = np.mean(np.abs(ir[-len(ir) // 4:]))
        assert first_quarter > last_quarter

    def test_deterministic(self):
        ir1 = generate_room_ir(SR)
        ir2 = generate_room_ir(SR)
        np.testing.assert_array_equal(ir1, ir2)


# ---------------------------------------------------------------------------
# apply_reverb
# ---------------------------------------------------------------------------

class TestApplyReverb:
    def test_output_same_length(self, tone_audio, room_ir):
        wet = apply_reverb(tone_audio, room_ir, mix=0.02)
        assert len(wet) == len(tone_audio)

    def test_zero_mix_returns_original(self, tone_audio, room_ir):
        result = apply_reverb(tone_audio, room_ir, mix=0.0)
        np.testing.assert_array_almost_equal(result, tone_audio, decimal=5)

    def test_nonzero_mix_differs(self, tone_audio, room_ir):
        result = apply_reverb(tone_audio, room_ir, mix=0.5)
        assert not np.allclose(result, tone_audio, atol=0.01)

    def test_returns_float32(self, tone_audio, room_ir):
        result = apply_reverb(tone_audio, room_ir)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# apply_speaker_volume
# ---------------------------------------------------------------------------

class TestApplySpeakerVolume:
    def test_no_map_returns_original(self, tone_audio):
        result = apply_speaker_volume(tone_audio, "alex")
        np.testing.assert_array_equal(result, tone_audio)

    def test_zero_db_returns_original(self, tone_audio):
        result = apply_speaker_volume(tone_audio, "alex", {"alex": 0.0})
        np.testing.assert_array_equal(result, tone_audio)

    def test_positive_db_boosts(self, tone_audio):
        result = apply_speaker_volume(tone_audio, "zara", {"zara": 6.0})
        assert np.max(np.abs(result)) > np.max(np.abs(tone_audio))

    def test_negative_db_cuts(self, tone_audio):
        result = apply_speaker_volume(tone_audio, "zara", {"zara": -6.0})
        assert np.max(np.abs(result)) < np.max(np.abs(tone_audio))

    def test_unknown_speaker_unchanged(self, tone_audio):
        result = apply_speaker_volume(tone_audio, "unknown", {"zara": 6.0})
        np.testing.assert_array_equal(result, tone_audio)

    def test_6db_roughly_doubles(self, tone_audio):
        result = apply_speaker_volume(tone_audio, "zara", {"zara": 6.0})
        ratio = np.max(np.abs(result)) / np.max(np.abs(tone_audio))
        assert ratio == pytest.approx(2.0, abs=0.05)


# ---------------------------------------------------------------------------
# rms_normalize
# ---------------------------------------------------------------------------

class TestRmsNormalize:
    def test_achieves_target_rms(self, tone_audio):
        result = rms_normalize(tone_audio, target_rms=0.1)
        actual_rms = np.sqrt(np.mean(result ** 2))
        assert actual_rms == pytest.approx(0.1, abs=0.005)

    def test_silent_audio_unchanged(self):
        silent = np.zeros(1000, dtype=np.float32)
        result = rms_normalize(silent, target_rms=0.1)
        np.testing.assert_array_equal(result, silent)

    def test_different_targets(self, tone_audio):
        result_low = rms_normalize(tone_audio, target_rms=0.05)
        result_high = rms_normalize(tone_audio, target_rms=0.2)
        rms_low = np.sqrt(np.mean(result_low ** 2))
        rms_high = np.sqrt(np.mean(result_high ** 2))
        assert rms_low < rms_high


# ---------------------------------------------------------------------------
# apply_clip_fades
# ---------------------------------------------------------------------------

class TestApplyClipFades:
    def test_starts_at_zero(self, tone_audio):
        result = apply_clip_fades(tone_audio, SR, fade_ms=20)
        assert result[0] == pytest.approx(0.0, abs=0.001)

    def test_ends_at_zero(self, tone_audio):
        result = apply_clip_fades(tone_audio, SR, fade_ms=20)
        assert result[-1] == pytest.approx(0.0, abs=0.001)

    def test_returns_copy(self, tone_audio):
        original = tone_audio.copy()
        apply_clip_fades(tone_audio, SR)
        np.testing.assert_array_equal(tone_audio, original)


# ---------------------------------------------------------------------------
# preprocess_line
# ---------------------------------------------------------------------------

class TestPreprocessLine:
    def test_returns_array(self, wav_dir):
        d, manifest, _ = wav_dir
        ir = generate_room_ir(SR)
        result = preprocess_line(d / "001_alex.wav", "alex", SR, room_ir=ir)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_with_volume_map(self, wav_dir):
        d, manifest, _ = wav_dir
        result = preprocess_line(
            d / "002_zara.wav", "zara", SR,
            volume_map={"zara": 3.0},
        )
        assert isinstance(result, np.ndarray)

    def test_without_reverb(self, wav_dir):
        d, manifest, _ = wav_dir
        result = preprocess_line(d / "001_alex.wav", "alex", SR, room_ir=None)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# preprocess_directory
# ---------------------------------------------------------------------------

class TestPreprocessDirectory:
    def test_processes_all_files(self, wav_dir):
        d, manifest, _ = wav_dir
        results = preprocess_directory(d, manifest, sr=SR)
        assert len(results) == 2
        assert all(r["status"] == "OK" for r in results)

    def test_handles_missing_files(self, wav_dir):
        d, _, _ = wav_dir
        manifest = [{"file": "nonexistent.wav", "speaker": "alex"}]
        results = preprocess_directory(d, manifest, sr=SR)
        assert results[0]["status"] == "MISSING"

    def test_dry_run_no_modification(self, wav_dir):
        d, manifest, _ = wav_dir
        original, _ = sf.read(str(d / "001_alex.wav"), dtype="float32")
        results = preprocess_directory(d, manifest, sr=SR, dry_run=True)
        after, _ = sf.read(str(d / "001_alex.wav"), dtype="float32")
        np.testing.assert_array_equal(original, after)
        assert all(r["status"] == "DRY" for r in results)

    def test_with_volume_map(self, wav_dir):
        d, manifest, _ = wav_dir
        results = preprocess_directory(
            d, manifest, sr=SR, volume_map={"zara": 2.5},
        )
        assert len(results) == 2


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

class TestCLI:
    def test_minimal_args(self):
        args = build_parser().parse_args(["dir/", "--manifest", "m.json"])
        assert args.input_dir == "dir/"
        assert args.manifest == "m.json"
        assert args.reverb_decay == 0.15
        assert args.reverb_mix == 0.02

    def test_all_flags(self):
        args = build_parser().parse_args([
            "dir/", "--manifest", "m.json",
            "--reverb-decay", "0.2", "--reverb-mix", "0.05",
            "--speaker-volume", '{"zara": 3.0}',
            "--target-rms", "0.15", "--fade-ms", "10",
            "--sr", "48000", "--dry-run",
        ])
        assert args.reverb_decay == 0.2
        assert args.reverb_mix == 0.05
        assert args.speaker_volume == '{"zara": 3.0}'
        assert args.target_rms == 0.15
        assert args.fade_ms == 10
        assert args.sr == 48000
        assert args.dry_run is True
