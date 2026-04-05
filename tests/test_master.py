"""Tests for generator/master.py — Pedalboard mastering chain."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import numpy as np
    from pedalboard import Pedalboard
    HAS_PEDALBOARD = True
except ImportError:
    HAS_PEDALBOARD = False

pytestmark = pytest.mark.skipif(not HAS_PEDALBOARD, reason="pedalboard not installed")

from generator.master import (
    measure_lufs,
    compute_gain_db,
    build_master_chain,
    master_audio,
    analyze_audio,
    build_parser,
)


# ---------------------------------------------------------------------------
# Loudness measurement
# ---------------------------------------------------------------------------

class TestMeasureLufs:
    def test_silence_is_very_quiet(self):
        audio = np.zeros((1, 44100), dtype=np.float32)
        lufs = measure_lufs(audio, 44100)
        assert lufs < -60

    def test_loud_signal_is_loud(self):
        # Full-scale sine wave
        t = np.linspace(0, 1, 44100, dtype=np.float32)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).reshape(1, -1)
        lufs = measure_lufs(audio, 44100)
        assert -10 < lufs < 0

    def test_mono_1d_array(self):
        audio = np.random.randn(44100).astype(np.float32) * 0.1
        lufs = measure_lufs(audio, 44100)
        assert -40 < lufs < 0


class TestComputeGainDb:
    def test_positive_gain(self):
        assert compute_gain_db(-20.0, -16.0) == pytest.approx(4.0)

    def test_negative_gain(self):
        assert compute_gain_db(-14.0, -16.0) == pytest.approx(-2.0)

    def test_zero_gain(self):
        assert compute_gain_db(-16.0, -16.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Mastering chain
# ---------------------------------------------------------------------------

class TestBuildMasterChain:
    def test_returns_pedalboard(self):
        chain = build_master_chain()
        assert isinstance(chain, Pedalboard)
        assert len(chain) >= 4  # highpass, low shelf, high shelf, limiter

    def test_no_compress(self):
        chain = build_master_chain(compress=False)
        # Should have fewer effects
        chain_with = build_master_chain(compress=True)
        assert len(chain) < len(chain_with)

    def test_no_gate(self):
        chain = build_master_chain(noise_gate=False)
        chain_with = build_master_chain(noise_gate=True)
        assert len(chain) < len(chain_with)

    def test_processes_audio(self):
        chain = build_master_chain()
        audio = np.random.randn(1, 44100).astype(np.float32) * 0.1
        result = chain(audio, 44100)
        assert result.shape == audio.shape
        assert not np.array_equal(result, audio)


# ---------------------------------------------------------------------------
# Full mastering
# ---------------------------------------------------------------------------

class TestMasterAudio:
    def test_masters_sine_wave(self, tmp_path):
        from pedalboard.io import AudioFile
        # Create test audio
        sr = 44100
        t = np.linspace(0, 2, sr * 2, dtype=np.float32)
        audio = (0.3 * np.sin(2 * np.pi * 440 * t)).reshape(1, -1)

        input_path = str(tmp_path / "input.wav")
        output_path = str(tmp_path / "output.wav")
        with AudioFile(input_path, 'w', sr, num_channels=1) as f:
            f.write(audio)

        result = master_audio(input_path, output_path)
        assert Path(output_path).exists()
        assert 'input_lufs' in result
        assert 'output_lufs' in result
        assert result['duration_s'] == pytest.approx(2.0, abs=0.1)
        # Output should be closer to -16 LUFS than input
        assert abs(result['output_lufs'] - (-16)) < abs(result['input_lufs'] - (-16))

    def test_custom_target(self, tmp_path):
        from pedalboard.io import AudioFile
        sr = 44100
        t = np.linspace(0, 1, sr, dtype=np.float32)
        audio = (0.2 * np.sin(2 * np.pi * 440 * t)).reshape(1, -1)

        input_path = str(tmp_path / "input.wav")
        output_path = str(tmp_path / "output.wav")
        with AudioFile(input_path, 'w', sr, num_channels=1) as f:
            f.write(audio)

        result = master_audio(input_path, output_path, target_lufs=-14.0)
        assert abs(result['output_lufs'] - (-14)) < 2.0


class TestAnalyzeAudio:
    def test_analyze_returns_measurements(self, tmp_path):
        from pedalboard.io import AudioFile
        sr = 44100
        audio = np.random.randn(1, sr * 2).astype(np.float32) * 0.1

        path = str(tmp_path / "test.wav")
        with AudioFile(path, 'w', sr, num_channels=1) as f:
            f.write(audio)

        info = analyze_audio(path)
        assert 'lufs' in info
        assert 'peak_db' in info
        assert 'rms_db' in info
        assert info['sample_rate'] == sr
        assert info['channels'] == 1
        assert info['duration_s'] == pytest.approx(2.0, abs=0.1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_minimal_args(self):
        parser = build_parser()
        args = parser.parse_args(["input.mp3", "-o", "output.mp3"])
        assert args.input == "input.mp3"
        assert args.output == "output.mp3"
        assert args.target_lufs == -16.0

    def test_analyze_flag(self):
        parser = build_parser()
        args = parser.parse_args(["input.mp3", "--analyze"])
        assert args.analyze is True
        assert args.output is None

    def test_all_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            "input.mp3", "-o", "output.mp3",
            "--target-lufs", "-14",
            "--highpass", "100",
            "--no-compress", "--no-gate",
        ])
        assert args.target_lufs == -14
        assert args.highpass == 100
        assert args.no_compress is True
        assert args.no_gate is True
