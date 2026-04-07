"""Tests for generator/analyze_voice.py — F0 and spectral centroid analysis."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generator"))
from analyze_voice import (
    build_parser,
    check_separation,
    compute_spectral_centroid,
    estimate_f0,
)

SR = 16000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tone_200hz():
    """1 second of 200Hz sine — simulates a mid-range male voice."""
    t = np.linspace(0, 1.0, SR, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 200 * t).astype(np.float32)


@pytest.fixture
def tone_300hz():
    """1 second of 300Hz sine — simulates a female voice."""
    t = np.linspace(0, 1.0, SR, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 300 * t).astype(np.float32)


@pytest.fixture
def tone_100hz():
    """1 second of 100Hz sine — simulates a deep male voice."""
    t = np.linspace(0, 1.0, SR, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 100 * t).astype(np.float32)


@pytest.fixture
def silence():
    return np.zeros(SR, dtype=np.float32)


# ---------------------------------------------------------------------------
# estimate_f0
# ---------------------------------------------------------------------------

class TestEstimateF0:
    def test_200hz_tone(self, tone_200hz):
        f0 = estimate_f0(tone_200hz, SR)
        assert f0 is not None
        assert f0 == pytest.approx(200, abs=15)

    def test_300hz_tone(self, tone_300hz):
        f0 = estimate_f0(tone_300hz, SR)
        assert f0 is not None
        assert f0 == pytest.approx(300, abs=15)

    def test_100hz_tone(self, tone_100hz):
        f0 = estimate_f0(tone_100hz, SR)
        assert f0 is not None
        assert f0 == pytest.approx(100, abs=15)

    def test_silence_returns_none(self, silence):
        f0 = estimate_f0(silence, SR)
        assert f0 is None

    def test_ordering(self, tone_100hz, tone_200hz, tone_300hz):
        f0_low = estimate_f0(tone_100hz, SR)
        f0_mid = estimate_f0(tone_200hz, SR)
        f0_high = estimate_f0(tone_300hz, SR)
        assert f0_low < f0_mid < f0_high


# ---------------------------------------------------------------------------
# compute_spectral_centroid
# ---------------------------------------------------------------------------

class TestSpectralCentroid:
    def test_returns_positive_value(self, tone_200hz):
        centroid = compute_spectral_centroid(tone_200hz, SR)
        assert centroid is not None
        assert centroid > 0

    def test_higher_tone_has_higher_centroid(self, tone_100hz, tone_300hz):
        c_low = compute_spectral_centroid(tone_100hz, SR)
        c_high = compute_spectral_centroid(tone_300hz, SR)
        assert c_high > c_low

    def test_silence_returns_none(self, silence):
        centroid = compute_spectral_centroid(silence, SR)
        assert centroid is None


# ---------------------------------------------------------------------------
# check_separation
# ---------------------------------------------------------------------------

class TestCheckSeparation:
    def test_well_separated_no_warnings(self):
        results = [
            {"file": "low.mp3", "f0_hz": 100},
            {"file": "mid.mp3", "f0_hz": 200},
            {"file": "high.mp3", "f0_hz": 350},
        ]
        warnings = check_separation(results, min_f0_gap=80)
        assert warnings == []

    def test_too_close_produces_warning(self):
        results = [
            {"file": "a.mp3", "f0_hz": 200},
            {"file": "b.mp3", "f0_hz": 230},
        ]
        warnings = check_separation(results, min_f0_gap=80)
        assert len(warnings) == 1
        assert "30 Hz apart" in warnings[0]

    def test_skips_none_f0(self):
        results = [
            {"file": "a.mp3", "f0_hz": None},
            {"file": "b.mp3", "f0_hz": 200},
        ]
        warnings = check_separation(results, min_f0_gap=80)
        assert warnings == []

    def test_three_voices_one_pair_close(self):
        results = [
            {"file": "low.mp3", "f0_hz": 100},
            {"file": "mid.mp3", "f0_hz": 150},  # too close to low
            {"file": "high.mp3", "f0_hz": 350},
        ]
        warnings = check_separation(results, min_f0_gap=80)
        assert len(warnings) == 1

    def test_empty_list(self):
        assert check_separation([], min_f0_gap=80) == []


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

class TestCLI:
    def test_minimal_args(self):
        args = build_parser().parse_args(["voice.mp3"])
        assert args.files == ["voice.mp3"]
        assert args.min_gap == 80
        assert args.json is False

    def test_multiple_files(self):
        args = build_parser().parse_args(["a.mp3", "b.mp3", "c.mp3"])
        assert len(args.files) == 3

    def test_all_flags(self):
        args = build_parser().parse_args([
            "a.mp3", "b.mp3", "--min-gap", "100", "--json",
        ])
        assert args.min_gap == 100
        assert args.json is True
