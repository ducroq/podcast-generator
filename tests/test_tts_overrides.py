"""Tests for generator/tts_overrides.py — TTS override loading, lookup, assembly."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generator"))
from tts_overrides import (
    apply_override_text,
    assemble_segments,
    build_parser,
    get_override,
    get_segment_pauses,
    get_segment_texts,
    is_segmented,
    load_overrides,
    validate_overrides,
)

SR = 24000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def overrides_file(tmp_path):
    """Create a sample overrides JSON file."""
    data = {
        "overrides": {
            "015": [
                {"text": "First segment.", "pause_after": 0.3},
                {"text": "Second segment."},
            ],
            "031": "Simple replacement text.",
        }
    }
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def segment_wavs(tmp_path):
    """Create short WAV segment files."""
    paths = []
    for i in range(3):
        t = np.linspace(0, 0.2, int(SR * 0.2), dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * (440 + i * 100) * t).astype(np.float32)
        path = tmp_path / f"seg_{i}.wav"
        sf.write(str(path), audio, SR)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# load_overrides
# ---------------------------------------------------------------------------

class TestLoadOverrides:
    def test_loads_valid_json(self, overrides_file):
        ov = load_overrides(overrides_file)
        assert "015" in ov
        assert "031" in ov

    def test_missing_key_returns_empty(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text('{"something_else": 1}')
        ov = load_overrides(path)
        assert ov == {}

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json at all")
        with pytest.raises(json.JSONDecodeError):
            load_overrides(path)


# ---------------------------------------------------------------------------
# get_override
# ---------------------------------------------------------------------------

class TestGetOverride:
    def test_lookup_by_int(self, overrides_file):
        ov = load_overrides(overrides_file)
        result = get_override(ov, 15)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_simple_string_override(self, overrides_file):
        ov = load_overrides(overrides_file)
        result = get_override(ov, 31)
        assert result == "Simple replacement text."

    def test_missing_returns_none(self, overrides_file):
        ov = load_overrides(overrides_file)
        assert get_override(ov, 999) is None

    def test_zero_padding(self, overrides_file):
        ov = load_overrides(overrides_file)
        # Line 15 should match key "015"
        assert get_override(ov, 15) is not None


# ---------------------------------------------------------------------------
# is_segmented
# ---------------------------------------------------------------------------

class TestIsSegmented:
    def test_list_is_segmented(self):
        assert is_segmented([{"text": "hi"}]) is True

    def test_string_not_segmented(self):
        assert is_segmented("hello") is False

    def test_none_not_segmented(self):
        assert is_segmented(None) is False


# ---------------------------------------------------------------------------
# apply_override_text
# ---------------------------------------------------------------------------

class TestApplyOverrideText:
    def test_none_returns_original(self):
        assert apply_override_text("original", None) == "original"

    def test_string_returns_replacement(self):
        assert apply_override_text("original", "new text") == "new text"

    def test_list_returns_segments(self):
        segments = [{"text": "a"}, {"text": "b"}]
        result = apply_override_text("original", segments)
        assert result == segments


# ---------------------------------------------------------------------------
# get_segment_texts / get_segment_pauses
# ---------------------------------------------------------------------------

class TestSegmentHelpers:
    def test_texts_from_segments(self):
        segments = [{"text": "hello", "pause_after": 0.3}, {"text": "world"}]
        assert get_segment_texts(segments) == ["hello", "world"]

    def test_pauses_from_segments(self):
        segments = [{"text": "hello", "pause_after": 0.3}, {"text": "world"}]
        assert get_segment_pauses(segments) == [0.3, 0.0]

    def test_texts_from_string(self):
        assert get_segment_texts("hello") == ["hello"]

    def test_pauses_from_string(self):
        assert get_segment_pauses("hello") == [0.0]


# ---------------------------------------------------------------------------
# assemble_segments
# ---------------------------------------------------------------------------

class TestAssembleSegments:
    def test_assembles_with_pauses(self, segment_wavs, tmp_path):
        output = tmp_path / "assembled.wav"
        pauses = [0.2, 0.1, 0.0]
        duration = assemble_segments(segment_wavs, pauses, SR, output)
        assert output.exists()
        # 3 segments * 0.2s + 0.2s + 0.1s pauses = 0.9s
        assert duration == pytest.approx(0.9, abs=0.05)

    def test_assembles_without_pauses(self, segment_wavs, tmp_path):
        output = tmp_path / "assembled.wav"
        pauses = [0.0, 0.0, 0.0]
        duration = assemble_segments(segment_wavs, pauses, SR, output)
        # 3 segments * 0.2s = 0.6s
        assert duration == pytest.approx(0.6, abs=0.05)

    def test_output_is_valid_wav(self, segment_wavs, tmp_path):
        output = tmp_path / "assembled.wav"
        assemble_segments(segment_wavs, [0.1, 0.0, 0.0], SR, output)
        audio, sr = sf.read(str(output), dtype="float32")
        assert sr == SR
        assert len(audio) > 0

    def test_single_segment(self, segment_wavs, tmp_path):
        output = tmp_path / "single.wav"
        duration = assemble_segments([segment_wavs[0]], [0.0], SR, output)
        assert duration == pytest.approx(0.2, abs=0.02)

    def test_empty_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No segments"):
            assemble_segments([], [], SR, tmp_path / "out.wav")


# ---------------------------------------------------------------------------
# validate_overrides
# ---------------------------------------------------------------------------

class TestValidateOverrides:
    def test_valid_returns_empty(self):
        ov = {
            "015": [{"text": "hi", "pause_after": 0.3}],
            "031": "replacement",
        }
        assert validate_overrides(ov) == []

    def test_non_numeric_key(self):
        issues = validate_overrides({"abc": "text"})
        assert len(issues) == 1
        assert "numeric" in issues[0]

    def test_empty_string(self):
        issues = validate_overrides({"001": ""})
        assert len(issues) == 1

    def test_missing_text_in_segment(self):
        issues = validate_overrides({"001": [{"pause_after": 0.3}]})
        assert any("missing 'text'" in i for i in issues)

    def test_negative_pause(self):
        issues = validate_overrides({"001": [{"text": "hi", "pause_after": -1}]})
        assert any("negative" in i for i in issues)

    def test_invalid_type(self):
        issues = validate_overrides({"001": 42})
        assert any("must be string or list" in i for i in issues)


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

class TestCLI:
    def test_list_flag(self):
        args = build_parser().parse_args(["overrides.json", "--list"])
        assert args.list is True

    def test_check_flag(self):
        args = build_parser().parse_args(["overrides.json", "--check", "15"])
        assert args.check == 15

    def test_validate_flag(self):
        args = build_parser().parse_args(["overrides.json", "--validate"])
        assert args.validate is True

    def test_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["overrides.json", "--list", "--validate"])
