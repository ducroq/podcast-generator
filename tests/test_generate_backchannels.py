"""Tests for generator/generate_backchannels.py — backchannel clip library."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generator"))
from generate_backchannels import (
    ALL_WORDS,
    BACKCHANNEL_WORDS,
    build_parser,
    generate_backchannel_clips,
    plan_backchannel_clips,
    get_all_words,
    get_words_for_type,
    load_backchannel_library,
    load_backchannel_manifest,
    list_library,
    save_manifest,
)

SR = 24000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bc_dir(tmp_path):
    """Create a directory with backchannel WAV clips."""
    d = tmp_path / "bc_clips"
    d.mkdir()
    for speaker in ["alex", "morgan"]:
        for i in range(3):
            t = np.linspace(0, 0.3, int(SR * 0.3), dtype=np.float32)
            audio = 0.2 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
            sf.write(str(d / f"bc_{speaker}_{i:02d}.wav"), audio, SR)
    return d


# ---------------------------------------------------------------------------
# Word library
# ---------------------------------------------------------------------------

class TestBackchannelWords:
    def test_categories_non_empty(self):
        for category, words in BACKCHANNEL_WORDS.items():
            assert len(words) > 0, f"Category '{category}' is empty"

    def test_all_words_non_empty(self):
        assert len(ALL_WORDS) > 0

    def test_get_words_for_type(self):
        assert len(get_words_for_type("agreement")) > 0
        assert len(get_words_for_type("surprise")) > 0
        assert get_words_for_type("nonexistent") == []

    def test_all_words_unique(self):
        assert len(ALL_WORDS) == len(set(ALL_WORDS))


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

class TestLoadBackchannelLibrary:
    def test_loads_clips_by_speaker(self, bc_dir):
        clips = load_backchannel_library(bc_dir)
        assert "alex" in clips
        assert "morgan" in clips
        assert len(clips["alex"]) == 3
        assert len(clips["morgan"]) == 3

    def test_clips_are_numpy_arrays(self, bc_dir):
        clips = load_backchannel_library(bc_dir)
        for speaker, arrays in clips.items():
            for arr in arrays:
                assert isinstance(arr, np.ndarray)
                assert len(arr) > 0

    def test_empty_dir_returns_empty(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        clips = load_backchannel_library(empty)
        assert clips == {}


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

class TestManifest:
    def test_save_and_load(self, tmp_path):
        manifest = [
            {"file": "bc_alex_00.wav", "word": "Mmhm.", "speaker": "alex"},
        ]
        save_manifest(manifest, tmp_path)
        loaded = load_backchannel_manifest(tmp_path)
        assert loaded == manifest

    def test_load_missing_returns_none(self, tmp_path):
        assert load_backchannel_manifest(tmp_path) is None


# ---------------------------------------------------------------------------
# Generation (stub)
# ---------------------------------------------------------------------------

class TestPlanBackchannelClips:
    def test_returns_manifest_list(self, tmp_path):
        config = {"name": "alex", "ref": "ref.mp3", "ref_text": "hello"}
        manifest = plan_backchannel_clips(config, tmp_path)
        assert len(manifest) == len(ALL_WORDS)
        assert all("file" in m for m in manifest)
        assert all("speaker" in m for m in manifest)
        assert all(m["speaker"] == "alex" for m in manifest)

    def test_custom_words(self, tmp_path):
        config = {"name": "morgan", "ref": "ref.mp3", "ref_text": "hello"}
        manifest = plan_backchannel_clips(config, tmp_path,
                                           words=["Yeah.", "Mm."])
        assert len(manifest) == 2


class TestGenerateBackchannelClips:
    def test_raises_not_implemented(self, tmp_path):
        config = {"name": "alex", "ref": "ref.mp3", "ref_text": "hello"}
        with pytest.raises(NotImplementedError, match="gpu-server"):
            generate_backchannel_clips(config, tmp_path)


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

class TestCLI:
    def test_list_flag(self):
        args = build_parser().parse_args(["--list", "bc_clips/"])
        assert args.list == "bc_clips/"

    def test_generate_flags(self):
        args = build_parser().parse_args([
            "--voices", "voices.json", "-o", "output/",
            "--engine", "chatterbox",
        ])
        assert args.voices == "voices.json"
        assert args.output_dir == "output/"
        assert args.engine == "chatterbox"
