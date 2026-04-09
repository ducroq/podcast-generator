"""Tests for generator/generate_tts.py — staged TTS with fallback chain.

All TTS engine calls are mocked — no GPU required.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import yaml

from generate_tts import (
    estimate_max_duration,
    _is_hallucinated,
    _get_adapter,
    _generate_with_guard,
    _resample_if_needed,
    generate_missing,
    QwenAdapter,
    ChatterboxAdapter,
    ADAPTERS,
)
from manifest import parse_script, build_manifest, save_manifest, load_manifest
from config import load_episode_config, EpisodeConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PODCAST_YAML = {
    "podcast": {"name": "Test"},
    "tts": {
        "language": "English",
        "temperature": 0.7,
        "repetition_penalty": 1.2,
        "hallucination": {
            "max_duration_per_word": 0.6,
            "retry_count": 2,
            "retry_temperature": 0.4,
            "retry_repetition_penalty": 1.3,
        },
    },
    "mix": {"target_sr": 24000, "backchannel": {}},
    "processing": {},
    "cast": {
        "alex": {
            "voice_ref": "alex.mp3",
            "ref_text": "Hello from Alex.",
            "engine": "qwen",
            "fallback": ["chatterbox"],
            "volume_db": 0.0,
        },
        "morgan": {
            "voice_ref": "morgan.mp3",
            "ref_text": "Hello from Morgan.",
            "engine": "qwen",
            "fallback": [],
            "volume_db": 0.0,
        },
    },
    "music": {},
    "review": {},
}

EPISODE_YAML = {
    "episode": {"number": 1, "title": "Test", "slug": "ep_test"},
    "podcast": "test-pod",
    "script": "scripts/test.txt",
    "work_dir": "work/test",
    "overrides": {},
    "force_fallback": [],
}

SCRIPT_TEXT = """\
====================
SECTION
====================

Alex: Hello world, this is a test line.

Morgan: Yes, it is indeed a test.

Alex: Another line from Alex.
"""


def _make_audio(sr=24000, duration=1.5):
    """Create a short test audio array."""
    samples = int(sr * duration)
    return np.random.randn(samples).astype(np.float32) * 0.1


@pytest.fixture
def pipeline_env(tmp_path):
    """Set up full pipeline environment: config + script + dirs."""
    # Podcast config
    pod_dir = tmp_path / "podcasts" / "test-pod"
    pod_dir.mkdir(parents=True)
    with open(pod_dir / "podcast.yaml", "w") as f:
        yaml.dump(PODCAST_YAML, f)

    # Episode config
    ep_dir = tmp_path / "podcasts" / "episodes"
    ep_dir.mkdir(parents=True)
    ep_path = ep_dir / "ep_test.yaml"
    with open(ep_path, "w") as f:
        yaml.dump(EPISODE_YAML, f)

    # Script
    scripts_dir = tmp_path / "podcasts" / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "test.txt").write_text(SCRIPT_TEXT, encoding="utf-8")

    # Voice refs
    voice_dir = tmp_path / "podcasts" / "voice_refs"
    voice_dir.mkdir(parents=True)
    for name in ("alex.mp3", "morgan.mp3"):
        audio = _make_audio(duration=5.0)
        sf.write(str(voice_dir / name), audio, 24000)

    # Work dir
    tts_dir = tmp_path / "podcasts" / "work" / "test" / "tts"
    tts_dir.mkdir(parents=True)

    cfg = load_episode_config(ep_path, podcast_dir=pod_dir)
    entries = parse_script(cfg.script_path())
    manifest = build_manifest(entries, audio_dir=cfg.tts_dir(),
                              script_path=str(cfg.script_path()), episode="ep_test")

    return {
        "cfg": cfg,
        "manifest": manifest,
        "tts_dir": tts_dir,
        "tmp_path": tmp_path,
    }


class MockAdapter:
    """Mock TTS adapter that returns predictable audio."""

    name = "mock"

    def __init__(self, duration=1.5, sr=24000, fail_count=0):
        self.duration = duration
        self.sr = sr
        self.fail_count = fail_count
        self._calls = 0
        self.loaded = False

    def load(self, device="cuda:0"):
        self.loaded = True

    def generate(self, text, voice_ref, ref_text=None, language=None, **kwargs):
        self._calls += 1
        if self._calls <= self.fail_count:
            # Simulate hallucination: 60s for a short line
            return _make_audio(self.sr, 60.0), self.sr
        return _make_audio(self.sr, self.duration), self.sr

    def unload(self):
        self.loaded = False


# ---------------------------------------------------------------------------
# Hallucination guard
# ---------------------------------------------------------------------------


class TestHallucinationGuard:
    def test_estimate_max_duration(self):
        assert estimate_max_duration("hello world") == 10.0  # floor
        assert estimate_max_duration("a " * 30) == 30 * 0.6  # 18.0

    def test_not_hallucinated(self):
        audio = _make_audio(24000, 2.0)
        bad, dur, max_dur = _is_hallucinated(audio, 24000, "Hello world test line")
        assert not bad

    def test_hallucinated(self):
        audio = _make_audio(24000, 60.0)
        bad, dur, max_dur = _is_hallucinated(audio, 24000, "Hello world")
        assert bad
        assert dur == pytest.approx(60.0, abs=0.1)

    def test_generate_with_guard_clean(self):
        adapter = MockAdapter(duration=1.5)
        adapter.load()
        audio, sr, attempts = _generate_with_guard(
            adapter, "Hello world", "/fake/ref.mp3", "ref text", "English",
            {"temperature": 0.7, "repetition_penalty": 1.2},
            {"max_duration_per_word": 0.6, "retry_count": 2},
        )
        assert audio is not None
        assert attempts == 1

    def test_generate_with_guard_retry_success(self):
        adapter = MockAdapter(duration=1.5, fail_count=1)
        adapter.load()
        audio, sr, attempts = _generate_with_guard(
            adapter, "Hello world", "/fake/ref.mp3", "ref text", "English",
            {"temperature": 0.7, "repetition_penalty": 1.2},
            {"max_duration_per_word": 0.6, "retry_count": 3},
        )
        assert audio is not None
        assert attempts == 2

    def test_generate_with_guard_all_retries_fail(self):
        adapter = MockAdapter(duration=1.5, fail_count=100)  # always hallucinate
        adapter.load()
        audio, sr, attempts = _generate_with_guard(
            adapter, "Hello world", "/fake/ref.mp3", "ref text", "English",
            {"temperature": 0.7, "repetition_penalty": 1.2},
            {"max_duration_per_word": 0.6, "retry_count": 2},
        )
        assert audio is None
        assert attempts == 3  # 1 initial + 2 retries


# ---------------------------------------------------------------------------
# Engine adapters
# ---------------------------------------------------------------------------


class TestAdapters:
    def test_get_adapter_qwen(self):
        adapter = _get_adapter("qwen")
        assert isinstance(adapter, QwenAdapter)

    def test_get_adapter_chatterbox(self):
        adapter = _get_adapter("chatterbox")
        assert isinstance(adapter, ChatterboxAdapter)

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown TTS engine"):
            _get_adapter("nonexistent")


# ---------------------------------------------------------------------------
# Resample
# ---------------------------------------------------------------------------


class TestResample:
    def test_no_resample_needed(self):
        audio = _make_audio(24000, 1.0)
        result = _resample_if_needed(audio, 24000, 24000)
        assert len(result) == len(audio)

    def test_resample_up(self):
        audio = _make_audio(16000, 1.0)
        result = _resample_if_needed(audio, 16000, 24000)
        assert len(result) == pytest.approx(24000, abs=10)


# ---------------------------------------------------------------------------
# generate_missing — integration tests with mock adapters
# ---------------------------------------------------------------------------


class TestGenerateMissing:
    def test_skip_existing_lines(self, pipeline_env):
        """Lines with existing audio are not re-generated."""
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        # Pre-create one file
        first_hash = list(manifest["lines"].keys())[0]
        first_info = manifest["lines"][first_hash]
        audio = _make_audio(24000, 2.0)
        sf.write(str(cfg.tts_dir() / first_info["file"]), audio, 24000)

        # Re-build manifest to pick up existing file
        from manifest import parse_script as ps, build_manifest as bm
        entries = ps(cfg.script_path())
        manifest = bm(entries, audio_dir=cfg.tts_dir())

        with patch.dict(ADAPTERS, {"qwen": lambda: MockAdapter()}):
            result = generate_missing(manifest, cfg)

        assert result["skipped"] >= 1
        # The existing file's status should be "exists"
        assert manifest["lines"][first_hash]["status"] == "exists"

    def test_generate_missing_line(self, pipeline_env):
        """Missing lines get generated and manifest is updated."""
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        mock = MockAdapter(duration=1.5)
        with patch.dict(ADAPTERS, {"qwen": lambda: mock}):
            result = generate_missing(manifest, cfg)

        assert result["generated"] == 3
        assert result["failed"] == 0
        # All lines should now be "exists"
        for info in manifest["lines"].values():
            assert info["status"] == "exists"
            assert info["duration"] is not None
            assert info["engine"] == "qwen"

    def test_hallucination_triggers_fallback(self, pipeline_env):
        """Hallucinated line falls back to next engine in chain."""
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        # Qwen always hallucinates, chatterbox works
        bad_qwen = MockAdapter(duration=1.5, fail_count=100)
        good_cb = MockAdapter(duration=1.5)

        with patch.dict(ADAPTERS, {
            "qwen": lambda: bad_qwen,
            "chatterbox": lambda: good_cb,
        }):
            result = generate_missing(manifest, cfg)

        # Alex has fallback [chatterbox], Morgan has no fallback
        alex_lines = [h for h, info in manifest["lines"].items() if info["speaker"] == "alex"]
        morgan_lines = [h for h, info in manifest["lines"].items() if info["speaker"] == "morgan"]

        # Alex's lines should succeed via chatterbox
        for h in alex_lines:
            assert manifest["lines"][h]["engine"] == "chatterbox"
            assert manifest["lines"][h]["status"] == "exists"

        # Morgan has no fallback — should fail
        for h in morgan_lines:
            assert manifest["lines"][h]["status"] == "failed"

    def test_existing_file_too_long(self, pipeline_env):
        """Existing file that exceeds duration limit is re-generated."""
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        # Create a too-long file for the first line
        first_hash = list(manifest["lines"].keys())[0]
        first_info = manifest["lines"][first_hash]
        long_audio = _make_audio(24000, 60.0)  # 60s for a short line
        sf.write(str(cfg.tts_dir() / first_info["file"]), long_audio, 24000)

        mock = MockAdapter(duration=1.5)
        with patch.dict(ADAPTERS, {"qwen": lambda: mock}):
            result = generate_missing(manifest, cfg)

        # Should have re-generated the long file
        assert result["generated"] == 3  # all 3, including the replaced one
        new_dur = manifest["lines"][first_hash]["duration"]
        assert new_dur < 5.0  # not 60s anymore

    def test_manifest_updated_after_gen(self, pipeline_env):
        """After generation, manifest reflects new files with durations."""
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        mock = MockAdapter(duration=2.0)
        with patch.dict(ADAPTERS, {"qwen": lambda: mock}):
            generate_missing(manifest, cfg)

        # Save and reload
        manifest_path = cfg.work_dir() / "manifest.json"
        save_manifest(manifest, manifest_path)
        loaded = load_manifest(manifest_path)

        for h, info in loaded["lines"].items():
            assert info["status"] == "exists"
            assert info["duration"] is not None
            assert info["engine"] == "qwen"

    def test_dry_run(self, pipeline_env):
        """Dry run reports what would be generated without calling TTS."""
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        result = generate_missing(manifest, cfg, dry_run=True)

        assert result["dry_run"] is True
        assert result["generated"] == 0
        # No files should be created
        tts_files = list(cfg.tts_dir().glob("*.wav"))
        assert len(tts_files) == 0
        # All lines still missing
        for info in manifest["lines"].values():
            assert info["status"] == "missing"

    def test_segmented_override(self, pipeline_env):
        """Override with segments generates concatenated audio."""
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        # Add a segmented override for the first line
        first_hash = list(manifest["lines"].keys())[0]
        cfg._data["overrides"] = {
            first_hash: {
                "segments": [
                    {"text": "Hello world.", "pause_after": 0.3},
                    {"text": "This is a test."},
                ],
            },
        }

        mock = MockAdapter(duration=1.0)
        with patch.dict(ADAPTERS, {"qwen": lambda: mock}):
            result = generate_missing(manifest, cfg)

        assert result["generated"] == 3
        # The segmented line should be longer than non-segmented ones
        # (2 segments + pause = ~2.3s vs 1.5s)
        first_dur = manifest["lines"][first_hash]["duration"]
        assert first_dur > 1.5

    def test_force_fallback(self, pipeline_env):
        """force_fallback overrides engine for specific lines."""
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        # Force first line to use chatterbox
        first_hash = list(manifest["lines"].keys())[0]
        cfg._data["force_fallback"] = [
            {"hash": first_hash, "engine": "chatterbox", "reason": "test"},
        ]

        qwen_mock = MockAdapter(duration=1.5)
        cb_mock = MockAdapter(duration=1.5)
        with patch.dict(ADAPTERS, {
            "qwen": lambda: qwen_mock,
            "chatterbox": lambda: cb_mock,
        }):
            result = generate_missing(manifest, cfg)

        assert result["generated"] == 3
        assert manifest["lines"][first_hash]["engine"] == "chatterbox"
