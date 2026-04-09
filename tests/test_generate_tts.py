"""Tests for generator/generate_tts.py — staged TTS with fallback chain.

All TTS engine calls are mocked — no GPU required.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import yaml

from generate_tts import (
    estimate_max_duration,
    _is_hallucinated,
    _get_adapter,
    _generate_with_guard,
    _generate_segmented,
    _try_fallback,
    _resample_if_needed,
    _resolve_engine,
    _get_line_tts_config,
    generate_missing,
    QwenAdapter,
    ChatterboxAdapter,
    ADAPTERS,
)
from manifest import (
    parse_script, build_manifest, save_manifest, load_manifest,
    STATUS_EXISTS, STATUS_FAILED, STATUS_MISSING,
)
from config import load_episode_config


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
    pod_dir = tmp_path / "podcasts" / "test-pod"
    pod_dir.mkdir(parents=True)
    with open(pod_dir / "podcast.yaml", "w") as f:
        yaml.dump(PODCAST_YAML, f)

    ep_dir = tmp_path / "podcasts" / "episodes"
    ep_dir.mkdir(parents=True)
    ep_path = ep_dir / "ep_test.yaml"
    with open(ep_path, "w") as f:
        yaml.dump(EPISODE_YAML, f)

    scripts_dir = tmp_path / "podcasts" / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "test.txt").write_text(SCRIPT_TEXT, encoding="utf-8")

    voice_dir = tmp_path / "podcasts" / "voice_refs"
    voice_dir.mkdir(parents=True)
    for name in ("alex.mp3", "morgan.mp3"):
        audio = _make_audio(duration=5.0)
        sf.write(str(voice_dir / name), audio, 24000)

    tts_dir = tmp_path / "podcasts" / "work" / "test" / "tts"
    tts_dir.mkdir(parents=True)

    cfg = load_episode_config(ep_path, podcast_dir=pod_dir)
    entries = parse_script(cfg.script_path())
    manifest = build_manifest(entries, audio_dir=cfg.tts_dir(),
                              script_path=str(cfg.script_path()), episode="ep_test")

    return {"cfg": cfg, "manifest": manifest, "tts_dir": tts_dir}


class MockAdapter:
    """Mock TTS adapter that returns predictable audio."""

    name = "mock"

    def __init__(self, duration=1.5, sr=24000, fail_count=0):
        self.duration = duration
        self.sr = sr
        self.fail_count = fail_count
        self._calls = 0
        self.loaded = False

    def load(self, **kwargs):
        self.loaded = True

    def generate(self, text, voice_ref, ref_text=None, language=None, **kwargs):
        self._calls += 1
        if self._calls <= self.fail_count:
            return _make_audio(self.sr, 60.0), self.sr
        return _make_audio(self.sr, self.duration), self.sr

    def unload(self):
        self.loaded = False


# ---------------------------------------------------------------------------
# Hallucination guard
# ---------------------------------------------------------------------------


class TestHallucinationGuard:
    def test_estimate_max_duration_floor(self):
        assert estimate_max_duration("hello world") == 10.0

    def test_estimate_max_duration_above_floor(self):
        assert estimate_max_duration("a " * 30) == 30 * 0.6

    def test_estimate_max_duration_boundary(self):
        # 17 words * 0.6 = 10.2 > floor of 10
        assert estimate_max_duration("word " * 17) == pytest.approx(17 * 0.6)

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
            {"temperature": 0.7}, {"max_duration_per_word": 0.6, "retry_count": 2},
        )
        assert audio is not None
        assert attempts == 1

    def test_generate_with_guard_retry_success(self):
        adapter = MockAdapter(duration=1.5, fail_count=1)
        adapter.load()
        audio, sr, attempts = _generate_with_guard(
            adapter, "Hello world", "/fake/ref.mp3", "ref text", "English",
            {"temperature": 0.7}, {"max_duration_per_word": 0.6, "retry_count": 3},
        )
        assert audio is not None
        assert attempts == 2

    def test_generate_with_guard_all_retries_fail(self):
        adapter = MockAdapter(duration=1.5, fail_count=100)
        adapter.load()
        audio, sr, attempts = _generate_with_guard(
            adapter, "Hello world", "/fake/ref.mp3", "ref text", "English",
            {"temperature": 0.7}, {"max_duration_per_word": 0.6, "retry_count": 2},
        )
        assert audio is None
        assert attempts == 3


# ---------------------------------------------------------------------------
# Segmented generation
# ---------------------------------------------------------------------------


class TestGenerateSegmented:
    def test_happy_path(self):
        adapter = MockAdapter(duration=1.0)
        adapter.load()
        segments = [
            {"text": "First part.", "pause_after": 0.3},
            {"text": "Second part."},
        ]
        audio, sr = _generate_segmented(
            adapter, segments, "/fake.mp3", "ref", "English",
            {"temperature": 0.7}, {"max_duration_per_word": 0.6}, 24000,
        )
        assert audio is not None
        # 1.0 + 0.3 pause + 1.0 = ~2.3s
        assert len(audio) / sr == pytest.approx(2.3, abs=0.1)

    def test_segment_hallucination_returns_none(self):
        adapter = MockAdapter(duration=1.0, fail_count=100)
        adapter.load()
        segments = [{"text": "Fails."}]
        audio, sr = _generate_segmented(
            adapter, segments, "/fake.mp3", "ref", "English",
            {"temperature": 0.7}, {"max_duration_per_word": 0.6, "retry_count": 1}, 24000,
        )
        assert audio is None

    def test_empty_segments_returns_none(self):
        adapter = MockAdapter()
        adapter.load()
        audio, sr = _generate_segmented(
            adapter, [], "/fake.mp3", "ref", "English",
            {"temperature": 0.7}, {}, 24000,
        )
        assert audio is None


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


class TestTryFallback:
    def test_fallback_success(self):
        mock = MockAdapter(duration=1.5)
        with patch.dict(ADAPTERS, {"chatterbox": lambda: mock}):
            info = {"speaker": "alex", "text": "Hello"}
            cfg = type("Cfg", (), {
                "voice_ref_path": lambda self, s: "/fake.mp3",
                "cast": lambda self, s: {"ref_text": "ref"},
            })()
            audio, sr, engine = _try_fallback(
                info, ["chatterbox"], cfg, "English",
                {"temperature": 0.7}, {"max_duration_per_word": 0.6}, None, 24000,
            )
        assert audio is not None
        assert engine == "chatterbox"

    def test_fallback_exhausted(self):
        mock = MockAdapter(duration=1.5, fail_count=100)
        with patch.dict(ADAPTERS, {"chatterbox": lambda: mock}):
            info = {"speaker": "alex", "text": "Hello"}
            cfg = type("Cfg", (), {
                "voice_ref_path": lambda self, s: "/fake.mp3",
                "cast": lambda self, s: {"ref_text": "ref"},
            })()
            audio, sr, engine = _try_fallback(
                info, ["chatterbox"], cfg, "English",
                {"temperature": 0.7}, {"max_duration_per_word": 0.6, "retry_count": 1},
                None, 24000,
            )
        assert audio is None
        assert engine is None

    def test_fallback_exception_continues(self):
        """Fallback engine that raises tries next engine."""
        class FailAdapter:
            name = "fail"
            def load(self, **kw): pass
            def generate(self, *a, **kw): raise RuntimeError("boom")
            def unload(self): pass

        good = MockAdapter(duration=1.5)
        with patch.dict(ADAPTERS, {"fail": lambda: FailAdapter(), "good": lambda: good}):
            info = {"speaker": "alex", "text": "Hello"}
            cfg = type("Cfg", (), {
                "voice_ref_path": lambda self, s: "/fake.mp3",
                "cast": lambda self, s: {"ref_text": "ref"},
            })()
            audio, sr, engine = _try_fallback(
                info, ["fail", "good"], cfg, "English",
                {"temperature": 0.7}, {"max_duration_per_word": 0.6}, None, 24000,
            )
        assert audio is not None
        assert engine == "good"


# ---------------------------------------------------------------------------
# Engine adapters
# ---------------------------------------------------------------------------


class TestAdapters:
    def test_get_adapter_qwen(self):
        assert isinstance(_get_adapter("qwen"), QwenAdapter)

    def test_get_adapter_chatterbox(self):
        assert isinstance(_get_adapter("chatterbox"), ChatterboxAdapter)

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

    def test_resample_down(self):
        audio = _make_audio(48000, 1.0)
        result = _resample_if_needed(audio, 48000, 24000)
        assert len(result) == pytest.approx(24000, abs=10)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestResolveEngine:
    def test_default_from_cast(self, pipeline_env):
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]
        h = list(manifest["lines"].keys())[0]
        info = manifest["lines"][h]
        engine = _resolve_engine(h, info, {}, cfg)
        assert engine == "qwen"

    def test_force_fallback_overrides(self, pipeline_env):
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]
        h = list(manifest["lines"].keys())[0]
        info = manifest["lines"][h]
        engine = _resolve_engine(h, info, {h: "chatterbox"}, cfg)
        assert engine == "chatterbox"

    def test_override_engine(self, pipeline_env):
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]
        h = list(manifest["lines"].keys())[0]
        info = manifest["lines"][h]
        cfg._data["overrides"] = {h: {"engine": "chatterbox"}}
        engine = _resolve_engine(h, info, {}, cfg)
        assert engine == "chatterbox"


class TestGetLineTtsConfig:
    def test_no_override(self, pipeline_env):
        cfg = pipeline_env["cfg"]
        base = {"temperature": 0.7, "repetition_penalty": 1.2}
        result = _get_line_tts_config("nonexistent_hash", cfg, base)
        assert result["temperature"] == 0.7

    def test_with_temperature_override(self, pipeline_env):
        cfg = pipeline_env["cfg"]
        h = list(pipeline_env["manifest"]["lines"].keys())[0]
        cfg._data["overrides"] = {h: {"temperature": 0.4}}
        base = {"temperature": 0.7, "repetition_penalty": 1.2}
        result = _get_line_tts_config(h, cfg, base)
        assert result["temperature"] == 0.4
        assert result["repetition_penalty"] == 1.2


# ---------------------------------------------------------------------------
# generate_missing — integration tests with mock adapters
# ---------------------------------------------------------------------------


class TestGenerateMissing:
    def test_skip_existing_lines(self, pipeline_env):
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        first_hash = list(manifest["lines"].keys())[0]
        first_info = manifest["lines"][first_hash]
        audio = _make_audio(24000, 2.0)
        sf.write(str(cfg.tts_dir() / first_info["file"]), audio, 24000)

        entries = parse_script(cfg.script_path())
        manifest = build_manifest(entries, audio_dir=cfg.tts_dir())

        with patch.dict(ADAPTERS, {"qwen": lambda: MockAdapter()}):
            result = generate_missing(manifest, cfg)

        assert result["skipped"] >= 1
        assert manifest["lines"][first_hash]["status"] == STATUS_EXISTS

    def test_generate_missing_line(self, pipeline_env):
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        with patch.dict(ADAPTERS, {"qwen": lambda: MockAdapter()}):
            result = generate_missing(manifest, cfg)

        assert result["generated"] == 3
        assert result["failed"] == 0
        for info in manifest["lines"].values():
            assert info["status"] == STATUS_EXISTS
            assert info["duration"] is not None
            assert info["engine"] == "qwen"

    def test_hallucination_triggers_fallback(self, pipeline_env):
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        with patch.dict(ADAPTERS, {
            "qwen": lambda: MockAdapter(fail_count=100),
            "chatterbox": lambda: MockAdapter(duration=1.5),
        }):
            result = generate_missing(manifest, cfg)

        alex_lines = [h for h, i in manifest["lines"].items() if i["speaker"] == "alex"]
        morgan_lines = [h for h, i in manifest["lines"].items() if i["speaker"] == "morgan"]

        for h in alex_lines:
            assert manifest["lines"][h]["engine"] == "chatterbox"
            assert manifest["lines"][h]["status"] == STATUS_EXISTS

        for h in morgan_lines:
            assert manifest["lines"][h]["status"] == STATUS_FAILED

    def test_fallback_resamples_correctly(self, pipeline_env):
        """Fallback engine at different SR gets resampled to target."""
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        # Qwen at 24kHz hallucinates, Chatterbox at 16kHz succeeds
        with patch.dict(ADAPTERS, {
            "qwen": lambda: MockAdapter(fail_count=100, sr=24000),
            "chatterbox": lambda: MockAdapter(duration=1.5, sr=16000),
        }):
            result = generate_missing(manifest, cfg)

        # Check Alex lines were resampled to 24kHz
        for h, info in manifest["lines"].items():
            if info["speaker"] == "alex" and info["status"] == STATUS_EXISTS:
                wav, sr = sf.read(str(cfg.tts_dir() / info["file"]))
                assert sr == 24000

    def test_existing_file_too_long(self, pipeline_env):
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        first_hash = list(manifest["lines"].keys())[0]
        first_info = manifest["lines"][first_hash]
        sf.write(str(cfg.tts_dir() / first_info["file"]), _make_audio(24000, 60.0), 24000)

        with patch.dict(ADAPTERS, {"qwen": lambda: MockAdapter()}):
            result = generate_missing(manifest, cfg)

        assert result["generated"] == 3
        assert manifest["lines"][first_hash]["duration"] < 5.0

    def test_manifest_updated_after_gen(self, pipeline_env):
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        with patch.dict(ADAPTERS, {"qwen": lambda: MockAdapter(duration=2.0)}):
            generate_missing(manifest, cfg)

        manifest_path = cfg.work_dir() / "manifest.json"
        save_manifest(manifest, manifest_path)
        loaded = load_manifest(manifest_path)

        for info in loaded["lines"].values():
            assert info["status"] == STATUS_EXISTS
            assert info["duration"] is not None

    def test_dry_run(self, pipeline_env):
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        result = generate_missing(manifest, cfg, dry_run=True)

        assert result["dry_run"] is True
        assert result["generated"] == 0
        assert len(list(cfg.tts_dir().glob("*.wav"))) == 0
        for info in manifest["lines"].values():
            assert info["status"] == STATUS_MISSING

    def test_segmented_override(self, pipeline_env):
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        first_hash = list(manifest["lines"].keys())[0]
        cfg._data["overrides"] = {
            first_hash: {
                "segments": [
                    {"text": "Hello.", "pause_after": 0.3},
                    {"text": "World."},
                ],
            },
        }

        with patch.dict(ADAPTERS, {"qwen": lambda: MockAdapter(duration=1.0)}):
            result = generate_missing(manifest, cfg)

        assert result["generated"] == 3
        assert manifest["lines"][first_hash]["duration"] > 1.5

    def test_force_fallback(self, pipeline_env):
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        first_hash = list(manifest["lines"].keys())[0]
        cfg._data["force_fallback"] = [
            {"hash": first_hash, "engine": "chatterbox", "reason": "test"},
        ]

        with patch.dict(ADAPTERS, {
            "qwen": lambda: MockAdapter(),
            "chatterbox": lambda: MockAdapter(),
        }):
            result = generate_missing(manifest, cfg)

        assert manifest["lines"][first_hash]["engine"] == "chatterbox"

    def test_per_line_temperature_override(self, pipeline_env):
        """Per-line temperature from config.overrides is applied."""
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        first_hash = list(manifest["lines"].keys())[0]
        cfg._data["overrides"] = {first_hash: {"temperature": 0.3}}

        calls = []
        class SpyAdapter(MockAdapter):
            def generate(self, text, voice_ref, ref_text=None, language=None, **kwargs):
                calls.append(kwargs.get("temperature"))
                return super().generate(text, voice_ref, ref_text, language, **kwargs)

        with patch.dict(ADAPTERS, {"qwen": lambda: SpyAdapter()}):
            generate_missing(manifest, cfg)

        # First line should have temp=0.3, others temp=0.7
        assert 0.3 in calls
        assert 0.7 in calls

    def test_exception_marks_failed(self, pipeline_env):
        """Exception during generation marks line as failed."""
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        class BoomAdapter(MockAdapter):
            def generate(self, *a, **kw):
                raise RuntimeError("GPU exploded")

        with patch.dict(ADAPTERS, {"qwen": lambda: BoomAdapter()}):
            result = generate_missing(manifest, cfg)

        assert result["failed"] == 3
        for info in manifest["lines"].values():
            assert info["status"] == STATUS_FAILED
            assert info["engine"] is None
