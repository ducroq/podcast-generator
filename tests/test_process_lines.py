"""Tests for generator/process_lines.py — batch audio processing."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from process_lines import (
    trim_silence,
    rms_normalize,
    apply_speaker_volume,
    apply_clip_fades,
    apply_reverb,
    generate_room_ir,
    process_one,
    process_all,
)
from manifest import parse_script, build_manifest, STATUS_EXISTS, STATUS_MISSING
from config import load_episode_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_speech(sr=24000, duration=2.0, silence_before=0.1, amplitude=0.3):
    """Create test audio: silence + speech-like signal."""
    silence = np.zeros(int(sr * silence_before), dtype=np.float32)
    speech = (np.random.randn(int(sr * duration)) * amplitude).astype(np.float32)
    return np.concatenate([silence, speech])


def _make_click_audio(sr=24000, duration=1.0):
    """Create test audio with an artificial click at the start."""
    audio = (np.random.randn(int(sr * duration)) * 0.05).astype(np.float32)
    # Insert a large jump at sample 100
    audio[100] = 0.5
    audio[101] = -0.3
    return audio


def _file_hash(path):
    """Get SHA256 of a file's contents."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PODCAST_YAML = {
    "podcast": {"name": "Test"},
    "tts": {"language": "English", "temperature": 0.7, "repetition_penalty": 1.2},
    "mix": {
        "target_sr": 24000,
        "reverb_mix": 0.02,
        "reverb_decay": 0.15,
        "backchannel": {},
    },
    "processing": {
        "fade_in_ms": 35,
        "fade_out_ms": 20,
        "pre_roll_ms": 40,
        "trim_threshold_db": -35,
        "click_check_ms": 50,
        "click_threshold": 0.08,
        "click_smooth_samples": 7,
        "rms_target": 0.1,
    },
    "cast": {
        "alex": {
            "voice_ref": "alex.mp3", "ref_text": "Hello.",
            "engine": "qwen", "fallback": [], "volume_db": 0.0,
        },
        "zara": {
            "voice_ref": "zara.mp3", "ref_text": "Hello.",
            "engine": "qwen", "fallback": [], "volume_db": 2.5,
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

Alex: Hello world test line.

Zara: Another test line from Zara.
"""


@pytest.fixture
def pipeline_env(tmp_path):
    """Full pipeline env with raw TTS files in tts_dir."""
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

    cfg = load_episode_config(ep_path, podcast_dir=pod_dir)

    # Create raw TTS files
    tts_dir = cfg.tts_dir()
    tts_dir.mkdir(parents=True)

    entries = parse_script(cfg.script_path())
    manifest = build_manifest(entries, script_path=str(cfg.script_path()))

    # Write a raw WAV for each line
    for h, info in manifest["lines"].items():
        audio = _make_speech(24000, 2.0, silence_before=0.1)
        sf.write(str(tts_dir / info["file"]), audio, 24000)
        info["status"] = STATUS_EXISTS
        info["duration"] = len(audio) / 24000

    return {"cfg": cfg, "manifest": manifest}


# ---------------------------------------------------------------------------
# Unit tests — individual processing functions
# ---------------------------------------------------------------------------


class TestTrimSilence:
    def test_trims_leading_silence(self):
        sr = 24000
        silence = np.zeros(int(sr * 0.5), dtype=np.float32)
        speech = np.ones(int(sr * 1.0), dtype=np.float32) * 0.1
        audio = np.concatenate([silence, speech])

        trimmed = trim_silence(audio, sr, threshold_db=-35, pre_roll_ms=40)
        # Should be much shorter than original (0.5s silence removed)
        assert len(trimmed) < len(audio)
        # Pre-roll: ~40ms of silence before speech
        pre_roll_samples = int(sr * 0.04)
        assert len(trimmed) == pytest.approx(len(speech) + pre_roll_samples, abs=sr * 0.015)

    def test_no_silence_to_trim(self):
        sr = 24000
        audio = np.ones(int(sr * 1.0), dtype=np.float32) * 0.1
        trimmed = trim_silence(audio, sr)
        # Pre-roll trim still applies but speech starts at sample 0
        assert len(trimmed) <= len(audio)

    def test_all_silence(self):
        sr = 24000
        audio = np.zeros(int(sr * 1.0), dtype=np.float32)
        trimmed = trim_silence(audio, sr)
        # Returns original when no speech detected
        assert len(trimmed) == len(audio)


class TestSpeechAwareFade:
    def test_fade_capped_to_silence(self):
        """Fade-in should not exceed leading silence duration."""
        sr = 24000
        # 10ms silence + speech
        silence = np.zeros(int(sr * 0.01), dtype=np.float32)
        speech = np.ones(int(sr * 0.5), dtype=np.float32) * 0.3
        audio = np.concatenate([silence, speech])

        faded = apply_clip_fades(audio.copy(), sr, fade_in_ms=35, fade_out_ms=20)
        # The speech at 10ms should not be attenuated much by the fade
        speech_start = int(sr * 0.01)
        # First speech sample should retain most of its amplitude
        assert abs(faded[speech_start + 5]) > 0.2  # still audible


class TestRMSNormalize:
    def test_normalize_to_target(self):
        audio = np.ones(24000, dtype=np.float32) * 0.5
        result = rms_normalize(audio, target_rms=0.1)
        rms = np.sqrt(np.mean(result ** 2))
        assert rms == pytest.approx(0.1, abs=0.001)

    def test_silent_audio_unchanged(self):
        audio = np.zeros(24000, dtype=np.float32)
        result = rms_normalize(audio, target_rms=0.1)
        assert np.all(result == 0)


class TestSpeakerVolume:
    def test_positive_db(self):
        audio = np.ones(100, dtype=np.float32) * 0.1
        result = apply_speaker_volume(audio, 6.0)
        # +6dB ≈ 2x amplitude
        assert np.mean(result) == pytest.approx(0.2, abs=0.01)

    def test_zero_db(self):
        audio = np.ones(100, dtype=np.float32) * 0.1
        result = apply_speaker_volume(audio, 0.0)
        assert np.allclose(result, audio)

    def test_negative_db(self):
        audio = np.ones(100, dtype=np.float32) * 0.1
        result = apply_speaker_volume(audio, -6.0)
        assert np.mean(result) == pytest.approx(0.05, abs=0.01)


class TestClickSuppression:
    def test_click_smoothed(self):
        sr = 24000
        audio = _make_click_audio(sr, 1.0)
        original_jump = abs(audio[101] - audio[100])
        assert original_jump > 0.5  # confirm click exists

        processed = apply_clip_fades(audio.copy(), sr)
        processed_jump = abs(processed[101] - processed[100])
        assert processed_jump < original_jump  # click reduced


class TestReverb:
    def test_reverb_changes_audio(self):
        sr = 24000
        audio = np.random.randn(sr).astype(np.float32) * 0.1
        ir = generate_room_ir(sr, 0.15)
        result = apply_reverb(audio, ir, mix=0.02)
        # Result should differ from dry signal
        assert not np.allclose(result, audio, atol=1e-6)
        # But should be close (low wet mix)
        assert np.corrcoef(audio, result)[0, 1] > 0.95


# ---------------------------------------------------------------------------
# Integration — process_all
# ---------------------------------------------------------------------------


class TestProcessAll:
    def test_creates_processed_files(self, pipeline_env):
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        result = process_all(manifest, cfg)

        assert result["processed"] == 2
        for info in manifest["lines"].values():
            processed_path = cfg.lines_dir() / info["file"]
            assert processed_path.exists()

    def test_raw_files_untouched(self, pipeline_env):
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        # Hash raw files before processing
        raw_hashes = {}
        for info in manifest["lines"].values():
            raw_path = cfg.tts_dir() / info["file"]
            raw_hashes[info["file"]] = _file_hash(raw_path)

        process_all(manifest, cfg)

        # Verify raw files are unchanged
        for info in manifest["lines"].values():
            raw_path = cfg.tts_dir() / info["file"]
            assert _file_hash(raw_path) == raw_hashes[info["file"]]

    def test_idempotent(self, pipeline_env):
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        process_all(manifest, cfg)

        # Hash processed files
        processed_hashes = {}
        for info in manifest["lines"].values():
            processed_path = cfg.lines_dir() / info["file"]
            processed_hashes[info["file"]] = _file_hash(processed_path)

        # Run again — should skip (files already exist)
        result = process_all(manifest, cfg)
        assert result["processed"] == 0
        assert result["skipped"] == 2

        # Files unchanged
        for info in manifest["lines"].values():
            processed_path = cfg.lines_dir() / info["file"]
            assert _file_hash(processed_path) == processed_hashes[info["file"]]

    def test_speaker_volume_applied(self, pipeline_env):
        """Zara (+2.5dB) should be louder than Alex (0dB) after processing."""
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        process_all(manifest, cfg)

        alex_rms = []
        zara_rms = []
        for info in manifest["lines"].values():
            audio, sr = sf.read(str(cfg.lines_dir() / info["file"]))
            rms = np.sqrt(np.mean(audio ** 2))
            if info["speaker"] == "alex":
                alex_rms.append(rms)
            elif info["speaker"] == "zara":
                zara_rms.append(rms)

        if alex_rms and zara_rms:
            assert np.mean(zara_rms) > np.mean(alex_rms)

    def test_skips_missing_lines(self, pipeline_env):
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        # Mark one line as missing
        first_hash = list(manifest["lines"].keys())[0]
        manifest["lines"][first_hash]["status"] = STATUS_MISSING

        result = process_all(manifest, cfg)
        assert result["processed"] == 1
        assert result["skipped"] == 1
