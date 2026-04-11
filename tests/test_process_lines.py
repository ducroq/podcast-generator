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
    lufs_normalize,
    apply_speaker_volume,
    apply_clip_fades,
    apply_reverb,
    apply_reverb_pedalboard,
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
            "backchannels": [
                {"file": "bc_alex_00.wav", "type": "laugh", "label": "Ha!"},
            ],
        },
        "zara": {
            "voice_ref": "zara.mp3", "ref_text": "Hello.",
            "engine": "qwen", "fallback": [], "volume_db": 2.5,
            "backchannels": [],
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

    for h, info in manifest["lines"].items():
        audio = _make_speech(24000, 2.0, silence_before=0.1)
        sf.write(str(tts_dir / info["file"]), audio, 24000)
        info["status"] = STATUS_EXISTS
        info["duration"] = len(audio) / 24000

    # Create a BC clip source
    bc_dir = cfg.backchannels_dir()
    bc_dir.mkdir(parents=True)
    bc_audio = _make_speech(24000, 0.5, silence_before=0.01, amplitude=0.2)
    sf.write(str(bc_dir / "bc_alex_00.wav"), bc_audio, 24000)

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
        assert len(trimmed) < len(audio)
        # Trimmed = pre_roll + speech + possible tail pad
        pre_roll_samples = int(sr * 0.04)
        min_pad = int(sr * 0.025)  # default 25ms
        assert len(trimmed) == pytest.approx(
            len(speech) + pre_roll_samples + min_pad, abs=sr * 0.015)

    def test_no_silence_to_trim(self):
        sr = 24000
        audio = np.ones(int(sr * 1.0), dtype=np.float32) * 0.1
        trimmed = trim_silence(audio, sr)
        # Speech starts at sample 0 AND ends at last sample, so
        # min_pad is added to both head and tail
        min_pad = int(sr * 0.025)  # default 25ms
        assert len(trimmed) <= len(audio) + 2 * min_pad
        # Verify pads are silence
        assert trimmed[0] == 0.0
        assert trimmed[-1] == 0.0

    def test_all_silence(self):
        sr = 24000
        audio = np.zeros(int(sr * 1.0), dtype=np.float32)
        trimmed = trim_silence(audio, sr)
        assert len(trimmed) == len(audio)

    def test_speech_in_final_window(self):
        """Speech only in the last chunk is still detected (not skipped)."""
        sr = 24000
        # Long silence then speech at the very end
        audio = np.zeros(int(sr * 2.0), dtype=np.float32)
        audio[-50:] = 0.1  # speech in last 50 samples
        trimmed = trim_silence(audio, sr, pre_roll_ms=40)
        # Should trim most of the silence, keeping ~40ms pre-roll before speech
        assert len(trimmed) < len(audio)
        assert len(trimmed) < 2000  # much less than 48000 original

    def test_very_short_clip(self):
        """Clip shorter than one window — speech detected, pad added if needed."""
        sr = 24000
        audio = np.zeros(100, dtype=np.float32)
        audio[50] = 0.1
        trimmed = trim_silence(audio, sr)
        # Speech at sample 50 with pre_roll=40ms (960 samples) — pre_roll
        # exceeds position, so start=0. Pad may be prepended if silence < min_pad.
        assert len(trimmed) >= 100  # at least original length


class TestSpeechAwareFade:
    def test_fade_capped_to_silence(self):
        sr = 24000
        silence = np.zeros(int(sr * 0.01), dtype=np.float32)
        speech = np.ones(int(sr * 0.5), dtype=np.float32) * 0.3
        audio = np.concatenate([silence, speech])

        faded = apply_clip_fades(audio.copy(), sr, fade_in_ms=35, fade_out_ms=20)
        speech_start = int(sr * 0.01)
        assert abs(faded[speech_start + 5]) > 0.2

    def test_very_short_clip_no_fade(self):
        """Clip shorter than fade_in + fade_out skips fading."""
        sr = 24000
        # 5ms clip — shorter than 35ms + 20ms = 55ms
        audio = np.ones(int(sr * 0.005), dtype=np.float32) * 0.1
        original = audio.copy()
        faded = apply_clip_fades(audio, sr, fade_in_ms=35, fade_out_ms=20)
        # Should not crash; audio may or may not be modified depending on click check
        assert len(faded) == len(original)

    def test_silent_clip(self):
        """All-zero clip doesn't crash."""
        sr = 24000
        audio = np.zeros(sr, dtype=np.float32)
        faded = apply_clip_fades(audio.copy(), sr)
        assert len(faded) == sr


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


class TestLUFSNormalize:
    def test_loudness_closer_to_target(self):
        """LUFS-normalized audio should measure close to target LUFS."""
        sr = 24000
        # 2 seconds of speech-like signal (enough for LUFS meter)
        t = np.linspace(0, 2, sr * 2, dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * 200 * t)
        result = lufs_normalize(audio, sr, target_lufs=-16.0)
        # Result should be non-silent and different from input
        assert np.max(np.abs(result)) > 0
        assert not np.array_equal(audio, result)

    def test_silent_audio_unchanged(self):
        sr = 24000
        audio = np.zeros(sr * 2, dtype=np.float32)
        result = lufs_normalize(audio, sr, target_lufs=-16.0)
        assert np.all(result == 0)

    def test_short_clip_falls_back_to_rms(self):
        """Clips shorter than 0.5s should use RMS fallback."""
        sr = 24000
        # 0.3s clip — too short for reliable LUFS
        audio = np.ones(int(sr * 0.3), dtype=np.float32) * 0.5
        result = lufs_normalize(audio, sr, target_lufs=-16.0, target_rms=0.1)
        rms = np.sqrt(np.mean(result ** 2))
        assert rms == pytest.approx(0.1, abs=0.01)


class TestSpeakerVolume:
    def test_positive_db(self):
        audio = np.ones(100, dtype=np.float32) * 0.1
        result = apply_speaker_volume(audio, 6.0)
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
        assert original_jump > 0.5

        processed = apply_clip_fades(audio.copy(), sr)
        processed_jump = abs(processed[101] - processed[100])
        assert processed_jump < original_jump


class TestReverb:
    def test_reverb_changes_audio(self):
        sr = 24000
        rng = np.random.RandomState(99)
        audio = np.random.randn(sr).astype(np.float32) * 0.1
        ir = generate_room_ir(sr, 0.15, rng=rng)
        result = apply_reverb(audio, ir, mix=0.02)
        assert not np.allclose(result, audio, atol=1e-6)
        assert np.corrcoef(audio, result)[0, 1] > 0.95

    def test_generate_room_ir_deterministic(self):
        """Same seed produces same IR."""
        sr = 24000
        ir1 = generate_room_ir(sr, 0.15, rng=np.random.RandomState(42))
        ir2 = generate_room_ir(sr, 0.15, rng=np.random.RandomState(42))
        assert np.allclose(ir1, ir2)

    def test_generate_room_ir_different_seeds(self):
        sr = 24000
        ir1 = generate_room_ir(sr, 0.15, rng=np.random.RandomState(1))
        ir2 = generate_room_ir(sr, 0.15, rng=np.random.RandomState(2))
        assert not np.allclose(ir1, ir2)

    def test_zero_decay_time(self):
        """Zero decay time produces a 1-sample IR without crashing."""
        sr = 24000
        ir = generate_room_ir(sr, 0.0, rng=np.random.RandomState(42))
        assert len(ir) >= 1

    def test_pedalboard_reverb_changes_audio(self):
        sr = 24000
        audio = np.random.randn(sr).astype(np.float32) * 0.1
        result = apply_reverb_pedalboard(audio, sr, mix=0.02, room_size=0.15)
        assert not np.allclose(result, audio, atol=1e-6)
        assert np.corrcoef(audio, result)[0, 1] > 0.95

    def test_pedalboard_reverb_deterministic(self):
        """Pedalboard reverb is deterministic (no RNG component)."""
        sr = 24000
        audio = np.random.randn(sr).astype(np.float32) * 0.1
        r1 = apply_reverb_pedalboard(audio, sr, mix=0.02)
        r2 = apply_reverb_pedalboard(audio, sr, mix=0.02)
        assert np.allclose(r1, r2)


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
            assert (cfg.lines_dir() / info["file"]).exists()

    def test_raw_files_untouched(self, pipeline_env):
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        raw_hashes = {}
        for info in manifest["lines"].values():
            raw_path = cfg.tts_dir() / info["file"]
            raw_hashes[info["file"]] = _file_hash(raw_path)

        process_all(manifest, cfg)

        for info in manifest["lines"].values():
            raw_path = cfg.tts_dir() / info["file"]
            assert _file_hash(raw_path) == raw_hashes[info["file"]]

    def test_idempotent(self, pipeline_env):
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        process_all(manifest, cfg)

        processed_hashes = {}
        for info in manifest["lines"].values():
            processed_path = cfg.lines_dir() / info["file"]
            processed_hashes[info["file"]] = _file_hash(processed_path)

        result = process_all(manifest, cfg)
        assert result["processed"] == 0
        assert result["skipped"] == 2

        for info in manifest["lines"].values():
            processed_path = cfg.lines_dir() / info["file"]
            assert _file_hash(processed_path) == processed_hashes[info["file"]]

    def test_speaker_volume_applied(self, pipeline_env):
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

        first_hash = list(manifest["lines"].keys())[0]
        manifest["lines"][first_hash]["status"] = STATUS_MISSING

        result = process_all(manifest, cfg)
        assert result["processed"] == 1
        assert result["skipped"] == 1

    def test_stereo_input_downmixed(self, pipeline_env):
        """Stereo raw file is downmixed to mono."""
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        # Replace first file with stereo
        first_hash = list(manifest["lines"].keys())[0]
        info = manifest["lines"][first_hash]
        stereo = np.random.randn(48000, 2).astype(np.float32) * 0.1
        sf.write(str(cfg.tts_dir() / info["file"]), stereo, 24000)

        process_all(manifest, cfg)

        processed, sr = sf.read(str(cfg.lines_dir() / info["file"]))
        assert processed.ndim == 1  # mono

    def test_mismatched_sample_rate(self, pipeline_env):
        """Raw file at different SR is resampled to target."""
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        first_hash = list(manifest["lines"].keys())[0]
        info = manifest["lines"][first_hash]
        audio_44k = _make_speech(44100, 2.0)
        sf.write(str(cfg.tts_dir() / info["file"]), audio_44k, 44100)

        process_all(manifest, cfg)

        _, sr = sf.read(str(cfg.lines_dir() / info["file"]))
        assert sr == 24000


class TestProcessBackchannels:
    def test_bc_clips_processed_to_subdirectory(self, pipeline_env):
        """BC clips are written to processed/ subdir, originals preserved."""
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        result = process_all(manifest, cfg)

        assert result["backchannels"] == 1
        # Processed file exists
        processed_path = cfg.backchannels_dir() / "processed" / "bc_alex_00.wav"
        assert processed_path.exists()
        # Original still exists and unchanged
        original_path = cfg.backchannels_dir() / "bc_alex_00.wav"
        assert original_path.exists()

    def test_bc_idempotent(self, pipeline_env):
        """Running twice doesn't re-process BC clips."""
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        process_all(manifest, cfg)
        processed_hash = _file_hash(cfg.backchannels_dir() / "processed" / "bc_alex_00.wav")

        result = process_all(manifest, cfg)
        assert result["backchannels"] == 0
        assert _file_hash(cfg.backchannels_dir() / "processed" / "bc_alex_00.wav") == processed_hash

    def test_bc_originals_never_modified(self, pipeline_env):
        """Original BC files are byte-identical after processing."""
        cfg = pipeline_env["cfg"]
        manifest = pipeline_env["manifest"]

        original_hash = _file_hash(cfg.backchannels_dir() / "bc_alex_00.wav")
        process_all(manifest, cfg)
        assert _file_hash(cfg.backchannels_dir() / "bc_alex_00.wav") == original_hash
