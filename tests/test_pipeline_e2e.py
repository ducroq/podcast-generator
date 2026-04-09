"""End-to-end smoke tests for the podcast pipeline.

Chains all 5 modules with MockAdapter (no GPU required):
    parse_script → build_manifest → generate_tts → process_lines → mix_podcast

Verifies:
- Modules chain correctly (output of each stage is consumed by the next)
- Final WAV exists with correct properties
- Work directory structure matches the plan
- Manifest is consistent at the end
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import yaml

from manifest import (
    parse_script, build_manifest, save_manifest, load_manifest,
    line_count, missing_lines, section_names,
    STATUS_EXISTS, STATUS_FAILED,
)
from config import load_episode_config
from generate_tts import generate_missing, ADAPTERS
from process_lines import process_all
from mix_podcast import mix_episode


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SR = 24000

PODCAST_YAML = {
    "podcast": {"name": "Smoke Test Podcast"},
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
    "mix": {
        "target_sr": SR,
        "speaker_change_pause": 0.10,
        "same_speaker_pause": 0.05,
        "interjection_pause": 0.03,
        "interjection_threshold": 1.5,
        "section_pause": 0.5,
        "reverb_mix": 0.02,
        "reverb_decay": 0.15,
        "room_tone_level": 0.002,
        "peak_limit_dbtp": -1.0,
        "tail_silence": 0.5,
        "backchannel": {
            "volume_db": -3.0,
            "overlap_ms": [100, 200],
            "duck_threshold": 0.5,
            "duck_level": 0.6,
            "spill_breathing_room": 0.05,
        },
    },
    "processing": {
        "fade_in_ms": 20,
        "fade_out_ms": 10,
        "pre_roll_ms": 20,
        "trim_threshold_db": -35,
        "click_check_ms": 30,
        "click_threshold": 0.08,
        "click_smooth_samples": 5,
        "rms_target": 0.1,
    },
    "cast": {
        "alex": {
            "voice_ref": "alex.mp3",
            "ref_text": "Hello from Alex.",
            "engine": "qwen",
            "fallback": ["chatterbox"],
            "volume_db": 0.0,
            "backchannels": [
                {"file": "bc_alex_00.wav", "type": "laugh", "label": "Ha!"},
            ],
        },
        "morgan": {
            "voice_ref": "morgan.mp3",
            "ref_text": "Hello from Morgan.",
            "engine": "qwen",
            "fallback": ["chatterbox"],
            "volume_db": 0.0,
            "backchannels": [
                {"file": "bc_morgan_00.wav", "type": "breath", "label": "(intake)"},
            ],
        },
        "zara": {
            "voice_ref": "zara.mp3",
            "ref_text": "Hello from Zara.",
            "engine": "qwen",
            "fallback": [],
            "volume_db": 2.0,
            "backchannels": [],
        },
    },
    "music": {
        "intro_bed": {
            "file": "music/bed.wav",
            "music_solo": 1.0,
            "fade_in": 0.5,
            "full_vol": 0.35,
            "duck_vol": 0.12,
            "post_voice": 1.0,
            "bleed_into_cold": 2.0,
        },
        "sting": {
            "file": "music/sting.wav",
            "fade_in": 0.3,
            "crossfade": 0.5,
            "vol_under_cold": 0.35,
            "cold_open_overlap": 1.0,
        },
    },
    "review": {"agents": []},
}

EPISODE_YAML = {
    "episode": {"number": 1, "title": "Smoke Test", "slug": "smoke"},
    "podcast": "smoke-pod",
    "script": "scripts/smoke.txt",
    "intro_lines": "scripts/smoke_intro.txt",
    "work_dir": "work/smoke",
    "aliases": {"narrator": "alex"},
    "overrides": {},
    "force_fallback": [],
}

SCRIPT_TEXT = """\
====================
COLD OPEN
====================

Alex: [earnest] Welcome to the smoke test. This is the cold open.

Morgan: [dry] A cold open response.

[Pause.]

====================
MAIN CONVERSATION
====================

Alex: Now we are in the main conversation.

Morgan: Indeed we are. Let me tell you something longer to test pacing.
[react: alex laugh]

Zara: [quick] Short.

Alex: And that wraps up our test episode.

[Beat.]

====================
CLOSE
====================

Alex: [quiet] Final thought for the listener.

[Silence.]

====================
END
====================
"""

INTRO_TEXT = """\
Alex: [warm] Welcome to the Smoke Test Podcast.
Morgan: [warm] Hello.
"""


def _make_audio(sr=SR, duration=1.5):
    samples = int(sr * duration)
    return (np.random.randn(samples) * 0.1).astype(np.float32)


class MockAdapter:
    """Mock TTS adapter — returns synthetic audio, no GPU."""
    name = "mock"

    def __init__(self, duration=1.5, sr=SR, fail_count=0):
        self.duration = duration
        self.sr = sr
        self.fail_count = fail_count
        self._calls = 0

    def load(self, **kwargs):
        pass

    def generate(self, text, voice_ref, ref_text=None, language=None, **kwargs):
        self._calls += 1
        if self._calls <= self.fail_count:
            return _make_audio(self.sr, 60.0), self.sr  # hallucination
        # Duration proportional to word count for realism
        words = len(text.split())
        dur = max(0.5, words * 0.15)
        return _make_audio(self.sr, dur), self.sr

    def unload(self):
        pass


@pytest.fixture
def e2e_env(tmp_path):
    """Set up the full pipeline environment."""
    # Podcast config
    pod_dir = tmp_path / "podcasts" / "smoke-pod"
    pod_dir.mkdir(parents=True)
    with open(pod_dir / "podcast.yaml", "w") as f:
        yaml.dump(PODCAST_YAML, f)

    # Episode config
    ep_dir = tmp_path / "podcasts" / "episodes"
    ep_dir.mkdir(parents=True)
    ep_path = ep_dir / "smoke.yaml"
    with open(ep_path, "w") as f:
        yaml.dump(EPISODE_YAML, f)

    # Script
    scripts_dir = tmp_path / "podcasts" / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "smoke.txt").write_text(SCRIPT_TEXT, encoding="utf-8")
    (scripts_dir / "smoke_intro.txt").write_text(INTRO_TEXT, encoding="utf-8")

    # Voice refs (dummy)
    voice_dir = tmp_path / "podcasts" / "voice_refs"
    voice_dir.mkdir(parents=True)
    for name in ("alex.mp3", "morgan.mp3", "zara.mp3"):
        sf.write(str(voice_dir / name), _make_audio(duration=3.0), SR)

    # Music assets
    music_dir = tmp_path / "podcasts" / "music"
    music_dir.mkdir(parents=True)
    sf.write(str(music_dir / "bed.wav"), _make_audio(duration=15.0), SR)
    sf.write(str(music_dir / "sting.wav"), _make_audio(duration=3.0), SR)

    # BC clips (raw — process_lines will create processed/ copies)
    cfg = load_episode_config(ep_path, podcast_dir=pod_dir)
    bc_dir = cfg.backchannels_dir()
    bc_dir.mkdir(parents=True)
    sf.write(str(bc_dir / "bc_alex_00.wav"), _make_audio(duration=0.8), SR)
    sf.write(str(bc_dir / "bc_morgan_00.wav"), _make_audio(duration=0.5), SR)

    return {"cfg": cfg, "ep_path": ep_path, "pod_dir": pod_dir}


# ---------------------------------------------------------------------------
# Happy path: full pipeline
# ---------------------------------------------------------------------------


class TestFullPipelineHappyPath:
    """Chain all 5 modules end-to-end with MockAdapter."""

    def test_full_chain(self, e2e_env):
        cfg = e2e_env["cfg"]

        # --- Stage 1: Parse + Manifest ---
        entries = parse_script(cfg.script_path())
        manifest = build_manifest(
            entries,
            audio_dir=cfg.tts_dir(),
            script_path=str(cfg.script_path()),
            episode="smoke",
        )
        assert line_count(manifest) == 7  # 2 cold + 4 conv + 1 close
        assert len(missing_lines(manifest)) == line_count(manifest)  # all missing
        assert len(section_names(manifest)) >= 3  # COLD OPEN, MAIN CONVERSATION, CLOSE (+ END)

        # --- Stage 2: Config already loaded ---
        assert cfg.cast("alex")["engine"] == "qwen"
        assert cfg.cast("narrator")["engine"] == "qwen"  # alias

        # --- Stage 3: Generate TTS ---
        with patch.dict(ADAPTERS, {"qwen": lambda: MockAdapter()}):
            gen_result = generate_missing(manifest, cfg)

        assert gen_result["generated"] == line_count(manifest)
        assert gen_result["failed"] == 0
        assert len(missing_lines(manifest)) == 0
        for info in manifest["lines"].values():
            assert info["status"] == STATUS_EXISTS
            assert info["duration"] is not None
            assert (cfg.tts_dir() / info["file"]).exists()

        # Save manifest for next stages
        manifest_path = cfg.work_dir() / "manifest.json"
        save_manifest(manifest, manifest_path)

        # --- Stage 4: Process lines ---
        proc_result = process_all(manifest, cfg)

        assert proc_result["processed"] == line_count(manifest)
        for info in manifest["lines"].values():
            assert (cfg.lines_dir() / info["file"]).exists()
        # Raw TTS files still exist
        for info in manifest["lines"].values():
            assert (cfg.tts_dir() / info["file"]).exists()
        # BC clips processed
        assert proc_result["backchannels"] >= 1
        assert (cfg.backchannels_dir() / "processed").exists()

        # --- Stage 5: Mix ---
        mix_result = mix_episode(manifest, cfg, seed=42)

        assert Path(mix_result["output"]).exists()
        assert mix_result["duration"] > 0
        assert len(mix_result["sections"]) >= 3

        # Verify output WAV properties
        audio, sr = sf.read(mix_result["output"])
        assert sr == SR
        assert len(audio) > 0
        peak = np.max(np.abs(audio))
        assert peak <= 10 ** (-1.0 / 20) + 0.001  # peak limited

        # Section WAVs exist
        section_files = list(cfg.sections_dir().glob("*.wav"))
        assert len(section_files) >= 3

        # Manifest is still consistent
        loaded = load_manifest(manifest_path)
        assert all(info["status"] == STATUS_EXISTS for info in loaded["lines"].values())

    def test_work_directory_structure(self, e2e_env):
        """Verify the work directory matches the planned layout."""
        cfg = e2e_env["cfg"]

        entries = parse_script(cfg.script_path())
        manifest = build_manifest(entries, script_path=str(cfg.script_path()))

        with patch.dict(ADAPTERS, {"qwen": lambda: MockAdapter()}):
            generate_missing(manifest, cfg)

        process_all(manifest, cfg)
        mix_episode(manifest, cfg, seed=42)

        work = cfg.work_dir()
        assert (work / "tts").is_dir()
        assert (work / "lines").is_dir()
        assert (work / "sections").is_dir()
        assert (work / "backchannels").is_dir()
        assert (work / "backchannels" / "processed").is_dir()
        assert (work / "mix.wav").is_file()

        # Count files
        tts_files = list((work / "tts").glob("*.wav"))
        lines_files = list((work / "lines").glob("*.wav"))
        assert len(tts_files) == len(lines_files)  # same count, same names
        assert len(tts_files) == line_count(manifest)


# ---------------------------------------------------------------------------
# Fallback path: primary engine fails
# ---------------------------------------------------------------------------


class TestPipelineWithFallback:
    """One speaker's primary engine always hallucinates → fallback fires."""

    def test_fallback_produces_valid_output(self, e2e_env):
        cfg = e2e_env["cfg"]

        entries = parse_script(cfg.script_path())
        manifest = build_manifest(entries, script_path=str(cfg.script_path()))

        # Qwen hallucinates, Chatterbox works
        with patch.dict(ADAPTERS, {
            "qwen": lambda: MockAdapter(fail_count=100),
            "chatterbox": lambda: MockAdapter(),
        }):
            gen_result = generate_missing(manifest, cfg)

        # Alex and Morgan have fallback [chatterbox] — should succeed
        # Zara has no fallback — should fail
        alex_morgan_lines = [
            h for h, info in manifest["lines"].items()
            if info["speaker"] in ("alex", "morgan")
        ]
        zara_lines = [
            h for h, info in manifest["lines"].items()
            if info["speaker"] == "zara"
        ]

        for h in alex_morgan_lines:
            assert manifest["lines"][h]["status"] == STATUS_EXISTS
            assert manifest["lines"][h]["engine"] == "chatterbox"

        for h in zara_lines:
            assert manifest["lines"][h]["status"] == STATUS_FAILED

        # Process and mix still work (failed lines become silence)
        process_all(manifest, cfg)
        mix_result = mix_episode(manifest, cfg, seed=42)

        assert Path(mix_result["output"]).exists()
        assert mix_result["duration"] > 0


# ---------------------------------------------------------------------------
# With intro voice
# ---------------------------------------------------------------------------


class TestPipelineWithIntro:
    """Full pipeline with intro voice produces longer output."""

    def test_intro_adds_to_duration(self, e2e_env):
        cfg = e2e_env["cfg"]

        entries = parse_script(cfg.script_path())
        manifest = build_manifest(entries, script_path=str(cfg.script_path()))

        with patch.dict(ADAPTERS, {"qwen": lambda: MockAdapter()}):
            generate_missing(manifest, cfg)

        process_all(manifest, cfg)

        # Without intro
        result_no_intro = mix_episode(manifest, cfg, seed=42)
        dur_no_intro = result_no_intro["duration"]

        # With intro
        intro_voice = _make_audio(duration=3.0)
        result_with_intro = mix_episode(
            manifest, cfg, intro_voice=intro_voice, seed=42,
        )
        dur_with_intro = result_with_intro["duration"]

        # Intro adds: music_solo (1.0) + voice (3.0) + post_voice (1.0) + pause (1.5) = ~6.5s extra
        assert dur_with_intro > dur_no_intro + 3.0

        # Output is valid
        audio, sr = sf.read(result_with_intro["output"])
        assert sr == SR
        assert np.max(np.abs(audio)) <= 10 ** (-1.0 / 20) + 0.001


# ---------------------------------------------------------------------------
# Manifest consistency
# ---------------------------------------------------------------------------


class TestManifestConsistency:
    """Manifest remains valid through the full pipeline."""

    def test_manifest_roundtrip(self, e2e_env):
        """Save/load manifest at each stage — always valid."""
        cfg = e2e_env["cfg"]
        manifest_path = cfg.work_dir() / "manifest.json"

        entries = parse_script(cfg.script_path())
        manifest = build_manifest(entries, script_path=str(cfg.script_path()), episode="smoke")
        save_manifest(manifest, manifest_path)

        # After generation
        with patch.dict(ADAPTERS, {"qwen": lambda: MockAdapter()}):
            generate_missing(manifest, cfg)
        save_manifest(manifest, manifest_path)
        loaded = load_manifest(manifest_path)
        assert all(info["status"] == STATUS_EXISTS for info in loaded["lines"].values())

        # After processing
        process_all(manifest, cfg)
        save_manifest(manifest, manifest_path)
        loaded = load_manifest(manifest_path)
        assert loaded["meta"]["episode"] == "smoke"

    def test_rerun_skips_existing(self, e2e_env):
        """Running the pipeline twice skips already-generated/processed files."""
        cfg = e2e_env["cfg"]

        entries = parse_script(cfg.script_path())
        manifest = build_manifest(entries, script_path=str(cfg.script_path()))

        # First run
        with patch.dict(ADAPTERS, {"qwen": lambda: MockAdapter()}):
            gen1 = generate_missing(manifest, cfg)
        proc1 = process_all(manifest, cfg)

        # Second run — everything should be skipped
        with patch.dict(ADAPTERS, {"qwen": lambda: MockAdapter()}):
            gen2 = generate_missing(manifest, cfg)
        proc2 = process_all(manifest, cfg)

        assert gen1["generated"] > 0
        assert gen2["generated"] == 0
        assert gen2["skipped"] == line_count(manifest)

        assert proc1["processed"] > 0
        assert proc2["processed"] == 0
        assert proc2["skipped"] == line_count(manifest)
