"""Tests for generator/mix_podcast.py — podcast episode mixer."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from mix_podcast import (
    build_section,
    build_intro_with_music,
    build_sting_transition,
    crossfade_into_conversation,
    extract_sections,
    generate_pink_noise,
    load_bc_clips,
    mix_episode,
)
from manifest import (
    parse_script, build_manifest, save_manifest,
    STATUS_EXISTS, STATUS_MISSING,
)
from config import load_episode_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 24000


def _make_audio(duration=2.0, amplitude=0.1, sr=SR):
    return (np.random.randn(int(sr * duration)) * amplitude).astype(np.float32)


def _make_tone(duration=1.0, freq=440, sr=SR):
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    return np.sin(2 * np.pi * freq * t) * 0.3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PODCAST_YAML = {
    "podcast": {"name": "Test"},
    "tts": {"language": "English", "temperature": 0.7, "repetition_penalty": 1.2},
    "mix": {
        "target_sr": SR,
        "speaker_change_pause": 0.15,
        "same_speaker_pause": 0.08,
        "interjection_pause": 0.05,
        "interjection_threshold": 1.5,
        "section_pause": 1.0,
        "reverb_mix": 0.02,
        "reverb_decay": 0.15,
        "room_tone_level": 0.002,
        "peak_limit_dbtp": -1.0,
        "tail_silence": 1.5,
        "backchannel": {
            "volume_db": -3.0,
            "overlap_ms": [200, 500],
            "duck_threshold": 0.5,
            "duck_level": 0.6,
            "spill_breathing_room": 0.08,
        },
    },
    "processing": {"fade_in_ms": 35, "fade_out_ms": 20, "rms_target": 0.1},
    "cast": {
        "alex": {
            "voice_ref": "alex.mp3", "ref_text": "Hello.",
            "engine": "qwen", "fallback": [], "volume_db": 0.0,
            "backchannels": [
                {"file": "bc_alex_00.wav", "type": "laugh", "label": "Ha!"},
            ],
        },
        "morgan": {
            "voice_ref": "morgan.mp3", "ref_text": "Hello.",
            "engine": "qwen", "fallback": [], "volume_db": 0.0,
            "backchannels": [
                {"file": "bc_morgan_00.wav", "type": "breath", "label": "(intake)"},
            ],
        },
    },
    "music": {
        "intro_bed": {
            "file": "music/bed.wav",
            "music_solo": 2.0,
            "fade_in": 1.0,
            "full_vol": 0.35,
            "duck_vol": 0.12,
            "post_voice": 2.0,
            "bleed_into_cold": 3.0,
        },
        "sting": {
            "file": "music/sting.wav",
            "fade_in": 0.5,
            "crossfade": 1.0,
            "vol_under_cold": 0.35,
            "cold_open_overlap": 1.5,
        },
    },
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
COLD OPEN
====================

Alex: [earnest] First cold open line from Alex.

Morgan: [dry] A response from Morgan.

====================
CONVERSATION
====================

Alex: Here is the main conversation starting.

Morgan: Yes, and Morgan continues.
[react: alex laugh]

Alex: Short reply.

Morgan: A longer line that goes on for a while with more content.
"""

SINGLE_SECTION_SCRIPT = """\
====================
ONLY SECTION
====================

Alex: First line.

Morgan: Second line.
"""


@pytest.fixture
def mix_env(tmp_path):
    """Full environment for mix testing."""
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

    entries = parse_script(cfg.script_path())
    manifest = build_manifest(entries, script_path=str(cfg.script_path()), episode="ep_test")

    lines_dir = cfg.lines_dir()
    lines_dir.mkdir(parents=True)
    for h, info in manifest["lines"].items():
        dur = 0.8 if "short" in info["text"].lower() else 2.0
        audio = _make_audio(duration=dur)
        sf.write(str(lines_dir / info["file"]), audio, SR)
        info["status"] = STATUS_EXISTS
        info["duration"] = dur

    bc_dir = cfg.backchannels_dir()
    processed_bc = bc_dir / "processed"
    processed_bc.mkdir(parents=True)
    sf.write(str(processed_bc / "bc_alex_00.wav"), _make_audio(0.8), SR)
    sf.write(str(processed_bc / "bc_morgan_00.wav"), _make_audio(0.5), SR)

    music_dir = tmp_path / "podcasts" / "music"
    music_dir.mkdir(parents=True)
    sf.write(str(music_dir / "bed.wav"), _make_tone(10.0, 220), SR)
    sf.write(str(music_dir / "sting.wav"), _make_tone(3.0, 440), SR)

    save_manifest(manifest, cfg.work_dir() / "manifest.json")
    cfg.sections_dir().mkdir(parents=True, exist_ok=True)

    return {"cfg": cfg, "manifest": manifest, "tmp_path": tmp_path}


# ---------------------------------------------------------------------------
# Section builder
# ---------------------------------------------------------------------------


class TestBuildSection:
    def test_section_produces_audio(self, mix_env):
        cfg = mix_env["cfg"]
        manifest = mix_env["manifest"]
        rng = np.random.default_rng(42)

        sections = extract_sections(manifest)
        _, entries = sections[0]
        audio = build_section(entries, manifest["lines"], cfg.lines_dir(),
                              SR, rng, cfg.mix)
        assert len(audio) > 0
        assert audio.dtype == np.float32

    def test_speaker_change_pause(self, mix_env):
        cfg = mix_env["cfg"]
        manifest = mix_env["manifest"]
        rng = np.random.default_rng(42)

        sections = extract_sections(manifest)
        _, entries = sections[0]
        audio = build_section(entries, manifest["lines"], cfg.lines_dir(),
                              SR, rng, cfg.mix)

        line_durs = sum(
            manifest["lines"][e["hash"]]["duration"]
            for e in entries if e["type"] == "line"
        )
        total_dur = len(audio) / SR
        gap = total_dur - line_durs
        assert 0.05 < gap < 0.5

    def test_interjection_pause(self, mix_env):
        cfg = mix_env["cfg"]
        manifest = mix_env["manifest"]
        rng = np.random.default_rng(42)

        sections = extract_sections(manifest)
        _, entries = sections[1]
        audio = build_section(entries, manifest["lines"], cfg.lines_dir(),
                              SR, rng, cfg.mix)
        assert len(audio) / SR > 3.0

    def test_backchannel_placed(self, mix_env):
        cfg = mix_env["cfg"]
        manifest = mix_env["manifest"]
        rng = np.random.default_rng(42)

        bc_clips = load_bc_clips(cfg, SR)
        assert len(bc_clips) > 0

        sections = extract_sections(manifest)
        _, entries = sections[1]
        audio = build_section(entries, manifest["lines"], cfg.lines_dir(),
                              SR, rng, cfg.mix, bc_clips=bc_clips)

        line_durs = sum(
            manifest["lines"][e["hash"]]["duration"]
            for e in entries if e["type"] == "line"
        )
        assert len(audio) / SR > line_durs

    def test_missing_line_produces_silence(self, mix_env):
        cfg = mix_env["cfg"]
        manifest = mix_env["manifest"]
        rng = np.random.default_rng(42)

        first_hash = [e["hash"] for e in manifest["order"] if e["type"] == "line"][0]
        manifest["lines"][first_hash]["status"] = STATUS_MISSING

        sections = extract_sections(manifest)
        _, entries = sections[0]
        audio = build_section(entries, manifest["lines"], cfg.lines_dir(),
                              SR, rng, cfg.mix)
        assert len(audio) > 0


# ---------------------------------------------------------------------------
# Intro + music
# ---------------------------------------------------------------------------


class TestIntroWithMusic:
    def test_builds_intro(self):
        intro_voice = _make_audio(5.0, 0.2)
        music_bed = _make_tone(30.0, 220)
        music_cfg = PODCAST_YAML["music"]["intro_bed"]

        intro, bleed = build_intro_with_music(intro_voice, music_bed, SR, music_cfg)

        assert len(intro) > 0
        assert len(bleed) > 0
        expected_dur = 2.0 + 5.0 + 2.0
        assert len(intro) / SR == pytest.approx(expected_dur, abs=0.1)

    def test_music_ducked_during_voice(self):
        """Music volume drops during voiceover section."""
        intro_voice = np.ones(int(SR * 3.0), dtype=np.float32) * 0.2
        music_bed = np.ones(int(SR * 30.0), dtype=np.float32) * 1.0
        music_cfg = PODCAST_YAML["music"]["intro_bed"]

        intro, bleed = build_intro_with_music(intro_voice, music_bed, SR, music_cfg)

        # Before voice (after fade-in): music at full_vol = 0.35
        pre_voice_sample = int(SR * 1.5)  # after fade_in (1.0s), before voice_start (2.0s)
        music_pre = abs(intro[pre_voice_sample] - intro_voice[0])  # subtract voice contribution
        # During voice: music at duck_vol = 0.12
        voice_mid = int(SR * 3.5)  # middle of voice section (2.0 + 1.5)
        # The intro at voice_mid = music * duck_vol + voice
        # So music component = intro[voice_mid] - voice[voice_mid - voice_start]
        voice_start_sample = int(SR * 2.0)
        music_during = intro[voice_mid] - intro_voice[voice_mid - voice_start_sample]

        # Music during voice should be significantly quieter than before voice
        assert abs(music_during) < abs(intro[pre_voice_sample]) * 0.8

    def test_short_music_bed_warns(self, caplog):
        """Short music bed logs a warning."""
        import logging
        with caplog.at_level(logging.WARNING):
            intro_voice = _make_audio(3.0, 0.2)
            music_bed = _make_tone(2.0, 220)  # too short
            music_cfg = PODCAST_YAML["music"]["intro_bed"]
            build_intro_with_music(intro_voice, music_bed, SR, music_cfg)
        assert "shorter than needed" in caplog.text


# ---------------------------------------------------------------------------
# Sting transition
# ---------------------------------------------------------------------------


class TestStingTransition:
    def test_sting_builds(self):
        cold_open = _make_audio(10.0)
        sting = _make_tone(3.0, 440)
        sting_cfg = PODCAST_YAML["music"]["sting"]

        cold_body, sting_zone = build_sting_transition(cold_open, sting, SR, sting_cfg)

        assert len(cold_body) > 0
        assert len(sting_zone) > 0
        assert len(cold_body) + len(sting_zone) >= len(cold_open)

    def test_crossfade_into_conversation(self):
        sting_zone = _make_tone(3.0, 440)
        conversation = _make_audio(20.0)

        result = crossfade_into_conversation(sting_zone, conversation, SR, 1.0)
        assert len(result) > len(conversation) - int(SR * 1.0)


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------


class TestExtractSections:
    def test_extracts_sections(self, mix_env):
        manifest = mix_env["manifest"]
        sections = extract_sections(manifest)
        assert len(sections) == 2
        assert sections[0][0] == "COLD OPEN"
        assert sections[1][0] == "CONVERSATION"

    def test_entries_are_correct_types(self, mix_env):
        manifest = mix_env["manifest"]
        sections = extract_sections(manifest)
        for _, entries in sections:
            for e in entries:
                assert e["type"] in ("line", "pause", "backchannel")

    def test_single_section(self, mix_env):
        """Script with one section produces one section."""
        tmp_path = mix_env["tmp_path"]
        script_path = tmp_path / "podcasts" / "scripts" / "single.txt"
        script_path.write_text(SINGLE_SECTION_SCRIPT, encoding="utf-8")

        entries = parse_script(script_path)
        manifest = build_manifest(entries)
        sections = extract_sections(manifest)
        assert len(sections) == 1
        assert sections[0][0] == "ONLY SECTION"

    def test_empty_manifest(self):
        manifest = {"meta": {"version": 1}, "lines": {}, "order": []}
        sections = extract_sections(manifest)
        assert sections == []


# ---------------------------------------------------------------------------
# Pink noise
# ---------------------------------------------------------------------------


class TestPinkNoise:
    def test_deterministic(self):
        p1 = generate_pink_noise(1000, np.random.default_rng(42))
        p2 = generate_pink_noise(1000, np.random.default_rng(42))
        assert np.allclose(p1, p2)

    def test_normalized(self):
        pink = generate_pink_noise(SR, np.random.default_rng(42))
        assert np.max(np.abs(pink)) == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Full mix integration
# ---------------------------------------------------------------------------


class TestMixEpisode:
    def test_produces_output(self, mix_env):
        cfg = mix_env["cfg"]
        manifest = mix_env["manifest"]

        result = mix_episode(manifest, cfg, seed=42)

        assert Path(result["output"]).exists()
        assert result["duration"] > 0
        assert len(result["sections"]) == 2

    def test_with_intro_voice(self, mix_env):
        """Passing intro_voice produces a longer episode with intro."""
        cfg = mix_env["cfg"]
        manifest = mix_env["manifest"]

        # Without intro
        result_no_intro = mix_episode(manifest, cfg, seed=42)
        dur_no_intro = result_no_intro["duration"]

        # With intro
        intro_voice = _make_audio(3.0, 0.2)
        result_with = mix_episode(manifest, cfg, intro_voice=intro_voice, seed=42)
        dur_with = result_with["duration"]

        assert dur_with > dur_no_intro

    def test_section_files_created(self, mix_env):
        cfg = mix_env["cfg"]
        manifest = mix_env["manifest"]

        mix_episode(manifest, cfg, seed=42)

        section_files = list(cfg.sections_dir().glob("*.wav"))
        assert len(section_files) >= 2

    def test_peak_limiting(self, mix_env):
        cfg = mix_env["cfg"]
        manifest = mix_env["manifest"]

        result = mix_episode(manifest, cfg, seed=42)

        audio, sr = sf.read(result["output"])
        peak = np.max(np.abs(audio))
        limit = 10 ** (-1.0 / 20)
        assert peak <= limit + 0.001

    def test_room_tone_present(self, mix_env):
        """Room tone prevents true silence in the tail."""
        cfg = mix_env["cfg"]
        manifest = mix_env["manifest"]

        result = mix_episode(manifest, cfg, seed=42)

        audio, sr = sf.read(result["output"])
        # Check the tail silence region — should have room tone, not true silence
        tail_start = len(audio) - int(sr * 1.0)
        tail = audio[tail_start:]
        tail_rms = np.sqrt(np.mean(tail ** 2))
        assert tail_rms > 0

    def test_deterministic_output(self, mix_env):
        cfg = mix_env["cfg"]
        manifest = mix_env["manifest"]

        result1 = mix_episode(manifest, cfg, seed=42)
        audio1, _ = sf.read(result1["output"])

        result2 = mix_episode(manifest, cfg, seed=42)
        audio2, _ = sf.read(result2["output"])

        assert np.allclose(audio1, audio2)

    def test_duration_reasonable(self, mix_env):
        cfg = mix_env["cfg"]
        manifest = mix_env["manifest"]

        total_line_dur = sum(
            info["duration"] for info in manifest["lines"].values()
            if info["status"] == STATUS_EXISTS
        )

        result = mix_episode(manifest, cfg, seed=42)

        assert result["duration"] > total_line_dur * 0.8
        assert result["duration"] < total_line_dur * 3.0

    def test_config_driven_constants(self, mix_env):
        """Changing a mix constant changes the output."""
        cfg = mix_env["cfg"]
        manifest = mix_env["manifest"]

        result1 = mix_episode(manifest, cfg, seed=42)
        audio1, _ = sf.read(result1["output"])

        # Change speaker_change_pause — use a fresh copy
        cfg._data["mix"] = {**cfg._data["mix"], "speaker_change_pause": 0.5}
        result2 = mix_episode(manifest, cfg, seed=42)
        audio2, _ = sf.read(result2["output"])

        assert len(audio1) != len(audio2)

    def test_no_sting(self, mix_env):
        """Episode without sting still produces valid output."""
        cfg = mix_env["cfg"]
        manifest = mix_env["manifest"]

        # Remove sting from music config
        cfg._data["music"] = {k: v for k, v in cfg._data["music"].items() if k != "sting"}

        result = mix_episode(manifest, cfg, seed=42)
        assert Path(result["output"]).exists()
        assert result["duration"] > 0

    def test_single_section_episode(self, mix_env):
        """Episode with one section (no cold open / conversation split)."""
        cfg = mix_env["cfg"]
        tmp_path = mix_env["tmp_path"]

        script_path = tmp_path / "podcasts" / "scripts" / "single.txt"
        script_path.write_text(SINGLE_SECTION_SCRIPT, encoding="utf-8")
        cfg._data["script"] = "scripts/single.txt"

        entries = parse_script(cfg.script_path())
        manifest = build_manifest(entries)

        # Create line files
        for h, info in manifest["lines"].items():
            audio = _make_audio(2.0)
            sf.write(str(cfg.lines_dir() / info["file"]), audio, SR)
            info["status"] = STATUS_EXISTS
            info["duration"] = 2.0

        result = mix_episode(manifest, cfg, seed=42)
        assert Path(result["output"]).exists()
        assert result["duration"] > 0
        assert len(result["sections"]) == 1
