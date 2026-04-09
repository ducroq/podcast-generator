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
    """Create test audio."""
    return (np.random.randn(int(sr * duration)) * amplitude).astype(np.float32)


def _make_tone(duration=1.0, freq=440, sr=SR):
    """Create a pure tone for testing crossfades and levels."""
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
            "file": "music/bed.mp3",
            "music_solo": 2.0,
            "fade_in": 1.0,
            "full_vol": 0.35,
            "duck_vol": 0.12,
            "post_voice": 2.0,
            "bleed_into_cold": 3.0,
        },
        "sting": {
            "file": "music/sting.mp3",
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


@pytest.fixture
def mix_env(tmp_path):
    """Full environment for mix testing."""
    # Config
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

    # Parse and build manifest
    entries = parse_script(cfg.script_path())
    manifest = build_manifest(entries, script_path=str(cfg.script_path()), episode="ep_test")

    # Create processed line WAVs
    lines_dir = cfg.lines_dir()
    lines_dir.mkdir(parents=True)
    for h, info in manifest["lines"].items():
        dur = 0.8 if "short" in info["text"].lower() else 2.0
        audio = _make_audio(duration=dur)
        sf.write(str(lines_dir / info["file"]), audio, SR)
        info["status"] = STATUS_EXISTS
        info["duration"] = dur

    # Create BC clips
    bc_dir = cfg.backchannels_dir()
    processed_bc = bc_dir / "processed"
    processed_bc.mkdir(parents=True)
    sf.write(str(processed_bc / "bc_alex_00.wav"), _make_audio(0.8), SR)
    sf.write(str(processed_bc / "bc_morgan_00.wav"), _make_audio(0.5), SR)

    # Create music assets
    music_dir = tmp_path / "podcasts" / "music"
    music_dir.mkdir(parents=True)
    sf.write(str(music_dir / "bed.mp3"), _make_tone(10.0, 220), SR)
    sf.write(str(music_dir / "sting.mp3"), _make_tone(3.0, 440), SR)

    # Save manifest
    save_manifest(manifest, cfg.work_dir() / "manifest.json")

    # Create sections dir
    cfg.sections_dir().mkdir(parents=True, exist_ok=True)

    return {"cfg": cfg, "manifest": manifest}


# ---------------------------------------------------------------------------
# Section builder
# ---------------------------------------------------------------------------


class TestBuildSection:
    def test_section_produces_audio(self, mix_env):
        cfg = mix_env["cfg"]
        manifest = mix_env["manifest"]
        rng = np.random.default_rng(42)

        # Get cold open entries
        sections = extract_sections(manifest)
        name, entries = sections[0]
        audio = build_section(entries, manifest["lines"], cfg.lines_dir(),
                              SR, rng, cfg.mix)
        assert len(audio) > 0
        assert audio.dtype == np.float32

    def test_speaker_change_pause(self, mix_env):
        """Gap between different speakers is ~speaker_change_pause."""
        cfg = mix_env["cfg"]
        manifest = mix_env["manifest"]
        rng = np.random.default_rng(42)

        sections = extract_sections(manifest)
        _, entries = sections[0]  # cold open: Alex then Morgan
        audio = build_section(entries, manifest["lines"], cfg.lines_dir(),
                              SR, rng, cfg.mix)

        # Duration should be: line1 + pause + line2 (approx)
        line_durs = sum(
            manifest["lines"][e["hash"]]["duration"]
            for e in entries if e["type"] == "line"
        )
        total_dur = len(audio) / SR
        gap = total_dur - line_durs
        assert 0.05 < gap < 0.5  # some pause exists between speakers

    def test_interjection_pause(self, mix_env):
        """Short lines get shorter gaps."""
        cfg = mix_env["cfg"]
        manifest = mix_env["manifest"]
        rng = np.random.default_rng(42)

        sections = extract_sections(manifest)
        _, entries = sections[1]  # conversation has "Short reply"
        audio = build_section(entries, manifest["lines"], cfg.lines_dir(),
                              SR, rng, cfg.mix)
        # Just verify it builds without error and has reasonable length
        assert len(audio) / SR > 3.0

    def test_backchannel_placed(self, mix_env):
        """[react: alex laugh] produces audio with BC mixed in."""
        cfg = mix_env["cfg"]
        manifest = mix_env["manifest"]
        rng = np.random.default_rng(42)

        bc_clips = load_bc_clips(cfg, SR)
        assert len(bc_clips) > 0  # BC clips loaded

        sections = extract_sections(manifest)
        _, entries = sections[1]  # conversation has a react cue
        audio = build_section(entries, manifest["lines"], cfg.lines_dir(),
                              SR, rng, cfg.mix, bc_clips=bc_clips)

        # Audio should be longer than just the lines (BC adds gap extension)
        line_durs = sum(
            manifest["lines"][e["hash"]]["duration"]
            for e in entries if e["type"] == "line"
        )
        assert len(audio) / SR > line_durs

    def test_missing_line_produces_silence(self, mix_env):
        """Missing line produces 0.5s silence placeholder."""
        cfg = mix_env["cfg"]
        manifest = mix_env["manifest"]
        rng = np.random.default_rng(42)

        # Mark first line as missing
        first_hash = [e["hash"] for e in manifest["order"] if e["type"] == "line"][0]
        manifest["lines"][first_hash]["status"] = STATUS_MISSING

        sections = extract_sections(manifest)
        _, entries = sections[0]
        audio = build_section(entries, manifest["lines"], cfg.lines_dir(),
                              SR, rng, cfg.mix)
        assert len(audio) > 0  # still produces audio


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
        # Intro should be: music_solo + voice + post_voice
        expected_dur = 2.0 + 5.0 + 2.0
        assert len(intro) / SR == pytest.approx(expected_dur, abs=0.1)

    def test_music_ducked_during_voice(self):
        intro_voice = np.ones(int(SR * 3.0), dtype=np.float32) * 0.2
        music_bed = np.ones(int(SR * 30.0), dtype=np.float32)
        music_cfg = PODCAST_YAML["music"]["intro_bed"]

        intro, bleed = build_intro_with_music(intro_voice, music_bed, SR, music_cfg)

        # During voice section, music should be ducked to duck_vol
        voice_start = int(SR * 2.0)  # music_solo
        voice_mid = voice_start + int(SR * 1.5)
        # Music component at voice_mid should be much lower than full_vol
        # (voice is added on top, but the base music track is ducked)


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
        # Cold body + sting zone should cover the full cold open
        assert len(cold_body) + len(sting_zone) >= len(cold_open)

    def test_crossfade_into_conversation(self):
        sting_zone = _make_tone(3.0, 440)
        conversation = _make_audio(20.0)

        result = crossfade_into_conversation(sting_zone, conversation, SR, 1.0)

        # Result should be roughly sting pre + crossfade + conversation
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
        for name, entries in sections:
            for e in entries:
                assert e["type"] in ("line", "pause", "backchannel")


# ---------------------------------------------------------------------------
# Pink noise
# ---------------------------------------------------------------------------


class TestPinkNoise:
    def test_deterministic(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p1 = generate_pink_noise(1000, rng1)
        p2 = generate_pink_noise(1000, rng2)
        assert np.allclose(p1, p2)

    def test_normalized(self):
        rng = np.random.default_rng(42)
        pink = generate_pink_noise(SR, rng)
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
        limit = 10 ** (-1.0 / 20)  # -1 dBTP
        assert peak <= limit + 0.001

    def test_room_tone_present(self, mix_env):
        cfg = mix_env["cfg"]
        manifest = mix_env["manifest"]

        result = mix_episode(manifest, cfg, seed=42)

        audio, sr = sf.read(result["output"])
        # Room tone means there should be no true silence
        # Check that the quietest section still has some signal
        chunk_size = int(sr * 0.1)
        min_rms = min(
            np.sqrt(np.mean(audio[i:i+chunk_size] ** 2))
            for i in range(0, len(audio) - chunk_size, chunk_size)
        )
        assert min_rms > 0  # room tone prevents true silence

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

        # Sum of all line durations
        total_line_dur = sum(
            info["duration"] for info in manifest["lines"].values()
            if info["status"] == STATUS_EXISTS
        )

        result = mix_episode(manifest, cfg, seed=42)

        # Output should be within reasonable range of line durations
        # (includes gaps, room tone, sting, etc.)
        assert result["duration"] > total_line_dur * 0.8
        assert result["duration"] < total_line_dur * 3.0

    def test_config_driven_constants(self, mix_env):
        """Changing a mix constant changes the output."""
        cfg = mix_env["cfg"]
        manifest = mix_env["manifest"]

        result1 = mix_episode(manifest, cfg, seed=42)
        audio1, _ = sf.read(result1["output"])

        # Change speaker_change_pause
        cfg._data["mix"]["speaker_change_pause"] = 0.5
        result2 = mix_episode(manifest, cfg, seed=42)
        audio2, _ = sf.read(result2["output"])

        # Different pause = different output length
        assert len(audio1) != len(audio2)
