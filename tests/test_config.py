"""Tests for generator/config.py — podcast + episode config loader."""

from pathlib import Path

import pytest
import yaml

from config import (
    load_podcast_config,
    load_episode_config,
    EpisodeConfig,
    VALID_ENGINES,
    _deep_merge,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PODCAST_YAML = {
    "podcast": {"name": "Test Podcast"},
    "tts": {"language": "English", "temperature": 0.7, "repetition_penalty": 1.2},
    "mix": {
        "target_sr": 24000,
        "speaker_change_pause": 0.15,
        "same_speaker_pause": 0.08,
        "interjection_pause": 0.05,
        "interjection_threshold": 1.5,
        "beat_pause": 0.5,
        "reverb_mix": 0.02,
        "reverb_decay": 0.15,
        "room_tone_level": 0.002,
        "peak_limit_dbtp": -1.0,
        "section_pause": 1.0,
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
            "voice_ref": "alex.mp3",
            "ref_text": "Hello from Alex.",
            "engine": "qwen",
            "fallback": ["chatterbox"],
            "volume_db": 0.0,
            "backchannels": [
                {"file": "bc_alex_00.wav", "type": "laugh", "label": "Ha!"},
                {"file": "bc_alex_01.wav", "type": "breath", "label": "(intake)"},
            ],
        },
        "morgan": {
            "voice_ref": "morgan.mp3",
            "ref_text": "Hello from Morgan.",
            "engine": "qwen",
            "fallback": ["chatterbox"],
            "volume_db": 0.0,
            "backchannels": [
                {"file": "bc_morgan_00.wav", "type": "laugh", "label": "Heh."},
            ],
        },
        "zara": {
            "voice_ref": "zara.mp3",
            "ref_text": "Hello from Zara.",
            "engine": "qwen",
            "fallback": ["chatterbox"],
            "volume_db": 2.5,
            "backchannels": [
                {"file": "bc_zara_00.wav", "type": "laugh", "label": "Ha!"},
                {"file": "bc_zara_01.wav", "type": "breath", "label": "(intake)"},
            ],
        },
    },
    "music": {
        "intro_bed": {
            "file": "music/bed.mp3",
            "music_solo": 4.0,
            "fade_in": 2.0,
            "full_vol": 0.35,
            "duck_vol": 0.12,
        },
        "sting": {
            "file": "music/sting.mp3",
            "fade_in": 0.5,
            "crossfade": 2.5,
        },
    },
    "review": {"agents": ["fidelity", "voice"]},
}

EPISODE_YAML = {
    "episode": {"number": 1, "title": "The Nodding", "slug": "ep01"},
    "podcast": "test-podcast",
    "script": "scripts/ep01.txt",
    "intro_lines": "scripts/ep01_intro.txt",
    "work_dir": "work/ep01",
    "aliases": {
        "junior manager": "alex",
        "team member 1": "morgan",
        "team member 2": "zara",
    },
    "overrides": {},
    "force_fallback": [],
}


@pytest.fixture
def podcast_dir(tmp_path):
    """Create podcast config directory."""
    d = tmp_path / "podcasts" / "test-podcast"
    d.mkdir(parents=True)
    with open(d / "podcast.yaml", "w") as f:
        yaml.dump(PODCAST_YAML, f)
    return d


@pytest.fixture
def episode_file(tmp_path, podcast_dir):
    """Create episode config file."""
    episodes_dir = tmp_path / "podcasts" / "episodes"
    episodes_dir.mkdir(parents=True)
    path = episodes_dir / "ep01.yaml"
    with open(path, "w") as f:
        yaml.dump(EPISODE_YAML, f)
    # Create the script file so path resolution works
    scripts_dir = tmp_path / "podcasts" / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "ep01.txt").write_text("Alex: Hello.\n", encoding="utf-8")
    (scripts_dir / "ep01_intro.txt").write_text("Alex: Intro.\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        assert _deep_merge(base, override) == {"a": 1, "b": 3}

    def test_nested_merge(self):
        base = {"mix": {"pause": 0.15, "reverb": 0.02}}
        override = {"mix": {"pause": 0.10}}
        result = _deep_merge(base, override)
        assert result["mix"]["pause"] == 0.10
        assert result["mix"]["reverb"] == 0.02

    def test_new_keys_added(self):
        result = _deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_base_unchanged(self):
        base = {"a": 1}
        _deep_merge(base, {"a": 2})
        assert base["a"] == 1

    def test_list_replacement_not_merge(self):
        """Episode list replaces podcast list entirely (no appending)."""
        base = {"fallback": ["chatterbox", "elevenlabs"]}
        override = {"fallback": ["elevenlabs"]}
        result = _deep_merge(base, override)
        assert result["fallback"] == ["elevenlabs"]


# ---------------------------------------------------------------------------
# Podcast config loading
# ---------------------------------------------------------------------------


class TestLoadPodcastConfig:
    def test_load_valid(self, podcast_dir):
        data = load_podcast_config(podcast_dir / "podcast.yaml")
        assert data["podcast"]["name"] == "Test Podcast"
        assert data["tts"]["temperature"] == 0.7
        assert "alex" in data["cast"]

    def test_missing_required_key(self, tmp_path):
        path = tmp_path / "bad.yaml"
        with open(path, "w") as f:
            yaml.dump({"podcast": {"name": "Bad"}, "tts": {}}, f)
        with pytest.raises(ValueError, match="missing required keys"):
            load_podcast_config(path)

    def test_unknown_engine(self, tmp_path):
        data = dict(PODCAST_YAML)
        data["cast"] = {"alex": {"engine": "unknown_engine", "voice_ref": "x.mp3"}}
        path = tmp_path / "bad_engine.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        with pytest.raises(ValueError, match="unknown engine"):
            load_podcast_config(path)

    def test_unknown_fallback_engine(self, tmp_path):
        data = dict(PODCAST_YAML)
        data["cast"] = {"alex": {"engine": "qwen", "fallback": ["badengine"], "voice_ref": "x.mp3"}}
        path = tmp_path / "bad_fb.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        with pytest.raises(ValueError, match="unknown fallback engine"):
            load_podcast_config(path)


# ---------------------------------------------------------------------------
# Episode config loading + merge
# ---------------------------------------------------------------------------


class TestLoadEpisodeConfig:
    def test_load_and_merge(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        assert cfg.episode["title"] == "The Nodding"
        assert cfg.episode["number"] == 1
        assert cfg.tts["temperature"] == 0.7
        assert cfg.mix["speaker_change_pause"] == 0.15

    def test_episode_overrides_podcast(self, tmp_path, podcast_dir):
        ep_data = dict(EPISODE_YAML)
        ep_data["tts"] = {"temperature": 0.5}
        episodes_dir = tmp_path / "podcasts" / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        path = episodes_dir / "ep_override.yaml"
        with open(path, "w") as f:
            yaml.dump(ep_data, f)
        cfg = load_episode_config(path, podcast_dir=podcast_dir)
        assert cfg.tts["temperature"] == 0.5
        assert cfg.tts["repetition_penalty"] == 1.2

    def test_missing_required_episode_key(self, tmp_path, podcast_dir):
        path = tmp_path / "bad_ep.yaml"
        with open(path, "w") as f:
            yaml.dump({"episode": {"title": "Bad"}}, f)
        with pytest.raises(ValueError, match="missing required keys"):
            load_episode_config(path, podcast_dir=podcast_dir)

    def test_auto_discovery_without_podcast_dir(self, tmp_path):
        """load_episode_config without podcast_dir uses convention."""
        # Create podcasts/test-podcast/podcast.yaml
        pd = tmp_path / "podcasts" / "test-podcast"
        pd.mkdir(parents=True)
        with open(pd / "podcast.yaml", "w") as f:
            yaml.dump(PODCAST_YAML, f)
        # Create podcasts/episodes/ep01.yaml
        ed = tmp_path / "podcasts" / "episodes"
        ed.mkdir(parents=True)
        ep_path = ed / "ep01.yaml"
        with open(ep_path, "w") as f:
            yaml.dump(EPISODE_YAML, f)
        # Load without podcast_dir
        cfg = load_episode_config(ep_path)
        assert cfg.episode["title"] == "The Nodding"
        assert cfg.tts["temperature"] == 0.7

    def test_force_fallback_invalid_engine(self, tmp_path, podcast_dir):
        """force_fallback with invalid engine raises ValueError."""
        ep_data = dict(EPISODE_YAML)
        ep_data["force_fallback"] = [{"hash": "abc", "engine": "badengine", "reason": "test"}]
        episodes_dir = tmp_path / "podcasts" / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        path = episodes_dir / "ep_bad_fb.yaml"
        with open(path, "w") as f:
            yaml.dump(ep_data, f)
        with pytest.raises(ValueError, match="force_fallback.*unknown engine"):
            load_episode_config(path, podcast_dir=podcast_dir)


# ---------------------------------------------------------------------------
# Cast lookup
# ---------------------------------------------------------------------------


class TestCastLookup:
    def test_lookup_by_name(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        alex = cfg.cast("alex")
        assert alex["voice_ref"] == "alex.mp3"
        assert alex["engine"] == "qwen"

    def test_lookup_by_alias(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        assert cfg.cast("junior manager")["voice_ref"] == "alex.mp3"

    def test_lookup_zara_by_alias(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        assert cfg.cast("team member 2")["voice_ref"] == "zara.mp3"

    def test_lookup_case_insensitive(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        assert cfg.cast("Alex")["voice_ref"] == "alex.mp3"
        assert cfg.cast("MORGAN")["voice_ref"] == "morgan.mp3"

    def test_unknown_speaker_raises(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        with pytest.raises(KeyError, match="Unknown cast member"):
            cfg.cast("felix")

    def test_cast_names(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        names = cfg.cast_names()
        assert "alex" in names
        assert "morgan" in names
        assert "zara" in names

    def test_all_aliases(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        aliases = cfg.all_aliases()
        assert aliases["junior manager"] == "alex"
        assert aliases["team member 1"] == "morgan"
        assert aliases["team member 2"] == "zara"


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


class TestPathResolution:
    def test_script_path(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        path = cfg.script_path()
        assert path.name == "ep01.txt"
        assert path.parent.name == "scripts"

    def test_intro_lines_path(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        path = cfg.intro_lines_path()
        assert path.name == "ep01_intro.txt"

    def test_work_dir(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        wd = cfg.work_dir()
        assert wd.name == "ep01"
        assert wd.parent.name == "work"

    def test_tts_dir(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        assert cfg.tts_dir().name == "tts"
        assert cfg.tts_dir().parent == cfg.work_dir()

    def test_lines_dir(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        assert cfg.lines_dir().name == "lines"
        assert cfg.lines_dir().parent == cfg.work_dir()

    def test_sections_dir(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        assert cfg.sections_dir().name == "sections"

    def test_backchannels_dir(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        assert cfg.backchannels_dir().name == "backchannels"

    def test_resolve_path(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        p = cfg.resolve_path("some/file.txt")
        assert str(p).endswith("some/file.txt") or str(p).endswith("some\\file.txt")

    def test_voice_refs_dir(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        vrd = cfg.voice_refs_dir()
        assert vrd.name == "voice_refs"

    def test_voice_ref_path(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        path = cfg.voice_ref_path("alex")
        assert path.name == "alex.mp3"
        assert "voice_refs" in str(path)


# ---------------------------------------------------------------------------
# Music assets
# ---------------------------------------------------------------------------


class TestMusicAssets:
    def test_music_asset_lookup(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        sting = cfg.music_asset("sting")
        assert sting is not None
        assert "sting.mp3" in sting["file"]
        assert sting["crossfade"] == 2.5

    def test_intro_bed_lookup(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        bed = cfg.music_asset("intro_bed")
        assert bed is not None
        assert bed["duck_vol"] == 0.12
        assert bed["music_solo"] == 4.0
        assert bed["fade_in"] == 2.0

    def test_missing_music_asset(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        assert cfg.music_asset("outro") is None


# ---------------------------------------------------------------------------
# Backchannel clips
# ---------------------------------------------------------------------------


class TestBackchannelClips:
    def test_clips_for_speaker(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        clips = cfg.backchannel_clips("alex")
        assert len(clips) == 2
        assert clips[0]["type"] == "laugh"
        assert clips[0]["label"] == "Ha!"

    def test_clips_paths_resolved(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        clips = cfg.backchannel_clips("alex")
        # File paths should be absolute, pointing into backchannels dir
        assert "backchannels" in clips[0]["file"]
        assert clips[0]["file"].endswith("bc_alex_00.wav")

    def test_clips_by_alias(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        clips = cfg.backchannel_clips("junior manager")
        assert len(clips) == 2  # same as alex

    def test_clips_for_zara(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        clips = cfg.backchannel_clips("zara")
        assert len(clips) == 2
        assert clips[0]["type"] == "laugh"

    def test_clips_for_unknown_speaker(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        clips = cfg.backchannel_clips("felix")
        assert clips == []

    def test_clips_have_required_fields(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        for name in cfg.cast_names():
            for clip in cfg.backchannel_clips(name):
                assert "file" in clip
                assert "type" in clip
                assert "label" in clip


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_processing(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        assert cfg.processing["fade_in_ms"] == 35
        assert cfg.processing["rms_target"] == 0.1

    def test_review(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        assert "fidelity" in cfg.review["agents"]

    def test_overrides(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        assert cfg.overrides == {}

    def test_force_fallback(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        assert cfg.force_fallback == []

    def test_backchannel_mix_params(self, episode_file, podcast_dir):
        cfg = load_episode_config(episode_file, podcast_dir=podcast_dir)
        bc = cfg.mix["backchannel"]
        assert bc["volume_db"] == -3.0
        assert bc["overlap_ms"] == [200, 500]
        assert bc["duck_threshold"] == 0.5
        assert bc["duck_level"] == 0.6
        assert bc["spill_breathing_room"] == 0.08
