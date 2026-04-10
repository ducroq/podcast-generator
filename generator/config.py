"""Episode configuration loader for podcast pipeline.

Loads a two-level config: podcast-level defaults + episode-level overrides.
Episode values override podcast values (scalars and lists replace; dicts
merge recursively). Provides cast lookup by name or alias, path resolution,
and validation.

Usage:
    from config import load_episode_config

    cfg = load_episode_config("podcasts/episodes/ep01.yaml",
                              podcast_dir="podcasts/it-is-both")
    voice = cfg.cast("morgan")
    voice = cfg.cast("team member 1")  # alias lookup
"""

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

REQUIRED_PODCAST_KEYS = {"podcast", "tts", "mix", "cast", "music"}
REQUIRED_EPISODE_KEYS = {"episode", "podcast", "script"}
VALID_ENGINES = {"qwen", "chatterbox", "elevenlabs"}  # elevenlabs uses separate API workflow


def _deep_merge(base, override):
    """Recursively merge override dict into base dict.

    - Dict values: merged recursively (episode can override individual keys).
    - Scalar and list values: replaced entirely (episode list replaces
      podcast list, no appending).
    - Base dict is not mutated.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class EpisodeConfig:
    """Merged podcast + episode configuration with lookup helpers.

    Speaker names are lowercased everywhere — both manifest.py (from script
    parsing) and this module use the same convention. This is a shared
    contract between the two modules.
    """

    def __init__(self, data, base_dir):
        self._data = data
        self._base_dir = Path(base_dir)
        self._alias_map = {}
        self._build_alias_map()

    def _build_alias_map(self):
        """Build alias -> speaker mapping from episode aliases."""
        for alias, speaker in self._data.get("aliases", {}).items():
            self._alias_map[alias.lower()] = speaker.lower()

    @property
    def episode(self):
        return self._data.get("episode", {})

    @property
    def tts(self):
        return self._data.get("tts", {})

    @property
    def mix(self):
        return self._data.get("mix", {})

    @property
    def music(self):
        return self._data.get("music", {})

    @property
    def processing(self):
        return self._data.get("processing", {})

    @property
    def review(self):
        return self._data.get("review", {})

    @property
    def overrides(self):
        """Hash-keyed TTS overrides for specific lines.

        Expected schema per entry:
            hash_key:
              segments:           # segmented generation (for pacing)
                - text: "First part."
                  pause_after: 0.3
                - text: "Second part."
              engine: chatterbox  # optional: override engine for this line
              temperature: 0.5   # optional: override temperature
        """
        return self._data.get("overrides", {})

    @property
    def force_fallback(self):
        """Lines that must use a fallback engine.

        Expected schema: list of dicts, each with:
            - hash: content hash of the line
            - reason: why the primary engine failed
            - engine: fallback engine to use (must be in VALID_ENGINES)
        """
        return self._data.get("force_fallback", [])

    # ----- Cast -----

    def cast(self, name):
        """Look up a cast member by name or alias (case-insensitive).

        Returns a dict with: voice_ref, ref_text, engine, fallback,
        volume_db, backchannels. Raises KeyError if not found.
        """
        name_lower = name.lower()
        resolved = self._alias_map.get(name_lower, name_lower)
        cast_data = self._data.get("cast", {})
        if resolved not in cast_data:
            raise KeyError(
                f"Unknown cast member '{name}' (resolved to '{resolved}'). "
                f"Available: {list(cast_data.keys())}"
            )
        return cast_data[resolved]

    def cast_names(self):
        """Return list of primary cast member names (not aliases)."""
        return list(self._data.get("cast", {}).keys())

    def all_aliases(self):
        """Return alias -> primary name mapping."""
        return dict(self._alias_map)

    # ----- Paths -----

    def resolve_path(self, relative_path):
        """Resolve a path relative to the config base directory (podcasts/)."""
        return self._base_dir / relative_path

    def script_path(self):
        """Resolved path to the episode script."""
        return self.resolve_path(self._data["script"])

    def intro_lines_path(self):
        """Resolved path to intro lines file, or None."""
        p = self._data.get("intro_lines")
        return self.resolve_path(p) if p else None

    def work_dir(self):
        """Resolved path to the episode work directory."""
        return self.resolve_path(
            self._data.get("work_dir", f"work/{self.episode.get('slug', 'default')}")
        )

    def tts_dir(self):
        """Work subdirectory for raw TTS output (never modified after generation)."""
        return self.work_dir() / "tts"

    def lines_dir(self):
        """Work subdirectory for processed lines (trimmed, faded, normalized)."""
        return self.work_dir() / "lines"

    def sections_dir(self):
        """Work subdirectory for per-section mixes."""
        return self.work_dir() / "sections"

    def backchannels_dir(self):
        """Work subdirectory for raw backchannel clips."""
        return self.work_dir() / "backchannels"

    def processed_backchannels_dir(self):
        """Work subdirectory for processed backchannel clips (faded, click-suppressed).

        Step 5 (mix) should read BC clips from here, not backchannels_dir().
        """
        return self.backchannels_dir() / "processed"

    # ----- Music -----

    def music_asset(self, name):
        """Get a music asset config by name (e.g. 'intro_bed', 'sting').

        Returns the asset dict with 'file' resolved to absolute path, or None.
        """
        asset = self.music.get(name)
        if asset and "file" in asset:
            resolved = dict(asset)
            resolved["file"] = str(self.resolve_path(asset["file"]))
            return resolved
        return asset

    # ----- Backchannels -----

    def backchannel_clips(self, speaker):
        """Get backchannel clip list for a speaker (resolved paths).

        Returns list of {file, type, label} dicts with file paths resolved
        relative to the backchannels work directory. Returns empty list for
        unknown speakers (graceful — mixer should handle no-clip situations).
        """
        speaker_lower = speaker.lower()
        resolved = self._alias_map.get(speaker_lower, speaker_lower)
        cast_data = self._data.get("cast", {}).get(resolved, {})
        clips = cast_data.get("backchannels", [])
        # Resolve file paths relative to backchannels dir
        bc_dir = self.backchannels_dir()
        return [
            {**clip, "file": str(bc_dir / clip["file"])}
            for clip in clips
        ]

    # ----- Voice refs -----

    def voice_refs_dir(self):
        """Directory where voice reference files are stored.

        Defaults to 'voice_refs' under the config base dir. Can be
        overridden in podcast.yaml via the voice_refs_dir key.

        Note: on the gpu-server this is typically ~/voice_refs/.
        The caller (generate_tts.py) is responsible for mapping this
        to the correct server path when running remotely.
        """
        return self.resolve_path(
            self._data.get("voice_refs_dir", "voice_refs")
        )

    def voice_ref_path(self, speaker):
        """Resolved path to a speaker's voice reference file."""
        info = self.cast(speaker)
        return self.voice_refs_dir() / info["voice_ref"]


# ---------------------------------------------------------------------------
# Loading & validation
# ---------------------------------------------------------------------------


def _load_yaml(path):
    """Load a YAML file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _validate_podcast(data, path):
    """Validate podcast-level config has required keys."""
    missing = REQUIRED_PODCAST_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Podcast config {path} missing required keys: {missing}")

    # Validate engines in cast
    for name, info in data.get("cast", {}).items():
        engine = info.get("engine", "")
        if engine and engine not in VALID_ENGINES:
            raise ValueError(
                f"Cast member '{name}' has unknown engine '{engine}'. "
                f"Valid: {VALID_ENGINES}"
            )
        for fb in info.get("fallback", []):
            if fb not in VALID_ENGINES:
                raise ValueError(
                    f"Cast member '{name}' has unknown fallback engine '{fb}'. "
                    f"Valid: {VALID_ENGINES}"
                )


def _validate_episode(data, path):
    """Validate episode-level config has required keys."""
    missing = REQUIRED_EPISODE_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Episode config {path} missing required keys: {missing}")

    # Validate force_fallback engine values
    for entry in data.get("force_fallback", []):
        engine = entry.get("engine", "")
        if engine and engine not in VALID_ENGINES:
            raise ValueError(
                f"force_fallback entry has unknown engine '{engine}'. "
                f"Valid: {VALID_ENGINES}"
            )


def load_podcast_config(path):
    """Load and validate a podcast-level config.

    Returns the raw dict (not an EpisodeConfig — use load_episode_config
    for the merged result).
    """
    data = _load_yaml(path)
    _validate_podcast(data, path)
    return data


def load_episode_config(episode_path, podcast_dir=None):
    """Load episode config, merge with podcast defaults, return EpisodeConfig.

    If podcast_dir is not specified, it's derived from the episode config's
    'podcast' field using the convention:
        podcasts/<podcast_name>/podcast.yaml
    where 'podcasts/' is the grandparent of the episode file.

    Merge order: podcast defaults <- episode overrides.
    """
    episode_path = Path(episode_path)
    ep_data = _load_yaml(episode_path)
    _validate_episode(ep_data, episode_path)

    # Find podcast config
    podcast_name = ep_data.get("podcast", "")
    if podcast_dir:
        podcast_path = Path(podcast_dir) / "podcast.yaml"
    else:
        # Convention: podcasts/<podcast_name>/podcast.yaml
        # relative to the episode file's grandparent (podcasts/)
        podcasts_root = episode_path.parent.parent
        podcast_path = podcasts_root / podcast_name / "podcast.yaml"

    podcast_data = _load_yaml(podcast_path)
    _validate_podcast(podcast_data, podcast_path)

    # Merge: podcast defaults <- episode overrides
    merged = _deep_merge(podcast_data, ep_data)

    # Base dir for path resolution is the podcasts/ root
    base_dir = episode_path.parent.parent

    return EpisodeConfig(merged, base_dir)
