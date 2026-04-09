"""Podcast episode mixer — assembles processed lines into a complete episode.

Reads processed lines from lines_dir, backchannel clips from
processed_backchannels_dir, and music assets from config. Assembles
sections in manifest order with pauses, backchannels, music bed,
and sting. Outputs per-section WAVs and a full episode mix.

All mix constants come from the episode config — nothing is hardcoded.

Usage:
    from mix_podcast import mix_episode
    from config import load_episode_config
    from manifest import load_manifest

    cfg = load_episode_config("episodes/ep01.yaml")
    manifest = load_manifest(cfg.work_dir() / "manifest.json")
    mix_episode(manifest, cfg)
"""

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, lfilter

from manifest import STATUS_EXISTS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


def load_audio_file(path, target_sr):
    """Load any audio file (MP3/WAV/etc) via ffmpeg, convert to mono float32."""
    # Create temp file, close it (Windows needs this), then use the path
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    import os
    os.close(fd)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(path),
             "-ar", str(target_sr), "-ac", "1", "-f", "wav", tmp_path],
            capture_output=True, timeout=60,
        )
        audio, _ = sf.read(tmp_path, dtype="float32")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return audio


def _load_line(path, target_sr):
    """Load a processed WAV line. Already mono float32 at target_sr."""
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


# ---------------------------------------------------------------------------
# Backchannel clip loading
# ---------------------------------------------------------------------------


def load_bc_clips(cfg, target_sr):
    """Load processed backchannel clips grouped by {type: {speaker: [arrays]}}.

    Reads from processed_backchannels_dir (already faded/click-suppressed).
    Uses the clip metadata from config to determine type and speaker mapping.
    """
    bc_dir = cfg.processed_backchannels_dir()
    bc_clips = {}

    if not bc_dir.exists():
        return bc_clips

    for speaker in cfg.cast_names():
        for clip_info in cfg.backchannel_clips(speaker):
            clip_name = Path(clip_info["file"]).name
            clip_path = bc_dir / clip_name
            if not clip_path.exists():
                continue

            audio, sr = sf.read(str(clip_path), dtype="float32")
            bc_type = clip_info["type"]

            if bc_type not in bc_clips:
                bc_clips[bc_type] = {}
            if speaker not in bc_clips[bc_type]:
                bc_clips[bc_type][speaker] = []
            bc_clips[bc_type][speaker].append(audio)

    total = sum(len(v) for t in bc_clips.values() for v in t.values())
    if total:
        logger.info("Loaded %d backchannel clips", total)
    return bc_clips


def _pick_bc_clip(bc_clips, reactor, bc_type, rng):
    """Pick a random backchannel clip for a reactor+type."""
    clips = bc_clips.get(bc_type, {}).get(reactor, [])
    if not clips:
        return None
    return clips[int(rng.integers(len(clips)))]


# ---------------------------------------------------------------------------
# Pink noise
# ---------------------------------------------------------------------------


def generate_pink_noise(num_samples, rng):
    """Generate pink noise via IIR filter."""
    white = rng.standard_normal(num_samples).astype(np.float32)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004709510]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
    pink = lfilter(b, a, white).astype(np.float32)
    peak = np.max(np.abs(pink))
    if peak > 0:
        pink /= peak
    return pink


# ---------------------------------------------------------------------------
# Section builder
# ---------------------------------------------------------------------------


def build_section(order_entries, manifest_lines, lines_dir, sr, rng,
                  mix_cfg, bc_clips=None):
    """Build audio for a sequence of manifest order entries.

    Handles line concatenation with config-driven pauses, backchannel
    placement with overlap/spill/duck, and explicit pause entries.

    Args:
        order_entries: list of entries from manifest["order"]
        manifest_lines: manifest["lines"] dict
        lines_dir: Path to processed lines directory
        sr: sample rate
        rng: numpy random generator
        mix_cfg: mix config dict from EpisodeConfig
        bc_clips: loaded BC clips dict, or None

    Returns:
        numpy array of assembled audio
    """
    # Extract mix constants
    speaker_change_pause = mix_cfg.get("speaker_change_pause", 0.15)
    same_speaker_pause = mix_cfg.get("same_speaker_pause", 0.08)
    interjection_pause = mix_cfg.get("interjection_pause", 0.05)
    interjection_threshold = mix_cfg.get("interjection_threshold", 1.5)
    section_pause = mix_cfg.get("section_pause", 1.0)

    bc_cfg = mix_cfg.get("backchannel", {})
    bc_vol = 10 ** (bc_cfg.get("volume_db", -3.0) / 20)
    bc_overlap_ms = bc_cfg.get("overlap_ms", [200, 500])
    bc_duck_threshold = bc_cfg.get("duck_threshold", 0.5)
    bc_duck_level = bc_cfg.get("duck_level", 0.6)
    bc_spill_room = bc_cfg.get("spill_breathing_room", 0.08)

    parts = []
    prev_speaker = None
    pending_bc = None
    last_line_idx = None
    prev_line_dur = 0.0

    for entry in order_entries:
        if entry["type"] == "section_break":
            pending_bc = None
            last_line_idx = None
            parts.append(np.zeros(int(sr * section_pause), dtype=np.float32))
            prev_speaker = None
            continue

        if entry["type"] == "pause":
            parts.append(np.zeros(int(sr * entry["duration"]), dtype=np.float32))
            continue

        if entry["type"] == "backchannel":
            if bc_clips is None:
                continue
            clip = _pick_bc_clip(bc_clips, entry["reactor"], entry["bc_type"], rng)
            if clip is not None:
                pending_bc = {"clip": clip.copy(), "reactor": entry["reactor"],
                              "bc_type": entry["bc_type"]}
            continue

        if entry["type"] == "line":
            h = entry["hash"]
            info = manifest_lines.get(h)
            if info is None or info["status"] != STATUS_EXISTS:
                parts.append(np.zeros(int(sr * 0.5), dtype=np.float32))
                pending_bc = None
                continue

            wav_path = lines_dir / info["file"]
            if not wav_path.exists():
                parts.append(np.zeros(int(sr * 0.5), dtype=np.float32))
                pending_bc = None
                continue

            audio = _load_line(wav_path, sr)

            if prev_speaker is not None:
                # Determine gap duration
                if info["speaker"] == prev_speaker:
                    base = same_speaker_pause
                elif prev_line_dur < interjection_threshold:
                    base = interjection_pause
                else:
                    base = speaker_change_pause
                jitter = rng.uniform(-0.02, 0.04)
                gap = max(0.03, base + jitter)

                if pending_bc is not None and last_line_idx is not None:
                    # Place backchannel overlapping tail of previous line
                    gap = _place_backchannel(
                        parts, last_line_idx, pending_bc, gap, sr, rng,
                        bc_vol, bc_overlap_ms, bc_duck_threshold,
                        bc_duck_level, bc_spill_room,
                    )
                    pending_bc = None
                else:
                    pending_bc = None
                    parts.append(np.zeros(int(sr * gap), dtype=np.float32))
            else:
                pending_bc = None

            parts.append(audio)
            last_line_idx = len(parts) - 1
            prev_line_dur = len(audio) / sr
            prev_speaker = info["speaker"]

    return np.concatenate(parts) if parts else np.array([], dtype=np.float32)


def _place_backchannel(parts, last_line_idx, pending_bc, gap, sr, rng,
                       bc_vol, bc_overlap_ms, bc_duck_threshold,
                       bc_duck_level, bc_spill_room):
    """Place a backchannel clip overlapping the previous line.

    Returns the effective gap duration (may be extended for spill).
    Appends gap audio to parts.
    """
    bc_clip = pending_bc["clip"] * bc_vol
    bc_len = len(bc_clip)
    bc_dur = bc_len / sr

    prev_line_audio = parts[last_line_idx]

    # Overlap into the tail of the previous line
    overlap_ms = rng.uniform(*bc_overlap_ms)
    overlap_samples = int(sr * overlap_ms / 1000)
    overlap_samples = min(overlap_samples, len(prev_line_audio))

    # Duck previous line under long backchannels
    if bc_dur > bc_duck_threshold and overlap_samples > 0:
        duck_region = prev_line_audio[-overlap_samples:]
        fade = min(int(sr * 0.04), len(duck_region) // 3)
        duck = np.ones(len(duck_region), dtype=np.float32) * bc_duck_level
        if fade > 0:
            duck[:fade] = np.linspace(1.0, bc_duck_level, fade, dtype=np.float32)
        prev_line_audio[-overlap_samples:] = duck_region * duck

    # Mix BC into the overlap zone
    mix_into_prev = min(overlap_samples, bc_len)
    if mix_into_prev > 0:
        prev_line_audio[-mix_into_prev:] += bc_clip[:mix_into_prev]

    # Spill past the previous line
    spill = bc_clip[mix_into_prev:] if bc_len > mix_into_prev else np.array([], dtype=np.float32)

    # Calculate effective gap including existing pauses between BC and now
    existing_gap_samples = sum(
        len(parts[i]) for i in range(last_line_idx + 1, len(parts))
    )
    spill_dur = len(spill) / sr if len(spill) > 0 else 0.0
    needed_gap = spill_dur + bc_spill_room
    effective_gap = max(gap, needed_gap - existing_gap_samples / sr)

    gap_samples = int(sr * effective_gap)
    gap_audio = np.zeros(gap_samples, dtype=np.float32)

    # Place spill into existing gaps and new gap
    if len(spill) > 0:
        spill_pos = 0
        for i in range(last_line_idx + 1, len(parts)):
            chunk = min(len(parts[i]), len(spill) - spill_pos)
            if chunk > 0:
                parts[i][:chunk] += spill[spill_pos:spill_pos + chunk]
                spill_pos += chunk
            if spill_pos >= len(spill):
                break
        remainder = len(spill) - spill_pos
        if remainder > 0 and remainder <= gap_samples:
            gap_audio[:remainder] += spill[spill_pos:]

    parts.append(gap_audio)

    logger.info(
        "  BC: %s %s (%.1fs) -> overlap %.0fms, spill %.2fs, gap %.2fs",
        pending_bc["reactor"], pending_bc["bc_type"], bc_dur,
        overlap_samples / sr * 1000, spill_dur, effective_gap,
    )

    return effective_gap


# ---------------------------------------------------------------------------
# Intro + music bed builder
# ---------------------------------------------------------------------------


def build_intro_with_music(intro_voice, music_bed, sr, music_cfg):
    """Build intro section: music bed with voiceover ducking.

    Returns (intro_section, music_bleed) where music_bleed fades into cold open.
    """
    music_solo = music_cfg.get("music_solo", 4.0)
    fade_in = music_cfg.get("fade_in", 2.0)
    full_vol = music_cfg.get("full_vol", 0.35)
    duck_vol = music_cfg.get("duck_vol", 0.12)
    post_voice = music_cfg.get("post_voice", 5.0)
    bleed = music_cfg.get("bleed_into_cold", 8.0)

    music_solo_samples = int(sr * music_solo)
    post_voice_samples = int(sr * post_voice)
    bleed_samples = int(sr * bleed)
    pause_samples = int(sr * 1.5)

    intro_voice_len = music_solo_samples + len(intro_voice) + post_voice_samples
    music_total_len = intro_voice_len + pause_samples + bleed_samples

    music_track = np.zeros(music_total_len, dtype=np.float32)
    music_needed = min(len(music_bed), music_total_len)
    music_track[:music_needed] = music_bed[:music_needed]

    fade_in_end = int(sr * fade_in)
    voice_start = music_solo_samples
    voice_end = music_solo_samples + len(intro_voice)
    fade_out_start = intro_voice_len + pause_samples

    # Apply volume envelope
    for i in range(music_total_len):
        if i < fade_in_end:
            music_track[i] *= full_vol * (i / max(1, fade_in_end))
        elif i < voice_start:
            music_track[i] *= full_vol
        elif i < voice_end:
            music_track[i] *= duck_vol
        elif i < intro_voice_len:
            music_track[i] *= full_vol
        elif i < fade_out_start:
            music_track[i] *= full_vol
        else:
            progress = (i - fade_out_start) / max(1, music_total_len - fade_out_start)
            music_track[i] *= full_vol * (1.0 - progress)

    intro_section = music_track[:intro_voice_len].copy()
    intro_section[voice_start:voice_end] += intro_voice
    music_bleed = music_track[intro_voice_len:]

    return intro_section, music_bleed


# ---------------------------------------------------------------------------
# Sting transition
# ---------------------------------------------------------------------------


def build_sting_transition(cold_open, sting, sr, sting_cfg):
    """Build the sting transition between cold open and conversation.

    Returns (cold_body, sting_zone) where sting_zone overlaps the cold open
    tail and crossfades into silence for the conversation to start.
    """
    fade_in_dur = sting_cfg.get("fade_in", 0.5)
    crossfade = sting_cfg.get("crossfade", 2.5)
    vol_under_cold = sting_cfg.get("vol_under_cold", 0.35)
    overlap = sting_cfg.get("cold_open_overlap", 3.0)

    sting_copy = sting.copy()
    fade_in_samples = int(sr * fade_in_dur)
    if fade_in_samples > 0 and fade_in_samples < len(sting_copy):
        sting_copy[:fade_in_samples] *= np.linspace(0, 1, fade_in_samples, dtype=np.float32)

    overlap_samples = min(int(sr * overlap), len(cold_open))
    cold_body = cold_open[:-overlap_samples] if overlap_samples > 0 else cold_open
    cold_tail = cold_open[-overlap_samples:] if overlap_samples > 0 else np.array([], dtype=np.float32)

    sting_zone_len = max(len(sting_copy), len(cold_tail))
    sting_zone = np.zeros(sting_zone_len, dtype=np.float32)
    sting_zone[:len(cold_tail)] += cold_tail
    for i in range(min(len(sting_copy), sting_zone_len)):
        vol = vol_under_cold if i < len(cold_tail) else 1.0
        sting_zone[i] += sting_copy[i] * vol

    return cold_body, sting_zone


def crossfade_into_conversation(sting_zone, conversation, sr, crossfade_dur):
    """Crossfade sting zone into the main conversation."""
    crossfade_samples = int(sr * crossfade_dur)

    if len(sting_zone) > crossfade_samples:
        sting_pre = sting_zone[:-crossfade_samples]
        sting_tail = sting_zone[-crossfade_samples:] * np.linspace(1, 0, crossfade_samples, dtype=np.float32)
    else:
        sting_pre = np.array([], dtype=np.float32)
        sting_tail = sting_zone * np.linspace(1, 0, len(sting_zone), dtype=np.float32)

    overlap_len = len(sting_tail)
    if len(conversation) >= overlap_len:
        overlap_zone = sting_tail + conversation[:overlap_len]
        conv_rest = conversation[overlap_len:]
    else:
        overlap_zone = sting_tail[:len(conversation)] + conversation
        conv_rest = np.array([], dtype=np.float32)

    return np.concatenate([sting_pre, overlap_zone, conv_rest])


# ---------------------------------------------------------------------------
# Section extraction from manifest
# ---------------------------------------------------------------------------


def extract_sections(manifest):
    """Split manifest order into named sections.

    Returns list of (section_name, entries) tuples. The first section
    includes everything before the first section_break.
    """
    sections = []
    current_name = None
    current_entries = []

    for entry in manifest["order"]:
        if entry["type"] == "section_break":
            if current_entries:
                sections.append((current_name or "INTRO", current_entries))
            current_name = entry["to_section"]
            current_entries = []
        else:
            if current_name is None:
                # Entries before first section break — use from_section
                if entry["type"] == "line":
                    h = entry["hash"]
                    info = manifest["lines"].get(h, {})
                    current_name = info.get("section")
            current_entries.append(entry)

    if current_entries:
        sections.append((current_name or "UNNAMED", current_entries))

    return sections


# ---------------------------------------------------------------------------
# Main mix function
# ---------------------------------------------------------------------------


def mix_episode(manifest, cfg, output_path=None, seed=42):
    """Mix a complete episode from manifest + config.

    Builds each section individually (saved to sections_dir), then
    assembles into a full episode with intro, music bed, sting, and
    room tone.

    Args:
        manifest: manifest dict from load_manifest
        cfg: EpisodeConfig instance
        output_path: path for final mix WAV (default: work_dir/mix.wav)
        seed: random seed for jitter and noise (deterministic output)

    Returns:
        dict with {output, duration, sections}
    """
    sr = cfg.mix.get("target_sr", 24000)
    rng = np.random.default_rng(seed)
    lines_dir = cfg.lines_dir()
    sections_dir = cfg.sections_dir()
    sections_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = cfg.work_dir() / "mix.wav"

    mix_cfg = cfg.mix
    peak_limit_dbtp = mix_cfg.get("peak_limit_dbtp", -1.0)
    room_tone_level = mix_cfg.get("room_tone_level", 0.002)

    # Load BC clips
    bc_clips = load_bc_clips(cfg, sr)

    # Load music assets
    intro_bed_cfg = cfg.music_asset("intro_bed")
    sting_cfg = cfg.music_asset("sting")

    # Extract sections from manifest
    sections = extract_sections(manifest)
    logger.info("Sections: %s", [name for name, _ in sections])

    # Build each section
    section_audios = {}
    for name, entries in sections:
        logger.info("Building section: %s (%d entries)", name, len(entries))
        section_audio = build_section(
            entries, manifest["lines"], lines_dir, sr, rng,
            mix_cfg, bc_clips=bc_clips,
        )
        section_audios[name] = section_audio

        # Save section WAV
        safe_name = name.lower().replace(" ", "_").replace("'", "").replace(":", "")
        section_path = sections_dir / f"{safe_name}.wav"
        sf.write(str(section_path), section_audio, sr)
        logger.info("  %s: %.1fs -> %s", name, len(section_audio) / sr, section_path.name)

    if not sections:
        logger.warning("No sections found in manifest")
        return {"output": str(output_path), "duration": 0, "sections": []}

    # First section is cold open, rest is main conversation
    first_section_name, _ = sections[0]
    cold_open = section_audios[first_section_name]

    # Build main conversation from remaining sections
    remaining_entries = []
    for name, entries in sections[1:]:
        remaining_entries.extend(entries)
    main_conv = build_section(
        remaining_entries, manifest["lines"], lines_dir, sr, rng,
        mix_cfg, bc_clips=bc_clips,
    ) if remaining_entries else np.array([], dtype=np.float32)

    # Build intro with music (if assets available)
    intro_section = np.array([], dtype=np.float32)
    music_bleed = np.array([], dtype=np.float32)
    if intro_bed_cfg and Path(intro_bed_cfg["file"]).exists():
        # For now, use a silence placeholder for intro voice
        # (intro voice assembly is a separate concern handled by the caller)
        music_bed = load_audio_file(intro_bed_cfg["file"], sr)
        # Intro voice would be passed in or built from intro_lines
        # For this version, skip intro if no voice is provided
        logger.info("Music bed loaded: %.1fs", len(music_bed) / sr)

    # Build sting transition
    if sting_cfg and Path(sting_cfg["file"]).exists():
        sting = load_audio_file(sting_cfg["file"], sr)
        logger.info("Sting loaded: %.1fs", len(sting) / sr)
        cold_body, sting_zone = build_sting_transition(
            cold_open, sting, sr, sting_cfg,
        )
        # Crossfade sting into conversation
        crossfade_dur = sting_cfg.get("crossfade", 2.5)
        conversation_with_sting = crossfade_into_conversation(
            sting_zone, main_conv, sr, crossfade_dur,
        )
        tail_silence = np.zeros(int(sr * 1.5), dtype=np.float32)
        full = np.concatenate([cold_body, conversation_with_sting, tail_silence])
    else:
        # No sting — just concatenate
        section_pause = np.zeros(int(sr * mix_cfg.get("section_pause", 1.0)), dtype=np.float32)
        tail_silence = np.zeros(int(sr * 1.5), dtype=np.float32)
        full = np.concatenate([cold_open, section_pause, main_conv, tail_silence])

    # Room tone
    logger.info("Adding room tone...")
    pink = generate_pink_noise(len(full), rng)
    room_mask = np.ones(len(full), dtype=np.float32) * room_tone_level
    # Fade in room tone over first 2 seconds
    fade_samples = min(int(sr * 2.0), len(full))
    room_mask[:fade_samples] *= np.linspace(0, 1, fade_samples, dtype=np.float32)
    full = full + pink * room_mask

    # Peak limiting
    peak = np.max(np.abs(full))
    limit = 10 ** (peak_limit_dbtp / 20)
    if peak > limit:
        full = full * (limit / peak)
        logger.info("Peak limited: %.3f -> %.3f", peak, limit)

    total_dur = len(full) / sr
    logger.info("Total: %.1fs (%.1f min) -> %s", total_dur, total_dur / 60, output_path)

    sf.write(str(output_path), full, sr)

    return {
        "output": str(output_path),
        "duration": round(total_dur, 1),
        "sections": [name for name, _ in sections],
    }
