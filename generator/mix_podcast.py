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
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

try:
    from pedalboard import Pedalboard, Limiter as PedalboardLimiter
    _HAS_PEDALBOARD = True
except ImportError:
    _HAS_PEDALBOARD = False

from manifest import STATUS_EXISTS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


def load_audio_file(path, target_sr):
    """Load any audio file (MP3/WAV/etc) via ffmpeg, convert to mono float32.

    Raises RuntimeError if ffmpeg fails.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(path),
             "-ar", str(target_sr), "-ac", "1", "-f", "wav", tmp_path],
            capture_output=True, timeout=60,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")[-200:]
            raise RuntimeError(f"ffmpeg failed for {path}: {stderr}")
        audio, _ = sf.read(tmp_path, dtype="float32")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return audio


def _load_line(path):
    """Load a processed WAV line (mono float32, already at target SR)."""
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

    Note: assumes flat filenames in backchannel clip config (no subdirectories).
    """
    bc_dir = cfg.processed_backchannels_dir()
    bc_clips = {}

    if not bc_dir.exists():
        return bc_clips

    for speaker in cfg.cast_names():
        for clip_info in cfg.backchannel_clips(speaker):
            # Extract just the filename — redirect from raw dir to processed dir
            clip_name = Path(clip_info["file"]).name
            clip_path = bc_dir / clip_name
            if not clip_path.exists():
                continue

            audio, sr = sf.read(str(clip_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            bc_type = clip_info["type"]

            if bc_type not in bc_clips:
                bc_clips[bc_type] = {}
            if speaker not in bc_clips[bc_type]:
                bc_clips[bc_type][speaker] = []
            bc_clips[bc_type][speaker].append(audio)

    total = sum(len(v) for t in bc_clips.values() for v in t.values())
    if total:
        logger.info("Loaded %d backchannel clips", total)
    else:
        logger.warning("No backchannel clips found — [react:] cues will be silent")
    return bc_clips


_BC_TYPE_FALLBACK = {
    "huh": "thinking",
    "hmm": "thinking",
    "gasp": "outbreath",
    "sigh": "outbreath",
    "breath": "outbreath",
}


def _pick_bc_clip(bc_clips, reactor, bc_type, rng):
    """Pick a random backchannel clip for a reactor+type.

    Falls back to related types if exact match not available
    (e.g. huh -> breath).
    """
    clips = bc_clips.get(bc_type, {}).get(reactor, [])
    if not clips:
        fallback = _BC_TYPE_FALLBACK.get(bc_type)
        if fallback:
            clips = bc_clips.get(fallback, {}).get(reactor, [])
    if not clips:
        logger.debug("No BC clip for [react: %s %s] (no fallback)", reactor, bc_type)
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
    """
    speaker_change_pause = mix_cfg.get("speaker_change_pause", 0.15)
    same_speaker_pause = mix_cfg.get("same_speaker_pause", 0.08)
    interjection_pause = mix_cfg.get("interjection_pause", 0.05)
    interjection_threshold = mix_cfg.get("interjection_threshold", 1.5)
    section_pause = mix_cfg.get("section_pause", 1.0)

    bc_cfg = mix_cfg.get("backchannel", {})
    bc_default_vol = bc_cfg.get("volume_db", -3.0)
    bc_vol_by_type = bc_cfg.get("volume_by_type", {})
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
                speaker = info["speaker"] if info else "unknown"
                logger.warning("Missing/failed line %s (%s) — substituting 0.5s silence", h, speaker)
                if pending_bc is not None:
                    logger.warning("  Discarding pending BC cue (%s) adjacent to missing line", pending_bc["reactor"])
                parts.append(np.zeros(int(sr * 0.5), dtype=np.float32))
                pending_bc = None
                continue

            wav_path = lines_dir / info["file"]
            if not wav_path.exists():
                logger.warning("WAV missing for line %s (%s): %s — substituting 0.5s silence", h, info["speaker"], wav_path.name)
                if pending_bc is not None:
                    logger.warning("  Discarding pending BC cue (%s) adjacent to missing line", pending_bc["reactor"])
                parts.append(np.zeros(int(sr * 0.5), dtype=np.float32))
                pending_bc = None
                continue

            audio = _load_line(wav_path)

            if prev_speaker is not None:
                if info["speaker"] == prev_speaker:
                    base = same_speaker_pause
                elif prev_line_dur < interjection_threshold:
                    base = interjection_pause
                else:
                    base = speaker_change_pause
                jitter = rng.uniform(-0.02, 0.04)
                gap = max(0.03, base + jitter)

                if pending_bc is not None and last_line_idx is not None:
                    vol_db = bc_vol_by_type.get(pending_bc["bc_type"], bc_default_vol)
                    bc_vol = 10 ** (vol_db / 20)
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
                if pending_bc is not None:
                    logger.debug("Discarding BC cue before first line in section")
                pending_bc = None

            # Micro-fade at boundaries to prevent join clicks
            fade_samples = min(int(sr * 0.002), len(audio) // 4)
            if fade_samples > 1:
                fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
                audio[-fade_samples:] *= fade_out

            parts.append(audio)
            last_line_idx = len(parts) - 1
            prev_line_dur = len(audio) / sr
            prev_speaker = info["speaker"]

    if not parts:
        return np.array([], dtype=np.float32)

    return np.concatenate(parts)


def _place_backchannel(parts, last_line_idx, bc_info, gap, sr, rng,
                       bc_vol, bc_overlap_ms, bc_duck_threshold,
                       bc_duck_level, bc_spill_room):
    """Place a backchannel clip overlapping the previous line.

    Mutates parts[last_line_idx] in-place for ducking.
    Appends gap audio to parts. Returns the effective gap duration.
    """
    bc_clip = bc_info["clip"] * bc_vol
    bc_len = len(bc_clip)
    bc_dur = bc_len / sr

    prev_line_audio = parts[last_line_idx]

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

    existing_gap_samples = sum(
        len(parts[i]) for i in range(last_line_idx + 1, len(parts))
    )
    spill_dur = len(spill) / sr if len(spill) > 0 else 0.0
    needed_gap = spill_dur + bc_spill_room
    effective_gap = max(gap, needed_gap - existing_gap_samples / sr)

    gap_samples = int(sr * effective_gap)
    gap_audio = np.zeros(gap_samples, dtype=np.float32)

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
        elif remainder > gap_samples:
            logger.warning("  BC spill truncated: %d samples (%.0fms) exceeded gap",
                           remainder - gap_samples, (remainder - gap_samples) / sr * 1000)

    parts.append(gap_audio)

    logger.info(
        "  BC: %s %s (%.1fs) -> overlap %.0fms, spill %.2fs, gap %.2fs",
        bc_info["reactor"], bc_info["bc_type"], bc_dur,
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
    fade_in = min(music_cfg.get("fade_in", 2.0), music_solo)  # clamp to avoid envelope gap
    full_vol = music_cfg.get("full_vol", 0.35)
    duck_vol = music_cfg.get("duck_vol", 0.12)
    post_voice = music_cfg.get("post_voice", 5.0)
    bleed = music_cfg.get("bleed_into_cold", 8.0)

    music_solo_samples = int(sr * music_solo)
    post_voice_samples = int(sr * post_voice)
    bleed_samples = int(sr * bleed)
    pause_samples = int(sr * music_cfg.get("pause_after_voice", 1.5))

    intro_voice_len = music_solo_samples + len(intro_voice) + post_voice_samples
    music_total_len = intro_voice_len + pause_samples + bleed_samples

    music_track = np.zeros(music_total_len, dtype=np.float32)
    music_needed = min(len(music_bed), music_total_len)
    if music_needed < music_total_len:
        logger.warning(
            "Music bed (%.1fs) shorter than needed (%.1fs) — tail will be silent",
            len(music_bed) / sr, music_total_len / sr,
        )
    music_track[:music_needed] = music_bed[:music_needed]

    # Build volume envelope as a separate array, then apply once.
    # This avoids double-multiplication when fade_in >= music_solo.
    fade_in_end = int(sr * fade_in)
    voice_start = music_solo_samples
    voice_end = music_solo_samples + len(intro_voice)
    fade_out_start = intro_voice_len + pause_samples

    envelope = np.zeros(music_total_len, dtype=np.float32)
    # Fade in (0 → full_vol)
    if fade_in_end > 0:
        envelope[:fade_in_end] = np.linspace(0, full_vol, fade_in_end, dtype=np.float32)
    # Full volume before voice
    envelope[fade_in_end:voice_start] = full_vol
    # Duck ramp into voice (50ms transition to avoid click)
    duck_ramp = min(int(sr * 0.05), (voice_end - voice_start) // 4)
    envelope[voice_start:voice_start + duck_ramp] = np.linspace(
        full_vol, duck_vol, duck_ramp, dtype=np.float32)
    envelope[voice_start + duck_ramp:voice_end] = duck_vol
    # Ramp back up after voice
    envelope[voice_end:voice_end + duck_ramp] = np.linspace(
        duck_vol, full_vol, duck_ramp, dtype=np.float32)
    envelope[voice_end + duck_ramp:fade_out_start] = full_vol
    # Fade out (full_vol → 0)
    fade_out_len = music_total_len - fade_out_start
    if fade_out_len > 0:
        envelope[fade_out_start:] = np.linspace(full_vol, 0, fade_out_len, dtype=np.float32)

    music_track *= envelope

    intro_section = music_track[:intro_voice_len].copy()
    intro_section[voice_start:voice_end] += intro_voice
    music_bleed = music_track[intro_voice_len:]

    return intro_section, music_bleed


# ---------------------------------------------------------------------------
# Sting transition
# ---------------------------------------------------------------------------


def build_sting_transition(cold_open, sting, sr, sting_cfg):
    """Build the sting transition between cold open and conversation.

    Returns (cold_body, sting_zone). The sting_zone contains the cold open
    tail mixed with the sting at reduced volume. The caller must then use
    crossfade_into_conversation() to fade the sting_zone into the
    conversation — the sting_zone does NOT fade out on its own.
    """
    fade_in_dur = sting_cfg.get("fade_in", 0.5)
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

    # Sting plays at vol_under_cold while cold tail is present, full volume after
    sting_len = min(len(sting_copy), sting_zone_len)
    cold_len = min(len(cold_tail), sting_len)
    vol_arr = np.ones(sting_len, dtype=np.float32)
    vol_arr[:cold_len] = vol_under_cold
    sting_zone[:sting_len] += sting_copy[:sting_len] * vol_arr

    return cold_body, sting_zone


def crossfade_into_conversation(sting_zone, conversation, sr, crossfade_dur):
    """Crossfade sting zone into the main conversation.

    The sting fades out while conversation plays at full volume on top.
    This is a one-sided fade: only the sting ramps down.  The conversation
    starts at full volume so first words aren't eaten by the crossfade.
    """
    crossfade_samples = int(sr * crossfade_dur)

    if len(sting_zone) > crossfade_samples:
        sting_pre = sting_zone[:-crossfade_samples]
        sting_tail = sting_zone[-crossfade_samples:] * np.linspace(1, 0, crossfade_samples, dtype=np.float32)
    else:
        sting_pre = np.array([], dtype=np.float32)
        sting_tail = sting_zone * np.linspace(1, 0, len(sting_zone), dtype=np.float32)

    # Conversation plays at full volume — sting fades underneath
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
            if current_name is None and entry["type"] == "line":
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


def mix_episode(manifest, cfg, output_path=None, intro_voice=None, seed=42):
    """Mix a complete episode from manifest + config.

    Builds each section individually (saved to sections_dir), then
    assembles into a full episode with intro, music bed, sting, and
    room tone.

    Args:
        manifest: manifest dict from load_manifest
        cfg: EpisodeConfig instance
        output_path: path for final mix WAV (default: work_dir/mix.wav)
        intro_voice: pre-built intro voiceover array, or None to skip intro
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
    tail_silence_dur = mix_cfg.get("tail_silence", 1.5)

    # Load BC clips
    bc_clips = load_bc_clips(cfg, sr)

    # Extract sections from manifest
    sections = extract_sections(manifest)
    logger.info("Sections: %s", [name for name, _ in sections])

    if not sections:
        logger.warning("No sections found in manifest")
        sf.write(str(output_path), np.array([], dtype=np.float32), sr)
        return {"output": str(output_path), "duration": 0, "sections": []}

    # Build each section and save individually
    section_audios = {}
    for name, entries in sections:
        logger.info("Building section: %s (%d entries)", name, len(entries))
        section_audio = build_section(
            entries, manifest["lines"], lines_dir, sr, rng,
            mix_cfg, bc_clips=bc_clips,
        )
        section_audios[name] = section_audio

        safe_name = name.lower().replace(" ", "_").replace("'", "").replace(":", "")
        section_path = sections_dir / f"{safe_name}.wav"
        sf.write(str(section_path), section_audio, sr)
        logger.info("  %s: %.1fs -> %s", name, len(section_audio) / sr, section_path.name)

    # Assemble from already-built sections.
    # If an INTRO section exists, use it as the intro voice (over music bed).
    # The first non-INTRO section is the cold open (sting plays after it).
    section_names_list = [name for name, _ in sections]
    body_sections = [(n, e) for n, e in sections if n.upper() != "INTRO"]

    # Extract intro voice from INTRO section if present
    if "INTRO" in section_audios:
        auto_intro = section_audios["INTRO"]
        if intro_voice is None or len(intro_voice) == 0:
            intro_voice = auto_intro
            logger.info("Using INTRO section as intro voice (%.1fs)", len(auto_intro) / sr)

    cold_open = section_audios[body_sections[0][0]] if body_sections else np.array([], dtype=np.float32)

    # Main conversation: everything after cold open
    section_pause_dur = mix_cfg.get("section_pause", 1.0)
    main_parts = []
    for name, _ in body_sections[1:]:
        if main_parts:
            main_parts.append(np.zeros(int(sr * section_pause_dur), dtype=np.float32))
        main_parts.append(section_audios[name])
    main_conv = np.concatenate(main_parts) if main_parts else np.array([], dtype=np.float32)

    # Build intro with music
    intro_section = np.array([], dtype=np.float32)
    music_bleed = np.array([], dtype=np.float32)
    intro_bed_cfg = cfg.music_asset("intro_bed")
    if intro_voice is not None and len(intro_voice) > 0 and intro_bed_cfg:
        if Path(intro_bed_cfg["file"]).exists():
            music_bed = load_audio_file(intro_bed_cfg["file"], sr)
            intro_section, music_bleed = build_intro_with_music(
                intro_voice, music_bed, sr, intro_bed_cfg,
            )
            logger.info("Intro: %.1fs (voice + music)", len(intro_section) / sr)

    # Build sting transition
    sting_cfg = cfg.music_asset("sting")
    if sting_cfg and Path(sting_cfg["file"]).exists():
        sting = load_audio_file(sting_cfg["file"], sr)
        logger.info("Sting loaded: %.1fs", len(sting) / sr)
        cold_body, sting_zone = build_sting_transition(cold_open, sting, sr, sting_cfg)
        crossfade_dur = sting_cfg.get("crossfade", 2.5)
        conversation_with_sting = crossfade_into_conversation(
            sting_zone, main_conv, sr, crossfade_dur,
        )
        tail_silence = np.zeros(int(sr * tail_silence_dur), dtype=np.float32)
        episode_body = np.concatenate([cold_body, conversation_with_sting, tail_silence])
    else:
        section_pause = np.zeros(int(sr * section_pause_dur), dtype=np.float32)
        tail_silence = np.zeros(int(sr * tail_silence_dur), dtype=np.float32)
        episode_body = np.concatenate([cold_open, section_pause, main_conv, tail_silence])

    # Prepend intro + music bleed
    if len(intro_section) > 0:
        pause_after_intro = np.zeros(int(sr * mix_cfg.get("pause_after_intro", 1.5)), dtype=np.float32)
        bleed_len = min(len(music_bleed), len(pause_after_intro) + len(episode_body))
        body_with_pause = np.concatenate([pause_after_intro, episode_body])
        body_with_pause[:bleed_len] += music_bleed[:bleed_len]
        full = np.concatenate([intro_section, body_with_pause])
    else:
        full = episode_body

    # Room tone
    logger.info("Adding room tone...")
    pink = generate_pink_noise(len(full), rng)
    room_mask = np.ones(len(full), dtype=np.float32) * room_tone_level
    fade_samples = min(int(sr * 2.0), len(full))
    room_mask[:fade_samples] *= np.linspace(0, 1, fade_samples, dtype=np.float32)
    room_mask[-fade_samples:] *= np.linspace(1, 0, fade_samples, dtype=np.float32)
    full = full + pink * room_mask

    # Peak limiting
    peak_before = float(np.max(np.abs(full)))
    limit = 10 ** (peak_limit_dbtp / 20)
    if _HAS_PEDALBOARD:
        board = Pedalboard([PedalboardLimiter(threshold_db=peak_limit_dbtp)])
        full = board(full[np.newaxis, :].astype(np.float32), sr)[0]
        # Hard-clip as safety net — the limiter handles dynamics gracefully
        # but may allow transient peaks slightly above threshold
        np.clip(full, -limit, limit, out=full)
        peak_after = float(np.max(np.abs(full)))
        if peak_before > limit:
            logger.info("Peak limited: %.3f -> %.3f", peak_before, peak_after)
    elif peak_before > limit:
        full = full * (limit / peak_before)
        logger.info("Peak limited: %.3f -> %.3f", peak_before, limit)

    total_dur = len(full) / sr
    logger.info("Total: %.1fs (%.1f min) -> %s", total_dur, total_dur / 60, output_path)

    sf.write(str(output_path), full, sr)

    return {
        "output": str(output_path),
        "duration": round(total_dur, 1),
        "sections": [name for name, _ in sections],
    }
