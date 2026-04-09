"""Batch audio processing for podcast pipeline.

Reads raw TTS files from tts_dir, applies processing (trim, fade, normalize,
reverb, click suppression, speaker volume), writes to lines_dir. Raw files
are never modified — processing always creates new copies.

Also processes backchannel clips with the same fade/click treatment.

Usage:
    from process_lines import process_all
    from config import load_episode_config
    from manifest import load_manifest

    cfg = load_episode_config("episodes/ep01.yaml")
    manifest = load_manifest(cfg.work_dir() / "manifest.json")
    process_all(manifest, cfg)
"""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from manifest import STATUS_EXISTS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio processing functions
# ---------------------------------------------------------------------------


def trim_silence(audio, sr, threshold_db=-35, pre_roll_ms=40):
    """Trim leading silence, keeping pre_roll_ms before first detected audio."""
    threshold = 10 ** (threshold_db / 20)
    window = max(1, int(sr * 0.01))
    abs_audio = np.abs(audio)
    pre_roll = int(sr * pre_roll_ms / 1000)
    for i in range(0, len(audio) - window, window):
        if np.max(abs_audio[i:i + window]) > threshold:
            start = max(0, i - pre_roll)
            return audio[start:]
    return audio


def rms_normalize(audio, target_rms=0.1):
    """Normalize audio to target RMS level."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        return audio * (target_rms / rms)
    return audio


def apply_speaker_volume(audio, volume_db):
    """Apply per-speaker volume offset in dB."""
    if volume_db != 0.0:
        return audio * (10 ** (volume_db / 20))
    return audio


def generate_room_ir(sr, decay_time=0.3):
    """Generate a synthetic room impulse response."""
    num_samples = int(sr * decay_time)
    ir = np.random.randn(num_samples).astype(np.float32)
    decay = np.exp(-np.linspace(0, 6, num_samples)).astype(np.float32)
    ir *= decay
    for delay_ms, gain in [(5, 0.4), (12, 0.25), (20, 0.15), (35, 0.1)]:
        pos = int(sr * delay_ms / 1000)
        if pos < num_samples:
            ir[pos] += gain
    ir /= np.max(np.abs(ir))
    return ir


def apply_reverb(audio, ir, mix=0.02):
    """Apply convolution reverb with wet/dry mix."""
    wet = fftconvolve(audio, ir, mode="full")[:len(audio)].astype(np.float32)
    return audio * (1 - mix) + wet * mix


def apply_clip_fades(audio, sr, fade_in_ms=35, fade_out_ms=20,
                     click_check_ms=50, click_threshold=0.08,
                     click_smooth_samples=7):
    """Fade edges and suppress clicks at clip boundaries.

    Fade-in is capped to the leading silence so it never eats into speech.
    Click suppression smooths large sample-to-sample jumps in the first/last
    check region.
    """
    # Detect where speech starts (first sample above -40 dB)
    speech_thresh = 10 ** (-40 / 20)
    speech_start = len(audio)
    for i in range(len(audio)):
        if abs(audio[i]) > speech_thresh:
            speech_start = i
            break

    # Fade-in: never longer than the silence before speech
    fade_in = min(int(sr * fade_in_ms / 1000), max(speech_start - 1, 1))
    fade_out = int(sr * fade_out_ms / 1000)
    if len(audio) > fade_in + fade_out:
        audio[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
        audio[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)

    # Click suppression in first/last check region
    check_samples = int(sr * click_check_ms / 1000)
    half_smooth = click_smooth_samples // 2
    for region in [audio[:check_samples], audio[-check_samples:]]:
        for i in range(1, len(region)):
            jump = abs(region[i] - region[i - 1])
            if jump > click_threshold:
                start = max(0, i - half_smooth)
                end = min(len(region), i + half_smooth + 1)
                region[start:end] = np.linspace(
                    region[start], region[min(end, len(region) - 1)],
                    end - start, dtype=np.float32,
                )
    return audio


def process_one(audio, sr, volume_db, room_ir, processing_cfg):
    """Apply the full processing chain to one audio clip.

    Processing order: trim → normalize → speaker volume → reverb → fades.
    """
    p = processing_cfg
    audio = trim_silence(
        audio, sr,
        threshold_db=p.get("trim_threshold_db", -35),
        pre_roll_ms=p.get("pre_roll_ms", 40),
    )
    audio = rms_normalize(audio, target_rms=p.get("rms_target", 0.1))
    audio = apply_speaker_volume(audio, volume_db)
    audio = apply_reverb(audio, room_ir, mix=p.get("reverb_mix", 0.02))
    audio = apply_clip_fades(
        audio, sr,
        fade_in_ms=p.get("fade_in_ms", 35),
        fade_out_ms=p.get("fade_out_ms", 20),
        click_check_ms=p.get("click_check_ms", 50),
        click_threshold=p.get("click_threshold", 0.08),
        click_smooth_samples=p.get("click_smooth_samples", 7),
    )
    return audio


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_all(manifest, cfg):
    """Process all lines from tts_dir → lines_dir.

    Reads raw TTS files, applies processing chain, writes to lines_dir.
    Also processes backchannel clips. Raw files in tts_dir are never modified.

    Args:
        manifest: manifest dict with lines
        cfg: EpisodeConfig instance

    Returns:
        dict with counts: {processed, skipped, total}
    """
    tts_dir = cfg.tts_dir()
    lines_dir = cfg.lines_dir()
    lines_dir.mkdir(parents=True, exist_ok=True)

    target_sr = cfg.mix.get("target_sr", 24000)
    processing_cfg = cfg.processing
    reverb_decay = cfg.mix.get("reverb_decay", 0.15)

    # Generate room IR (deterministic with fixed seed for consistency)
    rng = np.random.RandomState(42)
    room_ir = _generate_room_ir_seeded(target_sr, reverb_decay, rng)

    lines = manifest["lines"]
    processed = 0
    skipped = 0

    for h, info in lines.items():
        if info["status"] != STATUS_EXISTS:
            skipped += 1
            continue

        src = tts_dir / info["file"]
        dst = lines_dir / info["file"]

        if not src.exists():
            logger.warning("Source file missing: %s", src)
            skipped += 1
            continue

        if dst.exists():
            skipped += 1
            continue

        audio, file_sr = sf.read(str(src), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != target_sr:
            from scipy.signal import resample
            audio = resample(audio, int(len(audio) * target_sr / file_sr)).astype(np.float32)

        # Get speaker volume from config
        speaker = info["speaker"]
        try:
            volume_db = cfg.cast(speaker).get("volume_db", 0.0)
        except KeyError:
            volume_db = 0.0

        audio = process_one(audio, target_sr, volume_db, room_ir, processing_cfg)

        sf.write(str(dst), audio, target_sr)
        processed += 1

        if processed % 20 == 0:
            logger.info("Processed %d lines...", processed)

    logger.info("Lines: %d processed, %d skipped, %d total",
                processed, skipped, len(lines))

    # Process backchannel clips
    bc_processed = _process_backchannels(cfg, target_sr, processing_cfg, room_ir)

    return {"processed": processed, "skipped": skipped,
            "total": len(lines), "backchannels": bc_processed}


def _generate_room_ir_seeded(sr, decay_time, rng):
    """Generate room IR with a seeded RNG for deterministic output."""
    num_samples = int(sr * decay_time)
    ir = rng.randn(num_samples).astype(np.float32)
    decay = np.exp(-np.linspace(0, 6, num_samples)).astype(np.float32)
    ir *= decay
    for delay_ms, gain in [(5, 0.4), (12, 0.25), (20, 0.15), (35, 0.1)]:
        pos = int(sr * delay_ms / 1000)
        if pos < num_samples:
            ir[pos] += gain
    ir /= np.max(np.abs(ir))
    return ir


def _process_backchannels(cfg, target_sr, processing_cfg, room_ir):
    """Process backchannel clips: apply fades and click suppression.

    BC clips get lighter processing than dialogue lines — no reverb or
    RMS normalization, just fades and click suppression.
    """
    bc_dir = cfg.backchannels_dir()
    if not bc_dir.exists():
        return 0

    processed = 0
    for speaker in cfg.cast_names():
        for clip_info in cfg.backchannel_clips(speaker):
            src_path = Path(clip_info["file"])
            if not src_path.exists():
                continue

            # Check if already processed (same file, same location)
            audio, file_sr = sf.read(str(src_path), dtype="float32")
            if file_sr != target_sr:
                from scipy.signal import resample
                audio = resample(audio, int(len(audio) * target_sr / file_sr)).astype(np.float32)

            audio = apply_clip_fades(
                audio, target_sr,
                fade_in_ms=processing_cfg.get("fade_in_ms", 35),
                fade_out_ms=processing_cfg.get("fade_out_ms", 20),
                click_check_ms=processing_cfg.get("click_check_ms", 50),
                click_threshold=processing_cfg.get("click_threshold", 0.08),
                click_smooth_samples=processing_cfg.get("click_smooth_samples", 7),
            )

            sf.write(str(src_path), audio, target_sr)
            processed += 1

    if processed:
        logger.info("Backchannels: %d clips processed", processed)
    return processed
