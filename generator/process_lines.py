"""Batch audio processing for podcast pipeline.

Reads raw TTS files from tts_dir, applies processing (trim, fade, normalize,
reverb, click suppression, speaker volume), writes to lines_dir. Raw files
are never modified — processing always creates new copies.

Also processes backchannel clips with the same fade/click treatment, writing
processed copies to a subdirectory (originals preserved).

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
from scipy.signal import fftconvolve, resample

try:
    import pyloudnorm as pyln
    _HAS_PYLOUDNORM = True
except ImportError:
    pyln = None
    _HAS_PYLOUDNORM = False

try:
    from pedalboard import Pedalboard, Reverb as PedalboardReverb
    _HAS_PEDALBOARD = True
except ImportError:
    _HAS_PEDALBOARD = False

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

from manifest import STATUS_EXISTS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio processing functions
# ---------------------------------------------------------------------------


def trim_silence(audio, sr, threshold_db=-35, pre_roll_ms=40,
                 min_pad_ms=25):
    """Trim leading/trailing silence, keeping pre_roll_ms around speech.

    If speech starts or ends too close to a clip boundary (common Qwen3-TTS
    artifact), silence is padded to guarantee the micro-fade has room to
    ramp without eating into speech.
    """
    threshold = 10 ** (threshold_db / 20)
    window = max(1, int(sr * 0.01))
    abs_audio = np.abs(audio)
    pre_roll = int(sr * pre_roll_ms / 1000)
    min_pad = int(sr * min_pad_ms / 1000)

    # Find speech start
    speech_start = len(audio)
    for i in range(0, len(audio), window):
        chunk = abs_audio[i:i + window]
        if len(chunk) > 0 and np.max(chunk) > threshold:
            speech_start = i
            break

    if speech_start == len(audio):
        return audio  # all silence

    # Find speech end (scan backwards)
    speech_end = 0
    for i in range(len(audio) - 1, -1, -window):
        chunk = abs_audio[max(0, i - window):i + 1]
        if len(chunk) > 0 and np.max(chunk) > threshold:
            speech_end = i + 1
            break

    # Trim with pre-roll on both sides
    start = max(0, speech_start - pre_roll)
    end = min(len(audio), speech_end + pre_roll)
    trimmed = audio[start:end]

    # Pad start if speech is too close to the beginning
    silence_before = speech_start - start
    if silence_before < min_pad:
        head_pad = np.zeros(min_pad - silence_before, dtype=np.float32)
        trimmed = np.concatenate([head_pad, trimmed])

    # Pad end if speech is too close to the end
    silence_after = end - speech_end
    if silence_after < min_pad:
        tail_pad = np.zeros(min_pad - silence_after, dtype=np.float32)
        trimmed = np.concatenate([trimmed, tail_pad])

    return trimmed


def rms_normalize(audio, target_rms=0.1):
    """Normalize audio to target RMS level (legacy fallback)."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        return audio * (target_rms / rms)
    return audio


# Minimum duration for reliable LUFS measurement (pyloudnorm needs enough
# samples for the 400ms gating window).  Clips shorter than this fall back
# to RMS normalization.
_LUFS_MIN_DURATION = 0.5  # seconds


def lufs_normalize(audio, sr, target_lufs=-16.0, target_rms=0.1):
    """Normalize audio to target LUFS (broadcast standard).

    Uses pyloudnorm for perceptually correct loudness normalization.
    Falls back to RMS normalization when pyloudnorm is unavailable or
    the clip is too short for reliable LUFS measurement (<0.5s).
    """
    duration = len(audio) / sr
    if not _HAS_PYLOUDNORM or duration < _LUFS_MIN_DURATION:
        return rms_normalize(audio, target_rms)

    meter = pyln.Meter(sr)
    # pyloudnorm expects (samples, channels)
    mono_2d = audio.reshape(-1, 1)
    current_lufs = meter.integrated_loudness(mono_2d)

    # If the audio is effectively silent, LUFS returns -inf
    if current_lufs == float("-inf") or current_lufs < -70:
        return audio

    gain_db = target_lufs - current_lufs
    gain_linear = 10 ** (gain_db / 20)
    return (audio * gain_linear).astype(np.float32)


def _find_speech_boundaries(audio, sr, threshold_db=-35):
    """Find speech start and end positions in audio.

    Uses librosa onset detection when available (spectral flux — more
    robust for speech than amplitude thresholding). Falls back to
    sample-by-sample amplitude scanning.

    Returns:
        (speech_start, speech_end) sample indices.
        speech_start == len(audio) if no speech found.
    """
    threshold = 10 ** (threshold_db / 20)

    # Librosa path: spectral flux onset detection
    if _HAS_LIBROSA and len(audio) > sr * 0.05:
        onsets = librosa.onset.onset_detect(
            y=audio, sr=sr, units="samples", backtrack=True,
            hop_length=256, pre_max=3, post_max=3,
            pre_avg=3, post_avg=5, delta=0.05, wait=1,
        )
        if len(onsets) > 0:
            # First onset is speech start; refine with threshold scan
            # near the detected onset (librosa may overshoot by a hop)
            onset_sample = int(onsets[0])
            search_start = max(0, onset_sample - 512)
            for i in range(search_start, min(len(audio), onset_sample + 512)):
                if abs(audio[i]) > threshold:
                    speech_start = i
                    break
            else:
                speech_start = onset_sample

            # Speech end: scan backward from end
            speech_end = 0
            for i in range(len(audio) - 1, -1, -1):
                if abs(audio[i]) > threshold:
                    speech_end = i + 1
                    break
            return speech_start, speech_end

    # Legacy path: sample-by-sample threshold scanning
    speech_start = len(audio)
    for i in range(len(audio)):
        if abs(audio[i]) > threshold:
            speech_start = i
            break

    speech_end = 0
    for i in range(len(audio) - 1, -1, -1):
        if abs(audio[i]) > threshold:
            speech_end = i + 1
            break

    return speech_start, speech_end


def smooth_onset_glitches(audio, sr, threshold_db=-35,
                          scan_ms=10, reversal_threshold=0.025,
                          smooth_radius=4):
    """Smooth micro-glitches near speech onset and offset.

    Qwen3-TTS often produces a sharp sample reversal (direction change)
    in the first few ms of speech.  This scans the onset and offset
    regions for abnormal reversals and replaces them with a linear
    interpolation over a small window.

    Uses librosa onset detection when available for more accurate
    speech boundary finding (spectral flux vs amplitude threshold).

    Runs on raw TTS audio, before normalization (so amplitudes are
    at TTS output scale).
    """
    speech_start, speech_end = _find_speech_boundaries(audio, sr, threshold_db)
    if speech_start >= len(audio):
        return audio

    scan = int(sr * scan_ms / 1000)

    # Smooth onset and offset regions (up to 3 passes for nested glitches)
    for _ in range(3):
        fixed_any = False
        for region_start, region_end in [
            (speech_start, min(speech_start + scan, len(audio))),
            (max(speech_end - scan, 0), speech_end),
        ]:
            region = audio[region_start:region_end]
            if len(region) < 3:
                continue
            diffs = np.diff(region)
            for j in range(1, len(diffs)):
                if diffs[j] * diffs[j - 1] < 0:  # sign change
                    reversal = abs(diffs[j]) + abs(diffs[j - 1])
                    if reversal > reversal_threshold:
                        abs_start = max(0, region_start + j - smooth_radius)
                        abs_end = min(len(audio), region_start + j + smooth_radius + 1)
                        audio[abs_start:abs_end] = np.linspace(
                            audio[abs_start], audio[abs_end - 1],
                            abs_end - abs_start, dtype=np.float32,
                        )
                        fixed_any = True
        if not fixed_any:
            break
    return audio


def apply_speaker_volume(audio, volume_db):
    """Apply per-speaker volume offset in dB."""
    if volume_db != 0.0:
        return audio * (10 ** (volume_db / 20))
    return audio


def generate_room_ir(sr, decay_time=0.3, rng=None):
    """Generate a synthetic room impulse response.

    Args:
        sr: sample rate
        decay_time: reverb tail duration in seconds
        rng: numpy Generator for deterministic output. Uses default_rng if None.
    """
    if rng is None:
        rng = np.random.default_rng()
    num_samples = max(1, int(sr * decay_time))
    ir = rng.standard_normal(num_samples).astype(np.float32)
    decay = np.exp(-np.linspace(0, 6, num_samples)).astype(np.float32)
    ir *= decay
    for delay_ms, gain in [(5, 0.4), (12, 0.25), (20, 0.15), (35, 0.1)]:
        pos = int(sr * delay_ms / 1000)
        if pos < num_samples:
            ir[pos] += gain
    peak = np.max(np.abs(ir))
    if peak > 0:
        ir /= peak
    return ir


def apply_reverb(audio, ir, mix=0.02):
    """Apply convolution reverb with wet/dry mix (legacy path)."""
    wet = fftconvolve(audio, ir, mode="full")[:len(audio)].astype(np.float32)
    return audio * (1 - mix) + wet * mix


def apply_reverb_pedalboard(audio, sr, mix=0.02, room_size=0.15):
    """Apply reverb using Pedalboard (Spotify JUCE Freeverb).

    Higher quality than synthetic IR convolution, deterministic,
    and no IR generation needed.
    """
    board = Pedalboard([PedalboardReverb(
        room_size=room_size,
        wet_level=mix,
        dry_level=1.0 - mix,
        width=0.5,
    )])
    # Pedalboard expects (channels, samples) float32
    result = board(audio[np.newaxis, :].astype(np.float32), sr)
    return result[0]


def apply_clip_fades(audio, sr, fade_in_ms=35, fade_out_ms=20,
                     click_check_ms=50, click_threshold=0.08,
                     click_smooth_samples=7):
    """Suppress clicks at clip boundaries.

    Smooths large sample-to-sample jumps in the first/last check region.
    Boundary fades (ramp to/from zero) are handled separately by the
    boundary_fade_ms step in process_one — this function only does click
    suppression.

    Note: this function modifies the input array in-place for performance.
    Pass ``audio.copy()`` if you need to preserve the original.
    """
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


def process_one(audio, sr, volume_db, room_ir, processing_cfg, reverb_mix=0.02):
    """Apply the full processing chain to one audio clip.

    Processing order: trim -> normalize -> speaker volume -> reverb -> fades.
    Reverb is applied before fades so the reverb tail decays naturally
    under the fade-out rather than cutting off abruptly.

    Note: modifies ``audio`` in-place. Pass a copy if you need the original.
    """
    p = processing_cfg
    audio = trim_silence(
        audio, sr,
        threshold_db=p.get("trim_threshold_db", -35),
        pre_roll_ms=p.get("pre_roll_ms", 40),
        min_pad_ms=p.get("min_pad_ms", 25),
    )
    audio = smooth_onset_glitches(
        audio, sr,
        threshold_db=p.get("trim_threshold_db", -35),
        scan_ms=p.get("onset_scan_ms", 10),
        reversal_threshold=p.get("onset_reversal_threshold", 0.025),
        smooth_radius=p.get("onset_smooth_radius", 4),
    )
    audio = lufs_normalize(
        audio, sr,
        target_lufs=p.get("target_lufs", -20.0),
        target_rms=p.get("rms_target", 0.1),
    )
    audio = apply_speaker_volume(audio, volume_db)
    if _HAS_PEDALBOARD and p.get("use_pedalboard_reverb", True):
        audio = apply_reverb_pedalboard(audio, sr, mix=reverb_mix,
                                        room_size=p.get("reverb_decay", 0.15))
    else:
        audio = apply_reverb(audio, room_ir, mix=reverb_mix)
    # Boundary fade: S-curve (raised cosine) at head and tail.
    # Replaces the old apply_clip_fades which damaged speech transients.
    # Head fade is pad + overlap to smooth Qwen's hard onsets.
    # Tail fade is just the pad — capped to 10% of clip length
    # so short clips like "Thanks." don't lose their final consonant.
    pad_ms = p.get("min_pad_ms", 25)
    overlap_ms = p.get("boundary_overlap_ms", 15)
    fade_in = int(sr * (pad_ms + overlap_ms) / 1000)
    fade_out = int(sr * pad_ms / 1000)
    # Cap fades based on clip length — short clips get minimal fades
    # to avoid eating final consonants ("they", "Yeah")
    clip_dur = len(audio) / sr
    if clip_dur < 0.5:
        # Ultra-short clips: 5ms symmetric fade (compromise between
        # onset blip suppression and tail preservation)
        fade_in = int(sr * 0.005)
        fade_out = int(sr * 0.005)
    else:
        # Normal clips: cap tail to 5% of clip length
        fade_out = min(fade_out, len(audio) // 20)
        fade_in = max(fade_in, int(sr * 0.005))
        fade_out = max(fade_out, int(sr * 0.005))

    if len(audio) > fade_in + fade_out:
        t_in = np.linspace(0, 1, fade_in, dtype=np.float32)
        t_out = np.linspace(0, 1, fade_out, dtype=np.float32)
        scurve_in = 0.5 * (1 - np.cos(np.pi * t_in))
        scurve_out = 0.5 * (1 - np.cos(np.pi * t_out))
        audio[:fade_in] *= scurve_in
        audio[-fade_out:] *= scurve_out[::-1]

    # Peak safety cap — prevent clipping after the full chain
    # (normalization + speaker volume + reverb can push peaks above 1.0)
    peak_ceiling = p.get("peak_ceiling", 0.85)
    peak = np.max(np.abs(audio))
    if peak > peak_ceiling:
        audio = (audio * (peak_ceiling / peak)).astype(np.float32)

    return audio


# ---------------------------------------------------------------------------
# Onset quality detection
# ---------------------------------------------------------------------------


def check_onset_quality(audio, sr, threshold_db=-35,
                        attack_window_ms=5, attack_rms_max=0.08,
                        pre_speech_ms=2, pre_speech_peak_max=0.02,
                        min_silence_ms=1.0):
    """Check if a TTS clip has a clean speech onset.

    Detects two common Qwen3-TTS artifacts:
    1. Hard onset — speech energy rises too steeply (click/pop at start)
    2. Pre-speech spike — isolated energy before the main speech begins
    3. No lead-in — speech starts within the first ms (no natural ramp)

    Args:
        audio: mono float32 audio (raw TTS, before processing)
        sr: sample rate
        threshold_db: speech detection threshold
        attack_window_ms: window after speech start to measure attack slope
        attack_rms_max: max RMS in the attack window (higher = harder onset)
        pre_speech_ms: window before speech start to check for spikes
        pre_speech_peak_max: max peak in pre-speech window
        min_silence_ms: minimum silence before speech (below = flagged)

    Returns:
        dict with {clean: bool, issues: [str], metrics: {attack_rms, pre_speech_peak, speech_start_ms}}
    """
    issues = []

    speech_start, _ = _find_speech_boundaries(audio, sr, threshold_db)
    speech_start_ms = speech_start / sr * 1000

    # Speech starts too early — no natural lead-in silence
    if speech_start_ms < min_silence_ms:
        issues.append("no_silence_before_speech")

    # Attack slope: RMS in the first N ms of speech
    attack_samples = int(sr * attack_window_ms / 1000)
    attack_region = audio[speech_start:speech_start + attack_samples]
    attack_rms = float(np.sqrt(np.mean(attack_region ** 2))) if len(attack_region) > 0 else 0.0

    if attack_rms > attack_rms_max:
        issues.append(f"hard_onset(rms={attack_rms:.3f})")

    # Pre-speech spike: any energy before speech start
    pre_samples = int(sr * pre_speech_ms / 1000)
    if speech_start > pre_samples:
        pre_region = audio[speech_start - pre_samples:speech_start]
        pre_peak = float(np.max(np.abs(pre_region)))
    elif speech_start > 0:
        pre_region = audio[:speech_start]
        pre_peak = float(np.max(np.abs(pre_region))) if len(pre_region) > 0 else 0.0
    else:
        pre_peak = 0.0

    if pre_peak > pre_speech_peak_max:
        issues.append(f"pre_speech_spike(peak={pre_peak:.3f})")

    # Onset glitch: abnormal sample-to-sample reversal in first 5ms
    # of speech.  Qwen sometimes produces a micro-glitch (sudden
    # direction change) right at the start of a consonant.
    glitch_window = int(sr * 0.005)
    onset = audio[speech_start:speech_start + glitch_window]
    max_reversal = 0.0
    if len(onset) > 2:
        diffs = np.diff(onset)
        for j in range(1, len(diffs)):
            # Sign change with magnitude — a sudden reversal
            if diffs[j] * diffs[j - 1] < 0:  # opposite directions
                reversal = abs(diffs[j]) + abs(diffs[j - 1])
                max_reversal = max(max_reversal, reversal)
    if max_reversal > 0.03:
        issues.append(f"onset_glitch(reversal={max_reversal:.3f})")

    return {
        "clean": len(issues) == 0,
        "issues": issues,
        "metrics": {
            "attack_rms": round(attack_rms, 4),
            "pre_speech_peak": round(pre_peak, 4),
            "speech_start_ms": round(speech_start_ms, 1),
            "max_reversal": round(max_reversal, 4),
        },
    }


def check_all_onsets(manifest, cfg):
    """Check onset quality for all TTS lines.

    Uses thresholds from processing config so detection sensitivity
    matches the smoother settings.

    Returns:
        dict with {clean: int, flagged: int, flags: [{hash, speaker, text, issues, metrics}]}
    """
    tts_dir = cfg.tts_dir()
    target_sr = cfg.mix.get("target_sr", 24000)
    p = cfg.processing
    lines = manifest["lines"]
    flags = []
    clean = 0

    for h, info in lines.items():
        if info["status"] != STATUS_EXISTS:
            continue
        src = tts_dir / info["file"]
        if not src.exists():
            continue

        audio, file_sr = sf.read(str(src), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != target_sr:
            audio = resample(audio, int(len(audio) * target_sr / file_sr)).astype(np.float32)

        result = check_onset_quality(
            audio, target_sr,
            threshold_db=p.get("trim_threshold_db", -35),
            min_silence_ms=p.get("min_pad_ms", 25) / 5,  # flag if < 1/5 of pad
        )
        if result["clean"]:
            clean += 1
        else:
            flags.append({
                "hash": h,
                "speaker": info["speaker"],
                "text": info["text"][:60],
                "issues": result["issues"],
                "metrics": result["metrics"],
            })
            logger.warning(
                "Onset issue %s (%s): %s — %s",
                h, info["speaker"], ", ".join(result["issues"]),
                info["text"][:50],
            )

    if flags:
        logger.warning("Onset check: %d flagged, %d clean", len(flags), clean)
    else:
        logger.info("Onset check: all %d lines clean", clean)

    return {"clean": clean, "flagged": len(flags), "flags": flags}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_all(manifest, cfg):
    """Process all lines from tts_dir -> lines_dir.

    Reads raw TTS files, applies processing chain, writes to lines_dir.
    Also processes backchannel clips. Raw files in tts_dir are never modified.

    Args:
        manifest: manifest dict with lines
        cfg: EpisodeConfig instance

    Returns:
        dict with counts: {processed, skipped, total, backchannels}
    """
    tts_dir = cfg.tts_dir()
    lines_dir = cfg.lines_dir()
    lines_dir.mkdir(parents=True, exist_ok=True)

    target_sr = cfg.mix.get("target_sr", 24000)
    processing_cfg = cfg.processing
    reverb_decay = cfg.mix.get("reverb_decay", 0.15)
    reverb_mix = cfg.mix.get("reverb_mix", 0.02)

    # Generate room IR (deterministic with fixed seed for consistency)
    rng = np.random.default_rng(42)
    room_ir = generate_room_ir(target_sr, reverb_decay, rng=rng)

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

        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            skipped += 1
            continue

        audio, file_sr = sf.read(str(src), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != target_sr:
            audio = resample(audio, int(len(audio) * target_sr / file_sr)).astype(np.float32)

        # Get speaker volume from config
        speaker = info["speaker"]
        try:
            volume_db = cfg.cast(speaker).get("volume_db", 0.0)
        except KeyError:
            logger.warning("Unknown speaker '%s', using volume_db=0.0", speaker)
            volume_db = 0.0

        audio = process_one(audio, target_sr, volume_db, room_ir,
                            processing_cfg, reverb_mix=reverb_mix)

        sf.write(str(dst), audio, target_sr)
        processed += 1

        if processed % 20 == 0 and processed > 0:
            logger.info("Processed %d lines...", processed)

    logger.info("Lines: %d processed, %d skipped, %d total",
                processed, skipped, len(lines))

    # Process backchannel clips
    bc_processed = _process_backchannels(cfg, target_sr, processing_cfg)

    # Onset quality report — reads raw TTS files (tts_dir), not processed
    # output (lines_dir).  Reports artifacts in Qwen's output; the
    # smoother has already fixed these in the processed files.
    onset_report = check_all_onsets(manifest, cfg)

    return {"processed": processed, "skipped": skipped,
            "total": len(lines), "backchannels": bc_processed,
            "onset_flags": onset_report["flagged"]}


def _process_backchannels(cfg, target_sr, processing_cfg):
    """Process backchannel clips: apply fades and click suppression.

    BC clips get lighter processing than dialogue lines — no reverb or
    RMS normalization, just fades and click suppression.

    Processed clips are written to a 'processed' subdirectory inside
    backchannels_dir. Source clips are never modified.
    """
    bc_dir = cfg.backchannels_dir()
    if not bc_dir.exists():
        return 0

    out_dir = bc_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for speaker in cfg.cast_names():
        for clip_info in cfg.backchannel_clips(speaker):
            # clip_info["file"] is resolved by config — may be absolute
            src_path = Path(clip_info["file"])
            # If resolved path doesn't exist, try relative to bc_dir
            if not src_path.exists():
                src_path = bc_dir / src_path.name
            dst_path = out_dir / src_path.name

            if not src_path.exists():
                continue

            if dst_path.exists() and dst_path.stat().st_mtime >= src_path.stat().st_mtime:
                continue  # already processed and up to date

            audio, file_sr = sf.read(str(src_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if file_sr != target_sr:
                audio = resample(audio, int(len(audio) * target_sr / file_sr)).astype(np.float32)

            # BC clips get onset smoothing + micro-fade like dialogue,
            # but shorter fade (10ms) to preserve the punchy attack of
            # laughs and reactions.
            audio = smooth_onset_glitches(audio, target_sr)
            micro = int(target_sr * 0.010)  # 10 ms
            if len(audio) > 2 * micro:
                audio[:micro] *= np.linspace(0, 1, micro, dtype=np.float32)
                audio[-micro:] *= np.linspace(1, 0, micro, dtype=np.float32)

            sf.write(str(dst_path), audio, target_sr)
            processed += 1

    if processed:
        logger.info("Backchannels: %d clips processed -> %s", processed, out_dir)
    return processed
