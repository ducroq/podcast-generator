"""Staged TTS generation for podcast pipeline.

Reads an episode config and manifest, generates audio for missing lines
using a configurable engine with automatic fallback chain. Content-addressed:
files are named by text hash, not line index.

This module is designed to run on the gpu-server via SSH. Each engine
adapter handles model loading and generation independently.

Usage (on gpu-server):
    from generate_tts import generate_missing
    from config import load_episode_config
    from manifest import parse_script, build_manifest, save_manifest

    cfg = load_episode_config("episodes/ep01.yaml")
    entries = parse_script(cfg.script_path())
    manifest = build_manifest(entries, audio_dir=cfg.tts_dir())
    results = generate_missing(manifest, cfg)
    save_manifest(manifest, cfg.work_dir() / "manifest.json")
"""

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hallucination guard
# ---------------------------------------------------------------------------


def estimate_max_duration(text, max_per_word=0.6, floor=10.0):
    """Estimate reasonable max duration for text based on word count."""
    words = len(text.split())
    return max(floor, words * max_per_word)


def _is_hallucinated(audio, sr, text, max_per_word=0.6):
    """Check if generated audio is unreasonably long (hallucination)."""
    duration = len(audio) / sr
    max_dur = estimate_max_duration(text, max_per_word)
    return duration > max_dur, duration, max_dur


# ---------------------------------------------------------------------------
# Engine adapters
# ---------------------------------------------------------------------------

# Each adapter has the signature:
#   load_engine(device) -> model
#   generate_line(model, text, voice_ref, ref_text, language, **kwargs) -> (audio, sr)
#
# voice_ref is the absolute path to the reference audio file.
# Returns (numpy array float32, sample rate).


class QwenAdapter:
    """Adapter for Qwen3-TTS voice cloning."""

    name = "qwen"

    def __init__(self):
        self.model = None

    def load(self, device="cuda:0"):
        import torch
        from qwen_tts import Qwen3TTSModel
        logger.info("Loading Qwen3-TTS...")
        self.model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map=device, dtype=torch.bfloat16,
        )
        logger.info("Qwen3-TTS loaded.")

    def generate(self, text, voice_ref, ref_text, language="English",
                 temperature=0.7, repetition_penalty=1.2, **kwargs):
        import torch
        wavs, sr = self.model.generate_voice_clone(
            text=text, language=language,
            ref_audio=str(voice_ref), ref_text=ref_text,
            temperature=temperature, repetition_penalty=repetition_penalty,
        )
        audio = wavs[0].copy()
        del wavs
        torch.cuda.empty_cache()
        return audio, sr

    def unload(self):
        if self.model is not None:
            import torch
            del self.model
            self.model = None
            torch.cuda.empty_cache()


class ChatterboxAdapter:
    """Adapter for Chatterbox TTS voice cloning."""

    name = "chatterbox"

    def __init__(self):
        self.model = None
        self.sr = None

    def load(self, device="cuda:0"):
        from chatterbox.tts import ChatterboxTTS
        logger.info("Loading Chatterbox...")
        self.model = ChatterboxTTS.from_pretrained(device=device)
        self.sr = self.model.sr
        logger.info("Chatterbox loaded.")

    def generate(self, text, voice_ref, ref_text=None, language=None,
                 temperature=0.6, exaggeration=0.3, cfg_weight=0.7, **kwargs):
        wav = self.model.generate(
            text, audio_prompt_path=str(voice_ref),
            temperature=temperature, exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
        audio = wav.squeeze().cpu().numpy()
        return audio, self.sr

    def unload(self):
        if self.model is not None:
            import torch
            del self.model
            self.model = None
            torch.cuda.empty_cache()


ADAPTERS = {
    "qwen": QwenAdapter,
    "chatterbox": ChatterboxAdapter,
}


def _get_adapter(engine_name):
    """Get an engine adapter by name. Raises ValueError for unknown engines."""
    cls = ADAPTERS.get(engine_name)
    if cls is None:
        raise ValueError(f"Unknown TTS engine '{engine_name}'. Available: {list(ADAPTERS.keys())}")
    return cls()


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------


def _generate_with_guard(adapter, text, voice_ref, ref_text, language,
                         tts_config, retry_config):
    """Generate TTS with hallucination guard and retry logic.

    Returns (audio, sr, attempts) where attempts is the number of tries.
    """
    temp = tts_config.get("temperature", 0.7)
    rep_penalty = tts_config.get("repetition_penalty", 1.2)
    max_per_word = retry_config.get("max_duration_per_word", 0.6)
    retry_count = retry_config.get("retry_count", 3)
    retry_temp = retry_config.get("retry_temperature", 0.4)
    retry_rep = retry_config.get("retry_repetition_penalty", 1.3)

    audio, sr = adapter.generate(
        text, voice_ref, ref_text, language,
        temperature=temp, repetition_penalty=rep_penalty,
    )

    hallucinated, duration, max_dur = _is_hallucinated(audio, sr, text, max_per_word)
    if not hallucinated:
        return audio, sr, 1

    for attempt in range(retry_count):
        t = max(0.2, retry_temp - (attempt * 0.05))
        logger.warning(
            "HALLUCINATION (%.1fs > %.1fs), retry %d/%d (temp=%.2f)",
            duration, max_dur, attempt + 1, retry_count, t,
        )
        audio, sr = adapter.generate(
            text, voice_ref, ref_text, language,
            temperature=t, repetition_penalty=retry_rep,
        )
        hallucinated, duration, max_dur = _is_hallucinated(audio, sr, text, max_per_word)
        if not hallucinated:
            return audio, sr, attempt + 2

    logger.warning(
        "Still hallucinated after %d retries (%.1fs), marking as failed",
        retry_count, duration,
    )
    return None, sr, retry_count + 1


def _generate_segmented(adapter, segments, voice_ref, ref_text, language,
                        tts_config, retry_config, sr_target=24000):
    """Generate segmented line (multiple chunks with pauses between)."""
    parts = []
    sr = sr_target
    for seg in segments:
        text = seg["text"]
        audio, seg_sr, _ = _generate_with_guard(
            adapter, text, voice_ref, ref_text, language,
            tts_config, retry_config,
        )
        if audio is None:
            return None, sr
        sr = seg_sr
        parts.append(audio)
        pause = seg.get("pause_after", 0)
        if pause > 0:
            parts.append(np.zeros(int(sr * pause), dtype=np.float32))
    return np.concatenate(parts) if parts else None, sr


def _resample_if_needed(audio, sr, target_sr):
    """Resample audio to target sample rate if different."""
    if sr == target_sr:
        return audio
    from scipy.signal import resample
    new_len = int(len(audio) * target_sr / sr)
    return resample(audio, new_len).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_missing(manifest, cfg, dry_run=False):
    """Generate TTS for all missing lines in the manifest.

    Reads engine config from `cfg`, generates audio files into `cfg.tts_dir()`.
    Updates the manifest in-place with status, duration, and engine info.

    Args:
        manifest: manifest dict from build_manifest()
        cfg: EpisodeConfig instance
        dry_run: if True, report what would be generated without calling TTS

    Returns:
        dict with counts: {generated, skipped, failed, total}
    """
    lines = manifest["lines"]
    tts_dir = cfg.tts_dir()
    tts_dir.mkdir(parents=True, exist_ok=True)

    target_sr = cfg.mix.get("target_sr", 24000)
    language = cfg.tts.get("language", "English")
    tts_config = cfg.tts
    retry_config = cfg.tts.get("hallucination", {})
    max_per_word = retry_config.get("max_duration_per_word", 0.6)

    # Build force_fallback lookup: hash → engine
    force_fb = {entry["hash"]: entry["engine"] for entry in cfg.force_fallback}

    # Collect lines to generate
    to_generate = []
    for h, info in lines.items():
        out_path = tts_dir / info["file"]

        if out_path.exists():
            # Validate existing file duration
            existing_dur = sf.info(str(out_path)).duration
            max_dur = estimate_max_duration(info["text"], max_per_word)
            if existing_dur > max_dur:
                logger.info(
                    "%s exists but too long (%.1fs > %.1fs), will re-generate",
                    info["file"], existing_dur, max_dur,
                )
                out_path.unlink()
            else:
                info["status"] = "exists"
                info["duration"] = round(existing_dur, 2)
                continue

        to_generate.append(h)

    if dry_run:
        for h in to_generate:
            info = lines[h]
            engine = force_fb.get(h, cfg.cast(info["speaker"])["engine"])
            logger.info(
                "DRY RUN: would generate %s (%s, engine=%s): %s",
                info["file"], info["speaker"], engine, info["text"][:50],
            )
        return {"generated": 0, "skipped": len(lines) - len(to_generate),
                "failed": 0, "total": len(lines), "dry_run": True}

    if not to_generate:
        logger.info("All %d lines already exist.", len(lines))
        return {"generated": 0, "skipped": len(lines), "failed": 0, "total": len(lines)}

    logger.info("Generating %d / %d lines...", len(to_generate), len(lines))

    # Group by engine to minimize model loading
    engine_groups = {}
    for h in to_generate:
        info = lines[h]
        speaker = info["speaker"]
        engine = force_fb.get(h, cfg.cast(speaker)["engine"])
        engine_groups.setdefault(engine, []).append(h)

    generated = 0
    failed = 0

    for engine_name, hashes in engine_groups.items():
        adapter = _get_adapter(engine_name)
        adapter.load()

        try:
            for h in hashes:
                info = lines[h]
                speaker = info["speaker"]
                cast_info = cfg.cast(speaker)
                voice_ref = cfg.voice_ref_path(speaker)
                ref_text = cast_info.get("ref_text", "")
                out_path = tts_dir / info["file"]

                # Check for overrides (segmented generation)
                override = cfg.overrides.get(h, {})
                segments = override.get("segments") if isinstance(override, dict) else None

                logger.info(
                    "[%d/%d] %s: %s",
                    generated + failed + 1, len(to_generate),
                    speaker, info["text"][:55],
                )

                try:
                    if segments:
                        audio, sr = _generate_segmented(
                            adapter, segments, voice_ref, ref_text, language,
                            tts_config, retry_config, target_sr,
                        )
                    else:
                        audio, sr, _ = _generate_with_guard(
                            adapter, info["text"], voice_ref, ref_text, language,
                            tts_config, retry_config,
                        )

                    if audio is None:
                        # Primary engine failed — try fallback
                        fallback_chain = cast_info.get("fallback", [])
                        audio = _try_fallback(
                            h, info, fallback_chain, cfg, language,
                            tts_config, retry_config, segments,
                        )

                    if audio is not None:
                        audio = _resample_if_needed(audio, sr, target_sr)
                        sf.write(str(out_path), audio, target_sr)
                        duration = len(audio) / target_sr
                        info["status"] = "exists"
                        info["duration"] = round(duration, 2)
                        # Engine is set by _try_fallback if fallback was used,
                        # otherwise use the primary engine
                        if info.get("engine") is None:
                            info["engine"] = engine_name
                        generated += 1
                        logger.info("  -> %s (%.1fs, %s)", info["file"], duration, info["engine"])
                    else:
                        info["status"] = "failed"
                        info["engine"] = None
                        failed += 1
                        logger.error("  FAILED: %s (all engines exhausted)", info["file"])

                except Exception as exc:
                    info["status"] = "failed"
                    failed += 1
                    logger.error("  ERROR generating %s: %s", info["file"], str(exc)[:80])
        finally:
            adapter.unload()

    skipped = len(lines) - len(to_generate)
    logger.info(
        "Done: %d generated, %d skipped, %d failed (total: %d)",
        generated, skipped, failed, len(lines),
    )
    return {"generated": generated, "skipped": skipped, "failed": failed, "total": len(lines)}


def _try_fallback(h, info, fallback_chain, cfg, language, tts_config, retry_config, segments):
    """Try fallback engines for a line that failed on the primary engine."""
    for fb_engine in fallback_chain:
        logger.info("  Trying fallback engine: %s", fb_engine)
        fb_adapter = _get_adapter(fb_engine)
        fb_adapter.load()
        try:
            speaker = info["speaker"]
            voice_ref = cfg.voice_ref_path(speaker)
            ref_text = cfg.cast(speaker).get("ref_text", "")

            if segments:
                audio, sr = _generate_segmented(
                    fb_adapter, segments, voice_ref, ref_text, language,
                    tts_config, retry_config,
                )
            else:
                audio, sr, _ = _generate_with_guard(
                    fb_adapter, info["text"], voice_ref, ref_text, language,
                    tts_config, retry_config,
                )

            if audio is not None:
                info["engine"] = fb_engine
                return audio
        except Exception as exc:
            logger.warning("  Fallback %s failed: %s", fb_engine, str(exc)[:80])
        finally:
            fb_adapter.unload()

    return None
