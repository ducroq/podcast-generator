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

from manifest import STATUS_EXISTS, STATUS_FAILED, STATUS_MISSING
from process_lines import check_onset_quality

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hallucination guard
# ---------------------------------------------------------------------------


def estimate_max_duration(text, max_per_word=0.6, floor=10.0):
    """Estimate reasonable max duration for text based on word count."""
    words = len(text.split())
    return max(floor, words * max_per_word)


# ---------------------------------------------------------------------------
# Voice similarity scoring (automated "director")
# ---------------------------------------------------------------------------

# NOTE: Module-level singleton — not thread-safe. The generation pipeline
# is single-threaded; if parallelized in future, protect with threading.Lock.
_resemblyzer_encoder = None


def _score_voice_similarity(audio, sr, voice_ref_path):
    """Score how similar generated audio is to the reference voice.

    Uses resemblyzer d-vector cosine similarity. Returns float 0.0-1.0,
    or None if resemblyzer is unavailable or audio is too short.
    """
    global _resemblyzer_encoder
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError:
        return None

    # Too short for reliable embedding (~1600 samples at 16kHz)
    if len(audio) < sr * 0.1:
        return None

    try:
        if _resemblyzer_encoder is None:
            _resemblyzer_encoder = VoiceEncoder("cpu")

        wav_gen = preprocess_wav(audio, source_sr=sr)
        wav_ref = preprocess_wav(str(voice_ref_path))
        if len(wav_gen) < 1600 or len(wav_ref) < 1600:
            return None

        embed_gen = _resemblyzer_encoder.embed_utterance(wav_gen)
        embed_ref = _resemblyzer_encoder.embed_utterance(wav_ref)
        norm = np.linalg.norm(embed_gen) * np.linalg.norm(embed_ref)
        if norm < 1e-10:
            return None
        return float(np.dot(embed_gen, embed_ref) / norm)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Context-embedding for short lines
# ---------------------------------------------------------------------------

_whisper_model_cache = {}


def _get_whisper_model(model_size="base"):
    """Get or load a cached faster-whisper model for word-timestamp extraction."""
    if model_size not in _whisper_model_cache:
        try:
            from faster_whisper import WhisperModel
            device = "cuda"
            try:
                import torch
                if not torch.cuda.is_available():
                    device = "cpu"
            except ImportError:
                device = "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            _whisper_model_cache[model_size] = WhisperModel(
                model_size, device=device, compute_type=compute_type
            )
        except ImportError:
            return None
    return _whisper_model_cache.get(model_size)


def _find_order_index(manifest, current_hash):
    """Find the index of a line hash in manifest order. Returns None if not found."""
    for i, entry in enumerate(manifest.get("order", [])):
        if entry.get("type") == "line" and entry.get("hash") == current_hash:
            return i
    return None


def _find_same_speaker_context(manifest, current_hash, speaker, max_words=15):
    """Find the previous line from the same speaker (prefix context).

    Walks backwards from current_hash, skipping pauses and backchannels.
    Stops at section_break boundaries (don't cross sections).

    Returns the text (last max_words words) or None.
    """
    order = manifest.get("order", [])
    lines = manifest.get("lines", {})
    current_idx = _find_order_index(manifest, current_hash)

    if current_idx is None or current_idx == 0:
        return None

    for i in range(current_idx - 1, -1, -1):
        entry = order[i]
        if entry.get("type") == "section_break":
            return None
        if entry.get("type") == "line":
            prev_hash = entry["hash"]
            prev_info = lines.get(prev_hash, {})
            if prev_info.get("speaker", "").lower() == speaker.lower():
                text = prev_info.get("text", "")
                words = text.split()
                if len(words) > max_words:
                    words = words[-max_words:]
                return " ".join(words)

    return None


def _find_same_speaker_suffix(manifest, current_hash, speaker, max_words=15):
    """Find the next line from the same speaker (suffix context).

    Walks forward from current_hash. Stops at section_break.
    Returns the text (first max_words words) or None.
    """
    order = manifest.get("order", [])
    lines = manifest.get("lines", {})
    current_idx = _find_order_index(manifest, current_hash)

    if current_idx is None:
        return None

    for i in range(current_idx + 1, len(order)):
        entry = order[i]
        if entry.get("type") == "section_break":
            return None
        if entry.get("type") == "line":
            next_hash = entry["hash"]
            next_info = lines.get(next_hash, {})
            if next_info.get("speaker", "").lower() == speaker.lower():
                text = next_info.get("text", "")
                words = text.split()
                if len(words) > max_words:
                    words = words[:max_words]
                return " ".join(words)

    return None


def _extract_target_with_whisper(audio, sr, prefix_text, target_text, config,
                                  suffix_text=None):
    """Extract the target from a context-sandwich audio via word timestamps.

    The sandwich: [prefix] ... [TARGET] ... [suffix]
    Onset artifacts land in the prefix (cut away), tail artifacts land in
    the suffix (cut away). Only the clean middle is returned.

    Args:
        audio: combined audio array
        sr: sample rate
        prefix_text: context prepended before target (or None)
        target_text: the text we want to extract
        config: dict with min_extracted_duration, pad_before_ms, pad_after_ms
        suffix_text: context appended after target (or None)

    Returns extracted audio array, or None if extraction fails.
    """
    import tempfile

    model = _get_whisper_model(config.get("whisper_model", "base"))
    if model is None:
        return None

    min_duration = config.get("min_extracted_duration", 0.2)
    pad_before = config.get("pad_before_ms", 30) / 1000
    pad_after = config.get("pad_after_ms", 80) / 1000

    prefix_word_count = len(prefix_text.split()) if prefix_text else 0
    target_word_count = len(target_text.split())

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            sf.write(f.name, audio, sr)

        segments, _ = model.transcribe(temp_path, language="en",
                                        word_timestamps=True)

        words = []
        for segment in segments:
            if hasattr(segment, "words") and segment.words:
                for w in segment.words:
                    words.append((w.word.strip(), w.start, w.end))

        # Find where the target starts — the separator "..." may produce
        # 0-3 words in Whisper output, so we can't rely on exact word count.
        # Strategy: start at prefix_word_count, then search nearby for a word
        # that matches the first word of the target text.
        if len(words) <= prefix_word_count:
            return None

        target_first_word = target_text.split()[0].lower().strip(".,;:!?\"'")
        best_start_idx = prefix_word_count  # default

        # Search window around expected position (±3 words for separator drift)
        search_start = max(0, prefix_word_count - 1)
        search_end = min(len(words), prefix_word_count + 4)
        for i in range(search_start, search_end):
            whisper_word = words[i][0].lower().strip(".,;:!?\"'")
            if whisper_word.startswith(target_first_word[:3]) or \
               target_first_word.startswith(whisper_word[:3]):
                best_start_idx = i
                break

        target_start_word = words[best_start_idx]

        # Target ends before suffix (or at last word if no suffix)
        if suffix_text:
            target_end_idx = min(best_start_idx + target_word_count - 1,
                                 len(words) - 1)
            target_end_word = words[target_end_idx]
        else:
            target_end_word = words[-1]

        # Start: pad before first target word
        start_time = max(0, target_start_word[1] - pad_before)

        # End: depends on whether there's a suffix
        whisper_end = target_end_word[2]
        if suffix_text:
            # With suffix: use Whisper boundary + fixed pad (don't envelope-follow
            # or we'll bleed into the suffix which has continuous energy)
            suffix_start_idx = best_start_idx + target_word_count
            if suffix_start_idx < len(words):
                # Take 80% of the gap between target end and suffix start
                # (50% was too tight — cut off final consonant releases)
                suffix_start_time = words[suffix_start_idx][1]
                end_time = whisper_end + min(pad_after, (suffix_start_time - whisper_end) * 0.8)
            else:
                end_time = whisper_end + pad_after
        else:
            # No suffix: follow the audio envelope past Whisper's end timestamp
            # until energy drops to silence. Captures consonant releases
            # ("t" in "podcast", "s" in "sense").
            silence_threshold = 0.01
            scan_limit = min(whisper_end + 0.3, len(audio) / sr)
            scan_start = int(whisper_end * sr)
            scan_end = int(scan_limit * sr)
            envelope_end = whisper_end
            if scan_start < len(audio):
                window = int(sr * 0.01)  # 10ms RMS windows
                for i in range(scan_start, min(scan_end, len(audio) - window), window):
                    rms = float(np.sqrt(np.mean(audio[i:i + window] ** 2)))
                    if rms < silence_threshold:
                        envelope_end = i / sr
                        break
                else:
                    envelope_end = scan_limit
            end_time = envelope_end + 0.02  # 20ms safety pad
        end_time = min(end_time, len(audio) / sr)

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        extracted = audio[start_sample:end_sample]

        extracted_dur = len(extracted) / sr
        if extracted_dur < min_duration:
            return None

        return extracted.astype(np.float32)

    except Exception as e:
        logger.debug("Whisper extraction failed: %s", e)
        return None
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass


def _generate_with_context_embedding(adapter, text, voice_ref, ref_text,
                                      language, tts_config, retry_config,
                                      prefix_text, embed_config,
                                      suffix_text=None):
    """Generate a context-sandwich and extract the clean target via Whisper.

    Sandwich: [prefix] ... [TARGET] ... [suffix]
    - Prefix gives prosodic runway (onset artifacts land here)
    - Suffix absorbs trailing artifacts (tail cut-offs land here)
    - Whisper word timestamps locate the target boundaries
    - Only the clean middle is returned

    Returns (audio, sr) or (None, None) on failure.
    """
    separator = " ... "
    parts = []
    if prefix_text:
        parts.append(prefix_text)
    parts.append(text)
    if suffix_text:
        parts.append(suffix_text)
    combined_text = separator.join(parts)

    # Generate the combined utterance
    audio, sr, _ = _generate_with_guard(
        adapter, combined_text, voice_ref, ref_text, language,
        tts_config, retry_config,
    )

    if audio is None:
        logger.info("    context-embed: generation failed (hallucination guard)")
        return None, None

    # Extract just the target portion
    extracted = _extract_target_with_whisper(
        audio, sr, prefix_text, text, embed_config,
        suffix_text=suffix_text,
    )

    if extracted is None:
        logger.info("    context-embed: Whisper extraction failed, falling back")
        return None, None

    return extracted, sr


def _is_hallucinated(audio, sr, text, max_per_word=0.6):
    """Check if generated audio is unreasonably long or silent (hallucination)."""
    duration = len(audio) / sr
    max_dur = estimate_max_duration(text, max_per_word)
    if duration > max_dur:
        return True, duration, max_dur
    # All-silence detection: if RMS is near zero, Qwen generated nothing useful
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 0.001 and len(text.split()) > 0:
        return True, duration, max_dur
    return False, duration, max_dur


# ---------------------------------------------------------------------------
# Engine adapters
# ---------------------------------------------------------------------------

# Each adapter implements:
#   load(**kwargs)  — load the model (device is optional, ignored by API engines)
#   generate(text, voice_ref, ref_text, language, **kwargs) -> (audio, sr)
#   unload()        — release resources
#
# voice_ref is the absolute path to the reference audio file.
# Returns (numpy array float32, sample rate).


class QwenAdapter:
    """Adapter for Qwen3-TTS voice cloning."""

    name = "qwen"

    def __init__(self):
        self.model = None

    def load(self, device="cuda:0", **kwargs):
        import torch
        from qwen_tts import Qwen3TTSModel
        logger.info("Loading Qwen3-TTS...")
        self.model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map=device, dtype=torch.bfloat16,
        )
        logger.info("Qwen3-TTS loaded.")

    # Qwen3-TTS codec rate: 12 tokens per second of audio.
    CODEC_HZ = 12

    def generate(self, text, voice_ref, ref_text, language="English",
                 temperature=0.7, repetition_penalty=1.2,
                 max_per_word=0.6, **kwargs):
        import torch
        # Cap generation length to prevent runaway hallucination.
        # Much faster than waiting for full output then checking duration.
        max_dur = estimate_max_duration(text, max_per_word)
        max_tokens = int(max_dur * self.CODEC_HZ * 1.5)  # 50% headroom

        wavs, sr = self.model.generate_voice_clone(
            text=text, language=language,
            ref_audio=str(voice_ref), ref_text=ref_text,
            temperature=temperature, repetition_penalty=repetition_penalty,
            max_new_tokens=max_tokens,
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

    def load(self, device="cuda:0", **kwargs):
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
    audio is None if all retries were exhausted (hallucination).
    """
    temp = tts_config.get("temperature", 0.7)
    rep_penalty = tts_config.get("repetition_penalty", 1.2)
    max_per_word = retry_config.get("max_duration_per_word", 0.6)
    retry_count = min(retry_config.get("retry_count", 3), 10)
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
    """Generate segmented line (multiple chunks with pauses between).

    All segments are resampled to sr_target before concatenation to ensure
    consistent sample rate across the assembled line.

    Returns (audio, sr_target) or (None, sr_target) if any segment fails.
    """
    if not segments:
        return None, sr_target

    parts = []
    for seg in segments:
        text = seg["text"]
        audio, seg_sr, _ = _generate_with_guard(
            adapter, text, voice_ref, ref_text, language,
            tts_config, retry_config,
        )
        if audio is None:
            return None, sr_target
        # Resample each segment to target SR before concatenation
        audio = _resample_if_needed(audio, seg_sr, sr_target)
        parts.append(audio)
        pause = seg.get("pause_after", 0)
        if pause > 0:
            parts.append(np.zeros(int(sr_target * pause), dtype=np.float32))

    return np.concatenate(parts), sr_target


def _resample_if_needed(audio, sr, target_sr):
    """Resample audio to target sample rate if different."""
    if sr == target_sr:
        return audio
    from scipy.signal import resample
    new_len = int(len(audio) * target_sr / sr)
    return resample(audio, new_len).astype(np.float32)


def _resolve_engine(h, info, force_fb, cfg):
    """Determine which engine to use for a line.

    Checks force_fallback first, then per-line override, then cast default.
    """
    # Force fallback takes priority
    if h in force_fb:
        return force_fb[h]
    # Per-line override from config.overrides
    override = cfg.overrides.get(h, {})
    if isinstance(override, dict) and "engine" in override:
        return override["engine"]
    # Cast default
    return cfg.cast(info["speaker"])["engine"]


def _get_line_tts_config(h, cfg, base_tts_config):
    """Get TTS config for a specific line, applying per-line overrides.

    Per-line overrides (from config.overrides) can set temperature and
    other TTS parameters that override the episode-level defaults.
    """
    override = cfg.overrides.get(h, {})
    if not isinstance(override, dict):
        return base_tts_config
    # Only override keys that are present
    line_config = dict(base_tts_config)
    for key in ("temperature", "repetition_penalty"):
        if key in override:
            line_config[key] = override[key]
    return line_config


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


def _try_fallback(info, fallback_chain, cfg, language, tts_config,
                  retry_config, segments, sr_target):
    """Try fallback engines for a line that failed on the primary engine.

    Returns (audio, sr, engine_name) or (None, sr_target, None) if all
    fallbacks are exhausted.
    """
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
                    tts_config, retry_config, sr_target,
                )
            else:
                audio, sr, _ = _generate_with_guard(
                    fb_adapter, info["text"], voice_ref, ref_text, language,
                    tts_config, retry_config,
                )

            if audio is not None:
                return audio, sr, fb_engine
        except Exception as exc:
            logger.warning("  Fallback %s failed: %s", fb_engine, str(exc)[:80])
        finally:
            fb_adapter.unload()

    return None, sr_target, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_missing(manifest, cfg, dry_run=False):
    """Generate TTS for all missing lines in the manifest.

    Reads engine config from ``cfg``, generates audio files into
    ``cfg.tts_dir()``. Updates the manifest in-place with status, duration,
    and engine info.

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

    # Voice bank: dynamic reference selection across lines/episodes
    bank_cfg = tts_config.get("voice_bank", {})
    bank = None
    if bank_cfg.get("enabled", False):
        try:
            from voice_bank import VoiceBank
            bank_dir = cfg.voice_refs_dir() / bank_cfg.get("bank_dir", "dynamic")
            bank = VoiceBank(
                bank_dir,
                min_score=bank_cfg.get("min_score", 0.85),
                min_duration=bank_cfg.get("min_duration", 1.5),
            )
            logger.info("Voice bank loaded from %s", bank_dir)
        except Exception as e:
            logger.warning("Voice bank unavailable: %s", e)

    # Build force_fallback lookup: hash -> engine
    force_fb = {entry["hash"]: entry["engine"] for entry in cfg.force_fallback}

    # Collect lines to generate
    to_generate = []
    for h, info in lines.items():
        out_path = tts_dir / info["file"]

        if out_path.exists():
            existing_dur = sf.info(str(out_path)).duration
            max_dur = estimate_max_duration(info["text"], max_per_word)
            if existing_dur > max_dur:
                logger.info(
                    "%s exists but too long (%.1fs > %.1fs), will re-generate",
                    info["file"], existing_dur, max_dur,
                )
                out_path.unlink()
            else:
                info["status"] = STATUS_EXISTS
                info["duration"] = round(existing_dur, 2)
                continue

        to_generate.append(h)

    if dry_run:
        for h in to_generate:
            info = lines[h]
            engine = _resolve_engine(h, info, force_fb, cfg)
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
        engine = _resolve_engine(h, info, force_fb, cfg)
        engine_groups.setdefault(engine, []).append(h)

    generated = 0
    failed = 0
    progress = 0

    for engine_name, hashes in engine_groups.items():
        adapter = _get_adapter(engine_name)
        adapter.load()

        try:
            for h in hashes:
                progress += 1
                info = lines[h]
                speaker = info["speaker"]
                cast_info = cfg.cast(speaker)

                # Dynamic ref: use best-scoring line from bank if available
                dynamic = bank.best_ref(speaker) if bank else None
                if dynamic:
                    voice_ref, ref_text = dynamic
                else:
                    voice_ref = cfg.voice_ref_path(speaker)
                    ref_text = cast_info.get("ref_text", "")
                out_path = tts_dir / info["file"]

                # Check for overrides (segmented generation, per-line config)
                override = cfg.overrides.get(h, {})
                segments = override.get("segments") if isinstance(override, dict) else None
                line_tts_config = _get_line_tts_config(h, cfg, tts_config)

                logger.info(
                    "[%d/%d] %s: %s",
                    progress, len(to_generate),
                    speaker, info["text"][:55],
                )

                try:
                    if segments:
                        audio, sr = _generate_segmented(
                            adapter, segments, voice_ref, ref_text, language,
                            line_tts_config, retry_config, target_sr,
                        )
                    else:
                        word_count = len(info["text"].split())
                        embed_cfg = tts_config.get("context_embedding", {})
                        embed_enabled = embed_cfg.get("enabled", True)
                        max_wc = embed_cfg.get("max_word_count", 4)
                        short_attempts = min(tts_config.get("short_line_attempts", 3), 10)

                        audio, sr = None, None

                        # 1. Context-sandwich for short lines (≤max_wc words)
                        #    Prefix: same-speaker previous line (if available — voice consistency + onset)
                        #    Suffix: ALWAYS append filler "Hmm." (tail protection — Qwen
                        #    cuts short lines mid-word without it)
                        if embed_enabled and word_count <= max_wc:
                            ctx_max = embed_cfg.get("context_max_words", 15)
                            prefix = _find_same_speaker_context(
                                manifest, h, speaker, max_words=ctx_max)
                            filler = embed_cfg.get("tail_filler", "Hmm.")
                            # Always use filler, prefix is optional
                            audio, sr = _generate_with_context_embedding(
                                adapter, info["text"], voice_ref, ref_text,
                                language, line_tts_config, retry_config,
                                prefix or "", embed_cfg,
                                suffix_text=filler,
                            )
                            if audio is not None:
                                score = _score_voice_similarity(audio, sr, voice_ref)
                                logger.info("    sandwich: %.1fs (score=%.2f, prefix=%s)",
                                            len(audio) / sr, score or 0,
                                            "yes" if prefix else "no")

                        # 2. Fallback: multi-attempt with resemblyzer scoring
                        if audio is None:
                            attempts = short_attempts if word_count <= max_wc else 1
                            best_audio, best_sr, best_score = None, None, -1.0
                            for attempt_i in range(attempts):
                                a, s, _ = _generate_with_guard(
                                    adapter, info["text"], voice_ref, ref_text,
                                    language, line_tts_config, retry_config,
                                )
                                if a is not None:
                                    dur = len(a) / s
                                    max_dur = estimate_max_duration(info["text"])
                                    if dur <= max_dur:
                                        score = _score_voice_similarity(a, s, voice_ref)
                                        if score is None:
                                            score = dur / max_dur
                                        if score > best_score:
                                            best_audio, best_sr, best_score = a, s, score
                                            if attempts > 1:
                                                logger.info("    attempt %d/%d: %.1fs score=%.2f (best)",
                                                            attempt_i + 1, attempts, dur, score)
                            audio, sr = best_audio, best_sr

                    used_engine = engine_name
                    if audio is None:
                        # Primary engine failed — try fallback
                        fallback_chain = cast_info.get("fallback", [])
                        audio, sr, fb_engine = _try_fallback(
                            info, fallback_chain, cfg, language,
                            line_tts_config, retry_config, segments, target_sr,
                        )
                        if fb_engine is not None:
                            used_engine = fb_engine

                    if audio is not None:
                        audio = _resample_if_needed(audio, sr, target_sr)

                        # Onset quality check — regen if hard onset detected
                        # Skip for segmented lines (onset retry would destroy assembly)
                        onset = check_onset_quality(audio, target_sr) if not segments else {"clean": True, "issues": [], "metrics": {}}
                        onset_retries = 0
                        max_onset_retries = 2
                        while not onset["clean"] and any("hard_onset" in i for i in onset["issues"]) and onset_retries < max_onset_retries:
                            onset_retries += 1
                            retry_temp = max(0.3, line_tts_config.get("temperature", 0.7) - onset_retries * 0.15)
                            logger.warning(
                                "  HARD ONSET (rms=%.3f), retry %d/%d (temp=%.2f)",
                                onset["metrics"]["attack_rms"], onset_retries, max_onset_retries, retry_temp,
                            )
                            try:
                                audio2, sr2 = adapter.generate(
                                    info["text"], voice_ref, ref_text,
                                    language=language,
                                    temperature=retry_temp,
                                    repetition_penalty=line_tts_config.get("repetition_penalty", 1.2),
                                )
                                # Guard against hallucination in onset retry
                                hallucinated, _, _ = _is_hallucinated(
                                    audio2, sr2, info["text"], max_per_word)
                                if hallucinated:
                                    continue
                                audio2 = _resample_if_needed(audio2, sr2, target_sr)
                                onset2 = check_onset_quality(audio2, target_sr)
                                if onset2["clean"] or onset2["metrics"]["attack_rms"] < onset["metrics"]["attack_rms"]:
                                    audio = audio2
                                    onset = onset2
                            except Exception:
                                break

                        sf.write(str(out_path), audio, target_sr)
                        duration = len(audio) / target_sr
                        info["status"] = STATUS_EXISTS
                        info["duration"] = round(duration, 2)
                        info["engine"] = used_engine
                        generated += 1

                        # Score voice similarity for quality tracking
                        static_ref = cfg.voice_ref_path(speaker)
                        voice_score = _score_voice_similarity(audio, target_sr, static_ref)
                        if voice_score is not None:
                            info["voice_score"] = round(voice_score, 3)
                            # Update voice bank with this score
                            if bank and voice_score:
                                bank.update(speaker, audio, target_sr, voice_score,
                                            info["text"], cfg.episode.get("slug", "unknown"), h)

                        logger.info("  -> %s (%.1fs, %s, score=%.2f)",
                                    info["file"], duration, info["engine"],
                                    voice_score or 0)
                    else:
                        info["status"] = STATUS_FAILED
                        info["engine"] = None
                        failed += 1
                        logger.error("  FAILED: %s (all engines exhausted)", info["file"])

                except Exception as exc:
                    info["status"] = STATUS_FAILED
                    info["engine"] = None
                    failed += 1
                    logger.error("  ERROR generating %s: %s", info["file"], str(exc)[:80])
        finally:
            adapter.unload()

    skipped = len(lines) - len(to_generate)
    logger.info(
        "Done: %d generated, %d skipped, %d failed (total: %d)",
        generated, skipped, failed, len(lines),
    )

    # Save voice bank (persist best refs for future episodes)
    if bank:
        bank.save()

    return {"generated": generated, "skipped": skipped, "failed": failed, "total": len(lines)}


# ---------------------------------------------------------------------------
# Post-generation ASR validation
# ---------------------------------------------------------------------------


def validate_asr(manifest, cfg, language="en", model_size="base"):
    """Validate all generated lines by transcribing and comparing to input text.

    Must run AFTER generate_missing and AFTER the TTS model is unloaded,
    since Whisper needs its own VRAM.  Returns a dict with validation
    results and a list of hashes that should be regenerated.

    Args:
        manifest: manifest dict (lines must have status=exists and files on disk)
        cfg: EpisodeConfig instance
        language: 2-letter language code for Whisper
        model_size: Whisper model size (base, small, medium, large)

    Returns:
        {validated: int, failed_asr: int, regen_hashes: [str],
         results: [{hash, speaker, text, transcription, issues}]}
    """
    from validate_tts import transcribe, check_hallucination

    tts_dir = cfg.tts_dir()
    lines = manifest["lines"]
    validated = 0
    failed_asr = 0
    regen_hashes = []
    results = []

    total = sum(1 for info in lines.values() if info["status"] == STATUS_EXISTS)
    logger.info("ASR validation: %d lines to check (model=%s)", total, model_size)

    for i, (h, info) in enumerate(lines.items()):
        if info["status"] != STATUS_EXISTS:
            continue

        wav_path = tts_dir / info["file"]
        if not wav_path.exists():
            continue

        transcription = transcribe(str(wav_path), model_size=model_size,
                                    language=language)
        if transcription is None:
            logger.warning("  ASR failed for %s — Whisper returned no output", info["file"])
            results.append({
                "hash": h, "speaker": info["speaker"],
                "text": info["text"][:60], "transcription": None,
                "issues": ["TRANSCRIPTION_FAILED"],
            })
            failed_asr += 1
            regen_hashes.append(h)
            continue

        is_ok, issues = check_hallucination(info["text"], transcription)
        if is_ok:
            validated += 1
        else:
            failed_asr += 1
            regen_hashes.append(h)
            results.append({
                "hash": h, "speaker": info["speaker"],
                "text": info["text"][:60], "transcription": transcription[:60],
                "issues": issues,
            })
            logger.warning(
                "  [%d/%d] ASR MISMATCH %s (%s): %s",
                i + 1, total, h, info["speaker"], "; ".join(issues),
            )

        if (validated + failed_asr) % 20 == 0:
            logger.info("  ASR progress: %d/%d checked...", validated + failed_asr, total)

    checked = validated + failed_asr
    if total > 0 and checked == 0:
        logger.warning("ASR validation: %d lines expected but no WAV files found on disk!", total)
    logger.info(
        "ASR validation done: %d/%d checked, %d passed, %d failed, %d to regen",
        checked, total, validated, failed_asr, len(regen_hashes),
    )
    return {
        "validated": validated,
        "failed_asr": failed_asr,
        "regen_hashes": regen_hashes,
        "results": results,
    }
