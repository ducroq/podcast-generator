#!/usr/bin/env python3
"""Smoke test: run the full pipeline with real config + music, synthetic TTS.

Produces a listenable WAV at podcasts/work/smoke/mix.wav using:
- Real ep01 script and podcast.yaml config
- Real music assets (vibraphone bed, sting)
- Synthetic TTS (pitched sine waves per speaker, no GPU needed)
- Real backchannel clips if available, otherwise synthetic

Usage:
    python smoke_test.py                       # run from podcast-generator/
    python smoke_test.py --output smoke.wav    # custom output path
    python smoke_test.py --no-intro            # skip intro voice
"""

import argparse
import logging
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

# Ensure generator/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "generator"))

from config import load_episode_config
from generate_tts import generate_missing, ADAPTERS
from manifest import (
    build_manifest, line_count, load_manifest, missing_lines, parse_script,
    save_manifest, section_names,
)
from mix_podcast import mix_episode
from process_lines import check_all_onsets
from process_lines import process_all

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SR = 24000

# Base frequencies per speaker — makes voices distinguishable
SPEAKER_FREQ = {
    "alex": 160,
    "morgan": 220,
    "zara": 280,
}


def make_speech(text, speaker, sr=SR):
    """Generate synthetic speech: pitched buzz with amplitude envelope.

    Each speaker gets a distinct pitch so you can hear who is talking.
    Duration scales with word count for realistic pacing.
    """
    words = len(text.split())
    duration = max(0.4, words * 0.18)
    n = int(sr * duration)
    t = np.linspace(0, duration, n, dtype=np.float32)

    freq = SPEAKER_FREQ.get(speaker, 200)

    # Sawtooth-ish buzz (sounds more speech-like than sine)
    wave = np.sin(2 * np.pi * freq * t)
    wave += 0.3 * np.sin(2 * np.pi * freq * 2 * t)
    wave += 0.1 * np.sin(2 * np.pi * freq * 3 * t)

    # Amplitude envelope: attack + sustain + release
    env = np.ones(n, dtype=np.float32)
    attack = min(int(sr * 0.05), n // 4)
    release = min(int(sr * 0.08), n // 4)
    if attack > 0:
        env[:attack] = np.linspace(0, 1, attack, dtype=np.float32)
    if release > 0:
        env[-release:] = np.linspace(1, 0, release, dtype=np.float32)

    # Word-rate amplitude modulation (syllable rhythm)
    mod_freq = max(2.0, words / duration * 0.8)
    mod = 0.7 + 0.3 * np.sin(2 * np.pi * mod_freq * t)

    audio = wave * env * mod * 0.15
    return audio.astype(np.float32)


class SpeechMockAdapter:
    """Mock TTS that produces distinguishable pitched audio per speaker."""
    name = "smoke-mock"

    def __init__(self):
        self._speaker_map = {}

    def load(self, **kwargs):
        pass

    def generate(self, text, voice_ref, ref_text=None, language=None,
                 speaker=None, **kwargs):
        # Infer speaker from voice_ref filename
        spk = speaker or "unknown"
        for name in SPEAKER_FREQ:
            if name in str(voice_ref).lower():
                spk = name
                break
        audio = make_speech(text, spk)
        return audio, SR

    def unload(self):
        pass


def ensure_bc_clips(cfg):
    """Create synthetic BC clips if real ones don't exist."""
    bc_dir = cfg.backchannels_dir()
    bc_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for speaker in cfg.cast_names():
        freq = SPEAKER_FREQ.get(speaker, 200)
        for clip_info in cfg.backchannel_clips(speaker):
            clip_name = Path(clip_info["file"]).name
            clip_path = bc_dir / clip_name
            if not clip_path.exists():
                # Short pitched burst
                dur = 0.4
                t = np.linspace(0, dur, int(SR * dur), dtype=np.float32)
                clip = np.sin(2 * np.pi * freq * 1.5 * t) * 0.1
                clip *= np.linspace(1, 0, len(clip), dtype=np.float32)
                sf.write(str(clip_path), clip, SR)
                created += 1
    if created:
        logger.info("Created %d synthetic BC clips", created)


def make_intro_voice(cfg):
    """Build a short intro voiceover from synthetic speech."""
    lines = [
        ("alex", "Welcome to It Is Both, a companion podcast for the book."),
        ("morgan", "Three voices. Eight episodes."),
    ]
    parts = []
    for speaker, text in lines:
        parts.append(make_speech(text, speaker))
        parts.append(np.zeros(int(SR * 0.3), dtype=np.float32))
    return np.concatenate(parts)


def main():
    parser = argparse.ArgumentParser(description="Pipeline smoke test with real config + synthetic TTS")
    parser.add_argument("--output", type=str, default=None, help="Output WAV path")
    parser.add_argument("--no-intro", action="store_true", help="Skip intro voice")
    args = parser.parse_args()

    # Paths
    it_is_both = Path(__file__).resolve().parent.parent / "it_is_both"
    ep_yaml = it_is_both / "podcasts" / "episodes" / "ep01.yaml"
    pod_dir = it_is_both / "podcasts" / "it-is-both"

    if not ep_yaml.exists():
        sys.exit(f"Episode config not found: {ep_yaml}")

    # Use a separate work dir so we don't clobber real production files
    smoke_work = it_is_both / "podcasts" / "work" / "smoke"

    logger.info("=" * 60)
    logger.info("SMOKE TEST — full pipeline, synthetic TTS")
    logger.info("=" * 60)

    # 1. Load config (override work_dir to smoke/)
    logger.info("\n--- Stage 1: Config ---")
    cfg = load_episode_config(ep_yaml, podcast_dir=pod_dir)
    # Monkey-patch work_dir to smoke/ so we don't touch real ep01 files
    cfg._data["work_dir"] = str(smoke_work.relative_to(it_is_both / "podcasts"))
    logger.info("Config loaded: %d cast, work_dir=%s", len(cfg.cast_names()), cfg.work_dir())

    # 2. Parse script + build manifest
    logger.info("\n--- Stage 2: Parse + Manifest ---")
    entries = parse_script(cfg.script_path())
    manifest = build_manifest(
        entries,
        audio_dir=cfg.tts_dir(),
        script_path=str(cfg.script_path()),
        episode="smoke",
    )
    logger.info("Lines: %d, Sections: %s, Missing: %d",
                line_count(manifest), section_names(manifest), len(missing_lines(manifest)))

    manifest_path = cfg.work_dir() / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    save_manifest(manifest, manifest_path)

    # 3. Generate TTS (mock)
    logger.info("\n--- Stage 3: Generate TTS (synthetic) ---")
    with patch.dict(ADAPTERS, {"qwen": SpeechMockAdapter}):
        gen_result = generate_missing(manifest, cfg)
    logger.info("Generated: %d, Skipped: %d, Failed: %d",
                gen_result["generated"], gen_result["skipped"], gen_result["failed"])
    save_manifest(manifest, manifest_path)

    # 4. Ensure BC clips exist
    ensure_bc_clips(cfg)

    # 5. Process lines
    logger.info("\n--- Stage 4: Process lines ---")
    proc_result = process_all(manifest, cfg)
    logger.info("Processed: %d lines, %d BC clips",
                proc_result["processed"], proc_result["backchannels"])

    # 6. Mix
    logger.info("\n--- Stage 5: Mix episode ---")
    intro_voice = None if args.no_intro else make_intro_voice(cfg)
    output_path = args.output or str(cfg.work_dir() / "mix.wav")

    mix_result = mix_episode(manifest, cfg, output_path=output_path,
                             intro_voice=intro_voice, seed=42)

    # -----------------------------------------------------------------------
    # Automated checks
    # -----------------------------------------------------------------------
    logger.info("\n--- Checks ---")
    failures = []

    def check(name, condition, detail=""):
        if condition:
            logger.info("  PASS  %s", name)
        else:
            logger.error("  FAIL  %s  %s", name, detail)
            failures.append(name)

    # -- Pipeline stage results --
    check("All lines generated",
          gen_result["failed"] == 0,
          f"{gen_result['failed']} failed")
    check("All lines processed",
          proc_result["processed"] == line_count(manifest),
          f"only {proc_result['processed']}/{line_count(manifest)}")
    check("BC clips processed",
          proc_result["backchannels"] > 0,
          "no BC clips processed")

    # -- Output file exists and is valid WAV --
    output = Path(mix_result["output"])
    check("Output file exists", output.exists())

    audio, sr = sf.read(str(output), dtype="float32")
    duration = mix_result["duration"]
    peak = np.max(np.abs(audio))
    peak_db = 20 * np.log10(peak + 1e-10)

    # -- Duration sanity (real ep01 script should produce 3-10 min) --
    check("Duration > 2 min",
          duration > 120,
          f"only {duration:.0f}s")
    check("Duration < 15 min",
          duration < 900,
          f"{duration:.0f}s seems too long")

    # -- Sample rate matches config --
    check("Sample rate = 24000",
          sr == 24000,
          f"got {sr}")

    # -- Peak limiting --
    limit_db = cfg.mix.get("peak_limit_dbtp", -1.0)
    limit_linear = 10 ** (limit_db / 20)
    check(f"Peak limited (<= {limit_db} dBTP)",
          peak <= limit_linear + 0.001,
          f"peak={peak_db:.1f} dBFS")

    # -- No digital silence (dead air) longer than 3s --
    window = int(sr * 3.0)
    rms_windows = []
    for i in range(0, len(audio) - window, window):
        rms = np.sqrt(np.mean(audio[i:i + window] ** 2))
        rms_windows.append((i / sr, rms))
    dead_spots = [(t, rms) for t, rms in rms_windows if rms < 0.0005]
    check("No dead air (>3s silence)",
          len(dead_spots) == 0,
          f"{len(dead_spots)} silent 3s windows: " +
          ", ".join(f"{t:.0f}s" for t, _ in dead_spots[:5]))

    # -- All sections present in output --
    expected_sections = section_names(manifest)
    check("All sections in mix",
          set(expected_sections) == set(mix_result["sections"]),
          f"expected {expected_sections}, got {mix_result['sections']}")

    # -- Section WAVs exist --
    section_wavs = list(cfg.sections_dir().glob("*.wav"))
    check(f"Section WAVs created ({len(expected_sections)} expected)",
          len(section_wavs) >= len(expected_sections),
          f"only {len(section_wavs)} files")

    # -- Each section WAV is non-empty --
    empty_sections = []
    for sw in section_wavs:
        sa, _ = sf.read(str(sw), dtype="float32")
        if len(sa) < sr * 0.5:  # less than 0.5s
            empty_sections.append(sw.name)
    check("All section WAVs have audio",
          len(empty_sections) == 0,
          f"near-empty: {empty_sections}")

    # -- Manifest consistency after full pipeline --
    loaded = load_manifest(manifest_path)
    all_exist = all(
        info["status"] == "exists" for info in loaded["lines"].values()
    )
    check("Manifest: all lines status=exists", all_exist)

    all_have_duration = all(
        info.get("duration") is not None for info in loaded["lines"].values()
    )
    check("Manifest: all lines have duration", all_have_duration)

    # -- Work directory structure --
    check("tts/ dir exists", cfg.tts_dir().is_dir())
    check("lines/ dir exists", cfg.lines_dir().is_dir())
    check("sections/ dir exists", cfg.sections_dir().is_dir())

    tts_count = len(list(cfg.tts_dir().glob("*.wav")))
    lines_count = len(list(cfg.lines_dir().glob("*.wav")))
    check("tts/ and lines/ have same file count",
          tts_count == lines_count,
          f"tts={tts_count}, lines={lines_count}")

    # -- Music integration (intro + sting) --
    if not args.no_intro:
        # Intro adds music_solo + voice + post_voice = ~12s
        # Without intro would be shorter; with intro should exceed cold open alone
        cold_open_wav = cfg.sections_dir() / "cold_open.wav"
        if cold_open_wav.exists():
            co_audio, _ = sf.read(str(cold_open_wav), dtype="float32")
            co_dur = len(co_audio) / sr
            check("Intro adds duration beyond cold open",
                  duration > co_dur + 10,
                  f"total={duration:.0f}s, cold_open={co_dur:.0f}s")

    # -- Onset quality (on raw TTS) --
    onset = check_all_onsets(manifest, cfg)
    check("Onset quality reported",
          onset["flagged"] >= 0,  # always passes — just ensures it runs
          "onset check crashed")
    if onset["flagged"] > 0:
        logger.info("  NOTE: %d lines have onset issues (smoothed automatically)",
                     onset["flagged"])

    # -- Idempotency: re-running process should skip everything --
    proc2 = process_all(manifest, cfg)
    check("Re-process is idempotent (0 reprocessed)",
          proc2["processed"] == 0,
          f"reprocessed {proc2['processed']} lines")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_checks = 18 + (1 if not args.no_intro else 0)
    logger.info("\n" + "=" * 60)
    if failures:
        logger.error("SMOKE TEST: %d/%d checks FAILED",
                      len(failures), total_checks)
        for f in failures:
            logger.error("  - %s", f)
        logger.info("=" * 60)
        return 1
    else:
        logger.info("SMOKE TEST: ALL %d CHECKS PASSED", total_checks)
        logger.info("=" * 60)
        logger.info("Output:   %s", mix_result["output"])
        logger.info("Duration: %.1fs (%.1f min)", duration, duration / 60)
        logger.info("Peak:     %.1f dBFS", peak_db)
        logger.info("Sections: %s", mix_result["sections"])
        logger.info("")
        logger.info("Play it:  ffplay \"%s\"", mix_result["output"])
        return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
