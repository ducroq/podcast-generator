"""Full pipeline for It Is Both Episode 01: The Nodding.

Generates TTS with Qwen3-TTS, then uses podcast-generator tools for
clean, validate, preprocess, assemble intro, mix, and master.

Designed for gpu-server. Requires:
    - qwen-tts, torch, numpy, soundfile, scipy (TTS + mix)
    - faster-whisper (validation)
    - pedalboard, pyloudnorm (mastering)

Usage (on gpu-server):
    source ~/podcast-generator/vox-env/bin/activate
    python3 ~/podcast-generator/podcasts/it-is-both/pipeline_ep01.py \\
        ~/ep01_script.txt \\
        --intro-lines ~/ep01_intro_lines.txt \\
        --overrides ~/ep01_tts_overrides.json \\
        --music-bed ~/music_bed.mp3 \\
        --sting ~/sting.mp3 \\
        -o ~/ep01_final.wav

    # Re-run from mix step (skip TTS generation):
    python3 ~/podcast-generator/podcasts/it-is-both/pipeline_ep01.py \\
        ~/ep01_script.txt \\
        --intro-lines ~/ep01_intro_lines.txt \\
        --music-bed ~/music_bed.mp3 \\
        --sting ~/sting.mp3 \\
        --skip-tts -o ~/ep01_final.wav
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, lfilter

# ---------------------------------------------------------------------------
# Paths — resolve podcast-generator tools relative to this file
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GENERATOR = REPO_ROOT / "generator"

# ---------------------------------------------------------------------------
# Voice references (gpu-server paths)
# ---------------------------------------------------------------------------

VOICE_REFS = {
    "alex": {
        "ref": Path.home() / "voice_refs" / "alex_qwen_ref.mp3",
        "text": (
            "At some point, someone hands you a book. Good to Great. "
            "Lean Startup. Atomic Habits. It comes recommended with "
            "reverence, like scripture that happens to have a forward "
            "by a former CEO."
        ),
    },
    "morgan": {
        "ref": Path.home() / "voice_refs" / "lisa_qwen_ref.mp3",
        "text": (
            "Welcome back to another episode on Machine Learning for "
            "Engineers. Today, we are diving deep into AI explainability, "
            "a topic that is becoming critical for anyone building ML "
            "systems."
        ),
    },
    "zara": {
        "ref": Path.home() / "voice_refs" / "zara_qwen_ref.mp3",
        "text": (
            "Okay, so here is the thing nobody tells you about your first "
            "job search. That generic CV, you are blasting to 50 companies, "
            "it is not working. And that is not because there is something "
            "wrong with you. It is because the system is designed to filter "
            "you out."
        ),
    },
}

# Cold open characters map to host voices
VOICE_REFS["junior manager"] = VOICE_REFS["alex"]
VOICE_REFS["team member 1"] = VOICE_REFS["morgan"]
VOICE_REFS["team member 2"] = VOICE_REFS["zara"]

# Backchannel phrases per speaker
BC_PHRASES = {
    "morgan": ["Mmhm.", "Right.", "Right, right.", "Yeah.", "Mm."],
    "zara": ["Yeah.", "Huh.", "Mm.", "Right.", "Yeah, yeah."],
    "alex": ["Yeah.", "Mmhm.", "Right.", "Mm.", "Huh."],
}

# ---------------------------------------------------------------------------
# Mix constants (from production feedback)
# ---------------------------------------------------------------------------

TARGET_SR = 24000

# Pause durations (seconds)
SECTION_PAUSE = 1.0
SPEAKER_CHANGE_PAUSE = 0.15
SAME_SPEAKER_PAUSE = 0.08
BEAT_PAUSE = 0.5

# Sting placement
COLD_OPEN_OVERLAP = 3.0
STING_FADE_IN = 0.5
STING_CROSSFADE = 2.5

# Music bed
MUSIC_FADE_IN = 4.0
MUSIC_DUCK_VOL = 0.12
MUSIC_FULL_VOL = 0.35
MUSIC_POST_VOICE = 5.0
MUSIC_BLEED_INTO_COLD = 8.0

# Per-speaker volume adjustments (dB)
SPEAKER_VOLUME_DB = {
    "alex": 0.0,
    "morgan": 0.0,
    "zara": 2.5,
    "junior manager": 0.0,
    "team member 1": 0.0,
    "team member 2": 0.0,
}

# Room tone
ROOM_TONE_LEVEL = 0.002

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_lines(path):
    """Parse speaker lines from script. Returns list of {index, speaker, text}."""
    entries = []
    line_index = 0
    for raw in open(path, encoding="utf-8"):
        stripped = raw.strip()
        if not stripped or stripped.startswith("="):
            continue
        if stripped == stripped.upper() and len(stripped) > 3:
            continue
        if re.match(r"^\[.*\]$", stripped):
            continue
        match = re.match(r"(.+?):\s*(?:\[([^\]]*)\]\s*)?(.*)", stripped)
        if match:
            speaker = match.group(1).strip().lower()
            text = match.group(3).strip()
            if text:
                entries.append({"index": line_index, "speaker": speaker, "text": text})
                line_index += 1
    return entries


def parse_intro_lines(path):
    """Parse intro lines (simple speaker: text format)."""
    entries = []
    for i, raw in enumerate(open(path, encoding="utf-8")):
        stripped = raw.strip()
        if not stripped:
            continue
        match = re.match(r"(.+?):\s*(?:\[([^\]]*)\]\s*)?(.*)", stripped)
        if match:
            speaker = match.group(1).strip().lower()
            text = match.group(3).strip()
            if text:
                entries.append({"index": i, "speaker": speaker, "text": text})
    return entries


def parse_script_with_sections(path):
    """Parse script preserving section structure for mixing."""
    entries = []
    line_index = 0
    current_section = None
    for raw in open(path, encoding="utf-8"):
        stripped = raw.strip()
        if not stripped or stripped.startswith("=" * 10):
            continue
        if re.match(r"^[A-Z][A-Z\s\':,\-]+$", stripped):
            if current_section is not None:
                entries.append({"type": "section_break", "from": current_section, "to": stripped})
            current_section = stripped
            continue
        if re.match(r"^\[.*\]$", stripped):
            lower = stripped.lower()
            if any(w in lower for w in ["pause", "silence", "beat"]):
                if "two second" in lower or "three second" in lower:
                    entries.append({"type": "pause", "duration": 2.5})
                elif "long" in lower:
                    entries.append({"type": "pause", "duration": 1.5})
                else:
                    entries.append({"type": "pause", "duration": BEAT_PAUSE})
            continue
        match = re.match(r"(.+?):\s*(?:\[([^\]]*)\]\s*)?(.*)", stripped)
        if match:
            speaker = match.group(1).strip().lower()
            text = match.group(3).strip()
            if text:
                filename = f"{line_index:03d}_{speaker.replace(' ', '_')}.wav"
                entries.append({
                    "type": "line", "index": line_index,
                    "speaker": speaker, "text": text,
                    "file": filename, "section": current_section,
                })
                line_index += 1
    return entries


# ---------------------------------------------------------------------------
# Step 1: TTS generation (Qwen3-TTS)
# ---------------------------------------------------------------------------


def generate_tts(lines, output_dir, overrides=None, prefix="", retry=False):
    """Generate TTS for all lines using Qwen3-TTS.

    If retry=True, skips lines that already have WAV files on disk.
    Returns (manifest_entries, failed_entries).
    """
    import torch
    from qwen_tts import Qwen3TTSModel

    print("Loading Qwen3-TTS...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    print("Model loaded.")

    manifest = []
    failed = []

    for i, line in enumerate(lines):
        speaker = line["speaker"]
        text = line["text"]
        idx = line["index"]

        voice = VOICE_REFS.get(speaker)
        if not voice:
            print(f"  WARNING: No voice ref for '{speaker}', skipping")
            continue

        filename = f"{prefix}{idx:03d}_{speaker.replace(' ', '_')}.wav"
        out_path = output_dir / filename

        # Skip if already generated (for retry runs)
        if out_path.exists() and not retry:
            manifest.append({
                "index": idx, "speaker": speaker, "text": text,
                "file": filename, "engine": "qwen",
                "duration": sf.info(str(out_path)).duration,
            })
            continue

        # Check for overrides
        override_key = f"{idx:03d}"
        segments = None
        if overrides and override_key in overrides:
            ov = overrides[override_key]
            if isinstance(ov, str):
                text = ov
            elif isinstance(ov, list):
                segments = ov

        # Periodic VRAM cleanup to prevent OOM on long runs
        if i > 0 and i % 10 == 0:
            torch.cuda.empty_cache()

        print(f"  [{i + 1}/{len(lines)}] {speaker}: {text[:55]}...")

        try:
            if segments:
                parts = []
                for seg in segments:
                    wavs, sr = model.generate_voice_clone(
                        text=seg["text"],
                        language="English",
                        ref_audio=str(voice["ref"]),
                        ref_text=voice["text"],
                        temperature=0.7,
                        repetition_penalty=1.2,
                    )
                    parts.append(wavs[0].copy())
                    del wavs
                    torch.cuda.empty_cache()
                    pause = seg.get("pause_after", 0)
                    if pause > 0:
                        parts.append(np.zeros(int(sr * pause), dtype=np.float32))
                audio = np.concatenate(parts)
            else:
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language="English",
                    ref_audio=str(voice["ref"]),
                    ref_text=voice["text"],
                    temperature=0.7,
                    repetition_penalty=1.2,
                )
                audio = wavs[0].copy()
                del wavs
                torch.cuda.empty_cache()

            sf.write(str(out_path), audio, sr)
            duration = len(audio) / sr
            manifest.append({
                "index": idx, "speaker": speaker, "text": text,
                "file": filename, "engine": "qwen", "duration": duration,
            })
            print(f"    -> {filename} ({duration:.1f}s)")

        except Exception as e:
            err_msg = str(e)[:80]
            print(f"    ERROR: {err_msg}")
            failed.append({"index": idx, "speaker": speaker, "text": text, "error": err_msg})

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return manifest, failed


def generate_backchannels_tts(output_dir):
    """Generate backchannel clips using Qwen3-TTS."""
    import torch
    from qwen_tts import Qwen3TTSModel

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Qwen3-TTS for backchannels...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )

    for speaker, phrases in BC_PHRASES.items():
        voice = VOICE_REFS[speaker]
        print(f"  {speaker}:")
        for i, phrase in enumerate(phrases):
            print(f"    {phrase}...", end=" ")
            wavs, sr = model.generate_voice_clone(
                text=phrase,
                language="English",
                ref_audio=str(voice["ref"]),
                ref_text=voice["text"],
                temperature=0.5,
                repetition_penalty=1.2,
            )
            audio = wavs[0].copy()
            del wavs
            torch.cuda.empty_cache()

            # Apply 8ms fades
            fade_samples = int(sr * 0.008)
            if len(audio) > fade_samples * 2:
                audio[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples)
                audio[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)

            filename = f"bc_{speaker}_{i:02d}.wav"
            sf.write(str(output_dir / filename), audio, sr)
            print(f"{len(audio) / sr:.1f}s")

    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Step 2-4: Post-processing (delegates to podcast-generator tools)
# ---------------------------------------------------------------------------


def run_tool(script_name, args, check=True):
    """Run a podcast-generator tool as a subprocess."""
    cmd = [sys.executable, str(GENERATOR / script_name)] + [str(a) for a in args]
    print(f"  $ {script_name} {' '.join(str(a) for a in args)}")
    result = subprocess.run(cmd, capture_output=False, timeout=600)
    if check and result.returncode != 0:
        print(f"  WARNING: {script_name} exited with code {result.returncode}")
    return result.returncode


def clean_audio(audio_dir):
    """Clean TTS output: trim silence, fades, edge-only click repair.

    Only repairs clicks in the first/last 50ms — Qwen has glitches at
    clip boundaries but the full-file click repair destroys speech
    transients (consonant plosives register as false positives).
    """
    print("\n=== Step: Clean audio (edge-only) ===")
    audio_dir = Path(audio_dir)
    for wav_path in sorted(audio_dir.glob("*.wav")):
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Trim silence
        threshold = 10 ** (-35 / 20)
        window = max(1, int(sr * 0.01))
        # Leading
        for i in range(0, len(audio) - window, window):
            if np.max(np.abs(audio[i:i + window])) > threshold:
                audio = audio[max(0, i - int(sr * 0.005)):]
                break
        # Trailing
        for i in range(len(audio) - window, 0, -window):
            if np.max(np.abs(audio[i:i + window])) > threshold:
                audio = audio[:min(len(audio), i + window + int(sr * 0.02))]
                break

        # Edge-only click repair (first/last 50ms)
        edge_samples = int(sr * 0.05)
        click_threshold = 0.15
        for region in [audio[:edge_samples], audio[-edge_samples:]]:
            for i in range(1, len(region)):
                if abs(region[i] - region[i - 1]) > click_threshold:
                    start = max(0, i - 3)
                    end = min(len(region), i + 4)
                    region[start:end] = np.linspace(
                        region[start], region[end - 1], end - start,
                        dtype=np.float32,
                    )

        # Fades (8ms)
        fade = int(sr * 0.008)
        if len(audio) > fade * 2:
            audio[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

        sf.write(str(wav_path), audio, sr)
    print(f"  Cleaned {len(list(audio_dir.glob('*.wav')))} files (edge-only click repair)")


def validate_tts(audio_dir, manifest_path):
    """Validate TTS output against expected text via ASR."""
    print("\n=== Step: Validate TTS ===")
    return run_tool("validate_tts.py", [
        str(audio_dir), "--manifest", str(manifest_path),
        "--language", "en", "--engine", "qwen",
    ])


def preprocess_audio(audio_dir, manifest_path):
    """Apply room reverb, per-speaker volume, RMS normalize."""
    print("\n=== Step: Preprocess audio ===")
    speaker_vol = json.dumps({k: v for k, v in SPEAKER_VOLUME_DB.items() if v != 0.0})
    return run_tool("mix_preprocess.py", [
        str(audio_dir), "--manifest", str(manifest_path),
        "--reverb-decay", "0.15", "--reverb-mix", "0.02",
        "--speaker-volume", speaker_vol, "--target-rms", "0.1",
    ])


def assemble_intro(intro_dir, intro_lines_path, output_path):
    """Assemble intro line WAVs into single voiceover track."""
    print("\n=== Step: Assemble intro ===")
    speaker_pauses = json.dumps({"morgan": 0.4, "zara": 0.4})
    return run_tool("assemble_intro.py", [
        str(intro_dir), "--lines", str(intro_lines_path),
        "-o", str(output_path),
        "--default-pause", "0.15", "--speaker-pauses", speaker_pauses,
    ])


def master_audio(input_path, output_path):
    """Master: peak limit to -1 dBTP only. No EQ, no compression, no LUFS targeting.

    Preserves dynamics — podcast platforms normalize loudness anyway.
    """
    print("\n=== Step: Master (peak limit only) ===")
    result = subprocess.run([
        "ffmpeg", "-y", "-i", str(input_path),
        "-af", "alimiter=limit=0.891:attack=5:release=50:level=disabled",
        "-ar", "24000", str(output_path),
    ], capture_output=True, timeout=120)
    if result.returncode != 0:
        print(f"  WARNING: ffmpeg limiter exited with code {result.returncode}")
    else:
        print(f"  -> {output_path}")
    return result.returncode


# ---------------------------------------------------------------------------
# Step 5: Custom mix (episode-specific structure)
# ---------------------------------------------------------------------------


def load_audio_file(path, target_sr):
    """Load any audio file, convert to mono float32 at target sample rate."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", str(path),
            "-ar", str(target_sr), "-ac", "1", "-f", "wav", tmp.name,
        ], capture_output=True, timeout=60)
        audio, _ = sf.read(tmp.name, dtype="float32")
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    return audio


def generate_pink_noise(num_samples, rng):
    """Generate pink noise for room tone."""
    white = rng.standard_normal(num_samples).astype(np.float32)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004709510]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
    pink = lfilter(b, a, white).astype(np.float32)
    peak = np.max(np.abs(pink))
    if peak > 0:
        pink /= peak
    return pink


def load_line_audio(wav_path, sr):
    """Load a preprocessed line WAV (already cleaned, normalized, reverbed)."""
    audio, file_sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * sr / file_sr)).astype(np.float32)
    return audio


def build_section(entries, audio_dir, sr, rng, start_after_break=False):
    """Concatenate entries into a section with pauses and jitter."""
    parts = []
    prev_speaker = None
    past_break = not start_after_break

    for entry in entries:
        if entry["type"] == "section_break":
            if not past_break:
                past_break = True
                continue
            parts.append(np.zeros(int(sr * SECTION_PAUSE), dtype=np.float32))
            prev_speaker = None
            continue

        if not past_break:
            continue

        if entry["type"] == "pause":
            parts.append(np.zeros(int(sr * entry["duration"]), dtype=np.float32))
        elif entry["type"] == "line":
            wav_path = audio_dir / entry["file"]
            if not wav_path.exists():
                parts.append(np.zeros(int(sr * 0.5), dtype=np.float32))
                continue
            audio = load_line_audio(wav_path, sr)

            if prev_speaker is not None:
                base = SAME_SPEAKER_PAUSE if entry["speaker"] == prev_speaker else SPEAKER_CHANGE_PAUSE
                jitter = rng.uniform(-0.03, 0.06)
                pause = max(0.05, base + jitter)
                parts.append(np.zeros(int(sr * pause), dtype=np.float32))
            parts.append(audio)
            prev_speaker = entry["speaker"]

    return np.concatenate(parts) if parts else np.array([], dtype=np.float32)


def mix_episode(script_path, lines_dir, intro_wav_path, music_bed_path,
                sting_path, bc_dir, output_path, seed=42):
    """Custom mix: intro + music bed + cold open + sting + conversation.

    Structure:
      00:00  Music bed fades in (~4s solo)
      00:04  Intro voiceover (music ducked)
      ~00:40 Music returns full (~5s) then fades into cold open (~8s bleed)
      ~00:50 Cold open (no music after bleed)
      ~01:30 Sting crossfade into main conversation
      ~09:00 Closing question -> silence -> end
    """
    print("\n=== Step: Mix episode ===")

    sr = TARGET_SR
    rng = np.random.default_rng(seed)
    entries = parse_script_with_sections(script_path)

    # Load assets
    print("  Loading assets...")
    intro_voice, intro_sr = sf.read(str(intro_wav_path), dtype="float32")
    if intro_voice.ndim > 1:
        intro_voice = intro_voice.mean(axis=1)
    if intro_sr != sr:
        from scipy.signal import resample
        intro_voice = resample(intro_voice, int(len(intro_voice) * sr / intro_sr)).astype(np.float32)
    print(f"    Intro voice: {len(intro_voice) / sr:.1f}s")

    music_bed = load_audio_file(music_bed_path, sr)
    print(f"    Music bed: {len(music_bed) / sr:.1f}s")

    sting = load_audio_file(sting_path, sr)
    print(f"    Sting: {len(sting) / sr:.1f}s")

    # --- INTRO SECTION: music bed + voiceover ---
    print("  Building intro...")
    music_solo_samples = int(sr * MUSIC_FADE_IN)
    post_voice_samples = int(sr * MUSIC_POST_VOICE)
    bleed_samples = int(sr * MUSIC_BLEED_INTO_COLD)
    pause_samples = int(sr * 1.5)

    intro_voice_len = music_solo_samples + len(intro_voice) + post_voice_samples
    music_total_len = intro_voice_len + pause_samples + bleed_samples

    # Build music bed track with volume envelope
    music_track = np.zeros(music_total_len, dtype=np.float32)
    music_needed = min(len(music_bed), music_total_len)
    music_track[:music_needed] = music_bed[:music_needed]

    fade_in_end = int(sr * 2.0)
    voice_start = music_solo_samples
    voice_end = music_solo_samples + len(intro_voice)
    post_voice_end = intro_voice_len
    fade_out_start = intro_voice_len + pause_samples

    for i in range(music_total_len):
        if i < fade_in_end:
            music_track[i] *= MUSIC_FULL_VOL * (i / fade_in_end)
        elif i < voice_start:
            music_track[i] *= MUSIC_FULL_VOL
        elif i < voice_end:
            music_track[i] *= MUSIC_DUCK_VOL
        elif i < post_voice_end:
            music_track[i] *= MUSIC_FULL_VOL
        elif i < fade_out_start:
            music_track[i] *= MUSIC_FULL_VOL
        else:
            fade_progress = (i - fade_out_start) / max(1, music_total_len - fade_out_start)
            music_track[i] *= MUSIC_FULL_VOL * (1.0 - fade_progress)

    intro_section = music_track[:intro_voice_len].copy()
    intro_section[voice_start:voice_end] += intro_voice
    music_bleed = music_track[intro_voice_len:]

    print(f"    Intro section: {len(intro_section) / sr:.1f}s")
    print(f"    Music bleed: {len(music_bleed) / sr:.1f}s")

    # --- COLD OPEN ---
    print("  Building cold open...")
    cold_entries = []
    for entry in entries:
        if entry["type"] == "section_break":
            break
        cold_entries.append(entry)

    cold_open = build_section(cold_entries, lines_dir, sr, rng)
    print(f"    Cold open: {len(cold_open) / sr:.1f}s")

    # --- STING TRANSITION ---
    sting_copy = sting.copy()
    fade_in_samples = int(sr * STING_FADE_IN)
    sting_copy[:fade_in_samples] *= np.linspace(0, 1, fade_in_samples, dtype=np.float32)

    overlap_start = min(int(sr * COLD_OPEN_OVERLAP), len(cold_open))
    cold_body = cold_open[:-overlap_start] if overlap_start > 0 else cold_open
    cold_tail = cold_open[-overlap_start:] if overlap_start > 0 else np.array([], dtype=np.float32)

    sting_zone_len = max(len(sting_copy), len(cold_tail))
    sting_zone = np.zeros(sting_zone_len, dtype=np.float32)
    sting_zone[:len(cold_tail)] += cold_tail
    sting_vol = 0.35
    for i in range(min(len(sting_copy), sting_zone_len)):
        vol = sting_vol if i < len(cold_tail) else 1.0
        sting_zone[i] += sting_copy[i] * vol

    # --- MAIN CONVERSATION ---
    print("  Building main conversation...")
    main_conv = build_section(entries, lines_dir, sr, rng, start_after_break=True)
    print(f"    Main conversation: {len(main_conv) / sr:.1f}s")

    # Crossfade sting tail into conversation
    crossfade_samples = int(sr * STING_CROSSFADE)
    if len(sting_zone) > crossfade_samples:
        sting_pre = sting_zone[:-crossfade_samples]
        sting_tail = sting_zone[-crossfade_samples:] * np.linspace(1, 0, crossfade_samples, dtype=np.float32)
    else:
        sting_pre = np.array([], dtype=np.float32)
        sting_tail = sting_zone * np.linspace(1, 0, len(sting_zone), dtype=np.float32)

    overlap_len = len(sting_tail)
    if len(main_conv) >= overlap_len:
        overlap_zone = sting_tail + main_conv[:overlap_len]
        main_rest = main_conv[overlap_len:]
    else:
        overlap_zone = sting_tail[:len(main_conv)] + main_conv
        main_rest = np.array([], dtype=np.float32)

    tail_silence = np.zeros(int(sr * 1.5), dtype=np.float32)

    # --- FINAL ASSEMBLY ---
    cold_with_rest = np.concatenate([cold_body, sting_pre, overlap_zone, main_rest, tail_silence])
    pause_plus_cold = np.concatenate([
        np.zeros(int(sr * 1.5), dtype=np.float32),
        cold_with_rest,
    ])

    # Overlay music bleed
    bleed_len = min(len(music_bleed), len(pause_plus_cold))
    pause_plus_cold[:bleed_len] += music_bleed[:bleed_len]

    full = np.concatenate([intro_section, pause_plus_cold])

    # --- BACKCHANNELS ---
    if bc_dir and Path(bc_dir).exists() and any(Path(bc_dir).glob("bc_*.wav")):
        print("  Placing backchannels...")
        bc_clips = {}
        for bc_file in sorted(Path(bc_dir).glob("bc_*.wav")):
            parts_name = bc_file.stem.split("_")
            if len(parts_name) >= 3:
                speaker = parts_name[1]
                audio_bc, sr_bc = sf.read(str(bc_file), dtype="float32")
                if sr_bc != sr:
                    from scipy.signal import resample as sp_resample
                    audio_bc = sp_resample(audio_bc, int(len(audio_bc) * sr / sr_bc)).astype(np.float32)
                if speaker not in bc_clips:
                    bc_clips[speaker] = []
                bc_clips[speaker].append(audio_bc)

        STEM_MAP = {"junior manager": "alex", "team member 1": "morgan", "team member 2": "zara"}
        MAX_BC = 12
        MIN_GAP = 5
        MIN_TURN_DUR = 6.0
        bc_volume = 10 ** (-3.0 / 20)
        bc_count = 0

        # Build line positions — only for main conversation (after first section break).
        # The cold open is separately assembled, so its sample offsets in `full` are
        # not linear; placing backchannels there would land at wrong positions.
        # Main conversation starts at: intro_section + pause + cold_body + sting_pre + overlap_zone
        main_conv_start_in_full = (
            len(intro_section) + int(sr * 1.5) + len(cold_body) + len(sting_pre) + len(overlap_zone)
        )
        prev_spk = None
        pos_scan = main_conv_start_in_full
        line_positions = []
        past_first_break = False
        for entry in entries:
            if entry["type"] == "section_break":
                if not past_first_break:
                    past_first_break = True
                    continue
                # Subsequent section breaks within main conversation
                prev_spk = None
                pos_scan += int(sr * SECTION_PAUSE)
                continue
            if not past_first_break:
                continue
            if entry["type"] == "pause":
                pos_scan += int(sr * entry["duration"])
                continue
            if entry["type"] == "line":
                wav_path = lines_dir / entry["file"]
                if not wav_path.exists():
                    pos_scan += int(sr * 0.5)
                    continue
                info = sf.info(str(wav_path))
                line_dur = info.duration
                line_samples = int(line_dur * sr)

                if prev_spk is not None and entry["speaker"] != prev_spk:
                    pos_scan += int(sr * SPEAKER_CHANGE_PAUSE)
                elif prev_spk is not None:
                    pos_scan += int(sr * SAME_SPEAKER_PAUSE)

                line_positions.append({
                    "pos": pos_scan, "speaker": entry["speaker"], "duration": line_dur,
                })
                pos_scan += line_samples
                prev_spk = entry["speaker"]

        last_bc_idx = -MIN_GAP
        last_clip_per_speaker = {}
        for i in range(1, len(line_positions)):
            if bc_count >= MAX_BC:
                break
            prev_line = line_positions[i - 1]
            curr_line = line_positions[i]
            if (prev_line["speaker"] == curr_line["speaker"]
                    or prev_line["duration"] < MIN_TURN_DUR
                    or i - last_bc_idx < MIN_GAP):
                continue

            prev_stem = STEM_MAP.get(prev_line["speaker"], prev_line["speaker"])
            curr_stem = STEM_MAP.get(curr_line["speaker"], curr_line["speaker"])
            possible = [s for s in ["alex", "morgan", "zara"] if s != prev_stem and s != curr_stem]
            if not possible or possible[0] not in bc_clips:
                continue
            reactor = possible[0]

            available = list(range(len(bc_clips[reactor])))
            last_used = last_clip_per_speaker.get(reactor, -1)
            if last_used in available and len(available) > 1:
                available.remove(last_used)
            clip_idx = available[int(rng.integers(len(available)))]
            clip = bc_clips[reactor][clip_idx]
            last_clip_per_speaker[reactor] = clip_idx

            bc_pos = curr_line["pos"] - int(sr * 0.1)
            if 0 < bc_pos < len(full) - len(clip):
                full[bc_pos:bc_pos + len(clip)] += clip * bc_volume
                bc_count += 1
                last_bc_idx = i

        print(f"    Placed {bc_count} backchannels (max {MAX_BC})")

    # --- ROOM TONE ---
    print("  Adding room tone...")
    pink = generate_pink_noise(len(full), rng)
    dialogue_start = len(intro_section) + int(sr * 1.5) + int(sr * MUSIC_BLEED_INTO_COLD)
    room_tone_mask = np.zeros(len(full), dtype=np.float32)
    fade_start = len(intro_section)
    fade_end = min(dialogue_start, len(full))
    for i in range(fade_start, fade_end):
        room_tone_mask[i] = ROOM_TONE_LEVEL * ((i - fade_start) / max(1, fade_end - fade_start))
    room_tone_mask[fade_end:] = ROOM_TONE_LEVEL
    full = full + pink * room_tone_mask

    total_dur = len(full) / sr
    print(f"  Total: {total_dur:.1f}s ({total_dur / 60:.1f} min)")

    # Write premix (before mastering)
    sf.write(str(output_path), full, sr)
    print(f"  Premix: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline for It Is Both Ep01: TTS + clean + validate + mix + master",
    )
    parser.add_argument("script", help="Main episode script (.txt)")
    parser.add_argument("--intro-lines", required=True, help="Intro lines text file")
    parser.add_argument("--overrides", help="TTS overrides JSON")
    parser.add_argument("--music-bed", required=True, help="Music bed audio file")
    parser.add_argument("--sting", required=True, help="Transition sting audio file")
    parser.add_argument("--backchannels", help="Existing backchannel clips directory")
    parser.add_argument("--generate-backchannels", action="store_true",
                        help="Generate fresh backchannel clips")
    parser.add_argument("-o", "--output", default="~/ep01_final.wav", help="Output file path")
    parser.add_argument("--work-dir", default="~/ep01_pipeline", help="Working directory")
    parser.add_argument("--skip-tts", action="store_true", help="Skip TTS, reuse existing lines")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-generate only lines missing WAV files (skip existing)")
    parser.add_argument("--skip-clean", action="store_true", help="Skip audio cleaning")
    parser.add_argument("--skip-validate", action="store_true", help="Skip ASR validation")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip reverb/volume preprocessing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for mix (default: 42)")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).expanduser()
    lines_dir = work_dir / "lines"
    intro_dir = work_dir / "intro"
    bc_dir = work_dir / "backchannels"
    lines_dir.mkdir(parents=True, exist_ok=True)
    intro_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output).expanduser()
    premix_path = output_path.with_name(output_path.stem + "_premix.wav")

    # Load overrides
    overrides = None
    if args.overrides:
        with open(args.overrides) as f:
            overrides = json.load(f).get("overrides", {})
        print(f"Loaded {len(overrides)} TTS overrides")

    # Parse scripts
    main_lines = parse_lines(args.script)
    intro_lines = parse_intro_lines(args.intro_lines)
    print(f"Main script: {len(main_lines)} lines")
    print(f"Intro: {len(intro_lines)} lines")

    manifest_path = work_dir / "manifest.json"

    # ── Step 1: TTS generation ──
    if not args.skip_tts:
        print("\n=== Step 1: Generate TTS ===")

        retry = args.retry_failed

        print("\n--- Intro lines ---")
        intro_manifest, intro_failed = generate_tts(
            intro_lines, intro_dir, prefix="intro_", retry=retry,
        )
        print(f"Intro: {len(intro_manifest)}/{len(intro_lines)} generated")

        print("\n--- Main dialogue ---")
        main_manifest, main_failed = generate_tts(
            main_lines, lines_dir, overrides=overrides, retry=retry,
        )
        print(f"Main: {len(main_manifest)}/{len(main_lines)} generated, {len(main_failed)} failed")

        if args.generate_backchannels:
            print("\n--- Backchannels ---")
            generate_backchannels_tts(bc_dir)

        # Save manifests
        manifest = {
            "engine": "qwen", "intro": intro_manifest,
            "main": main_manifest, "failed": main_failed + intro_failed,
        }
        manifest_path = work_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest: {manifest_path}")

        if main_failed or intro_failed:
            total_failed = len(main_failed) + len(intro_failed)
            print(f"\nWARNING: {total_failed} lines failed:")
            for item in main_failed + intro_failed:
                print(f"  {item['index']:03d} {item['speaker']}: {item.get('error', '?')}")
    else:
        print("\n=== Step 1: Skipped (--skip-tts) ===")
        if not manifest_path.exists():
            print("  WARNING: No manifest.json found — validate and preprocess steps will be skipped.")
            print("  If audio was not preprocessed in a prior run, mix will use raw TTS output.")

    # ── Step 2: Clean audio ──
    if not args.skip_clean:
        clean_audio(lines_dir)
        clean_audio(intro_dir)
    else:
        print("\n=== Step 2: Skipped (--skip-clean) ===")

    # ── Step 3: Validate ──
    manifest_path = work_dir / "manifest.json"
    if not args.skip_validate and manifest_path.exists():
        with open(manifest_path) as f:
            manifest_data = json.load(f)
        # Manifest must live in audio dir (validate_tts resolves paths relative to it)
        val_manifest = lines_dir / "validation_manifest.json"
        with open(val_manifest, "w") as f:
            json.dump(manifest_data["main"], f, indent=2)
        validate_tts(lines_dir, val_manifest)
    else:
        print("\n=== Step 3: Skipped (--skip-validate or no manifest) ===")

    # ── Step 4: Preprocess ──
    if not args.skip_preprocess and manifest_path.exists():
        with open(manifest_path) as f:
            manifest_data = json.load(f)
        # Manifests must live in audio dirs (mix_preprocess resolves paths relative to them)
        main_pp_manifest = lines_dir / "preprocess_manifest.json"
        with open(main_pp_manifest, "w") as f:
            json.dump(manifest_data["main"], f, indent=2)
        preprocess_audio(lines_dir, main_pp_manifest)
        if manifest_data.get("intro"):
            intro_pp_manifest = intro_dir / "preprocess_manifest.json"
            with open(intro_pp_manifest, "w") as f:
                json.dump(manifest_data["intro"], f, indent=2)
            preprocess_audio(intro_dir, intro_pp_manifest)
    else:
        print("\n=== Step 4: Skipped (--skip-preprocess or no manifest) ===")

    # ── Step 5: Assemble intro ──
    intro_wav = work_dir / "intro_assembled.wav"
    assemble_intro(intro_dir, args.intro_lines, intro_wav)

    # ── Step 6: Determine backchannel directory ──
    if args.backchannels:
        bc_path = Path(args.backchannels).expanduser()
    elif bc_dir.exists() and any(bc_dir.glob("bc_*.wav")):
        bc_path = bc_dir
    else:
        bc_path = None

    # ── Step 7: Mix ──
    mix_episode(
        args.script, lines_dir, intro_wav,
        args.music_bed, args.sting, bc_path, premix_path,
        seed=args.seed,
    )

    # ── Step 8: Master ──
    master_audio(premix_path, output_path)

    # Clean up premix
    if premix_path.exists() and output_path.exists():
        premix_path.unlink()
        print(f"  Cleaned up premix: {premix_path}")

    print(f"\n=== DONE === Output: {output_path}")


if __name__ == "__main__":
    main()
