"""Single-pass pipeline for It Is Both Episode 01: The Nodding.

Generates TTS, validates, then mixes in one pass. Raw TTS files are never
modified — all processing happens in memory during the mix. This is the
approach that worked in v1 (mix_ep01_v2.py), extended with TTS generation
and backchannel support.

Usage (on gpu-server):
    source ~/podcast-generator/qwen-tts-env/bin/activate

    # Full pipeline
    python3 pipeline_ep01_v2.py ~/ep01_script.txt \\
        --intro-lines ~/podcast-generator/podcasts/it-is-both/ep01_intro_lines.txt \\
        --overrides ~/podcast-generator/podcasts/it-is-both/ep01_tts_overrides.json \\
        --music-bed ~/music_bed.mp3 --sting ~/sting.mp3 \\
        --backchannels ~/ep01_pipeline/backchannels \\
        -o ~/ep01_final.wav

    # Intro only (for testing)
    python3 pipeline_ep01_v2.py ~/ep01_script.txt \\
        --intro-lines ~/podcast-generator/podcasts/it-is-both/ep01_intro_lines.txt \\
        --music-bed ~/music_bed.mp3 --sting ~/sting.mp3 \\
        --intro-only -o ~/ep01_intro_test.wav

    # Skip TTS (reuse existing lines)
    python3 pipeline_ep01_v2.py ... --skip-tts -o ~/ep01_final.wav
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
# Voice references (gpu-server)
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
VOICE_REFS["junior manager"] = VOICE_REFS["alex"]
VOICE_REFS["team member 1"] = VOICE_REFS["morgan"]
VOICE_REFS["team member 2"] = VOICE_REFS["zara"]

# ---------------------------------------------------------------------------
# Mix constants
# ---------------------------------------------------------------------------

TARGET_SR = 24000

SECTION_PAUSE = 1.0
SPEAKER_CHANGE_PAUSE = 0.15
SAME_SPEAKER_PAUSE = 0.08
INTERJECTION_PAUSE = 0.05       # after short lines (<1.5s) — quick return
INTERJECTION_THRESHOLD = 1.5    # seconds — lines shorter than this are interjections
BEAT_PAUSE = 0.5

COLD_OPEN_OVERLAP = 3.0
STING_FADE_IN = 0.5
STING_CROSSFADE = 2.5

MUSIC_FADE_IN = 4.0
MUSIC_DUCK_VOL = 0.12
MUSIC_FULL_VOL = 0.35
MUSIC_POST_VOICE = 5.0
MUSIC_BLEED_INTO_COLD = 8.0

SPEAKER_VOLUME_DB = {
    "alex": 0.0, "morgan": 0.0, "zara": 2.5,
    "junior manager": 0.0, "team member 1": 0.0, "team member 2": 0.0,
}

REVERB_MIX = 0.02
REVERB_DECAY = 0.15
ROOM_TONE_LEVEL = 0.002


# ---------------------------------------------------------------------------
# Script parsers
# ---------------------------------------------------------------------------


def parse_lines(path):
    """Parse speaker lines for TTS generation."""
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
    """Parse intro lines."""
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
            # Inline backchannel: [react: speaker type]
            react_match = re.match(
                r"^\[react:\s*(\w+)\s+(\w+)\]$", lower,
            )
            if react_match:
                entries.append({
                    "type": "backchannel",
                    "reactor": react_match.group(1),
                    "bc_type": react_match.group(2),
                })
                continue
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
# TTS generation
# ---------------------------------------------------------------------------


MAX_DURATION_PER_WORD = 0.6   # seconds — expected max TTS duration per word
MAX_SEGMENT_DURATION = 10.0   # seconds — anything longer is a hallucination
HALLUCINATION_RETRIES = 3
RETRY_TEMPERATURE = 0.4
RETRY_REPETITION_PENALTY = 1.3


def _estimate_max_duration(text):
    """Estimate reasonable max duration for a text based on word count."""
    words = len(text.split())
    return max(MAX_SEGMENT_DURATION, words * MAX_DURATION_PER_WORD)


def _generate_one(model, text, voice, temperature=0.7, rep_penalty=1.2):
    """Generate TTS for a single text. Returns (audio, sr)."""
    import torch
    wavs, sr = model.generate_voice_clone(
        text=text, language="English",
        ref_audio=str(voice["ref"]), ref_text=voice["text"],
        temperature=temperature, repetition_penalty=rep_penalty,
    )
    audio = wavs[0].copy()
    del wavs; torch.cuda.empty_cache()
    return audio, sr


def _generate_with_guard(model, text, voice, temperature=0.7, rep_penalty=1.2):
    """Generate TTS with hallucination guard. Retries with lower temperature."""
    import torch
    max_dur = _estimate_max_duration(text)

    audio, sr = _generate_one(model, text, voice, temperature, rep_penalty)
    duration = len(audio) / sr

    if duration <= max_dur:
        return audio, sr

    # Hallucination detected — retry with progressively safer settings
    for attempt in range(HALLUCINATION_RETRIES):
        temp = RETRY_TEMPERATURE - (attempt * 0.05)  # 0.4, 0.35, 0.3
        print(f"    HALLUCINATION ({duration:.1f}s > {max_dur:.1f}s), "
              f"retry {attempt + 1}/{HALLUCINATION_RETRIES} (temp={temp:.2f})")
        audio, sr = _generate_one(
            model, text, voice, temperature=temp,
            rep_penalty=RETRY_REPETITION_PENALTY,
        )
        duration = len(audio) / sr
        if duration <= max_dur:
            return audio, sr

    print(f"    WARNING: still long after {HALLUCINATION_RETRIES} retries "
          f"({duration:.1f}s), using best attempt")
    return audio, sr


def generate_tts(lines, output_dir, overrides=None, prefix=""):
    """Generate TTS for all lines. Skips existing files.

    Includes hallucination guard: segments exceeding expected duration are
    automatically retried with lower temperature and higher repetition penalty.
    """
    import torch
    from qwen_tts import Qwen3TTSModel

    print("Loading Qwen3-TTS...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0", dtype=torch.bfloat16,
    )
    print("Model loaded.")

    manifest, failed = [], []

    for i, line in enumerate(lines):
        speaker, text, idx = line["speaker"], line["text"], line["index"]
        voice = VOICE_REFS.get(speaker)
        if not voice:
            print(f"  WARNING: No voice ref for '{speaker}', skipping")
            continue

        filename = f"{prefix}{idx:03d}_{speaker.replace(' ', '_')}.wav"
        out_path = output_dir / filename

        if out_path.exists():
            info = sf.info(str(out_path))
            # Validate existing file too
            max_dur = _estimate_max_duration(text)
            if info.duration > max_dur:
                print(f"  [{i + 1}/{len(lines)}] {filename} exists but "
                      f"too long ({info.duration:.1f}s > {max_dur:.1f}s), re-generating")
                out_path.unlink()
            else:
                manifest.append({
                    "index": idx, "speaker": speaker, "text": text,
                    "file": filename, "duration": info.duration,
                })
                continue

        override_key = f"{idx:03d}"
        segments = None
        if overrides and override_key in overrides:
            ov = overrides[override_key]
            if isinstance(ov, str):
                text = ov
            elif isinstance(ov, list):
                segments = ov

        if i > 0 and i % 10 == 0:
            torch.cuda.empty_cache()

        print(f"  [{i + 1}/{len(lines)}] {speaker}: {text[:55]}...")

        try:
            if segments:
                parts = []
                for seg in segments:
                    audio, sr = _generate_with_guard(
                        model, seg["text"], voice,
                    )
                    parts.append(audio)
                    pause = seg.get("pause_after", 0)
                    if pause > 0:
                        parts.append(np.zeros(int(sr * pause), dtype=np.float32))
                audio = np.concatenate(parts)
            else:
                audio, sr = _generate_with_guard(model, text, voice)

            sf.write(str(out_path), audio, sr)
            duration = len(audio) / sr
            manifest.append({
                "index": idx, "speaker": speaker, "text": text,
                "file": filename, "duration": duration,
            })
            print(f"    -> {filename} ({duration:.1f}s)")

        except Exception as e:
            print(f"    ERROR: {str(e)[:80]}")
            failed.append({"index": idx, "speaker": speaker, "text": text, "error": str(e)[:80]})

    del model; torch.cuda.empty_cache()
    return manifest, failed


# ---------------------------------------------------------------------------
# Inline audio processing (from v1 mix_ep01_v2.py — never writes to disk)
# ---------------------------------------------------------------------------


def trim_silence(audio, sr, threshold_db=-35):
    """Trim leading silence only."""
    threshold = 10 ** (threshold_db / 20)
    window = max(1, int(sr * 0.01))
    abs_audio = np.abs(audio)
    for i in range(0, len(audio) - window, window):
        if np.max(abs_audio[i:i + window]) > threshold:
            start = max(0, i - int(sr * 0.04))
            return audio[start:]
    return audio


def rms_normalize(audio, target_rms=0.1):
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        audio = audio * (target_rms / rms)
    return audio


def apply_speaker_volume(audio, speaker):
    db = SPEAKER_VOLUME_DB.get(speaker, 0.0)
    if db != 0.0:
        audio = audio * (10 ** (db / 20))
    return audio


def generate_room_ir(sr, decay_time=0.3):
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


def apply_reverb(audio, ir, mix=0.08):
    wet = fftconvolve(audio, ir, mode="full")[:len(audio)].astype(np.float32)
    return audio * (1 - mix) + wet * mix


def apply_clip_fades(audio, sr, fade_in_ms=35, fade_out_ms=20):
    """Fade edges and suppress clicks at clip boundaries only (first/last 50ms).

    Fade-in is capped to the leading silence so it never eats into speech.
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

    check_samples = int(sr * 0.05)
    for region in [audio[:check_samples], audio[-check_samples:]]:
        for i in range(1, len(region)):
            jump = abs(region[i] - region[i - 1])
            if jump > 0.08:
                start = max(0, i - 3)
                end = min(len(region), i + 4)
                region[start:end] = np.linspace(
                    region[start], region[min(end, len(region) - 1)],
                    end - start, dtype=np.float32,
                )
    return audio


def load_and_process_line(wav_path, speaker, sr, room_ir):
    """Load a line, process in memory. Never writes back."""
    audio, file_sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * sr / file_sr)).astype(np.float32)
    audio = trim_silence(audio, sr)
    audio = rms_normalize(audio, target_rms=0.1)
    audio = apply_speaker_volume(audio, speaker)
    audio = apply_reverb(audio, room_ir, mix=REVERB_MIX)
    audio = apply_clip_fades(audio, sr)
    return audio


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _pick_bc_clip(bc_clips, reactor, bc_type, rng):
    """Pick a random backchannel clip for a reactor+type. Returns audio or None."""
    clips = bc_clips.get(bc_type, {}).get(reactor, [])
    if not clips:
        return None
    return clips[int(rng.integers(len(clips)))]


BC_VOLUME_DB = -3.0  # backchannel level relative to dialogue
BC_OVERLAP_MS = (200, 500)  # how far back into previous line to start the BC


def build_section(entries, audio_dir, sr, room_ir, rng, start_after_break=False,
                  bc_clips=None):
    """Build audio from script entries. Loads and processes each line in memory.

    Backchannel entries (type "backchannel") are overlaid on the tail of the
    previous line. If the backchannel clip is longer than the overlap region,
    extra silence is inserted so the next speaker pauses until the reaction
    finishes — the primary speaker yields the floor to the laugh.
    """
    parts = []
    prev_speaker = None
    past_break = not start_after_break
    # Pending backchannel: clip to overlay on the NEXT gap
    pending_bc = None
    # Index into parts[] of the last spoken line (not pauses/silences)
    last_line_idx = None
    prev_line_dur = 0.0  # duration of previous line in seconds

    bc_vol = 10 ** (BC_VOLUME_DB / 20)

    for entry in entries:
        if entry["type"] == "section_break":
            if not past_break:
                past_break = True
                continue
            pending_bc = None
            last_line_idx = None
            parts.append(np.zeros(int(sr * SECTION_PAUSE), dtype=np.float32))
            prev_speaker = None
            continue

        if not past_break:
            continue

        if entry["type"] == "pause":
            parts.append(np.zeros(int(sr * entry["duration"]), dtype=np.float32))

        elif entry["type"] == "backchannel":
            if bc_clips is None:
                continue
            clip = _pick_bc_clip(bc_clips, entry["reactor"], entry["bc_type"], rng)
            if clip is not None:
                pending_bc = {"clip": clip.copy(), "reactor": entry["reactor"],
                              "bc_type": entry["bc_type"]}
            continue

        elif entry["type"] == "line":
            wav_path = audio_dir / entry["file"]
            if not wav_path.exists():
                parts.append(np.zeros(int(sr * 0.5), dtype=np.float32))
                pending_bc = None
                continue
            audio = load_and_process_line(wav_path, entry["speaker"], sr, room_ir)

            if prev_speaker is not None:
                if entry["speaker"] == prev_speaker:
                    base = SAME_SPEAKER_PAUSE
                elif prev_line_dur < INTERJECTION_THRESHOLD:
                    base = INTERJECTION_PAUSE
                else:
                    base = SPEAKER_CHANGE_PAUSE
                jitter = rng.uniform(-0.02, 0.04)
                gap = max(0.03, base + jitter)

                if pending_bc is not None and last_line_idx is not None:
                    # --- Place backchannel overlapping tail of previous LINE ---
                    bc_clip = pending_bc["clip"] * bc_vol
                    reactor = pending_bc["reactor"]
                    bc_type = pending_bc["bc_type"]
                    pending_bc = None
                    bc_len = len(bc_clip)
                    bc_dur = bc_len / sr

                    # Find the actual previous spoken audio (skip past pauses)
                    prev_line_audio = parts[last_line_idx]

                    # Overlap into the tail of the previous spoken audio
                    overlap_ms = rng.uniform(*BC_OVERLAP_MS)
                    overlap_samples = int(sr * overlap_ms / 1000)
                    overlap_samples = min(overlap_samples, len(prev_line_audio))

                    # Duck the previous line under the backchannel overlap
                    if bc_dur > 0.5 and overlap_samples > 0:
                        duck_region = prev_line_audio[-overlap_samples:]
                        fade = min(int(sr * 0.04), len(duck_region) // 3)
                        duck = np.ones(len(duck_region), dtype=np.float32) * 0.6
                        if fade > 0:
                            duck[:fade] = np.linspace(1.0, 0.6, fade, dtype=np.float32)
                        prev_line_audio[-overlap_samples:] = duck_region * duck

                    # Mix backchannel into the overlap zone of the previous line
                    mix_into_prev = min(overlap_samples, bc_len)
                    if mix_into_prev > 0:
                        prev_line_audio[-mix_into_prev:] += bc_clip[:mix_into_prev]

                    # Remaining backchannel audio that spills past the previous line
                    spill = bc_clip[mix_into_prev:] if bc_len > mix_into_prev else np.array([], dtype=np.float32)

                    # The gap must be long enough for the spill to finish.
                    # Include any pauses already appended between the BC cue
                    # and this line — they count toward the gap.
                    existing_gap_samples = sum(
                        len(parts[i]) for i in range(last_line_idx + 1, len(parts))
                    )
                    spill_dur = len(spill) / sr if len(spill) > 0 else 0.0
                    needed_gap = spill_dur + 0.08  # 80ms breathing room
                    extra_gap = max(gap, needed_gap - existing_gap_samples / sr)

                    gap_samples = int(sr * extra_gap)
                    gap_audio = np.zeros(gap_samples, dtype=np.float32)

                    # Place spill: it starts right after the previous line ends,
                    # which means it goes into the existing gap parts first,
                    # then into the new gap we're adding.
                    if len(spill) > 0:
                        spill_pos = 0
                        # Fill into existing gap parts (pauses between bc cue and now)
                        for i in range(last_line_idx + 1, len(parts)):
                            chunk = min(len(parts[i]), len(spill) - spill_pos)
                            if chunk > 0:
                                parts[i][:chunk] += spill[spill_pos:spill_pos + chunk]
                                spill_pos += chunk
                            if spill_pos >= len(spill):
                                break
                        # Remainder into the new gap
                        remainder = len(spill) - spill_pos
                        if remainder > 0 and remainder <= gap_samples:
                            gap_audio[:remainder] += spill[spill_pos:]

                    parts.append(gap_audio)
                    print(f"    BC: {reactor} {bc_type} ({bc_dur:.1f}s) -> "
                          f"overlap {overlap_samples/sr*1000:.0f}ms, "
                          f"spill {spill_dur:.2f}s, gap {extra_gap:.2f}s")
                else:
                    pending_bc = None
                    parts.append(np.zeros(int(sr * gap), dtype=np.float32))
            else:
                pending_bc = None

            parts.append(audio)
            last_line_idx = len(parts) - 1
            prev_line_dur = len(audio) / sr
            prev_speaker = entry["speaker"]

    return np.concatenate(parts) if parts else np.array([], dtype=np.float32)


def build_intro(intro_dir, intro_lines, sr, room_ir):
    """Assemble intro voiceover from individual line WAVs."""
    parts = []
    for line in intro_lines:
        speaker = line["speaker"]
        idx = line["index"]
        filename = f"intro_{idx:03d}_{speaker}.wav"
        wav_path = intro_dir / filename
        if not wav_path.exists():
            continue
        audio = load_and_process_line(wav_path, speaker, sr, room_ir)
        parts.append(audio)
        pause = 0.4 if speaker in ("morgan", "zara") else 0.15
        parts.append(np.zeros(int(sr * pause), dtype=np.float32))

    return np.concatenate(parts) if parts else np.array([], dtype=np.float32)


def load_audio_file(path, target_sr):
    """Load any audio file via ffmpeg, convert to mono float32."""
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
    white = rng.standard_normal(num_samples).astype(np.float32)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004709510]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
    pink = lfilter(b, a, white).astype(np.float32)
    peak = np.max(np.abs(pink))
    if peak > 0:
        pink /= peak
    return pink


# ---------------------------------------------------------------------------
# Single-pass mix
# ---------------------------------------------------------------------------


def load_bc_clips(bc_dir, sr):
    """Load backchannel clips grouped by type and speaker.

    Returns {"laugh": {"alex": [array, ...], ...}, "breath": {...}}
    Clips 00-04 = laugh, 05-08 = breath (by filename index).
    """
    bc_clips = {"laugh": {}, "breath": {}}
    if not bc_dir or not Path(bc_dir).exists():
        return bc_clips
    for bc_file in sorted(Path(bc_dir).glob("bc_*.wav")):
        parts_name = bc_file.stem.split("_")
        if len(parts_name) >= 3:
            speaker = parts_name[1]
            idx = int(parts_name[2])
            audio_bc, sr_bc = sf.read(str(bc_file), dtype="float32")
            if sr_bc != sr:
                from scipy.signal import resample as sp_resample
                audio_bc = sp_resample(audio_bc, int(len(audio_bc) * sr / sr_bc)).astype(np.float32)
            # Fade in/out to prevent clicks at clip boundaries
            audio_bc = apply_clip_fades(audio_bc, sr)
            bc_type = "laugh" if idx <= 4 else "breath"
            if speaker not in bc_clips[bc_type]:
                bc_clips[bc_type][speaker] = []
            bc_clips[bc_type][speaker].append(audio_bc)
    total = sum(len(v) for t in bc_clips.values() for v in t.values())
    if total:
        print(f"    Backchannel clips: {total} loaded")
    return bc_clips


def mix_episode(script_path, lines_dir, intro_voice, music_bed_path,
                sting_path, bc_dir, bc_script, output_path, seed=42,
                intro_only=False):
    """Mix everything in one pass. Inline [react:] cues are handled during
    section building. Legacy bc_script JSON is still supported as fallback."""
    print("\n=== Mixing ===")

    sr = TARGET_SR
    rng = np.random.default_rng(seed)
    entries = parse_script_with_sections(script_path)
    room_ir = generate_room_ir(sr, REVERB_DECAY)

    # Load backchannel clips (used by both inline and legacy paths)
    bc_clips = load_bc_clips(bc_dir, sr) if bc_dir else None

    # Check if script has inline backchannel cues
    has_inline_bc = any(e["type"] == "backchannel" for e in entries)

    # Load assets
    print("  Loading assets...")
    print(f"    Intro voice: {len(intro_voice) / sr:.1f}s")

    music_bed = load_audio_file(music_bed_path, sr)
    print(f"    Music bed: {len(music_bed) / sr:.1f}s")

    sting = load_audio_file(sting_path, sr)
    print(f"    Sting: {len(sting) / sr:.1f}s")

    # --- INTRO SECTION ---
    print("  Building intro...")
    music_solo_samples = int(sr * MUSIC_FADE_IN)
    post_voice_samples = int(sr * MUSIC_POST_VOICE)
    bleed_samples = int(sr * MUSIC_BLEED_INTO_COLD)
    pause_samples = int(sr * 1.5)

    intro_voice_len = music_solo_samples + len(intro_voice) + post_voice_samples
    music_total_len = intro_voice_len + pause_samples + bleed_samples

    music_track = np.zeros(music_total_len, dtype=np.float32)
    music_needed = min(len(music_bed), music_total_len)
    music_track[:music_needed] = music_bed[:music_needed]

    fade_in_end = int(sr * 2.0)
    voice_start = music_solo_samples
    voice_end = music_solo_samples + len(intro_voice)
    fade_out_start = intro_voice_len + pause_samples

    for i in range(music_total_len):
        if i < fade_in_end:
            music_track[i] *= MUSIC_FULL_VOL * (i / fade_in_end)
        elif i < voice_start:
            music_track[i] *= MUSIC_FULL_VOL
        elif i < voice_end:
            music_track[i] *= MUSIC_DUCK_VOL
        elif i < intro_voice_len:
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

    if intro_only:
        # Output just the intro for testing
        full = np.concatenate([intro_section, music_bleed])
        peak = np.max(np.abs(full))
        print(f"  Intro only: {len(full) / sr:.1f}s, peak={peak:.3f}")
        sf.write(str(output_path), full, sr)
        print(f"  -> {output_path}")
        return

    # --- COLD OPEN ---
    print("  Building cold open...")
    cold_entries = []
    for entry in entries:
        if entry["type"] == "section_break":
            break
        cold_entries.append(entry)

    cold_open = build_section(cold_entries, lines_dir, sr, room_ir, rng, bc_clips=bc_clips)
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
    main_conv = build_section(entries, lines_dir, sr, room_ir, rng, start_after_break=True, bc_clips=bc_clips)
    print(f"    Main conversation: {len(main_conv) / sr:.1f}s")

    # Crossfade sting into conversation
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

    # --- ASSEMBLE ---
    cold_with_rest = np.concatenate([cold_body, sting_pre, overlap_zone, main_rest, tail_silence])
    pause_plus_cold = np.concatenate([
        np.zeros(int(sr * 1.5), dtype=np.float32),
        cold_with_rest,
    ])

    bleed_len = min(len(music_bleed), len(pause_plus_cold))
    pause_plus_cold[:bleed_len] += music_bleed[:bleed_len]

    full = np.concatenate([intro_section, pause_plus_cold])

    # --- BACKCHANNELS ---
    if has_inline_bc:
        bc_count = sum(1 for e in entries if e["type"] == "backchannel")
        print(f"  Inline backchannels: {bc_count} cues processed during section build")

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

    # --- PEAK LIMIT ---
    peak = np.max(np.abs(full))
    limit = 10 ** (-1.0 / 20)  # -1 dBTP = 0.891
    if peak > limit:
        full = full * (limit / peak)
        print(f"  Peak limited: {peak:.3f} -> {limit:.3f}")

    total_dur = len(full) / sr
    print(f"  Total: {total_dur:.1f}s ({total_dur / 60:.1f} min)")

    sf.write(str(output_path), full, sr)
    print(f"  -> {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Single-pass pipeline for It Is Both Ep01")
    parser.add_argument("script", help="Episode script (.txt)")
    parser.add_argument("--intro-lines", required=True, help="Intro lines text file")
    parser.add_argument("--overrides", help="TTS overrides JSON")
    parser.add_argument("--music-bed", required=True, help="Music bed audio file")
    parser.add_argument("--sting", required=True, help="Sting audio file")
    parser.add_argument("--backchannels", help="Backchannel clips directory")
    parser.add_argument("--bc-script", help="Backchannel placement script (JSON)")
    parser.add_argument("-o", "--output", default="~/ep01_final.wav")
    parser.add_argument("--work-dir", default="~/ep01_pipeline")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--intro-only", action="store_true", help="Output intro section only")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    work_dir = Path(args.work_dir).expanduser()
    lines_dir = work_dir / "lines"
    intro_dir = work_dir / "intro"
    lines_dir.mkdir(parents=True, exist_ok=True)
    intro_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output).expanduser()

    overrides = None
    if args.overrides:
        with open(args.overrides) as f:
            overrides = json.load(f).get("overrides", {})
        print(f"Loaded {len(overrides)} TTS overrides")

    main_lines = parse_lines(args.script)
    intro_lines = parse_intro_lines(args.intro_lines)
    print(f"Script: {len(main_lines)} main + {len(intro_lines)} intro lines")

    # --- TTS ---
    if not args.skip_tts:
        print("\n=== Generating TTS ===")
        print("--- Intro ---")
        intro_manifest, intro_failed = generate_tts(intro_lines, intro_dir, prefix="intro_")
        print(f"Intro: {len(intro_manifest)}/{len(intro_lines)}")

        print("\n--- Main ---")
        main_manifest, main_failed = generate_tts(main_lines, lines_dir, overrides=overrides)
        print(f"Main: {len(main_manifest)}/{len(main_lines)}, {len(main_failed)} failed")

        manifest = {"intro": intro_manifest, "main": main_manifest, "failed": main_failed + intro_failed}
        with open(work_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        if main_failed or intro_failed:
            print(f"\nWARNING: {len(main_failed) + len(intro_failed)} failed:")
            for item in main_failed + intro_failed:
                print(f"  {item['index']:03d} {item['speaker']}: {item.get('error', '?')}")
    else:
        print("\n=== Skipping TTS ===")

    # --- Build intro voiceover ---
    sr = TARGET_SR
    room_ir = generate_room_ir(sr, REVERB_DECAY)
    intro_voice = build_intro(intro_dir, intro_lines, sr, room_ir)
    print(f"Intro voiceover: {len(intro_voice) / sr:.1f}s")

    # --- Load backchannel script ---
    bc_script = None
    if args.bc_script:
        with open(args.bc_script) as f:
            bc_script = json.load(f).get("placements", [])
        print(f"Backchannel script: {len(bc_script)} placements")

    # --- Mix ---
    mix_episode(
        args.script, lines_dir, intro_voice,
        args.music_bed, args.sting,
        args.backchannels, bc_script, output_path,
        seed=args.seed, intro_only=args.intro_only,
    )

    print(f"\n=== DONE ===")


if __name__ == "__main__":
    main()
