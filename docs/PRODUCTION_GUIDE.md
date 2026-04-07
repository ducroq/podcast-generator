# Podcast Production Guide

Shared production knowledge distilled from the Mondriaan podcast project (7 episodes, 3 languages, 300+ listeners/episode).

## Pipeline

```
Script → Generate (ElevenLabs, Chatterbox, or Qwen) → Validate → Trim silences → Add realism → Mix music in DAW → Master with FFmpeg
```

Each step is independent — you can retry any phase without redoing the others.

### Engine Selection

**Use one TTS engine per episode.** Mixing engines (e.g. Chatterbox + Qwen) within one episode creates audible acoustic mismatches — different noise floors, room characteristics, and tonal qualities.

| Priority | Engine | When to use |
|----------|--------|-------------|
| 1 | Qwen3-TTS | Cleanest output, best pacing, but hallucination risk on long lines |
| 2 | Chatterbox (tuned) | Reliable fallback for individual hallucinated lines |
| 3 | ElevenLabs | Best overall quality (paid), multilingual |

One Chatterbox line among 87 Qwen lines is not audibly different. A 50/50 mix is. If you must re-generate individual lines with a different engine, keep it under ~5% of total lines.

## Script Format

```
==================================================
SECTION NAME
==================================================

Speaker: [emotion] Dialogue text here.
Speaker: [thoughtful] More dialogue.
```

- Sections let you regenerate parts without re-doing the whole episode
- Audio tags (`[emotion]`) control delivery — see tag reference below
- Lines starting with `#`, `=`, or `[` (without speaker) are skipped

## Audio Tags

Tags only work with ElevenLabs `text_to_dialogue` API. They must always be in English, even in Dutch/German scripts.

**Emotions:** `[curious]` `[thoughtful]` `[excited]` `[warm]` `[calm]` `[serious]` `[skeptical]` `[surprised]` `[hopeful]` `[confident]` `[enthusiastic]` `[amused]` `[fascinated]` `[impressed]` `[passionate]`

**Delivery:** `[whispers]` `[pause]` `[slow]` `[rushed]`

**Natural sounds:** `[sighs]` `[chuckles]` `[clears throat]` `[exhales]`

Tags describe emotions, not actions — use `[warm]` not `[agreeing]`.

## ElevenLabs: Critical Decisions

### Use text_to_dialogue, not text_to_speech

The `text_to_speech` API speaks audio tags literally ("[curious]" becomes a spoken word) and produces mechanical delivery. The `text_to_dialogue` API understands conversation context, handles audio tags correctly, and produces natural speaker transitions. Quality difference is roughly 2x.

Generate full sections at once — not line by line. Full conversation context gives the model information about pacing and intonation that per-line generation lacks.

### Tuning parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `similarity_boost` | 0.95 | Faithful voice reproduction |
| `speed_adjustment` | -0.10 | Slightly slower = more natural pacing |
| `stability` | 0.3–0.5 | Lower = more expressive (varies by emotion) |
| `style` | 0.3–0.5 | Higher = more dramatic (varies by emotion) |

### Voice ID management

Use language-suffixed env vars: `VOICE_EMMA`, `VOICE_EMMA_DE`, `VOICE_EMMA_EN`. The `--lang` flag auto-selects the right suffix. One `.env` file holds all variants.

## Post-Processing

### Trim silences (after generation, before mixing)

`text_to_dialogue` produces excessive pauses (0.5–1.5s vs natural 0.2–0.4s). Trimming saves ~13% of audio length.

```bash
python generator/trim_silences.py input.mp3 output.mp3 --no-loudnorm
```

**Never loudnorm at this stage** — that's always the final step.

### Validate TTS output (after generation, before trim)

Qwen3-TTS can hallucinate — prepend extra text at the start of output. Always validate before continuing the pipeline, especially for Qwen-generated audio.

```bash
# Single file
python generator/validate_tts.py output.wav "The exact text you asked it to say"

# Batch via manifest (JSON array of {file, text} pairs)
python generator/validate_tts.py . --manifest manifest.json
```

The script transcribes output with Whisper and checks for:
- Extra words at start/end (hallucination)
- Low word overlap with expected text
- Suspicious duration (too long = runaway, too short = silent)

Exit code is 1 if any file is flagged — use in scripts to auto-retry.

**Mitigating Qwen hallucinations**: pass `temperature=0.7, repetition_penalty=1.2` to `generate_voice_clone()`.

### Add realism (after trim, before mixing)

Breaks the robotic sequential feel of TTS output:

```bash
# Preview effects without processing
python generator/add_realism.py input.mp3 --dry-run --seed 42

# Default settings (15% overlap, ±50-150ms jitter, pink noise room tone)
python generator/add_realism.py input.mp3 output.mp3 --seed 42

# More overlaps for heated debate
python generator/add_realism.py input.mp3 output.mp3 --overlap-chance 0.25 --seed 42

# With filler sounds (uh, mmhm, etc.)
python generator/add_realism.py input.mp3 output.mp3 --fillers-dir sounds/fillers/ --seed 42
```

| Effect | What | Default |
|--------|------|---------|
| Overlaps | Pull next speaker's start earlier so voices briefly collide | 15% of turns, 300-800ms |
| Jitter | Vary pause durations so timing isn't metronomic | ±50-150ms |
| Room tone | Synthetic pink noise underneath | On (use `--no-room-tone` to skip) |
| Fillers | Insert "uh", "mmhm" under long monologues | Off (needs `--fillers-dir`) |

Use `--seed` for reproducible results. The room tone here is lighter than the mastering chain's noise floor — they stack fine.

### Master (after mixing music, always last)

Full mastering chain with noise floor, compression, and loudness normalization:

```bash
ffmpeg -y -i input.wav \
  -filter_complex "
    anoisesrc=color=pink:sample_rate=24000:amplitude=0.003[noise];
    [0:a][noise]amix=inputs=2:weights=1 0.15:duration=first[mixed];
    [mixed]acompressor=threshold=-18dB:ratio=3:attack=5:release=100,loudnorm=I=-16:TP=-1.5:LRA=11[out]
  " \
  -map "[out]" -codec:a libmp3lame -b:a 192k master.mp3
```

| Step | What | Why |
|------|------|-----|
| Pink noise floor | `anoisesrc=color=pink:amplitude=0.003` mixed at 15% | Masks jarring digital silence between TTS clips, glues the episode together |
| Compression | `acompressor=threshold=-18dB:ratio=3` | Harmonizes levels across different speakers/TTS engines |
| Loudness norm | `loudnorm=I=-16:TP=-1.5:LRA=11` | Spotify/Apple podcast standard (-16 LUFS) |
| Bitrate | 192 kbps | Good quality for speech + music |

For ElevenLabs-only output (already well-leveled), you can skip the noise floor and compressor and just apply loudnorm:

```bash
ffmpeg -i final_mix.mp3 -af "loudnorm=I=-16:TP=-1.5:LRA=11" -codec:a libmp3lame -b:a 192k master.mp3
```

### Overlapping speech (advanced)

`text_to_dialogue` generates a single audio stream with sequential turns — speakers never actually talk over each other. For moments where you want genuine overlap (interruptions, reactive sounds under someone's turn, two people laughing together), you need to generate and mix separately.

**The hybrid approach:**

1. Generate the full episode with `text_to_dialogue` as usual — this gives you natural pacing and conversational context for 95% of the episode
2. Identify 2-3 moments where overlap would sell the scene (an excited interruption, a "mmhm" under a monologue, shared laughter)
3. Generate those specific lines separately — either with `text_to_speech` per line, or with Chatterbox on a GPU server
4. In your DAW, replace or layer those moments: pull the interrupting speaker's start earlier so they overlap the previous speaker's last words

**What works well for overlap:**
- Non-verbal sounds under a monologue: sighs, chuckles, "hmm" thinking sounds, exhales — these don't need conversational context so they sound natural even when generated separately
- Short reactive words: "Mmhm", "Right", "Wow", laughter
- Interruptions where one speaker cuts in on the last 1-2 words
- Two speakers reacting to a revelation at the same time

**What doesn't work:**
- Long simultaneous monologues — unintelligible, just like real life
- Overlapping every turn — exhausting to listen to, use sparingly

**DAW tips:**
- Crossfade the overlap region (50-100ms) to avoid clicks
- Lower the interrupted speaker's volume slightly in the overlap zone — the listener's ear follows the louder voice
- Keep overlaps short (0.5-1.5s) — just enough to break the sequential feel

Use this selectively. Two or three well-placed overlaps per episode add more realism than overlapping every other turn.

## Multilingual Scripts

When translating scripts to other languages:

1. **Audio tags stay in English** — `[curious]` not `[begeistert]` or `[nieuwsgierig]`
2. **Spell out numbers** for TTS — "nineteen twenty-two" not "1922"
3. **Use informal register** — podcasts are conversational (DE: "du/ihr" not "Sie")
4. **Use contractions** in English — "we'll" not "we will"
5. **Cross-reference** Claude translations with DeepL — merge the best of both
6. **DeepL quirks**: translates audio tags to target language (revert them), defaults to formal register (change it)

## Voice Design

Each voice needs a character brief covering:

- **Age and accent** — e.g., "Female, early 30s, soft Rhineland warmth"
- **Role** — host, expert, historical figure, narrator
- **Personality** — curious, authoritative, philosophical
- **Speech patterns** — animated for discoveries, measured for complex ideas
- **ElevenLabs settings** — Loudness %, Guidance %

The same character should sound like the same person across languages — same personality, different language. A Dutch accent on a historical Dutch figure is a feature, not a bug.

## FFmpeg (not pydub)

Python 3.13 removed the `audioop` module that pydub depends on. Use FFmpeg subprocess calls directly — it's faster, has no Python version constraints, and is the industry standard anyway.

## Qwen3-TTS Notes

- Excellent for English and German, no Dutch support
- Runs on GPU server (`source qwen-tts-env/bin/activate`)
- 1.7B model, needs ~9GB VRAM
- Voice cloning via ref audio + ref text transcript

### Critical: ref_text must match ref_audio exactly

Qwen3-TTS voice cloning requires the `ref_text` parameter to be the **actual transcript** of the reference audio. Mismatched text causes runaway generation where the model never emits a stop token.

**Solutions** (choose one):

1. **ASR transcription** (preferred, free): Transcribe existing voice refs with Whisper or Qwen3-ASR, then use the transcript as `ref_text`. Scripts: `generator/asr_whisper.py`, `generator/asr_qwen.py`.
2. **ElevenLabs generation**: Generate reference samples with known text via ElevenLabs, then use that exact text as `ref_text`. Script: `generator/elevenlabs/generate_qwen_refs.py`.

```bash
# Option 1: Transcribe existing refs (on GPU server)
source vox-env/bin/activate && python asr_whisper.py

# Option 2: Generate refs with ElevenLabs (run locally)
cd generator/elevenlabs && python generate_qwen_refs.py
scp qwen_refs/*.mp3 gpu-server:~/voice_refs/qwen_refs/
```

Always set `max_new_tokens=240` (~20s cap) as a safeguard against runaway generation.

### TTS pronunciation tips

- Use "Mondrian" not "Mondriaan" for English — TTS mispronounces the Dutch spelling
- Drop first names that TTS struggles with (e.g., "Piet") when context is clear
- Split long monologues into separate sentences with pauses for natural delivery

## Chatterbox Notes

- Excellent for English, poor for Dutch and German
- Runs on GPU server (`source vox-env/bin/activate`)
- Voice refs synced to the GPU server for cloning
- No audio tag support — emotion comes from voice reference selection only
- Clones speaking pace from reference audio — use refs with natural conversational pacing
- Includes its own mastering pipeline (compression + loudnorm at -16 LUFS)
