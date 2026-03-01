# Podcast Production Guide

Shared production knowledge distilled from the Mondriaan podcast project (7 episodes, 3 languages, 300+ listeners/episode).

## Pipeline

```
Script → Generate (ElevenLabs or Chatterbox) → Trim silences → Mix music in DAW → Master with FFmpeg
```

Each step is independent — you can retry any phase without redoing the others.

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

### Master (after mixing music, always last)

```bash
ffmpeg -i final_mix.mp3 -af "loudnorm=I=-16:TP=-1.5:LRA=11" -codec:a libmp3lame -b:a 192k master.mp3
```

| Parameter | Value | Why |
|-----------|-------|-----|
| I=-16 | -16 LUFS | Spotify/Apple podcast standard |
| TP=-1.5 | -1.5 dB true peak | Headroom, prevents clipping |
| LRA=11 | 11 LU range | Natural dynamics |
| 192k | 192 kbps | Good quality for speech + music |

Dynamic compression is usually unnecessary — ElevenLabs output is already well-leveled.

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

## Chatterbox Notes

- Excellent for English, poor for Dutch
- Runs on GPU server (RTX 4080), not locally
- Voice refs synced to `~/voice_refs/` on gpu-server
- No audio tag support — emotion comes from voice reference selection only
- Includes its own mastering pipeline (compression + loudnorm at -16 LUFS)
