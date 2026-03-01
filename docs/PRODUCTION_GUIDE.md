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

## Chatterbox Notes

- Excellent for English, poor for Dutch
- Runs on a GPU server, not locally
- Voice refs synced to the GPU server for cloning
- No audio tag support — emotion comes from voice reference selection only
- Includes its own mastering pipeline (compression + loudnorm at -16 LUFS)
