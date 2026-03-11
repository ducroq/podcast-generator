# Podcast Generator

Methodology and toolkit for producing AI-generated podcasts. Shared voice library (30+ synthetic voices), ElevenLabs and Chatterbox TTS engines, and guides for narrative design and production.

## Before You Start

| When | Read |
|------|------|
| Writing or reviewing a script | `docs/NARRATIVE_DESIGN.md` — three-voice model, dialogue dynamics, emotional arcs |
| Generating audio or post-processing | `docs/PRODUCTION_GUIDE.md` — TTS engine choice, audio tags, silence trimming, mastering |
| Creating a new voice | `voices/designs/TEMPLATE.md` — character brief format, ElevenLabs Voice Design settings |
| Adding a new podcast project | `README.md` — project structure and getting started |

## Hard Constraints

- **Never commit `.env`** — it contains API keys and voice IDs. Use `.env.example` as reference.
- **Audio tags must be in English** — `[excited]` not `[begeistert]`. ElevenLabs requires this even in Dutch/German scripts.
- **Use `text_to_dialogue`, not `text_to_speech`** — quality difference is ~2x. Generate full sections, not line by line.
- **Never loudnorm before mixing music** — loudnorm is always the final step, after DAW mixing.
- **All voices are 100% synthetic** — designed via ElevenLabs Voice Design v3, not cloned from real people.
- **Spell out numbers for TTS** — "nineteen twenty-two" not "1922".

## Architecture

```
generator/elevenlabs/     → Primary TTS engine (multilingual, paid API)
generator/chatterbox/     → English-only TTS (free, requires GPU)
generator/whisper/        → Whisper STT (transcription)
generator/asr_*.py        → ASR comparison scripts (Whisper vs Qwen3-ASR)
generator/qwen_bootstrap_refs.py → Bootstrap matched refs for Qwen3-TTS
generator/trim_silences.py → Post-processing (works with any engine)
voices/                   → Master voice library (voices.json + designs/ + *.mp3)
podcasts/                 → Per-podcast projects (scripts + generated audio)
docs/                     → Methodology guides
```

## Key Commands

```bash
# Generate episode
cd generator/elevenlabs
python generate_episode.py script.txt --lang de --output-dir output/

# Repair single line
python generate_single_line.py "Emma: [excited] This is amazing!" fix_01

# Trim silences (after generation, before mixing)
python generator/trim_silences.py input.mp3 output.mp3 --no-loudnorm

# Master (after mixing, always last)
ffmpeg -i mix.mp3 -af "loudnorm=I=-16:TP=-1.5:LRA=11" -codec:a libmp3lame -b:a 192k final.mp3
```

## Voice Library

Voice IDs are configured in `generator/elevenlabs/.env` with language suffixes: `VOICE_EMMA`, `VOICE_EMMA_DE`, `VOICE_EMMA_EN`. The `--lang` flag auto-selects the right variant.

Every voice needs a character design (`voices/designs/`) before generation. Same character, same personality across languages — only the language changes.
