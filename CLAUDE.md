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

## Voice Library (100% Synthetic, 30 voices)

Voice IDs are configured in `generator/elevenlabs/.env` with language suffixes: `VOICE_EMMA`, `VOICE_EMMA_DE`, `VOICE_EMMA_EN`. The `--lang` flag auto-selects the right variant.

Every voice needs a character design (`voices/designs/`) before generation. Same character, same personality across languages — only the language changes.

| Voice | Description | Project | Source |
|-------|-------------|---------|--------|
| alex.mp3 | Male host, warm, mid-Atlantic accent | It Is Both | ElevenLabs v3 |
| alex-dutch.mp3 | Male host, Dutch-European accent variant | It Is Both | ElevenLabs v3 |
| alex-english.mp3 | Male host, neutral English variant | It Is Both | ElevenLabs v3 |
| daan_en.mp3 | Male, English with subtle Dutch accent, news anchor | ovr.news | ElevenLabs v3 |
| daan_nl.mp3 | Male, warm Dutch news anchor (ABN) | ovr.news | ElevenLabs v3 |
| ember.mp3 | Female, Southern US accent | General | ElevenLabs |
| emma.mp3 | Female, Dutch accent | General | ElevenLabs |
| emma_english_sample.mp3 | Female, English bilingual sample | General | ElevenLabs |
| emma_nl_sample.mp3 | Female, Dutch bilingual sample | General | ElevenLabs |
| felix.mp3 | Male, warm baritone, wonder narrator, accent-free | Sol Invictus | ElevenLabs v3 |
| hugo.mp3 | Male, enthusiastic and professional | General | ElevenLabs v3 |
| jann.mp3 | Male, calm, confident and natural | General | ElevenLabs v3 |
| lisa.mp3 | Female host, warm, clear English | General | ElevenLabs |
| lucas.mp3 | Male, alternative host voice | General | ElevenLabs v3 |
| lucas_english_sample.mp3 | Male, English bilingual sample | General | ElevenLabs v3 |
| lucas_nl_sample.mp3 | Male, Dutch bilingual sample | General | ElevenLabs v3 |
| marc.mp3 | Male expert, calm, Dutch accent | General | ElevenLabs |
| narrator.mp3 | Male, neutral, clear English | General | Chatterbox native |
| oma.mp3 | Female, 55+, wise Dutch storyteller | Busara | ElevenLabs v3 |
| piet_english_sample.mp3 | Male, English bilingual sample | General | ElevenLabs v3 |
| piet_nl_sample.mp3 | Male, Dutch bilingual sample | General | ElevenLabs v3 |
| professor.mp3 | Male, 50+, measured academic, European accent | Digital Engineers | ElevenLabs v3 |
| ruth.mp3 | Female, friendly children's storyteller | Busara | ElevenLabs v3 |
| serafina.mp3 | Female, sensual and expressive | General | ElevenLabs v3 |
| serge.mp3 | Male, professional Dutch narrator | General | ElevenLabs v3 |
| sofie_en.mp3 | Female, professional anchor, subtle Dutch accent | ovr.news | ElevenLabs v3 |
| sofie_nl.mp3 | Female, professional Dutch news anchor | ovr.news | ElevenLabs v3 |
| sven.mp3 | Male student, skeptical, Dutch | General | ElevenLabs |
| victoria.mp3 | Female, British accent | General | ElevenLabs |
| zara.mp3 | Female, 23-28, energetic, accent-free | Grad Career | ElevenLabs v3 |

## TTS Engines (Tested)

| Engine | English | Dutch | Voice Cloning | Verdict |
|--------|---------|-------|---------------|---------|
| **Chatterbox** | Excellent | Bad | Yes | Best open-source for English |
| **Qwen3-TTS** | Excellent | - | Yes | Free, requires GPU, needs matched ref_text |
| **TADA** | Untested | - | Yes | HumeAI, 1:1 alignment, no hallucination claim |
| StyleTTS2 | Poor | - | Yes | Not recommended |
| Coqui XTTS | Good | Mediocre | Yes | Too many issues |
| ElevenLabs | Excellent | Good | Yes | Best overall (paid) |
| NotebookLM | Excellent | Excellent | No | Google's podcast generator |

## Workflow

- **English podcasts (local)**: Chatterbox or Qwen3-TTS on gpu-server (free, GPU)
- **Dutch podcasts**: ElevenLabs or NotebookLM
- **gpu-server access**: `ssh gpu-server` (Tailscale, user: hcl)
