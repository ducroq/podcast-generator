# Podcast Generator

<!-- agent-ready-projects: v1.6.0 -->

Methodology and toolkit for producing AI-generated podcasts. Shared voice library (30+ synthetic voices), multiple TTS engines (ElevenLabs, Chatterbox, TADA), and guides for narrative design and production.

## Before You Start

| When | Read |
|------|------|
| Writing or reviewing a script | `docs/NARRATIVE_DESIGN.md` — three-voice model, dialogue dynamics, emotional arcs |
| Generating audio or post-processing | `docs/PRODUCTION_GUIDE.md` — TTS engine choice, audio tags, silence trimming, realism, mastering |
| Creating a new voice | `voices/designs/TEMPLATE.md` — character brief format, ElevenLabs Voice Design settings |
| Adding a new podcast project | `README.md` — project structure and getting started |
| Stuck or debugging | `memory/gotcha-log.md` — problem-fix archive |
| Running generators on gpu-server | `docs/RUNBOOK.md` — venv paths, model loading, disk/VRAM constraints |
| Ending a session | Run `/curate` — review gotchas, promote patterns, update memory |

## Hard Constraints

- **Never commit `.env`** — it contains API keys and voice IDs. Use `.env.example` as reference.
- **Audio tags must be in English** — `[excited]` not `[begeistert]`. ElevenLabs requires this even in Dutch/German scripts.
- **Use `text_to_dialogue`, not `text_to_speech`** — quality difference is ~2x. Generate full sections, not line by line.
- **Never loudnorm before mixing music** — loudnorm is always the final step, after DAW mixing.
- **All voices are 100% synthetic** — designed via ElevenLabs Voice Design v3, not cloned from real people.
- **Spell out numbers for TTS** — "nineteen twenty-two" not "1922".

## Architecture

```
generator/elevenlabs/           → Primary TTS engine (multilingual, paid API)
  src/voice_settings.py         → Shared EMOTIONAL_VARIANTS, get_voice_settings(), parse_line()
  src/voice_config.py           → Voice ID mapping from .env
  src/dialogue_parser.py        → Script parser
generator/chatterbox/           → English-only TTS (free, requires GPU)
generator/tada/                 → TADA TTS (archived — env removed)
generator/whisper/              → Whisper STT (transcription)
generator/audio_utils.py        → Shared ffmpeg helpers (detect_silences, get_duration, get_sample_rate)
generator/quality_checks.py     → Optional quality checks (UTMOS MOS, speaker similarity, language ID)
generator/validate_tts.py       → Validation pipeline: ASR + quality checks → validation.json
generator/_transcribe_worker.py → Whisper subprocess worker (avoids code injection)
generator/add_realism.py        → Post-processing: overlaps, jitter, room tone
generator/trim_silences.py      → Silence trimming (loudnorm OFF by default)
generator/asr_*.py              → ASR comparison scripts (Whisper vs Qwen3-ASR)
generator/qwen_bootstrap_refs.py → Bootstrap matched refs for Qwen3-TTS
voices/                         → Master voice library (voices.json + designs/ + *.mp3)
podcasts/                       → Per-podcast projects (scripts + generated audio)
tests/                          → Test suite (80 tests, ~4s, no GPU needed)
docs/                           → Methodology guides
```

## Key Commands

```bash
# Generate episode
cd generator/elevenlabs
python generate_episode.py script.txt --lang de --output-dir output/

# Repair single line
python generate_single_line.py "Emma: [excited] This is amazing!" fix_01

# Trim silences (after generation, before mixing — loudnorm OFF by default)
python generator/trim_silences.py input.mp3 output.mp3

# Validate TTS output (always writes validation.json alongside audio)
python generator/validate_tts.py . --manifest manifest.json --language en --engine qwen
python generator/validate_tts.py . --manifest manifest.json --revalidate-flagged  # only re-check failures

# Master (after mixing, always last)
ffmpeg -i mix.mp3 -af "loudnorm=I=-16:TP=-1.5:LRA=11" -codec:a libmp3lame -b:a 192k final.mp3
```

## Quality Checks (on gpu-server)

Validation runs three tiers of checks (gracefully skips if dependencies not installed):

| Check | Package | What it catches | Threshold |
|-------|---------|----------------|-----------|
| ASR text comparison | faster-whisper | Hallucinations, missing/extra words | Word overlap < 70% |
| UTMOS MOS scoring | torch.hub (SpeechMOS) | Audio quality degradation, artifacts | < 3.5/5.0 |
| Speaker similarity | resemblyzer | Voice drift from reference | < 0.75 cosine sim |

Install on gpu-server: `pip install resemblyzer` (UTMOS loads via torch.hub automatically).

Results appear in `validation.json` under the `quality` field per entry.

## Testing

```bash
python -m pytest tests/ -v  # 80 tests, ~4 seconds, no GPU needed
```

Covers: audio_utils, voice_settings, hallucination detection, validation reports, add_realism filter graphs (end-to-end ffmpeg), trim_silences, full pipeline chain.

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

| Engine | English | Dutch | German | Voice Cloning | Verdict |
|--------|---------|-------|--------|---------------|---------|
| **Qwen3-TTS** | Excellent | No | Good | Yes | Best open-source overall (blind test winner, free, `pip install qwen-tts`) |
| **Chatterbox** | Excellent | Bad | - | Yes | Close second, very competitive on male voices |
| **TADA** | Good | No | Good | Yes | Fast (3-6x RT), no hallucination, but loses to both Qwen and Chatterbox |
| StyleTTS2 | Poor | - | - | Yes | Not recommended |
| Coqui XTTS | Good | Mediocre | - | Yes | Too many issues |
| ElevenLabs | Excellent | Good | Good | Yes | Best overall (paid) |
| NotebookLM | Excellent | Excellent | - | No | Google's podcast generator |

### Blind A/B Test Results (2026-04-01/02)

**TADA vs Chatterbox** (Lisa, Sven, Marc — 9 comparisons):
- **Chatterbox 8.5 — TADA 0.5** — Chatterbox dominates

**Qwen vs Chatterbox** (10 voices, 30 comparisons):
- **Chatterbox 17.5 — Qwen 12** — Chatterbox wins overall
- Qwen wins: Lisa (2.5–0.5), Zara (3–0), Sofie (2–1)
- Chatterbox wins: Emma (3–0), Lucas (3–0), Victoria (2.5–0.5), Felix (2.5–0.5), Ember (2–1), Marc (1.5–1)
- Tie: Sven (1.5–1.5)
- **Qwen hallucination issue**: prepends extra text on some samples (seen on Lucas)

### Recommended Engine Per Voice

Use different engines for different voices based on blind test results:

| Voice | Best Engine | Notes |
|-------|------------|-------|
| Lisa | Qwen | Strong preference |
| Zara | Qwen | Clean sweep |
| Sofie | Qwen | Slight edge |
| Sven | Either | Dead even |
| Emma | Chatterbox | Clean sweep |
| Lucas | Chatterbox | Clean sweep, Qwen hallucinated |
| Felix | Chatterbox | Strong preference |
| Victoria | Chatterbox | Strong preference |
| Ember | Chatterbox | |
| Marc | Chatterbox | Slight edge |

### Qwen3-TTS Notes

- **Install**: `pip install qwen-tts` (NOT via transformers directly)
- Uses `Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")`
- Voice cloning via `model.generate_voice_clone(text=..., language="English", ref_audio=..., ref_text=...)`
- **ref_text must match ref_audio** — transcribe with Whisper first, trim ref to clean sentence boundaries
- ~4GB VRAM, ~1-1.4x real-time on RTX 4080
- No Dutch support
- Previous runaway issues were caused by using raw transformers instead of the `qwen-tts` package
- **Hallucination risk**: may prepend extra text to output. Mitigate with `temperature=0.7`, `repetition_penalty=1.2`. Always validate output with ASR transcription check

### TADA Notes (archived — env removed)

- Lost blind test decisively, env deleted to free disk space
- Can be reinstalled with `pip install hume-tada` if needed for German
- 10 languages but no Dutch — English and German confirmed working
- No audio tags — expressiveness from reference clip only

## Post-Processing Pipeline

```
Generate → Validate (ASR) → Trim silences → Add realism → Mix music in DAW → Master with FFmpeg
```

Validation runs automatically after generation. Each output directory gets a `validation.json` with per-line ASR results. Flagged lines should be re-generated before proceeding. Use `--revalidate-flagged` to only re-check previously failed lines.

### add_realism.py (after trim_silences, before mixing)

Automatically adds natural podcast feel to generated audio:
- Randomly overlaps speaker turns (~15% of turns, 300-800ms)
- Jitters pause timing (±50-150ms) to break metronomic feel
- Synthetic pink noise room tone underneath
- Optional filler sounds ("uh", "mmhm") from audio directory

```bash
# Preview what will happen
python generator/add_realism.py input.mp3 --dry-run --seed 42

# Process
python generator/add_realism.py input.mp3 output.mp3 --seed 42

# Tune: more overlaps for heated debate, less for calm interview
python generator/add_realism.py input.mp3 output.mp3 --overlap-chance 0.25 --seed 42
```

## Workflow

- **English podcasts (local)**: Qwen3-TTS or Chatterbox on gpu-server (use per-voice recommendation table above)
- **German podcasts**: Qwen3-TTS or ElevenLabs on gpu-server
- **Dutch podcasts**: ElevenLabs or NotebookLM
- **gpu-server access**: `ssh gpu-server` (Tailscale, user: hcl)
