# Podcast Generator

A methodology and toolkit for producing AI-generated podcasts that sound like real conversations, not robot lectures.

This project captures everything learned from producing a multilingual podcast series — from narrative design and character voice creation to audio generation and mastering. The tools are built for ElevenLabs and Chatterbox TTS, but the methodology applies to any engine.

## Case Study: Mondriaan de Denker

The methodology was developed and validated on a 7-episode educational podcast about Piet Mondriaan's philosophy:

- **7 episodes** across **3 languages** (Dutch, English, German)
- **300+ plays per episode**, growing organically through Spotify, Apple Podcasts, and YouTube
- **Top 10-25% globally** in engagement — 72-147% average consumption rate
- 100+ subscribers within the first months, zero paid promotion

The series proves that AI-generated audio can hold attention when the writing, voice design, and production are treated with the same care as traditional podcasting.

## The Methodology

Two guides capture the production knowledge:

- **[Narrative Design Guide](docs/NARRATIVE_DESIGN.md)** — How to write scripts that sound like real conversations: the three-voice model, emotional arcs, dialogue dynamics, and series continuity.
- **[Production Guide](docs/PRODUCTION_GUIDE.md)** — The technical pipeline from script to published episode: TTS engine selection, audio tag usage, silence trimming, mastering parameters, and multilingual workflows.

## The Toolkit

### TTS Engines

| Engine | Best for | Quality | Cost |
|--------|----------|---------|------|
| **ElevenLabs** | Multilingual, Dutch, German | Excellent | Paid API |
| **Chatterbox** | English podcasts | Excellent | Free (GPU required) |

ElevenLabs' `text_to_dialogue` API is the key — it understands conversation context and audio tags, producing natural speaker transitions that per-line generation can't match.

### CLI Tools

```bash
# Generate a full episode from a dialogue script
python generator/elevenlabs/generate_episode.py script.txt --lang de --output-dir output/

# Repair a single line without regenerating the whole episode
python generator/elevenlabs/generate_single_line.py "Emma: [excited] This is amazing!" fix_01

# Generate voice samples for cloning/comparison
python generator/elevenlabs/generate_voice_samples.py

# Trim excessive TTS pauses (works with any engine's output)
python generator/trim_silences.py input.mp3 output.mp3 --no-loudnorm

# Final master (always last step, after mixing music)
ffmpeg -i mix.mp3 -af "loudnorm=I=-16:TP=-1.5:LRA=11" -codec:a libmp3lame -b:a 192k final.mp3
```

### Script Format

```
==================================================
SEGMENT NAME
==================================================

Emma: [curious] So what happened when he moved to Paris?
Lucas: [building] Everything changed. He saw Cubism and thought...
Lucas: [excited] ...this is what I've been looking for my entire life.
```

Audio tags like `[curious]`, `[excited]`, `[warm]` control emotional delivery through ElevenLabs' dialogue API. See the [Production Guide](docs/PRODUCTION_GUIDE.md) for the full tag reference.

## Voice Library

All 30+ voices are **100% synthetic** — designed from scratch using ElevenLabs Voice Design v3, not cloned from real people.

Each voice starts as a character brief: age, accent, personality, speech patterns, and TTS engine settings. The same character sounds like the same person across languages — same personality, different language. A Dutch accent on a historical Dutch figure is a feature, not a bug.

Voice designs live in `voices/designs/` (see [TEMPLATE.md](voices/designs/TEMPLATE.md) for the methodology). Voice metadata and reference audio are tracked in `voices/voices.json`.

## Project Structure

```
podcast-generator/
├── docs/                            # Methodology guides
│   ├── NARRATIVE_DESIGN.md          # Script writing methodology
│   └── PRODUCTION_GUIDE.md          # Technical production pipeline
├── generator/                       # TTS engines and tools
│   ├── elevenlabs/                  # ElevenLabs TTS (multilingual)
│   │   ├── generate_episode.py      # Full episode generator
│   │   ├── generate_single_line.py  # Single-line repair tool
│   │   ├── generate_voice_samples.py
│   │   ├── test_voice.py            # Voice ID tester
│   │   ├── src/                     # Shared modules
│   │   └── .env.example             # Configuration template
│   ├── chatterbox/                  # Chatterbox TTS (English, GPU)
│   │   └── generate_podcast.py
│   └── trim_silences.py             # Post-processing: shorten pauses
├── voices/                          # Master voice library
│   ├── voices.json                  # Full metadata for all voices
│   ├── designs/                     # Voice design specs per character
│   └── *.mp3                        # Reference audio files
└── podcasts/                        # Per-podcast projects
    └── my-podcast/
        ├── dialogen/                # Dialogue scripts
        └── productie/               # Generated audio
```

## Getting Started

1. **Clone this repo** and install dependencies:
   ```bash
   cd generator/elevenlabs
   pip install -r requirements.txt
   ```

2. **Set up your API key** — copy `.env.example` to `.env` and add your ElevenLabs API key and voice IDs:
   ```bash
   cp .env.example .env
   ```

3. **Design your voices** — use the [voice design template](voices/designs/TEMPLATE.md) to create character briefs, then generate voices in ElevenLabs Voice Design

4. **Write a script** — follow the [Narrative Design Guide](docs/NARRATIVE_DESIGN.md) for structure and dialogue patterns

5. **Generate audio**:
   ```bash
   python generate_episode.py ../../podcasts/my-podcast/dialogen/script.txt \
       --output-dir ../../podcasts/my-podcast/productie
   ```

6. **Post-process** — trim silences, mix music in a DAW, then master with FFmpeg (see [Production Guide](docs/PRODUCTION_GUIDE.md))
