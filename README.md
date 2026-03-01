# Podcast Generator

Multi-project podcast production hub with shared TTS engines and a master voice library.

## Structure

```
podcast-generator/
├── generator/                    # TTS engines and shared tools
│   ├── chatterbox/               # Chatterbox TTS (GPU, open-source, English-only)
│   │   └── generate_podcast.py   # Multi-speaker podcast generator
│   ├── elevenlabs/               # ElevenLabs TTS (paid API, multilingual)
│   │   ├── generate_episode.py   # Full episode generator (text_to_dialogue)
│   │   ├── generate_single_line.py  # Single-line repair tool
│   │   ├── generate_voice_samples.py  # Voice sample generator for cloning
│   │   ├── test_voice.py         # Voice ID tester
│   │   ├── src/                  # Shared modules (parser, voice config)
│   │   ├── requirements.txt
│   │   └── .env                  # API key + voice IDs
│   ├── trim_silences.py          # Shared: shorten pauses in any MP3
│   └── cleanup_server.sh         # GPU server maintenance
├── voices/                       # Master voice library (30+ synthetic voices)
│   ├── voices.json               # Full metadata for all voices
│   ├── designs/                  # Voice design specs (per language)
│   ├── clone_tests/              # TTS engine comparison samples
│   └── *.mp3                     # Reference audio files
└── podcasts/                     # Per-podcast projects
    ├── vision-at-the-edge/       # AI/tech education podcast
    │   ├── PODCAST_GUIDE.md
    │   ├── dialogen/             # Dialogue scripts
    │   ├── onderzoek/            # Research notes
    │   └── productie/            # Generated audio
    ├── ovr-news/                 # News podcast (scaffold)
    └── sol-invictus/             # Narrative podcast (scaffold)
```

## TTS Engines

| Engine | Best for | Quality | Cost |
|--------|----------|---------|------|
| **Chatterbox** | English podcasts | Excellent | Free (GPU required) |
| **ElevenLabs** | Multilingual, Dutch | Excellent | Paid API |

### Chatterbox (runs on gpu-server)

```bash
# Copy script to server, generate, copy back
scp script.txt gpu-server:~/
ssh gpu-server "cd ~/chatterbox && python generate_podcast.py ~/script.txt -o ~/output.mp3"
scp gpu-server:~/output.mp3 .
```

### ElevenLabs

```bash
cd generator/elevenlabs
pip install -r requirements.txt

# Generate full episode
python generate_episode.py ../../podcasts/vision-at-the-edge/dialogen/script.txt

# Generate with language variant and custom output
python generate_episode.py script.txt --lang de --output-dir ../../podcasts/mondriaan/productie

# Repair a single line
python generate_single_line.py "Emma: [excited] This is amazing!" fix_01

# Test a new voice ID
python test_voice.py script.txt "OPENING" --lucas "new_voice_id"
```

### Post-processing

```bash
# Trim excessive pauses (works with any engine's output)
python generator/trim_silences.py input.mp3 output.mp3 --no-loudnorm

# Final master (always last step, after mixing music)
ffmpeg -i mix.mp3 -af "loudnorm=I=-16:TP=-1.5:LRA=11" -codec:a libmp3lame -b:a 192k final.mp3
```

## Adding a New Podcast

1. Create a folder under `podcasts/`:
   ```
   podcasts/my-podcast/
   ├── dialogen/     # Scripts
   └── productie/    # Output audio
   ```
2. Write dialogue scripts in `Speaker: [emotion] Text` format
3. Generate with either engine, pointing `--output-dir` to your `productie/` folder

## Voice Library

All 30+ voices are 100% synthetic (ElevenLabs Voice Design v3). See `voices/voices.json` for full metadata. Voice reference files are synced to gpu-server at `~/voice_refs/` for Chatterbox use.
