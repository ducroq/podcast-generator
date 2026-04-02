# Backlog

## GPU Server Disk Space
- Server is at 98% (146/158GB). Need to free space before downloading new models.
- Candidates for cleanup: Qwen2.5-1.5B (2.9GB), multilingual-e5-large (2.2GB), gemma-3-1b-pt (1.9GB), faster-whisper-medium (1.5GB), paraphrase-multilingual-mpnet (1.1GB)
- Need ~4.2GB free for Qwen3-ASR-1.7B, or ~1.2GB for 0.6B variant

## Qwen3-ASR vs Whisper Comparison
- Goal: compare transcription quality to pick the best ASR for generating matched ref_text for Qwen3-TTS voice cloning
- Whisper (faster-whisper large-v3) already ran on all 26 English voice refs — results saved at `gpu-server:~/podcast-generator/asr_whisper_results.json`
- Qwen3-ASR blocked on disk space (see above)
- Scripts ready: `generator/asr_whisper.py`, `generator/asr_qwen.py`
- `qwen-asr` package installed in qwen-tts-env

## Qwen3-TTS Self-Ref Bootstrap (English voices)
- Goal: create matched ref_audio + ref_text pairs for all 29 voices so Qwen3-TTS can reliably clone them
- Pipeline: ASR transcribes original voice ref → use transcript + ref MP3 as matched pair → Qwen generates self-ref with that pair
- First attempt (mismatched ref_text) failed for ~half the voices — confirms matched text is essential
- Bootstrap script: `generator/qwen_bootstrap_refs.py` (needs updating once ASR transcripts are validated)
- Partial results in `gpu-server:~/voice_refs/qwen_self_refs/` (only ~half usable)

## German Mondriaan Episode (Qwen3-TTS)
- Generated and mastered once, but file got lost during cleanup
- ElevenLabs version still exists: `output/mondriaan_ep0/de/episode_0_de_mastered.mp3`
- Redo with Qwen3-TTS once self-refs are bootstrapped for German voices (emma_de, lucas_de, piet_de)
- Generation script ready: `output/generate_mondriaan_de_qwen.py`
