# Backlog

## GPU Server Disk Space [RESOLVED]
- Cleaned from 99% to 78% (35GB free) on 2026-04-02
- Removed: pip cache (18GB), TADA models, unused Qwen 0.6B, whisper-large

## Qwen3-ASR vs Whisper Comparison [RESOLVED]
- Using faster-whisper base for all ref transcription — works well enough for Qwen ref_text matching
- Scripts still available: `generator/asr_whisper.py`, `generator/asr_qwen.py`

## Qwen3-TTS Self-Ref Bootstrap (English voices) [RESOLVED]
- Solved by transcribing existing refs with Whisper and trimming to clean sentence boundaries
- Trimmed refs on gpu-server: `~/voice_refs/*_qwen_ref.mp3` (10 voices done)
- No need for ElevenLabs-generated refs — Whisper transcription is accurate enough
- Remaining voices can be bootstrapped the same way when needed

## German Mondriaan Episode (Qwen3-TTS)
- Generated and mastered once, but file got lost during cleanup
- ElevenLabs version still exists: `output/mondriaan_ep0/de/episode_0_de_mastered.mp3`
- Redo with Qwen3-TTS once self-refs are bootstrapped for German voices (emma_de, lucas_de, piet_de)
- Generation script ready: `output/generate_mondriaan_de_qwen.py`
