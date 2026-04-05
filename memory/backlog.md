# Backlog

## Bug Sweep (2026-04-04) [RESOLVED]
- Closed 10 issues in one session: #1-3 (quality checks already done), #5 (filler rendering), #6 (room tone duration), #7 (overlap default), #8 (emotion tags), #9 (stereo/mono), #10 (unknown speaker warning), #13 (section regex)
- Also fixed: custom room tone [room] label (pre-existing bug), room tone input overwriting filler inputs
- Tests: 80 → 104, all passing

## GPU Server Disk Space [RESOLVED]
- Cleaned from 99% to 78% (35GB free) on 2026-04-02
- Removed: pip cache (18GB), TADA models, unused Qwen 0.6B, whisper-large
- Dia2 installed and removed same day (2026-04-04)

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

## Script Generation Pipeline [RESOLVED]
- Implemented `generator/write_script.py` — 7-pass LLM pipeline (extract → draft → director → pronunciation → review → revise)
- Review pass: 3 parallel perspectives (source fidelity, target listener, narrative design)
- First episode produced: `podcasts/it-is-both/script_ep01_the_formula.txt` (Alex/Felix/Zara, ~14 min, 10/12 narrative design)
- Character definitions: `podcasts/it-is-both/characters/` with signature phrases and never-does rules
- 49 tests for write_script, 153 total. Issues #22 and #11 closed.

## Audio Pipeline Completion (2026-04-05) [RESOLVED]
- mix_episode.py: LUFS leveling, intro/outro crossfade, music bed ducking, loudnorm mastering
- Review found and fixed: sidechain ducking bugs (#25), cleanup on error (#26), edge cases (#27)
- master.py: Pedalboard DSP chain (highpass, EQ, compressor, gate, limiter, LUFS). Issue #15 closed.
- Auphonic (#16) superseded by mix_episode.py — closed.
- Resemble Enhance (#14) tested on gpu-server: neutral on denoiser, enhancer degrades quality at 4.4+ MOS. Closed.
- Breath insertion (#17): synthetic band-passed pink noise with inhale/exhale envelopes in add_realism.py
- Content-aware backchannels (#18): --script flag for placement based on dialogue content (questions, emotions, turn length)
- Show-don't-tell (#24): NARRATIVE_DESIGN.md updated + all write_script.py prompts updated
- It Is Both project files moved to it_is_both repo
- Tests: 104 → 218

## Dia2 Single-Pass TTS Evaluation [RESOLVED]
- Evaluated 2026-04-04: MOS 3.13 (voice cloned) vs our 4.3-4.5 — not competitive
- User confirmed: "really terrible" compared to Chatterbox/Qwen/ElevenLabs
- Removed from gpu-server. Issue #4 closed.

## Publish Pipeline (2026-04-05) [RESOLVED]
- publish.py: chapters.json (Podcasting 2.0), transcript.srt (speaker-labeled), show_notes.md
- Timestamps from cumulative section audio durations, proportional word-count per line
- Issues closed: #19 (publish), #23 (spectral matching — not needed)
- Tests: 218 → 270

## Emotional Pacing Framework (2026-04-05) [RESOLVED]
- NARRATIVE_DESIGN.md: tension mapping, breathing patterns, anticipation beats, peak-end rule
- Quality checklist reorganized into pacing/dialogue/content categories
- Issue #21 closed.

## LLM Pronunciation Pass (2026-04-05) [RESOLVED]
- write_script.py pass 4: replaces foreign proper nouns with phonetic respellings for TTS
- Always-on, skip with --no-pronunciation
- Driven by ovr.news international content needs
- Issue #12 closed (reframed from MFA to LLM approach).

## Prosody Reference Library
- Issue #20 still open — needs gpu-server session to generate emotion-tagged ref clips
- Plan: ElevenLabs generates emotion variants per voice → SCP to gpu-server → Qwen/Chatterbox use as refs
- Blocked by Ollama holding GPU VRAM
