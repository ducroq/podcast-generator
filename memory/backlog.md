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

## Pipeline Consolidation from It Is Both Ep01 (2026-04-07) [RESOLVED]
- Closed 12 of 13 GitHub issues (#28-39) — only #40 (vision doc) remains
- 10 new modules: clean_audio, tts_overrides, mix_preprocess, assemble_intro, generate_backchannels, place_backchannels, analyze_voice, export_stems, plus write_script segmentation pass and mix_episode extensions
- Modified 4 modules: mix_episode (validation gate, sting, bleed, peak-limit, backchannels), validate_tts (duration anomaly, voice calibration), write_script (pacing, short lines, segmentation), chatterbox/generate_podcast (tuned defaults)
- Security audit: 7 findings fixed (path traversal, temp files, input validation)
- Code review: 5 must-fix resolved (clipping guard, ffmpeg returncode, stub clarity, sys.exit in library)
- UX review: 13 findings, key ones fixed (dry-run gaps, import paths, warnings, JSON errors)
- Tests: 284 → 486, all passing, ~11s
- Reference source: it_is_both/podcasts/production/ (episode-specific scripts extracted and generalized)

## Prosody Reference Library (2026-04-05) [RESOLVED]
- 25 Chatterbox-generated prosody refs on gpu-server (5 voices x 5 emotions)
- Voices: emma, felix, lisa, daan_en, sofie_en
- Emotions: excited, calm, emphatic, contemplative, urgent
- prosody_selector.py maps script emotion tags to closest ref
- A/B tested: 11/12 good on first try, 12/12 after one re-roll
- Issue #20 closed.
