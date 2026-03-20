# ADR-002: Local TTS for Agentic Engineering Series

**Date:** 2026-03-19
**Status:** Accepted

## Context

The Agentic Engineering podcast series needs a TTS engine for production. The podcast-generator project supports multiple engines: ElevenLabs (paid, cloud), Chatterbox (free, GPU), Qwen3-TTS (free, GPU), and TADA (untested, GPU).

ElevenLabs' `text_to_dialogue` API offers the best quality — it understands conversation context, emotional tags, and speaker transitions. However, it is a paid service with per-character pricing, and using it creates a dependency on an external vendor for every episode.

## Decision

The Agentic Engineering series uses **local TTS engines** (Chatterbox, Qwen3-TTS, or TADA) running on gpu-server. ElevenLabs is not used for this series.

### Engine priority

1. **Chatterbox** — proven in production (Vision at the Edge series), best English quality among local models. Default choice.
2. **Qwen3-TTS** — excellent English quality, also supports German. Use when Chatterbox quality is insufficient or German episodes are needed. Requires matched `ref_text` for voice cloning.
3. **TADA (HumeAI)** — untested. 1:1 token alignment claims zero hallucination. Ten languages. Evaluate before production use.

### Voice assignments

| Character | Voice ref | Notes |
|-----------|----------|-------|
| Lisa | `lisa.mp3` | ElevenLabs-designed synthetic voice |
| Marc | `felix.mp3` | Warm baritone, accent-free — better pace for non-native listeners than `narrator.mp3` (too fast) or `marc.mp3` (number articulation issues) |
| Sven | `sven.mp3` | ElevenLabs-designed synthetic voice |

### Script implications

Scripts follow ADR-001 (dialogue writing style for local TTS): no emotion tags, emotion through word choice and structure, overlap markers as stage directions stripped before generation.

## Consequences

### Positive
- Zero per-episode cost (GPU time only)
- No vendor dependency — episodes can be produced indefinitely
- Full control over timing, mixing, and post-production
- Scripts portable across engines without modification

### Negative
- No emotion tags — writing must carry the entire emotional load
- No `text_to_dialogue` conversation awareness — each line generated independently
- Overlap and interruptions require manual DAW mixing
- Voice quality may be slightly below ElevenLabs for some speakers

### Risks
- Chatterbox may struggle with technical jargon or tool names (CLAUDE.md, .mdc, etc.) — test before production
- TADA is untested — evaluate on a sample before committing to a full episode

## References

- podcast-generator ADR-001: Dialogue Writing Style for Local TTS Engines
- podcast-generator `docs/PRODUCTION_GUIDE.md` — engine-specific notes, mastering pipeline
- HumeAI TADA: https://huggingface.co/HumeAI/tada-3b-ml
