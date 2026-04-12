# ADR-005: Resemblyzer as Automated Director

**Date:** 2026-04-12
**Status:** Accepted

## Context

TTS generation is probabilistic — each run produces different prosody, timing, and voice quality. The pipeline generates multiple attempts for short lines and needs to select the best one. Previously, selection was by longest duration (proxy for "more prosodic runway"). But longer isn't always better — a long take can drift from the character's voice.

In a recording studio, a director listens to takes and picks the one that sounds most like the character. We need an automated equivalent that doesn't require a human in the loop.

## Decision

Use resemblyzer (d-vector speaker embeddings) as the automated "director." For each attempt, compute cosine similarity against the speaker's voice reference. Pick the take with the highest score.

```python
score = cosine_similarity(
    encoder.embed_utterance(generated_audio),
    encoder.embed_utterance(reference_audio)
)
```

The directive is simple: **sound like yourself.**

### Integration points

1. **Multi-attempt selection:** Score each attempt, pick highest (not longest)
2. **Voice bank tracking:** Score every generated line, track the session best per speaker
3. **Quality metric in manifest:** `voice_score` field persisted for analysis

### Performance

- resemblyzer encoder loads once (module-level cache), runs on CPU
- ~50ms per scoring call (negligible vs. 3-10s TTS generation)
- Falls back to duration-based selection if resemblyzer unavailable

## Consequences

- Take selection is objective and consistent — no human taste needed
- Short lines that drift (0.4-0.6 similarity) get filtered out automatically
- The score is relative to the reference — if the reference itself is suboptimal, all scores are lower but relative ranking still works
- Enables the voice bank (ADR-006): best-scoring lines can become new references

## Alternatives Considered

- **UTMOS MOS scoring:** Measures audio quality, not voice identity. A high-quality wrong-voice take would score well
- **Human selection ("director mode"):** Accurate but blocks automation. Rejected per user requirement: "human out of the loop"
- **Duration-based (previous approach):** Correlation with quality exists but is weak. A 1.2s take at 0.92 similarity beats a 1.5s take at 0.71
- **Pitch/F0 matching:** Would catch register drift but not timbre drift. Resemblyzer captures both
