# ADR-009: Context Sandwich — Prefix and Suffix for All Lines

**Date:** 2026-04-12
**Status:** Accepted
**Supersedes:** ADR-004 (which applied context-embedding only to short lines)

## Context

ADR-004 introduced context-embedding for short lines (≤4 words): prepend the previous same-speaker line, generate combined, extract target via Whisper. This solved voice drift on short lines.

But two problems remained on ALL lines:
1. **Onset artifacts** — Qwen3-TTS produces micro-glitches (clicks, hard attacks) in the first few milliseconds of every generated clip. The pipeline spent 70+ smoothing passes per episode trying to fix these.
2. **Tail cut-offs** — The last consonant of a line sometimes gets clipped, especially on short lines but also on normal-length lines where the model stops abruptly.

Insight: these are the same problem voice actors solve with "pick-up lines" and "reading through" — you never start or stop cold. You always have a run-up and a run-out.

## Decision

Extend context-embedding to ALL lines (not just short ones) using a **sandwich** approach:

```
Generated: [prefix context] ... [TARGET LINE] ... [suffix context]
                                 ^^^^^^^^^^^^^^
                                 extract this — clean onset AND clean tail
```

- **Prefix:** Previous same-speaker line (last 15 words). Onset artifacts land here.
- **Suffix:** Next same-speaker line (first 15 words). Tail artifacts land here.
- **Extraction:** Whisper word-level timestamps find the target boundaries.
- **Padding:** 30ms before first target word, 80ms after last target word.

If no same-speaker prefix exists (first line in section, different speaker before), only suffix is used. If neither exists, fall back to normal generation.

## Consequences

- **Onset artifacts eliminated at source** — no more `smooth_onset_glitches` needed as primary defense. The artifacts exist but they're in the prefix, which is discarded.
- **Tail cut-offs eliminated** — the model trails off into the suffix, not into silence.
- **Generation time increases ~2-3x** — each line generates a longer utterance. Acceptable for offline podcast production.
- **Whisper extraction runs for every line** — adds ~2s per line on GPU. The model is cached module-level.
- **Fallback is safe** — if Whisper extraction fails for any line, the pipeline falls back to multi-attempt generation (same as before).
- **Voice consistency improves across ALL lines** — not just short ones. Every line has prosodic runway.

## The Voice Actor Analogy

A voice actor never cold-reads a single line. They read the previous line under their breath, deliver the target, and let the next line carry them through. The director then edits the tape to extract just the performance.

We're doing exactly that: generate in context, extract the performance.
