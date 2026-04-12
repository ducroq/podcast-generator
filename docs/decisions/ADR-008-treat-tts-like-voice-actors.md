# ADR-008: Treat TTS Engines Like Voice Actors, Not Machines

**Date:** 2026-04-12
**Status:** Accepted

## Context

When TTS-generated podcast lines sound wrong — collapsed short phrases, voice drift, clipped consonants — the instinct is to treat it as a model quality problem. Tune parameters. Retry. Switch engines. Force it to be correct.

But the failures aren't random. They follow a pattern that any recording engineer would recognize: the model is being asked to cold-read a two-word line, in perfect character voice, with no warm-up, no context, and no second opinion. A human voice actor given the same conditions would fail the same way.

This ADR establishes a production philosophy: **treat TTS engines with the same empathy and technique you'd use with a voice actor.** Give them what they need to perform. Then pick their best work.

## Decision

All TTS pipeline design decisions should be evaluated through the lens: "Would this help a voice actor?" If the answer is yes, the technique probably helps the TTS model too.

### The mapping

| Voice Actor Practice | Pipeline Implementation | ADR |
|---|---|---|
| **Warm-up before a session** — actors do throwaway lines to find the character voice | Use the best previous output as the voice reference, not a cold seed clip. The model "hears itself" at its best before starting. | ADR-006 |
| **Pick-up lines** — actors back up a full sentence when re-doing a tricky word | Prepend same-speaker context before short lines, generate the combined utterance, extract just the target via word-level timestamps. | ADR-004 |
| **Director picks the best take** — multiple takes, objective selection | Generate multiple attempts, score each with resemblyzer cosine similarity against the character reference. Pick the best. | ADR-005 |
| **Performance improves across sessions** — actors get better at a role over time | Persist the best-scoring output per speaker in a voice bank. Each episode starts better than the last. | ADR-006 |
| **Don't record in isolation** — ensemble energy, conversational rhythm | Same-speaker context gives the model conversational flow to build on, not a cold start from silence. | ADR-004 |
| **Monitor levels, don't clip** — engineer manages headroom per take, not just at mixdown | Per-line peak ceiling after all gain stages, so the mix-stage limiter barely activates. | ADR-007 |
| **Match the room** — consistent acoustic environment across takes | LUFS normalization ensures perceptual loudness consistency, not just signal amplitude matching. | ADR-003 |

### The reframe

The model isn't broken. The production was.

A voice-cloning TTS model generating "Both?" has:
- 0.5 seconds of audio to work with
- No prior conversational context
- A static reference clip recorded in different conditions
- One chance to nail the voice, the emotion, and the intonation

Of course it fails. Give it a 15-word run-up, a reference that matches what it actually produces, three attempts with an objective judge, and it delivers.

## Consequences

- Pipeline complexity increases (context lookup, Whisper extraction, voice bank) — but each piece solves a specific, measurable problem
- Generation time increases ~30% (multi-attempt + context-embedding for short lines) — acceptable for offline podcast production
- Human is removed from the quality loop — resemblyzer replaces the director's ear
- The approach generalizes: any voice-cloning TTS engine would benefit from these techniques, not just Qwen3-TTS
- **Opens a category:** "TTS production engineering" as a practice distinct from model training or prompt engineering

## Article Outline

This ADR and ADRs 003-007 form the basis for a technical article:

**Title:** "What if we stopped fighting TTS and started directing it?"

**Angle:** Empathy toward TTS engines. The same techniques that make human voice actors sound good — warm-up, context, multiple takes, a director — make TTS sound good too. The model isn't the problem; the production pipeline is.

**Structure:**

1. **The problem** — short TTS lines sound terrible (voice drift data, collapsed audio examples)
2. **The wrong instinct** — parameter tuning, engine switching, brute-force retries
3. **The reframe** — a voice actor would fail under the same conditions
4. **The techniques** — one section each for context-embedding, resemblyzer scoring, voice bank, peak management
5. **Results** — before/after resemblyzer scores, A/B audio comparison
6. **The generalization** — this applies to any voice-cloning TTS, not just our stack
7. **The category** — "TTS production engineering" deserves its own practice

**Audience:** Developers building TTS pipelines, podcast producers using synthetic voices, TTS researchers interested in production-side quality improvements.
