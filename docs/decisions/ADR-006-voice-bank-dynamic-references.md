# ADR-006: Voice Bank — Dynamic References Across Episodes

**Date:** 2026-04-12
**Status:** Accepted

## Context

Voice-cloning TTS uses a reference audio clip to define "what the speaker sounds like." Our static references were recorded once — they're good but not optimal for what Qwen3-TTS actually produces. The model's output has its own acoustic fingerprint (codec artifacts, specific frequency response) that differs from the original voice design clips.

Observation: when we scored all generated lines with resemblyzer, the best lines scored 0.93-0.96 against the static reference. But the static reference itself, if scored against its own TTS output, is only a ~0.84 match. The model's own best output is a better template for the model to reproduce than the external reference.

This mirrors how voice actors warm up: after a few takes, they've "found" the character. You don't keep going back to the audition tape — you use the best performance from today.

## Decision

Implement a persistent voice bank that tracks the best-scoring TTS output per speaker per episode. The best reference improves over time:

```
Episode 1: static ref (seed)     → best line scores 0.94 → saved
Episode 2: ep01 best (0.94) as ref → best line scores 0.96 → saved
Episode 3: ep02 best (0.96) as ref → convergence on "what this voice sounds like in this podcast"
```

### Voice bank structure

```
voice_refs/dynamic/
  voice_bank.json       ← index: {speaker: {score, file, text, episode, hash}}
  alex_ep01_best.wav    ← best-scoring Alex line from ep01
  morgan_ep01_best.wav
  zara_ep01_best.wav
```

### Selection logic

For each line during generation:
1. Check voice bank for a dynamic reference (best from previous sessions)
2. If available AND score > threshold (0.85): use it as voice_ref + ref_text
3. If not: fall back to static reference from podcast.yaml

### Update logic

After each successful generation:
1. Score the line against the static reference (not the dynamic one — consistent baseline)
2. If score > session best for that speaker AND duration > 1.5s: track as candidate
3. After all generation completes: save session bests that exceed the persisted index

### Guardrails

- **min_score: 0.85** — don't save mediocre lines as references
- **min_duration: 1.5s** — short lines are unreliable voice templates (not enough signal)
- **Score against static ref:** Always compare to the original seed, not to the dynamic ref (prevents drift accumulation)

## Consequences

- Voice consistency improves across episodes without manual intervention
- First few lines of each episode benefit from previous episode's learning
- The voice "converges" — later episodes sound more like one consistent character
- If a speaker's voice drifts badly in one episode (bad day for the model), the bank won't save it (score threshold protects)
- Bank is additive — never deletes old refs, only overwrites with better ones

## Alternatives Considered

- **Fixed reference forever:** Safe but leaves quality on the table. The static ref is a ceiling that real output regularly exceeds
- **Use most recent line as ref (chaining):** Dangerous drift — if line N is slightly off, lines N+1 through N+100 inherit the drift
- **Human-curated reference library:** Accurate but requires manual work each episode. Contradicts "human out of the loop" goal
- **Ensemble of references:** Use top-3 refs and average embeddings. More complex, marginal benefit over single-best for this corpus size
