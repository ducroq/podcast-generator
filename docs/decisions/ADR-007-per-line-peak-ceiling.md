# ADR-007: Per-Line Peak Ceiling Over Global Limiting

**Date:** 2026-04-12
**Status:** Accepted

## Context

The processing chain applies multiple gain stages per line: LUFS normalization, speaker volume offset (+2.5 dB for Zara), and reverb. Combined, these can push peaks above 1.0 (clipping). The mix stage then applies a limiter — but if many lines are clipping, the limiter works too hard and squashes dynamics globally.

In v3 of Episode 1, 117 of 124 processed lines peaked above 0.95, with many hitting 1.0. The mix-stage Pedalboard Limiter had to pull the entire waveform down from 1.17 to 0.89 — a 2.4 dB global reduction that killed dynamics.

## Decision

Apply a per-line peak ceiling (0.9 linear, ~-0.9 dBFS) at the end of the per-line processing chain, after all gain stages and reverb. This is a simple linear scaling:

```python
peak = np.max(np.abs(audio))
if peak > 0.9:
    audio *= (0.9 / peak)
```

This leaves headroom for the mix stage (backchannel overlap, music bed bleed) so the final Pedalboard Limiter barely activates.

### Why 0.9?

- Leaves ~1 dB headroom below 0 dBFS
- Backchannel overlap can add up to 3 dB in a 200ms window
- The mix-stage limiter at -1 dBTP catches any remaining peaks
- Post-ceiling, mix-stage peak went from 1.17 to 0.96 — limiter removes only 0.6 dB

### Short-line fade interaction

Ultra-short clips (< 0.5s) get 3ms micro-fades instead of the full S-curve fade. This prevents the fade from eating the final consonant. Combined with the peak ceiling:

```
Normal line: full S-curve fade (pad + overlap) → peak ceiling at 0.9
Short line:  3ms micro-fade only → peak ceiling at 0.9
```

## Consequences

- No more audible clipping on any speaker (was affecting all three voices in v3)
- Mix-stage limiter barely activates — dynamics preserved
- Perceptual loudness slightly lower per-line (0.9 ceiling vs. uncapped) but LUFS normalization already set the loudness correctly — the ceiling just prevents peak overshoot
- Configurable via `processing.peak_ceiling` in podcast.yaml (default 0.9)

## Alternatives Considered

- **Limiter per line (Pedalboard):** More sophisticated but adds processing time. The gain stages in this pipeline are all linear — simple scaling is exact. A limiter would be needed if we had non-linear processing (compression, saturation)
- **Lower LUFS target (-20 instead of -16):** Fixes clipping but makes everything quieter. The ceiling approach preserves loudness for lines that don't need capping
- **No per-line ceiling, just better mix-stage limiting:** Tried — the Pedalboard Limiter handles transients well but when most of the input is above threshold, it becomes a global compressor. Per-line prevention is better than mix-stage cure
