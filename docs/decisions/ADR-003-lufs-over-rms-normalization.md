# ADR-003: LUFS Over RMS for Loudness Normalization

**Date:** 2026-04-12
**Status:** Accepted

## Context

The pipeline normalized individual TTS lines using RMS (root mean square) — `audio * (target_rms / current_rms)`. This produced audibly inconsistent volume across speakers and line lengths because RMS doesn't account for perceptual loudness.

Two clips at the same RMS can sound very different in volume. A breathy whisper and a crisp consonant-heavy sentence have the same RMS at different perceived loudness. Short lines (1-3 words) were particularly affected — they'd get boosted to the same RMS as a 20-word sentence but sound quieter because less of the signal is speech energy.

## Decision

Replace RMS normalization with LUFS (Loudness Units relative to Full Scale) using pyloudnorm, the ITU-R BS.1770-4 standard for broadcast loudness measurement.

- **Target:** -16 LUFS per line (Spotify/Apple podcast standard)
- **Fallback:** RMS for clips < 0.5s (the LUFS meter's 400ms gating window is unreliable on very short audio)
- **Peak safety:** After normalization + reverb + speaker volume, cap at 0.9 linear to leave headroom for the mix-stage limiter

## Consequences

- Volume consistency improved across speakers — Zara no longer sounds louder than Morgan despite having a higher-F0 voice
- The mix-stage Pedalboard Limiter now barely activates (pre-limit peak dropped from 1.17 to 0.96)
- Per-line LUFS is not the same as program loudness — podcast platforms will still apply their own integrated loudness normalization on the final file. This is intentional: we normalize per-line for relative balance, platforms normalize the full episode for absolute level

## Alternatives Considered

- **Two-pass loudnorm (ffmpeg):** Rejected per production rules — squashes dynamics, makes quiet moments loud
- **RMS with perceptual weighting:** Possible but reinventing pyloudnorm worse
- **No per-line normalization:** Tried — speakers at different levels make the mix unusable
