# ADR-004: Context-Embedding for Short TTS Lines

**Date:** 2026-04-12
**Status:** Accepted

## Context

Voice-cloning TTS models (Qwen3-TTS, Chatterbox) produce poor results on lines with ≤4 words. The model needs enough text to "settle into" the cloned voice — short lines don't give it enough prosodic runway. Measured impact:

- Voice similarity (resemblyzer) drops from ~0.85 mean to 0.41-0.65 on short lines
- Audio duration collapses (0.2-0.4s for 2-word lines, losing final consonants)
- Intonation drifts from character voice

This is the single biggest quality gap in the pipeline. 30 of 124 lines in Episode 1 are ≤4 words.

## Decision

For short lines (≤4 words), prepend the same speaker's previous line as prosodic context, generate the combined utterance, then use faster-whisper word-level timestamps to surgically extract just the target audio.

```
Previous line (same speaker): "The book doesn't update. It just sits there."
Target line:                   "Did they?"

Generated: "...It just sits there. ... Did they?"
                                         ^^^^^^^^ extracted via word timestamps
```

This is directly inspired by voice actors doing "pick-up lines" — when an actor flubs a word, they back up a full sentence and re-do the run-up. The context gives the voice box (or TTS model) time to settle into the right register.

### Extraction method

1. Count words in context text → boundary is at word N+1 in Whisper output
2. Whisper transcribes with `word_timestamps=True` → per-word start/end times
3. Slice audio from `word[N].start - 30ms` to `word[-1].end + 50ms`
4. Validate: extracted duration ≥ 0.2s, otherwise fall back to multi-attempt

### Fallback chain

```
context-embedding → multi-attempt (3x, pick best resemblyzer score) → single attempt
```

## Consequences

- Short lines get the full voice identity of longer lines — the model has settled by the time it reaches the target words
- Adds ~2s per short line for the Whisper extraction step (acceptable — runs on GPU alongside TTS)
- Requires same-speaker context (stops at section boundaries). First lines in a section or lines after a different speaker fall back to multi-attempt
- The separator `" ... "` can occasionally be transcribed as a word by Whisper — the word-count boundary method handles this by being slightly tolerant

## Alternatives Considered

- **Padding with the speaker's ref_text:** Works but produces unnatural intonation (the ref_text is a generic paragraph, not conversational)
- **Generating longer and trimming by duration:** Doesn't preserve word boundaries — cuts mid-phoneme
- **Using a different TTS engine for short lines:** Violates single-engine-per-episode rule (acoustic mismatch)
- **Expanding all short lines in the script:** Destroys dramatic rhythm ("Both?" carries weight that "Wait, both of them?" doesn't)
