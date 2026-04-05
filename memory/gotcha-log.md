# Gotcha Log

<!-- Structured problem/solution journal. Append-only.
     Part of the self-learning loop: Capture → Surface → Promote → Retire.

     PROMOTION LIFECYCLE:
     - New entries start here (Capture phase)
     - At end-of-session, review for patterns (Surface phase)
     - When an entry recurs 2-3 times, promote it to the relevant topic file
       as an "if X, then Y" pattern (Promote phase)
     - When a gotcha's root cause is fixed, mark it [RESOLVED] (Retire phase)
     - Track what you've promoted in the "Promoted" section below

     When the root cause is fixed, mark it resolved here (don't delete). -->

<!-- Template for new entries:

### [Short description] (YYYY-MM-DD)
**Problem**: What went wrong or was confusing.
**Root cause**: Why it happened.
**Fix**: What solved it.

-->

### Chatterbox venv path is not ~/vox-env (2026-04-01)
**Problem**: Scripts assume `source ~/vox-env/bin/activate` but chatterbox is actually at `~/podcast-generator/vox-env/bin/activate`. The `~/vox-env/` exists but is broken/empty.
**Root cause**: Chatterbox was installed inside a repo clone on gpu-server, not in the home directory.
**Fix**: Use `source ~/podcast-generator/vox-env/bin/activate` for chatterbox.

### TADA encoder + model OOM on 16GB GPU [RESOLVED] (2026-04-01)
**Problem**: Loading TADA encoder and model simultaneously exceeds 16GB VRAM on RTX 4080.
**Root cause**: Encoder (~4GB) + model (~9GB bf16) + ref audio processing don't fit together.
**Fix**: Two-phase workflow: load encoder, create prompt, save `.pt`, free encoder, then load model with cached prompt. Trim ref audio to max 5-8s and resample to 24kHz before encoding.

### TADA produces silent output for some voices [RESOLVED] (2026-04-01)
**Problem**: Marc voice produced completely silent `.wav` file on one of three test lines.
**Root cause**: Unknown — possibly ref audio characteristics or alignment failure.
**Fix**: No fix yet. Check output file size/duration after generation; re-generate if silent.

### TADA EncoderOutput has no .to() method [RESOLVED] (2026-04-01)
**Problem**: `EncoderOutput.load('file.pt').to(device)` throws AttributeError.
**Root cause**: EncoderOutput dataclass doesn't implement `.to()`.
**Fix**: Manually move tensors: iterate attributes, check for CPU tensors, move each to device.

### Qwen3-TTS runaway generation [RESOLVED] (2026-03-09)
**Problem**: Output files are dramatically longer than expected — minutes of garbage audio.
**Root cause**: Using raw transformers `AutoModelForCausalLM` instead of the `qwen-tts` package. Also ref_text/ref_audio mismatches.
**Fix**: Use `pip install qwen-tts` and `Qwen3TTSModel.from_pretrained()` with `generate_voice_clone()`. Still transcribe ref audio with Whisper and trim to clean sentence boundaries.

### Qwen3-TTS not loadable via transformers (2026-04-01)
**Problem**: `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-TTS-...")` fails with `KeyError: 'qwen3_tts'` even on transformers 5.5.0.dev0.
**Root cause**: Qwen3-TTS has its own package; it's not a standard transformers model.
**Fix**: `pip install qwen-tts` and use `from qwen_tts import Qwen3TTSModel`.

### Ember voice sounds distant in TADA [RESOLVED] (2026-04-01)
**Problem**: TADA ember voice cloning produces audio that sounds far away, like in another room.
**Root cause**: TADA faithfully clones the acoustic environment from the ref audio, not just the voice. The ember.mp3 ref has reverb/room ambience baked in.
**Fix**: Use dry, close-mic reference audio for TADA. Cleaning up refs with highpass+compression before encoding did not help (also caused OOM due to 48kHz resampling).

### qwen-tts-env is empty [RESOLVED] (2026-04-01)
**Problem**: `~/qwen-tts-env/` on gpu-server has no packages installed (only pip and setuptools).
**Root cause**: Unknown — env may have been wiped or never fully set up.
**Fix**: Rebuilt with `pip install qwen-tts` (which pulls transformers 4.57.3, torch, etc). Works correctly now.

### gpu-server disk nearly full [RESOLVED] (2026-04-01)
**Problem**: 158GB disk at 99% usage (2.3GB free). Can't install new packages.
**Root cause**: HuggingFace cache (33GB) + multiple venvs (~13GB) + pip cache (18GB).
**Fix**: Purged pip cache, removed unused HF models (TADA, whisper-large, Qwen 0.6B), deleted tada-env. Down to 78% (35GB free).

### Qwen3-TTS hallucinations — prepends extra text (2026-04-02)
**Problem**: Qwen sometimes prepends extra speech at the start of output that wasn't in the input text. Observed on Lucas voice (2 of 3 samples).
**Root cause**: Likely sampling randomness in the autoregressive generation. ICL mode (with ref_text) can leak reference content.
**Fix**: Three-layer mitigation:
1. **Prevention**: Use `temperature=0.7, repetition_penalty=1.2` (lower randomness)
2. **Detection**: Transcribe output with Whisper, compare against input text. Flag if output is >20% longer or contains unmatched words at start
3. **Recovery**: Re-generate flagged samples (different seed gives different result)

### add_realism.py was completely broken at runtime (2026-04-04)
**Problem**: Every run of `add_realism.py` with overlaps, jitter, or room tone failed with ffmpeg filter graph errors. Nobody noticed because the script was never tested end-to-end.
**Root cause**: Two bugs: (1) `anullsrc` is a source filter that can't be comma-chained with `atrim` — needs two separate filter lines. (2) `amix` weights used space separator instead of pipe (`|`). Both were present since the original code and survived our first review round (we introduced the `anullsrc` issue when replacing `aevalsrc`).
**Fix**: Extracted `_silence_pad()` helper with proper two-filter separation. Fixed `amix` weights to use pipe. Added end-to-end ffmpeg test in `test_add_realism.py::TestEndToEndFilterGraph`.

### speechmos package API different than documented (2026-04-04)
**Problem**: `speechmos.predict()` doesn't exist. The `speechmos` pip package contains DNSMOS/PLCMOS, not UTMOS.
**Root cause**: Confused the `speechmos` package with the `SpeechMOS` torch.hub model (tarepan/SpeechMOS). Different projects, similar names.
**Fix**: Use `torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong")` instead. Works correctly, verified on gpu-server.

### Dia2 single-pass TTS quality not competitive (2026-04-04)
**Problem**: Dia2 (nari-labs/Dia2-2B) produced MOS 3.13 with voice cloning, 2.28 without. Subjectively "really terrible" compared to Chatterbox (4.31) and Qwen (4.56).
**Root cause**: Single-pass dialogue models optimize for turn-taking dynamics at the expense of per-voice audio quality. The 2B model is too small to match dedicated per-voice engines.
**Fix**: No fix — this is a model limitation. Closed issue #4. Keep per-line pipeline with Chatterbox/Qwen/ElevenLabs. Revisit when models mature.

### Custom room tone [room] label was never valid ffmpeg syntax [RESOLVED] (2026-04-04)
**Problem**: `build_filter_complex()` referenced custom room tone as `[room]` in the filter graph, which isn't a valid ffmpeg input label. Only synthetic pink noise (anoisesrc) worked because it doesn't reference an input file.
**Root cause**: The room tone input was added as `-i room_tone.wav` but referenced with a made-up label `[room]` instead of `[1:a]`. Nobody caught it because custom room tone was never tested (only synthetic pink noise was used in practice).
**Fix**: Room tone now tracked via `room_tone_input_idx` and referenced as `[N:a]`. Input ordering is: [0]=main audio, [1]=room tone (if present), [2+]=fillers.

### Filler insertion was completely stubbed out [RESOLVED] (2026-04-04)
**Problem**: `plan_realism()` assigned filler files to actions, but `build_filter_complex()` never rendered them. The `filler_inputs` parameter was dead code. The dry-run output even showed "Filler insertions: N" but zero fillers appeared in the output.
**Root cause**: Original implementation planned fillers but the ffmpeg rendering was never completed. The `filler_input_offset` variable was initialized but unused.
**Fix**: After concatenation, each filler is added as an extra ffmpeg input, delayed to its absolute timeline position (30-70% through the turn), and mixed at 0.3 volume via `amix`.

### Emotion tags must be from EMOTIONAL_VARIANTS, not invented (2026-04-05)
**Problem**: Script used [quiet], [skeptical], [building] which aren't in `voice_settings.py` EMOTIONAL_VARIANTS. TTS would produce unpredictable results.
**Root cause**: Script writing (human or LLM) naturally invents emotion labels that sound right but aren't in the supported set.
**Fix**: Validate all tags against EMOTIONAL_VARIANTS before generation. `validate_script()` checks format but not tag validity — consider adding tag validation.

### LLM-generated "direct quotes" may be fabricated (2026-04-05)
**Problem**: Script included a Phil Rosenzweig quote presented as verbatim that was actually a paraphrase/synthesis of his argument. In an episode about epistemic honesty, this is especially bad.
**Root cause**: LLMs confidently generate plausible-sounding quotes from known authors. The text matches the author's style and argument but isn't a real passage.
**Fix**: Always reframe LLM-generated attributions as paraphrases ("his argument boils down to...") unless you can verify the exact quote. The source fidelity review agent caught this.

### Resemble Enhance degrades synthetic speech quality (2026-04-05) [RESOLVED]
**Problem**: Resemble Enhance denoiser is neutral on synthetic TTS, enhancer drops MOS by 0.08-0.18. First measurement showed catastrophic drop (3.1→1.3) but that was a UTMOS measurement artifact.
**Root cause**: UTMOS was measuring 44.1kHz upsampled audio without resampling to 16kHz first. At 16kHz, denoiser is neutral, enhancer slightly degrades. Synthetic speech is already clean — nothing to denoise.
**Fix**: Don't use Resemble Enhance on synthetic TTS output. Always resample to 16kHz before UTMOS measurement.

### ffmpeg sidechaincompress attack=200ms too slow for speech (2026-04-05) [RESOLVED]
**Problem**: Music bed ducking let the first syllable of every sentence punch through at full volume.
**Root cause**: `attack=200ms` is appropriate for music compression but way too slow for speech transients. Speech needs 5-20ms attack.
**Fix**: Changed to `attack=10ms`, `release=800ms`. Also removed double-attenuation (amix weights on top of sidechaincompress).

## Promoted

<!-- Track gotchas that have been promoted to topic files or the memory index.
     This helps you avoid re-promoting and shows the loop is working.

     STATUS TAGS:
     - [PROMOTED] — lesson was moved up the stack (to a topic file, memory index, or project file)
     - [RESOLVED] — root cause was fixed; entry stays as history
-->

| Entry | Promoted to | Date |
|-------|------------|------|
| ML package APIs are unreliable | CLAUDE.md hard constraint candidate | 2026-04-04 |
| ffmpeg filter graphs must be e2e tested | `tests/test_add_realism.py::TestEndToEndFilterGraph` | 2026-04-04 |
