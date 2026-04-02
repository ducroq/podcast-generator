# Gotcha Log

<!-- Structured problem/solution journal. Append-only.
     Part of the self-learning loop: Capture > Surface > Promote > Retire. -->

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

## Promoted

| Entry | Promoted to | Date |
|-------|------------|------|
