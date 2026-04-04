# Runbook

## Principles

- **Ref audio quality determines output quality** — use dry, close-mic references. TADA and Qwen clone the room, not just the voice.
- **One engine per venv** — TTS engines have conflicting dependencies (especially transformers versions). Never mix them.
- **Cache voice prompts** — TADA encoder is too large to keep loaded alongside the model on 16GB. Encode once, save `.pt`, reuse.
- **Always verify output** — check file duration after generation. Silent or runaway outputs happen silently.

## gpu-server Access

```bash
ssh gpu-server  # Tailscale, user: hcl, RTX 4080 16GB
```

## Virtual Environments

| Engine | venv | Activate |
|--------|------|----------|
| Qwen3-TTS | `~/qwen-tts-env/` | `source ~/qwen-tts-env/bin/activate` |
| Chatterbox | `~/podcast-generator/vox-env/` | `source ~/podcast-generator/vox-env/bin/activate` |
| Whisper (faster-whisper) | `~/podcast-generator/vox-env/` | Same as Chatterbox |

**Note:** TADA env was removed (2026-04-01) to free disk. Qwen uses `qwen-tts` package (not raw transformers).

## Voice References

```bash
ls ~/voice_refs/          # Master voice refs (synced from local)
ls ~/voice_refs/qwen_refs/ # ElevenLabs-generated refs with known transcripts
```

Sync new refs: `scp voices/*.mp3 gpu-server:~/voice_refs/`

## Validating TTS Output

Always validate after Qwen generation (hallucination risk). Run on gpu-server (needs faster-whisper).
Validation always writes `validation.json` alongside the audio — no opt-in needed.

```bash
source ~/podcast-generator/vox-env/bin/activate

# Single file
python generator/validate_tts.py output.wav "Expected text here" --engine qwen

# Batch: create manifest.json with [{"file": "out1.wav", "text": "expected"}, ...]
python generator/validate_tts.py . --manifest manifest.json --language en --engine qwen

# After re-generating flagged files, only re-check the failures:
python generator/validate_tts.py . --manifest manifest.json --revalidate-flagged

# Exit code 1 = flagged files. Reports saved to validation.json in the audio directory.
```

### Validation report format

Each `validation.json` contains:
- `validated_at` — UTC timestamp
- `engine` — which TTS engine was used
- `language` — language code
- `summary` — total/ok/flagged/errors counts
- `results[]` — per-line: file, status, duration, expected_text, transcription, issues

Previous reports are rotated to `validation_prev.json` so you can compare across runs.

## Qwen3-TTS Workflow

```bash
source ~/qwen-tts-env/bin/activate

# Step 1: Transcribe ref audio (use vox-env for faster-whisper)
# source ~/podcast-generator/vox-env/bin/activate
# python -c "
# from faster_whisper import WhisperModel
# model = WhisperModel('base', device='cuda', compute_type='float16')
# segments, _ = model.transcribe('/home/hcl/voice_refs/VOICE.mp3', language='en')
# print(' '.join(s.text.strip() for s in segments))
# "

# Step 2: Trim ref to clean sentence boundary
# ffmpeg -y -i /home/hcl/voice_refs/VOICE.mp3 -t SECONDS -acodec copy /home/hcl/voice_refs/VOICE_qwen_ref.mp3

# Step 3: Generate
python -c "
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    'Qwen/Qwen3-TTS-12Hz-1.7B-Base',
    device_map='cuda:0',
    dtype=torch.bfloat16,
)
wavs, sr = model.generate_voice_clone(
    text='Your text here.',
    language='English',
    ref_audio='/home/hcl/voice_refs/VOICE_qwen_ref.mp3',
    ref_text='Exact transcript of the ref audio.',
)
sf.write('output.wav', wavs[0], sr)
"
```

Trimmed refs available: `ls ~/voice_refs/*_qwen_ref.mp3`

## Chatterbox Workflow

```bash
source ~/podcast-generator/vox-env/bin/activate

python -c "
from chatterbox.tts import ChatterboxTTS
import torchaudio
model = ChatterboxTTS.from_pretrained(device='cuda')
wav = model.generate('Your text here.', audio_prompt_path='/home/hcl/voice_refs/VOICE.mp3')
torchaudio.save('output.wav', wav.cpu(), model.sr)
"
```

## Disk Space

gpu-server disk is 158GB, currently near full (~99%). Main consumers:

| Path | Size | Notes |
|------|------|-------|
| `~/.cache/huggingface/hub/` | ~33GB | Model weights — clear unused models when switching engines |
| `~/tada-env/` | ~6GB | TADA + PyTorch |
| `~/podcast-generator/vox-env/` | ~7GB | Chatterbox + faster-whisper |

Before installing a new engine, check `df -h /` and clean up as needed.

## Common Problems

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| CUDA OOM during TADA encoding | Encoder + model both loaded | Use two-phase workflow (see above) |
| CUDA OOM with "Process XXXX has N GiB in use" | Previous process still holding VRAM | `nvidia-smi --query-compute-apps=pid --format=csv,noheader \| xargs -r kill -9` |
| TADA silent output | Unknown alignment issue | Re-generate; try different ref audio segment |
| Qwen output dramatically too long | ref_text doesn't match ref_audio | Transcribe ref with Whisper first, use exact transcript |
| Qwen prepends extra text | Hallucination from ICL mode | Use `temperature=0.7, repetition_penalty=1.2`; validate with `validate_tts.py`; re-generate flagged |
| `ModuleNotFoundError` | Wrong venv activated | Check venv table above |
| `No space left on device` | Disk full | Check `du -sh ~/.cache/huggingface/hub/models--*` and remove unused models |
