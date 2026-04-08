"""Regenerate hallucinated lines with tuned Chatterbox.

Usage (on gpu-server):
    source ~/podcast-generator/qwen-tts-env/bin/activate
    python3 ~/podcast-generator/podcasts/it-is-both/regen_hallucinated.py
"""

from pathlib import Path
import soundfile as sf
from chatterbox.tts import ChatterboxTTS

LINES_DIR = Path.home() / "ep01_pipeline" / "lines"

REGEN = [
    {
        "idx": 7,
        "speaker": "junior_manager",
        "ref": Path.home() / "voice_refs" / "alex_qwen_ref.mp3",
        "segments": [
            {"text": "we could, uh...", "pause": 0.3},
            {"text": "I actually haven't finished the book yet.", "pause": 0.2},
            {"text": "But the point is... systems."},
        ],
    },
    {
        "idx": 56,
        "speaker": "zara",
        "ref": Path.home() / "voice_refs" / "zara_qwen_ref.mp3",
        "segments": [
            {"text": "My Mondays changed. I stopped dreading the presentations.", "pause": 0.2},
            {"text": "Not because I got better at them.", "pause": 0.3},
            {"text": "Because I stopped thinking there was something wrong with me for hating them.", "pause": 0.3},
            {"text": "That was real. That actually happened to me."},
        ],
    },
]

import numpy as np

print("Loading Chatterbox...")
model = ChatterboxTTS.from_pretrained(device="cuda")

for line in REGEN:
    idx = line["idx"]
    spk = line["speaker"]
    ref = str(line["ref"])
    outfile = LINES_DIR / f"{idx:03d}_{spk}.wav"

    print(f"  [{idx:03d}] {spk}")
    parts = []
    for seg in line["segments"]:
        print(f"    {seg['text'][:50]}...", end=" ", flush=True)
        wav = model.generate(
            seg["text"], audio_prompt_path=ref,
            temperature=0.6, exaggeration=0.3, cfg_weight=0.7,
        )
        audio = wav.squeeze().cpu().numpy()
        parts.append(audio)
        pause = seg.get("pause", 0)
        if pause > 0:
            parts.append(np.zeros(int(24000 * pause), dtype=np.float32))
        print(f"{len(audio) / 24000:.1f}s")

    full = np.concatenate(parts)
    sf.write(str(outfile), full, 24000)
    print(f"    -> {outfile.name} ({len(full) / 24000:.1f}s total)")

print("Done")
