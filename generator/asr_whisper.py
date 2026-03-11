"""Transcribe English voice refs with Whisper large-v3."""

import json
import time
from pathlib import Path
from faster_whisper import WhisperModel

VOICE_REF_DIR = Path("/home/hcl/voice_refs")
ENGLISH_VOICES = [
    "alex.mp3", "alex-dutch.mp3", "alex-english.mp3", "daan_en.mp3",
    "ember.mp3", "emma.mp3", "emma_en.mp3", "emma_english_sample.mp3",
    "felix.mp3", "hugo.mp3", "jann.mp3", "lisa.mp3", "lucas.mp3",
    "lucas_en.mp3", "lucas_english_sample.mp3", "marc.mp3", "narrator.mp3",
    "piet_english_sample.mp3", "professor.mp3", "ruth.mp3", "serafina.mp3",
    "serge.mp3", "sofie_en.mp3", "sven.mp3", "victoria.mp3", "zara.mp3",
]

print("Loading Whisper large-v3...")
t0 = time.time()
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print(f"Loaded in {time.time() - t0:.1f}s")

results = {}
for fname in ENGLISH_VOICES:
    path = str(VOICE_REF_DIR / fname)
    segments, info = model.transcribe(path, language="en")
    text = " ".join(s.text.strip() for s in segments)
    results[fname] = text
    print(f"  {fname}: {text[:80]}...")

out_path = Path("/home/hcl/podcast-generator/asr_whisper_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {out_path}")
