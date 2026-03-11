"""Bootstrap Qwen3-TTS self-references for all voices.

Strategy: Use each original voice ref MP3 with a short sentence (5-10 words).
Short sentences are safe even with mismatched ref_text. Qwen generates its own
version, which becomes the new matched ref_audio + ref_text pair.

This gives us clean Qwen-native refs without needing ElevenLabs.
"""

import torch
import soundfile as sf
import os
import json
from pathlib import Path
from qwen_tts import Qwen3TTSModel

VOICE_REF_DIR = Path("/home/hcl/voice_refs")
OUTPUT_DIR = Path("/home/hcl/voice_refs/qwen_self_refs")

# Short, natural sentences per voice. Unique to help distinguish them.
# German sentences for _de variants; English for everything else.
VOICES = {
    # --- English voices ---
    "alex": {
        "ref": "alex.mp3",
        "text": "Welcome to the show, it is great to have you here with us today.",
        "lang": "English",
    },
    "alex-dutch": {
        "ref": "alex-dutch.mp3",
        "text": "Let me tell you something interesting about the way we see the world.",
        "lang": "English",
    },
    "alex-english": {
        "ref": "alex-english.mp3",
        "text": "There is always more than one way to look at a problem like this.",
        "lang": "English",
    },
    "daan_en": {
        "ref": "daan_en.mp3",
        "text": "Good evening, and welcome to tonight's edition of the news.",
        "lang": "English",
    },
    "ember": {
        "ref": "ember.mp3",
        "text": "You know what I find really fascinating about this whole story?",
        "lang": "English",
    },
    "emma": {
        "ref": "emma.mp3",
        "text": "Today we are going to explore a topic that is close to my heart.",
        "lang": "English",
    },
    "emma_en": {
        "ref": "emma_en.mp3",
        "text": "This is one of those moments that changes everything you thought you knew.",
        "lang": "English",
    },
    "emma_english_sample": {
        "ref": "emma_english_sample.mp3",
        "text": "I have been thinking about this question for quite some time now.",
        "lang": "English",
    },
    "felix": {
        "ref": "felix.mp3",
        "text": "Imagine standing at the edge of the unknown, looking out into the vast darkness.",
        "lang": "English",
    },
    "hugo": {
        "ref": "hugo.mp3",
        "text": "This is exactly the kind of breakthrough we have been waiting for.",
        "lang": "English",
    },
    "jann": {
        "ref": "jann.mp3",
        "text": "Let us take a step back and consider what this really means.",
        "lang": "English",
    },
    "lisa": {
        "ref": "lisa.mp3",
        "text": "I think the most important thing here is the human connection behind it all.",
        "lang": "English",
    },
    "lucas": {
        "ref": "lucas.mp3",
        "text": "The research shows something quite remarkable when you look at the data closely.",
        "lang": "English",
    },
    "lucas_en": {
        "ref": "lucas_en.mp3",
        "text": "What we discovered next was something nobody had anticipated at all.",
        "lang": "English",
    },
    "lucas_english_sample": {
        "ref": "lucas_english_sample.mp3",
        "text": "If you look at the evidence carefully, a very different picture emerges.",
        "lang": "English",
    },
    "marc": {
        "ref": "marc.mp3",
        "text": "From a scientific perspective, this finding is absolutely extraordinary.",
        "lang": "English",
    },
    "narrator": {
        "ref": "narrator.mp3",
        "text": "In the beginning, there was nothing but silence and the promise of what was to come.",
        "lang": "English",
    },
    "piet_english_sample": {
        "ref": "piet_english_sample.mp3",
        "text": "When I look at a painting, I see more than just colours on a canvas.",
        "lang": "English",
    },
    "professor": {
        "ref": "professor.mp3",
        "text": "The historical context here is absolutely essential to understanding this phenomenon.",
        "lang": "English",
    },
    "ruth": {
        "ref": "ruth.mp3",
        "text": "Once upon a time, in a land far away, there lived a very curious little fox.",
        "lang": "English",
    },
    "serafina": {
        "ref": "serafina.mp3",
        "text": "There is something deeply beautiful about the way music moves through us.",
        "lang": "English",
    },
    "serge": {
        "ref": "serge.mp3",
        "text": "Today we present a special report on the events of the past week.",
        "lang": "English",
    },
    "sofie_en": {
        "ref": "sofie_en.mp3",
        "text": "Coming up next, we have a story that will surprise you.",
        "lang": "English",
    },
    "sven": {
        "ref": "sven.mp3",
        "text": "I am not sure I agree with that, can you explain what you mean?",
        "lang": "English",
    },
    "victoria": {
        "ref": "victoria.mp3",
        "text": "The situation is rather more complicated than it appears at first glance.",
        "lang": "English",
    },
    "zara": {
        "ref": "zara.mp3",
        "text": "Oh my gosh, this is so exciting, I cannot wait to tell you about it!",
        "lang": "English",
    },
    # --- German voices ---
    "emma_de": {
        "ref": "emma_de.mp3",
        "text": "Willkommen zu einer neuen Folge, heute haben wir ein ganz besonderes Thema fuer euch.",
        "lang": "German",
    },
    "lucas_de": {
        "ref": "lucas_de.mp3",
        "text": "Die Forschung zeigt etwas ganz Bemerkenswertes, wenn man die Daten genauer betrachtet.",
        "lang": "German",
    },
    "piet_de": {
        "ref": "piet_de.mp3",
        "text": "Die Menschen verstehen nicht, dass es in meiner Kunst um das Finden geht.",
        "lang": "German",
    },
    # --- Skipping Dutch-only voices (Qwen doesn't support Dutch) ---
    # daan_nl, sofie_nl, oma, emma_nl_sample, lucas_nl_sample, piet_nl_sample
}

# Dutch voices we skip but document why
SKIPPED = ["daan_nl", "sofie_nl", "oma", "emma_nl_sample", "lucas_nl_sample", "piet_nl_sample"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Qwen3-TTS 1.7B model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    print(f"Model loaded. Generating refs for {len(VOICES)} voices.\n")

    results = {}
    failures = []

    for i, (name, cfg) in enumerate(VOICES.items()):
        ref_path = str(VOICE_REF_DIR / cfg["ref"])
        print(f"[{i+1}/{len(VOICES)}] {name} ({cfg['lang']}): {cfg['text'][:50]}...")

        try:
            wavs, sr = model.generate_voice_clone(
                text=cfg["text"],
                language=cfg["lang"],
                ref_audio=ref_path,
                ref_text=cfg["text"],  # Will be mismatched, but short text is safe
                max_new_tokens=240,
            )

            out_path = OUTPUT_DIR / f"{name}.wav"
            sf.write(str(out_path), wavs[0], sr)

            duration = len(wavs[0]) / sr
            print(f"    OK: {duration:.1f}s")

            results[name] = {
                "ref_audio": f"qwen_self_refs/{name}.wav",
                "ref_text": cfg["text"],
                "lang": cfg["lang"],
                "duration": round(duration, 1),
                "source_voice": cfg["ref"],
            }
        except Exception as e:
            print(f"    FAILED: {e}")
            failures.append(name)

    # Save metadata
    meta_path = OUTPUT_DIR / "ref_manifest.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n=== Done ===")
    print(f"Success: {len(results)}/{len(VOICES)}")
    if failures:
        print(f"Failed: {failures}")
    print(f"Skipped (Dutch-only, no Qwen support): {SKIPPED}")
    print(f"Manifest: {meta_path}")


if __name__ == "__main__":
    main()
