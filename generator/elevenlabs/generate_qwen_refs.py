"""Generate ElevenLabs voice samples to use as Qwen3-TTS reference audio.

Creates short samples with known text for each speaker, so Qwen gets
a perfect ref_audio + ref_text match for voice cloning.
"""

import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from src.voice_config import VoiceConfig

load_dotenv(SCRIPT_DIR / ".env")

API_KEY = os.getenv("ELEVENLABS_API_KEY")
voice_config = VoiceConfig(str(SCRIPT_DIR / ".env"))

# Reference texts — conversational, ~15-20 seconds each
# These exact texts will be passed as ref_text to Qwen
REFS = {
    "emma_en": {
        "voice_key": "EMMA",
        "text": (
            "Welcome back to another episode about Piet Mondriaan. Today we are "
            "looking at something I find truly fascinating. The moment he stopped "
            "painting trees and flowers and began seeing the world as pure lines "
            "and colors. What drives someone to strip everything away like that?"
        ),
    },
    "lucas_en": {
        "voice_key": "LUCAS",
        "text": (
            "What Mondriaan discovered in Paris changed everything. Cubism showed "
            "him that you could break reality apart, into planes, into forms, into "
            "relationships. But Mondriaan went further than Picasso or Braque ever "
            "did. He did not want to simplify things. He wanted to get to the core."
        ),
    },
    "piet_en": {
        "voice_key": "PIET",  # 56kxWBSW5DRJ5pZBsrew
        "text": (
            "People do not understand that my art is not about leaving things out. "
            "It is about finding. Every line, every plane, every colour, they are "
            "not less than nature. They are the essence of nature. That is what "
            "remains when you remove everything accidental."
        ),
    },
    "emma_de": {
        "voice_key": "EMMA_DE",
        "text": (
            "Willkommen zu einer neuen Folge ueber Piet Mondriaan. Heute sprechen "
            "wir ueber seine fruehen Jahre in den Niederlanden und den Moment, in "
            "dem er begann, die Welt in Linien und Farben zu sehen. Es ist eine "
            "Geschichte ueber Mut, Abstraktion und die Suche nach dem Wesentlichen."
        ),
    },
    "lucas_de": {
        "voice_key": "LUCAS_DE",
        "text": (
            "Was Mondriaan in Paris entdeckte, veraenderte alles. Der Kubismus "
            "zeigte ihm, dass man die Realitaet zerlegen konnte, in Flaechen, in "
            "Formen, in Beziehungen. Aber Mondriaan ging weiter als Picasso oder "
            "Braque. Er wollte zum Kern vordringen, zur reinen Wirklichkeit."
        ),
    },
    "piet_de": {
        "voice_key": "PIET_DE",
        "text": (
            "Die Menschen verstehen nicht, dass es in meiner Kunst nicht um das "
            "Weglassen geht, sondern um das Finden. Jede Linie, jede Flaeche, "
            "jede Farbe, sie sind nicht weniger als die Natur. Sie sind das "
            "Wesentliche der Natur."
        ),
    },
}


def generate_sample(name, config, output_dir):
    voice_id = voice_config.get_voice_id(config["voice_key"])
    text = config["text"]
    output_path = output_dir / f"qwen_ref_{name}.mp3"

    print(f"  {name} (voice: {config['voice_key']}, id: {voice_id[:8]}...)")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": API_KEY}
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.9,
            "style": 0.3,
        },
    }

    response = requests.post(url, json=data, headers=headers, timeout=(10, 300))
    if response.status_code != 200:
        print(f"    FAILED: {response.status_code} - {response.text[:200]}")
        return False

    with open(output_path, "wb") as f:
        f.write(response.content)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"    OK: {output_path.name} ({size_kb:.0f} KB)")
    return True


def main():
    output_dir = SCRIPT_DIR / "qwen_refs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating ElevenLabs samples for Qwen voice cloning...\n")

    for name, config in REFS.items():
        generate_sample(name, config, output_dir)

    # Write ref_text file for reference
    ref_text_path = output_dir / "ref_texts.txt"
    with open(ref_text_path, "w", encoding="utf-8") as f:
        for name, config in REFS.items():
            f.write(f"=== {name} ===\n{config['text']}\n\n")

    print(f"\nRef texts saved to: {ref_text_path}")
    print("Done! Upload qwen_refs/ to gpu-server for use with Qwen3-TTS.")


if __name__ == "__main__":
    main()
