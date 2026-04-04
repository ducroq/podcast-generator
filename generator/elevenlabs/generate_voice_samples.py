#!/usr/bin/env python3
"""
Generate ~1 minute voice samples for voice cloning reference.

Usage:
    python generate_voice_samples.py [--lang en|nl] [--output-dir DIR]
"""

import argparse
import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / '.env')

API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not API_KEY:
    print("ERROR: ELEVENLABS_API_KEY not found in .env")
    sys.exit(1)

VOICES = {
    "emma": {
        "voice_id": os.getenv("VOICE_EMMA"),
        "description": "Female host, curious, engaging",
    },
    "lucas": {
        "voice_id": os.getenv("VOICE_LUCAS"),
        "description": "Male host, knowledgeable, warm",
    },
    "piet": {
        "voice_id": os.getenv("VOICE_PIET"),
        "description": "Historical figure, thoughtful",
    },
}

SAMPLE_TEXTS = {
    "en": {
        "emma": (
            "Have you ever stood in front of a painting and felt like it was trying to tell you something? "
            "I remember the first time I visited a modern art museum. I was maybe twelve years old, and I walked "
            "into this enormous white room. There was nothing on the walls except one canvas with a single red "
            "square. Just a red square. And I thought, well, that's not very impressive, is it? But then I stood "
            "there for a while, and something shifted. The red became deeper, almost alive. I started noticing "
            "how it interacted with the light, with the space around it. It was as if the painting was breathing. "
            "That moment changed everything for me. Art isn't just about what you see on the surface. It's about "
            "what happens between the artwork and the viewer. It's a conversation, really. And the most fascinating "
            "conversations are the ones where you don't know exactly where they'll lead. Don't you think so?"
        ),
        "lucas": (
            "You know, there's a common misconception about the Dutch Golden Age that I'd love to clear up. "
            "People often think of it as this period of pure prosperity and cultural achievement, which it was, "
            "but there's so much more to the story. The seventeenth century Netherlands was a place of remarkable "
            "contradictions. On one hand, you had extraordinary artistic freedom. Painters like Rembrandt and "
            "Vermeer were pushing the boundaries of what art could express. On the other hand, the wealth that "
            "funded all of this came from trade routes that stretched across the globe, and not all of that trade "
            "was ethical by any standard. What fascinates me most is how the art from that period captures these "
            "tensions. Look closely at a Rembrandt portrait, and you'll see both pride and uncertainty in those "
            "eyes. The artists were documenting their world honestly, with all its beauty and complexity. That "
            "kind of honesty is what makes great art timeless."
        ),
        "piet": (
            "When I look at a tree, I do not see branches and leaves the way most people do. I see lines "
            "reaching upward, horizontal planes intersecting with vertical forces. Nature has its own geometry, "
            "a structure hidden beneath the surface of things. My entire life has been a journey toward "
            "understanding that structure. In Paris, I discovered that the old ways of painting were not enough. "
            "Impressionism, Cubism, they were steps along the path, but they didn't go far enough. I needed to "
            "find the essence, the pure relationships between form and color. Red, yellow, blue. Horizontal and "
            "vertical. These are not limitations. They are liberation. When you strip away everything that is "
            "particular, everything that is accidental, you arrive at something universal. A painting that speaks "
            "not to one person or one culture, but to the fundamental harmony that connects all things. That is "
            "what I have spent my life searching for. And I believe I have found it in the simplest of forms."
        ),
    },
    "nl": {
        "emma": (
            "Heb je ooit voor een schilderij gestaan en het gevoel gehad dat het iets tegen je probeerde te "
            "zeggen? Ik herinner me de eerste keer dat ik een museum voor moderne kunst bezocht. Ik was misschien "
            "twaalf jaar oud, en ik liep een enorme witte zaal binnen. Er hing niets aan de muren behalve één "
            "doek met een enkel rood vierkant. Gewoon een rood vierkant. En ik dacht, nou, dat is niet erg "
            "indrukwekkend, toch? Maar toen bleef ik er even staan, en er veranderde iets. Het rood werd dieper, "
            "bijna levend. Ik begon te merken hoe het samenwerkte met het licht, met de ruimte eromheen. Het was "
            "alsof het schilderij ademde. Dat moment heeft alles voor mij veranderd. Kunst gaat niet alleen over "
            "wat je aan de oppervlakte ziet. Het gaat over wat er gebeurt tussen het kunstwerk en de kijker. Het "
            "is eigenlijk een gesprek. En de meest fascinerende gesprekken zijn die waarvan je niet precies weet "
            "waar ze naartoe leiden. Vind je ook niet? Ik denk dat dat het mooie is van kunst. Het blijft je "
            "verrassen, elke keer weer opnieuw."
        ),
        "lucas": (
            "Weet je, er bestaat een veelvoorkomend misverstand over de Gouden Eeuw dat ik graag wil rechtzetten. "
            "Mensen denken er vaak aan als een periode van pure welvaart en culturele prestaties, en dat was het "
            "ook, maar er is zoveel meer aan het verhaal. Het zeventiende-eeuwse Nederland was een plek van "
            "opmerkelijke tegenstellingen. Aan de ene kant had je buitengewone artistieke vrijheid. Schilders "
            "zoals Rembrandt en Vermeer verlegden de grenzen van wat kunst kon uitdrukken. Aan de andere kant "
            "kwam de rijkdom die dit alles financierde van handelsroutes die zich over de hele wereld uitstrekten, "
            "en niet al die handel was ethisch verantwoord. Wat mij het meest fascineert is hoe de kunst uit die "
            "periode deze spanningen vastlegt. Kijk goed naar een portret van Rembrandt, en je ziet zowel trots "
            "als onzekerheid in die ogen. De kunstenaars documenteerden hun wereld eerlijk, met al haar schoonheid "
            "en complexiteit. Die eerlijkheid is wat grote kunst tijdloos maakt. En dat is precies wat we in "
            "onze podcast proberen te laten zien."
        ),
        "piet": (
            "Als ik naar een boom kijk, zie ik geen takken en bladeren zoals de meeste mensen dat doen. Ik zie "
            "lijnen die omhoog reiken, horizontale vlakken die kruisen met verticale krachten. De natuur heeft "
            "haar eigen geometrie, een structuur verborgen onder het oppervlak der dingen. Mijn hele leven is een "
            "reis geweest naar het begrijpen van die structuur. In Parijs ontdekte ik dat de oude manieren van "
            "schilderen niet voldoende waren. Impressionisme, kubisme, het waren stappen op het pad, maar ze "
            "gingen niet ver genoeg. Ik moest de essentie vinden, de zuivere verhoudingen tussen vorm en kleur. "
            "Rood, geel, blauw. Horizontaal en verticaal. Dit zijn geen beperkingen. Het is bevrijding. Wanneer "
            "je alles weghaalt wat bijzonder is, alles wat toevallig is, kom je uit bij iets universeels. Een "
            "schilderij dat niet spreekt tot één persoon of één cultuur, maar tot de fundamentele harmonie die "
            "alle dingen verbindt. Dat is waar ik mijn hele leven naar heb gezocht. En ik geloof dat ik het heb "
            "gevonden in de eenvoudigste vormen."
        ),
    },
}

MODELS = {
    "en": "eleven_turbo_v2_5",
    "nl": "eleven_multilingual_v2",
}

# Voice settings: high similarity for faithful voice capture, moderate stability
VOICE_SETTINGS = {
    "stability": 0.45,
    "similarity_boost": 0.95,
    "style": 0.35,
    "use_speaker_boost": True,
}


def generate_sample(name, voice_id, text, output_path, model_id):
    """Generate a voice sample using the REST TTS API."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": API_KEY}

    data = {
        "text": text,
        "model_id": model_id,
        "voice_settings": VOICE_SETTINGS,
    }

    print(f"  Generating {name}...")
    response = requests.post(url, json=data, headers=headers, timeout=(10, 300))

    if response.status_code != 200:
        print(f"  ERROR: {response.status_code} - {response.text[:200]}")
        return False

    with open(output_path, "wb") as f:
        f.write(response.content)

    size = output_path.stat().st_size
    # Rough estimate: 128kbps mp3 = 16KB/sec, so size/16000 ~ seconds
    est_seconds = size / 16000
    print(f"  Saved: {output_path} ({size:,} bytes, ~{est_seconds:.0f}s)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate voice samples for cloning")
    parser.add_argument("--lang", choices=["en", "nl"], default="en",
                        help="Language: en (English) or nl (Dutch)")
    parser.add_argument("--output-dir", help="Output directory (default: ./voice_samples)")
    args = parser.parse_args()

    lang = args.lang
    lang_label = {"en": "English", "nl": "Dutch"}[lang]
    model_id = MODELS[lang]

    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "voice_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("VOICE SAMPLE GENERATOR")
    print("=" * 50)
    print(f"Language: {lang_label}")
    print(f"Output: {output_dir}")
    print(f"Model: {model_id}")
    print(f"Settings: stability={VOICE_SETTINGS['stability']}, "
          f"similarity={VOICE_SETTINGS['similarity_boost']}, "
          f"style={VOICE_SETTINGS['style']}")
    print()

    results = {}
    for name, config in VOICES.items():
        voice_id = config["voice_id"]
        if not voice_id:
            print(f"  SKIP {name}: no voice ID configured")
            continue

        text = SAMPLE_TEXTS[lang][name]
        word_count = len(text.split())
        print(f"[{name.upper()}] {config['description']}")
        print(f"  Words: {word_count} (~{word_count / 150:.1f} min at 150 wpm)")

        output_path = output_dir / f"{name}_{lang}_sample.mp3"
        success = generate_sample(name, voice_id, text, output_path, model_id)
        results[name] = success
        print()

    # Summary
    print("=" * 50)
    print("RESULTS")
    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {name}: {status}")

    ok_count = sum(1 for v in results.values() if v)
    print(f"\n{ok_count}/{len(results)} {lang_label} samples generated in {output_dir}")


if __name__ == "__main__":
    main()
