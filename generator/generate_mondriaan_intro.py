import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from qwen_tts import Qwen3TTSModel

VOICE_REF_DIR = Path.home() / "voice_refs"
OUTPUT_DIR = Path.home() / "podcast-generator" / "mondriaan_output"

print("Loading Qwen3-TTS 1.7B model...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)
print("Model loaded.")

VOICES = {
    "emma_en": {
        "ref": str(VOICE_REF_DIR / "emma_en.mp3"),
        "ref_text": "Welcome back to another episode about Pete Mondrian. Today we are looking at something I find truly fascinating, the moment he stopped painting trees and flowers and began seeing the world as pure lines and colors. What drives someone to strip everything away like that? Lucas, I am curious, was this a gradual transformation or was there a turning point?",
        "lang": "English",
    },
    "lucas_en": {
        "ref": str(VOICE_REF_DIR / "lucas_en.mp3"),
        "ref_text": "What Mondrian discovered in Paris changed everything. Cubism showed him that you could break reality apart, into planes, into forms, into relationships. But Mondrian went further than Picasso or Braque ever did. He did not want to simplify things. He wanted to get to the core, to what he called pure reality. And that is a fundamentally different ambition.",
        "lang": "English",
    },
    "piet_en": {
        "ref": str(VOICE_REF_DIR / "piet_english_sample.mp3"),
        "ref_text": "When I look at a tree I do not see branches and leaves the way most people do. I see lines reaching upward, horizontal planes intersecting with vertical forces. Nature has its own geometry, a structure hidden beneath the surface of things. My entire life has been a journey toward understanding that structure.",
        "lang": "English",
    },
    "emma_de": {
        "ref": str(VOICE_REF_DIR / "emma_de.mp3"),
        "ref_text": "Willkommen zu einer neuen Folge ueber Pete Mondrian. Heute sprechen wir ueber seine fruehen Jahre in den Niederlanden und den Moment, in dem er begann, die Welt in Linien und Farben zu sehen. Es ist eine Geschichte ueber Mut, Abstraktion und die Suche nach dem Wesentlichen in der Kunst.",
        "lang": "German",
    },
    "lucas_de": {
        "ref": str(VOICE_REF_DIR / "lucas_de.mp3"),
        "ref_text": "Was Mondrian in Paris entdeckte, veraenderte alles. Der Kubismus zeigte ihm, dass man die Realitaet zerlegen konnte. In Flaechen, in Formen, in Beziehungen. Aber Mondrian ging weiter als Picasso oder Braque. Er wollte nicht die Dinge vereinfachen, er wollte zum Kern vordringen. Zudem, was er die reine Wirklichkeit nannte.",
        "lang": "German",
    },
    "piet_de": {
        "ref": str(VOICE_REF_DIR / "piet_de.mp3"),
        "ref_text": "Die Menschen verstehen nicht, dass es in meiner Kunst nicht um das Weglassen geht, sondern um das Finden. Jede Linie, jede Flaeche, jede Farbe, sie sind nicht weniger als die Natur, sie sind das Wesentliche der Natur. Das, was uebrig bleibt, wenn man alles Zufaellige entfernt. In meiner Erfahrung liegt die tiefste Schoenheit in der reinsten Form.",
        "lang": "German",
    },
}

SCRIPT_EN = [
    ("emma_en", "Welcome to Mondriaan the Thinker!"),
    ("lucas_en", "The podcast that takes you into the world behind the paintings of the famous Dutch painter Piet Mondriaan."),
    ("emma_en", "I'm Emma!"),
    ("lucas_en", "And I'm Lucas."),
    ("emma_en", "And this is the beginning of a special journey through the mind of a man we all think we know."),
    ("lucas_en", "For six weeks, we will delve into the hidden world of Mondriaan as a thinker."),
    ("lucas_en", "A world of essays that have remained virtually unknown until now."),
    ("emma_en", "Each week, we will get one step closer to answering the question: what did Mondriaan want to say with his paintings?"),
    ("emma_en", "Because, you are a visual artist, you speak through your paintings, so why would you suddenly need words?"),
    ("lucas_en", "In an interview from nineteen twenty-two, Mondriaan himself said:"),
    ("piet_en", "It is difficult to explain what I mean with my paintings. In them, I have expressed it as well as I could. The other side, what remains unsaid, is better brought out in an article."),
    ("emma_en", "The other side! What a beautiful metaphor! The story behind the painting, the world of thought behind the lines and planes."),
    ("lucas_en", "Because behind those seemingly simple little lines and colours, behind them lies an entire world of thought."),
    ("lucas_en", "A vision of how we can live together, how we can improve the world. How we can make the world, uhm..."),
    ("emma_en", "Improve the world with abstract art?"),
    ("lucas_en", "Sounds ambitious, but just wait until you hear his own words."),
    ("emma_en", "Alright, let us go discover that other side of Mondriaan!"),
    ("lucas_en", "Welcome to the unknown world of Mondriaan the Thinker!"),
]

SCRIPT_DE = [
    ("emma_de", "Willkommen bei Mondriaan der Denker!"),
    ("lucas_de", "Der Podcast, der euch in die Welt hinter den Gemaelden des beruehmten niederlaendischen Malers Piet Mondriaan mitnimmt."),
    ("emma_de", "Ich bin Emma!"),
    ("lucas_de", "Und ich bin Lucas."),
    ("emma_de", "Und dies ist der Beginn einer besonderen Reise durch den Geist eines Mannes, den wir alle zu kennen glauben."),
    ("lucas_de", "Sechs Wochen lang tauchen wir in die verborgene Welt von Mondriaan als Denker ein."),
    ("lucas_de", "Eine Welt von Essays, die bis heute nahezu unbekannt geblieben sind."),
    ("emma_de", "Jede Woche kommen wir der Antwort auf die Frage einen Schritt naeher: Was wollte Mondriaan mit seinen Gemaelden ausdruecken?"),
    ("emma_de", "Denn als bildender Kuenstler sprichst du durch deine Gemaelde, warum solltest du dann ploetzlich Worte brauchen?"),
    ("lucas_de", "In einem Interview aus dem Jahr neunzehnhundertzweiundzwanzig sagte Mondriaan selbst:"),
    ("piet_de", "Es ist schwierig zu erklaeren, was ich mit meinen Gemaelden meine. Darin habe ich es so gut wie moeglich ausgedrueckt. Die Rueckseite, die unausgesprochen bleibt, laesst sich besser in einem Artikel zum Ausdruck bringen."),
    ("emma_de", "Die Rueckseite! Was fuer eine wunderbare Metapher! Die Geschichte hinter dem Gemaelde, die Gedankenwelt hinter den Linien und Flaechen."),
    ("lucas_de", "Denn hinter diesen scheinbar einfachen Linien und Farben, dahinter verbirgt sich eine ganze Gedankenwelt."),
    ("lucas_de", "Eine Vision davon, wie wir zusammenleben koennen, wie wir die Welt verbessern koennen. Wie wir die Welt, aehm..."),
    ("emma_de", "Die Welt verbessern mit abstrakter Kunst?"),
    ("lucas_de", "Klingt ehrgeizig, aber wartet ab, bis ihr seine eigenen Worte hoert."),
    ("emma_de", "Gut, lasst uns diese andere Seite von Mondriaan entdecken!"),
    ("lucas_de", "Willkommen in der unbekannten Welt von Mondriaan, dem Denker!"),
]

import os
os.makedirs(str(OUTPUT_DIR), exist_ok=True)

for label, script in [("en", SCRIPT_EN), ("de", SCRIPT_DE)]:
    print(f"\n=== Generating {label.upper()} introduction ===")
    all_audio = []
    sr_out = None

    for i, (speaker, text) in enumerate(script):
        voice = VOICES[speaker]
        print(f"  [{i+1}/{len(script)}] {speaker}: {text[:60]}...")
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=voice["lang"],
            ref_audio=voice["ref"],
            ref_text=voice["ref_text"],
        )
        sr_out = sr
        all_audio.append(wavs[0])
        pause = np.zeros(int(sr * 0.4), dtype=np.float32)
        all_audio.append(pause)
        sf.write(str(OUTPUT_DIR / f"{label}_{i+1:02d}_{speaker}.wav"), wavs[0], sr)

    full = np.concatenate(all_audio)
    out_path = str(OUTPUT_DIR / f"mondriaan_intro_{label}_qwen.wav")
    sf.write(out_path, full, sr_out)
    print(f"  Saved full intro: {out_path}")

print("\nAll done!")
