"""Generate German Mondriaan intro episode using Qwen3-TTS with matched voice refs."""

import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from qwen_tts import Qwen3TTSModel

VOICE_REF_DIR = Path.home() / "voice_refs"

print("Loading Qwen3-TTS 1.7B model...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)
print("Model loaded.")

# Using qwen_refs with EXACTLY matched ref_text (generated via ElevenLabs with known text)
VOICES = {
    "emma_de": {
        "ref": str(VOICE_REF_DIR / "qwen_refs" / "qwen_ref_emma_de.mp3"),
        "ref_text": (
            "Willkommen zu einer neuen Folge ueber Piet Mondriaan. Heute sprechen "
            "wir ueber seine fruehen Jahre in den Niederlanden und den Moment, in "
            "dem er begann, die Welt in Linien und Farben zu sehen. Es ist eine "
            "Geschichte ueber Mut, Abstraktion und die Suche nach dem Wesentlichen."
        ),
        "lang": "German",
    },
    "lucas_de": {
        "ref": str(VOICE_REF_DIR / "qwen_refs" / "qwen_ref_lucas_de.mp3"),
        "ref_text": (
            "Was Mondriaan in Paris entdeckte, veraenderte alles. Der Kubismus "
            "zeigte ihm, dass man die Realitaet zerlegen konnte, in Flaechen, in "
            "Formen, in Beziehungen. Aber Mondriaan ging weiter als Picasso oder "
            "Braque. Er wollte zum Kern vordringen, zur reinen Wirklichkeit."
        ),
        "lang": "German",
    },
    "piet_de": {
        "ref": str(VOICE_REF_DIR / "qwen_refs" / "qwen_ref_piet_de.mp3"),
        "ref_text": (
            "Die Menschen verstehen nicht, dass es in meiner Kunst nicht um das "
            "Weglassen geht, sondern um das Finden. Jede Linie, jede Flaeche, "
            "jede Farbe, sie sind nicht weniger als die Natur. Sie sind das "
            "Wesentliche der Natur."
        ),
        "lang": "German",
    },
}

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
OUTPUT_DIR = Path.home() / "podcast-generator" / "mondriaan_output" / "de_qwen"
os.makedirs(str(OUTPUT_DIR), exist_ok=True)

print(f"\n=== Generating DE introduction ({len(SCRIPT_DE)} lines) ===")
all_audio = []
sr_out = None

for i, (speaker, text) in enumerate(SCRIPT_DE):
    voice = VOICES[speaker]
    print(f"  [{i+1}/{len(SCRIPT_DE)}] {speaker}: {text[:60]}...")
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=voice["lang"],
        ref_audio=voice["ref"],
        ref_text=voice["ref_text"],
        max_new_tokens=240,
    )
    sr_out = sr
    all_audio.append(wavs[0])
    pause = np.zeros(int(sr * 0.4), dtype=np.float32)
    all_audio.append(pause)
    sf.write(str(OUTPUT_DIR / f"de_{i+1:02d}_{speaker}.wav"), wavs[0], sr)

if not all_audio:
    print("WARNING: Script is empty, no audio generated.")
else:
    full = np.concatenate(all_audio)
    out_path = str(OUTPUT_DIR / "mondriaan_intro_de_qwen.wav")
    sf.write(out_path, full, sr_out)
    print(f"\nSaved full intro: {out_path}")
print("Done!")
