"""Generate laugh/chuckle backchannel clips for It Is Both cast.

Usage (on gpu-server):
    source ~/podcast-generator/qwen-tts-env/bin/activate
    python3 ~/podcast-generator/podcasts/it-is-both/generate_laughs.py -o ~/ep01_laughs/
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

VOICE_REFS = {
    "alex": {
        "ref": Path.home() / "voice_refs" / "alex_qwen_ref.mp3",
        "text": (
            "At some point, someone hands you a book. Good to Great. "
            "Lean Startup. Atomic Habits. It comes recommended with "
            "reverence, like scripture that happens to have a forward "
            "by a former CEO."
        ),
    },
    "morgan": {
        "ref": Path.home() / "voice_refs" / "lisa_qwen_ref.mp3",
        "text": (
            "Welcome back to another episode on Machine Learning for "
            "Engineers. Today, we are diving deep into AI explainability, "
            "a topic that is becoming critical for anyone building ML "
            "systems."
        ),
    },
    "zara": {
        "ref": Path.home() / "voice_refs" / "zara_qwen_ref.mp3",
        "text": (
            "Okay, so here is the thing nobody tells you about your first "
            "job search. That generic CV, you are blasting to 50 companies, "
            "it is not working. And that is not because there is something "
            "wrong with you. It is because the system is designed to filter "
            "you out."
        ),
    },
}

# Laugh/chuckle phrases — varying intensity and style
LAUGH_PHRASES = {
    "alex": [
        "Ha!",
        "Haha.",
        "Heh.",
        "Ha, yeah.",
        "Haha, right.",
        "Pfft.",
    ],
    "morgan": [
        "Ha!",
        "Heh.",
        "Hmph.",
        "Ha, exactly.",
        "Haha.",
        "Heh heh.",
    ],
    "zara": [
        "Ha!",
        "Haha!",
        "Heh.",
        "Pfft, yeah.",
        "Haha, okay.",
        "Ha, right.",
    ],
}


def main():
    parser = argparse.ArgumentParser(description="Generate laugh clips for backchannels")
    parser.add_argument("-o", "--output-dir", default="~/ep01_laughs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Qwen3-TTS...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    print("Model loaded.\n")

    for speaker, phrases in LAUGH_PHRASES.items():
        voice = VOICE_REFS[speaker]
        print(f"{speaker}:")
        for i, phrase in enumerate(phrases):
            print(f"  {phrase}...", end=" ", flush=True)
            try:
                wavs, sr = model.generate_voice_clone(
                    text=phrase,
                    language="English",
                    ref_audio=str(voice["ref"]),
                    ref_text=voice["text"],
                    temperature=0.5,
                    repetition_penalty=1.2,
                )
                audio = wavs[0].copy()
                del wavs
                torch.cuda.empty_cache()

                # 8ms fades
                fade = int(sr * 0.008)
                if len(audio) > fade * 2:
                    audio[:fade] *= np.linspace(0.0, 1.0, fade)
                    audio[-fade:] *= np.linspace(1.0, 0.0, fade)

                filename = f"laugh_{speaker}_{i:02d}.wav"
                sf.write(str(output_dir / filename), audio, sr)
                print(f"{len(audio) / sr:.1f}s -> {filename}")
            except Exception as e:
                print(f"FAILED: {str(e)[:60]}")

        torch.cuda.empty_cache()
        print()

    print(f"\nDone. Clips in {output_dir}")
    print("Listen and pick the best ones, then rename to bc_<speaker>_NN.wav for the pipeline.")


if __name__ == "__main__":
    main()
