#!/usr/bin/env python3
"""
Voice Tester — test different voice IDs on a section of your script.

Usage:
    python test_voice.py script.txt "OPENING" --lucas "new_voice_id_here"
    python test_voice.py script.txt "OPENING" --emma "new_voice_id_here"
"""

import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from elevenlabs.types import DialogueInput

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from src.voice_settings import get_voice_settings, parse_line

load_dotenv(SCRIPT_DIR / '.env')
client = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))

def main():
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description="Test different voice IDs on a section of your script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_voice.py script.txt "OPENING" --lucas "new_voice_id"
  python test_voice.py script.txt "OPENING" --emma "new_voice_id" --output-dir ./tests
"""
    )
    parser.add_argument("script", help="Path to dialogue script file")
    parser.add_argument("section", help="Section name to generate")
    parser.add_argument("--emma", help="Override Emma voice ID")
    parser.add_argument("--lucas", help="Override Lucas voice ID")
    parser.add_argument("--piet", help="Override Piet voice ID")
    parser.add_argument("--output-dir", help="Output directory (default: ./voice_tests)")

    args = parser.parse_args()

    script_path = Path(args.script)
    section_name = args.section

    # Get voice IDs (default from .env or override from command line)
    voice_ids = {
        "emma": args.emma or os.getenv('VOICE_EMMA'),
        "lucas": args.lucas or os.getenv('VOICE_LUCAS'),
        "piet": args.piet or os.getenv('VOICE_PIET'),
    }

    # Read and extract section
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    section_pattern = r'={20,}\s*' + re.escape(section_name) + r'.*?={20,}\s*\n(.*?)(?:={20,}|$)'
    match = re.search(section_pattern, content, re.DOTALL | re.IGNORECASE)

    if not match:
        print(f"ERROR: Section '{section_name}' not found")
        sys.exit(1)

    section_content = match.group(1).strip()
    lines = [line.strip() for line in section_content.split('\n') if line.strip()]

    # Build dialogue
    dialogue_inputs = []
    for line in lines:
        speaker, emotion, text = parse_line(line)
        if speaker and text:
            voice_id = voice_ids.get(speaker)
            if not voice_id:
                continue
            settings = get_voice_settings(speaker, emotion)
            dialogue_inputs.append(
                DialogueInput(text=text, voice_id=voice_id, voice_settings=settings)
            )

    # Generate
    print("VOICE TEST")
    print("="*60)
    print(f"Section: {section_name}")
    print(f"Lines: {len(dialogue_inputs)}")
    print(f"Voice IDs:")
    for speaker, vid in voice_ids.items():
        default = vid == os.getenv(f'VOICE_{speaker.upper()}')
        marker = " (DEFAULT)" if default else " (TESTING)"
        print(f"  {speaker}: {vid}{marker}")
    print()
    print("Generating...")

    audio = client.text_to_dialogue.convert(inputs=dialogue_inputs)

    # Save with timestamp
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "voice_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"voice_test_{section_name.lower().replace(' ', '_')}_{timestamp}.mp3"

    with open(output_path, 'wb') as f:
        for chunk in audio:
            f.write(chunk)

    print(f"Saved: {output_path}")
    print(f"Size: {output_path.stat().st_size/1024/1024:.1f} MB")
    print("="*60)

if __name__ == "__main__":
    main()
