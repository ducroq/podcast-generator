#!/usr/bin/env python3
"""
Generate audio for a single dialogue line (repair tool).

Usage:
    python generate_single_line.py "SPEAKER: [tone] dialogue text" [output_name] [--method dialogue|rest]

Example:
    python generate_single_line.py "Emma: [realizing] Omdat die... niet gebonden zijn!" emma_fix_01
    python generate_single_line.py "Lucas: [enthusiastic] Dit is geweldig!" lucas_fix_01 --method rest
"""

import sys
import os
import re
import requests
from pathlib import Path
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from elevenlabs.types import DialogueInput

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from src.dialogue_parser import DialogueParser
from src.voice_config import VoiceConfig
from src.voice_settings import (
    EMOTIONAL_VARIANTS, BASE_SIMILARITY, STABILITY_OFFSET,
    get_voice_settings,
)

# Voice settings for REST API (Lucas)
REST_SIMILARITY_BOOST = 0.8
REST_USE_SPEAKER_BOOST = True

# Speaker-to-method mapping (auto-detection)
SPEAKER_METHOD_MAP = {
    "emma": "dialogue",  # text_to_dialogue API with eleven_turbo_v2_5
    "lucas": "rest",     # REST API with eleven_multilingual_v2
    "piet": "rest",      # REST API with eleven_multilingual_v2
}

def parse_emotion_from_line(text):
    """Extract emotion tag from dialogue text like [realizing] or [thoughtful]"""
    match = re.match(r'\[(\w+(?:\s+\w+)*)\]\s*(.*)', text.strip())
    if match:
        emotion = match.group(1).lower()
        clean_text = match.group(2).strip()
        return emotion, clean_text
    return "default", text

def get_rest_voice_settings(speaker, emotion):
    """Get voice settings for REST API (dict format)"""
    variants = EMOTIONAL_VARIANTS.get(speaker, {})
    variant = variants.get(emotion, variants.get("default", {"stability": 0.4, "style": 0.4}))

    return {
        "stability": variant["stability"],
        "similarity_boost": REST_SIMILARITY_BOOST,
        "style": variant["style"],
        "use_speaker_boost": REST_USE_SPEAKER_BOOST
    }

def generate_with_rest_api(text, voice_id, output_path, voice_settings, api_key):
    """Generate audio using the REST text-to-speech API with eleven_multilingual_v2"""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": api_key}

    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": voice_settings,
        "enable_logging": False
    }

    response = requests.post(url, json=data, headers=headers, timeout=(10, 300))

    if response.status_code != 200:
        print(f"[ERROR] API Error: {response.status_code} - {response.text[:200]}")
        return False

    with open(output_path, "wb") as f:
        f.write(response.content)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True
    return False

def generate_with_dialogue_api(text, voice_id, output_path, voice_settings, api_key):
    """Generate audio using text_to_dialogue API with eleven_turbo_v2_5"""
    client = ElevenLabs(api_key=api_key)

    dialogue_input = DialogueInput(
        text=text,
        voice_id=voice_id,
        voice_settings=voice_settings
    )
    audio_generator = client.text_to_dialogue.convert(inputs=[dialogue_input])

    with open(output_path, 'wb') as f:
        for chunk in audio_generator:
            f.write(chunk)

    return os.path.exists(output_path) and os.path.getsize(output_path) > 0

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate audio for a single dialogue line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_single_line.py "Emma: [realizing] Omdat die... niet gebonden zijn!" emma_fix_01
  python generate_single_line.py "Lucas: [enthusiastic] Dit is geweldig!" lucas_fix_01
  python generate_single_line.py "Lucas: [calm] Test" test --method dialogue

Methods:
  dialogue - text_to_dialogue API with eleven_turbo_v2_5 (supports audio tags)
  rest     - REST text-to-speech API with eleven_multilingual_v2 (better for Dutch)

Auto-detection:
  Emma   -> dialogue (eleven_turbo_v2_5)
  Lucas  -> rest (eleven_multilingual_v2)
  Piet   -> rest (eleven_multilingual_v2)
"""
    )
    parser.add_argument("line", nargs="?", help="Dialogue line in format 'Speaker: [emotion] text'")
    parser.add_argument("output_name", nargs="?", default="repair", help="Output filename (without extension)")
    parser.add_argument("--method", "-m", choices=["dialogue", "rest"],
                        help="API method to use (auto-detected from speaker if not specified)")
    parser.add_argument("--output-dir", help="Output directory (default: ./repairs)")

    args = parser.parse_args()

    # If no line provided, use hardcoded test
    if args.line:
        input_line = args.line
        output_name = args.output_name
    else:
        input_line = "Emma: [excited] And that changes everything we thought we knew!"
        output_name = "repair_test"

    # Load environment variables
    load_dotenv(SCRIPT_DIR / '.env')
    api_key = os.getenv('ELEVENLABS_API_KEY')
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY not found in .env file")

    # Parse the dialogue line (parser now preserves emotion tags)
    dialogue_parser = DialogueParser()
    lines = dialogue_parser.parse_text(input_line)

    if not lines:
        print("Error: Could not parse the dialogue line")
        print("Expected format: [speaker]: [optional_tone] dialogue text")
        sys.exit(1)

    line = lines[0]
    speaker = line.speaker
    emotion = line.emotion
    text = line.text

    # Get voice mapping
    voice_config = VoiceConfig(str(SCRIPT_DIR / '.env'))
    voice_id = voice_config.get_voice_id(speaker.upper())

    clean_text = text
    speaker_lower = speaker.lower()

    # Determine method: explicit argument > auto-detection
    if args.method:
        method = args.method
    else:
        method = SPEAKER_METHOD_MAP.get(speaker_lower, "dialogue")

    # Get voice settings based on method
    if method == "dialogue":
        voice_settings = get_voice_settings(speaker_lower, emotion)
        model_name = "eleven_turbo_v2_5"
        settings_str = f"stability={voice_settings.stability:.2f}, similarity={voice_settings.similarity_boost:.2f}, style={voice_settings.style:.2f}"
    else:  # rest
        voice_settings = get_rest_voice_settings(speaker_lower, emotion)
        model_name = "eleven_multilingual_v2"
        settings_str = f"stability={voice_settings['stability']:.2f}, similarity={voice_settings['similarity_boost']:.2f}, style={voice_settings['style']:.2f}"

    print(f"Generating audio for:")
    print(f"  Speaker: {speaker.upper()}")
    print(f"  Emotion: {emotion}")
    print(f"  Text: {clean_text}")
    print(f"  Voice ID: {voice_id}")
    print(f"  Method: {method} ({model_name})")
    print(f"  Settings: {settings_str}")

    # Create output directory
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "repairs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_name}.mp3"

    # Generate audio using the appropriate method
    print(f"\nGenerating audio (using {method} API with {model_name})...")

    if method == "dialogue":
        success = generate_with_dialogue_api(clean_text, voice_id, output_path, voice_settings, api_key)
    else:
        success = generate_with_rest_api(clean_text, voice_id, output_path, voice_settings, api_key)

    if success:
        file_size = os.path.getsize(output_path)
        print(f"[SUCCESS] Audio saved to: {output_path} ({file_size:,} bytes)")
        print(f"\nTo use this in your podcast:")
        print(f"1. Listen to the repair file to verify it's correct")
        print(f"2. Replace the corresponding segment in your full podcast using audio editing software")
        print(f"   (or regenerate the full podcast if needed)")
    else:
        print(f"[FAILED] Could not generate audio")
        sys.exit(1)

if __name__ == "__main__":
    main()
