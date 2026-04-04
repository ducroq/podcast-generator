#!/usr/bin/env python3
"""
ElevenLabs Episode Generator (text_to_dialogue API)

Usage:
    python generate_episode.py ../../podcasts/vision-at-the-edge/dialogen/script.txt
    python generate_episode.py script.txt --section "OPENING"
    python generate_episode.py script.txt --lang de --output-dir ../../podcasts/mondriaan/productie
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
from src.voice_settings import get_voice_settings, parse_line, BASE_SIMILARITY, STABILITY_OFFSET

load_dotenv(SCRIPT_DIR / '.env')
client = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))

# Voice IDs from .env (suffix determined by --lang flag)
def load_voice_ids(lang_suffix=""):
    suffix = f"_{lang_suffix.upper()}" if lang_suffix else ""
    return {
        "emma": os.getenv(f'VOICE_EMMA{suffix}'),
        "lucas": os.getenv(f'VOICE_LUCAS{suffix}'),
        "piet": os.getenv(f'VOICE_PIET{suffix}'),
    }

VOICE_IDS = load_voice_ids()  # default, overridden in main() if --lang used

def find_sections(content):
    """Find all section names in the script"""
    sections = []
    pattern = r'={20,}\s*\n([^\n=]+)\s*\n={20,}'
    for match in re.finditer(pattern, content):
        section_name = match.group(1).strip()
        if section_name:
            sections.append(section_name)
    return sections

def generate_section(script_path, section_name, output_path):
    """Generate a section using text_to_dialogue API (high quality)"""
    print(f"\n{'='*60}")
    print(f"Section: {section_name}")
    print(f"{'='*60}")

    # Read script
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find section: match header block (===\nNAME\n===) then capture until next === or EOF
    section_pattern = (
        r'={20,}\s*\n'
        + re.escape(section_name)
        + r'\s*\n={20,}\s*\n'
        + r'(.*?)(?=\n={20,}|\Z)'
    )
    match = re.search(section_pattern, content, re.DOTALL | re.IGNORECASE)

    if not match:
        print(f"ERROR: Section not found: {section_name}")
        return False

    section_content = match.group(1).strip()
    lines = [line.strip() for line in section_content.split('\n') if line.strip()]

    # Build dialogue inputs
    dialogue_inputs = []
    for line in lines:
        speaker, emotion, text = parse_line(line)
        if speaker and text:
            voice_id = VOICE_IDS.get(speaker)
            if not voice_id:
                print(f"WARNING: Unknown speaker '{speaker}', skipping")
                continue

            settings = get_voice_settings(speaker, emotion)
            dialogue_inputs.append(
                DialogueInput(
                    text=text,
                    voice_id=voice_id,
                    voice_settings=settings
                )
            )

    if not dialogue_inputs:
        print(f"ERROR: No dialogue found in section")
        return False

    print(f"Lines: {len(dialogue_inputs)}")
    print(f"Settings: similarity={BASE_SIMILARITY}, stability_offset={STABILITY_OFFSET}")
    print(f"Generating audio...")

    # Generate audio using text_to_dialogue (best quality)
    try:
        audio = client.text_to_dialogue.convert(inputs=dialogue_inputs)

        # Save output
        with open(output_path, 'wb') as f:
            for chunk in audio:
                f.write(chunk)
    except Exception as e:
        print(f"ERROR: API call failed: {e}")
        return False

    file_size = output_path.stat().st_size
    print(f"Saved: {output_path}")
    print(f"Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")

    return True

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate podcast episode audio using ElevenLabs text_to_dialogue API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_episode.py ../../podcasts/vision-at-the-edge/dialogen/script.txt
  python generate_episode.py script.txt --section OPENING
  python generate_episode.py script.txt --lang de --output-dir ../../podcasts/mondriaan/productie
"""
    )
    parser.add_argument("script", help="Path to dialogue script file")
    parser.add_argument("--section", help="Generate only this section (case-insensitive)")
    parser.add_argument("--lang", default="", help="Language suffix for voice IDs (e.g., de, en)")
    parser.add_argument("--output-dir", help="Output directory (default: <script_dir>/../productie)")

    args = parser.parse_args()

    script_path = Path(args.script)
    if not script_path.exists():
        print(f"ERROR: Script file not found: {script_path}")
        sys.exit(1)

    # Read script to get episode info
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract episode number from filename or first line
    episode_name = script_path.stem

    # Find all sections
    sections = find_sections(content)

    if not sections:
        print("ERROR: No sections found in script")
        sys.exit(1)

    # Load voice IDs for the specified language
    global VOICE_IDS
    lang = args.lang
    if lang:
        VOICE_IDS = load_voice_ids(lang)
    # Always check for missing voice IDs
    missing = [k for k, v in VOICE_IDS.items() if not v]
    if missing:
        suffix = f"_{lang.upper()}" if lang else ""
        print(f"ERROR: Missing voice IDs: {', '.join(f'VOICE_{s.upper()}{suffix}' for s in missing)}")
        sys.exit(1)

    # Filter to specific section if requested
    if args.section:
        sections = [s for s in sections if s.upper() == args.section.upper()]
        if not sections:
            print(f"ERROR: Section '{args.section}' not found")
            print(f"Available sections: {', '.join(find_sections(content))}")
            sys.exit(1)

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir) / episode_name
    else:
        output_dir = script_path.resolve().parent.parent / "productie" / episode_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print header
    print("ELEVENLABS EPISODE GENERATOR")
    print(f"Script: {script_path}")
    print(f"Language: {lang.upper() if lang else 'NL (default)'}")
    print(f"API: text_to_dialogue (high quality)")
    print(f"Output: {output_dir}")
    print(f"Sections to generate: {len(sections)}")
    print()

    # Generate all sections
    successful = []
    failed = []

    for idx, section_name in enumerate(sections, 1):
        section_slug = re.sub(r'[^a-z0-9_\-]', '', section_name.replace(' ', '_').lower())
        if not section_slug:
            section_slug = f"section"
        output_file = output_dir / f"{episode_name}_{idx}_{section_slug}.mp3"

        success = generate_section(script_path, section_name, output_file)

        if success:
            successful.append(section_name)
        else:
            failed.append(section_name)

    # Summary
    print()
    print("="*60)
    print("GENERATION SUMMARY")
    print("="*60)
    print(f"Successful: {len(successful)}/{len(sections)}")
    print(f"Failed: {len(failed)}/{len(sections)}")

    if successful:
        print()
        print("Generated files:")
        for i, section in enumerate(successful, 1):
            print(f"  {i}. {section}")

    if failed:
        print()
        print("Failed sections:")
        for section in failed:
            print(f"  X {section}")

    print()
    print("="*60)
    if len(successful) == len(sections):
        print("SUCCESS! All sections generated.")
    print("Next: Combine sections in your DAW")
    print("="*60)

if __name__ == "__main__":
    main()
