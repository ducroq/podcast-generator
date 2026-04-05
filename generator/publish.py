#!/usr/bin/env python3
"""
Publish pipeline: generate chapters, transcript, and show notes for an episode.

Takes the section audio directory and the original script, then produces
Podcasting 2.0 JSON chapters, a speaker-labeled SRT transcript, and
markdown show notes.

Usage:
    # Basic: generate all publishing artifacts
    python generator/publish.py output/ep01/ --script script.txt

    # With intro offset (same intro used in mix_episode)
    python generator/publish.py output/ep01/ --script script.txt --intro intro.mp3

    # Custom output directory
    python generator/publish.py output/ep01/ --script script.txt -o published/

    # Preview without writing files
    python generator/publish.py output/ep01/ --script script.txt --dry-run
"""

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_utils import get_duration
from mix_episode import find_section_files

sys.path.insert(0, str(Path(__file__).resolve().parent / "elevenlabs"))
from src.dialogue_parser import DialogueParser


# ---------------------------------------------------------------------------
# Section parsing (duplicated from generate_episode.py:find_sections to avoid deep import)
# Keep in sync with generator/elevenlabs/generate_episode.py line ~40
# ---------------------------------------------------------------------------

SECTION_PATTERN = re.compile(r'={20,}\s*\n([^\n=]+)\s*\n={20,}')


def find_sections(content: str) -> list[str]:
    """Extract section names from a script."""
    return [m.group(1).strip() for m in SECTION_PATTERN.finditer(content) if m.group(1).strip()]


def extract_section_text(content: str, section_name: str) -> str:
    """Return the text between a section header and the next header (or EOF)."""
    pattern = (
        r'={20,}\s*\n'
        + re.escape(section_name)
        + r'\s*\n={20,}\s*\n'
        + r'(.*?)'
        + r'(?=={20,}\s*\n|\Z)'
    )
    match = re.search(pattern, content, re.DOTALL)
    return match.group(1).strip() if match else ""


# ---------------------------------------------------------------------------
# Chapters (Podcasting 2.0 JSON)
# ---------------------------------------------------------------------------

def compute_chapters(
    section_files: list[Path],
    section_names: list[str],
    intro_offset: float = 0.0,
) -> list[dict]:
    """Compute chapter start times from section audio durations.

    Returns a list of {"startTime": float, "title": str} dicts conforming
    to the Podcasting 2.0 JSON Chapters spec.
    """
    chapters = []
    cursor = intro_offset
    # Use the shorter of the two lists in case of mismatch
    count = min(len(section_files), len(section_names))
    for i in range(count):
        chapters.append({
            "startTime": round(cursor, 1),
            "title": section_names[i],
        })
        cursor += get_duration(str(section_files[i]))
    return chapters


def write_chapters_json(chapters: list[dict], output_path: Path) -> None:
    """Write Podcasting 2.0 JSON chapters file."""
    data = {"version": "1.2.0", "chapters": chapters}
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Transcript (SRT with speaker labels)
# ---------------------------------------------------------------------------

def estimate_line_timestamps(
    lines: list,  # List[DialogueLine]
    section_start: float,
    section_duration: float,
) -> list[dict]:
    """Distribute section duration across dialogue lines by word count.

    Returns list of {"start": float, "end": float, "speaker": str, "text": str}.
    """
    if not lines or section_duration <= 0:
        return []

    word_counts = [len(line.text.split()) for line in lines]
    total_words = sum(word_counts)
    if total_words == 0:
        # Equal distribution as fallback
        total_words = len(lines)
        word_counts = [1] * len(lines)

    entries = []
    cursor = section_start
    for i, line in enumerate(lines):
        fraction = word_counts[i] / total_words
        duration = section_duration * fraction
        entries.append({
            "start": round(cursor, 3),
            "end": round(cursor + duration, 3),
            "speaker": line.speaker,
            "text": line.text,
        })
        cursor += duration
    # Clamp last entry to exact section boundary to avoid float drift
    if entries:
        entries[-1]["end"] = round(section_start + section_duration, 3)
    return entries


def build_transcript(
    script_content: str,
    section_files: list[Path],
    section_names: list[str],
    intro_offset: float = 0.0,
) -> list[dict]:
    """Build a full transcript with estimated timestamps from script + section audio."""
    parser = DialogueParser()
    entries = []
    cursor = intro_offset
    count = min(len(section_files), len(section_names))

    for i in range(count):
        section_text = extract_section_text(script_content, section_names[i])
        lines = parser.parse_text(section_text)
        duration = get_duration(str(section_files[i]))
        entries.extend(estimate_line_timestamps(lines, cursor, duration))
        cursor += duration

    return entries


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm."""
    total_ms = round(seconds * 1000)
    h, remainder = divmod(total_ms, 3_600_000)
    m, remainder = divmod(remainder, 60_000)
    s, ms = divmod(remainder, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_srt(entries: list[dict]) -> str:
    """Format transcript entries as SRT with speaker labels."""
    blocks = []
    for i, e in enumerate(entries, 1):
        start = _format_srt_time(e["start"])
        end = _format_srt_time(e["end"])
        blocks.append(f"{i}\n{start} --> {end}\n[{e['speaker']}] {e['text']}")
    return "\n\n".join(blocks) + "\n" if blocks else ""


def write_transcript_srt(entries: list[dict], output_path: Path) -> None:
    """Write SRT transcript file."""
    output_path.write_text(format_srt(entries), encoding="utf-8")


# ---------------------------------------------------------------------------
# Show notes (Markdown)
# ---------------------------------------------------------------------------

def build_show_notes(
    section_names: list[str],
    speakers: list[str],
    script_content: str,
    title: str = "",
) -> str:
    """Generate markdown show notes from script metadata."""
    parser = DialogueParser()
    parts = []

    if title:
        parts.append(f"# {title}\n")

    # Speakers
    if speakers:
        parts.append("## Speakers\n")
        for s in speakers:
            parts.append(f"- {s}")
        parts.append("")

    # Sections with first dialogue line as topic hint
    if section_names:
        parts.append("## Sections\n")
        for name in section_names:
            section_text = extract_section_text(script_content, name)
            lines = parser.parse_text(section_text)
            hint = f" -- {textwrap.shorten(lines[0].text, 80, placeholder='...')}" if lines else ""
            parts.append(f"- **{name}**{hint}")
        parts.append("")

    return "\n".join(parts)


def write_show_notes(notes: str, output_path: Path) -> None:
    """Write show notes markdown file."""
    output_path.write_text(notes, encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def publish(
    section_dir: Path,
    script: Path,
    output_dir: Path,
    intro: Path | None = None,
    crossfade: float = 2.0,
    title: str = "",
    dry_run: bool = False,
) -> dict:
    """Generate all publishing artifacts for an episode.

    Returns a summary dict with paths to generated files.
    """
    # Read script
    script_content = script.read_text(encoding="utf-8")
    section_names = find_sections(script_content)

    # Find section audio files
    section_files = find_section_files(str(section_dir))

    # Compute intro offset
    intro_offset = 0.0
    if intro:
        intro_offset = max(0.0, get_duration(str(intro)) - crossfade)

    # Infer title from script filename if not provided
    if not title:
        title = script.stem.replace("_", " ").replace("-", " ").title()

    # Parse all speakers
    parser = DialogueParser()
    all_lines = parser.parse_text(script_content)
    speakers = parser.get_speakers(all_lines)

    # Handle no-sections case: treat entire script as one chapter
    if not section_names:
        section_names = [title]

    # Align section count: if fewer names than files, use file count
    if len(section_names) < len(section_files):
        # Pad with numbered sections
        for i in range(len(section_names), len(section_files)):
            section_names.append(f"Section {i + 1}")

    print(f"Episode: {title}")
    print(f"Sections: {len(section_names)}")
    print(f"Section audio files: {len(section_files)}")
    print(f"Speakers: {', '.join(speakers)}")
    if intro_offset > 0:
        print(f"Intro offset: {intro_offset:.1f}s")

    if dry_run:
        print("\n[dry run] Would generate:")
        print(f"  {output_dir / 'chapters.json'}")
        print(f"  {output_dir / 'transcript.srt'}")
        print(f"  {output_dir / 'show_notes.md'}")
        return {"dry_run": True}

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate chapters
    chapters = compute_chapters(section_files, section_names, intro_offset)
    chapters_path = output_dir / "chapters.json"
    write_chapters_json(chapters, chapters_path)
    print(f"\nChapters: {chapters_path} ({len(chapters)} chapters)")

    # Generate transcript
    entries = build_transcript(script_content, section_files, section_names, intro_offset)
    transcript_path = output_dir / "transcript.srt"
    write_transcript_srt(entries, transcript_path)
    print(f"Transcript: {transcript_path} ({len(entries)} lines)")

    # Generate show notes
    notes = build_show_notes(section_names, speakers, script_content, title)
    notes_path = output_dir / "show_notes.md"
    write_show_notes(notes, notes_path)
    print(f"Show notes: {notes_path}")

    return {
        "chapters": str(chapters_path),
        "transcript": str(transcript_path),
        "show_notes": str(notes_path),
        "chapter_count": len(chapters),
        "transcript_lines": len(entries),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate publishing artifacts (chapters, transcript, show notes) for an episode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python generator/publish.py output/ep01/ --script script.txt\n"
            "  python generator/publish.py output/ep01/ --script script.txt --intro intro.mp3\n"
            "  python generator/publish.py output/ep01/ --script script.txt -o published/ --dry-run\n"
        ),
    )
    p.add_argument("section_dir", help="Directory with per-section audio files")
    p.add_argument("--script", required=True, help="Path to dialogue script")
    p.add_argument("-o", "--output-dir", default=None, help="Output directory (default: published/ inside section_dir)")
    p.add_argument("--intro", default=None, help="Intro audio file (to calculate chapter time offset)")
    p.add_argument("--crossfade", type=float, default=2.0, help="Crossfade seconds used with intro (default: 2.0)")
    p.add_argument("--title", default="", help="Episode title override (default: from script filename)")
    p.add_argument("--dry-run", action="store_true", help="Show what would be generated without writing files")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    section_dir = Path(args.section_dir)
    script = Path(args.script)
    output_dir = Path(args.output_dir) if args.output_dir else section_dir / "published"
    intro = Path(args.intro) if args.intro else None

    if not script.exists():
        sys.exit(f"Script not found: {script}")
    if intro and not intro.exists():
        sys.exit(f"Intro file not found: {intro}")

    publish(
        section_dir=section_dir,
        script=script,
        output_dir=output_dir,
        intro=intro,
        crossfade=args.crossfade,
        title=args.title,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
