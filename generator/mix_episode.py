#!/usr/bin/env python3
"""
Episode mixer: assembles section audio into a finished episode.

Handles:
  - Per-speaker LUFS leveling (two-pass loudnorm)
  - Intro jingle with crossfade
  - Background music bed with ducking under speech
  - Outro with crossfade
  - Final loudness normalization to podcast standard (-16 LUFS)

Usage:
    # Basic: concat sections + master
    python generator/mix_episode.py output/ep01/ -o episode.mp3

    # Full mix: intro + music bed + outro
    python generator/mix_episode.py output/ep01/ -o episode.mp3 \
        --intro assets/intro.mp3 --outro assets/outro.mp3 \
        --music-bed assets/music.mp3

    # Level speakers first (per-section files already exist)
    python generator/mix_episode.py output/ep01/ -o episode.mp3 --level

    # Dry run: show what would happen
    python generator/mix_episode.py output/ep01/ --dry-run
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_utils import get_duration


# ---------------------------------------------------------------------------
# LUFS measurement & leveling
# ---------------------------------------------------------------------------

def measure_lufs(input_path: str) -> float:
    """Measure integrated loudness (LUFS) using ffmpeg loudnorm first pass."""
    cmd = [
        "ffmpeg", "-i", str(input_path),
        "-af", "loudnorm=print_format=json",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"LUFS measurement failed: {result.stderr[:200]}")

    # Parse the JSON block from stderr
    # loudnorm prints a JSON summary at the end of stderr
    json_match = re.search(r'\{[^}]*"input_i"[^}]*\}', result.stderr, re.DOTALL)
    if not json_match:
        raise RuntimeError(f"Could not parse loudnorm output for {input_path}")
    data = json.loads(json_match.group())
    return float(data["input_i"])


def apply_gain(input_path: str, output_path: str, gain_db: float) -> None:
    """Apply a fixed gain (dB) to an audio file."""
    if abs(gain_db) < 0.1:
        # No meaningful adjustment needed — just copy
        shutil.copy2(input_path, output_path)
        return

    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-af", f"volume={gain_db:+.1f}dB",
        "-codec:a", "libmp3lame", "-b:a", "192k",
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Gain adjustment failed: {result.stderr[:200]}")


def level_files(file_paths: list[str], target_lufs: float = -18.0,
                dry_run: bool = False) -> list[tuple[str, float, float]]:
    """Measure and level a set of audio files to a common LUFS target.

    Returns list of (path, original_lufs, gain_applied_db).
    Files are modified in-place (backed up as .bak if not dry_run).
    """
    results = []
    for path in file_paths:
        lufs = measure_lufs(path)
        gain = target_lufs - lufs
        results.append((path, lufs, gain))

        if not dry_run and abs(gain) >= 0.1:
            leveled = str(path) + ".leveled.mp3"
            bak = str(path) + ".bak"
            apply_gain(path, leveled, gain)
            Path(path).replace(bak)
            try:
                Path(leveled).replace(path)
            except OSError:
                # Restore original if the second rename fails
                Path(bak).replace(path)
                Path(leveled).unlink(missing_ok=True)
                raise

    return results


# ---------------------------------------------------------------------------
# Concatenation
# ---------------------------------------------------------------------------

def find_section_files(directory: str, exclude: str | None = None) -> list[Path]:
    """Find and sort section audio files in an output directory.

    Args:
        directory: Path to directory containing section .mp3 files.
        exclude: Optional filename to exclude (e.g. the output file).
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        sys.exit(f"Not a directory: {directory}")

    # Match pattern: *_N_*.mp3 where N is section index
    exclude_names = set()
    if exclude:
        exclude_names.add(Path(exclude).name)
    # Also exclude known intermediate file patterns
    skip_suffixes = (".work.mp3", ".leveled.mp3", ".bak", ".concat.txt")

    files = [
        f for f in dir_path.glob("*.mp3")
        if f.name not in exclude_names
        and not any(f.name.endswith(s) for s in skip_suffixes)
    ]
    if not files:
        sys.exit(f"No .mp3 files found in {directory}")

    # Sort by section index if present, otherwise alphabetically
    def sort_key(f):
        match = re.search(r'_(\d+)_', f.name)
        return int(match.group(1)) if match else f.name
    return sorted(files, key=sort_key)


def concat_files(file_paths: list[str], output_path: str) -> None:
    """Concatenate audio files using ffmpeg concat demuxer."""
    # Create a temporary concat list file
    list_path = Path(output_path).with_suffix(".concat.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for path in file_paths:
            # ffmpeg concat demuxer needs forward slashes and escaped quotes
            safe = str(Path(path).resolve()).replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{safe}'\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-codec:a", "libmp3lame", "-b:a", "192k",
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    list_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"Concat failed: {result.stderr[:300]}")


# ---------------------------------------------------------------------------
# Intro / outro with crossfade
# ---------------------------------------------------------------------------

def prepend_with_crossfade(main_path: str, intro_path: str, output_path: str,
                           crossfade_sec: float = 2.0) -> None:
    """Prepend an intro jingle to the main audio with a crossfade."""
    cmd = [
        "ffmpeg", "-y", "-i", str(intro_path), "-i", str(main_path),
        "-filter_complex",
        f"[0][1]acrossfade=d={crossfade_sec}:c1=tri:c2=tri",
        "-codec:a", "libmp3lame", "-b:a", "192k",
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Intro crossfade failed: {result.stderr[:300]}")


def append_with_crossfade(main_path: str, outro_path: str, output_path: str,
                          crossfade_sec: float = 2.0) -> None:
    """Append an outro to the main audio with a crossfade."""
    cmd = [
        "ffmpeg", "-y", "-i", str(main_path), "-i", str(outro_path),
        "-filter_complex",
        f"[0][1]acrossfade=d={crossfade_sec}:c1=tri:c2=tri",
        "-codec:a", "libmp3lame", "-b:a", "192k",
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Outro crossfade failed: {result.stderr[:300]}")


# ---------------------------------------------------------------------------
# Music bed with ducking
# ---------------------------------------------------------------------------

def mix_music_bed(speech_path: str, music_path: str, output_path: str,
                  music_volume: float = 0.12,
                  fade_in: float = 3.0, fade_out: float = 5.0) -> None:
    """Mix a music bed under speech with sidechain ducking.

    The music plays at `music_volume` during silence and is automatically
    compressed when speech is present via ffmpeg sidechaincompress.

    Args:
        speech_path: Main speech audio
        music_path: Background music (will be looped/trimmed to match)
        output_path: Output file
        music_volume: Music volume during gaps (0.0-1.0)
        fade_in: Music fade-in duration at start (seconds)
        fade_out: Music fade-out duration at end (seconds)
    """
    speech_duration = get_duration(speech_path)
    fade_out_start = max(0.0, speech_duration - fade_out)

    # Build filter: loop music to match speech length, apply fade, duck under speech
    filter_complex = (
        # Loop and trim music to match speech duration, apply fades
        f"[1]aloop=loop=-1:size=2000000000,atrim=0:{speech_duration:.1f},"
        f"afade=t=in:d={fade_in},afade=t=out:st={fade_out_start:.1f}:d={fade_out},"
        f"volume={music_volume}[music];"
        # Sidechain compress: speech signal ducks the music
        # attack=10ms (fast enough for speech transients), release=800ms (smooth recovery)
        f"[music][0]sidechaincompress="
        f"threshold=0.02:ratio=8:attack=10:release=800:level_sc=1[ducked];"
        # Mix speech + ducked music (no extra weights — sidechaincompress handles ducking)
        f"[0][ducked]amix=inputs=2:duration=first:normalize=0"
    )

    cmd = [
        "ffmpeg", "-y", "-i", str(speech_path), "-i", str(music_path),
        "-filter_complex", filter_complex,
        "-codec:a", "libmp3lame", "-b:a", "192k",
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Music bed mixing failed: {result.stderr[:300]}")


# ---------------------------------------------------------------------------
# Final mastering
# ---------------------------------------------------------------------------

def master_loudnorm(input_path: str, output_path: str,
                    target_i: float = -16.0, target_tp: float = -1.5,
                    target_lra: float = 11.0) -> None:
    """Apply loudness normalization (final mastering step)."""
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-af", f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}",
        "-codec:a", "libmp3lame", "-b:a", "192k",
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Loudnorm failed: {result.stderr[:300]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Mix section audio files into a finished podcast episode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python generator/mix_episode.py output/ep01/ -o episode.mp3\n"
               "  python generator/mix_episode.py output/ep01/ -o episode.mp3 --intro intro.mp3 --outro outro.mp3\n"
               "  python generator/mix_episode.py output/ep01/ -o episode.mp3 --music-bed music.mp3\n"
               "  python generator/mix_episode.py output/ep01/ -o episode.mp3 --level --dry-run\n",
    )
    p.add_argument("input_dir",
                    help="Directory containing section audio files (*.mp3)")
    p.add_argument("-o", "--output", default="episode_mixed.mp3",
                    help="Output file path (default: episode_mixed.mp3)")
    p.add_argument("--intro",
                    help="Intro jingle audio file (crossfaded in)")
    p.add_argument("--outro",
                    help="Outro audio file (crossfaded out)")
    p.add_argument("--music-bed",
                    help="Background music file (looped, ducked under speech)")
    p.add_argument("--music-volume", type=float, default=0.12,
                    help="Music bed volume during gaps (default: 0.12)")
    p.add_argument("--crossfade", type=float, default=2.0,
                    help="Crossfade duration in seconds for intro/outro (default: 2.0)")
    p.add_argument("--level", action="store_true",
                    help="Level section files to common LUFS before mixing")
    p.add_argument("--target-lufs", type=float, default=-18.0,
                    help="Target LUFS for per-section leveling (default: -18)")
    p.add_argument("--no-master", action="store_true",
                    help="Skip final loudnorm mastering")
    p.add_argument("--dry-run", action="store_true",
                    help="Show what would happen without processing")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # --- Find section files (exclude output to avoid picking it up on re-runs) ---
    sections = find_section_files(args.input_dir, exclude=args.output)
    print(f"Found {len(sections)} section(s):")
    for s in sections:
        dur = get_duration(str(s))
        print(f"  {s.name} ({dur:.1f}s)")

    section_paths = [str(s) for s in sections]

    # --- Level ---
    if args.level:
        print(f"\nLeveling to {args.target_lufs} LUFS...")
        results = level_files(section_paths, args.target_lufs, args.dry_run)
        for path, original, gain in results:
            status = "(skip)" if abs(gain) < 0.1 else f"{gain:+.1f}dB"
            print(f"  {Path(path).name}: {original:.1f} LUFS → {status}")

    if args.dry_run:
        print("\nDry run — would perform:")
        print(f"  1. Concat {len(sections)} sections")
        if args.intro:
            print(f"  2. Prepend intro: {args.intro} (crossfade {args.crossfade}s)")
        if args.outro:
            print(f"  3. Append outro: {args.outro} (crossfade {args.crossfade}s)")
        if args.music_bed:
            print(f"  4. Mix music bed: {args.music_bed} (volume {args.music_volume})")
        if not args.no_master:
            print(f"  5. Master: loudnorm -16 LUFS")
        print(f"  → {args.output}")
        return

    # --- Pipeline with cleanup on error ---
    work_path = str(Path(args.output).with_suffix(".work.mp3"))
    try:
        # --- Concat sections ---
        print(f"\nConcatenating {len(sections)} sections...")
        concat_files(section_paths, work_path)
        total_dur = get_duration(work_path)
        print(f"  Total: {total_dur:.1f}s ({total_dur / 60:.1f} min)")

        # --- Intro ---
        if args.intro:
            print(f"Prepending intro ({args.crossfade}s crossfade)...")
            next_path = work_path + ".intro.mp3"
            prepend_with_crossfade(work_path, args.intro, next_path, args.crossfade)
            Path(work_path).unlink()
            work_path = next_path

        # --- Outro ---
        if args.outro:
            print(f"Appending outro ({args.crossfade}s crossfade)...")
            next_path = work_path + ".outro.mp3"
            append_with_crossfade(work_path, args.outro, next_path, args.crossfade)
            Path(work_path).unlink()
            work_path = next_path

        # --- Music bed ---
        if args.music_bed:
            print(f"Mixing music bed (volume {args.music_volume})...")
            next_path = work_path + ".music.mp3"
            mix_music_bed(work_path, args.music_bed, next_path,
                          music_volume=args.music_volume)
            Path(work_path).unlink()
            work_path = next_path

        # --- Master ---
        if not args.no_master:
            print("Mastering (loudnorm -16 LUFS)...")
            master_loudnorm(work_path, args.output)
            Path(work_path).unlink()
        else:
            Path(work_path).rename(args.output)

    except Exception:
        # Clean up any intermediate files on failure
        Path(work_path).unlink(missing_ok=True)
        raise

    final_dur = get_duration(args.output)
    print(f"\nDone: {args.output} ({final_dur:.1f}s / {final_dur / 60:.1f} min)")


if __name__ == "__main__":
    main()
