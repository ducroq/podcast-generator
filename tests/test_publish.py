"""Tests for generator/publish.py — chapters, transcript, and show notes."""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from publish import (
    build_show_notes,
    build_transcript,
    compute_chapters,
    estimate_line_timestamps,
    extract_section_text,
    find_sections,
    format_srt,
    publish,
    _format_srt_time,
    build_parser,
    main,
)
from src.dialogue_parser import DialogueLine


# ---------------------------------------------------------------------------
# Sample script for testing
# ---------------------------------------------------------------------------

SAMPLE_SCRIPT = """\
==================================================
OPENING
==================================================

LISA: [excited] Welcome to the show!
MARC: [warm] Great to be here, Lisa.

==================================================
DISCUSSION
==================================================

LISA: [curious] So what is quantum computing?
MARC: [thoughtful] It uses qubits instead of classical bits.
LISA: [surprised] That sounds revolutionary.

==================================================
CLOSING
==================================================

LISA: [warm] Thanks for listening everyone.
MARC: [calm] See you next time.
"""

SCRIPT_NO_SECTIONS = """\
LISA: [excited] Welcome to the show!
MARC: [warm] Great to be here.
LISA: [curious] Let us dive in.
"""


# ---------------------------------------------------------------------------
# find_sections / extract_section_text
# ---------------------------------------------------------------------------

class TestFindSections:
    def test_finds_all_sections(self):
        assert find_sections(SAMPLE_SCRIPT) == ["OPENING", "DISCUSSION", "CLOSING"]

    def test_no_sections(self):
        assert find_sections(SCRIPT_NO_SECTIONS) == []

    def test_empty_string(self):
        assert find_sections("") == []


class TestExtractSectionText:
    def test_extracts_opening(self):
        text = extract_section_text(SAMPLE_SCRIPT, "OPENING")
        assert "Welcome to the show" in text
        assert "quantum computing" not in text

    def test_extracts_middle_section(self):
        text = extract_section_text(SAMPLE_SCRIPT, "DISCUSSION")
        assert "quantum computing" in text
        assert "Welcome to the show" not in text

    def test_extracts_last_section(self):
        text = extract_section_text(SAMPLE_SCRIPT, "CLOSING")
        assert "Thanks for listening" in text

    def test_missing_section(self):
        assert extract_section_text(SAMPLE_SCRIPT, "NONEXISTENT") == ""


# ---------------------------------------------------------------------------
# Chapters
# ---------------------------------------------------------------------------

class TestComputeChapters:
    @patch("publish.get_duration")
    def test_basic_chapters(self, mock_dur):
        mock_dur.side_effect = [30.0, 120.0, 45.0]
        files = [Path(f"section_{i}.mp3") for i in range(3)]
        names = ["Opening", "Discussion", "Closing"]

        chapters = compute_chapters(files, names)

        assert len(chapters) == 3
        assert chapters[0] == {"startTime": 0.0, "title": "Opening"}
        assert chapters[1] == {"startTime": 30.0, "title": "Discussion"}
        assert chapters[2] == {"startTime": 150.0, "title": "Closing"}

    @patch("publish.get_duration")
    def test_with_intro_offset(self, mock_dur):
        mock_dur.side_effect = [30.0, 120.0]
        files = [Path(f"section_{i}.mp3") for i in range(2)]
        names = ["Opening", "Discussion"]

        chapters = compute_chapters(files, names, intro_offset=15.0)

        assert chapters[0]["startTime"] == 15.0
        assert chapters[1]["startTime"] == 45.0

    @patch("publish.get_duration")
    def test_single_section(self, mock_dur):
        mock_dur.return_value = 300.0
        chapters = compute_chapters([Path("only.mp3")], ["Full Episode"])

        assert len(chapters) == 1
        assert chapters[0] == {"startTime": 0.0, "title": "Full Episode"}

    @patch("publish.get_duration")
    def test_mismatched_counts_uses_shorter(self, mock_dur):
        mock_dur.return_value = 60.0
        files = [Path(f"s_{i}.mp3") for i in range(3)]
        names = ["A", "B"]  # fewer names than files

        chapters = compute_chapters(files, names)
        assert len(chapters) == 2


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------

class TestEstimateLineTimestamps:
    def test_proportional_distribution(self):
        lines = [
            DialogueLine("LISA", "One two three", "excited"),      # 3 words
            DialogueLine("MARC", "One two three four five six", "calm"),  # 6 words
        ]
        entries = estimate_line_timestamps(lines, section_start=0.0, section_duration=9.0)

        assert len(entries) == 2
        # 3/9 words → 3.0s, 6/9 words → 6.0s
        assert entries[0]["start"] == 0.0
        assert entries[0]["end"] == 3.0
        assert entries[0]["speaker"] == "LISA"
        assert entries[1]["start"] == 3.0
        assert entries[1]["end"] == 9.0

    def test_with_offset(self):
        lines = [DialogueLine("LISA", "Hello world", "default")]
        entries = estimate_line_timestamps(lines, section_start=10.0, section_duration=5.0)

        assert entries[0]["start"] == 10.0
        assert entries[0]["end"] == 15.0

    def test_empty_lines(self):
        assert estimate_line_timestamps([], section_start=0.0, section_duration=10.0) == []

    def test_zero_duration(self):
        lines = [DialogueLine("LISA", "Hello", "default")]
        assert estimate_line_timestamps(lines, section_start=0.0, section_duration=0.0) == []

    def test_single_line(self):
        lines = [DialogueLine("MARC", "Just one line here", "calm")]
        entries = estimate_line_timestamps(lines, section_start=5.0, section_duration=10.0)

        assert len(entries) == 1
        assert entries[0]["start"] == 5.0
        assert entries[0]["end"] == 15.0


class TestFormatSrt:
    def test_basic_format(self):
        entries = [
            {"start": 0.0, "end": 3.5, "speaker": "LISA", "text": "Hello everyone."},
            {"start": 3.5, "end": 8.0, "speaker": "MARC", "text": "Welcome to the show."},
        ]
        srt = format_srt(entries)

        assert "1\n00:00:00,000 --> 00:00:03,500\n[LISA] Hello everyone." in srt
        assert "2\n00:00:03,500 --> 00:00:08,000\n[MARC] Welcome to the show." in srt

    def test_hour_timestamps(self):
        entries = [{"start": 3661.5, "end": 3665.0, "speaker": "LISA", "text": "Late in the show."}]
        srt = format_srt(entries)
        assert "01:01:01,500 --> 01:01:05,000" in srt

    def test_empty_entries(self):
        assert format_srt([]) == ""

    def test_ends_with_newline(self):
        entries = [{"start": 0.0, "end": 1.0, "speaker": "A", "text": "Hi."}]
        assert format_srt(entries).endswith("\n")


class TestFormatSrtTime:
    def test_zero(self):
        assert _format_srt_time(0.0) == "00:00:00,000"

    def test_milliseconds(self):
        assert _format_srt_time(1.234) == "00:00:01,234"

    def test_minutes(self):
        assert _format_srt_time(125.5) == "00:02:05,500"

    def test_hours(self):
        assert _format_srt_time(7200.0) == "02:00:00,000"


# ---------------------------------------------------------------------------
# Show notes
# ---------------------------------------------------------------------------

class TestBuildShowNotes:
    def test_basic_notes(self):
        notes = build_show_notes(
            section_names=["OPENING", "DISCUSSION", "CLOSING"],
            speakers=["LISA", "MARC"],
            script_content=SAMPLE_SCRIPT,
            title="Test Episode",
        )
        assert "# Test Episode" in notes
        assert "LISA" in notes
        assert "MARC" in notes
        assert "**OPENING**" in notes
        assert "**DISCUSSION**" in notes
        assert "**CLOSING**" in notes

    def test_includes_first_line_hints(self):
        notes = build_show_notes(
            section_names=["OPENING"],
            speakers=["LISA"],
            script_content=SAMPLE_SCRIPT,
        )
        assert "Welcome to the show" in notes

    def test_no_title(self):
        notes = build_show_notes([], [], "")
        assert "# " not in notes

    def test_no_sections(self):
        notes = build_show_notes([], ["LISA"], SCRIPT_NO_SECTIONS)
        assert "## Speakers" in notes
        assert "## Sections" not in notes


# ---------------------------------------------------------------------------
# Build transcript (integration with section parsing)
# ---------------------------------------------------------------------------

class TestBuildTranscript:
    @patch("publish.get_duration")
    def test_builds_from_script(self, mock_dur):
        mock_dur.side_effect = [10.0, 15.0, 8.0]
        files = [Path(f"section_{i}.mp3") for i in range(3)]
        names = ["OPENING", "DISCUSSION", "CLOSING"]

        entries = build_transcript(SAMPLE_SCRIPT, files, names)

        assert len(entries) == 7  # 2 + 3 + 2 lines
        # First entry is from OPENING
        assert entries[0]["speaker"] == "LISA"
        assert "Welcome" in entries[0]["text"]
        assert entries[0]["start"] == 0.0
        # Last entry is from CLOSING
        assert entries[-1]["speaker"] == "MARC"
        assert "next time" in entries[-1]["text"]

    @patch("publish.get_duration")
    def test_with_intro_offset(self, mock_dur):
        mock_dur.side_effect = [10.0, 15.0, 8.0]
        files = [Path(f"s_{i}.mp3") for i in range(3)]
        names = ["OPENING", "DISCUSSION", "CLOSING"]

        entries = build_transcript(SAMPLE_SCRIPT, files, names, intro_offset=5.0)

        assert entries[0]["start"] == 5.0


# ---------------------------------------------------------------------------
# Integration: publish() orchestrator
# ---------------------------------------------------------------------------

@pytest.fixture
def section_audio_files(tmp_path):
    """Create 3 short test audio files simulating section outputs."""
    files = []
    for i in range(3):
        path = tmp_path / f"episode_test_{i + 1}_section{i + 1}.mp3"
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono",
            "-t", str(3 + i),  # 3s, 4s, 5s
            "-codec:a", "libmp3lame", "-b:a", "64k",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"ffmpeg failed: {result.stderr[-200:]}"
        files.append(path)
    return files


@pytest.fixture
def sample_script(tmp_path):
    """Write sample script to a temp file."""
    path = tmp_path / "test_script.txt"
    path.write_text(SAMPLE_SCRIPT, encoding="utf-8")
    return path


class TestPublishEndToEnd:
    def test_generates_all_files(self, section_audio_files, sample_script, tmp_path):
        output_dir = tmp_path / "published"

        result = publish(
            section_dir=tmp_path,
            script=sample_script,
            output_dir=output_dir,
        )

        assert (output_dir / "chapters.json").exists()
        assert (output_dir / "transcript.srt").exists()
        assert (output_dir / "show_notes.md").exists()

        # Validate chapters.json
        chapters = json.loads((output_dir / "chapters.json").read_text())
        assert chapters["version"] == "1.2.0"
        assert len(chapters["chapters"]) == 3
        assert chapters["chapters"][0]["title"] == "OPENING"
        assert chapters["chapters"][0]["startTime"] == 0.0

        # Validate transcript.srt
        srt = (output_dir / "transcript.srt").read_text()
        assert "[LISA]" in srt
        assert "[MARC]" in srt
        assert "-->" in srt

        # Validate show_notes.md
        notes = (output_dir / "show_notes.md").read_text()
        assert "OPENING" in notes
        assert "LISA" in notes

        # Validate return value
        assert result["chapter_count"] == 3
        assert result["transcript_lines"] == 7

    def test_dry_run_no_files(self, section_audio_files, sample_script, tmp_path):
        output_dir = tmp_path / "published"

        result = publish(
            section_dir=tmp_path,
            script=sample_script,
            output_dir=output_dir,
            dry_run=True,
        )

        assert result["dry_run"] is True
        assert not output_dir.exists()

    def test_custom_title(self, section_audio_files, sample_script, tmp_path):
        output_dir = tmp_path / "published"

        publish(
            section_dir=tmp_path,
            script=sample_script,
            output_dir=output_dir,
            title="My Custom Episode",
        )

        notes = (output_dir / "show_notes.md").read_text()
        assert "# My Custom Episode" in notes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_build_parser_defaults(self):
        p = build_parser()
        args = p.parse_args(["mydir", "--script", "s.txt"])
        assert args.section_dir == "mydir"
        assert args.script == "s.txt"
        assert args.output_dir is None
        assert args.crossfade == 2.0
        assert args.dry_run is False

    def test_build_parser_all_args(self):
        p = build_parser()
        args = p.parse_args([
            "mydir", "--script", "s.txt",
            "-o", "out/", "--intro", "intro.mp3",
            "--crossfade", "3.0", "--title", "Ep 1", "--dry-run",
        ])
        assert args.output_dir == "out/"
        assert args.intro == "intro.mp3"
        assert args.crossfade == 3.0
        assert args.title == "Ep 1"
        assert args.dry_run is True

    def test_main_dry_run(self, section_audio_files, sample_script, tmp_path):
        main([
            str(tmp_path),
            "--script", str(sample_script),
            "--dry-run",
        ])
        # Should not create output directory
        assert not (tmp_path / "published").exists()
