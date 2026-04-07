"""Tests for generator/assemble_intro.py — intro voiceover assembly."""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generator"))
from assemble_intro import (
    assemble_intro,
    build_parser,
    parse_intro_lines,
)

SR = 24000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def intro_lines_file(tmp_path):
    """Create a sample intro lines text file."""
    content = (
        "Alex: [warm] Welcome to the show.\n"
        "Morgan: [excited] Thanks for having me!\n"
        "Zara: [calm] Great to be here.\n"
    )
    path = tmp_path / "intro_lines.txt"
    path.write_text(content)
    return path


@pytest.fixture
def intro_wav_dir(tmp_path):
    """Create WAV files matching intro line format."""
    d = tmp_path / "intro"
    d.mkdir()

    for i, speaker in enumerate(["alex", "morgan", "zara"]):
        t = np.linspace(0, 0.3, int(SR * 0.3), dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * (440 + i * 50) * t).astype(np.float32)
        sf.write(str(d / f"intro_{i:03d}_{speaker}.wav"), audio, SR)

    return d


# ---------------------------------------------------------------------------
# parse_intro_lines
# ---------------------------------------------------------------------------

class TestParseIntroLines:
    def test_parses_three_lines(self, intro_lines_file):
        lines = parse_intro_lines(intro_lines_file)
        assert len(lines) == 3

    def test_speaker_lowercase(self, intro_lines_file):
        lines = parse_intro_lines(intro_lines_file)
        assert lines[0]["speaker"] == "alex"
        assert lines[1]["speaker"] == "morgan"

    def test_text_extracted(self, intro_lines_file):
        lines = parse_intro_lines(intro_lines_file)
        assert lines[0]["text"] == "Welcome to the show."

    def test_skips_empty_lines(self, tmp_path):
        path = tmp_path / "lines.txt"
        path.write_text("\n\nAlex: Hello.\n\n")
        lines = parse_intro_lines(path)
        assert len(lines) == 1

    def test_handles_no_emotion_tag(self, tmp_path):
        path = tmp_path / "lines.txt"
        path.write_text("Alex: Just speaking normally.\n")
        lines = parse_intro_lines(path)
        assert lines[0]["text"] == "Just speaking normally."


# ---------------------------------------------------------------------------
# assemble_intro
# ---------------------------------------------------------------------------

class TestAssembleIntro:
    def test_assembles_output_file(self, intro_wav_dir, intro_lines_file, tmp_path):
        lines = parse_intro_lines(intro_lines_file)
        output = tmp_path / "assembled.wav"
        duration = assemble_intro(intro_wav_dir, lines, output, sr=SR)
        assert output.exists()
        assert duration > 0

    def test_duration_includes_pauses(self, intro_wav_dir, intro_lines_file, tmp_path):
        lines = parse_intro_lines(intro_lines_file)
        output = tmp_path / "assembled.wav"
        duration = assemble_intro(intro_wav_dir, lines, output, sr=SR,
                                   default_pause=0.5)
        # 3 lines * 0.3s + 3 pauses * 0.5s = 2.4s
        assert duration == pytest.approx(2.4, abs=0.1)

    def test_speaker_pauses_override(self, intro_wav_dir, intro_lines_file, tmp_path):
        lines = parse_intro_lines(intro_lines_file)
        output_default = tmp_path / "default.wav"
        output_custom = tmp_path / "custom.wav"

        dur_default = assemble_intro(
            intro_wav_dir, lines, output_default, sr=SR, default_pause=0.15,
        )
        dur_custom = assemble_intro(
            intro_wav_dir, lines, output_custom, sr=SR, default_pause=0.15,
            speaker_pauses={"morgan": 0.8, "zara": 0.8},
        )
        assert dur_custom > dur_default

    def test_missing_files_handled(self, tmp_path, intro_lines_file):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        lines = parse_intro_lines(intro_lines_file)
        output = tmp_path / "assembled.wav"
        duration = assemble_intro(empty_dir, lines, output, sr=SR)
        assert duration == 0.0

    def test_output_is_valid_wav(self, intro_wav_dir, intro_lines_file, tmp_path):
        lines = parse_intro_lines(intro_lines_file)
        output = tmp_path / "assembled.wav"
        assemble_intro(intro_wav_dir, lines, output, sr=SR)
        audio, file_sr = sf.read(str(output), dtype="float32")
        assert file_sr == SR
        assert len(audio) > 0


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

class TestCLI:
    def test_minimal_args(self):
        args = build_parser().parse_args(["dir/", "--lines", "intro.txt"])
        assert args.intro_dir == "dir/"
        assert args.lines == "intro.txt"
        assert args.sr == 24000
        assert args.default_pause == 0.15

    def test_all_flags(self):
        args = build_parser().parse_args([
            "dir/", "--lines", "intro.txt",
            "-o", "out.wav", "--sr", "48000",
            "--default-pause", "0.3",
            "--speaker-pauses", '{"morgan": 0.5}',
        ])
        assert args.output == "out.wav"
        assert args.sr == 48000
        assert args.default_pause == 0.3
        assert args.speaker_pauses == '{"morgan": 0.5}'
