"""Tests for generator/export_stems.py — per-speaker stem export."""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generator"))
from export_stems import (
    build_parser,
    build_timeline,
    export_stems,
    parse_script_timing,
)

SR = 24000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def script_file(tmp_path):
    content = (
        "Alex: [warm] Welcome to the show.\n"
        "Morgan: [excited] Thanks for having me.\n"
        "Alex: [curious] So tell me about the project.\n"
    )
    path = tmp_path / "script.txt"
    path.write_text(content)
    return path


@pytest.fixture
def wav_dir(tmp_path):
    """Create WAV files matching script line indices."""
    d = tmp_path / "lines"
    d.mkdir()
    for i, (speaker, dur) in enumerate([("alex", 0.5), ("morgan", 0.3), ("alex", 0.4)]):
        t = np.linspace(0, dur, int(SR * dur), dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        sf.write(str(d / f"{i:03d}_{speaker}.wav"), audio, SR)
    return d


# ---------------------------------------------------------------------------
# parse_script_timing
# ---------------------------------------------------------------------------

class TestParseScriptTiming:
    def test_parses_three_lines(self, script_file):
        entries = parse_script_timing(script_file)
        assert len(entries) == 3

    def test_speaker_lowercase(self, script_file):
        entries = parse_script_timing(script_file)
        assert entries[0]["speaker"] == "alex"
        assert entries[1]["speaker"] == "morgan"

    def test_sequential_indices(self, script_file):
        entries = parse_script_timing(script_file)
        assert [e["index"] for e in entries] == [0, 1, 2]

    def test_skips_headers(self, tmp_path):
        path = tmp_path / "s.txt"
        path.write_text("=" * 30 + "\nSECTION\n" + "=" * 30 + "\nAlex: [warm] Hi.\n")
        entries = parse_script_timing(path)
        assert len(entries) == 1


# ---------------------------------------------------------------------------
# build_timeline
# ---------------------------------------------------------------------------

class TestBuildTimeline:
    def test_builds_from_wav_files(self, wav_dir, script_file):
        entries = parse_script_timing(script_file)
        total, timeline = build_timeline(wav_dir, entries, sr=SR)
        assert total > 0
        assert len(timeline) == 3

    def test_timeline_positions_increase(self, wav_dir, script_file):
        entries = parse_script_timing(script_file)
        _, timeline = build_timeline(wav_dir, entries, sr=SR)
        starts = [t["start_sample"] for t in timeline]
        assert starts == sorted(starts)

    def test_empty_dir_returns_empty(self, tmp_path, script_file):
        empty = tmp_path / "empty"
        empty.mkdir()
        entries = parse_script_timing(script_file)
        total, timeline = build_timeline(empty, entries, sr=SR)
        assert total == 0
        assert timeline == []


# ---------------------------------------------------------------------------
# export_stems
# ---------------------------------------------------------------------------

class TestExportStems:
    def test_creates_per_speaker_stems(self, wav_dir, script_file, tmp_path):
        entries = parse_script_timing(script_file)
        out = tmp_path / "stems"
        stem_paths = export_stems(wav_dir, entries, out, sr=SR)
        assert "alex" in stem_paths
        assert "morgan" in stem_paths
        assert Path(stem_paths["alex"]).exists()
        assert Path(stem_paths["morgan"]).exists()

    def test_all_stems_same_length(self, wav_dir, script_file, tmp_path):
        entries = parse_script_timing(script_file)
        out = tmp_path / "stems"
        stem_paths = export_stems(wav_dir, entries, out, sr=SR)
        lengths = set()
        for path in stem_paths.values():
            audio, _ = sf.read(path, dtype="float32")
            lengths.add(len(audio))
        assert len(lengths) == 1  # all same length

    def test_creates_lof_file(self, wav_dir, script_file, tmp_path):
        entries = parse_script_timing(script_file)
        out = tmp_path / "stems"
        export_stems(wav_dir, entries, out, sr=SR)
        lof = out / "import.lof"
        assert lof.exists()
        content = lof.read_text()
        assert "stem_alex.wav" in content
        assert "stem_morgan.wav" in content

    def test_speaker_audio_at_correct_positions(self, wav_dir, script_file, tmp_path):
        entries = parse_script_timing(script_file)
        out = tmp_path / "stems"
        export_stems(wav_dir, entries, out, sr=SR)
        # Alex stem should have audio at start (line 0) and later (line 2)
        alex_audio, _ = sf.read(str(out / "stem_alex.wav"), dtype="float32")
        # First 100 samples should be non-zero (Alex's first line)
        assert np.max(np.abs(alex_audio[:100])) > 0

    def test_no_files_returns_empty(self, tmp_path, script_file):
        empty = tmp_path / "empty"
        empty.mkdir()
        entries = parse_script_timing(script_file)
        out = tmp_path / "stems"
        stem_paths = export_stems(empty, entries, out, sr=SR)
        assert stem_paths == {}


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

class TestCLI:
    def test_minimal_args(self):
        args = build_parser().parse_args(["dir/", "--script", "s.txt"])
        assert args.audio_dir == "dir/"
        assert args.script == "s.txt"
        assert args.sr == 24000

    def test_all_flags(self):
        args = build_parser().parse_args([
            "dir/", "--script", "s.txt", "-o", "out/",
            "--sr", "48000", "--speaker-pause", "0.2", "--same-pause", "0.1",
        ])
        assert args.output_dir == "out/"
        assert args.sr == 48000
        assert args.speaker_pause == 0.2
        assert args.same_pause == 0.1
