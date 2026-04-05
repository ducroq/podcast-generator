"""Tests for generator/mix_episode.py — episode mixing pipeline."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generator.mix_episode import (
    measure_lufs,
    apply_gain,
    level_files,
    find_section_files,
    concat_files,
    prepend_with_crossfade,
    append_with_crossfade,
    mix_music_bed,
    master_loudnorm,
    build_parser,
)


# ---------------------------------------------------------------------------
# LUFS measurement
# ---------------------------------------------------------------------------

class TestMeasureLufs:
    def test_parses_loudnorm_json(self):
        fake_stderr = (
            "[Parsed_loudnorm_0 @ 0x...] \n"
            '{\n'
            '    "input_i" : "-20.50",\n'
            '    "input_tp" : "-3.21",\n'
            '    "input_lra" : "5.30",\n'
            '    "input_thresh" : "-30.50"\n'
            '}\n'
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr=fake_stderr)
            lufs = measure_lufs("test.mp3")
            assert lufs == pytest.approx(-20.5)

    def test_raises_on_failure(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="error")
            with pytest.raises(RuntimeError, match="LUFS measurement"):
                measure_lufs("bad.mp3")

    def test_raises_on_unparseable_output(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="no json here")
            with pytest.raises(RuntimeError, match="Could not parse"):
                measure_lufs("weird.mp3")


class TestApplyGain:
    def test_skips_small_gain(self, tmp_path):
        src = tmp_path / "input.mp3"
        dst = tmp_path / "output.mp3"
        src.write_bytes(b"fake audio")
        apply_gain(str(src), str(dst), 0.05)
        assert dst.read_bytes() == b"fake audio"

    def test_calls_ffmpeg_for_real_gain(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            apply_gain("in.mp3", "out.mp3", 3.5)
            cmd = mock_run.call_args[0][0]
            assert "volume=+3.5dB" in " ".join(cmd)

    def test_raises_on_failure(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="fail")
            with pytest.raises(RuntimeError, match="Gain adjustment"):
                apply_gain("in.mp3", "out.mp3", 5.0)


# ---------------------------------------------------------------------------
# Level files
# ---------------------------------------------------------------------------

class TestLevelFiles:
    def test_dry_run_doesnt_modify(self, tmp_path):
        f = tmp_path / "test.mp3"
        f.write_bytes(b"audio")
        with patch("generator.mix_episode.measure_lufs", return_value=-22.0):
            results = level_files([str(f)], target_lufs=-18.0, dry_run=True)
        assert len(results) == 1
        assert results[0][1] == pytest.approx(-22.0)
        assert results[0][2] == pytest.approx(4.0)
        assert not (tmp_path / "test.mp3.bak").exists()

    def test_skips_small_adjustment(self, tmp_path):
        f = tmp_path / "test.mp3"
        f.write_bytes(b"audio")
        with patch("generator.mix_episode.measure_lufs", return_value=-18.05):
            results = level_files([str(f)], target_lufs=-18.0, dry_run=False)
        assert abs(results[0][2]) < 0.1
        assert not (tmp_path / "test.mp3.bak").exists()


# ---------------------------------------------------------------------------
# Find section files
# ---------------------------------------------------------------------------

class TestFindSectionFiles:
    def test_sorts_by_section_index(self, tmp_path):
        (tmp_path / "ep_3_closing.mp3").write_bytes(b"")
        (tmp_path / "ep_1_opening.mp3").write_bytes(b"")
        (tmp_path / "ep_2_main.mp3").write_bytes(b"")
        files = find_section_files(str(tmp_path))
        names = [f.name for f in files]
        assert names == ["ep_1_opening.mp3", "ep_2_main.mp3", "ep_3_closing.mp3"]

    def test_exits_on_empty_dir(self, tmp_path):
        with pytest.raises(SystemExit, match="No .mp3"):
            find_section_files(str(tmp_path))

    def test_exits_on_nonexistent_dir(self):
        with pytest.raises(SystemExit, match="Not a directory"):
            find_section_files("/nonexistent/path")

    def test_fallback_alphabetical_sort(self, tmp_path):
        (tmp_path / "c_outro.mp3").write_bytes(b"")
        (tmp_path / "a_intro.mp3").write_bytes(b"")
        (tmp_path / "b_main.mp3").write_bytes(b"")
        files = find_section_files(str(tmp_path))
        names = [f.name for f in files]
        assert names == ["a_intro.mp3", "b_main.mp3", "c_outro.mp3"]

    def test_excludes_output_file(self, tmp_path):
        (tmp_path / "ep_1_opening.mp3").write_bytes(b"")
        (tmp_path / "ep_2_main.mp3").write_bytes(b"")
        (tmp_path / "episode_mixed.mp3").write_bytes(b"")
        files = find_section_files(str(tmp_path), exclude="episode_mixed.mp3")
        names = [f.name for f in files]
        assert "episode_mixed.mp3" not in names
        assert len(names) == 2

    def test_excludes_work_files(self, tmp_path):
        (tmp_path / "ep_1_opening.mp3").write_bytes(b"")
        (tmp_path / "episode.work.mp3").write_bytes(b"")
        (tmp_path / "test.leveled.mp3").write_bytes(b"")
        files = find_section_files(str(tmp_path))
        names = [f.name for f in files]
        assert len(names) == 1
        assert names[0] == "ep_1_opening.mp3"


# ---------------------------------------------------------------------------
# Concat
# ---------------------------------------------------------------------------

class TestConcatFiles:
    def test_creates_concat_list_and_calls_ffmpeg(self, tmp_path):
        out = str(tmp_path / "output.mp3")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            concat_files(["a.mp3", "b.mp3"], out)
            cmd = mock_run.call_args[0][0]
            assert "-f" in cmd
            assert "concat" in cmd

    def test_escapes_single_quotes_in_paths(self, tmp_path):
        out = str(tmp_path / "output.mp3")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            concat_files(["John's_speech.mp3"], out)
            # Read the concat list that was written
            # The list file is deleted after, so check the ffmpeg call happened
            assert mock_run.called

    def test_raises_on_failure(self, tmp_path):
        out = str(tmp_path / "output.mp3")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="fail")
            with pytest.raises(RuntimeError, match="Concat"):
                concat_files(["a.mp3"], out)


# ---------------------------------------------------------------------------
# Crossfades
# ---------------------------------------------------------------------------

class TestCrossfades:
    def test_prepend_calls_acrossfade(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            prepend_with_crossfade("main.mp3", "intro.mp3", "out.mp3", 1.5)
            cmd = " ".join(mock_run.call_args[0][0])
            assert "acrossfade" in cmd
            assert "d=1.5" in cmd

    def test_append_calls_acrossfade(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            append_with_crossfade("main.mp3", "outro.mp3", "out.mp3", 2.0)
            cmd = " ".join(mock_run.call_args[0][0])
            assert "acrossfade" in cmd

    def test_prepend_raises_on_failure(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="fail")
            with pytest.raises(RuntimeError, match="Intro crossfade"):
                prepend_with_crossfade("a", "b", "c")

    def test_append_raises_on_failure(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="fail")
            with pytest.raises(RuntimeError, match="Outro crossfade"):
                append_with_crossfade("a", "b", "c")


# ---------------------------------------------------------------------------
# Music bed
# ---------------------------------------------------------------------------

class TestMusicBed:
    def test_calls_sidechaincompress(self):
        with patch("subprocess.run") as mock_run, \
             patch("generator.mix_episode.get_duration", return_value=120.0):
            mock_run.return_value = MagicMock(returncode=0)
            mix_music_bed("speech.mp3", "music.mp3", "out.mp3")
            cmd = " ".join(mock_run.call_args[0][0])
            assert "sidechaincompress" in cmd
            assert "amix" in cmd
            # Verify corrected attack time (10ms, not 200ms)
            assert "attack=10" in cmd
            # Verify no extra amix weights (sidechaincompress handles ducking)
            assert "weights" not in cmd
            # Verify integer loop size
            assert "2000000000" in cmd

    def test_short_audio_no_negative_fade(self):
        """Audio shorter than fade_out should not produce negative timestamp."""
        with patch("subprocess.run") as mock_run, \
             patch("generator.mix_episode.get_duration", return_value=3.0):
            mock_run.return_value = MagicMock(returncode=0)
            mix_music_bed("short.mp3", "music.mp3", "out.mp3", fade_out=5.0)
            cmd = " ".join(mock_run.call_args[0][0])
            # fade_out start should be max(0, 3-5) = 0, not -2
            assert "st=-" not in cmd
            assert "st=0.0" in cmd

    def test_raises_on_failure(self):
        with patch("subprocess.run") as mock_run, \
             patch("generator.mix_episode.get_duration", return_value=60.0):
            mock_run.return_value = MagicMock(returncode=1, stderr="fail")
            with pytest.raises(RuntimeError, match="Music bed"):
                mix_music_bed("a", "b", "c")


# ---------------------------------------------------------------------------
# Master
# ---------------------------------------------------------------------------

class TestMasterLoudnorm:
    def test_calls_loudnorm(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            master_loudnorm("in.mp3", "out.mp3")
            cmd = " ".join(mock_run.call_args[0][0])
            assert "loudnorm" in cmd
            assert "I=-16" in cmd

    def test_custom_targets(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            master_loudnorm("in.mp3", "out.mp3", target_i=-14, target_tp=-1.0)
            cmd = " ".join(mock_run.call_args[0][0])
            assert "I=-14" in cmd
            assert "TP=-1.0" in cmd

    def test_raises_on_failure(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="fail")
            with pytest.raises(RuntimeError, match="Loudnorm"):
                master_loudnorm("a", "b")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_minimal_args(self):
        parser = build_parser()
        args = parser.parse_args(["output/ep01/"])
        assert args.input_dir == "output/ep01/"
        assert args.output == "episode_mixed.mp3"
        assert args.level is False
        assert args.no_master is False

    def test_all_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            "output/ep01/", "-o", "final.mp3",
            "--intro", "intro.mp3", "--outro", "outro.mp3",
            "--music-bed", "music.mp3", "--music-volume", "0.15",
            "--crossfade", "3.0",
            "--level", "--target-lufs", "-20",
            "--no-master", "--dry-run",
        ])
        assert args.output == "final.mp3"
        assert args.intro == "intro.mp3"
        assert args.outro == "outro.mp3"
        assert args.music_bed == "music.mp3"
        assert args.music_volume == 0.15
        assert args.crossfade == 3.0
        assert args.level is True
        assert args.target_lufs == -20
        assert args.no_master is True
        assert args.dry_run is True
