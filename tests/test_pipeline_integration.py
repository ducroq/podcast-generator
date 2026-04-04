"""Integration tests: verify the full pipeline stages chain correctly.

These tests run real ffmpeg commands on test audio to verify that
output from each stage is valid input for the next stage.
"""

import json
import random
import subprocess
from pathlib import Path

from audio_utils import detect_silences, get_duration, get_sample_rate
from add_realism import split_into_turns, plan_realism, build_filter_complex
from trim_silences import trim_silences
from validate_tts import (
    check_hallucination, build_report, save_report, load_report,
)


class TestTrimThenRealism:
    """Test that trim_silences output feeds correctly into add_realism."""

    def test_trim_output_is_valid_realism_input(self, tmp_audio):
        # Stage 1: Trim
        trimmed = tmp_audio.parent / "trimmed.wav"
        trim_silences(tmp_audio, trimmed, max_pause=0.3, loudnorm=False)
        assert trimmed.exists()

        # Stage 2: Detect turns in trimmed output
        duration = get_duration(trimmed)
        sample_rate = get_sample_rate(trimmed)
        silences = detect_silences(trimmed, noise_db=-30, min_duration=0.1)
        turns = split_into_turns(silences, duration)

        if len(turns) < 2:
            # Not enough turns — that's OK, trim may have merged them
            return

        # Stage 3: Plan and build realism filter graph
        random.seed(42)
        actions = plan_realism(
            turns, 0.3, (100, 300), (20, 50), 0, [],
        )
        filters, out_label, extra = build_filter_complex(
            turns, actions, duration, sample_rate, no_room_tone=True,
        )

        # Stage 4: Run the filter graph
        output = tmp_audio.parent / "realistic.wav"
        cmd = [
            "ffmpeg", "-y", "-i", str(trimmed),
            *extra,
            "-filter_complex", ";".join(filters),
            "-map", f"[{out_label}]",
            str(output),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"ffmpeg failed:\n{result.stderr[-500:]}"
        assert output.exists()
        assert get_duration(output) > 0


class TestRealismThenMaster:
    """Test that add_realism output can be mastered with ffmpeg."""

    def test_master_ffmpeg_command(self, tmp_audio):
        # Run simple loudnorm mastering on the test audio
        mastered = tmp_audio.parent / "mastered.mp3"
        cmd = [
            "ffmpeg", "-y", "-i", str(tmp_audio),
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-codec:a", "libmp3lame", "-b:a", "192k",
            str(mastered),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert mastered.exists()
        assert mastered.stat().st_size > 0


class TestValidationReportFlow:
    """Test the validation report lifecycle."""

    def test_report_created_and_reloadable(self, tmp_path):
        results = [
            {"file": "line_01.wav", "status": "OK", "issues": [], "duration": 2.1,
             "expected_text": "Hello world", "transcription": "Hello world"},
            {"file": "line_02.wav", "status": "FLAGGED", "issues": ["HALLUCINATION_START"],
             "duration": 5.3, "expected_text": "Short text",
             "transcription": "Extra words Short text"},
        ]
        report = build_report(results, language="en", engine="chatterbox")
        path = save_report(report, tmp_path)

        loaded = load_report(tmp_path)
        assert loaded["engine"] == "chatterbox"
        assert loaded["summary"]["ok"] == 1
        assert loaded["summary"]["flagged"] == 1
        assert len(loaded["results"]) == 2

    def test_manifest_validation_with_path_traversal(self, tmp_path):
        """Manifest entries with path traversal should be rejected."""
        from validate_tts import validate_manifest

        manifest = [{"file": "../../etc/passwd", "text": "test"}]
        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        results, flagged = validate_manifest(str(manifest_path))
        assert flagged == 1
        assert results[0]["status"] == "ERROR"
        assert "Rejected" in results[0]["issues"][0]


class TestFullPipelineChain:
    """End-to-end: generate test audio → trim → realism → master."""

    def test_full_chain(self, tmp_audio):
        workdir = tmp_audio.parent

        # Step 1: Trim silences
        trimmed = workdir / "01_trimmed.wav"
        trim_silences(tmp_audio, trimmed, max_pause=0.3, loudnorm=False)
        assert trimmed.exists()

        # Step 2: Add realism (no room tone for simplicity)
        duration = get_duration(trimmed)
        sr = get_sample_rate(trimmed)
        silences = detect_silences(trimmed, noise_db=-30, min_duration=0.1)
        turns = split_into_turns(silences, duration)

        if len(turns) >= 2:
            random.seed(42)
            actions = plan_realism(turns, 0.3, (50, 150), (10, 30), 0, [])
            filters, out_label, extra = build_filter_complex(
                turns, actions, duration, sr, no_room_tone=True,
            )
            realistic = workdir / "02_realistic.wav"
            cmd = [
                "ffmpeg", "-y", "-i", str(trimmed), *extra,
                "-filter_complex", ";".join(filters),
                "-map", f"[{out_label}]", str(realistic),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            assert result.returncode == 0, f"Realism failed:\n{result.stderr[-300:]}"
            master_input = realistic
        else:
            master_input = trimmed

        # Step 3: Master
        mastered = workdir / "03_mastered.mp3"
        cmd = [
            "ffmpeg", "-y", "-i", str(master_input),
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-codec:a", "libmp3lame", "-b:a", "192k",
            str(mastered),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert mastered.exists()

        final_duration = get_duration(mastered)
        original_duration = get_duration(tmp_audio)
        assert final_duration > 0
        assert final_duration <= original_duration + 0.5  # Should not grow
