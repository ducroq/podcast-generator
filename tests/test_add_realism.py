"""Tests for generator/add_realism.py — realism post-processing."""

import random

from add_realism import (
    split_into_turns, plan_realism, build_filter_complex, _silence_pad, parse_range,
)


class TestSplitIntoTurns:
    def test_basic_split(self):
        silences = [
            {"start": 1.0, "end": 1.5, "duration": 0.5},
            {"start": 3.0, "end": 3.8, "duration": 0.8},
        ]
        turns = split_into_turns(silences, total_duration=5.0)
        assert len(turns) == 3
        # First turn: 0.0 - 1.0
        assert turns[0]["start"] == 0.0
        assert turns[0]["end"] == 1.0
        assert turns[0]["gap_after"] == 0.5
        # Second turn: 1.5 - 3.0
        assert turns[1]["start"] == 1.5
        assert turns[1]["end"] == 3.0
        # Third turn: 3.8 - 5.0
        assert turns[2]["start"] == 3.8
        assert turns[2]["end"] == 5.0
        assert turns[2]["gap_after"] == 0.0

    def test_no_silences(self):
        turns = split_into_turns([], total_duration=5.0)
        assert len(turns) == 1
        assert turns[0]["start"] == 0.0
        assert turns[0]["end"] == 5.0

    def test_leading_silence(self):
        """Audio starts with silence — should not lose the first speech turn."""
        silences = [{"start": 0.0, "end": 0.5, "duration": 0.5}]
        turns = split_into_turns(silences, total_duration=3.0)
        assert len(turns) >= 1
        # The speech after the silence should be captured
        assert any(t["start"] == 0.5 for t in turns)

    def test_very_short_gap_skipped(self):
        """Gaps shorter than 0.01s should not produce a turn."""
        silences = [{"start": 1.0, "end": 1.005, "duration": 0.005}]
        turns = split_into_turns(silences, total_duration=3.0)
        # Should be 1 or 2 turns, not 3 (the tiny gap doesn't split)
        assert len(turns) <= 2


class TestPlanRealism:
    def test_deterministic_with_seed(self):
        turns = [
            {"start": 0, "end": 1, "duration": 1, "gap_after": 0.5},
            {"start": 1.5, "end": 2.5, "duration": 1, "gap_after": 0.5},
            {"start": 3.0, "end": 4.0, "duration": 1, "gap_after": 0},
        ]
        random.seed(42)
        actions1 = plan_realism(turns, 0.5, (300, 800), (50, 150), 0, [], False)
        random.seed(42)
        actions2 = plan_realism(turns, 0.5, (300, 800), (50, 150), 0, [], False)
        assert actions1 == actions2

    def test_last_turn_never_gets_overlap(self):
        turns = [
            {"start": 0, "end": 1, "duration": 1, "gap_after": 0.5},
            {"start": 1.5, "end": 2.5, "duration": 1, "gap_after": 0},
        ]
        random.seed(42)
        actions = plan_realism(turns, 1.0, (100, 200), (50, 150), 0, [], False)
        assert actions[-1]["action"] == "normal"
        assert actions[-1]["overlap_ms"] == 0

    def test_zero_overlap_chance(self):
        turns = [
            {"start": 0, "end": 1, "duration": 1, "gap_after": 1.0},
            {"start": 2, "end": 3, "duration": 1, "gap_after": 0},
        ]
        actions = plan_realism(turns, 0.0, (300, 800), (50, 150), 0, [], False)
        assert all(a["action"] != "overlap" for a in actions)

    def test_filler_only_on_long_turns(self):
        turns = [
            {"start": 0, "end": 1, "duration": 1.0, "gap_after": 0.5},  # short
            {"start": 1.5, "end": 5.5, "duration": 4.0, "gap_after": 0.5},  # long
            {"start": 6.0, "end": 7.0, "duration": 1.0, "gap_after": 0},
        ]
        random.seed(0)
        actions = plan_realism(turns, 0, (300, 800), (50, 150), 1.0, ["filler.wav"], False)
        # Only the long turn (index 1) should potentially get a filler
        assert actions[0]["filler_file"] is None


class TestSilencePad:
    def test_produces_two_filters(self):
        filters = _silence_pad(44100, 0.5, "pad0")
        assert len(filters) == 2
        assert "anullsrc" in filters[0]
        assert "atrim" in filters[1]
        assert "[pad0]" in filters[1]

    def test_labels_are_unique(self):
        f1 = _silence_pad(44100, 0.5, "pad0")
        f2 = _silence_pad(44100, 0.3, "pad1")
        # No label collision
        assert "pad0" not in str(f2)
        assert "pad1" not in str(f1)


class TestBuildFilterComplex:
    def test_input_forced_to_mono(self):
        turns = [
            {"start": 0, "end": 1, "duration": 1, "gap_after": 0},
        ]
        actions = [
            {"turn_idx": 0, "action": "normal", "overlap_ms": 0, "jitter_ms": 0, "filler_file": None},
        ]
        filters, _, _ = build_filter_complex(
            turns, actions, 1.0, 44100, no_room_tone=True,
        )
        combined = ";".join(filters)
        # Input should be forced to mono before any processing
        assert "[0:a]aformat=channel_layouts=mono[inmono]" in combined
        # Turn extraction should reference [inmono], not [0:a]
        assert "[inmono]atrim=" in combined

    def test_basic_structure(self):
        turns = [
            {"start": 0, "end": 1, "duration": 1, "gap_after": 0.5},
            {"start": 1.5, "end": 2.5, "duration": 1, "gap_after": 0},
        ]
        actions = [
            {"turn_idx": 0, "action": "normal", "overlap_ms": 0, "jitter_ms": 0, "filler_file": None},
            {"turn_idx": 1, "action": "normal", "overlap_ms": 0, "jitter_ms": 0, "filler_file": None},
        ]
        filters, out_label, extra = build_filter_complex(
            turns, actions, 2.5, 44100, no_room_tone=True,
        )
        assert isinstance(filters, list)
        assert out_label == "joined"
        assert extra == []
        # Should have: 2 atrim + 2 anullsrc (for pad) + 1 concat = at least 5 filters
        combined = ";".join(filters)
        assert "concat" in combined
        assert "atrim" in combined

    def test_no_room_tone(self):
        turns = [{"start": 0, "end": 1, "duration": 1, "gap_after": 0}]
        actions = [{"turn_idx": 0, "action": "normal", "overlap_ms": 0, "jitter_ms": 0, "filler_file": None}]
        filters, out_label, _ = build_filter_complex(
            turns, actions, 1.0, 44100, no_room_tone=True,
        )
        combined = ";".join(filters)
        assert "anoisesrc" not in combined
        assert "roomnoise" not in combined
        assert out_label == "joined"

    def test_synthetic_room_tone(self):
        turns = [{"start": 0, "end": 1, "duration": 1, "gap_after": 0}]
        actions = [{"turn_idx": 0, "action": "normal", "overlap_ms": 0, "jitter_ms": 0, "filler_file": None}]
        filters, out_label, _ = build_filter_complex(
            turns, actions, 1.0, 44100, no_room_tone=False,
        )
        combined = ";".join(filters)
        assert "anoisesrc" in combined
        assert out_label == "roomed"

    def test_amix_uses_pipe_separator(self):
        """Regression: amix weights must use pipe, not space."""
        turns = [{"start": 0, "end": 1, "duration": 1, "gap_after": 0}]
        actions = [{"turn_idx": 0, "action": "normal", "overlap_ms": 0, "jitter_ms": 0, "filler_file": None}]
        filters, _, _ = build_filter_complex(
            turns, actions, 1.0, 44100, no_room_tone=False,
        )
        combined = ";".join(filters)
        # amix weights should use pipe separator
        assert "weights=1|" in combined

    def test_overlap_action_produces_silence_pad(self):
        turns = [
            {"start": 0, "end": 1, "duration": 1, "gap_after": 0.5},
            {"start": 1.5, "end": 2.5, "duration": 1, "gap_after": 0},
        ]
        actions = [
            {"turn_idx": 0, "action": "overlap", "overlap_ms": 300, "jitter_ms": 0, "filler_file": None},
            {"turn_idx": 1, "action": "normal", "overlap_ms": 0, "jitter_ms": 0, "filler_file": None},
        ]
        filters, _, _ = build_filter_complex(
            turns, actions, 2.5, 44100, no_room_tone=True,
        )
        combined = ";".join(filters)
        # Should have anullsrc for the pad (properly separated, not comma-chained)
        assert "anullsrc" in combined
        assert "[null_pad0]" in combined  # intermediate label from _silence_pad


    def test_filler_action_produces_input_and_mix(self):
        random.seed(42)
        turns = [
            {"start": 0, "end": 4, "duration": 4, "gap_after": 0.5},
            {"start": 4.5, "end": 5.5, "duration": 1, "gap_after": 0},
        ]
        actions = [
            {"turn_idx": 0, "action": "normal", "overlap_ms": 0, "jitter_ms": 0, "filler_file": "/tmp/uh.wav"},
            {"turn_idx": 1, "action": "normal", "overlap_ms": 0, "jitter_ms": 0, "filler_file": None},
        ]
        filters, out_label, extra_inputs = build_filter_complex(
            turns, actions, 5.5, 44100, no_room_tone=True,
        )
        combined = ";".join(filters)
        # Filler should be added as extra input
        assert "-i" in extra_inputs
        assert "/tmp/uh.wav" in extra_inputs
        # Filter graph should have adelay and volume for the filler
        assert "adelay=" in combined
        assert "volume=0.3" in combined
        # Should mix filler into output
        assert "amix=inputs=2:duration=first" in combined
        # Output label should be the filler mix, not 'joined'
        assert out_label.startswith("fmix")

    def test_filler_with_room_tone_correct_input_indices(self):
        random.seed(42)
        turns = [
            {"start": 0, "end": 4, "duration": 4, "gap_after": 0.5},
            {"start": 4.5, "end": 5.5, "duration": 1, "gap_after": 0},
        ]
        actions = [
            {"turn_idx": 0, "action": "normal", "overlap_ms": 0, "jitter_ms": 0, "filler_file": "/tmp/uh.wav"},
            {"turn_idx": 1, "action": "normal", "overlap_ms": 0, "jitter_ms": 0, "filler_file": None},
        ]
        filters, out_label, extra_inputs = build_filter_complex(
            turns, actions, 5.5, 44100,
            room_tone_path="/tmp/tone.wav", no_room_tone=False,
        )
        combined = ";".join(filters)
        # Room tone is input [1], filler is input [2]
        assert "[1:a]" in combined  # room tone
        assert "[2:a]" in combined  # filler
        # Both room tone and filler file should be in extra_inputs
        assert extra_inputs.count("-i") == 2
        assert out_label == "roomed"

    def test_no_filler_when_no_filler_file(self):
        turns = [
            {"start": 0, "end": 4, "duration": 4, "gap_after": 0},
        ]
        actions = [
            {"turn_idx": 0, "action": "normal", "overlap_ms": 0, "jitter_ms": 0, "filler_file": None},
        ]
        filters, out_label, extra_inputs = build_filter_complex(
            turns, actions, 4.0, 44100, no_room_tone=True,
        )
        combined = ";".join(filters)
        assert "adelay" not in combined
        assert "filler" not in combined
        assert extra_inputs == []


class TestParseRange:
    def test_basic_range(self):
        assert parse_range("300-800") == (300, 800)

    def test_single_value(self):
        assert parse_range("500") == (500, 500)

    def test_negative_rejected(self):
        import argparse
        import pytest
        with pytest.raises(argparse.ArgumentTypeError):
            parse_range("-5")


class TestEndToEndFilterGraph:
    """Test that the generated filter graph is valid ffmpeg syntax."""

    def test_filter_graph_runs_on_real_audio(self, tmp_audio):
        """Build a filter graph from real silence detection and verify ffmpeg accepts it."""
        import subprocess
        from audio_utils import detect_silences, get_duration, get_sample_rate

        duration = get_duration(tmp_audio)
        sample_rate = get_sample_rate(tmp_audio)
        silences = detect_silences(tmp_audio, noise_db=-30, min_duration=0.2)

        if len(silences) < 2:
            import pytest
            pytest.skip("Not enough silences detected in test fixture")

        turns = split_into_turns(silences, duration)

        random.seed(42)
        actions = plan_realism(
            turns, overlap_chance=0.5, overlap_range_ms=(100, 300),
            jitter_range_ms=(20, 50), filler_chance=0, fillers_available=[],
        )

        filters, out_label, extra_inputs = build_filter_complex(
            turns, actions, duration, sample_rate, no_room_tone=True,
        )

        filter_complex = ";".join(filters)
        output_path = tmp_audio.parent / "realism_output.wav"

        cmd = [
            "ffmpeg", "-y", "-i", str(tmp_audio),
            *extra_inputs,
            "-filter_complex", filter_complex,
            "-map", f"[{out_label}]",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"ffmpeg failed:\n{result.stderr[-500:]}"
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_stereo_input_handled_gracefully(self, tmp_audio_stereo):
        """Stereo input (e.g. from Chatterbox) should be downmixed to mono without error."""
        import subprocess
        from audio_utils import detect_silences, get_duration, get_sample_rate

        duration = get_duration(tmp_audio_stereo)
        sample_rate = get_sample_rate(tmp_audio_stereo)
        silences = detect_silences(tmp_audio_stereo, noise_db=-30, min_duration=0.2)

        if len(silences) < 2:
            import pytest
            pytest.skip("Not enough silences detected in test fixture")

        turns = split_into_turns(silences, duration)

        random.seed(42)
        actions = plan_realism(
            turns, overlap_chance=0.5, overlap_range_ms=(100, 300),
            jitter_range_ms=(20, 50), filler_chance=0, fillers_available=[],
        )

        filters, out_label, extra_inputs = build_filter_complex(
            turns, actions, duration, sample_rate, no_room_tone=False,
        )

        filter_complex = ";".join(filters)
        output_path = tmp_audio_stereo.parent / "stereo_realism_output.wav"

        cmd = [
            "ffmpeg", "-y", "-i", str(tmp_audio_stereo),
            *extra_inputs,
            "-filter_complex", filter_complex,
            "-map", f"[{out_label}]",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, f"ffmpeg failed on stereo input:\n{result.stderr[-500:]}"
        assert output_path.exists()
        assert output_path.stat().st_size > 0
