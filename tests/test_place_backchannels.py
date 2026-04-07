"""Tests for generator/place_backchannels.py — backchannel placement rules."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generator"))
from place_backchannels import (
    build_parser,
    place_backchannels,
    plan_backchannel_placement,
)

SR = 24000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def speakers():
    return ["alex", "morgan", "zara"]


@pytest.fixture
def line_positions():
    """Simulate a conversation with varied turn lengths and speaker changes."""
    rng = np.random.default_rng(42)
    positions = []
    pos = 0
    speakers = ["alex", "morgan", "zara"]

    for i in range(30):
        speaker = speakers[i % 3]
        # Every 5th line is a long turn (>6s)
        duration = 8.0 if i % 5 == 0 else 2.0 + rng.random() * 2
        positions.append({
            "pos_samples": pos,
            "speaker": speaker,
            "duration": duration,
        })
        pos += int(duration * SR) + int(0.15 * SR)

    return positions


@pytest.fixture
def bc_clips():
    """Synthetic backchannel clips for each speaker."""
    clips = {}
    for speaker in ["alex", "morgan", "zara"]:
        clips[speaker] = []
        for i in range(3):
            t = np.linspace(0, 0.3, int(SR * 0.3), dtype=np.float32)
            clip = 0.2 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
            clips[speaker].append(clip)
    return clips


# ---------------------------------------------------------------------------
# plan_backchannel_placement
# ---------------------------------------------------------------------------

class TestPlanPlacement:
    def test_returns_list(self, line_positions, speakers):
        placements = plan_backchannel_placement(line_positions, speakers)
        assert isinstance(placements, list)

    def test_respects_max_count(self, line_positions, speakers):
        placements = plan_backchannel_placement(
            line_positions, speakers, max_count=3,
        )
        assert len(placements) <= 3

    def test_respects_min_gap(self, line_positions, speakers):
        placements = plan_backchannel_placement(
            line_positions, speakers, min_gap=5,
        )
        for i in range(1, len(placements)):
            gap = placements[i]["after_line_idx"] - placements[i - 1]["after_line_idx"]
            assert gap >= 5

    def test_only_after_long_turns(self, line_positions, speakers):
        placements = plan_backchannel_placement(
            line_positions, speakers, min_turn_duration=6.0,
        )
        for p in placements:
            idx = p["after_line_idx"]
            assert line_positions[idx]["duration"] >= 6.0

    def test_third_person_selection(self, line_positions, speakers):
        placements = plan_backchannel_placement(line_positions, speakers)
        for p in placements:
            idx = p["after_line_idx"]
            prev_speaker = line_positions[idx]["speaker"]
            next_speaker = line_positions[idx + 1]["speaker"]
            reactor = p["reactor_speaker"]
            assert reactor != prev_speaker
            assert reactor != next_speaker

    def test_no_placements_for_short_turns(self, speakers):
        # All turns are short (< 6s)
        positions = [
            {"pos_samples": i * SR * 3, "speaker": speakers[i % 3], "duration": 2.0}
            for i in range(20)
        ]
        placements = plan_backchannel_placement(positions, speakers)
        assert len(placements) == 0

    def test_requires_speaker_change(self, speakers):
        # Same speaker throughout
        positions = [
            {"pos_samples": i * SR * 8, "speaker": "alex", "duration": 7.0}
            for i in range(10)
        ]
        placements = plan_backchannel_placement(positions, speakers)
        assert len(placements) == 0

    def test_deterministic_with_seed(self, line_positions, speakers):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        p1 = plan_backchannel_placement(line_positions, speakers, rng=rng1)
        p2 = plan_backchannel_placement(line_positions, speakers, rng=rng2)
        assert p1 == p2

    def test_empty_positions(self, speakers):
        placements = plan_backchannel_placement([], speakers)
        assert placements == []

    def test_two_speakers_only(self):
        positions = [
            {"pos_samples": 0, "speaker": "alex", "duration": 8.0},
            {"pos_samples": SR * 9, "speaker": "morgan", "duration": 2.0},
        ]
        # With only 2 speakers, no third person exists
        # but should still work if there are >2 total speakers
        placements = plan_backchannel_placement(
            positions, ["alex", "morgan", "zara"],
        )
        if placements:
            assert placements[0]["reactor_speaker"] == "zara"


# ---------------------------------------------------------------------------
# place_backchannels
# ---------------------------------------------------------------------------

class TestPlaceBackchannels:
    def test_places_clips(self, bc_clips):
        audio = np.zeros(SR * 60, dtype=np.float32)
        placements = [
            {"position_samples": SR * 10, "reactor_speaker": "morgan",
             "after_line_idx": 0},
            {"position_samples": SR * 30, "reactor_speaker": "zara",
             "after_line_idx": 5},
        ]
        result, count = place_backchannels(audio, placements, bc_clips)
        assert count == 2
        # Audio should be non-zero at placement positions
        assert np.max(np.abs(result[SR * 10:SR * 10 + 1000])) > 0

    def test_returns_copy(self, bc_clips):
        audio = np.zeros(SR * 10, dtype=np.float32)
        placements = [
            {"position_samples": SR * 5, "reactor_speaker": "alex",
             "after_line_idx": 0},
        ]
        result, _ = place_backchannels(audio, placements, bc_clips)
        # Original should be unchanged
        assert np.max(np.abs(audio)) == 0.0

    def test_skips_missing_speaker(self, bc_clips):
        audio = np.zeros(SR * 10, dtype=np.float32)
        placements = [
            {"position_samples": SR * 5, "reactor_speaker": "unknown",
             "after_line_idx": 0},
        ]
        result, count = place_backchannels(audio, placements, bc_clips)
        assert count == 0

    def test_skips_out_of_bounds(self, bc_clips):
        audio = np.zeros(SR * 1, dtype=np.float32)  # only 1 second
        placements = [
            {"position_samples": SR * 5, "reactor_speaker": "alex",
             "after_line_idx": 0},  # way past end
        ]
        result, count = place_backchannels(audio, placements, bc_clips)
        assert count == 0

    def test_volume_applied(self, bc_clips):
        audio = np.zeros(SR * 10, dtype=np.float32)
        placements = [
            {"position_samples": SR * 2, "reactor_speaker": "alex",
             "after_line_idx": 0},
        ]
        result_loud, _ = place_backchannels(audio, placements, bc_clips,
                                             volume_db=0.0)
        result_quiet, _ = place_backchannels(audio, placements, bc_clips,
                                              volume_db=-12.0)
        loud_peak = np.max(np.abs(result_loud[SR * 2:SR * 3]))
        quiet_peak = np.max(np.abs(result_quiet[SR * 2:SR * 3]))
        assert loud_peak > quiet_peak

    def test_empty_placements(self, bc_clips):
        audio = np.zeros(SR * 10, dtype=np.float32)
        result, count = place_backchannels(audio, [], bc_clips)
        assert count == 0
        np.testing.assert_array_equal(result, audio)


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

class TestCLI:
    def test_required_args(self):
        args = build_parser().parse_args([
            "mixed.wav", "--manifest", "m.json",
            "--backchannels", "bc/",
        ])
        assert args.input == "mixed.wav"
        assert args.manifest == "m.json"
        assert args.backchannels == "bc/"

    def test_all_flags(self):
        args = build_parser().parse_args([
            "mixed.wav", "--manifest", "m.json",
            "--backchannels", "bc/",
            "-o", "out.wav",
            "--max-count", "8", "--min-gap", "3",
            "--min-turn", "4.0", "--volume", "-6.0",
            "--seed", "123", "--dry-run",
        ])
        assert args.max_count == 8
        assert args.min_gap == 3
        assert args.min_turn == 4.0
        assert args.volume == -6.0
        assert args.seed == 123
        assert args.dry_run is True
