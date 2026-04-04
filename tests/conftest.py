"""Shared fixtures for podcast-generator tests."""

import subprocess
import sys
from pathlib import Path

import pytest

# Add generator/ to path so imports work
GENERATOR_DIR = Path(__file__).parent.parent / "generator"
sys.path.insert(0, str(GENERATOR_DIR))
sys.path.insert(0, str(GENERATOR_DIR / "elevenlabs"))


@pytest.fixture
def tmp_audio(tmp_path):
    """Generate a short test audio file: 3 speech-like tones with silences between them.

    Structure (5s total):
      0.0-0.8s  tone 440Hz  (simulated speech turn 1)
      0.8-1.5s  silence     (0.7s gap)
      1.5-2.5s  tone 330Hz  (simulated speech turn 2)
      2.5-3.5s  silence     (1.0s gap)
      3.5-4.5s  tone 440Hz  (simulated speech turn 3)
      4.5-5.0s  silence     (0.5s trailing)
    """
    path = tmp_path / "test_audio.wav"
    # Generate with ffmpeg: 3 tones separated by silences
    filter_str = (
        "aevalsrc=sin(440*2*PI*t):s=44100:d=0.8[t1];"
        "anullsrc=r=44100:cl=mono[n1];[n1]atrim=duration=0.7[s1];"
        "aevalsrc=sin(330*2*PI*t):s=44100:d=1.0[t2];"
        "anullsrc=r=44100:cl=mono[n2];[n2]atrim=duration=1.0[s2];"
        "aevalsrc=sin(440*2*PI*t):s=44100:d=1.0[t3];"
        "anullsrc=r=44100:cl=mono[n3];[n3]atrim=duration=0.5[s3];"
        "[t1][s1][t2][s2][t3][s3]concat=n=6:v=0:a=1[out]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-filter_complex", filter_str,
        "-map", "[out]",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"ffmpeg failed: {result.stderr[-300:]}"
    return path


@pytest.fixture
def tmp_audio_stereo(tmp_path):
    """Generate a short stereo test audio file (same structure as tmp_audio)."""
    path = tmp_path / "test_audio_stereo.wav"
    filter_str = (
        "aevalsrc=sin(440*2*PI*t):s=44100:d=0.8[t1];"
        "anullsrc=r=44100:cl=mono[n1];[n1]atrim=duration=0.7[s1];"
        "aevalsrc=sin(330*2*PI*t):s=44100:d=1.0[t2];"
        "anullsrc=r=44100:cl=mono[n2];[n2]atrim=duration=1.0[s2];"
        "aevalsrc=sin(440*2*PI*t):s=44100:d=1.0[t3];"
        "anullsrc=r=44100:cl=mono[n3];[n3]atrim=duration=0.5[s3];"
        "[t1][s1][t2][s2][t3][s3]concat=n=6:v=0:a=1,"
        "aformat=channel_layouts=stereo[out]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-filter_complex", filter_str,
        "-map", "[out]",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"ffmpeg failed: {result.stderr[-300:]}"
    return path


@pytest.fixture
def tmp_audio_mp3(tmp_audio, tmp_path):
    """Convert the test WAV to MP3."""
    mp3_path = tmp_path / "test_audio.mp3"
    cmd = ["ffmpeg", "-y", "-i", str(tmp_audio), "-codec:a", "libmp3lame", "-b:a", "128k", str(mp3_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    return mp3_path
