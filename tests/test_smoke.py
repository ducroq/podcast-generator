"""Smoke tests: lightweight integration tests for every CLI tool.

Exercises each tool's real entry point with minimal synthetic input.
Catches import errors, argparse misconfigurations, broken exit codes,
and integration issues that unit tests miss.

No GPU or API keys required. Runs in <30s.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

PROJECT_ROOT = Path(__file__).parent.parent
GENERATOR_DIR = PROJECT_ROOT / "generator"

# All CLI tools with a --help flag
ALL_CLI_TOOLS = [
    "generator/write_script.py",
    "generator/mix_episode.py",
    "generator/clean_audio.py",
    "generator/validate_tts.py",
    "generator/add_realism.py",
    "generator/master.py",
    "generator/publish.py",
    "generator/trim_silences.py",
    "generator/tts_overrides.py",
    "generator/mix_preprocess.py",
    "generator/assemble_intro.py",
    "generator/generate_backchannels.py",
    "generator/place_backchannels.py",
    "generator/analyze_voice.py",
    "generator/export_stems.py",
]


def run_cli(script, *args, timeout=15):
    """Run a generator script via subprocess, return CompletedProcess."""
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / script), *args],
        capture_output=True, text=True, timeout=timeout,
        cwd=str(PROJECT_ROOT),
    )


def make_wav(path, duration=2.0, freq=440, sr=24000):
    """Create a mono WAV sine wave using numpy."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * freq * t)
    sf.write(str(path), audio, sr)
    return path


def make_mp3(wav_path, mp3_path):
    """Convert WAV to MP3 via ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-codec:a", "libmp3lame",
         "-b:a", "128k", str(mp3_path)],
        capture_output=True, check=True,
    )
    return mp3_path


# ---------------------------------------------------------------------------
# Session-scoped fixtures (created once, shared across all tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def smoke_wav(tmp_path_factory):
    """A 2-second mono WAV sine wave."""
    path = tmp_path_factory.mktemp("smoke") / "tone.wav"
    return make_wav(path)


@pytest.fixture(scope="session")
def smoke_mp3(smoke_wav, tmp_path_factory):
    """The smoke WAV converted to MP3."""
    path = tmp_path_factory.mktemp("smoke_mp3") / "tone.mp3"
    return make_mp3(smoke_wav, path)


@pytest.fixture(scope="session")
def smoke_wav_with_silence(tmp_path_factory):
    """A WAV with speech-silence-speech pattern for trim testing."""
    path = tmp_path_factory.mktemp("smoke_sil") / "speech_silence.wav"
    sr = 24000
    tone = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr, dtype=np.float32))
    silence = np.zeros(sr, dtype=np.float32)
    audio = np.concatenate([tone, silence, tone])
    sf.write(str(path), audio, sr)
    return path


# ---------------------------------------------------------------------------
# Function-scoped fixtures (fresh copy per test)
# ---------------------------------------------------------------------------

@pytest.fixture
def work_wav(smoke_wav, tmp_path):
    """Copy of smoke_wav for tests that modify in-place."""
    dest = tmp_path / "work.wav"
    shutil.copy2(str(smoke_wav), str(dest))
    return dest


@pytest.fixture
def work_mp3(smoke_mp3, tmp_path):
    """Copy of smoke_mp3 for tests that modify in-place."""
    dest = tmp_path / "work.mp3"
    shutil.copy2(str(smoke_mp3), str(dest))
    return dest


@pytest.fixture
def section_dir(tmp_path):
    """Directory with numbered section MP3 files for mix_episode/publish."""
    d = tmp_path / "sections"
    d.mkdir()
    for i in range(1, 4):
        wav = d / f"section_{i:02d}.wav"
        make_wav(wav, duration=1.0, freq=300 + i * 100)
        mp3 = d / f"section_{i:02d}.mp3"
        make_mp3(wav, mp3)
        wav.unlink()
    return d


@pytest.fixture
def wav_dir(tmp_path):
    """Directory with numbered speaker WAV files for preprocess/stems."""
    d = tmp_path / "wavs"
    d.mkdir()
    make_wav(d / "000_alex.wav", duration=0.5, freq=200)
    make_wav(d / "001_morgan.wav", duration=0.5, freq=350)
    make_wav(d / "002_alex.wav", duration=0.5, freq=200)
    return d


@pytest.fixture
def script_file(tmp_path):
    """Minimal dialogue script."""
    path = tmp_path / "script.txt"
    path.write_text(
        "==================================================\n"
        "OPENING\n"
        "==================================================\n"
        "\n"
        "Alex: [warm] Welcome to the show.\n"
        "Morgan: [excited] Thanks for having me.\n"
        "Alex: [curious] So tell me about the project.\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def manifest_file(tmp_path, wav_dir):
    """JSON manifest matching wav_dir files."""
    manifest = [
        {"file": "000_alex.wav", "speaker": "alex",
         "text": "Welcome to the show.", "duration": 0.5},
        {"file": "001_morgan.wav", "speaker": "morgan",
         "text": "Thanks for having me.", "duration": 0.5},
        {"file": "002_alex.wav", "speaker": "alex",
         "text": "So tell me about the project.", "duration": 0.5},
    ]
    path = wav_dir / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


@pytest.fixture
def overrides_file(tmp_path):
    """JSON overrides file."""
    path = tmp_path / "overrides.json"
    data = {
        "overrides": {
            "000": [{"text": "Welcome to the show.", "pause_after": 0.2}]
        }
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def bc_dir(tmp_path):
    """Backchannel clip directory with a manifest."""
    d = tmp_path / "bc_clips"
    d.mkdir()
    # Create a short backchannel clip
    make_wav(d / "bc_morgan_00.wav", duration=0.3, freq=500)
    manifest = [
        {"file": "bc_morgan_00.wav", "speaker": "morgan", "type": "agreement"}
    ]
    (d / "bc_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d


# ===========================================================================
# Test classes
# ===========================================================================

class TestHelpFlags:
    """Every CLI tool should exit 0 on --help."""

    @pytest.mark.parametrize("script", ALL_CLI_TOOLS)
    def test_help_exits_zero(self, script):
        result = run_cli(script, "--help")
        assert result.returncode == 0, f"{script} --help failed:\n{result.stderr[:300]}"
        stdout = result.stdout.lower()
        assert "usage" in stdout or "optional" in stdout


class TestDryRun:
    """Tools with --dry-run should exit 0 and not create output files."""

    def test_clean_audio_dry_run(self, work_wav):
        from clean_audio import main
        mtime_before = work_wav.stat().st_mtime
        main([str(work_wav), "--dry-run"])
        assert work_wav.stat().st_mtime == mtime_before, "dry-run should not modify file"

    def test_mix_episode_dry_run(self, section_dir, tmp_path):
        from mix_episode import main
        output = tmp_path / "should_not_exist.mp3"
        main([str(section_dir), "-o", str(output), "--dry-run"])
        assert not output.exists(), "dry-run should not create output"

    def test_add_realism_dry_run(self, tmp_path, tmp_audio_mp3):
        output = tmp_path / "should_not_exist.mp3"
        result = run_cli("generator/add_realism.py", str(tmp_audio_mp3),
                         str(output), "--dry-run", "--seed", "42")
        assert result.returncode == 0, f"stdout: {result.stdout[:300]}\nstderr: {result.stderr[:300]}"
        assert not output.exists(), "dry-run should not create output"

    def test_mix_preprocess_dry_run(self, wav_dir, manifest_file):
        from mix_preprocess import main
        mtimes = {f.name: f.stat().st_mtime for f in wav_dir.glob("*.wav")}
        main([str(wav_dir), "--manifest", str(manifest_file), "--dry-run"])
        for f in wav_dir.glob("*.wav"):
            assert f.stat().st_mtime == mtimes[f.name], \
                f"dry-run should not modify {f.name}"

    def test_publish_dry_run(self, section_dir, script_file):
        from publish import main
        main([str(section_dir), "--script", str(script_file), "--dry-run"])
        assert not (section_dir / "published").exists(), \
            "dry-run should not create published directory"


class TestHappyPath:
    """Each tool's happy path with minimal synthetic input."""

    def test_clean_audio(self, work_wav):
        from clean_audio import main
        main([str(work_wav)])
        assert work_wav.exists()

    def test_trim_silences(self, smoke_wav_with_silence, tmp_path):
        output = tmp_path / "trimmed.mp3"
        result = run_cli("generator/trim_silences.py",
                         str(smoke_wav_with_silence), str(output))
        assert result.returncode == 0, result.stderr[:300]
        assert output.exists()

    def test_trim_silences_no_silences(self, smoke_wav, tmp_path):
        """A continuous tone has no silences — should succeed as no-op."""
        output = tmp_path / "trimmed.mp3"
        result = run_cli("generator/trim_silences.py", str(smoke_wav), str(output))
        assert result.returncode == 0, result.stderr[:300]
        assert "No silences" in result.stdout

    def test_master(self, work_mp3, tmp_path):
        from master import main
        output = tmp_path / "mastered.mp3"
        main([str(work_mp3), "-o", str(output)])
        assert output.exists()
        assert output.stat().st_size > 0

    def test_master_analyze(self, smoke_mp3):
        from master import main
        main([str(smoke_mp3), "--analyze"])

    def test_tts_overrides_list(self, overrides_file):
        from tts_overrides import main
        main([str(overrides_file), "--list"])

    def test_tts_overrides_validate(self, overrides_file):
        from tts_overrides import main
        main([str(overrides_file), "--validate"])

    def test_tts_overrides_check(self, overrides_file):
        from tts_overrides import main
        main([str(overrides_file), "--check", "0"])

    def test_mix_preprocess(self, wav_dir, manifest_file):
        from mix_preprocess import main
        mtimes_before = {f.name: f.stat().st_mtime for f in wav_dir.glob("*.wav")}
        main([str(wav_dir), "--manifest", str(manifest_file)])
        for f in wav_dir.glob("*.wav"):
            assert f.exists()
            assert f.stat().st_mtime >= mtimes_before[f.name], \
                f"{f.name} was not modified by preprocess"

    def test_assemble_intro(self, wav_dir, tmp_path):
        from assemble_intro import main
        lines_file = tmp_path / "intro_lines.txt"
        lines_file.write_text(
            "Alex: [warm] Welcome to the show.\n"
            "Morgan: [excited] Thanks for having me.\n",
            encoding="utf-8",
        )
        # Files must be named intro_NNN_speaker.wav
        intro_dir = tmp_path / "intro"
        intro_dir.mkdir()
        shutil.copy2(str(wav_dir / "000_alex.wav"), str(intro_dir / "intro_000_alex.wav"))
        shutil.copy2(str(wav_dir / "001_morgan.wav"), str(intro_dir / "intro_001_morgan.wav"))

        output = tmp_path / "intro.wav"
        main([str(intro_dir), "--lines", str(lines_file), "-o", str(output)])
        assert output.exists()
        assert output.stat().st_size > 0

    def test_analyze_voice(self, smoke_wav):
        from analyze_voice import main
        main([str(smoke_wav)])

    def test_analyze_voice_json(self, smoke_wav, capsys):
        from analyze_voice import main
        main([str(smoke_wav), "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_export_stems(self, wav_dir, script_file, tmp_path):
        from export_stems import main
        stems_dir = tmp_path / "stems"
        main([str(wav_dir), "--script", str(script_file),
              "-o", str(stems_dir)])
        assert stems_dir.exists()
        stem_files = list(stems_dir.glob("*.wav"))
        assert len(stem_files) >= 1

    def test_publish(self, section_dir, script_file):
        from publish import main
        main([str(section_dir), "--script", str(script_file)])
        pub_dir = section_dir / "published"
        assert pub_dir.exists()
        assert (pub_dir / "chapters.json").exists()

    def test_mix_episode_no_master(self, section_dir, tmp_path):
        from mix_episode import main
        output = tmp_path / "mixed.mp3"
        main([str(section_dir), "-o", str(output), "--no-master"])
        assert output.exists()
        assert output.stat().st_size > 0

    def test_generate_backchannels_list(self, bc_dir):
        from generate_backchannels import main
        main(["--list", str(bc_dir)])

    def test_validate_tts_hallucination_check(self):
        """Exercise the hallucination detection logic directly."""
        from validate_tts import check_hallucination
        # Clean transcription — no issues
        is_ok, issues = check_hallucination("hello world", "hello world")
        assert is_ok
        assert not issues
        # Extra words at start — hallucination
        is_ok, issues = check_hallucination("hello world", "garbage garbage hello world")
        assert any("HALLUCINATION_START" in i for i in issues)

    def test_prosody_selector_import(self):
        """ProsodySelector can be imported and instantiated with a synthetic manifest."""
        from prosody_selector import ProsodySelector
        # Doesn't need real files — just test the mapping logic
        selector = ProsodySelector.__new__(ProsodySelector)
        selector.manifest = {
            "emma": {
                "excited": {"file": "/fake/emma_excited.wav", "text": "hello"},
                "calm": {"file": "/fake/emma_calm.wav", "text": "hello"},
            }
        }
        selector.emotion_map = {"fascinated": "excited", "thoughtful": "calm"}
        assert selector.select("emma", "excited") == "/fake/emma_excited.wav"

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    )
    def test_write_script_extract_only(self, tmp_path):
        source = tmp_path / "source.txt"
        source.write_text("AI is transforming podcast production. "
                          "New tools allow automated generation of dialogue.",
                          encoding="utf-8")
        result = run_cli("generator/write_script.py", str(source),
                         "--cast", "lisa,marc", "--extract-only",
                         timeout=60)
        assert result.returncode == 0, result.stderr[:500]


class TestErrorPaths:
    """Bad input should produce non-zero exit codes, not tracebacks."""

    def test_clean_audio_missing_file(self):
        from clean_audio import main
        with pytest.raises(SystemExit):
            main(["nonexistent.wav"])

    def test_master_missing_file(self):
        from master import main
        with pytest.raises(SystemExit):
            main(["nonexistent.mp3", "-o", "out.mp3"])

    def test_master_no_output(self, smoke_mp3):
        from master import main
        with pytest.raises(SystemExit):
            main([str(smoke_mp3)])

    def test_tts_overrides_missing_file(self):
        from tts_overrides import main
        with pytest.raises(SystemExit):
            main(["nonexistent.json", "--list"])

    def test_mix_episode_missing_dir(self):
        from mix_episode import main
        with pytest.raises(SystemExit):
            main(["nonexistent_dir/"])

    def test_trim_silences_missing_file(self):
        result = run_cli("generator/trim_silences.py", "nonexistent.wav", "out.mp3")
        assert result.returncode != 0

    def test_add_realism_missing_file(self):
        result = run_cli("generator/add_realism.py", "nonexistent.mp3", "out.mp3")
        assert result.returncode != 0

    def test_publish_missing_script(self, section_dir):
        from publish import main
        with pytest.raises(SystemExit):
            main([str(section_dir), "--script", "nonexistent.txt"])

    def test_write_script_too_few_speakers(self, tmp_path):
        source = tmp_path / "source.txt"
        source.write_text("Test content.", encoding="utf-8")
        result = run_cli("generator/write_script.py", str(source),
                         "--cast", "solo")
        assert result.returncode != 0

    def test_mix_preprocess_missing_file(self, tmp_path, capsys):
        """Missing files should print [MISSING] warning, not crash."""
        from mix_preprocess import main
        d = tmp_path / "missing_test"
        d.mkdir()
        manifest = [{"file": "nonexistent.wav", "speaker": "alex",
                      "text": "hello", "duration": 0.5}]
        manifest_path = d / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        main([str(d), "--manifest", str(manifest_path)])
        captured = capsys.readouterr()
        assert "MISSING" in captured.out

    def test_mix_episode_no_args(self):
        result = run_cli("generator/mix_episode.py")
        assert result.returncode == 2  # argparse error
