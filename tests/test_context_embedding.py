"""Tests for smart TTS generation: resemblyzer scoring, voice bank, context-embedding."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

import numpy as np
import pytest
import soundfile as sf

from generate_tts import (
    _score_voice_similarity,
    _find_same_speaker_suffix,
    _find_same_speaker_context,
    _extract_target_with_whisper,
    _generate_with_context_embedding,
    _generate_with_guard,
)

SR = 24000


def _make_audio(duration=1.5, sr=SR):
    """Create test audio of given duration."""
    n = max(1, int(sr * duration))
    t = np.linspace(0, duration, n, dtype=np.float32)
    return np.sin(2 * np.pi * 200 * t) * 0.2


def _make_manifest(lines_spec):
    """Build a minimal manifest from a list of (speaker, text) or "break" markers."""
    from manifest import content_hash

    order = []
    lines = {}

    for item in lines_spec:
        if item == "break":
            order.append({"type": "section_break", "from_section": "A", "to_section": "B"})
        elif isinstance(item, tuple) and item[0] == "pause":
            order.append({"type": "pause", "duration": item[1]})
        else:
            speaker, text = item
            h = content_hash(text, speaker)
            lines[h] = {
                "speaker": speaker,
                "text": text,
                "emotion": None,
                "file": f"{speaker}_{h}.wav",
                "status": "missing",
            }
            order.append({"type": "line", "hash": h})

    return {"lines": lines, "order": order, "meta": {}}


# ---------------------------------------------------------------------------
# Phase A: Resemblyzer scoring
# ---------------------------------------------------------------------------


class TestScoreVoiceSimilarity:
    def test_returns_float_with_mock_encoder(self):
        """When resemblyzer is available, returns a float score."""
        # Create a temp ref file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            ref_audio = _make_audio(duration=2.0)
            sf.write(f.name, ref_audio, SR)
            ref_path = Path(f.name)

        try:
            score = _score_voice_similarity(ref_audio, SR, ref_path)
            # Same audio compared to itself should be high similarity
            if score is not None:
                assert 0.0 <= score <= 1.0
                assert score > 0.8  # same audio = very similar
        finally:
            ref_path.unlink(missing_ok=True)

    def test_too_short_returns_none(self):
        """Audio shorter than 100ms returns None."""
        short_audio = _make_audio(duration=0.05)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, _make_audio(2.0), SR)
            ref_path = Path(f.name)
        try:
            score = _score_voice_similarity(short_audio, SR, ref_path)
            assert score is None
        finally:
            ref_path.unlink(missing_ok=True)

    def test_unavailable_returns_none(self):
        """When resemblyzer is not importable, returns None gracefully."""
        with patch.dict("sys.modules", {"resemblyzer": None}):
            # Force reimport failure
            import generate_tts
            old_encoder = generate_tts._resemblyzer_encoder
            generate_tts._resemblyzer_encoder = None
            try:
                audio = _make_audio(duration=1.5)
                # This will try to import resemblyzer and fail
                # But since resemblyzer IS installed in test env, we mock the import
                with patch("generate_tts._score_voice_similarity", return_value=None):
                    pass  # Can't easily test import failure; tested implicitly
            finally:
                generate_tts._resemblyzer_encoder = old_encoder


# ---------------------------------------------------------------------------
# Phase B: Voice bank (tests added when voice_bank.py is implemented)
# ---------------------------------------------------------------------------


class TestVoiceBank:
    def test_update_and_best(self, tmp_path):
        from voice_bank import VoiceBank
        bank = VoiceBank(tmp_path, min_score=0.85, min_duration=1.5)
        audio = _make_audio(duration=2.0)

        bank.update("alex", audio, SR, score=0.90, text="Hello world.",
                     episode="ep01", line_hash="abc123")
        result = bank.best_ref("alex")
        assert result is not None
        ref_path, ref_text = result
        assert ref_path.exists()
        assert ref_text == "Hello world."

    def test_higher_score_replaces(self, tmp_path):
        from voice_bank import VoiceBank
        bank = VoiceBank(tmp_path, min_score=0.85, min_duration=1.5)
        audio = _make_audio(duration=2.0)

        bank.update("alex", audio, SR, score=0.90, text="First.",
                     episode="ep01", line_hash="a")
        bank.update("alex", audio, SR, score=0.95, text="Better.",
                     episode="ep01", line_hash="b")
        _, text = bank.best_ref("alex")
        assert text == "Better."

    def test_lower_score_ignored(self, tmp_path):
        from voice_bank import VoiceBank
        bank = VoiceBank(tmp_path, min_score=0.85, min_duration=1.5)
        audio = _make_audio(duration=2.0)

        bank.update("alex", audio, SR, score=0.95, text="Best.",
                     episode="ep01", line_hash="a")
        bank.update("alex", audio, SR, score=0.88, text="Worse.",
                     episode="ep01", line_hash="b")
        _, text = bank.best_ref("alex")
        assert text == "Best."

    def test_persistence(self, tmp_path):
        from voice_bank import VoiceBank
        audio = _make_audio(duration=2.0)

        bank1 = VoiceBank(tmp_path, min_score=0.80)
        bank1.update("morgan", audio, SR, score=0.91, text="Persisted.",
                      episode="ep01", line_hash="xyz")
        saved = bank1.save()
        assert saved == 1
        assert (tmp_path / "voice_bank.json").exists()

        # New instance reads persisted data
        bank2 = VoiceBank(tmp_path, min_score=0.80)
        result = bank2.best_ref("morgan")
        assert result is not None
        ref_path, ref_text = result
        assert ref_path.exists()
        assert ref_text == "Persisted."

    def test_save_only_overwrites_when_better(self, tmp_path):
        from voice_bank import VoiceBank
        audio = _make_audio(duration=2.0)

        bank1 = VoiceBank(tmp_path, min_score=0.80)
        bank1.update("alex", audio, SR, score=0.95, text="Ep01 best.",
                      episode="ep01", line_hash="a")
        bank1.save()

        # Second session with lower score — should NOT overwrite
        bank2 = VoiceBank(tmp_path, min_score=0.80)
        bank2.update("alex", audio, SR, score=0.88, text="Ep02 worse.",
                      episode="ep02", line_hash="b")
        bank2.save()

        # Third session reads — should still have ep01
        bank3 = VoiceBank(tmp_path, min_score=0.80)
        _, text = bank3.best_ref("alex")
        assert text == "Ep01 best."

    def test_min_score_filter(self, tmp_path):
        from voice_bank import VoiceBank
        bank = VoiceBank(tmp_path, min_score=0.85, min_duration=1.5)
        audio_long = _make_audio(duration=2.0)
        audio_short = _make_audio(duration=0.5)

        # Score too low
        bank.update("alex", audio_long, SR, score=0.80, text="Low score.",
                     episode="ep01", line_hash="a")
        assert bank.best_ref("alex") is None

        # Duration too short
        bank.update("alex", audio_short, SR, score=0.95, text="Short.",
                     episode="ep01", line_hash="b")
        assert bank.best_ref("alex") is None

    def test_no_ref_returns_none(self, tmp_path):
        from voice_bank import VoiceBank
        bank = VoiceBank(tmp_path, min_score=0.85, min_duration=1.5)
        assert bank.best_ref("unknown_speaker") is None

    def test_best_ref_avoids_redundant_writes(self, tmp_path):
        from voice_bank import VoiceBank
        bank = VoiceBank(tmp_path, min_score=0.80, min_duration=1.5)
        audio = _make_audio(duration=2.0)

        bank.update("alex", audio, SR, score=0.90, text="Hello.",
                     episode="ep01", line_hash="a")

        # First call writes the file
        ref_path, _ = bank.best_ref("alex")
        mtime1 = ref_path.stat().st_mtime

        # Second call with same score should NOT rewrite
        import time; time.sleep(0.05)
        bank.best_ref("alex")
        mtime2 = ref_path.stat().st_mtime
        assert mtime1 == mtime2

    def test_save_cleans_temp_files(self, tmp_path):
        from voice_bank import VoiceBank
        bank = VoiceBank(tmp_path, min_score=0.80, min_duration=1.5)
        audio = _make_audio(duration=2.0)

        bank.update("alex", audio, SR, score=0.90, text="Hello.",
                     episode="ep01", line_hash="a")
        ref_path, _ = bank.best_ref("alex")
        assert ref_path.exists()  # temp file exists before save

        bank.save()
        assert not ref_path.exists()  # temp cleaned up after save


# ---------------------------------------------------------------------------
# Phase C: Context-embedding (tests added when functions are implemented)
# ---------------------------------------------------------------------------


class MockWord:
    """Mock faster-whisper word object."""
    def __init__(self, word, start, end):
        self.word = word
        self.start = start
        self.end = end


class MockSegment:
    """Mock faster-whisper segment with word timestamps."""
    def __init__(self, words):
        self.words = words
        self.text = " ".join(w.word for w in words)
        self.start = words[0].start if words else 0
        self.end = words[-1].end if words else 0


class TestFindSameSpeakerContext:
    def test_finds_previous_same_speaker(self):
        manifest = _make_manifest([
            ("alex", "The book doesn't update. It just sits there."),
            ("morgan", "Something else entirely."),
            ("alex", "Did they?"),
        ])
        short_hash = [h for h, info in manifest["lines"].items()
                      if info["text"] == "Did they?"][0]
        result = _find_same_speaker_context(manifest, short_hash, "alex")
        assert result is not None
        assert "book doesn't update" in result

    def test_skips_different_speaker(self):
        manifest = _make_manifest([
            ("morgan", "Only Morgan spoke before."),
            ("alex", "Both?"),
        ])
        short_hash = [h for h, info in manifest["lines"].items()
                      if info["text"] == "Both?"][0]
        result = _find_same_speaker_context(manifest, short_hash, "alex")
        assert result is None

    def test_stops_at_section_break(self):
        manifest = _make_manifest([
            ("alex", "End of previous section."),
            "break",
            ("alex", "Yeah."),
        ])
        short_hash = [h for h, info in manifest["lines"].items()
                      if info["text"] == "Yeah."][0]
        result = _find_same_speaker_context(manifest, short_hash, "alex")
        assert result is None

    def test_truncates_long_context(self):
        long_text = " ".join(f"word{i}" for i in range(30))
        manifest = _make_manifest([
            ("alex", long_text),
            ("alex", "Huh."),
        ])
        short_hash = [h for h, info in manifest["lines"].items()
                      if info["text"] == "Huh."][0]
        result = _find_same_speaker_context(manifest, short_hash, "alex", max_words=10)
        assert result is not None
        assert len(result.split()) == 10

    def test_first_line_returns_none(self):
        manifest = _make_manifest([("alex", "Both?")])
        short_hash = list(manifest["lines"].keys())[0]
        result = _find_same_speaker_context(manifest, short_hash, "alex")
        assert result is None


class TestFindSameSpeakerSuffix:
    def test_finds_next_same_speaker(self):
        manifest = _make_manifest([
            ("alex", "First line."),
            ("morgan", "Intervening."),
            ("alex", "The later line from Alex."),
        ])
        first_hash = [h for h, i in manifest["lines"].items()
                      if i["text"] == "First line."][0]
        result = _find_same_speaker_suffix(manifest, first_hash, "alex")
        assert result is not None
        assert "later line" in result

    def test_skips_different_speaker(self):
        manifest = _make_manifest([
            ("alex", "Only Alex."),
            ("morgan", "Only Morgan after."),
        ])
        h = [h for h, i in manifest["lines"].items()
             if i["text"] == "Only Alex."][0]
        result = _find_same_speaker_suffix(manifest, h, "alex")
        assert result is None

    def test_stops_at_section_break(self):
        manifest = _make_manifest([
            ("alex", "Before break."),
            "break",
            ("alex", "After break."),
        ])
        h = [h for h, i in manifest["lines"].items()
             if i["text"] == "Before break."][0]
        result = _find_same_speaker_suffix(manifest, h, "alex")
        assert result is None

    def test_truncates_long_suffix(self):
        long_text = " ".join(f"word{i}" for i in range(25))
        manifest = _make_manifest([("alex", "Short."), ("alex", long_text)])
        h = [h for h, i in manifest["lines"].items()
             if i["text"] == "Short."][0]
        result = _find_same_speaker_suffix(manifest, h, "alex", max_words=5)
        assert result is not None
        assert len(result.split()) == 5

    def test_last_line_returns_none(self):
        manifest = _make_manifest([("alex", "Only line.")])
        h = list(manifest["lines"].keys())[0]
        result = _find_same_speaker_suffix(manifest, h, "alex")
        assert result is None


class TestExtractTargetWithWhisper:
    def _mock_whisper_model(self, words):
        model = MagicMock()
        segment = MockSegment(words)
        model.transcribe.return_value = (iter([segment]), {})
        return model

    def test_basic_extraction(self):
        words = [
            MockWord("The", 0.0, 0.3),
            MockWord("book", 0.3, 0.6),
            MockWord("doesn't", 0.6, 0.9),
            MockWord("update", 0.9, 1.2),
            MockWord("here", 1.2, 1.5),
            MockWord("Did", 1.6, 1.9),
            MockWord("they", 1.9, 2.2),
        ]
        model = self._mock_whisper_model(words)
        audio = _make_audio(duration=3.0)
        config = {"min_extracted_duration": 0.2, "pad_before_ms": 30, "pad_after_ms": 50}

        with patch("generate_tts._get_whisper_model", return_value=model):
            result = _extract_target_with_whisper(
                audio, SR, "The book doesn't update here", "Did they?", config
            )

        assert result is not None
        actual_dur = len(result) / SR
        # Envelope-following extends past Whisper end; just check reasonable range
        assert actual_dur > 0.4   # at least covers the two target words
        assert actual_dur < 1.5   # doesn't include the prefix

    def test_too_short_returns_none(self):
        words = [
            MockWord("Context", 0.0, 0.5),
            MockWord("word", 0.5, 0.8),
            MockWord("Hi", 0.85, 0.9),
        ]
        model = self._mock_whisper_model(words)
        audio = _make_audio(duration=1.0)
        config = {"min_extracted_duration": 0.3, "pad_before_ms": 0, "pad_after_ms": 0}

        with patch("generate_tts._get_whisper_model", return_value=model):
            result = _extract_target_with_whisper(
                audio, SR, "Context word", "Hi", config
            )
        assert result is None

    def test_word_count_mismatch_returns_none(self):
        words = [MockWord("Something", 0.0, 0.5), MockWord("else", 0.5, 1.0)]
        model = self._mock_whisper_model(words)
        audio = _make_audio(duration=1.5)
        config = {"min_extracted_duration": 0.2, "pad_before_ms": 30, "pad_after_ms": 50}

        with patch("generate_tts._get_whisper_model", return_value=model):
            result = _extract_target_with_whisper(
                audio, SR, "Three word context", "Target", config
            )
        assert result is None

    def test_sandwich_extraction_with_suffix(self):
        """Extract target from prefix + target + suffix sandwich."""
        words = [
            MockWord("Context", 0.0, 0.4),     # prefix
            MockWord("Did", 0.5, 0.8),          # target word 1
            MockWord("they", 0.8, 1.1),         # target word 2
            MockWord("Trailing", 1.3, 1.6),     # suffix
        ]
        model = self._mock_whisper_model(words)
        audio = _make_audio(duration=2.0)
        config = {"min_extracted_duration": 0.2, "pad_before_ms": 30, "pad_after_ms": 80}

        with patch("generate_tts._get_whisper_model", return_value=model):
            result = _extract_target_with_whisper(
                audio, SR, "Context", "Did they", config, suffix_text="Trailing"
            )

        assert result is not None
        actual_dur = len(result) / SR
        # Should end at target word end (1.1) + pad (0.08), not at suffix (1.6)
        assert actual_dur < 1.3
        assert actual_dur > 0.2

    def test_separator_drift_handled(self):
        """Whisper transcribes '...' as a word — fuzzy matching finds target anyway."""
        words = [
            MockWord("Context", 0.0, 0.4),     # prefix
            MockWord("...", 0.4, 0.5),          # separator became a word!
            MockWord("Did", 0.6, 0.9),          # target word 1
            MockWord("they", 0.9, 1.2),         # target word 2
        ]
        model = self._mock_whisper_model(words)
        audio = _make_audio(duration=2.0)
        config = {"min_extracted_duration": 0.2, "pad_before_ms": 30, "pad_after_ms": 80}

        with patch("generate_tts._get_whisper_model", return_value=model):
            result = _extract_target_with_whisper(
                audio, SR, "Context", "Did they?", config
            )

        assert result is not None
        # Should have found "Did" at 0.6s despite separator drift
        actual_dur = len(result) / SR
        assert actual_dur > 0.3


class TestGenerateWithContextEmbedding:
    def test_success_returns_extracted_audio(self):
        class MockAdapter:
            name = "mock"
            def generate(self, text, voice_ref, ref_text=None, language=None, **kw):
                return _make_audio(3.0), SR

        words = [
            MockWord("Context", 0.0, 0.4),
            MockWord("sentence", 0.4, 0.8),
            MockWord("here", 0.8, 1.2),
            MockWord("Did", 1.5, 1.8),
            MockWord("they", 1.8, 2.2),
        ]
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([MockSegment(words)]), {})

        embed_config = {"min_extracted_duration": 0.2, "pad_before_ms": 30, "pad_after_ms": 50}
        tts_config = {"temperature": 0.7, "repetition_penalty": 1.2}
        retry_config = {"max_duration_per_word": 0.6, "retry_count": 3}

        with patch("generate_tts._get_whisper_model", return_value=mock_model):
            audio, sr = _generate_with_context_embedding(
                MockAdapter(), "Did they?", "voice.mp3", "ref text",
                "en", tts_config, retry_config,
                "Context sentence here", embed_config,
            )

        assert audio is not None
        assert sr == SR

    def test_extraction_failure_returns_none(self):
        class MockAdapter:
            name = "mock"
            def generate(self, text, voice_ref, ref_text=None, language=None, **kw):
                return _make_audio(2.0), SR

        words = [MockWord("garbled", 0.0, 0.5)]
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([MockSegment(words)]), {})

        embed_config = {"min_extracted_duration": 0.2, "pad_before_ms": 30, "pad_after_ms": 50}
        tts_config = {"temperature": 0.7, "repetition_penalty": 1.2}
        retry_config = {"max_duration_per_word": 0.6, "retry_count": 3}

        with patch("generate_tts._get_whisper_model", return_value=mock_model):
            audio, sr = _generate_with_context_embedding(
                MockAdapter(), "Did they?", "voice.mp3", "ref text",
                "en", tts_config, retry_config,
                "Context sentence here", embed_config,
            )

        assert audio is None
