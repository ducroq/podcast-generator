"""Persistent voice reference bank — tracks best-scoring TTS output per speaker.

Saves the best line per speaker per episode as a dynamic voice reference.
References persist across episodes via a JSON index, so each episode
starts with the best-known voice for each character.

Usage:
    from voice_bank import VoiceBank

    bank = VoiceBank(bank_dir)
    bank.update("alex", audio, sr, score=0.94, text="...", episode="ep01", line_hash="abc123")
    ref_path, ref_text = bank.best_ref("alex")
    bank.save()  # persist after generation completes
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def _safe_filename(s):
    """Sanitize a string for use as a filename component (no path traversal)."""
    return re.sub(r"[^\w\-]", "_", s)


class VoiceBank:
    """Manages dynamic voice references that improve across episodes."""

    def __init__(self, bank_dir, min_score=0.85, min_duration=1.5):
        self.bank_dir = Path(bank_dir)
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.bank_dir / "voice_bank.json"
        self.min_score = min_score
        self.min_duration = min_duration
        self._index = self._load_index()
        self._session_best = {}  # speaker -> {score, audio, sr, text, hash, episode}
        self._written_scores = {}  # speaker -> last written score (avoids redundant writes)

    def _load_index(self):
        """Load existing index or return empty dict."""
        if self.index_path.exists():
            try:
                with open(self.index_path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("Could not load voice bank index, starting fresh")
        return {}

    def update(self, speaker, audio, sr, score, text, episode, line_hash):
        """Track a generated line's score. Keeps the session best per speaker."""
        duration = len(audio) / sr

        if score < self.min_score:
            return
        if duration < self.min_duration:
            return

        current_best = self._session_best.get(speaker)
        if current_best is None or score > current_best["score"]:
            self._session_best[speaker] = {
                "score": score,
                "audio": audio.copy(),
                "sr": sr,
                "text": text,
                "hash": line_hash,
                "episode": episode,
            }

    def best_ref(self, speaker):
        """Return (wav_path, ref_text) for the best known reference.

        Checks session best first, then persisted index.
        Returns None if no dynamic reference available.
        """
        session = self._session_best.get(speaker)
        if session:
            safe_name = _safe_filename(speaker)
            temp_path = self.bank_dir / f"{safe_name}_session_best.wav"
            # Only write when score improved since last write
            if self._written_scores.get(speaker) != session["score"]:
                sf.write(str(temp_path), session["audio"], session["sr"])
                self._written_scores[speaker] = session["score"]
            return temp_path, session["text"]

        # Persisted best from previous episodes
        entry = self._index.get(speaker)
        if entry:
            ref_path = self.bank_dir / entry["file"]
            if ref_path.exists():
                return ref_path, entry["text"]

        logger.debug("Voice bank: no dynamic ref for %s, using static", speaker)
        return None

    def best_score(self, speaker):
        """Return the best known score for a speaker (session or persisted)."""
        scores = []
        session = self._session_best.get(speaker)
        if session:
            scores.append(session["score"])
        persisted = self._index.get(speaker)
        if persisted:
            scores.append(persisted.get("score", 0))
        return max(scores) if scores else 0.0

    def save(self):
        """Persist session bests to disk. Call after generation completes.

        Uses write-then-rename for the JSON index to prevent corruption
        on crash. Only overwrites persisted ref if session best exceeds it.
        """
        saved = 0
        for speaker, entry in self._session_best.items():
            persisted = self._index.get(speaker)
            persisted_score = persisted.get("score", 0) if persisted else 0

            if entry["score"] > persisted_score:
                filename = f"{_safe_filename(speaker)}_{_safe_filename(entry['episode'])}_best.wav"
                wav_path = self.bank_dir / filename
                sf.write(str(wav_path), entry["audio"], entry["sr"])

                self._index[speaker] = {
                    "score": round(entry["score"], 3),
                    "file": filename,
                    "text": entry["text"],
                    "episode": entry["episode"],
                    "hash": entry["hash"],
                }
                saved += 1
                logger.info("Voice bank: saved %s (score=%.3f, ep=%s)",
                            speaker, entry["score"], entry["episode"])

        # Atomic-ish index write: write to temp, then rename
        if saved > 0:
            tmp_path = self.index_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._index, f, indent=2)
            tmp_path.replace(self.index_path)
            logger.info("Voice bank: %d speaker(s) updated", saved)

        # Clean up session temp files (use _safe_filename to match best_ref)
        for speaker in self._session_best:
            temp = self.bank_dir / f"{_safe_filename(speaker)}_session_best.wav"
            if temp.exists():
                temp.unlink()

        self._written_scores.clear()
        return saved
