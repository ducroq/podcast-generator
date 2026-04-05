#!/usr/bin/env python3
"""
Prosody reference selector: maps (voice, emotion) to the best-matching
reference audio clip for voice cloning with Qwen/Chatterbox.

Usage:
    from prosody_selector import ProsodySelector

    selector = ProsodySelector("/path/to/prosody_manifest.json")
    ref_path = selector.select("emma", "excited")
    # Returns: "/home/hcl/voice_refs/prosody_refs/emma_excited.wav"

    ref_path = selector.select("emma", "fascinated")
    # Returns: "excited" ref (closest match via EMOTION_MAP)

    ref_path = selector.select("unknown_voice", "calm")
    # Returns: None (voice not in manifest)
"""

import json
from pathlib import Path


# Map script emotion tags to prosody ref categories.
# Left: emotion tags used in scripts (from voice_settings.py EMOTIONAL_VARIANTS).
# Right: prosody ref categories (excited, calm, emphatic, contemplative, urgent).
EMOTION_MAP = {
    # → excited
    "excited": "excited",
    "enthusiastic": "excited",
    "passionate": "excited",
    "fascinated": "excited",
    "surprised": "excited",
    "amazed": "excited",
    "impressed": "excited",
    "realizing": "excited",
    # → calm
    "calm": "calm",
    "warm": "calm",
    "neutral": "calm",
    "casual": "calm",
    "quiet": "calm",
    "default": "calm",
    "friendly": "calm",
    # → emphatic
    "emphatic": "emphatic",
    "building": "emphatic",
    "confident": "emphatic",
    "explaining": "emphatic",
    "dramatic": "emphatic",
    "determined": "emphatic",
    # → contemplative
    "contemplative": "contemplative",
    "thoughtful": "contemplative",
    "curious": "contemplative",
    "reflective": "contemplative",
    "philosophical": "contemplative",
    "hesitant": "contemplative",
    "processing": "contemplative",
    # → urgent
    "urgent": "urgent",
    "serious": "urgent",
    "skeptical": "urgent",
    "concerned": "urgent",
    "rushed": "urgent",
}

# Default prosody when emotion is not in the map
DEFAULT_PROSODY = "calm"


class ProsodySelector:
    """Select prosody reference clips based on voice and emotion."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        with open(self.manifest_path, encoding="utf-8") as f:
            self.manifest = json.load(f)

    def voices(self) -> list[str]:
        """Return list of voices with prosody refs."""
        return sorted(self.manifest.keys())

    def emotions(self, voice: str) -> list[str]:
        """Return list of available prosody categories for a voice."""
        voice_data = self.manifest.get(voice.lower(), {})
        return sorted(voice_data.keys())

    def select(self, voice: str, emotion: str) -> str | None:
        """Select the best prosody ref for a voice and emotion.

        Returns the file path to the ref clip, or None if voice is not found.
        Falls back to DEFAULT_PROSODY if emotion is unmapped.
        """
        voice_key = voice.lower()
        voice_data = self.manifest.get(voice_key)
        if not voice_data:
            return None

        # Map script emotion to prosody category
        emotion_key = emotion.lower()
        prosody = EMOTION_MAP.get(emotion_key, DEFAULT_PROSODY)

        # Look up ref clip
        ref = voice_data.get(prosody)
        if not ref:
            # Fallback to default prosody
            ref = voice_data.get(DEFAULT_PROSODY)
        if not ref:
            # Last resort: return first available
            ref = next(iter(voice_data.values()), None)

        return ref["file"] if ref else None

    def select_with_text(self, voice: str, emotion: str) -> tuple[str, str] | None:
        """Like select(), but also returns the ref text for Qwen (which needs it).

        Returns (file_path, ref_text) or None if voice not found.
        """
        voice_key = voice.lower()
        voice_data = self.manifest.get(voice_key)
        if not voice_data:
            return None

        emotion_key = emotion.lower()
        prosody = EMOTION_MAP.get(emotion_key, DEFAULT_PROSODY)

        ref = voice_data.get(prosody) or voice_data.get(DEFAULT_PROSODY)
        if not ref:
            ref = next(iter(voice_data.values()), None)

        return (ref["file"], ref["text"]) if ref else None
