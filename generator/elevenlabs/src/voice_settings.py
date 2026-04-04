"""Shared voice settings for ElevenLabs generation scripts."""

from elevenlabs.types import VoiceSettings

BASE_SIMILARITY = 0.95
STABILITY_OFFSET = -0.10

EMOTIONAL_VARIANTS = {
    "emma": {
        "default": {"stability": 0.4, "style": 0.4},
        "enthusiastic": {"stability": 0.35, "style": 0.5},
        "excited": {"stability": 0.3, "style": 0.55},
        "curious": {"stability": 0.38, "style": 0.45},
        "thoughtful": {"stability": 0.5, "style": 0.3},
        "calm": {"stability": 0.55, "style": 0.25},
        "warm": {"stability": 0.45, "style": 0.35},
        "surprised": {"stability": 0.3, "style": 0.5},
        "amused": {"stability": 0.35, "style": 0.45},
        "listening": {"stability": 0.45, "style": 0.2},
        "understanding": {"stability": 0.42, "style": 0.38},
        "realizing": {"stability": 0.38, "style": 0.45},
        "fascinated": {"stability": 0.33, "style": 0.48},
        "agreeing": {"stability": 0.4, "style": 0.4},
        "interested": {"stability": 0.38, "style": 0.42},
        "hesitant": {"stability": 0.48, "style": 0.25},
        "laughing": {"stability": 0.3, "style": 0.55},
        "moved": {"stability": 0.5, "style": 0.35},
        "practical": {"stability": 0.4, "style": 0.35},
        "confused": {"stability": 0.35, "style": 0.4},
        "amazed": {"stability": 0.32, "style": 0.5},
        "processing": {"stability": 0.45, "style": 0.35},
        "impressed": {"stability": 0.38, "style": 0.45},
        "philosophical": {"stability": 0.5, "style": 0.3},
        "observant": {"stability": 0.42, "style": 0.38},
        "descriptive": {"stability": 0.4, "style": 0.4},
        "questioning": {"stability": 0.38, "style": 0.42},
        "insightful": {"stability": 0.38, "style": 0.45},
        "connecting": {"stability": 0.4, "style": 0.42},
        "intrigued": {"stability": 0.35, "style": 0.48},
        "anticipating": {"stability": 0.38, "style": 0.45},
    },
    "lucas": {
        "default": {"stability": 0.35, "style": 0.4},
        "warm": {"stability": 0.4, "style": 0.35},
        "enthusiastic": {"stability": 0.3, "style": 0.5},
        "thoughtful": {"stability": 0.45, "style": 0.3},
        "calm": {"stability": 0.5, "style": 0.25},
        "confident": {"stability": 0.38, "style": 0.45},
        "amused": {"stability": 0.32, "style": 0.48},
        "explanatory": {"stability": 0.38, "style": 0.42},
        "confirming": {"stability": 0.37, "style": 0.4},
        "mysterious": {"stability": 0.35, "style": 0.5},
        "revealing": {"stability": 0.33, "style": 0.48},
        "excited": {"stability": 0.3, "style": 0.52},
        "proud": {"stability": 0.38, "style": 0.42},
        "meaningful": {"stability": 0.42, "style": 0.38},
        "passionate": {"stability": 0.32, "style": 0.52},
        "emphatic": {"stability": 0.35, "style": 0.48},
        "agreeing": {"stability": 0.37, "style": 0.4},
        "encouraging": {"stability": 0.38, "style": 0.42},
        "friendly": {"stability": 0.4, "style": 0.38},
        "curious": {"stability": 0.35, "style": 0.45},
        "surprised": {"stability": 0.28, "style": 0.52},
        "listening": {"stability": 0.45, "style": 0.25},
        "informative": {"stability": 0.4, "style": 0.38},
        "dramatic": {"stability": 0.32, "style": 0.5},
        "descriptive": {"stability": 0.38, "style": 0.4},
        "admiring": {"stability": 0.4, "style": 0.4},
        "genuinely interested": {"stability": 0.38, "style": 0.42},
    },
    "piet": {
        "default": {"stability": 0.5, "style": 0.3},
        "thoughtful": {"stability": 0.6, "style": 0.2},
        "passionate": {"stability": 0.4, "style": 0.5},
        "determined": {"stability": 0.45, "style": 0.4},
    }
}


def get_voice_settings(speaker, emotion):
    """Get VoiceSettings for speaker and emotion."""
    variants = EMOTIONAL_VARIANTS.get(speaker, {})
    variant = variants.get(emotion, variants.get("default", {"stability": 0.4, "style": 0.4}))

    adjusted_stability = max(0.15, variant["stability"] + STABILITY_OFFSET)

    return VoiceSettings(
        stability=adjusted_stability,
        similarity_boost=BASE_SIMILARITY,
        style=variant["style"]
    )


def parse_line(line):
    """Parse a dialogue line into speaker, emotion, and text."""
    import re
    match = re.match(r'(\w+):\s*\[(\w+(?:\s+\w+)*)\]\s*(.*)', line.strip())
    if match:
        speaker = match.group(1).lower()
        emotion = match.group(2).lower()
        text = match.group(3).strip()
        return speaker, emotion, text
    return None, None, None
