"""
Voice configuration management for mapping speakers to ElevenLabs voices.
"""

import os
from typing import Dict
from dotenv import load_dotenv


class VoiceConfig:
    """Manage voice mappings for different speakers."""

    def __init__(self, env_file: str = '.env'):
        load_dotenv(env_file)
        self.api_key = os.getenv('ELEVENLABS_API_KEY')
        self.voice_map = self._load_voice_mappings()

        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not found in environment variables")

    def _load_voice_mappings(self) -> Dict[str, str]:
        """Load voice mappings from environment variables."""
        voice_map = {}

        for key, value in os.environ.items():
            if key.startswith('VOICE_') and value:
                speaker_name = key.replace('VOICE_', '')
                voice_map[speaker_name] = value

        return voice_map

    def get_voice_id(self, speaker: str) -> str:
        """Get the voice ID for a given speaker."""
        if speaker not in self.voice_map:
            raise ValueError(
                f"No voice mapping found for speaker '{speaker}'. "
                f"Available speakers: {list(self.voice_map.keys())}"
            )
        return self.voice_map[speaker]

    def has_speaker(self, speaker: str) -> bool:
        """Check if a voice mapping exists for a speaker."""
        return speaker in self.voice_map

    def get_all_speakers(self) -> list:
        """Get all configured speakers."""
        return list(self.voice_map.keys())
