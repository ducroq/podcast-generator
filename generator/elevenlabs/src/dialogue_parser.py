"""
Parse dialogue scripts from text files.

Expected format:
[speaker]: [tone] Dialogue text here
[speaker]: More dialogue

Also supports:
SPEAKER_NAME: Dialogue text here

Lines without a speaker prefix are treated as narrator text.
Special markers like (langzaam), (stilte), (pauze) are preserved.
"""

import re
from typing import List, Dict


class DialogueLine:
    """Represents a single line of dialogue."""

    def __init__(self, speaker: str, text: str, emotion: str = "default"):
        self.speaker = speaker.upper()
        self.text = text.strip()
        self.emotion = emotion.lower()

    def __repr__(self):
        return f"DialogueLine(speaker='{self.speaker}', emotion='{self.emotion}', text='{self.text[:30]}...')"


class DialogueParser:
    """Parse dialogue scripts into structured data."""

    def __init__(self, narrator_name: str = "NARRATOR"):
        self.narrator_name = narrator_name
        # Pattern to match "[speaker]: [tone] text" or "[speaker]: text"
        self.bracket_speaker_pattern = re.compile(r'^\[(\w+)\]:\s*(?:\[([\w\s]+)\]\s*)?(.+)$')
        # Pattern to match "SPEAKER: [tone] text" or "SPEAKER: text"
        self.caps_speaker_pattern = re.compile(r'^([A-Z_]+):\s*(?:\[([\w\s]+)\]\s*)?(.+)$')

    def parse_file(self, filepath: str) -> List[DialogueLine]:
        """Parse a dialogue script file and return a list of DialogueLine objects."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        return self.parse_text(content)

    def parse_text(self, text: str) -> List[DialogueLine]:
        """Parse dialogue text and return a list of DialogueLine objects."""
        lines = []

        for line in text.split('\n'):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip comments
            if line.startswith('#'):
                continue

            # Skip segment headers (lines starting with ==)
            if line.startswith('=='):
                continue

            # Try to match bracket speaker pattern first: [speaker]:
            match = self.bracket_speaker_pattern.match(line)
            if match:
                speaker = match.group(1)
                emotion = match.group(2) or "default"
                dialogue_text = match.group(3)
                lines.append(DialogueLine(speaker, dialogue_text, emotion))
                continue

            # Try to match caps speaker pattern: SPEAKER:
            match = self.caps_speaker_pattern.match(line)
            if match:
                speaker = match.group(1)
                emotion = match.group(2) or "default"
                dialogue_text = match.group(3)
                lines.append(DialogueLine(speaker, dialogue_text, emotion))
                continue

            # Lines without speaker are skipped (title lines, etc.)
            # You can uncomment the next line to treat them as narrator
            # lines.append(DialogueLine(self.narrator_name, line))

        return lines

    def get_speakers(self, dialogue_lines: List[DialogueLine]) -> List[str]:
        """Extract unique speaker names from dialogue lines."""
        speakers = set(line.speaker for line in dialogue_lines)
        return sorted(list(speakers))
