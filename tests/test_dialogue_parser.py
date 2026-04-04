"""Tests for generator/elevenlabs/src/dialogue_parser.py — emotion tag preservation."""

from src.dialogue_parser import DialogueParser, DialogueLine


class TestDialogueLine:
    def test_default_emotion(self):
        line = DialogueLine("Emma", "Hello world")
        assert line.emotion == "default"

    def test_custom_emotion(self):
        line = DialogueLine("Emma", "Hello world", "excited")
        assert line.emotion == "excited"

    def test_emotion_lowercased(self):
        line = DialogueLine("Emma", "Hello", "EXCITED")
        assert line.emotion == "excited"

    def test_repr_includes_emotion(self):
        line = DialogueLine("Emma", "Hello", "warm")
        assert "warm" in repr(line)


class TestDialogueParserEmotionTags:
    def setup_method(self):
        self.parser = DialogueParser()

    def test_bracket_speaker_with_emotion(self):
        lines = self.parser.parse_text("[Emma]: [excited] This is amazing!")
        assert len(lines) == 1
        assert lines[0].speaker == "EMMA"
        assert lines[0].emotion == "excited"
        assert lines[0].text == "This is amazing!"

    def test_bracket_speaker_without_emotion(self):
        lines = self.parser.parse_text("[Emma]: Just plain text here.")
        assert len(lines) == 1
        assert lines[0].emotion == "default"
        assert lines[0].text == "Just plain text here."

    def test_caps_speaker_with_emotion(self):
        lines = self.parser.parse_text("EMMA: [warm] Welcome to the show!")
        assert len(lines) == 1
        assert lines[0].speaker == "EMMA"
        assert lines[0].emotion == "warm"
        assert lines[0].text == "Welcome to the show!"

    def test_caps_speaker_without_emotion(self):
        lines = self.parser.parse_text("EMMA: No emotion here.")
        assert len(lines) == 1
        assert lines[0].emotion == "default"

    def test_multi_word_emotion(self):
        lines = self.parser.parse_text("[Lucas]: [genuinely interested] Tell me more.")
        assert len(lines) == 1
        assert lines[0].emotion == "genuinely interested"
        assert lines[0].text == "Tell me more."

    def test_multiple_lines_preserve_emotions(self):
        script = (
            "[Emma]: [excited] Welcome!\n"
            "[Lucas]: [calm] Thanks for having me.\n"
            "[Emma]: Let's get started.\n"
        )
        lines = self.parser.parse_text(script)
        assert len(lines) == 3
        assert lines[0].emotion == "excited"
        assert lines[1].emotion == "calm"
        assert lines[2].emotion == "default"

    def test_skips_comments_and_headers(self):
        script = (
            "# This is a comment\n"
            "====================\n"
            "[Emma]: [warm] Hello!\n"
        )
        lines = self.parser.parse_text(script)
        assert len(lines) == 1
        assert lines[0].emotion == "warm"
