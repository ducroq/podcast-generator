"""Tests for generator/write_script.py — script generation pipeline."""

import json
import re
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generator.write_script import (
    ingest_source,
    ingest_sources,
    validate_script,
    pass_extract,
    pass_draft,
    pass_director,
    pass_pronunciation,
    pass_review,
    pass_revise,
    format_review_summary,
    _parse_json_response,
    build_parser,
    LINE_PATTERN,
    EXTRACT_SYSTEM,
    DRAFT_SYSTEM,
    DIRECTOR_SYSTEM,
    PRONUNCIATION_SYSTEM,
    REVIEW_FIDELITY_SYSTEM,
    REVIEW_LISTENER_SYSTEM,
    REVIEW_NARRATIVE_SYSTEM,
    REVISE_SYSTEM,
)


# ---------------------------------------------------------------------------
# Source ingestion
# ---------------------------------------------------------------------------

class TestIngestSource:
    def test_read_text_file(self, tmp_path):
        f = tmp_path / "article.txt"
        f.write_text("Hello world", encoding="utf-8")
        assert ingest_source(str(f)) == "Hello world"

    def test_read_markdown_file(self, tmp_path):
        f = tmp_path / "notes.md"
        f.write_text("# Title\nBody", encoding="utf-8")
        assert "# Title" in ingest_source(str(f))

    def test_read_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("First", encoding="utf-8")
        (tmp_path / "b.txt").write_text("Second", encoding="utf-8")
        (tmp_path / "skip.jpg").write_text("ignored", encoding="utf-8")
        result = ingest_source(str(tmp_path))
        assert "First" in result
        assert "Second" in result
        assert "ignored" not in result

    def test_empty_directory_exits(self, tmp_path):
        (tmp_path / "photo.jpg").write_text("nope", encoding="utf-8")
        with pytest.raises(SystemExit, match="No .txt/.md/.pdf"):
            ingest_source(str(tmp_path))

    def test_missing_file_exits(self):
        with pytest.raises(SystemExit, match="not found"):
            ingest_source("/nonexistent/file.txt")

    def test_url_without_trafilatura(self):
        """URL ingestion fails gracefully when trafilatura is not installed."""
        with patch.dict("sys.modules", {"trafilatura": None}):
            with pytest.raises(SystemExit, match="trafilatura"):
                ingest_source("https://example.com/article")


class TestIngestSources:
    def test_multiple_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("Alpha", encoding="utf-8")
        (tmp_path / "b.txt").write_text("Beta", encoding="utf-8")
        result = ingest_sources([str(tmp_path / "a.txt"), str(tmp_path / "b.txt")])
        assert "Alpha" in result
        assert "Beta" in result
        assert "---" in result  # separator


# ---------------------------------------------------------------------------
# Format validation
# ---------------------------------------------------------------------------

class TestValidateScript:
    def test_valid_script(self):
        script = textwrap.dedent("""\
            Lisa: [curious] So what happened next?
            Marc: [thoughtful] Well, it's complicated.
            Sven: [skeptical] I don't buy it.
        """)
        assert validate_script(script) == []

    def test_blank_lines_ignored(self):
        script = "Lisa: [curious] Hello\n\n\nMarc: [warm] Hi\n"
        assert validate_script(script) == []

    def test_missing_emotion_tag_flagged(self):
        script = "Lisa: No emotion tag here."
        warnings = validate_script(script)
        assert len(warnings) == 1
        assert "Line 1" in warnings[0]

    def test_section_header_flagged(self):
        script = "=== OPENING ===\nLisa: [curious] Hello"
        warnings = validate_script(script)
        assert len(warnings) == 1  # the header line

    def test_multi_word_emotion(self):
        """Emotion tags like [building excitement] with spaces should be valid."""
        script = "Lisa: [building excitement] Wow!"
        assert validate_script(script) == []


class TestLinePattern:
    @pytest.mark.parametrize("line", [
        "Lisa: [curious] What happened?",
        "Marc: [warm] Let me explain.",
        "Sven: [skeptical] Really?",
        "Source_voice: [calm] In eighteen ninety-five...",
    ])
    def test_valid_lines(self, line):
        assert LINE_PATTERN.match(line)

    @pytest.mark.parametrize("line", [
        "Just some text without a speaker",
        "=== SEGMENT 1 ===",
        "",
        "Lisa: No brackets here",
    ])
    def test_invalid_lines(self, line):
        assert not LINE_PATTERN.match(line)


# ---------------------------------------------------------------------------
# LLM pass prompts (structure checks)
# ---------------------------------------------------------------------------

class TestPrompts:
    def test_extract_system_mentions_json(self):
        assert "JSON" in EXTRACT_SYSTEM

    def test_draft_system_has_format_placeholders(self):
        assert "{word_target}" in DRAFT_SYSTEM
        assert "{length_minutes}" in DRAFT_SYSTEM

    def test_draft_system_mentions_guard_rails(self):
        assert "agreement spiral" in DRAFT_SYSTEM.lower()
        assert "as you know" in DRAFT_SYSTEM.lower()

    def test_draft_system_mentions_show_dont_tell(self):
        assert "SHOW FIRST" in DRAFT_SYSTEM
        assert "EARN THE EXPLANATION" in DRAFT_SYSTEM
        assert "PERSONAL STORIES" in DRAFT_SYSTEM

    def test_director_system_mentions_format(self):
        assert "Speaker: [emotion]" in DIRECTOR_SYSTEM

    def test_review_fidelity_system_mentions_json(self):
        assert "JSON" in REVIEW_FIDELITY_SYSTEM or "json" in REVIEW_FIDELITY_SYSTEM.lower()

    def test_review_listener_system_mentions_attention(self):
        assert "ATTENTION_DRIFT" in REVIEW_LISTENER_SYSTEM

    def test_review_narrative_system_mentions_checklist(self):
        assert "PASS" in REVIEW_NARRATIVE_SYSTEM
        assert "FAIL" in REVIEW_NARRATIVE_SYSTEM

    def test_review_narrative_system_includes_show_dont_tell(self):
        assert "scene or anecdote" in REVIEW_NARRATIVE_SYSTEM
        assert "host, not the expert" in REVIEW_NARRATIVE_SYSTEM

    def test_review_listener_system_includes_telling_not_showing(self):
        assert "TELLING_NOT_SHOWING" in REVIEW_LISTENER_SYSTEM

    def test_revise_system_prioritizes_high_severity(self):
        assert "HIGH" in REVISE_SYSTEM


# ---------------------------------------------------------------------------
# LLM pass mocking
# ---------------------------------------------------------------------------

def make_mock_client(response_text: str):
    """Create a mock Anthropic client that returns a fixed response."""
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=response_text)]
    client.messages.create.return_value = msg
    return client


class TestPassExtract:
    def test_parses_json_response(self):
        brief = {"topic": "Test", "key_facts": ["fact1"], "surprising_findings": [],
                 "narrative_hooks": [], "potential_disagreements": [],
                 "concrete_examples": [], "quotes": [], "modern_relevance": [],
                 "suggested_arc": "linear"}
        client = make_mock_client(json.dumps(brief))
        result = pass_extract(client, "claude-sonnet-4-6", "some source text")
        assert result["topic"] == "Test"

    def test_strips_markdown_fences(self):
        brief = {"topic": "Fenced"}
        client = make_mock_client(f"```json\n{json.dumps(brief)}\n```")
        result = pass_extract(client, "claude-sonnet-4-6", "text")
        assert result["topic"] == "Fenced"

    def test_topic_override_included_in_prompt(self):
        client = make_mock_client(json.dumps({"topic": "X"}))
        pass_extract(client, "claude-sonnet-4-6", "text", topic_override="AI ethics")
        call_args = client.messages.create.call_args
        user_content = call_args.kwargs["messages"][0]["content"]
        assert "AI ethics" in user_content


class TestPassDraft:
    def test_returns_script_text(self):
        script = "Lisa: [curious] Hello\nMarc: [warm] Hi there"
        client = make_mock_client(script)
        brief = {"topic": "Test", "key_facts": ["fact"]}
        result = pass_draft(client, "claude-sonnet-4-6", brief,
                           ["Lisa", "Marc"], "en", 20)
        assert "Lisa:" in result
        assert "Marc:" in result

    def test_non_english_instruction(self):
        client = make_mock_client("Lisa: [curious] Hallo")
        brief = {"topic": "Test"}
        pass_draft(client, "claude-sonnet-4-6", brief, ["Lisa", "Marc"], "nl", 20)
        call_args = client.messages.create.call_args
        user_content = call_args.kwargs["messages"][0]["content"]
        assert "Dutch" in user_content

    def test_three_speakers_includes_source_voice(self):
        client = make_mock_client("Lisa: [curious] Hi")
        brief = {"topic": "Test"}
        pass_draft(client, "claude-sonnet-4-6", brief,
                   ["Lisa", "Marc", "Narrator"], "en", 15)
        call_args = client.messages.create.call_args
        user_content = call_args.kwargs["messages"][0]["content"]
        assert "source voice" in user_content.lower()


class TestPassDirector:
    def test_returns_improved_script(self):
        improved = "Lisa: [excited] Much better!\nMarc: [warm] Agreed."
        client = make_mock_client(improved)
        result = pass_director(client, "claude-sonnet-4-6", "Lisa: [flat] Okay")
        assert "excited" in result


# ---------------------------------------------------------------------------
# Pronunciation pass
# ---------------------------------------------------------------------------

class TestPassPronunciation:
    def test_returns_script(self):
        fixed = "Daan: [warm] Today we discuss Shee Jin-ping's latest move."
        client = make_mock_client(fixed)
        result = pass_pronunciation(client, "claude-sonnet-4-6",
                                   "Daan: [warm] Today we discuss Xi Jinping's latest move.")
        assert "Shee Jin-ping" in result

    def test_lang_included_in_prompt(self):
        client = make_mock_client("Daan: [warm] Hello")
        pass_pronunciation(client, "claude-sonnet-4-6", "Daan: [warm] Hello", lang="nl")
        call_args = client.messages.create.call_args
        user_content = call_args.kwargs["messages"][0]["content"]
        assert "Dutch" in user_content

    def test_english_default_lang(self):
        client = make_mock_client("Lisa: [curious] Hello")
        pass_pronunciation(client, "claude-sonnet-4-6", "Lisa: [curious] Hello")
        call_args = client.messages.create.call_args
        user_content = call_args.kwargs["messages"][0]["content"]
        assert "English" in user_content

    def test_preserves_format(self):
        """Mock returns expected format — verifies pass_pronunciation passes it through."""
        script = "Sofie: [curious] What about Volodymyr Zelenskyy?"
        fixed = "Sofie: [curious] What about Vo-lo-DEE-mir Ze-LEN-skee?"
        client = make_mock_client(fixed)
        result = pass_pronunciation(client, "claude-sonnet-4-6", script)
        assert result.startswith("Sofie: [curious]")

    def test_unknown_lang_fallback(self):
        """Unknown lang codes should pass through as-is without error."""
        client = make_mock_client("Daan: [warm] Hello")
        pass_pronunciation(client, "claude-sonnet-4-6", "Daan: [warm] Hello", lang="ja")
        call_args = client.messages.create.call_args
        user_content = call_args.kwargs["messages"][0]["content"]
        assert "ja" in user_content

    def test_strips_markdown_fences(self):
        """Model wrapping output in fences should be handled."""
        script = "Lisa: [curious] Hello"
        client = make_mock_client(f"```\n{script}\n```")
        result = pass_pronunciation(client, "claude-sonnet-4-6", script)
        assert result == script


class TestPronunciationSystem:
    def test_mentions_phonetic(self):
        assert "phonetic" in PRONUNCIATION_SYSTEM.lower()

    def test_mentions_tts(self):
        assert "TTS" in PRONUNCIATION_SYSTEM

    def test_mentions_foreign(self):
        assert "foreign" in PRONUNCIATION_SYSTEM.lower()

    def test_mentions_format_preservation(self):
        assert "Speaker: [emotion]" in PRONUNCIATION_SYSTEM


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestCLI:
    def test_minimal_args(self):
        parser = build_parser()
        args = parser.parse_args(["source.txt", "--cast", "lisa,marc"])
        assert args.sources == ["source.txt"]
        assert args.cast == "lisa,marc"
        assert args.lang == "en"
        assert args.length == 20

    def test_all_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            "a.txt", "b.pdf",
            "--cast", "lisa,marc,sven",
            "--lang", "de",
            "--length", "25",
            "--model", "claude-opus-4-6",
            "--topic", "AI ethics",
            "-o", "out.txt",
            "--no-director",
        ])
        assert args.sources == ["a.txt", "b.pdf"]
        assert args.lang == "de"
        assert args.length == 25
        assert args.model == "claude-opus-4-6"
        assert args.topic == "AI ethics"
        assert args.output == "out.txt"
        assert args.no_director is True

    def test_extract_only_flag(self):
        parser = build_parser()
        args = parser.parse_args(["source.txt", "--cast", "a,b", "--extract-only"])
        assert args.extract_only is True

    def test_cast_required(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["source.txt"])

    def test_review_flag(self):
        parser = build_parser()
        args = parser.parse_args(["source.txt", "--cast", "a,b", "--review",
                                   "--listener", "28yo new grad"])
        assert args.review is True
        assert args.listener == "28yo new grad"

    def test_review_only_flag(self):
        parser = build_parser()
        args = parser.parse_args(["source.txt", "--cast", "a,b",
                                   "--review-only", "existing_script.txt"])
        assert args.review_only == "existing_script.txt"

    def test_no_pronunciation_flag(self):
        parser = build_parser()
        args = parser.parse_args(["source.txt", "--cast", "a,b", "--no-pronunciation"])
        assert args.no_pronunciation is True

    def test_pronunciation_on_by_default(self):
        parser = build_parser()
        args = parser.parse_args(["source.txt", "--cast", "a,b"])
        assert args.no_pronunciation is False


# ---------------------------------------------------------------------------
# Review and revise passes
# ---------------------------------------------------------------------------

class TestParseJsonResponse:
    def test_plain_json(self):
        result = _parse_json_response('{"key": "value"}')
        assert result["key"] == "value"

    def test_markdown_fenced_json(self):
        result = _parse_json_response('```json\n{"key": "value"}\n```')
        assert result["key"] == "value"

    def test_invalid_json_returns_raw(self):
        result = _parse_json_response("not json at all")
        assert result["parse_error"] is True
        assert "not json" in result["raw"]


class TestPassReview:
    def test_returns_three_review_sections(self):
        """Review should call the LLM 3 times and return fidelity/listener/narrative."""
        fidelity = json.dumps({"issues": [], "faithful": ["good"]})
        listener = json.dumps({"issues": [], "strengths": ["engaging"], "verdict": "good"})
        narrative = json.dumps({"scores": [], "top_improvements": [], "overall_score": "10/12"})

        client = MagicMock()
        responses = [fidelity, listener, narrative]
        msgs = []
        for r in responses:
            msg = MagicMock()
            msg.content = [MagicMock(text=r)]
            msgs.append(msg)
        client.messages.create.side_effect = msgs

        result = pass_review(client, "claude-sonnet-4-6",
                            "Lisa: [curious] Hello", "source text")
        assert "fidelity" in result
        assert "listener" in result
        assert "narrative" in result
        assert client.messages.create.call_count == 3

    def test_listener_description_included(self):
        """Listener description should appear in the system prompt."""
        response = json.dumps({"issues": [], "strengths": [], "verdict": "ok"})
        client = MagicMock()
        # We only care about the second call (listener review)
        msg = MagicMock()
        msg.content = [MagicMock(text=response)]
        client.messages.create.return_value = msg

        pass_review(client, "claude-sonnet-4-6", "script", "source",
                   listener_desc="28yo professional")
        # Second call should have listener desc in system
        calls = client.messages.create.call_args_list
        listener_call = calls[1]
        assert "28yo professional" in listener_call.kwargs["system"]


class TestFormatReviewSummary:
    def test_formats_fidelity_issues(self):
        review = {
            "fidelity": {
                "issues": [
                    {"severity": "HIGH", "type": "DISTORTION",
                     "explanation": "Misquoted the source"},
                ],
                "faithful": []
            },
            "listener": {"issues": [], "verdict": "Solid episode"},
            "narrative": {"scores": [], "top_improvements": [], "overall_score": "9/12"},
        }
        summary = format_review_summary(review)
        assert "HIGH" in summary
        assert "DISTORTION" in summary
        assert "Solid episode" in summary
        assert "9/12" in summary

    def test_handles_empty_review(self):
        review = {"fidelity": {}, "listener": {}, "narrative": {}}
        summary = format_review_summary(review)
        assert "SOURCE FIDELITY" in summary


class TestPassRevise:
    def test_returns_revised_script(self):
        revised = "Lisa: [excited] Fixed version!\nMarc: [warm] Much better."
        client = make_mock_client(revised)
        review = {
            "fidelity": {"issues": [{"severity": "HIGH", "type": "DISTORTION",
                                     "explanation": "wrong fact"}]},
            "listener": {"issues": [], "verdict": "ok"},
            "narrative": {"scores": [], "top_improvements": []},
        }
        result = pass_revise(client, "claude-sonnet-4-6", "old script", review)
        assert "Fixed version" in result
