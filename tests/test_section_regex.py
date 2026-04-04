"""Tests for section parsing regex used in generate_episode.py."""

import re


def find_sections(content):
    """Find all section names in the script (copied from generate_episode.py)."""
    sections = []
    pattern = r'={20,}\s*\n([^\n=]+)\s*\n={20,}'
    for match in re.finditer(pattern, content):
        section_name = match.group(1).strip()
        if section_name:
            sections.append(section_name)
    return sections


def extract_section(content, section_name):
    """Extract section content (regex from generate_episode.py)."""
    section_pattern = (
        r'={20,}\s*\n'
        + re.escape(section_name)
        + r'\s*\n={20,}\s*\n'
        + r'(.*?)(?=\n={20,}|\Z)'
    )
    match = re.search(section_pattern, content, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


SCRIPT = """\
====================
OPENING
====================
Emma: [excited] Welcome to the show!
Lucas: [warm] Great to be here.

====================
DISCUSSION
====================
Emma: [curious] So what do you think?
Lucas: [thoughtful] It's complicated.

====================
CLOSING
====================
Emma: [warm] Thanks for listening!
"""


class TestFindSections:
    def test_finds_all_sections(self):
        assert find_sections(SCRIPT) == ["OPENING", "DISCUSSION", "CLOSING"]

    def test_no_sections(self):
        assert find_sections("Just plain text") == []


class TestExtractSection:
    def test_extracts_first_section(self):
        content = extract_section(SCRIPT, "OPENING")
        assert "Welcome to the show" in content
        assert "So what do you think" not in content

    def test_extracts_middle_section(self):
        content = extract_section(SCRIPT, "DISCUSSION")
        assert "So what do you think" in content
        assert "Welcome to the show" not in content
        assert "Thanks for listening" not in content

    def test_extracts_last_section(self):
        content = extract_section(SCRIPT, "CLOSING")
        assert "Thanks for listening" in content
        assert "So what do you think" not in content

    def test_section_not_found(self):
        assert extract_section(SCRIPT, "NONEXISTENT") is None

    def test_case_insensitive(self):
        content = extract_section(SCRIPT, "opening")
        assert "Welcome to the show" in content

    def test_no_cross_boundary_leaking(self):
        """The original bug: regex could match across section boundaries."""
        content = extract_section(SCRIPT, "OPENING")
        lines = [l for l in content.split('\n') if l.strip()]
        # OPENING has exactly 2 dialogue lines
        assert len(lines) == 2
