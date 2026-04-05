#!/usr/bin/env python3
"""
Script generator: sources → podcast dialogue script.

Uses a 3-pass LLM pipeline:
  1. Extract key facts, hooks, and narrative angles from sources
  2. Draft a dialogue script following NARRATIVE_DESIGN.md rules
  3. Dialogue director pass: punch up, fix pacing, validate format

Usage:
    python generator/write_script.py sources/article.txt --cast lisa,marc,sven
    python generator/write_script.py https://example.com/article --cast lisa,marc --lang en
    python generator/write_script.py paper.pdf notes.txt --cast lisa,marc,sven --length 20
    python generator/write_script.py sources/ --cast lisa,marc,sven -o script.txt
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import anthropic
except ImportError:
    sys.exit("Missing dependency: pip install anthropic")

# ---------------------------------------------------------------------------
# Source ingestion
# ---------------------------------------------------------------------------

def read_text_file(path: Path) -> str:
    """Read a plain text, markdown, or similar file."""
    return path.read_text(encoding="utf-8")


def read_pdf(path: Path) -> str:
    """Extract text from a PDF using pymupdf."""
    try:
        import pymupdf  # noqa: F811
    except ImportError:
        sys.exit(f"PDF support requires pymupdf: pip install pymupdf")
    doc = pymupdf.open(str(path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages)


def fetch_url(url: str) -> str:
    """Extract article text from a URL using trafilatura."""
    try:
        import trafilatura
    except ImportError:
        sys.exit("URL support requires trafilatura: pip install trafilatura")
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        sys.exit(f"Failed to download: {url}")
    text = trafilatura.extract(downloaded)
    if not text:
        sys.exit(f"Failed to extract text from: {url}")
    return text


def ingest_source(source: str) -> str:
    """Read a single source — file path, directory, or URL."""
    if source.startswith("http://") or source.startswith("https://"):
        return fetch_url(source)

    path = Path(source)
    if not path.exists():
        sys.exit(f"Source not found: {source}")

    if path.is_dir():
        texts = []
        for f in sorted(path.iterdir()):
            if f.suffix in (".txt", ".md", ".pdf"):
                texts.append(f"--- {f.name} ---\n{ingest_source(str(f))}")
        if not texts:
            sys.exit(f"No .txt/.md/.pdf files found in: {source}")
        return "\n\n".join(texts)

    if path.suffix == ".pdf":
        return read_pdf(path)

    return read_text_file(path)


def ingest_sources(sources: list[str]) -> str:
    """Read and concatenate all sources."""
    parts = []
    for src in sources:
        parts.append(ingest_source(src))
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def call_llm(client: anthropic.Anthropic, model: str, system: str, user: str,
             max_tokens: int = 8192) -> str:
    """Single Claude API call. Returns the text response."""
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


# ---------------------------------------------------------------------------
# Pass 1: Extract
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = """\
You are a research assistant preparing material for a podcast script writer.

Your job: analyze the source material and extract a structured brief that a \
script writer can turn into a compelling podcast episode.

Return valid JSON with this structure:
{
  "topic": "One-line topic summary",
  "angle": "The specific angle or thesis for this episode",
  "key_facts": ["fact 1", "fact 2", ...],
  "surprising_findings": ["thing most people don't know 1", ...],
  "narrative_hooks": ["hook that grabs attention 1", ...],
  "potential_disagreements": ["point where reasonable people disagree 1", ...],
  "concrete_examples": ["vivid, specific example 1", ...],
  "quotes": ["notable quote with attribution 1", ...],
  "modern_relevance": ["why this matters today 1", ...],
  "suggested_arc": "Brief description of how the episode could flow"
}

Focus on what is INTERESTING, SURPRISING, or DEBATABLE — not just what is \
important. Prioritize vivid details and concrete examples over abstractions. \
Extract at least 3 items per category when the source material allows."""


def pass_extract(client: anthropic.Anthropic, model: str, source_text: str,
                 topic_override: str | None = None) -> dict:
    """Pass 1: Extract structured brief from sources."""
    user_msg = f"Analyze this source material and extract a structured brief.\n\n"
    if topic_override:
        user_msg += f"Focus on this angle: {topic_override}\n\n"
    user_msg += f"SOURCE MATERIAL:\n\n{source_text}"

    response = call_llm(client, model, EXTRACT_SYSTEM, user_msg)
    # Strip markdown code fences if present
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Pass 2: Draft script
# ---------------------------------------------------------------------------

DRAFT_SYSTEM = """\
You are an expert podcast script writer. You write dialogue that sounds like \
real conversation between intelligent friends — never a presenter reading to \
an audience.

RULES (non-negotiable):
1. Every line must follow this format exactly: Speaker: [emotion] Dialogue text
2. Use ONLY these emotion tags: [warm], [curious], [surprised], [thoughtful], \
[excited], [skeptical], [emphatic], [building], [passionate], [calm], \
[neutral], [casual], [quiet], [realizing], [fascinated], [explaining]
3. Emotion tags must be in English even if the dialogue is in another language.
4. Spell out all numbers: "nineteen twenty-two" not "1922".

STRUCTURE — 5 segments:
- OPENING (2-3 min): Hook + rapport + preview of the "wow moment"
- SEGMENT 1 (context, 2-4 min): Plant a mystery or paradox
- SEGMENT 2 (deep dive, 3-5 min): Core concept exploration
- SEGMENT 3 (emotional pivot, 1-2 min): Source voice, quote, or key revelation
- SEGMENT 4 (application, 3-5 min): Concrete examples, specific references
- CLOSING (2-3 min): Emotional recap + call to action + next episode teaser

DIALOGUE DYNAMICS:
- Ping-Pong-Plus: each exchange escalates (Statement → Question → Deepening → Realization)
- Vary turn length DRAMATICALLY: mix single-word reactions ("...Forty years.") \
with 30-second monologues. This contrast creates rhythm.
- Include interruptions (em-dash mid-sentence), incomplete thoughts, self-corrections
- 2-3 thinking sounds per segment ("Well, actually...", "Hmm, how do I explain this...")
- At least 1 pushback/disagreement per segment from the host
- Backchanneling: short reactive lines between longer turns ("Three years?" / "Exactly.")
- Complete each other's thoughts occasionally

GUARD RAILS:
- NO agreement spiral — the host must challenge, question, push back
- NO "as you know, Bob" — never have characters explain things they'd both already know
- NO symmetric turn lengths — if every turn is 1-3 sentences, you've failed
- NO exposition dumps — break complex ideas across multiple exchanges
- At least 2-3 genuine emotional moments per episode (vulnerability, shared wonder, \
personal connection)

SPEAKING TIME:
- Host: ~35% (audience proxy — asks naive questions, reacts emotionally)
- Expert: ~60% (knowledge carrier — makes complex ideas accessible)
- Source voice (if 3 speakers): ~5% (historical figure, quote, or narrator — adds weight)

WORD COUNT: Target {word_target} words for a {length_minutes}-minute episode.

Do NOT include section headers or metadata. Output ONLY dialogue lines."""


def pass_draft(client: anthropic.Anthropic, model: str, brief: dict,
               cast: list[str], lang: str, length_minutes: int) -> str:
    """Pass 2: Generate dialogue script from brief."""
    word_target = length_minutes * 150  # ~150 words per minute for dialogue
    system = DRAFT_SYSTEM.format(
        word_target=word_target, length_minutes=length_minutes,
    )

    # Build cast description
    roles = ["host (audience proxy, asks questions, reacts)",
             "expert (knowledge carrier, explains, gives examples)"]
    if len(cast) > 2:
        roles.append("source voice (historical figure, narrator, or interviewee — used sparingly)")
    cast_desc = "\n".join(f"- {name}: {role}" for name, role in zip(cast, roles))

    lang_instruction = ""
    if lang != "en":
        lang_map = {"nl": "Dutch", "de": "German", "fr": "French", "es": "Spanish"}
        lang_name = lang_map.get(lang, lang)
        lang_instruction = (
            f"\n\nIMPORTANT: Write ALL dialogue in {lang_name}. "
            f"Emotion tags must still be in English (e.g., [excited] not [begeistert])."
        )

    user_msg = f"""Write a podcast script using this research brief and cast.

CAST:
{cast_desc}

RESEARCH BRIEF:
{json.dumps(brief, indent=2)}
{lang_instruction}

Write the full episode script now. Remember: every line must be "Speaker: [emotion] text"."""

    return call_llm(client, model, system, user_msg, max_tokens=16384)


# ---------------------------------------------------------------------------
# Pass 3: Dialogue director
# ---------------------------------------------------------------------------

DIRECTOR_SYSTEM = """\
You are a dialogue director for podcast scripts. Your job is to review and \
punch up an existing script draft to make it sound like a real conversation.

You receive a script draft. Return the IMPROVED script — same format, same \
speakers, same topic — but better.

WHAT TO FIX:
1. Agreement spiral: add disagreement, skepticism, pushback where it's too cozy
2. Turn-length uniformity: if turns are all similar length, vary them — add \
single-word reactions, longer monologues, interruptions
3. Missing emotional beats: ensure 2-3 genuine moments (vulnerability, wonder, \
personal connection)
4. "As you know, Bob": remove any line where characters explain things they'd \
both already know
5. Robotic transitions: replace "That's interesting" / "Great point" with \
specific reactions
6. Missing texture: add interruptions (em-dash), incomplete thoughts, \
self-corrections, backchanneling where natural
7. Flat endings: the closing should land emotionally, not just summarize

WHAT TO PRESERVE:
- The overall topic, structure, and factual content
- The speaker names and their roles
- The line format: Speaker: [emotion] text
- Key emotional moments that already work
- Approximate word count (don't pad or cut significantly)

WHAT TO CHECK:
- Every line has format "Speaker: [emotion] text"
- Emotion tags are valid English words in brackets
- Numbers are spelled out
- No section headers or metadata — just dialogue lines

Output ONLY the improved script. No commentary, no notes."""


def pass_director(client: anthropic.Anthropic, model: str, draft: str) -> str:
    """Pass 3: Dialogue director polish pass."""
    user_msg = f"Review and improve this podcast script:\n\n{draft}"
    return call_llm(client, model, DIRECTOR_SYSTEM, user_msg, max_tokens=16384)


# ---------------------------------------------------------------------------
# Format validation
# ---------------------------------------------------------------------------

LINE_PATTERN = __import__("re").compile(
    r'^[A-Za-z_]+:\s*\[[\w\s]+\]\s*.+$'
)


def validate_script(script: str) -> list[str]:
    """Check that every non-blank line matches the expected format. Returns warnings."""
    warnings = []
    for i, line in enumerate(script.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        if not LINE_PATTERN.match(stripped):
            warnings.append(f"Line {i}: unexpected format: {stripped[:80]}")
    return warnings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a podcast script from source material.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python generator/write_script.py article.txt --cast lisa,marc,sven\n"
               "  python generator/write_script.py https://example.com --cast lisa,marc\n"
               "  python generator/write_script.py paper.pdf notes.md --cast a,b,c --length 25\n",
    )
    p.add_argument("sources", nargs="+",
                    help="Source files (txt/md/pdf), directories, or URLs")
    p.add_argument("--cast", required=True,
                    help="Comma-separated speaker names (first=host, second=expert, third=source voice)")
    p.add_argument("--lang", default="en",
                    help="Target language code (default: en)")
    p.add_argument("--length", type=int, default=20,
                    help="Target episode length in minutes (default: 20)")
    p.add_argument("--model", default="claude-sonnet-4-6",
                    help="Claude model to use (default: claude-sonnet-4-6)")
    p.add_argument("--topic",
                    help="Override topic/angle (auto-detected from sources if omitted)")
    p.add_argument("-o", "--output",
                    help="Output file path (default: script_YYYYMMDD.txt)")
    p.add_argument("--extract-only", action="store_true",
                    help="Stop after extraction pass and print the brief as JSON")
    p.add_argument("--no-director", action="store_true",
                    help="Skip the dialogue director pass (faster, rougher output)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cast = [name.strip().capitalize() for name in args.cast.split(",")]

    if len(cast) < 2:
        sys.exit("Need at least 2 speakers (--cast host,expert)")
    if len(cast) > 3:
        sys.exit("Maximum 3 speakers supported")

    # Resolve API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Try loading from .env files in common locations
        try:
            from dotenv import load_dotenv
            for env_path in [Path(".env"), Path("generator/.env"),
                             Path("generator/elevenlabs/.env")]:
                if env_path.exists():
                    load_dotenv(env_path)
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        except ImportError:
            pass
    if not api_key:
        sys.exit("Set ANTHROPIC_API_KEY environment variable or add it to .env")

    client = anthropic.Anthropic(api_key=api_key)

    # --- Ingest sources ---
    print(f"Reading {len(args.sources)} source(s)...")
    source_text = ingest_sources(args.sources)
    print(f"  {len(source_text):,} characters ingested")

    # Truncate very long sources to avoid blowing context
    max_chars = 100_000
    if len(source_text) > max_chars:
        print(f"  Truncating to {max_chars:,} characters")
        source_text = source_text[:max_chars]

    # --- Pass 1: Extract ---
    print(f"\nPass 1/3: Extracting key facts and narrative hooks...")
    brief = pass_extract(client, args.model, source_text, args.topic)
    print(f"  Topic: {brief.get('topic', 'unknown')}")
    print(f"  Angle: {brief.get('angle', 'unknown')}")
    print(f"  {len(brief.get('key_facts', []))} facts, "
          f"{len(brief.get('surprising_findings', []))} surprises, "
          f"{len(brief.get('narrative_hooks', []))} hooks")

    if args.extract_only:
        print(json.dumps(brief, indent=2))
        return

    # --- Pass 2: Draft ---
    print(f"\nPass 2/3: Drafting {args.length}-minute script for {', '.join(cast)}...")
    draft = pass_draft(client, args.model, brief, cast, args.lang, args.length)
    draft_lines = [l for l in draft.splitlines() if l.strip()]
    print(f"  {len(draft_lines)} dialogue lines, ~{len(draft.split())} words")

    # --- Pass 3: Director ---
    if args.no_director:
        final = draft
        print("\nSkipping director pass (--no-director)")
    else:
        print(f"\nPass 3/3: Dialogue director polish...")
        final = pass_director(client, args.model, draft)
        final_lines = [l for l in final.splitlines() if l.strip()]
        print(f"  {len(final_lines)} dialogue lines, ~{len(final.split())} words")

    # --- Validate ---
    warnings = validate_script(final)
    if warnings:
        print(f"\nFormat warnings ({len(warnings)}):")
        for w in warnings[:10]:
            print(f"  {w}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")

    # --- Write output ---
    output_path = args.output or f"script_{datetime.now():%Y%m%d_%H%M%S}.txt"
    Path(output_path).write_text(final, encoding="utf-8")
    print(f"\nScript written to: {output_path}")
    print(f"Ready for: python generator/elevenlabs/generate_episode.py {output_path}")


if __name__ == "__main__":
    main()
