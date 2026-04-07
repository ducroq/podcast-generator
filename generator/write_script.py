#!/usr/bin/env python3
"""
Script generator: sources → podcast dialogue script.

Uses a 4-pass LLM pipeline:
  1. Extract key facts, hooks, and narrative angles from sources
  2. Draft a dialogue script following NARRATIVE_DESIGN.md rules
  3. Dialogue director pass: punch up, fix pacing, validate format
  4. Pronunciation pass: replace foreign proper nouns with phonetic respellings for TTS

Usage:
    python generator/write_script.py sources/article.txt --cast lisa,marc,sven
    python generator/write_script.py https://example.com/article --cast lisa,marc --lang en
    python generator/write_script.py paper.pdf notes.txt --cast lisa,marc,sven --length 20
    python generator/write_script.py sources/ --cast lisa,marc,sven -o script.txt
"""

import argparse
import json
import os
import re
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
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text
    except anthropic.APIError as e:
        sys.exit(f"Claude API error: {e}")


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
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        sys.exit(f"Extract pass returned malformed JSON: {e}\nResponse:\n{text[:500]}")


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

SHOW, DON'T TELL (critical):
- SHOW FIRST, NAME SECOND: before explaining any concept, ground it in a scene, \
anecdote, or concrete moment the listener recognizes. The listener should FEEL \
the phenomenon before anyone labels it.
- Characters must bring PERSONAL STORIES, not just positions. "I was in a meeting \
where..." or "I bought this book because..." — not "Take Apple, for instance..."
- EARN THE EXPLANATION: the host or skeptic should arrive at insights BEFORE the \
expert labels them. The Realization in Ping-Pong-Plus doesn't always belong to \
the expert — sometimes the host gets there first, sometimes it emerges from \
the conversation without anyone claiming it.
- The opening should SHOW the phenomenon, not describe it. Drop the listener into \
a recognizable moment.

DIALOGUE DYNAMICS:
- Ping-Pong-Plus: each exchange escalates (Statement → Question → Deepening → Realization)
- Vary turn length DRAMATICALLY: mix single-word reactions ("...Forty years.") \
with 30-second monologues. This contrast creates rhythm.
- Include interruptions (em-dash mid-sentence), incomplete thoughts, self-corrections
- 2-3 thinking sounds per segment ("Well, actually...", "Hmm, how do I explain this...")
- At least 1 pushback/disagreement per segment from the host
- Backchanneling: short reactive lines between longer turns ("Three years?" / "Exactly.")
- Complete each other's thoughts occasionally

TTS PACING (critical for audio generation):
- Use "..." (ellipsis) for thinking pauses, trailing off, collecting thoughts. \
TTS creates a held pause. Example: "I think... maybe part of this is just how people tell stories."
- Use "---" ONLY for interruptions and abrupt cut-offs. TTS creates sharp energy break.
- Use commas for micro-pauses within sentences.
- NEVER use "ehm" or "uh" as filler. Instead:
  - Older/professional characters: "well...", "right, but...", "I don't know if..."
  - Young/direct characters: "like...", "I mean...", "okay so..."
  - Vulnerable moments: use silence (just "...") — the pause IS the hesitation
- Lines over 40 words should have natural breathing points (comma, ellipsis, or \
em-dash) to prevent monotone TTS delivery.

GUARD RAILS:
- NO agreement spiral — the host must challenge, question, push back
- NO "as you know, Bob" — never have characters explain things they'd both already know
- NO symmetric turn lengths — if every turn is 1-3 sentences, you've failed
- NO exposition dumps — break complex ideas across multiple exchanges
- NO "expert explains, others react" — if the expert is the only one generating \
insights, you're writing a lecture, not a conversation
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
1. Telling before showing: if a concept is explained before the listener has \
felt it through a scene or anecdote, add the scene first. Move the label after.
2. Expert-only insights: if every realization belongs to the expert, give some \
to the host or skeptic. They should arrive at insights too.
3. Missing personal stories: characters should bring anecdotes ("I was in a \
meeting where..."), not just analytical positions ("Take Apple, for instance...")
4. Agreement spiral: add disagreement, skepticism, pushback where it's too cozy
5. Turn-length uniformity: if turns are all similar length, vary them — add \
single-word reactions, longer monologues, interruptions
6. Missing emotional beats: ensure 2-3 genuine moments (vulnerability, wonder, \
personal connection)
7. "As you know, Bob": remove any line where characters explain things they'd \
both already know
8. Robotic transitions: replace "That's interesting" / "Great point" with \
specific reactions
9. Missing texture: add interruptions (em-dash), incomplete thoughts, \
self-corrections, backchanneling where natural
10. Flat endings: the closing should land emotionally, not just summarize

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
- SHORT LINES (1-3 words): TTS engines produce unreliable output for very short \
utterances. Pad them naturally: "Both?" → "Wait... both?" or "Both of them?". \
"Yeah." → "Yeah, I think so." Consecutive short lines from the same speaker \
should be merged into one line. The ONLY acceptable 1-word lines are strong \
emotional reactions that need isolation ("...Forty years." / "Exactly.").

Output ONLY the improved script. No commentary, no notes."""


def pass_director(client: anthropic.Anthropic, model: str, draft: str) -> str:
    """Pass 3: Dialogue director polish pass."""
    user_msg = f"Review and improve this podcast script:\n\n{draft}"
    return call_llm(client, model, DIRECTOR_SYSTEM, user_msg, max_tokens=16384)


# ---------------------------------------------------------------------------
# Pass 3.5: Pronunciation (phonetic respellings for TTS)
# ---------------------------------------------------------------------------

PRONUNCIATION_SYSTEM = """\
You are a pronunciation specialist preparing podcast scripts for TTS \
(text-to-speech) engines. Your job: find foreign proper nouns, place names, \
and terms that a TTS engine will likely mispronounce, and replace them with \
phonetic respellings that guide the TTS toward correct pronunciation.

RULES:
1. Replace foreign names with phonetic respellings throughout the script. \
Use intuitive English-style spelling: "Zelenskyy" → "Zelenskee", \
"Xi Jinping" → "Shee Jin-ping", "Bundesverfassungsgericht" → "Bun-des-fer-FAH-sungs-ge-rikht".
2. Use hyphens to separate syllables and CAPS for stressed syllables when \
the stress pattern is non-obvious. Keep it readable, not IPA.
3. Do NOT respell names that English TTS handles well (Paris, Berlin, Tokyo, \
Einstein, Mozart). Only respell names that TTS engines commonly butcher.
4. Do NOT respell names that match the script language. A Dutch name in a \
Dutch script is fine. Only respell names from OTHER languages.
5. For repeated names, use the respelling consistently throughout.
6. Preserve the EXACT line format: Speaker: [emotion] text. \
Do not change dialogue content, emotions, structure, or meaning — ONLY \
fix pronunciation of foreign terms.
7. When uncertain, prefer the most widely-used English approximation.
8. Common patterns that need respelling: Chinese names (tonal languages), \
Arabic/Hebrew names, Eastern European names, African names, South/Southeast \
Asian names, German compound words, French silent letters.

OUTPUT: Return the full script with pronunciation fixes applied. \
If no fixes are needed, return the script unchanged. No commentary."""


def pass_pronunciation(client: anthropic.Anthropic, model: str, script: str,
                       lang: str = "en") -> str:
    """Pass 3.5: Replace foreign proper nouns with phonetic respellings for TTS."""
    lang_map = {"nl": "Dutch", "de": "German", "fr": "French", "es": "Spanish",
                "en": "English"}
    lang_name = lang_map.get(lang, lang)

    user_msg = (
        f"The script language is {lang_name}. "
        f"Fix pronunciation of foreign names (from other languages) for TTS.\n\n"
        f"SCRIPT:\n{script}"
    )
    result = call_llm(client, model, PRONUNCIATION_SYSTEM, user_msg, max_tokens=16384)
    # Strip markdown fences if model wraps output
    result = result.strip()
    if result.startswith("```"):
        result = result.split("\n", 1)[1]
    if result.endswith("```"):
        result = result.rsplit("```", 1)[0]
    return result.strip()


# ---------------------------------------------------------------------------
# Pass 4: Review (three parallel perspectives)
# ---------------------------------------------------------------------------

REVIEW_FIDELITY_SYSTEM = """\
You are a source fidelity reviewer for podcast scripts. Compare the script \
against the original source material and flag factual issues.

For each issue, report as a JSON object:
{
  "line": "first few words of the line",
  "type": "DISTORTION | INVENTION | OMISSION",
  "severity": "HIGH | MEDIUM | LOW",
  "explanation": "what's wrong and what the source actually says"
}

Return a JSON object:
{
  "issues": [...],
  "faithful": ["brief note on what the script gets right", ...]
}

Only flag simplifications that lose important meaning. Dialogue naturally \
condenses — that's fine. Flag meaning changes, invented claims, and \
important nuance that was dropped."""

REVIEW_LISTENER_SYSTEM = """\
You are reviewing a podcast script from the perspective of the target listener.

Flag issues that affect the listening experience:
- ATTENTION_DRIFT: where would the listener zone out, and why
- JARGON: unexplained terms for this audience
- PACING: sections that drag or rush
- PREACHY: where it feels like a lecture instead of a conversation
- TELLING_NOT_SHOWING: where a concept is explained before being grounded in \
a scene, anecdote, or recognizable moment the listener can feel
- SHAREABILITY: moments that would make them share the episode

For each issue, report as a JSON object:
{
  "type": "ATTENTION_DRIFT | JARGON | PACING | PREACHY | TELLING_NOT_SHOWING | SHAREABILITY",
  "line": "first few words of the line",
  "note": "what's wrong / what works and suggested fix"
}

Return a JSON object:
{
  "issues": [...],
  "strengths": ["what works well for this audience", ...],
  "verdict": "one-line overall assessment"
}"""

REVIEW_NARRATIVE_SYSTEM = """\
You are reviewing a podcast script against a narrative design checklist.

Score each item as PASS, PARTIAL, or FAIL with specific evidence.

CHECKLIST:
1. Sounds like a real conversation when read aloud
2. Turn lengths vary (single-word reactions mixed with monologues)
3. At least 2-3 interruptions or incomplete thoughts per segment
4. Host pushes back or disagrees at least once
5. Host asks questions the listener would ask
6. Expert gives concrete examples, not just abstractions
7. Modern relevance feels organic, not bolted on
8. 2-3 genuine emotional moments
9. Each segment has a clear emotional arc
10. Specific, actionable references (books, search terms)
11. Ping-Pong-Plus pattern (Statement → Question → Deepening → Realization)
12. Natural speech markers (self-corrections, thinking aloud, 2-3 per segment)
13. Each major concept is grounded in a scene or anecdote BEFORE being explained
14. At least one insight per episode is reached by the host, not the expert
15. Characters bring personal stories, not just analytical positions
16. Opening hooks show the phenomenon, not describe it

ALSO CHECK:
- Speaking time ratio: Host ~35%, Expert ~60%, Third voice ~5%
- Episode structure follows: Opening → Context → Deep dive → Pivot → Application → Closing
- Show-don't-tell ratio: are concepts experienced before labeled?

Return a JSON object:
{
  "scores": [
    {"item": "checklist item name", "score": "PASS|PARTIAL|FAIL", "evidence": "..."},
    ...
  ],
  "speaking_time": {"speaker1": "X%", ...},
  "structure_notes": "how well it follows the episode structure",
  "top_improvements": ["improvement 1", "improvement 2", "improvement 3"],
  "overall_score": "X/12"
}"""


def pass_review(client: anthropic.Anthropic, model: str, script: str,
                source_text: str, listener_desc: str = "") -> dict:
    """Pass 4: Run three review perspectives. Returns combined feedback dict."""
    results = {}

    # Fidelity review
    fidelity_msg = (
        f"Compare this script against the source material.\n\n"
        f"SOURCE MATERIAL:\n{source_text[:50000]}\n\n"
        f"SCRIPT:\n{script}"
    )
    fidelity_raw = call_llm(client, model, REVIEW_FIDELITY_SYSTEM, fidelity_msg)
    results["fidelity"] = _parse_json_response(fidelity_raw)

    # Listener review
    listener_system = REVIEW_LISTENER_SYSTEM
    if listener_desc:
        listener_system += f"\n\nTARGET LISTENER: {listener_desc}"
    listener_msg = f"Review this script:\n\n{script}"
    listener_raw = call_llm(client, model, listener_system, listener_msg)
    results["listener"] = _parse_json_response(listener_raw)

    # Narrative design review
    narrative_msg = f"Review this script:\n\n{script}"
    narrative_raw = call_llm(client, model, REVIEW_NARRATIVE_SYSTEM, narrative_msg)
    results["narrative"] = _parse_json_response(narrative_raw)

    return results


def format_review_summary(review: dict) -> str:
    """Format review results as a human-readable summary."""
    lines = []

    # Fidelity
    fidelity = review.get("fidelity", {})
    issues = fidelity.get("issues", [])
    high = [i for i in issues if i.get("severity") == "HIGH"]
    med = [i for i in issues if i.get("severity") == "MEDIUM"]
    lines.append(f"SOURCE FIDELITY: {len(high)} high, {len(med)} medium, "
                 f"{len(issues) - len(high) - len(med)} low issues")
    for i in high + med:
        lines.append(f"  [{i.get('severity')}] {i.get('type')}: {i.get('explanation', '')[:100]}")

    # Listener
    listener = review.get("listener", {})
    lines.append(f"\nLISTENER: {listener.get('verdict', 'no verdict')}")
    for i in listener.get("issues", [])[:5]:
        lines.append(f"  [{i.get('type')}] {i.get('note', '')[:100]}")

    # Narrative
    narrative = review.get("narrative", {})
    lines.append(f"\nNARRATIVE DESIGN: {narrative.get('overall_score', '?')}")
    fails = [s for s in narrative.get("scores", []) if s.get("score") == "FAIL"]
    partials = [s for s in narrative.get("scores", []) if s.get("score") == "PARTIAL"]
    for s in fails:
        lines.append(f"  FAIL: {s.get('item')} — {s.get('evidence', '')[:80]}")
    for s in partials[:3]:
        lines.append(f"  PARTIAL: {s.get('item')} — {s.get('evidence', '')[:80]}")
    for imp in narrative.get("top_improvements", []):
        lines.append(f"  -> {imp}")

    return "\n".join(lines)


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text, "parse_error": True}


# ---------------------------------------------------------------------------
# Pass 5: Revise (incorporate review feedback)
# ---------------------------------------------------------------------------

REVISE_SYSTEM = """\
You are revising a podcast script based on review feedback from three \
reviewers: a source fidelity checker, a target listener advocate, and a \
narrative design expert.

Fix the issues flagged by reviewers. Prioritize:
1. HIGH severity fidelity issues (factual errors) — must fix
2. FAIL items on narrative checklist — should fix
3. Listener attention drift / preachy sections — should fix
4. MEDIUM severity and PARTIAL items — fix if natural

DO NOT:
- Rewrite sections that reviewers didn't flag
- Change the topic, speakers, or overall structure
- Add content not supported by the source material
- Pad the script significantly (stay within ~10% of original word count)

Output ONLY the revised script. Same format: "Speaker: [emotion] text". \
No commentary."""


def pass_revise(client: anthropic.Anthropic, model: str, script: str,
                review: dict) -> str:
    """Pass 5: Revise script based on review feedback."""
    summary = format_review_summary(review)
    user_msg = (
        f"Revise this script based on the review feedback below.\n\n"
        f"REVIEW FEEDBACK:\n{summary}\n\n"
        f"SCRIPT:\n{script}"
    )
    return call_llm(client, model, REVISE_SYSTEM, user_msg, max_tokens=16384)


# ---------------------------------------------------------------------------
# Format validation
# ---------------------------------------------------------------------------

LINE_PATTERN = re.compile(
    r'^[A-Za-z_]+:\s*\[[\w\s]+\]\s*.+$'
)


# ---------------------------------------------------------------------------
# Pass: Segmentation (identifies long lines, generates TTS overrides)
# ---------------------------------------------------------------------------

SEGMENT_SYSTEM = """\
You are a TTS performance director. You receive a long dialogue line that \
needs to be split into segments with controlled pauses for natural delivery.

A TTS engine generating this line in one shot will rush through it. Your job: \
split it into segments with authored pause durations that create natural \
breathing, thinking, and dramatic pacing.

Rules:
- Each segment should be a natural phrase or clause (5-20 words)
- pause_after: 0.15-0.2 for comma-level micro-pauses
- pause_after: 0.25-0.35 for thought transitions
- pause_after: 0.4-0.6 for dramatic beats, emotional shifts, or trailing off
- The last segment should have pause_after: 0.0
- Preserve the EXACT text — do not rephrase, only split

Return valid JSON — an array of segment objects:
[
  {"text": "First part of the line...", "pause_after": 0.3},
  {"text": "second part continues here.", "pause_after": 0.0}
]

Output ONLY the JSON array. No commentary."""


def pass_segment(client, model, line_text):
    """Split a single long line into segments with pause durations.

    Returns list of segment dicts or None if segmentation fails.
    """
    user_msg = f"Split this dialogue line into segments:\n\n{line_text}"
    try:
        response = call_llm(client, model, SEGMENT_SYSTEM, user_msg, max_tokens=2048)
        # Parse JSON from response (may be wrapped in markdown fences)
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        segments = json.loads(text)
        if isinstance(segments, list) and all("text" in s for s in segments):
            return segments
    except (json.JSONDecodeError, IndexError, KeyError):
        pass
    return None


def generate_overrides(client, model, script, min_words=35):
    """Scan a script for long lines and generate TTS overrides.

    Returns an overrides dict: {"overrides": {"015": [...], ...}}
    """
    try:
        from elevenlabs.src.voice_settings import parse_line
    except ImportError:
        sys.exit("--segment requires elevenlabs module. Run from the repo root or check sys.path.")

    overrides = {}
    line_index = 0
    long_count = 0

    for raw in script.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        speaker, emotion, text = parse_line(stripped)
        if speaker and text:
            word_count = len(text.split())
            if word_count >= min_words:
                long_count += 1
                segments = pass_segment(client, model, text)
                if segments:
                    key = f"{line_index:03d}"
                    overrides[key] = segments
            line_index += 1

    return {"overrides": overrides}, long_count


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
    p.add_argument("--no-pronunciation", action="store_true",
                    help="Skip the pronunciation pass (phonetic respellings for foreign names)")
    p.add_argument("--review", action="store_true",
                    help="Run review + revise passes after director (adds 3 LLM calls)")
    p.add_argument("--listener",
                    help="Target listener description for review (e.g. '28yo new grad, gets handed business books')")
    p.add_argument("--review-only",
                    help="Review an existing script file (skip generation, just review)")
    p.add_argument("--segment", action="store_true",
                    help="Generate TTS overrides for long lines (>35 words)")
    p.add_argument("--segment-min-words", type=int, default=35,
                    help="Minimum word count to trigger segmentation (default: 35)")
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

    # --- Review-only mode ---
    if args.review_only:
        script_path = Path(args.review_only)
        if not script_path.exists():
            sys.exit(f"Script not found: {args.review_only}")
        script = script_path.read_text(encoding="utf-8")
        source_text = ingest_sources(args.sources)
        print(f"Reviewing {script_path.name} against {len(args.sources)} source(s)...")
        review = pass_review(client, args.model, script, source_text, args.listener or "")
        print(format_review_summary(review))
        review_path = script_path.with_suffix(".review.json")
        review_path.write_text(json.dumps(review, indent=2), encoding="utf-8")
        print(f"\nReview saved to: {review_path}")
        return

    # --- Ingest sources ---
    print(f"Reading {len(args.sources)} source(s)...")
    source_text = ingest_sources(args.sources)
    print(f"  {len(source_text):,} characters ingested")

    # Truncate very long sources to avoid blowing context
    max_chars = 100_000
    if len(source_text) > max_chars:
        print(f"  Truncating to {max_chars:,} characters")
        source_text = source_text[:max_chars]

    # Determine total passes
    total_passes = 4  # extract + draft + director + pronunciation
    if args.no_director:
        total_passes -= 1
    if args.no_pronunciation:
        total_passes -= 1
    if args.review:
        total_passes += 2  # review + potential revise
    pass_num = 0

    # --- Pass 1: Extract ---
    pass_num += 1
    print(f"\nPass {pass_num}/{total_passes}: Extracting key facts and narrative hooks...")
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
    pass_num += 1
    print(f"\nPass {pass_num}/{total_passes}: Drafting {args.length}-minute script for {', '.join(cast)}...")
    draft = pass_draft(client, args.model, brief, cast, args.lang, args.length)
    draft_lines = [l for l in draft.splitlines() if l.strip()]
    print(f"  {len(draft_lines)} dialogue lines, ~{len(draft.split())} words")

    # --- Pass 3: Director ---
    if args.no_director:
        final = draft
        print("\nSkipping director pass (--no-director)")
    else:
        pass_num += 1
        print(f"\nPass {pass_num}/{total_passes}: Dialogue director polish...")
        final = pass_director(client, args.model, draft)
        final_lines = [l for l in final.splitlines() if l.strip()]
        print(f"  {len(final_lines)} dialogue lines, ~{len(final.split())} words")

    # --- Pass 3.5: Pronunciation ---
    if args.no_pronunciation:
        print("\nSkipping pronunciation pass (--no-pronunciation)")
    else:
        pass_num += 1
        print(f"\nPass {pass_num}/{total_passes}: Pronunciation — phonetic respellings for foreign names...")
        final = pass_pronunciation(client, args.model, final, args.lang)
        final_lines = [l for l in final.splitlines() if l.strip()]
        print(f"  {len(final_lines)} dialogue lines, ~{len(final.split())} words")

    # --- Pass 4+5: Review & Revise ---
    if args.review:
        pass_num += 1
        print(f"\nPass {pass_num}/{total_passes}: Reviewing script (fidelity + listener + narrative)...")
        review = pass_review(client, args.model, final, source_text, args.listener or "")
        print(format_review_summary(review))

        # Write review JSON alongside script
        review_path = Path(args.output or "script").with_suffix(".review.json")
        review_path.write_text(json.dumps(review, indent=2), encoding="utf-8")
        print(f"\n  Review saved to: {review_path}")

        # Check if revision is warranted
        fidelity_issues = review.get("fidelity", {}).get("issues", [])
        high_issues = [i for i in fidelity_issues if i.get("severity") == "HIGH"]
        narrative_scores = review.get("narrative", {}).get("scores", [])
        fails = [s for s in narrative_scores if s.get("score") == "FAIL"]

        if high_issues or fails:
            pass_num += 1
            print(f"\nPass {pass_num}/{total_passes}: Revising script ({len(high_issues)} high-severity + {len(fails)} fails)...")
            final = pass_revise(client, args.model, final, review)
            final_lines = [l for l in final.splitlines() if l.strip()]
            print(f"  {len(final_lines)} dialogue lines, ~{len(final.split())} words")
        else:
            print("\n  No high-severity issues or fails — skipping revision pass.")

    # --- Validate ---
    warnings = validate_script(final)
    if warnings:
        print(f"\nFormat warnings ({len(warnings)}):")
        for w in warnings[:10]:
            print(f"  {w}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")

    # --- Determine output path (needed by segmentation) ---
    output_path = args.output or f"script_{datetime.now():%Y%m%d_%H%M%S}.txt"

    # --- Segmentation ---
    if args.segment:
        print(f"\nSegmentation: scanning for lines >{args.segment_min_words} words...")
        overrides_data, long_count = generate_overrides(
            client, args.model, final, min_words=args.segment_min_words,
        )
        n_overrides = len(overrides_data.get("overrides", {}))
        print(f"  {long_count} long lines found, {n_overrides} segmented")
        if n_overrides > 0:
            overrides_path = Path(output_path).with_suffix(".overrides.json")
            overrides_path.write_text(
                json.dumps(overrides_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  Overrides written to: {overrides_path}")

    # --- Write output ---
    Path(output_path).write_text(final, encoding="utf-8")
    print(f"\nScript written to: {output_path}")
    print(f"Ready for: python generator/elevenlabs/generate_episode.py {output_path}")


if __name__ == "__main__":
    main()
