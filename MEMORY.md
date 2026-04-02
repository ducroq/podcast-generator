# Memory

## Topic Files

| File | When to load | Key insight |
|------|-------------|-------------|
| `memory/gotcha-log.md` | Stuck or debugging | Problem-fix archive — Qwen hallucinations, venv paths |
| `docs/RUNBOOK.md` | Running anything on gpu-server | Venv paths, model loading, disk constraints |
| `memory/backlog.md` | Planning next work | German Mondriaan redo, Qwen self-ref bootstrap |

## Current State

- Full blind A/B tests complete (2026-04-02): **Chatterbox 17.5 — Qwen 12** across 10 voices, 30 comparisons
- Per-voice engine recommendations established — use Qwen for Lisa/Zara/Sofie, Chatterbox for the rest
- Qwen hallucination issue identified (prepends extra text), mitigation: lower temperature + ASR validation
- Qwen3-TTS working on gpu-server (`qwen-tts-env`, `pip install qwen-tts`)
- TADA env removed to free disk; HF cache also cleaned (~14GB freed)
- `add_realism.py` working — tested on ai_agent_culture episode, good results
- Test samples in `voices/clone_tests/ab_{voice}_blind_v2/` (Qwen vs Chatterbox)
- Agent-ready-projects framework adopted (v1.3.4): gotcha log, memory index, runbook, /curate skill

## Recently Promoted

<!-- Gotchas promoted to always-loaded visibility -->

## Key File Paths

- `generator/add_realism.py` — automated realism post-processing (overlaps, jitter, room tone)
- `generator/elevenlabs/generate_episode.py` — ElevenLabs episode generation
- `generator/elevenlabs/generate_single_line.py` — single line repair tool
- `generator/chatterbox/generate_podcast.py` — Chatterbox podcast generation
- `generator/validate_tts.py` — hallucination detection (ASR vs expected text)
- `generator/trim_silences.py` — silence trimming (after generation, before realism)
- `voices/clone_tests/` — A/B test samples organized by engine
- `voices/clone_tests/ab_{voice}_blind/` — blind test sets with `.key.json`
- `docs/decisions/` — ADRs

## Active Decisions

- ADR-001: Dialogue writing style for local TTS
- ADR-002: Local TTS for agentic engineering (Chatterbox + Qwen over ElevenLabs for English)
- Pending: Update ADR-002 — use per-voice engine selection (Qwen for Lisa/Zara/Sofie, Chatterbox for rest)
