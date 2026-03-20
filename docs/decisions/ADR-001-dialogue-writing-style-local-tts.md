# ADR-001: Dialogue Writing Style for Local TTS Engines

**Date:** 2026-03-18
**Status:** Accepted

## Context

Podcast production is moving from ElevenLabs to local TTS engines running on gpu-server: Chatterbox, Qwen3-TTS, and TADA. This changes what the script can rely on.

ElevenLabs' `text_to_dialogue` API understands conversation context, emotional tags (`[curious]`, `[excited]`), and speaker transitions. Local models do not. They process each line (or turn) independently with no emotional direction beyond the words themselves and the voice reference audio.

This means:

- **Emotion tags are dead weight.** `[warm]` does nothing in Chatterbox or Qwen3-TTS. Keeping them in scripts creates false confidence that delivery is being shaped.
- **The writing must carry the entire emotional load.** Word choice, sentence length, punctuation, and turn structure are the only tools for controlling how speech sounds.
- **LLM-generated scripts default to patterns that sound worse in TTS.** Long compound sentences, em dash chains, hedging phrases, and presentation verbs all flatten into monotone delivery when a local model reads them.

This ADR adapts the wire service writing discipline (see ovr.news ADR-026) for spoken dialogue targeting local TTS engines.

## Decision

All podcast scripts targeting local TTS engines follow the rules below. Scripts targeting ElevenLabs may still use emotion tags and `text_to_dialogue` conventions.

### Sentence and turn structure

1. **One idea per turn.** If a character makes a point and then asks a question, split it into two turns or let another character react in between.
2. **Two to four sentences per turn, maximum.** Monologues longer than four sentences flatten in TTS. If a character needs to explain something complex, break it with an interruption, a question, or a reaction from another speaker.
3. **Vary turn length dramatically.** Mix single-word reactions ("Wait."), short questions ("Since when?"), and 3-sentence explanations. Uniform turn length creates metronome pacing.
4. **First sentence of each turn does the work.** TTS models deliver the first sentence with the most natural energy. Front-load the point.

### Word choice

5. **Use contractions.** "We'll" not "we will." "That's" not "that is." "Doesn't" not "does not." Uncontracted speech sounds robotic in TTS.
6. **Specific over abstract.** "50 lines" not "lightweight." "4th time this week" not "repeatedly." "12 countries" not "multiple nations."
7. **Common words.** "Start" not "commence." "Use" not "utilize." "Show" not "demonstrate."
8. **Spell out numbers for TTS.** "Nineteen twenty-two" not "1922." "Forty-two percent" not "42%."
9. **No adjectives unless they are facts.** "First" is fact. "Groundbreaking" is editorializing.

### Emotion through structure, not tags

10. **Short sentences signal urgency or surprise.** "Wait. That's it? Fifty lines?" reads as disbelief without any tag.
11. **Longer sentences signal calm explanation.** "So what happens is the agent loads the project file at the start of every session, and everything in there is visible without any extra action." This reads as measured, explanatory.
12. **Incomplete thoughts signal excitement or interruption.** Use em dashes for cut-offs: "And then the whole architecture just—" / "Gone. Completely."
13. **Questions carry curiosity.** Don't write "he asked curiously." Write a question that is genuinely curious. "But if it only auto-loads fifty lines, what happens to the rest?"
14. **Repetition carries emphasis.** "Not the article. Not a summary of the article. The actual news." Three short beats land harder than one long sentence.

### Punctuation

15. **Em dashes only for interruptions and cut-offs.** Never as parenthetical connectors (that is the LLM tic ADR-026 identifies). One em dash per scene maximum for interruption. Zero is fine.
16. **Ellipses for trailing off.** "I mean, I thought it would be more..." — use sparingly, once or twice per episode.
17. **No semicolons.** Period or new turn.
18. **No parenthetical asides.** If it matters, give it its own sentence or turn.

### The never list (LLM anti-patterns)

19. **Never use presentation verbs.** "Highlights," "showcases," "underscores," "demonstrates" — these describe an article, not a conversation. Characters report, explain, react. They do not "highlight."
20. **Never hedge.** "It's worth noting that," "it's important to recognize," "interestingly enough" — cut all of these. Say the thing.
21. **Never editorialize with empty labels.** "Comprehensive," "innovative," "significant," "groundbreaking" — if it's impressive, show why with a specific fact.
22. **Never use rhetorical questions as transitions.** "But what does this mean for developers?" — this is a lecture pattern. Have a character genuinely ask or react instead.
23. **Never have a character summarize what another character just said.** "So what you're saying is..." — the listener already heard it. Move forward.
24. **Never open with definitions.** "AI agents are software programs that..." — open with an experience, a problem, or a reaction. Define through use, not declaration.

### Natural speech patterns

25. **Backchanneling between long turns.** If Marc explains for three sentences, Lisa can interject "Three years?" or "Huh." before he continues. This prevents wall-of-text delivery.
26. **Genuine disagreement.** Characters push back, not just nod. "I don't buy that" is more natural than "That's a great point, and building on that..."
27. **Reactions before responses.** People react emotionally before they respond intellectually. "Wait, what?" then "How is that even possible?" — not the reverse.
28. **Restarts and corrections.** "The thing is — no, actually, let me put it differently." This reads as thinking, not scripted. Aim for two to four per episode. Types that work well:
    - **Self-count corrections:** "Three sessions — no, wait. Four sessions." A character adjusts a number mid-thought.
    - **Mid-sentence restarts:** "That's exactly — yes. That's the right way to think about it." The character abandons one phrasing for a better one.
    - **Hedged recall:** "I think it's nine now." Uncertainty about a detail signals real memory, not a script.
    - **Word search:** "Hard to catch precisely because the prose is so — how do I put this — fluent." The pause signals genuine thought.
    - **Honest admission:** "I've seen — actually, I did this myself." Correcting a distancing "I've seen" to the more vulnerable first person.
    - **Scope correction:** "We do this for — well, we try to do it for contractors." The correction carries meaning: we attempt it but fail.
    - **Never overdo it.** More than four per episode sounds like the characters can't think straight. These should feel like moments of honesty, not a speech impediment.

### Overlap markers for DAW mixing

29. **Mark two to four overlap points per episode.** Local TTS generates one line at a time, producing separate audio files per turn. This gives full control over timing in the DAW. Use overlap to create the illusion of real conversation at key moments.

30. **Use `# [OVERLAP]` comments as stage directions.** These are stripped before TTS generation but visible in the script as a mixing guide. Each marker describes the specific timing for the DAW operator.

    Format:
    ```
    Sven: Fourteen words.
    # [OVERLAP] Marc echoes "Fourteen words" immediately, overlapping the tail of Sven's line

    Marc: Fourteen words that would have saved your Thursday night.
    ```

31. **Overlap works best for:**
    - **Fast reactions that cut in on the last one to two words.** "The audit log?" starting before the previous speaker finishes saying "audit log." This signals surprise or alarm.
    - **Echoes and confirmations.** One speaker repeating a phrase from the previous turn, overlapping its tail. Creates emphasis.
    - **Quick single-word responses.** "Eight." landing on the last syllable of a question. Signals certainty.
    - **Enthusiastic agreement.** "Someone has to do it" starting over the tail of a joke. Signals rapport.

32. **Overlap does not work for:**
    - **Long sentences over long sentences.** Becomes noise. Both lines are unintelligible.
    - **More than four overlaps per episode.** The effect loses impact and starts sounding like a production gimmick.
    - **Content-heavy lines.** If the listener needs to hear every word, don't overlap it. Overlap is for reactions and emphasis, not information.
    - **Back-to-back overlaps.** Space them across the episode. Two overlaps in thirty seconds sounds like an argument, not a conversation.

33. **Describe the timing in the marker.** Don't just write `# [OVERLAP]`. Specify which words overlap and how: "Marc starts 'It sounds like' over the tail of 'employee'" gives the DAW operator a clear instruction. Vague markers like "these lines overlap" will produce inconsistent results.

## Engine-specific notes

### Chatterbox
- Best English quality among local models. Proven in production.
- Marc uses `narrator.mp3` (Chatterbox native voice) instead of `marc.mp3` — the original had clarity issues with years ("2018" sounding like "2008").
- No emotion control. Voice character comes entirely from reference audio.

### Qwen3-TTS
- English and German support. Excellent quality.
- Voice cloning requires `ref_text` to match `ref_audio` exactly. Mismatched text causes runaway generation (model never stops).
- Use bootstrapped self-references from `qwen_bootstrap_refs.py` for reliable cloning.

### TADA (HumeAI, untested)
- 10 languages including English and German. No Dutch.
- 1:1 text-to-speech token alignment — claims zero hallucination (no swallowed or repeated words).
- Dynamic duration: model decides word timing naturally rather than fixed frame rate.
- Voice cloning from reference audio. No emotion tags.
- Based on Llama 3.2-3B (4B total parameters). Requires GPU.
- Needs evaluation before production use.

## Consequences

### Positive

- Scripts sound human when read by any TTS engine, not just the one they were tuned for.
- Portable across engines: same script works for Chatterbox, Qwen3-TTS, TADA, or future models.
- Wire discipline eliminates the most common LLM writing tics, making AI-generated scripts harder to identify as AI-generated.
- Rules are concrete enough to put directly into LLM script-generation prompts.

### Negative

- Writing is harder. Emotion tags were a shortcut. Now every line must earn its tone through word choice and structure.
- Scripts may need more revision passes. A flat-sounding line cannot be fixed by changing `[neutral]` to `[excited]`.
- Loss of fine-grained delivery control that ElevenLabs provided (whispering, sighing, specific pacing).

### Risks

- Over-constraining the script prompt may make LLM-generated dialogue feel stilted. The rules should guide, not straitjacket. Natural-sounding exceptions are fine.
- Some rules (turn length, sentence count) are guidelines, not hard limits. A five-sentence turn that flows well beats a two-sentence turn that feels chopped.

## References

- ovr.news ADR-026: Wire Service Writing Style for Summaries — the source of the "never list" and wire discipline principles
- podcast-generator `docs/NARRATIVE_DESIGN.md` — three-voice model, ping-pong-plus dynamics, episode structure
- podcast-generator `docs/PRODUCTION_GUIDE.md` — ElevenLabs emotion settings, Qwen3-TTS constraints, mastering pipeline
- AP Stylebook — sentence structure and word choice standards
- George Orwell, "Politics and the English Language" (1946) — "Never use a long word where a short one will do"
