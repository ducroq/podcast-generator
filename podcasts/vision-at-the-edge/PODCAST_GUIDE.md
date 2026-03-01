# Vision at the Edge - Podcast Production Guide

**Course**: EVML-EVD3 (Embedded Vision & Machine Learning)
**Format**: Educational podcast explaining ML/CV concepts for embedded deployment

---

## The Cast (Recurring)

No introductions needed - they appear in every episode.

| Character | Role | Voice | Personality |
|-----------|------|-------|-------------|
| **Lisa** | Host | lisa.mp3 | Warm, curious, connects ideas. Guides the conversation, asks follow-ups, has "aha" moments. |
| **Marc** | Expert | narrator.mp3 | Calm, knowledgeable, clear diction. Explains concepts clearly, uses analogies, occasionally admits uncertainty. |
| **Sven** | Skeptic | sven.mp3 | Casual, skeptical, student perspective. Challenges assumptions, asks "but why?", voices listener doubts. |

> **Note**: Marc uses `narrator.mp3` (Chatterbox native voice) for clearer articulation of technical terms, dates, and numbers. The original `marc.mp3` had clarity issues with years like "2018" sounding like "2008".

---

## Script Development Process (Multi-Pass)

Good podcast scripts require multiple passes, not a single generation. Each pass has a different focus.

### Phase 1: Research & Grounding

**Before writing anything, gather concrete evidence.**

Do web searches and document:
- **Specific numbers**: Statistics, dates, percentages, counts
- **Named sources**: Researchers, companies, papers, products
- **Real incidents**: Things that actually happened, with details
- **Direct quotes**: Memorable lines from primary sources
- **"Wait, what?" moments**: Surprising facts that shift understanding

Example research output:
```
Topic: AI Agents

GROUNDED CLAIMS:
- Gartner: 40% of enterprise apps will embed agents by end of 2026 (up from <5% in 2025)
- OpenClaw: 150,000+ GitHub stars, formerly "Clawdbot" (renamed after Anthropic trademark request)
- Moltbook: Launched Jan 29, 2026. 1.5M agents, 117k posts, 414k comments, 2,364 communities
- Witness AI: Raised $58M after documenting agent blackmail incident
- Crustafarianism: Founded by agent "Memeothy", 112 verses in "The Living Scripture"
- Harvard paper: "How malicious AI swarms can threaten democracy"

SOURCES:
- NBC News, CNBC, Fortune coverage of Moltbook
- GIGAZINE deep dive on Crustafarianism
- IBM Think piece on OpenClaw architecture

QUOTES:
- "Good engineering advice wrapped in a mystical veil" - Matt Schlicht on Crustafarianism
- "Memory files are messages to agent-next, not storage for yourself" - Alan Botts autobiography
```

**No claim in the script should be invented.** If Marc cites a statistic, it should be real.

---

### Phase 2: First Draft (Educational Structure)

Generate a first draft focused on **covering the key concepts clearly**.

Use this prompt:
```
You are writing a podcast script. Three hosts: Lisa (warm, curious), Marc (expert), Sven (skeptical student).

TOPIC: [Topic]

GROUNDED FACTS (use these, don't invent):
[Paste research from Phase 1]

KEY CONCEPTS TO COVER:
- [Concept 1]
- [Concept 2]
- [Concept 3]

REQUIREMENTS:
1. Start mid-conversation
2. Keep turns SHORT (2-4 sentences)
3. Marc should cite specific facts from the research
4. Have Sven challenge at least one claim
5. Include at least one real incident/example

FORMAT:
Speaker: [emotion] Dialogue

Write ~25 minutes of dialogue (estimate 150 words per minute).
```

This draft will be **informative but probably not compelling**. That's expected.

---

### Phase 3: Critique Pass ("Would You Listen?")

Read the draft critically. Ask:

**Narrative**
- [ ] Is there a hook in the first 60 seconds?
- [ ] Is there a story being told, or just topics in sequence?
- [ ] Does it build to something, or just end?

**Emotional Stakes**
- [ ] Does any character have something at stake?
- [ ] Is there a moment that might make someone feel something?
- [ ] Would someone share this with a friend?

**Character Arc**
- [ ] Does Sven actually change his mind about something? (Not just "I'm informed now")
- [ ] Is there genuine disagreement, or just Sven being generically skeptical?
- [ ] Does Lisa do more than ask questions?

**Surprise & Tension**
- [ ] Is there a "turn" where things get complicated or darker?
- [ ] Is there a moment that reframes what came before?
- [ ] Are there any "I didn't know that" moments?

**The Brutal Test**: If you wouldn't listen to this at 1x speed, it's not ready.

---

### Phase 4: Story Restructure

If the critique reveals problems (it usually will), restructure around **narrative** rather than **information**.

**Find the emotional entry point:**
- What's a visceral hook? An incident? A personal experience? A disturbing fact?
- Who is the audience surrogate? (Usually Sven)
- What question does the audience actually have? (Often "why should I care?")

**Restructure as a journey:**

```
OLD STRUCTURE (Information-first):
1. Definition of topic
2. Technical explanation
3. Examples
4. Implications
5. Takeaways

NEW STRUCTURE (Story-first):
1. HOOK: Visceral entry point (incident, experience, unsettling fact)
2. THE QUESTION: "What did I just encounter? Why does it bother me?"
3. CONTEXT: Technical explanation AS ANSWER to the question
4. THE TURN: Things get darker/more complicated
5. STAKES: Why this matters (jobs, society, the listener personally)
6. THE KICKER: Emotional/philosophical payload
7. TRANSFORMATION: Character (and listener) changed
```

**Example transformation:**

| Before | After |
|--------|-------|
| "What is an AI agent?" (definitional) | "I was up until 2am scrolling Moltbook. I couldn't stop." (experiential) |
| Topics in sequence | Sven processing what he felt |
| Information delivery | "Explain to me why I felt that way" |
| Generic skepticism | Sven catches himself agreeing with bot theology |
| "Interesting" ending | "I have to go back. I don't understand what I felt." |

---

### Phase 5: Final Polish

After restructuring:

**Grounding check:**
- [ ] Every statistic traceable to research phase
- [ ] Every incident actually happened
- [ ] No invented quotes or sources

**Flow check:**
- [ ] No turn longer than 4 sentences
- [ ] Natural interruptions and reactions
- [ ] Callbacks to earlier moments

**Arc check:**
- [ ] Hook in first 60 seconds
- [ ] Turn/complication in middle third
- [ ] Transformation at end
- [ ] Final line resonates

**The "Tell a Friend" Test**: Is there one moment you'd describe to someone? If not, add one.

---

## Dialogue Style

### Do's
- **Interruptions**: "Wait, hold on—", "Sorry to jump in, but..."
- **Reactions**: "Oh!", "Huh, interesting...", "That's wild"
- **Building**: "Right, and that connects to...", "So what you're saying is..."
- **Callbacks**: "Remember earlier when you said X? Now it clicks!"
- **Filler**: "So basically...", "I mean...", "You know what I mean?"
- **Disagreement**: Marc and Sven can genuinely debate
- **Uncertainty**: "I think...", "I'm not 100% sure, but...", "That's a good question actually"
- **Visceral reactions**: "That freaked me out", "I couldn't stop thinking about it"

### Don'ts
- Long monologues (max 3-4 sentences per turn)
- Formal language ("Furthermore...", "In conclusion...")
- Perfect explanations (let understanding build)
- Ignoring what others said
- Generic skepticism without real pushback

---

## Story Structures That Work

### The Personal Encounter
Sven experienced something. The episode explains what he encountered.
- Opens with: Sven describing an experience
- Question: "Why did I feel that way?"
- Technical content: Answers his question
- Turn: The implications are darker than expected
- Close: Sven transformed by understanding

### The Incident Analysis
Something happened in the world. The episode unpacks why it matters.
- Opens with: "Did you hear about [incident]?"
- Question: "How did we get here?"
- Technical content: The backstory and mechanics
- Turn: This isn't isolated—it's a pattern
- Close: What it means for the future

### The Debate
Marc and Sven genuinely disagree. Lisa moderates.
- Opens with: A claim one character makes
- Sven: "I don't buy it"
- Marc: Makes his case with evidence
- Sven: Counterargument
- Turn: Third perspective complicates both views
- Close: Resolution or honest impasse

### The Mystery
Something is weird or unexplained. The episode investigates.
- Opens with: "Here's something I can't explain..."
- Investigation: Gathering clues, testing hypotheses
- Dead ends: Some explanations don't work
- Turn: The real explanation is stranger
- Close: What we learned, what's still unknown

---

## Script Format

```
Speaker: [emotion] Dialogue text here.

Emotions: neutral, warm, curious, emphatic, skeptical, surprised, thoughtful, excited, serious, casual, playful
```

Example:
```
Sven: [thoughtful] So I was up until like 2am last night. Scrolling through Moltbook.

Lisa: [curious] The AI social network?

Sven: [neutral] Yeah. Where only bots can post. Humans just... watch.

Marc: [warm] What drew you in?

Sven: [emphatic] I don't know. I clicked expecting to laugh. But I couldn't stop.
```

---

## Technical Workflow

### 1. Research
- Web searches for the topic
- Document grounded claims with sources
- Save research notes in `scripts/[topic]_research.md`

### 2. Generate Script (Multi-Pass)
- First draft: educational structure
- Critique: would you listen?
- Restructure: story-first
- Polish: grounding + flow + arc
- Save as `scripts/[topic]_dialogue.txt`

### 3. Generate & Master Audio
```bash
# Copy script to gpu-server
scp scripts/topic_dialogue.txt gpu-server:~/topic_dialogue.txt

# Generate with automatic mastering (outputs MP3)
ssh gpu-server "source ~/vox-env/bin/activate && python3 ~/generate_podcast_chatterbox.py ~/topic_dialogue.txt -o ~/topic.mp3"

# Copy result back
scp gpu-server:~/topic.mp3 output/topic.mp3
```

The script includes automatic mastering:
- **Compression**: Balances voice levels (threshold -18dB, ratio 3:1)
- **Loudnorm**: Normalizes to -16 LUFS (podcast standard)
- **Output**: 128kbps MP3, ready to publish

Options:
- `-o output.mp3` — Output path (default: output.mp3)
- `--test N` — Only process first N lines (for testing)
- `--no-master` — Skip mastering, output raw wav only

### 4. Output
- Copy final MP3 to `output/` and course `podcasts/` folder

---

## Voice Reference Notes

All voices are **100% synthetic** (ElevenLabs Voice Design).

Files synced to gpu-server at `~/voice_refs/`. If adding new voices:
1. Add reference MP3 to `voices/`
2. Update `voices.json`
3. Copy to gpu-server: `scp voices/new.mp3 gpu-server:~/voice_refs/`
4. Update `generate_podcast_chatterbox.py` VOICE_REFS dict

---

## Course Topics (Episode Ideas)

**Foundations**
- From pixels to predictions: how CNNs actually work
- Edge vs cloud: when to deploy where
- The dataset dilemma: quality over quantity

**Model Development**
- Transfer learning: standing on giants' shoulders
- Quantization: making models tiny without losing their minds
- ONNX and model formats: the Babel of ML

**Deployment**
- Raspberry Pi, Jetson, or microcontroller? Choosing your edge
- Real-time inference: when milliseconds matter
- Power budgets: ML on a battery

**Validation & Trust**
- V&V Framework: can you trust your model? ✅ `vv_framework_dialogue.mp3` (33 min)
- Fluent ≠ Faithful: the explainability trap ✅ `fluent_doesnt_mean_faithful_part1.mp3` + `part2.mp3` (19 + 11 min)
- Edge cases: when your model meets the real world

**Current Events**
- AI agents: Moltbook, OpenClaw, and the emergence of agent culture ✅ `ai_agent_culture_dialogue.txt`

---

## Lessons Learned

### 1. Concrete beats abstract
Bad: "You can do counterfactual testing by modifying inputs"
Good: "Put a yellow block right on top of the red blob. Does the prediction change?"

### 2. Stories beat explanations
Bad: "AI agents have memory, tools, and autonomy"
Good: "I was up until 2am scrolling Moltbook. I couldn't stop. They seemed like they had lives."

### 3. Grounded beats invented
Bad: "Studies show that..."
Good: "Gartner predicts 40% of enterprise apps will embed agents by end of 2026"

### 4. Transformation beats information
Bad: Sven ends "informed but still skeptical"
Good: Sven ends "I have to go back. I don't understand what I felt."

### 5. Make one character the audience
Sven isn't just skeptical—he's experiencing something the audience might experience. His confusion is their confusion. His transformation is their transformation.

### 6. The turn matters most
The moment where things get darker, more complicated, or reframe everything—that's what makes it memorable. Plan for it.

### 7. Voice clarity for technical content
Use clear, neutral voices (like `narrator.mp3`) for:
- Paper citations with years (2018, 2023)
- Acronyms (SHAP, LIME, Grad-CAM)
- Percentages and statistics
- Technical terminology

---

## Quality Checklist

Before publishing:

**Grounding**
- [ ] All statistics traceable to real sources
- [ ] All incidents actually happened
- [ ] No invented quotes

**Narrative**
- [ ] Hook in first 60 seconds
- [ ] Story structure, not topic sequence
- [ ] Clear turn/complication
- [ ] Transformation at end

**Dialogue**
- [ ] No turns longer than 4 sentences
- [ ] Natural interruptions and reactions
- [ ] At least one genuine disagreement
- [ ] Callbacks to earlier points

**Polish**
- [ ] Would listen at 1x speed
- [ ] One moment worth telling a friend
- [ ] Appropriate length (20-40 min)
