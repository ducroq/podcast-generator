# Narrative Design Guide

How to write podcast scripts that sound like real conversations, not lectures. Based on lessons from producing a 7-episode series with 300+ plays per episode.

## The Core Rule

Intelligent friends talking — never a presenter reading to an audience. The listener should feel like they're overhearing a fascinating conversation, not attending a class.

## Three-Voice Model

Most educational podcasts benefit from three complementary roles:

| Role | Speaking time | Function |
|------|-------------|----------|
| **Host** | ~35% | Audience proxy. Asks "naive" questions, reacts emotionally, grounds abstract ideas |
| **Expert** | ~60% | Knowledge carrier. Makes complex ideas accessible, provides examples and stories |
| **Source voice** | ~5% | Historical figure, interviewee, or narrator. Adds authenticity and emotional weight |

**Character dynamics:**
- Host's energy rises → Expert matches and elevates
- Expert gets intense → Host grounds with a practical question
- Both get excited → Source voice brings wise calm

Each character needs signature phrases. The host says "Wait, so you mean..." The expert says "What's fascinating is..." These patterns make voices recognizable even without names.

## Episode Structure (5 Segments)

```
OPENING (2-3 min)     Hook + rapport + preview of the "wow moment"
SEGMENT 1 (2-4 min)   Context. Plant a mystery or paradox.
SEGMENT 2 (3-5 min)   Deep dive. Core concept exploration.
SEGMENT 3 (1-2 min)   Emotional punctuation. Source voice, quote, or key revelation.
SEGMENT 4 (3-5 min)   Application. Concrete examples, specific references.
SEGMENT 5 (2-4 min)   Modern relevance. Why this matters now.
CLOSING (2-3 min)      Emotional recap + call to action + next episode teaser
```

Segments are deliberately unequal. The short emotional segment (3) acts as a pivot between intellectual exploration and practical application.

**Word targets:** 15-20 min = ~2500-3000 words. 20-25 min = ~3000-4000 words.

## Dialogue Dynamics

### Ping-Pong-Plus

Not just back-and-forth. Each exchange escalates:

```
Statement → Question → Deepening → Realization
```

Bad:
```
Expert: Mondriaan wrote over 100 essays.
Host: That's a lot. What were they about?
Expert: Art and philosophy.
```

Good:
```
Expert: [warm] Here's something most people don't know. Mondriaan wrote over
a hundred essays. More pages of philosophy than paintings.
Host: [surprised] Wait — a hundred? I thought he was purely a visual artist.
Expert: [building] That's exactly the misconception. He saw writing and
painting as the same project. The essays ARE his art, just in a different medium.
Host: [realizing] So when people say "it's just coloured squares"...
Expert: [passionate] They're missing half the work. Literally half.
```

Each line advances understanding. The host's realization teaches the listener.

### Natural Speech

Use thinking sounds and self-corrections sparingly — 2-3 per segment maximum:
- "Well, actually..." (self-correction)
- "Hmm, how do I explain this..." (thinking aloud)
- "No wait, let me put that differently..." (authenticity)

Too many and it sounds scripted-trying-to-sound-unscripted. Too few and it sounds robotic.

### Conversational Texture

Real conversations aren't tidy. Scripts that feel alive use these patterns:

**Interruptions** — one speaker cuts in mid-sentence. Use an em-dash to mark the break:
```
Expert: [building] And what he realized was that the entire tradition of—
Host: [excited] Wait, you mean he rejected ALL of it?
Expert: [calm] Not rejected. Transcended.
```

**Completing each other's thoughts** — one trails off, the other finishes:
```
Host: [thoughtful] So if you take that logic further, then...
Expert: [completing] Then the painting isn't a picture anymore. It's a system.
```

**Backchanneling** — short reactive lines between longer turns. These break the rhythm of lecture-length monologues:
```
Expert: [explaining] He spent three years just painting trees.
Host: [surprised] Three years?
Expert: [warm] Three years. And each one got more abstract than the last.
```

**Pushback and disagreement** — not every exchange should escalate toward agreement. The host should challenge, and the expert should occasionally concede:
```
Host: [skeptical] But isn't that just... intellectualizing what's basically decoration?
Expert: [thoughtful] That's a fair challenge. And honestly, some art historians agree with you.
Expert: [building] But here's what changed my mind—
```

**Vary turn length dramatically.** A common trap is every turn being 1-3 sentences. Mix single-word reactions with 30-second monologues. The contrast creates rhythm:
```
Expert: [passionate] He believed — genuinely believed — that if you could find the
right balance between horizontal and vertical, between red and blue and yellow, you
weren't just making a painting. You were modelling the structure of reality itself.
That everything in the universe, from a jazz melody to a city grid to the way light
falls on water, follows the same underlying pattern of opposition and harmony. And
he spent forty years trying to prove it.
Host: [quiet] ...Forty years.
Expert: [warm] Forty years.
```

**Incomplete thoughts** — people don't speak in finished paragraphs. Let speakers lose their thread, restart, or change direction mid-sentence:
```
Expert: [thinking] The thing about Paris is... well, it wasn't just the art. It was—
how do I put this — it was the first time he was around people who thought the way he did.
```

These techniques work with `text_to_dialogue` as written — no post-production needed. The API handles pacing and intonation from the conversational context. For actual overlapping audio (two voices speaking simultaneously), you'd need to generate separately and layer in a DAW.

## Emotional Progressions

Plan emotional arcs per segment, not just content arcs.

**Discovery arc:** curious → interested → fascinated → impressed
**Confusion-to-clarity:** confused → hesitant → probing → realizing → excited
**Expert teaching:** calm → building → enthusiastic → passionate

Tag each line with the intended emotion. This drives both the writing and the TTS delivery:
```
Host: [curious] So what happened when he moved to Paris?
Expert: [building] Everything changed. He saw Cubism and thought...
Expert: [excited] ...this is what I've been looking for my entire life.
Host: [fascinated] You can hear that in his writing?
```

## Emotional Authenticity Moments

The best episodes have 2-3 moments where the conversation gets genuinely human:

- **Vulnerability:** The expert admits uncertainty. "Honestly, scholars still argue about this."
- **Shared wonder:** Both hosts are surprised by the same fact at once.
- **Personal connection:** "This changes how I think about..."
- **Empathy for the subject:** "Imagine knowing you're right but nobody understands."

These moments can't be faked. Write them from real emotional engagement with the material.

## Source Quotes / Historical Voice

When integrating quotes or a historical voice:

1. **Context** — Expert explains what's coming and why it matters
2. **Pause** — Let the listener prepare
3. **Quote** — Source voice delivers
4. **Pause** — Let it land
5. **Reaction** — Hosts process and connect it back to the theme

Never drop a quote without setup and payoff. The frame around the quote does as much work as the quote itself.

## Series Continuity

For multi-episode series, plan three layers:

**Episode-level:** Each episode is self-contained with its own arc and payoff.

**Series-level:** Plant seeds early that pay off later. Episode 1 mentions a concept in passing that becomes the focus of Episode 5. Reward listeners who follow the whole series.

**Character growth:** The host starts as a curious outsider and gradually becomes knowledgeable. By the final episode, they can articulate the core ideas themselves. This mirrors the listener's own journey.

## Connecting to the Present

Every historical or abstract topic needs a bridge to contemporary life. The pattern:

1. Identify the historical figure's concern
2. Name its modern equivalent
3. Let it emerge from dialogue, not as a lecture point

"Search for universality" → diversity and inclusion debates.
"Art as social change" → contemporary activist movements.
"Statements without words" → visual culture, emoji, data visualization.

The connection should feel like a natural realization, not a forced lesson.

## Practical References

When discussing specific works, places, or sources — make them findable:
- Name the museum, city, and room if possible
- Give a search term the listener can use
- Mention a book by title and author

Listeners who can follow up become loyal listeners.

## Quality Checklist

A script is ready when:
- [ ] Sounds like a real conversation when read aloud
- [ ] Turn lengths vary — mix single-word reactions with longer monologues
- [ ] At least 2-3 interruptions or incomplete thoughts per segment
- [ ] Host pushes back or disagrees at least once per episode
- [ ] Host asks questions the listener would ask
- [ ] Expert gives concrete examples, not just abstractions
- [ ] Source quotes integrate naturally with setup and payoff
- [ ] Modern relevance feels organic, not bolted on
- [ ] 2-3 genuine emotional moments per episode
- [ ] Each segment has a clear emotional arc
- [ ] Specific, actionable references (museums, books, search terms)
- [ ] Works standalone but rewards series listening
