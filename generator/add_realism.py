#!/usr/bin/env python3
"""
Add realism to generated podcast audio: overlapping speech, filler sounds,
breath insertion, timing jitter, and room tone. Works on any TTS engine output.

Pipeline position: after trim_silences, before final mastering.

Usage:
    python generator/add_realism.py input.mp3 [output.mp3] [options]

Options:
    --overlap-chance 0.25    Probability of overlapping a speaker turn (0-1)
    --overlap-ms 300-800     Overlap duration range in ms
    --jitter-ms 50-150       Random pause jitter range in ms
    --fillers-dir path       Directory with filler audio files (uh.wav, mmhm.wav, etc.)
    --filler-chance 0.10     Probability of inserting a filler during a long turn
    --room-tone path         Room tone / ambience audio file to loop underneath
    --room-tone-vol 0.08     Room tone mix volume (0-1)
    --seed 42                Random seed for reproducibility
    --no-room-tone           Skip synthetic room tone generation
    --dry-run                Print what would happen without processing
"""

import argparse
import random
import subprocess
import sys
from pathlib import Path

import re

from audio_utils import detect_silences, get_duration, get_sample_rate


def split_into_turns(silences, total_duration):
    """Split audio into speech turns based on silence boundaries.

    Returns list of turns: [{start, end, duration, gap_after}]
    where gap_after is the silence duration following this turn.
    """
    turns = []
    current_pos = 0.0

    for silence in silences:
        if silence['start'] >= current_pos + 0.01:
            turns.append({
                'start': current_pos,
                'end': silence['start'],
                'duration': silence['start'] - current_pos,
                'gap_after': silence['duration'],
            })
        current_pos = silence['end']

    # Final turn after last silence
    if current_pos < total_duration - 0.05:
        turns.append({
            'start': current_pos,
            'end': total_duration,
            'duration': total_duration - current_pos,
            'gap_after': 0.0,
        })

    return turns


# ---------------------------------------------------------------------------
# Content-aware backchannel placement
# ---------------------------------------------------------------------------

BACKCHANNEL_TYPES = {
    'agreement': ['mmhm', 'right', 'yeah'],
    'surprise': ['oh', 'huh', 'wow'],
    'tracking': ['right', 'mmhm', 'okay'],
}

# Emotions that suggest the listener would react with surprise/interest
SURPRISE_EMOTIONS = {'surprised', 'excited', 'passionate', 'emphatic', 'fascinated'}
# Emotions that suggest agreement/tracking backchannels
AGREEMENT_EMOTIONS = {'warm', 'calm', 'thoughtful', 'explaining', 'building'}


def parse_script_lines(script_path):
    """Parse a dialogue script into a list of (speaker, emotion, text) tuples.

    Only returns dialogue lines (skips blanks and section headers).
    """
    lines = []
    line_pattern = re.compile(r'^([A-Za-z_]+):\s*\[(\w[\w\s]*)\]\s*(.+)$')
    with open(script_path, 'r', encoding='utf-8') as f:
        for raw in f:
            raw = raw.strip()
            m = line_pattern.match(raw)
            if m:
                lines.append((m.group(1), m.group(2).lower(), m.group(3)))
    return lines


def select_backchannel(line_text, line_emotion):
    """Select a backchannel type based on the content of the preceding line.

    Returns a backchannel category ('agreement', 'surprise', 'tracking')
    or None if no backchannel is warranted.
    """
    # After a question → agreement/tracking
    if line_text.rstrip().endswith('?'):
        return 'agreement'

    # After a surprising or intense emotion → surprise
    if line_emotion in SURPRISE_EMOTIONS:
        return 'surprise'

    # After a long statement (3+ sentences) → tracking
    sentence_count = len(re.findall(r'[.!?]+', line_text))
    if sentence_count >= 3:
        return 'tracking'

    # After agreement-type emotions on substantial turns
    if line_emotion in AGREEMENT_EMOTIONS and len(line_text.split()) > 15:
        return 'agreement'

    return None


def plan_realism(turns, overlap_chance, overlap_range_ms, jitter_range_ms,
                 filler_chance, fillers_available, breath_chance=0.3,
                 breaths_available=None, script_lines=None, dry_run=False):
    """Decide which realism effects to apply to each turn gap.

    Returns list of actions per turn:
    [{turn_idx, action, overlap_ms, jitter_ms, filler_file, breath}]
    """
    actions = []

    for i, turn in enumerate(turns):
        action = {
            'turn_idx': i,
            'action': 'normal',
            'overlap_ms': 0,
            'jitter_ms': 0,
            'filler_file': None,
            'breath': None,  # 'inhale', 'exhale', or a file path
        }

        if i < len(turns) - 1:  # Not the last turn
            gap = turn['gap_after']

            # Overlap: pull next turn earlier so voices briefly collide
            # Only on gaps > 0.2s to avoid mangling tight transitions
            if gap > 0.2 and random.random() < overlap_chance:
                max_overlap = min(
                    overlap_range_ms[1],
                    int(gap * 1000 * 0.8)  # Don't overlap more than 80% of gap
                )
                min_overlap = min(overlap_range_ms[0], max_overlap)
                if min_overlap < max_overlap:
                    action['action'] = 'overlap'
                    action['overlap_ms'] = random.randint(min_overlap, max_overlap)

            # Jitter: vary pause timing so it's not metronomic
            if action['action'] == 'normal' and gap > 0.1:
                jitter = random.randint(-jitter_range_ms[1], jitter_range_ms[1])
                # Don't jitter into negative gap
                if gap * 1000 + jitter > 50:
                    action['jitter_ms'] = jitter

            # Filler / backchannel: insert "uh", "mmhm" etc.
            # Content-aware: if script lines available, select type based on content
            if turn['duration'] > 3.0 and random.random() < filler_chance:
                if script_lines and i < len(script_lines):
                    _, emotion, text = script_lines[i]
                    bc_type = select_backchannel(text, emotion)
                    if bc_type:
                        action['backchannel_type'] = bc_type
                        action['backchannel_word'] = random.choice(BACKCHANNEL_TYPES[bc_type])
                if fillers_available:
                    action['filler_file'] = random.choice(fillers_available)

            # Breath: insert between speaker turns
            # More likely before long turns (>2s) and at gaps >0.3s
            next_turn = turns[i + 1] if i + 1 < len(turns) else None
            if (gap > 0.3 and action['action'] != 'overlap'
                    and random.random() < breath_chance):
                if breaths_available:
                    action['breath'] = random.choice(breaths_available)
                elif next_turn and next_turn['duration'] > 2.0:
                    action['breath'] = 'inhale'
                else:
                    action['breath'] = 'exhale'

        actions.append(action)

    return actions


def _silence_pad(sample_rate, duration, label):
    """Generate a silence pad filter as two properly separated filters.

    anullsrc is a source filter and cannot be comma-chained with atrim.
    Returns a list of filter strings to be joined with ';'.
    """
    null_label = f'null_{label}'
    return [
        f'anullsrc=r={sample_rate}:cl=mono[{null_label}]',
        f'[{null_label}]atrim=duration={duration:.6f}[{label}]',
    ]


def _breath_filter(sample_rate, breath_type, label):
    """Generate a synthetic breath filter (inhale or exhale).

    Uses band-passed pink noise with an amplitude envelope to approximate
    the spectral and temporal shape of a human breath.

    Inhale: 0.4-0.6s, rising then falling envelope, 800-3500 Hz band
    Exhale: 0.3-0.5s, sharp attack then decay, 600-2500 Hz band
    """
    if breath_type == 'inhale':
        duration = random.uniform(0.4, 0.6)
        # Inhale: gradual rise, peaks at ~60%, then drops
        freq_lo, freq_hi = 800, 3500
        volume = random.uniform(0.015, 0.025)
    else:  # exhale
        duration = random.uniform(0.3, 0.5)
        # Exhale: sharper attack, lower frequency
        freq_lo, freq_hi = 600, 2500
        volume = random.uniform(0.010, 0.020)

    noise_label = f'bnoise_{label}'
    return [
        f'anoisesrc=color=pink:sample_rate={sample_rate}:'
        f'amplitude={volume}:duration={duration:.3f}[{noise_label}]',
        f'[{noise_label}]bandpass=frequency={int((freq_lo + freq_hi) / 2)}:'
        f'width_type=h:width={freq_hi - freq_lo},'
        f'afade=t=in:d={duration * 0.3:.3f},'
        f'afade=t=out:st={duration * 0.6:.3f}:d={duration * 0.4:.3f}[{label}]',
    ]


def build_filter_complex(turns, actions, total_duration, sample_rate,
                         room_tone_path=None, room_tone_vol=0.08,
                         no_room_tone=False):
    """Build ffmpeg filter_complex string for all realism effects.

    Strategy: extract each turn as a segment, adjust timing between segments,
    then concatenate. Overlaps are done by mixing the tail of one segment with
    the head of the next. Fillers are mixed in after concatenation at their
    planned positions within the timeline.
    """
    filters = []
    segments = []
    extra_inputs = []
    # Track next ffmpeg input index: [0]=main audio, then room tone if present, then fillers
    next_input_idx = 1
    room_tone_input_idx = None
    if room_tone_path and not no_room_tone:
        room_tone_input_idx = next_input_idx
        extra_inputs.extend(['-i', str(room_tone_path)])
        next_input_idx += 1

    # Force input to mono to avoid channel mismatch with generated silence pads
    # and noise sources (Chatterbox can output stereo, ElevenLabs outputs mono)
    filters.append('[0:a]aformat=channel_layouts=mono[inmono]')

    for i, (turn, action) in enumerate(zip(turns, actions)):
        seg_label = f's{i}'

        # Extract this turn's audio
        filters.append(
            f'[inmono]atrim={turn["start"]:.6f}:{turn["end"]:.6f},'
            f'asetpts=N/SR/TB[{seg_label}]'
        )

        if i < len(turns) - 1:
            gap = turn['gap_after']

            if action['action'] == 'overlap':
                # Overlap: shorten the gap and crossfade
                overlap_s = action['overlap_ms'] / 1000.0
                remaining_gap = max(0.05, gap - overlap_s)
                pad_label = f'pad{i}'
                filters.extend(_silence_pad(sample_rate, remaining_gap, pad_label))
                segments.append(f'[{seg_label}]')
                segments.append(f'[{pad_label}]')
            elif action['jitter_ms'] != 0:
                # Jitter: adjust gap duration
                new_gap = max(0.05, gap + action['jitter_ms'] / 1000.0)
                pad_label = f'pad{i}'
                filters.extend(_silence_pad(sample_rate, new_gap, pad_label))
                segments.append(f'[{seg_label}]')
                segments.append(f'[{pad_label}]')
            else:
                # Normal: keep original gap
                if gap > 0.01:
                    pad_label = f'pad{i}'
                    filters.extend(_silence_pad(sample_rate, gap, pad_label))
                    segments.append(f'[{seg_label}]')
                    segments.append(f'[{pad_label}]')
                else:
                    segments.append(f'[{seg_label}]')
        else:
            segments.append(f'[{seg_label}]')

    # Concatenate all segments
    n_segments = len(segments)
    concat_str = ''.join(segments) + f'concat=n={n_segments}:v=0:a=1[joined]'
    filters.append(concat_str)

    out_label = 'joined'

    # Breaths: insert synthetic breath sounds between turns
    # Breaths are placed just before the next turn in the gap
    breath_segments = []
    for i, action in enumerate(actions):
        breath = action.get('breath')
        if breath and isinstance(breath, str) and breath in ('inhale', 'exhale'):
            breath_label = f'breath{i}'
            filters.extend(_breath_filter(sample_rate, action['breath'], breath_label))
            breath_segments.append((i, breath_label))

    # If we have breaths, mix them at their gap positions after concatenation
    # We'll mix them after fillers (both use the same post-concat mixing approach)

    # Fillers: mix each filler at its planned absolute position
    # Compute absolute timeline positions by replaying overlap/jitter decisions
    timeline_pos = 0.0
    turn_abs_starts = []
    for i, (turn, action) in enumerate(zip(turns, actions)):
        turn_abs_starts.append(timeline_pos)
        timeline_pos += turn['duration']
        if i < len(turns) - 1:
            gap = turn['gap_after']
            if action['action'] == 'overlap':
                overlap_s = action['overlap_ms'] / 1000.0
                timeline_pos += max(0.05, gap - overlap_s)
            elif action['jitter_ms'] != 0:
                timeline_pos += max(0.05, gap + action['jitter_ms'] / 1000.0)
            else:
                if gap > 0.01:
                    timeline_pos += gap

    for i, (turn, action) in enumerate(zip(turns, actions)):
        if action['filler_file']:
            filler_idx = next_input_idx
            next_input_idx += 1
            extra_inputs.extend(['-i', action['filler_file']])
            # Place filler at a random point within the turn (30%-70% through)
            turn_start = turn_abs_starts[i]
            offset_in_turn = turn['duration'] * random.uniform(0.3, 0.7)
            delay_ms = int((turn_start + offset_in_turn) * 1000)
            filler_label = f'filler{i}'
            mix_label = f'fmix{i}'
            filters.append(
                f'[{filler_idx}:a]aformat=channel_layouts=mono,'
                f'volume=0.3,adelay={delay_ms}|{delay_ms}[{filler_label}]'
            )
            filters.append(
                f'[{out_label}][{filler_label}]amix=inputs=2:duration=first[{mix_label}]'
            )
            out_label = mix_label

    # Mix breaths at their gap positions (just before the next turn starts)
    for turn_idx, breath_label in breath_segments:
        # Place breath at end of gap, just before next turn
        if turn_idx + 1 < len(turn_abs_starts):
            next_start = turn_abs_starts[turn_idx + 1]
            # Breath ends ~50ms before next turn starts
            delay_ms = max(0, int((next_start - 0.5) * 1000))
        else:
            turn_start = turn_abs_starts[turn_idx]
            delay_ms = int((turn_start + turns[turn_idx]['duration']) * 1000)

        bmix_label = f'bmix{turn_idx}'
        filters.append(
            f'[{breath_label}]adelay={delay_ms}|{delay_ms}[bd{turn_idx}]'
        )
        filters.append(
            f'[{out_label}][bd{turn_idx}]amix=inputs=2:duration=first[{bmix_label}]'
        )
        out_label = bmix_label

    # Room tone: mix underneath (skip if no_room_tone)
    # Use computed timeline duration (accounts for overlaps/jitter) instead of input duration
    output_duration = timeline_pos if timeline_pos > 0 else total_duration
    if not no_room_tone:
        if room_tone_input_idx is not None:
            # Loop room tone to match duration, mix at low volume
            filters.append(
                f'[{room_tone_input_idx}:a]aformat=channel_layouts=mono,'
                f'aloop=loop=-1:size=2000000000,'
                f'atrim=duration={output_duration:.6f},'
                f'volume={room_tone_vol}[roomvol]'
            )
            filters.append(
                f'[{out_label}][roomvol]amix=inputs=2:weights=1|{room_tone_vol}:'
                f'duration=first[roomed]'
            )
            out_label = 'roomed'
        else:
            # Synthetic pink noise room tone
            filters.append(
                f'anoisesrc=color=pink:sample_rate={sample_rate}:'
                f'amplitude=0.002:duration={output_duration:.6f}[roomnoise]'
            )
            filters.append(
                f'[{out_label}][roomnoise]amix=inputs=2:weights=1|0.12:'
                f'duration=first[roomed]'
            )
            out_label = 'roomed'

    return filters, out_label, extra_inputs


def add_realism(input_path, output_path, overlap_chance=0.25,
                overlap_range_ms=(300, 800), jitter_range_ms=(50, 150),
                filler_chance=0.10, fillers_dir=None,
                breath_chance=0.3, breaths_dir=None,
                script_path=None,
                room_tone_path=None, room_tone_vol=0.08,
                no_room_tone=False, seed=None, dry_run=False):
    """Main pipeline: detect turns, plan effects, render with ffmpeg."""

    if seed is not None:
        random.seed(seed)

    print(f'Analyzing: {input_path}')
    total_duration = get_duration(input_path)
    sample_rate = get_sample_rate(input_path)
    silences = detect_silences(input_path)

    print(f'Duration: {total_duration:.1f}s, silences: {len(silences)}')

    turns = split_into_turns(silences, total_duration)
    print(f'Detected {len(turns)} speech turns')

    if len(turns) < 2:
        print('Not enough turns to add realism effects.')
        return False

    # Collect filler files
    fillers = []
    if fillers_dir:
        fillers_path = Path(fillers_dir)
        fillers = sorted([
            str(f) for f in fillers_path.glob('*.wav')
        ] + [
            str(f) for f in fillers_path.glob('*.mp3')
        ])
        if fillers:
            print(f'Filler sounds: {len(fillers)} files from {fillers_dir}')

    # Collect breath files (optional — synthetic breaths used if no files provided)
    breaths = []
    if breaths_dir:
        breaths_path = Path(breaths_dir)
        breaths = sorted([
            str(f) for f in breaths_path.glob('*.wav')
        ] + [
            str(f) for f in breaths_path.glob('*.mp3')
        ])
        if breaths:
            print(f'Breath sounds: {len(breaths)} files from {breaths_dir}')

    # Parse script for content-aware backchannel placement
    script_lines = None
    if script_path:
        script_lines = parse_script_lines(script_path)
        print(f'Script: {len(script_lines)} dialogue lines from {script_path}')

    # Plan effects
    actions = plan_realism(
        turns, overlap_chance, overlap_range_ms, jitter_range_ms,
        filler_chance, fillers, breath_chance, breaths,
        script_lines=script_lines, dry_run=dry_run,
    )

    # Stats
    overlaps = sum(1 for a in actions if a['action'] == 'overlap')
    jitters = sum(1 for a in actions if a['jitter_ms'] != 0)
    filler_inserts = sum(1 for a in actions if a.get('filler_file'))
    breath_inserts = sum(1 for a in actions if a.get('breath'))
    backchannel_inserts = sum(1 for a in actions if a.get('backchannel_type'))

    print(f'Planned effects:')
    print(f'  Overlaps: {overlaps}/{len(turns)} turns')
    print(f'  Jittered pauses: {jitters}/{len(turns)} turns')
    print(f'  Filler insertions: {filler_inserts}')
    print(f'  Backchannels: {backchannel_inserts}')
    print(f'  Breath insertions: {breath_inserts}')
    print(f'  Room tone: {"custom" if room_tone_path else "synthetic pink noise" if not no_room_tone else "none"}')

    if dry_run:
        print('\n--- Dry run detail ---')
        for i, (turn, action) in enumerate(zip(turns, actions)):
            effect = action['action']
            extra = ''
            if action['overlap_ms']:
                extra = f' ({action["overlap_ms"]}ms)'
            if action['jitter_ms']:
                extra += f' jitter={action["jitter_ms"]:+d}ms'
            if action['filler_file']:
                extra += f' filler={Path(action["filler_file"]).name}'
            if action.get('breath'):
                extra += f' breath={action["breath"]}'
            if action.get('backchannel_type'):
                extra += f' backchannel={action["backchannel_word"]}({action["backchannel_type"]})'
            print(f'  Turn {i}: {turn["start"]:.2f}-{turn["end"]:.2f}s '
                  f'({turn["duration"]:.2f}s) gap={turn["gap_after"]:.3f}s '
                  f'-> {effect}{extra}')
        return True

    # Build and run ffmpeg
    filters, out_label, extra_inputs = build_filter_complex(
        turns, actions, total_duration, sample_rate,
        room_tone_path=room_tone_path,
        room_tone_vol=room_tone_vol,
        no_room_tone=no_room_tone,
    )

    filter_complex = ';'.join(filters)

    cmd = [
        'ffmpeg', '-y', '-i', str(input_path),
        *extra_inputs,
        '-filter_complex', filter_complex,
        '-map', f'[{out_label}]',
        '-codec:a', 'libmp3lame', '-b:a', '192k',
        str(output_path)
    ]

    print('Processing...')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'ERROR: ffmpeg failed')
        print(result.stderr[-800:])
        return False

    new_duration = get_duration(output_path)
    in_size = Path(input_path).stat().st_size / 1024
    out_size = Path(output_path).stat().st_size / 1024
    print(f'Result: {total_duration:.1f}s -> {new_duration:.1f}s '
          f'({in_size:.0f}KB -> {out_size:.0f}KB)')
    print(f'Done: {output_path}')
    return True


def parse_range(s):
    """Parse 'min-max' into (min, max) tuple of positive ints."""
    parts = s.split('-', 1)
    if len(parts) == 2 and parts[0] and parts[1]:
        lo, hi = int(parts[0]), int(parts[1])
        if lo < 0 or hi < 0:
            raise argparse.ArgumentTypeError(f"Range values must be positive: {s}")
        if lo > hi:
            raise argparse.ArgumentTypeError(f"Min must be <= max: {s}")
        return (lo, hi)
    val = int(s)
    if val < 0:
        raise argparse.ArgumentTypeError(f"Value must be positive: {s}")
    return (val, val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add realism to generated podcast audio'
    )
    parser.add_argument('input', help='Input audio file (mp3/wav)')
    parser.add_argument('output', nargs='?', help='Output file (default: input_real.mp3)')
    parser.add_argument('--overlap-chance', type=float, default=0.25,
                        help='Probability of overlapping a turn (default: 0.25)')
    parser.add_argument('--overlap-ms', type=str, default='300-800',
                        help='Overlap duration range in ms (default: 300-800)')
    parser.add_argument('--jitter-ms', type=str, default='50-150',
                        help='Pause jitter range in ms (default: 50-150)')
    parser.add_argument('--fillers-dir', type=str, default=None,
                        help='Directory with filler audio files')
    parser.add_argument('--filler-chance', type=float, default=0.10,
                        help='Probability of filler during long turns (default: 0.10)')
    parser.add_argument('--breath-chance', type=float, default=0.3,
                        help='Probability of breath between turns (default: 0.3)')
    parser.add_argument('--breaths-dir', type=str, default=None,
                        help='Directory with breath audio files (uses synthetic if omitted)')
    parser.add_argument('--no-breaths', action='store_true',
                        help='Skip breath insertion entirely')
    parser.add_argument('--script', type=str, default=None,
                        help='Dialogue script for content-aware backchannel placement')
    parser.add_argument('--room-tone', type=str, default=None,
                        help='Room tone audio file to loop underneath')
    parser.add_argument('--room-tone-vol', type=float, default=0.08,
                        help='Room tone volume (default: 0.08)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--no-room-tone', action='store_true',
                        help='Skip room tone entirely')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print plan without processing')

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f'Error: {input_path} not found')
        sys.exit(1)

    output_path = Path(args.output) if args.output else \
        input_path.with_stem(input_path.stem + '_real')

    ok = add_realism(
        input_path, output_path,
        overlap_chance=args.overlap_chance,
        overlap_range_ms=parse_range(args.overlap_ms),
        jitter_range_ms=parse_range(args.jitter_ms),
        filler_chance=args.filler_chance,
        fillers_dir=args.fillers_dir,
        breath_chance=0.0 if args.no_breaths else args.breath_chance,
        breaths_dir=args.breaths_dir,
        script_path=args.script,
        room_tone_path=args.room_tone,
        room_tone_vol=args.room_tone_vol,
        no_room_tone=args.no_room_tone,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    if ok is False:
        sys.exit(1)
