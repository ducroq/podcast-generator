#!/usr/bin/env python3
"""
Add realism to generated podcast audio: overlapping speech, filler sounds,
timing jitter, and room tone. Works on any TTS engine output.

Pipeline position: after trim_silences, before final mastering.

Usage:
    python generator/add_realism.py input.mp3 [output.mp3] [options]

Options:
    --overlap-chance 0.15    Probability of overlapping a speaker turn (0-1)
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
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path


def detect_silences(input_path, noise_db=-35, min_duration=0.25):
    """Detect silence regions using ffmpeg silencedetect."""
    cmd = [
        'ffmpeg', '-i', str(input_path),
        '-af', f'silencedetect=noise={noise_db}dB:d={min_duration}',
        '-f', 'null', '-'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    import re
    silences = []
    starts = []
    for line in result.stderr.split('\n'):
        start_match = re.search(r'silence_start:\s*([\d.]+)', line)
        end_match = re.search(r'silence_end:\s*([\d.]+)\s*\|\s*silence_duration:\s*([\d.]+)', line)
        if start_match:
            starts.append(float(start_match.group(1)))
        if end_match and starts:
            silences.append({
                'start': starts.pop(0),
                'end': float(end_match.group(1)),
                'duration': float(end_match.group(2))
            })
    return silences


def get_duration(input_path):
    """Get audio duration in seconds."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(input_path)
    ]
    return float(subprocess.run(cmd, capture_output=True, text=True).stdout.strip())


def get_sample_rate(input_path):
    """Get audio sample rate."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=sample_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(input_path)
    ]
    return int(subprocess.run(cmd, capture_output=True, text=True).stdout.strip())


def split_into_turns(silences, total_duration):
    """Split audio into speech turns based on silence boundaries.

    Returns list of turns: [{start, end, duration, gap_after}]
    where gap_after is the silence duration following this turn.
    """
    turns = []
    current_pos = 0.0

    for silence in silences:
        if silence['start'] > current_pos + 0.05:
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


def plan_realism(turns, overlap_chance, overlap_range_ms, jitter_range_ms,
                 filler_chance, fillers_available, dry_run=False):
    """Decide which realism effects to apply to each turn gap.

    Returns list of actions per turn:
    [{turn_idx, action, overlap_ms, jitter_ms, filler_file}]
    """
    actions = []

    for i, turn in enumerate(turns):
        action = {
            'turn_idx': i,
            'action': 'normal',
            'overlap_ms': 0,
            'jitter_ms': 0,
            'filler_file': None,
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

            # Filler: insert "uh", "mmhm" etc. under long turns
            if turn['duration'] > 3.0 and fillers_available and random.random() < filler_chance:
                action['filler_file'] = random.choice(fillers_available)

        actions.append(action)

    return actions


def build_filter_complex(turns, actions, total_duration, sample_rate,
                         room_tone_path=None, room_tone_vol=0.08):
    """Build ffmpeg filter_complex string for all realism effects.

    Strategy: extract each turn as a segment, adjust timing between segments,
    then concatenate. Overlaps are done by mixing the tail of one segment with
    the head of the next.
    """
    filters = []
    segments = []
    input_idx = 0  # [0:a] is always the main audio
    extra_inputs = []

    for i, (turn, action) in enumerate(zip(turns, actions)):
        seg_label = f's{i}'

        # Extract this turn's audio
        filters.append(
            f'[0:a]atrim={turn["start"]:.6f}:{turn["end"]:.6f},'
            f'asetpts=N/SR/TB[{seg_label}]'
        )

        if i < len(turns) - 1:
            gap = turn['gap_after']

            if action['action'] == 'overlap':
                # Overlap: shorten the gap and crossfade
                overlap_s = action['overlap_ms'] / 1000.0
                remaining_gap = max(0.05, gap - overlap_s)
                # Create shortened silence
                pad_label = f'pad{i}'
                filters.append(
                    f'aevalsrc=0:d={remaining_gap:.6f}:s={sample_rate}[{pad_label}]'
                )
                segments.append(f'[{seg_label}]')
                segments.append(f'[{pad_label}]')
            elif action['jitter_ms'] != 0:
                # Jitter: adjust gap duration
                new_gap = max(0.05, gap + action['jitter_ms'] / 1000.0)
                pad_label = f'pad{i}'
                filters.append(
                    f'aevalsrc=0:d={new_gap:.6f}:s={sample_rate}[{pad_label}]'
                )
                segments.append(f'[{seg_label}]')
                segments.append(f'[{pad_label}]')
            else:
                # Normal: keep original gap
                if gap > 0.01:
                    pad_label = f'pad{i}'
                    filters.append(
                        f'aevalsrc=0:d={gap:.6f}:s={sample_rate}[{pad_label}]'
                    )
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

    # Room tone: mix underneath
    if room_tone_path:
        # Loop room tone to match duration, mix at low volume
        filters.append(
            f'[room]aloop=loop=-1:size=2e9,atrim=duration={total_duration:.6f},'
            f'volume={room_tone_vol}[roomvol]'
        )
        filters.append(
            f'[{out_label}][roomvol]amix=inputs=2:weights=1 {room_tone_vol}:'
            f'duration=first[roomed]'
        )
        out_label = 'roomed'
        extra_inputs = ['-i', str(room_tone_path)]
    else:
        # Synthetic pink noise room tone
        filters.append(
            f'anoisesrc=color=pink:sample_rate={sample_rate}:'
            f'amplitude=0.002:duration={total_duration:.6f}[roomnoise]'
        )
        filters.append(
            f'[{out_label}][roomnoise]amix=inputs=2:weights=1 0.12:'
            f'duration=first[roomed]'
        )
        out_label = 'roomed'

    return filters, out_label, extra_inputs


def add_realism(input_path, output_path, overlap_chance=0.15,
                overlap_range_ms=(300, 800), jitter_range_ms=(50, 150),
                filler_chance=0.10, fillers_dir=None,
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

    # Plan effects
    actions = plan_realism(
        turns, overlap_chance, overlap_range_ms, jitter_range_ms,
        filler_chance, fillers, dry_run
    )

    # Stats
    overlaps = sum(1 for a in actions if a['action'] == 'overlap')
    jitters = sum(1 for a in actions if a['jitter_ms'] != 0)
    filler_inserts = sum(1 for a in actions if a['filler_file'])

    print(f'Planned effects:')
    print(f'  Overlaps: {overlaps}/{len(turns)} turns')
    print(f'  Jittered pauses: {jitters}/{len(turns)} turns')
    print(f'  Filler insertions: {filler_inserts}')
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
            print(f'  Turn {i}: {turn["start"]:.2f}-{turn["end"]:.2f}s '
                  f'({turn["duration"]:.2f}s) gap={turn["gap_after"]:.3f}s '
                  f'-> {effect}{extra}')
        return True

    # Build and run ffmpeg
    use_room = not no_room_tone
    filters, out_label, extra_inputs = build_filter_complex(
        turns, actions, total_duration, sample_rate,
        room_tone_path if room_tone_path else None if not use_room else None,
        room_tone_vol
    )

    # If no_room_tone, rebuild without room tone
    if no_room_tone:
        # Remove the last two filter lines (room tone generation and mixing)
        # and use 'joined' as output
        filters_no_room = []
        for f in filters:
            if 'anoisesrc' in f or 'roomnoise' in f or 'roomed' in f:
                continue
            filters_no_room.append(f)
        filters = filters_no_room
        out_label = 'joined'

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
    """Parse 'min-max' into (min, max) tuple of ints."""
    parts = s.split('-')
    if len(parts) == 2:
        return (int(parts[0]), int(parts[1]))
    val = int(parts[0])
    return (val, val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add realism to generated podcast audio'
    )
    parser.add_argument('input', help='Input audio file (mp3/wav)')
    parser.add_argument('output', nargs='?', help='Output file (default: input_real.mp3)')
    parser.add_argument('--overlap-chance', type=float, default=0.15,
                        help='Probability of overlapping a turn (default: 0.15)')
    parser.add_argument('--overlap-ms', type=str, default='300-800',
                        help='Overlap duration range in ms (default: 300-800)')
    parser.add_argument('--jitter-ms', type=str, default='50-150',
                        help='Pause jitter range in ms (default: 50-150)')
    parser.add_argument('--fillers-dir', type=str, default=None,
                        help='Directory with filler audio files')
    parser.add_argument('--filler-chance', type=float, default=0.10,
                        help='Probability of filler during long turns (default: 0.10)')
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

    add_realism(
        input_path, output_path,
        overlap_chance=args.overlap_chance,
        overlap_range_ms=parse_range(args.overlap_ms),
        jitter_range_ms=parse_range(args.jitter_ms),
        filler_chance=args.filler_chance,
        fillers_dir=args.fillers_dir,
        room_tone_path=args.room_tone,
        room_tone_vol=args.room_tone_vol,
        no_room_tone=args.no_room_tone,
        seed=args.seed,
        dry_run=args.dry_run,
    )
