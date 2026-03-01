#!/usr/bin/env python3
"""
Chatterbox TTS Multi-Speaker Podcast Generator
Generates podcast audio with different voices per speaker.
Includes mastering step for balanced voice levels.
"""

import argparse
import re
import sys
import random
import subprocess
from pathlib import Path

import torch
import torchaudio
from chatterbox.tts import ChatterboxTTS

# Voice configuration: speaker -> reference audio path (None = default voice)
VOICE_REFS = {
    'ember': Path.home() / 'voice_refs' / 'ember.mp3',
    'emma': Path.home() / 'voice_refs' / 'emma.mp3',
    'lisa': Path.home() / 'voice_refs' / 'lisa.mp3',
    'marc': Path.home() / 'voice_refs' / 'narrator.mp3',
    'narrator': Path.home() / 'voice_refs' / 'narrator.mp3',
    'sven': Path.home() / 'voice_refs' / 'sven.mp3',
    'victoria': Path.home() / 'voice_refs' / 'victoria.mp3',
}

def parse_line(line):
    """Parse a dialogue line: 'Speaker: [emotion] Text'"""
    match = re.match(r'(\w+):\s*\[(\w+(?:\s+\w+)*)\]\s*(.*)', line.strip())
    if match:
        return match.group(1).lower(), match.group(2).lower(), match.group(3).strip()
    return None, None, None

def master_audio(input_path, output_path):
    """
    Apply mastering: compression + normalization.
    Balances voice levels and normalizes overall loudness.
    """
    print('Mastering audio...')
    
    # Compression (balances loud/quiet parts) + loudnorm (overall loudness)
    # - acompressor: threshold=-18dB, ratio=3:1, fast attack, medium release
    # - loudnorm: target -16 LUFS (podcast standard), true peak -1.5dB
    filter_chain = (
        'acompressor=threshold=-18dB:ratio=3:attack=5:release=100,'
        'loudnorm=I=-16:TP=-1.5:LRA=11'
    )
    
    cmd = [
        'ffmpeg', '-y', '-i', str(input_path),
        '-af', filter_chain,
        '-codec:a', 'libmp3lame', '-b:a', '128k',
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'Warning: mastering failed: {result.stderr}')
        return False
    return True

def generate_podcast(script_path, output_path, test_lines=None, skip_master=False):
    """Generate podcast from script with multi-speaker support"""
    print(f'Loading script: {script_path}')
    with open(script_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parse dialogue lines
    dialogues = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('[') or line.startswith('='):
            continue
        speaker, emotion, text = parse_line(line)
        if speaker and text:
            dialogues.append((speaker, emotion, text))
    
    if test_lines:
        dialogues = dialogues[:test_lines]
    
    print(f'Processing {len(dialogues)} dialogue lines')
    
    # Load model
    print('Loading Chatterbox model...')
    model = ChatterboxTTS.from_pretrained(device='cuda')
    sr = model.sr
    
    # Generate audio for each line
    audio_chunks = []
    current_speaker = None
    
    for i, (speaker, emotion, text) in enumerate(dialogues):
        # Get voice reference for this speaker
        voice_ref = VOICE_REFS.get(speaker)
        ref_path = str(voice_ref) if voice_ref and voice_ref.exists() else None
        ref_label = f'(ref: {voice_ref.name})' if ref_path else '(default voice)'
        
        print(f'  [{i+1}/{len(dialogues)}] {speaker} {ref_label}: {text[:40]}...')
        
        # Generate speech with or without voice reference
        if ref_path:
            wav = model.generate(text, audio_prompt_path=ref_path)
        else:
            wav = model.generate(text)
        
        audio_chunks.append(wav)
        
        # Add pause between lines (longer between different speakers)
        if speaker != current_speaker:
            pause_duration = random.uniform(0.15, 0.45)  # Speaker change
        else:
            pause_duration = random.uniform(0.10, 0.30)  # Same speaker continues
        pause = torch.zeros(1, int(sr * pause_duration))
        audio_chunks.append(pause)
        current_speaker = speaker
    
    # Concatenate all audio
    print('Concatenating audio...')
    full_audio = torch.cat(audio_chunks, dim=1)
    
    # Determine output paths
    output_path = Path(output_path)
    if output_path.suffix.lower() == '.mp3':
        wav_path = output_path.with_suffix('.wav')
        mp3_path = output_path
    else:
        wav_path = output_path
        mp3_path = output_path.with_suffix('.mp3')
    
    # Save wav
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(wav_path), full_audio, sr)
    
    duration = full_audio.shape[1] / sr
    print(f'Raw audio: {wav_path}')
    print(f'Duration: {duration/60:.1f} minutes ({duration:.0f}s)')
    
    # Master and convert to MP3
    if not skip_master:
        if master_audio(wav_path, mp3_path):
            size_mb = mp3_path.stat().st_size / (1024 * 1024)
            print(f'Mastered: {mp3_path} ({size_mb:.1f} MB)')
        else:
            print('Mastering failed, wav file available')
    else:
        print('Skipping mastering (--no-master flag)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('script', help='Path to dialogue script')
    parser.add_argument('-o', '--output', default='output.mp3', help='Output path (.mp3 or .wav)')
    parser.add_argument('--test', type=int, help='Only process first N lines')
    parser.add_argument('--no-master', action='store_true', help='Skip mastering step')
    args = parser.parse_args()
    
    generate_podcast(args.script, args.output, args.test, args.no_master)
