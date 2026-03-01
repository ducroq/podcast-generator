#!/bin/bash
# GPU Server Audio Cleanup Script
# Reorganizes ~/  audio files into a proper structure
# Does NOT touch: nexusmind-scorer, venvs, .ssh, .config, etc.

set -euo pipefail

HOME_DIR="$HOME"
cd "$HOME_DIR"

echo "=== GPU Server Audio Cleanup ==="
echo ""

# 1. Create directory structure
echo "Creating directory structure..."
mkdir -p ~/podcast-generator/{scripts,output,logs}
mkdir -p ~/podcast-generator/tts-tests
mkdir -p ~/whisper/output

# 2. Move podcast episodes (grouped by episode)
echo "Moving podcast episodes..."

# ai_agent_culture
mkdir -p ~/podcast-generator/output/ai_agent_culture
mv -v ~/ai_agent_culture.mp3          ~/podcast-generator/output/ai_agent_culture/ 2>/dev/null || true
mv -v ~/ai_agent_culture.wav          ~/podcast-generator/output/ai_agent_culture/ 2>/dev/null || true
mv -v ~/ai_agent_culture_dialogue.txt ~/podcast-generator/scripts/ai_agent_culture_dialogue.txt 2>/dev/null || true

# fluent_doesnt_mean_faithful (multiple parts/versions)
mkdir -p ~/podcast-generator/output/fluent_doesnt_mean_faithful
mv -v ~/fluent_doesnt_mean_faithful.mp3 ~/podcast-generator/output/fluent_doesnt_mean_faithful/ 2>/dev/null || true
mv -v ~/fluent_doesnt_mean_faithful.wav ~/podcast-generator/output/fluent_doesnt_mean_faithful/ 2>/dev/null || true
mv -v ~/fluent_part1.mp3               ~/podcast-generator/output/fluent_doesnt_mean_faithful/ 2>/dev/null || true
mv -v ~/fluent_part1.wav               ~/podcast-generator/output/fluent_doesnt_mean_faithful/ 2>/dev/null || true
mv -v ~/fluent_part1_v4.mp3            ~/podcast-generator/output/fluent_doesnt_mean_faithful/ 2>/dev/null || true
mv -v ~/fluent_part1_v4.wav            ~/podcast-generator/output/fluent_doesnt_mean_faithful/ 2>/dev/null || true
mv -v ~/fluent_part1_narrator.mp3      ~/podcast-generator/output/fluent_doesnt_mean_faithful/ 2>/dev/null || true
mv -v ~/fluent_part1_narrator.wav      ~/podcast-generator/output/fluent_doesnt_mean_faithful/ 2>/dev/null || true
mv -v ~/fluent_part2.mp3               ~/podcast-generator/output/fluent_doesnt_mean_faithful/ 2>/dev/null || true
mv -v ~/fluent_part2.wav               ~/podcast-generator/output/fluent_doesnt_mean_faithful/ 2>/dev/null || true
mv -v ~/fluent_part2_narrator.mp3      ~/podcast-generator/output/fluent_doesnt_mean_faithful/ 2>/dev/null || true
mv -v ~/fluent_part2_narrator.wav      ~/podcast-generator/output/fluent_doesnt_mean_faithful/ 2>/dev/null || true
mv -v ~/fluent_doesnt_mean_faithful_part1_script.txt ~/podcast-generator/scripts/ 2>/dev/null || true
mv -v ~/fluent_doesnt_mean_faithful_part2_script.txt ~/podcast-generator/scripts/ 2>/dev/null || true
mv -v ~/fluent_part1_script.txt        ~/podcast-generator/scripts/ 2>/dev/null || true
mv -v ~/fluent_part2_script.txt        ~/podcast-generator/scripts/ 2>/dev/null || true

# speckit_vibecoding
mkdir -p ~/podcast-generator/output/speckit_vibecoding
mv -v ~/speckit_vibecoding.mp3            ~/podcast-generator/output/speckit_vibecoding/ 2>/dev/null || true
mv -v ~/speckit_vibecoding.wav            ~/podcast-generator/output/speckit_vibecoding/ 2>/dev/null || true
mv -v ~/speckit_vibecoding_mastered.mp3   ~/podcast-generator/output/speckit_vibecoding/ 2>/dev/null || true
mv -v ~/speckit_vibecoding_normalized.mp3 ~/podcast-generator/output/speckit_vibecoding/ 2>/dev/null || true
mv -v ~/speckit_vibecoding_dialogue.txt   ~/podcast-generator/scripts/speckit_vibecoding_dialogue.txt 2>/dev/null || true

# speckit_vs_bmad
mkdir -p ~/podcast-generator/output/speckit_vs_bmad
mv -v ~/speckit_vs_bmad.mp3            ~/podcast-generator/output/speckit_vs_bmad/ 2>/dev/null || true
mv -v ~/speckit_vs_bmad.wav            ~/podcast-generator/output/speckit_vs_bmad/ 2>/dev/null || true
mv -v ~/speckit_vs_bmad_mastered.mp3   ~/podcast-generator/output/speckit_vs_bmad/ 2>/dev/null || true
mv -v ~/speckit_vs_bmad_normalized.mp3 ~/podcast-generator/output/speckit_vs_bmad/ 2>/dev/null || true
mv -v ~/speckit_vs_bmad_dialogue.txt   ~/podcast-generator/scripts/speckit_vs_bmad_dialogue.txt 2>/dev/null || true

# vv_framework
mkdir -p ~/podcast-generator/output/vv_framework
mv -v ~/vv_framework_chatterbox.wav ~/podcast-generator/output/vv_framework/ 2>/dev/null || true
mv -v ~/vv_framework_marc.wav       ~/podcast-generator/output/vv_framework/ 2>/dev/null || true
mv -v ~/vv_framework_narrator.mp3   ~/podcast-generator/output/vv_framework/ 2>/dev/null || true
mv -v ~/vv_framework_narrator.wav   ~/podcast-generator/output/vv_framework/ 2>/dev/null || true
mv -v ~/vv_framework_dialogue.txt      ~/podcast-generator/scripts/vv_framework_dialogue.txt 2>/dev/null || true
mv -v ~/vv_framework_v2_dialogue.txt   ~/podcast-generator/scripts/vv_framework_v2_dialogue.txt 2>/dev/null || true

# 3. Move the generator script
echo "Moving generator script..."
mv -v ~/generate_podcast_chatterbox.py ~/podcast-generator/ 2>/dev/null || true

# 4. Move logs
echo "Moving logs..."
mv -v ~/podcast.log     ~/podcast-generator/logs/ 2>/dev/null || true
mv -v ~/podcast_gen.log ~/podcast-generator/logs/ 2>/dev/null || true

# 5. Move TTS test samples
echo "Moving TTS test samples..."
mv -v ~/chatterbox_lisa_test.wav    ~/podcast-generator/tts-tests/ 2>/dev/null || true
mv -v ~/chatterbox_marc_test.wav    ~/podcast-generator/tts-tests/ 2>/dev/null || true
mv -v ~/chatterbox_sven_test.wav    ~/podcast-generator/tts-tests/ 2>/dev/null || true
mv -v ~/marc_chatterbox_test.wav    ~/podcast-generator/tts-tests/ 2>/dev/null || true
mv -v ~/styletts2_lisa_test.wav     ~/podcast-generator/tts-tests/ 2>/dev/null || true
mv -v ~/styletts2_marc_test.wav     ~/podcast-generator/tts-tests/ 2>/dev/null || true
mv -v ~/styletts2_sven_test.wav     ~/podcast-generator/tts-tests/ 2>/dev/null || true
mv -v ~/dutch_test.wav              ~/podcast-generator/tts-tests/ 2>/dev/null || true
mv -v ~/dutch_test_xtts.wav         ~/podcast-generator/tts-tests/ 2>/dev/null || true
mv -v ~/dutch_test_xtts_emma.wav    ~/podcast-generator/tts-tests/ 2>/dev/null || true
mv -v ~/dutch_test_xtts_lisa.wav    ~/podcast-generator/tts-tests/ 2>/dev/null || true
mv -v ~/dutch_test_xtts_sven.wav    ~/podcast-generator/tts-tests/ 2>/dev/null || true

# 6. Move whisper output
echo "Moving whisper files..."
mv -v ~/youtube-audio.mp3 ~/whisper/output/ 2>/dev/null || true
mv -v ~/youtube-audio.txt ~/whisper/output/ 2>/dev/null || true

# 7. Clean up stale __pycache__
echo "Removing stale __pycache__..."
rm -rf ~/__pycache__

echo ""
echo "=== Done! ==="
echo ""
echo "New structure:"
find ~/podcast-generator ~/whisper -type d | sort
echo ""
echo "Files remaining in ~/:"
find ~/ -maxdepth 1 -type f ! -name '.*' | sort
