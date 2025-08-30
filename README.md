# JA→EN Real-Time Desktop Audio Translator

Captures desktop audio on Linux (PulseAudio or PipeWire) and translates Japanese speech to English in real time in the terminal.

## Features
- Fully offline translation using Whisper/faster-whisper
- Supports GPU (CUDA + cuDNN) or CPU-only
- Optional bilingual display (original Japanese + English)
- Safe temp files and exception handling
- Terminal-based, no GUI required

## Installation

```bash
sudo pacman -S ffmpeg python-pip
python -m venv ja2en-rt
source ja2en-rt/bin/activate ## If you're using fish: source ~/ja2en-rt/bin/activate.fish
pip install -r requirements.txt

##-h or --help for options and usage
