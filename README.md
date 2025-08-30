# JA→EN Real-Time Desktop Audio Translator

Captures desktop audio on Linux (PulseAudio or PipeWire) and translates Japanese speech to English in real time in the terminal.

## Features
- Fully local translation using Whisper/faster-whisper
- Supports GPU (CUDA + cuDNN) or CPU-only
- Optional bilingual display (original Japanese + English)
- All in the terminal

## Troubleshooting

### 1. GPU errors
- On Arch Linux, install NVIDIA driver, CUDA, and cuDNN:
```bash
sudo pacman -Syu nvidia nvidia-utils cuda cudnn

# Reboot after installation.

# Verify sucessful installation:
nvidia-smi
ls /usr/lib/libcudnn*

# If you don’t have a GPU or want a simpler setup, force CPU mode:
./ja2en_.py --device cpu --compute-type int8 --model small
```
### 2. No text / audio issues
- Verify you're recording the correct PulseAudio/Pipewire monitor
```bash
pactl list short sources | grep monitor
```
- Test recording
```bash
ffmpeg -f pulse -i <monitor> -t 3 -ac 1 -ar 16000 test.wav
ffplay test.wav
```


## Installation
```bash
sudo pacman -S ffmpeg python-pip
python -m venv ja2en-rt
source ja2en-rt/bin/activate ## If you're using fish: source ~/ja2en-rt/bin/activate.fish
pip install -r requirements.txt

## -h or --help for options and usage
