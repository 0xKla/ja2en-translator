# Multi-Language Real-Time Desktop Audio Translator

## Features
- 70+ Language Support - Auto-detect or specify: Japanese, Chinese, German, Spanish, Korean, French, Russian, etc.
- Fully Local Translation - Uses OpenAI Whisper/faster-whisper
- GPU & CPU Support - Automatic CUDA detection with CPU fallback
- Real-Time Processing - Live translation with configurable chunk sizes
- Multiple Output Modes - Compact single-line or bilingual display
- Smart Audio Detection - Voice Activity Detection (VAD) filtering
- All in the terminal

## Installation
### Arch Linux
```bash
# Install system dependencies
sudo pacman -S ffmpeg python-pip

# Create virtual environment
python -m venv any2en
source any2en/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Make script executable
chmod +x any2en.py
```
### Other Linux Distros
```bash
# Ubuntu/Debian
sudo apt install ffmpeg python3-pip python3-venv

# Fedora
sudo dnf install ffmpeg python3-pip 

# Then follow the same venv setup above
```

## Usage
```bash
# Auto-detect language (default)
./any2en.py

# Specify language
./any2en.py --language zh --bilingual

# Specify audio source
./any2en.py --source alsa_output.pci-0000_00_1f.3.analog-stereo.monitor

# Higher quality with larger model 
./any2en.py --language auto --model medium --device cuda --compute-type float16

# Longer audio chunks for better accuracy
./any2en.py --language ja -chunk 5.0 --bilingual

# CPU-only mode
./any2en.py --language auto --device cpu --compute-type int8

# Use --help for all options
```

## Audio Source Setup
```bash
# List audio sources
./any2en.py --list-sources

# Test source with ffmpeg
ffmpeg -f pulse -i your_source -t 3 -ac 1 -ar 16000 test.wav
ffplay test.wav  # Should play back the recorded audio
```

## GPU Setup (Recommended)
```bash
# Install CUDA support
sudo pacman -S nvidia nvidia-utils cuda cudnn

# Reboot and verify installation
nvidia-smi
ls /usr/lib/libcudnn*

# Use GPU acceleration
./any2en.py --device cuda --compute-type float16 --model medium
```
