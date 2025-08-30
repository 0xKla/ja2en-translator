#!/usr/bin/env python3
"""
Real-time desktop audio JA -> EN translator.

Dependencies: faster-whisper, rich
Supports CPU and CUDA GPU.

CAPTURES ALL DESKTOP AUDIO FROM SELECTED Pulse/PipeWire SOURCE.
"""

import argparse
import os
import subprocess
import tempfile
from datetime import datetime
from faster_whisper import WhisperModel
from rich.console import Console

console = Console()

def record_chunk(pulse_source: str, seconds: float, out_path: str):
    """Record short audio chunk from desktop monitor source using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "pulse",
        "-i", pulse_source,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-t", str(seconds),
        "-y", out_path,
    ]
    subprocess.run(cmd, check=True)

def load_model(model_size: str, device: str, compute_type: str):
    """Load Whisper model, auto-detect GPU if requested."""
    if device == "auto":
        try:
            subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            device = "cuda"
        except Exception:
            device = "cpu"
    return WhisperModel(model_size, device=device, compute_type=compute_type)

def transcribe(model, audio_path: str, language: str, task: str, vad: bool):
    """Transcribe or translate audio chunk."""
    segments, _ = model.transcribe(
        audio_path,
        language=language,
        task=task,
        vad_filter=vad,
        vad_parameters=dict(min_silence_duration_ms=300),
        beam_size=5,
        condition_on_previous_text=True,
        no_speech_threshold=0.1,
    )
    return "".join(seg.text for seg in segments).strip()

def main():
    parser = argparse.ArgumentParser(description="Real-time JA->EN desktop audio translator")
    parser.add_argument("--source", default="default", help="Pulse/PipeWire monitor source")
    parser.add_argument("--model", default="small", help="Whisper model size (tiny, base, small, medium, large-v3)")
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--compute-type", default="int8", help="int8, int8_float16, float16, float32")
    parser.add_argument("--chunk", type=float, default=3.0, help="Seconds per audio chunk")
    parser.add_argument("--bilingual", action="store_true", help="Show Japanese transcription too")
    parser.add_argument("--show-timestamps", action="store_true", help="Prefix translations with timestamp")
    parser.add_argument("--no-vad", action="store_true", help="Disable voice activity detection filtering")
    args = parser.parse_args()

    model = load_model(args.model, args.device, args.compute_type)
    console.print(f"[bold green]Loaded model:[/] {args.model} on {args.device} ({args.compute_type})")

    vad = not args.no_vad

    # Warm-up model
    with tempfile.NamedTemporaryFile(suffix=".wav") as warm:
        open(warm.name, "wb").close()
        try:
            transcribe(model, warm.name, "ja", "translate", vad)
        except Exception:
            pass

    console.print("[bold cyan]Listening... Press Ctrl+C to quit[/]")

    try:
        while True:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                record_chunk(args.source, args.chunk, tmp_path)
                en_text = transcribe(model, tmp_path, "ja", "translate", vad)
                if en_text:
                    prefix = f"[{datetime.now().strftime('%H:%M:%S')}] " if args.show_timestamps else ""
                    console.print(f"{prefix}[bold yellow]EN:[/] {en_text}")
                    if args.bilingual:
                        ja_text = transcribe(model, tmp_path, "ja", "transcribe", vad)
                        if ja_text:
                            console.print(f"{prefix}[dim]JA:[/] {ja_text}")
            except Exception as e:
                console.print(f"[red]Error processing chunk:[/] {e}")
            finally:
                try:
                    os.remove(tmp_path)
                except FileNotFoundError:
                    pass

    except KeyboardInterrupt:
        console.print("\n[bold red]Stopped.[/]")

if __name__ == "__main__":
    main()

