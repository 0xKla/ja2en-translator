#!/usr/bin/env python3
"""
Real-time desktop audio multi-language -> EN translator.

Dependencies: faster-whisper, rich
Supports CPU and CUDA GPU with auto-language detection.

CAPTURES ALL DESKTOP AUDIO FROM SELECTED Pulse/PipeWire SOURCE.
"""

import argparse
import os
import subprocess
import tempfile
from datetime import datetime
from typing import Optional, Dict, Tuple
from faster_whisper import WhisperModel
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# Supported languages mapping (ISO 639-1 -> full name)
SUPPORTED_LANGUAGES = {
    "auto": "Auto-detect",
    "zh": "Chinese",
    "de": "German", 
    "es": "Spanish",
    "ru": "Russian",
    "ko": "Korean",
    "fr": "French",
    "ja": "Japanese",
    "pt": "Portuguese",
    "tr": "Turkish",
    "pl": "Polish",
    "ca": "Catalan",
    "nl": "Dutch",
    "ar": "Arabic",
    "sv": "Swedish",
    "it": "Italian",
    "id": "Indonesian",
    "hi": "Hindi",
    "fi": "Finnish",
    "vi": "Vietnamese",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "el": "Greek",
    "ms": "Malay",
    "cs": "Czech",
    "ro": "Romanian",
    "da": "Danish",
    "hu": "Hungarian",
    "ta": "Tamil",
    "no": "Norwegian",
    "th": "Thai",
    "ur": "Urdu",
    "hr": "Croatian",
    "bg": "Bulgarian",
    "lt": "Lithuanian",
    "la": "Latin",
    "mi": "Maori",
    "ml": "Malayalam",
    "cy": "Welsh",
    "sk": "Slovak",
    "te": "Telugu",
    "fa": "Persian",
    "lv": "Latvian",
    "bn": "Bengali",
    "sr": "Serbian",
    "az": "Azerbaijani",
    "sl": "Slovenian",
    "kn": "Kannada",
    "et": "Estonian",
    "mk": "Macedonian",
    "br": "Breton",
    "eu": "Basque",
    "is": "Icelandic",
    "hy": "Armenian",
    "ne": "Nepali",
    "mn": "Mongolian",
    "bs": "Bosnian",
    "kk": "Kazakh",
    "sq": "Albanian",
    "sw": "Swahili",
    "gl": "Galician",
    "mr": "Marathi",
    "pa": "Punjabi",
    "si": "Sinhala",
    "km": "Khmer",
    "sn": "Shona",
    "yo": "Yoruba",
    "so": "Somali",
    "af": "Afrikaans",
    "oc": "Occitan",
    "ka": "Georgian",
    "be": "Belarusian",
    "tg": "Tajik",
    "sd": "Sindhi",
    "gu": "Gujarati",
    "am": "Amharic",
    "yi": "Yiddish",
    "lo": "Lao",
    "uz": "Uzbek",
    "fo": "Faroese",
    "ht": "Haitian creole",
    "ps": "Pashto",
    "tk": "Turkmen",
    "nn": "Nynorsk",
    "mt": "Maltese",
    "sa": "Sanskrit",
    "lb": "Luxembourgish",
    "my": "Myanmar",
    "bo": "Tibetan",
    "tl": "Tagalog",
    "mg": "Malagasy",
    "as": "Assamese",
    "tt": "Tatar",
    "haw": "Hawaiian",
    "ln": "Lingala",
    "ha": "Hausa",
    "ba": "Bashkir",
    "jw": "Javanese",
    "su": "Sundanese",
}

class AudioTranslator:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.detected_lang_cache: Dict[str, int] = {}
        self.chunk_count = 0
        self.last_detected_lang = "en"  # Default fallback
        
    def record_chunk(self, pulse_source: str, seconds: float, out_path: str) -> bool:
        """Record short audio chunk from desktop monitor source using ffmpeg."""
        try:
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
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Audio recording failed:[/] {e}")
            return False
        except FileNotFoundError:
            console.print("[red]ffmpeg not found. Install with: sudo pacman -S ffmpeg[/]")
            return False

    def load_model(self, model_size: str, device: str, compute_type: str) -> WhisperModel:
        """Load Whisper model, auto-detect GPU if requested."""
        if device == "auto":
            try:
                result = subprocess.run(["nvidia-smi"], 
                                      stdout=subprocess.DEVNULL, 
                                      stderr=subprocess.DEVNULL, 
                                      check=True)
                device = "cuda"
                console.print("[green]✓ CUDA GPU detected[/]")
            except (subprocess.CalledProcessError, FileNotFoundError):
                device = "cpu"
                console.print("[yellow]⚠ No CUDA GPU detected, using CPU[/]")
        
        try:
            return WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as e:
            console.print(f"[red]Failed to load model:[/] {e}")
            console.print("[yellow]Falling back to CPU with int8[/]")
            return WhisperModel(model_size, device="cpu", compute_type="int8")

    def is_valid_audio_file(self, audio_path: str) -> bool:
        """Check if audio file is valid and has content."""
        try:
            return os.path.exists(audio_path) and os.path.getsize(audio_path) > 1024
        except OSError:
            return False

    def detect_language(self, audio_path: str) -> Tuple[str, float]:
        """Detect the language of the audio with confidence score."""
        if not self.is_valid_audio_file(audio_path):
            return self.last_detected_lang, 0.0
            
        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=None,  # Auto-detect
                task="transcribe",
                vad_filter=not self.args.no_vad,
                beam_size=1,  # Faster for detection
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
            )
            
            # Get detected language
            detected_lang = getattr(info, 'language', self.last_detected_lang)
            confidence = getattr(info, 'language_probability', 0.0)
            
            # Update last known good language
            if confidence > 0.3:
                self.last_detected_lang = detected_lang
            
            return detected_lang, confidence
            
        except Exception:
            return self.last_detected_lang, 0.0

    def transcribe_and_translate(self, audio_path: str, source_lang: str) -> Tuple[str, str, str]:
        """Transcribe and translate audio chunk."""
        if not self.is_valid_audio_file(audio_path):
            return "", "", source_lang
        
        try:
            # If auto-detect, determine language first
            if source_lang == "auto":
                detected_lang, confidence = self.detect_language(audio_path)
                
                # Cache frequently detected languages (only non-English)
                if detected_lang != "en" and confidence > 0.3:
                    self.detected_lang_cache[detected_lang] = self.detected_lang_cache.get(detected_lang, 0) + 1
                
                # Use detected language if confidence is reasonable
                if confidence > 0.3:
                    source_lang = detected_lang
                else:
                    # Fallback to most common cached language or last detected
                    if self.detected_lang_cache:
                        source_lang = max(self.detected_lang_cache, key=self.detected_lang_cache.get)
                    else:
                        source_lang = self.last_detected_lang
            else:
                detected_lang = source_lang
                confidence = 1.0

            # Get original text (transcription)
            segments_orig, _ = self.model.transcribe(
                audio_path,
                language=source_lang if source_lang != "auto" else None,
                task="transcribe",
                vad_filter=not self.args.no_vad,
                vad_parameters=dict(min_silence_duration_ms=300),
                beam_size=5,
                condition_on_previous_text=True,
                no_speech_threshold=0.6,
                temperature=0.0,
            )
            original_text = "".join(seg.text for seg in segments_orig).strip()

            # Skip translation if already English or no text
            if not original_text or source_lang == "en":
                return original_text, original_text, source_lang

            # Get translation
            segments_trans, _ = self.model.transcribe(
                audio_path,
                language=source_lang,
                task="translate",
                vad_filter=not self.args.no_vad,
                vad_parameters=dict(min_silence_duration_ms=300),
                beam_size=5,
                condition_on_previous_text=True,
                no_speech_threshold=0.6,
                temperature=0.1,
            )
            translated_text = "".join(seg.text for seg in segments_trans).strip()
            
            # If translation is empty but original isn't, fallback to original
            if not translated_text and original_text:
                translated_text = original_text

            return original_text, translated_text, source_lang

        except Exception:
            return "", "", source_lang

    def display_result(self, original: str, translated: str, detected_lang: str):
        """Display the transcription and translation results."""
        if not translated and not original:
            return
            
        self.chunk_count += 1
        timestamp = datetime.now().strftime('%H:%M:%S') if self.args.show_timestamps else ""
        
        # Language display
        lang_name = SUPPORTED_LANGUAGES.get(detected_lang, detected_lang.upper())
        
        if self.args.compact:
            # Compact single-line output
            if detected_lang != "en" and translated and translated != original:
                prefix = f"[{timestamp}] " if timestamp else ""
                console.print(f"{prefix}[bold blue]{lang_name}→EN:[/] {translated}")
            elif original:
                prefix = f"[{timestamp}] " if timestamp else ""
                console.print(f"{prefix}[bold green]{lang_name}:[/] {original}")
        else:
            # Rich formatted output
            if detected_lang != "en" and translated and translated != original:
                if self.args.show_original or self.args.bilingual:
                    console.print(f"[dim]{lang_name}:[/] {original}")
                console.print(f"[bold yellow]English:[/] {translated}")
            elif original:
                console.print(f"[bold green]{lang_name}:[/] {original}")
            else:
                # Show both even if translation failed
                if original:
                    console.print(f"[dim]{lang_name}:[/] {original}")
                if translated:
                    console.print(f"[bold yellow]English:[/] {translated}")
            
            if timestamp:
                console.print(f"[dim]└─ {timestamp}[/]")

    def list_audio_sources(self):
        """List available PulseAudio/PipeWire monitor sources."""
        try:
            result = subprocess.run(
                ["pactl", "list", "short", "sources"],
                capture_output=True, text=True, check=True
            )
            
            console.print("\n[bold cyan]Available Audio Sources:[/]")
            table = Table()
            table.add_column("Index", style="cyan")
            table.add_column("Name", style="white")
            table.add_column("Description", style="dim")
            
            for line in result.stdout.strip().split('\n'):
                if 'monitor' in line.lower():
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        table.add_row(parts[0], parts[1], "Desktop Audio Monitor")
            
            console.print(table)
            console.print("\n[yellow]Use --source <name> to select a specific source[/]\n")
            
        except subprocess.CalledProcessError:
            console.print("[red]Could not list audio sources. Make sure PulseAudio/PipeWire is running.[/]")

    def run(self):
        """Main translation loop."""
        # List sources if requested
        if self.args.list_sources:
            self.list_audio_sources()
            return

        # Load model
        console.print(f"[cyan]Loading Whisper model '{self.args.model}'...[/]")
        self.model = self.load_model(self.args.model, self.args.device, self.args.compute_type)
        
        lang_display = SUPPORTED_LANGUAGES.get(self.args.language, self.args.language)
        console.print(f"[bold green]✓ Model loaded:[/] {self.args.model} on {self.args.device} ({self.args.compute_type})")
        console.print(f"[bold blue]Source Language:[/] {lang_display}")

        # Main loop
        panel = Panel.fit(
            "[bold cyan]🎧 Listening for audio...[/]\n"
            "[dim]Press Ctrl+C to stop[/]",
            border_style="cyan"
        )
        console.print(panel)

        try:
            while True:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                
                try:
                    if self.record_chunk(self.args.source, self.args.chunk, tmp_path):
                        # Only process if file has actual content
                        if self.is_valid_audio_file(tmp_path):
                            original, translated, detected_lang = self.transcribe_and_translate(
                                tmp_path, self.args.language
                            )
                            if original or translated:  # Only display if we got results
                                self.display_result(original, translated, detected_lang)
                        
                except Exception:
                    pass  # Silent error handling - just continue
                    
                finally:
                    try:
                        os.remove(tmp_path)
                    except (OSError, FileNotFoundError):
                        pass

        except KeyboardInterrupt:
            console.print(f"\n[bold red]Stopped after {self.chunk_count} chunks.[/]")

def main():
    parser = argparse.ArgumentParser(
        description="Real-time multi-language->English desktop audio translator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Auto-detect language
  %(prog)s --language ja                      # Japanese to English
  %(prog)s --language zh --bilingual          # Chinese with original text
  %(prog)s --list-sources                     # Show available audio sources
  %(prog)s --language auto --compact          # Auto-detect, compact output
        """
    )
    
    parser.add_argument("--source", default="default", 
                       help="Pulse/PipeWire monitor source (use --list-sources to see options)")
    parser.add_argument("--language", default="auto", 
                       help=f"Source language code or 'auto' for detection. Supported: {', '.join(list(SUPPORTED_LANGUAGES.keys())[:10])}...")
    parser.add_argument("--model", default="small", 
                       help="Whisper model size: tiny, base, small, medium, large-v3 (larger = better but slower)")
    parser.add_argument("--device", default="auto", 
                       help="Processing device: cpu, cuda, or auto")
    parser.add_argument("--compute-type", default="int8", 
                       help="Computation precision: int8, int8_float16, float16, float32")
    parser.add_argument("--chunk", type=float, default=3.0, 
                       help="Audio chunk duration in seconds")
    parser.add_argument("--bilingual", action="store_true", 
                       help="Show original language text alongside English translation")
    parser.add_argument("--show-original", action="store_true",
                       help="Always show original text (alias for --bilingual)")
    parser.add_argument("--show-timestamps", action="store_true", 
                       help="Add timestamps to output")
    parser.add_argument("--compact", action="store_true",
                       help="Compact single-line output format")
    parser.add_argument("--no-vad", action="store_true", 
                       help="Disable voice activity detection filtering")
    parser.add_argument("--list-sources", action="store_true",
                       help="List available audio sources and exit")
    
    args = parser.parse_args()
    
    # Handle aliases
    if args.show_original:
        args.bilingual = True
    
    # Validate language
    if args.language not in SUPPORTED_LANGUAGES:
        console.print(f"[red]Unsupported language: {args.language}[/]")
        console.print("Supported languages:")
        for code, name in list(SUPPORTED_LANGUAGES.items())[:20]:
            console.print(f"  {code}: {name}")
        console.print("  ... and more. Use 'auto' for automatic detection.")
        return 1
    
    translator = AudioTranslator(args)
    translator.run()
    return 0

if __name__ == "__main__":
    exit(main())
