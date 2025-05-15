#!/usr/bin/env python
"""
Long Audio Generation Script

This script provides a command-line interface for generating long audio clips with Dia-TTS.
It automatically chunks text and handles the generation of long audio content.

Example usage:
  python generate_long_audio.py --text "This is a long text to convert to speech..." --output output.wav --voice S1
  
  Or with a text file:
  python generate_long_audio.py --text-file my_script.txt --output podcast.mp3 --voice my_voice.wav
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("long_audio_cli")

# Add the parent directory to the path to make imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import server modules
from config import config_manager
from engine import load_model
from app.utils.long_audio import generate_and_save_long_audio


def main():
    parser = argparse.ArgumentParser(description="Generate long audio with Dia-TTS")
    
    # Input text options (mutually exclusive)
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", type=str, help="Text to convert to speech")
    text_group.add_argument("--text-file", type=str, help="Path to a text file to convert to speech")
    
    # Output options
    parser.add_argument("--output", required=True, type=str, help="Output audio file path (.wav or .mp3)")
    
    # Voice options
    parser.add_argument("--voice", default="S1", type=str, 
                       help="Voice to use: 'S1', 'S2', 'dialogue', or path to a reference audio file")
    parser.add_argument("--transcript", type=str, help="Optional transcript for voice cloning")
    
    # Generation parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42, -1 for random)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed factor (0.5-2.0, default: 1.0)")
    parser.add_argument("--temperature", type=float, default=1.3, help="Temperature (0.1-1.5, default: 1.3)")
    parser.add_argument("--cfg-scale", type=float, default=3.0, help="CFG scale (1.0-5.0, default: 3.0)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling (0.1-1.0, default: 0.95)")
    parser.add_argument("--chunk-size", type=int, default=300, 
                       help="Maximum characters per chunk (100-1000, default: 300)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get the text content
    if args.text:
        text = args.text
    else:
        try:
            with open(args.text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            logger.info(f"Loaded text from {args.text_file} ({len(text)} characters)")
        except Exception as e:
            logger.error(f"Failed to read text file {args.text_file}: {e}")
            return 1
    
    # Prepare output path
    output_path = Path(args.output)
    if not output_path.parent.exists():
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create output directory {output_path.parent}: {e}")
            return 1
    
    # Determine voice mode and reference file
    voice_mode = "single_s1"
    clone_reference_filename = None
    
    if args.voice.lower() == "s1":
        voice_mode = "single_s1"
    elif args.voice.lower() == "s2":
        voice_mode = "single_s2"
    elif args.voice.lower() == "dialogue":
        voice_mode = "dialogue"
    elif os.path.isfile(args.voice):
        voice_mode = "clone"
        clone_reference_filename = args.voice
    elif args.voice.lower().endswith((".wav", ".mp3")):
        # Check if it's in the reference_audio directory
        from config import get_reference_audio_path
        ref_path = get_reference_audio_path()
        potential_path = os.path.join(ref_path, args.voice)
        if os.path.isfile(potential_path):
            voice_mode = "clone"
            clone_reference_filename = potential_path
        else:
            logger.error(f"Reference file not found: {args.voice}")
            return 1
    else:
        # Check if it's a predefined voice
        from utils import get_predefined_voices
        predefined_voices = get_predefined_voices()
        for voice_info in predefined_voices:
            if (args.voice.lower() == voice_info["display_name"].lower() or 
                args.voice.lower() == voice_info["filename"].lower()):
                voice_mode = "clone"
                from config import get_predefined_voices_path
                predefined_path = get_predefined_voices_path()
                clone_reference_filename = os.path.join(predefined_path, voice_info["filename"])
                break
        
        if voice_mode != "clone" and args.voice not in ["s1", "S1", "s2", "S2", "dialogue"]:
            logger.warning(f"Voice '{args.voice}' not recognized. Using S1 as fallback.")
    
    # Load the model (required before generation)
    logger.info("Loading Dia TTS model...")
    if not load_model():
        logger.error("Failed to load model")
        return 1
    
    # Define a simple progress callback
    def progress_callback(current, total, message):
        percent = int(100 * current / total) if total > 0 else 0
        sys.stdout.write(f"\r{message}: {percent}% ({current}/{total}) ")
        sys.stdout.flush()
    
    # Generate the audio
    logger.info(f"Generating audio for {len(text)} characters...")
    start_time = time.time()
    
    result_path = generate_and_save_long_audio(
        text=text,
        output_path=str(output_path),
        voice_mode=voice_mode,
        clone_reference_filename=clone_reference_filename,
        transcript=args.transcript,
        speed_factor=args.speed,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        max_chunk_chars=args.chunk_size,
        progress_callback=progress_callback
    )
    
    # Print a newline after progress updates
    print()
    
    if result_path:
        duration = time.time() - start_time
        logger.info(f"Successfully generated audio in {duration:.1f} seconds")
        logger.info(f"Output saved to: {result_path}")
        return 0
    else:
        logger.error("Failed to generate audio")
        return 1


if __name__ == "__main__":
    sys.exit(main())
