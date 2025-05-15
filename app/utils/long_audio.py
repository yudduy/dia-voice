"""
Utility for generating longer audio clips by chunking text and combining audio segments.
"""

import io
import os
import logging
import re
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

import numpy as np
from pydub import AudioSegment
import torch

# Import local modules safely
try:
    from ...utils import chunk_text_by_sentences, save_audio_to_file
    from ...engine import generate_speech
    from ...config import get_gen_default_cfg_scale, get_gen_default_temperature, get_gen_default_top_p, get_gen_default_cfg_filter_top_k
except (ImportError, ValueError):
    # Adjust import paths when module is used in different contexts
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from utils import chunk_text_by_sentences, save_audio_to_file
    from engine import generate_speech
    from config import get_gen_default_cfg_scale, get_gen_default_temperature, get_gen_default_top_p, get_gen_default_cfg_filter_top_k

logger = logging.getLogger(__name__)

# Estimated characters per second based on average speaking rate
CHARS_PER_SECOND = 15
# Target chunk duration in seconds
TARGET_CHUNK_DURATION = 20
# Maximum characters per chunk
MAX_CHARS_PER_CHUNK = TARGET_CHUNK_DURATION * CHARS_PER_SECOND


def split_text_into_chunks(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    """
    Split text into chunks respecting sentence boundaries when possible.
    
    Args:
        text: The input text to split
        max_chars: Maximum number of characters per chunk
        
    Returns:
        List of text chunks
    """
    # Extract speaker tag if present
    speaker_tag = ""
    speaker_match = re.match(r'(\[S\d+\])', text)
    if speaker_match:
        speaker_tag = speaker_match.group(1)
        text = text[len(speaker_tag):].strip()
    
    # Use the existing chunk_text_by_sentences utility from main utils.py
    # This handles speaker tags properly and respects sentence boundaries
    chunks = chunk_text_by_sentences(
        text if not speaker_tag else f"{speaker_tag} {text}", 
        chunk_size=max_chars,
        allow_multiple_tags=True
    )
    
    return chunks


def generate_long_audio(
    text: str, 
    voice_mode: str = "single_s1",
    clone_reference_filename: Optional[str] = None,
    transcript: Optional[str] = None,
    max_tokens: Optional[int] = None,
    cfg_scale: Optional[float] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    speed_factor: float = 0.94,
    cfg_filter_top_k: Optional[int] = None,
    seed: int = 42,
    enable_silence_trimming: bool = True,
    enable_internal_silence_fix: bool = True,
    enable_unvoiced_removal: bool = True,
    max_chunk_chars: int = MAX_CHARS_PER_CHUNK,
    progress_callback: Optional[callable] = None
) -> Optional[Tuple[np.ndarray, int]]:
    """
    Generates long audio by splitting text into chunks, processing each chunk separately,
    and then concatenating the results.
    
    Args:
        text: Text to convert to speech
        voice_mode: Voice mode (single_s1, single_s2, dialogue, clone)
        clone_reference_filename: Path to reference audio for voice cloning
        transcript: Optional transcript for cloning
        max_tokens: Maximum number of tokens to generate per chunk
        cfg_scale: Classifier-free guidance scale (defaults to config value if None)
        temperature: Sampling temperature (defaults to config value if None)
        top_p: Nucleus sampling probability (defaults to config value if None)
        speed_factor: Speech speed adjustment
        cfg_filter_top_k: Top-k filter for CFG (defaults to config value if None)
        seed: Random seed for generation
        enable_silence_trimming: Whether to trim leading/trailing silence
        enable_internal_silence_fix: Whether to fix long internal silences
        enable_unvoiced_removal: Whether to remove long unvoiced segments
        max_chunk_chars: Maximum characters per chunk
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Tuple of (audio_array, sample_rate) if successful, None if failed
    """
    # Set default generation parameters from config if not provided
    if cfg_scale is None:
        cfg_scale = get_gen_default_cfg_scale()
    if temperature is None:
        temperature = get_gen_default_temperature()
    if top_p is None:
        top_p = get_gen_default_top_p()
    if cfg_filter_top_k is None:
        cfg_filter_top_k = get_gen_default_cfg_filter_top_k()
    
    # Split text into chunks
    logger.info(f"Splitting long text ({len(text)} chars) into chunks of max {max_chunk_chars} chars...")
    text_chunks = split_text_into_chunks(text, max_chars=max_chunk_chars)
    
    if not text_chunks:
        logger.error("Failed to split text into chunks.")
        return None
    
    logger.info(f"Split text into {len(text_chunks)} chunks for processing.")
    
    # Process each chunk
    all_audio_arrays = []
    all_sample_rates = []
    sample_rate = None
    
    for i, chunk in enumerate(text_chunks):
        logger.info(f"Processing chunk {i+1}/{len(text_chunks)}: {len(chunk)} chars")
        
        if progress_callback:
            progress_callback(i, len(text_chunks), f"Generating chunk {i+1}/{len(text_chunks)}")
        
        # Generate audio for this chunk
        result = generate_speech(
            text_to_process=chunk,
            voice_mode=voice_mode,
            clone_reference_filename=clone_reference_filename,
            transcript=transcript,
            max_tokens=max_tokens,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            speed_factor=speed_factor,
            cfg_filter_top_k=cfg_filter_top_k,
            seed=seed,
            split_text=False,  # Important: we're already splitting the text here
            chunk_size=max_chunk_chars,
            enable_silence_trimming=enable_silence_trimming,
            enable_internal_silence_fix=enable_internal_silence_fix,
            enable_unvoiced_removal=enable_unvoiced_removal
        )
        
        if result is None:
            logger.error(f"Failed to generate audio for chunk {i+1}.")
            continue
        
        audio_array, chunk_sample_rate = result
        all_audio_arrays.append(audio_array)
        all_sample_rates.append(chunk_sample_rate)
        
        # Save sample rate (all chunks should have the same rate)
        if sample_rate is None:
            sample_rate = chunk_sample_rate
        elif sample_rate != chunk_sample_rate:
            logger.warning(f"Sample rate mismatch: {sample_rate} vs {chunk_sample_rate}")
        
        # Clear CUDA cache after each chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Check if we have any audio to concatenate
    if not all_audio_arrays:
        logger.error("No audio chunks were successfully generated.")
        return None
    
    # Concatenate all audio chunks
    logger.info(f"Concatenating {len(all_audio_arrays)} audio chunks...")
    if progress_callback:
        progress_callback(len(text_chunks), len(text_chunks), "Concatenating audio chunks")
    
    try:
        final_audio = np.concatenate(all_audio_arrays)
        logger.info(f"Successfully created long audio of shape {final_audio.shape}")
        return final_audio, sample_rate
    except Exception as e:
        logger.error(f"Error concatenating audio chunks: {e}")
        return None


def generate_and_save_long_audio(
    text: str,
    output_path: str,
    voice_mode: str = "single_s1",
    clone_reference_filename: Optional[str] = None,
    transcript: Optional[str] = None,
    max_tokens: Optional[int] = None,
    cfg_scale: Optional[float] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    speed_factor: float = 0.94,
    cfg_filter_top_k: Optional[int] = None,
    seed: int = 42,
    enable_silence_trimming: bool = True,
    enable_internal_silence_fix: bool = True,
    enable_unvoiced_removal: bool = True,
    max_chunk_chars: int = MAX_CHARS_PER_CHUNK,
    progress_callback: Optional[callable] = None
) -> Optional[str]:
    """
    Generates long audio and saves it to a file.
    
    Args:
        text: Text to convert to speech
        output_path: Path to save the output audio file
        (Other parameters same as generate_long_audio)
        
    Returns:
        Path to the saved audio file if successful, None if failed
    """
    # Generate the long audio
    result = generate_long_audio(
        text=text,
        voice_mode=voice_mode,
        clone_reference_filename=clone_reference_filename,
        transcript=transcript,
        max_tokens=max_tokens,
        cfg_scale=cfg_scale,
        temperature=temperature,
        top_p=top_p,
        speed_factor=speed_factor,
        cfg_filter_top_k=cfg_filter_top_k,
        seed=seed,
        enable_silence_trimming=enable_silence_trimming,
        enable_internal_silence_fix=enable_internal_silence_fix,
        enable_unvoiced_removal=enable_unvoiced_removal,
        max_chunk_chars=max_chunk_chars,
        progress_callback=progress_callback
    )
    
    if result is None:
        logger.error("Failed to generate long audio.")
        return None
    
    audio_array, sample_rate = result
    
    # Save the audio to file
    logger.info(f"Saving audio to {output_path}...")
    if progress_callback:
        progress_callback(1, 1, "Saving audio file")
    
    if save_audio_to_file(audio_array, sample_rate, output_path):
        logger.info(f"Successfully saved audio to {output_path}")
        return output_path
    else:
        logger.error(f"Failed to save audio to {output_path}")
        return None


# API wrapper for OpenAI compatibility
def generate_long_audio_api(
    text: str,
    voice: str = "S1",
    speed: float = 1.0,
    seed: int = 42,
    output_format: str = "mp3",
    max_chunk_chars: int = MAX_CHARS_PER_CHUNK
) -> Optional[bytes]:
    """
    API-compatible wrapper for generating long audio.
    
    Args:
        text: Text to convert to speech
        voice: Voice setting (S1, S2, dialogue, or a reference file)
        speed: Speech speed adjustment
        seed: Random seed for generation
        output_format: Output format (mp3 or wav)
        max_chunk_chars: Maximum characters per chunk
        
    Returns:
        Audio data as bytes if successful, None if failed
    """
    from ...utils import encode_audio
    
    # Map voice parameter to voice_mode
    voice_mode = "single_s1"
    clone_reference_filename = None
    
    if voice.lower() == "s1":
        voice_mode = "single_s1"
    elif voice.lower() == "s2":
        voice_mode = "single_s2"
    elif voice.lower() == "dialogue":
        voice_mode = "dialogue"
    elif voice.lower().endswith((".wav", ".mp3")):
        voice_mode = "clone"
        clone_reference_filename = voice
    else:
        # Check if it's a predefined voice
        from ...utils import get_predefined_voices
        predefined_voices = get_predefined_voices()
        for voice_info in predefined_voices:
            if (voice.lower() == voice_info["display_name"].lower() or 
                voice.lower() == voice_info["filename"].lower()):
                voice_mode = "clone"
                clone_reference_filename = voice_info["filename"]
                break
    
    # Generate the long audio
    result = generate_long_audio(
        text=text,
        voice_mode=voice_mode,
        clone_reference_filename=clone_reference_filename,
        speed_factor=speed,
        seed=seed,
        max_chunk_chars=max_chunk_chars
    )
    
    if result is None:
        logger.error("Failed to generate long audio for API request.")
        return None
    
    audio_array, sample_rate = result
    
    # Encode the audio in the requested format
    return encode_audio(audio_array, sample_rate, output_format)
