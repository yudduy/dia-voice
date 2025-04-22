# utils.py
# Utility functions for the Dia TTS server

import logging
import time
import os
import io
import numpy as np
import soundfile as sf
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# --- Audio Processing ---


def encode_audio(
    audio_array: np.ndarray, sample_rate: int, output_format: str = "opus"
) -> Optional[bytes]:
    """
    Encodes a NumPy audio array into the specified format in memory.

    Args:
        audio_array: NumPy array containing audio data (float32, range [-1, 1]).
        sample_rate: Sample rate of the audio data.
        output_format: Desired output format ('opus' or 'wav').

    Returns:
        Bytes object containing the encoded audio, or None on failure.
    """
    if audio_array is None or audio_array.size == 0:
        logger.warning("encode_audio received empty or None audio array.")
        return None

    start_time = time.time()
    output_buffer = io.BytesIO()

    try:
        if output_format == "opus":
            # Soundfile expects int16 for Opus usually, but let's try float32 first
            # It might convert internally or require specific subtypes.
            # If this fails, we might need to convert to int16 first:
            # audio_int16 = (audio_array * 32767).astype(np.int16)
            # sf.write(output_buffer, audio_int16, sample_rate, format='ogg', subtype='opus')
            sf.write(
                output_buffer, audio_array, sample_rate, format="ogg", subtype="opus"
            )
            content_type = "audio/ogg; codecs=opus"
        elif output_format == "wav":
            # WAV typically uses int16
            audio_int16 = (audio_array * 32767).astype(np.int16)
            sf.write(
                output_buffer, audio_int16, sample_rate, format="wav", subtype="pcm_16"
            )
            content_type = "audio/wav"
        else:
            logger.error(f"Unsupported output format requested: {output_format}")
            return None

        encoded_bytes = output_buffer.getvalue()
        end_time = time.time()
        logger.info(
            f"Encoded {len(encoded_bytes)} bytes to {output_format} in {end_time - start_time:.3f} seconds."
        )
        return encoded_bytes

    except ImportError:
        logger.critical(
            "`soundfile` or its dependency `libsndfile` not found/installed correctly. Cannot encode audio."
        )
        raise  # Re-raise critical error
    except Exception as e:
        logger.error(f"Error encoding audio to {output_format}: {e}", exc_info=True)
        return None


def save_audio_to_file(
    audio_array: np.ndarray, sample_rate: int, file_path: str
) -> bool:
    """
    Saves a NumPy audio array to a WAV file.

    Args:
        audio_array: NumPy array containing audio data (float32, range [-1, 1]).
        sample_rate: Sample rate of the audio data.
        file_path: Path to save the WAV file.

    Returns:
        True if saving was successful, False otherwise.
    """
    if audio_array is None or audio_array.size == 0:
        logger.warning("save_audio_to_file received empty or None audio array.")
        return False
    if not file_path.lower().endswith(".wav"):
        logger.warning(
            f"File path '{file_path}' does not end with .wav. Saving as WAV anyway."
        )
        # Optionally change the extension: file_path += ".wav"

    start_time = time.time()
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # WAV typically uses int16
        audio_int16 = (audio_array * 32767).astype(np.int16)
        sf.write(file_path, audio_int16, sample_rate, format="wav", subtype="pcm_16")

        end_time = time.time()
        logger.info(
            f"Saved WAV file to {file_path} in {end_time - start_time:.3f} seconds."
        )
        return True
    except ImportError:
        logger.critical(
            "`soundfile` or its dependency `libsndfile` not found/installed correctly. Cannot save audio."
        )
        return False  # Indicate failure
    except Exception as e:
        logger.error(f"Error saving WAV file to {file_path}: {e}", exc_info=True)
        return False


# --- Other Utilities (Optional) ---


class PerformanceMonitor:
    """Simple performance monitoring."""

    def __init__(self):
        self.start_time = time.time()
        self.events = []

    def record(self, event_name: str):
        self.events.append((event_name, time.time()))

    def report(self) -> str:
        report_lines = ["Performance Report:"]
        last_time = self.start_time
        total_duration = time.time() - self.start_time
        for name, timestamp in self.events:
            duration = timestamp - last_time
            report_lines.append(f"  - {name}: {duration:.3f}s")
            last_time = timestamp
        report_lines.append(f"Total Duration: {total_duration:.3f}s")
        return "\n".join(report_lines)
