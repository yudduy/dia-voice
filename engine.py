# engine.py
# Core Dia TTS model loading and generation logic

import logging
import time
import os
import torch
import numpy as np
from typing import Optional, Tuple
from huggingface_hub import hf_hub_download  # Import downloader

# Import Dia model class and config
try:
    from dia.model import Dia
    from dia.config import DiaConfig
except ImportError as e:
    # Log critical error if core components are missing
    logging.critical(
        f"Failed to import Dia model components: {e}. Ensure the 'dia' package exists and is importable.",
        exc_info=True,
    )

    # Define dummy classes/functions to prevent server crash on import,
    # but generation will fail later if these are used.
    class Dia:
        @staticmethod
        def load_model_from_files(*args, **kwargs):
            raise RuntimeError("Dia model package not available or failed to import.")

        def generate(*args, **kwargs):
            raise RuntimeError("Dia model package not available or failed to import.")

    class DiaConfig:
        pass


# Import configuration getters from our project's config.py
from config import (
    get_model_repo_id,
    get_model_cache_path,
    get_reference_audio_path,
    get_model_config_filename,
    get_model_weights_filename,
)

logger = logging.getLogger(__name__)  # Use standard logger name

# --- Global Variables ---
dia_model: Optional[Dia] = None
# model_config is now loaded within Dia.load_model_from_files, maybe remove global?
# Let's keep it for now if needed elsewhere, but populate it after loading.
model_config_instance: Optional[DiaConfig] = None
model_device: Optional[torch.device] = None
MODEL_LOADED = False
EXPECTED_SAMPLE_RATE = 44100  # Dia model and DAC typically operate at 44.1kHz

# --- Model Loading ---


def get_device() -> torch.device:
    """Determines the optimal torch device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        logger.info("CUDA is available, using GPU.")
        return torch.device("cuda")
    # Add MPS check for Apple Silicon GPUs
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Basic check is usually sufficient
        logger.info("MPS is available, using Apple Silicon GPU.")
        return torch.device("mps")
    else:
        logger.info("CUDA and MPS not available, using CPU.")
        return torch.device("cpu")


def load_model():
    """
    Loads the Dia TTS model and associated DAC model.
    Downloads model files based on configuration if they don't exist locally.
    Handles both .pth and .safetensors formats.
    """
    global dia_model, model_config_instance, model_device, MODEL_LOADED

    if MODEL_LOADED:
        logger.info("Dia model already loaded.")
        return True

    # Get configuration values
    repo_id = get_model_repo_id()
    config_filename = get_model_config_filename()
    weights_filename = get_model_weights_filename()
    cache_path = get_model_cache_path()  # Already absolute path
    model_device = get_device()

    logger.info(f"Attempting to load Dia model:")
    logger.info(f"  Repo ID: {repo_id}")
    logger.info(f"  Config File: {config_filename}")
    logger.info(f"  Weights File: {weights_filename}")
    logger.info(f"  Cache Directory: {cache_path}")
    logger.info(f"  Target Device: {model_device}")

    # Ensure cache directory exists
    try:
        os.makedirs(cache_path, exist_ok=True)
    except OSError as e:
        logger.error(
            f"Failed to create cache directory '{cache_path}': {e}", exc_info=True
        )
        # Depending on severity, might want to return False here
        # return False
        pass  # Continue and let hf_hub_download handle potential issues

    try:
        start_time = time.time()

        # --- Download Model Files ---
        logger.info(
            f"Downloading/finding configuration file '{config_filename}' from repo '{repo_id}'..."
        )
        local_config_path = hf_hub_download(
            repo_id=repo_id,
            filename=config_filename,
            cache_dir=cache_path,
            # force_download=False, # Default: only download if missing or outdated
            # resume_download=True, # Default: resume interrupted downloads
        )
        logger.info(f"Configuration file path: {local_config_path}")

        logger.info(
            f"Downloading/finding weights file '{weights_filename}' from repo '{repo_id}'..."
        )
        local_weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=weights_filename,
            cache_dir=cache_path,
        )
        logger.info(f"Weights file path: {local_weights_path}")

        # --- Load Model using the class method ---
        # The Dia class method now handles config loading, instantiation, weight loading, etc.
        dia_model = Dia.load_model_from_files(
            config_path=local_config_path,
            weights_path=local_weights_path,
            device=model_device,
        )

        # Store the config instance if needed globally (optional)
        model_config_instance = dia_model.config

        end_time = time.time()
        logger.info(
            f"Dia model loaded successfully in {end_time - start_time:.2f} seconds."
        )
        MODEL_LOADED = True
        return True

    except FileNotFoundError as e:
        logger.error(
            f"Model loading failed: Required file not found. {e}", exc_info=True
        )
        MODEL_LOADED = False
        return False
    except ImportError:
        # This catches if the 'dia' package itself is missing
        logger.critical(
            "Failed to load model: Dia package or its core dependencies not found.",
            exc_info=True,
        )
        MODEL_LOADED = False
        return False
    except Exception as e:
        # Catch other potential errors during download or loading
        logger.error(
            f"Error loading Dia model from repo '{repo_id}': {e}", exc_info=True
        )
        dia_model = None
        model_config_instance = None
        MODEL_LOADED = False
        return False


# --- Speech Generation ---


def generate_speech(
    text: str,
    voice_mode: str = "single_s1",
    clone_reference_filename: Optional[str] = None,
    max_tokens: Optional[int] = None,
    cfg_scale: float = 3.0,
    temperature: float = 1.3,
    top_p: float = 0.95,
    speed_factor: float = 0.94,  # Keep speed factor separate from model generation params
    cfg_filter_top_k: int = 35,
) -> Optional[Tuple[np.ndarray, int]]:
    """
    Generates speech using the loaded Dia model, handling voice modes and speed adjustment.

    Args:
        text: Text to synthesize.
        voice_mode: 'dialogue', 'single_s1', 'single_s2', 'clone'.
        clone_reference_filename: Filename for voice cloning (if mode is 'clone'). Located in reference audio path.
        max_tokens: Max generation tokens for the model's generate method.
        cfg_scale: CFG scale for the model's generate method.
        temperature: Sampling temperature for the model's generate method.
        top_p: Nucleus sampling p for the model's generate method.
        speed_factor: Factor to adjust the playback speed *after* generation (e.g., 0.9 = slower, 1.1 = faster).
        cfg_filter_top_k: CFG filter top K for the model's generate method.

    Returns:
        Tuple of (numpy_audio_array, sample_rate), or None on failure.
    """
    if not MODEL_LOADED or dia_model is None:
        logger.error("Dia model is not loaded. Cannot generate speech.")
        return None

    logger.info(f"Generating speech with mode: {voice_mode}")
    logger.debug(f"Input text (start): '{text[:100]}...'")
    # Log model generation parameters
    logger.debug(
        f"Model Params: max_tokens={max_tokens}, cfg={cfg_scale}, temp={temperature}, top_p={top_p}, top_k={cfg_filter_top_k}"
    )
    # Log post-processing parameters
    logger.debug(f"Post-processing Params: speed_factor={speed_factor}")

    audio_prompt_path = None
    processed_text = text  # Start with original text

    # --- Handle Voice Mode ---
    if voice_mode == "clone":
        if not clone_reference_filename:
            logger.error("Clone mode selected but no reference filename provided.")
            return None
        ref_base_path = get_reference_audio_path()  # Gets absolute path
        potential_path = os.path.join(ref_base_path, clone_reference_filename)
        if os.path.isfile(potential_path):
            audio_prompt_path = potential_path
            logger.info(f"Using audio prompt for cloning: {audio_prompt_path}")
            # Dia requires the transcript of the clone audio to be prepended to the target text.
            # The UI/API caller is responsible for constructing this combined text.
            logger.warning(
                "Clone mode active. Ensure the 'text' input includes the transcript of the reference audio for best results (e.g., '[S1] Reference transcript. [S1] Target text...')."
            )
            processed_text = text  # Use the combined text provided by the caller
        else:
            logger.error(f"Reference audio file not found: {potential_path}")
            return None  # Fail generation if reference file is missing
    elif voice_mode == "dialogue":
        # Assume text already contains [S1]/[S2] tags as required by the model
        logger.info("Using dialogue mode. Expecting [S1]/[S2] tags in input text.")
        if "[S1]" not in text and "[S2]" not in text:
            logger.warning(
                "Dialogue mode selected, but no [S1] or [S2] tags found in the input text."
            )
        processed_text = text  # Pass directly
    elif voice_mode == "single_s1":
        logger.info("Using single voice mode (S1).")
        # Check if text *already* contains tags, warn if so, as it might confuse the model
        if "[S1]" in text or "[S2]" in text:
            logger.warning(
                "Input text contains dialogue tags ([S1]/[S2]), but 'single_s1' mode was selected. Model behavior might be unexpected."
            )
        # Dia likely expects tags even for single speaker. Prepending [S1] might be safer.
        # Let's assume for now the model handles untagged text as S1, but this could be adjusted.
        # Consider: processed_text = f"[S1] {text}" # Option to enforce S1 tag
        processed_text = text  # Pass directly for now
    elif voice_mode == "single_s2":
        logger.info("Using single voice mode (S2).")
        if "[S1]" in text or "[S2]" in text:
            logger.warning(
                "Input text contains dialogue tags ([S1]/[S2]), but 'single_s2' mode was selected."
            )
        # Similar to S1, how to signal S2? Prepending [S2] seems logical if needed.
        # Consider: processed_text = f"[S2] {text}" # Option to enforce S2 tag
        processed_text = text  # Pass directly for now
    else:
        logger.error(
            f"Unsupported voice_mode: {voice_mode}. Defaulting to 'single_s1'."
        )
        processed_text = text  # Fallback

    # --- Call Dia Generate ---
    try:
        start_time = time.time()
        logger.info("Calling Dia model generate method...")

        # Call the model's generate method with appropriate parameters
        generated_audio_np = dia_model.generate(
            text=processed_text,
            audio_prompt_path=audio_prompt_path,
            max_tokens=max_tokens,  # Pass None if not specified, Dia uses its default
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            use_cfg_filter=True,  # Default from Dia's app.py, seems reasonable
            cfg_filter_top_k=cfg_filter_top_k,
            use_torch_compile=False,  # Keep False for stability unless specifically tested/enabled
        )
        gen_end_time = time.time()
        logger.info(
            f"Dia model generation finished in {gen_end_time - start_time:.2f} seconds."
        )

        if generated_audio_np is None or generated_audio_np.size == 0:
            logger.warning("Dia model returned None or empty audio array.")
            return None

        # --- Apply Speed Factor (Post-processing) ---
        # This mimics the logic in Dia's original app.py
        if speed_factor != 1.0:
            logger.info(f"Applying speed factor: {speed_factor}")
            original_len = len(generated_audio_np)
            # Ensure speed_factor is within a reasonable range to avoid extreme distortion
            # Adjust range based on observed quality (e.g., 0.5 to 2.0)
            speed_factor = max(0.5, min(speed_factor, 2.0))
            target_len = int(original_len / speed_factor)

            if target_len > 0 and target_len != original_len:
                logger.debug(
                    f"Resampling audio from {original_len} to {target_len} samples."
                )
                # Create time axes for original and resampled audio
                x_original = np.linspace(0, original_len - 1, original_len)
                x_resampled = np.linspace(0, original_len - 1, target_len)
                # Interpolate using numpy
                resampled_audio_np = np.interp(
                    x_resampled, x_original, generated_audio_np
                )
                final_audio_np = resampled_audio_np.astype(np.float32)  # Ensure float32
                logger.info(f"Audio resampled for {speed_factor:.2f}x speed.")
            else:
                logger.warning(
                    f"Skipping speed adjustment (factor: {speed_factor:.2f}). Target length invalid ({target_len}) or no change needed."
                )
                final_audio_np = generated_audio_np  # Use original audio
        else:
            logger.info("Speed factor is 1.0, no speed adjustment needed.")
            final_audio_np = generated_audio_np  # No speed change needed

        # Ensure output is float32 (DAC output should be, but good practice)
        if final_audio_np.dtype != np.float32:
            logger.warning(
                f"Generated audio was not float32 ({final_audio_np.dtype}), converting."
            )
            final_audio_np = final_audio_np.astype(np.float32)

        logger.info(
            f"Final audio ready. Shape: {final_audio_np.shape}, dtype: {final_audio_np.dtype}"
        )
        # Return the processed audio and the expected sample rate
        return final_audio_np, EXPECTED_SAMPLE_RATE

    except Exception as e:
        logger.error(
            f"Error during Dia generation or post-processing: {e}", exc_info=True
        )
        return None  # Return None on failure
