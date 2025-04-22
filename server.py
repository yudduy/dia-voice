# server.py
# Main FastAPI server for Dia TTS

import logging
import time
import os
import io
import uuid
import sys
import shutil  # For file copying
import yaml  # For loading presets
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, Literal, List, Dict, Any

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
    Form,
    UploadFile,
    File,
    BackgroundTasks,
)
from fastapi.responses import (
    StreamingResponse,
    JSONResponse,
    HTMLResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np

# Internal imports
from config import (
    config_manager,
    get_host,
    get_port,
    get_output_path,
    get_reference_audio_path,
    # register_config_routes is now defined locally
    get_model_cache_path,
    get_model_repo_id,
    get_model_config_filename,
    get_model_weights_filename,
    # Generation default getters
    get_gen_default_speed_factor,
    get_gen_default_cfg_scale,
    get_gen_default_temperature,
    get_gen_default_top_p,
    get_gen_default_cfg_filter_top_k,
    DEFAULT_CONFIG,
)
from models import OpenAITTSRequest, CustomTTSRequest, ErrorResponse
from engine import (
    load_model as load_dia_model,
    generate_speech,
    EXPECTED_SAMPLE_RATE,
    MODEL_LOADED,
)
from utils import encode_audio, save_audio_to_file, PerformanceMonitor

# Configure logging (Basic setup, can be enhanced)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
# Reduce verbosity of noisy libraries if needed
# logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
# logging.getLogger("watchfiles").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)  # Logger for this module

# --- Global Variables & Constants ---
PRESETS_FILE = "ui/presets.yaml"
loaded_presets: List[Dict[str, Any]] = []  # Cache presets in memory

# --- Helper Functions ---


def load_presets():
    """Loads presets from the YAML file."""
    global loaded_presets
    try:
        if os.path.exists(PRESETS_FILE):
            with open(PRESETS_FILE, "r", encoding="utf-8") as f:
                loaded_presets = yaml.safe_load(f)
                if not isinstance(loaded_presets, list):
                    logger.error(
                        f"Presets file '{PRESETS_FILE}' should contain a list, but found {type(loaded_presets)}. No presets loaded."
                    )
                    loaded_presets = []
                else:
                    logger.info(
                        f"Successfully loaded {len(loaded_presets)} presets from {PRESETS_FILE}."
                    )
        else:
            logger.warning(
                f"Presets file not found at '{PRESETS_FILE}'. No presets will be available."
            )
            loaded_presets = []
    except yaml.YAMLError as e:
        logger.error(
            f"Error parsing presets YAML file '{PRESETS_FILE}': {e}", exc_info=True
        )
        loaded_presets = []
    except Exception as e:
        logger.error(f"Error loading presets file '{PRESETS_FILE}': {e}", exc_info=True)
        loaded_presets = []


def get_valid_reference_files() -> list[str]:
    """Gets a list of valid audio files (.wav, .mp3) from the reference directory."""
    ref_path = get_reference_audio_path()
    valid_files = []
    allowed_extensions = (".wav", ".mp3")
    try:
        if os.path.isdir(ref_path):
            for filename in os.listdir(ref_path):
                if filename.lower().endswith(allowed_extensions):
                    # Optional: Add check for file size or basic validity if needed
                    valid_files.append(filename)
        else:
            logger.warning(f"Reference audio directory not found: {ref_path}")
    except Exception as e:
        logger.error(
            f"Error reading reference audio directory '{ref_path}': {e}", exc_info=True
        )
    return sorted(valid_files)


def sanitize_filename(filename: str) -> str:
    """Removes potentially unsafe characters and path components from a filename."""
    # Remove directory separators
    filename = os.path.basename(filename)
    # Keep only alphanumeric, underscore, hyphen, dot. Replace others with underscore.
    safe_chars = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    )
    sanitized = "".join(c if c in safe_chars else "_" for c in filename)
    # Prevent names starting with dot or consisting only of dots/spaces
    if not sanitized or sanitized.lstrip("._ ") == "":
        return f"uploaded_file_{uuid.uuid4().hex[:8]}"  # Generate a safe fallback name
    # Limit length
    max_len = 100
    if len(sanitized) > max_len:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[: max_len - len(ext)] + ext
    return sanitized


# --- Application Lifespan (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    logger.info("Starting Dia TTS server initialization...")
    # Ensure base directories exist
    os.makedirs(get_output_path(), exist_ok=True)
    os.makedirs(get_reference_audio_path(), exist_ok=True)
    os.makedirs(get_model_cache_path(), exist_ok=True)
    os.makedirs("ui", exist_ok=True)  # Ensure UI directory exists
    os.makedirs("static", exist_ok=True)  # For favicon etc.

    # Load presets from YAML file
    load_presets()

    # Load the main TTS model during startup
    if not load_dia_model():
        logger.critical(
            "Failed to load Dia model on startup. Server might not function correctly."
        )
        # Depending on desired behavior, could raise an exception here to stop startup
        # raise RuntimeError("Failed to load Dia model.")
    else:
        logger.info("Dia model loaded successfully.")

    yield  # Application runs here

    # Cleanup on shutdown
    logger.info("Application shutdown initiated...")
    # Add any specific cleanup needed for Dia model if required (e.g., release GPU memory explicitly if needed)
    logger.info("Application shutdown complete.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Dia TTS Server",
    description="Text-to-Speech server using the Dia model, providing API and Web UI.",
    version="1.1.0",  # Incremented version
    lifespan=lifespan,
)

# List of folders to check/create
folders = [
    "reference_audio",
    "model_cache",
    "outputs"
]

# Check each folder and create if it doesn't exist
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created directory: {folder}")
    else:
        print(f"Directory already exists: {folder}")

# --- Static Files and Templates ---
# Serve generated audio files from the configured output path
app.mount("/outputs", StaticFiles(directory=get_output_path()), name="outputs")
# Serve UI files (CSS, JS) from the 'ui' directory
app.mount("/ui", StaticFiles(directory="ui"), name="ui_static")
# Initialize Jinja2 templates to look in the 'ui' directory
templates = Jinja2Templates(directory="ui")


# --- Configuration Routes Definition ---
# Defined locally now instead of importing from config.py
def register_config_routes(app: FastAPI):
    """Adds configuration management endpoints to the FastAPI app."""
    logger.info(
        "Registering configuration routes (/get_config, /save_config, /restart_server, /save_generation_defaults)."
    )

    @app.get(
        "/get_config",
        tags=["Configuration"],
        summary="Get current server configuration",
    )
    async def get_current_config():
        """Returns the current server configuration values (from .env or defaults)."""
        logger.info("Request received for /get_config")
        return JSONResponse(content=config_manager.get_all())

    @app.post(
        "/save_config", tags=["Configuration"], summary="Save server configuration"
    )
    async def save_new_config(request: Request):
        """
        Saves updated server configuration values (Host, Port, Model paths, etc.)
        to the .env file. Requires server restart to apply most changes.
        """
        logger.info("Request received for /save_config")
        try:
            new_config_data = await request.json()
            if not isinstance(new_config_data, dict):
                raise ValueError("Request body must be a JSON object.")
            logger.debug(f"Received server config data to save: {new_config_data}")

            # Filter data to only include keys present in DEFAULT_CONFIG
            filtered_data = {
                k: v for k, v in new_config_data.items() if k in DEFAULT_CONFIG
            }
            unknown_keys = set(new_config_data.keys()) - set(filtered_data.keys())
            if unknown_keys:
                logger.warning(
                    f"Ignoring unknown keys in save_config request: {unknown_keys}"
                )

            config_manager.update(filtered_data)  # Update in memory first
            if config_manager.save():  # Attempt to save to .env
                logger.info("Server configuration saved successfully to .env.")
                return JSONResponse(
                    content={
                        "message": "Server configuration saved. Restart server to apply changes."
                    }
                )
            else:
                logger.error("Failed to save server configuration to .env file.")
                raise HTTPException(
                    status_code=500, detail="Failed to save configuration file."
                )
        except ValueError as ve:
            logger.error(f"Invalid data format for /save_config: {ve}")
            raise HTTPException(
                status_code=400, detail=f"Invalid request data: {str(ve)}"
            )
        except Exception as e:
            logger.error(f"Error processing /save_config request: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Internal server error during save: {str(e)}"
            )

    @app.post(
        "/save_generation_defaults",
        tags=["Configuration"],
        summary="Save default generation parameters",
    )
    async def save_generation_defaults(request: Request):
        """
        Saves the provided generation parameters (speed, cfg, temp, etc.)
        as the new defaults in the .env file. These are loaded by the UI on startup.
        """
        logger.info("Request received for /save_generation_defaults")
        try:
            gen_params = await request.json()
            if not isinstance(gen_params, dict):
                raise ValueError("Request body must be a JSON object.")
            logger.debug(f"Received generation defaults to save: {gen_params}")

            # Map received keys (e.g., 'speed_factor') to .env keys (e.g., 'GEN_DEFAULT_SPEED_FACTOR')
            defaults_to_save = {}
            key_map = {
                "speed_factor": "GEN_DEFAULT_SPEED_FACTOR",
                "cfg_scale": "GEN_DEFAULT_CFG_SCALE",
                "temperature": "GEN_DEFAULT_TEMPERATURE",
                "top_p": "GEN_DEFAULT_TOP_P",
                "cfg_filter_top_k": "GEN_DEFAULT_CFG_FILTER_TOP_K",
            }
            valid_keys_found = False
            for ui_key, env_key in key_map.items():
                if ui_key in gen_params:
                    # Basic validation could be added here (e.g., check if float/int)
                    defaults_to_save[env_key] = str(
                        gen_params[ui_key]
                    )  # Ensure saving as string
                    valid_keys_found = True
                else:
                    logger.warning(
                        f"Missing expected key '{ui_key}' in save_generation_defaults request."
                    )

            if not valid_keys_found:
                raise ValueError("No valid generation parameters found in the request.")

            config_manager.update(defaults_to_save)  # Update in memory
            if (
                config_manager.save()
            ):  # Save all current config (including these) to .env
                logger.info("Generation defaults saved successfully to .env.")
                return JSONResponse(content={"message": "Generation defaults saved."})
            else:
                logger.error("Failed to save generation defaults to .env file.")
                raise HTTPException(
                    status_code=500, detail="Failed to save configuration file."
                )
        except ValueError as ve:
            logger.error(f"Invalid data format for /save_generation_defaults: {ve}")
            raise HTTPException(
                status_code=400, detail=f"Invalid request data: {str(ve)}"
            )
        except Exception as e:
            logger.error(
                f"Error processing /save_generation_defaults request: {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500, detail=f"Internal server error during save: {str(e)}"
            )

    @app.post(
        "/restart_server",
        tags=["Configuration"],
        summary="Attempt to restart the server",
    )
    async def trigger_server_restart(background_tasks: BackgroundTasks):
        """
        Attempts to restart the server process.
        NOTE: This is highly dependent on how the server is run (e.g., with uvicorn --reload,
        or managed by systemd/supervisor). A simple exit might just stop the process.
        This implementation attempts a clean exit, relying on the runner to restart it.
        """
        logger.warning("Received request to restart server via API.")

        def _do_restart():
            time.sleep(1)  # Short delay to allow response to be sent
            logger.warning("Attempting clean exit for restart...")
            # Option 1: Clean exit (relies on Uvicorn reload or process manager)
            sys.exit(0)
            # Option 2: Forceful re-execution (use with caution, might not work as expected)
            # try:
            #     logger.warning("Attempting os.execv for restart...")
            #     os.execv(sys.executable, ['python'] + sys.argv)
            # except Exception as exec_e:
            #      logger.error(f"os.execv failed: {exec_e}. Server may not restart automatically.")
            #      # Fallback to sys.exit if execv fails
            #      sys.exit(1)

        background_tasks.add_task(_do_restart)
        return JSONResponse(
            content={
                "message": "Restart signal sent. Server should restart shortly if run with auto-reload."
            }
        )


# --- Register Configuration Routes ---
register_config_routes(app)


# --- API Endpoints ---


@app.post(
    "/v1/audio/speech",
    response_class=StreamingResponse,
    tags=["TTS Generation"],
    summary="Generate speech (OpenAI compatible)",
)
async def openai_tts_endpoint(request: OpenAITTSRequest):
    """
    Generates speech audio from text, compatible with the OpenAI TTS API structure.
    Maps the 'voice' parameter to Dia's voice modes ('S1', 'S2', 'dialogue', or filename for clone).
    """
    monitor = PerformanceMonitor()
    monitor.record("Request received")
    logger.info(
        f"Received OpenAI request: voice='{request.voice}', speed={request.speed}, format='{request.response_format}'"
    )
    logger.debug(f"Input text (start): '{request.input[:100]}...'")

    voice_mode = "single_s1"  # Default if mapping fails
    clone_ref_file = None
    ref_path = get_reference_audio_path()

    # --- Map OpenAI 'voice' parameter to Dia's modes ---
    voice_param = request.voice.strip()
    if voice_param.lower() == "dialogue":
        voice_mode = "dialogue"
    elif voice_param.lower() == "s1":
        voice_mode = "single_s1"
    elif voice_param.lower() == "s2":
        voice_mode = "single_s2"
    # Check if it looks like a filename for cloning (allow .wav or .mp3)
    elif voice_param.lower().endswith((".wav", ".mp3")):
        potential_path = os.path.join(ref_path, voice_param)
        # Check if the file actually exists in the reference directory
        if os.path.isfile(potential_path):
            voice_mode = "clone"
            clone_ref_file = voice_param  # Use the provided filename
            logger.info(
                f"OpenAI request mapped to clone mode with file: {clone_ref_file}"
            )
        else:
            logger.warning(
                f"Reference file '{voice_param}' specified in OpenAI request not found in '{ref_path}'. Defaulting voice mode."
            )
            # Fallback to default 'single_s1' if file not found
    else:
        logger.warning(
            f"Unrecognized OpenAI voice parameter '{voice_param}'. Defaulting voice mode to 'single_s1'."
        )
        # Fallback for any other value

    monitor.record("Parameters processed")

    try:
        # Call the core engine function using mapped parameters
        result = generate_speech(
            text=request.input,
            voice_mode=voice_mode,
            clone_reference_filename=clone_ref_file,
            speed_factor=request.speed,  # Pass speed factor for post-processing
            # Use Dia's configured defaults for other generation params unless mapped
            max_tokens=None,  # Let Dia use its default unless specified otherwise
            cfg_scale=get_gen_default_cfg_scale(),  # Use saved defaults
            temperature=get_gen_default_temperature(),
            top_p=get_gen_default_top_p(),
            cfg_filter_top_k=get_gen_default_cfg_filter_top_k(),
        )
        monitor.record("Generation complete")

        if result is None:
            logger.error("Speech generation failed (engine returned None).")
            raise HTTPException(status_code=500, detail="Speech generation failed.")

        audio_array, sample_rate = result

        if sample_rate != EXPECTED_SAMPLE_RATE:
            logger.warning(
                f"Engine returned sample rate {sample_rate}, but expected {EXPECTED_SAMPLE_RATE}. Encoding might assume {EXPECTED_SAMPLE_RATE}."
            )
            # Use EXPECTED_SAMPLE_RATE for encoding as it's what the model is trained for
            sample_rate = EXPECTED_SAMPLE_RATE

        # Encode the audio in memory to the requested format
        encoded_audio = encode_audio(audio_array, sample_rate, request.response_format)
        monitor.record("Audio encoding complete")

        if encoded_audio is None:
            logger.error(f"Failed to encode audio to format: {request.response_format}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to encode audio to {request.response_format}",
            )

        # Determine the correct media type for the response header
        media_type = "audio/opus" if request.response_format == "opus" else "audio/wav"
        # Note: OpenAI uses audio/opus, not audio/ogg;codecs=opus. Let's match OpenAI.

        logger.info(
            f"Successfully generated {len(encoded_audio)} bytes in format {request.response_format}"
        )
        logger.debug(monitor.report())

        # Stream the encoded audio back to the client
        return StreamingResponse(io.BytesIO(encoded_audio), media_type=media_type)

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly (e.g., from parameter validation)
        logger.error(f"HTTP exception during OpenAI request: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Error processing OpenAI TTS request: {e}", exc_info=True)
        logger.debug(monitor.report())
        # Return generic server error for unexpected issues
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post(
    "/tts",
    response_class=StreamingResponse,
    tags=["TTS Generation"],
    summary="Generate speech (Custom parameters)",
)
async def custom_tts_endpoint(request: CustomTTSRequest):
    """
    Generates speech audio from text using explicit Dia parameters.
    """
    monitor = PerformanceMonitor()
    monitor.record("Request received")
    logger.info(
        f"Received custom TTS request: mode='{request.voice_mode}', format='{request.output_format}'"
    )
    logger.debug(f"Input text (start): '{request.text[:100]}...'")
    logger.debug(
        f"Params: max_tokens={request.max_tokens}, cfg={request.cfg_scale}, temp={request.temperature}, top_p={request.top_p}, speed={request.speed_factor}, top_k={request.cfg_filter_top_k}"
    )

    clone_ref_file = None
    if request.voice_mode == "clone":
        if not request.clone_reference_filename:
            raise HTTPException(
                status_code=400,  # Bad request
                detail="Missing 'clone_reference_filename' which is required for clone mode.",
            )
        ref_path = get_reference_audio_path()
        potential_path = os.path.join(ref_path, request.clone_reference_filename)
        if not os.path.isfile(potential_path):
            logger.error(
                f"Reference audio file not found for clone mode: {potential_path}"
            )
            raise HTTPException(
                status_code=404,  # Not found
                detail=f"Reference audio file not found: {request.clone_reference_filename}",
            )
        clone_ref_file = request.clone_reference_filename
        logger.info(f"Custom request using clone mode with file: {clone_ref_file}")

    monitor.record("Parameters processed")

    try:
        # Call the core engine function with parameters from the request
        result = generate_speech(
            text=request.text,
            voice_mode=request.voice_mode,
            clone_reference_filename=clone_ref_file,
            max_tokens=request.max_tokens,  # Pass user value or None
            cfg_scale=request.cfg_scale,
            temperature=request.temperature,
            top_p=request.top_p,
            speed_factor=request.speed_factor,  # For post-processing
            cfg_filter_top_k=request.cfg_filter_top_k,
        )
        monitor.record("Generation complete")

        if result is None:
            logger.error("Speech generation failed (engine returned None).")
            raise HTTPException(status_code=500, detail="Speech generation failed.")

        audio_array, sample_rate = result

        if sample_rate != EXPECTED_SAMPLE_RATE:
            logger.warning(
                f"Engine returned sample rate {sample_rate}, expected {EXPECTED_SAMPLE_RATE}. Encoding will use {EXPECTED_SAMPLE_RATE}."
            )
            sample_rate = EXPECTED_SAMPLE_RATE

        # Encode the audio in memory
        encoded_audio = encode_audio(audio_array, sample_rate, request.output_format)
        monitor.record("Audio encoding complete")

        if encoded_audio is None:
            logger.error(f"Failed to encode audio to format: {request.output_format}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to encode audio to {request.output_format}",
            )

        # Determine media type
        media_type = "audio/opus" if request.output_format == "opus" else "audio/wav"

        logger.info(
            f"Successfully generated {len(encoded_audio)} bytes in format {request.output_format}"
        )
        logger.debug(monitor.report())

        # Stream the response
        return StreamingResponse(io.BytesIO(encoded_audio), media_type=media_type)

    except HTTPException as http_exc:
        logger.error(f"HTTP exception during custom TTS request: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Error processing custom TTS request: {e}", exc_info=True)
        logger.debug(monitor.report())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# --- Web UI Endpoints ---


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_web_ui(request: Request):
    """Serves the main TTS web interface."""
    logger.info("Serving TTS Web UI (index.html)")
    # Get current list of reference files for the clone dropdown
    reference_files = get_valid_reference_files()
    # Get current server config and default generation params
    current_config = config_manager.get_all()
    default_gen_params = {
        "speed_factor": get_gen_default_speed_factor(),
        "cfg_scale": get_gen_default_cfg_scale(),
        "temperature": get_gen_default_temperature(),
        "top_p": get_gen_default_top_p(),
        "cfg_filter_top_k": get_gen_default_cfg_filter_top_k(),
    }

    return templates.TemplateResponse(
        "index.html",  # Use the renamed file
        {
            "request": request,
            "reference_files": reference_files,
            "config": current_config,  # Pass current server config
            "presets": loaded_presets,  # Pass loaded presets
            "default_gen_params": default_gen_params,  # Pass default gen params
            # Add other variables needed by the template for initial state
            "error": None,
            "success": None,
            "output_file_url": None,
            "generation_time": None,
            "submitted_text": "",
            "submitted_voice_mode": "dialogue",  # Default to combined mode
            "submitted_clone_file": None,
            # Initial generation params will be set by default_gen_params
        },
    )


@app.post("/web/generate", response_class=HTMLResponse, include_in_schema=False)
async def handle_web_ui_generate(
    request: Request,
    text: str = Form(...),
    voice_mode: Literal["dialogue", "clone"] = Form(...),  # Updated modes
    clone_reference_select: Optional[str] = Form(None),
    # Generation parameters from form
    speed_factor: float = Form(...),  # Make required or use Depends with default
    cfg_scale: float = Form(...),
    temperature: float = Form(...),
    top_p: float = Form(...),
    cfg_filter_top_k: int = Form(...),
):
    """Handles the generation request from the web UI form."""
    logger.info(f"Web UI generation request: mode='{voice_mode}'")
    monitor = PerformanceMonitor()
    monitor.record("Web request received")

    output_file_url = None
    generation_time = None
    error_message = None
    success_message = None
    output_filename_base = "dia_output"  # Default base name

    # --- Pre-generation Validation ---
    if not text.strip():
        error_message = "Please enter some text to synthesize."

    clone_ref_file = None
    if voice_mode == "clone":
        if not clone_reference_select or clone_reference_select == "none":
            error_message = "Please select a reference audio file for clone mode."
        else:
            # Verify selected file still exists (important if files can be deleted)
            ref_path = get_reference_audio_path()
            potential_path = os.path.join(ref_path, clone_reference_select)
            if not os.path.isfile(potential_path):
                error_message = f"Selected reference file '{clone_reference_select}' no longer exists. Please refresh or upload."
                # Invalidate selection
                clone_ref_file = None
                clone_reference_select = None  # Clear submitted value for re-rendering
            else:
                clone_ref_file = clone_reference_select
                logger.info(f"Using selected reference file: {clone_ref_file}")

    # If validation failed, re-render the page with error and submitted values
    if error_message:
        logger.warning(f"Web UI validation error: {error_message}")
        reference_files = get_valid_reference_files()
        current_config = config_manager.get_all()
        default_gen_params = {  # Pass defaults again for consistency
            "speed_factor": get_gen_default_speed_factor(),
            "cfg_scale": get_gen_default_cfg_scale(),
            "temperature": get_gen_default_temperature(),
            "top_p": get_gen_default_top_p(),
            "cfg_filter_top_k": get_gen_default_cfg_filter_top_k(),
        }
        # Pass back the values the user submitted
        submitted_gen_params = {
            "speed_factor": speed_factor,
            "cfg_scale": cfg_scale,
            "temperature": temperature,
            "top_p": top_p,
            "cfg_filter_top_k": cfg_filter_top_k,
        }

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": error_message,
                "reference_files": reference_files,
                "config": current_config,
                "presets": loaded_presets,
                "default_gen_params": default_gen_params,  # Base defaults
                # Submitted values to repopulate form
                "submitted_text": text,
                "submitted_voice_mode": voice_mode,
                "submitted_clone_file": clone_reference_select,  # Use potentially invalidated value
                "submitted_gen_params": submitted_gen_params,  # Pass submitted params back
                # Ensure other necessary template variables are passed
                "success": None,
                "output_file_url": None,
                "generation_time": None,
            },
        )

    # --- Generation ---
    try:
        monitor.record("Parameters processed")
        # Call the core engine function
        result = generate_speech(
            text=text,
            voice_mode=voice_mode,
            clone_reference_filename=clone_ref_file,
            speed_factor=speed_factor,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            cfg_filter_top_k=cfg_filter_top_k,
            max_tokens=None,  # Use model default for UI simplicity
        )
        monitor.record("Generation complete")

        if result:
            audio_array, sample_rate = result
            output_path_base = get_output_path()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Create a more descriptive filename
            mode_tag = voice_mode
            if voice_mode == "clone" and clone_ref_file:
                safe_ref_name = sanitize_filename(os.path.splitext(clone_ref_file)[0])
                mode_tag = f"clone_{safe_ref_name[:20]}"  # Limit length
            output_filename = (
                f"{mode_tag}_{timestamp}.wav"  # Always save as WAV for simplicity
            )
            output_filepath = os.path.join(output_path_base, output_filename)

            # Save the audio to a WAV file
            saved = save_audio_to_file(audio_array, sample_rate, output_filepath)
            monitor.record("Audio saved")

            if saved:
                output_file_url = (
                    f"/outputs/{output_filename}"  # URL path for browser access
                )
                generation_time = (
                    monitor.events[-1][1] - monitor.start_time
                )  # Time until save complete
                success_message = f"Audio generated successfully!"
                logger.info(f"Web UI generated audio saved to: {output_filepath}")
            else:
                error_message = "Failed to save generated audio file."
                logger.error("Failed to save audio file from web UI request.")
        else:
            error_message = "Speech generation failed (engine returned None)."
            logger.error("Speech generation failed for web UI request.")

    except Exception as e:
        logger.error(f"Error processing web UI TTS request: {e}", exc_info=True)
        error_message = f"An unexpected error occurred: {str(e)}"

    logger.debug(monitor.report())

    # --- Re-render Template with Results ---
    reference_files = get_valid_reference_files()
    current_config = config_manager.get_all()
    default_gen_params = {
        "speed_factor": get_gen_default_speed_factor(),
        "cfg_scale": get_gen_default_cfg_scale(),
        "temperature": get_gen_default_temperature(),
        "top_p": get_gen_default_top_p(),
        "cfg_filter_top_k": get_gen_default_cfg_filter_top_k(),
    }
    # Pass back submitted values to repopulate form correctly
    submitted_gen_params = {
        "speed_factor": speed_factor,
        "cfg_scale": cfg_scale,
        "temperature": temperature,
        "top_p": top_p,
        "cfg_filter_top_k": cfg_filter_top_k,
    }

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "error": error_message,
            "success": success_message,
            "output_file_url": output_file_url,
            "generation_time": f"{generation_time:.2f}" if generation_time else None,
            "reference_files": reference_files,
            "config": current_config,
            "presets": loaded_presets,
            "default_gen_params": default_gen_params,  # Base defaults
            # Pass back submitted values
            "submitted_text": text,
            "submitted_voice_mode": voice_mode,
            "submitted_clone_file": clone_ref_file,  # Pass the validated filename back
            "submitted_gen_params": submitted_gen_params,  # Pass submitted params back
        },
    )


# --- Reference Audio Upload Endpoint ---
@app.post(
    "/upload_reference", tags=["Web UI Helpers"], summary="Upload reference audio files"
)
async def upload_reference_audio(files: List[UploadFile] = File(...)):
    """Handles uploading of reference audio files (.wav, .mp3) for voice cloning."""
    logger.info(f"Received request to upload {len(files)} reference audio file(s).")
    ref_path = get_reference_audio_path()
    uploaded_filenames = []
    errors = []
    allowed_mime_types = [
        "audio/wav",
        "audio/mpeg",
        "audio/x-wav",
    ]  # Common WAV/MP3 types
    allowed_extensions = [".wav", ".mp3"]

    for file in files:
        try:
            # Basic validation
            if not file.filename:
                errors.append("Received file with no filename.")
                continue

            # Sanitize filename
            safe_filename = sanitize_filename(file.filename)
            _, ext = os.path.splitext(safe_filename)
            if ext.lower() not in allowed_extensions:
                errors.append(
                    f"File '{file.filename}' has unsupported extension '{ext}'. Allowed: {allowed_extensions}"
                )
                continue

            # Check MIME type (more reliable than extension)
            if file.content_type not in allowed_mime_types:
                errors.append(
                    f"File '{file.filename}' has unsupported content type '{file.content_type}'. Allowed: {allowed_mime_types}"
                )
                continue

            # Construct full save path
            destination_path = os.path.join(ref_path, safe_filename)

            # Prevent overwriting existing files (optional, could add counter)
            if os.path.exists(destination_path):
                # Simple approach: skip if exists
                logger.warning(
                    f"Reference file '{safe_filename}' already exists. Skipping upload."
                )
                # Add to list so UI knows it's available, even if not newly uploaded this time
                if safe_filename not in uploaded_filenames:
                    uploaded_filenames.append(safe_filename)
                continue
                # Alternative: add counter like file_1.wav, file_2.wav

            # Save the file using shutil.copyfileobj for efficiency with large files
            try:
                with open(destination_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                logger.info(f"Successfully saved reference file: {destination_path}")
                uploaded_filenames.append(safe_filename)
            except Exception as save_exc:
                errors.append(f"Failed to save file '{safe_filename}': {save_exc}")
                logger.error(
                    f"Failed to save uploaded file '{safe_filename}' to '{destination_path}': {save_exc}",
                    exc_info=True,
                )
            finally:
                # Ensure the UploadFile resource is closed
                await file.close()

        except Exception as e:
            errors.append(
                f"Error processing file '{getattr(file, 'filename', 'unknown')}': {e}"
            )
            logger.error(
                f"Unexpected error processing uploaded file: {e}", exc_info=True
            )
            # Ensure file is closed even if other errors occur
            if file:
                await file.close()

    # Get the updated list of all valid files in the directory
    updated_file_list = get_valid_reference_files()

    response_data = {
        "message": f"Processed {len(files)} file(s).",
        "uploaded_files": uploaded_filenames,  # List of successfully saved *new* files this request
        "all_reference_files": updated_file_list,  # Complete current list
        "errors": errors,
    }

    status_code = (
        200 if not errors or len(errors) < len(files) else 400
    )  # OK if at least one succeeded, else Bad Request
    if errors:
        logger.warning(f"Upload completed with errors: {errors}")

    return JSONResponse(content=response_data, status_code=status_code)


# --- Health Check Endpoint ---
@app.get("/health", tags=["Server Status"], summary="Check server health")
async def health_check():
    """Basic health check, indicates if the server is running and if the model is loaded."""
    # MODEL_LOADED is updated by the engine's load_model function
    return {"status": "healthy", "model_loaded": MODEL_LOADED}


# --- Main Execution ---
if __name__ == "__main__":
    host = get_host()
    port = get_port()
    logger.info(f"Starting Dia TTS server on {host}:{port}")
    logger.info(f"Model Repository: {get_model_repo_id()}")
    logger.info(f"Model Config File: {get_model_config_filename()}")
    logger.info(f"Model Weights File: {get_model_weights_filename()}")
    logger.info(f"Model Cache Path: {get_model_cache_path()}")
    logger.info(f"Reference Audio Path: {get_reference_audio_path()}")
    logger.info(f"Output Path: {get_output_path()}")
    logger.info(
        f"Web UI available at http://{host if host != '0.0.0.0' else 'localhost'}:{port}/"
    )
    logger.info(
        f"API Docs available at http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs"
    )

    # Ensure UI directory and index.html exist for UI
    ui_dir = "ui"
    index_file = os.path.join(ui_dir, "index.html")
    if not os.path.isdir(ui_dir) or not os.path.isfile(index_file):
        logger.warning(
            f"'{ui_dir}' directory or '{index_file}' not found. Web UI may not work."
        )
        # Optionally create dummy files/dirs if needed for startup
        os.makedirs(ui_dir, exist_ok=True)
        if not os.path.isfile(index_file):
            try:
                with open(index_file, "w") as f:
                    f.write(
                        "<html><body>Web UI template missing. See project source for index.html.</body></html>"
                    )
                logger.info(f"Created dummy {index_file}.")
            except Exception as e:
                logger.error(f"Failed to create dummy {index_file}: {e}")

    # Run Uvicorn server
    uvicorn.run(
        "server:app",  # Use the format 'module:app_instance'
        host=host,
        port=port,
        reload=True,  # Enable auto-reload for development
        reload_dirs=[".", "ui"],  # Watch project root and UI directory
        reload_includes=[
            "*.py",
            "*.html",
            "*.css",
            "*.js",
            ".env",
            "*.yaml",
        ],  # Watch relevant file types
        lifespan="on",  # Use the lifespan context manager
        # workers=1 # Keep workers=1 when using reload=True
    )
