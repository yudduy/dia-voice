# models.py
# Pydantic models for API requests and potentially responses

from pydantic import BaseModel, Field
from typing import Optional, Literal

# --- Request Models ---


class OpenAITTSRequest(BaseModel):
    """Request model compatible with the OpenAI TTS API."""

    model: str = Field(
        default="dia-1.6b",
        description="Model identifier (ignored by this server, always uses Dia). Included for compatibility.",
    )
    input: str = Field(..., description="The text to synthesize.")
    voice: str = Field(
        default="S1",
        description="Voice mode or reference audio filename. Examples: 'S1', 'S2', 'dialogue', 'my_reference.wav'.",
    )
    response_format: Literal["opus", "wav"] = Field(
        default="opus", description="The desired audio output format."
    )
    speed: float = Field(
        default=1.0,
        ge=0.8,
        le=1.2,  # Dia speed factor range seems narrower
        description="Adjusts the speed of the generated audio (0.8 to 1.2).",
    )


class CustomTTSRequest(BaseModel):
    """Request model for the custom /tts endpoint."""

    text: str = Field(
        ...,
        description="The text to synthesize. For 'dialogue' mode, include [S1]/[S2] tags.",
    )
    voice_mode: Literal["dialogue", "single_s1", "single_s2", "clone"] = Field(
        default="single_s1", description="Specifies the generation mode."
    )
    clone_reference_filename: Optional[str] = Field(
        default=None,
        description="Filename of the reference audio within the configured reference path (required if voice_mode is 'clone').",
    )
    output_format: Literal["opus", "wav"] = Field(
        default="opus", description="The desired audio output format."
    )
    # Dia-specific generation parameters
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum number of audio tokens to generate (defaults to model's internal config value).",
    )
    cfg_scale: float = Field(
        default=3.0,
        ge=1.0,
        le=5.0,
        description="Classifier-Free Guidance scale (1.0-5.0).",
    )
    temperature: float = Field(
        default=1.3, ge=1.0, le=1.5, description="Sampling temperature (1.0-1.5)."
    )
    top_p: float = Field(
        default=0.95,
        ge=0.8,
        le=1.0,
        description="Nucleus sampling probability (0.8-1.0).",
    )
    speed_factor: float = Field(
        default=0.94,
        ge=0.8,
        le=1.0,  # Dia's default range seems to be <= 1.0
        description="Adjusts the speed of the generated audio (0.8 to 1.0).",
    )
    cfg_filter_top_k: int = Field(
        default=35, ge=15, le=50, description="Top k filter for CFG guidance (15-50)."
    )


# --- Response Models (Optional, can be simple dicts too) ---


class TTSResponse(BaseModel):
    """Basic response model for successful generation (if returning JSON)."""

    request_id: str
    status: str = "completed"
    generation_time_sec: float
    output_url: Optional[str] = None  # If saving file and returning URL


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str
