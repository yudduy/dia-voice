# config.py
# Configuration management for Dia TTS server

import os
import logging
from dotenv import load_dotenv, find_dotenv, set_key
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Default configuration values (used if not found in .env or environment)
DEFAULT_CONFIG = {
    # Server Settings
    "HOST": "0.0.0.0",
    "PORT": "8003",
    # Model Source Settings
    "DIA_MODEL_REPO_ID": "ttj/dia-1.6b-safetensors",  # Default to safetensors repo
    "DIA_MODEL_CONFIG_FILENAME": "config.json",  # Standard config filename
    "DIA_MODEL_WEIGHTS_FILENAME": "dia-v0_1_bf16.safetensors",  # Default to BF16 weights
    # Path Settings
    "DIA_MODEL_CACHE_PATH": "./model_cache",
    "REFERENCE_AUDIO_PATH": "./reference_audio",
    "OUTPUT_PATH": "./outputs",
    # Default Generation Parameters (can be overridden by user in UI/API)
    # These are saved to .env via the UI's "Save Generation Defaults" button
    "GEN_DEFAULT_SPEED_FACTOR": "0.90",  # Default speed slightly slower
    "GEN_DEFAULT_CFG_SCALE": "3.0",
    "GEN_DEFAULT_TEMPERATURE": "1.3",
    "GEN_DEFAULT_TOP_P": "0.95",
    "GEN_DEFAULT_CFG_FILTER_TOP_K": "35",
}


class ConfigManager:
    """Manages configuration for the TTS server with .env file support."""

    def __init__(self):
        """Initialize the configuration manager."""
        self.config = {}
        self.env_file = find_dotenv()

        if not self.env_file:
            self.env_file = os.path.join(os.getcwd(), ".env")
            logger.info(
                f"No .env file found, creating one with defaults at {self.env_file}"
            )
            self._create_default_env_file()
        else:
            logger.info(f"Loading configuration from: {self.env_file}")

        self.reload()

    def _create_default_env_file(self):
        """Create a default .env file with default values."""
        try:
            with open(self.env_file, "w") as f:
                for key, value in DEFAULT_CONFIG.items():
                    f.write(f"{key}={value}\n")
            logger.info("Created default .env file")
        except Exception as e:
            logger.error(f"Failed to create default .env file: {e}")

    def reload(self):
        """Reload configuration from .env file and environment variables."""
        load_dotenv(self.env_file, override=True)
        loaded_config = {}
        for key, default_value in DEFAULT_CONFIG.items():
            loaded_config[key] = os.environ.get(key, default_value)
        self.config = loaded_config
        logger.info("Configuration loaded/reloaded.")
        logger.debug(f"Current config: {self.config}")
        return self.config

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value in memory (does not save automatically)."""
        self.config[key] = value
        logger.debug(f"Configuration value set in memory: {key}={value}")

    def save(self) -> bool:
        """Save the current in-memory configuration to the .env file."""
        if not self.env_file:
            logger.error("Cannot save configuration, .env file path not set.")
            return False
        try:
            for key in DEFAULT_CONFIG.keys():
                if key not in self.config:
                    logger.warning(
                        f"Key '{key}' missing from current config, adding default value before saving."
                    )
                    self.config[key] = DEFAULT_CONFIG[key]
            for key, value in self.config.items():
                if key in DEFAULT_CONFIG:
                    set_key(self.env_file, key, str(value))
            logger.info(f"Configuration saved to {self.env_file}")
            return True
        except Exception as e:
            logger.error(
                f"Failed to save configuration to {self.env_file}: {e}", exc_info=True
            )
            return False

    def get_all(self) -> Dict[str, Any]:
        """Get all current configuration values."""
        return self.config.copy()

    def update(self, new_config: Dict[str, Any]) -> None:
        """Update multiple configuration values in memory from a dictionary."""
        updated_keys = []
        for key, value in new_config.items():
            if key in DEFAULT_CONFIG:
                self.config[key] = value
                updated_keys.append(key)
            else:
                logger.warning(
                    f"Attempted to update unknown config key: {key}. Ignoring."
                )
        if updated_keys:
            logger.debug(
                f"Configuration values updated in memory for keys: {updated_keys}"
            )

    def get_int(self, key: str, default: Optional[int] = None) -> int:
        """Get a configuration value as an integer, with error handling."""
        value_str = self.get(key)  # Get value which might be from env (str) or default
        if value_str is None:  # Key not found at all
            if default is not None:
                logger.warning(
                    f"Config key '{key}' not found, using provided default: {default}"
                )
                return default
            else:
                logger.error(
                    f"Mandatory config key '{key}' not found and no default provided. Returning 0."
                )
                return 0  # Or raise error

        try:
            return int(value_str)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid integer value '{value_str}' for config key '{key}', using default: {default}"
            )
            if isinstance(default, int):
                return default
            elif default is None:
                logger.error(
                    f"Cannot parse '{value_str}' as int for key '{key}' and no valid default. Returning 0."
                )
                return 0
            else:  # Default was provided but not an int
                logger.error(
                    f"Invalid default value type for key '{key}'. Cannot parse '{value_str}'. Returning 0."
                )
                return 0

    def get_float(self, key: str, default: Optional[float] = None) -> float:
        """Get a configuration value as a float, with error handling."""
        value_str = self.get(key)
        if value_str is None:
            if default is not None:
                logger.warning(
                    f"Config key '{key}' not found, using provided default: {default}"
                )
                return default
            else:
                logger.error(
                    f"Mandatory config key '{key}' not found and no default provided. Returning 0.0."
                )
                return 0.0

        try:
            return float(value_str)
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid float value '{value_str}' for config key '{key}', using default: {default}"
            )
            if isinstance(default, float):
                return default
            elif default is None:
                logger.error(
                    f"Cannot parse '{value_str}' as float for key '{key}' and no valid default. Returning 0.0."
                )
                return 0.0
            else:
                logger.error(
                    f"Invalid default value type for key '{key}'. Cannot parse '{value_str}'. Returning 0.0."
                )
                return 0.0


# --- Create a singleton instance for global access ---
config_manager = ConfigManager()


# --- Export common getters for easy access ---


# Server Settings
def get_host() -> str:
    """Gets the host address for the server."""
    return config_manager.get("HOST", DEFAULT_CONFIG["HOST"])


def get_port() -> int:
    """Gets the port number for the server."""
    # Ensure default is parsed correctly if get_int fails on env var
    return config_manager.get_int("PORT", int(DEFAULT_CONFIG["PORT"]))


# Model Source Settings
def get_model_repo_id() -> str:
    """Gets the Hugging Face repository ID for the model."""
    return config_manager.get("DIA_MODEL_REPO_ID", DEFAULT_CONFIG["DIA_MODEL_REPO_ID"])


def get_model_config_filename() -> str:
    """Gets the filename for the model's configuration file within the repo."""
    return config_manager.get(
        "DIA_MODEL_CONFIG_FILENAME", DEFAULT_CONFIG["DIA_MODEL_CONFIG_FILENAME"]
    )


def get_model_weights_filename() -> str:
    """Gets the filename for the model's weights file within the repo."""
    return config_manager.get(
        "DIA_MODEL_WEIGHTS_FILENAME", DEFAULT_CONFIG["DIA_MODEL_WEIGHTS_FILENAME"]
    )


# Path Settings
def get_model_cache_path() -> str:
    """Gets the local directory path for caching downloaded models."""
    return os.path.abspath(
        config_manager.get(
            "DIA_MODEL_CACHE_PATH", DEFAULT_CONFIG["DIA_MODEL_CACHE_PATH"]
        )
    )


def get_reference_audio_path() -> str:
    """Gets the local directory path for storing reference audio files for cloning."""
    return os.path.abspath(
        config_manager.get(
            "REFERENCE_AUDIO_PATH", DEFAULT_CONFIG["REFERENCE_AUDIO_PATH"]
        )
    )


def get_output_path() -> str:
    """Gets the local directory path for saving generated audio outputs."""
    return os.path.abspath(
        config_manager.get("OUTPUT_PATH", DEFAULT_CONFIG["OUTPUT_PATH"])
    )


# Default Generation Parameter Getters
def get_gen_default_speed_factor() -> float:
    """Gets the default speed factor for generation."""
    return config_manager.get_float(
        "GEN_DEFAULT_SPEED_FACTOR", float(DEFAULT_CONFIG["GEN_DEFAULT_SPEED_FACTOR"])
    )


def get_gen_default_cfg_scale() -> float:
    """Gets the default CFG scale for generation."""
    return config_manager.get_float(
        "GEN_DEFAULT_CFG_SCALE", float(DEFAULT_CONFIG["GEN_DEFAULT_CFG_SCALE"])
    )


def get_gen_default_temperature() -> float:
    """Gets the default temperature for generation."""
    return config_manager.get_float(
        "GEN_DEFAULT_TEMPERATURE", float(DEFAULT_CONFIG["GEN_DEFAULT_TEMPERATURE"])
    )


def get_gen_default_top_p() -> float:
    """Gets the default top_p for generation."""
    return config_manager.get_float(
        "GEN_DEFAULT_TOP_P", float(DEFAULT_CONFIG["GEN_DEFAULT_TOP_P"])
    )


def get_gen_default_cfg_filter_top_k() -> int:
    """Gets the default CFG filter top_k for generation."""
    return config_manager.get_int(
        "GEN_DEFAULT_CFG_FILTER_TOP_K",
        int(DEFAULT_CONFIG["GEN_DEFAULT_CFG_FILTER_TOP_K"]),
    )
