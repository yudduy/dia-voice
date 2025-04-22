# Dia TTS Server - Technical Documentation

**Version:** 1.0.0
**Date:** 2025-04-22

**Table of Contents:**

1.  [Overview](#1-overview)
2.  [Visual Overview](#2-visual-overview)
    *   [Directory Structure](#21-directory-structure)
    *   [Component Diagram](#22-component-diagram)
3.  [System Prerequisites](#3-system-prerequisites)
4.  [Installation and Setup](#4-installation-and-setup)
    *   [Cloning the Repository](#41-cloning-the-repository)
    *   [Setting up Python Virtual Environment](#42-setting-up-python-virtual-environment)
        *   [Windows Setup](#421-windows-setup)
        *   [Linux Setup (Debian/Ubuntu Example)](#422-linux-setup-debianubuntu-example)
    *   [Installing Dependencies](#43-installing-dependencies)
    *   [NVIDIA Driver and CUDA Setup (Required for GPU Acceleration)](#44-nvidia-driver-and-cuda-setup-required-for-gpu-acceleration)
        *   [Step 1: Check/Install NVIDIA Drivers](#441-step-1-checkinstall-nvidia-drivers)
        *   [Step 2: Install PyTorch with CUDA Support](#442-step-2-install-pytorch-with-cuda-support)
        *   [Step 3: Verify PyTorch CUDA Installation](#443-step-3-verify-pytorch-cuda-installation)
5.  [Configuration](#5-configuration)
    *   [Configuration Files (`.env` and `config.py`)](#51-configuration-files-env-and-configpy)
    *   [Configuration Parameters](#52-configuration-parameters)
6.  [Running the Server](#6-running-the-server)
7.  [Usage](#7-usage)
    *   [Web User Interface (Web UI)](#71-web-user-interface-web-ui)
        *   [Main Generation Form](#711-main-generation-form)
        *   [Presets](#712-presets)
        *   [Voice Cloning](#713-voice-cloning)
        *   [Generation Parameters](#714-generation-parameters)
        *   [Server Configuration (UI)](#715-server-configuration-ui)
        *   [Generated Audio Player](#716-generated-audio-player)
        *   [Theme Toggle](#717-theme-toggle)
    *   [API Endpoints](#72-api-endpoints)
        *   [POST /v1/audio/speech (OpenAI Compatible)](#721-post-v1audiospeech-openai-compatible)
        *   [POST /tts (Custom Parameters)](#722-post-tts-custom-parameters)
        *   [Configuration & Helper Endpoints](#723-configuration--helper-endpoints)
8.  [Troubleshooting](#8-troubleshooting)
9.  [Project Architecture](#9-project-architecture)
10. [License and Disclaimer](#10-license-and-disclaimer)

---

## 1. Overview

The Dia TTS Server provides a backend service and web interface for generating high-fidelity speech, including dialogue with multiple speakers and non-verbal sounds, using the Dia text-to-speech model family (originally from Nari Labs, with support for community conversions like SafeTensors).

This server is built using the FastAPI framework and offers both a RESTful API (including an OpenAI-compatible endpoint) and an interactive web UI powered by Jinja2, Tailwind CSS, and JavaScript. It supports voice cloning via audio prompts and allows configuration of various generation parameters.

**Key Features:**

*   **High-Quality TTS:** Leverages the Dia model for realistic speech synthesis.
*   **Dialogue Generation:** Supports `[S1]` and `[S2]` tags for multi-speaker dialogue.
*   **Non-Verbal Sounds:** Can generate sounds like `(laughs)`, `(sighs)`, etc., when included in the text.
*   **Voice Cloning:** Allows conditioning the output voice on a provided reference audio file.
*   **Flexible Model Loading:** Supports loading models from Hugging Face repositories, including both `.pth` and `.safetensors` formats (defaults to BF16 SafeTensors for efficiency).
*   **API Access:** Provides a custom API endpoint (`/tts`) and an OpenAI-compatible endpoint (`/v1/audio/speech`).
*   **Web Interface:** Offers an easy-to-use UI for text input, parameter adjustment, preset loading, reference audio management, and audio playback.
*   **Configuration:** Server settings, model sources, paths, and default generation parameters are configurable via an `.env` file.
*   **GPU Acceleration:** Utilizes NVIDIA GPUs via CUDA for significantly faster inference when available, falling back to CPU otherwise.

---

## 2. Visual Overview

### 2.1 Directory Structure

```
dia-tts-server/
│
├── .env                  # Local configuration overrides (user-created)
├── config.py             # Default configuration and management class
├── engine.py             # Core model loading and generation logic
├── models.py             # Pydantic models for API requests
├── requirements.txt      # Python dependencies
├── server.py             # Main FastAPI application, API endpoints, UI routes
├── utils.py              # Utility functions (audio encoding, saving, etc.)
│
├── dia/                  # Core Dia model implementation package
│   ├── __init__.py
│   ├── audio.py          # Audio processing helpers (delay, codebook conversion)
│   ├── config.py         # Pydantic models for Dia model architecture config
│   ├── layers.py         # Custom PyTorch layers for the Dia model
│   └── model.py          # Dia model class wrapper (loading, generation)
│
├── static/               # Static assets (e.g., favicon.ico)
│   └── favicon.ico
│
├── ui/                   # Web User Interface files
│   ├── index.html        # Main HTML template (Jinja2)
│   ├── presets.yaml      # Predefined UI examples
│   ├── script.js         # Frontend JavaScript logic
│   └── style.css         # Frontend CSS styling (Tailwind via CDN/build)
│
├── model_cache/          # Default directory for downloaded model files (configurable)
├── outputs/              # Default directory for saved audio output (configurable)
└── reference_audio/      # Default directory for voice cloning reference files (configurable)
```

### 2.2 Component Diagram

```
┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
│ User (Web UI /    │────→ │ FastAPI Server    │────→ │ TTS Engine        │────→ │ Dia Model Wrapper │
│ API Client)       │      │ (server.py)       │      │ (engine.py)       │      │ (dia/model.py)    │
└───────────────────┘      └─────────┬─────────┘      └─────────┬─────────┘      └─────────┬─────────┘
                                     │                          │                          │
                                     │ Uses                     │ Uses                     │ Uses
                                     ▼                          ▼                          ▼
                           ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
                           │ Configuration     │ ←─── │ .env File         │      │ Dia Model Layers  │
                           │ (config.py)       │      └───────────────────┘      │ (dia/layers.py)   │
                           └───────────────────┘                                 └───────────────────┘
                                     │                                                   │ Uses
                                     │ Uses                                                   │
                                     ▼                                                   │
                           ┌───────────────────┐                                         │ Uses
                           │ Utilities         │                                         ▼
                           │ (utils.py)        │                               ┌───────────────────┐
                           └───────────────────┘                               │ PyTorch / CUDA    │
                                     ▲                                         └───────────────────┘
                                     │ Uses                                             │ Uses
                                     │                                                  ▼
┌───────────────────┐      ┌───────────────────┐                           ┌───────────────────┐
│ Web UI Files      │ ←─── │ Jinja2 Templates  │                           │ DAC Model         │
│ (ui/)             │      └───────────────────┘                           │ (descript-audio..)│
└───────────────────┘               ▲                                      └───────────────────┘
                                    │ Renders                                        ▲
                                    │                                                │ Uses
                                    └────────────────────────────────────────────────┘
```

**Diagram Legend:**

*   Boxes represent major components or file groups.
*   Arrows (`→`) indicate primary data flow or control flow.
*   Lines with "Uses" indicate dependencies or function calls.

---

## 3. System Prerequisites

Before installing and running the Dia TTS Server, ensure your system meets the following requirements:

*   **Operating System:**
    *   Windows 10/11 (64-bit)
    *   Linux (Debian/Ubuntu recommended, other distributions may require adjustments)
*   **Python:** Python 3.10 or later (Python 3.10.x recommended based on tracebacks). Ensure Python and Pip are added to your system's PATH.
*   **Version Control:** Git (for cloning the repository).
*   **Internet Connection:** Required for downloading dependencies and model files.
*   **(Optional but Highly Recommended for Performance):**
    *   **NVIDIA GPU:** A CUDA-compatible NVIDIA GPU (Maxwell architecture or newer). Check compatibility [here](https://developer.nvidia.com/cuda-gpus). Sufficient VRAM is needed (BF16 model requires ~5-6GB, full precision ~10GB).
    *   **NVIDIA Drivers:** Latest appropriate drivers for your GPU and OS.
    *   **CUDA Toolkit:** Version compatible with the chosen PyTorch build (e.g., 11.8, 12.1). See [Section 4.4](#44-nvidia-driver-and-cuda-setup-required-for-gpu-acceleration).
*   **(Linux System Libraries):**
    *   `libsndfile1`: Required by the `soundfile` Python library for audio I/O. Install using your package manager (e.g., `sudo apt install libsndfile1` on Debian/Ubuntu).

---

## 4. Installation and Setup

Follow these steps to set up the project environment and install necessary dependencies.

### 4.1 Cloning the Repository

Open your terminal or command prompt and navigate to the directory where you want to store the project. Then, clone the repository:

```bash
git clone https://github.com/devnen/dia-tts-server.git # Replace with the actual repo URL if different
cd dia-tts-server
```

### 4.2 Setting up Python Virtual Environment

Using a virtual environment is strongly recommended to isolate project dependencies.

#### 4.2.1 Windows Setup

1.  **Open PowerShell or Command Prompt** in the project directory (`dia-tts-server`).
2.  **Create the virtual environment:**
    ```powershell
    python -m venv venv
    ```
3.  **Activate the virtual environment:**
    ```powershell
    .\venv\Scripts\activate
    ```
    Your terminal prompt should now be prefixed with `(venv)`.

#### 4.2.2 Linux Setup (Debian/Ubuntu Example)

1.  **Install prerequisites (if not already present):**
    ```bash
    sudo apt update
    sudo apt install python3 python3-venv python3-pip libsndfile1 -y
    ```
2.  **Open your terminal** in the project directory (`dia-tts-server`).
3.  **Create the virtual environment:**
    ```bash
    python3 -m venv venv
    ```
4.  **Activate the virtual environment:**
    ```bash
    source venv/bin/activate
    ```
    Your terminal prompt should now be prefixed with `(venv)`.

### 4.3 Installing Dependencies

With your virtual environment activated (`(venv)` prefix visible), install the required Python packages:

```bash
# Upgrade pip first (optional but good practice)
pip install --upgrade pip

# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

**Note:** This command installs the CPU-only version of PyTorch by default. If you have a compatible NVIDIA GPU and want acceleration, proceed to [Section 4.4](#44-nvidia-driver-and-cuda-setup-required-for-gpu-acceleration) **before** running the server.

### 4.4 NVIDIA Driver and CUDA Setup (Required for GPU Acceleration)

Follow these steps **only if you have a compatible NVIDIA GPU** and want faster inference.

#### 4.4.1 Step 1: Check/Install NVIDIA Drivers

1.  **Check Existing Driver:** Open Command Prompt (Windows) or Terminal (Linux) and run:
    ```bash
    nvidia-smi
    ```
2.  **Interpret Output:**
    *   If the command runs successfully, note the **Driver Version** and the **CUDA Version** listed in the top right corner. This CUDA version is the *maximum* supported by your current driver.
    *   If the command fails ("not recognized"), you need to install or update your NVIDIA drivers.
3.  **Install/Update Drivers:** Go to the [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx) page. Select your GPU model and OS, then download and install the latest recommended driver (Game Ready or Studio). **Reboot your computer** after installation. Run `nvidia-smi` again to confirm it works.

#### 4.4.2 Step 2: Install PyTorch with CUDA Support

1.  **Go to PyTorch Website:** Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).
2.  **Configure:** Select:
    *   **PyTorch Build:** Stable
    *   **Your OS:** Windows or Linux
    *   **Package:** Pip
    *   **Language:** Python
    *   **Compute Platform:** Choose the CUDA version **equal to or lower than** the version reported by `nvidia-smi`. For example, if `nvidia-smi` shows `CUDA Version: 12.4`, select `CUDA 12.1`. If it shows `11.8`, select `CUDA 11.8`. **Do not select a version higher than your driver supports.** (CUDA 12.1 or 11.8 are common stable choices).
3.  **Copy Command:** Copy the generated installation command. It will look similar to:
    ```bash
    # Example for CUDA 12.1 (Windows/Linux):
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    # Example for CUDA 11.8 (Windows/Linux):
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    *(Use `pip` instead of `pip3` if that's your command)*
4.  **Install in Activated venv:**
    *   Ensure your `(venv)` is active.
    *   **Uninstall CPU PyTorch first:**
        ```bash
        pip uninstall torch torchvision torchaudio -y
        ```
    *   **Paste and run the copied command** from the PyTorch website.

#### 4.4.3 Step 3: Verify PyTorch CUDA Installation

1.  With the `(venv)` still active, start a Python interpreter:
    ```bash
    python
    ```
2.  Run the following Python code:
    ```python
    import torch
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA version used by PyTorch: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device index: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA not available to PyTorch. Ensure drivers and CUDA-enabled PyTorch are installed correctly.")
    exit()
    ```
3.  If `CUDA available:` shows `True`, the setup was successful. If `False`, review driver installation and the PyTorch installation command.

---

## 5. Configuration

The server's behavior, including model selection, paths, and default generation parameters, is controlled via configuration settings.

### 5.1 Configuration Files (`.env` and `config.py`)

*   **`config.py`:** Defines the *default* values for all configuration parameters in the `DEFAULT_CONFIG` dictionary. It also contains the `ConfigManager` class and getter functions used by the application.
*   **`.env` File:** This file, located in the project root directory (`dia-tts-server/.env`), allows you to *override* the default values. Create this file if it doesn't exist. Settings are defined as `KEY=VALUE` pairs, one per line. The server reads this file on startup using `python-dotenv`.

**Priority:** Values set in the `.env` file take precedence over the defaults in `config.py`. Environment variables set directly in your system also override `.env` file values (though using `.env` is generally recommended for project-specific settings).

### 5.2 Configuration Parameters

The following parameters can be set in your `.env` file:

| Parameter Name (in `.env`)         | Default Value (`config.py`)        | Description                                                                                                | Example `.env` Value                 |
| :--------------------------------- | :--------------------------------- | :--------------------------------------------------------------------------------------------------------- | :----------------------------------- |
| **Server Settings**                |                                    |                                                                                                            |                                      |
| `HOST`                             | `0.0.0.0`                          | The network interface address the server listens on. `0.0.0.0` makes it accessible on your local network. | `127.0.0.1` (localhost only)         |
| `PORT`                             | `8003`                             | The port number the server listens on.                                                                     | `8080`                               |
| **Model Source Settings**          |                                    |                                                                                                            |                                      |
| `DIA_MODEL_REPO_ID`                | `ttj/dia-1.6b-safetensors`         | The Hugging Face repository ID containing the model files.                                                 | `nari-labs/Dia-1.6B`                 |
| `DIA_MODEL_CONFIG_FILENAME`        | `config.json`                      | The filename of the model's configuration JSON within the repository.                                      | `config.json`                        |
| `DIA_MODEL_WEIGHTS_FILENAME`       | `dia-v0_1_bf16.safetensors`        | The filename of the model weights file (`.safetensors` or `.pth`) within the repository to load.           | `dia-v0_1.safetensors` or `dia-v0_1.pth` |
| **Path Settings**                  |                                    |                                                                                                            |                                      |
| `DIA_MODEL_CACHE_PATH`             | `./model_cache`                    | Local directory to store downloaded model files. Relative paths are based on the project root.             | `/path/to/shared/cache`              |
| `REFERENCE_AUDIO_PATH`             | `./reference_audio`                | Local directory to store reference audio files (`.wav`, `.mp3`) used for voice cloning.                      | `./voices`                           |
| `OUTPUT_PATH`                      | `./outputs`                        | Local directory where generated audio files from the Web UI are saved.                                     | `./generated_speech`                 |
| **Default Generation Parameters**  |                                    | *(These set the initial UI values and can be saved via the UI)*                                            |                                      |
| `GEN_DEFAULT_SPEED_FACTOR`         | `0.90`                             | Default playback speed factor applied *after* generation (UI slider initial value).                        | `1.0`                                |
| `GEN_DEFAULT_CFG_SCALE`            | `3.0`                              | Default Classifier-Free Guidance scale (UI slider initial value).                                          | `2.5`                                |
| `GEN_DEFAULT_TEMPERATURE`          | `1.3`                              | Default sampling temperature (UI slider initial value).                                                    | `1.2`                                |
| `GEN_DEFAULT_TOP_P`                | `0.95`                             | Default nucleus sampling probability (UI slider initial value).                                            | `0.9`                                |
| `GEN_DEFAULT_CFG_FILTER_TOP_K`     | `35`                               | Default Top-K value for CFG filtering (UI slider initial value).                                           | `40`                                 |

**Example `.env` File (Using Original Nari Labs Model):**

```dotenv
# .env
# Example configuration to use the original Nari Labs model

HOST=0.0.0.0
PORT=8003

DIA_MODEL_REPO_ID=nari-labs/Dia-1.6B
DIA_MODEL_CONFIG_FILENAME=config.json
DIA_MODEL_WEIGHTS_FILENAME=dia-v0_1.pth

# Keep other paths as default or specify custom ones
# DIA_MODEL_CACHE_PATH=./model_cache
# REFERENCE_AUDIO_PATH=./reference_audio
# OUTPUT_PATH=./outputs

# Keep default generation parameters or override them
# GEN_DEFAULT_SPEED_FACTOR=0.90
# GEN_DEFAULT_CFG_SCALE=3.0
# GEN_DEFAULT_TEMPERATURE=1.3
# GEN_DEFAULT_TOP_P=0.95
# GEN_DEFAULT_CFG_FILTER_TOP_K=35
```

**Important:** You must **restart the server** after making changes to the `.env` file for them to take effect.

---

## 6. Running the Server

1.  **Activate Virtual Environment:** Ensure your virtual environment is activated (`(venv)` prefix).
    *   Windows: `.\venv\Scripts\activate`
    *   Linux: `source venv/bin/activate`
2.  **Navigate to Project Root:** Make sure your terminal is in the `dia-tts-server` directory.
3.  **Run the Server:**
    ```bash
    python server.py
    ```
4.  **Server Output:** You should see log messages indicating the server is starting, including:
    *   The configuration being used (repo ID, filenames, paths).
    *   The device being used (CPU or CUDA).
    *   Model loading progress (downloading if necessary).
    *   Confirmation that the server is running (e.g., `Uvicorn running on http://0.0.0.0:8003`).
    *   URLs for accessing the Web UI and API Docs.

5.  **Accessing the Server:**
    *   **Web UI:** Open your web browser and go to `http://localhost:PORT` (e.g., `http://localhost:8003` if using the default port). If running on a different machine or VM, replace `localhost` with the server's IP address.
    *   **API Docs:** Access the interactive API documentation (Swagger UI) at `http://localhost:PORT/docs`.
6.  **Stopping the Server:** Press `CTRL+C` in the terminal where the server is running.

**Auto-Reload:** The server is configured to run with `reload=True`. This means Uvicorn will automatically restart the server if it detects changes in `.py`, `.html`, `.css`, `.js`, `.env`, or `.yaml` files within the project or `ui` directory. This is useful for development but should generally be disabled in production.

---

## 7. Usage

The Dia TTS Server can be used via its Web UI or its API endpoints.

### 7.1 Web User Interface (Web UI)

Access the UI by navigating to the server's base URL (e.g., `http://localhost:8003`).

#### 7.1.1 Main Generation Form

*   **Text to speak:** Enter the text you want to synthesize.
    *   Use `[S1]` and `[S2]` tags to indicate speaker turns for dialogue.
    *   Include non-verbal cues like `(laughs)`, `(sighs)`, `(clears throat)` directly in the text where desired.
    *   For voice cloning, **prepend the exact transcript** of the selected reference audio before the text you want generated (e.g., `[S1] Reference transcript text. [S1] This is the new text to generate in the cloned voice.`).
*   **Voice Mode:** Select the desired generation mode:
    *   **Single / Dialogue (Use [S1]/[S2]):** Use this for single-speaker text (you can use `[S1]` or omit tags if the model handles it) or multi-speaker dialogue (using `[S1]` and `[S2]`).
    *   **Voice Clone (from Reference):** Enables voice cloning based on a selected audio file. Requires selecting a file below and prepending its transcript to the text input.
*   **Generate Speech Button:** Submits the text and settings to the server to start generation.

#### 7.1.2 Presets

*   Located below the Voice Mode selection.
*   Clicking a preset button (e.g., "Standard Dialogue", "Expressive Narration") will automatically populate the "Text to speak" area and the "Generation Parameters" sliders with predefined values, demonstrating different use cases.

#### 7.1.3 Voice Cloning

*   This section appears only when "Voice Clone" mode is selected.
*   **Reference Audio File Dropdown:** Lists available `.wav` and `.mp3` files found in the configured `REFERENCE_AUDIO_PATH`. Select the file whose voice you want to clone. Remember to prepend its transcript to the main text input.
*   **Load Button:** Click this to open your system's file browser. You can select one or more `.wav` or `.mp3` files to upload. The selected files will be copied to the server's `REFERENCE_AUDIO_PATH`, and the dropdown list will refresh automatically. The first newly uploaded file will be selected in the dropdown.

#### 7.1.4 Generation Parameters

*   Expand this section to fine-tune the generation process. These values correspond to the parameters used by the underlying Dia model.
*   **Sliders:** Adjust Speed Factor, CFG Scale, Temperature, Top P, and CFG Filter Top K. The current value is displayed next to the label.
*   **Save Generation Defaults Button:** Saves the *current* values of these sliders to the `.env` file (as `GEN_DEFAULT_...` keys). These saved values will become the default settings loaded into the UI the next time the server starts.

#### 7.1.5 Server Configuration (UI)

*   Expand this section to view and modify server-level settings stored in the `.env` file.
*   **Fields:** Edit Model Repo ID, Config/Weights Filenames, Cache/Reference/Output Paths, Host, and Port.
*   **Save Server Configuration Button:** Saves the values currently shown in these fields to the `.env` file. **A server restart is required** for most of these changes (especially model source or paths) to take effect.
*   **Restart Server Button:** (Appears after saving) Attempts to trigger a server restart. This works best if the server was started with `reload=True` or is managed by a process manager like systemd or Supervisor.

#### 7.1.6 Generated Audio Player

*   Appears below the main form after a successful generation.
*   **Waveform:** Visual representation of the generated audio.
*   **Play/Pause Button:** Controls audio playback.
*   **Download WAV Button:** Downloads the generated audio as a `.wav` file.
*   **Info:** Displays the voice mode used, generation time, and audio duration.

#### 7.1.7 Theme Toggle

*   Located in the top-right navigation bar.
*   Click the Sun/Moon icon to switch between Light and Dark themes. Your preference is saved in your browser's `localStorage`.

### 7.2 API Endpoints

Access the interactive API documentation via the `/docs` path (e.g., `http://localhost:8003/docs`).

#### 7.2.1 POST `/v1/audio/speech` (OpenAI Compatible)

*   **Purpose:** Provides an endpoint compatible with the basic OpenAI TTS API for easier integration with existing tools.
*   **Request Body:** (`application/json`) - Uses the `OpenAITTSRequest` model.
    | Field             | Type                     | Required | Description                                                                                                                               | Example                     |
    | :---------------- | :----------------------- | :------- | :---------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------- |
    | `model`           | string                   | No       | Ignored by this server (always uses Dia). Included for compatibility. Defaults to `dia-1.6b`.                                               | `"dia-1.6b"`                |
    | `input`           | string                   | Yes      | The text to synthesize. Use `[S1]`/`[S2]` tags for dialogue. For cloning, prepend reference transcript.                                    | `"Hello [S1] world."`       |
    | `voice`           | string                   | No       | Maps to Dia modes. Use `"S1"`, `"S2"`, `"dialogue"`, or the filename of a reference audio (e.g., `"my_ref.wav"`) for cloning. Defaults to `S1`. | `"dialogue"` or `"ref.mp3"` |
    | `response_format` | `"opus"` \| `"wav"`      | No       | Desired audio output format. Defaults to `opus`.                                                                                          | `"wav"`                     |
    | `speed`           | float                    | No       | Playback speed factor (0.5-2.0). Applied *after* generation. Defaults to `1.0`.                                                           | `0.9`                       |
*   **Response:**
    *   **Success (200 OK):** `StreamingResponse` containing the binary audio data (`audio/opus` or `audio/wav`).
    *   **Error:** Standard FastAPI JSON error response (e.g., 400, 404, 500).

#### 7.2.2 POST `/tts` (Custom Parameters)

*   **Purpose:** Allows generation using all specific Dia generation parameters.
*   **Request Body:** (`application/json`) - Uses the `CustomTTSRequest` model.
    | Field                      | Type                                   | Required | Description                                                                                                                               | Default     |
    | :------------------------- | :------------------------------------- | :------- | :---------------------------------------------------------------------------------------------------------------------------------------- | :---------- |
    | `text`                     | string                                 | Yes      | The text to synthesize. Use `[S1]`/`[S2]` tags. Prepend transcript for cloning.                                                           |             |
    | `voice_mode`               | `"dialogue"` \| `"clone"`              | No       | Generation mode. Note: `single_s1`/`single_s2` are handled via `dialogue` mode with appropriate tags in the text.                         | `dialogue`  |
    | `clone_reference_filename` | string \| null                         | No       | Filename of reference audio in `REFERENCE_AUDIO_PATH`. **Required if `voice_mode` is `clone`**.                                           | `null`      |
    | `output_format`            | `"opus"` \| `"wav"`                    | No       | Desired audio output format.                                                                                                              | `opus`      |
    | `max_tokens`               | integer \| null                        | No       | Maximum audio tokens to generate. `null` uses the model's default.                                                                        | `null`      |
    | `cfg_scale`                | float                                  | No       | Classifier-Free Guidance scale.                                                                                                           | `3.0`       |
    | `temperature`              | float                                  | No       | Sampling temperature.                                                                                                                     | `1.3`       |
    | `top_p`                    | float                                  | No       | Nucleus sampling probability.                                                                                                             | `0.95`      |
    | `speed_factor`             | float                                  | No       | Playback speed factor (0.5-2.0). Applied *after* generation.                                                                              | `0.90`      |
    | `cfg_filter_top_k`         | integer                                | No       | Top-K value for CFG filtering.                                                                                                            | `35`        |
*   **Response:**
    *   **Success (200 OK):** `StreamingResponse` containing the binary audio data (`audio/opus` or `audio/wav`).
    *   **Error:** Standard FastAPI JSON error response (e.g., 400, 404, 500).

#### 7.2.3 Configuration & Helper Endpoints

*   **GET `/get_config`:** Returns the current server configuration as JSON.
*   **POST `/save_config`:** Saves server configuration settings provided in the JSON request body to the `.env` file. Requires server restart.
*   **POST `/save_generation_defaults`:** Saves default generation parameters provided in the JSON request body to the `.env` file. Affects UI defaults on next load.
*   **POST `/restart_server`:** Attempts to trigger a server restart (reliability depends on execution environment).
*   **POST `/upload_reference`:** Uploads one or more audio files (`.wav`, `.mp3`) as `multipart/form-data` to the reference audio directory. Returns JSON with status and updated file list.
*   **GET `/health`:** Basic health check endpoint. Returns `{"status": "healthy", "model_loaded": true/false}`.

---

## 8. Troubleshooting

*   **Error: `CUDA available: False` or Slow Performance:**
    *   Verify NVIDIA drivers are installed correctly (`nvidia-smi` command).
    *   Ensure you installed the correct PyTorch version with CUDA support matching your driver (See [Section 4.4](#44-nvidia-driver-and-cuda-setup-required-for-gpu-acceleration)). Reinstall PyTorch using the command from the official website if unsure.
    *   Check if another process is using all GPU VRAM.
*   **Error: `ImportError: No module named 'dac'` (or `safetensors`, `yaml`, etc.):**
    *   Make sure your virtual environment is activated.
    *   Run `pip install -r requirements.txt` again to install missing dependencies.
    *   Specifically for `dac`, ensure you installed `descript-audio-codec` and not a different package named `dac`. Run `pip uninstall dac -y && pip install descript-audio-codec`.
*   **Error: `libsndfile library not found` (or similar `soundfile` error, mainly on Linux):**
    *   Install the system library: `sudo apt update && sudo apt install libsndfile1` (Debian/Ubuntu) or the equivalent for your distribution.
*   **Error: Model Download Fails (e.g., `HTTPError`, `ConnectionError`):**
    *   Check your internet connection.
    *   Verify the `DIA_MODEL_REPO_ID`, `DIA_MODEL_CONFIG_FILENAME`, and `DIA_MODEL_WEIGHTS_FILENAME` in your `.env` file (or defaults in `config.py`) are correct and accessible on Hugging Face Hub.
    *   Check Hugging Face Hub status if multiple downloads fail.
    *   Ensure the cache directory (`DIA_MODEL_CACHE_PATH`) is writable.
*   **Error: `RuntimeError: Failed to load DAC model...`:**
    *   This usually indicates an issue with the `descript-audio-codec` installation or version incompatibility. Ensure it's installed correctly (see `ImportError` above).
    *   Check logs for specific `AttributeError` messages (like missing `utils` or `download`) which might indicate version mismatches between the Dia code's expectation and the installed library. The current code expects `dac.utils.download()`.
*   **Error: `FileNotFoundError` during generation (Reference Audio):**
    *   Ensure the filename selected/provided for voice cloning exists in the configured `REFERENCE_AUDIO_PATH`.
    *   Check that the path in `config.py` or `.env` is correct and the server has permission to read from it.
*   **Error: Cannot Save Output/Reference Files (`PermissionError`, etc.):**
    *   Ensure the directories specified by `OUTPUT_PATH` and `REFERENCE_AUDIO_PATH` exist and the server process has write permissions to them.
*   **Web UI Issues (Buttons don't work, styles missing):**
    *   Clear your browser cache.
    *   Check the browser's developer console (usually F12) for JavaScript errors.
    *   Ensure `ui/script.js` and `ui/style.css` are being loaded correctly (check network tab in developer tools).
*   **Generation Cancel Button Doesn't Stop Process:**
    *   This is expected ("Fake Cancel"). The button currently only prevents the UI from processing the result when it eventually arrives. True cancellation is complex and not implemented. Clicking "Generate" again *will* cancel the *previous UI request's result processing* before starting the new one.

---

## 9. Project Architecture

*   **`server.py`:** The main entry point using FastAPI. Defines API routes, serves the Web UI using Jinja2, handles requests, and orchestrates calls to the engine.
*   **`engine.py`:** Responsible for loading the Dia model (including downloading files via `huggingface_hub`), managing the model instance, preparing inputs for the model's `generate` method based on user requests (handling voice modes), and calling the model's generation function. Also handles post-processing like speed adjustment.
*   **`config.py`:** Manages all configuration settings using default values and overrides from a `.env` file. Provides getter functions for easy access to settings.
*   **`dia/` package:** Contains the core implementation of the Dia model itself.
    *   `model.py`: Defines the `Dia` class, which wraps the underlying PyTorch model (`DiaModel`). It handles loading weights (`.pth` or `.safetensors`), loading the required DAC model, preparing inputs specifically for the `DiaModel` forward pass (including CFG logic), and running the autoregressive generation loop.
    *   `config.py` (within `dia/`): Defines Pydantic models representing the *structure* and hyperparameters of the Dia model architecture (encoder, decoder, data parameters). This is loaded from the `config.json` file associated with the model weights.
    *   `layers.py`: Contains custom PyTorch `nn.Module` implementations used within the `DiaModel` (e.g., Attention blocks, MLP blocks, RoPE).
    *   `audio.py`: Includes helper functions for audio processing specific to the model's tokenization and delay patterns (e.g., `audio_to_codebook`, `codebook_to_audio`, `apply_audio_delay`).
*   **`ui/` directory:** Contains all files related to the Web UI.
    *   `index.html`: The main Jinja2 template.
    *   `script.js`: Frontend JavaScript for interactivity, API calls, theme switching, etc.
    *   `presets.yaml`: Definitions for the UI preset examples.
*   **`utils.py`:** General utility functions, such as audio encoding (`encode_audio`) and saving (`save_audio_to_file`) using the `soundfile` library.
*   **Dependencies:** Relies heavily on `FastAPI`, `Uvicorn`, `PyTorch`, `torchaudio`, `huggingface_hub`, `safetensors`, `descript-audio-codec`, `soundfile`, `PyYAML`, `python-dotenv`, `pydantic`, and `Jinja2`.

---

## 10. License and Disclaimer

*   **License:** This project is licensed under the MIT License.
*   **Disclaimer:** This project offers a high-fidelity speech generation model intended solely for research and educational use. The following uses are **strictly forbidden**:
    *   **Identity Misuse**: Do not produce audio resembling real individuals without permission.
    *   **Deceptive Content**: Do not use this model to generate misleading content (e.g. fake news)
    *   **Illegal or Malicious Use**: Do not use this model for activities that are illegal or intended to cause harm.

    By using this model, you agree to uphold relevant legal standards and ethical responsibilities. The creators **are not responsible** for any misuse and firmly oppose any unethical usage of this technology.

---
