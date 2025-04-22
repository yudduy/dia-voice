# Dia TTS Server: OpenAI-Compatible API with Web UI

**Self-host the powerful [Nari Labs Dia TTS model](https://github.com/nari-labs/dia) with this enhanced FastAPI server! Features an intuitive Web UI, flexible API endpoints (including OpenAI-compatible `/v1/audio/speech`), support for realistic dialogue generation (`[S1]`/`[S2]`) and voice cloning.**

Defaults to efficient BF16 SafeTensors for reduced VRAM and faster inference, with support for original `.pth` weights. Runs accelerated on NVIDIA GPUs (CUDA) with CPU fallback.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-FastAPI-green.svg?style=for-the-badge)](https://fastapi.tiangolo.com/)
[![Model Format](https://img.shields.io/badge/Weights-SafeTensors%20/%20pth-orange.svg?style=for-the-badge)](https://github.com/huggingface/safetensors)
<!-- Add other relevant badges if applicable -->

<div align="center">
  <img src="screenshot-dark.png" alt="Dia TTS Server Web UI - Dark Mode" width="49%" />
  <img src="screenshot-light.png" alt="Dia TTS Server Web UI - Light Mode" width="49%" />
</div>

---

## 🗣️ Overview: Enhanced Dia TTS Access

The original [Dia model by Nari Labs](https://github.com/nari-labs/dia) provides incredible capabilities for generating realistic dialogue, complete with speaker turns and non-verbal sounds like `(laughs)` or `(sighs)`. This project builds upon that foundation by providing a robust **[FastAPI](https://fastapi.tiangolo.com/) server** that makes Dia significantly easier to use and integrate.

We solve the complexity of setting up and running the model by offering:

*   An **OpenAI-compatible API endpoint**, allowing you to use Dia TTS with tools expecting OpenAI's API structure.
*   A **modern Web UI** for easy experimentation, preset loading, reference audio management, and generation parameter tuning.
*   Support for both original `.pth` weights and modern, secure **[SafeTensors](https://github.com/huggingface/safetensors)**, defaulting to a **BF16 SafeTensors** version which uses roughly half the VRAM and offers improved speed.
*   Automatic **GPU (CUDA) acceleration** detection with fallback to CPU.
*   Simple configuration via an `.env` file.

This server is your gateway to leveraging Dia's advanced TTS capabilities seamlessly.

## ✅ Features

*   **Core Dia Capabilities (via [Nari Labs Dia](https://github.com/nari-labs/dia)):**
    *   🗣️ Generate multi-speaker dialogue using `[S1]` / `[S2]` tags.
    *   😂 Include non-verbal sounds like `(laughs)`, `(sighs)`, `(clears throat)`.
    *   🎭 Perform voice cloning using reference audio prompts.
*   **Enhanced Server & API:**
    *   ⚡ Built with the high-performance **[FastAPI](https://fastapi.tiangolo.com/)** framework.
    *   🤖 **OpenAI-Compatible API Endpoint** (`/v1/audio/speech`) for easy integration.
    *   ⚙️ **Custom API Endpoint** (`/tts`) exposing all Dia generation parameters.
    *   📄 Interactive API documentation via Swagger UI (`/docs`).
    *   🩺 Health check endpoint (`/health`).
*   **Intuitive Web User Interface:**
    *   🖱️ Modern, easy-to-use interface built with Jinja2 and Tailwind CSS.
    *   💡 **Presets:** Load example text and settings with one click.
    *   🎤 **Reference Audio Upload:** Easily upload `.wav`/`.mp3` files for voice cloning directly from the UI.
    *   🎛️ **Parameter Control:** Adjust generation settings (CFG Scale, Temperature, Speed, etc.) via sliders.
    *   💾 **Configuration Management:** View and save server settings and default generation parameters directly in the UI (updates `.env` file).
    *   🌓 **Light/Dark Mode:** Toggle between themes with preference saved locally.
    *   🔊 **Audio Player:** Integrated waveform player ([WaveSurfer.js](https://wavesurfer.xyz/)) for generated audio with download option.
*   **Flexible & Efficient Model Handling:**
    *   🔒 Supports loading secure **`.safetensors`** weights (default).
    *   💾 Supports loading original **`.pth`** weights.
    *   🚀 Defaults to **BF16 SafeTensors** for reduced memory footprint (~half size) and potentially faster inference. (Credit: [ttj/dia-1.6b-safetensors](https://huggingface.co/ttj/dia-1.6b-safetensors))
    *   🔄 Easily switch between model formats/versions via `.env` configuration.
    *   ☁️ Downloads models automatically from [Hugging Face Hub](https://huggingface.co/).
*   **Performance & Configuration:**
    *   💻 **GPU Acceleration:** Automatically uses NVIDIA CUDA if available, falls back to CPU.
    *   ⚙️ Simple configuration via `.env` file.
    *   📦 Uses standard Python virtual environments.

## 🔩 System Prerequisites

*   **Operating System:** Windows 10/11 (64-bit) or Linux (Debian/Ubuntu recommended).
*   **Python:** Version 3.10 or later ([Download](https://www.python.org/downloads/)).
*   **Git:** For cloning the repository ([Download](https://git-scm.com/downloads)).
*   **Internet:** For downloading dependencies and models.
*   **(Optional but HIGHLY Recommended for Performance):**
    *   **NVIDIA GPU:** CUDA-compatible (Maxwell architecture or newer). Check [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus). BF16 model needs ~5-6GB VRAM, full precision ~10GB+.
    *   **NVIDIA Drivers:** Latest version for your GPU/OS ([Download](https://www.nvidia.com/Download/index.aspx)).
    *   **CUDA Toolkit:** Compatible version (e.g., 11.8, 12.1) matching the PyTorch build you install.
*   **(Linux Only):**
    *   `libsndfile1`: Audio library needed by `soundfile`. Install via package manager (e.g., `sudo apt install libsndfile1`).

## 💻 Installation and Setup

Follow these steps carefully to get the server running.

**1. Clone the Repository**
```bash
git clone https://github.com/devnen/dia-tts-server.git
cd dia-tts-server
```

**2. Set up Python Virtual Environment**

Using a virtual environment is crucial!

*   **Windows (PowerShell):**
    ```powershell
    # In the dia-tts-server directory
    python -m venv venv
    .\venv\Scripts\activate
    # Your prompt should now start with (venv)
    ```

*   **Linux (Bash - Debian/Ubuntu Example):**
    ```bash
    # Ensure prerequisites are installed
    sudo apt update && sudo apt install python3 python3-venv python3-pip libsndfile1 -y

    # In the dia-tts-server directory
    python3 -m venv venv
    source venv/bin/activate
    # Your prompt should now start with (venv)
    ```

**3. Install Dependencies**

Make sure your virtual environment is activated (`(venv)` prefix visible).

```bash
# Upgrade pip (recommended)
pip install --upgrade pip

# Install project requirements
pip install -r requirements.txt
```

⭐ **Important:** This installs the *CPU-only* version of PyTorch by default. If you have an NVIDIA GPU, proceed to Step 4 **before** running the server for GPU acceleration.

**4. NVIDIA Driver and CUDA Setup (for GPU Acceleration)**

Skip this step if you only have a CPU.

*   **Step 4a: Check/Install NVIDIA Drivers**
    *   Run `nvidia-smi` in your terminal/command prompt.
    *   If it works, note the **CUDA Version** listed (e.g., 12.1, 11.8). This is the *maximum* your driver supports.
    *   If it fails, download and install the latest drivers from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx) and **reboot**. Verify with `nvidia-smi` again.

*   **Step 4b: Install PyTorch with CUDA Support**
    *   Go to the [Official PyTorch Website](https://pytorch.org/get-started/locally/).
    *   Use the configuration tool: Select **Stable**, **Windows/Linux**, **Pip**, **Python**, and the **CUDA version** that is **equal to or lower** than the one shown by `nvidia-smi` (e.g., if `nvidia-smi` shows 12.4, choose CUDA 12.1).
    *   Copy the generated command (it will include `--index-url https://download.pytorch.org/whl/cuXXX`).
    *   **In your activated `(venv)`:**
        ```bash
        # Uninstall the CPU version first!
        pip uninstall torch torchvision torchaudio -y

        # Paste and run the command copied from the PyTorch website
        # Example (replace with your actual command):
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```

*   **Step 4c: Verify PyTorch CUDA Installation**
    *   In your activated `(venv)`, run `python`.
    *   Inside the Python interpreter:
        ```python
        import torch
        print(f"PyTorch version: {torch.__version__}") # Should show +cuXXX
        print(f"CUDA available: {torch.cuda.is_available()}") # MUST be True
        if torch.cuda.is_available(): print(f"Device name: {torch.cuda.get_device_name(0)}")
        exit()
        ```
    *   If `CUDA available:` is `False`, double-check driver installation and the PyTorch install command.

## ⚙️ Configuration

Configure the server using a `.env` file in the project root (`dia-tts-server/.env`). Create this file if it doesn't exist. Values here override defaults from `config.py`.

**Current Default Configuration (from your `.env`):**

The server will currently use the following settings based on your provided `.env` file (or defaults if a key is missing):

*   **Model:** BF16 SafeTensors from `ttj/dia-1.6b-safetensors`
*   **Host:** `0.0.0.0`
*   **Port:** `8003`
*   **Paths:** `./model_cache`, `./reference_audio`, `./outputs`

**Example `.env` File Content (Reflecting your current setup):**

```dotenv
# .env - Current Active Configuration

# --- Server Settings ---
HOST=0.0.0.0
PORT=8003

# --- Dia Model Settings (Using BF16 SafeTensors) ---
DIA_MODEL_REPO_ID=ttj/dia-1.6b-safetensors
DIA_MODEL_CONFIG_FILENAME=config.json
DIA_MODEL_WEIGHTS_FILENAME=dia-v0_1_bf16.safetensors

# --- File Paths ---
DIA_MODEL_CACHE_PATH=./model_cache
REFERENCE_AUDIO_PATH=./reference_audio
OUTPUT_PATH=./outputs

# --- Generation Defaults (Loaded by UI, saved via UI button) ---
# GEN_DEFAULT_SPEED_FACTOR=0.90
# GEN_DEFAULT_CFG_SCALE=3.0
# GEN_DEFAULT_TEMPERATURE=1.3
# GEN_DEFAULT_TOP_P=0.95
# GEN_DEFAULT_CFG_FILTER_TOP_K=35
```

**To Use a Different Model:**

*   **Full Precision SafeTensors:**
    ```dotenv
    DIA_MODEL_REPO_ID=ttj/dia-1.6b-safetensors
    DIA_MODEL_WEIGHTS_FILENAME=dia-v0_1.safetensors
    ```
*   **Original Nari Labs `.pth` Model:**
    ```dotenv
    DIA_MODEL_REPO_ID=nari-labs/Dia-1.6B
    DIA_MODEL_WEIGHTS_FILENAME=dia-v0_1.pth
    ```

⭐ **Remember:** You **must restart the server** (`python server.py`) after changing the `.env` file! See `config.py` for all available options and their internal defaults.

## ▶️ Running the Server

1.  Activate the virtual environment: `source venv/bin/activate` (Linux) or `.\venv\Scripts\activate` (Windows).
2.  Run the server:
    ```bash
    python server.py
    ```
3.  Access the UI: Open `http://localhost:PORT` (e.g., `http://localhost:8003`) in your browser.
4.  Access API Docs: Open `http://localhost:PORT/docs`.
5.  Stop the server: Press `CTRL+C` in the terminal.

## 💡 Usage

### Web UI (`http://localhost:PORT`)

The most intuitive way to use the server:

*   **Text Input:** Enter your script with `[S1]`/`[S2]` tags and non-verbals like `(laughs)`. Prepend reference transcript for cloning.
*   **Voice Mode:** Choose `Single / Dialogue` or `Voice Clone`.
*   **Presets:** Click buttons to load examples.
*   **Reference Audio (Clone Mode):** Select an existing `.wav`/`.mp3` or click "Load" to upload new files.
*   **Generation Parameters:** Adjust sliders for speed, CFG, temperature, etc. Save your preferred defaults using the button within this section.
*   **Server Configuration:** View/edit `.env` settings (requires restart after saving).
*   **Generate Speech:** Starts the process. A loading overlay with a Cancel button appears.
*   **Audio Player:** Appears on success with playback/download.

### API Endpoints (`/docs` for details)

*   **`/v1/audio/speech` (POST):** OpenAI-compatible. Send JSON with `input`, `voice` (S1/S2/dialogue/filename.wav), `response_format`, `speed`.
*   **`/tts` (POST):** Custom endpoint. Send JSON with `text`, `voice_mode`, `clone_reference_filename`, `output_format`, and detailed generation parameters (`cfg_scale`, `temperature`, etc.).

## 🔍 Troubleshooting

*   **CUDA Not Available / Slow:** Check NVIDIA drivers (`nvidia-smi`), ensure correct CUDA-enabled PyTorch is installed (Installation Step 4).
*   **Import Errors (`dac`, `safetensors`, `yaml`):** Activate venv, run `pip install -r requirements.txt`. Ensure `descript-audio-codec` is installed.
*   **`libsndfile` Error (Linux):** Run `sudo apt install libsndfile1`.
*   **Model Download Fails:** Check internet, `.env` repo/filenames, [Hugging Face Hub status](https://status.huggingface.co/), cache path permissions.
*   **DAC Model Load Fails:** Ensure `descript-audio-codec` installed correctly. Check logs for `AttributeError` (might indicate version mismatch with `dac.utils.download` expectation).
*   **Reference File Not Found:** Check `REFERENCE_AUDIO_PATH` in `.env`, ensure file exists.
*   **Permission Errors (Saving Files):** Check write permissions for `OUTPUT_PATH` and `REFERENCE_AUDIO_PATH`.
*   **UI Issues:** Clear browser cache, check developer console (F12) for JS errors.
*   **Generation Cancel Button:** This is a "UI Cancel" - it stops the *frontend* from waiting/processing but doesn't instantly halt the backend model inference. Clicking Generate again cancels the previous UI wait.

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue to report bugs or suggest features, or submit a Pull Request for improvements.

## 📜 License

This project is licensed under the **MIT License**.

You can find it here: [https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT)

## 🙏 Acknowledgements

*   **Core Model:** This project heavily relies on the excellent **[Dia TTS model](https://github.com/nari-labs/dia)** developed by **[Nari Labs](https://github.com/nari-labs)**. Their work in creating and open-sourcing the model is greatly appreciated.
*   **SafeTensors Conversion:** Thank you to user **[ttj on Hugging Face](https://huggingface.co/ttj)** for providing the converted **[SafeTensors weights](https://huggingface.co/ttj/dia-1.6b-safetensors)** used as the default in this server.
*   **Core Libraries:**
    *   [FastAPI](https://fastapi.tiangolo.com/)
    *   [Uvicorn](https://www.uvicorn.org/)
    *   [PyTorch](https://pytorch.org/)
    *   [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/index) & [SafeTensors](https://github.com/huggingface/safetensors)
    *   [Descript Audio Codec (DAC)](https://github.com/descriptinc/descript-audio-codec)
    *   [SoundFile](https://python-soundfile.readthedocs.io/) & [libsndfile](http://www.mega-nerd.com/libsndfile/)
    *   [Jinja2](https://jinja.palletsprojects.com/)
    *   [WaveSurfer.js](https://wavesurfer.xyz/)
    *   [Tailwind CSS](https://tailwindcss.com/) (via CDN)

---
