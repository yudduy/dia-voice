# Long Audio Generation Guide for Dia-TTS-Server

This guide explains how to generate longer audio content using the Dia-TTS-Server.

## Overview

When generating audio content longer than ~30 seconds, the standard generation approach can have limitations:
- Memory usage becomes high during processing
- Generation may stall or fail
- Quality may suffer at segment boundaries

The long audio processing system handles these issues by:
1. Intelligently chunking the text into segments
2. Processing each segment individually
3. Combining segments into a seamless final audio file

## API Usage

### REST API

For longer content, use the dedicated long audio endpoint:

```bash
curl -X POST "http://localhost:8003/api/v1/tts/generate_long" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your long text here...",
    "voice_mode": "single_s1",
    "output_format": "mp3",
    "chunk_size": 300
  }' \
  --output long_speech.mp3
```

Parameters:
- `text`: Your input text (required)
- `voice_mode`: "single_s1", "single_s2", "dialogue", "clone", or "predefined" (default: "single_s1")
- `clone_reference_filename`: Reference audio filename when using "clone" or "predefined" mode
- `transcript`: Optional transcript text for voice cloning
- `output_format`: "opus" or "wav" (default: "opus")
- `chunk_size`: Target chunk size in characters (default: 300)
- `cfg_scale`: Classifier-free guidance scale (default: 3.0)
- `temperature`: Sampling temperature (default: 1.3)
- `top_p`: Nucleus sampling probability (default: 0.95)
- `speed_factor`: Speech speed adjustment (default: 0.94)
- `cfg_filter_top_k`: Top-k filter for CFG (default: 35)
- `seed`: Generation seed (-1 for random, default: 42)

### OpenAI-Compatible Endpoint

The standard OpenAI-compatible endpoint automatically switches to the long audio processor for content longer than 30 seconds:

```bash
curl -X POST "http://localhost:8003/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dia-1.6b",
    "input": "Your long text here...",
    "voice": "S1"
  }' \
  --output long_speech.mp3
```

### Command-Line Usage

For batch processing or scripting, use the `generate_long_audio.py` script:

```bash
python generate_long_audio.py --text "Your long text here..." --output output.wav --voice S1
```

Or with a text file:

```bash
python generate_long_audio.py --text-file script.txt --output podcast.mp3 --voice my_voice.wav
```

Arguments:
- `--text` or `--text-file`: Input text or path to text file (required, mutually exclusive)
- `--output`: Output audio file path (.wav or .mp3) (required)
- `--voice`: Voice to use: 'S1', 'S2', 'dialogue', or path to reference audio (default: 'S1')
- `--transcript`: Optional transcript for voice cloning
- `--seed`: Random seed (default: 42, -1 for random)
- `--speed`: Speed factor (0.5-2.0, default: 1.0)
- `--temperature`: Temperature (0.1-1.5, default: 1.3)
- `--cfg-scale`: CFG scale (1.0-5.0, default: 3.0)
- `--top-p`: Top-p sampling (0.1-1.0, default: 0.95)
- `--chunk-size`: Maximum characters per chunk (100-1000, default: 300)

## Voice Cloning with Long Audio

To clone a voice for long audio:

1. Upload your reference audio files to the `reference_audio` directory
2. Use the API with:
   ```json
   {
     "voice_mode": "clone",
     "clone_reference_filename": "your_voice.wav",
     "text": "Your long text here..."
   }
   ```

3. Or use the command-line:
   ```bash
   python generate_long_audio.py --text-file script.txt --output output.mp3 --voice your_voice.wav
   ```

## Tips for Best Results

- **Chunk Size**: The default chunk size (300 characters) works well for most content. Adjust if needed:
  - Smaller chunks (100-200): Better for technical content or complex words
  - Larger chunks (400-600): Better for narrative flow in storytelling

- **Speaker Tags**: For dialogue, always use [S1] and [S2] tags consistently:
  ```
  [S1] Hello there, how are you today?
  [S2] I'm doing well, thank you for asking!
  ```

- **Voice Consistency**: For best voice cloning results:
  - Use high-quality reference audio (clear speech, minimal background noise)
  - Keep reference audio 5-10 seconds in length
  - Match the speaking style of your reference in your script

- **Memory Usage**: If you encounter memory issues:
  - Reduce chunk size to process smaller segments
  - Close other applications to free system memory
  - Consider running on a machine with more RAM

## Implementation Details

The long audio system automatically:
- Splits text at natural sentence boundaries
- Maintains speaker tags across chunks
- Optimizes generation for each segment
- Handles memory management between chunks
- Applies consistent post-processing to the final audio

For developers: The core implementation is in `app/utils/long_audio.py`.
