# Narrate

Transform text into natural speech. Paste in articles, blogs, papers, or any text and get a high-quality audio version.

## Quick Start

### 1. Set up Fish Speech (TTS Backend)

Narrate uses [Fish Speech](https://github.com/fishaudio/fish-speech) for text-to-speech. You'll need to run it locally first.

**Option A: Docker (Recommended)**

```bash
# Clone Fish Speech
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# Start the API server
docker compose --profile server up
```

The API will be available at `http://localhost:8080`.

**Option B: Manual Install**

```bash
# Prerequisites (Linux/WSL)
sudo apt install portaudio19-dev libsox-dev ffmpeg

# Clone and install
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# Create environment (using conda)
conda create -n fish-speech python=3.12
conda activate fish-speech
pip install -e .[cu129]  # For NVIDIA GPU

# Or using uv (faster)
uv sync --python 3.12 --extra cu129

# Start API server
python -m tools.api_server \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path "checkpoints/openaudio-s1-mini" \
    --decoder-checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" \
    --decoder-config-name modded_dac_vq
```

> Note: Fish Speech requires ~12GB GPU memory for inference.

### 2. Run Narrate

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open `http://localhost:3000` in your browser.

## Usage

1. Paste your text into the text area
2. Select a voice model
3. Click "Generate Audio"
4. Listen or download the generated audio

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FISH_SPEECH_URL` | `http://127.0.0.1:8080` | Fish Speech API URL |

## Roadmap

- [ ] Voice cloning from reference audio
- [ ] Video support (extract audio tracks)
- [ ] Local model options
- [ ] Book/EPUB import
- [ ] Batch processing for long texts
- [ ] Multiple output formats (MP3, OGG)

## License

MIT
