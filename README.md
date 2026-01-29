# Narrate

Transform text into natural speech. Paste in articles, blogs, papers, or any text and get a high-quality audio version.

## Features

- **Local TTS**: Spark-TTS runs on Apple Silicon (no API key needed)
- **Cloud TTS**: ElevenLabs, OpenAI for highest quality
- **No Backend Storage**: API keys stored in browser only

## Quick Start

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and run
uv run python app.py
```

Open http://localhost:3000 in your browser.

## TTS Providers

### MLX-Audio (Local - Mac Only)

Free, runs locally on Apple Silicon. No API key needed.

**Models:**
- **Spark-TTS 0.5B** - Best quality, supports English and Chinese
- **Spark-TTS 0.5B 8-bit** - Faster, uses less memory

```bash
# Install MLX-Audio server (one command)
uv tool install mlx-audio --prerelease=allow --force --with uvicorn --with fastapi --with python-multipart --with webrtcvad --with "setuptools<81"

# Start the TTS server (in a separate terminal)
mlx_audio.server --host 0.0.0.0 --port 8000
```

First run downloads the model (~500MB). Then select "MLX-Audio (Local)" in the UI.

### ElevenLabs

Cloud TTS with natural voices. Get API key at [elevenlabs.io](https://elevenlabs.io).

**Models:** Flash v2.5 (fast), Multilingual v2 (best quality), Turbo v2.5 (balanced)

### OpenAI

Cloud TTS with GPT-4o voices. Get API key at [platform.openai.com](https://platform.openai.com).

**Models:** GPT-4o Mini TTS (best), TTS-1 (fast), TTS-1 HD (high quality)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MLX_AUDIO_URL` | `http://127.0.0.1:8000` | MLX-Audio server URL |
| `ELEVENLABS_API_KEY` | - | ElevenLabs API key (optional) |
| `OPENAI_API_KEY` | - | OpenAI API key (optional) |

API keys can also be entered in the UI and are stored in browser localStorage.

## Development

```bash
# Run unit tests
uv run --extra dev pytest tests/test_api.py -v

# Run integration tests (requires MLX-Audio server running)
uv run --extra dev pytest tests/test_integration.py -v -s

# Run all tests
uv run --extra dev pytest -v
```

## License

MIT
