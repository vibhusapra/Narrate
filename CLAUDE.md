# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
# Run the application (serves on http://localhost:3000)
uv run python app.py

# Run unit tests (mocked, no external dependencies)
uv run --extra dev pytest tests/test_api.py -v

# Run integration tests (requires MLX-Audio server or API keys)
uv run --extra dev pytest tests/test_integration.py -v -s

# Run all tests
uv run --extra dev pytest -v
```

### Local TTS Setup (MLX-Audio on Mac)

```bash
# Install MLX-Audio server (one-time)
uv tool install mlx-audio --prerelease=allow --force --with uvicorn --with fastapi --with python-multipart --with webrtcvad --with "setuptools<81"

# Start the TTS server (separate terminal)
mlx_audio.server --host 0.0.0.0 --port 8000
```

## Architecture

**Multi-provider TTS API** with three backends:
- **MLX-Audio** - Local TTS on Apple Silicon (Spark-TTS model), returns MP3
- **ElevenLabs** - Cloud TTS, requires API key, returns MP3
- **OpenAI** - Cloud TTS, requires API key, returns WAV

**Key files:**
- `app.py` - FastAPI backend with provider routing and TTS generation functions
- `static/index.html` - Single-page web UI (vanilla JS)
- `tests/test_api.py` - Unit tests with mocked HTTP calls
- `tests/test_integration.py` - Integration tests requiring real services

**API Endpoints:**
- `GET /api/providers` - Provider configs (models, voices, API key requirements)
- `GET /api/health` - Provider connectivity status
- `POST /api/tts` - Generate speech (accepts text, provider, model, voice, api_key)

**Environment variables:** `MLX_AUDIO_URL`, `ELEVENLABS_API_KEY`, `OPENAI_API_KEY`

## Testing Notes

- Unit tests mock `httpx.AsyncClient` to avoid external calls
- Integration tests use `@pytest.mark.skipif` to skip when services unavailable
- MLX-Audio returns MP3 (not WAV) - test assertions check for `audio/mpeg`
