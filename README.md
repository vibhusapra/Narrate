# Narrate

Transform text into natural speech. Paste in articles, blogs, papers, or any text and get a high-quality audio version.

## Features

- **Local TTS**: Spark-TTS runs on Apple Silicon (no API key needed)
- **Local Voice Cloning**: CSM-1B voice cloning from a short sample clip (Apple Silicon)
- **Cloud TTS**: ElevenLabs, OpenAI for highest quality
- **No Backend Storage**: API keys stored in browser only

## Quick Start

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and run
uv pip install -e .
uv pip install -e ".[local]"  # optional: MLX local providers
uv pip install -e ".[pocket-tts-mlx]"  # optional: direct MLX-enabled Pocket backend
uv run python app.py
```

Edit `.env` (optional) and add:

- `OPENAI_API_KEY`
- `ELEVENLABS_API_KEY`

Those values are read automatically by the app at startup.

By default, Narrate also auto-starts Pocket TTS on launch if `POCKET_TTS_URL` points to localhost
(default `http://127.0.0.1:8000`), so you get `pocket-tts` available from the same command.

Open http://localhost:3000 in your browser.

The web UI includes a live rough token-based cost estimate while typing (using tiktoken and the configured per-million-token rates).

## TTS Providers

### MLX-Audio (Local - Mac Only)

Free, runs locally on Apple Silicon. No API key needed.

Prereq (recommended): `brew install ffmpeg`

**Models:**
- **Spark-TTS 0.5B** - Best quality, supports English and Chinese
- **Spark-TTS 0.5B 8-bit** - Faster, uses less memory
- **CSM 1B 8-bit** - Voice cloning (recommended for speed)
- **CSM 1B** - Voice cloning (higher quality, slower)

```bash
# Install MLX-Audio server (one command)
uv tool install mlx-audio --prerelease=allow --force --with uvicorn --with fastapi --with python-multipart --with webrtcvad --with "setuptools<81"

# Start the TTS server (in a separate terminal)
mlx_audio.server --host 0.0.0.0 --port 8000
```

First run downloads model weights. Open http://localhost:3000, upload a short voice sample, pick a CSM model, and generate speech.

For voice cloning, upload a ~5–10 second sample and choose one of these transcript modes:

- `auto` (default): auto-transcribe from the clip when transcript is empty (requires `mlx-whisper`).
- `provided`: use your typed transcript only; reject empty transcript submissions.
- `off`: no STT request; your transcript is required.

The upload response includes transcript metadata (`transcript_status`, `transcript_source`, `duration_seconds`, `audio_duration_seconds`, etc.).
Use **Retranscribe** from the Voice Cloning panel for another pass, or re-upload with corrected transcript/stt settings.

`/api/voices/{voice_id}` allows transcript editing:
- `POST /api/voices/{voice_id}/retranscribe`
- `PATCH /api/voices/{voice_id}` (set manual transcript text)

Note: Apple Silicon is the target environment for local MLX paths (`mlx-audio`, `mlx-voice-clone`, `pocket-tts-mlx`).

### Pocket TTS (Kyutai)

Narrate now supports two Pocket paths:
- `pocket-tts` (HTTP server mode): starts or connects to a Pocket-compatible `/tts` server.
- `pocket-tts-mlx` (direct MLX mode): loads `pocket-tts-mlx` in-process for lower-latency local streaming.

Streaming local TTS with low-latency chunks and no hard text-length cap.

`pocket-tts` mode expects a Pocket-compatible HTTP service that exposes `/tts` (defaulting to `pocket-tts serve` via `POCKET_TTS_URL`).
This mode is auto-started when you run `uv run python app.py` (when `AUTO_START_POCKET_TTS=1`).

`pocket-tts-mlx` mode uses direct MLX inference in-process and is optimized for Apple Silicon live chunk streaming.
It requires `pocket-tts-mlx` to be installed but does **not** expose an HTTP serving API on its own.
`pocket-tts-mlx` targets Apple Silicon (`mlx`) environments.
`pocket-tts` is the compatibility/server-backed path; `pocket-tts-mlx` is the optimized direct path.

Both Pocket providers can use either the built-in Kyutai voices (`alba`, `marius`, etc.) or uploaded voice references from the
`Upload Reference Voice` panel for cloning-style behavior.

To disable auto-start or point to an existing server:

```bash
export AUTO_START_POCKET_TTS=0
export POCKET_TTS_URL=http://127.0.0.1:8000
```

Optional command override and startup timeout:

```bash
export POCKET_TTS_COMMAND=pocket-tts
export POCKET_TTS_STARTUP_TIMEOUT=120
```

If you prefer managing it manually, run:

```bash
pocket-tts serve --host 0.0.0.0 --port 8000
```

Install the direct MLX backend:

```bash
pip install pocket-tts-mlx
```

For `pocket-tts-mlx` users, keep `AUTO_START_POCKET_TTS` as-is (or set it to `0`) and select
`pocket-tts-mlx` in the UI provider list.

Set `POCKET_TTS_URL` if your Pocket server is not running at `http://127.0.0.1:8000`.

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
| `POCKET_TTS_URL` | `http://127.0.0.1:8000` | Pocket TTS server URL |
| `AUTO_START_POCKET_TTS` | `1` | Set `0` to prevent Narrate from auto-starting a local Pocket server |
| `POCKET_TTS_COMMAND` | `pocket-tts` | Executable used for auto-start |
| `POCKET_TTS_STARTUP_TIMEOUT` | `120` | Seconds to wait for auto-started Pocket service to become ready |
| `VOICE_SAMPLE_MAX_SECONDS` | `10` | Max seconds kept from uploaded voice sample (shorter = faster) |
| `SAVE_GENERATED_AUDIO` | `1` | Set `0` to disable saving generated outputs to `outputs/`. |
| `ELEVENLABS_API_KEY` | - | ElevenLabs API key (optional) |
| `OPENAI_API_KEY` | - | OpenAI API key (optional) |
| `OPENAI_GPT4O_MINI_TTS_COST_PER_MILLION_TOKENS` | `50.0` | Approximate rough rate used for OpenAI cost estimate |
| `OPENAI_TTS_1_COST_PER_MILLION_TOKENS` | `50.0` | Approximate rough rate used for OpenAI cost estimate |
| `OPENAI_TTS_1_HD_COST_PER_MILLION_TOKENS` | `80.0` | Approximate rough rate used for OpenAI cost estimate |
| `OPENAI_TTS_COST_PER_MILLION_TOKENS` | `50.0` | Fallback token-rate if model-specific value not set |
| `ELEVENLABS_TTS_COST_PER_MILLION_TOKENS` | `20.0` | Approximate rough rate used for ElevenLabs cost estimate |
| `OPENAI_TTS_MAX_INPUT_TOKENS` | `1800` | Token threshold before auto-chunking OpenAI TTS requests |
| `ELEVENLABS_TTS_MAX_INPUT_TOKENS` | `1800` | Token threshold before auto-chunking ElevenLabs TTS requests |

API keys can also be entered in the UI and are stored in browser localStorage.

Generated audio files are written to `/outputs` on each successful non-stream generation (and when `pocket-tts-mlx` streaming finishes), with the path returned in response header `X-Generated-Output`.

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
