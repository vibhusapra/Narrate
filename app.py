"""
Narrate - Text to Speech Audiobook Generator

Multi-provider TTS with local voice cloning (MLX-Audio) optimized for Apple Silicon.
"""

import asyncio
import base64
from contextlib import asynccontextmanager
import json
import inspect
import logging
import os
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    import tiktoken
except Exception:
    tiktoken = None


def _load_env_file(path: Path | str = ".env") -> None:
    """Load key/value pairs from a local .env file into process environment."""
    env_path = Path(path)
    if not env_path.exists():
        return

    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue

            if raw.startswith("export "):
                raw = raw[7:].strip()

            if "=" not in raw:
                continue

            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip().strip()

            if len(value) >= 2 and (
                (value.startswith('"') and value.endswith('"'))
                or (value.startswith("'") and value.endswith("'"))
            ):
                value = value[1:-1]

            if key and key not in os.environ:
                os.environ[key] = value
    except OSError as exc:
        logger = logging.getLogger("narrate")
        logger.warning("Failed to load .env file: %s", exc)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("narrate")

_load_env_file()


def _to_float_env(key: str, default: float) -> float:
    """Read an environment variable as float with fallback."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning("Invalid value for %s=%r; using default %s", key, value, default)
        return default


def _to_int_env(key: str, default: int) -> int:
    """Read an environment variable as int with fallback."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        parsed = int(value)
        if parsed <= 0:
            raise ValueError
        return parsed
    except (TypeError, ValueError):
        logger.warning("Invalid value for %s=%r; using default %s", key, value, default)
        return default


def _to_bool_env(key: str, default: bool = False) -> bool:
    """Read an environment variable as bool with fallback."""
    value = os.getenv(key)
    if value is None:
        return default

    value_normalized = value.strip().lower()
    if value_normalized in {"1", "true", "yes", "on"}:
        return True
    if value_normalized in {"0", "false", "no", "off"}:
        return False

    logger.warning("Invalid value for %s=%r; using default %s", key, value, default)
    return default


def _default_cost_rates() -> dict[str, dict[str, float]]:
    """Approximate token-based cost rates in USD per 1M tokens."""
    return {
        "openai": {
            "gpt-4o-mini-tts": _to_float_env("OPENAI_GPT4O_MINI_TTS_COST_PER_MILLION_TOKENS", 50.0),
            "tts-1": _to_float_env("OPENAI_TTS_1_COST_PER_MILLION_TOKENS", 50.0),
            "tts-1-hd": _to_float_env("OPENAI_TTS_1_HD_COST_PER_MILLION_TOKENS", 80.0),
            "default": _to_float_env("OPENAI_TTS_COST_PER_MILLION_TOKENS", 50.0),
        },
        "elevenlabs": {
            "default": _to_float_env("ELEVENLABS_TTS_COST_PER_MILLION_TOKENS", 20.0),
        },
        "mlx-audio": {
            "default": 0.0,
        },
        "mlx-voice-clone": {
            "default": 0.0,
        },
        "pocket-tts": {
            "default": 0.0,
        },
        "pocket-tts-mlx": {
            "default": 0.0,
        },
    }


# Optional: mlx-whisper for auto-transcription
try:
    import mlx_whisper

    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False


def _log_startup_banner() -> None:
    """Log the current runtime configuration on app startup."""
    logger.info("=" * 60)
    logger.info("Narrate - Text to Speech Server")
    logger.info("=" * 60)
    logger.info("MLX-Audio URL: %s", MLX_AUDIO_URL)
    logger.info("Pocket TTS URL: %s", POCKET_TTS_URL)
    logger.info("ElevenLabs API Key: %s", "✓ Set" if ELEVENLABS_API_KEY else "✗ Not set")
    logger.info("OpenAI API Key: %s", "✓ Set" if OPENAI_API_KEY else "✗ Not set")
    logger.info("MLX-Whisper: %s", "✓ Available" if HAS_WHISPER else "✗ Not available")
    logger.info("Voice sample max seconds: %.0fs", VOICE_SAMPLE_MAX_SECONDS)
    logger.info("Uploaded voices: %s", len(load_voices_metadata()))
    logger.info("=" * 60)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _log_startup_banner()
    await _start_pocket_tts_server_if_needed()
    if _pocket_tts_startup_error:
        logger.warning("Pocket TTS auto-start warning: %s", _pocket_tts_startup_error)
    try:
        yield
    finally:
        _stop_pocket_tts_server()


app = FastAPI(
    title="Narrate",
    description="Text to Speech Audiobook Generator",
    lifespan=lifespan,
)

# Voice uploads directory
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)
VOICES_METADATA_FILE = UPLOADS_DIR / "voices.json"

# Max upload size: 25 MB
MAX_UPLOAD_SIZE = 25 * 1024 * 1024

# Voice sample preprocessing (speed + reliability on macOS)
VOICE_SAMPLE_MAX_SECONDS = float(os.getenv("VOICE_SAMPLE_MAX_SECONDS", "10"))

# API Configuration
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)
MLX_AUDIO_URL = os.getenv("MLX_AUDIO_URL", "http://127.0.0.1:8000")
POCKET_TTS_URL = os.getenv("POCKET_TTS_URL", "http://127.0.0.1:8000")
POCKET_TTS_COMMAND = os.getenv("POCKET_TTS_COMMAND", "pocket-tts")
AUTO_START_POCKET_TTS = _to_bool_env("AUTO_START_POCKET_TTS", True)
POCKET_TTS_STARTUP_TIMEOUT = _to_float_env("POCKET_TTS_STARTUP_TIMEOUT", 120.0)
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SAVE_GENERATED_AUDIO = _to_bool_env("SAVE_GENERATED_AUDIO", True)
OPENAI_TTS_HARD_MAX_INPUT_TOKENS = 2000
OPENAI_TTS_MAX_INPUT_TOKENS = min(
    _to_int_env("OPENAI_TTS_MAX_INPUT_TOKENS", 1800),
    OPENAI_TTS_HARD_MAX_INPUT_TOKENS,
)
ELEVENLABS_TTS_MAX_INPUT_TOKENS = _to_int_env("ELEVENLABS_TTS_MAX_INPUT_TOKENS", 1800)

# Session storage for chunked generation (in-memory)
generation_sessions: Dict[str, dict] = {}
session_locks: Dict[str, asyncio.Lock] = {}
_pocket_tts_process: Optional[subprocess.Popen] = None
_pocket_tts_managed = False
_pocket_tts_startup_error: str | None = None
_pocket_tts_mlx_module: Any | None = None
_pocket_tts_mlx_runtime: Any | None = None
_pocket_tts_mlx_init_time_ms: float | None = None
_pocket_tts_mlx_sample_rate: int = 24000
_pocket_tts_mlx_channels: int = 1
_pocket_tts_mlx_lock = asyncio.Lock()

# Provider configurations
PROVIDERS = {
    "mlx-audio": {
        "name": "MLX-Audio (Local)",
        "description": "Local TTS on Apple Silicon",
        "supports_voice_id": False,
        "requires_voice_id": False,
        "requires_reference_transcript": False,
        "requires_api_key": False,
        "models": {
            "mlx-community/Spark-TTS-0.5B-bf16": "Spark TTS 0.5B (Best quality, EN/ZH)",
            "mlx-community/Spark-TTS-0.5B-8bit": "Spark TTS 0.5B 8-bit (Faster, less memory)",
        },
        "voices": {},
    },
    "elevenlabs": {
        "name": "ElevenLabs",
        "description": "Cloud TTS with natural voices",
        "supports_voice_id": False,
        "requires_voice_id": False,
        "requires_reference_transcript": False,
        "requires_api_key": True,
        "models": {
            "eleven_flash_v2_5": "Flash v2.5 (Low latency)",
            "eleven_multilingual_v2": "Multilingual v2 (Best quality)",
            "eleven_turbo_v2_5": "Turbo v2.5 (Balanced)",
        },
        "voices": {
            "21m00Tcm4TlvDq8ikWAM": "Rachel",
            "EXAVITQu4vr4xnSDxMaL": "Bella",
            "ErXwobaYiN019PkySvjV": "Antoni",
            "VR6AewLTigWG4xSOukaG": "Arnold",
            "pNInz6obpgDQGcFmaJgB": "Adam",
        },
    },
    "openai": {
        "name": "OpenAI",
        "description": "Cloud TTS with GPT-4o voices",
        "supports_voice_id": False,
        "requires_voice_id": False,
        "requires_reference_transcript": False,
        "requires_api_key": True,
        "models": {
            "gpt-4o-mini-tts": "GPT-4o Mini TTS (Best)",
            "tts-1": "TTS-1 (Fast)",
            "tts-1-hd": "TTS-1 HD (High quality)",
        },
        "voices": {
            "alloy": "Alloy",
            "ash": "Ash",
            "ballad": "Ballad",
            "coral": "Coral",
            "echo": "Echo",
            "fable": "Fable",
            "nova": "Nova",
            "onyx": "Onyx",
            "sage": "Sage",
            "shimmer": "Shimmer",
        },
    },
    "mlx-voice-clone": {
        "name": "MLX Voice Clone (Local)",
        "description": "Clone a voice from a short sample (≈3–10s)",
        "supports_voice_id": True,
        "requires_voice_id": True,
        "requires_reference_transcript": True,
        "requires_api_key": False,
        "requires_voice_upload": True,
        "models": {
            # CSM (Sesame) - best speed/quality tradeoff on Apple Silicon
            "mlx-community/csm-1b-8bit": "CSM 1B 8-bit (Fastest, recommended)",
            "mlx-community/csm-1b-fp16": "CSM 1B fp16 (Balanced)",
            "mlx-community/csm-1b": "CSM 1B bf16 (Best quality, slower)",
            # Keep Qwen3-TTS options for compatibility
            "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16": "Qwen3-TTS 0.6B (Fast)",
            "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16": "Qwen3-TTS 1.7B (Better quality)",
        },
        "voices": {},
    },
    "pocket-tts": {
        "name": "Pocket TTS (Kyutai)",
        "description": "Streaming local TTS with low-latency playback and long-text support. Supports preset Kyutai voices and uploaded reference voices for cloning.",
        "supports_voice_id": True,
        "requires_voice_id": False,
        "requires_reference_transcript": False,
        "requires_api_key": False,
        "models": {
            "default": "Pocket TTS default (no custom backend config)",
        },
        "voices": {
            "alba": "Alba",
            "marius": "Marius",
            "javert": "Javert",
            "jean": "Jean",
            "fantine": "Fantine",
            "cosette": "Cosette",
            "eponine": "Eponine",
            "azelma": "Azelma",
        },
    },
    "pocket-tts-mlx": {
        "name": "Pocket TTS (Kyutai + MLX)",
        "description": "Direct pocket-tts-mlx backend with live streaming chunks. Supports preset Kyutai voices and uploaded reference voices for cloning.",
        "supports_voice_id": True,
        "requires_voice_id": False,
        "requires_reference_transcript": False,
        "requires_api_key": False,
        "models": {
            "default": "Pocket TTS MLX",
        },
        "voices": {
            "alba": "Alba",
            "marius": "Marius",
            "javert": "Javert",
            "jean": "Jean",
            "fantine": "Fantine",
            "cosette": "Cosette",
            "eponine": "Eponine",
            "azelma": "Azelma",
        },
    },
}


class TTSRequest(BaseModel):
    text: str
    provider: str = "mlx-audio"
    model: str = "mlx-community/Spark-TTS-0.5B-bf16"
    voice: str | None = None
    instruct: str | None = None
    api_key: str | None = None
    voice_id: str | None = None  # For voice cloning - references uploaded voice
    stream: bool = False  # For providers that return streaming audio
    max_concurrent: int = 5  # Max parallel chunks for chunked generation


class VoiceUploadRetranscribeRequest(BaseModel):
    language: str | None = None
    stt_model: str | None = None


class VoiceTranscriptUpdate(BaseModel):
    transcript: str


class CostEstimateRequest(BaseModel):
    text: str
    provider: str = "mlx-audio"
    model: str = "mlx-community/Spark-TTS-0.5B-bf16"
    instruct: str | None = None


def estimate_tokens_for_text(text: str, model: str, provider: str) -> tuple[int, bool]:
    """Count tokens for cost estimation using a best-effort tokenizer."""
    if not text.strip():
        return 0, False

    if tiktoken is None:
        return len(text.split()), False

    provider_model = model or "cl100k_base"
    try:
        if provider == "openai":
            encoding = tiktoken.encoding_for_model(model)
        else:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text)), True
    except Exception as exc:
        logger.debug("Tokenization failed for provider=%s model=%s: %s", provider, provider_model, exc)
        try:
            return len(text.split()), False
        except Exception:
            return 0, False


def estimate_tts_cost(request: CostEstimateRequest) -> dict[str, float | int | None]:
    """Estimate cost for a TTS request using a rough token-based model."""
    token_text = request.text
    if request.instruct:
        token_text = f"{token_text}\n\n{request.instruct}"

    tokens, uses_tiktoken = estimate_tokens_for_text(token_text, request.model, request.provider)
    rates = _default_cost_rates().get(request.provider, {}).copy()
    rate_per_million = rates.get(request.model, rates.get("default", 0.0))
    estimated_cost = tokens / 1_000_000 * rate_per_million

    return {
        "input_tokens": tokens,
        "uses_tiktoken": uses_tiktoken,
        "rate_usd_per_1m_tokens": rate_per_million,
        "estimated_cost_usd": round(estimated_cost, 6) if rate_per_million > 0 else None,
    }


def load_voices_metadata() -> dict:
    """Load voice metadata from JSON file."""
    if VOICES_METADATA_FILE.exists():
        with open(VOICES_METADATA_FILE, "r") as f:
            return json.load(f)
    return {}


def save_voices_metadata(metadata: dict):
    """Save voice metadata to JSON file."""
    with open(VOICES_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)


def _build_output_path(provider: str, ext: str) -> Path:
    stem = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{provider}_{uuid.uuid4().hex[:10]}.{ext}"
    return OUTPUTS_DIR / stem


def save_generated_audio(audio_bytes: bytes, provider: str, ext: str) -> str | None:
    """Persist generated audio to the repository outputs folder and return relative path."""
    if not SAVE_GENERATED_AUDIO:
        return None
    if not audio_bytes:
        return None
    try:
        output_path = _build_output_path(provider, ext)
        output_path.write_bytes(audio_bytes)
        return str(output_path)
    except Exception as exc:
        logger.warning("Failed to save generated audio: %s", exc)
        return None


def transcribe_audio(
    audio_path: Path,
    language: str | None = None,
    stt_model: str | None = None,
) -> str:
    """Transcribe audio file using mlx-whisper (fast default)."""
    if not HAS_WHISPER:
        raise HTTPException(
            status_code=503,
            detail="STT not available: install mlx-whisper with `uv pip install mlx-whisper`.",
        )

    def _normalize_language(value: str | None) -> str | None:
        if not value:
            return None
        clean = value.strip()
        return clean if clean else None

    whisper_kwargs: dict[str, Any] = {
        "path_or_hf_repo": "mlx-community/whisper-tiny",
    }
    normalized_language = _normalize_language(language)
    if normalized_language:
        whisper_kwargs["language"] = normalized_language
    normalized_model = _normalize_language(stt_model)
    if normalized_model:
        whisper_kwargs["path_or_hf_repo"] = normalized_model

    result = mlx_whisper.transcribe(str(audio_path), **whisper_kwargs)
    return result.get("text", "").strip()


def _serialize_voice_metadata(voice_id: str, voice_info: dict[str, Any]) -> dict[str, Any]:
    """Serialize a voice payload for /api/voices listing and helper responses."""
    audio_duration_seconds = voice_info.get("audio_duration_seconds", voice_info.get("duration_seconds"))
    return {
        "id": voice_id,
        "voice_id": voice_id,
        "name": voice_info.get("name", ""),
        "transcript": voice_info.get("transcript", ""),
        "transcript_status": voice_info.get("transcript_status", "not_available"),
        "transcript_source": voice_info.get("transcript_source", "user"),
        "filename": voice_info.get("filename", ""),
        "ref_filename": voice_info.get("ref_filename"),
        "duration_seconds": audio_duration_seconds,
        "audio_duration_seconds": audio_duration_seconds,
        "sample_rate": voice_info.get("sample_rate"),
        "channels": voice_info.get("channels"),
        "transcription_error": voice_info.get("transcription_error"),
        "quality_warning": voice_info.get("quality_warning"),
    }


def _normalize_uploaded_voice_id(value: str | None) -> str | None:
    """Normalize uploaded reference IDs from UI-specific value formats."""
    if not value:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None

    prefix = "__uploaded__:"
    if cleaned.startswith(prefix):
        normalized = cleaned[len(prefix):].strip()
        return normalized if normalized else None

    return cleaned


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
        if parsed < 0:
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def _probe_audio_metadata(audio_path: Path) -> dict[str, float | int | None]:
    """Collect lightweight audio metadata for uploaded clips."""
    metadata = {
        "duration_seconds": None,
        "sample_rate": None,
        "channels": None,
    }

    try:
        import wave

        with wave.open(str(audio_path), "rb") as wav_file:
            metadata["sample_rate"] = wav_file.getframerate()
            metadata["channels"] = wav_file.getnchannels()
            metadata["duration_seconds"] = (
                wav_file.getnframes() / wav_file.getframerate()
                if wav_file.getframerate() > 0
                else None
            )
            return metadata
    except Exception:
        pass

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return metadata

    try:
        probe = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_entries",
                "format=duration",
                "-show_streams",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(probe.stdout or "{}")

        duration = _safe_float(
            ((data.get("format") or {}).get("duration") if isinstance(data.get("format"), dict) else None)
        )
        if duration is not None:
            metadata["duration_seconds"] = duration

        streams = data.get("streams") or []
        if isinstance(streams, list) and streams:
            for stream in streams:
                if stream.get("codec_type") == "audio":
                    metadata["sample_rate"] = int(stream.get("sample_rate", metadata["sample_rate"] or 0)) or metadata["sample_rate"]
                    metadata["channels"] = int(stream.get("channels", metadata["channels"] or 0)) or metadata["channels"]
                    break
    except Exception:
        logger.debug("Audio metadata probe failed", exc_info=True)

    return metadata


def _voice_upload_warning(duration_seconds: float | None) -> str | None:
    if duration_seconds is None:
        return None
    if duration_seconds < 0.8:
        return "Very short reference clip may reduce cloning quality. Try a longer sample (1.5s+)."
    max_seconds = max(1.0, VOICE_SAMPLE_MAX_SECONDS)
    if duration_seconds > max_seconds * 1.5:
        return (
            f"Long clip detected ({duration_seconds:.1f}s)."
            " For best results Narrate keeps a short trimmed segment."
        )
    return None


def preprocess_voice_sample(input_path: Path, voice_id: str) -> Optional[Path]:
    """
    Best-effort preprocessing for voice cloning:
    - Convert to mono 16k WAV
    - Trim to a short clip
    - Remove leading/trailing silence

    Returns processed path, or None on failure.
    """
    output_path = UPLOADS_DIR / f"{voice_id}_ref.wav"
    max_seconds = max(1.0, VOICE_SAMPLE_MAX_SECONDS)

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        logger.warning("ffmpeg not found; skipping voice preprocessing")
        return None

    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        # Remove silence then hard-trim to max length
        (
            "silenceremove=start_periods=1:start_silence=0.2:start_threshold=-35dB:"
            "stop_periods=1:stop_silence=0.2:stop_threshold=-35dB,"
            f"atrim=0:{max_seconds}"
        ),
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_path if output_path.exists() else None
    except Exception as e:
        logger.warning(f"Voice preprocessing failed (continuing with original): {e}")
        return None


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_sentences(sentences: List[str], max_words: int = 500) -> List[str]:
    """Group sentences into chunks up to max_words."""
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_word_count + word_count > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_text_by_chars(text: str, max_chars: int) -> List[str]:
    """Split text into chunks that do not exceed max_chars."""
    cleaned = text.strip()
    if not cleaned:
        return []

    if len(cleaned) <= max_chars:
        return [cleaned]

    chunks: List[str] = []
    current = ""

    for sentence in split_into_sentences(cleaned):
        # Hard split if a single sentence is still too long
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(sentence), max_chars):
                chunk = sentence[i : i + max_chars].strip()
                if chunk:
                    chunks.append(chunk)
            continue

        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def chunk_text_by_tokens(text: str, max_tokens: int, provider: str, model: str) -> List[str]:
    """Split text into chunks that stay within a token budget."""
    cleaned = text.strip()
    if not cleaned:
        return []
    if max_tokens <= 0:
        return [cleaned]

    full_tokens, _ = estimate_tokens_for_text(cleaned, model=model, provider=provider)
    if full_tokens <= max_tokens:
        return [cleaned]

    chunks: List[str] = []
    current = ""

    for sentence in split_into_sentences(cleaned):
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_tokens, _ = estimate_tokens_for_text(sentence, model=model, provider=provider)
        if sentence_tokens > max_tokens:
            if current:
                chunks.append(current)
                current = ""
            words = sentence.split()
            partial = ""
            for word in words:
                candidate = word if not partial else f"{partial} {word}"
                candidate_tokens, _ = estimate_tokens_for_text(candidate, model=model, provider=provider)
                if candidate_tokens > max_tokens:
                    if partial:
                        chunks.append(partial)
                        partial = word
                    else:
                        for i in range(0, len(word), 200):
                            piece = word[i : i + 200].strip()
                            if piece:
                                chunks.append(piece)
                        partial = ""
                else:
                    partial = candidate
            if partial:
                chunks.append(partial)
            continue

        candidate = sentence if not current else f"{current} {sentence}"
        candidate_tokens, _ = estimate_tokens_for_text(candidate, model=model, provider=provider)
        if candidate_tokens > max_tokens and current:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def is_input_too_long_error(error_text: str) -> bool:
    """Best-effort detection of provider input-length validation errors."""
    normalized = (error_text or "").lower()
    markers = (
        "maximum input limit",
        "max input",
        "string too long",
        "too many tokens",
        "too long",
        "over the maximum",
    )
    return any(marker in normalized for marker in markers)


def concatenate_audio_chunks(audio_chunks: List[bytes], output_ext: str = "mp3", crossfade_ms: int = 20) -> bytes:
    """Concatenate audio chunks using ffmpeg."""
    if not audio_chunks:
        raise ValueError("No audio chunks to concatenate")
    if len(audio_chunks) == 1:
        return audio_chunks[0]

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg is required to concatenate chunked audio")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        chunk_files: List[Path] = []
        for i, chunk_bytes in enumerate(audio_chunks):
            chunk_file = tmpdir_path / f"chunk_{i:04d}.{output_ext}"
            with open(chunk_file, "wb") as f:
                f.write(chunk_bytes)
            chunk_files.append(chunk_file)

        concat_list = tmpdir_path / "concat_list.txt"
        with open(concat_list, "w") as f:
            for chunk_file in chunk_files:
                safe_name = str(chunk_file).replace("'", "'\\''")
                f.write(f"file '{safe_name}'\n")

        output_file = tmpdir_path / f"combined.{output_ext}"

        ffmpeg_cmd = [ffmpeg_path, "-f", "concat", "-safe", "0", "-i", str(concat_list)]
        if output_ext == "wav":
            ffmpeg_cmd.extend(["-c:a", "pcm_s16le", "-y", str(output_file)])
        else:
            ffmpeg_cmd.extend(["-c", "copy", "-y", str(output_file)])

        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)

        with open(output_file, "rb") as f:
            return f.read()


def get_session_lock(session_id: str) -> asyncio.Lock:
    """Get or create lock for session."""
    if session_id not in session_locks:
        session_locks[session_id] = asyncio.Lock()
    return session_locks[session_id]


async def update_session_progress(session_id: str, completed_count: int, total_count: int):
    """Thread-safe session progress update."""
    lock = get_session_lock(session_id)
    async with lock:
        if session_id in generation_sessions:
            generation_sessions[session_id]["progress"] = {
                "current": completed_count,
                "total": total_count,
            }


async def generate_mlx_voice_clone_single(
    text: str,
    model: str,
    voice_id: str,
    instruct: str | None = None,
    timeout: float = 120.0,
) -> bytes:
    """Generate speech for a single chunk using MLX-Audio voice cloning."""
    if not voice_id:
        raise HTTPException(status_code=400, detail="Voice ID required for voice cloning")

    voices_metadata = load_voices_metadata()
    if voice_id not in voices_metadata:
        raise HTTPException(status_code=404, detail="Voice not found")

    voice_info = voices_metadata[voice_id]
    ref_filename = voice_info.get("ref_filename") or voice_info.get("filename")
    audio_path = UPLOADS_DIR / ref_filename if ref_filename else UPLOADS_DIR / voice_info["filename"]
    if not audio_path.exists() and voice_info.get("filename"):
        audio_path = UPLOADS_DIR / voice_info["filename"]
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Voice audio file not found")

    ref_text = voice_info.get("transcript", "")
    if not ref_text:
        raise HTTPException(status_code=400, detail="Voice transcript is required for cloning")

    timeout_config = httpx.Timeout(timeout, connect=10.0, read=timeout, write=10.0, pool=10.0)

    async with httpx.AsyncClient(timeout=timeout_config) as client:
        response = await client.post(
            f"{MLX_AUDIO_URL}/v1/audio/speech",
            json={
                "model": model,
                "input": text,
                "instruct": instruct,
                "ref_audio": str(audio_path.absolute()),
                "ref_text": ref_text,
            },
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"MLX-Audio voice clone error: {response.text}",
            )
        return response.content


async def generate_chunked_tts(
    chunks: List[str],
    model: str,
    voice_id: str,
    session_id: str,
    instruct: str | None = None,
    max_retries: int = 3,
    max_concurrent: int = 5,
) -> bytes:
    """Generate TTS for chunks in parallel with a concurrency limit."""
    total_chunks = len(chunks)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def generate_chunk_with_retry(index: int, chunk_text: str):
        async with semaphore:
            if (
                session_id in generation_sessions
                and generation_sessions[session_id]["status"] == "cancelled"
            ):
                raise Exception("Generation cancelled by user")

            for attempt in range(max_retries):
                try:
                    return (
                        index,
                        await generate_mlx_voice_clone_single(
                            chunk_text,
                            model,
                            voice_id,
                            instruct=instruct,
                            timeout=1800.0,
                        ),
                    )
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise Exception(f"Chunk {index+1} failed after {max_retries} retries: {e}")
                    await asyncio.sleep(2**attempt)

    tasks = [generate_chunk_with_retry(i, chunk) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    completed_count = 0
    audio_results: List[bytes | None] = [None] * total_chunks
    for result in results:
        if isinstance(result, Exception):
            raise result
        index, audio_data = result
        audio_results[index] = audio_data
        completed_count += 1
        await update_session_progress(session_id, completed_count, total_chunks)

    combined_audio = concatenate_audio_chunks([a for a in audio_results if a is not None], crossfade_ms=20)
    return combined_audio


async def generate_with_progress(
    text: str,
    provider: str,
    model: str,
    voice_id: str,
    session_id: str,
    instruct: str | None = None,
    max_concurrent: int = 5,
):
    """Background task orchestrator for chunked generation with progress tracking."""
    try:
        word_count = len(text.split())
        sentences = split_into_sentences(text)
        chunks = chunk_sentences(sentences, max_words=1000)

        if session_id in generation_sessions:
            generation_sessions[session_id].update(
                {
                    "word_count": word_count,
                    "chunks": len(chunks),
                    "progress": {"current": 0, "total": len(chunks)},
                }
            )

        audio_data = await generate_chunked_tts(
            chunks,
            model,
            voice_id,
            session_id,
            instruct=instruct,
            max_concurrent=max_concurrent,
        )

        if session_id in generation_sessions:
            generation_sessions[session_id].update(
                {
                    "status": "complete",
                    "audio_data": audio_data,
                    "progress": {"current": len(chunks), "total": len(chunks)},
                }
            )

    except Exception as e:
        logger.error(f"Session {session_id}: Generation failed: {e}", exc_info=True)
        if session_id in generation_sessions:
            generation_sessions[session_id].update(
                {
                    "status": "error",
                    "error": str(e),
                }
            )


async def _probe_pocket_tts(url: str, timeout_seconds: float = 2.0) -> bool:
    """Check whether Pocket TTS responds to /health."""
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(f"{url}/health")
            return response.status_code == 200
    except Exception:
        return False


def _is_local_pocket_url(url: str) -> bool:
    """Return True when POCKET_TTS_URL points to a local endpoint."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        return parsed.scheme.startswith("http") and hostname in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
    except Exception:
        return False


def _pocket_tts_host_and_port(url: str) -> tuple[str, int]:
    """Extract host and port from POCKET_TTS_URL with fallback defaults."""
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    return host, port


async def _start_pocket_tts_server_if_needed() -> None:
    """Start a local Pocket TTS server if configured and not already available."""
    global _pocket_tts_process, _pocket_tts_managed, _pocket_tts_startup_error

    _pocket_tts_startup_error = None

    if not AUTO_START_POCKET_TTS:
        logger.info("AUTO_START_POCKET_TTS is disabled.")
        return

    if not _is_local_pocket_url(POCKET_TTS_URL):
        logger.info("POCKET_TTS_URL is not local (%s); skipping auto-start.", POCKET_TTS_URL)
        return

    if await _probe_pocket_tts(POCKET_TTS_URL, timeout_seconds=1.0):
        logger.info("Pocket TTS already running at %s", POCKET_TTS_URL)
        return

    command_path = shutil.which(POCKET_TTS_COMMAND)
    if not command_path:
        _pocket_tts_startup_error = (
            f"Pocket TTS command not found: {POCKET_TTS_COMMAND}. "
            "Install pocket-tts or set POCKET_TTS_COMMAND."
        )
        logger.warning(_pocket_tts_startup_error)
        return

    host, port = _pocket_tts_host_and_port(POCKET_TTS_URL)

    logger.info(
        "Auto-starting Pocket TTS from Narrate (command=%s, host=%s, port=%s)",
        POCKET_TTS_COMMAND,
        host,
        port,
    )

    _pocket_tts_process = subprocess.Popen(
        [command_path, "serve", "--host", host, "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _pocket_tts_managed = True

    deadline = asyncio.get_running_loop().time() + max(10.0, POCKET_TTS_STARTUP_TIMEOUT)
    while asyncio.get_running_loop().time() < deadline:
        if await _probe_pocket_tts(POCKET_TTS_URL, timeout_seconds=1.0):
            logger.info("Pocket TTS started successfully at %s", POCKET_TTS_URL)
            return
        await asyncio.sleep(0.5)

    if _pocket_tts_process.poll() is not None:
        _pocket_tts_startup_error = "Pocket TTS process exited before becoming ready."
        _pocket_tts_process = None
        _pocket_tts_managed = False
        logger.error(_pocket_tts_startup_error)
        return

    _pocket_tts_startup_error = (
        f"Pocket TTS did not become ready within {POCKET_TTS_STARTUP_TIMEOUT:.0f}s."
    )
    logger.error(_pocket_tts_startup_error)


def _stop_pocket_tts_server() -> None:
    """Shut down the process started by Narrate, if any."""
    global _pocket_tts_process, _pocket_tts_managed

    if not _pocket_tts_managed or _pocket_tts_process is None:
        return

    process = _pocket_tts_process
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        logger.info("Pocket TTS process terminated.")

    _pocket_tts_process = None
    _pocket_tts_managed = False


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/providers")
async def get_providers():
    """Get available TTS providers and their models."""
    providers_response: dict = {pid: cfg.copy() for pid, cfg in PROVIDERS.items()}

    voices_metadata = load_voices_metadata()
    uploaded_voice_map = {vid: v["name"] for vid, v in voices_metadata.items()}

    for provider_id, provider in providers_response.items():
        if provider.get("supports_voice_id"):
            provider["uploaded_voices"] = uploaded_voice_map
            if provider.get("voices") is None:
                provider["voices"] = {}

    providers_response["mlx-voice-clone"]["voices"] = {
        vid: v["name"] for vid, v in voices_metadata.items()
    }
    return {"providers": providers_response}


@app.get("/api/health")
async def health_check():
    """Check provider connectivity."""
    status = {
        "mlx_audio": "unknown",
        "pocket_tts": "unknown",
        "pocket_tts_mlx": "unknown",
        "elevenlabs": "configured" if ELEVENLABS_API_KEY else "no_api_key",
        "openai": "configured" if OPENAI_API_KEY else "no_api_key",
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{MLX_AUDIO_URL}/v1/models")
            status["mlx_audio"] = "connected" if response.status_code == 200 else "error"
    except Exception:
        status["mlx_audio"] = "disconnected"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{POCKET_TTS_URL}/health")
            status["pocket_tts"] = "connected" if response.status_code == 200 else "error"
    except Exception:
        status["pocket_tts"] = "disconnected"
    try:
        import importlib

        importlib.import_module("pocket_tts_mlx")
        status["pocket_tts_mlx"] = "ready" if _pocket_tts_mlx_runtime is not None else "installed"
    except Exception:
        status["pocket_tts_mlx"] = "not_installed"

    return {"status": "ok", "providers": status}


@app.post("/api/cost-estimate")
async def cost_estimate(request: CostEstimateRequest):
    """Return a rough token-based cost estimate for a TTS request."""
    if request.provider not in PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {request.provider}")

    estimate = estimate_tts_cost(request)
    if request.provider in {"mlx-audio", "mlx-voice-clone", "pocket-tts"}:
        estimate["note"] = "Local providers: no API cost tracked here."
    elif request.provider == "pocket-tts-mlx":
        estimate["note"] = "Local providers: no API cost tracked here."

    return {
        "provider": request.provider,
        "model": request.model,
        **estimate,
    }


@app.post("/api/upload-voice")
async def upload_voice(
    file: UploadFile = File(...),
    name: str = Form(...),
    transcript: str = Form(""),
    transcript_mode: str = Form("auto"),
    language: str = Form(""),
    stt_model: str = Form(""),
):
    """Upload a voice sample for cloning with optional STT mode."""
    logger.info(
        f"Voice Upload: name='{name}', file='{file.filename}', type={file.content_type}"
    )

    normalized_mode = (transcript_mode or "auto").strip().lower()
    if normalized_mode not in {"auto", "provided", "off"}:
        raise HTTPException(status_code=400, detail="Invalid transcript_mode. Use auto, provided, or off.")

    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 25 MB)")

    voice_id = uuid.uuid4().hex[:12]

    ext = "wav"
    if file.content_type == "audio/mpeg":
        ext = "mp3"
    elif file.content_type == "audio/mp4":
        ext = "m4a"
    elif file.filename and "." in file.filename:
        ext = file.filename.rsplit(".", 1)[-1].lower()

    raw_audio_path = UPLOADS_DIR / f"{voice_id}.{ext}"
    with open(raw_audio_path, "wb") as f:
        f.write(content)

    processed_path = preprocess_voice_sample(raw_audio_path, voice_id=voice_id)
    audio_for_transcription = processed_path or raw_audio_path

    final_transcript = transcript.strip()
    transcript_status = "not_available"
    transcript_source = "user"
    transcription_error = None

    if final_transcript:
        transcript_status = "manual"
        transcript_source = "user"
    else:
        if normalized_mode == "provided":
            raise HTTPException(
                status_code=400,
                detail="Transcript mode is 'provided', but no transcript was supplied.",
            )
        if normalized_mode == "off":
            raise HTTPException(
                status_code=400,
                detail="Transcript mode is 'off'. Provide a transcript manually or set transcript_mode=auto.",
            )

        try:
            logger.info("Auto-transcribing audio with mlx-whisper...")
            final_transcript = transcribe_audio(
                audio_for_transcription,
                language=(language or None),
                stt_model=(stt_model or None),
            )
            transcript_status = "auto_transcribed" if final_transcript else "not_available"
            transcript_source = "mlx_whisper" if final_transcript else "user"
        except HTTPException:
            raise
        except Exception as exc:
            transcription_error = str(exc)
            raise HTTPException(
                status_code=500,
                detail="Failed to transcribe audio. Retry with transcript_mode=off and provide transcript manually.",
            )

    audio_metadata = _probe_audio_metadata(audio_for_transcription)
    quality_warning = _voice_upload_warning(audio_metadata["duration_seconds"])

    voices_metadata = load_voices_metadata()
    voices_metadata[voice_id] = {
        "name": name,
        "transcript": final_transcript,
        "filename": raw_audio_path.name,
        "ref_filename": processed_path.name if processed_path else None,
        "original_filename": file.filename,
        "transcript_status": transcript_status,
        "transcript_source": transcript_source,
        "transcription_error": transcription_error,
        "duration_seconds": audio_metadata.get("duration_seconds"),
        "audio_duration_seconds": audio_metadata.get("duration_seconds"),
        "sample_rate": audio_metadata.get("sample_rate"),
        "channels": audio_metadata.get("channels"),
        "quality_warning": quality_warning,
    }
    save_voices_metadata(voices_metadata)

    return {
        "voice_id": voice_id,
        "name": name,
        "filename": raw_audio_path.name,
        "ref_filename": processed_path.name if processed_path else None,
        "transcript": final_transcript,
        "transcript_status": transcript_status,
        "transcript_source": transcript_source,
        "transcription_error": transcription_error,
        "duration_seconds": audio_metadata.get("duration_seconds"),
        "audio_duration_seconds": audio_metadata.get("duration_seconds"),
        "sample_rate": audio_metadata.get("sample_rate"),
        "channels": audio_metadata.get("channels"),
        "quality_warning": quality_warning,
    }


@app.get("/api/voices")
async def list_voices():
    """List all uploaded voice samples."""
    voices_metadata = load_voices_metadata()
    voices = [_serialize_voice_metadata(vid, v) for vid, v in voices_metadata.items()]
    return {"voices": voices}


@app.post("/api/voices/{voice_id}/retranscribe")
async def retranscribe_voice(voice_id: str, request: VoiceUploadRetranscribeRequest):
    """Re-run STT for an uploaded reference voice."""
    voices_metadata = load_voices_metadata()
    if voice_id not in voices_metadata:
        raise HTTPException(status_code=404, detail="Voice not found")

    voice_path = _get_uploaded_voice_path(voice_id)
    metadata = voices_metadata[voice_id]

    transcript = transcribe_audio(
        voice_path,
        language=request.language,
        stt_model=request.stt_model,
    )
    metadata["transcript"] = transcript
    metadata["transcript_status"] = "auto_transcribed" if transcript else "not_available"
    metadata["transcript_source"] = "mlx_whisper" if transcript else "user"
    metadata["transcription_error"] = None
    save_voices_metadata(voices_metadata)

    return _serialize_voice_metadata(voice_id, metadata)


@app.patch("/api/voices/{voice_id}")
async def update_voice_metadata(voice_id: str, request: VoiceTranscriptUpdate):
    """Update uploaded voice transcript text."""
    voices_metadata = load_voices_metadata()
    if voice_id not in voices_metadata:
        raise HTTPException(status_code=404, detail="Voice not found")

    transcript = request.transcript.strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript cannot be empty")

    voice_info = voices_metadata[voice_id]
    voice_info["transcript"] = transcript
    voice_info["transcript_status"] = "manual"
    voice_info["transcript_source"] = "user"
    voice_info["transcription_error"] = None
    save_voices_metadata(voices_metadata)

    return _serialize_voice_metadata(voice_id, voice_info)


@app.delete("/api/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete an uploaded voice sample."""
    voices_metadata = load_voices_metadata()
    if voice_id not in voices_metadata:
        raise HTTPException(status_code=404, detail="Voice not found")

    voice_info = voices_metadata[voice_id]

    raw_audio_path = UPLOADS_DIR / voice_info["filename"]
    if raw_audio_path.exists():
        raw_audio_path.unlink()

    ref_filename = voice_info.get("ref_filename")
    if ref_filename:
        ref_audio_path = UPLOADS_DIR / ref_filename
        if ref_audio_path.exists() and ref_audio_path != raw_audio_path:
            ref_audio_path.unlink()

    del voices_metadata[voice_id]
    save_voices_metadata(voices_metadata)

    # Best-effort cleanup of any leftover processed refs
    for candidate in UPLOADS_DIR.glob(f"{voice_id}_ref.*"):
        if candidate.exists():
            candidate.unlink()

    return {"status": "deleted", "voice_id": voice_id}


async def generate_mlx_audio(text: str, model: str, voice: str | None, instruct: str | None) -> bytes:
    """Generate speech using MLX-Audio local server (OpenAI-compatible API)."""
    voice_id = voice or "af_heart"

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{MLX_AUDIO_URL}/v1/audio/speech",
            json={"model": model, "input": text, "voice": voice_id, "instruct": instruct},
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"MLX-Audio error: {response.text}")
        return response.content


async def generate_elevenlabs(text: str, model: str, voice: str | None, api_key: str, instruct: str | None) -> bytes:
    """Generate speech using ElevenLabs API."""
    if not api_key:
        raise HTTPException(status_code=400, detail="ElevenLabs API key required")

    voice_id = voice or "21m00Tcm4TlvDq8ikWAM"
    input_tokens, _ = estimate_tokens_for_text(text, model=model, provider="elevenlabs")
    if input_tokens > ELEVENLABS_TTS_MAX_INPUT_TOKENS:
        return await generate_elevenlabs_chunked(text, model, voice_id, api_key, instruct)

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json={
                "text": text,
                "model_id": model,
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
        )
        if response.status_code != 200:
            if response.status_code == 400 and is_input_too_long_error(response.text):
                return await generate_elevenlabs_chunked(text, model, voice_id, api_key, instruct)
            raise HTTPException(status_code=response.status_code, detail=f"ElevenLabs error: {response.text}")
        return response.content


async def generate_openai(text: str, model: str, voice: str | None, api_key: str, instruct: str | None) -> bytes:
    """Generate speech using OpenAI TTS API."""
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key required")

    voice_id = voice or "alloy"
    input_tokens, _ = estimate_tokens_for_text(text, model=model, provider="openai")
    if input_tokens > OPENAI_TTS_MAX_INPUT_TOKENS:
        return await generate_openai_chunked(text, model, voice_id, api_key, instruct)

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "input": text,
                "voice": voice_id,
                "response_format": "wav",
            },
        )
        if response.status_code != 200:
            if response.status_code == 400 and is_input_too_long_error(response.text):
                return await generate_openai_chunked(text, model, voice_id, api_key, instruct)
            raise HTTPException(status_code=response.status_code, detail=f"OpenAI error: {response.text}")
        return response.content


async def generate_openai_chunked(
    text: str, model: str, voice: str | None, api_key: str, instruct: str | None
) -> bytes:
    """Generate OpenAI TTS by chunking oversized requests and concatenating output WAV."""
    chunks = chunk_text_by_tokens(
        text,
        max_tokens=OPENAI_TTS_MAX_INPUT_TOKENS,
        provider="openai",
        model=model,
    )
    if not chunks:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    logger.info("OpenAI chunked generation: %d chunks", len(chunks))
    audio_chunks: List[bytes] = []
    for chunk_text in chunks:
        audio_chunks.append(await generate_openai(chunk_text, model, voice, api_key, instruct))

    try:
        return concatenate_audio_chunks(audio_chunks, output_ext="wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to concatenate OpenAI chunks: {e}") from e


async def generate_elevenlabs_chunked(
    text: str, model: str, voice: str | None, api_key: str, instruct: str | None
) -> bytes:
    """Generate ElevenLabs TTS by chunking oversized requests and concatenating output MP3."""
    chunks = chunk_text_by_tokens(
        text,
        max_tokens=ELEVENLABS_TTS_MAX_INPUT_TOKENS,
        provider="elevenlabs",
        model=model,
    )
    if not chunks:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    logger.info("ElevenLabs chunked generation: %d chunks", len(chunks))
    audio_chunks: List[bytes] = []
    for chunk_text in chunks:
        audio_chunks.append(await generate_elevenlabs(chunk_text, model, voice, api_key, instruct))

    try:
        return concatenate_audio_chunks(audio_chunks, output_ext="mp3")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to concatenate ElevenLabs chunks: {e}") from e


async def generate_mlx_voice_clone(text: str, model: str, voice_id: str | None, instruct: str | None) -> bytes:
    """Generate speech using MLX-Audio with voice cloning (CSM / Qwen3-TTS)."""
    if not voice_id:
        raise HTTPException(status_code=400, detail="Voice ID required for voice cloning")

    voices_metadata = load_voices_metadata()
    if voice_id not in voices_metadata:
        raise HTTPException(status_code=404, detail="Voice not found")

    voice_info = voices_metadata[voice_id]
    ref_filename = voice_info.get("ref_filename") or voice_info.get("filename")
    audio_path = UPLOADS_DIR / ref_filename if ref_filename else UPLOADS_DIR / voice_info["filename"]
    if not audio_path.exists() and voice_info.get("filename"):
        audio_path = UPLOADS_DIR / voice_info["filename"]
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Voice audio file not found")

    ref_text = voice_info.get("transcript", "")
    if not ref_text:
        raise HTTPException(status_code=400, detail="Voice transcript is required for cloning")

    word_count = len(text.split())
    timeout = min(max(60.0, 60.0 + (word_count * 2)), 3600.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{MLX_AUDIO_URL}/v1/audio/speech",
            json={
                "model": model,
                "input": text,
                "instruct": instruct,
                "ref_audio": str(audio_path.absolute()),
                "ref_text": ref_text,
            },
        )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"MLX-Audio voice clone error: {response.text}")
        return response.content


def _get_uploaded_voice_path(voice_id: str) -> Path:
    """Resolve an uploaded voice ID to a file path."""
    voices_metadata = load_voices_metadata()
    if voice_id not in voices_metadata:
        raise HTTPException(status_code=404, detail="Voice not found")

    voice_info = voices_metadata[voice_id]
    ref_filename = voice_info.get("ref_filename") or voice_info.get("filename")
    audio_path = UPLOADS_DIR / ref_filename if ref_filename else UPLOADS_DIR / voice_info["filename"]
    if not audio_path.exists() and voice_info.get("filename"):
        audio_path = UPLOADS_DIR / voice_info["filename"]
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Voice audio file not found")
    return audio_path


def _resolve_voice_reference(request: TTSRequest) -> tuple[str | None, str | None]:
    """Resolve built-in voice and uploaded reference voice IDs from request fields."""
    requested_voice = request.voice.strip() if request.voice else None
    requested_voice_id = _normalize_uploaded_voice_id(request.voice_id)

    # Frontend compatibility: allow legacy uploaded-voice prefix in the normal voice field.
    if not requested_voice_id and requested_voice:
        normalized_from_voice = _normalize_uploaded_voice_id(requested_voice)
        if normalized_from_voice:
            known_metadata = load_voices_metadata()
            if normalized_from_voice in known_metadata:
                requested_voice_id = normalized_from_voice
                requested_voice = None

    return requested_voice, requested_voice_id


async def generate_pocket_tts_stream(
    text: str, voice: str | None = None, voice_id: str | None = None
) -> AsyncGenerator[bytes, None]:
    """Generate Pocket TTS audio using the streaming /tts endpoint."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    form_data = {"text": text}
    files: dict[str, tuple[str, object, str]] | None = None
    uploaded_voice_file = None
    timeout = httpx.Timeout(1200.0, connect=5.0, read=1200.0, write=10.0, pool=10.0)

    if voice_id:
        audio_path = _get_uploaded_voice_path(voice_id)
        uploaded_voice_file = open(audio_path, "rb")
        files = {"voice_wav": (audio_path.name, uploaded_voice_file, "audio/wav")}
    else:
        form_data["voice_url"] = voice or "alba"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{POCKET_TTS_URL}/tts",
                data=form_data,
                files=files,
            ) as response:
                if response.status_code != 200:
                    error_text = (await response.aread()).decode("utf-8", errors="replace")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Pocket TTS error: {error_text}",
                    )

                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
    finally:
        if uploaded_voice_file:
            uploaded_voice_file.close()


async def generate_pocket_tts(
    text: str, voice: str | None = None, voice_id: str | None = None
) -> bytes:
    """Generate Pocket TTS audio and return all bytes."""
    chunks: List[bytes] = []
    async for chunk in generate_pocket_tts_stream(text, voice, voice_id):
        chunks.append(chunk)
    if not chunks:
        raise HTTPException(status_code=502, detail="Pocket TTS returned no audio data")
    return b"".join(chunks)


def _coerce_pcm16_bytes(chunk: Any) -> bytes:
    """Convert model output into int16 PCM bytes."""
    if chunk is None:
        return b""

    if isinstance(chunk, dict):
        for key in ("audio", "pcm", "chunk", "data", "bytes"):
            if key in chunk:
                return _coerce_pcm16_bytes(chunk[key])
        raise TypeError(f"Unsupported chunk dict keys: {list(chunk.keys())}")

    if isinstance(chunk, memoryview):
        return chunk.tobytes()
    if isinstance(chunk, (bytes, bytearray)):
        return bytes(chunk)

    try:
        import numpy as np
    except Exception:
        np = None

    if np is not None and isinstance(chunk, np.ndarray):
        if np.issubdtype(chunk.dtype, np.floating):
            chunk = (chunk * 32767.0).clip(-32768, 32767).astype(np.int16)
            return chunk.tobytes()
        if chunk.dtype == np.int16:
            return chunk.tobytes()
        if np.issubdtype(chunk.dtype, np.integer):
            chunk = chunk.astype(np.int16, copy=False)
            return chunk.tobytes()
        chunk = chunk.astype(np.int16, copy=False)
        return chunk.tobytes()

    if hasattr(chunk, "detach") and hasattr(chunk, "numpy"):
        try:
            np_chunk = chunk.detach().cpu().numpy()
            return _coerce_pcm16_bytes(np_chunk)
        except Exception:
            pass

    if isinstance(chunk, (list, tuple)):
        if not chunk:
            return b""
        if all(isinstance(v, int) for v in chunk):
            return b"".join(struct.pack("<h", max(-32768, min(32767, int(v))) ) for v in chunk)
        if all(isinstance(v, (int, float)) for v in chunk):
            return b"".join(
                struct.pack("<h", max(-32768, min(32767, int(float(v) * 32767.0))))
                for v in chunk
            )

    raise TypeError(f"Unsupported chunk type for PCM16 conversion: {type(chunk)!r}")


def _pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 24000, channels: int = 1) -> bytes:
    """Build a minimal WAV container around raw PCM16 mono/interleaved samples."""
    if sample_rate <= 0 or channels <= 0:
        sample_rate = 24000
        channels = 1

    data_size = len(pcm_bytes)
    fmt_chunk_size = 16
    audio_format = 1
    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    chunk_size = 36 + data_size

    header = bytearray()
    header.extend(b"RIFF")
    header.extend(struct.pack("<I", chunk_size))
    header.extend(b"WAVE")
    header.extend(b"fmt ")
    header.extend(struct.pack("<I", fmt_chunk_size))
    header.extend(struct.pack("<H", audio_format))
    header.extend(struct.pack("<H", channels))
    header.extend(struct.pack("<I", sample_rate))
    header.extend(struct.pack("<I", byte_rate))
    header.extend(struct.pack("<H", block_align))
    header.extend(struct.pack("<H", bits_per_sample))
    header.extend(b"data")
    header.extend(struct.pack("<I", data_size))
    header.extend(pcm_bytes)
    return bytes(header)


def _sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _resolve_pocket_tts_mlx_runtime(module: Any) -> Any:
    candidates = [
        "PocketTTS",
        "PocketTTSPipeline",
        "PocketTTSModel",
        "Generator",
        "TTS",
    ]
    for name in candidates:
        cls = getattr(module, name, None)
        if isinstance(cls, type):
            try:
                return cls()
            except Exception:
                logger.debug("Could not instantiate %s without args", name, exc_info=True)
                pass

    fn_names = ["stream", "generate", "synthesize", "tts", "predict", "infer", "run"]
    for name in fn_names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn

    if callable(module):
        return module

    raise HTTPException(
        status_code=503,
        detail="pocket-tts-mlx installed but no supported generator call found",
    )


def _invoke_pocket_tts_mlx(runtime: Any, text: str, voice: str, voice_file: Path | None) -> Any:
    """Invoke an unknown pocket-tts-mlx interface with best-effort argument probing."""
    runtime_targets: list[Any] = []
    runtime_targets.append(runtime)

    if not callable(runtime):
        for name in ("stream", "generate", "synthesize", "tts", "predict", "infer", "forward"):
            fn = getattr(runtime, name, None)
            if callable(fn):
                runtime_targets.append(fn)

    args_base = [text]
    kwargs_variants: list[tuple[list[Any], dict[str, Any]]] = [
        (args_base, {"voice": voice}),
        (args_base, {"voice_name": voice}),
        (args_base, {"speaker": voice}),
        (args_base, {}),
        (list(args_base), {"voice": voice, "text": text}),
    ]
    if voice_file is not None:
        kwargs_variants.insert(0, ([], {"text": text, "voice": voice, "voice_wav": str(voice_file)}))
        kwargs_variants.insert(0, ([], {"text": text, "voice_url": str(voice_file)}))
        kwargs_variants.insert(0, ([], {"text": text, "voice_file": str(voice_file)}))
        kwargs_variants.insert(0, ([], {"text": text, "reference_audio": str(voice_file)}))
    if not voice:
        kwargs_variants = [(args, kwargs) for args, kwargs in kwargs_variants if "voice" not in kwargs]

    for target in runtime_targets:
        if not callable(target):
            continue
        for args, kwargs in kwargs_variants:
            try:
                return target(*args, **kwargs)
            except TypeError as exc:
                logger.debug("Runtime call mismatch for target=%s args=%s kwargs=%s: %s", target, args, kwargs, exc)
                continue

    raise RuntimeError("No supported invocation signature for pocket-tts-mlx runtime")


async def _run_pocket_tts_mlx_stream(
    text: str, voice: str | None, voice_id: str | None
) -> AsyncGenerator[bytes, None]:
    voice_name = (voice or "alba").strip() or "alba"
    voice_file = None
    if voice_id:
        voice_file = _get_uploaded_voice_path(voice_id)

    runtime = await _get_pocket_tts_mlx_runtime()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    stop_event = threading.Event()
    task = None

    def producer() -> None:
        try:
            result = _invoke_pocket_tts_mlx(runtime, text, voice_name, voice_file)
            if inspect.isawaitable(result):
                result = asyncio.run(result)

            if isinstance(result, (bytes, bytearray, memoryview)):
                loop.call_soon_threadsafe(queue.put_nowait, ("chunk", _coerce_pcm16_bytes(result)))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
                return

            if inspect.isasyncgen(result):
                async def drain_async_gen() -> None:
                    async for chunk in result:
                        if stop_event.is_set():
                            return
                        loop.call_soon_threadsafe(queue.put_nowait, ("chunk", _coerce_pcm16_bytes(chunk)))
                asyncio.run(drain_async_gen())
            else:
                for chunk in iter(result):
                    if stop_event.is_set():
                        return
                    loop.call_soon_threadsafe(queue.put_nowait, ("chunk", _coerce_pcm16_bytes(chunk)))

            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
        except asyncio.CancelledError:
            stop_event.set()
            loop.call_soon_threadsafe(queue.put_nowait, ("cancelled", None))
        except Exception as exc:
            logger.exception("pocket-tts-mlx stream failed")
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

    task = asyncio.create_task(asyncio.to_thread(producer))

    try:
        while True:
            item = await queue.get()
            kind, value = item
            if kind == "chunk":
                if value:
                    yield value
            elif kind == "done":
                return
            elif kind == "error":
                raise HTTPException(status_code=500, detail=f"pocket-tts-mlx runtime error: {value}")
            elif kind == "cancelled":
                return
    finally:
        stop_event.set()
        if task is not None:
            try:
                await task
            except Exception:
                logger.debug("pocket-tts-mlx producer task completed with error", exc_info=True)


async def _get_pocket_tts_mlx_runtime() -> Any:
    """Load and cache pocket-tts-mlx runtime once per process."""
    global _pocket_tts_mlx_module, _pocket_tts_mlx_runtime, _pocket_tts_mlx_init_time_ms
    if _pocket_tts_mlx_module is not None and _pocket_tts_mlx_runtime is not None:
        return _pocket_tts_mlx_runtime

    async with _pocket_tts_mlx_lock:
        if _pocket_tts_mlx_module is not None and _pocket_tts_mlx_runtime is not None:
            return _pocket_tts_mlx_runtime

        start_time = time.perf_counter()
        try:
            import importlib
            module = importlib.import_module("pocket_tts_mlx")
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="pocket-tts-mlx is not installed. Install it with `pip install pocket-tts-mlx`.",
            ) from exc

        runtime = _resolve_pocket_tts_mlx_runtime(module)
        _pocket_tts_mlx_module = module
        _pocket_tts_mlx_runtime = runtime

        for attr in ("sample_rate", "sr", "sampling_rate", "sampleRate", "SAMPLE_RATE"):
            candidate = getattr(runtime, attr, None)
            if isinstance(candidate, int) and candidate > 0:
                _pocket_tts_mlx_sample_rate = candidate
                break
            if _pocket_tts_mlx_module is not None:
                module_candidate = getattr(_pocket_tts_mlx_module, attr, None)
                if isinstance(module_candidate, int) and module_candidate > 0:
                    _pocket_tts_mlx_sample_rate = module_candidate
                    break

        for attr in ("channels", "num_channels", "channel_count"):
            candidate = getattr(runtime, attr, None)
            if isinstance(candidate, int) and candidate > 0:
                _pocket_tts_mlx_channels = candidate
                break
            if _pocket_tts_mlx_module is not None:
                module_candidate = getattr(_pocket_tts_mlx_module, attr, None)
                if isinstance(module_candidate, int) and module_candidate > 0:
                    _pocket_tts_mlx_channels = module_candidate
                    break

        _pocket_tts_mlx_init_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info("Loaded pocket-tts-mlx runtime in %sms", _pocket_tts_mlx_init_time_ms)
        return runtime


async def generate_pocket_tts_mlx(request_text: str, voice: str | None, voice_id: str | None) -> bytes:
    chunks: List[bytes] = []
    async for chunk in _run_pocket_tts_mlx_stream(request_text, voice, voice_id):
        chunks.append(chunk)
    if not chunks:
        raise HTTPException(status_code=502, detail="pocket-tts-mlx returned no audio data")
    audio_data = _pcm_to_wav_bytes(
        b"".join(chunks),
        sample_rate=_pocket_tts_mlx_sample_rate,
        channels=_pocket_tts_mlx_channels,
    )
    save_generated_audio(audio_data, "pocket-tts-mlx", "wav")
    return audio_data


async def generate_pocket_tts_mlx_stream(
    request_text: str, voice: str | None = None, voice_id: str | None = None
) -> AsyncGenerator[str, None]:
    if not request_text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    await _get_pocket_tts_mlx_runtime()
    start = time.perf_counter()
    stream_init_ms = 0.0
    if _pocket_tts_mlx_init_time_ms is not None:
        stream_init_ms = _pocket_tts_mlx_init_time_ms
    else:
        stream_init_ms = round((time.perf_counter() - start) * 1000, 2)

    meta = {
        "sample_rate": _pocket_tts_mlx_sample_rate,
        "channels": _pocket_tts_mlx_channels,
        "sample_format": "pcm_s16le",
        "initialization_ms": stream_init_ms,
    }
    yielded_audio = False
    pcm_chunks: List[bytes] = []
    yield _sse_event("meta", meta)

    chunks = 0
    bytes_sent = 0
    errored = False
    try:
        async for chunk in _run_pocket_tts_mlx_stream(request_text, voice, voice_id):
            if not chunk:
                continue
            chunks += 1
            bytes_sent += len(chunk)
            yielded_audio = True
            pcm_chunks.append(chunk)
            yield _sse_event(
                "audio",
                {
                    "chunk_index": chunks,
                    "chunk": base64.b64encode(chunk).decode("ascii"),
                    "chunk_bytes": len(chunk),
                },
            )
    except HTTPException as exc:
        errored = True
        yield _sse_event("error", {"error": exc.detail})
    except Exception as exc:
        errored = True
        logger.exception("pocket-tts-mlx stream loop failed")
        yield _sse_event("error", {"error": f"Failed to stream: {exc}"})

    if errored:
        return

    if yielded_audio:
        wav_data = _pcm_to_wav_bytes(
            b"".join(pcm_chunks),
            sample_rate=_pocket_tts_mlx_sample_rate,
            channels=_pocket_tts_mlx_channels,
        )
        saved_output = save_generated_audio(wav_data, "pocket-tts-mlx", "wav")
        yield _sse_event(
            "done",
            {
                "status": "complete",
                "chunks": chunks,
                "bytes": bytes_sent,
                "sample_rate": _pocket_tts_mlx_sample_rate,
                **({"saved_to": saved_output} if saved_output else {}),
            },
        )
    else:
        yield _sse_event("error", {"error": "No audio chunks were produced."})


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using selected provider."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if request.provider not in PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {request.provider}")

    provider_cfg = PROVIDERS[request.provider]
    voice_input, voice_id = _resolve_voice_reference(request)

    if provider_cfg.get("requires_voice_id") and not voice_id and not voice_input:
        raise HTTPException(status_code=400, detail="voice_id required for this provider")

    if provider_cfg.get("requires_reference_transcript") and voice_id:
        voices_metadata = load_voices_metadata()
        if voice_id not in voices_metadata:
            raise HTTPException(status_code=404, detail="Voice not found")
        if not voices_metadata[voice_id].get("transcript", "").strip():
            raise HTTPException(
                status_code=400,
                detail=(
                    "Uploaded voice has no transcript."
                    " Retranscribe or provide a transcript when uploading."
                ),
            )

    try:
        api_key = request.api_key

        if request.provider == "mlx-audio":
            audio_data = await generate_mlx_audio(request.text, request.model, voice_input, request.instruct)
            media_type = "audio/mpeg"
            ext = "mp3"

        elif request.provider == "elevenlabs":
            api_key = api_key or ELEVENLABS_API_KEY
            audio_data = await generate_elevenlabs(request.text, request.model, voice_input, api_key, request.instruct)
            media_type = "audio/mpeg"
            ext = "mp3"

        elif request.provider == "openai":
            api_key = api_key or OPENAI_API_KEY
            audio_data = await generate_openai(request.text, request.model, voice_input, api_key, request.instruct)
            media_type = "audio/wav"
            ext = "wav"

        elif request.provider == "mlx-voice-clone":
            audio_data = await generate_mlx_voice_clone(request.text, request.model, voice_id, request.instruct)
            media_type = "audio/mpeg"
            ext = "mp3"
        elif request.provider == "pocket-tts":
            media_type = "audio/wav"
            ext = "wav"
            if request.stream:
                return StreamingResponse(
                    generate_pocket_tts_stream(request.text, voice_input, voice_id),
                    media_type=media_type,
                    headers={"Content-Disposition": f"attachment; filename=narrate_{uuid.uuid4().hex[:8]}.wav"},
                )
            audio_data = await generate_pocket_tts(request.text, voice_input, voice_id)
        elif request.provider == "pocket-tts-mlx":
            media_type = "audio/wav"
            ext = "wav"
            if request.stream:
                raise HTTPException(
                    status_code=400,
                    detail="pocket-tts-mlx streaming uses /api/tts/stream",
                )
            audio_data = await generate_pocket_tts_mlx(request.text, voice_input, voice_id)

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {request.provider}")

        saved_path = save_generated_audio(audio_data, request.provider, ext)
        response_headers = {
            "Content-Disposition": f"attachment; filename=narrate_{uuid.uuid4().hex[:8]}.{ext}",
        }
        if saved_path:
            response_headers["X-Generated-Output"] = saved_path

        return StreamingResponse(
            iter([audio_data]),
            media_type=media_type,
            headers=response_headers,
        )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="TTS generation timed out")
    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to {request.provider}. Check your connection or API settings. ({e})",
        )


@app.post("/api/tts/stream")
async def stream_text_to_speech(request: TTSRequest):
    """Stream PCM chunks using SSE for providers with real-time support."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if request.provider not in PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {request.provider}")

    if request.provider != "pocket-tts-mlx":
        raise HTTPException(
            status_code=400,
            detail="stream endpoint currently supports pocket-tts-mlx only",
        )

    voice_input, voice_id = _resolve_voice_reference(request)
    if request.provider == "pocket-tts-mlx":
        provider_cfg = PROVIDERS[request.provider]
        if provider_cfg.get("requires_reference_transcript") and voice_id:
            voices_metadata = load_voices_metadata()
            if voice_id not in voices_metadata:
                raise HTTPException(status_code=404, detail="Voice not found")
            if not voices_metadata[voice_id].get("transcript", "").strip():
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Uploaded voice has no transcript."
                        " Retranscribe or provide a transcript when uploading."
                    ),
                )

    try:
        await _get_pocket_tts_mlx_runtime()
        return StreamingResponse(
            generate_pocket_tts_mlx_stream(request.text, voice_input, voice_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start stream: {exc}") from exc


@app.post("/api/tts/chunked")
async def generate_chunked_audio(request: TTSRequest, background_tasks: BackgroundTasks):
    """Generate TTS for long text using chunking (mlx-voice-clone only)."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if request.provider != "mlx-voice-clone":
        raise HTTPException(status_code=400, detail="Chunked generation only supports mlx-voice-clone provider")

    _, voice_id = _resolve_voice_reference(request)
    if not voice_id:
        raise HTTPException(status_code=400, detail="Voice ID required for voice cloning")

    metadata = load_voices_metadata()
    if voice_id not in metadata:
        raise HTTPException(status_code=404, detail="Voice not found")
    if not metadata[voice_id].get("transcript", "").strip():
        raise HTTPException(
            status_code=400,
            detail="Uploaded voice has no transcript. Retranscribe or provide transcript at upload time.",
        )

    session_id = uuid.uuid4().hex
    generation_sessions[session_id] = {
        "status": "processing",
        "progress": {"current": 0, "total": 0},
        "created_at": datetime.now(),
        "provider": request.provider,
        "model": request.model,
        "voice_id": voice_id,
        "instruct": request.instruct,
    }

    background_tasks.add_task(
        generate_with_progress,
        text=request.text,
        provider=request.provider,
        model=request.model,
        voice_id=voice_id,
        session_id=session_id,
        instruct=request.instruct,
        max_concurrent=request.max_concurrent,
    )

    word_count = len(request.text.split())
    return {"session_id": session_id, "status": "processing", "word_count": word_count}


@app.get("/api/tts/status/{session_id}")
async def get_generation_status(session_id: str):
    if session_id not in generation_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = generation_sessions[session_id]
    return {
        "status": session["status"],
        "progress": session.get("progress"),
        "error": session.get("error"),
        "word_count": session.get("word_count"),
        "chunks": session.get("chunks"),
    }


@app.get("/api/tts/download/{session_id}")
async def download_generated_audio(session_id: str):
    if session_id not in generation_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = generation_sessions[session_id]
    if session["status"] != "complete":
        raise HTTPException(status_code=400, detail=f"Generation not complete (status: {session['status']})")

    audio_data = session["audio_data"]
    return StreamingResponse(
        iter([audio_data]),
        media_type="audio/mpeg",
        headers={"Content-Disposition": f"attachment; filename=narrate_{session_id[:8]}.mp3"},
    )


@app.post("/api/tts/cancel/{session_id}")
async def cancel_generation(session_id: str):
    if session_id not in generation_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    generation_sessions[session_id]["status"] = "cancelled"
    return {"status": "cancelled", "session_id": session_id}


# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3000)
