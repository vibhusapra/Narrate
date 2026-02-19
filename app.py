"""
Narrate - Text to Speech Audiobook Generator

Multi-provider TTS with local voice cloning (MLX-Audio) optimized for Apple Silicon.
"""

import asyncio
import json
import logging
import re
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("narrate")

# Optional: mlx-whisper for auto-transcription
try:
    import mlx_whisper

    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

app = FastAPI(title="Narrate", description="Text to Speech Audiobook Generator")

# Voice uploads directory
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)
VOICES_METADATA_FILE = UPLOADS_DIR / "voices.json"

# Max upload size: 25 MB
MAX_UPLOAD_SIZE = 25 * 1024 * 1024

# Voice sample preprocessing (speed + reliability on macOS)
VOICE_SAMPLE_MAX_SECONDS = float(os.getenv("VOICE_SAMPLE_MAX_SECONDS", "10"))

# API Configuration
MLX_AUDIO_URL = os.getenv("MLX_AUDIO_URL", "http://127.0.0.1:8000")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Session storage for chunked generation (in-memory)
generation_sessions: Dict[str, dict] = {}
session_locks: Dict[str, asyncio.Lock] = {}

# Provider configurations
PROVIDERS = {
    "mlx-audio": {
        "name": "MLX-Audio (Local)",
        "description": "Local TTS on Apple Silicon",
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
}


class TTSRequest(BaseModel):
    text: str
    provider: str = "mlx-audio"
    model: str = "mlx-community/Spark-TTS-0.5B-bf16"
    voice: str | None = None
    instruct: str | None = None
    api_key: str | None = None
    voice_id: str | None = None  # For voice cloning - references uploaded voice
    max_concurrent: int = 5  # Max parallel chunks for chunked generation


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


def transcribe_audio(audio_path: Path) -> str:
    """Transcribe audio file using mlx-whisper (fast default)."""
    if not HAS_WHISPER:
        raise HTTPException(
            status_code=400,
            detail="Transcript required. Install mlx-whisper for auto-transcription: uv pip install mlx-whisper",
        )

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo="mlx-community/whisper-tiny",
    )
    return result.get("text", "").strip()


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


def concatenate_audio_chunks(audio_chunks: List[bytes], crossfade_ms: int = 20) -> bytes:
    """Concatenate MP3 chunks using ffmpeg (fast, avoids Python audio libs)."""
    if not audio_chunks:
        raise ValueError("No audio chunks to concatenate")
    if len(audio_chunks) == 1:
        return audio_chunks[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        chunk_files: List[Path] = []
        for i, chunk_bytes in enumerate(audio_chunks):
            chunk_file = tmpdir_path / f"chunk_{i:04d}.mp3"
            with open(chunk_file, "wb") as f:
                f.write(chunk_bytes)
            chunk_files.append(chunk_file)

        concat_list = tmpdir_path / "concat_list.txt"
        with open(concat_list, "w") as f:
            for chunk_file in chunk_files:
                safe_name = str(chunk_file).replace("'", "'\\''")
                f.write(f"file '{safe_name}'\n")

        output_file = tmpdir_path / "combined.mp3"

        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-c",
                "copy",
                "-y",
                str(output_file),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

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


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("Narrate - Text to Speech Server")
    logger.info("=" * 60)
    logger.info(f"MLX-Audio URL: {MLX_AUDIO_URL}")
    logger.info(f"ElevenLabs API Key: {'✓ Set' if ELEVENLABS_API_KEY else '✗ Not set'}")
    logger.info(f"OpenAI API Key: {'✓ Set' if OPENAI_API_KEY else '✗ Not set'}")
    logger.info(f"MLX-Whisper: {'✓ Available' if HAS_WHISPER else '✗ Not available'}")
    logger.info(f"Voice sample max seconds: {VOICE_SAMPLE_MAX_SECONDS:.0f}s")
    logger.info(f"Uploaded voices: {len(load_voices_metadata())}")
    logger.info("=" * 60)


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/providers")
async def get_providers():
    """Get available TTS providers and their models."""
    providers_response: dict = {pid: cfg.copy() for pid, cfg in PROVIDERS.items()}

    voices_metadata = load_voices_metadata()
    providers_response["mlx-voice-clone"]["voices"] = {
        vid: v["name"] for vid, v in voices_metadata.items()
    }
    return {"providers": providers_response}


@app.get("/api/health")
async def health_check():
    """Check provider connectivity."""
    status = {
        "mlx_audio": "unknown",
        "elevenlabs": "configured" if ELEVENLABS_API_KEY else "no_api_key",
        "openai": "configured" if OPENAI_API_KEY else "no_api_key",
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{MLX_AUDIO_URL}/v1/models")
            status["mlx_audio"] = "connected" if response.status_code == 200 else "error"
    except Exception:
        status["mlx_audio"] = "disconnected"

    return {"status": "ok", "providers": status}


@app.post("/api/upload-voice")
async def upload_voice(
    file: UploadFile = File(...),
    name: str = Form(...),
    transcript: str = Form(""),
):
    """Upload a voice sample for cloning. Transcript is auto-generated if not provided."""
    logger.info(
        f"Voice Upload: name='{name}', file='{file.filename}', type={file.content_type}"
    )

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

    transcript = transcript.strip()
    if not transcript:
        logger.info("Auto-transcribing audio with mlx-whisper...")
        transcript = transcribe_audio(audio_for_transcription)

    voices_metadata = load_voices_metadata()
    voices_metadata[voice_id] = {
        "name": name,
        "transcript": transcript,
        "filename": raw_audio_path.name,
        "ref_filename": processed_path.name if processed_path else None,
        "original_filename": file.filename,
    }
    save_voices_metadata(voices_metadata)

    return {
        "voice_id": voice_id,
        "name": name,
        "filename": raw_audio_path.name,
        "ref_filename": processed_path.name if processed_path else None,
        "transcript": transcript,
    }


@app.get("/api/voices")
async def list_voices():
    """List all uploaded voice samples."""
    voices_metadata = load_voices_metadata()
    voices = [
        {
            "id": vid,
            "name": v["name"],
            "transcript": v.get("transcript", ""),
            "filename": v.get("filename", ""),
        }
        for vid, v in voices_metadata.items()
    ]
    return {"voices": voices}


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
            raise HTTPException(status_code=response.status_code, detail=f"ElevenLabs error: {response.text}")
        return response.content


async def generate_openai(text: str, model: str, voice: str | None, api_key: str, instruct: str | None) -> bytes:
    """Generate speech using OpenAI TTS API."""
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key required")

    voice_id = voice or "alloy"

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
            raise HTTPException(status_code=response.status_code, detail=f"OpenAI error: {response.text}")
        return response.content


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


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using selected provider."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if request.provider not in PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {request.provider}")

    try:
        api_key = request.api_key

        if request.provider == "mlx-audio":
            audio_data = await generate_mlx_audio(request.text, request.model, request.voice, request.instruct)
            media_type = "audio/mpeg"
            ext = "mp3"

        elif request.provider == "elevenlabs":
            api_key = api_key or ELEVENLABS_API_KEY
            audio_data = await generate_elevenlabs(request.text, request.model, request.voice, api_key, request.instruct)
            media_type = "audio/mpeg"
            ext = "mp3"

        elif request.provider == "openai":
            api_key = api_key or OPENAI_API_KEY
            audio_data = await generate_openai(request.text, request.model, request.voice, api_key, request.instruct)
            media_type = "audio/wav"
            ext = "wav"

        elif request.provider == "mlx-voice-clone":
            voice_id = request.voice_id or request.voice
            audio_data = await generate_mlx_voice_clone(request.text, request.model, voice_id, request.instruct)
            media_type = "audio/mpeg"
            ext = "mp3"

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {request.provider}")

        return StreamingResponse(
            iter([audio_data]),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename=narrate_{uuid.uuid4().hex[:8]}.{ext}"},
        )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="TTS generation timed out")
    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to {request.provider}. Check your connection or API settings. ({e})",
        )


@app.post("/api/tts/chunked")
async def generate_chunked_audio(request: TTSRequest, background_tasks: BackgroundTasks):
    """Generate TTS for long text using chunking (mlx-voice-clone only)."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if request.provider != "mlx-voice-clone":
        raise HTTPException(status_code=400, detail="Chunked generation only supports mlx-voice-clone provider")

    voice_id = request.voice_id or request.voice
    if not voice_id:
        raise HTTPException(status_code=400, detail="Voice ID required for voice cloning")

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
