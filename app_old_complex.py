"""
Narrate - Text to Speech Audiobook Generator
v0.2: Multi-provider TTS with voice cloning support
"""

import asyncio
import httpx
import json
import tempfile
import logging
import re
import subprocess
from typing import List, Dict
from datetime import datetime
from io import BytesIO
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import os
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
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

# API Configuration
MLX_AUDIO_URL = os.getenv("MLX_AUDIO_URL", "http://127.0.0.1:8000")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Session storage for chunked generation (in-memory)
generation_sessions: Dict[str, dict] = {}
session_locks: Dict[str, asyncio.Lock] = {}  # Thread-safe session updates

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
        "voices": {}  # Spark TTS uses default voice
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
        }
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
        }
    },
    "mlx-voice-clone": {
        "name": "MLX Voice Clone (Local)",
        "description": "Clone any voice with ~3-10 seconds of audio",
        "requires_api_key": False,
        "requires_voice_upload": True,
        "models": {
            "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16": "Qwen3-TTS 0.6B (Fast)",
            "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16": "Qwen3-TTS 1.7B (Better quality)",
        },
        "voices": {}  # Populated dynamically from uploaded voices
    }
}


class TTSRequest(BaseModel):
    text: str
    provider: str = "mlx-audio"
    model: str = "mlx-community/Spark-TTS-0.5B-bf16"
    voice: str | None = None
    api_key: str | None = None
    voice_id: str | None = None  # For voice cloning - references uploaded voice
    max_concurrent: int = 5  # Max parallel chunks for chunked generation


def load_voices_metadata() -> dict:
    """Load voice metadata from JSON file"""
    if VOICES_METADATA_FILE.exists():
        with open(VOICES_METADATA_FILE, "r") as f:
            return json.load(f)
    return {}


def save_voices_metadata(metadata: dict):
    """Save voice metadata to JSON file"""
    with open(VOICES_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)


def transcribe_audio(audio_path: Path) -> str:
    """Transcribe audio file using mlx-whisper"""
    if not HAS_WHISPER:
        raise HTTPException(
            status_code=400,
            detail="Transcript required. Install mlx-whisper for auto-transcription: uv pip install mlx-whisper"
        )

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo="mlx-community/whisper-tiny",
    )
    return result.get("text", "").strip()


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex."""
    # Split on sentence boundaries (., !, ?)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_sentences(sentences: List[str], max_words: int = 500) -> List[str]:
    """Group sentences into chunks up to max_words."""
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        # If adding this sentence exceeds max, start new chunk
        if current_word_count + word_count > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count

    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def concatenate_audio_chunks(audio_chunks: List[bytes], crossfade_ms: int = 20) -> bytes:
    """
    Concatenate audio chunks using ffmpeg.

    This function uses ffmpeg instead of pydub to avoid Python 3.13 compatibility issues.
    """
    if not audio_chunks:
        raise ValueError("No audio chunks to concatenate")

    if len(audio_chunks) == 1:
        return audio_chunks[0]

    # Create temporary directory for audio files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Write each chunk to a temporary file
        chunk_files = []
        for i, chunk_bytes in enumerate(audio_chunks):
            chunk_file = tmpdir_path / f"chunk_{i:04d}.mp3"
            with open(chunk_file, 'wb') as f:
                f.write(chunk_bytes)
            chunk_files.append(chunk_file)

        # Create concat file list for ffmpeg
        concat_list = tmpdir_path / "concat_list.txt"
        with open(concat_list, 'w') as f:
            for chunk_file in chunk_files:
                # Escape single quotes in filename
                safe_name = str(chunk_file).replace("'", "'\\''")
                f.write(f"file '{safe_name}'\n")

        # Output file
        output_file = tmpdir_path / "combined.mp3"

        # Use ffmpeg to concatenate
        # -f concat: use concat demuxer
        # -safe 0: allow absolute paths
        # -i: input concat list
        # -c copy: copy codec (no re-encoding for simple concatenation)
        # For crossfade, we'd need a more complex filter, but simple concat is faster
        try:
            subprocess.run([
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_list),
                '-c', 'copy',
                '-y',  # Overwrite output
                str(output_file)
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg concatenation failed: {e.stderr}")
            raise Exception(f"Failed to concatenate audio chunks: {e.stderr}")

        # Read the combined file
        with open(output_file, 'rb') as f:
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
                "total": total_count
            }


async def generate_chunked_tts(
    chunks: List[str],
    model: str,
    voice_id: str,
    session_id: str,
    max_retries: int = 3,
    max_concurrent: int = 5
) -> bytes:
    """
    Generate TTS for chunks IN PARALLEL with concurrency limit.
    Max N chunks processing simultaneously (limited by Semaphore).
    Updates progress as each chunk completes (out-of-order).
    """
    total_chunks = len(chunks)
    logger.info(f"Session {session_id}: Starting PARALLEL generation: {total_chunks} chunks (max {max_concurrent} concurrent)")

    # Semaphore to limit concurrent chunk processing
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create task for each chunk with index tracking
    async def generate_chunk_with_retry(index: int, chunk_text: str):
        """Generate single chunk with retry logic and concurrency limit."""
        async with semaphore:  # Acquire semaphore - only max_concurrent will run
            # Check if session was cancelled
            if session_id in generation_sessions and generation_sessions[session_id]["status"] == "cancelled":
                logger.info(f"Session {session_id}: Chunk {index+1}/{total_chunks} skipped (session cancelled)")
                raise Exception("Generation cancelled by user")

            for attempt in range(max_retries):
                try:
                    logger.info(f"Session {session_id}: Chunk {index+1}/{total_chunks} attempt {attempt+1}: {len(chunk_text)} chars, {len(chunk_text.split())} words")

                    chunk_audio = await generate_mlx_voice_clone_single(
                        chunk_text, model, voice_id, timeout=1800.0  # 30 minutes per chunk
                    )

                    logger.info(f"Session {session_id}: ✓ Chunk {index+1}/{total_chunks} complete ({len(chunk_audio)} bytes)")
                    return (index, chunk_audio)  # Return tuple with index

                except Exception as e:
                    logger.error(f"Session {session_id}: ✗ Chunk {index+1} attempt {attempt+1} failed: {type(e).__name__}: {e}")
                    if attempt == max_retries - 1:
                        raise Exception(f"Chunk {index+1} failed after {max_retries} retries: {e}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s

    # Launch ALL chunks in parallel
    tasks = [
        generate_chunk_with_retry(i, chunk)
        for i, chunk in enumerate(chunks)
    ]

    # Track progress as chunks complete
    completed_count = 0
    audio_results = [None] * total_chunks  # Pre-allocate array

    # Gather with exception handling
    logger.info(f"Session {session_id}: Launching {total_chunks} tasks in parallel...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    for result in results:
        if isinstance(result, Exception):
            # One or more chunks failed
            logger.error(f"Session {session_id}: Chunk generation failed: {result}")
            raise result
        else:
            index, audio_data = result
            audio_results[index] = audio_data
            completed_count += 1

            # Update progress (thread-safe)
            await update_session_progress(session_id, completed_count, total_chunks)

    # All chunks complete - concatenate in order
    logger.info(f"Session {session_id}: All chunks complete. Concatenating {len(audio_results)} audio files...")
    combined_audio = concatenate_audio_chunks(audio_results, crossfade_ms=20)
    logger.info(f"Session {session_id}: ✓ Final audio: {len(combined_audio)} bytes")

    return combined_audio


async def generate_mlx_voice_clone_single(
    text: str,
    model: str,
    voice_id: str,
    timeout: float = 120.0
) -> bytes:
    """
    Generate speech for a single chunk using MLX-Audio voice cloning.
    This is a helper function for chunked generation.
    """
    # Load voice metadata
    voices_metadata = load_voices_metadata()
    if voice_id not in voices_metadata:
        raise HTTPException(status_code=404, detail="Voice not found")

    voice_info = voices_metadata[voice_id]
    audio_path = UPLOADS_DIR / voice_info["filename"]

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Voice audio file not found")

    # Get transcript
    ref_text = voice_info.get("transcript", "")
    if not ref_text:
        raise HTTPException(status_code=400, detail="Voice transcript is required for cloning")

    # Call MLX-Audio with explicit timeout configuration
    timeout_config = httpx.Timeout(
        connect=10.0,      # 10s to establish connection
        read=timeout,      # Use provided timeout for reading response
        write=10.0,        # 10s for writing request
        pool=10.0          # 10s for acquiring connection from pool
    )

    logger.info(f"Sending request to MLX-Audio: {len(text)} chars, timeout={timeout}s")

    async with httpx.AsyncClient(timeout=timeout_config) as client:
        logger.info("Making POST request to MLX-Audio...")
        response = await client.post(
            f"{MLX_AUDIO_URL}/v1/audio/speech",
            json={
                "model": model,
                "input": text,
                "ref_audio": str(audio_path.absolute()),
                "ref_text": ref_text,
            },
        )

        logger.info(f"Received response from MLX-Audio: status={response.status_code}")

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"MLX-Audio voice clone error: {response.text}"
            )

        logger.info("Reading response content...")
        content = response.content
        logger.info(f"Received {len(content)} bytes from MLX-Audio")
        return content


async def generate_with_progress(
    text: str,
    provider: str,
    model: str,
    voice_id: str,
    session_id: str,
    max_concurrent: int = 5
):
    """
    Main orchestrator for chunked generation with progress tracking.
    Runs as a background task.
    """
    try:
        logger.info(f"Session {session_id}: === STARTING chunked generation ===")

        # Split text into chunks
        word_count = len(text.split())
        sentences = split_into_sentences(text)
        chunks = chunk_sentences(sentences, max_words=1000)  # Larger chunks for parallel processing

        logger.info(f"Session {session_id}: {word_count} words → {len(sentences)} sentences → {len(chunks)} chunks")

        # Update session with chunk info
        if session_id in generation_sessions:
            generation_sessions[session_id].update({
                "word_count": word_count,
                "chunks": len(chunks),
                "progress": {"current": 0, "total": len(chunks)}
            })
        else:
            logger.warning(f"Session {session_id}: Session not found in dict!")

        # Generate audio for all chunks
        logger.info(f"Session {session_id}: Calling generate_chunked_tts...")
        audio_data = await generate_chunked_tts(chunks, model, voice_id, session_id, max_concurrent=max_concurrent)
        logger.info(f"Session {session_id}: generate_chunked_tts returned {len(audio_data)} bytes")

        # Mark as complete
        if session_id in generation_sessions:
            generation_sessions[session_id].update({
                "status": "complete",
                "audio_data": audio_data,
                "progress": {"current": len(chunks), "total": len(chunks)}
            })
            logger.info(f"Session {session_id}: ✓✓✓ MARKED AS COMPLETE ✓✓✓")
        else:
            logger.error(f"Session {session_id}: Cannot mark complete - session not found!")

    except Exception as e:
        logger.error(f"Session {session_id}: ✗✗✗ Generation FAILED ✗✗✗: {e}", exc_info=True)
        if session_id in generation_sessions:
            generation_sessions[session_id].update({
                "status": "error",
                "error": str(e)
            })
        else:
            logger.error(f"Session {session_id}: Cannot mark error - session not found!")


@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    logger.info("=" * 60)
    logger.info("🎙️  Narrate - Text to Speech Server")
    logger.info("=" * 60)
    logger.info(f"MLX-Audio URL: {MLX_AUDIO_URL}")
    logger.info(f"ElevenLabs API Key: {'✓ Set' if ELEVENLABS_API_KEY else '✗ Not set'}")
    logger.info(f"OpenAI API Key: {'✓ Set' if OPENAI_API_KEY else '✗ Not set'}")
    logger.info(f"MLX-Whisper: {'✓ Available' if HAS_WHISPER else '✗ Not available'}")

    # Count uploaded voices
    voices_metadata = load_voices_metadata()
    logger.info(f"Uploaded voices: {len(voices_metadata)}")

    logger.info("Available providers:")
    for provider_id in PROVIDERS.keys():
        logger.info(f"  - {provider_id}")
    logger.info("=" * 60)


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/providers")
async def get_providers():
    """Get available TTS providers and their models"""
    # Clone PROVIDERS and add uploaded voices to mlx-voice-clone
    providers_response = {}
    for provider_id, config in PROVIDERS.items():
        providers_response[provider_id] = config.copy()

    # Add uploaded voices to mlx-voice-clone provider
    voices_metadata = load_voices_metadata()
    providers_response["mlx-voice-clone"]["voices"] = {
        vid: v["name"] for vid, v in voices_metadata.items()
    }

    return {"providers": providers_response}


@app.get("/api/health")
async def health_check():
    """Check provider connectivity"""
    status = {
        "mlx_audio": "unknown",
        "elevenlabs": "configured" if ELEVENLABS_API_KEY else "no_api_key",
        "openai": "configured" if OPENAI_API_KEY else "no_api_key",
    }

    # Check MLX-Audio
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
    transcript: str = Form("")
):
    """Upload a voice sample for cloning. Transcript is auto-generated if not provided."""
    logger.info(f"Voice Upload: name='{name}', file='{file.filename}', type={file.content_type}")

    # Validate file type
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")

    # Read file content
    content = await file.read()

    # Check file size
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 25 MB)")

    logger.info(f"  File size: {len(content) / 1024:.1f} KB")

    # Generate unique ID
    voice_id = uuid.uuid4().hex[:12]

    # Determine file extension from content type
    ext = "wav"  # Default to WAV
    if file.content_type == "audio/mpeg":
        ext = "mp3"
    elif file.content_type == "audio/mp4":
        ext = "m4a"
    elif file.filename and "." in file.filename:
        ext = file.filename.rsplit(".", 1)[-1].lower()

    # Save the audio file
    audio_path = UPLOADS_DIR / f"{voice_id}.{ext}"
    with open(audio_path, "wb") as f:
        f.write(content)

    # Auto-transcribe if no transcript provided
    transcript = transcript.strip()
    if not transcript:
        logger.info("  Auto-transcribing audio with mlx-whisper...")
        transcript = transcribe_audio(audio_path)
        logger.info(f"  Transcribed: {len(transcript)} chars")

    # Update metadata
    voices_metadata = load_voices_metadata()
    voices_metadata[voice_id] = {
        "name": name,
        "transcript": transcript,
        "filename": f"{voice_id}.{ext}",
        "original_filename": file.filename,
    }
    save_voices_metadata(voices_metadata)

    logger.info(f"✓ Voice saved with ID: {voice_id}")

    return {
        "voice_id": voice_id,
        "name": name,
        "filename": f"{voice_id}.{ext}",
        "transcript": transcript,
    }


@app.get("/api/voices")
async def list_voices():
    """List all uploaded voice samples"""
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
    """Delete an uploaded voice sample"""
    voices_metadata = load_voices_metadata()

    if voice_id not in voices_metadata:
        raise HTTPException(status_code=404, detail="Voice not found")

    voice_info = voices_metadata[voice_id]
    logger.info(f"Deleting voice: {voice_info['name']} (ID: {voice_id})")

    # Get filename and delete the audio file
    audio_path = UPLOADS_DIR / voice_info["filename"]
    if audio_path.exists():
        audio_path.unlink()

    # Remove from metadata
    del voices_metadata[voice_id]
    save_voices_metadata(voices_metadata)

    logger.info(f"✓ Voice deleted: {voice_id}")
    return {"status": "deleted", "voice_id": voice_id}


async def generate_mlx_audio(text: str, model: str, voice: str) -> bytes:
    """Generate speech using MLX-Audio local server (OpenAI-compatible API)"""
    voice_id = voice or "af_heart"

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{MLX_AUDIO_URL}/v1/audio/speech",
            json={
                "model": model,
                "input": text,
                "voice": voice_id,
            },
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"MLX-Audio error: {response.text}"
            )
        return response.content


async def generate_elevenlabs(text: str, model: str, voice: str, api_key: str) -> bytes:
    """Generate speech using ElevenLabs API"""
    if not api_key:
        raise HTTPException(status_code=400, detail="ElevenLabs API key required")

    voice_id = voice or "21m00Tcm4TlvDq8ikWAM"

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
            },
            json={
                "text": text,
                "model_id": model,
            },
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"ElevenLabs error: {response.text}"
            )
        return response.content


async def generate_openai(text: str, model: str, voice: str, api_key: str) -> bytes:
    """Generate speech using OpenAI API"""
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key required")

    voice_id = voice or "alloy"

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/audio/speech",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "input": text,
                "voice": voice_id,
                "response_format": "wav",
            },
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"OpenAI error: {response.text}"
            )
        return response.content


async def generate_mlx_voice_clone(text: str, model: str, voice_id: str) -> bytes:
    """Generate speech using MLX-Audio with voice cloning (Qwen3-TTS)"""
    if not voice_id:
        raise HTTPException(status_code=400, detail="Voice ID required for voice cloning")

    # Load voice metadata
    voices_metadata = load_voices_metadata()
    if voice_id not in voices_metadata:
        raise HTTPException(status_code=404, detail="Voice not found")

    voice_info = voices_metadata[voice_id]
    audio_path = UPLOADS_DIR / voice_info["filename"]

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Voice audio file not found")

    # Get transcript
    ref_text = voice_info.get("transcript", "")
    if not ref_text:
        raise HTTPException(status_code=400, detail="Voice transcript is required for cloning")

    # Calculate dynamic timeout based on text length
    # Base: 60s, add ~2s per word, max 3600s (1 hour) for non-chunked generation
    word_count = len(text.split())
    timeout = min(max(60.0, 60.0 + (word_count * 2)), 3600.0)

    # Log voice cloning details
    logger.info(f"Voice Cloning Details:")
    logger.info(f"  Voice: {voice_info['name']} (ID: {voice_id})")
    logger.info(f"  Reference audio: {audio_path.name}")
    logger.info(f"  Reference transcript: {len(ref_text)} chars")
    logger.info(f"  Target text: {len(text)} chars ({word_count} words)")
    logger.info(f"  Timeout: {timeout:.0f}s")
    if timeout > 300:
        logger.warning(f"⚠️  Long generation detected ({timeout:.0f}s timeout). This may take several minutes...")
    logger.info(f"Sending request to MLX-Audio...")

    # Call MLX-Audio with JSON - ref_audio param is file path for cloning
    # Voice cloning takes time, especially on first run (model loading)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{MLX_AUDIO_URL}/v1/audio/speech",
            json={
                "model": model,
                "input": text,
                "ref_audio": str(audio_path.absolute()),  # Path to reference audio
                "ref_text": ref_text,
            },
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"MLX-Audio voice clone error: {response.text}"
            )
        return response.content


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using selected provider"""

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if request.provider not in PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {request.provider}")

    # Log incoming request
    text_preview = request.text[:50] + "..." if len(request.text) > 50 else request.text
    word_count = len(request.text.split())
    logger.info(f"TTS Request: provider={request.provider}, model={request.model}, text='{text_preview}'")
    logger.info(f"  Text length: {len(request.text)} chars, {word_count} words")
    if request.voice_id:
        logger.info(f"  Using voice ID: {request.voice_id}")

    # Warn about very long texts
    if word_count > 500:
        logger.warning(f"⚠️  Large text detected ({word_count} words). Generation may take several minutes.")

    try:
        api_key = request.api_key

        if request.provider == "mlx-audio":
            logger.info(f"Generating with MLX-Audio Spark-TTS model={request.model}")
            audio_data = await generate_mlx_audio(request.text, request.model, request.voice)
            media_type = "audio/mpeg"
            ext = "mp3"

        elif request.provider == "elevenlabs":
            api_key = api_key or ELEVENLABS_API_KEY
            logger.info(f"Generating with ElevenLabs model={request.model}, voice={request.voice}")
            audio_data = await generate_elevenlabs(
                request.text, request.model, request.voice, api_key
            )
            media_type = "audio/mpeg"
            ext = "mp3"

        elif request.provider == "openai":
            api_key = api_key or OPENAI_API_KEY
            logger.info(f"Generating with OpenAI model={request.model}, voice={request.voice}")
            audio_data = await generate_openai(
                request.text, request.model, request.voice, api_key
            )
            media_type = "audio/wav"
            ext = "wav"

        elif request.provider == "mlx-voice-clone":
            # Use voice_id from request, or fall back to voice field
            voice_id = request.voice_id or request.voice
            logger.info(f"Generating with Voice Cloning model={request.model}, voice_id={voice_id}")
            audio_data = await generate_mlx_voice_clone(
                request.text, request.model, voice_id
            )
            media_type = "audio/mpeg"
            ext = "mp3"

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {request.provider}")

        logger.info(f"✓ Generated {len(audio_data)} bytes of audio ({media_type})")
        return StreamingResponse(
            iter([audio_data]),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=narrate_{uuid.uuid4().hex[:8]}.{ext}"
            }
        )

    except httpx.TimeoutException:
        logger.error(f"✗ TTS generation timed out (provider={request.provider})")
        raise HTTPException(status_code=504, detail="TTS generation timed out")
    except httpx.ConnectError as e:
        logger.error(f"✗ Cannot connect to {request.provider}: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to {request.provider}. Check your connection or API settings."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ TTS generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tts/chunked")
async def generate_chunked_audio(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    Generate TTS for long text using chunking.
    Returns session_id immediately, generates in background.
    Only supports mlx-voice-clone provider currently.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if request.provider != "mlx-voice-clone":
        raise HTTPException(
            status_code=400,
            detail="Chunked generation only supports mlx-voice-clone provider"
        )

    voice_id = request.voice_id or request.voice
    if not voice_id:
        raise HTTPException(status_code=400, detail="Voice ID required for voice cloning")

    # Generate unique session ID
    session_id = uuid.uuid4().hex

    # Initialize session
    generation_sessions[session_id] = {
        "status": "processing",
        "progress": {"current": 0, "total": 0},
        "created_at": datetime.now(),
        "provider": request.provider,
        "model": request.model,
        "voice_id": voice_id
    }

    # Start background task
    background_tasks.add_task(
        generate_with_progress,
        text=request.text,
        provider=request.provider,
        model=request.model,
        voice_id=voice_id,
        session_id=session_id,
        max_concurrent=request.max_concurrent
    )

    word_count = len(request.text.split())
    logger.info(f"Started chunked generation session {session_id} ({word_count} words)")

    return {
        "session_id": session_id,
        "status": "processing",
        "word_count": word_count
    }


@app.get("/api/tts/status/{session_id}")
async def get_generation_status(session_id: str):
    """
    Poll for generation progress.
    Returns: {"status": "processing|complete|error", "progress": {"current": 5, "total": 38}}
    """
    if session_id not in generation_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = generation_sessions[session_id]
    return {
        "status": session["status"],
        "progress": session.get("progress"),
        "error": session.get("error"),
        "word_count": session.get("word_count"),
        "chunks": session.get("chunks")
    }


@app.get("/api/tts/download/{session_id}")
async def download_generated_audio(session_id: str):
    """
    Download completed audio.
    """
    if session_id not in generation_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = generation_sessions[session_id]
    if session["status"] != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"Generation not complete (status: {session['status']})"
        )

    audio_data = session["audio_data"]
    logger.info(f"Downloading session {session_id}: {len(audio_data)} bytes")

    return StreamingResponse(
        iter([audio_data]),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f"attachment; filename=narrate_{session_id[:8]}.mp3"
        }
    )


@app.post("/api/tts/cancel/{session_id}")
async def cancel_generation(session_id: str):
    """
    Cancel an ongoing generation session.
    """
    if session_id not in generation_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # Mark session as cancelled
    generation_sessions[session_id]["status"] = "cancelled"
    logger.info(f"Session {session_id}: Marked as CANCELLED by user")

    return {"status": "cancelled", "session_id": session_id}


# Mount static files last
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
