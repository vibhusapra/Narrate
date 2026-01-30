"""
Narrate - Text to Speech Audiobook Generator
v0.2: Multi-provider TTS with voice cloning support
"""

import httpx
import json
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import os
import uuid

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
    # Validate file type
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")

    # Read file content
    content = await file.read()

    # Check file size
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 25 MB)")

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
        transcript = transcribe_audio(audio_path)

    # Update metadata
    voices_metadata = load_voices_metadata()
    voices_metadata[voice_id] = {
        "name": name,
        "transcript": transcript,
        "filename": f"{voice_id}.{ext}",
        "original_filename": file.filename,
    }
    save_voices_metadata(voices_metadata)

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

    # Get filename and delete the audio file
    voice_info = voices_metadata[voice_id]
    audio_path = UPLOADS_DIR / voice_info["filename"]
    if audio_path.exists():
        audio_path.unlink()

    # Remove from metadata
    del voices_metadata[voice_id]
    save_voices_metadata(voices_metadata)

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

    # Call MLX-Audio with JSON - ref_audio param is file path for cloning
    # Voice cloning takes time, especially on first run (model loading)
    async with httpx.AsyncClient(timeout=300.0) as client:
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

    try:
        api_key = request.api_key

        if request.provider == "mlx-audio":
            audio_data = await generate_mlx_audio(request.text, request.model, request.voice)
            media_type = "audio/mpeg"
            ext = "mp3"

        elif request.provider == "elevenlabs":
            api_key = api_key or ELEVENLABS_API_KEY
            audio_data = await generate_elevenlabs(
                request.text, request.model, request.voice, api_key
            )
            media_type = "audio/mpeg"
            ext = "mp3"

        elif request.provider == "openai":
            api_key = api_key or OPENAI_API_KEY
            audio_data = await generate_openai(
                request.text, request.model, request.voice, api_key
            )
            media_type = "audio/wav"
            ext = "wav"

        elif request.provider == "mlx-voice-clone":
            # Use voice_id from request, or fall back to voice field
            voice_id = request.voice_id or request.voice
            audio_data = await generate_mlx_voice_clone(
                request.text, request.model, voice_id
            )
            media_type = "audio/mpeg"
            ext = "mp3"

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {request.provider}")

        return StreamingResponse(
            iter([audio_data]),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=narrate_{uuid.uuid4().hex[:8]}.{ext}"
            }
        )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="TTS generation timed out")
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to {request.provider}. Check your connection or API settings."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount static files last
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
