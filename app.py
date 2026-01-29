"""
Narrate - Text to Speech Audiobook Generator
v0.1: Multi-provider TTS (MLX-Audio local, ElevenLabs, OpenAI)
"""

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import os
import uuid

app = FastAPI(title="Narrate", description="Text to Speech Audiobook Generator")

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
    }
}


class TTSRequest(BaseModel):
    text: str
    provider: str = "mlx-audio"
    model: str = "mlx-community/Spark-TTS-0.5B-bf16"
    voice: str | None = None
    api_key: str | None = None


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/providers")
async def get_providers():
    """Get available TTS providers and their models"""
    return {"providers": PROVIDERS}


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
