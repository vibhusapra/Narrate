"""
Narrate - Text to Speech Audiobook Generator
v0: Simple text input with Fish Speech TTS
"""

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import os
import tempfile
import uuid

app = FastAPI(title="Narrate", description="Text to Speech Audiobook Generator")

# Fish Speech API configuration
FISH_SPEECH_URL = os.getenv("FISH_SPEECH_URL", "http://127.0.0.1:8080")

# Available models/voices (expandable for future)
MODELS = {
    "openaudio-s1-mini": {
        "name": "OpenAudio S1 Mini",
        "description": "0.5B parameter model, fast inference"
    },
    "openaudio-s1": {
        "name": "OpenAudio S1",
        "description": "4B parameter model, highest quality"
    }
}


class TTSRequest(BaseModel):
    text: str
    model: str = "openaudio-s1-mini"
    reference_audio: str | None = None  # For future voice cloning


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/models")
async def get_models():
    """Get available TTS models"""
    return {"models": MODELS}


@app.get("/api/health")
async def health_check():
    """Check if Fish Speech API is available"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{FISH_SPEECH_URL}/")
            return {"status": "ok", "fish_speech": "connected"}
    except Exception as e:
        return {"status": "degraded", "fish_speech": "disconnected", "error": str(e)}


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using Fish Speech"""

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Fish Speech API endpoint for TTS
            # The API uses a streaming endpoint
            tts_payload = {
                "text": request.text,
                "format": "wav",
                "streaming": False,
            }

            response = await client.post(
                f"{FISH_SPEECH_URL}/v1/tts",
                json=tts_payload,
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Fish Speech API error: {response.text}"
                )

            # Return audio as streaming response
            return StreamingResponse(
                iter([response.content]),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"attachment; filename=narrate_{uuid.uuid4().hex[:8]}.wav"
                }
            )

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="TTS generation timed out")
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Fish Speech API. Make sure it's running on " + FISH_SPEECH_URL
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount static files last to avoid catching API routes
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
