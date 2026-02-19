"""
Narrate - Simplified Text to Speech with Voice Cloning
"""

import httpx
import json
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import os
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("narrate")

app = FastAPI(title="Narrate")

# Voice uploads directory
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)
VOICES_FILE = UPLOADS_DIR / "voices.json"

# Configuration
MLX_AUDIO_URL = os.getenv("MLX_AUDIO_URL", "http://127.0.0.1:8000")

# Auto-transcription
try:
    import mlx_whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

class TTSRequest(BaseModel):
    text: str
    model: str = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
    voice_id: Optional[str] = None

def load_voices():
    """Load voices metadata"""
    if VOICES_FILE.exists():
        with open(VOICES_FILE) as f:
            return json.load(f)
    return {}

def save_voices(voices):
    """Save voices metadata"""
    with open(VOICES_FILE, 'w') as f:
        json.dump(voices, f, indent=2)

@app.get("/")
async def root():
    """Serve the web UI"""
    return FileResponse("static/index_simple.html")

@app.get("/api/voices")
async def list_voices():
    """List all uploaded voices"""
    voices = load_voices()
    return {
        "voices": [
            {
                "id": vid,
                "name": v["name"],
                "duration": v.get("duration"),
                "transcript_length": len(v.get("transcript", ""))
            }
            for vid, v in voices.items()
        ]
    }

@app.post("/api/voices")
async def upload_voice(
    name: str = Form(...),
    audio: UploadFile = File(...),
    transcript: Optional[str] = Form(None)
):
    """Upload a voice for cloning"""
    # Generate voice ID
    voice_id = uuid.uuid4().hex[:12]

    # Save audio file
    ext = Path(audio.filename).suffix or ".mp3"
    filename = f"{voice_id}{ext}"
    audio_path = UPLOADS_DIR / filename

    content = await audio.read()
    with open(audio_path, 'wb') as f:
        f.write(content)

    # Auto-transcribe if no transcript provided
    if not transcript and HAS_WHISPER:
        try:
            result = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo="mlx-community/whisper-large-v3-turbo")
            transcript = result.get("text", "")
        except Exception as e:
            logger.warning(f"Auto-transcription failed: {e}")
            transcript = ""

    # Save metadata
    voices = load_voices()
    voices[voice_id] = {
        "name": name,
        "filename": filename,
        "transcript": transcript or "",
        "original_filename": audio.filename
    }
    save_voices(voices)

    logger.info(f"Uploaded voice: {name} (ID: {voice_id})")
    return {"voice_id": voice_id, "name": name}

@app.delete("/api/voices/{voice_id}")
async def delete_voice(voice_id: str):
    """Delete a voice"""
    voices = load_voices()
    if voice_id not in voices:
        raise HTTPException(404, "Voice not found")

    # Delete audio file
    audio_path = UPLOADS_DIR / voices[voice_id]["filename"]
    if audio_path.exists():
        audio_path.unlink()

    # Remove from metadata
    del voices[voice_id]
    save_voices(voices)

    return {"status": "deleted"}

@app.post("/api/tts")
async def generate_speech(request: TTSRequest):
    """Generate speech using voice cloning"""
    if not request.text.strip():
        raise HTTPException(400, "Text cannot be empty")

    if not request.voice_id:
        raise HTTPException(400, "Voice ID required")

    # Load voice
    voices = load_voices()
    if request.voice_id not in voices:
        raise HTTPException(404, "Voice not found")

    voice = voices[request.voice_id]
    audio_path = UPLOADS_DIR / voice["filename"]

    if not audio_path.exists():
        raise HTTPException(404, "Voice audio file not found")

    transcript = voice.get("transcript", "")
    if not transcript:
        raise HTTPException(400, "Voice needs transcript for cloning")

    # Calculate timeout: 60s base + 2s per word, max 30 minutes
    word_count = len(request.text.split())
    timeout = min(60 + (word_count * 2), 1800)

    logger.info(f"Generating: {word_count} words, timeout={timeout}s, voice={voice['name']}")

    try:
        # Call MLX-Audio
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            response = await client.post(
                f"{MLX_AUDIO_URL}/v1/audio/speech",
                json={
                    "model": request.model,
                    "input": request.text,
                    "ref_audio": str(audio_path.absolute()),
                    "ref_text": transcript
                }
            )

            if response.status_code != 200:
                raise HTTPException(response.status_code, f"MLX-Audio error: {response.text}")

            logger.info(f"✓ Generated {len(response.content)} bytes")
            return StreamingResponse(
                iter([response.content]),
                media_type="audio/mpeg",
                headers={"Content-Disposition": f"attachment; filename=narrate_{request.voice_id[:8]}.mp3"}
            )

    except httpx.TimeoutException:
        raise HTTPException(504, f"Generation timed out after {timeout}s. Try shorter text or check MLX-Audio server.")
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(500, f"Generation failed: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Check if MLX-Audio is available"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{MLX_AUDIO_URL}/v1/models")
            return {"status": "ok", "mlx_audio": response.status_code == 200}
    except:
        return {"status": "error", "mlx_audio": False}

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup():
    logger.info("=" * 60)
    logger.info("🎙️  Narrate - Text to Speech (Simplified)")
    logger.info("=" * 60)
    logger.info(f"MLX-Audio URL: {MLX_AUDIO_URL}")
    logger.info(f"MLX-Whisper: {'✓ Available' if HAS_WHISPER else '✗ Not available'}")
    logger.info(f"Uploaded voices: {len(load_voices())}")
    logger.info("=" * 60)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
