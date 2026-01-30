"""
Integration tests for TTS generation.
These tests require external services to be running or valid API keys.

Run with: uv run --extra dev pytest tests/test_integration.py -v -s
Skip with: uv run --extra dev pytest tests/test_api.py -v (unit tests only)
"""

import os
import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture
def client():
    return TestClient(app)


def is_mlx_audio_running():
    """Check if MLX-Audio server is available"""
    import httpx
    try:
        response = httpx.get("http://127.0.0.1:8000/v1/models", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def has_openai_key():
    """Check if OpenAI API key is available"""
    return bool(os.getenv("OPENAI_API_KEY"))


def has_elevenlabs_key():
    """Check if ElevenLabs API key is available"""
    return bool(os.getenv("ELEVENLABS_API_KEY"))


class TestMLXAudioIntegration:
    """Integration tests for MLX-Audio local TTS"""

    @pytest.mark.skipif(not is_mlx_audio_running(), reason="MLX-Audio server not running")
    def test_generate_audio_with_spark_tts(self, client):
        """Test generating audio with Spark TTS model"""
        response = client.post("/api/tts", json={
            "text": "Hello, this is a test.",
            "provider": "mlx-audio",
            "model": "mlx-community/Spark-TTS-0.5B-bf16",
        })

        assert response.status_code == 200, f"Error: {response.text}"
        assert response.headers["content-type"] == "audio/mpeg"
        assert len(response.content) > 1000  # Should have actual audio data

    @pytest.mark.skipif(not is_mlx_audio_running(), reason="MLX-Audio server not running")
    def test_generate_audio_with_spark_8bit(self, client):
        """Test generating audio with Spark TTS 8-bit model"""
        response = client.post("/api/tts", json={
            "text": "Hello, this is a test.",
            "provider": "mlx-audio",
            "model": "mlx-community/Spark-TTS-0.5B-8bit",
        })

        assert response.status_code == 200, f"Error: {response.text}"
        assert response.headers["content-type"] == "audio/mpeg"
        assert len(response.content) > 1000

    @pytest.mark.skipif(not is_mlx_audio_running(), reason="MLX-Audio server not running")
    def test_generate_long_text(self, client):
        """Test generating audio with longer text"""
        long_text = "This is a longer piece of text. " * 5
        response = client.post("/api/tts", json={
            "text": long_text,
            "provider": "mlx-audio",
            "model": "mlx-community/Spark-TTS-0.5B-bf16",
        })

        assert response.status_code == 200, f"Error: {response.text}"
        assert len(response.content) > 5000  # Longer audio


class TestOpenAIIntegration:
    """Integration tests for OpenAI TTS"""

    @pytest.mark.skipif(not has_openai_key(), reason="OPENAI_API_KEY not set")
    def test_generate_audio_openai(self, client):
        """Test generating audio with OpenAI"""
        response = client.post("/api/tts", json={
            "text": "Hello from OpenAI.",
            "provider": "openai",
            "model": "tts-1",
            "voice": "alloy",
            "api_key": os.getenv("OPENAI_API_KEY")
        })

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert len(response.content) > 1000

    @pytest.mark.skipif(not has_openai_key(), reason="OPENAI_API_KEY not set")
    def test_openai_different_voices(self, client):
        """Test different OpenAI voices"""
        voices = ["alloy", "nova", "shimmer"]

        for voice in voices:
            response = client.post("/api/tts", json={
                "text": "Test.",
                "provider": "openai",
                "model": "tts-1",
                "voice": voice,
                "api_key": os.getenv("OPENAI_API_KEY")
            })
            assert response.status_code == 200, f"Failed for voice: {voice}"


class TestElevenLabsIntegration:
    """Integration tests for ElevenLabs TTS"""

    @pytest.mark.skipif(not has_elevenlabs_key(), reason="ELEVENLABS_API_KEY not set")
    def test_generate_audio_elevenlabs(self, client):
        """Test generating audio with ElevenLabs"""
        response = client.post("/api/tts", json={
            "text": "Hello from ElevenLabs.",
            "provider": "elevenlabs",
            "model": "eleven_flash_v2_5",
            "voice": "21m00Tcm4TlvDq8ikWAM",  # Rachel
            "api_key": os.getenv("ELEVENLABS_API_KEY")
        })

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert len(response.content) > 1000


class TestAudioQuality:
    """Tests for audio output quality and format"""

    @pytest.mark.skipif(not is_mlx_audio_running(), reason="MLX-Audio server not running")
    def test_audio_is_valid_mp3(self, client):
        """Verify output is valid MP3 format"""
        response = client.post("/api/tts", json={
            "text": "Test audio format.",
            "provider": "mlx-audio",
            "model": "mlx-community/Spark-TTS-0.5B-bf16",
        })

        assert response.status_code == 200, f"Error: {response.text}"
        # MP3 files start with ID3 tag or frame sync
        assert response.content[:3] == b"ID3" or response.content[:2] == b"\xff\xfb"

    @pytest.mark.skipif(not is_mlx_audio_running(), reason="MLX-Audio server not running")
    def test_audio_has_reasonable_size(self, client):
        """Verify audio size is reasonable for text length"""
        short_response = client.post("/api/tts", json={
            "text": "Hi.",
            "provider": "mlx-audio",
            "model": "mlx-community/Spark-TTS-0.5B-bf16",
        })

        long_response = client.post("/api/tts", json={
            "text": "This is a much longer sentence with many more words.",
            "provider": "mlx-audio",
            "model": "mlx-community/Spark-TTS-0.5B-bf16",
        })

        assert short_response.status_code == 200, f"Error: {short_response.text}"
        assert long_response.status_code == 200, f"Error: {long_response.text}"
        # Longer text should produce larger audio
        assert len(long_response.content) > len(short_response.content)


# Path to dario.mp3 sample file
SAMPLE_AUDIO_PATH = os.path.join(os.path.dirname(__file__), "..", "dario.mp3")


def has_sample_audio():
    """Check if dario.mp3 sample file exists"""
    return os.path.exists(SAMPLE_AUDIO_PATH)


def has_whisper():
    """Check if mlx-whisper is available"""
    try:
        import mlx_whisper
        return True
    except ImportError:
        return False


@pytest.fixture
def clean_uploads_integration():
    """Clean up uploads directory before and after integration tests"""
    from app import UPLOADS_DIR, VOICES_METADATA_FILE

    # Clean before
    if UPLOADS_DIR.exists():
        for f in UPLOADS_DIR.iterdir():
            if f.is_file():
                f.unlink()

    yield

    # Clean after
    if UPLOADS_DIR.exists():
        for f in UPLOADS_DIR.iterdir():
            if f.is_file():
                f.unlink()


class TestVoiceUploadIntegration:
    """Integration tests for voice upload with real audio files"""

    @pytest.mark.skipif(not has_sample_audio(), reason="dario.mp3 sample file not found")
    def test_upload_real_audio_file(self, client, clean_uploads_integration):
        """Test uploading a real MP3 audio file"""
        with open(SAMPLE_AUDIO_PATH, "rb") as f:
            audio_data = f.read()

        response = client.post(
            "/api/upload-voice",
            files={"file": ("dario.mp3", audio_data, "audio/mpeg")},
            data={"name": "Dario Voice", "transcript": "This is Dario speaking."}
        )

        assert response.status_code == 200
        data = response.json()
        assert "voice_id" in data
        assert data["name"] == "Dario Voice"
        assert data["transcript"] == "This is Dario speaking."
        assert data["filename"].endswith(".mp3")

    @pytest.mark.skipif(
        not has_sample_audio() or not has_whisper(),
        reason="Requires dario.mp3 and mlx-whisper"
    )
    def test_upload_with_auto_transcription(self, client, clean_uploads_integration):
        """Test uploading audio with automatic transcription"""
        with open(SAMPLE_AUDIO_PATH, "rb") as f:
            audio_data = f.read()

        response = client.post(
            "/api/upload-voice",
            files={"file": ("dario.mp3", audio_data, "audio/mpeg")},
            data={"name": "Dario Auto", "transcript": ""}  # Empty transcript triggers auto-transcription
        )

        assert response.status_code == 200
        data = response.json()
        assert "voice_id" in data
        assert data["name"] == "Dario Auto"
        # Should have auto-transcribed text
        assert len(data["transcript"]) > 0
        print(f"Auto-transcribed: {data['transcript']}")

    @pytest.mark.skipif(not has_sample_audio(), reason="dario.mp3 sample file not found")
    def test_uploaded_voice_appears_in_list(self, client, clean_uploads_integration):
        """Test that uploaded voice appears in voice list"""
        with open(SAMPLE_AUDIO_PATH, "rb") as f:
            audio_data = f.read()

        # Upload voice
        upload_response = client.post(
            "/api/upload-voice",
            files={"file": ("dario.mp3", audio_data, "audio/mpeg")},
            data={"name": "Dario Test", "transcript": "Test transcript."}
        )
        voice_id = upload_response.json()["voice_id"]

        # List voices
        list_response = client.get("/api/voices")
        assert list_response.status_code == 200
        voices = list_response.json()["voices"]

        assert len(voices) == 1
        assert voices[0]["id"] == voice_id
        assert voices[0]["name"] == "Dario Test"

    @pytest.mark.skipif(not has_sample_audio(), reason="dario.mp3 sample file not found")
    def test_uploaded_voice_appears_in_providers(self, client, clean_uploads_integration):
        """Test that uploaded voice appears in mlx-voice-clone provider"""
        with open(SAMPLE_AUDIO_PATH, "rb") as f:
            audio_data = f.read()

        # Upload voice
        upload_response = client.post(
            "/api/upload-voice",
            files={"file": ("dario.mp3", audio_data, "audio/mpeg")},
            data={"name": "Dario Provider Test", "transcript": "Test transcript."}
        )
        voice_id = upload_response.json()["voice_id"]

        # Get providers
        providers_response = client.get("/api/providers")
        providers = providers_response.json()["providers"]

        # Check mlx-voice-clone has the uploaded voice
        voice_clone_voices = providers["mlx-voice-clone"]["voices"]
        assert voice_id in voice_clone_voices
        assert voice_clone_voices[voice_id] == "Dario Provider Test"


class TestVoiceCloningIntegration:
    """Integration tests for full voice cloning TTS workflow"""

    @pytest.mark.skipif(
        not has_sample_audio() or not is_mlx_audio_running(),
        reason="Requires dario.mp3 and MLX-Audio server"
    )
    def test_voice_clone_full_workflow(self, client, clean_uploads_integration):
        """Test complete voice cloning workflow: upload -> generate"""
        with open(SAMPLE_AUDIO_PATH, "rb") as f:
            audio_data = f.read()

        # Step 1: Upload voice
        upload_response = client.post(
            "/api/upload-voice",
            files={"file": ("dario.mp3", audio_data, "audio/mpeg")},
            data={"name": "Dario Clone", "transcript": "This is Dario speaking in a sample audio clip."}
        )
        assert upload_response.status_code == 200
        voice_id = upload_response.json()["voice_id"]
        print(f"Uploaded voice: {voice_id}")

        # Step 2: Generate TTS with cloned voice
        tts_response = client.post("/api/tts", json={
            "text": "Hello, this is a cloned voice speaking.",
            "provider": "mlx-voice-clone",
            "model": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
            "voice_id": voice_id
        })

        assert tts_response.status_code == 200, f"TTS Error: {tts_response.text}"
        assert tts_response.headers["content-type"] == "audio/mpeg"
        assert len(tts_response.content) > 1000  # Should have actual audio data
        print(f"Generated {len(tts_response.content)} bytes of cloned audio")

    @pytest.mark.skipif(
        not has_sample_audio() or not is_mlx_audio_running(),
        reason="Requires dario.mp3 and MLX-Audio server"
    )
    def test_voice_clone_larger_model(self, client, clean_uploads_integration):
        """Test voice cloning with larger 1.7B model"""
        with open(SAMPLE_AUDIO_PATH, "rb") as f:
            audio_data = f.read()

        # Upload voice
        upload_response = client.post(
            "/api/upload-voice",
            files={"file": ("dario.mp3", audio_data, "audio/mpeg")},
            data={"name": "Dario Large", "transcript": "This is Dario speaking in a sample audio clip."}
        )
        voice_id = upload_response.json()["voice_id"]

        # Generate with larger model
        tts_response = client.post("/api/tts", json={
            "text": "Testing the larger voice cloning model.",
            "provider": "mlx-voice-clone",
            "model": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            "voice_id": voice_id
        })

        assert tts_response.status_code == 200, f"TTS Error: {tts_response.text}"
        assert len(tts_response.content) > 1000

