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
