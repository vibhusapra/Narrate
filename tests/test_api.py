"""Tests for Narrate TTS API"""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app import app, PROVIDERS


@pytest.fixture
def client():
    return TestClient(app)


class TestProviders:
    """Test /api/providers endpoint"""

    def test_get_providers_returns_all_providers(self, client):
        response = client.get("/api/providers")
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        assert "mlx-audio" in data["providers"]
        assert "elevenlabs" in data["providers"]
        assert "openai" in data["providers"]

    def test_each_provider_has_required_fields(self, client):
        response = client.get("/api/providers")
        providers = response.json()["providers"]

        for provider_id, provider in providers.items():
            assert "name" in provider, f"{provider_id} missing name"
            assert "description" in provider, f"{provider_id} missing description"
            assert "requires_api_key" in provider, f"{provider_id} missing requires_api_key"
            assert "models" in provider, f"{provider_id} missing models"

    def test_cloud_providers_require_api_key(self, client):
        response = client.get("/api/providers")
        providers = response.json()["providers"]

        assert providers["elevenlabs"]["requires_api_key"] is True
        assert providers["openai"]["requires_api_key"] is True
        assert providers["mlx-audio"]["requires_api_key"] is False

    def test_cloud_providers_have_voices(self, client):
        response = client.get("/api/providers")
        providers = response.json()["providers"]

        # Cloud providers should have voices
        assert len(providers["elevenlabs"]["voices"]) > 0
        assert len(providers["openai"]["voices"]) > 0


class TestHealth:
    """Test /api/health endpoint"""

    def test_health_returns_status(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "providers" in data

    def test_health_checks_all_providers(self, client):
        response = client.get("/api/health")
        providers = response.json()["providers"]

        assert "mlx_audio" in providers
        assert "elevenlabs" in providers
        assert "openai" in providers


class TestTTS:
    """Test /api/tts endpoint"""

    def test_empty_text_returns_400(self, client):
        response = client.post("/api/tts", json={
            "text": "",
            "provider": "mlx-audio"
        })
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_whitespace_only_text_returns_400(self, client):
        response = client.post("/api/tts", json={
            "text": "   \n\t  ",
            "provider": "mlx-audio"
        })
        assert response.status_code == 400

    def test_unknown_provider_returns_400(self, client):
        response = client.post("/api/tts", json={
            "text": "Hello world",
            "provider": "unknown-provider"
        })
        assert response.status_code == 400
        assert "unknown provider" in response.json()["detail"].lower()

    def test_elevenlabs_without_api_key_returns_400(self, client):
        response = client.post("/api/tts", json={
            "text": "Hello world",
            "provider": "elevenlabs",
            "model": "eleven_flash_v2_5"
        })
        assert response.status_code == 400
        assert "api key" in response.json()["detail"].lower()

    def test_openai_without_api_key_returns_400(self, client):
        response = client.post("/api/tts", json={
            "text": "Hello world",
            "provider": "openai",
            "model": "tts-1"
        })
        assert response.status_code == 400
        assert "api key" in response.json()["detail"].lower()


class TestTTSWithMocks:
    """Test TTS generation with mocked external services"""

    @patch("app.httpx.AsyncClient")
    def test_mlx_audio_success(self, mock_client_class, client):
        # Setup mock
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.content = b"fake audio data"

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        response = client.post("/api/tts", json={
            "text": "Hello world",
            "provider": "mlx-audio",
            "model": "mlx-community/Kokoro-82M-bf16",
            "voice": "af_heart"
        })

        assert response.status_code == 200
        assert response.content == b"fake audio data"
        assert response.headers["content-type"] == "audio/mpeg"

    @patch("app.httpx.AsyncClient")
    def test_elevenlabs_success(self, mock_client_class, client):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.content = b"fake mp3 data"

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        response = client.post("/api/tts", json={
            "text": "Hello world",
            "provider": "elevenlabs",
            "model": "eleven_flash_v2_5",
            "voice": "21m00Tcm4TlvDq8ikWAM",
            "api_key": "test-api-key"
        })

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"

    @patch("app.httpx.AsyncClient")
    def test_openai_success(self, mock_client_class, client):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.content = b"fake wav data"

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        response = client.post("/api/tts", json={
            "text": "Hello world",
            "provider": "openai",
            "model": "tts-1",
            "voice": "alloy",
            "api_key": "test-api-key"
        })

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"

    @patch("app.httpx.AsyncClient")
    def test_mlx_audio_server_error(self, mock_client_class, client):
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        response = client.post("/api/tts", json={
            "text": "Hello world",
            "provider": "mlx-audio"
        })

        assert response.status_code == 500
        assert "MLX-Audio error" in response.json()["detail"]


class TestStaticFiles:
    """Test static file serving"""

    def test_index_returns_html(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Narrate" in response.text


class TestRequestValidation:
    """Test request body validation"""

    def test_missing_text_field(self, client):
        response = client.post("/api/tts", json={
            "provider": "mlx-audio"
        })
        assert response.status_code == 422  # Validation error

    @patch("app.httpx.AsyncClient")
    def test_default_provider_is_mlx_audio(self, mock_client_class, client):
        # Setup mock to capture the request
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.content = b"audio"

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        response = client.post("/api/tts", json={
            "text": "Hello"
        })

        assert response.status_code == 200
        # Verify it called MLX-Audio endpoint
        call_args = mock_client.post.call_args
        assert "8000" in call_args[0][0]  # MLX-Audio port

    @patch("app.httpx.AsyncClient")
    def test_accepts_optional_fields(self, mock_client_class, client):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.content = b"audio"

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        response = client.post("/api/tts", json={
            "text": "Hello",
            "provider": "openai",
            "model": "tts-1",
            "voice": "alloy",
            "api_key": "test-key"
        })

        assert response.status_code == 200
        # Verify OpenAI endpoint was called with correct params
        call_args = mock_client.post.call_args
        assert "openai.com" in call_args[0][0]
