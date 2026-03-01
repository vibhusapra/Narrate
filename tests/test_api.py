"""Tests for Narrate TTS API"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from app import (
    app, PROVIDERS, UPLOADS_DIR, VOICES_METADATA_FILE,
    split_into_sentences, chunk_sentences, chunk_text_by_tokens, concatenate_audio_chunks
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def clean_uploads():
    """Clean up uploads directory before and after test"""
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
            assert "supports_voice_id" in provider, f"{provider_id} missing supports_voice_id"
            assert "requires_voice_id" in provider, f"{provider_id} missing requires_voice_id"
            assert "requires_reference_transcript" in provider, f"{provider_id} missing requires_reference_transcript"

    def test_cloud_providers_require_api_key(self, client):
        response = client.get("/api/providers")
        providers = response.json()["providers"]

        assert providers["elevenlabs"]["requires_api_key"] is True
        assert providers["openai"]["requires_api_key"] is True
        assert providers["mlx-audio"]["requires_api_key"] is False
        assert providers["mlx-voice-clone"]["requires_api_key"] is False

    def test_voice_clone_provider_exists(self, client):
        response = client.get("/api/providers")
        providers = response.json()["providers"]

        assert "mlx-voice-clone" in providers
        assert providers["mlx-voice-clone"]["requires_voice_id"] is True
        assert providers["mlx-voice-clone"]["supports_voice_id"] is True
        assert providers["mlx-voice-clone"]["requires_reference_transcript"] is True
        assert "models" in providers["mlx-voice-clone"]
        assert len(providers["mlx-voice-clone"]["models"]) > 0

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


class TestCostEstimate:
    """Test rough cost estimation endpoint"""

    def test_cost_estimate_returns_tokens_and_cost(self, client):
        response = client.post("/api/cost-estimate", json={
            "text": "This is a quick test for rough cost estimates.",
            "provider": "openai",
            "model": "tts-1",
        })

        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "openai"
        assert data["model"] == "tts-1"
        assert data["input_tokens"] > 0
        assert isinstance(data["estimated_cost_usd"], float)
        assert isinstance(data["uses_tiktoken"], bool)

    def test_cost_estimate_local_provider_has_note(self, client):
        response = client.post("/api/cost-estimate", json={
            "text": "This is a quick test.",
            "provider": "mlx-audio",
            "model": "mlx-community/Spark-TTS-0.5B-bf16",
        })

        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "mlx-audio"
        assert data["estimated_cost_usd"] in [None, 0]
        assert "Local providers" in data.get("note", "")

    def test_cost_estimate_unknown_provider(self, client):
        response = client.post("/api/cost-estimate", json={
            "text": "Hello",
            "provider": "unknown-provider",
            "model": "foo",
        })
        assert response.status_code == 400
        assert "unknown provider" in response.json()["detail"].lower()


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
    def test_non_clone_provider_ignores_voice_id(self, mock_client_class, client):
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.content = b"fake wav"

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        response = client.post(
            "/api/tts",
            json={
                "text": "Hello world",
                "provider": "openai",
                "model": "tts-1",
                "voice": "alloy",
                "voice_id": "some-unused-id",
                "api_key": "test-api-key"
            }
        )

        assert response.status_code == 200
        assert response.content == b"fake wav"

    @patch("app.generate_openai_chunked", new_callable=AsyncMock)
    @patch("app.estimate_tokens_for_text")
    def test_openai_long_input_uses_chunked_path(self, mock_token_estimate, mock_openai_chunked, client):
        mock_token_estimate.return_value = (10000, True)
        mock_openai_chunked.return_value = b"chunked wav data"

        response = client.post("/api/tts", json={
            "text": "Long text",
            "provider": "openai",
            "model": "tts-1",
            "voice": "alloy",
            "api_key": "test-api-key"
        })

        assert response.status_code == 200
        assert response.content == b"chunked wav data"
        assert response.headers["content-type"] == "audio/wav"
        mock_openai_chunked.assert_awaited_once()

    @patch("app.generate_elevenlabs_chunked", new_callable=AsyncMock)
    @patch("app.estimate_tokens_for_text")
    def test_elevenlabs_long_input_uses_chunked_path(self, mock_token_estimate, mock_elevenlabs_chunked, client):
        mock_token_estimate.return_value = (10000, True)
        mock_elevenlabs_chunked.return_value = b"chunked mp3 data"

        response = client.post("/api/tts", json={
            "text": "Long text",
            "provider": "elevenlabs",
            "model": "eleven_flash_v2_5",
            "voice": "21m00Tcm4TlvDq8ikWAM",
            "api_key": "test-api-key"
        })

        assert response.status_code == 200
        assert response.content == b"chunked mp3 data"
        assert response.headers["content-type"] == "audio/mpeg"
        mock_elevenlabs_chunked.assert_awaited_once()

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


class TestVoiceManagement:
    """Test voice upload and management endpoints"""

    def test_list_voices_empty(self, client, clean_uploads):
        response = client.get("/api/voices")
        assert response.status_code == 200
        data = response.json()
        assert "voices" in data
        assert data["voices"] == []

    def test_upload_voice(self, client, clean_uploads):
        # Create a fake WAV file
        fake_audio = b"RIFF" + b"\x00" * 100  # Minimal WAV-like content

        response = client.post(
            "/api/upload-voice",
            files={"file": ("test.wav", fake_audio, "audio/wav")},
            data={"name": "Test Voice", "transcript": "Hello, this is a test."}
        )

        assert response.status_code == 200
        data = response.json()
        assert "voice_id" in data
        assert data["name"] == "Test Voice"
        assert data["transcript"] == "Hello, this is a test."

    def test_upload_voice_and_list(self, client, clean_uploads):
        # Upload a voice
        fake_audio = b"RIFF" + b"\x00" * 100

        upload_response = client.post(
            "/api/upload-voice",
            files={"file": ("test.wav", fake_audio, "audio/wav")},
            data={"name": "Test Voice", "transcript": "Hello, this is a test."}
        )
        voice_id = upload_response.json()["voice_id"]

        # List voices
        list_response = client.get("/api/voices")
        assert list_response.status_code == 200
        voices = list_response.json()["voices"]

        assert len(voices) == 1
        assert voices[0]["id"] == voice_id
        assert voices[0]["name"] == "Test Voice"
        assert voices[0]["transcript"] == "Hello, this is a test."

        assert "transcript_status" in voices[0]
        assert "transcript_source" in voices[0]
        assert "duration_seconds" in voices[0]
        assert "sample_rate" in voices[0]
        assert "channels" in voices[0]

    def test_delete_voice(self, client, clean_uploads):
        # Upload a voice
        fake_audio = b"RIFF" + b"\x00" * 100

        upload_response = client.post(
            "/api/upload-voice",
            files={"file": ("test.wav", fake_audio, "audio/wav")},
            data={"name": "Test Voice", "transcript": "Hello, this is a test."}
        )
        voice_id = upload_response.json()["voice_id"]

        # Delete the voice
        delete_response = client.delete(f"/api/voices/{voice_id}")
        assert delete_response.status_code == 200
        assert delete_response.json()["status"] == "deleted"

        # Verify it's gone
        list_response = client.get("/api/voices")
        voices = list_response.json()["voices"]
        assert len(voices) == 0

    def test_delete_nonexistent_voice(self, client, clean_uploads):
        response = client.delete("/api/voices/nonexistent123")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_upload_non_audio_file_returns_400(self, client, clean_uploads):
        response = client.post(
            "/api/upload-voice",
            files={"file": ("test.txt", b"not audio", "text/plain")},
            data={"name": "Test Voice", "transcript": "Hello"}
        )
        assert response.status_code == 400
        assert "audio" in response.json()["detail"].lower()

    def test_upload_voice_auto_transcribes(self, client, clean_uploads):
        # Mock the transcribe_audio function directly
        fake_audio = b"RIFF" + b"\x00" * 100

        with patch("app.transcribe_audio", return_value="Auto transcribed text"):
            response = client.post(
                "/api/upload-voice",
                files={"file": ("test.wav", fake_audio, "audio/wav")},
                data={"name": "Test Voice", "transcript": ""}  # Empty transcript
            )

        assert response.status_code == 200
        data = response.json()
        assert data["transcript"] == "Auto transcribed text"
        assert data["transcript_status"] == "auto_transcribed"
        assert data["transcript_source"] == "mlx_whisper"

    def test_upload_voice_without_transcript_and_no_whisper(self, client, clean_uploads):
        # When whisper is not available and no transcript provided
        fake_audio = b"RIFF" + b"\x00" * 100

        with patch("app.HAS_WHISPER", False):
            response = client.post(
                "/api/upload-voice",
                files={"file": ("test.wav", fake_audio, "audio/wav")},
                data={"name": "Test Voice", "transcript": ""}
            )

        assert response.status_code == 503
        assert "mlx-whisper" in response.json()["detail"].lower() or "install" in response.json()["detail"].lower()

    def test_upload_voice_mode_off_without_transcript(self, client, clean_uploads):
        fake_audio = b"RIFF" + b"\x00" * 100

        response = client.post(
            "/api/upload-voice",
            files={"file": ("test.wav", fake_audio, "audio/wav")},
            data={
                "name": "Test Voice",
                "transcript": "",
                "transcript_mode": "off",
            },
        )

        assert response.status_code == 400
        assert "transcript" in response.json()["detail"].lower()

    def test_upload_voice_mode_provided_skips_stt(self, client, clean_uploads):
        fake_audio = b"RIFF" + b"\x00" * 100

        with patch("app.transcribe_audio") as mocked_transcribe:
            response = client.post(
                "/api/upload-voice",
                files={"file": ("test.wav", fake_audio, "audio/wav")},
                data={"name": "Test Voice", "transcript": "Manual", "transcript_mode": "provided"},
            )

        assert response.status_code == 200
        assert response.json()["transcript"] == "Manual"
        mocked_transcribe.assert_not_called()

    @patch("app.transcribe_audio", return_value="Retranscribed text")
    def test_retranscribe_voice_endpoint_updates_transcript(self, client, clean_uploads):
        upload_response = client.post(
            "/api/upload-voice",
            files={"file": ("test.wav", b"RIFF" + b"\x00" * 100, "audio/wav")},
            data={"name": "Test Voice", "transcript": "seed"},
        )
        voice_id = upload_response.json()["voice_id"]

        response = client.post(f"/api/voices/{voice_id}/retranscribe", json={})
        assert response.status_code == 200
        payload = response.json()
        assert payload["voice_id"] == voice_id
        assert payload["transcript"] == "Retranscribed text"
        assert payload["transcript_status"] == "auto_transcribed"
        assert payload["transcript_source"] == "mlx_whisper"

    def test_update_voice_transcript_endpoint(self, client, clean_uploads):
        upload_response = client.post(
            "/api/upload-voice",
            files={"file": ("test.wav", b"RIFF" + b"\x00" * 100, "audio/wav")},
            data={"name": "Test Voice", "transcript": "seed"},
        )
        voice_id = upload_response.json()["voice_id"]

        response = client.patch(
            f"/api/voices/{voice_id}",
            json={"transcript": "edited transcript"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["transcript"] == "edited transcript"
        assert payload["transcript_status"] == "manual"


class TestVoiceCloneTTS:
    """Test TTS with voice cloning"""

    def test_voice_clone_without_voice_id_returns_400(self, client):
        response = client.post("/api/tts", json={
            "text": "Hello world",
            "provider": "mlx-voice-clone",
            "model": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
        })
        assert response.status_code == 400
        assert "voice" in response.json()["detail"].lower()

    def test_voice_clone_with_nonexistent_voice_returns_404(self, client, clean_uploads):
        response = client.post("/api/tts", json={
            "text": "Hello world",
            "provider": "mlx-voice-clone",
            "model": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
            "voice_id": "nonexistent123"
        })
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @patch("app.httpx.AsyncClient")
    def test_voice_clone_success(self, mock_client_class, client, clean_uploads):
        # First upload a voice
        fake_audio = b"RIFF" + b"\x00" * 100

        upload_response = client.post(
            "/api/upload-voice",
            files={"file": ("test.wav", fake_audio, "audio/wav")},
            data={"name": "Test Voice", "transcript": "Hello, this is a test."}
        )
        voice_id = upload_response.json()["voice_id"]

        # Setup mock for TTS call
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.content = b"fake cloned audio"

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        # Generate speech with cloned voice
        response = client.post("/api/tts", json={
            "text": "Hello world",
            "provider": "mlx-voice-clone",
            "model": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
            "voice_id": voice_id
        })

        assert response.status_code == 200
        assert response.content == b"fake cloned audio"
        assert response.headers["content-type"] == "audio/mpeg"


class TestChunkingFunctions:
    """Test text chunking and audio concatenation functions"""

    def test_split_into_sentences(self):
        text = "This is sentence one. This is sentence two! And this is sentence three?"
        sentences = split_into_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "This is sentence one."
        assert sentences[1] == "This is sentence two!"
        assert sentences[2] == "And this is sentence three?"

    def test_split_into_sentences_handles_newlines(self):
        text = "First sentence.\n\nSecond sentence. Third sentence!"
        sentences = split_into_sentences(text)
        assert len(sentences) == 3

    def test_chunk_sentences_under_limit(self):
        sentences = ["Short sentence.", "Another short one.", "One more."]
        chunks = chunk_sentences(sentences, max_words=50)
        # All sentences fit in one chunk
        assert len(chunks) == 1
        assert "Short sentence. Another short one. One more." in chunks[0]

    def test_chunk_sentences_over_limit(self):
        # Each sentence is ~10 words
        sentences = [
            "This is a sentence with exactly ten words in it here.",
            "Another sentence that also has ten words in it here.",
            "Yet another sentence with ten words in it here too."
        ]
        chunks = chunk_sentences(sentences, max_words=15)
        # Should split into multiple chunks
        assert len(chunks) == 3

    def test_chunk_sentences_respects_boundaries(self):
        sentences = [
            "Short.",
            "A " + " ".join(["word"] * 100),  # 100 words
            "Short again."
        ]
        chunks = chunk_sentences(sentences, max_words=50)
        # First short sentence might be alone, long sentence alone, last short sentence alone
        assert len(chunks) >= 2

    def test_chunk_text_by_tokens_keeps_existing_chunk_before_long_sentence(self):
        text = "Short start. " + ("word " * 5000) + ". End."
        chunks = chunk_text_by_tokens(
            text=text,
            max_tokens=50,
            provider="openai",
            model="tts-1",
        )
        assert len(chunks) > 1
        assert chunks[0].startswith("Short start")

    def test_concatenate_audio_chunks_single(self):
        # Single chunk should return as-is
        chunks = [b"audio data"]
        result = concatenate_audio_chunks(chunks)
        assert result == b"audio data"

    @patch("app.subprocess.run")
    def test_concatenate_audio_chunks_multiple(self, mock_run):
        # Mock subprocess.run to simulate successful ffmpeg execution
        mock_run.return_value = MagicMock(returncode=0)

        # Create fake MP3 data
        fake_mp3_header = b'\xff\xfb' + b'\x00' * 100
        chunks = [fake_mp3_header, fake_mp3_header]

        # Mock the file reading to return concatenated data
        with patch("builtins.open", create=True) as mock_open:
            # Configure mock to handle both writing chunks and reading result
            mock_file_write = MagicMock()
            mock_file_read = MagicMock()
            mock_file_read.read.return_value = b"concatenated audio data"

            def open_side_effect(path, mode, *args, **kwargs):
                if 'r' in mode and not 'b' in mode:
                    # Reading concat list as text
                    return MagicMock(__enter__=lambda s: MagicMock(write=lambda x: None), __exit__=lambda *a: None)
                elif 'w' in mode:
                    # Writing chunks
                    return MagicMock(__enter__=lambda s: mock_file_write, __exit__=lambda *a: None)
                elif 'r' in mode and 'b' in mode:
                    # Reading final combined file
                    return MagicMock(__enter__=lambda s: mock_file_read, __exit__=lambda *a: None)

            mock_open.side_effect = open_side_effect

            result = concatenate_audio_chunks(chunks)

        # Should have called ffmpeg
        assert mock_run.called

    def test_concatenate_empty_chunks_raises_error(self):
        with pytest.raises(ValueError, match="No audio chunks"):
            concatenate_audio_chunks([])


class TestChunkedTTSEndpoints:
    """Test chunked TTS generation endpoints"""

    def test_chunked_tts_requires_voice_clone_provider(self, client, clean_uploads):
        response = client.post("/api/tts/chunked", json={
            "text": "Hello world",
            "provider": "mlx-audio",
            "model": "some-model"
        })
        assert response.status_code == 400
        assert "chunked generation only supports mlx-voice-clone" in response.json()["detail"].lower()

    def test_chunked_tts_requires_voice_id(self, client, clean_uploads):
        response = client.post("/api/tts/chunked", json={
            "text": "Hello world",
            "provider": "mlx-voice-clone",
            "model": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
        })
        assert response.status_code == 400
        assert "voice" in response.json()["detail"].lower()

    def test_chunked_tts_returns_session_id(self, client, clean_uploads):
        # Upload a voice first
        fake_audio = b"RIFF" + b"\x00" * 100
        upload_response = client.post(
            "/api/upload-voice",
            files={"file": ("test.wav", fake_audio, "audio/wav")},
            data={"name": "Test Voice", "transcript": "Hello, this is a test."}
        )
        voice_id = upload_response.json()["voice_id"]

        # Start chunked generation
        response = client.post("/api/tts/chunked", json={
            "text": "This is a test. " * 100,  # Long text
            "provider": "mlx-voice-clone",
            "model": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
            "voice_id": voice_id
        })

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["status"] == "processing"
        assert "word_count" in data

    def test_status_endpoint_returns_progress(self, client, clean_uploads):
        # Upload a voice
        fake_audio = b"RIFF" + b"\x00" * 100
        upload_response = client.post(
            "/api/upload-voice",
            files={"file": ("test.wav", fake_audio, "audio/wav")},
            data={"name": "Test Voice", "transcript": "Hello, this is a test."}
        )
        voice_id = upload_response.json()["voice_id"]

        # Start chunked generation
        start_response = client.post("/api/tts/chunked", json={
            "text": "Test text. " * 50,
            "provider": "mlx-voice-clone",
            "model": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
            "voice_id": voice_id
        })
        session_id = start_response.json()["session_id"]

        # Check status
        status_response = client.get(f"/api/tts/status/{session_id}")
        assert status_response.status_code == 200
        status = status_response.json()
        assert "status" in status
        assert "progress" in status

    def test_status_nonexistent_session_returns_404(self, client):
        response = client.get("/api/tts/status/nonexistent-session")
        assert response.status_code == 404

    def test_download_endpoint_requires_completion(self, client, clean_uploads):
        # Upload a voice and start generation
        fake_audio = b"RIFF" + b"\x00" * 100
        upload_response = client.post(
            "/api/upload-voice",
            files={"file": ("test.wav", fake_audio, "audio/wav")},
            data={"name": "Test Voice", "transcript": "Hello, this is a test."}
        )
        voice_id = upload_response.json()["voice_id"]

        start_response = client.post("/api/tts/chunked", json={
            "text": "Test. " * 50,
            "provider": "mlx-voice-clone",
            "model": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
            "voice_id": voice_id
        })
        session_id = start_response.json()["session_id"]

        # Try to download before completion (should fail)
        download_response = client.get(f"/api/tts/download/{session_id}")
        assert download_response.status_code == 400
        assert "not complete" in download_response.json()["detail"].lower()

    def test_download_nonexistent_session_returns_404(self, client):
        response = client.get("/api/tts/download/nonexistent-session")
        assert response.status_code == 404
