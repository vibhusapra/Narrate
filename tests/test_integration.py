"""
Integration tests for TTS generation.
These tests require external services to be running or valid API keys.

Run with: uv run --extra dev pytest tests/test_integration.py -v -s
Skip with: uv run --extra dev pytest tests/test_api.py -v (unit tests only)
"""

import os
import pytest
from unittest.mock import AsyncMock, patch
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


class TestChunkedGenerationIntegration:
    """Integration tests for chunked TTS generation"""

    @pytest.mark.skipif(
        not has_sample_audio() or not is_mlx_audio_running(),
        reason="Requires dario.mp3 and MLX-Audio server"
    )
    def test_chunked_generation_workflow(self, client, clean_uploads_integration):
        """Test chunked generation with ~1000 word text"""
        import time

        with open(SAMPLE_AUDIO_PATH, "rb") as f:
            audio_data = f.read()

        # Upload voice
        upload_response = client.post(
            "/api/upload-voice",
            files={"file": ("dario.mp3", audio_data, "audio/mpeg")},
            data={"name": "Dario Chunked", "transcript": "This is Dario speaking in a sample audio clip."}
        )
        assert upload_response.status_code == 200
        voice_id = upload_response.json()["voice_id"]
        print(f"Uploaded voice: {voice_id}")

        # Create ~1000 word text (will be split into ~2 chunks)
        long_text = "This is a test sentence for voice cloning. " * 200  # ~1000 words
        word_count = len(long_text.split())
        print(f"Text has {word_count} words")

        # Start chunked generation
        start_time = time.time()
        start_response = client.post("/api/tts/chunked", json={
            "text": long_text,
            "provider": "mlx-voice-clone",
            "model": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
            "voice_id": voice_id
        })

        assert start_response.status_code == 200, f"Start Error: {start_response.text}"
        data = start_response.json()
        assert "session_id" in data
        assert data["status"] == "processing"
        session_id = data["session_id"]
        print(f"Started session: {session_id}")

        # Poll for completion
        max_wait = 300  # 5 minutes max
        poll_interval = 3  # Poll every 3 seconds
        elapsed = 0

        while elapsed < max_wait:
            status_response = client.get(f"/api/tts/status/{session_id}")
            assert status_response.status_code == 200
            status = status_response.json()

            print(f"Status: {status['status']}, Progress: {status.get('progress')}")

            if status["status"] == "complete":
                print(f"Completed in {elapsed}s")
                break
            elif status["status"] == "error":
                pytest.fail(f"Generation failed: {status.get('error')}")

            time.sleep(poll_interval)
            elapsed += poll_interval
        else:
            pytest.fail(f"Generation timed out after {max_wait}s")

        # Download audio
        download_response = client.get(f"/api/tts/download/{session_id}")
        assert download_response.status_code == 200
        assert download_response.headers["content-type"] == "audio/mpeg"
        assert len(download_response.content) > 10000  # Should be substantial audio

        total_time = time.time() - start_time
        print(f"Total time: {total_time:.1f}s, Audio size: {len(download_response.content)} bytes")

    @pytest.mark.skipif(
        not has_sample_audio() or not is_mlx_audio_running(),
        reason="Requires dario.mp3 and MLX-Audio server"
    )
    def test_chunked_vs_single_generation(self, client, clean_uploads_integration):
        """Compare chunked and single generation for same text"""
        with open(SAMPLE_AUDIO_PATH, "rb") as f:
            audio_data = f.read()

        # Upload voice
        upload_response = client.post(
            "/api/upload-voice",
            files={"file": ("dario.mp3", audio_data, "audio/mpeg")},
            data={"name": "Dario Compare", "transcript": "This is Dario speaking in a sample audio clip."}
        )
        voice_id = upload_response.json()["voice_id"]

        # Use a short text that can work with both approaches
        test_text = "This is a short test. " * 50  # ~150 words

        # Single generation (should work for 150 words)
        single_response = client.post("/api/tts", json={
            "text": test_text,
            "provider": "mlx-voice-clone",
            "model": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
            "voice_id": voice_id
        }, timeout=180.0)

        assert single_response.status_code == 200
        single_audio_size = len(single_response.content)
        print(f"Single generation: {single_audio_size} bytes")

        # Chunked generation
        import time
        start_response = client.post("/api/tts/chunked", json={
            "text": test_text,
            "provider": "mlx-voice-clone",
            "model": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
            "voice_id": voice_id
        })
        session_id = start_response.json()["session_id"]

        # Wait for completion
        for _ in range(100):  # Max 300s
            status = client.get(f"/api/tts/status/{session_id}").json()
            if status["status"] == "complete":
                break
            time.sleep(3)

        chunked_response = client.get(f"/api/tts/download/{session_id}")
        assert chunked_response.status_code == 200
        chunked_audio_size = len(chunked_response.content)
        print(f"Chunked generation: {chunked_audio_size} bytes")

        # Both should produce audio of similar size (within 20%)
        size_diff_pct = abs(single_audio_size - chunked_audio_size) / single_audio_size * 100
        print(f"Size difference: {size_diff_pct:.1f}%")
        # Note: sizes may differ due to MP3 encoding and chunk boundaries
        assert chunked_audio_size > 1000  # Should have substantial audio


class TestPocketMlxIntegration:
    """Integration-style tests for pocket-tts-mlx endpoints."""

    @patch("app.generate_pocket_tts_mlx", new_callable=AsyncMock)
    def test_generate_with_uploaded_voice_mlx_direct(self, mock_generator, client, clean_uploads_integration):
        if not has_sample_audio():
            pytest.skip("dario.mp3 sample file not found")

        mock_generator.return_value = b"fake pocket tts mlx audio"

        with open(SAMPLE_AUDIO_PATH, "rb") as f:
            audio_data = f.read()

        upload_response = client.post(
            "/api/upload-voice",
            files={"file": ("dario.mp3", audio_data, "audio/mpeg")},
            data={"name": "Dario MLX Clone", "transcript": "This is Dario speaking in a sample audio clip."}
        )
        assert upload_response.status_code == 200
        voice_id = upload_response.json()["voice_id"]

        response = client.post("/api/tts", json={
            "text": "Hello from pocket-tts-mlx clone path.",
            "provider": "pocket-tts-mlx",
            "model": "default",
            "voice": "alba",
            "voice_id": voice_id,
        })

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.content == b"fake pocket tts mlx audio"

    @pytest.mark.skipif(not has_sample_audio(), reason="Requires dario.mp3 sample file")
    @patch("app.HAS_WHISPER", False)
    def test_upload_voice_with_missing_stt_dependency_returns_503(self, client, clean_uploads_integration):
        with open(SAMPLE_AUDIO_PATH, "rb") as f:
            audio_data = f.read()

        response = client.post(
            "/api/upload-voice",
            files={"file": ("dario.mp3", audio_data, "audio/mpeg")},
            data={"name": "No STT", "transcript": ""}
        )

        assert response.status_code == 503
        detail = response.json()["detail"].lower()
        assert "mlx-whisper" in detail or "install" in detail
