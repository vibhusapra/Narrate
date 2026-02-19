#!/usr/bin/env python3
"""
Quick test script for chunked TTS generation.
Tests the chunking with a ~300 word text.
"""

import httpx
import time
import asyncio
from pathlib import Path

# Test configuration
NARRATE_URL = "http://localhost:3000"
SAMPLE_VOICE = "dario.mp3"
TEST_TEXT = "This is a test of the chunked generation system. " * 60  # ~300 words

async def main():
    print("=" * 60)
    print("Testing Chunked TTS Generation")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=300.0) as client:
        # Step 1: Upload voice sample
        print("\n1. Uploading voice sample...")
        with open(SAMPLE_VOICE, "rb") as f:
            files = {"file": ("dario.mp3", f.read(), "audio/mpeg")}
            data = {
                "name": "Dario Test",
                "transcript": "This is Dario speaking in a sample audio clip."
            }

            response = await client.post(
                f"{NARRATE_URL}/api/upload-voice",
                files=files,
                data=data
            )

            if response.status_code != 200:
                print(f"❌ Upload failed: {response.text}")
                return

            voice_id = response.json()["voice_id"]
            print(f"✅ Voice uploaded: {voice_id}")

        # Step 2: Start chunked generation
        print(f"\n2. Starting chunked generation...")
        word_count = len(TEST_TEXT.split())
        print(f"   Text: {word_count} words")

        response = await client.post(
            f"{NARRATE_URL}/api/tts/chunked",
            json={
                "text": TEST_TEXT,
                "provider": "mlx-voice-clone",
                "model": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
                "voice_id": voice_id
            }
        )

        if response.status_code != 200:
            print(f"❌ Start failed: {response.text}")
            return

        session_id = response.json()["session_id"]
        print(f"✅ Session started: {session_id}")

        # Step 3: Poll for progress
        print("\n3. Waiting for generation to complete...")
        start_time = time.time()

        while True:
            response = await client.get(f"{NARRATE_URL}/api/tts/status/{session_id}")
            status = response.json()

            if status["status"] == "processing":
                progress = status.get("progress", {})
                current = progress.get("current", 0)
                total = progress.get("total", 0)
                if total > 0:
                    pct = (current / total) * 100
                    print(f"   Progress: Chunk {current}/{total} ({pct:.0f}%)")
                else:
                    print(f"   Progress: Initializing...")

            elif status["status"] == "complete":
                elapsed = time.time() - start_time
                print(f"✅ Generation complete in {elapsed:.1f}s")
                break

            elif status["status"] == "error":
                print(f"❌ Generation failed: {status.get('error')}")
                return

            await asyncio.sleep(2)

        # Step 4: Download audio
        print("\n4. Downloading audio...")
        response = await client.get(f"{NARRATE_URL}/api/tts/download/{session_id}")

        if response.status_code != 200:
            print(f"❌ Download failed: {response.text}")
            return

        audio_size = len(response.content)
        output_file = f"test_output_{session_id[:8]}.mp3"

        with open(output_file, "wb") as f:
            f.write(response.content)

        print(f"✅ Audio downloaded: {audio_size:,} bytes")
        print(f"   Saved to: {output_file}")

        # Step 5: Cleanup
        print("\n5. Cleaning up...")
        await client.delete(f"{NARRATE_URL}/api/voices/{voice_id}")
        print(f"✅ Voice deleted")

    print("\n" + "=" * 60)
    print("✅ Test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
