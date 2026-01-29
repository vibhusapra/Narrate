#!/bin/bash
#
# Setup script for Fish Speech TTS backend
#

set -e

echo "=== Narrate - Fish Speech Setup ==="
echo ""

# Check if fish-speech directory already exists
if [ -d "fish-speech" ]; then
    echo "fish-speech directory already exists."
    read -p "Remove and re-clone? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf fish-speech
    else
        echo "Using existing fish-speech directory."
        cd fish-speech
    fi
fi

# Clone if not exists
if [ ! -d "fish-speech" ]; then
    echo "Cloning Fish Speech..."
    git clone https://github.com/fishaudio/fish-speech.git
    cd fish-speech
fi

# Check for Docker
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo ""
    echo "Docker detected. Starting Fish Speech API server..."
    echo "This may take a while on first run (downloading model weights)."
    echo ""
    docker compose --profile server up -d
    echo ""
    echo "Fish Speech API server starting in background."
    echo "API will be available at: http://localhost:8080"
    echo ""
    echo "To view logs: docker compose logs -f"
    echo "To stop: docker compose down"
else
    echo ""
    echo "Docker not found. Please install Fish Speech manually:"
    echo ""
    echo "  1. Install dependencies:"
    echo "     sudo apt install portaudio19-dev libsox-dev ffmpeg"
    echo ""
    echo "  2. Create conda environment:"
    echo "     conda create -n fish-speech python=3.12"
    echo "     conda activate fish-speech"
    echo "     pip install -e .[cu129]"
    echo ""
    echo "  3. Start API server:"
    echo "     python -m tools.api_server --listen 0.0.0.0:8080 \\"
    echo "       --llama-checkpoint-path checkpoints/openaudio-s1-mini \\"
    echo "       --decoder-checkpoint-path checkpoints/openaudio-s1-mini/codec.pth \\"
    echo "       --decoder-config-name modded_dac_vq"
    echo ""
fi

echo ""
echo "Once Fish Speech is running, start Narrate:"
echo "  cd .. && python app.py"
echo ""
