#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Add Homebrew to PATH (Critical for finding cmake, ffmpeg, livekit-server, etc)
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"

# Ensure Virtual Environment
if [[ ! -d "venv" ]]; then
    echo "Creating Python virtual environment..."
    /opt/homebrew/bin/python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
else
    source venv/bin/activate
fi

# Set Configuration
export PYTHON_CMD="$(pwd)/venv/bin/python"
export PIP_CMD="$(pwd)/venv/bin/pip"
export VISION_BACKEND="metal"
export LLM_QUANT="Q4_K_M"
# Simplex mode (voice only/interaction optimized) is now default, but enforcing it ensures consistency
export CPP_MODE="simplex"

# Run oneclick.sh with arguments
bash oneclick.sh "$@"
