#!/bin/bash
cd "$(dirname "$0")"

# Activate venv if it exists (assuming typical Python setup)
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Install dependencies if needed
pip install fastapi uvicorn requests

# Run the bridge
echo "Starting Unified Bridge Master..."
/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/venv/bin/python3 bridge_service.py
