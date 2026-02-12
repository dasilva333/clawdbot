
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
import uvicorn
import base64
import os
import argparse

app = FastAPI(title="Mock MiniCPM-o TTS API")

# Mock audio data (1 second of silence in OGG Opus format)
MOCK_OGG_OPUS = (
    "T2dnUwACAAAAAAAAAABJza0AAAAAAIsZmZEBHgF2b3JiaXMAAAAAAkSsAAD/////AHcBAP////+4AU9nZ1MAAAAAAAAAAAAASc2tAAEAAAC7uT1QDnm//////////////////wJ2b3JiaXMsAAAA"
    "dm9yYmlzLQEAAAB4nO19WXMbx5Zmdt8/oItIidxdLCWxH94SKVESqdt2X7gPpKiLZEiJpJ32/f0BkqI4j/F45ozk7H3YJ5MJHgC5AGT+/5/33//9//1//3//f/9//3//f/9//3//f/9//3//f/9/"
    "/3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//"
    "f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9/"
    "/3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//"
)

@app.post("/v1/tts/synthesize")
async def synthesize(
    text: str = Body(..., embed=True),
    voice: str = Body("default", embed=True),
    format: str = Body("opus", embed=True)
):
    print(f"Synthesizing text: '{text}' with voice: '{voice}' format: '{format}'")
    
    # In a real implementation, this would call llama.cpp-omni to generate audio
    # For now, return mock audio
    
    audio_bytes = base64.b64decode(MOCK_OGG_OPUS) 
    
    # The plugin expects raw binary in the response body, not JSON
    from fastapi.responses import Response
    return Response(content=audio_bytes, media_type="audio/ogg")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "mock-minicpm-tts-api"}

@app.get("/v1/voices")
async def list_voices():
    return {
        "default": "Default Voice",
        "mock_voice": "Mock Voice Description"
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock TTS API Server")
    parser.add_argument("--port", type=int, default=8087, help="Port to run the server on")
    args = parser.parse_args()
    
    print(f"Starting Mock TTS API on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
