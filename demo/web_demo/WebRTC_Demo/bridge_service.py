import os
import time
import requests
import uvicorn
import glob
import subprocess
import traceback
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import numpy as np
import soundfile as sf
import io
import librosa

app = FastAPI()

# Configuration
CPP_SERVER_URL = "http://127.0.0.1:19060"
BASE_OUTPUT_DIR = os.path.abspath("llama.cpp-omni/tools/omni/output_9060")
TEMP_PREFILL_DIR = os.path.join(os.path.dirname(__file__), "temp_prefill")

# Global state
prefill_counters = {}
VOICE_DIR = os.path.join(os.path.dirname(__file__), "assets", "voices")
current_voice_path = None
audio_queue = asyncio.Queue()
decode_in_progress = False

# Ensure dirs exist
os.makedirs(VOICE_DIR, exist_ok=True)

# Startup Cleanup: Clear output directory to ensure Round ID sync with C++ server
if os.path.exists(BASE_OUTPUT_DIR):
    import shutil
    try:
        shutil.rmtree(BASE_OUTPUT_DIR)
        print(f"[Startup] Cleared output directory: {BASE_OUTPUT_DIR}")
    except Exception as e:
        print(f"[Startup] Warning: Failed to clear output directory: {e}")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

class VoiceSetRequest(BaseModel):
    voice_id: str

def _get_voice_path(voice_id: str) -> Optional[str]:
    """Resolve voice_id to a WAV file path."""
    if not voice_id or voice_id == "default":
        return None
    path = os.path.join(VOICE_DIR, f"{voice_id}.wav")
    if os.path.exists(path):
        return os.path.abspath(path)
    for f in os.listdir(VOICE_DIR):
        if f.lower() == f"{voice_id.lower()}.wav":
            return os.path.abspath(os.path.join(VOICE_DIR, f))
    return None

class OpenAITTSRequest(BaseModel):
    """OpenAI-compatible TTS request format."""
    model: str = "minicpm"
    input: str
    voice: str = "default"
    response_format: Optional[str] = "opus"

async def handle_tcp_client(reader, writer):
    """Receive length-prefixed PCM bytes from Sidecar."""
    global audio_queue
    import struct
    addr = writer.get_extra_info('peername')
    print(f"[Bridge TCP] New Connection from {addr}")
    try:
        while True:
            header = await reader.readexactly(4)
            if not header: break
            
            length = struct.unpack('<I', header)[0]
            if length == 0:
                print("[Bridge TCP] RECEIVED: End-of-Turn (EOT).")
                await audio_queue.put(None)
                continue
            
            data = await reader.readexactly(length)
            if data == b"PING":
                print("[Bridge TCP] Handshake PING. Pipe is live.")
                writer.write(struct.pack('<I', 4) + b"PONG")
                await writer.drain()
                continue
            
            # LOGGING: See the bytes!
            print(f"[Bridge TCP] RECEIVED: {length} bytes of audio.")
            await audio_queue.put(data)
    except asyncio.IncompleteReadError:
        print("[Bridge TCP] Client disconnected (IncompleteRead).")
    except Exception as e:
        print(f"[Bridge TCP] Error: {e}")
    finally:
        # Only put None if we aren't already closed
        await audio_queue.put(None)
        writer.close()

async def start_tcp_server():
    """Start the TCP server to listen for audio chunks."""
    server = await asyncio.start_server(handle_tcp_client, '127.0.0.1', 18099)
    print(f'[Bridge TCP] Server active on 127.0.0.1:18099')
    async with server:
        await server.serve_forever()

@app.on_event("startup")
async def startup_event():
    """Start the background TCP server."""
    asyncio.create_task(start_tcp_server())

@app.get("/voices")
async def list_voices():
    """List available voices."""
    voices = []
    if os.path.exists(VOICE_DIR):
        voices = [f.replace(".wav", "") for f in os.listdir(VOICE_DIR) if f.endswith(".wav")]
    return {"voices": sorted(voices), "current": os.path.basename(current_voice_path) if current_voice_path else "default"}

@app.post("/voice/set")
async def set_voice(req: VoiceSetRequest):
    """Set the voice for the current session."""
    global current_voice_path
    voice_path = _get_voice_path(req.voice_id)
    if req.voice_id != "default" and not voice_path:
        raise HTTPException(status_code=404, detail=f"Voice '{req.voice_id}' not found")

    print(f"[Bridge] Setting voice to: {req.voice_id}")
    try:
        update_req = {
            "media_type": 2, # omni mode
            "duplex_mode": False, # simplex
            "language": "en" # Force English
        }
        if voice_path:
            update_req["voice_audio"] = voice_path
            
        resp = requests.post(
            f"{CPP_SERVER_URL}/v1/stream/update_session_config",
            json=update_req,
            timeout=10
        )
        if resp.status_code != 200:
             raise HTTPException(status_code=500, detail=f"C++ update failed: {resp.text}")
             
        current_voice_path = voice_path
        return {"status": "ok", "voice": req.voice_id}
    except Exception as e:
        print(f"[Bridge] Voice switch error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/audio/speech")
async def audio_speech(request: OpenAITTSRequest):
    """
    OpenAI-compatible TTS endpoint (Streaming version).
    Pipes audio chunks from C++ TCP socket directly to the HTTP response.
    """
    print(f"[Bridge] TTS Request: '{request.input}'")
    global audio_queue, decode_in_progress
    
    # Clear queue
    while not audio_queue.empty():
        audio_queue.get_nowait()
    
    async def tts_streamer():
        global decode_in_progress
        decode_in_progress = True
        
        # 1. Trigger C++ generation (fire-and-forget)
        def _trigger():
            try:
                requests.post(
                    f"{CPP_SERVER_URL}/v1/tts/inject_text",
                    json={"text": request.input},
                    timeout=30
                )
            except Exception as e:
                print(f"[Bridge] C++ Trigger error: {e}")

        asyncio.create_task(asyncio.to_thread(_trigger))
        
        try:
            print("[Bridge] TTS: Awaiting audio from pipe...")
            while True:
                chunk = await audio_queue.get()
                if chunk is None:
                    print("[Bridge] TTS: Stream finished (EOT).")
                    break
                
                # Yield raw PCM bytes (24kHz Mono 16-bit)
                yield chunk
        finally:
            decode_in_progress = False
            print("[Bridge] TTS: Stream handler closed.")

    return StreamingResponse(tts_streamer(), media_type="audio/pcm")

@app.post("/omni/decode")
async def trigger_decode(request: Request):
    """Open Pipe v2: Trigger C++ decode and yield bytes directly from the TCP queue."""
    guild_id = request.headers.get("X-Guild-ID", "default")
    print(f"[Bridge] Open Pipe: Direct Stream starting for guild={guild_id}")
    
    global audio_queue, decode_in_progress
    
    # Clear the queue
    while not audio_queue.empty():
        audio_queue.get_nowait()
    
    async def audio_streamer():
        global decode_in_progress
        decode_in_progress = True
        
        # 1. Start C++ decode (fire and forget)
        def _trigger():
            try:
                requests.post(f"{CPP_SERVER_URL}/v1/stream/decode", json={}, timeout=2)
            except requests.exceptions.ReadTimeout:
                pass
            except Exception as e:
                print(f"[Bridge] C++ Trigger failed: {e}")

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _trigger)
        
        try:
            print("[Bridge] Open Pipe: Awaiting bytes from TCP queue...")
            while True:
                # Wait for data from TCP queue. 
                # The Sidecar will send None (EOT) when the model is finished.
                chunk = await audio_queue.get()
                
                if chunk is None:
                    print("[Bridge] Open Pipe: Received EOT signal. Closing stream.")
                    break
                
                print(f"[Bridge] Open Pipe: YIELDING {len(chunk)} bytes to client.")
                yield chunk
                
        finally:
            decode_in_progress = False
            prefill_counters[guild_id] = 1
            print(f"[Bridge] Open Pipe: Stream closed for guild={guild_id}")

    return StreamingResponse(audio_streamer(), media_type="audio/pcm")

@app.post("/omni/streaming_prefill")
async def streaming_prefill(request: Request):
    """Receive raw PCM chunks and forward to C++ server."""
    guild_id = request.headers.get("X-Guild-ID", "default")
    pcm_data = await request.body()
    
    if not pcm_data:
        return {"status": "ok"}

    try:
        # Plugin sends 48000Hz Mono 16-bit PCM
        audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio_np) == 0:
            return {"status": "ok"}

        # Resample to 16000Hz (C++ server requirement)
        audio_16k = librosa.resample(audio_np, orig_sr=48000, target_sr=16000)
        
        os.makedirs(TEMP_PREFILL_DIR, exist_ok=True)
        cnt = prefill_counters.get(guild_id, 1)
        
        temp_path = os.path.abspath(os.path.join(TEMP_PREFILL_DIR, f"prefill_{guild_id}_{cnt}.wav"))
        sf.write(temp_path, audio_16k, 16000, format='WAV', subtype='PCM_16')
        
        cpp_req = {
            "audio_path_prefix": temp_path,
            "img_path_prefix": "",
            "cnt": cnt
        }
        
        print(f"[Bridge] Prefill guild={guild_id} cnt={cnt}")
        resp = requests.post(f"{CPP_SERVER_URL}/v1/stream/prefill", json=cpp_req, timeout=5)
        
        if resp.status_code == 200:
            prefill_counters[guild_id] = cnt + 1
            return {"status": "ok", "cnt": cnt}
        else:
            return {"status": "error", "message": resp.text}
            
    except Exception as e:
        print(f"[Bridge] Prefill exception: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health():
    """Health check endpoint with version info."""
    return {
        "status": "healthy",
        "message": "Scoped TTS Bridge active (v2026-02-11-v14)",
        "backend": "cpp"
    }

if __name__ == "__main__":
    print(f"Starting MacK-Scoped TTS Bridge (v2026-02-11-v14) on port 8090...")
    uvicorn.run(app, host="0.0.0.0", port=8090)
