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
import re

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

# 🔧 Phase 2: Inbound TCP State
inbound_reader = None
inbound_writer = None

async def get_inbound_connection():
    global inbound_reader, inbound_writer
    if inbound_writer is None:
        try:
            print("[Bridge] Connecting to C++ Inbound TCP on 18100...")
            inbound_reader, inbound_writer = await asyncio.open_connection('127.0.0.1', 18100)
            print("[Bridge] Connected to C++ Inbound TCP.")
        except Exception as e:
            print(f"[Bridge] Inbound TCP connection failed: {e}")
            inbound_writer = None
    return inbound_writer

async def send_inbound_pcm(data: bytes, index: int):
    writer = await get_inbound_connection()
    if writer:
        try:
            import struct
            # Protocol: [4-byte length][4-byte index][Data]
            writer.write(struct.pack('<II', len(data), index))
            writer.write(data)
            await writer.drain()
        except Exception as e:
            print(f"[Bridge] Send inbound PCM failed: {e}")
            global inbound_writer
            inbound_writer = None # Reset for reconnect

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
    OpenAI-compatible TTS endpoint (Pure TCP Pipe).
    Captures raw PCM from the C++ socket and returns a playable WAV.
    """
    print(f"[Bridge] TTS Request: '{request.input}'")
    global audio_queue, decode_in_progress
    
    # Clear queue of stale data
    while not audio_queue.empty():
        audio_queue.get_nowait()
    
    decode_in_progress = True
    start_time = time.time()
    collected_pcm = []
    received_any = False
    
    # 1. Trigger C++ generation
    def _trigger():
        try:
            print("[Bridge] Sending /v1/tts/inject_text to C++...")
            resp = requests.post(
                f"{CPP_SERVER_URL}/v1/tts/inject_text",
                json={"text": request.input},
                timeout=30
            )
            print(f"[Bridge] C++ Inject Response: {resp.status_code}")
            
            # 🔧 FIX: Must trigger decode after text injection to start generation!
            if resp.status_code == 200:
                print("[Bridge] Sending /v1/stream/decode to C++...")
                requests.post(f"{CPP_SERVER_URL}/v1/stream/decode", json={}, timeout=10)
                
        except Exception as e:
            print(f"[Bridge] C++ Trigger error: {e}")

    asyncio.create_task(asyncio.to_thread(_trigger))
    
    try:
        print("[Bridge] TTS: Awaiting bytes from TCP pipe...")
        while True:
            elapsed = time.time() - start_time
            # 🔧 25s hard stop to beat OpenClaw fallback
            if elapsed > 25:
                print(f"[Bridge] TTS: Hard stop reached ({elapsed:.1f}s). Processing collected data.")
                break
                
            try:
                # 🔧 Protocol Logic:
                # 1. Trust Engine EOT (chunk is None).
                # 2. Use 2s silence timeout as Fallback (once data has started).
                # 3. Use remaining 25s window as max wait for initial data.
                
                current_timeout = 2.0 if received_any else (25.0 - elapsed)
                chunk = await asyncio.wait_for(audio_queue.get(), timeout=max(0.1, current_timeout))
                
                if chunk is None:
                    print("[Bridge] TTS: Received native EOT signal from engine.")
                    break
                
                received_any = True
                collected_pcm.append(chunk)
            except asyncio.TimeoutError:
                if received_any:
                    print("[Bridge] TTS: Fallback triggered: 2s silence detected. Processing.")
                else:
                    print("[Bridge] TTS: Engine warmup timed out (25s).")
                break

    except Exception as e:
        err_msg = f"FATAL TTS ERROR: {str(e)}\n{traceback.format_exc()}"
        print("\n" + "!"*80)
        print(err_msg)
        print("!"*80 + "\n")
        return Response(status_code=500, content=err_msg)
    finally:
        decode_in_progress = False

    if not collected_pcm:
        print("[Bridge] TTS Error: No bytes received from pipe.")
        return Response(status_code=500, content="Engine produced 0 bytes via TCP")

    # 2. Return proper WAV (In-memory only, NO DISK I/O)
    full_pcm = b"".join(collected_pcm)
    audio_np = np.frombuffer(full_pcm, dtype=np.int16)
    
    out_buf = io.BytesIO()
    sf.write(out_buf, audio_np, 24000, format='WAV') # MiniCPM-o uses 24kHz
    out_buf.seek(0)
    
    data = out_buf.read()
    print(f"[Bridge] TTS: Returning WAV from pipe ({len(data)} bytes).")
    return Response(content=data, media_type="audio/wav")

@app.post("/omni/decode")
async def trigger_decode(request: Request):
    """Open Pipe v2: Trigger C++ decode and yield bytes directly from the TCP queue."""
    guild_id = request.headers.get("X-Guild-ID", "default")
    print(f"[Bridge] Open Pipe: Direct Stream starting for guild={guild_id}")
    
    # 🔧 Phase 2 FIX: Reset counter IMMEDIATELY for Turn 2 transition
    prefill_counters[guild_id] = 1
    
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
                # 🔧 Phase 2: Still use HTTP trigger for control flow
                requests.post(f"{CPP_SERVER_URL}/v1/stream/decode", json={}, timeout=10)
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
            # 🔧 [TURN 2 FIX] Reset counter for next turn to ensure C++ adds <|audio_start|>
            prefill_counters[guild_id] = 1
            print(f"[Bridge] Open Pipe: Stream closed for guild={guild_id}. Resetting counter to 1.")

    return StreamingResponse(audio_streamer(), media_type="audio/pcm")

@app.post("/omni/streaming_prefill")
async def streaming_prefill(request: Request):
    """Receive raw PCM chunks and forward to C++ server via TCP."""
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
        
        # Convert back to s16le bytes for TCP transmission
        pcm_16k = (audio_16k * 32767.0).astype(np.int16).tobytes()
        
        # 🔧 Phase 2: Send over TCP instead of saving file + HTTP
        cnt = prefill_counters.get(guild_id, 1)
        await send_inbound_pcm(pcm_16k, cnt)
        
        # Keep internal turn counter for logging
        prefill_counters[guild_id] = cnt + 1
        return {"status": "ok", "cnt": cnt}
            
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
