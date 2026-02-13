import os
import time
import json
import requests
import uvicorn
import glob
import subprocess
import traceback
import asyncio
from itertools import count
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
DEFAULT_REF_AUDIO = "/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/cpp_server/assets/default_ref_audio.wav"
VOICE_DIR = os.path.join(os.path.dirname(__file__), "assets", "voices")

# Global state
prefill_counters = {}
guild_audio_buffers = {} # 🔧 Buffer PCM chunks until decode
current_voice_path = DEFAULT_REF_AUDIO
audio_queue = asyncio.Queue()
decode_in_progress = False
voice_session_initialized = False
voice_session_guild_id = None
voice_session_lock = asyncio.Lock()
voice_turn_seq = count(1)
prefill_telemetry = {}

os.makedirs(VOICE_DIR, exist_ok=True)

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

def _log_latency(event: str, payload: dict) -> None:
    # Single-line JSON logs so latency can be grepped and parsed.
    print(f"[Latency][{event}] {json.dumps(payload, ensure_ascii=False, sort_keys=True)}")

def _fetch_prefill_status_sync():
    try:
        resp = requests.get(f"{CPP_SERVER_URL}/v1/stream/prefill_status", timeout=2)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None

async def _await_prefill_ready(previous_seq: int, guild_id: str, timeout_sec: float = 6.0, poll_sec: float = 0.02):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        status = await asyncio.to_thread(_fetch_prefill_status_sync)
        if status and status.get("success") is True:
            seq = int(status.get("prefill_signal_seq", 0))
            if seq > previous_seq:
                return True, seq
            if status.get("prefill_ready") is True and previous_seq < 0:
                return True, seq
        await asyncio.sleep(poll_sec)
    print(f"[Bridge] Prefill readiness timeout after {timeout_sec:.1f}s (guild={guild_id})")
    return False, previous_seq

async def handle_tcp_client(reader, writer):
    """Receive audio from Engine -> Bridge."""
    global audio_queue
    import struct
    try:
        while True:
            header = await reader.readexactly(4)
            if not header: break
            length = struct.unpack('<I', header)[0]
            if length == 0:
                await audio_queue.put(None)
                continue
            data = await reader.readexactly(length)
            await audio_queue.put(data)
    except Exception as e:
        pass
    finally:
        await audio_queue.put(None)
        writer.close()

async def start_tcp_server():
    server = await asyncio.start_server(handle_tcp_client, '127.0.0.1', 18099)
    async with server:
        await server.serve_forever()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(start_tcp_server())

class VoiceSetRequest(BaseModel):
    voice_id: str

class OpenAITTSRequest(BaseModel):
    model: str = "minicpm"
    input: str
    voice: str = "default"
    response_format: Optional[str] = "wav"
    prompt: Optional[str] = None
    temperature: Optional[float] = None

class SynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = "default"
    format: Optional[str] = "wav"
    prompt: Optional[str] = None
    temperature: Optional[float] = None

class InitSysPromptRequest(BaseModel):
    media_type: Optional[str] = "omni"
    system_prompt_prefix: Optional[str] = None
    system_prompt_suffix: Optional[str] = None
    temperature: Optional[float] = None
    voice_audio: Optional[str] = None
    language: Optional[str] = "en"

async def _ensure_voice_session(guild_id: str, temperature: float = 0.7) -> None:
    """Initialize the C++ omni session once per guild, keep context for later turns."""
    global voice_session_initialized, voice_session_guild_id

    async with voice_session_lock:
        needs_init = (not voice_session_initialized) or (voice_session_guild_id != guild_id)
        if not needs_init:
            return

        if voice_session_guild_id and voice_session_guild_id != guild_id:
            print(f"[Bridge] Guild switch: {voice_session_guild_id} -> {guild_id}; reinitializing session")
        else:
            print(f"[Bridge] Initializing voice session for guild={guild_id}")

        result = await init_sys_prompt(InitSysPromptRequest(temperature=temperature))
        if isinstance(result, dict):
            if result.get("status") == "error":
                raise HTTPException(status_code=500, detail=f"Session init failed: {result.get('message', 'unknown error')}")
            if result.get("success") is False:
                raise HTTPException(status_code=500, detail=f"Session init failed: {result}")

        voice_session_initialized = True
        voice_session_guild_id = guild_id

def _invalidate_voice_session(reason: str) -> None:
    """Force the next voice decode to reinitialize a clean omni session."""
    global voice_session_initialized, voice_session_guild_id
    if voice_session_initialized:
        print(f"[Bridge] Invalidating voice session (guild={voice_session_guild_id}): {reason}")
    voice_session_initialized = False
    voice_session_guild_id = None

def _get_voice_path(voice_id: str) -> Optional[str]:
    if not voice_id or voice_id == "default":
        return DEFAULT_REF_AUDIO
    path = os.path.join(VOICE_DIR, f"{voice_id}.wav")
    if os.path.exists(path):
        return os.path.abspath(path)
    for f in os.listdir(VOICE_DIR):
        if f.lower() == f"{voice_id.lower()}.wav":
            return os.path.abspath(os.path.join(VOICE_DIR, f))
    return None

def _pcm_to_wav_bytes(pcm_bytes: bytes) -> bytes:
    audio_np = np.frombuffer(pcm_bytes, dtype=np.int16)
    out_buf = io.BytesIO()
    sf.write(out_buf, audio_np, 24000, format='WAV')
    out_buf.seek(0)
    return out_buf.read()

def _wav_to_opus_bytes(wav_bytes: bytes) -> bytes:
    ffmpeg = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "wav",
            "-i",
            "pipe:0",
            "-c:a",
            "libopus",
            "-b:a",
            "64k",
            "-f",
            "opus",
            "pipe:1",
        ],
        input=wav_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if ffmpeg.returncode != 0:
        raise HTTPException(status_code=500, detail=f"ffmpeg opus conversion failed: {ffmpeg.stderr.decode(errors='ignore')[:300]}")
    return ffmpeg.stdout

def _clear_audio_queue() -> None:
    while not audio_queue.empty():
        audio_queue.get_nowait()

async def _collect_tts_pcm(text: str, prompt: Optional[str], temperature: Optional[float]) -> bytes:
    global decode_in_progress

    _clear_audio_queue()
    decode_in_progress = True
    start_time = time.time()
    started_pc = time.perf_counter()
    first_chunk_pc = None
    collected_pcm = []
    received_any = False
    trigger_metrics = {"status_code": None, "inject_ms": None, "decode_ms": None, "error": None}

    def _trigger():
        try:
            if prompt or temperature is not None:
                update_payload = {"media_type": 2}
                if prompt:
                    update_payload["system_prompt_prefix"] = prompt
                if temperature is not None:
                    update_payload["temperature"] = temperature
                requests.post(f"{CPP_SERVER_URL}/v1/stream/update_session_config", json=update_payload, timeout=10)

            final_text = f"Please read the following content. {text}"
            inject_started_pc = time.perf_counter()
            resp = requests.post(f"{CPP_SERVER_URL}/v1/tts/inject_text", json={"text": final_text}, timeout=30)
            trigger_metrics["inject_ms"] = round((time.perf_counter() - inject_started_pc) * 1000.0, 2)
            trigger_metrics["status_code"] = resp.status_code
            if resp.status_code == 200:
                decode_started_pc = time.perf_counter()
                requests.post(f"{CPP_SERVER_URL}/v1/stream/decode", json={}, timeout=10)
                trigger_metrics["decode_ms"] = round((time.perf_counter() - decode_started_pc) * 1000.0, 2)
        except Exception as e:
            trigger_metrics["error"] = str(e)
            print(f"[Bridge] TTS trigger failed: {e}")

    asyncio.create_task(asyncio.to_thread(_trigger))

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed > 25:
                break

            try:
                current_timeout = 2.0 if received_any else (25.0 - elapsed)
                chunk = await asyncio.wait_for(audio_queue.get(), timeout=max(0.1, current_timeout))
                if chunk is None:
                    # Ignore stale end markers that can arrive before a fresh stream starts.
                    if not received_any:
                        continue
                    break
                if first_chunk_pc is None:
                    first_chunk_pc = time.perf_counter()
                received_any = True
                collected_pcm.append(chunk)
            except asyncio.TimeoutError:
                break
    finally:
        decode_in_progress = False
        # TTS inject/decode mutates the same backend context used by voice calls.
        # Force voice path to re-init on next turn to avoid transcript/parrot drift.
        _invalidate_voice_session("tts request finished")

    total_pcm_bytes = sum(len(c) for c in collected_pcm)
    _log_latency("TTS", {
        "text_chars": len(text),
        "received_any": received_any,
        "status": "ok" if total_pcm_bytes > 0 else "empty",
        "total_ms": round((time.perf_counter() - started_pc) * 1000.0, 2),
        "first_audio_ms": round((first_chunk_pc - started_pc) * 1000.0, 2) if first_chunk_pc is not None else None,
        "pcm_bytes": total_pcm_bytes,
        "pcm_audio_sec": round(total_pcm_bytes / 48000.0, 3),  # 24kHz s16 mono
        "trigger_status": trigger_metrics["status_code"],
        "inject_post_ms": trigger_metrics["inject_ms"],
        "decode_post_ms": trigger_metrics["decode_ms"],
        "trigger_error": trigger_metrics["error"],
    })

    if not collected_pcm:
        raise HTTPException(status_code=500, detail="Engine produced 0 bytes via TCP")
    return b"".join(collected_pcm)

async def _stream_tts_pcm(text: str, prompt: Optional[str], temperature: Optional[float]):
    global decode_in_progress

    _clear_audio_queue()
    decode_in_progress = True
    start_time = time.time()
    started_pc = time.perf_counter()
    first_chunk_pc = None
    streamed_bytes = 0
    received_any = False
    trigger_metrics = {"status_code": None, "inject_ms": None, "decode_ms": None, "error": None}

    def _trigger():
        try:
            if prompt or temperature is not None:
                update_payload = {"media_type": 2}
                if prompt:
                    update_payload["system_prompt_prefix"] = prompt
                if temperature is not None:
                    update_payload["temperature"] = temperature
                requests.post(f"{CPP_SERVER_URL}/v1/stream/update_session_config", json=update_payload, timeout=10)

            final_text = f"Please read the following content. {text}"
            inject_started_pc = time.perf_counter()
            resp = requests.post(f"{CPP_SERVER_URL}/v1/tts/inject_text", json={"text": final_text}, timeout=30)
            trigger_metrics["inject_ms"] = round((time.perf_counter() - inject_started_pc) * 1000.0, 2)
            trigger_metrics["status_code"] = resp.status_code
            if resp.status_code == 200:
                decode_started_pc = time.perf_counter()
                requests.post(f"{CPP_SERVER_URL}/v1/stream/decode", json={}, timeout=10)
                trigger_metrics["decode_ms"] = round((time.perf_counter() - decode_started_pc) * 1000.0, 2)
        except Exception as e:
            trigger_metrics["error"] = str(e)
            print(f"[Bridge] Stream TTS trigger failed: {e}")

    asyncio.create_task(asyncio.to_thread(_trigger))

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed > 25:
                break

            try:
                current_timeout = 2.0 if received_any else (25.0 - elapsed)
                chunk = await asyncio.wait_for(audio_queue.get(), timeout=max(0.1, current_timeout))
                if chunk is None:
                    # Ignore stale end markers that can arrive before a fresh stream starts.
                    if not received_any:
                        continue
                    break
                if first_chunk_pc is None:
                    first_chunk_pc = time.perf_counter()
                received_any = True
                streamed_bytes += len(chunk)
                # Keep this endpoint raw PCM to match VoiceManager.playStream(..., isRaw=true).
                yield chunk
            except asyncio.TimeoutError:
                break
    finally:
        decode_in_progress = False
        # Keep TTS and voice-call sessions isolated across requests.
        _invalidate_voice_session("tts stream finished")
        _log_latency("TTSStream", {
            "text_chars": len(text),
            "received_any": received_any,
            "total_ms": round((time.perf_counter() - started_pc) * 1000.0, 2),
            "first_audio_ms": round((first_chunk_pc - started_pc) * 1000.0, 2) if first_chunk_pc is not None else None,
            "pcm_bytes": streamed_bytes,
            "pcm_audio_sec": round(streamed_bytes / 48000.0, 3),  # 24kHz s16 mono
            "trigger_status": trigger_metrics["status_code"],
            "inject_post_ms": trigger_metrics["inject_ms"],
            "decode_post_ms": trigger_metrics["decode_ms"],
            "trigger_error": trigger_metrics["error"],
        })

@app.post("/omni/init_sys_prompt")
async def init_sys_prompt(req: InitSysPromptRequest):
    """Bridge proxy for engine reconfiguration with official tag structure."""
    global current_voice_path
    try:
        # Official MiniCPM-o 4.5 System Block
        prefix = req.system_prompt_prefix or "Clone the voice in the provided audio prompt."
        if "<|im_start|>" not in prefix:
            # We open the system block AND the audio tag for the reference voice
            prefix = f"<|im_start|>system\n{prefix}\n<|audio_start|>"
            
        core_suffix = req.system_prompt_suffix or (
            "Your task is to be a helpful assistant using this voice pattern. "
            "Please answer the user's questions seriously and in a high quality. "
            "Please chat with the user in a high naturalness style. "
            "Always speak in English, regardless of the input language. "
            "Important: treat incoming audio as the user's intent and respond with new content. "
            "Do not transcribe, quote, or repeat the user's speech verbatim unless explicitly asked. "
            "Never begin your reply by mirroring the user's sentence. "
            "If the user asks a question, provide a direct semantic answer instead of repeating the question."
        )
        # We close the ref audio tag, add instructions, and close the system block
        # 🔧 Crucially, we do NOT open the user block here, we let the turn logic do it
        suffix = f"<|audio_end|>{core_suffix}<|im_end|>\n"
        
        if req.voice_audio:
            current_voice_path = req.voice_audio

        payload = {
            "media_type": 2 if req.media_type == "omni" else 1,
            "duplex_mode": False,
            "language": req.language or "en",
            "system_prompt_prefix": prefix,
            "system_prompt_suffix": suffix,
            "temperature": req.temperature if req.temperature is not None else 0.7,
            "voice_audio": current_voice_path,
            "model_dir": "/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/models/openbmb/MiniCPM-o-4_5-gguf",
            "tts_bin_dir": "/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/models/openbmb/MiniCPM-o-4_5-gguf/tts"
        }
        
        print(f"[Bridge] Re-init Context (Temp={payload['temperature']})")
        resp = requests.post(f"{CPP_SERVER_URL}/v1/stream/omni_init", json=payload, timeout=60)
        return resp.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/voices")
async def list_voices():
    voices = ["default"]
    if os.path.exists(VOICE_DIR):
        voices.extend([f.replace(".wav", "") for f in os.listdir(VOICE_DIR) if f.endswith(".wav")])
    return {"voices": sorted(set(voices)), "current": os.path.basename(current_voice_path) if current_voice_path else "default"}

@app.post("/voice/set")
async def set_voice(req: VoiceSetRequest):
    global current_voice_path
    voice_path = _get_voice_path(req.voice_id)
    if voice_path is None:
        raise HTTPException(status_code=404, detail=f"Voice '{req.voice_id}' not found")

    update_req = {
        "media_type": 2,
        "duplex_mode": False,
        "language": "en",
        "voice_audio": voice_path,
    }
    resp = requests.post(f"{CPP_SERVER_URL}/v1/stream/update_session_config", json=update_req, timeout=10)
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"C++ update failed: {resp.text}")

    current_voice_path = voice_path
    return {"status": "ok", "voice": req.voice_id}

@app.post("/audio/speech")
@app.post("/v1/audio/speech")
async def audio_speech(request: OpenAITTSRequest):
    pcm_bytes = await _collect_tts_pcm(request.input, request.prompt, request.temperature)
    wav_bytes = _pcm_to_wav_bytes(pcm_bytes)

    fmt = (request.response_format or "wav").lower()
    if fmt == "opus":
        opus_bytes = _wav_to_opus_bytes(wav_bytes)
        return Response(content=opus_bytes, media_type="audio/ogg")
    return Response(content=wav_bytes, media_type="audio/wav")

@app.post("/v1/tts/synthesize")
async def synthesize(req: SynthesizeRequest, stream: bool = False):
    if stream:
        return StreamingResponse(_stream_tts_pcm(req.text, req.prompt, req.temperature), media_type="audio/pcm")

    mapped = OpenAITTSRequest(
        input=req.text,
        voice=req.voice or "default",
        response_format=req.format or "wav",
        prompt=req.prompt,
        temperature=req.temperature,
    )
    return await audio_speech(mapped)

@app.post("/omni/decode")
async def trigger_decode(request: Request):
    """Decode the buffered guild audio as one user turn and stream back raw PCM."""
    guild_id = request.headers.get("X-Guild-ID", "default")
    print(f"[Bridge] Turn Start: {guild_id}")
    turn_id = next(voice_turn_seq)
    turn_started_pc = time.perf_counter()
    prefill_meta = prefill_telemetry.pop(guild_id, None)

    # Keep one initialized session per guild so multi-turn context survives.
    session_init_started_pc = time.perf_counter()
    await _ensure_voice_session(guild_id, temperature=0.7)
    session_init_ms = round((time.perf_counter() - session_init_started_pc) * 1000.0, 2)

    # 2. Flush queue
    global audio_queue, decode_in_progress
    while not audio_queue.empty(): audio_queue.get_nowait()

    # 3. Feed a user-audio turn (audio-only path; no text injection).
    # stream_decode() appends assistant generation prompt after async prefill completes.
    prefill_seq_before = -1
    prefill_status_supported = False
    prefill_probe_ms = None
    input_audio_sec = None
    send_inbound_ms = None
    prefill_wait_ms = None
    prefill_ready_seq = None
    prefill_ready = None
    prefill_wait_timed_out = None
    trigger_post_ms = None
    trigger_post_status = None
    trigger_post_error = None
    try:
        prefill_probe_started_pc = time.perf_counter()
        prefill_status = await asyncio.to_thread(_fetch_prefill_status_sync)
        prefill_probe_ms = round((time.perf_counter() - prefill_probe_started_pc) * 1000.0, 2)
        if prefill_status and prefill_status.get("success") is True:
            prefill_seq_before = int(prefill_status.get("prefill_signal_seq", 0))
            prefill_status_supported = True

        buffered_pcm = guild_audio_buffers.pop(guild_id, [])
        if not buffered_pcm:
            raise HTTPException(status_code=400, detail=f"No buffered audio for guild {guild_id}")

        # Send user audio embeddings via inbound TCP channel.
        full_pcm = b"".join(buffered_pcm)
        input_audio_sec = round(len(full_pcm) / 32000.0, 3)  # 16kHz s16 mono
        send_started_pc = time.perf_counter()
        await send_inbound_pcm(full_pcm, 1)

        # Send EOT to finalize TCP ingestion for this turn.
        writer = await get_inbound_connection()
        if writer:
            import struct
            writer.write(struct.pack('<II', 0, 1))
            await writer.drain()
        send_inbound_ms = round((time.perf_counter() - send_started_pc) * 1000.0, 2)

    except Exception as e:
        print(f"[Bridge] Injection failed: {e}")
        _log_latency("VoiceTurn", {
            "turn_id": turn_id,
            "guild": guild_id,
            "status": "inject_error",
            "error": str(e),
            "session_init_ms": session_init_ms,
            "prefill_probe_ms": prefill_probe_ms,
            "total_ms": round((time.perf_counter() - turn_started_pc) * 1000.0, 2),
        })
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Voice turn injection failed: {e}")

    if prefill_status_supported:
        prefill_wait_started_pc = time.perf_counter()
        ready, seq_seen = await _await_prefill_ready(prefill_seq_before, guild_id)
        prefill_wait_ms = round((time.perf_counter() - prefill_wait_started_pc) * 1000.0, 2)
        prefill_ready = ready
        prefill_ready_seq = seq_seen
        prefill_wait_timed_out = not ready
        if ready:
            print(f"[Bridge] Prefill ready via signal seq={seq_seen} (guild={guild_id})")
    else:
        print(f"[Bridge] Prefill status endpoint unavailable, decoding immediately (guild={guild_id})")

    async def audio_streamer():
        global decode_in_progress
        decode_in_progress = True
        stream_started_at = time.time()
        first_chunk_timeout_sec = 20.0
        idle_chunk_timeout_sec = 12.0
        idle_timeout_retries = 2
        max_stream_duration_sec = 120.0
        idle_timeouts = 0
        stream_started_pc = time.perf_counter()
        first_chunk_pc = None
        stream_output_bytes = 0
        stream_chunks = 0
        stream_end_reason = "completed"
        
        def _trigger():
            nonlocal trigger_post_ms, trigger_post_status, trigger_post_error
            try:
                # Let C++ own round index progression for multi-turn sessions.
                trigger_started_pc = time.perf_counter()
                resp = requests.post(f"{CPP_SERVER_URL}/v1/stream/decode", json={}, timeout=10)
                trigger_post_ms = round((time.perf_counter() - trigger_started_pc) * 1000.0, 2)
                trigger_post_status = resp.status_code
            except Exception as e:
                trigger_post_error = str(e)
                print(f"[Bridge] Trigger failed: {e}")

        asyncio.get_event_loop().run_in_executor(None, _trigger)
        
        received_any = False
        try:
            while True:
                if (time.time() - stream_started_at) > max_stream_duration_sec:
                    print(f"[Bridge] Stream timeout: exceeded {max_stream_duration_sec:.0f}s (guild={guild_id})")
                    stream_end_reason = "max_stream_timeout"
                    break
                try:
                    current_timeout = idle_chunk_timeout_sec if received_any else first_chunk_timeout_sec
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=current_timeout)
                    if chunk is None:
                        # Ignore stale end markers until we have actually started this stream.
                        if not received_any:
                            continue
                        break
                    if first_chunk_pc is None:
                        first_chunk_pc = time.perf_counter()
                    received_any = True
                    idle_timeouts = 0
                    stream_output_bytes += len(chunk)
                    stream_chunks += 1
                    yield chunk
                except asyncio.TimeoutError:
                    if not received_any:
                        print(f"[Bridge] Stream timeout before first chunk ({first_chunk_timeout_sec:.0f}s) guild={guild_id}")
                        stream_end_reason = "first_chunk_timeout"
                        break
                    idle_timeouts += 1
                    if idle_timeouts > idle_timeout_retries:
                        print(f"[Bridge] Stream idle timeout after {idle_timeouts} waits ({idle_chunk_timeout_sec:.0f}s each) guild={guild_id}")
                        stream_end_reason = "idle_timeout"
                        break
        finally:
            decode_in_progress = False
            total_turn_ms = round((time.perf_counter() - turn_started_pc) * 1000.0, 2)
            stream_total_ms = round((time.perf_counter() - stream_started_pc) * 1000.0, 2)
            first_audio_ms = round((first_chunk_pc - stream_started_pc) * 1000.0, 2) if first_chunk_pc is not None else None
            prefill_capture_ms = None
            decode_gap_ms = None
            if prefill_meta:
                prefill_capture_ms = round((prefill_meta["last_chunk_pc"] - prefill_meta["first_chunk_pc"]) * 1000.0, 2)
                decode_gap_ms = round((stream_started_pc - prefill_meta["last_chunk_pc"]) * 1000.0, 2)

            _log_latency("VoiceTurn", {
                "turn_id": turn_id,
                "guild": guild_id,
                "status": "ok" if received_any else "no_audio",
                "session_init_ms": session_init_ms,
                "prefill_probe_ms": prefill_probe_ms,
                "prefill_signal_supported": prefill_status_supported,
                "prefill_wait_ms": prefill_wait_ms,
                "prefill_ready": prefill_ready,
                "prefill_ready_seq": prefill_ready_seq,
                "prefill_wait_timed_out": prefill_wait_timed_out,
                "send_inbound_ms": send_inbound_ms,
                "input_audio_sec": input_audio_sec,
                "prefill_chunk_count": prefill_meta["chunk_count"] if prefill_meta else 0,
                "prefill_bytes_48k": prefill_meta["bytes_48k"] if prefill_meta else 0,
                "prefill_bytes_16k": prefill_meta["bytes_16k"] if prefill_meta else 0,
                "prefill_capture_ms": prefill_capture_ms,
                "decode_gap_ms": decode_gap_ms,
                "trigger_post_ms": trigger_post_ms,
                "trigger_post_status": trigger_post_status,
                "trigger_post_error": trigger_post_error,
                "first_audio_ms": first_audio_ms,
                "stream_total_ms": stream_total_ms,
                "stream_output_bytes": stream_output_bytes,
                "stream_output_audio_sec": round(stream_output_bytes / 48000.0, 3),  # 24kHz s16 mono
                "stream_chunks": stream_chunks,
                "stream_end_reason": stream_end_reason,
                "total_turn_ms": total_turn_ms,
            })

    return StreamingResponse(audio_streamer(), media_type="audio/pcm")

@app.post("/omni/streaming_prefill")
async def streaming_prefill(request: Request):
    guild_id = request.headers.get("X-Guild-ID", "default")
    pcm_data = await request.body()
    if not pcm_data: return {"status": "ok"}
    try:
        now_pc = time.perf_counter()
        audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio_np) == 0: return {"status": "ok"}
        audio_16k = librosa.resample(audio_np, orig_sr=48000, target_sr=16000)
        pcm_16k = (audio_16k * 32767.0).astype(np.int16).tobytes()
        if guild_id not in guild_audio_buffers: guild_audio_buffers[guild_id] = []
        guild_audio_buffers[guild_id].append(pcm_16k)
        meta = prefill_telemetry.get(guild_id)
        if meta is None:
            prefill_telemetry[guild_id] = {
                "first_chunk_pc": now_pc,
                "last_chunk_pc": now_pc,
                "chunk_count": 1,
                "bytes_48k": len(pcm_data),
                "bytes_16k": len(pcm_16k),
            }
        else:
            meta["last_chunk_pc"] = now_pc
            meta["chunk_count"] += 1
            meta["bytes_48k"] += len(pcm_data)
            meta["bytes_16k"] += len(pcm_16k)
        return {"status": "buffered"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "v3.2-official-wrap"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)
