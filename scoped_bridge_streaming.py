import os
import time
import requests
import uvicorn
import glob
import subprocess
import traceback
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# Configuration
CPP_SERVER_URL = "http://127.0.0.1:19060"
# Root output directory
BASE_OUTPUT_DIR = os.path.abspath("llama.cpp-omni/tools/omni/output_9060")

class TTSRequest(BaseModel):
    text: str
    voice_id: str = "default"

class OpenAITTSRequest(BaseModel):
    """OpenAI-compatible TTS request format."""
    model: str = "minicpm"
    input: str
    voice: str = "default"
    response_format: Optional[str] = "opus"

def _find_active_dir():
    """Find the latest round_XXX directory."""
    rounds = sorted(glob.glob(os.path.join(BASE_OUTPUT_DIR, "round_*")))
    if not rounds:
        return None
    return rounds[-1]

async def _chunk_generator(text: str):
    """Generator that watches for new WAV chunks and yields them immediately, stripping headers."""
    try:
        print(f"[Bridge] Starting streaming TTS: '{text[:50]}...'")
        
        # 1. Inject Text
        inject_resp = requests.post(
            f"{CPP_SERVER_URL}/v1/tts/inject_text",
            json={"text": text},
            timeout=10
        )
        inject_resp.raise_for_status()

        # 2. Trigger Decode
        decode_resp = requests.post(
            f"{CPP_SERVER_URL}/v1/stream/decode",
            json={},
            timeout=30
        )
        decode_resp.raise_for_status()

        # 3. Resolve the SPECIFIC round directory
        await asyncio.sleep(1.0)
        round_dir = _find_active_dir()
        if not round_dir:
            return

        output_dir = os.path.join(round_dir, "tts_wav")
        if not os.path.exists(output_dir):
            output_dir = round_dir
            
        flag_path = os.path.join(output_dir, "generation_done.flag")
        if not os.path.exists(flag_path) and os.path.exists(os.path.join(round_dir, "generation_done.flag")):
            flag_path = os.path.join(round_dir, "generation_done.flag")

        print(f"[Bridge] Streaming from: {output_dir}")

        # 4. Watch for chunks (Ordered)
        seen_chunks = set()
        next_chunk_index = 0
        start_time = time.time()
        
        while time.time() - start_time < 60: # 60s timeout
            # Look for specifically indexed chunks to ensure ordering
            pattern = os.path.join(output_dir, f"wav_{next_chunk_index:04d}.wav")
            candidates = glob.glob(pattern)
            
            if candidates:
                chunk_path = candidates[0]
                print(f"[Bridge] Yielding ordered chunk: {os.path.basename(chunk_path)}")
                with open(chunk_path, "rb") as f:
                    data = f.read()
                    # Strip 44-byte WAV header from every chunk to yield raw PCM
                    # This allows the Receiver to treat the entire stream as one PCM flow
                    if len(data) > 44:
                        yield data[44:]
                
                seen_chunks.add(chunk_path)
                next_chunk_index += 1
                continue # Check for next index immediately
            
            if os.path.exists(flag_path):
                # Double check for one last chunk that might have landed just before the flag
                pattern = os.path.join(output_dir, f"wav_{next_chunk_index:04d}.wav")
                if not glob.glob(pattern):
                    print("[Bridge] Streaming complete (flag found)")
                    break
                
            await asyncio.sleep(0.05) # Very tight loop for low latency

    except Exception as e:
        print(f"[Bridge] Streaming error: {e}")
        traceback.print_exc()

def _do_tts(text: str, response_format: str = "wav"):
    """Core TTS logic: inject text, decode, merge WAV chunks from the SPECIFIC round folder."""
    try:
        print(f"[Bridge] Starting TTS task: '{text[:50]}...' format={response_format}")
        
        # 1. Inject Text
        print(f"[Bridge] Injecting text to C++ server...")
        inject_resp = requests.post(
            f"{CPP_SERVER_URL}/v1/tts/inject_text",
            json={"text": text},
            timeout=10
        )
        inject_resp.raise_for_status()

        # 2. Trigger Decode
        print(f"[Bridge] Triggering decode...")
        decode_resp = requests.post(
            f"{CPP_SERVER_URL}/v1/stream/decode",
            json={},
            timeout=30
        )
        decode_resp.raise_for_status()

        # 3. Resolve the SPECIFIC round directory for this request
        time.sleep(1.5) # Wait for the new round folder to be created
        round_dir = _find_active_dir()
        if not round_dir:
            raise HTTPException(status_code=500, detail="No round directory created by server")
            
        # Target the tts_wav subfolder where chunks actually live
        output_dir = os.path.join(round_dir, "tts_wav")
        if not os.path.exists(output_dir):
            output_dir = round_dir # Fallback
            
        flag_path = os.path.join(output_dir, "generation_done.flag")
        # Check if flag is at the round level instead of tts_wav level
        if not os.path.exists(flag_path) and os.path.exists(os.path.join(round_dir, "generation_done.flag")):
            flag_path = os.path.join(round_dir, "generation_done.flag")

        print(f"[Bridge] Target directory: {output_dir}")
        print(f"[Bridge] Watching for flag: {flag_path}")

        # 4. Wait for generation to complete in THIS specific folder
        found_flag = False
        for i in range(120): # 60 seconds max
            if os.path.exists(flag_path):
                print(f"[Bridge] Flag found after {i*0.5}s")
                found_flag = True
                break
            time.sleep(0.5)

        if not found_flag:
            print("[Bridge] WARNING: Timeout waiting for generation_done.flag. Checking chunks in active dir anyway.")

        # 5. Collect and Sort WAV chunks ONLY from the current output_dir
        wav_files = sorted(glob.glob(os.path.join(output_dir, "wav_*.wav")))
        
        if not wav_files:
            print(f"[Bridge] ERROR: No WAV chunks found in {output_dir}")
            raise HTTPException(status_code=500, detail=f"No WAV chunks found in {output_dir}")

        print(f"[Bridge] Found {len(wav_files)} chunks in {os.path.basename(round_dir)}. Merging...")

        # 6. Concatenate using ffmpeg
        concat_list_path = os.path.join(round_dir, "concat_list.txt")
        with open(concat_list_path, "w") as f:
            for wav in wav_files:
                safe_path = os.path.abspath(wav).replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")

        merged_wav = os.path.join(round_dir, f"merged_{int(time.time())}.wav")
        
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path, "-c", "copy", merged_wav]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[Bridge] ffmpeg merge failed: {result.stderr}")
            raise HTTPException(status_code=500, detail="ffmpeg merge failed")

        if response_format == "opus":
            opus_file = merged_wav.replace(".wav", ".opus")
            print(f"[Bridge] Converting to Opus...")
            cmd_opus = ["ffmpeg", "-y", "-i", merged_wav, "-c:a", "libopus", "-b:a", "64k", "-f", "opus", opus_file]
            result_opus = subprocess.run(cmd_opus, capture_output=True, text=True)
            
            if result_opus.returncode != 0:
                print(f"[Bridge] Opus conversion failed: {result_opus.stderr}")
                return merged_wav
                
            if os.path.exists(merged_wav): os.remove(merged_wav)
            return opus_file

        return merged_wav

    except Exception as e:
        print(f"[Bridge] FATAL ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/tts/synthesize")
async def synthesize(req: TTSRequest, stream: bool = False):
    if stream:
        return StreamingResponse(_chunk_generator(req.text), media_type="audio/wav")
    
    output_file = _do_tts(req.text)
    return FileResponse(output_file, media_type="audio/wav")

@app.post("/audio/speech")
@app.post("/v1/audio/speech")
async def openai_tts(req: OpenAITTSRequest):
    print(f"[Bridge] Received OpenAI request: {req.input[:50]}...")
    output_file = _do_tts(req.input, response_format=req.response_format)
    
    mime_type = "audio/ogg" if output_file.endswith(".opus") else "audio/wav"
    
    with open(output_file, "rb") as f:
        audio_bytes = f.read()

    print(f"[Bridge] Sending {len(audio_bytes)} bytes")

    # Clean up generated file
    try: os.remove(output_file)
    except: pass

    return Response(
        content=audio_bytes,
        media_type=mime_type,
        headers={"Content-Disposition": f"attachment; filename=speech.{'opus' if output_file.endswith('.opus') else 'wav'}"}
    )

@app.get("/health")
@app.get("/v1/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    print(f"Starting Scoped TTS Bridge on port 8090...")
    uvicorn.run(app, host="0.0.0.0", port=8090)
