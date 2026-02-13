import requests
import sys
import os
import subprocess
import json
import time

BRIDGE_URL = "http://127.0.0.1:8090"
GUILD_ID = "harness-test-guild"

def run_test(wav_path):
    print(f"🚀 Starting Voice Harness Test: {wav_path}")
    
    # 1. Transcode to 48kHz Mono s16le (what the bridge expects)
    print("📢 Transcoding input...")
    # 🔧 Use absolute path for input and output to avoid relative issues
    abs_wav = os.path.abspath(wav_path)
    cmd = [
        "ffmpeg", "-y", "-i", abs_wav,
        "-ar", "48000", "-ac", "1", "-f", "s16le", "-"
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pcm_data, stderr = process.communicate()
    
    if not pcm_data:
        print(f"❌ Transcoding failed: {stderr.decode()}")
        return

    # 2. Inject in chunks (simulate streaming)
    print(f"💉 Injecting {len(pcm_data)} bytes...")
    chunk_size = 14400 # ~150ms
    for i in range(0, len(pcm_data), chunk_size):
        chunk = pcm_data[i:i+chunk_size]
        requests.post(f"{BRIDGE_URL}/omni/streaming_prefill", 
                      data=chunk, 
                      headers={"X-Guild-ID": GUILD_ID})
    
    # 🔧 [Sync Fix] Wait for the LLM thread in the engine to finish processing the audio
    # before triggering decode, otherwise we get KV cache position errors.
    print("⏳ Waiting for engine processing...")
    time.sleep(3) 

    # 3. Trigger Decode and capture stream
    print("📡 Triggering Decode & Capturing Stream...")
    start_time = time.time()
    response = requests.post(f"{BRIDGE_URL}/omni/decode", 
                            headers={"X-Guild-ID": GUILD_ID}, 
                            stream=True)
    
    output_path = "harness_response.raw"
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                f.write(chunk)
    
    end_time = time.time()
    print(f"✅ Captured response in {end_time - start_time:.2f}s")
    
    # 4. Convert to WAV for Whisper (MiniCPM-o is 24kHz)
    print("🎵 Converting response to WAV...")
    subprocess.run([
        "ffmpeg", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1",
        "-i", output_path, "harness_response.wav"
    ], stderr=subprocess.DEVNULL)
    
    print("✨ Harness complete. Response saved to harness_response.wav")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 voice_harness.py <input.wav>")
        sys.exit(1)
    run_test(sys.argv[1])
