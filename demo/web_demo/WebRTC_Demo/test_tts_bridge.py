import requests
import time
import sys
import os

# Configuration
BRIDGE_URL = "http://127.0.0.1:8090/audio/speech"
TEST_TEXT = "The quick brown fox jumps over the lazy dogs."
OUTPUT_FILE = "debug_tts_output.wav"

def test_tts():
    print(f"🚀 Sending TTS request to Bridge: '{TEST_TEXT}'")
    
    payload = {
        "model": "minicpm",
        "input": TEST_TEXT,
        "voice": "default",
        "response_format": "wav"
    }
    
    start_time = time.time()
    try:
        response = requests.post(BRIDGE_URL, json=payload, timeout=60)
        elapsed = time.time() - start_time
        
        print(f"⏱️ Request completed in {elapsed:.2f}s")
        print(f"📡 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            content_len = len(response.content)
            print(f"✅ Success! Received {content_len} bytes.")
            
            with open(OUTPUT_FILE, "wb") as f:
                f.write(response.content)
            print(f"💾 Saved response to {OUTPUT_FILE}")
            
            # Basic WAV header check
            if response.content.startswith(b'RIFF'):
                print("🎵 File has valid RIFF/WAV header.")
            else:
                print("⚠️ Warning: Received data does not have a standard WAV header.")
                
        else:
            print(f"❌ Error Response Body:\n{response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out after 60s.")
    except Exception as e:
        print(f"❌ FATAL: {str(e)}")

if __name__ == "__main__":
    test_tts()
