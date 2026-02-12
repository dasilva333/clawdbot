import socket
import json
import subprocess
import threading
import time
import os

# Configuration
BRIDGE_ADDR = ('127.0.0.1', 18099)
SERVICE_PATH = "/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/llama.cpp-omni/tools/omni/pyt2w/token2wav_service.py"

def start_mock_bridge():
    """Starts a simple TCP server to catch the audio stream."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(BRIDGE_ADDR)
    sock.listen(1)
    print(f"[Mock Bridge] Listening on {BRIDGE_ADDR}")
    
    conn, addr = sock.accept()
    print(f"[Mock Bridge] Connection from {addr}")
    
    total_received = 0
    try:
        while True:
            data = conn.recv(4096)
            if not data: break
            total_received += len(data)
            print(f"[Mock Bridge] Received {len(data)} bytes (Total: {total_received})")
    finally:
        conn.close()
        sock.close()
    print(f"[Mock Bridge] Stream finished. Received {total_received} total bytes.")

def run_test():
    # 1. Start bridge in background
    bridge_thread = threading.Thread(target=start_mock_bridge, daemon=True)
    bridge_thread.start()
    time.sleep(1)

    # 2. Start the Token2Wav service
    # We need to mock 'stepaudio2' because we don't want to load weights for a pipe test
    # We'll use a wrapper that mocks the class
    print("[Test] Starting Token2Wav service...")
    
    # Create a mock wrapper script
    mock_wrapper = "/tmp/mock_t2w.py"
    with open(mock_wrapper, "w") as f:
        f.write("""
import sys
import json
import numpy as np

# Mock the model
class MockToken2wav:
    def __init__(self, *args, **kwargs): pass
    def set_stream_cache(self, path): return {}, {}
    def stream(self, **kwargs):
        # Generate 1 second of "audio" (24000 samples)
        return np.random.uniform(-1, 1, 24000).astype(np.float32)

# Monkeypatch the service
import token2wav_service
token2wav_service.Token2wav = MockToken2wav
token2wav_service.main()
""")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(SERVICE_PATH)
    
    process = subprocess.Popen(
        [sys.executable, mock_wrapper],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )

    def read_stderr():
        for line in process.stderr:
            print(f"[Service Stderr] {line.strip()}")

    threading.Thread(target=read_stderr, daemon=True).start()

    # Wait for ready signal
    ready_line = process.stdout.readline()
    print(f"[Test] Service said: {ready_line.strip()}")

    # 3. Send commands
    print("[Test] Sending init...")
    process.stdin.write(json.dumps({"cmd": "init", "model_dir": "/tmp"}) + "\\n")
    process.stdin.flush()
    print(f"[Test] Response: {process.stdout.readline().strip()}")

    print("[Test] Sending set_ref_audio...")
    # Just need a path that exists
    process.stdin.write(json.dumps({"cmd": "set_ref_audio", "ref_audio_path": SERVICE_PATH}) + "\\n")
    process.stdin.flush()
    print(f"[Test] Response: {process.stdout.readline().strip()}")

    print("[Test] Sending process (Triggering audio generation)...")
    process.stdin.write(json.dumps({"cmd": "process", "tokens": [1,2,3], "last_chunk": False, "output_path": "/tmp/test.wav"}) + "\\n")
    process.stdin.flush()
    print(f"[Test] Response: {process.stdout.readline().strip()}")

    time.sleep(2) # Give time for socket transfer
    
    print("[Test] Shutting down...")
    process.stdin.write(json.dumps({"cmd": "quit"}) + "\\n")
    process.stdin.flush()
    process.terminate()

if __name__ == "__main__":
    import sys
    run_test()
