import os
import sys
import time
import torch

# 🚨 EMERGENCY LOG: Write to a file we can definitely read
DEBUG_LOG = "/tmp/t2w_emergency.log"
with open(DEBUG_LOG, "a") as f:
    f.write(f"[{time.ctime()}] Sidecar process starting. PID: {os.getpid()}\n")

# 🔧 Mandatory Redirect: Stop libraries from polluting stdout
_original_stdout = sys.stdout
_original_stderr = sys.stderr

class StderrRedirector:
    def write(self, text):
        _original_stderr.write(text)
        with open(DEBUG_LOG, "a") as f: f.write(f"[STDERR] {text}")
    def flush(self):
        _original_stderr.flush()

sys.stdout = StderrRedirector()

# ==============================================================================
# 🐵 NUCLEAR MONKEY PATCH: Force CPU/MPS for Mac (Aggressive Mocking)
# ==============================================================================
import torch
import types

def force_cpu_override():
    """Aggressively mocks torch.cuda to force CPU execution on Mac."""
    if torch.cuda.is_available():
        return

    print(f"[T2W-PY] ⚠️  CUDA not found. applying NUCLEAR CPU patch...", file=sys.stderr)
    
    # 1. Force is_available to False (so well-behaved code takes CPU path)
    torch.cuda.is_available = lambda: False
    
    # 2. Mock device count/current device
    torch.cuda.device_count = lambda: 0
    torch.cuda.current_device = lambda: 0
    torch.cuda.set_device = lambda device: None
    
    # 3. Mock synchronization/memory (no-ops)
    torch.cuda.synchronize = lambda device=None: None
    torch.cuda.empty_cache = lambda: None
    torch.cuda.reset_peak_memory_stats = lambda device=None: None
    torch.cuda.memory_stats = lambda device=None: {}
    torch.cuda.mem_get_info = lambda device=None: (1024*1024*1024*8, 1024*1024*1024*16) # Fake free memory
    
    # 4. Patch Tensor.cuda() and Module.cuda() to move to CPU (or MPS if desired)
    #    NOTE: Using 'cpu' for maximum compatibility with operators lacking MPS support
    target_device = 'cpu' 
    
    def custom_cuda(self, device=None, non_blocking=False):
        return self.to(device=target_device, non_blocking=non_blocking)
    torch.Tensor.cuda = custom_cuda
    torch.nn.Module.cuda = custom_cuda
    
    # 5. Patch torch.device to redirect 'cuda' to 'cpu'
    _orig_device = torch.device
    def custom_device(device_str, *args, **kwargs):
        if isinstance(device_str, str) and "cuda" in device_str:
            return _orig_device(target_device)
        return _orig_device(device_str, *args, **kwargs)
    torch.device = custom_device
    
    # 6. Patch torch.amp.autocast (disable it or map cuda to cpu)
    #    CPU autocast uses 'cpu', not 'cuda'
    _orig_autocast = torch.amp.autocast
    class CustomAutocast(_orig_autocast):
        def __init__(self, device_type, dtype=None, enabled=True, cache_enabled=None):
            if device_type == 'cuda':
                device_type = 'cpu'
                # bfloat16 is better for CPU autocast if supported, otherwise disable
                if dtype == torch.float16: 
                    dtype = torch.bfloat16 
                    enabled = False # float16 on CPU is often slow/unsupported
            super().__init__(device_type, dtype, enabled, cache_enabled)
    torch.amp.autocast = CustomAutocast

    # 7. Mock CUDAGraph (used in flashcosyvoice)
    class MockCUDAGraph:
        def capture_begin(self, pool=None): pass
        def capture_end(self): pass
        def replay(self): pass
        def reset(self): pass
    torch.cuda.CUDAGraph = MockCUDAGraph
    torch.cuda.graph = lambda graph, pool=None, stream=None: torch.cuda.stream(stream) # Dummy context manager

force_cpu_override()
# ==============================================================================

import json
import socket
import struct
import numpy as np
import traceback

# Restore stdout for JSON protocol
sys.stdout = _original_stdout

def log(msg):
    full_msg = f"[T2W-PY] {msg}"
    print(full_msg, file=sys.stderr, flush=True)
    with open(DEBUG_LOG, "a") as f: f.write(full_msg + "\n")

class Token2WavService:
    def __init__(self):
        self.token2wav = None
        self.stream_cache = None
        self.hift_cache = None
        self.ref_audio_path = None
        self.initialized = False
        self.bridge_socket = None
        self.bridge_addr = ("127.0.0.1", 18099)
        
    def _send_to_bridge(self, data: bytes):
        try:
            if self.bridge_socket is None:
                # Retry logic for initial connection
                for i in range(5):
                    try:
                        self.bridge_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        self.bridge_socket.settimeout(5.0)
                        self.bridge_socket.connect(self.bridge_addr)
                        log(f"Connected to Bridge TCP at {self.bridge_addr}")
                        break
                    except Exception as e:
                        log(f"Connection attempt {i+1}/5 failed: {e}")
                        time.sleep(1.0)
                        self.bridge_socket = None
                
                if self.bridge_socket is None:
                    log("CRITICAL: Failed to connect to Bridge after retries.")
                    return

            header = struct.pack('<I', len(data))
            self.bridge_socket.sendall(header + data)
            
            # 🔧 Bidirectional Sync: Wait for ACK (PONG) if we sent PING
            if data == b"PING":
                log("Sent PING. Waiting for PONG...")
                self.bridge_socket.settimeout(5.0) # Increased timeout
                try:
                    # Expect 4 bytes header + 4 bytes "PONG"
                    pong_header = self.bridge_socket.recv(4)
                    if not pong_header:
                        log("CRITICAL: Connection closed by Bridge during PONG wait")
                        return
                    
                    pong_len = struct.unpack('<I', pong_header)[0]
                    pong_data = self.bridge_socket.recv(pong_len)
                    
                    if pong_data == b"PONG":
                         log("✅ RECEIVED PONG. Bidirectional sync established.")
                    else:
                         log(f"❌ RECEIVED UNKNOWN DATA instead of PONG: {pong_data}")

                except Exception as e:
                    log(f"PING/PONG handshake FAILED: {e}")
                    # Don't proceed if handshake fails, it means the pipe is broken
                    self.bridge_socket = None
                    
        except Exception as e:
            log(f"TCP Send Failed: {e}")
            self.bridge_socket = None

    def init(self, model_dir, device="cuda:0", float16=True, n_timesteps=5):
        try:
            # Device override handled by monkey patch globally
            log(f"Loading weights from {model_dir}...")
            from stepaudio2 import Token2wav
            
            # CPU usually requires float32 to avoid "LayerNormKernelImpl" not implemented for Half
            if not torch.cuda.is_available():
                float16 = False
                log("Force-disabling float16 for CPU execution")

            self.token2wav = Token2wav(model_dir, float16=float16, n_timesteps=n_timesteps)
            
            self.initialized = True
            log("Model Initialized. Sending PING...")
            self._send_to_bridge(b"PING")
            return {"status": "ok"}
        except Exception as e:
            log(f"CRITICAL INIT ERROR: {e}")
            traceback.print_exc(file=sys.stderr)
            return {"status": "error", "message": str(e)}

    def set_ref_audio(self, ref_audio_path):
        try:
            self.ref_audio_path = ref_audio_path
            self.stream_cache, self.hift_cache = self.token2wav.set_stream_cache(ref_audio_path)
            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def process(self, tokens, last_chunk, output_path):
        if not self.initialized: 
            log("ERROR: Process called but model not initialized")
            return {"status": "error", "message": "Not initialized"}
        try:
            start_t = time.time()
            log(f"--- [Mouth] Processing segment ---")
            log(f"--- [Mouth] Tokens: {len(tokens)}, LastChunk: {last_chunk}")
            
            self.token2wav.stream_cache = self.stream_cache
            self.token2wav.hift_cache_dict = self.hift_cache
            
            # The Brain Call
            wav_data = self.token2wav.stream(
                generated_speech_tokens=tokens, 
                prompt_wav=self.ref_audio_path, 
                last_chunk=last_chunk, 
                return_waveform=True
            )
            
            self.stream_cache = self.token2wav.stream_cache
            self.hift_cache = self.token2wav.hift_cache_dict
            
            if wav_data is not None and len(wav_data) > 0:
                log(f"--- [Brain] Audio generated. Shape: {wav_data.shape}")
                if len(wav_data.shape) > 1: wav_data = wav_data.squeeze()
                
                # Format for the pipe
                pcm_data = (np.clip(wav_data, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                log(f"--- [Pipe] Sending {len(pcm_data)} PCM bytes...")
                
                self._send_to_bridge(pcm_data)
                log(f"--- [Pipe] SUCCESS: Streamed in {int((time.time()-start_t)*1000)}ms")
            else:
                log("--- [Brain] No audio samples generated for this chunk.")
                
            if last_chunk:
                log("--- [Mouth] Turn complete. Sending EOT signal.")
                self._send_to_bridge(b"") # 0-length = EOT
            
            return {"status": "ok"}
        except Exception as e:
            log(f"--- [CRITICAL] Process failure: {e}")
            traceback.print_exc(file=sys.stderr)
            return {"status": "error", "message": str(e)}

def main():
    service = Token2WavService()
    log("Process loop starting.")
    print(json.dumps({"status": "ready"}), flush=True)
    while True:
        line = sys.stdin.readline()
        if not line: break
        try:
            cmd = json.loads(line)
            c = cmd.get("cmd")
            if c == "init": res = service.init(cmd["model_dir"], cmd.get("device","cuda:0"))
            elif c == "set_ref_audio": res = service.set_ref_audio(cmd["ref_audio_path"])
            elif c == "process": res = service.process(cmd["tokens"], cmd["last_chunk"], cmd["output_path"])
            elif c == "quit": break
            else: res = {"status": "error", "message": "unknown"}
            print(json.dumps(res), flush=True)
        except Exception as e:
            log(f"Fatal loop error: {e}")
            break

if __name__ == "__main__":
    main()
