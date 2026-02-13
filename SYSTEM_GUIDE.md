# MiniCPM-o 4.5 Open Pipe System Guide

This guide outlines the architecture, management, and troubleshooting of the three-tier voice system.

## 🏗️ Architecture (The Three Pieces)

### 1. The Engine (C++)
*   **Location**: `/Users/richardpinedo/Projects/minicpm`
*   **Core Logic**: `demo/web_demo/WebRTC_Demo/llama.cpp-omni/tools/server/server.cpp` (HTTP API) and `tools/omni/omni.cpp` (Sampling/TTS).
*   **TCP Pipes**:
    *   **Port 18099**: Outbound raw PCM (Engine -> Bridge).
    *   **Port 18100**: Inbound raw PCM (Bridge -> Engine).
*   **Build**: `cd demo/web_demo/WebRTC_Demo && ./oneclick.sh`
*   **Logs**: `/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/.logs/cpp_server.log`

### 2. The Bridge (Python)
*   **Location**: `/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/bridge_service.py`
*   **Function**: Translates between Discord's world (Opus/PCM) and the Engine's TCP pipes. Handles VAD (Voice Activity Detection).
*   **HTTP Port**: `9060`
*   **Start**: `cd /Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo && ./run_bridge.sh`
*   **Logs**: Console output (standard redirect if backgrounded).

### 3. The Plugin (TypeScript/OpenClaw)
*   **Location**: `/Users/richardpinedo/Projects/openclaw-minicpm-tts`
*   **Function**: Discord Command handler (`/join`, `/leave`) and audio I/O via a dedicated `discord.js` client.
*   **Deploy**: `npm run deploy` (moves to `~/.openclaw/extensions/minicpm-tts/`).
*   **Logs**: `/tmp/openclaw/*.log` (Standard OpenClaw gateway logs).

---

## 🚀 Lifecycle Management

### Start Sequence
1.  **C++ Server**: `cd /Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo && ./run_demo.sh` (starts C++ and Bridge).
2.  **OpenClaw**: Ensure the gateway is running. The plugin loads on startup.

### Stop Sequence
*   **C++**: `pkill -f minicpm` or `kill $(cat .pids/cpp_server.pid)`
*   **Bridge**: `pkill -f bridge_service.py`
*   **OpenClaw**: `openclaw gateway stop`

### Status Check
*   **Ports**: `lsof -i :18099,18100,9060`
*   **PIDs**: `ls /Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/.pids/`

---

## 🛠️ Configuration & Tuning

### Dynamic Prompts/Temp
Use the bridge endpoint to update the engine live:
```bash
curl -X POST http://127.0.0.1:9060/omni/init_sys_prompt \
-H "Content-Type: application/json" \
-d '{"temperature": 0.8, "system_prompt_suffix": "..."}'
```

### Reference Voices
Reference WAVs should be provided to the bridge during injection/prefill if cloning is desired. Currently defaults to the model's base English voice.
