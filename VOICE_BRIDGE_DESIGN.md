# Discord Real-Time Voice Bridge (Omni Design)

This document outlines the architectural design for a low-latency, "raw audio" bidirectional voice bridge between Discord and the MiniCPM-o 4.5 inference stack.

## 1. Primary Objective
Establish a real-time voice connection where the AI agent acts as a live participant in a Discord voice channel, bypassing traditional STT/TTS text conversion layers to achieve "human-like" latency and emotional resonance.

## 2. Key Projects & Libraries

### Orchestration & Control
- **OpenClaw**: The central gateway and control plane.
- **MiniCPM-o 4.5**: The "Omni" multimodal model capable of end-to-end speech processing.

### Voice Transport (Discord)
- **@discordjs/voice**: The core Node.js implementation of the Discord Voice API.
- **discord-voice (avatarneil/discord-voice)**: Community skill reference for VAD, join/leave logic, and barge-in handling.

### Audio Processing
- **Prism-Media**: For real-time format conversion (Opus <-> PCM).
- **voice-agent (ricardotrevisan/voice-agent)**: Reference for "Audio-First" workflows and "Silent Delivery" (suppressing text).
- **FFmpeg**: Fallback for complex transcoding.

## 3. High-Level Architecture (The "Raw Path")

### Standalone Bridge Service (Node.js)
A robust Node.js service that terminates the Discord voice transport and interfaces with OpenClaw's control plane via WebSocket.

#### Inbound Flow (User -> AI)
1. **Discord UDP Stream**: Receive Opus audio packets via `@discordjs/voice` `VoiceReceiver`.
2. **Turn Segmentation**: Use `EndBehaviorType.AfterSilence` (tuned to ~500ms-1000ms) to identify user utterances.
3. **Format Normalization**: `prism.opus.Decoder` converts Opus to **PCM s16le (48k, 2 channels)**.
4. **Sliding Buffer**: Accumulate short PCM chunks (200ms - 500ms).
5. **Streaming Prefill**: POST raw chunks to the MiniCPM-o bridge `/omni/streaming_prefill` endpoint.

#### Outbound Flow (AI -> User)
1. **Generated Chunks**: MiniCPM-o produces audio chunks in response.
2. **Bridge Collection**: The Python bridge merges chunks.
3. **Audio Resource**: `prism.FFmpeg` decodes the merged buffer back to raw PCM for Discord.
4. **Playback**: Push to `@discordjs/voice` `AudioPlayer`.

## 4. Integration with OpenClaw Gateway
The Voice Bridge communicates with OpenClaw using the Gateway WebSocket API (`ws://127.0.0.1:18789`):
- **chat.send Pattern**: Emulate OpenClaw's "Talk Mode" by sending event markers or transcriptions (if the model generates them) to the session.
- **Session Management**: Each (Guild, Channel) maps to a stable OpenClaw session ID.
- **Interruption Signal**: Detect Discord `speaking` events and immediately trigger the Gateway's `omni/break` RPC method to stop current AI output.

## 5. Execution Roadmap

### Phase 1: Transport & Connectivity
- Register a new Discord Bot with `GuildVoiceStates` intents.
- Use `@discordjs/voice` to join a specific `voiceChannelId`.
- Verify the bot can maintain a stable UDP connection in the foreground.

### Phase 2: The "Listen" Loop
- Implement the `VoiceReceiver` and `prism` decoder.
- Buffer PCM data and log successful ingestion (Verify we are getting clean 48k PCM).

### Phase 3: The "Speak" Loop (S2S)
- Bridge the PCM buffer to the local Python server's prefill endpoint.
- Receive generated chunks and push to Discord's `AudioPlayer`.
- Verify end-to-end "Raw Path" latency.

## 6. Design Principles
- **S2S (Speech-to-Speech)**: Eliminate the STT -> LLM -> TTS bottleneck.
- **UDP Separation**: Respect Discord's separate media plane (UDP) vs control plane (Gateway).
- **Silent Delivery**: Audio is the primary interface; text is ephemeral or secondary.
- **Low-Latency Buffering**: Favor small, frequent data transfers over large batches.

---
*Status: Design Finalized / Ready for Phase 1*
