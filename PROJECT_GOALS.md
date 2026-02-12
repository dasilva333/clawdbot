# MiniCPM-o Bridge Project Goals

This document serves as the source of truth for the project's architectural direction and functional requirements. All future development must align with these four core goals.

## 1. No Messing with WAV Files (The "Open Pipe" Architecture)
- **Problem**: File-based batching (saving WAVs to disk) introduced massive latency and context corruption.
- **Requirement**: Zero disk I/O for audio. All audio data must flow through high-speed TCP sockets.
- **Implementation**:
    - **Inbound**: Raw PCM bytes from the user are streamed directly to the C++ engine via TCP (Port 18100).
    - **Outbound**: Raw PCM bytes from the C++ engine are streamed directly to the Bridge via TCP (Port 18099).

## 2. TTS On Demand
- **Problem**: The system needs a way to generate specific audio versions of text prompts.
- **Requirement**: An OpenAI-compatible `/audio/speech` endpoint that returns playable audio within 25 seconds.
- **Implementation**: 
    - The Bridge triggers the engine's `omni_inject_text` path and captures the resulting TCP stream. 
    - **Protocol**: The engine must send a native **End of Turn (EOT)** signal (length=0 chunk) immediately after the final audio token. 
    - **Fallback**: The Bridge uses a 2-second silence timeout ONLY as a fallback if the engine fails to send the EOT. 
    - Returns high-quality 24kHz Mono WAV files.

## 3. Bidirectional Voice Calls
- **Problem**: Discord real-time communication requires sub-second latency for a natural feel.
- **Requirement**: Full-duplex or high-responsiveness simplex voice interaction.
- **Implementation**:
    - Integrated turn-taking logic that correctly manages `<|audio_start|>` and `<|audio_end|>` tags.
    - Automatic state management to ensure Turn 2+ does not suffer from KV cache drift or tag bloat.

## 4. Adjustable Voices (Persona/Voice Cloning)
- **Problem**: Users want specific AI identities (e.g., Jarvis).
- **Requirement**: Ability to swap the reference audio file used for the model's timbre without rebuilding the core engine.
- **Technical Details**: See `VOICES_CONFIG.md` for implementation details on voice lookup and profile candidate lists.
- **Implementation**:
    - Hardcoded or dynamically configured paths to high-quality reference samples in `assets/voices/`.
    - Engine-level persona instructions hardcoded in the system prompt (e.g., "Always respond in English").
