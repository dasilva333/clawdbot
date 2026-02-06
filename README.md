# openclaw-minicpm-tts

Local TTS plugin for OpenClaw using MiniCPM-o 4.5 GGUF forced token decoding via llama.cpp-omni.

## Features

- Voice cloning from reference WAV files (5-15s of clear speech)
- Opus output for Telegram voice message compatibility
- Named voice aliases with server-side resolution
- Gateway methods for direct API access (synthesize, health, voices)
- 10.4 GB VRAM (Q5_K_M quantization) vs 18 GB bfloat16

## Architecture

```
OpenClaw ──POST──▶ GGUF TTS API (:8087) ──POST──▶ llama.cpp-omni (:8085)
                   FastAPI proxy                   Forced token decoding
                   WAV concat + opus               tokenize → hidden states → TTS vocoder
                   Voice resolution                WAV chunks on disk
```

~31s per sentence (TTS/T2W vocoder pipeline bottleneck).

## Configuration

### Core TTS Provider

The plugin integrates as a core TTS provider (`"minicpm"`) in OpenClaw's TTS engine.

**openclaw.json:**
```json
{
  "messages": {
    "tts": {
      "provider": "minicpm",
      "minicpm": {
        "endpoint": "http://192.168.1.119:8087",
        "defaultVoice": "default",
        "timeoutMs": 120000,
        "voices": {
          "gekko": "/home/comfy/minicpm-audio-api/voice_refs/gekko_sample.wav"
        }
      }
    }
  }
}
```

**Telegram commands:**
```
/tts provider minicpm    # Switch to local GGUF TTS
/tts status              # Verify provider active
/tts audio Hello world   # Synthesize and send as voice message
```

### Extension Plugin

**openclaw.json:**
```json
{
  "plugins": {
    "minicpm-tts": {
      "enabled": true,
      "endpoint": "http://192.168.1.119:8087",
      "defaultVoice": "default",
      "format": "opus",
      "timeoutMs": 120000
    }
  }
}
```

## Gateway Methods

### `minicpm.synthesize`

Returns base64-encoded audio.

```json
{
  "text": "Hello, how are you today?",
  "voice": "default"
}
```

Response:
```json
{
  "success": true,
  "format": "opus",
  "mimeType": "audio/ogg",
  "audioBase64": "T2dnUwAC...",
  "audioSize": 2271
}
```

### `minicpm.health`

```json
{
  "healthy": true,
  "endpoint": "http://192.168.1.119:8087",
  "format": "opus",
  "defaultVoice": "default"
}
```

### `minicpm.voices`

Lists available voice references and their server-side paths.

## Prerequisites

- GGUF TTS API running on target host (FastAPI proxy on port 8087 + llama.cpp-omni on port 8085)
- MiniCPM-o-4_5-Q5_K_M.gguf model loaded
- ffmpeg with libopus on the API host (for opus output)

## File Structure

```
├── openclaw.plugin.json   # Plugin manifest and config schema
├── index.ts               # Plugin entry point (gateway methods + service lifecycle)
├── package.json           # Plugin metadata
├── README.md              # This file
└── src/
    ├── config.ts          # Configuration types and defaults
    └── provider.ts        # GGUF TTS API client (synthesize, health, voices)
```

## License

MIT
