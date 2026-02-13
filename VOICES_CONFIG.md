# MiniCPM Voice Config Journal

Last updated: 2026-02-13

This file is our voice-cloning and voice-control research log for the current MiniCPM bridge stack.

## 1. Canonical Upstream References (MiniCPM-o-4_5 README)

Fetched reference file:
- `/tmp/MiniCPM-o-4_5-README.md`

Key anchors:
- Product claim (voice cloning + configurable voices): `/tmp/MiniCPM-o-4_5-README.md:32`
- "Simplex speech conversation with custom reference audio and character prompts": `/tmp/MiniCPM-o-4_5-README.md:970`
- "Click to show chat inference code.": `/tmp/MiniCPM-o-4_5-README.md:1167`
- "Click to show streaming inference code.": `/tmp/MiniCPM-o-4_5-README.md:1221`
- "Click to show custom voice conversation code.": `/tmp/MiniCPM-o-4_5-README.md:1471`
- "Click to show TTS code.": `/tmp/MiniCPM-o-4_5-README.md:1533`

Reference-audio usage in upstream examples:
- `assets/HT_ref_audio.wav`: `/tmp/MiniCPM-o-4_5-README.md:1178`, `/tmp/MiniCPM-o-4_5-README.md:1237`, `/tmp/MiniCPM-o-4_5-README.md:1439`, `/tmp/MiniCPM-o-4_5-README.md:1542`
- `assets/system_ref_audio.wav`: `/tmp/MiniCPM-o-4_5-README.md:1477`
- `assets/system_ref_audio_2.wav`: `/tmp/MiniCPM-o-4_5-README.md:1506`

## 2. Active Runtime Path in This Repo

Important: our live bridge path is `bridge_service.py`, not `cpp_server/minicpmo_cpp_http_server.py`.

Evidence:
- launcher executes bridge service directly: `demo/web_demo/WebRTC_Demo/start_bridge.sh:61`

Realtime voice call path:
1. OpenClaw plugin streams PCM chunks -> bridge prefill
   - `openclaw-minicpm-tts/src/provider.ts:149`
2. Plugin triggers decode on speech end
   - `openclaw-minicpm-tts/src/provider.ts:170`
   - `openclaw-minicpm-tts/src/voice/bridge.ts:23`
3. Bridge receives `/omni/streaming_prefill` + `/omni/decode`
   - `demo/web_demo/WebRTC_Demo/bridge_service.py:478`
   - `demo/web_demo/WebRTC_Demo/bridge_service.py:387`

## 3. Effective Knobs Today (What Is Actually Controllable)

### 3.1 Voice reference file (`voice_audio`)

- Stored as global `current_voice_path` in bridge:
  - `demo/web_demo/WebRTC_Demo/bridge_service.py:30`
- Used during session init payload:
  - `demo/web_demo/WebRTC_Demo/bridge_service.py:323`
- Can be switched by API command:
  - `demo/web_demo/WebRTC_Demo/bridge_service.py:341`
  - plugin command `/setvoice`: `openclaw-minicpm-tts/index.ts:212`

### 3.2 Prompt profile (`system_prompt_prefix` + `system_prompt_suffix`)

- Accepted by bridge init endpoint:
  - `demo/web_demo/WebRTC_Demo/bridge_service.py:291`
  - `demo/web_demo/WebRTC_Demo/bridge_service.py:320`
  - `demo/web_demo/WebRTC_Demo/bridge_service.py:321`
- Applied in C++ omni init/update:
  - `demo/web_demo/WebRTC_Demo/llama.cpp-omni/tools/omni/omni.cpp:3635`
  - `demo/web_demo/WebRTC_Demo/llama.cpp-omni/tools/server/server.cpp:6234`

### 3.3 Temperature

- Voice-session default in decode path currently fixed to `0.7`:
  - `demo/web_demo/WebRTC_Demo/bridge_service.py:394`
- Applied on init payload:
  - `demo/web_demo/WebRTC_Demo/bridge_service.py:322`

### 3.4 TTS "voice" parameter caveat

- `/v1/tts/synthesize` receives `voice`, but current synthesis flow does not map that value into a `voice_audio` lookup/update.
- Code path:
  - request model has `voice`: `demo/web_demo/WebRTC_Demo/bridge_service.py:101`
  - synth route maps `voice` into OpenAITTSRequest: `demo/web_demo/WebRTC_Demo/bridge_service.py:378`
  - `_collect_tts_pcm` / `_stream_tts_pcm` do not use `voice`:
    - `demo/web_demo/WebRTC_Demo/bridge_service.py:195`
    - `demo/web_demo/WebRTC_Demo/bridge_service.py:245`

## 4. Current Prompt Templates in Engine

Default simplex-en style in current C++:
- Prefix: `Clone the voice in the provided audio prompt.`
- Suffix: helpful-assistant + naturalness + English instruction
- Code:
  - `demo/web_demo/WebRTC_Demo/llama.cpp-omni/tools/omni/omni.cpp:3649`
  - `demo/web_demo/WebRTC_Demo/llama.cpp-omni/tools/omni/omni.cpp:3664`

Language switch also reassigns prompt templates:
- `demo/web_demo/WebRTC_Demo/llama.cpp-omni/tools/omni/omni.cpp:4308`

System prompt prefill with reference audio embedding:
- `demo/web_demo/WebRTC_Demo/llama.cpp-omni/tools/omni/omni.cpp:8903`
- `demo/web_demo/WebRTC_Demo/llama.cpp-omni/tools/omni/omni.cpp:8952`

## 5. Reference Audio Inventory and Measured Properties

Local voice files:
- `demo/web_demo/WebRTC_Demo/assets/voices/jarvis.wav` (16k mono s16le, ~2.968s)
- `demo/web_demo/WebRTC_Demo/assets/voices/trump.wav` (16k mono s16le, ~1.734s)
- `demo/web_demo/WebRTC_Demo/cpp_server/assets/default_ref_audio.wav` (16k mono s16le, ~2.968s)

Upstream sample refs measured:
- `/private/tmp/elon_ref.wav` (16k mono s16le, ~16.843s)
- `/private/tmp/minicpm_ref.wav` (16k mono s16le, ~6.016s)

Observation:
- Local files are format-correct, but notably shorter than upstream roleplay references.

## 6. Recon Conclusions (No Code Change Section)

1. The knobs exist at API level, but operationally they are sparse in day-to-day voice calls.
2. Voice calls now run audio-native (no text injection in voice decode path), which fixed parrot behavior.
3. Voice identity strength likely depends heavily on reference-audio quality/length and system profile content.
4. The plugin supports `/setvoice`, but there is no richer per-guild or per-agent prompt/profile control exposed in normal call flow.
5. There is no built-in STT/transcription HTTP endpoint in current omni server route table.
6. The TTS `voice` request field is currently not wired into reference-audio switching in the synthesis helper path.

## 7. Intent / Goal Catalog

Target outcomes for this project:
1. Reliable male/female timbre switching via reference WAV.
2. Reproducible roleplay profile behavior from system prompt text.
3. Voice call flow with practical knobs:
   - `voice_id`
   - `system_prompt_prefix`
   - `system_prompt_suffix`
   - `temperature`
4. Predictable behavior parity between:
   - `/v1/tts/synthesize`
   - realtime `/omni/streaming_prefill` + `/omni/decode`
5. Stable long-answer playback without premature cutoff (already improved by timeout tuning in bridge).

## 8. Candidate Test Voice Profiles

1. Jarvis (Paul Bettany) - sophisticated dry British cadence
2. Elon Musk - direct, technical, sparse fillers
3. Donald Trump - high-energy superlative rhythm
4. Morgan Freeman - deep, slow, authoritative
5. Snoop Dogg - relaxed rhythmic flow
6. Gordon Ramsay - energetic, emphatic
7. Arnold Schwarzenegger - accent + punchy delivery
8. David Attenborough - calm documentary cadence

## 9. Sidecar Voice Cloning PoC (Standalone Benchmark)

PoC location:
- `/Users/richardpinedo/Projects/minicpm/sidecar_voice_poc`

Successful reference clone run:
- `/Users/richardpinedo/Projects/minicpm/sidecar_voice_poc/runs/20260212_230537/report.md`
- `/Users/richardpinedo/Projects/minicpm/sidecar_voice_poc/runs/20260212_230537/summary.json`

Failure case with long reference clip:
- `/Users/richardpinedo/Projects/minicpm/sidecar_voice_poc/runs/20260212_230452/report.md`
- `/Users/richardpinedo/Projects/minicpm/sidecar_voice_poc/runs/20260212_230452/summary.json`

Measured behavior from successful run:
1. Sidecar init wall time: ~3.0s
2. `set_ref_audio` wall time: ~0.55s
3. First audio availability after first process send: ~0.44s
4. Chunk processing wall time: ~10.7s for ~43.8s synthesized output (pre-tokenized input)
5. Clone quality is clearly speaker-conditioned (Elon-like timbre audible), but still uncanny.

Known blockers:
1. Long reference clip can fail early with stream-cache shape mismatch (`1000` vs `1070`).
2. Current PoC path is CPU-only due upstream CUDA assumptions in `stepaudio2`.
3. This PoC is not integrated into live bridge flow; it proves feasibility, not production readiness.

## 10. Recommended Direction (C: Staged Hybrid Plan)

1. Keep current production pipeline as default and continue latency optimization there first.
2. Keep sidecar voice-clone work isolated as an opt-in experiment until it has stable long-ref behavior.
3. Define merge gate for sidecar integration:
   - no cache-shape crash on long refs
   - sub-1s first-audio in streamed mode
   - bounded quality/perf impact versus current ~8s baseline
4. Integrate cloning only after the above gate passes.
