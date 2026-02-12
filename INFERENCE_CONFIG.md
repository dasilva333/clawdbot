# MiniCPM-o Inference Configuration (Debug Mode)

## 1. Target Voice (Voice Cloning Reference)
The model uses a reference audio file to determine the output voice timbre.

- **Config File:** `cpp_server/minicpmo_cpp_http_server.py`
- **Current Logic:** 
    1. Checks `REF_AUDIO` environment variable.
    2. Fallback: `{LLAMACPP_ROOT}/tools/omni/assets/default_ref_audio.wav`
- **Location in Code:** `FIXED_TIMBRE_PATH` variable (approx. line 55 and inside `lifespan` function).

## 2. System Prompt Templates
These templates wrap the reference audio and define the model's persona/behavior.

- **Config File:** `llama.cpp-omni/tools/omni/omni.cpp`
- **Function:** `omni_init(...)` (approx. line 3640)
- **Simplex Mode Template (Current):**
    - **Prefix:** `<|im_start|>system\n模仿音频样本的音色并生成新的内容。\n<|audio_start|>`
    - **Suffix:** `<|audio_end|>你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。<|im_end|>\n<|im_start|>user\n`
- **Duplex Mode Template:**
    - **Prefix:** `<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>`
    - **Suffix:** `<|audio_end|><|im_end|>\n`

## 3. Language Switch Configuration
The templates can be swapped between Chinese ("zh") and English ("en").

- **Switch Logic:** `omni_set_language(...)` in `omni.cpp` (approx. line 4050).
- **Current Status:** Determined by the `language` parameter sent in the `/omni/init_sys_prompt` request. In the Python bridge (`minicpmo_cpp_http_server.py`), the default is currently set to **"zh"** (Chinese).

## 4. Per-Agent Voice Customization (Future Project)
Based on the current architecture, enabling different voices for different agents is feasible with minor code changes:

- **OpenClaw Side:** The `textToSpeech` function already handles an `overrides` object. We can pass a `voice` or `voiceId` from the agent's configuration.
- **Bridge Side:** The `OpenAITTSRequest` schema in `bridge_service.py` already has a `voice` field.
- **Implementation Path:** 
    1. Store multiple reference audio files (e.g., `jarvis.wav`, `snoop.wav`) in the `assets` folder.
    2. Update `_do_tts` in the bridge to look up the `voice_id` and use the corresponding file for the `voice_audio` parameter when calling the C++ server.

## 5. Potential Character Voice Profiles
Candidate voices for testing Cadence and Timbre extraction:
1. **Jarvis (Paul Bettany)** - Sophisticated, dry, British AI (The Daily Driver).
2. **Elon Musk** - Distinct pauses, tech-focused cadence.
3. **Donald Trump** - High-energy superlatives, very distinct rhythm.
4. **Morgan Freeman** - Deep resonance, slow and authoritative.
5. **Snoop Dogg** - Relaxed, rhythmic, "chill and friendly" fit.
6. **Gordon Ramsay** - High energy, aggressive inflections.
7. **Arnold Schwarzenegger** - Iconic accent and punchy delivery.
8. **David Attenborough** - Whispered, dramatic nature-documentary style.

## 6. Translation of Chinese Prompt
The default Chinese prompt translates to:
- **Prefix:** "Imitate the timbre of the audio sample and generate new content."
- **Suffix:** "Your task is to act as an assistant using this voice mode. Please reply to user's questions seriously and with high quality. Please chat with the user in a highly natural way. You are a helpful assistant developed by ModelBest: MiniCPM-Omni (literally 'Small Steel Cannon')."
