# Sidecar Voice Clone PoC Report

- Timestamp: `2026-02-12T23:05:15`
- Token source: `/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/llama.cpp-omni/tools/omni/output/round_000/tts_wav`
- Token chunks: `16`
- Sidecar script: `/Users/richardpinedo/Projects/minicpm/sidecar_voice_poc/mini_token2wav_sidecar.py`

## Results

### default_ref
- Status: `ok`
- Ref audio: `/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/cpp_server/assets/default_ref_audio.wav`
- Final wav: `/Users/richardpinedo/Projects/minicpm/sidecar_voice_poc/runs/20260212_230452/default_ref/default_ref.wav`
- Audio duration: `43.8s`
- Init wall: `3027.15ms`
- Set-ref wall: `553.03ms`
- First audio (from first process send): `442.61ms`
- Chunk processing wall: `10830.89ms`
- Profile end-to-end wall: `14985.82ms`
- RTF chunks-only: `0.247`
- RTF end-to-end: `0.342`
- Sidecar stderr log: `/Users/richardpinedo/Projects/minicpm/sidecar_voice_poc/runs/20260212_230452/default_ref/sidecar.stderr.log`

### elon_ref
- Status: `error`
- Ref audio: `/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/assets/voices/elon_musk_ref.wav`
- Final wav: `/Users/richardpinedo/Projects/minicpm/sidecar_voice_poc/runs/20260212_230452/elon_ref/elon_ref.wav`
- Audio duration: `1.48s`
- Init wall: `3700.74ms`
- Set-ref wall: `2441.54ms`
- First audio (from first process send): `549.57ms`
- Chunk processing wall: `593.64ms`
- Profile end-to-end wall: `7242.85ms`
- RTF chunks-only: `0.401`
- RTF end-to-end: `4.894`
- Sidecar stderr log: `/Users/richardpinedo/Projects/minicpm/sidecar_voice_poc/runs/20260212_230452/elon_ref/sidecar.stderr.log`
- Error chunk: `1`
- Error message: `The expanded size of the tensor (1000) must match the existing size (1070) at non-singleton dimension 2.  Target sizes: [2, 8, 1000, 128].  Tensor sizes: [2, 8, 1070, 128]`

