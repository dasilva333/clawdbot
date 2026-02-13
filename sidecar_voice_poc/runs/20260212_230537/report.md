# Sidecar Voice Clone PoC Report

- Timestamp: `2026-02-12T23:06:07`
- Token source: `/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/llama.cpp-omni/tools/omni/output/round_000/tts_wav`
- Token chunks: `16`
- Sidecar script: `/Users/richardpinedo/Projects/minicpm/sidecar_voice_poc/mini_token2wav_sidecar.py`

## Results

### default_ref
- Status: `ok`
- Ref audio: `/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/cpp_server/assets/default_ref_audio.wav`
- Final wav: `/Users/richardpinedo/Projects/minicpm/sidecar_voice_poc/runs/20260212_230537/default_ref/default_ref.wav`
- Audio duration: `43.8s`
- Init wall: `2976.19ms`
- Set-ref wall: `551.2ms`
- First audio (from first process send): `468.44ms`
- Chunk processing wall: `10758.49ms`
- Profile end-to-end wall: `14856.27ms`
- RTF chunks-only: `0.246`
- RTF end-to-end: `0.339`
- Sidecar stderr log: `/Users/richardpinedo/Projects/minicpm/sidecar_voice_poc/runs/20260212_230537/default_ref/sidecar.stderr.log`

### elon_ref
- Status: `ok`
- Ref audio: `/Users/richardpinedo/Projects/minicpm/sidecar_voice_poc/assets/elon_ref_3s.wav`
- Final wav: `/Users/richardpinedo/Projects/minicpm/sidecar_voice_poc/runs/20260212_230537/elon_ref/elon_ref.wav`
- Audio duration: `43.8s`
- Init wall: `3047.59ms`
- Set-ref wall: `553.22ms`
- First audio (from first process send): `443.29ms`
- Chunk processing wall: `10726.82ms`
- Profile end-to-end wall: `14896.4ms`
- RTF chunks-only: `0.245`
- RTF end-to-end: `0.34`
- Sidecar stderr log: `/Users/richardpinedo/Projects/minicpm/sidecar_voice_poc/runs/20260212_230537/elon_ref/sidecar.stderr.log`

