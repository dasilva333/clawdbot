# Sidecar Voice Clone PoC Report

- Timestamp: `2026-02-12T23:04:37`
- Token source: `/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/llama.cpp-omni/tools/omni/output/round_000/tts_wav`
- Token chunks: `4`
- Sidecar script: `/Users/richardpinedo/Projects/minicpm/sidecar_voice_poc/mini_token2wav_sidecar.py`

## Results

### default_ref
- Status: `ok`
- Ref audio: `/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/cpp_server/assets/default_ref_audio.wav`
- Final wav: `/Users/richardpinedo/Projects/minicpm/sidecar_voice_poc/runs/20260212_230423/default_ref/default_ref.wav`
- Audio duration: `8.52s`
- Init wall: `3110.74ms`
- Set-ref wall: `565.65ms`
- First audio (from first process send): `447.12ms`
- Chunk processing wall: `2292.58ms`
- Profile end-to-end wall: `6527.16ms`
- RTF chunks-only: `0.269`
- RTF end-to-end: `0.766`
- Sidecar stderr log: `/Users/richardpinedo/Projects/minicpm/sidecar_voice_poc/runs/20260212_230423/default_ref/sidecar.stderr.log`

### elon_ref
- Status: `error`
- Ref audio: `/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/assets/voices/elon_musk_ref.wav`
- Final wav: `/Users/richardpinedo/Projects/minicpm/sidecar_voice_poc/runs/20260212_230423/elon_ref/elon_ref.wav`
- Audio duration: `1.48s`
- Init wall: `3021.15ms`
- Set-ref wall: `2420.14ms`
- First audio (from first process send): `532.83ms`
- Chunk processing wall: `577.05ms`
- Profile end-to-end wall: `6525.5ms`
- RTF chunks-only: `0.39`
- RTF end-to-end: `4.409`
- Sidecar stderr log: `/Users/richardpinedo/Projects/minicpm/sidecar_voice_poc/runs/20260212_230423/elon_ref/sidecar.stderr.log`
- Error chunk: `1`
- Error message: `The expanded size of the tensor (1000) must match the existing size (1070) at non-singleton dimension 2.  Target sizes: [2, 8, 1000, 128].  Tensor sizes: [2, 8, 1070, 128]`

