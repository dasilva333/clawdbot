# Sidecar Voice PoC

Standalone proof-of-concept to benchmark the Python Token2Wav sidecar without touching Discord/OpenClaw/live bridge routing.

## What It Tests

- Uses real semantic token chunks from C++ output:
  - `demo/web_demo/WebRTC_Demo/llama.cpp-omni/tools/omni/output/round_000/tts_wav/audio_tokens_chunk_*.txt`
- Runs a minimal sidecar process over JSON lines (stdin/stdout)
- Synthesizes the same tokens twice:
  - default reference voice
  - Elon reference voice
- Emits:
  - output WAV files
  - per-profile latency/RTF metrics
  - JSON + Markdown report

## Run

```bash
cd /Users/richardpinedo/Projects/minicpm
./demo/web_demo/WebRTC_Demo/venv/bin/python3 sidecar_voice_poc/run_benchmark.py
```

Optional fast run:

```bash
./demo/web_demo/WebRTC_Demo/venv/bin/python3 sidecar_voice_poc/run_benchmark.py --max-chunks 4
```

## Outputs

Per run, files are written under:

- `sidecar_voice_poc/runs/<timestamp>/summary.json`
- `sidecar_voice_poc/runs/<timestamp>/report.md`
- `sidecar_voice_poc/runs/<timestamp>/default_ref/default_ref.wav`
- `sidecar_voice_poc/runs/<timestamp>/elon_ref/elon_ref.wav`

