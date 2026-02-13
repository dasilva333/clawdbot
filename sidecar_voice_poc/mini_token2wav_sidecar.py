#!/usr/bin/env python3
"""
Minimal standalone Token2Wav sidecar for PoC benchmarking.

Protocol:
- stdin: one JSON object per line
- stdout: one JSON object per line

Commands:
- {"cmd":"init","model_dir":"...","n_timesteps":5,"float16":false}
- {"cmd":"set_ref_audio","ref_audio_path":"..."}
- {"cmd":"process","tokens":[...],"last_chunk":false,"output_path":"..."}
- {"cmd":"quit"}
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def _patch_torch_for_cpu() -> None:
    """Force cpu behavior when CUDA is unavailable."""
    if torch.cuda.is_available():
        return

    torch.cuda.is_available = lambda: False  # type: ignore[assignment]
    torch.cuda.device_count = lambda: 0  # type: ignore[assignment]
    torch.cuda.current_device = lambda: 0  # type: ignore[assignment]
    torch.cuda.set_device = lambda _device: None  # type: ignore[assignment]
    torch.cuda.synchronize = lambda _device=None: None  # type: ignore[assignment]
    torch.cuda.empty_cache = lambda: None  # type: ignore[assignment]

    def _normalize_device_arg(device):  # noqa: ANN001
        if isinstance(device, str) and "cuda" in device.lower():
            return "cpu"
        return device

    def _tensor_cuda(self, device=None, non_blocking=False):  # noqa: ANN001
        return self.to(device="cpu", non_blocking=non_blocking)

    torch.Tensor.cuda = _tensor_cuda  # type: ignore[assignment]
    torch.nn.Module.cuda = _tensor_cuda  # type: ignore[assignment]

    # Some upstream code uses creators with device='cuda' directly.
    _orig_tensor = torch.tensor
    _orig_zeros = torch.zeros
    _orig_ones = torch.ones
    _orig_empty = torch.empty
    _orig_full = torch.full
    _orig_randn = torch.randn

    def _tensor_wrapper(*args, **kwargs):  # noqa: ANN001
        if "device" in kwargs:
            kwargs["device"] = _normalize_device_arg(kwargs["device"])
        return _orig_tensor(*args, **kwargs)

    def _zeros_wrapper(*args, **kwargs):  # noqa: ANN001
        if "device" in kwargs:
            kwargs["device"] = _normalize_device_arg(kwargs["device"])
        return _orig_zeros(*args, **kwargs)

    def _ones_wrapper(*args, **kwargs):  # noqa: ANN001
        if "device" in kwargs:
            kwargs["device"] = _normalize_device_arg(kwargs["device"])
        return _orig_ones(*args, **kwargs)

    def _empty_wrapper(*args, **kwargs):  # noqa: ANN001
        if "device" in kwargs:
            kwargs["device"] = _normalize_device_arg(kwargs["device"])
        return _orig_empty(*args, **kwargs)

    def _full_wrapper(*args, **kwargs):  # noqa: ANN001
        if "device" in kwargs:
            kwargs["device"] = _normalize_device_arg(kwargs["device"])
        return _orig_full(*args, **kwargs)

    def _randn_wrapper(*args, **kwargs):  # noqa: ANN001
        if "device" in kwargs:
            kwargs["device"] = _normalize_device_arg(kwargs["device"])
        return _orig_randn(*args, **kwargs)

    torch.tensor = _tensor_wrapper  # type: ignore[assignment]
    torch.zeros = _zeros_wrapper  # type: ignore[assignment]
    torch.ones = _ones_wrapper  # type: ignore[assignment]
    torch.empty = _empty_wrapper  # type: ignore[assignment]
    torch.full = _full_wrapper  # type: ignore[assignment]
    torch.randn = _randn_wrapper  # type: ignore[assignment]

    _orig_device = torch.device

    def _device(device_str, *args, **kwargs):  # noqa: ANN001
        if isinstance(device_str, str) and "cuda" in device_str.lower():
            return _orig_device("cpu")
        return _orig_device(device_str, *args, **kwargs)

    torch.device = _device  # type: ignore[assignment]


def _patch_torchaudio_load_fallback() -> None:
    """
    Patch torchaudio.load to fallback to soundfile when torchcodec is unavailable.
    This avoids hard-failing on Python/torchaudio combos without torchcodec wheels.
    """
    import torchaudio

    orig_load = torchaudio.load

    def safe_load(path, *args, **kwargs):  # noqa: ANN001
        try:
            return orig_load(path, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "torchcodec" not in msg and "load_with_torchcodec" not in msg:
                raise

            data, sr = sf.read(path, dtype="float32", always_2d=True)
            # soundfile -> [time, channels], torchaudio expects [channels, time]
            tensor = torch.from_numpy(data.T.copy())
            return tensor, sr

    torchaudio.load = safe_load  # type: ignore[assignment]


def _reply(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


class Token2WavSidecar:
    def __init__(self) -> None:
        self.token2wav = None
        self.stream_cache = None
        self.hift_cache = None
        self.ref_audio_path: str | None = None
        self.initialized = False

    def init(self, model_dir: str, n_timesteps: int = 5, float16: bool = False) -> dict:
        _patch_torch_for_cpu()
        _patch_torchaudio_load_fallback()

        from stepaudio2 import Token2wav

        # CPU path should avoid float16
        if not torch.cuda.is_available():
            float16 = False

        t0 = time.perf_counter()
        self.token2wav = Token2wav(model_dir, float16=float16, n_timesteps=n_timesteps)
        self.initialized = True
        return {
            "status": "ok",
            "init_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "cuda_available": bool(torch.cuda.is_available()),
            "float16": bool(float16),
            "model_dir": model_dir,
        }

    def set_ref_audio(self, ref_audio_path: str) -> dict:
        if not self.initialized or self.token2wav is None:
            return {"status": "error", "message": "sidecar not initialized"}

        t0 = time.perf_counter()
        self.stream_cache, self.hift_cache = self.token2wav.set_stream_cache(ref_audio_path)
        self.ref_audio_path = ref_audio_path
        return {
            "status": "ok",
            "set_ref_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "ref_audio_path": ref_audio_path,
        }

    def process(self, tokens: list[int], last_chunk: bool, output_path: str) -> dict:
        if not self.initialized or self.token2wav is None:
            return {"status": "error", "message": "sidecar not initialized"}
        if not self.ref_audio_path:
            return {"status": "error", "message": "ref audio not set"}
        if not tokens:
            return {"status": "error", "message": "tokens list is empty"}

        self.token2wav.stream_cache = self.stream_cache
        self.token2wav.hift_cache_dict = self.hift_cache

        t0 = time.perf_counter()
        wav_data = self.token2wav.stream(
            generated_speech_tokens=tokens,
            prompt_wav=self.ref_audio_path,
            last_chunk=last_chunk,
            return_waveform=True,
        )
        infer_ms = (time.perf_counter() - t0) * 1000.0

        self.stream_cache = self.token2wav.stream_cache
        self.hift_cache = self.token2wav.hift_cache_dict

        if wav_data is None:
            return {
                "status": "ok",
                "output_path": output_path,
                "samples": 0,
                "audio_sec": 0.0,
                "inference_ms": round(infer_ms, 2),
                "last_chunk": bool(last_chunk),
            }

        wav_np = np.asarray(wav_data, dtype=np.float32).squeeze()
        if wav_np.ndim != 1:
            wav_np = wav_np.reshape(-1)

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, wav_np, 24000)

        return {
            "status": "ok",
            "output_path": str(out_path),
            "samples": int(wav_np.shape[0]),
            "audio_sec": round(float(wav_np.shape[0]) / 24000.0, 6),
            "inference_ms": round(infer_ms, 2),
            "last_chunk": bool(last_chunk),
        }


def main() -> int:
    sidecar = Token2WavSidecar()
    _reply({"status": "ready", "pid": os.getpid()})

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _reply({"status": "error", "message": f"invalid JSON: {exc}"})
            continue

        cmd = req.get("cmd")
        try:
            if cmd == "init":
                resp = sidecar.init(
                    model_dir=str(req["model_dir"]),
                    n_timesteps=int(req.get("n_timesteps", 5)),
                    float16=bool(req.get("float16", False)),
                )
            elif cmd == "set_ref_audio":
                resp = sidecar.set_ref_audio(str(req["ref_audio_path"]))
            elif cmd == "process":
                tokens = [int(x) for x in req.get("tokens", [])]
                resp = sidecar.process(
                    tokens=tokens,
                    last_chunk=bool(req.get("last_chunk", False)),
                    output_path=str(req["output_path"]),
                )
            elif cmd == "quit":
                _reply({"status": "ok", "message": "bye"})
                return 0
            else:
                resp = {"status": "error", "message": f"unknown cmd: {cmd}"}
        except Exception as exc:  # noqa: BLE001
            resp = {
                "status": "error",
                "message": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }

        _reply(resp)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
