#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class Profile:
    name: str
    ref_audio: Path


def parse_tokens(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [int(x) for x in text.split(",") if x.strip()]


def chunk_key(path: Path) -> int:
    stem = path.stem  # audio_tokens_chunk_12
    try:
        return int(stem.split("_")[-1])
    except Exception:  # noqa: BLE001
        return 10**9


def send_cmd(proc: subprocess.Popen, payload: dict, max_lines: int = 200) -> dict:
    proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
    proc.stdin.flush()
    for _ in range(max_lines):
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("sidecar closed stdout unexpectedly")
        stripped = line.strip()
        if not stripped:
            continue
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            # Ignore noisy non-JSON stdout emitted by 3rd-party libs.
            continue
    raise RuntimeError("sidecar did not return JSON response")


def merge_wavs(chunks: list[Path], out_path: Path) -> tuple[float, int]:
    audio_parts: list[np.ndarray] = []
    total_samples = 0
    for chunk in chunks:
        if not chunk.exists():
            continue
        data, sr = sf.read(chunk, dtype="float32", always_2d=False)
        if sr != 24000:
            raise RuntimeError(f"Unexpected sample rate {sr} in {chunk}")
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr[:, 0]
        audio_parts.append(arr)
        total_samples += int(arr.shape[0])

    if audio_parts:
        merged = np.concatenate(audio_parts, axis=0)
    else:
        merged = np.zeros((0,), dtype=np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, merged, 24000)
    return (float(total_samples) / 24000.0, total_samples)


def run_profile(
    python_bin: Path,
    sidecar_script: Path,
    model_dir: Path,
    token_files: list[Path],
    profile: Profile,
    out_dir: Path,
) -> dict:
    proc = subprocess.Popen(
        [str(python_bin), str(sidecar_script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    ready = None
    for _ in range(200):
        line = proc.stdout.readline()
        if not line:
            break
        stripped = line.strip()
        if not stripped:
            continue
        try:
            cand = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if cand.get("status") == "ready":
            ready = cand
            break
    if ready is None:
        raise RuntimeError("sidecar not ready (no JSON ready message)")

    t_profile_start = time.perf_counter()

    t0 = time.perf_counter()
    init_resp = send_cmd(
        proc,
        {"cmd": "init", "model_dir": str(model_dir), "n_timesteps": 5, "float16": False},
    )
    init_wall_ms = (time.perf_counter() - t0) * 1000.0
    if init_resp.get("status") != "ok":
        raise RuntimeError(f"init failed: {init_resp}")

    t0 = time.perf_counter()
    set_ref_resp = send_cmd(proc, {"cmd": "set_ref_audio", "ref_audio_path": str(profile.ref_audio)})
    set_ref_wall_ms = (time.perf_counter() - t0) * 1000.0
    if set_ref_resp.get("status") != "ok":
        raise RuntimeError(f"set_ref_audio failed: {set_ref_resp}")

    profile_dir = out_dir / profile.name
    chunks_out_dir = profile_dir / "chunks"
    chunks_out_dir.mkdir(parents=True, exist_ok=True)

    chunk_rows = []
    chunk_out_files: list[Path] = []
    first_audio_wall_ms = None
    t_chunks_start = time.perf_counter()
    profile_error = None

    for idx, token_file in enumerate(token_files):
        tokens = parse_tokens(token_file)
        chunk_out = chunks_out_dir / f"chunk_{idx:03d}.wav"
        t0 = time.perf_counter()
        resp = send_cmd(
            proc,
            {
                "cmd": "process",
                "tokens": tokens,
                "last_chunk": idx == (len(token_files) - 1),
                "output_path": str(chunk_out),
            },
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        if resp.get("status") != "ok":
            profile_error = {
                "chunk_index": idx,
                "response": resp,
            }
            chunk_rows.append(
                {
                    "chunk_index": idx,
                    "token_file": str(token_file),
                    "token_count": len(tokens),
                    "wall_ms": round(wall_ms, 2),
                    "status": "error",
                    "response": resp,
                }
            )
            break

        audio_sec = float(resp.get("audio_sec", 0.0))
        if first_audio_wall_ms is None and audio_sec > 0:
            first_audio_wall_ms = (time.perf_counter() - t_chunks_start) * 1000.0

        chunk_rows.append(
            {
                "chunk_index": idx,
                "token_file": str(token_file),
                "token_count": len(tokens),
                "status": "ok",
                "inference_ms_reported": float(resp.get("inference_ms", 0.0)),
                "wall_ms": round(wall_ms, 2),
                "audio_sec": audio_sec,
                "output_path": str(chunk_out),
            }
        )
        if chunk_out.exists():
            chunk_out_files.append(chunk_out)

    try:
        send_cmd(proc, {"cmd": "quit"})
    finally:
        proc.wait(timeout=20)

    stderr_text = proc.stderr.read() if proc.stderr else ""
    (profile_dir / "sidecar.stderr.log").write_text(stderr_text, encoding="utf-8")

    final_wav = profile_dir / f"{profile.name}.wav"
    audio_sec_total, total_samples = merge_wavs(chunk_out_files, final_wav)

    chunks_wall_ms = sum(row["wall_ms"] for row in chunk_rows)
    profile_wall_ms = (time.perf_counter() - t_profile_start) * 1000.0

    return {
        "status": "ok" if profile_error is None else "error",
        "profile": profile.name,
        "ref_audio": str(profile.ref_audio),
        "ready": ready,
        "init": {
            "response": init_resp,
            "wall_ms": round(init_wall_ms, 2),
        },
        "set_ref_audio": {
            "response": set_ref_resp,
            "wall_ms": round(set_ref_wall_ms, 2),
        },
        "first_audio_ms_from_first_chunk_send": round(first_audio_wall_ms or 0.0, 2),
        "chunks_wall_ms": round(chunks_wall_ms, 2),
        "profile_wall_ms": round(profile_wall_ms, 2),
        "audio_total_sec": round(audio_sec_total, 4),
        "total_samples": total_samples,
        "rtf_chunks_only": round((chunks_wall_ms / 1000.0) / audio_sec_total, 3) if audio_sec_total > 0 else None,
        "rtf_profile_end_to_end": round((profile_wall_ms / 1000.0) / audio_sec_total, 3)
        if audio_sec_total > 0
        else None,
        "chunk_count": len(chunk_rows),
        "error": profile_error,
        "final_wav": str(final_wav),
        "chunks": chunk_rows,
        "stderr_log": str(profile_dir / "sidecar.stderr.log"),
    }


def write_report(out_dir: Path, summary: dict) -> Path:
    md = []
    md.append("# Sidecar Voice Clone PoC Report")
    md.append("")
    md.append(f"- Timestamp: `{summary['timestamp']}`")
    md.append(f"- Token source: `{summary['tokens_dir']}`")
    md.append(f"- Token chunks: `{summary['token_chunk_count']}`")
    md.append(f"- Sidecar script: `{summary['sidecar_script']}`")
    md.append("")
    md.append("## Results")
    md.append("")
    for row in summary["profiles"]:
        md.append(f"### {row['profile']}")
        md.append(f"- Status: `{row.get('status', 'unknown')}`")
        md.append(f"- Ref audio: `{row['ref_audio']}`")
        md.append(f"- Final wav: `{row['final_wav']}`")
        md.append(f"- Audio duration: `{row['audio_total_sec']}s`")
        md.append(f"- Init wall: `{row['init']['wall_ms']}ms`")
        md.append(f"- Set-ref wall: `{row['set_ref_audio']['wall_ms']}ms`")
        md.append(f"- First audio (from first process send): `{row['first_audio_ms_from_first_chunk_send']}ms`")
        md.append(f"- Chunk processing wall: `{row['chunks_wall_ms']}ms`")
        md.append(f"- Profile end-to-end wall: `{row['profile_wall_ms']}ms`")
        md.append(f"- RTF chunks-only: `{row['rtf_chunks_only']}`")
        md.append(f"- RTF end-to-end: `{row['rtf_profile_end_to_end']}`")
        md.append(f"- Sidecar stderr log: `{row['stderr_log']}`")
        if row.get("error"):
            md.append(f"- Error chunk: `{row['error']['chunk_index']}`")
            md.append(f"- Error message: `{row['error']['response'].get('message', 'unknown')}`")
        md.append("")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone sidecar benchmark for voice cloning PoC.")
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path("/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/venv/bin/python3"),
    )
    parser.add_argument(
        "--sidecar-script",
        type=Path,
        default=Path(__file__).with_name("mini_token2wav_sidecar.py"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/llama.cpp-omni/tools/omni/pyt2w/token2wav"),
    )
    parser.add_argument(
        "--tokens-dir",
        type=Path,
        default=Path("/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/llama.cpp-omni/tools/omni/output/round_000/tts_wav"),
    )
    parser.add_argument(
        "--default-ref",
        type=Path,
        default=Path("/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/cpp_server/assets/default_ref_audio.wav"),
    )
    parser.add_argument(
        "--elon-ref",
        type=Path,
        default=Path("/Users/richardpinedo/Projects/minicpm/demo/web_demo/WebRTC_Demo/assets/voices/elon_musk_ref.wav"),
    )
    parser.add_argument("--max-chunks", type=int, default=0, help="0 means all chunks")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    args = parser.parse_args()

    token_files = sorted(args.tokens_dir.glob("audio_tokens_chunk_*.txt"), key=chunk_key)
    if not token_files:
        raise SystemExit(f"No token chunk files found in {args.tokens_dir}")
    if args.max_chunks > 0:
        token_files = token_files[: args.max_chunks]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    profiles = [
        Profile(name="default_ref", ref_audio=args.default_ref),
        Profile(name="elon_ref", ref_audio=args.elon_ref),
    ]
    for p in profiles:
        if not p.ref_audio.exists():
            raise SystemExit(f"Missing ref audio: {p.ref_audio}")

    rows = []
    for profile in profiles:
        print(f"[run] profile={profile.name}", flush=True)
        row = run_profile(
            python_bin=args.python_bin,
            sidecar_script=args.sidecar_script,
            model_dir=args.model_dir,
            token_files=token_files,
            profile=profile,
            out_dir=args.out_dir,
        )
        rows.append(row)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python_bin": str(args.python_bin),
        "sidecar_script": str(args.sidecar_script),
        "model_dir": str(args.model_dir),
        "tokens_dir": str(args.tokens_dir),
        "token_chunk_count": len(token_files),
        "profiles": rows,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path = write_report(args.out_dir, summary)

    print("")
    print(f"summary: {summary_path}")
    print(f"report:  {report_path}")
    for row in rows:
        print(f"{row['profile']}: {row['final_wav']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
