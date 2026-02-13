#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import requests


DEFAULT_PROMPTS = [
    "hello how are you?",
    "what is the meaning of life?",
    "pick one between chicken broccoli and szechuan chicken and explain why.",
]


def slugify(text: str) -> str:
    out = []
    for c in text.lower():
        if c.isalnum():
            out.append(c)
        elif c in (" ", "-", "_"):
            out.append("_")
    return "".join(out).strip("_")[:48] or "prompt"


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def say_to_wav(text: str, aiff_path: Path, wav_path: Path, voice: str) -> None:
    run_cmd(["say", "-v", voice, "-o", str(aiff_path), text])
    run_cmd(["ffmpeg", "-y", "-i", str(aiff_path), "-ar", "48000", "-ac", "1", str(wav_path)])


def wav_to_pcm48_bytes(wav_path: Path) -> bytes:
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-ar", "48000", "-ac", "1", "-f", "s16le", "-"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout


def pcm24_raw_to_wav(raw_path: Path, wav_path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", str(raw_path), str(wav_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def run_case(
    bridge_url: str,
    guild_id: str,
    pcm_48k: bytes,
    chunk_bytes: int,
    prefill_wait_sec: float,
    raw_out_path: Path,
    decode_retries: int,
    retry_wait_sec: float,
) -> dict:
    last_error = None
    for attempt in range(1, decode_retries + 2):
        inject_start = time.perf_counter()
        chunk_count = 0
        for i in range(0, len(pcm_48k), chunk_bytes):
            chunk = pcm_48k[i : i + chunk_bytes]
            r = requests.post(
                f"{bridge_url}/omni/streaming_prefill",
                data=chunk,
                headers={"X-Guild-ID": guild_id},
                timeout=20,
            )
            r.raise_for_status()
            chunk_count += 1
        inject_ms = (time.perf_counter() - inject_start) * 1000.0

        if prefill_wait_sec > 0:
            time.sleep(prefill_wait_sec)

        decode_start = time.perf_counter()
        try:
            response = requests.post(
                f"{bridge_url}/omni/decode",
                headers={"X-Guild-ID": guild_id},
                stream=True,
                timeout=(20, 240),
            )
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt <= decode_retries:
                time.sleep(retry_wait_sec)
                continue
            raise

        first_chunk_ms = None
        total_bytes = 0
        stream_chunks = 0
        with raw_out_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=4096):
                if not chunk:
                    continue
                now = time.perf_counter()
                if first_chunk_ms is None:
                    first_chunk_ms = (now - decode_start) * 1000.0
                f.write(chunk)
                total_bytes += len(chunk)
                stream_chunks += 1
        total_decode_ms = (time.perf_counter() - decode_start) * 1000.0

        audio_sec = total_bytes / (24000.0 * 2.0)
        return {
            "attempt": attempt,
            "inject_ms": round(inject_ms, 2),
            "prefill_chunk_count": chunk_count,
            "first_audio_ms": round(first_chunk_ms or 0.0, 2),
            "total_decode_ms": round(total_decode_ms, 2),
            "response_bytes": total_bytes,
            "response_audio_sec": round(audio_sec, 3),
            "response_chunks": stream_chunks,
            "last_error": last_error,
        }

    raise RuntimeError(f"decode failed after retries: {last_error}")


def write_markdown_report(out_dir: Path, summary: dict) -> Path:
    lines = [
        "# Voice Decode Latency Benchmark",
        "",
        f"- Timestamp: `{summary['timestamp']}`",
        f"- Bridge URL: `{summary['bridge_url']}`",
        f"- Prefill wait: `{summary['prefill_wait_sec']}s`",
        f"- Prefill chunk bytes: `{summary['chunk_bytes']}`",
        f"- Voice synth for inputs: `{summary['say_voice']}`",
        "",
        "## Results",
        "",
        "| Case | Prompt | First audio (ms) | Total decode (ms) | Output audio (s) |",
        "|---|---|---:|---:|---:|",
    ]
    for case in summary["cases"]:
        lines.append(
            f"| {case['case_id']} | {case['prompt']} | {case['metrics']['first_audio_ms']} | "
            f"{case['metrics']['total_decode_ms']} | {case['metrics']['response_audio_sec']} |"
        )
    lines.append("")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure voice path first-audio latency via /omni/decode streaming.")
    parser.add_argument("--bridge-url", default="http://127.0.0.1:8090")
    parser.add_argument("--chunk-bytes", type=int, default=14400)
    parser.add_argument("--prefill-wait-sec", type=float, default=0.0)
    parser.add_argument("--decode-retries", type=int, default=2)
    parser.add_argument("--retry-wait-sec", type=float, default=4.0)
    parser.add_argument("--guild-id", default="", help="If set, reuse this guild ID for all prompts")
    parser.add_argument("--say-voice", default="Samantha")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "voice_eval" / f"latency_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--prompt", action="append", dest="prompts", help="Custom prompt (can pass multiple times)")
    args = parser.parse_args()

    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS

    health = requests.get(f"{args.bridge_url}/health", timeout=10)
    health.raise_for_status()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for idx, prompt in enumerate(prompts, start=1):
        case_id = f"{idx:02d}_{slugify(prompt)}"
        guild_id = args.guild_id or f"latency-bench-{run_id}-{idx:02d}"

        input_aiff = args.out_dir / f"{case_id}.aiff"
        input_wav = args.out_dir / f"{case_id}.wav"
        output_raw = args.out_dir / f"{case_id}.response.raw"
        output_wav = args.out_dir / f"{case_id}.response.wav"

        say_to_wav(prompt, input_aiff, input_wav, args.say_voice)
        pcm_48k = wav_to_pcm48_bytes(input_wav)
        metrics = run_case(
            bridge_url=args.bridge_url,
            guild_id=guild_id,
            pcm_48k=pcm_48k,
            chunk_bytes=args.chunk_bytes,
            prefill_wait_sec=args.prefill_wait_sec,
            raw_out_path=output_raw,
            decode_retries=args.decode_retries,
            retry_wait_sec=args.retry_wait_sec,
        )
        pcm24_raw_to_wav(output_raw, output_wav)

        cases.append(
            {
                "case_id": case_id,
                "guild_id": guild_id,
                "prompt": prompt,
                "input_aiff": str(input_aiff),
                "input_wav": str(input_wav),
                "output_raw": str(output_raw),
                "output_wav": str(output_wav),
                "metrics": metrics,
            }
        )

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "bridge_url": args.bridge_url,
        "prefill_wait_sec": args.prefill_wait_sec,
        "chunk_bytes": args.chunk_bytes,
        "decode_retries": args.decode_retries,
        "retry_wait_sec": args.retry_wait_sec,
        "say_voice": args.say_voice,
        "cases": cases,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = write_markdown_report(args.out_dir, summary)

    print(f"summary: {summary_path}")
    print(f"report:  {report_path}")
    for case in cases:
        m = case["metrics"]
        print(
            f"{case['case_id']}: first_audio_ms={m['first_audio_ms']} total_decode_ms={m['total_decode_ms']} "
            f"output_audio_sec={m['response_audio_sec']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
