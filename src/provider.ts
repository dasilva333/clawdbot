/**
 * MiniCPM-o TTS Provider
 *
 * Calls GGUF TTS API (FastAPI proxy on port 8087) which wraps
 * llama.cpp-omni's forced token decoding endpoint.
 * Returns Opus audio (OGG container) for Telegram voice compatibility.
 */

import type { ResolvedMiniCPMTTSConfig } from "./config";

interface SynthesizeResponse {
  detail?: string;
}

export class MiniCPMTTSProvider {
  private config: ResolvedMiniCPMTTSConfig;

  constructor(config: ResolvedMiniCPMTTSConfig) {
    this.config = config;
  }

  /**
   * Synthesize text to audio via GGUF TTS API.
   * Returns raw audio bytes (opus or wav depending on config).
   */
  async synthesize(text: string, voice?: string): Promise<{ buffer: Buffer; format: string; mimeType: string }> {
    const url = `${this.config.endpoint.replace(/\/+$/, "")}/v1/tts/synthesize`;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.config.timeoutMs);

    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          voice: voice || this.config.defaultVoice,
          format: this.config.format,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const body = await response.text().catch(() => "");
        throw new Error(`GGUF TTS API error (${response.status}): ${body.slice(0, 200)}`);
      }

      const buffer = Buffer.from(await response.arrayBuffer());
      const mimeType = this.config.format === "opus" ? "audio/ogg" : "audio/wav";
      return { buffer, format: this.config.format, mimeType };
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        throw new Error(`GGUF TTS API timeout after ${this.config.timeoutMs}ms`);
      }
      throw error;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.config.endpoint.replace(/\/+$/, "")}/health`, {
        method: "GET",
        signal: AbortSignal.timeout(5000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  async listVoices(): Promise<Record<string, unknown>> {
    try {
      const response = await fetch(`${this.config.endpoint.replace(/\/+$/, "")}/v1/voices`, {
        method: "GET",
        signal: AbortSignal.timeout(5000),
      });
      if (!response.ok) {
        return {};
      }
      return (await response.json()) as Record<string, unknown>;
    } catch {
      return {};
    }
  }
}
