/**
 * MiniCPM-o TTS Provider
 *
 * Calls GGUF TTS API (FastAPI proxy on port 8087) which wraps
 * llama.cpp-omni's forced token decoding endpoint.
 * Returns Opus audio (OGG container) for Telegram voice compatibility.
 */
import { Readable } from "node:stream";
export class MiniCPMTTSProvider {
    config;
    constructor(config) {
        this.config = config;
    }
    /**
     * Synthesize text to audio via GGUF TTS API.
     * Returns raw audio bytes (opus or wav depending on config).
     */
    async synthesize(text, voice) {
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
        }
        catch (error) {
            if (error instanceof Error && error.name === "AbortError") {
                throw new Error(`GGUF TTS API timeout after ${this.config.timeoutMs}ms`);
            }
            throw error;
        }
        finally {
            clearTimeout(timeoutId);
        }
    }
    /**
     * Synthesize text to a stream (Phase 1 Voice Bridge).
     */
    async synthesizeStream(text, voice) {
        const url = `${this.config.endpoint.replace(/\/+$/, "")}/v1/tts/synthesize?stream=true`;
        console.log(`[MiniCPM Provider] synthesizeStream: POST ${url} text="${text.slice(0, 50)}..."`);
        try {
            const response = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    text,
                    voice: voice || this.config.defaultVoice,
                    format: "wav",
                }),
            });
            if (!response.ok) {
                const body = await response.text().catch(() => "");
                console.error(`[MiniCPM Provider] API error: ${response.status} ${body.slice(0, 200)}`);
                throw new Error(`GGUF TTS API error (${response.status}): ${body.slice(0, 200)}`);
            }
            if (!response.body) {
                console.error("[MiniCPM Provider] API returned empty body");
                throw new Error("GGUF TTS API returned empty body for stream");
            }
            console.log("[MiniCPM Provider] Stream opened successfully");
            return Readable.fromWeb(response.body);
        }
        catch (error) {
            console.error(`[MiniCPM Provider] Connection failed: ${error.message}`);
            throw error;
        }
    }
    async healthCheck() {
        try {
            const response = await fetch(`${this.config.endpoint.replace(/\/+$/, "")}/health`, {
                method: "GET",
                signal: AbortSignal.timeout(5000),
            });
            return response.ok;
        }
        catch {
            return false;
        }
    }
    async listVoices() {
        try {
            const response = await fetch(`${this.config.endpoint.replace(/\/+$/, "")}/voices`, {
                method: "GET",
                signal: AbortSignal.timeout(5000),
            });
            if (!response.ok) {
                return [];
            }
            const data = (await response.json());
            return data.voices || [];
        }
        catch {
            return [];
        }
    }
    async setVoice(voiceId) {
        try {
            const response = await fetch(`${this.config.endpoint.replace(/\/+$/, "")}/voice/set`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ voice_id: voiceId }),
            });
            if (!response.ok) {
                const err = await response.text();
                throw new Error(`Voice switch failed: ${err}`);
            }
            const data = (await response.json());
            return data.voice;
        }
        catch (error) {
            console.error(`[MiniCPM Provider] setVoice failed: ${error.message}`);
            throw error;
        }
    }
    async prefill(guildId, pcmBuffer) {
        const url = `${this.config.endpoint.replace(/\/+$/, "")}/omni/streaming_prefill`;
        try {
            const response = await fetch(url, {
                method: "POST",
                headers: {
                    "Content-Type": "application/octet-stream",
                    "X-Guild-ID": guildId
                },
                body: pcmBuffer,
            });
            if (!response.ok) {
                throw new Error(`Prefill API error (${response.status})`);
            }
        }
        catch (error) {
            console.error(`[MiniCPM Provider] Prefill failed: ${error.message}`);
        }
    }
    async triggerDecode(guildId) {
        const url = `${this.config.endpoint.replace(/\/+$/, "")}/omni/decode`;
        try {
            const response = await fetch(url, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-Guild-ID": guildId
                },
            });
            if (!response.ok) {
                throw new Error(`Decode API error (${response.status})`);
            }
            if (!response.body) {
                return null;
            }
            return Readable.fromWeb(response.body);
        }
        catch (error) {
            console.error(`[MiniCPM Provider] Decode trigger failed: ${error.message}`);
            return null;
        }
    }
}
