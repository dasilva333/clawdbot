/**
 * MiniCPM-o TTS Plugin
 *
 * Integrates GGUF TTS API (forced token decoding via llama.cpp-omni)
 * as an OpenClaw plugin. Provides gateway methods for manual synthesis
 * and voice listing. Core TTS integration is handled by the "minicpm"
 * provider in src/tts/tts.ts.
 *
 * Architecture:
 * - GGUF TTS API (FastAPI, port 8087) → llama.cpp-omni (C++, port 8085)
 * - Forced token decoding: tokenize text → LLM forward pass → hidden states → TTS vocoder → WAV/Opus
 * - Voice cloning via reference audio files on the server
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { resolveConfig, type MiniCPMTTSConfig } from "./src/config";
import { MiniCPMTTSProvider } from "./src/provider";

const miniCPMTTSPlugin = {
  id: "minicpm-tts",
  name: "MiniCPM-o TTS",

  register(api: OpenClawPluginApi) {
    const rawConfig = api.pluginConfig as MiniCPMTTSConfig | undefined;
    const config = resolveConfig(rawConfig);

    if (!config.enabled) {
      api.logger.info("MiniCPM-o TTS plugin disabled in config");
      return;
    }

    const provider = new MiniCPMTTSProvider(config);

    /**
     * Gateway Method: minicpm.synthesize
     * Manual TTS synthesis — returns base64 audio
     */
    api.registerGatewayMethod("minicpm.synthesize", async ({ params, respond }) => {
      const { text, voice } = params as { text: string; voice?: string };

      if (!text || typeof text !== "string") {
        respond({ error: "Missing or invalid 'text' parameter" });
        return;
      }

      try {
        const result = await provider.synthesize(text, voice);
        respond({
          success: true,
          format: result.format,
          mimeType: result.mimeType,
          audioBase64: result.buffer.toString("base64"),
          audioSize: result.buffer.length,
        });
      } catch (error) {
        respond({
          error: error instanceof Error ? error.message : "Unknown error",
        });
      }
    });

    /**
     * Gateway Method: minicpm.health
     */
    api.registerGatewayMethod("minicpm.health", async ({ respond }) => {
      const healthy = await provider.healthCheck();
      respond({
        healthy,
        endpoint: config.endpoint,
        format: config.format,
        defaultVoice: config.defaultVoice,
      });
    });

    /**
     * Gateway Method: minicpm.voices
     * List available voice references
     */
    api.registerGatewayMethod("minicpm.voices", async ({ respond }) => {
      const voices = await provider.listVoices();
      respond(voices);
    });

    /**
     * Service: Lifecycle management
     */
    api.registerService({
      id: "minicpm-tts",
      start: async () => {
        api.logger.info(`[MiniCPM TTS] Starting with endpoint: ${config.endpoint}`);
        const healthy = await provider.healthCheck();
        if (!healthy) {
          api.logger.warn(
            `[MiniCPM TTS] Endpoint ${config.endpoint} not reachable. Check if gguf-tts-api is running.`
          );
        } else {
          api.logger.info(`[MiniCPM TTS] Health check passed`);
        }
      },
      stop: async () => {
        api.logger.info("[MiniCPM TTS] Stopping plugin");
      },
    });

    api.logger.info("[MiniCPM TTS] Plugin registered successfully");
  },
};

export default miniCPMTTSPlugin;
