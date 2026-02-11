/**
 * MiniCPM-o TTS Plugin Configuration
 */

export interface MiniCPMTTSConfig {
  /** Enable MiniCPM-o TTS */
  enabled?: boolean;

  /** Enable Discord Voice Bridge (Phase 1) */
  voiceEnabled?: boolean;

  /** GGUF TTS API endpoint (FastAPI proxy wrapping llama.cpp-omni) */
  endpoint?: string;

  /** Default voice name or ref audio path */
  defaultVoice?: string;

  /** Output format: wav or opus */
  format?: "wav" | "opus";

  /** Request timeout in milliseconds */
  timeoutMs?: number;
}

export interface ResolvedMiniCPMTTSConfig {
  enabled: boolean;
  voiceEnabled: boolean;
  endpoint: string;
  defaultVoice: string;
  format: "wav" | "opus";
  timeoutMs: number;
}

export function resolveConfig(config: MiniCPMTTSConfig = {}): ResolvedMiniCPMTTSConfig {
  return {
    enabled: config.enabled ?? false,
    voiceEnabled: config.voiceEnabled ?? false,
    endpoint: config.endpoint ?? "http://localhost:8087",
    defaultVoice: config.defaultVoice ?? "default",
    format: config.format ?? "opus",
    timeoutMs: config.timeoutMs ?? 120000,
  };
}
