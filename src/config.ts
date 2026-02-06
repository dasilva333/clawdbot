/**
 * MiniCPM-o TTS Plugin Configuration
 */

export interface MiniCPMTTSConfig {
  /** Enable MiniCPM-o TTS */
  enabled?: boolean;

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
  endpoint: string;
  defaultVoice: string;
  format: "wav" | "opus";
  timeoutMs: number;
}

export function resolveConfig(config: MiniCPMTTSConfig = {}): ResolvedMiniCPMTTSConfig {
  return {
    enabled: config.enabled ?? false,
    endpoint: config.endpoint ?? "http://192.168.1.119:8087",
    defaultVoice: config.defaultVoice ?? "default",
    format: config.format ?? "opus",
    timeoutMs: config.timeoutMs ?? 120000,
  };
}
