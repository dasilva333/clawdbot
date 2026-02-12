/**
 * MiniCPM-o TTS Plugin Configuration
 */
export function resolveConfig(config = {}) {
    return {
        enabled: config.enabled ?? false,
        voiceEnabled: config.voiceEnabled ?? false,
        endpoint: config.endpoint ?? "http://localhost:8087",
        defaultVoice: config.defaultVoice ?? "default",
        format: config.format ?? "opus",
        timeoutMs: config.timeoutMs ?? 120000,
    };
}
