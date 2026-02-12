
import { VoiceManager } from "./manager.js";
import { MiniCPMTTSProvider } from "../provider.js";
import type { PluginLogger } from "openclaw";

export class VoiceBridge {
  private manager: VoiceManager;
  private provider: MiniCPMTTSProvider;
  private logger: PluginLogger;
  private isDecoding = new Map<string, boolean>();

  constructor(manager: VoiceManager, provider: MiniCPMTTSProvider, logger: PluginLogger) {
    this.manager = manager;
    this.provider = provider;
    this.logger = logger;
  }

  async prefill(guildId: string, pcmBuffer: Buffer) {
    // Forward raw PCM chunks to the bridge
    await this.provider.prefill(guildId, pcmBuffer);
  }

  async triggerDecode(guildId: string) {
    if (this.isDecoding.get(guildId)) {
      this.logger.info(`[VoiceBridge] Decode already in progress for guild ${guildId}, skipping duplicate trigger.`);
      return;
    }
    
    try {
      this.isDecoding.set(guildId, true);
      this.logger.info(`[VoiceBridge] Triggering decode for guild ${guildId}...`);
      const stream = await this.provider.triggerDecode(guildId);
      if (stream) {
        this.logger.info(`[VoiceBridge] Received audio stream from decode trigger, playing...`);
        // Use isRaw=true for the Open Pipe PCM stream
        this.manager.playStream(guildId, stream, true);
      } else {
        this.logger.warn(`[VoiceBridge] Decode trigger returned no audio for guild ${guildId}`);
      }
    } catch (error) {
      this.logger.error(`[VoiceBridge] Decode trigger failed for ${guildId}: ${error}`);
    } finally {
      // Small cooldown to prevent rapid-fire triggers
      setTimeout(() => this.isDecoding.set(guildId, false), 2000);
    }
  }

  async speak(guildId: string, text: string, voice?: string) {
    try {
      this.logger.info(`[VoiceBridge] Generating stream for guild ${guildId}: "${text.slice(0, 50)}..."`);
      const stream = await this.provider.synthesizeStream(text, voice);
      // Use isRaw=true for the unified Open Pipe PCM stream
      this.manager.playStream(guildId, stream, true);
    } catch (error) {
      this.logger.error(`[VoiceBridge] Failed to play voice in ${guildId}: ${error}`);
    }
  }
}
