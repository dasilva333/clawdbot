import {
  joinVoiceChannel,
  createAudioPlayer,
  createAudioResource,
  AudioPlayerStatus,
  VoiceConnectionStatus,
  entersState,
  StreamType,
  EndBehaviorType,
  type VoiceConnection,
  type AudioPlayer,
} from "@discordjs/voice";
import { Readable } from "node:stream";
import type { PluginLogger } from "openclaw";
import prism from "prism-media";

export interface VoiceSession {
  connection: VoiceConnection;
  player: AudioPlayer;
  channelId: string;
  guildId: string;
}

export class VoiceManager {
  private sessions = new Map<string, VoiceSession>();
  private pcmBuffers = new Map<string, Buffer>(); // guildId -> Buffer
  private prefillQueue = new Map<string, Promise<void>>(); // guildId -> Last prefill promise
  private logger: PluginLogger;
  private onAudio: ((guildId: string, pcm: Buffer) => void) | undefined;
  private onSpeechEnd: ((guildId: string) => void) | undefined;

  constructor(logger: PluginLogger) {
    this.logger = logger;
  }

  setAudioHandler(handler: (guildId: string, pcm: Buffer) => void) {
    this.onAudio = handler;
  }

  setSpeechEndHandler(handler: (guildId: string) => void) {
    this.onSpeechEnd = handler;
  }

  async join(guildId: string, channelId: string, adapterCreator: any) {
    const existing = this.sessions.get(guildId);
    if (existing) {
      if (existing.channelId === channelId) return existing;
      existing.connection.destroy();
    }

    const connection = joinVoiceChannel({
      channelId,
      guildId,
      adapterCreator,
      selfDeaf: false,
      selfMute: false,
    });

    const player = createAudioPlayer();
    connection.subscribe(player);

    connection.on(VoiceConnectionStatus.Disconnected, async () => {
      try {
        await Promise.race([
          entersState(connection, VoiceConnectionStatus.Signalling, 5_000),
          entersState(connection, VoiceConnectionStatus.Connecting, 5_000),
        ]);
        // Reconnected
      } catch (error: any) {
        this.logger.warn(`[Voice] Connection to ${guildId} lost permanently: ${error.message}`);
        connection.destroy();
        this.sessions.delete(guildId);
      }
    });

    const activeSession: VoiceSession = { connection, player, channelId, guildId };
    this.sessions.set(guildId, activeSession);
    
    this.logger.info(`[Voice] Session stored for guild ${guildId} channel ${channelId}`);

    // Set up player event listeners for debugging
    player.on(AudioPlayerStatus.Playing, () => {
      this.logger.info(`[Voice] Audio player started playing in guild ${guildId}`);
    });
    player.on(AudioPlayerStatus.Idle, () => {
      this.logger.info(`[Voice] Audio player returned to idle in guild ${guildId}`);
    });
    player.on("error", (error: any) => {
      this.logger.error(`[Voice] Audio player error in guild ${guildId}: ${error.message}\n${error.stack}`);
    });

    // Start listening for audio
    this.startListening(activeSession);

    return activeSession;
  }

  private startListening(session: VoiceSession) {
    const receiver = session.connection.receiver;
    const guildId = session.guildId;
    this.logger.info(`[Voice] Started receiver listening for guild ${guildId}`);

    receiver.speaking.on("start", (userId) => {
      this.logger.info(`[Voice] User speaking: userId=${userId} in guild ${guildId}`);
      
      try {
        const opusStream = receiver.subscribe(userId, {
          end: {
            behavior: EndBehaviorType.AfterSilence,
            duration: 1500, // Wait 1.5s of silence before ending stream
          },
        });

        const decoder = new prism.opus.Decoder({
          rate: 48000,
          channels: 1,
          frameSize: 960,
        });

        opusStream.pipe(decoder);

        const BUFFER_THRESHOLD = 28800; // 300ms at 48kHz mono 16-bit

        decoder.on("data", (chunk: Buffer) => {
          let pcmBuffer = this.pcmBuffers.get(guildId) || Buffer.alloc(0);
          pcmBuffer = Buffer.concat([pcmBuffer, chunk]);
          
          if (pcmBuffer.length >= BUFFER_THRESHOLD) {
            const dataToSend = pcmBuffer;
            this.pcmBuffers.set(guildId, Buffer.alloc(0));
            this.logger.info(`[Voice] Queuing prefill (${dataToSend.length} bytes / ~${Math.round(dataToSend.length/192)}ms) for bridge`);
            
            if (this.onAudio) {
              const previous = this.prefillQueue.get(guildId) || Promise.resolve();
              const next = previous.then(() => this.onAudio!(guildId, dataToSend)).catch(err => {
                this.logger.error(`[Voice] Prefill failed: ${err.message}`);
              });
              this.prefillQueue.set(guildId, next);
            }
          } else {
            this.pcmBuffers.set(guildId, pcmBuffer);
          }
        });

        decoder.on("end", () => {
          const pcmBuffer = this.pcmBuffers.get(guildId) || Buffer.alloc(0);
          if (pcmBuffer.length > 0) {
            const dataToSend = pcmBuffer;
            this.pcmBuffers.set(guildId, Buffer.alloc(0));
            this.logger.info(`[Voice] Flushing remaining PCM (${dataToSend.length} bytes) on stream end`);
            
            if (this.onAudio) {
              const previous = this.prefillQueue.get(guildId) || Promise.resolve();
              const next = previous.then(() => this.onAudio!(guildId, dataToSend)).catch(err => {
                this.logger.error(`[Voice] Final prefill failed: ${err.message}`);
              });
              this.prefillQueue.set(guildId, next);
            }
          }
          this.logger.info(`[Voice] User finished speaking (stream end): userId=${userId}`);
          if (this.onSpeechEnd) {
            // Ensure all audio is sent before triggering decode
            const finalWait = this.prefillQueue.get(guildId) || Promise.resolve();
            finalWait.then(() => {
              this.logger.info(`[Voice] Prefill queue cleared, triggering onSpeechEnd for ${guildId}`);
              this.onSpeechEnd!(guildId);
            });
          }
        });

        decoder.on("error", (err) => {
          this.logger.error(`[Voice] Decoder error for user ${userId}: ${err.message}`);
        });
      } catch (err: any) {
        this.logger.error(`[Voice] Failed to subscribe to user ${userId}: ${err.message}`);
      }
    });
  }

  playStream(guildId: string, stream: Readable) {
    this.logger.info(`[Voice] playStream request for guild ${guildId}`);
    const activeSession = this.sessions.get(guildId);
    if (!activeSession) {
      this.logger.warn(`[Voice] Cannot play stream: no active session for guild ${guildId}`);
      return;
    }

    try {
      this.logger.info("[Voice] Preparing FFmpeg transcoder to 48k stereo...");
      // Let FFmpeg detect the input format and transcode to 48kHz stereo PCM for Discord
      const ffmpeg = new prism.FFmpeg({
        args: [
          "-analyzeduration", "0",
          "-loglevel", "0",
          "-i", "-",
          "-f", "s16le",
          "-ar", "48000",
          "-ac", "2",
        ],
      });

      const resource = createAudioResource(stream.pipe(ffmpeg), {
        inputType: StreamType.Raw,
      });

      this.logger.info(`[Voice] Playing resource in guild ${guildId}...`);
      activeSession.player.play(resource);
    } catch (err: any) {
      this.logger.error(`[Voice] playStream critical failure in guild ${guildId}: ${err.message}`);
    }
  }

  leave(guildId: string) {
    const session = this.sessions.get(guildId);
    if (session) {
      session.connection.destroy();
      this.sessions.delete(guildId);
      this.logger.info(`[Voice] Left voice in guild ${guildId}`);
    }
  }
}
