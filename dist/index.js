/**
 * MiniCPM-o TTS Plugin (Phase 1 Voice Bridge)
 *
 * Integrates GGUF TTS API as a voice ferry for Discord.
 */
import { Client, GatewayIntentBits } from "discord.js";
import { resolveConfig } from "./src/config.js";
import { MiniCPMTTSProvider } from "./src/provider.js";
import { VoiceManager } from "./src/voice/manager.js";
import { VoiceBridge } from "./src/voice/bridge.js";
const miniCPMTTSPlugin = {
    id: "minicpm-tts",
    name: "MiniCPM-o TTS",
    register(api) {
        const rawConfig = api.pluginConfig;
        const config = resolveConfig(rawConfig);
        api.logger.info(`[MiniCPM TTS] Plugin registration. ID: ${api.id}`);
        api.logger.info(`[MiniCPM TTS] Runtime structure: ${Object.keys(api.runtime || {}).join(", ")}`);
        if (api.runtime?.discord) {
            api.logger.info(`[MiniCPM TTS] Discord runtime methods: ${Object.keys(api.runtime.discord).join(", ")}`);
        }
        if (!config.enabled) {
            api.logger.info("MiniCPM-o TTS plugin disabled");
            return;
        }
        const provider = new MiniCPMTTSProvider(config);
        let voiceManager;
        let voiceBridge;
        let discordClient;
        api.logger.info("[MiniCPM TTS] Plugin registration started");
        /**
         * Voice Setup
         */
        if (config.voiceEnabled) {
            api.logger.info("[MiniCPM TTS] Voice capability enabled in config");
            voiceManager = new VoiceManager(api.logger);
            voiceBridge = new VoiceBridge(voiceManager, provider, api.logger);
            // Wire up the audio listener to the bridge prefill
            voiceManager.setAudioHandler(async (guildId, pcm) => {
                if (voiceBridge) {
                    await voiceBridge.prefill(guildId, pcm);
                }
            });
            voiceManager.setSpeechEndHandler(async (guildId) => {
                if (voiceBridge) {
                    api.logger.info(`[MiniCPM Voice] User speech ended for guild ${guildId}, triggering decode...`);
                    await voiceBridge.triggerDecode(guildId);
                }
            });
            // 🔧 [Compatibility Fix] OpenClaw core now uses Carbon client, which is incompatible with discord.js voice.
            // We will ALWAYS use a dedicated discord.js client for voice handling.
            api.logger.info("[MiniCPM TTS] Using dedicated Discord.js client for voice support (Carbon bypass).");
        }
        /**
         * Commands: /join, /leave
         */
        api.registerCommand({
            name: "join",
            description: "Join a voice channel (auto-joins yours if no ID provided)",
            requireAuth: false,
            handler: async (ctx) => {
                api.logger.info(`[MiniCPM Voice] /join command entered. Context: senderId=${ctx.senderId}, from=${ctx.from}, args=${ctx.args}`);
                try {
                    // 🔧 [Compatibility Fix] OpenClaw core now uses Carbon client, which is incompatible with discord.js.
                    // We always use a dedicated Discord.js client instance for voice.
                    if (!discordClient) {
                        api.logger.info("[MiniCPM Voice] Initializing dedicated Discord.js client...");
                        const token = api.config?.channels?.discord?.token;
                        if (token) {
                            discordClient = new Client({
                                intents: [
                                    GatewayIntentBits.Guilds,
                                    GatewayIntentBits.GuildVoiceStates,
                                    GatewayIntentBits.GuildMessages,
                                ],
                            });
                            await discordClient.login(token);
                            api.logger.info(`[MiniCPM Voice] Dedicated client logged in: ${discordClient.user?.tag}`);
                        }
                    }
                    if (!discordClient || !voiceManager) {
                        api.logger.error(`[MiniCPM Voice] Join aborted: client=${!!discordClient}, manager=${!!voiceManager}`);
                        return { content: "Voice bridge is not enabled." };
                    }
                    const fromParts = (ctx.from || "").split(":");
                    const sourceChannelId = fromParts[fromParts.length - 1];
                    api.logger.info(`[MiniCPM Voice] Parsed sourceChannelId: ${sourceChannelId}`);
                    // Determine target channel
                    let targetChannelId = ctx.args?.trim();
                    let guildId;
                    if (targetChannelId) {
                        targetChannelId = targetChannelId.replace(/[<#>]/g, "").split("/").pop() || targetChannelId;
                        api.logger.info(`[MiniCPM Voice] Resolving manual targetChannelId: ${targetChannelId}`);
                        const channel = await discordClient.channels.fetch(targetChannelId).catch((e) => {
                            api.logger.error(`[MiniCPM Voice] Failed to fetch target channel ${targetChannelId}: ${e.message}`);
                            return null;
                        });
                        if (channel && "guild" in channel) {
                            guildId = channel.guild.id;
                            api.logger.info(`[MiniCPM Voice] Resolved guildId: ${guildId}`);
                        }
                        else {
                            api.logger.warn(`[MiniCPM Voice] Channel ${targetChannelId} not found or not in a guild`);
                            return { content: `Could not find voice channel with ID: ${targetChannelId}` };
                        }
                    }
                    else {
                        api.logger.info(`[MiniCPM Voice] No target ID; auto-resolving from user voice state...`);
                        const sourceChannel = await discordClient.channels.fetch(sourceChannelId).catch((e) => {
                            api.logger.error(`[MiniCPM Voice] Failed to fetch source channel ${sourceChannelId}: ${e.message}`);
                            return null;
                        });
                        if (!sourceChannel || !("guild" in sourceChannel)) {
                            api.logger.error(`[MiniCPM Voice] Source channel ${sourceChannelId} has no guild context`);
                            return { content: "Could not find guild context." };
                        }
                        const guild = sourceChannel.guild;
                        guildId = guild.id;
                        api.logger.info(`[MiniCPM Voice] Auto-resolved guildId: ${guildId} (${guild.name})`);
                        api.logger.info(`[MiniCPM Voice] Fetching member profile for ${ctx.senderId}...`);
                        const member = await guild.members.fetch(ctx.senderId).catch((e) => {
                            api.logger.error(`[MiniCPM Voice] Member fetch failed: ${e.message}`);
                            return null;
                        });
                        if (!member || !member.voice.channelId) {
                            api.logger.warn(`[MiniCPM Voice] User ${ctx.senderId} voice state: ${member ? 'Not in channel' : 'Member not found'}`);
                            return { content: "You must be in a voice channel to use auto-join. Otherwise, provide a channel ID." };
                        }
                        targetChannelId = member.voice.channelId;
                        api.logger.info(`[MiniCPM Voice] Found member in voice channel: ${targetChannelId}`);
                    }
                    if (!guildId || !targetChannelId) {
                        api.logger.error(`[MiniCPM Voice] Resolution failed: guildId=${guildId}, targetChannelId=${targetChannelId}`);
                        return { content: "Failed to resolve guild or channel IDs." };
                    }
                    api.logger.info(`[MiniCPM Voice] Initiating voiceManager.join for guild ${guildId}, channel ${targetChannelId}...`);
                    const guildObj = await discordClient.guilds.fetch(guildId);
                    await voiceManager.join(guildId, targetChannelId, guildObj.voiceAdapterCreator);
                    api.logger.info("[MiniCPM Voice] voiceManager.join successful");
                    return { content: `Joined <#${targetChannelId}> 🍎` };
                }
                catch (error) {
                    api.logger.error(`[MiniCPM Voice] CRITICAL: /join handler exception: ${error.message}\n${error.stack}`);
                    return { content: `Error: ${error.message}` };
                }
            },
        });
        api.registerCommand({
            name: "leave",
            description: "Leave the current voice channel",
            requireAuth: false,
            handler: async (ctx) => {
                api.logger.info(`[MiniCPM Voice] /leave command entered. senderId=${ctx.senderId}, from=${ctx.from}`);
                console.log(`[MiniCPM Voice] /leave command entered. senderId=${ctx.senderId}, from=${ctx.from}`);
                if (!voiceManager) {
                    api.logger.error("[MiniCPM Voice] Leave failed: voiceManager is null");
                    return { content: "Voice bridge is not enabled." };
                }
                try {
                    const fromParts = (ctx.from || "").split(":");
                    const channelId = fromParts[fromParts.length - 1];
                    const client = discordClient || api.runtime?.discord?.getClient?.();
                    api.logger.info(`[MiniCPM Voice] Resolving guild context for channel ${channelId}...`);
                    const channel = await client?.channels.fetch(channelId).catch((e) => {
                        api.logger.error(`[MiniCPM Voice] Channel fetch failed: ${e.message}`);
                        return null;
                    });
                    const guildId = channel?.guild?.id;
                    if (!guildId) {
                        api.logger.warn(`[MiniCPM Voice] /leave failed: could not resolve guildId`);
                        return { content: "Could not find guild context." };
                    }
                    api.logger.info(`[MiniCPM Voice] Executing voiceManager.leave for guild ${guildId}`);
                    voiceManager.leave(guildId);
                    return { content: "Left voice channel." };
                }
                catch (error) {
                    api.logger.error(`[MiniCPM Voice] /leave handler exception: ${error.message}\n${error.stack}`);
                    return { content: `Error: ${error.message}` };
                }
            },
        });
        api.registerCommand({
            name: "setvoice",
            description: "List or set the voice (e.g. /setvoice jarvis)",
            requireAuth: false,
            handler: async (ctx) => {
                const arg = ctx.args?.trim();
                if (!arg) {
                    // List voices
                    const voices = await provider.listVoices();
                    if (voices.length === 0) {
                        return { content: "No custom voices found in `assets/voices`." };
                    }
                    return { content: `**Available Voices:**\n${voices.map(v => `• \`/setvoice ${v}\``).join("\n")}` };
                }
                // Set voice
                try {
                    const newVoice = await provider.setVoice(arg);
                    return { content: `✅ Switched voice to: **${newVoice}**` };
                }
                catch (e) {
                    return { content: `❌ Failed to switch voice: ${e.message}` };
                }
            },
        });
        /**
         * Hook: Speak on response (from chat)
         */
        api.on("message_sent", async (event, ctx) => {
            api.logger.debug(`[MiniCPM Voice] Hook message_sent: channelId=${ctx.channelId} content="${event.content.slice(0, 30)}..."`);
            try {
                if (!voiceBridge || !ctx.channelId.includes("discord"))
                    return;
                // Use the dedicated Discord.js client if initialized
                const client = discordClient;
                if (!client) {
                    // If no dedicated client, we are not in a voice channel
                    return;
                }
                api.logger.info(`[MiniCPM Voice] Hook resolving guild for channel ${ctx.channelId}...`);
                const channel = await client.channels.fetch(ctx.channelId).catch(() => null);
                const guildId = channel?.guild?.id;
                if (guildId) {
                    api.logger.info(`[MiniCPM Voice] Hook initiating auto-speech for guild ${guildId}`);
                    await voiceBridge.speak(guildId, event.content);
                }
                else {
                    api.logger.debug(`[MiniCPM Voice] Hook could not resolve guildId for ${ctx.channelId}`);
                }
            }
            catch (error) {
                api.logger.error(`[MiniCPM Voice] message_sent hook exception: ${error.message}`);
            }
        });
        /**
         * Fallback Hook: Log chat activity & Text Command Fallback
         */
        api.on("message_received", async (event, ctx) => {
            api.logger.debug(`[MiniCPM Voice] Hook message_received: from=${event.from} channel=${ctx.channelId} content="${event.content.slice(0, 50)}..."`);
            // Fallback for /setvoice
            if (event.content.startsWith("/setvoice")) {
                const arg = event.content.replace("/setvoice", "").trim();
                api.logger.info(`[MiniCPM Voice] Text command detected: /setvoice "${arg}"`);
                try {
                    if (!arg) {
                        const voices = await provider.listVoices();
                        api.logger.info(`[MiniCPM Voice] Available voices: ${voices.join(", ")}`);
                    }
                    else {
                        const newVoice = await provider.setVoice(arg);
                        api.logger.info(`[MiniCPM Voice] Voice switched to: ${newVoice}`);
                    }
                }
                catch (e) {
                    api.logger.error(`[MiniCPM Voice] /setvoice failed: ${e.message}`);
                }
            }
        });
        /**
         * Legacy Gateway Methods
         */
        api.registerGatewayMethod("minicpm.synthesize", async ({ params, respond }) => {
            const { text, voice } = params;
            try {
                const result = await provider.synthesize(text, voice);
                respond({
                    success: true,
                    format: result.format,
                    mimeType: result.mimeType,
                    audioBase64: result.buffer.toString("base64"),
                });
            }
            catch (error) {
                respond({ error: error instanceof Error ? error.message : "Unknown error" });
            }
        });
        api.registerService({
            id: "minicpm-tts",
            start: () => api.logger.info("[MiniCPM TTS] Service started"),
            stop: () => {
                discordClient?.destroy();
                api.logger.info("[MiniCPM TTS] Service stopped");
            },
        });
    },
};
export default miniCPMTTSPlugin;
