import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  loadWebMedia: vi.fn(),
}));

vi.mock("../media.js", () => ({
  loadWebMedia: mocks.loadWebMedia,
}));

import { deliverWebReply } from "./deliver-reply.js";

function makeMsg() {
  return {
    from: "+15550001111",
    to: "+15550002222",
    id: "msg-1",
    reply: vi.fn().mockResolvedValue(undefined),
    sendMedia: vi.fn().mockResolvedValue(undefined),
  };
}

function makeReplyLogger() {
  return {
    info: vi.fn(),
    warn: vi.fn(),
  };
}

type DeliverMsg = Parameters<typeof deliverWebReply>[0]["msg"];

describe("deliverWebReply audio caption mode", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("defaults to caption mode for audio media", async () => {
    const msg = makeMsg();
    const replyLogger = makeReplyLogger();
    mocks.loadWebMedia.mockResolvedValue({
      buffer: Buffer.from("audio"),
      kind: "audio",
      contentType: "audio/ogg; codecs=opus",
    });

    await deliverWebReply({
      replyResult: { text: "hola mundo", mediaUrl: "https://example.com/audio.ogg" },
      msg: msg as DeliverMsg,
      maxMediaBytes: 1024 * 1024,
      textLimit: 4000,
      replyLogger,
    });

    expect(msg.reply).not.toHaveBeenCalled();
    expect(msg.sendMedia).toHaveBeenCalledWith(
      expect.objectContaining({
        ptt: true,
        caption: "hola mundo",
      }),
    );
  });

  it("supports separate mode by sending text before audio", async () => {
    const msg = makeMsg();
    const replyLogger = makeReplyLogger();
    mocks.loadWebMedia.mockResolvedValue({
      buffer: Buffer.from("audio"),
      kind: "audio",
      contentType: "audio/ogg; codecs=opus",
    });

    await deliverWebReply({
      replyResult: { text: "hola mundo", mediaUrl: "https://example.com/audio.ogg" },
      msg: msg as DeliverMsg,
      maxMediaBytes: 1024 * 1024,
      textLimit: 4000,
      replyLogger,
      audioCaptionMode: "separate",
    });

    expect(msg.reply).toHaveBeenCalledTimes(1);
    expect(msg.reply).toHaveBeenCalledWith("hola mundo");
    expect(msg.sendMedia).toHaveBeenCalledWith(
      expect.objectContaining({
        ptt: true,
        caption: undefined,
      }),
    );
  });

  it("does not change caption behavior for non-audio media", async () => {
    const msg = makeMsg();
    const replyLogger = makeReplyLogger();
    mocks.loadWebMedia.mockResolvedValue({
      buffer: Buffer.from("image"),
      kind: "image",
      contentType: "image/jpeg",
    });

    await deliverWebReply({
      replyResult: { text: "hola mundo", mediaUrl: "https://example.com/image.jpg" },
      msg: msg as DeliverMsg,
      maxMediaBytes: 1024 * 1024,
      textLimit: 4000,
      replyLogger,
      audioCaptionMode: "separate",
    });

    expect(msg.reply).not.toHaveBeenCalled();
    expect(msg.sendMedia).toHaveBeenCalledWith(
      expect.objectContaining({
        caption: "hola mundo",
      }),
    );
  });
});
