export interface SseEvent {
  event: string;
  data: string;
}

interface LineEnding {
  index: number;
  length: number;
}

function findLineEnding(buffer: string, endOfStream: boolean): LineEnding | null {
  for (let index = 0; index < buffer.length; index += 1) {
    const character = buffer[index];
    if (character === "\n") return { index, length: 1 };
    if (character !== "\r") continue;

    if (index === buffer.length - 1 && !endOfStream) return null;
    return {
      index,
      length: buffer[index + 1] === "\n" ? 2 : 1,
    };
  }
  return null;
}

/**
 * Incrementally parses a UTF-8 server-sent event stream without assuming that
 * transport chunks align with either characters, lines, or event boundaries.
 */
export async function* parseSseStream(
  stream: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): AsyncGenerator<SseEvent> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let eventName = "";
  let dataLines: string[] = [];
  let reachedEnd = false;
  const abort = () => { void reader.cancel().catch(() => undefined); };
  signal?.addEventListener("abort", abort, { once: true });

  function consumeLine(line: string): SseEvent | null {
    if (line === "") {
      if (dataLines.length === 0) {
        eventName = "";
        return null;
      }

      const parsed = {
        event: eventName || "message",
        data: dataLines.join("\n"),
      };
      eventName = "";
      dataLines = [];
      return parsed;
    }

    if (line.startsWith(":")) return null;

    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);

    if (field === "event") eventName = value;
    if (field === "data") dataLines.push(value);
    return null;
  }

  try {
    while (true) {
      signal?.throwIfAborted();
      const chunk = await reader.read();
      signal?.throwIfAborted();
      if (chunk.done) {
        buffer += decoder.decode();
        reachedEnd = true;
      } else {
        buffer += decoder.decode(chunk.value, { stream: true });
      }

      while (true) {
        const ending = findLineEnding(buffer, reachedEnd);
        if (!ending) break;
        const line = buffer.slice(0, ending.index);
        buffer = buffer.slice(ending.index + ending.length);
        const event = consumeLine(line);
        if (event) {
          signal?.throwIfAborted();
          yield event;
        }
      }

      if (!reachedEnd) continue;

      // An unterminated final line is parsed, but an event is dispatched only
      // after a blank line, matching the SSE framing contract.
      if (buffer) consumeLine(buffer);
      return;
    }
  } finally {
    signal?.removeEventListener("abort", abort);
    if (!reachedEnd) {
      try {
        await reader.cancel();
      } catch {
        // The underlying fetch may already have been aborted.
      }
    }
    reader.releaseLock();
  }
}
