import assert from "node:assert/strict";
import test from "node:test";

import { ApiError, sendChatStream } from "./api.ts";
import type { ChatResponse } from "./types.ts";

const encoder = new TextEncoder();

function sseEvent(name: string, data: unknown): string {
  return `event: ${name}\r\ndata: ${JSON.stringify(data)}\r\n\r\n`;
}

function chunkedStream(source: string, chunkSizes: number[]): ReadableStream<Uint8Array> {
  const bytes = encoder.encode(source);
  let offset = 0;
  let chunkIndex = 0;

  return new ReadableStream<Uint8Array>({
    pull(controller) {
      if (offset === bytes.length) {
        controller.close();
        return;
      }
      const requestedSize = chunkSizes[chunkIndex % chunkSizes.length];
      chunkIndex += 1;
      const end = Math.min(offset + requestedSize, bytes.length);
      controller.enqueue(bytes.slice(offset, end));
      offset = end;
    },
  });
}

function streamResponse(source: string): Response {
  return new Response(chunkedStream(source, [1, 2, 5, 3, 8]), {
    headers: { "Content-Type": "text/event-stream; charset=utf-8" },
  });
}

async function withFetchResponse<T>(response: Response, run: () => Promise<T>): Promise<T> {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input, init) => {
    assert.equal(input, "/api/chat/stream");
    assert.equal(init?.method, "POST");
    assert.equal(new Headers(init?.headers).get("accept"), "text/event-stream");
    return response;
  }) as typeof fetch;

  try {
    return await run();
  } finally {
    globalThis.fetch = originalFetch;
  }
}

const finalResponse: ChatResponse = {
  conversation_id: "conversation-1",
  message_id: "message-2",
  response: "Hello, 世界",
  metadata: {
    model: "test-model",
    latency_ms: 42,
    input_tokens: 11,
    output_tokens: 3,
    validator: { retry_attempted: false, retry_passed: null },
    tools: [],
    memory: [],
  },
};

test("image SSE sends explicit consent only to the local chat proxy", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input, init) => {
    assert.equal(input, "/api/chat/stream");
    assert.deepEqual(JSON.parse(String(init?.body)), {
      conversation_id: null, message: "Describe", asset_ids: ["current-image"], allow_remote_vision: true,
    });
    return streamResponse(sseEvent("start", { conversation_id: null }) + sseEvent("text", { delta: finalResponse.response })
      + sseEvent("final", finalResponse) + sseEvent("done", {}));
  }) as typeof fetch;
  try {
    assert.deepEqual(await sendChatStream({
      conversation_id: null, message: "Describe", asset_ids: ["current-image"], allow_remote_vision: true,
    }, { onText: () => undefined }), finalResponse);
  } finally { globalThis.fetch = originalFetch; }
});

test("POST SSE emits multiple deltas that reconstruct the final response", async () => {
  const source = [
    ": keepalive\r\n\r\n",
    sseEvent("start", { conversation_id: null }),
    sseEvent("text", { delta: "Hello, " }),
    sseEvent("text", { delta: "世界" }),
    sseEvent("final", finalResponse),
    sseEvent("done", {}),
  ].join("");
  const deltas: string[] = [];
  let starts = 0;
  let completed = false;
  let receivedFinal: ChatResponse | null = null;

  const result = await withFetchResponse(streamResponse(source), () =>
    sendChatStream(
      { conversation_id: null, message: "Say hello" },
      {
        onStart: () => {
          starts += 1;
        },
        onText: (delta) => deltas.push(delta),
        onFinal: (response) => {
          receivedFinal = response;
        },
        onDone: () => {
          completed = true;
        },
      },
    ),
  );

  assert.equal(starts, 1);
  assert.deepEqual(deltas, ["Hello, ", "世界"]);
  assert.equal(deltas.join(""), finalResponse.response);
  assert.deepEqual(receivedFinal, finalResponse);
  assert.deepEqual(result, finalResponse);
  assert.equal(result.metadata.input_tokens, 11);
  assert.equal(result.metadata.output_tokens, 3);
  assert.equal(completed, true);
});

test("rejects a stream that ends before final and done", async () => {
  const source =
    sseEvent("start", { conversation_id: "conversation-1" }) +
    sseEvent("text", { delta: "partial" });

  await assert.rejects(
    withFetchResponse(streamResponse(source), () =>
      sendChatStream(
        { conversation_id: "conversation-1", message: "Continue" },
        { onText: () => undefined },
      ),
    ),
    (error: unknown) => {
      assert.ok(error instanceof ApiError);
      assert.equal(error.message, "Stream ended before completion");
      assert.equal(error.backendReached, true);
      return true;
    },
  );
});

test("rejects a final response that differs from emitted deltas", async () => {
  const source = [
    sseEvent("start", { conversation_id: null }),
    sseEvent("text", { delta: "discarded candidate" }),
    sseEvent("final", finalResponse),
    sseEvent("done", {}),
  ].join("");

  await assert.rejects(
    withFetchResponse(streamResponse(source), () =>
      sendChatStream(
        { conversation_id: null, message: "Mismatch" },
        { onText: () => undefined },
      ),
    ),
    (error: unknown) => {
      assert.ok(error instanceof ApiError);
      assert.equal(error.message, "Streamed text did not match the final response");
      return true;
    },
  );
});

test("surfaces a server error event as a reached-backend error", async () => {
  const source =
    sseEvent("start", { conversation_id: null }) +
    sseEvent("error", { detail: "Generation failed" });

  await assert.rejects(
    withFetchResponse(streamResponse(source), () =>
      sendChatStream(
        { conversation_id: null, message: "Fail" },
        { onText: () => undefined },
      ),
    ),
    (error: unknown) => {
      assert.ok(error instanceof ApiError);
      assert.equal(error.message, "Generation failed");
      assert.equal(error.backendReached, true);
      return true;
    },
  );
});
