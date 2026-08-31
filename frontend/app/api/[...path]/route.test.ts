import assert from "node:assert/strict";
import test, { afterEach, beforeEach } from "node:test";

import { DELETE, GET, PATCH, POST } from "./route.ts";

const LOCAL_TOKEN = "LOCAL_API_SECRET_91233_secure_test_padding";
const originalFetch = globalThis.fetch;
const originalOrigin = process.env.AMITAI_API_ORIGIN;
const originalToken = process.env.AMITAI_LOCAL_API_TOKEN;
const originalAllowLan = process.env.AMITAI_ALLOW_LAN;

function context(...path: string[]) {
  return { params: Promise.resolve({ path }) };
}

beforeEach(() => {
  process.env.AMITAI_API_ORIGIN = "http://127.0.0.1:8000";
  process.env.AMITAI_LOCAL_API_TOKEN = LOCAL_TOKEN;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  if (originalOrigin === undefined) delete process.env.AMITAI_API_ORIGIN;
  else process.env.AMITAI_API_ORIGIN = originalOrigin;
  if (originalToken === undefined) delete process.env.AMITAI_LOCAL_API_TOKEN;
  else process.env.AMITAI_LOCAL_API_TOKEN = originalToken;
  if (originalAllowLan === undefined) delete process.env.AMITAI_ALLOW_LAN;
  else process.env.AMITAI_ALLOW_LAN = originalAllowLan;
});

test("server proxy adds local auth without returning it to browser code", async () => {
  const upstreamRequests: Array<{ input: string; init?: RequestInit }> = [];
  globalThis.fetch = (async (input, init) => {
    upstreamRequests.push({ input: String(input), init });
    return Response.json([{ id: "conversation-1" }]);
  }) as typeof fetch;

  const response = await GET(
    new Request("http://127.0.0.1:3000/api/conversations?limit=5", {
      headers: { Authorization: "Bearer browser-controlled-value" },
    }),
    context("conversations"),
  );

  const upstreamRequest = upstreamRequests[0];
  assert.ok(upstreamRequest);
  assert.equal(upstreamRequest.input, "http://127.0.0.1:8000/api/conversations?limit=5");
  const forwarded = new Headers(upstreamRequest.init?.headers);
  assert.equal(forwarded.get("authorization"), `Bearer ${LOCAL_TOKEN}`);
  assert.equal(response.headers.get("authorization"), null);
  assert.equal(await response.text(), '[{"id":"conversation-1"}]');
});

test("server proxy preserves SSE streaming without waiting for completion", async () => {
  const encoder = new TextEncoder();
  let release = () => undefined;
  const upstreamStream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encoder.encode('event: text\ndata: {"delta":"First"}\n\n'));
      release = () => {
        controller.enqueue(encoder.encode('event: done\ndata: {}\n\n'));
        controller.close();
      };
    },
  });
  let forwardedSignal: AbortSignal | null | undefined;
  globalThis.fetch = (async (_input, init) => {
    forwardedSignal = init?.signal;
    return new Response(upstreamStream, {
      headers: {
        "Content-Type": "text/event-stream",
        "X-Accel-Buffering": "no",
      },
    });
  }) as typeof fetch;

  const request = new Request("http://127.0.0.1:3000/api/chat/stream", {
    method: "POST",
    headers: {
      Accept: "text/event-stream",
      "Content-Type": "application/json",
      Origin: "http://127.0.0.1:3000",
    },
    body: JSON.stringify({ conversation_id: null, message: "Stream" }),
  });
  const response = await POST(request, context("chat", "stream"));
  const reader = response.body!.getReader();
  const first = await reader.read();

  assert.equal(new TextDecoder().decode(first.value), 'event: text\ndata: {"delta":"First"}\n\n');
  assert.equal(response.headers.get("content-type"), "text/event-stream");
  assert.equal(response.headers.get("x-accel-buffering"), "no");
  assert.equal(forwardedSignal, request.signal);

  release();
  const second = await reader.read();
  assert.equal(new TextDecoder().decode(second.value), "event: done\ndata: {}\n\n");
  assert.equal((await reader.read()).done, true);
});

test("server proxy fails closed without exposing a missing secret", async () => {
  delete process.env.AMITAI_LOCAL_API_TOKEN;
  let fetchCalled = false;
  globalThis.fetch = (async () => {
    fetchCalled = true;
    throw new Error("must not run");
  }) as typeof fetch;

  const response = await GET(
    new Request("http://127.0.0.1:3000/api/memory"),
    context("memory"),
  );

  assert.equal(response.status, 503);
  assert.equal(fetchCalled, false);
  assert.equal(await response.text(), '{"detail":"Local API proxy is not configured"}');
  assert.equal(response.headers.get("authorization"), null);
});

test("server proxy refuses a non-loopback backend unless LAN access is explicit", async () => {
  process.env.AMITAI_API_ORIGIN = "https://gpu.example";
  delete process.env.AMITAI_ALLOW_LAN;
  let fetchCalled = false;
  globalThis.fetch = (async () => {
    fetchCalled = true;
    throw new Error("must not run");
  }) as typeof fetch;

  const response = await GET(
    new Request("http://127.0.0.1:3000/api/conversations"),
    context("conversations"),
  );

  assert.equal(response.status, 503);
  assert.equal(fetchCalled, false);
});

test("server proxy rejects cross-origin state changes before forwarding", async () => {
  let fetchCalls = 0;
  globalThis.fetch = (async () => {
    fetchCalls += 1;
    throw new Error("must not run");
  }) as typeof fetch;

  const handlers = [POST, PATCH, DELETE];
  const rejectedOrigins = [
    "https://untrusted.example",
    "http://127.0.0.1:4000",
    "http://localhost:3000",
    "null",
  ];

  for (const handler of handlers) {
    for (const origin of rejectedOrigins) {
      const response = await handler(
        new Request("http://127.0.0.1:3000/api/memory/item-1", {
          method: handler.name,
          headers: {
            "Content-Type": "application/json",
            Origin: origin,
          },
          body: handler === DELETE ? undefined : JSON.stringify({ value: "private" }),
        }),
        context("memory", "item-1"),
      );

      assert.equal(response.status, 403);
      assert.equal(await response.text(), '{"detail":"Cross-origin request denied"}');
    }
  }

  assert.equal(fetchCalls, 0);
});

test("server proxy accepts matching localhost browser origin", async () => {
  let upstreamCalls = 0;
  globalThis.fetch = (async () => {
    upstreamCalls += 1;
    return Response.json({ id: "memory-1" }, { status: 201 });
  }) as typeof fetch;

  const response = await POST(
    new Request("http://localhost:3000/api/memory", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Origin: "http://localhost:3000",
      },
      body: JSON.stringify({ category: "preference", key: "ui.theme", value: "dark" }),
    }),
    context("memory"),
  );

  assert.equal(response.status, 201);
  assert.equal(upstreamCalls, 1);
});
