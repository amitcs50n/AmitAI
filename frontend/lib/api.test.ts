import assert from "node:assert/strict";
import test from "node:test";

import {
  ApiError,
  createMemory,
  deleteMemory,
  listMemories,
  updateMemory,
} from "./api.ts";
import type { MemoryRecord } from "./types.ts";

const memory: MemoryRecord = {
  id: "memory/one",
  operation: "current",
  category: "preference",
  key: "ui.theme",
  value: "dark",
  status: "active",
  source: { conversation_id: null, message_id: null },
  updated_at: "2026-08-30T10:00:00Z",
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

test("memory list client uses active, search, category, and deleted routes", async () => {
  const originalFetch = globalThis.fetch;
  const requests: string[] = [];
  globalThis.fetch = (async (input) => {
    requests.push(String(input));
    return jsonResponse([memory]);
  }) as typeof fetch;

  try {
    await listMemories();
    await listMemories({ query: "UI theme + color" });
    await listMemories({ category: "project" });
    await listMemories({ category: "instruction", status: "deleted" });
  } finally {
    globalThis.fetch = originalFetch;
  }

  assert.deepEqual(requests, [
    "/api/memory",
    "/api/memory?query=UI+theme+%2B+color",
    "/api/memory?category=project",
    "/api/memory?category=instruction&status=deleted",
  ]);
});

test("memory mutation client uses typed JSON routes and encoded ids", async () => {
  const originalFetch = globalThis.fetch;
  const requests: Array<{ input: string; init?: RequestInit }> = [];
  globalThis.fetch = (async (input, init) => {
    requests.push({ input: String(input), init });
    if (init?.method === "DELETE") return new Response(null, { status: 204 });
    return jsonResponse(memory);
  }) as typeof fetch;

  try {
    await createMemory({ category: "preference", key: "ui.theme", value: "dark" });
    await updateMemory("memory/one", { value: "light" });
    await deleteMemory("memory/one");
  } finally {
    globalThis.fetch = originalFetch;
  }

  assert.equal(requests[0].input, "/api/memory");
  assert.equal(requests[0].init?.method, "POST");
  assert.deepEqual(JSON.parse(String(requests[0].init?.body)), {
    category: "preference",
    key: "ui.theme",
    value: "dark",
  });
  assert.equal(requests[1].input, "/api/memory/memory%2Fone");
  assert.equal(requests[1].init?.method, "PATCH");
  assert.deepEqual(JSON.parse(String(requests[1].init?.body)), { value: "light" });
  assert.equal(requests[2].input, "/api/memory/memory%2Fone");
  assert.equal(requests[2].init?.method, "DELETE");
});

test("memory client preserves 409 and 422 ApiError details", async () => {
  const originalFetch = globalThis.fetch;
  const responses = [
    jsonResponse({ detail: "Memory changed concurrently" }, 409),
    jsonResponse({ detail: "Memory value is invalid" }, 422),
  ];
  globalThis.fetch = (async () => responses.shift()!) as typeof fetch;

  try {
    await assert.rejects(updateMemory("one", { value: "light" }), (error: unknown) => {
      assert.ok(error instanceof ApiError);
      assert.equal(error.status, 409);
      assert.equal(error.backendReached, true);
      assert.equal(error.message, "Memory changed concurrently");
      return true;
    });
    await assert.rejects(
      createMemory({ category: "profile", key: "display.name", value: "" }),
      (error: unknown) => {
        assert.ok(error instanceof ApiError);
        assert.equal(error.status, 422);
        assert.equal(error.message, "Memory value is invalid");
        return true;
      },
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("memory client reports an unavailable backend", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async () => {
    throw new TypeError("fetch failed");
  }) as typeof fetch;

  try {
    await assert.rejects(listMemories(), (error: unknown) => {
      assert.ok(error instanceof ApiError);
      assert.equal(error.status, null);
      assert.equal(error.backendReached, false);
      return true;
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});
