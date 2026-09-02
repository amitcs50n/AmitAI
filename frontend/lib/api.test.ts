import assert from "node:assert/strict";
import test from "node:test";

import {
  ApiError,
  createMemory,
  deleteMemory,
  listMemories,
  updateMemory,
  uploadImage,
  deleteAsset,
  assetContentUrl,
  sendChat,
  getCapabilities,
} from "./api.ts";
import type { MemoryRecord } from "./types.ts";

const memory: MemoryRecord = {
  id: "memory/one",
  operation: "current",
  category: "preference",
  key: "ui.theme",
  value: "dark",
  sensitivity: "local_only",
  status: "active",
  source: { conversation_id: null, message_id: null },
  updated_at: "2026-08-30T10:00:00Z",
};

test("image client uploads multipart, encodes ID routes and sends only attachment IDs in chat", async () => {
  const originalFetch = globalThis.fetch;
  const requests: Array<{ input: string; init?: RequestInit }> = [];
  globalThis.fetch = (async (input, init) => {
    requests.push({ input: String(input), init });
    return init?.method === "DELETE" ? new Response(null, { status: 204 }) : jsonResponse({ id: "asset-id" });
  }) as typeof fetch;
  try {
    const file = new File(["explicit-bytes"], "photo.png", { type: "image/png" });
    await uploadImage(file);
    const form = requests[0].init?.body;
    assert.ok(form instanceof FormData);
    assert.equal(form.get("persistence_mode"), "temporary");
    assert.equal(await (form.get("file") as File).text(), "explicit-bytes");
    assert.equal(new Headers(requests[0].init?.headers).get("content-type"), null);
    await deleteAsset("a/b");
    assert.equal(requests[1].input, "/api/assets/a%2Fb");
    assert.equal(assetContentUrl("a/b"), "/api/assets/a%2Fb/content");
    await sendChat({ conversation_id: null, message: "Look", asset_ids: ["asset-id"] });
    assert.deepEqual(JSON.parse(String(requests[2].init?.body)), { conversation_id: null, message: "Look", asset_ids: ["asset-id"] });
    assert.doesNotMatch(String(requests[2].init?.body), /explicit-bytes|photo\.png/);
  } finally { globalThis.fetch = originalFetch; }
});

test("capability client uses the exact safe route and chat carries only boolean consent", async () => {
  const originalFetch = globalThis.fetch;
  const requests: Array<{ input: string; init?: RequestInit }> = [];
  globalThis.fetch = (async (input, init) => {
    requests.push({ input: String(input), init });
    return String(input).endsWith("capabilities")
      ? jsonResponse({ vision: { enabled: true, scope: "remote" } })
      : jsonResponse({ conversation_id: "c", message_id: "m", response: "ok", metadata: {} });
  }) as typeof fetch;
  try {
    assert.deepEqual(await getCapabilities(), { vision: { enabled: true, scope: "remote" } });
    await sendChat({ conversation_id: null, message: "Look", asset_ids: ["asset-id"], allow_remote_vision: true });
  } finally { globalThis.fetch = originalFetch; }
  assert.equal(requests[0].input, "/api/capabilities");
  assert.deepEqual(JSON.parse(String(requests[1].init?.body)), {
    conversation_id: null, message: "Look", asset_ids: ["asset-id"], allow_remote_vision: true,
  });
});

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

test("memory list client uses active, search, category, and deleted routes", async () => {
  const originalFetch = globalThis.fetch;
  const requests: Array<{ input: string; init?: RequestInit }> = [];
  globalThis.fetch = (async (input, init) => {
    requests.push({ input: String(input), init });
    return jsonResponse([memory]);
  }) as typeof fetch;

  try {
    await listMemories();
    await listMemories({ query: "  SEARCH_CANARY + color  ", category: "profile", status: "deleted" });
    await listMemories({ category: "project" });
    await listMemories({ category: "instruction", status: "deleted" });
  } finally {
    globalThis.fetch = originalFetch;
  }

  assert.deepEqual(requests.map(({ input }) => input), [
    "/api/memory",
    "/api/memory/search",
    "/api/memory?category=project",
    "/api/memory?category=instruction&status=deleted",
  ]);
  assert.equal(requests[1].init?.method, "POST");
  assert.equal(new Headers(requests[1].init?.headers).get("content-type"), "application/json");
  assert.deepEqual(JSON.parse(String(requests[1].init?.body)), { query: "SEARCH_CANARY + color" });
  assert.ok(requests.every(({ input }) => !input.includes("SEARCH_CANARY")));
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
    await updateMemory("memory/one", { sensitivity: "remote_allowed" });
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
  assert.equal(requests[3].input, "/api/memory/memory%2Fone");
  assert.equal(requests[3].init?.method, "PATCH");
  assert.deepEqual(JSON.parse(String(requests[3].init?.body)), { sensitivity: "remote_allowed" });
  assert.ok(requests.every(({ input }) => !input.includes("remote_allowed")));
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
