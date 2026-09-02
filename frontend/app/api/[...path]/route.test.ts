import assert from "node:assert/strict";
import fs, { mkdtempSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { syncBuiltinESMExports } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test, { afterEach, beforeEach } from "node:test";

import { DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT } from "./route.ts";
import { MAX_REQUEST_BODY_BYTES, MAX_UPLOAD_BODY_BYTES, resolveProxyRoute } from "../../../lib/proxyPolicy.ts";

const LOCAL_TOKEN = "ab".repeat(32);
const ID = "a2466cb5-3b48-4efa-8fca-ae039e76886a";
const BROWSER = "http://127.0.0.1:3000";
const handlers = { GET, HEAD, POST, PATCH, DELETE, PUT, OPTIONS };
const originalFetch = globalThis.fetch;
const originalOrigin = process.env.AMITAI_API_ORIGIN;
const originalTokenFile = process.env.AMITAI_LOCAL_API_TOKEN_FILE;
const originalLegacyToken = process.env.AMITAI_LOCAL_API_TOKEN;
const originalAllowLan = process.env.AMITAI_ALLOW_LAN;
let privateDirectory = "";
let tokenFile = "";

function context(...path: string[]) {
  return { params: Promise.resolve({ path }) };
}

test("only exact asset upload route accepts bounded multipart with one supported image", async () => {
  const calls: Array<{ path: string; init?: RequestInit }> = [];
  globalThis.fetch = (async (input, init) => {
    calls.push({ path: String(input), init });
    return Response.json({ id: ID }, { status: 201 });
  }) as typeof fetch;
  const form = new FormData();
  form.set("file", new Blob(["image-bytes"], { type: "image/png" }), "photo.png");
  form.set("persistence_mode", "temporary");
  const request = () => new Request(`${BROWSER}/api/assets`, { method: "POST", headers: { Origin: BROWSER }, body: form });
  assert.equal((await POST(request(), context("assets"))).status, 201);
  assert.equal(calls[0].path, "http://127.0.0.1:8000/api/assets");
  assert.ok(calls[0].init?.body instanceof ArrayBuffer);
  const forwarded = new Headers(calls[0].init?.headers);
  assert.equal(forwarded.get("authorization"), `Bearer ${LOCAL_TOKEN}`);
  assert.match(forwarded.get("content-type") ?? "", /^multipart\/form-data;/);
  const body = await new Response(calls[0].init?.body, { headers: forwarded }).formData();
  assert.equal(await (body.get("file") as File).text(), "image-bytes");

  assert.equal((await POST(new Request(`${BROWSER}/api/chat`, { method: "POST", headers: { Origin: BROWSER }, body: form }), context("chat"))).status, 415);
  form.set("path", "C:/private.png");
  assert.equal((await POST(request(), context("assets"))).status, 400);
  form.delete("path");
  form.append("file", new Blob(["second"], { type: "image/png" }), "second.png");
  assert.equal((await POST(request(), context("assets"))).status, 400);
  assert.equal(calls.length, 1);
  assert.equal(resolveProxyRoute("POST", ["assets", "import-path"]), null);
  assert.equal(resolveProxyRoute("GET", ["assets"]), null);
  assert.equal(resolveProxyRoute("GET", ["assets", ID, "content"])?.body, "none");
});

test("upload rejection preserves origin, query, byte caps and token isolation", async () => {
  let called = false;
  globalThis.fetch = (async () => { called = true; return Response.json({}); }) as typeof fetch;
  const headers = { Origin: BROWSER, "Content-Type": "multipart/form-data; boundary=x" };
  const tooLarge = new Request(`${BROWSER}/api/assets`, { method: "POST", headers, body: new Uint8Array(MAX_UPLOAD_BODY_BYTES + 1) });
  assert.equal((await POST(tooLarge, context("assets"))).status, 413);
  assert.equal((await POST(new Request(`${BROWSER}/api/assets`, { method: "POST", headers: { ...headers, Origin: "https://evil.example" }, body: "private" }), context("assets"))).status, 403);
  assert.equal((await POST(new Request(`${BROWSER}/api/assets?path=private`, { method: "POST", headers, body: "private" }), context("assets"))).status, 400);
  assert.equal((await POST(new Request(`${BROWSER}/api/assets`, { method: "POST", headers, body: "invalid multipart" }), context("assets"))).status, 400);
  assert.equal(called, false);
});

beforeEach(() => {
  privateDirectory = mkdtempSync(join(tmpdir(), "amitai-token-"));
  tokenFile = join(privateDirectory, "local-api-token");
  writeFileSync(tokenFile, `${LOCAL_TOKEN}\n`, { encoding: "utf8", mode: 0o600 });
  process.env.AMITAI_API_ORIGIN = "http://127.0.0.1:8000";
  process.env.AMITAI_LOCAL_API_TOKEN_FILE = tokenFile;
  process.env.AMITAI_LOCAL_API_TOKEN = "legacy-token-must-never-be-used".repeat(2);
  delete process.env.AMITAI_ALLOW_LAN;
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  if (originalOrigin === undefined) delete process.env.AMITAI_API_ORIGIN;
  else process.env.AMITAI_API_ORIGIN = originalOrigin;
  if (originalTokenFile === undefined) delete process.env.AMITAI_LOCAL_API_TOKEN_FILE;
  else process.env.AMITAI_LOCAL_API_TOKEN_FILE = originalTokenFile;
  if (originalLegacyToken === undefined) delete process.env.AMITAI_LOCAL_API_TOKEN;
  else process.env.AMITAI_LOCAL_API_TOKEN = originalLegacyToken;
  if (originalAllowLan === undefined) delete process.env.AMITAI_ALLOW_LAN;
  else process.env.AMITAI_ALLOW_LAN = originalAllowLan;
  rmSync(privateDirectory, { recursive: true, force: true });
});

test("server proxy adds local auth without returning it to browser code", async () => {
  const upstreamRequests: Array<{ input: string; init?: RequestInit }> = [];
  globalThis.fetch = (async (input, init) => {
    upstreamRequests.push({ input: String(input), init });
    return Response.json([{ id: "conversation-1" }]);
  }) as typeof fetch;

  const response = await GET(
    new Request("http://127.0.0.1:3000/api/conversations", {
      headers: {
        Authorization: "Bearer browser-controlled-value",
        Cookie: "private-browser-cookie",
        "Proxy-Authorization": "attacker-proxy-credential",
        Forwarded: "host=attacker",
        "X-Forwarded-Host": "attacker",
        "X-Forwarded-For": "attacker",
        "X-Forwarded-Proto": "https",
        Host: "127.0.0.1:3000",
      },
    }),
    context("conversations"),
  );

  const upstreamRequest = upstreamRequests[0];
  assert.ok(upstreamRequest);
  assert.equal(upstreamRequest.input, "http://127.0.0.1:8000/api/conversations");
  const forwarded = new Headers(upstreamRequest.init?.headers);
  assert.equal(forwarded.get("authorization"), `Bearer ${LOCAL_TOKEN}`);
  assert.deepEqual([...forwarded.keys()], ["authorization"]);
  assert.equal(upstreamRequest.init?.body, undefined);
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
  assert.equal(response.headers.get("cache-control"), "no-store, no-transform");
  assert.equal(forwardedSignal, request.signal);

  release();
  const second = await reader.read();
  assert.equal(new TextDecoder().decode(second.value), "event: done\ndata: {}\n\n");
  assert.equal((await reader.read()).done, true);
});

test("server proxy fails closed without exposing a missing secret", async () => {
  unlinkSync(tokenFile);
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

test("server proxy rereads a replacement token file on every request", async () => {
  const replacement = "cd".repeat(32);
  const forwarded: string[] = [];
  globalThis.fetch = (async (_input, init) => {
    forwarded.push(new Headers(init?.headers).get("authorization") ?? "");
    return Response.json([]);
  }) as typeof fetch;

  const request = () =>
    GET(new Request("http://127.0.0.1:3000/api/conversations"), context("conversations"));
  await request();
  writeFileSync(tokenFile, `${replacement}\n`, { encoding: "utf8", mode: 0o600 });
  await request();

  assert.deepEqual(forwarded, [`Bearer ${LOCAL_TOKEN}`, `Bearer ${replacement}`]);
});

test("server proxy rejects a malformed runtime token file", async () => {
  writeFileSync(tokenFile, "not-a-valid-token\n", "utf8");
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

test("server proxy rejects a relative runtime token path", async () => {
  process.env.AMITAI_LOCAL_API_TOKEN_FILE = "relative/local-api-token";
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
    undefined,
    "https://untrusted.example",
    "http://127.0.0.1:4000",
    "http://localhost:3000",
    "null",
    "not-an-origin",
    `${BROWSER}/path`,
    `${BROWSER}/`,
    `${BROWSER}?secret=canary`,
    "http://user:password@127.0.0.1:3000",
    `${BROWSER}, https://attacker.example`,
  ];

  for (const handler of handlers) {
    for (const origin of rejectedOrigins) {
      const response = await handler(
        new Request("http://127.0.0.1:3000/api/memory/item-1", {
          method: handler.name,
          headers: {
            "Content-Type": "application/json",
            ...(origin === undefined ? {} : { Origin: origin }),
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

function browserRequest(method: string, path: string[], suffix = "", body?: string): Request {
  return new Request(`${BROWSER}/api/${path.join("/")}${suffix}`, {
    method,
    headers: { Origin: BROWSER, "Content-Type": "application/json" },
    body: body ?? (["POST", "PATCH"].includes(method) ? "{}" : undefined),
  });
}

const approved: Array<[string[], Array<keyof typeof handlers>]> = [
  [["health"], ["GET"]],
  [["conversations"], ["GET", "POST"]],
  [["conversations", ID], ["GET", "PATCH", "DELETE"]],
  [["chat"], ["POST"]],
  [["chat", "stream"], ["POST"]],
  [["memory"], ["GET", "POST"]],
  [["memory", "search"], ["POST"]],
  [["memory", ID], ["PATCH", "DELETE"]],
];

test("only explicit route/method pairs can reach upstream; HEAD and OPTIONS fail closed", async () => {
  let calls = 0;
  globalThis.fetch = (async (input, init) => {
    calls += 1;
    assert.match(String(input), /^http:\/\/127\.0\.0\.1:8000\/api\//);
    assert.equal(init?.redirect, "manual");
    assert.equal(init?.cache, "no-store");
    return Response.json({ ok: true });
  }) as typeof fetch;
  for (const [path, allowed] of approved) {
    for (const method of Object.keys(handlers) as Array<keyof typeof handlers>) {
      const before = calls;
      const response = await handlers[method](browserRequest(method, path), context(...path));
      assert.equal(response.status, allowed.includes(method) ? 200 : 404, `${method} ${path}`);
      assert.equal(calls - before, allowed.includes(method) ? 1 : 0);
      assert.equal(response.headers.get("cache-control"), "no-store");
    }
  }
  assert.equal(resolveProxyRoute("toString", ["memory"]), null);
});

test("hostile paths and unknown routes never read the token or fetch upstream", async (t) => {
  let reads = 0;
  const read = fs.readFileSync;
  const mockedRead = t.mock.method(fs, "readFileSync", (...args: Parameters<typeof read>) => {
    reads += 1;
    return read(...args);
  });
  syncBuiltinESMExports();
  globalThis.fetch = (async () => { assert.fail("must not fetch"); }) as typeof fetch;
  const hostile = [
    [], ["admin"], ["docs"], ["openapi.json"], ["v1", "generate"], ["memory", "search", "extra"],
    ...["", ".", "..", "../health", "\\health", "a/b", "%2f", "%5c", "%252f", "%2e%2e",
      "%", "%GG", "a\u0000b", "a\nb", "x".repeat(4000), "not-a-uuid", `${ID}.json`,
      ID.replace(/-/g, ""), `{${ID}}`].map((id) => ["conversations", id]),
    ["memory", "non-uuid"], ["health", ""], ["%68ealth"],
  ];
  try {
    for (const path of hostile) {
      // Route params are tested directly, including forms the URL parser normalizes.
      const response = await GET(new Request(`${BROWSER}/api/invalid`), context(...path));
      assert.equal(response.status, 404, JSON.stringify(path));
      assert.equal(response.headers.get("cache-control"), "no-store");
    }
    for (const encoded of ["/api/%6demory", "/api//memory", "/api/memory/"]) {
      assert.equal((await GET(new Request(`${BROWSER}${encoded}`), context("memory"))).status, 404);
    }
    assert.equal((await GET(new Request(`${BROWSER}/api/conversations/%61${ID.slice(1)}`), context("conversations", ID))).status, 404);
    assert.equal(reads, 0);
  } finally {
    mockedRead.mock.restore();
    syncBuiltinESMExports();
  }
});

test("loopback host and fetch metadata checks apply to reads and mutations even with LAN opted in", async () => {
  process.env.AMITAI_ALLOW_LAN = "1";
  unlinkSync(tokenFile);
  globalThis.fetch = (async () => { assert.fail("must not fetch"); }) as typeof fetch;
  for (const method of ["GET", "POST", "PATCH", "DELETE"] as const) {
    for (const host of ["192.168.1.10", "attacker.example", "0.0.0.0"]) {
      const origin = `http://${host}:3000`;
      const response = await handlers[method](new Request(`${origin}/api/memory`, {
        method, headers: { Origin: origin },
      }), context("memory"));
      assert.equal(response.status, 403);
    }
    const response = await handlers[method](new Request(`${BROWSER}/api/memory`, {
      method, headers: { Origin: BROWSER, "Sec-Fetch-Site": "cross-site" },
    }), context("memory"));
    assert.equal(response.status, 403);
    for (const host of ["attacker.example:3000", "localhost:3000", "127.0.0.1:4000"]) {
      const rebinding = await handlers[method](new Request(`${BROWSER}/api/memory`, {
        method, headers: { Origin: BROWSER, Host: host },
      }), context("memory"));
      assert.equal(rebinding.status, 403);
    }
  }
});

test("matching loopback origins and non-cross-site fetch metadata are accepted", async () => {
  globalThis.fetch = (async () => Response.json([])) as typeof fetch;
  for (const origin of [BROWSER, "http://localhost:3000", "http://[::1]:3000"]) {
    for (const site of [undefined, "same-origin", "same-site", "none"]) {
      const response = await POST(new Request(`${origin}/api/conversations`, {
        method: "POST",
        headers: { Origin: origin, ...(site ? { "Sec-Fetch-Site": site } : {}) },
      }), context("conversations"));
      assert.equal(response.status, 200);
    }
  }
});

test("query allowlist rejects URL canaries, duplicates and unknown keys before token access", async () => {
  const canary = "PRIVATE_SEARCH_CANARY_998877";
  unlinkSync(tokenFile);
  globalThis.fetch = (async () => { assert.fail("must not fetch"); }) as typeof fetch;
  for (const [path, methods] of approved) {
    for (const method of methods) {
      const response = await handlers[method](browserRequest(method, path, `?query=${canary}`), context(...path));
      assert.equal(response.status, 400);
      assert.doesNotMatch(await response.text(), new RegExp(canary));
    }
  }
  for (const query of [
    "status=active&status=active", "category=profile&category=project", "category=invalid",
    "status=stale", "category=", "status=", "q=private", "value=private", "debug=1", "limit=5",
    "category=project&secret=canary", "status=active&%73tatus=deleted",
  ]) {
    assert.equal((await GET(browserRequest("GET", ["memory"], `?${query}`), context("memory"))).status, 400);
  }
});

test("all rejection stages run before any runtime token-file read", async (t) => {
  let reads = 0;
  const mockedRead = t.mock.method(fs, "readFileSync", () => {
    reads += 1;
    return `${LOCAL_TOKEN}\n`;
  });
  syncBuiltinESMExports();
  globalThis.fetch = (async () => { assert.fail("must not fetch"); }) as typeof fetch;
  try {
    const attempts: Array<[Request, string[], number]> = [
      [new Request(`${BROWSER}/api/chat`, { method: "POST" }), ["chat"], 403],
      [new Request(`${BROWSER}/api/memory`, { headers: { "Sec-Fetch-Site": "cross-site" } }), ["memory"], 403],
      [browserRequest("GET", ["admin"]), ["admin"], 404],
      [browserRequest("GET", ["memory"], "?query=PRIVATE"), ["memory"], 400],
      [browserRequest("POST", ["chat"], "", "{invalid"), ["chat"], 400],
      [new Request(`${BROWSER}/api/chat`, { method: "POST", headers: { Origin: BROWSER }, body: "{}" }), ["chat"], 415],
      [browserRequest("POST", ["chat"], "", "x".repeat(MAX_REQUEST_BODY_BYTES + 1)), ["chat"], 413],
    ];
    for (const [request, path, status] of attempts) {
      const response = await handlers[request.method as keyof typeof handlers](request, context(...path));
      assert.equal(response.status, status);
    }
    assert.equal(reads, 0);
  } finally {
    mockedRead.mock.restore();
    syncBuiltinESMExports();
  }
});

test("only memory filter queries are reconstructed and search is exclusively a JSON body", async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  globalThis.fetch = (async (input, init) => {
    calls.push({ url: String(input), init });
    return Response.json([]);
  }) as typeof fetch;
  for (const category of ["preference", "profile", "project", "workflow", "instruction"]) {
    for (const status of ["active", "deleted"]) {
      const suffix = `?category=${category}&status=${status}`;
      const response = await GET(browserRequest("GET", ["memory"], suffix), context("memory"));
      assert.equal(response.status, 200);
      assert.equal(calls.at(-1)?.url, `http://127.0.0.1:8000/api/memory${suffix}`);
    }
  }
  const canary = "SEARCH_BODY_ONLY_CANARY_928374";
  const response = await POST(browserRequest("POST", ["memory", "search"], "", JSON.stringify({ query: canary })), context("memory", "search"));
  assert.equal(response.status, 200);
  assert.equal(calls.at(-1)?.url, "http://127.0.0.1:8000/api/memory/search");
  assert.deepEqual(JSON.parse(String(calls.at(-1)?.init?.body)), { query: canary });
  assert.ok(calls.every(({ url }) => !url.includes(canary)));
});

test("JSON route bodies are required, object-only and content-type checked before configuration", async () => {
  unlinkSync(tokenFile);
  globalThis.fetch = (async () => { assert.fail("must not fetch"); }) as typeof fetch;
  const jsonRoutes = approved.flatMap(([path, methods]) => methods
    .filter((method) => ["POST", "PATCH"].includes(method))
    .map((method) => ({ path, method })));
  for (const { path, method } of jsonRoutes) {
    for (const body of ["{bad", "[]", "null", "42", '"string"', ""]) {
      assert.equal((await handlers[method](browserRequest(method, path, "", body), context(...path))).status, 400);
    }
    for (const type of [undefined, "text/plain", "application/x-www-form-urlencoded", "multipart/form-data", "application/jsonp"]) {
      const response = await handlers[method](new Request(`${BROWSER}/api/${path.join("/")}`, {
        method, headers: { Origin: BROWSER, ...(type ? { "Content-Type": type } : {}) }, body: "{}",
      }), context(...path));
      assert.equal(response.status, 415);
    }
  }
  for (const [path, method] of [[["chat"], "POST"], [["memory", ID], "PATCH"]] as const) {
    const response = await handlers[method](new Request(`${BROWSER}/api/${path.join("/")}`, {
      method, headers: { Origin: BROWSER, "Content-Type": "application/json" },
    }), context(...path));
    assert.equal(response.status, 400);
  }
});

test("actual body bytes are bounded regardless of Content-Length, with producer cancellation", async () => {
  globalThis.fetch = (async () => Response.json({ ok: true })) as typeof fetch;
  const body = JSON.stringify({ message: "x".repeat(MAX_REQUEST_BODY_BYTES - 14) });
  assert.equal(new TextEncoder().encode(body).byteLength, MAX_REQUEST_BODY_BYTES);
  assert.equal((await POST(browserRequest("POST", ["chat"], "", body), context("chat"))).status, 200);
  for (const contentLength of [undefined, "1"]) {
    let cancelled = false;
    const request = new Request(`${BROWSER}/api/chat`, {
      method: "POST", duplex: "half",
      headers: { Origin: BROWSER, "Content-Type": "application/json", ...(contentLength ? { "Content-Length": contentLength } : {}) },
      body: new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(body));
          controller.enqueue(new TextEncoder().encode(" "));
        },
        cancel() { cancelled = true; },
      }),
    } as RequestInit);
    const response = await POST(request, context("chat"));
    assert.equal(response.status, 413);
    assert.equal(cancelled, true);
    assert.equal(response.headers.get("cache-control"), "no-store");
  }
  const multiByte = JSON.stringify({ message: "😀".repeat(MAX_REQUEST_BODY_BYTES / 3) });
  assert.equal((await POST(browserRequest("POST", ["chat"], "", multiByte), context("chat"))).status, 413);
});

test("optional create body and JSON charset work; reads/deletes reject bodies", async () => {
  globalThis.fetch = (async () => Response.json({ ok: true })) as typeof fetch;
  assert.equal((await POST(new Request(`${BROWSER}/api/conversations`, {
    method: "POST", headers: { Origin: BROWSER },
  }), context("conversations"))).status, 200);
  assert.equal((await POST(new Request(`${BROWSER}/api/chat`, {
    method: "POST", headers: { Origin: BROWSER, "Content-Type": "application/json; charset=utf-8" }, body: "{}",
  }), context("chat"))).status, 200);
  for (const method of ["GET", "DELETE"] as const) {
    const path = method === "GET" ? ["memory"] : ["memory", ID];
    const request = browserRequest(method, path);
    // Fetch's constructor forbids GET bodies; exercise the server-facing contract too.
    Object.defineProperty(request, "body", { value: new ReadableStream({
      start(controller) { controller.enqueue(new TextEncoder().encode("{}")); controller.close(); },
    }) });
    assert.equal((await handlers[method](request, context(...path))).status, 400);
  }
});

test("Next-style empty request streams permit bodyless conversation creation and deletion", async () => {
  globalThis.fetch = (async (_input, init) => {
    assert.equal(init?.body, undefined);
    return Response.json({ ok: true });
  }) as typeof fetch;
  for (const method of ["POST", "DELETE"] as const) {
    const path = method === "POST" ? ["conversations"] : ["memory", ID];
    const request = new Request(`${BROWSER}/api/${path.join("/")}`, {
      method, headers: { Origin: BROWSER, "Content-Length": "0" }, duplex: "half",
      body: new ReadableStream({ start(controller) { controller.close(); } }),
    } as RequestInit);
    assert.equal((await handlers[method](request, context(...path))).status, 200);
  }
  const inaccessibleGetBody = new Request(`${BROWSER}/api/memory`, { headers: { "Content-Length": "2" } });
  assert.equal((await GET(inaccessibleGetBody, context("memory"))).status, 400);
});

test("response headers are allowlisted and all backend/error responses are no-store", async () => {
  for (const status of [200, 401, 403, 404, 413, 415, 422, 503]) {
    globalThis.fetch = (async () => Response.json({ detail: "safe" }, { status, headers: {
      "Cache-Control": "public, max-age=3600", "Set-Cookie": "secret=1", Server: "private",
      Via: "private", "X-Powered-By": "private", Authorization: LOCAL_TOKEN,
      "WWW-Authenticate": "private", "X-Debug": "private", "Access-Control-Allow-Origin": "*",
    } })) as typeof fetch;
    const response = await GET(browserRequest("GET", ["memory"]), context("memory"));
    assert.equal(response.status, status);
    assert.deepEqual([...response.headers.keys()].sort(), ["cache-control", "content-type"]);
    assert.equal(response.headers.get("cache-control"), "no-store");
  }
  globalThis.fetch = (async () => { throw new Error(`secret ${tokenFile} ${LOCAL_TOKEN}`); }) as typeof fetch;
  const failed = await GET(browserRequest("GET", ["memory"]), context("memory"));
  assert.equal(failed.status, 502);
  assert.equal(await failed.text(), '{"detail":"Local AmitAI backend is unavailable"}');
  assert.equal(failed.headers.get("cache-control"), "no-store");
});

test("request cancellation propagates to the upstream streaming fetch", async () => {
  const abort = new AbortController();
  let signal: AbortSignal | null | undefined;
  globalThis.fetch = (async (_input, init) => {
    signal = init?.signal;
    return new Response(new ReadableStream(), { headers: { "Content-Type": "text/event-stream" } });
  }) as typeof fetch;
  const request = new Request(`${BROWSER}/api/chat/stream`, {
    method: "POST", signal: abort.signal,
    headers: { Origin: BROWSER, "Content-Type": "application/json" }, body: "{}",
  });
  const response = await POST(request, context("chat", "stream"));
  assert.equal(signal, request.signal);
  abort.abort();
  assert.equal(signal?.aborted, true);
  await response.body?.cancel();
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
