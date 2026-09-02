import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";
import { JSDOM } from "jsdom";
import type { Root } from "react-dom/client";
import type { UploadedAsset } from "../lib/types.ts";

const dom = new JSDOM("<!doctype html><html><body></body></html>", { url: "http://localhost" });
for (const [name, value] of Object.entries({
  window: dom.window, document: dom.window.document, navigator: dom.window.navigator,
  HTMLElement: dom.window.HTMLElement, HTMLInputElement: dom.window.HTMLInputElement,
  HTMLTextAreaElement: dom.window.HTMLTextAreaElement,
  localStorage: dom.window.localStorage, sessionStorage: dom.window.sessionStorage,
  IS_REACT_ACT_ENVIRONMENT: true,
})) Object.defineProperty(globalThis, name, { configurable: true, value });
dom.window.HTMLElement.prototype.scrollIntoView = () => undefined;

const { act } = await import("react");
const { createRoot } = await import("react-dom/client");
const { Composer } = await import("./Composer.tsx");
const { Message } = await import("./Message.tsx");

const ASSET: UploadedAsset = {
  id: "a2466cb5-3b48-4efa-8fca-ae039e76886a", kind: "image",
  original_filename: "photo.png", content_type: "image/png", byte_size: 123,
  width: 12, height: 8, sha256: "a".repeat(64), created_at: "2026-09-02T00:00:00Z",
  conversation_id: null, persistence_mode: "temporary", processing_scope: "local_only",
};
let container: HTMLDivElement;
let root: Root;
let originalFetch: typeof fetch;

async function settle(): Promise<void> {
  await act(async () => new Promise((resolve) => setTimeout(resolve, 20)));
}

beforeEach(() => {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  originalFetch = globalThis.fetch;
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  globalThis.fetch = originalFetch;
});

test("explicit select uploads, previews, attaches and clears without web storage", async () => {
  const calls: Array<{ path: string; init?: RequestInit }> = [];
  globalThis.fetch = (async (input, init) => {
    calls.push({ path: String(input), init });
    return Response.json(ASSET, { status: 201, headers: { "Content-Type": "application/json" } });
  }) as typeof fetch;
  let sent: { text: string; ids: string[] } | null = null;
  await act(async () => root.render(<Composer enterToSend onSend={(text, assets) => {
    sent = { text, ids: (assets ?? []).map((asset) => asset.id) };
  }} />));
  const input = container.querySelector('input[type="file"]') as HTMLInputElement;
  const file = new Blob(["real-png-bytes"], { type: "image/png" });
  Object.defineProperty(file, "name", { value: "photo.png" });
  Object.defineProperty(input, "files", { configurable: true, value: [file] });
  await act(async () => input.dispatchEvent(new dom.window.Event("change", { bubbles: true })));
  await settle();
  assert.equal(calls[0].path, "/api/assets");
  assert.ok(calls[0].init?.body instanceof FormData);
  assert.equal(container.querySelector("img")?.getAttribute("src"), `/api/assets/${ASSET.id}/content`);
  assert.match(container.textContent ?? "", /photo\.png[\s\S]*Local only/);
  assert.match(container.textContent ?? "", /Vision requires a local model; one image per message/);
  assert.doesNotMatch(container.textContent ?? "", /analysis is not enabled/);

  const textarea = container.querySelector("textarea")!;
  const setter = Object.getOwnPropertyDescriptor(dom.window.HTMLTextAreaElement.prototype, "value")!.set!;
  await act(async () => {
    setter.call(textarea, "What is shown?");
    textarea.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
    textarea.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
  });
  await act(async () => (container.querySelector('button[type="submit"]') as HTMLButtonElement).click());
  assert.deepEqual(sent, { text: "What is shown?", ids: [ASSET.id] });
  assert.equal(container.querySelector("img"), null);
  assert.equal(localStorage.length, 0);
  assert.equal(sessionStorage.length, 0);
});

test("attachment can be removed and persisted history renders backend preview", async () => {
  const methods: string[] = [];
  globalThis.fetch = (async (_input, init) => {
    methods.push(init?.method ?? "GET");
    return init?.method === "DELETE"
      ? new Response(null, { status: 204 })
      : Response.json(ASSET, { status: 201, headers: { "Content-Type": "application/json" } });
  }) as typeof fetch;
  await act(async () => root.render(<Composer enterToSend onSend={() => undefined} />));
  const input = container.querySelector('input[type="file"]') as HTMLInputElement;
  const file = new Blob(["bytes"], { type: "image/png" });
  Object.defineProperty(file, "name", { value: "photo.png" });
  Object.defineProperty(input, "files", { configurable: true, value: [file] });
  await act(async () => input.dispatchEvent(new dom.window.Event("change", { bubbles: true })));
  await settle();
  await act(async () => (container.querySelector("button[aria-label^='Remove']") as HTMLButtonElement).click());
  await settle();
  assert.deepEqual(methods, ["POST", "DELETE"]);
  assert.equal(container.querySelector("img"), null);

  await act(async () => root.render(<Message message={{
    id: "m", conversation_id: "c", role: "user", content: "Attached",
    created_at: "2026-09-02T00:00:00Z", metadata: null,
    assets: [{ ...ASSET, conversation_id: "c", persistence_mode: "conversation" }],
  }} showTimestamp={false} wrapCode={false} />));
  assert.equal(container.querySelector("img")?.getAttribute("src"), `/api/assets/${ASSET.id}/content`);
});
