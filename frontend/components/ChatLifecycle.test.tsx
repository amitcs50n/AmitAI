import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";
import { JSDOM } from "jsdom";
import type { Root } from "react-dom/client";
import type { ChatRequest, ChatResponse, ConversationDetail, InferenceMode, Message, UploadedAsset } from "../lib/types.ts";

// React's event system must see the DOM before react-dom is imported.
const dom = new JSDOM("<!doctype html><html><body></body></html>", { url: "http://localhost" });
for (const [name, value] of Object.entries({
  window: dom.window, document: dom.window.document, navigator: dom.window.navigator,
  HTMLElement: dom.window.HTMLElement, HTMLInputElement: dom.window.HTMLInputElement,
  HTMLTextAreaElement: dom.window.HTMLTextAreaElement,
  localStorage: dom.window.localStorage, sessionStorage: dom.window.sessionStorage,
  IS_REACT_ACT_ENVIRONMENT: true,
})) Object.defineProperty(globalThis, name, { configurable: true, value });

const { act } = await import("react");
const { createRoot } = await import("react-dom/client");
const { AmitaiApp } = await import("./AmitaiApp.tsx");
const { ChatView } = await import("./ChatView.tsx");
const { Composer } = await import("./Composer.tsx");
const { DEFAULT_PREFERENCES } = await import("../lib/types.ts");

const NOW = "2026-09-03T00:00:00Z";
const QUESTION = "PRIVATE_USER_CANARY";
const ANSWER = "PRIVATE_ANSWER_CANARY";
const PARTIAL = "PRIVATE_PARTIAL_CANARY";
const metadata: ChatResponse["metadata"] = {
  model: "fake", latency_ms: 10, input_tokens: 4, output_tokens: 3,
  validator: { retry_attempted: false, retry_passed: null }, tools: [], memory: [],
};
function message(id: string, role: string, content: string): Message {
  return { id, role, content, conversation_id: "a", created_at: NOW, metadata: role === "assistant" ? metadata : null };
}
function detail(id = "a", messages: Message[] = []): ConversationDetail {
  return { id, title: `Conversation ${id}`, created_at: NOW, updated_at: NOW, archived: false, messages };
}
function final(response = ANSWER, conversationId = "a"): ChatResponse {
  return { conversation_id: conversationId, message_id: "saved-assistant", response, metadata };
}
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

class ControlledStream {
  controller!: ReadableStreamDefaultController<Uint8Array>;
  cancelled = false;
  closed = false;
  readonly response: Response;
  constructor(readonly signal: AbortSignal, readonly payload: ChatRequest) {
    // Deliberately do NOT wire signal into the source: the SSE reader must
    // cancel a pending read too, not rely on fetch mocks to clean it up.
    const body = new ReadableStream<Uint8Array>({
      start: (controller) => { this.controller = controller; },
      cancel: () => { this.cancelled = true; },
    });
    this.response = new Response(body, { headers: { "Content-Type": "text/event-stream" } });
  }
  raw(data: string) { this.controller.enqueue(new TextEncoder().encode(data)); }
  event(event: string, data: unknown) { this.raw(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`); }
  start() { this.event("start", { conversation_id: this.payload.conversation_id }); }
  text(delta: string) { this.event("text", { delta }); }
  close() { this.closed = true; this.controller.close(); }
}

let container: HTMLDivElement;
let root: Root;
let mounted: boolean;
let originalFetch: typeof fetch;
let streams: ControlledStream[];
let list: ConversationDetail[];
let remote: boolean;
let inferenceMode: InferenceMode;
let getHistory: (id: string) => Promise<Response>;
let fetchChat: ((init: RequestInit) => Promise<Response>) | null;
const ASSET: UploadedAsset = {
  id: "test-image", kind: "image", original_filename: "photo.png", content_type: "image/png",
  byte_size: 12, width: 2, height: 2, sha256: "a".repeat(64), created_at: NOW,
  conversation_id: null, persistence_mode: "temporary", processing_scope: "local_only",
};

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  mounted = true;
  streams = [];
  list = [];
  remote = false;
  inferenceMode = "local";
  fetchChat = null;
  getHistory = async (id) => Response.json(detail(id, [message("saved-user", "user", QUESTION), message("saved-assistant", "assistant", ANSWER)]));
  originalFetch = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    const path = String(input);
    if (path === "/api/capabilities") return Response.json({ vision: { enabled: inferenceMode !== "mock", scope: remote ? "remote" : "local" }, inference: { mode: inferenceMode } });
    if (path === "/api/conversations") return Response.json(list);
    if (path.startsWith("/api/conversations/")) return getHistory(path.split("/").at(-1)!);
    if (path === "/api/assets") return Response.json(ASSET, { status: 201 });
    if (path === "/api/chat/stream") {
      if (fetchChat) return fetchChat(init!);
      assert.ok(init?.signal);
      assert.equal(init.method, "POST");
      const stream = new ControlledStream(init.signal, JSON.parse(String(init.body)) as ChatRequest);
      streams.push(stream);
      assert.equal(streams.filter((item) => !item.signal.aborted && !item.cancelled && !item.closed).length, 1, "at most one live request");
      return stream.response;
    }
    throw new Error(`Unexpected test route: ${path}`);
  };
});

afterEach(async () => {
  if (mounted) await act(async () => root.unmount());
  container.remove();
  globalThis.fetch = originalFetch;
  Reflect.deleteProperty(dom.window, "requestAnimationFrame");
  Reflect.deleteProperty(dom.window, "cancelAnimationFrame");
});

async function settle() { await act(async () => { await new Promise((resolve) => setTimeout(resolve, 25)); }); }
async function boot() { await act(async () => root.render(<AmitaiApp />)); await settle(); }
function button(name: string): HTMLButtonElement {
  const found = [...container.querySelectorAll("button")].find((element) => element.getAttribute("aria-label") === name || element.textContent?.trim() === name);
  assert.ok(found, `missing button ${name}`);
  return found;
}
async function click(name: string) { await act(async () => button(name).click()); }
async function type(text: string) {
  const input = container.querySelector("textarea")!;
  assert.ok(input);
  await act(async () => {
    Object.getOwnPropertyDescriptor(dom.window.HTMLTextAreaElement.prototype, "value")!.set!.call(input, text);
    input.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
  });
}
async function send(text = QUESTION) { await type(text); await click("Send message"); }
function articles() { return [...container.querySelectorAll("article")].map((element) => element.textContent ?? ""); }
function assertPrivateStorage() {
  const serialized = JSON.stringify({ local: { ...localStorage }, session: { ...sessionStorage } });
  for (const canary of [QUESTION, ANSWER, PARTIAL]) assert.ok(!serialized.includes(canary));
  assert.equal(sessionStorage.length, 0);
  assert.ok(Object.keys(localStorage).every((key) => ["amitai-selected-conversation", "amitai-ui-preferences-v1"].includes(key)));
}
async function complete(stream: ControlledStream, response = ANSWER) {
  await act(async () => { stream.text(response); stream.event("final", final(response)); stream.event("done", {}); stream.close(); });
  await settle();
}

for (const [detail, expected, status] of [
  ["Remote inference blocked by local privacy policy", /blocked by your local privacy policy/, 422],
  ["Remote vision disclosure is not enabled", /Remote image sharing is not enabled/, 403],
  ["Local API proxy is not configured", /local API connection is not configured/, 503],
  ["Local AmitAI backend is unavailable", /local Aevon backend is unavailable/, 502],
  ["PRIVATE_INTERNAL_ERROR_CANARY", /Generation failed/, 500],
] as const) {
  for (const streaming of [false, true]) test(`safe error UI for ${detail} (stream=${streaming})`, async () => {
    if (!streaming) fetchChat = async () => Response.json({ detail }, { status });
    await boot();
    await send();
    if (streaming) await act(async () => { streams[0].start(); streams[0].event("error", { detail }); streams[0].close(); });
    await settle();
    assert.match(container.textContent!, expected);
    assert.doesNotMatch(container.textContent!, /PRIVATE_INTERNAL_ERROR_CANARY/);
    assert.equal(button("Retry").disabled, false);
    assert.equal(articles().length, 1);
    if (!streaming && (status === 502 || status === 503)) assert.match(container.textContent!, /Disconnected/);
  });
}

for (const [mode, label] of [["mock", "Mock · no model inference"], ["local", "Local inference"], ["remote", "Remote inference"], ["unknown", "Unavailable"]] as const) {
  test(`Settings shows server inference mode ${mode}`, async () => {
    inferenceMode = mode;
    remote = mode === "remote";
    await boot();
    await click("Settings");
    assert.match(container.textContent!, new RegExp(label));
    assert.match(container.textContent!, /Calculator/);
    assert.doesNotMatch(container.textContent!, /GitHub|WebNot configured|FilesNot configured|PythonNot configured/);
    inferenceMode = "unknown";
    await click("Refresh capabilities");
    await settle();
    assert.match(container.textContent!, /Configured inferenceUnavailable/);
  });
}

test("real app: pending -> start -> incremental deltas -> final -> done -> persisted history", async () => {
  const persisted = deferred<Response>();
  getHistory = () => persisted.promise;
  localStorage.setItem("amitai-ui-preferences-v1", JSON.stringify({ showTimestamps: true }));
  await boot();
  await send();
  assert.deepEqual(articles().map((text) => text.includes(QUESTION)), [true]);
  assert.equal(button("Stop generation").disabled, false);
  assert.ok(container.querySelector('[aria-label="Aevon is responding"]'));
  const stream = streams[0];
  await act(async () => stream.start());
  assert.match(container.textContent!, /Connected/);
  await act(async () => stream.text("PRIVATE_"));
  assert.equal(container.querySelector('[aria-label="Aevon is responding"]'), null);
  assert.match(articles()[1], /PRIVATE_/);
  const status = container.querySelector('[role="status"]')!;
  const announced = status.textContent;
  await act(async () => stream.text("ANSWER_CANARY"));
  assert.equal(status.textContent, announced, "no per-delta screen-reader announcement");
  assert.equal(container.querySelector("article [aria-live]"), null);
  await act(async () => stream.event("final", final()));
  assert.equal(container.querySelector('[aria-label="Stop generation"]'), null, "final is already committed");
  await act(async () => { stream.event("done", {}); stream.close(); });
  assert.equal(articles().length, 2, "fallback replaces rather than duplicates transients");
  assert.match(articles()[1], new RegExp(ANSWER));
  await act(async () => persisted.resolve(Response.json(detail("a", [message("saved-user", "user", QUESTION), message("saved-assistant", "assistant", ANSWER)]))));
  assert.equal(container.querySelectorAll(`time[datetime="${NOW}"]`).length, 2, "server history replaced optimistic timestamps");
  assert.equal(container.querySelector("textarea")!.disabled, false);
  assertPrivateStorage();
});

for (const partial of [false, true]) test(`Stop ${partial ? "after partial text" : "before first text"} is neutral and retry reuses exactly one user turn`, async () => {
  await boot();
  await send();
  const stream = streams[0];
  await act(async () => { stream.start(); if (partial) stream.text(PARTIAL); });
  button("Stop generation").focus();
  assert.equal(document.activeElement, button("Stop generation"));
  await click("Stop generation");
  assert.equal(stream.signal.aborted, true);
  assert.equal(stream.cancelled, true, "pending reader was cancelled");
  assert.equal(articles().length, 1);
  assert.ok(articles()[0].includes(QUESTION));
  assert.doesNotMatch(container.textContent!, /PRIVATE_PARTIAL_CANARY|Generation failed|Disconnected/);
  assert.match(container.textContent!, /Generation stopped/);
  assert.equal(container.querySelector("textarea")!.disabled, false);
  assert.equal(document.activeElement, container.querySelector("textarea"));
  assert.equal(streams.length, 1, "no automatic retry");
  await click("Retry");
  assert.equal(streams.length, 2);
  assert.deepEqual(streams[1].payload, stream.payload);
  assert.equal(articles().length, 1);
  await act(async () => streams[1].start());
  await complete(streams[1]);
  assert.equal(articles().length, 2);
  assert.equal(articles().filter((text) => text.includes(QUESTION)).length, 1);
  assertPrivateStorage();
});

test("synchronous duplicate submits are refused; an aborted late fetch cannot overwrite the next stream", async () => {
  const late = deferred<Response>();
  let oldSignal!: AbortSignal;
  fetchChat = async (init) => { oldSignal = init.signal!; return late.promise; };
  await boot();
  await send();
  await click("Stop generation");
  assert.equal(oldSignal.aborted, true);
  fetchChat = null;
  await type("Fresh question");
  const form = container.querySelector("form")!;
  await act(async () => {
    form.dispatchEvent(new dom.window.Event("submit", { bubbles: true, cancelable: true }));
    form.dispatchEvent(new dom.window.Event("submit", { bubbles: true, cancelable: true }));
  });
  assert.equal(streams.length, 1);
  const discarded = new ControlledStream(oldSignal, { conversation_id: null, message: QUESTION });
  discarded.start(); discarded.text(PARTIAL); discarded.event("final", final(PARTIAL)); discarded.event("done", {});
  await act(async () => late.resolve(discarded.response));
  assert.equal(discarded.cancelled, true);
  assert.equal(streams[0].signal.aborted, false);
  assert.equal(button("Stop generation").disabled, false, "old finally cannot clear new sending state");
  assert.doesNotMatch(container.textContent!, new RegExp(PARTIAL));
  assert.deepEqual(articles(), ["Fresh question"]);
  await click("Stop generation");
});

test("unmount aborts the active fetch and releases a quiet stream reader", async () => {
  await boot(); await send();
  await act(async () => { streams[0].start(); streams[0].text(PARTIAL); });
  await act(async () => root.unmount()); mounted = false;
  assert.equal(streams[0].signal.aborted, true);
  assert.equal(streams[0].cancelled, true);
  assert.equal(container.textContent, "");
  assertPrivateStorage();
});

test("network failure is distinct from Stop and retains a retryable pending user", async () => {
  fetchChat = async () => { throw new TypeError("network unavailable"); };
  await boot(); await send();
  assert.match(container.textContent!, /Unable to connect to Aevon/);
  assert.match(container.textContent!, /Disconnected/);
  assert.doesNotMatch(container.textContent!, /Generation stopped/);
  assert.equal(button("Retry").disabled, false);
  assert.deepEqual(articles(), [QUESTION]);
  assert.equal(container.querySelector("textarea")!.disabled, false);
});

for (const failure of ["malformed", "incomplete", "server", "body-network"] as const) test(`${failure} stream removes partial output and reports a normal failure`, async () => {
  await boot(); await send();
  const stream = streams[0];
  await act(async () => { stream.start(); stream.text(PARTIAL); });
  assert.match(articles()[1], new RegExp(PARTIAL));
  await act(async () => {
    if (failure === "malformed") stream.raw("event: text\ndata: {invalid-json}\n\n");
    else if (failure === "server") stream.event("error", { detail: "Assistant generation failed" });
    else if (failure === "body-network") stream.controller.error(new TypeError("connection lost"));
    else stream.close();
  });
  assert.match(container.textContent!, /Generation failed\. Try again/);
  assert.doesNotMatch(container.textContent!, /Generation stopped/);
  assert.deepEqual(articles(), [QUESTION]);
  assert.equal(button("Retry").disabled, false);
  assertPrivateStorage();
});

test("history reload failure keeps the saved fallback and offers reload, not generation retry", async () => {
  getHistory = async () => { throw new TypeError("offline"); };
  await boot(); await send();
  await act(async () => streams[0].start());
  await complete(streams[0]);
  assert.equal(articles().length, 2);
  assert.match(articles()[1], new RegExp(ANSWER));
  assert.match(container.textContent!, /response was saved, but persisted history could not be reloaded/);
  assert.doesNotMatch(container.textContent!, /Generation failed/);
  assert.equal(container.querySelector("textarea")!.disabled, false);
  getHistory = async () => Response.json(detail("a"));
  await click("Retry");
  assert.equal(streams.length, 1, "reload never regenerates a saved response");
});

test("final without done preserves confirmed saved output but reports stream failure without resend", async () => {
  await boot(); await send();
  await act(async () => {
    streams[0].start(); streams[0].text(ANSWER); streams[0].event("final", final()); streams[0].close();
  });
  assert.equal(articles().length, 2);
  assert.match(container.textContent!, /stream did not finish normally/);
  assert.doesNotMatch(container.textContent!, /Generation stopped/);
  await click("Retry");
  assert.equal(streams.length, 1);
});

test("late persisted-history reload cannot replace a newer selected conversation", async () => {
  const oldHistory = deferred<Response>();
  list = [detail("a"), detail("b")];
  let aReads = 0;
  getHistory = async (id) => id === "a" && ++aReads > 1 ? oldHistory.promise : Response.json(detail(id, [message(`${id}-old`, "user", `History ${id}`)]));
  await boot(); await send();
  await act(async () => streams[0].start());
  await complete(streams[0]);
  await click("Conversation b");
  assert.deepEqual(articles(), ["History b"]);
  await act(async () => oldHistory.resolve(Response.json(detail("a", [message("stale", "user", "STALE HISTORY")]))));
  assert.deepEqual(articles(), ["History b"]);
  assert.equal(localStorage.getItem("amitai-selected-conversation"), "b");
});

test("late history reload cannot erase a newer send in the same conversation", async () => {
  const oldHistory = deferred<Response>();
  getHistory = () => oldHistory.promise;
  await boot(); await send();
  await act(async () => streams[0].start());
  await complete(streams[0]);
  await send("Second turn");
  await act(async () => oldHistory.resolve(Response.json(detail("a"))));
  assert.equal(articles().length, 3);
  assert.match(articles()[2], /Second turn/);
  assert.equal(button("Stop generation").disabled, false);
  await click("Stop generation");
});

test("stopped remote image keeps its asset but every retry requires new explicit consent", async () => {
  remote = true;
  await boot();
  const input = container.querySelector('input[type="file"]')!;
  const file = new Blob(["image"], { type: "image/png" });
  Object.defineProperty(file, "name", { value: "photo.png" });
  Object.defineProperty(input, "files", { value: [file], configurable: true });
  await act(async () => input.dispatchEvent(new dom.window.Event("change", { bubbles: true })));
  await type(QUESTION);
  assert.equal(button("Send message").disabled, true);
  await act(async () => (container.querySelector('input[type="checkbox"]') as HTMLInputElement).click());
  await click("Send message");
  assert.equal(streams[0].payload.allow_remote_vision, true);
  await click("Stop generation");
  for (let retry = 1; retry <= 2; retry++) {
    const checkbox = container.querySelector('input[type="checkbox"]') as HTMLInputElement;
    assert.equal(checkbox.checked, false);
    assert.equal(button("Retry").disabled, true);
    await click("Retry"); assert.equal(streams.length, retry);
    await act(async () => checkbox.click());
    await click("Retry");
    assert.deepEqual(streams[retry].payload.asset_ids, [ASSET.id]);
    assert.equal(streams[retry].payload.allow_remote_vision, true);
    assert.equal(articles().length, 1);
    await click("Stop generation");
  }
  assertPrivateStorage();
});

for (const enterToSend of [true, false]) test(`keyboard send policy is unchanged (enterToSend=${enterToSend})`, async () => {
  const sent: string[] = [];
  await act(async () => root.render(<Composer enterToSend={enterToSend} onSend={(text) => { sent.push(text); }} />));
  await type(QUESTION);
  const textarea = container.querySelector("textarea")!;
  async function key(options: KeyboardEventInit) {
    const event = new dom.window.KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true, ...options });
    await act(async () => textarea.dispatchEvent(event));
    return event.defaultPrevented;
  }
  assert.equal(await key({ isComposing: true, ctrlKey: !enterToSend }), false);
  assert.equal(await key(enterToSend ? { shiftKey: true } : {}), false);
  assert.deepEqual(sent, []);
  assert.equal(await key(enterToSend ? {} : { ctrlKey: true }), true);
  assert.deepEqual(sent, [QUESTION]);
  assertPrivateStorage();
});

// Controlled geometry and animation frames make scroll tests deterministic,
// without pretending JSDOM implements layout or using wall-clock timing.
function scrollHarness() {
  let height = 1000;
  let top = 0;
  let writes = 0;
  let reads = 0;
  let nextFrame = 0;
  const frames = new Map<number, FrameRequestCallback>();
  dom.window.requestAnimationFrame = (callback) => { frames.set(++nextFrame, callback); return nextFrame; };
  dom.window.cancelAnimationFrame = (id) => { frames.delete(id); };
  const props = {
    messages: [message("history", "assistant", "Older history")],
    pendingMessage: null as Message | null, streamingMessage: null as Message | null,
    loading: false, sending: false, loadError: null, sendError: null,
    preferences: DEFAULT_PREFERENCES, onSend: async () => undefined,
    onRetryLoad: () => undefined, onRetrySend: () => undefined,
  };
  let viewport: HTMLDivElement;
  async function render() {
    await act(async () => root.render(<ChatView {...props} />));
    viewport = container.querySelector('[aria-label="Chat messages"]')!;
    Object.defineProperties(viewport, {
      scrollHeight: { configurable: true, get: () => { reads++; return height; } },
      clientHeight: { configurable: true, get: () => 200 },
      scrollTop: { configurable: true, get: () => top, set: (value: number) => { writes++; top = Math.max(0, Math.min(height - 200, value)); } },
    });
  }
  async function frame() {
    await act(async () => {
      const callbacks = [...frames.values()]; frames.clear();
      for (const callback of callbacks) callback(0);
    });
  }
  async function scroll(value: number) {
    top = value;
    await act(async () => viewport.dispatchEvent(new dom.window.Event("scroll")));
  }
  return { props, render, frame, scroll,
    set height(value: number) { height = value; }, get top() { return top; },
    get writes() { return writes; }, get reads() { return reads; }, get frames() { return frames.size; },
  };
}

test("smart scroll follows near bottom, coalesces chunks, pauses on scroll-up, and jump resumes", async () => {
  const scroll = scrollHarness();
  scroll.props.pendingMessage = message("pending", "user", QUESTION);
  scroll.props.sending = true;
  await scroll.render(); await scroll.frame();
  assert.equal(scroll.top, 800);
  await scroll.scroll(750); // 50px is within the 96px threshold.
  const writes = scroll.writes;
  const reads = scroll.reads;
  for (let index = 1; index <= 4; index++) {
    scroll.props.streamingMessage = message("streaming", "assistant", "chunk".repeat(index));
    scroll.height = 1000 + index * 100;
    await scroll.render();
  }
  assert.equal(scroll.reads, reads, "no layout reads per chunk render");
  assert.equal(scroll.frames, 1);
  await scroll.frame();
  assert.equal(scroll.writes, writes + 1);
  assert.equal(scroll.top, 1200);
  await scroll.scroll(200);
  scroll.props.streamingMessage = message("streaming", "assistant", "another chunk");
  scroll.height = 1700;
  await scroll.render(); await scroll.frame();
  assert.equal(scroll.top, 200, "streaming must not yank a reader back down");
  const jump = button("Jump to latest");
  assert.equal(jump.type, "button");
  assert.equal(jump.disabled, false);
  jump.focus(); assert.equal(document.activeElement, jump);
  await click("Jump to latest"); await scroll.frame();
  assert.equal(scroll.top, 1500);
  assert.equal(container.querySelector('[aria-label="Jump to latest"]'), null);
  scroll.height = 1800;
  scroll.props.streamingMessage = message("streaming", "assistant", "more and more chunks");
  await scroll.render(); await scroll.frame();
  assert.equal(scroll.top, 1600);
});

test("natural return near bottom restores follow; queued frame respects subsequent user scroll", async () => {
  const scroll = scrollHarness();
  await scroll.render(); await scroll.frame();
  await scroll.scroll(100);
  await scroll.scroll(740);
  scroll.height = 1400;
  scroll.props.streamingMessage = message("streaming", "assistant", "first");
  await scroll.render(); await scroll.frame();
  assert.equal(scroll.top, 1200);
  scroll.height = 1600;
  scroll.props.streamingMessage = message("streaming", "assistant", "first second");
  await scroll.render();
  await scroll.scroll(100); // user moves after render, before the scheduled frame
  await scroll.frame();
  assert.equal(scroll.top, 100);
});

test("new user turn intentionally follows, but unchanged history rerenders do not scroll", async () => {
  const scroll = scrollHarness();
  await scroll.render(); await scroll.frame();
  await scroll.scroll(100);
  const writes = scroll.writes;
  scroll.props.messages = [...scroll.props.messages];
  await scroll.render(); await scroll.frame();
  assert.equal(scroll.writes, writes);
  assert.equal(scroll.top, 100);
  scroll.props.pendingMessage = message("new-turn", "user", QUESTION);
  scroll.props.sending = true;
  scroll.height = 1300;
  await scroll.render(); await scroll.frame();
  assert.equal(scroll.top, 1100);
  assert.equal(container.querySelector('[aria-label="Jump to latest"]'), null);
  scroll.props.streamingMessage = message("streaming", "assistant", "queued");
  await scroll.render();
  await act(async () => root.unmount()); mounted = false;
  assert.equal(scroll.frames, 0, "scheduled frame is cancelled on unmount");
});

test("first saved conversation keeps its viewport and respects scroll-up through final/history reload", async () => {
  await boot(); await send();
  const viewport = container.querySelector('[aria-label="Chat messages"]')!;
  Object.defineProperties(viewport, {
    scrollHeight: { configurable: true, value: 2000 },
    clientHeight: { configurable: true, value: 200 },
  });
  await act(async () => { streams[0].start(); streams[0].text("PRIVATE_"); });
  await settle();
  await act(async () => {
    viewport.scrollTop = 100;
    viewport.dispatchEvent(new dom.window.Event("scroll"));
  });
  await act(async () => {
    streams[0].text("ANSWER_CANARY");
    streams[0].event("final", final());
    streams[0].event("done", {});
    streams[0].close();
  });
  await settle();
  assert.equal(container.querySelector('[aria-label="Chat messages"]'), viewport);
  assert.equal(viewport.scrollTop, 100);
  assert.equal(button("Jump to latest").disabled, false);
});
