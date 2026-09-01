import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";

import { JSDOM } from "jsdom";
import type { ReactElement } from "react";
import type { Root } from "react-dom/client";

import { DEFAULT_PREFERENCES, type MemoryRecord } from "../lib/types.ts";

function installDomGlobals(target: JSDOM): void {
  Object.defineProperty(globalThis, "window", { configurable: true, value: target.window });
  Object.defineProperty(globalThis, "document", { configurable: true, value: target.window.document });
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: target.window.navigator });
  Object.defineProperty(globalThis, "HTMLElement", { configurable: true, value: target.window.HTMLElement });
  Object.defineProperty(globalThis, "HTMLInputElement", { configurable: true, value: target.window.HTMLInputElement });
  Object.defineProperty(globalThis, "HTMLTextAreaElement", { configurable: true, value: target.window.HTMLTextAreaElement });
  Object.defineProperty(globalThis, "HTMLSelectElement", { configurable: true, value: target.window.HTMLSelectElement });
  Object.defineProperty(globalThis, "IS_REACT_ACT_ENVIRONMENT", { configurable: true, value: true });
  target.window.HTMLElement.prototype.scrollIntoView = () => undefined;
}

const bootstrapDom = new JSDOM("<!doctype html><html><body></body></html>", {
  url: "http://localhost",
});
installDomGlobals(bootstrapDom);

const { act } = await import("react");
const { createRoot } = await import("react-dom/client");
const { ChatView } = await import("./ChatView.tsx");
const { MemoryView } = await import("./MemoryView.tsx");
const { PreferencesView } = await import("./PreferencesView.tsx");

interface FetchCall {
  input: string;
  init?: RequestInit;
}

let dom: JSDOM;
let container: HTMLDivElement;
let root: Root;
let originalFetch: typeof fetch;

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function memory(overrides: Partial<MemoryRecord> = {}): MemoryRecord {
  return {
    id: "memory-1",
    operation: "current",
    category: "preference",
    key: "ui.theme",
    value: "dark",
    status: "active",
    source: { conversation_id: null, message_id: null },
    updated_at: "2026-08-30T10:00:00Z",
    ...overrides,
  };
}

function mockFetch(responses: Response[]): FetchCall[] {
  const calls: FetchCall[] = [];
  globalThis.fetch = (async (input, init) => {
    calls.push({ input: String(input), init });
    const response = responses.shift();
    if (!response) throw new Error(`Unexpected fetch: ${String(input)}`);
    return response;
  }) as typeof fetch;
  return calls;
}

async function settle(milliseconds = 25): Promise<void> {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, milliseconds));
  });
}

async function render(element: ReactElement): Promise<void> {
  await act(async () => root.render(element));
  await settle();
}

async function setControlValue(
  control: HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement,
  value: string,
): Promise<void> {
  const prototype =
    control instanceof dom.window.HTMLTextAreaElement
      ? dom.window.HTMLTextAreaElement.prototype
      : control instanceof dom.window.HTMLSelectElement
        ? dom.window.HTMLSelectElement.prototype
        : dom.window.HTMLInputElement.prototype;
  const setter = Object.getOwnPropertyDescriptor(prototype, "value")?.set;
  await act(async () => {
    setter?.call(control, value);
    control.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
    control.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
  });
}

async function click(element: Element): Promise<void> {
  await act(async () => {
    element.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
  });
  await settle();
}

async function submit(form: HTMLFormElement): Promise<void> {
  await act(async () => {
    form.dispatchEvent(new dom.window.Event("submit", { bubbles: true, cancelable: true }));
  });
  await settle();
}

beforeEach(() => {
  dom = new JSDOM("<!doctype html><html><body><div id='root'></div></body></html>", {
    url: "http://localhost",
  });
  installDomGlobals(dom);
  container = dom.window.document.querySelector("#root") as HTMLDivElement;
  root = createRoot(container);
  originalFetch = globalThis.fetch;
});

afterEach(async () => {
  await act(async () => root.unmount());
  globalThis.fetch = originalFetch;
  dom.window.close();
});

test("lists active memories with category, key, value, and timestamp", async () => {
  const calls = mockFetch([jsonResponse([memory()])]);

  await render(<MemoryView />);

  assert.equal(calls[0].input, "/api/memory");
  assert.match(container.textContent ?? "", /preference/i);
  assert.match(container.textContent ?? "", /ui\.theme/);
  assert.match(container.textContent ?? "", /dark/);
  assert.match(container.textContent ?? "", /Updated/);
});

test("shows the active-memory empty state", async () => {
  mockFetch([jsonResponse([])]);

  await render(<MemoryView />);

  assert.match(container.textContent ?? "", /No memories yet/);
});

test("search and category filtering refetch with the correct query", async () => {
  const calls = mockFetch([
    jsonResponse([]),
    jsonResponse([memory()]),
    jsonResponse([memory({ category: "project", key: "app.name", value: "AmitAI" })]),
  ]);
  await render(<MemoryView />);

  const search = container.querySelector('input[placeholder="Search memories"]') as HTMLInputElement;
  await setControlValue(search, "UI theme");
  await submit(container.querySelector('form[role="search"]') as HTMLFormElement);
  assert.equal(calls[1].input, "/api/memory/search");
  assert.equal(calls[1].init?.method, "POST");
  assert.deepEqual(JSON.parse(String(calls[1].init?.body)), { query: "UI theme" });

  const category = container.querySelector('select[aria-label="Memory category"]') as HTMLSelectElement;
  await setControlValue(category, "project");
  await settle();
  assert.equal(calls[2].input, "/api/memory?category=project");
});

test("edits a memory and refetches without changing the current filter", async () => {
  const updated = memory({ value: "light" });
  const calls = mockFetch([
    jsonResponse([memory()]),
    jsonResponse(updated),
    jsonResponse([updated]),
  ]);
  await render(<MemoryView />);

  await click(container.querySelector('button[aria-label="Edit ui.theme"]')!);
  const textarea = container.querySelector('[role="dialog"] textarea') as HTMLTextAreaElement;
  await setControlValue(textarea, "light");
  await submit(container.querySelector('[role="dialog"] form') as HTMLFormElement);

  assert.equal(calls[1].input, "/api/memory/memory-1");
  assert.equal(calls[1].init?.method, "PATCH");
  assert.deepEqual(JSON.parse(String(calls[1].init?.body)), { value: "light" });
  assert.equal(calls[2].input, "/api/memory");
  assert.match(container.textContent ?? "", /light/);
});

test("forget requires confirmation and refetches after deletion", async () => {
  const calls = mockFetch([
    jsonResponse([memory()]),
    new Response(null, { status: 204 }),
    jsonResponse([]),
  ]);
  await render(<MemoryView />);

  await click(container.querySelector('button[aria-label="Forget ui.theme"]')!);
  assert.match(container.textContent ?? "", /tombstoned and its stored value redacted/);
  assert.equal(calls.length, 1);

  const confirm = [...container.querySelectorAll("button")].find(
    (button) => button.textContent?.trim() === "Forget memory",
  );
  await click(confirm!);

  assert.equal(calls[1].input, "/api/memory/memory-1");
  assert.equal(calls[1].init?.method, "DELETE");
  assert.equal(calls[2].input, "/api/memory");
  assert.match(container.textContent ?? "", /No memories yet/);
});

test("a 409 edit conflict is surfaced in the editor", async () => {
  mockFetch([
    jsonResponse([memory()]),
    jsonResponse({ detail: "Memory changed concurrently" }, 409),
  ]);
  await render(<MemoryView />);

  await click(container.querySelector('button[aria-label="Edit ui.theme"]')!);
  const textarea = container.querySelector('[role="dialog"] textarea') as HTMLTextAreaElement;
  await setControlValue(textarea, "light");
  await submit(container.querySelector('[role="dialog"] form') as HTMLFormElement);

  assert.match(container.textContent ?? "", /changed elsewhere\. Refresh and try again/);
  assert.ok(container.querySelector('[role="dialog"]'));
});

test("forgotten memories never render a returned raw value", async () => {
  const secret = "SHOULD-NOT-RENDER";
  const calls = mockFetch([
    jsonResponse([]),
    jsonResponse([
      memory({ status: "deleted", operation: "current", value: secret }),
    ]),
  ]);
  await render(<MemoryView />);

  const forgottenButton = [...container.querySelectorAll("button")].find(
    (button) => button.textContent?.trim() === "Forgotten",
  );
  await click(forgottenButton!);

  assert.equal(calls[1].input, "/api/memory?status=deleted");
  assert.doesNotMatch(container.textContent ?? "", new RegExp(secret));
  assert.match(container.textContent ?? "", /Value redacted/);
});

test("chat and browser preferences still render with their existing controls", async () => {
  await render(
    <ChatView
      loadError={null}
      loading={false}
      messages={[]}
      onRetryLoad={() => undefined}
      onRetrySend={() => undefined}
      onSend={async () => undefined}
      pendingMessage={null}
      preferences={DEFAULT_PREFERENCES}
      sendError={null}
      sending={false}
      streamingMessage={null}
    />,
  );
  assert.match(container.textContent ?? "", /What are we working on/);
  assert.ok(container.querySelector('textarea[aria-label="Write to Aevon"]'));

  await act(async () => root.render(
    <PreferencesView onChange={() => undefined} preferences={DEFAULT_PREFERENCES} />,
  ));
  assert.match(container.textContent ?? "", /Preferences/);
  assert.equal(container.querySelectorAll('[role="switch"]').length, 4);
});
