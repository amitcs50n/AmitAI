import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { once } from "node:events";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { JSDOM } from "jsdom";

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
const proxy = await import("../app/api/[...path]/route.ts");

test("UI -> authenticated proxy -> real runtime: text, memory, calculator, image, cancel, retry and reload", { timeout: 60000 }, async () => {
  const repository = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
  const directory = await mkdtemp(join(tmpdir(), "aevon-ui-test-"));
  const tokenFile = join(directory, "local-api-token");
  await writeFile(tokenFile, randomBytes(32).toString("hex"), { mode: 0o600 });
  const originalFetch = globalThis.fetch;
  const oldOrigin = process.env.AMITAI_API_ORIGIN;
  const oldTokenFile = process.env.AMITAI_LOCAL_API_TOKEN_FILE;
  const child = spawn(process.env.AMITAI_TEST_PYTHON ?? "python", ["-m", "tests.ui_runtime_fixture", directory], {
    cwd: repository, windowsHide: true,
    env: { ...process.env, PYTHONUNBUFFERED: "1", HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1", CUDA_VISIBLE_DEVICES: "" },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let port: number | undefined;
  let output = "";
  let errorOutput = "";
  let childError: Error | null = null;
  const events: string[] = [];
  child.on("error", (error) => { childError = error; });
  child.stderr.on("data", (data) => { errorOutput += String(data); });
  child.stdout.on("data", (data) => {
    output += String(data);
    while (output.includes("\n")) {
      const end = output.indexOf("\n");
      const item = JSON.parse(output.slice(0, end));
      output = output.slice(end + 1);
      if (item.port) port = item.port;
      if (item.event) events.push(item.event);
    }
  });
  const container = document.createElement("div");
  document.body.append(container);
  let root = createRoot(container);
  let mounted = false;
  const requests: string[] = [];
  async function until(predicate: () => boolean | Promise<boolean>, reason: string) {
    const deadline = Date.now() + 10000;
    while (Date.now() < deadline) {
      if (childError) throw childError;
      if (child.exitCode !== null) throw new Error(`CPU fixture exited: ${errorOutput}`);
      if (await predicate()) return;
      await act(async () => new Promise((done) => setTimeout(done, 25)));
    }
    assert.fail(`Timed out: ${reason}\n${container.textContent}\n${errorOutput}`);
  }
  function button(name: string) {
    const found = [...container.querySelectorAll("button")].find((item) => item.getAttribute("aria-label") === name || item.textContent?.trim() === name);
    assert.ok(found, `Missing button: ${name}`);
    return found;
  }
  async function click(name: string) { await act(async () => button(name).click()); }
  async function type(input: HTMLInputElement | HTMLTextAreaElement, value: string) {
    await act(async () => {
      const prototype = input.tagName === "INPUT" ? dom.window.HTMLInputElement.prototype : dom.window.HTMLTextAreaElement.prototype;
      Object.getOwnPropertyDescriptor(prototype, "value")!.set!.call(input, value);
      input.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
    });
  }
  async function send(value: string) { await type(container.querySelector("textarea")!, value); await click("Send message"); }
  const articles = () => [...container.querySelectorAll("article")].map((item) => item.textContent ?? "");
  async function finished(answer: string) {
    await until(() => !!articles().at(-1)?.includes(answer) && !container.querySelector("textarea")?.disabled, answer);
    assert.doesNotMatch(container.textContent!, /<tool_call|<tool_result|PRIVATE_ENGINE_ERROR_CANARY/);
  }
  async function mount() {
    await act(async () => root.render(<AmitaiApp />));
    mounted = true;
    await until(() => !!container.querySelector("textarea") && !container.textContent?.includes("Loading conversation"), "app ready");
  }
  async function upload() {
    const file = new File([await readFile(join(directory, "red.png"))], "red.png", { type: "image/png" });
    const input = container.querySelector('input[type="file"]')!;
    Object.defineProperty(input, "files", { configurable: true, value: [file] });
    await act(async () => input.dispatchEvent(new dom.window.Event("change", { bubbles: true })));
    await until(() => !!container.querySelector('[aria-label="Remove red.png"]') && !button("Remove red.png").disabled, "secure image upload");
  }
  try {
    await until(() => !!port, "fixture startup");
    const origin = `http://127.0.0.1:${port}`;
    process.env.AMITAI_API_ORIGIN = origin;
    process.env.AMITAI_LOCAL_API_TOKEN_FILE = tokenFile;
    globalThis.fetch = async (input, init) => {
      const path = String(input);
      if (path.startsWith(`${origin}/`)) return originalFetch(input, init);
      assert.ok(path.startsWith("/api/"), "only existing API routes and fixture loopback are allowed");
      requests.push(`${init?.method ?? "GET"} ${path}`);
      const request = new Request(`http://localhost${path}`, { ...init, headers: { ...init?.headers, Origin: "http://localhost" } });
      const handler = ({ GET: proxy.GET, POST: proxy.POST, PATCH: proxy.PATCH, DELETE: proxy.DELETE })[request.method as "GET" | "POST" | "PATCH" | "DELETE"];
      return handler(request, { params: Promise.resolve({ path: new URL(request.url).pathname.slice(5).split("/") }) });
    };
    await until(async () => { try { return (await fetch("/api/capabilities")).ok; } catch { return false; } }, "authenticated backend ready");
    await mount();
    await send("Hello");
    await until(() => articles().at(-1)?.includes("Hello") === true && !!container.querySelector('[aria-label="Stop generation"]'), "incremental text before final");
    await finished("Hello from the CPU fixture.");
    const firstId = localStorage.getItem("amitai-selected-conversation");
    assert.ok(firstId);
    await send("What is 17 * 83?");
    await finished("1411");
    assert.ok(events.includes("calculator-result-used"));
    assert.match(container.textContent!, /calculator/i);

    await click("Memory");
    await until(() => container.textContent?.includes("New memory") === true, "memory view");
    await click("New memory");
    await type(container.querySelector('[role="dialog"] input')!, "ui.theme");
    await type(container.querySelector('[role="dialog"] textarea')!, "BLUE_CANARY");
    await click("Create memory");
    await until(() => !container.querySelector('[role="dialog"]') && container.textContent?.includes("BLUE_CANARY") === true, "memory persisted");
    await act(async () => root.unmount()); mounted = false;
    root = createRoot(container);
    await mount();
    assert.equal(articles().length, 4, "reload reads persisted turns");
    await send("What is my ui.theme?");
    await finished("BLUE_CANARY");
    assert.ok(events.includes("memory-used"));

    await upload();
    const preview = container.querySelector("img")!.getAttribute("src")!;
    assert.equal((await fetch(preview)).status, 200, "preview uses authenticated asset path");
    await click("Remove red.png");
    await until(() => !container.querySelector("img"), "attachment removed");
    assert.equal((await fetch(preview)).status, 404);
    await upload();
    await send("Describe image");
    await until(() => articles().at(-1)?.includes("Red") === true && !!container.querySelector('[aria-label="Stop generation"]'), "incremental vision before final");
    await finished("Red square shown.");
    assert.ok(events.includes("image-decoded"));
    assert.ok(container.querySelector("article img"), "image persists in server history");

    for (const vision of [false, true]) {
      if (vision) await upload();
      const before = articles().length;
      await send(vision ? "Pause image" : "Pause text");
      await until(() => articles().at(-1)?.includes("Partial answer") === true, "partial generation");
      await click("Stop generation");
      await until(() => events.includes(vision ? "vision-cancelled" : "text-cancelled"), "producer cancellation");
      assert.doesNotMatch(container.textContent!, /Partial answer/);
      const persisted = await (await fetch(`/api/conversations/${firstId}`)).json();
      assert.equal(persisted.messages.length, before, "cancel never persisted a partial turn");
      await click("Retry");
      await finished(vision ? "Red square shown." : "Hello from the CPU fixture.");
      assert.equal(articles().length, before + 2);
    }
    await upload();
    await send("Fail image");
    await until(() => container.textContent?.includes("Generation failed") === true, "sanitized vision failure");
    assert.doesNotMatch(container.textContent!, /PRIVATE_ENGINE_ERROR_CANARY/);
    await click("Retry");
    await finished("Red square shown.");
    await click("New conversation");
    await send("Second conversation");
    await finished("Hello from the CPU fixture.");
    assert.notEqual(localStorage.getItem("amitai-selected-conversation"), firstId);
    await click("Hello");
    await until(() => articles().length === 14, "switch to persisted first conversation");
    assert.match(articles().at(-1)!, /Red square shown/);
    const stored = JSON.stringify({ ...localStorage, ...sessionStorage });
    assert.doesNotMatch(stored, /BLUE_CANARY|Red square|Pause|1411/);
    assert.ok(requests.includes("POST /api/chat/stream"));
    assert.ok(requests.includes("POST /api/memory"));
  } finally {
    if (mounted) await act(async () => root.unmount());
    container.remove();
    globalThis.fetch = originalFetch;
    if (oldOrigin === undefined) delete process.env.AMITAI_API_ORIGIN; else process.env.AMITAI_API_ORIGIN = oldOrigin;
    if (oldTokenFile === undefined) delete process.env.AMITAI_LOCAL_API_TOKEN_FILE; else process.env.AMITAI_LOCAL_API_TOKEN_FILE = oldTokenFile;
    if (child.exitCode === null && !childError) { const closed = once(child, "close"); child.kill(); await closed; }
    assert.equal(dirname(resolve(directory)), resolve(tmpdir()));
    assert.ok(basename(directory).startsWith("aevon-ui-test-"));
    await rm(directory, { recursive: true, force: true });
  }
});
