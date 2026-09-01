import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { securityHeaders } from "./securityHeaders.ts";
import nextConfig from "../next.config.ts";

interface SourceFile {
  path: string;
  source: string;
}

function sourceFiles(directory: string): SourceFile[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    if ([".next", "node_modules"].includes(entry.name)) return [];
    const path = `${directory}/${entry.name}`;
    if (entry.isDirectory()) return sourceFiles(path);
    if (!/\.(ts|tsx)$/.test(entry.name) || entry.name.includes(".test.")) return [];
    return [{ path, source: readFileSync(path, "utf8") }];
  });
}

const frontendRoot = fileURLToPath(new URL("../", import.meta.url));
const sources = sourceFiles(frontendRoot);
const packageJson = JSON.parse(readFileSync(`${frontendRoot}/package.json`, "utf8")) as {
  scripts: Record<string, string>;
};
const appSource = sources.find(({ path }) => path.endsWith("/components/AmitaiApp.tsx"))!.source;
const browserSources = sources.filter(
  ({ path, source }) =>
    source.startsWith('"use client"') ||
    path.endsWith("/lib/api.ts") ||
    path.endsWith("/lib/sse.ts"),
);

test("browser code persists only harmless UI preferences and selected conversation id", () => {
  const writes = [...appSource.matchAll(/localStorage\.setItem\(([^,]+),/g)].map(
    (match) => match[1].trim(),
  );

  assert.ok(writes.length > 0);
  assert.deepEqual(new Set(writes), new Set(["PREFERENCES_KEY", "SELECTED_CONVERSATION_KEY"]));
  for (const { path, source } of sources) {
    assert.doesNotMatch(source, /sessionStorage|indexedDB/);
    assert.doesNotMatch(source, /NEXT_PUBLIC_.*TOKEN/, path);
    if (source.includes("localStorage")) {
      assert.match(path, /\/components\/AmitaiApp\.tsx$/);
    }
  }
  for (const { path, source } of browserSources) {
    assert.doesNotMatch(source, /AMITAI_LOCAL_API_TOKEN/, path);
    assert.doesNotMatch(source, /Authorization\s*:/, path);
  }
});

test("supported Next launch commands bind to loopback", () => {
  assert.equal(packageJson.scripts.dev, "next dev -H 127.0.0.1");
  assert.equal(packageJson.scripts.start, "next start -H 127.0.0.1");
  assert.doesNotMatch(packageJson.scripts.dev, /0\.0\.0\.0/);
  assert.doesNotMatch(packageJson.scripts.start, /0\.0\.0\.0/);
});

test("production headers restrict framing, capabilities and all external sources without HSTS or eval", () => {
  const headers = Object.fromEntries(securityHeaders(true).map(({ key, value }) => [key.toLowerCase(), value]));
  for (const [key, expected] of Object.entries({
    "x-content-type-options": "nosniff", "referrer-policy": "no-referrer", "x-frame-options": "DENY",
    "cross-origin-opener-policy": "same-origin", "cross-origin-resource-policy": "same-origin",
    "x-dns-prefetch-control": "off", "x-permitted-cross-domain-policies": "none",
  })) assert.equal(headers[key], expected);
  for (const capability of ["camera", "microphone", "geolocation", "payment", "usb", "serial"]) {
    assert.ok(headers["permissions-policy"].split(/,\s*/).includes(`${capability}=()`));
  }
  assert.equal(headers["strict-transport-security"], undefined);
  const csp = headers["content-security-policy"];
  const directives = new Map(csp.split(/;\s*/).map((item) => {
    const [key, ...sources] = item.split(/\s+/);
    return [key, sources];
  }));
  for (const key of ["default-src", "connect-src", "form-action", "media-src"]) {
    assert.deepEqual(directives.get(key), ["'self'"]);
  }
  for (const key of ["base-uri", "object-src", "frame-ancestors"]) {
    assert.deepEqual(directives.get(key), ["'none'"]);
  }
  assert.deepEqual(directives.get("script-src"), ["'self'", "'unsafe-inline'"]);
  assert.deepEqual(directives.get("style-src"), ["'self'", "'unsafe-inline'"]);
  assert.deepEqual(directives.get("img-src"), ["'self'", "data:", "blob:"]);
  assert.deepEqual(directives.get("font-src"), ["'self'", "data:"]);
  assert.deepEqual(directives.get("worker-src"), ["'self'", "blob:"]);
  assert.doesNotMatch(csp, /unsafe-eval|\*|https?:|wss?:|\.com|\.net/);
  assert.match(securityHeaders(false).find(({ key }) => key === "Content-Security-Policy")!.value, /'unsafe-eval'/);
});

test("Next installs global headers and API no-store defaults without exposing its version", async () => {
  assert.equal(nextConfig.poweredByHeader, false);
  assert.equal(nextConfig.skipProxyUrlNormalize, true);
  const rules = await nextConfig.headers!();
  assert.deepEqual(rules.find(({ source }) => source === "/:path*")?.headers,
    securityHeaders(process.env.NODE_ENV === "production"));
  assert.deepEqual(rules.find(({ source }) => source === "/api/:path*")?.headers,
    [{ key: "Cache-Control", value: "no-store" }]);
  assert.deepEqual(rules.at(-1), { source: "/api/chat/stream", headers: [
    { key: "Cache-Control", value: "no-store, no-transform" },
  ] });
});

test("frontend audit excludes content logging, search URL transport and browser secrets", () => {
  for (const { path, source } of sources) {
    assert.doesNotMatch(source, /incomingUrl\.search|console\.(log|error)\(/, path);
    assert.doesNotMatch(source, /[?&]query=|parameters\.set\("query"/, path);
  }
  for (const { path, source } of browserSources) {
    assert.doesNotMatch(source, /AMITAI_(?:DB_KEY|REMOTE_INFERENCE_TOKEN|API_ORIGIN)|runpod|Authorization/i, path);
  }
});
