import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

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
