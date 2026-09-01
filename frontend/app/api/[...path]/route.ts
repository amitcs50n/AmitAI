import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { isAbsolute, join } from "node:path";
import {
  hasAllowedBrowserOrigin,
  isLoopbackHostname,
  resolveProxyQuery,
  resolveProxyRoute,
  validateProxyBody,
} from "../../../lib/proxyPolicy.ts";

const DEFAULT_BACKEND_ORIGIN = "http://127.0.0.1:8000";
const LOCAL_API_TOKEN_PATTERN = /^[0-9a-f]{64}$/;
const FORWARDED_REQUEST_HEADERS = ["accept", "content-type"] as const;
const FORWARDED_RESPONSE_HEADERS = [
  "content-type",
  "x-accel-buffering",
] as const;

interface RouteContext {
  params: Promise<{ path: string[] }>;
}

function jsonError(detail: string, status: number): Response {
  return Response.json(
    { detail },
    {
      status,
      headers: { "Cache-Control": "no-store" },
    },
  );
}

function localApiTokenFile(): string | null {
  const override = process.env.AMITAI_LOCAL_API_TOKEN_FILE;
  if (override) return isAbsolute(override) ? override : null;
  if (process.platform === "win32") {
    const root = process.env.LOCALAPPDATA;
    return root ? join(root, "AmitAI", "runtime", "local-api-token") : null;
  }
  if (process.platform === "darwin") {
    return join(homedir(), "Library", "Application Support", "AmitAI", "runtime", "local-api-token");
  }
  const runtimeRoot = process.env.XDG_RUNTIME_DIR;
  if (runtimeRoot) return join(runtimeRoot, "amitai", "local-api-token");
  const stateRoot = process.env.XDG_STATE_HOME ?? join(homedir(), ".local", "state");
  return join(stateRoot, "amitai", "runtime", "local-api-token");
}

function readLocalApiToken(): string | null {
  const path = localApiTokenFile();
  if (!path) return null;
  try {
    const raw = readFileSync(/* turbopackIgnore: true */ path, {
      encoding: "utf8",
      flag: "r",
    });
    if (!/^[0-9a-f]{64}\r?\n?$/.test(raw)) return null;
    const token = raw.replace(/\r?\n$/, "");
    return LOCAL_API_TOKEN_PATTERN.test(token) ? token : null;
  } catch {
    return null;
  }
}

function backendConfiguration(): { origin: string; token: string } | null {
  const token = readLocalApiToken();
  if (!token) return null;

  const configuredOrigin = process.env.AMITAI_API_ORIGIN ?? DEFAULT_BACKEND_ORIGIN;
  let parsed: URL;
  try {
    parsed = new URL(configuredOrigin);
  } catch {
    return null;
  }
  if (
    !["http:", "https:"].includes(parsed.protocol) ||
    parsed.username ||
    parsed.password ||
    parsed.search ||
    parsed.hash ||
    (parsed.pathname !== "/" && parsed.pathname !== "")
  ) {
    return null;
  }
  if (!isLoopbackHostname(parsed.hostname) && process.env.AMITAI_ALLOW_LAN !== "1") {
    return null;
  }
  return { origin: parsed.origin, token };
}

function requestHeaders(request: Request, token: string): Headers {
  const headers = new Headers({ Authorization: `Bearer ${token}` });
  for (const name of FORWARDED_REQUEST_HEADERS) {
    const value = request.headers.get(name);
    if (value !== null) headers.set(name, value);
  }
  return headers;
}

function responseHeaders(upstream: Response): Headers {
  const headers = new Headers({ "Cache-Control": "no-store" });
  for (const name of FORWARDED_RESPONSE_HEADERS) {
    const value = upstream.headers.get(name);
    if (value !== null) headers.set(name, value);
  }
  if (headers.get("content-type")?.toLowerCase().startsWith("text/event-stream")) {
    headers.set("Cache-Control", "no-store, no-transform");
    headers.set("X-Accel-Buffering", "no");
  }
  return headers;
}

async function proxyRequest(request: Request, context: RouteContext): Promise<Response> {
  if (!hasAllowedBrowserOrigin(request)) {
    return jsonError("Cross-origin request denied", 403);
  }

  const { path } = await context.params;
  const route = resolveProxyRoute(request.method, path);
  if (!route) return jsonError("Request not allowed", 404);
  const url = new URL(request.url);
  // Params are decoded by Next. Require the original URL to use that same
  // canonical path, not encoded aliases or extra separators normalized by routing.
  if (url.pathname !== route.pathname) return jsonError("Request not allowed", 404);
  const query = resolveProxyQuery(route, url.searchParams);
  if (query === null) return jsonError("Invalid request", 400);
  const validated = await validateProxyBody(request, route.body);
  if ("status" in validated) return jsonError(validated.detail, validated.status);

  const configuration = backendConfiguration();
  if (!configuration) return jsonError("Local API proxy is not configured", 503);

  const upstreamUrl = `${configuration.origin}${route.pathname}${query}`;
  const init: RequestInit = {
    method: request.method,
    headers: requestHeaders(request, configuration.token),
    cache: "no-store",
    redirect: "manual",
    signal: request.signal,
    body: validated.body,
  };

  let upstream: Response;
  try {
    upstream = await fetch(upstreamUrl, init);
  } catch {
    return jsonError("Local AmitAI backend is unavailable", 502);
  }

  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: responseHeaders(upstream),
  });
}

export function GET(request: Request, context: RouteContext): Promise<Response> {
  return proxyRequest(request, context);
}

export function HEAD(request: Request, context: RouteContext): Promise<Response> {
  return proxyRequest(request, context);
}

export function POST(request: Request, context: RouteContext): Promise<Response> {
  return proxyRequest(request, context);
}

export function PATCH(request: Request, context: RouteContext): Promise<Response> {
  return proxyRequest(request, context);
}

export function DELETE(request: Request, context: RouteContext): Promise<Response> {
  return proxyRequest(request, context);
}

// Explicit handlers prevent Next from automatically accepting OPTIONS/HEAD.
export function PUT(request: Request, context: RouteContext): Promise<Response> {
  return proxyRequest(request, context);
}

export function OPTIONS(request: Request, context: RouteContext): Promise<Response> {
  return proxyRequest(request, context);
}
