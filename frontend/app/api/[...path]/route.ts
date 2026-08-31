const DEFAULT_BACKEND_ORIGIN = "http://127.0.0.1:8000";
const MIN_LOCAL_API_TOKEN_CHARS = 32;
const FORWARDED_REQUEST_HEADERS = ["accept", "content-type"] as const;
const FORWARDED_RESPONSE_HEADERS = [
  "cache-control",
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

function isLoopbackHostname(hostname: string): boolean {
  const normalized = hostname.toLowerCase().replace(/^\[|\]$/g, "");
  return normalized === "localhost" || normalized === "127.0.0.1" || normalized === "::1";
}

function isStateChangingMethod(method: string): boolean {
  return method === "POST" || method === "PATCH" || method === "DELETE";
}

function hasAllowedBrowserOrigin(request: Request): boolean {
  if (!isStateChangingMethod(request.method)) return true;

  const origin = request.headers.get("origin");
  if (origin === null) return true;

  try {
    const browserOrigin = new URL(origin);
    const aevonOrigin = new URL(request.url);
    return (
      isLoopbackHostname(browserOrigin.hostname) &&
      isLoopbackHostname(aevonOrigin.hostname) &&
      browserOrigin.origin === aevonOrigin.origin
    );
  } catch {
    return false;
  }
}

function backendConfiguration(): { origin: string; token: string } | null {
  const token = process.env.AMITAI_LOCAL_API_TOKEN?.trim();
  if (!token || token.length < MIN_LOCAL_API_TOKEN_CHARS) return null;

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
  const headers = new Headers();
  for (const name of FORWARDED_RESPONSE_HEADERS) {
    const value = upstream.headers.get(name);
    if (value !== null) headers.set(name, value);
  }
  return headers;
}

async function proxyRequest(request: Request, context: RouteContext): Promise<Response> {
  if (!hasAllowedBrowserOrigin(request)) {
    return jsonError("Cross-origin request denied", 403);
  }

  const configuration = backendConfiguration();
  if (!configuration) return jsonError("Local API proxy is not configured", 503);

  const { path } = await context.params;
  if (!path.length) return jsonError("Not found", 404);
  const encodedPath = path.map((segment) => encodeURIComponent(segment)).join("/");
  const incomingUrl = new URL(request.url);
  const upstreamUrl = `${configuration.origin}/api/${encodedPath}${incomingUrl.search}`;
  const hasBody = request.method !== "GET" && request.method !== "HEAD" && request.body !== null;
  const init: RequestInit & { duplex?: "half" } = {
    method: request.method,
    headers: requestHeaders(request, configuration.token),
    cache: "no-store",
    redirect: "manual",
    signal: request.signal,
    ...(hasBody ? { body: request.body, duplex: "half" } : {}),
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
