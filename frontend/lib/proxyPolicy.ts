// Pure browser-boundary policy. Never reads configuration, tokens, or local state.
export const MAX_REQUEST_BODY_BYTES = 256 * 1024;

type BodyPolicy = "none" | "json" | "optional-json";
export interface ProxyRoute {
  pathname: string;
  body: BodyPolicy;
  memoryFilters: boolean;
}

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const CATEGORIES = new Set(["preference", "profile", "project", "workflow", "instruction"]);
const ROUTES: ReadonlyArray<{
  path: readonly string[];
  methods: Readonly<Partial<Record<string, BodyPolicy>>>;
}> = [
  { path: ["health"], methods: { GET: "none" } },
  { path: ["conversations"], methods: { GET: "none", POST: "optional-json" } },
  { path: ["conversations", ":uuid"], methods: { GET: "none", PATCH: "json", DELETE: "none" } },
  { path: ["chat"], methods: { POST: "json" } },
  { path: ["chat", "stream"], methods: { POST: "json" } },
  { path: ["memory"], methods: { GET: "none", POST: "json" } },
  { path: ["memory", "search"], methods: { POST: "json" } },
  { path: ["memory", ":uuid"], methods: { PATCH: "json", DELETE: "none" } },
];

export function isLoopbackHostname(hostname: string): boolean {
  return ["localhost", "127.0.0.1", "[::1]", "::1"].includes(hostname.toLowerCase());
}

export function hasAllowedBrowserOrigin(request: Request): boolean {
  const target = new URL(request.url);
  if (!isLoopbackHostname(target.hostname) || !["http:", "https:"].includes(target.protocol)) {
    return false;
  }
  // Next constructs absolute URLs using the listener host. Reject a differing
  // browser Host too, so DNS rebinding cannot hide behind that construction.
  const host = request.headers.get("host");
  if (host !== null && host !== target.host) return false;
  if (request.headers.get("sec-fetch-site")?.toLowerCase() === "cross-site") return false;
  const origin = request.headers.get("origin");
  const mutation = ["POST", "PATCH", "DELETE"].includes(request.method);
  if (origin === null) return !mutation;
  // Compare the serialized Origin itself: credentials, paths, null, multiple origins,
  // aliases, and a different port must not normalize into an accepted origin.
  return origin === target.origin;
}

export function resolveProxyRoute(method: string, segments: readonly string[]): ProxyRoute | null {
  // Approved literals/UUIDs need no decoding. Reject percent escapes, separators,
  // dot segments, controls, empty and oversized segments instead of salvaging them.
  if (segments.some((part) => !/^[a-zA-Z0-9-]{1,36}$/.test(part))) return null;
  const match = ROUTES.find((route) =>
    route.path.length === segments.length &&
    route.path.every((part, index) => part === ":uuid" ? UUID.test(segments[index]) : part === segments[index]),
  );
  const body = match && Object.hasOwn(match.methods, method) ? match.methods[method] : undefined;
  if (!body) return null;
  return {
    pathname: `/api/${segments.join("/")}`,
    body,
    memoryFilters: method === "GET" && segments.length === 1 && segments[0] === "memory",
  };
}

export function resolveProxyQuery(route: ProxyRoute, parameters: URLSearchParams): string | null {
  const allowed = new URLSearchParams();
  for (const [key, value] of parameters) {
    if (!route.memoryFilters || allowed.has(key)) return null;
    if (key === "status" && ["active", "deleted"].includes(value)) allowed.set(key, value);
    else if (key === "category" && CATEGORIES.has(value)) allowed.set(key, value);
    else return null;
  }
  return allowed.size ? `?${allowed.toString()}` : "";
}

type BodyResult = { body?: string } | { status: number; detail: string };

export async function validateProxyBody(request: Request, policy: BodyPolicy): Promise<BodyResult> {
  const contentType = request.headers.get("content-type");
  const jsonType = /^application\/json(?:\s*;\s*charset\s*=\s*(?:utf-8|"utf-8"))?\s*$/i.test(contentType ?? "");
  if (policy === "none" && (
    Number(request.headers.get("content-length") ?? "0") !== 0 || request.headers.has("transfer-encoding")
  )) return { status: 400, detail: "Invalid request" };
  if (policy !== "none" && !jsonType && (policy === "json" || contentType !== null)) {
    return { status: 415, detail: "Unsupported content type" };
  }
  if (!request.body) {
    if (policy === "none" || (policy === "optional-json" && contentType === null)) return {};
    return { status: 400, detail: "Invalid request" };
  }

  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (policy === "none" && size > 0) {
        void reader.cancel().catch(() => undefined);
        return { status: 400, detail: "Invalid request" };
      }
      if (size > MAX_REQUEST_BODY_BYTES) {
        // Don't wait for a producer to acknowledge cancellation before rejecting.
        void reader.cancel().catch(() => undefined);
        return { status: 413, detail: "Request body too large" };
      }
      chunks.push(value);
    }
    // Node/Next may provide an empty ReadableStream even when no body was sent.
    if (size === 0 && (policy === "none" || (policy === "optional-json" && contentType === null))) {
      return {};
    }
    if (!jsonType) return { status: 415, detail: "Unsupported content type" };
    const bytes = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
    const body = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    const parsed: unknown = JSON.parse(body);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { status: 400, detail: "Invalid request" };
    }
    return { body };
  } catch {
    return { status: 400, detail: "Invalid request" };
  } finally {
    reader.releaseLock();
  }
}
