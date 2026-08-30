import type {
  ChatRequest,
  ChatResponse,
  ChatStreamError,
  ChatStreamStart,
  ChatStreamText,
  Conversation,
  ConversationDetail,
  MemoryCreateInput,
  MemoryListOptions,
  MemoryRecord,
  MemoryUpdateInput,
} from "@/lib/types";
import { parseSseStream } from "./sse.ts";

export class ApiError extends Error {
  public readonly status: number | null;
  public readonly backendReached: boolean;

  constructor(
    message: string,
    status: number | null,
    backendReached: boolean,
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.backendReached = backendReached;
  }
}

async function readError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    return typeof body.detail === "string" ? body.detail : "Request failed";
  } catch {
    return "Request failed";
  }
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;

  try {
    response = await fetch(path, {
      cache: "no-store",
      ...init,
      headers: {
        Accept: "application/json",
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...init?.headers,
      },
    });
  } catch (error) {
    throw new ApiError(
      error instanceof Error ? error.message : "Unable to reach the backend",
      null,
      false,
    );
  }

  if (!response.ok) {
    const contentType = response.headers.get("content-type") ?? "";
    throw new ApiError(
      await readError(response),
      response.status,
      contentType.toLowerCase().includes("application/json"),
    );
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export function listConversations(): Promise<Conversation[]> {
  return apiFetch<Conversation[]>("/api/conversations");
}

export function getConversation(id: string): Promise<ConversationDetail> {
  return apiFetch<ConversationDetail>(`/api/conversations/${encodeURIComponent(id)}`);
}

export function createConversation(title?: string): Promise<Conversation> {
  return apiFetch<Conversation>("/api/conversations", {
    method: "POST",
    ...(title ? { body: JSON.stringify({ title }) } : {}),
  });
}

export function renameConversation(id: string, title: string): Promise<Conversation> {
  return apiFetch<Conversation>(`/api/conversations/${encodeURIComponent(id)}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });
}

export function deleteConversation(id: string): Promise<void> {
  return apiFetch<void>(`/api/conversations/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}

export function listMemories(options: MemoryListOptions = {}): Promise<MemoryRecord[]> {
  const query = options.query?.trim();
  const parameters = new URLSearchParams();
  if (query) {
    parameters.set("query", query);
  } else {
    if (options.category) parameters.set("category", options.category);
    if (options.status && options.status !== "active") {
      parameters.set("status", options.status);
    }
  }
  const suffix = parameters.size ? `?${parameters.toString()}` : "";
  return apiFetch<MemoryRecord[]>(`/api/memory${suffix}`);
}

export function createMemory(payload: MemoryCreateInput): Promise<MemoryRecord> {
  return apiFetch<MemoryRecord>("/api/memory", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateMemory(id: string, payload: MemoryUpdateInput): Promise<MemoryRecord> {
  return apiFetch<MemoryRecord>(`/api/memory/${encodeURIComponent(id)}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export function deleteMemory(id: string): Promise<void> {
  return apiFetch<void>(`/api/memory/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}

export function sendChat(payload: ChatRequest): Promise<ChatResponse> {
  return apiFetch<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export interface ChatStreamHandlers {
  onStart?: (data: ChatStreamStart) => void;
  onText: (delta: string) => void;
  onFinal?: (response: ChatResponse) => void;
  onDone?: () => void;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNullableString(value: unknown): value is string | null {
  return value === null || typeof value === "string";
}

function isNullableNumber(value: unknown): value is number | null {
  return value === null || typeof value === "number";
}

function isChatStreamError(value: unknown): value is ChatStreamError {
  return isRecord(value) && typeof value.detail === "string";
}

function isChatStreamText(value: unknown): value is ChatStreamText {
  return isRecord(value) && typeof value.delta === "string";
}

function streamProtocolError(status: number, detail = "Invalid streaming response"): ApiError {
  return new ApiError(detail, status, true);
}

function parseStreamData(data: string, status: number): unknown {
  try {
    return JSON.parse(data) as unknown;
  } catch {
    throw streamProtocolError(status);
  }
}

function parseFinalResponse(value: unknown, status: number): ChatResponse {
  const metadata = isRecord(value) && isRecord(value.metadata) ? value.metadata : null;
  const validator = metadata && isRecord(metadata.validator) ? metadata.validator : null;
  if (
    !isRecord(value) ||
    typeof value.conversation_id !== "string" ||
    typeof value.message_id !== "string" ||
    typeof value.response !== "string" ||
    !metadata ||
    !isNullableString(metadata.model) ||
    !isNullableNumber(metadata.latency_ms) ||
    !isNullableNumber(metadata.input_tokens) ||
    !isNullableNumber(metadata.output_tokens) ||
    !validator ||
    typeof validator.retry_attempted !== "boolean" ||
    !(validator.retry_passed === null || typeof validator.retry_passed === "boolean") ||
    !Array.isArray(metadata.tools) ||
    !Array.isArray(metadata.memory)
  ) {
    throw streamProtocolError(status);
  }
  return value as unknown as ChatResponse;
}

export async function sendChatStream(
  payload: ChatRequest,
  handlers: ChatStreamHandlers,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  let response: Response;

  try {
    response = await fetch("/api/chat/stream", {
      method: "POST",
      cache: "no-store",
      headers: {
        Accept: "text/event-stream",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
      signal,
    });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new ApiError(
      error instanceof Error ? error.message : "Unable to reach the backend",
      null,
      false,
    );
  }

  if (!response.ok) {
    throw new ApiError(await readError(response), response.status, true);
  }

  const contentType = response.headers.get("content-type")?.toLowerCase() ?? "";
  if (!contentType.includes("text/event-stream") || !response.body) {
    throw streamProtocolError(response.status);
  }

  let started = false;
  let emittedText = "";
  let finalResponse: ChatResponse | null = null;

  try {
    for await (const event of parseSseStream(response.body)) {
      if (!["error", "start", "text", "final", "done"].includes(event.event)) continue;
      const data = parseStreamData(event.data, response.status);

      if (event.event === "error") {
        const detail = isChatStreamError(data) ? data.detail : "Assistant generation failed";
        throw new ApiError(detail, response.status, true);
      }

      if (event.event === "start") {
        if (
          started ||
          finalResponse ||
          !isRecord(data) ||
          !isNullableString(data.conversation_id)
        ) {
          throw streamProtocolError(response.status);
        }
        started = true;
        handlers.onStart?.({ conversation_id: data.conversation_id });
        continue;
      }

      if (event.event === "text") {
        if (
          !started ||
          finalResponse ||
          !isChatStreamText(data)
        ) {
          throw streamProtocolError(response.status);
        }
        const delta = data.delta;
        emittedText += delta;
        handlers.onText(delta);
        continue;
      }

      if (event.event === "final") {
        if (!started || finalResponse) throw streamProtocolError(response.status);
        finalResponse = parseFinalResponse(data, response.status);
        if (emittedText !== finalResponse.response) {
          throw streamProtocolError(response.status, "Streamed text did not match the final response");
        }
        handlers.onFinal?.(finalResponse);
        continue;
      }

      if (event.event === "done") {
        if (!finalResponse || !isRecord(data)) throw streamProtocolError(response.status);
        handlers.onDone?.();
        return finalResponse;
      }
    }
  } catch (error) {
    if (signal?.aborted || error instanceof ApiError) throw error;
    throw new ApiError(
      error instanceof Error ? error.message : "Streaming response failed",
      response.status,
      true,
    );
  }

  throw streamProtocolError(response.status, "Stream ended before completion");
}

export function backendResponded(error: unknown): boolean {
  return error instanceof ApiError && error.backendReached;
}
