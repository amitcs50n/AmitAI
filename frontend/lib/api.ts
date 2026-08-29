import type {
  ChatRequest,
  ChatResponse,
  Conversation,
  ConversationDetail,
} from "@/lib/types";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number | null,
    public readonly backendReached: boolean,
  ) {
    super(message);
    this.name = "ApiError";
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

export function sendChat(payload: ChatRequest): Promise<ChatResponse> {
  return apiFetch<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function backendResponded(error: unknown): boolean {
  return error instanceof ApiError && error.backendReached;
}
