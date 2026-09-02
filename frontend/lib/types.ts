export type ConnectionState = "connecting" | "connected" | "disconnected";

export type AppView = "chat" | "memory" | "settings" | "preferences" | "security";

export const MEMORY_CATEGORIES = [
  "preference",
  "profile",
  "project",
  "workflow",
  "instruction",
] as const;

export type MemoryCategory = (typeof MEMORY_CATEGORIES)[number];
export type MemoryStatus = "active" | "deleted";
export type MemorySensitivity = "local_only" | "remote_allowed";
export type MemoryOperation = "current" | "retrieved" | "stored" | "updated" | "deleted";

export interface MemorySource {
  conversation_id: string | null;
  message_id: string | null;
}

export interface MemoryRecord {
  id: string;
  operation: MemoryOperation;
  category: MemoryCategory;
  key: string;
  value: string | null;
  sensitivity: MemorySensitivity;
  status: MemoryStatus;
  source: MemorySource;
  updated_at: string;
}

export interface MemoryReference {
  id: string;
  operation: MemoryOperation;
  category: MemoryCategory;
  key: string;
  status: MemoryStatus;
  source: MemorySource;
  updated_at: string;
}

export interface MemoryCreateInput {
  category: MemoryCategory;
  key: string;
  value: string;
  sensitivity?: MemorySensitivity;
}

export interface MemoryUpdateInput {
  value?: string;
  sensitivity?: MemorySensitivity;
}

export interface MemoryListOptions {
  query?: string;
  category?: MemoryCategory;
  status?: MemoryStatus;
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  archived: boolean;
}

export interface MessageMetadata {
  model: string | null;
  latency_ms: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  validator: Record<string, unknown> | null;
  tools: unknown[] | null;
  memory: MemoryReference[] | null;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: string;
  content: string;
  created_at: string;
  metadata: MessageMetadata | null;
  assets?: UploadedAsset[];
}

export interface UploadedAsset {
  id: string;
  kind: "image";
  original_filename: string;
  content_type: "image/png";
  byte_size: number;
  width: number;
  height: number;
  sha256: string;
  created_at: string;
  conversation_id: string | null;
  persistence_mode: "temporary" | "conversation";
  processing_scope: "local_only";
}

export interface ConversationDetail extends Conversation {
  messages: Message[];
}

export interface ChatRequest {
  conversation_id: string | null;
  message: string;
  asset_ids?: string[];
}

export interface ChatValidatorMetadata {
  retry_attempted: boolean;
  retry_passed: boolean | null;
  [key: string]: unknown;
}

export interface ChatMetadata {
  model: string | null;
  latency_ms: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  validator: ChatValidatorMetadata;
  tools: unknown[];
  memory: MemoryReference[];
}

export interface ChatResponse {
  conversation_id: string;
  message_id: string;
  response: string;
  metadata: ChatMetadata;
}

export interface ChatStreamStart {
  conversation_id: string | null;
}

export interface ChatStreamText {
  delta: string;
}

export interface ChatStreamError {
  detail: string;
}

export interface UiPreferences {
  enterToSend: boolean;
  showTimestamps: boolean;
  compactMessages: boolean;
  wrapCode: boolean;
  developerMode: boolean;
  showToolActivity: boolean;
  showValidatorDetails: boolean;
}

export const DEFAULT_PREFERENCES: UiPreferences = {
  enterToSend: true,
  showTimestamps: false,
  compactMessages: false,
  wrapCode: false,
  developerMode: false,
  showToolActivity: true,
  showValidatorDetails: true,
};
