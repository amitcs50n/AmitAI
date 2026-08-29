export type ConnectionState = "connecting" | "connected" | "disconnected";

export type AppView = "chat" | "settings" | "preferences" | "security";

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
  memory: unknown[] | null;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: string;
  content: string;
  created_at: string;
  metadata: MessageMetadata | null;
}

export interface ConversationDetail extends Conversation {
  messages: Message[];
}

export interface ChatRequest {
  conversation_id: string | null;
  message: string;
}

export interface ChatValidatorMetadata {
  retry_attempted: boolean;
  retry_passed: boolean | null;
  [key: string]: unknown;
}

export interface ChatMetadata {
  model: string | null;
  latency_ms: number | null;
  validator: ChatValidatorMetadata;
  tools: unknown[];
  memory: unknown[];
}

export interface ChatResponse {
  conversation_id: string;
  message_id: string;
  response: string;
  metadata: ChatMetadata;
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
