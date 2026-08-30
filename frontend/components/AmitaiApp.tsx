"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { RotateCcw, X } from "lucide-react";

import {
  backendResponded,
  deleteConversation,
  getConversation,
  listConversations,
  renameConversation,
  sendChatStream,
} from "@/lib/api";
import type {
  AppView,
  ChatMetadata,
  ConnectionState,
  Conversation,
  ConversationDetail,
  Message,
  MessageMetadata,
  UiPreferences,
} from "@/lib/types";
import { DEFAULT_PREFERENCES } from "@/lib/types";
import { ChatView } from "@/components/ChatView";
import { DeveloperDrawer } from "@/components/DeveloperDrawer";
import { MemoryView } from "@/components/MemoryView";
import { PreferencesView } from "@/components/PreferencesView";
import { RenameDialog } from "@/components/RenameDialog";
import { SecurityView } from "@/components/SecurityView";
import { SettingsView } from "@/components/SettingsView";
import { Sidebar } from "@/components/Sidebar";
import { TopBar } from "@/components/TopBar";

const PREFERENCES_KEY = "amitai-ui-preferences-v1";
const SELECTED_CONVERSATION_KEY = "amitai-selected-conversation";

function loadInitialPreferences(): UiPreferences {
  if (typeof window === "undefined") return DEFAULT_PREFERENCES;
  try {
    const saved = localStorage.getItem(PREFERENCES_KEY);
    return saved
      ? { ...DEFAULT_PREFERENCES, ...(JSON.parse(saved) as Partial<UiPreferences>) }
      : DEFAULT_PREFERENCES;
  } catch {
    return DEFAULT_PREFERENCES;
  }
}

interface ActionAlert {
  message: string;
  retry?: () => void;
}

function temporaryUserMessage(content: string, conversationId: string | null): Message {
  return {
    id: `pending-${crypto.randomUUID()}`,
    conversation_id: conversationId ?? "pending",
    role: "user",
    content,
    created_at: new Date().toISOString(),
    metadata: null,
  };
}

function responseMetadata(metadata: ChatMetadata): MessageMetadata {
  return {
    model: metadata.model,
    latency_ms: metadata.latency_ms,
    input_tokens: metadata.input_tokens,
    output_tokens: metadata.output_tokens,
    validator: metadata.validator,
    tools: metadata.tools,
    memory: metadata.memory,
  };
}

function updateConnectionFromError(
  error: unknown,
  setConnection: (state: ConnectionState) => void,
) {
  setConnection(backendResponded(error) ? "connected" : "disconnected");
}

export function AmitaiApp() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [conversation, setConversation] = useState<ConversationDetail | null>(null);
  const [view, setView] = useState<AppView>("chat");
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [developerOpen, setDeveloperOpen] = useState(false);
  const [loadingConversation, setLoadingConversation] = useState(true);
  const [sending, setSending] = useState(false);
  const [pendingMessage, setPendingMessage] = useState<Message | null>(null);
  const [streamingMessage, setStreamingMessage] = useState<Message | null>(null);
  const [failedInput, setFailedInput] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [sendError, setSendError] = useState<string | null>(null);
  const [actionAlert, setActionAlert] = useState<ActionAlert | null>(null);
  const [renameOpen, setRenameOpen] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [renameError, setRenameError] = useState<string | null>(null);
  const [preferences, setPreferences] = useState<UiPreferences>(DEFAULT_PREFERENCES);
  const conversationRequestRef = useRef(0);
  const streamAbortRef = useRef<AbortController | null>(null);

  const markConnected = useCallback(() => setConnection("connected"), []);

  const loadConversation = useCallback(async (id: string) => {
    const requestId = ++conversationRequestRef.current;
    setLoadingConversation(true);
    setLoadError(null);
    setPendingMessage(null);
    setStreamingMessage(null);
    setFailedInput(null);
    setSendError(null);
    try {
      const detail = await getConversation(id);
      if (requestId !== conversationRequestRef.current) return;
      setConversation(detail);
      setSelectedId(id);
      localStorage.setItem(SELECTED_CONVERSATION_KEY, id);
      markConnected();
    } catch (error) {
      if (requestId !== conversationRequestRef.current) return;
      setConversation(null);
      updateConnectionFromError(error, setConnection);
      setLoadError(
        backendResponded(error)
          ? "Unable to load this conversation."
          : "Unable to connect to Aevon.",
      );
    } finally {
      if (requestId === conversationRequestRef.current) setLoadingConversation(false);
    }
  }, [markConnected]);

  const initialize = useCallback(async () => {
    const requestId = ++conversationRequestRef.current;
    setConnection("connecting");
    setLoadingConversation(true);
    setLoadError(null);
    try {
      const listed = await listConversations();
      if (requestId !== conversationRequestRef.current) return;
      setConversations(listed);
      markConnected();

      const remembered = localStorage.getItem(SELECTED_CONVERSATION_KEY);
      const initialId =
        (remembered && listed.some((item) => item.id === remembered) ? remembered : null) ??
        listed[0]?.id ??
        null;
      if (initialId) {
        setSelectedId(initialId);
        try {
          const detail = await getConversation(initialId);
          if (requestId !== conversationRequestRef.current) return;
          setConversation(detail);
          markConnected();
        } catch (error) {
          if (requestId !== conversationRequestRef.current) return;
          setConversation(null);
          updateConnectionFromError(error, setConnection);
          setLoadError(
            backendResponded(error)
              ? "Unable to load this conversation."
              : "Unable to connect to Aevon.",
          );
        }
      } else {
        setSelectedId(null);
        setConversation(null);
      }
    } catch (error) {
      if (requestId !== conversationRequestRef.current) return;
      setConversations([]);
      setConversation(null);
      updateConnectionFromError(error, setConnection);
      setLoadError(
        backendResponded(error)
          ? "Unable to load conversations."
          : "Unable to connect to Aevon.",
      );
    } finally {
      if (requestId === conversationRequestRef.current) setLoadingConversation(false);
    }
  }, [markConnected]);

  useEffect(() => {
    const initializeTimer = window.setTimeout(() => {
      setPreferences(loadInitialPreferences());
      if (window.innerWidth < 1024) setSidebarOpen(false);
      void initialize();
    }, 0);
    return () => {
      window.clearTimeout(initializeTimer);
      streamAbortRef.current?.abort();
    };
  }, [initialize]);

  function updatePreferences(patch: Partial<UiPreferences>) {
    setPreferences((current) => {
      const next = { ...current, ...patch };
      localStorage.setItem(PREFERENCES_KEY, JSON.stringify(next));
      return next;
    });
  }

  async function refreshConversationList() {
    try {
      const listed = await listConversations();
      setConversations(listed);
      markConnected();
    } catch (error) {
      updateConnectionFromError(error, setConnection);
      setActionAlert({ message: "Conversation saved, but the sidebar could not refresh.", retry: () => void refreshConversationList() });
    }
  }

  function handleNewConversation() {
    if (sending) return;
    conversationRequestRef.current += 1;
    setLoadingConversation(false);
    setView("chat");
    setSelectedId(null);
    setConversation(null);
    setPendingMessage(null);
    setStreamingMessage(null);
    setFailedInput(null);
    setSendError(null);
    setLoadError(null);
    setActionAlert(null);
    localStorage.removeItem(SELECTED_CONVERSATION_KEY);
    if (window.innerWidth < 1024) setSidebarOpen(false);
  }

  function handleSelectConversation(id: string) {
    if (sending) return;
    setView("chat");
    setSelectedId(id);
    if (window.innerWidth < 1024) setSidebarOpen(false);
    void loadConversation(id);
  }

  function handleNavigate(nextView: AppView) {
    setView(nextView);
    if (window.innerWidth < 1024) setSidebarOpen(false);
  }

  async function submitMessage(message: string, retry = false) {
    const targetId = selectedId;
    const userMessage = retry && pendingMessage ? pendingMessage : temporaryUserMessage(message, targetId);
    const streamMessageId = `streaming-${crypto.randomUUID()}`;
    const streamCreatedAt = new Date().toISOString();
    const abortController = new AbortController();
    let streamedText = "";

    streamAbortRef.current?.abort();
    streamAbortRef.current = abortController;
    if (!retry) setPendingMessage(userMessage);
    setStreamingMessage(null);
    setSending(true);
    setSendError(null);
    setFailedInput(null);

    try {
      const result = await sendChatStream(
        { conversation_id: targetId, message },
        {
          onStart: markConnected,
          onText: (delta) => {
            streamedText += delta;
            if (!streamedText) return;
            setStreamingMessage({
              id: streamMessageId,
              conversation_id: targetId ?? "pending",
              role: "assistant",
              content: streamedText,
              created_at: streamCreatedAt,
              metadata: null,
            });
          },
          onFinal: (finalResponse) => {
            setStreamingMessage({
              id: finalResponse.message_id,
              conversation_id: finalResponse.conversation_id,
              role: "assistant",
              content: finalResponse.response,
              created_at: streamCreatedAt,
              metadata: responseMetadata(finalResponse.metadata),
            });
          },
        },
        abortController.signal,
      );
      markConnected();
      const assistantMessage: Message = {
        id: result.message_id,
        conversation_id: result.conversation_id,
        role: "assistant",
        content: result.response,
        created_at: new Date().toISOString(),
        metadata: responseMetadata(result.metadata),
      };
      const now = new Date().toISOString();
      const fallback: ConversationDetail = conversation
        ? {
            ...conversation,
            id: result.conversation_id,
            updated_at: now,
            messages: [...conversation.messages, userMessage, assistantMessage],
          }
        : {
            id: result.conversation_id,
            title: "Conversation",
            created_at: now,
            updated_at: now,
            archived: false,
            messages: [userMessage, assistantMessage],
          };

      setSelectedId(result.conversation_id);
      setConversation(fallback);
      setPendingMessage(null);
      setStreamingMessage(null);
      localStorage.setItem(SELECTED_CONVERSATION_KEY, result.conversation_id);

      try {
        const persisted = await getConversation(result.conversation_id);
        setConversation(persisted);
        markConnected();
      } catch (error) {
        updateConnectionFromError(error, setConnection);
        setActionAlert({
          message: "Your response was saved, but persisted history could not be reloaded.",
          retry: () => void loadConversation(result.conversation_id),
        });
      }
      await refreshConversationList();
    } catch (error) {
      setStreamingMessage(null);
      if (!abortController.signal.aborted) {
        updateConnectionFromError(error, setConnection);
        setFailedInput(message);
        setSendError(
          backendResponded(error) ? "Generation failed. Try again." : "Unable to connect to Aevon.",
        );
      }
    } finally {
      if (streamAbortRef.current === abortController) streamAbortRef.current = null;
      setSending(false);
    }
  }

  function retrySend() {
    if (failedInput && !sending) void submitMessage(failedInput, true);
  }

  async function performDelete(id: string) {
    try {
      await deleteConversation(id);
      markConnected();
      const remaining = conversations.filter((item) => item.id !== id);
      setConversations(remaining);
      setConversation(null);
      setPendingMessage(null);
      setStreamingMessage(null);
      const nextId = remaining[0]?.id ?? null;
      setSelectedId(nextId);
      if (nextId) {
        localStorage.setItem(SELECTED_CONVERSATION_KEY, nextId);
        await loadConversation(nextId);
      } else {
        localStorage.removeItem(SELECTED_CONVERSATION_KEY);
        setLoadError(null);
      }
    } catch (error) {
      updateConnectionFromError(error, setConnection);
      setActionAlert({
        message: "Unable to delete this conversation.",
        retry: () => void performDelete(id),
      });
    }
  }

  function handleDelete() {
    if (!selectedId || !conversation || sending || loadingConversation) return;
    if (window.confirm(`Delete “${conversation.title}”? This cannot be undone.`)) {
      void performDelete(selectedId);
    }
  }

  async function handleRename(title: string) {
    if (!selectedId || sending || loadingConversation) return;
    setRenaming(true);
    setRenameError(null);
    try {
      const renamed = await renameConversation(selectedId, title);
      markConnected();
      setConversations((current) => current.map((item) => (item.id === renamed.id ? renamed : item)));
      setConversation((current) => (current ? { ...current, ...renamed } : current));
      setRenameOpen(false);
      await refreshConversationList();
    } catch (error) {
      updateConnectionFromError(error, setConnection);
      setRenameError("Unable to rename this conversation. Try again.");
    } finally {
      setRenaming(false);
    }
  }

  function handleExport() {
    if (!conversation) return;
    const blob = new Blob([JSON.stringify(conversation, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${conversation.title.replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "").toLowerCase() || "aevon-conversation"}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  const latestMetadata = useMemo(() => {
    const messages = conversation?.messages ?? [];
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      if (messages[index].role === "assistant" && messages[index].metadata) return messages[index].metadata;
    }
    return null;
  }, [conversation]);

  return (
    <main className="flex h-dvh min-h-[32rem] overflow-hidden bg-[#0d0e0e] text-[#ded8d2] selection:bg-[#9f6842]/50">
      <Sidebar
        conversations={conversations}
        creating={false}
        currentView={view}
        interactionDisabled={sending || loadingConversation}
        onClose={() => setSidebarOpen(false)}
        onNavigate={handleNavigate}
        onNewConversation={handleNewConversation}
        onSelectConversation={handleSelectConversation}
        open={sidebarOpen}
        selectedId={selectedId}
      />
      <div className="relative flex min-w-0 flex-1 flex-col bg-[#0d0e0e]">
        <TopBar
          connection={connection}
          conversationActionsDisabled={sending || loadingConversation}
          hasConversation={Boolean(conversation)}
          onDelete={handleDelete}
          onDeveloperDetails={() => setDeveloperOpen(true)}
          onExport={handleExport}
          onOpenSidebar={() => setSidebarOpen(true)}
          onRename={() => {
            setRenameError(null);
            setRenameOpen(true);
          }}
        />

        {view === "chat" ? (
          <ChatView
            loadError={loadError}
            loading={loadingConversation}
            messages={conversation?.messages ?? []}
            onRetryLoad={() => (selectedId ? void loadConversation(selectedId) : void initialize())}
            onRetrySend={retrySend}
            onSend={(message) => submitMessage(message)}
            pendingMessage={pendingMessage}
            preferences={preferences}
            sendError={sendError}
            sending={sending}
            streamingMessage={streamingMessage}
          />
        ) : view === "memory" ? (
          <MemoryView />
        ) : view === "settings" ? (
          <SettingsView connection={connection} onChange={updatePreferences} preferences={preferences} />
        ) : view === "preferences" ? (
          <PreferencesView onChange={updatePreferences} preferences={preferences} />
        ) : (
          <SecurityView />
        )}

        {actionAlert ? (
          <div className="absolute bottom-5 left-1/2 z-30 flex w-[calc(100%-2rem)] max-w-lg -translate-x-1/2 items-center gap-3 rounded-xl border border-[#754735]/70 bg-[#241813] px-4 py-3 text-sm text-[#e0cfc4] shadow-xl">
            <span className="min-w-0 flex-1">{actionAlert.message}</span>
            {actionAlert.retry ? (
              <button
                className="inline-flex shrink-0 items-center gap-1.5 text-[#dca778] hover:text-[#efbf93] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]"
                onClick={actionAlert.retry}
                type="button"
              >
                <RotateCcw aria-hidden="true" className="h-3.5 w-3.5" />
                Retry
              </button>
            ) : null}
            <button
              aria-label="Dismiss notification"
              className="shrink-0 rounded-md p-1 text-[#a79b92] hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]"
              onClick={() => setActionAlert(null)}
              type="button"
            >
              <X aria-hidden="true" className="h-4 w-4" />
            </button>
          </div>
        ) : null}
      </div>

      <DeveloperDrawer metadata={latestMetadata} onClose={() => setDeveloperOpen(false)} open={developerOpen} />
      {renameOpen ? (
        <RenameDialog
          error={renameError}
          initialTitle={conversation?.title ?? ""}
          onCancel={() => setRenameOpen(false)}
          onRename={(title) => void handleRename(title)}
          saving={renaming}
        />
      ) : null}
    </main>
  );
}
