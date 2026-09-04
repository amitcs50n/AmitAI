"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { RotateCcw, X } from "lucide-react";

import {
  backendResponded,
  chatErrorMessage,
  deleteConversation,
  getConversation,
  listConversations,
  renameConversation,
  sendChatStream,
  getCapabilities,
} from "@/lib/api";
import type {
  AppView,
  ChatMetadata,
  ChatResponse,
  ConnectionState,
  InferenceMode,
  Conversation,
  ConversationDetail,
  Message,
  MessageMetadata,
  UiPreferences,
  UploadedAsset,
  VisionCapability,
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

function temporaryUserMessage(content: string, conversationId: string | null, assets: UploadedAsset[]): Message {
  return {
    id: `pending-${crypto.randomUUID()}`,
    conversation_id: conversationId ?? "pending",
    role: "user",
    content,
    created_at: new Date().toISOString(),
    metadata: null,
    assets,
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
  // Stable across a new conversation's first save, so completion does not
  // remount ChatView and reset a reader who has scrolled away from the bottom.
  const [chatViewKey, setChatViewKey] = useState(0);
  const [conversation, setConversation] = useState<ConversationDetail | null>(null);
  const [view, setView] = useState<AppView>("chat");
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [developerOpen, setDeveloperOpen] = useState(false);
  const [loadingConversation, setLoadingConversation] = useState(true);
  const [sending, setSending] = useState(false);
  const [finishing, setFinishing] = useState(false);
  const [stopped, setStopped] = useState(false);
  const [pendingMessage, setPendingMessage] = useState<Message | null>(null);
  const [streamingMessage, setStreamingMessage] = useState<Message | null>(null);
  const [failedInput, setFailedInput] = useState<string | null>(null);
  const [vision, setVision] = useState<VisionCapability | null>(null);
  const [inferenceMode, setInferenceMode] = useState<InferenceMode>("unknown");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [sendError, setSendError] = useState<string | null>(null);
  const [actionAlert, setActionAlert] = useState<ActionAlert | null>(null);
  const [renameOpen, setRenameOpen] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [renameError, setRenameError] = useState<string | null>(null);
  const [preferences, setPreferences] = useState<UiPreferences>(DEFAULT_PREFERENCES);
  const conversationRequestRef = useRef(0);
  const streamAbortRef = useRef<AbortController | null>(null);
  const committedRef = useRef(false);
  const mountedRef = useRef(false);

  const markConnected = useCallback(() => setConnection("connected"), []);
  const loadCapabilities = useCallback(async () => {
    try {
      const capabilities = await getCapabilities();
      if (mountedRef.current) {
        setVision(capabilities.vision);
        setInferenceMode(capabilities.inference?.mode ?? "unknown");
      }
    } catch {
      if (mountedRef.current) { setVision(null); setInferenceMode("unknown"); }
    }
  }, []);

  const loadConversation = useCallback(async (id: string) => {
    setChatViewKey((current) => current + 1);
    setStopped(false);
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
    mountedRef.current = true;
    const initializeTimer = window.setTimeout(() => {
      setPreferences(loadInitialPreferences());
      if (window.innerWidth < 1024) setSidebarOpen(false);
      void initialize();
      void loadCapabilities();
    }, 0);
    return () => {
      window.clearTimeout(initializeTimer);
      mountedRef.current = false;
      conversationRequestRef.current += 1;
      streamAbortRef.current?.abort();
      streamAbortRef.current = null;
    };
  }, [initialize, loadCapabilities]);

  function updatePreferences(patch: Partial<UiPreferences>) {
    setPreferences((current) => {
      const next = { ...current, ...patch };
      localStorage.setItem(PREFERENCES_KEY, JSON.stringify(next));
      return next;
    });
  }

  async function refreshConversationList() {
    const requestId = conversationRequestRef.current;
    try {
      const listed = await listConversations();
      if (!mountedRef.current || requestId !== conversationRequestRef.current) return;
      setConversations(listed);
      markConnected();
    } catch (error) {
      if (!mountedRef.current || requestId !== conversationRequestRef.current) return;
      updateConnectionFromError(error, setConnection);
      setActionAlert({ message: "Conversation saved, but the sidebar could not refresh.", retry: () => void refreshConversationList() });
    }
  }

  function handleNewConversation() {
    if (streamAbortRef.current) return;
    setChatViewKey((current) => current + 1);
    setStopped(false);
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
    if (streamAbortRef.current) return;
    setView("chat");
    setSelectedId(id);
    if (window.innerWidth < 1024) setSidebarOpen(false);
    void loadConversation(id);
  }

  function handleNavigate(nextView: AppView) {
    setView(nextView);
    if (window.innerWidth < 1024) setSidebarOpen(false);
  }

  async function submitMessage(message: string, retry = false, assets: UploadedAsset[] = [], allowRemoteVision = false) {
    // Reserve ownership synchronously, before React state updates or any await.
    if (streamAbortRef.current || !mountedRef.current) return;
    const requestId = ++conversationRequestRef.current;
    const targetId = selectedId;
    const userMessage = retry && pendingMessage ? pendingMessage : temporaryUserMessage(message, targetId, assets);
    const streamMessageId = `streaming-${crypto.randomUUID()}`;
    const streamCreatedAt = new Date().toISOString();
    const abortController = new AbortController();
    let streamedText = "";
    const committed: { response: ChatResponse | null } = { response: null };
    const isCurrent = () => mountedRef.current && streamAbortRef.current === abortController;
    const isSelected = () => mountedRef.current && requestId === conversationRequestRef.current;

    streamAbortRef.current = abortController;
    committedRef.current = false;
    if (!retry) setPendingMessage(userMessage);
    setStreamingMessage(null);
    setSending(true);
    setFinishing(false);
    setStopped(false);
    setActionAlert(null);
    setSendError(null);
    setFailedInput(null);

    function showSavedResponse(result: ChatResponse) {
      const now = new Date().toISOString();
      const assistantMessage: Message = {
        id: result.message_id,
        conversation_id: result.conversation_id,
        role: "assistant",
        content: result.response,
        created_at: now,
        metadata: responseMetadata(result.metadata),
      };
      const fallback: ConversationDetail = {
        ...(conversation ?? { title: "Conversation", created_at: now, archived: false }),
        id: result.conversation_id,
        updated_at: now,
        messages: [...(conversation?.messages ?? []), userMessage, assistantMessage],
      };
      setSelectedId(result.conversation_id);
      setConversation(fallback);
      setPendingMessage(null);
      setStreamingMessage(null);
      localStorage.setItem(SELECTED_CONVERSATION_KEY, result.conversation_id);
    }

    try {
      const result = await sendChatStream(
        { conversation_id: targetId, message, asset_ids: (userMessage.assets ?? []).map((asset) => asset.id), allow_remote_vision: allowRemoteVision },
        {
          onStart: () => { if (isCurrent()) markConnected(); },
          onText: (delta) => {
            if (!isCurrent()) return;
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
            if (!isCurrent()) return;
            // The server commits before final; Stop cannot undo a saved turn.
            committed.response = finalResponse;
            committedRef.current = true;
            setFinishing(true);
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
      if (!isCurrent()) return;
      markConnected();
      showSavedResponse(result);
      streamAbortRef.current = null;
      setSending(false);
      setFinishing(false);

      try {
        const persisted = await getConversation(result.conversation_id);
        if (!isSelected()) return;
        setConversation(persisted);
        markConnected();
      } catch (error) {
        if (!isSelected()) return;
        updateConnectionFromError(error, setConnection);
        setActionAlert({
          message: "Your response was saved, but persisted history could not be reloaded.",
          retry: () => void loadConversation(result.conversation_id),
        });
      }
      await refreshConversationList();
    } catch (error) {
      if (!isCurrent()) return;
      setStreamingMessage(null);
      if (!abortController.signal.aborted) {
        updateConnectionFromError(error, setConnection);
        if (committed.response) {
          // Missing done is a failure, but retrying a confirmed commit would
          // duplicate the turn. Keep the saved fallback and offer a reload.
          const saved = committed.response;
          showSavedResponse(saved);
          setActionAlert({
            message: "Your response was saved, but the stream did not finish normally. Reload history to verify it.",
            retry: () => void loadConversation(saved.conversation_id),
          });
          return;
        }
        setFailedInput(message);
        setSendError(
          chatErrorMessage(error),
        );
      }
    } finally {
      if (isCurrent()) {
        streamAbortRef.current = null;
        setSending(false);
        setFinishing(false);
      }
    }
  }

  function stopGeneration() {
    const controller = streamAbortRef.current;
    if (!controller || committedRef.current) return;
    streamAbortRef.current = null; // Invalidate callbacks before abort settles.
    controller.abort();
    setStreamingMessage(null);
    setSending(false);
    setFinishing(false);
    setStopped(true);
    setSendError(null);
    setFailedInput(pendingMessage?.content ?? null);
  }

  function retrySend(allowRemoteVision = false) {
    if (failedInput && !sending) void submitMessage(failedInput, true, [], allowRemoteVision);
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
            key={chatViewKey}
            loadError={loadError}
            loading={loadingConversation}
            messages={conversation?.messages ?? []}
            onRetryLoad={() => (selectedId ? void loadConversation(selectedId) : void initialize())}
            onRetrySend={retrySend}
            onSend={(message, assets, consent) => submitMessage(message, false, assets, consent)}
            vision={vision}
            onReloadCapabilities={() => void loadCapabilities()}
            pendingMessage={pendingMessage}
            preferences={preferences}
            sendError={sendError}
            sending={sending}
            stopped={stopped}
            onStop={sending && !finishing ? stopGeneration : undefined}
            streamingMessage={streamingMessage}
          />
        ) : view === "memory" ? (
          <MemoryView />
        ) : view === "settings" ? (
          <SettingsView connection={connection} inferenceMode={inferenceMode} vision={vision} onReloadCapabilities={() => void loadCapabilities()} onChange={updatePreferences} preferences={preferences} />
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
