"use client";

import { useState } from "react";
import { ArrowDown, LoaderCircle, RotateCcw } from "lucide-react";

import type { Message as ChatMessage, UiPreferences, UploadedAsset, VisionCapability } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Composer } from "@/components/Composer";
import { Message } from "@/components/Message";
import { RemoteVisionConsent } from "@/components/RemoteVisionConsent";
import { useChatScroll } from "@/components/useChatScroll";

interface ChatViewProps {
  messages: ChatMessage[];
  pendingMessage: ChatMessage | null;
  streamingMessage: ChatMessage | null;
  loading: boolean;
  sending: boolean;
  stopped?: boolean;
  onStop?: () => void;
  loadError: string | null;
  sendError: string | null;
  preferences: UiPreferences;
  onSend: (message: string, assets?: UploadedAsset[], allowRemoteVision?: boolean) => Promise<void>;
  vision?: VisionCapability | null;
  onReloadCapabilities?: () => void;
  onRetryLoad: () => void;
  onRetrySend: (allowRemoteVision?: boolean) => void;
}

function AssistantWaiting() {
  return (
    <div aria-label="Aevon is responding" className="flex items-center gap-2 pl-16 text-sm text-[#8f8983]">
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#b77d51]" />
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#b77d51] [animation-delay:120ms]" />
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#b77d51] [animation-delay:240ms]" />
    </div>
  );
}

export function ChatView({
  messages,
  pendingMessage,
  streamingMessage,
  loading,
  sending,
  stopped = false,
  onStop,
  loadError,
  sendError,
  preferences,
  onSend,
  onRetryLoad,
  onRetrySend,
  vision,
  onReloadCapabilities,
}: ChatViewProps) {
  const [retryConsentId, setRetryConsentId] = useState<string | null>(null);
  const retryHasImage = !!pendingMessage?.assets?.length;
  const retryRemote = retryHasImage && vision?.scope === "remote";
  const retryConsent = !!pendingMessage && retryConsentId === pendingMessage.id;
  const visionAvailable = vision?.enabled && (vision.scope === "local" || vision.scope === "remote");
  const retryBlocked = sending || (retryHasImage && (!visionAvailable || pendingMessage!.assets!.length > 1 || (retryRemote && !retryConsent)));
  const visibleMessages = [
    ...messages,
    ...(pendingMessage ? [pendingMessage] : []),
    ...(streamingMessage ? [streamingMessage] : []),
  ];
  const streamingContentLength = streamingMessage?.content.length ?? 0;
  const empty = !loading && !loadError && visibleMessages.length === 0;
  const { viewportRef, onScroll, showLatest, jumpToLatest } = useChatScroll({
    loading,
    sending,
    pendingId: pendingMessage?.id,
    contentVersion: `${messages.at(-1)?.id}:${visibleMessages.length}:${streamingContentLength}:${sending}:${stopped}:${sendError}`,
  });

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <p aria-live="polite" aria-atomic="true" className="sr-only" role="status">
        {sending ? "Aevon is responding." : stopped ? "Generation stopped." : sendError ? "Response failed." : "Ready to send."}
      </p>
      <div aria-label="Chat messages" className="min-h-0 flex-1 overflow-y-auto" onScroll={onScroll} ref={viewportRef} role="region" tabIndex={0}>
        <div
          className={cn(
            "mx-auto flex min-h-full w-full max-w-[58rem] flex-col px-5 sm:px-8",
            preferences.compactMessages ? "py-6" : "py-10",
          )}
        >
          {loading ? (
            <div className="flex flex-1 items-center justify-center text-sm text-[#928b85]">
              <LoaderCircle aria-hidden="true" className="mr-2 h-4 w-4 animate-spin text-[#bd8254]" />
              Loading conversation…
            </div>
          ) : loadError ? (
            <div className="m-auto max-w-md rounded-xl border border-[#754735]/60 bg-[#261812]/70 p-5 text-center">
              <p className="text-sm text-[#e0cfc4]">{loadError}</p>
              <button
                className="mt-4 inline-flex items-center gap-2 rounded-lg border border-[#986442]/60 px-3 py-2 text-sm text-[#dca778] hover:bg-[#37251a] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]"
                onClick={onRetryLoad}
                type="button"
              >
                <RotateCcw aria-hidden="true" className="h-4 w-4" />
                Retry
              </button>
            </div>
          ) : visibleMessages.length === 0 ? (
            <div className="flex flex-1 flex-col items-center justify-center pb-10 text-center">
              <h2 className="font-serif text-5xl tracking-[-0.03em] text-[#eee8e1] sm:text-6xl">Aevon</h2>
              <p className="mt-3 text-base text-[#948d86]">What are we working on?</p>
              <div className="mt-8 w-full max-w-[52rem] text-left">
                <Composer disabled={sending} onStop={onStop} enterToSend={preferences.enterToSend} onSend={onSend} vision={vision} onReloadCapabilities={onReloadCapabilities} />
              </div>
            </div>
          ) : (
            <div className={cn("flex flex-col", preferences.compactMessages ? "gap-6" : "gap-10")}>
              {visibleMessages.map((message) => (
                <Message
                  key={message.id}
                  message={message}
                  showTimestamp={preferences.showTimestamps}
                  wrapCode={preferences.wrapCode}
                />
              ))}
              {sending && !streamingContentLength ? <AssistantWaiting /> : null}
              {sendError || stopped ? (
                <div className="ml-16 flex flex-wrap items-center gap-3 rounded-xl border border-[#754735]/60 bg-[#261812]/70 px-4 py-3 text-sm text-[#e0cfc4]">
                  <span>{stopped ? "Generation stopped. You can retry this message." : sendError}</span>
                  {retryHasImage && !visionAvailable ? <p role="status">Vision is unavailable for this image. <button disabled={sending} onClick={onReloadCapabilities} type="button">Retry capabilities</button></p> : null}
                  {retryRemote ? <RemoteVisionConsent checked={retryConsent} disabled={sending} onChange={(checked) => setRetryConsentId(checked ? pendingMessage!.id : null)} /> : null}
                  <button
                    className="inline-flex items-center gap-1.5 text-[#dca778] hover:text-[#efbf93] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]"
                    disabled={retryBlocked}
                    onClick={() => { setRetryConsentId(null); onRetrySend(retryRemote && retryConsent); }}
                    type="button"
                  >
                    <RotateCcw aria-hidden="true" className="h-3.5 w-3.5" />
                    Retry
                  </button>
                </div>
              ) : null}
            </div>
          )}
        </div>
      </div>
      {showLatest ? (
        <button aria-label="Jump to latest" className="mx-auto mt-2 inline-flex shrink-0 items-center gap-2 rounded-full border border-[#805a3d]/65 bg-[#191713] px-3 py-1.5 text-xs text-[#dca778] hover:text-[#efbf93] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]" onClick={jumpToLatest} type="button">
          <ArrowDown aria-hidden="true" className="h-3.5 w-3.5" />Jump to latest
        </button>
      ) : null}
      {!empty && !loading && !loadError ? (
        <div className="shrink-0 bg-[#0d0e0e] px-5 pb-5 pt-3 sm:px-8 sm:pb-7">
          <div className="mx-auto w-full max-w-[52rem]">
            <Composer disabled={sending} onStop={onStop} enterToSend={preferences.enterToSend} onSend={onSend} vision={vision} onReloadCapabilities={onReloadCapabilities} />
          </div>
        </div>
      ) : null}
    </div>
  );
}
