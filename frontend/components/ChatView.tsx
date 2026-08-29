"use client";

import { useEffect, useRef } from "react";
import { LoaderCircle, RotateCcw } from "lucide-react";

import type { Message as ChatMessage, UiPreferences } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Composer } from "@/components/Composer";
import { Message } from "@/components/Message";

interface ChatViewProps {
  messages: ChatMessage[];
  pendingMessage: ChatMessage | null;
  streamingMessage: ChatMessage | null;
  loading: boolean;
  sending: boolean;
  loadError: string | null;
  sendError: string | null;
  preferences: UiPreferences;
  onSend: (message: string) => Promise<void>;
  onRetryLoad: () => void;
  onRetrySend: () => void;
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
  loadError,
  sendError,
  preferences,
  onSend,
  onRetryLoad,
  onRetrySend,
}: ChatViewProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const visibleMessages = [
    ...messages,
    ...(pendingMessage ? [pendingMessage] : []),
    ...(streamingMessage ? [streamingMessage] : []),
  ];
  const streamingContentLength = streamingMessage?.content.length ?? 0;
  const hasStreamingMessage = streamingMessage !== null;
  const empty = !loading && !loadError && visibleMessages.length === 0;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: hasStreamingMessage ? "auto" : "smooth",
      block: "end",
    });
  }, [visibleMessages.length, sending, streamingContentLength, hasStreamingMessage]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto">
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
                <Composer disabled={sending} enterToSend={preferences.enterToSend} onSend={onSend} />
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
              {sendError ? (
                <div className="ml-16 flex flex-wrap items-center gap-3 rounded-xl border border-[#754735]/60 bg-[#261812]/70 px-4 py-3 text-sm text-[#e0cfc4]">
                  <span>{sendError}</span>
                  <button
                    className="inline-flex items-center gap-1.5 text-[#dca778] hover:text-[#efbf93] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]"
                    onClick={onRetrySend}
                    type="button"
                  >
                    <RotateCcw aria-hidden="true" className="h-3.5 w-3.5" />
                    Retry
                  </button>
                </div>
              ) : null}
              <div ref={bottomRef} />
            </div>
          )}
        </div>
      </div>
      {!empty && !loading && !loadError ? (
        <div className="shrink-0 bg-[#0d0e0e] px-5 pb-5 pt-3 sm:px-8 sm:pb-7">
          <div className="mx-auto w-full max-w-[52rem]">
            <Composer disabled={sending} enterToSend={preferences.enterToSend} onSend={onSend} />
          </div>
        </div>
      ) : null}
    </div>
  );
}
