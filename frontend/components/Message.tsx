import type { Message as ChatMessage } from "@/lib/types";
import { formatMessageTime } from "@/lib/dates";
import { AevonMonogram } from "@/components/AevonMonogram";
import { MarkdownContent } from "@/components/MarkdownContent";
import { ToolEventCard } from "@/components/ToolEventCard";
import { AssetPreview } from "@/components/AssetPreview";

interface MessageProps {
  message: ChatMessage;
  showTimestamp: boolean;
  wrapCode: boolean;
}

export function Message({ message, showTimestamp, wrapCode }: MessageProps) {
  if (message.role === "user") {
    return (
      <article className="ml-auto max-w-[78%] sm:max-w-[70%]">
        <div className="rounded-2xl rounded-tr-sm border border-[#6f5138]/45 bg-[#2a211a]/85 px-4 py-3 font-serif text-[1.02rem] leading-7 text-[#eee8e1] shadow-sm">
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
          {message.assets?.length ? <div className="mt-2 flex flex-wrap gap-2">
            {message.assets.map((asset) => <AssetPreview asset={asset} key={asset.id} />)}
          </div> : null}
        </div>
        {showTimestamp ? (
          <time className="mt-1.5 block text-right text-[0.68rem] text-[#746d67]" dateTime={message.created_at}>
            {formatMessageTime(message.created_at)}
          </time>
        ) : null}
      </article>
    );
  }

  return (
    <article className="flex max-w-[52rem] items-start gap-3.5 sm:gap-5">
      <AevonMonogram className="mt-0.5" />
      <div className="min-w-0 flex-1">
        <div className="mb-2 flex items-center gap-2">
          <span className="text-sm font-medium text-[#eee8e1]">Aevon</span>
          {showTimestamp ? (
            <time className="text-[0.68rem] text-[#746d67]" dateTime={message.created_at}>
              {formatMessageTime(message.created_at)}
            </time>
          ) : null}
        </div>
        <MarkdownContent content={message.content} wrapCode={wrapCode} />
        {message.metadata?.tools?.map((event, index) => (
          <ToolEventCard event={event} key={`tool-${message.id}-${index}`} kind="tool" />
        ))}
        {message.metadata?.memory?.map((event, index) => (
          <ToolEventCard event={event} key={`memory-${message.id}-${index}`} kind="memory" />
        ))}
      </div>
    </article>
  );
}
