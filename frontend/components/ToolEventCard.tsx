import { Brain, ExternalLink, FileText, Wrench } from "lucide-react";

import { describeEvent } from "@/lib/utils";

interface ToolEventCardProps {
  kind: "source" | "tool" | "memory";
  event: unknown;
}

const labels = {
  source: "Source",
  tool: "Tool",
  memory: "Memory",
};

export function ToolEventCard({ kind, event }: ToolEventCardProps) {
  const { title, detail, url } = describeEvent(event);
  const Icon = kind === "memory" ? Brain : kind === "source" ? FileText : Wrench;

  return (
    <div className="mt-3 flex gap-3 rounded-xl border border-[#8e613e]/45 bg-[#17130f]/70 px-4 py-3 text-sm text-[#d7d0c9]">
      <Icon aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0 text-[#c48959]" />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className="text-[0.7rem] font-semibold uppercase tracking-[0.16em] text-[#b77d51]">
            {labels[kind]}
          </span>
          {url ? (
            <a
              className="inline-flex items-center gap-1 break-words text-[#eee8e1] underline decoration-[#8e613e] underline-offset-4 hover:text-white"
              href={url}
              rel="noreferrer"
              target="_blank"
            >
              {title}
              <ExternalLink aria-hidden="true" className="h-3 w-3" />
            </a>
          ) : (
            <span className="break-words text-[#eee8e1]">{title}</span>
          )}
        </div>
        {detail ? (
          <pre className="mt-1.5 whitespace-pre-wrap break-words font-sans text-xs leading-5 text-[#aaa29b]">
            {detail}
          </pre>
        ) : null}
      </div>
    </div>
  );
}
