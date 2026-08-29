"use client";

import { X } from "lucide-react";

import type { MessageMetadata } from "@/lib/types";
import { ToolEventCard } from "@/components/ToolEventCard";

interface DeveloperDrawerProps {
  metadata: MessageMetadata | null;
  open: boolean;
  onClose: () => void;
}

function yesNo(value: unknown): string {
  if (value === true) return "Yes";
  if (value === false) return "No";
  return "—";
}

function DetailRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-[#4c392b]/40 py-3">
      <dt className="text-xs uppercase tracking-[0.12em] text-[#817a74]">{label}</dt>
      <dd className="break-all text-right text-sm text-[#ded7d0]">{value}</dd>
    </div>
  );
}

export function DeveloperDrawer({ metadata, open, onClose }: DeveloperDrawerProps) {
  const validator = metadata?.validator ?? null;
  const retryAttempted = validator?.retry_attempted;
  const retryPassed = validator?.retry_passed;
  const tools = metadata?.tools ?? [];
  const memory = metadata?.memory ?? [];

  if (!open) return null;

  return (
    <>
      <button
        aria-label="Close developer details"
        className="fixed inset-0 z-40 bg-black/55"
        onClick={onClose}
        type="button"
      />
      <aside
        aria-label="Developer details"
        className="fixed inset-y-0 right-0 z-50 w-full max-w-[390px] border-l border-[#775138]/60 bg-[#0e0f0f] shadow-2xl shadow-black/60"
      >
        <div className="flex h-full flex-col">
          <div className="flex items-center justify-between border-b border-[#60452f]/55 px-6 py-5">
            <div>
              <h2 className="font-serif text-2xl text-[#eee8e1]">Developer details</h2>
              <p className="mt-1 text-xs text-[#817a74]">Latest Aevon response metadata</p>
            </div>
            <button
              aria-label="Close developer details"
              className="rounded-lg p-2 text-[#918a83] hover:bg-white/5 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]"
              onClick={onClose}
              type="button"
            >
              <X aria-hidden="true" className="h-5 w-5" />
            </button>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto px-6 py-5">
            <dl>
              <DetailRow label="Model" value={metadata?.model ?? "—"} />
              <DetailRow label="Latency" value={metadata?.latency_ms == null ? "—" : `${metadata.latency_ms} ms`} />
              <DetailRow label="Tools" value={tools.length} />
              <DetailRow label="Memory" value={memory.length} />
            </dl>
            <section className="mt-7">
              <h3 className="text-xs font-semibold uppercase tracking-[0.16em] text-[#bc8254]">Validator</h3>
              <dl className="mt-2">
                <DetailRow label="Retry attempted" value={yesNo(retryAttempted)} />
                <DetailRow label="Retry passed" value={yesNo(retryPassed)} />
              </dl>
            </section>
            {tools.length ? (
              <section className="mt-7">
                <h3 className="text-xs font-semibold uppercase tracking-[0.16em] text-[#bc8254]">Tool activity</h3>
                {tools.map((event, index) => (
                  <ToolEventCard event={event} key={`drawer-tool-${index}`} kind="tool" />
                ))}
              </section>
            ) : null}
            {memory.length ? (
              <section className="mt-7">
                <h3 className="text-xs font-semibold uppercase tracking-[0.16em] text-[#bc8254]">Memory activity</h3>
                {memory.map((event, index) => (
                  <ToolEventCard event={event} key={`drawer-memory-${index}`} kind="memory" />
                ))}
              </section>
            ) : null}
            {!metadata ? (
              <p className="mt-7 rounded-xl border border-[#5b4433]/45 bg-[#15130f] p-4 text-sm leading-6 text-[#8f8881]">
                Metadata will appear after Aevon responds in this conversation.
              </p>
            ) : null}
          </div>
        </div>
      </aside>
    </>
  );
}
