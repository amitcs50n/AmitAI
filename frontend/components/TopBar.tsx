"use client";

import { useEffect, useRef, useState } from "react";
import { Braces, Download, Menu, MoreHorizontal, Pencil, Trash2 } from "lucide-react";

import type { ConnectionState } from "@/lib/types";
import { cn } from "@/lib/utils";

interface TopBarProps {
  connection: ConnectionState;
  hasConversation: boolean;
  conversationActionsDisabled: boolean;
  onDelete: () => void;
  onDeveloperDetails: () => void;
  onExport: () => void;
  onOpenSidebar: () => void;
  onRename: () => void;
}

const connectionLabels: Record<ConnectionState, string> = {
  connecting: "Connecting…",
  connected: "Connected",
  disconnected: "Disconnected",
};

export function TopBar({
  connection,
  hasConversation,
  conversationActionsDisabled,
  onDelete,
  onDeveloperDetails,
  onExport,
  onOpenSidebar,
  onRename,
}: TopBarProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function closeMenu(event: MouseEvent) {
      if (!menuRef.current?.contains(event.target as Node)) setMenuOpen(false);
    }
    document.addEventListener("mousedown", closeMenu);
    return () => document.removeEventListener("mousedown", closeMenu);
  }, []);

  function run(action: () => void) {
    setMenuOpen(false);
    action();
  }

  return (
    <header className="relative z-20 shrink-0 px-5 pt-4 sm:px-8 sm:pt-5">
      <div className="mx-auto flex h-[4.6rem] w-full max-w-[62rem] items-start justify-between border-b border-[#755137]/55">
        <div className="flex items-start gap-3">
          <button
            aria-label="Open conversation sidebar"
            className="mt-1 rounded-lg p-2 text-[#98918a] hover:bg-white/5 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]"
            onClick={onOpenSidebar}
            type="button"
          >
            <Menu aria-hidden="true" className="h-5 w-5" />
          </button>
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <h1 className="font-serif text-[2.3rem] leading-none tracking-[-0.035em] text-[#f0ebe6] sm:text-[2.65rem]">Aevon</h1>
            <div className="inline-flex items-center gap-2 text-xs text-[#aa9e94] sm:text-sm">
              <span
                aria-hidden="true"
                className={cn(
                  "h-2 w-2 rounded-full",
                  connection === "connected" && "bg-[#c58b5b]",
                  connection === "connecting" && "animate-pulse bg-[#8e6d53]",
                  connection === "disconnected" && "bg-[#805044]",
                )}
              />
              <span aria-live="polite">{connectionLabels[connection]}</span>
            </div>
          </div>
        </div>

        <div className="relative" ref={menuRef}>
          <button
            aria-expanded={menuOpen}
            aria-haspopup="menu"
            aria-label="Conversation menu"
            className="rounded-lg p-2.5 text-[#98918a] hover:bg-white/5 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]"
            onClick={() => setMenuOpen((value) => !value)}
            type="button"
          >
            <MoreHorizontal aria-hidden="true" className="h-5 w-5" />
          </button>
          {menuOpen ? (
            <div
              className="absolute right-0 top-11 z-50 w-56 rounded-xl border border-[#6f4e36]/70 bg-[#121211] p-1.5 text-sm shadow-2xl shadow-black/50"
              role="menu"
            >
              <button
                className="flex w-full items-center gap-2 rounded-lg px-3 py-2.5 text-left text-[#d7d0c9] hover:bg-[#251c16] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]"
                onClick={() => run(onDeveloperDetails)}
                role="menuitem"
                type="button"
              >
                <Braces aria-hidden="true" className="h-4 w-4 text-[#bd8254]" />
                Developer details
              </button>
              <button
                className="flex w-full items-center gap-2 rounded-lg px-3 py-2.5 text-left text-[#d7d0c9] hover:bg-[#251c16] disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]"
                disabled={!hasConversation || conversationActionsDisabled}
                onClick={() => run(onRename)}
                role="menuitem"
                type="button"
              >
                <Pencil aria-hidden="true" className="h-4 w-4 text-[#bd8254]" />
                Rename conversation
              </button>
              <button
                className="flex w-full items-center gap-2 rounded-lg px-3 py-2.5 text-left text-[#d7d0c9] hover:bg-[#251c16] disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]"
                disabled={!hasConversation || conversationActionsDisabled}
                onClick={() => run(onExport)}
                role="menuitem"
                type="button"
              >
                <Download aria-hidden="true" className="h-4 w-4 text-[#bd8254]" />
                Export conversation
              </button>
              <div className="my-1 border-t border-[#5e4432]/45" />
              <button
                className="flex w-full items-center gap-2 rounded-lg px-3 py-2.5 text-left text-[#d99b8a] hover:bg-[#301b17] disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]"
                disabled={!hasConversation || conversationActionsDisabled}
                onClick={() => run(onDelete)}
                role="menuitem"
                type="button"
              >
                <Trash2 aria-hidden="true" className="h-4 w-4" />
                Delete conversation
              </button>
            </div>
          ) : null}
        </div>
      </div>
    </header>
  );
}
