"use client";

import { useMemo, useState } from "react";
import { Brain, KeyRound, PanelLeftClose, Plus, Search, Settings, SlidersHorizontal, X } from "lucide-react";

import { CONVERSATION_GROUPS, groupConversations } from "@/lib/dates";
import type { AppView, Conversation } from "@/lib/types";
import { cn } from "@/lib/utils";

interface SidebarProps {
  conversations: Conversation[];
  selectedId: string | null;
  currentView: AppView;
  open: boolean;
  creating: boolean;
  interactionDisabled: boolean;
  onClose: () => void;
  onNewConversation: () => void;
  onSelectConversation: (id: string) => void;
  onNavigate: (view: AppView) => void;
}

const navItems = [
  { view: "memory" as const, label: "Memory", Icon: Brain },
  { view: "settings" as const, label: "Settings", Icon: Settings },
  { view: "preferences" as const, label: "Preferences", Icon: SlidersHorizontal },
  { view: "security" as const, label: "Security & Keys", Icon: KeyRound },
];

export function Sidebar({
  conversations,
  selectedId,
  currentView,
  open,
  creating,
  interactionDisabled,
  onClose,
  onNewConversation,
  onSelectConversation,
  onNavigate,
}: SidebarProps) {
  const [search, setSearch] = useState("");
  const filtered = useMemo(() => {
    const query = search.trim().toLocaleLowerCase();
    return query
      ? conversations.filter((conversation) => conversation.title.toLocaleLowerCase().includes(query))
      : conversations;
  }, [conversations, search]);
  const groups = useMemo(() => groupConversations(filtered), [filtered]);

  return (
    <>
      {open ? (
        <button
          aria-label="Close conversation sidebar"
          className="fixed inset-0 z-30 bg-black/60 lg:hidden"
          onClick={onClose}
          type="button"
        />
      ) : null}
      <aside
        aria-hidden={!open}
        aria-label="Conversation navigation"
        className={cn(
          "fixed inset-y-0 left-0 z-40 flex w-[270px] flex-col border-r border-[#7b5235]/55 bg-[#090a09] transition-transform duration-200 lg:static lg:z-auto lg:shrink-0",
          open ? "translate-x-0" : "-translate-x-full lg:hidden",
        )}
        inert={!open}
      >
        <div className="flex items-start justify-between px-6 pb-5 pt-6">
          <div>
            <div className="font-serif text-[2.35rem] leading-none tracking-[-0.035em] text-[#f0ebe6]">AmitAI</div>
            <div className="mt-1.5 text-[0.68rem] uppercase tracking-[0.32em] text-[#bd8254]">private</div>
          </div>
          <button
            aria-label="Collapse sidebar"
            className="mt-1 rounded-md p-2 text-[#8d857e] hover:bg-white/5 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]"
            onClick={onClose}
            type="button"
          >
            <PanelLeftClose aria-hidden="true" className="hidden h-4 w-4 lg:block" />
            <X aria-hidden="true" className="h-5 w-5 lg:hidden" />
          </button>
        </div>

        <div className="mx-5 border-t border-[#7b5235]/55 pt-3">
          <button
            className="flex w-full items-center gap-2 rounded-lg px-1 py-2.5 text-left text-sm text-[#ded8d2] hover:text-white disabled:cursor-wait disabled:opacity-60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]"
            disabled={creating || interactionDisabled}
            onClick={onNewConversation}
            type="button"
          >
            <Plus aria-hidden="true" className="h-4 w-4 text-[#c48a5c]" />
            {creating ? "Creating…" : "New conversation"}
          </button>
        </div>

        <div className="px-5 py-3">
          <label className="relative block">
            <span className="sr-only">Search conversations</span>
            <Search aria-hidden="true" className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[#bd8254]" />
            <input
              className="h-11 w-full rounded-xl border border-[#805a3d]/60 bg-[#111110] pl-10 pr-3 text-sm text-[#eee8e1] outline-none placeholder:text-[#77716b] focus:border-[#b4784b] focus:ring-1 focus:ring-[#b4784b]/25"
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search"
              type="search"
              value={search}
            />
          </label>
        </div>

        <nav className="min-h-0 flex-1 overflow-y-auto px-3 pb-4" aria-label="Conversations">
          {CONVERSATION_GROUPS.map((group) =>
            groups[group].length ? (
              <section className="mt-4" key={group}>
                <h2 className="px-3 text-[0.67rem] font-semibold uppercase tracking-[0.2em] text-[#bd8254]">
                  {group}
                </h2>
                <div className="mt-1.5 space-y-0.5">
                  {groups[group].map((conversation) => {
                    const selected = currentView === "chat" && selectedId === conversation.id;
                    return (
                      <button
                        aria-current={selected ? "page" : undefined}
                        className={cn(
                          "relative w-full truncate rounded-r-lg px-4 py-2 text-left text-sm transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[#bd8254]",
                          selected
                            ? "bg-[#2a211a] text-[#f1eae3] before:absolute before:inset-y-0 before:left-0 before:w-0.5 before:rounded-full before:bg-[#c88857]"
                            : "text-[#b9b2ab] hover:bg-white/[0.035] hover:text-[#e4ded8]",
                        )}
                        key={conversation.id}
                        disabled={interactionDisabled}
                        onClick={() => onSelectConversation(conversation.id)}
                        title={conversation.title}
                        type="button"
                      >
                        {conversation.title}
                      </button>
                    );
                  })}
                </div>
              </section>
            ) : null,
          )}
          {filtered.length === 0 ? (
            <p className="px-3 py-8 text-center text-xs leading-5 text-[#77716b]">
              {search ? "No matching conversations." : "No conversations yet."}
            </p>
          ) : null}
        </nav>

        <nav className="border-t border-[#5f4431]/40 px-4 py-4" aria-label="Application settings">
          {navItems.map(({ view, label, Icon }) => (
            <button
              className={cn(
                "flex w-full items-center gap-3 rounded-lg px-2 py-2.5 text-sm transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]",
                currentView === view ? "bg-[#251d17] text-[#f0e8e1]" : "text-[#bbb4ad] hover:bg-white/[0.035] hover:text-white",
              )}
              key={view}
              onClick={() => onNavigate(view)}
              type="button"
            >
              <Icon aria-hidden="true" className="h-[1.1rem] w-[1.1rem] text-[#c88959]" />
              {label}
            </button>
          ))}
        </nav>
      </aside>
    </>
  );
}
