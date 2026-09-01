"use client";

import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import { Brain, Pencil, Plus, RotateCcw, Search, Trash2, X } from "lucide-react";

import {
  ApiError,
  createMemory,
  deleteMemory,
  listMemories,
  updateMemory,
} from "@/lib/api";
import type { MemoryCategory, MemoryRecord, MemorySensitivity, MemoryStatus, MemoryUpdateInput } from "@/lib/types";
import { MEMORY_CATEGORIES } from "@/lib/types";

type CategoryFilter = MemoryCategory | "all";

function formatUpdatedAt(timestamp: string): string {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "Unknown";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

export function memoryErrorMessage(error: unknown, action: "load" | "save" | "forget"): string {
  if (!(error instanceof ApiError)) return "Something went wrong. Try again.";
  if (!error.backendReached) return "Unable to connect to Aevon.";
  if (error.status === 409) return "This memory changed elsewhere. Refresh and try again.";
  if (error.status === 404) return "This memory no longer exists. Refresh the list.";
  if (error.status === 422) return error.message || "Check the memory fields and try again.";
  if (action === "load") return "Unable to load memories.";
  if (action === "forget") return "Unable to forget this memory.";
  return "Unable to save this memory.";
}

interface MemoryEditorDialogProps {
  memory: MemoryRecord | null;
  error: string | null;
  saving: boolean;
  onCancel: () => void;
  onSave: (input: { category: MemoryCategory; key: string; value: string; sensitivity: MemorySensitivity }) => void;
}

function MemoryEditorDialog({
  memory,
  error,
  saving,
  onCancel,
  onSave,
}: MemoryEditorDialogProps) {
  const [category, setCategory] = useState<MemoryCategory>(
    memory?.category ?? "preference",
  );
  const [key, setKey] = useState(memory?.key ?? "");
  const [value, setValue] = useState(memory?.value ?? "");
  const [sensitivity, setSensitivity] = useState<MemorySensitivity>(memory?.sensitivity ?? "local_only");
  const editing = memory !== null;
  const unchanged = editing && value.trim() === memory.value && sensitivity === memory.sensitivity;

  function submit(event: FormEvent) {
    event.preventDefault();
    const normalizedKey = key.trim();
    const normalizedValue = value.trim();
    if (!normalizedKey || !normalizedValue || saving || unchanged) return;
    onSave({ category, key: normalizedKey, value: normalizedValue, sensitivity });
  }

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/65 p-4" role="presentation">
      <div
        aria-labelledby="memory-editor-title"
        aria-modal="true"
        className="w-full max-w-lg rounded-2xl border border-[#755138]/65 bg-[#121211] p-5 shadow-2xl"
        role="dialog"
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="font-serif text-2xl text-[#eee8e1]" id="memory-editor-title">
              {editing ? "Edit memory" : "Create memory"}
            </h2>
            <p className="mt-1 text-xs leading-5 text-[#8c857f]">
              Structured memories are explicit and never captured automatically.
            </p>
          </div>
          <button
            aria-label="Close memory editor"
            className="rounded-lg p-2 text-[#918a83] hover:bg-white/5 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]"
            disabled={saving}
            onClick={onCancel}
            type="button"
          >
            <X aria-hidden="true" className="h-5 w-5" />
          </button>
        </div>

        <form className="mt-5 space-y-4" onSubmit={submit}>
          <label className="block text-xs uppercase tracking-[0.15em] text-[#9a9189]">
            Category
            <select
              className="mt-2 h-11 w-full rounded-xl border border-[#765038]/65 bg-[#0d0e0e] px-3 text-sm text-[#eee8e1] outline-none focus:border-[#bd8254] focus:ring-1 focus:ring-[#bd8254]/30"
              disabled={editing || saving}
              onChange={(event) => setCategory(event.target.value as MemoryCategory)}
              value={category}
            >
              {MEMORY_CATEGORIES.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>

          <label className="block text-xs uppercase tracking-[0.15em] text-[#9a9189]">
            Key
            <input
              autoFocus={!editing}
              className="mt-2 h-11 w-full rounded-xl border border-[#765038]/65 bg-[#0d0e0e] px-3 font-mono text-sm text-[#eee8e1] outline-none placeholder:text-[#6f6963] focus:border-[#bd8254] focus:ring-1 focus:ring-[#bd8254]/30"
              disabled={editing || saving}
              maxLength={128}
              onChange={(event) => setKey(event.target.value)}
              pattern="[A-Za-z0-9]+([._-][A-Za-z0-9]+)*"
              placeholder="ui.theme"
              required
              value={key}
            />
          </label>

          <label className="block text-xs uppercase tracking-[0.15em] text-[#9a9189]">
            Value
            <textarea
              autoFocus={editing}
              className="mt-2 min-h-28 w-full resize-y rounded-xl border border-[#765038]/65 bg-[#0d0e0e] px-3 py-2.5 text-sm leading-6 text-[#eee8e1] outline-none placeholder:text-[#6f6963] focus:border-[#bd8254] focus:ring-1 focus:ring-[#bd8254]/30"
              disabled={saving}
              maxLength={1000}
              onChange={(event) => setValue(event.target.value)}
              placeholder="What should Aevon remember?"
              required
              value={value}
            />
          </label>

          <label className="block text-xs uppercase tracking-[0.15em] text-[#9a9189]">
            Inference access
            <select
              className="mt-2 h-11 w-full rounded-xl border border-[#765038]/65 bg-[#0d0e0e] px-3 text-sm text-[#eee8e1] outline-none focus:border-[#bd8254]"
              disabled={saving}
              onChange={(event) => setSensitivity(event.target.value as MemorySensitivity)}
              value={sensitivity}
            >
              <option value="local_only">Local only</option>
              <option value="remote_allowed">Remote allowed</option>
            </select>
          </label>
          <p className="text-xs leading-5 text-[#8c857f]">
            Remote allowed lets a remote inference provider read this memory when relevant.
          </p>

          {error ? (
            <p aria-live="polite" className="rounded-lg border border-[#754735]/60 bg-[#241813] px-3 py-2 text-sm text-[#dda08d]">
              {error}
            </p>
          ) : null}

          <div className="flex justify-end gap-2 pt-1">
            <button
              className="rounded-lg px-4 py-2 text-sm text-[#b8b0a9] hover:bg-white/5 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]"
              disabled={saving}
              onClick={onCancel}
              type="button"
            >
              Cancel
            </button>
            <button
              className="rounded-lg bg-[#a87349] px-4 py-2 text-sm font-medium text-[#0c0b0a] hover:bg-[#bd8558] disabled:cursor-wait disabled:opacity-60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#e4b487]"
              disabled={saving || !key.trim() || !value.trim() || unchanged}
              type="submit"
            >
              {saving ? "Saving…" : editing ? "Save changes" : "Create memory"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

interface ForgetDialogProps {
  memory: MemoryRecord;
  error: string | null;
  forgetting: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}

function ForgetDialog({ memory, error, forgetting, onCancel, onConfirm }: ForgetDialogProps) {
  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/65 p-4" role="presentation">
      <div
        aria-labelledby="forget-memory-title"
        aria-modal="true"
        className="w-full max-w-md rounded-2xl border border-[#755138]/65 bg-[#121211] p-5 shadow-2xl"
        role="alertdialog"
      >
        <h2 className="font-serif text-2xl text-[#eee8e1]" id="forget-memory-title">
          Forget this memory?
        </h2>
        <p className="mt-3 text-sm leading-6 text-[#aaa19a]">
          <span className="font-mono text-[#e2d8cf]">{memory.key}</span> will be tombstoned and its stored value redacted. This cannot be undone.
        </p>
        {error ? (
          <p aria-live="polite" className="mt-3 rounded-lg border border-[#754735]/60 bg-[#241813] px-3 py-2 text-sm text-[#dda08d]">
            {error}
          </p>
        ) : null}
        <div className="mt-5 flex justify-end gap-2">
          <button
            className="rounded-lg px-4 py-2 text-sm text-[#b8b0a9] hover:bg-white/5 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]"
            disabled={forgetting}
            onClick={onCancel}
            type="button"
          >
            Cancel
          </button>
          <button
            className="rounded-lg bg-[#8d4f42] px-4 py-2 text-sm font-medium text-[#f6e9e4] hover:bg-[#a76050] disabled:cursor-wait disabled:opacity-60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#dc9a87]"
            disabled={forgetting}
            onClick={onConfirm}
            type="button"
          >
            {forgetting ? "Forgetting…" : "Forget memory"}
          </button>
        </div>
      </div>
    </div>
  );
}

export function MemoryView() {
  const [memories, setMemories] = useState<MemoryRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [searchDraft, setSearchDraft] = useState("");
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState<CategoryFilter>("all");
  const [status, setStatus] = useState<MemoryStatus>("active");
  const [editing, setEditing] = useState<MemoryRecord | "create" | null>(null);
  const [forgetting, setForgetting] = useState<MemoryRecord | null>(null);
  const [mutationPending, setMutationPending] = useState(false);
  const [mutationError, setMutationError] = useState<string | null>(null);
  const requestRef = useRef(0);

  const load = useCallback(async () => {
    const requestId = ++requestRef.current;
    setLoading(true);
    setLoadError(null);
    try {
      const listed = await listMemories(
        query
          ? { query }
          : {
              ...(category === "all" ? {} : { category }),
              status,
            },
      );
      if (requestId !== requestRef.current) return;
      setMemories(listed);
    } catch (error) {
      if (requestId !== requestRef.current) return;
      setLoadError(memoryErrorMessage(error, "load"));
    } finally {
      if (requestId === requestRef.current) setLoading(false);
    }
  }, [category, query, status]);

  useEffect(() => {
    const timer = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(timer);
  }, [load]);

  function submitSearch(event: FormEvent) {
    event.preventDefault();
    const nextQuery = searchDraft.trim();
    if (nextQuery) {
      setCategory("all");
      setStatus("active");
    }
    if (nextQuery === query) {
      void load();
    } else {
      setQuery(nextQuery);
    }
  }

  function changeCategory(nextCategory: CategoryFilter) {
    setSearchDraft("");
    setQuery("");
    setCategory(nextCategory);
  }

  function changeStatus(nextStatus: MemoryStatus) {
    setSearchDraft("");
    setQuery("");
    setStatus(nextStatus);
  }

  async function saveMemory(input: {
    category: MemoryCategory;
    key: string;
    value: string;
    sensitivity: MemorySensitivity;
  }) {
    if (mutationPending) return;
    setMutationPending(true);
    setMutationError(null);
    try {
      if (editing === "create") {
        await createMemory(input);
      } else if (editing) {
        const changes: MemoryUpdateInput = {};
        if (input.value !== editing.value) changes.value = input.value;
        if (input.sensitivity !== editing.sensitivity) changes.sensitivity = input.sensitivity;
        if (Object.keys(changes).length === 0) return;
        await updateMemory(editing.id, changes);
      } else {
        return;
      }
      setEditing(null);
      await load();
    } catch (error) {
      setMutationError(memoryErrorMessage(error, "save"));
    } finally {
      setMutationPending(false);
    }
  }

  async function confirmForget() {
    if (!forgetting || mutationPending) return;
    setMutationPending(true);
    setMutationError(null);
    try {
      await deleteMemory(forgetting.id);
      setForgetting(null);
      await load();
    } catch (error) {
      setMutationError(memoryErrorMessage(error, "forget"));
    } finally {
      setMutationPending(false);
    }
  }

  const emptyMessage = query
    ? "No active memories match this search."
    : status === "deleted"
      ? "No forgotten memories."
      : category === "all"
        ? "No memories yet. Create one explicitly or use a supported memory command in chat."
        : `No ${category} memories.`;

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto w-full max-w-5xl px-6 py-10 sm:px-10">
        <div className="flex flex-col gap-5 border-b border-[#4d392b]/45 pb-7 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <div className="flex items-center gap-3">
              <Brain aria-hidden="true" className="h-6 w-6 text-[#c68a5a]" />
              <h2 className="font-serif text-4xl text-[#eee8e1]">Memory</h2>
            </div>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-[#8c857f]">
              View and manage explicit structured context. Aevon never captures ordinary conversation automatically.
            </p>
          </div>
          <button
            className="inline-flex h-10 items-center justify-center gap-2 self-start rounded-xl bg-[#a87349] px-4 text-sm font-medium text-[#0c0b0a] hover:bg-[#bd8558] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#e4b487] sm:self-auto"
            onClick={() => {
              setMutationError(null);
              setEditing("create");
            }}
            type="button"
          >
            <Plus aria-hidden="true" className="h-4 w-4" />
            New memory
          </button>
        </div>

        <div className="mt-6 grid gap-3 lg:grid-cols-[minmax(0,1fr)_12rem_auto]">
          <form className="flex gap-2" onSubmit={submitSearch} role="search">
            <label className="relative min-w-0 flex-1">
              <span className="sr-only">Search memories</span>
              <Search aria-hidden="true" className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[#bd8254]" />
              <input
                className="h-11 w-full rounded-xl border border-[#805a3d]/60 bg-[#111110] pl-10 pr-3 text-sm text-[#eee8e1] outline-none placeholder:text-[#77716b] focus:border-[#b4784b] focus:ring-1 focus:ring-[#b4784b]/25"
                onChange={(event) => setSearchDraft(event.target.value)}
                placeholder="Search memories"
                type="search"
                value={searchDraft}
              />
            </label>
            <button
              className="rounded-xl border border-[#765038]/65 bg-[#171410] px-4 text-sm text-[#d7cfc8] hover:bg-[#251d17] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]"
              type="submit"
            >
              Search
            </button>
          </form>

          <label>
            <span className="sr-only">Filter memories by category</span>
            <select
              aria-label="Memory category"
              className="h-11 w-full rounded-xl border border-[#805a3d]/60 bg-[#111110] px-3 text-sm capitalize text-[#ded7d0] outline-none focus:border-[#b4784b] focus:ring-1 focus:ring-[#b4784b]/25"
              onChange={(event) => changeCategory(event.target.value as CategoryFilter)}
              value={category}
            >
              <option value="all">All categories</option>
              {MEMORY_CATEGORIES.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
          </label>

          <div className="inline-flex h-11 rounded-xl border border-[#805a3d]/60 bg-[#111110] p-1" role="group" aria-label="Memory status">
            {(["active", "deleted"] as const).map((item) => (
              <button
                aria-pressed={status === item}
                className={`rounded-lg px-3 text-sm capitalize transition ${status === item ? "bg-[#34271d] text-[#f0e8e1]" : "text-[#908880] hover:text-[#d9d1ca]"}`}
                key={item}
                onClick={() => changeStatus(item)}
                type="button"
              >
                {item === "deleted" ? "Forgotten" : "Active"}
              </button>
            ))}
          </div>
        </div>

        {query ? (
          <div className="mt-3 flex items-center gap-2 text-xs text-[#948b83]">
            <span>Searching active memories for “{query}”</span>
            <button
              className="text-[#cf9667] hover:text-[#e8b082]"
              onClick={() => {
                setSearchDraft("");
                setQuery("");
              }}
              type="button"
            >
              Clear
            </button>
          </div>
        ) : null}

        {loadError ? (
          <div className="mt-8 rounded-2xl border border-[#754735]/65 bg-[#211713] p-5">
            <p className="text-sm text-[#dda08d]">{loadError}</p>
            <button
              className="mt-3 inline-flex items-center gap-2 text-sm text-[#dca778] hover:text-[#efbf93] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]"
              onClick={() => void load()}
              type="button"
            >
              <RotateCcw aria-hidden="true" className="h-4 w-4" />
              Retry
            </button>
          </div>
        ) : null}

        {!loadError && loading && memories.length === 0 ? (
          <div aria-label="Loading memories" className="mt-8 space-y-3">
            {[0, 1, 2].map((item) => (
              <div className="h-32 animate-pulse rounded-2xl border border-[#4d392b]/45 bg-[#111211]" key={item} />
            ))}
          </div>
        ) : null}

        {!loadError && !loading && memories.length === 0 ? (
          <div className="mt-8 rounded-2xl border border-dashed border-[#614630]/65 bg-[#11110f] px-6 py-12 text-center">
            <Brain aria-hidden="true" className="mx-auto h-7 w-7 text-[#8d6345]" />
            <p className="mx-auto mt-3 max-w-lg text-sm leading-6 text-[#8f8881]">{emptyMessage}</p>
          </div>
        ) : null}

        {!loadError && memories.length > 0 ? (
          <div className="mt-8 grid gap-4 xl:grid-cols-2" aria-busy={loading} aria-live="polite">
            {memories.map((memory) => (
              <article className="rounded-2xl border border-[#5d4533]/55 bg-[#111211] p-5" key={memory.id}>
                <div className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="rounded-full border border-[#765038]/55 bg-[#211a15] px-2.5 py-1 text-[0.65rem] font-semibold uppercase tracking-[0.13em] text-[#cb9365]">
                        {memory.category}
                      </span>
                      <span className={`text-[0.68rem] uppercase tracking-[0.13em] ${memory.status === "deleted" ? "text-[#9a7469]" : "text-[#7f8d79]"}`}>
                        {memory.status === "deleted" ? "Forgotten" : "Active"}
                      </span>
                      {memory.status === "active" ? (
                        <span className="text-[0.68rem] text-[#b69c86]">
                          {memory.sensitivity === "remote_allowed" ? "Remote allowed" : "Local only"}
                        </span>
                      ) : null}
                    </div>
                    <h3 className="mt-3 break-all font-mono text-sm text-[#e9e2db]">{memory.key}</h3>
                  </div>
                  {memory.status === "active" ? (
                    <div className="flex shrink-0 gap-1">
                      <button
                        aria-label={`Edit ${memory.key}`}
                        className="rounded-lg p-2 text-[#a89d94] hover:bg-[#251d17] hover:text-[#e3b28a] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]"
                        onClick={() => {
                          setMutationError(null);
                          setEditing(memory);
                        }}
                        type="button"
                      >
                        <Pencil aria-hidden="true" className="h-4 w-4" />
                      </button>
                      <button
                        aria-label={`Forget ${memory.key}`}
                        className="rounded-lg p-2 text-[#a89d94] hover:bg-[#2b1b17] hover:text-[#d99482] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]"
                        onClick={() => {
                          setMutationError(null);
                          setForgetting(memory);
                        }}
                        type="button"
                      >
                        <Trash2 aria-hidden="true" className="h-4 w-4" />
                      </button>
                    </div>
                  ) : null}
                </div>
                <p className={`mt-4 whitespace-pre-wrap break-words text-sm leading-6 ${memory.status === "deleted" ? "italic text-[#756f69]" : "text-[#c9c1ba]"}`}>
                  {memory.status === "active" ? (memory.value ?? "No value returned.") : "Value redacted"}
                </p>
                <p className="mt-4 border-t border-[#49362a]/45 pt-3 text-xs text-[#77716b]">
                  Updated {formatUpdatedAt(memory.updated_at)}
                </p>
              </article>
            ))}
          </div>
        ) : null}
      </div>

      {editing ? (
        <MemoryEditorDialog
          error={mutationError}
          key={editing === "create" ? "create-memory" : editing.id}
          memory={editing === "create" ? null : editing}
          onCancel={() => {
            if (!mutationPending) setEditing(null);
          }}
          onSave={(input) => void saveMemory(input)}
          saving={mutationPending}
        />
      ) : null}

      {forgetting ? (
        <ForgetDialog
          error={mutationError}
          forgetting={mutationPending}
          memory={forgetting}
          onCancel={() => {
            if (!mutationPending) setForgetting(null);
          }}
          onConfirm={() => void confirmForget()}
        />
      ) : null}
    </div>
  );
}
