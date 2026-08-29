"use client";

import { useState, type FormEvent } from "react";
import { X } from "lucide-react";

interface RenameDialogProps {
  initialTitle: string;
  saving: boolean;
  error: string | null;
  onCancel: () => void;
  onRename: (title: string) => void;
}

export function RenameDialog({
  initialTitle,
  saving,
  error,
  onCancel,
  onRename,
}: RenameDialogProps) {
  const [title, setTitle] = useState(initialTitle);

  function submit(event: FormEvent) {
    event.preventDefault();
    if (title.trim()) onRename(title.trim());
  }

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/65 p-4" role="presentation">
      <div aria-labelledby="rename-title" aria-modal="true" className="w-full max-w-md rounded-2xl border border-[#755138]/65 bg-[#121211] p-5 shadow-2xl" role="dialog">
        <div className="flex items-center justify-between">
          <h2 className="font-serif text-2xl text-[#eee8e1]" id="rename-title">
            Rename conversation
          </h2>
          <button
            aria-label="Cancel rename"
            className="rounded-lg p-2 text-[#918a83] hover:bg-white/5 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]"
            disabled={saving}
            onClick={onCancel}
            type="button"
          >
            <X aria-hidden="true" className="h-5 w-5" />
          </button>
        </div>
        <form className="mt-5" onSubmit={submit}>
          <label className="text-xs uppercase tracking-[0.15em] text-[#9a9189]" htmlFor="conversation-title">
            Title
          </label>
          <input
            autoFocus
            className="mt-2 h-11 w-full rounded-xl border border-[#765038]/65 bg-[#0d0e0e] px-3 text-sm text-[#eee8e1] outline-none focus:border-[#bd8254] focus:ring-1 focus:ring-[#bd8254]/30"
            disabled={saving}
            id="conversation-title"
            maxLength={200}
            onChange={(event) => setTitle(event.target.value)}
            value={title}
          />
          {error ? <p className="mt-2 text-sm text-[#d79784]">{error}</p> : null}
          <div className="mt-5 flex justify-end gap-2">
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
              disabled={saving || !title.trim()}
              type="submit"
            >
              {saving ? "Saving…" : "Save"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
