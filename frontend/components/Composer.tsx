"use client";

import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { ArrowUp, Paperclip } from "lucide-react";

interface ComposerProps {
  disabled?: boolean;
  enterToSend: boolean;
  onSend: (message: string) => Promise<void> | void;
}

export function Composer({ disabled = false, enterToSend, onSend }: ComposerProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "0px";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 176)}px`;
  }, [value]);

  async function submit(event?: FormEvent) {
    event?.preventDefault();
    const message = value.trim();
    if (!message || disabled) return;
    setValue("");
    await onSend(message);
    textareaRef.current?.focus();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    const requestedSend = enterToSend
      ? event.key === "Enter" && !event.shiftKey
      : event.key === "Enter" && (event.ctrlKey || event.metaKey);
    if (!requestedSend || event.nativeEvent.isComposing) return;
    event.preventDefault();
    void submit();
  }

  return (
    <form className="w-full" onSubmit={submit}>
      <div className="flex min-h-[4.5rem] items-end gap-2 rounded-2xl border border-[#805a3d]/65 bg-[#111212] p-2.5 pl-3 shadow-[0_18px_55px_rgba(0,0,0,0.24)] transition focus-within:border-[#b4784b]/80 focus-within:ring-1 focus-within:ring-[#b4784b]/25">
        <button
          aria-label="Attach a file (coming soon)"
          className="mb-0.5 flex h-10 w-10 shrink-0 cursor-not-allowed items-center justify-center rounded-full text-[#8f8983] opacity-70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]"
          disabled
          title="Attachments are coming later"
          type="button"
        >
          <Paperclip aria-hidden="true" className="h-5 w-5" />
        </button>
        <label className="sr-only" htmlFor="aevon-composer">
          Write to Aevon
        </label>
        <textarea
          aria-label="Write to Aevon"
          className="max-h-44 min-h-10 flex-1 resize-none bg-transparent py-2.5 text-[0.98rem] leading-6 text-[#eee8e1] outline-none placeholder:text-[#77736f] disabled:cursor-not-allowed disabled:opacity-60"
          disabled={disabled}
          id="aevon-composer"
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Write to Aevon..."
          ref={textareaRef}
          rows={1}
          value={value}
        />
        <button
          aria-label={disabled ? "Waiting for Aevon" : "Send message"}
          className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-[#a87349] text-[#0c0b0a] transition hover:bg-[#bd8558] disabled:cursor-not-allowed disabled:bg-[#4c3b2d] disabled:text-[#867568] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#e4b487] focus-visible:ring-offset-2 focus-visible:ring-offset-[#111212]"
          disabled={disabled || !value.trim()}
          type="submit"
        >
          <ArrowUp aria-hidden="true" className="h-5 w-5" strokeWidth={2} />
        </button>
      </div>
    </form>
  );
}
