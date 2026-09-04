"use client";

import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { ArrowUp, Paperclip, Square } from "lucide-react";
import { ApiError, deleteAsset, uploadImage } from "@/lib/api";
import type { UploadedAsset, VisionCapability } from "@/lib/types";
import { AssetPreview } from "@/components/AssetPreview";
import { RemoteVisionConsent } from "@/components/RemoteVisionConsent";

interface ComposerProps {
  disabled?: boolean;
  onStop?: () => void;
  enterToSend: boolean;
  onSend: (message: string, assets?: UploadedAsset[], allowRemoteVision?: boolean) => Promise<void> | void;
  vision?: VisionCapability | null;
  onReloadCapabilities?: () => void;
}

export function Composer({ disabled = false, onStop, enterToSend, onSend, vision, onReloadCapabilities }: ComposerProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [assets, setAssets] = useState<UploadedAsset[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const busy = useRef(false);
  const staged = useRef<UploadedAsset[]>([]);
  const mounted = useRef(true);
  const wasDisabled = useRef(disabled);
  const [consentAssetId, setConsentAssetId] = useState<string | null>(null);
  const remote = vision?.scope === "remote";
  const consent = assets.length === 1 && consentAssetId === assets[0].id;
  const visionAvailable = vision?.enabled && (vision.scope === "local" || vision.scope === "remote");
  const imageBlocked = assets.length > 0 && (!visionAvailable || assets.length > 1 || (remote && !consent));

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      // Best effort only; the server TTL also covers refresh/offline/abandoned tabs.
      for (const asset of staged.current) void deleteAsset(asset.id).catch(() => undefined);
      staged.current = [];
    };
  }, []);

  function setStaged(next: UploadedAsset[]) {
    setConsentAssetId(null);
    staged.current = next;
    setAssets(next);
  }

  async function selectImage(file: File | undefined) {
    if (!file || busy.current || disabled || staged.current.length >= 1) return;
    if (!["image/png", "image/jpeg", "image/webp"].includes(file.type) || !file.size || file.size > 20 * 1024 * 1024) {
      setUploadError("Choose a PNG, JPEG or WebP image up to 20 MiB.");
      return;
    }
    busy.current = true;
    setUploading(true);
    setUploadError(null);
    try {
      const asset = await uploadImage(file);
      if (!mounted.current) {
        await deleteAsset(asset.id);
        return;
      }
      setStaged([...staged.current, asset]);
    } catch (error) {
      if (mounted.current) setUploadError(error instanceof ApiError ? error.message : "Image upload failed.");
    } finally {
      busy.current = false;
      if (mounted.current) setUploading(false);
    }
  }

  async function removeImage(id: string) {
    if (busy.current || disabled) return;
    busy.current = true;
    setUploading(true);
    try {
      await deleteAsset(id);
      setStaged(staged.current.filter((asset) => asset.id !== id));
      setUploadError(null);
    } catch {
      setUploadError("Could not remove the image. Try again.");
    } finally {
      busy.current = false;
      setUploading(false);
    }
  }

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "0px";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 176)}px`;
  }, [value]);

  useEffect(() => {
    if (wasDisabled.current && !disabled) textareaRef.current?.focus();
    wasDisabled.current = disabled;
  }, [disabled]);

  async function submit(event?: FormEvent) {
    event?.preventDefault();
    const message = value.trim();
    if (!message || disabled || busy.current || imageBlocked) return;
    const attachments = staged.current;
    const allowRemoteVision = remote && consent;
    setStaged([]); // Ownership passes to the pending chat, including retry state.
    setValue("");
    await onSend(message, attachments, allowRemoteVision);
    if (mounted.current && !textareaRef.current?.disabled) textareaRef.current?.focus();
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
      <input accept="image/png,image/jpeg,image/webp" aria-label="Select image" className="sr-only" disabled={disabled || uploading || assets.length >= 1} onChange={(event) => {
        const file = event.target.files?.[0];
        event.target.value = "";
        void selectImage(file);
      }} ref={fileRef} type="file" />
      {assets.length ? <div className="mb-3 flex flex-wrap gap-2">
        {assets.map((asset) => <div key={asset.id}>
          <AssetPreview asset={asset} />
          <button aria-label={`Remove ${asset.original_filename}`} className="mt-1 text-xs text-[#dca778] disabled:opacity-50" disabled={disabled || uploading} onClick={() => void removeImage(asset.id)} type="button">Remove</button>
        </div>)}
        <p className="w-full text-xs text-[#948d86]">Stored locally with this chat when sent. One image per message.</p>
        {!visionAvailable ? <p className="text-xs text-[#948d86]" role="status">{vision ? "Vision is not enabled for the configured provider." : "Vision capabilities unavailable."} <button onClick={onReloadCapabilities} type="button">Retry capabilities</button></p> : null}
        {remote && assets.length === 1 ? <RemoteVisionConsent checked={consent} disabled={disabled || uploading} onChange={(checked) => setConsentAssetId(checked ? assets[0].id : null)} /> : null}
      </div> : null}
      {uploading ? <p className="mb-2 text-xs text-[#948d86]" role="status">Updating image attachment…</p> : null}
      {uploadError ? <p className="mb-2 text-sm text-[#e0b49b]" role="alert">{uploadError}</p> : null}
      <div className="flex min-h-[4.5rem] items-end gap-2 rounded-2xl border border-[#805a3d]/65 bg-[#111212] p-2.5 pl-3 shadow-[0_18px_55px_rgba(0,0,0,0.24)] transition focus-within:border-[#b4784b]/80 focus-within:ring-1 focus-within:ring-[#b4784b]/25">
        <button
          aria-label="Attach image"
          className="mb-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-[#bcb1a7] disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#bd8254]"
          disabled={disabled || uploading || assets.length >= 1}
          title="Upload PNG, JPEG or WebP locally. Remote vision requires separate consent."
          onClick={() => fileRef.current?.click()}
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
          aria-label={onStop ? "Stop generation" : disabled ? "Waiting for Aevon" : "Send message"}
          className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-[#a87349] text-[#0c0b0a] transition hover:bg-[#bd8558] disabled:cursor-not-allowed disabled:bg-[#4c3b2d] disabled:text-[#867568] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#e4b487] focus-visible:ring-offset-2 focus-visible:ring-offset-[#111212]"
          disabled={!onStop && (disabled || uploading || !value.trim() || imageBlocked)}
          onClick={onStop}
          type={onStop ? "button" : "submit"}
        >
          {onStop ? <Square aria-hidden="true" className="h-4 w-4" fill="currentColor" /> : <ArrowUp aria-hidden="true" className="h-5 w-5" strokeWidth={2} />}
        </button>
      </div>
    </form>
  );
}
