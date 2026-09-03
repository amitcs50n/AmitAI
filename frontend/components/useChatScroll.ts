"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export const CHAT_BOTTOM_THRESHOLD = 96;

/** Scroll only this conversation viewport, at most once per animation frame. */
export function useChatScroll({ loading, pendingId, sending, contentVersion }: {
  loading: boolean;
  pendingId: string | undefined;
  sending: boolean;
  contentVersion: string;
}) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const following = useRef(true);
  const cancelFrame = useRef<(() => void) | null>(null);
  const [showLatest, setShowLatest] = useState(false);

  const schedule = useCallback(() => {
    if (cancelFrame.current) return;
    const update = () => {
      cancelFrame.current = null;
      const viewport = viewportRef.current;
      if (!viewport) return;
      if (following.current) viewport.scrollTop = viewport.scrollHeight;
      setShowLatest(viewport.scrollHeight - viewport.clientHeight - viewport.scrollTop > CHAT_BOTTOM_THRESHOLD);
    };
    // The fallback also supports non-visual DOM test environments.
    if (window.requestAnimationFrame) {
      const frame = window.requestAnimationFrame(update);
      cancelFrame.current = () => window.cancelAnimationFrame(frame);
    } else {
      const timer = window.setTimeout(update, 16);
      cancelFrame.current = () => window.clearTimeout(timer);
    }
  }, []);

  const jumpToLatest = useCallback(() => {
    following.current = true;
    schedule();
  }, [schedule]);

  function onScroll() {
    const viewport = viewportRef.current;
    if (!viewport) return;
    following.current = viewport.scrollHeight - viewport.clientHeight - viewport.scrollTop <= CHAT_BOTTOM_THRESHOLD;
    setShowLatest(!following.current);
  }

  useEffect(() => {
    if (!loading) jumpToLatest();
  }, [loading, jumpToLatest]);

  useEffect(() => {
    if (pendingId && sending) jumpToLatest();
  }, [pendingId, sending, jumpToLatest]);

  useEffect(() => { schedule(); }, [contentVersion, schedule]);

  useEffect(() => {
    // Images/markdown may change height after the corresponding text render.
    const content = viewportRef.current?.firstElementChild;
    const observer = typeof ResizeObserver !== "undefined" ? new ResizeObserver(schedule) : null;
    if (content) observer?.observe(content);
    return () => {
      observer?.disconnect();
      cancelFrame.current?.();
      cancelFrame.current = null;
    };
  }, [schedule]);

  return { viewportRef, onScroll, showLatest, jumpToLatest };
}
