"use client";

import { useEffect, useRef } from "react";

export type Hotkeys = Record<string, ((e: KeyboardEvent) => void) | false | undefined>;

function isTyping(t: EventTarget | null): boolean {
  if (!(t instanceof HTMLElement)) return false;
  if (t.isContentEditable) return true;
  if (t instanceof HTMLTextAreaElement || t instanceof HTMLSelectElement) return true;
  if (t instanceof HTMLInputElement) {
    return !["range", "checkbox", "radio", "button", "submit", "reset", "color"].includes(t.type);
  }
  return false;
}

export function useHotkeys(keys: Hotkeys) {
  const ref = useRef(keys);
  ref.current = keys;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
      if (isTyping(e.target)) {
        if (e.key === "Escape") (e.target as HTMLElement).blur();
        else return;
      }
      const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
      const fn = ref.current[k] ?? ref.current[e.key];
      if (!fn) return;
      e.preventDefault();
      fn(e);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}

export const SHORTCUTS: [string, string][] = [
  ["SPACE", "Done reading this sentence"],
  ["?", "How it works"],
  ["C", "Compare steered vs baseline"],
  ["E", "Export session as JSON"],
  ["K", "Show these shortcuts"],
  ["ESC", "Close overlays"],
];
