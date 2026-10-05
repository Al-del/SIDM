"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export type ToastKind = "error" | "ok" | "info";
export type Toast = { id: number; kind: ToastKind; text: string; count: number };
export type Notify = (kind: ToastKind, text: string) => void;

const TTL: Record<ToastKind, number> = { error: 7000, ok: 3500, info: 4500 };
const MAX = 4;

export function useToasts() {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const timers = useRef(new Map<number, ReturnType<typeof setTimeout>>());
  const nextId = useRef(1);

  const dismiss = useCallback((id: number) => {
    const t = timers.current.get(id);
    if (t) clearTimeout(t);
    timers.current.delete(id);
    setToasts((ts) => ts.filter((x) => x.id !== id));
  }, []);

  const arm = useCallback((id: number, kind: ToastKind) => {
    const old = timers.current.get(id);
    if (old) clearTimeout(old);
    timers.current.set(id, setTimeout(() => dismiss(id), TTL[kind]));
  }, [dismiss]);

  const notify = useCallback<Notify>((kind, text) => {
    setToasts((ts) => {
      const dup = ts.find((x) => x.kind === kind && x.text === text);
      if (dup) {
        arm(dup.id, kind);
        return ts.map((x) => (x === dup ? { ...x, count: x.count + 1 } : x));
      }
      const id = nextId.current++;
      arm(id, kind);
      const next = [...ts, { id, kind, text, count: 1 }];
      for (const old of next.slice(0, Math.max(0, next.length - MAX))) {
        const t = timers.current.get(old.id);
        if (t) clearTimeout(t);
        timers.current.delete(old.id);
      }
      return next.slice(-MAX);
    });
  }, [arm]);

  useEffect(() => {
    const map = timers.current;
    return () => { map.forEach(clearTimeout); map.clear(); };
  }, []);

  return { toasts, notify, dismiss };
}
