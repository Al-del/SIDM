"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export type ToastKind = "error" | "ok" | "info";
export type Toast = { id: number; kind: ToastKind; text: string; count: number };
export type Notify = (kind: ToastKind, text: string) => void;

const TTL: Record<ToastKind, number> = { error: 7000, ok: 3500, info: 4500 };
const MAX = 4;

export function useToasts() {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const list = useRef<Toast[]>([]);
  const timers = useRef(new Map<number, ReturnType<typeof setTimeout>>());
  const nextId = useRef(1);

  const commit = useCallback((next: Toast[]) => {
    list.current = next;
    setToasts(next);
  }, []);

  const drop = useCallback((id: number) => {
    const t = timers.current.get(id);
    if (t) clearTimeout(t);
    timers.current.delete(id);
  }, []);

  const dismiss = useCallback((id: number) => {
    drop(id);
    commit(list.current.filter((x) => x.id !== id));
  }, [commit, drop]);

  const arm = useCallback((id: number, kind: ToastKind) => {
    drop(id);
    timers.current.set(id, setTimeout(() => dismiss(id), TTL[kind]));
  }, [dismiss, drop]);

  const notify = useCallback<Notify>((kind, text) => {
    const cur = list.current;
    const dup = cur.find((x) => x.kind === kind && x.text === text);
    if (dup) {
      arm(dup.id, kind);
      commit(cur.map((x) => (x === dup ? { ...x, count: x.count + 1 } : x)));
      return;
    }
    const id = nextId.current++;
    arm(id, kind);
    const next = [...cur, { id, kind, text, count: 1 }];
    next.slice(0, Math.max(0, next.length - MAX)).forEach((old) => drop(old.id));
    commit(next.slice(-MAX));
  }, [arm, commit, drop]);

  useEffect(() => {
    const map = timers.current;
    return () => { map.forEach(clearTimeout); map.clear(); };
  }, []);

  return { toasts, notify, dismiss };
}
