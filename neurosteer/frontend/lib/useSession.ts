"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { API, errorText, post, streamErrorText } from "@/lib/api";
import type { Decode, Entry, Plan, Settings, Token } from "@/lib/api";

export type Phase = "idle" | "generating" | "reading" | "decoding" | "ready" | "complete";

export const MAX_READ = 9;

type Options = {
  settings: Settings | null;
  onError: (message: string) => void;
  
  onSentence?: () => void;
};

export function useSession({ settings, onError, onSentence }: Options) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [asked, setAsked] = useState("");
  const [readout, setReadout] = useState<string | null>(null);
  const [live, setLive] = useState<Token[]>([]);
  const [entries, setEntries] = useState<Entry[]>([]);
  const [decode, setDecode] = useState<Decode | null>(null);
  const [plan, setPlan] = useState<Plan>(null);
  const [injecting, setInjecting] = useState(false);
  
  const [readStartedAt, setReadStartedAt] = useState(0);
  const [steer, setSteer] = useState(true);
  const [auto, setAuto] = useState(true);
  const [t0, setT0] = useState<number | null>(null);

  const phaseRef = useRef<Phase>("idle");
  const autoRef = useRef(auto);
  const steerRef = useRef(steer);
  const settingsRef = useRef(settings);
  const onErrorRef = useRef(onError);
  const onSentenceRef = useRef(onSentence);
  const readStart = useRef(0);
  const esRef = useRef<EventSource | null>(null);
  const endRef = useRef(false);
  const generateRef = useRef<() => void>(() => {});
  
  const runRef = useRef(0);
  const startingRef = useRef(false);
  autoRef.current = auto;
  steerRef.current = steer;
  settingsRef.current = settings;
  onErrorRef.current = onError;
  onSentenceRef.current = onSentence;

  const go = useCallback((p: Phase) => { phaseRef.current = p; setPhase(p); }, []);

  const endReading = useCallback(async () => {
    if (phaseRef.current !== "reading") return;
    go("decoding");
    const run = runRef.current;
    try {
      const d = await post<Decode>("/api/read/end");
      if (run !== runRef.current) return;
      setDecode(d);
      setEntries((es) => es.map((e, i) => (i === es.length - 1 ? { ...e, decode: d } : e)));
      onSentenceRef.current?.();
      if (endRef.current) go("complete");
      else if (autoRef.current) generateRef.current();
      else go("ready");
    } catch (e) {
      if (run !== runRef.current) return;
      onErrorRef.current(errorText(e));
      go("ready");
    }
  }, [go]);

  const startReading = useCallback(async () => {
    const run = runRef.current;
    try {
      await post("/api/read/start");
    } catch (e) {
      if (run !== runRef.current) return;
      onErrorRef.current(errorText(e));
      go("ready");
      return;
    }
    if (run !== runRef.current) return;
    readStart.current = performance.now();
    setReadStartedAt(readStart.current);
    go("reading");
  }, [go]);

  const generate = useCallback(() => {
    esRef.current?.close();
    go("generating");
    setLive([]);
    setPlan(null);
    const es = new EventSource(`${API}/api/generate${steerRef.current ? "" : "?steer=0"}`);
    esRef.current = es;
    es.addEventListener("plan", (ev) => {
      const p = JSON.parse((ev as MessageEvent).data) as Plan;
      setPlan(p);
      setInjecting(!!p);
    });
    es.addEventListener("token", (ev) => {
      setInjecting(false);
      const t = JSON.parse((ev as MessageEvent).data) as Token;
      setLive((l) => [...l, t]);
    });
    es.addEventListener("done", (ev) => {
      es.close();
      if (esRef.current === es) esRef.current = null;
      const rec = JSON.parse((ev as MessageEvent).data) as Entry;
      endRef.current = rec.end;
      if (rec.decode === null) post<{ text: string | null }>("/api/readout").then((r) => setReadout(r.text)).catch(() => {});
      if (!rec.sentence) { setLive([]); go("complete"); onSentenceRef.current?.(); return; }
      setEntries((e) => [...e, rec]);
      startReading();
    });

    es.onerror = (ev) => {
      es.close();
      if (esRef.current === es) esRef.current = null;
      if (phaseRef.current === "generating") {
        setInjecting(false);
        onErrorRef.current(streamErrorText(ev, "Generation failed: no active session, Qwen busy, or backend unreachable"));
        go("ready");
      }
    };
  }, [go, startReading]);
  generateRef.current = generate;

  useEffect(() => {
    if (phase !== "reading") return;
    const left = MAX_READ * 1000 - (performance.now() - readStart.current);
    const id = setTimeout(endReading, Math.max(0, left));
    return () => clearTimeout(id);
  }, [phase, endReading]);

  useEffect(() => () => esRef.current?.close(), []);

  const start = useCallback(async (question: string) => {
    if (startingRef.current) return;
    startingRef.current = true;
    esRef.current?.close();
    esRef.current = null;
    const run = ++runRef.current;
    try {
      await post("/api/session", { question, settings: settingsRef.current });
    } catch (e) {
      onErrorRef.current(errorText(e));
      return;
    } finally {
      startingRef.current = false;
    }
    if (run !== runRef.current) return;
    setAsked(question);
    endRef.current = false;
    setEntries([]);
    setDecode(null);
    setPlan(null);
    setReadout(null);
    setT0(Date.now());
    generateRef.current();
  }, []);

  const stop = useCallback(() => {
    esRef.current?.close();
    esRef.current = null;
    autoRef.current = false;
    setAuto(false);
    setInjecting(false);
    if (phaseRef.current === "reading") endReading();
    else if (phaseRef.current === "generating") go("ready");
  }, [endReading, go]);

  const newSession = useCallback(() => {
    runRef.current++;
    esRef.current?.close();
    esRef.current = null;
    setT0(null);
    go("idle");
  }, [go]);

  
  const clear = useCallback(() => {
    runRef.current++;
    esRef.current?.close();
    esRef.current = null;
    endRef.current = false;
    setEntries([]);
    setLive([]);
    setDecode(null);
    setPlan(null);
    setReadout(null);
    setInjecting(false);
    setAsked("");
    setT0(null);
    go("idle");
  }, [go]);

  return {
    phase, asked, readout, live, entries, decode, plan, injecting, readStartedAt, steer, auto, t0,
    setSteer, setAuto, start, stop, next: generate, endReading, newSession, clear,
  };
}
