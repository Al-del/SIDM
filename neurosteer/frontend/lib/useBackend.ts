"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { errorText, get, getOptional, post } from "@/lib/api";
import type { Health, Montage, PresetName, Presets, Settings, Status } from "@/lib/api";
import type { Notify } from "@/lib/useToasts";

const POLL_MS = 2000;

export function useBackend(notify: Notify) {
  const [status, setStatus] = useState<Status | null>(null);
  const [montage, setMontage] = useState<Montage | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [health, setHealth] = useState<Health | null | undefined>(undefined);
  const [presets, setPresets] = useState<Presets | null | undefined>(undefined);
  const [unreachable, setUnreachable] = useState(false);
  const [switching, setSwitching] = useState<string | null>(null);

  const notifyRef = useRef(notify);
  notifyRef.current = notify;
  const montageRef = useRef(montage);
  montageRef.current = montage;
  const probed = useRef(false);
  
  const link = useRef(0);
  const pending = useRef<Partial<Settings>>({});
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let alive = true;
    const probe = async () => {
      probed.current = true;
      const [h, p] = await Promise.all([
        getOptional<Health>("/api/health").catch(() => null),
        getOptional<Presets>("/api/presets").catch(() => null),
      ]);
      if (!alive) return;
      setHealth(h);
      setPresets(p);
    };
    const poll = async () => {
      try {
        const s = await get<Status>("/api/status");
        if (!alive) return;
        setStatus(s);
        setUnreachable(false);
        if (link.current === 2) notifyRef.current("ok", "Backend link restored");
        link.current = 1;
        setSettings((cur) => cur ?? s.settings);
        if (!montageRef.current) get<Montage>("/api/montage").then((m) => alive && setMontage(m)).catch(() => {});
        if (!probed.current) probe();
      } catch {
        if (!alive) return;
        setUnreachable(true);
        if (link.current === 1) {
          link.current = 2;
          notifyRef.current("error", "Lost the link to the backend. Retrying…");
        }
      }
    };
    poll();
    const id = setInterval(poll, POLL_MS);
    return () => {
      alive = false;
      clearInterval(id);
      if (saveTimer.current) clearTimeout(saveTimer.current);
    };
  }, []);

  
  const updateSetting = useCallback((k: keyof Settings, v: number | boolean) => {
    setSettings((s) => (s ? { ...s, [k]: v } : s));
    pending.current = { ...pending.current, [k]: v };
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => {
      const patch = pending.current;
      pending.current = {};
      post<Settings>("/api/settings", patch).catch((e) => notifyRef.current("error", errorText(e)));
    }, 200);
  }, []);

  const applyPreset = useCallback(async (name: PresetName) => {
    if (saveTimer.current) clearTimeout(saveTimer.current);
    pending.current = {};
    try {
      const s = await post<Settings>("/api/settings", { preset: name });
      setSettings((cur) => ({ ...(cur ?? s), ...s }));
      notifyRef.current("ok", `Steering preset: ${name}`);
    } catch (e) {
      notifyRef.current("error", errorText(e));
    }
  }, []);

  
  const resync = useCallback(async () => {
    try {
      const s = await get<Status>("/api/status");
      setStatus(s);
      setSettings(s.settings);
    } catch {}
  }, []);

  const setSource = useCallback(async (kind: string) => {
    try {
      setSwitching(kind);
      await post("/api/source", { kind });
      setStatus(await get<Status>("/api/status"));
      setMontage(await get<Montage>("/api/montage"));
      notifyRef.current("ok", `Signal source: ${kind}`);
    } catch (e) {
      notifyRef.current("error", errorText(e));
    } finally {
      setSwitching(null);
    }
  }, []);

  const mock = !!(status?.mock || status?.llm?.mock);
  const ready = !!status && !status.loading && status.decoder && !!status.llm;

  return {
    status, montage, settings, health, presets, unreachable, switching, mock, ready,
    updateSetting, applyPreset, setSource, resync,
  };
}
