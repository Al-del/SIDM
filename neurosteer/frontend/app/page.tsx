"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import LeftRail from "@/components/LeftRail";
import Raster from "@/components/Raster";
import SemanticPanel from "@/components/SemanticPanel";
import TopBar from "@/components/TopBar";
import Boot from "@/components/Boot";
import Reader from "@/components/Reader";
import Transcript from "@/components/Transcript";
import Toasts from "@/components/Toasts";
import { API, errorText, get, post } from "@/lib/api";
import type { Montage, Settings, Status } from "@/lib/api";
import { useSession } from "@/lib/useSession";
import { useToasts } from "@/lib/useToasts";
import { useEEG } from "@/lib/useEEG";

export default function Page() {
  const [status, setStatus] = useState<Status | null>(null);
  const [montage, setMontage] = useState<Montage | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [question, setQuestion] = useState("Why do we dream?");
  const [bootError, setBootError] = useState<string | null>(null);
  const [switching, setSwitching] = useState<string | null>(null);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const { toasts, notify, dismiss } = useToasts();
  const onError = useCallback((m: string) => notify("error", m), [notify]);
  const session = useSession({ settings, onError });
  const { phase, asked, readout, live, entries, decode, plan, injecting, readT, steer, auto, t0 } = session;
  const { traces, metrics, connected, rate } = useEEG(montage?.display.length ?? 0);

  useEffect(() => {
    get<Montage>("/api/montage").then(setMontage).catch(() => setBootError("Backend unreachable at " + API));
    let alive = true;
    const poll = async () => {
      try {
        const s = await get<Status>("/api/status");
        if (!alive) return;
        setStatus(s);
        setSettings((cur) => cur ?? s.settings);
      } catch {}
    };
    poll();
    const id = setInterval(poll, 2000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === "Space" && phase === "reading") { e.preventDefault(); session.endReading(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [phase, session.endReading]);

  const start = () => session.start(question);

  const onSetting = (k: keyof Settings, v: number | boolean) => {
    setSettings((s) => {
      if (!s) return s;
      const next = { ...s, [k]: v };
      if (saveTimer.current) clearTimeout(saveTimer.current);
      saveTimer.current = setTimeout(() => post("/api/settings", { [k]: v }).catch((e) => onError(errorText(e))), 200);
      return next;
    });
  };

  const setSource = async (kind: string) => {
    try {
      setSwitching(kind);
      await post("/api/source", { kind });
      setStatus(await get<Status>("/api/status"));
      setMontage(await get<Montage>("/api/montage"));
      notify("ok", `Signal source: ${kind}`);
    } catch (e) {
      onError(errorText(e));
    } finally {
      setSwitching(null);
    }
  };

  const ready = !!status && !status.loading && status.decoder && !!status.llm;
  const current = phase === "generating" ? live : entries[entries.length - 1]?.tokens ?? [];
  const stageOn = {
    READ: phase === "reading",
    DECODE: phase === "decoding",
    INJECT: phase === "generating" && injecting,
    GENERATE: phase === "generating" && !injecting,
  };
  const eeg = status?.eeg;

  return (
    <div className="shell">
      <Boot status={status} error={!status ? bootError : null} hidden={ready} />

      <TopBar stageOn={stageOn} connected={connected} status={status} t0={t0} />

      <main className="main">
        <aside className="col">
          <LeftRail montage={montage} metrics={metrics} reading={phase === "reading"} eeg={eeg} decode={decode}
            switching={switching} onSource={setSource} />
        </aside>

        <section className="col reader">
          <Reader phase={phase} asked={asked} tokens={current} count={entries.length} injecting={injecting} readT={readT}
            translator={!!status?.llm?.translator} question={question} ready={ready}
            onQuestion={setQuestion} onStart={start} onNext={session.next} onNewSession={session.newSession} onStop={session.stop} />
          <Transcript entries={entries} />
        </section>

        <aside className="col">
          <SemanticPanel readout={readout} decode={decode} plan={plan} settings={settings} steer={steer} auto={auto}
            onSetting={onSetting} onSteer={session.setSteer} onAuto={session.setAuto} />
        </aside>
      </main>

      <Toasts toasts={toasts} onDismiss={dismiss} />
      <Raster traces={traces} labels={montage?.display_labels ?? []} rate={rate} source={eeg?.label} />
    </div>
  );
}
