"use client";

import { useCallback, useEffect, useState } from "react";
import LeftRail from "@/components/LeftRail";
import Raster from "@/components/Raster";
import SemanticPanel from "@/components/SemanticPanel";
import TopBar from "@/components/TopBar";
import Boot from "@/components/Boot";
import Reader from "@/components/Reader";
import Transcript from "@/components/Transcript";
import Toasts from "@/components/Toasts";
import { API } from "@/lib/api";
import { useBackend } from "@/lib/useBackend";
import { useSession } from "@/lib/useSession";
import { useToasts } from "@/lib/useToasts";
import { useEEG } from "@/lib/useEEG";

export default function Page() {
  const [question, setQuestion] = useState("Why do we dream?");
  const { toasts, notify, dismiss } = useToasts();
  const onError = useCallback((m: string) => notify("error", m), [notify]);
  const backend = useBackend(notify);
  const { status, montage, settings, ready } = backend;
  const session = useSession({ settings, onError });
  const { phase, asked, readout, live, entries, decode, plan, injecting, readT, steer, auto, t0 } = session;
  const { traces, metrics, connected, rate } = useEEG(montage?.display.length ?? 0);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === "Space" && phase === "reading") { e.preventDefault(); session.endReading(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [phase, session.endReading]);

  const start = () => session.start(question);
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
      <Boot status={status} error={backend.unreachable && !status ? `Backend unreachable at ${API}` : null} hidden={ready} />

      <TopBar stageOn={stageOn} connected={connected} status={status} t0={t0} />

      <main className="main">
        <aside className="col">
          <LeftRail montage={montage} metrics={metrics} reading={phase === "reading"} eeg={eeg} decode={decode}
            switching={backend.switching} onSource={backend.setSource} />
        </aside>

        <section className="col reader">
          <Reader phase={phase} asked={asked} tokens={current} count={entries.length} injecting={injecting} readT={readT}
            translator={!!status?.llm?.translator} question={question} ready={ready}
            onQuestion={setQuestion} onStart={start} onNext={session.next} onNewSession={session.newSession} onStop={session.stop} />
          <Transcript entries={entries} />
        </section>

        <aside className="col">
          <SemanticPanel readout={readout} decode={decode} plan={plan} settings={settings} steer={steer} auto={auto}
            onSetting={backend.updateSetting} onSteer={session.setSteer} onAuto={session.setAuto} />
        </aside>
      </main>

      <Toasts toasts={toasts} onDismiss={dismiss} />
      <Raster traces={traces} labels={montage?.display_labels ?? []} rate={rate} source={eeg?.label} />
    </div>
  );
}
