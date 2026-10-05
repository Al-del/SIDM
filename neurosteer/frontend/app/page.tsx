"use client";

import { useCallback, useState } from "react";
import LeftRail from "@/components/LeftRail";
import Raster from "@/components/Raster";
import SemanticPanel from "@/components/SemanticPanel";
import TopBar from "@/components/TopBar";
import Boot from "@/components/Boot";
import Reader from "@/components/Reader";
import Transcript from "@/components/Transcript";
import Toasts from "@/components/Toasts";
import ShortcutHint from "@/components/ShortcutHint";
import HowItWorks from "@/components/HowItWorks";
import StatsStrip from "@/components/StatsStrip";
import SessionActions from "@/components/SessionActions";
import ComparePanel from "@/components/ComparePanel";
import { ApiError, errorText, post } from "@/lib/api";
import { exportSession } from "@/lib/exportSession";
import { useBackend } from "@/lib/useBackend";
import { useSession } from "@/lib/useSession";
import { useToasts } from "@/lib/useToasts";
import { useEEG } from "@/lib/useEEG";
import { useHotkeys } from "@/lib/useHotkeys";
import { useStats } from "@/lib/useStats";

type Overlay = "keys" | "help" | "compare" | null;

export default function Page() {
  const [question, setQuestion] = useState("Why do we dream?");
  const { toasts, notify, dismiss } = useToasts();
  const onError = useCallback((m: string) => notify("error", m), [notify]);
  const backend = useBackend(notify);
  const { status, montage, settings, ready } = backend;
  const [statsTick, setStatsTick] = useState(0);
  const onSentence = useCallback(() => setStatsTick((t) => t + 1), []);
  const session = useSession({ settings, onError, onSentence });
  const { phase, asked, readout, live, entries, decode, plan, injecting, readStartedAt, steer, auto, t0 } = session;
  const { traces, metrics, connected, rate } = useEEG(montage?.display.length ?? 0);
  const { stats, remote } = useStats(statsTick, entries);

  const [overlay, setOverlay] = useState<Overlay>(null);
  const [sideOpen, setSideOpen] = useState(false);
  const modalOpen = overlay === "help" || overlay === "compare";
  const toggle = (o: Exclude<Overlay, null>) => setOverlay((cur) => (cur === o ? null : o));

  useHotkeys({
    " ": phase === "reading" && !modalOpen && session.endReading,
    k: () => toggle("keys"),
    "?": () => toggle("help"),
    Escape: () => { setOverlay(null); setSideOpen(false); },
    e: !modalOpen && (() => doExport()),
    c: () => (overlay === "compare" ? setOverlay(null) : openCompare()),
  });

  const start = () => session.start(question);

  const [exporting, setExporting] = useState(false);
  const doExport = async () => {
    if (exporting || !entries.length) return;
    setExporting(true);
    try {
      const from = await exportSession({ question: asked, entries, settings });
      notify("ok", from === "backend" ? "Session exported" : "Session exported from the browser copy (no /api/export)");
    } catch (e) {
      onError(errorText(e));
    } finally {
      setExporting(false);
    }
  };

  const busy = phase === "generating" || phase === "reading" || phase === "decoding";
  const compareHint = backend.health === null ? "This backend has no /api/compare"
    : !entries.length ? "Ask a question first: compare needs a sentence to continue from"
    : busy ? "Pause the loop (Stop) to compare" : "Generate the next sentence with and without steering (C)";
  const canCompare = backend.health !== null && entries.length > 0 && !busy;
  const openCompare = () => {
    if (canCompare) setOverlay("compare");
    else notify("info", compareHint);
  };

  const doReset = async () => {
    session.clear();
    try {
      await post("/api/reset");
      notify("ok", "Session reset");
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) notify("info", "Cleared locally; this backend has no /api/reset");
      else onError(errorText(e));
    }
    backend.resync();
    setStatsTick((t) => t + 1);
  };
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
      <Boot status={status} mock={backend.mock} unreachable={backend.unreachable} hidden={ready} onHelp={() => setOverlay("help")} />

      <TopBar stageOn={stageOn} connected={connected} status={status} health={backend.health} mock={backend.mock} offline={backend.unreachable && !!status} t0={t0}>
        <div className="tools">
          <button className={`tool ${overlay === "help" ? "on" : ""}`} onClick={() => toggle("help")}
            aria-label="How it works" title="How it works (?)"><span className="long">HOW IT WORKS</span><span className="short">?</span></button>
          <button className={`tool panel-toggle ${sideOpen ? "on" : ""}`} onClick={() => setSideOpen((o) => !o)}
            aria-expanded={sideOpen} aria-controls="side-panel" title="Semantic panel and steering controls">PANEL</button>
          <button className={`tool ${overlay === "keys" ? "on" : ""}`} onClick={() => toggle("keys")}
            aria-expanded={overlay === "keys"} aria-label="Keyboard shortcuts" title="Keyboard shortcuts (K)">KEYS</button>
          <ShortcutHint open={overlay === "keys"} onClose={() => setOverlay(null)} />
        </div>
      </TopBar>

      <main className={`main ${sideOpen ? "side-open" : ""}`}>
        <aside className="col">
          <LeftRail montage={montage} metrics={metrics} reading={phase === "reading"} eeg={eeg} decode={decode}
            switching={backend.switching} onSource={backend.setSource} />
        </aside>

        <section className="col reader">
          <Reader phase={phase} asked={asked} tokens={current} count={entries.length} injecting={injecting} readStartedAt={readStartedAt}
            translator={!!status?.llm?.translator} question={question} ready={ready}
            onQuestion={setQuestion} onStart={start} onNext={session.next} onNewSession={session.newSession} onStop={session.stop} />
          <div className="sessionbar">
            <StatsStrip stats={stats} remote={remote} />
            <SessionActions onExport={doExport} onReset={doReset} exporting={exporting} canExport={entries.length > 0}
              compare={backend.health === null ? undefined : { onClick: openCompare, disabled: !canCompare, hint: compareHint }} />
          </div>
          <Transcript entries={entries} />
        </section>

        <div className="side-scrim" onClick={() => setSideOpen(false)} aria-hidden />
        <aside className="col side" id="side-panel" aria-label="Semantic decode and steering">
          <SemanticPanel readout={readout} decode={decode} plan={plan} settings={settings} steer={steer} auto={auto}
            onSetting={backend.updateSetting} onSteer={session.setSteer} onAuto={session.setAuto}
            presets={backend.presets} onPreset={backend.applyPreset} />
        </aside>
      </main>

      <HowItWorks open={overlay === "help"} onClose={() => setOverlay(null)} source={eeg?.label} mock={backend.mock} />
      <ComparePanel open={overlay === "compare"} onClose={() => setOverlay(null)} question={asked} />
      <Toasts toasts={toasts} onDismiss={dismiss} />
      <Raster traces={traces} labels={montage?.display_labels ?? []} rate={rate} source={eeg?.label} />
    </div>
  );
}
