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
import { API } from "@/lib/api";
import { useBackend } from "@/lib/useBackend";
import { useSession } from "@/lib/useSession";
import { useToasts } from "@/lib/useToasts";
import { useEEG } from "@/lib/useEEG";
import { useHotkeys } from "@/lib/useHotkeys";
import { useStats } from "@/lib/useStats";

type Overlay = "keys" | "help" | null;

export default function Page() {
  const [question, setQuestion] = useState("Why do we dream?");
  const { toasts, notify, dismiss } = useToasts();
  const onError = useCallback((m: string) => notify("error", m), [notify]);
  const backend = useBackend(notify);
  const { status, montage, settings, ready } = backend;
  const [statsTick, setStatsTick] = useState(0);
  const onSentence = useCallback(() => setStatsTick((t) => t + 1), []);
  const session = useSession({ settings, onError, onSentence });
  const { phase, asked, readout, live, entries, decode, plan, injecting, readT, steer, auto, t0 } = session;
  const { traces, metrics, connected, rate } = useEEG(montage?.display.length ?? 0);
  const { stats, remote } = useStats(statsTick, entries);

  const [overlay, setOverlay] = useState<Overlay>(null);
  const toggle = (o: Exclude<Overlay, null>) => setOverlay((cur) => (cur === o ? null : o));

  useHotkeys({
    " ": phase === "reading" && session.endReading,
    k: () => toggle("keys"),
    "?": () => toggle("help"),
    Escape: () => setOverlay(null),
  });

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
      <Boot status={status} mock={backend.mock} error={backend.unreachable && !status ? `Backend unreachable at ${API}` : null} hidden={ready} />

      <TopBar stageOn={stageOn} connected={connected} status={status} health={backend.health} mock={backend.mock} t0={t0}>
        <div className="tools">
          <button className={`tool ${overlay === "help" ? "on" : ""}`} onClick={() => toggle("help")}
            aria-label="How it works" title="How it works (?)">HOW IT WORKS</button>
          <button className={`tool ${overlay === "keys" ? "on" : ""}`} onClick={() => toggle("keys")}
            aria-expanded={overlay === "keys"} aria-label="Keyboard shortcuts" title="Keyboard shortcuts (K)">KEYS</button>
          <ShortcutHint open={overlay === "keys"} onClose={() => setOverlay(null)} />
        </div>
      </TopBar>

      <main className="main">
        <aside className="col">
          <LeftRail montage={montage} metrics={metrics} reading={phase === "reading"} eeg={eeg} decode={decode}
            switching={backend.switching} onSource={backend.setSource} />
        </aside>

        <section className="col reader">
          <Reader phase={phase} asked={asked} tokens={current} count={entries.length} injecting={injecting} readT={readT}
            translator={!!status?.llm?.translator} question={question} ready={ready}
            onQuestion={setQuestion} onStart={start} onNext={session.next} onNewSession={session.newSession} onStop={session.stop} />
          <div className="sessionbar">
            <StatsStrip stats={stats} remote={remote} />
          </div>
          <Transcript entries={entries} />
        </section>

        <aside className="col">
          <SemanticPanel readout={readout} decode={decode} plan={plan} settings={settings} steer={steer} auto={auto}
            onSetting={backend.updateSetting} onSteer={session.setSteer} onAuto={session.setAuto} />
        </aside>
      </main>

      <HowItWorks open={overlay === "help"} onClose={() => setOverlay(null)} source={eeg?.label} mock={backend.mock} />
      <Toasts toasts={toasts} onDismiss={dismiss} />
      <Raster traces={traces} labels={montage?.display_labels ?? []} rate={rate} source={eeg?.label} />
    </div>
  );
}
