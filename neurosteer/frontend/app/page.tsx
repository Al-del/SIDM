"use client";

import { useEffect, useRef, useState } from "react";
import LeftRail from "@/components/LeftRail";
import Raster from "@/components/Raster";
import SemanticPanel from "@/components/SemanticPanel";
import TopBar from "@/components/TopBar";
import Boot from "@/components/Boot";
import { API, get, post } from "@/lib/api";
import type { Montage, Settings, Status } from "@/lib/api";
import { MAX_READ, useSession } from "@/lib/useSession";
import { useEEG } from "@/lib/useEEG";

export default function Page() {
  const [status, setStatus] = useState<Status | null>(null);
  const [montage, setMontage] = useState<Montage | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [question, setQuestion] = useState("Why do we dream?");
  const [error, setError] = useState<string | null>(null);
  const [switching, setSwitching] = useState<string | null>(null);
  const tailRef = useRef<HTMLDivElement>(null);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const session = useSession({ settings, onError: setError });
  const { phase, asked, readout, live, entries, decode, plan, injecting, readT, steer, auto, t0 } = session;
  const { traces, metrics, connected, rate } = useEEG(montage?.display.length ?? 0);

  useEffect(() => {
    get<Montage>("/api/montage").then(setMontage).catch(() => setError("Backend unreachable at " + API));
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
    tailRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [entries.length]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === "Space" && phase === "reading") { e.preventDefault(); session.endReading(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [phase, session.endReading]);

  const start = () => {
    setError(null);
    session.start(question);
  };

  const onSetting = (k: keyof Settings, v: number | boolean) => {
    setSettings((s) => {
      if (!s) return s;
      const next = { ...s, [k]: v };
      if (saveTimer.current) clearTimeout(saveTimer.current);
      saveTimer.current = setTimeout(() => post("/api/settings", { [k]: v }).catch(() => {}), 200);
      return next;
    });
  };

  const setSource = async (kind: string) => {
    try {
      setSwitching(kind);
      await post("/api/source", { kind });
      setStatus(await get<Status>("/api/status"));
      setMontage(await get<Montage>("/api/montage"));
      setError(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setSwitching(null);
    }
  };

  const ready = !!status && !status.loading && status.decoder && !!status.llm;
  const current = phase === "generating" ? live : entries[entries.length - 1]?.tokens ?? [];
  const history = entries;
  const stageOn = {
    READ: phase === "reading",
    DECODE: phase === "decoding",
    INJECT: phase === "generating" && injecting,
    GENERATE: phase === "generating" && !injecting,
  };
  const eeg = status?.eeg;

  return (
    <div className="shell">
      <Boot status={status} error={error && !status ? error : null} hidden={ready} />

      <TopBar stageOn={stageOn} connected={connected} status={status} t0={t0} />

      <main className="main">
        <aside className="col">
          <LeftRail montage={montage} metrics={metrics} reading={phase === "reading"} eeg={eeg} decode={decode}
            switching={switching} onSource={setSource} />
        </aside>

        <section className="col reader">
          <div className="stagebox">
            <span className="corner tl" /><span className="corner tr" /><span className="corner bl" /><span className="corner br" />
            <div className="stage-meta">
              <span className="label num">{entries.length ? `SENTENCE ${String(entries.length + (phase === "generating" ? 1 : 0)).padStart(2, "0")}` : "SESSION"}</span>
              <span className="label">{phase === "idle" ? "standby" : phase}{status?.llm?.translator ? " · translator" : ""}</span>
            </div>

            {phase === "idle" ? (
              <div className="start">
                <h1>Ask anything.<br /><em>It answers as you read.</em></h1>
                <div className="field">
                  <span className="label">Question</span>
                  <input value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Ask a question"
                    onKeyDown={(e) => e.key === "Enter" && ready && question.trim() && start()} />
                </div>
                <div className="actions">
                  <button className="btn primary" disabled={!ready || !question.trim()} onClick={start}>Ask</button>
                  <span className="label">One sentence at a time · SPACE when read</span>
                </div>
              </div>
            ) : (
              <>
                <p className="question"><b>Q</b><span>{asked}</span></p>
                <div className="sentence">
                  {current.map((t, i) => (
                    <span key={i} className={`tok ${t.steered ? "steer" : ""}`} title={t.steered ? `bias +${t.bias}` : undefined}>{t.text}</span>
                  ))}
                  {phase === "generating" && <span className="caret" />}
                  {phase === "decoding" && <span className="scan" />}
                </div>
                <div className="timer">
                  {phase === "reading" && (
                    <>
                      <div className="timer-track"><div className="timer-fill" style={{ width: `${(readT / MAX_READ) * 100}%` }} /></div>
                      <span className="label num">{readT.toFixed(1)} / {MAX_READ}.0 s</span>
                      <span className="key">SPACE</span>
                    </>
                  )}
                  {phase === "decoding" && <span className="label">Decoding epoch through RAG-Mosaic…</span>}
                  {phase === "generating" && <span className="label">{injecting ? "Injecting neural context…" : "Qwen generating"}</span>}
                  {phase === "complete" && (
                    <div className="actions">
                      <span className="label" style={{ color: "var(--signal)" }}>Answer complete · {entries.length} sentences</span>
                      <button className="btn primary" onClick={session.newSession}>Ask another</button>
                    </div>
                  )}
                  {phase === "ready" && (
                    <div className="actions">
                      <button className="btn primary" onClick={session.next}>Next sentence</button>
                      <button className="btn ghost" onClick={session.newSession}>New session</button>
                    </div>
                  )}
                  {(phase === "reading" || phase === "generating") && (
                    <button className="btn ghost" style={{ marginLeft: "auto" }} onClick={session.stop}>Stop</button>
                  )}
                </div>
                {error && <p className="notice" style={{ maxWidth: 520 }}>{error}</p>}
              </>
            )}
          </div>

          <div className="transcript">
            {history.length === 0 && <p className="empty" style={{ padding: "20px 34px" }}>The answer builds here sentence by sentence. Amber words were pushed by the neural bias; → chips were injected into Qwen, ← chips were decoded from your EEG while reading.</p>}
            {history.map((e) => (
              <div className="tr-row" key={e.id}>
                <span className="tr-idx num">{String(e.id + 1).padStart(2, "0")}</span>
                <div>
                  <p className="tr-text">
                    {e.tokens.map((t, i) => <span key={i} className={t.steered ? "steer" : ""}>{t.text}</span>)}
                  </p>
                  <div className="chips">
                    {e.plan?.units.slice(0, 6).map((u) => <span key={"o" + u} className="chip out">→ {u}</span>)}
                    {e.decode?.units.slice(0, 6).map((u) => <span key={"i" + u.unit} className="chip in">← {u.word}</span>)}
                    <span className="chip num">{e.ms} ms</span>
                  </div>
                </div>
              </div>
            ))}
            <div ref={tailRef} />
          </div>
        </section>

        <aside className="col">
          <SemanticPanel readout={readout} decode={decode} plan={plan} settings={settings} steer={steer} auto={auto}
            onSetting={onSetting} onSteer={session.setSteer} onAuto={session.setAuto} />
        </aside>
      </main>

      <Raster traces={traces} labels={montage?.display_labels ?? []} rate={rate} source={eeg?.label} />
    </div>
  );
}
