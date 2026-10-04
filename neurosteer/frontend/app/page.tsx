"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ElectrodeArray from "@/components/ElectrodeArray";
import Raster from "@/components/Raster";
import SemanticPanel from "@/components/SemanticPanel";
import Logo from "@/components/Logo";
import { API, get, post } from "@/lib/api";
import type { Decode, Entry, Montage, Plan, Settings, Status, Token } from "@/lib/api";
import { useEEG } from "@/lib/useEEG";

type Phase = "idle" | "generating" | "reading" | "decoding" | "ready" | "complete";
const MAX_READ = 9;
const BANDS = ["delta", "theta", "alpha", "beta", "gamma"];

export default function Page() {
  const [status, setStatus] = useState<Status | null>(null);
  const [montage, setMontage] = useState<Montage | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [question, setQuestion] = useState("Why do we dream?");
  const [asked, setAsked] = useState("");
  const [readout, setReadout] = useState<string | null>(null);
  const [live, setLive] = useState<Token[]>([]);
  const [entries, setEntries] = useState<Entry[]>([]);
  const [decode, setDecode] = useState<Decode | null>(null);
  const [plan, setPlan] = useState<Plan>(null);
  const [injecting, setInjecting] = useState(false);
  const [readT, setReadT] = useState(0);
  const [steer, setSteer] = useState(true);
  const [auto, setAuto] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [t0, setT0] = useState<number | null>(null);
  const [clock, setClock] = useState(0);

  const phaseRef = useRef<Phase>("idle");
  const autoRef = useRef(auto);
  const steerRef = useRef(steer);
  const readStart = useRef(0);
  const esRef = useRef<EventSource | null>(null);
  const endRef = useRef(false);
  const tailRef = useRef<HTMLDivElement>(null);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  autoRef.current = auto;
  steerRef.current = steer;

  const go = (p: Phase) => { phaseRef.current = p; setPhase(p); };
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
    if (t0 === null) return;
    const id = setInterval(() => setClock(Date.now() - t0), 250);
    return () => clearInterval(id);
  }, [t0]);

  const endReading = useCallback(async () => {
    if (phaseRef.current !== "reading") return;
    go("decoding");
    try {
      const d = await post<Decode>("/api/read/end");
      setDecode(d);
      setEntries((es) => es.map((e, i) => (i === es.length - 1 ? { ...e, decode: d } : e)));
      if (endRef.current) go("complete");
      else if (autoRef.current) generate();
      else go("ready");
    } catch (e) {
      setError(String(e));
      go("ready");
    }
  }, []);

  const startReading = useCallback(async () => {
    await post("/api/read/start");
    readStart.current = performance.now();
    setReadT(0);
    go("reading");
  }, []);

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
      const rec = JSON.parse((ev as MessageEvent).data) as Entry;
      endRef.current = rec.end;
      if (rec.decode === null) post<{ text: string | null }>("/api/readout").then((r) => setReadout(r.text)).catch(() => {});
      if (!rec.sentence) { setLive([]); go("complete"); return; }
      setEntries((e) => [...e, rec]);
      startReading();
    });
    es.onerror = () => {
      es.close();
      if (phaseRef.current === "generating") { setError("Generation stream failed"); go("ready"); }
    };
  }, [startReading]);

  useEffect(() => {
    tailRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [entries.length]);

  useEffect(() => {
    if (phase !== "reading") return;
    let raf = 0;
    const tick = () => {
      const t = (performance.now() - readStart.current) / 1000;
      setReadT(t);
      if (t >= MAX_READ) { endReading(); return; }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [phase, endReading]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === "Space" && phaseRef.current === "reading") { e.preventDefault(); endReading(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [endReading]);

  const start = async () => {
    setError(null);
    await post("/api/session", { question, settings });
    setAsked(question);
    endRef.current = false;
    setEntries([]);
    setDecode(null);
    setPlan(null);
    setReadout(null);
    setT0(Date.now());
    generate();
  };

  const stop = () => {
    esRef.current?.close();
    setAuto(false);
    if (phaseRef.current === "reading") endReading();
    else if (phaseRef.current === "generating") go("ready");
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
      await post("/api/source", { kind });
      setStatus(await get<Status>("/api/status"));
      setError(null);
    } catch (e) {
      setError(String(e));
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
  const mmss = (ms: number) => {
    const s = Math.floor(ms / 1000);
    return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}.${Math.floor((ms % 1000) / 100)}`;
  };

  return (
    <div className="shell">
      <Boot status={status} error={error && !status ? error : null} hidden={ready} />

      <header className="topbar">
        <div className="brand">
          <Logo />
          <span className="brand-name">NEUROSTEER</span>
          <span className="brand-sub mono">EEG → QWEN / CLOSED LOOP</span>
        </div>
        <nav className="pipeline">
          {(["READ", "DECODE", "INJECT", "GENERATE"] as const).map((s, i) => (
            <div key={s} style={{ display: "flex", alignItems: "center" }}>
              {i > 0 && <span className={`wire ${stageOn[s] ? "live" : ""}`} />}
              <span className={`stage ${stageOn[s] ? "on" : ""} ${s === "INJECT" || s === "READ" ? "inj" : ""}`}>
                <span className="pip" />{s}
              </span>
            </div>
          ))}
        </nav>
        <div className="links">
          <span className="link"><span className={`dot ${connected ? "ok" : "err"}`} />EEG</span>
          <span className="link"><span className={`dot ${status?.decoder ? "ok" : status?.errors.decoder ? "err" : "wait"}`} />DECODER</span>
          <span className="link"><span className={`dot ${status?.llm ? "ok" : status?.errors.llm ? "err" : "wait"}`} />QWEN</span>
          <span className="clock num">T+{mmss(clock)}</span>
        </div>
      </header>

      <main className="main">
        <aside className="col">
          <div className="section">
            <div className="section-head">
              <span className="label">Electrode array</span>
              <span className="label num">{eeg?.channels ?? 105} CH · {eeg?.fs ?? 250} HZ</span>
            </div>
            <ElectrodeArray montage={montage} metrics={metrics} reading={phase === "reading"} />
            <div className="array-readout">
              <div className="kv"><span className="label">Montage</span><span className="v mono" style={{ fontSize: 11 }}>HCGSN-128 / 105</span></div>
              <div className="kv"><span className="label">Epoch</span><span className="v num">{decode ? `${decode.seconds.toFixed(2)} s` : "—"}</span></div>
            </div>
          </div>
          <div className="section">
            <div className="section-head"><span className="label">Relative band power</span></div>
            <div className="bands">
              {BANDS.map((b) => (
                <div className="band" key={b}>
                  <div className="band-bar" style={{ height: `${Math.max(3, (metrics?.bands[b] ?? 0) * 140)}%` }} />
                  <span className="band-name">{b.slice(0, 3).toUpperCase()}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="section">
            <div className="section-head"><span className="label">Signal source</span></div>
            <div className="seg">
              {[["replay", "REPLAY"], ["synthetic", "SYNTH"], ["lsl", "LSL"]].map(([k, n]) => (
                <button key={k} className={eeg?.kind === k ? "on" : ""} onClick={() => setSource(k)}>{n}</button>
              ))}
            </div>
            {eeg?.kind === "synthetic" && (
              <p className="notice">Synthetic signal. Decoded units are not brain-derived; use this only to test the pipeline.</p>
            )}
            {eeg?.kind === "replay" && (
              <p className="notice">Real held-out ZuCo EEG, but recorded on other sentences. Units reflect that recording, not what you read here.</p>
            )}
            {eeg?.kind === "lsl" && <p className="notice real">Live headset via Lab Streaming Layer. {eeg.label}</p>}
          </div>
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
                      <button className="btn primary" onClick={() => { setT0(null); setClock(0); go("idle"); }}>Ask another</button>
                    </div>
                  )}
                  {phase === "ready" && (
                    <div className="actions">
                      <button className="btn primary" onClick={generate}>Next sentence</button>
                      <button className="btn ghost" onClick={() => { setT0(null); setClock(0); go("idle"); }}>New session</button>
                    </div>
                  )}
                  {(phase === "reading" || phase === "generating") && (
                    <button className="btn ghost" style={{ marginLeft: "auto" }} onClick={stop}>Stop</button>
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
            onSetting={onSetting} onSteer={setSteer} onAuto={setAuto} />
        </aside>
      </main>

      <Raster traces={traces} labels={montage?.display_labels ?? []} rate={rate} source={eeg?.label} />
    </div>
  );
}

function Boot({ status, error, hidden }: { status: Status | null; error: string | null; hidden: boolean }) {
  const line = (name: string, ok: boolean, err?: string) => (
    <div>
      <span>{name.padEnd(30, ".")}</span>{" "}
      {err ? <span className="err">FAULT</span> : ok ? <span className="ok">ONLINE</span> : <span>LOADING</span>}
    </div>
  );
  return (
    <div className={`boot ${hidden ? "gone" : ""}`}>
      <Logo size={40} />
      <span className="label" style={{ letterSpacing: "0.4em" }}>NEUROSTEER · INITIALISING</span>
      <div className="boot-lines">
        {line("FLASK LINK", !!status, error ?? undefined)}
        {line("EEG ACQUISITION", !!status?.eeg.kind)}
        {line("RAG-MOSAIC DECODER", !!status?.decoder, status?.errors.decoder)}
        {line(`QWEN · ${status?.llm?.name ?? "Qwen3-1.7B"}`.toUpperCase(), !!status?.llm, status?.errors.llm)}
        {(error || status?.errors.llm || status?.errors.decoder) && (
          <div className="err" style={{ marginTop: 10, maxWidth: 420 }}>{error ?? status?.errors.llm ?? status?.errors.decoder}</div>
        )}
      </div>
    </div>
  );
}
