"use client";

import { useState } from "react";
import Logo from "@/components/Logo";
import { API } from "@/lib/api";
import type { Status } from "@/lib/api";

type Props = {
  status: Status | null;
  mock: boolean;
  unreachable: boolean;
  hidden: boolean;
  onHelp: () => void;
};

export default function Boot({ status, mock, unreachable, hidden, onHelp }: Props) {
  const [dismissed, setDismissed] = useState(false);
  const fault = !status && unreachable;
  const modelError = status?.errors.llm ?? status?.errors.decoder;

  const line = (name: string, ok: boolean, err?: boolean) => (
    <div>
      <span>{name.padEnd(30, ".")}</span>{" "}
      {err ? <span className="err">FAULT</span> : ok ? <span className="ok">ONLINE</span> : fault ? <span>STANDBY</span> : <span className="wait">LOADING</span>}
    </div>
  );

  return (
    <div className={`boot ${hidden || dismissed ? "gone" : ""}`} role="dialog" aria-label="Starting up" aria-hidden={hidden || dismissed}>
      <Logo size={40} />
      <span className="label" style={{ letterSpacing: "0.4em" }}>NEUROSTEER · {fault ? "NO LINK" : "INITIALISING"}</span>
      <div className={`boot-bar ${fault ? "fault" : ""}`} aria-hidden><span /></div>
      <div className="boot-lines" aria-live="polite">
        {line("FLASK LINK", !!status, fault)}
        {line("EEG ACQUISITION", !!status?.eeg.kind)}
        {line("RAG-MOSAIC DECODER", !!status?.decoder, !!status?.errors.decoder)}
        {line(`QWEN · ${status?.llm?.name ?? "Qwen3-1.7B"}`.toUpperCase(), !!status?.llm, !!status?.errors.llm)}
        {mock && <div className="warn">DEMO MODE · models are simulated</div>}
        {fault && (
          <div className="boot-help">
            <p className="err">Backend unreachable at {API}. Retrying every 2 s.</p>
            <p>Start it with <code>./start.sh</code> (Flask on :5050), or point the UI elsewhere with <code>NEXT_PUBLIC_API</code>.</p>
          </div>
        )}
        {modelError && <div className="err" style={{ marginTop: 10, maxWidth: 420 }}>{modelError}</div>}
      </div>
      {(fault || modelError) && (
        <div className="actions">
          <button className="btn" onClick={onHelp}>How it works</button>
          <button className="btn ghost" onClick={() => setDismissed(true)}>Look around without it</button>
        </div>
      )}
    </div>
  );
}
