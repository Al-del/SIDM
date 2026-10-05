import type { ReactNode } from "react";
import Logo from "@/components/Logo";
import Pipeline from "@/components/Pipeline";
import type { Stage } from "@/components/Pipeline";
import Clock from "@/components/Clock";
import type { Health, Status } from "@/lib/api";

type Props = {
  stageOn: Record<Stage, boolean>;
  connected: boolean;
  status: Status | null;
  health: Health | null | undefined;
  mock: boolean;
  
  offline: boolean;
  t0: number | null;
  children?: ReactNode;
};

const uptime = (s: number) => (s < 3600 ? `${Math.floor(s / 60)} min` : `${(s / 3600).toFixed(1)} h`);

export default function TopBar({ stageOn, connected, status: live, health, mock, offline, t0, children }: Props) {
  const status = offline ? null : live;
  const llm = status?.llm;
  return (
    <header className="topbar">
      <div className="brand">
        <Logo />
        <span className="brand-name">NEUROSTEER</span>
        <span className="brand-sub mono"
          title={health ? `backend v${health.version} · up ${uptime(health.uptime_s)}` : undefined}>
          EEG → QWEN / CLOSED LOOP{health?.version ? ` · v${health.version}` : ""}
        </span>
        {mock && (
          <span className="demo-badge" role="status"
            title="The backend is running simulated models: tokens and decodes are synthetic, for showing the interface only.">
            <span className="demo-dot" />DEMO MODE<span className="demo-long">&nbsp;· simulated models</span>
          </span>
        )}
      </div>
      <Pipeline on={stageOn} />
      {offline && <span className="offline-badge" role="alert">NO BACKEND LINK</span>}
      <div className="links">
        <span className="link" title={connected ? "EEG stream connected" : "EEG stream offline"}>
          <span className={`dot ${connected ? "ok" : "err"}`} /><span className="link-name">EEG</span>
        </span>
        <span className="link" title={status?.errors.decoder ?? (status?.decoder ? "RAG-Mosaic decoder loaded" : "Decoder loading")}>
          <span className={`dot ${offline ? "err" : status?.decoder ? "ok" : status?.errors.decoder ? "err" : "wait"}`} /><span className="link-name">DECODER</span>
        </span>
        <span className="link"
          title={status?.errors.llm ?? (llm ? `${llm.name} · ${llm.layers} layers · steer L${llm.steer_layer} · ${llm.device}${llm.mock ? " · simulated" : ""}` : "LLM loading")}>
          <span className={`dot ${offline ? "err" : llm ? (llm.mock ? "mock" : "ok") : status?.errors.llm ? "err" : "wait"}`} /><span className="link-name">QWEN</span>
        </span>
        <Clock t0={t0} />
        {children}
      </div>
    </header>
  );
}
