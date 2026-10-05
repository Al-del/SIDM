import Logo from "@/components/Logo";
import Pipeline from "@/components/Pipeline";
import type { Stage } from "@/components/Pipeline";
import Clock from "@/components/Clock";
import type { Status } from "@/lib/api";

type Props = {
  stageOn: Record<Stage, boolean>;
  connected: boolean;
  status: Status | null;
  t0: number | null;
};

export default function TopBar({ stageOn, connected, status, t0 }: Props) {
  return (
    <header className="topbar">
      <div className="brand">
        <Logo />
        <span className="brand-name">NEUROSTEER</span>
        <span className="brand-sub mono">EEG → QWEN / CLOSED LOOP</span>
      </div>
      <Pipeline on={stageOn} />
      <div className="links">
        <span className="link"><span className={`dot ${connected ? "ok" : "err"}`} />EEG</span>
        <span className="link"><span className={`dot ${status?.decoder ? "ok" : status?.errors.decoder ? "err" : "wait"}`} />DECODER</span>
        <span className="link"><span className={`dot ${status?.llm ? "ok" : status?.errors.llm ? "err" : "wait"}`} />QWEN</span>
        <Clock t0={t0} />
      </div>
    </header>
  );
}
