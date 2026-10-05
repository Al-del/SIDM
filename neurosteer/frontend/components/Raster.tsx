"use client";

import { memo, useEffect, useRef } from "react";
import type { MutableRefObject } from "react";
import type { Traces } from "@/lib/useEEG";
import { WINDOW_SECONDS } from "@/lib/useEEG";

type Props = {
  traces: MutableRefObject<Traces | null>;
  labels: string[];
  rate: number;
  source?: string;
};

const MONO = "500 9px 'JetBrains Mono', ui-monospace, monospace";

function Raster({ traces, labels, rate, source }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = canvas.current;
    if (!cv) return;
    const ctx = cv.getContext("2d")!;
    const scale: number[] = [];
    let raf = 0;
    let lastTotal = -1, lastW = 0, lastH = 0;
    let trace: CanvasGradient | null = null, head: CanvasGradient | null = null, sweep: CanvasGradient | null = null;

    const draw = () => {
      raf = requestAnimationFrame(draw);
      const dpr = window.devicePixelRatio || 1;
      const W = cv.clientWidth, H = cv.clientHeight;
      const tr = traces.current;

      if (tr && tr.total === lastTotal && W === lastW && H === lastH) return;
      if (cv.width !== W * dpr || cv.height !== H * dpr) {
        cv.width = W * dpr;
        cv.height = H * dpr;
      }
      if (W !== lastW || !trace) {
        trace = ctx.createLinearGradient(0, 0, W, 0);
        trace.addColorStop(0, "rgba(127,227,255,0.04)");
        trace.addColorStop(0.7, "rgba(127,227,255,0.5)");
        trace.addColorStop(1, "rgba(200,245,255,0.95)");
        head = ctx.createLinearGradient(W - 40, 0, W, 0);
        head.addColorStop(0, "rgba(220,250,255,0)");
        head.addColorStop(1, "rgba(235,252,255,1)");
        sweep = ctx.createLinearGradient(W - 80, 0, W, 0);
        sweep.addColorStop(0, "rgba(127,227,255,0)");
        sweep.addColorStop(1, "rgba(127,227,255,0.14)");
      }
      lastTotal = tr ? tr.total : -1; lastW = W; lastH = H;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      ctx.strokeStyle = "rgba(205,225,245,0.06)";
      ctx.lineWidth = 1;
      ctx.font = MONO;
      ctx.fillStyle = "rgba(138,150,162,0.55)";
      for (let s = 0; s <= WINDOW_SECONDS; s++) {
        const x = Math.round((s / WINDOW_SECONDS) * W) + 0.5;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
        if (s < WINDOW_SECONDS && s > 0) ctx.fillText(`-${WINDOW_SECONDS - s}s`, x + 4, H - 5);
      }
      if (!tr) return;
      const { size, head: h0, data, reading } = tr;
      const C = data.length;
      const row = H / C;
      const dx = W / size;

      ctx.fillStyle = "rgba(205,225,245,0.018)";
      for (let c = 0; c < C; c += 2) ctx.fillRect(0, row * c, W, row);

      let runStart = -1;
      for (let i = 0; i <= size; i++) {
        const on = i < size && reading[(h0 + i) % size] === 1;
        if (on && runStart < 0) runStart = i;
        if (!on && runStart >= 0) {
          const x0 = runStart * dx, w = (i - runStart) * dx;
          ctx.fillStyle = "rgba(255,176,74,0.08)";
          ctx.fillRect(x0, 0, w, H);
          ctx.fillStyle = "rgba(255,176,74,0.6)";
          ctx.fillRect(x0, 0, 1, H);
          if (w > 60) { ctx.fillStyle = "rgba(255,176,74,0.85)"; ctx.fillText("EPOCH", x0 + 5, 11); }
          runStart = -1;
        }
      }

      ctx.lineWidth = 1;
      for (let c = 0; c < C; c++) {
        const d = data[c];
        let m = 0, v = 0;
        for (let i = 0; i < size; i += 4) m += d[i];
        m /= size / 4;
        for (let i = 0; i < size; i += 4) v += (d[i] - m) ** 2;
        const sd = Math.sqrt(v / (size / 4)) || 1;
        scale[c] = scale[c] ? scale[c] * 0.97 + sd * 0.03 : sd;
        const y0 = row * (c + 0.5);
        const k = (row * 0.42) / (scale[c] * 2.2);
        const lim = row * 0.6;
        ctx.strokeStyle = trace!;
        ctx.beginPath();
        for (let i = 0; i < size; i++) {
          const y = y0 - Math.max(-lim, Math.min(lim, (d[(h0 + i) % size] - m) * k));
          if (i === 0) ctx.moveTo(0, y); else ctx.lineTo(i * dx, y);
        }
        ctx.stroke();

        const from = Math.max(0, Math.floor(size - 40 / dx));
        ctx.strokeStyle = head!;
        ctx.beginPath();
        for (let i = from; i < size; i++) {
          const y = y0 - Math.max(-lim, Math.min(lim, (d[(h0 + i) % size] - m) * k));
          if (i === from) ctx.moveTo(i * dx, y); else ctx.lineTo(i * dx, y);
        }
        ctx.stroke();
      }

      ctx.fillStyle = sweep!;
      ctx.fillRect(W - 80, 0, 80, H);
      ctx.fillStyle = "rgba(200,245,255,0.9)";
      ctx.fillRect(W - 1, 0, 1, H);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [traces]);

  return (
    <div className="raster" role="img" aria-label={`Live EEG, ${labels.length} channels over the last ${WINDOW_SECONDS} seconds`}>
      <div className="raster-labels" aria-hidden>
        {labels.map((l) => <span key={l}>{l}</span>)}
      </div>
      <div className="raster-plot"><canvas ref={canvas} /></div>
      <div className="raster-side">
        <div className="kv"><span className="label">Window</span><span className="v num">{WINDOW_SECONDS}.0 s</span></div>
        <div className="kv"><span className="label">Display</span><span className="v num">{labels.length} / 105 ch</span></div>
        <div className="kv">
          <span className="label">Stream</span>
          <span className="v num"><span className={`live-dot ${rate > 0 ? "on" : ""}`} />{rate} S/s</span>
        </div>
        <div className="kv"><span className="label">Source</span><span className="v mono" style={{ fontSize: 10, color: "var(--dim)" }}>{source ?? "—"}</span></div>
      </div>
    </div>
  );
}

export default memo(Raster);
