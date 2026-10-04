"use client";

import { useEffect, useRef } from "react";
import type { MutableRefObject } from "react";
import type { Traces } from "@/lib/useEEG";
import { WINDOW_SECONDS } from "@/lib/useEEG";

type Props = {
  traces: MutableRefObject<Traces | null>;
  labels: string[];
  rate: number;
  source?: string;
};

export default function Raster({ traces, labels, rate, source }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = canvas.current;
    if (!cv) return;
    const ctx = cv.getContext("2d")!;
    const scale: number[] = [];
    let raf = 0;

    const draw = () => {
      raf = requestAnimationFrame(draw);
      const dpr = window.devicePixelRatio || 1;
      const W = cv.clientWidth, H = cv.clientHeight;
      if (cv.width !== W * dpr || cv.height !== H * dpr) {
        cv.width = W * dpr;
        cv.height = H * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      const tr = traces.current;

      ctx.strokeStyle = "rgba(205,225,245,0.06)";
      ctx.lineWidth = 1;
      for (let s = 0; s <= WINDOW_SECONDS; s++) {
        const x = Math.round((s / WINDOW_SECONDS) * W) + 0.5;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
      }
      if (!tr) return;
      const { size, head, data, reading } = tr;
      const C = data.length;
      const row = H / C;
      const dx = W / size;

      ctx.fillStyle = "rgba(255,176,74,0.07)";
      let runStart = -1;
      for (let i = 0; i <= size; i++) {
        const on = i < size && reading[(head + i) % size] === 1;
        if (on && runStart < 0) runStart = i;
        if (!on && runStart >= 0) {
          ctx.fillRect(runStart * dx, 0, (i - runStart) * dx, H);
          ctx.fillStyle = "rgba(255,176,74,0.55)";
          ctx.fillRect(runStart * dx, 0, 1, H);
          ctx.fillStyle = "rgba(255,176,74,0.07)";
          runStart = -1;
        }
      }

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
        const g = ctx.createLinearGradient(0, 0, W, 0);
        g.addColorStop(0, "rgba(127,227,255,0.05)");
        g.addColorStop(0.75, "rgba(127,227,255,0.55)");
        g.addColorStop(1, "rgba(200,245,255,0.95)");
        ctx.strokeStyle = g;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < size; i++) {
          const val = d[(head + i) % size] - m;
          const y = y0 - Math.max(-row * 0.6, Math.min(row * 0.6, val * k));
          if (i === 0) ctx.moveTo(0, y); else ctx.lineTo(i * dx, y);
        }
        ctx.stroke();
      }

      const sweep = ctx.createLinearGradient(W - 60, 0, W, 0);
      sweep.addColorStop(0, "rgba(127,227,255,0)");
      sweep.addColorStop(1, "rgba(127,227,255,0.12)");
      ctx.fillStyle = sweep;
      ctx.fillRect(W - 60, 0, 60, H);
      ctx.fillStyle = "rgba(200,245,255,0.9)";
      ctx.fillRect(W - 1, 0, 1, H);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [traces]);

  return (
    <div className="raster">
      <div className="raster-labels">
        {labels.map((l) => <span key={l}>{l}</span>)}
      </div>
      <div className="raster-plot"><canvas ref={canvas} /></div>
      <div className="raster-side">
        <div className="kv"><span className="label">Window</span><span className="v num">{WINDOW_SECONDS}.0 s</span></div>
        <div className="kv"><span className="label">Display</span><span className="v num">{labels.length} / 105 ch</span></div>
        <div className="kv"><span className="label">Stream</span><span className="v num">{rate} S/s</span></div>
        <div className="kv"><span className="label">Source</span><span className="v mono" style={{ fontSize: 10, color: "var(--dim)" }}>{source ?? "—"}</span></div>
      </div>
    </div>
  );
}
