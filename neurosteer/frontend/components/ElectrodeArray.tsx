"use client";

import { memo, useEffect, useMemo, useRef } from "react";
import type { Montage } from "@/lib/api";
import type { Metrics } from "@/lib/useEEG";

const C = 150;
const HEAD = 104;

const GRID = 72;
const K = 6;

type Props = { montage: Montage | null; metrics: Metrics | null; reading: boolean; sensors: number[] };

function ramp(v: number, reading: boolean): [number, number, number, number] {
  const t = Math.min(1, Math.max(0, reading ? v * 0.8 + 0.2 : v));
  const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
  let r, g, b;
  if (t < 0.6) {
    const u = t / 0.6;
    r = lerp(12, 127, u); g = lerp(40, 227, u); b = lerp(70, 255, u);
  } else {
    const u = (t - 0.6) / 0.4;
    r = lerp(127, 255, u); g = lerp(227, 176, u); b = lerp(255, 74, u);
  }
  return [r, g, b, Math.round(255 * (0.08 + 0.5 * t * t))];
}

function ElectrodeArray({ montage, metrics, reading, sensors }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);

  const geo = useMemo(() => {
    if (!montage) return null;
    const maxR = Math.max(...montage.pos.map(([x, y]) => Math.hypot(x, y)));
    const k = (HEAD + 8) / maxR;
    const pts = montage.pos.map(([x, y]) => [C + x * k, C - y * k] as const);
    const order = montage.pos
      .map(([x, y], i) => ({ i, a: Math.atan2(x, y) }))
      .sort((p, q) => p.a - q.a)
      .map((p) => p.i);
    return { pts, order };
  }, [montage]);

  const idw = useMemo(() => {
    if (!geo) return null;
    const idx = new Int32Array(GRID * GRID * K);
    const w = new Float32Array(GRID * GRID * K);
    const inside = new Uint8Array(GRID * GRID);
    const scale = (2 * HEAD) / GRID;
    const d = new Float32Array(geo.pts.length);
    for (let py = 0; py < GRID; py++) {
      for (let px = 0; px < GRID; px++) {
        const x = C - HEAD + (px + 0.5) * scale, y = C - HEAD + (py + 0.5) * scale;
        const p = py * GRID + px;
        if (Math.hypot(x - C, y - C) > HEAD) continue;
        inside[p] = 1;
        for (let e = 0; e < geo.pts.length; e++) d[e] = (geo.pts[e][0] - x) ** 2 + (geo.pts[e][1] - y) ** 2;
        const near = Array.from(d.keys()).sort((a, b) => d[a] - d[b]).slice(0, K);
        let sum = 0;
        near.forEach((e, j) => { const ww = 1 / (d[e] + 30); idx[p * K + j] = e; w[p * K + j] = ww; sum += ww; });
        for (let j = 0; j < K; j++) w[p * K + j] /= sum;
      }
    }
    return { idx, w, inside };
  }, [geo]);

  const level = useMemo(() => {
    if (!metrics) return null;
    const s = [...metrics.rms].sort((a, b) => a - b);
    const p95 = s[Math.floor(s.length * 0.95)] || 1;
    return metrics.rms.map((v) => Math.min(1, v / p95));
  }, [metrics]);

  useEffect(() => {
    const cv = canvas.current;
    if (!cv || !idw) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(GRID, GRID);
    for (let p = 0; p < GRID * GRID; p++) {
      if (!idw.inside[p]) continue;
      let v = 0;
      if (level) for (let j = 0; j < K; j++) v += (level[idw.idx[p * K + j]] ?? 0) * idw.w[p * K + j];
      const [r, g, b, a] = ramp(v, reading);
      img.data[p * 4] = r; img.data[p * 4 + 1] = g; img.data[p * 4 + 2] = b; img.data[p * 4 + 3] = level ? a : 10;
    }
    ctx.putImageData(img, 0, 0);
  }, [idw, level, reading]);

  const accent = reading ? "var(--inject)" : "var(--signal)";
  const pad = ((C - HEAD) / 300) * 100;

  return (
    <div className={`array-wrap ${reading ? "reading" : ""}`} role="img"
      aria-label={`Scalp map of ${montage?.labels.length ?? 0} electrodes, coloured by signal amplitude${reading ? ", recording" : ""}`}>
      <canvas ref={canvas} width={GRID} height={GRID} className="topo"
        style={{ left: `${pad}%`, top: `${pad}%`, width: `${100 - 2 * pad}%`, height: `${100 - 2 * pad}%` }} />
      <svg viewBox="0 0 300 300" aria-hidden>
        <defs>
          <filter id="elec-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="2.4" result="b" />
            <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>
        <g className="spin">
          <circle cx={C} cy={C} r={142} fill="none" stroke="rgba(205,225,245,0.18)" strokeDasharray="1 5" />
          {[0, 90, 180, 270].map((a) => (
            <line key={a} x1={C} y1={C - 146} x2={C} y2={C - 136} stroke="rgba(205,225,245,0.5)"
              transform={`rotate(${a} ${C} ${C})`} />
          ))}
        </g>
        <g className="spin-rev">
          <circle cx={C} cy={C} r={134} fill="none" stroke="rgba(205,225,245,0.07)" strokeDasharray="22 6 2 6" />
        </g>
        {geo && level && geo.order.map((ch, j) => {
          const a = (j / geo.order.length) * Math.PI * 2 - Math.PI / 2;
          const r0 = 118, r1 = r0 + 3 + level[ch] * 12;
          return (
            <line key={ch} x1={C + Math.cos(a) * r0} y1={C + Math.sin(a) * r0}
              x2={C + Math.cos(a) * r1} y2={C + Math.sin(a) * r1}
              stroke={accent} strokeOpacity={0.25 + level[ch] * 0.6} strokeWidth={1}
              style={{ transition: "all 260ms linear" }} />
          );
        })}
        <circle cx={C} cy={C} r={HEAD} fill="none" stroke="rgba(205,225,245,0.2)" />
        <path d={`M ${C - 8} ${C - HEAD + 1} L ${C} ${C - HEAD - 9} L ${C + 8} ${C - HEAD + 1}`}
          fill="none" stroke="rgba(205,225,245,0.3)" />
        <line x1={C - HEAD} y1={C} x2={C + HEAD} y2={C} stroke="rgba(205,225,245,0.05)" />
        <line x1={C} y1={C - HEAD} x2={C} y2={C + HEAD} stroke="rgba(205,225,245,0.05)" />
        {geo && sensors.map((i, j) => {
          const [x, y] = geo.pts[i];
          return (
            <g key={`s${j}`}>
              <circle cx={x} cy={y} r={7} fill="none" stroke="var(--inject)" strokeWidth={1} className="sensor" />
              <text x={x} y={y - 10} textAnchor="middle" className="sensor-label">{montage?.display_labels[j]}</text>
            </g>
          );
        })}
        {geo && geo.pts.map(([x, y], i) => {
          const v = level ? level[i] : 0;
          return (
            <rect key={i} x={x - 1.3} y={y - 1.3} width={2.6} height={2.6}
              fill={v > 0.6 ? accent : "rgba(228,235,242,0.75)"} opacity={0.3 + v * 0.6}
              style={{ transition: "opacity 260ms linear" }}>
              <title>{montage?.labels[i]}</title>
            </rect>
          );
        })}
        {}
        <g filter="url(#elec-glow)">
          {geo && level && geo.pts.map(([x, y], i) => level[i] > 0.7 && (
            <circle key={i} cx={x} cy={y} r={1.6 + (level[i] - 0.7) * 6} fill={accent} opacity={0.9} />
          ))}
        </g>
      </svg>
    </div>
  );
}

export default memo(ElectrodeArray);
