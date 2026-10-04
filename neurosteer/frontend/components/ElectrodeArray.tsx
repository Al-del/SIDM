"use client";

import { useMemo } from "react";
import type { Montage } from "@/lib/api";
import type { Metrics } from "@/lib/useEEG";

const C = 150;
const HEAD = 104;

export default function ElectrodeArray({ montage, metrics, reading }: {
  montage: Montage | null; metrics: Metrics | null; reading: boolean;
}) {
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

  const level = useMemo(() => {
    if (!metrics) return null;
    const s = [...metrics.rms].sort((a, b) => a - b);
    const p95 = s[Math.floor(s.length * 0.95)] || 1;
    return metrics.rms.map((v) => Math.min(1, v / p95));
  }, [metrics]);

  const accent = reading ? "var(--inject)" : "var(--signal)";

  return (
    <div className="array-wrap">
      <svg viewBox="0 0 300 300">
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
        <circle cx={C} cy={C} r={HEAD} fill="rgba(127,227,255,0.015)" stroke="rgba(205,225,245,0.16)" />
        <path d={`M ${C - 8} ${C - HEAD + 1} L ${C} ${C - HEAD - 9} L ${C + 8} ${C - HEAD + 1}`}
          fill="none" stroke="rgba(205,225,245,0.3)" />
        <line x1={C - HEAD} y1={C} x2={C + HEAD} y2={C} stroke="rgba(205,225,245,0.05)" />
        <line x1={C} y1={C - HEAD} x2={C} y2={C + HEAD} stroke="rgba(205,225,245,0.05)" />
        {geo && geo.pts.map(([x, y], i) => {
          const v = level ? level[i] : 0;
          return (
            <g key={i}>
              <circle className="elec" cx={x} cy={y} r={2 + v * 5} fill={accent} opacity={v * 0.22} />
              <rect x={x - 1.3} y={y - 1.3} width={2.6} height={2.6}
                fill={v > 0.6 ? accent : "rgba(228,235,242,0.75)"} opacity={0.35 + v * 0.65}
                style={{ transition: "opacity 260ms linear" }}>
                <title>{montage?.labels[i]}</title>
              </rect>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
