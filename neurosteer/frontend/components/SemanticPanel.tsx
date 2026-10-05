"use client";

import type { Decode, Plan, Settings } from "@/lib/api";
import AlignmentMeter from "@/components/AlignmentMeter";

type Props = {
  readout: string | null;
  decode: Decode | null;
  plan: Plan;
  settings: Settings | null;
  steer: boolean;
  auto: boolean;
  onSetting: (k: keyof Settings, v: number | boolean) => void;
  onSteer: (v: boolean) => void;
  onAuto: (v: boolean) => void;
};

function cellColor(v: number) {
  const a = Math.min(1, Math.abs(v)) * 0.95;
  return v >= 0 ? `rgba(255,176,74,${a})` : `rgba(127,227,255,${a})`;
}

const SLIDERS: { key: keyof Settings; name: string; min: number; max: number; step: number; hint: string }[] = [
  { key: "prefix", name: "Neural prefix", min: 0, max: 3, step: 0.1, hint: "post-embedding" },
  { key: "prefix_tokens", name: "Slot tokens", min: 0, max: 12, step: 1, hint: "translated slots" },
  { key: "residual", name: "Residual Δh", min: 0, max: 1.2, step: 0.05, hint: "layer activation" },
  { key: "bias", name: "Logit bias", min: 0, max: 8, step: 0.25, hint: "token boost" },
  { key: "temperature", name: "Temperature", min: 0.2, max: 1.4, step: 0.05, hint: "sampling" },
  { key: "max_sentences", name: "Max sentences", min: 2, max: 16, step: 1, hint: "answer length" },
];

export default function SemanticPanel({ readout, decode, plan, settings, steer, auto, onSetting, onSteer, onAuto }: Props) {
  const maxW = decode?.units[0]?.weight || 1;
  return (
    <>
      <div className="section">
        <div className="section-head">
          <span className="label">Decoded semantic units</span>
          {decode && <span className="label num">{decode.seconds.toFixed(1)}s · {decode.latency_ms}ms</span>}
        </div>
        {typeof decode?.alignment === "number" && <div style={{ marginBottom: 12 }}><AlignmentMeter value={decode.alignment} /></div>}
        {!decode && <p className="empty">No epoch yet. Units appear after the first sentence has been read.</p>}
        {decode?.units.map((u, i) => (
          <div className="unit" key={`${u.unit}-${i}`} style={{ animationDelay: `${i * 35}ms` }}>
            <span className="unit-word">
              {u.word}
              {u.words.length > 1 && <span className="unit-alt mono">{u.words.slice(1, 3).join(" ")}</span>}
            </span>
            <span className="unit-bar"><span style={{ width: `${(u.weight / maxW) * 100}%` }} /></span>
            <span className="unit-w num">{u.weight.toFixed(2)}</span>
          </div>
        ))}
      </div>

      <div className="section">
        <div className="section-head"><span className="label">Qwen reads from ẑ</span></div>
        {readout ? <p className="readout">“{readout}”</p>
          : <p className="empty">The translated sentence vector, decoded back to text by frozen Qwen.</p>}
      </div>

      <div className="section">
        <div className="section-head">
          <span className="label">Injection into Qwen</span>
          {plan && <span className="label num" style={{ color: "var(--inject)" }}>{plan.mode === "translator" ? "256 → 2048 · " : "LOOKUP · "}L{plan.layer}</span>}
        </div>
        {!plan && <p className="empty">Waiting for a decoded epoch to steer the next sentence.</p>}
        {plan?.prefix && (
          <div className="strips">
            {plan.prefix.map((row, i) => (
              <div className="strip" key={i}>
                <span className={`strip-name ${plan.labels[i]?.startsWith("z") ? "z" : ""}`}>{plan.labels[i] === undefined ? "" : plan.labels[i].startsWith("z") ? `ẑ·${plan.labels[i].slice(1)}` : plan.labels[i]}</span>
                <div className="cells">
                  {row.map((v, j) => (
                    <span key={j} className="cell" style={{ background: cellColor(v), animationDelay: `${i * 40 + j * 6}ms` }} />
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
        {plan && (
          <div className="inj-stats">
            <div className="kv"><span className="label">Prefix</span><span className="v num">{plan.prefix ? plan.prefix.length : 0} tok</span></div>
            <div className="kv"><span className="label">‖Δh‖</span><span className="v num">{plan.residual_norm}</span></div>
            <div className="kv"><span className="label">Bias</span><span className="v num">{plan.bias_tokens} ids</span></div>
          </div>
        )}
      </div>

      <div className="section">
        <div className="section-head"><span className="label">Steering</span></div>
        <div className="toggle" onClick={() => onSteer(!steer)} style={{ borderTop: 0 }}>
          <span className="label" style={{ color: "var(--text)" }}>Neural steering</span>
          <span className={`switch ${steer ? "on" : ""}`} />
        </div>
        <div className="toggle" onClick={() => onAuto(!auto)}>
          <span className="label" style={{ color: "var(--text)" }}>Closed loop</span>
          <span className={`switch ${auto ? "on" : ""}`} />
        </div>
        <div className="toggle" onClick={() => settings && onSetting("hint", !settings.hint)} style={{ marginBottom: 16 }}>
          <span className="label" style={{ color: "var(--text)" }}>Text hint in prompt</span>
          <span className={`switch ${settings?.hint ? "on" : ""}`} />
        </div>
        {settings && SLIDERS.map((s) => (
          <div className="ctrl" key={s.key} style={{ opacity: steer || s.key === "temperature" || s.key === "max_sentences" ? 1 : 0.35 }}>
            <span className="label"><span style={{ color: "var(--text)" }}>{s.name}</span><span>{s.hint}</span></span>
            <input type="range" min={s.min} max={s.max} step={s.step} value={settings[s.key] as number}
              onChange={(e) => onSetting(s.key, Number(e.target.value))} />
            <output className="num">{Number(settings[s.key]).toFixed(s.step < 1 ? 2 : 0)}</output>
          </div>
        ))}
      </div>

      <div className="section">
        <div className="section-head"><span className="label">Nearest corpus sentences (ẑ)</span></div>
        {!decode && <p className="empty">—</p>}
        {decode?.neighbours.map((n, i) => (
          <div className="nb" key={i}><span>{n.sentence}</span><span className="num" style={{ textAlign: "right" }}>{n.sim.toFixed(2)}</span></div>
        ))}
      </div>
    </>
  );
}
