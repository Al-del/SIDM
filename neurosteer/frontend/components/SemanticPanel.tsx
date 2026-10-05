"use client";

import { PRESETS } from "@/lib/api";
import type { Decode, Plan, PresetName, Presets, Settings } from "@/lib/api";
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
  presets?: Presets | null;
  onPreset?: (name: PresetName) => void;
};

function activePreset(presets: Presets, s: Settings): PresetName | null {
  for (const name of PRESETS) {
    const p = presets[name];
    if (!p) continue;
    const keys = Object.keys(p) as (keyof Settings)[];
    if (keys.length && keys.every((k) => {
      const a = p[k], b = s[k];
      return typeof a === "number" && typeof b === "number" ? Math.abs(a - b) < 1e-6 : a === b;
    })) return name;
  }
  return null;
}

function cellColor(v: number) {
  const a = Math.min(1, Math.abs(v)) * 0.95;
  return v >= 0 ? `rgba(255,176,74,${a})` : `rgba(127,227,255,${a})`;
}

const SLIDERS: { key: keyof Settings; name: string; min: number; max: number; step: number; hint: string }[] = [
  { key: "topic", name: "Topic match", min: 0, max: 3, step: 0.1, hint: "re-rank by question" },
  { key: "prefix", name: "Neural prefix", min: 0, max: 3, step: 0.1, hint: "post-embedding" },
  { key: "prefix_tokens", name: "Slot tokens", min: 0, max: 12, step: 1, hint: "translated slots" },
  { key: "residual", name: "Residual Δh", min: 0, max: 1.2, step: 0.05, hint: "layer activation" },
  { key: "bias", name: "Logit bias", min: 0, max: 8, step: 0.25, hint: "token boost" },
  { key: "temperature", name: "Temperature", min: 0.2, max: 1.4, step: 0.05, hint: "sampling" },
  { key: "min_sentences", name: "Min sentences", min: 1, max: 12, step: 1, hint: "never ends earlier" },
  { key: "max_sentences", name: "Max sentences", min: 2, max: 16, step: 1, hint: "answer length" },
];

function Toggle({ name, hint, on, onChange, first, last }: {
  name: string; hint: string; on: boolean; onChange: (v: boolean) => void; first?: boolean; last?: boolean;
}) {
  return (
    <button type="button" role="switch" aria-checked={on} className="toggle" title={hint} onClick={() => onChange(!on)}
      style={{ ...(first ? { borderTop: 0 } : {}), ...(last ? { marginBottom: 16 } : {}) }}>
      <span className="label" style={{ color: "var(--text)" }}>{name}</span>
      <span className={`switch ${on ? "on" : ""}`} aria-hidden />
    </button>
  );
}

export default function SemanticPanel({ readout, decode, plan, settings, steer, auto, onSetting, onSteer, onAuto, presets, onPreset }: Props) {
  const maxW = decode?.units[0]?.weight || 1;
  const preset = presets && settings ? activePreset(presets, settings) : null;
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
        {decode?.topic && (
          <p className="empty" style={{ marginTop: 10 }}>
            Matched to the question&apos;s topic (×{decode.topic.weight.toFixed(1)}): <span className="mono">{decode.topic.nearest.join(" · ")}</span>
          </p>
        )}
      </div>

      <div className="section">
        <div className="section-head"><span className="label">Qwen reads from ẑ</span></div>
        {readout ? <p className="readout">“{readout}”</p>
          : <p className="empty">The translated sentence vector, decoded back to text by frozen Qwen.</p>}
      </div>

      <div className="section">
        <div className="section-head">
          <span className="label">Injection into Qwen</span>
          {plan && <span className="label num" style={{ color: "var(--inject)" }}>{plan.mode === "translator" ? "256 → 2048 · " : plan.mode === "mock" ? "SIMULATED · " : "LOOKUP · "}L{plan.layer}</span>}
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
        <div className="section-head">
          <span className="label">Steering</span>
          {presets && <span className="label" style={{ color: preset ? "var(--inject)" : "var(--faint)" }}>{preset ?? "custom"}</span>}
        </div>
        {presets && onPreset && (
          <div className="seg presets" role="radiogroup" aria-label="Steering preset">
            {PRESETS.filter((n) => presets[n]).map((n) => (
              <button key={n} role="radio" aria-checked={preset === n} className={preset === n ? "on" : ""}
                onClick={() => onPreset(n)} title={Object.entries(presets[n]).map(([k, v]) => `${k} ${v}`).join(" · ")}>
                {n.toUpperCase()}
              </button>
            ))}
          </div>
        )}
        <Toggle name="Neural steering" on={steer} onChange={onSteer} first
          hint="Off: the next sentences are generated without any EEG-derived input" />
        <Toggle name="Closed loop" on={auto} onChange={onAuto}
          hint="On: generate the next sentence automatically after each decode" />
        <Toggle name="Text hint in prompt" on={!!settings?.hint} onChange={(v) => settings && onSetting("hint", v)} last
          hint="Also list the decoded words in the prompt text: strongest, but least neural" />
        {settings && SLIDERS.map((s) => (
          <div className="ctrl" key={s.key} style={{ opacity: steer || s.key === "temperature" || s.key === "max_sentences" ? 1 : 0.35 }}>
            <span className="label"><span style={{ color: "var(--text)" }}>{s.name}</span><span>{s.hint}</span></span>
            <input type="range" aria-label={s.name} min={s.min} max={s.max} step={s.step} value={settings[s.key] as number}
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
