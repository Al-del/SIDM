export const STAGES = ["READ", "DECODE", "INJECT", "GENERATE"] as const;
export type Stage = (typeof STAGES)[number];

const SUB: Record<Stage, string> = {
  READ: "EEG epoch",
  DECODE: "RAG-Mosaic",
  INJECT: "prefix · Δh · bias",
  GENERATE: "Qwen3-1.7B",
};
const AMBER: Record<Stage, boolean> = { READ: true, DECODE: false, INJECT: true, GENERATE: false };

export default function Pipeline({ on }: { on: Record<Stage, boolean> }) {
  const active = STAGES.findIndex((s) => on[s]);
  return (
    <nav className={`pipeline ${active >= 0 ? "running" : ""}`} aria-label="Closed-loop pipeline">
      <span className={`loop ${on.READ ? "live" : ""}`} aria-hidden />
      {STAGES.map((s, i) => (
        <div key={s} className="pipe-cell">
          {i > 0 && (
            <span className={`wire ${on[s] ? "live" : ""} ${AMBER[s] ? "amber" : ""}`} aria-hidden>
              <i /><i /><i />
            </span>
          )}
          <span className={`stage ${on[s] ? "on" : ""} ${active > i ? "done" : ""} ${AMBER[s] ? "inj" : ""}`}
            aria-current={on[s] ? "step" : undefined}>
            <span className="pip" />
            <span className="stage-text">
              <span className="stage-name">{s}</span>
              <span className="stage-sub">{SUB[s]}</span>
            </span>
          </span>
        </div>
      ))}
    </nav>
  );
}
