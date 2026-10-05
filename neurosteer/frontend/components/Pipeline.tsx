export const STAGES = ["READ", "DECODE", "INJECT", "GENERATE"] as const;
export type Stage = (typeof STAGES)[number];

export default function Pipeline({ on }: { on: Record<Stage, boolean> }) {
  return (
    <nav className="pipeline">
      {STAGES.map((s, i) => (
        <div key={s} style={{ display: "flex", alignItems: "center" }}>
          {i > 0 && <span className={`wire ${on[s] ? "live" : ""}`} />}
          <span className={`stage ${on[s] ? "on" : ""} ${s === "INJECT" || s === "READ" ? "inj" : ""}`}>
            <span className="pip" />{s}
          </span>
        </div>
      ))}
    </nav>
  );
}
