import type { Token } from "@/lib/api";
import { MAX_READ } from "@/lib/useSession";
import type { Phase } from "@/lib/useSession";

type Props = {
  phase: Phase;
  asked: string;
  tokens: Token[];
  count: number;
  injecting: boolean;
  readT: number;
  translator: boolean;
  question: string;
  ready: boolean;
  onQuestion: (q: string) => void;
  onStart: () => void;
  onNext: () => void;
  onNewSession: () => void;
  onStop: () => void;
};

export default function Reader(p: Props) {
  const { phase, tokens, count } = p;
  return (
    <div className="stagebox">
      <span className="corner tl" /><span className="corner tr" /><span className="corner bl" /><span className="corner br" />
      <div className="stage-meta">
        <span className="label num">{count ? `SENTENCE ${String(count + (phase === "generating" ? 1 : 0)).padStart(2, "0")}` : "SESSION"}</span>
        <span className="label">{phase === "idle" ? "standby" : phase}{p.translator ? " · translator" : ""}</span>
      </div>

      {phase === "idle" ? (
        <div className="start">
          <h1>Ask anything.<br /><em>It answers as you read.</em></h1>
          <div className="field">
            <span className="label">Question</span>
            <input value={p.question} onChange={(e) => p.onQuestion(e.target.value)} placeholder="Ask a question"
              onKeyDown={(e) => e.key === "Enter" && p.ready && p.question.trim() && p.onStart()} />
          </div>
          <div className="actions">
            <button className="btn primary" disabled={!p.ready || !p.question.trim()} onClick={p.onStart}>Ask</button>
            <span className="label">One sentence at a time · SPACE when read</span>
          </div>
        </div>
      ) : (
        <>
          <p className="question"><b>Q</b><span>{p.asked}</span></p>
          <div className="sentence">
            {tokens.map((t, i) => (
              <span key={i} className={`tok ${t.steered ? "steer" : ""}`} title={t.steered ? `bias +${t.bias}` : undefined}>{t.text}</span>
            ))}
            {phase === "generating" && <span className="caret" />}
            {phase === "decoding" && <span className="scan" />}
          </div>
          <div className="timer">
            {phase === "reading" && (
              <>
                <div className="timer-track"><div className="timer-fill" style={{ width: `${(p.readT / MAX_READ) * 100}%` }} /></div>
                <span className="label num">{p.readT.toFixed(1)} / {MAX_READ}.0 s</span>
                <span className="key">SPACE</span>
              </>
            )}
            {phase === "decoding" && <span className="label">Decoding epoch through RAG-Mosaic…</span>}
            {phase === "generating" && <span className="label">{p.injecting ? "Injecting neural context…" : "Qwen generating"}</span>}
            {phase === "complete" && (
              <div className="actions">
                <span className="label" style={{ color: "var(--signal)" }}>Answer complete · {count} sentences</span>
                <button className="btn primary" onClick={p.onNewSession}>Ask another</button>
              </div>
            )}
            {phase === "ready" && (
              <div className="actions">
                <button className="btn primary" onClick={p.onNext}>Next sentence</button>
                <button className="btn ghost" onClick={p.onNewSession}>New session</button>
              </div>
            )}
            {(phase === "reading" || phase === "generating") && (
              <button className="btn ghost" style={{ marginLeft: "auto" }} onClick={p.onStop}>Stop</button>
            )}
          </div>
        </>
      )}
    </div>
  );
}
