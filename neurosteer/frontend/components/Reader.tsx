import type { Token } from "@/lib/api";
import ReadRing from "@/components/ReadRing";
import Tokens from "@/components/Tokens";
import type { Phase } from "@/lib/useSession";

type Props = {
  phase: Phase;
  asked: string;
  tokens: Token[];
  count: number;
  injecting: boolean;
  readStartedAt: number;
  translator: boolean;
  question: string;
  ready: boolean;
  onQuestion: (q: string) => void;
  onStart: () => void;
  onNext: () => void;
  onNewSession: () => void;
  onStop: () => void;
};

const PHASE_SAY: Record<Phase, string> = {
  idle: "",
  generating: "Qwen is writing the next sentence",
  reading: "Read the sentence, then press space",
  decoding: "Decoding the EEG epoch",
  ready: "Paused",
  complete: "Answer complete",
};

const EXAMPLES = [
  "Why do we dream?",
  "How do bees find their way home?",
  "What makes a melody sound sad?",
  "Why is the ocean salty?",
];

export default function Reader(p: Props) {
  const { phase, tokens, count } = p;
  return (
    <section className="stagebox" aria-label="Reader">
      <span className="sr-only" aria-live="polite">{PHASE_SAY[phase]}</span>
      <span className="corner tl" /><span className="corner tr" /><span className="corner bl" /><span className="corner br" />
      <div className="stage-meta">
        <span className="label num">{count ? `SENTENCE ${String(count + (phase === "generating" ? 1 : 0)).padStart(2, "0")}` : "SESSION"}</span>
        <span className="label">{phase === "idle" ? "standby" : phase}{p.translator ? " · translator" : ""}</span>
      </div>

      {phase === "idle" ? (
        <div className="start">
          <h1>Ask anything.<br /><em>It answers as you read.</em></h1>
          <div className="field">
            <label className="label" htmlFor="question">Question</label>
            <input id="question" value={p.question} onChange={(e) => p.onQuestion(e.target.value)} placeholder="Ask a question"
              autoComplete="off" spellCheck={false}
              onKeyDown={(e) => e.key === "Enter" && p.ready && p.question.trim() && p.onStart()} />
          </div>
          <div className="examples" aria-label="Example questions">
            {EXAMPLES.map((q) => (
              <button key={q} className={`example ${p.question === q ? "on" : ""}`} onClick={() => p.onQuestion(q)}>{q}</button>
            ))}
          </div>
          <div className="actions">
            <button className="btn primary" disabled={!p.ready || !p.question.trim()} onClick={p.onStart}>Ask</button>
            <span className="label">{p.ready ? "Enter to ask · SPACE when you've read a sentence" : "Waiting for the models to load…"}</span>
          </div>
          <ol className="howto">
            <li><span className="num">01</span><b>Read</b> each sentence as it appears while EEG records</li>
            <li><span className="num">02</span><b>Decode</b> the epoch into semantic vectors</li>
            <li><span className="num">03</span><b>Steer</b> Qwen&apos;s next sentence with them</li>
          </ol>
        </div>
      ) : (
        <>
          <p className="question"><b>Q</b><span>{p.asked}</span></p>
          <div className="sentence">
            <Tokens tokens={tokens} animate />
            {phase === "generating" && <span className="caret" />}
            {phase === "decoding" && <span className="scan" />}
          </div>
          <div className="timer">
            {phase === "reading" && (
              <>
                <ReadRing startedAt={p.readStartedAt} />
                <span className="label">Reading · press <kbd className="key">SPACE</kbd> when done</span>
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
    </section>
  );
}
