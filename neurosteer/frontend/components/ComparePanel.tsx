"use client";

import { useEffect, useMemo, useState } from "react";
import Modal from "@/components/Modal";
import { API, streamErrorText } from "@/lib/api";
import type { CompareDone, CompareLane, CompareSummary, CompareToken, Token } from "@/lib/api";

type Props = { open: boolean; onClose: () => void; question: string };
type Run = {
  tokens: Record<CompareLane, Token[]>;
  ms: Partial<Record<CompareLane, number>>;
  summary: CompareSummary | null;
  error: string | null;
};
const EMPTY: Run = { tokens: { steered: [], baseline: [] }, ms: {}, summary: null, error: null };

const norm = (w: string) => w.toLowerCase().replace(/[^\p{L}\p{N}'-]/gu, "");
const words = (ts: Token[]) => ts.map((t) => t.text).join("").split(/\s+/).map(norm).filter(Boolean);

function useCompare(open: boolean, runId: number) {
  const [run, setRun] = useState<Run>(EMPTY);
  useEffect(() => {
    if (!open) return;
    setRun(EMPTY);
    let got = false;
    const es = new EventSource(`${API}/api/compare`);
    es.addEventListener("token", (ev) => {
      got = true;
      const t = JSON.parse((ev as MessageEvent).data) as CompareToken;
      const lane: CompareLane = t.lane === "baseline" ? "baseline" : "steered";
      setRun((r) => ({ ...r, tokens: { ...r.tokens, [lane]: [...r.tokens[lane], t] } }));
    });
    es.addEventListener("done", (ev) => {
      const d = JSON.parse((ev as MessageEvent).data) as CompareDone;
      setRun((r) => ({ ...r, ms: { ...r.ms, [d.lane]: d.ms } }));
    });
    es.addEventListener("summary", (ev) => {
      es.close();
      setRun((r) => ({ ...r, summary: JSON.parse((ev as MessageEvent).data) as CompareSummary }));
    });
    es.onerror = (ev) => {
      es.close();
      setRun((r) => (r.summary ? r : {
        ...r,
        error: streamErrorText(ev, got ? "The compare stream was interrupted."
          : "Compare is unavailable: no active session, Qwen is busy generating, or the backend lacks /api/compare."),
      }));
    };
    return () => es.close();
  }, [open, runId]);
  return run;
}

function Lane({ name, kind, tokens, ms, mark, streaming }: {
  name: string; kind: CompareLane; tokens: Token[]; ms?: number; mark: Set<string> | null; streaming: boolean;
}) {
  return (
    <div className={`lane ${kind}`}>
      <div className="lane-head">
        <span className="label">{name}</span>
        <span className="label num">{ms != null ? `${ms} ms` : streaming ? "streaming" : ""}</span>
      </div>
      <p className="lane-text">
        {tokens.map((t, i) => {
          const only = mark && kind === "steered" && mark.has(norm(t.text));
          return (
            <span key={i} className={`tok ${t.steered ? "steer" : ""} ${only ? "only" : ""}`}
              title={only ? "Only in the steered sentence" : t.steered ? `bias +${t.bias}` : undefined}>{t.text}</span>
          );
        })}
        {streaming && <span className="caret" />}
        {!tokens.length && !streaming && <span className="empty">—</span>}
      </p>
    </div>
  );
}

export default function ComparePanel({ open, onClose, question }: Props) {
  const [runId, setRunId] = useState(0);
  const run = useCompare(open, runId);
  const finished = !!run.summary || !!run.error;

  const mark = useMemo(() => {
    if (run.summary) return new Set(run.summary.steered_words.map(norm));
    if (run.ms.steered == null || run.ms.baseline == null) return null;
    const base = new Set(words(run.tokens.baseline));
    return new Set(words(run.tokens.steered).filter((w) => !base.has(w)));
  }, [run]);

  const overlap = run.summary ? (run.summary.overlap > 1 ? run.summary.overlap / 100 : run.summary.overlap) : null;

  return (
    <Modal open={open} onClose={onClose} title="Steered vs baseline" kicker="A/B compare · next sentence" wide>
      <p className="cmp-intro">
        The same next sentence for <b>“{question || "this session"}”</b>, generated twice from the same history:
        once with the neural prefix, residual Δh and logit bias from the last EEG decode, once without. The transcript is not changed.
      </p>
      <div className="lanes">
        <Lane name="Steered · neural channels on" kind="steered" tokens={run.tokens.steered} ms={run.ms.steered} mark={mark}
          streaming={!finished && run.ms.steered == null} />
        <Lane name="Baseline · steering off" kind="baseline" tokens={run.tokens.baseline} ms={run.ms.baseline} mark={null}
          streaming={!finished && run.ms.baseline == null} />
      </div>

      <div className="cmp-foot">
        {run.error ? <p className="notice" style={{ margin: 0 }}>{run.error}</p> : (
          <div className="cmp-stats">
            <div className="kv">
              <span className="label">Word overlap</span>
              <span className="cmp-big num">{overlap == null ? "—" : `${Math.round(overlap * 100)}%`}</span>
            </div>
            <div className="overlap-bar" aria-hidden>
              <span className="ov-shared" style={{ width: `${(overlap ?? 0) * 100}%` }} />
            </div>
            <div className="kv">
              <span className="label">Only in steered</span>
              <span className="cmp-words">
                {mark && mark.size ? [...mark].slice(0, 10).map((w) => <span key={w} className="chip out">{w}</span>)
                  : <span className="empty">{finished || mark ? "none" : "waiting for both lanes…"}</span>}
              </span>
            </div>
          </div>
        )}
        <button className="btn" onClick={() => setRunId((n) => n + 1)} disabled={!finished}>Run again</button>
      </div>
      <p className="cmp-note">
        Amber underline: token picked while the logit bias favoured it. Highlighted: words that appear only in the steered lane.
        A single pair of samples is anecdotal; sampling noise alone changes wording, so run it a few times.
      </p>
    </Modal>
  );
}
