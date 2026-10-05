"use client";

import { useEffect, useRef } from "react";
import type { Entry } from "@/lib/api";
import AlignmentMeter from "@/components/AlignmentMeter";
import Tokens from "@/components/Tokens";

function Legend() {
  return (
    <div className="legend">
      <p className="empty">The answer builds here, one sentence per row. How to read a row:</p>
      <dl>
        <div><dt><span className="steer-sample">amber word</span></dt><dd>token picked while the neural logit bias favoured it</dd></div>
        <div><dt><span className="chip out">→ memory</span></dt><dd>units injected into Qwen to steer this sentence</dd></div>
        <div><dt><span className="chip in">← sleep</span></dt><dd>units decoded from the EEG while you read it</dd></div>
        <div><dt><AlignmentMeter value={0.6} compact /></dt><dd>how well that decode matched the sentence on screen</dd></div>
      </dl>
    </div>
  );
}

const time = (at: string | number) => new Date(typeof at === "number" ? at * 1000 : at).toLocaleTimeString();

export default function Transcript({ entries }: { entries: Entry[] }) {
  const tailRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    tailRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [entries.length]);

  return (
    <div className="transcript" aria-label="Transcript">
      {entries.length === 0 && <Legend />}
      {entries.map((e) => (
        <div className="tr-row" key={e.id}>
          <span className="tr-idx num" title={e.at != null ? time(e.at) : undefined}>{String(e.id + 1).padStart(2, "0")}</span>
          <div>
            <p className="tr-text">
              <Tokens tokens={e.tokens} />
            </p>
            <div className="chips">
              {e.plan?.units.slice(0, 6).map((u) => <span key={"o" + u} className="chip out">→ {u}</span>)}
              {e.decode?.units.slice(0, 6).map((u) => <span key={"i" + u.unit} className="chip in">← {u.word}</span>)}
              <span className="chip num">{e.ms} ms</span>
              <AlignmentMeter value={e.decode?.alignment} compact />
            </div>
          </div>
        </div>
      ))}
      <div ref={tailRef} />
    </div>
  );
}
