"use client";

import { useEffect, useRef } from "react";
import type { Entry } from "@/lib/api";
import AlignmentMeter from "@/components/AlignmentMeter";

export default function Transcript({ entries }: { entries: Entry[] }) {
  const tailRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    tailRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [entries.length]);

  return (
    <div className="transcript">
      {entries.length === 0 && <p className="empty" style={{ padding: "20px 34px" }}>The answer builds here sentence by sentence. Amber words were pushed by the neural bias; → chips were injected into Qwen, ← chips were decoded from your EEG while reading.</p>}
      {entries.map((e) => (
        <div className="tr-row" key={e.id}>
          <span className="tr-idx num">{String(e.id + 1).padStart(2, "0")}</span>
          <div>
            <p className="tr-text">
              {e.tokens.map((t, i) => <span key={i} className={t.steered ? "steer" : ""}>{t.text}</span>)}
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
