"use client";

import { useEffect, useMemo, useState } from "react";
import { MAX_READ } from "@/lib/useSession";

const R = 19;
const LEN = 2 * Math.PI * R;

export default function ReadRing({ startedAt }: { startedAt: number }) {
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    const id = setInterval(() => setNow(performance.now()), 100);
    return () => clearInterval(id);
  }, [startedAt]);
  const t = Math.min(MAX_READ, Math.max(0, (now - startedAt) / 1000));

  const anim = useMemo(() => ({
    animationDuration: `${MAX_READ}s`,
    animationDelay: `${-Math.max(0, (performance.now() - startedAt) / 1000)}s`,
  }), [startedAt]);
  return (
    <>
      <span className="ring" role="progressbar" aria-label="Reading time" aria-valuemin={0} aria-valuemax={MAX_READ}
        aria-valuenow={Number(t.toFixed(1))}>
        <svg viewBox="0 0 44 44" aria-hidden>
          <circle className="ring-track" cx={22} cy={22} r={R} />
          <circle key={startedAt} className="ring-fill" cx={22} cy={22} r={R}
            style={{ strokeDasharray: LEN, ["--len" as string]: LEN, ...anim }} />
        </svg>
        <span className="ring-v num">{(MAX_READ - t).toFixed(1)}</span>
      </span>
      <span key={startedAt} className="ring-sweep" style={anim} aria-hidden />
    </>
  );
}
