"use client";

import { useEffect, useState } from "react";

const mmss = (ms: number) => {
  const s = Math.floor(ms / 1000);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}.${Math.floor((ms % 1000) / 100)}`;
};

export default function Clock({ t0 }: { t0: number | null }) {
  const [now, setNow] = useState(0);
  useEffect(() => {
    if (t0 === null) return;
    const id = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(id);
  }, [t0]);
  const ms = t0 === null ? 0 : Math.max(0, now - t0);
  return <span className="clock num" aria-label="Session time">T+{mmss(ms)}</span>;
}
