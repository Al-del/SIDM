"use client";

import { useEffect, useMemo, useState } from "react";
import { getOptional } from "@/lib/api";
import type { Entry, Stats } from "@/lib/api";

const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);

export function localStats(entries: Entry[]): Stats {
  const tokens = entries.flatMap((e) => e.tokens);
  return {
    sentences: entries.length,
    mean_alignment: mean(entries.flatMap((e) => (typeof e.decode?.alignment === "number" ? [e.decode.alignment] : []))),
    mean_latency_ms: mean(entries.flatMap((e) => (e.decode ? [e.decode.latency_ms] : []))),
    steered_tokens: tokens.filter((t) => t.steered).length,
    total_tokens: tokens.length,
  };
}

export function useStats(version: number, entries: Entry[]) {
  const [remote, setRemote] = useState<Stats | null>(null);
  const [available, setAvailable] = useState(true);
  const local = useMemo(() => localStats(entries), [entries]);

  useEffect(() => {
    if (!available) return;
    let alive = true;
    getOptional<Stats>("/api/stats")
      .then((s) => {
        if (!alive) return;
        if (s === null) setAvailable(false);
        setRemote(s);
      })
      .catch(() => alive && setRemote(null));
    return () => { alive = false; };
  }, [version, available]);

  return { stats: remote ?? local, remote: !!remote };
}
