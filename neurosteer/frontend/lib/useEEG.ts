"use client";

import { useEffect, useRef, useState } from "react";
import { API } from "./api";

export const WINDOW_SECONDS = 8;

export type Traces = {
  data: Float32Array[];
  reading: Uint8Array;
  head: number;
  size: number;
  fs: number;
  total: number;
};

export type Metrics = { rms: number[]; bands: Record<string, number> };

export function useEEG(channels: number) {
  const traces = useRef<Traces | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [connected, setConnected] = useState(false);
  const [rate, setRate] = useState(0);

  useEffect(() => {
    if (!channels) return;

    traces.current = null;
    const es = new EventSource(`${API}/api/eeg`);
    let count = 0;
    let t0 = performance.now();
    es.onopen = () => setConnected(true);

    es.onerror = () => setConnected(false);
    es.addEventListener("eeg", (ev) => {
      const f = JSON.parse((ev as MessageEvent).data) as { fs: number; samples: number[][]; reading: boolean };
      const n = f.samples[0]?.length ?? 0;
      let tr = traces.current;
      if (!tr || tr.fs !== f.fs || tr.data.length !== channels) {
        const size = Math.round(f.fs * WINDOW_SECONDS);
        tr = {
          data: Array.from({ length: channels }, () => new Float32Array(size)),
          reading: new Uint8Array(size), head: 0, size, fs: f.fs, total: 0,
        };
        traces.current = tr;
      }
      for (let i = 0; i < n; i++) {
        const p = (tr.head + i) % tr.size;
        for (let c = 0; c < channels; c++) tr.data[c][p] = f.samples[c]?.[i] ?? 0;
        tr.reading[p] = f.reading ? 1 : 0;
      }
      tr.head = (tr.head + n) % tr.size;
      tr.total += n;
      count += n;
      const now = performance.now();
      if (now - t0 > 1000) {
        setRate(Math.round((count * 1000) / (now - t0)));
        count = 0;
        t0 = now;
      }
    });
    es.addEventListener("metrics", (ev) => setMetrics(JSON.parse((ev as MessageEvent).data)));
    return () => { es.close(); setConnected(false); };
  }, [channels]);

  return { traces, metrics, connected, rate };
}
