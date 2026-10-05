export const API = process.env.NEXT_PUBLIC_API ?? "http://127.0.0.1:5050";

export type Settings = {
  prefix: number;
  prefix_tokens: number;
  bias: number;
  residual: number;
  hint: boolean;
  temperature: number;
  max_tokens: number;
  max_sentences: number;
};

export type Unit = { unit: number; word: string; words: string[]; weight: number };

export type Decode = {
  units: Unit[];
  slots: { slot: number; p_active: number; candidates: { words: string[]; cos: number }[] }[];
  neighbours: { sentence: string; sim: number }[];
  seconds: number;
  latency_ms: number;
  brain_derived: boolean;
  source: string;
  
  alignment?: number | null;
};

export type Plan = {
  units: string[];
  weights: number[];
  prefix: number[][] | null;
  labels: string[];
  mode: "translator" | "lookup" | "mock";
  bias_tokens: number;
  residual_norm: number;
  layer: number;
} | null;

export type Token = { text: string; steered: boolean; bias: number };

export type Entry = {
  id: number;
  sentence: string;
  tokens: Token[];
  plan: Plan;
  decode: Decode | null;
  ms: number;
  end: boolean;
  
  at?: string | number;
};

export type Status = {
  eeg: {
    kind: string | null; label?: string; fs?: number; channels?: number; brain_derived?: boolean; live?: boolean;
    headset_channels?: string[]; unknown_labels?: string[]; model?: string; battery?: number | null;
  };
  decoder: boolean;
  llm: {
    name: string; layers: number; hidden: number; steer_layer: number; device: string; translator: boolean; mock?: boolean;
  } | null;
  loading: boolean;
  errors: Record<string, string>;
  settings: Settings;
  
  mock?: boolean;
};

export type Montage = {
  labels: string[]; pos: [number, number][]; display: number[]; display_labels: string[]; sensors: number[];
};

export type Health = { ok: boolean; uptime_s: number; version: string };

export type Stats = {
  sentences: number;
  mean_alignment: number | null;
  mean_latency_ms: number | null;
  mean_decode_ms?: number | null;
  steered_tokens: number;
  total_tokens: number;
};

export const PRESETS = ["subtle", "balanced", "strong", "off"] as const;
export type PresetName = (typeof PRESETS)[number];
export type Presets = Record<PresetName, Partial<Settings>>;

export type CompareLane = "steered" | "baseline";
export type CompareToken = Token & { lane: CompareLane };
export type CompareDone = { lane: CompareLane; sentence: string; ms: number };
export type CompareSummary = { overlap: number; steered_words: string[] };

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function parse<T>(r: Response): Promise<T> {
  let data: unknown = null;
  try {
    data = await r.json();
  } catch {
    if (r.ok) throw new ApiError(`Malformed response from ${r.url}`, r.status);
  }
  if (!r.ok) {
    const msg = (data as { error?: string } | null)?.error;
    throw new ApiError(msg ?? `${r.status} ${r.statusText}`.trim(), r.status);
  }
  return data as T;
}

export async function post<T>(path: string, body: unknown = {}): Promise<T> {
  const r = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return parse<T>(r);
}

export async function get<T>(path: string): Promise<T> {
  return parse<T>(await fetch(`${API}${path}`));
}

export async function getOptional<T>(path: string): Promise<T | null> {
  try {
    return await get<T>(path);
  } catch (e) {
    if (e instanceof ApiError && (e.status === 404 || e.status === 405)) return null;
    throw e;
  }
}

export function streamErrorText(ev: Event, fallback: string): string {
  if (ev instanceof MessageEvent && typeof ev.data === "string" && ev.data) {
    try {
      const d = JSON.parse(ev.data) as { error?: string; message?: string };
      return d.error ?? d.message ?? fallback;
    } catch {
      return ev.data;
    }
  }
  return fallback;
}

export function errorText(e: unknown): string {
  if (e instanceof ApiError && e.status === 400) return e.message || "No active session: ask a question first.";
  if (e instanceof ApiError && e.status === 409) return "Qwen is already generating. Wait for it to finish.";
  if (e instanceof TypeError) return `Backend unreachable at ${API}`;
  return e instanceof Error ? e.message : String(e);
}
