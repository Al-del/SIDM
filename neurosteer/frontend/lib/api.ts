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
};

export type Plan = {
  units: string[];
  weights: number[];
  prefix: number[][] | null;
  labels: string[];
  mode: "translator" | "lookup";
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
};

export type Status = {
  eeg: {
    kind: string | null; label?: string; fs?: number; channels?: number; brain_derived?: boolean; live?: boolean;
    headset_channels?: string[]; unknown_labels?: string[]; model?: string; battery?: number | null;
  };
  decoder: boolean;
  llm: { name: string; layers: number; hidden: number; steer_layer: number; device: string; translator: boolean } | null;
  loading: boolean;
  errors: Record<string, string>;
  settings: Settings;
};

export type Montage = {
  labels: string[]; pos: [number, number][]; display: number[]; display_labels: string[]; sensors: number[];
};

export async function post<T>(path: string, body: unknown = {}): Promise<T> {
  const r = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error ?? r.statusText);
  return data as T;
}

export async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${API}${path}`);
  return (await r.json()) as T;
}
