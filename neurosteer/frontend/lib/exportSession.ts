import { API, ApiError } from "@/lib/api";
import type { Entry, Settings } from "@/lib/api";

function save(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const stamp = () => new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);

export async function exportSession(local: { question: string; entries: Entry[]; settings: Settings | null }): Promise<"backend" | "local"> {
  try {
    const r = await fetch(`${API}/api/export`);
    if (!r.ok) throw new ApiError(r.statusText, r.status);
    save(await r.blob(), `neurosteer-session-${stamp()}.json`);
    return "backend";
  } catch (e) {
    if (e instanceof ApiError && e.status !== 404 && e.status !== 405) throw e;
    const body = { exported_at: new Date().toISOString(), source: "browser", ...local };
    save(new Blob([JSON.stringify(body, null, 2)], { type: "application/json" }), `neurosteer-session-${stamp()}.json`);
    return "local";
  }
}
