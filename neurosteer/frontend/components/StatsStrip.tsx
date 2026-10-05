import { ALIGNMENT_HELP } from "@/components/AlignmentMeter";
import type { Stats } from "@/lib/api";

function Bar({ v, amber }: { v: number; amber?: boolean }) {
  return <span className={`kpi-bar ${amber ? "amber" : ""}`}><span style={{ width: `${Math.max(0, Math.min(1, v)) * 100}%` }} /></span>;
}

export default function StatsStrip({ stats, remote }: { stats: Stats; remote: boolean }) {
  const steeredShare = stats.total_tokens ? stats.steered_tokens / stats.total_tokens : 0;
  return (
    <div className="kpis" aria-label="Session statistics" title={remote ? "From /api/stats" : "Computed from this transcript"}>
      <div className="kpi">
        <span className="label">Sentences</span>
        <span className="kpi-v num">{String(stats.sentences).padStart(2, "0")}</span>
      </div>
      <div className="kpi" title={ALIGNMENT_HELP}>
        <span className="label">Alignment</span>
        <span className="kpi-v num">{stats.mean_alignment == null ? "—" : stats.mean_alignment.toFixed(2)}</span>
        <Bar v={stats.mean_alignment ?? 0} />
      </div>
      <div className="kpi" title={`Mean time to generate a sentence${stats.mean_decode_ms != null ? `; EEG decode ${Math.round(stats.mean_decode_ms)} ms` : ""}`}>
        <span className="label">Latency{stats.mean_decode_ms != null ? ` · dec ${Math.round(stats.mean_decode_ms)}` : ""}</span>
        <span className="kpi-v num">{stats.mean_latency_ms == null ? "—" : Math.round(stats.mean_latency_ms)}<small>{stats.mean_latency_ms == null ? "" : " ms"}</small></span>
      </div>
      <div className="kpi" title={`${stats.steered_tokens} of ${stats.total_tokens} tokens were picked while the logit bias favoured them`}>
        <span className="label">Steered</span>
        <span className="kpi-v num">{stats.total_tokens ? Math.round(steeredShare * 100) : "—"}<small>{stats.total_tokens ? " %" : ""}</small></span>
        <Bar v={steeredShare} amber />
      </div>
    </div>
  );
}
