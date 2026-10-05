const SEGMENTS = 10;

export const ALIGNMENT_HELP =
  "Alignment (0–1): the backend's similarity between what the decoder read from this EEG epoch and the sentence that was on screen. A sanity signal, not mind reading; replay and synthetic sources are not expected to align.";

export default function AlignmentMeter({ value, compact }: { value: number | null | undefined; compact?: boolean }) {
  if (typeof value !== "number" || Number.isNaN(value)) return null;
  const v = Math.max(0, Math.min(1, value));
  const lit = Math.round(v * SEGMENTS);
  return (
    <span className={`align ${compact ? "compact" : ""}`} title={ALIGNMENT_HELP}
      role="meter" aria-valuemin={0} aria-valuemax={1} aria-valuenow={Number(v.toFixed(2))} aria-label="EEG–sentence alignment">
      <span className="align-name">ALIGN</span>
      <span className="align-segs" aria-hidden>
        {Array.from({ length: SEGMENTS }, (_, i) => (
          <i key={i} className={i < lit ? (v >= 0.6 ? "hi" : v >= 0.35 ? "mid" : "lo") : ""} />
        ))}
      </span>
      <span className="align-v num">{v.toFixed(2)}</span>
    </span>
  );
}
