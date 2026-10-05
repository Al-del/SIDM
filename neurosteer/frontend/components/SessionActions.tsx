"use client";

import { useEffect, useState } from "react";

type Props = {
  onExport: () => void;
  onReset: () => void;
  exporting: boolean;
  canExport: boolean;
  compare?: { onClick: () => void; disabled: boolean; hint: string };
};

export default function SessionActions({ onExport, onReset, exporting, canExport, compare }: Props) {
  const [confirm, setConfirm] = useState(false);
  useEffect(() => {
    if (!confirm) return;
    const id = setTimeout(() => setConfirm(false), 3000);
    return () => clearTimeout(id);
  }, [confirm]);

  return (
    <div className="session-actions">
      {compare && (
        <button className="tool accent" onClick={compare.onClick} disabled={compare.disabled} title={compare.hint}>
          A/B COMPARE <kbd>C</kbd>
        </button>
      )}
      <button className="tool" onClick={onExport} disabled={!canExport || exporting} title="Download this session as JSON (E)">
        {exporting ? "EXPORTING…" : <>EXPORT <kbd>E</kbd></>}
      </button>
      <button className={`tool danger ${confirm ? "on" : ""}`}
        onClick={() => { if (confirm) { setConfirm(false); onReset(); } else setConfirm(true); }}
        title="Clear the transcript and reset the backend session">
        {confirm ? "CONFIRM RESET" : "RESET"}
      </button>
    </div>
  );
}
