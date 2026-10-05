import type { Toast } from "@/lib/useToasts";

const TAG = { error: "FAULT", ok: "OK", info: "NOTE" } as const;

export default function Toasts({ toasts, onDismiss }: { toasts: Toast[]; onDismiss: (id: number) => void }) {
  return (
    <div className="toasts" role="region" aria-label="Notifications" aria-live="polite">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.kind}`} role={t.kind === "error" ? "alert" : "status"}>
          <span className="toast-tag mono">{TAG[t.kind]}{t.count > 1 ? ` ×${t.count}` : ""}</span>
          <span className="toast-text">{t.text}</span>
          <button className="toast-x" onClick={() => onDismiss(t.id)} aria-label="Dismiss notification">×</button>
        </div>
      ))}
    </div>
  );
}
