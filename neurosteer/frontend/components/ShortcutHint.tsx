import { SHORTCUTS } from "@/lib/useHotkeys";

export function ShortcutList() {
  return (
    <dl className="keys-list">
      {SHORTCUTS.map(([k, d]) => (
        <div key={k}><dt><kbd className="key">{k}</kbd></dt><dd>{d}</dd></div>
      ))}
    </dl>
  );
}

export default function ShortcutHint({ open, onClose }: { open: boolean; onClose: () => void }) {
  if (!open) return null;
  return (
    <>
      <div className="popover-scrim" onClick={onClose} aria-hidden />
      <div className="popover" role="dialog" aria-label="Keyboard shortcuts">
        <span className="label">Keyboard</span>
        <ShortcutList />
      </div>
    </>
  );
}
