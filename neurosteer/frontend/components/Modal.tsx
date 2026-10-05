"use client";

import { useEffect, useRef } from "react";
import type { ReactNode } from "react";

type Props = {
  open: boolean;
  onClose: () => void;
  title: string;
  kicker?: string;
  wide?: boolean;
  children: ReactNode;
};

export default function Modal({ open, onClose, title, kicker, wide, children }: Props) {
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const prev = document.activeElement as HTMLElement | null;
    box.current?.querySelector<HTMLElement>("[data-autofocus]")?.focus();
    const onTab = (e: KeyboardEvent) => {
      if (e.key !== "Tab" || !box.current) return;
      const f = box.current.querySelectorAll<HTMLElement>("button:not(:disabled), a[href], input, [tabindex]:not([tabindex='-1'])");
      if (!f.length) return;
      const first = f[0], last = f[f.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    };
    document.addEventListener("keydown", onTab);
    return () => {
      document.removeEventListener("keydown", onTab);
      prev?.focus?.();
    };
  }, [open]);

  if (!open) return null;
  return (
    <div className="modal-back" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div ref={box} className={`modal ${wide ? "wide" : ""}`} role="dialog" aria-modal="true" aria-label={title}>
        <span className="corner tl" /><span className="corner tr" /><span className="corner bl" /><span className="corner br" />
        <header className="modal-head">
          <div>
            {kicker && <span className="label">{kicker}</span>}
            <h2>{title}</h2>
          </div>
          <button className="tool" onClick={onClose} data-autofocus aria-label="Close">ESC</button>
        </header>
        <div className="modal-body">{children}</div>
      </div>
    </div>
  );
}
