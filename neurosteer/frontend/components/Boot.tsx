import Logo from "@/components/Logo";
import type { Status } from "@/lib/api";

export default function Boot({ status, error, hidden }: { status: Status | null; error: string | null; hidden: boolean }) {
  const line = (name: string, ok: boolean, err?: string) => (
    <div>
      <span>{name.padEnd(30, ".")}</span>{" "}
      {err ? <span className="err">FAULT</span> : ok ? <span className="ok">ONLINE</span> : <span>LOADING</span>}
    </div>
  );
  return (
    <div className={`boot ${hidden ? "gone" : ""}`}>
      <Logo size={40} />
      <span className="label" style={{ letterSpacing: "0.4em" }}>NEUROSTEER · INITIALISING</span>
      <div className="boot-lines">
        {line("FLASK LINK", !!status, error ?? undefined)}
        {line("EEG ACQUISITION", !!status?.eeg.kind)}
        {line("RAG-MOSAIC DECODER", !!status?.decoder, status?.errors.decoder)}
        {line(`QWEN · ${status?.llm?.name ?? "Qwen3-1.7B"}`.toUpperCase(), !!status?.llm, status?.errors.llm)}
        {(error || status?.errors.llm || status?.errors.decoder) && (
          <div className="err" style={{ marginTop: 10, maxWidth: 420 }}>{error ?? status?.errors.llm ?? status?.errors.decoder}</div>
        )}
      </div>
    </div>
  );
}
