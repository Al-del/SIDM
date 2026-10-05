import ElectrodeArray from "@/components/ElectrodeArray";
import type { Decode, Montage, Status } from "@/lib/api";
import type { Metrics } from "@/lib/useEEG";

const BANDS = ["delta", "theta", "alpha", "beta", "gamma"];
const SOURCES = [["brainaccess", "BA DIRECT"], ["lsl", "LSL"], ["replay", "REPLAY"], ["synthetic", "SYNTH"]] as const;

type Props = {
  montage: Montage | null;
  metrics: Metrics | null;
  reading: boolean;
  eeg: Status["eeg"] | undefined;
  decode: Decode | null;
  switching: string | null;
  onSource: (kind: string) => void;
};

export default function LeftRail({ montage, metrics, reading, eeg, decode, switching, onSource }: Props) {
  return (
    <>
      <div className="section">
        <div className="section-head">
          <span className="label">Electrode array</span>
          <span className="label num">{eeg?.channels ?? 105} CH · {eeg?.fs ?? 250} HZ</span>
        </div>
        <ElectrodeArray montage={montage} metrics={metrics} reading={reading} sensors={montage?.sensors ?? []} />
        <div className="array-readout">
          <div className="kv"><span className="label">Montage</span><span className="v mono" style={{ fontSize: 11 }}>{eeg?.live ? `${eeg.headset_channels?.length ?? 0} → 105` : "HCGSN-128 / 105"}</span></div>
          <div className="kv"><span className="label">Epoch</span><span className="v num">{decode ? `${decode.seconds.toFixed(2)} s` : "—"}</span></div>
        </div>
      </div>
      <div className="section">
        <div className="section-head"><span className="label">Relative band power</span></div>
        <div className="bands">
          {BANDS.map((b) => (
            <div className="band" key={b}>
              <div className="band-bar" style={{ height: `${Math.max(3, (metrics?.bands[b] ?? 0) * 140)}%` }} />
              <span className="band-name">{b.slice(0, 3).toUpperCase()}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="section">
        <div className="section-head"><span className="label">Signal source</span></div>
        <div className="seg">
          {SOURCES.map(([k, n]) => (
            <button key={k} className={`${eeg?.kind === k ? "on" : ""} ${switching === k ? "busy" : ""}`}
              disabled={!!switching} onClick={() => onSource(k)}>{switching === k ? "LINKING" : n}</button>
          ))}
        </div>
        {eeg?.kind === "synthetic" && (
          <p className="notice">Synthetic signal. Decoded units are not brain-derived; use this only to test the pipeline.</p>
        )}
        {eeg?.kind === "replay" && (
          <p className="notice">Real held-out ZuCo EEG, but recorded on other sentences. Units reflect that recording, not what you read here.</p>
        )}
        {eeg?.live && (
          <p className="notice real">
            Live headset{eeg.model ? ` · BrainAccess ${eeg.model}` : ""}{eeg.battery != null ? ` · battery ${eeg.battery}%` : ""}.
            {" "}{eeg.headset_channels?.length} channels ({eeg.headset_channels?.join(", ")}) band-passed 1–40 Hz and
            interpolated onto the 105-channel layout the decoder was trained on. Expect weak decoding with few channels.
            {eeg.unknown_labels?.length ? ` Unplaced: ${eeg.unknown_labels.join(", ")}.` : ""}
          </p>
        )}
      </div>
    </>
  );
}
