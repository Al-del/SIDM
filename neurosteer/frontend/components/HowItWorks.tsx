import Modal from "@/components/Modal";
import { ShortcutList } from "@/components/ShortcutHint";

type Props = { open: boolean; onClose: () => void; source?: string | null; mock: boolean };

function Node({ x, y, w, h, title, sub, accent }: { x: number; y: number; w: number; h: number; title: string; sub: string[]; accent?: boolean }) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} className={`hw-node ${accent ? "accent" : ""}`} />
      <text x={x + 12} y={y + 22} className="hw-title">{title}</text>
      {sub.map((s, i) => <text key={i} x={x + 12} y={y + 40 + i * 14} className="hw-sub">{s}</text>)}
    </g>
  );
}

function Diagram() {
  return (
    <svg viewBox="0 0 960 330" className="hw-svg" role="img"
      aria-label="Diagram: you read a sentence, EEG is recorded, the RAG-Mosaic decoder turns it into semantic vectors, a translator maps them into Qwen's embedding space, and they steer Qwen's next sentence through a neural prefix, a residual nudge at layer 14 and a logit bias.">
      <defs>
        <marker id="hw-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M0 0 L8 4 L0 8 z" fill="currentColor" />
        </marker>
      </defs>

      {}
      <Node x={20} y={130} w={130} h={78} title="YOU" sub={["read one sentence", "(≤ 9 s)"]} />
      <Node x={195} y={130} w={140} h={78} title="EEG" sub={["105 channels", "1 epoch / sentence"]} />
      <Node x={380} y={130} w={165} h={78} title="RAG-MOSAIC" sub={["EEG → ẑ + word-slot", "vectors (256-d)"]} accent />
      <Node x={590} y={150} w={135} h={78} title="TRANSLATOR" sub={["MLP 256 → 2048", "into Qwen space"]} accent />

      <g className="hw-flow signal">
        <line x1={150} y1={169} x2={191} y2={169} markerEnd="url(#hw-arrow)" />
        <line x1={330} y1={169} x2={376} y2={169} markerEnd="url(#hw-arrow)" />
        <path d="M545 189 L586 189" markerEnd="url(#hw-arrow)" />
      </g>

      {}
      <g>
        <rect x={770} y={24} width={170} height={282} className="hw-qwen" />
        <text x={786} y={46} className="hw-title">QWEN3-1.7B</text>
        <text x={786} y={60} className="hw-sub">frozen weights</text>
        <rect x={786} y={74} width={138} height={34} className="hw-layer" />
        <text x={798} y={95} className="hw-sub">output logits</text>
        <rect x={786} y={116} width={138} height={26} className="hw-layer dim" />
        <text x={798} y={133} className="hw-sub">layers 15–28</text>
        <rect x={786} y={150} width={138} height={34} className="hw-layer hot" />
        <text x={798} y={171} className="hw-sub">layer 14 · residual</text>
        <rect x={786} y={192} width={138} height={26} className="hw-layer dim" />
        <text x={798} y={209} className="hw-sub">layers 1–13</text>
        <rect x={786} y={226} width={138} height={34} className="hw-layer" />
        <text x={798} y={247} className="hw-sub">input embeddings</text>
        <text x={786} y={290} className="hw-sub">→ next sentence</text>
      </g>

      {}
      <g className="hw-flow inject">
        <path d="M462 130 L462 91 L782 91" markerEnd="url(#hw-arrow)" />
        <path d="M725 172 L782 167" markerEnd="url(#hw-arrow)" />
        <path d="M690 228 L690 243 L782 243" markerEnd="url(#hw-arrow)" />
      </g>
      <text x={476} y={83} className="hw-chan">3 · LOGIT BIAS · boost the decoded words</text>
      <text x={732} y={160} className="hw-chan">2 · Δh</text>
      <text x={600} y={262} className="hw-chan">1 · NEURAL PREFIX</text>

      {}
      <g className="hw-flow loop">
        <path d="M855 306 L855 318 L85 318 L85 212" markerEnd="url(#hw-arrow)" />
      </g>
      <text x={400} y={312} className="hw-sub" textAnchor="middle">the new sentence appears on screen · the loop repeats</text>
    </svg>
  );
}

export default function HowItWorks({ open, onClose, source, mock }: Props) {
  return (
    <Modal open={open} onClose={onClose} title="A brain-in-the-loop language model" kicker="How it works" wide>
      <Diagram />

      <div className="hw-steps">
        <div>
          <span className="hw-n num">01</span>
          <h3>Read</h3>
          <p>Qwen writes its answer one sentence at a time. While you read a sentence, the app records an EEG epoch of up to 9 seconds. Press SPACE when you&apos;re done.</p>
        </div>
        <div>
          <span className="hw-n num">02</span>
          <h3>Decode</h3>
          <p>RAG-Mosaic, an EEG→text decoder trained on the ZuCo reading dataset, turns the epoch into semantic vectors: a sentence-level gist (ẑ) and a few word slots. It acts as the middle man between brain and language model.</p>
        </div>
        <div>
          <span className="hw-n num">03</span>
          <h3>Steer</h3>
          <p>A small trained translator maps those vectors into Qwen&apos;s own embedding space. They nudge the next sentence three ways: as extra “neural” prompt tokens, as a push on layer 14&apos;s hidden state, and as a bias towards the decoded words, which are underlined in amber.</p>
        </div>
      </div>

      <div className="hw-caveats">
        <span className="label" style={{ color: "var(--inject)" }}>Read this before judging the output</span>
        <ul>
          {mock && <li><b>Demo mode is on.</b> The backend is running simulated models; the tokens and decodes are synthetic and only show the interface.</li>}
          <li><b>REPLAY isn&apos;t your brain.</b> It streams real EEG from the held-out ZuCo set, recorded on other people reading other sentences. <b>SYNTH</b> is generated signal. Only a live headset (LSL / BA DIRECT) records you{source ? `; the current source is ${source}` : ""}.</li>
          <li><b>The signal is weak.</b> The decoder&apos;s EEG-specific information on new users is small, and a consumer headset with 4–8 channels is interpolated onto the 105-channel layout it was trained on: that keeps the shape of the input, not the information.</li>
          <li><b>Steering is subtle by design.</b> Strong settings break grammar. Use Compare (C) to see the same sentence with and without the neural channels; alignment scores are a backend similarity measure, not mind reading.</li>
        </ul>
      </div>

      <div className="hw-keys">
        <span className="label">Keyboard</span>
        <ShortcutList />
      </div>
    </Modal>
  );
}
