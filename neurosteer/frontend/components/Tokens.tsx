import type { Token } from "@/lib/api";

type Props = {
  tokens: Token[];
  
  animate?: boolean;
  
  highlight?: (t: Token) => boolean;
};

export default function Tokens({ tokens, animate, highlight }: Props) {
  return (
    <>
      {tokens.map((t, i) => {
        const m = /^(\s*)([\s\S]*)$/.exec(t.text)!;
        const hi = highlight?.(t);
        const cls = `tok ${animate ? "in" : ""} ${t.steered ? "steer" : ""} ${hi ? "only" : ""}`;
        return (
          <span key={i}>
            {m[1]}
            <span className={cls} title={hi ? "Only in the steered sentence" : t.steered ? `logit bias +${t.bias}` : undefined}>{m[2]}</span>
          </span>
        );
      })}
    </>
  );
}
