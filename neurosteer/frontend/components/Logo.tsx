export default function Logo({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden>
      <circle cx="16" cy="16" r="14.5" stroke="#e4ebf2" strokeOpacity="0.55" />
      <circle cx="16" cy="16" r="5" stroke="#7fe3ff" />
      {[0, 45, 90, 135, 180, 225, 270, 315].map((a) => (
        <line key={a} x1="16" y1="11" x2="16" y2={a % 90 === 0 ? 3 : 6} stroke="#e4ebf2" strokeOpacity={a % 90 === 0 ? 0.9 : 0.4}
          transform={`rotate(${a} 16 16)`} />
      ))}
      <circle cx="16" cy="16" r="1.4" fill="#ffb04a" />
    </svg>
  );
}
