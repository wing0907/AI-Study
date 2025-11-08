// src/components/ShieldMark.tsx
export default function ShieldMark({ size = 16 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      aria-hidden="true"
      style={{ color: '#70b6ff' }}
    >
      <defs>
        <linearGradient id="g1" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#88c7ff" />
          <stop offset="100%" stopColor="#4da1ff" />
        </linearGradient>
      </defs>
      <path
        d="M12 2l7 3v6c0 4.7-3.2 8.9-7 10-3.8-1.1-7-5.3-7-10V5l7-3z"
        fill="url(#g1)"
        stroke="currentColor"
        strokeWidth="0.6"
        opacity="0.95"
      />
      <path
        d="M9 12l2 2 4-4"
        fill="none"
        stroke="#0b1220"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
