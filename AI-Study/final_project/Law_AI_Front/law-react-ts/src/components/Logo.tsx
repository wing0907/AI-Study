type Props = {
  size?: number;
  showWordmark?: boolean;
};

export default function Logo({ size = 20, showWordmark = true }: Props) {
  return (
    <div className="logo" style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
      {/* 방패 + 학사모 느낌의 심볼 (시계 아이콘 제거) */}
      <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden>
        <path fill="#4da1ff" d="M12 2l8 3v6c0 5-3.4 9.1-8 11-4.6-1.9-8-6-8-11V5l8-3z"/>
        <path fill="#0b1220" d="M12 5l6 2.2-6 2.3L6 7.2 12 5z"/>
      </svg>
      {showWordmark && <span className="wordmark">LawAI</span>}
    </div>
  );
}
