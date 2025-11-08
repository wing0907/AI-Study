type Props = {
  placement: 'upper' | 'lower';
  opacity?: number;
};

export default function PatternLayer({ placement, opacity = 0.5 }: Props) {
  if (placement === 'upper') {
    // 상단 헤일로
    const style: React.CSSProperties = {
      position: 'absolute',
      inset: 0,
      pointerEvents: 'none',
      opacity,
      background:
        'radial-gradient(900px 380px at 50% 0%, rgba(77,161,255,.20), rgba(11,18,32,0) 70%)',
      zIndex: 0,
    };
    return <div style={style} aria-hidden />;
  }

  // 하단 도트/그리드
  const style: React.CSSProperties = {
    position: 'absolute',
    inset: 0,
    pointerEvents: 'none',
    opacity,
    backgroundImage:
      'radial-gradient(rgba(143,164,201,.25) 1px, transparent 1.2px)',
    backgroundSize: '22px 22px',
    backgroundPosition: 'center 35vh',
    zIndex: 0,
  };
  return <div style={style} aria-hidden />;
}
