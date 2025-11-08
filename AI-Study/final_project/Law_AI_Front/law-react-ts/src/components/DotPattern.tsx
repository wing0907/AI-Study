// src/components/DotPattern.tsx
import * as React from 'react';

type Props = {
  placement?: 'full' | 'lower';
  opacity?: number;
};

export default function DotPattern({ placement = 'lower', opacity = 0.45 }: Props) {
  const style: React.CSSProperties = 
    placement === 'lower'
      ? { height: '48vh', top: 'auto', bottom: 0, opacity }
      : { height: '100%', top: 0, opacity };

  return <div className="dotGrid" style={style} aria-hidden="true" />;
}
