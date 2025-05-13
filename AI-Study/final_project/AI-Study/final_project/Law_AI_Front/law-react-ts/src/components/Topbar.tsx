// src/components/Topbar.tsx
import Logo from './Logo';
import { useNavigate } from 'react-router-dom';

export default function Topbar() {
  const nav = useNavigate();
  return (
    <header className="topbar">
      <div className="topbarLeft" onClick={() => nav('/landing')} role="button" aria-label="홈으로">
        <Logo size={18} showWordmark />
      </div>

      <nav className="topbarNav">
        <button className="link" onClick={() => nav('/solution')}>솔루션</button>
        <button className="link" onClick={() => nav('/cases')}>적용사례</button>
        <button className="link" onClick={() => nav('/pricing')}>요금</button>
      </nav>

      <div className="topbarRight">
        <button className="btn ghost sm" onClick={() => nav('/login')}>로그인</button>
      </div>
    </header>
  );
}
