// src/pages/Login.tsx
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Topbar from '../components/Topbar';

export default function Login() {
  const nav = useNavigate();
  const [email, setEmail] = useState('');
  const [pw, setPw] = useState('');
  const [loading, setLoading] = useState(false);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim() || !pw.trim()) return;
    setLoading(true);
    try {
      // TODO: 실제 로그인 API 연동
      await new Promise((r) => setTimeout(r, 500));
      // 로그인 후 메인으로
      nav('/landing', { replace: true });
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Topbar />
      <main
        style={{
          maxWidth: 420,
          margin: '8vh auto',
          padding: 20,
        }}
      >
        <div className="glass" style={{ padding: 20, borderRadius: 16 }}>
          <h1 style={{ marginTop: 0, marginBottom: 8 }}>로그인</h1>
          <p style={{ marginTop: 0, color: 'var(--muted)' }}>
            LawAI 계정으로 로그인하세요.
          </p>

          <form onSubmit={onSubmit} style={{ display: 'grid', gap: 12 }}>
            <label style={{ display: 'grid', gap: 6 }}>
              <span style={{ fontSize: 13, color: 'var(--muted)' }}>이메일</span>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                style={{
                  height: 42,
                  borderRadius: 12,
                  background: 'rgba(10,16,28,.55)',
                  border: '1px solid var(--border)',
                  color: 'var(--text)',
                  padding: '0 14px',
                }}
              />
            </label>

            <label style={{ display: 'grid', gap: 6 }}>
              <span style={{ fontSize: 13, color: 'var(--muted)' }}>비밀번호</span>
              <input
                type="password"
                value={pw}
                onChange={(e) => setPw(e.target.value)}
                placeholder="••••••••"
                style={{
                  height: 42,
                  borderRadius: 12,
                  background: 'rgba(10,16,28,.55)',
                  border: '1px solid var(--border)',
                  color: 'var(--text)',
                  padding: '0 14px',
                }}
              />
            </label>

            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <button
                type="button"
                className="btn"
                onClick={() => nav('/landing')}
              >
                돌아가기
              </button>
              <button
                type="submit"
                className="btn primary"
                disabled={loading}
              >
                {loading ? '로그인 중…' : '로그인'}
              </button>
            </div>
          </form>
        </div>
      </main>
    </>
  );
}
