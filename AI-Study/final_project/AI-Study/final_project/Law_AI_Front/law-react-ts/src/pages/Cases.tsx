// src/pages/Cases.tsx
import Topbar from '../components/Topbar';

export default function Cases() {
  return (
    <>
      <Topbar />
      <main style={{ maxWidth: 980, margin: '6vh auto', padding: '0 20px' }}>
        <header style={{ textAlign: 'center', marginBottom: 24 }}>
          <h1 style={{ margin: 0 }}>적용사례</h1>
          <p style={{ marginTop: 8, color: 'var(--muted)' }}>
            실제 로펌/법무팀/개별 변호사 분들의 도입 사례를 확인하세요.
          </p>
        </header>

        <section style={{ display: 'grid', gap: 14 }}>
          {[1, 2, 3].map((n) => (
            <article key={n} className="glass" style={{ borderRadius: 16, padding: 16 }}>
              <h3 style={{ marginTop: 0 }}>케이스 스터디 #{n}</h3>
              <p style={{ color: 'var(--muted)' }}>
                도입 배경 · 문제 · 해결 과정 · 효과(시간 절감, 정확도 개선 등)를 요약합니다.
              </p>
            </article>
          ))}
        </section>
      </main>
    </>
  );
}
