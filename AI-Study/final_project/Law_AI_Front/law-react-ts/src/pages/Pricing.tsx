// src/pages/Pricing.tsx
import Topbar from '../components/Topbar';

export default function Pricing() {
  return (
    <>
      <Topbar />
      <main style={{ maxWidth: 980, margin: '6vh auto', padding: '0 20px' }}>
        <header style={{ textAlign: 'center', marginBottom: 24 }}>
          <h1 style={{ margin: 0 }}>요금</h1>
          <p style={{ marginTop: 8, color: 'var(--muted)' }}>
            팀 규모/사용량에 맞게 유연한 플랜을 제공합니다.
          </p>
        </header>

        <section
          style={{
            display: 'grid',
            gap: 14,
            gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
          }}
        >
          <article className="glass" style={{ borderRadius: 16, padding: 16 }}>
            <h3 style={{ marginTop: 0 }}>Basic</h3>
            <ul style={{ margin: '8px 0', paddingLeft: 18, color: 'var(--muted)' }}>
              <li>개인/소규모 팀</li>
              <li>리서치 기본 기능</li>
              <li>월 N회 결과 생성</li>
            </ul>
            <button className="btn primary" style={{ width: '100%' }}>시작하기</button>
          </article>

          <article className="glass" style={{ borderRadius: 16, padding: 16 }}>
            <h3 style={{ marginTop: 0 }}>Pro</h3>
            <ul style={{ margin: '8px 0', paddingLeft: 18, color: 'var(--muted)' }}>
              <li>중형 팀</li>
              <li>전략 시뮬레이션 포함</li>
              <li>우선 지원</li>
            </ul>
            <button className="btn primary" style={{ width: '100%' }}>시작하기</button>
          </article>

          <article className="glass" style={{ borderRadius: 16, padding: 16 }}>
            <h3 style={{ marginTop: 0 }}>Enterprise</h3>
            <ul style={{ margin: '8px 0', paddingLeft: 18, color: 'var(--muted)' }}>
              <li>대규모/보안 요구</li>
              <li>온프레미스/전용 모델</li>
              <li>맞춤형 SLA</li>
            </ul>
            <button className="btn primary" style={{ width: '100%' }}>상담 요청</button>
          </article>
        </section>
      </main>
    </>
  );
}
