// src/pages/Solution.tsx
import Topbar from '../components/Topbar';

export default function Solution() {
  return (
    <>
      <Topbar />
      <main style={{ maxWidth: 980, margin: '6vh auto', padding: '0 20px' }}>
        <header style={{ textAlign: 'center', marginBottom: 24 }}>
          <h1 style={{ margin: 0 }}>솔루션</h1>
          <p style={{ marginTop: 8, color: 'var(--muted)' }}>
            LawAI가 제공하는 지능형 리서치/시뮬레이션/증거보관 솔루션을 소개합니다.
          </p>
        </header>

        <section
          style={{
            display: 'grid',
            gap: 14,
            gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
          }}
        >
          <article className="glass" style={{ borderRadius: 16, padding: 16 }}>
            <h3 style={{ marginTop: 0 }}>지능형 리서치</h3>
            <p style={{ color: 'var(--muted)' }}>
              로컬 인덱스 기반 RAG로 신뢰 가능한 인용과 함께 답변을 제공합니다.
            </p>
          </article>
          <article className="glass" style={{ borderRadius: 16, padding: 16 }}>
            <h3 style={{ marginTop: 0 }}>전략 시뮬레이션</h3>
            <p style={{ color: 'var(--muted)' }}>
              사실관계/주장을 넣으면 반박/증거/리스크가 포함된 전략을 생성합니다.
            </p>
          </article>
          <article className="glass" style={{ borderRadius: 16, padding: 16 }}>
            <h3 style={{ marginTop: 0 }}>증거 보관함</h3>
            <p style={{ color: 'var(--muted)' }}>
              이미지·음성·파일 첨부를 사건별 폴더로 정리하고 추적할 수 있습니다.
            </p>
          </article>
        </section>
      </main>
    </>
  );
}
