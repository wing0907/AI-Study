import { useEffect, useRef, useState } from 'react';
import PatternLayer from '../components/PatternLayer';
import Topbar from '../components/Topbar';
import SearchDrawer from '../components/SearchDrawer';
type RetrievalItem = Record<string, any>;
type Result = { id: string; title: string; snippet: string };
/** --- 백엔드 호출: /api/answer (Nginx가 8501로 프록시) --- */
async function askBackend(query: string, noLLM = false): Promise<{ answer: string; retrieval?: RetrievalItem[] }> {
  const res = await fetch('/api/answer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, no_llm: noLLM }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`HTTP ${res.status} ${text}`);
  }
  return res.json();
}
export default function Landing() {
  const [query, setQuery] = useState('');
  const [openAttach, setOpenAttach] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);
  // 숨겨진 파일 입력 (아이콘 클릭 시 트리거)
  const imageInputRef = useRef<HTMLInputElement>(null);
  const audioInputRef = useRef<HTMLInputElement>(null);
  const fileInputRef  = useRef<HTMLInputElement>(null);
  const [showResults, setShowResults] = useState(false);
  const [results, setResults] = useState<Result[]>([]);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string>('');
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [history, setHistory] = useState<string[]>([]);
  // 바깥 클릭 & ESC → 첨부 메뉴 닫기
  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (!searchRef.current) return;
      if (!searchRef.current.contains(e.target as Node)) setOpenAttach(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpenAttach(false);
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, []);
  /** ------------------- 실제 검색 실행 (백엔드 호출) ------------------- */
  const runSearch = async (textRaw: string) => {
    const text = textRaw.trim();
    if (!text || loading) return;
    // 검색어 히스토리 업데이트
    setHistory(prev => {
      const rest = prev.filter(x => x !== text);
      return [text, ...rest].slice(0, 50);
    });
    setOpenAttach(false);
    setLoading(true);
    setErrorMsg('');
    setShowResults(true);
    setResults(prev => {
      // UX: 직전 결과는 잠깐 유지, 로딩 카드 추가
      return [
        ...prev.slice(0, 0), // 이전 결과 지우고 싶으면 0, 유지하고 싶으면 prev 그대로
        { id: 'loading', title: '생성 중…', snippet: '문서를 검색하고 요약을 생성하는 중입니다.' },
      ];
    });
    try {
      const data = await askBackend(text, false);
      // 메인 답변을 첫 카드로
      const main: Result = {
        id: `ans_${Date.now()}`,
        title: `“${text}”에 대한 답변`,
        snippet: data.answer || '(응답이 비어 있습니다.)',
      };
      // retrieval이 있다면 상위 3개 정도를 하이라이트 카드로 뒤에 추가
      const extras: Result[] = (data.retrieval || []).slice(0, 3).map((r, i) => {
        // 판례/법령 등 메타가 있을 수 있으니 최대한 읽기 좋게 타이틀 구성
        const title =
          r.사건명 ? `근거 #${i + 1}: ${r.사건명}` :
          r.law && r.article_no ? `근거 #${i + 1}: ${r.law} 제${r.article_no}조` :
          `근거 #${i + 1}`;
        const snippet = (r.text as string) || r.snippet || JSON.stringify(r).slice(0, 300);
        return {
          id: `ret_${Date.now()}_${i}`,
          title,
          snippet,
        };
      });
      setResults([main, ...extras]);
    } catch (err: any) {
      const msg = err?.message || String(err);
      setErrorMsg(`:경고: 서버 오류: ${msg}`);
      setResults([
        {
          id: `err_${Date.now()}`,
          title: '오류',
          snippet: `백엔드 호출에 실패했습니다.\n${msg}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };
  const onEnter = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') runSearch(query);
  };
  const selectHistory = (q: string) => {
    setQuery(q);
    void runSearch(q);
    setDrawerOpen(false);
  };
  const clearHistory = () => setHistory([]);
  // 아이콘 클릭 → 숨은 input 트리거
  const pickImage = () => imageInputRef.current?.click();
  const pickAudio = () => audioInputRef.current?.click();
  const pickFile  = () => fileInputRef.current?.click();
  // 파일 선택 시 닫기(필요하면 업로드 처리 추가)
  const afterPick = () => setOpenAttach(false);
  return (
    <div className={`landingRoot ${showResults ? 'hasResults' : ''}`}>
      <Topbar />
      <PatternLayer placement="upper" opacity={0.46} />
      <PatternLayer placement="lower" opacity={0.50} />
      {/* 결과가 '검색바 위' */}
      {showResults && (
        <section className="resultsStack resultsTop">
          {errorMsg && (
            <article className="resultCard glass" style={{ borderLeft: '3px solid #FF6961' }}>
              <div className="resultTitle">에러</div>
              <div className="resultSnippet">{errorMsg}</div>
            </article>
          )}
          {results.map(r => (
            <article key={r.id} className="resultCard glass">
              <div className="resultTitle">
                {r.title}
                {r.id === 'loading' && ' :모래가_내려오고_있는_모래시계:'}
              </div>
              <div className="resultSnippet" style={{ whiteSpace: 'pre-wrap' }}>{r.snippet}</div>
            </article>
          ))}
        </section>
      )}
      <section className="landingHero">
        {!showResults && (
          <>
            <h1 className="landingTitle">법률가를 위한 올인원 AI 어시스턴트</h1>
            <p className="landingSub">
              지능형 리서치 · 전략 시뮬레이션 · 증거 보관함을 하나로.
            </p>
          </>
        )}
        <div className={`searchBar ${showResults ? 'docked' : ''}`} ref={searchRef}>
          <button
            className={`iconBtn ${openAttach ? 'active' : ''}`}
            onClick={() => setOpenAttach(v => !v)}
            aria-label="첨부"
            type="button"
            title="첨부"
            disabled={loading}
          >
            ＋
          </button>
          <input
            placeholder="무엇을 도와드릴까요?"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={onEnter}
            disabled={loading}
          />
          <button
            className="btn primary"
            onClick={() => runSearch(query)}
            disabled={loading}
          >
            {loading ? '검색 중…' : '검색'}
          </button>
          {/* 첨부 메뉴: 픽토그램 아이콘 */}
          {openAttach && !loading && (
            <div className="attachMenu icons">
              <button className="chip icon" onClick={pickImage} aria-label="이미지 첨부" title="이미지">
                {/* image-pictogram (outline) */}
                <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
                  <path fill="currentColor"
                    d="M19 3H5a2 2 0 0 0-2 2v14l.01.01A2 2 0 0 0 5 21h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2m0 16H5V5h14v14M8.5 11.5A1.5 1.5 0 1 1 10 10a1.5 1.5 0 0 1-1.5 1.5M6 17l3.5-4.5 2.5 3 3.5-4.5 4.5 6H6Z"/>
                </svg>
              </button>
              <button className="chip icon" onClick={pickAudio} aria-label="음성 첨부" title="음성">
                {/* mic-pictogram */}
                <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
                  <path fill="currentColor"
                    d="M12 14a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v5a3 3 0 0 0 3 3m5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11Z"/>
                </svg>
              </button>
              <button className="chip icon" onClick={pickFile} aria-label="파일 첨부" title="파일">
                {/* paperclip-pictogram */}
                <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
                  <path fill="currentColor"
                    d="M7.5 20A4.5 4.5 0 0 1 3 15.5c0-1.2.47-2.34 1.32-3.18L13.3 3.35A4.5 4.5 0 1 1 19.66 9.7l-7.07 7.07a3 3 0 1 1-4.24-4.24l6.01-6.01 1.41 1.41-6.01 6.01a1 1 0 1 0 1.42 1.42l7.07-7.07a2.5 2.5 0 1 0-3.54-3.54L6.75 11.9A3.5 3.5 0 0 0 7.5 20Z"/>
                </svg>
              </button>
            </div>
          )}
          {/* 숨겨진 input (선택 시 자동 닫힘) */}
          <input
            ref={imageInputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={afterPick}
          />
          <input
            ref={audioInputRef}
            type="file"
            accept="audio/*"
            hidden
            onChange={afterPick}
          />
          <input
            ref={fileInputRef}
            type="file"
            hidden
            onChange={afterPick}
          />
        </div>
      </section>
      <SearchDrawer
        open={drawerOpen}
        history={history}
        onSelect={selectHistory}
        onClear={clearHistory}
        onToggle={() => setDrawerOpen(o => !o)}
      />
    </div>
  );
}





