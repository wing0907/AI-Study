// src/App.tsx
import { useEffect, useMemo, useState } from 'react';
import {
  Routes,
  Route,
  Navigate,
  useLocation,
  useNavigate,
} from 'react-router-dom';
import Sidebar from './components/Sidebar';
import ChatList from './components/ChatList';
import AttachFlyout from './components/AttachFlyout';
import BottomDock from './components/BottomDock';
import Logo from './components/Logo';
import './styles.css';
// Pages
import Landing from './pages/Landing';
import Solution from './pages/Solution';
import Cases from './pages/Cases';
import Pricing from './pages/Pricing';
import Login from './pages/Login';
// types
import type { AttachmentsState, Message, PageKey } from './types';
/** --- API 호출 헬퍼: /api/answer (Nginx가 8501로 프록시) --- */
async function askBackend(query: string, noLLM = false): Promise<{ answer: string; retrieval?: any[] }> {
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
/* ---------------- Path <-> PageKey 매핑 ---------------- */
const pathToPage = (path: string): PageKey => {
  switch (path) {
    case '/simulation': return 'simulation';
    case '/evidence':   return 'evidence';
    case '/history':    return 'history';
    case '/settings':   return 'settings';
    case '/research':
    default:            return 'research';
  }
};
const pageToPath = (p: PageKey): string => {
  switch (p) {
    case 'simulation': return '/simulation';
    case 'evidence':   return '/evidence';
    case 'history':    return '/history';
    case 'settings':   return '/settings';
    case 'research':
    default:           return '/research';
  }
};
/* ---------------- 메인 셸 레이아웃 ----------------
   사이드바 + 탑바 + 컨테이너 + (조건부) 입력 Dock
--------------------------------------------------- */
function MainShell({ pageKey }: { pageKey: PageKey }) {
  const navigate = useNavigate();
  const location = useLocation();
  // URL과 동기화되는 현재 페이지
  const [page, _setPage] = useState<PageKey>(pageKey);
  useEffect(() => { _setPage(pathToPage(location.pathname)); }, [location.pathname]);
  // 전역 상태
  const [summary, setSummary] = useState('');
  const [researchMsgs, setResearchMsgs] = useState<Message[]>([]);
  const [simMsgs, setSimMsgs] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [attachOpen, setAttachOpen] = useState(false);
  const [, setAttachments] = useState<AttachmentsState>({ images: [], audio: null, files: [] });
  const [loading, setLoading] = useState(false);
  const setPage = (p: PageKey) => {
    _setPage(p);
    const to = pageToPath(p);
    if (location.pathname !== to) navigate(to);
  };
  const onPick = (type: keyof AttachmentsState, payload: any) =>
    setAttachments(prev => ({ ...prev, [type]: payload }));
  /** ------------------- 핵심: 백엔드 호출 연결 ------------------- */
  const handleSend = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput('');
    setAttachOpen(false);
    if (page === 'research') {
      // 1) 사용자 메시지 먼저 추가
      setResearchMsgs(prev => [...prev, { role: 'user' as const, text }]);
      // 2) 로딩 표시용 플레이스홀더
      const loadingMsg: Message = { role: 'assistant', text: '생성 중…' };
      setResearchMsgs(prev => [...prev, loadingMsg]);
      setLoading(true);
      try {
        // 3) 백엔드 호출 (LLM 사용; 디버깅 시 no_llm를 true로)
        const res = await askBackend(text, false);
        // 4) 로딩 메시지 교체
        setResearchMsgs(prev => {
          const copy = [...prev];
          const lastIdx = copy.length - 1;
          if (lastIdx >= 0 && copy[lastIdx].text === '생성 중…') {
            copy[lastIdx] = { role: 'assistant', text: res.answer || '응답이 비어 있습니다.' };
          } else {
            copy.push({ role: 'assistant', text: res.answer || '응답이 비어 있습니다.' });
          }
          return copy;
        });
        // 5) 간단 요약 업데이트
        setSummary(`요약: ${res.answer?.slice(0, 40) || ''}`);
      } catch (err: any) {
        setResearchMsgs(prev => {
          const copy = [...prev];
          const lastIdx = copy.length - 1;
          if (lastIdx >= 0 && copy[lastIdx].text === '생성 중…') {
            copy[lastIdx] = { role: 'assistant', text: `:경고: 서버 오류: ${err?.message || err}` };
          } else {
            copy.push({ role: 'assistant', text: `:경고: 서버 오류: ${err?.message || err}` });
          }
          return copy;
        });
      } finally {
        setLoading(false);
      }
    } else if (page === 'simulation') {
      // 필요 시 시뮬레이션도 같은 API를 쓰거나 별도 엔드포인트로 분리 가능
      setSimMsgs(prev => [
        ...prev,
        { role: 'user' as const, text },
        { role: 'assistant' as const, text: '전략 초안(예시)\n1) 쟁점 정리 …\n2) 반박 포인트 …\n3) 증거 매칭 …\n4) 리스크/대안 …' },
      ]);
      setSummary('요약: 전략 초안이 생성되었습니다.');
    }
  };
  const PageBody = useMemo(() => {
    switch (page) {
      case 'research':
        return (
          <>
            <div className="pageTitle">지능형 리서치</div>
            <p className="pageCap">질문을 입력하면 관련 법률 문서 일부를 검색해 보여줍니다.</p>
            <div className="divider" />
            <ChatList messages={researchMsgs} />
          </>
        );
      case 'simulation':
        return (
          <>
            <div className="pageTitle">전략 시뮬레이션</div>
            <p className="pageCap">사건 시나리오를 입력하고 전략을 시뮬레이션합니다.</p>
            <div className="divider" />
            <ChatList messages={simMsgs} />
          </>
        );
      case 'evidence':
        return (
          <>
            <div className="pageTitle">증거 보관함</div>
            <p className="pageCap">업로드한 파일/이미지/음성을 관리하고, 사건별 폴더로 정리하세요.</p>
          </>
        );
      case 'history':
        return (
          <>
            <div className="pageTitle">히스토리</div>
            <p className="pageCap">이전 검색/시뮬레이션 세션을 날짜/사건별로 확인합니다.</p>
          </>
        );
      case 'settings':
      default:
        return (
          <>
            <div className="pageTitle">설정</div>
            <p className="pageCap">모델/엔드포인트/테마/데이터 경로 등을 설정합니다.</p>
          </>
        );
    }
  }, [page, researchMsgs, simMsgs]);
  const showDock = page === 'research' || page === 'simulation';
  return (
    <div className="app">
      <Sidebar page={page} setPage={setPage} summary={summary} />
      <main className="main">
        {/* 상단 바 */}
        <header className="topbar">
          <div className="topbarLeft">
            <Logo size={16} showWordmark />
          </div>
        <div className="topbarRight">
            <button className="btn" onClick={() => navigate('/login')}>로그인</button>
            <button className="btn primary" onClick={() => navigate('/research')}>무료 가입</button>
          </div>
        </header>
        <div className="container">{PageBody}</div>
        {showDock && (
          <BottomDock
            value={input}
            setValue={setInput}
            onSend={handleSend}
            attachOpen={attachOpen}
            setAttachOpen={setAttachOpen}
            AttachMenu={
              <AttachFlyout
                open={attachOpen}
                onPick={onPick}
                onClose={() => setAttachOpen(false)}
              />
            }
          />
        )}
      </main>
    </div>
  );
}
/* ---------------- 루트 라우터 ---------------- */
export default function App() {
  return (
    <Routes>
      {/* 기본은 랜딩으로 */}
      <Route path="/" element={<Navigate to="/landing" replace />} />
      {/* 랜딩 & 상단 메뉴 */}
      <Route path="/landing" element={<Landing />} />
      <Route path="/solution" element={<Solution />} />
      <Route path="/cases" element={<Cases />} />
      <Route path="/pricing" element={<Pricing />} />
      {/* 인증 */}
      <Route path="/login" element={<Login />} />
      {/* 기능 영역: MainShell 사용 */}
      <Route path="/research"  element={<MainShell pageKey="research" />} />
      <Route path="/simulation" element={<MainShell pageKey="simulation" />} />
      <Route path="/evidence"   element={<MainShell pageKey="evidence" />} />
      <Route path="/history"    element={<MainShell pageKey="history" />} />
      <Route path="/settings"   element={<MainShell pageKey="settings" />} />
      {/* 기타 → 랜딩 */}
      <Route path="*" element={<Navigate to="/landing" replace />} />
    </Routes>
  );
}