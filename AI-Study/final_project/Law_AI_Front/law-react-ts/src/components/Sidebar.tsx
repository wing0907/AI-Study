import Logo from './Logo';
import type { PageKey } from '../types';

export default function Sidebar({
  page, setPage, summary,
}:{ page:PageKey; setPage:(p:PageKey)=>void; summary:string }) {
  return (
    <aside className="sidebar">
      <div className="brandRow">
        <Logo size={14} showWordmark />
        <div className="logoText">LawAI</div>
      </div>

      <div className="nav">
        <button className={page==='research'?'active':''} onClick={()=>setPage('research')}>지능형 리서치</button>
        <button className={page==='simulation'?'active':''} onClick={()=>setPage('simulation')}>전략 시뮬레이션</button>
        <div className="divider" />
        <button className={page==='evidence'?'active':''} onClick={()=>setPage('evidence')}>증거 보관함</button>
        <button className={page==='history'?'active':''} onClick={()=>setPage('history')}>히스토리</button>
        <button className={page==='settings'?'active':''} onClick={()=>setPage('settings')}>설정</button>
      </div>

      <div className="summary">
        <div className="summaryTitle">요약</div>
        <div className="summaryBox">{summary || '검색/시뮬레이션 후 요약이 여기에 표시됩니다.'}</div>
      </div>
    </aside>
  );
}
