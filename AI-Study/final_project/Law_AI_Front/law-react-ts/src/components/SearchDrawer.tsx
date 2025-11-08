// src/components/SearchDrawer.tsx
import { useEffect, useRef } from 'react';

interface Props {
  open: boolean;
  history: string[];
  onSelect: (q: string) => void;
  onClear: () => void;
  onToggle: () => void;
}

/**
 * 오프캔버스 검색 기록 드로어
 * - position: fixed 로 문서 레이아웃에 영향 X
 * - transform 으로 슬라이드 인/아웃
 * - 모든 클래스는 sd* 접두어로 격리
 */
export default function SearchDrawer({
  open,
  history,
  onSelect,
  onClear,
  onToggle,
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);

  // 바깥 클릭 시 닫기
  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (!open) return;
      if (!wrapRef.current) return;
      if (!wrapRef.current.contains(e.target as Node)) {
        onToggle();
      }
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [open, onToggle]);

  return (
    <>
      {/* Drawer Panel */}
      <aside
        ref={wrapRef}
        className={`sdDrawer ${open ? 'sdOpen' : ''}`}
        aria-hidden={!open}
      >
        <div className="sdHeader">
          <div className="sdTitle">검색기록</div>
          <button className="btn ghost sm" onClick={onClear}>
            모두 지우기
          </button>
        </div>

        <div className="sdBody">
          {history.length === 0 ? (
            <div className="sdMuted">아직 검색 기록이 없습니다.</div>
          ) : (
            <ul className="sdList">
              {history.map((q, i) => (
                <li key={`${q}-${i}`}>
                  <button
                    className="sdItem"
                    onClick={() => onSelect(q)}
                    title={q}
                  >
                    <span className="sdDot" />
                    <span className="sdEllipsis">{q}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>

      {/* Toggle Handle */}
      <button
        className={`sdHandle ${open ? 'sdShift' : ''}`}
        onClick={onToggle}
        aria-label="검색기록 열기/닫기"
        type="button"
      >
        {open ? '⟨' : '⟩'}
      </button>
    </>
  );
}
