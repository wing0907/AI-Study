import { useEffect, useRef } from 'react';
import type { AttachmentsState } from '../types';

export default function AttachFlyout({
  open, onPick, onClose
}:{ open:boolean; onPick:(type:keyof AttachmentsState, payload:any)=>void; onClose:()=>void }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(()=>{
    if(!open) return;
    const onDown = (e:MouseEvent)=>{ if(ref.current && !ref.current.contains(e.target as Node)) onClose(); };
    const onEsc = (e:KeyboardEvent)=>{ if(e.key==='Escape') onClose(); };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onEsc);
    return ()=>{ document.removeEventListener('mousedown', onDown); document.removeEventListener('keydown', onEsc); };
  },[open, onClose]);

  if(!open) return null;

  return (
    <div ref={ref} className="attachPopover">
      <label className="chip">🖼 이미지
        <input type="file" accept="image/*" multiple onChange={(e)=> onPick('images', Array.from(e.target.files||[]))}/>
      </label>
      <label className="chip">🎙 음성
        <input type="file" accept="audio/*" onChange={(e)=> onPick('audio', e.target.files?.[0]||null)}/>
      </label>
      <label className="chip">📎 파일
        <input type="file" multiple onChange={(e)=> onPick('files', Array.from(e.target.files||[]))}/>
      </label>
    </div>
  );
}
