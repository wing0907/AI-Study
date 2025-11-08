import React from 'react';

type Props = {
  value:string;
  setValue:(v:string)=>void;
  onSend:()=>void;
  attachOpen:boolean;
  setAttachOpen:(v:boolean)=>void;
  AttachMenu:React.ReactNode;
};

export default function BottomDock({
  value, setValue, onSend, attachOpen, setAttachOpen, AttachMenu
}:Props){
  return (
    <div className="dock">
      <div className="dockInner">
        {attachOpen && AttachMenu}
        <div className="inputGroup">
          <button
            className={`iconBtn ${attachOpen ? 'active':''}`}
            aria-label="첨부"
            onClick={()=>setAttachOpen(!attachOpen)}
          >＋</button>

          <input
            placeholder="무엇을 도와드릴까요?"
            value={value}
            onChange={(e)=>setValue(e.target.value)}
            onKeyDown={(e)=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); onSend(); } }}
          />

          <button className="sendBtn" onClick={onSend}>전송</button>
        </div>
      </div>
    </div>
  );
}
