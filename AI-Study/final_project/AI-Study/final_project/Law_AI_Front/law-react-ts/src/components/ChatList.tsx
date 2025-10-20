// src/components/ChatList.tsx
import type { Message } from '../types';

export default function ChatList({ messages }:{messages:Message[]}) {
  return (
    <div className="glass">
      <div className="chatList">
        {messages.length===0 && (
          <div className="chatMessage">아직 메시지가 없습니다.</div>
        )}
        {messages.map((m, i)=>(
          <div key={i} className={`chatMessage ${m.role}`}>
            <div className="role">{m.role==='user'?'사용자':'LawAI'}</div>
            <div>{m.text}</div>
            {/* 필요하면 메타(시간 등) */}
            {/* <div className="meta">오후 3:41</div> */}
          </div>
        ))}
      </div>
    </div>
  );
}
