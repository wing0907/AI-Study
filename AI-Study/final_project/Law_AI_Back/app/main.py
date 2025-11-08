# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.settings import settings
from app.routers.search import router as search_router
from app.retriever import count_docs  # ← get_vectordb(오타) 제거

# (선택) 진단용
import os
import chromadb

app = FastAPI()

# CORS
origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

# ---- 진단용 Chroma 라우트들 ----
@app.get("/chroma/collections")
def chroma_collections():
    try:
        client = chromadb.PersistentClient(path=settings.VECTOR_DIR)
        cols = client.list_collections()
        return {"ok": True, "collections": [c.name for c in cols]}
    except Exception as e:
        return {"ok": False, "error": f"collections_failed: {e}"}

@app.get("/chroma/count")
def chroma_count():
    try:
        client = chromadb.PersistentClient(path=settings.VECTOR_DIR)
        out = {}
        for c in client.list_collections():
            try:
                out[c.name] = c.count()
            except Exception as e:
                out[c.name] = f"error: {e}"
        return {"ok": True, "counts": out}
    except Exception as e:
        return {"ok": False, "error": f"count_failed: {e}"}

@app.get("/chroma/which")
def chroma_which():
    try:
        client = chromadb.PersistentClient(path=settings.VECTOR_DIR)
        name = (os.getenv("CHROMA_COLLECTION") or settings.CHROMA_COLLECTION or "").strip()
        cols = client.list_collections()
        names = [c.name for c in cols]
        if not name or name not in names:
            # 문서수 최댓값으로 자동 선택
            best = None
            best_cnt = -1
            for c in cols:
                try:
                    cnt = c.count()
                except Exception:
                    cnt = 0
                if cnt > best_cnt:
                    best, best_cnt = c, cnt
            name = best.name if best else None
            if not name:
                return {"ok": False, "error": "no_collections_found"}
        count = client.get_collection(name=name).count()
        return {"ok": True, "collection": name, "docs_count": count}
    except Exception as e:
        return {"ok": False, "error": f"which_failed: {e}"}
# -------------------------------

# 검색 라우터
app.include_router(search_router)

# 시작 로그
@app.on_event("startup")
def _startup_log():
    try:
        n = count_docs()
        print(f"[Chroma] path = {settings.VECTOR_DIR}")
        print(f"[Chroma] using collection = {settings.CHROMA_COLLECTION or '(auto)'} | docs = {n}")
    except Exception as e:
        print(f"[Chroma] load failed: {e}")
