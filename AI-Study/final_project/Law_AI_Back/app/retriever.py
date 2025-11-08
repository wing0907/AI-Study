# app/retriever.py
from __future__ import annotations

import os
import chromadb
from typing import Optional

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from app.settings import settings


def _embedding():
    # langchain-ollama 0.3.x: base_url 파라미터 사용
    return OllamaEmbeddings(model=settings.EMBED_MODEL, base_url=settings.OLLAMA_HOST)


def _choose_collection_name(client: chromadb.ClientAPI) -> str:
    """
    1순위: .env 의 CHROMA_COLLECTION
    2순위: 존재 컬렉션 중 문서 수가 가장 큰 것
    """
    wanted = (os.getenv("CHROMA_COLLECTION") or settings.CHROMA_COLLECTION or "").strip()
    if wanted:
        for c in client.list_collections():
            if c.name == wanted:
                return wanted
        # 없으면 아래 fall-back

    cols = client.list_collections()
    if not cols:
        raise RuntimeError("Chroma에 컬렉션이 없습니다. VECTOR_DIR 경로를 확인하세요.")

    best = None
    best_cnt = -1
    for c in cols:
        try:
            cnt = c.count()  # int 반환
        except Exception:
            cnt = 0
        if cnt > best_cnt:
            best, best_cnt = c, cnt

    if not best:
        raise RuntimeError("사용 가능한 컬렉션을 찾지 못했습니다.")
    return best.name


def get_vectorstore() -> Chroma:
    """
    기존 퍼시스턴트 인덱스를 langchain_chroma.Chroma 로 연다.
    """
    client = chromadb.PersistentClient(path=settings.VECTOR_DIR)
    name = _choose_collection_name(client)
    vs = Chroma(
        client=client,
        collection_name=name,
        embedding_function=_embedding(),
    )
    return vs


def get_retriever(k: int = 5):
    vs = get_vectorstore()
    return vs.as_retriever(search_kwargs={"k": k})


def count_docs() -> int:
    client = chromadb.PersistentClient(path=settings.VECTOR_DIR)
    name = _choose_collection_name(client)
    col = client.get_collection(name=name)
    return col.count()
